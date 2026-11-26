"""Row-invariant matrix-unit law for Flash-Next AST tree tiles (opt-in, prototype).

Fast-route contract (AGENTS.md, Pierre 2026-11-26): a tile row need not match
its ordinary one-token call, but its result must not depend on which or how
many other rows share the forward, so the route is exact with itself across
batch widths. The default tree law (``flash_next_tree``) meets that by running
the one-row scalar kernels over every row, which is ALU bound and makes a
16-row tile cost about eight ordinary steps. This law keeps the invariance and
moves the two expensive stages onto matrix units:

* projections run as ``invariant_prefill.quantized_matmul`` (two batches of at
  least 33 rows: the NAX ``qmm`` 64-row tile, never split K, whatever the row
  count);
* routed experts are grouped: the tile's (row, slot) pairs are sorted by expert,
  each expert's rows are cut into fixed ``group_rows`` chunks (padding lanes
  repeat row 0 and are never read back), and
  one ``gather_qmm`` per projection reads each touched expert once. The chunk
  height is fixed, so the kernel configuration does not depend on the tile;
  slots are combined per row in slot order with float32 accumulation;
* the shared expert runs on the invariant projections.

Routing (router and shared-expert gate projections), norms, HC, GDN and QSA keep
the default tree law, which is already row-invariant. ``moe`` groups on the
host (one synchronisation per MoE layer, measured about 55 ms per tile);
``moe_device`` sorts on the GPU and runs the segmented NAX expert kernels of
``moe_nax_gather`` (no host sync, weights read once per expert run).

Never a default; selected only by an explicit executor argument.
"""
from __future__ import annotations

from collections import Counter

import mlx.core as mx
import numpy as np

from . import invariant_prefill as _inv
from .activations import swiglu

LAW = "flash_tree_row_invariant_mma_v1"
# MLX runs a quantized gather below its qmv batch limit (25 rows for the
# Flash-Next expert shapes, ``qmm_row_pad.vector_limit``) on the per-row vector
# kernel, where every padded row costs a full matvec; 32 reaches the matrix
# ``gather_qmm``. 8 and 16 stay selectable for attribution.
GROUP_ROW_CHOICES = (8, 16, 32, 64)

STATS: Counter = Counter()


def project(module, x):
    """Row-invariant matrix projection for an affine quantized linear, else None."""
    if "scales" not in module or getattr(module, "mode", "affine") != "affine":
        return None
    out = _inv.quantized_matmul(
        x, module["weight"], module["scales"], module.get("biases"),
        group_size=module.group_size, bits=module.bits, mode="affine",
    )
    if "bias" in module:
        out = out + module["bias"]
    STATS["projection_calls"] += 1
    return out


def _linear(module, x):
    out = project(module, x)
    if out is None:
        raise ValueError(f"row-invariant law needs an affine quantized linear, got {type(module).__name__}")
    return out


def _groups(host_inds, group_rows):
    """(row ids [G, group_rows], experts [G], flat position of each (row, slot))."""
    rows, top_k = host_inds.shape
    flat = host_inds.reshape(-1)
    order = np.argsort(flat, kind="stable")  # (row, slot) order kept within an expert
    experts = flat[order]
    starts = np.flatnonzero(np.r_[True, experts[1:] != experts[:-1]])
    ends = np.r_[starts[1:], len(order)]
    group_rows_ids, group_experts = [], []
    position = np.empty(len(order), np.int32)
    for start, end in zip(starts, ends):
        for chunk in range(start, end, group_rows):
            members = order[chunk:min(chunk + group_rows, end)]
            base = len(group_experts) * group_rows
            position[members] = base + np.arange(len(members), dtype=np.int32)
            ids = np.zeros(group_rows, np.int32)
            ids[:len(members)] = members // top_k
            group_rows_ids.append(ids)
            group_experts.append(int(experts[start]))
    return (np.stack(group_rows_ids), np.asarray(group_experts, np.uint32),
            position.reshape(rows, top_k), len(starts))


def _gather(x, linear, experts):
    return mx.gather_qmm(
        x, linear["weight"], linear["scales"], linear.get("biases"),
        rhs_indices=experts, transpose=True,
        group_size=linear.group_size, bits=linear.bits, mode=linear.mode,
    )


def moe(block, x, inds, scores, gate_logit, *, group_rows=16):
    """``routed + sigmoid(gate_logit) * shared_expert(x)`` for every row: [rows, hidden]."""
    if group_rows not in GROUP_ROW_CHOICES:
        raise ValueError(f"group_rows must be one of {GROUP_ROW_CHOICES}")
    sw = block.switch_mlp
    for linear in (sw.gate_proj, sw.up_proj, sw.down_proj):
        if getattr(linear, "mode", None) != "affine" or "scales" not in linear or "bias" in linear:
            raise ValueError("row-invariant expert law needs affine quantized experts")
    hidden = x.shape[-1]
    rows = x.reshape(-1, hidden)
    count = int(rows.shape[0])
    top_k = int(inds.shape[-1])
    host = np.asarray(inds.reshape(count, top_k)).astype(np.int64)  # host sync
    ids, experts, position, distinct = _groups(host, group_rows)
    groups = len(experts)
    gx = mx.take(rows, mx.array(ids.reshape(-1)), axis=0).reshape(groups, group_rows, hidden)
    experts = mx.array(experts)
    h = swiglu(_gather(gx, sw.gate_proj, experts), _gather(gx, sw.up_proj, experts))
    y = _gather(h, sw.down_proj, experts).reshape(groups * group_rows, hidden)
    picked = mx.take(y, mx.array(position.reshape(-1)), axis=0).reshape(count, top_k, hidden)
    weights = scores.reshape(count, top_k).astype(mx.float32)
    routed = picked[:, 0].astype(mx.float32) * weights[:, 0:1]
    for slot in range(1, top_k):
        routed = routed + picked[:, slot].astype(mx.float32) * weights[:, slot:slot + 1]
    shared = block.shared_expert
    s = _linear(shared.down_proj, swiglu(_linear(shared.gate_proj, rows), _linear(shared.up_proj, rows)))
    out = routed + mx.sigmoid(gate_logit.reshape(count, 1).astype(mx.float32)) * s.astype(mx.float32)
    STATS["moe_calls"] += 1
    STATS["moe_groups"] += groups
    STATS["moe_distinct_experts"] += distinct
    STATS["moe_slots"] += count * top_k
    STATS["moe_pad_rows"] += groups * group_rows - count * top_k
    return out.astype(x.dtype)


def moe_device(block, x, inds, scores, gate_logit):
    """``moe`` without a host sync: the segmented NAX expert kernels.

    The (row, slot) pairs are sorted by expert on the GPU, and
    ``moe_nax_gather.sorted_gather_qmm_swiglu_split`` (gate + up + SwiGLU,
    rows read through the row map) and ``sorted_gather_qmm`` (down) run
    one tile per expert run, cut by the kernels' own GPU pre-pass. Each
    output element is bit-identical to MLX's sorted gather (the same tensor
    ops in the same K order), which does not depend on the other rows, so
    the law stays row-invariant. Returns None when the kernels decline
    (not an M5 host, layout, or an unverified instantiation); the caller
    then fails closed rather than mixing laws.
    """
    from . import moe_nax_gather as NAX

    sw = block.switch_mlp
    for linear in (sw.gate_proj, sw.up_proj, sw.down_proj):
        if getattr(linear, "mode", None) != "affine" or "scales" not in linear or "bias" in linear:
            raise ValueError("row-invariant expert law needs affine quantized experts")
    hidden = x.shape[-1]
    rows = x.reshape(-1, hidden)
    count = int(rows.shape[0])
    top_k = int(inds.shape[-1])
    flat = inds.reshape(-1).astype(mx.uint32)
    order = mx.argsort(flat).astype(mx.uint32)
    sorted_experts = mx.take(flat, order)
    row_map = (order // top_k).astype(mx.uint32)
    table = lambda linear: (linear["weight"], linear["scales"], linear.get("biases"))
    fmt = dict(group_size=int(sw.gate_proj.group_size), bits=int(sw.gate_proj.bits), mode="affine")
    h = NAX.sorted_gather_qmm_swiglu_split(rows[:, None, :], table(sw.gate_proj), table(sw.up_proj),
                                           sorted_experts, row_map=row_map, **fmt)
    if h is None:
        return None
    down = sw.down_proj
    y = NAX.sorted_gather_qmm(h, down["weight"], down["scales"], down.get("biases"), sorted_experts,
                              group_size=int(down.group_size), bits=int(down.bits), mode="affine")
    if y is None:
        return None
    inverse = mx.argsort(order)
    picked = mx.take(y.reshape(-1, hidden), inverse, axis=0).reshape(count, top_k, hidden)
    weights = scores.reshape(count, top_k).astype(mx.float32)
    routed = picked[:, 0].astype(mx.float32) * weights[:, 0:1]
    for slot in range(1, top_k):
        routed = routed + picked[:, slot].astype(mx.float32) * weights[:, slot:slot + 1]
    shared = block.shared_expert
    s = _linear(shared.down_proj, swiglu(_linear(shared.gate_proj, rows), _linear(shared.up_proj, rows)))
    out = routed + mx.sigmoid(gate_logit.reshape(count, 1).astype(mx.float32)) * s.astype(mx.float32)
    STATS["moe_device_calls"] += 1
    STATS["moe_slots"] += count * top_k
    return out.astype(x.dtype)


def status(*, reset=False):
    out = {"law": LAW, "counts": dict(STATS)}
    if reset:
        STATS.clear()
    return out
