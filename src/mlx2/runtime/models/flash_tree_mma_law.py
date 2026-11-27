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


_COMPILED = {}


def _compiled_projection(group_size, bits, has_biases):
    """One compiled call for pad + two-batch qmm + unpad (fewer graph nodes
    to build; the same ``invariant_prefill.quantized_matmul`` arithmetic)."""
    key = ("projection", group_size, bits, has_biases)
    fn = _COMPILED.get(key)
    if fn is None:
        if has_biases:
            def run(x, w, scales, biases):
                return _inv.quantized_matmul(x, w, scales, biases, group_size=group_size,
                                             bits=bits, mode="affine")
        else:
            def run(x, w, scales):
                return _inv.quantized_matmul(x, w, scales, None, group_size=group_size,
                                             bits=bits, mode="affine")
        fn = _COMPILED[key] = mx.compile(run)
    return fn


def _combine(picked, weights, shared, gate):
    """``sum_slot score * expert + sigmoid(gate) * shared`` in float32, slot order."""
    top_k = picked.shape[1]
    routed = picked[:, 0].astype(mx.float32) * weights[:, 0:1]
    for slot in range(1, top_k):
        routed = routed + picked[:, slot].astype(mx.float32) * weights[:, slot:slot + 1]
    out = routed + mx.sigmoid(gate.astype(mx.float32)) * shared.astype(mx.float32)
    return out.astype(picked.dtype)


_combine_compiled = mx.compile(_combine)
FUSED_GLUE = True  # compiled projection wrapper and MoE combine (fast-route law only)


_PADDED = {}
_ZEROS = {}
CACHED_PADDING = True  # padded inputs shared per forward + cached weight views


def begin_forward():
    """Drop the previous forward's padded-input memo (the executor calls this)."""
    _PADDED.clear()


def _zeros(rows, width, dtype):
    key = (rows, width, dtype)
    z = _ZEROS.get(key)
    if z is None:
        z = mx.zeros((rows, width), dtype=dtype)
        mx.eval(z)
        _ZEROS[key] = z
    return z


def _padded(x):
    """``x`` as the [2, m, K] zero-padded batch of ``invariant_prefill`` (same
    values), built once per input array per forward."""
    hit = _PADDED.get(id(x))
    if hit is not None and hit[0] is x:
        return hit[1], hit[2]
    width = int(x.shape[-1])
    rows = x.reshape(-1, width)
    n = int(rows.shape[0])
    m = max(_inv.MIN_BATCH_ROWS, -(-n // _inv.LINEAR_BATCHES))
    total = _inv.LINEAR_BATCHES * m
    if total > n:
        rows = mx.concatenate([rows, _zeros(total - n, width, rows.dtype)], axis=0)
    batched = rows.reshape(_inv.LINEAR_BATCHES, m, width)
    _PADDED[id(x)] = (x, batched, n)
    return batched, n


def _wide(module):
    """Broadcast [2, ...] views of the module's tables, evaluated once."""
    weight = module["weight"]
    cached = getattr(module, "_mlx2_tree_wide", None)
    if cached is None or cached[0] is not weight:
        tables = [module["weight"], module["scales"], module.get("biases")]
        wide = [None if t is None else mx.broadcast_to(t, (_inv.LINEAR_BATCHES, *t.shape)) for t in tables]
        mx.eval([w for w in wide if w is not None])
        cached = (weight, tuple(wide))
        object.__setattr__(module, "_mlx2_tree_wide", cached)
    return cached[1]


def project(module, x):
    """Row-invariant matrix projection for an affine quantized linear, else None."""
    if "scales" not in module or getattr(module, "mode", "affine") != "affine":
        return None
    biases = module.get("biases")
    if CACHED_PADDING:
        batched, n = _padded(x)
        w, sc, b = _wide(module)
        out = mx.quantized_matmul(batched, w, sc, b, transpose=True,
                                  group_size=module.group_size, bits=module.bits, mode="affine")
        out = out.reshape(-1, out.shape[-1])[:n].reshape(*x.shape[:-1], out.shape[-1])
    elif FUSED_GLUE:
        fn = _compiled_projection(int(module.group_size), int(module.bits), biases is not None)
        args = (x, module["weight"], module["scales"]) + ((biases,) if biases is not None else ())
        out = fn(*args)
    else:
        out = _inv.quantized_matmul(
            x, module["weight"], module["scales"], biases,
            group_size=module.group_size, bits=module.bits, mode="affine",
        )
    if "bias" in module:
        out = out + module["bias"]
    STATS["projection_calls"] += 1
    return out


def gate_project(module, x):
    """Router / shared-expert gate: one row-invariant call for all rows.

    Quantized affine gates take ``project``; dense ``Linear`` gates take
    ``invariant_prefill.dense_matmul`` on a two-copy ``W.T`` stack cached on
    the module (MLX's dense gemv/gemm choice follows the row count, the stack
    pins one GEMM configuration). None: unsupported, caller keeps its law.
    """
    out = project(module, x)
    if out is not None:
        return out
    weight = module.get("weight") if hasattr(module, "get") else None
    if weight is None or weight.ndim != 2 or "scales" in module:
        return None
    cached = getattr(module, "_mlx2_tree_gate_stack", None)
    if cached is None or cached[0] is not weight:
        cached = (weight, _inv.stack_dense(weight))
        object.__setattr__(module, "_mlx2_tree_gate_stack", cached)
    out = _inv.dense_matmul(x, cached[1], int(weight.shape[0]))
    if "bias" in module:
        out = out + module["bias"]
    STATS["gate_calls"] += 1
    return out


def route(block, gates):
    """Stock-identical routing in one launch (``qwen4_moe_window.router_topk``,
    one row per simdgroup, so row-invariant), or None when the model's routing
    contract does not admit it (the caller keeps the stock routing)."""
    from . import qwen4_moe_window as W

    contract = getattr(block, "moe_routing_contract", None)
    logits = gates.reshape(-1, gates.shape[-1])
    # A tree tile is a verify forward (the consumer name the contract knows).
    refusal = W.admit_router_topk(logits, int(block.top_k), bool(block.norm_topk_prob),
                                  contract=contract, consumer="verify")
    if refusal is not None:
        STATS["router_kernel_refused:" + refusal[:60]] += 1
        return None
    STATS["router_kernel_calls"] += 1
    return W.router_topk(logits, top_k=int(block.top_k), contract=contract, consumer="verify")


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
    shared = block.shared_expert
    s = _linear(shared.down_proj, swiglu(_linear(shared.gate_proj, rows), _linear(shared.up_proj, rows)))
    combine = _combine_compiled if FUSED_GLUE else _combine
    out = combine(picked, weights, s, gate_logit.reshape(count, 1))
    STATS["moe_calls"] += 1
    STATS["moe_groups"] += groups
    STATS["moe_distinct_experts"] += distinct
    STATS["moe_slots"] += count * top_k
    STATS["moe_pad_rows"] += groups * group_rows - count * top_k
    return out.astype(x.dtype)


EXPERT_PLAN = None  # None: moe_nax_gather picks; "bm32": 32-row tiles (tile rows are mostly padding)


def _expert_plan(NAX):
    if EXPERT_PLAN is None:
        return None
    if EXPERT_PLAN == "bm32":
        return NAX.Plan(NAX._SCHED_DB, 32, 64, 0, 0)
    raise ValueError(f"unknown expert plan {EXPERT_PLAN!r}")


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
    plan = _expert_plan(NAX)
    h = NAX.sorted_gather_qmm_swiglu_split(rows[:, None, :], table(sw.gate_proj), table(sw.up_proj),
                                           sorted_experts, row_map=row_map, plan=plan, **fmt)
    if h is None:
        return None
    down = sw.down_proj
    y = NAX.sorted_gather_qmm(h, down["weight"], down["scales"], down.get("biases"), sorted_experts,
                              group_size=int(down.group_size), bits=int(down.bits), mode="affine",
                              plan=plan)
    if y is None:
        return None
    inverse = mx.argsort(order)
    picked = mx.take(y.reshape(-1, hidden), inverse, axis=0).reshape(count, top_k, hidden)
    weights = scores.reshape(count, top_k).astype(mx.float32)
    shared = block.shared_expert
    s = _linear(shared.down_proj, swiglu(_linear(shared.gate_proj, rows), _linear(shared.up_proj, rows)))
    combine = _combine_compiled if FUSED_GLUE else _combine
    out = combine(picked, weights, s, gate_logit.reshape(count, 1))
    STATS["moe_device_calls"] += 1
    STATS["moe_slots"] += count * top_k
    return out.astype(x.dtype)


def moe_lane(block, x, inds, scores, gate_logit):
    """``moe`` on the tensor-unit expert-grouped kernels of
    ``qwen36_moe_grouped.lane_grouped_rows`` (GPU-planned expert groups, one
    16-row tile per group instead of a 64-row NAX tile, per-(row, slot)
    outputs). A row's outputs do not depend on its group's other members.
    None when the grouped kernels refuse (layout, rows, host)."""
    from . import qwen36_moe_grouped as G

    sw = block.switch_mlp
    hidden = x.shape[-1]
    rows = x.reshape(-1, hidden)
    count = int(rows.shape[0])
    top_k = int(inds.shape[-1])
    gate, up, down = sw.gate_proj, sw.up_proj, sw.down_proj
    refusal = G.admission_refusal(rows, inds, gate, up, down) or G.lane_refusal(gate, up, down)
    if refusal is not None:
        STATS["moe_lane_refused:" + refusal[:60]] += 1
        return None
    picked = G.lane_grouped_rows(rows, inds.reshape(count, top_k), gate, up, down).reshape(count, top_k, hidden)
    weights = scores.reshape(count, top_k).astype(mx.float32)
    shared = block.shared_expert
    s = _linear(shared.down_proj, swiglu(_linear(shared.gate_proj, rows), _linear(shared.up_proj, rows)))
    combine = _combine_compiled if FUSED_GLUE else _combine
    out = combine(picked, weights, s, gate_logit.reshape(count, 1))
    STATS["moe_lane_calls"] += 1
    return out.astype(x.dtype)


def _hc_mix_body(raw_mix, normed, hc_count):
    """``mean(sigmoid(raw_mix) * streams)`` over the HC streams, as explicit
    adds in stream order (no reduction whose tiling follows the row count)."""
    rows = raw_mix.shape[0]
    hidden = raw_mix.shape[-1] // hc_count
    weights = mx.sigmoid(raw_mix).reshape(rows, hc_count, hidden)
    streams = normed.reshape(rows, hc_count, hidden)
    total = weights[:, 0] * streams[:, 0]
    for index in range(1, hc_count):
        total = total + weights[:, index] * streams[:, index]
    return total * (1.0 / hc_count)


_hc_mix_compiled = mx.compile(_hc_mix_body, shapeless=False)


def hc_rows(module, x):
    """Row-invariant 16-row HC (``GatedResidual``) for a tree tile.

    The one-row HC decode law re-reads the down / up / inject weights for
    every row (16x the weight traffic at 16 rows). Here the grouped RMS norm
    keeps its one-row law per row and the elementwise steps run per row, but
    the three projections run once for all rows through ``project`` /
    ``gate_project`` (one weight read each). Not ordinary-exact (matrix
    projections); row-invariant. Returns ``(mixed, inject | None)`` with
    ``mixed`` [rows, hidden] and ``inject`` [rows, hc] (finished
    ``2 * sigmoid(raw / hc)``), or None when a projection is unsupported.
    """
    hc = int(module.hc_count)
    width = int(x.shape[-1])
    flat = x.reshape(-1, width)
    normed = module.hc_norm(flat.reshape(-1, 1, width)).reshape(flat.shape)
    gate_input = gate_project(module.input_mix_weight_down, normed)
    if gate_input is None:
        return None
    weights = nn_silu(gate_input / hc)
    raw_mix = gate_project(module.input_mix_weight_up, weights)
    if raw_mix is None:
        return None
    mix = _hc_mix_compiled if FUSED_GLUE else _hc_mix_body
    mixed = mix(raw_mix, normed, hc)
    inject = None
    if "block_inject_weight" in module:
        raw_gate = gate_project(module.block_inject_weight, normed)
        if raw_gate is None:
            return None
        inject = 2 * mx.sigmoid(raw_gate / hc)
    STATS["hc_rows_calls"] += 1
    return mixed, inject


def nn_silu(x):
    import mlx.nn as nn

    return nn.silu(x)


def status(*, reset=False):
    out = {"law": LAW, "counts": dict(STATS)}
    if reset:
        STATS.clear()
    return out
