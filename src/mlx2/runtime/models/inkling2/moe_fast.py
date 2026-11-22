# SPDX-License-Identifier: Apache-2.0
"""Opt-in fast paths for the Inkling 2 sparse MoE (all default off).

Original mlx2 code (2026-11-21).  It reuses mlx2's shared MoE pieces rather
than new kernels: ``switch_layers`` sort/unsort and gather modules, the
segmented NAX sorted gather (``moe_nax_gather``, reached through the expert
modules' own calls) and the sorted-order weighted sum
(``qwen4_moe_weighted_sum``).  See docs/PROVENANCE.md.

``inkling2_moe_fused`` (``fused_call``), the same arithmetic in fewer launches:

- **Routing** is the reference ``_route`` under ``mx.compile``: the sigmoid,
  bias add and negation fuse into one kernel and the normalization into
  another; the matmul, ``argpartition`` and gathers stay the same primitives.
- **Gate and up** run as one gather over a stacked ``[gate; up]`` table.
  Each output row of a quantized gather is its own dot product, so the
  stacked rows equal the two separate gathers.  The original gate/up
  modules become row views of the stacked table, so memory does not grow
  and the reference path still reads the same bytes.
- **Scale and activation.** NVFP4 gate and up share one per-expert
  ``weight_scale_2`` (both come from the checkpoint's ``w13`` amax); it is
  applied to the stacked output and the softcapped SwiGLU follows in one
  compiled kernel, with the reference's FP32 multiply and BF16 rounding.
- **Down scale and combine.** The down projection runs unscaled (a twin
  module on the same arrays); its per-expert scale and the weighted sum are
  one compiled kernel plus the reference reduction.  The scale is
  elementwise, so applying it after the unsort equals applying it before.
- One ``take`` reads both per-expert scales from a stacked ``[E, 2]`` table.

``inkling2_moe_weighted_sum`` (needs ``inkling2_moe_fused``; GPU, sorted
multi-token calls, i.e. prefill chunks and verify windows): the down rows stay in sorted order and
``qwen4_moe_weighted_sum`` reads them through ``inv_order``, replaying MLX's
column-reduction order, so the unsorted ``[T, K, D]`` copy and the FP32
product are never materialized.

Every path is intended to be bit-identical to the reference; the CPU and
Metal tests check it.  Anything outside the supported layouts declines,
counted, and the caller runs the reference ``SwitchGLU``.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from .. import invariant_prefill as _invariant
from .. import switch_grouped_experts as _grouped
from ..switch_layers import (
    QuantizedSwitchLinear,
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
    _switch_sort_decision,
)


def _route(x, weight, bias, top_k, route_scale):
    # Same ops as ``language._route`` (kept textually identical).
    scores = mx.sigmoid(x.astype(mx.float32) @ weight.astype(mx.float32).T)
    indices = mx.argpartition(-(scores + bias), top_k - 1, axis=-1)[..., :top_k]
    indices = mx.stop_gradient(indices)
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20) * route_scale
    return indices, weights


route_compiled = mx.compile(_route)


def _scale_law(y, scale):
    # ``nvfp4_linear._scale_law``: FP32 multiply, rounded back.
    return (y.astype(mx.float32) * scale).astype(y.dtype)


def _swiglu_law(up, gate, gate_cap, up_cap):
    # ``language._softcapped_swiglu``.
    dtype = gate.dtype
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    return (
        mx.tanh(gate / gate_cap)
        * gate_cap
        * mx.sigmoid(gate)
        * mx.tanh(up / up_cap)
        * up_cap
    ).astype(dtype)


@mx.compile
def _gate_up_act(gate_up, gate_cap, up_cap):
    n = gate_up.shape[-1] // 2
    return _swiglu_law(gate_up[..., n:], gate_up[..., :n], gate_cap, up_cap)


@mx.compile
def _scaled_gate_up_act(gate_up, scale, gate_cap, up_cap):
    n = gate_up.shape[-1] // 2
    gate = _scale_law(gate_up[..., :n], scale)
    up = _scale_law(gate_up[..., n:], scale)
    return _swiglu_law(up, gate, gate_cap, up_cap)


@mx.compile
def _combine(out, weights):
    # ``language._combine_routes``.
    return (out.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(out.dtype)


@mx.compile
def _scaled_combine(out, scale, weights):
    out = _scale_law(out, scale)
    return (out.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(out.dtype)


_scale_rows = mx.compile(_scale_law)


def _nvfp4_type():
    from ...nvfp4_linear import ScaledNVFP4SwitchLinear

    return ScaledNVFP4SwitchLinear


def _raw_nvfp4(weight, scales, scale2):
    """A ScaledNVFP4SwitchLinear on existing arrays that returns the
    unscaled product (``scale_folded`` on this instance only)."""
    cls = _nvfp4_type()
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    module.weight, module.scales, module.weight_scale_2 = weight, scales, scale2
    module.group_size, module.bits, module.mode = 16, 4, "nvfp4"
    module.scale_folded = True
    module.freeze()
    return module


def _quantized(weight, scales, like):
    module = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
    nn.Module.__init__(module)
    module.weight, module.scales, module.biases = weight, scales, None
    module.group_size, module.bits, module.mode = like.group_size, like.bits, like.mode
    module.freeze()
    return module


def _dense(weight):
    module = SwitchLinear.__new__(SwitchLinear)
    nn.Module.__init__(module)
    module.weight = weight
    return module


class FusedExperts:
    """The stacked gate/up table, the unscaled down twin and the scale table.

    Not an ``nn.Module``: it is kept off the parameter tree (its arrays are
    the same buffers the reference modules now view)."""

    def __init__(self, gate_up, down, scales):
        self.gate_up, self.down, self.scales = gate_up, down, scales


def source_key(switch):
    """The arrays a build was made from (weights, scales, global scales) and
    the fold flags; any replacement makes the build stale."""
    key = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        proj = switch[name]
        key += [proj.get(leaf) for leaf in ("weight", "scales", "weight_scale_2")]
        key.append(bool(getattr(proj, "scale_folded", False)))
    return tuple(key)


def build(switch) -> tuple[FusedExperts | None, str | None]:
    """Stack ``switch``'s gate/up tables (``(experts, None)`` or ``(None, why)``).

    Rewrites ``gate_proj`` / ``up_proj`` arrays as row views of the stacked
    table, so the reference path keeps working on the same bytes."""
    gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj
    nvfp4 = _nvfp4_type()
    kinds = {type(gate), type(up)}
    if len(kinds) != 1:
        return None, "gate_up_types_differ"
    kind = kinds.pop()
    if "bias" in gate or "bias" in up or "bias" in down:
        return None, "bias"
    n = int(gate.weight.shape[1])
    if int(up.weight.shape[1]) != n:
        return None, "gate_up_shapes_differ"
    if kind is nvfp4:
        if type(down) is not nvfp4:
            return None, "down_type"
        if gate.scale_folded or up.scale_folded or down.scale_folded:
            return None, "scales_folded"
        if not bool(mx.array_equal(gate["weight_scale_2"], up["weight_scale_2"]).item()):
            return None, "gate_up_scales_differ"
        weight = mx.concatenate([gate["weight"], up["weight"]], axis=1)
        qscales = mx.concatenate([gate["scales"], up["scales"]], axis=1)
        table = mx.stack([gate["weight_scale_2"], down["weight_scale_2"]], axis=-1)
        table = table.astype(mx.float32)
        mx.eval(weight, qscales, table)
        fused = _raw_nvfp4(weight, qscales, gate["weight_scale_2"])
        down_raw = _raw_nvfp4(down["weight"], down["scales"], down["weight_scale_2"])
        for proj, rows in ((gate, slice(0, n)), (up, slice(n, 2 * n))):
            proj.weight, proj.scales = weight[:, rows], qscales[:, rows]
        experts = FusedExperts(fused, down_raw, table)
    elif kind is QuantizedSwitchLinear:
        if (gate.group_size, gate.bits, gate.mode) != (up.group_size, up.bits, up.mode):
            return None, "gate_up_formats_differ"
        if gate.get("biases") is not None or up.get("biases") is not None:
            return None, "affine_biases"
        weight = mx.concatenate([gate["weight"], up["weight"]], axis=1)
        qscales = mx.concatenate([gate["scales"], up["scales"]], axis=1)
        mx.eval(weight, qscales)
        fused = _quantized(weight, qscales, gate)
        for proj, rows in ((gate, slice(0, n)), (up, slice(n, 2 * n))):
            proj.weight, proj.scales = weight[:, rows], qscales[:, rows]
        experts = FusedExperts(fused, down, None)
    elif kind is SwitchLinear:
        weight = mx.concatenate([gate["weight"], up["weight"]], axis=1)
        mx.eval(weight)
        fused = _dense(weight)
        gate.weight, up.weight = weight[:, :n], weight[:, n:]
        experts = FusedExperts(fused, down, None)
    else:
        return None, f"expert_type_{kind.__name__}"
    return experts, None


def same_source(held, key) -> bool:
    return len(held) == len(key) and all(a is b for a, b in zip(held, key))


def decline_reason() -> str | None:
    if _invariant.active():
        return "invariant_prefill"
    if _grouped.MODE != "off":
        return "grouped_experts_mode"
    return None


def _weighted_sum_admitted(y, inv_order, weights, indices, do_sort):
    from .. import qwen4_moe_weighted_sum as ws

    if not ws.runtime_supported():
        return "not_gpu"
    admission = ws.admit_moe_weighted_sum(
        x_sorted=y, inv_order=inv_order, scores=weights, indices=indices,
        do_sort=do_sort, training=False,
    )
    return None if admission.accepted else admission.reason


def fused_call(moe, experts, x, weighted_sum=False):
    """``moe(x)`` through the stacked experts; counters live on ``moe``."""
    switch = moe.switch_mlp
    counts = moe.__dict__.setdefault("moe_fast_counts", {})

    def note(key):
        counts[key] = counts.get(key, 0) + 1

    from .language import _fp32

    indices, weights = route_compiled(
        x, _fp32(moe, "gate_weight"), moe.e_score_correction_bias, moe.top_k, moe.route_scale
    )
    xe = mx.expand_dims(x, (-2, -3))
    token_length = int(indices.shape[-2]) if indices.ndim >= 2 else 1
    do_sort, counter = _switch_sort_decision(
        switch._sort_policy, assignments=int(indices.size), token_length=token_length,
        invariant=False,
    )
    switch._sort_counts[counter] += 1
    idx, inv_order = indices, None
    if do_sort:
        xe, idx, inv_order = _gather_sort(xe, indices)
    gate_up = experts.gate_up(xe, idx, sorted_indices=do_sort)
    act = switch.activation
    scales = None if experts.scales is None else experts.scales[idx]
    if scales is None:
        hidden = _gate_up_act(gate_up, act.gate_cap, act.up_cap)
    else:
        hidden = _scaled_gate_up_act(gate_up, scales[..., 0:1, None], act.gate_cap, act.up_cap)
    y = experts.down(hidden, idx, sorted_indices=do_sort)
    note("fused")
    # Multi-token sorted calls only (prefill chunks, verify windows): a
    # one-token batch at any width keeps the reference combine.
    if do_sort and weighted_sum and token_length > 1:
        reason = _weighted_sum_admitted(y, inv_order, weights, indices, do_sort)
        if reason is None:
            from .. import qwen4_moe_weighted_sum as ws

            if scales is not None:
                y = _scale_rows(y, scales[..., 1:2, None])
            note("weighted_sum")
            return ws.moe_weighted_sum(y, inv_order, weights).astype(y.dtype)
        note("weighted_sum_declined:" + reason)
    if do_sort:
        y = _scatter_unsort(y, inv_order, indices.shape)
        if scales is not None:
            scales = _scatter_unsort(scales, inv_order, indices.shape)
    y = y.squeeze(-2)
    if scales is None:
        return _combine(y, weights)
    return _scaled_combine(y, scales[..., 1:2], weights)
