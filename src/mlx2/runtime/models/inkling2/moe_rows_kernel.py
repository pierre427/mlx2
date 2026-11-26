# SPDX-License-Identifier: Apache-2.0
# Mechanism (row-exact verify experts run with the pairs of one output tile
# back to back, so pairs that share an expert read its tile from cache) from
# oMLX (Apache-2.0, https://github.com/jundot/omlx) #5572 / #5617 / #5638,
# as described in those pull requests; no oMLX code copied.  The per-pair
# arithmetic is moe_decode_kernel's (MLX 0.32.4 fp_qmv_impl, MIT).  See
# docs/PROVENANCE.md and provenance/inkling2-omlx-moe-rows.json.
"""``inkling2_moe_rows_kernel``: the routed-expert forward of a 2-16 row
Inkling 2 ``SparseMoE`` call (a verify block or a short prefill chunk) with
every row computing exactly its one-token arithmetic (default off).

Below MLX's streaming floor every (row, expert) pair of a multi-row call is
its own ``gather_qmv`` launch slice that re-reads its expert's weights, even
when the next row chose the same expert (consecutive tokens share most of
their experts).  ``moe_panel_kernel`` read each distinct expert once by
giving one threadgroup to each expert and looping over its rows; that leaves
few threadgroups and was slower at every width measured.  This module keeps
one threadgroup per pair, as the shipped gather does, and only changes the
order of the grid:

1. ``gate_up``: grid x = pair (``row * top_k + slot``), grid y = 4-row output
   tile.  Threadgroups are dispatched x first, so every pair of one tile runs
   back to back and an expert tile named by several pairs is read from cache
   after the first.  Each threadgroup is ``moe_decode_kernel``'s gate/up
   threadgroup for that pair (two simdgroups: gate rows and up rows, the BF16
   rounding of the gather, the per-expert ``weight_scale_2`` law and the
   softcapped SwiGLU), written per pair.
2. ``down``: grid x = row, grid y = 4-row output tile, ``top_k`` simdgroups
   per threadgroup: ``moe_decode_kernel``'s down threadgroup for that row
   (the same gather order, scale law, routing normalization and weighted sum
   in MLX's ``col_reduce_small`` order), BF16 out.

So each row's output is the one-token kernel's output for that row, which
the decode-kernel parity sweep found bitwise equal to MLX's one-token
gather; the multi-row shipped route runs the same per-pair ``fp_qmv_impl``
(it never streams in this range), so the target is bitwise equality with the
shipped fused route at the same width.  The router is the shipped op
sequence at the shipped shape (``moe_decode_kernel.route_compiled``).  The
Metal parity script checks both on the real checkpoint.

Declines (counted in ``moe_fast_counts`` as ``rows_kernel_declined:<why>``;
the caller runs the shipped route): ``moe_decode_kernel``'s (not on the GPU,
non-BF16 activations, act-quant, non-NVFP4 experts such as the Mini's MXFP8
layers 20-23, top_k above 8, widths where MLX picks ``qmv_fast``), fewer than
2 or more than ``MAX_ROWS`` rows, and any call where the shipped route would
stream (``gather_qmm_rhs``) or combine through ``qwen4_moe_weighted_sum``
(``moe_panel_kernel``'s rules, whose orders this module does not reproduce).
"""

from __future__ import annotations

import mlx.core as mx

from . import moe_decode_kernel as _mdk

MAX_ROWS = 16

_GATE_UP_ROWS_SOURCE = r"""
    const uint pair = threadgroup_position_in_grid.x;
    const uint tg = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lid = thread_index_in_simdgroup;
    const uint row = pair / TOPK;
    const uint slot = pair - row * TOPK;
    const uint e = order[row * NE + slot];
    const uint row0 = sg * I + tg * 4;
    const size_t first = (size_t)e * (2 * I) + row0;
    float result[4] = {0, 0, 0, 0};
    mlx2_rows4<bfloat16_t, H>((const device uint8_t*)w + first * (H / 2),
                              s + first * (H / 16), x + (size_t)row * H, lid, result);
    for (int r = 0; r < 4; r++) result[r] = simd_sum(result[r]);
    threadgroup float up_vals[4];
    const float scale = scale2[e * 2];
    if (sg == 1 && lid < 4) {
      up_vals[lid] = mlx2_scaled(result[lid], scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lid < 4) {
      float gate = mlx2_scaled(result[lid], scale);
      hidden[(size_t)pair * I + tg * 4 + lid] = static_cast<bfloat16_t>(
          mlx2_swiglu(up_vals[lid], gate, GATE_CAP, UP_CAP));
    }
"""

_DOWN_ROWS_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint tg = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lid = thread_index_in_simdgroup;
    const device uint* ord = order + (size_t)row * NE;
    const device float* sc = scores + (size_t)row * NE;
    const uint e = ord[sg];
    const uint row0 = tg * 4;
    const size_t first = (size_t)e * H + row0;
    float result[4] = {0, 0, 0, 0};
    mlx2_rows4<bfloat16_t, I>((const device uint8_t*)w + first * (I / 2),
                              s + first * (I / 16),
                              hidden + ((size_t)row * TOPK + sg) * I, lid, result);
    for (int r = 0; r < 4; r++) result[r] = simd_sum(result[r]);
    threadgroup float part[TOPK * 4];
    if (lid < 4) {
      part[sg * 4 + lid] = mlx2_scaled(result[lid], scale2[e * 2 + 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lid < 4) {
      float total_s = 0;
      if (ROUTE_HALVES) {
        float a = 0, b = 0;
        for (int k = 0; k < TOPK / 2; k++) a = mlx2_add(sc[ord[k]], a);
        for (int k = TOPK / 2; k < TOPK; k++) b = mlx2_add(sc[ord[k]], b);
        total_s = mlx2_add(a, b);
      } else {
        for (int k = 0; k < TOPK; k++) total_s = mlx2_add(sc[ord[k]], total_s);
      }
      const float denom = mlx2_add(total_s, EPS);
      float total = 0;
      for (int k = 0; k < TOPK; k++) {
        float weight = sc[ord[k]] / denom;
        weight = weight * ROUTE_SCALE;
        float p = part[k * 4 + lid] * weight;
        total = mlx2_add(p, total);
      }
      y[(size_t)row * H + row0 + lid] = static_cast<bfloat16_t>(total);
    }
"""

_kernels: dict = {}


def _kernel(kind, contract, route_sum, caps):
    key = (kind, contract, route_sum, caps)
    kernel = _kernels.get(key)
    if kernel is None:
        gate_cap, up_cap, route_scale = caps
        consts = (f"#define GATE_CAP {_mdk._f32_literal(gate_cap)}\n"
                  f"#define UP_CAP {_mdk._f32_literal(up_cap)}\n"
                  f"#define ROUTE_SCALE {_mdk._f32_literal(route_scale)}\n"
                  f"#define EPS {_mdk._f32_literal(1e-20)}\n"
                  f"#define ROUTE_HALVES {1 if route_sum == 'halves' else 0}\n")
        header = consts + _mdk._header(contract)
        if kind == "gate_up":
            kernel = mx.fast.metal_kernel(
                name=f"inkling2_moe_rows_gate_up_{contract}",
                input_names=["x", "order", "w", "s", "scale2"],
                output_names=["hidden"], source=_GATE_UP_ROWS_SOURCE, header=header,
            )
        else:
            kernel = mx.fast.metal_kernel(
                name=f"inkling2_moe_rows_down_{contract}_{route_sum}",
                input_names=["hidden", "order", "scores", "w", "s", "scale2"],
                output_names=["y"], source=_DOWN_ROWS_SOURCE, header=header,
            )
        _kernels[key] = kernel
    return kernel


def _count(moe, key):
    counts = moe.__dict__.setdefault("moe_fast_counts", {})
    counts[key] = counts.get(key, 0) + 1


def device_ok():
    """GPU only (CPU tests patch this together with ``launch``)."""
    return mx.default_device() == mx.gpu


def decline_reason(moe, experts, x, weighted_sum=False):
    """None when the kernels take ``moe(x)``."""
    from . import moe_panel_kernel as _panel

    if not device_ok():
        return "not_gpu"
    hidden = int(x.shape[-1])
    rows = int(x.size) // hidden if hidden else 0
    if not 2 <= rows <= MAX_ROWS:
        return "rows"
    # The one-token kernel's own rules (dtype, act-quant, NVFP4 experts,
    # top_k, the qmv_fast widths), applied to one row of the call.
    reason = _mdk.decline_reason(moe, experts, x.reshape(-1)[:hidden].reshape(1, 1, hidden))
    if reason is not None:
        return reason
    token_length = int(x.shape[-2]) if x.ndim >= 2 else 1
    streams = _panel._shipped_streams(moe, experts, "nvfp4", rows, token_length)
    if streams is not None:
        return streams
    if _panel.shipped_weighted_sum_engages(moe, rows, token_length, weighted_sum):
        return "weighted_sum_order_unproven"
    return None


def launch(moe, experts, x, order, scores, *, contract=None, route_sum=None,
           return_hidden=False):
    """The two kernels: ``y`` with ``x``'s shape (BF16); with ``return_hidden``
    also the ``[rows * top_k, inter]`` activation rows (pair order)."""
    contract = _mdk.QDOT_CONTRACT if contract is None else contract
    route_sum = _mdk.ROUTE_SUM if route_sum is None else route_sum
    if contract not in _mdk._CONTRACTS or route_sum not in _mdk._ROUTE_SUMS:
        raise ValueError(f"moe_rows_kernel: contract {contract!r} / route_sum {route_sum!r}")
    hidden = int(x.shape[-1])
    rows = int(x.size) // hidden
    inter = int(experts.down["scales"].shape[-1]) * 16
    top_k = int(moe.top_k)
    act = moe.switch_mlp.activation
    caps = (float(act.gate_cap), float(act.up_cap), float(moe.route_scale))
    order = order.reshape(rows, -1)
    scores = scores.reshape(rows, -1)
    num = int(order.shape[-1])
    gate_up = _kernel("gate_up", contract, route_sum, caps)
    (hidden_rows,) = gate_up(
        inputs=[x.reshape(rows, hidden), order, experts.gate_up["weight"],
                experts.gate_up["scales"], experts.scales],
        template=[("H", hidden), ("I", inter), ("TOPK", top_k), ("NE", num)],
        grid=(rows * top_k * 64, inter // 4, 1), threadgroup=(64, 1, 1),
        output_shapes=[(rows * top_k, inter)], output_dtypes=[mx.bfloat16],
    )
    down = _kernel("down", contract, route_sum, caps)
    (y,) = down(
        inputs=[hidden_rows, order, scores, experts.down["weight"], experts.down["scales"],
                experts.scales],
        template=[("H", hidden), ("I", inter), ("TOPK", top_k), ("NE", num)],
        grid=(rows * 32 * top_k, hidden // 4, 1), threadgroup=(32 * top_k, 1, 1),
        output_shapes=[(rows, hidden)], output_dtypes=[mx.bfloat16],
    )
    y = y.reshape(x.shape)
    return (y, hidden_rows) if return_hidden else y


def law(moe, experts, x, order=None, scores=None):
    """The target: each row through the shipped fused route as a one-token
    call (``order``/``scores`` unused: the one-token call routes itself).
    The CPU tests run the plumbing against it; the Metal parity script
    compares the kernels with it and with the shipped multi-row call."""
    from . import moe_fast

    del order, scores
    hidden = int(x.shape[-1])
    rows = int(x.size) // hidden
    xs = x.reshape(rows, 1, hidden)
    armed = moe.__dict__.get("_moe_rows_kernel", False)
    object.__setattr__(moe, "_moe_rows_kernel", False)  # the shipped route, not this hook
    try:
        out = [moe_fast.fused_call(moe, experts, xs[r : r + 1]) for r in range(rows)]
    finally:
        object.__setattr__(moe, "_moe_rows_kernel", armed)
    return mx.concatenate([o.reshape(1, hidden) for o in out], axis=0).reshape(x.shape)


def call(moe, experts, x, router_weight=None, weighted_sum=False):
    """``moe(x)`` through the kernels, or None (counted) for the shipped route."""
    reason = decline_reason(moe, experts, x, weighted_sum)
    if reason is not None:
        _count(moe, "rows_kernel_declined:" + reason)
        return None
    if router_weight is None:
        from .language import _fp32

        router_weight = _fp32(moe, "gate_weight")
    order, scores = _mdk.route_compiled(x, router_weight, moe.e_score_correction_bias,
                                        moe.top_k)
    y = launch(moe, experts, x, order, scores)
    _count(moe, "rows_kernel")
    _count(moe, "fused")
    counts = moe.switch_mlp._sort_counts
    counts["rows_kernel"] = counts.get("rows_kernel", 0) + 1
    return y
