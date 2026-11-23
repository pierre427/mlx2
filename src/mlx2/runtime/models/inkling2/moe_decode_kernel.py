# SPDX-License-Identifier: Apache-2.0
"""``inkling2_moe_decode_kernel``: the one-token NVFP4 routed-expert forward
of the Inkling 2 ``SparseMoE`` in two Metal launches (default off).

Original mlx2 code (2026-11-22).  The per-row dot-product order, the FP4
(E2M1) and FP8 (E4M3) decodes reproduce MLX 0.32.4's ``fp_qmv_impl`` /
``qdot`` / ``qdot_safe`` / ``fp4_e2m1`` / ``fp8_e4m3`` from
``mlx/backend/metal/kernels/{fp_quantized,fp4,fp8}.h`` (MIT); the epilogues
reproduce the shipped ``moe_fast`` compiled kernels op for op.  See
docs/PROVENANCE.md.

What runs, for one token (``x.size == hidden``) on NVFP4 experts:

1. ``router`` (compiled, the shipped ops): ``sigmoid(x @ Wr.T)``, the bias
   add and negation, ``argpartition`` (MLX sorts on the GPU).  Its outputs
   are the full order and the scores; the kernels read ``order[:top_k]``
   and ``scores[order[k]]`` in place (no slice copy, no ``take``).
2. ``gate_up`` kernel: per (expert slot, 4 hidden rows) one threadgroup of
   two simdgroups; simdgroup 0 computes the 4 gate rows and simdgroup 1 the
   4 up rows of the stacked table with MLX's one-token gather lane layout
   and summation order, then the BF16 rounding of the gather, the per-expert
   ``weight_scale_2`` law (FP32 multiply, BF16 round) and the softcapped
   SwiGLU (``moe_fast._swiglu_law``) are applied in-register.
3. ``down`` kernel: per 4 output rows one threadgroup of ``top_k``
   simdgroups, one per routed expert; the same gather order, BF16 rounding,
   down ``weight_scale_2`` law, then the routing weight normalization
   (``scores / (sum + 1e-20) * route_scale``) and the weighted sum in MLX's
   ``col_reduce_small`` order (``p_j + total`` for j = 0..K-1), BF16 out.

So a decode MoE call is ~6 launches (x cast, router GEMV, sigmoid/bias,
argsort, two kernels) instead of ~16 (the same router plus slice/take, two
normalization launches, two gathers with their index aranges, the scale
take, the activation, the down-scale multiply, the reduction and the cast).

Exactness is the target, not a claim: every rounding point of the shipped
route is kept, but two orders are MLX dispatch details this module cannot
see from Python, so each is a module switch the Metal parity test sweeps on
the real checkpoint (``scripts/research/inkling2_moe_decode_kernel_ab.py``):

- ``QDOT_CONTRACT``: the FP contraction mode the dot products are compiled
  with (MLX's own kernels are built offline with ``-fno-fast-math``; whether
  that fuses ``x0*d0 + x1*d1`` into an FMA is the compiler's default).
- ``ROUTE_SUM``: the order of the 8-score normalization sum (``sequential``
  as MLX's ``row_reduce_small``, or ``halves`` as its ``all_reduce`` with 4
  reads per thread: ``(s0+s1+s2+s3) + (s4+...+s7)``).  The sweep found
  ``sequential`` (the default).

Declines (counted in ``moe_fast_counts`` as ``decode_kernel_declined:<why>``,
caller runs the shipped route): not on the GPU, more than one token, non-BF16
activations, act-quant on, non-NVFP4 experts (the MXFP8 layers 20-23 of the
Mini), top_k above 8 (``top_k``: MLX's column reduction changes order there),
or a width where MLX would pick its ``qmv_fast`` kernel (K % 512 == 0, a
different order this module does not reproduce).

Counters: an engaged call counts ``decode_kernel`` and ``fused`` (so the
receipt's ``moe_fused_calls`` still counts every fused-route call, as with
``inkling2_compile_decode_blocks``); ``moe_decode_kernel_calls`` is the
kernel's share of it.
"""

from __future__ import annotations

import struct

import mlx.core as mx

# Module switches swept by the Metal parity script (see the docstring).
# Defaults from the real-checkpoint sweep (Inkling-2-Mini@440956c8, M5 Max,
# MLX 0.32.4; swept at mlx2 193b0f95a, whose default was still "halves", and
# confirmed as the module default at 6b0946589): default/on/off x sequential are bitwise on
# all 44 NVFP4 visits of a decode step (output and activation rows); halves
# differs on 4/44 (the route sum is MLX row_reduce_small's sequential loop),
# "fast" contraction differs on 1/44 activations.
QDOT_CONTRACT = "default"  # "default" (no pragma), "on", "off" or "fast"
ROUTE_SUM = "sequential"  # "sequential" or "halves"
_CONTRACTS = ("default", "on", "off", "fast")
_ROUTE_SUMS = ("sequential", "halves")
_FAST_QMV_K = 512  # MLX gather_qmv_fast K alignment for 4 bits (8 * 2 * 32)
_MAX_TOP_K = 8  # col_reduce_small: one y-thread per routed row up to 8


def _route(x, weight, bias, top_k):
    # The shipped ``moe_fast._route`` up to the argpartition (same ops).
    scores = mx.sigmoid(x.astype(mx.float32) @ weight.astype(mx.float32).T)
    order = mx.argpartition(-(scores + bias), top_k - 1, axis=-1)
    return mx.stop_gradient(order), scores


route_compiled = mx.compile(_route)


def _f32_literal(value):
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    return f"as_type<float>(0x{bits:08x}u)"


def _header(contract):
    pragma = "" if contract == "default" else f"#pragma METAL fp contract({contract})\n"
    return pragma + r"""
// MLX fp4_e2m1::operator float16_t (fp4.h), via Dequantize<4>.
inline float mlx2_dq4(uint8_t bits) {
  half converted = as_type<half>(ushort((bits & 7) << 9));
  converted *= 16384.0;
  return static_cast<float>(bits & 8 ? -converted : converted);
}

// MLX fp8_e4m3::operator float16_t (fp8.h), via dequantize_scale<float, 16>.
inline float mlx2_e4m3(uint8_t bits) {
  uint16_t v = bits & 127;
  uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;
  uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
  half converted = as_type<half>(u);
  return static_cast<float>(converted * 256.0);
}

// MLX qdot_safe<float, 8, 4> (equal to qdot for n == 8).
inline float mlx2_qdot(const device uint8_t* w, const thread float* x, float scale, int n) {
  float accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (n / 4); i++) {
    accum +=
        (x[4 * i] * mlx2_dq4(ws[i]) +
         x[4 * i + 1] * mlx2_dq4(ws[i] >> 4) +
         x[4 * i + 2] * mlx2_dq4(ws[i] >> 8) +
         x[4 * i + 3] * mlx2_dq4(ws[i] >> 12));
  }
  return scale * accum;
}

// MLX fp_qmv_impl (packs_per_thread 1, 8 values per lane, 256-value blocks,
// the out_vec_size >= 8 branch) for 4 consecutive rows; result before the
// simd_sum.  ``w``/``s`` point at the first row, ``x`` at the vector.
template <typename T, int KDIM>
inline void mlx2_rows4(const device uint8_t* w, const device uint8_t* s,
                       const device T* x, uint lid, thread float* result) {
  constexpr int block_size = 256;
  constexpr int row_w = KDIM / 2;
  constexpr int row_g = KDIM / 16;
  float x_thread[8];
  w += lid * 4;
  s += lid / 2;
  x += lid * 8;
  int k = 0;
  for (; k < KDIM - block_size; k += block_size) {
    for (int i = 0; i < 8; i++) x_thread[i] = x[i];
    for (int row = 0; row < 4; row++) {
      float sc = mlx2_e4m3(s[row * row_g]);
      result[row] += mlx2_qdot(w + row * row_w, x_thread, sc, 8);
    }
    w += block_size / 2;
    s += block_size / 16;
    x += block_size;
  }
  const int remaining = clamp(static_cast<int>(KDIM - k - lid * 8), 0, 8);
  if (remaining > 0) {
    for (int i = 0; i < remaining; i++) x_thread[i] = x[i];
    for (int i = remaining; i < 8; i++) x_thread[i] = 0;
    for (int row = 0; row < 4; row++) {
      float sc = mlx2_e4m3(s[row * row_g]);
      result[row] += mlx2_qdot(w + row * row_w, x_thread, sc, remaining);
    }
  }
}
""" + r"""
#pragma METAL fp contract(off)

// moe_fast._scale_law: FP32 multiply of the BF16 product, BF16 round.
inline float mlx2_scaled(float raw, float scale) {
  bfloat16_t y = static_cast<bfloat16_t>(raw);
  return static_cast<float>(static_cast<bfloat16_t>(static_cast<float>(y) * scale));
}

// MLX Sigmoid (unary_ops.h).
inline float mlx2_sigmoid(float x) {
  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return (x < 0) ? y : 1 - y;
}

// moe_fast._swiglu_law on BF16-valued inputs.
inline float mlx2_swiglu(float up, float gate, float gate_cap, float up_cap) {
  float a = metal::precise::tanh(gate / gate_cap);
  a = a * gate_cap;
  a = a * mlx2_sigmoid(gate);
  a = a * metal::precise::tanh(up / up_cap);
  return a * up_cap;
}

inline float mlx2_add(float a, float b) { return a + b; }
"""


_GATE_UP_SOURCE = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint slot = threadgroup_position_in_grid.y;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lid = thread_index_in_simdgroup;
    const uint e = order[slot];
    const uint row0 = sg * I + tg * 4;
    const size_t first = (size_t)e * (2 * I) + row0;
    float result[4] = {0, 0, 0, 0};
    mlx2_rows4<bfloat16_t, H>((const device uint8_t*)w + first * (H / 2),
                              s + first * (H / 16), x, lid, result);
    for (int r = 0; r < 4; r++) result[r] = simd_sum(result[r]);
    threadgroup float up_vals[4];
    const float scale = scale2[e * 2];
    if (sg == 1 && lid < 4) {
      up_vals[lid] = mlx2_scaled(result[lid], scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lid < 4) {
      float gate = mlx2_scaled(result[lid], scale);
      hidden[slot * I + tg * 4 + lid] = static_cast<bfloat16_t>(
          mlx2_swiglu(up_vals[lid], gate, GATE_CAP, UP_CAP));
    }
"""

_DOWN_SOURCE = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lid = thread_index_in_simdgroup;
    const uint e = order[sg];
    const uint row0 = tg * 4;
    const size_t first = (size_t)e * H + row0;
    float result[4] = {0, 0, 0, 0};
    mlx2_rows4<bfloat16_t, I>((const device uint8_t*)w + first * (I / 2),
                              s + first * (I / 16), hidden + sg * I, lid, result);
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
        for (int k = 0; k < TOPK / 2; k++) a = mlx2_add(scores[order[k]], a);
        for (int k = TOPK / 2; k < TOPK; k++) b = mlx2_add(scores[order[k]], b);
        total_s = mlx2_add(a, b);
      } else {
        for (int k = 0; k < TOPK; k++) total_s = mlx2_add(scores[order[k]], total_s);
      }
      const float denom = mlx2_add(total_s, EPS);
      float total = 0;
      for (int k = 0; k < TOPK; k++) {
        float weight = scores[order[k]] / denom;
        weight = weight * ROUTE_SCALE;
        float p = part[k * 4 + lid] * weight;
        total = mlx2_add(p, total);
      }
      y[row0 + lid] = static_cast<bfloat16_t>(total);
    }
"""

_kernels: dict = {}


def _kernel(kind, contract, route_sum, caps):
    key = (kind, contract, route_sum, caps)
    kernel = _kernels.get(key)
    if kernel is None:
        gate_cap, up_cap, route_scale = caps
        consts = (f"#define GATE_CAP {_f32_literal(gate_cap)}\n"
                  f"#define UP_CAP {_f32_literal(up_cap)}\n"
                  f"#define ROUTE_SCALE {_f32_literal(route_scale)}\n"
                  f"#define EPS {_f32_literal(1e-20)}\n"
                  f"#define ROUTE_HALVES {1 if route_sum == 'halves' else 0}\n")
        if kind == "gate_up":
            kernel = mx.fast.metal_kernel(
                name=f"inkling2_moe_decode_gate_up_{contract}",
                input_names=["x", "order", "w", "s", "scale2"],
                output_names=["hidden"],
                source=_GATE_UP_SOURCE, header=consts + _header(contract),
            )
        else:
            kernel = mx.fast.metal_kernel(
                name=f"inkling2_moe_decode_down_{contract}_{route_sum}",
                input_names=["hidden", "order", "scores", "w", "s", "scale2"],
                output_names=["y"],
                source=_DOWN_SOURCE, header=consts + _header(contract),
            )
        _kernels[key] = kernel
    return kernel


def _count(moe, key):
    counts = moe.__dict__.setdefault("moe_fast_counts", {})
    counts[key] = counts.get(key, 0) + 1


def device_ok():
    """The kernels run on the GPU only (CPU tests patch this together with
    ``launch`` to run the plumbing against ``law``)."""
    return mx.default_device() == mx.gpu


def decline_reason(moe, experts, x):
    from ...nvfp4_linear import ScaledNVFP4SwitchLinear

    if not device_ok():
        return "not_gpu"
    hidden = int(x.shape[-1])
    if x.size != hidden:
        return "rows"
    if x.dtype != mx.bfloat16:
        return "dtype"
    if moe.__dict__.get("_act_quant"):
        return "act_quant"
    if (experts is None or experts.scales is None
            or type(experts.gate_up) is not ScaledNVFP4SwitchLinear
            or type(experts.down) is not ScaledNVFP4SwitchLinear):
        return "expert_format"
    inter = int(experts.down["scales"].shape[-1]) * 16
    top_k = int(moe.top_k)
    if not 1 <= top_k <= _MAX_TOP_K or (ROUTE_SUM == "halves" and top_k % 2):
        # The weighted sum reproduces MLX col_reduce_small only while its
        # threadgroup has one y-thread per routed row (top_k <= 8); above
        # that the y-threads stride by 8 first, and at 32 rows MLX picks
        # another reduction kernel. Swept on the GPU: top_k 2 (tiny) and 8.
        return "top_k"
    if (hidden % 16 or inter % 16 or hidden % 4
            or hidden % _FAST_QMV_K == 0 or inter % _FAST_QMV_K == 0):
        # K % 512 == 0: MLX would run gather_qmv_fast (16 values per lane),
        # a different order.
        return "geometry"
    if int(experts.gate_up["weight"].shape[1]) != 2 * inter:
        return "geometry"
    return None


def launch(moe, experts, x, order, scores, *, contract=None, route_sum=None,
           return_hidden=False):
    """The two kernels: ``y`` with ``x``'s shape (BF16); with
    ``return_hidden`` also the ``[top_k, inter]`` activation rows."""
    contract = QDOT_CONTRACT if contract is None else contract
    route_sum = ROUTE_SUM if route_sum is None else route_sum
    if contract not in _CONTRACTS or route_sum not in _ROUTE_SUMS:
        raise ValueError(f"moe_decode_kernel: contract {contract!r} / route_sum {route_sum!r}")
    hidden = int(x.shape[-1])
    inter = int(experts.down["scales"].shape[-1]) * 16
    top_k = int(moe.top_k)
    act = moe.switch_mlp.activation
    caps = (float(act.gate_cap), float(act.up_cap), float(moe.route_scale))
    order = order.reshape(-1)
    scores = scores.reshape(-1)
    gate_up = _kernel("gate_up", contract, route_sum, caps)
    (hidden_rows,) = gate_up(
        inputs=[x.reshape(hidden), order, experts.gate_up["weight"],
                experts.gate_up["scales"], experts.scales],
        template=[("H", hidden), ("I", inter)],
        grid=(inter // 4 * 64, top_k, 1), threadgroup=(64, 1, 1),
        output_shapes=[(top_k, inter)], output_dtypes=[mx.bfloat16],
    )
    down = _kernel("down", contract, route_sum, caps)
    (y,) = down(
        inputs=[hidden_rows, order, scores, experts.down["weight"], experts.down["scales"],
                experts.scales],
        template=[("H", hidden), ("I", inter), ("TOPK", top_k)],
        grid=(hidden // 4 * 32 * top_k, 1, 1), threadgroup=(32 * top_k, 1, 1),
        output_shapes=[(hidden,)], output_dtypes=[mx.bfloat16],
    )
    y = y.reshape(x.shape)
    return (y, hidden_rows) if return_hidden else y


def law(moe, experts, x, order, scores):
    """The kernels' arithmetic in MLX ops (rounding points, slot order and
    epilogues; the dot products in MLX's own order, not the kernel's lane
    order).  The CPU tests run the plumbing against it; the Metal parity
    script uses it as a third reference."""
    from .moe_fast import _scale_law, _swiglu_law

    top_k = int(moe.top_k)
    idx = order.reshape(-1)[:top_k]
    s = scores.reshape(-1)[idx]
    if ROUTE_SUM == "halves":
        half = top_k // 2
        total = s[:half].sum() + s[half:].sum()
    else:
        total = s.sum()
    weights = s / (total + 1e-20) * moe.route_scale
    xe = x.reshape(1, 1, -1)
    table = experts.scales[idx]
    gate_up = experts.gate_up(mx.expand_dims(xe, (-2, -3)), idx.reshape(1, 1, -1))
    inter = gate_up.shape[-1] // 2
    gate = _scale_law(gate_up[..., :inter], table[:, 0:1, None])
    up = _scale_law(gate_up[..., inter:], table[:, 0:1, None])
    act = moe.switch_mlp.activation
    h = _swiglu_law(up, gate, act.gate_cap, act.up_cap)
    down = experts.down(h, idx.reshape(1, 1, -1)).squeeze(-2)
    down = _scale_law(down, table[:, 1:2])
    out = (down.astype(mx.float32) * weights[:, None]).sum(axis=-2).astype(down.dtype)
    return out.reshape(x.shape)


def call(moe, experts, x):
    """``moe(x)`` through the kernels, or None (counted) for the shipped route."""
    from .language import _fp32

    reason = decline_reason(moe, experts, x)
    if reason is not None:
        _count(moe, "decode_kernel_declined:" + reason)
        return None
    order, scores = route_compiled(x, _fp32(moe, "gate_weight"), moe.e_score_correction_bias,
                                   moe.top_k)
    y = launch(moe, experts, x, order, scores)
    # Counted after the launch was built. ``fused`` too, as the compiled
    # decode block does: these calls are fused-route calls, so the receipt's
    # moe_fused_calls keeps counting every call that took a fused route.
    _count(moe, "decode_kernel")
    _count(moe, "fused")
    return y
