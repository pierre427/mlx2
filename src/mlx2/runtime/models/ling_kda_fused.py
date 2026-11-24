# SPDX-License-Identifier: MIT
"""One-launch Ling KDA decode body (``ling3_flash`` key ``kda_fused=decode_v1``).

Port of the GLM-5.3 ``glm53_kda_fused=decode_v1`` launch structure
(``glm53_kda_fused.py``) to Ling's own reference chain. For one B1 decode
token, one Metal launch per KDA layer covers everything in
``bailing_hybrid.KimiDeltaAttention.__call__`` between the input projections
and ``o_proj``:

    three ShortConv1d windows (q, k, v): concat(conv state, raw) -> depthwise
        conv1d (4 taps) -> nn.silu; new conv state = rows 1..3
    q = scale**2 * rms_norm(q), k = scale * rms_norm(k)   (mx.fast, eps 1e-6)
    g = exp(lower_bound * sigmoid(exp(A_log) * (a + dt_bias)))
                                     (gated_delta.compute_lower_bound_g)
    beta = sigmoid(b)                (fp32, gated_delta_update)
    gated delta rule over the 32 x 128 x 128 fp32 state (gated_delta_step_vec)
    o_norm(y) * sigmoid(gate)        (KimiDeltaAttention.output_norm_gate)

One threadgroup owns one head: the conv is depthwise and every norm reduces
over one head's 128 channels, so no cross-head boundary exists.

Exactness: the outputs, the three conv states and the recurrent state equal
the reference chain bit for bit. Ling's chain runs in the activation dtype
(bf16), not GLM's fp32, so every rounding point of the reference is a
``static_cast<T>`` here. The kernels reproduced (MLX v0.32.4; mlx2
gated_delta.py):

* conv: ``depthwise_conv_1d`` (AOT): ``float acc = 0; acc += float(in) * w``
  over the taps, conv state rows first, then ``static_cast<T>(acc)``.
  ``CONV_FMA`` / ``RMS_FMA`` select a fused or a separate multiply-add; for
  bf16 operands every product is exact in fp32 (barring results below
  2**-126), so the choice cannot change a bit and the Metal tests show all
  four choices agree. They are kept as test switches only.
* SiLU: ``nn.silu`` is ``mx.compile``'d: ``x * sigmoid(x)`` in T with the
  JIT Sigmoid. ``qwen4_fused_gdn.served_silu_exp`` probes that form against
  ``nn.silu`` over every bf16 encoding and supplies the matching spelling.
* q/k: ``rms_single_row`` (AOT, axis 128, 32 threads, 4 reads) with the
  no-weight ``array(1, T)`` weight at stride 0, then the eager bf16 multiply
  by the scalar rounded to T (``SCALE_Q`` / ``SCALE_K``, taken from MLX's own
  scalar promotion).
* g: ``compute_lower_bound_g`` is ``mx.compile``'d in fp32: ``Exp`` is
  ``metal::precise::exp``, the Sigmoid is the compiled one (served SiLU
  spelling).
* beta: eager fp32 ``mx.sigmoid`` (``ling_fused._GATE`` spelling for fp32).
* delta rule: the ``gated_delta_step_vec`` loop verbatim, with q/k/v held as
  T exactly as the reference reads them from device memory, and the readout
  cast to T (``InT`` = bf16: the fp32 state's exponent range fits bf16, so
  the reference does not widen).
* output: ``rms_single_row`` with ``o_norm.weight``, then the eager bf16
  ``mx.sigmoid`` (``ling_fused._GATE`` spelling) and the eager bf16 multiply.

Steps that are separate AOT kernels in the reference are evaluated under
``fp contract(off)``, with the AOT kernels' contractions written as
``metal::fma``; JIT-compiled steps keep their own expressions.

Launch shape: 1024 threads per threadgroup, verified on M5 Max only. A
pipeline whose ``maxTotalThreadsPerThreadgroup`` is lower would fail at eval
time rather than decline; run the Metal tests on such a host before enabling.

Counters: while this body engages, the per-piece counters it replaces
(``separate_conv``, ``merged_conv``, ``eager_norm``, ``fused_norm``) stop
counting; A/B those pieces with ``kda_fused`` off.

Callers keep the reference chain for differentiable-reference forwards, prefill, B > 1, speculation (the APCv2
rollback recorder needs the eager scan operands), masks, ragged lengths, CPU,
other dtypes and other shapes (``ling_kda_fused_admits``).
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

HEADS, DIM, TAPS = 32, 128, 4
INNER = HEADS * DIM
THREADS = 1024

# Contraction of the ahead-of-time MLX kernels (-fno-fast-math): the Metal
# tests pin these against the venv's MLX build.
CONV_FMA = True
RMS_FMA = True


def _hexf(value):
    """A float32 literal that names exactly ``np.float32(value)``."""
    return float(np.float32(value)).hex() + "f"


def _mac(acc, x, w, fused):
    """``acc += x * w`` as one AOT kernel statement compiles it."""
    if fused:
        return f"{acc} = metal::fma({x}, {w}, {acc});"
    return f"{{ const float p_ = {x} * {w}; {acc} = {acc} + p_; }}"


def _eager_sigmoid(dtype_name, x, out, spelling):
    """``ling_fused._SIGMOID`` (the eager ``mx.sigmoid`` body its gate
    probes) on ``x`` of type ``dtype_name``, stored in ``out``."""
    from .ling_fused import _SIGMOID

    body = _SIGMOID.replace("metal::precise::exp", spelling).replace("T ", f"{dtype_name} ")
    return f"{{ const {dtype_name} x = {x}; {dtype_name} y; {body.strip()} {out} = y; }}"


def scalar_in(dtype, value):
    """The bits MLX gives a Python float scalar multiplied into ``dtype``."""
    one = mx.ones((1,), dtype=dtype, stream=mx.cpu)
    return float((one * value).astype(mx.float32, stream=mx.cpu).item())


_SOURCE = r"""
const uint h = threadgroup_position_in_grid.x;
const uint t = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
constexpr uint NSG = NTHREADS / 32;
constexpr int n_per_t = DIM / 32;

threadgroup T qkv[3 * DIM];
threadgroup float g_tg[DIM];
threadgroup float beta_tg[1];
threadgroup T y_tg[DIM];

// ---- three ShortConv1d windows: depthwise conv (AOT), nn.silu (compiled) --
if (t < 3 * DIM) {
    const uint part = t / DIM;
    const uint c = h * DIM + (t % DIM);
    const device T* raw = part == 0 ? q_raw : (part == 1 ? k_raw : v_raw);
    const device T* st = part == 0 ? q_conv : (part == 1 ? k_conv : v_conv);
    const device T* w = (part == 0 ? q_w : (part == 1 ? k_w : v_w)) + size_t(c) * TAPS;
    device T* so = part == 0 ? q_conv_out : (part == 1 ? k_conv_out : v_conv_out);
    float acc = 0.0f;
    {
        #pragma clang fp contract(off)
        const float c0 = static_cast<float>(st[c]);
        const float c1 = static_cast<float>(st[INNER + c]);
        const float c2 = static_cast<float>(st[2 * INNER + c]);
        const float m = static_cast<float>(raw[c]);
        const float w0 = static_cast<float>(w[0]);
        const float w1 = static_cast<float>(w[1]);
        const float w2 = static_cast<float>(w[2]);
        const float w3 = static_cast<float>(w[3]);
        CONV_TAP0
        CONV_TAP1
        CONV_TAP2
        CONV_TAP3
    }
    so[c] = st[INNER + c];
    so[INNER + c] = st[2 * INNER + c];
    so[2 * INNER + c] = raw[c];
    const T xb = static_cast<T>(acc);
    qkv[t] = xb * mlx_sigmoid_fast(xb);
}
// ---- decay (compute_lower_bound_g, compiled fp32) and beta (eager fp32) ---
if (t < DIM) {
    const uint c = h * DIM + t;
    const float ea = metal::precise::exp(static_cast<float>(A_log[h]));
    const float shifted = static_cast<float>(a[c]) + static_cast<float>(dt_bias[c]);
    const float scaled = ea * shifted;
    const float sa = mlx_sigmoid_fast<float>(scaled);
    const float bounded = LOWER_BOUND * sa;
    g_tg[t] = metal::precise::exp(bounded);
    if (t == 0) {
        #pragma clang fp contract(off)
        float sb;
        SIGMOID_BETA
        beta_tg[0] = sb;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- q, k: rms_single_row (no weight), then the eager scalar multiply -----
if (sg < 2) {
    #pragma clang fp contract(off)
    threadgroup T* x = qkv + sg * DIM;
    float tx[4];
    float acc = 0.0f;
    for (int i = 0; i < 4; i++) {
        tx[i] = static_cast<float>(x[4 * lane + i]);
    }
    QK_SQ0
    QK_SQ1
    QK_SQ2
    QK_SQ3
    acc = simd_sum(acc);
    acc = simd_sum(lane == 0 ? acc : 0.0f);
    const float inv = metal::precise::rsqrt(acc / float(DIM) + QK_EPS);
    const T one = static_cast<T>(1.0f);
    const T scale = static_cast<T>(sg == 0 ? SCALE_Q : SCALE_K);
    for (int i = 0; i < 4; i++) {
        const T normed = one * static_cast<T>(tx[i] * inv);
        x[4 * lane + i] = normed * scale;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- gated delta rule: gated_delta_step_vec, one simdgroup per value row --
{
    const threadgroup T* q_ = qkv;
    const threadgroup T* k_ = qkv + DIM;
    const threadgroup T* v_ = qkv + 2 * DIM;
    const threadgroup float* g_ = g_tg;
    const float beta_h = beta_tg[0];
    const uint dk_idx = lane;
    for (uint dv_idx = sg; dv_idx < DIM; dv_idx += NSG) {
        auto i_state = state_in + (size_t(h) * DIM + dv_idx) * DIM;
        auto o_state = state_out + (size_t(h) * DIM + dv_idx) * DIM;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }
        float kv_mem = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] * g_[s_idx];
          kv_mem += state[i] * k_[s_idx];
        }
        kv_mem = simd_sum(kv_mem);

        auto delta = (v_[dv_idx] - kv_mem) * beta_h;

        float out = 0.0f;
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = state[i] + k_[s_idx] * delta;
          out += state[i] * q_[s_idx];
        }
        out = simd_sum(out);
        if (thread_index_in_simdgroup == 0) {
          y_tg[dv_idx] = static_cast<T>(out);
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<float>(state[i]);
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- o_norm: rms_single_row (axis 128) * eager sigmoid(gate) --------------
if (sg == 0) {
    #pragma clang fp contract(off)
    float tx[4];
    float acc = 0.0f;
    for (int i = 0; i < 4; i++) {
        tx[i] = static_cast<float>(y_tg[4 * lane + i]);
    }
    RMS_SQ0
    RMS_SQ1
    RMS_SQ2
    RMS_SQ3
    acc = simd_sum(acc);
    acc = simd_sum(lane == 0 ? acc : 0.0f);
    const float inv = metal::precise::rsqrt(acc / float(DIM) + RMS_EPS);
    for (int i = 0; i < 4; i++) {
        const uint e = 4 * lane + i;
        const T normed = norm[e] * static_cast<T>(tx[i] * inv);
        T sgt;
        SIGMOID_GATE
        o_gated[h * DIM + e] = normed * sgt;
    }
}
"""

_PLACEHOLDERS = ("CONV_TAP", "QK_SQ", "RMS_SQ", "SIGMOID_BETA", "SIGMOID_GATE", "LOWER_BOUND",
                 "QK_EPS", "RMS_EPS", "SCALE_Q", "SCALE_K")


def kernel_source(*, rms_eps, lower_bound, scale_q, scale_k, gate_exp, beta_exp,
                  conv_fma=CONV_FMA, rms_fma=RMS_FMA):
    """The Metal body for one set of constants, sigmoid spellings and pins."""
    source = _SOURCE
    rows = ("c0", "c1", "c2", "m")
    for tap in range(TAPS):
        source = source.replace(f"CONV_TAP{tap}", _mac("acc", rows[tap], f"w{tap}", conv_fma))
        source = source.replace(f"QK_SQ{tap}", _mac("acc", f"tx[{tap}]", f"tx[{tap}]", rms_fma))
        source = source.replace(f"RMS_SQ{tap}", _mac("acc", f"tx[{tap}]", f"tx[{tap}]", rms_fma))
    return (
        source.replace("SIGMOID_BETA", _eager_sigmoid("float", "static_cast<float>(b[h])", "sb",
                                                      beta_exp))
        .replace("SIGMOID_GATE", _eager_sigmoid("T", "gate[h * DIM + e]", "sgt", gate_exp))
        .replace("LOWER_BOUND", _hexf(lower_bound))
        .replace("QK_EPS", _hexf(1e-6))
        .replace("RMS_EPS", _hexf(rms_eps))
        .replace("SCALE_Q", _hexf(scale_q))
        .replace("SCALE_K", _hexf(scale_k))
    )


def served_spellings():
    """``(silu, eager bf16 sigmoid, eager fp32 sigmoid)`` exp spellings the
    served ops use, or None when any probe found no matching form."""
    from .ling_fused import _GATE
    from .qwen4_fused_gdn import served_silu_exp

    found = (served_silu_exp(), _GATE.spelling(mx.bfloat16), _GATE.spelling(mx.float32))
    return None if any(value is None for value in found) else found


def _suffix(spelling):
    return "p" if "precise" in spelling else "f"


@lru_cache(maxsize=8)
def _kernel(rms_eps, lower_bound, spellings, conv_fma=CONV_FMA, rms_fma=RMS_FMA):
    from .qwen4_fused_gdn import _served_silu_header

    silu_exp, gate_exp, beta_exp = spellings
    scale = float(DIM) ** -0.5
    source = kernel_source(
        rms_eps=rms_eps, lower_bound=lower_bound,
        scale_q=scalar_in(mx.bfloat16, scale**2), scale_k=scalar_in(mx.bfloat16, scale),
        gate_exp=gate_exp, beta_exp=beta_exp, conv_fma=conv_fma, rms_fma=rms_fma,
    )
    tag = (f"{np.float32(rms_eps).view(np.uint32):08x}_{np.float32(lower_bound).view(np.uint32):08x}"
           f"_{''.join(_suffix(s) for s in spellings)}_{int(conv_fma)}{int(rms_fma)}")
    return mx.fast.metal_kernel(
        name="mlx2_ling_kda_fused_decode_v1_" + tag,
        input_names=["q_raw", "k_raw", "v_raw", "q_conv", "k_conv", "v_conv", "q_w", "k_w",
                     "v_w", "a", "b", "A_log", "dt_bias", "state_in", "gate", "norm"],
        output_names=["o_gated", "q_conv_out", "k_conv_out", "v_conv_out", "state_out"],
        header=_served_silu_header(silu_exp),
        source=source,
        ensure_row_contiguous=True,
    )


def ling_kda_fused_refusal(raw, convs, weights, a, b, A_log, dt_bias, state, gate, norm):
    """None when the fused body reproduces the reference chain for these
    operands, else the reason it declines.

    B1, one token, Ling's KDA geometry (32 heads x 128, 4 conv taps), bf16
    activations, conv states and weights, fp32 recurrent state, on the Metal
    GPU with every served sigmoid spelling probed.
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return "not on the Metal GPU"
    bf16 = (*raw, *convs, *weights, a, b, gate, norm)
    if any(x is None or x.dtype != mx.bfloat16 for x in bf16):
        return "activation, conv or norm dtype is not bfloat16"
    if A_log.dtype not in (mx.float32, mx.bfloat16) or dt_bias.dtype not in (mx.float32, mx.bfloat16):
        return "A_log or dt_bias dtype"
    if state is None or state.dtype != mx.float32 or state.shape != (1, HEADS, DIM, DIM):
        return "recurrent state is not fp32 [1, 32, 128, 128]"
    if any(x.shape != (1, 1, INNER) for x in raw):
        return "not a B1 one-token projection"
    if any(x.shape != (1, TAPS - 1, INNER) for x in convs):
        return "conv state geometry"
    if any(x.shape != (INNER, TAPS, 1) for x in weights):
        return "conv weight geometry"
    if a.size != INNER or a.shape[:2] != (1, 1) or gate.size != INNER or gate.shape[:2] != (1, 1):
        return "decay or gate geometry"
    if b.size != HEADS or b.shape[:2] != (1, 1):
        return "beta geometry"
    if A_log.size != HEADS or dt_bias.size != INNER or norm.shape != (DIM,):
        return "A_log, dt_bias or norm geometry"
    if served_spellings() is None:
        return "a served sigmoid matches no kernel form"
    return None


def ling_kda_fused_decode(raw, convs, weights, a, b, A_log, dt_bias, state, gate, norm, *,
                          rms_eps, lower_bound, _contraction=(CONV_FMA, RMS_FMA)):
    """``(normed gated output [1, 1, 4096] bf16, (q, k, v) conv states
    [1, 3, 4096] bf16, recurrent state [1, 32, 128, 128] fp32)``.

    Equal bit for bit to ``KimiDeltaAttention.__call__``'s chain at B = L = 1
    up to ``o_proj`` (see the module docstring), in one launch of 32
    threadgroups. Callers check ``ling_kda_fused_refusal`` first.
    """
    spellings = served_spellings()
    outputs = _kernel(float(rms_eps), float(lower_bound), spellings, *_contraction)(
        inputs=[
            *(x.reshape(INNER) for x in raw),
            *(x.reshape(TAPS - 1, INNER) for x in convs),
            *(x.reshape(INNER * TAPS) for x in weights),
            a.reshape(INNER),
            b.reshape(HEADS),
            A_log.reshape(HEADS),
            dt_bias.reshape(INNER),
            state,
            gate.reshape(INNER),
            norm,
        ],
        template=[
            ("T", mx.bfloat16),
            ("DIM", DIM),
            ("INNER", INNER),
            ("TAPS", TAPS),
            ("NTHREADS", THREADS),
        ],
        grid=(HEADS * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(1, 1, INNER)] + [(1, TAPS - 1, INNER)] * 3 + [state.shape],
        output_dtypes=[mx.bfloat16] * 4 + [mx.float32],
    )
    out, q_conv, k_conv, v_conv, new_state = outputs
    return out, (q_conv, k_conv, v_conv), new_state
