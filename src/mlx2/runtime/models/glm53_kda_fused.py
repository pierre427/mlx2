"""Original single-launch GLM-5.3 KDA decode body (``glm53_kda_fused=decode_v1``).

One Metal launch per KDA layer per B1 decode token covers everything between
the input projections and ``attn_output`` in ``Glm53GGUFModel._kda``:

    conv window = concat(conv state, qkv)          new conv state = rows 1..3
    depthwise conv1d (4 taps) -> SiLU -> split q, k, v
    q, k L2 norms (epsilon floors the norm), q * dim**-0.5
    decay = exp(-5 * sigmoid((a + dt_bias) * ssm_a)), beta = sigmoid(beta)
    gated delta rule over the 64 x 128 x 128 fp32 state (gated_delta_step_vec)
    rms_norm(y, ssm_norm) * sigmoid(gate)

One threadgroup owns one head: every step above is head-local (the conv is
depthwise, the norms reduce over one head's 128 channels), so no cross-head
reduction boundary exists and a single launch suffices.

Exactness: the outputs and both states equal the reference op chain bit for
bit, one IEEE operation per reference operation in the reference order. The
orders reproduced (MLX v0.32.4, the venv's build; mlx2 gated_delta.py):

* conv: MLX ``depthwise_conv_1d`` (``conv1d`` fast path for groups == C):
  ``acc = 0; acc += in[t] * w[t]`` for taps t = 0..3, conv state rows first.
  That kernel is built ahead of time with ``-fno-fast-math``, which contracts
  ``acc += x * w`` into one fused multiply-add (``CONV_FMA``; the Metal tests
  pin it).
* SiLU: ``nn.silu`` = ``x * sigmoid(x)`` with MLX's ``Sigmoid`` functor.
* L2 norm: ``q * q`` is its own kernel (each square rounded), then
  ``row_reduce_simple`` (row 128, one simdgroup): lane l folds
  ``total = sq + total`` over elements 4l..4l+3 from 0, then ``simd_sum``.
  ``maximum`` is MLX's ``Maximum`` (NaN propagates), sqrt and division IEEE.
* delta rule: the ``gated_delta_step_vec`` inner loop verbatim (lane l owns
  key elements 4l..4l+3 of a value row, two ``simd_sum`` reductions).
* rms: MLX ``rms_single_row`` for an axis of 128 (32 threads, 4 reads):
  ``acc += x * x`` (``RMS_FMA``, contracted like the conv), ``simd_sum``,
  the cross-simdgroup ``simd_sum`` over one value and zeros,
  ``precise::rsqrt(acc / 128 + eps)``, ``w * (x * inv)``, then ``* gate``.

Contraction is pinned explicitly: steps that are separate MLX kernels in the
reference are evaluated under ``fp contract(off)``; the AOT kernels' fused
multiply-adds are written as ``metal::fma``. The delta rule keeps the JIT
source's own expressions, compiled under the same (default, safe) options as
the reference kernel.

Callers keep the reference chain for prefill, B > 1, CPU, non-fp32 inputs or
other shapes (``kda_fused_admits``); on Metal the GPU tests in
tests/test_glm53_kda_fused.py compare the launch with the chain bitwise,
chained over several decode steps.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

HEADS, DIM, TAPS = 64, 128, 4
INNER = HEADS * DIM
CHANNELS = 3 * INNER
THREADS = 1024


def _hexf(value):
    """A float32 literal that names exactly ``np.float32(value)``."""
    return float(np.float32(value)).hex() + "f"


def _sigmoid(x, out):
    """MLX's Sigmoid functor (mx.sigmoid's bytes), as in glm53_glue_metal."""
    return (
        f"{{ const float y_ = 1 / (1 + metal::precise::exp(metal::abs({x})));"
        f" {out} = ({x} < 0) ? y_ : 1 - y_; }}"
    )


def _mac(acc, x, w, fused):
    """``acc += x * w`` as one AOT kernel statement compiles it."""
    if fused:
        return f"{acc} = metal::fma({x}, {w}, {acc});"
    return f"{{ const float p_ = {x} * {w}; {acc} = {acc} + p_; }}"


_SOURCE = r"""
const uint h = threadgroup_position_in_grid.x;
const uint t = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
constexpr uint NSG = NTHREADS / 32;
constexpr int n_per_t = DIM / 32;

threadgroup float qkv[3 * DIM];
threadgroup float g_tg[DIM];
threadgroup float gate_tg[DIM];
threadgroup float y_tg[DIM];
threadgroup float beta_tg[1];

// ---- conv window, depthwise conv (4 taps), SiLU; new conv state ----------
if (t < 3 * DIM) {
    #pragma clang fp contract(off)
    const uint part = t / DIM;
    const uint c = part * INNER + h * DIM + (t % DIM);
    const float c0 = conv[c];
    const float c1 = conv[CHANNELS + c];
    const float c2 = conv[2 * CHANNELS + c];
    const float m = mixed[c];
    conv_out[c] = c1;
    conv_out[CHANNELS + c] = c2;
    conv_out[2 * CHANNELS + c] = m;
    const device float* w = filters + size_t(c) * TAPS;
    float acc = 0.0f;
    CONV_TAP0
    CONV_TAP1
    CONV_TAP2
    CONV_TAP3
    float s;
    SIGMOID_ACC
    qkv[t] = acc * s;
}
// ---- gate epilogue (kda_small's law) -------------------------------------
if (t < DIM) {
    #pragma clang fp contract(off)
    const uint c = h * DIM + t;
    const float shifted = a[c] + dt_bias[c];
    const float scaled = shifted * ssm_a[h];
    float sa;
    SIGMOID_SA
    const float negated = -5.0f * sa;
    g_tg[t] = metal::precise::exp(negated);
    float sgt;
    SIGMOID_SG
    gate_tg[t] = sgt;
    if (t == 0) {
        float sb;
        SIGMOID_SB
        beta_tg[0] = sb;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- q, k L2 norms: square kernel, row_reduce_simple, max/sqrt/divide -----
if (sg < 2) {
    #pragma clang fp contract(off)
    threadgroup float* x = qkv + sg * DIM;
    float v4[4];
    for (int i = 0; i < 4; i++) {
        v4[i] = x[4 * lane + i];
    }
    float sq[4];
    for (int i = 0; i < 4; i++) {
        sq[i] = v4[i] * v4[i];
    }
    float total = 0.0f;
    for (int i = 0; i < 4; i++) {
        total = sq[i] + total;
    }
    total = simd_sum(total);
    const float nrm = metal::precise::sqrt(total);
    const float den = metal::isnan(nrm) ? nrm : (nrm > L2_EPS ? nrm : L2_EPS);
    for (int i = 0; i < 4; i++) {
        float r = v4[i] / den;
        if (sg == 0) {
            r = r * Q_SCALE;
        }
        x[4 * lane + i] = r;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- gated delta rule: gated_delta_step_vec, one simdgroup per value row --
{
    const threadgroup float* q_ = qkv;
    const threadgroup float* k_ = qkv + DIM;
    const threadgroup float* v_ = qkv + 2 * DIM;
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
          y_tg[dv_idx] = out;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = state[i];
        }
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- rms_single_row (axis 128), then * sigmoid(gate) -----------------------
if (sg == 0) {
    #pragma clang fp contract(off)
    float tx[4];
    float acc = 0.0f;
    for (int i = 0; i < 4; i++) {
        tx[i] = y_tg[4 * lane + i];
    }
    RMS_SQ0
    RMS_SQ1
    RMS_SQ2
    RMS_SQ3
    acc = simd_sum(acc);
    acc = simd_sum(lane == 0 ? acc : 0.0f);
    const float mean = acc / float(DIM);
    const float inv = metal::precise::rsqrt(mean + RMS_EPS);
    for (int i = 0; i < 4; i++) {
        const uint e = 4 * lane + i;
        const float normed = norm[e] * (tx[i] * inv);
        y[h * DIM + e] = normed * gate_tg[e];
    }
}
"""

# Contraction of the ahead-of-time MLX kernels (-fno-fast-math): the Metal
# tests pin these against the venv's MLX build.
CONV_FMA = True
RMS_FMA = True


def kernel_source(rms_eps, conv_fma=CONV_FMA, rms_fma=RMS_FMA):
    """The Metal body for one rms epsilon and the pinned contraction choices."""
    source = _SOURCE
    for tap in range(TAPS):
        rows = ("c0", "c1", "c2", "m")
        source = source.replace(f"CONV_TAP{tap}", _mac("acc", rows[tap], f"w[{tap}]", conv_fma))
        source = source.replace(f"RMS_SQ{tap}", _mac("acc", f"tx[{tap}]", f"tx[{tap}]", rms_fma))
    return (
        source.replace("SIGMOID_ACC", _sigmoid("acc", "s"))
        .replace("SIGMOID_SA", _sigmoid("scaled", "sa"))
        .replace("SIGMOID_SG", _sigmoid("gate[c]", "sgt"))
        .replace("SIGMOID_SB", _sigmoid("beta[h]", "sb"))
        .replace("L2_EPS", _hexf(1e-6))
        .replace("Q_SCALE", _hexf(DIM**-0.5))
        .replace("RMS_EPS", _hexf(rms_eps))
    )


@lru_cache(maxsize=8)
def _kernel(rms_eps, conv_fma=CONV_FMA, rms_fma=RMS_FMA):
    source = kernel_source(rms_eps, conv_fma, rms_fma)
    tag = f"{np.float32(rms_eps).view(np.uint32):08x}_{int(conv_fma)}{int(rms_fma)}"
    return mx.fast.metal_kernel(
        name="mlx2_glm53_kda_fused_decode_v1_" + tag,
        input_names=["mixed", "conv", "filters", "a", "dt_bias", "ssm_a", "beta",
                     "gate", "state_in", "norm"],
        output_names=["y", "conv_out", "state_out"],
        source=source,
        ensure_row_contiguous=True,
    )


def kda_fused_admits(mixed, conv, filters, a, dt_bias, ssm_a, beta, gate, state, norm):
    """True when the fused decode body reproduces the reference chain for these inputs.

    B1, one token, the GLM-5.3 KDA geometry, fp32 everything, on the Metal GPU.
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    arrays = (mixed, conv, filters, a, dt_bias, ssm_a, beta, gate, state, norm)
    if any(x is None or x.dtype != mx.float32 for x in arrays):
        return False
    return (
        mixed.shape == (1, 1, CHANNELS)
        and conv.shape == (1, TAPS - 1, CHANNELS)
        and filters.size == CHANNELS * TAPS
        and a.size == INNER and a.shape[:2] == (1, 1)
        and gate.size == INNER and gate.shape[:2] == (1, 1)
        and beta.size == HEADS and beta.shape[:2] == (1, 1)
        and dt_bias.size == INNER
        and ssm_a.size == HEADS
        and state.shape == (1, HEADS, DIM, DIM)
        and norm.shape == (DIM,)
    )


def fused_kda_decode(mixed, conv, filters, a, dt_bias, ssm_a, beta, gate, state, norm, *,
                     rms_eps, _contraction=(CONV_FMA, RMS_FMA)):
    """(pre-output y [1, 1, 8192], new conv [1, 3, 24576], new state [1, 64, 128, 128]).

    Equal bit for bit to ``Glm53GGUFModel._kda``'s reference chain at L = 1
    (see the module docstring), in one launch of 64 threadgroups.
    """
    y, conv_out, state_out = _kernel(float(rms_eps), *_contraction)(
        inputs=[
            mixed.reshape(CHANNELS),
            conv.reshape(TAPS - 1, CHANNELS),
            filters.reshape(CHANNELS * TAPS),
            a.reshape(INNER),
            dt_bias.reshape(INNER),
            ssm_a.reshape(HEADS),
            beta.reshape(HEADS),
            gate.reshape(INNER),
            state,
            norm,
        ],
        template=[
            ("DIM", DIM),
            ("INNER", INNER),
            ("CHANNELS", CHANNELS),
            ("TAPS", TAPS),
            ("NTHREADS", THREADS),
        ],
        grid=(HEADS * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(1, 1, INNER), (1, TAPS - 1, CHANNELS), state.shape],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return y, conv_out, state_out


# ---- decode_v2: the low-rank f_b / g_b prologue inside the same launch -------
#
# ``a = ssm_f_b(f_a)`` and ``gate = ssm_g_b(g_a)`` are Q4_0 [128 -> 8192]
# projections that the reference runs as a second ``kblock_multi`` launch the
# body then waits on. Head h needs only rows 128h .. 128h+127 of each, i.e.
# 256 rows against the 128-wide f_a / g_a outputs, so decode_v2 computes them
# as the body's first phase (8 rounds of one row per simdgroup) and drops that
# launch.
#
# Law: Q4_0 always runs kblock_multi's EXACT branch (``exact_law``, whatever
# the row policy): all 32 lanes of one simdgroup per output row, lane l
# accumulating k = l, l+32, l+64, l+96 from 0.0f in that order as
# ``sums += xv * gguf_k_value(row, k, 2)`` (the shared device-pointer decoder
# of k_metal.HEADER), then ``simd_sum``. The prologue is that statement
# verbatim, compiled under the same (default) JIT options; ``LOWRANK_MAC``
# pins the contraction and the Metal tests show the alternatives differ.
# The rest of the body is decode_v1's source with the gate epilogue reading
# the prologue's threadgroup rows instead of device ``a`` / ``gate``; the
# epilogue moves behind the first barrier (beside the L2 norms) so the
# prologue and the conv window share one phase and no barrier is added.

LOWRANK_K = 128
Q4_0, Q4_0_ROW_BYTES = 2, 72  # 4 blocks of 18 bytes per 128-wide row

# "verbatim" (kblock_multi's own expression), "fma" or "off".
LOWRANK_MAC = "verbatim"

_LOWRANK_PROLOGUE = r"""threadgroup float lr_tg[2 * DIM];

// ---- low-rank prologue: this head's ssm_f_b / ssm_g_b rows (Q4_0 exact law) --
for (uint r = sg; r < 2 * DIM; r += NSG) {
    const uint part = r / DIM;
    device const uchar* wrow = (part == 0 ? f_b : g_b) + size_t(h * DIM + r % DIM) * LR_ROW_BYTES;
    const device float* xin = part == 0 ? f_a : g_a;
    float sum = 0.0f;
    for (uint k = lane; k < LR_K; k += 32) {
        const float xv = xin[k];
        LOWRANK_MAC_STMT
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        lr_tg[r] = sum;
        PROBE_STORE
    }
}
"""

_LOWRANK_MACS = {
    "verbatim": "sum+=xv*gguf_k_value(wrow,k,2);",
    "fma": "sum = metal::fma(xv, gguf_k_value(wrow, k, 2), sum);",
    "off": "{\n            #pragma clang fp contract(off)\n"
           "            const float p_ = xv * gguf_k_value(wrow, k, 2);\n"
           "            sum = sum + p_;\n        }",
}


def _split_v1_source():
    """decode_v1's source cut at the gate epilogue, asserted against drift."""
    begin = _SOURCE.index("// ---- gate epilogue")
    barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    end = _SOURCE.index(barrier, begin)
    decls_end = _SOURCE.index("threadgroup float beta_tg[1];\n") + len("threadgroup float beta_tg[1];\n")
    head, conv = _SOURCE[:decls_end], _SOURCE[decls_end:begin]
    epilogue, rest = _SOURCE[begin:end], _SOURCE[end + len(barrier):]
    assert "a[c]" in epilogue and "SIGMOID_SG" in epilogue and "// ---- conv window" in conv
    assert rest.lstrip().startswith("// ---- q, k L2 norms")
    return head, conv, epilogue, barrier, rest


def lowrank_kernel_source(rms_eps, conv_fma=CONV_FMA, rms_fma=RMS_FMA, mac=LOWRANK_MAC,
                          probe=False):
    """decode_v2's Metal body: the prologue, then decode_v1 with threadgroup gates.

    ``probe`` (tests only) also stores each prologue row to a fourth output
    ``lr_out`` laid out as kblock_multi's [f_b rows | g_b rows].
    """
    head, conv, epilogue, barrier, rest = _split_v1_source()
    epilogue = epilogue.replace("a[c] + dt_bias[c]", "lr_tg[t] + dt_bias[c]")
    store = "lr_out[(r / DIM) * INNER + h * DIM + r % DIM] = sum;" if probe else ""
    prologue = _LOWRANK_PROLOGUE.replace("LOWRANK_MAC_STMT", _LOWRANK_MACS[mac])
    prologue = prologue.replace("        PROBE_STORE\n", f"        {store}\n" if store else "")
    source = head + prologue + conv + barrier + epilogue + rest
    for tap in range(TAPS):
        rows = ("c0", "c1", "c2", "m")
        source = source.replace(f"CONV_TAP{tap}", _mac("acc", rows[tap], f"w[{tap}]", conv_fma))
        source = source.replace(f"RMS_SQ{tap}", _mac("acc", f"tx[{tap}]", f"tx[{tap}]", rms_fma))
    return (
        source.replace("SIGMOID_ACC", _sigmoid("acc", "s"))
        .replace("SIGMOID_SA", _sigmoid("scaled", "sa"))
        .replace("SIGMOID_SG", _sigmoid("lr_tg[DIM + t]", "sgt"))
        .replace("SIGMOID_SB", _sigmoid("beta[h]", "sb"))
        .replace("L2_EPS", _hexf(1e-6))
        .replace("Q_SCALE", _hexf(DIM**-0.5))
        .replace("RMS_EPS", _hexf(rms_eps))
    )


@lru_cache(maxsize=8)
def _lowrank_kernel(rms_eps, conv_fma=CONV_FMA, rms_fma=RMS_FMA, mac=LOWRANK_MAC, probe=False):
    from ..gguf.k_metal import HEADER

    source = lowrank_kernel_source(rms_eps, conv_fma, rms_fma, mac, probe)
    tag = (f"{np.float32(rms_eps).view(np.uint32):08x}_{int(conv_fma)}{int(rms_fma)}_{mac}"
           + ("_probe" if probe else ""))
    return mx.fast.metal_kernel(
        name="mlx2_glm53_kda_fused_decode_v2_" + tag,
        input_names=["mixed", "conv", "filters", "f_a", "g_a", "f_b", "g_b", "dt_bias", "ssm_a",
                     "beta", "state_in", "norm"],
        output_names=["y", "conv_out", "state_out"] + (["lr_out"] if probe else []),
        header=HEADER,
        source=source,
        ensure_row_contiguous=True,
    )


def _lowrank_matrix_admits(w):
    """A Q4_0 [8192, 128] PackedMatrix on its flat uint8 device buffer."""
    data = getattr(w, "data", None)
    return (
        getattr(w, "ggml_type", None) == Q4_0
        and tuple(getattr(w, "shape", ())) == (INNER, LOWRANK_K)
        and getattr(w, "row_bytes", None) == Q4_0_ROW_BYTES
        and data is not None and data.dtype == mx.uint8 and data.ndim == 1
        and data.size == INNER * Q4_0_ROW_BYTES
    )


def kda_fused_lowrank_admits(mixed, conv, filters, f_a_out, g_a_out, f_b, g_b, dt_bias,
                             stored_a, beta, state, norm):
    """True when decode_v2 reproduces ``kblock_multi([f_a, g_a], [f_b, g_b])``
    followed by decode_v1 bit for bit: decode_v1's admission, fp32 128-wide
    f_a / g_a rows of one token, and both low-rank matrices Q4_0 [8192, 128].
    """
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    if not (_lowrank_matrix_admits(f_b) and _lowrank_matrix_admits(g_b)):
        return False
    lowrank = (f_a_out, g_a_out)
    if any(x is None or x.dtype != mx.float32 or x.size != LOWRANK_K or x.shape[:2] != (1, 1)
           for x in lowrank):
        return False
    arrays = (mixed, conv, filters, dt_bias, stored_a, beta, state, norm)
    if any(x is None or x.dtype != mx.float32 for x in arrays):
        return False
    return (
        mixed.shape == (1, 1, CHANNELS)
        and conv.shape == (1, TAPS - 1, CHANNELS)
        and filters.size == CHANNELS * TAPS
        and beta.size == HEADS and beta.shape[:2] == (1, 1)
        and dt_bias.size == INNER
        and stored_a.size == HEADS
        and state.shape == (1, HEADS, DIM, DIM)
        and norm.shape == (DIM,)
    )


def fused_kda_decode_lowrank(mixed, conv, filters, f_a_out, g_a_out, f_b, g_b, dt_bias,
                             stored_a, beta, state, norm, *, rms_eps,
                             _contraction=(CONV_FMA, RMS_FMA), _mac=LOWRANK_MAC, _probe=False):
    """decode_v2: (pre-output y [1, 1, 8192], new conv, new state) from the
    f_a / g_a outputs and the Q4_0 ``ssm_f_b`` / ``ssm_g_b`` PackedMatrix pair,
    in one launch of 64 threadgroups; equal bit for bit to the second
    ``kblock_multi`` launch followed by ``fused_kda_decode``. ``_probe``
    (tests only) appends the raw prologue rows [1, 16384] to the result.
    """
    outputs = _lowrank_kernel(float(rms_eps), *_contraction, _mac, bool(_probe))(
        inputs=[
            mixed.reshape(CHANNELS),
            conv.reshape(TAPS - 1, CHANNELS),
            filters.reshape(CHANNELS * TAPS),
            f_a_out.reshape(LOWRANK_K),
            g_a_out.reshape(LOWRANK_K),
            f_b.data,
            g_b.data,
            dt_bias.reshape(INNER),
            stored_a.reshape(HEADS),
            beta.reshape(HEADS),
            state,
            norm,
        ],
        template=[
            ("DIM", DIM),
            ("INNER", INNER),
            ("CHANNELS", CHANNELS),
            ("TAPS", TAPS),
            ("NTHREADS", THREADS),
            ("LR_K", LOWRANK_K),
            ("LR_ROW_BYTES", Q4_0_ROW_BYTES),
        ],
        grid=(HEADS * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(1, 1, INNER), (1, TAPS - 1, CHANNELS), state.shape]
        + ([(1, 2 * INNER)] if _probe else []),
        output_dtypes=[mx.float32] * (4 if _probe else 3),
    )
    return tuple(outputs)
