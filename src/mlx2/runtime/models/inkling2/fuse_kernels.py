# SPDX-License-Identifier: Apache-2.0
"""Near-exact Metal kernels for ``inkling2_fuse_sconv_kernel`` and
``inkling2_fuse_residual_kernel`` (original mlx2 code, 2026-11-22).

Both are GPU only (``fuse_sconv_kernel`` default on since 2026-11-22,
``fuse_residual_kernel`` default off) and decline (return ``None``, counted) on
anything outside their layout, so the caller runs the exact path.  Their
arithmetic follows the reference op for op in FP32, but the summation order
of the reductions is the kernels' own, so results are near-exact, not
bit-identical; ``tests/test_inkling2_fuse_metal.py`` measures the gap.

- ``sconv``: a static K=4 short convolution of up to 8 rows (decode and
  verify windows) with no padding mask: the FP32 depthwise conv over
  ``[state; x]``, ``(conv + x) / sqrt(K + 1)``, the BF16 rounding, the
  decoder's residual tail (``fuse.SCONV_TAILS``) and the next state, in one
  launch.  The reference is conv1d (an unfold plus a grouped GEMM on the
  GPU), two concatenations, casts and the elementwise tail.
- ``residual``: a one-row attention-residual mix, ``softmax(rms(S) q /
  sqrt(D)) rms(S)`` over the blocks plus the partial, in one launch instead
  of a stack, casts, an RMS norm, two GEMVs, the scale and a four-op
  softmax.  The loop messages (the same law over the previous pass's bank)
  use it too.
"""

from __future__ import annotations

import math

import mlx.core as mx

from . import fuse as _fuse

_SCONV_ROWS = 8
_TAIL_IDS = {"plain": 0, "attn_boundary": 1, "attn_residual": 2, "mlp_residual": 3}
_SCONV_SOURCE = r"""
    uint c = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    if (c >= C) return;
    const float inv = inv_arr[0];
    float w[4];
    for (int j = 0; j < 4; ++j) w[j] = wt[c * 4 + j];
    float ctx[3 + L];
    for (int j = 0; j < 3; ++j) ctx[j] = HAS_STATE ? state[(b * 3 + j) * C + c] : 0.0f;
    for (int i = 0; i < L; ++i) ctx[3 + i] = float(x[(b * L + i) * C + c]);
    for (int i = 0; i < L; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < 4; ++j) acc += w[j] * ctx[i + j];
        float xf = ctx[3 + i];
        float y = float(T((acc + xf) / inv));
        size_t o = (b * L + i) * C + c;
        if (TAIL == 0) {
            out[o] = OutT(y);
        } else if (TAIL == 1) {
            out[o] = OutT(y);
        } else if (TAIL == 2) {
            out[o] = OutT(y + float(e0[o]));
        } else {
            out[o] = OutT(float(e0[o]) + y);
        }
    }
    for (int j = 0; j < 3; ++j) nstate[(b * 3 + j) * C + c] = ctx[L + j];
"""
_sconv_kernels = {}


def _decline(module, reason):
    _fuse.count(module, "kernel_declined:" + reason)
    return None


def sconv(conv, x, cache, mask, tail, extra):
    """One-launch static short convolution, or ``None`` to run the exact path."""
    from .language import _fp32, _speculative_record

    if mx.default_device() != mx.gpu:
        return _decline(conv, "not_gpu")
    if mask is not None:
        return _decline(conv, "mask")
    if conv.kernel_size != 4 or cache is None:
        return _decline(conv, "layout")
    b, length, channels = x.shape
    if length > _SCONV_ROWS:
        return _decline(conv, "rows")
    state = cache[conv.conv_idx]
    if state is not None and (state.shape != (b, 3, channels) or state.dtype != mx.float32):
        return _decline(conv, "state")
    if getattr(cache, "speculating", False):
        # The rollback record keeps the FP32 inputs (same values).
        _speculative_record(cache, conv.conv_idx, state, x.astype(mx.float32), mask)
    tail_id = _TAIL_IDS[tail]
    out_dtype = {0: x.dtype, 1: mx.float32, 2: mx.float32}.get(tail_id)
    if tail_id == 3:
        out_dtype = extra[1].dtype
        e0 = extra[0]
    elif tail_id == 2:
        e0 = extra[0]
    else:
        e0 = mx.zeros((1,), mx.float32)
    kernel = _sconv_kernels.get("k")
    if kernel is None:
        kernel = _sconv_kernels["k"] = mx.fast.metal_kernel(
            name="inkling2_fuse_sconv",
            input_names=["x", "state", "wt", "inv_arr", "e0"],
            output_names=["out", "nstate"],
            source=_SCONV_SOURCE,
        )
    weight = _fp32(conv.conv, "weight").reshape(channels, 4)
    out, nstate = kernel(
        inputs=[x, state if state is not None else mx.zeros((1,), mx.float32), weight,
                mx.array([math.sqrt(5)], mx.float32), e0],
        template=[("T", x.dtype), ("OutT", out_dtype), ("C", channels), ("L", length),
                  ("TAIL", tail_id), ("HAS_STATE", state is not None)],
        grid=(channels, b, 1),
        threadgroup=(min(256, channels), 1, 1),
        output_shapes=[(b, length, channels), (b, 3, channels)],
        output_dtypes=[out_dtype, mx.float32],
    )
    cache[conv.conv_idx] = nstate
    _fuse.count(conv, "sconv_kernel")
    return out


_RES_TG = 256
_res_kernels = {}
# Rounding-point experiment (2026-11-22, the lesson from omlx#5538 and
# Rapid-MLX #5918): compile the residual, dynamic-conv and message-mix
# kernels with ``#pragma METAL fp math_mode(safe)`` and ``fp contract(off)``
# so no product is fused into an add the reference rounds separately. A
# module switch for the Metal tests (tests/test_inkling2_fuse_metal.py)
# to measure; not an execution-policy option.
SAFE_MATH = False
_SAFE_HEADER = "#pragma METAL fp math_mode(safe)\n#pragma METAL fp contract(off)\n"


def _header():
    return _SAFE_HEADER if SAFE_MATH else ""


def _residual_source(n):
    loads = "\n".join(
        f"        v[{i}] = float(b{i}[d]);" for i in range(n - 1)
    ) + f"\n        v[{n - 1}] = float(p[d]);"
    return r"""
    constexpr int NS = TG / 32;
    const uint tid = thread_position_in_threadgroup.x;
    const uint qi = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float red[N * NS];
    threadgroup float inv_s[N];
    threadgroup float wts[N];
    float v[N];
    float acc[N];
    for (int i = 0; i < N; ++i) acc[i] = 0.0f;
    for (int d = tid; d < D; d += TG) {
LOADS
        for (int i = 0; i < N; ++i) acc[i] += v[i] * v[i];
    }
    for (int i = 0; i < N; ++i) {
        float s = simd_sum(acc[i]);
        if (lane == 0) red[i * NS + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < N) {
        float t = 0.0f;
        for (int g = 0; g < NS; ++g) t += red[tid * NS + g];
        inv_s[tid] = metal::precise::rsqrt(t / float(D) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N; ++i) acc[i] = 0.0f;
    for (int d = tid; d < D; d += TG) {
LOADS
        const float qd = float(q[qi * D + d]);
        for (int i = 0; i < N; ++i) acc[i] += (v[i] * inv_s[i]) * qd;
    }
    for (int i = 0; i < N; ++i) {
        float s = simd_sum(acc[i]);
        if (lane == 0) red[i * NS + sg] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sc[N];
        float m = -INFINITY;
        for (int i = 0; i < N; ++i) {
            float t = 0.0f;
            for (int g = 0; g < NS; ++g) t += red[i * NS + g];
            sc[i] = t * scale[0];
            m = max(m, sc[i]);
        }
        float l = 0.0f;
        for (int i = 0; i < N; ++i) { sc[i] = metal::precise::exp(sc[i] - m); l += sc[i]; }
        for (int i = 0; i < N; ++i) wts[i] = sc[i] / l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = tid; d < D; d += TG) {
LOADS
        float o = 0.0f;
        for (int i = 0; i < N; ++i) o += wts[i] * (v[i] * inv_s[i]);
        out[qi * D + d] = o;
    }
""".replace("LOADS", loads)


def softmax_mix(module, states, query):
    """``softmax(rms(S) q^T / sqrt(D), over S) rms(S)`` for one row.

    states: arrays of D elements (one row each), the last one any dtype;
    query: [nq, D] or [D].  Returns [1, 1, nq, D] FP32, or ``None``."""
    width = states[-1].shape[-1]
    n = len(states)
    if n > 16 or any(s.size != width for s in states):
        return _decline(module, "layout")
    kernel = _res_kernels.get((n, SAFE_MATH))
    if kernel is None:
        kernel = _res_kernels[(n, SAFE_MATH)] = mx.fast.metal_kernel(
            name=f"inkling2_fuse_residual_{n}{'_safe' if SAFE_MATH else ''}",
            input_names=[f"b{i}" for i in range(n - 1)] + ["p", "q", "scale"],
            output_names=["out"],
            source=_residual_source(n),
            header=_header(),
        )
    q = query.reshape(-1, width)
    nq = q.shape[0]
    out = kernel(
        inputs=[*(s.reshape(width) for s in states[:-1]), states[-1].reshape(width), q,
                mx.array([width ** -0.5], mx.float32)],
        template=[("N", n), ("D", width), ("TG", _RES_TG)],
        grid=(nq * _RES_TG, 1, 1),
        threadgroup=(_RES_TG, 1, 1),
        output_shapes=[(1, 1, nq, width)],
        output_dtypes=[mx.float32],
    )[0]
    return out


def residual(module, blocks, partial):
    """One-launch attention-residual mix, or ``None`` to run the exact path."""
    if mx.default_device() != mx.gpu:
        return _decline(module, "not_gpu")
    if not module.normalize_values:
        return _decline(module, "values_not_normalized")
    if partial.shape[0] * partial.shape[1] != 1:
        return _decline(module, "rows")
    out = softmax_mix(module, [*blocks, partial], module.query)
    if out is not None:
        _fuse.count(module, "residual_kernel")
    return out


# ----- dynamic short conv (global-layer k/v), one launch ------------------
#
# ``inkling2_fuse_dconv_kernel`` (original mlx2 code, 2026-11-22; the idea of
# a one-launch dynamic conv is from omlx#5538 and Rapid-MLX #5918, both
# reimplemented here against mlx2's own reference, ``fuse._dynamic_conv``).
# Per-token taps ``static_t[c, lag] + dyn[b, t, head, lag]``, summed lag 0
# (current token) first from 0, then ``(out + x) / sqrt(2K + 1)`` rounded to
# the activation dtype, and the next state, in one launch instead of the
# context concat, the compiled lag loop and the state slice. ``FMA`` selects
# a fused multiply-add per lag (1) or a separate multiply and add (0): which
# one matches the compiled reference bit for bit is a Metal-test question
# (tests/test_inkling2_fuse_metal.py), so both are kept.
_DCONV_SOURCE = r"""
    uint c = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    if (c >= C) return;
    const float inv = inv_arr[0];
    const uint head = c / (C / H);
    float st[4];
    for (int lag = 0; lag < 4; ++lag) st[lag] = static_t[c * 4 + lag];
    float ctx[3 + L];
    for (int j = 0; j < 3; ++j) ctx[j] = HAS_STATE ? state[(b * 3 + j) * C + c] : 0.0f;
    for (int i = 0; i < L; ++i) ctx[3 + i] = float(x[(b * L + i) * C + c]);
    for (int i = 0; i < L; ++i) {
        const size_t drow = ((size_t)(b * L + i) * H + head) * 4;
        float out = 0.0f;
        for (int lag = 0; lag < 4; ++lag) {
            const float w = st[lag] + dyn[drow + lag];
            const float v = ctx[3 - lag + i];
            if (FMA) {
                out = metal::fma(v, w, out);
            } else {
                const float term = v * w;
                out = out + term;
            }
        }
        const float xf = ctx[3 + i];
        out_y[(b * L + i) * C + c] = T((out + xf) / inv);
    }
    for (int j = 0; j < 3; ++j) nstate[(b * 3 + j) * C + c] = ctx[L + j];
"""
_dconv_kernels = {}
DCONV_FMA = 0


def dconv(conv, x, dynamic, cache, mask):
    """One-launch ``DynamicShortConvolution`` (decode and verify rows), or
    ``None`` to run the exact path. ``dynamic``: [B, L, H, 1, K] FP32."""
    from .language import _fp32, _speculative_record

    if mx.default_device() != mx.gpu:
        return _decline(conv, "not_gpu")
    if mask is not None:
        return _decline(conv, "mask")
    if conv.kernel_size != 4 or cache is None:
        return _decline(conv, "layout")
    b, length, channels = x.shape
    if length > _SCONV_ROWS:
        return _decline(conv, "rows")
    state = cache[conv.conv_idx]
    if state is not None and (state.shape != (b, 3, channels) or state.dtype != mx.float32):
        return _decline(conv, "state")
    if getattr(cache, "speculating", False):
        _speculative_record(cache, conv.conv_idx, state, x.astype(mx.float32), mask)
    kernel = _dconv_kernels.get(SAFE_MATH)
    if kernel is None:
        kernel = _dconv_kernels[SAFE_MATH] = mx.fast.metal_kernel(
            name=f"inkling2_fuse_dconv{'_safe' if SAFE_MATH else ''}",
            input_names=["x", "state", "static_t", "dyn", "inv_arr"],
            output_names=["out_y", "nstate"],
            source=_DCONV_SOURCE,
            header=_header(),
        )
    static_t = _fp32(conv, "static_weight").T  # [C, K]
    out, nstate = kernel(
        inputs=[x, state if state is not None else mx.zeros((1,), mx.float32),
                mx.contiguous(static_t), dynamic.reshape(b, length, conv.num_heads * 4),
                mx.array([math.sqrt(9)], mx.float32)],
        template=[("T", x.dtype), ("C", channels), ("L", length), ("H", conv.num_heads),
                  ("HAS_STATE", state is not None), ("FMA", int(DCONV_FMA))],
        grid=(channels, b, 1),
        threadgroup=(min(256, channels), 1, 1),
        output_shapes=[(b, length, channels), (b, 3, channels)],
        output_dtypes=[x.dtype, mx.float32],
    )
    cache[conv.conv_idx] = nstate
    _fuse.count(conv, "dconv_kernel")
    return out


# ----- loop-message mix (forget/gain GEMVs, tanh, blend), one launch ------
#
# ``inkling2_fuse_message_kernel`` (original mlx2 code, 2026-11-22; the
# mechanism is from Rapid-MLX #5919 ``message_mix_decode``, reimplemented):
# for one-row controller calls, ``f = tanh(h . forget[c])``,
# ``g = tanh(h . gain[c])`` over the gate rank and ``(1 - f) x + g m`` in one
# launch instead of two GEMVs, two tanh and the compiled blend. The rank-64
# dot products are summed in index order (the GEMVs' order is their own), so
# this is near-exact.
_MIX_SOURCE = r"""
    uint c = thread_position_in_grid.x;
    uint n = thread_position_in_grid.y;
    if (c >= W) return;
    const device float* hrow = hidden + n * R;
    const device float* frow = forget + ((size_t)n * W + c) * R;
    const device float* grow = gain + ((size_t)n * W + c) * R;
    float fs = 0.0f;
    float gs = 0.0f;
    for (int r = 0; r < R; ++r) {
        fs += hrow[r] * frow[r];
        gs += hrow[r] * grow[r];
    }
    const float f = metal::precise::tanh(fs);
    const float g = metal::precise::tanh(gs);
    const float xv = x[(NX == 1 ? 0 : n) * W + c];
    const float mv = message[n * W + c];
    out[n * W + c] = (1.0f - f) * xv + g * mv;
"""
_mix_kernels = {}


def message_mix(module, hidden, forget, gain, x, message):
    """``hidden`` [N, R], ``forget``/``gain`` [N, W, R], ``x`` [NX, W] (NX 1
    or N), ``message`` [N, W], all FP32 -> [N, W] FP32; or ``None``."""
    if mx.default_device() != mx.gpu:
        return _decline(module, "message_not_gpu")
    if any(a.dtype != mx.float32 for a in (hidden, forget, gain, x, message)):
        return _decline(module, "message_dtype")
    n, rank = hidden.shape
    width = forget.shape[1]
    kernel = _mix_kernels.get(SAFE_MATH)
    if kernel is None:
        kernel = _mix_kernels[SAFE_MATH] = mx.fast.metal_kernel(
            name=f"inkling2_fuse_message_mix{'_safe' if SAFE_MATH else ''}",
            input_names=["hidden", "forget", "gain", "x", "message"],
            output_names=["out"],
            source=_MIX_SOURCE,
            header=_header(),
        )
    (out,) = kernel(
        inputs=[hidden, forget, gain, x, message],
        template=[("W", width), ("R", rank), ("NX", int(x.shape[0]))],
        grid=(width, n, 1),
        threadgroup=(min(256, width), 1, 1),
        output_shapes=[(n, width)],
        output_dtypes=[mx.float32],
    )
    _fuse.count(module, "message_mix_kernel")
    return out


# ----- BF16 loop-message controllers, row-independent (fuse_loop_bf16) -----
#
# ``inkling2_fuse_loop_bf16`` (original mlx2 code, 2026-11-26; the idea is
# from omlx#5586 "fused BF16 loop-controller matmuls", reimplemented): the
# controllers' published BF16 ``down``/``forget``/``gain`` weights are read
# directly and widened in registers, instead of reading the FP32 copies
# ``_fp32`` caches (half the controller weight bytes per visit).  Widening
# BF16 to FP32 is exact, so the products are the reference's products; the
# reductions run in this kernel's own order (near-exact against the MLX
# GEMMs).  Every output row is computed on its own (one simdgroup per
# ``down`` output, one thread per gate channel), so a row's values do not
# depend on how many rows share the call: verify rows equal one-row decode.
#
# down: hidden[row, r] = bf16(sum_k bf16(pair[row, k]) * down[c, r, k]) with
# pair = [rms(x), rms(m)] -- the reference casts the pair to BF16, runs a
# BF16 GEMM (FP32 accumulation, one BF16 round) and widens the result.
_LOOP_DOWN_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint r = thread_position_in_grid.y;
    const uint row = thread_position_in_grid.z;
    if (r >= R) return;
    const uint n = row / C;
    const uint c = row % C;
    const device float* xr = nx + ((size_t)n * CX + (CX == 1 ? 0 : c)) * W;
    const device float* mr = nm + (size_t)row * W;
    const device bfloat16_t* dr = down + ((size_t)c * R + r) * (2 * W);
    float s = 0.0f;
    for (uint k = lane; k < W; k += 32) {
        s += float(bfloat16_t(xr[k])) * float(dr[k]);
    }
    for (uint k = lane; k < W; k += 32) {
        s += float(bfloat16_t(mr[k])) * float(dr[W + k]);
    }
    s = simd_sum(s);
    if (lane == 0) hidden[(size_t)row * R + r] = float(bfloat16_t(s));
"""
# gates + blend: ``_MIX_SOURCE``'s arithmetic with BF16 weight loads, per
# (channel, controller, row); bitwise with ``message_mix`` on the FP32 copies.
_LOOP_GATES_SOURCE = r"""
    const uint w = thread_position_in_grid.x;
    const uint c = thread_position_in_grid.y;
    const uint n = thread_position_in_grid.z;
    if (w >= W) return;
    const uint row = n * C + c;
    const device float* hrow = hidden + (size_t)row * R;
    const device bfloat16_t* frow = forget + ((size_t)c * W + w) * R;
    const device bfloat16_t* grow = gain + ((size_t)c * W + w) * R;
    float fs = 0.0f;
    float gs = 0.0f;
    for (int r = 0; r < R; ++r) {
        fs += hrow[r] * float(frow[r]);
        gs += hrow[r] * float(grow[r]);
    }
    const float f = metal::precise::tanh(fs);
    const float g = metal::precise::tanh(gs);
    const float xv = x[((size_t)n * CX + (CX == 1 ? 0 : c)) * W + w];
    const float mv = message[(size_t)row * W + w];
    out[(size_t)row * W + w] = (1.0f - f) * xv + g * mv;
"""
# No row cap: the kernels compute every row on its own, so a call of any
# size gives each row the arithmetic of a one-row call (a cap on the call
# size would make a lane's numerics depend on how many lanes share the
# forward, and cold prefill differ from a short warm suffix).
_loop_kernels = {}


def _loop_kernel(name, source, inputs, outputs):
    key = (name, SAFE_MATH)
    kernel = _loop_kernels.get(key)
    if kernel is None:
        kernel = _loop_kernels[key] = mx.fast.metal_kernel(
            name=f"inkling2_{name}{'_safe' if SAFE_MATH else ''}",
            input_names=inputs, output_names=outputs, source=source, header=_header())
    return kernel


def loop_controllers(module, x, message, x_norm, message_norm, down, forget, gain):
    """``C`` loop-message controllers over any ``N`` rows, or ``None`` (counted
    decline) to run the reference ops.

    x / x_norm: [N, CX, W] FP32 (CX 1 = one input shared by all C);
    message / message_norm: [N, C, W]; down [C, R, 2W], forget / gain
    [C, W, R], all BF16.  Returns the blended inputs [N, C, W] FP32."""
    if mx.default_device() != mx.gpu:
        return _decline(module, "loop_bf16_not_gpu")
    if any(w.dtype != mx.bfloat16 for w in (down, forget, gain)):
        return _decline(module, "loop_bf16_weights")
    n, c, width = message.shape
    rank = down.shape[1]
    cx = x.shape[1]
    f32 = mx.float32
    template = [("W", width), ("R", rank), ("C", c), ("CX", cx)]
    (hidden,) = _loop_kernel("loop_down_bf16", _LOOP_DOWN_SOURCE, ["nx", "nm", "down"],
                             ["hidden"])(
        inputs=[x_norm.astype(f32), message_norm.astype(f32), down],
        template=template,
        grid=(32, rank, n * c),
        threadgroup=(32, min(8, rank), 1),
        output_shapes=[(n * c, rank)],
        output_dtypes=[f32],
    )
    (out,) = _loop_kernel("loop_gates_bf16", _LOOP_GATES_SOURCE,
                          ["hidden", "forget", "gain", "x", "message"], ["out"])(
        inputs=[hidden, forget, gain, x.astype(f32), message.astype(f32)],
        template=template,
        grid=(width, c, n),
        threadgroup=(min(256, width), 1, 1),
        output_shapes=[(n, c, width)],
        output_dtypes=[f32],
    )
    _fuse.count(module, "loop_bf16_kernel", n * c)
    return out
