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

``RESIDUAL_MLX_ORDER`` (module switch, default ON since 2026-11-26, Pierre)
swaps the residual and message mix for a launch that sums in the order of
the MLX kernels the reference dispatches, bit-identical to them on the
checkpoint (unit @981e86bcb); see the comment above ``mlx_order_plan``.
False restores the earlier near-exact kernel.
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


def softmax_mix(module, states, query, law="residual"):
    """``softmax(rms(S) q^T / sqrt(D), over S) rms(S)`` for one row.

    states: arrays of D elements (one row each), the last one any dtype;
    query: [nq, D] or [D].  Returns [1, 1, nq, D] FP32, or ``None``.
    ``law`` names the reference the call replaces (``"residual"``:
    ``language._attention_residual``; ``"message"``: the decoder's loop
    messages); only the ``RESIDUAL_MLX_ORDER`` candidate reads it."""
    width = states[-1].shape[-1]
    n = len(states)
    if n > 16 or any(s.size != width for s in states):
        return _decline(module, "layout")
    if RESIDUAL_MLX_ORDER:
        out = _softmax_mix_mlx_order(module, states, query, law)
        if out is not None:
            return out
        # A shape (or release) the MLX-order plan does not restate keeps the
        # earlier near-exact kernel, the default before 2026-11-26, rather
        # than dropping to the composed ops (counted mlx_order:<reason>).
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


# ----- MLX-order residual / message mix (candidate, module switch) --------
#
# ``RESIDUAL_MLX_ORDER`` (default True since 2026-11-26, Pierre: probe @79920fe96
# bitwise 64/64 cells, checkpoint @981e86bcb logits bitwise against the
# reference ops at 1K/16K, decode +3-5% over them and ~3% under the earlier
# near-exact kernel in a sanity A/B; False restores that kernel): the
# same one-launch mix as ``_residual_source``, but every reduction sums in
# the order of the MLX 0.32.5 kernel the composed reference dispatches for
# the call's shape (idea: oMLX #5557, "repeat the arithmetic of the MLX ops
# they replace", description only; the orders below are restated from the
# MLX v0.32.5 source, MIT, see provenance/inkling2-residual-mlx-order.json).
# One threadgroup of ``32 * ceil(ceil(D / 4) / 32)`` threads (576 at D 2304)
# runs all four stages for all query rows:
#
# (a) keys = ``mx.fast.rms_norm(S, None, 1e-6)``: MLX ``rms_single_row``
#     (rms_norm.metal:12-81; normalization.cpp:57-80 picks it for rows of at
#     most 4096 with exactly this threadgroup): four consecutive values per
#     thread squared and summed in order, ``simd_sum``, the per-simdgroup
#     sums ``simd_sum``-ed over a zeroed 32-slot array,
#     ``precise::rsqrt(sum / D + eps)``, key = ``x * inv`` (the weight is
#     ``array(1)``, fast.cpp:110).
# (b) scores: one dot per (state, query) row pair when either side has one
#     row (matmul.cpp:1659 ``gemv``; gemv_axbpy 1277-1295 picks
#     ``bm1_bn8_sm1_sn32_tn4`` for D >= 16 * rows: thread t of 256 sums k =
#     4t + j + 1024 i in order, ``simd_shuffle_down`` 16..1, then the eight
#     simdgroup partials in order, gemv.h:107-245); otherwise split-K
#     (matmul.cpp:970-990, 520-675: 16x16 tiles, ``P = min(max(2,
#     next_pow2(K / 16)), 32)`` partitions of ``(K / 16) / P`` 16-wide
#     iterations, the last partition also takes the rest; each partition an
#     8x8x8 ``simdgroup_multiply_accumulate`` chain from zero
#     (steel_gemm_splitk.h, mma.h:513-539), then ``0 + P0 + P1 + ...``
#     (gemm_splitk_accum)). FP32 never takes ``gemv_wide`` (FP16/BF16 only,
#     matmul.cpp:1446). Then ``* D**-0.5`` as its own rounding.
# (c) residual: ``mx.softmax(axis=-2)`` is not the Softmax primitive
#     (ops.cpp:4276-4305: max, ``precise::exp`` (unary_ops.h:177), sum,
#     divide); the sum is ``row_reduce_small`` for one query (sequential,
#     reduce_row.h:199-235) and ``col_reduce_small`` for several
#     (reduce_col.h:4-80; reduce.cpp:586-650: ``Y = min(8, N)`` row
#     strides, then the Y partials in order).  message: ``axis=-1`` is the
#     primitive, ``softmax_single_row`` (softmax.h:10-100, softmax.cpp:56-70:
#     one simdgroup, four values per lane, ``fast::exp``, ``simd_sum`` twice,
#     ``x * (1 / sum)``).
# (d) mix: one query row is ``gemv_t`` (gemv_axbpy 1243-1276:
#     ``bm1_bn16_sm8_sn4_tm4_tn4``; gemv.h:297-430: lane row c sums k = 4c..
#     4c + 3 in order, then ``(r0 + r4 + r2 + r6) + (r1 + r5 + r3 + r7)`` by
#     shuffles); several rows are the regular steel GEMM (K <= 16, one BK
#     tile, BK 32 on 'd' devices else 16, matmul.cpp:91-176; the unaligned K
#     tile first, steel_gemm_fused.h:133), an 8x8x8 MMA chain from zero.
#
# Products that MLX writes as ``acc += a * b`` go through ``MO_MAC`` so the
# probe can test the compiler's contraction choice (``RESIDUAL_MLX_MAC``:
# "source" keeps MLX's expression, "fma" forces ``fma``, "nocontract" adds
# ``fp contract(off)``).  Routes are restated, not read from MLX at run time:
# ``mlx_order_plan`` declines (counted ``mlx_order:<reason>``) any shape
# whose route it does not restate, and the caller then runs the reference.
# FP32 GEMMs take TF32/NAX routes when ``MLX_ENABLE_TF32`` is on (mlx2 pins
# it off, process_env.py), so those routes decline under TF32.
RESIDUAL_MLX_ORDER = True
RESIDUAL_MLX_MAC = "source"
_MO_MACS = {
    "source": ("", "acc += (a) * (b)"),
    "fma": ("", "acc = metal::fma((a), (b), acc)"),
    "nocontract": ("#pragma METAL fp contract(off)\n", "acc += (a) * (b)"),
}
_mo_kernels = {}
_ARCH = []
# The MLX release whose dispatch and orders are restated here. Two things
# this module cannot see (review 2026-11-26): MLX latches MLX_ENABLE_TF32 at
# its first FP32 matmul, so the pin must be applied before any (mlx2's
# process_env does; the plan reads the environment), and this kernel is
# compiled by the OS Metal compiler while MLX's come prebuilt, so a macOS
# update could change a contraction choice. Re-run
# scripts/research/inkling2_residual_bitwise_probe.py after either changes.
MLX_ORDER_VERSION = "0.32.5"
# A threadgroup MLX would refuse at eval time (above the pipeline's
# maxTotalThreadsPerThreadgroup) cannot fall back, so the plan stays well
# under 1024 threads (D <= 3072; the checkpoint's 2304 takes 576).
MLX_ORDER_MAX_THREADS = 768


def _architecture():
    if not _ARCH:
        try:
            info = (getattr(mx, "device_info", None) or mx.metal.device_info)()
            _ARCH.append(str(info.get("architecture", "")))
        except Exception:  # noqa: BLE001 - no device info: declined (see the plan)
            _ARCH.append("")
    return _ARCH[0]


def _ceil_div(a, b):
    return -(-a // b)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def mlx_order_plan(law, n, nq, width, *, tf32=None, arch=None):
    """MLX 0.32.5's routes for one ``law`` call of ``n`` states, ``nq``
    query rows and width ``width``: ``(plan, None)`` or ``(None, reason)``."""
    if law not in ("residual", "message"):
        return None, "law"
    if not 1 <= n <= 16:
        return None, "states"
    if not 1 <= nq <= 8:
        return None, "queries"
    if mx.__version__ != MLX_ORDER_VERSION:
        # The orders are MLX 0.32.5's; another release may dispatch other
        # kernels or tiles, so it runs the reference ops rather than a
        # near-exact launch that would still count as MLX-order.
        return None, "mlx_version"
    d = width
    tg = 32 * _ceil_div(_ceil_div(d, 4), 32)
    if d % 8 or not 256 <= tg <= MLX_ORDER_MAX_THREADS:
        return None, "width"
    if tf32 is None:
        from mlx2.process_env import tf32_enabled

        tf32 = tf32_enabled()
    arch = _architecture() if arch is None else arch
    if not arch:
        # The mix GEMM's K tile depends on the device class ('d' takes BK
        # 32); without device info the route is not known.
        return None, "architecture"
    sa, sb = (n, nq) if law == "residual" else (nq, n)
    plan = {"law": law, "n": n, "nq": nq, "d": d, "tg": tg, "sa": sa, "sb": sb}
    if min(sa, sb) == 1:
        # One dot per row pair. sa == sb == 1 is MLX's dot_product; the
        # weight is then exp(s - s) / 1 = 1 for any finite score, so the
        # order of that one dot does not reach the output.
        if d <= 64 or d < 16 * max(sa, sb):
            return None, "scores_gemv_tiles"
        plan["scores"] = "dot" if sa == sb == 1 else "gemv"
    else:
        if tf32:
            return None, "tf32"
        tk = d // 16
        threshold = 2048 if arch[-1:] in ("s", "d") else 1024
        if _ceil_div(sa, 16) * _ceil_div(sb, 16) > threshold or tk < 8 or d < max(sa, sb):
            return None, "scores_route"
        if d % 16:
            return None, "scores_k"
        parts = min(max(2, _next_pow2(tk // (_ceil_div(sa, 32) * _ceil_div(sb, 32)))), 32)
        iters = tk // parts
        plan.update(scores="splitk", parts=parts, iters=iters,
                    iters_last=tk - (parts - 1) * iters)
    if law == "residual":
        if nq > 4:
            return None, "softmax_cols"
        plan["softmax"] = "row_small" if nq == 1 else "col_small"
        plan["col_rows"] = min(8, n)
    else:
        plan["softmax"] = "block"
    if nq == 1:
        plan["mix"] = "gemv_t"
    else:
        if tf32:
            return None, "tf32"
        plan["mix"] = "gemm"
        plan["bk"] = 32 if arch.endswith("d") else 16
    return plan, None


def _mo_defines(plan, x, q, key):
    """``#define`` lines for the stage snippets: ``x(i, d)`` a raw state
    value, ``q(j, d)`` a query value, ``key(i, d)`` a normed state value."""
    a, b = ("MO_KEY", "MO_Q") if plan["law"] == "residual" else ("MO_Q", "MO_KEY")
    return "\n".join([
        f"#define MO_X(i, d) ({x})", f"#define MO_Q(j, d) ({q})", f"#define MO_KEY(i, d) ({key})",
        f"#define MO_A(r, k) {a}(r, k)", f"#define MO_B(r, k) {b}(r, k)",
    ])


_MO_DECL = r"""
    constexpr int MO_N = {n};
    constexpr int MO_D = {d};
    constexpr int MO_TG = {tg};
    constexpr int MO_NS = MO_TG / 32;
    constexpr int MO_NQ = {nq};
    constexpr int MO_SA = {sa};
    constexpr int MO_SB = {sb};
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const short mo_qid = short(lane / 4);
    const short mo_fm = (mo_qid & 4) + short((lane / 2) % 4);
    const short mo_fn = (mo_qid & 2) * 2 + short(lane % 2) * 2;
    threadgroup float mo_lsum[MO_N * 32];
    threadgroup float mo_inv[MO_N];
    threadgroup float mo_red[{red}];
    threadgroup float mo_s[MO_SA * MO_SB];
    threadgroup float mo_w[MO_NQ * MO_N];
    (void)mo_fm; (void)mo_fn; (void)mo_lsum; (void)mo_red;
"""

_MO_RMS = r"""
    if (sg == 0) {
        for (int i = 0; i < MO_N; ++i) mo_lsum[i * 32 + lane] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < MO_N; ++i) {
        float acc = 0;
        float xv[4];
        const uint base = tid * 4;
        if (base + 4 <= uint(MO_D)) {
            for (int j = 0; j < 4; j++) {
                xv[j] = MO_X(i, base + j);
                MO_MAC(acc, xv[j], xv[j]);
            }
        } else {
            for (int j = 0; j < 4; j++) {
                xv[j] = (base + j < uint(MO_D)) ? MO_X(i, base + j) : 0;
                MO_MAC(acc, xv[j], xv[j]);
            }
        }
        acc = simd_sum(acc);
        if (lane == 0) mo_lsum[i * 32 + sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (int i = 0; i < MO_N; ++i) {
            float acc = simd_sum(mo_lsum[i * 32 + lane]);
            if (lane == 0) mo_inv[i] = metal::precise::rsqrt(acc / uint(MO_D) + eps[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

# gemv (bm1_bn8_sm1_sn32_tn4): threads 0..255 are MLX's 8 x 32 lanes.
_MO_SCORES_GEMV = r"""
    if (tid < 256) {
        for (int a = 0; a < MO_SA; ++a) {
            for (int b = 0; b < MO_SB; ++b) {
                float r = 0;
                int bn = int(tid) * 4;
                for (int it = 0; it < MO_D / 1024; ++it) {
                    float vc[4];
                    float mv[4];
                    for (int j = 0; j < 4; ++j) vc[j] = MO_B(b, bn + j);
                    for (int j = 0; j < 4; ++j) mv[j] = MO_A(a, bn + j);
                    for (int j = 0; j < 4; ++j) MO_MAC(r, mv[j], vc[j]);
                    bn += 1024;
                }
                if (MO_D % 1024 > 0) {
                    float vc[4];
                    float mv[4];
                    for (int j = 0; j < 4; ++j) vc[j] = (bn + j < MO_D) ? MO_B(b, bn + j) : 0.0f;
                    for (int j = 0; j < 4; ++j) mv[j] = (bn + j < MO_D) ? MO_A(a, bn + j) : 0.0f;
                    for (int j = 0; j < 4; ++j) MO_MAC(r, mv[j], vc[j]);
                }
                for (ushort sn = 16; sn >= 1; sn >>= 1) r += simd_shuffle_down(r, sn);
                if (lane == 0) mo_red[(a * MO_SB + b) * 8 + sg] = r;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(tid); e < MO_SA * MO_SB; e += MO_TG) {
        float r = mo_red[e * 8];
        for (int g = 1; g < 8; ++g) r += mo_red[e * 8 + g];
        mo_s[e] = r * scale[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

# split-K (bm16_bn16_bk16_wm2_wn2): one simdgroup per (8x8 tile, partition).
_MO_SCORES_SPLITK = r"""
    {
        constexpr int FA = (MO_SA + 7) / 8;
        constexpr int FB = (MO_SB + 7) / 8;
        constexpr int NP = {parts};
        for (int task = int(sg); task < FA * FB * NP; task += MO_NS) {
            const int kp = task % NP;
            const int fa = (task / NP) / FB;
            const int fb = (task / NP) % FB;
            const int ar = fa * 8 + mo_fm;
            const int bc = fb * 8 + mo_fn;
            const int k0 = kp * {iters} * 16;
            const int iters = (kp == NP - 1) ? {iters_last} : {iters};
            simdgroup_matrix<float, 8, 8> mc;
            mc.thread_elements()[0] = 0.0f;
            mc.thread_elements()[1] = 0.0f;
            for (int it = 0; it < iters; ++it) {
                for (int kk = 0; kk < 16; kk += 8) {
                    const int k = k0 + it * 16 + kk;
                    simdgroup_matrix<float, 8, 8> ma;
                    simdgroup_matrix<float, 8, 8> mb;
                    ma.thread_elements()[0] = ar < MO_SA ? MO_A(ar, k + mo_fn) : 0.0f;
                    ma.thread_elements()[1] = ar < MO_SA ? MO_A(ar, k + mo_fn + 1) : 0.0f;
                    mb.thread_elements()[0] = bc < MO_SB ? MO_B(bc, k + mo_fm) : 0.0f;
                    mb.thread_elements()[1] = bc + 1 < MO_SB ? MO_B(bc + 1, k + mo_fm) : 0.0f;
                    simdgroup_multiply_accumulate(mc, ma, mb, mc);
                }
            }
            if (ar < MO_SA && bc < MO_SB)
                mo_red[(kp * MO_SA + ar) * MO_SB + bc] = mc.thread_elements()[0];
            if (ar < MO_SA && bc + 1 < MO_SB)
                mo_red[(kp * MO_SA + ar) * MO_SB + bc + 1] = mc.thread_elements()[1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = int(tid); e < MO_SA * MO_SB; e += MO_TG) {
        float o = 0;
        for (int kp = 0; kp < {parts}; ++kp) o += mo_red[kp * MO_SA * MO_SB + e];
        mo_s[e] = o * scale[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

# residual softmax over the states (axis -2 of [.., N, NQ]): composed ops.
_MO_SOFTMAX_COMPOSED = r"""
    for (int qr = int(tid); qr < MO_NQ; qr += MO_TG) {
        float m = mo_s[qr];
        for (int i = 1; i < MO_N; ++i) m = metal::max(m, mo_s[i * MO_NQ + qr]);
        float e[MO_N];
        for (int i = 0; i < MO_N; ++i) e[i] = metal::precise::exp(mo_s[i * MO_NQ + qr] - m);
        float total;
SUM
        for (int i = 0; i < MO_N; ++i) mo_w[qr * MO_N + i] = e[i] / total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""
_MO_SUM_ROW = r"""
        total = 0;
        for (int i = 0; i < MO_N; ++i) total = e[i] + total;
"""
_MO_SUM_COL = r"""
        float part[{y}];
        for (int y = 0; y < {y}; ++y) {
            part[y] = 0;
            for (int i = y; i < MO_N; i += {y}) part[y] = e[i] + part[y];
        }
        total = part[0];
        for (int y = 1; y < {y}; ++y) total = part[y] + total;
"""

# message softmax over the bank (axis -1 of [.., NQ, N]): softmax_single_row.
_MO_SOFTMAX_BLOCK = r"""
    if (sg == 0) {
        for (int qr = 0; qr < MO_NQ; ++qr) {
            float ld[4];
            for (int j = 0; j < 4; ++j) {
                const int c = int(lane) * 4 + j;
                ld[j] = c < MO_N ? mo_s[qr * MO_N + c] : -INFINITY;
            }
            float mval = -metal::numeric_limits<float>::max();
            for (int j = 0; j < 4; ++j) mval = (mval < ld[j]) ? ld[j] : mval;
            mval = simd_max(mval);
            float nrm = 0;
            for (int j = 0; j < 4; ++j) {
                float ex = metal::fast::exp(ld[j] - mval);
                ld[j] = ex;
                nrm += ex;
            }
            nrm = simd_sum(nrm);
            nrm = simd_sum(lane == 0 ? nrm : 0.0f);
            nrm = 1 / nrm;
            for (int j = 0; j < 4; ++j) {
                const int c = int(lane) * 4 + j;
                if (c < MO_N) mo_w[qr * MO_N + c] = ld[j] * nrm;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""

# gemv_t (bm1_bn16_sm8_sn4_tm4_tn4), one query row: per output, the eight
# lane rows' partials and MLX's shuffle tree.
_MO_MIX_GEMV_T = r"""
    for (int d = int(tid); d < MO_D; d += MO_TG) {
        float r[8];
        for (int c = 0; c < 8; ++c) {
            r[c] = 0;
            int bm = 4 * c;
            for (int it = 0; it < MO_N / 32; ++it) {
                for (int tm = 0; tm < 4; ++tm) {
                    const float vc = mo_w[bm + tm];
                    const float mv = MO_KEY(bm + tm, d);
                    MO_MAC(r[c], vc, mv);
                }
                bm += 32;
            }
            if (MO_N % 32 > 0) {
                for (int tm = 0; tm < 4 && bm + tm < MO_N; ++tm) {
                    const float vc = mo_w[bm + tm];
                    const float mv = MO_KEY(bm + tm, d);
                    MO_MAC(r[c], vc, mv);
                }
            }
        }
        for (int c = 0; c < 4; ++c) r[c] += r[c + 4];
        for (int c = 0; c < 2; ++c) r[c] += r[c + 2];
        r[0] += r[1];
        out[d] = r[0];
    }
"""

# steel GEMM, several query rows: one simdgroup per 8-column tile.
_MO_MIX_GEMM = r"""
    for (int fd = int(sg); fd < MO_D / 8; fd += MO_NS) {
        simdgroup_matrix<float, 8, 8> mc;
        mc.thread_elements()[0] = 0.0f;
        mc.thread_elements()[1] = 0.0f;
        const int col = fd * 8 + mo_fn;
        for (int c = 0; c < {bk} / 8; ++c) {
            const int k = c * 8;
            simdgroup_matrix<float, 8, 8> ma;
            simdgroup_matrix<float, 8, 8> mb;
            ma.thread_elements()[0] = (mo_fm < MO_NQ && k + mo_fn < MO_N) ? mo_w[mo_fm * MO_N + k + mo_fn] : 0.0f;
            ma.thread_elements()[1] = (mo_fm < MO_NQ && k + mo_fn + 1 < MO_N) ? mo_w[mo_fm * MO_N + k + mo_fn + 1] : 0.0f;
            mb.thread_elements()[0] = (k + mo_fm < MO_N) ? MO_KEY(k + mo_fm, col) : 0.0f;
            mb.thread_elements()[1] = (k + mo_fm < MO_N) ? MO_KEY(k + mo_fm, col + 1) : 0.0f;
            simdgroup_multiply_accumulate(mc, ma, mb, mc);
        }
        if (mo_fm < MO_NQ) {
            out[mo_fm * MO_D + col] = mc.thread_elements()[0];
            out[mo_fm * MO_D + col + 1] = mc.thread_elements()[1];
        }
    }
"""


def _mo_red_size(plan):
    if plan["scores"] == "splitk":
        return plan["parts"] * plan["sa"] * plan["sb"]
    return plan["sa"] * plan["sb"] * 8


def mo_phase(plan, phase):
    """The Metal text of one stage (``rms``, ``scores``, ``softmax``,
    ``mix``) for ``plan``; the probe builds single-stage kernels from it."""
    if phase == "rms":
        return _MO_RMS
    if phase == "scores":
        if plan["scores"] == "splitk":
            return (_MO_SCORES_SPLITK.replace("{parts}", str(plan["parts"]))
                    .replace("{iters_last}", str(plan["iters_last"]))
                    .replace("{iters}", str(plan["iters"])))
        return _MO_SCORES_GEMV
    if phase == "softmax":
        if plan["softmax"] == "block":
            return _MO_SOFTMAX_BLOCK
        total = (_MO_SUM_ROW if plan["softmax"] == "row_small"
                 else _MO_SUM_COL.replace("{y}", str(plan["col_rows"])))
        return _MO_SOFTMAX_COMPOSED.replace("SUM", total.strip("\n"))
    if phase == "mix":
        if plan["mix"] == "gemm":
            return _MO_MIX_GEMM.replace("{bk}", str(plan["bk"]))
        return _MO_MIX_GEMV_T
    raise ValueError(phase)


def mo_source(plan, phases, defines, pre="", post="", mac=None):
    """``(header, source)`` for ``phases`` of ``plan`` between ``pre`` and
    ``post`` glue; ``defines`` from ``_mo_defines``."""
    pragma, mac_expr = _MO_MACS[mac or RESIDUAL_MLX_MAC]
    decl = _MO_DECL
    for key in ("n", "d", "tg", "nq", "sa", "sb"):
        decl = decl.replace("{" + key + "}", str(plan[key]))
    decl = decl.replace("{red}", str(_mo_red_size(plan)))
    body = "\n".join([defines, f"#define MO_MAC(acc, a, b) {mac_expr}", decl, pre,
                      *(mo_phase(plan, p) for p in phases), post])
    return pragma, body


def _mo_state_expr(n):
    expr = "float(p[(d)])"
    for i in range(n - 2, -1, -1):
        expr = f"(i) == {i} ? float(b{i}[(d)]) : ({expr})"
    return expr


def _softmax_mix_mlx_order(module, states, query, law):
    width = states[-1].shape[-1]
    n = len(states)
    q = query.reshape(-1, width)
    nq = q.shape[0]
    plan, reason = mlx_order_plan(law, n, nq, width)
    if plan is None:
        return _decline(module, "mlx_order:" + reason)
    mac = RESIDUAL_MLX_MAC
    key = (law, n, nq, width, tuple(sorted(plan.items())), mac)
    kernel = _mo_kernels.get(key)
    if kernel is None:
        defines = _mo_defines(plan, _mo_state_expr(n), "float(q[(j) * MO_D + (d)])",
                              "MO_X(i, d) * mo_inv[(i)]")
        header, source = mo_source(plan, ("rms", "scores", "softmax", "mix"), defines, mac=mac)
        kernel = _mo_kernels[key] = mx.fast.metal_kernel(
            name=f"inkling2_fuse_residual_mlx_{law}_{n}_{nq}_{mac}",
            input_names=[f"b{i}" for i in range(n - 1)] + ["p", "q", "scale", "eps"],
            output_names=["out"],
            source=source,
            header=header,
        )
    out = kernel(
        inputs=[*(s.reshape(width) for s in states[:-1]), states[-1].reshape(width), q,
                mx.array([width ** -0.5], mx.float32), mx.array([1e-6], mx.float32)],
        grid=(plan["tg"], 1, 1),
        threadgroup=(plan["tg"], 1, 1),
        output_shapes=[(1, 1, nq, width)],
        output_dtypes=[mx.float32],
    )[0]
    _fuse.count(module, "mlx_order_kernel:" + law)
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
