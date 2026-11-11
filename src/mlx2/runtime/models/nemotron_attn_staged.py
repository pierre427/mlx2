# SPDX-License-Identifier: MIT
"""Opt-in Nemotron-H causal prefill attention with staged K/V tiles (GQA 16).

Reimplemented from the layout described in nace-ai/kapitsa PR #890
(``nh_attn_ts_s8``): one simdgroup per query token whose 16 query heads form
one 16-row matrix-unit tile, eight tokens per threadgroup, and each 32-key
tile of K and V loaded once into threadgroup memory (16 KB) for all eight
simdgroups.  No Kapitsa code was read or copied; see
provenance/kapitsa-nemotron-prefill.json.  The matrix-unit fragment layout
(``mpp::tensor_ops::matmul2d`` 16x32x16, cooperative-tensor element order,
online max/sum update) is this tree's ``qwen4_qsa_nax`` / Inkling NAX
prefill tile, with device K/V reads replaced by threadgroup reads.

Admission: B >= 1, 16 query heads per KV head, head_dim 128, 16-bit
activations, a string ``"causal"`` mask (bottom-right aligned, as MLX's
SDPA), no sinks, no quantized or bucketed cache, and at least
``MIN_ROWS`` query rows.  Everything else stays on
``mx.fast.scaled_dot_product_attention``.

Not bit-identical to MLX's SDPA (different tile shapes, ``exp`` instead of
``exp2``, fp16-class P in the P x V product), so the switch is APCv2-bound.

Switch: ``MLX2_NEMOTRON_ATTN_STAGED`` (off by default; any value but empty/0/false/off/no enables).

Why it stays off (M5 Max, MLX 0.32.4, 2026-11-10, Super attention geometry
32 q / 2 kv heads, D 128, bf16, 4096 query rows): MLX's SDPA ran 50-54
TFLOP/s at 4K-64K keys and 46 at 131K; this kernel ran 21-22 TFLOP/s
(2.4x slower), with 4, 8 or 16 tokens per threadgroup making no
difference, so it is issue-bound in the 16-row tile, not K/V-bandwidth
bound.  Kapitsa's gain came from replacing its own unstaged per-token
kernel (13-24 TFLOP/s past 128K); MLX's steel attention already stages
K/V tiles in threadgroup memory per query block, so the premise of
PR #890 does not apply to mlx2's default path.
"""

from __future__ import annotations

import os
import threading

import mlx.core as mx

ATTN_STAGED_ENV = "MLX2_NEMOTRON_ATTN_STAGED"
MIN_ROWS = 32
TOKENS_PER_GROUP = 8
_OFF = {"", "0", "false", "off", "no"}  # apc_numerics "flag" rule

_counts = {"staged": 0, "refused": 0}
_refusals: dict[str, int] = {}

_HEADER = (
    "#include <metal_stdlib>\n#include <metal_simdgroup>\n"
    "#include <metal_simdgroup_matrix>\n"
    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
    "using namespace metal;\n"
)

_SOURCE = r"""
    constexpr int D = 128;
    constexpr int GQA = 16;
    constexpr int TK = 32;                 // keys per staged tile
    constexpr int NKS = D / 16;            // QK reduction steps
    constexpr int NNT = D / 32;            // PV output tiles
    constexpr int NT = TPG * 32;

    const int L = dims[0], TOT = dims[1], OFF = dims[2];
    const float qk_scale = scale[0];
    const ushort sg = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const int tid = thread_position_in_threadgroup.x;
    const int t0 = int(threadgroup_position_in_grid.y) * TPG;
    const uint bkv = threadgroup_position_in_grid.z;
    const uint b = bkv / NKVH;
    const uint hkv = bkv % NKVH;
    const int tok = t0 + sg;
    const bool live = tok < L;
    const int limit = OFF + tok;           // last visible key position

    const short qid = lane >> 2;
    const short fm0 = ((qid & 4) | ((lane >> 1) & 3));
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;

    constexpr int DP = D + 8;              // padded row: spreads the 8 fragment rows over banks
    threadgroup T Ks[TK * DP];
    threadgroup T Vs[TK * DP];
    typedef metal::vec<T, 4> t4;

    const size_t k_head = (size_t)b * k_strides[0] + (size_t)hkv * k_strides[1];
    const size_t v_head = (size_t)b * v_strides[0] + (size_t)hkv * v_strides[1];

    int head_of[2];
    for (short rr = 0; rr < 2; ++rr) head_of[rr] = int(hkv) * GQA + fm0 + rr * 8;

    typedef metal::vec<T, 8> frag_t;
    frag_t qf[NKS];
    for (short ks = 0; ks < NKS; ks++) {
        for (short i = 0; i < 8; i++) {
            short rr = (i >> 2);
            int c = ks * 16 + fn0 + (i % 4);
            qf[ks][i] = live
                ? q[(size_t)b * q_strides[0] + head_of[rr] * q_strides[1]
                    + (size_t)tok * q_strides[2] + c * q_strides[3]]
                : T(0);
        }
    }

    float of[NNT][16];
    for (short t = 0; t < NNT; t++)
        for (short i = 0; i < 16; i++) of[t][i] = 0.0f;
    float rmax_r[2] = {-INFINITY, -INFINITY};
    float rsum_r[2] = {0.0f, 0.0f};

    constexpr auto desc_qk = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto desc_pv = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // 16-byte row loads only for a unit-stride, 8-aligned head dim (cache
    // slices of a contiguous buffer); any other layout loads per element.
    const bool VEC_KV = k_strides[3] == 1 && v_strides[3] == 1
        && (k_strides[0] | k_strides[1] | k_strides[2]
            | v_strides[0] | v_strides[1] | v_strides[2]) % 8 == 0;
    const int end = min(TOT, OFF + min(t0 + TPG, L));
    for (int u0 = 0; u0 < end; u0 += TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Stage the tile: rows u0..u0+31, zero past the cache end.
        for (int f = tid; f < TK * D / 8; f += NT) {
            const int r = f / (D / 8), c = (f % (D / 8)) * 8;
            const int kp = u0 + r;
            if (VEC_KV) {
                uint4 kv = uint4(0), vv = uint4(0);
                if (kp < TOT) {
                    kv = *(const device uint4*)(k + k_head + (size_t)kp * k_strides[2] + c);
                    vv = *(const device uint4*)(v + v_head + (size_t)kp * v_strides[2] + c);
                }
                *(threadgroup uint4*)(Ks + r * DP + c) = kv;
                *(threadgroup uint4*)(Vs + r * DP + c) = vv;
            } else {
                for (short e = 0; e < 8; ++e) {
                    Ks[r * DP + c + e] = kp < TOT
                        ? k[k_head + (size_t)kp * k_strides[2] + (size_t)(c + e) * k_strides[3]] : T(0);
                    Vs[r * DP + c + e] = kp < TOT
                        ? v[v_head + (size_t)kp * v_strides[2] + (size_t)(c + e) * v_strides[3]] : T(0);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!live || u0 > limit) continue;

        short krow[4];
        for (short qq = 0; qq < 4; ++qq) krow[qq] = fm0 + qq * 8;

        float s[16];
        {
            mpp::tensor_ops::matmul2d<desc_qk, metal::execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<T, T, float>();
            auto cb = op.get_right_input_cooperative_tensor<T, T, float>();
            auto cc = op.get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ca)>,
                metal::remove_addrspace_t<decltype(cb)>, float>();
            for (short i = 0; i < 16; i++) cc[i] = 0.0f;
            for (short ks = 0; ks < NKS; ks++) {
                for (short i = 0; i < 8; i++) ca[i] = qf[ks][i];
                const int c = ks * 16 + fn0;
                for (short hh = 0; hh < 2; hh++) {
                    const t4 a0 = *(const threadgroup t4*)(Ks + krow[hh] * DP + c);
                    const t4 a1 = *(const threadgroup t4*)(Ks + krow[2 + hh] * DP + c);
                    for (short e = 0; e < 4; e++) {
                        cb[hh * 4 + e] = a0[e];
                        cb[8 + hh * 4 + e] = a1[e];
                    }
                }
                op.run(ca, cb, cc);
            }
            for (short i = 0; i < 16; i++) s[i] = cc[i] * qk_scale;
        }
        // The mask depends on the key only: every row is this token.
        for (short f = 0; f < 2; ++f)
            for (short co = 0; co < 4; ++co) {
                const int kp = u0 + f * 16 + fn0 + co;
                if (kp > limit || kp >= TOT) {
                    s[f * 8 + co] = -INFINITY;
                    s[f * 8 + 4 + co] = -INFINITY;
                }
            }

        float rowmax[2];
        for (short rr = 0; rr < 2; rr++) {
            float m = -INFINITY;
            for (short f = 0; f < 2; f++)
                for (short co = 0; co < 4; co++)
                    m = metal::max(m, s[f * 8 + rr * 4 + co]);
            m = metal::max(m, metal::simd_shuffle_xor(m, ushort(1)));
            m = metal::max(m, metal::simd_shuffle_xor(m, ushort(8)));
            rowmax[rr] = m;
        }
        float alpha[2], p[16];
        for (short rr = 0; rr < 2; rr++) {
            float mn = metal::max(rmax_r[rr], rowmax[rr]);
            bool ok = mn > -INFINITY;
            alpha[rr] = ok ? metal::fast::exp(rmax_r[rr] - mn) : 1.0f;
            float sm = 0.0f;
            for (short f = 0; f < 2; f++)
                for (short co = 0; co < 4; co++) {
                    short idx = f * 8 + rr * 4 + co;
                    float pv = ok ? metal::fast::exp(s[idx] - mn) : 0.0f;
                    p[idx] = pv;
                    sm += pv;
                }
            sm += metal::simd_shuffle_xor(sm, ushort(1));
            sm += metal::simd_shuffle_xor(sm, ushort(8));
            rmax_r[rr] = mn;
            rsum_r[rr] = rsum_r[rr] * alpha[rr] + sm;
        }

        for (short t = 0; t < NNT; t++) {
            for (short f = 0; f < 2; f++)
                for (short rr = 0; rr < 2; rr++)
                    for (short co = 0; co < 4; co++)
                        of[t][f * 8 + rr * 4 + co] *= alpha[rr];
            const int nbase = t * 32;
            mpp::tensor_ops::matmul2d<desc_pv, metal::execution_simdgroup> op;
            auto ca = op.get_left_input_cooperative_tensor<T, T, float>();
            auto cb = op.get_right_input_cooperative_tensor<T, T, float>();
            auto cc = op.get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ca)>,
                metal::remove_addrspace_t<decltype(cb)>, float>();
            for (short i = 0; i < 16; i++) cc[i] = of[t][i];
            for (short ks = 0; ks < 2; ks++) {
                for (short i = 0; i < 8; i++) ca[i] = (T)p[ks * 8 + i];
                const int co = nbase + fn0;
                for (short hh = 0; hh < 2; hh++) {
                    const short r = krow[ks * 2 + hh];
                    const t4 a0 = *(const threadgroup t4*)(Vs + r * DP + co);
                    const t4 a1 = *(const threadgroup t4*)(Vs + r * DP + co + 16);
                    for (short e = 0; e < 4; e++) {
                        cb[hh * 4 + e] = a0[e];
                        cb[8 + hh * 4 + e] = a1[e];
                    }
                }
                op.run(ca, cb, cc);
            }
            for (short i = 0; i < 16; i++) of[t][i] = cc[i];
        }
    }

    if (!live) return;
    for (short t = 0; t < NNT; t++)
        for (short f = 0; f < 2; f++)
            for (short rr = 0; rr < 2; rr++) {
                float inv = (rsum_r[rr] > 0.0f) ? (1.0f / rsum_r[rr]) : 0.0f;
                for (short co = 0; co < 4; co++) {
                    int c = t * 32 + f * 16 + fn0 + co;
                    out[(((size_t)b * NQH + head_of[rr]) * L + tok) * D + c] =
                        T(of[t][f * 8 + rr * 4 + co] * inv);
                }
            }
"""

_kernel = None
_LOCK = threading.Lock()  # kernel cache and counters


def _get_kernel():
    global _kernel
    with _LOCK:
        return _build_kernel()


def _build_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="mlx2_nemotron_attn_staged",
            input_names=["q", "k", "v", "scale", "dims"],
            output_names=["out"],
            header=_HEADER,
            source=_SOURCE,
            ensure_row_contiguous=False,
        )
    return _kernel


def enabled(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(ATTN_STAGED_ENV, "")).strip().lower() not in _OFF


def refusal(queries, keys, values, cache, mask) -> str | None:
    """Why this call stays on MLX SDPA, or ``None`` when it is admitted."""
    if not (isinstance(mask, str) and mask == "causal"):
        return "mask is not the string causal"
    if cache is not None and (
        hasattr(cache, "bits") or getattr(cache, "bucketed_attention", None) is not None
    ):
        return "quantized or bucketed cache"
    if queries.ndim != 4 or keys.ndim != 4 or values.shape != keys.shape:
        return "rank or K/V shape"
    b, hq, length, dim = queries.shape
    if keys.shape[0] != b or dim != 128 or keys.shape[3] != 128:
        return f"head_dim {dim}"
    if keys.shape[1] <= 0 or hq != 16 * keys.shape[1]:
        return f"GQA {hq}/{keys.shape[1]}"
    if length < MIN_ROWS:
        return "short query"
    if keys.shape[2] < length:
        return "fewer keys than queries"
    if queries.dtype not in (mx.bfloat16, mx.float16) or any(
        a.dtype != queries.dtype for a in (keys, values)
    ):
        return "requires matching 16-bit q/k/v"
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "requires Metal"
    return None


def staged_attention(queries, keys, values, *, scale: float):
    b, hq, length, dim = queries.shape
    hk = keys.shape[1]
    total = keys.shape[2]
    tpg = TOKENS_PER_GROUP
    return _get_kernel()(
        inputs=[
            queries,
            keys,
            values,
            mx.array([scale], mx.float32),
            mx.array([length, total, total - length], mx.int32),
        ],
        template=[("T", queries.dtype), ("NQH", hq), ("NKVH", hk), ("TPG", tpg)],
        grid=(32 * tpg, (length + tpg - 1) // tpg, b * hk),
        threadgroup=(32 * tpg, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]


def maybe_staged_attention(queries, keys, values, *, cache, scale, mask):
    """The staged-tile output when the switch is on and admits the call."""
    if not enabled():
        return None
    reason = refusal(queries, keys, values, cache, mask)
    if reason is not None:
        with _LOCK:
            _counts["refused"] += 1
            _refusals[reason] = _refusals.get(reason, 0) + 1
        return None
    with _LOCK:
        _counts["staged"] += 1
    return staged_attention(queries, keys, values, scale=scale)


def status() -> dict:
    return {
        "switch": ATTN_STAGED_ENV,
        "enabled": enabled(),
        "counts": dict(_counts),
        "refusal_reasons": dict(_refusals),
    }
