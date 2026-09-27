# SPDX-License-Identifier: Apache-2.0
# Research-only extraction of jundot/omlx at 3e9703518984c294c2ed989cf968727d0e0ffa56.
# See provenance/omlx3958-wide-verify-sdpa.json. No runtime route is modified.
"""Isolated dense B1 M4-8 simdgroup-matrix verify attention candidate.

Importing this module does not import MLX or compile Metal. The MPP tensor_ops
arm is deliberately absent; physical accelerator attribution requires a trace.
"""

from __future__ import annotations


def admission_reasons(q_shape, k_shape, v_shape, *, q_dtype: str, k_dtype: str,
                      v_dtype: str, cache_type: str, cache_offset: int,
                      mask: str, left_padding: int = 0, right_padding: int = 0,
                      sinks: bool = False, quantized: bool = False) -> list[str]:
    """Validate only the current Qwen3.8-27B plain dense verify geometry.

    ``cache_offset`` is the offset *after* the verify rows have been appended.
    A nonempty reason list means the candidate must not be called.
    """
    reasons = []
    if len(q_shape) != 4 or len(k_shape) != 4 or len(v_shape) != 4:
        return ["rank"]
    b, hq, m, dq = q_shape
    bk, hk, t, dk = k_shape
    bv, hv, tv, dv = v_shape
    if (b, bk, bv) != (1, 1, 1):
        reasons.append("batch")
    if (hq, hk, hv) != (24, 4, 4) or (dq, dk, dv) != (256, 256, 256):
        reasons.append("heads_or_dim")
    if not 4 <= m <= 8:
        reasons.append("verify_width")
    if t != tv or t < max(m, 8) or cache_offset != t:
        reasons.append("cache_length")
    if q_dtype not in ("bfloat16", "float16") or (q_dtype, k_dtype, v_dtype) != (q_dtype,) * 3:
        reasons.append("dtype")
    if cache_type != "KVCache" or quantized:
        reasons.append("cache_type")
    if mask != "causal" or left_padding or right_padding or sinks:
        reasons.append("mask_or_padding")
    return reasons


def dtype_name(dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def wide_attention(queries, keys, values, *, cache, scale: float, mask: str,
                   left_padding: int = 0, right_padding: int = 0,
                   sinks: bool = False):
    """Research-only call after normal KVCache.update_and_fetch; no cache writes."""
    from mlx2.runtime.models.cache import KVCache
    reasons = admission_reasons(
        queries.shape, keys.shape, values.shape,
        q_dtype=dtype_name(queries.dtype), k_dtype=dtype_name(keys.dtype),
        v_dtype=dtype_name(values.dtype),
        cache_type="KVCache" if type(cache) is KVCache else type(cache).__name__,
        cache_offset=cache.offset, mask=mask, left_padding=left_padding,
        right_padding=right_padding, sinks=sinks, quantized=hasattr(cache, "bits"),
    )
    if type(cache) is not KVCache:
        reasons.append("exact_cache_class")
    if reasons:
        raise ValueError("ineligible isolated wide SDPA: " + ",".join(sorted(set(reasons))))
    return _wide_causal_sdpa(queries, keys, values, scale)


# Verify blocks wider than the vector kernel's row budget: one threadgroup per
# query head and key split holds all eight rows. Its four simdgroups each own
# 64 head dims, so partial scores meet in threadgroup memory once per 32-key
# block; a second kernel merges the splits.
_WIDE_PARTIAL = """
    constexpr int D = 256;
    constexpr int BK = 32;
    constexpr int SS = 36;   // padded partial-S row stride (floats)
    constexpr int PS = 40;   // padded P row stride (bf16)
    uint tid = thread_position_in_threadgroup.x;
    uint sg = tid / 32;
    uint lane = tid % 32;
    // Heads vary fastest so the query heads of one KV head read a split together.
    int qh = int(threadgroup_position_in_grid.x);
    int split = int(threadgroup_position_in_grid.y);
    int h = qh / G;
    int T = int(params[0]);
    int L = int(params[1]);
    int chunk = int(params[2]);
    float scale = as_type<float>(params[4]);
    int t_begin = split * chunk;
    int t_end = min(t_begin + chunk, T);

    threadgroup float Sp[4 * 8 * SS];
    threadgroup T_ P[8 * PS];
    threadgroup float st_m[8];
    threadgroup float st_l[8];
    threadgroup float st_a[8];

    // This simdgroup's 64-dim slice of the eight query rows, kept in registers.
    const device T_* qp = q + qh * 8 * D + int(sg) * 64;
    simdgroup_matrix<T_, 8, 8> qa[8];
    for (int dk = 0; dk < 8; ++dk)
        simdgroup_load(qa[dk], qp + dk * 8, D);
    const device T_* kh = k + h * k_strides[1] + int(sg) * 64;
    const device T_* vh = v + h * v_strides[1] + int(sg) * 64;
    int kst = int(k_strides[2]);
    int vst = int(v_strides[2]);

    simdgroup_matrix<float, 8, 8> o[8];
    for (int j = 0; j < 8; ++j) o[j] = simdgroup_matrix<float, 8, 8>(0.0f);
    if (tid < 8) {
        st_m[tid] = -INFINITY;
        st_l[tid] = 0.0f;
    }
    short qid = short(lane / 4);
    short fm = (qid & 4) + short((lane / 2) % 4);

    for (int t0 = t_begin; t0 < t_end; t0 += BK) {
        // Partial scores over this simdgroup's dims for four 8-key tiles.
        for (int c = 0; c < 4; ++c) {
            int ts = min(t0 + c * 8, T - 8);
            simdgroup_matrix<float, 8, 8> s = simdgroup_matrix<float, 8, 8>(0.0f);
            for (int dk = 0; dk < 8; ++dk) {
                simdgroup_matrix<T_, 8, 8> kb;
                simdgroup_load(kb, kh + ts * kst + dk * 8, kst, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s, qa[dk], kb, s);
            }
            simdgroup_store(s, Sp + int(sg) * 8 * SS + c * 8, SS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax: 16 threads per row, two columns each.
        {
            int r = int(tid) / 16;
            int cp = (int(tid) % 16) * 2;
            float sv[2];
            float rmax = -INFINITY;
            for (int e = 0; e < 2; ++e) {
                int col = cp + e;
                int c = col / 8;
                int base = t0 + c * 8;
                int key = min(base, T - 8) + (col - c * 8);
                bool ok = r < L && key >= base && key < t_end && key <= T - L + r;
                float val = Sp[r * SS + col] + Sp[8 * SS + r * SS + col]
                    + Sp[16 * SS + r * SS + col] + Sp[24 * SS + r * SS + col];
                sv[e] = ok ? val * scale : -INFINITY;
                rmax = max(rmax, sv[e]);
            }
            for (int off = 1; off < 16; off <<= 1)
                rmax = max(rmax, simd_shuffle_xor(rmax, ushort(off)));
            float m_old = st_m[r];
            float m_new = max(m_old, rmax);
            float rsum = 0.0f;
            for (int e = 0; e < 2; ++e) {
                float p = m_new == -INFINITY ? 0.0f : exp(sv[e] - m_new);
                rsum += p;
                P[r * PS + cp + e] = T_(p);
            }
            for (int off = 1; off < 16; off <<= 1)
                rsum += simd_shuffle_xor(rsum, ushort(off));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if ((tid % 16) == 0) {
                float alpha = m_new == -INFINITY ? 1.0f : exp(m_old - m_new);
                st_a[r] = alpha;
                st_l[r] = st_l[r] * alpha + rsum;
                st_m[r] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float alpha = st_a[fm];
        for (int j = 0; j < 8; ++j) {
            o[j].thread_elements()[0] *= alpha;
            o[j].thread_elements()[1] *= alpha;
        }
        for (int kt = 0; kt < 4; ++kt) {
            simdgroup_matrix<T_, 8, 8> pa;
            simdgroup_load(pa, P + kt * 8, PS);
            int ts = min(t0 + kt * 8, T - 8);
            // Issue the tile's value loads before the products that use them.
            simdgroup_matrix<T_, 8, 8> vb[8];
            for (int j = 0; j < 8; ++j)
                simdgroup_load(vb[j], vh + ts * vst + j * 8, vst);
            for (int j = 0; j < 8; ++j)
                simdgroup_multiply_accumulate(o[j], pa, vb[j], o[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* ob = o_part + ((split * H + qh) * 8) * D + int(sg) * 64;
    for (int j = 0; j < 8; ++j)
        simdgroup_store(o[j], ob + j * 8, D);
    if (tid < 8) {
        ml_part[((split * H + qh) * 8 + int(tid)) * 2] = st_m[tid];
        ml_part[((split * H + qh) * 8 + int(tid)) * 2 + 1] = st_l[tid];
    }
"""

_WIDE_COMBINE = """
    constexpr int D = 256;
    uint d = thread_position_in_threadgroup.x;
    int r = int(threadgroup_position_in_grid.x);
    int qh = int(threadgroup_position_in_grid.y);
    int L = int(params[1]);
    int n_splits = int(params[3]);
    float M = -INFINITY;
    for (int s = 0; s < n_splits; ++s)
        M = max(M, ml_part[((s * H + qh) * 8 + r) * 2]);
    float total = 0.0f, acc = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        int idx = (s * H + qh) * 8 + r;
        float m = ml_part[idx * 2];
        if (m == -INFINITY) continue;
        float w = exp(m - M);
        total += w * ml_part[idx * 2 + 1];
        acc += w * o_part[idx * D + d];
    }
    out[(qh * L + r) * D + d] = T_(acc / total);
"""

_WIDE_KERNELS: dict = {}
_WIDE_MAX_ROWS = 8
_WIDE_MIN_ROWS = 4


def _wide_kernels():
    import mlx.core as mx
    if not _WIDE_KERNELS:
        _WIDE_KERNELS["partial"] = mx.fast.metal_kernel(
            name="mlx2_research_omlx3958_wide_partial",
            input_names=["q", "k", "v", "params"],
            output_names=["o_part", "ml_part"],
            source=_WIDE_PARTIAL,
            # Keys and values are strided cache views; the kernel reads their
            # strides instead of copying them.
            ensure_row_contiguous=False,
        )
        _WIDE_KERNELS["combine"] = mx.fast.metal_kernel(
            name="mlx2_research_omlx3958_wide_combine",
            input_names=["o_part", "ml_part", "params"],
            output_names=["out"],
            source=_WIDE_COMBINE,
        )
    return _WIDE_KERNELS


def _wide_causal_sdpa(queries, keys, values, scale):
    import mlx.core as mx
    import struct
    """Causal verify attention for up to eight rows at head_dim 256."""
    _, heads, q_len, dim = queries.shape
    kv_heads, kv_len = keys.shape[1], keys.shape[2]
    rows = queries[0]
    if q_len < _WIDE_MAX_ROWS:
        pad = mx.zeros((heads, _WIDE_MAX_ROWS - q_len, dim), dtype=queries.dtype)
        rows = mx.concatenate([rows, pad], axis=1)
    rows = mx.contiguous(rows)
    # About sixteen key splits per head, measured best on M3 Ultra up to 32k
    # keys. Longer caches miss the system cache, where short splits keep the
    # query heads of a KV head on the same keys (-20% at 64k).
    chunk = min(2048, max(512, (-(-kv_len // 16) + 31) // 32 * 32))
    if kv_len > 32768:
        chunk = 512
    n_splits = -(-kv_len // chunk)
    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
    params = mx.array([kv_len, q_len, chunk, n_splits, scale_bits], dtype=mx.uint32)
    kernels = _wide_kernels()
    template = [("T_", queries.dtype), ("G", heads // kv_heads), ("H", heads)]
    o_part, ml_part = kernels["partial"](
        inputs=[rows, keys, values, params],
        template=template,
        grid=(heads * 128, n_splits, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[
            (n_splits, heads, _WIDE_MAX_ROWS, dim),
            (n_splits, heads, _WIDE_MAX_ROWS, 2),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    (out,) = kernels["combine"](
        inputs=[o_part, ml_part, params],
        template=template,
        grid=(q_len * 256, heads, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, heads, q_len, dim)],
        output_dtypes=[queries.dtype],
    )
    return out
