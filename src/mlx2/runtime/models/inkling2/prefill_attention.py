# SPDX-License-Identifier: Apache-2.0
"""Opt-in Inkling 2 prefill attention paths (all default off, unqualified).

Original mlx2 code (2026-11-22); see docs/PROVENANCE.md.

``inkling2_prefill_mask_with_sink`` (``banded_mask_with_sink``): the composed
prefill path appends each query's sink logit to the dense relative-bias mask
with a second full-width concatenation (one more write and read of the whole
mask per slice). Here the sink is column ``size`` of the same gather: the
bias table gets the sink logit as an extra entry ``E`` and the sink column's
gather index points at it, so the mask comes out of one gather plus the same
selects. The values are the reference's, copied, so the mask and the SDPA
result are bit-identical.
"""

import mlx.core as mx

from . import fuse as _fuse

NEG = -1e30


def banded_mask_with_sink(attn, rel, proj, sinks, q_offset, size, sliding, rel_extent,
                          left_padding=None):
    """``concat([banded_additive_mask(...), sinks], -1)`` in one gather.

    rel: [B, LQ, H, d_rel]; proj: [d_rel, E]; sinks: [B, H, LQ, 1] (the
    sink logits). Returns [B, H, LQ, size + 1] in rel's dtype. Index
    tensors are memoized per forward like ``fuse.banded_mask``'s (an empty
    memo when no fusion holder is set, so they are rebuilt per call)."""
    b, lq, h, _ = rel.shape
    dtype = rel.dtype
    size, extent, q_offset = int(size), int(rel_extent), int(q_offset)
    key = ("mask_sink", lq, size, q_offset, int(sliding), extent)
    memo = _fuse.forward_memo(attn)
    held = memo.get(key)
    if held is None:
        dist = (mx.arange(lq) + q_offset)[:, None] - mx.arange(size)[None, :]
        sink_col = mx.full((lq, 1), extent, dtype=dist.dtype)
        gidx = mx.concatenate([mx.clip(dist, 0, extent - 1), sink_col], axis=1)
        no = mx.zeros((lq, 1), dtype=mx.bool_)
        far = mx.concatenate([dist >= extent, no], axis=1)
        masked = dist < 0
        if sliding > 0:
            masked = masked | (dist >= sliding)
        masked = mx.concatenate([masked, no], axis=1)
        held = memo[key] = (gidx[None, None], far[None, None], masked[None, None])
        _fuse.count(attn, "mask_sink_index_built")
    gidx, far, masked = held
    table = mx.concatenate([(rel @ proj).transpose(0, 2, 1, 3), sinks.astype(dtype)], axis=-1)
    pb = mx.take_along_axis(table, mx.broadcast_to(gidx, (b, h, lq, size + 1)), axis=-1)
    mask = _fuse._mask_select(pb, far, masked, mx.array(0.0, dtype), mx.array(NEG, dtype))
    if left_padding is not None:
        pkey = ("pad_sink", size, id(left_padding))
        valid = memo.get(pkey)
        if valid is None or valid[0] is not left_padding:
            keys = mx.arange(size + 1)[None, :]
            ok = (keys >= left_padding[:, None]) | (keys == size)
            valid = memo[pkey] = (left_padding, ok[:, None, None, :])
        mask = _fuse._mask_pad(mask, valid[1], mx.array(NEG, dtype))
    return mask.astype(dtype)


# ----- inkling2_prefill_tiled_attention: GQA-packed tiled flash prefill -----
#
# Original mlx2 kernel (2026-11-22). Mechanism references, no code copied:
# jundot/omlx PR 5526 / ``flash_banded_attention`` (Apache-2.0: half-staged
# Q/K/V tiles from BF16 bit patterns, scores in the log2 domain, the online
# softmax started from the sink logit, transposed K tile padded against bank
# conflicts) and mlx2's own split-K port (splitk.py: GQA row packing). New
# here: prefill-sized query tiles whose packed rows interleave the G query
# heads of one KV head (packed row = position * G + member), so a tile's
# rows span only PR / G query positions (tight causal key range) while every
# K/V tile is read once for all G heads; a bias-free fast path for key tiles
# entirely beyond the relative extent; per-simdgroup causal tile skipping.
#
# score(i, j) = scale * q_i . k_j + bias_i[d]   (bias only for d < E)
# d = OFF + i - j; masked when d < 0, j < left_pad, or WINDOW > 0 and
# d >= WINDOW. The sink is a zero-value key with logit sink_i: the softmax
# starts at m = sink_i, l = 1.
_TILED_SOURCE = r"""
    constexpr int G = HQ / HKV;
    constexpr int NT = NSG * 32;
    constexpr int PR = NSG * 8;
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int qt = threadgroup_position_in_grid.x;
    const int hkv = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int L = dims[0];
    const int S = dims[1];
    const int OFF = dims[2];
    const int lp = max(left_pad[b], 0);
    const int n_rows = L * G;

    // 8x8 fragment layout: this lane holds row fm, columns fn and fn + 1.
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int sg_first = qt * PR + int(sg) * 8;
    const int prow = sg_first + fm;
    const bool row_ok = prow < n_rows;
    const int q_pos = row_ok ? prow / G : 0;
    const int h = hkv * G + (row_ok ? prow % G : 0);
    const int a_row = OFF + q_pos;
    const float qscale = scale[0] * LOG2E;

    // Key range of the whole tile (uniform across the threadgroup).
    const int first_pos = (qt * PR) / G;
    const int last_pos = min((qt * PR + PR - 1) / G, L - 1);
    const int a_first = OFF + first_pos;
    const int a_last = min(OFF + last_pos, S - 1);
    int lo = WINDOW > 0 ? max(0, a_first - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int k_begin = lo / BK * BK;
    // This simdgroup's last query position: key tiles past its diagonal
    // are fully masked for its rows and skipped.
    const bool sg_live = sg_first < n_rows;
    const int sg_last = OFF + min((sg_first + 7) / G, L - 1);

    threadgroup ST Ks[D * LDK];
    threadgroup ST Vs[BK * LDV];

    simdgroup_matrix<ST, 8, 8> Qf[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            const device T* qp = q + b * q_strides[0] + h * q_strides[1]
                + q_pos * q_strides[2] + (db * 8 + fn) * q_strides[3];
            qv = float2(float(qp[0]), float(qp[q_strides[3]]));
        }
        Qf[db].thread_elements()[0] = ST(qv.x);
        Qf[db].thread_elements()[1] = ST(qv.y);
    }
    simdgroup_float8x8 Of[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    const size_t row_index = ((size_t)b * L + q_pos) * HQ + h;
    float m = row_ok ? sinks[row_index] * LOG2E : 0.0f;
    float l = 1.0f;
    const device T* relp = rel + row_index * E;

    const device T* kb_ptr = k + b * k_strides[0] + hkv * k_strides[1];
    const device T* vb_ptr = v + b * v_strides[0] + hkv * v_strides[1];
    const bool vec_kv = VEC_OK
        && k_strides[3] == 1 && v_strides[3] == 1
        && (k_strides[2] % 8 == 0) && (v_strides[2] % 8 == 0)
        && ((reinterpret_cast<ulong>(kb_ptr) & 15ul) == 0)
        && ((reinterpret_cast<ulong>(vb_ptr) & 15ul) == 0);

    for (int k0 = k_begin; k0 <= a_last; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vec_kv) {
            // Eight BF16 values (16 bytes) per load; a BF16 is the high half
            // of an FP32, so the widening is a shift (exact), then to half.
            for (int c = int(tid); c < BK * D / 8; c += NT) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const int key = k0 + r;
                uint4 kw = uint4(0u);
                uint4 vw = uint4(0u);
                if (key <= a_last) {
                    kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);
                    vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);
                }
                threadgroup ST* kd = Ks + col * LDK + r;
                threadgroup ST* vd = Vs + r * LDV + col;
                for (int j = 0; j < 4; ++j) {
                    kd[(2 * j) * LDK] = ST(as_type<float>(kw[j] << 16));
                    kd[(2 * j + 1) * LDK] = ST(as_type<float>(kw[j] & 0xFFFF0000u));
                    vd[2 * j] = ST(as_type<float>(vw[j] << 16));
                    vd[2 * j + 1] = ST(as_type<float>(vw[j] & 0xFFFF0000u));
                }
            }
        } else {
            for (int c = int(tid); c < BK * D; c += NT) {
                const int r = c / D;
                const int col = c % D;
                const int key = k0 + r;
                float kv = 0.0f;
                float vv = 0.0f;
                if (key <= a_last) {
                    kv = float(kb_ptr[key * k_strides[2] + col * k_strides[3]]);
                    vv = float(vb_ptr[key * v_strides[2] + col * v_strides[3]]);
                }
                Ks[col * LDK + r] = ST(kv);
                Vs[r * LDV + col] = ST(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!sg_live || k0 > sg_last) {
            continue;
        }

        simdgroup_float8x8 Sf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_matrix<ST, 8, 8> Kt;
                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }
        float rmax = -INFINITY;
        if (WINDOW == 0 && k0 >= lp && k0 + BK - 1 <= a_first - E && k0 + BK - 1 <= a_last) {
            // Every key of the tile is visible to every row of the tile and
            // past the bias band: scale only.
            for (int kb = 0; kb < BK / 8; ++kb) {
                for (int e = 0; e < 2; ++e) {
                    const float s = row_ok ? Sf[kb].thread_elements()[e] * qscale : -INFINITY;
                    Sf[kb].thread_elements()[e] = s;
                    rmax = max(rmax, s);
                }
            }
        } else {
            for (int kb = 0; kb < BK / 8; ++kb) {
                for (int e = 0; e < 2; ++e) {
                    const int key = k0 + kb * 8 + fn + e;
                    const int d = a_row - key;
                    const bool vis = row_ok && key <= a_last && d >= 0 && key >= lp
                        && (WINDOW == 0 || d < WINDOW);
                    float s = -INFINITY;
                    if (vis) {
                        s = Sf[kb].thread_elements()[e] * qscale;
                        if (d < E) {
                            s += float(relp[d]) * LOG2E;
                        }
                    }
                    Sf[kb].thread_elements()[e] = s;
                    rmax = max(rmax, s);
                }
            }
        }
        // The four lanes of a fragment row differ in lane bits 0 and 3.
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        // m starts at the sink logit; guard a -inf sink over a masked tile.
        const float m_new = max(m, rmax);
        const float factor = (m_new == -INFINITY) ? 1.0f : fast::exp2(m - m_new);
        float rsum = 0.0f;
        simdgroup_matrix<ST, 8, 8> Pf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float sv = Sf[kb].thread_elements()[e];
                const float p = (sv == -INFINITY) ? 0.0f : fast::exp2(sv - m_new);
                Pf[kb].thread_elements()[e] = ST(p);
                rsum += p;
            }
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * factor + rsum;
        m = m_new;
        // LAZY_RESCALE: a row whose running max did not move has factor
        // 1.0f, and x * 1.0f == x bit for bit for finite, infinite, -0 and
        // NaN x (only a subnormal accumulator under fast-math flushing could
        // differ, which needs a max jump of > 126 in log2 units). l keeps
        // its own update. Per thread: the accumulators are its registers.
        if (!LAZY_RESCALE || factor != 1.0f) {
            for (int db = 0; db < D / 8; ++db) {
                Of[db].thread_elements()[0] *= factor;
                Of[db].thread_elements()[1] *= factor;
            }
        }
        for (int db = 0; db < D / 8; ++db) {
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_matrix<ST, 8, 8> Vm;
                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vm, Of[db]);
            }
        }
    }
    if (row_ok) {
        const float inv = 1.0f / l;
        device T* op = out + (((size_t)b * HQ + h) * L + q_pos) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(Of[db].thread_elements()[0] * inv);
            op[db * 8 + fn + 1] = T(Of[db].thread_elements()[1] * inv);
        }
    }
"""
# Simdgroups per threadgroup (packed rows = 8 * NSG). 4 (128 threads) is the
# thread count the GPU-tested split-K kernel runs with the same per-thread
# state; 8 halves the K/V tile passes again if the pipeline admits 256
# threads (the Metal timing test records both).
TILED_NSG = 4
# ``inkling2_prefill_lazy_rescale``: skip the O-accumulator rescale for rows
# whose running max did not move (factor exactly 1). Bitwise identical by
# construction in practice (see the kernel comment for the subnormal caveat);
# idea from gufo-org/gufo#1420 (reimplemented, no code copied).
# Default off until its A/B is reviewed. Serving: the adapter option
# ``inkling2_prefill_lazy_rescale`` (per call); ``set_lazy_rescale`` is the
# process-wide research switch the probes use.
TILED_LAZY_RESCALE = False
_tiled_kernel = None
LOG2E = 1.4426950408889634


def set_lazy_rescale(enabled):
    """Switch ``inkling2_prefill_lazy_rescale`` process-wide (bitwise-neutral)."""
    global TILED_LAZY_RESCALE
    TILED_LAZY_RESCALE = bool(enabled)


def tiled_block_keys(dtype):
    """Key-tile width: 32 with half staging (BF16 inputs; 19 KB of tiles at
    D = 128), 16 with FP32 staging (21 KB)."""
    return 16 if dtype == mx.float32 else 32


def decline_reason(q, k, v, *, n_heads, n_kv, head_dim, device=None):
    """None when ``tiled_prefill_attention`` can run these arrays, else why."""
    if q.shape[2] < 2:
        return "rows"
    if (mx.default_device() if device is None else device) != mx.gpu:
        return "cpu"
    if not (isinstance(k, mx.array) and isinstance(v, mx.array)):
        return "cache_format"
    if q.dtype not in (mx.bfloat16, mx.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        return "dtype"
    if head_dim % 8 or head_dim > 128 or n_heads % n_kv:
        return "geometry"
    return None


def tiled_prefill_attention(q, k, v, rel, sinks, *, offset, left_padding, window, scale,
                            lazy_rescale=None):
    """Prefill attention with Inkling's relative bias, window, padding and sink.

    q: [B, H, L, D]; k, v: [B, Hk, S, D] (any strides); rel: [B, L, H, E],
    the per-query bias at distances below E; sinks: [B, L, H]; queries at
    key positions ``offset + i``. Returns [B, H, L, D] in q's dtype. GPU
    only; check ``decline_reason`` first. Near-exact: BF16 inputs stage in
    FP16 (Q/K L2-clipped, V RMS-normalized, so in range), FP32 accumulation
    and online softmax; the reduction order differs from SDPA."""
    global _tiled_kernel
    if _tiled_kernel is None:
        _tiled_kernel = mx.fast.metal_kernel(
            name="inkling2_prefill_tiled_attention",
            input_names=["q", "k", "v", "rel", "sinks", "left_pad", "dims", "scale"],
            output_names=["out"],
            source=_TILED_SOURCE,
            ensure_row_contiguous=False,
        )
    b, hq, length, dim = q.shape
    hkv, size = k.shape[1], k.shape[2]
    g = hq // hkv
    nsg = min(TILED_NSG, max(1, (length * g + 7) // 8))
    tiles = (length * g + nsg * 8 - 1) // (nsg * 8)
    pad = (mx.zeros((b,), mx.int32) if left_padding is None
           else mx.maximum(left_padding, 0).astype(mx.int32))
    half = q.dtype == mx.bfloat16
    return _tiled_kernel(
        inputs=[q, k, v, mx.contiguous(rel.astype(q.dtype)),
                mx.contiguous(sinks.astype(mx.float32)), pad,
                mx.array([length, size, int(offset)], dtype=mx.int32),
                mx.array([float(scale)], dtype=mx.float32)],
        template=[("T", q.dtype), ("ST", mx.float16 if half else mx.float32), ("D", dim),
                  ("HQ", hq), ("HKV", hkv), ("E", rel.shape[-1]), ("WINDOW", int(window)),
                  ("NSG", nsg), ("BK", tiled_block_keys(q.dtype)), ("VEC_OK", int(half)),
                  ("LAZY_RESCALE", int(TILED_LAZY_RESCALE if lazy_rescale is None
                                       else bool(lazy_rescale)))],
        grid=(tiles * nsg * 32, hkv, b),
        threadgroup=(nsg * 32, 1, 1),
        output_shapes=[(b, hq, length, dim)],
        output_dtypes=[q.dtype],
    )[0]


def tiled_reference(q, k, v, rel, sinks, *, offset, left_padding, window, scale, bk=None):
    """The kernel's algorithm in MLX ops (CPU tests, A/B debugging).

    Same arguments and result as ``tiled_prefill_attention``: key tiles of
    ``bk`` at multiples of ``bk`` in order, the online softmax in the log2
    domain from the sink, FP16 staging of Q/K/V and the probabilities for
    BF16 inputs, FP32 accumulation. Tiles the kernel skips (outside the
    window, before the padding, past a simdgroup's diagonal) are fully
    masked for those rows, and a fully masked tile leaves (m, l, acc)
    unchanged, so processing every tile is the same law."""
    b, hq, length, dim = q.shape
    hkv, size = k.shape[1], k.shape[2]
    g = hq // hkv
    bk = bk or tiled_block_keys(q.dtype)
    stage = mx.float16 if q.dtype == mx.bfloat16 else mx.float32

    def staged(x):
        return x.astype(stage).astype(mx.float32)

    qf = staged(q)
    kf = staged(mx.repeat(k, g, axis=1))
    vf = staged(mx.repeat(v, g, axis=1))
    extent = rel.shape[-1]
    bias = rel.astype(q.dtype).astype(mx.float32).transpose(0, 2, 1, 3) * LOG2E  # [B,H,L,E]
    m = sinks.astype(mx.float32).transpose(0, 2, 1)[..., None] * LOG2E  # [B,H,L,1]
    l = mx.ones_like(m)
    acc = mx.zeros((b, hq, length, dim), mx.float32)
    rows = mx.arange(length)[:, None] + int(offset)
    pad = (mx.zeros((b,), mx.int32) if left_padding is None
           else mx.maximum(left_padding, 0).astype(mx.int32))
    last = min(int(offset) + length, size)
    for k0 in range(0, last, bk):
        k1 = min(k0 + bk, last)
        keys = mx.arange(k0, k1)[None, :]
        d = rows - keys  # [L, bk]
        vis = d >= 0
        if window:
            vis = vis & (d < int(window))
        vis = vis[None, None] & (keys[None, None] >= pad[:, None, None, None])
        s = (qf @ kf[:, :, k0:k1].transpose(0, 1, 3, 2)) * (float(scale) * LOG2E)
        gidx = mx.broadcast_to(mx.clip(d, 0, extent - 1)[None, None], s.shape)
        s = s + mx.where((d < extent)[None, None], mx.take_along_axis(bias, gidx, axis=-1), 0.0)
        s = mx.where(vis, s, -mx.inf)
        m_new = mx.maximum(m, s.max(axis=-1, keepdims=True))
        factor = mx.power(2.0, m - m_new)
        p = mx.where(vis, mx.power(2.0, s - m_new), 0.0)
        l = l * factor + p.sum(axis=-1, keepdims=True)
        acc = acc * factor + staged(p) @ vf[:, :, k0:k1]
        m = m_new
    return (acc / l).astype(q.dtype)
