# SPDX-License-Identifier: MIT
"""Fused QSA block selection: relu head-sum, causal mask, top-k, compaction.

The unfused selection runs per-head matmuls, several elementwise passes over
an ``[L, n_blocks]`` fp32 score tensor, ``argpartition`` over up to 64K blocks
per query row, then an argsort to compact the chosen ids.  Here one batched
GEMM produces per-head dot products ``[H, L, n_blocks]`` and one kernel per
query row does the rest:

1. ``score = sum_h relu(dot_h)`` over the row's valid blocks only (block ``j``
   is valid iff ``4j + 3 <= q_pos``, a prefix ``j < (q_pos + 1) // 4``), stored
   as an order-preserving key ``bits(score) + 1`` (scores are >= 0);
2. an 8-bit radix select over the keys finds the k-th largest key ``T`` and
   how many keys equal to ``T`` are still needed;
3. a tiled scan emits every key above ``T`` and the first needed keys equal
   to ``T``, in ascending block order.

The output is already the sorted, prefix-packed id list ``QSACompactBlocks``
wants.  Ties at the threshold resolve to the lowest block ids; the reference
``topk``/``argpartition`` resolve them arbitrarily, so selections agree on
every block above the threshold and may differ only among exact ties.
"""

from __future__ import annotations

import mlx.core as mx

TG = 1024

_SOURCE = """
    const uint row  = threadgroup_position_in_grid.x;
    const uint tid  = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg   = simdgroup_index_in_threadgroup;
    const uint ROWS = dims[0];
    const uint NB   = dims[1];
    const uint K    = dims[2];

    threadgroup atomic_uint hist[256];
    threadgroup uint shared_sel[2];
    threadgroup uint scan_tot[TG / 32 + 1];

    device uint* out_ids = ids + (size_t)row * K;
    const int qp = qpos[row];
    const int nv_i = (qp + 1) / 4;
    const uint nv = (uint)metal::clamp(nv_i, 0, (int)NB);

    if (nv <= K) {
        // Every valid block is selected: ids 0..nv-1, zero padding.
        for (uint j = tid; j < K; j += TG) out_ids[j] = (j < nv) ? j : 0u;
        if (tid == 0) counts[row] = nv;
        return;
    }

    for (uint i = tid; i < 256; i += TG)
        atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 0: fused relu head-sum into order-preserving keys, top-byte histogram.
    // Each thread later re-reads only the indices it wrote (same stride).
    const size_t head_stride = (size_t)ROWS * NB;
    const size_t row_base = (size_t)row * NB;
    device uint* krow = keys + row_base;
    for (uint j = tid; j < nv; j += TG) {
        float s = 0.0f;
        for (int h = 0; h < H; h++) s += metal::max(S[h * head_stride + row_base + j], 0.0f);
        uint key = as_type<uint>(s) + 1u;
        krow[j] = key;
        atomic_fetch_add_explicit(&hist[key >> 24], 1u, memory_order_relaxed);
    }

    uint prefix = 0u, pmask = 0u, remaining = K;
    for (int pass = 0; pass < 4; pass++) {
        const uint shift = 24u - 8u * (uint)pass;
        if (pass > 0) {
            for (uint i = tid; i < 256; i += TG)
                atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint j = tid; j < nv; j += TG) {
                uint key = krow[j];
                if ((key & pmask) == prefix)
                    atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint cum = 0u, digit = 0u;
            for (int b = 255; b >= 0; b--) {
                uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (cum + c >= remaining) { digit = (uint)b; break; }
                cum += c;
            }
            shared_sel[0] = digit;
            shared_sel[1] = remaining - cum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= shared_sel[0] << shift;
        pmask |= 255u << shift;
        remaining = shared_sel[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint T = prefix;
    const uint need_eq = remaining;

    // Ordered emission, one TG-wide tile at a time.  One packed scan carries
    // the above-threshold count (low 16 bits) and the at-threshold count.
    uint out_run = 0u, eq_run = 0u;
    for (uint base = 0; base < nv; base += TG) {
        const uint j = base + tid;
        const uint key = (j < nv) ? krow[j] : 0u;
        const uint gt = (key > T) ? 1u : 0u;
        const uint eq = (j < nv && key == T) ? 1u : 0u;
        const uint v = gt | (eq << 16);
        const uint incl = simd_prefix_inclusive_sum(v);
        if (lane == 31) scan_tot[sg] = incl;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            const uint t = (lane < TG / 32) ? scan_tot[lane] : 0u;
            const uint e = simd_prefix_exclusive_sum(t);
            if (lane < TG / 32) scan_tot[lane] = e;
            if (lane == TG / 32 - 1) scan_tot[TG / 32] = e + t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint excl = scan_tot[sg] + incl - v;
        const uint total = scan_tot[TG / 32];
        const uint gt_ex = excl & 0xffffu, eq_ex = excl >> 16;
        const uint eq_avail = (need_eq > eq_run) ? need_eq - eq_run : 0u;
        if (gt || (eq && eq_ex < eq_avail))
            out_ids[out_run + gt_ex + metal::min(eq_ex, eq_avail)] = j;
        out_run += (total & 0xffffu) + metal::min(total >> 16, eq_avail);
        eq_run += total >> 16;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) counts[row] = K;
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="qsa_fused_select",
            input_names=["S", "qpos", "dims"],
            output_names=["ids", "counts", "keys"],
            source=_SOURCE,
        )
    return _KERNEL


def fused_select(queries: mx.array, blocks: mx.array, q_pos: mx.array, topk: int,
                 *, q_chunk: int = 512):
    """Top-``topk`` blocks per query, sorted ascending and prefix-packed.

    ``queries`` [H, L, D] and ``blocks`` [n_blocks, D] (float32); ``q_pos`` [L]
    int32 absolute query positions.  Returns ``(ids [L, topk] int32,
    counts [L] int32)``; ``ids[l, :counts[l]]`` are the selected blocks.
    """
    (heads, length, _) = queries.shape
    n_blocks = int(blocks.shape[0])
    kernel = _kernel()
    kb_t = blocks.T
    ids_parts, count_parts = [], []
    for start in range(0, length, q_chunk):
        rows = min(q_chunk, length - start)
        dots = queries[:, start:start + rows] @ kb_t  # [H, rows, n_blocks]
        (ids, counts, _) = kernel(
            inputs=[dots, q_pos[start:start + rows].astype(mx.int32),
                    mx.array([rows, n_blocks, topk], dtype=mx.uint32)],
            template=[("H", heads), ("TG", TG)],
            grid=(TG * rows, 1, 1),
            threadgroup=(TG, 1, 1),
            output_shapes=[(rows, topk), (rows,), (rows, n_blocks)],
            output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
        )
        ids_parts.append(ids)
        count_parts.append(counts)
    ids = ids_parts[0] if len(ids_parts) == 1 else mx.concatenate(ids_parts)
    counts = count_parts[0] if len(count_parts) == 1 else mx.concatenate(count_parts)
    return ids.astype(mx.int32), counts.astype(mx.int32)
