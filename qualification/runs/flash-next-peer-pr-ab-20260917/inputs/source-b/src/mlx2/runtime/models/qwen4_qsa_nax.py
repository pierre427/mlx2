# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import mlx.core as mx

BLOCK = 4
MTILE = 16
HEADER = "\n#include <metal_stdlib>\n#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\n"
SOURCE = "\n    constexpr int NSG = 4;\n    constexpr int DS  = D / NSG;\n    constexpr int NKS = DS / 16;\n    constexpr int NNT = DS / 32;\n\n    // Query length, KV length and ids stride are runtime uniforms, not template\n    // constants: every prefill-chunk width then shares ONE compiled pipeline.\n    const int L   = dims[0];   // query length\n    const int TOT = dims[1];   // KV length (physical_width)\n    const int U   = dims[2];   // per-token ids stride (u_width)\n\n    const ushort sg   = simdgroup_index_in_threadgroup;\n    const ushort lane = thread_index_in_simdgroup;\n    const uint  tok   = threadgroup_position_in_grid.y;   // query token\n    const uint  bkv   = threadgroup_position_in_grid.z;   // b * NKVH + hkv\n    const uint  b     = bkv / NKVH;\n    const uint  hkv   = bkv % NKVH;\n\n    const short qid = lane >> 2;\n    const short fm0 = ((qid & 4) | ((lane >> 1) & 3));    // row = query head\n    const short fn0 = ((qid & 2) | (lane & 1)) * 4;\n\n    threadgroup float sS[NSG * 32 * 16];\n\n    const size_t k_head =\n        (size_t)b * k_strides[0] + (size_t)hkv * k_strides[1];\n    const size_t v_head =\n        (size_t)b * v_strides[0] + (size_t)hkv * v_strides[1];\n\n    const int  lpad      = left_pad[b];\n    const uint tile_base = (b * L + tok) * U;\n    const uint cnt       = counts[b * L + tok];\n    const uint nsel      = n_sel[b * L + tok];\n\n    const int qp       = qpos[b * L + tok];\n    const int complete = ((qp + 1) / BS) * BS;\n\n    // Row r is query head hkv*GQA + r.  Rows GQA..15 are idle.\n    int head_of[2];\n    bool row_live[2];\n    for (short rr = 0; rr < 2; rr++) {\n        int r = fm0 + rr * 8;\n        row_live[rr] = (r < GQA);\n        head_of[rr] = (int)hkv * GQA + r;\n    }\n\n    typedef metal::vec<T, 8> frag_t;\n    frag_t qf[NKS];\n    for (short ks = 0; ks < NKS; ks++) {\n        for (short i = 0; i < 8; i++) {\n            short rr = (i >> 2);\n            int c = sg * DS + ks * 16 + fn0 + (i % 4);\n            qf[ks][i] = row_live[rr]\n                ? q[((size_t)(b * NQH + head_of[rr]) * L + tok) * D + c]\n                : T(0);\n        }\n    }\n\n    float of[NNT][16];\n    for (short t = 0; t < NNT; t++)\n        for (short i = 0; i < 16; i++) of[t][i] = 0.0f;\n    float rmax_r[2] = {-INFINITY, -INFINITY};\n    float rsum_r[2] = {0.0f, 0.0f};\n\n    constexpr auto desc_qk = mpp::tensor_ops::matmul2d_descriptor(\n        16, 32, 16, false, true, true,\n        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);\n    constexpr auto desc_pv = mpp::tensor_ops::matmul2d_descriptor(\n        16, 32, 16, false, false, true,\n        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);\n\n    const float qk_scale = scale[0];\n\n    for (uint u0 = 0; u0 < cnt; u0 += 8) {\n\n        uint physical_rows[4];\n        for (short qq = 0; qq < 4; qq++) {\n            int n = fm0 + qq * 8;\n            uint u = u0 + (uint)(n >> 2);\n            int bid = (u < cnt) ? (int)ids[tile_base + u] : 0;\n            int phys = lpad + bid * BS + (n & 3);\n            physical_rows[qq] = (uint)metal::clamp(phys, 0, (int)TOT - 1);\n        }\n\n        {\n            mpp::tensor_ops::matmul2d<desc_qk, metal::execution_simdgroup> op;\n            auto ca = op.get_left_input_cooperative_tensor<T, T, float>();\n            auto cb = op.get_right_input_cooperative_tensor<T, T, float>();\n            auto cc = op.get_destination_cooperative_tensor<\n                metal::remove_addrspace_t<decltype(ca)>,\n                metal::remove_addrspace_t<decltype(cb)>, float>();\n            for (short i = 0; i < 16; i++) cc[i] = 0.0f;\n            for (short ks = 0; ks < NKS; ks++) {\n                for (short i = 0; i < 8; i++) ca[i] = qf[ks][i];\n                for (short i = 0; i < 8; i++) {\n                    int c = sg * DS + ks * 16 + fn0 + (i % 4);\n                    const uint physical = physical_rows[(i >> 2)];\n                    const uint physical_2 = physical_rows[2 + (i >> 2)];\n                    cb[i] = k[\n                        k_head + (size_t)physical * k_strides[2]\n                        + (size_t)c * k_strides[3]\n                    ];\n                    cb[8 + i] = k[\n                        k_head + (size_t)physical_2 * k_strides[2]\n                        + (size_t)c * k_strides[3]\n                    ];\n                }\n                op.run(ca, cb, cc);\n            }\n            for (short i = 0; i < 16; i++)\n                sS[(sg * 32 + lane) * 16 + i] = cc[i];\n        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n        float s[16];\n        for (short i = 0; i < 16; i++) {\n            s[i] = (sS[(0 * 32 + lane) * 16 + i] + sS[(1 * 32 + lane) * 16 + i]\n                  + sS[(2 * 32 + lane) * 16 + i] + sS[(3 * 32 + lane) * 16 + i])\n                 * qk_scale;\n        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n        // One selection for the whole tile: the mask has NO row dependence.\n        bool colok[8];\n        for (short f = 0; f < 2; f++) {\n            for (short co = 0; co < 4; co++) {\n                int n = f * 16 + fn0 + co;\n                uint u = u0 + (uint)(n >> 2);\n                bool inb = (u < cnt);\n                int bid = inb ? (int)ids[tile_base + u] : 0;\n                int kl = bid * BS + (n & 3);\n                colok[f * 4 + co] = inb && (kl >= 0) && (kl <= qp)\n                                    && ((u < nsel) || (kl >= complete));\n            }\n        }\n        for (short f = 0; f < 2; f++)\n            for (short rr = 0; rr < 2; rr++)\n                for (short co = 0; co < 4; co++)\n                    if (!colok[f * 4 + co]) s[f * 8 + rr * 4 + co] = -INFINITY;\n\n        // The mask is shared across rows; the scores are not.  Max and sum\n        // stay per-row.\n        float rowmax[2];\n        for (short rr = 0; rr < 2; rr++) {\n            float m = -INFINITY;\n            for (short f = 0; f < 2; f++)\n                for (short co = 0; co < 4; co++)\n                    m = metal::max(m, s[f * 8 + rr * 4 + co]);\n            m = metal::max(m, metal::simd_shuffle_xor(m, ushort(1)));\n            m = metal::max(m, metal::simd_shuffle_xor(m, ushort(8)));\n            rowmax[rr] = m;\n        }\n\n        float alpha[2], p[16], newmax[2], newsum[2];\n        for (short rr = 0; rr < 2; rr++) {\n            float mn = metal::max(rmax_r[rr], rowmax[rr]);\n            bool live = mn > -INFINITY;\n            alpha[rr] = live ? metal::fast::exp(rmax_r[rr] - mn) : 1.0f;\n            float sm = 0.0f;\n            for (short f = 0; f < 2; f++) {\n                for (short co = 0; co < 4; co++) {\n                    short idx = f * 8 + rr * 4 + co;\n                    float pv = live ? metal::fast::exp(s[idx] - mn) : 0.0f;\n                    p[idx] = pv;\n                    sm += pv;\n                }\n            }\n            sm += metal::simd_shuffle_xor(sm, ushort(1));\n            sm += metal::simd_shuffle_xor(sm, ushort(8));\n            newmax[rr] = mn;\n            newsum[rr] = rsum_r[rr] * alpha[rr] + sm;\n        }\n        for (short rr = 0; rr < 2; rr++) {\n            rmax_r[rr] = newmax[rr];\n            rsum_r[rr] = newsum[rr];\n        }\n\n        for (short t = 0; t < NNT; t++) {\n            for (short f = 0; f < 2; f++)\n                for (short rr = 0; rr < 2; rr++)\n                    for (short co = 0; co < 4; co++)\n                        of[t][f * 8 + rr * 4 + co] *= alpha[rr];\n\n            int nbase = sg * DS + t * 32;\n            mpp::tensor_ops::matmul2d<desc_pv, metal::execution_simdgroup> op;\n            auto ca = op.get_left_input_cooperative_tensor<T, T, float>();\n            auto cb = op.get_right_input_cooperative_tensor<T, T, float>();\n            auto cc = op.get_destination_cooperative_tensor<\n                metal::remove_addrspace_t<decltype(ca)>,\n                metal::remove_addrspace_t<decltype(cb)>, float>();\n            for (short i = 0; i < 16; i++) cc[i] = of[t][i];\n            for (short ks = 0; ks < 2; ks++) {\n                for (short i = 0; i < 8; i++) ca[i] = (T)p[ks * 8 + i];\n                for (short i = 0; i < 8; i++) {\n                    uint physical = physical_rows[ks * 2 + (i >> 2)];\n                    int co = nbase + fn0 + (i % 4);\n                    cb[i] = v[\n                        v_head + (size_t)physical * v_strides[2]\n                        + (size_t)co * v_strides[3]\n                    ];\n                    cb[8 + i] = v[\n                        v_head + (size_t)physical * v_strides[2]\n                        + (size_t)(co + 16) * v_strides[3]\n                    ];\n                }\n                op.run(ca, cb, cc);\n            }\n            for (short i = 0; i < 16; i++) of[t][i] = cc[i];\n        }\n    }\n\n    for (short t = 0; t < NNT; t++) {\n        for (short f = 0; f < 2; f++) {\n            for (short rr = 0; rr < 2; rr++) {\n                if (!row_live[rr]) continue;\n                float inv = (rsum_r[rr] > 0.0f) ? (1.0f / rsum_r[rr]) : 0.0f;\n                for (short co = 0; co < 4; co++) {\n                    int c = sg * DS + t * 32 + f * 16 + fn0 + co;\n                    out[((size_t)(b * NQH + head_of[rr]) * L + tok) * D + c] =\n                        of[t][f * 8 + rr * 4 + co] * inv;\n                }\n            }\n        }\n    }\n"
_KERNEL = mx.fast.metal_kernel(
    name="nax_qsa_attn_b",
    input_names=[
        "q",
        "k",
        "v",
        "ids",
        "counts",
        "n_sel",
        "qpos",
        "left_pad",
        "scale",
        "dims",
    ],
    output_names=["out"],
    header=HEADER,
    source=SOURCE,
    ensure_row_contiguous=False,
)


def block_sparse_layout_supported(head_dim, n_heads, n_kv_heads, block_size):
    """Whether the head-tiled NAX kernel can serve this attention geometry.

    Constraints come from the tile shape, not the KV length: the D axis is
    split across NSG=4 simdgroups then 16/32-wide NAX tiles (so DS = D/4 must
    be a multiple of 32), one KV head's query heads must fit a 16-row tile,
    and the block size is fixed at 4 (the QSA compression ratio).
    """
    if n_kv_heads <= 0 or n_heads % n_kv_heads:
        return False
    ds = head_dim // 4
    return (
        head_dim % 4 == 0
        and ds % 32 == 0
        and (block_size == BLOCK)
        and (n_heads // n_kv_heads <= MTILE)
    )


_NAX_AVAILABLE = None


def nax_kernel_available():
    """True iff the MPP/NAX kernel is validated and runs on this device.

    The minimal zero probe only establishes compilation: on M3 Pro the kernel
    compiles and returns finite zeros but corrupts nonzero production geometry.
    Keep every caller (including the direct-decode and stage-one experimental
    routes) inside the M5 envelope that passed the dense numerical oracle.
    Within that envelope the memoized probe still catches missing Metal 4 / MPP
    support so callers fall back instead of crashing a forward pass.
    """
    global _NAX_AVAILABLE
    if _NAX_AVAILABLE is not None:
        return _NAX_AVAILABLE
    if not mx.metal.is_available():
        _NAX_AVAILABLE = False
        return False
    try:
        if "M5" not in str(mx.device_info().get("device_name", "")):
            _NAX_AVAILABLE = False
            return False
    except (AttributeError, RuntimeError, TypeError):
        _NAX_AVAILABLE = False
        return False
    try:
        (total, dim) = (8, 256)
        q = mx.zeros((1, 2, 2, dim), dtype=mx.bfloat16)
        k = mx.zeros((1, 1, total, dim), dtype=mx.bfloat16)
        v = mx.zeros((1, 1, total, dim), dtype=mx.bfloat16)
        ids = mx.zeros((1, 2, 8), dtype=mx.uint32)
        counts = mx.ones((1, 2), dtype=mx.uint32)
        n_sel = mx.zeros((1, 2), dtype=mx.uint32)
        qpos = mx.array([[0, 1]], dtype=mx.int32)
        left_pad = mx.zeros((1,), dtype=mx.int32)
        out = nax_qsa_attention(
            q,
            k,
            v,
            ids,
            counts,
            n_sel,
            qpos,
            left_pad,
            scale=1.0 / 16.0,
            u_width=8,
            total=total,
            n_kv_heads=1,
        )
        mx.eval(out)
        _NAX_AVAILABLE = True
    except Exception:
        _NAX_AVAILABLE = False
    return _NAX_AVAILABLE


def compact_blocks_to_kernel_inputs(cb):
    """Turn a ``QSACompactBlocks`` into the kernel's per-token block list.

    ``cb.block_ids`` is the sorted, prefix-packed set of genuinely SELECTED
    logical blocks (``cb.block_counts`` long, suffix zeroed).  The kernel also
    needs the incomplete TAIL block appended at slot ``n_sel``; a slot at or
    past ``n_sel`` is admitted only by ``kl >= complete`` (the causal tail),
    never by membership.  The tail block is ``q_pos // block_size`` and is
    dropped from ``counts`` when it duplicates the last selected block
    (``q_pos == block_size - 1 mod block_size`` with that block selected), so
    the block list stays a set.

    Returns ``(ids, counts, n_sel, u_width, q_pos, left_pad, total)`` ready for
    ``nax_qsa_attention``.  The width is derived from the static block-id
    width, so this never syncs the device.
    """
    block_ids = cb.block_ids.astype(mx.int32)
    n_sel = cb.block_counts.astype(mx.int32)
    bs = cb.block_size
    total = cb.physical_width
    (batch, length, k_width) = block_ids.shape
    n_ext = -(-total // bs)
    q_pos = (cb.tail_stop - 1).astype(mx.int32)
    if q_pos.shape != (batch, length):
        q_pos = mx.broadcast_to(q_pos, (batch, length))
    tail = mx.clip(q_pos // bs, 0, n_ext - 1)
    last = mx.take_along_axis(
        block_ids, mx.clip(n_sel - 1, 0, k_width - 1)[..., None], axis=-1
    )[..., 0]
    dup = (n_sel > 0) & (last == tail)
    counts = n_sel + mx.where(dup, 0, 1)
    u_width = max(8, (k_width + 1 + 7) // 8 * 8)
    pad = u_width - k_width
    ids_full = mx.concatenate(
        [block_ids, mx.full((batch, length, pad), n_ext, dtype=mx.int32)], axis=-1
    )
    ids_full = mx.put_along_axis(ids_full, n_sel[..., None], tail[..., None], axis=-1)
    ids = mx.where(ids_full < n_ext, ids_full, 0).astype(mx.uint32)
    left_pad = cb.left_padding
    if left_pad is None:
        left_pad = mx.zeros((batch,), dtype=mx.int32)
    else:
        left_pad = left_pad.astype(mx.int32)
    return (
        ids,
        counts.astype(mx.uint32),
        n_sel.astype(mx.uint32),
        u_width,
        q_pos,
        left_pad,
        total,
    )


def compact_token_validity(cb):
    """Return compact token coordinates and the shared validity predicate."""
    (ids, counts, n_sel, u_width, q_pos, left_pad, total) = (
        compact_blocks_to_kernel_inputs(cb)
    )
    block_size = int(cb.block_size)
    logical = ids.astype(mx.int32)[..., None] * block_size + mx.arange(
        block_size, dtype=mx.int32
    )
    slots = mx.arange(u_width, dtype=mx.int32)[None, None, :, None]
    present = slots < counts.astype(mx.int32)[..., None, None]
    selected = slots < n_sel.astype(mx.int32)[..., None, None]
    tail = (logical >= cb.tail_start[..., None, None]) & (
        logical < cb.tail_stop[..., None, None]
    )
    valid = present & (selected | tail)
    physical = logical + left_pad[:, None, None, None]
    valid = valid & (physical >= 0) & (physical < total)
    valid = valid & (logical <= q_pos[..., None, None])
    physical = mx.clip(physical, 0, total - 1)
    if cb.causal_mask is not None:
        (batch, length) = ids.shape[:2]
        causal = mx.broadcast_to(cb.causal_mask, (batch, 1, length, total))[:, 0]
        gathered = mx.take_along_axis(
            causal, physical.reshape(batch, length, -1), axis=-1
        ).reshape(physical.shape)
        valid = valid & gathered
    return (ids, counts, n_sel, u_width, q_pos, left_pad, total, physical, valid)


def nax_qsa_attention(
    q, k, v, ids, counts, n_sel, q_pos, left_pad, *, scale, u_width, total, n_kv_heads
):
    """Block-sparse QSA attention on NAX.  q [B,H,L,D], k/v [B,HKV,T,D] -> fp32.

    ``ids``/``counts``/``n_sel``/``q_pos``/``left_pad`` come from
    ``compact_blocks_to_kernel_inputs``.  Output is fp32 [B,H,L,D]; the caller
    casts to the attention dtype.
    """
    (batch, nqh, length, dim) = q.shape
    gqa = nqh // n_kv_heads
    assert gqa <= MTILE, "one token's heads must fit a 16-row NAX tile"
    (out,) = _KERNEL(
        inputs=[
            mx.contiguous(q),
            k,
            v,
            mx.contiguous(ids.astype(mx.uint32)),
            mx.contiguous(counts.astype(mx.uint32)),
            mx.contiguous(n_sel.astype(mx.uint32)),
            mx.contiguous(mx.broadcast_to(q_pos, (batch, length)).astype(mx.int32)),
            mx.contiguous(left_pad.astype(mx.int32)),
            mx.array([scale], dtype=mx.float32),
            mx.array([length, total, u_width], dtype=mx.int32),
        ],
        template=[
            ("T", q.dtype),
            ("D", dim),
            ("NQH", nqh),
            ("NKVH", n_kv_heads),
            ("GQA", gqa),
            ("BS", BLOCK),
        ],
        grid=(128, length, batch * n_kv_heads),
        threadgroup=(128, 1, 1),
        output_shapes=[(batch, nqh, length, dim)],
        output_dtypes=[mx.float32],
    )
    return out
