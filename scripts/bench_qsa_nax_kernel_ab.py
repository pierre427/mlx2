#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Research microbench. The A2 variant kernel's 16-byte row loads with a shared
# Q/K dim permutation, its 16x32x32 cooperative-tensor layouts and the fp16
# hi+lo P packing follow jundot/omlx #5084 at
# b0541976f5e5031c8ecd4c728f5a5810b6af4fdf,
# omlx/patches/mlx_vlm_minimax_m3_compat/vendor/mlx_vlm/models/minimax_m3_vl/msa_nax.py
# (Apache-2.0). Everything else (grid, masks, ids ABI, 4-simdgroup D split,
# softmax) is mlx2's qwen4_qsa_nax kernel. Not a serving route.
"""Kernel-only A/B of the NAX block-sparse QSA prefill attention.

Arms (synthetic bf16 q/k/v, synthetic per-token top-512 block selections):

* ``A0c`` -- ``compact_blocks_to_kernel_inputs`` alone (MLX ops);
* ``A0k`` -- today's ``nax_qsa_attention`` on precomputed inputs;
* ``A1k`` -- today's kernel with each row's blocks replaced by its most
  recent ``n_sel`` complete blocks (contiguous: the gather-locality ceiling);
* ``A2k`` -- variant: vec8 row loads, 16x32x32 QK and PV ops, P as bf16;
* ``A2h`` -- variant: as A2k with P as fp16 hi + lo (PV K-width doubled).

Per arm: median ms over interleaved reps, TFLOP/s (4 * rows * heads * keys *
D), error against a float64 reference on the bf16 operands for sampled rows,
and bitwise chunk invariance (one launch vs four row slices).

Usage: ``bench_qsa_nax_kernel_ab.py owned OUT_DIR`` (CPG lease + paired GPU
locks) or ``bench_qsa_nax_kernel_ab.py unit OUT_DIR [--quick]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TOPK = 512
LOCAL_BLOCKS = 64
BLOCK = 4
D = 256
HEADS = 24
GEOMETRIES = {
    # Synthetic tensors with the attention geometry of these checkpoints.
    "fn": {"kv_heads": 2, "geometry_of":
           "Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP@sha256:25c2da42836b90dd13fc63e1b0d49aa765512de8ff07a22384595bf5b3fd5248"},
    "27b": {"kv_heads": 4, "geometry_of":
            "Qwen3.8-27B-MLX-MXFP8@6ee88ed04dce8f87958ad4df9fee0b402b6347f3f5e8318a945407806ff45f6d"},
}
CONTEXTS = (16384, 32768, 131072)
ROWS = 8192
REPS = 5
REF_ROWS = 48

# ---------------------------------------------------------------------------
# Variant kernel (A2).  Same inputs, grid and ids ABI as qwen4_qsa_nax.

_VARIANT_HEAD = r"""
    constexpr int NSG = 4;
    constexpr int DS  = D / NSG;      // 64 dims per simdgroup
    constexpr int NJ  = DS / 32;      // 16x32x32 QK ops per step
    constexpr int NNT = DS / 32;      // 32-wide output tiles

    const int L   = dims[0];
    const int TOT = dims[1];
    const int U   = dims[2];

    const ushort sg   = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const uint  tok   = threadgroup_position_in_grid.y;
    const uint  bkv   = threadgroup_position_in_grid.z;
    const uint  b     = bkv / NKVH;
    const uint  hkv   = bkv % NKVH;

    const short qid = lane >> 2;
    const short fm0 = ((qid & 4) | ((lane >> 1) & 3));
    const short fn0 = ((qid & 2) | (lane & 1)) * 4;
    const short fcol = 2 * fn0;   // 8 contiguous dims feed a fragment pair

    threadgroup float sS[NSG * 32 * 16];

    const int64_t skl = k_strides[2];
    const int64_t svl = v_strides[2];
    const device T* kbase = k + (int64_t)b * k_strides[0] + (int64_t)hkv * k_strides[1] + sg * DS + fcol;
    const device T* vbase = v + (int64_t)b * v_strides[0] + (int64_t)hkv * v_strides[1] + sg * DS + fcol;

    const int  lpad      = left_pad[b];
    const uint tile_base = (b * L + tok) * U;
    const uint cnt       = counts[b * L + tok];
    const uint nsel      = n_sel[b * L + tok];
    const int qp       = qpos[b * L + tok];
    const int complete = ((qp + 1) / BS) * BS;

    int head_of[2];
    bool row_live[2];
    for (short rr = 0; rr < 2; rr++) {
        int r = fm0 + rr * 8;
        row_live[rr] = (r < GQA);
        head_of[rr] = (int)hkv * GQA + r;
    }

    metal::vec<T, 8> qa[NJ][2];
    for (short jj = 0; jj < NJ; jj++) {
        for (short rr = 0; rr < 2; rr++) {
            qa[jj][rr] = row_live[rr]
                ? *(const device metal::vec<T, 8>*)(q + ((size_t)(b * NQH + head_of[rr]) * L + tok) * D
                                                     + sg * DS + 32 * jj + fcol)
                : metal::vec<T, 8>(T(0));
        }
    }

    float of[NNT][16];
    for (short t = 0; t < NNT; t++)
        for (short i = 0; i < 16; i++) of[t][i] = 0.0f;
    float rmax_r[2] = {-INFINITY, -INFINITY};
    float rsum_r[2] = {0.0f, 0.0f};

    constexpr auto desc_qk = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto desc_pv = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 32, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    const float qk_scale = scale[0];

    for (uint u0 = 0; u0 < cnt; u0 += 8) {
        // Keys fm0, fm0+8, fm0+16, fm0+24 of this 32-key step.
        uint prow[4];
        for (short qq = 0; qq < 4; qq++) {
            int n = fm0 + qq * 8;
            uint u = u0 + (uint)(n >> 2);
            int bid = (u < cnt) ? (int)ids[tile_base + u] : 0;
            int phys = lpad + bid * BS + (n & 3);
            prow[qq] = (uint)metal::clamp(phys, 0, (int)TOT - 1);
        }
        {
            mpp::tensor_ops::matmul2d<desc_qk, metal::execution_simdgroup> op;
            auto ca = op.template get_left_input_cooperative_tensor<T, T, float>();
            auto cb = op.template get_right_input_cooperative_tensor<T, T, float>();
            auto cc = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ca)>,
                metal::remove_addrspace_t<decltype(cb)>, float>();
            for (short i = 0; i < 16; i++) cc[i] = 0.0f;
            for (short jj = 0; jj < NJ; jj++) {
                metal::vec<T, 8> kr[4];
                for (short r = 0; r < 4; r++)
                    kr[r] = *(const device metal::vec<T, 8>*)(kbase + (int64_t)prow[r] * skl + 32 * jj);
                for (short j = 0; j < 4; j++) {
                    ca[j] = qa[jj][0][j];
                    ca[4 + j] = qa[jj][1][j];
                    ca[8 + j] = qa[jj][0][4 + j];
                    ca[12 + j] = qa[jj][1][4 + j];
                    cb[j] = kr[0][j];
                    cb[4 + j] = kr[1][j];
                    cb[8 + j] = kr[2][j];
                    cb[12 + j] = kr[3][j];
                    cb[16 + j] = kr[0][4 + j];
                    cb[20 + j] = kr[1][4 + j];
                    cb[24 + j] = kr[2][4 + j];
                    cb[28 + j] = kr[3][4 + j];
                }
                op.run(ca, cb, cc);
            }
            for (short i = 0; i < 16; i++)
                sS[(sg * 32 + lane) * 16 + i] = cc[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float s[16];
        for (short i = 0; i < 16; i++) {
            s[i] = (sS[(0 * 32 + lane) * 16 + i] + sS[(1 * 32 + lane) * 16 + i]
                  + sS[(2 * 32 + lane) * 16 + i] + sS[(3 * 32 + lane) * 16 + i])
                 * qk_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        bool colok[8];
        for (short f = 0; f < 2; f++) {
            for (short co = 0; co < 4; co++) {
                int n = f * 16 + fn0 + co;
                uint u = u0 + (uint)(n >> 2);
                bool inb = (u < cnt);
                int bid = inb ? (int)ids[tile_base + u] : 0;
                int kl = bid * BS + (n & 3);
                colok[f * 4 + co] = inb && (kl >= 0) && (kl <= qp)
                                    && ((u < nsel) || (kl >= complete));
            }
        }
        for (short f = 0; f < 2; f++)
            for (short rr = 0; rr < 2; rr++)
                for (short co = 0; co < 4; co++)
                    if (!colok[f * 4 + co]) s[f * 8 + rr * 4 + co] = -INFINITY;

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
            bool live = mn > -INFINITY;
            alpha[rr] = live ? metal::fast::exp(rmax_r[rr] - mn) : 1.0f;
            float sm = 0.0f;
            for (short f = 0; f < 2; f++) {
                for (short co = 0; co < 4; co++) {
                    short idx = f * 8 + rr * 4 + co;
                    float pv = live ? metal::fast::exp(s[idx] - mn) : 0.0f;
                    p[idx] = pv;
                    sm += pv;
                }
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
            mpp::tensor_ops::matmul2d<desc_pv, metal::execution_simdgroup> op;
"""

_PV_BF16 = r"""
            auto ca = op.template get_left_input_cooperative_tensor<T, T, float>();
            auto cb = op.template get_right_input_cooperative_tensor<T, T, float>();
            auto cc = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ca)>,
                metal::remove_addrspace_t<decltype(cb)>, float>();
            for (short i = 0; i < 16; i++) cc[i] = of[t][i];
            metal::vec<T, 8> vr[4];
            for (short r = 0; r < 4; r++)
                vr[r] = *(const device metal::vec<T, 8>*)(vbase + (int64_t)prow[r] * svl + 32 * t);
            for (short i = 0; i < 16; i++) ca[i] = (T)p[i];
            for (short j = 0; j < 4; j++) {
                cb[j] = vr[0][j];
                cb[4 + j] = vr[1][j];
                cb[8 + j] = vr[0][4 + j];
                cb[12 + j] = vr[1][4 + j];
                cb[16 + j] = vr[2][j];
                cb[20 + j] = vr[3][j];
                cb[24 + j] = vr[2][4 + j];
                cb[28 + j] = vr[3][4 + j];
            }
            op.run(ca, cb, cc);
            for (short i = 0; i < 16; i++) of[t][i] = cc[i];
        }
    }
"""

_PV_HILO = r"""
            auto ca = op.template get_left_input_cooperative_tensor<half, T, float>();
            auto cb = op.template get_right_input_cooperative_tensor<half, T, float>();
            auto cc = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ca)>,
                metal::remove_addrspace_t<decltype(cb)>, float>();
            for (short i = 0; i < 16; i++) cc[i] = of[t][i];
            for (short f = 0; f < 2; f++) {
                const metal::vec<T, 8> ra = *(const device metal::vec<T, 8>*)(vbase + (int64_t)prow[2 * f] * svl + 32 * t);
                const metal::vec<T, 8> rb = *(const device metal::vec<T, 8>*)(vbase + (int64_t)prow[2 * f + 1] * svl + 32 * t);
                for (short j = 0; j < 4; j++) {
                    cb[j] = ra[j];
                    cb[4 + j] = rb[j];
                    cb[8 + j] = ra[4 + j];
                    cb[12 + j] = rb[4 + j];
                    cb[16 + j] = ra[j];
                    cb[20 + j] = rb[j];
                    cb[24 + j] = ra[4 + j];
                    cb[28 + j] = rb[4 + j];
                }
                for (short i = 0; i < 8; i++) {
                    const half hi = half(p[f * 8 + i]);
                    ca[i] = hi;
                    ca[8 + i] = half(p[f * 8 + i] - float(hi));
                }
                op.run(ca, cb, cc);
            }
            for (short i = 0; i < 16; i++) of[t][i] = cc[i];
        }
    }
"""

_VARIANT_TAIL = r"""
    for (short t = 0; t < NNT; t++) {
        for (short f = 0; f < 2; f++) {
            for (short rr = 0; rr < 2; rr++) {
                if (!row_live[rr]) continue;
                float inv = (rsum_r[rr] > 0.0f) ? (1.0f / rsum_r[rr]) : 0.0f;
                for (short co = 0; co < 4; co++) {
                    int c = sg * DS + t * 32 + fcol + f * 4 + co;
                    out[((size_t)(b * NQH + head_of[rr]) * L + tok) * D + c] =
                        of[t][f * 8 + rr * 4 + co] * inv;
                }
            }
        }
    }
"""

_VARIANTS = {}


def _variant_kernel(kind):
    import mlx.core as mx
    from mlx2.runtime.models import qwen4_qsa_nax as nax

    if kind not in _VARIANTS:
        pv = {"bf16": _PV_BF16, "hilo": _PV_HILO}[kind]
        _VARIANTS[kind] = mx.fast.metal_kernel(
            name=f"bench_nax_qsa_attn_v2_{kind}",
            input_names=["q", "k", "v", "ids", "counts", "n_sel", "qpos",
                         "left_pad", "scale", "dims"],
            output_names=["out"],
            header=nax.HEADER,
            source=_VARIANT_HEAD + pv + _VARIANT_TAIL,
            ensure_row_contiguous=False,
        )
    return _VARIANTS[kind]


def variant_attention(kind, q, k, v, ids, counts, n_sel, q_pos, left_pad, *,
                      scale, u_width, total, n_kv_heads):
    import mlx.core as mx

    batch, nqh, length, dim = q.shape
    assert dim == D and nqh % n_kv_heads == 0 and nqh // n_kv_heads <= 16
    (out,) = _variant_kernel(kind)(
        inputs=[mx.contiguous(q), k, v,
                mx.contiguous(ids.astype(mx.uint32)),
                mx.contiguous(counts.astype(mx.uint32)),
                mx.contiguous(n_sel.astype(mx.uint32)),
                mx.contiguous(mx.broadcast_to(q_pos, (batch, length)).astype(mx.int32)),
                mx.contiguous(left_pad.astype(mx.int32)),
                mx.array([scale], dtype=mx.float32),
                mx.array([length, total, u_width], dtype=mx.int32)],
        template=[("T", q.dtype), ("D", dim), ("NQH", nqh), ("NKVH", n_kv_heads),
                  ("GQA", nqh // n_kv_heads), ("BS", BLOCK)],
        grid=(128, length, batch * n_kv_heads),
        threadgroup=(128, 1, 1),
        output_shapes=[(batch, nqh, length, dim)],
        output_dtypes=[mx.float32],
    )
    return out


# ---------------------------------------------------------------------------
# Inputs


def make_selection(start, rows, topk, *, seed, contiguous=False):
    """Per-row sorted ids: the last LOCAL_BLOCKS complete blocks plus random
    earlier ones (or, contiguous, the most recent ``n_sel`` blocks)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    ids = np.zeros((1, rows, topk), dtype=np.int32)
    counts = np.zeros((1, rows), dtype=np.int32)
    for i in range(rows):
        p = start + i
        nb = (p + 1) // BLOCK
        k = min(topk, nb)
        if contiguous or nb <= topk:
            sel = np.arange(nb - k, nb, dtype=np.int32)
        else:
            local = np.arange(nb - LOCAL_BLOCKS, nb, dtype=np.int32)
            far = rng.choice(nb - LOCAL_BLOCKS, k - LOCAL_BLOCKS, replace=False).astype(np.int32)
            sel = np.sort(np.concatenate([far, local]))
        ids[0, i, :k] = sel
        counts[0, i] = k
    return ids, counts


def make_case(geom, context, rows, *, topk=TOPK, seed=20261105, contiguous=False, tensors=None):
    import mlx.core as mx
    import numpy as np

    kvh = GEOMETRIES[geom]["kv_heads"]
    start = context - rows
    if tensors is not None:
        q, k, v = tensors
    else:
        key = mx.random.key(seed)
        kq, kk, kv = mx.random.split(key, 3)
        q = mx.random.normal((1, HEADS, rows, D), key=kq).astype(mx.bfloat16)
        k = mx.random.normal((1, kvh, context, D), key=kk).astype(mx.bfloat16)
        v = mx.random.normal((1, kvh, context, D), key=kv).astype(mx.bfloat16)
    ids, counts = make_selection(start, rows, topk, seed=seed, contiguous=contiguous)
    qp = np.arange(start, context, dtype=np.int32)[None]
    cb = SimpleNamespace(
        block_ids=mx.array(ids), block_counts=mx.array(counts),
        tail_start=mx.array((qp + 1) // BLOCK * BLOCK), tail_stop=mx.array(qp + 1),
        left_padding=None, block_size=BLOCK, physical_width=context, causal_mask=None)
    mx.eval(q, k, v, cb.block_ids, cb.block_counts, cb.tail_start, cb.tail_stop)
    return SimpleNamespace(geom=geom, context=context, rows=rows, kvh=kvh, start=start,
                           q=q, k=k, v=v, cb=cb, ids_np=ids, counts_np=counts, qp_np=qp[0])


def compact(case):
    from mlx2.runtime.models.qwen4_qsa_nax import compact_blocks_to_kernel_inputs

    return compact_blocks_to_kernel_inputs(case.cb)


def run_arm(arm, case, kin):
    from mlx2.runtime.models.qwen4_qsa_nax import nax_qsa_attention

    ids, counts, n_sel, u_width, q_pos, left_pad, total = kin
    kw = dict(scale=D ** -0.5, u_width=u_width, total=total, n_kv_heads=case.kvh)
    if arm in ("A0k", "A1k"):
        # Pinned: the in-tree law v2 is selectable process-wide
        # (MLX_QWEN4_QSA_NAX_LAW), and these arms are the v1 baseline.
        return nax_qsa_attention(case.q, case.k, case.v, ids, counts, n_sel, q_pos, left_pad,
                                 law="v1", **kw)
    kind = {"A2k": "bf16", "A2h": "hilo"}[arm]
    return variant_attention(kind, case.q, case.k, case.v, ids, counts, n_sel, q_pos, left_pad, **kw)


def visible_keys(case):
    """Total attended key positions over all rows (selected blocks + tail)."""
    import numpy as np

    qp = case.qp_np
    complete = (qp + 1) // BLOCK * BLOCK
    return int(np.sum(case.counts_np[0]) * BLOCK + np.sum(qp + 1 - complete))


def reference_rows(case, rows):
    """float64 attention for sampled rows on the bf16 operands: [len(rows), H, D]."""
    import mlx.core as mx
    import numpy as np

    q = np.array(case.q.astype(mx.float32))[0]
    k = np.array(case.k.astype(mx.float32))[0]
    v = np.array(case.v.astype(mx.float32))[0]
    gqa = HEADS // case.kvh
    out = np.zeros((len(rows), HEADS, D))
    for n, i in enumerate(rows):
        p = int(case.qp_np[i])
        complete = (p + 1) // BLOCK * BLOCK
        sel = case.ids_np[0, i, : case.counts_np[0, i]]
        keys = np.concatenate([(sel[:, None] * BLOCK + np.arange(BLOCK)).ravel(),
                               np.arange(complete, p + 1)])
        for h in range(HEADS):
            kk = k[h // gqa, keys].astype(np.float64)
            s = kk @ q[h, i].astype(np.float64) * D ** -0.5
            w = np.exp(s - s.max())
            out[n, h] = (w / w.sum()) @ v[h // gqa, keys].astype(np.float64)
    return out


def err(got_rows, ref):
    import numpy as np

    d = got_rows - ref
    return {"rel_l2": float(np.linalg.norm(d) / np.linalg.norm(ref)),
            "max_abs": float(np.abs(d).max()),
            "finite": bool(np.isfinite(got_rows).all())}


def time_fn(fn):
    import mlx.core as mx

    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    mx.synchronize()
    return (time.perf_counter() - t0) * 1e3


def chunk_invariant(arm, case, kin, full):
    """Bitwise: four row-slice launches vs the one full launch."""
    import mlx.core as mx

    ids, counts, n_sel, u_width, q_pos, left_pad, total = kin
    parts = []
    step = case.rows // 4
    for a in range(0, case.rows, step):
        sub = SimpleNamespace(**vars(case))
        sub.q = case.q[:, :, a:a + step]
        skin = (ids[:, a:a + step], counts[:, a:a + step], n_sel[:, a:a + step], u_width,
                q_pos[:, a:a + step], left_pad, total)
        parts.append(run_arm(arm, sub, skin))
    joined = mx.concatenate(parts, axis=2)
    return bool(mx.array_equal(joined, full).item())


# ---------------------------------------------------------------------------


def host_sample():
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    m = re.search(r"Swapouts:\s+(\d+)", vm)
    return {"time": time.time(), "swapouts": int(m.group(1)) if m else None}


def source():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    files = {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
             for rel in ("scripts/bench_qsa_nax_kernel_ab.py",
                         "src/mlx2/runtime/models/qwen4_qsa_nax.py")}
    return {"commit": head, "src_dirty": bool(dirty), "files": files}


def atomic(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    os.replace(tmp, path)


def one_case(geom, context, rows, reps, ref_n, arms_ok, log):
    import mlx.core as mx
    import numpy as np

    case = make_case(geom, context, rows)
    contig = make_case(geom, context, rows, contiguous=True, tensors=(case.q, case.k, case.v))
    kin = compact(case)
    mx.eval(*[x for x in kin if isinstance(x, mx.array)])
    kin_c = compact(contig)
    mx.eval(*[x for x in kin_c if isinstance(x, mx.array)])
    arms = ["A0c", "A0k", "A1k"] + [a for a in ("A2k", "A2h") if arms_ok.get(a, True)]
    fns = {
        "A0c": lambda: [x for x in compact(case) if isinstance(x, mx.array)],
        "A0k": lambda: run_arm("A0k", case, kin),
        "A1k": lambda: run_arm("A1k", contig, kin_c),
        "A2k": lambda: run_arm("A2k", case, kin),
        "A2h": lambda: run_arm("A2h", case, kin),
    }
    for a in arms:          # warm-up / compile
        time_fn(fns[a]); time_fn(fns[a])
    times = {a: [] for a in arms}
    for r in range(reps):
        order = arms if r % 2 == 0 else arms[::-1]
        for a in order:
            times[a].append(time_fn(fns[a]))
    keys = visible_keys(case)
    flops = 4.0 * HEADS * keys * D
    sample = sorted(set(np.linspace(0, rows - 1, ref_n).astype(int).tolist()))
    ref = reference_rows(case, sample)
    row = {"geometry": geom, "geometry_of": GEOMETRIES[geom]["geometry_of"], "context": context,
           "rows": rows, "kv_heads": case.kvh, "visible_keys": keys, "gflop": flops / 1e9,
           "arms": {}}
    full_out = {}
    for a in arms:
        med = statistics.median(times[a])
        entry = {"ms": med, "ms_all": times[a]}
        if a != "A0c":
            entry["tflops"] = flops / (med * 1e-3) / 1e12
        if a in ("A0k", "A2k", "A2h"):
            out = fns[a]()
            mx.eval(out)
            full_out[a] = out
            got = np.array(out[0][:, sample].transpose(1, 0, 2)).astype(np.float64)
            entry["vs_fp64"] = err(got, ref)
            entry["chunk_invariant_bitwise"] = chunk_invariant(a, case, kin, out)
        row["arms"][a] = entry
    for a in ("A2k", "A2h"):
        if a in full_out:
            row["arms"][a]["bitwise_equal_to_A0k"] = bool(mx.array_equal(full_out[a], full_out["A0k"]).item())
    log(json.dumps({k: row[k] for k in ("geometry", "context")} |
                   {a: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                        for kk, vv in e.items() if kk in ("ms", "tflops")} for a, e in row["arms"].items()}))
    # Rebind instead of ``del``: the timing lambdas in ``fns`` close over these.
    fns = case = contig = kin = kin_c = full_out = None
    mx.clear_cache()
    return row


def smoke(log):
    """Tiny-shape compile + numerics gate per arm; a failing variant is skipped."""
    import mlx.core as mx
    import numpy as np

    ok = {}
    detail = {}
    for geom in GEOMETRIES:
        case = make_case(geom, 2048, 64, topk=96)
        kin = compact(case)
        sample = list(range(0, 64, 4))
        ref = reference_rows(case, sample)
        for a in ("A0k", "A2k", "A2h"):
            try:
                out = run_arm(a, case, kin)
                mx.eval(out)
                got = np.array(out[0][:, sample].transpose(1, 0, 2)).astype(np.float64)
                e = err(got, ref)
                passed = e["finite"] and e["rel_l2"] < 2e-2
            except Exception as exc:  # noqa: BLE001 -- recorded, the arm is skipped
                e, passed = {"error": repr(exc)[:500]}, False
            detail[f"{geom}:{a}"] = e
            ok[a] = ok.get(a, True) and passed
    log("smoke " + json.dumps(detail))
    return ok, detail


def unit(out_dir: Path, quick: bool):
    import mlx.core as mx

    mx.set_cache_limit(4 << 30)
    log_path = out_dir / "bench.log"

    def log(msg):
        with log_path.open("a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        print(msg, flush=True)

    report = {"source": source(), "device": mx.device_info().get("device_name"),
              "mlx": mx.__version__, "scope": "kernel-only research microbench on synthetic tensors; "
              "not qualification, no model checkpoint loaded",
              "selection": f"top-{TOPK} blocks of {BLOCK}: last {LOCAL_BLOCKS} complete blocks + random "
              "earlier blocks, plus the tail block", "before": host_sample(), "rows": []}
    ok, detail = smoke(log)
    report["smoke"] = {"ok": ok, "detail": detail}
    if not ok.get("A0k"):
        report["aborted"] = "baseline kernel failed the smoke gate"
        atomic(out_dir / "report.json", report)
        return 1
    contexts = (32768,) if quick else CONTEXTS
    for geom in GEOMETRIES:
        for ctx in contexts:
            before = host_sample()["swapouts"]
            report["rows"].append(one_case(geom, ctx, ROWS, 2 if quick else REPS,
                                           8 if quick else REF_ROWS, ok, log))
            after = host_sample()["swapouts"]
            atomic(out_dir / "report.partial.json", report)
            if before is not None and after is not None and after > before:
                report["aborted"] = f"swapouts rose {before} -> {after}"
                atomic(out_dir / "report.json", report)
                return 2
    report["after"] = host_sample()
    atomic(out_dir / "report.json", report)
    return 0


def owned(out_dir: Path, job_id: str, quick: bool):
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    with DesktopCPGLease(out_dir, job_id, checkpoint="synthetic-qsa-nax-kernel",
                         source_revision=source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out_dir / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out_dir)]
        if quick:
            command.append("--quick")
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--job-id", default="qsa-nax-kernel-ab-20261105")
    args = parser.parse_args()
    if args.mode == "unit":
        return unit(args.out, args.quick)
    return owned(args.out, args.job_id, args.quick)


if __name__ == "__main__":
    raise SystemExit(main())
