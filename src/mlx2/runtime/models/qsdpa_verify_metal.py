"""Verify-block attention over affine-quantized (kv_q8) K/V, direct read (Metal).

Self-MTP verify (K=2 gives 3 rows) and copy drafts (up to 8 rows) send
2-8 query rows per request through attention. Before this module those rows
took the composed path in ``models/base.py``: ``quantized_matmul`` for
Q.K^T, a materialized masked softmax, then ``quantized_matmul`` for P.V with
the GQA group as a broadcast axis. This kernel reads each KV head's int8
K/V once per step for all of its query heads and all verify rows.

Design (the verify tile of Splash's paged attention, adapted to mlx2's
cache format; see docs/PROVENANCE.md and provenance/splash-verify-attention):

* One threadgroup owns one (batch row, KV head, fused-row group, history
  split). Its query tile is the fused matrix M = rows x query heads of the
  group, M <= 32 (larger blocks use two or more fused-row groups).
* Per 32-key tile the threadgroup stages the tile's packed K codes and their
  scales/biases in threadgroup memory (the next tile's are already in
  flight in registers), computes S = Q.K^T with 8x8 simdgroup matrix
  products whose K operand is dequantized (``scale * code + bias`` per
  group) as it is loaded, runs an online softmax per fused row with its
  causal limit (4 threads per row), stages the V codes in the same buffer
  and accumulates O += P.V in simdgroup matrices, rescaled when a row's
  running maximum grows.
* Each split writes a normalized partial (o/l, l, m) per fused row; an
  MLX-op merge combines the splits (the decode kernel's merge).
* Causality is per query row: row r of an L-row block sees keys
  ``t <= N - L + r``. Batched calls take per-row left padding
  (``BatchQuantizedKVCache``): keys ``t < left_padding[b]`` are skipped and
  the history split covers only ``[left_padding[b], N)``.

What differs from Splash: Splash stores symmetric int8 with one fp32 scale
per (token, KV head) in 32-token pages and runs QK/PV as Metal 4 tensor ops;
mlx2's kv_q8 is MLX's packed affine format (uint32 words, a scale and a
bias per 64 channels), read from strided live cache views with no copies,
so the tile dequantizes inside the simdgroup_matrix operand loads.

Covered: 1-8 query rows, head_dim 128 or 256, group size 32/64/128, 2/4/8
bits for keys and values independently, any GQA ratio, bf16/fp16, causal
masks with optional left padding. Everything else stays on the composed
path; see ``use_verify_kernel``. ``MLX2_QSDPA_VERIFY_KERNEL=0`` disables it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import mlx.core as mx

MAX_ROWS = 8
MAX_FUSED = 32  # fused rows (verify rows x query heads) per threadgroup
TILE = 32
SIMDGROUPS = 4

# Below this many cached positions the composed path is about as fast or
# faster (launch + merge overhead). On an M5 Max the kernel is 0.5-0.8x the
# composed path at 1-4K, 0.84-1.46x at 16K, 1.2-2.0x at 64K and 1.4-2.6x at
# 128K (27B and Flash-Next geometries, 1-8 rows, B1/B4;
# qualification/runs/sp-verifyattn-20260925/bench_verify.log).
MIN_CONTEXT = 32768


_SOURCE = r"""
    #define VATT_UNROLL _Pragma("clang loop unroll(full)")
    constexpr int TN = 32;                  // keys per tile (one 8-key block per simdgroup)
    constexpr int NTH = NS * 32;
    constexpr int QS = D + 8;
    constexpr int RB = FRB / 8;
    constexpr int FR = L * G;
    constexpr int NG = D / GS;
    constexpr int KW = D * KB / 32;         // packed words per key row
    constexpr int VWN = D * VB / 32;
    constexpr int CW = (KW > VWN ? KW : VWN) + 4;
    constexpr int KPT = TN * KW / NTH;      // words each thread stages per tile
    constexpr int VPT = TN * VWN / NTH;
    constexpr int SPT = (TN * NG + NTH - 1) / NTH;
    constexpr uint KMASK = (1u << KB) - 1u;
    constexpr uint VMASK = (1u << VB) - 1u;
    constexpr int CPS = D / NS;             // output columns per simdgroup
    constexpr int CF = CPS / 8;

    threadgroup half q_sh[FRB * QS];
    threadgroup uint c_sh[TN * CW];         // raw packed codes of the K (then V) tile
    threadgroup float cs_sh[TN * NG];
    threadgroup float cb_sh[TN * NG];
    threadgroup float s_sh[FRB * TN];
    threadgroup float fac_sh[FRB];
    threadgroup float l_sh[FRB];
    threadgroup float m_sh[FRB];

    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = sg * 32 + lane;
    const uint gy = threadgroup_position_in_grid.y;
    const int block = int(threadgroup_position_in_grid.z);
    const int blocks = int(threadgroups_per_grid.z);

    const int Hkv = kw_shape[1];
    const int N = kw_shape[2];
    const int Hq = q_shape[1];
    const int fg = int(gy) % FG;
    const int bh = int(gy) / FG;
    const int b = bh / Hkv;
    const int hkv = bh % Hkv;
    const int f0 = fg * FRB;
    const int nf = min(FRB, FR - f0);

    for (uint idx = tid; idx < uint(FRB * D); idx += NTH) {
        const int f = int(idx) / D, d = int(idx) % D;
        float val = 0.0f;
        if (f < nf) {
            const int fu = f0 + f, r = fu / G, j = fu % G;
            val = float(q[b * q_strides[0] + (hkv * G + j) * q_strides[1]
                          + r * q_strides[2] + d * q_strides[3]]) * scale[0];
        }
        q_sh[f * QS + d] = half(val);
    }

    const int lp = lpad[b];
    const int span = max(0, N - lp);
    const int chunk = (span + blocks - 1) / blocks;
    const int kstart = lp + block * chunk;
    const int kend = min(N, kstart + chunk);

    // softmax ownership: 4 threads per fused row, 8 keys each
    const int srow = int(tid) / 4;
    const int spart = int(tid) % 4;
    int my_limit = 0;
    if (srow < nf) my_limit = min(kend, N - L + (f0 + srow) / G + 1);
    float mrow = -INFINITY, lrow = 0.0f;

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;

    simdgroup_float8x8 acc[RB][CF];
    VATT_UNROLL for (int i = 0; i < RB; i++)
        VATT_UNROLL for (int c = 0; c < CF; c++)
            acc[i][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    const device uint32_t* kwb = kw + b * kw_strides[0] + hkv * kw_strides[1];
    const device T* ksb = ks + b * ks_strides[0] + hkv * ks_strides[1];
    const device T* kbb = kb + b * kb_strides[0] + hkv * kb_strides[1];
    const device uint32_t* vwb = vw + b * vw_strides[0] + hkv * vw_strides[1];
    const device T* vsb = vs + b * vs_strides[0] + hkv * vs_strides[1];
    const device T* vbb = vb + b * vb_strides[0] + hkv * vb_strides[1];

    // this thread's slice of a tile: KPT (VPT) consecutive words of one row
    const int kt_row = int(tid) * KPT / KW, kt_word = int(tid) * KPT % KW;
    const int vt_row = int(tid) * VPT / VWN, vt_word = int(tid) * VPT % VWN;
    uint pk[KPT], pv[VPT];
    float pks[SPT], pkb[SPT], pvs[SPT], pvb[SPT];

    #define LOAD_K(T0) {                                                        \
        const int tt = (T0) + kt_row;                                           \
        const bool ok = tt < kend;                                              \
        VATT_UNROLL for (int w = 0; w < KPT; w++)                                    \
            pk[w] = ok ? kwb[tt * kw_strides[2] + kt_word + w] : 0u;            \
        VATT_UNROLL for (int i = 0; i < SPT; i++) {                                  \
            const int e = int(tid) + i * NTH, t2 = (T0) + e / NG, g = e % NG;   \
            const bool ok2 = e < TN * NG && t2 < kend;                          \
            pks[i] = ok2 ? float(ksb[t2 * ks_strides[2] + g]) : 0.0f;           \
            pkb[i] = ok2 ? float(kbb[t2 * kb_strides[2] + g]) : 0.0f;           \
        } }
    #define LOAD_V(T0) {                                                        \
        const int tt = (T0) + vt_row;                                           \
        const bool ok = tt < kend;                                              \
        VATT_UNROLL for (int w = 0; w < VPT; w++)                                    \
            pv[w] = ok ? vwb[tt * vw_strides[2] + vt_word + w] : 0u;            \
        VATT_UNROLL for (int i = 0; i < SPT; i++) {                                  \
            const int e = int(tid) + i * NTH, t2 = (T0) + e / NG, g = e % NG;   \
            const bool ok2 = e < TN * NG && t2 < kend;                          \
            pvs[i] = ok2 ? float(vsb[t2 * vs_strides[2] + g]) : 0.0f;           \
            pvb[i] = ok2 ? float(vbb[t2 * vb_strides[2] + g]) : 0.0f;           \
        } }

    if (kstart < kend) {
        LOAD_K(kstart);
        LOAD_V(kstart);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int tile = kstart; tile < kend; tile += TN) {
        VATT_UNROLL for (int w = 0; w < KPT; w++) c_sh[kt_row * CW + kt_word + w] = pk[w];
        VATT_UNROLL for (int i = 0; i < SPT; i++) {
            const int e = int(tid) + i * NTH;
            if (e < TN * NG) { cs_sh[e] = pks[i]; cb_sh[e] = pkb[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tile + TN < kend) LOAD_K(tile + TN);   // in flight during QK

        // S = Q . K^T: this simdgroup owns keys [8*sg, 8*sg+8) for every row block
        {
            simdgroup_float8x8 sacc[RB];
            VATT_UNROLL for (int i = 0; i < RB; i++)
                sacc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            const int t0 = int(sg) * 8 + fn;
            const threadgroup uint* r0 = c_sh + t0 * CW;
            const threadgroup uint* r1 = r0 + CW;
            VATT_UNROLL for (int g = 0; g < NG; g++) {
                const float sc0 = cs_sh[t0 * NG + g], bi0 = cb_sh[t0 * NG + g];
                const float sc1 = cs_sh[(t0 + 1) * NG + g], bi1 = cb_sh[(t0 + 1) * NG + g];
                VATT_UNROLL for (int kk = 0; kk < GS / 8; kk++) {
                    const int k = g * (GS / 8) + kk;
                    const int d = k * 8 + fm;
                    const int wi = d * KB / 32, sh = (d * KB) % 32;
                    simdgroup_float8x8 bm;
                    bm.thread_elements()[0] = sc0 * float((r0[wi] >> sh) & KMASK) + bi0;
                    bm.thread_elements()[1] = sc1 * float((r1[wi] >> sh) & KMASK) + bi1;
                    VATT_UNROLL for (int rb = 0; rb < RB; rb++) {
                        const threadgroup half* qa = q_sh + (rb * 8 + fm) * QS + k * 8 + fn;
                        simdgroup_float8x8 a;
                        a.thread_elements()[0] = float(qa[0]);
                        a.thread_elements()[1] = float(qa[1]);
                        simdgroup_multiply_accumulate(sacc[rb], a, bm, sacc[rb]);
                    }
                }
            }
            VATT_UNROLL for (int rb = 0; rb < RB; rb++)
                simdgroup_store(sacc[rb], s_sh + rb * 8 * TN + int(sg) * 8, TN);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax: 4 threads per fused row, causal limit per verify row
        if (srow < FRB) {
            threadgroup float* sr = s_sh + srow * TN + spart * 8;
            const int t0 = tile + spart * 8;
            float x[8];
            float tmax = -INFINITY;
            VATT_UNROLL for (int u = 0; u < 8; u++) {
                x[u] = sr[u];
                if (t0 + u < my_limit) tmax = max(tmax, x[u]);
            }
            tmax = max(tmax, simd_shuffle_xor(tmax, 1));
            tmax = max(tmax, simd_shuffle_xor(tmax, 2));
            const float nm = max(mrow, tmax);
            const float fac = mrow == -INFINITY ? 0.0f : fast::exp(mrow - nm);
            float sum = 0.0f;
            VATT_UNROLL for (int u = 0; u < 8; u++) {
                const float p = t0 + u < my_limit ? fast::exp(x[u] - nm) : 0.0f;
                sr[u] = p;
                sum += p;
            }
            sum += simd_shuffle_xor(sum, 1);
            sum += simd_shuffle_xor(sum, 2);
            lrow = lrow * fac + sum;
            mrow = nm;
            if (spart == 0) fac_sh[srow] = fac;
        }
        // V tile codes replace K's (K is no longer read)
        VATT_UNROLL for (int w = 0; w < VPT; w++) c_sh[vt_row * CW + vt_word + w] = pv[w];
        VATT_UNROLL for (int i = 0; i < SPT; i++) {
            const int e = int(tid) + i * NTH;
            if (e < TN * NG) { cs_sh[e] = pvs[i]; cb_sh[e] = pvb[i]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tile + TN < kend) LOAD_V(tile + TN);   // in flight during PV

        // O = O * fac + P . V; this simdgroup owns output columns [sg*CPS, +CPS)
        VATT_UNROLL for (int rb = 0; rb < RB; rb++) {
            const float f = fac_sh[rb * 8 + fm];
            VATT_UNROLL for (int c = 0; c < CF; c++) {
                acc[rb][c].thread_elements()[0] *= f;
                acc[rb][c].thread_elements()[1] *= f;
            }
        }
        VATT_UNROLL for (int kk = 0; kk < TN / 8; kk++) {
            simdgroup_float8x8 a[RB];
            VATT_UNROLL for (int rb = 0; rb < RB; rb++)
                simdgroup_load(a[rb], s_sh + rb * 8 * TN + kk * 8, TN);
            const int t = kk * 8 + fm;
            const threadgroup uint* vr = c_sh + t * CW;
            VATT_UNROLL for (int c = 0; c < CF; c++) {
                const int d0 = int(sg) * CPS + c * 8 + fn;
                const int g = d0 / GS;
                const float sc = cs_sh[t * NG + g], bi = cb_sh[t * NG + g];
                const uint word = vr[d0 * VB / 32];
                const int sh = (d0 * VB) % 32;
                simdgroup_float8x8 bm;
                bm.thread_elements()[0] = sc * float((word >> sh) & VMASK) + bi;
                bm.thread_elements()[1] = sc * float((word >> (sh + VB)) & VMASK) + bi;
                VATT_UNROLL for (int rb = 0; rb < RB; rb++)
                    simdgroup_multiply_accumulate(acc[rb][c], a[rb], bm, acc[rb][c]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #undef LOAD_K
    #undef LOAD_V

    if (srow < FRB && spart == 0) {
        l_sh[srow] = lrow;
        m_sh[srow] = mrow;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int P = blocks;
    VATT_UNROLL for (int rb = 0; rb < RB; rb++) {
        const int fr = rb * 8 + fm;
        if (fr < nf) {
            const int fu = f0 + fr, r = fu / G, j = fu % G;
            const size_t row = size_t(((b * Hq + hkv * G + j) * L + r) * P + block);
            const float lv = l_sh[fr];
            const float inv = lv > 0.0f ? 1.0f / lv : 0.0f;
            VATT_UNROLL for (int c = 0; c < CF; c++) {
                const int col = int(sg) * CPS + c * 8 + fn;
                part_o[row * D + col] = acc[rb][c].thread_elements()[0] * inv;
                part_o[row * D + col + 1] = acc[rb][c].thread_elements()[1] * inv;
            }
        }
    }
    if (tid < uint(nf)) {
        const int fu = f0 + int(tid), r = fu / G, j = fu % G;
        const size_t row = size_t(((b * Hq + hkv * G + j) * L + r) * P + block);
        const float lv = l_sh[tid];
        part_l[row] = lv;
        part_m[row] = lv > 0.0f ? m_sh[tid] : -1.0e30f;
    }
    #undef VATT_UNROLL
"""


@lru_cache(maxsize=None)
def _kernel():
    return mx.fast.metal_kernel(
        name="mlx2_gqa_quantized_verify_attention",
        input_names=["q", "kw", "ks", "kb", "vw", "vs", "vb", "scale", "lpad"],
        output_names=["part_o", "part_l", "part_m"],
        source=_SOURCE,
        ensure_row_contiguous=False,
    )


# Host-side engagement counts (plain int bumps; read by qualification tracers
# to prove which path a served request actually ran).
STATS = {"verify_kernel_calls": 0, "verify_kernel_left_padded": 0,
         "composed_calls": 0}
STATS_BY_ROWS: dict[str, int] = {}


def note(path: str, rows: int, count: int = 1) -> None:
    STATS[path + "_calls"] = STATS.get(path + "_calls", 0) + count
    key = f"{path}_L{rows}"
    STATS_BY_ROWS[key] = STATS_BY_ROWS.get(key, 0) + count


# ------------------------------------------------------------------ planning


def fused_groups(rows: int, gqa: int) -> tuple[int, int]:
    """(groups, fused rows per group, padded to 8) for ``rows x gqa`` fused rows."""
    fused = rows * gqa
    groups = -(-fused // MAX_FUSED)
    per = -(-fused // groups)
    return groups, -(-per // 8) * 8


def verify_blocks(n: int, batch: int, kv_heads: int, groups: int) -> int:
    """History splits: about 512 threadgroups, at least 256 keys per split."""
    lanes = max(1, batch * kv_heads * groups)
    return max(1, min(-(-512 // lanes), max(1, n // 256)))


def split_ranges(n: int, left_padding: int, blocks: int) -> list[tuple[int, int]]:
    """The ``[start, end)`` key range each split owns (the kernel's arithmetic)."""
    span = max(0, n - left_padding)
    chunk = -(-span // blocks)
    out = []
    for block in range(blocks):
        start = left_padding + block * chunk
        out.append((start, min(n, start + chunk)))
    return out


def row_limit(n: int, rows: int, row: int) -> int:
    """Exclusive causal key limit of verify row ``row`` in an ``rows``-row block."""
    return n - rows + row + 1


# ------------------------------------------------------------- mask contract

# Masks built by batch caches' ``make_mask`` whose meaning is exactly
# "causal over the last L rows, keys below left_padding[b] invisible". The
# attention seam only sees an array; identity against this small registry
# proves where it came from without a device sync. Entries hold the mask
# itself, so an id cannot be reused while it is registered.
_CAUSAL_MASKS: dict[int, tuple] = {}
_REGISTRY_LIMIT = 16
NOT_CAUSAL = object()


def register_causal_mask(mask, left_padding, *, kind: str = "left_padded"):
    """Record that ``mask`` is causal with per-row ``left_padding`` (an mx array)."""
    if not isinstance(mask, mx.array):
        return mask
    if len(_CAUSAL_MASKS) >= _REGISTRY_LIMIT:
        _CAUSAL_MASKS.pop(next(iter(_CAUSAL_MASKS)))
    _CAUSAL_MASKS[id(mask)] = (mask, left_padding, kind)
    return mask


def lift_causal_mask(mask):
    """``mask[None, None]`` for a 2-D mask, carrying its registration.

    Indexing builds a new array, so a lifted registered mask would otherwise
    lose its provenance and look unregistered (non-plain) to consumers."""
    if not isinstance(mask, mx.array) or mask.ndim != 2:
        return mask
    lifted = mask[None, None, :, :]
    entry = _CAUSAL_MASKS.get(id(mask))
    if entry is not None and entry[0] is mask:
        register_causal_mask(lifted, entry[1], kind=entry[2])
    return lifted


def registered_mask_kind(mask) -> Optional[str]:
    entry = _CAUSAL_MASKS.get(id(mask)) if isinstance(mask, mx.array) else None
    if entry is None or entry[0] is not mask:
        return None
    return entry[2]


def causal_left_padding(mask, batch: int, rows: int, keys: int):
    """Left padding (int32, one per batch row) if ``mask`` is causal, else NOT_CAUSAL."""
    zeros = None
    if mask is None:
        return zeros if rows == 1 else NOT_CAUSAL
    if isinstance(mask, str):
        return zeros if mask == "causal" else NOT_CAUSAL
    entry = _CAUSAL_MASKS.get(id(mask))
    if entry is None or entry[0] is not mask or entry[2] != "left_padded":
        return NOT_CAUSAL
    left_padding = entry[1]
    if mask.shape[-2:] != (rows, keys):
        return NOT_CAUSAL
    if left_padding is None:
        return None
    if left_padding.ndim != 1 or left_padding.shape[0] != batch:
        return NOT_CAUSAL
    return left_padding


# ------------------------------------------------------------------- gating


def verify_attention_supported(queries, q_keys, q_values, *, group_size, key_bits,
                               value_bits) -> bool:
    """True when the kernel covers these shapes (the mask is checked separately)."""
    if mx.default_device() != mx.gpu:
        return False
    if len(q_keys) != 3 or len(q_values) != 3:  # normalized caches carry extra planes
        return False
    B, Hq, L, D = queries.shape
    Hkv = q_keys[0].shape[1]
    if not 1 <= L <= MAX_ROWS or Hq % Hkv or D not in (128, 256):
        return False
    if q_values[1].shape[-1] * group_size != D:  # value head dim == key head dim
        return False
    if group_size not in (32, 64, 128) or key_bits not in (2, 4, 8) or value_bits not in (2, 4, 8):
        return False
    return queries.dtype in (mx.bfloat16, mx.float16) and q_keys[1].dtype == queries.dtype


def use_verify_kernel(queries, q_keys, q_values, *, group_size, key_bits, value_bits) -> bool:
    """Supported, long enough to win, and not disabled by MLX2_QSDPA_VERIFY_KERNEL=0."""
    if not verify_kernel_enabled():
        return False
    if q_keys[0].shape[2] < MIN_CONTEXT:
        return False
    return verify_attention_supported(queries, q_keys, q_values, group_size=group_size,
                                      key_bits=key_bits, value_bits=value_bits)


def verify_kernel_enabled() -> bool:
    return os.environ.get("MLX2_QSDPA_VERIFY_KERNEL", "1") != "0"


# -------------------------------------------------------------------- kernel


def merge_partials(part_o, part_l, part_m):
    """Combine per-split normalized partials (..., P, D) into (..., D)."""
    top = part_m.max(axis=-1, keepdims=True)
    weight = part_l * mx.exp(part_m - top)
    return (weight[..., None] * part_o).sum(axis=-2) / weight.sum(axis=-1, keepdims=True)


def gqa_quantized_verify_attention(
    queries: mx.array,
    q_keys: tuple,
    q_values: tuple,
    *,
    scale: float,
    group_size: int = 64,
    key_bits: int = 8,
    value_bits: int = 8,
    left_padding: Optional[mx.array] = None,
    blocks: Optional[int] = None,
) -> mx.array:
    """``(B, Hq, L, D)`` causal attention of the last ``L`` positions over quantized K/V.

    Query row ``r`` sees keys ``[left_padding[b], N - L + r]``.
    """
    B, Hq, L, D = queries.shape
    Hkv, N = q_keys[0].shape[1], q_keys[0].shape[2]
    G = Hq // Hkv
    groups, per = fused_groups(L, G)
    blocks = blocks or verify_blocks(N, B, Hkv, groups)
    if left_padding is None:
        lpad = mx.zeros((B,), dtype=mx.int32)
    else:
        lpad = left_padding.astype(mx.int32)
        STATS["verify_kernel_left_padded"] += 1
    note("verify_kernel", L)
    part_o, part_l, part_m = _kernel()(
        inputs=[queries, *q_keys, *q_values, mx.array([scale], dtype=mx.float32), lpad],
        template=[("T", queries.dtype), ("D", D), ("G", G), ("L", L), ("KB", key_bits),
                  ("VB", value_bits), ("GS", group_size), ("FRB", per), ("FG", groups),
                  ("NS", SIMDGROUPS)],
        grid=(32 * SIMDGROUPS, B * Hkv * groups, blocks),
        threadgroup=(32 * SIMDGROUPS, 1, 1),
        output_shapes=[(B, Hq, L, blocks, D), (B, Hq, L, blocks), (B, Hq, L, blocks)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return merge_partials(part_o, part_l, part_m).astype(queries.dtype)


# ------------------------------------------------------ CPU reference of the plan


def reference_verify_attention(
    queries, q_keys, q_values, *, scale, group_size=64, key_bits=8, value_bits=8,
    left_padding=None, blocks=None,
):
    """The kernel's partition, fused-row mapping, causal limits and merge in MLX ops.

    Slow and CPU-friendly: it exists so the indexing logic the kernel encodes
    (split ranges, per-row limits, left padding, fused groups, partial merge)
    is testable without Metal against dequantize + SDPA with an explicit mask.
    """
    B, Hq, L, D = queries.shape
    Hkv, N = q_keys[0].shape[1], q_keys[0].shape[2]
    G = Hq // Hkv
    groups, per = fused_groups(L, G)
    blocks = blocks or verify_blocks(N, B, Hkv, groups)
    keys = mx.dequantize(*q_keys, group_size=group_size, bits=key_bits).astype(mx.float32)
    values = mx.dequantize(*q_values, group_size=group_size, bits=value_bits).astype(mx.float32)
    q = queries.astype(mx.float32) * scale
    pads = [0] * B if left_padding is None else [int(x) for x in left_padding.tolist()]
    rows_out = []
    for b in range(B):
        ranges = split_ranges(N, pads[b], blocks)
        per_head = []
        for hkv in range(Hkv):
            fused_out = [None] * (L * G)
            for fg in range(groups):
                for f in range(per):
                    fu = fg * per + f
                    if fu >= L * G:
                        continue
                    r, j = divmod(fu, G)
                    limit = row_limit(N, L, r)
                    po, pl, pm = [], [], []
                    for start, end in ranges:
                        end = min(end, limit)
                        if end <= start:
                            po.append(mx.zeros((D,)))
                            pl.append(mx.array(0.0))
                            pm.append(mx.array(-1.0e30))
                            continue
                        s = keys[b, hkv, start:end] @ q[b, hkv * G + j, r]
                        m = s.max()
                        p = mx.exp(s - m)
                        lsum = p.sum()
                        po.append((p @ values[b, hkv, start:end]) / lsum)
                        pl.append(lsum)
                        pm.append(m)
                    fused_out[fu] = merge_partials(mx.stack(po), mx.stack(pl), mx.stack(pm))
            per_head.append(fused_out)
        # (Hq, L, D) from fused rows (r, j) per KV head
        heads = []
        for hkv in range(Hkv):
            for j in range(G):
                heads.append(mx.stack([per_head[hkv][r * G + j] for r in range(L)]))
        rows_out.append(mx.stack(heads))
    return mx.stack(rows_out).astype(queries.dtype)
