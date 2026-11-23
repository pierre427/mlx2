# SPDX-License-Identifier: Apache-2.0
# The two Metal kernels are int8 variants of the split-K kernels in splitk.py
# (mlx2's port of jundot/omlx PR 5511, Apache-2.0): the tile math, bias,
# window, padding and sink handling are splitk.py's, only the K/V loads
# dequantize. The int8 KV mechanism (symmetric group-64 codes, FP16 scales,
# attention reading the codes directly) comes from g13-engine c91c0180 (MIT),
# mechanism only, no code copied. See docs/PROVENANCE.md and
# provenance/inkling2-kv-q8.json.
"""Opt-in int8 K/V for Inkling 2's global attention visits (``inkling2_kv_q8``).

Approximate (it changes outputs) and unqualified. Only the 8 global visits
(plain ``KVCache`` / ``BatchKVCache`` planes) are quantized; the sliding
window rings stay in the activation dtype.

Storage format v2 (one row per (batch, KV head, token)): ``D`` int8 codes
at byte 0, then the FP16 scale of each ``GROUP``-wide channel group at byte
``D`` (two little-endian bytes each), then zero padding to a multiple of 16
bytes: width ``packed_width(D)`` = 144 for the Mini's D = 128. Every row
starts 16-byte aligned, so the kernels load codes 16 bytes per thread and
scales as ``half``; a plane that is not last-axis contiguous or not 16-byte
aligned takes an in-kernel byte-load path with the same arithmetic, so it
is never misread. (v1, ``-kvq8``, packed rows at width 132 and was read a
byte at a time.) Symmetric: ``x ~= code * scale`` with
``scale = fp16(max|x_group| / 127)`` and ``code = clip(round(x / scale),
-127, 127)`` (0 when the scale is 0). Because codes and scales share one
array per K and V, the cache classes are unchanged: ``KVCache`` growth,
offset, ``trim`` (speculative rollback), ``BatchKVCache`` merge, extract,
filter and left padding, and APCv2 freeze/restore all move rows without
looking at their bytes. The format is named by the model's APCv2 layout
(``LAYOUT_SUFFIX``) and the adapter's numerics operation, so a Q8 entry is
never restored into a BF16 lane or the reverse; the attention also refuses a
plane in the other format. While the option is unqualified the adapter
refuses APCv2 reuse and publication of Q8 lanes outright
(``adapters/inkling2.py`` ``KV_Q8_PUBLICATION_QUALIFIED``).

Attention never dequantizes the whole prefix:

* GPU, 1..31 query rows (decode, DSpark verify, short prefill tails): the
  kernels below read the codes directly.
* GPU, 32 rows or more (prefill chunks): the same split-K kernel over
  31-row query tiles (``tiled_attention``), each tile at its own query
  offset; the cache is untouched.
* CPU or declined geometry: ``blockwise_attention`` dequantizes key blocks
  of at most ``KEY_BLOCK`` keys and combines them with an online softmax
  (the sink is its starting term).

``reference_attention`` (dequantize everything, then the composed path's
dense mask + sink column + SDPA) is the test oracle only; serving never
calls it, and ``full_prefix_dequant_calls()`` counts any call.

Precision: dequantized K/V tiles are staged in FP16 for BF16 queries (FP32
for FP32 queries), FP32 accumulation and softmax, as in splitk.py. The
L == 1 kernel keeps FP32 registers.
"""

import mlx.core as mx

from . import splitk

GROUP = 64
QMAX = 127.0
LAYOUT_SUFFIX = "-kvq8v2"
FORMAT = ("v2: symmetric int8 codes, group 64 along head_dim (head_dim if smaller), "
          "FP16 scales after the codes, rows zero-padded to 16 bytes")
ROW_ALIGN = 16
MAX_ROWS = splitk.MAX_ROWS
PREFILL_TILE = MAX_ROWS - 1  # query rows per kernel launch for long chunks
KEY_BLOCK = 4096  # keys dequantized at a time by the blockwise fallback
_SCORE_BUDGET = 64 << 20  # bytes of FP32 scores per blockwise pass
_FULL_PREFIX = [0]  # reference_attention calls (test oracle; 0 in serving)

_HEADER = r"""
// FP16 scale stored as two little-endian int8 bytes at row[at], row[at + 1]
// (byte path: any stride or alignment).
inline float q8_scale(const device int8_t* row, int at, int64_t stride) {
    ushort lo = ushort(uchar(row[at * stride]));
    ushort hi = ushort(uchar(row[(at + 1) * stride]));
    return float(as_type<half>(ushort(lo | (hi << 8))));
}

// Sixteen dequantized channels c0..c0+15 of row ``key`` (0 when !valid).
// vec: one 16-byte code load plus one half scale load (the row is 16-byte
// aligned and last-axis contiguous); otherwise byte loads. Both compute
// float(code) * float(scale).
inline void q8_row16(const device int8_t* base, int key, int64_t s2, int64_t s3,
                     int c0, int at, bool valid, bool vec, thread float* out) {
    if (!valid) {
        for (int i = 0; i < 16; ++i) out[i] = 0.0f;
        return;
    }
    const device int8_t* row = base + key * s2;
    if (vec) {
        const float sc = float(*(const device half*)(row + at));
        const int4 w = *(const device int4*)(row + c0);
        const char4 c[4] = {as_type<char4>(w.x), as_type<char4>(w.y),
                            as_type<char4>(w.z), as_type<char4>(w.w)};
        for (int j = 0; j < 4; ++j) {
            out[4 * j] = float(c[j].x) * sc;
            out[4 * j + 1] = float(c[j].y) * sc;
            out[4 * j + 2] = float(c[j].z) * sc;
            out[4 * j + 3] = float(c[j].w) * sc;
        }
    } else {
        const float sc = q8_scale(row, at, s3);
        for (int i = 0; i < 16; ++i) out[i] = float(row[(c0 + i) * s3]) * sc;
    }
}

inline bool q8_vec_ok(const device int8_t* kb, const device int8_t* vb,
                      int64_t ks2, int64_t ks3, int64_t vs2, int64_t vs3) {
    return ks3 == 1 && vs3 == 1 && (ks2 % 16) == 0 && (vs2 % 16) == 0
        && ((reinterpret_cast<ulong>(kb) & 15ul) == 0)
        && ((reinterpret_cast<ulong>(vb) & 15ul) == 0);
}
"""

# splitk._PART_SOURCE with int8 K/V rows; only the tile loads differ.
_PART_SOURCE = r"""
    constexpr int BK = 16;
    constexpr int NT = NSG * 32;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int sp = threadgroup_position_in_grid.x;
    const int hkv_qt = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;

    const int L = dims[0];
    const int S = dims[1];
    const int NS = dims[2];
    const int OFF = dims[3];
    const int lp = max(dims[4 + b], 0);

    const int n_rows = G * L;
    const int BQ = NSG * 8;
    const int nqt = (n_rows + BQ - 1) / BQ;
    const int hkv = hkv_qt / nqt;
    const int qt = hkv_qt % nqt;

    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int packed_row = qt * BQ + int(sg) * 8 + fm;
    const bool row_ok = packed_row < n_rows;
    const int g_idx = row_ok ? (packed_row / L) : 0;
    const int q_pos = row_ok ? (packed_row % L) : 0;
    const int h = hkv * G + g_idx;
    const int a_row = OFF + q_pos;
    const float qscale = scale[0] * LOG2E;

    const int a_first = OFF;
    const int a_last = min(OFF + L, S) - 1;
    int lo = WINDOW > 0 ? max(0, a_first - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int nk = max(0, a_last + 1 - lo);
    const int per_sp = ((nk + NS - 1) / NS + BK - 1) / BK * BK;
    const int k_start = lo + sp * per_sp;
    const int k_end = min(a_last + 1, k_start + per_sp);

    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
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
    float m = (SINGLE && row_ok) ? float(sink[(b * L + q_pos) * HQ + h]) * LOG2E : -INFINITY;
    float l = SINGLE ? 1.0f : 0.0f;
    const device T* relp = rel + ((size_t)(b * L + q_pos) * HQ + h) * R;

    const device int8_t* kb_ptr = k + b * k_strides[0] + hkv * k_strides[1];
    const device int8_t* vb_ptr = v + b * v_strides[0] + hkv * v_strides[1];
    const int64_t ks2 = k_strides[2];
    const int64_t vs2 = v_strides[2];
    const int64_t ks3 = k_strides[3];
    const int64_t vs3 = v_strides[3];
    const bool vec = q8_vec_ok(kb_ptr, vb_ptr, ks2, ks3, vs2, vs3);

    for (int k0 = k_start; k0 < k_end; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // 16 channels per thread per row: one 16-byte code load each.
        for (int c = int(tid); c < BK * (D / 16); c += NT) {
            const int r = c / (D / 16);
            const int col = (c % (D / 16)) * 16;
            const int key = k0 + r;
            const int at = D + 2 * (col / GROUP);
            float kv[16];
            float vv[16];
            q8_row16(kb_ptr, key, ks2, ks3, col, at, key < k_end, vec, kv);
            q8_row16(vb_ptr, key, vs2, vs3, col, at, key < k_end, vec, vv);
            for (int j = 0; j < 16; ++j) {
                Ks[(col + j) * LDK + r] = ST(kv[j]);
                Vs[r * LDV + col + j] = ST(vv[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (qt * BQ + int(sg) * 8 >= n_rows) {
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
        if (WINDOW == 0 && k0 >= lp && k0 + BK <= k_end && k0 + BK <= a_first - R + 1) {
            for (int kb = 0; kb < BK / 8; ++kb) {
                float s0 = row_ok ? Sf[kb].thread_elements()[0] * qscale : -INFINITY;
                float s1 = row_ok ? Sf[kb].thread_elements()[1] * qscale : -INFINITY;
                Sf[kb].thread_elements()[0] = s0;
                Sf[kb].thread_elements()[1] = s1;
                rmax = max(rmax, max(s0, s1));
            }
        } else {
            for (int kb = 0; kb < BK / 8; ++kb) {
                for (int e = 0; e < 2; ++e) {
                    const int key = k0 + kb * 8 + fn + e;
                    const int d = a_row - key;
                    const bool vis = row_ok && key < k_end && d >= 0 && key >= lp
                        && (WINDOW == 0 || d < WINDOW);
                    float s = Sf[kb].thread_elements()[e] * qscale;
                    if (vis) {
                        if (d < R) {
                            s += float(relp[d]) * LOG2E;
                        }
                    } else {
                        s = -INFINITY;
                    }
                    Sf[kb].thread_elements()[e] = s;
                    rmax = max(rmax, s);
                }
            }
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const float m_new = max(m, rmax);
        const float factor = (m == -INFINITY || m_new == -INFINITY)
            ? 0.0f : fast::exp2(m - m_new);
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
        for (int db = 0; db < D / 8; ++db) {
            Of[db].thread_elements()[0] *= factor;
            Of[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_matrix<ST, 8, 8> Vm;
                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vm, Of[db]);
            }
        }
    }

    if (row_ok) {
        if (SINGLE) {
            const float inv = 1.0f / l;
            size_t base_o = (((size_t)b * HQ + h) * L + q_pos) * D;
            for (int db = 0; db < D / 8; ++db) {
                out_o[base_o + db * 8 + fn] = OT(Of[db].thread_elements()[0] * inv);
                out_o[base_o + db * 8 + fn + 1] = OT(Of[db].thread_elements()[1] * inv);
            }
        } else {
            size_t base_ml = (((size_t)b * HQ + h) * L + q_pos) * NS + sp;
            if (fn == 0) {
                out_m[base_ml] = m;
                out_l[base_ml] = l;
            }
            size_t base_o = base_ml * D;
            for (int db = 0; db < D / 8; ++db) {
                out_o[base_o + db * 8 + fn] = OT(Of[db].thread_elements()[0]);
                out_o[base_o + db * 8 + fn + 1] = OT(Of[db].thread_elements()[1]);
            }
        }
    }
"""

# One-row decode (L == 1), register streaming, after splitk._DIRECT_SOURCE
# but with 16-byte rows: LPK = D / 16 lanes share one key (16 channels each,
# one 16-byte code load), so a simdgroup streams KPS = 32 / LPK keys per
# step. Each key slot keeps its own online softmax; the slots are merged by
# shuffles, then the simdgroups through threadgroup memory, and the splits by
# splitk's merge kernel. The softmax partition differs from v1's (one key per
# simdgroup step): near-exact, not bitwise, against it.
_DIRECT_SOURCE = r"""
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int CPL = 16;
    constexpr int LPK = D / CPL;
    constexpr int KPS = 32 / LPK;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int sp = threadgroup_position_in_grid.x;
    const int hkv = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;

    const int S = dims[0];
    const int NS = dims[1];
    const int OFF = dims[2];
    const int lp = max(dims[3 + b], 0);
    const float qscale = scale[0] * LOG2E;

    const int a_row = OFF;
    const int a_last = min(a_row, S - 1);
    int lo = WINDOW > 0 ? max(0, a_row - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int nk = max(0, a_last + 1 - lo);
    const int per_sp = (nk + NS - 1) / NS;
    const int k_start = lo + sp * per_sp;
    const int k_end = min(a_last + 1, k_start + per_sp);

    const int slot = int(lane) / LPK;
    const int c0 = (int(lane) % LPK) * CPL;
    const int at = D + 2 * (c0 / GROUP);
    float qv[G][CPL];
    float ov[G][CPL];
    float mv[G];
    float lv[G];
    const device T* relp[G];
    for (int g_idx = 0; g_idx < G; ++g_idx) {
        int h = hkv * G + g_idx;
        const device T* qp = q + b * q_strides[0] + h * q_strides[1] + c0 * q_strides[3];
        for (int i = 0; i < CPL; ++i) {
            qv[g_idx][i] = float(qp[i * q_strides[3]]) * qscale;
            ov[g_idx][i] = 0.0f;
        }
        mv[g_idx] = -INFINITY;
        lv[g_idx] = 0.0f;
        relp[g_idx] = rel + ((size_t)b * HQ + h) * R;
    }

    const device int8_t* kb_ptr = k + b * k_strides[0] + hkv * k_strides[1];
    const device int8_t* vb_ptr = v + b * v_strides[0] + hkv * v_strides[1];
    const int64_t ks2 = k_strides[2];
    const int64_t vs2 = v_strides[2];
    const int64_t ks3 = k_strides[3];
    const int64_t vs3 = v_strides[3];
    const bool vec = q8_vec_ok(kb_ptr, vb_ptr, ks2, ks3, vs2, vs3);

    // Uniform trip count per simdgroup (the shuffles need every lane).
    for (int base = k_start + int(sg) * KPS; base < k_end; base += NSG * KPS) {
        const int key = base + slot;
        const bool valid = key < k_end;
        float kv[CPL];
        float vv[CPL];
        q8_row16(kb_ptr, key, ks2, ks3, c0, at, valid, vec, kv);
        q8_row16(vb_ptr, key, vs2, vs3, c0, at, valid, vec, vv);
        const int d = a_row - key;
        for (int g_idx = 0; g_idx < G; ++g_idx) {
            float dot = 0.0f;
            for (int i = 0; i < CPL; ++i) dot += kv[i] * qv[g_idx][i];
            for (int off = 1; off < LPK; off <<= 1) dot += simd_shuffle_xor(dot, off);
            if (valid) {
                if (d < R) dot += float(relp[g_idx][d]) * LOG2E;
                float m_new = max(mv[g_idx], dot);
                float f_old = (mv[g_idx] == -INFINITY) ? 0.0f : fast::exp2(mv[g_idx] - m_new);
                float p = fast::exp2(dot - m_new);
                lv[g_idx] = lv[g_idx] * f_old + p;
                mv[g_idx] = m_new;
                for (int i = 0; i < CPL; ++i) {
                    ov[g_idx][i] = ov[g_idx][i] * f_old + p * vv[i];
                }
            }
        }
    }
    // Merge the key slots (lanes with the same channels differ in slot bits).
    for (int off = LPK; off < 32; off <<= 1) {
        for (int g_idx = 0; g_idx < G; ++g_idx) {
            const float mo = simd_shuffle_xor(mv[g_idx], off);
            const float lo2 = simd_shuffle_xor(lv[g_idx], off);
            const float mn = max(mv[g_idx], mo);
            const float a0 = (mv[g_idx] == -INFINITY) ? 0.0f : fast::exp2(mv[g_idx] - mn);
            const float a1 = (mo == -INFINITY) ? 0.0f : fast::exp2(mo - mn);
            lv[g_idx] = lv[g_idx] * a0 + lo2 * a1;
            mv[g_idx] = mn;
            for (int i = 0; i < CPL; ++i) {
                const float oo = simd_shuffle_xor(ov[g_idx][i], off);
                ov[g_idx][i] = ov[g_idx][i] * a0 + oo * a1;
            }
        }
    }

    threadgroup float tg_m[NSG][G];
    threadgroup float tg_l[NSG][G];
    threadgroup float tg_o[NSG][G][D];
    if (slot == 0) {
        for (int g_idx = 0; g_idx < G; ++g_idx) {
            if (lane == 0) {
                tg_m[sg][g_idx] = mv[g_idx];
                tg_l[sg][g_idx] = lv[g_idx];
            }
            for (int i = 0; i < CPL; ++i) {
                tg_o[sg][g_idx][c0 + i] = ov[g_idx][i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && slot == 0) {
        for (int g_idx = 0; g_idx < G; ++g_idx) {
            int h = hkv * G + g_idx;
            float m = -INFINITY;
            float l = 0.0f;
            float oo[CPL];
            for (int i = 0; i < CPL; ++i) oo[i] = 0.0f;
            for (int s = 0; s < NSG; ++s) {
                float ls = tg_l[s][g_idx];
                if (ls == 0.0f) continue;
                float ms = tg_m[s][g_idx];
                float mn = max(m, ms);
                float a0 = (m == -INFINITY) ? 0.0f : fast::exp2(m - mn);
                float a1 = fast::exp2(ms - mn);
                l = l * a0 + ls * a1;
                m = mn;
                for (int i = 0; i < CPL; ++i) {
                    oo[i] = oo[i] * a0 + tg_o[s][g_idx][c0 + i] * a1;
                }
            }
            size_t base_ml = ((size_t)b * HQ + h) * NS + sp;
            if (lane == 0) {
                out_m[base_ml] = m;
                out_l[base_ml] = l;
            }
            size_t base_o = base_ml * D;
            for (int i = 0; i < CPL; ++i) {
                out_o[base_o + c0 + i] = oo[i];
            }
        }
    }
"""

_kernels = {}


def _kernel(name, inputs, outputs, source):
    kernel = _kernels.get(name)
    if kernel is None:
        kernel = _kernels[name] = mx.fast.metal_kernel(
            name=f"inkling2_kv_q8_{name}", input_names=inputs, output_names=outputs,
            source=source, header=_HEADER, ensure_row_contiguous=False,
        )
    return kernel


# ----- format -----


def group_size(head_dim):
    """The scale group: 64 channels, or the whole row when it is narrower."""
    return GROUP if head_dim >= GROUP else int(head_dim)


def geometry_reason(head_dim):
    """None when ``head_dim`` has a Q8 layout, else why not."""
    if type(head_dim) is not int or head_dim <= 0 or head_dim % 2:
        return "head_dim"
    if head_dim % group_size(head_dim):
        return "group"
    return None


def packed_width(head_dim):
    """Row bytes: codes, FP16 scales, zero padding to ``ROW_ALIGN``."""
    raw = head_dim + 2 * (head_dim // group_size(head_dim))
    return -(-raw // ROW_ALIGN) * ROW_ALIGN


def pack(x):
    """[..., D] floats -> [..., D + 2 D / group] int8 (codes, then FP16 scale bytes)."""
    *lead, dim = x.shape
    group = group_size(dim)
    n = dim // group
    xf = x.astype(mx.float32).reshape(*lead, n, group)
    amax = mx.max(mx.abs(xf), axis=-1, keepdims=True)
    scale = (amax / QMAX).astype(mx.float16)
    sf = scale.astype(mx.float32)
    live = sf > 0
    codes = mx.where(live, mx.clip(mx.round(xf / mx.where(live, sf, 1.0)), -QMAX, QMAX), 0.0)
    scale_bytes = mx.contiguous(scale.reshape(*lead, n)).view(mx.int8)
    parts = [codes.astype(mx.int8).reshape(*lead, dim), scale_bytes]
    spare = packed_width(dim) - dim - 2 * n
    if spare:
        parts.append(mx.zeros((*lead, spare), mx.int8))
    return mx.concatenate(parts, axis=-1)


def unpack(packed, head_dim):
    """(int8 codes [..., D], FP16 scales [..., D / group]) of packed rows."""
    if packed.dtype != mx.int8 or packed.shape[-1] != packed_width(head_dim):
        raise ValueError(
            f"Inkling 2 kv_q8 rows must be int8 of width {packed_width(head_dim)}, got "
            f"{packed.dtype} width {packed.shape[-1]}")
    n = head_dim // group_size(head_dim)
    codes = packed[..., :head_dim]
    scales = mx.contiguous(packed[..., head_dim:head_dim + 2 * n]).view(mx.float16)
    return codes, scales


def dequantize(packed, head_dim, dtype):
    codes, scales = unpack(packed, head_dim)
    *lead, _ = codes.shape
    group = group_size(head_dim)
    out = (codes.astype(mx.float32).reshape(*lead, head_dim // group, group)
           * scales.astype(mx.float32)[..., None])
    return out.reshape(*lead, head_dim).astype(dtype)


def roundtrip(x):
    """``dequantize(pack(x))`` in x's dtype (the stored value of x)."""
    return dequantize(pack(x), x.shape[-1], x.dtype)


def kv_bytes(cache_planes):
    """{"q8": bytes, "other": bytes} held by int8 and other K/V buffers."""
    out = {"q8": 0, "other": 0}
    for plane in cache_planes:
        for name in ("keys", "values"):
            value = getattr(plane, name, None)
            if isinstance(value, mx.array):
                out["q8" if value.dtype == mx.int8 else "other"] += value.nbytes
    return out


# ----- attention -----


def decline_reason(q, k, v, *, n_heads, n_kv, head_dim, device=None):
    """None when ``q8_attention`` can run these arrays, else a reason."""
    if not 1 <= q.shape[2] < MAX_ROWS:
        return "rows"
    if (mx.default_device() if device is None else device) != mx.gpu:
        return "cpu"
    if q.dtype not in (mx.bfloat16, mx.float32):
        return "dtype"
    if k.dtype != mx.int8 or v.dtype != mx.int8:
        return "cache_format"
    width = packed_width(head_dim)
    if k.shape[-1] != width or v.shape[-1] != width:
        return "cache_format"
    if head_dim % 32 or head_dim > 128 or n_heads % n_kv:
        return "geometry"
    # 16-channel loads must stay inside one scale group, and D / 16 lanes
    # per key must tile a simdgroup (D in 32, 64, 128).
    if geometry_reason(head_dim) or group_size(head_dim) % 16 or 32 % (head_dim // 16):
        return "geometry"
    return None


def q8_attention(q, k, v, rel, sinks, *, offset, left_padding, window, scale):
    """``splitk.gqa_splitk_attention`` over packed int8 K/V.

    q [B, H, L, D]; k, v [B, Hk, S, D + 2 D / group] int8 rows (any strides);
    rel [B, L, H, E]; sinks [B, L, H]; queries at key positions
    ``offset + i``. Returns [B, H, L, D] in q's dtype. GPU only; check
    ``decline_reason`` first.
    """
    b, hq, length, dim = q.shape
    hkv, size = k.shape[1], k.shape[2]
    g = hq // hkv
    group = group_size(dim)
    extent = rel.shape[-1]
    window = int(window)
    offset = int(offset)
    pad = (mx.zeros((b,), mx.int32) if left_padding is None
           else mx.maximum(left_padding, 0).astype(mx.int32))
    sc = mx.array([float(scale)], dtype=mx.float32)
    sink = mx.contiguous(sinks.astype(mx.float32))
    rel = mx.contiguous(rel.astype(q.dtype))
    n_keys = min(size, offset + length)
    if window:
        n_keys = min(n_keys, window + length - 1)
    merge = splitk._kernel("merge", ["part_o", "part_m", "part_l", "sink"], ["out"],
                           splitk._MERGE_SOURCE)

    def merged(po, pm, pl, ns):
        return merge(
            inputs=[po, pm, pl, sink],
            template=[("T", q.dtype), ("D", dim), ("HQ", hq), ("L", length), ("NS", ns)],
            grid=(b * hq * length * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(b, hq, length, dim)],
            output_dtypes=[q.dtype],
        )[0]

    if length == 1 and g <= splitk._DIRECT_MAX_GROUP and n_keys >= splitk._DIRECT_MIN_KEYS:
        direct = _kernel("direct", ["q", "k", "v", "rel", "dims", "scale"],
                         ["out_o", "out_m", "out_l"], _DIRECT_SOURCE)
        nsg = 4
        ns = splitk._splits(n_keys, b * hkv, direct=True)
        dims = mx.concatenate([mx.array([size, ns, offset], dtype=mx.int32), pad])
        po, pm, pl = direct(
            inputs=[q, k, v, rel, dims, sc],
            template=[("T", q.dtype), ("D", dim), ("HQ", hq), ("G", g), ("R", extent),
                      ("WINDOW", window), ("NSG", nsg), ("GROUP", group)],
            grid=(ns * nsg * 32, hkv, b),
            threadgroup=(nsg * 32, 1, 1),
            output_shapes=[(b, hq, 1, ns, dim), (b, hq, 1, ns), (b, hq, 1, ns)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        return merged(po, pm, pl, ns)

    part = _kernel("part", ["q", "k", "v", "rel", "sink", "dims", "scale"],
                   ["out_o", "out_m", "out_l"], _PART_SOURCE)
    n_rows = g * length
    nsg = min(4, max(1, (n_rows + 7) // 8))
    nqt = (n_rows + nsg * 8 - 1) // (nsg * 8)
    ns = splitk._splits(n_keys, b * hkv * nqt, direct=False)
    dims = mx.concatenate([mx.array([length, size, ns, offset], dtype=mx.int32), pad])
    template = [
        ("T", q.dtype), ("ST", mx.float32 if q.dtype == mx.float32 else mx.float16),
        ("OT", q.dtype if ns == 1 else mx.float32), ("D", dim), ("HQ", hq), ("G", g),
        ("R", extent), ("WINDOW", window), ("NSG", nsg), ("SINGLE", int(ns == 1)),
        ("GROUP", group),
    ]
    nt = nsg * 32
    if ns == 1:
        return part(
            inputs=[q, k, v, rel, sink, dims, sc], template=template,
            grid=(nt, hkv * nqt, b), threadgroup=(nt, 1, 1),
            output_shapes=[(b, hq, length, dim), (1,), (1,)],
            output_dtypes=[q.dtype, mx.float32, mx.float32],
        )[0]
    po, pm, pl = part(
        inputs=[q, k, v, rel, sink, dims, sc], template=template,
        grid=(ns * nt, hkv * nqt, b), threadgroup=(nt, 1, 1),
        output_shapes=[(b, hq, length, ns, dim), (b, hq, length, ns), (b, hq, length, ns)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return merged(po, pm, pl, ns)


def _count(counts, key, n=1):
    if counts is not None:
        counts[key] = counts.get(key, 0) + n


def tiled_attention(q, k, v, rel, sinks, *, offset, left_padding, window, scale, counts=None):
    """``q8_attention`` over ``PREFILL_TILE``-row query tiles (prefill chunks).

    Tile ``a`` holds queries at key positions ``offset + a ..``; its bias rows
    and sinks are the tile's own, so the law is the per-row law unchanged.
    Each tile reads the int8 K/V in place. GPU only."""
    length = q.shape[2]
    outs = []
    for a in range(0, length, PREFILL_TILE):
        e = min(length, a + PREFILL_TILE)
        outs.append(q8_attention(
            q[:, :, a:e], k, v, rel[:, a:e], sinks[:, a:e], offset=int(offset) + a,
            left_padding=left_padding, window=window, scale=scale))
        _count(counts, "prefill_tiles")
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


def blockwise_attention(q, k, v, rel, sinks, *, offset, left_padding, window, scale,
                        block=None, counts=None):
    """The fallback (CPU, declined geometry): the composed path's law over
    key blocks of at most ``block`` (default ``KEY_BLOCK``) keys.

    Each block's codes are dequantized to FP32, scored with the relative
    bias (distances below E), causal/window/left-padding masks, and merged
    by an online softmax that starts from the sink (logit ``sink``, value
    0). Query rows are processed in passes bounded by ``_SCORE_BUDGET``.
    FP32 throughout; the result is cast to q's dtype."""
    block = int(block or KEY_BLOCK)
    b, hq, length, dim = q.shape
    hkv, size = k.shape[1], k.shape[2]
    g = hq // hkv
    extent = rel.shape[-1]
    offset, window = int(offset), int(window)
    pad = None if left_padding is None else mx.maximum(left_padding, 0).astype(mx.int32)
    rows_per_pass = max(1, _SCORE_BUDGET // max(1, b * hq * block * 4))
    outs = []
    for r0 in range(0, length, rows_per_pass):
        r1 = min(length, r0 + rows_per_pass)
        n = r1 - r0
        qf = (q[:, :, r0:r1].astype(mx.float32) * float(scale)).reshape(b, hkv, g, n, dim)
        rel_t = rel[:, r0:r1].astype(mx.float32).transpose(0, 2, 1, 3)  # [B, H, n, E]
        m = sinks[:, r0:r1].astype(mx.float32).transpose(0, 2, 1)  # [B, H, n]
        den = mx.ones_like(m)
        acc = mx.zeros((b, hq, n, dim), mx.float32)
        first = offset + r0
        last = min(size, offset + r1) - 1
        lo = max(0, first - window + 1) if window else 0
        positions = mx.arange(first, first + n)[:, None]
        for s0 in range(lo, last + 1, block):
            s1 = min(last + 1, s0 + block)
            kb = dequantize(k[:, :, s0:s1], dim, mx.float32)
            vb = dequantize(v[:, :, s0:s1], dim, mx.float32)
            _count(counts, "block_dequant_calls")
            scores = (qf @ kb[:, :, None].swapaxes(-1, -2)).reshape(b, hq, n, s1 - s0)
            dist = positions - mx.arange(s0, s1)[None, :]
            if first - (s1 - 1) < extent:  # some (row, key) inside the bias band
                idx = mx.broadcast_to(mx.clip(dist, 0, extent - 1)[None, None],
                                      (b, hq, n, s1 - s0))
                bias = mx.take_along_axis(rel_t, idx, axis=-1)
                scores = scores + mx.where((dist < extent)[None, None], bias, 0.0)
            invalid = dist < 0
            if window:
                invalid = invalid | (dist >= window)
            invalid = invalid[None, None]
            if pad is not None:
                invalid = invalid | (mx.arange(s0, s1)[None, :] < pad[:, None])[:, None, None, :]
            scores = mx.where(invalid, -mx.inf, scores)
            m_new = mx.maximum(m, scores.max(axis=-1))
            p = mx.exp(scores - m_new[..., None])
            alpha = mx.exp(m - m_new)
            den = den * alpha + p.sum(axis=-1)
            pv = (p.reshape(b, hkv, g, n, s1 - s0) @ vb[:, :, None]).reshape(b, hq, n, dim)
            acc = acc * alpha[..., None] + pv
            m = m_new
        outs.append(acc / den[..., None])
    out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
    return out.astype(q.dtype)


def full_prefix_dequant_calls():
    """Calls of the whole-prefix dequantize (``reference_attention``); a
    serving path never makes one."""
    return _FULL_PREFIX[0]


def reference_attention(q, k, v, rel, sinks, **kwargs):
    """TEST ORACLE ONLY (serving never calls it): dequantize the whole K/V,
    then the composed path's dense-mask + sink-column SDPA
    (``splitk.reference_attention``). Counted (``full_prefix_dequant_calls``)."""
    _FULL_PREFIX[0] += 1
    dim = q.shape[-1]
    return splitk.reference_attention(
        q, dequantize(k, dim, q.dtype), dequantize(v, dim, q.dtype), rel, sinks, **kwargs)
