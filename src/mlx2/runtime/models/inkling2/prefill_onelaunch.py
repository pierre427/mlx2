"""Inkling 2 one-launch prompt attention (``inkling2_prefill_onelaunch_attention``).

Original mlx2 code (2026-11-23), a clean-room implementation of
``spec-prefill-attention.md`` (see provenance/inkling2-prefill-onelaunch.json).
Default off, unqualified.

One GPU launch per (layer visit, prefill chunk) computes Inkling 2's prompt
attention directly from the cache: Q, K, V, the relative-bias profiles and the
sink logits are read in place through their strides; no bias plane, mask or
concatenated sink row is built. For query row i at logical key index
``p = offset + i``::

    d = p - j;  visible = d >= 0 and j >= pad[b] and (W == 0 or d < W)
    logit(i, j) = scale * q_i . k_j + (rel[b, i, h, d] if d < E)
    out_i = sum_vis exp(logit - m) v_j / (sum_vis exp(logit - m) + exp(sink_i - m))

Algorithm: FlashAttention-2 style key tiles with an online softmax in the
log2 domain, started from the sink (running max = sink logit, running sum 1,
accumulator 0). Everything after the 16-bit loads is FP32: Q, K, V widen
exactly into FP32 operands, the probabilities stay FP32 through the P.V
product, and the output rounds once.

Chunk-width invariance (bitwise): key tiles of ``BK`` keys start at
multiples of ``BK`` in each sequence's own token frame (origin at its first
real token, so after left padding and independent of any window crop or ring
rotation), are visited in ascending order, and a tile a row cannot see is an
exact identity on that row's (max, sum, accumulator): the max is unchanged,
the rescale factor is exactly 1 and the added terms are exactly zero. The
bias-free fast path (tiles wholly visible and at distance >= E from every
row of a simdgroup) adds nothing where the general path adds nothing, so its
logits are the general path's. A row's result therefore depends only on its
absolute position and the data, not on the chunk, its place in the chunk, or
the rows sharing its threadgroup. Keys are never split across threadgroups.

Two kernel forms compute the same function and differ only in rounding order:

- ``simd``: Metal ``simdgroup_matrix`` 8x8 FP32 multiply-accumulate (every
  Apple GPU); one simdgroup per (query head, 8 query positions).
- ``tensor``: Metal 4 tensor operations (``MetalPerformancePrimitives``
  ``matmul2d``, written against Apple's public API, no MLX NAX header) on GPU
  generation 17+ (M5 class) with macOS 26.2+; one simdgroup per (query head,
  16 query positions). S and P pass through threadgroup memory so the softmax
  never depends on the implementation-defined cooperative-tensor layout.

The G query heads of one KV head run in one threadgroup, so each staged K/V
tile is read once for all of them.

Ring reads (``ring=(rot, size)``) are exercised by synthetic inputs only. On
the model path a prompt chunk (two rows or more) appends through the rotating
caches' ``_update_concat``, which leaves the buffer chronological, so
``Attention._ring_layout`` reports rotation 0 there and the route reads the
buffer with ``rot`` = the window crop start only; the wrapped (rot > 0,
across-the-end) read happens only in the synthetic ring tests and the
first-use self-check.

Tensor form, partial tiles: a simdgroup whose block has fewer than 16 query
rows (the chunk's last block) runs ``matmul2d`` on a Q tensor with fewer rows,
Apple's edge-checked path. Bitwise chunk invariance of those rows is to be
shown on M5 (Metal test N3 includes widths that produce partial blocks); the
simd form has no edge path (rows past the chunk are zero operands).
"""

from __future__ import annotations

import os
import platform
import re

import mlx.core as mx

FORMS = ("simd", "tensor")
BLOCK_KEYS = 32  # BK: part of the arithmetic (tile boundaries); fixed per process
ROWS = {"simd": 8, "tensor": 16}  # query positions per simdgroup
SIMDGROUP_TARGET = 6  # simdgroups per threadgroup aimed for (G * position blocks)
THREADGROUP_MEMORY_LIMIT = 32768
THREAD_LIMIT = 1024
# Policy (unmeasured): calls whose key count is below this decline to the
# composed path. 0 until the A/B (spec N11) measures the crossover.
MIN_KEYS = 0
LOG2E = 1.4426950408889634
SINK_CLAMP = 1e30  # sinks are clamped to +-1e30 (finite: no inf arithmetic in-kernel)

TOLERANCE = {mx.bfloat16: 3.5e-3, mx.float16: 5e-4}  # N1 relative RMS vs float64


# --------------------------------------------------------------------------
# Kernel sources
# --------------------------------------------------------------------------

_HEADER_SIMD = "#include <metal_simdgroup>\nusing namespace metal;\n"
_HEADER_TENSOR = (
    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
    "using namespace metal;\nusing namespace mpp::tensor_ops;\n"
)

# Shared prologue: indices, the threadgroup's key range, the aligned first
# tile. Template constants: T (q/k/v/rel dtype), TS (sink dtype), D, HQ, HKV,
# E, WINDOW, NPB (position blocks per threadgroup), BK, RS (positions per
# simdgroup), NO_SKIP / NO_FAST (debug builds: visit every tile of the
# buffer / never take the bias-free path).
_PROLOGUE = r"""
    constexpr int G = HQ / HKV;
    constexpr int NSG = G * NPB;
    constexpr int NT = NSG * 32;
    constexpr int PT = NPB * RS;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int tile = int(threadgroup_position_in_grid.x);
    const int hkv = int(threadgroup_position_in_grid.y);
    const int b = int(threadgroup_position_in_grid.z);
    const int L = dims[0];      // query rows
    const int NK = dims[1];     // logical keys
    const int OFF = dims[2];    // logical key index of query row 0
    const int ROT = dims[3];    // logical key j -> physical row (j + ROT) % PHYS
    const int PHYS = dims[4];
    const int pad = max(pads[b], 0);   // keys j < pad are masked
    const int org = origins[b];        // logical index of the tile grid origin
    const int member = int(sg) % G;
    const int pblk = int(sg) / G;
    const int h = hkv * G + member;
    const int p_tg0 = tile * PT;
    const int p_tg1 = min(p_tg0 + PT - 1, L - 1);
    const int p_sg0 = p_tg0 + pblk * RS;
    const int p_sg1 = min(p_sg0 + RS - 1, L - 1);
    const bool sg_live = p_sg0 < L;
    const int sg_a0 = OFF + p_sg0;
    const int sg_a1 = OFF + p_sg1;
    int sg_lo = pad;
    if (WINDOW > 0) {
        sg_lo = max(sg_lo, sg_a0 - WINDOW + 1);
    }
    // Keys the threadgroup visits: [first key its earliest row can see,
    // its latest row's own key]; the debug build visits the whole buffer.
    int lo = pad;
    if (WINDOW > 0) {
        lo = max(lo, OFF + p_tg0 - WINDOW + 1);
    }
    int hi = OFF + p_tg1;
    if (NO_SKIP) {
        lo = 0;
        hi = NK - 1;
    }
    // Tile starts sit at org + n * BK (floor division: org may be negative).
    const int rel_lo = lo - org;
    const int n_lo = rel_lo >= 0 ? rel_lo / BK : -((-rel_lo + BK - 1) / BK);
    const int k_first = org + n_lo * BK;
    const float qscale = scale[0] * LOG2E;
    const device T* kb_ptr = k + b * k_strides[0] + hkv * k_strides[1];
    const device T* vb_ptr = v + b * v_strides[0] + hkv * v_strides[1];
    const bool vec_kv = k_strides[3] == 1 && v_strides[3] == 1
        && (k_strides[2] % 8) == 0 && (v_strides[2] % 8) == 0
        && ((reinterpret_cast<ulong>(kb_ptr) & 15ul) == 0)
        && ((reinterpret_cast<ulong>(vb_ptr) & 15ul) == 0);
"""

# Stage keys k0 .. k0 + BK - 1 into Ks / Vs ([key][LDS], raw 16-bit values).
# Keys outside [pad, min(hi, NK - 1)] are zero; they are masked for every row.
_STAGE = r"""
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = int(tid); c < BK * (D / 8); c += NT) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int j = k0 + r;
            threadgroup T* kd = Ks + r * LDS + col;
            threadgroup T* vd = Vs + r * LDS + col;
            if (j >= pad && j <= hi && j < NK) {
                int phys = j + ROT;
                if (phys >= PHYS) {
                    phys -= PHYS;
                }
                const device T* ks = kb_ptr + (int64_t)phys * k_strides[2];
                const device T* vs = vb_ptr + (int64_t)phys * v_strides[2];
                if (vec_kv) {
                    *(threadgroup uint4*)kd = *(const device uint4*)(ks + col);
                    *(threadgroup uint4*)vd = *(const device uint4*)(vs + col);
                } else {
                    for (int e = 0; e < 8; ++e) {
                        kd[e] = ks[(col + e) * k_strides[3]];
                        vd[e] = vs[(col + e) * v_strides[3]];
                    }
                }
            } else {
                for (int e = 0; e < 8; ++e) {
                    kd[e] = T(0.0f);
                    vd[e] = T(0.0f);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!NO_SKIP && (!sg_live || k0 > sg_a1 || k0 + BK - 1 < sg_lo)) {
            continue;  // fully masked for this simdgroup's rows: an identity
        }
        // Bias-free tile: every key visible to every row of this simdgroup
        // and at distance >= E. Same logits as the general path, which adds
        // no bias there either.
        const bool fast = !NO_FAST && k0 >= pad && k0 + BK - 1 <= sg_a0 - E
            && (WINDOW == 0 || sg_a1 - k0 < WINDOW);
"""

_SIMD_BODY = r"""
    constexpr int LDS = D + 8;
    threadgroup T Ks[BK * LDS];
    threadgroup T Vs[BK * LDS];
    // 8x8 fragment layout: this lane holds row fm, columns fn and fn + 1.
    const short qid = short(lane / 4);
    const short fm = (qid & 4) + short((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + short(lane % 2) * 2;
    const int pos = p_sg0 + fm;
    const bool row_ok = sg_live && pos < L;
    const int a_row = OFF + (row_ok ? pos : 0);
    const device T* qrow = q + b * q_strides[0] + h * q_strides[1]
        + (int64_t)(row_ok ? pos : 0) * q_strides[2];
    const device T* relrow = rel + b * rel_strides[0] + (int64_t)(row_ok ? pos : 0) * rel_strides[1]
        + h * rel_strides[2];
    simdgroup_float8x8 Qf[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        for (int e = 0; e < 2; ++e) {
            Qf[db].thread_elements()[e] = row_ok
                ? float(qrow[(int64_t)(db * 8 + fn + e) * q_strides[3]]) : 0.0f;
        }
    }
    simdgroup_float8x8 Of[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = 0.0f;
    bool sink_nan = false;
    if (row_ok) {
        const float sv = float(sinks[b * sinks_strides[0] + (int64_t)pos * sinks_strides[1]
                                     + h * sinks_strides[2]]);
        // A NaN sink makes the row NaN, as in the composed softmax (bit
        // test: fast math may fold isnan).
        sink_nan = (as_type<uint>(sv) & 0x7fffffffu) > 0x7f800000u;
        m = clamp(sv, -1e30f, 1e30f) * LOG2E;
    }
    float l = 1.0f;

    for (int k0 = k_first; k0 <= hi; k0 += BK) {
""" + _STAGE + r"""
        simdgroup_float8x8 Sf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_float8x8 Kt;  // element (d = fm, key = fn + e)
                for (int e = 0; e < 2; ++e) {
                    Kt.thread_elements()[e] = float(Ks[(kb * 8 + fn + e) * LDS + db * 8 + fm]);
                }
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }
        uint vis_bits = 0u;
        float rmax = m;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                float s = Sf[kb].thread_elements()[e] * qscale;
                bool vis;
                if (fast) {
                    vis = true;
                } else {
                    const int j = k0 + kb * 8 + fn + e;
                    const int d = a_row - j;
                    vis = row_ok && d >= 0 && j >= pad && (WINDOW == 0 || d < WINDOW);
                    if (vis && d < E) {
                        s += float(relrow[(int64_t)d * rel_strides[3]]) * LOG2E;
                    }
                }
                Sf[kb].thread_elements()[e] = s;
                if (vis) {
                    vis_bits |= 1u << (kb * 2 + e);
                    rmax = max(rmax, s);
                }
            }
        }
        // The four lanes holding a fragment row differ in lane bits 0 and 3.
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const float m_new = rmax;
        const float factor = (m_new > m) ? fast::exp2(m - m_new) : 1.0f;
        float rsum = 0.0f;
        simdgroup_float8x8 Pf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const bool vis = (vis_bits >> (kb * 2 + e)) & 1u;
                const float p = vis ? fast::exp2(Sf[kb].thread_elements()[e] - m_new) : 0.0f;
                Pf[kb].thread_elements()[e] = p;
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
                simdgroup_float8x8 Vt;  // element (key = fm, d = fn + e)
                for (int e = 0; e < 2; ++e) {
                    Vt.thread_elements()[e] = float(Vs[(kb * 8 + fm) * LDS + db * 8 + fn + e]);
                }
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vt, Of[db]);
            }
        }
    }
    if (row_ok) {
        const float inv = 1.0f / l;
        device T* op = out + (((int64_t)b * L + pos) * HQ + h) * D;
        for (int db = 0; db < D / 8; ++db) {
            for (int e = 0; e < 2; ++e) {
                float o = Of[db].thread_elements()[e] * inv;
                o = (o == 0.0f) ? 0.0f : o;  // one zero sign whatever tiles added zeros
                o = sink_nan ? NAN : o;
                op[db * 8 + fn + e] = T(o);
            }
        }
    }
"""

_TENSOR_BODY = r"""
    constexpr int LDS = D;
    threadgroup T Ks[BK * LDS];
    threadgroup T Vs[BK * LDS];
    threadgroup float Ps[NSG * RS * BK];
    threadgroup float Fs[NSG * RS];
    threadgroup float* Pw = Ps + sg * RS * BK;
    threadgroup float* Fw = Fs + sg * RS;
    // Softmax lanes: row r = lane / 2 holds columns [c0, c0 + BK / 2).
    const int r = int(lane) / 2;
    const int c0 = (int(lane) % 2) * (BK / 2);
    const int pos = p_sg0 + r;
    const bool row_ok = sg_live && pos < L;
    const int a_row = OFF + (row_ok ? pos : 0);
    const int n_rows = sg_live ? (p_sg1 - p_sg0 + 1) : 1;
    const device T* relrow = rel + b * rel_strides[0] + (int64_t)(row_ok ? pos : 0) * rel_strides[1]
        + h * rel_strides[2];
    const int64_t q_off = b * q_strides[0] + h * q_strides[1]
        + (int64_t)(sg_live ? p_sg0 : 0) * q_strides[2];
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tQ(
        (device T*)(q + q_off), dextents<int32_t, 2>(D, n_rows),
        array<int32_t, 2>({1, int32_t(q_strides[2])}));
    tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline> tK(Ks, dextents<int32_t, 2>(D, BK));
    tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline> tV(Vs, dextents<int32_t, 2>(D, BK));
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tP(Pw, dextents<int32_t, 2>(BK, RS));
    constexpr auto desc_s = matmul2d_descriptor(RS, BK, D, false, true, false,
                                                matmul2d_descriptor::mode::multiply);
    constexpr auto desc_o = matmul2d_descriptor(RS, D, BK, false, false, false,
                                                matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_s, execution_simdgroup> op_s;
    matmul2d<desc_o, execution_simdgroup> op_o;
    auto Oc = op_o.template get_destination_cooperative_tensor<decltype(tP), decltype(tV), float>();
    for (uint16_t i = 0; i < Oc.get_capacity(); ++i) {
        Oc[i] = 0.0f;
    }
    float m = 0.0f;
    bool sink_nan = false;
    if (row_ok) {
        const float sv = float(sinks[b * sinks_strides[0] + (int64_t)pos * sinks_strides[1]
                                     + h * sinks_strides[2]]);
        // A NaN sink makes the row NaN, as in the composed softmax (bit
        // test: fast math may fold isnan).
        sink_nan = (as_type<uint>(sv) & 0x7fffffffu) > 0x7f800000u;
        m = clamp(sv, -1e30f, 1e30f) * LOG2E;
    }
    float l = 1.0f;
    // The wrapper guarantees unit D stride (declared by the caller or a
    // counted copy); a broken declaration fails loudly rather than reading
    // the wrong elements.
    const bool q_unit = q_strides[3] == 1;

    for (int k0 = k_first; k0 <= hi; k0 += BK) {
""" + _STAGE + r"""
        {
            auto Sc = op_s.template get_destination_cooperative_tensor<decltype(tQ), decltype(tK), float>();
            op_s.run(tQ, tK, Sc);
            Sc.store(tP);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float sv[BK / 2];
        uint vis_bits = 0u;
        float rmax = m;
        for (int c = 0; c < BK / 2; ++c) {
            float s = Pw[r * BK + c0 + c] * qscale;
            bool vis;
            if (fast) {
                vis = true;
            } else {
                const int j = k0 + c0 + c;
                const int d = a_row - j;
                vis = row_ok && d >= 0 && j >= pad && (WINDOW == 0 || d < WINDOW);
                if (vis && d < E) {
                    s += float(relrow[(int64_t)d * rel_strides[3]]) * LOG2E;
                }
            }
            sv[c] = s;
            if (vis) {
                vis_bits |= 1u << c;
                rmax = max(rmax, s);
            }
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        const float m_new = rmax;
        const float factor = (m_new > m) ? fast::exp2(m - m_new) : 1.0f;
        float rsum = 0.0f;
        for (int c = 0; c < BK / 2; ++c) {
            const float p = ((vis_bits >> c) & 1u) ? fast::exp2(sv[c] - m_new) : 0.0f;
            Pw[r * BK + c0 + c] = p;
            rsum += p;
        }
        rsum += simd_shuffle_xor(rsum, 1);
        l = l * factor + rsum;
        m = m_new;
        if (c0 == 0) {
            Fw[r] = factor;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint16_t i = 0; i < Oc.get_capacity(); ++i) {
            if (Oc.is_valid_element(i)) {
                const auto ix = Oc.get_multidimensional_index(i);
                Oc[i] *= Fw[int(ix[1])];
            }
        }
        op_o.run(tP, tV, Oc);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (c0 == 0) {
        Fw[r] = sink_nan ? NAN : 1.0f / l;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (sg_live) {
        for (uint16_t i = 0; i < Oc.get_capacity(); ++i) {
            if (Oc.is_valid_element(i)) {
                const auto ix = Oc.get_multidimensional_index(i);
                const int row = int(ix[1]);
                const int col = int(ix[0]);
                const int p = p_sg0 + row;
                if (row < RS && p < L && col < D) {
                    float o = Oc[i] * Fw[row];
                    o = (o == 0.0f) ? 0.0f : o;
                    if (!q_unit) {
                        o = NAN;  // contract violation only (see the wrapper)
                    }
                    out[(((int64_t)b * L + p) * HQ + h) * D + col] = T(o);
                }
            }
        }
    }
"""

INPUT_NAMES = ["q", "k", "v", "rel", "sinks", "pads", "origins", "dims", "scale"]
_kernels = {}


def kernel_source(form):
    """The kernel body ``mx.fast.metal_kernel`` compiles for ``form``."""
    if form == "simd":
        return _PROLOGUE + _SIMD_BODY
    if form == "tensor":
        return _PROLOGUE + _TENSOR_BODY
    raise ValueError(f"unknown form {form!r}")


def kernel_header(form):
    return _HEADER_TENSOR if form == "tensor" else _HEADER_SIMD


def kernel_name(form):
    return f"inkling2_prefill_onelaunch_{form}"


def _kernel(form):
    kernel = _kernels.get(form)
    if kernel is None:
        kernel = _kernels[form] = mx.fast.metal_kernel(
            name=kernel_name(form),
            input_names=INPUT_NAMES,
            output_names=["out"],
            source=kernel_source(form),
            header=kernel_header(form),
            ensure_row_contiguous=False,
        )
    return kernel


# --------------------------------------------------------------------------
# Geometry and device gates
# --------------------------------------------------------------------------

def position_blocks(form, group):
    """Position blocks per threadgroup (G * NPB simdgroups). Per-row results
    do not depend on it (only the tiles a threadgroup skips do)."""
    return max(1, SIMDGROUP_TARGET // max(1, group))


def threadgroup_bytes(form, dim, group, npb, bk=BLOCK_KEYS, itemsize=2):
    if form == "simd":
        return 2 * bk * (dim + 8) * itemsize
    nsg = group * npb
    return 2 * bk * dim * itemsize + nsg * ROWS["tensor"] * bk * 4 + nsg * ROWS["tensor"] * 4


def template(form, dtype, sink_dtype, *, dim, n_heads, n_kv, extent, window,
             no_skip=False, no_fast=False):
    group = n_heads // n_kv
    return [("T", dtype), ("TS", sink_dtype), ("D", int(dim)), ("HQ", int(n_heads)),
            ("HKV", int(n_kv)), ("E", int(extent)), ("WINDOW", int(window)),
            ("NPB", position_blocks(form, group)), ("BK", BLOCK_KEYS), ("RS", ROWS[form]),
            ("NO_SKIP", bool(no_skip)), ("NO_FAST", bool(no_fast))]


_ARCH = re.compile(r"^applegpu_g(\d+)([a-z])$")
_KNOWN_CLASSES = {"g", "s", "d", "c", "p"}


def tensor_units(architecture=None, macos=None):
    """(available, reason) for Metal 4 tensor operations on this GPU.

    Generation 17+ (phone class ``p``: 18+) and macOS 26.2+; any
    architecture string or class letter not recognised fails closed."""
    if architecture is None:
        try:
            info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
            architecture = str(info.get("architecture", ""))
        except Exception as error:  # noqa: BLE001 - no Metal device
            return False, f"device_info: {error}"
    match = _ARCH.match(str(architecture))
    if match is None:
        return False, f"unrecognised architecture {architecture!r}"
    gen, cls = int(match.group(1)), match.group(2)
    if cls not in _KNOWN_CLASSES:
        return False, f"unrecognised GPU class {cls!r}"
    if gen < (18 if cls == "p" else 17):
        return False, f"generation {gen}{cls} has no tensor units"
    if macos is None:
        macos = platform.mac_ver()[0]
    try:
        parts = [int(x) for x in str(macos).split(".")[:2]]
    except ValueError:
        return False, f"unrecognised macOS version {macos!r}"
    if not parts or tuple(parts + [0])[:2] < (26, 2):
        return False, f"macOS {macos} < 26.2"
    return True, f"{architecture} macOS {macos}"


def select_form(architecture=None, macos=None):
    """The form this process uses, or None. ``MLX2_INKLING2_ONELAUNCH_FORM``:
    ``auto`` (default: tensor where available, else simd), ``simd``,
    ``tensor`` (None without tensor units), ``off``."""
    want = os.environ.get("MLX2_INKLING2_ONELAUNCH_FORM", "auto").strip().lower()
    if want == "off":
        return None
    if want == "simd":
        return "simd"
    has_tensor, _ = tensor_units(architecture, macos)
    if want == "tensor":
        return "tensor" if has_tensor else None
    if want != "auto":
        return None
    return "tensor" if has_tensor else "simd"


def decline_reason(q, k, v, rel, sinks, *, window, form="simd", ring=None, device=None):
    """None when ``onelaunch_prefill_attention`` can run these arrays, else why."""
    if (mx.default_device() if device is None else device) != mx.gpu:
        return "cpu"
    if form not in FORMS:
        return "no_form"
    if not all(isinstance(a, mx.array) for a in (q, k, v, rel, sinks)):
        return "cache_format"
    if q.dtype not in (mx.bfloat16, mx.float16):
        return "dtype"
    if k.dtype != q.dtype or v.dtype != q.dtype or rel.dtype != q.dtype:
        return "dtype"
    if sinks.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return "dtype"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or rel.ndim != 4 or sinks.ndim != 3:
        return "geometry"
    b, hq, length, dim = q.shape
    hkv, phys = k.shape[1], k.shape[2]
    if k.shape != v.shape or k.shape[0] != b or k.shape[3] != dim:
        return "geometry"
    if hkv < 1 or hq % hkv:
        return "geometry"
    if rel.shape[:3] != (b, length, hq) or tuple(sinks.shape) != (b, length, hq):
        return "geometry"
    step = 16 if form == "tensor" else 8
    if dim % step or dim > 128 or dim < 16:
        return "geometry"
    group = hq // hkv
    npb = position_blocks(form, group)
    if group * npb * 32 > THREAD_LIMIT:
        return "threadgroup"
    if threadgroup_bytes(form, dim, group, npb) > THREADGROUP_MEMORY_LIMIT:
        return "threadgroup"
    if int(window) < 0:
        return "geometry"
    keys = phys if ring is None else int(ring[1])
    if ring is not None and not (0 <= int(ring[0]) < phys and 0 < keys <= phys):
        return "ring"
    if length < 2:
        return "rows"
    if length > keys:
        return "geometry"
    if keys < MIN_KEYS:
        return "crossover"
    return None


def _origins(b, left_padding, align_padding, abs_base):
    if align_padding is not None:
        base = align_padding.astype(mx.int32)
    elif left_padding is not None:
        if abs_base:
            raise ValueError("onelaunch_prefill_attention: abs_base with left_padding "
                             "needs align_padding (the padding before the crop)")
        base = left_padding.astype(mx.int32)
    else:
        base = mx.zeros((b,), mx.int32)
    return base - int(abs_base)


def onelaunch_prefill_attention(q, k, v, rel, sinks, *, offset, left_padding, window, scale,
                                abs_base=0, ring=None, gate=None, align_padding=None,
                                form="simd", no_skip=False, no_fast=False, q_unit_stride=False):
    """Prompt attention with Inkling 2's relative bias, window, padding and sink.

    q: [B, H, L, D], any strides. The tensor form reads Q as a strided
    tensor with unit stride on D; MLX does not expose strides to Python, so
    the caller declares it (``q_unit_stride=True``: language.py's q is a
    head transpose of a fresh [B, L, H, D] array) and otherwise the tensor
    form takes ``mx.contiguous(q)`` (``q_copied`` in the route counters);
    k, v: [B, Hkv, S_phys, D], any strides, holding the history and the
    chunk's rows; rel: [B, L, H, E] bias profiles; sinks: [B, L, H] (any
    float dtype). Row i sits at logical key ``offset + i``. ``ring=(rot,
    size)``: logical key j < size lives at physical row (j + rot) % S_phys
    (default: the buffer as is). ``left_padding`` [B] masks keys j <
    left_padding[b]. Tile boundaries sit at multiples of BK in each row's own
    token frame: ``abs_base`` is the absolute position of logical key 0 and
    ``align_padding`` [B] the padding in that absolute frame (defaults to
    ``left_padding``; required when both abs_base and left_padding are set).

    Returns [B, H, L, D] in q's dtype (a transposed view of a [B, L, H, D]
    buffer). GPU only; check ``decline_reason`` first. ``gate`` is not
    fused (the caller applies it); ``no_skip`` / ``no_fast`` are the debug
    builds of spec test N7."""
    if gate is not None:
        raise ValueError("onelaunch_prefill_attention: gate fusion is not implemented")
    if form == "tensor" and not q_unit_stride:
        q = mx.contiguous(q)
    b, hq, length, dim = q.shape
    hkv, phys = k.shape[1], k.shape[2]
    rot, keys = (0, phys) if ring is None else (int(ring[0]), int(ring[1]))
    pads = (mx.zeros((b,), mx.int32) if left_padding is None
            else mx.maximum(left_padding, 0).astype(mx.int32))
    origins = _origins(b, left_padding, align_padding, abs_base)
    group = hq // hkv
    per_tg = position_blocks(form, group) * ROWS[form]
    tiles = (length + per_tg - 1) // per_tg
    threads = group * position_blocks(form, group) * 32
    out = _kernel(form)(
        inputs=[q, k, v, rel, sinks, pads, origins,
                mx.array([length, keys, int(offset), rot, phys], dtype=mx.int32),
                mx.array([float(scale)], dtype=mx.float32)],
        template=template(form, q.dtype, sinks.dtype, dim=dim, n_heads=hq, n_kv=hkv,
                          extent=rel.shape[-1], window=window, no_skip=no_skip,
                          no_fast=no_fast),
        grid=(tiles * threads, hkv, b),
        threadgroup=(threads, 1, 1),
        output_shapes=[(b, length, hq, dim)],
        output_dtypes=[q.dtype],
    )[0]
    return out.transpose(0, 2, 1, 3)


# --------------------------------------------------------------------------
# References (CPU): the law in float64, and the kernel's algorithm in MLX ops
# --------------------------------------------------------------------------

def _ring_view(a, ring):
    if ring is None:
        return a
    rot, size = int(ring[0]), int(ring[1])
    return mx.take(a, (mx.arange(size, dtype=mx.int32) + rot) % a.shape[2], axis=2)


def reference64(q, k, v, rel, sinks, *, offset, left_padding, window, scale, ring=None,
                **_unused):
    """Spec section 2.1 in float64 (NumPy) on the given rounded inputs."""
    import numpy as np

    k, v = _ring_view(k, ring), _ring_view(v, ring)
    f = lambda a: np.array(a.astype(mx.float32), dtype=np.float64)  # noqa: E731
    qn, kn, vn, rn, sn = f(q), f(k), f(v), f(rel), f(sinks)
    b, hq, length, dim = qn.shape
    hkv, size = kn.shape[1], kn.shape[2]
    g = hq // hkv
    kn = np.repeat(kn, g, axis=1)
    vn = np.repeat(vn, g, axis=1)
    extent = rn.shape[-1]
    pos = np.arange(length)[:, None] + int(offset)
    keys = np.arange(size)[None, :]
    d = pos - keys
    vis = d >= 0
    if window:
        vis &= d < int(window)
    pads = (np.zeros(b, dtype=np.int64) if left_padding is None
            else np.maximum(np.array(left_padding), 0))
    logits = np.einsum("bhld,bhsd->bhls", qn, kn) * float(scale)
    bias = np.take_along_axis(rn.transpose(0, 2, 1, 3),
                              np.broadcast_to(np.clip(d, 0, extent - 1), (b, hq, length, size)),
                              axis=-1)
    logits = logits + np.where(d < extent, bias, 0.0)
    visb = vis[None, None] & (keys[None, None] >= pads[:, None, None, None])
    sink = np.clip(sn.transpose(0, 2, 1), -SINK_CLAMP, SINK_CLAMP)[..., None]  # [B, H, L, 1]
    masked = np.where(visb, logits, -np.inf)
    mrow = np.maximum(masked.max(axis=-1, keepdims=True), sink)
    w = np.where(visb, np.exp(masked - mrow), 0.0)
    den = w.sum(axis=-1, keepdims=True) + np.exp(sink - mrow)
    return np.einsum("bhls,bhsd->bhld", w, vn) / den


def emulate(q, k, v, rel, sinks, *, offset, left_padding, window, scale, abs_base=0,
            ring=None, align_padding=None, bk=BLOCK_KEYS):
    """The kernel's algorithm in MLX ops (CPU tests, debugging).

    Same arguments and result as ``onelaunch_prefill_attention``: key tiles
    of ``bk`` aligned to each row's token frame, ascending, the online
    softmax in the log2 domain from the clamped sink, FP32 operands, FP32
    probabilities, one rounding at the end. Every aligned tile of the buffer
    is processed for every row (the kernel skips the ones that are identities
    for its rows)."""
    k, v = _ring_view(k, ring), _ring_view(v, ring)
    b, hq, length, dim = q.shape
    hkv, size = k.shape[1], k.shape[2]
    g = hq // hkv
    f32 = mx.float32
    qf = q.astype(f32)
    kf = mx.repeat(k, g, axis=1).astype(f32)
    vf = mx.repeat(v, g, axis=1).astype(f32)
    extent = rel.shape[-1]
    bias = rel.astype(f32).transpose(0, 2, 1, 3) * LOG2E  # [B, H, L, E]
    m = mx.clip(sinks.astype(f32), -SINK_CLAMP, SINK_CLAMP).transpose(0, 2, 1)[..., None] * LOG2E
    l = mx.ones_like(m)
    acc = mx.zeros((b, hq, length, dim), f32)
    pads = (mx.zeros((b,), mx.int32) if left_padding is None
            else mx.maximum(left_padding, 0).astype(mx.int32))
    origins = _origins(b, left_padding, align_padding, abs_base)
    rows = mx.arange(length)[:, None] + int(offset)
    # Per batch row the tile grid is shifted by origins[b] % bk; process the
    # union of grids by giving each batch row its own tile index per key.
    keys_all = mx.arange(size)
    tile_of = (keys_all[None, :] - origins[:, None]) // bk  # [B, S] floor division
    t_lo = int(mx.min(tile_of).item()) if size else 0
    t_hi = int(mx.max(tile_of).item()) if size else -1
    scores = (qf @ kf.transpose(0, 1, 3, 2)) * (float(scale) * LOG2E)  # [B, H, L, S]
    d = rows - keys_all[None, :]
    gidx = mx.broadcast_to(mx.clip(d, 0, extent - 1)[None, None], scores.shape)
    scores = scores + mx.where((d < extent)[None, None],
                               mx.take_along_axis(bias, gidx, axis=-1), 0.0)
    vis = d >= 0
    if window:
        vis = vis & (d < int(window))
    vis = vis[None, None] & (keys_all[None, None, None, :] >= pads[:, None, None, None])
    for t in range(t_lo, t_hi + 1):
        in_tile = (tile_of == t)[:, None, None, :]  # [B, 1, 1, S]
        tv = vis & in_tile
        s = mx.where(tv, scores, -mx.inf)
        m_new = mx.maximum(m, s.max(axis=-1, keepdims=True))
        factor = mx.where(m_new > m, mx.power(2.0, m - m_new), 1.0)
        p = mx.where(tv, mx.power(2.0, scores - m_new), 0.0)
        l = l * factor + p.sum(axis=-1, keepdims=True)
        acc = acc * factor + p @ vf
        m = m_new
    out = acc / l
    # A NaN sink makes its row NaN (the composed softmax does the same).
    nan_row = mx.isnan(sinks.astype(f32)).transpose(0, 2, 1)[..., None]
    return mx.where(nan_row, mx.nan, out).astype(q.dtype)


# --------------------------------------------------------------------------
# Spec N1 metrics and the first-use self-check
# --------------------------------------------------------------------------

def numerics(got, ref64, dtype):
    """(relative RMS error, all finite, fraction correctly rounded)."""
    import numpy as np

    g = np.array(got.astype(mx.float32), dtype=np.float64)
    finite = bool(np.isfinite(g).all())
    den = float(np.sqrt(np.mean(ref64 ** 2))) or 1.0
    rel_rms = float(np.sqrt(np.mean((g - ref64) ** 2))) / den
    rounded = np.array(mx.array(ref64.astype(np.float32)).astype(dtype).astype(mx.float32),
                       dtype=np.float64)
    return rel_rms, finite, float(np.mean(g == rounded))


def synthetic(seed, *, b, hq, hkv, length, size, dim, extent, dtype=mx.bfloat16,
              pads=None):
    """Spec section 4 synthetic inputs: q, k ~ N(0, 2.2^2) clipped to L2 22.63,
    v RMS-normalised, profiles N(0, 2.2^2) clipped to 8, sinks N(0, 3^2);
    all rounded to ``dtype``. Returns (q [B,H,L,D], k, v [B,Hkv,S,D],
    rel [B,L,H,E], sinks [B,L,H])."""
    key = mx.random.key(seed)
    ks = mx.random.split(key, 5)

    def clip(x, t):
        n = mx.linalg.norm(x, axis=-1, keepdims=True)
        return x * mx.minimum(1.0, t / mx.maximum(n, 1e-12))

    q = clip(mx.random.normal((b, hq, length, dim), key=ks[0]) * 2.2, 22.63)
    k = clip(mx.random.normal((b, hkv, size, dim), key=ks[1]) * 2.2, 22.63)
    v = mx.random.normal((b, hkv, size, dim), key=ks[2])
    v = v * mx.rsqrt(mx.mean(v * v, axis=-1, keepdims=True) + 1e-6)
    rel = clip(mx.random.normal((b, length, hq, extent), key=ks[3]) * 2.2, 8.0)
    sinks = mx.random.normal((b, length, hq), key=ks[4]) * 3.0
    return tuple(a.astype(dtype) for a in (q, k, v, rel, sinks))


def composed(q, k, v, rel, sinks, *, offset, left_padding, window, scale, ring=None, **_unused):
    """The composed path (banded mask + zero sink row + SDPA), the ordinary
    reference this route replaces."""
    from .splitk import reference_attention

    return reference_attention(q, k, v, rel, sinks.astype(q.dtype), offset=offset,
                               left_padding=left_padding, window=window, scale=scale, ring=ring)


_self_checks = {}
# Exception text of failures that say nothing about the kernel's correctness
# (allocation): such a self-check is not cached and is retried on a later call.
TRANSIENT = "transient:"
_TRANSIENT_MARKERS = ("out of memory", "insufficient memory", "outofmemory",
                      "resource", "memoryerror", "timeout")


def _check_case(form, args, kw, dtype, tol):
    got = onelaunch_prefill_attention(*args, form=form, **kw)
    stock = composed(*args, **kw)
    mx.eval(got, stock)
    ref = reference64(*args, **kw)
    err, finite, exact = numerics(got, ref, dtype)
    s_err, _, s_exact = numerics(stock, ref, dtype)
    if not finite:
        return got, "nonfinite"
    if err > tol or err > max(s_err, 0.0) * 1.5 + 1e-6:
        return got, f"error {err:.3g} (stock {s_err:.3g})"
    if exact + 1e-9 < s_exact - 0.02:
        return got, f"rounded {exact:.3f} < stock {s_exact:.3f}"
    return got, None


def self_check(form, dtype, *, dim, n_heads, n_kv, extent, window, sink_dtype=None):
    """Spec N1 once per (form, dtype, window kind, geometry) on the device.

    Cases: B = 1 (the per-batch inputs in the constant address space) and
    B = 8 with left padding {0, 1, 31, 100, ...} (device address space),
    each against float64 and the composed path; a ring read bitwise equal to
    its chronological copy; the B = 8 rows chunked at two widths bitwise
    equal to the whole call. Returns None when it passes, else the reason. A
    correctness failure is cached for the process (decline ``first_use``); a
    transient failure (allocation) is returned prefixed ``transient:`` and not
    cached."""
    key = (form, dtype, sink_dtype, bool(window), int(dim), int(n_heads), int(n_kv),
           int(extent), int(window))
    if key in _self_checks:
        return _self_checks[key]
    reason = None
    tol = TOLERANCE[dtype]
    try:
        size = (window or extent) + 300
        length = 40
        scale = dim ** -0.5

        def case(seed, b, pads):
            args = synthetic(seed, b=b, hq=n_heads, hkv=n_kv, length=length, size=size,
                             dim=dim, extent=extent, dtype=dtype)
            if sink_dtype is not None:
                args = args[:4] + (args[4].astype(sink_dtype),)
            kw = dict(offset=size - length, window=window, scale=scale,
                      left_padding=None if pads is None else mx.array(pads, dtype=mx.int32))
            return args, kw

        args, kw = case(1234, 1, None)
        _, reason = _check_case(form, args, kw, dtype, tol)
        if reason is None:
            pads = [0, 1, 31, 100, 5, 300, size - length + 3, 2][:8]
            bargs, bkw = case(1235, 8, pads)
            whole, reason = _check_case(form, bargs, bkw, dtype, tol)
        if reason is None:
            q, k, v, rel, sinks = args
            rot = 211 % size
            ring = onelaunch_prefill_attention(
                q, mx.roll(k, rot, axis=2), mx.roll(v, rot, axis=2), rel, sinks, form=form,
                ring=(rot, size), **kw)
            plain = onelaunch_prefill_attention(*args, form=form, **kw)
            if not bool(mx.array_equal(ring, plain).item()):
                reason = "ring not bitwise"
        if reason is None:
            q, k, v, rel, sinks = bargs
            cut = 17
            parts = []
            for a, e in ((0, cut), (cut, length)):
                parts.append(onelaunch_prefill_attention(
                    q[:, :, a:e], k[:, :, : size - length + e], v[:, :, : size - length + e],
                    rel[:, a:e], sinks[:, a:e], form=form, offset=size - length + a,
                    left_padding=bkw["left_padding"], window=window, scale=scale))
            if not bool(mx.array_equal(mx.concatenate(parts, axis=2), whole).item()):
                reason = "chunk widths not bitwise"
    except Exception as error:  # noqa: BLE001 - pipeline limits, compile errors, OOM
        text = f"{type(error).__name__}: {str(error)[:200]}"
        if any(m in text.lower() for m in _TRANSIENT_MARKERS):
            return TRANSIENT + text
        reason = "exception: " + text
    _self_checks[key] = reason
    return reason


# --------------------------------------------------------------------------
# Offline compile (xcrun, CPU only): the translation units metal_kernel builds
# --------------------------------------------------------------------------

_TYPES = {mx.bfloat16: "bfloat16_t", mx.float16: "float16_t", mx.float32: "float",
          mx.int32: "int32_t"}
_ATTRS = (("thread_position_in_threadgroup", "uint3"), ("threadgroup_position_in_grid", "uint3"))


def translation_unit(form, tmpl, *, small_batch=True):
    """The source ``mx.fast.metal_kernel`` builds for ``form`` and template
    ``tmpl`` (MLX 0.32: inputs under 8 elements in the constant address
    space; ``<name>_strides`` as ``constant int64_t*`` when referenced)."""
    source = kernel_source(form)
    shapes = {"pads": 1 if small_batch else 8, "origins": 1 if small_batch else 8,
              "dims": 5, "scale": 1}
    dtypes = dict(tmpl)
    in_types = {"q": dtypes["T"], "k": dtypes["T"], "v": dtypes["T"], "rel": dtypes["T"],
                "sinks": dtypes["TS"], "pads": mx.int32, "origins": mx.int32,
                "dims": mx.int32, "scale": mx.float32}
    sig, index = [], 0
    for name in INPUT_NAMES:
        space = "constant" if shapes.get(name, 1 << 20) < 8 else "device"
        sig.append(f"  const {space} {_TYPES[in_types[name]]}* {name} [[buffer({index})]]")
        index += 1
        if f"{name}_strides" in source:
            sig.append(f"  const constant int64_t* {name}_strides [[buffer({index})]]")
            index += 1
    sig.append(f"  device {_TYPES[dtypes['T']]}* out [[buffer({index})]]")
    for attr, kind in _ATTRS:
        if attr in source:
            sig.append(f"  {kind} {attr} [[{attr}]]")
    params, args = [], []
    for key, value in tmpl:
        if isinstance(value, mx.Dtype):
            params.append(f"typename {key}")
            args.append(_TYPES[value])
        elif isinstance(value, bool):
            params.append(f"bool {key}")
            args.append("true" if value else "false")
        else:
            params.append(f"int {key}")
            args.append(str(int(value)))
    func = f"custom_kernel_{kernel_name(form)}"
    inst = f"{func}<{', '.join(args)}>"
    return "\n".join([
        "#include <metal_stdlib>", '#include "mlx/backend/metal/kernels/utils.h"',
        kernel_header(form),
        f"template <{', '.join(params)}>",
        f"[[kernel]] void {func}(\n" + ",\n".join(sig) + ") {\n" + source + "\n}\n",
        f'template [[host_name("{func}_i")]] [[kernel]] decltype({inst}) {inst};',
    ])


def model_variants(*, dim=128, n_heads=24, n_kv=8, extents=(1024,), windows=(0, 1024)):
    """Every (form, template) the route can generate for one geometry:
    both forms, BF16/FP16 activations with sinks in the activation dtype or
    FP32, each window kind, and the two debug builds."""
    out = []
    for form in FORMS:
        for dtype in (mx.bfloat16, mx.float16):
            for sink_dtype in (dtype, mx.float32):
                for extent in extents:
                    for window in windows:
                        for no_skip, no_fast in ((False, False), (True, False), (False, True)):
                            out.append((form, template(
                                form, dtype, sink_dtype, dim=dim, n_heads=n_heads, n_kv=n_kv,
                                extent=extent, window=window, no_skip=no_skip,
                                no_fast=no_fast)))
    return out


def compile_offline(form, tmpl, *, small_batch=True, workdir=None):
    """Compile one variant with ``xcrun -sdk macosx metal -c`` (no GPU).
    Returns None on success, else the compiler's error text."""
    import subprocess
    import tempfile
    from pathlib import Path

    include = Path(mx.__file__).resolve().parent / "include"
    std = "metal4.0" if form == "tensor" else "metal3.1"
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        src = Path(tmp) / "unit.metal"
        src.write_text(translation_unit(form, tmpl, small_batch=small_batch))
        result = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", f"-std={std}", "-I", str(include), "-c",
             str(src), "-o", str(Path(tmp) / "unit.air")],
            capture_output=True, text=True)
    return None if result.returncode == 0 else result.stderr[-4000:]


# --------------------------------------------------------------------------
# Route (language.py): selection, declines, counters
# --------------------------------------------------------------------------

def count(attn, key):
    counts = attn.__dict__.setdefault("prefill_onelaunch_counts", {})
    counts[key] = counts.get(key, 0) + 1


def route(attn, q, k, v, rel, sinks, *, offset, left_padding, window, scale, abs_base,
          align_padding, ring, q_unit_stride=False):
    """The route's output, or None after counting the decline reason.

    ``calls:*`` is counted only once the launch was built; an exception from
    it declines (``launch``) instead of propagating. Pipeline errors raised
    when the graph is evaluated are not seen here: the first-use self-check
    evaluates the variants this geometry generates (both address spaces,
    padding, a ring, two chunk widths) before the route serves."""
    form = select_form()
    reason = "no_form" if form is None else decline_reason(
        q, k, v, rel, sinks, window=window, form=form, ring=ring)
    if reason is None:
        reason = self_check(form, q.dtype, dim=q.shape[-1], n_heads=q.shape[1],
                            n_kv=k.shape[1], extent=rel.shape[-1], window=window,
                            sink_dtype=sinks.dtype)
        if reason is not None and reason.startswith(TRANSIENT):
            reason = "first_use_transient"  # not cached: retried on the next call
        elif reason is not None:
            attn.__dict__.setdefault("prefill_onelaunch_first_use", {})[form] = reason
            reason = "first_use"
    if reason is not None:
        count(attn, "declined:" + reason)
        return None
    try:
        out = onelaunch_prefill_attention(
            q, k, v, rel, sinks, offset=offset, left_padding=left_padding, window=window,
            scale=scale, abs_base=abs_base, ring=ring, align_padding=align_padding, form=form,
            q_unit_stride=q_unit_stride)
    except Exception:  # noqa: BLE001 - fail closed to the composed path, counted
        count(attn, "declined:launch")
        return None
    count(attn, "calls:" + form)
    count(attn, "calls:" + ("window" if window else "global"))
    if form == "tensor" and not q_unit_stride:
        count(attn, "q_copied")
    if ring is not None and (int(ring[0]) or int(ring[1]) != k.shape[2]):
        count(attn, "ring_in_place")
    return out


def receipt(layers):
    """Summed counters for ``kernel_receipt``."""
    total, first_use = {}, {}
    for attn in layers:
        for key, value in attn.__dict__.get("prefill_onelaunch_counts", {}).items():
            total[key] = total.get(key, 0) + value
        first_use.update(attn.__dict__.get("prefill_onelaunch_first_use", {}))
    calls = {k.split(":", 1)[1]: total.pop(k) for k in sorted(total) if k.startswith("calls:")}
    ring = total.pop("ring_in_place", 0)
    copies = total.pop("ring_copies", 0)
    q_copied = total.pop("q_copied", 0)
    return {"calls": sum(calls.get(f, 0) for f in FORMS), "by_kind": calls,
            "ring_in_place": ring, "ring_copies": copies, "q_copied": q_copied,
            "declines": {k.split(":", 1)[1]: v for k, v in sorted(total.items())},
            "first_use_failures": dict(sorted(first_use.items())),
            "qualification": "unqualified"}


__all__ = ["BLOCK_KEYS", "FORMS", "compile_offline", "composed", "decline_reason", "emulate",
           "model_variants", "numerics", "onelaunch_prefill_attention", "receipt",
           "reference64", "route", "select_form", "self_check", "synthetic", "tensor_units",
           "translation_unit"]

