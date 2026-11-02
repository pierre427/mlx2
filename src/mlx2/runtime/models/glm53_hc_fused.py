# SPDX-License-Identifier: MIT
"""Opt-in GLM-5.3 HC boundary with the flat norm and F32 mixing gemv fused (``glm53_hc_fused``).

Under ``glm53_hc=mega_v1`` one B1 decode HC boundary (per half-layer) is, in
launch order: the previous half's four-stream expand (deepseek_v4_hc
EXPAND_SOURCE), the flat ``rms_norm`` of the 4x4096 stream, the native F32
mixing projection ``z @ fn.T`` ([16384 -> 24], MLX's ``gemv`` kernel) and the
mega ``post`` kernel (Sinkhorn split, weighted sum, half norm). This module
folds the first three into one launch and, optionally, the fourth as well:

``v1``       one ``project`` kernel (expand when the half has one, flat norm,
             mixing gemv), then the unchanged mega ``post`` kernel. Two
             launches per half instead of three or four.
``tail_v1``  the same kernel where the last threadgroup to finish its gemv
             tile runs the mega ``post`` body itself (MSL 3.2+ device-scope
             fences and atomics carry the 24 mixes across threadgroups). One
             launch per half.

Exactness (bit for bit, B1 single-token rows only):

* expand: ``glm_hc_expand`` from glm53_hc_mega's header, compiled with the same
  default options as EXPAND_SOURCE; each value is computed once per
  threadgroup, kept in registers and reused for the norm, the store, the gemv
  operand and (tail) the weighted sum.
* flat norm: glm53_hc_mega's ``pre`` law (1024 lanes, four reads per lane per
  4096 chunk, fma accumulation, ``simd_sum`` twice, precise rsqrt) and
  ``1*(x*inv)``; the gemv operand is that value, staged through threadgroup
  memory one 4096 chunk at a time, never re-rounded.
* gemv: MLX 0.32.4 ``GEMVKernel`` (gemv.h) at the geometry Matmul::eval_gpu
  selects for a [1,K] x [K,N] product with a transposed [N,K] row-major
  matrix, N < 4096 and K >= 16N: BM=1, BN=8, SM=1, SN=32, TM=4, TN=4. Every
  output keeps that kernel's arithmetic: lane v = 32 s + l of simdgroup s
  accumulates ``result += mat * vec`` (fused by the compiler, as in the
  metallib) over k = 4v + 1024 it + tn in that order, then
  ``simd_shuffle_down`` 16, 8, 4, 2, 1, then simdgroup 0 adds simdgroups 1..7
  in order. MLX runs four rows per lane (TM=4) on N/4 threadgroups; here
  ``RPT`` rows per threadgroup (default 1: 24 threadgroups) only spreads the
  same per-row chains over more cores.
* post: glm53_hc_mega's post body, verbatim pieces and order.

Equality with the mega_v1 composition is a first-use GPU gate (``self_check``)
and a GPU test, not an assumption. The cross-threadgroup tail uses a per-call
launch uint32 counter (fresh per call, so never shared by two launches
that MLX could run concurrently) that the last threadgroup resets.
See provenance/glm53-decode-dispatch-fusion.json.
"""

from __future__ import annotations

from functools import lru_cache

from . import glm53_hc_mega as mega

POLICIES = ("reference", "v1", "tail_v1")
HC, DIM, FLAT, MIXES = mega.HC, mega.DIM, mega.FLAT, 24
THREADS = 1024
# MLX gemv geometry for N=24, K=16384 (see module docstring).
BN, SN, TM, TN = 8, 32, 4, 4
BLOCK_N = BN * SN * TN
# Output rows per threadgroup (1, 2 or 4; 4 is MLX's 6-threadgroup split).
# Values do not depend on it: each row's chains and reductions are fixed.
RPT = 1
TILES = MIXES // RPT
_SCHEDULED = {"project": 0, "project_expand": 0, "tail": 0, "tail_expand": 0}

# Shared device helpers. Placed after glm53_hc_mega's header, whose trailing
# `#pragma clang fp contract(off)` governs the kernel bodies; the gemv step is
# compiled with contraction on, as MLX's metallib compiles gemv.h.
GEMV_HEADER = r"""
#pragma clang fp contract(on)
inline float glm_gemv_acc(float acc, float a, float b) { acc += a * b; return acc; }
#pragma clang fp contract(off)
"""

_HEADER = mega._HEADER + GEMV_HEADER + r"""
// Stream accessors (MLX binds small inputs as constant, large as device).
template<typename H> struct glm_hc_plain {
    H h;
    float operator[](uint i) const { return h[i]; }
};
template<typename B, typename R, typename P, typename C> struct glm_hc_lazy {
    B br; R rr; P pr; C cr;
    float operator[](uint i) const { return glm_hc_expand(br, rr, pr, cr, i / 4096u, i % 4096u, 4096u); }
};
template<typename H> inline glm_hc_plain<H> glm_hc_make_plain(H h) { return {h}; }
// This thread's 16 register values as the stream (index = c*4096 + base + i).
struct glm_hc_regs {
    thread const float *vals; uint base;
    float operator[](uint i) const { return vals[(i / 4096u) * 4u + (i % 4096u - base)]; }
};
template<typename B, typename R, typename P, typename C>
inline glm_hc_lazy<B, R, P, C> glm_hc_make_lazy(B b, R r, P p, C c) { return {b, r, p, c}; }
"""

# Body. Thread p owns flat indices c*4096 + 4p + i (c, i < 4): the rms_norm
# lane layout, and also (stream c, d = 4p + i), the post kernel's layout. Its
# 16 values stay in registers. The gemv operands of chunk c are staged in
# threadgroup memory (4096 floats) and read back by the gemv lanes, so no
# value is loaded or derived twice. RPT output rows per threadgroup: simdgroup
# 8r + s is MLX gemv simdgroup s of row tile*RPT + r (one TM=1 chain per lane;
# MLX's TM=4 lanes run the same per-row chain).
_PROJECT_SOURCE = r"""
threadgroup float partials[32];
threadgroup float inv_shared;
threadgroup float gemv_part[RPT * BN];
threadgroup float stage[4096];
const uint tile = threadgroup_position_in_grid.x;
const uint lid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
HV_DECL
float vals[16];
float acc = 0.0f;
for (uint c = 0; c < 4; c++) {
    for (uint i = 0; i < 4; i++) {
        const uint index = c * 4096 + lid * 4 + i;
        const float v = hv[index];
        vals[c * 4 + i] = v;
        STORE_H
        acc = FLATFMA ? glm_hc_sq_fma(acc, v) : glm_hc_sq_add(acc, v);
    }
}
acc = simd_sum(acc);
if (lane == 0) partials[sg] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    const float total = simd_sum(partials[lane]);
    if (lane == 0) inv_shared = glm_hc_inv(total, float(FLAT), constants[1]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float inv = inv_shared;
// MLX GEMVKernel<float, 1, BN=8, 1, 32, TM, TN=4>: lane v = 32 s + l of row
// `row` accumulates k = 4v + 1024 it + tn, it = 0..15, in order.
const uint gr = sg / BN, gs = sg % BN;
const bool gemv_lane = gr < RPT;
const uint row = tile * RPT + gr;
const uint v = gs * 32 + lane;
// F32 or exact-BF16 fn (glm53_f32_weights): same float operand values.
const device auto *mat = w + size_t(gemv_lane ? row : 0) * FLAT;
float result = 0.0f;
for (uint c = 0; c < 4; c++) {
    for (uint i = 0; i < 4; i++) stage[lid * 4 + i] = glm_hc_scale(1.0f, vals[c * 4 + i], inv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gemv_lane) {
        for (uint j = 0; j < 4; j++) {
            const uint local = v * TN + j * BLOCKN;
            for (uint tn = 0; tn < TN; tn++) {
                result = glm_gemv_acc(result, mat[c * 4096 + local + tn], stage[local + tn]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (ushort sn = 16; sn >= 1; sn >>= 1) result += simd_shuffle_down(result, sn);
if (gemv_lane && lane == 0) gemv_part[gr * BN + gs] = result;
threadgroup_barrier(mem_flags::mem_threadgroup);
TAIL
"""

_STORE_MIXES = r"""
if (lid < RPT) {
    float r = gemv_part[lid * BN];
    for (int sgn = 1; sgn < BN; sgn++) r += gemv_part[lid * BN + sgn];
    mixes[tile * RPT + lid] = r;
}
"""

# Last-threadgroup tail: publish this tile's mixes with device atomics, count
# arrivals, and let the last arrival run glm53_hc_mega's post body with h from
# its own registers.
_TAIL = r"""
threadgroup uint last_shared;
threadgroup float mix_tg[24];
threadgroup float coeff[24];
threadgroup float inv_post;
device atomic_uint *published = (device atomic_uint *)(mixes);
if (lid < RPT) {
    float r = gemv_part[lid * BN];
    for (int sgn = 1; sgn < BN; sgn++) r += gemv_part[lid * BN + sgn];
    atomic_store_explicit(published + tile * RPT + lid, as_type<uint>(r), memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
if (lid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    device atomic_uint *count = (device atomic_uint *)(counter);
    const uint arrived = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
    uint last = arrived == uint(TILES - 1) ? 1u : 0u;
    if (last) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        atomic_store_explicit(count, 0u, memory_order_relaxed);
    }
    last_shared = last;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (last_shared == 0u) return;
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
if (lid < 24) mix_tg[lid] = as_type<float>(atomic_load_explicit(published + lid, memory_order_relaxed));
threadgroup_barrier(mem_flags::mem_threadgroup);
// glm53_hc_mega _POST_SOURCE, row 0, mixes from mix_tg and h from registers.
if (LANES) { if (sg == 0) glm_hc_split_lanes(mix_tg, scale, base, constants[0], ITERS, lane, coeff); }
else if (lid == 0) glm_hc_split(mix_tg, scale, base, constants[0], ITERS, coeff);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid < 24) out_coeff[lid] = coeff[lid];
const glm_hc_regs own{vals, lid * 4};
float xs[4];
float pacc = 0.0f;
for (uint i = 0; i < 4; i++) {
    const uint d = lid * 4 + i;
    xs[i] = glm_hc_sum(own, coeff, d, DIM);
    pacc = glm_hc_sq_add(pacc, xs[i]);
}
pacc = simd_sum(pacc);
if (lane == 0) partials[sg] = pacc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    const float total = simd_sum(partials[lane]);
    if (lane == 0) inv_post = glm_hc_inv(total, float(DIM), constants[1]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float pinv = inv_post;
for (uint i = 0; i < 4; i++) {
    const uint d = lid * 4 + i;
    x[d] = glm_hc_scale(weight[d], xs[i], pinv);
}
"""


def _source(expand, tail):
    if expand:
        decl = "const auto hv = glm_hc_make_lazy(branch, residual, post, comb);"
        # Contiguous slices of h, one per tile, so the stores coalesce.
        store = "if (index >= tile * HSLICE && index < (tile + 1) * HSLICE) h_out[index] = v;"
    else:
        decl = "const auto hv = glm_hc_make_plain(h);"
        store = ""
    return (_PROJECT_SOURCE.replace("HV_DECL", decl).replace("STORE_H", store)
            .replace("TAIL", _TAIL if tail else _STORE_MIXES))


@lru_cache(maxsize=8)
def _kernel(expand, tail, bf16_weight=False):
    import mlx.core as mx

    names = ["branch", "residual", "post", "comb"] if expand else ["h"]
    names += ["w", "constants"]
    outs = ["h_out"] if expand else []
    if tail:
        names += ["scale", "base", "weight", "counter"]
        outs += ["mixes", "out_coeff", "x"]
    else:
        outs += ["mixes"]
    name = ("mlx2_glm53_hc_fused_" + ("expand_" if expand else "") + ("tail_" if tail else "") + "v1"
            + ("_bf16w" if bf16_weight else ""))
    return mx.fast.metal_kernel(name=name, input_names=names, output_names=outs, header=_HEADER,
                                source=_source(expand, tail), ensure_row_contiguous=True)


def _template(expand, tail, iterations):
    template = [("TG", THREADS), ("DIM", DIM), ("FLAT", FLAT), ("FLATFMA", int(mega.FLAT_FMA)),
                ("BN", BN), ("TN", TN), ("BLOCKN", BLOCK_N), ("RPT", RPT), ("TILES", TILES)]
    if expand:
        template.append(("HSLICE", -(-FLAT // TILES)))
    if tail:
        template += [("ITERS", iterations), ("LANES", int(mega.LANES))]
    return template


def admits(h, fn):
    """True for one B1 row with the HC4x4096 stream and an F32 [24, 16384] fn
    (or its exact BF16 copy, glm53_f32_weights=bf16_v1: same operand values)."""
    import mlx.core as mx

    return (
        mx.default_device() == mx.gpu
        and mx.metal.is_available()
        and tuple(h.shape) == (1, 1, HC, DIM)
        and h.dtype == mx.float32
        and fn is not None
        and tuple(fn.shape) == (MIXES, FLAT)
        and fn.dtype in (mx.float32, mx.bfloat16)
    )


def counter():
    """A fresh arrival counter for one tail launch (glm53_glue_metal.arrival_counter)."""
    from .glm53_glue_metal import arrival_counter

    return arrival_counter()


def project(h, fn, *, rms_eps, expand=None):
    """(h, mixes [1,1,24]) for one B1 row; ``expand`` = (branch, residual, post, comb)
    computes h as the previous half's expand first (returned as the new h)."""
    import mlx.core as mx

    constants = mega._constants(1e-6, rms_eps)
    if expand is None:
        (mixes,) = _kernel(False, False, fn.dtype == mx.bfloat16)(
            inputs=[h.astype(mx.float32), fn, constants], template=_template(False, False, 0),
            grid=(TILES * THREADS, 1, 1), threadgroup=(THREADS, 1, 1),
            output_shapes=[(1, 1, MIXES)], output_dtypes=[mx.float32])
        _SCHEDULED["project"] += 1
        return h, mixes
    branch, residual, post_, comb = expand
    h_out, mixes = _kernel(True, False, fn.dtype == mx.bfloat16)(
        inputs=[branch.astype(mx.float32), residual.astype(mx.float32), post_.astype(mx.float32),
                comb.astype(mx.float32), fn, constants], template=_template(True, False, 0),
        grid=(TILES * THREADS, 1, 1), threadgroup=(THREADS, 1, 1),
        output_shapes=[residual.shape, (1, 1, MIXES)], output_dtypes=[mx.float32, mx.float32])
    _SCHEDULED["project_expand"] += 1
    return h_out, mixes


def boundary(h, fn, scale, base, weight, counter_=None, *, rms_eps, hc_eps=1e-6, iterations=20, expand=None):
    """(h, x, pre, post, comb) of one B1 HC boundary in one launch (``tail_v1``).

    Every launch gets its own arrival counter by default: two launches that
    shared one and were not ordered by a data dependency could run
    concurrently and mix arrivals. Pass ``counter_`` only for a launch whose
    counter no unordered launch uses.
    """
    import mlx.core as mx

    if counter_ is None:
        counter_ = counter()

    constants = mega._constants(hc_eps, rms_eps)
    tail = [scale.astype(mx.float32), base.astype(mx.float32), weight.astype(mx.float32), counter_]
    shapes = [(1, 1, MIXES), (1, 1, MIXES), (1, 1, DIM)]
    if expand is None:
        mixes, coeff, x = _kernel(False, True, fn.dtype == mx.bfloat16)(
            inputs=[h.astype(mx.float32), fn, constants, *tail], template=_template(False, True, iterations),
            grid=(TILES * THREADS, 1, 1), threadgroup=(THREADS, 1, 1),
            output_shapes=shapes, output_dtypes=[mx.float32] * 3)
        _SCHEDULED["tail"] += 1
    else:
        branch, residual, post_, comb = expand
        h, mixes, coeff, x = _kernel(True, True, fn.dtype == mx.bfloat16)(
            inputs=[branch.astype(mx.float32), residual.astype(mx.float32), post_.astype(mx.float32),
                    comb.astype(mx.float32), fn, constants, *tail],
            template=_template(True, True, iterations),
            grid=(TILES * THREADS, 1, 1), threadgroup=(THREADS, 1, 1),
            output_shapes=[residual.shape, *shapes], output_dtypes=[mx.float32] * 4)
        _SCHEDULED["tail_expand"] += 1
    del mixes
    return h, x, coeff[..., :4], coeff[..., 4:8], coeff[..., 8:].reshape(1, 1, 4, 4)


def reference(h, fn, scale, base, weight, *, rms_eps, expand=None):
    """The glm53_hc=mega_v1 composition this module replaces (Glm53GGUFModel._hc_in)."""
    import mlx.core as mx

    from .deepseek_v4_hc import expand as hc_expand

    if expand is not None:
        h = hc_expand(*expand)
    z = mx.fast.rms_norm(h.reshape(1, 1, -1).astype(mx.float32), None, rms_eps)
    mixes = z @ fn.T
    x, pre, post_, comb = mega.post(h, mixes, scale, base, weight, rms_eps=rms_eps)
    return h, x, pre, post_, comb, mixes


@lru_cache(maxsize=2)
def self_check(rms_eps=1e-5):
    """First-use bitwise GPU gate of both variants against the mega_v1 composition."""
    import mlx.core as mx
    import numpy as np

    mega.self_check(rms_eps)
    rng = np.random.default_rng(5303)
    for trial, magnitude in enumerate((0.05, 1.0, 7.0, 30.0, 300.0)):
        h = mx.array((rng.normal(size=(1, 1, HC, DIM)) * magnitude).astype(np.float32))
        branch = mx.array((rng.normal(size=(1, 1, DIM)) * magnitude).astype(np.float32))
        post_ = mx.array(rng.uniform(0, 2, size=(1, 1, 4)).astype(np.float32))
        comb = mx.array(rng.uniform(0, 1, size=(1, 1, 4, 4)).astype(np.float32))
        fn = mx.array((rng.normal(size=(MIXES, FLAT)) * 0.02).astype(np.float32))
        scale = mx.array(rng.uniform(0.1, 2.0, size=3).astype(np.float32))
        base = mx.array(rng.normal(size=24).astype(np.float32))
        weight = mx.array(rng.normal(size=DIM).astype(np.float32))
        for expand in (None, (branch, h, post_, comb)):
            ref_h, ref_x, ref_pre, ref_post, ref_comb, ref_mixes = reference(
                h, fn, scale, base, weight, rms_eps=rms_eps, expand=expand)
            got_h, got_mixes = project(h, fn, rms_eps=rms_eps, expand=expand)
            tail = boundary(h, fn, scale, base, weight, rms_eps=rms_eps, expand=expand)
            pairs = [(got_h, ref_h), (got_mixes, ref_mixes), (tail[0], ref_h), (tail[1], ref_x),
                     (tail[2], ref_pre), (tail[3], ref_post), (tail[4], ref_comb)]
            mx.eval(pairs)
            for index, (a, b) in enumerate(pairs):
                av, bv = np.array(a), np.array(b)
                if av.shape != bv.shape or not np.array_equal(av.view(np.uint32), bv.view(np.uint32)):
                    raise ValueError(
                        f"GLM HC fused bit gate failed (trial {trial}, expand {expand is not None}, output {index})")
    return True


def route_status(policy="reference"):
    return {"policy": policy, "implemented": True, "qualified": False, "selected_by_default": False,
            "scheduled_calls": dict(_SCHEDULED), "requires": "glm53_hc=mega_v1, B1 single-token rows, Metal"}
