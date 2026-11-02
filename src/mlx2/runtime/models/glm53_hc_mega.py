# SPDX-License-Identifier: MIT
"""Opt-in GLM-5.3 mHC4 boundary in two kernels per half-layer (``mega_v1``).

The fused_fp32 GLM boundary runs, per half-layer: ``rms_norm`` of the flat
4x4096 stream, the native F32 mixing matmul, the single-thread Sinkhorn split,
the four-stream weighted sum, ``rms_norm`` with the half's norm weight and,
after the branch, the four-stream expand. This module keeps the stock mixing
matmul and replaces the rest with two kernels, one threadgroup per row:

``post``  Sinkhorn split (deepseek_v4_hc.SPLIT_SOURCE body, verbatim, on one
          thread), the weighted sum (SUM_SOURCE expression), and the norm in
          ``mx.fast.rms_norm``'s order (four reads per lane, ``simd_sum``, then
          ``simd_sum`` of the 32 partials, ``precise::rsqrt(sum/N + eps)``,
          ``w * (x * inv)``).
``pre``   the expand (EXPAND_SOURCE expression) fused with the next half's
          flat ``rms_norm`` (looped order over 16384 values, no weight).

Verbatim pieces compile under the same default options as deepseek_v4_hc;
only the norm reductions are written with FP contraction off (the law the DS4.1
Source mega kernel matched against MLX 0.32.4). Equality with the ordinary
composition is a first-use GPU gate (``self_check``), not an assumption.
Inspired by deepseek_v41_source_hc_mega (ea78d82ff); GLM's HC normalizes the
flat stream before the mixing matmul, so the kernels differ.
"""

from functools import lru_cache

POLICY = "mega_v1"
LANES = True  # lane-parallel Sinkhorn (False: the serial body on one thread)
FLAT_FMA = True  # flat (looped) rms_norm accumulation as fma(v, v, acc)
HC, DIM = 4, 4096
FLAT = HC * DIM
THREADS = 1024
_SCHEDULED = {"pre": 0, "post": 0}

_HEADER = r"""
// Verbatim pieces (default FP options, as deepseek_v4_hc's kernels).
template<typename M, typename S, typename B>
inline void glm_hc_split(M mixes, S scale, B base, float e, uint ITERS, threadgroup float *out) {
    float c[16];
    for(uint i=0;i<4;i++) {
      const float pre=mixes[i]*scale[0]+base[i];
      const float post=mixes[4+i]*scale[1]+base[4+i];
      out[i]=1.0f/(1.0f+exp(-pre))+e;
      out[4+i]=2.0f/(1.0f+exp(-post));
    }
    for(uint r=0;r<4;r++) {
      float maximum=-INFINITY;
      for(uint s=0;s<4;s++) { c[r*4+s]=mixes[8+r*4+s]*scale[2]+base[8+r*4+s]; maximum=max(maximum,c[r*4+s]); }
      float sum=0.0f;
      for(uint s=0;s<4;s++) { c[r*4+s]=exp(c[r*4+s]-maximum); sum+=c[r*4+s]; }
      for(uint s=0;s<4;s++) c[r*4+s]=c[r*4+s]/sum+e;
    }
    for(uint s=0;s<4;s++) {
      float sum=0.0f;
      for(uint r=0;r<4;r++) sum+=c[r*4+s];
      for(uint r=0;r<4;r++) c[r*4+s]/=(sum+e);
    }
    for(uint it=1;it<ITERS;it++) {
      for(uint r=0;r<4;r++) {
        float sum=0.0f;
        for(uint s=0;s<4;s++) sum+=c[r*4+s];
        for(uint s=0;s<4;s++) c[r*4+s]/=(sum+e);
      }
      for(uint s=0;s<4;s++) {
        float sum=0.0f;
        for(uint r=0;r<4;r++) sum+=c[r*4+s];
        for(uint r=0;r<4;r++) c[r*4+s]/=(sum+e);
      }
    }
    for(uint i=0;i<16;i++) out[8+i]=c[i];
}
template<typename V>
inline float glm_hc_sum(V value, threadgroup const float *weights, uint d, uint DIMS) {
    float sum=0.0f;
    for(uint h=0;h<4;h++) sum+=value[h*DIMS+d]*weights[h];
    return sum;
}
template<typename Br, typename R, typename P, typename C>
inline float glm_hc_expand(Br branch, R residual, P post, C comb, uint h, uint d, uint DIMS) {
    float sum=branch[d]*post[h];
    for(uint s=0;s<4;s++) sum+=residual[s*DIMS+d]*comb[s*4+h];
    return sum;
}
// Lane-parallel form of glm_hc_split for simdgroup 0: lane e<16 owns comb
// element (r,s)=(e/4,e%4), lanes 16..19 pre, 20..23 post. Each four-term sum
// gathers its operands by simd_shuffle and adds them from 0 in the serial
// body's order, so every operation is the serial one on another lane.
template<typename M, typename S, typename B>
inline void glm_hc_split_lanes(M mixes, S scale, B base, float e, uint ITERS, uint lane, threadgroup float *out) {
    if(lane>=16 && lane<20) {
        const uint i=lane-16;
        const float pre=mixes[i]*scale[0]+base[i];
        out[i]=1.0f/(1.0f+exp(-pre))+e;
    } else if(lane>=20 && lane<24) {
        const uint i=lane-20;
        const float post=mixes[4+i]*scale[1]+base[4+i];
        out[4+i]=2.0f/(1.0f+exp(-post));
    }
    const uint el=lane&15u, r=el/4, s=el%4;
    float c=mixes[8+el]*scale[2]+base[8+el];
    float maximum=-INFINITY;
    for(uint k=0;k<4;k++) maximum=max(maximum,simd_shuffle(c,ushort(r*4+k)));
    c=exp(c-maximum);
    {
        float sum=0.0f;
        for(uint k=0;k<4;k++) sum+=simd_shuffle(c,ushort(r*4+k));
        c=c/sum+e;
    }
    {
        float sum=0.0f;
        for(uint k=0;k<4;k++) sum+=simd_shuffle(c,ushort(k*4+s));
        c/=(sum+e);
    }
    for(uint it=1;it<ITERS;it++) {
        {
            float sum=0.0f;
            for(uint k=0;k<4;k++) sum+=simd_shuffle(c,ushort(r*4+k));
            c/=(sum+e);
        }
        {
            float sum=0.0f;
            for(uint k=0;k<4;k++) sum+=simd_shuffle(c,ushort(k*4+s));
            c/=(sum+e);
        }
    }
    if(lane<16) out[8+lane]=c;
}
// Norm reductions in mx.fast.rms_norm's order, FP contraction off.
#pragma clang fp contract(off)
inline float glm_hc_sq_add(float acc, float v) { const float sq=v*v; return acc+sq; }
inline float glm_hc_sq_fma(float acc, float v) { return metal::fma(v, v, acc); }
inline float glm_hc_inv(float total, float n, float eps) { const float mean=total/n; return metal::precise::rsqrt(mean+eps); }
inline float glm_hc_scale(float w, float x, float inv) { const float y=x*inv; return w*y; }
"""

_POST_SOURCE = r"""
threadgroup float coeff[24];
threadgroup float partials[32];
threadgroup float inv_shared;
const uint row=threadgroup_position_in_grid.x;
const uint lid=thread_position_in_threadgroup.x;
const uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
auto hr=h+size_t(row)*FLAT;
if(LANES) { if(sg==0) glm_hc_split_lanes(mixes+row*24, scale, base, constants[0], ITERS, lane, coeff); }
else if(lid==0) glm_hc_split(mixes+row*24, scale, base, constants[0], ITERS, coeff);
threadgroup_barrier(mem_flags::mem_threadgroup);
if(lid<24) out_coeff[row*24+lid]=coeff[lid];
float xs[4];
float acc=0.0f;
for(uint i=0;i<4;i++) {
    const uint d=lid*4+i;
    xs[i]=glm_hc_sum(hr, coeff, d, DIM);
    acc=glm_hc_sq_add(acc, xs[i]);
}
acc=simd_sum(acc);
if(lane==0) partials[sg]=acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0) {
    const float total=simd_sum(partials[lane]);
    if(lane==0) inv_shared=glm_hc_inv(total, float(DIM), constants[1]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float inv=inv_shared;
for(uint i=0;i<4;i++) {
    const uint d=lid*4+i;
    x[size_t(row)*DIM+d]=glm_hc_scale(weight[d], xs[i], inv);
}
"""

_PRE_SOURCE = r"""
threadgroup float partials[32];
threadgroup float inv_shared;
const uint row=threadgroup_position_in_grid.x;
const uint lid=thread_position_in_threadgroup.x;
const uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
auto br=branch+size_t(row)*DIM;
auto rr=residual+size_t(row)*FLAT;
auto pr=post+row*4;
auto cr=comb+row*16;
float vals[16];
float acc=0.0f;
for(uint r=0,k=0;r<FLAT;r+=TG*4) {
    for(uint i=0;i<4;i++,k++) {
        const uint index=r+lid*4+i;
        const float v=glm_hc_expand(br, rr, pr, cr, index/DIM, index%DIM, DIM);
        vals[k]=v;
        h_out[size_t(row)*FLAT+index]=v;
        acc=FLATFMA ? glm_hc_sq_fma(acc, v) : glm_hc_sq_add(acc, v);
    }
}
acc=simd_sum(acc);
if(lane==0) partials[sg]=acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0) {
    const float total=simd_sum(partials[lane]);
    if(lane==0) inv_shared=glm_hc_inv(total, float(FLAT), constants[1]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float inv=inv_shared;
for(uint r=0,k=0;r<FLAT;r+=TG*4)
    for(uint i=0;i<4;i++,k++) z[size_t(row)*FLAT+r+lid*4+i]=glm_hc_scale(1.0f, vals[k], inv);
"""


@lru_cache(maxsize=2)
def _kernel(kind):
    import mlx.core as mx

    if kind == "post":
        names, outs, source = ["h", "mixes", "scale", "base", "weight", "constants"], ["out_coeff", "x"], _POST_SOURCE
    else:
        names, outs, source = ["branch", "residual", "post", "comb", "constants"], ["h_out", "z"], _PRE_SOURCE
    return mx.fast.metal_kernel(name=f"mlx2_glm53_hc_mega_{kind}_v1", input_names=names, output_names=outs,
                                header=_HEADER, source=source, ensure_row_contiguous=True)


@lru_cache(maxsize=8)
def _constants(hc_eps, rms_eps):
    """The kernels' [hc_eps, rms_eps] input; built once per pair (a constant
    input array, so reusing it changes no value)."""
    import mlx.core as mx

    return mx.array([hc_eps, rms_eps], dtype=mx.float32)


def post(h, mixes, scale, base, weight, *, rms_eps, hc_eps=1e-6, iterations=20):
    """(x normalized with weight, pre, post, comb) from h [...,4,4096] and mixes [...,24]."""
    import mlx.core as mx

    lead = h.shape[:-2]
    rows = 1
    for n in lead:
        rows *= n
    coeff, x = _kernel("post")(
        inputs=[h.astype(mx.float32), mixes.astype(mx.float32), scale.astype(mx.float32), base.astype(mx.float32),
                weight.astype(mx.float32), _constants(hc_eps, rms_eps)],
        template=[("TG", THREADS), ("DIM", DIM), ("FLAT", FLAT), ("ITERS", iterations), ("LANES", int(LANES))],
        grid=(rows * THREADS, 1, 1), threadgroup=(THREADS, 1, 1),
        output_shapes=[(*lead, 24), (*lead, DIM)], output_dtypes=[mx.float32, mx.float32])
    _SCHEDULED["post"] += 1
    return x, coeff[..., :4], coeff[..., 4:8], coeff[..., 8:].reshape(*lead, 4, 4)


def pre(branch, residual, post_, comb, *, rms_eps):
    """(expanded h [...,4,4096], rms-normalized flat z [...,16384]) in one kernel."""
    import mlx.core as mx

    lead = residual.shape[:-2]
    rows = 1
    for n in lead:
        rows *= n
    h, z = _kernel("pre")(
        inputs=[branch.astype(mx.float32), residual.astype(mx.float32), post_.astype(mx.float32),
                comb.astype(mx.float32), _constants(1e-6, rms_eps)],
        template=[("TG", THREADS), ("DIM", DIM), ("FLAT", FLAT), ("FLATFMA", int(FLAT_FMA))],
        grid=(rows * THREADS, 1, 1), threadgroup=(THREADS, 1, 1),
        output_shapes=[residual.shape, (*lead, FLAT)], output_dtypes=[mx.float32, mx.float32])
    _SCHEDULED["pre"] += 1
    return h, z


def reference_post(h, mixes, scale, base, weight, *, rms_eps, hc_eps=1e-6, iterations=20):
    import mlx.core as mx

    from .deepseek_v4_hc import split_sinkhorn, weighted_sum

    pre_, post_, comb = split_sinkhorn(mixes, scale, base, iterations=iterations, eps=hc_eps)
    x = mx.fast.rms_norm(weighted_sum(h, pre_).astype(mx.float32), weight, rms_eps)
    return x, pre_, post_, comb


def reference_pre(branch, residual, post_, comb, *, rms_eps):
    import mlx.core as mx

    from .deepseek_v4_hc import expand

    h = expand(branch, residual, post_, comb)
    return h, mx.fast.rms_norm(h.reshape(*h.shape[:-2], -1).astype(mx.float32), None, rms_eps)


@lru_cache(maxsize=1)
def self_check(rms_eps=1e-5):
    """First-use bitwise GPU gate against the fused_fp32 composition."""
    import mlx.core as mx
    import numpy as np

    rng = np.random.default_rng(5301)
    # Served geometry only: one row. MLX picks other rms_norm kernels for
    # more rows (64 rows of 4096 failed this gate), so multi-row calls stay
    # on the ordinary composition (glm53_gguf._hc_in).
    for trial, (magnitude, rows) in enumerate(((0.05, 1), (1.0, 1), (7.0, 1), (30.0, 1), (300.0, 1))):
        h = mx.array((rng.normal(size=(1, rows, HC, DIM)) * magnitude).astype(np.float32))
        mixes = mx.array(rng.normal(size=(1, rows, 24)).astype(np.float32) * 3)
        scale = mx.array(rng.uniform(0.1, 2.0, size=3).astype(np.float32))
        base = mx.array(rng.normal(size=24).astype(np.float32))
        weight = mx.array(rng.normal(size=DIM).astype(np.float32))
        branch = mx.array((rng.normal(size=(1, rows, DIM)) * magnitude).astype(np.float32))
        actual = post(h, mixes, scale, base, weight, rms_eps=rms_eps)
        expected = reference_post(h, mixes, scale, base, weight, rms_eps=rms_eps)
        a2 = pre(branch, h, actual[2], actual[3], rms_eps=rms_eps)
        e2 = reference_pre(branch, h, actual[2], actual[3], rms_eps=rms_eps)
        pairs = list(zip(actual, expected)) + list(zip(a2, e2))
        mx.eval(pairs)
        for index, (a, b) in enumerate(pairs):
            av, bv = np.array(a), np.array(b)
            if av.shape != bv.shape or not np.array_equal(av.view(np.uint32), bv.view(np.uint32)):
                diff = int(np.count_nonzero(av.view(np.uint32) != bv.view(np.uint32))) if av.shape == bv.shape else -1
                raise ValueError(f"GLM HC mega bit gate failed (trial {trial}, output {index}, {diff} words)")
    return True


def route_status():
    return {"policy": POLICY, "implemented": True, "qualified": False, "selected_by_default": False,
            "scheduled_calls": dict(_SCHEDULED), "stock_frontiers": ["hc_*_fn F32 mixing matmul"]}
