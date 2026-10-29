# SPDX-License-Identifier: MIT
"""Opt-in single-pass Source mHC decode boundary (``mega_v1``), B1 M1 only.

The ordinary Source decode boundary runs about 38 dependent launches per layer
for hyper-connection work: per half, the stock coefficient matmul, then
square, sum, mean scale, epsilon add, rsqrt, multiply, Sinkhorn split, the
stream collapse multiply and sum, a BF16 round, the norm-weight cast,
``fast.rms_norm`` and another round; and per residual update a multiply, the
stock comb matmul, an add and a round. This module keeps both stock matmuls
(``flat @ fn.T`` and ``comb^T @ residual``) and replaces everything else with
two kernels:

``pre``    one 1024-thread threadgroup: the mixes RMS scale in MLX's
           ``row_reduce_looped`` law (1024 lanes, four reads, ``x + total``,
           ``simd_sum`` then threadgroup ``simd_sum``; mean as ``sum * (1/N)``),
           ``precise::rsqrt``, the unchanged Source Sinkhorn body from
           ``deepseek_v41_source_hc`` on one lane, the four-stream collapse in
           column-reduce order, a BF16 round, ``rms_looped``'s 1024-lane
           ``fast.rms_norm`` law with ``precise::rsqrt`` and ``w * (x * inv)``,
           and the final BF16 round.
``store``  ``bf16(post * y + mixed)``, with ``y = branch`` or
           ``y = bf16(routed + shared)`` for the FFN finish.

The reduction laws were matched empirically against installed MLX 0.32.4 on
this machine; equality is a first-use GPU gate (``self_check``), not an
assumption. Any mismatch raises before serving. Other row counts, dtypes and
geometries are refused by ``can_pre``/``can_store`` and stay on the ordinary
composition. See provenance/deepseek-v41-source-hc-mega.json.
"""
from functools import lru_cache
import hashlib
import math
from pathlib import Path

POLICY = 'mega_v1'
POLICIES = ('reference', POLICY)
HC = 4
DIM = 5120
FLAT = HC * DIM
THREADS = 1024
_SCHEDULED = {}

_HEADER = r'''
#pragma clang fp contract(off)
inline float mega_bf16(float x) {
    uint bits=as_type<uint>(x);
    return as_type<float>((bits+0x7fffu+((bits>>16)&1u))&0xffff0000u);
}
'''

_PRE_SOURCE = r'''
threadgroup float sum_partials[32];
threadgroup float rms_partials[32];
threadgroup float shared_scalars[2];
threadgroup float tg_mixes[24];
threadgroup float collapsed[DIM];
const uint lid=thread_position_in_threadgroup.x;
const uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
// (1) mean(square(flat)) in row_reduce_looped order.
float total=0.0f;
for(uint r=0;r<FLAT;r+=TG*4) {
    for(uint i=0;i<4;i++) {
        const float v=h[r+lid*4+i];
        const float sq=v*v;
        total=sq+total;
    }
}
total=simd_sum(total);
if(lane==0)sum_partials[sg]=total;
// (2) collapse with the previous pre, BF16 round, rms_looped accumulation.
float acc=0.0f;
for(uint r=0;r<DIM;r+=TG*4) {
    if(r+lid*4+4<=DIM) {
        for(uint i=0;i<4;i++) {
            const uint d=r+lid*4+i;
            float c=0.0f;
            for(uint s=0;s<4;s++) {
                const float product=h[s*DIM+d]*prev_pre[s];
                c=product+c;
            }
            const float xi=mega_bf16(c);
            collapsed[d]=xi;
            const float sq=xi*xi;
            acc=acc+sq;
        }
    }
}
acc=simd_sum(acc);
if(sg==0)rms_partials[lane]=0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(lane==0)rms_partials[sg]=acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0) {
    const float tv=(lid<TG/32)?sum_partials[lid]:0.0f;
    const float flat_total=simd_sum(tv);
    const float rv=rms_partials[lane];
    const float rms_total=simd_sum(rv);
    if(lane==0) {
        const float mean=flat_total*constants[0];
        const float denominator=mean+constants[1];
        shared_scalars[0]=metal::precise::rsqrt(denominator);
        const float rms_mean=rms_total/float(DIM);
        const float rms_denominator=rms_mean+constants[1];
        shared_scalars[1]=metal::precise::rsqrt(rms_denominator);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(lid<24)tg_mixes[lid]=raw[lid]*shared_scalars[0];
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0) {
SINKHORN_BODY
}
const float inverse=shared_scalars[1];
for(uint r=0;r<DIM;r+=TG*4) {
    if(r+lid*4+4<=DIM) {
        for(uint i=0;i<4;i++) {
            const uint d=r+lid*4+i;
            const float normalized=collapsed[d]*inverse;
            const float weighted=static_cast<float>(weight[d])*normalized;
            x[d]=mega_bf16(weighted);
        }
    }
}
'''

_STORE_SOURCE = r'''
const uint i=thread_position_in_grid.x;
if(i>=DIM)return;
BRANCH_VALUE
for(uint s=0;s<4;s++) {
    const uint offset=s*DIM+i;
    const float contribution=post[s]*y;
    const float raw=contribution+mixed[offset];
    out[offset]=mega_bf16(raw);
}
'''


def _once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Source HC mega pinned anchor changed: ' + old)
    return text.replace(old, new)


# Lane-parallel transcription of the selected Source split body
# (deepseek_v41_source_hc.SOURCE with reduction="sequential" and
# softmax_exp="metal::fast::exp", the arguments Block.mixes uses). Lane
# L<16 owns comb element L=(r*4+s); lanes 16..23 own pre/post. Each four-term
# sum gathers its operands by simd_shuffle and then runs the same
# source_hc_sum4(...,0) on every lane, so every add and division is the
# serial body's own operation in the same order; only its lane changes.
_PARALLEL_SINKHORN = r'''
const float epsilon=constants[2];
if(lane>=16 && lane<20) {
    const uint i=lane-16;
    const float pre_product=tg_mixes[i]*scale[0];
    const float pre_logit=pre_product+base[i];
    coeff[i]=source_hc_sigmoid(pre_logit)+epsilon;
} else if(lane>=20 && lane<24) {
    const uint i=lane-20;
    const float post_product=tg_mixes[4+i]*scale[1];
    const float post_logit=post_product+base[4+i];
    coeff[4+i]=2.0f*source_hc_sigmoid(post_logit);
}
const uint e=lane&15u, r=e/4, s=e%4;
const float product=tg_mixes[8+e]*scale[2];
const float logit=product+base[8+e];
float maximum=-FLT_MAX;
for(uint k=0;k<4;k++) {
    const float v=simd_shuffle(logit,ushort(r*4+k));
    maximum=maximum<v ? v : maximum;
}
const float value=metal::fast::exp(logit-maximum);
const float total=source_hc_sum4(simd_shuffle(value,ushort(r*4)),simd_shuffle(value,ushort(r*4+1)),
                                 simd_shuffle(value,ushort(r*4+2)),simd_shuffle(value,ushort(r*4+3)),0);
const float normalizer=1.0f/total;
float c=value*normalizer+epsilon;
{
    const float sum=source_hc_sum4(simd_shuffle(c,ushort(s)),simd_shuffle(c,ushort(4+s)),
                                   simd_shuffle(c,ushort(8+s)),simd_shuffle(c,ushort(12+s)),0);
    const float denominator=sum+epsilon;
    c=c/denominator;
}
for(uint iteration=1;iteration<ITERS;iteration++) {
    {
        const float sum=source_hc_sum4(simd_shuffle(c,ushort(r*4)),simd_shuffle(c,ushort(r*4+1)),
                                       simd_shuffle(c,ushort(r*4+2)),simd_shuffle(c,ushort(r*4+3)),0);
        const float denominator=sum+epsilon;
        c=c/denominator;
    }
    {
        const float sum=source_hc_sum4(simd_shuffle(c,ushort(s)),simd_shuffle(c,ushort(4+s)),
                                       simd_shuffle(c,ushort(8+s)),simd_shuffle(c,ushort(12+s)),0);
        const float denominator=sum+epsilon;
        c=c/denominator;
    }
}
if(lane<16)coeff[8+lane]=c;
'''


def _sinkhorn_body(sigmoid_exp):
    """Source split helpers (sigmoid spelling, sum4) and the parallel body.

    The serial body is checked for the anchors this transcription mirrors,
    so a change to the selected Source split fails closed here.
    """
    from .deepseek_v41_source_hc import compilation_source
    header, source = compilation_source(sigmoid_exp=sigmoid_exp)
    for anchor in ('out[row*24+i]=source_hc_sigmoid(pre_logit)+epsilon;',
                   'out[row*24+4+i]=2.0f*source_hc_sigmoid(post_logit);',
                   'values[s]=metal::fast::exp(c[r*4+s]-maximum);',
                   'source_hc_sum4(values[0],values[1],values[2],values[3],0);',
                   'c[r*4+s]=values[s]*inverse+epsilon;',
                   'source_hc_sum4(c[s],c[4+s],c[8+s],c[12+s],0);',
                   'source_hc_sum4(c[i],c[i+1],c[i+2],c[i+3],0);'):
        if not source.count(anchor):
            raise RuntimeError('Source HC mega: selected Sinkhorn body changed: ' + anchor)
    helpers = _once(header, '#pragma clang fp contract(off)\n', '')
    return helpers, _PARALLEL_SINKHORN


def compilation_source(kind, *, sigmoid_exp='metal::precise::exp', shared=False):
    """Metal header/source for CPU inspection; no device work."""
    if kind == 'pre':
        helpers, body = _sinkhorn_body(sigmoid_exp)
        return dict(header=_HEADER + helpers, source=_PRE_SOURCE.replace('SINKHORN_BODY', body),
                    inputs=['h', 'raw', 'prev_pre', 'weight', 'scale', 'base', 'constants'],
                    outputs=['coeff', 'x'])
    if kind == 'store':
        value = ('const float y=mega_bf16(branch[i]+shared[i]);' if shared
                 else 'const float y=branch[i];')
        inputs = ['branch', 'mixed', 'post'] + (['shared'] if shared else [])
        return dict(header=_HEADER, source=_STORE_SOURCE.replace('BRANCH_VALUE', value),
                    inputs=inputs, outputs=['out'])
    raise ValueError('Source HC mega unknown kernel kind')


@lru_cache(maxsize=1)
def implementation_identity():
    """Bind this file, the reused Sinkhorn file and the installed MLX pins."""
    from . import deepseek_v41_source_hc as source_hc
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(source_hc.__file__)):
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
    digest.update(repr(sorted(source_hc.INSTALLED_PINS.items())).encode())
    return hashlib.sha256((POLICY + ':' + digest.hexdigest()).encode()).hexdigest()


def _device():
    from .deepseek_v41_source_hc import _device as owned_device, _verify_installed
    mx = owned_device()
    _verify_installed()
    return mx


@lru_cache(maxsize=4)
def _kernel(kind, sigmoid_exp, shared):
    import mlx.core as mx
    code = compilation_source(kind, sigmoid_exp=sigmoid_exp, shared=shared)
    tag = sigmoid_exp.replace('::', '_')
    return mx.fast.metal_kernel(name=f'mlx2_v41_source_hc_mega_{kind}_{int(shared)}_{tag}_v1',
        input_names=code['inputs'], output_names=code['outputs'], header=code['header'],
        source=code['source'], ensure_row_contiguous=True, compile_options={'math_mode': 'safe'})


def _count(key):
    _SCHEDULED[key] = _SCHEDULED.get(key, 0) + 1


def can_pre(h, prev_pre, fn, scale, base, weight):
    import mlx.core as mx
    values = (h, prev_pre, fn, scale, base, weight)
    if not all(isinstance(v, mx.array) for v in values):
        return False
    return (h.shape == (1, 1, HC, DIM) and h.dtype == mx.float32
            and prev_pre.shape == (1, 1, HC) and prev_pre.dtype == mx.float32
            and fn.shape == (24, FLAT) and fn.dtype == mx.float32
            and scale.shape == (3,) and scale.dtype == mx.float32
            and base.shape == (24,) and base.dtype == mx.float32
            and weight.shape == (DIM,) and weight.dtype in (mx.float32, mx.bfloat16))


def can_store(branch, residual, post, comb, shared=None):
    import mlx.core as mx
    values = (branch, residual, post, comb) + (() if shared is None else (shared,))
    if not all(isinstance(v, mx.array) and v.dtype == mx.float32 for v in values):
        return False
    return (branch.shape == (1, 1, DIM) and residual.shape == (1, 1, HC, DIM)
            and post.shape == (1, 1, HC) and comb.shape == (1, 1, HC, HC)
            and (shared is None or shared.shape == branch.shape))


def _checked_options(iterations, hc_eps, rms_eps):
    if type(iterations) is not int or not 1 <= iterations <= 100:
        raise ValueError('Source HC mega iterations must be in1..100')
    for value in (hc_eps, rms_eps):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError('Source HC mega epsilon must be positive finite')


def pre(h, prev_pre, hc, weight, *, iterations, hc_eps, rms_eps):
    """(pre, post, comb, normalized BF16-valued x) for one decode row.

    ``hc`` is the Block's FP32 (fn, scale, base). The coefficient matmul is
    the ordinary stock expression; all other arithmetic is the fused kernel.
    """
    fn, scale, base = hc
    if not can_pre(h, prev_pre, fn, scale, base, weight):
        raise ValueError('Source HC mega requires B1 M1 HC4x5120 FP32 geometry')
    _checked_options(iterations, hc_eps, rms_eps)
    mx = _device()
    from .deepseek_v41_source_hc import sigmoid_spelling
    import numpy as np
    flat = h.reshape(1, 1, FLAT)
    raw = flat @ fn.T
    constants = mx.array(np.array([1.0 / FLAT, rms_eps, hc_eps], dtype=np.float32))
    coeff, x = _kernel('pre', sigmoid_spelling(), False)(
        inputs=[h, raw, prev_pre, weight, scale, base, constants],
        template=[('TG', THREADS), ('FLAT', FLAT), ('DIM', DIM), ('ITERS', iterations)],
        grid=(THREADS, 1, 1), threadgroup=(THREADS, 1, 1),
        output_shapes=[(1, 1, 24), (1, 1, DIM)], output_dtypes=[mx.float32, mx.float32])
    _count('pre')
    return coeff[..., :4], coeff[..., 4:8], coeff[..., 8:].reshape(1, 1, HC, HC), x


def store(branch, residual, post, comb, shared=None):
    """bf16(post*y + comb^T@residual); ``shared`` selects the FFN finish."""
    if not can_store(branch, residual, post, comb, shared):
        raise ValueError('Source HC mega store requires B1 M1 HC4x5120 FP32 geometry')
    mx = _device()
    from .deepseek_v41_source_hc import sigmoid_spelling
    mixed = mx.matmul(mx.swapaxes(comb, -1, -2), residual.astype(mx.float32))
    inputs = [branch, mixed, post] + ([] if shared is None else [shared])
    out = _kernel('store', sigmoid_spelling(), shared is not None)(
        inputs=inputs, template=[('DIM', DIM)], grid=(DIM, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[residual.shape], output_dtypes=[mx.float32])[0]
    _count('store_shared' if shared is not None else 'store')
    return out


def reference_pre(h, prev_pre, hc, weight, *, iterations, hc_eps, rms_eps):
    """The ordinary Source composition (default coefficient policy), as served."""
    import mlx.core as mx
    from .deepseek_v41_fused import rounded
    from .deepseek_v41_source_hc import split
    fn, scale, base = hc
    flat = h.reshape(*h.shape[:2], -1).astype(mx.float32)
    mixes = (flat @ fn.T) * mx.rsqrt(mx.mean(mx.square(flat), -1, keepdims=True) + rms_eps)
    p, q, c = split(mixes, scale, base, iterations=iterations, eps=hc_eps)
    collapsed = rounded(mx.sum(h.astype(mx.float32) * prev_pre[..., None], -2))
    x = rounded(mx.fast.rms_norm(collapsed.astype(mx.float32), weight.astype(mx.float32), rms_eps))
    return p, q, c, x


def reference_store(branch, residual, post, comb, shared=None):
    import mlx.core as mx
    from .deepseek_v4 import hc_expand
    from .deepseek_v41_fused import rounded
    if shared is not None:
        branch = rounded(branch + shared)
    return rounded(hc_expand(branch, residual, post, comb))


def _bits(value):
    import numpy as np
    return np.array(value).view(np.uint32)


@lru_cache(maxsize=1)
def self_check(iterations=20, hc_eps=1e-6, rms_eps=1e-20):
    """First-use bitwise GPU gate at production geometry; raises on mismatch."""
    import mlx.core as mx
    import numpy as np
    rng = np.random.default_rng(4111)
    for trial in range(4):
        magnitude = (0.05, 1.0, 7.0, 30.0)[trial]
        h = mx.array((rng.normal(size=(1, 1, HC, DIM)) * magnitude).astype(np.float32))
        h = h.astype(mx.bfloat16).astype(mx.float32)
        fn = mx.array((rng.normal(size=(24, FLAT)) * 0.01).astype(np.float32))
        scale = mx.array(rng.uniform(0.1, 2.0, size=3).astype(np.float32))
        base = mx.array(rng.normal(size=24).astype(np.float32))
        prev = mx.array(rng.uniform(0.0, 1.2, size=(1, 1, HC)).astype(np.float32))
        weight = mx.array(rng.normal(size=DIM).astype(np.float32)).astype(
            mx.bfloat16 if trial % 2 else mx.float32)
        hc = (fn, scale, base)
        options = dict(iterations=iterations, hc_eps=hc_eps, rms_eps=rms_eps)
        actual, expected = pre(h, prev, hc, weight, **options), reference_pre(h, prev, hc, weight, **options)
        branch = mx.array((rng.normal(size=(1, 1, DIM)) * magnitude).astype(np.float32))
        shared = mx.array((rng.normal(size=(1, 1, DIM)) * magnitude).astype(np.float32))
        pairs = list(zip(actual, expected))
        for extra in (None, shared):
            pairs.append((store(branch, h, actual[1], actual[2], extra),
                          reference_store(branch, h, actual[1], actual[2], extra)))
        mx.eval(pairs)
        for index, (a, b) in enumerate(pairs):
            if a.shape != b.shape or not np.array_equal(_bits(a), _bits(b)):
                raise ValueError(f'Source HC mega first-use bit gate failed (trial {trial}, output {index})')
    return True


def route_status():
    return {'policy': POLICY, 'implemented': True, 'qualified': False, 'selected_by_default': False,
            'scheduled_calls': dict(_SCHEDULED), 'gpu_completion_not_implied': True,
            'stock_frontiers': ['flat @ fn.T', 'comb^T @ residual']}
