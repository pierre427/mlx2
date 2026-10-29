# SPDX-License-Identifier: MIT
"""Explicit Source B1 short-band attention candidates.

Both profiles change the reference reduction graph and require their own APCv2
identity. Joint math runs on the caller's device; native dispatch refuses CPU.
Importing or generating a kernel specification performs no GPU work.
Provenance: provenance/deepseek-v41-short-attention-candidates.json.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

JOINT_PROFILE = 'source_short_decode_joint_f32_v1'
NATIVE_PROFILE = 'source_short_decode_grouped_f32_v1'
HEADS, DIM, MAX_LOCAL, MAX_SELECTED = 64, 512, 128, 512


def geometry(q, local, mask, pooled, indices, valid, sinks, *, gathered=None):
    """Structural admission only; route values retain the selector's contract."""
    import mlx.core as mx
    if (q.shape != (1, HEADS, 1, DIM) or q.dtype != mx.float32
            or local.ndim != 4 or local.shape[:2] != (1, 1)
            or local.shape[-1] != DIM or not 1 <= local.shape[2] <= MAX_LOCAL
            or local.dtype != mx.float32 or mask.dtype != mx.bool_
            or mask.shape != (1, 1, 1, local.shape[2])
            or sinks.shape != (HEADS,) or sinks.dtype != mx.float32):
        raise ValueError('Source short attention requires F32 B1/H64/M1/D512 and bool local mask')
    count = 0
    if pooled is None:
        if any(x is not None for x in (indices, valid, gathered)):
            raise ValueError('Source short attention cannot select rows without pooled storage')
    else:
        if (pooled.ndim != 3 or pooled.shape[0] != 1 or pooled.shape[2] != DIM
                or pooled.dtype != mx.float32 or indices is None or valid is None
                or indices.ndim != 3 or indices.shape[:2] != (1, 1)
                or not 0 <= indices.shape[-1] <= min(MAX_SELECTED, pooled.shape[1])
                or indices.dtype not in (mx.int32, mx.uint32)
                or valid.shape != indices.shape or valid.dtype != mx.bool_):
            raise ValueError('Source short attention selected-row geometry mismatch')
        count = indices.shape[-1]
        if gathered is not None and (gathered.shape != (1, 1, count, DIM)
                                     or gathered.dtype != mx.float32):
            raise ValueError('Source short attention reused gather geometry mismatch')
    return dict(local_rows=local.shape[2], selected_rows=count,
                band_rows=local.shape[2]+count, output_shape=q.shape)


def _operands(q, local, mask, pooled, indices, valid, sinks, gathered):
    import mlx.core as mx
    p = geometry(q, local, mask, pooled, indices, valid, sinks, gathered=gathered)
    if p['selected_rows']:
        # Invalid slots may carry arbitrary indices and NaN padding. They must
        # neither read an invalid address nor poison a zero-weight PV product.
        if gathered is None:
            safe = mx.where(valid, indices, mx.zeros_like(indices))
            rows = pooled[0, safe[0]][None]
        else:
            rows = gathered
        selected = mx.where(valid.reshape(1, 1, -1, 1), rows, 0.0)
        keep = mx.concatenate((mask.reshape(1, 1, -1), valid), axis=-1)
    else:
        selected = mx.zeros((1, 1, 0, DIM), mx.float32)
        keep = mask.reshape(1, 1, -1)
    local_safe = mx.where(mask.reshape(1, 1, -1, 1), local, 0.0)
    return p, local_safe, selected, keep


def joint_attention(q, local, mask, pooled, indices, valid, sinks, *,
                    gathered=None, numerical_profile):
    """One F32 joined score/softmax/PV graph, including the zero-valued sink.

    Masked storage is zeroed before QK/PV. This is a distinct numerical profile,
    rather than a claim of raw identity to the two-segment reference graph.
    """
    if numerical_profile != JOINT_PROFILE:
        raise ValueError('Source joint attention requires its explicit numerical profile')
    import mlx.core as mx
    _, local_safe, selected, keep = _operands(
        q, local, mask, pooled, indices, valid, sinks, gathered)
    kv = mx.concatenate((local_safe, selected), axis=2)
    scores = (q * DIM**-0.5) @ mx.swapaxes(kv, -1, -2)
    scores = mx.where(keep[:, :, None], scores, -mx.inf)
    full = mx.concatenate((scores, sinks.reshape(1, HEADS, 1, 1)), axis=-1)
    norm = mx.logsumexp(full, axis=-1, keepdims=True)
    return mx.exp(scores-norm) @ kv


def kernel_spec(*, head_group=2):
    """Generate one of three finite shader families; lengths remain dynamic."""
    if type(head_group) is not int or head_group not in (1, 2, 4):
        raise ValueError('Source short attention head group must be 1, 2 or 4')
    source = r'''
constexpr uint D=512, MAX_BAND=640, HG=HEAD_GROUP;
const uint tid=thread_index_in_threadgroup;
const uint lane=tid%32, sg=tid/32;
const uint head0=threadgroup_position_in_grid.x*HG;
const uint L=lengths[0], P=lengths[1], N=L+P;
threadgroup float weights[HG*MAX_BAND];
threadgroup float partials[8];
threadgroup float scalar;
for(uint key=sg;key<N;key+=8) {
    const bool live=keep[key];
    float accum[HG];
    for(uint h=0;h<HG;++h)accum[h]=0.0f;
    if(live) {
        for(uint d=lane;d<D;d+=32) {
            const float value=key<L?local[key*D+d]:selected[(key-L)*D+d];
            for(uint h=0;h<HG;++h)
                accum[h]+=(q[(head0+h)*D+d]*0.04419417382415922f)*value;
        }
    }
    for(uint h=0;h<HG;++h) {
        const float dot=simd_sum(accum[h]);
        if(lane==0)weights[h*MAX_BAND+key]=live?dot:-INFINITY;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint h=0;h<HG;++h) {
    float maximum=tid==0?sinks[head0+h]:-INFINITY;
    for(uint key=tid;key<N;key+=256)maximum=max(maximum,weights[h*MAX_BAND+key]);
    maximum=simd_max(maximum);
    if(lane==0)partials[sg]=maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        const float merged=simd_max(lane<8?partials[lane]:-INFINITY);
        if(lane==0)scalar=merged;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float top=scalar;
    float denominator=tid==0?metal::exp(sinks[head0+h]-top):0.0f;
    for(uint key=tid;key<N;key+=256) {
        const float mass=metal::exp(weights[h*MAX_BAND+key]-top);
        weights[h*MAX_BAND+key]=mass;
        denominator+=mass;
    }
    denominator=simd_sum(denominator);
    if(lane==0)partials[sg]=denominator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        const float merged=simd_sum(lane<8?partials[lane]:0.0f);
        if(lane==0)scalar=merged;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total=scalar;
    for(uint key=tid;key<N;key+=256)weights[h*MAX_BAND+key]/=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for(uint d=tid;d<D;d+=256) {
    float accum[HG];
    for(uint h=0;h<HG;++h)accum[h]=0.0f;
    for(uint key=0;key<N;++key)if(keep[key]) {
        const float value=key<L?local[key*D+d]:selected[(key-L)*D+d];
        for(uint h=0;h<HG;++h)accum[h]+=weights[h*MAX_BAND+key]*value;
    }
    for(uint h=0;h<HG;++h)out[(head0+h)*D+d]=accum[h];
}
'''.replace('HEAD_GROUP', str(head_group))
    return dict(profile=NATIVE_PROFILE, head_group=head_group,
                header='#include <metal_stdlib>\nusing namespace metal;\n',
                source=source, input_names=['q', 'local', 'selected', 'keep', 'sinks', 'lengths'],
                output_names=['out'], grid=(HEADS//head_group*256, 1, 1),
                threadgroup=(256, 1, 1), threadgroup_scratch_bytes=(head_group*640+9)*4,
                output_shape=(1, HEADS, 1, DIM), dynamic_lengths=True,
                numerical_contract='F32 shared-key grouped SIMD QK; merged sink softmax; ordered F32 PV; distinct unqualified profile')


@lru_cache(maxsize=3)
def _kernel(head_group):
    import mlx.core as mx
    s = kernel_spec(head_group=head_group)
    return mx.fast.metal_kernel(name=f'mlx2_source_short_attention_h{head_group}_v1',
        input_names=s['input_names'], output_names=s['output_names'],
        header=s['header'], source=s['source'], ensure_row_contiguous=True)


def native_attention(q, local, mask, pooled, indices, valid, sinks, *,
                     gathered=None, numerical_profile, head_group=2):
    """Explicit native candidate; device admission precedes kernel creation."""
    if numerical_profile != NATIVE_PROFILE:
        raise ValueError('Source native short attention requires its numerical profile')
    import mlx.core as mx
    p = geometry(q, local, mask, pooled, indices, valid, sinks, gathered=gathered)
    s = kernel_spec(head_group=head_group)
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise ValueError('Source native short attention requires an owned Metal device')
    _, local_safe, selected, keep = _operands(
        q, local, mask, pooled, indices, valid, sinks, gathered)
    lengths = mx.array([p['local_rows'], p['selected_rows']], mx.uint32)
    return _kernel(head_group)(inputs=[q, local_safe, selected, keep, sinks, lengths],
        grid=s['grid'], threadgroup=s['threadgroup'], output_shapes=[s['output_shape']],
        output_dtypes=[mx.float32])[0]


def implementation_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def dispatch(q, local, mask, pooled, indices, valid, sinks, *, policy,
             gathered=None, reference, counters):
    """Adapter helper: unsupported geometry returns the caller-owned reference."""
    if policy not in (JOINT_PROFILE, NATIVE_PROFILE):
        raise ValueError('Unknown Source short attention policy')
    try:
        geometry(q, local, mask, pooled, indices, valid, sinks, gathered=gathered)
    except (ValueError, TypeError, AttributeError, IndexError):
        key = 'source_short_attention_reference_geometry'
        counters[key] = counters.get(key, 0)+1
        return reference()
    if policy == NATIVE_PROFILE:
        import mlx.core as mx
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            key = 'source_short_attention_reference_device'
            counters[key] = counters.get(key, 0)+1
            return reference()
    function = joint_attention if policy == JOINT_PROFILE else native_attention
    value = function(q, local, mask, pooled, indices, valid, sinks,
                     gathered=gathered, numerical_profile=policy)
    key = 'source_short_attention_joint' if policy == JOINT_PROFILE else 'source_short_attention_native'
    counters[key] = counters.get(key, 0)+1
    return value
