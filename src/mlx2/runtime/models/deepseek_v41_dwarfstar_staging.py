# SPDX-License-Identifier: MIT
"""Opt-in exact preparation fusion; original attention arithmetic stays intact.

Adapted packed copy/cast pattern from antirez/ds4 metal/cpy.metal at ee7e840.
Retain provenance/dwarfstar-flash-attention.NOTICE.txt. The compressed source
is F32 here, so its cast matches the existing MLX preparation instead of the
upstream F16 bit-copy. No weight/cache representation or persistent owner.
"""
from functools import lru_cache

HEADER='#include <metal_stdlib>\nusing namespace metal;\n'
SOURCE=r'''
const uint gid=thread_position_in_grid.x;
constexpr uint row_vecs=128;
const uint raw_vecs=LOCAL_ROWS*row_vecs;
const uint total_vecs=(LOCAL_ROWS+POOL_ROWS)*row_vecs;
device packed_half4* dst_half=(device packed_half4*)out_kv;
if(gid<raw_vecs) {
 const device packed_float4* raw=(const device packed_float4*)raw_src;
 const float4 value=float4(raw[gid]);
 dst_half[gid]=packed_half4(half4(value));
} else if(gid<total_vecs) {
 const device packed_float4* comp=(const device packed_float4*)comp_src;
 const float4 value=float4(comp[gid-raw_vecs]);
 dst_half[gid]=packed_half4(half4(value));
} else if(gid<PADDED_ROWS*row_vecs) {
 dst_half[gid]=packed_half4(half4(0.0h));
}
if(gid<PADDED_ROWS) {
 const bool live=gid<LOCAL_ROWS ? local_mask[gid] :
     gid<LOCAL_ROWS+POOL_ROWS ? valid[gid-LOCAL_ROWS] : false;
 out_mask[gid]=live ? half(0.0f) : half(-65504.0f);
}
if(gid==0)out_blocks[0]=char(0);
'''


def layout(local_rows,pool_rows):
    from .deepseek_v41_dwarfstar_attention import plan
    p=plan(1,local_rows,pool_rows);elements=p['padded_keys']*128
    return dict(local_rows=local_rows,pool_rows=pool_rows,padded_rows=p['padded_keys'],
        grid=(elements,1,1),threadgroup=(256,1,1),
        output_shapes=[(p['padded_keys'],512),(1,p['padded_keys']),(1,)],
        input_bindings=4,output_bindings=3,threadgroup_memory_bytes=0,
        logical_F32_concat_intermediate_eliminated_bytes=p['padded_keys']*512*4,
        accounting_scope='Logical intermediatepayload, notmeasuredpeak/DRAM')


def geometry(raw,pooled,local_mask,valid):
    import mlx.core as mx
    if raw.ndim!=4 or raw.shape[:2]!=(1,1)or raw.shape[-1]!=512:
        raise ValueError('Fused decode stage requires onechronologicallocalKVlatent')
    if pooled.ndim!=3 or pooled.shape[0]!=1 or pooled.shape[-1]!=512:
        raise ValueError('Fused decode stage requires oneglobalKVlatent')
    p=layout(raw.shape[2],pooled.shape[1])
    if (local_mask.shape!=(1,1,1,p['local_rows'])or valid.shape!=(1,1,p['pool_rows'])
            or raw.dtype!=mx.float32 or pooled.dtype!=mx.float32
            or local_mask.dtype!=mx.bool_ or valid.dtype!=mx.bool_):
        raise ValueError('Fused decode staging dtype/maskgeometry mismatch')
    return p


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='mlx2_dwarfstar_fused_decode_stage_v1',
        input_names=['raw_src','comp_src','local_mask','valid'],output_names=['out_kv','out_mask','out_blocks'],
        header=HEADER,source=SOURCE,ensure_row_contiguous=True,compile_options={'math_mode':'fast'})


def stage(raw,pooled,local_mask,valid):
    import mlx.core as mx
    p=geometry(raw,pooled,local_mask,valid)
    if mx.default_device()!=mx.gpu or not mx.metal.is_available():
        raise ValueError('Fused decode stage requires caller-owned Metal GPU')
    return _kernel()(inputs=[raw,pooled,local_mask,valid],
        template=[('LOCAL_ROWS',p['local_rows']),('POOL_ROWS',p['pool_rows']),('PADDED_ROWS',p['padded_rows'])],
        grid=p['grid'],threadgroup=p['threadgroup'],output_shapes=p['output_shapes'],output_dtypes=[mx.float16,mx.float16,mx.int8])


def reference_numpy(np,raw,pooled,local_mask,valid):
    """CPU preparation oracle; never used as an inference fallback."""
    p=layout(raw.shape[2],pooled.shape[1]);n=p['padded_rows'];keys=p['local_rows']+p['pool_rows']
    bank=np.zeros((n,512),np.float16)
    bank[:p['local_rows']]=raw[0,0].astype(np.float16)
    bank[p['local_rows']:keys]=pooled[0].astype(np.float16)
    mask=np.full((1,n),np.float16(-65504));live=np.concatenate((local_mask[0,0],valid[0]),1)
    mask[:,:keys]=np.where(live,np.float16(0),np.float16(-65504))
    return bank,mask,np.zeros((1,),np.int8)


def compilation_source(local_rows=101,pool_rows=50):
    p=layout(local_rows,pool_rows)
    constants=f'constant uint LOCAL_ROWS={local_rows},POOL_ROWS={pool_rows},PADDED_ROWS={p["padded_rows"]};\n'
    return HEADER+constants+'''kernel void stage_probe(
const device float* raw_src [[buffer(0)]],const device float* comp_src [[buffer(1)]],
const device bool* local_mask [[buffer(2)]],const device bool* valid [[buffer(3)]],
device half* out_kv [[buffer(4)]],device half* out_mask [[buffer(5)]],device char* out_blocks [[buffer(6)]],
uint3 thread_position_in_grid [[thread_position_in_grid]]) {
'''+SOURCE+'\n}\n'
