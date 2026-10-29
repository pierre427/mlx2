# SPDX-License-Identifier: MIT
"""Unwired Source output preparation; all stock matrix products stay external.

Provenance: deepseek-v41-source-output-chain.json. These candidates retain
both BF16 owner boundaries and FP8 group32 semantics. They need actual GPU
component proof; CPU/source/offline evidence is not serving qualification.
"""
from functools import lru_cache


def layout(rows,heads=64,dim=512,groups=8):
    if (type(rows)is not int or rows not in (1,99)or heads!=64 or dim!=512 or groups!=8):
        raise ValueError('Source output chain admits only captured H64/D512/G8 M1/M99')
    return dict(input_shape=(1,heads,rows,dim),grouped_shape=(1,rows,groups,heads//groups*dim),
        wo_a_shape=(1,rows,groups,1024),prepared_wo_b_shape=(1,rows,8192),
        elements=heads*rows*dim,group32_blocks=heads*rows*dim//32,
        input_owner_round_retained=True,inverse_rotary_owner_round_retained=True,
        wo_a_input_owner_round_retained=True,coefficient_repack_bytes=0,
        logical_intermediate_attention_round_bytes=heads*rows*dim*4,
        logical_intermediate_head_major_output_bytes=heads*rows*dim*4)


def source(kind):
    from .deepseek_v41_quant import FUSED_HEADER,FUSED_SOURCE
    from .deepseek_v41_fused import HEADER
    if kind=='wo_a':
        expression=r'''
const uint coordinate=i%DIM,pos=(i/DIM)%LENGTH,head=i/(DIM*LENGTH);
float rotated=bf16_value(value[i]);
if(coordinate>=DIM-ROPE) {
 uint j=coordinate-(DIM-ROPE),pair=j/2,base=i-(j%2);
 float a=bf16_value(value[base]),b=bf16_value(value[base+1]);
 float c=cosines[pos*(ROPE/2)+pair],s=sines[pos*(ROPE/2)+pair];
 rotated=(j%2)==0 ? a*c-b*s : a*s+b*c;
}
const float v=bf16_value(rotated);
'''
        body=FUSED_SOURCE.replace('const float v=value[i];',expression)
        body=body.replace('out[i]=quantized*scale;',r'''
const uint destination=((pos*GROUPS+head/HEADS_PER_GROUP)*HEADS_PER_GROUP+head%HEADS_PER_GROUP)*DIM+coordinate;
out[destination]=bf16_value(quantized*scale);
''')
        names=['value','inverse','cosines','sines']
    elif kind=='wo_b':
        body=FUSED_SOURCE.replace('const float v=value[i];','const float v=bf16_value(value[i]);')
        names=['value','inverse']
    else:raise ValueError('Unknown Source output-chain stage')
    return dict(header=FUSED_HEADER+HEADER,source=body,input_names=names,output_names=['out'])


@lru_cache(maxsize=2)
def _kernel(kind):
    import mlx.core as mx
    spec=source(kind)
    return mx.fast.metal_kernel(name='mlx2_source_output_chain_'+kind+'_v1',**spec,ensure_row_contiguous=True)


def _gpu():
    import mlx.core as mx
    if mx.default_device()!=mx.gpu or not mx.metal.is_available():
        raise ValueError('Source output chain requires caller-owned Metal GPU')


def prepare_wo_a(value,rope,offset=0):
    """Raw attention F32[1,64,T,512] → groupedQAT F32[1,T,8,4096]."""
    import mlx.core as mx
    if value.ndim!=4 or value.dtype!=mx.float32:raise ValueError('Source output-chain attention requires F32/H64')
    p=layout(value.shape[2])
    if value.shape!=p['input_shape']or rope.dims!=64 or type(offset)is not int or offset<0:
        raise ValueError('Source output-chain attention/rotary geometry')
    _gpu()
    angles=-(mx.arange(offset,offset+value.shape[2]).astype(mx.float32)[:,None]*rope.inv[None])
    return _kernel('wo_a')(inputs=[value,mx.array([1/448],mx.float32),mx.cos(angles),mx.sin(angles)],
        template=[('ELEMENTS',value.size),('BLOCK',32),('KIND',0),('DIM',512),('LENGTH',value.shape[2]),
                  ('ROPE',64),('GROUPS',8),('HEADS_PER_GROUP',8)],
        grid=(value.size,1,1),threadgroup=(128,1,1),output_shapes=[p['grouped_shape']],output_dtypes=[mx.float32])[0]


def prepare_wo_b(value):
    """Raw grouped WoA F32[1,T,8,1024] → denseQAT F32[1,T,8192]."""
    import mlx.core as mx
    if value.ndim!=4 or value.dtype!=mx.float32:raise ValueError('Source output-chain WoA requires F32/G8')
    p=layout(value.shape[1])
    if value.shape!=p['wo_a_shape']:raise ValueError('Source output-chain grouped WoA geometry')
    _gpu()
    return _kernel('wo_b')(inputs=[value,mx.array([1/448],mx.float32)],
        template=[('ELEMENTS',value.size),('BLOCK',32),('KIND',0)],
        grid=(value.size,1,1),threadgroup=(128,1,1),output_shapes=[p['prepared_wo_b_shape']],output_dtypes=[mx.float32])[0]


def compilation_source(kind):
    s=source(kind)
    constants='constant uint ELEMENTS=3244032,BLOCK=32,KIND=0,DIM=512,LENGTH=99,ROPE=64,GROUPS=8,HEADS_PER_GROUP=8;\n'
    args='const device float* value [[buffer(0)]],const device float* inverse [[buffer(1)]],'
    if kind=='wo_a':args+='const device float* cosines [[buffer(2)]],const device float* sines [[buffer(3)]],device float* out [[buffer(4)]],'
    else:args+='device float* out [[buffer(2)]],'
    return '#include <metal_stdlib>\nusing namespace metal;\n'+s['header']+constants+'kernel void output_chain('+args+'uint3 thread_position_in_grid [[thread_position_in_grid]]){\n'+s['source']+'\n}\n'
