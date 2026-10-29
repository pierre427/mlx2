# SPDX-License-Identifier: MIT
"""Source BF16/SwiGLU/score/FP8 epilogue without intermediate arrays.

See provenance/deepseek-v41-source-middle-fusion.json. Only routed math is
changed. The sigmoid spelling is checked against the installed served op.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def _exp_gate():
    import mlx.core as mx
    import mlx.nn as nn
    from ..models.served_exp import ServedExpGate
    return ServedExpGate(
        'v41_source_middle', served_name='compiled nn.silu(float32)', served=nn.silu,
        body='float z=1.0f/(1.0f+metal::exp(metal::abs(x))); y=x*(x<0.0f ? z : 1.0f-z);',
    )


def exp_spelling():
    import mlx.core as mx
    if mx.default_device()!=mx.gpu or not mx.metal.is_available():
        raise ValueError('Source fused middle requires an owned Metal execution window')
    spelling=_exp_gate().spelling(mx.float32)
    if spelling is None:
        raise ValueError('Source fused middle cannot reproduce served F32 sigmoid')
    return spelling


def plan(gate, up, scores, slots):
    import mlx.core as mx
    if (gate.ndim!=2 or up.shape!=gate.shape or gate.dtype!=mx.float32 or up.dtype!=mx.float32
            or not all(gate.shape) or gate.shape[1]%32
            or scores.ndim!=1 or scores.dtype!=mx.float32 or not scores.size
            or slots.ndim!=1 or slots.dtype!=mx.uint32 or slots.size!=gate.shape[0]):
        raise ValueError('Source fused middle geometry/dtype mismatch')
    return gate.shape


@lru_cache(maxsize=2)
def _kernel(spelling):
    import mlx.core as mx
    from ..models.deepseek_v41_quant import FUSED_HEADER
    if spelling not in ('metal::exp','metal::precise::exp'):
        raise ValueError('Source fused middle unknown sigmoid spelling')
    source=r'''
const uint i=thread_position_in_grid.x;
if(i>=ELEMENTS) return;
const float g=min(float(bfloat(gate[i])),10.0f);
const float u=clamp(float(bfloat(up[i])),-10.0f,10.0f);
const float z=1.0f/(1.0f+EXP(abs(g)));
const float sigmoid=g<0.0f ? z : 1.0f-z;
const float silu=g*sigmoid;
const float activation=silu*u;
const float weighted=activation*scores[slots[i/WIDTH]];
const float v=float(bfloat(weighted));
float maximum=abs(v);
for(uint step=16;step>0;step/=2) maximum=max(maximum,simd_shuffle_xor(maximum,step));
const uint bits=as_type<uint>(max(maximum,1e-4f)*inverse[0]);
const uint exponent=((bits>>23)&255u)+uint((bits&0x7fffffu)!=0u);
const float scale=as_type<float>(exponent<<23);
const float normalized=v/scale;
out[i]=v41_e4m3_round(clamp(normalized,-448.0f,448.0f))*scale;
'''.replace('EXP(',spelling+'(')
    return mx.fast.metal_kernel(
        name='mlx2_source_middle_fused_'+('precise' if 'precise' in spelling else 'fast'),
        input_names=['gate','up','scores','slots','inverse'],output_names=['out'],
        header=FUSED_HEADER,source=source,ensure_row_contiguous=True,
    )


def middle(gate, up, scores, slots):
    """Raw F32 projections to exact-law F32 down input; slots are trusted routes."""
    import mlx.core as mx
    rows,width=plan(gate,up,scores,slots)
    return _kernel(exp_spelling())(
        inputs=[gate,up,scores,slots,mx.array([1/448],mx.float32)],
        template=[('ELEMENTS',rows*width),('WIDTH',width)],
        grid=(rows*width,1,1),threadgroup=(128,1,1),
        output_shapes=[(rows,width)],output_dtypes=[mx.float32],
    )[0]
