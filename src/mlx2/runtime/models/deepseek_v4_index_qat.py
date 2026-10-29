"""Explicit fused E2M1 group32 QAT after the unchanged MLX Hadamard transform."""
from functools import lru_cache
import math

SOURCE = r'''
const uint i=thread_position_in_grid.x;
if(i>=ELEMENTS) return;
const float maximum=max(simd_max(abs(value[i])),7.052966104933725e-38f);
const float scale=precise::exp2(ceil(precise::log2(maximum/6.0f)));
const float v=value[i]/scale;
const float a=abs(v);
// Adjacent E2M1 midpoint ties select the even code, matching the reference
// eight-distance argmin. All boundaries are exactly representable in FP32.
float q;
if(a<=0.25f) q=0.0f;
else if(a<0.75f) q=0.5f;
else if(a<=1.25f) q=1.0f;
else if(a<1.75f) q=1.5f;
else if(a<=2.5f) q=2.0f;
else if(a<3.5f) q=3.0f;
else if(a<=5.0f) q=4.0f;
else q=6.0f;
// sign(+-0) is zero in the reference; nonzero negative inputs quantizing to
// zero retain a negative zero. Multiplication by a power of two is separate.
const float signed_q=(v==0.0f)?0.0f:copysign(q,v);
out[i]=signed_q*scale;
'''
_SCHEDULED = 0


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(name="mlx2_deepseek_v4_index_fp4_epilogue_v1",
        input_names=["value"],output_names=["out"],source=SOURCE,ensure_row_contiguous=True)


def quantize_rotated(value):
    """FP32 [...,128], one SIMD group per contiguous 32-coordinate scale group."""
    import mlx.core as mx
    global _SCHEDULED
    if value.ndim<1 or value.shape[-1]!=128 or value.dtype!=mx.float32 or not value.size:
        raise ValueError("fused index QAT requires nonempty FP32 [...,128]")
    if mx.default_device()!=mx.gpu or not mx.metal.is_available():
        raise ValueError("fused index QAT requires an owned Metal GPU execution window")
    count=math.prod(value.shape)
    result=_kernel()(inputs=[value],template=[("ELEMENTS",count)],grid=(count,1,1),threadgroup=(128,1,1),
                     output_shapes=[value.shape],output_dtypes=[mx.float32])[0]
    _SCHEDULED+=1
    return result


def index_fp4(value):
    import mlx.core as mx
    if value.ndim<1 or value.dtype!=mx.float32 or value.shape[-1]!=128:
        raise ValueError("fused index QAT requires FP32 [...,128]")
    return quantize_rotated(mx.hadamard_transform(value,scale=128**-0.5))


def route_status():
    return dict(policy="fused_index_fp4_fp32",scheduled_calls=_SCHEDULED,
                observation="scheduled only; not evidence of completed GPU work")


def compilation_source():
    return '''#include <metal_stdlib>
using namespace metal;
kernel void check_index_fp4(device const float *value [[buffer(0)]],
device float *out [[buffer(1)]],uint3 thread_position_in_grid [[thread_position_in_grid]]) {
constexpr uint ELEMENTS=128;
''' + SOURCE + '\n}\n'


def reference_rotated(value):
    """Independent NumPy eight-level distance/priority oracle, not thresholds."""
    import numpy as np
    value=np.asarray(value,dtype=np.float32)
    groups=value.reshape(*value.shape[:-1],4,32)
    maximum=np.maximum(np.max(np.abs(groups),axis=-1,keepdims=True),np.float32(7.052966104933725e-38))
    scale=np.exp2(np.ceil(np.log2(maximum/np.float32(6))))
    scaled=groups/scale
    levels=np.array([0,.5,1,1.5,2,3,4,6],dtype=np.float32)
    distance=np.abs(np.abs(scaled)[...,None]-levels)
    minimum=np.min(distance,axis=-1,keepdims=True)
    codes=np.arange(8);priority=np.where(codes%2==0,codes,codes+8)
    chosen=np.argmin(np.where(distance==minimum,priority,100),axis=-1)
    return ((levels[chosen]*np.sign(scaled))*scale).reshape(value.shape)
