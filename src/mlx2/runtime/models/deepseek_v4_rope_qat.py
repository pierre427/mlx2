"""Opt-in FP32 RoPE + E4M3FN QAT epilogue; no automatic selection.

One SIMD group per nonrotary group64 and one per rotary tail64. Existing
Rotary.inv supplies all YaRN frequency choices. See provenance entry.
"""
from functools import lru_cache
import math

SOURCE = r'''
const uint row=threadgroup_position_in_grid.x;
const uint group=thread_index_in_threadgroup/32;
const uint lane=thread_index_in_threadgroup%32;
if(row>=ROWS) return;
if(group<7) {
  const uint first=row*512+group*64+lane;
  const float a=value[first], b=value[first+32];
  const float maximum=max(simd_max(max(abs(a),abs(b))),1.0e-4f);
  const float scale=exp2(ceil(log2(maximum/448.0f)));
  for(uint i=0;i<2;i++) {
    const uint index=first+i*32;
    const float v=clamp(value[index]/scale,-448.0f,448.0f);
    const float exponent=floor(log2(max(abs(v),0.015625f)));
    const float step=exp2(max(exponent-3.0f,-9.0f));
    out[index]=(rint(v/step)*step)*scale;
  }
} else {
  const uint index=row*512+448+lane*2;
  const float angle=positions[row%LENGTH]*inv_freq[lane];
  const float c=precise::cos(angle), s=precise::sin(angle);
  const float a=value[index], b=value[index+1];
  // Match separate graph multiplies/adds; prohibit implicit FMA contraction.
  volatile float ac=a*c, bs=b*s, as=a*s, bc=b*c;
  out[index]=ac-bs;
  out[index+1]=as+bc;
}
'''
_SCHEDULED = 0


def route_status():
    return {'policy':'fused_rope_fp8_fp32', 'scheduled_calls':_SCHEDULED,
            'observation':'scheduled only; does not establish GPU completion'}


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx
    return mx.fast.metal_kernel(name='mlx2_deepseek_v4_rope_fp8_v1',
        input_names=['value','inv_freq','positions'], output_names=['out'],
        source=SOURCE, ensure_row_contiguous=True)


def rope_fp8(value, inv_freq, positions):
    """Mandatory QAT prefix448 + traditional rotary tail64; FP32 only.

    Inputs [..., length, 512], [32], [length]. Explicit positions may be pool
    starts; no cached absolute offset or hidden frequency policy is introduced.
    """
    import mlx.core as mx
    global _SCHEDULED
    if (value.ndim<2 or value.shape[-1]!=512 or inv_freq.shape!=(32,)
            or positions.shape!=(value.shape[-2],)
            or any(a.dtype!=mx.float32 for a in (value, inv_freq, positions))):
        raise ValueError('fused RoPE/QAT requires FP32 [...,length,512], [32], [length]')
    rows=math.prod(value.shape[:-1])
    if not rows: raise ValueError('fused RoPE/QAT requires nonempty rows')
    if mx.default_device()!=mx.gpu or not mx.metal.is_available():
        raise ValueError('fused RoPE/QAT requires an owned Metal GPU execution window')
    result=_kernel()(inputs=[value,inv_freq,positions],
        template=[('ROWS',rows),('LENGTH',value.shape[-2])],
        grid=(rows*256,1,1),threadgroup=(256,1,1),
        output_shapes=[value.shape],output_dtypes=[mx.float32])[0]
    _SCHEDULED+=1
    return result


def reference_rope_fp8(value, inv_freq, positions):
    """Independent exhaustive-code nearest-even QAT + NumPy FP32 RoPE oracle."""
    import numpy as np
    value=np.asarray(value,dtype=np.float32)
    inv=np.asarray(inv_freq,dtype=np.float32); positions=np.asarray(positions,dtype=np.float32)
    if value.ndim<2 or value.shape[-1]!=512 or inv.shape!=(32,) or positions.shape!=(value.shape[-2],):
        raise ValueError('invalid reference RoPE/QAT geometry')
    prefix=value[...,:448].reshape(*value.shape[:-1],7,64)
    maximum=np.maximum(np.max(np.abs(prefix),axis=-1,keepdims=True),np.float32(1e-4))
    scale=np.exp2(np.ceil(np.log2(maximum/np.float32(448))))
    v=np.clip(prefix/scale,-448,448)
    codes=np.arange(127); exponent=codes>>3; fraction=codes&7
    levels=np.where(exponent==0,fraction/512,(1+fraction/8)*np.exp2(exponent-7)).astype(np.float32)
    magnitude=np.abs(v)
    lower=np.clip(np.searchsorted(levels,magnitude,side='right')-1,0,126)
    upper=np.minimum(lower+1,126)
    dl=magnitude-levels[lower]; du=levels[upper]-magnitude
    chosen=np.where((du<dl)|((du==dl)&((upper%2)==0)),upper,lower)
    rounded=np.copysign(levels[chosen],v)*scale
    angle=positions[:,None]*inv[None]
    c=np.cos(angle); s=np.sin(angle)
    a=value[...,448::2]; b=value[...,449::2]
    tail=np.stack((a*c-b*s,a*s+b*c),axis=-1).reshape(*value.shape[:-1],64)
    return np.concatenate((rounded.reshape(*value.shape[:-1],448),tail),axis=-1)


def compilation_source():
    return '''#include <metal_stdlib>
using namespace metal;
kernel void check_rope_fp8(device const float *value [[buffer(0)]],
device const float *inv_freq [[buffer(1)]], device const float *positions [[buffer(2)]],
device float *out [[buffer(3)]],
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]) {
constexpr uint ROWS=9,LENGTH=9;
''' + SOURCE + '\n}\n'
