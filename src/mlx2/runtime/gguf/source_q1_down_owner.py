# SPDX-License-Identifier: MIT
"""Unwired Q1 DOWN candidate: unchanged dot law, fused BF16 owner store.

The original packed-Q1 body is inherited through ``source_q1.SOURCE``. Only
the last output store changes. GPU raw/owner parity remains a required gate;
CPU/source checks do not establish it. See provenance/source-down-actual.
"""
from functools import lru_cache
import hashlib

from .source_q1 import SOURCE
PINNED_SOURCE_SHA256 = '222a8e8eddfcf5e6f99570bf49b89dc0986ab191e61d83a4aad0b677380deb18'


def kernel_spec(*, return_raw=False):
    if type(return_raw) is not bool:
        raise ValueError('return_raw must be boolean')
    if hashlib.sha256(SOURCE.encode()).hexdigest() != PINNED_SOURCE_SHA256:
        raise ValueError('Source Q1 dot program changed; bounded candidate requires review')
    body = (SOURCE.replace('GROUP_SIZE', '64')
            .replace('OUTPUTS_PER_THREADGROUP', '8')
            .replace('OUTPUTS_PER_SIMDGROUP', '4')
            .replace('ROW_VALID', 'output_row < output_dims'))
    old = 'static_cast<T>(accumulators[row]);'
    if body.count(old) != 1:
        raise RuntimeError('Pinned Source Q1 final store changed')
    new = 'float(bfloat(accumulators[row]));'
    if return_raw:
        new += '\n            raw[input_row * output_dims + output_row] = accumulators[row];'
    body = body.replace(old, new)
    return dict(input_names=['x', 'weight', 'scales', 'biases'],
                output_names=['out', 'raw'] if return_raw else ['out'],
                source=body, threadgroup=(64, 1, 1),
                dot_body_modifications=0, owner_store='F32(BF16(raw F32))')


def plan(x, weight, scales, biases):
    import mlx.core as mx
    if (x.ndim != 2 or x.dtype != mx.float32 or x.shape[1] != 2304
            or type(x.shape[0]) is not int or not 1 <= x.shape[0] <= 99
            or weight.dtype != mx.uint32 or weight.shape != (5120, 72)
            or scales.dtype != mx.float16 or biases.dtype != mx.float16
            or scales.shape != (5120, 36) or biases.shape != scales.shape):
        raise ValueError('Bounded Q1 DOWN owner geometry/dtype mismatch')
    return dict(rows=x.shape[0], output_shape=(x.shape[0], 5120),
                grid=(640*64, x.shape[0], 1), threadgroup=(64, 1, 1),
                logical_input_bytes=x.shape[0]*2304*4,
                logical_output_bytes=x.shape[0]*5120*4,
                coefficient_bytes=5120*72*4 + 2*5120*36*2)


@lru_cache(maxsize=2)
def _kernel(return_raw):
    import mlx.core as mx
    spec = kernel_spec(return_raw=return_raw)
    return mx.fast.metal_kernel(
        name='mlx2_source_q1_down_owner_v1' + ('_raw' if return_raw else ''),
        input_names=spec['input_names'], output_names=spec['output_names'],
        source=spec['source'])


def project(x, weight, scales, biases, *, return_raw=False):
    """Caller retains immutable coefficient inputs through GPU completion."""
    import mlx.core as mx
    if type(return_raw) is not bool:
        raise ValueError('return_raw must be boolean')
    p = plan(x, weight, scales, biases)
    if mx.default_device() != mx.gpu:
        raise ValueError('Q1 DOWN owner requires caller-owned GPU')
    outputs = _kernel(return_raw)(
        inputs=[x, weight, scales, biases],
        grid=p['grid'], threadgroup=p['threadgroup'],
        output_shapes=[p['output_shape']]*(2 if return_raw else 1),
        output_dtypes=[mx.float32]*(2 if return_raw else 1))
    return tuple(outputs) if return_raw else outputs[0]


def compilation_source(*, return_raw=False):
    """Offline Metal entry with the same argument types and built-in IDs."""
    spec = kernel_spec(return_raw=return_raw)
    raw = ', device float* raw [[buffer(5)]]' if return_raw else ''
    return '''#include <metal_stdlib>
#include "mlx/backend/metal/kernels/bf16.h"
using namespace metal;
kernel void q1_down_owner(device const float* x [[buffer(0)]],
 device const uint* weight [[buffer(1)]], device const half* scales [[buffer(2)]],
 device const half* biases [[buffer(3)]], device float* out [[buffer(4)]]RAW,
 uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
 uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
 uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
 constexpr uint x_shape[2]={99,2304};
 constexpr uint weight_shape[2]={5120,72};
 constexpr uint scales_shape[2]={5120,36};
BODY
}
'''.replace('RAW', raw).replace('BODY', spec['source'])
