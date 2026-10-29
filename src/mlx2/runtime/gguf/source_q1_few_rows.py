# SPDX-License-Identifier: MIT
"""Unwired M2..7 Q1 coefficient reuse; M1 retains the reference consumer.

The intended lane arithmetic matches source_q1, but changed register/compiler
scheduling still needs raw GPU parity. This is a distinct unqualified profile.
No device work or MLX import occurs on import. Provenance: source-q1-few-rows.
"""
from functools import lru_cache
import hashlib

import numpy as np

from .source_q1 import SOURCE

PROFILE = 'source_q1_g64_few_rows_v1'
PINNED_DOT_SHA256 = '222a8e8eddfcf5e6f99570bf49b89dc0986ab191e61d83a4aad0b677380deb18'
SHAPES = {'gate': (5120, 2304), 'up': (5120, 2304), 'down': (2304, 5120)}


def plan(projection, rows, *, mapped=False, source_rows=None):
    if type(projection) is not str or projection not in SHAPES or type(rows) is not int or not 2 <= rows <= 7 or type(mapped) is not bool:
        raise ValueError('Source Q1 few-row candidate requires gate/up/down and local M2..7')
    if source_rows is None:
        source_rows = rows
    if (type(source_rows) is not int or not 1 <= source_rows <= 1024
            or (not mapped and source_rows != rows)):
        raise ValueError('Source Q1 few-row source row extent is invalid')
    k, n = SHAPES[projection]
    return dict(profile=PROFILE, projection=projection, rows=rows, mapped=mapped,
                source_rows=source_rows, input_dims=k, output_dims=n,
                output_shape=(rows,n), weight_shape=(n,k//32), parameter_shape=(n,k//64),
                grid=(((n+7)//8)*64,1,1), threadgroup=(64,1,1),
                group_size=64, coefficient_load_row_reuse=rows,
                coefficient_repack_bytes=0, qualification='unqualified',
                numerical_contract='per-row stock K512/lane16 affine accumulation; GPU raw gate pending')


def kernel_spec(rows, *, mapped=False):
    if type(rows) is not int or not 2 <= rows <= 7 or type(mapped) is not bool:
        raise ValueError('Source Q1 few-row kernel requires M2..7 and boolean layout')
    if hashlib.sha256(SOURCE.encode()).hexdigest() != PINNED_DOT_SHA256:
        raise RuntimeError('Source Q1 dot program changed; few-row consumer requires review')
    row_expr = 'rows[v]' if mapped else 'v'
    source = r'''
const uint lane=thread_index_in_simdgroup;
const uint simd_group=simdgroup_index_in_threadgroup;
const uint input_dims=x_shape[1], output_dims=weight_shape[0];
const uint groups=scales_shape[1];
const uint output_start=threadgroup_position_in_grid.x*8+simd_group*4;
float accumulators[LOCAL_ROWS][4] = {{0.0f}};
constexpr uint VALUES_PER_THREAD=16;
constexpr uint BLOCK_SIZE=VALUES_PER_THREAD*32;
for(uint block_start=lane*VALUES_PER_THREAD; block_start<input_dims; block_start+=BLOCK_SIZE) {
    const uint group=block_start/64;
    const uint packed_column=block_start>>5;
    const uint packed_shift=block_start&31;
    ushort packed_words[4];
    float scale_values[4],bias_values[4];
    // One packed-word/parameter load serves every independent input row.
    for(uint row=0;row<4;++row) {
        const uint output_row=output_start+row;
        packed_words[row]=0;scale_values[row]=0.0f;bias_values[row]=0.0f;
        if(output_row<output_dims) {
            packed_words[row]=ushort(weight[output_row*weight_shape[1]+packed_column]>>packed_shift);
            const uint parameter_index=output_row*groups+group;
            scale_values[row]=float(scales[parameter_index]);
            bias_values[row]=float(biases[parameter_index]);
        }
    }
    for(uint v=0;v<LOCAL_ROWS;++v) {
        const uint source_row=ROW_EXPRESSION;
        if(source_row>=x_shape[0]) continue;
        float x_thread[VALUES_PER_THREAD];
        float total_sum=0.0f;
        #pragma clang loop unroll(full)
        for(uint i=0;i<VALUES_PER_THREAD;++i) {
            float value=float(x[source_row*input_dims+block_start+i]);
            x_thread[i]=value;
            total_sum+=value;
        }
        for(uint row=0;row<4;++row) {
            const uint output_row=output_start+row;
            if(output_row<output_dims) {
                const ushort packed=packed_words[row];
                float selected_sum=0.0f;
                #pragma clang loop unroll(full)
                for(uint i=0;i<VALUES_PER_THREAD;++i) {
                    selected_sum+=(packed & (ushort(1)<<i)) ? x_thread[i] : 0.0f;
                }
                accumulators[v][row]+=selected_sum*scale_values[row];
                accumulators[v][row]+=total_sum*bias_values[row];
            }
        }
    }
}
for(uint v=0;v<LOCAL_ROWS;++v) {
    const uint source_row=ROW_EXPRESSION;
    for(uint row=0;row<4;++row) {
        accumulators[v][row]=simd_sum(accumulators[v][row]);
        const uint output_row=output_start+row;
        if(lane==0 && output_row<output_dims) {
            // The CPU-issued map is validated before upload. This guard also
            // avoids an out-of-bounds load if the descriptor is corrupted.
            out[v*output_dims+output_row]=source_row<x_shape[0] ? accumulators[v][row] : as_type<float>(0x7fc00000u);
        }
    }
}
'''.replace('LOCAL_ROWS',str(rows)).replace('ROW_EXPRESSION',row_expr)
    return dict(input_names=['x']+(['rows'] if mapped else [])+['weight','scales','biases'],
                output_names=['out'], source=source, threadgroup=(64,1,1),
                profile=PROFILE, rows=rows, mapped=mapped,
                requires_raw_gpu_parity=True, coefficient_repack_bytes=0)


@lru_cache(maxsize=12)
def _kernel(rows, mapped):
    import mlx.core as mx
    spec=kernel_spec(rows,mapped=mapped)
    return mx.fast.metal_kernel(name=f'mlx2_q1_few_rows_m{rows}_map{int(mapped)}_v1',
                               input_names=spec['input_names'], output_names=spec['output_names'],
                               source=spec['source'], ensure_row_contiguous=True)


def project(x, weight, scales, biases, *, projection, rows=None):
    """Explicit GPU probe consumer; CPU map input prevents a hidden readback.

    Caller holds expert/input/map owners through output completion. This API
    is not selected by the ordinary Source provider or a serving default.
    """
    import mlx.core as mx
    mapped=rows is not None
    if (x.ndim!=2 or x.dtype!=mx.float32 or not 1<=x.shape[0]<=1024):
        raise ValueError('Source Q1 few-row activation geometry/dtype mismatch')
    if mapped:
        if (not isinstance(rows,np.ndarray) or rows.dtype!=np.uint32 or rows.ndim!=1
                or not 2<=rows.size<=7 or np.any(rows>=x.shape[0])):
            raise ValueError('Source Q1 few-row map must be bounded CPU uint32 metadata')
        row_count=int(rows.size)
    else:
        row_count=int(x.shape[0])
    p=plan(projection,row_count,mapped=mapped,source_rows=int(x.shape[0]))
    if (x.shape[1]!=p['input_dims'] or weight.dtype!=mx.uint32 or weight.shape!=p['weight_shape']
            or scales.dtype!=mx.float16 or biases.dtype!=mx.float16
            or scales.shape!=p['parameter_shape'] or biases.shape!=scales.shape):
        raise ValueError('Source Q1 few-row packed group64 geometry/dtype mismatch')
    if mx.default_device()!=mx.gpu or not mx.metal.is_available():
        raise ValueError('Source Q1 few-row probe requires caller-owned GPU approval')
    inputs=[x]
    if mapped:inputs.append(mx.array(np.array(rows,copy=True),mx.uint32))
    inputs.extend((weight,scales,biases))
    return _kernel(row_count,mapped)(inputs=inputs, grid=p['grid'],threadgroup=p['threadgroup'],
        output_shapes=[p['output_shape']],output_dtypes=[mx.float32])[0]


def lane_oracle(x, weight, scales, biases, *, rows=None, reuse=False):
    """Small CPU synthetic oracle returns pre-simd lane accumulators.

    Stock-style and reuse-style schedules are independent loops. It does not
    claim to emulate Metal simd_sum/code generation. Synthetic N may be small
    so narrow output tails can be checked without loading checkpoint weights.
    """
    if (not isinstance(x,np.ndarray) or x.dtype!=np.float32 or x.ndim!=2
            or x.shape[1]%64 or not x.shape[1] or not isinstance(weight,np.ndarray)
            or weight.dtype!=np.uint32 or weight.ndim!=2 or weight.shape[1]*32!=x.shape[1]
            or not isinstance(scales,np.ndarray) or not isinstance(biases,np.ndarray)
            or scales.dtype!=np.float16 or biases.dtype!=np.float16
            or scales.shape!=biases.shape or scales.shape!=(weight.shape[0],x.shape[1]//64)):
        raise ValueError('Source Q1 few-row CPU oracle shape/dtype mismatch')
    selected=np.arange(x.shape[0],dtype=np.uint32) if rows is None else rows
    if (not isinstance(selected,np.ndarray) or selected.dtype!=np.uint32 or selected.ndim!=1
            or not 1<=selected.size<=7 or np.any(selected>=x.shape[0]) or type(reuse)is not bool):
        raise ValueError('Source Q1 few-row CPU oracle invalid rows')
    result=np.zeros((selected.size,weight.shape[0],32),np.float32)
    f=np.float32
    if not reuse:
        # Reference follows one invocation's input row through every output.
        for v,source_row in enumerate(selected):
            for out in range(weight.shape[0]):
                for lane in range(32):
                    acc=f(0)
                    for start in range(lane*16,x.shape[1],512):
                        total,chosen=f(0),f(0)
                        packed=int(weight[out,start//32])>>(start%32)
                        for i in range(16):
                            value=x[int(source_row),start+i]
                            total=f(total+value)
                            chosen=f(chosen+(value if packed&(1<<i) else f(0)))
                        acc=f(acc+f(chosen*f(scales[out,start//64])))
                        acc=f(acc+f(total*f(biases[out,start//64])))
                    result[v,out,lane]=acc
    else:
        # Candidate loads codes/parameters once, then visits its input rows.
        for lane in range(32):
            for start in range(lane*16,x.shape[1],512):
                for out in range(weight.shape[0]):
                    packed=int(weight[out,start//32])>>(start%32)
                    scale,bias=f(scales[out,start//64]),f(biases[out,start//64])
                    for v,source_row in enumerate(selected):
                        values=x[int(source_row),start:start+16]
                        total,chosen=f(0),f(0)
                        for i,value in enumerate(values):
                            total=f(total+value)
                            chosen=f(chosen+(value if packed&(1<<i) else f(0)))
                        result[v,out,lane]=f(result[v,out,lane]+f(chosen*scale))
                        result[v,out,lane]=f(result[v,out,lane]+f(total*bias))
    return result
