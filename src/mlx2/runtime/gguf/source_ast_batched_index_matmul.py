# SPDX-License-Identifier: MIT
"""Explicit, unqualified M32/K128 index-score batch candidates.

This wrapper preserves the installed regular SIMD GEMM arithmetic body. Its
dispatch is a candidate for actual GPU comparison, not inferred parity from an
older host source. The existing admitted M64 attention helper is unchanged.
"""
from functools import lru_cache
import hashlib
import os
from ...process_env import TF32_ENV
from pathlib import Path

PROFILE = 'source_ast_index_m32_regular_candidate_v1'
IMPLEMENTATIONS = ('pinned_regular_s64_v1', 'native_batch_v1')


def plan(batches, n, *, implementation='pinned_regular_s64_v1'):
    if (type(batches) is not int or not 1 <= batches <= 16
            or type(n) is not int or not 129 <= n <= 0x7fffffff // 32
            or implementation not in IMPLEMENTATIONS):
        raise ValueError('Source AST index matrix candidate geometry unsupported')
    return dict(profile=PROFILE, implementation=implementation, batches=batches,
        M=32, N=n, K=128, transpose_b=True, BM=64, BN=64, BK=16,
        WM=2, WN=2, architecture='applegpu_g17s', align_M=False,
        align_N=n % 64 == 0, align_K=True, ldA=128, ldB=128,
        output_shape=(batches, 32, n), grid=(((n+63)//64)*32, 2, batches*2),
        threadgroup=(32, 2, 2), qualified=False, exact_gpu_claim=False,
        dispatch_scope='Regular-only N>K; installed B1 parity requires actual role controls')


def kernel_spec(batches, n):
    from .source_ast_batched_splitk import _installed, _header
    d = plan(batches, n)
    stock = (_installed()/'steel/gemm/kernels/steel_gemm_fused.h').read_text()
    begin = stock.index('  using gemm_kernel = GEMMKernel<')
    end = stock.index('\n}\n', begin)
    body = stock[begin:end]
    prefix = f'''
using namespace mlx::steel;
using T=float;using AccumType=float;
constexpr int BM=64,BN=64,BK=16,WM=2,WN=2;
constexpr bool transpose_a=false,transpose_b=true;
constexpr bool has_batch=false,use_out_source=false,do_axpby=false;
constexpr bool align_M=false,align_N={'true' if d['align_N'] else 'false'},align_K=true;
const uint simd_lane_id=thread_index_in_simdgroup;
const uint simd_group_id=simdgroup_index_in_threadgroup;
const uint3 physical_tid=threadgroup_position_in_grid;
const uint3 tid=uint3(physical_tid.x,physical_tid.y,0u);
const device float* A=a+size_t(physical_tid.z)*{32*128};
const device float* B=b+size_t(physical_tid.z)*{n*128};
const device float* C=a;
device float* D=result_buffer+size_t(physical_tid.z)*{32*n};
const GEMMParams params_value={{32,{n},128,128,128,{n},{(n+63)//64},1,0,0,0,0,8,1}};
thread const GEMMParams* params=&params_value;
const GEMMAddMMParams unused_addmm={{{n},1,0,1.0f,0.0f}};
thread const GEMMAddMMParams* addmm_params=&unused_addmm;
const constant int* batch_shape=nullptr;
const constant int64_t* batch_strides=nullptr;
'''
    return dict(descriptor=d, header=_header(), source=prefix+body,
        input_names=['a', 'b'], output_names=['result_buffer'])


@lru_cache(maxsize=128)
def _kernel(batches, n):
    import mlx.core as mx
    spec = kernel_spec(batches, n)
    spec.pop('descriptor')
    return mx.fast.metal_kernel(name=f'mlx2_ast_index_m32_s64_g{batches}_n{n}',
        ensure_row_contiguous=True, **spec)


def private_matmul(left, right, *, check_live, implementation='pinned_regular_s64_v1'):
    """Schedule G independent index products over equal chronological counts.

    left[G,32,128], right[G,128,N]. Coefficients are compact NT index rows,
    never a padding of all speculative physical columns. No eval or CPU IDs.
    """
    import mlx.core as mx
    if not callable(check_live) or check_live() is not True:
        raise ValueError('Source AST index matrix requires live caller ownership')
    if mx.default_device() != mx.gpu:
        raise ValueError('Source AST index matrix requires parent-owned GPU')
    if os.environ.get(TF32_ENV) != '0':
        raise ValueError('Source AST index matrix requires explicit TF32 disabled')
    if (left.ndim != 3 or right.ndim != 3 or left.dtype != mx.float32
            or right.dtype != mx.float32 or left.shape[1:] != (32, 128)
            or right.shape[0] != left.shape[0] or right.shape[1] != 128):
        raise ValueError('Source AST index matrix input geometry/dtype mismatch')
    batches, n = left.shape[0], right.shape[-1]
    d = plan(batches, n, implementation=implementation)
    if mx.metal.device_info().get('architecture') != d['architecture']:
        raise ValueError('Source AST index matrix requires pinned applegpu_g17s')
    a = mx.contiguous(left)
    compact = mx.contiguous(mx.swapaxes(right, -1, -2))
    if implementation == 'native_batch_v1':
        storage = mx.concatenate((a.reshape(batches, 32*128),
                                  mx.zeros((batches, 1), mx.float32)), axis=1)
        a = mx.as_strided(storage, (batches, 32, 128), (32*128+1, 128, 1))
        return mx.matmul(a, mx.swapaxes(compact, -1, -2))
    return _kernel(batches, n)(inputs=[a, compact], grid=d['grid'],
        threadgroup=d['threadgroup'], output_shapes=[d['output_shape']],
        output_dtypes=[mx.float32])[0]


def implementation_identity():
    from .source_ast_batched_splitk import implementation_identity as installed_closure
    return hashlib.sha256(Path(__file__).read_bytes()+installed_closure().encode()).hexdigest()
