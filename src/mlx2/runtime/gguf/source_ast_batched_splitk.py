# SPDX-License-Identifier: MIT
"""Default-off ordinary F32 split-K/regular batches; actual GPU gate required.

Original project wrapper. Installed MLX MIT helper notices are preserved in
expanded shader headers. Provenance: deepseek-v41-source-ast-batched-splitk.json.
"""
from functools import lru_cache
import hashlib
import os
from ...process_env import TF32_ENV
from pathlib import Path
import re

PROFILE='source_ast_batched_stock_float_candidate_v2'
HASHES={'steel/gemm/kernels/steel_gemm_fused.h':'ed49c8b9c73957f0cbc9dea30f484cf84fb821aece68233d6d1967856200d5c7','steel/gemm/gemm.h': 'c84af31e2c57154f2a8a24fa7f9fe2449765cce9a66f3f035846ed3c03b6a8b0', 'steel/gemm/loader.h': 'c1b82153670b1e18371a6e88fdbcf440af692957775c2f837700315ab8a5112e', 'steel/defines.h': 'b03cea6a7d5cfe814e6838d8536e2aacdcafd2744cb408e954fda336ce21759a', 'steel/gemm/mma.h': '9b37709de40c3acaa5411875e52c8d26e6de093261741f671fbdab9c39a852b7', 'steel/gemm/transforms.h': '73487e29beab5af9478bc931c49b5e8d66937c2fa2651499867f47ac5529afc2', 'steel/utils.h': 'd4c36298145d1c5617a94f07dbfabf6ac932afe2fd7ea0c9291f1fb385c5404a', 'steel/utils/integral_constant.h': '886eccaac4f8f00eab5a10551fdb5315df7ab7462aba13d69bb95765c1a4735e', 'steel/utils/type_traits.h': '4766c5a3809d3cf7740e63acc5703ef76b6b4e85d8baf16602c9d517a2fd426c', 'steel/gemm/params.h': '6711e72e32310eafb088895d5a6d20fcfca814158f5b38eb72b6e69d9d773e43', 'steel/gemm/kernels/steel_gemm_splitk.h': '2ada2a7ea548f6c4b0e7962b4a2f163e8a212620ecc46e078a06647df12b0635'}


def supports_splitk(m,n,k):
    """Bounded intersection of all local-device SIMD split-K dispatch gates."""
    return (all(type(v)is int for v in(m,n,k)) and m==64 and 2<=n<=512
            and 128<=k<=512 and k>=max(m,n))


def supports(m,n,k):
    return (supports_splitk(m,n,k) or (all(type(v)is int for v in(m,n,k))
            and m==64 and n==512 and 1<=k<=512))


def plan(batches,m,n,k,*,transpose_b):
    if type(batches)is not int or not 1<=batches<=16 or type(transpose_b)is not bool or not supports(m,n,k):
        raise ValueError('Source AST stock split-K candidate geometry unsupported')
    if not supports_splitk(m,n,k):
        return dict(profile=PROFILE,mode='regular',architecture='applegpu_g17s',batches=batches,M=m,N=n,K=k,
            transpose_b=transpose_b,BM=64,BN=64,BK=16,WM=2,WN=2,ldA=k,ldB=k if transpose_b else n,
            output_shape=(batches,m,n),grid=(((n+63)//64)*32,((m+63)//64)*2,batches*2),
            threadgroup=(32,2,2),has_batch=False,use_out_source=False,align_M=m%64==0,
            align_N=n%64==0,align_K=k%16==0,qualified=False,exact_gpu_claim=False)
    tm=(m+31)//32;tn=(n+31)//32;tk=k//16
    ratio=tk//(tm*tn);power=1<<(max(1,ratio)-1).bit_length()
    partitions=min(max(2,power),32)
    iterations=(k//16)//partitions;partition_size=iterations*16
    bm=16 if m<40 else 32;bn=16 if n<40 else 32
    return dict(profile=PROFILE,mode='splitk',batches=batches,M=m,N=n,K=k,transpose_b=transpose_b,
        BM=bm,BN=bn,BK=16,WM=2,WN=2,partitions=partitions,
        partition_size=partition_size,k_iterations=iterations,ldA=k,ldB=k if transpose_b else n,
        split_shape=(batches,partitions,m,n),output_shape=(batches,m,n),
        grid=(((n+bn-1)//bn)*32,((m+bm-1)//bm)*2,batches*partitions*2),
        threadgroup=(32,2,2),qualified=False,exact_gpu_claim=False)


@lru_cache(maxsize=1)
def _installed():
    from .source_dense import _installed as pinned
    _,base=pinned()
    for name,expected in HASHES.items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Source AST split-K installed helper changed; re-audit required:'+name)
    return base


@lru_cache(maxsize=1)
def _header():
    base=_installed();seen=set()
    def expand(name):
        if name in seen:return ''
        seen.add(name);result=[]
        for line in(base/name).read_text().splitlines():
            match=re.fullmatch(r'#include "mlx/backend/metal/kernels/([^"]+)"',line)
            if match:result.append(expand(match.group(1)))
            elif line!='#pragma once':result.append(line)
        return '\n'.join(result)
    # metal_kernel appends its entry declaration immediately after this text.
    # The installed closure ends in a // namespace comment: terminate it.
    return expand('steel/gemm/gemm.h')+'\n'


def kernel_specs(batches,m,n,k,*,transpose_b):
    """CPU source builder; original loop and MMA closure remain installed/pinned."""
    d=plan(batches,m,n,k,transpose_b=transpose_b);base=_installed()
    if d['mode']=='regular':return _regular_specs(d)
    stock=(base/'steel/gemm/kernels/steel_gemm_splitk.h').read_text()
    start=stock.index('  using gemm_kernel = GEMMKernel<')
    stop=stock.index('\n}\n',start)
    body=stock[start:stop]
    old='const int tid_z = tid.z;'
    if body.count(old)!=1:raise RuntimeError('Pinned split-K partition index changed')
    body=body.replace(old,f'const int tid_z = tid.z % {d["partitions"]};')
    prefix=f"""
using namespace mlx::steel;
using T=float;using U=float;
constexpr int BM={d['BM']},BN={d['BN']},BK=16,WM=2,WN=2;
constexpr bool transpose_a=false,transpose_b={'true'if transpose_b else'false'};
constexpr bool MN_aligned={'true'if m%d['BM']==0 and n%d['BN']==0 else'false'},K_aligned={'true'if k%16==0 else'false'};
const uint simd_lane_id=thread_index_in_simdgroup;
const uint simd_group_id=simdgroup_index_in_threadgroup;
const uint3 tid=threadgroup_position_in_grid;
const uint batch=tid.z/{d['partitions']};
const device float* A=a+size_t(batch)*{m*k};
const device float* B=b+size_t(batch)*{n*k};
device float* C=split+size_t(batch)*{d['partitions']*m*n};
const GEMMSpiltKParams params_value={{{m},{n},{k},{k},{d['ldB']},{n},{(n+d['BN']-1)//d['BN']},{(m+d['BM']-1)//d['BM']},{d['partitions']},{m*n},{d['partition_size']},0,{d['k_iterations']}}};
thread const GEMMSpiltKParams* params=&params_value;
"""
    # Preserve accumulation statements byte-for-byte; batch addresses and gid
    # ownership change, while each scalar's ascending partition chain does not.
    begin=stock.index('  // Ajust D and C',stock.index('void gemm_splitk_accum('))
    end=stock.index('\n}\n',begin)
    accum_body=stock[begin:end]
    accum_prefix=f"""
using namespace mlx::steel;
using AccT=float;using OutT=float;using Epilogue=TransformNone<float,float>;
const uint3 index=thread_position_in_grid;
if(index.x>={n}u||index.y>={m}u||index.z>={batches}u)return;
const uint2 gid=index.xy;
const int k_partitions={d['partitions']},partition_stride={m*n},ldd={n};
const device float* C_split=split+size_t(index.z)*{d['partitions']*m*n};
device float* D=result_buffer+size_t(index.z)*{m*n};
"""
    return dict(descriptor=d,split=dict(header=_header(),source=prefix+body,input_names=['a','b'],output_names=['split']),
        accum=dict(header=_header(),source=accum_prefix+accum_body,input_names=['split'],output_names=['result_buffer']))


def _regular_specs(d):
    base=_installed();stock=(base/'steel/gemm/kernels/steel_gemm_fused.h').read_text()
    begin=stock.index('  using gemm_kernel = GEMMKernel<');end=stock.index('\n}\n',begin)
    body=stock[begin:end]
    m,n,k,batches=(d[v]for v in('M','N','K','batches'))
    # B1 body retains its zero batch index and flags. Independent owner offsets
    # are applied before that unchanged body, never inside its arithmetic.
    prefix=f"""
using namespace mlx::steel;
using T=float;using AccumType=float;
constexpr int BM=64,BN=64,BK=16,WM=2,WN=2;
constexpr bool transpose_a=false,transpose_b={'true'if d['transpose_b']else'false'};
constexpr bool has_batch=false,use_out_source=false,do_axpby=false;
constexpr bool align_M={'true'if d['align_M']else'false'},align_N={'true'if d['align_N']else'false'},align_K={'true'if d['align_K']else'false'};
const uint simd_lane_id=thread_index_in_simdgroup;
const uint simd_group_id=simdgroup_index_in_threadgroup;
const uint3 physical_tid=threadgroup_position_in_grid;
const uint3 tid=uint3(physical_tid.x,physical_tid.y,0u);
const device float* A=a+size_t(physical_tid.z)*{m*k};
const device float* B=b+size_t(physical_tid.z)*{n*k};
const device float* C=a;
device float* D=result_buffer+size_t(physical_tid.z)*{m*n};
const GEMMParams params_value={{{m},{n},{k},{k},{d['ldB']},{n},{(n+63)//64},{(m+63)//64},0,0,0,0,{k//16},1}};
thread const GEMMParams* params=&params_value;
const GEMMAddMMParams unused_addmm={{{n},1,0,1.0f,0.0f}};
thread const GEMMAddMMParams* addmm_params=&unused_addmm;
const constant int* batch_shape=nullptr;
const constant int64_t* batch_strides=nullptr;
"""
    return dict(descriptor=d,regular=dict(header=_header(),source=prefix+body,input_names=['a','b'],output_names=['result_buffer']))


@lru_cache(maxsize=128)
def _kernels(batches,m,n,k,transpose_b):
    import mlx.core as mx
    specs=kernel_specs(batches,m,n,k,transpose_b=transpose_b)
    key=hashlib.sha256(repr((batches,m,n,k,transpose_b)).encode()).hexdigest()[:16]
    stages=('split','accum')if specs['descriptor']['mode']=='splitk'else('regular',)
    return tuple(mx.fast.metal_kernel(name='mlx2_ast_stock_float_'+stage+'_'+key,
        ensure_row_contiguous=True,**specs[stage])for stage in stages)


def private_matmul(left,right,*,transpose_b,check_live):
    """Schedule G ordinary M64 products with split-K or pinned S regular law.

    right is logically[G,K,N]; transpose_b=True retains ordinary QK nt law by
    preparing its compact[G,N,K] storage. Caller retains outputs through wait.
    Noneligible geometries are refused; caller chooses its explicit other law.
    """
    import mlx.core as mx
    if not callable(check_live)or check_live()is not True:
        raise ValueError('Source AST split-K requires caller live ownership')
    if mx.default_device()!=mx.gpu:
        raise ValueError('Source AST split-K requires parent-owned GPU')
    if os.environ.get(TF32_ENV)!='0':
        raise ValueError('Source AST split-K requires explicit TF32 disabled')
    if (left.ndim!=3 or right.ndim!=3 or left.dtype!=mx.float32 or right.dtype!=mx.float32
            or left.shape[0]!=right.shape[0]or left.shape[-1]!=right.shape[-2]):
        raise ValueError('Source AST split-K input geometry/dtype mismatch')
    batches,m,k=left.shape;n=right.shape[-1]
    d=plan(batches,m,n,k,transpose_b=transpose_b)
    a=mx.contiguous(left);b=mx.contiguous(mx.swapaxes(right,-1,-2)if transpose_b else right)
    if d['mode']=='regular':
        if mx.metal.device_info().get('architecture')!='applegpu_g17s':
            raise ValueError('Source AST ordinary regularGEMM candidate requires pinned applegpu_g17s')
        first,=_kernels(batches,m,n,k,transpose_b)
        return first(inputs=[a,b],grid=d['grid'],threadgroup=d['threadgroup'],
            output_shapes=[d['output_shape']],output_dtypes=[mx.float32])[0]
    first,second=_kernels(batches,m,n,k,transpose_b)
    partial=first(inputs=[a,b],grid=d['grid'],threadgroup=d['threadgroup'],
        output_shapes=[d['split_shape']],output_dtypes=[mx.float32])[0]
    return second(inputs=[partial],grid=(n,m,batches),threadgroup=(min(16,n),min(16,m),1),
        output_shapes=[d['output_shape']],output_dtypes=[mx.float32])[0]


def implementation_identity():
    _installed()
    from .source_dense import LIBRARY_SHA256
    return hashlib.sha256(Path(__file__).read_bytes()+LIBRARY_SHA256.encode()+repr(sorted(HASHES.items())).encode()).hexdigest()
