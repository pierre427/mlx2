# SPDX-License-Identifier: MIT
"""Source WoA: BF16 coefficient storage, strict legacy F32 K8 MMA candidate.

Exactness is a parent GPU-probe gate, never inferred from source geometry.
See provenance/deepseek-v41-source-dense-legacy.json. No GPU work on import.
"""
from functools import lru_cache
import hashlib


GROUPS, OUTPUT, WIDTH = 8, 1024, 4096
MMA_SHA256 = '9b37709de40c3acaa5411875e52c8d26e6de093261741f671fbdab9c39a852b7'


def plan(rows, *, tile_m=16, tile_n=64, simdgroups=4):
    """CPU-only ownership plan; one full K8 chain per output, never split K."""
    if type(rows) is not int or not 2 <= rows <= 1024:
        raise ValueError('Source legacy WoA row domain is2..1024; M1 retains GEMV')
    if (any(type(v) is not int for v in (tile_m,tile_n,simdgroups))
            or tile_m not in (8,16,32) or tile_n not in (32,64,128)
            or simdgroups not in (2,4,8) or tile_n % (8*simdgroups)
            or not 1 <= tile_n//simdgroups//8 <= 4):
        raise ValueError('Source legacy WoA unsupported fragment tile')
    threads = simdgroups*32
    return {'rows':rows,'tile_m':tile_m,'tile_n':tile_n,'simdgroups':simdgroups,
        'm_fragments':tile_m//8,'n_fragments_per_simd':tile_n//simdgroups//8,
        'k_step':8,'k_steps':WIDTH//8,'split_k':1,
        'grid':(OUTPUT//tile_n*threads,(rows+tile_m-1)//tile_m,GROUPS),
        'threadgroup':(threads,1,1),'output_shape':(1,rows,GROUPS,OUTPUT),
        'global_f32_weight_copy_bytes':0,'threadgroup_scratch_bytes':0,
        'coefficient_bytes':GROUPS*OUTPUT*WIDTH*2,'numerics':'candidate_legacy_f32_k8',
        'qualification':'unqualified'}


def _geometry(x,weight,**options):
    import mlx.core as mx
    if (x.ndim != 4 or x.shape[0] != 1 or x.shape[2:] != (GROUPS,WIDTH)
            or x.dtype != mx.float32 or weight.ndim != 2
            or weight.shape != (GROUPS*OUTPUT,WIDTH) or weight.dtype != mx.bfloat16):
        raise ValueError('Source legacy WoA input/coefficient geometry or dtype mismatch')
    return plan(x.shape[1],**options)


def _device_supported():
    from .source_dense import device_supported
    from ...process_env import tf32_enabled
    return device_supported() and not tf32_enabled()


def supports(x,weight,*,check_device=True,**options):
    try:
        _geometry(x,weight,**options)
        return not check_device or _device_supported()
    except (ValueError,TypeError,AttributeError):
        return False


@lru_cache(maxsize=1)
def _verify_installed():
    from .source_dense import _installed
    _,headers = _installed()
    if hashlib.sha256((headers/'steel/gemm/mma.h').read_bytes()).hexdigest() != MMA_SHA256:
        raise RuntimeError('Source legacy WoA installed MMA changed; re-audit required')


_HEADER = r"""// Derived installed MLX BaseMMAFrag<float,8,8> coordinate/arithmetic,
// Apple2024, MIT. License retained in source-mxfp4-installed-MLX-LICENSE.txt.
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#define SOURCE_LEGACY_UNROLL _Pragma("clang loop unroll(full)")
"""


def kernel_spec(rows,*,tile_m=16,tile_n=64,simdgroups=4):
    p = plan(rows,tile_m=tile_m,tile_n=tile_n,simdgroups=simdgroups)
    _verify_installed()
    source = f"constexpr int M={rows}, MF={p['m_fragments']}, NF={p['n_fragments_per_simd']};\n" + r"""
const uint lane=thread_index_in_simdgroup;
const uint sg=simdgroup_index_in_threadgroup;
const uint group=threadgroup_position_in_grid.z;
const uint m_base=threadgroup_position_in_grid.y*__TILE_M__;
const uint n_base=threadgroup_position_in_grid.x*__TILE_N__+sg*(8*NF);
// Pinned BaseMMAFrag<float,8,8> lane coordinates, two contiguous columns/lane.
const short qid=short(lane)/4;
const short fm=(qid&4)+((short(lane)/2)%4);
const short fn=(qid&2)*2+(short(lane)%2)*2;
simdgroup_matrix<float,8,8> acc[MF][NF];
SOURCE_LEGACY_UNROLL for(int m=0;m<MF;m++)
    SOURCE_LEGACY_UNROLL for(int n=0;n<NF;n++)
        acc[m][n]=make_filled_simdgroup_matrix<float,8,8>(0.0f);
// Each output retains exactly one sequential zero-initialized K8 MMA chain.
// Unlike native NAX, this is the strict F32 legacy simdgroup primitive.
#pragma clang loop unroll(disable)
for(int k0=0;k0<4096;k0+=8) {
    simdgroup_barrier(mem_flags::mem_none);
    simdgroup_matrix<float,8,8> a[MF];
    SOURCE_LEGACY_UNROLL for(int m=0;m<MF;m++) {
        const uint row=m_base+m*8+fm;
        thread auto& v=a[m].thread_elements();
        v[0]=row<M ? x[(row*8+group)*4096+k0+fn] : 0.0f;
        v[1]=row<M ? x[(row*8+group)*4096+k0+fn+1] : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_none);
    simdgroup_matrix<float,8,8> b[NF];
    SOURCE_LEGACY_UNROLL for(int n=0;n<NF;n++) {
        const uint col=n_base+n*8+fn;
        thread auto& v=b[n].thread_elements();
        v[0]=float(weight[(group*1024+col)*4096+k0+fm]);
        v[1]=float(weight[(group*1024+col+1)*4096+k0+fm]);
    }
    simdgroup_barrier(mem_flags::mem_none);
    SOURCE_LEGACY_UNROLL for(int m=0;m<MF;m++)
        SOURCE_LEGACY_UNROLL for(int n=0;n<NF;n++) {
            const int serp=(m%2) ? NF-1-n : n;
            simdgroup_multiply_accumulate(acc[m][serp],a[m],b[serp],acc[m][serp]);
        }
}
SOURCE_LEGACY_UNROLL for(int m=0;m<MF;m++)
    SOURCE_LEGACY_UNROLL for(int n=0;n<NF;n++) {
        const uint row=m_base+m*8+fm;
        const uint col=n_base+n*8+fn;
        thread auto& v=acc[m][n].thread_elements();
        if(row<M) {
            out[(row*8+group)*1024+col]=v[0];
            out[(row*8+group)*1024+col+1]=v[1];
        }
    }
"""
    source = source.replace('__TILE_M__',str(tile_m)).replace('__TILE_N__',str(tile_n))
    return {**p,'header':_HEADER,'source':source,'input_names':['x','weight'],'output_names':['out']}


@lru_cache(maxsize=16)
def _kernel(rows,tile_m,tile_n,simdgroups):
    import mlx.core as mx
    spec=kernel_spec(rows,tile_m=tile_m,tile_n=tile_n,simdgroups=simdgroups)
    return mx.fast.metal_kernel(name=f'mlx2_source_woa_legacy_r{rows}_m{tile_m}_n{tile_n}_s{simdgroups}',
        input_names=spec['input_names'],output_names=spec['output_names'],
        header=spec['header'],source=spec['source'],ensure_row_contiguous=True)


def project(x,weight,*,tile_m=16,tile_n=64,simdgroups=4):
    """Explicit unqualified candidate; no activation QAT or output rounding."""
    import mlx.core as mx
    p=_geometry(x,weight,tile_m=tile_m,tile_n=tile_n,simdgroups=simdgroups)
    if not _device_supported():
        raise ValueError('Source legacy WoA requires M5 Max and TF32-off caller-owned GPU')
    return _kernel(p['rows'],tile_m,tile_n,simdgroups)(inputs=[x,weight],
        grid=p['grid'],threadgroup=p['threadgroup'],output_shapes=[p['output_shape']],
        output_dtypes=[mx.float32])[0]


def compilation_source(rows,**options):
    s=kernel_spec(rows,**options)
    return ('#include "mlx/backend/metal/kernels/utils.h"\n'+s['header']
        +'\nusing namespace metal;\nkernel void source_dense_legacy_offline(\n'
        +'device const float* x [[buffer(0)]],device const bfloat* weight [[buffer(1)]],\n'
        +'device float* out [[buffer(2)]],uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],\n'
        +'uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],\n'
        +'uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {\n'+s['source']+'\n}\n')
