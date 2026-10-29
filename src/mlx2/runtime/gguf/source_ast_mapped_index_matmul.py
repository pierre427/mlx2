# SPDX-License-Identifier: MIT
"""Unselected chronological M32 index consumer over a shared key table.

Reuses the frozen mapped B loader without changing active index dispatch.
Every caller must reject nonzero GPU status before publishing model state.
"""
from functools import lru_cache
import hashlib
import os
from ...process_env import TF32_ENV
from pathlib import Path

from .source_ast_mapped_kv_matmul import MappedProduct

PROFILE = 'source_ast_mapped_index_m32_candidate_v1'


def plan(batches, n, *, table_rows):
    from .source_ast_batched_index_matmul import plan as ordinary_plan
    d = dict(ordinary_plan(batches, n, implementation='pinned_regular_s64_v1'))
    if type(table_rows) is not int or not 1 <= table_rows <= 0x7fffffff:
        raise ValueError('Source AST mapped index table row count invalid')
    d.update(profile=PROFILE, table_rows=table_rows, table_dim=128,
        map_columns=n, map_shape=(batches,n), map_bytes=batches*n*4,
        eliminated_gather_bytes=batches*n*128*4,
        materialized_query_index_keys=False, map_validation_launches=1,
        explicit_cross_query_threadgroup_reuse=False,
        publication_requires_zero_map_status=True, qualified=False, selected=False,
        exact_gpu_claim=False, performance_claim=False)
    return d


def kernel_specs(batches, n, *, table_rows):
    from .source_ast_batched_index_matmul import kernel_spec as ordinary_spec
    from .source_ast_mapped_kv_matmul import _mapped_header, _replace_once, _zero_invalid
    d = plan(batches,n,table_rows=table_rows)
    spec = dict(ordinary_spec(batches,n));spec.pop('descriptor')
    source = spec['source']
    source = _replace_once(source,
        f'const device float* B=b+size_t(physical_tid.z)*{n*128};',
        'const device float* B=b;', 'shared index key table')
    source = _replace_once(source,
        '  B += transpose_b ? c_col_long * params->ldb : c_col_long;',
        '  const size_t mapped_b_offset = transpose_b ? c_col_long * params->ldb : c_col_long;',
        'index virtual column offset')
    source = _replace_once(source,
        '  thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);',
        f'  thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id, '
        f'row_map+size_t(physical_tid.z)*{n}, mapped_b_offset);', 'mapped index loader')
    source = _replace_once(source, '  using gemm_kernel = GEMMKernel<',
        _zero_invalid(d,'regular')+'  using gemm_kernel = GEMMKernel<', 'zero invalid index query')
    spec.update(header=_mapped_header(),source=source,input_names=['a','b','row_map','map_status'])
    validator=f'''
const uint batch=threadgroup_position_in_grid.y;
const uint lane=thread_index_in_simdgroup;
bool bad=false;
for(uint i=lane;i<{n}u;i+=32u)
  bad=bad||(row_map[size_t(batch)*{n}+i]>={table_rows}u);
const bool any_bad=simd_any(bad);
if(lane==0u)map_status[batch]=uint(any_bad);
'''
    return dict(descriptor=d,regular=spec,validate=dict(
        header='#include <metal_stdlib>\nusing namespace metal;\n',source=validator,
        input_names=['row_map'],output_names=['map_status']))


@lru_cache(maxsize=128)
def _kernels(batches,n,table_rows):
    import mlx.core as mx
    specs=kernel_specs(batches,n,table_rows=table_rows)
    key=hashlib.sha256(repr((batches,n,table_rows,implementation_identity())).encode()).hexdigest()[:16]
    return tuple(mx.fast.metal_kernel(name='mlx2_ast_mapped_index_'+stage+'_'+key,
        ensure_row_contiguous=True,**specs[stage])for stage in('validate','regular'))


def private_matmul(left,table,row_map,*,check_live):
    """Return head scores[G,32,N] and GPU map status[G], without GxN key gather.

    left[G,32,128], table[P,128], row_map[G,N]. GPU IDs stay on device.
    The caller owns the table/maps until completion and checks all status roots
    once before publication. This prototype cannot admit publication itself.
    """
    import mlx.core as mx
    if not callable(check_live)or check_live()is not True:
        raise ValueError('Source AST mapped index requires live caller ownership')
    if mx.default_device()!=mx.gpu:
        raise ValueError('Source AST mapped index requires parent-owned GPU')
    if os.environ.get(TF32_ENV)!='0':
        raise ValueError('Source AST mapped index requires explicit TF32 disabled')
    if(left.ndim!=3 or left.shape[1:]!=(32,128)or left.dtype!=mx.float32
            or table.ndim!=2 or table.shape[1]!=128 or table.dtype!=mx.float32
            or row_map.ndim!=2 or row_map.shape[0]!=left.shape[0]
            or row_map.dtype not in(mx.uint32,mx.int32)):
        raise ValueError('Source AST mapped index geometry/dtype mismatch')
    batches,n=left.shape[0],row_map.shape[1]
    d=plan(batches,n,table_rows=table.shape[0])
    if mx.metal.device_info().get('architecture')!='applegpu_g17s':
        raise ValueError('Source AST mapped index requires pinned applegpu_g17s')
    validate,matrix=_kernels(batches,n,table.shape[0])
    maps=mx.contiguous(row_map.astype(mx.uint32))
    status=validate(inputs=[maps],grid=(32,batches,1),threadgroup=(32,1,1),
        output_shapes=[(batches,)],output_dtypes=[mx.uint32])[0]
    output=matrix(inputs=[mx.contiguous(left),mx.contiguous(table),maps,status],
        grid=d['grid'],threadgroup=d['threadgroup'],output_shapes=[d['output_shape']],
        output_dtypes=[mx.float32])[0]
    return MappedProduct(output,status,d)


def implementation_identity():
    from .source_ast_batched_index_matmul import implementation_identity as index_identity
    from .source_ast_mapped_kv_matmul import implementation_identity as mapped_identity
    return hashlib.sha256(Path(__file__).read_bytes()+index_identity().encode()+mapped_identity().encode()).hexdigest()
