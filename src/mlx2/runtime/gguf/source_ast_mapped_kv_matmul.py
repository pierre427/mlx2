# SPDX-License-Identifier: MIT
"""Unselected shared-table M64 products with exact-body mapped B loads.

Only B addressing changes. Installed MMA/loop/tail/partition statements remain
unchanged. GPU map status MUST be checked before using/publishing any output.
No active adapter, cache, numerical identity or default selects this prototype.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from ...process_env import TF32_ENV
from pathlib import Path

PROFILE = 'source_ast_mapped_kv_m64_candidate_v1'


@dataclass(frozen=True)
class MappedProduct:
    output: object
    status: object
    descriptor: dict


def plan(batches, m, n, k, *, transpose_b, table_rows):
    from .source_ast_batched_splitk import plan as ordinary_plan
    d = ordinary_plan(batches, m, n, k, transpose_b=transpose_b)
    if type(table_rows) is not int or not 1 <= table_rows <= 0x7fffffff:
        raise ValueError('Source AST mapped KV table row count invalid')
    d = dict(d)
    d.update(profile=PROFILE, table_rows=table_rows, table_dim=k if transpose_b else n,
        map_columns=n if transpose_b else k, map_shape=(batches, n if transpose_b else k),
        map_bytes=batches*(n if transpose_b else k)*4,
        eliminated_gather_bytes=batches*n*k*4, materialized_query_KV=False,
        explicit_cross_query_threadgroup_reuse=False, map_validation_launches=1,
        publication_requires_zero_map_status=True, qualified=False, selected=False,
        exact_gpu_claim=False, performance_claim=False)
    return d


def _replace_once(text, old, new, label):
    if text.count(old) != 1:
        raise RuntimeError('Pinned mapped KV source anchor changed: '+label)
    return text.replace(old, new, 1)


@lru_cache(maxsize=1)
def _mapped_header():
    from .source_ast_batched_splitk import _installed, _header
    source = (_installed()/'steel/gemm/loader.h').read_text()
    source = source[source.index('template <'):source.rindex('};')+2]
    source = _replace_once(source, 'struct BlockLoader {', 'struct MappedBlockLoader {', 'loader class')
    source = _replace_once(source, 'METAL_FUNC BlockLoader(', 'METAL_FUNC MappedBlockLoader(', 'loader constructor')
    source = _replace_once(source, '  const device T* src;',
        '  size_t src;\n  const device T* table;\n  const device uint* row_map;', 'logical source cursor')
    source = _replace_once(source,
        '      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread',
        '      ushort simd_lane_id [[thread_index_in_simdgroup]],\n'
        '      const device uint* map_ = nullptr,\n      size_t logical_base = 0) thread', 'map arguments')
    source = _replace_once(source, '        src(src_ + bi * src_ld + bj) {}',
        '        src(logical_base + bi * src_ld + bj), table(src_), row_map(map_) {}', 'cursor initializer')
    address = '''  // Translate only the load address; logical K movement stays stock.
  METAL_FUNC const device T* address(size_t offset) const thread {
    const size_t logical_row = offset / size_t(src_ld);
    const size_t feature = offset % size_t(src_ld);
    const size_t physical_row = row_map ? size_t(row_map[logical_row]) : logical_row;
    return table + physical_row * size_t(src_ld) + feature;
  }

'''
    source = _replace_once(source, '  /* Apply operation to threadgroup without bound checking */',
        address+'  /* Apply operation to threadgroup without bound checking */', 'address function')
    source = _replace_once(source, '(&src[i * src_ld])', '(address(src + i * src_ld))', 'unsafe address')
    source = _replace_once(source, 'src[(tmp_idx[j] ? i * src_ld + j : 0)]',
        '*address(src + (tmp_idx[j] ? i * src_ld + j : 0))', 'safe address')
    header = _header()
    anchor = 'template <bool M_aligned, bool N_aligned, bool K_aligned>\nstruct LoopAlignment {};'
    header = _replace_once(header, anchor, source+'\n\n'+anchor, 'mapped loader insertion')
    return _replace_once(header, '  using loader_b_t = BlockLoader<',
        '  using loader_b_t = MappedBlockLoader<', 'only B loader alias')


def _zero_invalid(d, stage):
    m, n, bm, bn = (d[x] for x in ('M', 'N', 'BM', 'BN'))
    if stage == 'split':
        groups = d['partitions']; batch = 'batch'; gx = 'tid.x'; gy = 'tid.y'
        target = f'split[(size_t(batch)*{groups}+tid.z%{groups})*{m*n}+size_t(row)*{n}+col]'
    else:
        batch = 'physical_tid.z'; gx = 'physical_tid.x'; gy = 'physical_tid.y'
        target = f'result_buffer[size_t(physical_tid.z)*{m*n}+size_t(row)*{n}+col]'
    return f'''  // Invalid maps never reach B loads; caller still rejects status.
  if (map_status[{batch}] != 0u) {{
    for (uint i=simd_group_id*32+simd_lane_id;i<{bm*bn}u;i+=128u) {{
      const uint row={gy}*{bm}+i/{bn};
      const uint col={gx}*{bn}+i%{bn};
      if(row<{m}u&&col<{n}u){target}=0.0f;
    }}
    return;
  }}
'''


def kernel_specs(batches, m, n, k, *, transpose_b, table_rows):
    from .source_ast_batched_splitk import kernel_specs as ordinary_specs
    d = plan(batches, m, n, k, transpose_b=transpose_b, table_rows=table_rows)
    original = ordinary_specs(batches, m, n, k, transpose_b=transpose_b)
    stage = 'split' if d['mode'] == 'splitk' else 'regular'
    spec = dict(original[stage]); source = spec['source']
    batch = 'batch' if stage == 'split' else 'physical_tid.z'
    source = _replace_once(source,
        f'const device float* B=b+size_t({batch})*{n*k};',
        'const device float* B=b;', 'shared B base')
    if stage == 'split':
        old = ('  B += transpose_b ? (k_start_long + c_col_long * params->ldb)\n'
               '                   : (c_col_long + k_start_long * params->ldb);')
        new = ('  const size_t mapped_b_offset = transpose_b ? (k_start_long + c_col_long * params->ldb)\n'
               '                   : (c_col_long + k_start_long * params->ldb);')
    else:
        old = '  B += transpose_b ? c_col_long * params->ldb : c_col_long;'
        new = '  const size_t mapped_b_offset = transpose_b ? c_col_long * params->ldb : c_col_long;'
    source = _replace_once(source, old, new, 'logical tile offset')
    source = _replace_once(source,
        '  thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);',
        f'  thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id, '
        f'row_map+size_t({batch})*{d["map_columns"]}, mapped_b_offset);', 'mapped B loader construction')
    source = _replace_once(source, '  using gemm_kernel = GEMMKernel<',
        _zero_invalid(d, stage)+'  using gemm_kernel = GEMMKernel<', 'zero invalid batch')
    spec.update(header=_mapped_header(), source=source,
        input_names=['a', 'b', 'row_map', 'map_status'])
    result = dict(descriptor=d, **{stage: spec})
    if stage == 'split':result['accum'] = original['accum']
    validator = f'''
const uint batch=threadgroup_position_in_grid.y;
const uint lane=thread_index_in_simdgroup;
bool bad=false;
for(uint i=lane;i<{d['map_columns']}u;i+=32u)
  bad=bad||(row_map[size_t(batch)*{d['map_columns']}+i]>={table_rows}u);
const bool any_bad=simd_any(bad);
if(lane==0u)map_status[batch]=uint(any_bad);
'''
    result['validate'] = dict(header='#include <metal_stdlib>\nusing namespace metal;\n',
        source=validator, input_names=['row_map'], output_names=['map_status'])
    return result


@lru_cache(maxsize=128)
def _kernels(batches, m, n, k, transpose_b, table_rows):
    import mlx.core as mx
    specs = kernel_specs(batches, m, n, k, transpose_b=transpose_b, table_rows=table_rows)
    key = hashlib.sha256(repr((batches,m,n,k,transpose_b,table_rows,implementation_identity())).encode()).hexdigest()[:16]
    stages = ('validate','split','accum') if specs['descriptor']['mode']=='splitk' else ('validate','regular')
    return tuple(mx.fast.metal_kernel(name='mlx2_ast_mapped_kv_'+stage+'_'+key,
        ensure_row_contiguous=True, **specs[stage]) for stage in stages)


def private_matmul(left, table, row_map, *, transpose_b, check_live):
    """Return output+mandatory GPU status; retain both until owned completion.

    NT maps each query's output columns; NN maps its chronological reduction
    rows. A single common table[T,D] replaces gathered[G,C,D] coefficients.
    This function cannot authorize cache/state publication.
    """
    import mlx.core as mx
    if not callable(check_live) or check_live() is not True:
        raise ValueError('Source AST mapped KV requires live caller ownership')
    if mx.default_device() != mx.gpu:
        raise ValueError('Source AST mapped KV requires parent-owned GPU')
    if os.environ.get(TF32_ENV) != '0':
        raise ValueError('Source AST mapped KV requires explicit TF32 disabled')
    if (left.ndim!=3 or table.ndim!=2 or row_map.ndim!=2
            or left.dtype!=mx.float32 or table.dtype!=mx.float32
            or row_map.dtype not in(mx.uint32,mx.int32) or row_map.shape[0]!=left.shape[0]):
        raise ValueError('Source AST mapped KV input geometry/dtype mismatch')
    batches,m,k = left.shape
    n = row_map.shape[1] if transpose_b else table.shape[1]
    if (transpose_b and table.shape[1]!=k) or (not transpose_b and row_map.shape[1]!=k):
        raise ValueError('Source AST mapped KV row-map dimension mismatch')
    d = plan(batches,m,n,k,transpose_b=transpose_b,table_rows=table.shape[0])
    if mx.metal.device_info().get('architecture') != 'applegpu_g17s':
        raise ValueError('Source AST mapped KV requires pinned applegpu_g17s')
    kernels = _kernels(batches,m,n,k,transpose_b,table.shape[0])
    maps = mx.contiguous(row_map.astype(mx.uint32)); a=mx.contiguous(left); b=mx.contiguous(table)
    status = kernels[0](inputs=[maps], grid=(32,batches,1), threadgroup=(32,1,1),
        output_shapes=[(batches,)], output_dtypes=[mx.uint32])[0]
    if d['mode']=='splitk':
        partial = kernels[1](inputs=[a,b,maps,status], grid=d['grid'], threadgroup=d['threadgroup'],
            output_shapes=[d['split_shape']],output_dtypes=[mx.float32])[0]
        output = kernels[2](inputs=[partial], grid=(n,m,batches),
            threadgroup=(min(16,n),min(16,m),1), output_shapes=[d['output_shape']],
            output_dtypes=[mx.float32])[0]
    else:
        output = kernels[1](inputs=[a,b,maps,status],grid=d['grid'],threadgroup=d['threadgroup'],
            output_shapes=[d['output_shape']],output_dtypes=[mx.float32])[0]
    return MappedProduct(output,status,d)


def implementation_identity():
    from .source_ast_batched_splitk import implementation_identity as ordinary_identity
    return hashlib.sha256(Path(__file__).read_bytes()+ordinary_identity().encode()).hexdigest()
