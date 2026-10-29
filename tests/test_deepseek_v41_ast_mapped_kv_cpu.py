"""Mapped loader source/ownership contract only; no actual GPU parity claim."""
import mlx.core as mx
import pytest

from mlx2.runtime.gguf import source_ast_batched_splitk as ordinary
from mlx2.runtime.gguf import source_ast_mapped_kv_matmul as mapped


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, 'metal_kernel', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('GPU forbidden')))


@pytest.mark.parametrize('n,k,transpose', ((11,512,True),(512,11,False),(128,512,True),(512,128,False),(512,512,True),(512,512,False)))
def test_descriptor_retains_ordinary_geometry_and_replaces_gather_with_maps(n,k,transpose):
    before = ordinary.plan(3,64,n,k,transpose_b=transpose)
    after = mapped.plan(3,64,n,k,transpose_b=transpose,table_rows=2048)
    for key in ('mode','M','N','K','BM','BN','BK','WM','WN','grid','threadgroup','output_shape'):
        assert before[key] == after[key]
    assert after['map_columns'] == (n if transpose else k)
    assert after['map_bytes']*512 == after['eliminated_gather_bytes']
    assert not after['materialized_query_KV'] and not after['explicit_cross_query_threadgroup_reuse']
    assert after['publication_requires_zero_map_status'] and not after['selected']
    assert not after['qualified'] and not after['exact_gpu_claim']


@pytest.mark.parametrize('n,k,transpose', ((11,512,True),(512,11,False),(512,512,True),(512,512,False)))
def test_only_b_addressing_changes_and_stock_mma_tail_partitions_stay(n,k,transpose):
    before = ordinary.kernel_specs(3,64,n,k,transpose_b=transpose)
    after = mapped.kernel_specs(3,64,n,k,transpose_b=transpose,table_rows=2048)
    stage = 'split' if before['descriptor']['mode']=='splitk' else 'regular'
    old, new = before[stage]['source'], after[stage]['source']
    assert old.count('mma_op.mma(As, Bs);') == new.count('mma_op.mma(As, Bs);')
    assert old.count('gemm_kernel::gemm_loop(') == new.count('gemm_kernel::gemm_loop(')
    assert old.count('threadgroup_barrier(') == new.count('threadgroup_barrier(')
    assert 'row_map+size_t(' in new and 'mapped_b_offset' in new
    assert 'const device float* B=b;' in new
    assert 'if (map_status[' in new and new.index('if (map_status[') < new.index('  using gemm_kernel')
    assert after[stage]['input_names'] == ['a','b','row_map','map_status']
    assert 'using loader_a_t = BlockLoader<' in after[stage]['header']
    assert 'using loader_b_t = MappedBlockLoader<' in after[stage]['header']
    if stage=='split':assert before['accum'] == after['accum']
    else:
        for statement in ('loader_b.src += k_jump_b;', 'loader_b.src -= k_jump_b;', 'loader_b.next();'):
            assert old.count(statement) == new.count(statement)


def test_mapped_loader_uses_logical_cursor_and_preserves_safe_zero_and_vector_loads():
    header = mapped._mapped_header()
    block = header[header.index('struct MappedBlockLoader {'):]
    assert 'size_t src;' in block
    assert 'const size_t logical_row = offset / size_t(src_ld);' in block
    assert 'const size_t feature = offset % size_t(src_ld);' in block
    assert 'row_map ? size_t(row_map[logical_row]) : logical_row' in block
    assert '*((threadgroup ReadVector*)(&dst[i * dst_ld]))' in block
    assert '(address(src + i * src_ld))' in block
    assert '*address(src + (tmp_idx[j] ? i * src_ld + j : 0))' in block
    assert 'src_tile_dim.x <= 0 || src_tile_dim.y <= 0' in block
    assert 'src += tile_stride;' in block


def test_validator_checks_all_map_rows_and_declares_gpu_status_dependency():
    spec = mapped.kernel_specs(3,64,11,512,transpose_b=True,table_rows=143)
    source = spec['validate']['source']
    assert 'for(uint i=lane;i<11u;i+=32u)' in source
    assert '>=143u' in source and 'simd_any(bad)' in source
    assert spec['validate']['output_names'] == ['map_status']
    assert 'map_status' in spec['split']['input_names']


def test_cpu_and_missing_owner_refuse_before_kernel_build():
    left=mx.zeros((3,64,512),mx.float32);table=mx.zeros((143,512),mx.float32);rows=mx.zeros((3,11),mx.uint32)
    with pytest.raises(ValueError,match='live caller'):
        mapped.private_matmul(left,table,rows,transpose_b=True,check_live=lambda:False)
    with pytest.raises(ValueError,match='parent-owned GPU'):
        mapped.private_matmul(left,table,rows,transpose_b=True,check_live=lambda:True)
    for value in (0,-1,False,2**32):
        with pytest.raises(ValueError,match='row count'):
            mapped.plan(3,64,11,512,transpose_b=True,table_rows=value)
