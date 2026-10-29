"""Mapped index source/ownership tests; GPU numerical proof remains separate."""
import mlx.core as mx
import pytest

from mlx2.runtime.gguf import source_ast_batched_index_matmul as ordinary
from mlx2.runtime.gguf import source_ast_mapped_index_matmul as mapped


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('GPU forbidden')))


@pytest.mark.parametrize('n',(129,513,1025,8195,16384))
def test_shared_table_plan_keeps_ordinary_m32_geometry_and_logical_count(n):
    old=ordinary.plan(3,n);new=mapped.plan(3,n,table_rows=n+16)
    for key in('M','N','K','BM','BN','BK','WM','WN','grid','threadgroup','output_shape'):
        assert old[key]==new[key]
    assert new['map_shape']==(3,n) and new['map_bytes']==3*n*4
    assert new['eliminated_gather_bytes']==new['map_bytes']*128
    assert not new['materialized_query_index_keys']
    assert not new['explicit_cross_query_threadgroup_reuse']
    assert new['publication_requires_zero_map_status']
    assert not new['selected'] and not new['qualified'] and not new['exact_gpu_claim']


def test_only_b_addresses_change_with_m32_guards_and_status_dependency():
    old=ordinary.kernel_spec(3,513);new=mapped.kernel_specs(3,513,table_rows=521)
    before,after=old['source'],new['regular']['source']
    assert 'constexpr bool align_M=false,align_N=false,align_K=true;' in after
    for statement in('mma_op.mma(As, Bs);','gemm_kernel::gemm_loop(',
                     'threadgroup_barrier(','loader_b.src += k_jump_b;',
                     'loader_b.src -= k_jump_b;','loader_b.next();'):
        assert before.count(statement)==after.count(statement)
    assert 'const device float* B=b;' in after
    assert 'row_map+size_t(physical_tid.z)*513' in after
    assert 'using loader_a_t = BlockLoader<' in new['regular']['header']
    assert 'using loader_b_t = MappedBlockLoader<' in new['regular']['header']
    assert new['regular']['input_names']==['a','b','row_map','map_status']
    assert after.index('if (map_status[')<after.index('  using gemm_kernel')
    assert 'if(row<32u&&col<513u)' in after
    assert '>=521u' in new['validate']['source'] and 'simd_any(bad)' in new['validate']['source']


def test_cpu_or_missing_caller_refuses_before_any_shader_build():
    q=mx.zeros((3,32,128),mx.float32);table=mx.zeros((521,128),mx.float32);rows=mx.zeros((3,513),mx.int32)
    with pytest.raises(ValueError,match='live caller'):
        mapped.private_matmul(q,table,rows,check_live=lambda:False)
    with pytest.raises(ValueError,match='parent-owned GPU'):
        mapped.private_matmul(q,table,rows,check_live=lambda:True)
    for size in(0,-1,False,2**32):
        with pytest.raises(ValueError,match='row count'):
            mapped.plan(3,513,table_rows=size)
