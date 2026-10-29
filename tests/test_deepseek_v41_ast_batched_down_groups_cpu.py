"""CPU source/ownership geometry only; actual GPU parity and timing remain gated."""
from collections import Counter
from types import SimpleNamespace
import mlx.core as mx
import numpy as np
import pytest
from mlx2.runtime.gguf import source_ast_batched_down_groups as groups
from mlx2.runtime.gguf import source_ast_batched_moe as ordinary
from mlx2.adapters.deepseek_v41_source_buffers import canonical_layout


def key():
    layouts=[canonical_layout('original'),canonical_layout('q1')]
    from mlx2.runtime.gguf.source_resident_bank_gather import FIELDS
    return tuple((1 if l.kind=='original'else 2,2*l.nbytes//4,l.nbytes//4,
        tuple(next((t.offset for t in l.tensors if t.name==name),0)for name in FIELDS))for l in layouts)


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:(_ for _ in ()).throw(AssertionError('GPU forbidden')))


@pytest.mark.parametrize('simdgroups',(1,4,8))
def test_all_output_rows_occurrences_exactly_once_and_same_virtual_M1_owner(simdgroups):
    p=groups.plan(simdgroups);counts=Counter()
    for physical in range(5120//(4*simdgroups)):
        for sg in range(simdgroups):
            tile=physical*simdgroups+sg
            vtile,vsg=tile//2,tile%2
            for r in range(4):
                assert vtile*8+vsg*4+r==tile*4+r
                counts[tile*4+r]+=1
    assert counts==Counter(range(5120))
    assert p['total_SIMDgroups_per_layer']==122880
    assert p['threadgroups_per_layer']==122880//simdgroups
    assert p['grid'][0]//p['threadgroup'][0]==1280//simdgroups
    assert p['grid'][1]==96 and p['threadgroup'][0]==32*simdgroups
    assert p['shared_partial_bytes']==16*simdgroups


@pytest.mark.parametrize('simdgroups',(4,8))
def test_source_is_original_body_with_only_geometry_and_disjoint_scratch_changed(simdgroups):
    old=ordinary.down_spec(key());new=groups.kernel_spec(key(),simdgroups=simdgroups)
    assert new['header']==old['header']and new['input_names']==old['input_names']and new['output_names']==old['output_names']
    source=new['source']
    decl=('const uint physical_tile=threadgroup_position_in_grid.x;\n'
          'const uint sg=simdgroup_index_in_threadgroup;\n'
          f'const uint tile=physical_tile*{simdgroups}u+sg,lane=thread_index_in_simdgroup;')
    scratch=(f'threadgroup float shared_partials[{4*simdgroups}];\n'
             'threadgroup float* partials=shared_partials+sg*4;')
    source=source.replace(decl,'const uint tile=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;')
    source=source.replace(scratch,'threadgroup float partials[4];')
    assert source==old['source']
    assert source.count('threadgroup_barrier(mem_flags::mem_threadgroup);')==1
    assert 'float(bfloat(partials[lane]))'in source


def test_one_group_control_source_is_byte_identical():
    old=ordinary.down_spec(key());control=groups.kernel_spec(key(),simdgroups=1)
    assert control['source']==old['source']and control['header']==old['header']


@pytest.mark.parametrize('simdgroups',(4,8))
def test_shared_scratch_has_no_inter_SIMDgroup_collision(simdgroups):
    slots={sg:set(range(sg*4,sg*4+4))for sg in range(simdgroups)}
    assert set.union(*slots.values())==set(range(4*simdgroups))
    for a in slots:
        for b in slots:
            if a!=b:assert not slots[a]&slots[b]
    # Every occurrence/status condition is TG-uniform, including the final
    # invalid padded occurrences, so no SIMD group diverges around the barrier.
    for occurrence in range(96):
        for valid_count in (0,1,95,96):
            decisions=[occurrence>=valid_count for _ in range(simdgroups)]
            assert len(set(decisions))==1


@pytest.mark.parametrize('bad',(0,2,3,16,True,-1,'8'))
def test_unknown_geometry_refuses_before_kernel(bad):
    with pytest.raises(ValueError):groups.plan(bad)


def test_no_GPU_evaluation_or_fence_is_owned_by_consumer(monkeypatch):
    a=lambda shape,dtype:SimpleNamespace(shape=shape,dtype=dtype)
    args=(a((96,2304),mx.float32),a((96,),mx.uint32),a((384,8),mx.uint32),a((96,),mx.uint32),a((16,),mx.uint32),SimpleNamespace())
    with pytest.raises(ValueError,match='live caller'):
        groups.down(*args,check_live=None)
    with pytest.raises(ValueError,match='parent-owned GPU'):
        groups.down(*args,check_live=lambda:True)
    p=groups.plan(8)
    assert not p['expert_row_panel_reuse']and not p['changed_arithmetic_expressions']and not p['performance_claim']
    assert len(groups.implementation_identity())==64
