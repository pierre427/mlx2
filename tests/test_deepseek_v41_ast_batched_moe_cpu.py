"""Resident M16 GPU source/owner contracts only; no GPU/checkpoint qualification."""
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import pytest
from mlx2.adapters import deepseek_v41_ast_batched_moe as adapter
from mlx2.runtime.gguf import source_ast_batched_moe as kernels


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:pytest.fail('GPU forbidden'))
    yield
    mx.set_default_device(old)


class Backend:
    any=staticmethod(np.any);all=staticmethod(np.all);stack=staticmethod(np.stack);isfinite=staticmethod(np.isfinite);where=staticmethod(np.where)
    def __init__(self,fail=False):self.fail=fail;self.calls=0
    def eval(self,*roots):
        self.calls+=1
        if self.fail:raise RuntimeError('failed completion')


class Bank:
    generation=7
    def __init__(self):self.leases=0;self.checks=[]
    @contextmanager
    def lease(self,*,ownercheck):
        ownercheck();self.leases+=1
        try:yield self
        finally:self.leases-=1
    def assert_current(self,*,contents=True):self.checks.append(contents)


def setup(*,fail=False):
    bank=Bank();weights=SimpleNamespace(expert_stream=SimpleNamespace(_resident_bank=bank),
        source_resident_gather_policy='resident_bank_v1',source_resident_bank_session=None,
        expert_math_policy='routed_fused_fp32',scheduled_expert_calls={},_ready=lambda:None)
    backend=Backend(fail);scope=adapter.resident_scope(weights,lambda:True,bank=bank,backend=backend)
    return weights,bank,backend,scope


def result(status=0,finite=True):
    out=np.full((16,8),1. if finite else np.nan)
    return SimpleNamespace(output=out,status=np.array([status],np.uint32),roots=(out,))


def test_tile_scope_pins_whole_bank_and_requires_explicit_status_gate():
    w,b,xp,scope=setup()
    with scope:
        assert b.leases==1 and scope.check_live()
        scope.retain(result());scope.retain(result())
        receipt=scope.complete('privatecache')
        assert receipt['completed']and receipt['moe_tiles']==2 and xp.calls==1
    assert b.leases==0 and w._source_ast_bank_scope is None
    assert b.checks and all(v is False for v in b.checks)


@pytest.mark.parametrize('status,finite',[(1,True),(8,True),(0,False)])
def test_missing_stale_or_nonfinite_tile_never_falls_back(status,finite):
    w,b,xp,scope=setup()
    with pytest.raises(ValueError,match='no streaming fallback'):
        with scope:scope.retain(result(status,finite));scope.complete()
    assert b.leases==0 and scope.receipt()['ordinary_fallbacks']==0


def test_missing_completion_gate_refuses_even_if_outputs_were_finite():
    w,b,xp,scope=setup()
    with pytest.raises(RuntimeError,match='required before publication'):
        with scope:scope.retain(result())
    assert b.leases==0


def test_failed_drain_keeps_bank_lease_and_all_encoded_owners():
    w,b,xp,scope=setup(fail=True);r=result();private=object()
    with pytest.raises(RuntimeError,match='failed completion'):
        with scope:scope.retain(r);scope.complete(private)
    assert scope.poisoned and scope in adapter._QUARANTINE and b.leases==1
    assert scope.external_roots==(private,)and any(v is r.output for v in scope.roots)
    assert w._source_ast_bank_scope is scope


def test_owner_callback_can_come_only_from_prepared_session():
    w,b,xp,scope=setup();w.source_resident_bank_session=SimpleNamespace(bank=b,ownercheck=lambda:True)
    assert adapter.resident_scope(w,backend=xp).bank is b
    w.source_resident_bank_session=None
    with pytest.raises(ValueError,match='ownership'):adapter.resident_scope(w,bank=b)


def test_default_route_and_unprepared_or_active_bank_refuse():
    w,b,xp,scope=setup();w.source_resident_gather_policy='reference'
    with pytest.raises(ValueError,match='explicit'):adapter.resident_scope(w,lambda:True,bank=b)
    w.source_resident_gather_policy='resident_bank_v1';w.expert_stream._resident_bank=None
    with pytest.raises(ValueError,match='attached'):adapter.resident_scope(w,lambda:True,bank=b)
    w.expert_stream._resident_bank=b;w.source_resident_bank_session=SimpleNamespace(active=True)
    with pytest.raises(ValueError,match='active'):adapter.resident_scope(w,lambda:True,bank=b)


def test_moe_routes_gpu_arrays_unchanged_and_keeps_row_qat(monkeypatch):
    from mlx2.runtime.models import deepseek_v41_quant as q
    prepared=[];calls=[]
    monkeypatch.setattr(q,'fused_quantize',lambda value,kind:prepared.append((value,kind))or value)
    w,b,xp,scope=setup()
    def consumer(x,ids,scores,lease,**options):
        calls.append((x,ids,scores,lease,options));return result()
    scope.consumer=consumer
    ids=mx.zeros((16,6),mx.uint32);scores=mx.ones((16,6),mx.float32)
    with scope:
        output=adapter.moe_tile(w,mx.ones((16,5120),mx.float32),ids,scores,'blk.3')
        assert output.shape==(16,8);scope.complete()
    assert calls[0][1]is ids and calls[0][2]is scores and calls[0][3]is b
    assert calls[0][4]['layer']==3 and prepared[0][1]=='fp8'
    assert not hasattr(w,'source_route_host_materializations')
    assert w.scheduled_expert_calls['source_ast_resident_gpu_occurrences']==96


def test_source_generation_uses_grouped_private_m1_law_and_cached_coefficients():
    key=((1,100000000,4700160,(0,1,0,2,3,0,4,5,0)),(2,100000000,1658880,(0,1,2,3,4,5,6,7,8)))
    route=kernels.route_spec(key,3,7)
    assert 'const uint vecs=2u;'in route['source']and 'count>12u'not in route['source']
    gu=kernels.gu_spec(key)
    assert 'qdot_cached' in gu['header']and 'const thread uint16_t* ws' in gu['header']
    assert 'float gates[2][4]'in gu['source']and 'for(uint v=0;v<active;++v)'in gu['source']
    assert gu['source'].index('ushort gp[4][4]')<gu['source'].index('for(uint v=0;v<active;++v)')
    assert 'fp_qmv_wide'not in gu['source']and 'split_k'not in gu['source']
    assert len(gu['input_names'])+len(gu['output_names'])==9+len(key)
    down=kernels.down_spec(key)
    assert 'fp_qmv_to_shared_impl<float,32,4,false>'in down['source']
    assert 'float(bfloat(partials[lane]))'in down['source']
    combine=kernels.combine_spec()['source']
    assert 'volatile float total=0.0f' in combine and 'slots[j-1]'in combine


def test_cached_original_panel_walk_matches_pinned_byte_stride_without_crossing_rows():
    key=((1,100000000,4700160,(0,1,0,2,3,0,4,5,0)),)
    source=kernels.gu_spec(key)['source']
    assert '(out_row+r)*1280+k/4+lane*4+i'in source
    # Pinned M1 qdot walks byte pointers by256 for every512 activation values.
    # Register-panel ushort indices must denote exactly those same eight bytes.
    for k in range(0,5120,512):
        for lane in range(32):
            first=k//4+lane*4
            assert first*2==k//2+lane*8
            assert 0<=first and first+3<1280


def hybrid_result(*,stale=False,unknown=False,bad_good_row=False):
    groups=np.zeros((384,8),np.uint32);groups[1,[0,2,5]]=(1,0,25)
    status=np.zeros(16,np.uint32);status[0]=25
    out=np.ones((16,8),np.float32);out[0]=np.nan
    if stale:groups[2,[0,2,5]]=(1,1,8);status[0]|=8
    if unknown:status[0]|=128
    if bad_good_row:out[1]=np.nan
    return SimpleNamespace(output=out,status=status,route_meta=groups,roots=(out,status,groups))


def test_hybrid_missing_group_is_typed_only_after_drain_and_lease_release():
    w,b,xp,scope=setup();scope.recover_misses=True
    with pytest.raises(adapter.ResidentCohortMiss)as caught:
        with scope:scope.retain(hybrid_result());scope.complete('private')
    assert caught.value.drained and b.leases==0 and scope.drained and not scope.active


@pytest.mark.parametrize('change',[{'stale':True},{'unknown':True},{'bad_good_row':True}])
def test_hybrid_missing_does_not_hide_stale_unknown_or_independent_nonfinite(change):
    w,b,xp,scope=setup();scope.recover_misses=True
    with pytest.raises(ValueError)as caught:
        with scope:scope.retain(hybrid_result(**change));scope.complete()
    assert not isinstance(caught.value,adapter.ResidentCohortMiss)
    assert b.leases==0


def test_hybrid_masks_only_failed_private_rows_and_still_requires_typed_status_gate(monkeypatch):
    from mlx2.runtime.models import deepseek_v41_quant as quant
    monkeypatch.setattr(quant,'fused_quantize',lambda value,kind:value)
    w,b,xp,scope=setup();scope.recover_misses=True
    failed=hybrid_result();failed.output=mx.array(failed.output);failed.status=mx.array(failed.status)
    failed.route_meta=mx.array(failed.route_meta);failed.roots=(failed.output,failed.status,failed.route_meta)
    scope.backend=mx
    scope.consumer=lambda *a,**k:failed
    with pytest.raises(adapter.ResidentCohortMiss):
        with scope:
            output=adapter.moe_tile(w,mx.ones((16,5120),mx.float32),mx.zeros((16,6),mx.uint32),
                mx.ones((16,6),mx.float32),'blk.0')
            np.testing.assert_array_equal(np.asarray(output)[0],np.zeros(8))
            np.testing.assert_array_equal(np.asarray(output)[1:],np.ones((15,8)))
            assert np.isnan(np.asarray(failed.output)[0]).all()
            scope.complete()
    assert b.leases==0
