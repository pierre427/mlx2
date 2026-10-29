"""Native policy insertion/loader lifecycle; CPU references, no GPU dot proof."""
import numpy as np
import mlx.core as mx
import pytest

from mlx2.adapters import deepseek_v41_source_native_mx as policy
from test_deepseek_v41_source_prefill_integration_cpu import case,run,expected
from test_deepseek_v41_source_buffers_cpu import lfu_fixture


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:pytest.fail('CPU fixture attempted Metal'))
    yield
    mx.set_default_device(old)


@pytest.mark.parametrize('owned',[True,False])
def test_native_original_gu_down_keep_middle_score_rows_and_global_assembly(monkeypatch,owned):
    c=case(monkeypatch,[[5,0,3,2,4,1]]*17,owned=owned)
    c.w.source_native_mx_policy=policy.POLICY
    calls=[]
    def identity(data):return next(e for e,d in c.bank.items()if d is data)
    def gate_up(owner,x,data,prefix,expert,*,prefill_rows):
        assert owner is c.w and prefix=='blk.0'and prefill_rows==17
        assert c.store._active and len(c.store._leased)==6 and expert==identity(data)
        calls.append((expert,'gu',x.shape[0]))
        return mx.full((len(x),2304),expert+1,mx.float32),mx.full((len(x),2304),expert+1,mx.float32)
    def down(owner,x,data,prefix,expert,*,prefill_rows):
        if data.layout.kind=='q1':return None
        assert c.store._active and len(c.store._leased)==6 and expert==identity(data)
        calls.append((expert,'down',x.shape[0]))
        return mx.repeat(((expert+1)*x[:,0])[:,None],5120,axis=1)
    monkeypatch.setattr(policy,'maybe_gate_up',gate_up);monkeypatch.setattr(policy,'maybe_down',down)
    assert run(c).tobytes()==expected(c).tobytes()
    assert calls==[(e,'gu',17)for e in (0,2,4)]+[(e,'down',17)for e in (0,2,4)]
    assert c.calls['projections']==[(e,'down',17)for e in (1,3,5)]
    assert len(c.calls['qat'])==1 and len(c.calls['combine'])==1
    assert c.w.scheduled_expert_calls['official_source_prefill_ordinary_down_jobs']==3
    assert c.w.scheduled_expert_calls['official_source_prefill_assembly_parts']==6
    assert len(c.acquired)==6 and c.store.stats['lookups']==1
    assert bool(c.store._leased)is(not owned)


@pytest.mark.parametrize('failure',['gu','down'])
@pytest.mark.parametrize('owned',[True,False])
def test_selected_native_fault_preserves_existing_provider_completion(monkeypatch,failure,owned):
    c=case(monkeypatch,[[5,0,3,2,4,1]]*17,owned=owned)
    c.w.source_native_mx_policy=policy.POLICY
    def fail(*args,**kwargs):
        assert c.store._active and len(c.store._leased)==6
        c.store._pending_host['sentinel']=object();c.store._pending_uploads['sentinel']={}
        raise RuntimeError('injected native '+failure)
    if failure=='gu':monkeypatch.setattr(policy,'maybe_gate_up',fail)
    else:
        monkeypatch.setattr(policy,'maybe_gate_up',lambda owner,x,data,prefix,e,**kw:(mx.full((len(x),2304),e+1,mx.float32),mx.full((len(x),2304),e+1,mx.float32)))
        monkeypatch.setattr(policy,'maybe_down',fail)
    with pytest.raises(RuntimeError,match='injected native'):run(c)
    if owned:
        assert not c.store._active and not c.store._leased and not c.store._pending_host and len(c.fences)==1
    else:
        assert c.store._active and len(c.store._leased)==6 and c.store._pending_host and not c.fences


@pytest.mark.parametrize('mixed',[False,True])
def test_loader_observer_runs_once_on_original_miss_after_fence_before_host_release(tmp_path,monkeypatch,mixed):
    store=lfu_fixture(tmp_path,monkeypatch,mixed=mixed)
    events=[];original=mx.eval
    def evaluate(*leaves):events.append('upload_fence');return original(*leaves)
    monkeypatch.setattr(mx,'eval',evaluate)
    def observer(key,host,packed):
        assert events[-1]=='upload_fence'
        assert store._pending_host[key]is host and store._pending_uploads[key]is packed
        assert key not in store._cache and key not in store._leased
        events.append(('observer',key))
    store._packed_original_loaded=observer
    with store.window():
        first=store.acquire('blk.0',0)
        assert store.acquire('blk.0',0)is first
        if mixed:store.acquire('blk.0',1)
    with store.window():assert store.acquire('blk.0',0)is first
    assert [event for event in events if isinstance(event,tuple)]==[('observer',(0,0))]
    assert not store._pending_host and not store._pending_uploads
    store.close()


@pytest.mark.parametrize('error',[RuntimeError,KeyboardInterrupt])
@pytest.mark.parametrize('fence_failure',[False,True])
def test_failed_observer_keeps_pending_backing_host_until_successful_fence(tmp_path,monkeypatch,error,fence_failure):
    from mlx2.adapters.deepseek_v41_mlx_weights import _UPLOAD_QUARANTINE
    store=lfu_fixture(tmp_path,monkeypatch)
    def observer(key,host,packed):
        assert store._pending_host[key]is host and store._pending_uploads[key]is packed
        raise error('injected observer')
    store._packed_original_loaded=observer
    if fence_failure:
        def reject():raise RuntimeError('injected fence')
        store._synchronize=reject
    with pytest.raises((RuntimeError,KeyboardInterrupt)):
        with store.window():store.acquire('blk.0',0)
    assert bool(store._pending_host)==fence_failure and bool(store._pending_uploads)==fence_failure
    if fence_failure:assert _UPLOAD_QUARANTINE[id(store)]is store
    store._synchronize=lambda:None
    store.close()
    assert not store._pending_host and not store._pending_uploads and id(store)not in _UPLOAD_QUARANTINE
