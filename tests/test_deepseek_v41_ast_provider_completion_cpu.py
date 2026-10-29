"""Real NativeExpertStore/BoundedSourceCompletion ownership regression.

No real checkpoint or GPU; arrays are evaluated on CPU and Metal is forbidden.
"""
from collections import OrderedDict
from contextlib import contextmanager
from types import SimpleNamespace
import threading
import mlx.core as mx
import pytest
from mlx2.adapters import deepseek_v41_ast_batched_moe as bank_scope_module
from mlx2.adapters import deepseek_v41_ast_bank_manager as manager_module
from mlx2.adapters import deepseek_v41_mlx_weights as native
from mlx2.adapters.deepseek_v41_source_completion import BoundedSourceCompletion
from mlx2.adapters.deepseek_v41_ast_batched_target import _completed_provider_window


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k:pytest.fail('GPU forbidden'))
    monkeypatch.setattr(bank_scope_module,'_QUARANTINE',[])
    monkeypatch.setattr(manager_module,'_QUARANTINE',[])
    monkeypatch.setattr(native,'_quarantine_upload',lambda owner:None)
    yield
    mx.set_default_device(old)


class Bank:
    generation=3
    def __init__(self,store):
        self.store=store;self.leases=0
        self.plan=SimpleNamespace(payload_bytes=8<<20,keys=frozenset({(0,1)}))
    def assert_current(self,**kwargs):assert self.store._resident_bank is self
    @contextmanager
    def lease(self,*,ownercheck):
        ownercheck();self.leases+=1
        try:yield self
        finally:self.leases-=1
    def receipt(self):return {'active_leases':self.leases,'published':True}


def fixture():
    store=object.__new__(native.NativeExpertStore)
    store._lock=threading.RLock();store._closed=store._parked=store._active=False
    store._error=store._owner=None;store._leased={};store._pending_uploads={};store._pending_host={}
    store._completion_collector=None;store.pool=SimpleNamespace(failure=None)
    store.stats={'windows':0};store._cache=OrderedDict();store._cache_bytes=0
    store.minimum_cache_bytes=1<<20;store.cache_budget_bytes=store.maximum_cache_bytes=4<<30
    store._adaptive_staging=1<<20
    weights=SimpleNamespace(expert_stream=store,source_resident_gather_policy='reference',source_resident_bank_session=None)
    model=SimpleNamespace(weights=weights,_weight_executions=0,_mlx2_ds41_ast_active=None,_mlx2_ds41_ast_poisoned=False)
    bank=Bank(store);store._resident_bank=bank;events=[]
    store._synchronize=lambda:events.append(('provider_fence',bank.leases))
    return model,store,bank,events


def tile(*,miss):
    output=mx.ones((16,8),mx.float32);status=mx.zeros(16,mx.uint32);groups=mx.zeros((384,8),mx.uint32)
    if miss:
        output[0]=float('nan');status[0]=25;groups[1,0]=1;groups[1,2]=0;groups[1,5]=25
    else:
        output[0]=float('nan');status[0]=8;groups[1,0]=1;groups[1,2]=1;groups[1,5]=8
    return SimpleNamespace(output=output,status=status,route_meta=groups,roots=(output,status,groups))


def run_private(model,store,bank,*,problem='miss',fixed=True):
    completion=BoundedSourceCompletion(store,backend=mx)
    scope=bank_scope_module.resident_scope(model.weights,lambda:True,bank=bank,backend=mx,recover_misses=True)
    logits=mx.ones((1,16,8));features=mx.ones((1,16,4))
    result=tile(miss=problem!='status')
    try:
        with scope:
            if fixed:
                with _completed_provider_window(completion.execution(),scope)as finish:
                    with completion.scope():
                        completion.note_acquire((0,1));store._leased[(0,1)]=object()
                        completion.evaluate(logits,features)
                    scope.retain(result)
                    if problem=='integrity':raise ValueError('source integrity failure')
                    finish(logits,features,(logits,features))
            else:
                # Exact historical error placement: the actual native window
                # treats ANY propagated exception as provider damage.
                with completion.execution():
                    with completion.scope():completion.evaluate(logits,features)
                    scope.retain(result);scope.complete(logits,features)
    except BaseException as error:
        error.test_scope=scope;error.test_completion=completion
        raise


def test_old_placement_reproduces_real_native_window_poison_without_fake_provider():
    model,store,bank,events=fixture();model.weights.source_resident_gather_policy='resident_bank_v1'
    with pytest.raises(bank_scope_module.ResidentCohortMiss):run_private(model,store,bank,fixed=False)
    assert store._error and not store._active and bank.leases==0
    with pytest.raises(ValueError,match='closed, parked or failed'):store._ready()


def test_expected_miss_exits_real_bounded_window_clean_before_exact_m1_retry():
    model,store,bank,events=fixture();calls=[];completed=[]
    manager=manager_module.SourceASTBankManager(model,ownercheck=lambda:True,available_bytes=lambda:2<<30)
    manager.bank=bank
    def attempt(law):
        calls.append(law)
        if law==manager_module.RESIDENT:
            try:run_private(model,store,bank)
            except bank_scope_module.ResidentCohortMiss as error:
                completed.append(error.test_completion);raise
        store._ready();events.append(('m1_ready',bank.leases,store._active,store._error))
        return SimpleNamespace(receipt={'physical_target_invocations':1,'grouped_moe_calls':40})
    result=manager.execute(model,attempt)
    assert calls==[manager_module.RESIDENT,manager_module.M1]
    assert events==[('provider_fence',1),('m1_ready',0,False,None)]
    assert completed[0].stats['final_drains']==1 and completed[0].stats['error_drains']==0
    assert result.receipt['physical_target_invocations']==2 and result.receipt['hybrid_bank']['exact_M1_recovery']
    assert model.weights._source_ast_bank_scope is None and not manager.poisoned


@pytest.mark.parametrize('problem',['status','integrity'])
def test_unknown_status_and_integrity_errors_still_poison_actual_native_window(problem):
    model,store,bank,events=fixture();model.weights.source_resident_gather_policy='resident_bank_v1'
    with pytest.raises(ValueError)as caught:run_private(model,store,bank,problem=problem)
    assert not isinstance(caught.value,bank_scope_module.ResidentCohortMiss)
    assert store._error and not store._active and bank.leases==0
    assert caught.value.test_completion.stats['error_drains']==1
    with pytest.raises(ValueError,match='closed, parked or failed'):store._ready()


def test_failed_native_final_fence_quarantines_bank_even_when_outputs_were_drained():
    model,store,bank,events=fixture();model.weights.source_resident_gather_policy='resident_bank_v1'
    def failed():events.append(('provider_fence',bank.leases));raise RuntimeError('native final fence failed')
    store._synchronize=failed
    with pytest.raises(RuntimeError,match='bank lease quarantined')as caught:run_private(model,store,bank)
    scope=caught.value.test_scope
    assert store._active and store._error=='native final fence failed'
    assert scope.poisoned and scope.active and scope.drained and bank.leases==1
    assert model.weights._source_ast_bank_scope is scope
