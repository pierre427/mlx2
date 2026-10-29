"""Host-only Source APC resource metadata; no GPU/checkpoint is loaded."""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
import threading
import numpy as np
import pytest
from mlx2.adapters import deepseek_v41_source_checkpoint_resources as helper
from mlx2.adapters.deepseek_v41_source_buffers import ORIGINAL_BYTES,Q1_BYTES
from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
from mlx2.runtime.expert_residency_hints import MAX_HINT_EXPERTS

class Store:
    def __init__(self,layers=2,experts=3):
        self.tables={'blk.'+str(layer):tuple((layer,e)for e in range(experts))for layer in range(layers)}
        self._cache=OrderedDict();self._cache_bytes=0;self._lock=threading.RLock()
        self._active=False;self._leased={};self._pending_uploads={};self._pending_host={};self._completion_collector=None
        self._resident_bank=None;self._resident_bank_preparing=None;self._resident_bank_replacing=None
        self._closed=self._parked=False;self._error=None;self._owner=None
        self.binding='source';self.q1_binding='q1';self.hot_binding='hot'
        self.minimum_cache_bytes=ORIGINAL_BYTES;self._adaptive_staging=ORIGINAL_BYTES;self._adaptive_working_set=1
        self.cache_budget_bytes=self.maximum_cache_bytes=512<<20
        self.pool=SimpleNamespace(failure=None,counters=lambda:{})
        self.windows=0;self.loads=[];self.fail_allocation=False;self.fail_fence=False
    def kind(self,key):return 'q1'if key[1]%2 else'original'
    def add(self,key):
        size=Q1_BYTES if self.kind(key)=='q1'else ORIGINAL_BYTES
        if key not in self._cache:self._cache_bytes+=size
        self._cache[key]=SimpleNamespace(allocation_nbytes=size)
    def drop(self,key):self._cache_bytes-=self._cache.pop(key).allocation_nbytes
    @contextmanager
    def window(self):
        self.windows+=1;self._active=True
        try:yield
        except BaseException as error:self._error=str(error);raise
        finally:
            if self.fail_fence:raise RuntimeError('unsafe completion')
            self._leased.clear();self._pending_uploads.clear();self._pending_host.clear();self._active=False
    def acquire(self,prefix,index):
        key=self.tables[prefix][index];self.loads.append(key)
        if self.fail_allocation:raise MemoryError('actual optional upload allocation')
        self.add(key)


def fixture(*,layers=2,experts=3,tokens=(1,2)):
    store=Store(layers,experts);weights=SimpleNamespace(expert_stream=store,artifact={'identity':{'fingerprint':'checkpoint@fingerprint'}},
        expert_math_policy='routed_fused_fp32',source_resident_bank_session=None)
    model=SimpleNamespace(identity='source-model',weights=weights,args=SimpleNamespace(num_hidden_layers=layers),_weight_executions=0)
    adapter=SimpleNamespace(model=model);cache=[]
    for _ in range(layers):
        row=DeepSeekV41LayerCache(0,xp=np,identity=model.identity,head_dim=2,index_head_dim=2)
        row.offset=len(tokens);row.keys=np.zeros((1,1,len(tokens),2),np.float32);cache.append(row)
    return adapter,store,cache,list(tokens)


def published(keys=((0,0),(0,1),(1,2))):
    adapter,store,cache,tokens=fixture()
    for key in keys:store.add(key)
    hint=helper.snapshot(adapter,cache,tokens)
    return adapter,store,cache,tokens,hint


def test_inventory_is_advisory_bound_and_does_not_read_gpu_state_or_claim_routes(monkeypatch):
    adapter,store,cache,tokens,hint=published()
    assert hint.phase=='resident_snapshot'and hint.coverage=='advisory'and not hint.truncated
    assert hint.experts==((0,0),(0,1),(1,2))and store.windows==0
    for row in cache:
        monkeypatch.setattr(row,'validate_state',lambda:pytest.fail('GPU tail/state validation forbidden'))
    assert helper.snapshot(adapter,cache,tokens)==hint
    assert helper.snapshot(adapter,cache,tokens+[3])is None


def test_bounded_inventory_marks_truncation_and_has_no_resource_owners():
    adapter,store,cache,tokens=fixture(layers=40,experts=384)
    for key in list(helper._registry(store))[:MAX_HINT_EXPERTS+5]:store.add(key)
    hint=helper.snapshot(adapter,cache,tokens)
    assert len(hint.experts)==MAX_HINT_EXPERTS and hint.truncated and hint.nbytes<=64<<10
    assert not any('path'in key or 'buffer'in key for key in hint.to_manifest())


@pytest.mark.parametrize('change',[{'source_revision':'old'},{'quantization':'old'},{'layout':'old'},{'resource_abi':'old'},{'checkpoint':'foreign'}])
def test_stale_resource_binding_drops_only_hint_and_does_no_io(change):
    adapter,store,cache,tokens,hint=published();before=store._cache_bytes
    report=helper.restore(adapter,replace(hint,**change),cache,available_bytes=lambda:1<<30)
    assert not report['hint_hit']and not store.windows and store._cache_bytes==before
    assert all(row.offset==len(tokens)for row in cache)


def test_unknown_snapshot_revision_and_private_or_foreign_cache_decline():
    adapter,store,cache,tokens,hint=published()
    assert not helper.restore(adapter,replace(hint,snapshot_revision='unknown'),cache)['hint_hit']
    cache[0].identity='foreign'
    assert not helper.restore(adapter,hint,cache)['hint_hit']
    cache[0].identity=adapter.model.identity;cache[0].ast_private_profile='private'
    assert helper.snapshot(adapter,cache,tokens)is None


def test_bounded_partial_prefetch_remeasures_actual_availability_before_each_load():
    adapter,store,cache,tokens,hint=published();store.drop((0,1));store.drop((1,2))
    availability=iter((128<<20,0));report=helper.restore(adapter,hint,cache,available_bytes=lambda:next(availability))
    assert report['hint_hit']and report['prefetched']==1 and report['resource_resident']==2
    assert store.loads==[(0,1)]and store.windows==1 and not report['bank_constructed']
    assert report['reason']=='actual_availability_or_operator_working_headroom'
    assert report['reservation_bytes']==0 and report['precision_unchanged']


@pytest.mark.parametrize('available',[None,0,1])
def test_pressure_or_unknown_availability_preserves_kv_and_normal_lazy_path(available):
    adapter,store,cache,tokens,hint=published();store.drop((0,1));before=store._cache_bytes
    report=helper.restore(adapter,hint,cache,available_bytes=lambda:available)
    assert report['hint_hit']and report['prefetched']==0 and not store.windows
    assert store._cache_bytes==before and store._error is None and all(row.offset==2 for row in cache)


def test_operator_maximum_and_active_or_pinned_owners_never_get_bypassed():
    adapter,store,cache,tokens,hint=published();store.drop((0,1))
    store.cache_budget_bytes=store._cache_bytes
    assert helper.restore(adapter,hint,cache,available_bytes=lambda:1<<30)['prefetched']==0
    store._resident_bank=object()
    assert helper.restore(adapter,hint,cache,available_bytes=lambda:1<<30)['reason']=='existing_pinned_bank_preserved'
    store._active=True
    assert helper.restore(adapter,hint,cache,available_bytes=lambda:1<<30)['reason']=='expert_resources_active'
    assert not store.windows


def test_cancel_declines_before_any_window_and_shorter_inventory_keeps_origin_identity():
    adapter,store,cache,tokens,hint=published();store.drop((0,1))
    assert helper.restore(adapter,hint,cache,available_bytes=lambda:1<<30,cancelled=lambda:True)['prefetched']==0
    assert not store.windows
    for row in cache:row.offset=1;row.keys=row.keys[:,:,:1]
    report=helper.restore(adapter,hint,cache,available_bytes=lambda:0)
    assert report['hint_hit']and report['coverage']=='advisory'and hint.covered_tokens==2


def test_completed_optional_allocation_failure_does_not_poison_lazy_provider():
    adapter,store,cache,tokens,hint=published();store.drop((0,1));store.fail_allocation=True
    report=helper.restore(adapter,hint,cache,available_bytes=lambda:1<<30)
    assert report['reason']=='completed_optional_allocation_declined'and store._error is None
    store.fail_allocation=False
    with store.window():store.acquire('blk.0',1)
    assert (0,1)in store._cache


def test_failed_completion_is_never_cleared_or_claimed_as_safe_pressure():
    adapter,store,cache,tokens,hint=published();store.drop((0,1));store.fail_allocation=store.fail_fence=True
    with pytest.raises(RuntimeError,match='unsafe completion'):
        helper.restore(adapter,hint,cache,available_bytes=lambda:1<<30)
    assert store._active and store._error is not None


@pytest.mark.parametrize('owner',['lease','pending_upload','manager'])
def test_optional_restore_respects_existing_lease_upload_and_manager_ownership(owner):
    adapter,store,cache,tokens,hint=published();store.drop((0,1))
    bound=dict(helper.binding(adapter))
    if owner=='lease':store._leased[(0,0)]=store._cache[(0,0)]
    elif owner=='pending_upload':store._pending_uploads[(0,1)]={}
    else:
        adapter.model._source_ast_bank_manager=SimpleNamespace(active=True,poisoned=False,closed=False,
            hint_binding=lambda:bound,check_live=lambda:pytest.fail('active manager must not be touched'))
    report=helper.restore(adapter,hint,cache,available_bytes=lambda:1<<30)
    assert report['hint_hit']and report['prefetched']==0 and not store.windows


def test_real_source_adapter_delegates_attach_and_restore_through_generic_optional_hooks(monkeypatch):
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    from mlx2.runtime.checkpoint_resources import publication_hints,restore_resources
    adapter,store,cache,tokens=fixture();store.add((0,0));store.add((0,1))
    owner=object.__new__(DeepSeekV41SourceAdapter);owner.model=adapter.model
    hint,receipt=publication_hints(owner,cache,tokens)
    assert hint is not None and receipt['action']=='attached'and receipt['coverage']=='advisory'
    store.drop((0,1));monkeypatch.setattr(helper,'_available',lambda callback:0)
    lookup=SimpleNamespace(hit=True,cache=cache,cached_tokens=len(tokens),expert_hints=hint)
    restored=restore_resources(owner,lookup,tokens)
    assert restored['action']=='adapter'and restored['numeric_hit_unchanged']
    assert restored['resource']['hint_hit']and restored['resource']['resource_resident']==1
    assert restored['resource']['prefetched']==0 and lookup.hit and not store.windows


def test_resource_binding_changes_with_helper_semantics_and_live_numerical_profile(monkeypatch):
    adapter,store,cache,tokens=fixture()
    monkeypatch.setattr(helper,'implementation_identity',lambda:'helper-v1')
    first=helper.binding(adapter)
    monkeypatch.setattr(helper,'implementation_identity',lambda:'helper-v2')
    assert helper.binding(adapter)!=first
    second=helper.binding(adapter);adapter.model.identity='another-live-numerical-profile'
    assert helper.binding(adapter)!=second


def test_manager_binding_closure_contains_helper_and_hint_abi_and_separates_live_profiles(monkeypatch):
    import hashlib
    from pathlib import Path
    from mlx2.adapters import deepseek_v41_ast_target as target
    from mlx2.adapters.deepseek_v41_ast_bank_manager import SourceASTBankManager
    source=dict(target.implementation_identity());root=Path(__file__).parents[1]
    for name in ('src/mlx2/adapters/deepseek_v41_source_checkpoint_resources.py','src/mlx2/runtime/expert_residency_hints.py'):
        assert source[name]==hashlib.sha256((root/name).read_bytes()).hexdigest()
    monkeypatch.setattr(target,'source_revision_token',lambda law,forward:'immutable-code-closure')
    adapter,store,cache,tokens=fixture();manager=SourceASTBankManager(adapter.model,ownercheck=lambda:True)
    adapter.model._source_ast_bank_manager=manager
    first=helper.binding(adapter);adapter.model.identity='another-profile'
    assert helper.binding(adapter)!=first
