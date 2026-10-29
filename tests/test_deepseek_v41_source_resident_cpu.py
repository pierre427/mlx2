"""Synthetic CPU resource/transaction contracts; no checkpoint/GPU executed."""
import copy
import gc
import weakref

import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_source_buffers import PackedExpert, canonical_layout
from mlx2.adapters.deepseek_v41_source_resident import (
    ForwardTransaction, ResidentCapacityError, ResidentCapabilityError,
    ResidentResourceSet, require_serving_backend, resident_capabilities,
)
from mlx2.runtime.gguf.source_resident_probe import descriptors, kernel_spec


class Owner:
    pass


def packed(kind='original'):
    # A deliberately opaque injected resource, not an allocated coefficient bank.
    return PackedExpert(Owner(), canonical_layout(kind), {})


def install(store, key, *, kind='original', wait=None):
    expert = packed(kind)
    serial = store.stage(key, expert, host_owner=Owner(), staging_bytes=expert.layout.nbytes,
                         wait_upload=wait or (lambda expert: None))
    store.publish(key)
    return expert, serial


def store(formats=None, *, budget=200_000_000, capacity=15):
    return ResidentResourceSet(formats or {(0, 0): 'original', (0, 1): 'q1'},
                               budget_bytes=budget, max_experts=capacity)


def test_pending_upload_owns_host_and_destination_until_success():
    s=store(); expert=packed(); host=Owner(); wh=weakref.ref(host); wb=weakref.ref(expert.backing)
    class Upload:
        failed=True
        def __call__(self, expert):
            if self.failed:raise EOFError('injected short read/upload failure')
    upload=Upload()
    s.stage((0,0),expert,host_owner=host,staging_bytes=expert.layout.nbytes+128,wait_upload=upload)
    charged=2*expert.layout.nbytes+128
    del host,expert; gc.collect()
    with pytest.raises(EOFError):s.publish((0,0))
    assert wh() is not None and wb() is not None and s.live_bytes==charged
    assert s.receipt()['pending_uploads']==1 and s.generation==0
    with pytest.raises(ValueError,match='no admitted'):s.epoch(layer=0)
    upload.failed=False;s.publish((0,0));gc.collect()
    assert wh() is None and wb() is not None
    assert s.live_bytes==18_800_640 and s.generation==1


def test_all_reachable_resources_pin_even_if_route_never_uses_them():
    s=store();install(s,(0,0));install(s,(0,1),kind='q1')
    epoch=s.epoch(layer=0);root=Owner();epoch.note_roots(root)
    assert s.receipt()['pinned_resources']==2
    for key in ((0,0),(0,1)):
        with pytest.raises(ResidentCapacityError,match='pinned'):s.evict(key)
    with pytest.raises(ResidentCapacityError,match='unpinned'):s.evict_for(1)
    assert s.receipt()['resident']==2
    epoch.retire(lambda roots:None)
    assert s.evict_for(1)==((0,0),)


def test_repeated_roots_and_failed_completion_keep_all_branches_alive():
    s=store();install(s,(0,0));epoch=s.epoch(layer=0)
    a,b=Owner(),Owner();wa,wb=weakref.ref(a),weakref.ref(b)
    epoch.note_roots(a);epoch.note_roots(a,b)
    del a,b;gc.collect();seen=[]
    def failed(roots):
        seen.extend(roots);raise RuntimeError('device failure')
    with pytest.raises(RuntimeError,match='device failure'):epoch.retire(failed)
    assert len(seen)==2 and wa() is not None and wb() is not None
    assert s.receipt()['pinned_resources']==1
    seen.clear();epoch.retire(lambda roots:None);gc.collect()
    assert wa() is None and wb() is None
    with pytest.raises(RuntimeError,match='retired'):epoch.note_roots(Owner())


def test_generation_and_serial_prevent_reused_key_inheriting_old_identity():
    s=store();_,first=install(s,(0,0));old=s.generation
    e=s.epoch(layer=0);e.note_roots(Owner());e.retire(lambda roots:None)
    s.evict((0,0));_,second=install(s,(0,0))
    assert second>first
    with pytest.raises(ValueError,match='Stale'):s.epoch(layer=0,expected_generation=old)
    epoch=s.epoch(layer=0,expected_generation=s.generation)
    table,sizes=descriptors(epoch)
    assert table[0,2]==second and sizes[0]==18_800_640
    epoch.note_roots(Owner());epoch.retire(lambda roots:None)


def test_capacity_counts_full_staging_not_only_payload():
    s=store(budget=2*18_800_640-1)
    with pytest.raises(ResidentCapacityError,match='staging'):
        s.stage((0,0),packed(),host_owner=Owner(),staging_bytes=18_800_640,
                wait_upload=lambda expert:None)
    assert s.live_bytes==0


def test_fixed_precision_map_and_binding_limit_are_not_relaxed():
    s=store(capacity=1);install(s,(0,0))
    with pytest.raises(ValueError,match='precision'):
        s.stage((0,1),packed('original'),host_owner=Owner(),staging_bytes=18_800_640,
                wait_upload=lambda expert:None)
    with pytest.raises(ResidentCapacityError,match='binding'):
        s.stage((0,1),packed('q1'),host_owner=Owner(),staging_bytes=6_635_520,
                wait_upload=lambda expert:None)
    assert s.receipt()['resident']==1


def test_eviction_preflight_does_not_partially_mutate_on_insufficient_victims():
    s=store();install(s,(0,0));install(s,(0,1),kind='q1');generation=s.generation
    with pytest.raises(ResidentCapacityError,match='unpinned'):s.evict_for(200_000_000)
    assert s.receipt()['resident']==2 and s.generation==generation


def test_two_independent_epochs_release_only_their_completion_owner():
    s=store();install(s,(0,0));a,b=s.epoch(layer=0),s.epoch(layer=0)
    a.note_roots(Owner());b.note_roots(Owner());a.retire(lambda roots:None)
    with pytest.raises(ResidentCapacityError):s.evict((0,0))
    b.retire(lambda roots:None);s.evict((0,0))
    assert s.live_bytes==0


@pytest.mark.parametrize('key',[(True,0),(40,0),(0,384),(-1,0),(0,)])
def test_bad_expert_identity_refused(key):
    with pytest.raises(ValueError):ResidentResourceSet({key:'q1'},budget_bytes=1)


def test_mutable_key_and_boolean_generation_are_refused():
    s=store();install(s,(0,0))
    with pytest.raises(ValueError):s.evict([0,0])
    with pytest.raises(ValueError):s.epoch(layer=0,expected_generation=True)


@pytest.mark.parametrize('budget,capacity',[(0,1),(True,1),(1,0),(1,16),(1,True)])
def test_bad_capacity_refused(budget,capacity):
    with pytest.raises(ValueError):store(budget=budget,capacity=capacity)


def test_descriptor_table_is_dense_readonly_and_does_not_expose_addresses():
    s=store({(0,383):'original',(0,2):'q1'});install(s,(0,383));install(s,(0,2),kind='q1')
    e=s.epoch(layer=0);table,sizes=descriptors(e)
    assert table.shape==(384,4) and table.dtype==np.uint32
    assert table[2,:2].tolist()==[1,2] and table[383,:2].tolist()==[2,1]
    assert np.flatnonzero(table[:,0]).tolist()==[2,383]
    assert sizes.tolist()==[6_635_520,18_800_640]
    assert not table.flags.writeable and not sizes.flags.writeable
    e.note_roots(Owner());e.retire(lambda roots:None)
    with pytest.raises(RuntimeError):descriptors(e)


@pytest.mark.parametrize('count',[1,6,15])
def test_uncompiled_probe_declares_every_reachable_backing_and_guards_before_load(count):
    spec=kernel_spec(count)
    assert spec['input_names'][-count:]==[f'expert{i}'for i in range(count)]
    source=spec['source']
    assert source.index('id>=384')<source.index('table[id*4]')
    assert source.index('encoded==0')<source.index('backing[0]')
    assert source.index('generation[0]!=')<source.index('backing[0]')
    assert 'reinterpret_cast'not in source and 'ulong'not in source
    assert spec['output_names']==['probe','status']


@pytest.mark.parametrize('count',[0,16,True,2.5])
def test_unbounded_probe_refused(count):
    with pytest.raises(ValueError):kernel_spec(count)


def test_pilot_cannot_be_selected_as_production_backend():
    with pytest.raises(ResidentCapabilityError,match='no maintained dynamic-resource'):
        require_serving_backend()
    r=store().receipt()
    assert not r['selected']and not r['qualified']and not r['observed_used']
    capabilities=resident_capabilities()
    assert capabilities['component_pilot_implemented']
    assert not capabilities['serving_backend_available']
    assert not capabilities['component_pilot_gpu_observed']


def transaction(published, *, retries=1, commit=None, discard=None):
    def default_commit(dst,src,expected_revision):
        # Test backend atomically replaces its logical plane.
        assert dst.get('revision',tuple(dst['state']))==expected_revision
        dst['state']=src['state'];return 'committed'
    return ForwardTransaction(published,fork=copy.deepcopy,validate_fork=_isolated,
                              revision=lambda state:state.get('revision',tuple(state['state'])),commit=commit or default_commit,
                              discard=discard or (lambda state:None),max_replays=retries)


def _isolated(left,right):
    def owners(value):
        if isinstance(value,dict):
            yield id(value)
            for child in value.values():yield from owners(child)
        elif isinstance(value,list):
            yield id(value)
            for child in value:yield from owners(child)
    return not set(owners(left)).intersection(owners(right))


def test_miss_discards_private_state_and_replay_starts_from_same_frontier():
    published={'state':[1,2]};tx=transaction(published)
    working=tx.begin();working['state'].append(3)
    assert published=={'state':[1,2]}
    assert tx.finish(status='miss',wait=lambda state:None,missing=((0,5),))=='ordinary_replay_required'
    working=tx.begin();assert working=={'state':[1,2]}
    working['state'].append(4)
    assert tx.finish(status='ok',wait=lambda state:None)=='committed'
    assert published=={'state':[1,2,4]}


@pytest.mark.parametrize('status',['cancelled','error','stale','miss'])
def test_failure_never_publishes_private_partial_state(status):
    published={'state':[1]};tx=transaction(published,retries=0);tx.begin()['state'].append(99)
    assert tx.finish(status=status,wait=lambda state:None)=='refused_without_publication'
    assert published=={'state':[1]}
    with pytest.raises(RuntimeError):tx.begin()


def test_failed_completion_retains_private_state_until_explicit_drain_retry():
    published={'state':[1]};discarded=[];tx=transaction(published,discard=discarded.append)
    working=tx.begin();working['state'].append(9)
    def failed(state):raise RuntimeError('injected pending GPU failure')
    with pytest.raises(RuntimeError):tx.finish(status='cancelled',wait=failed)
    assert tx.working is working and not discarded and published=={'state':[1]}
    tx.finish(status='cancelled',wait=lambda state:None)
    assert discarded==[working]and published=={'state':[1]}


def test_retries_are_bounded_and_failed_commit_keeps_private_owner():
    published={'state':[1]};tx=transaction(published)
    tx.begin();assert tx.finish(status='miss',wait=lambda state:None)=='ordinary_replay_required'
    tx.begin();assert tx.finish(status='miss',wait=lambda state:None)=='refused_without_publication'
    def failed(dst,src,revision):raise RuntimeError('commit refused before mutation')
    tx=transaction(published,commit=failed);working=tx.begin()
    with pytest.raises(RuntimeError):tx.finish(status='ok',wait=lambda state:None)
    assert tx.working is working and published=={'state':[1]}


def test_aliasing_fork_and_successful_missing_status_are_refused():
    p={'state':[1]};tx=ForwardTransaction(p,fork=lambda x:x,validate_fork=_isolated,
            revision=lambda state:0,commit=lambda *a:None,discard=lambda *a:None)
    with pytest.raises(ValueError,match='private'):tx.begin()
    tx=transaction(p);tx.begin()
    with pytest.raises(ValueError,match='missing'):tx.finish(status='ok',wait=lambda state:None,missing=((0,0),))
    assert p=={'state':[1]}


def test_publication_does_not_alias_callers_working_state():
    published={'state':[1],'revision':0};tx=transaction(published)
    working=tx.begin();working['state'].append(2)
    tx.finish(status='ok',wait=lambda state:None)
    working['state'].append(999)
    assert published['state']==[1,2]


def test_shallow_fork_is_refused_and_retained_for_drain():
    published={'state':[1]};discarded=[];tx=ForwardTransaction(published,fork=copy.copy,
        validate_fork=_isolated,revision=lambda state:0,commit=lambda *a:None,discard=discarded.append)
    with pytest.raises(ValueError,match='private'):tx.begin()
    assert tx.working is not None
    assert tx.finish(status='cancelled',wait=lambda state:None)=='refused_without_publication'
    assert not discarded and published=={'state':[1]}
    assert tx.receipt()['quarantined_invalid_forks']==1


def test_revision_change_during_completion_refuses_publication():
    published={'state':[1],'revision':7};tx=transaction(published);tx.begin()['state'].append(2)
    def moved(state):published['revision']=8
    with pytest.raises(ValueError,match='revision changed'):tx.finish(status='ok',wait=moved)
    assert published=={'state':[1],'revision':8}and tx.working is not None
    tx.finish(status='cancelled',wait=lambda state:None)


def test_failed_publication_copy_wait_retains_both_owners_until_cancel_drain():
    published={'state':[1]};discarded=[];tx=transaction(published,discard=discarded.append)
    working=tx.begin();working['state'].append(2);calls=[]
    def pending(state):
        calls.append(state)
        if state is not working:raise TimeoutError('copy incomplete')
    with pytest.raises(TimeoutError):tx.finish(status='ok',wait=pending)
    candidate=tx._candidate
    assert candidate is not None and tx.working is working and not discarded
    assert published=={'state':[1]}
    tx.finish(status='cancelled',wait=lambda state:None)
    assert len(discarded)==2 and any(v is candidate for v in discarded)


def test_authoritative_change_during_publication_copy_wait_also_refuses():
    published={'state':[1],'revision':1};tx=transaction(published);working=tx.begin()
    def moved(state):
        if state is not working:published['revision']=2
    with pytest.raises(ValueError,match='before publication'):tx.finish(status='ok',wait=moved)
    assert published=={'state':[1],'revision':2}and tx._candidate is not None
    tx.finish(status='cancelled',wait=lambda state:None)


def test_pending_epochs_are_bounded_and_failed_wait_does_not_restore_capacity():
    s=store();install(s,(0,0));a,b=s.epoch(layer=0),s.epoch(layer=0)
    a.note_roots(Owner());b.note_roots(Owner())
    with pytest.raises(ResidentCapacityError,match='epoch bound'):s.epoch(layer=0)
    def failed(roots):raise TimeoutError('still encoded')
    with pytest.raises(TimeoutError):a.retire(failed)
    with pytest.raises(ResidentCapacityError,match='epoch bound'):s.epoch(layer=0)
    a.retire(lambda roots:None)
    c=s.epoch(layer=0);c.note_roots(Owner())
    b.retire(lambda roots:None);c.retire(lambda roots:None)
    assert s.receipt()['active_epochs']==0


def test_mutable_authoritative_revision_refused_before_fork():
    p={'state':[1]};tx=ForwardTransaction(p,fork=copy.deepcopy,validate_fork=_isolated,
        revision=lambda state:[1],commit=lambda *a:None,discard=lambda *a:None)
    with pytest.raises(ValueError,match='immutable'):tx.begin()
    assert tx.working is None


def test_rejected_publication_alias_cannot_reach_discard_hook():
    published={'state':[1]};forks=[];discarded=[]
    def fork(state):
        forks.append(state)
        return copy.deepcopy(state)if len(forks)==1 else published
    tx=ForwardTransaction(published,fork=fork,validate_fork=_isolated,
        revision=lambda state:0,commit=lambda *a:None,discard=discarded.append)
    working=tx.begin()
    with pytest.raises(ValueError,match='independently'):tx.finish(status='ok',wait=lambda state:None)
    tx.finish(status='cancelled',wait=lambda state:None)
    assert len(discarded)==1 and discarded[0]is working
    assert all(owner is not published for owner in discarded)
    assert tx.receipt()['quarantined_invalid_forks']==1
