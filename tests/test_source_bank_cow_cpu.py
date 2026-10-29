"""Exact synthetic Source cache COW ownership on mx.cpu; GPU kernels forbidden."""
from types import SimpleNamespace
import mlx.core as mx
import numpy as np
import pytest
from mlx2.adapters.deepseek_v41_source_bank_session import PreparedBankSession,_fork,_isolated
from test_source_bank_session_cpu import Bank
from test_deepseek_v41_cache_frontiers_cpu import row,advance

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:pytest.fail('GPU forbidden'))
    yield
    mx.set_default_device(old)

def state():
    result=advance(row(mx),100);mx.eval(result.state)
    assert result._capacity==256 and result._pool_count==50
    return result

def capture(cache):
    return [None if v is None else np.array(v)for v in cache.state]

def exact(cache,values):
    for actual,expected in zip(cache.state,values):
        assert (actual is None)==(expected is None)
        if actual is not None:assert np.array_equal(np.array(actual).view(np.uint8),expected.view(np.uint8))

def session_for(cache,*,cancelled=None,backend=mx):
    weights=SimpleNamespace(source_resident_gather_policy='resident_bank_v1');bank=Bank()
    session=PreparedBankSession(weights,bank,ownercheck=lambda:True,cancelled=cancelled,backend=backend)
    assert session.cache_fork_policy=='physical_copy_v1'
    session.set_cache_fork_policy('descriptor_cow_v1');return session,bank

def test_descriptor_fork_preserves_full_capacity_and_source_on_private_append():
    source=state();before=capture(source);raw=np.array(source._pooled)
    clone=_fork([source],'descriptor_cow_v1')[0]
    assert _isolated([source],[clone])and clone._capacity==256 and clone._pooled.shape==source._pooled.shape
    advance(clone,102);mx.eval(clone.state)
    assert source.offset==100 and clone.offset==102 and clone._capacity==256
    exact(source,before);assert np.array_equal(np.array(source._pooled),raw)
    # Existing physical control serializes only live pooled rows, losing slack.
    physical=_fork([source],'physical_copy_v1')[0];assert physical._capacity==50
    advance(physical,102);mx.eval(physical.state);exact(clone,capture(physical))

def test_cow_success_publication_owns_independent_wrappers_after_working_mutation():
    cache=[state()];session,bank=session_for(cache);refs=[]
    def run(private):
        refs.append(private);advance(private[0],102)
        session.statuses.append(mx.array([0],mx.uint32));return(mx.array([[4.]],mx.float32),None)
    session.forward(None,None,cache,run)
    assert cache[0].offset==102 and cache[0]._capacity==256 and bank.leases==0
    published=capture(cache[0]);private=refs[0][0]
    private._pooled[:,0]=99;private.keys[:]=88;mx.eval(private.state)
    exact(cache[0],published)
    assert session.stats['publication_descriptor_evaluations']==1 and session.stats['publication_copy_evaluations']==0

def test_cow_miss_replay_sees_exact_original_prefix():
    cache=[state()];before=capture(cache[0]);session,bank=session_for(cache);seen=[]
    def run(private):
        seen.append((session.replaying,private[0].offset));exact(private[0],before)
        advance(private[0],102)
        if session.active:session.statuses.append(mx.array([1],mx.uint32))
        return(mx.array([[4.]],mx.float32),None)
    session.forward(None,None,cache,run)
    assert seen==[(False,100),(True,100)]and cache[0].offset==102 and bank.leases==0
    assert session.stats['ordinary_replays']==1 and cache[0]._capacity==256

def test_cow_cancellation_keeps_all_published_planes_and_capacity():
    cache=[state()];before=capture(cache[0]);checks=[0]
    def cancelled():checks[0]+=1;return checks[0]>1
    session,bank=session_for(cache,cancelled=cancelled)
    def run(private):
        advance(private[0],102);session.statuses.append(mx.array([0],mx.uint32));return(mx.array([[4.]],mx.float32),None)
    with pytest.raises(InterruptedError):session.forward(None,None,cache,run)
    assert cache[0].offset==100 and cache[0]._capacity==256 and bank.leases==0;exact(cache[0],before)

def test_cow_failed_drain_retains_alias_roots_and_bank_epoch():
    cache=[state()];before=capture(cache[0])
    class Failed:
        any=staticmethod(mx.any);all=staticmethod(mx.all);stack=staticmethod(mx.stack);isfinite=staticmethod(mx.isfinite)
        def eval(self,*roots):raise RuntimeError('failed drain')
    session,bank=session_for(cache,backend=Failed())
    def run(private):
        advance(private[0],102);session.statuses.append(mx.array([0],mx.uint32));return(mx.array([[4.]],mx.float32),None)
    with pytest.raises(RuntimeError,match='failed drain'):session.forward(None,None,cache,run)
    assert session.poisoned and bank.leases==1 and session._quarantined[0][2].working is not None
    assert cache[0].offset==100 and cache[0]._capacity==256;exact(cache[0],before)
    session._quarantined[0][0].__exit__(None,None,None)

def test_numpy_cow_policy_remains_physical_copy():
    source=advance(row(np),100);clone=_fork([source],'descriptor_cow_v1')[0]
    assert not np.shares_memory(source._pooled,clone._pooled)
    advance(clone,102);assert source.offset==100

def test_fork_variant_change_requires_idle_healthy_epoch():
    cache=[state()];session,bank=session_for(cache);session.active=True
    with pytest.raises(RuntimeError,match='idle healthy'):session.set_cache_fork_policy('physical_copy_v1')
    session.active=False
    with pytest.raises(ValueError,match='Invalid'):session.set_cache_fork_policy('unknown')
