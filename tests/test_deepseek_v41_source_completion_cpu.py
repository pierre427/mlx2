"""CPU dependency/ownership oracles for bounded immutable Source epochs.

Uses the real NativeExpertStore.window cleanup and LFU acquire cache-hit path.
No model payload, disk reads, GPU context or GPU synchronization is used.
"""
from collections import OrderedDict
import threading
from types import SimpleNamespace

import pytest

from mlx2.adapters.deepseek_v41_source_completion import BoundedSourceCompletion
from mlx2.adapters.deepseek_v41_mlx_weights import NativeExpertStore
from mlx2.adapters.deepseek_v41_source_lfu import LFUSourceExpertStore, LFUIndex


class FakeArray:
    def __init__(self, name, dependencies=(), *, shape=(1,), dtype='float32'):
        self.name, self.dependencies = name, tuple(dependencies)
        self.complete = False
        self.shape,self.ndim,self.dtype = shape,len(shape),dtype
        self.size = 1
        for extent in shape:self.size *= extent
        self.indices,self.reshape_shapes = [],[]
    def reshape(self, *shape):
        self.reshape_shapes.append(shape)
        return FakeArray('reshape:'+self.name, (self,),shape=shape,dtype=self.dtype)
    def __getitem__(self, key):
        self.indices.append(key)
        return FakeArray('slice:'+self.name, (self,),shape=(),dtype=self.dtype)
    def astype(self,dtype):
        return FakeArray('astype:'+self.name,(self,),shape=self.shape,dtype=dtype)


class FakeBackend:
    array = FakeArray
    float32 = 'float32'
    def __init__(self):
        self.scheduled = []
        self.evaluated = []
        self.copies = []
        self.fences = []
        self.fail_eval = self.fail_fence = self.fail_async = False
        self.stream = None
    def async_eval(self, *arrays):
        if self.fail_async:
            raise RuntimeError('injected async submission failure')
        self.scheduled.extend(arrays)
    def asarray(self, value, *, copy=False):
        assert copy is True, 'completion must use a real output-dependent copy'
        result = FakeArray('copy:'+value.name, (value,))
        self.copies.append(result)
        return result
    def concatenate(self, values, axis=0):
        return FakeArray('concatenate', tuple(values))
    def eval(self, *arrays):
        if self.fail_eval:
            raise RuntimeError('injected marker evaluation failure')
        def finish(value):
            for dependency in value.dependencies:
                finish(dependency)
            value.complete = True
        for value in arrays:
            self.evaluated.append(value)
            finish(value)
    def synchronize(self):
        # Snapshot before the real window releases any provider references.
        self.fences.append(set(self.stream._leased))
        if self.fail_fence:
            raise RuntimeError('injected provider fence failure')
        old = self.fail_eval
        self.fail_eval = False
        try:
            self.eval(*self.scheduled)
        finally:
            self.fail_eval = old


@pytest.fixture
def make_completion(monkeypatch):
    from mlx2.adapters import deepseek_v41_source_lfu as lfu
    from mlx2.adapters import deepseek_v41_mlx_weights as native
    quarantined = []
    monkeypatch.setattr(native, '_quarantine_upload', lambda owner:quarantined.append(owner))
    monkeypatch.setattr(native, '_release_upload', lambda owner:None)
    monkeypatch.setattr(lfu, 'expert_names', lambda *key:('dummy.weight',))
    monkeypatch.setattr(lfu, 'checked_file', lambda *args:None)
    def make(maximum=2):
        backend = FakeBackend()
        store = LFUSourceExpertStore.__new__(LFUSourceExpertStore)
        store._lock = threading.RLock()
        store._active = store._closed = store._parked = False
        store._owner = store._error = None
        store._synchronize = backend.synchronize
        store._cache = OrderedDict(((0,i),{'weight':object()}) for i in range(8))
        store._leased = {}
        store._pending_uploads = {};store._pending_host = {}
        store._cache_bytes = 8
        store.q1_root = None;store.hot = frozenset()
        store.root = '/fake-unopened-artifact'
        store.artifact = {'tensors':{'dummy.weight':{'file':'unopened'}},'seals':{'unopened':{}}}
        store.tables = {'blk.0':tuple((0,i) for i in range(384))}
        store.index = LFUIndex()
        store.stats = {'windows':0,'cache_hits':0,'original_hits':0}
        store.pool = SimpleNamespace(failure=None,close=lambda:None,
            counters=lambda:{'weight_read_inflight':0})
        backend.stream = store
        coordinator = BoundedSourceCompletion(store,maximum=maximum,backend=backend)
        return coordinator,store,backend,quarantined
    return make


def submit(coordinator, store, expert, *, name=None):
    with coordinator.scope():
        store.acquire('blk.0',expert)
        output = FakeArray(name or f'out{expert}')
        coordinator.evaluate(output)
    return output


@pytest.mark.parametrize('maximum', (1,2))
def test_real_window_final_drain_and_bounded_epoch_count(make_completion, maximum):
    c,s,b,_ = make_completion(maximum)
    outputs = []
    with c.execution():
        for i in range(7):
            outputs.append(submit(c,s,i))
            assert 1 <= len(c._epochs) <= maximum
            assert len(s._leased) <= maximum
        assert c.stats['max_pending_epochs'] == maximum
        assert b.fences == []
    assert all(value.complete for value in outputs)
    assert c.stats['submitted_epochs'] == c.stats['retired_epochs'] == 7
    assert c.stats['lease_releases'] == 7
    assert c.stats['final_drains'] == 1 and c.stats['error_drains'] == 0
    assert len(b.fences) == 1 and b.fences[0] == set()
    assert not c.active and not s._active and s._completion_collector is None
    assert not s._leased and not s._pending_uploads and not s._pending_host
    assert not c._epochs and not c._references


def test_repeated_lfu_key_is_collected_before_already_leased_return(make_completion):
    c,s,b,_ = make_completion()
    with c.execution():
        first = submit(c,s,0,name='first')
        cached_payload = s._leased[(0,0)]
        second = submit(c,s,0,name='second')
        assert c._references[(0,0)] == 2
        assert len(s._leased) == 1 and s.stats['cache_hits'] == 1
        with c.scope():
            # Opening epoch3 retires epoch1; epoch2 still protects the key.
            assert first.complete and not second.complete
            assert s._leased[(0,0)] is cached_payload
            assert c._references[(0,0)] == 1
            s.acquire('blk.0',1)
            c.evaluate(FakeArray('third'))
        assert c.stats['lease_releases'] == 0
    assert second.complete and c.stats['lease_releases'] == 2


def test_marker_is_copy_and_depends_on_output_producer(make_completion):
    c,s,b,_ = make_completion(1)
    producer = FakeArray('producer')
    output = FakeArray('computed-output',(producer,))
    with c.execution():
        with c.scope():
            s.acquire('blk.0',0)
            c.evaluate({'h':output})
        assert not producer.complete and not output.complete
        marker = c._epochs[0][0]
        assert marker in b.copies
        with c.scope():
            assert producer.complete and output.complete
            c.evaluate(FakeArray('next'))
    assert len(b.copies) == 2


def test_marker_covers_all_submitted_independent_roots(make_completion):
    c,s,b,_ = make_completion(1)
    tap = FakeArray('independent-tap')
    expert_consumer = FakeArray('expert-consumer')
    with c.execution():
        with c.scope():
            s.acquire('blk.0',0)
            c.evaluate([tap,{'hidden':expert_consumer}])
        with c.scope():
            assert expert_consumer.complete, 'expert lease retired before independent output completed'
            assert tap.complete
            c.evaluate(FakeArray('next'))


def test_repeated_evaluate_preserves_every_submitted_root(make_completion):
    c,s,b,_ = make_completion(1)
    first,second = FakeArray('first'),FakeArray('second')
    with c.execution():
        with c.scope():
            s.acquire('blk.0',0)
            c.evaluate(first)
            c.evaluate(second)
        with c.scope():
            assert first.complete and second.complete, 'later evaluate lost prior output dependencies'
            c.evaluate(FakeArray('next'))


def test_mixed_dtype_scalar_and_strided_roots_get_small_witnesses(make_completion):
    c,s,b,_ = make_completion(1)
    roots = [FakeArray('large-strided-cache',shape=(1,8192,64),dtype='bfloat16'),
             FakeArray('hash-index',shape=(4,),dtype='int64'),
             FakeArray('scalar',shape=(),dtype='float16')]
    with c.execution():
        with c.scope():
            s.acquire('blk.0',0)
            c.evaluate(roots)
            # Identity duplicates must not multiply marker branches.
            c.evaluate(roots[0])
            assert len(c._values) == 3
        with c.scope():
            assert all(root.complete for root in roots)
            assert roots[0].indices == [(0,0,0)]
            assert roots[1].indices == [(0,)] and roots[2].indices == []
            assert not any(root.reshape_shapes for root in roots)
            c.evaluate(FakeArray('next'))


def test_empty_root_is_skipped_but_live_branch_protects_experts(make_completion):
    c,s,b,_ = make_completion(1)
    empty = FakeArray('empty-kv',shape=(1,0,512))
    live = FakeArray('live')
    with c.execution():
        with c.scope():
            s.acquire('blk.0',0)
            c.evaluate(empty,live)
        assert empty.indices == []
        with c.scope():
            assert live.complete
            c.evaluate(FakeArray('next'))


def test_only_empty_outputs_refuse_and_fence_leases(make_completion):
    c,s,b,_ = make_completion()
    with pytest.raises(ValueError,match='empty outputs'):
        with c.execution():
            with c.scope():
                s.acquire('blk.0',0)
                c.evaluate(FakeArray('empty',shape=(0,)))
    assert b.fences == [{(0,0)}]
    assert not s._active and not s._leased and not c._epochs


def test_scope_without_expert_acquisition_and_empty_execution_allowed(make_completion):
    c,s,b,_ = make_completion()
    with c.execution():pass
    with c.execution():
        with c.scope():c.evaluate(FakeArray('embedding-no-experts'))
    assert c.stats['executions'] == c.stats['final_drains'] == 2
    assert c.stats['lease_releases'] == 0 and c.stats['submitted_epochs'] == 1
    assert len(b.fences) == 1 and not c._references


def test_validation_failure_before_weight_scope_does_not_poison_provider(make_completion):
    c,s,b,_=make_completion()
    with pytest.raises(ValueError,match='input validation'):
        with c.execution():raise ValueError('input validation')
    assert s._error is None and not s._active and not b.fences
    with c.execution():submit(c,s,0)
    assert len(b.fences)==1 and not s._leased


def test_duplicate_acquisition_within_one_scope_is_single_reference(make_completion):
    c,s,b,_ = make_completion()
    with c.execution():
        with c.scope():
            first = s.acquire('blk.0',0)
            assert s.acquire('blk.0',0) is first
            c.evaluate(FakeArray('output'))
        assert c._references[(0,0)] == 1
    assert c.stats['lease_releases'] == 1


def test_marker_submission_failure_fences_before_provider_cleanup(make_completion):
    c,s,b,_ = make_completion()
    with pytest.raises(RuntimeError,match='async submission'):
        with c.execution():
            with c.scope():
                s.acquire('blk.0',0)
                c.evaluate(FakeArray('output'))
                b.fail_async = True
    assert c.stats['submitted_epochs'] == 0
    assert b.fences == [{(0,0)}] and not s._leased


def test_final_provider_fence_failure_preserves_pending_owners(make_completion):
    c,s,b,quarantined = make_completion()
    host,upload = object(),object()
    with pytest.raises(RuntimeError,match='provider fence'):
        with c.execution():
            output = submit(c,s,0)
            s._pending_host['pending'] = host
            s._pending_uploads['pending'] = upload
            b.fail_fence = True
    assert output.complete and not c._epochs
    assert s._active and s._pending_host['pending'] is host
    assert s._pending_uploads['pending'] is upload
    assert quarantined == [s] and s._completion_collector is None
    with pytest.raises(ValueError,match='already active'):
        with c.execution():pass
    b.fail_fence = False
    NativeExpertStore.close(s)
    assert not s._active and not s._pending_host and not s._pending_uploads


def test_source_weight_hooks_use_one_outer_window_and_bounded_scopes(make_completion):
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    c,s,b,_ = make_completion()
    weights = SourceV41Weights.__new__(SourceV41Weights)
    weights.expert_stream = s
    weights.source_completion_policy = 'bounded_2';weights._source_completion = c
    weights._ready = lambda:None
    with weights.execution_scope():
        for expert in range(4):
            with weights.dense_window():
                s.acquire('blk.0',expert)
                weights.evaluate_dense_window({'h':FakeArray(f'block{expert}')})
        assert b.fences == []
    assert len(b.fences) == 1 and c.stats['submitted_epochs'] == 4
    assert c.stats['max_pending_epochs'] == 2
    assert not s._active and not s._leased


@pytest.mark.parametrize('operation', ('scope','note','evaluate'))
def test_scope_operations_outside_execution_fail_closed(make_completion,operation):
    c,s,b,_ = make_completion()
    with pytest.raises(ValueError,match='Source'):
        if operation == 'scope':
            with c.scope():pass
        elif operation == 'note':c.note_acquire((0,0))
        else:c.evaluate(FakeArray('outside'))
    assert not s._active and not c.active and b.fences == []


@pytest.mark.parametrize('empty_evaluate', (False,True))
def test_missing_output_errors_and_real_window_cleans(make_completion,empty_evaluate):
    c,s,b,_ = make_completion()
    with pytest.raises(ValueError,match='output'):
        with c.execution():
            with c.scope():
                s.acquire('blk.0',0)
                if empty_evaluate:c.evaluate(None,{'number':1},[],())
    assert b.fences == [{(0,0)}]
    assert not s._active and not s._leased and not c._epochs
    assert c.stats['error_drains'] == 1


def test_nested_execution_and_scope_refuse_without_disturbing_outer(make_completion):
    c,s,b,_ = make_completion()
    with c.execution():
        collector = s._completion_collector
        with pytest.raises(ValueError,match='already active'):
            with c.execution():pass
        assert c.active and s._completion_collector is collector
        with c.scope():
            with pytest.raises(ValueError,match='nonnested'):
                with c.scope():pass
            s.acquire('blk.0',0)
            c.evaluate(FakeArray('outer'))
    assert c.stats['executions'] == 1 and c.stats['error_drains'] == 0


class Cancelled(BaseException):pass


@pytest.mark.parametrize('error', (RuntimeError('model failure'),Cancelled('cancelled')))
@pytest.mark.parametrize('where', ('between','inside'))
def test_exception_and_cancellation_fence_before_releasing(make_completion,error,where):
    c,s,b,_ = make_completion()
    with pytest.raises(type(error),match=str(error)):
        with c.execution():
            submit(c,s,0)
            if where == 'inside':
                with c.scope():
                    s.acquire('blk.0',1)
                    raise error
            raise error
    assert b.fences == [{(0,0)} if where == 'between' else {(0,0),(0,1)}]
    assert not s._active and not s._leased and not c._epochs
    assert c.stats['error_drains'] == 1 and c.stats['final_drains'] == 0


@pytest.mark.parametrize('fence_fails', (False,True))
def test_marker_eval_failure_retains_ownership_until_provider_fence(make_completion,fence_fails):
    c,s,b,quarantined = make_completion(1)
    with pytest.raises(RuntimeError,match='provider fence' if fence_fails else 'marker evaluation'):
        with c.execution():
            submit(c,s,0)
            b.fail_eval = True;b.fail_fence = fence_fails
            with c.scope():pass
    assert b.fences == [{(0,0)}] and c.stats['lease_releases'] == 0
    assert not c.active and s._completion_collector is None
    if fence_fails:
        assert s._active and set(s._leased) == {(0,0)}
        assert len(c._epochs) == 1 and c._references[(0,0)] == 1
        assert quarantined == [s]
        with pytest.raises(ValueError,match='undrained'):
            with c.execution():pass
        b.fail_fence = b.fail_eval = False
        NativeExpertStore.close(s)
        assert not s._active and not s._leased
    else:
        assert not s._active and not s._leased and not c._epochs
        assert quarantined == []


def test_async_submission_failure_fences_live_leases(make_completion):
    c,s,b,_ = make_completion()
    with pytest.raises(RuntimeError,match='async submission'):
        with c.execution():
            with c.scope():
                s.acquire('blk.0',0)
                b.fail_async = True
                c.evaluate(FakeArray('failed-output'))
    assert b.fences == [{(0,0)}]
    assert not s._active and not s._leased and not c._epochs


def test_native_synchronous_fallback_keeps_per_scope_fence(make_completion,monkeypatch):
    import mlx.core as mx
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    c,s,b,_ = make_completion()
    weights = SourceV41Weights.__new__(SourceV41Weights)
    weights.expert_stream = s
    weights.source_completion_policy = 'reference';weights._source_completion = None
    weights._ready = lambda:None
    evaluated = []
    monkeypatch.setattr(mx,'eval',lambda values:evaluated.extend(values))
    with mx.stream(mx.cpu):
        value = mx.array([1],mx.float32)
        with weights.execution_scope():
            for _ in range(2):
                with weights.dense_window():
                    s.acquire('blk.0',0)
                    weights.evaluate_dense_window({'value':value})
                assert not s._active and not s._leased
    assert evaluated == [value,value]
    assert len(b.fences) == 2 and c.stats['submitted_epochs'] == 0


@pytest.mark.parametrize('maximum', (0,3,True,1.0))
def test_invalid_epoch_bound_refused(make_completion,maximum):
    with pytest.raises(ValueError,match='one or two'):
        make_completion(maximum)
