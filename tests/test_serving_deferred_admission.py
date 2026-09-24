"""Actual worker retries with owned warm leases and continuing active execution."""
from types import SimpleNamespace as NS
import threading
import time
import pytest


@pytest.fixture
def deferred_engine(monkeypatch):
    from mlx2 import serving, memory
    from mlx2.runtime import apc_v2, generate, os_memory
    import mlx.core as mx
    state = dict(free=21 * 2**30, lookups=[], tokenizations=[], branches=[], cycles=0,
                 recover=False, pending=threading.Event(), allocator_reclaims=0)
    class Branch(list):
        def __init__(self, large):
            super().__init__([NS(nbytes=2 * 2**30 if large else 0)])
            self.closed = 0
            state['branches'].append(self)
        def close(self): self.closed += 1
    class APC:
        def __init__(self, **kw): self.apc_stats = {}
        def key(self, *a, **kw): return 'key'
        def lookup(self, key, tokens, **kw):
            state['lookups'].append(len(tokens))
            if len(tokens) > 10: state['pending'].set()
            return NS(cache=Branch(len(tokens) > 10), cached_tokens=len(tokens)-1,
                      remaining_tokens=[2], sidecar=None, miss_reason=None)
        def store(self, *a, **kw): pass
        def spill_idle_entries(self): pass
        def evict_oldest_unleased(self): return False
        def clear(self): pass
        def __len__(self): return 0
    class Batch:
        scheduler_stats = {}
        def __init__(self, *a, **kw): self.pending = {}; self.uid = 0
        def insert(self, *a, **kw):
            uid = self.uid; self.uid += 1; self.pending[uid] = True
            return [uid]
        def next(self):
            state['cycles'] += 1
            if state['recover'] and state['pending'].is_set() and state['cycles'] >= 3:
                state['free'] = 100 * 2**30
            if state['recover'] and state['free'] < 30 * 2**30:
                return [], []
            result = [NS(uid=uid, execution_width=len(self.pending), finish_reason='length',
                token=3, mtp_state=None, all_tokens=[1,2,3], prompt_cache=[], mtp_receipt=None)
                for uid in self.pending]
            self.pending.clear()
            return [], result
        def remove(self, uids):
            for uid in uids: self.pending.pop(uid, None)
        def close(self): pass
    class Detokenizer:
        last_segment = 'ok'
        def reset(self): pass
        def add_token(self, t): pass
        def finalize(self): pass
    class Adapter:
        max_context = 2000
        identity = {'fingerprint': 'fake'}
        environment = {}; layout = 'fake'; model = None
        tokenizer = NS(vocab_size=100, eos_token_ids=[], detokenizer=Detokenizer())
        def __init__(self, path): pass
        def profile_name(self, mtp): return 'fake'
        def execution_config(self, **kw): return {'num_draft': 0}
        def prompt_tokens(self, request):
            state['tokenizations'].append(request.get('large', False))
            return [1] * (1000 if request.get('large') else 2)
        def output_parser(self, request):
            return NS(push=lambda *a, **kw: [], stopped=False, tool_count=0)
        def diagnostics(self): return {}
        def close(self): pass
    monkeypatch.setattr(serving, 'runtime_identity', lambda: {'source_sha256': 'fake'})
    monkeypatch.setattr(memory, 'execution_headroom', lambda: state['free'])
    # Admission reserves are host-scaled from installed RAM and Metal's
    # advisory (16+4 GiB on the 128 GiB calibration host, 3+~1 GiB on a
    # 36 GiB M3). The fake 21 GiB of headroom is sized against the
    # calibration reserves, so pin those readings instead of probing the
    # machine running the test.
    monkeypatch.setattr(memory, 'host_memory_gib', lambda: 128.0)
    monkeypatch.setattr(memory, 'metal_advisory_gib', lambda: 112.0)
    monkeypatch.setattr(os_memory, 'physical_footprint_bytes', lambda: 0)
    monkeypatch.setattr(apc_v2, 'APCv2', APC)
    monkeypatch.setattr(generate, 'BatchGenerator', Batch)
    monkeypatch.setattr(mx, 'synchronize', lambda: None)
    def clear_cache():
        state['allocator_reclaims'] += 1
    monkeypatch.setattr(mx, 'clear_cache', clear_cache)
    monkeypatch.setattr(serving.ServingEngine, 'MEMORY_ADMISSION_RETRY', .01)
    engine = serving.ServingEngine('fake', adapter_factory=Adapter,
        qualification_mode=True, mtp=False)
    assert engine.ready.wait(5)
    yield engine, state
    engine.close()
    assert not engine.error


def test_transient_admission_keeps_warm_lease_and_active_work_progresses(deferred_engine):
    engine, state = deferred_engine
    state['recover'] = True
    active = engine.submit({'max_tokens': 1})
    waiting = engine.submit({'max_tokens': 1, 'large': True})
    assert active.events.get(timeout=5)['finish_reason'] == 'length'
    result = waiting.events.get(timeout=5)
    assert result['finish_reason'] == 'length'
    assert result['receipt']['cached_tokens'] == 999
    assert state['cycles'] >= 4
    assert state['lookups'] == [2, 1000]  # Original lease, no repeated lookup.
    assert state['tokenizations'] == [False, True]
    assert engine.counts['memory_admission_deferred'] == 1
    assert engine.counts['memory_admission_retries'] >= 1
    assert engine.counts['memory_cache_reclaims_before_reject'] >= 1
    assert state['allocator_reclaims'] >= 1
    assert all(branch.closed == 1 for branch in state['branches'])
    assert waiting.admission_hit is None and waiting.cache_branch is None


@pytest.mark.parametrize('ending', ['timeout', 'cancel', 'shutdown'])
def test_pending_lease_released_on_every_terminal_path(deferred_engine, monkeypatch, ending):
    engine, state = deferred_engine
    monkeypatch.setattr(engine, 'MEMORY_ADMISSION_TIMEOUT', .06)
    waiting = engine.submit({'max_tokens': 1, 'large': True})
    assert state['pending'].wait(2)
    if ending == 'cancel': waiting.cancelled.set()
    elif ending == 'shutdown': engine.close()
    result = waiting.events.get(timeout=5)
    assert 'error' in result
    if ending == 'timeout':
        assert result['status'] == 429 and 'deadline' in result['error']
        assert engine.counts['memory_admission_timeouts'] == 1
        assert engine.counts['memory_cache_reclaims_before_reject'] >= 1
        assert state['allocator_reclaims'] >= 1
    elif ending == 'cancel': assert result['error'] == 'cancelled'
    else: assert result['status'] == 503
    deadline = time.monotonic() + 1
    while waiting.cache_branch is not None and time.monotonic() < deadline:
        time.sleep(.001)
    assert waiting.admission_hit is None and waiting.admission_tokens is None
    assert state['branches'][0].closed == 1
    assert state['lookups'] == [1000]
    assert state['cycles'] == 0
