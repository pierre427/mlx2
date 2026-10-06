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
    state = dict(free=21 * 2**30, lookups=[], stores=[], tokenizations=[],
                 branches=[], cycles=0,
                 recover=False, pending=threading.Event(), allocator_reclaims=0,
                 cold=False, native_uids=set(), native_jobs={},
                 native_allocations=[], native_retired=[], native_fail=False,
                 native_hold=False, native_waiting=threading.Event(),
                 native_release=threading.Event())
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
            if state['cold']:
                return NS(cache=None, cached_tokens=0,
                          remaining_tokens=list(tokens), sidecar=None,
                          miss_reason='cold')
            return NS(cache=Branch(len(tokens) > 10), cached_tokens=len(tokens)-1,
                      remaining_tokens=[2], sidecar=None, miss_reason=None)
        def store(self, *a, **kw): state['stores'].append((a, kw))
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
            result = []
            self.failures = []
            for uid in tuple(self.pending):
                native = uid in state['native_uids']
                if native and state['native_hold']:
                    state['native_waiting'].set()
                    state['native_release'].wait(2)
                    if state['native_jobs'][uid].cancelled.is_set():
                        continue
                if native and state['native_fail']:
                    self.failures.append({'uid': uid, 'reason': 'injected native terminal failure'})
                    state['native_retired'].append(uid)
                    state['native_uids'].discard(uid)
                    continue
                result.append(NS(uid=uid, execution_width=len(self.pending),
                    finish_reason='length', token=3, mtp_state=None,
                    all_tokens=[1,2,3], prompt_cache=None if native else [],
                    mtp_receipt=({'route': 'native_qwen3_paged',
                        'implemented': True, 'qualified': False,
                        'selected': True, 'observed_used': True,
                        'apcv2': 'native_checkpoint_unavailable'} if native else None)))
                if native:
                    state['native_retired'].append(uid)
                    state['native_uids'].discard(uid)
            self.pending.clear()
            return [], result
        def remove(self, uids):
            for uid in uids:
                self.pending.pop(uid, None)
                if uid in state['native_uids']:
                    state['native_uids'].remove(uid)
                    state['native_retired'].append(uid)
        def take_lane_failures(self):
            failures = getattr(self, 'failures', [])
            self.failures = []
            return failures
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


def test_native_http_worker_cold_policy_receipt_and_failure_retirement(
        deferred_engine, monkeypatch):
    """Exercise real HTTP and the serving worker with a host-only native stand-in."""
    import json
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    from mlx2 import serving
    from mlx2.server import handler_for

    engine, state = deferred_engine
    state['cold'] = True
    original = serving.install_explicit_native_qwen3_request

    def install(batch, adapter, job, *, prompt_tokens, maximum,
                lifecycle_lock, **paths):
        if job.cached_tokens or job.request.get('skip_writing_prefix_cache') is not True:
            # Run the production refusal before the stand-in can create state.
            return original(batch, adapter, job, prompt_tokens=prompt_tokens,
                            maximum=maximum, lifecycle_lock=lifecycle_lock,
                            **paths)
        state['native_uids'].add(job.uid)
        state['native_jobs'][job.uid] = job
        state['native_allocations'].append(job.uid)
        return {'route': 'native_qwen3_paged', 'implemented': True,
                'qualified': False, 'selected': True, 'observed_used': False}

    monkeypatch.setattr(serving, 'install_explicit_native_qwen3_request', install)
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'

    def post(**options):
        body = {'model': 'fake', 'messages': [{'role': 'user', 'content': 'hi'}],
                'max_tokens': 1, **options}
        return urlopen(Request(base + '/v1/chat/completions',
                               data=json.dumps(body).encode(),
                               headers={'Content-Type': 'application/json'}),
                       timeout=5)

    try:
        with post() as response:
            ordinary = json.load(response)
        assert ordinary['mlx2']['route'] != 'native_qwen3_paged'
        assert not state['native_retired']

        with pytest.raises(HTTPError) as write_refusal:
            post(paged_native_qwen3=True, temperature=0)
        assert write_refusal.value.code == 400
        assert 'skip_writing_prefix_cache' in json.load(write_refusal.value)['error']['message']

        stores_before_native = len(state['stores'])
        with post(paged_native_qwen3=True, skip_writing_prefix_cache=True,
                  temperature=0) as response:
            cold = json.load(response)
        route = cold['mlx2']['route_receipt']
        assert cold['mlx2']['cached_tokens'] == 0
        assert cold['mlx2']['route'] == 'native_qwen3_paged'
        assert route['implemented'] and route['selected'] and route['observed_used']
        assert route['qualified'] is False
        assert route['apcv2'] == 'native_checkpoint_unavailable'
        assert cold['mlx2']['request_controls']['sampling'] == {'temperature': 0}
        assert len(state['stores']) == stores_before_native
        assert len(state['native_retired']) == 1 and not state['native_uids']
        assert len(state['native_allocations']) == 1

        state['cold'] = False
        with pytest.raises(HTTPError) as warm_refusal:
            post(paged_native_qwen3=True, skip_writing_prefix_cache=True,
                 temperature=0)
        assert warm_refusal.value.code == 400
        assert 'pinned price/source paths' in json.load(warm_refusal.value)['error']['message']
        assert len(state['native_retired']) == 1 and not state['native_uids']
        assert len(state['native_allocations']) == 1

        state['cold'] = True
        state['native_fail'] = True
        with pytest.raises(HTTPError) as failed:
            post(paged_native_qwen3=True, skip_writing_prefix_cache=True,
                 temperature=0)
        assert failed.value.code == 500
        assert 'injected native terminal failure' in json.load(failed.value)['error']['message']
        assert len(state['native_retired']) == 2 and not state['native_uids']
        with post() as response:
            assert json.load(response)['choices'][0]['finish_reason'] == 'length'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_native_live_worker_cancellation_retires_private_lane(
        deferred_engine, monkeypatch):
    from mlx2 import serving

    engine, state = deferred_engine
    state['cold'] = True
    state['native_hold'] = True

    def install(batch, adapter, job, **kwargs):
        state['native_uids'].add(job.uid)
        state['native_jobs'][job.uid] = job
        state['native_allocations'].append(job.uid)
        return {'route': 'native_qwen3_paged', 'implemented': True,
                'qualified': False, 'selected': True, 'observed_used': False}

    monkeypatch.setattr(serving, 'install_explicit_native_qwen3_request', install)
    job = engine.submit({'max_tokens': 1, 'paged_native_qwen3': True,
                         'skip_writing_prefix_cache': True, 'temperature': 0})
    assert state['native_waiting'].wait(3)
    job.cancelled.set()
    state['native_release'].set()
    result = job.events.get(timeout=5)
    assert result['error'] == 'cancelled'
    assert state['native_retired'] == [job.uid]
    assert state['native_retired'].count(job.uid) == 1
    assert not state['native_uids']
    state['native_hold'] = False
    ordinary = engine.submit({'max_tokens': 1})
    assert ordinary.events.get(timeout=5)['finish_reason'] == 'length'


def test_idle_worker_reaps_terminal_native_admission_owner(deferred_engine):
    from mlx2 import serving

    _engine, _state = deferred_engine
    calls = []
    owner = NS(
        fully_retired=True,
        reap_retired=lambda: calls.append("retired"),
        reap_quarantine=lambda: calls.append("quarantine"),
    )
    writer = NS(poisoned=False)
    record = (owner, writer, None)
    serving._NATIVE_ADMISSION_ORPHANS.append(record)
    try:
        deadline = time.monotonic() + 2
        while record in serving._NATIVE_ADMISSION_ORPHANS and time.monotonic() < deadline:
            time.sleep(0.01)
        assert record not in serving._NATIVE_ADMISSION_ORPHANS
        assert calls == ["retired", "quarantine"]
    finally:
        if record in serving._NATIVE_ADMISSION_ORPHANS:
            serving._NATIVE_ADMISSION_ORPHANS.remove(record)
