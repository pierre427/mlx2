"""Synthetic overlap_v1 expert-I/O contracts; no model load or model-path GPU run."""
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_buffers as buffers
from mlx2.adapters import deepseek_v41_source_weights as weights
from tests.test_deepseek_v41_source_buffers_cpu import original_file

LAYERS, EXPERTS = 2, 16


def overlap_store(tmp_path, monkeypatch, *, predictor=None, enable=True, experts_per_tok=2):
    from mlx2.adapters import deepseek_v41_source_lfu as lfu
    artifact, _ = original_file(tmp_path)
    base = {k: dict(v) for k, v in artifact['tensors'].items()}
    for layer in range(LAYERS):
        for e in range(EXPERTS):
            artifact['tensors'].update({k.replace('layers.0.', f'layers.{layer}.').replace('experts.0.', f'experts.{e}.'): dict(v)
                                        for k, v in base.items()})
    artifact['args'] = SimpleNamespace(num_hidden_layers=LAYERS, n_routed_experts=EXPERTS,
                                       num_experts_per_tok=experts_per_tok)

    class Pool:  # Ordinary synchronous demand reader, as in the buffer tests.
        failure = None
        def __init__(self, *args, **kwargs): self.calls = []
        def call(self, function, *args):
            self.calls.append(function.__name__)
            return function(*args)
        def close(self): pass
        def counters(self): return {'weight_read_inflight': 0}
    monkeypatch.setattr(weights, 'ReadPool', Pool)
    budget = 8*buffers.ORIGINAL_BYTES
    bounds = dict(expert_cache_bytes=budget, minimum_expert_cache_bytes=buffers.ORIGINAL_BYTES,
                  expert_table_bytes=LAYERS*EXPERTS*buffers.ORIGINAL_BYTES, manifest_binding_sha256='synthetic-no-model',
                  expert_active_bound_bytes=budget, expert_staging_bound_bytes=0)
    store = lfu.LFUSourceExpertStore(artifact, bounds, {'mode': 'original_lfu', 'budget_bytes': budget},
                                     packed_buffers=True)
    store._synchronize = lambda: None
    if enable:
        store.enable_expert_overlap(predictor)
    return store


def routes(*rows):
    import mlx.core as mx
    return mx.array(np.array(rows, np.uint32))


def backing(arrays):
    return np.asarray(arrays.backing).copy()


def test_policy_admission_is_opt_in_and_fails_closed():
    from mlx2.adapters.deepseek_v41_source import source_policy
    experts = {'mode': 'original_lfu', 'budget_bytes': 90_000_000_000}
    assert 'deepseek_source_expert_io' not in source_policy({'deepseek_source_experts': experts})
    selected = source_policy({'deepseek_source_experts': experts, 'deepseek_source_expert_io': 'overlap_v1'})
    assert selected['deepseek_source_expert_io'] == 'overlap_v1'
    for extra in ({}, ):
        with pytest.raises(ValueError, match='overlap_v1'):
            source_policy({'deepseek_source_expert_io': 'overlap_v1', **extra})
    base = {'deepseek_source_experts': experts, 'deepseek_source_expert_io': 'overlap_v1'}
    for extra in ({'deepseek_source_submission': 'shared_first_v1'},
                  {'deepseek_physical_batch': 'b2'},
                  {'deepseek_source_resident_gather': 'resident_bank_v1', 'deepseek_source_expert_math': 'routed_fused_fp32'}):
        with pytest.raises(ValueError):
            source_policy({**base, **extra})
    with pytest.raises(ValueError):
        source_policy({**base, 'deepseek_source_expert_io': 'overlap_v2'})
    with pytest.raises(ValueError, match='expert I/O'):
        weights.SourceV41Weights(None, expert_cache_bytes=0, source_expert_io_policy='overlap_v2')
    with pytest.raises(ValueError, match='overlap_v1 requires'):
        weights.SourceV41Weights(None, expert_cache_bytes=0, source_expert_io_policy='overlap_v1')


def test_early_reads_upload_identical_bytes_and_keep_lfu_accounting(tmp_path, monkeypatch):
    import mlx.core as mx
    reference = overlap_store(tmp_path, monkeypatch, enable=False)
    with mx.stream(mx.cpu), reference.window():
        expected = {e: backing(reference.acquire('blk.0', e)) for e in (3, 5)}
    reference_touch = dict(reference.index.last)
    reference.close()
    store = overlap_store(tmp_path, monkeypatch)
    calls = []
    real = buffers.materialize
    def spy(host, *, stream=None):
        calls.append(stream)
        return real(host, stream=stream)
    from mlx2.adapters import deepseek_v41_source_lfu as lfu
    monkeypatch.setattr(lfu, 'materialize', spy)
    ticket = store.prepare('blk.0', routes([5, 3]))
    assert set(ticket.demand) == {(0, 3), (0, 5)} and not ticket.queued
    assert not store._cache and not store._pending_host and not store._leased
    assert not store.index.counts and not store.index.last  # prepare is not an LFU observation
    with store.window():
        got = {e: backing(store.acquire('blk.0', e)) for e in (3, 5)}
    store.finish_ticket(ticket)
    assert all(np.array_equal(got[e], expected[e]) for e in expected)
    assert calls == [mx.cpu, mx.cpu]
    assert dict(store.index.last) == reference_touch and not store.index.counts
    stats = store.stats
    assert stats['overlap_consumed_bytes'] == 2*buffers.ORIGINAL_BYTES
    assert stats['overlap_sync_reads'] == 0 and stats['overlap_route_readbacks'] == 1
    assert stats['cache_misses'] == 2 and stats['bytes_read'] == 2*buffers.ORIGINAL_BYTES
    assert stats['upload_offqueue_evals'] == 2
    assert store.pool.calls == []
    assert not store._leased and not store._pending_uploads and not store._pending_host
    receipt = store.receipt()['expert_io_overlap']
    assert receipt['policy'] == 'overlap_v1' and receipt['max_pending_reads'] == weights.OVERLAP_PENDING
    assert receipt['counters']['overlap_consumed_bytes'] == 2*buffers.ORIGINAL_BYTES
    store.close()


def test_cached_hits_issue_no_reads_and_unticketed_misses_stay_synchronous(tmp_path, monkeypatch):
    import mlx.core as mx
    store = overlap_store(tmp_path, monkeypatch)
    with mx.stream(mx.cpu):
        with store.window():
            store.acquire('blk.0', 1)
        assert store.pool.calls == ['read_original_packed']
        ticket = store.prepare('blk.0', routes([1, 2]))
        assert set(ticket.demand) == {(0, 2)}
        with store.window():
            store.acquire('blk.0', 1)
            store.acquire('blk.0', 2)
        store.finish_ticket(ticket)
    assert store.stats['cache_hits'] == 1 and store.pool.calls == ['read_original_packed']
    store.close()


def test_predicted_next_layer_reads_are_hints_only(tmp_path, monkeypatch):
    import mlx.core as mx
    seen = []
    def predictor(hidden, layer):
        seen.append(layer)
        return mx.array(np.array([[7, 9]], np.uint32))
    store = overlap_store(tmp_path, monkeypatch, predictor=predictor)
    hidden = mx.zeros((1, 1, 8))
    with mx.stream(mx.cpu):
        ticket = store.prepare('blk.0', routes([0, 1]), route_hidden=hidden)
        assert seen == [1] and set(store._overlap['hints']) == {(1, 7), (1, 9)}
        with store.window():
            store.acquire('blk.0', 0); store.acquire('blk.0', 1)
        store.finish_ticket(ticket)
        # Hints never upload, cache, lease or count.
        assert not any(k[0] == 1 for k in store._cache) and not store.index.counts
        ticket = store.prepare('blk.1', routes([9, 4]), route_hidden=hidden)
        assert seen == [1] and (1, 9) in ticket.hinted and (1, 4) in ticket.demand
        assert not store._overlap['hints']  # (1, 7) was retired as waste
        with store.window():
            store.acquire('blk.1', 4); store.acquire('blk.1', 9)
        store.finish_ticket(ticket)
    s = store.stats
    assert s['overlap_hint_issued'] == 2 and s['overlap_hint_used'] == 1 and s['overlap_hint_wasted'] == 1
    assert s['overlap_hint_predictions'] == 1 and s['overlap_sync_reads'] == 0
    assert s['overlap_route_readbacks'] == 2 and store.pool.calls == []
    store.end_context()
    store.close()


def test_read_pool_pending_bound_and_cancel_keep_store_healthy(tmp_path, monkeypatch):
    import mlx.core as mx
    gate = threading.Event()
    from mlx2.adapters import deepseek_v41_source_lfu as lfu
    real = lfu.read_original_packed
    def slow(*args):
        gate.wait(10)
        return real(*args)
    monkeypatch.setattr(lfu, 'read_original_packed', slow)
    store = overlap_store(tmp_path, monkeypatch, experts_per_tok=1)
    ticket = store.prepare('blk.0', routes(*[[e] for e in range(EXPERTS)]))
    assert len(ticket.demand) == weights.OVERLAP_PENDING
    assert len(ticket.queued) == EXPERTS-weights.OVERLAP_PENDING
    assert store._overlap['pool'].counters()['weight_read_inflight'] <= weights.OVERLAP_PENDING
    with pytest.raises(ValueError, match='already owned'):
        store.prepare('blk.0', routes([0]))
    with pytest.raises(ValueError, match='ticket still owns'):
        store.park()
    store.cancel(ticket)
    store.cancel(ticket)  # idempotent for the block's error path
    gate.set()
    store._drain_overlap()
    assert store._overlap_inflight() == 0 and not store._cache
    with mx.stream(mx.cpu), store.window():
        store.acquire('blk.0', 0)
    assert store._error is None
    store.park()
    store.close()


def test_failed_upload_poisons_store_and_refuses_tickets(tmp_path, monkeypatch):
    import mlx.core as mx
    store = overlap_store(tmp_path, monkeypatch)
    ticket = store.prepare('blk.0', routes([0, 1]))
    real_eval = mx.eval
    def fail(*args):
        raise RuntimeError('injected upload failure')
    monkeypatch.setattr(mx, 'eval', fail)
    with mx.stream(mx.cpu), pytest.raises(RuntimeError, match='injected'):
        with store.window():
            store.acquire('blk.0', 0)
    monkeypatch.setattr(mx, 'eval', real_eval)
    store.cancel(ticket)
    assert store._error is not None
    with pytest.raises(ValueError, match='failed'):
        store.prepare('blk.0', routes([0, 1]))
    store.close()


def test_demand_read_failure_raises_but_hint_failure_falls_back(tmp_path, monkeypatch):
    import mlx.core as mx
    from mlx2.adapters import deepseek_v41_source_lfu as lfu
    real = lfu.read_original_packed
    bad = set()
    def flaky(artifact, layer, expert):
        if (layer, expert) in bad:
            raise OSError('injected read failure')
        return real(artifact, layer, expert)
    monkeypatch.setattr(lfu, 'read_original_packed', flaky)
    store = overlap_store(tmp_path, monkeypatch, predictor=lambda h, l: mx.array(np.array([[3, 4]], np.uint32)))
    bad.add((1, 3))
    with mx.stream(mx.cpu):
        ticket = store.prepare('blk.0', routes([0, 1]), route_hidden=mx.zeros((1, 8)))
        with store.window():
            store.acquire('blk.0', 0); store.acquire('blk.0', 1)
        store.finish_ticket(ticket)
        for future in store._overlap['hints'].values():
            try: future.result(10)
            except OSError: pass
        bad.clear()
        ticket = store.prepare('blk.1', routes([3, 4]))
        with store.window():
            first = store.acquire('blk.1', 3)
            store.acquire('blk.1', 4)
        store.finish_ticket(ticket)
        assert store.stats['overlap_hint_failed'] == 1 and store._overlap['hints_disabled']
        assert store.pool.calls == ['flaky'] and isinstance(first, buffers.PackedExpert)
        bad.add((0, 9))
        ticket = store.prepare('blk.0', routes([9, 0]))
        with pytest.raises(OSError, match='injected'):
            with store.window():
                store.acquire('blk.0', 9)
        store.cancel(ticket)
    assert store._error is not None
    store.close()


def test_moe_ticket_must_match_policy_prefix_and_owner(tmp_path, monkeypatch):
    import mlx.core as mx
    store = overlap_store(tmp_path, monkeypatch)
    owner = weights.SourceV41Weights.__new__(weights.SourceV41Weights)
    owner._ready = lambda: None
    owner.expert_stream, owner.source_resident_bank_session = store, None
    owner.source_expert_io_policy = 'reference'
    ids = routes([0, 1])
    ticket = store.prepare('blk.0', ids)
    x, scores = mx.zeros((1, 8)), mx.zeros((1, 2))
    with pytest.raises(ValueError, match='ticket does not match'):
        owner.moe(x, ids, scores, 'blk.0', ticket=ticket)
    owner.source_expert_io_policy = 'overlap_v1'
    with pytest.raises(ValueError, match='ticket does not match'):
        owner.moe(x, ids, scores, 'blk.1', ticket=ticket)
    store.cancel(ticket)
    with pytest.raises(ValueError, match='ticket does not match'):
        owner.moe(x, ids, scores, 'blk.0', ticket=ticket)
    store.close()


def test_predictor_follows_router_law_on_identical_input():
    import mlx.core as mx
    from mlx2.runtime.models.deepseek_v4 import route_experts
    rng = np.random.default_rng(3)
    w = rng.standard_normal((16, 8)).astype(np.float32)
    b = rng.standard_normal(16).astype(np.float32)
    x = rng.standard_normal((1, 1, 8)).astype(np.float32)
    owner = weights.SourceV41Weights.__new__(weights.SourceV41Weights)
    owner.tensors = {'blk.1.ffn_gate_inp.weight': None, 'blk.1.exp_probs_b.bias': None}
    arrays = {'blk.1.ffn_gate_inp.weight': mx.array(w).astype(mx.bfloat16), 'blk.1.exp_probs_b.bias': mx.array(b)}
    owner.array = arrays.__getitem__
    owner.artifact = {'args': SimpleNamespace(hidden_size=8, num_experts_per_tok=4)}
    with mx.stream(mx.cpu):
        predicted = owner._predict_next_routes(mx.array(x), 1)
        hx = mx.array(x).astype(mx.bfloat16).astype(mx.float32)
        exact, _ = route_experts(hx.reshape(1, 8) @ arrays['blk.1.ffn_gate_inp.weight'].astype(mx.float32).T,
                                 top_k=4, scale=1.0, correction=mx.array(b))
        assert set(np.asarray(predicted).ravel()) == set(np.asarray(exact).ravel())
        assert owner._predict_next_routes(mx.zeros((2, 8)), 1) is None
        assert owner._predict_next_routes(mx.zeros((1, 8)), 2) is None


def test_offqueue_views_are_bitwise_identical_for_gpu_consumers():
    """Small synthetic Metal check: CPU-stream view build, same GPU kernel result."""
    import mlx.core as mx
    if not mx.metal.is_available():
        pytest.skip('Metal unavailable')
    layout = buffers.canonical_layout('original')
    rng = np.random.default_rng(11)
    slab = rng.integers(0, 256, layout.nbytes, dtype=np.uint8)
    for t in layout.tensors:
        if t.name.endswith('.scales'):
            slab[t.offset:t.offset+t.nbytes] = rng.integers(118, 136, t.nbytes, dtype=np.uint8)
    host = buffers.HostExpert(slab, layout)
    x = mx.array(rng.standard_normal((1, 5120)).astype(np.float32))
    results = []
    for stream in (None, mx.cpu):
        packed = buffers.materialize(host) if stream is None else buffers.materialize(host, stream=stream)
        mx.eval(packed.backing, *packed.values())
        with mx.stream(mx.gpu):
            y = mx.quantized_matmul(x, packed['gate.qweight'], scales=packed['gate.scales'], mode='mxfp4')
            mx.eval(y)
        results.append(np.asarray(y))
    assert np.isfinite(results[0]).all() and np.array_equal(results[0], results[1])


def test_demand_only_variant_admits_and_labels_without_hints(tmp_path, monkeypatch):
    from mlx2.adapters.deepseek_v41_source import source_policy
    experts = {'mode': 'original_lfu', 'budget_bytes': 90_000_000_000}
    selected = source_policy({'deepseek_source_experts': experts, 'deepseek_source_expert_io': 'overlap_demand_v1'})
    assert selected['deepseek_source_expert_io'] == 'overlap_demand_v1'
    with pytest.raises(ValueError, match='overlap_v1 requires'):
        weights.SourceV41Weights(None, expert_cache_bytes=0, source_expert_io_policy='overlap_demand_v1')
    store = overlap_store(tmp_path, monkeypatch)
    receipt = store.overlap_receipt()
    assert receipt['predictor'] is False and receipt['counters']['overlap_hint_issued'] == 0
    store.close()
