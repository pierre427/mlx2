"""CPU-only Source expert streaming policies: priority loader, cycle-aware
retention, storage-tier accounting, multi-layer lookahead, ridge heads.

Synthetic data only; no model load and no model-path GPU run.
"""
from collections import OrderedDict
from types import SimpleNamespace
import random
import threading

import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_buffers as buffers
from mlx2.adapters import deepseek_v41_source_stream as stream
from mlx2.adapters import deepseek_v41_source_weights as weights
from mlx2.runtime.priority_read_pool import PriorityReadPool
from tests.test_deepseek_v41_source_buffers_cpu import original_file

# ------------------------------------------------------------ priority pool


def test_priority_pool_runs_demand_before_queued_speculation_and_promotes():
    pool = PriorityReadPool(1, 'test-priority', timeout=30, max_pending=8)
    gate, order = threading.Event(), []
    blocker = pool.submit(lambda: gate.wait(10))
    spec = [pool.submit_ranked(depth, order.append, f'spec{depth}') for depth in (3, 1, 2)]
    promoted = pool.submit_ranked(4, order.append, 'promoted')
    demand = pool.submit(order.append, 'demand')
    assert pool.promote(promoted) is True
    assert pool.promote(blocker) is False  # running calls are never moved
    gate.set()
    for future in (blocker, *spec, promoted, demand):
        future.result(timeout=10)
    # Demand first (FIFO among rank 0, the promotion came after it), then by depth.
    assert order == ['demand', 'promoted', 'spec1', 'spec2', 'spec3']
    counters = pool.counters()
    assert counters['priority_demand_submitted'] == 2 and counters['priority_speculative_submitted'] == 4
    assert counters['priority_promotions'] == 1 and counters['priority_queued'] == 0
    assert pool.close() is True


def test_priority_pool_cancelled_speculation_never_runs():
    pool = PriorityReadPool(1, 'test-priority-cancel', timeout=30, max_pending=4)
    gate, ran = threading.Event(), []
    blocker = pool.submit(lambda: gate.wait(10))
    stale = pool.submit_ranked(2, ran.append, 'stale')
    assert stale.cancel()
    keep = pool.submit_ranked(1, ran.append, 'keep')
    gate.set()
    blocker.result(10); keep.result(10)
    assert ran == ['keep']
    with pytest.raises(ValueError):
        pool.submit_ranked(-1, ran.append, 'bad')
    pool.close()

# ------------------------------------------------------ cycle-aware retention


def test_cycle_aware_prefers_just_ran_layer_and_protects_leases():
    index = stream.CycleAwareIndex(4, sample=0)
    for layer in range(4):
        index.observe_counts(layer, [(0, 1)])
    index.observe_counts(3, [(1, 1), (2, 1)])
    for key in [(l, 0) for l in range(4)] + [(3, 1), (3, 2)]:
        index.touch(key)
    # Position is layer 3: layer 0 runs next (distance 1), layer 3 is furthest.
    assert [index.distance((l, 0)) for l in range(4)] == [1, 2, 3, 4]
    resident = {(l, 0): 1 for l in range(4)}
    resident.update({(3, 1): 1, (3, 2): 1})
    assert index.victim(resident, {}, lambda k: True) == (3, 0)
    assert index.victim(resident, {(3, 0): 1}, lambda k: True) == (2, 0)
    # The current observation's experts, still to be acquired, go last.
    assert index.victim(resident, {(3, 0): 1, (3, 1): 1}, lambda k: k[0] == 3) == (3, 2)
    assert index.victim({(3, 1): 1, (2, 0): 1}, {}, lambda k: True) == (2, 0)
    with pytest.raises(MemoryError):
        index.victim(resident, resident, lambda k: True)
    with pytest.raises(ValueError):
        index.observe_counts(4, [(0, 1)])


def _trace(layers=12, experts=24, top_k=4, tokens=300, seed=1):
    rng = np.random.default_rng(seed)
    weights_ = rng.dirichlet(np.full(experts, 0.3), size=layers)
    return [[(l, tuple(int(e) for e in rng.choice(experts, top_k, replace=False, p=weights_[l])))
             for l in range(layers)] for _ in range(tokens)]


def _misses(trace, capacity, index=None):
    cache, misses = OrderedDict(), 0
    for token in trace:
        for layer, chosen in token:
            if index is not None:
                index.observe_counts(layer, [(e, 1) for e in chosen])
            for e in chosen:
                key = (layer, e)
                if key in cache:
                    cache.move_to_end(key)
                else:
                    misses += 1
                    while len(cache) >= capacity:
                        leased = {(layer, x): 1 for x in chosen}
                        victim = (next(k for k in cache if k not in leased) if index is None
                                  else index.victim(cache, leased, lambda k: True))
                        cache.pop(victim)
                    cache[key] = 1
                if index is not None:
                    index.touch(key)
    return misses


def test_cycle_aware_beats_lru_below_one_token_working_set():
    trace = _trace()
    working_set = 12*4
    for capacity in (working_set//2, working_set*3//4):
        lru = _misses(trace, capacity)
        cycle = _misses(trace, capacity, stream.CycleAwareIndex(12, sample=0))
        assert cycle < lru, (capacity, cycle, lru)


def test_cycle_policy_keys_validate():
    from mlx2.adapters.deepseek_v41_source_lfu import parse_policy
    base = {'mode': 'original_lfu', 'budget_bytes': 90_000_000_000}
    ok = parse_policy({**base, 'retention': 'cycle_aware_v1', 'retention_alpha': 0.2, 'retention_sample': 0})
    assert ok['retention'] == 'cycle_aware_v1'
    for bad in ({'retention_alpha': 0.1},
                {'retention': 'cycle_aware_v1', 'retention_alpha': 0},
                {'retention': 'cycle_aware_v1', 'retention_sample': -1},
                {'lookahead_depth': 2},
                {'lookahead_depth': 5, 'loader': 'priority_v1'},
                {'lookahead_heads': 'relative.npz'},
                {'loader': 'lifo'},
                {'storage_tiers': 'on'},
                {'tier_declared': {'a': 'external'}},
                {'storage_tiers': 'accounting_v1', 'tier_declared': {'a': 'tape'}}):
        with pytest.raises(ValueError):
            parse_policy({**base, **bad})
    assert parse_policy({**base, 'lookahead_depth': 3, 'loader': 'priority_v1',
                         'storage_tiers': 'accounting_v1', 'tier_declared': {'x': 'external'}})

# -------------------------------------------------------------- storage tiers


def test_tier_classification_declared_symlink_and_chain(tmp_path):
    target = tmp_path/'real'
    target.mkdir()
    (target/'shard.safetensors').write_bytes(b'x')
    link = tmp_path/'shard.safetensors'
    link.symlink_to(target/'shard.safetensors')
    assert stream.classify_path(link) == stream.TIER_INTERNAL  # tmp lives on the system volume
    assert stream.classify_path(link, {str(target): stream.TIER_EXTERNAL}) == stream.TIER_EXTERNAL
    assert stream.classify_path(link, {'shard.safetensors': stream.TIER_REMOTE}) == stream.TIER_REMOTE
    with pytest.raises(ValueError):
        stream.classify_path(link, {'shard.safetensors': 'tape'})
    chain = stream.TierChain([stream.StorageTier(stream.TIER_EXTERNAL, lambda k: True),
                              stream.StorageTier(stream.TIER_INTERNAL, lambda k: k[0] < 2)])
    chain.insert(stream.StorageTier(stream.TIER_REMOTE, lambda k: k[0] < 4))
    assert [t.name for t in chain.tiers] == ['internal_nvme', 'remote_layer', 'external']
    assert [chain.locate((l, 0)) for l in (0, 3, 5)] == ['internal_nvme', 'remote_layer', 'external']
    accounting = stream.TierAccounting(chain)
    accounting.hit(); accounting.miss((0, 0), 10, 5); accounting.miss((5, 0), 7)
    counters = accounting.receipt()['counters']
    assert counters == {'tier_bytes_external': 7, 'tier_bytes_internal_nvme': 10, 'tier_hits_ram': 1,
                        'tier_misses_external': 1, 'tier_misses_internal_nvme': 1,
                        'tier_read_ns_external': 0, 'tier_read_ns_internal_nvme': 5}

    class Remote(stream.StorageTier):
        def read_job(self, key):
            return (print, key)
    with pytest.raises(ValueError, match='not admitted'):
        stream.TierChain([Remote(stream.TIER_REMOTE, lambda k: True)]).locate((0, 0))


def test_recommend_placement_costs_misses_and_engram_rows_per_byte():
    tensors = {
        'layers.1.engram.embed.weight': dict(file='e.safetensors', size=50),
        'layers.0.attn.wq.weight': dict(file='d.safetensors', size=20),
        'layers.0.ffn.experts.1.w1.weight': dict(file='x1.safetensors', size=30),
        'layers.0.ffn.experts.2.w1.weight': dict(file='x2.safetensors', size=30),
    }
    artifact = dict(tensors=tensors, seals={})
    freq = {(0, 2): 9, (0, 1): 1}
    internal, external, scores = stream.recommend_placement(artifact, 100, expert_frequency=freq, ram_expert_bytes=0)
    assert internal == ['e.safetensors', 'x2.safetensors', 'd.safetensors'] and external == ['x1.safetensors']
    assert scores['d.safetensors'] == 0 and scores['x2.safetensors'] == pytest.approx(9*scores['x1.safetensors'])
    # The hot expert fits RAM: only the cold one still pays T7 misses.
    _, _, scores = stream.recommend_placement(artifact, 100, expert_frequency=freq, ram_expert_bytes=30)
    assert scores['x2.safetensors'] == 0 and scores['x1.safetensors'] > 0
    with pytest.raises(ValueError):
        stream.recommend_placement(artifact, 100, t7_speed=1)


def test_engram_io_policy_is_opt_in_and_lossless_settings():
    from mlx2.adapters.deepseek_v41_source import source_policy
    assert stream.engram_io_settings('reference', 32) == {'workers': 8, 'cache_bytes': None}
    assert stream.engram_io_settings('tiered_v1', 2) == {'workers': 16, 'cache_bytes': 256 << 20}
    with pytest.raises(ValueError):
        stream.engram_io_settings('tiered_v2', 2)
    assert source_policy({'deepseek_source_engram_io': 'tiered_v1'})['deepseek_source_engram_io'] == 'tiered_v1'
    assert 'deepseek_source_engram_io' not in source_policy()
    with pytest.raises(ValueError):
        source_policy({'deepseek_source_engram_io': 'tiered_v2'})


def test_engram_io_receipt_reports_tier_and_cache(tmp_path):
    from mlx2.runtime.models.deepseek_v41_engram import EngramRows, file_stamp
    path = tmp_path/'engram.bin'
    path.write_bytes(bytes(264*64))
    settings = stream.engram_io_settings('tiered_v1', 2)
    table = EngramRows(path, offset=0, rows=64, expected_stat=file_stamp(path.stat()),
                       workers=settings['workers'], cache_bytes=settings['cache_bytes'])
    try:
        table.read(np.array([1, 2, 1]))
        receipt = stream.engram_io_receipt('tiered_v1', {14: table})
        row = receipt['tables']['14']
        assert row['storage_tier'] == stream.TIER_INTERNAL and row['workers'] == 16
        assert row['cache_budget_bytes'] == 256 << 20 and row['cache_misses'] == 2
        assert receipt['status'] == 'implemented_untested_unqualified'
    finally:
        table.close()

# ---------------------------------------------------------------- ridge heads


def test_ridge_head_recovers_residual_and_binds_checkpoint(tmp_path):
    rng = np.random.default_rng(0)
    n, d, e, k = 400, 16, 12, 3
    x = rng.standard_normal((n, d))
    base = x @ rng.standard_normal((d, e))*0.3
    truth = base + x @ rng.standard_normal((d, e))
    weight, bias = stream.fit_ridge_head(x[:300], base[:300], truth[:300], lam=1.0)
    trained = base[300:] + x[300:] @ weight.T + bias
    top = lambda s: np.argsort(-s, axis=1)[:, :k]
    assert stream.recall_at(top(trained), top(truth[300:])) > stream.recall_at(top(base[300:]), top(truth[300:])) + 0.2
    path = tmp_path/'heads.npz'
    digest = stream.save_heads(path, {(1, 2): (weight, bias)}, binding='ckpt', hidden_size=d, experts=e, lam=1.0)
    heads = stream.RidgeLookaheadHeads(path, binding='ckpt', hidden_size=d, experts=e)
    assert heads.sha256 == digest and heads.get(1, 2)[0].dtype == np.float16 and heads.get(1, 1) is None
    with pytest.raises(ValueError, match='bind'):
        stream.RidgeLookaheadHeads(path, binding='other', hidden_size=d, experts=e)
    with pytest.raises(ValueError):
        stream.fit_ridge_head(x, base, truth, lam=0)

# ------------------------------------------------- store integration (exact)

LAYERS, EXPERTS = 4, 16


def lookahead_store(tmp_path, monkeypatch, policy_extra, *, predictor=None, enable=True):
    from mlx2.adapters import deepseek_v41_source_lfu as lfu
    artifact, _ = original_file(tmp_path)
    base = {k: dict(v) for k, v in artifact['tensors'].items()}
    for layer in range(LAYERS):
        for e in range(EXPERTS):
            artifact['tensors'].update({k.replace('layers.0.', f'layers.{layer}.').replace('experts.0.', f'experts.{e}.'): dict(v)
                                        for k, v in base.items()})
    artifact['args'] = SimpleNamespace(num_hidden_layers=LAYERS, n_routed_experts=EXPERTS, num_experts_per_tok=2,
                                       hidden_size=8)

    class Pool:
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
    store = lfu.LFUSourceExpertStore(artifact, bounds, {'mode': 'original_lfu', 'budget_bytes': budget, **policy_extra},
                                     packed_buffers=True)
    store._synchronize = lambda: None
    if enable:
        store.enable_expert_overlap(predictor)
    return store


def routes(*rows):
    import mlx.core as mx
    return mx.array(np.array(rows, np.uint32))


def test_depth_two_lookahead_keeps_future_hints_supersedes_and_stays_exact(tmp_path, monkeypatch):
    import mlx.core as mx
    plan = {1: [[3, 4]], 2: [[5, 6]], 3: [[7, 8]]}
    calls = []

    def predictor(hidden, layer):
        calls.append(layer)
        # A deeper guess for layer 2 (made at layer 0) differs from the
        # shallower one (made at layer 1): [9, 6] then [5, 6].
        if layer == 2 and len([c for c in calls if c == 2]) == 1:
            return mx.array(np.array([[9, 6]], np.uint32))
        return mx.array(np.array(plan[layer], np.uint32))

    reference = lookahead_store(tmp_path, monkeypatch, {}, enable=False)
    with mx.stream(mx.cpu), reference.window():
        expected = {(l, e): np.asarray(reference.acquire(f'blk.{l}', e).backing).copy()
                    for l, e in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6))}
    reference.close()

    store = lookahead_store(tmp_path, monkeypatch, {'lookahead_depth': 2, 'loader': 'priority_v1'},
                            predictor=predictor)
    assert isinstance(store._overlap['pool'], PriorityReadPool)
    hidden = mx.zeros((1, 1, 8))
    got = {}
    with mx.stream(mx.cpu):
        for layer, chosen in ((0, [1, 2]), (1, [3, 4]), (2, [5, 6])):
            ticket = store.prepare(f'blk.{layer}', routes(chosen), route_hidden=hidden)
            if layer == 0:
                assert set(store._overlap['hints']) == {(1, 3), (1, 4), (2, 9), (2, 6)}
                assert store._overlap['hint_depth'][(2, 9)] == 2
            if layer == 1:
                # (2, 9) superseded by the fresher depth-1 guess; (2, 6) kept, now depth 1.
                assert set(store._overlap['hints']) >= {(2, 5), (2, 6), (3, 7), (3, 8)}
                assert (2, 9) not in store._overlap['hints'] and store._overlap['hint_depth'][(2, 6)] == 1
            with store.window():
                for e in chosen:
                    got[(layer, e)] = np.asarray(store.acquire(f'blk.{layer}', e).backing).copy()
            store.finish_ticket(ticket)
    assert all(np.array_equal(got[k], expected[k]) for k in expected)
    assert calls == [1, 2, 2, 3, 3]  # layer 3 has no layer 4: depth truncates at the last layer
    s = store.stats
    assert s['lookahead_hint_matched_d1'] == 4 and s['overlap_hint_used'] == 4
    assert s['lookahead_hint_superseded'] == 1 and s['lookahead_untrained_predictions'] == 5
    assert s['overlap_sync_reads'] == 0 and s['cache_misses'] == 6
    receipt = store.receipt()['expert_io_overlap']
    assert receipt['lookahead']['depth'] == 2 and receipt['lookahead']['loader'] == 'priority_v1'
    assert receipt['max_pending_reads'] == weights.OVERLAP_PENDING + weights.OVERLAP_HINTS
    assert receipt['lookahead']['status'] == 'implemented_untested_unqualified'
    assert receipt['lookahead']['loader_counters']['priority_speculative_submitted'] >= 6
    store.end_context()
    store.close()


def test_default_overlap_receipt_has_no_lookahead_section(tmp_path, monkeypatch):
    import mlx.core as mx
    store = lookahead_store(tmp_path, monkeypatch, {}, predictor=lambda h, l: mx.array(np.array([[1, 2]], np.uint32)))
    receipt = store.receipt()['expert_io_overlap']
    assert 'lookahead' not in receipt and receipt['max_pending_reads'] == weights.OVERLAP_PENDING
    assert not any(k.startswith('lookahead_') for k in store.stats)
    assert not isinstance(store._overlap['pool'], PriorityReadPool)
    store.close()


def test_lookahead_refuses_demand_only_and_fifo_depth(tmp_path, monkeypatch):
    store = lookahead_store(tmp_path, monkeypatch, {'lookahead_depth': 2, 'loader': 'priority_v1'}, enable=False)
    with pytest.raises(ValueError, match='route predictor'):
        store.enable_expert_overlap(None)
    store.close()


def test_lookahead_keys_fail_closed_without_overlap_v1():
    from mlx2.adapters.deepseek_v41_source import source_policy
    experts = {'mode': 'original_lfu', 'budget_bytes': 90_000_000_000, 'lookahead_depth': 2, 'loader': 'priority_v1'}
    with pytest.raises(ValueError, match='overlap_v1'):
        source_policy({'deepseek_source_experts': experts})
    with pytest.raises(ValueError, match='overlap_v1'):
        source_policy({'deepseek_source_experts': experts, 'deepseek_source_expert_io': 'overlap_demand_v1'})
    selected = source_policy({'deepseek_source_experts': experts, 'deepseek_source_expert_io': 'overlap_v1'})
    assert selected['deepseek_source_expert_io'] == 'overlap_v1'
    # Retention and tier accounting do not need overlap.
    assert source_policy({'deepseek_source_experts': {'mode': 'original_lfu', 'budget_bytes': 90_000_000_000,
                                                      'retention': 'cycle_aware_v1', 'storage_tiers': 'accounting_v1'}})


def test_multirow_ticket_drops_future_hints_and_repredict_promotes(tmp_path, monkeypatch):
    import mlx.core as mx
    store = lookahead_store(tmp_path, monkeypatch, {'lookahead_depth': 3, 'loader': 'priority_v1'},
                            predictor=lambda h, l: mx.array(np.array([[10, 11]], np.uint32)))
    promoted = []
    real = store._overlap['pool'].promote
    store._overlap['pool'].promote = lambda future, rank=0: promoted.append(rank) or real(future, rank)
    hidden = mx.zeros((1, 1, 8))
    with mx.stream(mx.cpu):
        store.cancel(store.prepare('blk.0', routes([0, 1]), route_hidden=hidden))
        assert {k[0] for k in store._overlap['hints']} == {1, 2, 3}
        assert store._overlap['hint_depth'][(3, 10)] == 3
        store.cancel(store.prepare('blk.1', routes([10, 11]), route_hidden=hidden))
        # (2, *) and (3, *) re-predicted at depths 1 and 2: re-ranked in the queue.
        assert store._overlap['hint_depth'][(2, 10)] == 1 and store._overlap['hint_depth'][(3, 10)] == 2
        assert 1 in promoted and 2 in promoted
        store.cancel(store.prepare('blk.1', routes([1, 2], [3, 4])))
        assert not store._overlap['hints'] and not store._overlap['hint_depth']
    store.end_context()
    store._drain_overlap()
    store.close()


def test_ridge_head_predictions_pass_head_and_count(tmp_path, monkeypatch):
    import mlx.core as mx
    path = tmp_path/'heads.npz'
    stream.save_heads(path, {(1, 1): (np.zeros((EXPERTS, 8), np.float32), np.zeros(EXPERTS, np.float32))},
                      binding='synthetic-no-model', hidden_size=8, experts=EXPERTS, lam=100.0)
    seen = []

    def predictor(hidden, layer, head=None):
        seen.append((layer, head is not None))
        return mx.array(np.array([[1, 2]], np.uint32))
    store = lookahead_store(tmp_path, monkeypatch, {'lookahead_heads': str(path)}, predictor=predictor)
    with mx.stream(mx.cpu):
        ticket = store.prepare('blk.0', routes([0, 1]), route_hidden=mx.zeros((1, 1, 8)))
        store.cancel(ticket)
        ticket = store.prepare('blk.1', routes([1, 2]), route_hidden=mx.zeros((1, 1, 8)))
        store.cancel(ticket)
    assert seen == [(1, True), (2, False)]
    assert store.stats['lookahead_head_predictions'] == 1 and store.stats['lookahead_untrained_predictions'] == 1
    assert store.receipt()['expert_io_overlap']['lookahead']['heads']['heads'] == 1
    store.end_context()
    store._drain_overlap()
    store.close()


def test_cycle_retention_and_tier_accounting_in_store(tmp_path, monkeypatch):
    import mlx.core as mx
    store = lookahead_store(tmp_path, monkeypatch, {'retention': 'cycle_aware_v1', 'retention_sample': 0,
                                                    'storage_tiers': 'accounting_v1'}, enable=False)
    assert isinstance(store.index, stream.CycleAwareIndex) and store.index.layers == LAYERS
    with mx.stream(mx.cpu):
        for _ in range(2):
            for layer in range(LAYERS):
                store.observe_routes(f'blk.{layer}', np.array([[0, 1]], np.uint32))
                with store.window():
                    store.acquire(f'blk.{layer}', 0); store.acquire(f'blk.{layer}', 1)
    counters = store.receipt()['storage_tiers']['counters']
    misses = store.stats['cache_misses']
    assert counters['tier_misses_internal_nvme'] == misses
    assert counters['tier_bytes_internal_nvme'] == misses*buffers.ORIGINAL_BYTES
    assert counters.get('tier_hits_ram', 0) == store.stats['cache_hits']
    retention = store.receipt()['retention']
    assert retention['policy'] == 'cycle_aware_v1' and retention['status'] == 'implemented_untested_unqualified'
    store.close()
