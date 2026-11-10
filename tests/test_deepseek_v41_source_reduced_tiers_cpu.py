"""Reduced-precision tiers: Q1 pool, per-layer budgets, resolver, Q2 tier, rank tiers.

CPU synthetic stores only (scaled expert sizes, MLX pinned to the CPU device):
no model load, no Metal work.
"""
from collections import defaultdict

import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_lfu as lfu
from mlx2.adapters import deepseek_v41_source_q1_on_miss as q1m
from mlx2.adapters import deepseek_v41_source_tiers as tiers
from tests.test_deepseek_v41_source_residency_cpu import O, Q, small  # noqa: F401 - fixture
from tests.test_deepseek_v41_source_q1_on_miss_cpu import call, experts, rule, store

GB = 10 ** 9


# ---- Cached Q1 pool (pool_bytes), per-layer LRU ----------------------------

def pooled(*, pool_layers=(2, 2), max_weight=0.10):
    """Total 12 O; Q1 pool carved per layer (sizes in Q1 objects)."""
    s = store(max_weight=max_weight)
    total = s.cache_budget_bytes
    settings = {'max_weight': max_weight, 'layers': [0, 1], 'pool_bytes': sum(pool_layers) * Q,
                'overrides': [{'layers': l, 'q1_bytes': n * Q} for l, n in enumerate(pool_layers)]}
    s.policy['q1_on_miss'] = settings
    s.q1_on_miss = q1m.Q1OnMiss(settings, root='/q1b', records=defaultdict(lambda: {'file': 'q1', 'size': Q + 8}),
                                bank_binding='d' * 64, layers=2, budget_bytes=total)
    s._carve_q1_pool()
    s.configure_adaptive_residency(available_bytes=lambda: s.available[0], working_set_experts=1)
    return s


def test_pool_policy_is_carved_from_the_total_and_validated():
    parsed = lfu.parse_policy(experts(budget_bytes=90_000_000_000,
                                      q1_on_miss={'max_weight': 0.1, 'pool_bytes': 15_000_000_000}))
    assert parsed['q1_on_miss']['pool_bytes'] == 15_000_000_000
    for bad in (0, 1, 90_000_000_000, 89_990_000_000, True, 1.5e10):
        with pytest.raises(ValueError):
            lfu.parse_policy(experts(budget_bytes=90_000_000_000, q1_on_miss={'max_weight': 0.1, 'pool_bytes': bad}))
    with pytest.raises(ValueError, match='only q1'):
        lfu.parse_policy(experts(q1_on_miss={'max_weight': 0.1, 'pool_bytes': GB,
                                             'overrides': [{'layers': 0, 'q4_bytes': 1}]}))
    with pytest.raises(ValueError, match='exceeds budget_bytes'):
        lfu.parse_policy(experts(budget_bytes=90 * GB, q1_on_miss={
            'max_weight': 0.1, 'pool_bytes': GB, 'overrides': [{'layers': [0, 39], 'q1_bytes': 3 * GB}]}))
    with pytest.raises(ValueError, match='mutually exclusive'):
        lfu.parse_policy(experts(tiers={'usage_map': '/u.json'}))


def test_pool_hits_and_lru_evicts_within_its_layer_only(small):
    s = pooled(pool_layers=(2, 1))
    assert s.maximum_cache_bytes == s.expert_cache_total_bytes - 3 * Q
    for e in (1, 2):
        call(s, 0, [9, e], [0.99, 0.01])
    reads = len(s.reads)
    call(s, 0, [9, 1], [0.99, 0.01])            # pool hit (layer 0), no Q1 read
    assert len(s.reads) == reads
    call(s, 1, [9, 5], [0.99, 0.01])            # layer 1 has its own budget
    call(s, 0, [9, 3], [0.99, 0.01])            # layer 0 full: LRU evicts expert 2, not 1
    pool = s.q1_on_miss.pools['q1']
    assert set(pool.layers[0]) == {1, 3} and set(pool.layers[1]) == {5}
    assert pool.stats['evictions'] == [1, 0] and pool.stats['hits'] == [1, 0]
    assert all(k not in s._cache for k in ((0, 1), (0, 3), (1, 5)))
    r = s.receipt()['q1_on_miss']['pools']['q1']
    assert r['per_layer']['objects'] == [2, 1] and r['live_bytes'] == 3 * Q and r['eviction'] == 'lru_within_layer'
    assert s.receipt()['expert_cache_total_maximum_bytes'] == s.expert_cache_total_bytes


def test_originals_and_pool_never_evict_each_other(small):
    s = pooled(pool_layers=(2, 2))
    call(s, 0, [9, 1], [0.99, 0.01])
    pool = s.q1_on_miss.pools['q1']
    for e in range(10, 16):                      # many original misses
        call(s, 1, [e], [1.0])
    assert s._cache_bytes <= s.maximum_cache_bytes and 1 in pool.layers[0]
    assert s.live_bytes() == s._cache_bytes + pool.live_bytes   # pool counted once


def test_memory_pressure_shrinks_pool_with_the_total(small):
    s = pooled(pool_layers=(2, 2))
    for e in (1, 2):
        call(s, 0, [9, e], [0.99, 0.01])
    pool = s.q1_on_miss.pools['q1']
    s.available[0] = 0                           # nothing free: total target collapses
    s.adaptive_residency_update()
    assert pool.targets[0] < 2 * Q and pool.live[0] <= pool.targets[0]
    assert pool.stats['pressure_evictions'][0] >= 1
    assert s._adaptive_last['reduced_pools']['q1']['capacity_bytes'] == 4 * Q
    s.available[0] = 1 << 40
    s.adaptive_residency_update()
    assert pool.targets == pool.budgets


def test_salt_includes_pool_and_layout(small):
    a, b = pooled(pool_layers=(2, 2)), pooled(pool_layers=(3, 1))
    assert len({a.q1_on_miss.salt, b.q1_on_miss.salt, rule().salt}) == 3


# ---- Tier resolver ---------------------------------------------------------

USAGE = [[384 - e for e in range(384)] for _ in range(40)]   # expert 0 most used everywhere


def layout(overrides=None, *, budget=90 * GB, q2=15 * GB, q1=15 * GB, q4=None, usage=USAGE):
    totals = {'q2': q2, 'q1': q1, **({} if q4 is None else {'q4': q4})}
    return tiers.resolve(budget_bytes=budget, totals=totals, usage=usage, overrides=overrides)


def test_resolver_default_bands_and_remainder_rule():
    out = layout()
    assert out['budgets'][0] == {'q4': 60 * GB // 40, 'q2': 15 * GB // 40, 'q1': 15 * GB // 40}
    n4, n2 = 60 * GB // 40 // tiers.ORIGINAL_BYTES, 15 * GB // 40 // tiers.Q2_BYTES
    row = out['map'][5]
    assert row[:n4] == '4' * n4 and row[n4:n4 + n2] == '2' * n2 and set(row[n4 + n2:]) == {'1'}
    assert out['counts'][5] == {'q4': n4, 'q2': n2, 'q1': 384 - n4 - n2}


def test_resolver_spans_counts_and_precedence():
    # Defaults leave 20 GB of headroom for the overrides (ablation-ladder shape).
    out = layout([{'layers': [13, 26], 'q4_experts': 64, 'q2_experts': 64, 'q1_experts': 192},
                  {'layers': 20, 'q4_experts': 1, 'q2_experts': 0, 'q1_experts': 383}],
                 q4=50 * GB, q2=10 * GB, q1=10 * GB)
    assert out['counts'][13] == {'q4': 64, 'q2': 64, 'q1': 256}       # remainder -> q1
    assert out['counts'][20] == {'q4': 1, 'q2': 0, 'q1': 383}          # later entry wins
    assert out['map'][20][0] == '4' and out['budgets'][20]['q1'] == 383 * tiers.Q1_BYTES
    assert out['counts'][12]['q4'] == 50 * GB // 40 // tiers.ORIGINAL_BYTES
    for span in (40, [5, 2], [-1, 3], 'x'):
        with pytest.raises(ValueError, match='layers'):
            layout([{'layers': span, 'q1_bytes': 1}])


def test_resolver_refuses_overflow_and_bad_counts():
    with pytest.raises(ValueError, match='exceeds budget_bytes'):
        layout([{'layers': [13, 26], 'q4_experts': 384}])
    with pytest.raises(ValueError, match='sum to'):
        layout([{'layers': 3, 'q4_experts': 200, 'q2_experts': 100, 'q1_experts': 100}])
    with pytest.raises(ValueError, match='both'):
        layout([{'layers': 3, 'q1_bytes': 1, 'q1_experts': 1}])
    with pytest.raises(ValueError, match='usage map'):
        layout([{'layers': 3, 'only': 'q1'}], usage=None)


def test_resolver_selection_overrides_and_identity():
    base = layout()
    chosen = [{'layers': 7, 'tier': 'q4', 'experts': [383, 382]},
              {'layers': [30, 39], 'only': 'q1'},
              {'layers': [0, 1], 'tier': 'q2', 'top': 3}]
    forced = layout(chosen)
    assert forced['map'][7][383] == '4' and forced['map'][7][382] == '4'
    assert set(forced['map'][35]) == {'1'} and forced['map'][0][:3] == '222'
    assert base['identity'] != forced['identity'] and layout(chosen)['identity'] == forced['identity']
    rank = layout([{'layers': 32, 'rank_tiers': {'q4': 1, 'rest': 'q1'}}])
    assert rank['rank_tiers'][32] == {'q4': 1, 'q2': 0, 'rest': 'q1'} and rank['identity'] != base['identity']
    for bad in ({'q4': 7, 'rest': 'q1'}, {'q4': 1}, {'q4': 1, 'rest': 'q3'}):
        with pytest.raises(ValueError, match='rank_tiers'):
            layout([{'layers': 32, 'rank_tiers': bad}])
    with pytest.raises(ValueError, match='not configured'):
        tiers.resolve(budget_bytes=90 * GB, totals={'q1': GB}, usage=USAGE, tiers=('q4', 'q1'),
                      overrides=[{'layers': 1, 'q2_experts': 3}])


# ---- Tiers store: map, Q2 pool, transient guard, rank tiers ------------------

def tiered(monkeypatch, overrides, *, max_weight=1.0):
    monkeypatch.setattr(q1m, 'Q2_BYTES', 2 * Q)
    s = store()
    total = s.cache_budget_bytes
    usage = [[16 - e for e in range(16)] for _ in range(2)]
    settings = {'usage_map': '/u.json', 'q2_root': '/q2', 'q2_bytes': 2 * 2 * Q, 'q1_bytes': 2 * 2 * Q,
                'max_weight': max_weight, 'layers': [0, 1], 'overrides': overrides}
    s.policy = {'mode': 'original_lfu', 'budget_bytes': total, 'q1_root': '/q1b', 'tiers': settings}
    q2 = q1m.Bank('q2', '/q2', defaultdict(lambda: {'file': 'q2', 'size': 2 * Q + 8}), 'f' * 64, 2 * Q)
    s.q1_on_miss = q1m.Q1OnMiss(settings, root='/q1b', records=defaultdict(lambda: {'file': 'q1', 'size': Q + 8}),
                                bank_binding='d' * 64, layers=2, mode='tiers', budget_bytes=total, q2_bank=q2,
                                usage=usage, usage_binding='e' * 64, experts=16)
    s._carve_q1_pool()
    s.configure_adaptive_residency(available_bytes=lambda: s.available[0], working_set_experts=1)
    base = s._read_expert

    def read(key, function, *args):
        if function.__name__ == 'q2_read':
            s.reads.append((key, 'q2_read'))
            return {'gate.q2weight': np.zeros(2 * Q, np.uint8)}
        return base(key, function, *args)
    s._read_expert = read
    return s


def test_tier_map_routes_each_miss_to_its_tier(small, monkeypatch):
    s = tiered(monkeypatch, [{'layers': [0, 1], 'only': 'q1'},
                             {'layers': 0, 'tier': 'q4', 'experts': [0]},
                             {'layers': 0, 'tier': 'q2', 'experts': [1]}])
    _, out = call(s, 0, [0, 1, 2], [0.5, 0.3, 0.2])
    assert s.reads == [((0, 0), 'read_tensors'), ((0, 1), 'q2_read'), ((0, 2), 'q1_read')]
    assert (0, 0) in s._cache and (0, 1) not in s._cache and q1m.is_q2(out[1])
    r = s.q1_on_miss.receipt()
    assert r['per_layer_served_calls'] == {'q4': [1, 0], 'q2': [1, 0], 'q1': [1, 0]}
    assert r['weight_served']['q2'] == pytest.approx(0.3) and r['weight_served']['q1'] == pytest.approx(0.2)
    assert r['pools']['q2']['per_layer']['objects'] == [1, 0]
    call(s, 0, [1, 2], [0.5, 0.5])                    # both pool hits
    assert len(s.reads) == 3
    assert s.q2_tier_active('blk.0')


def test_heavy_reduced_selection_reads_original_transiently(small, monkeypatch):
    s = tiered(monkeypatch, [{'layers': [0, 1], 'only': 'q1'}], max_weight=0.25)
    call(s, 0, [3, 4], [0.9, 0.1])
    assert s.reads == [((0, 3), 'read_tensors'), ((0, 4), 'q1_read')]
    assert (0, 3) not in s._cache                      # transient: never cached
    assert s.q1_on_miss.stats['original_transient_reads'] == 1


def test_rank_tiers_choose_precision_per_call(small, monkeypatch):
    s = tiered(monkeypatch, [{'layers': [0, 1], 'only': 'q1'},
                             {'layers': 1, 'tier': 'q4', 'experts': [7]},
                             {'layers': 1, 'rank_tiers': {'q4': 1, 'q2': 0, 'rest': 'q1'}}])
    call(s, 1, [7, 8], [0.6, 0.4])     # rank 1 q4-mapped -> cached original; rank 2 -> q1
    call(s, 1, [8, 7], [0.7, 0.3])     # 8 rank 1, not q4-mapped -> transient original; 7 resident
    assert s.reads == [((1, 7), 'read_tensors'), ((1, 8), 'q1_read'), ((1, 8), 'read_tensors')]
    assert (1, 7) in s._cache and (1, 8) not in s._cache and 8 in s.q1_on_miss.pools['q1'].layers[1]
    assert s.q1_on_miss.receipt()['per_layer_served_calls']['q4'] == [0, 3]


def test_moe_routes_serves_q2_objects_without_fused_down(monkeypatch):
    import mlx.core as mx
    from tests.test_deepseek_v41_source_batch_provider_cpu import _case
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        case = _case(monkeypatch, ids=[[9, 1, 7, 2, 6, 0]])
        w = mx.zeros((2304, 5120), mx.float16); d = mx.zeros((5120, 2304), mx.float16)
        arrays = {}
        for kind, value in (('gate', w), ('up', w), ('down', d)):
            q, sc, b = mx.quantize(value, group_size=64, bits=2)
            arrays.update({kind + '.q2weight': q, kind + '.q2scales': sc, kind + '.q2biases': b})
        for e in list(case.bank):
            case.bank[e] = q1m.Q2Expert(arrays)
        case.store.q2_tier_active = lambda prefix: True
        from mlx2.runtime.gguf import source_middle_fusion
        seen = []
        monkeypatch.setattr(source_middle_fusion, 'middle',
                            lambda gate, up, scores, slots: seen.append(gate.shape) or gate[:, :2304])
        output = case.w.moe(case.x, case.ids, case.scores, 'blk.0'); mx.eval(output)
        assert case.w.scheduled_expert_calls['official_source_q2_tier_experts'] == 6
        assert case.calls['down'] == [] and case.calls['batch'] == []
        assert output.shape == (1, 5120) and seen == [(1, 2304)] * 6
    finally:
        mx.set_default_device(previous)


# ---- Review fixes: reservations, call-scoped leases, completion -------------

def test_fixed_mode_reserve_and_resume_include_the_pools(small):
    s = pooled(pool_layers=(2, 2))
    s._adaptive_residency_enabled = False                   # legacy fixed reservation contract
    base = s.cache_budget_bytes + s.plan.transient_bound_bytes
    assert s.reserved_bytes() == base + 4 * Q
    assert s.resume_reservation_bytes() == base + 4 * Q
    assert s.resume_reservation_bytes() == s.reserved_bytes()


def test_served_refs_end_with_the_call_and_park_clears_everything(small):
    s = store()                                              # uncached Q1 (no pool)
    ids = np.array([[1, 2]], np.uint32)
    with s.window():
        with s.q1_on_miss_scope('blk.0', ids, np.array([[0.97, 0.03]], np.float32)):
            s.acquire('blk.0', 2)
            assert (0, 2) in s.q1_on_miss.served
        assert s.q1_on_miss.served == {} and (0, 2) in s.q1_on_miss.served_ids
        assert s._q1_pool_unleased_bytes() == 0              # call done: no refs held by the rule
    s.end_context()
    p = pooled()
    call(p, 0, [9, 1], [0.99, 0.01])
    p.park()
    rule = p.q1_on_miss
    assert rule.pool_live == 0 and rule.served == {} and rule.served_ids == {} and rule.armed is None


def test_pools_under_bounded_completion_report_every_lease(small):
    s = pooled(pool_layers=(2, 2))
    collected = []
    s._completion_collector = collected.append
    call(s, 0, [9, 1], [0.99, 0.01])                         # original miss + Q1 bank read
    call(s, 0, [9, 1], [0.99, 0.01])                         # original hit + Q1 pool hit
    assert collected == [(0, 9), (0, 1), (0, 9), (0, 1)]
    assert s.q1_on_miss.pools['q1'].stats['hits'][0] == 1


def test_main_route_drop_composes_with_reduced_tiers():
    from mlx2.adapters.deepseek_v41_source import source_policy
    from mlx2.adapters.deepseek_v41_source_approx import DROP_KEY, DROP_SPANS
    assert source_policy({DROP_KEY: 'drop_min_v1'})[DROP_KEY] == 'drop_min_v1'   # alone: main's option works
    p = source_policy({'deepseek_source_experts': experts(), DROP_KEY: 'drop_min_v1',
                       DROP_SPANS: [{'layers': [13, 26], 'count': 2}]})
    assert p[DROP_KEY] == 'drop_min_v1' and 'q1_on_miss' in p['deepseek_source_experts']


def test_route_drop_redirects_collapse_before_rank_tiers(small, monkeypatch):
    # route_drop redirects a dropped slot to the row's top expert with weight 0. arm() must
    # collapse it, rank only the kept experts, and never read the dropped expert.
    s = tiered(monkeypatch, [{'layers': [0, 1], 'only': 'q1'},
                             {'layers': 1, 'tier': 'q4', 'experts': [7]},
                             {'layers': 1, 'rank_tiers': {'q4': 1, 'q2': 1, 'rest': 'q1'}}])
    call(s, 1, [7, 8, 9, 7], [0.5, 0.3, 0.2, 0.0])     # slot 4 was dropped expert 10 -> now 7, weight 0
    assert s.reads == [((1, 7), 'read_tensors'), ((1, 8), 'q2_read'), ((1, 9), 'q1_read')]
    stats = s.q1_on_miss.stats
    assert stats['dropped_slots_collapsed'] == 1 and stats['invalid_route_calls'] == 0
    calls = s.q1_on_miss.receipt()['per_layer_served_calls']
    assert calls['q2'] == [0, 1] and calls['q1'] == [0, 1]


def test_nonzero_duplicate_ids_still_serve_exact(small, monkeypatch):
    s = tiered(monkeypatch, [{'layers': [0, 1], 'only': 'q1'}])
    call(s, 1, [7, 7], [0.5, 0.5])
    assert s.q1_on_miss.stats['invalid_route_calls'] == 1


def test_composed_operations_share_one_private_profile(monkeypatch):
    from types import SimpleNamespace
    from mlx2.adapters.deepseek_v41_source_approx import stamp_cache
    rule = SimpleNamespace(salt='a' * 64)
    monkeypatch.setattr(q1m, 'store_rule', lambda adapter: rule)
    monkeypatch.setattr(q1m, 'enabled', lambda policy: True)
    model = SimpleNamespace(source_approx_private_profile='mlx2.ds41.source-approx.v1:drop')
    q1m.configure_model(SimpleNamespace(kernel_policy={}, model=model))
    assert model.source_approx_private_profile == model.source_combined_private_profile
    assert model.source_combined_private_profile.startswith('mlx2.ds41.source-approx.v1:drop+')

    class Row:
        ast_private_profile = None
    rows = [Row()]
    stamp_cache(rows, model.source_approx_private_profile)          # route drop stamps first
    stamp_cache(rows, model.source_combined_private_profile)        # tiers stamp the same: no conflict
    alone = SimpleNamespace()
    q1m.configure_model(SimpleNamespace(kernel_policy={}, model=alone))
    assert not hasattr(alone, 'source_combined_private_profile')


def test_strict_rank_cap_ignores_resident_original(small, monkeypatch):
    # "Q2 as the highest quant": with strict rank tiers, an unleased resident original
    # does not override the requested reduced tier.
    s = tiered(monkeypatch, [{'layers': [0, 1], 'only': 'q1'},
                             {'layers': 1, 'tier': 'q4', 'experts': [7]},
                             {'layers': 1, 'rank_tiers': {'q4': 1, 'q2': 0, 'rest': 'q1'}}])
    call(s, 1, [7, 8], [0.6, 0.4])                     # 7 becomes a resident original
    assert (1, 7) in s._cache
    s2 = tiered(monkeypatch, [{'layers': [0, 1], 'only': 'q1'},
                              {'layers': 1, 'rank_tiers': {'q4': 0, 'q2': 6, 'rest': 'q1', 'strict': True}}])
    s2._cache[(1, 7)] = s._cache[(1, 7)]               # resident original on the capped layer
    s2.reads.clear()
    call(s2, 1, [7], [1.0])
    assert s2.reads == [((1, 7), 'q2_read')]
    assert s2.q1_on_miss.stats['strict_cap_bypassed_resident'] == 1


def test_strict_flag_validation():
    with pytest.raises(ValueError):
        tiers.resolve(budget_bytes=90 * GB, totals={'q2': GB, 'q1': GB}, usage=USAGE, tiers=('q4', 'q2', 'q1'),
                      overrides=[{'layers': 1, 'rank_tiers': {'q4': 0, 'q2': 6, 'rest': 'q1', 'strict': 'yes'}}])
