"""Route bias composed with route drop and reduced expert tiers (CPU, synthetic only).

No checkpoint, GPU or timing: policy parsing, the router chain against a NumPy
oracle (native -> bias -> drop -> redirect), residency including reduced pools,
and the identity contract (composed salts distinct, each operation alone
byte-for-byte the binding it always had).
"""
import hashlib
import json
import threading
from collections import OrderedDict
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_deepseek_v41_cpu import Weights, config

from mlx2.adapters import deepseek_v41_source_approx as approx
from mlx2.adapters import deepseek_v41_source_q1_on_miss as q1m
from mlx2.adapters import deepseek_v41_source_route_bias as rb
from mlx2.adapters import deepseek_v41_source_route_drop as rd
from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.adapters.deepseek_v41_source_approx import (
    DROP_COUNT, DROP_KEY, DROP_LAYERS, DROP_MAX_WEIGHT, DROP_PROTECT, ROUTE_ADAPTIVE, ROUTE_KEY, ROUTE_LAMBDA,
    approximate_model_class,
)
from mlx2.adapters.deepseek_v41_source_route_bias import ResidentRouteBias, reference_selection, resident_mask
from mlx2.adapters.deepseek_v41_source_route_drop import RankDrop, reference_drop, reference_redirect
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model

LFU = {'mode': 'original_lfu', 'budget_bytes': 75_000_000_000}
TIERED = dict(LFU, q1_root='/models/q1-bank', q1_on_miss={'max_weight': 0.10, 'layers': [0, 39]})


class Store:
    def __init__(self, layers, experts, resident=(), pools=None):
        self._lock = threading.RLock()
        self.tables = {f'blk.{i}': tuple((i, e) for e in range(experts)) for i in range(layers)}
        self._cache = OrderedDict((key, object()) for key in resident)
        self._leased = {}
        if pools is not None:
            self.q1_on_miss = SimpleNamespace(pools={
                name: SimpleNamespace(layers=[dict.fromkeys(held.get(i, ())) for i in range(layers)])
                for name, held in pools.items()})


# ---------------------------------------------------------------- policy

@pytest.mark.parametrize('extra', [
    {DROP_KEY: 'drop_min_v1'},
    {DROP_KEY: 'drop_miss_v1', DROP_PROTECT: 2, DROP_MAX_WEIGHT: 0.2},
])
def test_route_bias_and_drop_parse_together(extra):
    p = source_policy({'deepseek_source_experts': LFU, ROUTE_KEY: 'resident_guarded_v1', **extra})
    assert p[ROUTE_KEY] == 'resident_guarded_v1' and p[DROP_KEY] == extra[DROP_KEY]
    assert approx.selected_operations(p) == ('route_resident_guarded_v1', 'route_' + extra[DROP_KEY])
    assert approx.composition(p) == {'router_chain': approx.ROUTE_DROP_COMPOSITION}


def test_route_bias_with_tiers_and_all_three_parse():
    p = source_policy({'deepseek_source_experts': TIERED, ROUTE_KEY: 'resident_v1'})
    assert q1m.enabled(p) and p[ROUTE_KEY] == 'resident_v1'
    assert approx.composition(p) == {'route_bias_residency': approx.ROUTE_TIERS_RESIDENCY}
    p3 = source_policy({'deepseek_source_experts': TIERED, ROUTE_KEY: 'resident_v1', DROP_KEY: 'drop_miss_v1'})
    assert q1m.enabled(p3) and approx.composition(p3) == {
        'router_chain': approx.ROUTE_DROP_COMPOSITION, 'route_bias_residency': approx.ROUTE_TIERS_RESIDENCY}
    assert approx.apc_refusal_reason(p3) == (approx.APC_REASON_PREFIX
                                             + 'route_resident_v1+route_drop_miss_v1')
    assert q1m.apc_refusal_reason(p3) == q1m.APC_REASON


@pytest.mark.parametrize('extra', [
    {'deepseek_physical_batch': 'b2'},
    {'draft_model': 'embedded_dspark', 'num_draft': 3},
    {'deepseek_source_resident_gather': 'resident_v1'},
])
def test_composed_routes_still_refuse_uncomposed_laws(extra):
    with pytest.raises(ValueError):
        source_policy({'deepseek_source_experts': LFU, ROUTE_KEY: 'resident_v1', DROP_KEY: 'drop_min_v1', **extra})


@pytest.mark.parametrize('extra', [
    {'deepseek_source_expert_io': 'overlap_v1'},
    {'deepseek_source_prefill_readahead': 'layer_ahead_v1'},
    {'deepseek_physical_batch': 'b2'},
])
def test_bias_with_tiers_still_refuses_uncomposed_laws(extra):
    with pytest.raises(ValueError):
        source_policy({'deepseek_source_experts': TIERED, ROUTE_KEY: 'resident_v1', **extra})


def test_route_bias_still_requires_an_expert_store():
    with pytest.raises(ValueError, match='deepseek_source_experts'):
        source_policy({ROUTE_KEY: 'resident_v1', DROP_KEY: 'drop_min_v1'})


# ---------------------------------------------------------------- residency

def test_bias_residency_counts_reduced_pool_members():
    plain = Store(2, 8, [(0, 1), (1, 2)])
    tiered = Store(2, 8, [(0, 1), (1, 2)], pools={'q1': {0: (4,)}, 'q2': {0: (6,), 1: (7,)}})
    args = SimpleNamespace(n_routed_experts=8)
    alone = ResidentRouteBias(mode='resident_v1', store=plain, args=args, lam=0.35)
    both = ResidentRouteBias(mode='resident_v1', store=tiered, args=args, lam=0.35)
    for layer in (0, 1):   # no tiers bound: exactly the original-cache/lease mask
        np.testing.assert_array_equal(alone.residency(layer), resident_mask(plain, layer, 8))
    assert np.flatnonzero(both.residency(0)).tolist() == [1, 4, 6]
    assert np.flatnonzero(both.residency(1)).tolist() == [2, 7]
    np.testing.assert_array_equal(both.residency(0), rd.resident_or_pooled(tiered, 0, 8))


# ---------------------------------------------------------------- router chain

E, K, SCALE = 16, 6, 1.5


def _drop_miss_oracle(ids, w, resident, *, count, protect, max_weight):
    w = np.asarray(w, np.float64).copy()
    total = w.sum()
    rank = np.argsort(np.argsort(-w, kind='stable'), kind='stable')
    cand = [i for i in range(len(w)) if rank[i] >= protect and not resident[ids[i]] and w[i] < max_weight * total]
    for i in sorted(cand, key=lambda i: w[i])[:count]:
        w[i] = 0.0
    return w * (total / w.sum())


def _chain_block(logits, correction, store, *, bias, drop):
    c = SimpleNamespace(n_routed_experts=E, num_experts_per_tok=K, routed_scaling_factor=SCALE, norm_topk_prob=True)
    model = SimpleNamespace(source_route_bias=bias, source_route_drop=drop, pointwise_policy='reference',
                            linear=lambda x, name, rounded=False: mx.array(logits[None, None]))
    block = SimpleNamespace(model=model, layer=0, c=c, prefix='blk.0.', correction=mx.array(correction))
    model.layers = [block]
    bias.args = c
    return model, block


def _pairs(ids, w):
    ids, w = np.asarray(ids).reshape(-1), np.asarray(w, np.float64).reshape(-1)
    return sorted(zip(ids.tolist(), w.tolist()))


def _assert_route(got, want, case):
    assert [i for i, _ in got] == [i for i, _ in want], case
    np.testing.assert_allclose([v for _, v in got], [v for _, v in want], rtol=1e-5, atol=1e-6, err_msg=str(case))


@pytest.mark.parametrize('drop_mode', ['drop_min_v1', 'drop_miss_v1'])
@pytest.mark.parametrize('order', ['bias_first', 'drop_first'])
def test_router_chain_matches_oracle_bias_then_drop(monkeypatch, drop_mode, order):
    def unexpected(block):
        return lambda x, image_mask=None: pytest.fail('B1 rows must not take the native router twice')
    monkeypatch.setattr(rb, '_native_router', unexpected)
    monkeypatch.setattr(rd, '_native_router', unexpected)
    rng = np.random.default_rng(11)
    checked = 0
    for case in range(120):
        logits = rng.normal(0, 2.0, E).astype(np.float32)
        correction = rng.normal(0, 0.1, E).astype(np.float32)
        resident_ids = [e for e in range(E) if rng.random() < 0.4]
        pooled_ids = [e for e in range(E) if rng.random() < 0.2]
        store = Store(1, E, [(0, e) for e in resident_ids], pools={'q1': {0: pooled_ids}})
        resident = np.zeros(E, bool); resident[resident_ids] = True; resident[pooled_ids] = True
        lam = 1.0
        scores = np.sqrt(np.log1p(np.exp(logits.astype(np.float64))))
        chosen, w, selection = reference_selection(scores, correction, resident, lam, mode='resident_v1',
                                                   top_k=K, scale=SCALE, return_selection=True)
        if min(v[K - 1] - v[K] for v in (np.sort(scores + correction)[::-1], np.sort(selection)[::-1])) < 1e-4:
            continue
        if drop_mode == 'drop_min_v1':
            dropped = reference_drop(w, 2)
        else:
            dropped = _drop_miss_oracle(chosen, w, resident, count=2, protect=2, max_weight=0.25)
        want = _pairs(reference_redirect(chosen, dropped), dropped)
        bias = ResidentRouteBias(mode='resident_v1', store=store, args=None, lam=lam)
        drop = RankDrop(mode=drop_mode, layers=[0, 0], count=2, store=store, protect=2, max_weight=0.25)
        model, block = _chain_block(logits, correction, store, bias=bias, drop=drop)
        installs = (rb.install_router_wrappers, rd.install_router_wrappers)
        for install in (installs if order == 'bias_first' else installs[::-1]) * 2:   # idempotent
            install(model)
        ids, got = block.router(mx.zeros((1, 1, 4)))
        _assert_route(_pairs(ids, got), want, case)
        assert bias.stats['router_calls'] == 1 and drop.stats['dropped_layer_rows'] == 1
        checked += 1
    assert checked > 50


def test_chain_keeps_native_rows_native_and_counts_once(monkeypatch):
    calls = []
    native = (mx.array([[[0, 1, 2, 3, 4, 5]] * 3], mx.uint32), mx.full((1, 3, 6), 0.25, mx.float32))

    def recorder(block):
        def router(x, image_mask=None):
            calls.append(x.shape)
            return native
        return router
    monkeypatch.setattr(rb, '_native_router', recorder)
    monkeypatch.setattr(rd, '_native_router', recorder)
    store = Store(1, E)
    bias = ResidentRouteBias(mode='resident_v1', store=store, args=None, lam=1.0)
    drop = RankDrop(mode='drop_min_v1', layers=[0, 0])
    model, block = _chain_block(np.zeros(E, np.float32), np.zeros(E, np.float32), store, bias=bias, drop=drop)
    rd.install_router_wrappers(model); rb.install_router_wrappers(model)
    out = block.router(mx.zeros((1, 3, 4)))
    assert out is native and calls == [(1, 3, 4)]               # one native call for a prefill chunk
    assert bias.stats['native_rows'] == 3 and drop.stats['native_rows'] == 3
    assert bias.stats['router_calls'] == 0 and drop.stats['dropped_layer_rows'] == 0


def test_composed_model_decodes_stamps_and_reports_both(monkeypatch):
    c = config()
    store = Store(c.num_hidden_layers, c.n_routed_experts, [(i, 0) for i in range(c.num_hidden_layers)])
    policy = {ROUTE_KEY: 'resident_v1', ROUTE_ADAPTIVE: 'off', ROUTE_LAMBDA: 3.0,
              DROP_KEY: 'drop_min_v1', DROP_LAYERS: [0, c.num_hidden_layers - 1], DROP_COUNT: 1}
    model = approximate_model_class(DeepSeekV41Model, policy)(c, Weights(c), identity='synthetic-route@1')
    model.source_approx_private_profile = approx.private_profile(policy)
    model.source_route_bias = ResidentRouteBias.from_policy(policy, store, c)
    model.source_route_drop = RankDrop.from_policy(policy, store=store)
    cache = model.make_cache()
    model(mx.array([[1, 2, 3, 4]], mx.int32), cache)
    for token in (5, 6):
        out = model(mx.array([[token]], mx.int32), cache)
        assert np.isfinite(np.asarray(out.astype(mx.float32))).all()
    bias, drop = model.source_route_bias, model.source_route_drop
    assert bias.stats['router_calls'] == 2 * c.num_hidden_layers
    assert drop.stats['dropped_layer_rows'] == 2 * c.num_hidden_layers
    assert all(row.ast_private_profile == approx.private_profile(policy) for row in cache)
    receipt = approx.receipt(policy, model)
    assert receipt['route_bias']['stats']['router_calls'] == 2 * c.num_hidden_layers
    assert receipt['route_drop']['stats']['decode_forwards'] == 2
    assert receipt['composition'] == {'router_chain': approx.ROUTE_DROP_COMPOSITION}
    from mlx2.runtime.models.deepseek_v41 import Block
    model.layers = [Block(model, i) for i in range(c.num_hidden_layers)]     # restore rebuilds blocks
    model(mx.array([[7]], mx.int32), cache)
    assert bias.stats['router_calls'] == 3 * c.num_hidden_layers
    assert drop.stats['dropped_layer_rows'] == 3 * c.num_hidden_layers


# ---------------------------------------------------------------- identity

BASE = ['adapters/deepseek_v41_source_approx.py', 'runtime/models/deepseek_v41.py',
        'runtime/models/deepseek_v41_cache.py']
BIAS_FILES = ['adapters/deepseek_v41_source_route_bias.py', 'runtime/models/deepseek_v4.py',
              'runtime/models/deepseek_v41_fused.py']
DROP_FILES = ['adapters/deepseek_v41_source_route_drop.py', 'runtime/models/deepseek_v4.py',
              'runtime/models/deepseek_v41_fused.py']


def _historical_salt(policy, files):
    """The binding every operation used before composition existed."""
    binding = {key: policy[key] for key in sorted(approx.POLICY_KEYS) if key in policy}
    binding['implementation'] = approx._file_identity(BASE + files)
    binding['qualification'] = 'unqualified'
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@pytest.mark.parametrize('raw,files', [
    ({'deepseek_source_experts': LFU, ROUTE_KEY: 'resident_v1'}, BIAS_FILES),
    ({'deepseek_source_experts': LFU, ROUTE_KEY: 'resident_guarded_v1', ROUTE_LAMBDA: 1.0}, BIAS_FILES),
    ({DROP_KEY: 'drop_min_v1'}, DROP_FILES),
    ({'deepseek_source_experts': LFU, DROP_KEY: 'drop_miss_v1'}, DROP_FILES),
    ({'deepseek_source_experts': TIERED, DROP_KEY: 'drop_min_v1'}, DROP_FILES),   # existing drop+tiers
])
def test_each_operation_alone_keeps_its_salt_and_profile(raw, files):
    policy = source_policy(raw)
    assert approx.composition(policy) == {}
    salt = approx.identity_salt(policy)
    assert salt == _historical_salt(policy, files)
    assert approx.private_profile(policy) == (approx.PRIVATE_PROFILE_PREFIX
                                              + '+'.join(approx.selected_operations(policy)) + ':' + salt[:32])


def test_composed_salts_are_distinct_and_cover_every_parameter():
    bias = source_policy({'deepseek_source_experts': LFU, ROUTE_KEY: 'resident_v1'})
    drop = source_policy({'deepseek_source_experts': LFU, DROP_KEY: 'drop_min_v1'})
    both = source_policy({'deepseek_source_experts': LFU, ROUTE_KEY: 'resident_v1', DROP_KEY: 'drop_min_v1'})
    both_l = source_policy({'deepseek_source_experts': LFU, ROUTE_KEY: 'resident_v1', ROUTE_LAMBDA: 0.5,
                            DROP_KEY: 'drop_min_v1'})
    both_c = source_policy({'deepseek_source_experts': LFU, ROUTE_KEY: 'resident_v1', DROP_KEY: 'drop_min_v1',
                            DROP_COUNT: 2})
    tiers = source_policy({'deepseek_source_experts': TIERED, ROUTE_KEY: 'resident_v1'})
    three = source_policy({'deepseek_source_experts': TIERED, ROUTE_KEY: 'resident_v1', DROP_KEY: 'drop_min_v1'})
    salts = [approx.identity_salt(p) for p in (bias, drop, both, both_l, both_c, tiers, three)]
    assert len(set(salts)) == len(salts)
    profile = approx.private_profile(both)
    assert profile.startswith(approx.PRIVATE_PROFILE_PREFIX + 'route_resident_v1+route_drop_min_v1:')
    # Bias + tiers hashes the residency helper it executes.
    assert approx.implementation_identity(tiers) != approx.implementation_identity(bias)


def test_bias_with_tiers_shares_one_combined_profile_and_salt(monkeypatch):
    policy = source_policy({'deepseek_source_experts': TIERED, ROUTE_KEY: 'resident_v1', DROP_KEY: 'drop_min_v1'})
    rule = SimpleNamespace(salt='b' * 64)
    monkeypatch.setattr(q1m, 'store_rule', lambda adapter: rule)
    model = SimpleNamespace(source_approx_private_profile=approx.private_profile(policy))
    q1m.configure_model(SimpleNamespace(kernel_policy=policy, model=model))
    combined = model.source_combined_private_profile
    assert combined == approx.private_profile(policy) + '+' + q1m.private_profile(rule.salt)
    assert model.source_approx_private_profile == combined     # bias, drop and tiers all stamp this
    assert model.source_combined_identity_salt == q1m.combined_identity_salt(approx.identity_salt(policy), rule.salt)
    receipt = approx.receipt(policy, model)
    assert receipt['combined_private_cache_profile'] == combined
    assert receipt['combined_identity_salt'] == model.source_combined_identity_salt
    assert receipt['route_bias']['residency'] == 'original cache, lease or reduced pool'

    class Row:
        ast_private_profile = None
    rows = [Row()]
    approx.stamp_cache(rows, model.source_approx_private_profile)
    approx.stamp_cache(rows, combined)
    assert rows[0].ast_private_profile == combined
