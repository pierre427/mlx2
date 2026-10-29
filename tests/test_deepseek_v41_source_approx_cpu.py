"""Approximate DS4.1 Source operations (YOCO prefill skip, resident route bias).

Synthetic CPU contracts only: no checkpoint, GPU or timing.
"""
import threading
from collections import OrderedDict
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_deepseek_v41_cpu import Weights, config

from mlx2.adapters import deepseek_v41_source_approx as approx
from mlx2.adapters.deepseek_v41_mlx import DeepSeekV41MLXAdapter
from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter, source_policy
from mlx2.adapters.deepseek_v41_source_approx import (
    APC_REASON_PREFIX,
    ROUTE_ADAPTIVE,
    ROUTE_KEY,
    ROUTE_LAMBDA,
    ROUTE_MARGIN,
    YOCO_KEEP,
    YOCO_KEY,
    YOCO_LAYERS,
    YOCO_MIN_ROWS,
    approximate_model_class,
)
from mlx2.adapters.deepseek_v41_source_route_bias import (
    ResidentRouteBias,
    _margin_lambda_mx,
    biased_selection,
    install_router_wrappers,
    margin_lambda,
    reference_selection,
    resident_mask,
)
from mlx2.adapters.deepseek_v41_source_yoco import (
    configure_yoco,
    exact_reference_prefill,
    partition,
    plan_layers,
)
from mlx2.runtime.models.deepseek_v4 import route_experts
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model

EXPERTS = {'mode': 'original_lfu', 'budget_bytes': 1}


def yoco_config(**kw):
    values = dict(num_hidden_layers=6, compress_ratios=(0, 2, 2, 1, 1, 1), kv_source_layer_ids=(1, 3),
                  index_source_layer_ids=(1, 3, 4), candidate_source_layer_id=3, sliding_window=4)
    values.update(kw)
    return config(**values)


def yoco_pair(c, *, keep, min_rows, chunk=3):
    policy = {YOCO_KEY: 'yoco_v1', YOCO_LAYERS: [4, 5], YOCO_KEEP: keep, YOCO_MIN_ROWS: min_rows}
    model = approximate_model_class(DeepSeekV41Model, policy)(c, Weights(c), identity='synthetic-yoco@1')
    model.recommended_prefill_rows = lambda maximum, context_tokens=0: min(maximum, chunk)
    model.source_approx_private_profile = approx.private_profile(policy)
    configure_yoco(model, policy)
    ref = DeepSeekV41Model(c, Weights(c), identity='synthetic-yoco@1')
    return model, ref, policy


def chunk_major(ref, ids, cache, chunk):
    out = None
    for s in range(0, ids.shape[1], chunk):
        final = s + chunk >= ids.shape[1]
        out = ref(ids[:, s:s + chunk], cache, logits_to_keep=1 if final else None)
    return out[:, -1:]


def prompt(rows, c):
    rng = np.random.default_rng(rows)
    return mx.array(rng.integers(0, c.vocab_size, (1, rows)), mx.int32)


def _f(a):
    return np.asarray(a.astype(mx.float32))


# ---------------------------------------------------------------- YOCO

def test_yoco_reduces_to_exact_when_tail_covers_the_dependency_cone():
    """Falsifier: keep = W + (W-1) covers the exact CED cone for 2 decoders."""
    c = yoco_config()
    model, ref, _ = yoco_pair(c, keep=7, min_rows=10)
    ids = prompt(23, c)
    cache, rc = model.make_cache(), ref.make_cache()
    actual = model(ids, cache)
    expected = chunk_major(ref, ids, rc, 3)
    assert model.source_prefill_approx_last['admitted'] is True
    np.testing.assert_allclose(_f(actual), _f(expected), rtol=1e-5, atol=1e-6)
    for a, b in zip(cache, rc):
        assert a.offset == b.offset == 23
        np.testing.assert_allclose(_f(a.keys), _f(b.keys), rtol=1e-5, atol=1e-6)
    nxt = mx.array([[5]], mx.int32)
    np.testing.assert_allclose(_f(model(nxt, cache)), _f(ref(nxt, rc)), rtol=1e-5, atol=1e-6)


def test_yoco_short_tail_is_a_real_approximation_and_skips_decoder_reads():
    c = yoco_config()
    model, ref, policy = yoco_pair(c, keep=4, min_rows=10)
    seen = []
    original = model.weights.moe

    def moe(x, ids, scores, prefix, **kw):
        seen.append((prefix, x.shape[0]))
        return original(x, ids, scores, prefix, **kw)
    model.weights.moe = moe
    ids = prompt(23, c)
    cache, rc = model.make_cache(), ref.make_cache()
    actual = model(ids, cache)
    expected = chunk_major(ref, ids, rc, 3)
    assert np.all(np.isfinite(_f(actual)))
    assert not np.allclose(_f(cache[5].keys), _f(rc[5].keys), atol=1e-6)
    # Encoder caches are exact; the first decoder's KV is a projection of exact inputs.
    for i in range(5):
        np.testing.assert_allclose(_f(cache[i].keys), _f(rc[i].keys), rtol=1e-5, atol=1e-6)
    decoder_rows = [rows for prefix, rows in seen if prefix in ('blk.4', 'blk.5')]
    assert decoder_rows == [4, 4]
    assert sum(rows for prefix, rows in seen if prefix == 'blk.0') == 23
    last = model.source_prefill_approx_last
    assert last['skipped_decoder_layer_rows'] == (23 - 4) * 2 and last['qualification'] == 'unqualified'
    assert {row.offset for row in cache} == {23}
    assert all(row.ast_private_profile == approx.private_profile(policy) for row in cache)
    assert not any(row._in_forward for row in cache)
    del expected


def test_yoco_admission_refusals_run_exact_ordinary():
    c = yoco_config()
    model, ref, _ = yoco_pair(c, keep=4, min_rows=10)
    ids = prompt(9, c)  # below the minimum
    cache, rc = model.make_cache(), ref.make_cache()
    np.testing.assert_array_equal(_f(model(ids, cache)), _f(ref(ids, rc)))
    assert model.source_prefill_approx_last is None
    assert all(row.ast_private_profile is None for row in cache)
    # All-row logits always run full depth.
    cache = model.make_cache()
    out = model._forward(prompt(12, c), cache, logits_to_keep=0)[0]
    assert out.shape[1] == 12 and all(row.ast_private_profile is None for row in cache)


def test_wide_call_below_minimum_uses_exact_chunk_sequence():
    c = yoco_config()
    model, ref, _ = yoco_pair(c, keep=4, min_rows=2000, chunk=512)
    ids = prompt(1030, c)
    cache, rc = model.make_cache(), ref.make_cache()
    actual = model(ids, cache)
    expected = chunk_major(ref, ids, rc, 512)
    np.testing.assert_array_equal(_f(actual), _f(expected))
    for a, b in zip(cache, rc):
        assert a.offset == b.offset and a.ast_private_profile is None
        for p, q in zip(a.state, b.state):
            assert (p is None) == (q is None)
            if p is not None:
                np.testing.assert_array_equal(np.asarray(p), np.asarray(q))
    assert model.scheduled_kv_calls['source_yoco_exact_fallback_waves'] == 1
    assert model.source_prefill_approx_last['reason'] == 'prompt_below_minimum_rows'
    control = model.make_cache()
    np.testing.assert_array_equal(_f(exact_reference_prefill(model, ids, control)), _f(expected))


def test_partition_puts_whole_tail_in_final_sub_chunk():
    for rows, chunk, keep in ((5000, 1024, 140), (4097, 1024, 140), (4096, 1024, 140), (2100, 1000, 1024)):
        bounds, first = partition(rows, chunk, keep)
        assert bounds[0][0] == 0 and bounds[-1][1] == rows
        assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
        assert all(1 <= e - s <= 1024 for s, e in bounds)
        assert bounds[-1][1] - bounds[-1][0] - first == keep
    with pytest.raises(ValueError):
        partition(100, 1024, 140)


def test_plan_layers_fail_closed_on_production_geometry():
    from mlx2.runtime.models.deepseek_v41_config import DeepSeekV41Config
    args = DeepSeekV41Config()
    assert plan_layers(args, [21, 39], 140) == (21, 39)
    for layers, keep, match in (([20, 39], 140, 'global KV pool'), ([21, 38], 140, 'final layer'),
                                ([14, 39], 140, 'global KV'), ([21, 39], 100, 'sliding window')):
        with pytest.raises(ValueError, match=match):
            plan_layers(args, layers, keep)


# ---------------------------------------------------------------- policy

def test_policy_default_off_and_parameters():
    default = source_policy()
    assert not set(default) & approx.POLICY_KEYS
    assert source_policy({YOCO_KEY: 'off'}) == default
    value = source_policy({YOCO_KEY: 'yoco_v1'})
    assert value[YOCO_LAYERS] == [21, 39] and value[YOCO_KEEP] == 140 and value[YOCO_MIN_ROWS] == 4096
    value = source_policy({YOCO_KEY: 'yoco_v1', YOCO_LAYERS: '22-39', YOCO_KEEP: 256, YOCO_MIN_ROWS: 8192})
    assert value[YOCO_LAYERS] == [22, 39] and value[YOCO_KEEP] == 256
    for bad in ({YOCO_KEY: 'yoco'}, {YOCO_KEEP: 140}, {YOCO_KEY: 'yoco_v1', YOCO_KEEP: 0},
                {YOCO_KEY: 'yoco_v1', YOCO_KEEP: 2048}, {YOCO_KEY: 'yoco_v1', YOCO_MIN_ROWS: 1024},
                {YOCO_KEY: 'yoco_v1', YOCO_LAYERS: [30, 20]}, {YOCO_KEY: 'yoco_v1', YOCO_KEEP: True},
                {ROUTE_KEY: 'resident_v1'}, {ROUTE_LAMBDA: 0.3}):
        with pytest.raises(ValueError):
            source_policy(bad)


def test_yoco_refuses_incompatible_routes():
    for extra in ({'deepseek_source_prefill_order': 'layer_major_v1'}, {'deepseek_ced': 'dependency_exact'},
                  {'deepseek_vision': 'enabled'}, {'draft_model': 'prompt_lookup'},
                  {'deepseek_kv_storage': 'native_packed', 'deepseek_candidate_score': 'candidate_only',
                   'deepseek_source_attention_math': 'all_retained_broadcast',
                   'deepseek_index_policy': 'finite_all_retained'},
                  {'deepseek_source_prefill_partition': 'canonical_m1_v1'}):
        with pytest.raises(ValueError):
            source_policy({YOCO_KEY: 'yoco_v1', **extra})


def test_route_policy_validation():
    base = {'deepseek_source_experts': dict(EXPERTS)}
    value = approx.validate_policy({ROUTE_KEY: 'resident_guarded_v1'}, base)
    assert value == {ROUTE_KEY: 'resident_guarded_v1', ROUTE_ADAPTIVE: 'off', ROUTE_LAMBDA: 0.35}
    value = approx.validate_policy({ROUTE_KEY: 'resident_v1', ROUTE_ADAPTIVE: 'margin_v1'}, base)
    assert value[ROUTE_MARGIN] == approx.ROUTE_DEFAULT_MARGIN and ROUTE_LAMBDA not in value
    for raw in ({ROUTE_KEY: 'resident_v1', ROUTE_LAMBDA: 0.0}, {ROUTE_KEY: 'resident_v1', ROUTE_LAMBDA: 4.0},
                {ROUTE_KEY: 'resident_v1', ROUTE_LAMBDA: float('nan')},
                {ROUTE_KEY: 'resident_v1', ROUTE_ADAPTIVE: 'margin_v1', ROUTE_LAMBDA: 0.2},
                {ROUTE_KEY: 'resident_v1', ROUTE_MARGIN: dict(approx.ROUTE_DEFAULT_MARGIN)},
                {ROUTE_KEY: 'resident_v1', ROUTE_ADAPTIVE: 'margin_v1',
                 ROUTE_MARGIN: dict(minimum=0.5, maximum=0.1, hard_gap=1.0, easy_gap=2.0)},
                {ROUTE_KEY: 'resident_v1', ROUTE_ADAPTIVE: 'margin_v1',
                 ROUTE_MARGIN: dict(minimum=0.1, maximum=0.5, hard_gap=2.0, easy_gap=2.0)}):
        with pytest.raises(ValueError):
            approx.validate_policy(raw, base)
    with pytest.raises(ValueError, match='deepseek_source_experts'):
        approx.validate_policy({ROUTE_KEY: 'resident_v1'}, {})
    for extra in ({'deepseek_physical_batch': 'b2'}, {'draft_model': 'embedded_dspark'},
                  {'deepseek_source_resident_gather': 'resident_bank_v1'},
                  {'deepseek_source_experts': dict(EXPERTS, residency='dynamic_precision_v1')}):
        with pytest.raises(ValueError):
            approx.validate_policy({ROUTE_KEY: 'resident_v1'}, {**base, **extra})


# ---------------------------------------------------------------- isolation

def _adapter(policy):
    a = object.__new__(DeepSeekV41SourceAdapter)
    a.kernel_policy = dict(policy)
    a.numerics_revision = a.profile = a.numeric_profile = 'base'
    return a


def test_identity_salt_is_distinct_per_operation_and_parameter():
    a = _adapter({})
    assert approx.bind_identity(a) is None and (a.numerics_revision, a.profile, a.numeric_profile) == ('base',) * 3
    yoco = {YOCO_KEY: 'yoco_v1', YOCO_LAYERS: [21, 39], YOCO_KEEP: 140, YOCO_MIN_ROWS: 4096}
    route = {ROUTE_KEY: 'resident_v1', ROUTE_ADAPTIVE: 'off', ROUTE_LAMBDA: 0.35}
    salts = set()
    for policy in (yoco, {**yoco, YOCO_KEEP: 256}, route, {**route, ROUTE_LAMBDA: 0.5},
                   {**route, ROUTE_KEY: 'resident_guarded_v1'}, {**yoco, **route}):
        a = _adapter(policy)
        salt = approx.bind_identity(a)
        assert salt and a.numerics_revision != 'base' and a.profile.startswith('base:approx-')
        assert a.numeric_profile.endswith('_approximate_unqualified')
        salts.add(salt)
    assert len(salts) == 6
    assert approx.private_profile({**yoco, **route}).startswith(approx.PRIVATE_PROFILE_PREFIX + 'yoco_v1+route_resident_v1:')


def test_apcv2_publication_and_reuse_refused(monkeypatch):
    def base(self, *, max_lanes, prefill_step):
        if not 1 <= prefill_step <= 1024:
            raise ValueError('V4.1 requires prefill chunks1..1024')
        return {'prefill_step_size': prefill_step}
    monkeypatch.setattr(DeepSeekV41MLXAdapter, 'execution_config', base)
    policy = source_policy({YOCO_KEY: 'yoco_v1'})
    a = _adapter(policy)
    result = a.execution_config(max_lanes=1, prefill_step=65536)
    assert result['prefill_step_size'] == 65536
    assert result['apcv2_reuse_disabled_reason'] == APC_REASON_PREFIX + 'yoco_v1'
    assert a.apc_publication_refusal([], [1, 2]) == APC_REASON_PREFIX + 'yoco_v1'
    with pytest.raises(ValueError, match='minimum prompt rows'):
        a.execution_config(max_lanes=1, prefill_step=1024)
    a = _adapter({ROUTE_KEY: 'resident_v1', ROUTE_ADAPTIVE: 'off', ROUTE_LAMBDA: 0.35})
    assert a.execution_config(max_lanes=1, prefill_step=512)['apcv2_reuse_disabled_reason'].startswith(APC_REASON_PREFIX)
    assert a.apc_publication_refusal([], [1]) == APC_REASON_PREFIX + 'route_resident_v1'
    assert approx.PUBLICATION_QUALIFIED is False
    # Default route is untouched.
    a = _adapter({})
    assert a.apc_publication_refusal([], [1]) is None
    assert a.execution_config(max_lanes=1, prefill_step=512)['apcv2_reuse_disabled_reason'] == 'adaptive_prefill_partitions_not_qualified'


def test_receipts_label_approximate_unqualified():
    off = approx.receipt({})
    assert off['approximate'] is False and off['apcv2_publication'] == 'not affected'
    policy = {YOCO_KEY: 'yoco_v1', YOCO_LAYERS: [21, 39], YOCO_KEEP: 140, YOCO_MIN_ROWS: 4096,
              ROUTE_KEY: 'resident_guarded_v1', ROUTE_ADAPTIVE: 'margin_v1', ROUTE_MARGIN: dict(approx.ROUTE_DEFAULT_MARGIN)}
    value = approx.receipt(policy)
    assert value['approximate'] is True and value['qualification'] == 'unqualified'
    assert value['default_selected'] is False and value['apcv2_publication'].startswith('refused: ')
    assert value['yoco']['qualification'] == value['route_bias']['qualification'] == 'unqualified'
    assert value['route_bias']['margin_values_measured'] is False


def test_stamp_refuses_foreign_private_profile():
    row = SimpleNamespace(ast_private_profile=None)
    approx.stamp_cache([row], 'p')
    approx.stamp_cache([row], 'p')
    with pytest.raises(ValueError):
        approx.stamp_cache([row], 'q')


def test_model_class_is_identity_when_off():
    assert approximate_model_class(DeepSeekV41Model, {}) is DeepSeekV41Model


# ---------------------------------------------------------------- route bias math

def _logits(scores):
    return np.log(np.expm1(np.asarray(scores, np.float64) ** 2)).astype(np.float32)


def _mx_select(logits, bias, resident, lam, mode, k=6, scale=1.5):
    lg = mx.array(np.asarray(logits, np.float32))[None, None]
    corr = mx.array(np.asarray(bias, np.float32))
    native = route_experts(lg, top_k=k, scale=scale, correction=corr, normalize=True)
    ids, w, changed = biased_selection(lg, corr, resident, lam, mode=mode, top_k=k, scale=scale,
                                       normalize=True, native_indices=native[0], native_weights=native[1])
    ids, w = np.asarray(ids).reshape(-1), np.asarray(w).reshape(-1)
    order = np.argsort(ids)
    return ids[order], w[order], bool(np.asarray(changed).reshape(-1)[0]), np.asarray(native[0]).reshape(-1)


@pytest.mark.parametrize('mode', ['resident_v1', 'resident_guarded_v1'])
def test_selection_matches_numpy_oracle_on_random_rows(mode):
    rng = np.random.default_rng(7)
    checked = 0
    for case in range(300):
        experts = 32
        logits = rng.normal(0, 2.0, experts).astype(np.float32)
        bias = rng.normal(0, 0.1, experts).astype(np.float32)
        resident = rng.random(experts) < rng.uniform(0.2, 0.9)
        lam = float(rng.choice([0.05, 0.35, 1.0, 3.0]))
        scores = np.sqrt(np.log1p(np.exp(logits.astype(np.float64))))
        ref_ids, ref_w, selection = reference_selection(scores, bias, resident, lam, mode=mode, top_k=6,
                                                        scale=1.5, return_selection=True)
        ordered = np.sort(scores + bias)[::-1], np.sort(selection)[::-1]
        if min(v[5] - v[6] for v in ordered) < 1e-4:
            continue  # float32 near-ties at the top-6 boundary are not an oracle question
        ids, w, _, _ = _mx_select(logits, bias, resident, lam, mode)
        assert list(ids) == list(ref_ids), case
        np.testing.assert_allclose(w, ref_w, rtol=1e-5)
        checked += 1
    assert checked > 100


def test_selection_hand_cases():
    scores = np.array([3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 1.0, 0.99, 0.5, 0.4], np.float32)
    bias = np.zeros(10, np.float32)
    every = np.ones(10, bool)
    # All native resident: native result, bit for bit.
    ids, w, changed, _ = _mx_select(_logits(scores), bias, every, 3.0, 'resident_v1')
    assert not changed and list(ids) == list(range(6))
    # Native expert 0 missing: plain bias swaps in the best resident (6).
    resident = every.copy(); resident[0] = False
    ids, w, changed, _ = _mx_select(_logits(scores), bias, resident, 2.5, 'resident_v1')
    assert changed and list(ids) == [1, 2, 3, 4, 5, 6]
    s = np.sqrt(np.log1p(np.expm1(scores.astype(np.float64) ** 2)))
    np.testing.assert_allclose(w, s[[1, 2, 3, 4, 5, 6]] / s[[1, 2, 3, 4, 5, 6]].sum() * 1.5, rtol=1e-5)
    # Small lambda cannot overcome the 2.0 score gap.
    ids, _, changed, _ = _mx_select(_logits(scores), bias, resident, 0.5, 'resident_v1')
    assert not changed and list(ids) == list(range(6))
    # Guarded: missing expert 0 leads best resident (1.0) by 2.0 >= width 0.5: protected.
    ids, _, changed, _ = _mx_select(_logits(scores), bias, resident, 3.0, 'resident_guarded_v1')
    assert not changed and list(ids) == list(range(6))
    # Guarded: missing expert 5 is not clearly ahead of resident 6 when they nearly tie.
    close = scores.copy(); close[6] = 2.45
    resident = every.copy(); resident[5] = False
    ids, _, changed, _ = _mx_select(_logits(close), bias, resident, 0.35, 'resident_guarded_v1')
    assert changed and list(ids) == [0, 1, 2, 3, 4, 6]
    # No resident replacement outside the native set: native.
    resident = np.zeros(10, bool); resident[:5] = True
    ids, _, changed, _ = _mx_select(_logits(close), bias, resident, 3.0, 'resident_guarded_v1')
    assert not changed and list(ids) == list(range(6))


def test_margin_lambda_matches_oracle():
    margin = dict(minimum=0.1, maximum=0.5, hard_gap=1.0, easy_gap=3.0)
    for row in (np.array([5.0, 1.0, 0.0]), np.array([2.0, 1.5, -np.inf]), np.array([2.0, 0.0, 0.5]),
                np.array([np.nan, 1.0, 2.0]), np.array([np.inf, 1.0]), np.array([1.0, -np.inf])):
        got = float(_margin_lambda_mx(mx.array(row.astype(np.float32)), margin).item())
        assert got == pytest.approx(margin_lambda(row, margin), abs=1e-6)
    assert margin_lambda([1.0, 0.0], dict(margin, maximum=0.1)) == 0.1


class Store:
    def __init__(self, layers, experts, resident):
        self._lock = threading.RLock()
        self.tables = {f'blk.{i}': tuple((i, e) for e in range(experts)) for i in range(layers)}
        self._cache = OrderedDict((key, object()) for key in resident)
        self._leased = {}
        self.stats = dict(cache_hits=0)


def test_resident_mask_reads_without_mutation():
    store = Store(2, 4, [(0, 3), (0, 1), (1, 2)])
    store._leased[(1, 0)] = object()
    order, stats = list(store._cache), dict(store.stats)
    assert list(resident_mask(store, 0, 4)) == [False, True, False, True]
    assert list(resident_mask(store, 1, 4)) == [True, False, True, False]
    assert list(store._cache) == order and store.stats == stats
    with pytest.raises(ValueError):
        resident_mask(store, 0, 5)
    with pytest.raises(ValueError):
        resident_mask(SimpleNamespace(), 0, 4)


def _route_model(c, store, *, lam, mode='resident_v1'):
    policy = {ROUTE_KEY: mode, ROUTE_ADAPTIVE: 'off', ROUTE_LAMBDA: lam}
    model = approximate_model_class(DeepSeekV41Model, policy)(c, Weights(c), identity='synthetic-route@1')
    model.source_approx_private_profile = approx.private_profile(policy)
    model.source_route_bias = ResidentRouteBias.from_policy(policy, store, c)
    return model, policy


def test_route_bias_passthrough_is_bitwise_and_bias_changes_decode_only():
    c = config()
    every = Store(c.num_hidden_layers, c.n_routed_experts,
                  [(i, e) for i in range(c.num_hidden_layers) for e in range(c.n_routed_experts)])
    model, policy = _route_model(c, every, lam=3.0)
    ref = DeepSeekV41Model(c, Weights(c), identity='synthetic-route@1')
    ids = mx.array([[1, 2, 3, 4, 5]], mx.int32)
    cache, rc = model.make_cache(), ref.make_cache()
    np.testing.assert_array_equal(_f(model(ids, cache)), _f(ref(ids, rc)))
    assert model.source_route_bias.stats['router_calls'] == 0  # prefill rows keep native routing
    for token in (6, 7, 8):
        step = mx.array([[token]], mx.int32)
        np.testing.assert_array_equal(_f(model(step, cache)), _f(ref(step, rc)))
    assert model.source_route_bias.stats['router_calls'] == 3 * c.num_hidden_layers
    assert model.source_route_bias.receipt()['recent_changed_layer_rows'] == 0
    # Only expert 0 of each layer resident and a large lambda: decode changes.
    sparse = Store(c.num_hidden_layers, c.n_routed_experts, [(i, 0) for i in range(c.num_hidden_layers)])
    model, policy = _route_model(c, sparse, lam=3.0)
    ref = DeepSeekV41Model(c, Weights(c), identity='synthetic-route@1')
    cache, rc = model.make_cache(), ref.make_cache()
    model(ids, cache), ref(ids, rc)
    outs = []
    for token in (6, 7, 8):
        step = mx.array([[token]], mx.int32)
        outs.append((_f(model(step, cache)), _f(ref(step, rc))))
    assert any(not np.array_equal(a, b) for a, b in outs)
    assert model.source_route_bias.receipt()['recent_changed_layer_rows'] > 0
    assert all(row.ast_private_profile == approx.private_profile(policy) for row in cache)
    with model.source_route_bias.suspend():
        step = mx.array([[9]], mx.int32)
        calls = model.source_route_bias.stats['router_calls']
        model(step, cache)
        assert model.source_route_bias.stats['router_calls'] == calls


def test_router_wrappers_survive_block_rebuild():
    c = config()
    store = Store(c.num_hidden_layers, c.n_routed_experts, [])
    model, _ = _route_model(c, store, lam=0.35)
    install_router_wrappers(model)
    assert all(getattr(b, '_mlx2_route_bias_wrapped', False) for b in model.layers)
    from mlx2.runtime.models.deepseek_v41 import Block
    model.layers = [Block(model, i) for i in range(c.num_hidden_layers)]
    model(mx.array([[1]], mx.int32), model.make_cache())
    assert all(getattr(b, '_mlx2_route_bias_wrapped', False) for b in model.layers)


def test_adaptive_margin_lambda_follows_previous_logits():
    c = config()
    store = Store(c.num_hidden_layers, c.n_routed_experts, [])
    policy = {ROUTE_KEY: 'resident_v1', ROUTE_ADAPTIVE: 'margin_v1',
              ROUTE_MARGIN: dict(minimum=0.0, maximum=2.0, hard_gap=0.0, easy_gap=1e-6)}
    model = approximate_model_class(DeepSeekV41Model, policy)(c, Weights(c), identity='synthetic-route@1')
    model.source_approx_private_profile = approx.private_profile(policy)
    bias = model.source_route_bias = ResidentRouteBias.from_policy(policy, store, c)
    assert bias.current_lambda() == 0.0
    logits = model(mx.array([[1, 2, 3]], mx.int32), model.make_cache())
    expected = margin_lambda(np.asarray(logits[0, -1]), policy[ROUTE_MARGIN])
    assert float(bias.current_lambda().item()) == pytest.approx(expected, abs=1e-6)
