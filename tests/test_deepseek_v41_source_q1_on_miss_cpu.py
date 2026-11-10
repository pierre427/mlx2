"""Opt-in approximate q1_on_miss: policy surface, store rule and identity contract.

CPU synthetic stores only (scaled expert sizes, MLX pinned to the CPU device):
no model load, no Metal work. Reuses the residency test's synthetic store.
"""
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_lfu as lfu
from mlx2.adapters import deepseek_v41_source_q1_on_miss as q1m
from mlx2.adapters import deepseek_v41_source_defaults as defaults
from mlx2.adapters.deepseek_v41_source import source_policy
from tests.test_deepseek_v41_source_residency_cpu import O, Q, make, small  # noqa: F401 - fixture

ROOT = '/models/q1-bank'


def experts(**extra):
    value = {'mode': 'original_lfu', 'budget_bytes': 75_000_000_000, 'q1_root': ROOT,
             'q1_on_miss': {'max_weight': 0.10, 'layers': [0, 39]}}
    value.update(extra)
    return value


# ---- Policy surface -----------------------------------------------------

def test_policy_is_opt_in_and_normalised():
    plain = lfu.parse_policy({'mode': 'original_lfu', 'budget_bytes': 75_000_000_000})
    assert 'q1_on_miss' not in plain and 'q1_root' not in plain
    parsed = lfu.parse_policy(experts())
    assert parsed['q1_on_miss'] == {'max_weight': 0.10, 'layers': [0, 39]}
    assert lfu.parse_policy(experts(q1_on_miss={'max_weight': 1}))['q1_on_miss'] == {
        'max_weight': 1.0, 'layers': [0, 1023]}


@pytest.mark.parametrize('bad', [
    {'max_weight': 0}, {'max_weight': -0.1}, {'max_weight': 1.5}, {'max_weight': True},
    {'max_weight': float('nan')}, {'max_weight': '0.1'}, {}, {'layers': [0, 39]},
    {'max_weight': 0.1, 'layers': [5, 2]}, {'max_weight': 0.1, 'layers': [0]},
    {'max_weight': 0.1, 'layers': [-1, 3]}, {'max_weight': 0.1, 'extra': 1}, 0.1, None,
])
def test_policy_rejects_malformed_rule(bad):
    with pytest.raises(ValueError, match='q1_on_miss'):
        lfu.parse_policy(experts(q1_on_miss=bad))


def test_policy_fails_closed_on_mode_and_root():
    with pytest.raises(ValueError, match='original_lfu'):
        lfu.parse_policy(dict(experts(), mode='hot_original_q1', hot_bytes=60_000_000_000,
                              hotset_path='/hot.json'))
    no_root = experts(); no_root.pop('q1_root')
    with pytest.raises(ValueError, match='q1_root'):
        lfu.parse_policy(no_root)
    with pytest.raises(ValueError, match='q1_root'):
        lfu.parse_policy(experts(q1_root='relative/bank'))
    with pytest.raises(ValueError, match='Q1 options'):
        lfu.parse_policy({'mode': 'original_lfu', 'budget_bytes': 75_000_000_000, 'q1_root': ROOT})
    with pytest.raises(ValueError, match='direct_reads'):
        lfu.parse_policy(experts(direct_reads=True))


def test_source_policy_accepts_ordinary_b1_and_refuses_uncomposed_laws():
    policy = source_policy({'deepseek_source_experts': experts()})
    assert policy['deepseek_source_experts']['q1_on_miss']['max_weight'] == 0.10
    assert q1m.enabled(policy) and q1m.apc_refusal_reason(policy) == q1m.APC_REASON
    assert q1m.apc_refusal_reason(source_policy({})) is None
    for extra in ({'deepseek_source_expert_io': 'overlap_v1'},
                  {'deepseek_source_prefill_readahead': 'layer_ahead_v1'},
                  {'draft_model': 'prompt_lookup'}):
        with pytest.raises(ValueError):
            source_policy({'deepseek_source_experts': experts(), **extra})


def test_combination_refusals_are_named():
    policy = {'deepseek_source_experts': experts(), 'deepseek_physical_batch': 'b2',
              'draft_model': 'embedded_dspark'}
    with pytest.raises(ValueError) as error:
        q1m.validate_combination(policy)
    for name in ('deepseek_physical_batch', 'draft_model'):
        assert name in str(error.value)
    q1m.validate_combination({'deepseek_physical_batch': 'b2'})  # off: nothing to refuse


def test_measured_default_bundle_does_not_apply():
    assert not defaults.default_route_policy_applies('ordinary', {'deepseek_source_experts': experts()})


# ---- Store rule -------------------------------------------------------------

def store(*, max_weight=0.10, layers=(0, 1)):
    s = make(mode='shared_v1')
    s.q1_root = None; s.q1_records = {}; s.q1_binding = None; s.hot = frozenset(); s.hot_binding = None
    s.payload_ceiling = s.cache_budget_bytes - O
    s.policy = {'mode': 'original_lfu', 'budget_bytes': s.cache_budget_bytes, 'q1_root': '/q1b',
                'q1_on_miss': {'max_weight': max_weight, 'layers': list(layers)}}
    s.q1_on_miss = q1m.Q1OnMiss(s.policy['q1_on_miss'], root='/q1b',
                                records=defaultdict(lambda: {'file': 'q1', 'size': Q + 8}),
                                bank_binding='d' * 64, layers=2)
    s.stats.update(q1_on_miss_reads=0, q1_on_miss_bytes_read=0)
    return s


def call(s, layer, ids, weights, acquire=None):
    """One MoE call inside a window; returns the leased arrays by expert."""
    ids = np.array([ids], np.uint32)
    scores = np.array([weights], np.float32)
    out = {}
    with s.window():
        with s.q1_on_miss_scope(f'blk.{layer}', ids, scores) as armed:
            s.observe_routes(f'blk.{layer}', ids)
            for expert in (acquire if acquire is not None else ids.reshape(-1)):
                out[int(expert)] = s.acquire(f'blk.{layer}', int(expert))
    s.end_context()
    return armed, out


def test_low_weight_miss_reads_q1_and_never_enters_the_cache(small):
    s = store()
    # Raw weights sum to 2.0 (routed scale): renormalised 0.6/0.32/0.08.
    armed, out = call(s, 0, [1, 2, 3], [1.2, 0.64, 0.16])
    assert armed
    assert s.reads == [((0, 1), 'read_tensors'), ((0, 2), 'read_tensors'), ((0, 3), 'q1_read')]
    assert (0, 3) not in s._cache and {(0, 1), (0, 2)} <= set(s._cache)
    assert sum(v.nbytes for v in out[3].values()) == Q
    assert s.stats['q1_on_miss_reads'] == 1 and s.stats['q1_on_miss_bytes_read'] == Q + 8
    assert s.stats['cache_misses'] == 2 and s.stats['original_misses'] == 2
    receipt = s.q1_on_miss.receipt()
    assert receipt['counters']['q1_reads'] == 1 and receipt['counters']['below_threshold'] == 1
    assert receipt['per_layer_q1_served'] == [1, 0] and receipt['pool'] is None
    assert receipt['weight_served_from_q1'] == pytest.approx(0.08)
    assert receipt['approximate'] and receipt['qualification'] == 'unqualified'
    assert not s._leased and not s.q1_on_miss.armed


def test_resident_or_heavy_or_out_of_range_experts_stay_exact(small):
    s = store(layers=(1, 1))
    call(s, 1, [4, 5], [0.95, 0.05])            # (1,5) miss -> Q1
    assert s.reads[-1] == ((1, 5), 'q1_read')
    call(s, 1, [5, 6], [0.5, 0.5])              # heavy: exact original, cached
    assert s.reads[-1] == ((1, 6), 'read_tensors') and (1, 5) in s._cache
    reads = len(s.reads)
    call(s, 1, [4, 5], [0.95, 0.05])            # now resident: exact hit, no read
    assert len(s.reads) == reads
    assert s.q1_on_miss.stats['below_threshold_resident'] == 1
    call(s, 0, [7, 8], [0.99, 0.01])            # layer 0 outside [1, 1]
    assert s.reads[-1] == ((0, 8), 'read_tensors') and (0, 8) in s._cache
    assert s.q1_on_miss.stats['q1_reads'] == 1
    assert s.q1_on_miss.stats['decode_calls'] == 4 and s.q1_on_miss.stats['armed_calls'] == 3


def test_multi_row_calls_are_never_armed(small):
    s = store()
    ids = np.array([[1, 2], [3, 4]], np.uint32)
    scores = np.array([[0.99, 0.01], [0.99, 0.01]], np.float32)
    with s.window():
        with s.q1_on_miss_scope('blk.0', ids, scores) as armed:
            for expert in (1, 2, 3, 4):
                s.acquire('blk.0', expert)
    assert armed is False
    assert all(function == 'read_tensors' for _, function in s.reads)
    assert s.q1_on_miss.stats['decode_calls'] == 0


def test_q1_lease_cannot_leak_to_an_unarmed_acquire(small):
    s = store()
    ids = np.array([[1, 2]], np.uint32)
    with s.window():
        with s.q1_on_miss_scope('blk.0', ids, np.array([[0.97, 0.03]], np.float32)):
            first = s.acquire('blk.0', 2)
            assert s.acquire('blk.0', 2) is first     # same armed call: same lease
        with pytest.raises(ValueError, match='outside its armed call'):
            s.acquire('blk.0', 2)
    # A later window drops the lease: the same expert is read exactly.
    _, out = call(s, 0, [2, 9], [0.5, 0.5])
    assert s.reads[-2] == ((0, 2), 'read_tensors') and (0, 2) in s._cache


def test_scope_validates_geometry_and_weights(small):
    s = store()
    with s.window():
        with pytest.raises(ValueError, match='geometry'):
            with s.q1_on_miss_scope('blk.0', np.array([[1, 2]], np.uint32), np.array([[1.0]], np.float32)):
                pass
    # Malformed routing values serve the call exact and are counted; the
    # provider stays usable (no poisoned window).
    for ids, weights in (([1, 2], [1.0, float('nan')]), ([1, 2], [0.0, 0.0]),
                         ([1, 2], [float('inf'), 0.01]), ([1, 1], [0.5, 0.5])):
        with s.window():
            with s.q1_on_miss_scope('blk.0', np.array([ids], np.uint32), np.array([weights], np.float32)) as armed:
                assert armed is False
                s.acquire('blk.0', 2)
    assert s.q1_on_miss.stats['invalid_route_calls'] == 4
    assert all(function == 'read_tensors' for _, function in s.reads)
    call(s, 0, [3, 4], [0.97, 0.03])                  # next valid call still arms
    assert s.reads[-1] == ((0, 4), 'q1_read')


# ---- Identity / APCv2 contract ------------------------------------------

def rule(max_weight=0.10, bank='d' * 64):
    return q1m.Q1OnMiss({'max_weight': max_weight, 'layers': [0, 39]}, root='/q1b', records={},
                        bank_binding=bank, layers=40)


def test_salt_binds_parameters_and_bank(small):
    base = rule().salt
    assert len(base) == 64
    assert rule(0.2).salt != base and rule(bank='e' * 64).salt != base and rule().salt == base
    s = store()
    assert s.apc_revision().endswith(':' + q1m.NAME + '=' + s.q1_on_miss.salt)
    receipt = s.receipt()
    assert receipt['q1_on_miss']['identity_salt'] == s.q1_on_miss.salt
    assert receipt['numeric_profile'].endswith('q1_on_miss_v1_approximate_unqualified')


def test_bind_identity_salts_every_profile_field():
    r = rule()
    adapter = SimpleNamespace(kernel_policy={'deepseek_source_experts': experts()},
                              weight_stream=SimpleNamespace(q1_on_miss=r),
                              numerics_revision='n', profile='p', numeric_profile='np')
    assert q1m.bind_identity(adapter) == r.salt
    assert r.salt in adapter.numerics_revision and adapter.profile == 'p:q1miss-' + r.salt[:16]
    assert adapter.numeric_profile.endswith('q1_on_miss_v1_approximate_unqualified')
    off = SimpleNamespace(kernel_policy={}, weight_stream=None, numerics_revision='n', profile='p', numeric_profile='np')
    assert q1m.bind_identity(off) is None and off.numerics_revision == 'n'
    unbound = SimpleNamespace(kernel_policy={'deepseek_source_experts': experts()},
                              weight_stream=SimpleNamespace(), numerics_revision='n', profile='p', numeric_profile='np')
    with pytest.raises(ValueError, match='did not bind'):
        q1m.bind_identity(unbound)
    assert q1m.diagnostics(adapter)['apcv2_publication'] == 'refused: ' + q1m.APC_REASON
    assert q1m.diagnostics(off) == {'selected': False}


def test_model_class_stamps_only_forwards_that_served_q1():
    class Base:
        def _forward(self, inputs, cache=None, **options):
            if inputs == 'q1':
                self.source_q1_on_miss.stats['q1_served'] += 1
            return inputs

    assert q1m.model_class(Base, {}) is Base
    model = q1m.model_class(Base, {'deepseek_source_experts': experts()})()
    model.source_q1_on_miss = r = rule()
    rows = [SimpleNamespace(ast_private_profile=None)]
    model._forward('exact', rows)
    assert rows[0].ast_private_profile is None
    model._forward('q1', rows)
    assert rows[0].ast_private_profile == q1m.private_profile(r.salt)
    assert r.stats['stamped_forwards'] == 1


# ---- Weights wiring (CPU provider oracle; no Metal) ---------------------

@pytest.mark.parametrize('enabled', [False, True])
def test_moe_routes_arms_the_scope_inside_the_window_only_when_enabled(monkeypatch, enabled):
    from contextlib import contextmanager
    import mlx.core as mx
    from tests.test_deepseek_v41_source_batch_provider_cpu import _case
    case = _case(monkeypatch)
    entered = []
    if enabled:
        case.store.q1_on_miss = object()

        @contextmanager
        def scope(prefix, values, scores):
            entered.append((prefix, values.copy(), np.array(scores), case.store._active, len(case.acquired)))
            yield True
            entered.append('exit')
        case.store.q1_on_miss_scope = scope
    output = case.w.moe(case.x, case.ids, case.scores, 'blk.0'); mx.eval(output)
    assert len(case.acquired) == 6
    if not enabled:
        assert entered == []
        return
    (prefix, values, scores, active, acquired_before), done = entered
    assert prefix == 'blk.0' and np.array_equal(values, case.ids) and active and acquired_before == 0
    assert np.array_equal(scores, np.array(case.scores)) and done == 'exit'
