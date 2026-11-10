"""Bug sweep 2026-11-09, lane ds41tiers (DST-1..DST-6 + the rows_moe shield note).

CPU only: synthetic policies, stores, kernels and ladder directories; no
checkpoint, no Metal kernel.
"""
import importlib.util
import json
import pathlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.adapters import deepseek_v41_dspark_chain as chain
from mlx2.adapters import deepseek_v41_dspark_chain_moe as rows_moe
from mlx2.adapters import deepseek_v41_source_approx as approx
from mlx2.adapters import deepseek_v41_source_lfu as lfu
from mlx2.adapters import deepseek_v41_source_q1_on_miss as q1m
from mlx2.adapters.deepseek_v41_source import source_policy
from mlx2.adapters.deepseek_v41_source_approx import DROP_KEY, DROP_LAYERS

GB = 10 ** 9
ROOT = Path(__file__).resolve().parents[1]
LFU = {'mode': 'original_lfu', 'budget_bytes': 75_000_000_000}
TIERED = dict(LFU, q1_root='/models/q1-bank', q1_on_miss={'max_weight': 0.10, 'layers': [0, 39]})


# ---- DST-1: selection overrides through the policy parser ------------------

def _tiers_policy(overrides):
    return {'mode': 'original_lfu', 'budget_bytes': 75_000_000_000, 'q1_root': '/models/q1-bank',
            'tiers': {'usage_map': '/maps/usage.json', 'q1_bytes': 5 * GB, 'overrides': overrides}}


def test_dst1_tier_selection_overrides_pass_the_policy_parser():
    chosen = [{'layers': [30, 39], 'only': 'q1'}, {'layers': [0, 1], 'tier': 'q4', 'top': 3},
              {'layers': 5, 'tier': 'q1', 'experts': [3, 17]}]
    parsed = lfu.parse_policy(_tiers_policy(chosen))
    assert parsed['tiers']['overrides'] == chosen
    # A budget overflow in the same list still fails closed at parse time.
    with pytest.raises(ValueError, match='exceeds budget_bytes'):
        lfu.parse_policy(_tiers_policy(chosen + [{'layers': [0, 39], 'q4_experts': 384}]))


# ---- DST-2: ablation ladder, truncated baseline ----------------------------

def _ladder():
    path = ROOT / 'scripts' / 'research' / 'ds41_atlas' / 'ablation_ladder.py'
    spec = importlib.util.spec_from_file_location('ablation_ladder_sweep1109', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tok(token, top):
    return {'token': token, 'logprob': top[0][1], 'top': top}


def _reply(text, tokens, finish):
    return {'gsm8k-7': {'reply': text, 'tokens': tokens, 'finish': finish, 'completion_tokens': len(tokens),
                        'decode_tok_s': 2.0, 'experts': {'v41_expert_cache_misses': 0,
                                                         'v41_expert_demand_wait_ns': 0}}}


def test_dst2_truncated_baseline_rows_are_oracle_na_and_finished_arms_are_judged(tmp_path):
    m = _ladder()
    shared = [_tok('Each', [('Each', -0.1), ('The', -2.5)]), _tok(' crate', [(' crate', -0.2), (' box', -1.9)])]
    base = shared + [_tok(' holds', [(' holds', -0.4), (' has', -0.7)]),
                     _tok(' forty', [(' forty', -0.3), (' 40', -1.6)])]
    # Near-tie divergence at token 2 (both margins <= 0.5 nat, same top two), arm also truncated.
    near = shared + [_tok(' has', [(' has', -0.5), (' holds', -0.6)]),
                     _tok(' eight', [(' eight', -0.2), (' 8', -1.4)])]
    # Diverges with a large margin and stops with a wrong final answer.
    wrong = shared + [_tok(' weighs', [(' weighs', -0.1), (' holds', -3.0)]),
                      _tok(' ####', [(' ####', -0.1), (' so', -2.2)])]
    arms = {'exact': _reply('Each crate holds forty jars and the shelves', base, 'length'),
            'near': _reply('Each crate has eight rows of jars stacked', near, 'length'),
            'wrong': _reply('Each crate weighs a lot.\n#### 7', wrong, 'stop')}
    for name, replies in arms.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / 'replies.json').write_text(json.dumps(replies))
        (tmp_path / name / 'done.json').write_text('{}')
    table = m.oracles({'gsm8k-7': {'reference': '160'}})
    rows = {r['arm']: r for r in m.summarize(tmp_path, list(arms), table)}
    exact, nr, wr = (rows[n]['per_prompt']['gsm8k-7'] for n in ('exact', 'near', 'wrong'))
    # Exact token parity needs no oracle: it stays identical even on a truncated baseline.
    assert exact['class'] == 'identical' and exact['tokens_identical'] is True and exact['oracle_pass'] is None
    assert rows['exact']['identical'] == 1 and rows['exact']['oracle_na'] == 0
    assert nr['class'] == 'oracle_na' and nr['first_divergence'] == 2 and nr['oracle_pass'] is None
    assert wr['class'] == 'diverged_fail' and wr['oracle_pass'] is False
    assert rows['near']['near_tie'] == 0 and rows['near']['diverged_pass'] == 0 and rows['near']['oracle_na'] == 1
    assert '| oracle n/a |' in (tmp_path / 'summary.md').read_text()


# ---- DST-3 + shield note: rows_moe declines, observes the call width -------

class _Store:
    def __init__(self, experts):
        self.experts, self.acquired, self._active, self.stats = experts, [], False, {}
        self.windows, self.observed = [], []

    def acquire(self, prefix, expert):
        assert self._active, 'acquired outside the window'
        self.acquired.append((prefix, expert))
        return self.experts[expert]

    def observe_moe_call(self, prefix, rows):
        self.observed.append((prefix, rows))

    @contextmanager
    def window(self):
        self._active = True
        self.windows.append('open')
        try:
            yield
        finally:
            self._active = False
            self.windows.append('closed')


def _experts(count=12, hidden=8, inner=5, seed=11):
    rng = np.random.default_rng(seed)
    return {e: {k: mx.array(rng.normal(0, .3, shape).astype(np.float32))
                for k, shape in (('gate', (inner, hidden)), ('up', (inner, hidden)), ('down', (hidden, inner)))}
            for e in range(count)}


def _parks(routes, original=None):
    rng = np.random.default_rng(5)
    out = []
    for route in routes:
        x = mx.array(rng.normal(0, 1, (1, 8)).astype(np.float32))
        scores = mx.array(rng.uniform(0.1, 1, (1, 6)).astype(np.float32))
        out.append(chain._Park('moe', 'blk.3', (x, mx.array([route], mx.uint32), scores, 'blk.3'), {}, original))
    return out


ROUTES = [[0, 2, 4, 6, 8, 10], [1, 2, 3, 6, 9, 11]]


def _weights(store):
    return SimpleNamespace(expert_math_policy='routed_fused_fp32', expert_stream=store,
                           source_resident_bank_session=None, _ready=lambda: None,
                           scheduled_expert_calls={}, source_multirow_dense_contract=True)


def test_dst3_executor_falls_back_before_any_acquire_when_row_kernels_refuse_the_device(monkeypatch):
    store = _Store(_experts())
    stats = {}
    executor = chain.SourceRowsExecutor(_weights(store), stats)
    monkeypatch.setattr(executor, '_gpu', lambda: True)     # a GPU host that is not an M5 Max (here: CPU)
    calls = []
    parks = _parks(ROUTES, original=lambda *a, **k: calls.append(a) or ('row', len(calls)))
    assert executor.run(parks) == [('row', 1), ('row', 2)]
    assert stats['lockstep_moe_declines'] == 1 and stats['lockstep_row_sites'] == 2
    assert store.acquired == [] and store.windows == []


def test_dst3_kernel_precondition_inside_the_loop_declines_and_closes_the_window():
    store = _Store(_experts())

    def refusing_middle(prepared, rows, scores, data):
        raise ValueError('Source AST M1 GU requires an owned M5 Max GPU window')
    kernels = SimpleNamespace(prepare=lambda x: x, middle=refusing_middle,
                              down=lambda a, d: a, combine=lambda *a: None)
    with pytest.raises(rows_moe.Decline):
        rows_moe.rows_moe(_weights(store), _parks(ROUTES), {}, kernels=kernels)
    assert store.windows == ['open', 'closed'] and not store._active   # window-scoped leases released


def test_review_in_loop_kernel_refusal_is_counted_apart_from_precondition_declines(monkeypatch):
    store = _Store(_experts())
    stats = {}
    executor = chain.SourceRowsExecutor(_weights(store), stats)
    monkeypatch.setattr(executor, '_gpu', lambda: True)
    monkeypatch.setattr(rows_moe, '_kernel_device_supported', lambda: True)

    def refusing_middle(prepared, rows, scores, data):
        raise ValueError('Source AST M1 GU requires original/Q1 occurrence rows1..16')
    monkeypatch.setattr(rows_moe, '_kernels', lambda: SimpleNamespace(
        prepare=lambda x: x, middle=refusing_middle, down=lambda a, d: a, combine=lambda *a: None))
    calls = []
    parks = _parks(ROUTES, original=lambda *a, **k: calls.append(a) or ('row', len(calls)))
    assert executor.run(parks) == [('row', 1), ('row', 2)]
    assert stats['lockstep_moe_kernel_declines'] == 1 and 'lockstep_moe_declines' not in stats
    assert stats['lockstep_row_sites'] == 2


def test_rows_moe_reports_its_call_width_to_the_prefill_shield_hook():
    store = _Store(_experts())
    ident = lambda a, d: a
    kernels = SimpleNamespace(prepare=lambda x: x, middle=lambda p, r, s, d: p[np.array(r).tolist()],
                              down=ident, combine=lambda parts, values, order: mx.zeros((2, 8)))
    rows_moe.rows_moe(_weights(store), _parks(ROUTES), {}, kernels=kernels)
    assert store.observed == [('blk.3', 2)]
    assert store.acquired and store.acquired[0][0] == 'blk.3'


# ---- DST-4: route drop layers beyond the checkpoint ------------------------

def _drop_adapter(layers):
    policy = source_policy({'deepseek_source_experts': LFU, DROP_KEY: 'drop_min_v1', DROP_LAYERS: layers})
    model = SimpleNamespace(args=SimpleNamespace(num_hidden_layers=40))
    return SimpleNamespace(kernel_policy=policy, model=model, weight_stream=None)


@pytest.mark.parametrize('layers', [[40, 60], [38, 45]])
def test_dst4_route_drop_refuses_layers_outside_the_checkpoint(layers):
    with pytest.raises(ValueError, match='outside this checkpoint'):
        approx.configure_model(_drop_adapter(layers))


def test_dst4_route_drop_inside_the_checkpoint_still_configures():
    adapter = _drop_adapter([13, 39])
    approx.configure_model(adapter)
    assert adapter.model.source_route_drop.applies(39) and not adapter.model.source_route_drop.applies(12)


# ---- DST-5: digests cover the math-carrying files --------------------------

def test_dst5_drop_miss_digest_binds_route_bias(monkeypatch):
    seen = []
    monkeypatch.setattr(approx, '_file_identity', lambda names: seen.append(list(names)) or 'x')
    approx.implementation_identity(source_policy({'deepseek_source_experts': LFU, DROP_KEY: 'drop_miss_v1'}))
    approx.implementation_identity(source_policy({'deepseek_source_experts': LFU, DROP_KEY: 'drop_min_v1'}))
    assert 'adapters/deepseek_v41_source_route_bias.py' in seen[0]
    assert 'adapters/deepseek_v41_source_route_bias.py' not in seen[1]     # drop_min alone unchanged


def test_dst5_tier_digest_binds_the_q2_reference_law(monkeypatch):
    read = []
    monkeypatch.setattr(pathlib.Path, 'read_bytes', lambda self: read.append(self.as_posix()) or b'')
    q1m.implementation_identity()
    assert any(p.endswith('adapters/deepseek_v41_source_weights.py') for p in read)


# ---- DST-6: tier receipt names the profile the rows carry ------------------

def test_dst6_tier_receipt_reports_the_combined_profile_under_composition(monkeypatch):
    policy = source_policy({'deepseek_source_experts': TIERED, DROP_KEY: 'drop_min_v1'})
    rule = SimpleNamespace(salt='c' * 64, receipt=lambda: {'private_cache_profile': q1m.private_profile('c' * 64)})
    monkeypatch.setattr(q1m, 'store_rule', lambda adapter: rule)
    model = SimpleNamespace(source_approx_private_profile=approx.private_profile(policy))
    adapter = SimpleNamespace(kernel_policy=policy, model=model)
    q1m.configure_model(adapter)
    value = q1m.diagnostics(adapter)
    # Q1OnMissSourceModel._forward stamps source_combined_private_profile when set.
    assert value['combined_private_cache_profile'] == model.source_combined_private_profile
    assert value['combined_identity_salt'] == model.source_combined_identity_salt
    alone = SimpleNamespace(kernel_policy=source_policy({'deepseek_source_experts': TIERED}),
                            model=SimpleNamespace())
    assert 'combined_private_cache_profile' not in q1m.diagnostics(alone)
