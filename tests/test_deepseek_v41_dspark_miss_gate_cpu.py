"""DSpark expert-miss cost admission (expert_miss_cost_v1): law, gate and route.

CPU only: a synthetic clock and provider counters; no model, no Metal.
"""
from types import SimpleNamespace
import time

import pytest

from mlx2.adapters import deepseek_v41_dspark_miss_gate as gate
from mlx2.adapters.deepseek_v41_dspark_external import DeepSeekV41DSparkDraftModel
from mlx2.adapters.deepseek_v41_dspark_policy import split_chain_policy
from mlx2.runtime.external_speculative import DraftUnavailable

GIB = 1 << 30


def test_law_matches_the_ds4_admission_inequality():
    common = dict(plain_tokens_per_second=10.0, propose_seconds=0.03, verify_nonread_seconds=0.1,
                  useful_tokens=3.0, read_bytes=4 * GIB, read_seconds=0.4)
    admit = gate.decide(predicted_miss_bytes=0.5 * GIB, **common)    # 0.03+0.1+0.05 <= 0.3
    assert admit['admit'] and admit['priced'] and admit['bandwidth_source'] == 'request'
    assert admit['expected_seconds'] == pytest.approx(0.18) and admit['plain_seconds'] == pytest.approx(0.3)
    reject = gate.decide(predicted_miss_bytes=2 * GIB, **common)     # 0.13+0.2 > 0.3
    assert not reject['admit'] and reject['action'] == 'plain'
    assert gate.decide(**{**common, 'plain_tokens_per_second': 0.0}, predicted_miss_bytes=0)['reason'] == 'observe_plain_rate'
    assert gate.decide(**{**common, 'propose_seconds': None}, predicted_miss_bytes=0)['reason'] == 'price_speculation'
    unknown = gate.decide(**{**common, 'read_bytes': 0, 'read_seconds': 0.0}, predicted_miss_bytes=GIB)
    assert unknown['admit'] and unknown['reason'] == 'read_bandwidth_unknown'
    lifetime = gate.decide(**{**common, 'read_bytes': 0, 'read_seconds': 0.0}, predicted_miss_bytes=GIB,
                           fallback_read_bytes=10 * GIB, fallback_read_seconds=1.0)
    assert lifetime['priced'] and lifetime['bandwidth_source'] == 'provider_lifetime'
    zero = gate.decide(**{**common, 'read_bytes': 0, 'read_seconds': 0.0}, predicted_miss_bytes=0)
    assert zero['priced'] and zero['miss_seconds'] == 0.0    # no reads: compute alone decides
    for bad in (float('nan'), -1.0, float('inf')):
        with pytest.raises(ValueError):
            gate.decide(predicted_miss_bytes=bad, **common)


def test_policy_is_opt_in_and_embedded_only():
    assert split_chain_policy({}, embedded=True) == {}
    value = {'deepseek_dspark_admission': 'expert_miss_cost_v1'}
    assert split_chain_policy(dict(value), embedded=True) == value
    assert split_chain_policy({'deepseek_dspark_admission': 'off'}, embedded=True) == {}
    with pytest.raises(ValueError, match='embedded'):
        split_chain_policy(dict(value), embedded=False)
    with pytest.raises(ValueError):
        split_chain_policy({'deepseek_dspark_admission': 'always'}, embedded=True)


class Clock:
    def __init__(self):
        self.ns = 10**12

    def advance(self, seconds):
        self.ns += int(seconds * 1e9)


class Hidden:
    def __init__(self, rows):
        self.shape = (1, rows, 12)


def harness(monkeypatch, *, admission='expert_miss_cost_v1'):
    clock = Clock()
    monkeypatch.setattr(time, 'perf_counter_ns', lambda: clock.ns)
    monkeypatch.setattr(time, 'perf_counter', lambda: clock.ns / 1e9)
    stats = {'bytes_read': 0, 'demand_wait_ns': 0}
    model = DeepSeekV41DSparkDraftModel.__new__(DeepSeekV41DSparkDraftModel)
    model.admission, model.admission_stats = admission, {}
    model.backbone = SimpleNamespace(target_weights=SimpleNamespace(expert_stream=SimpleNamespace(stats=stats)))
    calls = []

    def scheduled(anchors, hidden, cache, length, rngs, temperatures, **kwargs):
        calls.append(length)
        clock.advance(0.03 if length else 0.005)
        tokens = [7] * length
        return [tokens], [[None] * length]
    model._scheduled_draft = scheduled
    plane = SimpleNamespace()
    state = dict(committed=1)

    def round_(*, verify_seconds, miss_bytes=0, read_seconds=0.0, committed=None):
        result = model.draft_distributions([5], Hidden(state['committed']), [plane], 3, [None], [0.0])
        proposed = len(result[0][0])
        clock.advance(verify_seconds)
        stats['bytes_read'] += miss_bytes
        stats['demand_wait_ns'] += int(read_seconds * 1e9)
        state['committed'] = committed if committed is not None else proposed + 1
        return proposed
    return model, round_, calls, plane


def test_gate_probes_plain_prices_then_admits_cheap_speculation(monkeypatch):
    model, round_, calls, plane = harness(monkeypatch)
    assert round_(verify_seconds=0.06) == 0 and round_(verify_seconds=0.06) == 0   # plain probes
    assert round_(verify_seconds=0.12, committed=4) == 3                           # priced by a real round
    for _ in range(3):
        assert round_(verify_seconds=0.12, committed=4) == 3
    receipt = plane._mlx2_dspark_miss_gate.receipt()
    assert receipt['last_decision']['admit'] and receipt['last_decision']['priced']
    assert receipt['plain_tokens_per_second'] == pytest.approx(1 / 0.065)
    assert model.admission_stats['priced_admits'] >= 3 and 'handoffs' not in model.admission_stats
    assert calls[:2] == [0, 0] and set(calls[2:]) == {3}


def test_gate_hands_off_when_expert_misses_make_speculation_lose(monkeypatch):
    model, round_, calls, plane = harness(monkeypatch)
    round_(verify_seconds=0.06)
    round_(verify_seconds=0.06)
    # The verify round reads 2 GiB of missed experts at 10 GiB/s.
    round_(verify_seconds=0.35, miss_bytes=2 * GIB, read_seconds=0.2, committed=4)
    assert round_(verify_seconds=0.06) == 0                    # first rejection: one plain round
    with pytest.raises(DraftUnavailable) as caught:
        round_(verify_seconds=0.06)
    assert caught.value.ordinary_reason == gate.REASON
    last = model.admission_stats['last_handoff']['last_decision']
    assert last['reason'] == 'expected_round_exceeds_plain_time' and last['handoff']
    assert last['expected_seconds'] > last['plain_seconds']
    assert last['bandwidth_source'] == 'request' and last['bytes_per_second'] == pytest.approx(10 * GIB)


def test_off_and_zero_length_requests_bypass_the_gate(monkeypatch):
    model, round_, calls, plane = harness(monkeypatch, admission='off')
    for _ in range(3):
        assert round_(verify_seconds=0.5, miss_bytes=8 * GIB, read_seconds=0.1) == 3
    assert not hasattr(plane, '_mlx2_dspark_miss_gate') and model.admission_stats == {}
    model, round_, calls, plane = harness(monkeypatch)
    model.draft_distributions([5], Hidden(1), [plane], 0, [None], [0.0])
    assert not hasattr(plane, '_mlx2_dspark_miss_gate')


def test_gate_state_is_request_local(monkeypatch):
    model, round_, calls, plane = harness(monkeypatch)
    round_(verify_seconds=0.06)
    other = SimpleNamespace()
    model.draft_distributions([5], Hidden(1), [other], 3, [None], [0.0])
    assert plane._mlx2_dspark_miss_gate is not other._mlx2_dspark_miss_gate
    assert other._mlx2_dspark_miss_gate.plain_rounds == 0
