"""CPU-only complete-request pairing, mechanism and rate contracts."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/research'))
SPEC = importlib.util.spec_from_file_location('owned_perf_gate', ROOT / 'scripts/research/tree15_owned_request_perf_gate.py')
perf = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(perf)


def request(width, route, cap):
    count = 98 if cap == 192 else cap
    mode = 'b2plus_shared_ordinary' if width == 2 else 'b1_tree_eligible' if route == 'auto' else 'b1_serial_ordinary'
    return {'wall_seconds': 15., 'response': {
        'choices': [{'message': {'content': 'exact output'}, 'finish_reason': 'stop' if cap == 192 else 'length'}],
        'usage': {'prompt_tokens': 6988, 'completion_tokens': count},
        'mlx2': {'qualified': False, 'cache_policy': 'disabled', 'apcv2_lookup': False, 'apcv2_store': False,
                 'identity': 'law', 'cache_layout': 'opaque', 'output_token_ids_sha256': 'actual-token-hash',
                 'drafted': 15 if width == 1 and route == 'auto' else 0,
                 'active_width_modes': {mode: 3}, 'active_width_counts': {str(width): 3},
                 'timing': {'clock': 'host_monotonic_ns', 'cached_prompt_tokens': 0, 'prompt_tokens': 6988,
                            'completion_tokens': count, 'prefill_to_first_token_seconds': 10.,
                            'first_token_offset_seconds': 10., 'decode_seconds': 5., 'decode_tokens_per_second': (count-1)/5.}}}}


def arm(width, route, bodies):
    return {'width': width, 'route_requested': route,
            'requests': [request(width, route, b['max_tokens']) for b in bodies[:width]],
            'complete_cohort_wall_seconds': 20. if route == 'serial' else 19.,
            'b1_tree_edges_observed': width == 1 and route == 'auto'}


def test_three_fixed_pairs_alternate_orders_and_require_exact_ordinary_control(monkeypatch):
    calls = []
    def fake_arm(base, bodies, width, route):
        calls.append((width, route, [b['max_tokens'] for b in bodies]))
        return arm(width, route, bodies)
    monkeypatch.setattr(perf, 'run_arm', fake_arm)
    monkeypatch.setattr(perf, 'host_snapshot', lambda: {'thermal': 'fixture'})
    persisted = []
    pairs = perf.measured_pairs('host', [{'max_tokens': 192}, {'max_tokens': 64}], 'law', persist=lambda: persisted.append(1))
    assert [(w,r) for w,r,_ in calls] == [(1,'serial'),(1,'auto'),(2,'serial'),(2,'auto'),
        (1,'auto'),(1,'serial'),(2,'auto'),(2,'serial'),(1,'serial'),(1,'auto'),(2,'serial'),(2,'auto')]
    assert all(caps == ([192] if width == 1 else [64,64]) for width, _, caps in calls)
    assert len(pairs) == 6 and len(persisted) == 18 and all(p['exact_native_serial'] for p in pairs)
    summary = perf.pair_summary(pairs)
    assert summary['1']['pairs'] == summary['2']['pairs'] == 3
    assert summary['1']['median_auto_over_serial_complete_wall_ratio'] == .95
    assert 'same shared ordinary' in summary['2']['comparison']
    pairs[0]['auto']['requests'][0]['response']['usage']['completion_tokens'] = 97
    assert not perf.gate.exact(pairs[0]['auto']['requests'][0], pairs[0]['serial']['requests'][0])


def test_b2_requires_actual_shared_width_and_exposes_auto_b1_edges():
    a = arm(2, 'auto', [{'max_tokens':64}, {'max_tokens':64}])
    perf.check_arm(a, 'law', [64,64])
    first = a['requests'][0]['response']['mlx2']
    first['drafted'] = 15
    with pytest.raises(RuntimeError, match='observed auto B1 edge'): perf.check_arm(a, 'law', [64,64])
    first['active_width_modes']['b1_tree_eligible'] = 1
    perf.check_arm(a, 'law', [64,64])
    a['route_requested'] = 'serial'
    with pytest.raises(RuntimeError, match='observed auto B1 edge'): perf.check_arm(a, 'law', [64,64])
    a = arm(2, 'serial', [{'max_tokens':64}, {'max_tokens':64}])
    a['requests'][1]['response']['mlx2']['active_width_counts'] = {'1': 3}
    with pytest.raises(RuntimeError, match='width2'): perf.check_arm(a, 'law', [64,64])


def test_complete_cohort_rate_uses_actual_tokens_and_wall_with_prefill(monkeypatch):
    monkeypatch.setattr(perf.gate, 'post', lambda base, body, route: request(2, route, body['max_tokens']))
    monkeypatch.setattr(perf.gate, 'wait_idle', lambda base: {'active_requests':0, 'pending_requests':0, 'opening_requests':0})
    times = iter([100.,120.]); monkeypatch.setattr(perf.time, 'monotonic', lambda: next(times))
    result = perf.run_arm('fixture', [{'max_tokens':64}, {'max_tokens':64}], 2, 'serial')
    assert result['aggregate_completion_tokens'] == 128
    assert result['aggregate_complete_request_tokens_per_second'] == 6.4
    assert 'includes prefill' in result['rate_definition']
    assert result['per_request_metrics'][0]['native_decode_seconds_including_peer_prefill_pauses'] == 5.
    assert result['per_request_metrics'][0]['peer_prefill_seconds_overlap_not_subtracted'] == [10.]


def test_perf_deadline_extension_keeps_strict420_max_and_transition240(monkeypatch, tmp_path):
    monkeypatch.setattr(perf.gate, 'source_binding', lambda _: {'files': {}})
    monkeypatch.setattr(perf.gate, 'lease_identity', lambda: {'session':'fixture'})
    args = SimpleNamespace(source_commit='pin', output=tmp_path/'receipt.json', deadline_seconds=421, rss_gib=48)
    with pytest.raises(ValueError, match='420'): perf.gate.supervise(args, max_deadline_seconds=420)
    args.deadline_seconds = 241
    with pytest.raises(ValueError, match='240'): perf.gate.supervise(args)
