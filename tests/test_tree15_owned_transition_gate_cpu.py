"""Host contracts for the fresh gate; no model or native runtime imports."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('transition_gate', ROOT / 'scripts/research/tree15_owned_transition_gate.py')
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def row():
    return {'response': {'mlx2': {
        'output_token_ids_sha256': 'token-hash', 'identity': 'law', 'cache_layout': 'opaque',
        'mode_transitions': [{'mode': mode, 'active_width': width} for mode, width in [
            ('b1_tree_eligible', 1), ('b2plus_shared_ordinary', 2), ('b1_tree_eligible', 1)]],
        'drafted': 15, 'qualified': False, 'cache_policy': 'disabled',
        'apcv2_lookup': False, 'apcv2_store': False},
        'choices': [{'message': {'content': 'actual output'}, 'finish_reason': 'length'}],
        'usage': {'prompt_tokens': 6950, 'completion_tokens': 192}}}


def test_gate_requires_ordered_actual_width_and_exact_same_native_law():
    a = row(); b = copy.deepcopy(a)
    assert gate.exact(a, b) and gate.check_transition(a)
    b['response']['mlx2']['identity'] = 'another-native-law'
    assert not gate.exact(a, b)
    b = copy.deepcopy(a); b['response']['mlx2']['output_token_ids_sha256'] = None
    assert not gate.exact(b, b)
    b = copy.deepcopy(a); b['response']['mlx2']['mode_transitions'].reverse()
    # Reversing the symmetric history is inconclusive; drop the return instead.
    b['response']['mlx2']['mode_transitions'].pop()
    assert not gate.check_transition(b)
    b = copy.deepcopy(a); b['response']['mlx2']['mode_transitions'][1]['active_width'] = 1
    assert not gate.check_transition(b)


def test_gate_poll_requires_a_new_completed_native_round(monkeypatch):
    states = iter([{'sequence': 2, 'mode': 'b2plus_shared_ordinary'},
                   {'sequence': 3, 'mode': 'b1_tree_eligible'},
                   {'sequence': 4, 'mode': 'b2plus_shared_ordinary'}])
    monkeypatch.setattr(gate, 'get', lambda *args: {'tensorfold_owned': {'last_round': next(states)}})
    monkeypatch.setattr(gate.time, 'sleep', lambda _: None)
    assert gate.wait_round('fixture', 'b2plus_shared_ordinary', after=2) == 4


def test_gate_lease_requires_both_owner_records_and_live_pid(tmp_path, monkeypatch):
    monkeypatch.setenv('GPUQ_SESSION', 's'); monkeypatch.setenv('GPUQ_LEASE', 'l')
    paths = [tmp_path / 'a', tmp_path / 'b']
    owner = {'session': 's', 'lease_id': 'l', 'pid': 123}
    for p in paths:
        p.mkdir(); (p / 'owner.json').write_text(json.dumps(owner))
    real_path = gate.Path
    monkeypatch.setattr(gate, 'Path', lambda name: paths[0] if name == '/Users/Shared/mlxuag/gpu.lock' else paths[1])
    calls = []; monkeypatch.setattr(gate.os, 'kill', lambda pid, sig: calls.append((pid, sig)))
    assert gate.lease_identity() == owner and calls == [(123, 0), (123, 0)]
    (paths[1] / 'owner.json').write_text(json.dumps({**owner, 'lease_id': 'other'}))
    with pytest.raises(RuntimeError, match='ownership'):
        gate.lease_identity()


def test_gate_source_binding_refuses_dirty_or_different_head(monkeypatch):
    results = iter(['wanted', ' M src/mlx2/server.py'])
    monkeypatch.setattr(gate.subprocess, 'check_output', lambda *args, **kwargs: next(results))
    with pytest.raises(RuntimeError, match='clean'):
        gate.source_binding('wanted')
    monkeypatch.setattr(gate.subprocess, 'check_output', lambda *args, **kwargs: 'wrong')
    with pytest.raises(RuntimeError, match='exact'):
        gate.source_binding('wanted')


@pytest.mark.parametrize('rss,clock,reason', [(49 << 30, 0, 'process_tree_rss'), (0, 241, 'wall_deadline')])
def test_supervisor_kills_entire_group_on_wall_or_rss(tmp_path, monkeypatch, rss, clock, reason):
    import sys
    from types import SimpleNamespace
    class Child:
        pid = 321
        returncode = None
        def poll(self): return self.returncode
        def wait(self, timeout): self.returncode = -15; return self.returncode
    child = Child(); launch = {}
    def spawn(command, **kwargs):
        launch.update(kwargs); launch['command'] = command; return child
    monkeypatch.setattr(gate.subprocess, 'Popen', spawn)
    monkeypatch.setattr(gate, 'source_binding', lambda _: {'revision': 'pin'})
    monkeypatch.setattr(gate, 'lease_identity', lambda: {'session': 'lease'})
    ticks = iter([0, clock, clock, clock])
    monkeypatch.setattr(gate.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(gate.time, 'sleep', lambda _: None)
    class Process:
        def __init__(self, pid): pass
        def children(self, recursive): return []
        def is_running(self): return True
        def memory_info(self): return SimpleNamespace(rss=rss)
    monkeypatch.setitem(sys.modules, 'psutil', SimpleNamespace(Process=Process, Error=RuntimeError))
    signals = []; monkeypatch.setattr(gate.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    output = tmp_path / 'receipt.json'
    args = SimpleNamespace(source_commit='pin', deadline_seconds=240, rss_gib=48, output=output)
    assert gate.supervise(args) == 1
    assert launch['start_new_session'] is True and launch['command'][-1] == '--child'
    assert signals[0] == (321, gate.signal.SIGTERM)
    assert json.loads(output.read_text())['supervisor']['failure'] == reason


@pytest.mark.parametrize('count,reason,cap,valid', [
    (98, 'stop', 192, True), (64, 'length', 64, True),
    (63, 'length', 64, False), (193, 'stop', 192, False),
    (0, 'stop', 192, False), (True, 'stop', 192, False),
    (98, 'cancelled', 192, False)])
def test_completion_cap_accepts_eos_only_with_valid_finish_semantics(count, reason, cap, valid):
    result = row()
    result['response']['usage']['completion_tokens'] = count
    result['response']['choices'][0]['finish_reason'] = reason
    assert gate.valid_completion(result, cap) is valid


def test_eos_cap_fix_preserves_exact_actual_usage_and_finish_comparison():
    actual = row(); control = copy.deepcopy(actual)
    for item in (actual, control):
        item['response']['usage']['completion_tokens'] = 98
        item['response']['choices'][0]['finish_reason'] = 'stop'
    assert gate.valid_completion(actual, 192) and gate.exact(actual, control)
    control['response']['usage']['completion_tokens'] = 97
    assert not gate.exact(actual, control)
    control = copy.deepcopy(actual); control['response']['choices'][0]['finish_reason'] = 'length'
    assert not gate.exact(actual, control)
    actual['response']['mlx2']['mode_transitions'].pop()
    assert not gate.check_transition(actual)
