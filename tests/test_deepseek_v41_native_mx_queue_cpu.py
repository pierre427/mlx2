"""Ready-queue coordination tests: fake child/CPG and temporary locks only."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name='scripts/research/probe_ds41_native_mx.py'):
    spec = importlib.util.spec_from_file_location('_native_mx_queue_'+Path(name).stem, ROOT/name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def queued(module):
    return dict(queue_ready=True, queue_policy=dict(module.QUEUE_POLICY))


def test_default_spec_remains_unqueued():
    assert load().validate_queue_spec({}) is False


def test_queue_spec_is_explicit_and_sealed():
    module = load()
    assert module.validate_queue_spec(queued(module)) is True


@pytest.mark.parametrize('bad', ['false', 'missing', 'policy_only', 'timeout', 'physical', 'int_false', 'extra'])
def test_queue_spec_refuses_unsealed_changed_or_idle_physical_ownership(bad):
    module = load()
    spec = queued(module)
    if bad == 'false': spec['queue_ready'] = False
    elif bad == 'missing': spec.pop('queue_policy')
    elif bad == 'policy_only': spec.pop('queue_ready')
    elif bad == 'timeout': spec['queue_policy']['wait_seconds'] = 1800
    elif bad == 'physical': spec['queue_policy']['physical_locks_owned_while_queued'] = True
    elif bad == 'int_false': spec['queue_policy']['physical_locks_owned_while_queued'] = 0
    else: spec['queue_policy']['priority'] = 'first'
    with pytest.raises(ValueError): module.validate_queue_spec(spec)


@pytest.mark.parametrize('foreign,waiters', [(True, False), (False, True), (True, True)])
def test_default_contention_refusal_unchanged_but_ready_can_join(foreign, waiters):
    module = load()
    before = {'foreign_model_processes': [{'pid': 123}] if foreign else []}
    ledger = [{'name': 'prior'}] if waiters else []
    with pytest.raises(RuntimeError): module.prelease_admission(before, ledger, queue_ready=False)
    module.prelease_admission(before, ledger, queue_ready=True)


class Clock:
    def __init__(self, step=1): self.now, self.step = 0, step
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += self.step


class Child:
    returncode = 0
    def __init__(self, out, states): self.out, self.states = out, iter(states)
    def poll(self):
        value = next(self.states, 'waiting')
        if value == 'done': return 0
        if value != 'missing': (self.out/'paired-locks.json').write_text(json.dumps({'status': value}))
        return None


def test_queue_renewal_host_checks_and_separate_running_deadline(tmp_path, monkeypatch):
    module = load()
    clock = Clock(200)
    checks, floors = [], []
    lease = SimpleNamespace(check=lambda: checks.append(clock.now))
    monkeypatch.setattr(module, 'host_admission', lambda floor: floors.append(floor))
    child = Child(tmp_path, ['waiting', 'waiting', 'waiting', 'running', 'running', 'running', 'done'])
    assert module.monitor_child(child, lease, tmp_path, queue_ready=True, clock=clock, sleep=clock.sleep) == 0
    assert checks == [0, 200, 400, 600, 800, 1000]
    assert floors == [9 << 30]*3
    state = json.loads((tmp_path/'ready-queue-state.json').read_text())
    assert state['running_observed'] and state['task_claim_is_physical_gpu_lock'] is False


@pytest.mark.parametrize('state', ['waiting', 'missing'])
def test_queue_deadline_is_bounded_even_before_wrapper_receipt(tmp_path, monkeypatch, state):
    module = load()
    monkeypatch.setattr(module, 'host_admission', lambda floor: None)
    clock = Clock(601)
    with pytest.raises(TimeoutError, match='ready queue'):
        module.monitor_child(Child(tmp_path, [state]*3), SimpleNamespace(check=lambda: None),
            tmp_path, queue_ready=True, clock=clock, sleep=clock.sleep)


def test_running_deadline_and_default_deadline_stay_bounded(tmp_path, monkeypatch):
    module = load()
    monkeypatch.setattr(module, 'host_admission', lambda floor: None)
    for queue in (False, True):
        clock = Clock(601)
        with pytest.raises(TimeoutError, match='component deadline'):
            module.monitor_child(Child(tmp_path, ['running']*3), SimpleNamespace(check=lambda: None),
                tmp_path, queue_ready=queue, clock=clock, sleep=clock.sleep)


def test_lease_failure_or_live_memory_refusal_stops_monitor_before_dispatch(tmp_path, monkeypatch):
    module = load()
    def reject(): raise RuntimeError('coordination lease lost')
    with pytest.raises(RuntimeError, match='lease lost'):
        module.monitor_child(Child(tmp_path, ['waiting']), SimpleNamespace(check=reject), tmp_path, queue_ready=True)
    def low(floor): raise MemoryError('fresh live floor refused')
    monkeypatch.setattr(module, 'host_admission', low)
    with pytest.raises(MemoryError):
        module.monitor_child(Child(tmp_path, ['waiting']), SimpleNamespace(check=lambda: None), tmp_path, queue_ready=True)


def test_unit_cli_cannot_enable_queue_outside_sealed_spec(tmp_path):
    with pytest.raises(SystemExit): load().main(['unit', '--output', str(tmp_path), '--queue-ready'])


@pytest.mark.parametrize('interrupt', [False, True])
def test_existing_wrapper_waits_without_locks_and_preserves_predecessor(tmp_path, monkeypatch, interrupt):
    """Exercise unchanged wrapper using fake owner and two temp lock paths."""
    wrapper = load('scripts/run_with_gpu_locks.py')
    locks = (tmp_path/'a.lock', tmp_path/'b.lock')
    for path in locks:
        path.mkdir()
        (path/'owner.json').write_text('{"foreign_fixture":true}')
    waiters = tmp_path/'waiters'
    waiters.mkdir()
    prior = waiters/'prior.fixture.1.1'
    prior.touch()
    monkeypatch.setattr(wrapper, 'LOCKS', locks)
    monkeypatch.setattr(wrapper, 'WAITERS', waiters)
    monkeypatch.setattr(wrapper, 'foreign_model_processes', lambda: [])
    turns, calls = [], []
    def earlier(mine, created): return prior.name if not turns else None
    monkeypatch.setattr(wrapper, 'earlier_waiter', earlier)
    def sleep(seconds):
        assert not calls  # Neither CPG validation, acquire nor child launch yet.
        assert all(json.loads((path/'owner.json').read_text()) == {'foreign_fixture': True} for path in locks)
        assert len(list(waiters.iterdir())) == 2
        turns.append(seconds)
        if interrupt: raise KeyboardInterrupt()
        if len(turns) == 2:
            for path in locks:
                (path/'owner.json').unlink()
                path.rmdir()
                path.touch()
    monkeypatch.setattr(wrapper.time, 'sleep', sleep)
    def observe(*args):
        calls.append('observe')
        return dict(cpg_task='t', cpg_session='s', cpg_generation=1, agent_id='a'), {}
    monkeypatch.setattr(wrapper, 'observe_cpg', observe)
    acquire = wrapper.acquire
    def owned(owner):
        calls.append('acquire')
        return acquire(owner)
    monkeypatch.setattr(wrapper, 'acquire', owned)
    class Immediate:
        pid, returncode = 4242, 0
        def poll(self): return 0
        def wait(self, timeout=None): return 0
    def launch(*args, **kwargs):
        calls.append('launch')
        return Immediate()
    monkeypatch.setattr(wrapper.subprocess, 'Popen', launch)
    receipt = tmp_path/'paired.json'
    result = wrapper.main(['--cpg-owner-lock', 'fixture', '--cpg-task', 't', '--session', 's',
        '--label', 'native', '--receipt', str(receipt), '--', 'never-executed-fixture'])
    assert result == (130 if interrupt else 0)
    assert list(waiters.iterdir()) == [prior]
    if interrupt:
        assert not calls and all(path.is_dir() for path in locks)
    else:
        assert calls == ['observe', 'acquire', 'launch']
        assert all(path.is_file() and path.stat().st_size == 0 for path in locks)
