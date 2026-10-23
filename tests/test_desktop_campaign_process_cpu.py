"""Process identity cleanup with a synthetic process table; no child is spawned."""
from types import SimpleNamespace

import pytest

from mlx2 import desktop_campaign_process as owned


@pytest.fixture
def table(monkeypatch):
    rows, signals = {}, []
    class Process:
        def __init__(self, pid, created, session, parent=None, stubborn=False):
            self.pid, self.created, self.session = pid, created, session
            self.parent, self.stubborn, self.alive = parent, stubborn, True
            rows[pid] = self
        def create_time(self):
            return self.created
        def children(self, recursive=False):
            return [row for row in rows.values() if row.parent == self.pid]
        def is_running(self):
            return rows.get(self.pid) is self and self.alive
        def status(self):
            return "running" if self.is_running() else owned.psutil.STATUS_ZOMBIE
        def send_signal(self, signum):
            if not self.is_running():
                raise owned.psutil.NoSuchProcess(self.pid)
            signals.append((self.pid, self.created, signum))
            if not self.stubborn:
                self.alive = False
    def lookup(pid):
        if pid not in rows:
            raise owned.psutil.NoSuchProcess(pid)
        return rows[pid]
    monkeypatch.setattr(owned.psutil, "Process", lookup)
    monkeypatch.setattr(owned.psutil, "process_iter", lambda: list(rows.values()))
    monkeypatch.setattr(owned.os, "getsid", lambda pid: lookup(pid).session)
    clock = [0.0]
    monkeypatch.setattr(owned.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(owned.time, "sleep", lambda amount: clock.__setitem__(0, clock[0] + amount))
    leader = Process(10, 100, 10)
    raw = SimpleNamespace(pid=10, args=["synthetic"], returncode=None, poll=lambda: None, wait=lambda timeout: 0)
    return SimpleNamespace(rows=rows, signals=signals, Process=Process, leader=leader, raw=raw)


def test_reaped_leader_does_not_skip_retained_independent_descendant(table):
    child = table.Process(20, 101, 20, parent=10)
    process = owned.OwnedProcess(table.raw)
    # The child becomes orphaned in its own independent session.
    child.parent = None
    del table.rows[10]
    table.raw.returncode = 1
    table.raw.poll = lambda: 1
    process.stop(timeout=1)
    assert not child.alive
    assert [row[0] for row in table.signals] == [20]


def test_recycled_leader_pid_and_its_foreign_children_never_receive_signal(table):
    child = table.Process(20, 101, 20, parent=10)
    process = owned.OwnedProcess(table.raw)
    replacement = table.Process(10, 200, 10)
    foreign = table.Process(30, 201, 10, parent=10)
    process.stop(timeout=1)
    assert replacement.alive and foreign.alive
    assert not child.alive
    assert all(pid == 20 for pid, _created, _signal in table.signals)


def test_same_session_orphan_discovered_after_leader_reaped(table):
    process = owned.OwnedProcess(table.raw)
    del table.rows[10]
    child = table.Process(20, 101, 10)
    process.stop(timeout=1)
    assert not child.alive
    assert [row[0] for row in table.signals] == [20]


def test_cleanup_failure_is_not_reported_as_success(table):
    table.Process(20, 101, 20, parent=10, stubborn=True)
    process = owned.OwnedProcess(table.raw)
    with pytest.raises(RuntimeError, match="cleanup incomplete"):
        process.stop(timeout=0)
    assert table.signals
    assert all(signum == owned.signal.SIGKILL for _pid, _created, signum in table.signals)


def test_retained_child_new_independent_descendants_survive_no_cleanup_gap(table):
    child = table.Process(20, 101, 20, parent=10)
    process = owned.OwnedProcess(table.raw)
    del table.rows[10]
    child.parent = None
    grandchild = table.Process(30, 102, 30, parent=20)
    process.stop(timeout=1)
    assert not child.alive and not grandchild.alive
    assert {row[0] for row in table.signals} == {20, 30}
