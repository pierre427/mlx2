"""Host protocol for terminal-proved native request retirement."""

from types import SimpleNamespace

from mlx2.runtime.paged_native_retirement import reap_native_request_owner


class Owner:
    def __init__(self):
        self.calls = []

    def reap_retired(self):
        self.calls.append("retired")

    def reap_quarantine(self):
        self.calls.append("quarantine")

    def reap_failed_after_teardown(self):
        self.calls.append("failed_teardown")


def test_healthy_native_owner_uses_ordinary_terminal_reap():
    owner = Owner()
    writer = SimpleNamespace(poisoned=False)
    reap_native_request_owner(owner, writer)
    assert owner.calls == ["retired", "quarantine"]


def test_failed_command_retains_arena_until_every_epoch_is_terminal():
    owner = Owner()
    calls = []
    writer = SimpleNamespace(
        poisoned=True, pending_epochs=(4,),
        ledger=SimpleNamespace(pending_count=1),
        teardown_failed_arena=lambda: calls.append("teardown"))
    reap_native_request_owner(owner, writer)
    assert not calls and not owner.calls
    writer.pending_epochs = ()
    writer.ledger.pending_count = 0
    reap_native_request_owner(owner, writer)
    assert calls == ["teardown"]
    assert owner.calls == ["failed_teardown"]


def test_retirement_drains_late_reads_and_writes_before_failed_teardown():
    owner = Owner()
    calls = []
    writer = SimpleNamespace(
        poisoned=True, pending_epochs=(9,),
        ledger=SimpleNamespace(pending_count=2),
        teardown_failed_arena=lambda: calls.append("teardown"))

    def drain_reads():
        calls.append("read")
        writer.ledger.pending_count -= 1
        return 1

    def drain_writes():
        calls.append("write")
        writer.pending_epochs = ()
        writer.ledger.pending_count -= 1

    backend = SimpleNamespace(drain_failed_read_events=drain_reads)
    writer.poll_completions = drain_writes
    reap_native_request_owner(owner, writer, backend)
    assert calls == ["read", "write", "teardown"]
    assert owner.calls == ["failed_teardown"]
