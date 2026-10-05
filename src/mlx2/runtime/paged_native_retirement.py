"""Retire a native request only after terminal write/read proof."""

from __future__ import annotations


def reap_native_request_owner(owner, writer, backend=None) -> None:
    """Use one-way arena teardown for a terminal failed command buffer.

    An ambiguous submission remains pinned. The same failed callback path is
    used for host-injected and genuine Metal failure; only a GPU test can
    establish which caused a particular event.
    """
    # A failed asynchronous read may have timed out before its callback. The
    # backend retains that exact read lease; drain late events before asking
    # whether the arena is safe to tear down. Write and read queues differ.
    drain = getattr(backend, "drain_failed_read_events", None)
    if callable(drain):
        drain()
    poll = getattr(writer, "poll_completions", None)
    if callable(poll) and writer.pending_epochs:
        poll()
    if writer.poisoned:
        if writer.pending_epochs or writer.ledger.pending_count:
            return
        writer.teardown_failed_arena()
        owner.reap_failed_after_teardown()
        return
    owner.reap_retired()
    owner.reap_quarantine()


__all__ = ["reap_native_request_owner"]
