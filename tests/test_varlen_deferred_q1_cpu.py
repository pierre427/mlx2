"""CPU fault boundaries for default-off staged Q1 evaluation deferral."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.paged_kv_write import NativeWriteBackend
from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend


def _backend(roots=()):
    arena = object.__new__(NativeWriteBackend)
    arena._closed = False
    arena.defer_staged_q1_eval = False
    arena._deferred_q1_write_roots = list(roots)
    arena._deferred_q1_read_roots = []
    arena.deferred_q1_failure_flushes = 0
    writer = SimpleNamespace(backend=arena, pending_epochs={},
                             ledger=SimpleNamespace(pending_count=0),
                             poisoned=False)
    backend = object.__new__(NativeQwen3PagedBackend)
    backend.writer = writer
    backend.timeout_s = .01
    backend._failed = False
    backend._orphaned_reads = {}
    backend.terminal_successes = 0
    return arena, backend


def test_deferred_q1_abort_closes_prepared_read_before_root_flush(monkeypatch):
    arena, backend = _backend((mx.array([1], dtype=mx.uint8),))
    use = SimpleNamespace(state="prepared")
    use.abort_before_submit = lambda: setattr(use, "state", "closed")
    observed = []
    monkeypatch.setattr(mx, "eval", lambda *roots: observed.append(roots))
    backend.abort_deferred_q1((), (use,))
    assert use.state == "closed"
    assert observed == [(arena._deferred_q1_write_roots[0],)]
    assert arena.deferred_q1_failure_flushes == 1
    assert backend._failed and backend.writer.poisoned
    assert not backend.retain_deferred_q1_roots((use,))


def test_deferred_q1_ambiguous_read_keeps_roots_after_writes_complete(monkeypatch):
    root = mx.array([1], dtype=mx.uint8)
    arena, backend = _backend((root,))
    read = SimpleNamespace(state="submitted", lease=SimpleNamespace(epoch=27))
    monkeypatch.setattr(mx, "eval", lambda *_roots: (_ for _ in ()).throw(
        RuntimeError("ambiguous evaluation")))
    with pytest.raises(RuntimeError, match="ambiguous evaluation"):
        backend.abort_deferred_q1((), (read,))
    assert not backend.writer.pending_epochs and backend.writer.ledger.pending_count == 0
    assert backend._orphaned_reads == {27: read}
    assert backend.writer.poisoned and backend._failed
    assert backend.retain_deferred_q1_roots((read,))
    arena.end_deferred_q1(retain_roots=backend.retain_deferred_q1_roots((read,)))
    assert arena.deferred_q1_roots() == (root,)


def test_deferred_q1_roots_clear_only_after_read_terminal():
    root = mx.array([1], dtype=mx.uint8)
    arena, backend = _backend((root,))
    read = SimpleNamespace(state="closed", lease=SimpleNamespace(epoch=27))
    backend._orphaned_reads[27] = read
    assert backend.retain_deferred_q1_roots((read,))
    del backend._orphaned_reads[27]
    assert not backend.retain_deferred_q1_roots((read,))
    arena.end_deferred_q1(retain_roots=False)
    assert arena.deferred_q1_roots() == ()


def test_grouped_write_defers_eager_submission_only_inside_explicit_scope():
    arena, _backend_owner = _backend()
    root = mx.array([1], dtype=mx.uint8)
    submissions = []
    arena._native = SimpleNamespace(grouped_q1_write=lambda *_args, **_kwargs: root)
    arena._arena = object()
    arena.stream = object()
    arena._mx = SimpleNamespace(async_eval=lambda value: submissions.append(value))
    arena.grouped_write_async_evals = 0
    arena.deferred_q1_write_roots = 0
    arena.begin_deferred_q1()
    assert arena.grouped_q1_write(None, None, (0, 1), (0, 0), 2, 128, 1) is root
    assert submissions == [] and arena.deferred_q1_roots() == (root,)
    assert arena.deferred_q1_write_roots == 1
    arena.end_deferred_q1()
    assert arena.deferred_q1_roots() == ()
    assert arena.grouped_q1_write(None, None, (0, 1), (0, 0), 2, 128, 2) is root
    assert submissions == [root] and arena.grouped_write_async_evals == 1
