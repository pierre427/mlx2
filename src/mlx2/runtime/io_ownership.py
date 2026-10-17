"""Bounded worker ownership for filesystem reads; no device operations.

A deadline revokes consumption, not the underlying syscall. Fixed daemon
workers keep their buffers and descriptor owner until they return. Quarantine
is scoped to an owner key (the storage a pool serves): a new pool for the same
owner is refused while abandoned work remains, and a process-wide cap bounds
the hung workers of all owners together. See peer-survey I/O provenance and
provenance/sweep-20261016-readpool-owner-scope.json.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout

CONTROL_TIMEOUT_SECONDS = 60.0
FINALIZE_TIMEOUT_SECONDS = 2.0
# Hung workers quarantine may retain across every owner in the process. Once
# reached, no owner may build a new pool until some of that work drains.
MAX_QUARANTINED_WORKERS = 64
log = logging.getLogger(__name__)


class CloseAction:
    """Stable pure-I/O finalizer identity, retained until actual completion."""

    def __init__(self, close):
        self.close = close


class ReadTimeout(RuntimeError):
    """A streamed read exceeded its deadline; reload only after it drains."""


class ReadQueueFull(RuntimeError):
    """An optional I/O queue has reached its bounded ownership window."""


class QuarantineCapReached(ReadTimeout):
    """The process-wide hung-worker cap refuses new pools for every owner."""


# Quarantined worker token (a ReadPool, or a writer an owner registered with
# quarantine_worker) -> (owner key, hung workers it may retain).
_QUARANTINED_READ_POOLS = {}
_READ_POOLS_LOCK = threading.Lock()


def _after_fork():
    global _QUARANTINED_READ_POOLS, _READ_POOLS_LOCK
    _QUARANTINED_READ_POOLS = {}
    _READ_POOLS_LOCK = threading.Lock()


def storage_owner(kind, path):
    """Quarantine owner key for one storage resource; string work only, no I/O."""
    return (str(kind), os.path.abspath(os.path.expanduser(os.fspath(path))))


def quarantine_worker(token, owner, workers=1):
    """Retain a non-pool worker (e.g. a spill writer) in owner quarantine."""
    with _READ_POOLS_LOCK:
        _QUARANTINED_READ_POOLS[token] = (owner, max(1, int(workers)))


def release_worker(token):
    with _READ_POOLS_LOCK:
        _QUARANTINED_READ_POOLS.pop(token, None)


def quarantined_workers():
    with _READ_POOLS_LOCK:
        return sum(workers for _owner, workers in _QUARANTINED_READ_POOLS.values())


def _refuse_new_pool_locked(owner):
    hung = sum(workers for _owner, workers in _QUARANTINED_READ_POOLS.values())
    if hung >= MAX_QUARANTINED_WORKERS:
        raise QuarantineCapReached(
            f"{hung} hung I/O workers reached the process cap "
            f"MAX_QUARANTINED_WORKERS={MAX_QUARANTINED_WORKERS}; no new I/O pool "
            "for any owner until some drain")
    # owner=None is anonymous: its quarantine counts toward the cap only.
    if owner is not None and any(item == owner for item, _workers
                                 in _QUARANTINED_READ_POOLS.values()):
        raise ReadTimeout(f"previous I/O reads have not drained for owner {owner!r}")


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


class ReadPool:
    """Fixed, daemon I/O workers with explicit late-read ownership.

    ThreadPoolExecutor joins running calls at interpreter exit even after
    shutdown(wait=False). An uninterruptible read would therefore undo the
    serving timeout during shutdown. These workers are daemon threads, retain
    their arguments until the call returns, and never run MLX operations.
    A timed-out pool refuses further submissions. While such a pool still
    owns reads (or an uncertain close), creating another pool for the same
    ``owner`` is refused, so an unload/reload never overlaps new work with a
    hung worker on the same storage. Other owners are unaffected until the
    hung workers of all owners reach ``MAX_QUARANTINED_WORKERS``; then every
    new pool is refused with :class:`QuarantineCapReached`. ``owner`` is a
    hashable key supplied by the creator, normally ``storage_owner(...)``;
    ``None`` is anonymous and only counts toward the cap.
    """

    def __init__(self, max_workers, thread_name_prefix, *, timeout=None, max_pending=None,
                 owner=None):
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("I/O timeout must be finite and positive")
        if max_pending is not None and max_pending <= 0:
            raise ValueError("I/O pending limit must be positive")
        with _READ_POOLS_LOCK:
            _refuse_new_pool_locked(owner)
        self.owner = owner
        self._lock = threading.Lock()
        self._owner_pid = os.getpid()
        self._timeout = timeout
        self._max_pending = max_pending
        self._deadlines = {}
        self._changed = threading.Event()
        self._queue = queue.SimpleQueue()
        self._pending = set()
        self._closed = False
        self._failure = None
        self._timeouts = 0
        self._close_handles = None
        self._handles_owner = None
        self._finalizer = None
        self._threads = []
        self._max_workers = max(1, int(max_workers))
        self._thread_name = thread_name_prefix
        self._watcher = None

    def submit(self, function, *args):
        if os.getpid() != self._owner_pid:
            raise RuntimeError("I/O pool crossed fork; reopen its owning resource")
        with self._lock:
            if self._failure is not None:
                raise ReadTimeout(self._failure)
            if self._closed:
                raise RuntimeError("I/O reader is closed")
            # A finished future stays in _pending until its done-callback
            # runs, which can be after result() has already woken the caller.
            if (self._max_pending is not None
                    and sum(not item.done() for item in self._pending) >= self._max_pending):
                raise ReadQueueFull("I/O queue ownership window is full")
            future = Future()
            self._pending.add(future)
            if self._timeout is not None:
                self._deadlines[future] = time.monotonic() + self._timeout
            future.add_done_callback(self._completed)
            self._queue.put((future, function, args))
            if len(self._threads) < min(self._max_workers, len(self._pending)):
                thread = threading.Thread(target=self._run, daemon=True,
                                          name=f"{self._thread_name}-{len(self._threads)}")
                self._threads.append(thread)
                thread.start()
            if self._timeout is not None and self._watcher is None:
                self._watcher = threading.Thread(target=self._watch, daemon=True,
                                                name=f"{self._thread_name}-deadline")
                self._watcher.start()
            self._changed.set()
            return future

    def _completed(self, future):
        with self._lock:
            self._pending.discard(future)
            self._deadlines.pop(future, None)
            self._changed.set()
            if not self._pending:
                if self._close_handles is not None:
                    self._start_finalizer_locked()
                elif self._finalizer is None:
                    with _READ_POOLS_LOCK:
                        _QUARANTINED_READ_POOLS.pop(self, None)

    def _quarantine_locked(self):
        # Caller holds self._lock. Bound, not exact: any busy worker may hang.
        workers = min(len(self._pending), len(self._threads))
        workers += int(self._finalizer is not None and not self._finalizer.done())
        quarantine_worker(self, self.owner, workers)

    def call(self, function, *args):
        """Run pure host I/O with a bounded caller wait and revoked late result."""
        future = self.submit(function, *args)
        try:
            return future.result(timeout=self._timeout or CONTROL_TIMEOUT_SECONDS)
        except FutureTimeout as error:
            failure = ReadTimeout("I/O operation exceeded its deadline")
            self.fail(failure)
            raise failure from error

    def _start_finalizer_locked(self):
        if self._finalizer is not None:
            return
        self._finalizer = Future()
        close, self._close_handles = self._close_handles, None

        def finalize():
            try:
                close()
            except BaseException as error:  # noqa: BLE001 - uncertain close must retain ownership
                with self._lock:
                    self._failure = self._failure or f"I/O finalizer failed: {error}"
                    self._quarantine_locked()
                self._finalizer.set_exception(error.with_traceback(None))
                log.error("I/O finalizer failed; owner remains quarantined: %s", error)
            else:
                # Publish completion only after the resource close returned.
                # In particular, a timed-out os.close must never be retried on
                # a descriptor number the kernel could already have reused.
                with self._lock, _READ_POOLS_LOCK:
                    _QUARANTINED_READ_POOLS.pop(self, None)
                self._finalizer.set_result(None)

        threading.Thread(target=finalize, daemon=True,
                         name=f"{self._thread_name}-close").start()

    def _watch(self):
        while True:
            with self._lock:
                if self._closed:
                    return
                self._changed.clear()
                delay = (max(0.0, min(self._deadlines.values()) - time.monotonic())
                         if self._deadlines else None)
            if not self._changed.wait(delay):
                with self._lock:
                    expired = any(deadline <= time.monotonic()
                                  for deadline in self._deadlines.values())
                if expired:
                    self.fail(ReadTimeout("I/O worker exceeded its deadline"))
                    return

    @property
    def failure(self):
        with self._lock:
            return self._failure

    def publish(self, function, *args):
        """Commit a bounded host result atomically against close/failure.

        The callback must not do I/O, MLX work, or reenter this pool. This is
        for cache publication only; it never authorizes another storage call.
        """
        with self._lock:
            if self._closed or self._failure is not None:
                return False
            expired = any(deadline <= time.monotonic()
                          for deadline in self._deadlines.values())
            if not expired:
                function(*args)
                return True
        self.fail(ReadTimeout("I/O worker exceeded its deadline"))
        return False

    def _run(self):
        while True:
            task = self._queue.get()
            if task is None:
                return
            future, function, args = task
            if future.set_running_or_notify_cancel():
                try:
                    result = function(*args)
                    with self._lock:
                        deadline = self._deadlines.get(future)
                        expired = deadline is not None and deadline <= time.monotonic()
                        failure = self._failure
                    if expired:
                        error = ReadTimeout("I/O worker exceeded its deadline")
                        self.fail(error)
                        del result
                        raise error
                    if failure is not None:
                        del result
                        raise ReadTimeout(failure)
                    future.set_result(result)
                    del result
                except BaseException as error:  # noqa: BLE001 - report on the model thread
                    future.set_exception(error)
            # Do not retain the previous read's bytes while waiting for work.
            del task, future, function, args

    def fail(self, reason):
        with self._lock:
            first = self._failure is None
            self._failure = self._failure or str(reason)
            self._timeouts += int(first and isinstance(reason, ReadTimeout))
            if self._pending:
                self._quarantine_locked()
        self.close()

    def counters(self):
        with self._lock:
            return {
                "weight_read_inflight": len(self._pending),
                "weight_read_failed": int(self._failure is not None),
                "weight_read_timeouts_total": self._timeouts,
                "io_finalizer_inflight": int(self._finalizer is not None and not self._finalizer.done()),
            }

    def close(self, handles=None, *, timeout=None):
        """Revoke work and finalize descriptors once, with bounded caller wait.

        Running reads and the finalizer retain their exact owners. A late or
        failed close blocks replacement pools for the same owner; no retry can
        close a reused fd.
        The callback must perform only host I/O, never device cleanup.
        """
        with self._lock:
            if handles is not None and self._handles_owner is None:
                self._handles_owner = handles
                self._close_handles = handles.close
            elif handles is not None and self._handles_owner is not handles:
                raise ValueError("a read pool cannot close different descriptor owners")
            if not self._closed:
                self._closed = True
                self._changed.set()
                for _thread in self._threads:
                    self._queue.put(None)
            pending = tuple(self._pending)
            if pending:
                self._failure = self._failure or "I/O reader closed with reads in flight"
                self._quarantine_locked()
            if not pending and self._close_handles is not None:
                self._start_finalizer_locked()
            finalizer = self._finalizer
        # Cancellation only discards queued calls, never a running syscall.
        # Callbacks acquire our lock, so cancel outside it.
        for future in pending:
            future.cancel()
        if finalizer is not None and handles is not None:
            try:
                finalizer.result(timeout=FINALIZE_TIMEOUT_SECONDS if timeout is None else timeout)
            except FutureTimeout:
                with self._lock:
                    if not finalizer.done():
                        self._failure = self._failure or "I/O finalizer exceeded its deadline"
                        self._quarantine_locked()
                log.warning("I/O finalizer deadline exceeded; owner remains quarantined")
                return False
            except BaseException:  # noqa: BLE001 - reported by finalizer, never retry uncertain close
                return False
        return not pending and (finalizer is None or finalizer.done())


def close_read_pool(pool, handles):
    if isinstance(pool, ReadPool):
        pool.close(handles)
    else:
        # Compatibility for externally supplied test executors. Production
        # installs always use ReadPool and its deferred descriptor close.
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if handles is not None:
            handles.close()


def call_io(name, function, *args, timeout=None, owner=None):
    """One bounded host-only inventory operation with locally owned handles.

    The function must close its own files before returning and must not return
    an unowned descriptor. Timed-out workers stay quarantined (for ``owner``)
    until that scope actually exits; this helper never performs device work.
    """
    pool = ReadPool(1, name, timeout=CONTROL_TIMEOUT_SECONDS if timeout is None else timeout,
                    max_pending=1, owner=owner)
    try:
        return pool.call(function, *args)
    finally:
        pool.close()
