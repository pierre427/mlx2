# SPDX-License-Identifier: Apache-2.0
"""DPara candidate lifetime and scheduling, independent of an MLX device.

The callback boundary admits CPU tests and explicit placement by a caller.
Submitting work to a thread does not prove GPU execution overlap.
"""

from collections.abc import Callable
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any


@dataclass(frozen=True)
class DParaBinding:
    request_id: str
    target_revision: str
    draft_revision: str
    generation: int

    def __post_init__(self):
        if (
            any(
                not isinstance(value, str) or not value
                for value in (
                    self.request_id,
                    self.target_revision,
                    self.draft_revision,
                )
            )
            or type(self.generation) is not int
            or self.generation < 0
        ):
            raise ValueError(
                "DPara requires request, revisions and nonnegative generation"
            )

    def next(self):
        return DParaBinding(
            self.request_id,
            self.target_revision,
            self.draft_revision,
            self.generation + 1,
        )


@dataclass(frozen=True)
class DParaVerification:
    binding: DParaBinding
    accepted: int
    bonus: int
    target_features: Any


class DParaPrepared:
    """One-shot, request-owned branch set; rejected state is never published.

    The model owns ``payload``. Consumers select through resolve and must not
    mutate its immutable arrays. A discarded ticket cannot be revived.
    """

    def __init__(self, binding, spine, context, payload):
        if hasattr(context, "binding") and context.binding != binding:
            raise ValueError("DPara prepared handle/context binding mismatch")
        self._binding = binding
        self._spine = tuple(spine)
        self._context = context
        self._payload = payload
        self._state = "ready"
        self._lock = Lock()
        self._discard_requested = Event()

    @property
    def binding(self):
        return self._binding

    @property
    def spine(self):
        return self._spine

    @property
    def context(self):
        return self._context

    @property
    def state(self):
        with self._lock:
            return self._state

    @property
    def payload(self):
        with self._lock:
            if self._state != "ready" or self._discard_requested.is_set():
                raise ValueError("DPara ticket is no longer ready")
            return self._payload

    def resolve(self, verification, finalize):
        with self._lock:
            if self._state != "ready" or self._discard_requested.is_set():
                raise ValueError("DPara ticket is no longer ready")
            if verification.binding != self.binding:
                raise ValueError("DPara request/revision/generation mismatch")
            if type(
                verification.accepted
            ) is not int or not 0 <= verification.accepted < len(self.spine):
                raise ValueError("DPara accepted length is outside prepared branches")
            # Consume before callbacks: exceptions also invalidate tentative state.
            self._state = "resolving"
            try:
                result = finalize(self._payload, verification)
                if self._discard_requested.is_set():
                    raise ValueError("DPara ticket was discarded during finalization")
            except BaseException:
                self._state = "discarded"
                self._payload = None
                raise
            self._state = "resolved"
            self._payload = None
            return result

    def discard(self):
        self._discard_requested.set()
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._state == "ready":
                self._state = "discarded"
                self._payload = None
        finally:
            self._lock.release()


class DParaRound:
    """Overlap precomputation with the supplied verifier; no implicit commit."""

    def __init__(self, binding: DParaBinding, future: Future):
        self._binding = binding
        self._future = future
        self._cancelled = False
        self._cancel_event = Event()
        self._consumed = False
        self._lock = Lock()
        future.add_done_callback(self._discard_if_cancelled)

    @property
    def binding(self):
        return self._binding

    def _discard_if_cancelled(self, future):
        with self._lock:
            cancelled = self._cancelled
        if cancelled and not future.cancelled() and future.exception() is None:
            future.result().discard()

    def cancel(self):
        self._cancel_event.set()
        with self._lock:
            self._cancelled = True
        self._future.cancel()
        if self._future.done():
            self._discard_if_cancelled(self._future)

    def finish(self, verification: DParaVerification, resolve: Callable):
        try:
            prepared = self._future.result()
        except BaseException:
            self.cancel()
            raise
        with self._lock:
            if self._cancelled or self._consumed:
                prepared.discard()
                raise ValueError("DPara round was cancelled or already consumed")
            if verification.binding != self.binding or prepared.binding != self.binding:
                prepared.discard()
                self._consumed = True
                raise ValueError("DPara request/revision/generation mismatch")
            self._consumed = True
        try:
            result = resolve(prepared, verification)
            # A cancellation during the head invalidates the prospective result
            # even though its local tensor work may have completed.
            if self._cancel_event.is_set():
                raise ValueError("DPara round was cancelled during finalization")
            return result
        except BaseException:
            prepared.discard()
            raise


def launch_dpara(binding: DParaBinding, precompute: Callable, *, executor: Executor):
    """Launch independent precompute; caller may verify on its own thread."""
    return DParaRound(binding, executor.submit(precompute))


def run_dpara_round(binding, precompute, verify, resolve, *, executor):
    """A complete prepare/verify/barrier/finalize round with failure cleanup."""
    pending = launch_dpara(binding, precompute, executor=executor)
    try:
        verified = verify()
        return pending.finish(verified, resolve)
    except BaseException:
        pending.cancel()
        raise
