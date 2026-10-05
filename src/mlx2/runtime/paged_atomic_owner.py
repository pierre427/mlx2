"""Default-off host owner for an atomic paged request state boundary.

The owner stores request state as one private snapshot. Serving adapters must
materialize their tensor state into request-private rows before calling stage;
this host implementation does not bind a model, cache, or scheduler.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from typing import Any, Iterable

from .paged_request_transaction import CandidateRequest, STATE_PLANES


class AtomicOwnerError(RuntimeError):
    """The atomic request boundary cannot be safely published."""


@dataclass(frozen=True)
class RequestSnapshot:
    revision: str
    generation: int
    planes: tuple[tuple[str, tuple[Any, ...]], ...]

    def rows(self, plane: str) -> tuple[Any, ...]:
        return dict(self.planes)[plane]


class PagedAtomicRequestOwner:
    """One request's revision-bound state, published by one pointer swap.

    ``snapshot`` returns a deep copy so callers cannot mutate public rows.
    All writes, including reference-path updates, must go through this owner.
    The experimental candidate capability is disabled unless explicitly set.
    """

    def __init__(self, revision: str, initial: dict[str, Iterable[Any]], *,
                 supported_planes: tuple[str, ...] = STATE_PLANES,
                 enabled: bool = False) -> None:
        if not isinstance(revision, str) or not revision:
            raise ValueError("revision is required")
        if (type(supported_planes) is not tuple or not supported_planes or
                tuple(p for p in STATE_PLANES if p in supported_planes) != supported_planes or
                "kv" not in supported_planes):
            raise ValueError("supported planes must be ordered, unique and include kv")
        if set(initial) != set(supported_planes):
            raise ValueError("initial state must cover supported planes exactly")
        self._lock = threading.RLock()
        self.supported_planes = supported_planes
        self._enabled = enabled is True
        self._public = RequestSnapshot(revision, 0, self._copy_planes(initial))

    @property
    def atomic_publish(self) -> bool:
        return self._enabled

    def snapshot(self) -> RequestSnapshot:
        with self._lock:
            return copy.deepcopy(self._public)

    def _copy_planes(self, source: dict[str, Iterable[Any]]) -> tuple[tuple[str, tuple[Any, ...]], ...]:
        return tuple((plane, tuple(copy.deepcopy(tuple(source[plane]))))
                     for plane in self.supported_planes)

    def replace_reference(self, revision: str, state: dict[str, Iterable[Any]]) -> RequestSnapshot:
        """Advance the ordinary reference path under the same request lock."""
        if not isinstance(revision, str) or not revision:
            raise ValueError("revision is required")
        if set(state) != set(self.supported_planes):
            raise ValueError("reference state must cover supported planes exactly")
        planes = self._copy_planes(state)
        with self._lock:
            self._public = RequestSnapshot(revision, self._public.generation + 1, planes)
            return copy.deepcopy(self._public)

    def begin(self, request: CandidateRequest) -> "PrivateCandidate":
        if type(request) is not CandidateRequest:
            raise TypeError("request must be CandidateRequest")
        with self._lock:
            if not self._enabled:
                raise AtomicOwnerError("paged atomic owner is disabled")
            if any(p not in self.supported_planes for p in request.planes):
                raise AtomicOwnerError("requested state plane is unsupported")
            if request.revision != self._public.revision:
                raise AtomicOwnerError("request revision differs from public state")
            return PrivateCandidate(self, request, self._public)


class PrivateCandidate:
    def __init__(self, owner: PagedAtomicRequestOwner, request: CandidateRequest,
                 origin: RequestSnapshot) -> None:
        self._owner = owner
        self._request = request
        self._origin = origin
        self._staged: dict[str, tuple[Any, ...]] = {}
        self._closed = False

    def stage(self, plane: str, rows: Iterable[Any]) -> None:
        """Stage exactly one proposed row per token for a requested plane."""
        if self._closed:
            raise AtomicOwnerError("candidate is closed")
        if plane not in self._request.planes or plane in self._staged:
            raise AtomicOwnerError("plane is unrequested or already staged")
        copied = tuple(copy.deepcopy(tuple(rows)))
        if len(copied) != self._request.proposed_rows:
            raise AtomicOwnerError("staged rows must cover the complete proposal")
        self._staged[plane] = copied

    def prepare(self, accepted_rows: int) -> "PreparedCandidate":
        if self._closed:
            raise AtomicOwnerError("candidate is closed")
        if type(accepted_rows) is not int or not 0 <= accepted_rows <= self._request.proposed_rows:
            raise ValueError("accepted rows must be an integer prefix within the proposal")
        if set(self._staged) != set(self._request.planes):
            raise AtomicOwnerError("all requested state planes must be staged")
        next_planes = dict(self._origin.planes)
        for plane in self._request.planes:
            next_planes[plane] = next_planes[plane] + self._staged[plane][:accepted_rows]
        successor = RequestSnapshot(self._request.revision,
                                    self._origin.generation + 1,
                                    tuple((p, next_planes[p]) for p in self._owner.supported_planes))
        self._closed = True
        self._staged.clear()
        return PreparedCandidate(self._owner, self._request, self._origin, successor)

    def rollback(self) -> None:
        self._staged.clear()
        self._closed = True


class PreparedCandidate:
    def __init__(self, owner: PagedAtomicRequestOwner, request: CandidateRequest,
                 origin: RequestSnapshot, successor: RequestSnapshot) -> None:
        self._owner = owner
        self.planes = request.planes
        self.revision = request.revision
        self._origin = origin
        self._successor: RequestSnapshot | None = successor

    def publish(self) -> None:
        with self._owner._lock:
            if self._successor is None:
                raise AtomicOwnerError("prepared boundary is closed")
            if self._owner._public is not self._origin:
                raise AtomicOwnerError("public revision or generation drifted")
            # No fallible operation follows the sole public pointer assignment.
            self._owner._public = self._successor
            self._successor = None

    def rollback(self) -> None:
        self._successor = None


__all__ = ["AtomicOwnerError", "PagedAtomicRequestOwner", "RequestSnapshot"]
