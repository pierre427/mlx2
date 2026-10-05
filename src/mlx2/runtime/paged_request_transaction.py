"""Default-off, host-side boundary for one paged request candidate.

This module neither executes a model nor owns a cache. A serving adapter must
provide one revision-bound owner that stages every requested state plane and
publishes the accepted prefix in one atomic operation. An owner that cannot
provide that contract is refused before a candidate runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .paged_pack_scheduler import PagedPackDecision

STATE_PLANES = ("kv", "gdn", "qsa", "mtp")


@dataclass(frozen=True)
class CandidateRequest:
    lane_id: int
    revision: str
    proposed_rows: int
    planes: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.lane_id) is not int or self.lane_id < 0:
            raise ValueError("lane_id must be a nonnegative integer")
        if not isinstance(self.revision, str) or not self.revision:
            raise ValueError("revision is required")
        if type(self.proposed_rows) is not int or self.proposed_rows < 1:
            raise ValueError("proposed_rows must be a positive integer")
        if (type(self.planes) is not tuple or not self.planes or
                "kv" not in self.planes or
                tuple(plane for plane in STATE_PLANES if plane in self.planes) != self.planes):
            raise ValueError("planes must be ordered, unique known planes including kv")


class PreparedBoundary(Protocol):
    """A complete, unpublished accepted prefix across all requested planes."""

    planes: tuple[str, ...]
    revision: str

    def publish(self) -> None:
        """Publish all planes under one request lock/pointer swap.

        If this raises, no public plane may have advanced. Readers must not
        observe an intermediate plane or a partially accepted token.
        """

    def rollback(self) -> None:
        """Discard unpublished state; safe after a failed publish."""


class CandidateBoundary(Protocol):
    """Private speculative state. No public state changes before publish."""

    def prepare(self, accepted_rows: int) -> PreparedBoundary: ...

    def rollback(self) -> None: ...


class AtomicRequestOwner(Protocol):
    """Adapter capability; begin must fail without leaving a partial branch."""

    supported_planes: tuple[str, ...]
    atomic_publish: bool

    def begin(self, request: CandidateRequest) -> CandidateBoundary: ...


@dataclass(frozen=True)
class CandidateReceipt:
    lane_id: int
    revision: str
    profile_id: str | None
    proposed_rows: int
    accepted_rows: int
    planes: tuple[str, ...]
    published: bool
    reason: str


def _receipt(request: CandidateRequest, decision: PagedPackDecision,
             reason: str, *, accepted: int = 0,
             published: bool = False) -> CandidateReceipt:
    return CandidateReceipt(request.lane_id, request.revision, decision.profile_id,
                            request.proposed_rows, accepted, request.planes,
                            published, reason)


def execute_paged_request(
    decision: PagedPackDecision,
    request: CandidateRequest,
    owner: AtomicRequestOwner,
    run_candidate: Callable[[CandidateBoundary], int],
    *,
    permit_candidate: bool = False,
    cancelled: Callable[[], bool] = lambda: False,
) -> CandidateReceipt:
    """Run one selected request, then publish only its accepted prefix.

    The pack decision reserves decode/verify before any optional prefill. The
    caller runs this per lane under its normal scheduling/lifecycle lock. A
    speculative verifier may return zero, a partial prefix, or every row.
    Exceptions propagate after private state is discarded. This function
    cannot supply atomicity for an owner with separate public plane commits.
    """
    if type(decision) is not PagedPackDecision:
        raise TypeError("decision must be a PagedPackDecision")
    if not permit_candidate:
        return _receipt(request, decision, "paged_request_disabled")
    if not decision.accepted:
        return _receipt(request, decision, decision.reason)
    selected = {row.lane_id: row.rows for row in decision.reserved}
    if decision.prefill_lane_id is not None:
        selected[decision.prefill_lane_id] = decision.prefill_rows
    if selected.get(request.lane_id) != request.proposed_rows:
        return _receipt(request, decision, "request_not_selected")
    if (owner.atomic_publish is not True or
            any(plane not in owner.supported_planes for plane in request.planes)):
        return _receipt(request, decision, "atomic_state_capability_missing")
    if cancelled():
        return _receipt(request, decision, "cancelled")

    candidate = owner.begin(request)
    prepared: PreparedBoundary | None = None
    rolled_back = False
    try:
        accepted = run_candidate(candidate)
        if type(accepted) is not int or not 0 <= accepted <= request.proposed_rows:
            raise ValueError("accepted rows must be an integer prefix within the proposal")
        if cancelled():
            rolled_back = True
            candidate.rollback()
            return _receipt(request, decision, "cancelled")
        prepared = candidate.prepare(accepted)
        if prepared.planes != request.planes or prepared.revision != request.revision:
            raise ValueError("prepared boundary changed revision or state planes")
        if cancelled():
            rolled_back = True
            prepared.rollback()
            return _receipt(request, decision, "cancelled")
        prepared.publish()
        return _receipt(request, decision, "accepted", accepted=accepted,
                        published=True)
    except BaseException:
        if not rolled_back:
            if prepared is not None:
                prepared.rollback()
            else:
                candidate.rollback()
        raise


__all__ = [
    "STATE_PLANES",
    "AtomicRequestOwner",
    "CandidateBoundary",
    "CandidateReceipt",
    "CandidateRequest",
    "PreparedBoundary",
    "execute_paged_request",
]
