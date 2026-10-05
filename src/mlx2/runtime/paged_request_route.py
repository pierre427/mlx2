"""Default-off host coordinator for a priced, atomic paged request candidate.

This seam never calls the ordinary model path. A refused receipt leaves it
available for the caller; no native/model callback runs until every gate passes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .paged_atomic_owner import PagedAtomicRequestOwner, PrivateCandidate
from .paged_pack_scheduler import (
    PagedPackDecision, PrefillOffer, PrefillOption, ReservedRows, choose_paged_pack,
)
from .paged_pack_price import (
    MeasuredPackPrice, ResearchCalibratedPrice, ResearchWarmCalibratedPrice,
)
from .paged_request_transaction import CandidateRequest, CandidateReceipt, execute_paged_request


@dataclass(frozen=True)
class RouteCapability:
    """Caller-attested route coverage; qualification remains informational."""

    implemented: bool
    qualified: bool
    required_planes: tuple[str, ...]
    profile_id: str


@dataclass(frozen=True)
class PagedRouteReceipt:
    lane_id: int
    revision: str
    implemented: bool
    qualified: bool
    selected: bool
    observed_used: bool
    published: bool
    reason: str
    profile_id: str | None
    accepted_rows: int
    decision: PagedPackDecision | None
    candidate_receipt: CandidateReceipt | None


def coordinate_paged_request(
    request: CandidateRequest,
    owner: PagedAtomicRequestOwner,
    capability: RouteCapability,
    reserved: tuple[ReservedRows, ...],
    offers: tuple[PrefillOffer, ...],
    *,
    price: MeasuredPackPrice | ResearchCalibratedPrice | ResearchWarmCalibratedPrice | None,
    live_identity: dict[str, str],
    context_tokens: tuple[int, ...],
    row_capacity: int,
    free_pages: int,
    run_candidate: Callable[[PrivateCandidate], int],
    permit_candidate: bool = False,
    permit_research_calibration: bool = False,
    cancelled: Callable[[], bool] = lambda: False,
) -> PagedRouteReceipt:
    """Gate a candidate, price/select the pack, then publish one request boundary.

    A refusal returns without invoking ``run_candidate``. Exceptions from the
    candidate propagate after the transaction rolls back private state.
    """
    def receipt(reason: str, *, decision: PagedPackDecision | None = None,
                candidate: CandidateReceipt | None = None,
                selected: bool = False) -> PagedRouteReceipt:
        return PagedRouteReceipt(
            request.lane_id, request.revision, capability.implemented is True,
            capability.qualified is True, selected,
            candidate.published if candidate else False,
            candidate.published if candidate else False,
            reason, decision.profile_id if decision else None,
            candidate.accepted_rows if candidate else 0, decision, candidate,
        )

    if not permit_candidate:
        return receipt("paged_route_disabled")
    if capability.implemented is not True:
        return receipt("route_not_implemented")
    if (not capability.required_planes or
            request.planes != capability.required_planes or
            any(plane not in owner.supported_planes for plane in request.planes) or
            owner.atomic_publish is not True):
        return receipt("atomic_state_capability_missing")
    research_price = type(price) is ResearchCalibratedPrice
    if research_price and (not permit_research_calibration or
                           request.proposed_rows != 63 or reserved or
                           len(offers) != 1 or offers[0].lane_id != request.lane_id or
                           offers[0].options != (PrefillOption(63, 0),) or
                           context_tokens != (63,)):
        return receipt("research_calibration_shape_refused")
    warm_research_price = type(price) is ResearchWarmCalibratedPrice
    if warm_research_price and (not permit_research_calibration or
                                request.proposed_rows != 1 or reserved or
                                len(offers) != 1 or offers[0].lane_id != request.lane_id or
                                offers[0].options != (PrefillOption(1, 0),) or
                                context_tokens != (64,)):
        return receipt("warm_research_calibration_shape_refused")
    if (type(price) not in (MeasuredPackPrice, ResearchCalibratedPrice,
                            ResearchWarmCalibratedPrice) or
            not price.validated or
            dict(price.identity) != live_identity or
            price.context_tokens != context_tokens or
            not isinstance(capability.profile_id, str) or
            not capability.profile_id or
            price.profile_id != capability.profile_id):
        return receipt("source_bound_price_missing")
    snapshot = owner.snapshot()
    try:
        current_revision = snapshot.revision
    finally:
        close_snapshot = getattr(snapshot, "close", None)
        if close_snapshot is not None:
            close_snapshot()
    if current_revision != request.revision:
        return receipt("request_revision_drifted")
    if cancelled():
        return receipt("cancelled")
    try:
        decision = choose_paged_pack(reserved, offers, price=price,
                                     row_capacity=row_capacity, free_pages=free_pages,
                                     permit_candidate=True)
    except (ValueError, TypeError, AttributeError):
        return receipt("pack_price_or_input_invalid")
    if not decision.accepted:
        return receipt(decision.reason, decision=decision)
    selected = {row.lane_id: row.rows for row in decision.reserved}
    if decision.prefill_lane_id is not None:
        selected[decision.prefill_lane_id] = decision.prefill_rows
    if selected.get(request.lane_id) != request.proposed_rows:
        return receipt("request_not_selected", decision=decision)
    if cancelled():
        return receipt("cancelled", decision=decision, selected=True)
    candidate = execute_paged_request(decision, request, owner, run_candidate,
                                      permit_candidate=True, cancelled=cancelled)
    return receipt(candidate.reason, decision=decision, candidate=candidate,
                   selected=True)


__all__ = ["PagedRouteReceipt", "RouteCapability", "coordinate_paged_request"]
