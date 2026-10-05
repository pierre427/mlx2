"""Default-off BatchGenerator lifecycle bridge for a paged host probe.

The bridge binds a candidate to a live request UID and its cancellation/removal
boundary. It deliberately cannot select a serving route: BatchGenerator still
owns the ordinary model cache and sampler, while the paged token-page owner has
no atomic handoff into either. A host probe may exercise pricing and the
request-private row transaction, but its publication is not model KV use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

from .generate import BatchGenerator
from .paged_atomic_owner import PagedAtomicRequestOwner, PrivateCandidate
from .paged_pack_price import MeasuredPackPrice
from .paged_pack_scheduler import PrefillOffer, ReservedRows
from .paged_request_route import (
    PagedRouteReceipt,
    RouteCapability,
    coordinate_paged_request,
)
from .paged_request_transaction import CandidateRequest


@dataclass(frozen=True)
class PagedBatchProbeReceipt:
    uid: int
    reason: str
    generator_stage: str | None
    serving_selected: bool
    serving_observed_used: bool
    host_probe: PagedRouteReceipt | None


_STAGES = ("queued", "prefill", "decode", "plain_fallback")


def probe_paged_batch_request(
    generator: BatchGenerator,
    lifecycle_lock: RLock,
    request: CandidateRequest,
    owner: PagedAtomicRequestOwner,
    capability: RouteCapability,
    reserved: tuple[ReservedRows, ...],
    offers: tuple[PrefillOffer, ...],
    *,
    price: MeasuredPackPrice | None,
    live_identity: dict[str, str],
    context_tokens: tuple[int, ...],
    row_capacity: int,
    free_pages: int,
    run_host_candidate: Callable[[PrivateCandidate], int],
    permit_host_probe: bool = False,
    cancelled: Callable[[], bool] = lambda: False,
) -> PagedBatchProbeReceipt:
    """Run a private host transaction for one live BatchGenerator UID.

    The caller must use ``lifecycle_lock`` for insert, next, and remove too.
    Neither a successful probe nor its row publication changes the generator.
    """
    if type(generator) is not BatchGenerator:
        raise TypeError("a BatchGenerator is required")
    if type(request) is not CandidateRequest:
        raise TypeError("a CandidateRequest is required")
    if not hasattr(lifecycle_lock, "__enter__") or not hasattr(lifecycle_lock, "__exit__"):
        raise TypeError("a lifecycle lock is required")

    def receipt(reason: str, stage: str | None = None,
                host: PagedRouteReceipt | None = None) -> PagedBatchProbeReceipt:
        return PagedBatchProbeReceipt(request.lane_id, reason, stage, False, False, host)

    if not permit_host_probe:
        return receipt("paged_host_probe_disabled")
    with lifecycle_lock:
        found = generator._find_uids((request.lane_id,)).get(request.lane_id)
        if found is None:
            return receipt("request_not_live")
        stage = _STAGES[found[0]]
        if cancelled():
            return receipt("cancelled", stage)
        if stage != "queued":
            return receipt("generator_stage_not_supported", stage)
        sequence = generator._unprocessed_sequences[found[1]]
        queued_rows = sum(len(part) for part in sequence[1])
        actual_context = len(sequence[4]) + queued_rows
        if context_tokens != (actual_context,):
            return receipt("live_context_mismatch", stage)
        if (reserved or len(offers) != 1 or offers[0].lane_id != request.lane_id or
                not any(option.rows == request.proposed_rows for option in offers[0].options) or
                request.proposed_rows > queued_rows):
            return receipt("queued_prefill_shape_mismatch", stage)
        # No native page table or sampler is published into BatchGenerator.
        # The coordinator's "selected" and "observed_used" concern only the
        # private host owner and must never become a serving route receipt.
        host = coordinate_paged_request(
            request, owner, capability, reserved, offers, price=price,
            live_identity=live_identity, context_tokens=context_tokens,
            row_capacity=row_capacity, free_pages=free_pages,
            run_candidate=run_host_candidate, permit_candidate=True,
            cancelled=lambda: cancelled() or
                request.lane_id not in generator._find_uids((request.lane_id,)),
        )
        return receipt(host.reason, stage, host)


__all__ = ["PagedBatchProbeReceipt", "probe_paged_batch_request"]
