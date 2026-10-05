"""Default-off, host-only admission for a mixed paged forward.

The planner reserves every ready decode/verify row first.  It then considers
bounded prefill slices using a caller-supplied, source-bound measured cost
profile.  It does not mutate a queue, cache, or model; those transactions are
owned by the serving runtime after a plan is accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


def _positive(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative(name: str, value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class ReservedRows:
    lane_id: int
    phase: str
    rows: int
    new_pages: int
    deadline_ms: float

    def __post_init__(self) -> None:
        _nonnegative("lane_id", self.lane_id)
        if self.phase not in ("decode", "verify"):
            raise ValueError("reserved phase must be decode or verify")
        _positive("rows", self.rows)
        _nonnegative("new_pages", self.new_pages)
        if not math.isfinite(self.deadline_ms) or self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be finite and positive")


@dataclass(frozen=True)
class PrefillOption:
    rows: int
    new_pages: int

    def __post_init__(self) -> None:
        _positive("rows", self.rows)
        _nonnegative("new_pages", self.new_pages)


@dataclass(frozen=True)
class PrefillOffer:
    lane_id: int
    options: tuple[PrefillOption, ...]

    def __post_init__(self) -> None:
        _nonnegative("lane_id", self.lane_id)
        if type(self.options) is not tuple or not self.options:
            raise ValueError("prefill options must be a nonempty tuple")
        if any(type(option) is not PrefillOption for option in self.options):
            raise TypeError("prefill options must be exact PrefillOption values")
        sizes = [option.rows for option in self.options]
        if sizes != sorted(set(sizes)):
            raise ValueError("prefill option rows must increase without duplicates")


class PackPrice(Protocol):
    """A measured profile pinned to source, artifact, hardware and kernels."""

    profile_id: str

    def estimate_ms(self, reserved: tuple[ReservedRows, ...],
                    prefill: tuple[int, PrefillOption] | None) -> float: ...


@dataclass(frozen=True)
class PagedPackDecision:
    accepted: bool
    reason: str
    reserved: tuple[ReservedRows, ...]
    prefill_lane_id: int | None
    prefill_rows: int
    estimated_ms: float | None
    profile_id: str | None


def choose_paged_pack(
    reserved: tuple[ReservedRows, ...],
    offers: tuple[PrefillOffer, ...],
    *,
    price: PackPrice | None,
    row_capacity: int,
    free_pages: int,
    permit_candidate: bool = False,
    permit_research_graph: bool = False,
) -> PagedPackDecision:
    """Choose at most one prefill slice after mandatory decoder reservation.

    Offers are passed in the caller's ageing/fairness order. The oldest
    feasible offer wins; within that offer the largest feasible slice wins.
    The common forward must fit every ready decoder's deadline.
    An unpriced pack is refused rather than guessed from token count.
    """
    _positive("row_capacity", row_capacity)
    _nonnegative("free_pages", free_pages)
    if type(reserved) is not tuple or type(offers) is not tuple:
        raise TypeError("reserved and offers must be tuples")
    if any(type(item) is not ReservedRows for item in reserved):
        raise TypeError("reserved rows must use ReservedRows")
    if any(type(item) is not PrefillOffer for item in offers):
        raise TypeError("prefill offers must use PrefillOffer")
    ids = [item.lane_id for item in reserved] + [item.lane_id for item in offers]
    if len(ids) != len(set(ids)):
        raise ValueError("each lane must appear exactly once")
    mandatory_rows = sum(item.rows for item in reserved)
    mandatory_pages = sum(item.new_pages for item in reserved)

    def reject(reason: str, *, estimate: float | None = None) -> PagedPackDecision:
        return PagedPackDecision(False, reason, reserved, None, 0, estimate,
                                 getattr(price, "profile_id", None))

    if not permit_candidate:
        return reject("paged_pack_disabled")
    if mandatory_rows > row_capacity or mandatory_pages > free_pages:
        return reject("mandatory_decode_capacity")
    if price is None or not isinstance(getattr(price, "profile_id", None), str) or not price.profile_id:
        return reject("source_bound_price_missing")
    from .paged_pack_price import ResearchGraphB2Price
    if type(price) is ResearchGraphB2Price and (
            not permit_research_graph or not price.validated):
        return reject("research_graph_price_disabled")

    def estimate(prefill: tuple[int, PrefillOption] | None) -> float:
        value = price.estimate_ms(reserved, prefill)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError("pack cost estimate must be finite and nonnegative")
        return float(value)

    base_ms = estimate(None)
    deadline = min((item.deadline_ms for item in reserved), default=math.inf)
    if base_ms > deadline:
        return reject("mandatory_decode_deadline", estimate=base_ms)
    best: tuple[int, int, PrefillOption, float] | None = None
    for priority, offer in enumerate(offers):
        for option in offer.options:
            if mandatory_rows + option.rows > row_capacity or mandatory_pages + option.new_pages > free_pages:
                continue
            cost = estimate((offer.lane_id, option))
            if cost > deadline:
                continue
            if (best is None or priority < best[0] or
                    (priority == best[0] and
                     (option.rows > best[2].rows or
                      (option.rows == best[2].rows and cost < best[3])))):
                best = (priority, offer.lane_id, option, cost)
    if best is None:
        return PagedPackDecision(True, "decode_only", reserved, None, 0, base_ms, price.profile_id)
    _, lane_id, option, cost = best
    return PagedPackDecision(True, "mixed_pack", reserved, lane_id, option.rows, cost, price.profile_id)


__all__ = [
    "PackPrice",
    "PagedPackDecision",
    "PrefillOffer",
    "PrefillOption",
    "ReservedRows",
    "choose_paged_pack",
]
