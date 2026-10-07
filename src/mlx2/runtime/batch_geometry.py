"""Capability-aware geometry planning for continuous-batching rounds.

The planner is deliberately host-only.  It does not infer model families and
it never makes a route available: adapters declare the geometries they can
execute, while the scheduler supplies the live prefill/decode work.  A caller
may then apply the returned plan or fail closed if the execution seam changed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

Phase = Literal["prefill", "decode"]
Geometry = Literal[
    "rectangular", "bucketed_rectangular", "packed", "mixed_decode_first"
]


@dataclass(frozen=True)
class GeometryLane:
    uid: int
    phase: Phase
    tokens: int
    draft_tokens: int = 0

    def __post_init__(self) -> None:
        if self.phase not in ("prefill", "decode"):
            raise ValueError("lane phase must be prefill or decode")
        if isinstance(self.tokens, bool) or not isinstance(self.tokens, int):
            raise TypeError("lane tokens must be an integer")
        if self.tokens < 1:
            raise ValueError("lane tokens must be positive")
        if (
            isinstance(self.draft_tokens, bool)
            or not isinstance(self.draft_tokens, int)
            or self.draft_tokens < 0
        ):
            raise ValueError("draft_tokens must be a nonnegative integer")
        if self.phase == "prefill" and self.draft_tokens:
            raise ValueError("prefill lanes cannot carry draft tokens")


@dataclass(frozen=True)
class GeometryCapabilities:
    """Adapter/runtime capabilities, never model-name policy."""

    packed_prefill: bool = False
    packed_verify: bool = False
    mixed_forward: bool = False
    max_lanes: int = 1
    token_budget: int = 2048
    prefill_chunk: int = 512
    row_tile: int = 64

    def __post_init__(self) -> None:
        for name in ("max_lanes", "token_budget", "prefill_chunk", "row_tile"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class GeometryPolicy:
    """Selection thresholds; default-off at the serving-policy boundary."""

    packed_padding_fraction: float = 0.20
    bucket_padding_fraction: float = 0.12
    max_bucket_ratio: float = 1.50

    def __post_init__(self) -> None:
        for name in ("packed_padding_fraction", "bucket_padding_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not 0 <= float(value) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.max_bucket_ratio < 1:
            raise ValueError("max_bucket_ratio must be at least one")


@dataclass(frozen=True)
class GeometrySchedulerPolicy:
    """Default-off server policy for ordinary scheduler cohort shaping.

    Capabilities are intentionally absent.  The runtime derives them from the
    adapter/model execution seams, so a policy cannot manufacture a packed or
    mixed route.
    """

    enabled: bool = False
    token_budget: int | None = None
    selection: GeometryPolicy = GeometryPolicy()

    @classmethod
    def from_value(
        cls, value: Mapping[str, Any] | bool | None
    ) -> GeometrySchedulerPolicy:
        if value in (None, False):
            return cls()
        if value is True:
            return cls(enabled=True)
        if not isinstance(value, Mapping):
            raise TypeError("batch_geometry must be boolean or an object")
        # ``packed_padding_fraction`` is a planner (research replay) knob;
        # the scheduler never reads it, so the server policy refuses it.
        allowed = {
            "token_budget", "bucket_padding_fraction", "max_bucket_ratio",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown batch_geometry settings: {sorted(unknown)}")
        budget = value.get("token_budget")
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, int) or budget < 1
        ):
            raise ValueError("batch_geometry token_budget must be positive")
        return cls(
            enabled=True,
            token_budget=budget,
            selection=GeometryPolicy(
                bucket_padding_fraction=value.get("bucket_padding_fraction", .12),
                max_bucket_ratio=value.get("max_bucket_ratio", 1.50),
            ),
        )

    def as_dict(self) -> dict[str, int | float]:
        result: dict[str, int | float] = {
            "bucket_padding_fraction": self.selection.bucket_padding_fraction,
            "max_bucket_ratio": self.selection.max_bucket_ratio,
        }
        if self.token_budget is not None:
            result["token_budget"] = self.token_budget
        return result


@dataclass(frozen=True)
class GeometryPlan:
    geometry: Geometry
    lane_uids: tuple[int, ...]
    decode_uids: tuple[int, ...]
    prefill_uids: tuple[int, ...]
    prefill_rows: int
    decode_rows: int
    verification_rows: int
    real_rows: int
    charged_rows: int
    padding_rows: int
    padding_fraction: float
    token_budget: int
    reason: str
    implemented: bool = True
    qualified: bool = False
    selected_by_default: bool = False

    def receipt(self) -> dict:
        return {"schema": "mlx2.batch-geometry-plan.v1", **asdict(self)}


def _take_decode(lanes: Sequence[GeometryLane], caps: GeometryCapabilities):
    chosen, charged = [], 0
    for lane in lanes:
        if lane.phase != "decode" or len(chosen) >= caps.max_lanes:
            continue
        width = 1 + lane.draft_tokens
        if chosen and charged + width > caps.token_budget:
            break
        chosen.append(lane)
        charged += width
    return chosen, charged


def _take_prefill(
    lanes: Sequence[GeometryLane],
    caps: GeometryCapabilities,
    budget: int,
    *,
    max_lanes: int | None = None,
    allow_first_overflow: bool = True,
):
    lane_limit = caps.max_lanes if max_lanes is None else max_lanes
    chosen = []
    for lane in lanes:
        if lane.phase != "prefill" or len(chosen) >= lane_limit:
            continue
        width = min(lane.tokens, caps.prefill_chunk)
        # Rectangular execution charges every lane at the widest admitted
        # width.  A prefill-only round lets its first lane progress even when it
        # exceeds the soft token budget.  Once decode work owns part of the
        # round, prefill must fit both the remaining lane and token budgets.
        widths = [min(item.tokens, caps.prefill_chunk) for item in chosen] + [width]
        if max(widths) * len(widths) > budget and (
            chosen or not allow_first_overflow
        ):
            continue
        chosen.append(lane)
    return chosen


def _bucket(lanes: Sequence[GeometryLane], ratio: float) -> list[GeometryLane]:
    if not lanes:
        return []
    ordered = sorted(lanes, key=lambda lane: (lane.tokens, lane.uid))
    best: list[GeometryLane] = []
    for start in range(len(ordered)):
        group = [ordered[start]]
        floor = ordered[start].tokens
        for lane in ordered[start + 1 :]:
            if lane.tokens / floor > ratio:
                break
            group.append(lane)
        if len(group) > len(best):
            best = group
    return best


def trim_rectangular_cohort(
    candidate_lengths: Sequence[int],
    selected_indices: Sequence[int],
    *,
    active_lengths: Sequence[int] = (),
    token_budget: int,
) -> tuple[tuple[int, ...], dict[str, int]]:
    """Trim a padding-aware cohort to a rectangular scheduled-token budget.

    The queue head (the first selected index) is never removed, so progress is
    unconditional.  Existing active prompt rows are charged but cannot be
    evicted.  Among companion rows, each trim removes the row that minimizes
    the remaining padded charge; this is deterministic and host-only.
    """
    if isinstance(token_budget, bool) or not isinstance(token_budget, int):
        raise TypeError("token_budget must be an integer")
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    lengths = tuple(int(value) for value in candidate_lengths)
    active = tuple(int(value) for value in active_lengths if int(value) > 0)
    selected = list(dict.fromkeys(int(index) for index in selected_indices))
    if any(value < 0 for value in lengths) or any(value < 0 for value in active):
        raise ValueError("cohort lengths must be nonnegative")
    if any(not 0 <= index < len(lengths) for index in selected):
        raise ValueError("selected cohort index is outside candidate lengths")

    def accounting(indices):
        rows = active + tuple(lengths[index] for index in indices if lengths[index])
        real = sum(rows)
        padded = max(rows) * len(rows) if rows else 0
        return real, padded

    original = tuple(selected)
    _, original_padded = accounting(selected)
    while len(selected) > 1 and accounting(selected)[1] > token_budget:
        # Position zero is the oldest selected row and remains the progress
        # anchor.  Ties remove the latest companion, minimizing reordering.
        position = min(
            range(1, len(selected)),
            key=lambda pos: (
                accounting(selected[:pos] + selected[pos + 1 :])[1],
                -accounting(selected[:pos] + selected[pos + 1 :])[0],
                -pos,
            ),
        )
        selected.pop(position)
    real, padded = accounting(selected)
    return tuple(selected), {
        "candidate_rows": len(original),
        "selected_rows": len(selected),
        "deferred_rows": len(original) - len(selected),
        "real_rows": real,
        "charged_rows": padded,
        "padding_rows": max(0, padded - real),
        "original_charged_rows": original_padded,
    }


def plan_batch_geometry(
    lanes: Iterable[GeometryLane],
    capabilities: GeometryCapabilities,
    policy: GeometryPolicy | None = None,
) -> GeometryPlan:
    """Plan one decode-first round and account for speculative rows.

    ``draft_tokens`` is charged even when packed verification is unavailable;
    it is work the target must verify, not free output capacity.  Packed
    verification is therefore a capability gate rather than an accounting
    shortcut.
    """

    policy = policy or GeometryPolicy()
    lanes = tuple(lanes)
    if not lanes:
        raise ValueError("at least one lane is required")
    decode, decode_charge = _take_decode(lanes, capabilities)
    remaining_lanes = max(0, capabilities.max_lanes - len(decode))
    remaining_budget = capabilities.token_budget - decode_charge
    prefill = _take_prefill(
        lanes,
        capabilities,
        remaining_budget,
        max_lanes=remaining_lanes,
        allow_first_overflow=not decode,
    )

    if decode and prefill and capabilities.mixed_forward:
        geometry: Geometry = "mixed_decode_first"
        reason = "decode rows run first and share one bounded forward with prefill"
    elif decode:
        geometry = "packed" if capabilities.packed_verify else "rectangular"
        reason = (
            "packed speculative verification is adapter-declared"
            if capabilities.packed_verify
            else "decode and speculative verification use the ordinary reference"
        )
        prefill = []
    else:
        widths = [min(lane.tokens, capabilities.prefill_chunk) for lane in prefill]
        real = sum(widths)
        padded = max(widths) * len(widths)
        padding_fraction = 0.0 if not padded else (padded - real) / padded
        if (
            capabilities.packed_prefill
            and padding_fraction >= policy.packed_padding_fraction
        ):
            geometry = "packed"
            reason = "declared packed prefill removes material rectangular padding"
        elif padding_fraction >= policy.bucket_padding_fraction and len(prefill) > 1:
            bucketed = _bucket(prefill, policy.max_bucket_ratio)
            if bucketed and len(bucketed) < len(prefill):
                prefill = bucketed
                geometry = "bucketed_rectangular"
                reason = "length bucket reduces padding without a packed capability"
            else:
                geometry = "rectangular"
                reason = "current cohort is already a compact length bucket"
        else:
            geometry = "rectangular"
            reason = "padding is below the packed or bucketing threshold"

    prefill_widths = [
        min(lane.tokens, capabilities.prefill_chunk) for lane in prefill
    ]
    prefill_real = sum(prefill_widths)
    rectangular_prefill = (
        max(prefill_widths) * len(prefill_widths) if prefill_widths else 0
    )
    decode_rows = len(decode)
    verification_rows = sum(lane.draft_tokens for lane in decode)
    real_rows = prefill_real + decode_rows + verification_rows
    if geometry in ("packed", "mixed_decode_first"):
        charged_rows = real_rows
    else:
        charged_rows = rectangular_prefill + decode_rows + verification_rows
    padding_rows = max(0, charged_rows - real_rows)
    # Report the padding that the selected packed plan avoided, too.  It is
    # the decision signal an operator needs even though it is not charged.
    decision_padding = max(0, rectangular_prefill - prefill_real)
    padding_fraction = (
        0.0
        if rectangular_prefill == 0
        else decision_padding / rectangular_prefill
    )
    selected = tuple(decode) + tuple(prefill)
    return GeometryPlan(
        geometry=geometry,
        lane_uids=tuple(lane.uid for lane in selected),
        decode_uids=tuple(lane.uid for lane in decode),
        prefill_uids=tuple(lane.uid for lane in prefill),
        prefill_rows=prefill_real,
        decode_rows=decode_rows,
        verification_rows=verification_rows,
        real_rows=real_rows,
        charged_rows=charged_rows,
        padding_rows=(decision_padding if geometry == "packed" else padding_rows),
        padding_fraction=padding_fraction,
        token_budget=capabilities.token_budget,
        reason=reason,
    )


__all__ = [
    "GeometryCapabilities",
    "GeometryLane",
    "GeometryPlan",
    "GeometryPolicy",
    "GeometrySchedulerPolicy",
    "plan_batch_geometry",
    "trim_rectangular_cohort",
]
