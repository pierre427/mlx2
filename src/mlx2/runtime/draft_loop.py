"""Confidence-gated self-MTP draft looping.

A lane drafts in stages.  At the end of each stage except the last, it sums
the draft head's log-probability of the tokens that stage drafted.  At or
above ``threshold`` it drafts the next stage; below it, it stops, and only
what it drafted is verified.  One target forward still verifies each lane's
whole draft and the acceptance rule is unchanged, so outputs keep the
target's distribution exactly.  The gate only decides how many rows reach
the verify forward.

Stages are uniform (``stage``: every ``stage`` drafts, up to the lane's
``num_draft``), explicit (``boundaries``: increasing draft depths such as
``[3, 7]``), or explicit per cohort width (``by_width``: ``{"1": [3, 9],
"2": [3, 7]}``).  Explicit boundaries own the depth ceiling: a lane whose
``num_draft`` reaches the first boundary may draft up to the last one, so the
configured depth (and everything sized from it) stays the base stage.  A
lane below the first boundary, for example one an adaptive-depth controller
has shortened, is not gated.  ``max_width`` limits a single topology to
cohorts no wider than the one it was probed at; ``by_width`` names one
topology per probed width and leaves every other width at fixed depth.

``cohort`` says who decides in a cohort of several lanes.  Self-MTP verifies
a cohort with padded rows (every lane gets the deepest lane's row count), so
one lane's extension is paid by all.  ``lane`` lets each lane decide alone;
``any`` extends every lane at a decision point as soon as one passes, so the
padded rows carry real drafts instead of padding.

The rule is DLoop's (arXiv 2610.07659, "confidence-gated drafting loop"):
a non-positive score cannot be offset by a confident position, so one
uncertain token stops the loop.  The paper pairs it with retraining for
parallel drafters; an autoregressive MTP head already conditions on its own
hidden states, so this path needs no new weights.  Original mlx2 code; see
``provenance/dloop-untrained-gate.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

COHORT_RULES = ("lane", "any")


def _valid_boundaries(values) -> bool:
    return (
        isinstance(values, tuple)
        and len(values) >= 2
        and not any(isinstance(v, bool) or not isinstance(v, int) for v in values)
        and values[0] >= 1
        and all(b > a for a, b in zip(values, values[1:]))
    )


@dataclass(frozen=True)
class DraftLoopPolicy:
    threshold: float
    stage: Optional[int] = None
    boundaries: Optional[Tuple[int, ...]] = None
    max_width: Optional[int] = None
    by_width: Optional[Tuple[Tuple[int, Tuple[int, ...]], ...]] = None
    cohort: str = "lane"

    def __post_init__(self) -> None:
        if sum(x is not None for x in (self.stage, self.boundaries, self.by_width)) != 1:
            raise ValueError("draft_loop needs exactly one of stage, boundaries or by_width")
        if self.stage is not None and (
            isinstance(self.stage, bool) or not isinstance(self.stage, int) or self.stage < 1
        ):
            raise ValueError("draft_loop stage must be a positive integer")
        if self.boundaries is not None and not _valid_boundaries(self.boundaries):
            raise ValueError(
                "draft_loop boundaries must be two or more increasing positive depths"
            )
        if self.by_width is not None:
            widths = [w for w, _ in self.by_width]
            if (
                not self.by_width
                or any(isinstance(w, bool) or not isinstance(w, int) or w < 1 for w in widths)
                or len(set(widths)) != len(widths)
                or not all(_valid_boundaries(b) for _, b in self.by_width)
            ):
                raise ValueError(
                    "draft_loop by_width maps distinct positive widths to boundaries"
                )
            if self.max_width is not None:
                raise ValueError("draft_loop by_width already names its widths; drop max_width")
        if self.max_width is not None and (
            isinstance(self.max_width, bool)
            or not isinstance(self.max_width, int)
            or self.max_width < 1
        ):
            raise ValueError("draft_loop max_width must be a positive integer")
        if self.cohort not in COHORT_RULES:
            raise ValueError(f"draft_loop cohort must be one of {COHORT_RULES}")
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(float(self.threshold))
            or float(self.threshold) > 0.0
        ):
            raise ValueError("draft_loop threshold must be a finite log-probability <= 0")

    @classmethod
    def from_value(cls, value: Any) -> Optional["DraftLoopPolicy"]:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        keys = set(value) - {"max_width", "cohort"} if isinstance(value, Mapping) else None
        if keys not in (
            {"stage", "threshold"}, {"boundaries", "threshold"}, {"by_width", "threshold"}
        ):
            raise ValueError(
                "draft_loop must be {stage, threshold}, {boundaries, threshold} or"
                " {by_width, threshold}, with an optional max_width or cohort"
            )
        threshold = value["threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("draft_loop threshold must be a finite log-probability <= 0")
        boundaries = value.get("boundaries")
        if boundaries is not None:
            if not isinstance(boundaries, (list, tuple)):
                raise ValueError("draft_loop boundaries must be a list of depths")
            boundaries = tuple(boundaries)
        by_width = value.get("by_width")
        if by_width is not None:
            if not isinstance(by_width, Mapping):
                raise ValueError("draft_loop by_width must map widths to boundaries")
            items = []
            for width, ends in by_width.items():
                try:
                    width = int(width)
                except (TypeError, ValueError):
                    raise ValueError("draft_loop by_width keys must be widths") from None
                if not isinstance(ends, (list, tuple)):
                    raise ValueError("draft_loop by_width values must be lists of depths")
                items.append((width, tuple(ends)))
            by_width = tuple(sorted(items))
        return cls(
            threshold=float(threshold),
            stage=value.get("stage"),
            boundaries=boundaries,
            max_width=value.get("max_width"),
            by_width=by_width,
            cohort=value.get("cohort", "lane"),
        )

    def for_width(self, width: int) -> Optional["DraftLoopPolicy"]:
        """The single-topology policy a cohort of ``width`` lanes runs, or None."""
        width = int(width)
        if self.by_width is not None:
            ends = dict(self.by_width).get(width)
            if ends is None:
                return None
            return DraftLoopPolicy(threshold=self.threshold, boundaries=ends, cohort=self.cohort)
        if self.max_width is not None and width > self.max_width:
            return None
        return self

    def _single(self) -> None:
        if self.by_width is not None:
            raise ValueError("resolve a by_width draft loop with for_width() first")

    def limit(self, num_draft: int) -> int:
        """Most drafts a lane configured at ``num_draft`` may draft."""
        num_draft = int(num_draft)
        if self.by_width is not None:
            # Admission sizes every lane for the deepest width's ceiling.
            reach = [b[-1] for _, b in self.by_width if num_draft >= b[0]]
            return max([num_draft, *reach])
        if self.boundaries is not None and num_draft >= self.boundaries[0]:
            return max(num_draft, self.boundaries[-1])
        return num_draft

    def ends(self, limit: int) -> Tuple[int, ...]:
        """Stage-end depths for a lane allowed ``limit`` drafts this round."""
        self._single()
        limit = int(limit)
        if self.boundaries is not None:
            ends = [b for b in self.boundaries if b < limit]
        else:
            ends = list(range(self.stage, limit, self.stage))
        return tuple(ends) + (limit,)

    def applies(self, num_draft: int) -> bool:
        """True when a lane configured at ``num_draft`` has a decision to make."""
        self._single()
        num_draft = int(num_draft)
        if self.boundaries is not None and num_draft < self.boundaries[0]:
            return False
        return len(self.ends(self.limit(num_draft))) > 1

    def gates(self, num_draft: int) -> bool:
        return self.applies(num_draft)

    def decision(self, depth: int, num_draft: int) -> Optional[int]:
        """At ``depth`` drafts, the first depth of the stage to score, or None.

        None means ``depth`` is not a decision point for this lane: either it
        is inside a stage, or it is the lane's last allowed depth.
        """
        ends = self.ends(num_draft)
        if depth not in ends[:-1]:
            return None
        index = ends.index(depth)
        return ends[index - 1] if index else 0

    def receipt(self) -> dict:
        return {
            "implemented": True,
            "qualified": False,
            "selected": True,
            "stage": self.stage,
            "boundaries": list(self.boundaries) if self.boundaries is not None else None,
            "by_width": (
                {str(w): list(b) for w, b in self.by_width} if self.by_width is not None else None
            ),
            "max_width": self.max_width,
            "cohort": self.cohort,
            "threshold": float(self.threshold),
            "rule": "sum_draft_logprob_per_stage_v1",
        }


def cohort_draft_depth(config: Optional[Mapping[str, Any]], *, lanes: int = 1) -> int:
    """Most drafts one lane may carry in a cohort of ``lanes`` under ``config``.

    The base is ``num_draft``; a ``draft_loop`` raises it to the loop's
    ceiling (``DraftLoopPolicy.limit``) for cohorts no wider than its
    ``max_width``.  This is the depth the verify forward can really see, so
    every bound sized from the configured depth (int8 row bound, admission
    row cap, verify-transient calibration, exact-row receipts) reads it.
    """
    config = config or {}
    base = int(config.get("num_draft") or 0)
    loop = DraftLoopPolicy.from_value(config.get("draft_loop"))
    if loop is None:
        return base
    # ``for_width`` resolves a ``by_width`` policy to the topology this
    # cohort width runs (None: fixed depth) and applies ``max_width``.
    loop = loop.for_width(int(lanes))
    if loop is None:
        return base
    return loop.limit(base)


def draft_depth_ceiling(config: Optional[Mapping[str, Any]]) -> int:
    """Most drafts any lane may carry under ``config`` at any cohort width.

    ``DraftLoopPolicy.limit`` on a ``by_width`` policy is the deepest reach
    over every listed width, so this is the ceiling the adapter's exact row
    window must admit for the whole route.
    """
    config = config or {}
    base = int(config.get("num_draft") or 0)
    loop = DraftLoopPolicy.from_value(config.get("draft_loop"))
    return base if loop is None else loop.limit(base)


def cohort_proposal_depths(
    config: Optional[Mapping[str, Any]],
    *,
    max_lanes: int,
    copy_draft_policy: Any = None,
) -> dict[int, int]:
    """Widest proposal a lane may verify at each cohort width ``1..max_lanes``.

    Combines the loop-aware draft depth with the self-MTP copy-draft span
    the same cohort may verify in place of head drafts (``copy_draft.
    cohort_copy_cap``, fed the loop-aware depth exactly as the executor
    feeds it).  Width is the key, so a caller can apply its own per-row
    slack and take the maximum.
    """
    depths = {}
    for lanes in range(1, int(max_lanes) + 1):
        depth = cohort_draft_depth(config, lanes=lanes)
        if copy_draft_policy is not None and copy_draft_policy.enabled:
            from .copy_draft import cohort_copy_cap

            span = cohort_copy_cap(
                copy_draft_policy, lanes=lanes, head_depths=(depth,) * lanes
            )
            depth = max(depth, span)
        depths[lanes] = depth
    return depths


__all__ = [
    "COHORT_RULES",
    "DraftLoopPolicy",
    "cohort_draft_depth",
    "cohort_proposal_depths",
    "draft_depth_ceiling",
]
