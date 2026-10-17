"""Scheduler-owned proposal topology helpers.

Draft adapters may expose ranked proposal paths, but the runtime owns branch
ordering, staged verification cohorts, early stopping and winner semantics.
Nothing in this module depends on a model family or proposer implementation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median


ORDERED_ALTERNATIVE_STRATEGIES = frozenset(
    {
        "distinct_root",
        "shared_prefix",
        "paired_continuations",
        "confidence_tree",
    }
)


@dataclass(frozen=True)
class TopologyLatchDecision:
    active_branches: int
    trunk_tokens: int
    reason: str
    reprobe: bool
    completed_samples: int
    window_samples: int


class AdaptiveTopologyLatch:
    """Lagged acceptance/cost controller for one bound drafter and shape.

    Only completed verification batches enter the history.  Current proposal
    confidence may choose tree edges inside the admitted shape, but it cannot
    alter the latch decision that admitted that shape.
    """

    def __init__(
        self,
        configured_branches,
        *,
        window=8,
        shrink_after=3,
        park_rounds=4,
        reprobe_interval=16,
    ):
        if type(configured_branches) is not int or not 2 <= configured_branches <= 4:
            raise ValueError("adaptive topology latch requires two to four branches")
        if min(window, shrink_after, park_rounds, reprobe_interval) < 1:
            raise ValueError("adaptive topology latch bounds must be positive")
        self.configured_branches = configured_branches
        self.active_branches = configured_branches
        self.window = deque(maxlen=window)
        self.shrink_after = shrink_after
        self.park_rounds = park_rounds
        self.reprobe_interval = reprobe_interval
        self.no_alternative_win = 0
        self.park_remaining = 0
        self.decisions = 0
        self.completed_samples = 0

    def choose(self, suggested_trunk, proposal_count, verification_tile):
        """Choose from prior samples, before observing the current batch."""

        self.decisions += 1
        reprobe = self.decisions % self.reprobe_interval == 0
        reason = "configured_width"
        active = self.active_branches
        if self.park_remaining and not reprobe:
            self.park_remaining -= 1
            active = 1
            reason = "parked_after_repeated_no_alternative_win"
        elif reprobe:
            active = self.configured_branches
            reason = "periodic_reprobe"
        elif active < self.configured_branches:
            reason = "reduced_after_prior_completed_batches"

        maximum = min(verification_tile - 1, proposal_count - 1)
        trunk = max(1, min(int(suggested_trunk), maximum))
        if self.window:
            primary_depth = int(median(row["branch_accepts"][0] for row in self.window))
            alternative_wins = sum(row["selected_branch"] > 1 for row in self.window)
            if primary_depth <= trunk:
                trunk = max(1, min(primary_depth, maximum))
            elif alternative_wins == 0:
                trunk = min(maximum, trunk + 1)
        return TopologyLatchDecision(
            active_branches=active,
            trunk_tokens=trunk,
            reason=reason,
            reprobe=reprobe,
            completed_samples=self.completed_samples,
            window_samples=len(self.window),
        )

    def observe(
        self,
        *,
        branch_accepts,
        proposal_counts,
        selected_branch,
        unique_target_rows,
    ):
        """Update future decisions from one fully completed verification batch."""

        accepts = tuple(int(value) for value in branch_accepts)
        if not accepts or any(value < 0 for value in accepts[:selected_branch]):
            # Staged/unvisited rows are still recorded, but never promoted as
            # evidence that an unvisited alternative lost.
            visited = tuple(value for value in accepts if value >= 0)
        else:
            visited = accepts
        counts = _proposal_counts(proposal_counts, len(accepts))
        selected_count = counts[selected_branch - 1]
        self.window.append(
            {
                "branch_accepts": accepts,
                "proposal_counts": counts,
                "selected_proposal_count": selected_count,
                "selected_branch": int(selected_branch),
                "unique_target_rows": int(unique_target_rows),
                "committed_tokens": max(1, int(accepts[selected_branch - 1]) + 1),
                "visited_branches": len(visited),
            }
        )
        self.completed_samples += 1
        if selected_branch > 1:
            self.no_alternative_win = 0
            self.active_branches = min(
                self.configured_branches, max(self.active_branches, selected_branch)
            )
            self.park_remaining = 0
        else:
            self.no_alternative_win += 1
            if self.no_alternative_win >= self.shrink_after:
                self.active_branches = max(1, self.active_branches - 1)
                self.no_alternative_win = 0
                if self.active_branches == 1:
                    self.park_remaining = self.park_rounds

    def snapshot(self):
        rows = tuple(self.window)
        return {
            "configured_branches": self.configured_branches,
            "active_branches": self.active_branches,
            "completed_samples": self.completed_samples,
            "window_samples": len(rows),
            "alternative_wins": sum(row["selected_branch"] > 1 for row in rows),
            "full_accepts": sum(
                row["branch_accepts"][row["selected_branch"] - 1]
                == row["selected_proposal_count"]
                for row in rows
            ),
            "unique_rows_per_commit": (
                sum(row["unique_target_rows"] for row in rows)
                / sum(row["committed_tokens"] for row in rows)
                if rows
                else None
            ),
            "park_remaining": self.park_remaining,
            "next_periodic_reprobe_in": self.reprobe_interval
            - (self.decisions % self.reprobe_interval),
        }


def validate_ordered_alternative_strategy(strategy, branch_count):
    if strategy not in ORDERED_ALTERNATIVE_STRATEGIES:
        raise ValueError("unsupported progressive alternative strategy")
    if strategy in {
        "shared_prefix",
        "paired_continuations",
        "confidence_tree",
    } and branch_count <= 1:
        raise ValueError(
            "shared-prefix progressive alternatives require multiple branches"
        )


def _proposal_counts(proposal_counts, branch_count):
    if type(proposal_counts) is int:
        counts = (proposal_counts,) * branch_count
    else:
        counts = tuple(int(value) for value in proposal_counts)
    if len(counts) != branch_count or any(value < 1 for value in counts):
        raise ValueError("ordered alternatives require one positive depth per branch")
    return counts


def ordered_alternative_winner(decisions, proposal_counts):
    """Choose the first complete ordered branch, else preserve B1 semantics."""

    counts = _proposal_counts(proposal_counts, len(decisions))
    full = tuple(
        index
        for index, (decision, count) in enumerate(zip(decisions, counts, strict=True))
        if int(decision.accepted) == count
    )
    return (full[0] if full else 0), full


def ordered_branch_outcomes(accepted, proposal_counts):
    counts = _proposal_counts(proposal_counts, len(accepted))
    return tuple(
        "unvisited"
        if value < 0
        else "full_accept"
        if value == count
        else "partial_or_rejected"
        for value, count in zip(accepted, counts, strict=True)
    )


def staged_branch_cohorts(branch_count, cohort_width=2):
    """Return ordered half-open branch cohorts for staged verification."""

    if type(branch_count) is not int or branch_count < 1:
        raise ValueError("branch_count must be a positive integer")
    if type(cohort_width) is not int or cohort_width < 1:
        raise ValueError("cohort_width must be a positive integer")
    return tuple(
        (start, min(start + cohort_width, branch_count))
        for start in range(0, branch_count, cohort_width)
    )


def staged_branch_indices(branch_count, strategy, cohort_width=2):
    """Return ordered branch-index cohorts for physical verification.

    Paired continuations are stored as ``B1, B2, B3(B1), B4(B2)`` so their
    physical cohorts must follow the proposal families rather than contiguous
    list order.  This makes B3 an actual continuation experiment beside B1,
    and B4 beside B2, while preserving ranked family order and early stopping.
    """

    validate_ordered_alternative_strategy(strategy, branch_count)
    if strategy == "paired_continuations":
        if branch_count > 4:
            raise ValueError("paired continuations support at most four branches")
        if branch_count <= 2:
            return tuple((index,) for index in range(branch_count))
        first = tuple(index for index in (0, 2) if index < branch_count)
        second = tuple(index for index in (1, 3) if index < branch_count)
        return tuple(cohort for cohort in (first, second) if cohort)
    if strategy == "confidence_tree":
        if branch_count > 4:
            raise ValueError("confidence tree supports at most four branches")
        return (tuple(range(branch_count)),)
    return tuple(
        tuple(range(start, stop))
        for start, stop in staged_branch_cohorts(branch_count, cohort_width)
    )


def confidence_tree_trunk_length(
    confidence,
    proposal_count,
    verification_tile,
    *,
    strong_log_odds=0.0,
):
    """Count a bounded high-confidence trunk before tree fan-out.

    The target tile includes the current anchor, so a tile of four can carry
    at most three linear trunk proposals.  At least one suffix token remains
    for the tree children.
    """

    if type(proposal_count) is not int or proposal_count < 2:
        raise ValueError("confidence tree requires at least two proposal tokens")
    if type(verification_tile) is not int or verification_tile < 2:
        raise ValueError("confidence tree requires a target tile of at least two")
    if len(confidence) < proposal_count:
        raise ValueError("confidence tree requires one feature per proposal token")
    maximum = min(verification_tile - 1, proposal_count - 1)
    trunk = 0
    while trunk < maximum and float(confidence[trunk]) >= strong_log_odds:
        trunk += 1
    return max(1, trunk)


def confidence_tree_branch_depths(proposal_count, branch_count, trunk_tokens):
    """Allocate full top paths and short lower-ranked continuations.

    B1 and B2 retain the complete proposer chains.  B3 and B4 retain the
    confidence-selected trunk plus their first distinct continuation.  The
    shared trunk is verified once, so the suffix tile contains useful unique
    nodes and can become genuinely ragged as lower-ranked paths finish.
    """

    if type(proposal_count) is not int or proposal_count < 2:
        raise ValueError("confidence tree requires at least two proposal tokens")
    if type(branch_count) is not int or not 1 <= branch_count <= 4:
        raise ValueError("confidence tree requires one to four branches")
    if type(trunk_tokens) is not int or not 1 <= trunk_tokens < proposal_count:
        raise ValueError("confidence tree trunk must leave a continuation token")
    short = min(proposal_count, trunk_tokens + 1)
    return tuple(
        proposal_count if index < 2 else short for index in range(branch_count)
    )


def ordered_alternative_proposals(
    proposer,
    primary_tokens,
    *,
    branch_count,
    strategy,
    shared_prefix_tokens,
):
    """Request model-private paths through the generic proposer capability."""

    validate_ordered_alternative_strategy(strategy, branch_count)
    provider = getattr(proposer, "ordered_alternative_proposals", None)
    if not callable(provider):
        raise ValueError("drafter does not provide ordered alternative proposals")
    rows = provider(
        primary_tokens,
        branch_count=branch_count,
        strategy=strategy,
        shared_prefix_tokens=shared_prefix_tokens,
    )
    if not isinstance(rows, (list, tuple)) or len(rows) != len(primary_tokens):
        raise ValueError("drafter returned invalid ordered alternative rows")
    return rows
