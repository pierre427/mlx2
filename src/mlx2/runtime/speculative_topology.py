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
    decision_token: int


@dataclass(frozen=True)
class LiveFrontierAdmission:
    """One request-private branch admitted to the shared target frontier."""

    session_slot: int
    branch_index: int
    proposal_count: int


class OrderedLiveFrontierPlanner:
    """Pure ordered admission state for cross-session AST packing.

    A session has at most one live branch.  Branch ``n + 1`` becomes eligible
    only after branch ``n`` completed conclusively non-full.  The first full
    branch therefore wins without scheduling any higher rank; exhausting all
    branches leaves B1 as the caller's exact fallback.
    """

    def __init__(self, proposal_counts):
        counts = tuple(tuple(int(value) for value in row) for row in proposal_counts)
        if not 1 <= len(counts) <= 4:
            raise ValueError("live frontier requires one to four sessions")
        if any(
            not 2 <= len(row) <= 4 or any(value < 1 for value in row)
            for row in counts
        ):
            raise ValueError("live frontier requires two to four positive branch depths")
        self.proposal_counts = counts
        self._started = False
        self._active = {}
        self._visited = [set() for _row in counts]
        self._retired = set()
        self._winner = [None] * len(counts)
        self.rows_admitted = 0
        self.refills = 0

    def _admit(self, session_slot, branch_index):
        if session_slot in self._retired or session_slot in self._active:
            raise ValueError("live frontier session is not eligible for admission")
        if branch_index != len(self._visited[session_slot]):
            raise ValueError("live frontier branch admission is out of order")
        self._visited[session_slot].add(branch_index)
        self._active[session_slot] = branch_index
        self.rows_admitted += 1
        return LiveFrontierAdmission(
            session_slot=session_slot,
            branch_index=branch_index,
            proposal_count=self.proposal_counts[session_slot][branch_index],
        )

    def start(self, session_slots=None):
        if self._started:
            raise ValueError("live frontier was already started")
        self._started = True
        slots = tuple(
            range(len(self.proposal_counts))
            if session_slots is None
            else tuple(int(slot) for slot in session_slots)
        )
        if len(set(slots)) != len(slots) or any(
            not 0 <= slot < len(self.proposal_counts) for slot in slots
        ):
            raise ValueError("live frontier start contains an invalid session slot")
        return tuple(self._admit(slot, 0) for slot in slots)

    def retire_unadmitted(self, session_slot):
        """Retire a session whose shared trunk rejected before branch admission."""

        session_slot = int(session_slot)
        if session_slot in self._retired or session_slot in self._active:
            raise ValueError("live frontier session cannot be retired")
        self._retired.add(session_slot)

    def complete(self, session_slot, branch_index, *, full):
        """Retire a winner/exhausted session or immediately admit its next rank."""

        session_slot = int(session_slot)
        branch_index = int(branch_index)
        if self._active.get(session_slot) != branch_index:
            raise ValueError("live frontier completion does not match the active branch")
        del self._active[session_slot]
        if bool(full):
            self._winner[session_slot] = branch_index
            self._retired.add(session_slot)
            return None
        next_branch = branch_index + 1
        if next_branch == len(self.proposal_counts[session_slot]):
            self._retired.add(session_slot)
            return None
        self.refills += 1
        return self._admit(session_slot, next_branch)

    @property
    def finished(self):
        return len(self._retired) == len(self.proposal_counts) and not self._active

    def snapshot(self):
        total_rows = sum(len(row) for row in self.proposal_counts)
        total_tokens = sum(sum(row) for row in self.proposal_counts)
        scheduled_tokens = sum(
            self.proposal_counts[slot][branch]
            for slot, branches in enumerate(self._visited)
            for branch in branches
        )
        return {
            "mode": "ordered_live_frontier",
            "rows_admitted": self.rows_admitted,
            "sessions_retired": len(self._retired),
            "branches_unvisited": total_rows - self.rows_admitted,
            "proposal_tokens_not_scheduled": total_tokens - scheduled_tokens,
            "refills": self.refills,
            "winners": [None if value is None else value + 1 for value in self._winner],
            "finished": self.finished,
        }


class AdaptiveSpeculationTopology:
    """Adaptive Speculation Topology (AST) for one bound proposer and shape.

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
        max_park_rounds=64,
    ):
        if type(configured_branches) is not int or not 2 <= configured_branches <= 4:
            raise ValueError("adaptive topology latch requires two to four branches")
        if min(window, shrink_after, park_rounds, reprobe_interval) < 1:
            raise ValueError("adaptive topology latch bounds must be positive")
        if max_park_rounds < park_rounds:
            raise ValueError("adaptive topology max_park_rounds must cover park_rounds")
        self.configured_branches = configured_branches
        self.active_branches = configured_branches
        self.window = deque(maxlen=window)
        self.shrink_after = shrink_after
        self.park_rounds = park_rounds
        self.max_park_rounds = max_park_rounds
        # A park that expires re-engages the economical two-way check; each
        # park that ends without a win doubles the next one, up to the cap.
        self.park_length = park_rounds
        self.park_expiries = 0
        self.last_park_expiry_token = 0
        self.reprobe_interval = reprobe_interval
        self.no_alternative_win = 0
        self.park_remaining = 0
        self.decisions = 0
        self.completed_samples = 0
        self.completed_sessions = 0
        self.censored_sessions = 0
        self.primary_only_samples = 0
        self.primary_only_sessions = 0
        self.downshifts = 0
        self.periodic_reprobes = 0
        self.accepted_token_uplift = 0
        self.avoidable_target_rows = 0
        self.last_applied_token = 0

    def _choose_shape(self):
        """Preview the next shape without consuming adaptive state."""

        token = self.decisions + 1
        reprobe = token % self.reprobe_interval == 0
        reason = "configured_width"
        active = self.active_branches
        if self.park_remaining and not reprobe:
            active = 1
            reason = "parked_after_repeated_no_alternative_win"
        elif reprobe:
            active = self.configured_branches
            reason = "periodic_reprobe"
        elif active < self.configured_branches:
            reason = (
                "park_expired_economical_check"
                if self.park_expiries and self.last_park_expiry_token == self.decisions
                else "reduced_after_prior_completed_batches"
            )
        return active, reason, reprobe, token

    def _choose_trunk(self, suggested_trunk, proposal_count, verification_tile):
        maximum = min(verification_tile - 1, proposal_count - 1)
        trunk = max(1, min(int(suggested_trunk), maximum))
        if self.window:
            primary_depth = int(median(row["branch_accepts"][0] for row in self.window))
            alternative_wins = sum(row["selected_branch"] > 1 for row in self.window)
            if primary_depth <= trunk:
                trunk = max(1, min(primary_depth, maximum))
            elif alternative_wins == 0:
                trunk = min(maximum, trunk + 1)
        return trunk

    def choose(self, suggested_trunk, proposal_count, verification_tile):
        """Choose from prior samples, before observing the current batch."""

        active, reason, reprobe, token = self._choose_shape()
        trunk = self._choose_trunk(
            suggested_trunk, proposal_count, verification_tile
        )
        return TopologyLatchDecision(
            active_branches=active,
            trunk_tokens=trunk,
            reason=reason,
            reprobe=reprobe,
            completed_samples=self.completed_samples,
            window_samples=len(self.window),
            decision_token=token,
        )

    def choose_many(self, suggested_trunks, proposal_counts, verification_tile):
        """Choose one lagged shape for a cohort and per-session tree trunks."""

        suggested_trunks = tuple(suggested_trunks)
        proposal_counts = tuple(proposal_counts)
        if not suggested_trunks or len(suggested_trunks) != len(proposal_counts):
            raise ValueError("AST cohort choice requires one depth per session")
        active, reason, reprobe, token = self._choose_shape()
        return tuple(
            TopologyLatchDecision(
                active_branches=active,
                trunk_tokens=self._choose_trunk(
                    suggested, count, verification_tile
                ),
                reason=reason,
                reprobe=reprobe,
                completed_samples=self.completed_samples,
                window_samples=len(self.window),
                decision_token=token,
            )
            for suggested, count in zip(
                suggested_trunks, proposal_counts, strict=True
            )
        )

    def observe(
        self,
        *,
        branch_accepts,
        proposal_counts,
        selected_branch,
        unique_target_rows,
    ):
        """Update future decisions from one fully completed B1 batch."""

        self.observe_many(
            [
                {
                    "branch_accepts": branch_accepts,
                    "proposal_counts": proposal_counts,
                    "selected_branch": selected_branch,
                    "unique_target_rows": unique_target_rows,
                    "outcome": "committed_conclusive",
                }
            ],
            decision_token=self.decisions + 1,
        )

    def observe_many(self, observations, *, decision_token):
        """Apply one order-independent, post-commit cohort observation."""

        observations = tuple(dict(row) for row in observations)
        token = int(decision_token)
        if not observations:
            raise ValueError("AST cohort observation requires at least one session")
        if token == self.last_applied_token:
            return False
        if token != self.decisions + 1:
            raise ValueError("AST cohort observation token is stale or out of order")

        rows = []
        conclusive = []
        for observation in observations:
            outcome = observation.get("outcome", "committed_conclusive")
            if outcome not in {
                "committed_conclusive",
                "committed_censored",
                "cancelled",
                "faulted",
            }:
                raise ValueError("unknown AST cohort observation outcome")
            accepts = tuple(int(value) for value in observation["branch_accepts"])
            selected_branch = int(observation["selected_branch"])
            if not 1 <= selected_branch <= len(accepts):
                raise ValueError("AST selected branch is outside the observed topology")
            counts = _proposal_counts(observation["proposal_counts"], len(accepts))
            selected_count = counts[selected_branch - 1]
            visited = tuple(value for value in accepts if value >= 0)
            row = {
                "branch_accepts": accepts,
                "proposal_counts": counts,
                "selected_proposal_count": selected_count,
                "selected_branch": selected_branch,
                "unique_target_rows": int(observation["unique_target_rows"]),
                "committed_tokens": max(1, int(accepts[selected_branch - 1]) + 1),
                "primary_committed_tokens": max(1, int(accepts[0]) + 1),
                "visited_branches": len(visited),
                "outcome": outcome,
            }
            row["accepted_token_uplift"] = max(
                0,
                row["committed_tokens"] - row["primary_committed_tokens"],
            )
            rows.append(row)
            if outcome == "committed_conclusive":
                conclusive.append(row)

        reprobe = token % self.reprobe_interval == 0
        if not reprobe:
            self._tick_park(token)
        self.decisions = token
        self.last_applied_token = token
        self.completed_samples += 1
        self.completed_sessions += len(rows)
        self.censored_sessions += len(rows) - len(conclusive)
        self.window.extend(conclusive)

        accepted_token_uplift = sum(
            row["accepted_token_uplift"]
            for row in conclusive
        )
        avoidable_target_rows = sum(
            max(0, row["unique_target_rows"] - row["committed_tokens"])
            for row in conclusive
        )
        self.accepted_token_uplift += accepted_token_uplift
        self.avoidable_target_rows += avoidable_target_rows
        if reprobe:
            self.periodic_reprobes += 1

        alternative_winners = [
            row["selected_branch"]
            for row in conclusive
            if row["selected_branch"] > 1
            and row["accepted_token_uplift"] > 0
        ]
        if alternative_winners:
            self.no_alternative_win = 0
            self.active_branches = min(
                self.configured_branches,
                max(self.active_branches, max(alternative_winners)),
            )
            self.park_remaining = 0
            self.park_length = self.park_rounds
        elif (
            len(conclusive) == len(rows)
            and accepted_token_uplift == 0
            and avoidable_target_rows > 0
        ):
            self.no_alternative_win += 1
            if self.no_alternative_win >= self.shrink_after:
                previous = self.active_branches
                # Four-way verification is the exploratory shape; two-way is
                # the economical alternative check.  Width three preserves
                # most of the former cost while discarding only one chance to
                # win, so it is not a useful steady-state latch position.
                self.active_branches = 2 if self.active_branches > 2 else 1
                self.downshifts += int(self.active_branches < previous)
                self.no_alternative_win = 0
                if self.active_branches == 1:
                    self.park_remaining = self.park_length
                    self.park_length = min(2 * self.park_length, self.max_park_rounds)
        return True

    def observe_primary_only(self, *, decision_token, session_count):
        """Advance a parked decision after its exact primary rows commit.

        No alternative was executed, so this advances only the sparse-audit
        clock.  It contributes neither a win nor a no-uplift sample and cannot
        itself change the active width.  Calling it after commit keeps retries,
        faults and cancellations from consuming an audit epoch.
        """

        token = int(decision_token)
        session_count = int(session_count)
        if session_count < 1:
            raise ValueError("AST primary-only observation needs a live session")
        if token == self.last_applied_token:
            return False
        if token != self.decisions + 1:
            raise ValueError("AST primary-only observation token is stale or out of order")
        reprobe = token % self.reprobe_interval == 0
        if reprobe:
            raise ValueError("AST periodic reprobe cannot be recorded as primary-only")
        self._tick_park(token)
        self.decisions = token
        self.last_applied_token = token
        self.completed_samples += 1
        self.completed_sessions += session_count
        self.primary_only_samples += 1
        self.primary_only_sessions += session_count
        return True

    def _tick_park(self, token):
        if not self.park_remaining:
            return
        self.park_remaining -= 1
        if not self.park_remaining and self.active_branches == 1:
            self.active_branches = min(2, self.configured_branches)
            self.park_expiries += 1
            self.last_park_expiry_token = int(token)

    def snapshot(self):
        rows = tuple(self.window)
        return {
            "configured_branches": self.configured_branches,
            "active_branches": self.active_branches,
            "completed_samples": self.completed_samples,
            "completed_sessions": self.completed_sessions,
            "censored_sessions": self.censored_sessions,
            "primary_only_samples": self.primary_only_samples,
            "primary_only_sessions": self.primary_only_sessions,
            "window_samples": len(rows),
            "alternative_selections": sum(
                row["selected_branch"] > 1 for row in rows
            ),
            "alternative_wins": sum(
                row["selected_branch"] > 1
                and row["accepted_token_uplift"] > 0
                for row in rows
            ),
            "zero_uplift_alternative_selections": sum(
                row["selected_branch"] > 1
                and row["accepted_token_uplift"] == 0
                for row in rows
            ),
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
            "park_length": self.park_length,
            "park_expiries": self.park_expiries,
            "downshifts": self.downshifts,
            "periodic_reprobes": self.periodic_reprobes,
            "accepted_token_uplift": self.accepted_token_uplift,
            "avoidable_target_rows": self.avoidable_target_rows,
            "last_applied_token": self.last_applied_token,
            "next_periodic_reprobe_in": self.reprobe_interval
            - (self.decisions % self.reprobe_interval),
        }


class PerSessionSpeculationTopology:
    """One AST latch per request session.

    Each session decides its own branch width from its own completed batches,
    keeps its own decision clock (reprobes, parks) and is never narrowed or
    widened by another session sharing its physical launch.  Session state is
    created on first use and released when the request leaves the executor.
    """

    scope = "request_session"

    def __init__(self, configured_branches, **settings):
        # Validate once, eagerly, so a bad configuration fails at startup.
        AdaptiveSpeculationTopology(configured_branches, **settings)
        self.configured_branches = configured_branches
        self._settings = dict(settings)
        self._sessions = {}
        self.released_sessions = 0

    def session(self, uid):
        uid = int(uid)
        latch = self._sessions.get(uid)
        if latch is None:
            latch = AdaptiveSpeculationTopology(
                self.configured_branches, **self._settings
            )
            self._sessions[uid] = latch
        return latch

    def release(self, uid):
        if self._sessions.pop(int(uid), None) is not None:
            self.released_sessions += 1

    def choose_many(
        self, suggested_trunks, proposal_counts, verification_tile, *, session_uids
    ):
        """One decision per session, each from that session's own history."""

        suggested_trunks = tuple(suggested_trunks)
        proposal_counts = tuple(proposal_counts)
        session_uids = tuple(session_uids)
        if (
            not suggested_trunks
            or len(suggested_trunks) != len(proposal_counts)
            or len(session_uids) != len(proposal_counts)
        ):
            raise ValueError("AST cohort choice requires one depth per session")
        return tuple(
            self.session(uid).choose(suggested, count, verification_tile)
            for uid, suggested, count in zip(
                session_uids, suggested_trunks, proposal_counts, strict=True
            )
        )

    def observe_many(self, observations, *, session_uids, decision_tokens):
        """Apply each session's committed row to that session's latch."""

        observations = tuple(observations)
        session_uids = tuple(session_uids)
        decision_tokens = tuple(decision_tokens)
        if not observations or not (
            len(observations) == len(session_uids) == len(decision_tokens)
        ):
            raise ValueError("AST observation requires one token per session")
        applied = False
        for uid, observation, token in zip(
            session_uids, observations, decision_tokens, strict=True
        ):
            applied = (
                self.session(uid).observe_many([observation], decision_token=token)
                or applied
            )
        return applied

    def observe_primary_only(self, *, session_uids, decision_tokens):
        applied = False
        for uid, token in zip(session_uids, decision_tokens, strict=True):
            applied = (
                self.session(uid).observe_primary_only(
                    decision_token=token, session_count=1
                )
                or applied
            )
        return applied

    def snapshot(self, uid=None):
        if uid is not None:
            return {"scope": self.scope, "session": int(uid), **self.session(uid).snapshot()}
        # A session that has not committed a decision carries no state: its
        # latch only previews, so omitting it keeps a failed round's snapshot
        # identical to the one before it.
        sessions = {
            str(key): latch.snapshot()
            for key, latch in sorted(self._sessions.items())
            if latch.decisions
            or latch.completed_samples
            or latch.window
            or latch.active_branches != latch.configured_branches
            or latch.park_remaining
        }
        return {
            "scope": self.scope,
            "configured_branches": self.configured_branches,
            "live_sessions": len(sessions),
            "released_sessions": self.released_sessions,
            "active_branches_by_session": {
                key: snap["active_branches"] for key, snap in sessions.items()
            },
            "sessions": sessions,
        }


# Deprecated class names remain import-compatible for persisted callers.
AdaptiveProposalTopology = AdaptiveSpeculationTopology
AdaptiveTopologyLatch = AdaptiveSpeculationTopology


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


def ordered_branch_commit_counts(decisions):
    """Return the exact token count each verified branch would commit."""

    return tuple(
        max(
            0,
            int(decision.commit_count)
            if decision.commit_count is not None
            else min(int(decision.accepted) + 1, len(decision.emitted)),
        )
        for decision in decisions
    )


def ordered_alternative_winner(
    decisions, proposal_counts, *, allow_partial_uplift=False
):
    """Choose the first complete branch that does not lose B1 token utility.

    B1 remains the exact fallback.  A shorter confidence-tail branch can be
    fully accepted while committing fewer tokens than B1's partial path; such
    a branch is still reported as full evidence, but cannot replace B1.

    A complete-path provider may opt into exact partial-prefix utility.  Its
    target-verified branch owns a private cache and RNG transaction, so a
    branch that commits strictly more exact tokens than the ordinary B1 path
    is useful even when a later proposal rejects.  Ties retain the existing
    full-branch/B1 order.
    """

    counts = _proposal_counts(proposal_counts, len(decisions))
    committed = ordered_branch_commit_counts(decisions)
    full = tuple(
        index
        for index, (decision, count) in enumerate(zip(decisions, counts, strict=True))
        if int(decision.accepted) == count
    )
    useful_full = tuple(
        index for index in full if committed[index] >= committed[0]
    )
    winner = useful_full[0] if useful_full else 0
    if allow_partial_uplift:
        best = max(
            range(len(committed)),
            key=lambda index: (committed[index], -index),
        )
        if committed[best] > committed[winner]:
            winner = best
    return winner, full


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
    if type(trunk_tokens) is not int or not 0 <= trunk_tokens < proposal_count:
        raise ValueError("confidence tree trunk must leave a continuation token")
    short = min(proposal_count, trunk_tokens + 1)
    return tuple(
        proposal_count if index < 2 else short for index in range(branch_count)
    )


def processors_branch_safe(processors) -> bool:
    """Whether private AST/progressive branches may evaluate ``processors``.

    Every processor must declare the stateless branch contract: a law that
    depends only on the supplied history (``history_pure``), no mutable
    receipt or grammar state (``progressive_branch_safe``), and its forbidden
    ids per length (``forbidden_token_ids_at_length``).  Verification still
    applies the processors to the target, so an unselected or masked branch
    row changes nothing.  An empty list is safe.
    """
    return all(
        getattr(processor, "progressive_branch_safe", False) is True
        and getattr(processor, "history_pure", False) is True
        and callable(getattr(processor, "forbidden_token_ids_at_length", None))
        for processor in processors
    )


def ordered_alternative_proposals(
    proposer,
    primary_tokens,
    *,
    branch_count,
    strategy,
    shared_prefix_tokens,
    allow_adaptive_prefix_search=False,
):
    """Request model-private paths through the generic proposer capability."""

    validate_ordered_alternative_strategy(strategy, branch_count)
    provider = getattr(proposer, "ordered_alternative_proposals", None)
    if not callable(provider):
        raise ValueError("drafter does not provide ordered alternative proposals")
    options = {
        "branch_count": branch_count,
        "strategy": strategy,
        "shared_prefix_tokens": shared_prefix_tokens,
    }
    if allow_adaptive_prefix_search:
        options["allow_adaptive_prefix_search"] = True
    rows = provider(primary_tokens, **options)
    if not isinstance(rows, (list, tuple)) or len(rows) != len(primary_tokens):
        raise ValueError("drafter returned invalid ordered alternative rows")
    return rows
