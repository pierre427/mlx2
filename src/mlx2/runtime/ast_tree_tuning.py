"""Session-local measured source allocation; no tensors or model-specific policy.

Only complete, uncensored rounds train emitted-token throughput. The sixteen-row
target capacity is not a count of sessions or proposal paths. Expected tiles is
a conservative no-sharing source bound, not a predicted execution measurement.
"""

from collections import OrderedDict
from dataclasses import asdict, dataclass
import math


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


@dataclass(frozen=True)
class ASTTreeSourcePlan:
    depth: int
    branches: int
    max_tile_rows: int
    max_tiles: int
    expected_tiles: int
    ready_sessions: int
    reason: str
    cost_samples: int
    terminal_censored: bool
    target_sessions: int = 1
    source_depth_limit: int | None = None

    def receipt(self):
        return asdict(self)


class ASTTreeTuner:
    """Bounded EWMA and deterministic adjacent exploration for one exact scope.

    The caller supplies a source/model/session-bound immutable scope and never
    transfers this instance to another session. No histories or cache state are
    retained. Explicit fixed plans bypass both adaptation and pressure shaping.

    Startup seeds a chain and keeps it until it has ``INCUMBENT_SAMPLES``
    measurements; a new adjacent shape is probed at most every
    ``PROBE_PERIOD`` rounds, and a measured challenger (``CANDIDATE_SAMPLES``
    or more) replaces the champion only when it beats it by ``HYSTERESIS``.
    """

    INCUMBENT_SAMPLES = 4
    CANDIDATE_SAMPLES = 2
    PROBE_PERIOD = 8
    HYSTERESIS = 1.10
    # Startup width; 1 is a chain.  Tests that exercise wide startup trees
    # raise it explicitly rather than relying on the production seed.
    SEED_BRANCHES = 1

    def __init__(
        self,
        scope,
        *,
        max_depth,
        minimum_depth=1,
        max_branches=4,
        max_tile_rows=16,
        fixed_depth=None,
        fixed_branches=None,
    ):
        try:
            valid_scope = isinstance(scope, (str, tuple)) and bool(scope)
            hash(scope)
        except TypeError:
            valid_scope = False
        if (
            not valid_scope
            or not _integer(max_depth, 1, 15)
            or not _integer(minimum_depth, 1, 15)
            or not (_integer(max_branches, 1, 4) and _integer(max_tile_rows, 2, 16))
        ):
            raise ValueError("AST tuner requires an immutable scope and legal geometry")
        if (fixed_depth is None) != (fixed_branches is None) or (
            fixed_depth is not None
            and not (
                _integer(fixed_depth, 1, max_depth)
                and _integer(fixed_branches, 1, max_branches)
            )
        ):
            raise ValueError("AST fixed source geometry is incomplete or unsupported")
        self.scope = scope
        self.max_depth, self.max_branches = max_depth, max_branches
        self.minimum_depth = min(minimum_depth, max_depth)
        self.max_tile_rows = max_tile_rows
        self.fixed = None if fixed_depth is None else (fixed_depth, fixed_branches)
        self._stats, self._champions, self._probes = OrderedDict(), {}, {}
        self._rounds, self._skipped = {}, {}
        self._pending = self._last_plan = None

    def _budget(self, pressure, target_sessions):
        quota = self.max_tile_rows // target_sessions
        if self.minimum_depth >= quota and target_sessions > 1:
            raise ValueError("AST cohort quota cannot preserve the proposal floor")
        # The physical cohort quota already accounts for shared target rows.
        # Reserve a second tile's proposals; a rejection cancels stale work.
        # Dividing by ready pressure again would starve the second tile.
        return quota, 2, max(self.minimum_depth, 2 * quota - 1)

    def _legal(self, choice, budget, depth_limit):
        depth, branches = choice
        return (
            self.minimum_depth <= depth <= depth_limit
            and 1 <= branches <= self.max_branches
            and depth * branches <= budget
        )

    def _sample(self, group, choice):
        return self._stats.get((*group, *choice))

    @staticmethod
    def _rate(sample):
        return sample["emitted"] / sample["seconds"]

    def _choose(self, group, budget):
        # Seed the cheapest full-depth shape (a chain).  Wider trees cost a
        # full extra tile per round and must earn their place by measurement;
        # seeding the most expensive legal shape spent the startup rounds on
        # the widest tree and kept re-electing it from noisy samples.
        depth = min(group[2], budget)
        seed = (
            depth,
            1
            if group[1] > 1
            else max(1, min(self.max_branches, self.SEED_BRANCHES, budget // depth)),
        )
        incumbent = self._champions.setdefault(group, seed)
        current = self._sample(group, incumbent)
        measured = [
            (key[3:], value)
            for key, value in self._stats.items()
            if key[:3] == group
            and value["samples"] >= self.CANDIDATE_SAMPLES
            and self._legal(key[3:], budget, group[2])
        ]
        if current and current["samples"] >= self.INCUMBENT_SAMPLES and measured:
            best, sample = max(measured, key=lambda item: self._rate(item[1]))
            if self._rate(sample) > self._rate(current) * self.HYSTERESIS:
                incumbent = self._champions[group] = best
                current = sample
        if current is None or current["samples"] < self.INCUMBENT_SAMPLES:
            return incumbent, "startup_measurement"
        probe = self._probes.get(group)
        if probe is not None:
            sample = self._sample(group, probe)
            if sample is None or sample["samples"] < self.CANDIDATE_SAMPLES:
                return probe, "adjacent_exploration"
            del self._probes[group]
        count = self._rounds.get(group, 0)
        depth, branches = incumbent
        neighbors = [
            (depth, max(1, branches // 2)),
            (depth, branches - 1),
            (max(1, depth // 2), branches),
            ((depth + 1) // 2, branches),
            (depth - 1, branches),
            (depth, branches + 1),
            (depth * 2, branches),
            (depth + 1, branches),
        ]
        neighbors = list(
            dict.fromkeys(
                item
                for item in neighbors
                if item != incumbent
                and self._legal(item, budget, group[2])
                and (group[1] == 1 or item[1] == 1)
            )
        )
        if count % self.PROBE_PERIOD == 2:
            unseen = [
                item
                for item in neighbors
                if self._sample(group, item) is None
                or self._sample(group, item)["samples"] < self.CANDIDATE_SAMPLES
            ]
            if unseen:
                self._probes[group] = unseen[0]
                return unseen[0], "adjacent_exploration"
        # Recheck a known adjacent choice occasionally; no benchmark-only work.
        if neighbors and count and count % 16 == 0:
            candidate = min(
                neighbors,
                key=lambda item: (
                    self._sample(group, item)["last"]
                    if self._sample(group, item)
                    else -1
                ),
            )
            return candidate, "adjacent_recheck"
        return incumbent, "measured_throughput_hysteresis"

    def plan(self, *, remaining_tokens, ready_sessions=1, target_sessions=1,
             source_depth_limit=None):
        if (
            not _integer(remaining_tokens, 0, 2**63 - 1)
            or not _integer(ready_sessions, 1, 2**31 - 1)
            or not _integer(target_sessions, 1, min(4, ready_sessions))
        ):
            raise ValueError(
                "AST source planning needs nonnegative headroom and live pressure"
            )
        depth_limit = self.max_depth if source_depth_limit is None else source_depth_limit
        if not _integer(depth_limit, self.minimum_depth, self.max_depth):
            raise ValueError("AST source depth limit must preserve the proposal floor")
        # A configured work cap is not a terminal response budget. Scope the
        # champion to that cap so clipped proposals train their actual shape.
        # Beyond sixteen ready sessions the configured source floor is binding.
        pressure = min(ready_sessions, 16)
        group = (pressure, target_sessions, depth_limit)
        tile_rows, max_tiles, budget = self._budget(pressure, target_sessions)
        if self.fixed is None:
            choice, reason = self._choose(group, budget)
        else:
            choice, reason = (min(self.fixed[0], depth_limit), self.fixed[1]), "explicit_fixed"
            max_tiles = max(2, math.ceil((1 + choice[0] * choice[1]) / tile_rows))
        depth = min(choice[0], max(0, remaining_tokens - 1))
        branches = choice[1] if depth else 1
        sample = self._sample(group, choice)
        result = ASTTreeSourcePlan(
            depth,
            branches,
            tile_rows,
            max_tiles,
            math.ceil((1 + depth * branches) / tile_rows) if remaining_tokens else 0,
            ready_sessions,
            "terminal_anchor" if not depth else reason,
            0 if sample is None else sample["samples"],
            remaining_tokens < depth_limit + 1,
            target_sessions,
            depth_limit,
        )
        self._pending = self._last_plan = result
        return result

    def observe(
        self,
        plan,
        *,
        seconds,
        emitted,
        accepted,
        unique_rows,
        launches,
        completed=True,
        terminal=False,
        rejected=False,
    ):
        """Observe only successful full-round time, including draft and publication.

        ``rejected`` means the round itself was discarded/refused. Ordinary
        proposal-token rejection (including accepted=0) is valid training data.
        No evaluation/synchronization or tensor access occurs here.
        """
        if plan is not self._pending:
            reason = "foreign_or_stale_plan"
        else:
            self._pending = None
            reason = (
                "fixed"
                if self.fixed is not None
                else "incomplete"
                if not completed
                else "rejected_round"
                if rejected
                else "terminal"
                if terminal or plan.terminal_censored or not plan.depth
                else None
            )
        if reason is None and not (
            type(seconds) in (float, int)
            and math.isfinite(seconds)
            and seconds > 0
            and _integer(emitted, 1, plan.depth + 1)
            and _integer(accepted, 0, min(plan.depth, emitted))
            and _integer(unique_rows, 1, 1 + plan.depth * plan.branches)
            and _integer(launches, 1, unique_rows)
        ):
            reason = "invalid_measurement"
        if reason is not None:
            self._skipped[reason] = self._skipped.get(reason, 0) + 1
            return False
        group = (min(plan.ready_sessions, 16), plan.target_sessions, plan.source_depth_limit)
        self._rounds[group] = self._rounds.get(group, 0) + 1
        key = (*group, plan.depth, plan.branches)
        old = self._stats.get(key)
        values = dict(
            seconds=float(seconds),
            emitted=emitted,
            accepted=accepted,
            unique_rows=unique_rows,
            launches=launches,
        )
        if old:
            values = {
                key: 0.75 * old[key] + 0.25 * value for key, value in values.items()
            }
        values.update(
            samples=1 if old is None else old["samples"] + 1,
            last=self._rounds[group],
        )
        self._stats[key] = values
        self._stats.move_to_end(key)
        while len(self._stats) > 32:
            self._stats.popitem(last=False)
        return True

    def receipt(self):
        return {
            "scope": self.scope,
            "implemented": True,
            "defaulted": False,
            "policy": "session_ewma_adjacent_exploration_v1",
            "timing_boundary": "completed_draft_verify_publish_round",
            "timing_attribution": "per_request_yield_over_full_cohort_turn",
            "expected_tiles_authority": "no_sharing_source_upper_bound",
            "maximum_tile_rows": self.max_tile_rows,
            "minimum_depth": self.minimum_depth,
            "fixed": self.fixed,
            "cost_samples": sum(value["samples"] for value in self._stats.values()),
            "skipped": dict(self._skipped),
            "history_limit": 32,
            "measurements": [
                dict(
                    ready_sessions=key[0],
                    target_sessions=key[1],
                    source_depth_limit=key[2],
                    depth=key[3],
                    branches=key[4],
                    **value,
                    emitted_tokens_per_second=self._rate(value),
                )
                for key, value in self._stats.items()
            ],
            "last_plan": None if self._last_plan is None else self._last_plan.receipt(),
            "optimality_claim": False,
        }
