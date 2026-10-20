"""Bounded, tensor-independent AST planning over an exact execution contract.

Hardware/kernel costs are supplied by the adapter boundary. They are priors,
not benchmark results. Physical batching never changes request ownership or
the ordered branch frontier. Every plan includes its actual row placement.
"""

from __future__ import annotations

import math
from statistics import median
from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache

SCHEMA = "mlx2.ast-execution-plan.v1"
MAX_READY_ROWS = 16  # The current branch-transaction protocol's hard bound.
# Launch-cost calibration: timed launches are bucketed by physical row count,
# each bucket contributes its median prior and median seconds once it holds
# enough samples, and a monotone (isotonic) curve through those medians
# predicts the seconds of a shape never launched. Medians and monotonicity
# make it robust to the heavy-tailed timing noise that defeats a least-squares
# intercept; a curve that pools into one level carries no shape information
# and is not used.
LAUNCH_FIT_MIN_SAMPLES = 8
LAUNCH_BUCKET_MIN_SAMPLES = 4
LAUNCH_FIT_WINDOW = 256


@dataclass(frozen=True)
class NumericalEquivalenceOracle:
    """An exact backend verdict bound to every mutable execution identity.

    The planner never runs an oracle and never treats a source hash as a
    numerical verdict.  An adapter may publish a backend only after its own
    oracle has produced this complete, affirmative binding.
    """

    profile_binding: str
    device: str
    build: str
    source_revision_token: str
    evidence: str
    equivalent: bool

    def __post_init__(self):
        if type(self.equivalent) is not bool or self.equivalent is not True:
            raise ValueError("AST alternative backend requires a passing numerical oracle")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.profile_binding,
                self.device,
                self.build,
                self.source_revision_token,
                self.evidence,
            )
        ):
            raise ValueError("AST numerical oracle binding is incomplete")


@dataclass(frozen=True)
class AlternativeExecutionBackend:
    """Adapter-owned, exact, single-session target backend and cost law."""

    backend: str
    profile_binding: str
    device: str
    build: str
    source_revision_token: str
    input_widths: tuple[int, ...]
    row_costs: tuple[float, ...]
    launch_cost: float = 0.0
    context_row_cost: float = 0.0
    context_windows: tuple[tuple[int, float], ...] = ()
    cost_authority: str = "adapter_declared_weight_passes"
    maximum_sessions: int = 1
    oracle: NumericalEquivalenceOracle | None = None
    timing_boundary: str = "target_forward"

    def __post_init__(self):
        if not self.backend or self.backend == "padded_row_exact":
            raise ValueError("AST alternative backend identity is invalid")
        if self.maximum_sessions != 1:
            raise ValueError("AST alternative backend is single-session only")
        if self.timing_boundary not in ("target_forward", "target_successor"):
            raise ValueError("AST alternative timing boundary is invalid")
        if not self.input_widths or any(
            type(width) is not int or width < 1 for width in self.input_widths
        ):
            raise ValueError("AST alternative backend requires exact input widths")
        if len(self.row_costs) < max(self.input_widths) or any(
            not math.isfinite(cost) or cost <= 0 for cost in self.row_costs
        ):
            raise ValueError("AST alternative backend cost law is incomplete")
        if any(
            not math.isfinite(cost) or cost < 0
            for cost in (self.launch_cost, self.context_row_cost)
        ):
            raise ValueError("AST alternative backend costs must be nonnegative")
        if any(
            type(window) is not int
            or window < 1
            or not math.isfinite(cost)
            or cost < 0
            for window, cost in self.context_windows
        ):
            raise ValueError(
                "AST alternative attention windows require finite valid geometry"
            )
        oracle = self.oracle
        if oracle is None or (
            oracle.profile_binding != self.profile_binding
            or oracle.device != self.device
            or oracle.build != self.build
            or oracle.source_revision_token != self.source_revision_token
        ):
            raise ValueError(
                "AST alternative backend numerical oracle does not match its binding"
            )

    def cost(self, length, context=0):
        if type(length) is not int or length < 1 or context < 0:
            raise ValueError("AST alternative backend geometry is invalid")
        if length not in self.input_widths:
            return math.inf
        window_work = sum(
            cost * length * min(context, window)
            for window, cost in self.context_windows
        )
        return (
            self.launch_cost
            + self.row_costs[length - 1]
            + self.context_row_cost * length * context
            + window_work
        )


@dataclass(frozen=True)
class ExecutionProfile:
    binding: str
    device: str
    build: str
    max_lanes: int
    tile_widths: tuple[int, ...]
    terminal_widths: tuple[int, ...]
    packed_min_width: int
    # Relative weight traffic for each actual slab row count, indexed at R-1.
    row_costs: tuple[float, ...]
    launch_cost: float
    context_row_cost: float
    draft_cost: float
    cost_authority: str = "kernel_geometry_prior"
    backend: str = "padded_row_exact"
    # Subset of row_costs consumed by row-independent, adapter-opted stages.
    # State/attention still use the rectangular transaction ABI.
    valid_row_costs: tuple[float, ...] = ()
    reusable_draft_lattice: bool = False
    max_position_rows: int | None = None
    context_windows: tuple[tuple[int, float], ...] = ()
    sequential_draft: bool = False
    alternative_backends: tuple[AlternativeExecutionBackend, ...] = ()

    def __post_init__(self):
        if not self.binding or not self.device or not self.build:
            raise ValueError("AST profile requires source/device/build binding")
        if type(self.max_lanes) is not int or not 1 <= self.max_lanes <= 8:
            raise ValueError("AST profile lane limit exceeds transaction contract")
        if type(self.packed_min_width) is not int or self.packed_min_width < 1:
            raise ValueError("AST packed minimum must be positive")
        if self.max_position_rows is not None and (type(self.max_position_rows) is not int
                                                  or self.max_position_rows < 1):
            raise ValueError("AST kernel row capacity must be positive")
        for widths in (self.tile_widths, self.terminal_widths):
            if not widths or any(type(n) is not int or n < 1 for n in widths):
                raise ValueError("AST requires positive exact width declarations")
        if any(not math.isfinite(n) or n <= 0 for n in self.row_costs):
            raise ValueError("AST row costs must be finite and positive")
        if len(self.row_costs) < self.max_lanes * max(self.terminal_widths):
            raise ValueError("AST cost profile does not cover its physical geometry")
        if self.valid_row_costs and (len(self.valid_row_costs) != len(self.row_costs)
                or any(not math.isfinite(v) or not 0 <= v <= r
                       for v, r in zip(self.valid_row_costs, self.row_costs))):
            raise ValueError("AST compacted cost must be a subset of physical cost")
        if any(not math.isfinite(n) or n < 0 for n in (
            self.launch_cost, self.context_row_cost, self.draft_cost,
        )):
            raise ValueError("AST costs must be finite and nonnegative")
        if any(type(w) is not int or w < 1 or not math.isfinite(c) or c < 0
               for w, c in self.context_windows):
            raise ValueError("AST attention windows must have finite valid geometry")
        names = [backend.backend for backend in self.alternative_backends]
        if len(names) != len(set(names)) or self.backend in names:
            raise ValueError("AST execution backend identities must be distinct")
        for backend in self.alternative_backends:
            if (
                backend.profile_binding != self.binding
                or backend.device != self.device
                or backend.build != self.build
            ):
                raise ValueError(
                    "AST alternative backend does not match the execution profile"
                )

    def cost(self, lengths, contexts=()):
        lengths = tuple(lengths)
        if not lengths or len(lengths) > self.max_lanes:
            raise ValueError("AST physical launch exceeds adapter lane capability")
        if any(type(n) is not int or n < 1 for n in lengths):
            raise ValueError("AST input lengths must be positive integers")
        width = max(lengths)
        if len(lengths) > 1 and width < self.packed_min_width:
            return math.inf
        if contexts and (len(contexts) != len(lengths) or any(c < 0 for c in contexts)):
            raise ValueError("AST context costs require one value per row")
        rows = len(lengths) * width
        # Conservative common capacity: even a compacted launch must fit
        # every declared row consumer. No undeclared chunking/fallback.
        if self.max_position_rows is not None and rows > self.max_position_rows:
            return math.inf
        if rows > len(self.row_costs):
            return math.inf
        context = sum(n * c for n, c in zip(lengths, contexts)) if contexts else 0
        work = self.row_costs[rows - 1]
        if self.valid_row_costs:
            work += self.valid_row_costs[sum(lengths) - 1] - self.valid_row_costs[rows - 1]
        window_work = sum(cost * sum(n * min(c, window) for n, c in zip(lengths, contexts))
                          for window, cost in self.context_windows)
        return self.launch_cost + work + self.context_row_cost * context + window_work

    def alternative(self, backend):
        return next(
            (candidate for candidate in self.alternative_backends
             if candidate.backend == backend),
            None,
        )


@dataclass(frozen=True)
class RequestWork:
    uid: int
    maximum_depth: int
    context_length: int
    branch_limit: int
    acceptance: float = 0.5
    reprobe: bool = False
    # Shared trunk the confidence tree verifies once before branches fan out
    # (branches one and two keep full chains, later branches keep the trunk
    # plus one token: ``confidence_tree_branch_depths``).
    trunk: int = 1
    # Optional adapter-owned, per-position distribution features.  Every
    # value is sum(q_i^2) for the whole proposal law at that position, never
    # q_i(sampled).  The external route publishes only lagged completed-round
    # features, so this round's depth is fixed before its proposal draw.
    collision_probabilities: tuple[float, ...] = ()


@dataclass(frozen=True)
class SourceBudget:
    uid: int
    depth: int
    branches: int
    node_budget: int
    tile: int
    predicted_utility: float
    predicted_cost: float
    reason: str
    survival_priors: tuple[float, ...] = ()
    survival_authority: str = "lagged_acceptance"

    def receipt(self):
        return asdict(self)


@dataclass(frozen=True)
class ReadyRow:
    index: int
    uid: int
    remaining: int
    context_length: int
    acceptance: float = 1.0


@dataclass(frozen=True)
class LaunchPlan:
    indices: tuple[int, ...]
    tile: int
    input_lengths: tuple[int, ...]
    predicted_cost: float
    cohort_cost: float
    maximum_batch_cost: float
    batch_count: int
    backend: str
    context_lengths: tuple[int, ...] = ()
    batch_row_counts: tuple[int, ...] = ()
    expected_cohort_progress: float = 0.0
    acceptance_priors: tuple[float, ...] = ()

    def receipt(self):
        result = asdict(self)
        result.update(schema=SCHEMA, logical_rows=sum(self.input_lengths),
                      physical_rows=len(self.indices) * max(self.input_lengths),
                      padding_rows=len(self.indices) * max(self.input_lengths) - sum(self.input_lengths))
        return result


class ASTExecutionPlanner:
    """Plan source work and actual launches from the same bound cost profile.

Plans never wait for a future request. The oldest ready row always runs in
the next launch; other rows may move to reduce padding and balance costs.
This bounds starvation while allowing live-frontier refill and retirement.
"""

    def __init__(self, profile, *, tile_cap, fixed_tile=True, source_limit=15,
                 minimum_depth=1, cohort_sources=False, ordered_branches=False):
        if not isinstance(profile, ExecutionProfile):
            raise TypeError("AST planner requires a bound execution profile")
        if type(tile_cap) is not int or tile_cap < 1:
            raise ValueError("AST tile cap must be positive")
        if not 1 <= minimum_depth <= 15 or not 1 <= source_limit <= 15:
            raise ValueError("AST source bounds are invalid")
        self.profile = profile
        self.tiles = tuple(t for t in profile.tile_widths if t <= tile_cap and all(
            n in profile.terminal_widths for n in range(2, t + 2)))
        if fixed_tile:
            self.tiles = tuple(t for t in self.tiles if t == tile_cap)
        if not self.tiles:
            raise ValueError("AST has no exact tile under the requested bound")
        self.source_limit = source_limit
        self.minimum_depth = minimum_depth
        self.cohort_sources = bool(cohort_sources)
        self.ordered_branches = bool(ordered_branches)
        self.observed_scale = 1.0
        self.cost_samples = 0
        self._survival = {}
        self._depth_survival = {}
        self._shape_costs = {}
        # Observed committed-token uplift per extra verified row from
        # multi-branch rounds, per request: (uplift sum, extra-row sum, n).
        self._branch_uplift = {}
        # Adapter-opted distribution-only proposal features from the most
        # recent completed proposal round, keyed by request.  These are a
        # causal prior for the next round, never current sampled-token data.
        self._collision_probabilities = {}
        self._collision_feature_observations = 0
        # (physical rows, geometry prior, seconds) of timed launches, and the
        # robust per-row-count seconds curve over prior cost when justified:
        # a tuple of (prior, seconds, rows, samples) points, nondecreasing.
        self._launch_samples = deque(maxlen=LAUNCH_FIT_WINDOW)
        self._launch_curve = None
        self._backend_launches = {}
        # Numerical admission is separate from economics. Each alternative
        # learns only from its own already-synchronized target invocations.
        self._alternative_samples = {}
        self._cold_seconds_per_unit = None

    def acceptance(self, uid, depth=None):
        accepted, tested = (self._survival.get(uid, (0, 0)) if depth is None
                            else self._depth_survival.get((uid, depth), self._survival.get(uid, (0, 0))))
        return (accepted + 1) / (tested + 2)

    def observe_commit(self, uid, accepted, tested, *, censored=False, source_depth=None):
        # Rejection is a failure at only the first rejected position. Tokens
        # beyond that position were not tested. Terminal/stop samples censor
        # the unobserved tail instead of teaching the planner a rejection.
        if tested < 0 or accepted < 0 or accepted > tested:
            raise ValueError("AST acceptance observation is inconsistent")
        tested = min(tested, accepted + (not censored))
        a, n = self._survival.get(uid, (0, 0))
        # Bounded effective history follows changing workloads.
        if n >= 64:
            a, n = a / 2, n / 2
        self._survival[uid] = (a + accepted, n + tested)
        if source_depth is not None:
            key = (uid, source_depth)
            a, n = self._depth_survival.get(key, (0, 0))
            if n >= 64:
                a, n = a / 2, n / 2
            self._depth_survival[key] = (a + accepted, n + tested)

    def observe_branches(self, uid, extra_rows, uplift):
        """Learn what extra branch rows actually bought for one request.

        ``extra_rows`` are target rows beyond the primary chain's own rows and
        ``uplift`` the committed tokens the selected branch added over the
        primary. Only completed multi-branch rounds teach this; a primary-only
        round says nothing about branches. Bounded like the survival history.
        """
        if extra_rows < 0 or uplift < 0:
            raise ValueError("AST branch observation is inconsistent")
        if not extra_rows:
            return
        total, rows, count = self._branch_uplift.get(uid, (0.0, 0.0, 0))
        if count >= 32:
            total, rows, count = total / 2, rows / 2, count // 2
        self._branch_uplift[uid] = (total + uplift, rows + extra_rows, count + 1)

    def branch_uplift(self, uid):
        """(uplift per extra row, samples); None rate before any sample."""
        total, rows, count = self._branch_uplift.get(uid, (0.0, 0.0, 0))
        return (None if not rows else total / rows), count

    def observe_collision_probabilities(self, uid, probabilities):
        """Publish one law-only feature row for a later source decision."""
        values = tuple(float(value) for value in probabilities)
        if (
            not values
            or len(values) > 15
            or any(not math.isfinite(value) or not 0 < value <= 1 for value in values)
        ):
            raise ValueError("AST collision probabilities must be finite in (0, 1]")
        self._collision_probabilities[uid] = values
        self._collision_feature_observations += 1

    def release(self, uid):
        self._survival.pop(uid, None)
        self._branch_uplift.pop(uid, None)
        self._collision_probabilities.pop(uid, None)
        self._depth_survival = {key: value for key, value in self._depth_survival.items() if key[0] != uid}

    def observe_launch(self, plan, seconds):
        # The caller times an existing synchronization, never adds an eval.
        self._backend_launches[plan.backend] = (
            self._backend_launches.get(plan.backend, 0) + 1
        )
        if plan.backend != self.profile.backend:
            candidate = self.profile.alternative(plan.backend)
            if candidate is None or len(plan.input_lengths) != 1:
                return
            if candidate.timing_boundary != "target_forward":
                # This observation ends at logits/features materialization.
                # A deferred successor/replay is not free: preserve its
                # complete offline prior rather than train on a partial time.
                return
            prior = self.declared_backend_cost(
                plan.backend, plan.input_lengths, plan.context_lengths
            )
            if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(prior) or prior <= 0:
                return
            if self._cold_seconds_per_unit is None:
                self._cold_seconds_per_unit = seconds / prior
            samples = self._alternative_samples.setdefault(
                plan.backend, deque(maxlen=LAUNCH_FIT_WINDOW)
            )
            samples.append((
                self._shape_key(
                    plan.input_lengths,
                    plan.context_lengths or (0,) * len(plan.input_lengths),
                    plan.backend,
                ),
                prior, seconds,
            ))
            # Never train the ordinary backend's curve with TensorFold (or
            # any other alternative)'s timings.
            return
        if not math.isfinite(seconds) or seconds <= 0 or plan.predicted_cost <= 0:
            return
        prior = self.declared_backend_cost(
            plan.backend, plan.input_lengths, plan.context_lengths
        )
        scale = seconds / prior
        if self._cold_seconds_per_unit is None:
            self._cold_seconds_per_unit = scale
        self.observed_scale = scale if not self.cost_samples else (
            0.875 * self.observed_scale + 0.125 * scale)
        self.cost_samples += 1
        key = self._shape_key(
            plan.input_lengths, plan.context_lengths, plan.backend
        )
        previous, count = self._shape_costs.get(key, (scale, 0))
        self._shape_costs[key] = (previous * 0.875 + scale * 0.125, count + 1)
        # Per-executor, source-bound and bounded; no cross-artifact cache.
        if len(self._shape_costs) > 128:
            del self._shape_costs[next(iter(self._shape_costs))]
        self._launch_samples.append(
            (len(plan.indices) * max(plan.input_lengths), prior, seconds))
        self._refit_launch_curve()

    def _refit_launch_curve(self):
        """Robust per-row-count seconds curve over the geometry prior.

        The cold prior prices memory work only, so without calibration three
        one-row launches cost what one three-row launch costs. Each physical
        row count with enough timed launches contributes one point (median
        prior, median seconds); pool-adjacent-violators makes the points
        nondecreasing in prior; a curve that collapses to one level is
        discarded. Nothing here reads a tensor size as a time.
        """
        self._launch_curve = None
        samples = self._launch_samples
        if len(samples) < LAUNCH_FIT_MIN_SAMPLES:
            return
        buckets = {}
        for rows, prior, seconds in samples:
            buckets.setdefault(rows, []).append((prior, seconds))
        points = []
        for rows, pairs in sorted(buckets.items()):
            if len(pairs) < LAUNCH_BUCKET_MIN_SAMPLES:
                continue
            priors = sorted(p for p, _ in pairs)
            times = sorted(t for _, t in pairs)
            points.append([priors[len(priors) // 2], times[len(times) // 2], rows, len(pairs)])
        points.sort(key=lambda point: (point[0], point[2]))
        if len(points) < 2 or points[0][0] == points[-1][0]:
            return
        # Pool-adjacent-violators: weighted isotonic regression of seconds on
        # the prior, so a noisier bucket cannot make a wider launch cheaper.
        blocks = []
        for prior, seconds, rows, weight in points:
            blocks.append([prior, seconds, rows, weight, seconds * weight, weight, prior * weight])
            while len(blocks) > 1 and blocks[-2][1] > blocks[-1][1]:
                tail = blocks.pop()
                head = blocks[-1]
                head[4] += tail[4]
                head[5] += tail[5]
                head[6] += tail[6]
                head[1] = head[4] / head[5]
                head[0] = head[6] / head[5]
                head[3] += tail[3]
                head[2] = (head[2], tail[2]) if type(head[2]) is int else head[2] + (tail[2],)
        levels = {round(block[1], 12) for block in blocks}
        if len(levels) < 2:
            return
        self._launch_curve = tuple((block[0], block[1], block[2], block[3]) for block in blocks)

    def _curve_seconds(self, prior):
        curve = self._launch_curve
        if prior <= curve[0][0]:
            # Nothing observed costs less than the cheapest observed launch.
            return curve[0][1]
        for (x0, y0, _r0, _n0), (x1, y1, _r1, _n1) in zip(curve, curve[1:]):
            if prior <= x1:
                return y0 + (y1 - y0) * (prior - x0) / (x1 - x0)
        (x0, y0, _r0, _n0), (x1, y1, _r1, _n1) = curve[-2], curve[-1]
        return y1 + max(0.0, (y1 - y0) / (x1 - x0)) * (prior - x1)

    def launch_curve_receipt(self):
        curve = self._launch_curve
        if curve is None:
            return {"launch_fit_samples": len(self._launch_samples), "launch_curve": None,
                    "fitted_launch_seconds": None, "fitted_seconds_per_cost_unit": None,
                    "fitted_launch_cost_units": None}
        (x0, y0, _r, _n), (x1, y1, _r1, _n1) = curve[0], curve[1]
        floor = max(0.0, y0 - (y1 - y0) / (x1 - x0) * x0)
        (xa, ya, _ra, _na), (xb, yb, _rb, _nb) = curve[-2], curve[-1]
        return {"launch_fit_samples": len(self._launch_samples),
                "launch_curve": [{"prior": p, "seconds": t, "rows": list(r) if type(r) is tuple else r,
                                  "samples": n} for p, t, r, n in curve],
                "fitted_launch_seconds": floor,
                "fitted_seconds_per_cost_unit": max(0.0, (yb - ya) / (xb - xa)),
                "fitted_launch_cost_units": floor / self.observed_scale,
                "launch_estimator": "per_row_count_median_isotonic_v1"}

    @staticmethod
    def _shape_key(lengths, contexts, backend="padded_row_exact"):
        return (
            backend,
            tuple(sorted(zip(lengths, (int(c).bit_length() for c in contexts)))),
        )

    def cost(self, lengths, contexts):
        prior = self.profile.cost(lengths, contexts)
        if not math.isfinite(prior) or not self.cost_samples:
            return prior
        # Everything below is predicted seconds divided by the global
        # seconds-per-unit, i.e. equivalent work units: the robust curve for
        # a shape never launched, the geometry prior when no curve exists.
        baseline = (prior if self._launch_curve is None
                    else self._curve_seconds(prior) / self.observed_scale)
        observed = self._shape_costs.get(
            self._shape_key(lengths, contexts, self.profile.backend)
        )
        if observed is None:
            return baseline
        scale, count = observed
        # Shrink sparse observations toward the baseline. The ratio removes
        # global load/clock drift; only relative shape costs matter.
        weight = count / (count + 4)
        ratio = min(4.0, max(0.25, scale / self.observed_scale))
        return baseline + weight * (prior * ratio - baseline)

    def backend_cost(self, backend, lengths, contexts):
        """Compare backend-local observations in one common cost unit."""

        if backend == self.profile.backend:
            return self.cost(lengths, contexts)
        candidate = self.profile.alternative(backend)
        if candidate is None or len(lengths) != 1:
            return math.inf
        if contexts and len(contexts) != 1:
            raise ValueError("AST alternative backend requires one context")
        prior = candidate.cost(lengths[0], contexts[0] if contexts else 0)
        samples = self._alternative_samples.get(backend)
        if not math.isfinite(prior) or not samples:
            return prior
        scale = median(seconds / cost for _, cost, seconds in samples)
        key = self._shape_key(lengths, contexts or (0,) * len(lengths), backend)
        local = [seconds / cost for shape, cost, seconds in samples if shape == key]
        if local:
            weight = len(local) / (len(local) + 4)
            scale += weight * (median(local) - scale)
        # cost() is expressed in the primary's observed unit. Before it has
        # samples, keep every backend's geometry prior on the first measured
        # backend's common scale, rather than comparing seconds to work units.
        unit = self.observed_scale if self.cost_samples else self._cold_seconds_per_unit
        return prior * scale / unit

    def alternative_timing_receipt(self, backend):
        samples = self._alternative_samples.get(backend, ())
        candidate = self.profile.alternative(backend)
        boundary = candidate.timing_boundary
        return {
            "timing_boundary": boundary,
            "live_timing_withheld": boundary != "target_forward",
            "cost_samples": len(samples),
            "observed_shape_count": len({shape for shape, _, _ in samples}),
            "seconds_per_cost_unit": (
                median(seconds / prior for _, prior, seconds in samples)
                if samples else None
            ),
            "timing_authority": (
                "backend_local_observed_target_invocations_v1"
                if samples else candidate.cost_authority
            ),
        }

    def declared_backend_cost(self, backend, lengths, contexts):
        """Uncalibrated adapter cost used to fit backend-local timings."""

        if backend == self.profile.backend:
            return self.profile.cost(lengths, contexts)
        candidate = self.profile.alternative(backend)
        if candidate is None or len(lengths) != 1:
            return math.inf
        return candidate.cost(lengths[0], contexts[0] if contexts else 0)

    def _best_launch_cost(self, lengths, contexts):
        primary = self.cost(lengths, contexts)
        best = (primary, self.profile.backend)
        if len(lengths) == 1:
            for backend in self.profile.alternative_backends:
                value = self.backend_cost(backend.backend, lengths, contexts)
                # Strictly cheaper: an equality keeps the ordinary/reference
                # executor selected. Cost priors never break a tie toward a
                # specialized backend.
                if value < best[0]:
                    best = (value, backend.backend)
        return best

    @staticmethod
    def _branch_rows(branches, trunk, start, end):
        """Row lengths one tile launch carries for positions [start, end).

        The confidence tree verifies the trunk once. Branch one is the full
        chain (it is the trunk before fan-out), branch two a full chain past
        the trunk, every later branch the trunk plus one token.
        """
        rows = [end - start]
        if branches > 1:
            suffix = end - max(start, trunk)
            if suffix > 0:
                rows.append(suffix)
            if start <= trunk < end:
                rows.extend([1] * (branches - 2))
        return rows

    @staticmethod
    def _prefix_survival(priors, position):
        return math.prod(priors[:position])

    def _round_cost(self, depth, branches, tile, context, survival, trunk=1,
                    cohort_width=1):
        # Later tiles execute only after a surviving prefix. Price each
        # branch's conditional work, not every full path unconditionally,
        # and never charge the shared trunk once per branch.
        trunk = max(1, min(int(trunk), depth - 1)) if branches > 1 else depth
        total = 0.0
        priors = (
            tuple(float(value) for value in survival[:depth])
            if isinstance(survival, (tuple, list))
            else (float(survival),) * depth
        )
        if len(priors) < depth or any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in priors
        ):
            raise ValueError("AST per-position survival prior is invalid")

        def cohort_cost(lengths, contexts):
            # Source cohorts may exceed one target transaction. Price the
            # balanced physical partitions instead of sending an oversized
            # hypothetical launch to the hard-capacity cost law.
            capacity = max(1, self.profile.max_lanes // len(lengths))
            groups = math.ceil(cohort_width / capacity)
            width, extra = divmod(cohort_width, groups)
            return sum(
                self._best_launch_cost(
                    lengths * (width + int(group < extra)),
                    contexts * (width + int(group < extra)),
                )[0]
                for group in range(groups)
            ) / cohort_width

        if self.ordered_branches and branches > 1:
            # The live frontier admits one branch per request, not all
            # siblings in a single launch. The shared trunk is a separate
            # nonterminal transaction; each suffix pays its own launch.
            # Do not optimistically discount later branches before their
            # measured committed-token uplift has justified admission.
            def sequence_cost(start, count, terminal):
                value = 0.0
                for position in range(0, count, tile):
                    end = min(count, position + tile)
                    length = end - position + int(terminal and end == count)
                    target = cohort_cost(
                        [length], [context + start + position],
                    )
                    value += target * self._prefix_survival(
                        priors, start + position
                    )
                return value

            prefix = sequence_cost(0, trunk, False)
            if cohort_width > 1:
                return (
                    prefix + 2 * sequence_cost(trunk, depth - trunk, True)
                    + max(0, branches - 2) * sequence_cost(trunk, 1, True)
                )
            # A singleton uses the shared-prefix executor, which admits all
            # siblings together. It still pays a distinct trunk invocation.
            suffixes = [depth - trunk] * 2 + [1] * max(0, branches - 2)
            for start in range(0, max(suffixes), tile):
                lengths = [
                    min(tile, count - start) + int(count <= start + tile)
                    for count in suffixes if count > start
                ]
                value, _ = self._best_launch_cost(
                    lengths, [context + trunk + start] * len(lengths)
                )
                prefix += value * self._prefix_survival(priors, trunk + start)
            return prefix
        for start in range(0, depth, tile):
            end = min(depth, start + tile)
            rows = self._branch_rows(branches, trunk, start, end)
            terminal = int(end == depth)
            prefix_survival = self._prefix_survival(priors, start)
            target = cohort_cost(
                [n + terminal for n in rows],
                [context + start] * len(rows),
            )
            total += target * (
                1 - (1 - prefix_survival) ** branches)
        return total

    def source_budgets(self, requests):
        requests = tuple(requests)
        if not requests or len({r.uid for r in requests}) != len(requests):
            raise ValueError("AST source plan requires distinct requests")
        # A cohort receives capacity before an individual request can spend it
        # on siblings. Reprobes share the same hard cohort capacity and may
        # explore unknown uplift, but never a provably uneconomic shape.
        fair_rows = max(1, self.profile.max_lanes // len(requests))
        result = []
        options = {}
        live_width = sum(request.maximum_depth > 0 for request in requests)
        for request in requests:
            if (not 1 <= request.branch_limit <= 4 or request.maximum_depth < 0
                    or request.maximum_depth > 15 or request.context_length < 0):
                raise ValueError("AST request bounds are invalid")
            p = request.acceptance
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("AST acceptance prior must be a probability")
            collision = (
                tuple(request.collision_probabilities)
                or self._collision_probabilities.get(request.uid, ())
            )
            if collision and (
                len(collision) > 15
                or any(
                    not math.isfinite(value) or not 0 < value <= 1
                    for value in collision
                )
            ):
                raise ValueError(
                    "AST collision prior must be a bounded probability row"
                )
            if request.maximum_depth == 0:
                result.append(SourceBudget(request.uid, 0, 1, 0, self.tiles[0], 1, 0,
                                           "terminal_primary"))
                continue
            limit = min(request.branch_limit, fair_rows)
            if request.maximum_depth < 2 or self.source_limit < 2:
                limit = 1
            candidates = []
            for depth in range(min(self.minimum_depth, request.maximum_depth),
                               request.maximum_depth + 1):
                # DFlash block length changes its denoising distribution.
                # Do not pool a short block's observations into a longer one.
                depth_p = self.acceptance(request.uid, depth) if (request.uid, depth) in self._depth_survival else p
                priors = (
                    tuple(collision[:depth])
                    + (depth_p,) * max(0, depth - len(collision))
                    if collision
                    else (depth_p,) * depth
                )
                for branches in range(1, limit + 1):
                    if branches > 1 and depth < 2:
                        continue
                    if branches > 1 and depth > self.source_limit:
                        continue
                    # A branch budget is bounded work, not fabricated token
                    # confidence. The union prior stands in until exact
                    # multi-branch commits report what extra rows bought.
                    prefix_survival = [
                        self._prefix_survival(priors, position)
                        for position in range(1, depth + 1)
                    ]
                    primary = 1 + sum(prefix_survival)
                    union = 1 + sum(
                        1 - (1 - value) ** branches for value in prefix_survival
                    )
                    trunk = max(1, min(int(request.trunk), depth - 1)) if branches > 1 else depth
                    extra_rows = (depth - trunk) + max(0, branches - 2) if branches > 1 else 0
                    rate, samples = self.branch_uplift(request.uid)
                    weight = samples / (samples + 4)
                    prior_uplift = union - primary
                    learned = prior_uplift if rate is None else rate * extra_rows
                    utility = primary + prior_uplift + weight * (learned - prior_uplift)
                    if self.ordered_branches:
                        # Correlated sibling proposals are not independent
                        # chances of acceptance. Explore on the existing
                        # reprobe clock, then pay for siblings only from
                        # completed, request-owned uplift evidence.
                        utility = primary + (
                            0.0 if rate is None else
                            min(depth + 1 - primary, rate * extra_rows)
                        )
                    nodes = min(self.source_limit, depth * branches)
                    for tile in self.tiles:
                        width = live_width if self.cohort_sources else 1
                        target = self._round_cost(
                            depth, branches, tile, request.context_length, priors, trunk,
                            cohort_width=width,
                        )
                        passes = 1 + (branches > 1 and not self.profile.reusable_draft_lattice)
                        cost = target + self.profile.draft_cost * passes * (depth if self.profile.sequential_draft else 1)
                        if math.isfinite(cost) and cost > 0:
                            candidates.append((utility / cost, -cost, -nodes, depth,
                                               branches, tile, utility, cost, priors,
                                               target))
            if not candidates:
                raise ValueError("AST request has no feasible exact source plan")
            reprobe = False
            if request.reprobe:
                primary = max(row for row in candidates if row[4] == 1)
                feasible = [
                    row for row in candidates
                    if (row[3] > primary[3] or row[4] > 1)
                    # Even perfect acceptance cannot commit more than the
                    # proposal depth plus its terminal bonus. Do not spend
                    # a periodic probe on a shape that cannot beat the
                    # already available primary, even at that upper bound.
                    and (row[3] + 1) / row[7] > primary[0]
                ]
                if feasible:
                    shape = max((row[3], row[4]) for row in feasible)
                    candidates = [
                        row for row in feasible if (row[3], row[4]) == shape
                    ]
                    reprobe = True
            options[request.uid] = candidates
            _, _, _, depth, branches, tile, utility, cost, priors, _ = max(candidates)
            result.append(SourceBudget(request.uid, depth, branches,
                                       min(self.source_limit, depth * branches), tile,
                                       utility, cost, "bounded_reprobe" if reprobe
                                       else "learned_branch_uplift" if self.branch_uplift(request.uid)[1]
                                       else "hardware_cohort_cost_prior",
                                       tuple(priors),
                                       "proposal_collision_probability_lagged_v1"
                                       if collision else "lagged_acceptance"))
        if self.cohort_sources and live_width > 1:
            # Depth is part of the draft-call shape. Independent per-request
            # choices fragment an otherwise batchable cohort. Choose one
            # cohort depth, clipped only by each request's terminal bound,
            # while preserving per-request acceptance and branch evidence.
            choices = []
            for depth in range(self.minimum_depth, max(r.maximum_depth for r in requests) + 1):
                selected = []
                for request in requests:
                    if not request.maximum_depth:
                        continue
                    at_depth = [
                        row for row in options[request.uid]
                        if row[3] == min(depth, request.maximum_depth)
                    ]
                    if not at_depth:
                        break
                    selected.append(max(at_depth))
                else:
                    # Terminal clipping can leave distinct draft shapes;
                    # each distinct depth still pays a draft invocation.
                    draft_groups = {}
                    for row in selected:
                        draft_groups[row[3]] = max(
                            draft_groups.get(row[3], 0.0), row[7] - row[9]
                        )
                    cost = sum(row[9] for row in selected) + sum(draft_groups.values())
                    utility = sum(row[6] for row in selected)
                    choices.append((utility / cost, -cost, depth, selected))
            if choices:
                selected = iter(max(choices, key=lambda row: row[:3])[3])
                for index, plan in enumerate(result):
                    if not plan.depth:
                        continue
                    _, _, _, depth, branches, tile, utility, cost, priors, _ = next(selected)
                    result[index] = SourceBudget(
                        plan.uid, depth, branches,
                        min(self.source_limit, depth * branches), tile,
                        utility, cost, (
                            "bounded_reprobe" if plan.reason == "bounded_reprobe"
                            else "cohort_batched_source"
                        ),
                        tuple(priors), plan.survival_authority,
                    )
        return tuple(result)

    def plan_launch(self, ready, *, terminal, max_lanes):
        ready = tuple(ready)
        if not ready or len(ready) > MAX_READY_ROWS or len({r.index for r in ready}) != len(ready):
            raise ValueError("AST ready frontier is invalid")
        if any(r.remaining < 0 or (r.remaining == 0 and not terminal)
               or r.context_length < 0 for r in ready):
            raise ValueError("AST ready row geometry is invalid")
        if any(not math.isfinite(r.acceptance) or not 0 <= r.acceptance <= 1 for r in ready):
            raise ValueError("AST ready survival priors must be probabilities")
        cap = min(max_lanes, self.profile.max_lanes)
        if cap < 1:
            raise ValueError("AST launch capacity must be positive")
        plans = []
        for tile in self.tiles:
            lengths = {r.index: min(tile, r.remaining) + int(terminal and r.remaining <= tile)
                       for r in ready}
            rows = sorted(ready, key=lambda r: (lengths[r.index], r.context_length, r.index))

            def cost(group, lengths=lengths):
                return self._best_launch_cost(
                    [lengths[r.index] for r in group],
                    [r.context_length for r in group],
                )[0]

            @lru_cache(None)
            def partition(start, rows=tuple(rows), cost=cost):
                if start == len(rows):
                    return 0.0, 0.0, 0, ()
                choices = []
                for end in range(start + 1, min(len(rows), start + cap) + 1):
                    group = tuple(rows[start:end])
                    value = cost(group)
                    if not math.isfinite(value):
                        continue
                    tail, maximum, occupancy, groups = partition(end)
                    if not math.isfinite(tail):
                        continue
                    choices.append((value + tail, max(value, maximum),
                                    max(len(group), occupancy), (group,) + groups))
                if not choices:
                    return math.inf, math.inf, 0, ()
                # Equal predicted work is not evidence that four serial B1
                # launches beat one B4 launch. Preserve batching before
                # minimizing an individual launch's cost; genuinely cheaper
                # splits still win on total cost and hard caps are unchanged.
                return min(choices, key=lambda x: (x[0], len(x[3]), x[1], x[2]))

            total, maximum, _, groups = partition(0)
            if not math.isfinite(total):
                continue
            groups = [list(group) for group in groups]
            # Equal-length rows have no padding tradeoff. Redistribute their
            # context work across the chosen capacities, largest first, to
            # avoid one batch containing every expensive long-context row.
            if len(groups) > 1 and len(set(lengths.values())) == 1:
                capacities = [len(group) for group in groups]
                balanced = [[] for _ in groups]
                for row in sorted(rows, key=lambda r: (-r.context_length, r.index)):
                    eligible = [i for i, group in enumerate(balanced) if len(group) < capacities[i]]
                    slot = min(eligible, key=lambda i: (
                        cost(balanced[i] + [row]), len(balanced[i]), i))
                    balanced[slot].append(row)
                if sum(map(cost, balanced)) <= total + 1e-9:
                    groups = balanced
                    maximum = max(map(cost, groups))
            # Replan after each actual launch/refill. The oldest member must
            # run now, so repeated packing cannot indefinitely defer a row.
            group = next(g for g in groups if ready[0] in g)
            order = {r.index: i for i, r in enumerate(ready)}
            group.sort(key=lambda r: order[r.index])
            # Raw row throughput rewards speculatively computing a suffix
            # whose predecessor is likely to reject. Price useful decision
            # progress instead: the first target law is always needed, later
            # laws only after survival. Siblings share one request's utility
            # rather than manufacturing several guaranteed first tokens.
            by_request = {}
            for row in rows:
                by_request.setdefault(row.uid, []).append(row)
            useful = sum(
                1 - math.prod(1 - row.acceptance ** position
                              for row in group if lengths[row.index] > position)
                for group in by_request.values()
                for position in range(max(lengths[row.index] for row in group)))
            group_cost, backend = self._best_launch_cost(
                [lengths[r.index] for r in group],
                [r.context_length for r in group],
            )
            plans.append((total / useful, maximum, -tile, LaunchPlan(
                tuple(r.index for r in group), tile,
                tuple(lengths[r.index] for r in group), group_cost, total,
                maximum, len(groups), backend,
                tuple(r.context_length for r in group), tuple(len(g) for g in groups),
                useful, tuple(r.acceptance for r in group))))
        if not plans:
            raise ValueError("AST ready frontier has no feasible exact physical plan")
        return min(plans, key=lambda p: p[:3])[-1]

    def receipt(self):
        return {"schema": SCHEMA, "binding": self.profile.binding,
                "device": self.profile.device, "build": self.profile.build,
                "backend": self.profile.backend, "tiles": self.tiles,
                "max_lanes": self.profile.max_lanes,
                "max_position_rows": self.profile.max_position_rows,
                "packed_min_width": self.profile.packed_min_width,
                "terminal_input_widths": self.profile.terminal_widths,
                "source_limit": self.source_limit,
                "reusable_draft_lattice": self.profile.reusable_draft_lattice,
                "source_depth_authority": (
                    "cohort_batched_cost" if self.cohort_sources else "request_cost"
                ),
                "branch_execution": (
                    "cohort_ordered_singleton_shared_prefix"
                    if self.ordered_branches else "parallel"
                ),
                "cost_authority": self.profile.cost_authority,
                "cost_samples": self.cost_samples,
                "observed_shape_count": len(self._shape_costs),
                "seconds_per_cost_unit": self.observed_scale if self.cost_samples else None,
                **self.launch_curve_receipt(),
                "branch_authority": "planner_learned_uplift_v1",
                "branch_feedback_requests": len(self._branch_uplift),
                "collision_feature": "sum_q_squared_per_position_v1",
                "collision_feature_requests": len(self._collision_probabilities),
                "collision_feature_observations": self._collision_feature_observations,
                "collision_feature_phase": "prior_completed_round_before_current_draw",
                "alternative_backends": [
                    {
                        "backend": backend.backend,
                        "maximum_sessions": backend.maximum_sessions,
                        "input_widths": backend.input_widths,
                        "cost_authority": backend.cost_authority,
                        "source_revision_token": backend.source_revision_token,
                        "numerical_oracle": backend.oracle.evidence,
                        "qualified": False,
                        "selected": True,
                        "observed_used": self._backend_launches.get(
                            backend.backend, 0
                        )
                        > 0,
                        "launches": self._backend_launches.get(
                            backend.backend, 0
                        ),
                        **self.alternative_timing_receipt(backend.backend),
                    }
                    for backend in self.profile.alternative_backends
                ],
                "qualified": False}
