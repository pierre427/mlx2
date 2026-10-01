"""Default-off, host-side acceptance calibration and lagged cohort budgeting.

The IRLS estimator is adapted from vLLM (Apache-2.0); see provenance. Budget
selection is original: it only reads earlier committed rounds, before current
proposal draws. Costs are supplied by the caller, not inferred performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def proposal_feature(law):
    """Logit(max q) of the actual normalized proposal law, never q(sampled)."""
    values = np.asarray(law, dtype=np.float64)
    if (
        values.ndim != 1
        or not values.size
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
        or not np.isclose(values.sum(), 1.0, atol=1e-6)
    ):
        raise ValueError(
            "adaptive verification requires a finite normalized proposal law"
        )
    peak = float(values.max())
    if peak >= 1.0:
        return 40.0
    return max(-40.0, min(40.0, math.log(peak) - math.log1p(-peak)))


def _sigmoid(value):
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


class OnlineAcceptanceEstimator:
    """Shared nonnegative slope and per-position intercepts; censored labels.

    Only accepted proposals and the first rejected proposal have labels. Later
    positions contribute feature history but never rejection labels. Damped
    Newton updates use bounded O(depth) sufficient statistics.
    """

    def __init__(self, depth, *, refit_interval=100, damping=50.0, l2=1e-3):
        self.depth = int(depth)
        self.refit_interval = int(refit_interval)
        self.damping = float(damping)
        self.l2 = float(l2)
        if (
            self.depth < 1
            or self.refit_interval < 1
            or not np.isfinite(self.damping)
            or not np.isfinite(self.l2)
            or self.damping < 0
            or self.l2 <= 0
        ):
            raise ValueError("invalid acceptance estimator configuration")
        self.slope = 0.5
        self.intercepts = np.full(self.depth, -0.2, dtype=np.float64)
        self.info = np.zeros((self.depth, 3), dtype=np.float64)
        self.grad = np.zeros((self.depth, 2), dtype=np.float64)
        self.counts = np.zeros(self.depth, dtype=np.int64)
        self.observed_counts = np.zeros(self.depth, dtype=np.int64)
        self.feature_counts = np.zeros(self.depth, dtype=np.int64)
        self.feature_means = np.zeros(self.depth, dtype=np.float64)
        self.rounds = 0
        self.refits = 0

    def predict(self, features):
        features = np.asarray(features, dtype=np.float64)
        if (
            features.ndim != 1
            or len(features) > self.depth
            or not np.all(np.isfinite(features))
            or not np.isfinite(self.slope)
            or self.slope < 0
            or not np.all(np.isfinite(self.intercepts))
        ):
            raise ValueError("nonfinite or invalid acceptance estimator state")
        return _sigmoid(self.slope * features + self.intercepts[: len(features)])

    def observe(self, features, accepted, *, rejected):
        features = np.asarray(features, dtype=np.float64)
        if (
            features.ndim != 1
            or len(features) > self.depth
            or not np.all(np.isfinite(features))
            or not 0 <= accepted <= len(features)
            or (rejected and accepted == len(features))
        ):
            raise ValueError("invalid censored acceptance observation")
        predictions = self.predict(features)
        size = len(features)
        self.feature_counts[:size] += 1
        self.feature_means[:size] += (
            features - self.feature_means[:size]
        ) / self.feature_counts[:size]
        observed = int(accepted) + int(bool(rejected))
        x, pred = features[:observed], predictions[:observed]
        labels = np.arange(observed) < accepted
        weight, residual = pred * (1 - pred), labels.astype(float) - pred
        self.info[:observed] += np.stack((weight * x * x, weight * x, weight), axis=1)
        self.grad[:observed] += np.stack((residual * x, residual), axis=1)
        self.counts[:observed] += 1
        self.observed_counts[:observed] += 1

    def finish_round(self):
        self.rounds += 1
        if self.rounds % self.refit_interval == 0:
            self.refit()

    def refit(self):
        a = float(self.info[:, 0].sum()) + self.l2
        b = self.info[:, 1]
        c = self.info[:, 2] + self.l2
        g0, g1 = float(self.grad[:, 0].sum()), self.grad[:, 1]
        denominator = a - float((b * b / c).sum())
        total = int(self.counts.sum())
        step = (g0 - float((b * g1 / c).sum())) / denominator
        step *= total / (total + self.damping) if total else 0.0
        if not np.isfinite(step):
            step = 0.0
        step = max(-self.slope, float(step))
        bias_step = (g1 - b * step) / c
        bias_step *= self.counts / (self.counts + self.damping + (self.counts == 0))
        bias_step = np.where(np.isfinite(bias_step), bias_step, 0.0)
        # Bound the change of every logit over the clamped feature domain.
        # Saturated one-hot features otherwise produce enormous Newton steps
        # from almost-zero curvature and can alternate between certainty poles.
        maximum_change = float(np.max(40.0 * abs(step) + np.abs(bias_step)))
        scale = min(1.0, 2.0 / maximum_change) if maximum_change else 1.0
        self.slope += float(step) * scale
        self.intercepts += bias_step * scale
        self.info.fill(0)
        self.grad.fill(0)
        self.counts.fill(0)
        self.refits += 1


@dataclass(frozen=True)
class AdaptiveVerificationPolicy:
    """Explicit cost model in common units, indexed by proposal depth incl. zero.

    ``verification_costs[k]`` measures an entire cohort verifying k proposals
    plus a bonus row. This table is only valid for the configured cohort width;
    per-request mode can supply additional tables in verification_costs_by_cohort.
    The draft cost is a conservative upper bound per proposal group. Missing
    group costs use the fixed-depth backstop, not a guessed speedup.
    """

    verification_costs: tuple = ()
    draft_cost: float = 0.0
    cohort_size: int = 1
    min_observations: int = 32
    full_depth_interval: int = 32
    min_gain: float = 0.05
    minimum_depth: int = 1
    refit_interval: int = 100
    mode: str = "cohort"
    verification_costs_by_cohort: tuple = ()
    continuation_costs: tuple = ()

    @classmethod
    def from_value(cls, value, depth):
        if value is None or value is False:
            return None
        if not isinstance(value, dict):
            raise ValueError("adaptive_verification must be a cost-model dictionary")  # noqa: TRY004

        def table(values):
            if (
                not isinstance(values, (list, tuple))
                or len(values) != depth + 1
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                    for v in values
                )
            ):
                raise ValueError("invalid adaptive verification costs")
            return tuple(float(v) for v in values)

        try:
            fields = dict(value)
            pool_mapping = fields.pop("continuation_costs", {})
            if not isinstance(pool_mapping, dict):
                raise ValueError(  # noqa: TRY004
                    "continuation_costs must map physical path widths to depth costs"
                )
            pool_converted = {}
            for width, values in pool_mapping.items():
                if type(width) is str and width.isdecimal():
                    width = int(width)
                if (
                    type(width) is not int
                    or not 1 <= width <= 15
                    or width in pool_converted
                ):
                    raise ValueError("invalid continuation physical path width")
                pool_converted[width] = table(values)
            fields["continuation_costs"] = tuple(sorted(pool_converted.items()))
            if "verification_costs" in fields:
                fields["verification_costs"] = table(fields["verification_costs"])
            elif not pool_converted:
                raise ValueError("adaptive verification requires measured cost tables")
            mapping = fields.pop("verification_costs_by_cohort", {})
            if not isinstance(mapping, dict):
                raise ValueError("verification_costs_by_cohort must be a dictionary")  # noqa: TRY004
            converted = {}
            for width, values in mapping.items():
                if type(width) is str and width.isdecimal():
                    width = int(width)
                if type(width) is not int or width < 1 or width in converted:
                    raise ValueError("invalid cost model cohort size")
                converted[width] = table(values)
            fields["verification_costs_by_cohort"] = tuple(sorted(converted.items()))
            policy = cls(**fields)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid adaptive verification cost model") from error
        if (
            type(policy.draft_cost) not in (int, float)
            or not math.isfinite(policy.draft_cost)
            or policy.draft_cost < 0
            or type(policy.min_gain) not in (int, float)
            or not math.isfinite(policy.min_gain)
            or not 0 <= policy.min_gain < 1
        ):
            raise ValueError("invalid adaptive verification costs or gain")
        for name in (
            "cohort_size",
            "min_observations",
            "full_depth_interval",
            "minimum_depth",
            "refit_interval",
        ):
            number = getattr(policy, name)
            if type(number) is not int or number < 1:
                raise ValueError(
                    f"adaptive verification {name} must be a positive integer"
                )
        if policy.minimum_depth > depth or policy.mode not in ("cohort", "per_request"):
            raise ValueError("invalid adaptive verification depth or mode")
        if (
            policy.verification_costs
            and policy.cohort_size in converted
            and converted[policy.cohort_size] != policy.verification_costs
        ):
            raise ValueError("conflicting adaptive verification cohort cost tables")
        return policy

    def costs(self, cohort_size):
        if cohort_size == self.cohort_size:
            return self.verification_costs or None
        return dict(self.verification_costs_by_cohort).get(cohort_size)

    def choose_continuation_shape(
        self, maximum_width, maximum_depth, coverage, rounds, path_lengths=None
    ):
        """Costs bind one request's physical (path width,depth+bonus) forward.

        Historical top-W prefix coverage is conditional on reached positions.
        Missing labels use the declared Beta(1,1) prior. No current target draw,
        synthetic joint q or chain-cohort timing participates in admission.
        """
        full = (maximum_width, maximum_depth)
        tables = dict(self.continuation_costs)
        if (
            maximum_width not in tables
            or rounds % self.full_depth_interval == 0
            or maximum_depth < self.minimum_depth
        ):
            return full

        def progress(width, depth):
            survival, expected = 1.0, 1.0
            for position in range(depth):
                yes, no = coverage.get((width, position), (0, 0))
                if type(yes) is not int or type(no) is not int or min(yes, no) < 0:
                    raise ValueError("invalid continuation coverage observations")
                survival *= (yes + 1) / (yes + no + 2)
                expected += survival
            return expected

        base_yes, base_no = coverage.get((maximum_width, 0), (0, 0))
        if base_yes + base_no < self.min_observations:
            return full
        baseline = (self.draft_cost + tables[maximum_width][maximum_depth]) / progress(
            *full
        )
        choices = [(baseline, -maximum_width, -maximum_depth, full)]
        for width, costs in tables.items():
            if width > maximum_width:
                continue
            yes, no = coverage.get((width, 0), (0, 0))
            if yes + no < self.min_observations:
                continue
            bound = (
                maximum_depth
                if path_lengths is None
                else min(maximum_depth, max(path_lengths[:width]))
            )
            for depth in range(self.minimum_depth, bound + 1):
                cost = (self.draft_cost + costs[depth]) / progress(width, depth)
                choices.append((cost, -width, -depth, (width, depth)))
        cost, _, _, chosen = min(choices)
        return chosen if cost < baseline * (1 - self.min_gain) else full

    def choose_from_features(self, estimator, maximum, features, cohort_size=1):
        costs = self.costs(cohort_size)
        if (
            maximum <= self.minimum_depth
            or costs is None
            or features is None
            or len(features) < maximum
            or estimator.observed_counts.sum() < self.min_observations
            or estimator.rounds % self.full_depth_interval == 0
        ):
            return maximum
        try:
            probabilities = estimator.predict(features[:maximum])
        except ValueError:
            return maximum
        expected = 1.0 + np.cumsum(np.cumprod(probabilities))
        depths = range(self.minimum_depth, maximum + 1)
        scores = {
            k: float(expected[k - 1]) / (self.draft_cost + costs[k]) for k in depths
        }
        best = max(depths, key=lambda k: (scores[k], k))
        return best if scores[best] > scores[maximum] * (1 + self.min_gain) else maximum

    def request_depths(self, estimator, maxima, features):
        """Use valid group costs to beat the original fixed-depth cohort.

        The estimator is a scheduling surrogate. All decisions must precede
        stochastic proposal draws; deterministic proposal hooks can provide
        current confidence because their full block uses no proposal randomness.
        """
        baseline = min(maxima, default=0)
        fallback = [baseline] * len(maxima)
        full_costs, singleton_costs = self.costs(len(maxima)), self.costs(1)
        if (
            full_costs is None
            or singleton_costs is None
            or baseline < self.minimum_depth
            or estimator.observed_counts.sum() < self.min_observations
            or estimator.rounds % self.full_depth_interval == 0
            or any(
                f is None or len(f) < maximum for f, maximum in zip(features, maxima)
            )
        ):
            return fallback
        try:
            expected = [
                1.0 + np.cumsum(np.cumprod(estimator.predict(f[:maximum])))
                for f, maximum in zip(features, maxima)
            ]
        except ValueError:
            return fallback
        chosen = [
            self.choose_from_features(estimator, maximum, f)
            for maximum, f in zip(maxima, features)
        ]
        grouped = {}
        for row, count in enumerate(chosen):
            grouped.setdefault(count, []).append(row)
        # Charge the configured draft upper bound for each distinct proposal
        # depth; stochastic drafting may require separate forwards. This is
        # conservative for deterministic full-block drafters that share work.
        cost = float(self.draft_cost) * max(1, len(grouped))
        for count, indices in grouped.items():
            left = len(indices)
            while left:
                width = max(
                    size for size in range(1, left + 1) if self.costs(size) is not None
                )
                cost += self.costs(width)[count]
                left -= width
        proposed_score = (
            sum(e[k - 1] if k else 1.0 for e, k in zip(expected, chosen)) / cost
        )
        baseline_score = sum(e[baseline - 1] for e in expected) / (
            self.draft_cost + full_costs[baseline]
        )
        return (
            chosen
            if proposed_score > baseline_score * (1 + self.min_gain)
            else fallback
        )

    def choose_depth(self, estimator, maximum, cohort_size):
        maximum = int(maximum)
        if (
            maximum <= self.minimum_depth
            or cohort_size != self.cohort_size
            or estimator.observed_counts.sum() < self.min_observations
            or np.any(estimator.feature_counts[:maximum] == 0)
            or estimator.rounds % self.full_depth_interval == 0
        ):
            return maximum
        try:
            probabilities = estimator.predict(estimator.feature_means[:maximum])
        except ValueError:
            return maximum
        expected = 1.0 + np.cumsum(np.cumprod(probabilities))
        depths = range(self.minimum_depth, maximum + 1)
        scores = {
            k: float(expected[k - 1]) / (self.draft_cost + self.verification_costs[k])
            for k in depths
        }
        best = max(depths, key=lambda k: (scores[k], k))
        return best if scores[best] > scores[maximum] * (1 + self.min_gain) else maximum
