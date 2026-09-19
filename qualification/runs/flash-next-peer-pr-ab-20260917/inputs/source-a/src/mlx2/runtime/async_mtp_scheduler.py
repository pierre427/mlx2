"""CPU-only policy and replay prototype for request-local MTP scheduling.

This module is deliberately disconnected from the serving runtime.  It models
draft/verify economics from recorded acceptance-prefix observations; it does
not implement ASPIRE's mixed forward or make any route selectable.  Design
provenance is recorded in ``provenance/aspire-scheduler-prototype.json``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True)
class CostModel:
    """ASPIRE-style linear forward-time model in arbitrary cost units."""

    beta_model: float
    beta_mlp: float
    beta_attn: float

    def __post_init__(self) -> None:
        if self.beta_model < 0 or self.beta_mlp < 0 or self.beta_attn < 0:
            raise ValueError("cost coefficients must be non-negative")
        if self.beta_model + self.beta_mlp + self.beta_attn == 0:
            raise ValueError("at least one cost coefficient must be positive")

    def draft_cost(self, context: RequestContext) -> float:
        return (
            self.beta_model
            + self.beta_mlp * context.batch_size
            + self.beta_attn
            * context.batch_size
            * context.sparse_context_length
        )

    def verify_cost(self, context: RequestContext, draft_length: int) -> float:
        if draft_length < 0:
            raise ValueError("draft_length must be non-negative")
        return (
            self.beta_model
            + self.beta_mlp * context.batch_size * (draft_length + 1)
            + self.beta_attn * context.batch_size * context.context_length
        )


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    context_length: int
    sparse_context_length: int
    batch_size: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if not 0 < self.sparse_context_length <= self.context_length:
            raise ValueError(
                "sparse_context_length must be in [1, context_length]"
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class TraceRecord:
    """One counterfactual verification opportunity.

    ``accepted_prefix`` is the number of consecutive correct draft tokens when
    probing up to ``max_depth``.  Replaying a shorter choice therefore accepts
    ``min(accepted_prefix, chosen_depth)``.  This is an offline approximation;
    it does not reproduce policy-dependent model state trajectories.
    """

    request_id: str
    round_index: int
    context_length: int
    sparse_context_length: int
    batch_size: int
    max_depth: int
    accepted_prefix: int

    def __post_init__(self) -> None:
        RequestContext(
            self.request_id,
            self.context_length,
            self.sparse_context_length,
            self.batch_size,
        )
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative")
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if not 0 <= self.accepted_prefix <= self.max_depth:
            raise ValueError("accepted_prefix must be in [0, max_depth]")

    @property
    def context(self) -> RequestContext:
        return RequestContext(
            self.request_id,
            self.context_length,
            self.sparse_context_length,
            self.batch_size,
        )


class DraftPolicy(Protocol):
    def choose_depth(self, context: RequestContext) -> int: ...

    def observe(self, drafted: int, accepted: int) -> None: ...


@dataclass
class FixedDepthPolicy:
    depth: int = 2

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError("depth must be non-negative")

    def choose_depth(self, context: RequestContext) -> int:
        del context
        return self.depth

    def observe(self, drafted: int, accepted: int) -> None:
        _validate_observation(drafted, accepted)


@dataclass
class FeedbackDepthPolicy:
    """Small FSM baseline: grow after full acceptance, shrink otherwise."""

    initial_depth: int = 2
    minimum_depth: int = 0
    maximum_depth: int = 8

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_depth <= self.initial_depth <= self.maximum_depth:
            raise ValueError("expected minimum <= initial <= maximum depth")
        self._depth = self.initial_depth

    def choose_depth(self, context: RequestContext) -> int:
        del context
        return self._depth

    def observe(self, drafted: int, accepted: int) -> None:
        _validate_observation(drafted, accepted)
        if drafted > 0 and accepted == drafted:
            self._depth = min(self.maximum_depth, self._depth + 1)
        else:
            self._depth = max(self.minimum_depth, self._depth - 1)


@dataclass
class AspireDepthPolicy:
    """Request-local acceptance/cost policy inspired by ASPIRE section 4.2.

    The target is recomputed while walking the hypothetical draft depth, as it
    would be at successive scheduler decisions.  This only chooses a depth; it
    does not provide the mixed-forward execution needed to realize it online.
    """

    cost_model: CostModel
    maximum_depth: int = 8
    initial_depth: int = 3
    alpha_prior: float = 0.9
    smoothing: float = 0.8

    def __post_init__(self) -> None:
        if self.maximum_depth < 0:
            raise ValueError("maximum_depth must be non-negative")
        if not 0 <= self.initial_depth <= self.maximum_depth:
            raise ValueError("initial_depth must be in [0, maximum_depth]")
        if not 0 <= self.alpha_prior <= 1:
            raise ValueError("alpha_prior must be in [0, 1]")
        if not 0 <= self.smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.alpha = self.alpha_prior
        self.observations = 0

    def choose_depth(self, context: RequestContext) -> int:
        if self.observations == 0:
            return self.initial_depth
        draft_cost = self.cost_model.draft_cost(context)
        for current_depth in range(self.maximum_depth + 1):
            verify_cost = self.cost_model.verify_cost(context, current_depth)
            target = _target_depth(
                alpha=self.alpha,
                cost_ratio=draft_cost / verify_cost,
                maximum_depth=self.maximum_depth,
            )
            if current_depth >= target or current_depth == self.maximum_depth:
                return current_depth
        raise AssertionError("unreachable depth selection")

    def observe(self, drafted: int, accepted: int) -> None:
        _validate_observation(drafted, accepted)
        if drafted == 0:
            return
        estimate = accepted / min(accepted + 1, drafted)
        self.alpha = self.smoothing * self.alpha + (1 - self.smoothing) * estimate
        self.observations += 1


def _validate_observation(drafted: int, accepted: int) -> None:
    if drafted < 0 or not 0 <= accepted <= drafted:
        raise ValueError("expected 0 <= accepted <= drafted")


def _expected_tokens(alpha: float, depth: int) -> float:
    if alpha == 1:
        return float(depth + 1)
    return (1 - alpha ** (depth + 1)) / (1 - alpha)


def _target_depth(alpha: float, cost_ratio: float, maximum_depth: int) -> int:
    if cost_ratio < 0:
        raise ValueError("cost_ratio must be non-negative")
    best_depth = 0
    best_score = -1.0
    for depth in range(maximum_depth + 1):
        score = _expected_tokens(alpha, depth) / (1 + cost_ratio * depth)
        if score > best_score:
            best_score = score
            best_depth = depth
    return best_depth


@dataclass(frozen=True)
class ReplayResult:
    policy: str
    records: int
    drafted_tokens: int
    accepted_tokens: int
    committed_tokens: int
    normalized_cost: float
    zero_depth_rounds: int
    full_accept_rounds: int

    @property
    def throughput(self) -> float:
        return self.committed_tokens / self.normalized_cost

    @property
    def acceptance_rate(self) -> float:
        if self.drafted_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.drafted_tokens

    @property
    def mean_depth(self) -> float:
        if self.records == 0:
            return 0.0
        return self.drafted_tokens / self.records

    def as_dict(self) -> dict[str, int | float | str]:
        result = asdict(self)
        result.update(
            throughput=self.throughput,
            acceptance_rate=self.acceptance_rate,
            mean_depth=self.mean_depth,
        )
        return result


PolicyFactory = Callable[[], DraftPolicy]


def replay_trace(
    records: Iterable[TraceRecord],
    *,
    policy_name: str,
    policy_factory: PolicyFactory,
    cost_model: CostModel,
) -> ReplayResult:
    """Replay independent counterfactual rounds with request-local policy state."""

    policies: dict[str, DraftPolicy] = {}
    record_count = drafted_tokens = accepted_tokens = committed_tokens = 0
    zero_depth_rounds = full_accept_rounds = 0
    normalized_cost = 0.0
    last_round: dict[str, int] = {}

    for record in records:
        previous = last_round.get(record.request_id, -1)
        if record.round_index <= previous:
            raise ValueError("round_index must increase within each request")
        last_round[record.request_id] = record.round_index
        policy = policies.setdefault(record.request_id, policy_factory())
        depth = policy.choose_depth(record.context)
        if not 0 <= depth <= record.max_depth:
            raise ValueError(
                f"policy chose depth {depth} outside trace cap {record.max_depth}"
            )
        accepted = min(record.accepted_prefix, depth)
        policy.observe(depth, accepted)

        record_count += 1
        drafted_tokens += depth
        accepted_tokens += accepted
        committed_tokens += accepted + 1
        zero_depth_rounds += depth == 0
        full_accept_rounds += accepted == depth
        normalized_cost += (
            depth * cost_model.draft_cost(record.context)
            + cost_model.verify_cost(record.context, depth)
        )

    return ReplayResult(
        policy=policy_name,
        records=record_count,
        drafted_tokens=drafted_tokens,
        accepted_tokens=accepted_tokens,
        committed_tokens=committed_tokens,
        normalized_cost=normalized_cost,
        zero_depth_rounds=zero_depth_rounds,
        full_accept_rounds=full_accept_rounds,
    )
