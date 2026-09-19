"""Host-only adaptive policies for continuous serving.

These controllers intentionally own no model or cache state.  Callers sample
them only at closed scheduler/verification boundaries, then apply one decision
to the whole physical cohort.  This keeps policy experimentation out of the
transactional cache machinery.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_COUNTER_MAX = (1 << 63) - 1


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _bump(counters: dict[str, int], key: str, amount: int = 1) -> None:
    counters[key] = min(_COUNTER_MAX, int(counters.get(key, 0)) + int(amount))


@dataclass
class DecodeTimeFairness:
    """Bound prefill stalls and repay their wall-time cost with decode work."""

    enabled: bool = False
    fair_share: float = 0.5
    stall_target_ms: float = 500.0
    fallback_cap: int = 512
    floor: int = 64
    grid: int = 64
    debt_seconds: float = 0.0
    best_prefill_tokens_per_second: float = 0.0
    counters: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.fair_share) or self.fair_share < 0:
            raise ValueError("decode fair_share must be finite and nonnegative")
        if not math.isfinite(self.stall_target_ms) or self.stall_target_ms <= 0:
            raise ValueError("decode stall_target_ms must be finite and positive")
        if min(self.fallback_cap, self.floor, self.grid) < 1:
            raise ValueError("decode fairness token limits must be positive")
        for name in (
            "prefill_chunks",
            "debt_deferrals",
            "debt_repayments",
            "cap_clamps",
        ):
            self.counters.setdefault(name, 0)

    def cap(self, configured: int, *, contended: bool) -> int:
        configured = max(1, int(configured))
        if not self.enabled or not contended:
            return configured
        if self.best_prefill_tokens_per_second > 0:
            value = int(
                self.best_prefill_tokens_per_second * self.stall_target_ms / 1000.0
            )
            value = max(self.grid, (value // self.grid) * self.grid)
        else:
            value = self.fallback_cap
        value = max(self.floor, min(configured, value))
        if value < configured:
            _bump(self.counters, "cap_clamps")
        return value

    def may_prefill(self, *, contended: bool) -> bool:
        if not self.enabled or not contended:
            if not contended:
                self.debt_seconds = 0.0
            return True
        if self.debt_seconds > 0:
            _bump(self.counters, "debt_deferrals")
            return False
        return True

    def observe_decode(self, seconds: float) -> None:
        paid = min(self.debt_seconds, max(0.0, float(seconds)))
        self.debt_seconds -= paid
        if self.debt_seconds < 1e-12:
            self.debt_seconds = 0.0
        if paid:
            _bump(self.counters, "debt_repayments")

    def observe_prefill(self, tokens: int, seconds: float, *, contended: bool) -> None:
        seconds = max(0.0, float(seconds))
        tokens = max(0, int(tokens))
        if tokens and seconds:
            # Contention only depresses a sample; a running maximum avoids a
            # feedback loop where smaller measured chunks cause smaller caps.
            self.best_prefill_tokens_per_second = max(
                self.best_prefill_tokens_per_second, tokens / seconds
            )
        if self.enabled and contended and seconds:
            self.debt_seconds += seconds * self.fair_share
            _bump(self.counters, "prefill_chunks")


@dataclass
class CohortAdaptiveMTPDepth:
    """One adaptive draft-depth decision for an entire physical cohort.

    Depth zero is an exact ordinary target round over the still-owned target
    state.  It is a temporary park, not a lane migration.  Periodic depth-one
    probes make re-entry deterministic and prevent a permanently parked route.
    """

    max_depth: int
    ewma_alpha: float = 0.25
    shrink_gate: float = 0.35
    grow_gate: float = 0.80
    loss_rounds: int = 3
    gain_rounds: int = 3
    park_rounds: int = 4
    current_depth: int | None = None
    acceptance_ewma: float | None = None
    bad_streak: int = 0
    good_streak: int = 0
    parked_remaining: int = 0
    counters: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.max_depth = _positive_integer(
            self.max_depth, name="adaptive MTP max_depth"
        )
        self.ewma_alpha = _finite_number(
            self.ewma_alpha, name="adaptive MTP ewma_alpha"
        )
        self.shrink_gate = _finite_number(
            self.shrink_gate, name="adaptive MTP shrink_gate"
        )
        self.grow_gate = _finite_number(
            self.grow_gate, name="adaptive MTP grow_gate"
        )
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError("adaptive MTP ewma_alpha must be in (0, 1]")
        if not 0 <= self.shrink_gate < self.grow_gate <= 1:
            raise ValueError("adaptive MTP gates must satisfy 0 <= shrink < grow <= 1")
        self.loss_rounds = _positive_integer(
            self.loss_rounds, name="adaptive MTP loss_rounds"
        )
        self.gain_rounds = _positive_integer(
            self.gain_rounds, name="adaptive MTP gain_rounds"
        )
        self.park_rounds = _positive_integer(
            self.park_rounds, name="adaptive MTP park_rounds"
        )
        if self.current_depth is not None and (
            isinstance(self.current_depth, bool)
            or not isinstance(self.current_depth, int)
        ):
            raise ValueError("adaptive MTP current_depth must be an integer")
        self.current_depth = (
            self.max_depth if self.current_depth is None else self.current_depth
        )
        if not 0 <= self.current_depth <= self.max_depth:
            raise ValueError("adaptive MTP current_depth is out of range")
        for name in ("boundaries", "depth_changes", "parks", "reentries", "probes"):
            self.counters.setdefault(name, 0)

    def select(self, *, admitted_cap: int | None = None) -> int:
        """Choose at a closed boundary; never returns lane-specific depths."""
        cap = (
            self.max_depth
            if admitted_cap is None
            else max(0, min(self.max_depth, int(admitted_cap)))
        )
        depth = min(int(self.current_depth), cap)
        if self.parked_remaining > 0:
            self.parked_remaining -= 1
            if self.parked_remaining > 0:
                depth = 0
            else:
                depth = min(1, cap)
                self.current_depth = depth
                _bump(self.counters, "reentries")
                _bump(self.counters, "probes")
        _bump(self.counters, "boundaries")
        return depth

    def observe(self, proposed: int, accepted: int) -> None:
        proposed, accepted = int(proposed), int(accepted)
        if proposed < 0 or not 0 <= accepted <= proposed:
            raise ValueError("invalid adaptive MTP observation")
        if proposed == 0:
            return
        ratio = accepted / proposed
        self.acceptance_ewma = (
            ratio
            if self.acceptance_ewma is None
            else self.ewma_alpha * ratio
            + (1.0 - self.ewma_alpha) * self.acceptance_ewma
        )
        if self.acceptance_ewma < self.shrink_gate:
            self.bad_streak += 1
            self.good_streak = 0
        elif self.acceptance_ewma >= self.grow_gate:
            self.good_streak += 1
            self.bad_streak = 0
        else:
            self.bad_streak = self.good_streak = 0
        if self.bad_streak >= self.loss_rounds:
            old = int(self.current_depth)
            if old > 1:
                self.current_depth = old - 1
            else:
                self.current_depth = 0
                # One extra countdown boundary performs the re-entry; the
                # configured number remains the exact count of K=0 rounds.
                self.parked_remaining = self.park_rounds + 1
                _bump(self.counters, "parks")
            self.bad_streak = 0
            if self.current_depth != old:
                _bump(self.counters, "depth_changes")
        elif (
            self.good_streak >= self.gain_rounds
            and self.current_depth < self.max_depth
        ):
            old = int(self.current_depth)
            self.current_depth = min(self.max_depth, max(1, old + 1))
            self.good_streak = 0
            if self.current_depth != old:
                _bump(self.counters, "depth_changes")


@dataclass(frozen=True)
class AdaptiveMTPDepthPolicy:
    """Validated serving selection for the cohort-wide depth controller."""

    enabled: bool = False
    ewma_alpha: float = 0.25
    shrink_gate: float = 0.35
    grow_gate: float = 0.80
    loss_rounds: int = 3
    gain_rounds: int = 3
    park_rounds: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("adaptive MTP enabled must be boolean")
        # Reuse the executable controller's validation so the serving contract
        # cannot accept a policy that later fails after model allocation.
        CohortAdaptiveMTPDepth(max_depth=1, **self.controller_kwargs())

    @classmethod
    def from_value(
        cls, value: bool | Mapping[str, Any] | AdaptiveMTPDepthPolicy | None
    ) -> AdaptiveMTPDepthPolicy:
        if value is None or value is False:
            return cls()
        if value is True:
            return cls(enabled=True)
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("adaptive_mtp_depth must be a boolean or object")
        allowed = {
            "enabled",
            "ewma_alpha",
            "shrink_gate",
            "grow_gate",
            "loss_rounds",
            "gain_rounds",
            "park_rounds",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown adaptive MTP settings: {sorted(unknown)}")
        return cls(**dict(value))

    def controller_kwargs(self) -> dict[str, float | int]:
        return {
            "ewma_alpha": self.ewma_alpha,
            "shrink_gate": self.shrink_gate,
            "grow_gate": self.grow_gate,
            "loss_rounds": self.loss_rounds,
            "gain_rounds": self.gain_rounds,
            "park_rounds": self.park_rounds,
        }

    def as_dict(self) -> dict[str, bool | float | int]:
        return {"enabled": self.enabled, **self.controller_kwargs()}
