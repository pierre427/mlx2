# SPDX-License-Identifier: Apache-2.0
"""Request-owned cost-per-committed-token speculative handoff.

The controller in this module owns policy only.  Executors remain responsible
for observing a closed committed boundary and for moving exact target state to
their ordinary decoder.  State is intentionally lane-local: timing evidence is
never shared across requests, tenants, draft providers, or execution widths.

The feature is default-off at every caller.  A selected controller first
collects ordinary target samples, then speculative samples in the same timing
domain and physical width.  A sustained speculative loss produces one terminal
decision.  It can never re-enter speculation for that request.

Target cost grows with context for both routes, so a terminal decision may
only use the speculative samples that directly follow an ordinary baseline
(``speculative_samples + loss_decisions - 1``, the fewest that can decide).
A sustained loss against an older baseline does not decide: it discards both
windows and asks for a fresh ordinary probe at the current context.  A
speculator that keeps winning pays no extra probes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Hashable, Mapping


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unit_interval(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be in (0, 1]")
    return value


@dataclass(frozen=True)
class CostPerCommitHandoffSettings:
    """Validated, conservative and default-off decision settings."""

    enabled: bool = False
    ordinary_samples: int = 4
    speculative_samples: int = 4
    loss_ratio: float = 0.97
    loss_decisions: int = 3
    window: int = 16

    _KEYS = frozenset(
        {
            "enabled",
            "ordinary_samples",
            "speculative_samples",
            "loss_ratio",
            "loss_decisions",
            "window",
        }
    )

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("cost-per-commit handoff enabled must be boolean")
        for name in (
            "ordinary_samples",
            "speculative_samples",
            "loss_decisions",
            "window",
        ):
            _positive_int(getattr(self, name), f"cost-per-commit {name}")
        _unit_interval(self.loss_ratio, "cost-per-commit loss_ratio")
        if self.window < max(self.ordinary_samples, self.speculative_samples):
            raise ValueError(
                "cost-per-commit window must cover both sample minima"
            )

    @classmethod
    def from_value(
        cls,
        value: bool | Mapping[str, Any] | "CostPerCommitHandoffSettings" | None,
    ) -> "CostPerCommitHandoffSettings | None":
        if value is None or value is False:
            return None
        if isinstance(value, cls):
            return value if value.enabled else None
        if value is True:
            return cls(enabled=True)
        if not isinstance(value, Mapping):
            raise ValueError("cost-per-commit handoff must be an object or boolean")
        unknown = set(value) - cls._KEYS
        if unknown:
            raise ValueError(
                f"unknown cost-per-commit handoff settings: {sorted(unknown)}"
            )
        if value.get("enabled", True) is not True:
            if set(value) - {"enabled"}:
                raise ValueError(
                    "disabled cost-per-commit handoff cannot carry settings"
                )
            return None
        return cls(**{"enabled": True, **dict(value)})

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ordinary_samples": self.ordinary_samples,
            "speculative_samples": self.speculative_samples,
            "loss_ratio": self.loss_ratio,
            "loss_decisions": self.loss_decisions,
            "window": self.window,
        }


class CostPerCommitLaneState:
    """One request's one-way handoff evidence and terminal decision."""

    def __init__(
        self,
        settings: CostPerCommitHandoffSettings,
        *,
        lane_uid: Hashable,
        owner_scope: Hashable,
    ) -> None:
        if not isinstance(settings, CostPerCommitHandoffSettings):
            raise TypeError("validated cost-per-commit settings are required")
        if not settings.enabled:
            raise ValueError("cost-per-commit handoff settings are disabled")
        try:
            hash((lane_uid, owner_scope))
        except TypeError as error:
            raise TypeError("lane uid and owner scope must be hashable") from error
        self.settings = settings
        self.lane_uid = lane_uid
        self.owner_scope = owner_scope
        self._ordinary: dict[int, deque[tuple[int, int]]] = {}
        self._speculative: dict[int, deque[tuple[int, int]]] = {}
        self._loss_streak: dict[int, int] = {}
        # Speculative samples taken since this width's ordinary baseline was
        # last sampled; bounds how stale a baseline a decision may use.
        self._speculative_since_ordinary: dict[int, int] = {}
        self._ordinary_refreshes = 0
        self._decision: dict[str, Any] | None = None
        self._cancelled = False
        self._invalid_samples = 0
        self._observations = 0
        self._valid_samples = 0

    @staticmethod
    def _cost(samples: deque[tuple[int, int]]) -> float | None:
        committed = sum(count for _elapsed, count in samples)
        return (
            None
            if committed <= 0
            else sum(elapsed for elapsed, _count in samples) / committed
        )

    def needs_ordinary_probe(self, execution_width: int) -> bool:
        if self._decision is not None or self._cancelled:
            return False
        samples = self._ordinary.get(int(execution_width), ())
        return len(samples) < self.settings.ordinary_samples

    @property
    def handed_off(self) -> bool:
        return self._decision is not None

    def cancel(self) -> None:
        self._cancelled = True
        self._ordinary.clear()
        self._speculative.clear()
        self._loss_streak.clear()
        self._speculative_since_ordinary.clear()

    def observe(
        self,
        *,
        kind: str,
        elapsed_ns: int,
        committed_tokens: int,
        execution_width: int,
        lane_uid: Hashable,
        owner_scope: Hashable,
    ) -> dict[str, Any] | None:
        """Observe one closed boundary and return a new terminal decision.

        Identity must match exactly.  A mismatch raises instead of allowing a
        caller to mix another lane's or tenant's cost evidence into this lane.
        Invalid timing samples are ignored and counted; they never trigger a
        handoff.
        """
        if lane_uid != self.lane_uid or owner_scope != self.owner_scope:
            raise ValueError("cost-per-commit handoff owner identity mismatch")
        if kind not in {"ordinary", "speculative"}:
            raise ValueError("cost-per-commit sample kind must be ordinary or speculative")
        if self._decision is not None or self._cancelled:
            return None
        self._observations += 1
        if (
            type(elapsed_ns) is not int
            or type(committed_tokens) is not int
            or type(execution_width) is not int
            or elapsed_ns <= 0
            or committed_tokens <= 0
            or execution_width <= 0
            or not math.isfinite(float(elapsed_ns))
        ):
            self._invalid_samples += 1
            return None
        self._valid_samples += 1
        table = self._ordinary if kind == "ordinary" else self._speculative
        samples = table.setdefault(
            execution_width, deque(maxlen=self.settings.window)
        )
        samples.append((elapsed_ns, committed_tokens))
        if kind == "ordinary":
            self._speculative_since_ordinary[execution_width] = 0
            return None
        ordinary = self._ordinary.get(execution_width)
        if ordinary is None or len(ordinary) < self.settings.ordinary_samples:
            return None
        since = self._speculative_since_ordinary.get(execution_width, 0) + 1
        self._speculative_since_ordinary[execution_width] = since
        if len(samples) < self.settings.speculative_samples:
            return None
        ordinary_cost = self._cost(ordinary)
        speculative_cost = self._cost(samples)
        if ordinary_cost is None or speculative_cost is None:
            return None
        losing = speculative_cost >= ordinary_cost / self.settings.loss_ratio
        streak = self._loss_streak.get(execution_width, 0)
        streak = streak + 1 if losing else 0
        self._loss_streak[execution_width] = streak
        if streak < self.settings.loss_decisions:
            return None
        if since > (
            self.settings.speculative_samples + self.settings.loss_decisions - 1
        ):
            # The baseline predates the speculative window.  Re-measure both
            # routes at the current context before any one-way decision.
            del self._ordinary[execution_width]
            samples.clear()
            self._loss_streak[execution_width] = 0
            self._speculative_since_ordinary[execution_width] = 0
            self._ordinary_refreshes += 1
            return None
        self._decision = {
            "reason": "cost_per_committed_token",
            "one_way": True,
            "execution_width": execution_width,
            "ordinary_ns_per_committed_token": ordinary_cost,
            "speculative_ns_per_committed_token": speculative_cost,
            "speculative_to_ordinary_cost_ratio": speculative_cost / ordinary_cost,
            "loss_ratio_threshold": self.settings.loss_ratio,
            "loss_decisions": streak,
            "speculative_samples_since_ordinary": since,
        }
        return dict(self._decision)

    def receipt(self) -> dict[str, Any]:
        def costs(table):
            return {
                str(width): {
                    "samples": len(samples),
                    "committed_tokens": sum(count for _elapsed, count in samples),
                    "ns_per_committed_token": self._cost(samples),
                }
                for width, samples in sorted(table.items())
            }

        return {
            "implemented": True,
            "qualified": False,
            "selected": True,
            "defaulted": False,
            "observed_used": self._observations > 0,
            "observed_sampling": self._valid_samples > 0,
            "handoff_triggered": self._decision is not None,
            "one_way": True,
            "scope": "request_lane",
            "settings": self.settings.as_dict(),
            "ordinary": costs(self._ordinary),
            "speculative": costs(self._speculative),
            "decision": None if self._decision is None else dict(self._decision),
            "cancelled": self._cancelled,
            "invalid_samples": self._invalid_samples,
            "ordinary_refreshes": self._ordinary_refreshes,
            "observations": self._observations,
            "valid_samples": self._valid_samples,
        }


def cohort_needs_ordinary_probe(states, execution_width: int) -> bool:
    """One late lane makes the whole physical cohort run an ordinary probe.

    Native and external batching cannot mix verify depths inside one physical
    cohort.  Existing lanes may collect another ordinary sample; their
    speculative windows remain intact, while the late lane is guaranteed to
    finish its own baseline.
    """
    return any(
        state is not None and state.needs_ordinary_probe(execution_width)
        for state in states
    )


__all__ = [
    "CostPerCommitHandoffSettings",
    "CostPerCommitLaneState",
    "cohort_needs_ordinary_probe",
]
