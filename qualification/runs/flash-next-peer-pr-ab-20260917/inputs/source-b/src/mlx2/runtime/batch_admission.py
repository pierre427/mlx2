# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, Dict, Iterable, Optional


@dataclass(frozen=True)
class AdmissionState:
    """Scheduler-facing state for one request.

    ``projected_units`` is the total work at completion, while
    ``completed_units`` records current progress.  Units are defined by the
    scheduler (tokens for autoregressive generation, denoising steps for
    diffusion).  ``resident_bytes`` must include state already counted in the
    scheduler's live-byte total.
    """

    uid: Any
    projected_units: int
    completed_units: int = 0
    resident_bytes: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if any(
            (
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in (self.projected_units, self.completed_units)
            )
        ):
            raise TypeError("work units must be non-bool integers")
        if self.projected_units < 0 or self.completed_units < 0:
            raise ValueError("work units must be non-negative")
        if self.completed_units > self.projected_units:
            raise ValueError("completed_units cannot exceed projected_units")
        if not math.isfinite(self.resident_bytes) or self.resident_bytes < 0:
            raise ValueError("resident_bytes must be finite and non-negative")


class LinearStateCost:
    """Fixed plus per-unit state cost for dense and hybrid AR caches.

    ``allocation_step_units`` describes a shared stepped allocation such as
    ``BatchKVCache``. A cache resumed from an unaligned logical length can
    retain up to ``step - 1`` extra units, so projections use the alignment-
    independent envelope ``units + step - 1`` rather than assuming that total
    capacity is step-aligned. ``cohort_bytes`` additionally models the shared
    cohort width, where every row pays for the longest row's allocation.
    """

    def __init__(
        self,
        fixed_bytes: float,
        bytes_per_unit: float,
        max_units: Optional[int] = None,
        allocation_step_units: Optional[int] = None,
    ):
        values = (fixed_bytes, bytes_per_unit)
        if not all((math.isfinite(x) and x >= 0 for x in values)):
            raise ValueError("state costs must be finite and non-negative")
        if max_units is not None and (
            isinstance(max_units, bool)
            or not isinstance(max_units, Integral)
            or max_units <= 0
        ):
            raise ValueError("max_units must be a positive non-bool integer")
        if allocation_step_units is not None and (
            isinstance(allocation_step_units, bool)
            or not isinstance(allocation_step_units, Integral)
            or allocation_step_units <= 0
        ):
            raise ValueError(
                "allocation_step_units must be a positive non-bool integer"
            )
        self.fixed_bytes = fixed_bytes
        self.bytes_per_unit = bytes_per_unit
        self.max_units = max_units
        self.allocation_step_units = allocation_step_units

    def _capped_units(self, state: AdmissionState) -> int:
        units = state.projected_units
        if self.max_units is not None:
            units = min(units, self.max_units)
        return units

    def _allocated_units(self, units: int) -> int:
        if self.allocation_step_units is None or units == 0:
            return units
        return units + self.allocation_step_units - 1

    def __call__(self, state: AdmissionState) -> float:
        units = self._allocated_units(self._capped_units(state))
        return self.fixed_bytes + self.bytes_per_unit * units

    def cohort_bytes(self, states: Iterable[AdmissionState]) -> float:
        """Project total bytes for rows sharing one batched allocation width.

        Without stepped allocation geometry, rows remain independently linear.
        With it, the whole cohort is charged at the alignment-independent
        envelope of the maximum projected width, safely covering resumed
        ``BatchKVCache`` allocations whose existing width is not step-aligned.
        """
        states = tuple(states)
        if not states:
            return 0.0
        if self.allocation_step_units is None:
            return sum((self(state) for state in states))
        max_units = max((self._capped_units(state) for state in states))
        allocated_units = self._allocated_units(max_units)
        return len(states) * (self.fixed_bytes + self.bytes_per_unit * allocated_units)


class StateBudget:
    """Admission policy over arbitrary model state.

    ``project`` returns the peak bytes a request can occupy over its remaining
    lifetime, not merely its bytes at the next step.  This distinction lets a
    diffusion adapter account for timestep-dependent activation peaks without
    pretending that its state is a token-linear KV cache.
    """

    def __init__(self, budget_bytes: float, project: Callable[[AdmissionState], float]):
        if not callable(project):
            raise TypeError("project must be callable")
        self.budget_bytes = budget_bytes
        self.project = project

    @property
    def budget_bytes(self) -> float:
        return self._budget_bytes

    @budget_bytes.setter
    def budget_bytes(self, value: float):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("budget_bytes must be finite and positive")
        self._budget_bytes = value

    def projected_bytes(self, state: AdmissionState) -> float:
        projected = self.project(state)
        if not math.isfinite(projected) or projected < 0:
            raise ValueError("projected state bytes must be finite and non-negative")
        return projected

    def remaining_bytes(self, state: AdmissionState) -> float:
        return max(self.projected_bytes(state) - state.resident_bytes, 0.0)

    def admitted_prefix(
        self,
        candidates: Iterable[AdmissionState],
        *,
        live_bytes: float,
        active: Iterable[AdmissionState] = (),
        allow_oversized_if_idle: bool = True,
    ) -> int:
        """Return how many candidates fit, preserving the supplied order.

        A caller that reorders for padding efficiency must pass the exact
        selected order here; a count computed for one cohort cannot constrain a
        different cohort.  Removed requests naturally free headroom because
        all accounting is recomputed from the supplied live state.
        """
        if not math.isfinite(live_bytes) or live_bytes < 0:
            raise ValueError("live_bytes must be finite and non-negative")
        active = tuple(active)
        committed = live_bytes + sum((self.remaining_bytes(x) for x in active))
        admitted = 0
        for state in candidates:
            need = self.remaining_bytes(state)
            if committed + need > self.budget_bytes:
                if allow_oversized_if_idle and admitted == 0 and (not active):
                    return 1
                break
            committed += need
            admitted += 1
        return admitted
