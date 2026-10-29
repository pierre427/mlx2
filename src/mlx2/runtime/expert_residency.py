# SPDX-License-Identifier: Apache-2.0
"""Model-neutral demand-only expert admission. See Basalt provenance (2026-10-24).

Scores control retention, never routing, bytes, arithmetic, or speculative
acceptance. A bounded vector per projection prevents unbounded request history.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AdaptiveResidencySettings:
    enabled: bool = False
    decay: float = 0.9
    decay_every: int = 32
    hysteresis: float = 1.5
    max_swaps: int = 2
    prompt_weight: float = 2.0
    profile_path: str | None = None

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("adaptive residency enabled must be boolean")
        for key in ("decay_every", "max_swaps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"adaptive residency {key} must be a positive integer")
        for key in ("decay", "hysteresis", "prompt_weight"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"adaptive residency {key} must be finite")
        if not 0 < self.decay < 1 or self.hysteresis < 0 or self.prompt_weight < 0:
            raise ValueError(
                "adaptive residency decay must be in (0,1); weights must be nonnegative"
            )
        if self.profile_path is not None and (
            not isinstance(self.profile_path, str) or not self.profile_path
        ):
            raise ValueError(
                "adaptive residency profile_path must be a nonempty string or null"
            )
        if self.profile_path and not self.enabled:
            raise ValueError("adaptive residency profile_path requires enabled=true")

    @classmethod
    def parse(cls, value=None):
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("invalid adaptive residency settings")
        return cls(**value)

    def as_dict(self):
        return asdict(self)


class AdaptiveExpertResidency:
    def __init__(self, num_experts: int, settings: AdaptiveResidencySettings):
        if type(num_experts) is not int or num_experts < 1:
            raise ValueError("num_experts must be positive")
        self.settings = settings
        self.scores = [0.0] * num_experts
        self.calls = 0
        self.swaps = 0
        self.ephemeral = 0

    def observe(self, wanted, *, prefill=False):
        wanted = tuple(dict.fromkeys(wanted))
        if any(type(i) is not int or not 0 <= i < len(self.scores) for i in wanted):
            raise ValueError("expert id outside projection")
        self.calls += 1
        if self.calls % self.settings.decay_every == 0:
            self.scores = [score * self.settings.decay for score in self.scores]
        weight = self.settings.prompt_weight if prefill else 1.0
        for expert in wanted:
            self.scores[expert] += weight

    def retain(self, resident, wanted, capacity):
        """Keep hot residents; only demand-fetched candidates may enter.

        Resident order is LRU order and breaks score ties deterministically.
        Even an over-capacity gather is returned whole by the caller; this
        decision selects only the persistent bank after that gather.
        """
        keep = list(resident)
        while len(keep) > capacity:
            keep.remove(min(keep, key=lambda i: self.scores[i]))
        swaps = 0
        for expert in sorted(
            set(wanted) - set(keep), key=lambda i: (-self.scores[i], i)
        ):
            if len(keep) < capacity:
                keep.append(expert)
            elif swaps < self.settings.max_swaps:
                victim = min(keep, key=lambda i: self.scores[i])
                if self.scores[expert] > self.scores[victim] + self.settings.hysteresis:
                    keep.remove(victim)
                    keep.append(expert)
                    swaps += 1
        self.swaps += swaps
        self.ephemeral += len(set(wanted) - set(keep))
        return frozenset(keep)

    def restore_scores(self, scores):
        if not isinstance(scores, list) or len(scores) != len(self.scores):
            raise ValueError("residency profile geometry mismatch")
        if any(
            isinstance(s, bool)
            or not isinstance(s, (int, float))
            or not math.isfinite(s)
            or s < 0
            for s in scores
        ):
            raise ValueError("residency profile scores must be finite and nonnegative")
        self.scores = list(scores)
