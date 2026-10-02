# SPDX-License-Identifier: Apache-2.0
# Adapted from jundot/omlx #4112; see provenance/omlx-4112-park-memory.json.
"""Measured, adaptive MTP park decision with memory that outlives a cohort.

Mined from oMLX #4112 (``ParkMemory`` and the clear-loss rule in
``omlx/patches/mlx_lm_mtp/batch_policy.py``, Apache-2.0); see
``provenance/omlx-4112-park-memory.md``.

The static ``MTPOrdinaryHandoffPolicy`` hands a native-MTP cohort to the
ordinary batcher once its width exceeds ``max_mtp_width``.  This module lets a
route replace that fixed width with a measured comparison, per cohort width,
of two rates taken in the *same* timing domain:

* MTP: tokens emitted by one native-MTP round / wall interval between two
  consecutive pure-MTP decode rounds at that width;
* ordinary: tokens emitted by one ordinary batched step / wall interval
  between two consecutive pure-ordinary decode rounds at that width.

Both are scheduler-boundary intervals of ``BatchGenerator.next`` with no
prompt work queued, so they include the same per-round host overhead.  Rounds
that mix an MTP cohort with handed-off ordinary lanes, or that overlap prompt
work, are not sampled (``mtp_adaptive_park_samples_skipped``).

Decision (``decide``), at every closed MTP boundary of a cohort of width ``w``:

1. A live verdict at ``k <= w`` parks immediately (``park_memory``): MTP does
   not get cheaper per row as rows are added.
2. No ordinary rate at ``w`` (measured, or bounded from a wider measured
   width, see ``_ordinary_rate``): the static threshold decides (cold
   start); handing off then measures ordinary decode.
3. Ordinary known, MTP not yet: MTP keeps running and is measured (probe).
4. Both known, one decision per fresh MTP sample: ``mtp < 0.9 x ordinary``
   for 4 decisions in a row (clear loss), or ``mtp <= 1.03 x ordinary`` for
   16 (sustained loss), records a verdict at ``w`` and parks
   (``measured_loss``).  ``mtp > 1.03 x ordinary`` for 32 decisions in a
   row clears every verdict at ``k <= w``.  Verdict persistence plus the
   asymmetric streaks are the hysteresis.

A verdict parks ``cooldown_cohorts`` cohorts, then expires and the next
cohort at that width re-measures MTP from scratch.  Each repeated loss at the
same width doubles its cooldown (capped), as oMLX doubles its cooldown.  oMLX
counts the cooldown in decode steps because its park is reversible inside the
cohort; mlx2's handoff is one-way per lane, so a re-measurement costs a whole
cohort's MTP ramp and the cooldown is counted in parked cohorts instead (a
2026-10-02 GPU run with a 128-step cooldown re-probed, and lost, on every
wave; see qualification/runs/port-park-memory-20261002).

Reproducibility.  Handing a cohort to the ordinary batcher is exact at width 1
(greedy MTP verify and ordinary decode commit the same tokens), but at width
greater than one MTP verify and ordinary batched decode run different
batch-shaped reductions, so a near-tie can resolve differently and later
tokens diverge.  The static policy makes the switch point a function of the
width sequence alone.  The adaptive policy makes it a function of measured
wall time as well, so two runs of the same request schedule may hand off at
different rounds and produce different batched output.  Every handoff
records its decision (reason, rates, ratio, verdict width) in the lane's
route receipt, and ``snapshot()`` exposes the memory, so a run can be
explained after the fact; it cannot be replayed bit-exactly.  Keep the
adaptive park off where batched reproducibility matters.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
from typing import Any, Mapping, Optional
import weakref

KILL_SWITCH_ENV = "MLX2_MTP_ADAPTIVE_PARK"
_COUNTER_MAX = (1 << 62) - 1


def adaptive_park_kill_switch_engaged() -> bool:
    """``MLX2_MTP_ADAPTIVE_PARK=0`` forces the static threshold."""
    value = os.environ.get(KILL_SWITCH_ENV)
    if value is None:
        return False
    return value.strip().casefold() in {"0", "false", "off", "no"}


def _bump(stats: Optional[dict], key: str, amount: int = 1) -> None:
    if stats is None:
        return
    stats[key] = min(int(stats.get(key, 0)) + int(amount), _COUNTER_MAX)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _ratio(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not 0.0 < value <= 4.0:
        raise ValueError(f"{name} must be in (0, 4]")
    return value


@dataclass(frozen=True)
class AdaptiveParkSettings:
    """Validated settings for the measured park (default off)."""

    enabled: bool = False
    # oMLX #4112 constants: a clear loss (< 0.9x ordinary) parks after 4
    # decisions; a sustained one (MTP not ahead by more than 3%) after 16; a
    # cohort that stays ahead for 32 decisions clears verdicts at <= its width.
    clear_loss_ratio: float = 0.9
    clear_loss_decisions: int = 4
    loss_ratio: float = 1.03
    loss_decisions: int = 16
    hold_ratio: float = 1.03
    hold_decisions: int = 32
    min_samples: int = 4
    ema_alpha: float = 0.25
    cooldown_cohorts: int = 2
    max_cooldown_cohorts: int = 64

    _KEYS = (
        "enabled",
        "clear_loss_ratio",
        "clear_loss_decisions",
        "loss_ratio",
        "loss_decisions",
        "hold_ratio",
        "hold_decisions",
        "min_samples",
        "ema_alpha",
        "cooldown_cohorts",
        "max_cooldown_cohorts",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("adaptive_park enabled must be boolean")
        clear = _ratio(self.clear_loss_ratio, "adaptive_park clear_loss_ratio")
        loss = _ratio(self.loss_ratio, "adaptive_park loss_ratio")
        hold = _ratio(self.hold_ratio, "adaptive_park hold_ratio")
        if not clear <= loss <= hold:
            raise ValueError(
                "adaptive_park requires clear_loss_ratio <= loss_ratio <= "
                "hold_ratio (hysteresis)"
            )
        for name in (
            "clear_loss_decisions",
            "loss_decisions",
            "hold_decisions",
            "min_samples",
            "cooldown_cohorts",
            "max_cooldown_cohorts",
        ):
            _positive_int(getattr(self, name), f"adaptive_park {name}")
        if self.max_cooldown_cohorts < self.cooldown_cohorts:
            raise ValueError(
                "adaptive_park max_cooldown_cohorts must be >= cooldown_cohorts"
            )
        alpha = self.ema_alpha
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (
            0.0 < float(alpha) <= 1.0
        ):
            raise ValueError("adaptive_park ema_alpha must be in (0, 1]")

    @classmethod
    def from_value(
        cls, value: bool | Mapping[str, Any] | "AdaptiveParkSettings" | None
    ) -> "AdaptiveParkSettings | None":
        if value is None or value is False:
            return None
        if isinstance(value, cls):
            return value if value.enabled else None
        if value is True:
            return cls(enabled=True)
        if not isinstance(value, Mapping):
            raise ValueError("adaptive_park must be an object or boolean")
        unknown = set(value) - set(cls._KEYS)
        if unknown:
            raise ValueError(f"unknown adaptive_park settings: {sorted(unknown)}")
        if value.get("enabled", True) is not True:
            if set(value) - {"enabled"}:
                raise ValueError("adaptive_park settings require enabled: true")
            return None
        return cls(**{"enabled": True, **dict(value)})

    def as_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._KEYS}


class _Rate:
    __slots__ = ("ema", "samples")

    def __init__(self) -> None:
        self.ema = 0.0
        self.samples = 0

    def add(self, value: float, alpha: float) -> None:
        self.ema = value if self.samples == 0 else (
            alpha * value + (1.0 - alpha) * self.ema
        )
        self.samples = min(self.samples + 1, _COUNTER_MAX)


class ParkMemory:
    """Per-width MTP/ordinary rates and park verdicts for one route.

    Lives longer than any cohort (see ``park_memory_for``).  All mutation is
    under one lock; the scheduler calls it from its own thread only, but the
    status endpoint reads ``snapshot()`` from another.
    """

    def __init__(
        self,
        settings: AdaptiveParkSettings,
        *,
        static_max_width: int,
        stats: Optional[dict] = None,
    ) -> None:
        if not isinstance(settings, AdaptiveParkSettings) or not settings.enabled:
            raise ValueError("ParkMemory requires enabled AdaptiveParkSettings")
        self.settings = settings
        self.static_max_width = _positive_int(
            static_max_width, "adaptive_park static_max_width"
        )
        self.stats = stats
        self._lock = threading.Lock()
        self._rates: dict[tuple[str, int], _Rate] = {}
        # width -> {"cooldown": cohorts, "remaining": cohorts still to park}
        self._verdicts: dict[int, dict[str, int]] = {}
        self._streak_width: Optional[int] = None
        self._clear_losing = 0
        self._losing = 0
        self._held = 0
        self._last_seen_samples: Optional[int] = None

    def bind_stats(self, stats: Optional[dict]) -> None:
        with self._lock:
            self.stats = stats

    # -- measurement -------------------------------------------------------
    def observe(self, kind: str, width: int, tokens: int, seconds: float) -> bool:
        """Record one decode-round sample; returns whether it was accepted."""
        if kind not in ("mtp", "ordinary"):
            raise ValueError(f"unknown park-memory sample kind {kind!r}")
        if width < 1 or tokens < 1 or not seconds > 0.0:
            _bump(self.stats, "mtp_adaptive_park_samples_skipped")
            return False
        with self._lock:
            rate = self._rates.setdefault((kind, int(width)), _Rate())
            rate.add(float(tokens) / (float(seconds) * 1000.0), self.settings.ema_alpha)
        _bump(self.stats, f"mtp_adaptive_park_{kind}_samples")
        return True

    def note_skipped(self) -> None:
        _bump(self.stats, "mtp_adaptive_park_samples_skipped")

    def _expire_locked(self, width: int) -> None:
        """Re-measure MTP from scratch at and above an expired verdict."""
        for key in [k for k in self._rates if k[0] == "mtp" and k[1] >= width]:
            del self._rates[key]

    def _rate(self, kind: str, width: int) -> Optional[_Rate]:
        rate = self._rates.get((kind, width))
        if rate is None or rate.samples < self.settings.min_samples:
            return None
        return rate

    def _ordinary_rate(self, width: int) -> Optional[tuple[float, str]]:
        """Measured ordinary rate at ``width``, else a conservative bound.

        Ordinary batched step time does not decrease with width, so the step
        time measured at the nearest wider width ``v`` bounds the step time at
        ``width`` from above and ``rate(v) * width / v`` bounds the ordinary
        rate from below.  Comparing MTP against a lower bound can miss a park
        but cannot invent one.
        """
        exact = self._rate("ordinary", width)
        if exact is not None:
            return exact.ema, "measured"
        wider = sorted(
            v
            for (kind, v), rate in self._rates.items()
            if kind == "ordinary" and v > width
            and rate.samples >= self.settings.min_samples
        )
        if not wider:
            return None
        v = wider[0]
        return self._rates[("ordinary", v)].ema * width / v, f"bound_from_width_{v}"

    def _live_verdict(self, width: int) -> Optional[int]:
        live = [
            k for k, v in self._verdicts.items() if k <= width and v["remaining"] > 0
        ]
        return min(live) if live else None

    # -- decisions ---------------------------------------------------------
    def peek(self, width: int) -> Optional[dict[str, Any]]:
        """Decision for ``width`` without advancing any streak."""
        with self._lock:
            return self._peek_locked(int(width))

    def _peek_locked(self, width: int) -> Optional[dict[str, Any]]:
        verdict = self._live_verdict(width)
        if verdict is not None:
            return {
                "reason": "park_memory",
                "verdict_width": verdict,
                "remaining_cohorts": self._verdicts[verdict]["remaining"],
            }
        if self._ordinary_rate(width) is None:
            if width > self.static_max_width:
                return {
                    "reason": "static_width_threshold",
                    "adaptive": "cold_start_no_ordinary_rate",
                    "max_mtp_width": self.static_max_width,
                }
        return None

    def would_park(self, width: int) -> bool:
        return self.peek(width) is not None

    def decide(self, width: int) -> Optional[dict[str, Any]]:
        """Decision at a closed MTP boundary; advances the loss/hold streaks."""
        width = int(width)
        cleared = 0
        with self._lock:
            if self._streak_width != width:
                self._streak_width = width
                self._clear_losing = 0
                self._losing = 0
                self._held = 0
                self._last_seen_samples = None
            early = self._peek_locked(width)
            expired = False
            if early is not None:
                result = early
                if early["reason"] == "park_memory":
                    # This cohort is parked; count it against the cooldown.
                    verdict = self._verdicts[early["verdict_width"]]
                    verdict["remaining"] -= 1
                    if verdict["remaining"] == 0:
                        self._expire_locked(early["verdict_width"])
                        expired = True
            else:
                ordinary = self._ordinary_rate(width)
                mtp = self._rate("mtp", width)
                result = None
                if ordinary is not None and mtp is None:
                    pass  # MTP is being measured at this width (probe).
                elif ordinary is not None and mtp.samples != self._last_seen_samples:
                    # One decision per fresh MTP sample, never per boundary.
                    self._last_seen_samples = mtp.samples
                    ordinary_rate, ordinary_source = ordinary
                    ratio = mtp.ema / max(ordinary_rate, 1e-12)
                    s = self.settings
                    self._clear_losing = (
                        self._clear_losing + 1 if ratio < s.clear_loss_ratio else 0
                    )
                    self._losing = self._losing + 1 if ratio <= s.loss_ratio else 0
                    self._held = self._held + 1 if ratio > s.hold_ratio else 0
                    _bump(self.stats, "mtp_adaptive_park_decisions")
                    if self._held >= s.hold_decisions:
                        doomed = [k for k in self._verdicts if k <= width]
                        for k in doomed:
                            del self._verdicts[k]
                        cleared = len(doomed)
                        self._held = 0
                    if (
                        self._clear_losing >= s.clear_loss_decisions
                        or self._losing >= s.loss_decisions
                    ):
                        previous = self._verdicts.get(width)
                        cooldown = (
                            s.cooldown_cohorts
                            if previous is None
                            else min(
                                s.max_cooldown_cohorts, previous["cooldown"] * 2
                            )
                        )
                        self._verdicts[width] = {
                            "cooldown": cooldown,
                            "remaining": cooldown,
                        }
                        rule = (
                            "clear_loss"
                            if self._clear_losing >= s.clear_loss_decisions
                            else "sustained_loss"
                        )
                        self._clear_losing = 0
                        self._losing = 0
                        _bump(self.stats, "mtp_adaptive_park_verdicts_set")
                        result = {
                            "reason": "measured_loss",
                            "rule": rule,
                            "verdict_width": width,
                            "cooldown_cohorts": cooldown,
                            "mtp_tokens_per_ms": round(mtp.ema, 6),
                            "ordinary_tokens_per_ms": round(ordinary_rate, 6),
                            "ordinary_rate_source": ordinary_source,
                            "ratio": round(ratio, 6),
                        }
        if cleared:
            _bump(self.stats, "mtp_adaptive_park_verdicts_cleared", cleared)
        if expired:
            _bump(self.stats, "mtp_adaptive_park_verdicts_expired")
        if result is not None:
            _bump(self.stats, f"mtp_adaptive_park_{result['reason']}")
        return result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "settings": self.settings.as_dict(),
                "static_max_width": self.static_max_width,
                "rates": {
                    f"{kind}:{width}": {
                        "tokens_per_ms": round(rate.ema, 6),
                        "samples": rate.samples,
                    }
                    for (kind, width), rate in sorted(self._rates.items())
                },
                "verdicts": {
                    str(width): dict(verdict)
                    for width, verdict in sorted(self._verdicts.items())
                },
            }


# One memory per live model object and route key, like oMLX's per-model
# ParkMemory: a BatchGenerator may be rebuilt for a new wave or cohort and must
# not re-measure a known loss.  The weak reference keeps a reused ``id`` from
# inheriting another model's verdicts.
_MEMORIES: dict[tuple[int, str], tuple[Any, ParkMemory]] = {}
_MEMORIES_LOCK = threading.Lock()


def park_memory_for(
    model: Any,
    settings: AdaptiveParkSettings,
    *,
    static_max_width: int,
    route_key: str,
    stats: Optional[dict] = None,
) -> ParkMemory:
    key = (
        id(model),
        f"{route_key}|{static_max_width}|{sorted(settings.as_dict().items())}",
    )
    with _MEMORIES_LOCK:
        for stale in [k for k, (ref, _) in _MEMORIES.items() if ref() is None]:
            del _MEMORIES[stale]
        entry = _MEMORIES.get(key)
        if entry is None or entry[0]() is not model:
            memory = ParkMemory(
                settings, static_max_width=static_max_width, stats=stats
            )
            try:
                ref = weakref.ref(model)
            except TypeError:
                return memory
            _MEMORIES[key] = (ref, memory)
            return memory
    memory = entry[1]
    memory.bind_stats(stats)
    return memory


def reset_park_memories(model: Any = None) -> None:
    """Forget verdicts (for one model, or all); tests and route reloads."""
    with _MEMORIES_LOCK:
        for key in [
            k for k in _MEMORIES if model is None or k[0] == id(model)
        ]:
            del _MEMORIES[key]
