"""Opt-in power governor: steer measured GPU + DRAM power by shaping work.

macOS gives userspace no DVFS or power-limit control, so the governor never
touches clocks.  It shapes the work the scheduler submits, through one hook
the worker calls before forming each round (:meth:`PowerGovernor.round_hint`),
and it reads power from :class:`mlx2.power_telemetry.PowerTelemetry`, which
calls :meth:`PowerGovernor.observe` once per closed sampling interval.

Modes
-----
``off`` / ``max_throughput``
    No intervention: the hint is neutral and nothing is actuated.
``efficient``
    Maximise tokens per joule.  On the M5 Max a B4 decode round costs about
    the same watts as a B1 round and yields 2.1-2.6x the tokens, so the only
    lever here is width: while the engine is idle, the admission window is
    held open up to ``efficient_hold_ms`` so near-simultaneous arrivals start
    as one cohort.  It never paces (idle gaps cost 17-24 % more J/token in the
    thermal-q38 duty runs) and never caps lanes.
``budget``
    Hold rolling GPU + DRAM power under ``budget_watts``.  A ramped integral
    controller with hysteresis moves one throttle ``u`` in ``[0, 1]`` across
    an ordered actuator ladder.  It steps on each closed interval's mean
    power, not on the rolling-window mean: holding every interval near the
    budget is how the window is held under it.  Stepping on the lagging
    window mean instead makes the ramp overshoot into a limit cycle (6-40 W
    on the thermal-q38 plant) that breaks the rolling budget.  ``pace`` lowers the busy duty cycle by
    resting between rounds (the only lever that measurably lowers watts:
    idle GPU + DRAM is ~1 W, active 35-45 W).  ``lanes`` caps how many lanes
    admission may fill (an active lane is never evicted); measured decode
    power is nearly flat in width (ordinary B1 35 W vs B4 38 W, and MTP or
    DFlash2 B1 draw *more* than B4), so it is the second rung by default.
    The idle-admission hold of ``efficient`` also applies.

Guarantees: pacing never drops the busy duty below ``min_duty`` and never
rests more than ``MAX_PACE_SECONDS`` at once, and the lane cap never goes
below ``lane_floor`` (>= 1), so every admitted request keeps progressing and
the queue keeps draining in order.  Pacing does not change token outputs.
A lane cap or a hold can change batch width, and so batch-width numerics; a
request it touched is labelled ``numerics: "batch_width_may_differ"``.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass

MODES = ("off", "max_throughput", "efficient", "budget")
#: Modes that change nothing; ``off`` at startup means no governor at all.
NEUTRAL_MODES = frozenset({"off", "max_throughput"})
ACTUATORS = ("pace", "lanes")
RECEIPT_SCHEMA = "mlx2.power-governor.v1"


class PowerSignalUnavailable(RuntimeError):
    """Budget mode was requested without a power measurement to hold it."""


@dataclass(frozen=True)
class PowerGovernorPolicy:
    mode: str = "off"
    budget_watts: float | None = None
    #: Rolling window the budget is stated over; reported as the compliance
    #: view (``within_budget``).  The controller steps on each interval.
    window_seconds: float = 30.0
    #: Idle-admission hold for ``efficient`` and ``budget`` (0 disables).
    efficient_hold_ms: float = 50.0
    actuator_order: tuple = ("pace", "lanes")
    #: Lowest busy fraction pacing may impose.
    min_duty: float = 0.1
    lane_floor: int = 1
    #: Release only below ``budget * (1 - hysteresis)``.
    hysteresis: float = 0.1

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @classmethod
    def from_value(cls, value) -> PowerGovernorPolicy:
        """Parse ``None``/``False`` (off), a mode string or an object."""
        if value is None or value is False:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            value = {"mode": value}
        if not isinstance(value, dict):
            raise ValueError("power_governor must be a mode string or an object")
        fields = set(cls.__dataclass_fields__)
        unknown = set(value) - fields
        if unknown:
            raise ValueError(
                "unknown power_governor fields: " + ", ".join(sorted(unknown))
            )
        mode = value.get("mode", "off")
        if mode not in MODES:
            raise ValueError(f"power_governor mode must be one of {', '.join(MODES)}")
        budget = value.get("budget_watts")
        if budget is not None:
            budget = _number(budget, "budget_watts", 1.0, 1000.0)
        if mode == "budget" and budget is None:
            raise ValueError("power_governor budget mode requires budget_watts")
        window = _number(value.get("window_seconds", 30.0), "window_seconds", 1.0, 3600.0)
        hold = _number(value.get("efficient_hold_ms", 50.0), "efficient_hold_ms", 0.0, 1000.0)
        min_duty = _number(value.get("min_duty", 0.1), "min_duty", 0.05, 1.0)
        hysteresis = _number(value.get("hysteresis", 0.1), "hysteresis", 0.0, 0.5)
        order = tuple(value.get("actuator_order", ("pace", "lanes")))
        if not order or len(set(order)) != len(order) or set(order) - set(ACTUATORS):
            raise ValueError(
                "power_governor actuator_order must name distinct actuators from "
                + ", ".join(ACTUATORS)
            )
        floor = value.get("lane_floor", 1)
        if isinstance(floor, bool) or not isinstance(floor, int) or floor < 1:
            raise ValueError("power_governor lane_floor must be a positive integer")
        return cls(
            mode=mode,
            budget_watts=budget,
            window_seconds=window,
            efficient_hold_ms=hold,
            actuator_order=order,
            min_duty=min_duty,
            lane_floor=floor,
            hysteresis=hysteresis,
        )

    def as_dict(self) -> dict:
        value = asdict(self)
        value["actuator_order"] = list(self.actuator_order)
        return value


def _number(value, name, low, high) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"power_governor {name} must be a number")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"power_governor {name} must be {low:g} to {high:g}")
    return value


@dataclass(frozen=True)
class RoundHint:
    """What the scheduler applies to the round it is about to form.

    ``lane_cap`` None means ``max_lanes``; ``coalesce_seconds`` None means the
    configured idle-admission window.
    """

    lane_cap: int | None = None
    pace_seconds: float = 0.0
    coalesce_seconds: float | None = None


NEUTRAL_HINT = RoundHint()


class _Account:
    __slots__ = ("hold_ms", "lane_cap_min", "modes", "pace_ms")

    def __init__(self, mode):
        self.modes = {mode}
        self.pace_ms = 0.0
        self.lane_cap_min = None
        self.hold_ms = 0.0


class PowerGovernor:
    """Closed-loop work shaper; thread-safe, clock injectable for tests."""

    #: Throttle change allowed per power sample (the ramp).
    MAX_STEP = 0.1
    #: Throttle change per unit of relative power error.
    GAIN = 0.5
    #: Seconds below the release threshold before the throttle eases.
    RELEASE_HOLD_SECONDS = 3.0
    #: Rest is owed per round and paid in gaps of at least this long, so the
    #: GPU can leave its active state instead of flickering every token.
    MIN_PACE_SECONDS = 0.05
    #: Longest single gap; bounds cancellation and deadline latency.
    MAX_PACE_SECONDS = 1.0
    #: The signal is stale after this many seconds (or three intervals)
    #: without a sample; budget mode then holds its throttle.
    STALE_SECONDS = 5.0

    def __init__(
        self, policy, *, max_lanes: int, signal_interval_seconds=1.0,
        clock=time.monotonic,
    ):
        self.policy = PowerGovernorPolicy.from_value(policy)
        self.max_lanes = max(1, int(max_lanes))
        self._stale_after = max(self.STALE_SECONDS, 3.0 * float(signal_interval_seconds))
        self._clock = clock
        self._lock = threading.Lock()
        self._mode = self.policy.mode
        self._budget = self.policy.budget_watts
        self._throttle = 0.0
        self._below_since = None
        self._samples = deque()
        self._last_sample_at = None
        self._last_watts = None
        self._observed = 0
        self._owed = 0.0
        self._round_started = None
        self._last_hint_held = False
        # Whether this engine ran a lane during the current telemetry interval
        # (anti-windup: host-wide power is only ours to correct when it did).
        # None = no engine has reported rounds yet: treat as active.
        self._active_now = None
        self._active_in_interval = None
        self._accounts: dict[str, _Account] = {}
        self._counts = {
            "paced_rounds": 0,
            "pace_seconds": 0.0,
            "capped_rounds": 0,
            "held_admissions": 0,
            "mode_changes": 0,
            "tighten_steps": 0,
            "release_steps": 0,
            "idle_tighten_skipped": 0,
        }
        self._last_change = {
            "at": time.time(), "mode": self._mode, "budget_watts": self._budget,
            "source": "startup",
        }
        self.signal_reason = None

    # -- control surface -----------------------------------------------------

    def configure(self, *, mode=None, budget_watts=None, source="admin") -> dict:
        """Change mode and/or budget at runtime (the grid-signal hook)."""
        with self._lock:
            mode = self._mode if mode is None else mode
            if mode not in MODES:
                raise ValueError(f"mode must be one of {', '.join(MODES)}")
            budget = self._budget if budget_watts is None else _number(
                budget_watts, "watts", 1.0, 1000.0
            )
            if mode == "budget" and budget is None:
                raise ValueError("budget mode requires watts")
            if mode == "budget" and self.signal_reason is not None:
                raise PowerSignalUnavailable(
                    "budget mode needs a power signal: " + self.signal_reason
                )
            if mode != self._mode:
                self._counts["mode_changes"] += 1
                if mode != "budget":
                    # Leaving budget releases at once; the other modes never
                    # pace or cap, so there is nothing to ramp down.
                    self._throttle = 0.0
                    self._owed = 0.0
                self._below_since = None
            self._mode, self._budget = mode, budget
            self._last_change = {
                "at": time.time(), "mode": mode, "budget_watts": budget,
                "source": source,
            }
            return self._status_locked()

    @property
    def mode(self) -> str:
        return self._mode

    # -- measurement ---------------------------------------------------------

    def observe(self, seconds: float, watts: float) -> None:
        """One closed telemetry interval of GPU + DRAM mean power."""
        seconds, watts = float(seconds), float(watts)
        if not (math.isfinite(seconds) and seconds > 0 and math.isfinite(watts)):
            return
        with self._lock:
            now = self._clock()
            self._samples.append((now, seconds, watts * seconds))
            horizon = now - self.policy.window_seconds
            while self._samples and self._samples[0][0] <= horizon:
                self._samples.popleft()
            self._last_sample_at = now
            self._last_watts = watts
            self._observed += 1
            if self._mode == "budget":
                self._step_locked(
                    now, watts, active=self._active_in_interval is not False
                )
            self._active_in_interval = self._active_now

    def _step_locked(self, now, watts, *, active=True) -> None:
        budget = self._budget
        release_at = budget * (1.0 - self.policy.hysteresis)
        if watts > budget:
            self._below_since = None
            if not active:
                # Anti-windup: the reading is host-wide.  With no lane of
                # ours running in the interval the throttle cannot lower it,
                # so tightening would only bank throttle for our next round.
                self._counts["idle_tighten_skipped"] += 1
                return
            step = min(self.MAX_STEP, self.GAIN * (watts - budget) / budget)
            if self._throttle < 1.0:
                self._throttle = min(1.0, self._throttle + step)
                self._counts["tighten_steps"] += 1
        elif watts < release_at:
            if self._below_since is None:
                self._below_since = now
            if now - self._below_since >= self.RELEASE_HOLD_SECONDS and self._throttle > 0:
                step = min(self.MAX_STEP, self.GAIN * (release_at - watts) / budget)
                self._throttle = max(0.0, self._throttle - step)
                self._counts["release_steps"] += 1
        else:
            self._below_since = None  # inside the band: hold

    # -- actuators -----------------------------------------------------------

    def _rung_locked(self, name) -> float:
        """Fraction of ``name``'s range the throttle currently uses."""
        order = self.policy.actuator_order
        if name not in order:
            return 0.0
        span = 1.0 / len(order)
        start = order.index(name) * span
        return min(1.0, max(0.0, (self._throttle - start) / span))

    def _duty_locked(self) -> float:
        if self._mode != "budget":
            return 1.0
        return 1.0 - self._rung_locked("pace") * (1.0 - self.policy.min_duty)

    def _lane_cap_locked(self) -> int:
        floor = min(self.policy.lane_floor, self.max_lanes)
        if self._mode != "budget":
            return self.max_lanes
        level = self._rung_locked("lanes")
        return max(floor, self.max_lanes - round(level * (self.max_lanes - floor)))

    def round_hint(self, active_ids=()) -> RoundHint:
        """Called once before each round with the ids of the active lanes."""
        with self._lock:
            mode = self._mode
            active = tuple(active_ids)
            if mode in NEUTRAL_MODES:
                self._round_started = None
                self._owed = 0.0
                self._last_hint_held = False
                self._track_locked(active, mode, 0.0, None)
                return NEUTRAL_HINT
            now = self._clock()
            self._active_now = bool(active)
            if active:
                self._active_in_interval = True
            elif self._active_in_interval is None:
                self._active_in_interval = False
            duty = self._duty_locked()
            pace = 0.0
            if active and self._round_started is not None and duty < 1.0:
                busy = max(0.0, now - self._round_started)
                self._owed += busy * (1.0 - duty) / duty
            if not active:
                # Idle time is rest already; nothing is owed across it.
                self._owed = 0.0
            if self._owed >= self.MIN_PACE_SECONDS:
                pace = min(self._owed, self.MAX_PACE_SECONDS)
                self._owed -= pace
                self._counts["paced_rounds"] += 1
                self._counts["pace_seconds"] += pace
            self._round_started = now + pace if active else None
            cap = self._lane_cap_locked()
            if cap < self.max_lanes and active:
                self._counts["capped_rounds"] += 1
            hold = None
            if not active and self.policy.efficient_hold_ms > 0 and self.max_lanes > 1:
                # A single-lane server has no wider cohort to wait for.
                hold = self.policy.efficient_hold_ms / 1000.0
            self._track_locked(active, mode, pace, cap if cap < self.max_lanes else None)
            self._last_hint_held = hold is not None
            return RoundHint(
                lane_cap=cap if cap < self.max_lanes else None,
                pace_seconds=pace,
                coalesce_seconds=hold,
            )

    def _track_locked(self, active, mode, pace, cap) -> None:
        live = set(active)
        for request_id in [key for key in self._accounts if key not in live]:
            # Left the lanes without a receipt (failed or cancelled).
            del self._accounts[request_id]
        for request_id in active:
            account = self._accounts.get(request_id)
            if account is None:
                account = self._accounts[request_id] = _Account(mode)
                if self._last_hint_held:
                    # Admitted while the previous hint held the idle window.
                    account.hold_ms = self.policy.efficient_hold_ms
                    self._counts["held_admissions"] += 1
            account.modes.add(mode)
            account.pace_ms += pace * 1000.0
            if cap is not None:
                account.lane_cap_min = (
                    cap if account.lane_cap_min is None
                    else min(account.lane_cap_min, cap)
                )

    # -- views ---------------------------------------------------------------

    def request_receipt(self, request_id) -> dict:
        """Close ``request_id``'s account for its terminal receipt."""
        with self._lock:
            account = self._accounts.pop(request_id, None) or _Account(self._mode)
            account.modes.add(self._mode)
            touched_width = account.lane_cap_min is not None or account.hold_ms > 0
            return {
                "schema": RECEIPT_SCHEMA,
                "mode": self._mode,
                "modes": sorted(account.modes),
                "budget_watts": self._budget if "budget" in account.modes else None,
                "pace_ms": account.pace_ms,
                "lane_cap_min": account.lane_cap_min,
                "coalesce_hold_ms": account.hold_ms,
                "numerics": "batch_width_may_differ" if touched_width else "unchanged",
            }

    def _window_locked(self) -> dict:
        seconds = sum(record[1] for record in self._samples)
        joules = sum(record[2] for record in self._samples)
        return {
            "seconds": seconds,
            "samples": len(self._samples),
            "gpu_dram_watts": joules / seconds if seconds else None,
        }

    def _status_locked(self) -> dict:
        now = self._clock()
        window = self._window_locked()
        stale = self._last_sample_at is None or now - self._last_sample_at > self._stale_after
        budget_mode = self._mode == "budget"
        if self._mode in NEUTRAL_MODES:
            state = "neutral"
        elif not budget_mode:
            state = "efficient"
        elif stale:
            state = "holding_no_signal"
        elif self._throttle > 0:
            state = "throttling"
        else:
            state = "within_budget"
        average = window["gpu_dram_watts"]
        return {
            "mode": self._mode,
            "budget_watts": self._budget,
            "state": state,
            "policy": self.policy.as_dict(),
            "signal": {
                "available": self.signal_reason is None,
                "reason": self.signal_reason,
                "stale": stale,
                "observed_samples": self._observed,
                "last_gpu_dram_watts": self._last_watts,
            },
            "window": {
                **window,
                "within_budget": (
                    average <= self._budget
                    if budget_mode and average is not None
                    else None
                ),
            },
            "throttle": self._throttle,
            "actuators": {
                "duty": self._duty_locked(),
                "lane_cap": self._lane_cap_locked(),
                "max_lanes": self.max_lanes,
                "idle_hold_ms": (
                    0.0
                    if self._mode in NEUTRAL_MODES or self.max_lanes == 1
                    else self.policy.efficient_hold_ms
                ),
                "owed_pace_seconds": self._owed,
            },
            "counts": dict(self._counts),
            "tracked_requests": len(self._accounts),
            "last_change": dict(self._last_change),
            "numerics": (
                "pacing is output-neutral; lane caps and idle holds change batch "
                "width, so batch-width numerics may differ"
            ),
        }

    def status(self) -> dict:
        with self._lock:
            return self._status_locked()
