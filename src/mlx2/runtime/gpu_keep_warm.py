"""Opt-in GPU keep-warm ticker for the serving worker's idle loop.

After roughly 1.5 s without GPU work an Apple GPU enters an idle power state,
and the next command buffer stalls for up to ~1.7 s (the stall grows with the
idle gap and the resident footprint).  Back-to-back benchmarks never see it;
every agent tool round or interactive turn does.  oMLX #3974 measured MiMo
pp512 TTFT 2.0 s -> 0.52 s after a 6 s idle by submitting a one-element
kernel every 0.5 s while nothing runs; CPU activity during the gap does not
help.

The policy here is default off.  When on, the serving worker calls
:meth:`GpuKeepWarm.note_work` on every round that has active lanes and
:meth:`GpuKeepWarm.maybe_tick` from its idle branch, on the same (MLX) thread,
so a tick never runs while a real request is being stepped.  Ticks run only
within ``window_seconds`` of the last real work: a server that has gone quiet
is left to sleep.  The op is a one-element add with no effect on any cache,
model state or output; the policy is host power management, not route
identity.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

#: ``engine.counts`` keys; present only when the policy is enabled.
COUNTERS = {
    "gpu_keep_warm_ticks": "tick",
    "gpu_keep_warm_idle_windows": "idle_window",
    "gpu_keep_warm_expired_windows": "expired_window",
    "gpu_keep_warm_warm_resumes": "warm_resume",
    "gpu_keep_warm_failures": "failure",
}


@dataclass(frozen=True)
class GpuKeepWarmPolicy:
    enabled: bool = False
    window_seconds: float = 120.0
    interval_seconds: float = 0.5

    @classmethod
    def from_value(cls, value) -> "GpuKeepWarmPolicy":
        """Parse ``None``/``False`` (off), ``True`` or an object."""
        if value is None or value is False:
            return cls()
        if value is True:
            return cls(enabled=True)
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError("gpu_keep_warm must be a boolean or an object")
        unknown = set(value) - {"enabled", "window_seconds", "interval_seconds"}
        if unknown:
            raise ValueError(
                "unknown gpu_keep_warm fields: " + ", ".join(sorted(unknown))
            )
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("gpu_keep_warm enabled must be boolean")
        window = _seconds(value.get("window_seconds", 120.0), "window_seconds")
        interval = _seconds(value.get("interval_seconds", 0.5), "interval_seconds")
        if not 0.05 <= interval <= 10:
            raise ValueError("gpu_keep_warm interval_seconds must be 0.05 to 10")
        if not interval <= window <= 86400:
            raise ValueError(
                "gpu_keep_warm window_seconds must be at least interval_seconds "
                "and at most 86400"
            )
        return cls(enabled=enabled, window_seconds=window, interval_seconds=interval)

    def as_dict(self) -> dict:
        return asdict(self)


def _seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"gpu_keep_warm {name} must be a number of seconds")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"gpu_keep_warm {name} must be finite and positive")
    return value


def _tiny_gpu_op():
    """Submit and wait for one one-element kernel on the default device."""
    import mlx.core as mx

    mx.eval(mx.add(mx.array(1.0), mx.array(1.0)))


class GpuKeepWarm:
    """Idle-loop scheduler for keep-warm ticks (worker thread only).

    ``counts`` is the engine's counter mapping.  ``clock`` and ``op`` are
    injectable for tests.  A failing op disables the ticker for the life of
    the process and is counted; it never fails the serving loop.
    """

    def __init__(self, policy, counts, *, clock=time.monotonic, op=_tiny_gpu_op):
        self.policy = GpuKeepWarmPolicy.from_value(policy)
        self.counts = counts
        self.clock = clock
        self.op = op
        self._last_work = None
        self._last_tick = None
        self._window_open = False
        self._ticked_in_window = False
        self._failed = False
        if self.policy.enabled:
            for key in COUNTERS:
                counts.setdefault(key, 0)

    @property
    def enabled(self) -> bool:
        return self.policy.enabled and not self._failed

    def note_work(self, now=None) -> None:
        """Record a round that stepped real work (any active lane)."""
        if not self.policy.enabled:
            return
        if self._window_open and self._ticked_in_window:
            # This request arrived while the ticker held the GPU awake.
            self.counts["gpu_keep_warm_warm_resumes"] += 1
        self._window_open = False
        self._ticked_in_window = False
        self._last_work = self.clock() if now is None else now

    def maybe_tick(self, now=None, *, pending_work=False) -> bool:
        """Run one tick if it is due; return whether one ran.

        ``pending_work`` is true when a request is already queued: it will
        wake the GPU itself, so the tick steps aside.
        """
        if not self.enabled or self._last_work is None:
            return False
        now = self.clock() if now is None else now
        if not self._window_open:
            self._window_open = True
            self.counts["gpu_keep_warm_idle_windows"] += 1
        if now - self._last_work > self.policy.window_seconds:
            # The window ran out: stop until real work starts a new one.
            self.counts["gpu_keep_warm_expired_windows"] += 1
            self._last_work = None
            self._window_open = False
            self._ticked_in_window = False
            return False
        if pending_work:
            return False
        # The GPU was busy until the last real work, so the first tick of a
        # window is due one interval after it.
        last = self._last_work
        if self._last_tick is not None and self._last_tick > last:
            last = self._last_tick
        if now - last < self.policy.interval_seconds:
            return False
        try:
            self.op()
        except Exception:  # keep-warm is advisory; serving must not fail
            self._failed = True
            self.counts["gpu_keep_warm_failures"] += 1
            return False
        self._last_tick = now
        self._ticked_in_window = True
        self.counts["gpu_keep_warm_ticks"] += 1
        return True

    def status(self) -> dict:
        return {
            **self.policy.as_dict(),
            "active": self.enabled,
            "failed": self._failed,
            "counts": {
                event: int(self.counts.get(key, 0)) for key, event in COUNTERS.items()
            },
        }
