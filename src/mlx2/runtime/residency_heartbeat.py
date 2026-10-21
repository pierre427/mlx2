"""Keep the wired weights wired across idle pauses (a residency heartbeat).

Metal lets a residency set go about 2 s after the process's last command
buffer, requested or not.  MLX (0.32.4, ``mlx/backend/metal/resident.cpp``
and ``device.cpp``) puts its one wired set -- the weights, admitted by
``runtime/weight_residency.wire_serving_weights`` -- on *every* command
queue and calls ``requestResidency()`` once, when the set is built.  A
command buffer on a queue that holds a set waits until all of it is wired
again, so the first prefill after a pause of a few seconds pays the re-wire
of the whole model before its first kernel runs.

MLX exposes no ``requestResidency`` and no handle on the set, so this module
uses the smallest mechanism the API has: while the server is idle, a timer
thread commits one tiny command buffer (a one-element add) every
``interval_seconds`` on *its own* thread-local GPU stream.  That queue holds
the same set (MLX attaches it to every queue), so the set's last-use clock
never reaches Metal's ~2 s release and the weights stay wired.

Safety, by construction:

* Its own stream.  MLX gives every thread its own default stream
  (``default_stream_storage`` is ``thread_local``); the heartbeat builds and
  evaluates only on its thread, never on the generation stream, and records
  both stream indices in its status.
* No cross-stream fence.  The pulse's inputs are host-built constants with
  no producing stream; its output is consumed by nothing else, so no
  ``Fence`` (the ``MLX_METAL_FAST_SYNCH`` spin kernel of
  apc-warm-hit-hang-20261021, c46d82799) can form.
* No lock shared with generation.  It takes only its own condition lock,
  and never while it evaluates; ``mark`` (called by the serving loop) never
  waits on the GPU.
* No pulse while serving.  The serving loop marks itself busy while it has
  lanes; a busy engine already keeps the set wired, so steady-state decode
  and prefill never share the GPU with a pulse.
* Bounded.  It stops ``keep_seconds`` after the last activity, on memory
  pressure (kernel ``memorystatus`` level WARN or CRITICAL, polled at every
  beat, plus any engine pressure signal), on shutdown, and after any pulse
  error.  Only new activity re-arms it, so under pressure macOS can take an
  idle model's wired memory back as before.
* Nothing on the CPU backend, or when the weights were not wired.

Design input: Kapitsa (Apache-2.0) PR #206, ``src/metal/gpu.mm`` heartbeat
(a 500 ms GCD timer calling ``requestResidency`` for up to 600 s after the
last commit, stopped at memory-pressure WARN).  No code was taken; the
mechanism differs (a command buffer on a queue holding the set, because MLX
exposes no set handle).
"""

from __future__ import annotations

import re
import threading
import time
from typing import Callable, Optional

MAX_KEEP_SECONDS = 3600.0
#: Server default idle window (seconds); 0 is off.  On from the M5 idle-pause
#: measurement (qualification/runs/residency-heartbeat-20261021): after a
#: 3-90 s pause, first-token latency on an APCv2-hit prompt fell from
#: 183-204 to 70-78 ms (Qwen3.8-27B U4) and 260-337 to 37-45 ms (Qwen3.6-35B
#: 4-bit), with no pulse during decode and decode unchanged within noise.
DEFAULT_KEEP_SECONDS = 600.0
DEFAULT_INTERVAL_SECONDS = 0.5
MECHANISM = (
    "tiny command buffer on the heartbeat thread's own GPU stream "
    "(MLX attaches its wired residency set to every queue)"
)


def validate_keep_seconds(value) -> float:
    """The idle window in seconds: 0 disables, at most one hour."""
    if isinstance(value, bool):
        raise ValueError("residency keep seconds must be a number")
    try:
        keep = float(value)
    except (TypeError, ValueError):
        raise ValueError("residency keep seconds must be a number") from None
    if not (0.0 <= keep <= MAX_KEEP_SECONDS):
        raise ValueError(
            f"residency keep seconds must be from 0 to {MAX_KEEP_SECONDS:g}"
        )
    return keep


def _metal_available() -> bool:
    try:
        import mlx.core as mx

        return bool(mx.metal.is_available())
    except Exception:  # noqa: BLE001 - a non-Metal build has nothing to keep
        return False


def stream_index(stream) -> Optional[int]:
    """An MLX stream's index (``Stream(Device(gpu, 0), 3)`` -> 3), or None."""
    if stream is None:
        return None
    match = re.search(r",\s*(\d+)\)\s*$", repr(stream))
    return int(match.group(1)) if match else None


def metal_pulse_factory():
    """Build the pulse on the calling (heartbeat) thread.

    Returns ``(pulse, stream_index)``.  The stream is this thread's own
    default GPU stream; the inputs are evaluated host constants.
    """
    import mlx.core as mx

    stream = mx.default_stream(mx.gpu)
    one = mx.array(1, dtype=mx.uint32)
    mx.eval(one)

    def pulse() -> None:
        mx.eval(mx.add(one, one, stream=stream))

    return pulse, stream_index(stream)


def _default_pressure() -> int:
    from .os_memory import host_pressure_level

    level = host_pressure_level()
    return 0 if level is None else int(level)


class ResidencyHeartbeat:
    """Idle-window timer that keeps the wired set requested between requests."""

    def __init__(
        self,
        keep_seconds: float,
        *,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        pulse_factory: Optional[Callable[[], tuple]] = None,
        pressure: Optional[Callable[[], int]] = None,
        available: Optional[bool] = None,
        wired: bool = True,
        generation_stream_index: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.keep_seconds = validate_keep_seconds(keep_seconds)
        if not interval_seconds > 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = float(interval_seconds)
        self._pulse_factory = pulse_factory or metal_pulse_factory
        self._pressure = pressure or _default_pressure
        self._available = _metal_available() if available is None else bool(available)
        self._wired = bool(wired)
        self._clock = clock
        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._stopped = False
        self._armed = False
        self._busy = False
        self._last_activity = clock()
        self.generation_stream_index = generation_stream_index
        self.stream_index: Optional[int] = None
        self.state = "off" if self.keep_seconds == 0 else "configured"
        self.reason: Optional[str] = None
        self.pulses = 0
        self.pulse_ms_last: Optional[float] = None
        self.pulse_ms_max = 0.0
        self.idle_expiries = 0
        self.pressure_stops = 0
        self.restarts = 0
        self.last_pressure_level = 0
        self.error: Optional[str] = None

    # -- lifecycle -----------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start the timer thread; False when off or there is nothing to keep."""
        if self.keep_seconds == 0:
            self.state, self.reason = "off", "disabled by configuration"
            return False
        if not self._available:
            self.state, self.reason = "unavailable", "no Metal device (CPU backend)"
            return False
        if not self._wired:
            self.state, self.reason = "unavailable", "weights were not wired"
            return False
        with self._cond:
            if self._thread is not None:
                return True
            self._armed = True
            self._last_activity = self._clock()
            self.state, self.reason = "idle", None
            self._thread = threading.Thread(
                target=self._loop, name="mlx2-residency-heartbeat", daemon=True
            )
            self._thread.start()
        return True

    def stop(self, reason: str = "shutdown", timeout: float = 2.0) -> None:
        with self._cond:
            self._stopped = True
            if self.state not in ("off", "unavailable", "failed"):
                self.state, self.reason = "stopped", reason
            self._cond.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def mark(self, busy: bool) -> None:
        """Serving-loop hook: called each iteration with whether lanes exist.

        Never waits on the GPU: it takes only the heartbeat's own lock, which
        the timer never holds while it evaluates.
        """
        if self._thread is None:
            return
        busy = bool(busy)
        if not busy and not self._busy:
            return
        with self._cond:
            self._last_activity = self._clock()
            self._busy = busy
            if busy and not self._armed and not self._stopped:
                self._armed = True
                self.restarts += 1
                self.state, self.reason = "serving", None
                self._cond.notify_all()
            elif busy:
                self.state = "serving"
            else:
                self.state = "idle"

    # -- timer ---------------------------------------------------------------
    def _decide(self) -> bool:
        """One beat, under the lock: True when this beat should pulse."""
        if self._busy:
            return False
        level = int(self._pressure() or 0)
        self.last_pressure_level = level
        if level >= 1:
            self._armed = False
            self.pressure_stops += 1
            self.state, self.reason = "pressure", f"memory pressure level {level}"
            return False
        idle = self._clock() - self._last_activity
        if idle > self.keep_seconds:
            self._armed = False
            self.idle_expiries += 1
            self.state, self.reason = "idle_expired", f"idle over {self.keep_seconds:g} s"
            return False
        return idle >= self.interval_seconds

    def _loop(self) -> None:
        try:
            pulse, self.stream_index = self._pulse_factory()
            if (
                self.generation_stream_index is not None
                and self.stream_index == self.generation_stream_index
            ):
                raise RuntimeError("heartbeat stream is the generation stream")
            while True:
                with self._cond:
                    while not self._stopped and not self._armed:
                        self._cond.wait()
                    if self._stopped:
                        return
                    self._cond.wait(self.interval_seconds)
                    if self._stopped:
                        return
                    if not self._armed:
                        continue
                    fire = self._decide()
                if not fire:
                    continue
                started = time.perf_counter()
                pulse()
                elapsed = (time.perf_counter() - started) * 1000.0
                with self._cond:
                    self.pulses += 1
                    self.pulse_ms_last = round(elapsed, 3)
                    self.pulse_ms_max = max(self.pulse_ms_max, round(elapsed, 3))
                    if self.state not in ("stopped", "serving"):
                        self.state = "beating"
        except Exception as exc:  # noqa: BLE001 - a failed pulse turns it off
            with self._cond:
                self.state, self.error = "failed", f"{type(exc).__name__}: {exc}"
                self._armed = False

    def status(self) -> dict:
        with self._cond:
            idle = max(0.0, self._clock() - self._last_activity)
            return {
                "implemented": True,
                "qualified": False,
                "enabled": self.keep_seconds > 0,
                "state": self.state,
                "reason": self.reason,
                "running": self.running,
                "keep_seconds": self.keep_seconds,
                "interval_seconds": self.interval_seconds,
                "idle_seconds": round(idle, 3),
                "pulses": self.pulses,
                "pulse_ms_last": self.pulse_ms_last,
                "pulse_ms_max": self.pulse_ms_max,
                "idle_expiries": self.idle_expiries,
                "pressure_stops": self.pressure_stops,
                "restarts": self.restarts,
                "last_pressure_level": self.last_pressure_level,
                "stream_index": self.stream_index,
                "generation_stream_index": self.generation_stream_index,
                "mechanism": MECHANISM,
                "error": self.error,
            }


__all__ = [
    "DEFAULT_INTERVAL_SECONDS",
    "DEFAULT_KEEP_SECONDS",
    "MAX_KEEP_SECONDS",
    "ResidencyHeartbeat",
    "metal_pulse_factory",
    "stream_index",
    "validate_keep_seconds",
]
