"""Report a device evaluation that never returns.

A wedged GPU (apc-warm-hit-hang-20261021: a ``fence_wait`` spin kernel under
``MLX_METAL_FAST_SYNCH=1`` whose producer never ran) leaves the generation
thread blocked inside ``mx.eval`` with no error, the GPU at 100%, and the
service answering ``/health`` with ``ok`` for as long as it lives.  MLX has no
way to interrupt an evaluation and Metal no way to cancel a resident kernel,
so the watchdog cannot recover the eval.  What it can do:

* mark the service unhealthy (``/health`` 503, ``status()["healthy"]`` false)
  with a receipt naming the phase, its lanes and how long it has run;
* log the generation thread's Python stack once, so the wedge is diagnosable
  without ``sample``;
* optionally end the process (``MLX2_EVAL_WATCHDOG_ABORT=1``) after a
  faulthandler dump, for a supervisor that restarts it.  Off by default: an
  orphaned spin kernel can outlive its process (ml-explore/mlx#3830), so a
  restart is not guaranteed to help.

The bound is ``MLX2_EVAL_WATCHDOG_SECONDS`` (default 600; ``0`` disables).
A phase that returns after the watchdog tripped clears the stall and logs
the recovery, so a legitimately long evaluation never poisons the service.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import sys
import threading
import time
import traceback

log = logging.getLogger(__name__)

DEFAULT_SECONDS = 600.0
SECONDS_ENV = "MLX2_EVAL_WATCHDOG_SECONDS"
ABORT_ENV = "MLX2_EVAL_WATCHDOG_ABORT"
ABORT_EXIT_CODE = 70


class EvalWatchdog:
    """Watch one thread's device phases; report one that overruns its bound."""

    def __init__(
        self,
        seconds: float,
        *,
        model_label: str = "",
        abort: bool = False,
        interval: float | None = None,
        stop_event: threading.Event | None = None,
    ):
        self.seconds = float(seconds)
        self.model_label = model_label
        self.abort = bool(abort)
        self.interval = float(interval) if interval else max(0.05, min(5.0, self.seconds / 4))
        self._stop = stop_event or threading.Event()
        self._lock = threading.Lock()
        self._phase: dict | None = None
        self._stall: dict | None = None
        self._thread: threading.Thread | None = None
        self._watched: threading.Thread | None = None
        self.trips = 0
        self.recoveries = 0

    @classmethod
    def from_environment(cls, **kwargs) -> EvalWatchdog:
        raw = os.environ.get(SECONDS_ENV, "")
        try:
            seconds = float(raw) if raw.strip() else DEFAULT_SECONDS
        except ValueError:
            log.warning("%s=%r is not a number; using %s", SECONDS_ENV, raw, DEFAULT_SECONDS)
            seconds = DEFAULT_SECONDS
        abort = os.environ.get(ABORT_ENV, "0") == "1"
        return cls(max(0.0, seconds), abort=abort, **kwargs)

    @property
    def enabled(self) -> bool:
        return self.seconds > 0

    # Generation-thread side -------------------------------------------------

    def enter(self, phase: str, *, lanes: int = 0) -> None:
        """The watched thread is about to block in a device evaluation."""
        if not self.enabled:
            return
        with self._lock:
            self._phase = {"phase": str(phase), "lanes": int(lanes), "started": time.monotonic()}

    def leave(self) -> None:
        """The device evaluation returned; clear any stall it had raised."""
        if not self.enabled:
            return
        with self._lock:
            phase, self._phase = self._phase, None
            stall, self._stall = self._stall, None
        if stall is not None:
            elapsed = time.monotonic() - (phase or stall)["started"]
            self.recoveries += 1
            log.error(
                "eval watchdog: %s returned after %.1f s (bound %.0f s); service healthy again",
                stall["phase"], elapsed, self.seconds,
            )

    # Monitor side -----------------------------------------------------------

    def watch(self, thread: threading.Thread) -> None:
        """Start the monitor for ``thread``; a no-op when disabled."""
        if not self.enabled or self._thread is not None:
            return
        self._watched = thread
        self._thread = threading.Thread(
            target=self._monitor, name="mlx2-eval-watchdog", daemon=True
        )
        self._thread.start()

    def check(self, now: float | None = None) -> dict | None:
        """One monitor pass: trip on an overrun phase.  Returns the stall, if any."""
        now = time.monotonic() if now is None else now
        with self._lock:
            phase = self._phase
            if phase is None or self._stall is not None:
                return self._stall
            elapsed = now - phase["started"]
            if elapsed <= self.seconds:
                return None
            self._stall = {
                **phase,
                "elapsed_seconds": round(elapsed, 1),
                "bound_seconds": self.seconds,
                "model": self.model_label,
                "tripped_at": time.time(),
            }
            stall = dict(self._stall)
        self.trips += 1
        log.error(
            "eval watchdog: %s has not returned after %.1f s (bound %.0f s, %d lanes, %s); "
            "service reported unhealthy; the device evaluation cannot be interrupted",
            stall["phase"], stall["elapsed_seconds"], self.seconds, stall["lanes"], self.model_label,
        )
        stack = self._watched_stack()
        if stack:
            log.error("eval watchdog: generation thread stack:\n%s", stack)
        if self.abort:
            log.critical("eval watchdog: %s=1, ending the process (exit %d)", ABORT_ENV, ABORT_EXIT_CODE)
            try:
                faulthandler.dump_traceback(all_threads=True)
            finally:
                os._exit(ABORT_EXIT_CODE)
        return stall

    def _monitor(self) -> None:
        while not self._stop.wait(self.interval):
            watched = self._watched
            if watched is not None and not watched.is_alive():
                return
            try:
                self.check()
            except Exception:
                log.exception("eval watchdog: check failed")

    def _watched_stack(self) -> str:
        watched = self._watched
        if watched is None or watched.ident is None:
            return ""
        frame = sys._current_frames().get(watched.ident)
        if frame is None:
            return ""
        return "".join(traceback.format_stack(frame))

    # Reporting --------------------------------------------------------------

    @property
    def stalled(self) -> bool:
        with self._lock:
            return self._stall is not None

    def status(self) -> dict:
        with self._lock:
            phase = dict(self._phase) if self._phase else None
            stall = dict(self._stall) if self._stall else None
        if phase is not None:
            phase["elapsed_seconds"] = round(time.monotonic() - phase.pop("started"), 1)
        if stall is not None:
            stall.pop("started", None)
        return {
            "enabled": self.enabled,
            "bound_seconds": self.seconds,
            "abort_on_trip": self.abort,
            "phase": phase,
            "stall": stall,
            "trips": self.trips,
            "recoveries": self.recoveries,
        }
