"""Event totals over the last 10 seconds, minute, 5 minutes and 30 minutes.

Lifetime counters say how much happened; they cannot say whether it happened
while a request was slow.  A ``RollingCounter`` keeps timestamped events for
the longest window and sums them per window on demand, so ``/v1/status`` can
show, for example, how many APCv2 entries were spilled to disk in the last
minute and how long those spills blocked the model thread.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable

WINDOWS = (("10s", 10.0), ("1m", 60.0), ("5m", 300.0), ("30m", 1800.0))
HORIZON = max(seconds for _, seconds in WINDOWS)
# A bound on retained events, far above any real spill or restore rate over
# 30 minutes.  When it is reached the oldest events drop first, so the short
# windows stay exact and the longest ones may undercount (reported).
MAX_EVENTS = 100_000


class RollingCounter:
    """Thread-safe windowed sums of numeric event fields."""

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        max_events: int = MAX_EVENTS,
    ):
        self._clock = clock
        self._events: deque = deque()
        self._max_events = int(max_events)
        self._dropped = 0
        self._lock = threading.Lock()

    def add(self, **fields: float) -> None:
        """Record one event; every field is summed per window (``count`` is implicit)."""
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            if len(self._events) >= self._max_events:
                self._events.popleft()
                self._dropped += 1
            self._events.append((now, {k: float(v) for k, v in fields.items()}))

    def snapshot(self) -> dict:
        """``{"10s": {"count": n, <field>: sum, ...}, "1m": ..., "5m": ..., "30m": ...}``."""
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            events = tuple(self._events)
            dropped = self._dropped
        out = {}
        for label, seconds in WINDOWS:
            totals: dict = {"count": 0}
            for stamp, fields in events:
                if now - stamp <= seconds:
                    totals["count"] += 1
                    for key, value in fields.items():
                        totals[key] = totals.get(key, 0.0) + value
            out[label] = {
                key: (round(value, 6) if isinstance(value, float) else value)
                for key, value in totals.items()
            }
        if dropped:
            out["events_dropped_at_cap"] = dropped
        return out

    def _prune_locked(self, now: float) -> None:
        while self._events and now - self._events[0][0] > HORIZON:
            self._events.popleft()
