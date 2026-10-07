"""Opt-in serving-loop timeline trace (diagnostics only; off by default).

Set ``MLX2_LOOP_TRACE=/path/trace.jsonl`` to make the generation worker write
one JSON line per serving-loop iteration: wall-clock phase boundaries
(``time.perf_counter``, which on macOS is the host-wide mach clock, so a
client in another process can align its token arrivals with it) and the
scheduler/APC events that ran inside the iteration (decode phases, prefill
forwards, APC stores, spills and restores, each with its duration).

It exists to locate where a decoding neighbour's token gap is spent when the
scheduler's own gap attribution does not see it.  With the variable unset
every hook is a single attribute check.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

_PATH = os.environ.get("MLX2_LOOP_TRACE") or None
_events: list[tuple] = []
_lock = threading.Lock()
_handle = None


def enabled() -> bool:
    return _PATH is not None


def event(_name: str, start: float, end: float | None = None, **fields: Any) -> None:
    """Record one event; ``start``/``end`` are ``time.perf_counter`` values."""
    if _PATH is None:
        return
    record = {"k": _name, "t0": round(start, 6)}
    if end is not None:
        record["ms"] = round((end - start) * 1000.0, 3)
    record.update(fields)
    with _lock:
        _events.append(record)


def flush(iteration: dict | None) -> None:
    """Write one loop iteration with every event recorded since the last one."""
    global _handle
    if _PATH is None or iteration is None:
        return
    with _lock:
        events = list(_events)
        _events.clear()
    iteration = dict(iteration)
    iteration["events"] = events
    try:
        if _handle is None:
            _handle = open(_PATH, "a", buffering=1)  # noqa: SIM115 - process-lifetime log
        _handle.write(json.dumps(iteration, default=str) + "\n")
    except OSError:
        pass


def now() -> float:
    return time.perf_counter()
