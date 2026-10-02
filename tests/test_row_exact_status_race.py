"""RowExactVerify.status() copies its counters under the lock (sweep
2026-10-02 V3).  The generation worker's _close() inserts stage/route keys
under ``_lock`` while /v1/status runs status() on an HTTP thread; iterating
the dict unlocked raised "dictionary changed size during iteration"."""

import sys
import threading
import weakref

from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.models.qwen4_row_exact import RowExactVerify


def _handle():
    h = object.__new__(RowExactVerify)
    h.model = None
    h.enabled = True
    h._lock = threading.Lock()
    h._swapped = []
    h._pending = None
    h._live = weakref.WeakSet()
    h.static_refusals = {}
    h._gdn = []
    h.counts = {"windows": 0, "windows_row_exact": 0, "windows_not_exact": 0,
                "rows": 0, "one_row_passthrough": 0, "stages": {}, "failures": {}}
    return h


def test_status_holds_the_lock():
    h = _handle()

    class Probe:
        held = []

        def __enter__(self):
            Probe.held.append(True)

        def __exit__(self, *exc):
            return False

    h._lock = Probe()
    h.status()
    assert Probe.held, "status() read the counters without taking _lock"


def test_status_does_not_race_close():
    h = _handle()
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    stop = threading.Event()

    def worker():
        i = 0
        while not stop.is_set():
            record = REV.Window(4)
            for j in range(50):
                record.note(f"stage{i}_{j}", "route")
            h._close(record)
            i += 1

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    errors = []
    try:
        for _ in range(3000):
            try:
                h.status()
            except RuntimeError as exc:
                errors.append(exc)
    finally:
        stop.set()
        t.join()
        sys.setswitchinterval(previous)
    assert not errors, errors[0]
