# SPDX-License-Identifier: MIT
"""Read-ahead hints for hashed-row SSD reads (PLE and Engram tables).

A hint tells the kernel which byte ranges a reader is about to ``pread`` so
the device sees them at once instead of one ``pread`` at a time: Darwin's
``fcntl(F_RDADVISE)``, ``posix_fadvise(POSIX_FADV_WILLNEED)`` elsewhere.  A
refused hint is never an error; the rows are read the ordinary way and are
byte-identical either way.  Every caller keeps the hint opt-in and default
off until an A/B shows it helps on that reader (recon 2026-11-13 found it
slower on multi-worker PLE reads).

Mechanism first seen in omlx #5225 (PLE) and #5274/#5464 (Engram
``MADV_WILLNEED`` before row copies, batched across a verify round); see
docs/PROVENANCE.md.  ``ReadAdvisor`` keeps cheap integer counters only; wall
clock is taken only when ``MLX_ENGRAM_READ_ADVISE_TIMING=1``.
"""

from __future__ import annotations

import os
import struct
import sys
import threading
import time

F_RDADVISE = 44  # <sys/fcntl.h>; struct radvisory { off_t ra_offset; int ra_count; }
TIMING_ENV = "MLX_ENGRAM_READ_ADVISE_TIMING"


def strict_env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if raw == "1":
        return True
    if raw == "0":
        return False
    raise ValueError(f"{name} must be 0 or 1")


def read_advise_backend() -> str | None:
    if sys.platform == "darwin":
        return "f_rdadvise"
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_WILLNEED"):
        return "posix_fadvise"
    return None


def advise_spans(fd, backend: str, spans) -> int:
    """Hint every ``(offset, length)`` span; return how many were accepted.

    One syscall per span.  Darwin's F_RDADVISEV (up to 64 ranges per call)
    would batch these; not used yet.
    """
    import fcntl

    advised = 0
    for offset, length in spans:
        try:
            if backend == "f_rdadvise":
                fcntl.fcntl(fd, F_RDADVISE, struct.pack("qi4x", int(offset), int(length)))
            else:
                os.posix_fadvise(fd, int(offset), int(length), os.POSIX_FADV_WILLNEED)
            advised += 1
        except (OSError, ValueError, OverflowError):
            pass
    return advised


def advise_rows(fd, backend: str, offsets, row_bytes: int) -> int:
    """Hint ``row_bytes`` at each offset (the PLE row shape)."""
    return advise_spans(fd, backend, ((offset, row_bytes) for offset in offsets))


class ReadAdvisor:
    """One reader's opt-in hint switch and its counters.

    ``env`` is the reader's strict 0/1 switch (default off).  When on and the
    platform has no backend the reader is refused at construction rather than
    silently serving without the hint it was asked for.
    """

    def __init__(self, env: str, *, enabled: bool | None = None):
        self.env = env
        self.enabled = strict_env_flag(env, False) if enabled is None else bool(enabled)
        self.backend = read_advise_backend() if self.enabled else None
        if self.enabled and self.backend is None:
            raise ValueError(f"{env}=1 needs F_RDADVISE or posix_fadvise on this platform")
        self.timing = self.enabled and strict_env_flag(TIMING_ENV, False)
        self._lock = threading.Lock()
        self.calls = 0
        self.rows_advised = 0
        self.spans_requested = 0
        self.spans_accepted = 0
        self.failures = 0
        self.ns = 0

    def advise(self, fd, spans, rows: int | None = None) -> int:
        """Hint ``spans`` (``(offset, length)`` pairs covering ``rows`` rows).

        Returns the accepted span count; a refused span counts as a failure
        and is never raised.
        """
        spans = list(spans)
        if not self.enabled or not spans:
            return 0
        started = time.perf_counter_ns() if self.timing else 0
        accepted = advise_spans(fd, self.backend, spans)
        elapsed = time.perf_counter_ns() - started if self.timing else 0
        with self._lock:
            self.calls += 1
            self.rows_advised += len(spans) if rows is None else int(rows)
            self.spans_requested += len(spans)
            self.spans_accepted += accepted
            self.failures += len(spans) - accepted
            self.ns += elapsed
        return accepted

    def record_failure(self, spans: int) -> None:
        """Count spans whose hint could not even be queued."""
        with self._lock:
            self.failures += int(spans)

    def status(self) -> dict:
        with self._lock:
            out = {"configured": self.enabled, "qualified": False, "default": False,
                   "env": self.env, "backend": self.backend, "calls": self.calls,
                   "rows_advised": self.rows_advised,
                   "spans_requested": self.spans_requested,
                   "spans_accepted": self.spans_accepted, "failures": self.failures}
            if self.timing:
                out["advise_ns"] = self.ns
            return out
