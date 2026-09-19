# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
import threading


class CacheCapsuleGeneration:
    """Small generation authority shared by APC invalidation and workers."""

    def __init__(self, initial: int = 0):
        self._value = int(initial)
        self._lock = threading.Lock()

    @property
    def current(self) -> int:
        with self._lock:
            return self._value

    def advance(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
