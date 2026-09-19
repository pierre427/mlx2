"""Bounded host-side runtime telemetry."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from threading import RLock
from time import monotonic_ns
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    timestamp_ns: int
    name: str
    fields: Mapping[str, Any]


class RuntimeTelemetry:
    schema = "mlx2.runtime.v1"
    overflow_event = "__other__"

    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._events: deque[RuntimeEvent] = deque(maxlen=capacity)
        self._counts: Counter[str] = Counter()
        self._counts_capacity = capacity
        self._lock = RLock()

    def emit(self, name: str, **fields: Any) -> RuntimeEvent:
        event = RuntimeEvent(monotonic_ns(), name, MappingProxyType(dict(fields)))
        with self._lock:
            self._events.append(event)
            counter_name = name
            if name not in self._counts and len(self._counts) >= self._counts_capacity:
                counter_name = self.overflow_event
            self._counts[counter_name] += 1
        return event

    def snapshot(self) -> tuple[RuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def counts(self) -> Mapping[str, int]:
        with self._lock:
            return MappingProxyType(dict(self._counts))
