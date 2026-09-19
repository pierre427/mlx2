"""Identity-safe cache ownership with invalidation leases."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable
import weakref

from .contracts import Fidelity, StatePlane


@dataclass(frozen=True, slots=True)
class CacheFingerprint:
    schema_version: str
    plane: StatePlane
    model_revision: str
    adapter_revision: str | None
    configuration_hash: str
    tokenizer_hash: str
    template_hash: str
    layout: str
    token_hash: str
    segment_start: int
    segment_end: int
    fidelity: Fidelity = Fidelity.EXACT

    def __post_init__(self) -> None:
        required = {
            "schema_version": self.schema_version,
            "model_revision": self.model_revision,
            "configuration_hash": self.configuration_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "template_hash": self.template_hash,
            "layout": self.layout,
            "token_hash": self.token_hash,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"empty cache identity fields: {', '.join(missing)}")
        if self.segment_start < 0 or self.segment_end <= self.segment_start:
            raise ValueError("cache segment must be a non-empty half-open range")


class CacheMiss(LookupError):
    pass


class CacheLease:
    def __init__(self, value: Any, release: Callable[[], None]) -> None:
        self.value = value
        self._finalizer = weakref.finalize(self, release)

    def close(self) -> None:
        self._finalizer()

    def __enter__(self) -> Any:
        return self.value

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(slots=True)
class _Entry:
    value: Any
    pins: int = 0
    invalidated: bool = False


class CacheOwner:
    def __init__(self) -> None:
        self._lock = RLock()
        self._entries: dict[CacheFingerprint, _Entry] = {}

    def put(self, key: CacheFingerprint, value: Any) -> None:
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None and existing.pins:
                raise RuntimeError("cannot replace a leased cache entry")
            self._entries[key] = _Entry(value=value)

    def acquire(self, key: CacheFingerprint) -> CacheLease:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.invalidated:
                raise CacheMiss(key)
            entry.pins += 1

        def release() -> None:
            with self._lock:
                current = self._entries.get(key)
                if current is not entry:
                    return
                current.pins -= 1
                if current.pins == 0 and current.invalidated:
                    del self._entries[key]

        return CacheLease(entry.value, release)

    def invalidate(self, key: CacheFingerprint) -> bool:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False
            entry.invalidated = True
            if entry.pins == 0:
                del self._entries[key]
            return True

    def __len__(self) -> int:
        with self._lock:
            return sum(not entry.invalidated for entry in self._entries.values())
