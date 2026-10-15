"""Small host-only counter owner for submission and model-worker telemetry.

Compound updates use add/maximum; locking __getitem__ and __setitem__ alone
would not make ``mapping[key] += n`` atomic. Read a coherent report through
snapshot, without holding the serving lifecycle lock.
"""

from collections import Counter
from collections.abc import MutableMapping
from threading import Lock


class HostCounters(MutableMapping):
    def __init__(self, initial=None):
        self._values = Counter(initial or {})
        self._lock = Lock()

    def __getitem__(self, key):
        with self._lock:
            return self._values[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._values[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._values[key]

    def __iter__(self):
        return iter(self.snapshot())

    def __len__(self):
        with self._lock:
            return len(self._values)

    def __contains__(self, key):
        with self._lock:
            return key in self._values

    def get(self, key, default=None):
        with self._lock:
            return self._values.get(key, default)

    def add(self, key, amount=1):
        with self._lock:
            self._values[key] += amount
            return self._values[key]

    def maximum(self, key, value):
        with self._lock:
            self._values[key] = max(self._values[key], value)
            return self._values[key]

    def setdefault(self, key, default=0):
        with self._lock:
            return self._values.setdefault(key, default)

    def update(self, iterable=None, /, **kwargs):
        """Counter-compatible additive update, atomic for this owner.

        Stage caller iterables before taking the owner lock: a generator must
        not execute arbitrary code while holding the telemetry lock, and a
        self-update must not recursively acquire it.
        """
        additions = Counter()
        if iterable is self:
            iterable = self.snapshot()
        if iterable is not None:
            additions.update(iterable)
        additions.update(kwargs)
        with self._lock:
            self._values.update(additions)

    def snapshot(self):
        with self._lock:
            return dict(self._values)


def count_add(counters, key, amount=1):
    """Atomic on the engine owner; keep plain mappings usable in host fixtures."""
    if isinstance(counters, HostCounters):
        return counters.add(key, amount)
    counters[key] = counters.get(key, 0) + amount
    return counters[key]


def count_max(counters, key, value):
    if isinstance(counters, HostCounters):
        return counters.maximum(key, value)
    counters[key] = max(counters.get(key, 0), value)
    return counters[key]


def counter_snapshot(counters):
    return counters.snapshot() if isinstance(counters, HostCounters) else dict(counters)
