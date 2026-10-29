# SPDX-License-Identifier: Apache-2.0
"""Quiescent immutable-model residency, independent of prefix-cache math.

Providers own file formats, device allocation and fresh host-memory admission.
The serving evaluation owner supplies synchronization and physical measurements.
An optional FIFO reservation pool coordinates in-process owners; it does not
schedule GPU execution, evict active owners or replace CPG/paired-lock ownership.
"""

from collections import deque
from dataclasses import dataclass
import json
import math
import threading
import time


def _bytes(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be nonnegative integer bytes")
    return value


def _binding(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded) > 65536 or not isinstance(value, dict) or not value:
        raise ValueError("model residency needs a bounded source/profile binding")
    return json.loads(encoded)


@dataclass(eq=False)
class Reservation:
    pool: object
    owner: object
    bytes: int
    deadline: float
    granted: bool = False
    closed: bool = False


class ResidencyBudget:
    """Nonblocking FIFO capacity reservations; active owners are protected."""

    def __init__(self, total_bytes, *, max_waiters=32, clock=time.monotonic):
        self.total_bytes = _bytes(total_bytes, "total_bytes")
        if not self.total_bytes or type(max_waiters) is not int or not 1 <= max_waiters <= 1024:
            raise ValueError("invalid residency budget bounds")
        self.max_waiters, self.clock = max_waiters, clock
        self._lock = threading.Lock()
        self._queue, self._active = deque(), {}

    def request(self, owner, size_bytes, *, timeout=60.0):
        size_bytes = _bytes(size_bytes, "size_bytes")
        if not 0 < size_bytes <= self.total_bytes:
            raise ValueError("reservation cannot fit residency budget")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError("reservation timeout must be finite and bounded")
        with self._lock:
            if len(self._queue) >= self.max_waiters:
                raise RuntimeError("residency waiter queue is full")
            if owner in self._active or any(item.owner == owner for item in self._queue):
                raise ValueError("residency owner already has a reservation")
            ticket = Reservation(self, owner, size_bytes, self.clock()+timeout)
            self._queue.append(ticket)
            return ticket

    def try_grant(self, ticket):
        with self._lock:
            if ticket.pool is not self or ticket.closed:
                raise ValueError("foreign or closed residency reservation")
            if ticket.granted:
                if self._active.get(ticket.owner) is not ticket:
                    raise ValueError("stale residency reservation")
                return True
            if ticket not in self._queue:
                raise ValueError("unknown residency reservation")
            if self.clock() >= ticket.deadline:
                self._queue.remove(ticket); ticket.closed = True
                raise TimeoutError("residency reservation expired")
            # Expired head waiters cannot indefinitely starve a live owner.
            while self._queue and self.clock() >= self._queue[0].deadline:
                expired = self._queue.popleft(); expired.closed = True
            available = self.total_bytes-sum(item.bytes for item in self._active.values())
            if not self._queue or self._queue[0] is not ticket or ticket.bytes > available:
                return False
            self._queue.popleft(); ticket.granted = True
            self._active[ticket.owner] = ticket
            return True

    def release(self, ticket):
        with self._lock:
            if ticket.pool is not self:
                raise ValueError("foreign residency reservation")
            if ticket.closed:
                return
            if ticket.granted:
                if self._active.get(ticket.owner) is not ticket:
                    raise ValueError("stale residency reservation")
                del self._active[ticket.owner]
            else:
                self._queue.remove(ticket)
            ticket.closed = True

    def receipt(self):
        with self._lock:
            reserved = sum(item.bytes for item in self._active.values())
            return dict(total_bytes=self.total_bytes, reserved_bytes=reserved,
                        available_bytes=self.total_bytes-reserved, active_owners=len(self._active),
                        waiting_owners=len(self._queue), policy="fifo; no active-owner eviction")


class ModelResidency:
    """Evaluation-owner transaction around an adapter's immutable weights."""

    def __init__(self, provider, *, identity, synchronize, measure, budget=None):
        required = ("binding", "resident_bytes", "park", "activate", "admit", "receipt",
                    "retirement_receipt")
        if any(not callable(getattr(provider, name, None)) for name in required):
            raise ValueError("adapter lacks a complete model residency provider")
        self.provider, self.identity = provider, _binding(identity)
        self.synchronize, self.measure = synchronize, measure
        self.budget, self._reservation = budget, None
        self.owner = threading.get_ident()
        self.binding = _binding(provider.binding())
        self.state, self.generation, self.last = "active", 0, None
        if budget is not None:
            required_bytes = _bytes(provider.reservation_bytes(), "reservation_bytes")
            self._reservation = budget.request(self, required_bytes)
            try:
                if not budget.try_grant(self._reservation):
                    raise MemoryError("loaded model cannot acquire shared residency capacity")
            except BaseException:
                budget.release(self._reservation); self._reservation = None
                raise

    def _owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("model residency crossed evaluation owner")

    def _check(self):
        if _binding(self.provider.binding()) != self.binding:
            raise ValueError("model residency source/profile changed")

    def _measure(self):
        value = self.measure()
        if not isinstance(value, dict):
            raise ValueError("physical memory measurement must be a mapping")
        return {name: _bytes(size, name) for name, size in value.items()}

    def park(self):
        self._owner()
        if self.state == "parked":
            self._check()
            return self.receipt()
        if self.state != "active":
            raise RuntimeError("model residency is not active")
        self._check(); self.synchronize()
        before, logical = self._measure(), _bytes(self.provider.resident_bytes(), "resident_bytes")
        started = time.monotonic()
        try:
            detail = self.provider.park()
            self.synchronize(); self._check()
            if _bytes(self.provider.resident_bytes(), "resident_bytes"):
                raise RuntimeError("model provider retained payload after park")
            after = self._measure()
        except BaseException as error:
            self.state = "failed"
            self.last = dict(operation="park", status="failed", error=f"{type(error).__name__}: {error}")
            raise
        if self._reservation is not None:
            self.budget.release(self._reservation); self._reservation = None
        self.state = "parked"; self.generation += 1
        self.last = dict(operation="park", status="completed", seconds=time.monotonic()-started,
                         logical_payload_bytes_before=logical, logical_payload_bytes_after=0,
                         memory_before=before, memory_after=after, provider=detail)
        return self.receipt()

    def activate(self):
        self._owner()
        if self.state == "active":
            self._check()
            return self.receipt()
        if self.state != "parked":
            raise RuntimeError("failed park requires a complete model reload")
        try:
            self._check()
            if self.budget is not None:
                if self._reservation is None:
                    self._reservation = self.budget.request(self, _bytes(self.provider.reservation_bytes(), "reservation_bytes"))
                if not self.budget.try_grant(self._reservation):
                    return {**self.receipt(), "status": "waiting_for_budget"}
        except BaseException as error:
            if self._reservation is not None:
                self.budget.release(self._reservation); self._reservation = None
            self.last = dict(operation="activate", status="refused", error=f"{type(error).__name__}: {error}")
            raise
        started = time.monotonic()
        try:
            before = self._measure()
            self.provider.admit()  # Fresh host/advisory check immediately before allocation.
            detail = self.provider.activate()
            self.synchronize(); self._check()
            logical = _bytes(self.provider.resident_bytes(), "resident_bytes")
            after = self._measure()
        except BaseException as error:
            # A failed activation may have allocated a partial working set.
            # Re-park before releasing capacity; uncertain cleanup stays closed.
            try:
                self.provider.park(); self.synchronize()
                if _bytes(self.provider.resident_bytes(), "resident_bytes"):
                    raise RuntimeError("failed activation retained model payload")
            except BaseException:
                self.state = "failed"
            else:
                if self._reservation is not None:
                    self.budget.release(self._reservation); self._reservation = None
            self.last = dict(operation="activate", status="failed", error=f"{type(error).__name__}: {error}")
            raise
        self.state = "active"; self.generation += 1
        self.last = dict(operation="activate", status="completed", seconds=time.monotonic()-started,
                         logical_payload_bytes_after=logical, memory_before=before,
                         memory_after=after, provider=detail)
        return self.receipt()

    def cancel_activation(self):
        self._owner()
        if self.state != "parked":
            raise RuntimeError("only a parked activation waiter can be cancelled")
        if self._reservation is not None:
            self.budget.release(self._reservation); self._reservation = None
        return self.receipt()

    def retire(self):
        """Release accounting after tensor and asynchronous ownership drains.

        Adapter.close() may detach its objects while a quarantined I/O worker
        still owns staging. The provider retains those owners and supplies an
        explicit proof; a zero tensor-payload estimate alone is insufficient.
        """
        self._owner()
        retirement = None
        try:
            self.synchronize()
            if _bytes(self.provider.resident_bytes(), "resident_bytes"):
                raise RuntimeError("closed adapter still retains model payload")
            retirement = _binding(self.provider.retirement_receipt())
            if type(retirement.get("ownership_drained")) is not bool:
                raise ValueError("retirement receipt must explicitly prove ownership drain")
            retained = _bytes(retirement.get("retained_bytes"), "retirement retained_bytes")
            if not retirement["ownership_drained"] or retained:
                raise RuntimeError("closed adapter still retains asynchronous ownership")
        except BaseException as error:
            self.state = "failed"
            self.last = dict(operation="retire", status="failed",
                             error=f"{type(error).__name__}: {error}", provider=retirement)
            raise
        if self._reservation is not None:
            self.budget.release(self._reservation); self._reservation = None
        self.state = "closed"
        self.last = dict(operation="retire", status="completed", provider=retirement)
        return self.receipt()

    def receipt(self):
        return dict(schema="mlx2.model-residency.v1", state=self.state, generation=self.generation,
                    identity=self.identity, binding=self.binding, last=self.last,
                    budget=self.budget.receipt() if self.budget is not None else None,
                    qualification="unqualified")
