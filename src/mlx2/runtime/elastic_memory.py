# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral memory lending at explicit execution boundaries.

Only registered reconstructible resources are resized. Claims protect live
KV, temporary workspaces, or other owners; no cache state is discarded here.
Logical budgets are not evidence of physical reclamation. Admission must
remeasure its backend after a resize. See Basalt provenance (2026-10-24).
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass


def byte_count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer byte count")
    return value


class MemoryBudgetRefused(RuntimeError):
    pass


@dataclass
class ReclaimableMemory:
    minimum_bytes: int
    maximum_bytes: int
    resize: Callable[[int], int]
    budget_bytes: int | None = None

    def __post_init__(self):
        byte_count(self.minimum_bytes, "minimum_bytes")
        byte_count(self.maximum_bytes, "maximum_bytes")
        if self.minimum_bytes > self.maximum_bytes or not callable(self.resize):
            raise ValueError("invalid reclaimable memory resource")
        if self.budget_bytes is None:
            self.budget_bytes = self.maximum_bytes
        byte_count(self.budget_bytes, "budget_bytes")
        if not self.minimum_bytes <= self.budget_bytes <= self.maximum_bytes:
            raise ValueError("resource budget outside bounds")


class ElasticMemoryCoordinator:
    """One execution owner, named persistent claims and scoped workspace loans.

    Resizing fails closed: a backend callback failure poisons the coordinator,
    because it may have partially changed ownership. No loan is granted and no
    subsequent allocation may infer that the former budget still holds.
    """

    def __init__(self, total_bytes, resources, *, synchronize=lambda: None):
        self.total_bytes = byte_count(total_bytes, "total_bytes")
        self.resources = dict(resources)
        if not self.resources or any(
            not isinstance(k, str) or not k or not isinstance(v, ReclaimableMemory)
            for k, v in self.resources.items()
        ):
            raise ValueError("resources must be named reclaimable memory providers")
        if len({id(r) for r in self.resources.values()}) != len(
            self.resources
        ) or not callable(synchronize):
            raise ValueError(
                "resources must have distinct ownership and a synchronization callback"
            )
        if sum(r.maximum_bytes for r in self.resources.values()) > self.total_bytes:
            raise ValueError("maximum resource budgets exceed total")
        self.synchronize = synchronize
        self.claims = {}
        self._scoped = set()
        self._owner = None
        self._executing = 0
        self._lock = threading.RLock()
        self.error = None
        self.revision = 0
        self._targets = {name: r.budget_bytes for name, r in self.resources.items()}

    def bind_owner(self):
        owner = threading.get_ident()
        if self._owner is not None and self._owner != owner:
            raise MemoryBudgetRefused(
                "elastic memory belongs to another execution thread"
            )
        self._owner = owner

    def _boundary(self):
        self.bind_owner()
        if self.error:
            raise MemoryBudgetRefused(f"elastic memory provider failed: {self.error}")
        if self._executing:
            raise MemoryBudgetRefused("cannot resize inside an execution window")

    @contextmanager
    def execution_window(self):
        with self._lock:
            self._boundary()
            self._executing += 1
        try:
            yield
        finally:
            with self._lock:
                self._executing -= 1

    def set_claim(self, owner, size_bytes):
        with self._lock:
            self._boundary()
            if not isinstance(owner, str) or not owner:
                raise ValueError("claim owner must be a nonempty string")
            if owner in self._scoped:
                raise MemoryBudgetRefused("cannot replace a scoped workspace claim")
            byte_count(size_bytes, "claim")
            proposed = {**self.claims, owner: size_bytes}
            floor = sum(r.minimum_bytes for r in self.resources.values())
            room = self.total_bytes - sum(proposed.values())
            if room < floor:
                raise MemoryBudgetRefused(
                    "claim would consume protected resource floors"
                )
            span = sum(
                r.maximum_bytes - r.minimum_bytes for r in self.resources.values()
            )
            extra = min(span, room - floor)
            targets = {
                name: r.minimum_bytes
                + (extra * (r.maximum_bytes - r.minimum_bytes) // span if span else 0)
                for name, r in self.resources.items()
            }
            changes = [
                (name, r, targets[name])
                for name, r in self.resources.items()
                if targets[name] != self._targets[name]
            ]
            if changes:
                try:
                    self.synchronize()
                    # Reclaim before granting any growth to another provider.
                    for name, resource, target in sorted(
                        changes, key=lambda item: item[2] - item[1].budget_bytes
                    ):
                        actual = byte_count(resource.resize(target), "provider result")
                        if not resource.minimum_bytes <= actual <= target:
                            raise MemoryBudgetRefused(
                                f"{name} returned an unsafe resource budget"
                            )
                        resource.budget_bytes = actual
                except BaseException as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    raise
                self.revision += 1
                self._targets = targets
            self.claims = {key: value for key, value in proposed.items() if value}
            return self.receipt()

    @contextmanager
    def borrow(self, owner, size_bytes):
        if owner in self.claims or owner in self._scoped:
            raise MemoryBudgetRefused("workspace owner already has a claim")
        self.set_claim(owner, size_bytes)
        self._scoped.add(owner)
        try:
            yield
        finally:
            self._scoped.remove(owner)
            self.set_claim(owner, 0)

    def receipt(self):
        with self._lock:
            return {
                "schema": "mlx2.elastic-memory.v1",
                "revision": self.revision,
                "total_bytes": self.total_bytes,
                "claims": dict(self.claims),
                "resources": {
                    name: {
                        "minimum_bytes": r.minimum_bytes,
                        "maximum_bytes": r.maximum_bytes,
                        "budget_bytes": r.budget_bytes,
                    }
                    for name, r in self.resources.items()
                },
                "failed": self.error,
                "qualification": "unqualified",
            }


class ElasticHeadroom:
    """Measured admission adapter; reserve changes are charged exactly once.

    ``required`` includes the provider's current unfilled reservation. Extract
    that term before rebalancing, then return the new requirement. The next
    headroom sample must observe the backend; logical evictions add no credit.
    """

    def __init__(self, measure, unfilled_reserve, rebalance):
        self.measure = measure
        self.unfilled_reserve = unfilled_reserve
        self.rebalance = rebalance

    def __call__(self):
        return self.measure()

    def prepare(self, required):
        if (
            isinstance(required, bool)
            or not isinstance(required, (int, float))
            or not math.isfinite(required)
            or required < 0
        ):
            raise ValueError("required headroom must be finite and nonnegative")
        free = self.measure()
        byte_count(free, "measured headroom")
        before = byte_count(self.unfilled_reserve(), "unfilled reserve")
        protected = max(0, math.ceil(required) - before)
        self.rebalance(available_bytes=free, protected_bytes=protected)
        return protected + byte_count(self.unfilled_reserve(), "unfilled reserve")
