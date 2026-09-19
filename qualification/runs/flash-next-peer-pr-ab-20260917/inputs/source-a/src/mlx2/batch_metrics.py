"""Bounded host-only serving telemetry.

The hot path records monotonic-clock timestamps and integer counters only.  It
never reads or synchronizes device state; callers may attach an independently
collected memory snapshot when rendering the status endpoint.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Callable, Mapping


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _jain(values: list[float]) -> float | None:
    if not values:
        return None
    total = sum(values)
    square_total = sum(value * value for value in values)
    if square_total <= 0:
        return None
    return total * total / (len(values) * square_total)


@dataclass
class _RequestState:
    request_id: str
    tenant_id: str
    enqueued_at: float
    dequeued_at: float | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    tokens: int = 0
    inter_token_ms: list[float] = field(default_factory=list)
    mechanism: str = "pending"


class BatchRuntimeMetrics:
    """Thread-safe bounded request/event history for operational inspection."""

    def __init__(
        self,
        history_size: int = 512,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size < 1:
            raise ValueError("history_size must be a positive integer")
        self._clock = clock
        self._lock = threading.Lock()
        self._active: dict[str, _RequestState] = {}
        self._completed: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._events: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._counters: Counter[str] = Counter()
        self._admission: Counter[str] = Counter()
        self._batch_sizes: Counter[int] = Counter()
        self._mechanisms: Counter[str] = Counter()
        self._active_lanes = 0
        self._peak_active_lanes = 0

    def _event(self, kind: str, request_id: str = "", **fields: Any) -> None:
        self._events.append(
            {"kind": kind, "request_id": request_id, "at": self._clock(), **fields}
        )

    def admitted(self, request_id: str, tenant_id: str, queue_depth: int) -> None:
        with self._lock:
            self._active[request_id] = _RequestState(
                request_id, tenant_id or "default", self._clock()
            )
            self._counters["admitted"] += 1
            self._event("admitted", request_id, queue_depth=max(0, queue_depth))

    def rejected(self, reason: str, queue_depth: int) -> None:
        with self._lock:
            self._counters["rejected"] += 1
            self._admission[reason] += 1
            self._event("rejected", reason=reason, queue_depth=max(0, queue_depth))

    def dequeued(self, request_id: str, queue_depth: int) -> None:
        with self._lock:
            state = self._active.get(request_id)
            if state is not None and state.dequeued_at is None:
                state.dequeued_at = self._clock()
            self._event("dequeued", request_id, queue_depth=max(0, queue_depth))

    def lane_attached(self, request_id: str, active_lanes: int, mechanism: str) -> None:
        with self._lock:
            state = self._active.get(request_id)
            if state is not None:
                state.mechanism = mechanism
            self._active_lanes = max(0, active_lanes)
            self._peak_active_lanes = max(self._peak_active_lanes, self._active_lanes)
            self._mechanisms[mechanism] += 1
            self._event(
                "lane_attached",
                request_id,
                active_lanes=self._active_lanes,
                mechanism=mechanism,
            )

    def batch_cycle(self, active_lanes: int, queued_lanes: int = 0) -> None:
        with self._lock:
            self._active_lanes = max(0, active_lanes)
            self._peak_active_lanes = max(self._peak_active_lanes, self._active_lanes)
            self._batch_sizes[self._active_lanes] += 1
            self._counters["batch_cycles"] += 1
            if queued_lanes:
                self._counters["queued_lane_cycles"] += 1

    def token(self, request_id: str) -> None:
        now = self._clock()
        with self._lock:
            state = self._active.get(request_id)
            if state is None:
                return
            if state.first_token_at is None:
                state.first_token_at = now
                self._event("first_token", request_id)
            if state.last_token_at is not None:
                state.inter_token_ms.append((now - state.last_token_at) * 1000.0)
            state.last_token_at = now
            state.tokens += 1
            self._counters["tokens_delivered"] += 1

    def fault(self, request_id: str, kind: str) -> None:
        with self._lock:
            self._counters[f"fault_{kind}"] += 1
            self._event("fault", request_id, fault=kind)

    def terminal(self, request_id: str, status: str) -> None:
        now = self._clock()
        with self._lock:
            state = self._active.pop(request_id, None)
            if state is None:
                return
            queue_ms = ((state.dequeued_at or now) - state.enqueued_at) * 1000.0
            ttft_ms = (
                None
                if state.first_token_at is None
                else (state.first_token_at - state.enqueued_at) * 1000.0
            )
            service_ms = max(
                0.0, (now - (state.dequeued_at or state.enqueued_at)) * 1000.0
            )
            self._completed.append(
                {
                    "request_id": state.request_id,
                    "tenant_id": state.tenant_id,
                    "status": status,
                    "queue_ms": queue_ms,
                    "ttft_ms": ttft_ms,
                    "service_ms": service_ms,
                    "tokens": state.tokens,
                    "itl_ms": list(state.inter_token_ms),
                    "mechanism": state.mechanism,
                }
            )
            self._counters[f"terminal_{status}"] += 1
            self._active_lanes = min(self._active_lanes, len(self._active))
            self._event("terminal", request_id, status=status, tokens=state.tokens)

    def snapshot(
        self,
        *,
        queue_depth: int = 0,
        memory: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            completed = list(self._completed)
            active = list(self._active.values())
            counters = dict(self._counters)
            admission = dict(self._admission)
            batch_sizes = {str(key): value for key, value in sorted(self._batch_sizes.items())}
            mechanisms = dict(self._mechanisms)
            events = list(self._events)
            active_lanes = self._active_lanes
            peak_active_lanes = self._peak_active_lanes
        queue_values = [row["queue_ms"] for row in completed]
        ttft_values = [row["ttft_ms"] for row in completed if row["ttft_ms"] is not None]
        itl_values = [value for row in completed for value in row["itl_ms"]]
        tenant_tokens: Counter[str] = Counter()
        tenant_service: Counter[str] = Counter()
        for row in completed:
            tenant_tokens[row["tenant_id"]] += row["tokens"]
            tenant_service[row["tenant_id"]] += row["service_ms"] / 1000.0
        tenant_rates = {
            tenant: tenant_tokens[tenant] / seconds
            for tenant, seconds in tenant_service.items()
            if seconds > 0
        }
        return {
            "schema": "mlx2.batch-runtime.v1",
            "counters": counters,
            "admission_decisions": admission,
            "gauges": {
                "inflight_requests": len(active),
                "queue_depth": max(0, queue_depth),
                "active_lanes": active_lanes,
                "peak_active_lanes": peak_active_lanes,
            },
            "latency_ms": {
                "queue": _distribution(queue_values),
                "ttft": _distribution(ttft_values),
                "itl": _distribution(itl_values),
            },
            "fairness": {
                "jain_tenant_token_rate": _jain(list(tenant_rates.values())),
                "tenant_token_rates": tenant_rates,
            },
            "batch_composition": batch_sizes,
            "mechanism_receipts": mechanisms,
            "memory": dict(memory or {}),
            "active_requests": [state.request_id for state in active],
            "completed_requests": [
                {key: value for key, value in row.items() if key != "itl_ms"}
                for row in completed
            ],
            "events": events,
        }


@dataclass(frozen=True)
class BatchFaultSpec:
    kind: str
    after_tokens: int = 0

    KINDS = frozenset({"lane_abort", "cache_evict", "cache_reallocate"})

    @classmethod
    def parse(cls, value: Any, *, enabled: bool) -> "BatchFaultSpec | None":
        if value is None:
            return None
        if not enabled:
            raise ValueError("batch fault injection is disabled on this server")
        if not isinstance(value, dict) or value.get("kind") not in cls.KINDS:
            raise ValueError(f"mlx_fault.kind must be one of {sorted(cls.KINDS)}")
        after_tokens = value.get("after_tokens", 0)
        if isinstance(after_tokens, bool) or not isinstance(after_tokens, int) or after_tokens < 0:
            raise ValueError("mlx_fault.after_tokens must be a non-negative integer")
        if value["kind"] != "lane_abort" and after_tokens:
            raise ValueError("after_tokens is only valid for lane_abort")
        return cls(value["kind"], after_tokens)
