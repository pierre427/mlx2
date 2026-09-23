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

from .prometheus import (
    BATCH_SIZE_BUCKETS,
    REQUEST_DURATION_BUCKETS,
    TOKEN_COUNT_BUCKETS,
    TOKEN_LATENCY_BUCKETS,
    CumulativeHistogram,
)


_ADMISSION_REASONS = frozenset({"maximum_inflight", "parallel_sample_footprint"})
_MECHANISMS = frozenset({"ordinary", "self_mtp", "external_draft", "prompt_lookup"})
_HTTP_METHODS = frozenset({"GET", "POST"})
# One bounded route vocabulary for both the server, which names each
# request's route, and HttpRuntimeMetrics, which admits only these names as
# label values. Keeping them in one place stops a server route from silently
# collapsing into route="other".
_HTTP_EXACT_ROUTES = {
    "/metrics": "metrics",
    "/health": "health",
    "/v1/models": "models",
    "/v1/status": "status",
    "/v1/status/batching": "batching_status",
    "/v1/completions": "completions",
    "/v1/chat/completions": "chat_completions",
    "/v1/responses": "responses",
    "/v1/files": "files",
    "/v1/batches": "batches",
    "/v1/embeddings": "embeddings",
    "/v1/rerank": "rerank",
    "/v1/messages": "anthropic_messages",
    "/v1/messages/count_tokens": "anthropic_count_tokens",
    "/tokenize": "tokenize",
    "/apply-template": "apply_template",
    "/v1/load_lora_adapter": "load_lora_adapter",
    "/v1/unload_lora_adapter": "unload_lora_adapter",
}
_HTTP_PREFIX_ROUTES = frozenset(
    {
        "apc_sessions",
        "admin",
        "response_input_items",
        "responses_resource",
        "files_resource",
        "batches_resource",
    }
)
_HTTP_ROUTES = frozenset(
    {*_HTTP_EXACT_ROUTES.values(), *_HTTP_PREFIX_ROUTES, "other"}
)


def http_metric_route(path: str) -> str:
    """Map a request path to its bounded route label."""
    path = path.split("?", 1)[0]
    if path.startswith("/v1/apc/sessions"):
        return "apc_sessions"
    if path.startswith("/v1/admin/"):
        return "admin"
    if path.startswith("/v1/responses/") and path.endswith("/input_items"):
        return "response_input_items"
    if path.startswith("/v1/responses/"):
        return "responses_resource"
    if path.startswith("/v1/files/"):
        return "files_resource"
    if path.startswith("/v1/batches/"):
        return "batches_resource"
    return _HTTP_EXACT_ROUTES.get(path, "other")


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
    attached_at: float | None = None
    tokens: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    prompt_recorded: bool = False
    finish_reason: str | None = None
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
        self._terminal_reasons: Counter[tuple[str, str]] = Counter()
        self._active_lanes = 0
        self._peak_active_lanes = 0
        self._histograms = {
            "mlx2_time_to_first_token_seconds": CumulativeHistogram(
                REQUEST_DURATION_BUCKETS
            ),
            "mlx2_inter_token_latency_seconds": CumulativeHistogram(
                TOKEN_LATENCY_BUCKETS
            ),
            "mlx2_request_time_per_output_token_seconds": CumulativeHistogram(
                TOKEN_LATENCY_BUCKETS
            ),
            "mlx2_e2e_request_latency_seconds": CumulativeHistogram(
                REQUEST_DURATION_BUCKETS
            ),
            "mlx2_request_queue_time_seconds": CumulativeHistogram(
                REQUEST_DURATION_BUCKETS
            ),
            "mlx2_request_prefill_time_seconds": CumulativeHistogram(
                REQUEST_DURATION_BUCKETS
            ),
            "mlx2_request_decode_time_seconds": CumulativeHistogram(
                REQUEST_DURATION_BUCKETS
            ),
            "mlx2_request_inference_time_seconds": CumulativeHistogram(
                REQUEST_DURATION_BUCKETS
            ),
            "mlx2_request_prompt_tokens": CumulativeHistogram(TOKEN_COUNT_BUCKETS),
            "mlx2_request_generation_tokens": CumulativeHistogram(
                TOKEN_COUNT_BUCKETS
            ),
            "mlx2_batch_size": CumulativeHistogram(BATCH_SIZE_BUCKETS),
        }

    def _event(self, kind: str, request_id: str = "", **fields: Any) -> float:
        timestamp = self._clock()
        self._events.append(
            {"kind": kind, "request_id": request_id, "at": timestamp, **fields}
        )
        return timestamp

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
                self._histograms["mlx2_request_queue_time_seconds"].observe(
                    state.dequeued_at - state.enqueued_at
                )
            self._event("dequeued", request_id, queue_depth=max(0, queue_depth))

    def prompt(self, request_id: str, prompt_tokens: int, cached_tokens: int = 0) -> None:
        """Record authoritative prompt accounting once after APCv2 admission."""

        with self._lock:
            state = self._active.get(request_id)
            if state is None or state.prompt_recorded:
                return
            state.prompt_tokens = max(0, int(prompt_tokens))
            state.cached_tokens = max(0, min(int(cached_tokens), state.prompt_tokens))
            state.prompt_recorded = True
            self._counters["prompt_tokens"] += state.prompt_tokens
            self._counters["cached_prompt_tokens"] += state.cached_tokens
            self._histograms["mlx2_request_prompt_tokens"].observe(
                state.prompt_tokens
            )

    def lane_attached(self, request_id: str, active_lanes: int, mechanism: str) -> None:
        with self._lock:
            state = self._active.get(request_id)
            if state is not None:
                state.mechanism = mechanism
            self._active_lanes = max(0, active_lanes)
            self._peak_active_lanes = max(self._peak_active_lanes, self._active_lanes)
            # A memory-preempted request is attached again when it replays;
            # the engagement counter counts requests, so only the first
            # attachment of a tracked request counts.
            if state is None or state.attached_at is None:
                self._mechanisms[mechanism] += 1
            attached_at = self._event(
                "lane_attached",
                request_id,
                active_lanes=self._active_lanes,
                mechanism=mechanism,
            )
            if state is not None and state.attached_at is None:
                state.attached_at = attached_at

    def batch_cycle(self, active_lanes: int, queued_lanes: int = 0) -> None:
        with self._lock:
            self._active_lanes = max(0, active_lanes)
            self._peak_active_lanes = max(self._peak_active_lanes, self._active_lanes)
            self._batch_sizes[self._active_lanes] += 1
            self._counters["batch_cycles"] += 1
            self._histograms["mlx2_batch_size"].observe(self._active_lanes)
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
                self._histograms["mlx2_time_to_first_token_seconds"].observe(
                    now - state.enqueued_at
                )
                if state.attached_at is not None:
                    self._histograms[
                        "mlx2_request_prefill_time_seconds"
                    ].observe(now - state.attached_at)
                self._event("first_token", request_id)
            if state.last_token_at is not None:
                interval = now - state.last_token_at
                state.inter_token_ms.append(interval * 1000.0)
                self._histograms["mlx2_inter_token_latency_seconds"].observe(
                    interval
                )
            state.last_token_at = now
            state.tokens += 1
            self._counters["tokens_delivered"] += 1

    def fault(self, request_id: str, kind: str) -> None:
        with self._lock:
            self._counters[f"fault_{kind}"] += 1
            self._event("fault", request_id, fault=kind)

    def finishing(self, request_id: str, finish_reason: str | None) -> None:
        """Attach a bounded public finish reason before terminal ownership ends."""

        with self._lock:
            state = self._active.get(request_id)
            if state is not None:
                state.finish_reason = finish_reason

    def terminal(
        self, request_id: str, status: str, finish_reason: str | None = None
    ) -> None:
        now = self._clock()
        with self._lock:
            state = self._active.pop(request_id, None)
            if state is None:
                # The engine registers some jobs before it publishes them: a
                # batch-cohort member can expire (429) before its cohort
                # fills, and an APCv2 fan-out sibling can fail (503) with its
                # leader. The client still received a terminal response, so
                # count the outcome; without admission timestamps the job
                # contributes to no latency histogram.
                self._count_terminal(status, finish_reason)
                self._event("terminal", request_id, status=status, tokens=0)
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
            e2e_seconds = max(0.0, now - state.enqueued_at)
            service_seconds = service_ms / 1000.0
            self._histograms["mlx2_request_inference_time_seconds"].observe(
                service_seconds
            )
            if state.first_token_at is not None:
                self._histograms["mlx2_request_decode_time_seconds"].observe(
                    max(0.0, now - state.first_token_at)
                )
            self._histograms["mlx2_e2e_request_latency_seconds"].observe(
                e2e_seconds
            )
            self._histograms["mlx2_request_generation_tokens"].observe(
                state.tokens
            )
            if state.tokens:
                self._histograms[
                    "mlx2_request_time_per_output_token_seconds"
                ].observe(service_seconds / state.tokens)
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
            self._count_terminal(status, finish_reason or state.finish_reason)
            self._active_lanes = min(self._active_lanes, len(self._active))
            self._event("terminal", request_id, status=status, tokens=state.tokens)

    def _count_terminal(self, status: str, finish_reason: str | None) -> None:
        self._counters[f"terminal_{status}"] += 1
        bounded_reason = (
            finish_reason
            if finish_reason in {"stop", "length", "tool_calls", "eos"}
            else "none"
        )
        self._terminal_reasons[(status, bounded_reason)] += 1

    def prometheus_snapshot(self, *, queue_depth: int = 0) -> dict[str, Any]:
        """Return cumulative aggregate telemetry without request identifiers."""

        help_text = {
            "mlx2_num_requests_running": "Requests currently owned by the serving engine.",
            "mlx2_num_requests_waiting": "Requests waiting for an execution lane.",
            "mlx2_active_lanes": "Execution lanes active in the most recent scheduler cycle.",
            "mlx2_peak_active_lanes": "Highest number of simultaneously active execution lanes.",
            "mlx2_jain_tenant_token_rate": "Jain fairness index over the bounded completed-request window.",
            "mlx2_requests_total": "Terminal requests by bounded outcome.",
            "mlx2_admission_decisions_total": "Request admission decisions by bounded reason.",
            "mlx2_prompt_tokens_total": "Prompt tokens admitted by the serving engine.",
            "mlx2_generation_tokens_total": "Generated tokens delivered by the serving engine.",
            "mlx2_request_cached_prompt_tokens_total": "Prompt tokens served from APCv2 for admitted requests.",
            "mlx2_scheduler_cycles_total": "Serving scheduler cycles by bounded condition.",
            "mlx2_capability_engagements_total": "Requests attached to a selected generation mechanism.",
            "mlx2_time_to_first_token_seconds": "Time from request admission to first generated token.",
            "mlx2_inter_token_latency_seconds": "Time between consecutive delivered output tokens.",
            "mlx2_request_time_per_output_token_seconds": "Per-request service time divided by delivered output tokens.",
            "mlx2_e2e_request_latency_seconds": "Time from request admission through terminal completion.",
            "mlx2_request_queue_time_seconds": "Time from request admission to scheduler dequeue.",
            "mlx2_request_prefill_time_seconds": "Time from execution-lane attachment to first generated token.",
            "mlx2_request_decode_time_seconds": "Time from first generated token to terminal completion.",
            "mlx2_request_inference_time_seconds": "Time from scheduler dequeue to terminal completion.",
            "mlx2_request_prompt_tokens": "Prompt token count per admitted request.",
            "mlx2_request_generation_tokens": "Generated token count per terminal request.",
            "mlx2_batch_size": "Active execution lanes observed per scheduler cycle.",
        }
        with self._lock:
            counters = dict(self._counters)
            admission = dict(self._admission)
            mechanisms = dict(self._mechanisms)
            terminal_reasons = dict(self._terminal_reasons)
            completed = list(self._completed)
            histograms = {
                name: histogram.snapshot()
                for name, histogram in self._histograms.items()
            }
            gauges = {
                "mlx2_num_requests_running": len(self._active),
                "mlx2_num_requests_waiting": max(0, int(queue_depth)),
                "mlx2_active_lanes": self._active_lanes,
                "mlx2_peak_active_lanes": self._peak_active_lanes,
            }
        tenant_tokens: Counter[str] = Counter()
        tenant_service: Counter[str] = Counter()
        for row in completed:
            tenant_tokens[row["tenant_id"]] += row["tokens"]
            tenant_service[row["tenant_id"]] += row["service_ms"] / 1000.0
        rates = [
            tenant_tokens[tenant] / seconds
            for tenant, seconds in tenant_service.items()
            if seconds > 0
        ]
        fairness = _jain(rates)
        if fairness is not None:
            gauges["mlx2_jain_tenant_token_rate"] = fairness

        samples: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}

        def counter(name: str, value: int, **labels: str) -> None:
            key = (name, tuple(sorted(labels.items())))
            samples[key] = samples.get(key, 0) + int(value)

        for outcome in ("completed", "cancelled", "failed"):
            matching = {
                reason: value
                for (status, reason), value in terminal_reasons.items()
                if status == outcome
            }
            if not matching:
                matching = {"none": counters.get(f"terminal_{outcome}", 0)}
            for reason, value in sorted(matching.items()):
                counter(
                    "mlx2_requests_total",
                    value,
                    outcome=outcome,
                    finished_reason=reason,
                )
        counter(
            "mlx2_admission_decisions_total",
            counters.get("admitted", 0),
            decision="admitted",
            reason="accepted",
        )
        for reason, value in sorted(admission.items()):
            counter(
                "mlx2_admission_decisions_total",
                value,
                decision="rejected",
                reason=reason if reason in _ADMISSION_REASONS else "other",
            )
        counter("mlx2_prompt_tokens_total", counters.get("prompt_tokens", 0))
        counter("mlx2_generation_tokens_total", counters.get("tokens_delivered", 0))
        counter(
            "mlx2_request_cached_prompt_tokens_total",
            counters.get("cached_prompt_tokens", 0),
        )
        counter(
            "mlx2_scheduler_cycles_total",
            counters.get("batch_cycles", 0),
            condition="all",
        )
        counter(
            "mlx2_scheduler_cycles_total",
            counters.get("queued_lane_cycles", 0),
            condition="queued",
        )
        for mechanism, value in sorted(mechanisms.items()):
            counter(
                "mlx2_capability_engagements_total",
                value,
                capability=mechanism if mechanism in _MECHANISMS else "other",
                outcome="engaged",
            )
        return {
            "help": help_text,
            "gauges": gauges,
            "counters": samples,
            "histograms": histograms,
        }

    def snapshot(
        self,
        *,
        queue_depth: int = 0,
        memory: Mapping[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """The bounded history; only ``tenant_id``'s requests when one is given.

        A tenant-scoped view keeps the aggregates (counters, gauges, latency
        distributions, the Jain index) but lists only that tenant's own
        requests, events and token rate: request ids and other tenants' ids
        stay private.
        """
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
        fairness = _jain(list(tenant_rates.values()))
        inflight = len(active)
        if tenant_id is not None:
            owner = str(tenant_id or "default")
            active = [state for state in active if state.tenant_id == owner]
            completed = [row for row in completed if row["tenant_id"] == owner]
            owned = {state.request_id for state in active} | {
                row["request_id"] for row in completed
            }
            # Events without a request of this tenant (rejections carry no
            # request at all) cannot be attributed, so they are left out.
            events = [event for event in events if event.get("request_id") in owned]
            tenant_rates = {
                tenant: rate for tenant, rate in tenant_rates.items() if tenant == owner
            }
        return {
            "schema": "mlx2.batch-runtime.v1",
            "counters": counters,
            "admission_decisions": admission,
            "gauges": {
                "inflight_requests": inflight,
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
                "jain_tenant_token_rate": fairness,
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


class HttpRuntimeMetrics:
    """Bounded host-only HTTP request counters and duration histograms."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._requests: Counter[tuple[str, str, str]] = Counter()
        self._durations = {
            route: CumulativeHistogram(REQUEST_DURATION_BUCKETS)
            for route in _HTTP_ROUTES
        }

    def started(self) -> float:
        return self._clock()

    def completed(
        self, method: str, route: str, status: int, started_at: float
    ) -> None:
        method = method if method in _HTTP_METHODS else "other"
        route = route if route in _HTTP_ROUTES else "other"
        status_class = f"{max(0, min(9, int(status) // 100))}xx"
        duration = max(0.0, self._clock() - float(started_at))
        with self._lock:
            self._requests[(method, route, status_class)] += 1
            self._durations[route].observe(duration)

    def prometheus_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": dict(self._requests),
                "durations": {
                    route: histogram.snapshot()
                    for route, histogram in self._durations.items()
                },
            }


@dataclass(frozen=True)
class BatchFaultSpec:
    kind: str
    after_tokens: int = 0

    KINDS = frozenset(
        {"lane_abort", "cache_evict", "cache_reallocate", "memory_preempt"}
    )

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
        if value["kind"] not in {"lane_abort", "memory_preempt"} and after_tokens:
            raise ValueError(
                "after_tokens is only valid for lane_abort and memory_preempt"
            )
        return cls(value["kind"], after_tokens)
