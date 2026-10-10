"""Bounded Prometheus telemetry for standalone decision-model serving."""

from __future__ import annotations

import platform
import resource
import threading
import time
from collections.abc import Callable
from typing import Any

from ..prometheus import (
    REQUEST_DURATION_BUCKETS,
    TOKEN_COUNT_BUCKETS,
    CumulativeHistogram,
    PrometheusBuilder,
)

# Process counters behind /v1/status.  ``executions`` counts every request
# whose model forward began (in flight, succeeded or failed); it is what
# "observed_used" means, while ``requests`` counts successes only.
COUNTER_NAMES = ("requests", "failures", "refusals", "input_tokens", "executions")


def new_counters() -> dict[str, int]:
    return dict.fromkeys(COUNTER_NAMES, 0)


class DecisionRuntimeMetrics:
    """Host-only decision execution metrics; never synchronizes the device."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._created_at = time.time()
        self._lock = threading.Lock()
        self._load_seconds = 0.0
        self._inflight = 0
        self._peak_inflight = 0
        self._durations = CumulativeHistogram(REQUEST_DURATION_BUCKETS)
        self._input_tokens = CumulativeHistogram(TOKEN_COUNT_BUCKETS)

    def loading_started(self) -> float:
        return self._clock()

    def loaded(self, started_at: float) -> None:
        with self._lock:
            self._load_seconds = max(0.0, self._clock() - float(started_at))

    def execution_started(self) -> float:
        with self._lock:
            self._inflight += 1
            self._peak_inflight = max(self._peak_inflight, self._inflight)
        return self._clock()

    def execution_finished(self, started_at: float, *, input_tokens: int = 0) -> None:
        duration = max(0.0, self._clock() - float(started_at))
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self._durations.observe(duration)
            if input_tokens > 0:
                self._input_tokens.observe(input_tokens)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "created_at": self._created_at,
                "load_seconds": self._load_seconds,
                "inflight": self._inflight,
                "peak_inflight": self._peak_inflight,
                "durations": self._durations.snapshot(),
                "input_tokens": self._input_tokens.snapshot(),
            }


def render_decision_metrics(engine) -> str:
    """Render one deterministic, low-cardinality Prometheus snapshot."""

    builder = PrometheusBuilder()
    status = engine.status()
    route = status["route"]
    labels = {"family": route["family"], "variant": route["variant"]}
    builder.gauge(
        "mlx2_decision_route_info",
        "Selected decision route identity; the sample value is always one.",
        1,
        {
            **labels,
            "qualification": route["qualification"],
            "fingerprint_kind": route["artifact_fingerprint_kind"],
            "revision": route.get("artifact_revision") or "unavailable",
        },
    )
    builder.gauge(
        "mlx2_decision_qualified",
        "Whether the selected decision route has a validated qualification receipt.",
        1 if route["qualified"] else 0,
        labels,
    )
    counters = status["counters"]
    for outcome, key in (
        ("success", "requests"),
        ("refusal", "refusals"),
        ("failure", "failures"),
    ):
        builder.counter(
            "mlx2_decision_requests_total",
            "Decision requests by bounded terminal outcome.",
            counters[key],
            {**labels, "outcome": outcome},
        )
    builder.counter(
        "mlx2_decision_input_tokens_total",
        "Input tokens processed by successful decision requests.",
        counters["input_tokens"],
        labels,
    )

    metrics = engine.decision_metrics.snapshot()
    builder.gauge(
        "mlx2_decision_inflight_requests",
        "Decision requests currently holding the serialized decision lane.",
        metrics["inflight"],
        labels,
    )
    builder.gauge(
        "mlx2_decision_peak_inflight_requests",
        "Peak decision requests concurrently holding the serialized lane.",
        metrics["peak_inflight"],
        labels,
    )
    builder.gauge(
        "mlx2_decision_model_load_seconds",
        "Seconds spent loading and evaluating the resident decision model.",
        metrics["load_seconds"],
        labels,
    )
    builder.gauge(
        "mlx2_decision_process_start_time_seconds",
        "Unix time when decision telemetry was initialized.",
        metrics["created_at"],
        labels,
    )
    builder.histogram(
        "mlx2_decision_request_duration_seconds",
        "Serialized decision dispatch duration, including prompt construction.",
        metrics["durations"],
        labels,
    )
    builder.histogram(
        "mlx2_decision_request_input_tokens",
        "Input tokens processed by successful decision requests.",
        metrics["input_tokens"],
        labels,
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    builder.gauge(
        "mlx2_decision_process_peak_resident_memory_bytes",
        "Peak resident process memory in bytes.",
        peak_rss,
        labels,
    )
    builder.counter(
        "mlx2_decision_process_cpu_seconds_total",
        "User and system CPU seconds consumed by the decision process.",
        float(usage.ru_utime) + float(usage.ru_stime),
        labels,
    )

    http = engine.http_metrics.prometheus_snapshot()
    for (method, route_name, status_class), value in sorted(http["requests"].items()):
        builder.counter(
            "mlx2_http_requests_total",
            "HTTP requests by bounded method, route, and status class.",
            value,
            {"method": method, "route": route_name, "status_class": status_class},
        )
    for route_name, histogram in sorted(http["durations"].items()):
        builder.histogram(
            "mlx2_http_request_duration_seconds",
            "HTTP request duration by bounded route.",
            histogram,
            {"route": route_name},
        )
    return builder.render()
