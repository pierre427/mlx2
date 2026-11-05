"""Bounded, host-only telemetry for experimental weight-space operations.

This process-local collector never stores model, program, tensor, tenant or
session identities. Scrapes do not load checkpoints or materialize tensors.
"""

from __future__ import annotations

import math
import threading
import time
import weakref
from collections import Counter
from functools import wraps

OPERATIONS = frozenset({
    "program_create", "program_inspect", "program_export", "program_load", "lora_import",
    "directional_repair", "compose", "truncate", "projection_evaluate", "direction_estimate",
    "abliteration_compile", "retention_evaluate", "memory_calibrate", "memory_publish",
    "memory_mount", "failure_report", "repair_propose", "repair_trial", "repair_promote",
    "repair_rollback", "repair_expire", "repair_reject", "quantization_overlay",
    "layer_excision", "quality_evaluate", "perplexity_evaluate", "tool_dispatch",
    "expert_mask_compile", "expert_budget_plan", "runtime_compatibility",
})
OUTCOMES = frozenset({"ok", "error", "rejected", "expired", "cancelled"})


def unconfigured_runtime_state():
    """Mechanism state for a weight-space feature whose manager is not built.

    The code is present (implemented) but nothing is qualified, selected or
    used; status() and /metrics share this so the two cannot disagree.
    """
    return {"implemented": True, "qualified": False, "selected": False,
            "observed_used": False}


DURATION_BUCKETS = (0.001, 0.01, 0.1, 1.0, 10.0, 60.0, 300.0, 3600.0)
OVERLAY_GAUGES = {
    "quantization_overlay_bytes": "resident_bytes",
    "quantization_overlay_tensors": "tensors",
    "quantization_overlay_candidates": "candidates",
    "quantization_overlay_budget_bytes": "budget_bytes",
    "quantization_overlay_construction_reserved_bytes": "construction_reserved_bytes",
    "quantization_overlay_construction_budget_bytes": "construction_budget_bytes",
    "quantization_overlay_payload_reserved_bytes": "payload_reserved_bytes",
}
GAUGES = frozenset({
    "active_candidates", "compiled_program_bytes", "live_enabled", "live_execution_enabled",
    "promotion_recovery_required",
    "live_automatic_promotion", "quantization_overlay_bytes", "quantization_overlay_tensors",
    "quantization_overlay_candidates",
    "quantization_error_mse", "layer_excision_layers", "quality_score_delta", "perplexity_delta",
    "memory_calibration_error_mse", "projection_delta_norm", "retention_error_mse",
}) | frozenset(OVERLAY_GAUGES)
EXPERT_COUNTERS = frozenset({
    "requests", "hits", "faults", "evictions", "chunks", "streamed_bytes", "pacing_waits",
    "cancelled_bytes", "prefetch_requests", "prefetch_skips",
})
EXPERT_GAUGES = frozenset({
    "resident_bytes", "loading_bytes", "pinned_bytes", "overlay_bytes", "reserved_bytes",
    "budget_bytes", "resident_experts", "pending_experts", "leases", "peak_payload_bytes",
})


def _finite(value, *, signed=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("telemetry values must be finite numbers")
    if not math.isfinite(value) or (not signed and value < 0):
        raise ValueError("telemetry values must be finite and nonnegative")
    return value


class WeightSpaceMetrics:
    """Thread-safe fixed-label collector; independent instances support host tests."""

    def __init__(self):
        self.lock = threading.RLock()
        self.events = Counter()
        self.bytes = Counter()
        self.durations = {}
        self.gauges = dict.fromkeys(GAUGES, 0)
        self.measured = set()
        self.experts = weakref.WeakKeyDictionary()
        self.expert_totals = Counter()
        self.repair_sources = weakref.WeakKeyDictionary()
        self.overlay_sources = weakref.WeakSet()

    def record(self, operation, outcome="ok", *, duration_seconds=None, bytes_count=0):
        if operation not in OPERATIONS or outcome not in OUTCOMES:
            raise ValueError("weight-space telemetry labels must be allowlisted")
        _finite(bytes_count)
        if duration_seconds is not None:
            _finite(duration_seconds)
        with self.lock:
            self.events[operation, outcome] += 1
            self.bytes[operation] += bytes_count
            if duration_seconds is not None:
                row = self.durations.setdefault(operation, {"count": 0, "sum": 0.0,
                                                           "buckets": [0] * len(DURATION_BUCKETS)})
                row["count"] += 1
                row["sum"] += duration_seconds
                for index, bound in enumerate(DURATION_BUCKETS):
                    row["buckets"][index] += duration_seconds <= bound

    def set_gauge(self, name, value):
        if name not in GAUGES:
            raise ValueError("weight-space gauge must be allowlisted")
        _finite(value, signed=name in {"quality_score_delta", "perplexity_delta"})
        with self.lock:
            self.gauges[name] = value
            self.measured.add(name)

    def processed_bytes(self, operation, value):
        if operation not in OPERATIONS:
            raise ValueError("weight-space operation must be allowlisted")
        _finite(value)
        with self.lock:
            self.bytes[operation] += value

    def active_candidates(self, controller, count, *, recovery_required=0):
        _finite(count)
        _finite(recovery_required)
        with self.lock:
            self.repair_sources[controller] = {"active": count, "recovery": recovery_required}
            self.measured.update({"active_candidates", "promotion_recovery_required"})

    def register_experts(self, controller):
        with self.lock:
            self.experts.setdefault(controller, {})

    def register_overlays(self, bank):
        with self.lock:
            self.overlay_sources.add(bank)
            self.measured.update(OVERLAY_GAUGES)

    def sample_experts(self, controller):
        # Avoid inversion with a residency-controller lock. A caller may invoke
        # this after each state mutation, and scrapes also refresh live sources.
        snapshot = controller.metrics_snapshot()
        with self.lock:
            previous = self.experts.setdefault(controller, {})
            for key in EXPERT_COUNTERS:
                value = _finite(snapshot.get(key, 0))
                self.expert_totals[key] += max(0, value - previous.get(key, 0))
            self.experts[controller] = {
                key: max(previous.get(key, 0), _finite(snapshot.get(key, 0)))
                if key in EXPERT_COUNTERS else _finite(snapshot.get(key, 0))
                for key in EXPERT_COUNTERS | EXPERT_GAUGES}

    def snapshot(self):
        with self.lock:
            sources = list(self.experts)
            banks = list(self.overlay_sources)
        for source in sources:
            self.sample_experts(source)
        overlays = [bank.snapshot() for bank in banks]
        with self.lock:
            return {
                "events": dict(self.events), "bytes": dict(self.bytes),
                "durations": {key: {**row, "buckets": list(row["buckets"])}
                              for key, row in self.durations.items()},
                "gauges": {**self.gauges,
                           "active_candidates": sum(row["active"] for row in self.repair_sources.values()),
                           "promotion_recovery_required": sum(row["recovery"] for row in self.repair_sources.values()),
                           **{name: sum(_finite(row.get(field, 0)) for row in overlays)
                              for name, field in OVERLAY_GAUGES.items()}},
                "measured": sorted(self.measured),
                "expert_counters": dict(self.expert_totals),
                "expert_gauges": {key: sum(row.get(key, 0) for row in self.experts.values())
                                  for key in EXPERT_GAUGES},
            }


_METRICS = WeightSpaceMetrics()


def record_weight_space(operation, outcome="ok", *, duration_seconds=None, bytes_count=0):
    _METRICS.record(operation, outcome, duration_seconds=duration_seconds, bytes_count=bytes_count)


def set_weight_space_gauge(name, value):
    _METRICS.set_gauge(name, value)


def clear_weight_space_gauge(name):
    if name not in GAUGES:
        raise ValueError("weight-space gauge must be allowlisted")
    with _METRICS.lock:
        _METRICS.gauges[name] = 0
        _METRICS.measured.discard(name)


def record_weight_space_bytes(operation, value):
    _METRICS.processed_bytes(operation, value)


def set_active_weight_candidates(controller, count, *, recovery_required=0):
    _METRICS.active_candidates(controller, count, recovery_required=recovery_required)


def register_expert_metrics(controller):
    _METRICS.register_experts(controller)


def register_overlay_metrics(bank):
    _METRICS.register_overlays(bank)


def update_expert_metrics(controller):
    _METRICS.sample_experts(controller)


def weight_space_metrics_snapshot():
    return _METRICS.snapshot()


def track_weight_space(operation):
    """Observe completion and duration without altering exceptions or results."""
    if operation not in OPERATIONS:
        raise ValueError("weight-space operation must be allowlisted")

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            started = time.monotonic()
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                record_weight_space(operation, "cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit))
                                    else "error", duration_seconds=time.monotonic() - started)
                raise
            outcome = "rejected" if isinstance(result, dict) and result.get("state") == "rejected" else "ok"
            record_weight_space(operation, outcome, duration_seconds=time.monotonic() - started)
            return result
        return wrapped
    return decorate
