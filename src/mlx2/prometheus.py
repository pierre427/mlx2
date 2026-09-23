"""Prometheus exposition for mlx2 host-side telemetry.

The exporter deliberately owns no device objects and performs no destructive
reads. Runtime components publish bounded host-side snapshots; this module
translates those snapshots into a stable, low-cardinality scrape contract.
Detailed request identifiers and route receipts remain on the JSON status
endpoints and are never promoted to labels here.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
import re
import time
from typing import Any, Iterable, Mapping


CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
SCHEMA_VERSION = "2"

REQUEST_DURATION_BUCKETS = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
)
TOKEN_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    2.5,
)
TOKEN_COUNT_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32, 64)

_METRIC_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")
_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


@dataclass(frozen=True, slots=True)
class HistogramSnapshot:
    buckets: tuple[float, ...]
    bucket_counts: tuple[int, ...]
    count: int
    total: float


class CumulativeHistogram:
    """A small host-only cumulative histogram with immutable bucket edges."""

    def __init__(self, buckets: Iterable[float]) -> None:
        values = tuple(float(value) for value in buckets)
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("histogram buckets must be finite and non-empty")
        if values != tuple(sorted(set(values))):
            raise ValueError("histogram buckets must be unique and increasing")
        self._buckets = values
        self._bin_counts = [0] * (len(values) + 1)
        self._count = 0
        self._total = 0.0

    def observe(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value) or value < 0:
            return
        self._count += 1
        self._total += value
        self._bin_counts[bisect_left(self._buckets, value)] += 1

    def snapshot(self) -> HistogramSnapshot:
        cumulative = []
        running = 0
        for count in self._bin_counts[:-1]:
            running += count
            cumulative.append(running)
        return HistogramSnapshot(
            self._buckets,
            tuple(cumulative),
            self._count,
            self._total,
        )


def _escape_help(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n")


def _escape_label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _number(value: int | float) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if value == math.inf:
        return "+Inf"
    if value == -math.inf:
        return "-Inf"
    return format(value, ".17g")


class PrometheusBuilder:
    """Deterministic Prometheus text-format builder with family validation."""

    def __init__(self) -> None:
        self._families: dict[str, tuple[str, str]] = {}
        # Samples are kept per family so render() can emit each family as one
        # contiguous group, as the text format requires. Each entry is a sort
        # key and the lines it contributes; a histogram label set contributes
        # its buckets, sum, and count as one unit so bucket order survives.
        self._samples: dict[str, list[tuple[str, tuple[str, ...]]]] = {}

    def _family(self, name: str, kind: str, help_text: str) -> None:
        if not _METRIC_RE.fullmatch(name):
            raise ValueError(f"invalid metric name: {name!r}")
        definition = (kind, help_text)
        previous = self._families.setdefault(name, definition)
        if previous != definition:
            raise ValueError(f"conflicting metric family: {name}")

    @staticmethod
    def _labels(labels: Mapping[str, Any] | None) -> str:
        if not labels:
            return ""
        parts = []
        for name, value in sorted(labels.items()):
            if not _LABEL_RE.fullmatch(name):
                raise ValueError(f"invalid label name: {name!r}")
            parts.append(f'{name}="{_escape_label(value)}"')
        return "{" + ",".join(parts) + "}"

    def sample(
        self,
        name: str,
        kind: str,
        help_text: str,
        value: int | float,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        self._family(name, kind, help_text)
        line = f"{name}{self._labels(labels)} {_number(value)}"
        self._samples.setdefault(name, []).append((line, (line,)))

    def counter(
        self,
        name: str,
        help_text: str,
        value: int | float,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        if not name.endswith("_total"):
            raise ValueError("counter names must end in _total")
        self.sample(name, "counter", help_text, value, labels)

    def gauge(
        self,
        name: str,
        help_text: str,
        value: int | float,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        self.sample(name, "gauge", help_text, value, labels)

    def histogram(
        self,
        name: str,
        help_text: str,
        snapshot: HistogramSnapshot,
        labels: Mapping[str, Any] | None = None,
    ) -> None:
        self._family(name, "histogram", help_text)
        base_labels = dict(labels or {})
        lines = []
        for boundary, count in zip(snapshot.buckets, snapshot.bucket_counts):
            bucket_labels = {**base_labels, "le": _number(boundary)}
            lines.append(f"{name}_bucket{self._labels(bucket_labels)} {count}")
        lines.append(
            f"{name}_bucket{self._labels({**base_labels, 'le': '+Inf'})} {snapshot.count}"
        )
        suffix = self._labels(base_labels)
        lines.append(f"{name}_sum{suffix} {_number(snapshot.total)}")
        lines.append(f"{name}_count{suffix} {snapshot.count}")
        self._samples.setdefault(name, []).append((suffix, tuple(lines)))

    def render(self) -> str:
        # Each family is one group: HELP, TYPE, then its samples. Parsers
        # attach samples to the most recent TYPE line, so interleaving
        # families, or string-sorting histogram buckets, leaves every sample
        # untyped and buckets out of order.
        lines = []
        for name, (kind, help_text) in sorted(self._families.items()):
            lines.extend(
                (f"# HELP {name} {_escape_help(help_text)}", f"# TYPE {name} {kind}")
            )
            for _key, group in sorted(self._samples.get(name, ())):
                lines.extend(group)
        return "\n".join(lines) + "\n"


from .agent_compat import COUNTERS as _AGENT_COMPAT_COUNTERS

_ENGINE_EVENTS = {
    "apcv2_fanout_groups": ("apcv2_fanout", "groups"),
    "apcv2_fanout_lanes": ("apcv2_fanout", "lanes"),
    "apcv2_fanout_boundary_misses": ("apcv2_fanout", "boundary_miss"),
    "apcv2_fanout_store_failures": ("apcv2_fanout", "store_failure"),
    "apcv2_fanout_boundaries": ("apcv2_fanout", "boundary_reused"),
    "apcv2_store_failures": ("apcv2", "store_failure"),
    "apcv2_store_skipped_approximate": ("apcv2", "store_skipped_approximate"),
    "apcv2_write_suppressed_requests": ("apcv2", "write_suppressed_request"),
    "apcv2_fanout_write_suppressed": ("apcv2_fanout", "write_suppressed"),
    "apc_interior_checkpoints_captured": ("apcv2_interior", "captured"),
    "apc_interior_checkpoints_published": ("apcv2_interior", "published"),
    "apc_interior_checkpoints_degraded": ("apcv2_interior", "degraded"),
    "apc_interior_checkpoints_skipped_route": ("apcv2_interior", "skipped_route"),
    "apc_interior_checkpoints_skipped_write_suppressed": (
        "apcv2_interior", "skipped_write_suppressed"
    ),
    "apc_interior_checkpoints_skipped_approximate": (
        "apcv2_interior", "skipped_approximate"
    ),
    "apc_interior_checkpoints_skipped_publish_failed": (
        "apcv2_interior", "skipped_publish_failed"
    ),
    "apc_junction_checkpoints_planned": ("apcv2_junction", "planned"),
    "apc_junction_checkpoints_degraded": ("apcv2_junction", "degraded"),
    "apc_junction_checkpoints_captured": ("apcv2_junction", "captured"),
    "apc_rolling_checkpoints_planned": ("apcv2_rolling", "planned"),
    "apc_rolling_checkpoints_degraded": ("apcv2_rolling", "degraded"),
    "apc_rolling_checkpoints_published": ("apcv2_rolling", "published"),
    "apc_rolling_checkpoints_cancel_published": ("apcv2_rolling", "cancel_published"),
    "apc_rolling_checkpoints_retired": ("apcv2_rolling", "retired"),
    "apc_rolling_checkpoints_retire_deferred": ("apcv2_rolling", "retire_deferred"),
    "apc_rolling_checkpoints_retire_shared": ("apcv2_rolling", "retire_shared"),
    "apc_rolling_checkpoints_skipped_write_suppressed": (
        "apcv2_rolling", "skipped_write_suppressed"
    ),
    "apc_rolling_checkpoints_skipped_publish_failed": (
        "apcv2_rolling", "skipped_publish_failed"
    ),
    "apc_junction_checkpoints_published": ("apcv2_junction", "published"),
    "apc_junction_checkpoints_skipped_write_suppressed": (
        "apcv2_junction", "skipped_write_suppressed"
    ),
    "apc_junction_checkpoints_skipped_approximate": (
        "apcv2_junction", "skipped_approximate"
    ),
    "apc_junction_checkpoints_skipped_publish_failed": (
        "apcv2_junction", "skipped_publish_failed"
    ),
    "apc_interior_positions_planned_turn": ("apcv2_interior", "planned_turn"),
    "apc_interior_positions_planned_tail": ("apcv2_interior", "planned_tail"),
    "apc_interior_positions_planned_lattice": (
        "apcv2_interior", "planned_lattice"
    ),
    "apc_interior_positions_skipped_media": ("apcv2_interior", "skipped_media"),
    "apc_interior_positions_headroom_capped": (
        "apcv2_interior", "headroom_capped"
    ),
    "apc_interior_hits": ("apcv2_interior", "admitted_hit"),
    "apc_interior_hits_turn_boundary": ("apcv2_interior", "turn_boundary_hit"),
    "apc_interior_hit_tokens": ("apcv2_interior", "hit_token"),
    "apc_interior_turn_marker_missing": ("apcv2_interior", "turn_marker_missing"),
    "approximate_kv_applied": ("approximate_kv", "applied"),
    "approximate_kv_declined": ("approximate_kv", "declined"),
    "approximate_kv_requantized_prefix_hits": (
        "approximate_kv",
        "requantized_prefix_hit",
    ),
    "approximate_kv_fanout_bypassed": ("approximate_kv", "fanout_bypassed"),
    "spomin_exact_boundary_stores": ("spomin", "exact_boundary_stored"),
    "spomin_exact_boundary_store_failures": (
        "spomin",
        "exact_boundary_store_failure",
    ),
    "spomin_exact_boundary_snapshot_failures": (
        "spomin",
        "exact_boundary_snapshot_failure",
    ),
    "batch_cohort_attachment_failures": ("batch_cohort", "attachment_failure"),
    "batch_cohort_jobs_failed_closed": ("batch_cohort", "job_failed_closed"),
    "batch_cohort_jobs_released": ("batch_cohort", "job_released"),
    "batch_cohort_jobs_staged": ("batch_cohort", "job_staged"),
    "batch_cohort_jobs_timed_out": ("batch_cohort", "job_timed_out"),
    "batch_cohort_releases": ("batch_cohort", "release"),
    "batch_cohort_scheduler_failures": ("batch_cohort", "scheduler_failure"),
    "batch_cohort_staged_cancellations": ("batch_cohort", "staged_cancellation"),
    "batch_cohort_timeouts": ("batch_cohort", "timeout"),
    "cache_capsule_width_fallbacks": ("cache_capsule", "width_fallback"),
    "cache_capsule_deadline_fallbacks": ("cache_capsule", "deadline_fallback"),
    "cache_capsule_incompatible_fallbacks": ("cache_capsule", "incompatible_fallback"),
    "cache_capsule_prepare_fallbacks": ("cache_capsule", "prepare_fallback"),
    "cache_capsule_prepared": ("cache_capsule", "prepared"),
    "memory_admission_deferred": ("memory_admission", "deferred"),
    "memory_admission_retries": ("memory_admission", "retry"),
    "memory_admission_timeouts": ("memory_admission", "timeout"),
    "memory_admission_depth_floor_admits": ("memory_admission", "depth_floor_admit"),
    "memory_pressure_evictions": ("memory_admission", "cache_eviction"),
    "memory_cache_reclaims_before_reject": (
        "memory_admission", "allocator_reclaim_before_reject"
    ),
    "memory_preemptions": ("memory_preemption", "preempted"),
    "memory_preemptions_stall": ("memory_preemption", "preempted_stall"),
    "memory_preemptions_pressure": ("memory_preemption", "preempted_pressure"),
    "memory_preemptions_fault": ("memory_preemption", "preempted_fault"),
    "preempted_replays": ("memory_preemption", "replayed"),
    # Qualification-mode fault injection that produced no preemption: the
    # eligibility rule declined the lane, or the threshold was never reached.
    "memory_preemption_fault_declined": ("memory_preemption", "fault_declined"),
    "memory_preemption_fault_unfired": ("memory_preemption", "fault_unfired"),
    "memory_preemption_drain_cancellations": (
        "memory_preemption", "drain_cancelled"
    ),
    "mtp_sidecar_missing_misses": ("mtp", "sidecar_missing_cache_miss"),
    "structured_output_failures": ("structured_output", "failed_closed"),
    "structured_output_dead_ends": ("structured_output", "dead_end"),
    "structured_output_completed": ("structured_output", "completed"),
    "structured_engine_automaton": ("structured_output", "engine_automaton"),
    "structured_engine_scanner": ("structured_output", "engine_scanner"),
    "structured_deferred": ("structured_output", "deferred_past_thinking"),
    "thinking_budget_forced_closes": ("thinking_budget", "forced_close"),
    "tool_call_parse_fallbacks": ("tool_calls", "parse_fallback"),
    "tool_call_constraint_failures": ("tool_calls", "parallel_bound_failure"),
    "tool_call_constraint_truncations": ("tool_calls", "parallel_bound_truncated"),
    "constrained_tool_grammar_engagements": ("tool_calls", "decode_grammar_engaged"),
    "constrained_tool_grammar_skips": ("tool_calls", "decode_grammar_skipped"),
    "constrained_tool_grammar_auto_engagements": (
        "tool_calls",
        "decode_grammar_auto_engaged",
    ),
    "constrained_tool_grammar_streams": ("tool_calls", "decode_grammar_streamed"),
    "tolerant_tool_marker_requests": ("tool_calls", "tolerant_markers_requested"),
    "schema_ref_failures": ("json_schema", "reference_failure"),
    "reasoning_signature_rejections": ("reasoning_signature", "rejected"),
    "finish_stop": ("request_finish", "stop"),
    "finish_length": ("request_finish", "length"),
    "finish_unknown": ("request_finish", "unknown"),
    "client_disconnects": ("request_finish", "client_disconnect"),
    "cancelled": ("request_finish", "cancelled"),
    "embeddings_completed": ("representation", "embedding_completed"),
    "embeddings_failed": ("representation", "embedding_failed"),
    "embeddings_prompt_tokens": ("representation", "embedding_prompt_token"),
    "embeddings_vectors": ("representation", "embedding_vector"),
    "rerank_completed": ("representation", "rerank_completed"),
    "rerank_failed": ("representation", "rerank_failed"),
    "rerank_prompt_tokens": ("representation", "rerank_prompt_token"),
    "rerank_documents": ("representation", "rerank_document"),
    "lora_load_completed": ("lora", "loaded"),
    "lora_load_failed": ("lora", "load_failed"),
    "lora_unload_completed": ("lora", "unloaded"),
    "lora_unload_failed": ("lora", "unload_failed"),
    "lora_load_drain_timeouts": ("lora", "load_drain_timeout"),
    "lora_unload_drain_timeouts": ("lora", "unload_drain_timeout"),
    "multimodal_rejected": ("multimodal", "unsupported_adapter"),
    "gemma3n_video_requests": ("multimodal", "gemma3n_video_request"),
    "gemma3n_video_frames": ("multimodal", "gemma3n_video_frame"),
    "gemma3n_video_frame_batches": ("multimodal", "gemma3n_video_frame_batch"),
    "minicpmo_vision_batches": ("multimodal", "minicpmo_vision_batch"),
    "minicpmo_vision_slices": ("multimodal", "minicpmo_vision_slice"),
    "minicpmo_audio_chunks": ("multimodal", "minicpmo_audio_chunk"),
    "multimodal_apcv2_boundary_misses": ("multimodal", "apcv2_boundary_miss"),
    "output_audio_completed": ("output_audio", "completed"),
    "output_audio_failed": ("output_audio", "failed"),
    "output_audio_requests": ("output_audio", "request"),
    "output_audio_bytes": ("output_audio", "byte"),
    # MoE expert disk streaming (server-owned moe_expert_streaming policy).
    "stream_page_ins_total": ("expert_stream", "page_in"),
    "stream_page_in_bytes_total": ("expert_stream", "page_in_byte"),
    "stream_expert_hits_total": ("expert_stream", "hit"),
    "stream_expert_misses_total": ("expert_stream", "miss"),
    "stream_evictions_total": ("expert_stream", "eviction"),
    "stream_admission_refusals_total": ("expert_stream", "working_set_refusal"),
    # Atlas collection only. These count observations, never pinned bytes:
    # the atlas does not influence residency in this iteration.
    "atlas_observations_total": ("expert_atlas", "observation"),
    "atlas_trace_records_total": ("expert_atlas", "trace_record"),
    "atlas_trace_dropped_total": ("expert_atlas", "trace_dropped"),
}
# Agent-client wire compatibility (``--agent-compat``) mechanism counters.
_ENGINE_EVENTS.update(
    {key: ("agent_compat", event) for key, event in _AGENT_COMPAT_COUNTERS.items()}
)

# Default-off mechanisms.  Their counters do not exist on a server that never
# enabled the policy, and an unconditional zero series would make a default
# /metrics scrape differ from main's byte for byte.  These are rendered only
# once the engine actually holds the key.
_OPTIONAL_COMPONENTS = frozenset(
    {"apcv2_rolling", "apcv2_junction", "memory_preemption"}
)
_OPTIONAL_ENGINE_EVENTS = {
    key: value
    for key, value in _ENGINE_EVENTS.items()
    if value[0] in _OPTIONAL_COMPONENTS
}
for _optional_key in _OPTIONAL_ENGINE_EVENTS:
    del _ENGINE_EVENTS[_optional_key]

_MEMORY_GAUGES = {
    "metal_active_bytes": "mlx2_memory_active_bytes",
    "metal_peak_bytes": "mlx2_memory_peak_bytes",
    "process_physical_footprint_bytes": "mlx2_process_physical_footprint_bytes",
    "headroom_bytes": "mlx2_memory_headroom_bytes",
    "memory_waiting": "mlx2_memory_waiting_requests",
    # Present only when the server-owned host_memory_signals policy is on.
    # Pressure level: 0 normal, 1 warn, 2 critical (after fall hysteresis).
    "host_memory_pressure_level": "mlx2_host_memory_pressure_level",
    "host_memory_available_bytes": "mlx2_host_memory_available_bytes",
}

_STREAM_GAUGES = {
    # Bytes of expert weight currently held by the bounded per-layer LRU.
    # Present only when the server-owned moe_expert_streaming policy is on.
    "stream_resident_bytes": "mlx2_expert_stream_resident_bytes",
}

_SCHEDULER_GAUGES = frozenset(
    {"target_max_width", "draft_max_width", "reservation_bytes"}
)

# Export schemas are intentionally independent of runtime dictionary keys.
# Adding a private diagnostic therefore cannot silently create a public label
# value; a maintainer must add it to this reviewable registry first.
_SCHEDULER_EVENTS = frozenset(
    {
        "prefill_rounds",
        "prefill_only_rounds",
        "decode_priority_release_rounds",
        "decode_priority_deferred_rounds",
        "adaptive_prefill_release_rounds",
        "adaptive_prefill_slack_deferred_rounds",
        "adaptive_prefill_deadline_forced_rounds",
        "adaptive_prefill_apc_priority_admissions",
        "mtp_short_prefill_interleaved",
        "short_prefill_overflow_admissions",
        "cache_capsule_fallbacks",
        "cache_capsule_engaged",
        "atomic_cohort_admission_failures",
        "atomic_cohort_lifecycle_failures",
        "atomic_cohort_prefill_holds",
        "starved_mtp_plain_fallbacks",
        "external_rounds",
        "accepted_proposals",
        "proposed_tokens",
        "ordinary_rounds",
        "cancelled",
        "paired_cache_resumes",
        "segmented_transactions",
        "segmented_rollbacks",
        "draft_fallbacks",
        "external_draft_masked_positions",
        "external_ordinary_fast_path_rounds",
        "external_ordinary_fast_path_lanes",
        "external_draft_context_skipped",
        "external_taps_skipped",
        "external_transactions_skipped",
        "external_cow_snapshots",
        "external_cow_fallbacks",
        "fly_relaxed_accepts",
        "external_pairwise_selection_groups",
        "external_pairwise_selection_lanes",
        "memory_deferred",
        "memory_pressure_evictions",
        "stream_page_ins_total",
        "stream_page_in_bytes_total",
        "stream_expert_hits_total",
        "stream_expert_misses_total",
        "stream_evictions_total",
        "stream_admission_refusals_total",
        "atlas_observations_total",
        "atlas_trace_records_total",
        "atlas_trace_dropped_total",
        "pld_cycles",
        "pld_retrieval_cycles",
        "pld_plain_cycles",
        "pld_proposed",
        "pld_accepted",
        "pld_bonus",
        "pld_fallbacks",
        "pld_rollbacks",
        "pld_prefill_rounds",
        "pld_rotating_replay_rounds",
        "pld_rotating_replay_replayed_tokens",
        "pld_rotating_replay_refusals",
        "pld_rotating_replay_rebuilds",
        "pld_cow_snapshots",
        "pld_cow_fallbacks",
        "decode_fairness_prefill_chunks",
        "prefill_chunk_rounds_recorded",
        "prefill_chunk_varied_requests",
        "decode_fairness_debt_deferrals",
        "decode_fairness_debt_repayments",
        "decode_fairness_cap_clamps",
        "prefill_scheduling_bypasses",
        "prefill_scheduling_bypass_forced",
        "prefill_scheduling_one_slice_clamps",
        "adaptive_mtp_boundaries",
        "adaptive_mtp_depth_changes",
        "adaptive_mtp_parks",
        "adaptive_mtp_reentries",
        "adaptive_mtp_probes",
        "adaptive_mtp_cost_probes",
        "adaptive_mtp_cost_depth_changes",
        "adaptive_mtp_depth_decreases_concurrent",
        "adaptive_mtp_depth_recoveries_alone",
        "mtp_ordinary_handoff_events",
        "mtp_ordinary_handoff_lanes",
        "mtp_ordinary_handoff_static_width_threshold",
        "mtp_ordinary_handoff_segmented_width_lock",
        "mtp_confidence_feature_cycles",
        "mtp_acceptance_log_records",
        "self_mtp_zero_fast_rounds",
        "self_mtp_zero_draft_forwards_skipped",
        "self_mtp_zero_proposal_roundtrips_skipped",
        "self_mtp_copy_rounds",
        "self_mtp_copy_proposed_tokens",
        "self_mtp_copy_accepted_tokens",
        "self_mtp_copy_probe_rounds",
        "self_mtp_copy_gate_declines",
        "self_mtp_copy_lookup_misses",
        "apc_interior_checkpoints_skipped_trimmable",
        "apc_interior_checkpoints_skipped_inexact",
        "apc_rolling_checkpoints_captured",
        "apc_rolling_checkpoints_skipped_inexact",
        "apc_rolling_checkpoints_skipped_pressure",
        "apc_junction_checkpoints_captured",
        "apc_junction_checkpoints_skipped_inexact",
    }
)
_PREFILL_CHUNK_LABELS = frozenset(
    str(1 << exponent) for exponent in range(4, 14)
)
_ADAPTIVE_MTP_WIDTH_BUCKETS = frozenset({"1", "2", "3-4", "5-8", "9-16", "17+"})
_ADAPTIVE_MTP_DEPTHS = frozenset(str(depth) for depth in range(9))

_SEGMENTED_MTP_TIMERS = frozenset(
    {"proposal_ns", "commit_ns", "async_qsa_promotion_wait_ns"}
)
_SEGMENTED_MTP_GAUGES = frozenset(
    {
        "private_delta_base_tokens_last",
        "private_delta_base_tokens_min",
        "private_delta_base_tokens_max",
    }
)
_SEGMENTED_MTP_BYTES = frozenset(
    {
        "recurrent_state_materialized_bytes",
        "private_delta_duplicate_base_storage_bytes_not_formed_cumulative",
        "shared_qsa_base_bytes",
        "shared_qsa_private_bytes",
        "shared_qsa_materialized_bytes",
        "full_prefix_materialized_bytes",
        "async_qsa_promotion_patched_bytes",
        # Accumulated once per promotion, so it is a lifetime byte total, not
        # a current or high-water reservation.
        "async_qsa_promotion_reserved_bytes",
    }
)
_SEGMENTED_MTP_EVENTS = frozenset(
    {
        "requests", "physical_join_flag_reconciled", "width_lock_plain_fallbacks",
        "engaged", "declined", "failures", "fallback_physical_b2",
        "physical_b2_formations", "b1_target_forwards", "b1_draft_forwards",
        "batched_target_forwards", "batched_draft_forwards",
        "true_batched_requests", "true_batched_engaged", "true_batched_declined",
        "live_width_change_deferrals", "layer_local_materializations",
        "recurrent_state_materializations", "row_state_splits",
        "segmented_attention_calls", "private_delta_requests",
        "private_delta_declines", "private_delta_preflight_declines",
        "private_delta_late_gather_fallbacks", "private_delta_attention_calls",
        "private_delta_rows", "private_delta_base_tokens_cumulative",
        "shared_qsa_rows", "shared_qsa_materializations",
        "shared_qsa_batched_selections", "shared_qsa_policy_checks",
        "shared_qsa_policy_admitted", "shared_qsa_policy_declined",
        "shared_qsa_policy_context_tokens_cumulative",
        "shared_qsa_policy_remaining_tokens_cumulative",
        "shared_qsa_policy_cutoff_tokens_cumulative", "exact_set_fold_requests",
        "exact_set_fold_declines", "exact_set_fold_preflight_declines",
        "exact_set_fold_private_fallbacks", "exact_set_fold_attention_calls",
        "exact_set_fold_rows", "exact_set_fold_device_proofs",
        "independent_lineages_consumed", "full_prefix_materializations",
        "transaction_branches", "transaction_promotions", "transaction_rejections",
        "transaction_canonicalizations", "committed_cycles", "accepted_zero",
        "accepted_partial", "accepted_all", "zero_depth_fast_rounds",
        "async_qsa_promotion_requests",
        "async_qsa_promotion_queued", "async_qsa_promotion_engaged",
        "async_qsa_promotion_declined", "async_qsa_promotion_declined_shared_suffix",
        "async_qsa_promotion_declined_memory", "async_qsa_admission_settled",
        "async_qsa_promotion_failures", "async_qsa_prequeue_requests",
        "async_qsa_prequeue_queued", "async_qsa_prequeue_bound",
        "async_qsa_prequeue_declined", "async_qsa_shared_prefix_fused_layers",
        "async_qsa_budget_checks", "async_qsa_budget_retained_segmented",
        "async_qsa_budget_promotions", "async_qsa_budget_remaining_tokens_cumulative",
        "async_qsa_budget_cutoff_tokens_cumulative", "device_synchronizations",
    }
    | {
        f"private_delta_width_{width}_{event}"
        for width in (*range(1, 10), "other")
        for event in ("requests", "engaged", "declined")
    }
)

_MULTI_LORA_EVENTS = frozenset(
    {
        "forwards", "base_only_forwards", "mixed_forwards",
        "single_adapter_forwards", "rows_base", "rows_adapter", "slot_hits",
        "slot_loads", "slot_evictions", "slot_deferred", "registrations",
        "unregistrations", "structural_wraps", "materializations",
        "delta_applications", "delta_skips",
    }
)

_SPOMIN_OPERATIONS = frozenset(
    {
        "applied", "declined", "prepared",
        "reason:disabled", "reason:transcript_unavailable",
        "reason:cache_not_request_private", "reason:transcript_prompt_mismatch",
        "reason:mtp_state_active", "reason:recurrent_state_unrepairable",
        "reason:importance_scores_unavailable", "reason:below_pressure",
        "reason:target_unreachable", "reason:stale_epoch",
        "reason:request_not_quiescent", "reason:device_work_not_drained",
        "reason:backend_unavailable", "reason:capability_refused",
        "reason:revision_changed", "reason:plan_invalid", "reason:backend_refused",
        "reason:committed", "reason:backend_error", "reason:closed_without_apply",
    }
)
_COW_GAUGES = frozenset({"active_leases", "peak_leases"})
_COW_BYTES = frozenset(
    {
        "source_bytes", "descriptor_bytes", "avoided_copy_bytes",
        "materialized_bytes", "fallback_bytes",
    }
)
_COW_TIMERS = frozenset({"freeze_ns", "branch_ns", "fallback_ns"})
_COW_EVENTS = frozenset(
    {
        "sources", "freeze_failures", "branches", "branch_failures",
        "stale_rejections", "invalidations", "descriptor_aliases",
        "materializations", "fallback_deepcopies", "releases",
    }
)


def _copy_engine_state(engine: Any) -> tuple[dict[str, Any], dict[str, int], int]:
    lock = getattr(engine, "lock", None)
    if lock is not None and hasattr(engine, "snapshot"):
        with lock:
            snapshot = dict(getattr(engine, "snapshot", {}))
            snapshot["service_state"] = str(
                getattr(engine, "_service_state", "serving")
            )
            counts = dict(getattr(engine, "counts", {}))
            # Expert-stream and atlas counters live on the stream handle, not
            # in engine.counts; merge them the way status() does, or /metrics
            # reports them as zero while streaming is active.
            stream_counters = getattr(engine, "expert_stream_counters", None)
            if callable(stream_counters):
                counts.update(stream_counters())
            queue_depth = int(getattr(engine, "queued_jobs", 0))
        return snapshot, counts, queue_depth
    status = dict(engine.status())
    return status, dict(status.get("counts") or {}), int(status.get("queue_depth", 0))


def _add_batch_metrics(
    builder: PrometheusBuilder, engine: Any, queue_depth: int
) -> None:
    metrics = getattr(engine, "batch_metrics", None)
    if metrics is None or not hasattr(metrics, "prometheus_snapshot"):
        try:
            gauges = dict(engine.batching_status().get("gauges") or {})
        except (AttributeError, TypeError):
            gauges = {}
        builder.gauge(
            "mlx2_num_requests_waiting",
            "Requests waiting for an execution lane.",
            gauges.get("queue_depth", queue_depth),
        )
        builder.gauge(
            "mlx2_num_requests_running",
            "Requests currently owned by the serving engine.",
            gauges.get("inflight_requests", 0),
        )
        return

    data = metrics.prometheus_snapshot(queue_depth=queue_depth)
    for name, value in data["gauges"].items():
        builder.gauge(name, data["help"][name], value)
    for (name, labels), value in data["counters"].items():
        builder.counter(name, data["help"][name], value, dict(labels))
    for name, histogram in data["histograms"].items():
        builder.histogram(name, data["help"][name], histogram)


def _add_http_metrics(builder: PrometheusBuilder, engine: Any) -> None:
    metrics = getattr(engine, "http_metrics", None)
    if metrics is None or not hasattr(metrics, "prometheus_snapshot"):
        return
    data = metrics.prometheus_snapshot()
    for (method, route, status_class), value in sorted(data["requests"].items()):
        builder.counter(
            "mlx2_http_requests_total",
            "HTTP requests by bounded method, route, and status class.",
            value,
            {"method": method, "route": route, "status_class": status_class},
        )
    for route, histogram in sorted(data["durations"].items()):
        builder.histogram(
            "mlx2_http_request_duration_seconds",
            "HTTP request duration by bounded route.",
            histogram,
            {"route": route},
        )

    tracer = getattr(engine, "request_tracer", None)
    if tracer is not None and hasattr(tracer, "prometheus_snapshot"):
        trace_data = tracer.prometheus_snapshot()
        for outcome in ("started", "completed", "failed", "exported"):
            builder.counter(
                "mlx2_trace_spans_total",
                "Optional OTLP request spans by bounded lifecycle outcome.",
                trace_data.get(outcome, 0),
                {"outcome": outcome},
            )
        for stage in ("setup", "export"):
            builder.counter(
                "mlx2_trace_export_errors_total",
                "Optional OTLP tracing setup and asynchronous export errors.",
                trace_data.get(f"{stage}_errors", 0),
                {"stage": stage},
            )


def _add_response_store(builder: PrometheusBuilder, engine: Any, counts) -> None:
    store = getattr(engine, "response_store", None)
    if store is None or not hasattr(store, "status"):
        return
    status = store.status()
    builder.gauge(
        "mlx2_response_store_entries",
        "Responses objects currently retained in process-local state.",
        int(status.get("entries", 0)),
    )
    builder.gauge(
        "mlx2_response_store_bytes",
        "Serialized bytes currently retained in process-local Responses state.",
        int(status.get("bytes", 0)),
    )
    for event in ("stores", "hits", "misses"):
        builder.counter(
            f"mlx2_response_store_{event}_total",
            f"Process-local Responses store {event}.",
            int(status.get(event, 0)),
        )
    for reason in ("ttl", "lru", "bytes"):
        builder.counter(
            "mlx2_response_store_evictions_total",
            "Process-local Responses store evictions by bounded reason.",
            int(status.get(f"evictions_{reason}", 0)),
            {"reason": reason},
        )
    for reason in ("too_large", "write_error"):
        builder.counter(
            "mlx2_response_store_skipped_total",
            "Completed Responses objects skipped by process-local state retention.",
            int(status.get(f"skipped_{reason}", 0)),
            {"reason": reason},
        )
    builder.counter(
        "mlx2_thinking_signature_rejections_total",
        "Signed reasoning input blocks dropped after authentication failure.",
        int(counts.get("thinking_signature_rejections", 0)),
    )


def _add_capabilities(builder: PrometheusBuilder, snapshot: Mapping[str, Any]) -> None:
    qualification = str(snapshot.get("qualification") or "unavailable")
    profile = str(snapshot.get("profile") or "unavailable")
    builder.gauge(
        "mlx2_route_info",
        "Selected route identity; the sample value is always one.",
        1,
        {"profile": profile, "qualification": qualification},
    )
    selected = frozenset(
        str(value)
        for value in snapshot.get(
            "selected_capabilities", snapshot.get("capabilities") or ()
        )
    )
    implemented = frozenset(
        str(value)
        for value in snapshot.get("implemented_capabilities", selected)
    )
    qualified = frozenset(
        str(value)
        for value in snapshot.get(
            "qualified_capabilities",
            selected if qualification == "qualified" else (),
        )
    )
    for capability in sorted(implemented | selected | qualified):
        for state, value in (
            ("implemented", int(capability in implemented)),
            ("selected", int(capability in selected)),
            ("qualified", int(capability in qualified)),
        ):
            builder.gauge(
                "mlx2_capability_state",
                "Bounded lifecycle state for a declared route capability.",
                value,
                {"capability": capability, "state": state},
            )


def _add_apcv2(
    builder: PrometheusBuilder,
    apc: Mapping[str, Any],
    live_lifetime: Mapping[str, Any] | None = None,
) -> None:
    # Prefer the live counters: ``snapshot["apcv2"]`` is refreshed at most
    # once a second and is seeded all-zero at readiness, so reading hits from
    # it reports 0 for a run shorter than the cadence.
    lifetime = dict(live_lifetime if live_lifetime is not None
                    else (apc.get("lifetime") or {}))
    for key in ("lookups", "hits", "misses", "stores"):
        builder.counter(
            f"mlx2_prefix_cache_{key}_total",
            f"Lifetime APCv2 prefix-cache {key}.",
            int(lifetime.get(key, 0)),
        )
    builder.counter(
        "mlx2_cached_prompt_tokens_total",
        "Prompt tokens reused from APCv2 over the process lifetime.",
        int(lifetime.get("cached_tokens", 0)),
    )
    builder.counter(
        "mlx2_prefix_cache_query_tokens_total",
        "Prompt tokens presented to APCv2 lookups.",
        int(lifetime.get("queried_tokens", 0)),
    )
    builder.counter(
        "mlx2_prefix_cache_interior_hits_total",
        "APCv2 hits served from budgeted interior checkpoints.",
        int(lifetime.get("interior_hits", 0)),
    )
    builder.counter(
        "mlx2_prefix_cache_rolling_hits_total",
        "APCv2 hits served from disposable rolling prefill checkpoints.",
        int(lifetime.get("rolling_hits", 0)),
    )
    builder.counter(
        "mlx2_prefix_cache_junction_hits_total",
        "APCv2 hits served from junction checkpoints.",
        int(lifetime.get("junction_hits", 0)),
    )
    reuse = dict(apc.get("reuse_telemetry") or {})
    for key, metric, help_text in (
        (
            "hit_age_seconds",
            "mlx2_prefix_cache_hit_age_seconds",
            "Age of an APCv2 entry when it serves a hit.",
        ),
        (
            "eviction_age_seconds",
            "mlx2_prefix_cache_eviction_age_seconds",
            "Age of an APCv2 entry when it is evicted or dropped.",
        ),
        (
            "eviction_idle_seconds",
            "mlx2_prefix_cache_eviction_idle_seconds",
            "Idle time since last access when an APCv2 entry is evicted or dropped.",
        ),
        (
            "eviction_hit_count",
            "mlx2_prefix_cache_eviction_hits",
            "Hit count of an APCv2 entry when it is evicted or dropped.",
        ),
    ):
        raw = reuse.get(key)
        if not isinstance(raw, Mapping):
            continue
        builder.histogram(
            metric,
            help_text,
            HistogramSnapshot(
                tuple(float(value) for value in raw.get("buckets", ())),
                tuple(int(value) for value in raw.get("bucket_counts", ())),
                int(raw.get("count", 0)),
                float(raw.get("sum", 0.0)),
            ),
        )
    layers = dict(apc.get("layer_segments") or {})
    builder.gauge(
        "mlx2_prefix_cache_entries",
        "Resident APCv2 prefix-cache entries.",
        int(layers.get("entries", 0)),
    )
    builder.gauge(
        "mlx2_prefix_cache_logical_bytes",
        "Logical bytes represented by resident APCv2 segmented state.",
        int(layers.get("logical_bytes", 0)),
    )
    capacity = int(apc.get("resident_max_bytes", 0))
    if capacity > 0:
        builder.gauge(
            "mlx2_prefix_cache_capacity_bytes",
            "Configured APCv2 resident byte capacity.",
            capacity,
        )
        builder.gauge(
            "mlx2_prefix_cache_usage_ratio",
            "APCv2 resident bytes divided by configured resident capacity.",
            min(
                1.0,
                max(
                    0.0,
                    int((apc.get("idle_disk") or {}).get("resident_bytes", 0))
                    / capacity,
                ),
            ),
        )
    disk = dict(apc.get("idle_disk") or {})
    for key in ("resident_bytes", "disk_bytes", "disk_entries"):
        builder.gauge(
            f"mlx2_prefix_cache_{key}",
            f"Current APCv2 {key.replace('_', ' ')}.",
            int(disk.get(key, 0)),
        )
    for key in (
        "idle_spills",
        "pressure_spills",
        "restores",
        "restore_failures",
        "restore_budget_deferrals",
        "spill_failures",
        "disk_evictions",
        "restore_digest_failures",
    ):
        builder.counter(
            "mlx2_prefix_cache_disk_events_total",
            "APCv2 disk-tier events by bounded operation.",
            int(disk.get(key, 0)),
            {"operation": key},
        )
    for key in (
        "parks",
        "resumes",
        "prefetch_restores_ok",
        "prefetch_restores_abandoned",
        "prefetch_restores_failed",
        "prefetch_restores_cancelled",
        "prefetch_hits",
        "prefetch_misses",
        "session_deletes",
        "pin_expiries",
        "pin_cap_rejections",
        "publication_rejections",
        "persistence_io_failures",
        "persisted_writes",
        "quarantined",
        "quarantine_evictions",
    ):
        builder.counter(
            "mlx2_prefix_cache_session_events_total",
            "APCv2 session and persistent-tier events by bounded operation.",
            int(disk.get(key, 0)),
            {"operation": key},
        )
    for key, direction in (("bytes_written", "written"), ("bytes_read", "read")):
        builder.counter(
            "mlx2_prefix_cache_disk_bytes_total",
            "Bytes transferred by the APCv2 disk tier.",
            int(disk.get(key, 0)),
            {"direction": direction},
        )

    rescan = dict((apc.get("persistence") or {}).get("rescan") or {})
    quarantine = dict((apc.get("persistence") or {}).get("quarantine") or {})
    for key in ("entries", "bytes"):
        builder.gauge(
            f"mlx2_prefix_cache_quarantine_{key}",
            f"Current APCv2 quarantine {key}.",
            int(quarantine.get(key, 0)),
        )
    builder.counter(
        "mlx2_prefix_cache_rescan_registered_total",
        "Persisted APCv2 entries registered by startup rescan.",
        int(rescan.get("registered", 0)),
    )
    builder.gauge(
        "mlx2_prefix_cache_rescan_registered_bytes",
        "Persisted APCv2 bytes registered by startup rescan.",
        int(rescan.get("registered_bytes", 0)),
    )
    builder.gauge(
        "mlx2_prefix_cache_rescan_seconds",
        "Duration of the APCv2 startup manifest rescan.",
        float(rescan.get("elapsed_seconds", 0.0)),
    )
    for reason in ("identity_mismatch", "unknown_schema", "corruption"):
        builder.counter(
            "mlx2_prefix_cache_rescan_discarded_total",
            "Persisted APCv2 entries discarded at startup by bounded reason.",
            int((rescan.get("discarded") or {}).get(reason, 0)),
            {"reason": reason},
        )

    capsules = dict(apc.get("cache_capsules") or {})
    builder.gauge(
        "mlx2_cache_capsule_reserved_bytes",
        "Bytes currently reserved for APCv2 cache capsules.",
        int(capsules.get("reserved_bytes", 0)),
    )
    for key in ("reservations", "reservation_rejections"):
        builder.counter(
            "mlx2_cache_capsule_capacity_events_total",
            "APCv2 cache-capsule capacity events.",
            int(capsules.get(key, 0)),
            {"event": key},
        )

    cow = dict(apc.get("cow") or {})
    for key, value in cow.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if key in _COW_GAUGES:
            builder.gauge(
                "mlx2_prefix_cache_cow_leases",
                "Current and peak APCv2 COW cache leases.",
                value,
                {"kind": key.removesuffix("_leases")},
            )
        elif key in _COW_BYTES:
            builder.counter(
                "mlx2_prefix_cache_cow_bytes_total",
                "APCv2 COW bytes by bounded operation.",
                value,
                {"operation": key.removesuffix("_bytes")},
            )
        elif key in _COW_TIMERS:
            builder.counter(
                "mlx2_prefix_cache_cow_time_seconds_total",
                "Cumulative APCv2 COW host timing when timing is enabled.",
                float(value) / 1_000_000_000.0,
                {"operation": key.removesuffix("_ns")},
            )
        elif key in _COW_EVENTS:
            builder.counter(
                "mlx2_prefix_cache_cow_events_total",
                "APCv2 COW events by bounded operation.",
                value,
                {"operation": key},
            )


def _scheduler_mechanism(key: str) -> str:
    if key.startswith("pld_"):
        return "prompt_lookup"
    if key.startswith("mtp_ordinary_handoff_"):
        return "mtp_ordinary_handoff"
    if key.startswith("adaptive_mtp_"):
        return "adaptive_mtp"
    if key.startswith("self_mtp_copy_"):
        return "self_mtp_copy_draft"
    if key.startswith(("mtp_confidence_", "mtp_acceptance_")):
        # The draft-feature probe and the acceptance log ride the self-MTP
        # round and are independent of the adaptive-depth controller; a run
        # with only the log on must not look like adaptive depth ran.
        return "self_mtp"
    if key.startswith("self_mtp_"):
        return "self_mtp"
    if key.startswith("prefill_chunk_"):
        return "prefill_chunk"
    if key.startswith("decode_fairness_"):
        return "decode_fairness"
    if key.startswith("prefill_scheduling_"):
        return "prefill_scheduling"
    if key.startswith("cache_capsule_"):
        return "cache_capsule"
    if key.startswith("adaptive_prefill_"):
        return "adaptive_prefill"
    if key in {
        "accepted_proposals",
        "proposed_tokens",
        "external_rounds",
        "draft_fallbacks",
        "external_draft_masked_positions",
        "external_ordinary_fast_path_rounds",
        "external_ordinary_fast_path_lanes",
        "external_draft_context_skipped",
        "external_taps_skipped",
        "external_transactions_skipped",
        "external_cow_snapshots",
        "external_cow_fallbacks",
        "external_pairwise_selection_groups",
        "external_pairwise_selection_lanes",
    }:
        return "external_speculative"
    if key == "fly_relaxed_accepts":
        return "fly_verification"
    return "scheduler"


def _add_scheduler(builder: PrometheusBuilder, scheduler: Mapping[str, Any]) -> None:
    for key, raw in sorted(scheduler.items()):
        if key == "adaptive_mtp_cost_model":
            buckets = (raw or {}).get("buckets", {}) if isinstance(raw, Mapping) else {}
            for width_bucket, state in sorted(buckets.items()):
                if width_bucket not in _ADAPTIVE_MTP_WIDTH_BUCKETS or not isinstance(
                    state, Mapping
                ):
                    continue
                chosen_depth = state.get("chosen_depth")
                if isinstance(chosen_depth, (int, float)) and not isinstance(
                    chosen_depth, bool
                ):
                    builder.gauge(
                        "mlx2_adaptive_mtp_chosen_depth",
                        "Current cost-aware native-MTP depth by bounded compute-width bucket.",
                        chosen_depth,
                        {"width_bucket": width_bucket},
                    )
                estimates = state.get("goodput_tokens_per_second") or {}
                if not isinstance(estimates, Mapping):
                    continue
                for depth, estimate in sorted(estimates.items()):
                    depth_label = str(depth)
                    if depth_label not in _ADAPTIVE_MTP_DEPTHS:
                        continue
                    if not isinstance(estimate, (int, float)) or isinstance(
                        estimate, bool
                    ):
                        continue
                    builder.gauge(
                        "mlx2_adaptive_mtp_goodput_tokens_per_second",
                        "EWMA committed-token goodput by bounded width bucket and depth.",
                        estimate,
                        {"depth": depth_label, "width_bucket": width_bucket},
                    )
            continue
        if key == "adaptive_prefill_chunk_histogram":
            bounded_chunks: dict[str, int] = {}
            for chunk, count in (raw or {}).items():
                token_label = str(chunk)
                if token_label not in _PREFILL_CHUNK_LABELS:
                    token_label = "other"
                bounded_chunks[token_label] = (
                    bounded_chunks.get(token_label, 0) + int(count)
                )
            for token_label, count in sorted(bounded_chunks.items()):
                builder.counter(
                    "mlx2_scheduler_prefill_chunks_total",
                    "Adaptive prefill chunks selected, grouped by bounded configured size.",
                    count,
                    {"tokens": token_label},
                )
            continue
        if key not in _SCHEDULER_EVENTS and key not in _SCHEDULER_GAUGES:
            continue
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            continue
        mechanism = _scheduler_mechanism(key)
        if key in _SCHEDULER_GAUGES:
            builder.gauge(
                "mlx2_scheduler_state",
                "Current scheduler state and bounded high-water values.",
                raw,
                {"mechanism": mechanism, "state": key},
            )
        else:
            builder.counter(
                "mlx2_scheduler_events_total",
                "Scheduler and speculative-path events by bounded mechanism and event.",
                raw,
                {"mechanism": mechanism, "event": key},
            )


def _add_segmented_mtp(builder: PrometheusBuilder, stats: Mapping[str, Any]) -> None:
    for key, raw in sorted(stats.items()):
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            continue
        if key in _SEGMENTED_MTP_TIMERS:
            builder.counter(
                "mlx2_segmented_mtp_time_seconds_total",
                "Cumulative segmented self-MTP host timing when explicitly enabled.",
                float(raw) / 1_000_000_000.0,
                {"operation": key.removesuffix("_ns")},
            )
        elif key in _SEGMENTED_MTP_BYTES:
            builder.counter(
                "mlx2_segmented_mtp_bytes_total",
                "Segmented self-MTP bytes by bounded operation.",
                raw,
                {"operation": key.removesuffix("_cumulative").removesuffix("_bytes")},
            )
        elif key in _SEGMENTED_MTP_GAUGES:
            builder.gauge(
                "mlx2_segmented_mtp_state",
                "Current or high-water segmented self-MTP state.",
                raw,
                {"state": key},
            )
        elif key in _SEGMENTED_MTP_EVENTS:
            builder.counter(
                "mlx2_segmented_mtp_events_total",
                "Segmented self-MTP engagement and lifecycle events.",
                raw,
                {"event": key},
            )


def _add_cache_capsules(builder: PrometheusBuilder, stats: Mapping[str, Any]) -> None:
    for outcome in (
        "requests",
        "primary_successes",
        "fallbacks",
        "timeouts",
        "errors",
        "stale",
        "capacity_rejections",
        "late_disposals",
    ):
        builder.counter(
            "mlx2_cache_capsule_operations_total",
            "Cache-capsule operations by bounded outcome.",
            int(stats.get(outcome, 0)),
            {"outcome": outcome},
        )


def _add_execution(builder: PrometheusBuilder, execution: Mapping[str, Any]) -> None:
    """Export only reviewed advanced-path fields from adapter diagnostics."""

    media_cache = execution.get("media_feature_cache")
    if isinstance(media_cache, Mapping):
        for key in (
            "hits",
            "misses",
            "stores",
            "evictions",
            "oversize_skips",
            "clears",
            "cleared_entries",
        ):
            builder.counter(
                "mlx2_media_feature_cache_events_total",
                "Projected media-feature cache events by bounded outcome.",
                int(media_cache.get(key, 0)),
                {"event": key},
            )
        builder.gauge(
            "mlx2_media_feature_cache_entries",
            "Projected media-feature cache entries currently resident.",
            int(media_cache.get("entries", 0)),
        )
        builder.gauge(
            "mlx2_media_feature_cache_bytes",
            "Projected media-feature cache bytes currently resident.",
            int(media_cache.get("bytes", 0)),
        )

    moe = execution.get("moe")
    if isinstance(moe, Mapping):
        dispatches = moe.get("dispatches")
        if isinstance(dispatches, Mapping):
            for mode in ("scalar", "tile4"):
                builder.counter(
                    "mlx2_moe_dispatches_total",
                    "MoE expert dispatches by bounded kernel mode.",
                    int(dispatches.get(mode, 0)),
                    {"mode": mode},
                )
        builder.counter(
            "mlx2_moe_router_calls_total",
            "MoE router calls.",
            int(moe.get("router_calls", 0)),
        )
        builder.counter(
            "mlx2_moe_fallbacks_total",
            "MoE fused-expert fallbacks.",
            int(moe.get("fallbacks", 0)),
        )
        builder.gauge(
            "mlx2_moe_fused_gate_up_layers",
            "Model layers using the fused MoE gate-up path.",
            int(moe.get("fused_gate_up_layers", 0)),
        )

    levers = execution.get("round_levers")
    if isinstance(levers, Mapping):
        for key in (
            "ple_tail_prefetch_requests", "ple_tail_prefetch_declined",
            "ple_tail_prefetch_failures", "ple_prefetch_submitted",
            "ple_dq_hits", "ple_dq_misses", "device_sampled_drafts",
            "hedge_built", "hedge_hit", "hedge_miss", "hedge_consumed",
            "hedge_skipped", "hedge_discarded", "eager_async_evals",
            "eager_dispatch_forwards", "eager_dispatch_row_declines",
            "qsa_pooled_key_cache_hits", "qsa_pooled_key_cache_misses",
            "qsa_scatter_chosen_calls",
        ):
            builder.counter(
                "mlx2_advanced_path_events_total",
                "Advanced execution-path events from a bounded schema.",
                int(levers.get(key, 0)),
                {"event": key},
            )
        builder.counter(
            "mlx2_ple_prefetched_rows_total",
            "PLE rows submitted for prefetch.",
            int(levers.get("ple_prefetch_rows", 0)),
        )
        builder.counter(
            "mlx2_ple_tail_prefetched_tables_total",
            "PLE tail tables requested for prefetch.",
            int(levers.get("ple_tail_prefetch_tables", 0)),
        )

    compile_status = execution.get("ple_compile")
    if isinstance(compile_status, Mapping):
        builder.gauge(
            "mlx2_ple_compile_enabled",
            "Whether the compiled PLE chain is selected.",
            int(bool(compile_status.get("enabled", False))),
        )
        builder.gauge(
            "mlx2_ple_compile_cache_limit",
            "Configured upper bound on compiled PLE cache entries.",
            int(compile_status.get("cache_max", 0)),
        )
        counts = compile_status.get("counts")
        if isinstance(counts, Mapping):
            for event in (
                "builds", "hits", "fallbacks", "overflow", "skips",
                "retraces", "invalidations",
            ):
                builder.counter(
                    "mlx2_ple_compile_events_total",
                    "Compiled PLE lifecycle events.",
                    int(counts.get(event, 0)),
                    {"event": event},
                )

    gdn = execution.get("fused_gdn")
    if isinstance(gdn, Mapping):
        for event in (
            "fused_calls", "fused_outproj_calls", "fallbacks", "verify_calls",
            "verify_fallbacks", "replay_verify_calls", "replay_rollback_calls",
            "replay_fallbacks", "catchup_calls", "catchup_fallbacks",
        ):
            builder.counter(
                "mlx2_fused_gdn_events_total",
                "Fused GDN execution events.",
                int(gdn.get(event, 0)),
                {"event": event},
            )
        builder.counter(
            "mlx2_fused_gdn_rollback_tokens_total",
            "Tokens rolled back by fused GDN replay.",
            int(gdn.get("replay_rollback_tokens", 0)),
        )
        if "replay_dynamic_rollback_calls" in gdn:
            # Present only when device-count rollback is selected.
            builder.counter(
                "mlx2_fused_gdn_events_total",
                "Fused GDN execution events.",
                int(gdn.get("replay_dynamic_rollback_calls", 0)),
                {"event": "replay_dynamic_rollback_calls"},
            )

    amendment = execution.get("qsa_mtp_amendment")
    if isinstance(amendment, Mapping):
        for event in ("calls", "noops", "amendments", "failures"):
            builder.counter(
                "mlx2_qsa_mtp_amendment_events_total",
                "Shared-QSA MTP amendment lifecycle events.",
                int(amendment.get(event, 0)),
                {"event": event},
            )
        builder.counter(
            "mlx2_qsa_mtp_amendment_blocks_total",
            "Blocks appended by shared-QSA MTP amendment.",
            int(amendment.get("appended_blocks", 0)),
        )
        builder.gauge(
            "mlx2_qsa_mtp_amendment_max_blocks",
            "Largest block count appended by one shared-QSA MTP amendment.",
            int(amendment.get("max_appended_blocks", 0)),
        )

    indexed = execution.get("indexed_qsa")
    if isinstance(indexed, Mapping):
        builder.gauge(
            "mlx2_indexed_qsa_enabled",
            "Whether indexed QSA is enabled.",
            int(bool(indexed.get("enabled", False))),
        )
        builder.counter(
            "mlx2_indexed_qsa_fallbacks_total",
            "Indexed-QSA fallbacks.",
            int(indexed.get("fallbacks", 0)),
        )
        attestation = indexed.get("device_attestation")
        if isinstance(attestation, Mapping):
            for state in ("expected", "observed", "mismatches"):
                builder.counter(
                    "mlx2_indexed_qsa_attestations_total",
                    "Indexed-QSA device attestation outcomes.",
                    int(attestation.get(state, 0)),
                    {"state": state},
                )
            builder.gauge(
                "mlx2_indexed_qsa_attestations_pending",
                "Indexed-QSA device attestations awaiting reconciliation.",
                int(attestation.get("pending", 0)),
            )


_INT8_PREFILL_OUTCOMES = {
    "engaged_calls": "engaged",
    "fallback_rows": "below_row_threshold",
    "fallback_dtype": "unsupported_dtype",
    "fallback_unbound": "unbound_module",
}


def _add_int8_prefill(builder: PrometheusBuilder, engine: Any) -> None:
    """Int8 NAX prefill counters: host ints on the handle, always exported."""

    handle = getattr(engine, "int8_prefill_handle", None)
    policy = getattr(engine, "int8_prefill_policy", None)
    counts = dict(getattr(handle, "counts", None) or {})
    builder.gauge(
        "mlx2_int8_prefill_enabled",
        "Whether W8A8 int8 NAX prefill is installed on the served model.",
        int(bool(getattr(handle, "active", False))),
        {"scope": str(getattr(policy, "scope", "off")) if handle else "off"},
    )
    for key, outcome in _INT8_PREFILL_OUTCOMES.items():
        builder.counter(
            "mlx2_int8_prefill_calls_total",
            "Projection calls seen by int8 prefill, by outcome.",
            int(counts.get(key, 0)),
            {"outcome": outcome},
        )
    builder.counter(
        "mlx2_int8_prefill_rows_total",
        "Activation rows (tokens x lanes) computed by the int8 prefill GEMM.",
        int(counts.get("engaged_rows", 0)),
    )
    builder.gauge(
        "mlx2_int8_prefill_weight_bytes",
        "Bytes held by cached int8 weight copies.",
        int(handle.weight_bytes()) if handle is not None else 0,
    )
    builder.gauge(
        "mlx2_int8_prefill_weight_copy_bytes",
        "Int8 weight-copy bytes (resident plus worst per-call transient).",
        int(handle.weight_copy_bytes()) if handle is not None else 0,
    )


def _add_verify_bitexact(builder: PrometheusBuilder, engine: Any) -> None:
    """Bit-exact verify mode: host counters plus mlx's host-side route counter."""

    handle = getattr(engine, "verify_bitexact_handle", None)
    counts = dict(getattr(handle, "counts", None) or {})
    builder.gauge(
        "mlx2_verify_bitexact_enabled",
        "Whether the batch-invariant quantized matmul mode is active.",
        int(bool(getattr(handle, "active", False))),
    )
    builder.counter(
        "mlx2_verify_bitexact_dispatches_total",
        "Quantized matmul dispatches routed by mlx's bit-exact mode.",
        int(handle.dispatches()) if handle is not None else 0,
    )
    for outcome in ("true", "false"):
        builder.counter(
            "mlx2_verify_bitexact_receipts_total",
            "Terminal receipts by verify_bitexact claim.",
            int(counts.get(f"receipts_{outcome}", 0)),
            {"claim": outcome},
        )


def render_engine_metrics(engine: Any) -> str:
    """Render one non-destructive scrape from host-side engine snapshots."""

    snapshot, counts, queue_depth = _copy_engine_state(engine)
    builder = PrometheusBuilder()
    builder.gauge(
        "mlx2_telemetry_schema_info",
        "mlx2 Prometheus schema identity; the sample value is always one.",
        1,
        {"version": SCHEMA_VERSION},
    )
    builder.gauge(
        "mlx2_build_info",
        "mlx2 runtime build identity; the sample value is always one.",
        1,
        {
            "version": str(snapshot.get("version") or "0.1.0"),
            "revision": str(
                (snapshot.get("runtime") or {}).get("source_sha256") or "unknown"
            ),
        },
    )
    builder.gauge(
        "mlx2_model_info",
        "Loaded model route identity; the sample value is always one.",
        1,
        {
            "model_name": str(snapshot.get("model") or "unavailable"),
            "profile": str(snapshot.get("profile") or "unavailable"),
            "qualification": str(snapshot.get("qualification") or "unavailable"),
        },
    )
    builder.gauge(
        "mlx2_ready",
        "Whether the serving engine has published a ready snapshot.",
        int(snapshot.get("state") == "ready"),
    )
    service_state = str(snapshot.get("service_state") or "serving")
    for state in ("serving", "draining", "quiesced", "suspended"):
        builder.gauge(
            "mlx2_server_state",
            "Current quiesce lifecycle state; exactly one bounded state is one.",
            int(service_state == state),
            {"state": state},
        )
    for key, metric, help_text in (
        ("quiesce_requests", "mlx2_quiesce_requests_total", "Admin quiesce requests."),
        ("drains_completed", "mlx2_drains_completed_total", "Drains completed without timeout."),
        ("drain_timeouts", "mlx2_drain_timeouts_total", "Drains that reached their deadline."),
        ("jobs_drained", "mlx2_jobs_drained_total", "Accepted generation jobs completed during drain."),
        ("suspends", "mlx2_suspends_total", "Completed cache suspension operations."),
        ("suspended_entries", "mlx2_suspended_entries_total", "APCv2 resident entries suspended to disk."),
        ("suspended_bytes", "mlx2_suspended_bytes_total", "APCv2 resident bytes suspended to disk."),
        ("suspend_failures", "mlx2_suspend_failures_total", "APCv2 entries or operations that failed suspension."),
        ("resumes", "mlx2_resumes_total", "Transitions that reopened model admission."),
        ("prefetches_queued", "mlx2_prefetches_queued_total", "Admin resume session prefetches queued."),
        ("prefetches_cancelled", "mlx2_prefetches_cancelled_total", "Session prefetches cancelled while admission was closed."),
    ):
        builder.counter(metric, help_text, int(counts.get(key, 0)))
    for endpoint_class in (
        "generation",
        "embeddings",
        "rerank",
        "audio",
        "batch",
        "session_prefetch",
    ):
        builder.counter(
            "mlx2_admissions_rejected_total",
            "Model-executing admissions rejected while the service is not serving.",
            int(counts.get(f"admissions_rejected_{endpoint_class}", 0)),
            {"endpoint_class": endpoint_class},
        )
    started_at = getattr(engine, "started_at", None)
    if isinstance(started_at, (int, float)) and not isinstance(started_at, bool):
        builder.gauge(
            "mlx2_process_uptime_seconds",
            "Monotonic seconds since the serving engine was constructed.",
            max(0.0, time.monotonic() - float(started_at)),
        )
    builder.gauge(
        "mlx2_telemetry_source_available",
        "Whether the host-side telemetry snapshot was rendered successfully.",
        1,
    )
    _add_batch_metrics(builder, engine, queue_depth)
    _add_http_metrics(builder, engine)
    _add_capabilities(builder, snapshot)
    resources = getattr(engine, "api_resources", None)
    if isinstance(resources, Mapping):
        allowed_counts = {
            "responses": {
                "stores",
                "retrievals",
                "continuations",
                "deletes",
                "evictions",
                "misses",
                "continuation_misses",
                "delete_misses",
                "restored",
                "restore_failures",
            },
            "files": {
                "uploads",
                "retrievals",
                "deletes",
                "evictions",
                "misses",
                "delete_misses",
                "lists",
                "restored",
                "restore_failures",
            },
        }
        for name in ("responses", "files", "batches"):
            resource = resources.get(name)
            status = resource.status() if hasattr(resource, "status") else {}
            if not isinstance(status, Mapping):
                continue
            current = status.get(
                {"responses": "entries", "files": "files", "batches": "batches"}[name],
                0,
            )
            builder.gauge(
                "mlx2_api_resources",
                "Current bounded API resources.",
                int(current),
                {"resource": name},
            )
            builder.gauge(
                "mlx2_api_resource_durable",
                "Whether an API resource is persisted across restart.",
                int(bool(status.get("durable", False))),
                {"resource": name},
            )
            for event in sorted(allowed_counts.get(name, ())):
                builder.counter(
                    "mlx2_api_resource_events_total",
                    "Bounded API resource lifecycle events.",
                    int((status.get("counts") or {}).get(event, 0)),
                    {"resource": name, "event": event},
                )
        batch_status = resources.get("batches")
        batch_snapshot = (
            batch_status.status() if hasattr(batch_status, "status") else {}
        )
        for state in (
            "validating",
            "in_progress",
            "cancelling",
            "completed",
            "failed",
            "cancelled",
        ):
            builder.gauge(
                "mlx2_api_batches",
                "Local Batch API jobs by bounded lifecycle state.",
                int((batch_snapshot.get("states") or {}).get(state, 0)),
                {"state": state},
            )
        for outcome in ("completed", "failed"):
            builder.counter(
                "mlx2_api_batch_requests_total",
                "Local Batch API row outcomes.",
                int(batch_snapshot.get(f"requests_{outcome}", 0)),
                {"outcome": outcome},
            )
        tool_resource = resources.get("mcp_tools")
        tool_snapshot = (
            tool_resource.status() if hasattr(tool_resource, "status") else {}
        )
        for event in ("preparations", "tools_exposed", "executions", "execution_failures"):
            builder.counter(
                "mlx2_mcp_tool_events_total",
                "Allowlisted MCP tool backend lifecycle events.",
                int((tool_snapshot.get("counts") or {}).get(event, 0)),
                {"event": event},
            )

    for key, metric_name in _MEMORY_GAUGES.items():
        value = snapshot.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            builder.gauge(metric_name, f"Current mlx2 {key.replace('_', ' ')}.", value)
    for key, metric_name in _STREAM_GAUGES.items():
        value = counts.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            builder.gauge(metric_name, f"Current mlx2 {key.replace('_', ' ')}.", value)
    for key, (component, event) in sorted(_ENGINE_EVENTS.items()):
        builder.counter(
            "mlx2_runtime_events_total",
            "Advanced runtime events from bounded mlx2 lifecycle enums.",
            int(counts.get(key, 0)),
            {"component": component, "event": event},
        )
    for key, (component, event) in sorted(_OPTIONAL_ENGINE_EVENTS.items()):
        if key in counts:
            builder.counter(
                "mlx2_runtime_events_total",
                "Advanced runtime events from bounded mlx2 lifecycle enums.",
                int(counts[key]),
                {"component": component, "event": event},
            )
    _add_int8_prefill(builder, engine)
    _add_verify_bitexact(builder, engine)
    builder.gauge(
        "mlx2_peak_observed_batch_width",
        "Widest ordinary compute width any completed request has observed.",
        int(counts.get("peak_observed_width", 0)),
    )
    builder.counter(
        "mlx2_telemetry_export_errors_total",
        "Prometheus exposition failures.",
        int(counts.get("telemetry_export_errors", 0)),
    )
    for capability, reason, key in (
        ("batch_cohort", "attachment", "batch_cohort_attachment_failures"),
        ("batch_cohort", "scheduler", "batch_cohort_scheduler_failures"),
        ("structured_output", "constraint", "structured_output_failures"),
        ("tool_calls", "parallel_bound", "tool_call_constraint_failures"),
    ):
        builder.counter(
            "mlx2_fail_closed_total",
            "Requests or cohorts failed closed when a capability invariant was not met.",
            int(counts.get(key, 0)),
            {"capability": capability, "reason": reason},
        )

    apc = snapshot.get("apcv2")
    if isinstance(apc, Mapping):
        live = getattr(getattr(engine, "apc", None), "lifetime_stats", None)
        _add_apcv2(builder, apc, live() if callable(live) else None)
    scheduler = snapshot.get("scheduler")
    if isinstance(scheduler, Mapping):
        _add_scheduler(builder, scheduler)
    segmented = snapshot.get("segmented_self_mtp")
    if isinstance(segmented, Mapping):
        _add_segmented_mtp(builder, segmented)
    capsules = snapshot.get("cache_capsules")
    if isinstance(capsules, Mapping):
        _add_cache_capsules(builder, capsules)
    execution = snapshot.get("execution")
    if isinstance(execution, Mapping):
        _add_execution(builder, execution)
    spomin = snapshot.get("spomin_live_surgery")
    if isinstance(spomin, Mapping):
        for event, value in sorted((spomin.get("counts") or {}).items()):
            if (
                event in _SPOMIN_OPERATIONS
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                builder.counter(
                    "mlx2_spomin_operations_total",
                    "Live Spomin state-surgery operations by bounded outcome.",
                    value,
                    {"operation": event},
                )
    manager = getattr(engine, "multi_lora", None)
    multi_lora = manager.status() if manager is not None else None
    if isinstance(multi_lora, Mapping) and multi_lora.get("enabled"):
        for event, value in sorted((multi_lora.get("counts") or {}).items()):
            if (
                event in _MULTI_LORA_EVENTS
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                builder.counter(
                    "mlx2_multi_lora_events_total",
                    "Concurrent multi-LoRA mechanism events (forwards, rows, slots).",
                    value,
                    {"event": event},
                )
        builder.gauge(
            "mlx2_multi_lora_resident_adapters",
            "LoRA adapters resident in device slots.",
            len(multi_lora.get("resident") or {}),
        )
        builder.gauge(
            "mlx2_multi_lora_reserved_bytes",
            "Bytes preallocated for multi-LoRA slot tensors.",
            int(multi_lora.get("reserved_bytes") or 0),
        )
    return builder.render()
