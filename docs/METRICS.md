# Metrics and telemetry

mlx2 exposes aggregate Prometheus text format 0.0.4 at `GET /metrics`. The
endpoint is a non-destructive view over host-side counters and immutable
snapshots: a scrape does not clear state, synchronize a device, inspect an
array, or load a model. Detailed request history, route receipts and evidence
remain on `/v1/status` and `/v1/status/batching`.

The contract follows Prometheus conventions rather than another runtime's
namespace. Metrics use the `mlx2_` prefix, seconds and bytes as base units,
monotonic `_total` counters, cumulative histograms, and bounded labels. The
schema identity is `mlx2_telemetry_schema_info{version="2"}`. Version 2 adds
explicit capability lifecycle sets, bounded advanced-path schemas, transport
metrics, and truthful scrape failure signalling.

## Core request contract

| Family | Type | Meaning |
|---|---|---|
| `mlx2_requests_total{outcome,finished_reason}` | counter | Terminal requests; both labels use closed enums |
| `mlx2_admission_decisions_total{decision,reason}` | counter | Accepted and rejected submissions |
| `mlx2_prompt_tokens_total` | counter | Prompt tokens recorded after APCv2 admission |
| `mlx2_generation_tokens_total` | counter | Output tokens delivered by the engine |
| `mlx2_request_cached_prompt_tokens_total` | counter | Cached prompt tokens attached to admitted requests |
| `mlx2_num_requests_running`, `mlx2_num_requests_waiting` | gauge | Current ownership and queue state |
| `mlx2_active_lanes`, `mlx2_peak_active_lanes` | gauge | Current and process high-water execution width |
| `mlx2_time_to_first_token_seconds` | histogram | Admission through first output token |
| `mlx2_inter_token_latency_seconds` | histogram | Consecutive delivered-token interval |
| `mlx2_request_time_per_output_token_seconds` | histogram | Per-request service time divided by delivered tokens |
| `mlx2_e2e_request_latency_seconds` | histogram | Admission through terminal completion |
| `mlx2_request_queue_time_seconds` | histogram | Admission through scheduler dequeue |
| `mlx2_request_prefill_time_seconds` | histogram | Lane attachment through first output token |
| `mlx2_request_decode_time_seconds` | histogram | First output token through terminal completion |
| `mlx2_request_inference_time_seconds` | histogram | Scheduler dequeue through terminal completion |
| `mlx2_request_prompt_tokens`, `mlx2_request_generation_tokens` | histogram | Request-size distributions |
| `mlx2_batch_size` | histogram | Active lanes per scheduler cycle |

`rate(mlx2_generation_tokens_total[1m])` is the canonical aggregate generation
rate. mlx2 does not publish a precomputed throughput claim.

## Quiesce lifecycle

| Family | Type | Meaning |
|---|---|---|
| `mlx2_server_state{state}` | gauge | One-hot `serving`, `draining`, `quiesced`, or `suspended` process state |
| `mlx2_quiesce_requests_total` | counter | Valid quiesce operations received, including idempotent requests |
| `mlx2_drains_completed_total` | counter | Drains that reached the worker idle boundary |
| `mlx2_drain_timeouts_total` | counter | Drains that terminated remaining accepted work at their deadline |
| `mlx2_jobs_drained_total` | counter | Accepted generation jobs finalized while draining, excluding timeout termination |
| `mlx2_admissions_rejected_total{endpoint_class}` | counter | Closed-admission rejections for the bounded generation, embeddings, rerank, audio, batch, and session-prefetch classes |
| `mlx2_suspends_total` | counter | Cache-only suspends completed |
| `mlx2_suspended_entries_total`, `mlx2_suspended_bytes_total` | counter | Exact resident APCv2 entries and bytes spilled by suspend |
| `mlx2_suspend_failures_total` | counter | Individual entry or whole suspend-operation failures |
| `mlx2_resumes_total` | counter | Actual non-serving to serving transitions; idempotent serving requests do not increment it |
| `mlx2_prefetches_queued_total` | counter | Admin-requested session prefetches accepted after resume |
| `mlx2_prefetches_cancelled_total` | counter | Queued admin or APCv2 session prefetches cancelled when admission closes |

The JSON status endpoint also reports the current state and since-timestamps,
last transition result, all lifecycle counters, and the per-class rejection
map. These metrics are host counters and do not synchronize MLX device state.

## Advanced mechanisms

- Route state: `mlx2_route_info`, `mlx2_capability_state`,
  `mlx2_capability_engagements_total`, and `mlx2_fail_closed_total` keep
  declared/selected/qualified state separate from observed engagement.
- Flash-Next execution: bounded MoE dispatch/router/fallback counters, PLE
  prefetch and compilation lifecycle, fused-GDN calls/fallbacks/rollback
  tokens (plus `event="replay_dynamic_rollback_calls"`, only when device-count
  rollback is selected), indexed-QSA fallbacks and device attestations, and shared-QSA MTP
  amendment calls/blocks. Private diagnostic dictionary keys are not labels.
- Sanity counters (`mlx2_runtime_events_total`): `component="request_finish"`
  with `event` in `stop`, `length`, `unknown`, `cancelled`,
  `client_disconnect`; `component="structured_output"` adds
  `engine_automaton`, `engine_scanner`, `deferred_past_thinking`, and
  `dead_end` (a grammar step with no valid token continuation).
  `mlx2_peak_observed_batch_width` is the widest ordinary compute width a
  completed request has reported. Prompt-lookup rotating-replay counters
  (`pld_rotating_replay_*`) arrive through the scheduler events, as do the
  committed-boundary snapshot counters `external_cow_snapshots` /
  `external_cow_fallbacks` (external draft route) and `pld_cow_snapshots` /
  `pld_cow_fallbacks` (prompt lookup): descriptor-COW captures versus deep-copy
  fallbacks for graphs that cannot be frozen.
  Tool-call events separately count default-off decode-grammar engagements,
  grammar skips that retain main's post-generation validation, tolerant-marker
  requests, and actual parse fallbacks. With `constrained_tool_grammar_auto`,
  `decode_grammar_auto_engaged` counts engagements of the `auto` text-or-calls
  and calls-or-answer shapes (also counted in `decode_grammar_engaged`); with
  `tool_grammar_streaming`, `decode_grammar_streamed` counts HTTP streams whose
  tool calls were sent as they completed instead of buffered.
- APCv2: lifetime lookups, hits, misses, stores and cached tokens; resident and
  disk bytes/entries; spill/restore events and bytes; COW leases,
  materializations, byte accounting and optional host timing. Session
  lifecycle, startup rescan, write suppression, exact interior
  capture/publish/degrade, reuse and reclaim-before-evict use bounded events.
  Interior placement adds `apcv2_interior` events: `planned_turn`,
  `planned_tail`, `planned_lattice`, `skipped_media`, `headroom_capped`,
  `admitted_hit`, `turn_boundary_hit`, `hit_token` and `turn_marker_missing`.
  `/v1/status.apcv2.interior` reports resident interior entries, bytes and
  reused entries.
  Rolling prefill checkpoints report `apcv2_rolling` runtime events
  (`planned`, `degraded`, `published`, `cancel_published`, `retired`,
  `retire_deferred`, `retire_shared`, `skipped_*`), scheduler events
  `apc_rolling_checkpoints_{captured,skipped_inexact,skipped_pressure}`, and
  `mlx2_prefix_cache_rolling_hits_total`. Junction checkpoints use the same
  shape (`apcv2_junction`, `apc_junction_checkpoints_*`,
  `mlx2_prefix_cache_junction_hits_total`).
  The four `mlx2_prefix_cache_{hits,interior_hits,rolling_hits,junction_hits}`
  series are read live from `APCv2.lifetime_stats()`, not from the `apcv2`
  status snapshot. That snapshot aggregates per-entry segment statistics, so
  serving refreshes it at most once a second and seeds it all-zero at
  readiness; exporting hit counts from it reported 0 for any run shorter than
  the cadence, which is how a working junction hit was read as
  `junction_hits: 0` on a GPU gate. `/v1/status.apcv2` still carries the
  periodic snapshot and still lags by up to a second -- prefer `/metrics` or
  the engine's own `counts` when gating on a hit.
- Per-round verification histograms (`verify_span_hist`,
  `verify_accept_hist`) are **receipt** fields on the self-MTP, external-draft
  and prompt-lookup routes, not Prometheus series: their key space is the
  route's draft depth and they are per-request, so exporting them would add
  request-scoped cardinality for something a receipt consumer already gets
  exactly. The scheduler-level aggregates they decompose
  (`pld_accepted`, `accepted_proposals`, the self-MTP cycle counters) are
  exported as before and are unchanged. See `docs/SERVING.md`, "Per-round
  verification histograms in the receipt".
- Cache capsules: capacity reservations and rejections plus request, success,
  fallback, timeout, stale, error and late-disposal outcomes.
- Scheduling/speculation: ordinary, adaptive-prefill, prompt-lookup, external
  draft, adaptive-MTP and cache-capsule events through bounded `mechanism` and
  `event` labels. Configured-width/high-water values are gauges.
  Default-off SRPT prefill scheduling reports `mechanism="prefill_scheduling"`
  with `event` in `prefill_scheduling_bypasses` (older requests overtaken),
  `prefill_scheduling_bypass_forced` (bypass-cap services) and
  `prefill_scheduling_one_slice_clamps` (long slices bounded because a peer
  fit one slice); the keys exist only when the policy is selected.
  Processor-probe masking, permanent-ordinary fast paths, self-MTP depth-zero
  rounds and FLy relaxed accepts retain always-on bounded host counters while
  their output-changing policies remain default-off. Batched DFlash2 pairwise
  selection (`pairwise_selection: "batched"`) adds
  `external_pairwise_selection_groups`/`_lanes` events under
  `mechanism="external_speculative"`; the keys exist only when it is enabled.
- Segmented self-MTP: engagement/lifecycle events, materialized bytes, optional
  timing, transaction outcomes, forwards, attention calls and explicit device
  synchronization counts. Free-form decline details are intentionally omitted.
- Approximate KV, atomic cohorts, structured output, memory admission and
  Spomin live surgery: bounded operation/outcome counters. Spomin additionally
  reports exact pre-surgery boundary stores plus snapshot/store failures, so a
  repeated long prompt cannot silently fall back to full re-prefill. Full
  revision-bound receipts remain JSON-only.
- Host memory signals (only with `execution_policy.host_memory_signals`
  enabled; otherwise absent): `mlx2_host_memory_pressure_level` is the kernel
  memorystatus level after fall hysteresis (0 normal, 1 warn, 2 critical) and
  `mlx2_host_memory_available_bytes` is the Mach-statistics host estimate that
  then also feeds `mlx2_memory_headroom_bytes`.
- MoE expert disk streaming (only with `execution_policy.moe_expert_streaming`
  enabled; otherwise absent). `mlx2_runtime_events_total{component="expert_stream"}`
  counts `page_in`, `page_in_byte`, `hit`, `miss`, `eviction` and
  `working_set_refusal`; `mlx2_expert_stream_resident_bytes` is the bytes the
  bounded per-layer LRU currently holds, which never exceeds the configured
  `cache_gib` ceiling. `component="expert_atlas"` counts `observation`,
  `trace_record` and `trace_dropped`: the atlas **collects only** and never
  influences residency, so there is deliberately no pinned-bytes or
  pinned-hit-rate metric to read. Page-in counts are the diagnostic for this
  feature; throughput measured on a streamed configuration is not a benchmark
  and must not be recorded as one.
- Memory preemption (default-off): `component="memory_preemption"` with
  `event` in `preempted`, `preempted_stall`, `preempted_pressure`,
  `preempted_fault` (qualification-mode injection), `replayed`
  and `drain_cancelled` (`memory_preemptions*`, `preempted_replays`,
  `memory_preemption_drain_cancellations` in `/v1/status` counts, present
  there only when the policy is enabled).
- Process-local API resources: `mlx2_api_resources` reports bounded Responses,
  Files and Batch objects; `mlx2_api_resource_events_total` reports bounded
  store/retrieve/continue/delete/evict outcomes; `mlx2_api_batches` and
  `mlx2_api_batch_requests_total` expose batch lifecycle and row outcomes.
  `/v1/status.api_resources` carries byte/capacity details. File names, paths,
  response IDs, batch IDs and tenant IDs never become Prometheus labels.
- API compatibility counters include rejected reasoning signatures,
  history-budget closes, tool-grammar engagements/skips and schema-reference
  failures. APC interior names have one exporter owner; the
  disjointness regression test prevents duplicate series.

## Cardinality and privacy

Prometheus labels never include request or tenant IDs, prompts, generated
text, cache keys, filesystem paths, artifact fingerprints, or free-form
exception/detail strings. Model, profile, capability, mechanism, operation,
outcome and reason labels come from runtime-owned bounded values. Tenant-level
rates remain in the diagnostic JSON endpoint; Prometheus exports only the
aggregate Jain fairness gauge.

Unavailable measurements are omitted rather than emitted as false zeroes.
Families that are part of an active subsystem remain present at zero so alert
queries do not have to infer whether a selected mechanism was merely idle.

`mlx2_ready`, `mlx2_process_uptime_seconds`, and
`mlx2_telemetry_source_available` cover the exporter itself. A rendering
failure returns HTTP 503 with `mlx2_telemetry_source_available 0`; it does not
return a partial 200 response. HTTP request counts and duration histograms use
closed route and status-class labels.

## Compatibility and evolution

The core concepts and units align with vLLM, SGLang and TensorRT-LLM, while
mlx2 retains its own namespace. The implementation is original mlx2 code; no
peer exporter source was copied. Published family meanings, label names and
histogram boundaries are compatibility contracts. A breaking change requires
a schema-version increment and one release of overlap or an explicit migration
note.

Optional OTLP/HTTP request tracing is enabled with
`--otlp-traces-endpoint URL` after installing `mlx2[observability]`. It imports
OpenTelemetry only when selected, extracts W3C trace context, exports through
the SDK batch processor, and excludes prompt/output content. Span lifecycle
counters are exposed through `mlx2_trace_spans_total`; setup and asynchronous
export failures are counted by `mlx2_trace_export_errors_total`.

The deployable examples in `observability/` include a Grafana dashboard and
Prometheus alert rules. Thresholds are conservative examples and should be
tuned to the service's own SLOs; they are not performance claims.

## Qualification boundary

CPU qualification covers text-format structure, escaping, cumulative
histograms, monotonic counters, concurrent scrapes, scrape non-mutation,
HTTP content type, advanced-mechanism mapping and absence of sensitive labels.
It does not qualify live Metal instrumentation, model execution, metric values
under a real workload, a live OTLP collector, or alert thresholds.
