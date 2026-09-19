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
  tokens, indexed-QSA fallbacks and device attestations, and shared-QSA MTP
  amendment calls/blocks. Private diagnostic dictionary keys are not labels.
- Sanity counters (`mlx2_runtime_events_total`): `component="request_finish"`
  with `event` in `stop`, `length`, `unknown`, `cancelled`,
  `client_disconnect`; `component="structured_output"` adds
  `engine_automaton`, `engine_scanner`, `deferred_past_thinking`, and
  `dead_end` (a grammar step with no valid token continuation).
  `mlx2_peak_observed_batch_width` is the widest ordinary compute width a
  completed request has reported. Prompt-lookup rotating-replay counters
  (`pld_rotating_replay_*`) arrive through the scheduler events.
  Tool-call events separately count default-off decode-grammar engagements,
  grammar skips that retain main's post-generation validation, tolerant-marker
  requests, and actual parse fallbacks.
- APCv2: lifetime lookups, hits, misses, stores and cached tokens; resident and
  disk bytes/entries; spill/restore events and bytes; COW leases,
  materializations, byte accounting and optional host timing. Session
  lifecycle, startup rescan, write suppression, exact interior
  capture/publish/degrade, reuse and reclaim-before-evict use bounded events.
- Cache capsules: capacity reservations and rejections plus request, success,
  fallback, timeout, stale, error and late-disposal outcomes.
- Scheduling/speculation: ordinary, adaptive-prefill, prompt-lookup, external
  draft, adaptive-MTP and cache-capsule events through bounded `mechanism` and
  `event` labels. Configured-width/high-water values are gauges.
  Processor-probe masking, permanent-ordinary fast paths, self-MTP depth-zero
  rounds and FLy relaxed accepts retain always-on bounded host counters while
  their output-changing policies remain default-off.
- Segmented self-MTP: engagement/lifecycle events, materialized bytes, optional
  timing, transaction outcomes, forwards, attention calls and explicit device
  synchronization counts. Free-form decline details are intentionally omitted.
- Approximate KV, atomic cohorts, structured output, memory admission and
  Spomin live surgery: bounded operation/outcome counters. Spomin additionally
  reports exact pre-surgery boundary stores plus snapshot/store failures, so a
  repeated long prompt cannot silently fall back to full re-prefill. Full
  revision-bound receipts remain JSON-only.
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
