from collections import Counter
import re
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from mlx2.batch_metrics import BatchRuntimeMetrics, HttpRuntimeMetrics
from mlx2.api_resources import BatchManager, FileStore, ResponseStore
from mlx2.prometheus import (
    CONTENT_TYPE,
    CumulativeHistogram,
    PrometheusBuilder,
    render_engine_metrics,
)
from mlx2.server import handler_for


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 0.01
        return self.value


def exercised_batch_metrics():
    metrics = BatchRuntimeMetrics(clock=Clock())
    metrics.admitted("secret-request", "secret-tenant", 0)
    metrics.dequeued("secret-request", 0)
    metrics.prompt("secret-request", 12, 8)
    metrics.lane_attached("secret-request", 1, "self_mtp")
    metrics.batch_cycle(1)
    metrics.token("secret-request")
    metrics.token("secret-request")
    metrics.finishing("secret-request", "stop")
    metrics.terminal("secret-request", "completed")
    metrics.rejected("maximum_inflight", 1)
    return metrics


class FakeEngine:
    model_path = "fixture"

    def __init__(self):
        self.lock = threading.Lock()
        self.queued_jobs = 0
        self.counts = Counter(
            {
                "apcv2_fanout_groups": 2,
                "apc_interior_checkpoints_captured": 5,
                "apc_interior_checkpoints_degraded": 3,
                "approximate_kv_applied": 1,
                "cache_capsule_prepared": 3,
                "memory_admission_deferred": 4,
                "spomin_exact_boundary_stores": 1,
                "structured_output_dead_ends": 1,
                "tool_call_constraint_failures": 1,
            }
        )
        self.batch_metrics = exercised_batch_metrics()
        self.snapshot = {
            "state": "ready",
            "model": "fixture",
            "profile": "qualified-profile",
            "qualification": "qualified",
            "runtime": {"source_sha256": "abc123"},
            "capabilities": ["mtp", "segmented_mtp"],
            "implemented_capabilities": ["mtp", "segmented_mtp", "ordinary"],
            "selected_capabilities": ["mtp", "segmented_mtp"],
            "qualified_capabilities": ["mtp"],
            "metal_active_bytes": 100,
            "metal_peak_bytes": 200,
            "process_physical_footprint_bytes": 300,
            "headroom_bytes": 400,
            "memory_waiting": 0,
            "scheduler": {
                "prefill_rounds": 7,
                "pld_proposed": 8,
                "accepted_proposals": 9,
                "fly_relaxed_accepts": 2,
                "self_mtp_zero_fast_rounds": 3,
                "apc_interior_checkpoints_captured": 5,
                "target_max_width": 2,
                "adaptive_prefill_chunk_histogram": {128: 3, 256: 1},
                "private-runtime-key": 999,
            },
            "cache_capsules": {
                "requests": 4,
                "primary_successes": 3,
                "fallbacks": 1,
            },
            "segmented_self_mtp": {
                "requests": 2,
                "engaged": 1,
                "declined": 1,
                "materialized_bytes": 64,
                "proposal_ns": 1_000_000_000,
                "device_synchronizations": 0,
                "private_delta_decline_reasons": {"not-a-label": 1},
                "private-segmented-key": 999,
            },
            "apcv2": {
                "lifetime": {
                    "lookups": 4,
                    "hits": 3,
                    "misses": 1,
                    "cached_tokens": 128,
                    "stores": 2,
                    "interior_hits": 1,
                },
                "reuse_telemetry": {
                    "hit_age_seconds": {
                        "buckets": (1, 5, 30),
                        "bucket_counts": (0, 1, 2),
                        "count": 3,
                        "sum": 42.0,
                    },
                    "eviction_age_seconds": {
                        "buckets": (1, 5, 30),
                        "bucket_counts": (0, 1, 1),
                        "count": 2,
                        "sum": 61.0,
                    },
                    "eviction_idle_seconds": {
                        "buckets": (1, 5, 30),
                        "bucket_counts": (1, 1, 2),
                        "count": 2,
                        "sum": 31.0,
                    },
                    "eviction_hit_count": {
                        "buckets": (0, 1, 2),
                        "bucket_counts": (1, 1, 2),
                        "count": 2,
                        "sum": 2.0,
                    },
                },
                "layer_segments": {
                    "entries": 2,
                    "logical_bytes": 4096,
                },
                "idle_disk": {
                    "resident_bytes": 4096,
                    "disk_bytes": 1024,
                    "disk_entries": 1,
                    "restores": 1,
                    "bytes_read": 256,
                    "parks": 2,
                    "prefetch_restores_ok": 1,
                    "prefetch_hits": 1,
                    "persisted_writes": 3,
                    "restore_digest_failures": 1,
                },
                "persistence": {
                    "rescan": {
                        "registered": 2,
                        "registered_bytes": 1024,
                        "elapsed_seconds": 0.01,
                        "discarded": {"identity_mismatch": 1},
                    }
                },
                "cache_capsules": {
                    "reservations": 2,
                    "reservation_rejections": 1,
                    "reserved_bytes": 512,
                },
                "cow": {
                    "branches": 2,
                    "active_leases": 1,
                    "descriptor_bytes": 2048,
                    "branch_ns": 1_000_000,
                    "planes": {"attention_kv": {"descriptor_aliases": 2}},
                },
            },
            "spomin_live_surgery": {
                "enabled": True,
                "counts": {"applied": 1, "declined": 2, "secret-outcome": 99},
            },
            "execution": {
                "media_feature_cache": {
                    "entries": 2,
                    "bytes": 4096,
                    "hits": 3,
                    "misses": 1,
                    "stores": 2,
                    "evictions": 0,
                    "secret": 99,
                },
                "moe": {
                    "fused_gate_up_layers": 8,
                    "dispatches": {"scalar": 2, "tile4": 3, "secret": 99},
                    "fallbacks": 1,
                    "router_calls": 5,
                },
                "round_levers": {
                    "ple_dq_hits": 4,
                    "ple_prefetch_rows": 32,
                    "secret-lever": 99,
                },
                "ple_compile": {
                    "enabled": True,
                    "cache_max": 32,
                    "counts": {"builds": 1, "hits": 2, "secret": 99},
                },
                "fused_gdn": {"fused_calls": 3, "replay_rollback_tokens": 7},
                "qsa_mtp_amendment": {
                    "calls": 2,
                    "amendments": 1,
                    "appended_blocks": 3,
                    "max_appended_blocks": 2,
                },
                "indexed_qsa": {
                    "enabled": True,
                    "fallbacks": 1,
                    "device_attestation": {
                        "expected": 4,
                        "observed": 4,
                        "mismatches": 0,
                        "pending": 0,
                    },
                },
            },
        }

    def status(self):
        return {**self.snapshot, "healthy": True, "error": None}

    def batching_status(self):
        return self.batch_metrics.snapshot(queue_depth=self.queued_jobs)


_SAMPLE_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*(?:\{.*\})? [-+0-9.eENaInf]+$")


def assert_valid_prometheus_text(text):
    assert text.endswith("\n")
    helps = set()
    types = set()
    samples = 0
    for line in text.splitlines():
        if line.startswith("# HELP "):
            helps.add(line.split()[2])
        elif line.startswith("# TYPE "):
            fields = line.split()
            assert fields[3] in {"counter", "gauge", "histogram"}
            types.add(fields[2])
        else:
            assert _SAMPLE_RE.fullmatch(line), line
            samples += 1
    assert helps == types
    assert samples > 0


def test_cumulative_histogram_is_monotonic_and_immutable():
    histogram = CumulativeHistogram((0.1, 1.0))
    histogram.observe(0.05)
    first = histogram.snapshot()
    histogram.observe(0.5)
    second = histogram.snapshot()
    assert first.bucket_counts == (1, 1)
    assert second.bucket_counts == (1, 2)
    assert second.count == 2
    assert first.count == 1


def test_builder_escapes_labels_and_emits_histogram_metadata():
    histogram = CumulativeHistogram((0.1, 1.0))
    histogram.observe(0.2)
    builder = PrometheusBuilder()
    builder.counter("mlx2_test_total", "test counter", 1, {"outcome": 'a"b'})
    builder.histogram("mlx2_test_seconds", "test histogram", histogram.snapshot())
    rendered = builder.render()
    assert 'outcome="a\\"b"' in rendered
    assert "# TYPE mlx2_test_seconds histogram" in rendered
    assert 'mlx2_test_seconds_bucket{le="+Inf"} 1' in rendered
    assert_valid_prometheus_text(rendered)


def test_engine_exposition_is_non_destructive_bounded_and_advanced():
    engine = FakeEngine()
    before = engine.batch_metrics.snapshot()
    first = render_engine_metrics(engine)
    second = render_engine_metrics(engine)
    after = engine.batch_metrics.snapshot()
    assert first == second
    assert before == after
    assert_valid_prometheus_text(first)
    for secret in (
        "secret-request", "secret-tenant", "not-a-label", "private-runtime-key",
        "private-segmented-key", "secret-outcome", "secret-lever",
    ):
        assert secret not in first
    assert 'mlx2_requests_total{finished_reason="stop",outcome="completed"} 1' in first
    assert "mlx2_time_to_first_token_seconds_bucket" in first
    assert "mlx2_prefix_cache_hits_total 3" in first
    assert 'mlx2_prefix_cache_session_events_total{operation="parks"} 2' in first
    assert "mlx2_prefix_cache_rescan_registered_total 2" in first
    assert 'mlx2_prefix_cache_rescan_discarded_total{reason="identity_mismatch"} 1' in first
    assert 'component="approximate_kv",event="applied"' in first
    assert 'component="apcv2_interior",event="degraded"} 3' in first
    assert 'component="thinking_budget",event="forced_close"' in first
    assert 'component="tool_calls",event="parse_fallback"' in first
    assert 'component="tool_calls",event="parallel_bound_failure"' in first
    assert 'capability="tool_calls",reason="parallel_bound"' in first
    assert 'component="json_schema",event="reference_failure"' in first
    assert 'component="structured_output",event="dead_end"} 1' in first
    assert 'mechanism="prompt_lookup"' in first
    assert 'event="fly_relaxed_accepts",mechanism="fly_verification"' in first
    assert 'event="self_mtp_zero_fast_rounds",mechanism="self_mtp"' in first
    assert 'mlx2_segmented_mtp_events_total{event="engaged"} 1' in first
    assert 'mlx2_spomin_operations_total{operation="applied"} 1' in first
    assert 'component="spomin",event="exact_boundary_stored"' in first
    assert 'mlx2_cache_capsule_operations_total{outcome="primary_successes"} 3' in first
    assert 'mlx2_capability_state{capability="ordinary",state="implemented"} 1' in first
    assert 'mlx2_capability_state{capability="ordinary",state="selected"} 0' in first
    assert 'mlx2_capability_state{capability="segmented_mtp",state="qualified"} 0' in first
    assert 'mlx2_moe_dispatches_total{mode="tile4"} 3' in first
    assert 'mlx2_advanced_path_events_total{event="ple_dq_hits"} 4' in first
    assert 'mlx2_media_feature_cache_events_total{event="hits"} 3' in first
    assert "mlx2_media_feature_cache_entries 2" in first
    assert "mlx2_media_feature_cache_bytes 4096" in first
    assert "mlx2_fused_gdn_rollback_tokens_total 7" in first
    assert "mlx2_telemetry_source_available 1" in first


def test_quiesce_state_and_counters_are_exported_once():
    engine = FakeEngine()
    engine._service_state = "suspended"
    engine.counts.update(
        {
            "quiesce_requests": 2,
            "drains_completed": 1,
            "drain_timeouts": 1,
            "jobs_drained": 3,
            "admissions_rejected_generation": 4,
            "suspends": 1,
            "suspended_entries": 5,
            "suspended_bytes": 4096,
            "suspend_failures": 1,
            "resumes": 2,
            "prefetches_queued": 3,
            "prefetches_cancelled": 2,
        }
    )
    rendered = render_engine_metrics(engine)
    assert 'mlx2_server_state{state="suspended"} 1' in rendered
    assert 'mlx2_server_state{state="serving"} 0' in rendered
    assert "mlx2_quiesce_requests_total 2" in rendered
    assert "mlx2_drains_completed_total 1" in rendered
    assert "mlx2_drain_timeouts_total 1" in rendered
    assert "mlx2_jobs_drained_total 3" in rendered
    assert "mlx2_suspends_total 1" in rendered
    assert "mlx2_suspended_entries_total 5" in rendered
    assert "mlx2_suspended_bytes_total 4096" in rendered
    assert "mlx2_suspend_failures_total 1" in rendered
    assert "mlx2_resumes_total 2" in rendered
    assert "mlx2_prefetches_queued_total 3" in rendered
    assert "mlx2_prefetches_cancelled_total 2" in rendered
    assert (
        'mlx2_admissions_rejected_total{endpoint_class="generation"} 4'
        in rendered
    )
    sample_lines = [
        line
        for line in rendered.splitlines()
        if line and not line.startswith("#")
    ]
    assert len(sample_lines) == len(set(line.rsplit(" ", 1)[0] for line in sample_lines))
    assert_valid_prometheus_text(rendered)


def test_api_resource_metrics_are_bounded_and_non_destructive():
    engine = FakeEngine()
    responses = ResponseStore()
    responses.put(
        "tenant",
        {"id": "resp_test", "object": "response"},
        [{"role": "user", "content": "x"}],
    )
    responses.get("tenant", "resp_test")
    files = FileStore()
    files.create(
        "tenant",
        filename="x.txt",
        purpose="user_data",
        content_type="text/plain",
        content=b"x",
    )
    batches = BatchManager(files, lambda *_args: (200, {}))
    engine.api_resources = {
        "responses": responses,
        "files": files,
        "batches": batches,
    }
    first = render_engine_metrics(engine)
    second = render_engine_metrics(engine)
    assert first == second
    assert 'mlx2_api_resources{resource="responses"} 1' in first
    assert (
        'mlx2_api_resource_events_total{event="stores",resource="responses"} 1'
        in first
    )
    assert (
        'mlx2_api_resource_events_total{event="retrievals",resource="responses"} 1'
        in first
    )
    assert 'mlx2_api_resources{resource="files"} 1' in first
    assert 'mlx2_api_batches{state="completed"} 0' in first
    assert_valid_prometheus_text(first)


def test_apcv2_interior_event_names_are_exported_on_only_one_surface():
    rendered = render_engine_metrics(FakeEngine())
    runtime = set()
    scheduler = set()
    for line in rendered.splitlines():
        if line.startswith("mlx2_runtime_events_total{") and (
            'component="apcv2_interior"' in line
        ):
            event = re.search(r'event="([^"]+)"', line).group(1)
            runtime.add("apc_interior_checkpoints_" + event)
        elif line.startswith("mlx2_scheduler_events_total{"):
            event = re.search(r'event="([^"]+)"', line).group(1)
            if event.startswith("apc_interior_checkpoints_"):
                scheduler.add(event)

    assert "apc_interior_checkpoints_captured" in runtime
    assert runtime.isdisjoint(scheduler), {
        "duplicated_event_names": sorted(runtime & scheduler),
        "runtime": sorted(runtime),
        "scheduler": sorted(scheduler),
    }


def test_concurrent_scrapes_are_stable():
    engine = FakeEngine()
    outputs = []

    def scrape():
        for _ in range(20):
            outputs.append(render_engine_metrics(engine))

    threads = [threading.Thread(target=scrape) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(outputs) == 160
    assert len(set(outputs)) == 1


def test_exposition_parses_with_official_prometheus_client_when_installed():
    parser = pytest.importorskip("prometheus_client.parser")
    engine = FakeEngine()
    engine.http_metrics = HttpRuntimeMetrics()
    families = list(parser.text_string_to_metric_families(render_engine_metrics(engine)))
    assert families
    assert any(family.name == "mlx2_requests" for family in families)


def test_free_form_admission_and_mechanism_values_collapse_to_other():
    metrics = BatchRuntimeMetrics(clock=Clock())
    metrics.rejected("secret-reason", 0)
    metrics.admitted("request", "tenant", 0)
    metrics.lane_attached("request", 1, "secret-mechanism")
    data = metrics.prometheus_snapshot()
    labels = {labels: value for (_name, labels), value in data["counters"].items()}
    rendered_labels = repr(labels)
    assert "secret-reason" not in rendered_labels
    assert "secret-mechanism" not in rendered_labels
    assert "other" in rendered_labels


def test_http_metrics_endpoint_uses_prometheus_content_type():
    engine = FakeEngine()
    engine.http_metrics = HttpRuntimeMetrics()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/metrics") as response:
            payload = response.read().decode()
            assert response.headers["Content-Type"] == CONTENT_TYPE
            assert response.status == 200
        assert_valid_prometheus_text(payload)
        with urlopen(f"http://127.0.0.1:{server.server_port}/metrics") as response:
            second_payload = response.read().decode()
        assert (
            'mlx2_http_requests_total{method="GET",route="metrics",status_class="2xx"} 1'
            in second_payload
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_http_metrics_endpoint_returns_503_when_export_fails():
    class BrokenEngine(FakeEngine):
        def prometheus_metrics(self):
            raise RuntimeError("fixture exporter failure")

    engine = BrokenEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        try:
            urlopen(f"http://127.0.0.1:{server.server_port}/metrics")
        except HTTPError as error:
            payload = error.read().decode()
            assert error.code == 503
            assert "mlx2_telemetry_source_available 0" in payload
            assert "mlx2_telemetry_export_errors_total 1" in payload
        else:
            raise AssertionError("failed scrape returned success")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
