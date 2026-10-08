import pytest

from mlx2.batch_metrics import BatchFaultSpec, BatchRuntimeMetrics, HttpRuntimeMetrics
from mlx2.tracing import OptionalRequestTracer


def test_bounded_metrics_report_latency_fairness_and_mechanism():
    ticks = iter([0.0, 0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 1.0, 1.1, 1.2])
    metrics = BatchRuntimeMetrics(history_size=4, clock=lambda: next(ticks))
    metrics.admitted("a", "tenant-a", 0)
    metrics.dequeued("a", 0)
    metrics.lane_attached("a", 1, "ordinary")
    metrics.token("a")
    metrics.token("a")
    metrics.terminal("a", "completed")
    snapshot = metrics.snapshot(queue_depth=0, memory={"headroom_bytes": 10})
    assert snapshot["schema"] == "mlx2.batch-runtime.v1"
    assert snapshot["latency_ms"]["ttft"]["count"] == 1
    assert snapshot["latency_ms"]["itl"]["count"] == 1
    assert snapshot["fairness"]["jain_tenant_token_rate"] == 1.0
    assert snapshot["mechanism_receipts"] == {"ordinary": 1}
    assert snapshot["memory"] == {"headroom_bytes": 10}


def test_replayed_attachment_counts_one_engagement():
    metrics = BatchRuntimeMetrics(clock=lambda: 1.0)
    metrics.admitted("r1", "t", 0)
    metrics.dequeued("r1", 0)
    metrics.lane_attached("r1", 1, "ordinary")
    # Memory preemption evicts the lane and the request replays on a new one.
    metrics.lane_attached("r1", 1, "ordinary")
    metrics.token("r1")
    metrics.terminal("r1", "completed", "stop")
    counters = metrics.prometheus_snapshot()["counters"]
    engagements = {
        dict(labels)["capability"]: value
        for (name, labels), value in counters.items()
        if name == "mlx2_capability_engagements_total"
    }
    assert engagements == {"ordinary": 1}
    assert metrics.snapshot()["mechanism_receipts"] == {"ordinary": 1}


def test_fault_spec_is_qualification_only_and_bounded():
    assert BatchFaultSpec.parse(None, enabled=False) is None
    spec = BatchFaultSpec.parse(
        {"kind": "lane_abort", "after_tokens": 2}, enabled=True
    )
    assert spec == BatchFaultSpec("lane_abort", 2)
    for value in (
        {"kind": "unknown"},
        {"kind": "cache_evict", "after_tokens": 1},
        {"kind": "lane_abort", "after_tokens": -1},
    ):
        try:
            BatchFaultSpec.parse(value, enabled=True)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid fault spec {value}")
    try:
        BatchFaultSpec.parse({"kind": "lane_abort"}, enabled=False)
    except ValueError:
        pass
    else:
        raise AssertionError("fault injection was enabled outside qualification mode")


def test_http_metrics_use_bounded_route_method_and_status_labels():
    ticks = iter([1.0, 1.25, 2.0, 2.5])
    metrics = HttpRuntimeMetrics(clock=lambda: next(ticks))
    first = metrics.started()
    metrics.completed("POST", "chat_completions", 200, first)
    second = metrics.started()
    metrics.completed("DELETE", "private-route", 599, second)
    snapshot = metrics.prometheus_snapshot()
    assert snapshot["requests"] == {
        ("POST", "chat_completions", "2xx"): 1,
        ("other", "other", "5xx"): 1,
    }
    assert snapshot["durations"]["chat_completions"].count == 1
    assert snapshot["durations"]["other"].count == 1


def test_every_server_route_label_is_admitted_by_http_metrics():
    from mlx2.batch_metrics import http_metric_route

    paths = (
        "/metrics", "/health", "/v1/models", "/v1/status", "/v1/status/batching",
        "/v1/completions", "/v1/chat/completions", "/v1/responses", "/v1/files",
        "/v1/batches", "/v1/embeddings", "/v1/rerank", "/v1/messages",
        "/v1/messages/count_tokens", "/tokenize", "/apply-template",
        "/v1/load_lora_adapter", "/v1/unload_lora_adapter", "/v1/apc/sessions/s",
        "/v1/admin/quiesce", "/v1/responses/r/input_items", "/v1/responses/r",
        "/v1/files/f", "/v1/batches/b",
    )
    metrics = HttpRuntimeMetrics(clock=lambda: 0.0)
    labels = set()
    for path in paths:
        label = http_metric_route(path + "?x=1")
        assert label != "other", path
        labels.add(label)
        metrics.completed("POST", label, 200, 0.0)
    snapshot = metrics.prometheus_snapshot()
    assert {route for _method, route, _status in snapshot["requests"]} == labels
    assert len(labels) == len(paths)
    assert http_metric_route("/v1/unknown") == "other"


def test_disabled_optional_tracer_imports_no_opentelemetry_and_is_inert():
    tracer = OptionalRequestTracer()
    assert tracer.enabled is False
    trace = tracer.start("POST chat_completions", {}, {"http.route": "chat_completions"})
    trace.finish(200)
    assert tracer.prometheus_snapshot() == {
        "started": 0,
        "completed": 0,
        "failed": 0,
        "exported": 0,
        "setup_errors": 0,
        "export_errors": 0,
    }


def test_optional_tracer_initializes_async_otlp_processor_when_installed():
    pytest.importorskip("opentelemetry.sdk.trace")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    tracer = OptionalRequestTracer("http://127.0.0.1:9/v1/traces")
    try:
        assert tracer.enabled is True
    finally:
        tracer.close()


def _jittered_clock(seed=0):
    import random

    rng = random.Random(seed)
    now = [0.0]

    def clock():
        now[0] += rng.uniform(0.010, 0.030)
        return now[0]

    return clock


def test_inter_token_samples_stay_bounded_for_long_outputs():
    # history_size bounds the completed rows; the per-row ITL samples must be
    # bounded too, or 512 long reasoning outputs pin ~128 MiB of floats and
    # every /v1/status/batching read sorts millions of them.
    import gc
    import tracemalloc

    metrics = BatchRuntimeMetrics(history_size=2, clock=_jittered_clock())
    gc.collect()
    tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        for index in range(2):
            request_id = f"long-{index}"
            metrics.admitted(request_id, "tenant", 0)
            metrics.dequeued(request_id, 0)
            metrics.lane_attached(request_id, 1, "ordinary")
            for _ in range(50_000):
                metrics.token(request_id)
            metrics.terminal(request_id, "completed", "length")
        gc.collect()
        retained, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # 100K unbounded float samples cost ~3.2 MB; a bounded sample is far less.
    assert retained - baseline < 512 * 1024
    snapshot = metrics.snapshot()
    assert snapshot["latency_ms"]["itl"]["p50"] is not None
    assert 0 < snapshot["latency_ms"]["itl"]["count"] <= 2 * 256
    assert snapshot["counters"]["tokens_delivered"] == 100_000


def test_bounded_inter_token_sample_covers_the_whole_decode():
    # A slow start and a fast tail: the bounded sample must still see both,
    # not just the most recent gaps.
    ticks = [0.0]
    gaps = [0.100] * 1000 + [0.010] * 1000

    def clock():
        return ticks[0]

    metrics = BatchRuntimeMetrics(history_size=1, clock=clock)
    metrics.admitted("r", "t", 0)
    metrics.dequeued("r", 0)
    metrics.lane_attached("r", 1, "ordinary")
    metrics.token("r")
    for gap in gaps:
        ticks[0] += gap
        metrics.token("r")
    metrics.terminal("r", "completed", "length")
    itl = metrics.snapshot()["latency_ms"]["itl"]
    assert itl["p95"] == pytest.approx(100.0)
    assert itl["p50"] < 100.0


def test_batching_status_computes_distributions_outside_engine_lock():
    import threading
    from types import SimpleNamespace as NS

    from mlx2.serving import ServingEngine

    lock = threading.Lock()
    seen = {}

    class Metrics:
        def snapshot(self, **kwargs):
            seen["engine_lock_held"] = lock.locked()
            return {"gauges": {}, **kwargs}

    engine = NS(lock=lock, snapshot={}, queued_jobs=3, batch_metrics=Metrics())
    status = ServingEngine.batching_status(engine)
    assert status["queue_depth"] == 3
    # Sorting the ITL history under engine.lock blocks dequeue, submit and the
    # worker's 1 Hz snapshot publish for the whole computation.
    assert seen["engine_lock_held"] is False
