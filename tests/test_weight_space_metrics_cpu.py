"""Host metrics and real HTTP scrape tests; synthetic CPU fixtures, no model run."""

import gc
import json
import threading
import weakref
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from test_prometheus import FakeEngine, assert_valid_prometheus_text
from test_weight_space_self_repair_cpu import proposed, setup

from mlx2.prometheus import render_engine_metrics
from mlx2.runtime import weight_space_metrics as metrics
from mlx2.runtime.weight_space_metrics import WeightSpaceMetrics
from mlx2.server import handler_for


def test_bounded_collector_histograms_and_explicit_measurement_availability():
    collector = WeightSpaceMetrics()
    collector.record("program_create", duration_seconds=0.25, bytes_count=128)
    collector.record("program_create", "error", duration_seconds=1.5)
    collector.set_gauge("perplexity_delta", -0.5)
    snapshot = collector.snapshot()
    assert snapshot == collector.snapshot()
    assert snapshot["events"] == {("program_create", "ok"): 1, ("program_create", "error"): 1}
    assert snapshot["durations"]["program_create"]["count"] == 2
    assert snapshot["durations"]["program_create"]["sum"] == 1.75
    assert snapshot["durations"]["program_create"]["buckets"] == [0, 0, 0, 1, 2, 2, 2, 2]
    assert snapshot["measured"] == ["perplexity_delta"]
    for operation, outcome in [("private-session-id", "ok"), ("program_create", "secret-error")]:
        with pytest.raises(ValueError, match="allowlisted"):
            collector.record(operation, outcome)
    with pytest.raises(ValueError, match="finite"):
        collector.set_gauge("perplexity_delta", float("nan"))


def test_real_repair_operations_publish_candidate_rejection_promotion_and_rollback(tmp_path, monkeypatch):
    collector = WeightSpaceMetrics()
    monkeypatch.setattr(metrics, "_METRICS", collector)
    controller, backend, _ = setup(tmp_path)
    candidate = proposed(controller)
    assert collector.snapshot()["gauges"]["active_candidates"] == 1
    backend.fail_case = "heldout"
    assert controller.trial(candidate["id"])["state"] == "rejected"
    assert collector.snapshot()["gauges"]["active_candidates"] == 0
    backend.fail_case = None
    candidate = proposed(controller)
    assert controller.trial(candidate["id"])["state"] == "promoted"
    controller.rollback(candidate["id"], "CPU test")
    snapshot = collector.snapshot()
    for key in [("program_export", "ok"), ("program_load", "ok"), ("program_inspect", "ok"),
                ("failure_report", "ok"), ("repair_propose", "ok"), ("repair_trial", "rejected"),
                ("repair_promote", "ok"), ("repair_rollback", "ok"), ("repair_reject", "rejected")]:
        assert snapshot["events"][key] > 0
    assert snapshot["gauges"]["active_candidates"] == 0
    assert snapshot["bytes"]["program_export"] > 0


def test_expert_lifetime_totals_survive_release_without_double_counting():
    class Experts:
        def __init__(self):
            self.requests, self.resident = 1, 64

        def metrics_snapshot(self):
            return {"requests": self.requests, "resident_bytes": self.resident, "budget_bytes": 128}

    collector = WeightSpaceMetrics()
    source = Experts()
    collector.register_experts(source)
    assert collector.snapshot()["expert_counters"]["requests"] == 1
    source.requests, source.resident = 3, 0
    collector.sample_experts(source)
    assert collector.snapshot()["expert_counters"]["requests"] == 3
    del source
    gc.collect()
    snapshot = collector.snapshot()
    assert snapshot["expert_counters"]["requests"] == 3
    assert snapshot["expert_gauges"]["resident_bytes"] == 0
    assert snapshot["expert_gauges"]["budget_bytes"] == 0


def test_overlay_banks_aggregate_live_payload_and_release_on_collection():
    class Bank:
        def snapshot(self):
            return {"resident_bytes": 128, "tensors": 2, "candidates": 3}

    collector = WeightSpaceMetrics()
    first, second = Bank(), Bank()
    collector.register_overlays(first)
    collector.register_overlays(second)
    assert collector.snapshot()["gauges"]["quantization_overlay_bytes"] == 256
    assert collector.snapshot()["gauges"]["quantization_overlay_tensors"] == 4
    assert collector.snapshot()["gauges"]["quantization_overlay_candidates"] == 6
    del first
    gc.collect()
    assert collector.snapshot()["gauges"]["quantization_overlay_bytes"] == 128


def test_actual_quantization_and_excision_publish_metrics_and_banks_release(tmp_path, monkeypatch):
    from mlx2.experimental.weight_space.excision import LayerExcisionPlan, execute_numpy
    from mlx2.experimental.weight_space.overlays import (
        InMemoryOverlayBank,
        QuantizationSpec,
        TensorTarget,
        compare_quality,
        teacher_forced_quality,
    )

    collector = WeightSpaceMetrics()
    monkeypatch.setattr(metrics, "_METRICS", collector)
    checkpoint, revision = "metrics-cpu-fixture@" + "a" * 64, "e448f070b84642bfc5d11dc04719e40336326a68"
    bank = InMemoryOverlayBank(checkpoint=checkpoint, source_revision=revision, budget_bytes=1024)
    bank.create("q4", TensorTarget(0, None, "linear"), [[1, 2], [3, 4]],
                source_format="float32", spec=QuantizationSpec(4, 4))
    expected_bytes = bank.snapshot()["resident_bytes"]
    assert collector.snapshot()["gauges"]["quantization_overlay_bytes"] == expected_bytes > 0
    assert collector.snapshot()["gauges"]["quantization_overlay_tensors"] == 1
    assert collector.snapshot()["gauges"]["quantization_overlay_candidates"] == 1
    assert collector.snapshot()["gauges"]["quantization_overlay_budget_bytes"] == 1024
    assert collector.snapshot()["gauges"]["quantization_overlay_construction_budget_bytes"] > 0
    assert collector.snapshot()["gauges"]["quantization_overlay_construction_reserved_bytes"] == 0
    assert collector.snapshot()["gauges"]["quantization_overlay_payload_reserved_bytes"] == 0
    with pytest.raises(ValueError, match="exists"):
        bank.create("q4", TensorTarget(0, None, "linear"), [[1]],
                    source_format="float32", spec=QuantizationSpec(4, 4))
    plan = LayerExcisionPlan(checkpoint, revision, 3, (1,), 2)
    execute_numpy([lambda x: x + 1] * 3, np.zeros((1, 2)), plan)
    reference = teacher_forced_quality([[2, 0], [0, 2]], [0, 1],
                                      tokenizer_fingerprint="tokenizer-fixture", input_digest="inputs-fixture")
    candidate = teacher_forced_quality([[1, 0], [0, 1]], [0, 1],
                                      tokenizer_fingerprint="tokenizer-fixture", input_digest="inputs-fixture")
    comparison = compare_quality(reference, candidate)
    snapshot = collector.snapshot()
    assert snapshot["events"]["quantization_overlay", "ok"] == 1
    assert snapshot["events"]["quantization_overlay", "error"] == 1
    assert snapshot["events"]["layer_excision", "ok"] == 1
    assert snapshot["events"]["perplexity_evaluate", "ok"] == 2
    assert snapshot["gauges"]["layer_excision_layers"] == 1
    assert snapshot["gauges"]["perplexity_delta"] == comparison["delta_perplexity"]
    assert snapshot["gauges"]["quality_score_delta"] == comparison["delta_nll_nats"]
    pointer = weakref.ref(bank)
    del bank
    gc.collect()
    assert pointer() is None
    assert collector.snapshot()["gauges"]["quantization_overlay_bytes"] == 0


def test_actual_expert_lifetime_counters_survive_close_and_collection(monkeypatch):
    from test_weight_space_experts_cpu import fixture_controller, load

    collector = WeightSpaceMetrics()
    monkeypatch.setattr(metrics, "_METRICS", collector)
    controller, data, _ = fixture_controller()
    for key in data:
        load(controller, key)
    controller.close()
    before = collector.snapshot()
    assert before["expert_counters"]["streamed_bytes"] == 24
    assert before["expert_counters"]["evictions"] == 1
    pointer = weakref.ref(controller)
    del controller
    gc.collect()
    assert pointer() is None
    after = collector.snapshot()
    assert after["expert_counters"] == before["expert_counters"]
    assert after["expert_gauges"]["resident_bytes"] == 0
    assert after["expert_gauges"]["budget_bytes"] == 0


def test_metrics_http_endpoint_exposes_new_families_and_no_identity_labels():
    engine = FakeEngine()
    engine.weight_space_metrics = WeightSpaceMetrics()
    engine.weight_space_metrics.record("quantization_overlay", duration_seconds=0.1, bytes_count=256)
    engine.weight_space_metrics.set_gauge("quality_score_delta", -0.25)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/metrics", timeout=5) as response:
            payload = response.read().decode()
        assert_valid_prometheus_text(payload)
        assert 'mlx2_weight_space_operations_total{operation="quantization_overlay",outcome="ok"} 1' in payload
        assert 'mlx2_weight_space_processed_bytes_total{operation="quantization_overlay"} 256' in payload
        assert "mlx2_weight_space_quality_score_delta -0.25" in payload
        assert "mlx2_weight_space_live_enabled 0" in payload
        assert 'mlx2_weight_space_measurement_available{measurement="quality_score_delta"} 1' in payload
        assert 'mlx2_weight_space_measurement_available{measurement="perplexity_delta"} 0' in payload
        assert "mlx2_weight_space_expert_streamed_bytes_total 0" in payload
        new_lines = "\n".join(line for line in payload.splitlines() if line.startswith("mlx2_weight_space_"))
        assert all(label not in new_lines for label in ("session=", "tenant=", "target=", "program_id=", "expert_id="))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_private_request_activity_is_not_exposed_to_public_or_tenant_scrapes():
    engine = FakeEngine()
    engine.weight_space_metrics = WeightSpaceMetrics()
    engine.weight_space_metrics.record("repair_trial")
    assert "mlx2_weight_space_" not in render_engine_metrics(engine, include_request_metrics=False)
    assert "mlx2_weight_space_" not in render_engine_metrics(engine, tenant_id="tenant")


def test_runtime_surgery_metrics_keep_selected_observed_and_qualified_separate():
    engine = FakeEngine()
    engine.weight_space_metrics = WeightSpaceMetrics()
    engine.weight_overlays = SimpleNamespace(status=lambda: {
        "implemented": True, "qualified": False, "selected": True, "observed_used": False,
        "counts": {"selected": 1, "applied": 0}, "bound_requests": 1,
        "dequantized_bytes": 64, "materialized_budget_bytes": 256,
    })
    engine.layer_excision = SimpleNamespace(status=lambda: {
        "implemented": True, "qualified": False, "selected": True, "observed_used": True,
        "counts": {"selected": 2, "skipped_blocks": 3, "healing_applications": 1},
        "bound_requests": 1, "healing_bytes": 16,
    })
    engine.expert_masks = SimpleNamespace(status=lambda: {
        "implemented": True, "qualified": False, "selected": True, "observed_used": True,
        "counts": {"selected": 1, "masked_router_calls": 2}, "bound_requests": 1,
    })
    payload = render_engine_metrics(engine)
    assert 'mlx2_weight_space_runtime_state{feature="quantization_overlay",state="selected"} 1' in payload
    assert 'mlx2_weight_space_runtime_state{feature="quantization_overlay",state="observed_used"} 0' in payload
    assert 'mlx2_weight_space_runtime_state{feature="layer_excision",state="observed_used"} 1' in payload
    assert 'mlx2_weight_space_runtime_state{feature="layer_excision",state="qualified"} 0' in payload
    assert 'mlx2_weight_space_expert_mask_events_total{event="masked_router_calls"} 2' in payload
    assert "mlx2_weight_space_overlay_dequantized_bytes 64" in payload
    assert "mlx2_weight_space_layer_excision_healing_bytes 16" in payload


def test_rejected_live_tool_dispatch_is_visible_on_actual_metrics_endpoint(tmp_path, monkeypatch):
    pytest.importorskip("jsonschema")
    from test_weight_space_live_cpu import live

    collector = WeightSpaceMetrics()
    monkeypatch.setattr(metrics, "_METRICS", collector)
    _, _, engine, service = live.__wrapped__(tmp_path)
    engine.multi_lora.status = lambda: {"enabled": True, "counts": {}}
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(
        engine, weight_space_service=service, admin_token="operator-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.dumps({"tool": "secret-arbitrary-tool", "arguments": {}}).encode()
        request = Request(f"http://127.0.0.1:{server.server_port}/v1/admin/weight-space/call",
                          data=body, headers={"Authorization": "Bearer operator-token",
                                              "Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=5)
        assert error.value.code == 400
        with urlopen(f"http://127.0.0.1:{server.server_port}/metrics", timeout=5) as response:
            payload = response.read().decode()
        assert 'mlx2_weight_space_operations_total{operation="tool_dispatch",outcome="error"} 1' in payload
        assert "secret-arbitrary-tool" not in payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
