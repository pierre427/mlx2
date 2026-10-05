"""CPU-only price identity and benchmark-plan checks."""

import importlib.util
import json
from pathlib import Path

import pytest

from mlx2.runtime.paged_pack_price import PACK_STEP_UNIT, SCHEMA, evidence_sha256, load_price
from mlx2.runtime.paged_pack_scheduler import PrefillOption, ReservedRows, choose_paged_pack


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/research/varlen_pack_price_bench.py"
spec = importlib.util.spec_from_file_location("varlen_pack_price_bench", SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


IDENTITY = {"host": "test-host", "hardware": "M5 Max", "artifact_sha256": "a" * 64,
            "source_commit": "b" * 40, "source_tree_sha256": "c" * 64,
            "mlx_wheel_version": "test", "mlx_wheel_sha256": "d" * 64,
            "kernel_sha256": "e" * 64}
READY = (ReservedRows(7, "decode", 1, 0, 100),)


def receipt(scope="complete_request"):
    plain = {"request_inserted": True, "admitted": True,
             "cache_transaction_published": True, "response_emitted": True,
             "sampled_output": True, "synchronized": True,
             "request_removed": True, "request_state_released": True,
             "route_receipt": {"route": "ordinary"},
             "ordinary_model_forward_calls": 1, "output_token_id": 42}
    paged = {**plain, "route_receipt": {"route": "native_qwen3_paged",
                                           "selected": True, "observed_used": True},
             "ordinary_model_forward_calls": 0, "model_layers": 1,
             "paged_read_calls": 1, "terminal_successes": 1,
             "pending_native_epochs": 0, "retained_pages": 0}
    cases = []
    for rows in (0, 3):
        step = {"scope": PACK_STEP_UNIT, "reserved": [["decode", 1]],
                "prefill_rows": rows, "context_tokens": [63, 65, 129],
                "active_lane_count": 1 + bool(rows), "generator_step": True,
                "sampled_response": True, "synchronized": True}
        cases.append({"reserved": [["decode", 1]], "prefill_rows": rows,
                      "pack_step_ms": [4.0 + rows, 5.0 + rows, 6.0 + rows],
                      "ordinary_pack_step_ms": [5.0 + rows, 6.0 + rows, 7.0 + rows],
                      "complete_native_request_ms": [8.0 + rows, 9.0 + rows, 10.0 + rows],
                      "ordinary_request_ms": [9.0 + rows, 10.0 + rows, 11.0 + rows],
                      "request_proofs": [
                          {"ordinary": {**plain, "scheduler_step": step,
                                        "scheduler_step_ms": 5.0 + rows + i},
                           "paged": {**paged, "scheduler_step": step,
                                     "scheduler_step_ms": 4.0 + rows + i}}
                          for i in range(3)],
                      "kernel_engagement": {"paged_read_calls": 3,
                                            "terminal_successes": 3}})
    value = {"schema": SCHEMA, "status": "measured", "measurement_scope": scope,
            "price_unit": PACK_STEP_UNIT,
            "gpu_executed": True, "gpuq_owner": {"session": "test", "lease_id": "test-1", "pid": 1},
            "measured_route": "native_qwen3_paged",
            "profile_id": "host:artifact:source:wheel:request", "identity": IDENTITY,
            "context_tokens": [63, 65, 129],
            "cases": cases}
    value["complete_request_evidence_sha256"] = evidence_sha256(value)
    return value


def load(tmp_path, data, **kwargs):
    path = tmp_path / "price.json"
    path.write_text(json.dumps(data))
    return load_price(path, live_identity=kwargs.get("identity", IDENTITY),
                      context_tokens=kwargs.get("contexts", (63, 65, 129)))


def test_dry_plan_is_bounded_and_has_all_mandatory_rows():
    plan = bench.dry_run()
    assert plan["gpu_executed"] is False
    assert plan["measurement_scope"] == "complete_model_forward"
    assert plan["hard_seconds"] == 180 and plan["max_resident_bytes"] == 24 * 1024**3
    assert plan["planned_forwards"] == 48
    assert {1, 3, 17} <= {rows for shape in bench.CASES for _, rows in shape}
    assert {63, 65, 129} <= set(plan["context_tokens"])


def test_artifact_manifest_binds_model_bytes(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "weights.safetensors"
    weights.write_bytes(b"weights")
    manifest = tmp_path / "artifact.json"
    manifest.write_text(json.dumps({"root": str(model), "files": {
        weights.name: bench.sha256(weights)}}))
    bench.verify_artifact_manifest(manifest)
    weights.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="bytes differ"):
        bench.verify_artifact_manifest(manifest)


def test_exact_request_price_can_drive_default_off_selector(tmp_path):
    price = load(tmp_path, receipt())
    assert price.estimate_ms(READY, None) == 6.0
    assert price.estimate_ms((), None) == 0.0
    decision = choose_paged_pack(READY, (), price=price, row_capacity=32,
                                 free_pages=1, permit_candidate=True)
    assert decision.accepted and decision.reason == "decode_only"
    assert choose_paged_pack(READY, (), price=price, row_capacity=32,
                             free_pages=1).reason == "paged_pack_disabled"
    with pytest.raises(ValueError, match="unmeasured"):
        price.estimate_ms(READY, (8, PrefillOption(17, 0)))


@pytest.mark.parametrize("scope", ["attention_kernel_only", "complete_model_forward", None])
def test_non_request_timings_never_load_for_serving(tmp_path, scope):
    with pytest.raises(ValueError, match="complete-request"):
        load(tmp_path, receipt(scope))


def test_identity_context_engagement_and_raw_times_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="identity differs"):
        load(tmp_path, receipt(), identity={**IDENTITY, "host": "another-host"})
    with pytest.raises(ValueError, match="context lengths"):
        load(tmp_path, receipt(), contexts=(63, 65, 130))
    bad = receipt()
    bad["cases"][0]["kernel_engagement"]["paged_read_calls"] = 0
    with pytest.raises(ValueError, match="engagement"):
        load(tmp_path, bad)
    bad = receipt()
    bad["cases"][0]["pack_step_ms"] = [1.0]
    with pytest.raises(ValueError, match="three"):
        load(tmp_path, bad)
    bad = receipt()
    bad["complete_request_evidence_sha256"] = ""
    with pytest.raises(ValueError, match="digest"):
        load(tmp_path, bad)
    bad = receipt()
    bad["cases"][0]["pack_step_ms"][0] += 1
    with pytest.raises(ValueError, match="scheduler step"):
        load(tmp_path, bad)
    bad = receipt()
    bad["cases"][0]["request_proofs"][0]["paged"]["route_receipt"]["observed_used"] = False
    with pytest.raises(ValueError, match="observed native"):
        load(tmp_path, bad)


def test_cold_request_total_cannot_be_relabelled_as_pack_step(tmp_path):
    bad = receipt()
    bad.pop("price_unit")
    bad["complete_request_evidence_sha256"] = evidence_sha256(bad)
    with pytest.raises(ValueError, match="pack-step price unit"):
        load(tmp_path, bad)

    bad = receipt()
    bad["cases"][0]["forward_ms"] = bad["cases"][0].pop("pack_step_ms")
    bad["complete_request_evidence_sha256"] = evidence_sha256(bad)
    with pytest.raises(ValueError, match="complete-forward timing"):
        load(tmp_path, bad)


def test_declared_shape_and_context_must_match_observed_step(tmp_path):
    for field, value in (("reserved", [["verify", 3]]),
                         ("context_tokens", [63, 65, 130]),
                         ("generator_step", False)):
        bad = receipt()
        bad["cases"][0]["request_proofs"][0]["paged"]["scheduler_step"] = {
            **bad["cases"][0]["request_proofs"][0]["paged"]["scheduler_step"],
            field: value}
        bad["complete_request_evidence_sha256"] = evidence_sha256(bad)
        with pytest.raises(ValueError, match="scheduler step shape"):
            load(tmp_path, bad)


def test_complete_request_total_must_cover_pack_step(tmp_path):
    bad = receipt()
    bad["cases"][0]["complete_native_request_ms"][0] = 1.0
    bad["complete_request_evidence_sha256"] = evidence_sha256(bad)
    with pytest.raises(ValueError, match="separate paired pack-step"):
        load(tmp_path, bad)
