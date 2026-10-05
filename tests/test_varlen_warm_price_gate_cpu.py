"""Fail-closed host contracts for the unselected APCv2 warm price bootstrap."""

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

from mlx2.runtime.paged_pack_price import (
    load_price, load_research_warm_calibration,
)
from mlx2.runtime.paged_pack_scheduler import PrefillOption, ReservedRows
from mlx2.runtime.paged_pack_scheduler import PrefillOffer
from mlx2.runtime.paged_atomic_owner import PagedAtomicRequestOwner
from mlx2.runtime.paged_request_route import RouteCapability, coordinate_paged_request
from mlx2.runtime.paged_request_transaction import CandidateRequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
import varlen_http_cold_warm_price_gate as http_gate
import varlen_cold_warm_gate_run as combined
import varlen_warm_request_price as warm


IDENTITY = {"host": "test", "hardware": "M5", "artifact_sha256": "a" * 64,
            "source_commit": "b" * 40, "source_tree_sha256": "c" * 64,
            "mlx_wheel_version": "test", "mlx_wheel_sha256": "d" * 64,
            "kernel_sha256": "e" * 64}


def _proof(arm):
    proof = {"arm": arm, "request_inserted": True, "admitted": True,
             "cache_transaction_published": True, "response_emitted": True,
             "sampled_output": True, "synchronized": True,
             "request_removed": True, "request_state_released": True,
             "output_token_ids": [10, 11], "apcv2_cached_tokens": 63,
             "apcv2_stores_delta": 0, "q1_step_ms": 1.0,
             "ordinary_model_forward_calls": 2}
    if arm == "paged":
        proof.update(route_receipt={"route": "native_qwen3_paged",
                                    "research_executed": True,
                                    "selected": False, "observed_used": False,
                                    "apcv2_restored_tokens": 63},
                     first_response_receipt={"research_executed": True,
                                             "selected": False,
                                             "observed_used": False,
                                             "native_read_calls": 28,
                                             "terminal_successes": 28},
                     second_response_receipt={"research_executed": True,
                                              "selected": False,
                                              "observed_used": False,
                                              "native_read_calls": 56,
                                              "terminal_successes": 56},
                     ordinary_model_forward_calls=0, paged_read_calls=56,
                     terminal_successes=56, pending_native_epochs=0,
                     retained_pages=0, owner_fully_retired=True,
                     writer_poisoned=False)
    return proof


def _receipt():
    shape = {"prompt_tokens": 64, "cached_tokens": 63, "suffix_rows": 1,
             "output_tokens": 2, "decode_rows_after_prefill": 1}
    result = {"schema": warm.SCHEMA, "status": "calibrated",
              "gpu_executed": True, "measurement_scope": warm.SCOPE,
              "measured_route": "native_qwen3_paged", "profile_id": "warm64-test",
              "identity": dict(IDENTITY), "context_tokens": [64],
              "gpuq_owner": {"session": "test", "lease_id": "test-warm", "pid": 4},
              "request_shape": shape,
              "peak_resident_bytes": 4096,
              "cases": [{"warm_native_request_ms": [5.0, 6.0, 7.0],
                         "warm_ordinary_request_ms": [2.0, 3.0, 4.0],
                         "native_q1_step_ms": [1.0] * 3,
                         "ordinary_q1_step_ms": [1.0] * 3,
                         "request_proofs": [{"ordinary": _proof("ordinary"),
                                             "paged": _proof("paged")}
                                            for _ in range(3)],
                         "kernel_engagement": {"paged_read_calls": 168,
                                               "terminal_successes": 168}}],
              "qualified": False, "selected": False,
              "serving_selected": False, "research_executed": True,
              "price_usable": False}
    _digest(result)
    return result


def _digest(result):
    result.pop("research_warm_evidence_sha256", None)
    payload = json.dumps(result, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    result["research_warm_evidence_sha256"] = hashlib.sha256(payload).hexdigest()


def _load(tmp_path, result):
    path = tmp_path / "warm.json"
    path.write_text(json.dumps(result))
    return load_research_warm_calibration(
        path, live_identity=IDENTITY, context_tokens=(64,))


def test_real_shape_research_loader_is_distinct_from_serving_price(tmp_path):
    data = _receipt()
    price = _load(tmp_path, data)
    assert price.validated and price.warm_bound_ms == 7.0
    assert price.estimate_ms((), (1, PrefillOption(1, 0))) == 7.0
    assert price.estimate_ms((ReservedRows(1, "decode", 1, 0, 10),), None) == 1.0
    with pytest.raises(ValueError, match="warm research calibration admits"):
        price.estimate_ms((), (1, PrefillOption(2, 0)))
    with pytest.raises(ValueError, match="measured price receipt"):
        load_price(tmp_path / "warm.json", live_identity=IDENTITY,
                   context_tokens=(64,))


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(selected=True),
    lambda d: d.update(price_usable=True),
    lambda d: d["request_shape"].update(cached_tokens=62),
    lambda d: d["cases"][0]["request_proofs"][0]["paged"]["route_receipt"].update(
        selected=True),
    lambda d: d["cases"][0]["request_proofs"][0]["paged"]["route_receipt"].update(
        apcv2_restored_tokens=62),
    lambda d: d["cases"][0]["request_proofs"][0]["paged"].update(
        pending_native_epochs=1),
    lambda d: d["cases"][0]["request_proofs"][0]["paged"].update(
        owner_fully_retired=False),
    lambda d: d["cases"][0]["request_proofs"][0]["paged"].update(
        output_token_ids=[10, 12]),
    lambda d: d["cases"][0]["request_proofs"][0]["paged"]["first_response_receipt"].update(
        terminal_successes=27),
])
def test_loader_refuses_semantic_forgery_even_with_rehashed_body(tmp_path, mutation):
    data = copy.deepcopy(_receipt())
    mutation(data)
    _digest(data)
    with pytest.raises(ValueError):
        _load(tmp_path, data)


def test_loader_refuses_edited_raw_digest_and_wrong_identity(tmp_path):
    data = _receipt()
    data["cases"][0]["warm_native_request_ms"][0] += 1
    with pytest.raises(ValueError, match="digest"):
        _load(tmp_path, data)
    data = _receipt()
    data["identity"]["source_commit"] = "f" * 40
    _digest(data)
    with pytest.raises(ValueError, match="identity"):
        _load(tmp_path, data)


def test_http_gate_preflight_requires_same_source(monkeypatch):
    cold_manifest = {"paths": {"artifact": "a", "mlx_wheel": "b", "kernel": "c"}}
    warm_manifest = copy.deepcopy(cold_manifest)
    monkeypatch.setattr(http_gate, "cold_preflight", lambda _: {"identity": IDENTITY})
    monkeypatch.setattr(http_gate, "warm_preflight", lambda _: {"identity": IDENTITY})
    assert http_gate.preflight(cold_manifest, warm_manifest)["gpu_executed"] is False
    monkeypatch.setattr(http_gate, "warm_preflight", lambda _: {"identity": {**IDENTITY,
                                                                           "kernel_sha256": "f" * 64}})
    with pytest.raises(RuntimeError, match="different live sources"):
        http_gate.preflight(cold_manifest, warm_manifest)


def test_http_route_must_report_genuine_selected_and_observed_use():
    route = {"route": "native_qwen3_paged", "implemented": True,
             "qualified": False, "selected": True, "observed_used": True,
             "price_provenance": "research_calibrated",
             "price_evidence_sha256": "a" * 64,
             "ordinary_forward_calls": 0, "native_read_calls": 56,
             "terminal_successes": 56, "apcv2_restored_tokens": 63}
    body = {"mlx2": {"route": "native_qwen3_paged", "cached_tokens": 63,
                      "route_receipt": route}}
    samples = [(10, {"native_read_calls": 28}),
               (11, {"native_read_calls": 56, "terminal_successes": 56})]
    assert http_gate._native_response(body, cached_tokens=63,
                                      evidence="a" * 64, samples=samples) is route
    for field in ("selected", "observed_used"):
        bad = copy.deepcopy(body)
        bad["mlx2"]["route_receipt"][field] = False
        with pytest.raises(RuntimeError, match="proof missing"):
            http_gate._native_response(bad, cached_tokens=63,
                                       evidence="a" * 64, samples=samples)
    bad = copy.deepcopy(body)
    bad["mlx2"]["route_receipt"]["apcv2_restored_tokens"] = 62
    with pytest.raises(RuntimeError, match="proof missing"):
        http_gate._native_response(bad, cached_tokens=63,
                                   evidence="a" * 64, samples=samples)


def test_http_warm_prompt_requires_exact_token_prefix():
    class Tokenizer:
        def decode(self, tokens):
            return ",".join(map(str, tokens))

    class Adapter:
        tokenizer = Tokenizer()

        def prompt_tokens(self, request):
            return [int(token) for token in request["prompt"].split(",")]

    prefix = [1000] * 63
    text, tokens = http_gate._warm_prompt(Adapter(), Tokenizer().decode(prefix), prefix)
    assert len(tokens) == 64 and tokens[:63] == prefix
    assert text.endswith(",1000")


def test_warm_research_price_needs_explicit_permit_and_exact_suffix(tmp_path):
    price = _load(tmp_path, _receipt())
    owner = PagedAtomicRequestOwner("r1", {"kv": ["base"] * 63},
                                    supported_planes=("kv",), enabled=True)
    request = CandidateRequest(7, "r1", 1, ("kv",))
    capability = RouteCapability(True, False, ("kv",), price.profile_id)

    def run(candidate):
        candidate.stage("kv", ["suffix"])
        return 1

    def probe(*, permit, rows=1):
        return coordinate_paged_request(
            request, owner, capability, (),
            (PrefillOffer(7, (PrefillOption(rows, 0),)),),
            price=price, live_identity=IDENTITY, context_tokens=(64,),
            row_capacity=1, free_pages=2, run_candidate=run,
            permit_candidate=True, permit_research_calibration=permit)

    assert probe(permit=False).reason == "warm_research_calibration_shape_refused"
    assert probe(permit=True, rows=2).reason == "warm_research_calibration_shape_refused"
    accepted = probe(permit=True)
    assert accepted.published and accepted.selected


def test_calibration_orchestrator_pairs_three_real_arm_interfaces(monkeypatch, tmp_path):
    import mlx.core as mx

    calls = []

    class FakeDriver:
        def __init__(self, manifest):
            assert manifest["profile_id"] == "warm64-test"

        def run_request(self, arm):
            calls.append(arm)
            proof = copy.deepcopy(_proof(arm))
            proof["q1_step_ms"] = 1e-9
            return proof

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(warm, "preflight", lambda _: {"identity": IDENTITY})
    monkeypatch.setattr(warm, "_gpuq_owner", lambda: {
        "session": "test", "lease_id": "test-warm", "pid": 4})
    monkeypatch.setattr(warm, "WarmRequestDriver", FakeDriver)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    manifest = {"profile_id": "warm64-test", "identity": dict(IDENTITY),
                "request_shape": _receipt()["request_shape"]}
    result = warm.execute(manifest)
    assert calls == ["ordinary", "paged"] * 3 + ["closed"]
    assert result["selected"] is False and result["price_usable"] is False
    assert result["cases"][0]["kernel_engagement"] == {
        "paged_read_calls": 168, "terminal_successes": 168}
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(result))
    assert load_research_warm_calibration(
        path, live_identity=IDENTITY, context_tokens=(64,)).validated


def test_frozen_gate_manifests_share_exact_source_and_shape(monkeypatch, tmp_path):
    from mlx2.runtime import paged_price_identity

    monkeypatch.setattr(paged_price_identity, "compute_live_price_identity",
                        lambda *args, **kwargs: IDENTITY)
    monkeypatch.setattr("varlen_pack_price_bench.verify_artifact_manifest",
                        lambda _: {"root": str(tmp_path / "model")})
    output = tmp_path / "frozen"
    result = combined.freeze(tmp_path / "artifact.json", tmp_path / "mlx.whl",
                             tmp_path / "kernel.so", output)
    cold = json.loads((output / "cold-manifest.json").read_text())
    warm_manifest = json.loads((output / "warm-manifest.json").read_text())
    assert result["status"] == "frozen"
    assert cold["identity"] == warm_manifest["identity"] == IDENTITY
    assert cold["paths"] == warm_manifest["paths"]
    assert cold["context_tokens"] == [63]
    assert warm_manifest["context_tokens"] == [64]
    assert warm_manifest["request_shape"]["cached_tokens"] == 63
    assert warm_manifest["hard_seconds"] == 180


def test_combined_gate_stops_after_failed_warm_stage_and_keeps_receipts(
    monkeypatch, tmp_path,
):
    cold_manifest = tmp_path / "cold-manifest.json"
    warm_manifest = tmp_path / "warm-manifest.json"
    cold_manifest.write_text("{}")
    warm_manifest.write_text("{}")
    monkeypatch.setattr(combined, "preflight", lambda *_: {"identity": IDENTITY})
    monkeypatch.setattr(combined, "_gpuq_owner", lambda: {
        "session": "test", "lease_id": "test-warm", "pid": 4})
    stages = []

    def stage(label, command, output, deadline):
        stages.append(label)
        (output / f"{label}.json").write_text("preserved")
        if label == "warm":
            raise RuntimeError("warm parity failed")
        return {"label": label}

    monkeypatch.setattr(combined, "_run_stage", stage)
    output = tmp_path / "gate"
    with pytest.raises(RuntimeError, match="warm parity failed"):
        combined.run(cold_manifest, warm_manifest, output)
    assert stages == ["cold", "warm"]
    assert (output / "cold.json").read_text() == "preserved"
    assert (output / "warm.json").read_text() == "preserved"
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "failed" and run["current_stage"] == "warm"
    assert run["stages"] == [{"label": "cold"}]


def test_http_ordinary_arms_require_exact_cache_counts():
    http_gate._ordinary_response({"mlx2": {"route": "ordinary",
                                           "cached_tokens": 0}}, "cold_ordinary")
    http_gate._ordinary_response({"mlx2": {"route": "ordinary",
                                           "cached_tokens": 63}}, "warm_ordinary")
    for label, cached in (("cold_ordinary", 1), ("warm_ordinary", 0)):
        with pytest.raises(RuntimeError, match="APCv2 hit differs"):
            http_gate._ordinary_response(
                {"mlx2": {"route": "ordinary", "cached_tokens": cached}}, label)
    with pytest.raises(RuntimeError, match="ordinary route"):
        http_gate._ordinary_response(
            {"mlx2": {"route": "native_qwen3_paged", "cached_tokens": 63}},
            "warm_ordinary")


def test_combined_stage_kills_child_on_sampled_rss_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(combined, "_child_rss_bytes",
                        lambda _pid: combined.MAX_BYTES + 1)
    with pytest.raises(MemoryError, match="sampled RSS"):
        combined._run_stage("warm", [sys.executable, "-c",
                                      "import time; time.sleep(10)"],
                            tmp_path, time.monotonic() + 5)
    receipt = json.loads((tmp_path / "warm.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["error_type"] == "MemoryError"
    assert receipt["sampled_peak_rss_bytes"] == combined.MAX_BYTES + 1
    assert (tmp_path / "warm.log").is_file()
