"""Audit driver contracts without model loading or a GPU lease."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def unit():
    path = Path(__file__).parents[1] / "docs/audits/2026-11-03-bug-sweep/gpu_artifact_unit.py"
    spec = importlib.util.spec_from_file_location("bug_sweep_gpu_unit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response(tokens, *, cached=0, route="ordinary", width=1):
    return {"mlx2": {"token_trace": {"token_ids": tokens, "execution_widths": [width] * len(tokens)},
                     "prompt_tokens": 4096, "cached_tokens": cached, "route": route,
                     "profile": "candidate", "cache": "apcv2", "cache_checkpoint_role": None},
            "audit": {"functional_oracle": True}}


def test_parity_requires_explicit_apcv2_cold_and_selected_route(unit):
    cold, warm = response([1, 2]), response([1, 2], cached=4095)
    assert unit.compare(cold, warm, "ordinary")["passed"]
    cold["mlx2"]["cached_tokens"] = 1
    assert not unit.compare(cold, warm, "ordinary")["passed"]
    cold["mlx2"]["cached_tokens"] = 0
    del cold["mlx2"]["cache_checkpoint_role"]
    assert not unit.compare(cold, warm, "ordinary")["passed"]
    cold["mlx2"]["cache_checkpoint_role"] = None
    assert not unit.compare(cold, warm, "native_mtp")["passed"]


@pytest.mark.parametrize("left,right,index", [([1, 2], [1, 3], 1), ([1], [1, 2], 1), ([1], [1], None)])
def test_first_difference_is_token_position_not_text(unit, left, right, index):
    assert unit.first_difference(left, right) == index


def test_verifier_rows_cannot_relax_physical_b1_parity(unit):
    cold, warm = response([1, 2], width=4), response([1, 3], cached=4095, width=4)
    result = unit.first_position_evidence(cold, warm, cold, warm, 1)
    assert result == {"classification": "unclassified", "reason": "physical B1 requires exact parity"}


def test_identifier_oracle_normalizes_only_dashes(unit):
    assert unit.oracle("SWEEP–7439–AZ", "SWEEP-7439-AZ")
    assert not unit.oracle("SWEEP-7439-Az", "SWEEP-7439-AZ")


def test_count_request_uses_anthropic_schema_and_implicit_thinking_off(unit, monkeypatch):
    from mlx2.anthropic_compat import anthropic_request_model, anthropic_request_to_chat

    calls = []

    def count(base, path, body):
        assert path == "/v1/messages/count_tokens"
        assert anthropic_request_model(body) == "model"
        assert anthropic_request_to_chat(body, count_tokens=True)["enable_thinking"] is False
        calls.append(body)
        return {"input_tokens": len(body["messages"][0]["content"].split())}

    monkeypatch.setattr(unit, "http", count)
    _, tokens = unit.prompt_for("http://localhost", "model", 512, "CODE", "unit")
    assert tokens <= 512 and len(calls) > 1


def test_http_error_keeps_server_diagnostic_body(unit, monkeypatch):
    import io
    import urllib.error

    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("http://localhost/count", 400, "invalid", {},
                                     io.BytesIO(b'{"error":"unsupported field"}'))

    monkeypatch.setattr(unit.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError, match="POST /count: HTTP 400:.*unsupported field"):
        unit.http("http://localhost", "/count", {"model": "test"})


@pytest.mark.parametrize("stats,copy,passed", [({}, {}, False), ({"draft_proposed": 2}, {}, True),
                                            ({}, {"copy_proposed": 3}, True)])
def test_native_selection_without_observed_proposals_cannot_pass(unit, stats, copy, passed):
    value = response([1])
    value["mlx2"]["mtp"] = {"route": "segmented_self_mtp", "stats": stats, "copy_draft": copy}
    assert unit.native_engagement(value)["passed"] is passed


def test_qsa_diagnostics_keep_admission_and_observation_separate(unit):
    before = {"execution": {"qsa_b1_selected": {"enabled": True, "calls": 2}}}
    after = {"execution": {"qsa_b1_selected": {"enabled": True, "calls": 8, "declines": {}}}}
    result = unit.feature_diagnostics(before, after, ordinary=True, prompt_tokens=4096)
    assert result["qsa_b1_selected"]["calls_during_requests"] == 6
    assert result["qsa_b1_selected"]["eligible_context_and_route"] is True
    assert unit.feature_diagnostics({}, {}, ordinary=True, prompt_tokens=512)["qsa_b1_selected"]["selected"] is False


def test_final_status_waits_for_post_completion_worker_publication(unit, monkeypatch):
    now, counters = [0.], iter([2, 2, 3])
    monkeypatch.setattr(unit.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(unit.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(unit, "http", lambda *args, **kw: {
        "host_snapshot_timing": {"collections": next(counters)}, "execution": {"ready": True}})
    status, receipt = unit.fresh_worker_status("http://localhost")
    assert status["host_snapshot_timing"]["collections"] == 3
    assert receipt["initial_collection"] == 2 and receipt["polls"] == 3
    assert receipt["elapsed_seconds"] == pytest.approx(.2)


def test_final_status_refuses_stale_counters_after_bounded_wait(unit, monkeypatch):
    now = [0.]
    monkeypatch.setattr(unit.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(unit.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(unit, "http", lambda *args, **kw: {"host_snapshot_timing": {"collections": 2}})
    with pytest.raises(RuntimeError, match="did not refresh.*within 0.3s"):
        unit.fresh_worker_status("http://localhost", timeout=.3)
    assert now[0] == pytest.approx(.3)


def test_final_status_requires_worker_publication_evidence(unit, monkeypatch):
    monkeypatch.setattr(unit, "http", lambda *args, **kw: {"counts": {"completed": 2}})
    with pytest.raises(RuntimeError, match="collection counter is absent"):
        unit.fresh_worker_status("http://localhost")
