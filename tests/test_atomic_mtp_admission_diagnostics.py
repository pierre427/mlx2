"""CPU-only evidence binding for failed atomic MTP admission boundaries."""

import json

import pytest
from test_admission_progress import _atomic_joining_batch, _SizedCache

from mlx2.runtime.memory_policy import (
    SelfMTPLaneAdmissionController,
    _make_self_mtp_admission_callback,
)


def rows(width=4):
    return tuple((uid, 16384, 2, True, 1.0, 0.25) for uid in range(width))


def test_atomic_failure_binds_reclaim_eviction_and_settle_evidence():
    free = [20.0]
    releases = iter([21.0, 21.25, 21.5])
    entries = [object(), object()]

    def reclaim():
        free[0] = next(releases)

    def evict():
        return bool(entries and entries.pop())

    callback = _make_self_mtp_admission_callback(
        free_memory=lambda: free[0], reclaim_memory=reclaim,
        evict_unused_cache=evict,
        settle_memory=lambda timeout: {
            "exit": "quiet", "waited": 0.01, "released_bytes": 0,
        },
    )
    refused = callback.atomic(rows())
    assert refused == {uid: "queue" for uid in range(4)}
    diagnostic = refused.diagnostics
    assert diagnostic["initial_free_gib"] == 20.0
    assert diagnostic["final_free_gib"] == 21.5
    assert diagnostic["usable_gib"] == 1.5
    assert diagnostic["estimated_gib"] == 0.0
    assert diagnostic["required_gib_by_depth"]["1"] > 1.5
    assert diagnostic["required_gib_by_depth"]["2"] > diagnostic["required_gib_by_depth"]["1"]
    assert diagnostic["reclaim_calls"] == 3
    assert diagnostic["evicted_checkpoints"] == 2
    assert diagnostic["eviction_attempts"] == 3
    assert diagnostic["eviction_attempt_cap"] == 32
    assert diagnostic["settle_budget_seconds"] == 1.0
    assert diagnostic["settle_attempts"] == 3
    assert len(diagnostic["settle_reports"]) == 3
    assert all(report["exit"] == "quiet" for report in diagnostic["settle_reports"])
    assert diagnostic["stage"] == "queue"
    assert diagnostic["modes"] == ["queue"] * 4
    assert diagnostic["draft_depths"] == [None] * 4

    snapshot = json.dumps(diagnostic, sort_keys=True, allow_nan=False)
    free[0] = 100.0
    assert callback.atomic(rows()) == {uid: 2 for uid in range(4)}
    assert json.dumps(diagnostic, sort_keys=True, allow_nan=False) == snapshot


def test_atomic_capacity_cap_is_visible_without_evicting_apc():
    evictions = []
    callback = _make_self_mtp_admission_callback(
        SelfMTPLaneAdmissionController(saturation_lane_cap=2),
        free_memory=lambda: 100.0,
        evict_unused_cache=lambda: evictions.append(True) or True,
    )
    decision = callback.atomic(rows())
    assert decision == {uid: "queue" for uid in range(4)}
    diagnostic = decision.diagnostics
    assert diagnostic["saturation_lane_cap"] == 2
    assert diagnostic["required_gib_by_depth"]["2"] < diagnostic["usable_gib"]
    assert diagnostic["evicted_checkpoints"] == 0
    assert evictions == []


def test_atomic_unmeasurable_headroom_and_eviction_trace_are_bounded():
    callback = _make_self_mtp_admission_callback(
        free_memory=lambda: float("nan"),
        evict_unused_cache=lambda: True,
    )
    diagnostic = callback.atomic(rows(100)).diagnostics
    assert diagnostic["initial_free_gib"] is None
    assert diagnostic["final_free_gib"] is None
    assert diagnostic["row_count"] == 100
    assert len(diagnostic["rows"]) == len(diagnostic["modes"]) == 64
    json.dumps(diagnostic, allow_nan=False)

    callback = _make_self_mtp_admission_callback(
        free_memory=lambda: 20.0,
        reclaim_memory=lambda: None,
        evict_unused_cache=lambda: True,
        settle_memory=lambda timeout: None,
    )
    diagnostic = callback.atomic(rows()).diagnostics
    assert diagnostic["eviction_attempts"] == 32
    assert diagnostic["reclaim_calls"] == 33
    assert len(diagnostic["settle_reports"]) == 33


def test_joining_failure_retains_exact_decision_after_later_admission():
    free = [20.0]
    callback = _make_self_mtp_admission_callback(free_memory=lambda: free[0])
    batch = _atomic_joining_batch({})
    batch.mtp_admission = callback
    assert batch._admit_mtp_joining(20) == 0
    failure = batch.take_atomic_cohort_failures()[0]
    diagnostic = failure["admission"]
    assert diagnostic["boundary"] == "joining"
    assert diagnostic["row_count"] == 20
    assert [row["uid"] for row in diagnostic["rows"]] == list(failure["uids"])
    assert failure["reason"] == (
        "declared batch cohort was not wholly admitted for speculative decoding"
    )
    snapshot = json.dumps(failure, sort_keys=True)
    free[0] = 100.0
    assert callback.atomic(rows()) == {uid: 2 for uid in range(4)}
    assert json.dumps(failure, sort_keys=True) == snapshot


def test_bound_violation_retains_controller_approval_and_scheduler_refusal():
    batch = _atomic_joining_batch({})
    batch.mtp_admission = _make_self_mtp_admission_callback(
        free_memory=lambda: 100.0,
        controller=SelfMTPLaneAdmissionController(saturation_lane_cap=20),
    )
    sequence = batch._unprocessed_sequences[0]
    batch._unprocessed_sequences[0] = (*sequence[:3], [_SizedCache(8, 4)], *sequence[4:])
    batch._mtp_prefill_resident = {0}
    batch._mtp_prefill_projection_bytes = {0: 1}
    assert batch._admit_mtp_joining(20) == 0
    diagnostic = batch.take_atomic_cohort_failures()[0]["admission"]
    assert diagnostic["stage"] == "full"
    assert diagnostic["bound_violation_uids"] == [0]


def test_serving_atomic_failure_preserves_diagnostic_and_safe_message(monkeypatch, caplog):
    from test_apc_hits_hybrid_gdn_self_mtp import make_adapter, tiny_qwen38_mtp

    from mlx2 import memory, serving
    from mlx2.runtime import memory_policy, os_memory

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "cpu-test"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(memory_policy, "_make_self_mtp_admission_callback", lambda *a, **kw:
        _make_self_mtp_admission_callback(free_memory=lambda: 20.0))
    model, vocab = tiny_qwen38_mtp()
    engine = serving.ServingEngine(
        "tiny", adapter_factory=make_adapter(model, vocab), qualification_mode=True,
        mtp=True, max_lanes=4, prefill_step=16,
    )
    try:
        assert engine.ready.wait(10), engine.error
        jobs = [engine.submit({
            "tokens": [1, 2, 3, 4], "max_tokens": 4, "temperature": 0,
            "batch_cohort": {"id": "diagnostic-cpu", "size": 4},
        }) for _ in range(4)]
        for job in jobs:
            while True:
                event = job.events.get(timeout=10)
                if "error" in event:
                    break
            assert event["error"] == (
                "declared batch cohort was not wholly admitted for speculative decoding"
            )
            assert event["status"] == 429
            assert event["code"] == "batch_cohort_admission_failed"
            diagnostic = event["mlx2"]["admission_failure"]
            assert diagnostic["status"] == 429
            assert diagnostic["request_id"] == job.id
            assert diagnostic["uid"] in {row["uid"] for row in diagnostic["rows"]}
            assert diagnostic["row_count"] == 4
            assert diagnostic["final_free_gib"] == 20.0
        assert "Atomic cohort admission refused:" in caplog.text
    finally:
        engine.close()


@pytest.mark.parametrize("stream", [False, True])
def test_http_error_keeps_safe_message_and_request_bound_diagnostics(stream):
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError

    from test_anthropic_compat import AnthropicEngine, _post

    from mlx2.server import handler_for
    from mlx2.serving import Job

    safe_message = "declared batch cohort was not wholly admitted for speculative decoding"
    callback = _make_self_mtp_admission_callback(free_memory=lambda: 20.0)
    receipt = callback.atomic(rows()).diagnostics
    engine = AnthropicEngine()

    def submit(request, *, tenant_id="default"):
        job = Job(request)
        job.tenant_id = tenant_id
        # Open the HTTP stream before the refusal, as sliced prefill does.
        if stream:
            job.events.put({"delta": {"content": ""}})
        job.events.put({
            "error": safe_message, "status": 429,
            "code": "batch_cohort_admission_failed",
            "mlx2": {"admission_failure": {**receipt, "status": 429}},
        })
        return job

    engine.submit = submit
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        body = {"model": "fixture", "messages": [{"role": "user", "content": "hi"}],
                "stream": stream}
        if stream:
            with _post(base, "/v1/chat/completions", body) as response:
                assert response.status == 200
                events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                          if line.startswith("data: ") and line[6:] != "[DONE]"]
            error = next(event for event in events if "error" in event)
        else:
            with pytest.raises(HTTPError) as caught:
                _post(base, "/v1/chat/completions", body)
            assert caught.value.code == 429
            error = json.load(caught.value)
        assert error["error"]["message"] == safe_message
        assert error["error"]["code"] == "batch_cohort_admission_failed"
        assert error["mlx2"]["admission_failure"] == {**receipt, "status": 429}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
