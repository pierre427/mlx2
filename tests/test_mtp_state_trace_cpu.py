"""Bounded readback and native-MTP/APCv2 contracts; conftest selects CPU."""

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen4_mtp

from mlx2.runtime import hybrid_speculative as H
from mlx2.runtime import mtp_state_trace as T
from mlx2.runtime.apc_v2 import APCKey, APCv2, MTPAPCSidecar


def test_disabled_trace_never_inspects_state():
    class Poison:
        @property
        def state(self):
            raise AssertionError("disabled diagnostics touched a cache")

    T.state("off", target=Poison())
    T.sidecar_state("off", Poison(), Poison())
    T.round_state("off", Poison(), Poison(), Poison(), Poison())


def test_bounds_and_raw_bfloat_digest():
    trace = T.MTPStateTrace(max_events=2, max_leaves=2, max_bytes_per_event=8)
    small = mx.array([1, 2], dtype=mx.bfloat16)
    with T.activate(trace):
        T.state("one", target=[small, mx.ones((1024,))])
        T.state("two", target=[small, small, small])
        T.state("dropped", target=object())
    receipt = trace.receipt()
    assert len(receipt["events"]) == 2 and receipt["dropped_events"] == 1
    assert all(event["readback_bytes"] <= 8 for event in receipt["events"])
    assert all(len(event["tensors"]) <= 2 for event in receipt["events"])
    assert receipt["events"][0]["tensors"][0]["sha256"]
    assert receipt["events"][0]["tensors"][1]["omitted"] == "readback_budget"
    assert all(event["complete"] is False for event in receipt["events"])
    json.dumps(receipt, allow_nan=False)
    receipt["events"].clear()
    assert len(trace.events) == 2


def test_scope_restores_after_failure_and_errors_do_not_escape():
    class BadCache:
        @property
        def state(self):
            raise RuntimeError("secret must not appear")

    trace = T.MTPStateTrace()
    with pytest.raises(ValueError), T.activate(trace):
        T.state("bad", target=BadCache())
        raise ValueError()
    T.state("off", target=BadCache())
    assert len(trace.events) == 1
    assert trace.events[0]["diagnostic_error"] == "RuntimeError"
    assert "secret" not in json.dumps(trace.receipt())


def _prepare(model, prompt, *, cache=None, sidecar=None, boundary=None):
    return H.prepare_self_mtp_lane(
        mx.array(prompt, mx.uint32),
        model,
        uid=7,
        max_tokens=32,
        prompt_cache=cache,
        mtp_state=sidecar,
        lane_rng=None,
        num_draft=2,
        sampling_temp=0,
        sampling_top_p=1,
        sampling_top_k=0,
        sampling_min_p=0,
        accept_rule="residual",
        logits_processors=[],
        prefill_step_size=3,
        share_index_state=False,
        prompt_boundary_out=boundary,
    )


def _rounds(model, detached, first):
    batch = H.attach_self_mtp_lanes(model, None, [detached])
    output = [first.token]
    for _ in range(3):
        proposal = H.propose_batched_self_mtp(model, batch)
        output.extend(token.token for token in proposal.outputs[0])
        H.commit_batched_self_mtp(
            batch, proposal, emitted_counts=[len(proposal.outputs[0])], terminal=[False]
        )
    _, rows = H.detach_self_mtp_lanes(model, batch, [0])
    return output, rows[0]


def test_cpu_cold_restore_and_first_rollback_are_observational():
    model, _ = tiny_qwen4_mtp()
    prompt = [3, 4, 5, 6, 7, 8, 9]
    baseline, _ = _rounds(model, *_prepare(model, prompt))
    trace = T.MTPStateTrace(token_index=2)
    boundary = {}
    apc = APCv2(max_size=2, layout_name="mtp-state-trace-cpu")
    with T.activate(trace):
        cold, _ = _rounds(model, *_prepare(model, prompt, boundary=boundary))
        apc.store(
            APCKey("cpu"),
            prompt[:-1],
            boundary["target_cache"],
            sidecar=MTPAPCSidecar(boundary["mtp_state"], len(prompt) - 1),
        )
        hit = apc.lookup(APCKey("cpu"), prompt, require_sidecar=True)
        assert hit.hit and hit.cached_tokens == len(prompt) - 1
        warm, _ = _rounds(
            model,
            *_prepare(
                model, hit.remaining_tokens, cache=hit.cache, sidecar=hit.sidecar.state
            ),
        )
    assert baseline == cold == warm
    stages = {event["stage"] for event in trace.events}
    assert {
        "boundary_capture_source",
        "descriptor_capture",
        "boundary_capture_saved",
        "apc_sidecar_restore",
        "mtp_restore_validated",
        "prepare_catchup",
        "lane_ready",
        "verify_before_rollback",
        "verify_after_rollback",
        "selected_token_logprobs",
        "verify_ids",
        "commit_complete",
    } <= stages
    assert all("diagnostic_error" not in event for event in trace.events)
    source = next(e for e in trace.events if e["stage"] == "boundary_capture_source")
    saved = next(e for e in trace.events if e["stage"] == "boundary_capture_saved")
    assert source["tensors"] == saved["tensors"]
    assert source["offsets"] == saved["offsets"]
    top = [e for e in trace.events if e["stage"] == "selected_token_logprobs"]
    assert len(top) == 2 and all(e["token_index"] == 2 for e in top)
    assert all(e["selected"] == cold[1] for e in top)
    assert all(e["complete"] for e in trace.events if "complete" in e)
    json.dumps(trace.receipt(), allow_nan=False)
    if callable(getattr(hit.cache, "close", None)):
        hit.cache.close()
    apc.clear(release_memory=False)


@pytest.mark.parametrize(
    "kw",
    [
        {"max_events": 0},
        {"max_events": 257},
        {"max_leaves": 513},
        {"max_bytes_per_event": 2**30},
        {"token_index": 0},
    ],
)
def test_invalid_limits_rejected(kw):
    with pytest.raises(ValueError):
        T.MTPStateTrace(**kw)


def test_handoff_observer_preserves_detached_state():
    from mlx2.runtime.generate import MTPGenerationBatch

    model, _ = tiny_qwen4_mtp()
    detached, first = _prepare(model, [3, 4, 5, 6])
    owner = SimpleNamespace(
        ordinary_handoff_policy=SimpleNamespace(as_dict=lambda: {"enabled": True}),
        _observed_widths_by_uid={7: {1}},
        _async_qsa_receipts_by_uid={},
    )
    package = SimpleNamespace(detached=detached, num_tokens=0, initial_output=first)
    trace = T.MTPStateTrace()
    with T.activate(trace):
        T.state(
            "before",
            target=detached.caches.target,
            draft=detached.caches.draft,
            seed=detached.lane.seed_h,
        )
        MTPGenerationBatch._mark_plain_handoff(
            owner, [package], decision={"reason": "diagnostic_test"}, projected_width=1
        )
        T.state(
            "after",
            target=detached.caches.target,
            draft=detached.caches.draft,
            seed=detached.lane.seed_h,
        )
    assert [e["stage"] for e in trace.events] == ["before", "ordinary_handoff", "after"]
    assert (
        trace.events[0]["tensors"]
        == trace.events[1]["tensors"]
        == trace.events[2]["tensors"]
    )
    assert package.handoff_receipt["engaged"] is True
    assert package.initial_output.token == first.token


def test_only_selected_position_reads_top_two(monkeypatch):
    trace = T.MTPStateTrace(token_index=87)
    values = mx.array([[-3.0, -0.1, -0.2], [-0.2, -3.0, -0.1]])
    calls = []
    partition = mx.argpartition

    def counted(*args, **kwargs):
        calls.append(True)
        return partition(*args, **kwargs)

    monkeypatch.setattr(mx, "argpartition", counted)
    for ntoks in (1, 85, 88):
        trace.verification(
            lane=SimpleNamespace(uid=4, ntoks=ntoks),
            drafts=[1],
            accepted=1,
            logprobs=values,
            bonus=2,
        )
    assert len(calls) == 1
    event = trace.events[0]
    assert event["token_index"] == 87 and event["selected"] == 2
    assert [p["token"] for p in event["top_two"]] == [2, 0]
    assert event["margin"] == pytest.approx(0.1)
    T.MTPStateTrace().verification(
        lane=SimpleNamespace(uid=4, ntoks=85),
        drafts=[1],
        accepted=1,
        logprobs=values,
        bonus=2,
    )
    assert len(calls) == 1


def test_sidecar_error_is_redacted():
    class BadSidecar:
        @property
        def state(self):
            raise ValueError("secret sidecar")

    trace = T.MTPStateTrace()
    with T.activate(trace):
        T.sidecar_state("restore", [], BadSidecar())
    assert trace.events == [{"stage": "restore", "diagnostic_error": "ValueError"}]
