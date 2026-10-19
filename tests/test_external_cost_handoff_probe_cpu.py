"""The external cost-per-commit ordinary probe must not pay for draft work.

L3-2 (sweep 2026-10-12): the ordinary baseline was a forced depth-0
speculative round, which also ran a draft-context append (a draft forward)
that ``_ordinary_round`` -- the route a handed-off lane actually runs --
never performs.  The probe now defers that append to the next speculative
round, keeping the draft plane paired through ``lane.tail``.
"""
import mlx.core as mx

mx.set_default_device(mx.cpu)

from test_external_dflash2_cpu import drain, generator, tiny

from mlx2.runtime import external_speculative as es

PROMPT = [1, 2, 3, 4, 5]
SETTINGS = {
    "enabled": True,
    "ordinary_samples": 2,
    "speculative_samples": 2,
    # A 100x speculative loss is needed to hand off: the lane keeps
    # speculating, so the deferred draft context must be consumed later.
    "loss_ratio": 0.01,
    "loss_decisions": 1,
    "window": 4,
}
_APPEND = es.ExternalDraftBatchGenerator._append_context
_ROUND = es.ExternalDraftBatchGenerator._round


def _run(monkeypatch, max_tokens=24, **kwargs):
    model, draft = tiny()
    seen = []
    kind = {"now": None}
    def append(self, lane, token):
        seen.append(kind["now"])
        return _APPEND(self, lane, token)

    def tagged(self, cohort, *, forced_depth=None):
        kind["now"] = "ordinary_probe" if forced_depth == 0 else "other"
        return _ROUND(self, cohort, forced_depth=forced_depth)

    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_append_context", append)
    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_round", tagged)
    engine = generator(model, draft, **kwargs)
    uid = engine.insert(
        [list(PROMPT)], max_tokens=[max_tokens], sampling_configs=[{"sampling_temp": 0.0}]
    )[0]
    outputs, finishes = drain(engine)
    return outputs[uid], finishes[uid], seen


def test_ordinary_probe_skips_draft_append_and_preserves_output(monkeypatch):
    baseline, _final, _seen = _run(monkeypatch)
    tokens, final, seen = _run(monkeypatch, cost_per_commit_handoff=SETTINGS)
    receipt = final.speculative_receipt["cost_per_commit_handoff"]
    # The probe rounds ran and were booked as ordinary samples ...
    assert receipt["ordinary"]["1"]["samples"] == SETTINGS["ordinary_samples"]
    assert not receipt["handoff_triggered"]
    # ... without any draft-context append (draft forward) inside them.
    assert seen.count("ordinary_probe") == 0, (
        "ordinary baseline samples include draft-forward work the ordinary "
        "route never performs"
    )
    # Speculation resumed on a paired draft plane and the output is exact.
    assert final.speculative_receipt["external_rounds"] > 0
    assert tokens == baseline
    final.cache_sidecar.validate("test", len(final.all_tokens))


def test_explicit_ordinary_route_bypasses_cost_probe_and_uses_direct_pipeline():
    model, draft = tiny()
    engine = generator(model, draft, cost_per_commit_handoff=SETTINGS)
    uid = engine.insert(
        [list(PROMPT)],
        max_tokens=[8],
        sampling_configs=[{"sampling_temp": 0.0, "emit_logprobs": False}],
    )[0]
    engine.disable_speculation(uid)
    outputs, final = drain(engine)
    assert len(outputs[uid]) == 8
    assert final[uid].finish_reason == "length"
    stats = engine.scheduler_stats
    assert stats["ordinary_rounds"] == 8
    assert stats["external_ordinary_fast_path_rounds"] == 8
    assert stats["segmented_transactions"] == 0
    assert stats["external_rounds"] == 0


def test_request_finishing_inside_the_probe_publishes_a_paired_sidecar(monkeypatch):
    baseline, _final, _seen = _run(monkeypatch, max_tokens=2)
    tokens, final, seen = _run(
        monkeypatch,
        max_tokens=2,
        cost_per_commit_handoff={**SETTINGS, "ordinary_samples": 4},
    )
    receipt = final.speculative_receipt["cost_per_commit_handoff"]
    assert receipt["ordinary"]["1"]["samples"] >= 1
    assert not receipt["speculative"]
    assert seen.count("ordinary_probe") == 0
    assert tokens == baseline
    # The deferred context rides in the published tail, still paired.
    state = final.cache_sidecar
    state.validate("test", len(final.all_tokens))
    assert int(state.state[1].shape[1]) > 1


def test_opt_in_round_timing_survives_a_short_request(monkeypatch):
    monkeypatch.setenv("MLX2_EXTERNAL_ROUND_TIMING", "1")
    _tokens, final, _seen = _run(monkeypatch, max_tokens=8)
    timing = final.speculative_receipt["round_timing"]
    assert timing["device_synchronized"] is False
    assert timing["device_synchronizations"] == 0
    assert timing["scope"] == "executor_shared_since_request_admission"
    assert timing["request_exclusive"] is False
    assert timing["performance_claim"] is False
    assert timing["phase_total_nanoseconds"] > 0
    assert timing["phase_nanoseconds"]["draft"] > 0
    assert timing["phase_nanoseconds"]["emit"] > 0


def test_opt_in_synchronized_round_timing_is_labelled(monkeypatch):
    monkeypatch.setenv("MLX2_EXTERNAL_ROUND_TIMING", "1")
    monkeypatch.setenv("MLX2_EXTERNAL_ROUND_TIMING_SYNC", "1")
    _tokens, final, _seen = _run(monkeypatch, max_tokens=8)
    timing = final.speculative_receipt["round_timing"]
    assert timing["device_synchronized"] is True
    assert timing["device_synchronizations"] > 0
    assert timing["performance_claim"] is False
