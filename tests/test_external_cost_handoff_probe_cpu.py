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
