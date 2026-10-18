"""Reversible per-session cost park on the external executor.

A parked session runs target-only depth-0 rounds that defer the draft append
(the draft plane stays paired through ``lane.tail``), periodically re-probes
speculation, and re-engages when speculation wins again.  Output must stay
identical to the unparked route at every step.
"""
import mlx.core as mx

mx.set_default_device(mx.cpu)

from test_external_dflash2_cpu import drain, generator, tiny

from mlx2.runtime import external_speculative as es

PROMPT = [1, 2, 3, 4, 5]
_APPEND = es.ExternalDraftBatchGenerator._append_context
_ROUND = es.ExternalDraftBatchGenerator._round


def _settings(**overrides):
    return {
        "enabled": True,
        "reversible": True,
        "ordinary_samples": 2,
        "speculative_samples": 2,
        "loss_decisions": 1,
        "window": 4,
        "reprobe_tokens": 3,
        "reprobe_max_tokens": 6,
        **overrides,
    }


def _run(monkeypatch, *, speculative_ns, max_tokens=40, handoff=None):
    """``speculative_ns(round_index)`` prices a speculative round; an
    ordinary (depth-0) round always costs 100 ns on a synthetic clock."""
    model, draft = tiny()
    clock = {"now": 1, "spec_rounds": 0, "kind": None}
    appends = []

    def fake_ns():
        return clock["now"]

    def append(self, lane, token):
        appends.append(clock["kind"])
        return _APPEND(self, lane, token)

    def timed(self, cohort, *, forced_depth=None):
        parked = all(es._speculation_parked(lane) for lane in cohort)
        kind = "ordinary" if forced_depth == 0 or parked else "speculative"
        clock["kind"] = "parked_forced" if forced_depth == 0 and parked else kind
        result = _ROUND(self, cohort, forced_depth=forced_depth)
        if kind == "ordinary":
            clock["now"] += 100
        else:
            clock["now"] += speculative_ns(clock["spec_rounds"])
            clock["spec_rounds"] += 1
        return result

    monkeypatch.setattr(es.time, "perf_counter_ns", fake_ns)
    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_append_context", append)
    monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_round", timed)
    kwargs = {} if handoff is None else {"cost_per_commit_handoff": handoff}
    engine = generator(model, draft, **kwargs)
    uid = engine.insert(
        [list(PROMPT)], max_tokens=[max_tokens], sampling_configs=[{"sampling_temp": 0.0}]
    )[0]
    outputs, finishes = drain(engine)
    return outputs[uid], finishes[uid], appends, engine


def test_losing_session_parks_reprobes_and_keeps_exact_output(monkeypatch):
    baseline, _, _, _ = _run(monkeypatch, speculative_ns=lambda _i: 100_000)
    tokens, final, appends, engine = _run(
        monkeypatch, speculative_ns=lambda _i: 100_000, handoff=_settings()
    )
    assert tokens == baseline
    receipt = final.speculative_receipt["cost_per_commit_handoff"]
    reversible = receipt["reversible"]
    assert receipt["one_way"] is False and not receipt["handoff_triggered"]
    assert reversible["parks"] >= 1
    assert reversible["reprobes"] >= 1 and reversible["failed_reprobes"] >= 1
    assert reversible["reengagements"] == 0
    # The executor never made the lane permanently ordinary.
    assert final.speculative_receipt["ordinary_fallback"] is False
    assert engine.scheduler_stats.get("external_cost_parked_rounds", 0) > 0
    # Parked target-only rounds performed no draft-context append.
    assert "parked_forced" not in appends
    final.cache_sidecar.validate("test", len(final.all_tokens))


def test_parked_session_reengages_when_speculation_wins_again(monkeypatch):
    # Speculation loses for its first rounds, then becomes cheap.
    price = lambda i: 100_000 if i < 2 else 10
    baseline, _, _, _ = _run(monkeypatch, speculative_ns=price)
    tokens, final, _appends, _engine = _run(
        monkeypatch, speculative_ns=price, handoff=_settings()
    )
    assert tokens == baseline
    reversible = final.speculative_receipt["cost_per_commit_handoff"]["reversible"]
    assert reversible["parks"] == 1
    assert reversible["reengagements"] == 1
    assert reversible["state"] == "speculating"
    events = [event["event"] for event in reversible["events"]]
    assert events[:3] == ["park", "reprobe", "reengage"]
    final.cache_sidecar.validate("test", len(final.all_tokens))


def test_deferred_context_cap_flushes_without_breaking_pairing(monkeypatch):
    baseline, _, _, _ = _run(monkeypatch, speculative_ns=lambda _i: 100_000)
    tokens, final, _appends, engine = _run(
        monkeypatch,
        speculative_ns=lambda _i: 100_000,
        handoff=_settings(deferred_context_cap=2, reprobe_tokens=6, reprobe_max_tokens=6),
    )
    assert tokens == baseline
    assert engine.scheduler_stats.get("external_cost_parked_flush_rounds", 0) > 0
    final.cache_sidecar.validate("test", len(final.all_tokens))
