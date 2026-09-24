"""Qualification is confidence, not permission to run (AGENTS.md).

With neither a qualification receipt nor qualification mode, the engine still
loads and serves, and labels itself "unqualified" everywhere a client or an
operator can see it.  Test-only extras and approximate operations stay gated.
"""

import pytest

from mlx2.serving import ServingEngine


def _engine(monkeypatch, **kw):
    from route_harness import make_adapter, patch_host, tiny_qwen38_mtp

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    engine = ServingEngine(
        "tiny", adapter_factory=make_adapter(model, vocab),
        max_lanes=1, max_inflight=32, prefill_step=16, **kw,
    )
    assert engine.ready.wait(120), engine.error
    assert engine.error is None, engine.error
    return engine


@pytest.mark.parametrize("mtp", [False, True])
def test_an_unqualified_route_loads_serves_and_says_so(monkeypatch, mtp):
    from route_harness import collect

    engine = _engine(monkeypatch, mtp=mtp)
    try:
        status = engine.status()
        assert status["state"] == "ready"
        assert status["qualification"] == "unqualified"
        assert status["route_receipt"] == "unqualified"
        assert status["qualified_capabilities"] == []
        assert status["selected_capabilities"]
        output = collect(engine.submit({"tokens": [3, 5, 7], "max_tokens": 4}), timeout=60)
        assert output.get("error") is None
        assert len(output["tokens"]) == 4
        assert output["receipt"]["qualification"] == "unqualified"
        assert output["receipt"]["route_receipt"] == "unqualified"
    finally:
        engine.close()


def test_mtp_ordinary_handoff_runs_unqualified(monkeypatch):
    # Handoff is on by default for MTP models and exact, so it is not a
    # reason to refuse an unqualified model.
    engine = _engine(
        monkeypatch, mtp=True,
        execution_policy={"mtp_ordinary_handoff": {"enabled": True, "max_mtp_width": 4}},
    )
    try:
        assert engine.status()["qualification"] == "unqualified"
    finally:
        engine.close()


def test_candidate_mode_still_reports_candidate(monkeypatch):
    engine = _engine(monkeypatch, mtp=False, qualification_mode=True)
    try:
        assert engine.status()["qualification"] == "candidate"
        assert engine.status()["route_receipt"] == "candidate_validation"
    finally:
        engine.close()


def test_test_only_extras_stay_with_qualification_mode():
    with pytest.raises(ValueError, match="qualification mode"):
        ServingEngine("unused", mtp=True, mtp_acceptance_log="/dev/null")
