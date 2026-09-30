"""CPU tests for scripts/qualify_copy_draft_threshold.py.

Boundary tests drive the real ``CopyDraftState`` (host-only) at agreements
15/16/31/32; gate falsifiers use synthetic run records; the smoke runs the
deterministic tiny Flash-Next-class model on CPU. Real cells need
--i-own-the-gpu and belong to the GPU owner.
"""

import copy
import json
import subprocess
import sys
import types

import pytest

from scripts import qualify_copy_draft_threshold as Q


def _args(*extra):
    return Q.resolve_args(Q.build_parser(), ["--tiny", "--out", "/dev/null", *extra])


def test_import_is_mlx_free():
    probe = subprocess.run(
        [sys.executable, "-c", "import sys, scripts.qualify_copy_draft_threshold; print('mlx.core' in sys.modules)"],
        capture_output=True, text=True, env={"PYTHONPATH": "src:."})
    assert probe.returncode == 0 and probe.stdout.strip() == "False", probe.stderr


def test_arms_differ_in_strong_match_only_and_baseline_is_the_adapter_default():
    assert Q.adapter_default_copy_policy() == Q.BASELINE == Q.ARMS["s32"]
    diff = {k for k in Q.ARMS["s32"] if Q.ARMS["s32"][k] != Q.ARMS["s16"][k]}
    assert diff == {"strong_match"} and Q.ARMS["s16"]["strong_match"] == 16
    assert Q.UPSTREAM["file_blob"] == "219be71ca6899e26033be7f6dec87ad386cce228"


# ---------------------------------------------------------------- 15/16/31/32 boundary

def _state(arm, agreement, width):
    """A real CopyDraftState whose best source agrees for exactly ``agreement`` tokens."""
    from mlx2.runtime.copy_draft import CopyDraftPolicy, CopyDraftState

    quote = list(range(100, 100 + agreement))        # the quoted span, distinct tokens
    tail = list(range(300, 320))                     # what followed it the first time
    context = [7] + quote + tail + [9] + quote       # 7 != 9 stops the agreement at exactly N
    state = CopyDraftState(CopyDraftPolicy(**Q.ARMS[arm]), context)
    state.width = width
    return state, tail


@pytest.mark.parametrize("agreement,s32_span,s16_span", [(15, 7, 7), (16, 7, 16), (31, 7, 16), (32, 16, 16)])
def test_threshold_boundaries_through_the_real_plan(agreement, s32_span, s16_span):
    for arm, want in (("s32", s32_span), ("s16", s16_span)):
        state, tail = _state(arm, agreement, width=16)
        assert state._find()[1] == agreement
        with Q.PlanProbe() as probe:
            span, decision = state.plan(head_depth=2, cap=16)
        assert decision == "copy" and span == tail[:want]
        plans = Q.summarize_plans(probe.events)
        in_band = Q.BAND[0] <= agreement <= Q.BAND[1]
        assert plans["band_rounds"] == plans["differential_rounds"] == int(in_band)
        assert plans["differential_wide_spans"] == int(in_band and arm == "s16")
        assert Q.engagement_problems(arm, plans) == []


def test_a_narrow_sizer_width_is_not_differential():
    for arm in Q.ARMS:
        state, tail = _state(arm, 20, width=7)
        with Q.PlanProbe() as probe:
            span, _ = state.plan(head_depth=2, cap=16)
        assert span == tail[:7]
        plans = Q.summarize_plans(probe.events)
        assert plans["band_rounds"] == 1 and plans["differential_rounds"] == 0


def test_probe_restores_plan_even_on_error():
    from mlx2.runtime.copy_draft import CopyDraftState

    original = CopyDraftState.plan
    with pytest.raises(RuntimeError):
        with Q.PlanProbe():
            assert CopyDraftState.plan is not original
            raise RuntimeError("boom")
    assert CopyDraftState.plan is original


def test_engagement_invariants_catch_an_unbound_threshold():
    base = Q.summarize_plans([{"agreement": 20, "width": 16, "cap": 16, "decision": "copy", "span": 16,
                               "max_span": 7}])
    assert Q.engagement_problems("s32", base)[0].startswith("s32: copied more than max_span")
    cand = Q.summarize_plans([{"agreement": 20, "width": 16, "cap": 16, "decision": "copy", "span": 7,
                               "max_span": 7}])
    assert Q.engagement_problems("s16", cand)[0].startswith("s16: a differential round did not copy")


# ---------------------------------------------------------------- gate falsifiers

def _run(arm, *, differential=0, band=None, copy_rounds=3):
    band = differential if band is None else band
    run = {"arm": arm, "tokens": [1, 2, 3], "logprob_rows": ["a", "b", "c"], "finish_reason": "length",
           "failures": [], "verify_cap": 17, "decode_s": 1.0,
           "memory": {"active_bytes": 1, "cache_bytes": 0, "peak_bytes": 2},
           "target_state": {"status": "complete", "sha256": "t"}, "draft_state": {"status": "complete", "sha256": "d"},
           "continuation": {"status": "complete", "tokens": [4, 5],
                            "final_state": {"status": "complete", "sha256": "c"}}}
    if arm == "ordinary":
        return run
    run.update(held_copy_policy=dict(Q.ARMS[arm]), route="segmented_self_mtp",
               copy_draft={"enabled": True, "policy": {k: v for k, v in Q.ARMS[arm].items() if k != "enabled"},
                           "copy_rounds": copy_rounds},
               plans={"band_rounds": band, "differential_rounds": differential,
                      "differential_wide_spans": differential if arm == "s16" else 0,
                      "band_spans_over_max_span": differential if arm == "s16" else 0})
    return run


def _cells(pairs=2):
    def cell(cls):
        return {"reference": _run("ordinary"),
                "pairs": [{"s32": _run("s32", differential=2 if cls == "copy" else 0,
                                       copy_rounds=3 if cls == "copy" else 0),
                           "s16": _run("s16", differential=2 if cls == "copy" else 0,
                                       copy_rounds=3 if cls == "copy" else 0)} for _ in range(pairs)]}
    return {"copy": [cell("copy")], "control": [cell("control")]}


EVAL_ARGS = types.SimpleNamespace(gen=3, logprob_rows=3, continuation_tokens=2)


def _evaluate(edit=None):
    cells = _cells()
    if edit:
        edit(cells)
    return Q.evaluate(cells, EVAL_ARGS)


def test_complete_engaged_synthetic_cells_pass():
    result = _evaluate()
    assert result["verdict"] == "pass", result
    assert result["differential_rounds_copy"] == 4
    assert result["latency"]["copy0"]["median"] == 1.0


def _copy_pair(k, arm):
    return lambda c: c["copy"][0]["pairs"][k][arm]


@pytest.mark.parametrize("edit,reason", [
    (lambda c: [p["s16"]["plans"].update(differential_rounds=0, differential_wide_spans=0)
                for p in c["copy"][0]["pairs"]], "s16: no strong-threshold differential round"),
    (lambda c: c["control"][0]["pairs"][0]["s32"]["plans"].update(band_rounds=1),
     "control0 pair 0 s32: negative control reached agreement 16..31"),
    (lambda c: c["copy"][0]["pairs"][1]["s32"]["copy_draft"].update(copy_rounds=0),
     "copy0 pair 1 s32: copy prompt never copied"),
    (lambda c: c["copy"][0]["pairs"][0]["s16"].update(copy_draft=None), "copy0 pair 0 s16: no copy-draft receipt"),
    (lambda c: c["copy"][0]["pairs"][0]["s16"]["held_copy_policy"].update(strong_match=32),
     "copy0 pair 0 s16: generator copy-draft policy differs"),
    (lambda c: c["copy"][0]["pairs"][0]["s16"]["copy_draft"]["policy"].update(strong_match=32),
     "copy0 pair 0 s16: copy-draft receipt policy differs"),
    (lambda c: c["copy"][0]["pairs"][0]["s32"].update(verify_cap=8), "copy0 pair 0 s32: fused GDN verify cap 8"),
    (lambda c: c["copy"][0]["pairs"][0]["s32"]["plans"].update(band_spans_over_max_span=1),
     "copy0 pair 0 s32: s32: copied more than max_span"),
    (lambda c: c["copy"][0]["pairs"][0]["s16"]["plans"].update(differential_wide_spans=1),
     "copy0 pair 0 s16: s16: a differential round did not copy"),
    (lambda c: c["copy"][0]["pairs"][0]["s16"].update(finish_reason=None), "copy0 pair 0 s16: early stop"),
    (lambda c: c["copy"][0]["pairs"][0]["s16"]["failures"].append("bounded: stopped"),
     "copy0 pair 0 s16: bounded"),
    (lambda c: c["copy"][0]["pairs"][0]["s16"].update(route=None), "copy0 pair 0 s16: no self-MTP route receipt"),
    (lambda c: c["copy"][0]["reference"].update(finish_reason=None), "copy0 ordinary: early stop"),
    (lambda c: c.pop("control"), "no control prompts ran"),
    (lambda c: c["copy"][0].update(pairs=[]), "copy0: no timed pairs"),
])
def test_missing_or_unbound_evidence_is_refused(edit, reason):
    result = _evaluate(edit)
    assert result["verdict"] == "refused"
    assert any(r.startswith(reason) for r in result["refusals"]), result["refusals"]


@pytest.mark.parametrize("edit,difference", [
    (lambda c: c["copy"][0]["pairs"][0]["s16"].update(tokens=[1, 2, 4]), "copy0 pair 0 s32 vs s16: tokens differ"),
    (lambda c: c["copy"][0]["pairs"][0]["s16"].update(logprob_rows=["a", "x", "c"]),
     "copy0 pair 0 s32 vs s16: logprob row bits differ"),
    (lambda c: c["copy"][0]["pairs"][0]["s16"].update(logprob_rows=["a", "b", "late"]),   # late drift
     "copy0 pair 0 s32 vs s16: logprob row bits differ"),
    (lambda c: c["copy"][0]["pairs"][0]["s16"]["draft_state"].update(sha256="x"),
     "copy0 pair 0 s32 vs s16: draft_state digest differs"),
    (lambda c: c["control"][0]["pairs"][1]["s16"]["target_state"].update(sha256="x"),
     "control0 pair 1 s32 vs s16: target_state digest differs"),
    (lambda c: c["copy"][0]["pairs"][0]["s16"]["continuation"].update(tokens=[4, 6]),
     "copy0 pair 0 s32 vs s16: continuation differs"),
    (lambda c: [p["s32"].update(tokens=[1, 2, 9]) for p in c["copy"][0]["pairs"][1:]],
     "copy0 s32 pair 0 vs 1: tokens differ"),
])
def test_bit_differences_are_counterexamples(edit, difference):
    result = _evaluate(edit)
    assert result["verdict"] == "counterexample"
    assert difference in result["differences"], result["differences"]


@pytest.mark.parametrize("edit,item", [
    (lambda c: c["copy"][0]["pairs"][0]["s16"]["logprob_rows"].__setitem__(1, None),
     "copy0 pair 0 s32 vs s16: logprob rows unavailable"),
    (lambda c: c["copy"][0]["pairs"][0]["s16"].update(target_state={"status": "unavailable", "sha256": None}),
     "copy0 pair 0 s32 vs s16: target_state complete/unavailable"),
    (lambda c: c["copy"][0]["pairs"][0]["s32"].update(continuation={"status": "unavailable"}),
     "copy0 pair 0 s32 vs s16: continuation unavailable"),
    # complete status wrapping an unavailable nested digest
    (lambda c: [p["s16"]["continuation"].update(final_state={"status": "unavailable", "sha256": None})
                for p in c["copy"][0]["pairs"]],
     "copy0 pair 0 s32 vs s16: continuation final state digest is not complete"),
    (lambda c: [p[a]["continuation"].update(final_state={"status": "complete", "sha256": None})
                for p in c["copy"][0]["pairs"] for a in p],
     "copy0 pair 0 s32 vs s16: continuation final state digest is not complete"),
    (lambda c: [p[a]["continuation"].update(tokens=[4]) for p in c["copy"][0]["pairs"] for a in p],
     "copy0 pair 0 s32 vs s16: continuation has 1 of 2 tokens"),
    # truncated logprob coverage in both runs, identical where sampled
    (lambda c: [p[a].update(logprob_rows=["a", "b"]) for p in c["copy"][0]["pairs"] for a in p],
     "copy0 pair 0 s32 vs s16: logprob rows cover 2 of 3 emitted tokens"),
    (lambda c: c["copy"][0]["pairs"][0]["s16"].pop("memory"), "copy0 pair 0 s32 vs s16: memory not captured (b)"),
    (lambda c: c["copy"][0]["pairs"][0]["s32"]["memory"].update(peak_bytes=None),
     "copy0 pair 0 s32 vs s16: memory not captured (a)"),
])
def test_missing_bits_are_never_equal(edit, item):
    result = _evaluate(edit)
    assert result["verdict"] == "exact_with_unavailable_parts"
    assert item in result["incomparable"]


def test_a_sampled_prefix_never_hides_late_drift():
    """16 sampled rows of a longer run: equal prefix, drift only beyond the
    sample. The pre-repair gate passed this; it must not pass now."""
    cells = _cells()
    for cls in cells.values():
        cls[0]["reference"]["tokens"] = list(range(32))
        for pair in cls[0]["pairs"]:
            for run in pair.values():
                run["tokens"] = list(range(32))
                run["logprob_rows"] = [f"r{i}" for i in range(16)]
    args = types.SimpleNamespace(gen=32, logprob_rows=16, continuation_tokens=2)
    result = Q.evaluate(cells, args)
    assert result["verdict"] == "exact_with_unavailable_parts"
    assert "copy0 pair 0 s32 vs s16: logprob rows cover 16 of 32 emitted tokens" in result["incomparable"]


def test_logprob_rows_default_to_every_emitted_token():
    assert _args().logprob_rows == _args().gen == 24
    assert _args("--gen", "40").logprob_rows == 40 and _args("--logprob-rows", "5").logprob_rows == 5


def test_continuation_problem_requires_count_and_complete_digest():
    good = {"status": "complete", "tokens": [1, 2], "final_state": {"status": "complete", "sha256": "x"}}
    assert Q.continuation_problem(good, 2) is None
    assert Q.continuation_problem(good, 3) == "continuation has 2 of 3 tokens"
    assert Q.continuation_problem({**good, "final_state": {"status": "metadata_unavailable", "sha256": None}}, 2)
    assert Q.continuation_problem({**good, "tokens": None}, 2) == "continuation has no of 2 tokens"
    assert Q.continuation_problem(None, 2) == "continuation unavailable"


def test_disabled_logprob_rows_are_labelled_unavailable():
    cells = _cells()
    for cls in cells.values():
        for pair in cls[0]["pairs"]:
            for run in pair.values():
                run["logprob_rows"] = []
    result = Q.evaluate(cells, types.SimpleNamespace(gen=3, logprob_rows=0, continuation_tokens=2))
    assert result["verdict"] == "exact_with_unavailable_parts"
    assert any("(disabled (--logprob-rows 0))" in i for i in result["incomparable"])


# ---------------------------------------------------------------- CLI

@pytest.mark.parametrize("argv,message", [
    (["--model", "m"], "--i-own-the-gpu"),
    (["--i-own-the-gpu"], "--model is required"),
    (["--tiny", "--i-own-the-gpu"], "random CPU model"),
    (["--tiny", "--pairs", "0"], "--pairs 1..8"),
    (["--tiny", "--pairs", "9"], "--pairs 1..8"),
    (["--tiny", "--gen", "513"], "--gen and --warmup-gen 1..512"),
    (["--tiny", "--warmup-gen", "0"], "--gen and --warmup-gen 1..512"),
    (["--tiny", "--continuation-tokens", "65"], "--continuation-tokens 0..64"),
    (["--tiny", "--logprob-rows", "-1"], "--logprob-rows 0..512"),
    (["--tiny", "--time-limit-s", "nan"], "finite"),
])
def test_argument_refusals(argv, message, capsys):
    with pytest.raises(SystemExit):
        Q.resolve_args(Q.build_parser(), [*argv, "--out", "/dev/null"])
    assert message in capsys.readouterr().err


# ---------------------------------------------------------------- tiny CPU smoke

@pytest.fixture(scope="module")
def tiny():
    return Q.model_mode(_args("--pairs", "2"))[1]


def test_tiny_smoke_captures_full_evidence_and_refuses_without_engagement(tiny):
    """The random tiny model never continues its context, so copy drafts never
    fire: the cell must refuse, while every exactness artefact is present."""
    from mlx2.runtime.copy_draft import CopyDraftState
    from mlx2.runtime.models import qwen4_fused_gdn_verify as FV

    assert tiny["verdict"] == "refused" and tiny["differences"] == [] and tiny["incomparable"] == []
    assert any(r.startswith("s16: no strong-threshold differential round") for r in tiny["refusals"])
    assert all("never copied" in r or "differential" in r for r in tiny["refusals"])
    assert CopyDraftState.plan.__qualname__ == "CopyDraftState.plan"   # observer removed
    assert FV.MAX_VERIFY_STEPS == tiny["protocol"]["verify_cap_restored_to"]
    assert tiny["protocol"]["adapter_default_copy_policy"] == Q.BASELINE
    for cls in ("copy", "control"):
        cell = tiny[f"cells"][cls][0]
        assert cell["orders"] == [["s32", "s16"], ["s16", "s32"]]
        for pair in cell["pairs"]:
            for arm, run in pair.items():
                assert run["held_copy_policy"]["strong_match"] == Q.ARMS[arm]["strong_match"]
                assert run["verify_cap"] == 17 and run["plans"]["plans"] > 0
                assert run["target_state"]["status"] == run["draft_state"]["status"] == "complete"
                assert run["continuation"]["status"] == "complete" and len(run["continuation"]["tokens"]) == 4
                assert run["continuation"]["final_state"]["status"] == "complete"
                assert len(run["logprob_rows"]) == len(run["tokens"]) and None not in run["logprob_rows"]
                assert set(run["memory"]) >= {"active_bytes", "peak_bytes", "cache_bytes"}


def test_a_baseline_that_is_not_the_adapter_default_is_refused(monkeypatch):
    monkeypatch.setattr(Q, "adapter_default_copy_policy", lambda: dict(Q.BASELINE, strong_match=24))
    record = Q.model_mode(_args("--pairs", "1"))[1]
    assert record["verdict"] == "refused"
    assert record["refusals"][0].startswith("baseline is not the local Flash-Next default")


def test_main_writes_a_bound_record(tmp_path):
    out = tmp_path / "r.json"
    assert Q.main(["--tiny", "--pairs", "1", "--out", str(out)]) == 1
    data = json.loads(out.read_text())
    assert data["schema"] == Q.SCHEMA and data["upstream"]["pull"] == 616
    assert data["identity"]["files"]["src/mlx2/runtime/copy_draft.py"]
    assert data["identity"]["mlx"]["device"].startswith("Device(cpu")
    assert "TINY" in data["scope"] and "not a controlled performance run" in data["latency"]["note"]
