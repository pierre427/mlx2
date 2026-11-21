"""Bug sweep 2026-11-21, lane fix-amoeba-fn-b: regressions for AMF2-1..12 and the langgate site of AMFN-6.

CPU only: no model, no server. Each test names the finding it pins.
"""
from __future__ import annotations

import builtins
import dis
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import amoeba_langgate_lib as G  # noqa: E402


def _mod(name):
    return importlib.import_module(name)


# ----------------------------------------------------------------------------- AMF2-1

def test_export_functions_load_no_undefined_global():
    """AMF2-1: `main` used `mx` with no import (NameError after the partition tile set was written)."""
    ex = _mod("amoeba27b_export")
    known = set(vars(ex)) | set(dir(builtins))
    missing = {}
    for name, fn in vars(ex).items():
        if callable(fn) and getattr(fn, "__module__", None) == ex.__name__ and hasattr(fn, "__code__"):
            todo = [fn.__code__]
            while todo:
                code = todo.pop()
                todo += [c for c in code.co_consts if hasattr(c, "co_code")]
                bad = {i.argval for i in dis.get_instructions(code) if i.opname == "LOAD_GLOBAL"} - known
                if bad:
                    missing.setdefault(name, set()).update(bad)
    assert not missing, missing


# ----------------------------------------------------------------------------- AMF2-2 / AMF2-3 / AMFN-6

class _CountingLID:
    def __init__(self, answer=None):
        self.calls, self.answer = [], answer or {"en": 0.99}

    def probs(self, text):
        self.calls.append(text)
        return dict(self.answer)


@pytest.mark.parametrize("text", ["東京の天気は？", "猫が好き", "今日は何曜日？"])
def test_short_japanese_with_few_kana_opens_jpn(text):
    """AMF2-2: 1-3 kana beside Han was neither 'kana' nor 'Han-only': nothing opened."""
    opened, info = G.GatePolicy(G.GateConfig(), lid=_CountingLID()).detect(text)
    assert "jpn" in opened, info
    # the script-only check between full checks opens it too
    assert "jpn" in G.GatePolicy(G.GateConfig()).detect(text, full=False)[0]


def test_han_only_and_single_stray_kana_rules_unchanged():
    pol = G.GatePolicy(G.GateConfig())
    assert pol.detect("北京是中华人民共和国的首都")[0] >= {"cmn", "jpn"}        # Han-only: both, as before
    assert pol.detect("The mark の alone")[0] == set()                         # one stray kana, no Han: nothing
    assert pol.detect("猫が")[0] == set()                                      # under script_min_chars letters


def test_greek_prose_fails_open_and_greek_maths_does_not():
    """AMF2-3: Greek prose reached neither LID nor the unknown-script flag."""
    greek = "Η Αθήνα είναι η πρωτεύουσα της Ελλάδας και μια από τις αρχαιότερες πόλεις του κόσμου."
    for lid in (_CountingLID(), None):
        opened, info = G.GatePolicy(G.GateConfig(), lid=lid).detect(greek)
        assert opened >= set(G.LANG10), info
        assert any(f.startswith("unknown_script:greek=") for f in info["flags"])
    maths = "The angle θ satisfies sin(θ + φ) = α·β, where λ is the wavelength and μ the mean of the sample."
    opened, info = G.GatePolicy(G.GateConfig(), lid=_CountingLID()).detect(maths)
    assert opened == set() and info["flags"] == []
    assert G.GatePolicy(G.GateConfig(), lid=None).detect("α + β = γ, δ")[0] == set()   # symbols alone: too few letters
    off = G.GatePolicy(G.GateConfig(on_uncertain="none"), lid=None).detect(greek)[0]
    assert off == set()                                                        # the fail-open switch still governs


def test_decide_nan_open_arm_is_invalid():
    """AMFN-6 (langgate site): a NaN open-vs-full difference passed the sanity gate."""
    rows = {("open", "english"): {"mean": float("nan")}, ("auto", "english"): {"mean": 0.0}}
    v = G.decide(rows, {"needed": 10, "missed": 0}, None, {"english": "english"})
    assert v["verdict"] == "INVALID"
    rows[("open", "english")] = {"mean": 0.0}
    assert G.decide(rows, {"needed": 10, "missed": 0}, None, {"english": "english"})["verdict"] != "INVALID"


# ----------------------------------------------------------------------------- AMF2-4 / AMF2-8 / AMF2-7 (served, spec)

def _resp(text, tokens=None):
    r = {"choices": [{"message": {"content": text}}]}
    if tokens is not None:
        r["choices"][0]["logprobs"] = {"content": [
            {"token": t, "logprob": -0.1, "top_logprobs": [{"token": t, "logprob": -0.1}]} for t in tokens]}
    return r


def test_no_logprobs_is_no_evidence_not_identical(monkeypatch):
    """AMF2-4: two responses with no logprob entries counted as identical; an exact arm passed unchecked."""
    sv, sc = _mod("amoeba27b_served"), _mod("amoeba27b_spec_check")
    probes = [("Capital of France?", "Paris"), ("Capital of Peru?", "Lima"), ("Capital of Kenya?", "Nairobi")]
    monkeypatch.setattr(sv, "PROBES", probes)
    ref = [_resp("Paris"), _resp("Lima", []), _resp("Nairobi", ["N", "airobi"])]
    arm = [_resp("Madrid"), _resp("Lima", []), _resp("Nairobi", ["N", "airobi"])]
    for out in (sc.compare(ref, arm), sv.classify_pairs(ref, arm)):
        assert out["counts"]["identical"] == 1 and out["counts"]["no_evidence"] == 2
        assert [r["verdict"] for r in out["divergent"]] == ["no_evidence", "no_evidence"]
    cmp_ = sc.compare(ref, arm)
    assert cmp_["exact_at_b1"] is False
    # the served verdict rule (every count outside the allowed labels must be zero) fails on it
    assert any(cmp_["counts"][k] for k in cmp_["counts"] if k not in ("identical", "near_tie_equivalent"))
    full = [_resp(t, [t]) for t in ("Paris", "Lima", "Nairobi")]
    assert sc.compare(full, full)["exact_at_b1"] is True


def test_oracle_reads_unicode_dashes_as_ascii_hyphen():
    """AMF2-8: the dash normalisation was replace("-", "-")."""
    sv = _mod("amoeba27b_served")
    for dash in ("‐", "‑", "–", "—", "−", "－"):
        assert sv.oracle(f"The capital is Port{dash}au{dash}Prince.", "Port-au-Prince"), hex(ord(dash))
    assert sv.oracle("The answer is −42.", "-42")
    assert not sv.oracle("Port au Prince", "Port-au-Prince")                   # every other character stays exact
    assert not sv.oracle("Port_au_Prince", "Port-au-Prince")


def test_spec_check_and_served_report_failures_for_a_nonzero_exit():
    """AMF2-7: fail / error / no_reference verdicts, dead servers and footprint aborts ended with exit 0."""
    sv, sc = _mod("amoeba27b_served"), _mod("amoeba27b_spec_check")
    ok = {"layers": {"required": {"gates_row_invariant": True, "advice": None}},
          "served": {"ref": {"verdict": "reference"}, "ast": {"verdict": "pass"}}}
    assert sc.failures(ok) == [] and sc.failures({}) == []
    bad = {"layers": {"required": {"gates_row_invariant": np.False_, "lane_exact_row_invariant_and_covered": True,
                                   "advice": "serve with amoeba_tiles_gate_per_row=true"}},
           "served": {"ref": {"verdict": "reference"}, "a": {"verdict": "fail"}, "b": {"verdict": "error"},
                      "c": {"verdict": "no_reference"}}}
    assert sc.failures(bad) == ["layers.gates_row_invariant", "served.a:fail", "served.b:error",
                                "served.c:no_reference"]
    arms = {"full": {"widths": {"B1": {"rc": 0, "requests": 20}, "B4": {"rc": 1, "requests": 20}}},   # low score: a result
            "dead": {"error": "RuntimeError('server not healthy')", "widths": {}},
            "big": {"server_aborted_on_footprint": True, "widths": {}},
            "lost": {"widths": {"B1": {"rc": 1, "rounds": 5}}}}
    assert set(sv.failed_arms({"arms": arms})) == {"dead", "big", "lost"}
    assert sv.failed_arms({"arms": {"full": arms["full"]}}) == {}


# ----------------------------------------------------------------------------- AMF2-12 / AMF2-7 (det served)

def test_det_served_scores_false_fire_and_never_reports_unscored_as_zero_of_n():
    """AMF2-12: false_fire / prose rows were tallied as ok 0 of n; an HTTP error body looked like a reply."""
    ds = _mod("amoeba27b_det_served")
    ff = ds.FALSE_FIRE[0]
    want = ff.split(": ", 1)[1]
    replies = [("false_fire", want, _resp(want)),                                       # verbatim
               ("false_fire", want, _resp(want.replace("555-1234", "555‑1234"))),   # typeset dash: still verbatim
               ("false_fire", want, _resp("Call 6660 on 12/25/2024 at 10:30.")),         # a tile fired on the number
               ("false_fire", None, _resp("In 1999 the number 42 was popular.")),        # free-form: unscored
               ("prose", None, _resp("A keeper climbs the stairs each evening.")),
               ("sanity_mul", "36", _resp("36")),
               ("sanity_mul", "48", {"http_error": 500, "body": "boom"})]
    score, oks = {}, []
    for task, exp, r in replies:
        _, ok, err = ds.score_row(task, exp, r)
        ds.tally(score, task, ok, err)
        oks.append(ok)
    assert oks == [True, True, False, None, None, True, False]
    assert score["false_fire"] == {"n": 3, "ok": 2, "unscored": 1, "http_error": 0}
    assert score["prose"] == {"n": 0, "ok": 0, "unscored": 1, "http_error": 0}
    assert score["sanity_mul"] == {"n": 2, "ok": 1, "unscored": 0, "http_error": 1}
    assert ds.failed_arms({"arms": {"a": {"footprint": {"gb": 40.0, "killed": None}},
                                    "b": {"error": "RuntimeError('b: server exited 1')", "footprint": {"killed": None}},
                                    "c": {"footprint": {"gb": 81.0, "killed": "footprint 81.0 GB > 80 GB"}}}}).keys() \
        == {"b", "c"}


# ----------------------------------------------------------------------------- AMF2-5 / AMF2-6 (box partition)

def _box(root, name, mask, t2, held=0.02):
    d = root / f"box-{name}"
    d.mkdir(parents=True)
    np.savez(d / "export_core.npz", core=(t2 | mask).astype(np.uint8))
    (d / "carve.json").write_text(json.dumps({
        "inspect": {"paths": {"layerwise_grad": {"ok": True}}}, "export_core": {"held": {"mean_d_ce": held}},
        "fixed_only_held": {"mean_d_ce": 0.3},
        "bisect": {"zero": {"0.05": {"random_same_counts_held": {"mean_d_ce": 0.2}}}}}))


def _partition(tmp_path, n_boxes, nan_box=None):
    L, E = 4, 16 + 2 * n_boxes
    t2 = np.zeros((L, E), bool)
    t2[:, :16] = True
    root = tmp_path / "boxes"
    for i in range(n_boxes):
        m = np.zeros((L, E), bool)
        m[:, 16 + 2 * i:18 + 2 * i] = True
        _box(root, f"t{i:03d}", m, t2, held=float("nan") if i == nan_box else 0.02)
    np.savez(tmp_path / "t2.npz", core=t2.astype(np.uint8))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_fn_box_partition.py"), "--t2", str(tmp_path / "t2.npz"),
                        "--boxes-root", str(root), "--out", str(out), "--min-tile", "2"],
                       capture_output=True, text=True, cwd=ROOT)
    return r, out


def test_box_partition_nan_held_fails_the_box(tmp_path):
    """AMF2-5: a NaN held d_CE passed both gates."""
    r, out = _partition(tmp_path, 3, nan_box=1)
    assert r.returncode == 0, r.stderr
    p = json.loads((out / "partition.json").read_text())
    assert sorted(p["passing"]) == ["t000", "t002"]
    assert p["verdicts"]["t001"]["pass"] is False and len(p["verdicts"]["t001"]["why"]) == 2


def test_box_partition_refuses_more_boxes_than_the_bit_field_holds(tmp_path):
    """AMF2-6: boxes 65+ were shifted out of the int64 tile code and dropped silently."""
    r, out = _partition(tmp_path, 66)
    assert r.returncode != 0 and "at most 64" in r.stderr
    assert not (out / "partition.json").exists()


def test_box_partition_64_boxes_still_partition(tmp_path):
    r, out = _partition(tmp_path, 64)
    assert r.returncode == 0, r.stderr
    p = json.loads((out / "partition.json").read_text())
    assert p["coverage"]["partition"] and p["distinct_boxes"] == 64
    assert sum(t["kind"] == "box" for t in p["tiles"]) == 64


# ----------------------------------------------------------------------------- AMF2-9

def test_fn_window_parts_refuses_k_alone_and_unknown_parts():
    """AMF2-9: --parts k ran nothing and ended 'done', exit 0."""
    W = _mod("amoeba_nexttok_fn_window")
    assert W._parts_arg("q,k") == "q,k" and W._parts_arg("q") == "q" and W._parts_arg("") == ""
    for spec in ("k", "x", "q,z", "kq"):
        with pytest.raises(Exception) as e:
            W._parts_arg(spec)
        assert "parts" in str(e.value)
    args = [a for a in W.parser()._actions if a.dest == "parts"][0]
    assert args.type is W._parts_arg


# ----------------------------------------------------------------------------- AMF2-10

def test_router_threshold_report_names_in_sample_numbers_and_cross_estimates():
    """AMF2-10: recall / FPR measured on the rows that set the threshold were reported as held."""
    R = _mod("amoeba27b_router")
    rng = np.random.default_rng(3)
    pos, neg = rng.normal(2.0, 1.0, 400), rng.normal(0.0, 1.0, 600)
    pf, nf = np.arange(400) % 2, np.arange(600) % 2
    thr, rep = R.threshold_report(pos, neg, pf, nf, 0.98)
    assert thr == pytest.approx(float(np.quantile(pos, 0.02)))                 # the shipped threshold is unchanged
    assert "held_recall" not in rep and "held_fpr" not in rep
    assert rep["threshold_set_recall"] >= 0.98                                 # by construction: in-sample
    # two-fold: each fold's threshold is measured on the other fold only
    hit = sum(int((pos[pf != f] >= np.quantile(pos[pf == f], 0.02)).sum()) for f in (0, 1))
    assert rep["cross_recall"] == pytest.approx(hit / 400)
    assert 0.9 < rep["cross_recall"] < 1.0 and 0 < rep["cross_fpr"] < 1
    thr0, rep0 = R.threshold_report([], neg, [], nf, 0.98)
    assert thr0 == 0.0 and rep0["threshold_set_recall"] is None and rep0["cross_recall"] is None
    assert rep0["threshold_from"].startswith("none")
    _, one = R.threshold_report(pos[:3], neg, [0, 0, 0], nf, 0.98)             # positives in one fold only
    assert one["cross_recall"] is None and one["threshold_set_recall"] is not None
    assert {R.fold_of(f"src{i}") for i in range(40)} == {0, 1}


# ----------------------------------------------------------------------------- AMF2-11

def test_spec_proxy_counts_no_token_past_the_end():
    """AMF2-11: a last draft run that reached the end still counted a verifier token (25 of 24)."""
    Bx = _mod("amoeba27b_boxes")
    assert Bx.spec_proxy([True] * 24, 8) == (24, 3)
    assert Bx.spec_proxy([False] * 5, 8) == (5, 5)
    assert Bx.spec_proxy([True, True, False, True], 8) == (4, 2)               # run 2 + verifier, then run 1 at the end
    assert Bx.spec_proxy([True] * 9, 8) == (9, 1)                              # 8 drafts + the verifier's token
    for agree in ([True, False, True, True, False, False, True], [True] * 17, [False, True] * 6):
        assert Bx.spec_proxy(agree, 4)[0] == len(agree)
