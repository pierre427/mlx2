"""CPU checks of the near-tie contract used to classify Ripple divergences."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "classify", Path(__file__).parents[1] / "scripts/research/ripple_divergence_classify.py")
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def w(*pairs):
    return {"selected": pairs[0][0], "selected_logprob": pairs[0][1], "top": list(pairs)}


def test_two_way_near_tie():
    assert c.contract(w((1, -0.5), (2, -0.6)), w((2, -0.55), (1, -0.6))) == ("near_tie_equivalent", "two-way")


def test_margin_above_ceiling_is_not_two_way():
    verdict, _ = c.contract(w((1, -0.1), (2, -0.9)), w((2, -0.4), (1, -0.5)))
    assert verdict != "near_tie_equivalent" or _ != "two-way"


def test_exact_tie_with_third_token_in_other_slot():
    control = w((1, -0.7), (2, -0.7), (3, -0.9))
    arm = w((2, -0.6), (3, -0.8), (1, -0.95))
    assert c.contract(control, arm) == ("near_tie_equivalent", "exact-tie")


def test_multi_way_and_missing_tokens():
    control = w((1, -1.0), (3, -1.05), (2, -1.1))
    arm = w((2, -1.0), (4, -1.02), (1, -1.2))
    assert c.contract(control, arm) == ("near_tie_equivalent", "multi-way")
    far = w((2, -0.1), (4, -2.0), (5, -3.0))
    assert c.contract(w((1, -0.5), (2, -0.6)), far)[0] == "failed"
    close_missing = w((2, -1.0), (4, -1.05), (5, -1.1))
    assert c.contract(w((1, -1.0), (2, -1.1)), close_missing)[0] == "unclassified"


def test_oracle_flags_degenerate_and_off_topic():
    assert c.oracle("decode0", list(range(40)), "the printing press")["passed"]
    assert not c.oracle("decode0", [7] * 20, "the printing press")["passed"]
    assert not c.oracle("decode2", list(range(40)), "a refrigerator")["passed"]
    assert not c.oracle("copy0", list(range(40)), "def but nothing else")["passed"]


def test_oracle_rejecting_the_reference_leaves_the_row_unclassified(tmp_path, monkeypatch):
    import json

    class Tok:
        @staticmethod
        def decode(ids):
            return "def reviewed"

    import transformers
    monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "from_pretrained", lambda *a, **k: Tok())
    row = {"arm": "x", "cell": "copy0", "position": 3, "control_token": 1, "arm_token": 2,
           "control_tokens": [7] * 20, "arm_tokens": list(range(20)), "prefix": [7, 7, 7]}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"rows": [row]}))
    win = lambda sel, other: {"selected": sel, "selected_logprob": -0.5, "top": [[sel, -0.5], [other, -0.5]]}
    entry = lambda arm, w: {"row_arm": "x", "cell": "copy0", "position": 3, "reproduced": True, "window": w}
    windows = tmp_path / "w.json"
    windows.write_text(json.dumps({"arms": {"control": {"rows": [entry("control", win(1, 2))]},
                                            "x": {"rows": [entry("x", win(2, 1))]}}}))
    out = tmp_path / "out.json"
    monkeypatch.setattr("sys.argv", ["c", "--plan", str(plan), "--windows", str(windows),
                                     "--tokenizer", str(tmp_path), "--output", str(out)])
    c.main()
    result = json.loads(out.read_text())["rows"][0]
    assert result["verdict"] == "unclassified" and "reference" in result["reason"]
