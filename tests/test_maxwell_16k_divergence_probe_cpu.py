import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import maxwell_16k_divergence_probe as probe  # noqa: E402

PROSE = ("The problems cover money and cost questions, time and hour calculations, ages, "
         "percent changes, rates of speed and distance, fractions and ratios, and totals from "
         "buying and selling. " * 2)


def _trace(path, arm, ids, text=PROSE):
    row = {"content": text, "text_sha256": arm, "finish_reason": "stop",
           "token_trace": {"token_ids": ids}}
    path.write_text(json.dumps({"arm": arm, "prompt_sha256": "p", "rows": [row, row]}))


def _window(path, arm, position, selected, tops):
    content = [{"id": selected, "top_logprobs": [{"id": k, "logprob": v} for k, v in tops.items()]}]
    path.write_text(json.dumps({"arm": arm, "rows": [{"position": position, "route": arm,
                                                      "logprobs": {"content": content}}]}))


def _run(tmp_path, ref_tops, cand_tops, ref_sel=7, cand_sel=9):
    _trace(tmp_path / "r.json", "ord", [1, 2, 3, ref_sel, 4])
    _trace(tmp_path / "c.json", "mtp", [1, 2, 3, cand_sel])
    probe.main(["compare", "--reference", str(tmp_path / "r.json"), "--candidate",
                str(tmp_path / "c.json"), "--output", str(tmp_path / "d.json")])
    div = json.loads((tmp_path / "d.json").read_text())
    assert div["position"] == 4 and div["shared_prefix_tokens"] == 3
    _window(tmp_path / "rw.json", "ord", 4, ref_sel, ref_tops)
    _window(tmp_path / "cw.json", "mtp", 4, cand_sel, cand_tops)
    probe.main(["classify", "--reference", str(tmp_path / "rw.json"), "--candidate",
                str(tmp_path / "cw.json"), "--reference-trace", str(tmp_path / "r.json"),
                "--candidate-trace", str(tmp_path / "c.json"), "--divergence",
                str(tmp_path / "d.json"), "--output", str(tmp_path / "v.json")])
    return json.loads((tmp_path / "v.json").read_text())


def test_near_tie_two_way(tmp_path):
    v = _run(tmp_path, {"7": -0.6, "9": -0.8, "5": -3.0}, {"9": -0.6, "7": -0.7, "5": -3.0})
    assert v["verdict"] == "near_tie_equivalent" and v["clause"] == "two_way"


def test_high_margin(tmp_path):
    v = _run(tmp_path, {"7": -0.1, "9": -2.5, "5": -3.0}, {"9": -0.6, "7": -0.7, "5": -3.0})
    assert v["verdict"] == "high_margin_divergence"


def test_window_must_reproduce_trace(tmp_path):
    v = _run(tmp_path, {"7": -0.6, "9": -0.8}, {"7": -0.6, "9": -0.8}, cand_sel=9)
    # candidate window selected 9 as traced, reference 7: reproduces; now break it
    _window(tmp_path / "cw.json", "mtp", 4, 7, {"7": -0.6, "9": -0.8})
    probe.main(["classify", "--reference", str(tmp_path / "rw.json"), "--candidate",
                str(tmp_path / "cw.json"), "--reference-trace", str(tmp_path / "r.json"),
                "--candidate-trace", str(tmp_path / "c.json"), "--divergence",
                str(tmp_path / "d.json"), "--output", str(tmp_path / "v.json")])
    assert json.loads((tmp_path / "v.json").read_text())["verdict"] == "unclassified"
    assert v["verdict"] == "near_tie_equivalent"


def test_oracle_rejects_loops():
    assert probe.oracle(PROSE, "stop")["pass"]
    assert not probe.oracle("the cost of the apples is high and " * 20, "length")["pass"]


def _gsm(path, texts, correct, concurrency=1):
    rows = [{"test_index": i, "text": t, "predicted": t[-1], "correct": c, "route": "native_mtp"}
            for i, (t, c) in enumerate(zip(texts, correct))]
    path.write_text(json.dumps({"arms": [{"rows": rows, "concurrency": concurrency,
                                          "summary": {"accuracy": sum(correct) / len(correct)}}]}))


def test_widths(tmp_path):
    _gsm(tmp_path / "o.json", ["a1", "b2"], [True, False])
    _gsm(tmp_path / "b1.json", ["a1", "c2"], [True, True])
    _gsm(tmp_path / "b4.json", ["a1", "c3"], [True, False], 4)
    probe.main(["widths", "--ordinary", str(tmp_path / "o.json"), "--b1", str(tmp_path / "b1.json"),
                "--b4", str(tmp_path / "b4.json"), "--output", str(tmp_path / "w.json")])
    w = json.loads((tmp_path / "w.json").read_text())
    assert not w["b1_equals_b4"] and w["b1_b4_texts_differ"] == [1] and w["b1_b4_answers_differ"] == [1]
    assert w["route_correct_ordinary_wrong"] == [1] and w["b4_concurrency"] == 4
