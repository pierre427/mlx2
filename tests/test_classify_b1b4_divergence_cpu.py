import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from classify_b1b4_divergence import classify_pair  # noqa: E402


def _row(tokens, windows, correct=True):
    return {"correct": correct, "logprob_windows": [[t, w] for t, w in zip(tokens, windows)]}


W = {5: -0.1, 6: -2.0}


def test_identical_near_tie_high_margin_and_oracle():
    same = _row([5, 5], [W, W])
    assert classify_pair(same, same)["verdict"] == "identical"
    tie_a = _row([5, 1], [W, {1: -0.5, 2: -0.6}])
    tie_b = _row([5, 2], [W, {2: -0.5, 1: -0.6}])
    got = classify_pair(tie_a, tie_b)
    assert got["verdict"] == "near_tie_equivalent" and got["first_divergent_token"] == 1
    far_b = _row([5, 2], [W, {2: -0.1, 1: -3.0}])
    assert classify_pair(tie_a, far_b)["verdict"] == "high_margin"
    wrong = _row([5, 2], [W, {2: -0.5, 1: -0.6}], correct=False)
    assert classify_pair(tie_a, wrong)["verdict"] == "near_tie_oracle_failed"
    missing = _row([5, 3], [W, {3: -0.1, 4: -0.2}])
    assert classify_pair(tie_a, missing)["verdict"] == "unclassified"
    assert classify_pair({"correct": True}, tie_b)["verdict"] == "unclassified"
