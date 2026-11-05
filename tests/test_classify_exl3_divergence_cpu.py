import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from classify_exl3_divergence import classify  # noqa: E402


def arm(tokens, windows, text="bonjour"):
    return {"token_ids": tokens, "windows": windows, "text": text}


SAME = [[1, -0.1], [2, -0.2], [3, -3.0], [4, -4.0], [5, -5.0]]


def test_exact_and_length_only():
    assert classify(arm([1, 2], [SAME, SAME]), arm([1, 2], [SAME, SAME]))["classification"] == "exact"
    assert classify(arm([1, 2], [SAME]), arm([1], [SAME]))["classification"] == "unclassified"


def test_top_two_near_tie():
    a = arm([7, 1], [SAME, [[1, -0.30], [2, -0.40], [3, -3.0], [4, -4.0], [5, -5.0]]])
    b = arm([7, 2], [SAME, [[2, -0.35], [1, -0.45], [3, -3.0], [4, -4.0], [5, -5.0]]])
    row = classify(a, b)
    assert (row["first_divergence"], row["clause"], row["classification"]) == (1, "top_two", "near_tie_equivalent")


def test_exact_tie_clause():
    a = arm([1], [[[1, -0.5], [2, -0.5], [9, -0.6], [4, -4.0], [5, -5.0]]])
    b = arm([2], [[[2, -0.4], [9, -0.7], [1, -0.8], [4, -4.0], [5, -5.0]]])
    row = classify(a, b)
    assert row["clause"] == "exact_tie" and row["classification"] == "near_tie_equivalent"


def test_multi_way_clause_and_failure():
    a = arm([1], [[[1, -0.50], [3, -0.55], [2, -0.60], [4, -4.0], [5, -5.0]]])
    b = arm([2], [[[2, -0.50], [4, -0.55], [1, -0.70], [3, -4.0], [5, -5.0]]])
    assert classify(a, b)["clause"] == "multi_way"
    far = arm([2], [[[2, -0.1], [4, -0.2], [1, -1.5], [3, -4.0], [5, -5.0]]])
    assert classify(a, far)["classification"] == "failed"


def test_missing_token_and_oracle():
    a = arm([1], [[[1, -0.1], [2, -0.2], [3, -0.3], [4, -0.35], [5, -0.4]]])
    b = arm([6], [[[6, -0.1], [7, -0.2], [8, -0.3], [9, -0.35], [10, -0.4]]])
    assert classify(a, b)["classification"] == "unclassified"
    c = arm([7, 2], [SAME, [[2, -0.35], [1, -0.45], [3, -3.0], [4, -4.0], [5, -5.0]]], text="hello")
    d = arm([7, 1], [SAME, [[1, -0.30], [2, -0.40], [3, -3.0], [4, -4.0], [5, -5.0]]])
    assert classify(c, d)["classification"] == "unclassified"
