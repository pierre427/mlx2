import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import hy4_abba_ladder as L


def lane(ids, ok=True, decode=50.0, ttft=1.0):
    return {"lane": 0, "token_ids": ids, "oracle_pass": ok, "decode_tok_s": decode, "ttft_s": ttft,
            "prefill_tok_s": 1000.0}


def report(a_ids, b_ids, a2=None, b2=None, ok=True):
    def run(n, arm, ids, decode):
        return {"run": n, "arm": arm, "cells": [{"cell": 1024, "width": 1, "aggregate_tok_s": decode,
                                                  "lanes": [lane(ids, ok, decode)]}]}
    return {"runs": [run(1, "A", a_ids, 50.0), run(2, "B", b_ids, 55.0),
                     run(3, "B", b2 or b_ids, 55.0), run(4, "A", a2 or a_ids, 50.0)]}


def test_first_divergence_and_window():
    assert L.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert L.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert L.first_divergence([1, 2], [1, 2, 3]) == 2
    rows = [{"id": 5, "logprob": -0.1, "top_logprobs": [{"id": 6, "logprob": -0.3}]}]
    assert L.window_at(rows, 0) == {5: -0.1, 6: -0.3}
    assert L.window_at(rows, 3) is None


def test_exact_rows_and_perf_ratio():
    v = L.verdict(report([1, 2, 3], [1, 2, 3]), {"A": {}, "B": {}})
    assert v["summary"]["statuses"] == {"token_exact": 1}
    assert v["perf_mean_of_runs"]["1024x1"]["decode_B_over_A"] == 1.1
    assert v["rows"][0]["same_arm_repeat_equal"] == {"A": True, "B": True}


def test_near_tie_needs_both_windows_and_the_oracle():
    key = (1024, 1, 0, 1)
    win_a, win_b = {2: -0.6, 9: -0.8}, {9: -0.6, 2: -0.7}
    rep = report([1, 2, 3], [1, 9, 3])
    assert L.verdict(rep, {"A": {key: win_a}, "B": {key: win_b}})["rows"][0]["status"] == "near_tie_equivalent"
    assert L.verdict(rep, {"A": {key: win_a}, "B": {}})["rows"][0]["status"] == "unclassified"
    high = {"A": {key: {2: -0.01, 9: -4.0}}, "B": {key: {9: -0.01, 2: -4.0}}}
    assert L.verdict(rep, high)["rows"][0]["status"] == "high_margin"
    bad = report([1, 2, 3], [1, 9, 3], ok=False)
    assert L.verdict(bad, {"A": {key: win_a}, "B": {key: win_b}})["rows"][0]["status"] == "high_margin"


def test_same_arm_repeat_difference_is_reported():
    v = L.verdict(report([1, 2, 3], [1, 2, 3], a2=[1, 2, 4]), {"A": {}, "B": {}})
    assert v["rows"][0]["same_arm_repeat_equal"]["A"] is False
