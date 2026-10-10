from performance_assessment import assess
from qualification_verdict import verdict


def run(
    n,
    *,
    needle=True,
    decode=100.0,
    prefill=1000.0,
    noise=False,
    apc=1.0,
    admission=True,
    thermal_state=0,
):
    token = "NEEDLE-A-0"
    content = f"answer {token}" if needle else "answer missing"
    prompt_tokens = 4096
    cached_tokens = int(prompt_tokens * apc)
    cold = {
        "done": True,
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 12,
        "cached_tokens": 20,
    }
    warm = {
        "done": True,
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 12,
        "cached_tokens": cached_tokens,
    }
    return {
        "run": n,
        "measured": True,
        "needles": [token],
        "requests": {"cold": {"rows": [cold]}, "warm": {"rows": [warm]}},
        "summary": {
            "decode_tokens_per_second": decode,
            "prefill_tokens_per_second": prefill,
        },
        "thermal_pre": {
            "stable": admission,
            "safe_admitted": admission,
            "samples": [{"thermal_state": thermal_state}],
        },
        "thermal_post": {"breached": False, "samples": []},
        "contaminated": noise,
        "contamination": ["swapouts rose"] if noise else [],
        "swapouts": {"delta": 10 if noise else 0},
        "foreign_activity": [],
    }


def ladder(*runs, error=None, attempts=(), expected_cells=((4096, 1),)):
    cells = [
        {
            "requested_tokens": requested,
            "width": width,
            "error": error,
            "runs": list(runs) if (requested, width) == (4096, 1) else [],
            "attempts": list(attempts),
        }
        for requested, width in expected_cells
    ]
    return {
        "runs_per_cell": 3,
        "finished_at": 1.0,
        "model": "m",
        "route": "r",
        "host": "test-host",
        "max_context": 4096,
        "expected_cells": [
            {"requested_tokens": t, "width": w} for t, w in expected_cells
        ],
        "initial": {
            "artifact": "a",
            "runtime": {"source_sha256": "a" * 64, "mlx_native_sha256": "b" * 64},
            "settings": {"mtp": False},
        },
        "cells": cells,
    }


def test_swap_noise_is_recorded_but_does_not_fail_correctness():
    result = verdict(ladder(run(0, noise=True), run(1), run(2)))
    assert result["qualified"] is True
    assert result["measurement_noise"]


def test_speed_drop_does_not_fail_correctness_but_performance_flags_it():
    reference = ladder(run(0), run(1), run(2))
    slower = ladder(run(0, decode=60), run(1, decode=65), run(2, decode=62))
    assert verdict(slower)["qualified"] is True
    result = assess(slower, reference)
    assert result["quoteable"] is False
    assert any("decode_tokens_per_second" in failure for failure in result["failures"])


def test_missed_raw_retrieval_needle_fails():
    assert verdict(ladder(run(0), run(1, needle=False), run(2)))["qualified"] is False


def test_warm_apc_under_90_percent_fails():
    assert verdict(ladder(run(0, apc=0.89), run(1), run(2)))["qualified"] is False


def test_incomplete_three_repetition_ladder_fails():
    assert verdict(ladder(run(0), run(1)))["qualified"] is False


def test_environmental_noise_prevents_performance_quoteability_only():
    reference = ladder(run(0), run(1), run(2))
    noisy = ladder(run(0, noise=True), run(1), run(2))
    assert verdict(noisy)["qualified"] is True
    assert assess(noisy, reference)["quoteable"] is False


def test_clean_matching_three_repetition_runs_are_performance_assessable():
    reference = ladder(run(0), run(1), run(2))
    current = ladder(
        run(0, decode=95, prefill=950),
        run(1, decode=97, prefill=970),
        run(2, decode=96, prefill=960),
    )
    assert assess(current, reference)["quoteable"] is True


def test_fair_safe_admission_can_qualify_but_not_support_performance_quote():
    fair = ladder(
        run(0, thermal_state=1), run(1, thermal_state=1), run(2, thermal_state=1)
    )
    nominal = ladder(run(0), run(1), run(2))
    assert verdict(fair)["qualified"] is True
    assert assess(fair, nominal)["quoteable"] is False


def test_environmental_noise_from_retried_attempt_is_preserved():
    result = verdict(ladder(run(0), run(1), run(2), attempts=[run(0, noise=True)]))
    assert result["qualified"] is True
    assert result["measurement_noise"]


def test_missing_plan_cell_coverage_fails():
    report = ladder(run(0), run(1), run(2), expected_cells=((4096, 1), (8192, 1)))
    assert verdict(report)["qualified"] is False


def test_malformed_summary_cannot_hide_incomplete_raw_requests():
    report = ladder(run(0), run(1), run(2))
    report["cells"][0]["runs"][0]["requests"]["warm"]["rows"] = []
    assert verdict(report)["qualified"] is False


def test_malformed_report_rejected_without_crashing():
    result = verdict(
        {
            "initial": [],
            "cells": [None],
            "runs_per_cell": 3,
            "expected_cells": [{"requested_tokens": [], "width": {}}],
        }
    )
    assert result["qualified"] is False
    assert result["failures"]


def test_safe_admission_refusal_returns_pending_not_functional_failure():
    report = ladder(run(0), run(1))
    report["pending_admission"] = {"cell": {"requested_tokens": 4096, "width": 1}}
    result = verdict(report)
    assert result["status"] == "pending_admission"
    assert result["qualified"] is False
    assert not result["failures"]
