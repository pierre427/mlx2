"""Real serving-status integration with a scripted CPU fixture, no artifact."""

import json

import pytest
from test_structured_deferral import scripted_engine as _scripted_engine

from mlx2.runtime.models import qsdpa_verify_metal as qvm

scripted_engine = _scripted_engine


def test_fp_decode_status_distinguishes_selection_and_execution(scripted_engine, monkeypatch):
    from mlx2.runtime.models import qsdpa_decode_metal as qdm

    monkeypatch.setattr(qdm, "STATS", {"decode_kernel_calls": 0,
        "fp_decode_kernel_calls": 0, "fp_decode_gqa16_calls": 0})
    monkeypatch.delenv("MLX2_FP_DECODE_KERNEL", raising=False)
    build, _state = scripted_engine
    engine = build(declare_marker=True, max_lanes=1, max_inflight=1)
    initial = engine.status()["fp_decode_attention"]
    assert not initial["selected"] and not initial["observed_host_dispatch"]
    monkeypatch.setenv("MLX2_FP_DECODE_KERNEL", "1")
    selected = engine.status()["fp_decode_attention"]
    assert selected["selected"] and not selected["observed_host_dispatch"]
    qdm.STATS["fp_decode_kernel_calls"] += 1
    qdm.STATS["fp_decode_gqa16_calls"] += 1
    observed = engine.status()["fp_decode_attention"]
    assert observed["observed_host_dispatch"] and observed["counts"]["fp_decode_gqa16_calls"] == 1
    assert observed["basis"] == "selected_host_graph_branch"
    assert selected["counts"]["fp_decode_gqa16_calls"] == 0


def test_serving_status_roundtrips_bounded_host_branch_snapshots(
    scripted_engine, monkeypatch,
):
    # Isolate observations without resetting another test's process counters.
    monkeypatch.setattr(
        qvm, "_SELECTED_HOST_BRANCH_COUNTS",
        dict.fromkeys(qvm.SELECTED_HOST_BRANCHES, 0),
    )
    monkeypatch.setattr(qvm, "_SELECTED_HOST_BRANCH_EPOCH", 0)
    build, _state = scripted_engine
    engine = build(declare_marker=True, max_lanes=1, max_inflight=1)
    initial = engine.status()["qsdpa_verify"]
    baseline = json.loads(json.dumps(initial["selected_host_branches"]))
    for number, branch in enumerate(qvm.SELECTED_HOST_BRANCHES, 1):
        for _ in range(number):
            qvm.note_selected_host_branch(branch)

    observed = json.loads(json.dumps(engine.status()["qsdpa_verify"]))
    assert observed["counts"] == initial["counts"]
    assert observed["by_rows"] == initial["by_rows"]
    assert observed["selected_host_branches"]["basis"] == "selected_host_graph_branch"
    assert observed["selected_host_branches"]["scope"] == "process"
    assert observed["selected_host_branches"]["counts"] == {
        "direct_decode": 1,
        "direct_verify": 2,
        "composed": 3,
        "composed_tiled": 4,
        "dequantized_flash": 5,
    }
    assert qvm.selected_host_branch_delta(baseline)["counts"] == (
        observed["selected_host_branches"]["counts"]
    )
    # JSON/status snapshots are detached from the live observations.
    observed["selected_host_branches"]["counts"]["composed"] = 999
    assert engine.status()["qsdpa_verify"]["selected_host_branches"]["counts"]["composed"] == 3
    reset = qvm.reset_selected_host_branches()
    assert all(value == 0 for value in reset["counts"].values())
    with pytest.raises(ValueError, match="reset"):
        qvm.selected_host_branch_delta(baseline)
