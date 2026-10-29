"""Real serving-status integration with a scripted CPU fixture, no artifact."""

import json

import pytest
from test_structured_deferral import scripted_engine as _scripted_engine

from mlx2.runtime.models import qsdpa_verify_metal as qvm

scripted_engine = _scripted_engine


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
