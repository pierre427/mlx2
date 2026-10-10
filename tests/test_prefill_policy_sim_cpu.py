"""Host-model invariants; these do not establish tensor or device parity."""

import importlib.util
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/research/prefill_policy_sim.py"
spec = importlib.util.spec_from_file_location("prefill_policy_sim", PATH)
sim = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sim
spec.loader.exec_module(sim)


@pytest.mark.parametrize(
    "order", ["fcfs", "round_robin", "srpt", "bounded_srpt", "aging"]
)
@pytest.mark.parametrize("budget", [None, 10.0, 0.1])
def test_every_admitted_request_makes_progress_without_lost_rows(order, budget):
    requests = [sim.Request(i, i, 17 + i * 13, 4) for i in range(8)]
    report = sim.simulate(requests, order=order, chunk=16, round_budget=budget)
    assert report["completed"] == len(requests)
    for row in report["requests"]:
        assert row["prefilled"] == row["prefill"] == sum(row["chunks"])
        assert row["delivered"] == row["output"]
        assert all(0 < n <= 16 for n in row["chunks"])


def test_early_publication_changes_delivery_not_total_work_for_fixed_arrivals():
    requests = [sim.Request(0, 0, 0, 10), sim.Request(1, 0, 256, 4)]
    early = sim.simulate(requests, publication="early")
    late = sim.simulate(requests, publication="late")
    assert early["makespan"] == late["makespan"]
    assert (
        early["requests"][0]["first_delivery"] < late["requests"][0]["first_delivery"]
    )
    assert [(r["prefilled"], r["delivered"]) for r in early["requests"]] == [
        (r["prefilled"], r["delivered"]) for r in late["requests"]
    ]


def test_cancellation_releases_residency_and_suppresses_late_tokens():
    requests = [sim.Request(0, 0, 8192, 4, cancel_at=10), sim.Request(1, 1, 16, 4)]
    report = sim.simulate(requests, max_resident=1, chunk=16)
    assert report["cancelled"] == 1
    assert report["completed"] == 1
    first, peer = report["requests"]
    assert first["delivered"] == 0
    assert peer["delivered"] == 4


def test_bounded_srpt_improves_long_request_progress_over_unbounded_srpt():
    requests = sim.scenarios()["long_among_short"]
    bounded = sim.simulate(requests, order="bounded_srpt")
    shortest = sim.simulate(requests, order="srpt")
    # Compare the same token geometry. This assertion is about this finite trace.
    assert (
        bounded["requests"][0]["first_prefill_service"]
        < shortest["requests"][0]["first_prefill_service"]
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk": 0},
        {"chunk": True},
        {"max_resident": 0},
        {"max_rounds": 0},
        {"round_budget": 0},
        {"round_budget": float("nan")},
        {"aging_weight": float("inf")},
        {"publication": "bad"},
        {"order": "bad"},
    ],
)
def test_invalid_policies_fail_before_simulation(kwargs):
    with pytest.raises(ValueError):
        sim.simulate([sim.Request(0, 0, 1, 1)], **kwargs)


def test_ids_cancellation_and_bounded_execution_contracts():
    with pytest.raises(ValueError, match="unique"):
        sim.simulate([sim.Request(0, 0, 1, 1)] * 2)
    with pytest.raises(ValueError, match="arrival order"):
        sim.simulate([sim.Request(2, 0, 1, 1), sim.Request(1, 1, 1, 1)])
    with pytest.raises(ValueError, match="cancellation"):
        sim.Request(0, 2, 1, 1, cancel_at=1)
    with pytest.raises(RuntimeError, match="bounded rounds"):
        sim.simulate([sim.Request(0, 0, 100, 4)], max_rounds=1)
