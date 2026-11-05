"""Proof-window verdicts cannot promote a replay that did not reproduce."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "inkling_divergence_probe", SCRIPTS / "probe_inkling_divergence.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def rows():
    return (
        {
            "reproduced": True,
            "zero_based_index": 2,
            "selected_prefix": [100, 200, 10],
            "top": [(10, -0.1), (11, -0.3)],
        },
        {
            "reproduced": True,
            "zero_based_index": 2,
            "selected_prefix": [100, 200, 11],
            "top": [(11, -0.1), (10, -0.4)],
        },
    )


def test_prose_margin_evidence_does_not_become_semantic_qualification():
    left, right = rows()
    result = probe.classify(left, right, 10, 11, semantic_oracle=False)
    assert result["classification"] == "near_tie_margin_only_prose_oracle_not_qualified"
    assert result["cross_selected_margins_nats"] == pytest.approx([0.2, 0.3])
    assert (
        probe.classify(left, right, 10, 11, semantic_oracle=True)["classification"]
        == "near_tie_equivalent"
    )


@pytest.mark.parametrize(
    "mutation,verdict",
    [
        ("earlier_choice", "not_reproduced"),
        ("prefix", "no_shared_prefix"),
        ("position", "no_shared_prefix"),
        ("margin", "high_margin_divergence"),
        ("absent_bounded", "high_margin_divergence_window_bound"),
        ("absent_unbounded", "outside_window"),
    ],
)
def test_failed_window_contract_never_passes(mutation, verdict):
    left, right = rows()
    if mutation == "earlier_choice":
        left["reproduced"] = False
    elif mutation == "prefix":
        left["selected_prefix"][0] = 101
    elif mutation == "position":
        left["zero_based_index"] = 3
    elif mutation == "margin":
        left["top"] = [(10, -0.1), (11, -0.7)]
    else:
        left["top"] = [(10, -0.1), (12, -0.8 if mutation == "absent_bounded" else -0.2)]
    assert (
        probe.classify(left, right, 10, 11, semantic_oracle=True)["classification"]
        == verdict
    )


def schedule():
    return {
        "schema": "mlx2.prefill-slice-schedule.v1",
        "start": 0,
        "covered": 9,
        "slices": 3,
        "boundaries": [4, 5, 9],
        "boundaries_truncated": False,
        "deviation_reason": "width",
    }


def test_replay_executes_off_grid_slices_without_rounding_to_configured_step():
    class Model:
        def __init__(self):
            self.inputs = []

        def make_cache(self):
            return []

        def __call__(self, x, cache):
            self.inputs.append(x.tolist()[0])
            return probe.mx.array([[[1.0, 0.0]]])

    model = Model()
    boundaries = probe.replay_boundaries(schedule(), 10, serial=True)
    probe.window(model, list(range(10)), [0], 0, 4, boundaries=boundaries)
    assert model.inputs == [[0, 1, 2, 3], [4], [5, 6, 7, 8], [9]]
    ordinary = Model()
    probe.window(ordinary, list(range(10)), [0], 0, 4)
    assert ordinary.inputs == [[0, 1, 2, 3], [4, 5, 6, 7], [8], [9]]


@pytest.mark.parametrize(
    "change",
    [
        {"start": 4},
        {"covered": 8},
        {"slices": 4},
        {"boundaries_truncated": True},
        {"boundaries": [4, 5]},
        {"boundaries": [4, 4, 9]},
        {"boundaries": [4, 5, 10]},
        {"boundaries": [4, 5.0, 9]},
        {"boundaries": [True, 5, 9]},
        {"deviation_reason": "batched_prefill"},
        {"deviation_reason": "untracked_slice"},
    ],
)
def test_incomplete_or_unreplayable_schedule_refuses(change):
    with pytest.raises(ValueError):
        probe.replay_boundaries({**schedule(), **change}, 10, serial=True)


def test_serial_replay_rejects_parallel_origin_and_missing_schedule():
    with pytest.raises(ValueError):
        probe.replay_boundaries(schedule(), 10, serial=False)
    with pytest.raises(ValueError):
        probe.replay_boundaries(None, 10, serial=True)
