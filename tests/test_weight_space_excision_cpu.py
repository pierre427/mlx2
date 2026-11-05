"""Synthetic CPU boundary repair, identity and real bypass semantics."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from mlx2.experimental.weight_space.excision import (
    BoundaryHealing,
    ExcisionCapture,
    LayerExcisionPlan,
    execute_numpy,
    fit_boundary_healing,
)
from mlx2.experimental.weight_space.memory import CalibrationSplit

CHECKPOINT = "boundary-fixture@" + "b" * 64
SOURCE = "e448f070b84642bfc5d11dc04719e40336326a68"


def plan():
    return LayerExcisionPlan(CHECKPOINT, SOURCE, 24, (21,), 3)


def capture(p):
    x = np.random.default_rng(19).normal(size=(16, 3)).astype("f4")
    residual = x @ np.array([[.5, 0, 0], [0, 0, 0], [0, 0, 0]], dtype="f4")
    ids = tuple(f"row-{i}" for i in range(16))
    c = ExcisionCapture(p.fingerprint, ids, x, x + residual, "c" * 64)
    return c, CalibrationSplit(ids[:8], ids[8:])


def test_remove_21_is_real_bypass_and_keeps_original_layer_indices():
    calls = []

    def layer(index):
        def forward(x):
            calls.append(index)
            return x + index
        return forward

    p = plan()
    value = execute_numpy([layer(i) for i in range(24)], np.ones((2, 3)), p)
    assert calls == [i for i in range(24) if i != 21]
    np.testing.assert_array_equal(value, np.ones((2, 3)) + sum(calls))
    assert p.receipt()["healing_window"] == [20, 22]
    assert not p.receipt()["checkpoint_weights_deleted"]


def test_healing_fits_calibration_only_and_scores_heldout():
    p = plan()
    c, split = capture(p)
    healed, report = fit_boundary_healing(p, c, split=split, rank=1, ridge=0)
    assert healed.healing.boundary_layer == 22
    assert healed.fingerprint != p.fingerprint
    assert report["heldout"]["residual_mse"] < 1e-12
    assert report["heldout"]["baseline_mse"] > 0
    assert report["selection_split"] == "calibration_only"
    assert not report["qualified"] and not report["task_quality_evaluated"]
    changed = c.reference_outputs.copy()
    changed[8:] += 50
    adversarial = replace(c, reference_outputs=changed)
    second, bad = fit_boundary_healing(p, adversarial, split=split, rank=1, ridge=0)
    np.testing.assert_array_equal(healed.healing.a, second.healing.a)
    np.testing.assert_array_equal(healed.healing.b, second.healing.b)
    assert bad["heldout"]["residual_mse"] > 2000
    assert bad["calibration_digest"] == report["calibration_digest"]
    assert bad["heldout_digest"] != report["heldout_digest"]


def test_healing_applies_after_retained_boundary_not_in_removed_layer():
    p = LayerExcisionPlan(CHECKPOINT, SOURCE, 4, (1,), 1)
    healing = BoundaryHealing(2, [[1]], [[2]], "a" * 64, "b" * 64, "c" * 64)
    p = replace(p, healing=healing)
    layers = [lambda x: x + 1, lambda x: x + 1000, lambda x: x * 2, lambda x: x + 3]
    # ((1 + 1) * 2) * (1 + 2) + 3, with layer 1 bypassed.
    np.testing.assert_array_equal(execute_numpy(layers, [[1]], p), [[15]])


@pytest.mark.parametrize("removed", [(), (0,), (23,), (22, 21), (1, 3), (1, 1), (True,), (1, 10**12)])
def test_invalid_or_unbounded_removal_is_rejected(removed):
    with pytest.raises(ValueError, match="contiguous interior"):
        replace(plan(), removed_layers=removed)


def test_plans_and_capture_factors_are_immutable_and_source_bound():
    p = plan()
    c, split = capture(p)
    healed, _ = fit_boundary_healing(p, c, split=split)
    with pytest.raises(FrozenInstanceError):
        healed.removed_layers = (20,)
    with pytest.raises(ValueError):
        healed.healing.a.flags.writeable = True
    with pytest.raises(ValueError, match="exact unhealed"):
        fit_boundary_healing(replace(p, source_revision="changed"), c, split=split)
    with pytest.raises(ValueError, match="exact unhealed"):
        fit_boundary_healing(healed, c, split=split)


def test_capture_and_healing_geometry_fail_closed():
    p = plan()
    c, split = capture(p)
    with pytest.raises(ValueError, match="geometry"):
        replace(c, reference_outputs=np.zeros((16, 2)))
    with pytest.raises(ValueError, match="boundary"):
        replace(p, healing=BoundaryHealing(20, np.ones((3, 1)), np.ones((1, 3)), "a" * 64, "b" * 64, "c" * 64))
    with pytest.raises(ValueError, match="exact disjoint"):
        fit_boundary_healing(p, c, split=CalibrationSplit(split.calibration_ids, split.heldout_ids[:-1]))
    with pytest.raises(ValueError, match="nonnegative"):
        fit_boundary_healing(p, c, split=split, ridge=float("nan"))
    with pytest.raises(ValueError, match="workspace budget"):
        fit_boundary_healing(p, c, split=split, workspace_budget_bytes=1)
    with pytest.raises(ValueError, match="hidden geometry"):
        execute_numpy([lambda x: x[:, :2]] * 24, np.ones((2, 3)), p)


@pytest.mark.parametrize("checkpoint,source", [("artifact@main", SOURCE), (CHECKPOINT, "HEAD"),
                                             ("space artifact@hash", SOURCE)])
def test_excision_identity_refuses_mutable_aliases(checkpoint, source):
    with pytest.raises(ValueError):
        replace(plan(), checkpoint=checkpoint, source_revision=source)
