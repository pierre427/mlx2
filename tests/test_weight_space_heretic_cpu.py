"""Synthetic CPU checks of projection math; no model/behavior qualification."""

import numpy as np
import pytest

from mlx2.experimental.weight_space.heretic import (
    LayerProfile,
    compile_abliteration,
    estimate_direction,
    retention_metrics,
)

CHECKPOINT = "synthetic-cpu-fixture@" + "e" * 64
REVISION = "f" * 40


def compile_fixture(w, u, **options):
    return compile_abliteration(
        {"layer": w}, {"layer": u}, {"layer": {"layer": 2, "component": "mlp_output"}},
        profiles={"mlp_output": LayerProfile(peak_layer=2, radius=1, peak_strength=0.5)},
        checkpoint=CHECKPOINT, source_revision=REVISION,
        provenance={"source": "synthetic CPU projection fixture"}, **options,
    )


def test_mean_direction_orthogonalization_and_orientation():
    reference = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)
    contrast = np.array([[2, 2, 0], [2, 2, 0]], dtype=np.float32)
    plain = estimate_direction(reference, contrast)
    np.testing.assert_allclose(plain.direction, np.array([1, 2, 0]) / np.sqrt(5), atol=1e-7)
    projected = estimate_direction(reference, contrast, orthogonalize_reference=True)
    np.testing.assert_allclose(projected.direction, [0, 1, 0], atol=1e-7)
    assert projected.report["reference_mean_projection"] == pytest.approx(0)
    with pytest.raises(ValueError):
        projected.direction.flags.writeable = True


def test_winsorization_is_per_capture_row_over_components():
    reference = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32)
    contrast = np.array([[2, 2, 1000], [2, 2, 1000]], dtype=np.float32)
    estimate = estimate_direction(reference, contrast, winsorization_quantile=0.5)
    np.testing.assert_allclose(estimate.direction, np.ones(3) / np.sqrt(3), atol=1e-7)


def test_layer_profile_interpolates_and_zero_radius_has_no_division():
    profile = LayerProfile(peak_layer=4, radius=2, peak_strength=2, edge_strength=0.5)
    assert profile.strength(4) == 2
    assert profile.strength(3) == 1.25
    assert profile.strength(2) == 0.5
    assert profile.strength(1) == 0
    assert LayerProfile(2, 0).strength(2) == 1
    assert LayerProfile(2, 0).strength(3) == 0


@pytest.mark.parametrize("mode", ["none", "pre"])
def test_projection_factor_orientation_matches_direct_equation(mode):
    w = np.array([[1, 2, 3], [4, 2, 1]], dtype=np.float32)
    u = np.array([1, 1], dtype=np.float32) / np.sqrt(2)
    p = compile_fixture(w, u, row_normalization=mode)
    a, b = p.factor_map["layer"]
    if mode == "none":
        expected = -0.5 * np.outer(u, u @ w)
    else:
        norms = np.linalg.norm(w, axis=1)
        expected = -0.5 * norms[:, None] * np.outer(u, u @ (w / norms[:, None]))
    np.testing.assert_allclose((a @ b).T, expected, rtol=1e-6, atol=1e-6)
    assert not p.approximate


def test_full_normalization_preserves_rows_before_explicit_rank_reduction():
    w = np.array([[1, 2, 3], [4, 2, 1]], dtype=np.float32)
    u = np.array([1, 1], dtype=np.float32) / np.sqrt(2)
    complete = compile_fixture(w, u, row_normalization="full", full_rank=2)
    a, b = complete.factor_map["layer"]
    np.testing.assert_allclose(np.linalg.norm(w + (a @ b).T, axis=1), np.linalg.norm(w, axis=1), rtol=1e-6)
    truncated = compile_fixture(w, u, row_normalization="full", full_rank=1)
    receipt = truncated.provenance["targets"]["layer"]
    assert truncated.approximate
    assert receipt["discarded_svd_frobenius_norm"] > 0
    assert receipt["maximum_row_norm_drift"] > 0


def test_disabled_layer_is_excluded_and_all_disabled_refused():
    w = np.ones((2, 3), dtype=np.float32)
    u = np.array([1, 0], dtype=np.float32)
    p = compile_abliteration(
        {"active": w, "inactive": w}, {"active": u, "inactive": u},
        {"active": {"layer": 1, "component": "mlp_output"}, "inactive": {"layer": 5, "component": "mlp_output"}},
        profiles={"mlp_output": LayerProfile(1, 0)}, checkpoint=CHECKPOINT,
        source_revision=REVISION, provenance={"source": "test"},
    )
    assert set(p.factor_map) == {"active"}
    assert p.provenance["targets"]["inactive"]["reason"] == "layer_component_gate"
    with pytest.raises(ValueError, match="all output targets"):
        compile_abliteration(
            {"layer": w}, {"layer": u}, {"layer": {"layer": 5, "component": "mlp_output"}},
            profiles={"mlp_output": LayerProfile(1, 0)}, checkpoint=CHECKPOINT,
            source_revision=REVISION, provenance={"source": "test"},
        )


def test_retention_metrics_separate_preservation_from_steering():
    reference = np.array([[0, 0], [1, 0]], dtype=np.float32)
    contrast = np.array([[4, 0], [5, 0]], dtype=np.float32)
    report = retention_metrics(reference, contrast, reference, contrast - 3, reference_mse_ceiling=0)
    assert report["reference_mse"] == 0
    assert report["reference_array_gate_passed"] is True
    assert report["ara_optimizer_executed"] is False
    assert report["task_retention_verified"] is False
    changed = retention_metrics(reference, contrast, reference + 1, contrast, reference_mse_ceiling=0.5)
    assert changed["reference_array_gate_passed"] is False


def test_degenerate_capture_and_invalid_component_fail_closed():
    with pytest.raises(ValueError, match="degenerate"):
        estimate_direction(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="output components"):
        compile_abliteration(
            {"layer": np.eye(2)}, {"layer": np.array([1, 0])}, {"layer": {"layer": 1, "component": "embedding"}},
            profiles={"embedding": LayerProfile(1, 0)}, checkpoint=CHECKPOINT,
            source_revision=REVISION, provenance={"source": "test"},
        )
