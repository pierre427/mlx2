"""Sweep 2026-10-06 (capsule lane, CAP-1): trajectory compiler keeps waypoints exact.

Each stored increment is the segment sum since the previous retained waypoint,
so ``anchor + cumsum`` lands on the retained states, and the compiler reports
both the selected-delta and the waypoint error.
"""


import math

import numpy as np
import pytest

from mlx2.runtime.activation_capsules import (
    ActivationCapsuleSpec,
    Authority,
    AuthorityScope,
    CapsuleProvenance,
    Geometry,
    ModelBindings,
    NormGateBounds,
    PayloadKind,
    PositionConvention,
    TapBinding,
    compile_pruned_trajectory,
)
from mlx2.runtime.activation_injection import (
    ACTIVATION_CAPSULE_REVISION,
    QWEN35_TARGET_INJECTION,
    ActivationInjectionBridge,
    prepare_loaded_activation_injection,
)

STATES = np.array(
    [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (4, 6)], dtype=np.float64
)


def _bridge(width=2):
    eye = np.zeros((4, width), np.float32)
    return ActivationInjectionBridge(
        artifact_fingerprint="p" * 8,
        bindings={"model": "m", "tokenizer": "t", "runtime": "r"},
        capsule_revision=ACTIVATION_CAPSULE_REVISION,
        state_dim=4,
        hidden_dim=width,
        injection_layer=0,
        key_projection=eye,
        value_projection=eye,
        direction_projection=eye,
    )


def _manifest(compiled, steps, width=2):
    spec = ActivationCapsuleSpec(
        payload_kind=PayloadKind.ORDERED_TRAJECTORY,
        bindings=ModelBindings("m", "m", "t", "r", "p" * 8),
        tap=TapBinding(0, "post-block", 0, QWEN35_TARGET_INJECTION),
        geometry=Geometry(hidden_width=width, sequence_length=steps),
        dtype="<f4",
        normalization="none",
        position=PositionConvention("none", "v1"),
        provenance=CapsuleProvenance("probe", "1", ("a" * 64,)),
        authority=Authority("tenant", AuthorityScope.SESSION, "s"),
        bounds=NormGateBounds(maximum_relative_norm=0.5, maximum_gate=0.5),
        ordered_transition_indices=compiled.transition_indices,
        metadata={"representation_space": "target_hidden"},
    )
    return spec.to_dict()


def _prepared_waypoints(compiled):
    """What the adapter actually builds (before row normalisation)."""
    deltas = compiled.coefficients.astype(np.float64) @ compiled.basis.astype(np.float64)
    return compiled.anchor.astype(np.float64)[None, :] + np.cumsum(deltas, axis=0)


def _angle_deg(a, b):
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _run():
    compiled = compile_pruned_trajectory(STATES, maximum_transitions=1, maximum_rank=1)
    prepared = prepare_loaded_activation_injection(
        _manifest(compiled, len(STATES)),
        compiled.tensors(),
        _bridge(),
        capsule_digest="c" * 64,
        gate=0.2,
        tokens=[1, 2, 3],
        prefill_step=16,
    )
    return compiled, prepared


def test_review_probe_adapter_key_follows_the_true_state():
    compiled, prepared = _run()
    assert compiled.transition_indices == (5,)
    waypoint = _prepared_waypoints(compiled)[0]
    truth = STATES[5]
    assert np.allclose(waypoint, truth, atol=1e-6)
    assert _angle_deg(waypoint, truth) < 1e-3
    # The adapter's normalised key row is now the direction of state 5, (4, 6).
    key = np.asarray(prepared.memory["keys"])[0]
    assert np.allclose(key, truth / np.linalg.norm(truth), atol=1e-6)
    assert prepared.memory["operation"] == "continuous_prefix"


def test_expected_retained_waypoint_is_exact():
    compiled, _prepared = _run()
    waypoint = _prepared_waypoints(compiled)[0]
    # Waypoint labelled transition index 5 must be state 5 when rank is full.
    assert np.allclose(waypoint, STATES[compiled.transition_indices[0]])


def test_expected_reported_error_bounds_the_waypoint_error():
    compiled, _prepared = _run()
    waypoints = _prepared_waypoints(compiled)
    truth = STATES[list(compiled.transition_indices)]
    displacement = truth - STATES[0]
    waypoint_error = np.linalg.norm(waypoints - truth) / np.linalg.norm(displacement)
    assert compiled.reconstruction_relative_error >= waypoint_error - 1e-9

REVIEW = np.array([(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (4, 6)], dtype=np.float64)


def test_review_probe_waypoint_is_exact_and_both_errors_reported():
    compiled = compile_pruned_trajectory(REVIEW, maximum_transitions=1, maximum_rank=1)
    assert compiled.transition_indices == (5,)
    assert np.allclose(compiled.reconstructed_waypoints()[0], (4.0, 6.0))
    assert compiled.selected_delta_relative_error < 1e-7
    assert compiled.waypoint_relative_error < 1e-7
    assert compiled.waypoint_max_angle_degrees < 1e-3
    # Four unit steps of 9 total norm were merged into the one segment.
    assert compiled.discarded_motion_fraction == pytest.approx(4 / 9)
    assert compiled.tail_relative_displacement == 0.0


@pytest.mark.parametrize("seed", range(20))
def test_full_rank_retained_waypoints_are_exact(seed):
    rng = np.random.default_rng(seed)
    states = np.cumsum(rng.normal(size=(24, 16)), axis=0) + rng.normal(size=16) * 5
    compiled = compile_pruned_trajectory(
        states, maximum_transitions=6, maximum_rank=6, dtype="<f8"
    )
    truth = states[list(compiled.transition_indices)]
    assert np.allclose(compiled.reconstructed_waypoints(), truth, atol=1e-9)
    assert compiled.waypoint_relative_error < 1e-12


@pytest.mark.parametrize("seed", range(20))
def test_reported_waypoint_error_is_honest_at_low_rank(seed):
    rng = np.random.default_rng(100 + seed)
    states = np.cumsum(rng.normal(size=(32, 24)), axis=0)
    compiled = compile_pruned_trajectory(states, maximum_transitions=8, maximum_rank=2)
    truth = states[list(compiled.transition_indices)]
    measured = np.linalg.norm(compiled.reconstructed_waypoints() - truth) / np.linalg.norm(
        truth - states[0]
    )
    assert compiled.waypoint_relative_error == pytest.approx(measured, rel=1e-4, abs=1e-6)
    assert compiled.reconstruction_relative_error >= compiled.waypoint_relative_error


@pytest.mark.parametrize("seed", range(10))
def test_displacement_basis_beats_old_increment_basis_on_waypoints(seed):
    """Eckart-Young: the waypoint-displacement basis is optimal for cumsum."""
    rng = np.random.default_rng(200 + seed)
    states = np.cumsum(rng.normal(size=(20, 12)) * rng.uniform(0.1, 3, size=(20, 1)), axis=0)
    compiled = compile_pruned_trajectory(states, maximum_transitions=6, maximum_rank=2, dtype="<f8")
    rows = [index - 1 for index in compiled.transition_indices]
    truth = states[list(compiled.transition_indices)]
    # Old law: SVD of the raw selected increments, then cumsum from anchor.
    raw = np.diff(states, axis=0)[rows]
    _u, _s, vh = np.linalg.svd(raw, full_matrices=False)
    basis = vh[:2]
    old = states[0] + np.cumsum(raw @ basis.T @ basis, axis=0)
    old_error = np.linalg.norm(old - truth) / np.linalg.norm(truth - states[0])
    assert compiled.waypoint_relative_error <= old_error + 1e-9


def test_include_endpoint_retains_final_state_and_zeroes_tail():
    states = np.array([(0, 0), (5, 0), (5, 1), (5, 1.5)], dtype=np.float64)
    plain = compile_pruned_trajectory(states, maximum_transitions=1, maximum_rank=1)
    assert plain.transition_indices == (1,)
    assert plain.tail_relative_displacement > 0.25
    ended = compile_pruned_trajectory(
        states, maximum_transitions=1, maximum_rank=1, include_endpoint=True
    )
    assert ended.transition_indices == (3,)
    assert np.allclose(ended.reconstructed_waypoints()[-1], states[-1], atol=1e-6)
    assert ended.tail_relative_displacement == 0.0


def test_zero_trajectory_stays_finite_and_errors_zero():
    compiled = compile_pruned_trajectory(np.zeros((4, 3)), maximum_transitions=2, maximum_rank=1)
    assert np.isfinite(compiled.reconstructed_waypoints()).all()
    assert compiled.reconstruction_relative_error == 0.0
    assert compiled.waypoint_relative_error == 0.0


def test_quality_metadata_is_finite_json_scalars():
    compiled = compile_pruned_trajectory(REVIEW, maximum_transitions=2, maximum_rank=1)
    quality = compiled.quality()
    assert set(quality) >= {"selected_delta_relative_error", "waypoint_relative_error"}
    assert all(isinstance(value, float) and np.isfinite(value) for value in quality.values())
