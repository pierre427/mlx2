from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.runtime.models import qwen4_exp
from mlx2.runtime.models import qwen4_fused_gdn_verify as fused
from scripts.bench_qwen4_gdn_replay_cpu import (
    _case,
    bf16_round,
    reconstruct_state,
    replay_states,
    restore_conv_window,
    run_probe,
    traffic_estimate,
)


@pytest.mark.parametrize("width", [3, 8])
def test_cpu_reconstruction_matches_every_partial_acceptance(width):
    initial, keys, corrections, decay = _case(width)
    snapshots = replay_states(initial, keys, corrections, decay)
    for accepted in range(1, width):
        actual = reconstruct_state(initial, keys, corrections, decay, accepted)
        assert np.array_equal(actual, snapshots[accepted - 1])


@pytest.mark.parametrize("width", [3, 8])
def test_cpu_reconstruction_matches_state_dependent_gdn_corrections(width):
    """Exercise the part ReplaySSM depends on: ``u`` comes from live state."""
    rng = np.random.default_rng(100 + width)
    hk, hv, dk, dv = 2, 6, 16, 16
    initial = rng.normal(size=(hv, dv, dk)).astype(np.float32)
    keys = np.empty((width - 1, hk, dk), dtype=np.float32)
    corrections = np.empty((width - 1, hv, dv), dtype=np.float32)
    decay = rng.uniform(0.8, 1.0, size=(width - 1, hv)).astype(np.float32)
    values = rng.normal(size=(width - 1, hv, dv)).astype(np.float32)
    beta = rng.uniform(0.0, 1.0, size=(width - 1, hv)).astype(np.float32)
    key_for_value_head = np.arange(hv) // (hv // hk)
    state = initial.copy()
    expected = []

    for step in range(width - 1):
        keys[step] = bf16_round(rng.normal(size=(hk, dk)).astype(np.float32))
        key = keys[step, key_for_value_head]
        state *= decay[step, :, None, None]
        recalled = (state * key[:, None, :]).sum(axis=-1, dtype=np.float32)
        corrections[step] = (values[step] - recalled) * beta[step, :, None]
        state += corrections[step, :, :, None] * key[:, None, :]
        expected.append(state.copy())

    for accepted in range(1, width):
        actual = reconstruct_state(initial, keys, corrections, decay, accepted)
        assert np.array_equal(actual, expected[accepted - 1])


@pytest.mark.parametrize("width", [3, 8])
def test_conv_window_is_derived_from_checkpoint_and_qkv(width):
    keep, dim = 3, 7
    initial = np.arange(keep * dim, dtype=np.float32).reshape(keep, dim)
    qkv = np.arange(width * dim, dtype=np.float32).reshape(width, dim) + 1000
    combined = np.concatenate([initial, qkv], axis=0)
    for accepted in range(1, width):
        assert np.array_equal(
            restore_conv_window(initial, qkv, accepted),
            combined[accepted : accepted + keep],
        )


def test_compact_kernel_is_distinct_and_does_not_emit_snapshots():
    assert "state_snapshots" not in fused._REPLAY_SOURCE
    assert "conv_snapshots" not in fused._REPLAY_SOURCE
    assert "replay_keys" in fused._REPLAY_SOURCE
    assert "replay_corrections" in fused._REPLAY_SOURCE
    assert "replay_decay" in fused._REPLAY_SOURCE
    assert "st = st * decay;" in fused._RECONSTRUCT_SOURCE
    assert "st = st + key * correction;" in fused._RECONSTRUCT_SOURCE


def test_compact_replay_core_default_is_off_and_profile_can_select_it():
    assert qwen4_exp._FUSED_GDN_REPLAY_ROLLBACK is False
    assert qwen4_exp._FUSED_GDN_REPLAY_ROLLBACK_MODES == ("snapshots", "compact")
    holder = SimpleNamespace(fused_gdn_replay_rollback_mode="snapshots")
    qwen4_exp.GatedDeltaNet.set_fused_gdn_replay_rollback_mode(holder, "compact")
    assert holder.fused_gdn_replay_rollback_mode == "compact"
    with pytest.raises(ValueError, match="unknown fused GDN replay rollback mode"):
        qwen4_exp.GatedDeltaNet.set_fused_gdn_replay_rollback_mode(holder, "silent")


@pytest.mark.parametrize("accepted,tape_steps", [(0, 2), (3, 2), (1, 0)])
def test_reconstruction_admission_fails_closed(accepted, tape_steps):
    with pytest.raises(ValueError):
        fused.validate_qwen4_gdn_replay_acceptance(accepted, tape_steps)


def test_static_traffic_estimate_and_harness_labels():
    for width in (3, 8):
        traffic = traffic_estimate(width)
        assert (
            traffic["compact_output_bytes_per_layer"]
            < traffic["snapshot_output_bytes_per_layer"]
        )
    report = run_probe((3, 8), repeats=2)
    assert report["gpu_speedup_claimed"] is False
    assert all(
        result["all_partial_lengths_bit_exact"] for result in report["results"]
    )
