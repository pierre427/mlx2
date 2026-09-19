"""CPU-only correctness and traffic probe for compact Qwen4 GDN rollback.

The timings are synthetic NumPy costs, not Metal or end-to-end speedups.  The
byte estimates describe verify-kernel rollback outputs at the stated geometry.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def bf16_round(values: np.ndarray) -> np.ndarray:
    """Round float32 values to bfloat16 precision, retaining float32 storage."""
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def replay_states(
    initial: np.ndarray,
    keys: np.ndarray,
    corrections: np.ndarray,
    decay: np.ndarray,
) -> list[np.ndarray]:
    """Materialize the exact float32 recurrence after every tape position."""
    state = np.array(initial, dtype=np.float32, copy=True)
    hv, _, _ = state.shape
    hk = keys.shape[1]
    if hv % hk:
        raise ValueError("value heads must be divisible by key heads")
    key_for_value_head = np.arange(hv) // (hv // hk)
    snapshots = []
    for step in range(keys.shape[0]):
        state *= decay[step, :, None, None]
        state += (
            corrections[step, :, :, None]
            * keys[step, key_for_value_head, None, :]
        )
        snapshots.append(state.copy())
    return snapshots


def reconstruct_state(
    initial: np.ndarray,
    keys: np.ndarray,
    corrections: np.ndarray,
    decay: np.ndarray,
    accepted: int,
) -> np.ndarray:
    if not 1 <= accepted <= keys.shape[0]:
        raise ValueError(
            f"accepted prefix {accepted} outside compact replay tape 1..{keys.shape[0]}"
        )
    return replay_states(
        initial, keys[:accepted], corrections[:accepted], decay[:accepted]
    )[-1]


def restore_conv_window(
    initial_conv: np.ndarray, qkv: np.ndarray, accepted: int
) -> np.ndarray:
    keep = initial_conv.shape[0]
    if not 1 <= accepted < qkv.shape[0]:
        raise ValueError(
            f"accepted prefix {accepted} must be a partial width of {qkv.shape[0]}"
        )
    combined = np.concatenate([initial_conv, qkv], axis=0)
    return np.ascontiguousarray(combined[accepted : accepted + keep])


def traffic_estimate(
    width: int,
    *,
    hk: int = 16,
    hv: int = 48,
    dk: int = 128,
    dv: int = 128,
    conv_dim: int = 10240,
    conv_kernel: int = 4,
    layers: int = 36,
) -> dict[str, int | float]:
    if width < 2:
        raise ValueError("verify width must be at least 2")
    tape_steps = width - 1
    state_bytes = hv * dv * dk * 4
    conv_bytes = (conv_kernel - 1) * conv_dim * 2
    snapshot_outputs = tape_steps * (state_bytes + conv_bytes)
    compact_outputs = tape_steps * (hk * dk * 2 + hv * dv * 4 + hv * 4)
    return {
        "verify_width": width,
        "snapshot_output_bytes_per_layer": snapshot_outputs,
        "compact_output_bytes_per_layer": compact_outputs,
        "output_byte_reduction_ratio": snapshot_outputs / compact_outputs,
        "snapshot_output_bytes_all_layers": snapshot_outputs * layers,
        "compact_output_bytes_all_layers": compact_outputs * layers,
        "retained_existing_qkv_input_bytes_per_layer": width * conv_dim * 2,
    }


def _case(width: int, seed: int = 7):
    rng = np.random.default_rng(seed + width)
    hk, hv, dk, dv = 2, 6, 16, 16
    initial = rng.normal(size=(hv, dv, dk)).astype(np.float32)
    keys = bf16_round(rng.normal(size=(width - 1, hk, dk)).astype(np.float32))
    corrections = rng.normal(size=(width - 1, hv, dv)).astype(np.float32)
    decay = rng.uniform(0.8, 1.0, size=(width - 1, hv)).astype(np.float32)
    return initial, keys, corrections, decay


def run_probe(widths=(3, 8), repeats: int = 200) -> dict:
    results = []
    for width in widths:
        initial, keys, corrections, decay = _case(width)
        snapshots = replay_states(initial, keys, corrections, decay)
        exact = all(
            np.array_equal(
                reconstruct_state(initial, keys, corrections, decay, accepted),
                snapshots[accepted - 1],
            )
            for accepted in range(1, width)
        )
        accepted = max(1, (width - 1) // 2)

        started = time.perf_counter_ns()
        for _ in range(repeats):
            np.stack(snapshots, axis=0).copy()
        snapshot_materialize_ns = (time.perf_counter_ns() - started) / repeats

        started = time.perf_counter_ns()
        for _ in range(repeats):
            (keys.copy(), corrections.copy(), decay.copy())
        tape_materialize_ns = (time.perf_counter_ns() - started) / repeats

        started = time.perf_counter_ns()
        for _ in range(repeats):
            snapshots[accepted - 1].copy()
        snapshot_restore_ns = (time.perf_counter_ns() - started) / repeats

        started = time.perf_counter_ns()
        for _ in range(repeats):
            reconstruct_state(initial, keys, corrections, decay, accepted)
        replay_restore_ns = (time.perf_counter_ns() - started) / repeats

        results.append(
            {
                "width": width,
                "all_partial_lengths_bit_exact": exact,
                "synthetic_cpu_ns": {
                    "snapshot_artifact_copy": snapshot_materialize_ns,
                    "compact_tape_copy": tape_materialize_ns,
                    "snapshot_partial_restore_copy": snapshot_restore_ns,
                    "compact_partial_reconstruct": replay_restore_ns,
                },
                "traffic": traffic_estimate(width),
            }
        )
    return {
        "label": "CPU synthetic microbenchmark and static byte estimate",
        "gpu_speedup_claimed": False,
        "notes": [
            "Timings use tiny NumPy tensors and do not predict Metal latency.",
            (
                "Traffic counts rollback outputs only; retained qkv is an "
                "existing verify input."
            ),
            (
                "Full acceptance uses the fused verify kernel's final state "
                "and does not reconstruct."
            ),
        ],
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", default="3,8")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    widths = tuple(int(value) for value in args.widths.split(","))
    report = run_probe(widths, args.repeats)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
