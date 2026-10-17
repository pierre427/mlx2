#!/usr/bin/env python3
"""Owned exactness probe for Super ReLU2/down/FP32 top-22 fusion."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mlx.core as mx
import nemotron35_probe_common as common
from mlx import nn

from mlx2.runtime.models.nemotron_relu2_down import relu2_down_combine
from mlx2.runtime.models.switch_layers import (
    QuantizedSwitchLinear,
    _scatter_unsort,
)


def bits_sha(value) -> str:
    return hashlib.sha256(bytes(memoryview(mx.array(value).view(mx.uint16)))).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Refusing to overwrite nonempty probe evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    owner = common.require_owner(args.cpg_owner_lock)
    before = common.host_snapshot()
    errors = common.admission_errors(before, minimum_available_bytes=4 * common.GIB)
    if errors:
        raise RuntimeError("; ".join(errors))

    cases = []
    for seed, scale in ((31, 0.25), (37, 1.0), (41, 3.0), (43, 8.0)):
        mx.random.seed(seed)
        down = QuantizedSwitchLinear(2688, 1024, 1, bias=False)
        down.scales = down.scales.astype(mx.bfloat16)
        down.biases = down.biases.astype(mx.bfloat16)
        down.eval()
        pre = (mx.random.normal((1, 1, 22, 2688)) * scale).astype(mx.bfloat16)
        indices = mx.zeros((1, 1, 22), dtype=mx.uint32)
        raw_scores = mx.random.uniform(shape=(1, 1, 22)).astype(mx.float32)
        scores = raw_scores / raw_scores.sum(axis=-1, keepdims=True)
        mx.eval(down.parameters(), pre, indices, scores)

        expert = down(nn.relu2(pre.reshape(22, 1, 2688)), indices.reshape(22))
        expert = expert.reshape(1, 1, 22, 1024)
        expected = (expert * scores[..., None]).sum(axis=-2).astype(expert.dtype)
        actual = relu2_down_combine(pre, indices, scores, down)
        mx.eval(expected, actual)
        mismatch_count = int(mx.sum(expected.view(mx.uint16) != actual.view(mx.uint16)).item())
        cases.append({
            "seed": seed,
            "scale": scale,
            "exact": mismatch_count == 0,
            "mismatch_count": mismatch_count,
            "expected_sha256": bits_sha(expected),
            "actual_sha256": bits_sha(actual),
        })
        del down, pre, indices, scores, expert, expected, actual
        mx.clear_cache()
    # Exercise the actual B1 top-22 sorted gather over distinct expert table
    # offsets. The fused kernel receives restored slot order, while the served
    # reference computes fc2 in expert-sorted order and then scatters back.
    mx.random.seed(47)
    down = QuantizedSwitchLinear(2688, 1024, 32, bias=False)
    down.scales = down.scales.astype(mx.bfloat16)
    down.biases = down.biases.astype(mx.bfloat16)
    down.eval()
    pre = mx.random.normal((1, 1, 22, 2688)).astype(mx.bfloat16)
    indices = mx.array([[[
        21, 0, 20, 1, 19, 2, 18, 3, 17, 4, 16,
        5, 15, 6, 14, 7, 13, 8, 12, 9, 11, 10,
    ]]], dtype=mx.uint32)
    raw_scores = mx.random.uniform(shape=(1, 1, 22)).astype(mx.float32)
    scores = raw_scores / raw_scores.sum(axis=-1, keepdims=True)
    activated = nn.relu2(pre)
    order = mx.argsort(indices.reshape(22))
    inv_order = mx.argsort(order)
    activated_sorted = activated.reshape(22, 1, 2688)[order]
    indices_sorted = indices.reshape(22)[order]
    expert_sorted = down(activated_sorted, indices_sorted, sorted_indices=True)
    expert = _scatter_unsort(expert_sorted, inv_order, indices.shape).squeeze(-2)
    expected = (expert * scores[..., None]).sum(axis=-2).astype(expert.dtype)
    expert_unsorted = down(
        activated.reshape(22, 1, 2688), indices.reshape(22), sorted_indices=False
    ).reshape(1, 1, 22, 1024)
    expected_unsorted = (
        expert_unsorted * scores[..., None]
    ).sum(axis=-2).astype(expert_unsorted.dtype)
    actual = relu2_down_combine(pre, indices, scores, down)
    one_hot_mismatches = []
    for slot in range(22):
        one_hot = mx.zeros((1, 1, 22), dtype=mx.float32)
        one_hot[..., slot] = 1.0
        isolated = relu2_down_combine(pre, indices, one_hot, down)
        isolated_expected = expert[..., slot, :]
        mx.eval(isolated, isolated_expected)
        count = int(mx.sum(
            isolated.view(mx.uint16) != isolated_expected.view(mx.uint16)
        ).item())
        if count:
            one_hot_mismatches.append({"slot": slot, "expert": int(indices[..., slot].item()), "count": count})
    mx.eval(expected, expected_unsorted, actual)
    mismatch_count = int(mx.sum(expected.view(mx.uint16) != actual.view(mx.uint16)).item())
    unsorted_mismatch_count = int(
        mx.sum(expected_unsorted.view(mx.uint16) != actual.view(mx.uint16)).item()
    )
    cases.append({
        "seed": 47,
        "scale": 1.0,
        "kind": "shuffled_multi_expert_sorted_reference",
        "exact": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "unsorted_reference_mismatch_count": unsorted_mismatch_count,
        "unsorted_expected_sha256": bits_sha(expected_unsorted),
        "one_hot_mismatches": one_hot_mismatches,
        "expected_sha256": bits_sha(expected),
        "actual_sha256": bits_sha(actual),
    })
    after = common.host_snapshot()
    contamination = [
        reason for reason in common.contamination(before, after)
        if reason != "swap-outs increased"
    ]
    receipt = {
        "schema": "mlx2.nemotron35-super-relu2-down-fp32-top22-probe.v1",
        "owner": owner,
        "source": common.source_identity(),
        "dependencies": common.runtime_dependencies(),
        "before": before,
        "after": after,
        "swap_ignored_by_user": True,
        "cases": cases,
        "passed": all(case["exact"] for case in cases) and not contamination,
        "contamination": contamination,
    }
    common.atomic_json(args.out_dir / "result.json", receipt)
    if not receipt["passed"]:
        raise RuntimeError("Super ReLU2/down component exactness probe failed")


if __name__ == "__main__":
    main()
