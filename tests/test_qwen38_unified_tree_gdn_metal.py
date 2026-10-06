"""Exact parity for the mlx2-owned Qwen3.8 tree/cohort GDN backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.runtime.models import qwen38_tree_gdn as owned

def _reference():
    configured = os.environ.get("MLX2_TENSORFOLD_SOURCE")
    if not configured:
        pytest.skip("MLX2_TENSORFOLD_SOURCE is required for source-bound parity")
    tensorfold = Path(configured)
    if not (tensorfold / "src/tensorfold").is_dir():
        pytest.skip("MLX2_TENSORFOLD_SOURCE does not contain src/tensorfold")
    source = str(tensorfold / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from tensorfold.kernels.qwen.dense.v1 import lane_glue, stream_gdn

    return stream_gdn, lane_glue


def _equal(left, right):
    mx.eval(left, right)
    assert bool(mx.array_equal(left, right).item())


def test_owned_tree_backend_matches_corrected_tensorfold_all_operations():
    reference, reference_glue = _reference()
    parents = ([-1, 0, 0, 1, 1], [-1, 0, 1, 1, 2])
    nk, nv, dk, dv, taps = 16, 48, 128, 128, 4
    rows = sum(len(row) for row in parents)
    channels = 2 * nk * dk + nv * dv
    zba_width = nv * dv + 2 * nv

    def draw(shape, seed, dtype=mx.bfloat16):
        return (mx.random.normal(shape, key=mx.random.key(seed)) * 0.05).astype(dtype)

    qkv = draw((1, rows, channels), 1)
    zba = draw((1, rows, zba_width), 2)
    conv_weight = draw((channels, taps, 1), 3)
    conv_states = [draw((1, taps - 1, channels), 10 + lane) for lane in range(2)]
    recurrent = [draw((1, nv, dv, dk), 20 + lane, mx.float32) for lane in range(2)]
    a_log = draw((nv,), 30)
    dt_bias = draw((nv,), 31)

    reference_plan = reference.ConvPlan(parents, taps - 1)
    owned_plan = owned.ConvPlan(parents, taps - 1)
    reference_pre = reference.gdn_pre(
        qkv,
        conv_states,
        conv_weight,
        reference_plan,
        zba,
        a_log,
        dt_bias,
        nk=nk,
        nv=nv,
        dk=dk,
        dv=dv,
    )
    owned_pre = owned.gdn_pre(
        qkv,
        conv_states,
        conv_weight,
        owned_plan,
        zba,
        a_log,
        dt_bias,
        nk=nk,
        nv=nv,
        dk=dk,
        dv=dv,
    )
    for left, right in zip(reference_pre, owned_pre):
        _equal(left, right)

    first_rows = len(parents[0])
    first_windows = reference_plan.windows[: first_rows * taps]
    split_a = zba[:, :first_rows, -nv:]
    split_b = zba[:, :first_rows, nv * dv : nv * dv + nv]
    split_pre = owned.gdn_pre_split(
        qkv[:, :first_rows],
        conv_states[0],
        conv_weight,
        first_windows,
        split_a,
        split_b,
        a_log,
        dt_bias,
        nk=nk,
        nv=nv,
        dk=dk,
        dv=dv,
    )
    glue_pre = reference_glue.gdn_pre(
        qkv[:, :first_rows],
        conv_states[0],
        conv_weight,
        first_windows,
        split_a,
        split_b,
        a_log,
        dt_bias,
        nk=nk,
        nv=nv,
        dk=dk,
        dv=dv,
    )
    for left, right in zip(glue_pre, split_pre):
        _equal(left, right)

    reference_tree = reference.tree(
        *reference_pre[:5], recurrent, reference.TreePlan(parents)
    )
    owned_tree = owned.tree(*owned_pre[:5], recurrent, owned.TreePlan(parents))
    _equal(reference_tree, owned_tree)

    paths = ([0, 1, 3], [0, 1, 2, 4])
    firsts = (0, len(parents[0]))
    reference_commit = reference.CommitPlan(paths, firsts, taps - 1)
    owned_commit = owned.CommitPlan(paths, firsts, taps - 1)
    reference_replay = reference.replay(*reference_pre[:5], recurrent, reference_commit)
    owned_replay = owned.replay(*owned_pre[:5], recurrent, owned_commit)
    for left, right in zip(reference_replay, owned_replay):
        _equal(left, right)

    reference_tails = reference.conv_tails(conv_states, qkv, reference_commit)
    owned_tails = owned.conv_tails(conv_states, qkv, owned_commit)
    for left, right in zip(reference_tails, owned_tails):
        _equal(left, right)

    stats = owned.stats()
    assert stats["backend"] == "mlx2_qwen38_unified_tree_gdn_v1"
    assert stats["pre_calls"] >= 1
    assert stats["tree_calls"] >= 1
    assert stats["replay_calls"] >= 1
    assert stats["tail_calls"] >= 1
