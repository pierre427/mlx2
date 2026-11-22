"""CPU checks for scripts/check_qwen38_27b_compact_replay_memory.py (no model load)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.qwen_gdn_contracts import QWEN38_GDN_CONTRACT
from mlx2.runtime.models.qwen38_state_plane import QWEN38_STATE_PLANE_OPERATIONS
import json
import os

from scripts.check_qwen38_27b_compact_replay_memory import (
    GPU_LOCK,
    analytic_report,
    estimate,
    gdn_geometry,
    parse_args,
    refuse_nonempty_evidence_dir,
    verify_gpu_lock,
    write_atomic,
)

G = QWEN38_GDN_CONTRACT.geometry
CONFIG = {"text_config": {
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "linear_num_key_heads": G.num_key_heads,
    "linear_num_value_heads": G.num_value_heads,
    "linear_key_head_dim": G.key_head_dim,
    "linear_value_head_dim": G.value_head_dim,
    "linear_conv_kernel_dim": G.conv_kernel,
}}


def test_analytic_mode_needs_no_gpu_flags():
    a = parse_args(["--model", "/m", "--analytic", "--verify-widths", "4", "17", "--lanes", "1", "8"])
    assert a.analytic and a.verify_widths == [4, 17] and a.lanes == [1, 8]
    assert a.state_dtype == "float32"


@pytest.mark.parametrize("argv", [
    ["--model", "/m"],                                   # GPU mode without --out/lock
    ["--model", "/m", "--out", "/tmp/x.json"],           # no --i-own-the-gpu
    ["--model", "/m", "--i-own-the-gpu"],                # no --out
    ["--model", "/m", "--analytic", "--verify-widths", "1"],
    ["--model", "/m", "--analytic", "--verify-widths", "18"],
    ["--model", "/m", "--analytic", "--lanes", "0"],
    ["--model", "/m", "--out", "/o", "--i-own-the-gpu", "--modes", "compact"],
    ["--model", "/m", "--out", "/o", "--i-own-the-gpu", "--modes", "compact", "compact"],
    ["--model", "/m", "--out", "/o", "--i-own-the-gpu", "--modes", "bogus", "compact"],
])
def test_bad_arguments_are_refused(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_gpu_mode_arguments():
    a = parse_args(["--model", "/m", "--out", "/o.json", "--i-own-the-gpu", "--widths", "1", "8"])
    assert not a.analytic and a.widths == [1, 8] and a.modes == ["snapshots", "compact"]


@pytest.mark.parametrize("steps", [2, 4, 9, 17])
def test_estimate_matches_state_plane_accounting(steps):
    """The analytic tape/snapshot bytes are the provider's own accounting."""
    conv_dim = 2 * G.num_key_heads * G.key_head_dim + G.num_value_heads * G.value_head_dim
    cache = ArraysCache(2)
    cache[0] = mx.zeros((1, G.conv_kernel - 1, conv_dim), mx.bfloat16)
    cache[1] = mx.zeros((1, G.num_value_heads, G.value_head_dim, G.key_head_dim), mx.float32)
    cache.start_speculation()
    auth = QWEN38_STATE_PLANE_OPERATIONS.authorize_compact_gdn_replay(
        cache, QWEN38_GDN_CONTRACT, steps
    )
    layer = estimate(gdn_geometry(CONFIG), steps)["per_layer"]
    assert layer["compact_tape"] == auth.tape_bytes
    assert layer["snapshot_record"] == auth.tape_bytes + auth.snapshot_bytes_avoided


def test_report_counts_only_b1_as_current_saving():
    report = analytic_report(CONFIG, [4], [1, 4])
    assert report["geometry"]["gdn_layers"] == 3
    row = report["rows"][0]
    one, four = row["lanes"]["1"], row["lanes"]["4"]
    assert one["current_saving"] == one["potential_saving"] > 0
    assert four["current_saving"] == 0 and four["potential_saving"] == 4 * one["potential_saving"]


def test_default_lock_path_is_the_shared_host_lock():
    assert str(GPU_LOCK) == "/Users/Shared/mlxuag/gpu.lock"


def _lock(tmp_path, owner):
    lock = tmp_path / "gpu.lock"
    lock.mkdir()
    (lock / "owner.json").write_text(owner if isinstance(owner, str) else json.dumps(owner))
    return lock


def test_gpu_lock_owned_by_an_ancestor_is_accepted(tmp_path):
    lock = _lock(tmp_path, {"pid": 4242, "label": "x"})
    seen = []

    def ancestors(pid):
        seen.append(pid)
        return {pid, 4242, 77}

    assert verify_gpu_lock(lock, pid=999, ancestors=ancestors)["pid"] == 4242
    assert seen == [999]


def test_gpu_lock_default_ancestor_chain_includes_this_process(tmp_path):
    lock = _lock(tmp_path, {"pid": os.getppid()})
    assert verify_gpu_lock(lock)["pid"] == os.getppid()


@pytest.mark.parametrize("owner", [
    {"pid": 4242},          # not an ancestor
    {"pid": "4242"},        # not an integer
    {"label": "no pid"},
    "not json",
    [4242],
])
def test_gpu_lock_with_foreign_or_bad_owner_is_refused(tmp_path, owner):
    lock = _lock(tmp_path, owner)
    with pytest.raises(SystemExit, match="refusing"):
        verify_gpu_lock(lock, pid=999, ancestors=lambda pid: {pid, 77})


def test_idle_sentinel_file_or_missing_lock_is_refused(tmp_path):
    sentinel = tmp_path / "gpu.lock"
    sentinel.write_text("")
    with pytest.raises(SystemExit, match="not a held lock directory"):
        verify_gpu_lock(sentinel, pid=1, ancestors=lambda pid: {pid})
    with pytest.raises(SystemExit, match="not a held lock directory"):
        verify_gpu_lock(tmp_path / "absent", pid=1, ancestors=lambda pid: {pid})
    empty = tmp_path / "empty.lock"
    empty.mkdir()
    with pytest.raises(SystemExit, match="unreadable"):
        verify_gpu_lock(empty, pid=1, ancestors=lambda pid: {pid})


def test_evidence_directory_must_be_absent_or_empty(tmp_path):
    assert refuse_nonempty_evidence_dir(tmp_path / "new") == tmp_path / "new"
    (tmp_path / "empty").mkdir()
    refuse_nonempty_evidence_dir(tmp_path / "empty")
    full = tmp_path / "full"
    full.mkdir()
    (full / "run.log").write_text("")  # even an empty file makes it nonempty
    with pytest.raises(SystemExit, match="nonempty"):
        refuse_nonempty_evidence_dir(full)
    stray = tmp_path / "file.json"
    stray.write_text("{}")
    with pytest.raises(SystemExit, match="nonempty"):
        refuse_nonempty_evidence_dir(stray)


def test_write_atomic_replaces_without_leaving_temp(tmp_path):
    path = tmp_path / "r" / "result.json"
    write_atomic(path, {"cells": [1]})
    write_atomic(path, {"cells": [1, 2]})
    assert json.loads(path.read_text()) == {"cells": [1, 2]}
    assert sorted(p.name for p in path.parent.iterdir()) == ["result.json"]
