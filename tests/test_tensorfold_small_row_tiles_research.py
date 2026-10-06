"""CPU/static gates for the TensorFold #196 tile crossover benchmark."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlx2.runtime.lane import matmul

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "research" / "bench_tensorfold_small_row_tiles.py"


def _module():
    spec = importlib.util.spec_from_file_location("tensorfold_small_row_tiles", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_current_mlx2_lane_keeps_the_32_column_strategy():
    bench = _module()
    assert (matmul.NT, matmul.ROW_BLOCK, matmul.MAX_ROWS) == (32, 16, 128)
    info = bench.metadata()
    assert info["current_mlx2"]["already_implemented"] is True
    assert info["current_mlx2"]["production_change"] is False


def test_source_selection_law_only_narrows_long_slice_one_stream_weights():
    bench = _module()
    assert bench.narrow_strategy(streams=1, widest=32, split_k=8)
    assert not bench.narrow_strategy(streams=2, widest=32, split_k=8)
    assert not bench.narrow_strategy(streams=1, widest=64, split_k=8)
    assert not bench.narrow_strategy(streams=1, widest=32, split_k=4)
    with pytest.raises(ValueError):
        bench.narrow_strategy(streams=0, widest=32, split_k=8)


def test_shipped_qwen38_shapes_have_the_many_slice_geometry_from_pr196():
    bench = _module()
    for k, n in bench.SHAPES.values():
        assert matmul.split_k(n, k, 64, 4) > 4
        assert n % 64 == 0 and k % 64 == 0


def test_benchmark_covers_both_sides_of_the_reported_row_crossover():
    bench = _module()
    rows = bench.metadata()["execution_plan"]["cases"][0]["rows"]
    assert {1, 16, 17, 32, 64, 128}.issubset(rows)
    assert "tensorfold_nt32" in bench.metadata()["arms"]
    assert "tensorfold_nt64" in bench.metadata()["arms"]


def test_external_kernel_checkout_is_exact_revision_clean_and_byte_pinned():
    bench = _module()
    source = bench.metadata()["tensorfold_checkout"]
    assert source["ready"] is True
    assert source["exact_revision"] is True
    kernel = source["files"][bench.KERNEL_RELATIVE_PATH]
    assert kernel["clean"] is True
    assert kernel["observed_sha256"] == bench.KERNEL_SHA256
    assert kernel["revision_sha256"] == bench.KERNEL_SHA256


def test_checkout_verifier_refuses_a_dirty_relevant_source_file(tmp_path):
    bench = _module()
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    source = root / "kernel.py"
    source.write_text("PIN = 1\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "pin"], cwd=root, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert bench.pinned_checkout_snapshot(
        root,
        expected_revision=revision,
        files={"kernel.py": expected_hash},
        require_match=True,
    )["ready"]
    source.write_text("PIN = 2\n")
    with pytest.raises(RuntimeError, match="does not match"):
        bench.pinned_checkout_snapshot(
            root,
            expected_revision=revision,
            files={"kernel.py": expected_hash},
            require_match=True,
        )


def test_describe_receipt_expands_layout_costs_and_cli_execution_order():
    bench = _module()
    command = [
        sys.executable,
        str(SCRIPT),
        "--describe",
        "--rows",
        "32,64",
        "--shapes",
        "out_proj",
        "--warmups",
        "1",
        "--rounds",
        "3",
        "--seed",
        "11",
    ]
    receipt = json.loads(
        subprocess.run(command, check=True, capture_output=True).stdout
    )
    plan = receipt["execution_plan"]
    assert plan["seed"] == 11
    assert plan["warmups_per_arm"] == 1
    assert plan["measured_rounds_per_arm"] == 3
    assert len(plan["cases"]) == 1
    case = plan["cases"][0]
    assert case["name"] == "out_proj"
    assert case["rows"] == [32, 64]
    assert case["input_allocation"] == [64, 6144]
    assert case["estimated_bytes"]["plain_weight"] == 6144 * 5120 // 2
    assert case["estimated_bytes"]["three_weight_layouts"] == 3 * 6144 * 5120 // 2
    assert receipt["environment"]["mlx"]["exact_match"]
    assert receipt["tensorfold_checkout"]["ready"]
    command = receipt["deferred_gpu_command"]
    assert command[command.index("--rows") + 1] == "32,64"
    assert command[command.index("--shapes") + 1] == "out_proj"
    assert command[command.index("--tensorfold-root") + 1] == str(
        bench.DEFAULT_TENSORFOLD_ROOT
    )
    assert "--run-gpu" in command and "--i-hold-gpu-lease" in command
