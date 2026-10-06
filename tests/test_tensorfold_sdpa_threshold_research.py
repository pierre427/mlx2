"""CPU/static gates for the TensorFold #197 viability benchmark."""

from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.models import base

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "research" / "bench_tensorfold_sdpa_threshold.py"


def _module():
    spec = importlib.util.spec_from_file_location(
        "tensorfold_sdpa_threshold_bench", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_current_mlx2_already_covers_both_sides_of_the_source_threshold():
    bench = _module()
    assert base._FUSED_D256_MIN_L == 256
    assert base._FUSED_D256_MAX_L == bench.FUSED_ROWS == 1024
    assert bench.current_mlx2_route(255) == "mlx_stock_dispatch"
    assert bench.current_mlx2_route(256) == "mlx2_force_fused"
    assert bench.current_mlx2_route(1023) == "mlx2_force_fused"
    assert bench.current_mlx2_route(1024) == "mlx_stock_dispatch"
    assert bench.metadata()["current_mlx2"]["already_implemented"] is True


def test_idle_prefill_still_reaches_stock_2048_but_contended_slices_stop_at_512():
    parameters = inspect.signature(BatchGenerator.__init__).parameters
    assert parameters["prefill_step_size"].default == 2048
    assert parameters["adaptive_prefill_slices"].default == (64, 128, 256, 512)


def test_pre197_partition_plan_covers_queries_once_and_preserves_visible_keys():
    bench = _module()
    parts = bench.partition_plan(1500, 20001, tensor_units=True)
    assert parts[0] == (0, 128, 18629)
    assert parts[-1] == (1408, 1500, 20001)
    assert [begin for begin, _, _ in parts] == [0, *[end for _, end, _ in parts[:-1]]]
    assert sum(end - begin for begin, end, _ in parts) == 1500
    assert all(visible == 20001 - 1500 + end for _, end, visible in parts)


def test_pre197_non_tensor_plan_aligns_intermediate_key_boundaries_and_avoids_short_tail():
    bench = _module()
    parts = bench.partition_plan(273, 8193, tensor_units=False)
    assert all(visible % 16 == 0 for _, end, visible in parts[:-1])
    assert parts[-1][1] == 273
    assert not 0 < parts[-1][1] - parts[-1][0] <= 16


def test_benchmark_matrix_straddles_the_dispatch_boundary_and_never_selects_policy():
    bench = _module()
    info = bench.metadata()
    rows = [case["rows"] for case in info["execution_plan"]["cases"]]
    assert 1023 in rows and 1024 in rows
    assert info["current_mlx2"]["production_change"] is False
    assert info["status"] == "research_harness_unexecuted"


def test_capability_tuple_does_not_synthesize_unsupported_mask_or_sink_routes():
    """Regression for the independent-capability bug fixed by oMLX #3461."""

    bench = _module()
    assert bench.current_mlx2_route(512, mask="causal") == "mlx2_force_fused"
    assert bench.current_mlx2_route(512, mask=None) == "mlx_stock_dispatch"
    assert bench.current_mlx2_route(512, mask=object()) == "mlx_stock_dispatch"
    assert (
        bench.current_mlx2_route(512, mask="causal", sinks=object())
        == "mlx_stock_dispatch"
    )
    assert (
        bench.current_mlx2_route(512, mask="causal", dtype="float32")
        == "mlx_stock_dispatch"
    )
    assert bench.current_mlx2_route(512, head_dim=128) == "mlx_stock_dispatch"


def test_runtime_pin_refuses_an_installed_mlx_version_drift():
    bench = _module()
    assert bench.require_distribution("mlx", bench.EXPECTED_MLX_VERSION)["exact_match"]
    with pytest.raises(RuntimeError, match="distribution drift"):
        bench.require_distribution("mlx", "0.0-impossible")


def test_describe_receipt_expands_exact_cli_case_geometry_without_importing_mlx():
    command = [
        sys.executable,
        str(SCRIPT),
        "--describe",
        "--rows",
        "1023,1024",
        "--keys",
        "4096",
        "--query-heads",
        "8",
        "--kv-heads",
        "2",
        "--warmups",
        "1",
        "--rounds",
        "3",
        "--seed",
        "7",
    ]
    receipt = json.loads(
        subprocess.run(command, check=True, capture_output=True).stdout
    )
    plan = receipt["execution_plan"]
    assert plan["seed"] == 7
    assert plan["warmups_per_arm"] == 1
    assert plan["measured_rounds_per_arm"] == 3
    assert [case["inputs"]["queries"] for case in plan["cases"]] == [
        [1, 8, 1023, 256],
        [1, 8, 1024, 256],
    ]
    assert plan["cases"][0]["expected_mlx2_route"] == "mlx2_force_fused"
    assert plan["cases"][1]["expected_mlx2_route"] == "mlx_stock_dispatch"
    assert receipt["environment"]["mlx"]["observed"]
    assert receipt["environment"]["mlx"]["exact_match"]
    command = receipt["deferred_gpu_command"]
    assert command[command.index("--rows") + 1] == "1023,1024"
    assert command[command.index("--keys") + 1] == "4096"
    assert "--run-gpu" in command and "--i-hold-gpu-lease" in command
