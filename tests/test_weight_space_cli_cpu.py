"""CLI CPU fixtures reject every MLX import; no model artifact is executed."""

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from mlx2.experimental.weight_space.evaluation import (
    evaluate_projections,
    round_bfloat16,
)
from mlx2.experimental.weight_space.programs import create_program, load_program

CHECKPOINT = "synthetic-cpu-fixture@" + "c" * 64
REVISION = "d" * 40


def run_cli(*args, ok=True):
    code = """
import builtins, sys
original_import = builtins.__import__
def cpu_only(name, *args, **kwargs):
    if name == 'mlx' or name.startswith('mlx.'):
        raise AssertionError('CPU CLI attempted an MLX import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = cpu_only
from mlx2.experimental.weight_space.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    environment["MLX2_PREFLIGHT_CPU_ONLY"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code, *(str(arg) for arg in args)],
        env=environment, capture_output=True, text=True, timeout=30, check=False,
    )
    if ok:
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert result.returncode == 2, result.stderr
    return json.loads(result.stderr)


def repair_fixture(tmp_path):
    arrays = tmp_path / "arrays.npz"
    w = np.arange(12, dtype=np.float32).reshape(3, 4) / 11
    u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    x = np.array([[0.1, 0.2, 0.3, 0.4], [1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    np.savez(arrays, w=w, u=u, x=x)
    spec = tmp_path / "repair.json"
    spec.write_text(json.dumps({
        "checkpoint": CHECKPOINT, "source_revision": REVISION,
        "provenance": {"source": "synthetic CPU fixture"}, "strength": 0.7,
        "targets": {"layer": {"weight": "w", "direction": "u", "selected_columns": [1, 3]}},
    }))
    return arrays, spec


def test_cli_repair_inspect_compose_truncate_and_projection_evaluation(tmp_path):
    arrays, spec = repair_fixture(tmp_path)
    out = tmp_path / "repair"
    repair = run_cli("repair", "--arrays", arrays, "--spec", spec, "--output", out)
    assert repair["checkpoint"] == CHECKPOINT
    assert repair["status"]["qualified"] is False
    inspected = run_cli("inspect", out)
    assert inspected["program_id"] == repair["program_id"]
    composed = tmp_path / "composed"
    result = run_cli("compose", out, out, "--scale", "1", "--scale", "0.5", "--output", composed)
    assert result["rank"] == 2
    truncated = tmp_path / "truncated"
    result = run_cli("truncate", composed, "--rank", "1", "--output", truncated)
    assert result["approximate"] is True
    projection = tmp_path / "projection.json"
    projection.write_text(json.dumps({"targets": {"layer": {"weight": "w", "inputs": "x"}}}))
    receipt_dir = tmp_path / "projection-result"
    result = run_cli(
        "evaluate-projections", out, "--arrays", arrays, "--spec", projection,
        "--output", receipt_dir, "--simulate-bfloat16",
    )
    assert result["model_executed"] is False
    assert result["checkpoint_tensor_origin_verified"] is False
    assert result["targets"][0]["float32"]["zero_vs_base"]["exact_array_equal"] is True
    assert result["targets"][0]["simulated_bfloat16"]["device_arithmetic_reproduced"] is False
    assert (receipt_dir / "report.json").is_file()
    run_cli("repair", "--arrays", arrays, "--spec", spec, "--output", out, ok=False)


def test_cli_import_existing_lora(tmp_path):
    from safetensors.numpy import save_file

    source = tmp_path / "source"
    source.mkdir()
    save_file({"layer.lora_a": np.ones((3, 1), dtype=np.float32), "layer.lora_b": np.ones((1, 2), dtype=np.float32)}, str(source / "adapters.safetensors"))
    (source / "adapter_config.json").write_text(json.dumps({
        "lora_parameters": {"rank": 1, "scale": 0.25, "dropout": 0, "keys": ["layer"]},
    }))
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"source": "synthetic trained-format adapter"}))
    out = tmp_path / "imported"
    result = run_cli("import-lora", source, "--checkpoint", CHECKPOINT, "--source-revision", REVISION, "--provenance", provenance, "--output", out)
    assert result["construction"] == "imported_mlx2_lora"
    a, b = load_program(out).factor_map["layer"]
    np.testing.assert_array_equal(a @ b, np.full((3, 2), 0.25, dtype=np.float32))


def test_cli_calibrates_memory_from_explicit_disjoint_capture_split(tmp_path):
    x = np.array([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float32)
    y = x @ np.array([[1, 2], [2, 4]], dtype=np.float32)
    arrays = tmp_path / "memory.npz"
    np.savez(arrays, x=x, y=y)
    spec = tmp_path / "memory.json"
    spec.write_text(json.dumps({
        "source_revision": REVISION,
        "source": {
            "semantic_digest": "a" * 64, "session_revision": 1, "context_model": "test",
            "tenant": "tenant-a", "session": "session-a", "checkpoint": CHECKPOINT,
            "tokenizer_binding": "tokenizer-test", "runtime_binding": "runtime-test",
            "edge_ids": ["b" * 64],
        },
        "split": {"calibration_ids": ["a", "b", "c"], "heldout_ids": ["d"]},
        "captures": [{"target": "layer", "sample_ids": ["a", "b", "c", "d"], "inputs": "x", "target_deltas": "y"}],
        "rank": 1, "ridge": 0,
    }))
    result = run_cli("calibrate-memory", "--arrays", arrays, "--spec", spec, "--output", tmp_path / "memory-program")
    assert result["approximate"] is True
    assert result["provenance"]["promotion_scope"] == "session"
    assert result["provenance"]["calibration"]["evidence_scope"] == "supplied_activation_reconstruction_only"


def test_cli_rejects_object_npz_without_loading_pickle(tmp_path):
    arrays, spec = repair_fixture(tmp_path)
    np.savez(arrays, w=np.array([{"bad": "pickle"}], dtype=object))
    result = run_cli("repair", "--arrays", arrays, "--spec", spec, "--output", tmp_path / "bad", ok=False)
    assert "plain numeric geometry" in result["error"]
    assert not (tmp_path / "bad").exists()


def test_cli_rejects_forged_huge_npy_shape_before_numpy_allocates(tmp_path, monkeypatch):
    from mlx2.experimental.weight_space.cli import _arrays

    payload = io.BytesIO()
    np.lib.format.write_array_header_1_0(payload, {
        "descr": "<f4", "fortran_order": False, "shape": (1_000_000_000,),
    })
    payload.write(b"1234")
    path = tmp_path / "forged.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("weights.npy", payload.getvalue())

    def forbidden(*_, **__):
        raise AssertionError("NumPy allocation reached before header preflight")

    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="declared geometry"):
        _arrays(path)


def test_bfloat16_rounding_ties_to_even_and_fused_difference():
    halfway = np.array([1 + 1 / 256, 1 + 3 / 256], dtype=np.float32)
    np.testing.assert_array_equal(round_bfloat16(halfway), [1, 1 + 2 / 128])
    p = create_program(
        {"layer": (np.array([[1.0]], dtype=np.float32), np.array([[0.003]], dtype=np.float32))},
        checkpoint=CHECKPOINT, source_revision=REVISION,
        construction="synthetic_cpu_test", provenance={"source": "BF16 arithmetic fixture"},
    )
    result = evaluate_projections(p, {"layer": np.array([[1.0]])}, {"layer": np.array([[3.0]])}, simulate_bfloat16=True)
    assert result["targets"][0]["float32"]["dynamic_vs_fused"]["max_absolute_error"] < 1e-6
    assert result["targets"][0]["simulated_bfloat16"]["dynamic_vs_fused"]["max_absolute_error"] > 0


def test_projection_geometry_is_checked_before_arithmetic():
    p = create_program(
        {"layer": (np.ones((3, 1)), np.ones((1, 2)))}, checkpoint=CHECKPOINT,
        source_revision=REVISION, construction="synthetic_cpu_test", provenance={"source": "test"},
    )
    with pytest.raises(ValueError, match="dimensions"):
        evaluate_projections(p, {"layer": np.ones((2, 3))}, {"layer": np.ones((2, 4))})


def test_cli_abliteration_is_cpu_only_and_explicitly_gated(tmp_path):
    arrays = tmp_path / "direction.npz"
    np.savez(
        arrays, w=np.array([[1, 2], [3, 4]], dtype=np.float32),
        reference=np.array([[1, 0], [1, 0]], dtype=np.float32),
        contrast=np.array([[1, 1], [1, 1]], dtype=np.float32),
    )
    spec = tmp_path / "abliteration.json"
    spec.write_text(json.dumps({
        "checkpoint": CHECKPOINT, "source_revision": REVISION,
        "provenance": {"source": "synthetic CPU fixture"},
        "profiles": {"mlp_output": {"peak_layer": 2, "radius": 0, "peak_strength": 0.5}},
        "targets": {"layer": {"weight": "w", "reference": "reference", "contrast": "contrast", "layer": 2, "component": "mlp_output"}},
        "orthogonalize_reference": True,
    }))
    result = run_cli("abliterate", "--spec", spec, "--arrays", arrays, "--output", tmp_path / "ablated")
    assert result["construction"] == "contrastive_directional_projection"
    assert result["provenance"]["targets"]["layer"]["strength"] == 0.5
    assert result["provenance"]["behavioral_quality_evaluated"] is False
