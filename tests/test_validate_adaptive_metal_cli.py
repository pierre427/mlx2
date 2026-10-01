"""CPU-only CLI gates and an actual tiny runtime exercise of diagnostic logic."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_adaptive_metal.py"


def load():
    spec = importlib.util.spec_from_file_location("adaptive_metal_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli(tmp_path, *extra):
    # An import shadow proves dry-run and validation errors never import MLX.
    shadow = tmp_path / "mlx"
    shadow.mkdir(exist_ok=True)
    (shadow / "__init__.py").write_text(
        "raise AssertionError('MLX import in CPU-only CLI')"
    )
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--target",
            "/missing/target",
            "--draft",
            "/missing/head",
            "--out",
            str(tmp_path / "receipt.json"),
            *extra,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_dry_run_prints_complete_cost_grid_without_importing_mlx(tmp_path):
    result = cli(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["depths"] == list(range(16)) and plan["cohorts"] == [1, 2, 4]
    assert plan["model_loaded"] is False and plan["performance_qualified"] is False
    assert not (tmp_path / "receipt.json").exists()
    assert plan["target_verify_row_exact"] is False


def test_row_exact_option_is_explicit_tensor_free_and_strict(tmp_path):
    selected = cli(tmp_path, "--dry-run", "--target-verify-row-exact")
    assert selected.returncode == 0, selected.stderr
    assert json.loads(selected.stdout)["target_verify_row_exact"] is True
    invalid = cli(tmp_path, "--dry-run", "--target-verify-row-exact", "false")
    assert invalid.returncode != 0 and "unrecognized arguments" in invalid.stderr
    script = load()
    args = script.parser().parse_args(
        ["--target", "/a", "--draft", "/b", "--out", "/c", "--dry-run"]
    )
    args.target_verify_row_exact = 1
    with pytest.raises(ValueError, match="must be boolean"):
        script.plan(args)


@pytest.mark.parametrize(
    "extra,message",
    [
        ([], "--i-own-the-gpu required"),
        (["--dry-run", "--timeout", "901"], "timeout must be"),
        (["--dry-run", "--num-draft", "16"], "num_draft must be"),
        (["--dry-run", "--repetitions", "4"], "repetitions must be"),
        (["--dry-run", "--requests", "3"], "at least four"),
    ],
)
def test_cli_rejects_ownership_or_unbounded_inputs_before_import(
    tmp_path, extra, message
):
    result = cli(tmp_path, *extra)
    assert result.returncode != 0 and message in result.stderr
    assert "MLX import" not in result.stderr


def test_audit_rejects_geometry_or_censored_label_receipt_mismatch():
    script = load()
    frames = [{"adaptive": True, "shapes": [[2, 4], [1, 2]], "saved": 3}]
    stats = {
        "external_adaptive_target_rows": 10,
        "external_adaptive_trimmed_target_rows": 3,
    }
    good = script.audit_trace(frames, stats, [3, 1, 0], [3, 1, 0])
    assert good["mixed_depth_rounds"] == 1 and good["observed_used"]
    with pytest.raises(AssertionError, match="geometry"):
        script.audit_trace(
            frames, {**stats, "external_adaptive_target_rows": 12}, [3, 1, 0], [3, 1, 0]
        )
    with pytest.raises(AssertionError, match="reached rejection"):
        script.audit_trace(frames, stats, [3, 1, 0], [3, 1, 1])


@pytest.mark.parametrize("corrupt_first_cell", [False, True])
def test_full_diagnostic_logic_with_actual_tiny_cpu_model_explicit_metal_substitute(
    tmp_path, monkeypatch, corrupt_first_cell
):
    from test_standard_xpress_serving_cpu import tiny

    from mlx2.adapters import standard_decoder
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    mx.set_default_device(mx.cpu)
    model, draft = tiny()
    adapter = SimpleNamespace(
        model=model,
        draft_model=draft,
        identity={"fingerprint": "cpu-tiny"},
        tokenizer=SimpleNamespace(encode=lambda text, **kw: [ord(c) % 8 for c in text]),
    )

    created = [0]

    def create(**kwargs):
        created[0] += 1
        e = ExternalDraftBatchGenerator(
            model, draft_model=draft, binding="cpu-tiny", num_draft=2, **kwargs
        )
        if corrupt_first_cell and created[0] == 4:
            original = e.next
            changed = [False]

            def corrupt():
                ready, responses = original()
                if responses and not changed[0]:
                    responses[0].token = (responses[0].token + 1) % 9
                    changed[0] = True
                return ready, responses

            e.next = corrupt
        return e

    adapter.create_external_batch = create
    monkeypatch.setattr(
        standard_decoder, "StandardDecoderAdapter", lambda *a, **kw: adapter
    )
    original_device = mx.set_default_device
    monkeypatch.setattr(
        mx, "set_default_device", lambda device: original_device(mx.cpu)
    )
    monkeypatch.setattr(mx, "synchronize", lambda: None)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(
        mx.metal, "device_info", lambda: {"device_name": "CPU unit substitute"}
    )
    script = load()
    monkeypatch.setattr(
        script, "artifact_identity", lambda path: {"cpu_unit_fixture": True}
    )
    args = script.parser().parse_args(
        [
            "--target",
            "/synthetic",
            "--draft",
            "/synthetic/head",
            "--out",
            str(tmp_path / "receipt.json"),
            "--num-draft",
            "2",
            "--context-tokens",
            "4",
            "--max-tokens",
            "4",
            "--requests",
            "4",
            "--repetitions",
            "1",
            "--min-observations",
            "1",
        ]
    )
    receipt = script.plan(args)
    script.run(args, receipt)
    assert receipt["passed"] is (not corrupt_first_cell) and receipt["model_loaded"]
    assert len(receipt["generation_checks"]) == 7
    assert all(
        c.get("ordinary_greedy_exact") for c in receipt["generation_checks"][1:-1]
    )
    assert [c["depth"] for c in receipt["backstop_checks"]] == [2, 2, 2]
    assert all(
        len(c["median_seconds"]) == 3 for c in receipt["cost_measurements"].values()
    )
    assert (
        receipt["generation_checks"][-1]["sampled_target_law_max_absolute_error"] < 1e-4
    )
    assert receipt["performance_qualified"] is False
    if corrupt_first_cell:
        first = receipt["generation_checks"][0]
        assert first["passed"] is False and first["ordinary_greedy_exact"] is False
        mismatch = first["token_evidence"][0]["first_mismatch"]
        assert mismatch["position"] == 0 and mismatch["actual"] != mismatch["expected"]
        assert (
            first["token_evidence"][0]["actual_tokens"]
            and first["token_evidence"][0]["expected_tokens"]
        )
        assert all(c["passed"] for c in receipt["generation_checks"][1:])
        assert all(c["passed"] for c in receipt["backstop_checks"])


def test_artifact_binding_detects_equal_size_equal_timestamp_corruption(tmp_path):
    script = load()
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"ABCD")
    (tmp_path / "config.json").write_text("{}")
    before = script.artifact_identity(tmp_path)
    stamp = weights.stat().st_mtime_ns
    weights.write_bytes(b"ABCE")
    os.utime(weights, ns=(stamp, stamp))
    after = script.artifact_identity(tmp_path)
    assert (
        before["files"]["model.safetensors"]["size"]
        == after["files"]["model.safetensors"]["size"]
    )
    assert (
        before["files"]["model.safetensors"]["sha256"]
        != after["files"]["model.safetensors"]["sha256"]
    )
