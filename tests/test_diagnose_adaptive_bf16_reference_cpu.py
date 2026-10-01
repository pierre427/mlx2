"""Tensor-free CLI guards and strictly CPU target-reference diagnostics."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_map
from test_standard_xpress_serving_cpu import tiny

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/diagnose_adaptive_bf16_reference.py"
)


def load():
    spec = importlib.util.spec_from_file_location(
        "adaptive_reference_diagnostic", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli(tmp_path, *flags):
    shadow = tmp_path / "mlx"
    shadow.mkdir(exist_ok=True)
    (shadow / "__init__.py").write_text(
        "raise AssertionError('tensor import in dry-run')"
    )
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "/missing/model",
            "--adaptive-receipt",
            "/missing/input.json",
            "--out",
            str(tmp_path / "out.json"),
            *flags,
        ],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )


def test_dry_run_is_tensor_free_and_never_reads_receipt_or_allocates_model(tmp_path):
    result = cli(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["max_contexts"] == 4 and not plan["will_execute"]
    assert not plan["qualified"] and not plan["performance_claim"]
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize(
    "flags,error",
    [
        ([], "--i-own-the-gpu"),
        (["--dry-run", "--deadline-seconds", "901"], "deadline-seconds"),
        (["--dry-run", "--max-contexts", "15"], "invalid choice"),
    ],
)
def test_cli_refuses_without_ownership_or_beyond_bounds(tmp_path, flags, error):
    result = cli(tmp_path, *flags)
    assert result.returncode != 0 and error in result.stderr
    assert "tensor import" not in result.stderr
    assert not (tmp_path / "out.json").exists()


def test_input_artifact_or_receipt_cannot_be_overwritten():
    script = load()
    for destination in ("/model/config.json", "/input.json"):
        args = script.parser().parse_args(
            [
                "--model",
                "/model",
                "--adaptive-receipt",
                "/input.json",
                "--out",
                destination,
                "--dry-run",
            ]
        )
        with pytest.raises(ValueError, match="overwrite"):
            script.preflight(args)


def receipt():
    evidence = [
        {
            "uid": uid,
            "request_index": uid,
            "prompt_tokens": [1, 2],
            "actual_tokens": [3, 4, 5],
        }
        for uid in range(2)
    ]
    mismatch = [
        {
            "uid": uid,
            "request_index": uid,
            "position": position,
            "context_tokens": [1, 2] + [3, 4, 5][:position],
            "largest_error_token_id": 1,
            "actual_probability": 0.2,
            "expected_probability": 0.3,
        }
        for uid in range(2)
        for position in range(3)
    ]
    return {
        "generation_checks": [
            {
                "temperature": 0.8,
                "token_evidence": evidence,
                "sampled_law_mismatches": mismatch,
            }
        ]
    }


def test_context_selection_spans_requests_and_checks_exact_delivered_prefix():
    script = load()
    selected = script.select_contexts(receipt(), 4)
    assert [(c["request_index"], c["position"]) for c in selected] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (0, 2),
    ]
    bad = receipt()
    bad["generation_checks"][0]["sampled_law_mismatches"][0]["context_tokens"] = [1, 3]
    with pytest.raises(ValueError, match="delivered token prefix"):
        script.select_contexts(bad, 4)
    bad = receipt()
    bad["generation_checks"][0]["token_evidence"][0]["actual_tokens"][0] = True
    with pytest.raises(ValueError, match="invalid or unbounded"):
        script.select_contexts(bad, 4)
    with pytest.raises(ValueError, match="no sampled"):
        script.select_contexts({}, 4)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_actual_cpu_bf16_target_same_frozen_prefix_and_flag_restore(position):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        script = load()
        model, _ = tiny()
        model.update(
            tree_map(lambda value: value.astype(mx.bfloat16), model.parameters())
        )
        prompt, actual = [1, 2, 3, 4], [5, 6, 7, 8]
        serving, _, _ = script.reference(
            model, mx, prompt, actual, position, 128, whole=False
        )
        whole, _, _ = script.reference(
            model, mx, prompt, actual, position, 128, whole=True
        )
        context = {
            "uid": 0,
            "request_index": 0,
            "position": position,
            "prompt_tokens": prompt,
            "actual_tokens": actual,
            "context_tokens": prompt + actual[:position],
            "largest_error_token_id": 0,
            "actual_probability": float(
                script.probability(serving[0, -1].astype(mx.float32))[0]
            ),
            "expected_probability": float(
                script.probability(whole[0, -1].astype(mx.float32))[0]
            ),
        }
        original_modules = [(n, id(m)) for n, m in model.named_modules()]
        result = script.diagnose_context(model, mx, context, 128)
        assert (
            result["passed"]
            and result["same_frozen_prefix_row_exact_vs_original_s1"]["bitwise_equal"]
        )
        assert result["same_frozen_prefix_reached_row_vs_serving"]["bitwise_equal"]
        assert result["target_flag_restored"] and not model._target_verify_row_exact
        assert original_modules == [(n, id(m)) for n, m in model.named_modules()]
        assert result["row_exact_input_tokens"][0] == (prompt + actual[:position])[-1]
        bad = copy.deepcopy(context)
        bad["actual_probability"] = 1.0
        assert not script.diagnose_context(model, mx, bad, 128)["passed"]
    finally:
        mx.set_default_device(previous)
