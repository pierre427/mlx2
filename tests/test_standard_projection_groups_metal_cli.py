"""Tensor-free CLI admission and isolated CPU tests of the diagnostic."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/diagnose_standard_projection_groups_metal.py"
)
spec = importlib.util.spec_from_file_location("projection_diagnostic", SCRIPT)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def args(*extra):
    return diag.parser().parse_args(
        ["--model", "/missing-model", "--out", "/missing-output", *extra]
    )


def test_missing_gpu_ownership_and_bounded_cases():
    with pytest.raises(ValueError, match="i-own-the-gpu"):
        diag.preflight(args())
    for key, value in (
        ("rows", "15"),
        ("rows", "33"),
        ("queries", "2"),
        ("queries", "17"),
        ("repetitions", "4"),
        ("timeout-seconds", "901"),
    ):
        with pytest.raises(ValueError):
            diag.preflight(args("--dry-run", "--" + key, value))


def test_dry_run_never_imports_tensor_libraries():
    code = """
import sys,runpy
class Guard:
 def find_spec(self,fullname,*args,**kwargs):
  if fullname.startswith(('mlx','numpy','transformers','psutil')):raise RuntimeError(fullname)
sys.meta_path.insert(0,Guard())
sys.argv=[sys.argv[1],'--model','/missing-model','--out','/missing-output','--dry-run','--include-vmap']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["groups"] == [1, 2, 4, 8, 16] and plan["rows"] == 32
    assert plan["full_law_geometry"] == [15, 4]
    assert plan["layouts"] == ["sequence", "batch", "vmap"]
    assert (
        not plan["will_execute"]
        and not plan["qualified"]
        and not plan["production_math_changed"]
    )


@pytest.mark.parametrize("layout", ["sequence", "batch", "vmap"])
def test_groups_preserve_values_row_order_rank_and_tail(layout):
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    values = mx.arange(24).reshape(3, 2, 4)
    operation = lambda value: mx.concatenate([value, value * 2], axis=-1)
    expected = operation(values)
    for group in diag.GROUPS:
        actual = diag.grouped_projection(operation, values, group, layout)
        assert actual.shape == (3, 2, 8) and mx.array_equal(actual, expected).item()
    with pytest.raises(ValueError):
        diag.grouped_projection(operation, values, 3, layout)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("fault", [False, True])
def test_complete_diagnostic_real_cpu_bf16_target_and_restoration(
    monkeypatch, tied, fault
):
    import mlx.core as mx
    from mlx import nn
    from test_standard_xpress_serving_cpu import tiny

    from mlx2.adapters import standard_decoder as adapters
    from mlx2.runtime.models import standard_decoder as standard

    mx.set_default_device(mx.cpu)
    model, _ = tiny(tied)
    model.apply(lambda value: value.astype(mx.bfloat16))
    mx.eval(model.parameters())
    old_linear, old_embedding, old_project = (
        nn.Linear.__call__,
        nn.Embedding.as_linear,
        standard._project,
    )

    class Adapter:
        def __init__(self, *a, **kw):
            self.model = model
            self.identity = {"fingerprint": "cpu-tiny"}
            self.closed = False

        def prompt_tokens(self, request):
            return (
                [1, 2, 3, 4]
                if "French" in request["messages"][0]["content"]
                else [2, 4, 6, 3]
            )

        def close(self):
            self.closed = True

    adapter = Adapter()
    monkeypatch.setattr(adapters, "StandardDecoderAdapter", lambda *a, **kw: adapter)
    sys.path.insert(0, str(SCRIPT.parent))
    import validate_adaptive_metal as adaptive
    import validate_xpress_metal_matrix as matrix

    monkeypatch.setattr(
        matrix, "float32_artifact_forecast", lambda paths: (1000, 10, {"BF16": 500})
    )
    monkeypatch.setattr(
        adaptive, "artifact_identity", lambda path: {"cpu_fixture": True}
    )
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {
            "device_name": "M3 CPU unit substitute",
            "max_recommended_working_set_size": 100 << 30,
        },
    )
    original_set = mx.set_default_device
    monkeypatch.setattr(mx, "gpu", mx.cpu)

    def cpu_only(device):
        assert device == mx.cpu
        return original_set(mx.cpu)

    monkeypatch.setattr(mx, "set_default_device", cpu_only)
    selected = args(
        "--i-own-the-gpu", "--rows", "16", "--repetitions", "1", "--include-vmap"
    )
    report = diag.preflight(selected)
    if fault:
        original_grouped = diag.grouped_projection
        calls = [0]

        def fail_later(operation, value, group, layout):
            if value.shape[:2] == (15, 4):
                calls[0] += 1
                if calls[0] == 8:
                    raise RuntimeError("injected later target projector failure")
            return original_grouped(operation, value, group, layout)

        monkeypatch.setattr(diag, "grouped_projection", fail_later)
        with pytest.raises(RuntimeError, match="later target projector"):
            diag.run(selected, report)
        assert report["temporary_hooks_restored"] and adapter.closed
        assert nn.Linear.__call__ is old_linear
        assert nn.Embedding.as_linear is old_embedding
        assert standard._project is old_project
        assert model._target_verify_row_exact is False
        return
    diag.run(selected, report)
    assert report["passed"] and report["temporary_hooks_restored"] and adapter.closed
    assert (
        nn.Linear.__call__ is old_linear
        and nn.Embedding.as_linear is old_embedding
        and standard._project is old_project
    )
    assert len(report["projector_results"]) == 7 * len(model.layers) + 1
    assert all(item["equal"] for item in report["capture_matches_untraced_original_S1"])
    assert all(len(item["results"]) == 15 for item in report["projector_results"])
    assert all(item["results"][0]["equal"] for item in report["projector_results"])
    assert all(
        item["results"][0]["b15_fixed_input"]["shape"][:2] == [15, 1]
        for item in report["projector_results"]
    )
    check = report["frozen_cache_full_law_check"]
    assert check["query_geometry"] == [15, 4] and len(check["rows"]) == 60
    assert check["all_greedy_equal"] and not check["qualified"]
    assert not report["production_math_changed"] and not report["performance_claim"]


def test_native_bits_distinguishes_signed_zero_without_float_expansion():
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    positive = mx.array([0.0], dtype=mx.bfloat16)
    negative = mx.array([-0.0], dtype=mx.bfloat16)
    assert mx.array_equal(positive, negative).item()
    assert diag.native_bits(positive) != diag.native_bits(negative)
