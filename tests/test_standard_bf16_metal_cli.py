"""Tensor-free admission and a complete original diagnostic on tiny CPU BF16."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_standard_bf16_metal.py"
spec = importlib.util.spec_from_file_location("standard_bf16_diagnostic", SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def args(*extra):
    return diagnostic.parser().parse_args(
        [
            "--model",
            "/missing-target",
            "--draft",
            "/missing-draft",
            "--out",
            "/missing-output",
            *extra,
        ]
    )


def test_gpu_ownership_required():
    with pytest.raises(ValueError, match="i-own-the-gpu"):
        diagnostic.preflight(args())


@pytest.mark.parametrize("value", ["0", "901"])
def test_deadline_bounded(value):
    with pytest.raises(ValueError):
        diagnostic.preflight(args("--dry-run", "--deadline-seconds", value))


def test_tensor_free_dryrun():
    code = """
import sys,runpy
class Guard:
 def find_spec(self,fullname,*args,**kwargs):
  if fullname.startswith(('mlx','numpy','transformers','psutil')):raise RuntimeError(fullname)
sys.meta_path.insert(0,Guard())
sys.argv=[sys.argv[1],'--model','/missing-target','--draft','/missing-draft','--out','/missing-output','--dry-run']
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
    assert plan["will_execute"] is False and plan["production_math_changed"] is False
    assert plan["captured_verify_block"] == 2 and plan["num_draft"] == 15


def test_numeric_delta_reports_rms_and_rejects_shape_mismatch():
    assert diagnostic.delta([1.0, 2.0], [1.0, 4.0]) == {
        "equal": False,
        "max_abs": 2.0,
        "rms": pytest.approx(2**0.5),
    }
    with pytest.raises(ValueError):
        diagnostic.delta([1], [1, 2])


@pytest.mark.parametrize("tied", [False, True])
def test_full_diagnostic_runs_tiny_cpu_bf16_restores_patches(monkeypatch, tied):
    import mlx.core as mx
    from mlx import nn
    from test_standard_xpress_serving_cpu import tiny

    from mlx2.adapters import standard_decoder as adapter_module
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.models import standard_decoder as model_module

    sys.path.insert(0, str(SCRIPT.parent))
    import validate_xpress_metal_matrix as helpers

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    model, draft = tiny(tied)
    model.apply(lambda value: value.astype(mx.bfloat16))
    draft.apply(lambda value: value.astype(mx.bfloat16))
    mx.eval(model.parameters(), draft.parameters())
    linear = nn.Linear.__call__
    embedding_linear = nn.Embedding.as_linear
    attention = model_module.scaled_dot_product_attention

    class Adapter:
        def __init__(self, *args, **kwargs):
            self.identity = {"fingerprint": "cpu-test"}
            self.model = model
            self.draft_model = draft

        def create_external_batch(self, **kwargs):
            return ExternalDraftBatchGenerator(
                model, draft_model=draft, binding="cpu-test", num_draft=3, **kwargs
            )

        def prompt_tokens(self, request):
            return [1, 2, 3, 4]

        def close(self):
            pass

    # The diagnostic's GPU selection is replaced with CPU; hard assert every
    # device assignment stays CPU. No Metal kernel or model run occurs here.
    original_set = mx.set_default_device

    def cpu_only(device):
        assert device == mx.cpu
        return original_set(mx.cpu)

    monkeypatch.setattr(mx, "gpu", mx.cpu)
    monkeypatch.setattr(mx, "set_default_device", cpu_only)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {
            "device_name": "M3 CPU test",
            "max_recommended_working_set_size": 100 * 2**30,
        },
    )
    monkeypatch.setattr(
        helpers, "float32_artifact_forecast", lambda paths: (1000, 10, {"BF16": 250})
    )
    monkeypatch.setattr(adapter_module, "StandardDecoderAdapter", Adapter)
    report = diagnostic.preflight(args("--i-own-the-gpu"))
    try:
        diagnostic.run(args("--i-own-the-gpu"), report)
        assert report["passed"] and report["temporary_patches_restored"]
        assert report["trace_reproduces_actual_logits"]["equal"]
        assert len(report["comparisons"]) == 12
        assert report["scalar_vector_rope"]["equal"]
        assert len(report["fixed_input_projection_controls"]) == 8
        assert all(frame["per_row"] for frame in report["comparisons"])
        assert nn.Linear.__call__ is linear
        assert nn.Embedding.as_linear is embedding_linear
        assert model_module.scaled_dot_product_attention is attention
    finally:
        original_set(previous)
        sys.path.remove(str(SCRIPT.parent))
