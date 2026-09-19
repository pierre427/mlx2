from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
import pytest

from mlx2.runtime.models import qwen4_fused_moe, qwen4_moe_router


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_qwen4_moe_m3_candidate.py"


class _Array:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def _m3_down_inputs():
    return (
        _Array((1, 3, 10, 640), mx.bfloat16),
        _Array((1, 3, 10), mx.uint32),
        _Array((1, 3, 10), mx.bfloat16),
        _Array((512, 2560, 80), mx.uint32),
        _Array((512, 2560, 10), mx.bfloat16),
        _Array((512, 2560, 10), mx.bfloat16),
    )


def _pretend_metal(monkeypatch, module):
    monkeypatch.setattr(module.mx, "default_device", lambda: module.mx.gpu)
    monkeypatch.setattr(module.mx.metal, "is_available", lambda: True)


def test_m3_fused_down_is_default_closed_and_explicitly_candidate(monkeypatch):
    _pretend_metal(monkeypatch, qwen4_fused_moe)
    inputs = _m3_down_inputs()

    closed = qwen4_fused_moe.admit_qwen4_fused_down(*inputs)
    assert not closed.accepted
    assert "not qualified" in closed.reason

    candidate = qwen4_fused_moe.admit_qwen4_fused_down(
        *inputs, candidate_token_widths=(3,)
    )
    assert candidate.accepted
    assert candidate.tokens == 3
    assert qwen4_fused_moe.QUALIFIED_TOKEN_WIDTHS == (1,)
    assert qwen4_fused_moe.CANDIDATE_TOKEN_WIDTHS == (3,)


def test_fused_down_rejects_invented_candidate_width():
    with pytest.raises(ValueError, match="unsupported fused-down candidate"):
        qwen4_fused_moe.admit_qwen4_fused_down(
            *_m3_down_inputs(), candidate_token_widths=(7,)
        )


def test_m3_router_is_default_closed_and_explicitly_candidate(monkeypatch):
    _pretend_metal(monkeypatch, qwen4_moe_router)
    gates = _Array((1, 3, 512), mx.bfloat16)

    closed = qwen4_moe_router.admit_qwen4_moe_router(
        gates, top_k=10, norm_topk_prob=True
    )
    assert not closed.accepted
    assert "not qualified" in closed.reason

    candidate = qwen4_moe_router.admit_qwen4_moe_router(
        gates,
        top_k=10,
        norm_topk_prob=True,
        candidate_token_widths=(3,),
    )
    assert candidate.accepted
    assert "thread_position_in_grid.z" in qwen4_moe_router._SOURCE
    assert "route_base + j" in qwen4_moe_router._SOURCE
    assert qwen4_moe_router.QUALIFIED_TOKEN_WIDTHS == (1,)

    production = qwen4_moe_router.admit_qwen4_moe_router(
        _Array((1, 1, 512), mx.bfloat16), top_k=10, norm_topk_prob=True
    )
    assert production.accepted
    odd_shape = qwen4_moe_router.admit_qwen4_moe_router(
        _Array((3, 1, 512), mx.bfloat16),
        top_k=10,
        norm_topk_prob=True,
        candidate_token_widths=(3,),
    )
    assert not odd_shape.accepted


def test_describe_is_json_and_does_not_require_a_gpu():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--describe"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["schema"] == "mlx2.qwen4-moe-m3-candidate-benchmark.v2"
    assert report["status"] == "implemented_candidate_unqualified"
    assert report["token_width"] == 3
    assert report["engagement_gate"] == [
        "candidate_router_calls > 0",
        "candidate_down_calls > 0",
    ]


def test_gpu_run_refuses_without_explicit_lease_attestation(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-gpu", "--output", str(tmp_path / "x.json")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--run-gpu requires --i-hold-gpu-lease" in result.stderr
    assert not (tmp_path / "x.json").exists()


def test_benchmark_module_keeps_counterbalanced_and_parity_metadata():
    spec = importlib.util.spec_from_file_location("qwen4_m3_bench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.METADATA["arms"]["stock"].startswith("precise softmax")
    assert "tile4" in module.METADATA["arms"]["candidate"]
    assert "not route qualification" in module.METADATA["qualification"]
