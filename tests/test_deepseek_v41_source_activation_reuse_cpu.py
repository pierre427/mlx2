"""Compare routed activation reuse with the original repeated-QAT graph."""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.runtime.models.deepseek_v4 import limited_swiglu
from mlx2.runtime.models import deepseek_v41_quant as qat


def _fixture(rows, *, hidden=32, intermediate=64, experts=4, topk=2):
    rng = np.random.default_rng(41025)
    bank = {}
    for expert in range(experts):
        data = {}
        for kind in ("gate", "up", "down"):
            n, k = (hidden, intermediate) if kind == "down" else (intermediate, hidden)
            data[kind + ".qweight"] = mx.array(rng.integers(0, 2**32, (n, k // 8), dtype=np.uint32))
            data[kind + ".scales"] = mx.array(rng.integers(117, 125, (n, k // 32), dtype=np.uint8))
        bank[expert] = data
    ids = np.array([rng.choice(experts, topk, replace=False) for _ in range(rows)], np.uint32)
    x = mx.array(rng.normal(size=(rows, hidden)).astype(np.float32))
    scores = mx.array(rng.uniform(.1, .9, (rows, topk)).astype(np.float32))
    store = SimpleNamespace(_active=True, acquire=lambda prefix, expert: bank[expert], stats={"lookups": 0})
    weights = object.__new__(SourceV41Weights)
    weights._ready = lambda: None
    weights.artifact = {"args": SimpleNamespace(hidden_size=hidden, num_experts_per_tok=topk)}
    weights.expert_stream = store
    weights.scheduled_expert_calls = {}
    return weights, bank, x, ids, scores


def _original(bank, x, ids, scores):
    """Original three-QATs-per-expert law, grouped at the same row shape."""
    rounded = lambda v: v.astype(mx.bfloat16).astype(mx.float32)
    outputs = {}
    for expert in np.unique(ids):
        row_ids, slot_ids = np.where(ids == expert)
        selected = x[mx.array(row_ids, mx.uint32)]
        data = bank[int(expert)]

        def project(value, kind):
            return mx.quantized_matmul(qat.quantize(value, "fp8").astype(mx.float32),
                                      data[kind + ".qweight"], scales=data[kind + ".scales"], mode="mxfp4")

        gate = rounded(project(selected, "gate"))
        up = rounded(project(selected, "up"))
        middle = rounded(limited_swiglu(gate, up) * scores[mx.array(row_ids), mx.array(slot_ids), None])
        result = rounded(project(middle, "down"))
        for i, row in enumerate(row_ids):
            outputs[(int(row), int(expert))] = result[i]
    rows = []
    for row, route in enumerate(ids):
        value = mx.zeros((x.shape[1],), mx.float32)
        for expert in sorted(route):
            value = value + outputs[(row, int(expert))]
        rows.append(value)
    return mx.stack(rows)


@pytest.mark.parametrize("rows", [1, 7, 64])
def test_exact_cpu_moe_and_one_gate_up_activation(rows, monkeypatch):
    with mx.stream(mx.cpu):
        weights, bank, x, ids, scores = _fixture(rows)
        expected = _original(bank, x, ids, scores)
        mx.eval(expected)
        quantizations, projections = [], []
        original_quantize, original_qmm = qat.quantize, mx.quantized_matmul

        def quantize(value, kind):
            quantizations.append((value.shape, kind))
            return original_quantize(value, kind)

        def qmm(value, *args, **kwargs):
            projections.append(value)
            return original_qmm(value, *args, **kwargs)

        monkeypatch.setattr(qat, "quantize", quantize)
        monkeypatch.setattr(mx, "quantized_matmul", qmm)
        actual = weights.moe(x, ids, scores, "blk.0")
        mx.eval(actual)
        assert np.array(actual).tobytes() == np.array(expected).tobytes()
        assert len(quantizations) == 2 * len(np.unique(ids))
        assert len(projections) == 3 * len(np.unique(ids))
        assert all(projections[i] is projections[i + 1] for i in range(0, len(projections), 3))
        assert all(kind == "fp8" for _, kind in quantizations)
