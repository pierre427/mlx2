"""Synthetic mixed-format route/reuse oracle; no Metal or model qualification.

Q1 is unpacked on CPU solely to exercise the provider's dispatch and assembly.
This does not test the Metal Q1 kernel or compare Q1 against original weights.
"""
from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.runtime.gguf import source_q1
from mlx2.runtime.models import deepseek_v41_quant as qat
from mlx2.runtime.models.deepseek_v4 import limited_swiglu


def _q1_cpu(x, weight, scales, biases):
    packed = np.array(weight)
    bits = ((packed[..., None] >> np.arange(32, dtype=np.uint32)) & 1).reshape(
        packed.shape[0], -1
    ).astype(np.float32)
    dense = bits * np.repeat(np.array(scales).astype(np.float32), 64, axis=1)
    dense += np.repeat(np.array(biases).astype(np.float32), 64, axis=1)
    return x.astype(mx.float32) @ mx.array(dense).T


def _fixture(rows, dtype, owned):
    rng = np.random.default_rng(410251)
    hidden, intermediate, topk = 64, 128, 3
    bank, identities = {}, {}
    for expert in range(4):
        data = {}
        for kind in ("gate", "up", "down"):
            n, k = (hidden, intermediate) if kind == "down" else (intermediate, hidden)
            q1 = expert % 2 == 1
            data[kind + ".qweight"] = mx.array(
                rng.integers(0, 2**32, (n, k // (32 if q1 else 8)), dtype=np.uint32)
            )
            if q1:
                data[kind + ".scales"] = mx.array(rng.uniform(.01, .05, (n, k // 64)).astype(np.float16))
                data[kind + ".biases"] = mx.array(rng.uniform(-.04, -.01, (n, k // 64)).astype(np.float16))
            else:
                data[kind + ".scales"] = mx.array(rng.integers(117, 124, (n, k // 32), dtype=np.uint8))
            identities[id(data[kind + ".qweight"])] = (expert, kind, "q1" if q1 else "original")
        bank[expert] = data
    patterns = np.array([[3, 0, 2], [1, 3, 0], [2, 1, 3], [2, 0, 1]], np.uint32)
    ids = patterns[np.arange(rows) % len(patterns)]
    values = rng.normal(size=(rows, hidden)).astype(np.float32)
    values[0, :32] = 0  # A complete zero QAT block, alongside a nonzero block.
    x = mx.array(values).astype(dtype)
    scores = mx.array(rng.uniform(.05, .95, (rows, topk)).astype(np.float32))
    acquired, observed, windows = [], [], []
    store = SimpleNamespace(_active=not owned, stats={"lookups": 0})

    def acquire(prefix, expert):
        acquired.append((prefix, expert))
        return bank[expert]

    @contextmanager
    def window():
        assert not store._active
        store._active = True
        windows.append("enter")
        try:
            yield
        finally:
            store._active = False
            windows.append("exit")

    store.acquire, store.window = acquire, window
    store.observe_routes = lambda prefix, route: observed.append((prefix, route.copy()))
    weights = object.__new__(SourceV41Weights)
    weights._ready = lambda: None
    weights.artifact = {"args": SimpleNamespace(hidden_size=hidden, num_experts_per_tok=topk)}
    weights.expert_stream, weights.scheduled_expert_calls = store, {}
    return weights, bank, identities, x, ids, scores, acquired, observed, windows


def _repeated_qat_reference(bank, x, ids, scores):
    """Prior mixed projection law, with independent per-row route assembly."""
    rounded = lambda v: v.astype(mx.bfloat16).astype(mx.float32)
    outputs = {}
    for expert in np.unique(ids):
        row_ids, slots = np.where(ids == expert)
        data = bank[int(expert)]
        selected = x[mx.array(row_ids, mx.uint32)]

        def project(value, kind):
            activation = qat.quantize(value, "fp8")
            if kind + ".biases" in data:
                return _q1_cpu(activation, data[kind + ".qweight"], data[kind + ".scales"], data[kind + ".biases"])
            return mx.quantized_matmul(activation.astype(mx.float32), data[kind + ".qweight"],
                                       scales=data[kind + ".scales"], mode="mxfp4")

        gate, up = rounded(project(selected, "gate")), rounded(project(selected, "up"))
        middle = rounded(limited_swiglu(gate, up) * scores[mx.array(row_ids), mx.array(slots), None])
        result = rounded(project(middle, "down"))
        for index, row in enumerate(row_ids):
            outputs[(int(row), int(expert))] = result[index]
    result = []
    for row, route in enumerate(ids):
        total = mx.zeros((x.shape[1],), mx.float32)
        for expert in sorted(route):
            total = total + outputs[(row, int(expert))]
        result.append(total)
    return mx.stack(result)


@pytest.mark.parametrize("rows", [1, 7, 64])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("owned", [False, True])
def test_mixed_reference_qat_reused_and_routes_restored(rows, dtype, owned, monkeypatch):
    def no_metal(*args, **kwargs):
        raise AssertionError("CPU synthetic test must not construct a Metal kernel")

    monkeypatch.setattr(mx.fast, "metal_kernel", no_metal)
    with mx.stream(mx.cpu):
        weights, bank, identities, x, ids, scores, acquired, observed, windows = _fixture(rows, dtype, owned)
        expected = _repeated_qat_reference(bank, x, ids, scores)
        mx.eval(expected)
        quantizations, projections = [], []
        reference_quantize, reference_qmm = qat.quantize, mx.quantized_matmul

        def quantize(value, kind):
            quantizations.append((value.shape, kind))
            return reference_quantize(value, kind)

        def qmm(activation, weight, **kwargs):
            projections.append((identities[id(weight)], activation))
            return reference_qmm(activation, weight, **kwargs)

        def q1(activation, weight, scales, biases):
            projections.append((identities[id(weight)], activation))
            return _q1_cpu(activation, weight, scales, biases)

        monkeypatch.setattr(qat, "quantize", quantize)
        monkeypatch.setattr(mx, "quantized_matmul", qmm)
        monkeypatch.setattr(source_q1, "matmul", q1)
        actual = weights.moe(x, ids, scores, "blk.0")
        mx.eval(actual)
        assert np.array(actual).tobytes() == np.array(expected).tobytes()
        experts = [int(v) for v in np.unique(ids)]
        assert acquired == [("blk.0", expert) for expert in experts]
        assert len(observed) == 1 and observed[0][0] == "blk.0"
        assert np.array_equal(observed[0][1], ids)
        assert len(quantizations) == 2 * len(experts)
        assert all(kind == "fp8" for _, kind in quantizations)
        assert len(projections) == 3 * len(experts)
        assert {identity[2] for identity, _ in projections} == {"original", "q1"}
        for i, expert in enumerate(experts):
            group = projections[3 * i:3 * i + 3]
            assert [(identity[0], identity[1]) for identity, _ in group] == [
                (expert, "gate"), (expert, "up"), (expert, "down")
            ]
            assert group[0][1] is group[1][1]
            assert group[2][1] is not group[0][1]
            assert all(activation.dtype == mx.float32 for _, activation in group)
        assert weights.expert_stream.stats["lookups"] == 1
        assert windows == (["enter", "exit"] if owned else [])
        assert weights.expert_stream._active is (not owned)
