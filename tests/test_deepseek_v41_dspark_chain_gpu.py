"""GPU gate: the lockstep executor's multirow sites equal ordinary Source rows.

The originals are the real SourceV41Weights methods over synthetic coefficients
at production geometry. Bitwise per row; no checkpoint load.
"""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal GPU required")

from mlx2.adapters.deepseek_v41_dspark_chain import (
    SourceRowsExecutor,
    _Park,
)
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights


@pytest.fixture(autouse=True)
def _gpu_default():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class FakeSource(SourceV41Weights):
    """Real Source projection methods; synthetic coefficients, no artifact."""

    def __init__(self, qat, loads):
        self.source_dense_qat_policy = qat
        self.source_bf16_loads_policy = loads
        self.source_qakv_policy = "qat_reuse" if qat == "fused_fp32" else "reference"
        self.scheduled_dense_calls, self.scheduled_grouped_calls = {}, {}
        self._quant, self._dense = {}, {}
        key = mx.random.key(5)
        for name, (n, k) in {"blk.0.attn_output_b.weight": (5120, 8192),
                             "blk.0.ffn_gate_shexp.weight": (2304, 5120),
                             "blk.0.ffn_up_shexp.weight": (2304, 5120),
                             "blk.0.ffn_down_shexp.weight": (5120, 2304),
                             "blk.0.attn_q_a.weight": (1280, 5120), "blk.0.attn_kv.weight": (512, 5120),
                             "blk.0.attn_q_b.weight": (32768, 1280)}.items():
            key, sub = mx.random.split(key)
            q, s = mx.quantize(mx.random.normal((n, k), key=sub) * 0.05, mode="mxfp8")[:2]
            self._quant[name] = (q, s)
        self._dense["blk.0.ffn_gate_inp.weight"] = mx.random.normal((384, 5120), key=key) * 0.02
        head = (mx.random.normal((129280, 5120), key=mx.random.key(9)) * 0.02).astype(mx.bfloat16)
        self._dense["output.weight"] = head if loads == "direct_f32" else head.astype(mx.float32)
        self._dense["blk.0.attn_output_a.weight"] = (
            mx.random.normal((8192, 4096), key=mx.random.key(10)) * 0.02).astype(mx.bfloat16)
        mx.eval(self._quant, self._dense)

    def close(self):
        return None

    def _ready(self):
        return True

    def _ensure_dense(self):
        return True

    def array(self, name):
        return self._dense[name]


def _parks(weights, kind, name, rows, shape):
    xs = [mx.random.normal(shape, key=mx.random.key(r + 1)).astype(mx.bfloat16).astype(mx.float32)
          for r in range(rows)]
    if kind == "linear":
        apply = weights.linear(name)
        return [_Park(kind, name, (x,), {}, apply) for x in xs]
    if kind == "grouped":
        return [_Park(kind, name, (x, name), {"groups": 8}, weights.grouped) for x in xs]
    if kind == "paired":
        return [_Park(kind, name, (x, *name), {}, weights.paired_project) for x in xs]
    return [_Park(kind, name, (x, *name), {}, weights.gate_up) for x in xs]


def _bits(value):
    return np.array(value).view(np.uint32)


@pytest.mark.parametrize("qat", ["reference", "fused_fp32"])
@pytest.mark.parametrize("loads", ["woa_f32", "direct_f32"])
@pytest.mark.parametrize("rows", [2, 6])
def test_executor_sites_equal_ordinary_rows_bitwise(qat, loads, rows):
    with mx.stream(mx.gpu):
        weights = FakeSource(qat, loads)
        stats = {}
        executor = SourceRowsExecutor(weights, stats)
        cases = [("linear", "blk.0.attn_output_b.weight", (1, 1, 8192)),
                 ("linear", "blk.0.ffn_down_shexp.weight", (1, 1, 2304)),
                 ("linear", "blk.0.ffn_gate_inp.weight", (1, 1, 5120)),
                 ("linear", "output.weight", (1, 1, 5120)),
                 ("grouped", "blk.0.attn_output_a.weight", (1, 1, 8, 4096)),
                 ("gate_up", ("blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight"), (1, 1, 5120)),
                 ("linear", "blk.0.attn_q_b.weight", (1, 1, 1280))]
        if qat == "fused_fp32":
            cases.append(("paired", ("blk.0.attn_q_a.weight", "blk.0.attn_kv.weight"), (1, 1, 5120)))
        for kind, name, shape in cases:
            parks = _parks(weights, kind, name, rows, shape)
            expected = [p.original(*p.args, **p.kwargs) for p in parks]
            actual = executor.run(parks)
            mx.eval(expected, actual)
            for e, a in zip(expected, actual):
                for x, y in zip(e if isinstance(e, tuple) else (e,), a if isinstance(a, tuple) else (a,)):
                    assert x.shape == y.shape and x.dtype == y.dtype, (kind, name)
                    assert np.array_equal(_bits(x), _bits(y)), (kind, name)
    assert stats["lockstep_multirow_sites"] == len(cases), stats
    assert stats["lockstep_multirow_grouped_calls"] == 1


def test_synthetic_model_lockstep_equals_token_major_on_gpu():
    from test_deepseek_v41_cpu import Weights, config
    from test_deepseek_v41_dspark_chain_cpu import _assert_state, _state

    from mlx2.adapters import deepseek_v41_dspark_chain as chain
    from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
    mx.set_default_device(mx.gpu)
    c = config(sliding_window=4)
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@v41-lockstep-gpu")
    cache = model.make_cache()
    model(mx.array([[i % 19 for i in range(9)]], mx.uint32), cache)
    base = [s.snapshot() for s in cache]
    tokens = mx.array([[1, 2, 3, 4, 5]], mx.uint32)
    reference = [model.forward_with_taps(tokens[:, i:i + 1], cache, (2, 4)) for i in range(5)]
    expected = _state(cache)
    for state, saved in zip(cache, base):
        state.restore(saved)
    logits, taps, _ = chain.verify(model, tokens, cache, (2, 4))
    mx.eval(logits, taps)
    assert np.array_equal(_bits(mx.concatenate([r[0] for r in reference], 1)), _bits(logits))
    assert np.array_equal(_bits(mx.concatenate([r[1] for r in reference], 1)), _bits(taps))
    _assert_state(expected, _state(cache))
