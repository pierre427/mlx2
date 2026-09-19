"""Opt-in exact Metal gates; run only under the root's exclusive GPU lease."""
import os
from unittest.mock import patch
import mlx.core as mx
import pytest
from mlx2.runtime.models import qwen4_qsa_indexed as indexed
from mlx2.runtime.models import qwen4_qsa_indexed_merge as merge
from test_qwen4_qsa_indexed import _real_bf16_fixture, _real_bf16_m3_fixture

pytestmark = pytest.mark.skipif(os.environ.get("MLX2_TEST_OPTIONAL_METAL") != "1", reason="explicit exclusive GPU gate required")


@pytest.mark.parametrize("fixture", [_real_bf16_fixture, _real_bf16_m3_fixture])
def test_native_output_gate_is_bit_exact_and_observed(fixture):
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        q, k, v, compact = fixture()
        batch, heads, length, dim = q.shape
        for seed in range(16):
            gate = (mx.random.normal((batch, length, heads * dim), key=mx.random.key(seed)) * 12).astype(q.dtype)
            with patch.dict(os.environ, {"MLX_QWEN4_QSA_INDEXED_FUSED_MERGE": "0", "MLX_QWEN4_QSA_INDEXED_FUSED_GATE": "0"}):
                ordinary = indexed.qwen4_qsa_indexed_attention(q, k, v, compact, scale=dim**-0.5)
                expected = merge.mlx_apply_output_gate(ordinary, gate)
            merge.fused_merge_status(reset=True)
            with patch.dict(os.environ, {"MLX_QWEN4_QSA_INDEXED_FUSED_MERGE": "0", "MLX_QWEN4_QSA_INDEXED_FUSED_GATE": "1"}):
                actual = indexed.qwen4_qsa_indexed_attention(q, k, v, compact, scale=dim**-0.5, output_gate=gate)
                mx.eval(actual, expected)
                assert mx.array_equal(actual, expected).item()
                status = merge.fused_merge_status()
                assert status["gate_engaged"] and status["gate_path"] == "native_sdpa_merge"
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("length", [1, 3, 8])
def test_sequential_fused_merge_is_bit_exact_and_observed(dtype, length):
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        m = mx.random.normal((2, 24, length, 128), key=mx.random.key(length)) * 18
        l = mx.exp(mx.random.normal(m.shape, key=mx.random.key(length + 10)) * 4)
        o = (mx.random.normal((*m.shape, 256), key=mx.random.key(length + 20)) * 100).astype(dtype)
        m[..., :3] = -mx.inf
        l[..., :3] = 0
        o[..., :3, :] = 0
        expected = merge.mlx_sequential_merge(m, l, o, output_dtype=dtype)
        merge.fused_merge_status(reset=True)
        with patch.dict(os.environ, {"MLX_QWEN4_QSA_INDEXED_FUSED_MERGE": "1", "MLX_QWEN4_QSA_INDEXED_FUSED_GATE": "0"}):
            actual = merge.combine_indexed_partials(m, l, o, output_dtype=dtype)
            mx.eval(actual, expected)
            assert mx.array_equal(actual, expected).item()
            status = merge.fused_merge_status()
            assert status["engaged"] and status["fallbacks"] == 0
    finally:
        mx.set_default_device(previous)
