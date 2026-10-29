# SPDX-License-Identifier: MIT
"""Real-Metal bitwise parity, synthetic tensors only; no checkpoint or quality claim.

Run under the GPU lock: MLX2_RUN_METAL_TESTS=1. Peak allocation is about 4 GB
(the head case holds BF16 and prepared F32 129280x5120 coefficients).
"""
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.gguf import source_dense as dense
from mlx2.runtime.gguf import source_dense_mxfp8 as m8

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available() or not dense._hardware_supported(),
    reason="set MLX2_RUN_METAL_TESTS=1 on the M5 Max (under the GPU lock) for the real-Metal gate",
)
G, N, K = m8.GROUPS, m8.N, m8.K


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def bits(value):
    mx.eval(value)
    return np.asarray(value).view(np.uint32)


def woa(seed, scale_low=117, scale_high=124, all_codes=False, scale=None):
    rng = np.random.default_rng(seed)
    if all_codes:
        codes = np.resize(np.arange(256, dtype=np.uint8), G*N*K).reshape(G*N, K)
    else:
        codes = rng.integers(0, 256, size=(G*N, K), dtype=np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x11  # E4M3 NaN codes are not weights
    q = mx.array(codes.view('<u4'))
    if scale is None:
        blocks = rng.integers(scale_low, scale_high, size=(G*N//32, K//32), dtype=np.uint8)
        s = mx.repeat(mx.array(blocks), 32, axis=0)  # same layout the Source loader builds
    else:
        s = mx.full((G*N, K//32), scale, mx.uint8)
    return q, s


@pytest.mark.parametrize('seed', range(4))
def test_woa_mxfp8_gemv_bitwise_equals_bf16_expansion_consumers(seed):
    q, s = woa(seed)
    expanded = mx.dequantize(q, scales=s, mode='mxfp8', dtype=mx.bfloat16)
    x = mx.random.normal((1, 1, G, K), key=mx.random.key(seed)).astype(mx.bfloat16).astype(mx.float32)
    got = bits(m8.project(x, q, s))
    np.testing.assert_array_equal(got, bits(dense.project(x, expanded, 'wo_a')))
    stock = mx.einsum('btgi,goi->btgo', x, expanded.astype(mx.float32).reshape(G, N, K))
    np.testing.assert_array_equal(got, bits(stock))


@pytest.mark.parametrize('scale', (m8.SCALE_MIN, 60, 127, 200, m8.SCALE_MAX))
def test_every_code_at_envelope_scales(scale):
    q, s = woa(0, all_codes=True, scale=scale)
    expanded = mx.dequantize(q, scales=s, mode='mxfp8', dtype=mx.bfloat16)
    np.testing.assert_array_equal(bits(expanded.astype(mx.float32)),
                                  bits(mx.dequantize(q, scales=s, mode='mxfp8', dtype=mx.float32)))
    x = mx.random.normal((1, 1, G, K), key=mx.random.key(scale)).astype(mx.float32)
    np.testing.assert_array_equal(bits(m8.project(x, q, s)), bits(dense.project(x, expanded, 'wo_a')))


def test_weights_grouped_decode_matches_bf16_profile_on_device():
    from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
    name = 'blk.3.attn_output_a.weight'
    q, s = woa(11)
    expanded = mx.dequantize(q, scales=s, mode='mxfp8', dtype=mx.bfloat16)

    def owner(storage, dense_values):
        w = SourceV41Weights.__new__(SourceV41Weights)
        w.source_bf16_loads_policy, w.source_woa_storage_policy, w.source_woa_pp_policy = 'woa_f32', storage, 'reference'
        w.scheduled_dense_calls, w.scheduled_grouped_calls = {}, {}
        w._ensure_dense = w._ready = lambda: None
        w.source_resident_bank_session = None
        w._dense = dense_values
        return w
    stored = owner('mxfp8_gemv_v1', {name+'.q': q, name+'.s': s})
    reference = owner('reference', {name: expanded})
    for rows in (1, 2, 9):
        x = mx.random.normal((1, rows, G, K), key=mx.random.key(rows)).astype(mx.float32)
        np.testing.assert_array_equal(bits(stored.grouped(x, name)), bits(reference.grouped(x, name)))
    assert stored.scheduled_grouped_calls['source_mxfp8_grouped_gemv_f32'] == 1
    assert stored.scheduled_grouped_calls['source_mxfp8_grouped_bf16_expansion'] == 2
    assert reference.scheduled_grouped_calls['source_bf16_grouped_direct_f32'] == 1


def test_head_direct_bf16_bitwise_equals_prepared_f32_reference():
    v, d = dense.SHAPES['head'][1:]
    weight = (mx.random.normal((v, d), key=mx.random.key(3)) * 0.02).astype(mx.bfloat16)
    prepared = weight.astype(mx.float32)
    mx.eval(weight, prepared)
    for seed in range(2):
        x = mx.random.normal((1, 1, d), key=mx.random.key(seed)).astype(mx.float32)
        np.testing.assert_array_equal(bits(dense.project(x, weight, 'head')), bits(x @ prepared.T))
    # Multi-row fallback consumes the identical F32 coefficient values.
    x = mx.random.normal((1, 5, d), key=mx.random.key(9)).astype(mx.float32)
    np.testing.assert_array_equal(bits(x @ weight.astype(mx.float32).T), bits(x @ prepared.T))
