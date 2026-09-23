"""The quantized decode SDPA path: GQA folded into rows for Q.K^T (2026-09-23).

The reference below is the previous formulation (GQA as a broadcast batch
axis for both products), kept verbatim so the new path is checked against
exactly what it replaced.
"""

import mlx.core as mx
import pytest
from mlx.utils import tree_map

from mlx2.runtime.models.base import quantized_scaled_dot_product_attention


def broadcast_reference(queries, q_keys, q_values, scale, mask, group_size, key_bits, value_bits):
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads
    queries = queries * scale
    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)
    scores = mx.quantized_matmul(queries, *q_keys, transpose=True, group_size=group_size, bits=key_bits)
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            mask = mx.arange(kL - qL, kL)[:, None] >= mx.arange(kL)[None]
        if n_repeats > 1 and mask.ndim == scores.ndim - 1:
            mask = mx.expand_dims(mask, -3) if mask.shape[-3] == 1 else mx.unflatten(mask, -3, (n_kv_heads, n_repeats))
        scores = mx.where(mask, scores, mx.finfo(scores.dtype).min) if mask.dtype == mx.bool_ else scores + mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(scores, *q_values, transpose=False, group_size=group_size, bits=value_bits)
    return mx.reshape(out, (B, n_q_heads, L, D)) if n_repeats > 1 else out


def _inputs(Hq, Hkv, L, T=300, D=64, key_bits=8, value_bits=8, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((1, Hq, L, D))
    k = mx.random.normal((1, Hkv, T, D))
    v = mx.random.normal((1, Hkv, T, D))
    return (q, mx.quantize(k, group_size=64, bits=key_bits),
            mx.quantize(v, group_size=64, bits=value_bits), k, v)


def _masks(Hq, L, T):
    visible = mx.arange(T)[None, :] <= mx.arange(T - L, T)[:, None]
    return {
        "none": None,
        "causal": "causal",
        "bool_shared": visible[None, None],
        "bool_per_head": mx.broadcast_to(visible, (1, Hq, L, T)),
        "additive": mx.where(visible, 0.0, -1e9)[None, None],
    }


@pytest.mark.parametrize("Hq,Hkv", [(4, 4), (8, 4), (8, 2), (16, 2)])
@pytest.mark.parametrize("L", [1, 3])
@pytest.mark.parametrize("bits", [(8, 8), (8, 4), (4, 4)])
@pytest.mark.parametrize("mask_kind", ["none", "causal", "bool_shared", "bool_per_head", "additive"])
def test_matches_previous_broadcast_formulation(Hq, Hkv, L, bits, mask_kind):
    q, qk, qv, _, _ = _inputs(Hq, Hkv, L, key_bits=bits[0], value_bits=bits[1])
    mask = _masks(Hq, L, 300)[mask_kind]
    got = quantized_scaled_dot_product_attention(
        q, qk, qv, scale=0.125, mask=mask, group_size=64, key_bits=bits[0], value_bits=bits[1])
    want = broadcast_reference(q, qk, qv, 0.125, mask, 64, bits[0], bits[1])
    assert got.shape == (1, Hq, L, 64)
    assert mx.allclose(got, want, atol=1e-5, rtol=1e-5).item()


@pytest.mark.parametrize("Hq,Hkv", [(8, 2), (16, 2)])
def test_matches_dequantized_attention(Hq, Hkv):
    q, qk, qv, _, _ = _inputs(Hq, Hkv, 1)
    got = quantized_scaled_dot_product_attention(q, qk, qv, scale=0.125, mask=None, group_size=64, bits=8)
    k = mx.dequantize(*qk, group_size=64, bits=8)
    v = mx.dequantize(*qv, group_size=64, bits=8)
    want = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
    assert mx.allclose(got, want, atol=1e-4).item()


def test_does_not_scale_the_callers_queries_in_place():
    q, qk, qv, _, _ = _inputs(8, 2, 1)
    before = mx.array(q.tolist())
    quantized_scaled_dot_product_attention(q, qk, qv, scale=0.125, mask=None, group_size=64, bits=8)
    assert mx.array_equal(q, before).item()
