"""CPU tests for per-layer quantized-key token selection (experimental)."""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.cache import KVCache, QuantizedKVCache
from mlx2.runtime.quantized_key_index import (
    QuantizedKeyIndexCache,
    install_index,
    quantize_full_attention,
)
from mlx2.runtime.topk_index_reuse import (
    attention_probs,
    relative_error,
    select_mask,
    selection_scores,
)


def _filled(cache, T, Hkv=2, D=64, seed=0, chunks=(1,)):
    """Feed T rows of random K/V into ``cache`` in the given chunk sizes (cycled)."""
    mx.random.seed(seed)
    k = mx.random.normal((1, Hkv, T, D))
    v = mx.random.normal((1, Hkv, T, D))
    pos, i = 0, 0
    while pos < T:
        n = min(chunks[i % len(chunks)], T - pos)
        cache.update_and_fetch(k[:, :, pos : pos + n], v[:, :, pos : pos + n])
        pos, i = pos + n, i + 1
    return k, v


def _peaked(T=400, D=64, peaks=(17, 203), seed=1):
    """Keys where the planted positions dominate a matching query."""
    mx.random.seed(seed)
    k = np.array(mx.random.normal((1, 2, T, D))) * 0.05
    direction = np.ones(D, dtype=np.float32) / np.sqrt(D)
    for p in peaks:
        k[0, :, p] = direction * 8.0
    q = mx.array(np.broadcast_to(direction * 8.0, (1, 4, 1, D)).copy())
    v = mx.random.normal((1, 2, T, D))
    return q, mx.array(k), v


def test_incremental_index_matches_bulk_quantization():
    cache = QuantizedKeyIndexCache(bits=8, budget=8, window=4)
    k, _ = _filled(cache, 300, chunks=(200,))
    cache._sync_index()  # bulk build over 200+100 rows crossing a step boundary
    for _ in range(3):
        extra = mx.random.normal((1, 2, 1, 64))
        cache.update_and_fetch(extra, extra)
        k = mx.concatenate([k, extra], axis=2)
        cache._sync_index()
    want = mx.quantize(k, group_size=64, bits=8)
    for got, ref in zip(cache._qkeys, want):
        assert mx.array_equal(got[..., : cache.offset, :], ref).item()
    assert cache._q_offset == cache.offset == 303


def test_short_context_takes_stock_path():
    cache = QuantizedKeyIndexCache(budget=64, window=16)
    _filled(cache, 60, chunks=(60,))
    cache.armed = True
    assert cache.bucketed_attention(mx.zeros((1, 4, 1, 64)), 0.125, None) is None
    assert cache.counts == {"dense": 1}


def test_unarmed_and_prefill_rows_take_stock_path():
    cache = QuantizedKeyIndexCache(budget=8, window=4)
    _filled(cache, 100, chunks=(100,))
    assert cache.bucketed_attention(mx.zeros((1, 4, 1, 64)), 0.125, None) is None
    cache.armed = True
    assert cache.bucketed_attention(mx.zeros((1, 4, 3, 64)), 0.125, None) is None
    assert cache.counts == {"dense": 2}


def test_exact_selection_matches_reference_topk():
    cache = QuantizedKeyIndexCache(budget=12, window=4, exact_scores=True)
    k, _ = _filled(cache, 90, chunks=(90,))
    q = mx.random.normal((1, 4, 1, 64))
    idx = np.array(cache.select(q, k, 0.125))[0, 0]
    probs = attention_probs(q, k, 0.125)
    ref = np.array(select_mask(selection_scores(probs, 2, "shared"), budget=12, window=4))[0, 0]
    assert len(idx) == len(set(idx.tolist())) == 16
    assert set(idx.tolist()) == set(np.flatnonzero(ref).tolist())


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_scores_keep_the_planted_peaks(bits):
    q, k, v = _peaked()
    cache = QuantizedKeyIndexCache(bits=bits, budget=8, window=4)
    cache.update_and_fetch(k, v)
    cache.armed = True
    idx = set(np.array(cache.select(q, k, 0.125))[0, 0].tolist())
    assert {17, 203} <= idx
    out = cache.bucketed_attention(q, 0.125, None)
    oracle = QuantizedKeyIndexCache(budget=8, window=4, exact_scores=True)
    oracle.update_and_fetch(k, v)
    oracle.armed = True
    dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.125)
    # ~6% of the mass is spread over the background, so any 12-row set errs
    # by ~0.066; the quantized selection must do as well as the exact one.
    exact_err = relative_error(dense, oracle.bucketed_attention(q, 0.125, None)).max().item()
    assert relative_error(dense, out).max().item() <= exact_err + 0.01
    assert cache.counts == {"indexed": 1}


def test_indexed_output_is_sdpa_over_the_selected_rows():
    cache = QuantizedKeyIndexCache(bits=8, budget=10, window=6, sharing="kv_head")
    k, v = _filled(cache, 120, chunks=(120,))
    cache.armed = True
    q = mx.random.normal((1, 4, 1, 64))
    idx = cache.select(q, k, 0.125)  # (1, 2, 16), one set per KV head
    assert idx.shape == (1, 2, 16)
    rows = idx[..., None]
    want = mx.fast.scaled_dot_product_attention(
        q, mx.take_along_axis(k, rows, axis=2), mx.take_along_axis(v, rows, axis=2), scale=0.125
    )
    assert mx.allclose(cache.bucketed_attention(q, 0.125, None), want, atol=1e-5).item()


def test_trim_rewinds_the_index():
    cache = QuantizedKeyIndexCache(budget=8, window=4)
    _filled(cache, 50, chunks=(50,))
    cache._sync_index()
    cache.trim(5)
    assert cache._q_offset == cache.offset == 45


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        QuantizedKeyIndexCache(budget=0)
    with pytest.raises(ValueError):
        QuantizedKeyIndexCache(sharing="per_query")


# --- tiny models -------------------------------------------------------------


def _tiny_qwen():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=2,
        mtp_num_hidden_layers=0, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=4096,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    for layer in model.model.layers:
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            attention.v_proj.weight = attention.v_proj.weight * 8.0
            attention.o_proj.weight = attention.o_proj.weight * 8.0
    mx.eval(model.parameters())
    return model


def _tiny_muse():
    from mlx2.adapters.muse_glimmer_config import ModelArgs
    from mlx2.runtime.models.muse_glimmer import Model

    args = ModelArgs(
        hidden_size=64, num_hidden_layers=8, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        vocab_size=128, sliding_window=8, max_position_embeddings=4096,
    )
    mx.random.seed(11)
    model = Model(args)
    model.eval()
    mx.eval(model.parameters())
    return model


TOKENS = [int(x) for x in np.random.default_rng(5).integers(1, 120, size=72)]


def _run(model, arm):
    cache = model.make_cache()
    indexed = []
    if arm.get("index"):
        cache, indexed = install_index(cache, **arm["index"])
    ids = mx.array([TOKENS], dtype=mx.uint32)
    mx.eval(model(ids[:, :48], cache=cache))
    if arm.get("quantize"):
        cache, converted = quantize_full_attention(cache, **arm["quantize"])
        assert converted >= 2
    for c in indexed:
        c.armed = True
    out = []
    for pos in range(48, len(TOKENS)):
        logits = model(ids[:, pos : pos + 1], cache=cache)[0, -1]
        mx.eval(logits)
        out.append(logits)
    return mx.stack(out), indexed, cache


@pytest.mark.parametrize("factory", [_tiny_qwen, _tiny_muse], ids=["qwen", "muse"])
def test_install_swaps_only_full_attention(factory):
    model = factory()
    fresh = model.make_cache()
    swapped, indexed = install_index(fresh, budget=8, window=4)
    fa = [i for i, c in enumerate(fresh) if type(c) is KVCache]
    assert len(indexed) == len(fa) >= 2
    for i, (a, b) in enumerate(zip(fresh, swapped)):
        assert (b in indexed) if i in fa else (b is a)


@pytest.mark.parametrize("factory", [_tiny_qwen, _tiny_muse], ids=["qwen", "muse"])
def test_large_budget_is_bit_identical_to_dense(factory):
    model = factory()
    dense, _, _ = _run(model, {})
    same, indexed, _ = _run(model, {"index": {"budget": 64, "window": 64}})
    assert mx.array_equal(dense, same).item()
    assert all(c.counts["indexed"] == 0 for c in indexed)


@pytest.mark.parametrize("factory", [_tiny_qwen, _tiny_muse], ids=["qwen", "muse"])
@pytest.mark.parametrize("bits", [8, 4])
def test_small_budget_selects_every_decode_step(factory, bits):
    model = factory()
    dense, _, _ = _run(model, {})
    sparse, indexed, _ = _run(model, {"index": {"bits": bits, "budget": 6, "window": 4}})
    steps = len(TOKENS) - 48
    assert all(c.counts["indexed"] == steps for c in indexed)
    assert not mx.allclose(dense, sparse, atol=1e-3).item()


@pytest.mark.parametrize("factory", [_tiny_qwen, _tiny_muse], ids=["qwen", "muse"])
def test_kv_q8_conversion_runs_and_stays_close(factory):
    model = factory()
    dense, _, _ = _run(model, {})
    q8, _, cache = _run(model, {"quantize": {"key_bits": 8, "value_bits": 8}})
    assert sum(isinstance(c, QuantizedKVCache) for c in cache) >= 2
    assert mx.allclose(dense, q8, atol=0.5).item()
    assert not mx.array_equal(dense, q8).item()  # the quantized path really ran
