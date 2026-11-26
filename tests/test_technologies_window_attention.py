"""Window attention technology: exact CPU parity with every migrated copy.

Each ``_legacy_*`` function is the body a model or cache carried before it
called ``mlx2.runtime.technologies.window_attention`` (mlx2 d46e7a3ea),
moved here verbatim (``self`` attributes turned into arguments).  Masks are
compared with ``mx.array_equal`` plus dtype and shape; the reference
attention in every dtype the model runs.
"""

from __future__ import annotations

import itertools

import mlx.core as mx
import pytest

from mlx2.runtime.technologies import dispatch
from mlx2.runtime.technologies import window_attention as WA

mx.set_default_device(mx.cpu)


def _grid(*axes):
    return list(itertools.product(*axes))


def _same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype, a.shape, b.shape)
    assert mx.array_equal(a, b).item()


# --------------------------------------------------------------------------
# models/base.py create_causal_mask
# --------------------------------------------------------------------------


def _legacy_base_create_causal_mask(N, offset=0, window_size=None, right_padding=None,
                                    left_padding=None):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < mx.expand_dims(offset + N - right_padding, (1, 2, 3)))
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= rinds)
    return mask


@pytest.mark.parametrize("N,offset,window", _grid(
    (1, 2, 5, 17), (0, 1, 7, 40), (None, 1, 3, 8, 64)))
def test_causal_window_mask_matches_base(N, offset, window):
    from mlx2.runtime.models.base import create_causal_mask

    want = _legacy_base_create_causal_mask(N, offset, window)
    _same(WA.causal_window_mask(N, offset, window), want)
    _same(create_causal_mask(N, offset, window_size=window), want)


def test_causal_window_mask_matches_base_with_padding():
    from mlx2.runtime.models.base import create_causal_mask

    left = mx.array([0, 2, 5])
    right = mx.array([1, 0, 3])
    for N, offset, window in _grid((1, 4), (0, 6), (None, 3)):
        for lp, rp in ((left, None), (None, right), (left, right)):
            want = _legacy_base_create_causal_mask(N, offset, window, rp, lp)
            _same(WA.causal_window_mask(N, offset, window, rp, lp), want)
            _same(create_causal_mask(N, offset, window, rp, lp), want)


# --------------------------------------------------------------------------
# models/cache.py
# --------------------------------------------------------------------------


def _legacy_rotating_keep_mask(N, offset, window_size, keep):
    rinds = mx.arange(offset + N)[None]
    linds = mx.arange(offset, offset + N)[:, None]
    return (linds >= rinds) & (
        (rinds < keep)
        | (linds < rinds + window_size - keep)
    )


@pytest.mark.parametrize("N,offset,window,keep", _grid(
    (2, 5, 9), (0, 3, 11), (4, 8), (1, 2, 3)))
def test_sink_prefix_window_mask_matches_rotating_cache(N, offset, window, keep):
    _same(WA.sink_prefix_window_mask(N, offset, window, keep),
          _legacy_rotating_keep_mask(N, offset, window, keep))


def _legacy_ring_decode(offset, _idx, max_size, window_size):
    idx = _idx
    if idx >= max_size:
        idx = 0
    if offset < max_size:
        mask_size = offset + 1
    else:
        mask_size = max_size
    mask = mx.arange(mask_size) >= mask_size - window_size
    mask = mx.roll(mask, shift=idx + 1)
    return mask


@pytest.mark.parametrize("offset,idx,max_size,window", [
    (o, i, m, w) for m in (4, 8, 13) for w in (1, 3) if w < m
    for o in range(w, 3 * m) for i in (0, 1, m - 1, m, m + 2)])
def test_ring_decode_window_mask_matches_rotating_caches(offset, idx, max_size, window):
    _same(WA.ring_decode_window_mask(offset, idx, max_size, window),
          _legacy_ring_decode(offset, idx, max_size, window))


def _legacy_batch_rotating_band(N, offset, window_size):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    mask &= linds < rinds + window_size
    return mask, rinds


@pytest.mark.parametrize("N,offset,window", _grid((1, 3, 8), (0, 2, 15), (1, 4, 16)))
def test_batch_rotating_band_matches(N, offset, window):
    want, want_rinds = _legacy_batch_rotating_band(N, offset, window)
    rinds = mx.arange(offset + N)[None]
    _same(WA.causal_window_mask(N, offset, window_size=window), want)
    _same(rinds, want_rinds)


def _legacy_sink_window_cache(query_positions, key_positions, sink_size, effective):
    keys = mx.array(key_positions, dtype=mx.int32)[None]
    queries = mx.array(query_positions, dtype=mx.int32)[:, None]
    causal = queries >= keys
    return causal & ((keys < sink_size) | (keys >= queries - effective))


@pytest.mark.parametrize("sink,effective", _grid((0, 1, 4), (1, 3, 6)))
def test_window_keep_matches_sink_window_cache(sink, effective):
    active = [0, 1, 2, 3, 9, 10, 11, 12]
    offset, length = 13, 4
    keys = active + list(range(offset, offset + length))
    queries = list(range(offset, offset + length))
    want = _legacy_sink_window_cache(queries, keys, sink, effective)
    got = WA.window_keep(mx.array(queries, dtype=mx.int32), mx.array(keys, dtype=mx.int32),
                         effective, sink_tokens=sink, window_counts_query=False)
    _same(got, want)


def test_cache_objects_build_the_same_masks():
    """End to end through the cache classes that now call the technology."""
    from mlx2.runtime.models.cache import RotatingKVCache

    for keep in (0, 2):
        cache = RotatingKVCache(max_size=8, keep=keep)
        k = mx.zeros((1, 1, 11, 4))
        cache.update_and_fetch(k, k)
        got = cache.make_mask(3, return_array=True)
        offset = min(cache.max_size - 1, cache.offset)
        want = (_legacy_base_create_causal_mask(3, offset, window_size=8) if keep == 0
                else _legacy_rotating_keep_mask(3, offset, 8, keep))
        _same(got, want)
    cache = RotatingKVCache(max_size=8, keep=0)
    for _ in range(12):
        cache.update_and_fetch(mx.zeros((1, 1, 1, 4)), mx.zeros((1, 1, 1, 4)))
    _same(cache.make_mask(1, window_size=3),
          _legacy_ring_decode(cache.offset, cache._idx, cache.max_size, 3))


# --------------------------------------------------------------------------
# models/mimo_v2.py
# --------------------------------------------------------------------------


def _legacy_mimo_attention_mask(T, window):
    q = mx.arange(T)[:, None]
    k = mx.arange(T)[None, :]
    keep = k <= q
    if window is not None:
        keep = keep & (k > q - window)
    return mx.where(keep, mx.array(0.0, mx.float32), mx.array(-mx.inf, mx.float32))


def _legacy_mimo_attention_core(q, k, v, *, heads, kv_heads, head_dim, mask, sink, row_chunk,
                                dtype):
    B, _, T, _ = q.shape
    groups = heads // kv_heads
    if groups > 1:
        k = mx.repeat(k, groups, axis=1)
        v = mx.repeat(v, groups, axis=1)
    scale = head_dim ** -0.5
    step = row_chunk or T
    outs = []
    for r0 in range(0, T, step):
        r1 = min(T, r0 + step)
        s = (q[:, :, r0:r1] @ k.transpose(0, 1, 3, 2)) * scale
        s = s.astype(mx.float32) + mask[r0:r1]
        if sink is not None:
            col = mx.broadcast_to(sink.astype(mx.float32).reshape(1, -1, 1, 1), (B, heads, r1 - r0, 1))
            s = mx.concatenate([s, col], axis=-1)
        p = mx.softmax(s - mx.max(s, axis=-1, keepdims=True), axis=-1).astype(dtype)
        if sink is not None:
            p = p[..., :-1]
        outs.append(p @ v)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


@pytest.mark.parametrize("T,window", _grid((1, 2, 9, 33), (None, 1, 4, 128)))
def test_mimo_attention_mask(T, window):
    from mlx2.runtime.models import mimo_v2

    _same(mimo_v2.attention_mask(T, window), _legacy_mimo_attention_mask(T, window))


@pytest.mark.parametrize("dtype,window,sinked,row_chunk,gqa", _grid(
    (mx.bfloat16, mx.float16, mx.float32), (None, 3), (False, True), (None, 3), (1, 4)))
def test_mimo_attention_core_matches_reference(dtype, window, sinked, row_chunk, gqa):
    heads, kv_heads, hd, vhd, T = 8, 8 // gqa, 16, 8, 7
    key = mx.random.key(11)
    k1, k2, k3, k4 = mx.random.split(key, 4)
    q = mx.random.normal((2, heads, T, hd), key=k1).astype(dtype)
    k = mx.random.normal((2, kv_heads, T, hd), key=k2).astype(dtype)
    v = mx.random.normal((2, kv_heads, T, vhd), key=k3).astype(dtype)
    # Ties: duplicate a key so two scores are exactly equal.
    k = mx.concatenate([k[:, :, :1], k[:, :, :1], k[:, :, 2:]], axis=2)
    sink = mx.random.normal((heads,), key=k4).astype(dtype) if sinked else None
    mask = _legacy_mimo_attention_mask(T, window)
    want = _legacy_mimo_attention_core(q, k, v, heads=heads, kv_heads=kv_heads, head_dim=hd,
                                       mask=mask, sink=sink, row_chunk=row_chunk, dtype=dtype)
    spec = WA.WindowAttentionSpec(sliding_window=window, sink_logits=sinked,
                                  scale=hd ** -0.5, mask_form="additive")
    got = WA.reference_attention(q, k, v, spec, mask=mask, sinks=sink, row_chunk=row_chunk)
    _same(got, want)


def test_mimo_attention_layer_end_to_end():
    """mimo_v2.attention (projections + RoPE + core) against the legacy core."""
    from mlx2.runtime.models import mimo_v2

    spec = mimo_v2.AttnSpec(True, 4, 2, 16, 8, 8, 10000.0, True, 3)
    d, T = 32, 6
    keys = mx.random.split(mx.random.key(3), 6)
    w = {
        "self_attn.q_proj.weight": mx.random.normal((4 * 16, d), key=keys[0]).astype(mx.bfloat16),
        "self_attn.k_proj.weight": mx.random.normal((2 * 16, d), key=keys[1]).astype(mx.bfloat16),
        "self_attn.v_proj.weight": mx.random.normal((2 * 8, d), key=keys[2]).astype(mx.bfloat16),
        "self_attn.o_proj.weight": mx.random.normal((d, 4 * 8), key=keys[3]).astype(mx.bfloat16),
        "self_attn.attention_sink_bias": mx.random.normal((4,), key=keys[4]).astype(mx.bfloat16),
    }
    x = mx.random.normal((1, T, d), key=keys[5]).astype(mx.bfloat16)

    class Cfg:
        value_scale = None

    got = mimo_v2.attention(w, x, spec, Cfg(), row_chunk=4)
    # Legacy: the same projections/RoPE, then the legacy core.
    B = 1
    lin = mimo_v2.linear
    q = lin(x, w["self_attn.q_proj.weight"]).reshape(B, T, 4, 16).transpose(0, 2, 1, 3)
    k = lin(x, w["self_attn.k_proj.weight"]).reshape(B, T, 2, 16).transpose(0, 2, 1, 3)
    v = lin(x, w["self_attn.v_proj.weight"]).reshape(B, T, 2, 8).transpose(0, 2, 1, 3)
    cos, sin = mimo_v2.rope_cos_sin(T, spec.rope_dim, spec.rope_theta, x.dtype)
    q = mimo_v2.apply_partial_rope(q, cos, sin, spec.rope_dim)
    k = mimo_v2.apply_partial_rope(k, cos, sin, spec.rope_dim)
    o = _legacy_mimo_attention_core(q, k, v, heads=4, kv_heads=2, head_dim=16,
                                    mask=_legacy_mimo_attention_mask(T, 3),
                                    sink=w["self_attn.attention_sink_bias"], row_chunk=4,
                                    dtype=x.dtype)
    o = o.transpose(0, 2, 1, 3).reshape(B, T, 4 * 8)
    _same(got, lin(o, w["self_attn.o_proj.weight"]))


# --------------------------------------------------------------------------
# hy_v4_flash sliding band / deepseek_v41 local window
# --------------------------------------------------------------------------


def _legacy_hy_allowed(qpos, kpos, window):
    return (kpos[None, :] <= qpos[:, None]) & (kpos[None, :] > qpos[:, None] - window)


def _legacy_deepseek_local(start, length, local_length, window):
    queries = mx.arange(start, start + length)[:, None]
    keys = mx.arange(start + length - local_length, start + length)
    return ((queries >= keys) & (queries < keys + window))[None, None]


@pytest.mark.parametrize("start,L,lo,window", _grid((0, 5, 130), (1, 4, 9), (0, 3), (1, 4, 128)))
def test_window_keep_matches_hy_v4_flash_band(start, L, lo, window):
    lo = max(0, min(lo, start))
    qpos = mx.arange(start, start + L)
    kpos = mx.arange(lo, start + L)
    _same(WA.window_keep(qpos, kpos, window), _legacy_hy_allowed(qpos, kpos, window))


@pytest.mark.parametrize("start,length,local,window", [
    (0, 1, 1, 4), (0, 5, 5, 4), (7, 3, 6, 4), (100, 8, 12, 128), (3, 2, 5, 1)])
def test_window_keep_matches_deepseek_v41_local(start, length, local, window):
    keys = mx.arange(start + length - local, start + length)
    got = WA.window_keep(mx.arange(start, start + length), keys, window)[None, None]
    _same(got, _legacy_deepseek_local(start, length, local, window))


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------


def test_spec_from_gemma_style_config():
    spec = WA.WindowAttentionSpec.from_config({
        "sliding_window": 1024, "sliding_window_pattern": 6, "num_hidden_layers": 12,
        "attn_logit_softcapping": 50.0, "query_pre_attn_scalar": 256, "head_dim": 128,
    })
    assert spec.sliding_window == 1024 and spec.logit_softcap == 50.0
    assert spec.scale == 256 ** -0.5
    assert [spec.window_for_layer(i) for i in range(12)] == [
        1024, 1024, 1024, 1024, 1024, None] * 2


def test_spec_from_layer_types_and_mimo_pattern():
    spec = WA.WindowAttentionSpec.from_config({
        "sliding_window": 128, "head_dim": 64,
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
        "num_sink_tokens": 4})
    assert spec.layer_windowed == (True, False, True) and spec.sink_tokens == 4
    assert spec.scale == 64 ** -0.5
    mimo = WA.WindowAttentionSpec.from_config({
        "sliding_window_size": 128, "hybrid_layer_pattern": [0, 1, 1, 0],
        "add_swa_attention_sink_bias": True, "hidden_size": 256, "num_attention_heads": 4})
    assert mimo.layer_windowed == (False, True, True, False) and mimo.sink_logits
    assert mimo.window_for_layer(1) == 128 and mimo.window_for_layer(0) is None
    assert mimo.scale == 64 ** -0.5
    block = WA.WindowAttentionSpec.from_config({"sliding_window": 8, "hybrid_block_size": 2,
                                                "num_hidden_layers": 4})
    assert block.layer_windowed == (True, False, True, False)
    assert WA.WindowAttentionSpec.from_config({}).window_for_layer(3) is None
    with pytest.raises(TypeError):
        WA.WindowAttentionSpec.from_config({}, not_a_field=1)
    with pytest.raises(ValueError):
        WA.WindowAttentionSpec(mask_form="float")


def test_reference_softcap_and_bool_mask():
    q = mx.random.normal((1, 2, 3, 8), key=mx.random.key(1))
    k = mx.random.normal((1, 2, 3, 8), key=mx.random.key(2)) * 20
    keep = WA.window_keep(mx.arange(3), mx.arange(3), 2)
    spec = WA.WindowAttentionSpec(logit_softcap=5.0, scale=1.0)
    got = WA.reference_attention(q, k, k, spec, mask=keep)
    s = mx.tanh((q @ k.transpose(0, 1, 3, 2)).astype(mx.float32) / 5.0) * 5.0
    s = s + WA.additive_mask(keep)
    p = mx.softmax(s - mx.max(s, axis=-1, keepdims=True), axis=-1)
    _same(got, p @ k)


# --------------------------------------------------------------------------
# DENSE registry (models.base.scaled_dot_product_attention float branch)
# --------------------------------------------------------------------------


def _qkv(L, D=128):
    q = mx.zeros((1, 4, L, D), dtype=mx.bfloat16)
    k = mx.zeros((1, 2, 64, D), dtype=mx.bfloat16)
    return q, k, k


def test_dense_registry_name_and_order():
    assert WA.DENSE.operation == "window_attention.dense"
    assert [p.name for p in WA.DENSE._paths] == [
        "fp_verify_tensor", "sdpa_vector_d512_tk2", "qsdpa_fp_decode"]


def test_dense_falls_back_to_stock_with_bounded_refusal_codes(monkeypatch):
    for name in ("MLX2_FP_VERIFY_KERNEL", "MLX2_SDPA_D512_TK2", "MLX2_FP_DECODE_KERNEL"):
        monkeypatch.delenv(name, raising=False)
    assert not WA.dense_fast_paths_requested()
    monkeypatch.setenv("MLX2_FP_DECODE_KERNEL", "1")
    assert WA.dense_fast_paths_requested()
    dispatch.reset_counters()
    q, k, v = _qkv(1)
    run, receipt = WA.DENSE.select(None, queries=q, keys=k, values=v, mask=None, sinks=None)
    assert receipt.selected == "reference" and run is WA._stock  # CPU: no kernel admits
    assert {dispatch.refusal_code(r) for r in receipt.refusals.values()} <= {
        "geometry", "not_admitted"}
    snap = dispatch.counters_snapshot()
    assert snap["selections"] == [{"technology": "window_attention", "operation": "dense",
                                   "path": "reference", "count": 1}]


def test_dense_selects_in_legacy_order(monkeypatch):
    from mlx2.runtime.models import fp_verify_tensor, qsdpa_decode_metal
    from mlx2.runtime.models import sdpa_vector_d512 as d512

    monkeypatch.setattr(fp_verify_tensor, "use_kernel", lambda *a, **k: True)
    monkeypatch.setattr(d512, "use_kernel", lambda *a, **k: True)
    monkeypatch.setattr(qsdpa_decode_metal, "use_fp_decode_kernel", lambda *a, **k: True)
    pick = lambda L, D=128, sinks=None: WA.DENSE.select(  # noqa: E731
        None, queries=_qkv(L, D)[0], keys=_qkv(L, D)[1], values=_qkv(L, D)[2],
        mask=None, sinks=sinks)[1].selected
    assert pick(4) == "fp_verify_tensor"
    assert pick(1, 512) == "sdpa_vector_d512_tk2"
    assert pick(1, 128) == "qsdpa_fp_decode"
    assert pick(1, 128, sinks=mx.zeros((4,))) == "reference"
    assert pick(17) == "reference"


def test_base_dispatch_routes_through_dense(monkeypatch):
    from mlx2.runtime.models import base
    from mlx2.runtime.models import sdpa_vector_d512 as d512

    calls = []
    monkeypatch.setenv("MLX2_SDPA_D512_TK2", "1")
    monkeypatch.setattr(d512, "use_kernel", lambda *a, **k: True)
    monkeypatch.setattr(d512, "attention", lambda q, k, v, *, scale: calls.append(scale) or q)
    q, k, v = _qkv(1, 512)
    out = base.scaled_dot_product_attention(q, k, v, cache=None, scale=0.5, mask=None)
    assert calls == [0.5] and out is q
    for name in ("MLX2_FP_VERIFY_KERNEL", "MLX2_SDPA_D512_TK2", "MLX2_FP_DECODE_KERNEL"):
        monkeypatch.delenv(name, raising=False)
    calls.clear()
    base.scaled_dot_product_attention(q, k, v, cache=None, scale=0.5, mask=None)
    assert calls == []


def test_fast_path_table_names_real_admission_functions():
    import importlib

    for module, fn in WA.FAST_PATHS.values():
        assert callable(getattr(importlib.import_module(module), fn))


def test_base_hot_path_never_consults_the_registry_without_a_flag(monkeypatch):
    from mlx2.runtime.models import base

    for name in ("MLX2_FP_VERIFY_KERNEL", "MLX2_SDPA_D512_TK2", "MLX2_FP_DECODE_KERNEL"):
        monkeypatch.delenv(name, raising=False)

    def refuse(*a, **k):
        raise AssertionError("registry consulted")

    monkeypatch.setattr(WA.DENSE, "select", refuse)
    monkeypatch.setattr(base, "fast_sdpa", lambda q, k, v, **kw: q)
    for L, D in ((1, 128), (1, 512), (4, 128), (32, 128)):
        q, k, v = _qkv(L, D)
        assert base.scaled_dot_product_attention(q, k, v, cache=None, scale=0.5, mask=None) is q
    # A flag for a path this shape cannot take still skips the lookup.
    monkeypatch.setenv("MLX2_SDPA_D512_TK2", "1")
    q, k, v = _qkv(1, 128)
    assert base.scaled_dot_product_attention(q, k, v, cache=None, scale=0.5, mask=None) is q
    monkeypatch.setenv("MLX2_FP_VERIFY_KERNEL", "1")
    q, k, v = _qkv(32, 128)
    assert base.scaled_dot_product_attention(q, k, v, cache=None, scale=0.5, mask=None) is q


@pytest.mark.parametrize("q_dtype,x_dtype", [(mx.float32, mx.bfloat16), (mx.bfloat16, mx.float16)])
def test_reference_rounds_probabilities_to_the_requested_dtype(q_dtype, x_dtype):
    """MiMo rounds its probabilities to the layer input dtype (x.dtype)."""
    keys = mx.random.split(mx.random.key(5), 4)
    q = mx.random.normal((1, 4, 5, 16), key=keys[0]).astype(q_dtype)
    k = mx.random.normal((1, 2, 5, 16), key=keys[1]).astype(q_dtype)
    v = mx.random.normal((1, 2, 5, 8), key=keys[2]).astype(q_dtype)
    sink = mx.random.normal((4,), key=keys[3])
    mask = _legacy_mimo_attention_mask(5, 3)
    want = _legacy_mimo_attention_core(q, k, v, heads=4, kv_heads=2, head_dim=16, mask=mask,
                                       sink=sink, row_chunk=2, dtype=x_dtype)
    spec = WA.WindowAttentionSpec(sliding_window=3, sink_logits=True, scale=16 ** -0.5,
                                  mask_form="additive")
    got = WA.reference_attention(q, k, v, spec, mask=mask, sinks=sink, row_chunk=2,
                                 probs_dtype=x_dtype)
    _same(got, want)
