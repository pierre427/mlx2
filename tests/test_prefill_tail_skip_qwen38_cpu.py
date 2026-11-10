"""Prompt-chunk tier-2 tail on the Qwen3.8/3.6 hybrid trunk (CPU, tiny models).

MLX2_PREFILL_TAIL_SKIP=1 (default off): the last layer's state writes (full
attention K/V, or the whole GDN mixer) run over every row; q, attention
output, o_proj, MLP/MoE, final norm and head run on the last row only.  Cache
state and the next decode step must be bitwise identical to the full forward;
the checkpoint row's logits must be close.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten  # noqa: E402

from mlx2.runtime import prefill_tail_skip as pts  # noqa: E402
from mlx2.runtime.models.cache import (  # noqa: E402
    ArraysCache,
    BatchKVCache,
    KVCache,
    QuantizedKVCache,
)
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows  # noqa: E402

VOCAB = 128


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _dense(layers=4, interval=4, seed=7):
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel
    from mlx2.seeded_fixtures import legacy_conv_init

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=layers, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, vocab_size=VOCAB, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=interval,
        mtp_num_hidden_layers=0, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=512,
    )
    mx.random.seed(seed)
    with legacy_conv_init():
        model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    return model


def _sparse(seed=11):
    from mlx2.runtime.models.qwen36_35b import Model, ModelArgs

    text = {
        "model_type": "qwen3_5_moe_text", "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 8, "vocab_size": VOCAB, "linear_num_key_heads": 2,
        "linear_num_value_heads": 4, "linear_key_head_dim": 8,
        "linear_value_head_dim": 8, "linear_conv_kernel_dim": 3,
        "full_attention_interval": 4, "mtp_num_hidden_layers": 0,
        "partial_rotary_factor": 0.5, "rope_parameters": None,
        "max_position_embeddings": 128, "num_experts": 4, "num_experts_per_tok": 2,
        "moe_intermediate_size": 24, "shared_expert_intermediate_size": 32,
    }
    mx.random.seed(seed)
    model = Model(ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": text}))
    model.eval()
    mx.eval(model.parameters())
    return model


def _batch_cache(model, padding):
    return [
        ArraysCache(size=2, left_padding=list(padding)) if layer.is_linear
        else BatchKVCache(list(padding))
        for layer in model.layers
    ]


def _quantized_cache(model):
    return [
        ArraysCache(size=2) if layer.is_linear else QuantizedKVCache(group_size=32, bits=8)
        for layer in model.layers
    ]


def _prefill(model, chunks, cache, lengths=None):
    outs = []
    for chunk in chunks:
        tokens = mx.array(chunk, dtype=mx.int32)
        with prefill_checkpoint_rows(
            lengths or [tokens.shape[1]] * tokens.shape[0], width=tokens.shape[1]
        ) as request:
            logits = model(tokens, cache)
            mx.eval(logits)
        outs.append((logits, request.applied))
    return outs


def _state(cache):
    flat = []
    for entry in cache:
        flat.extend(v for _, v in tree_flatten(entry.state))
        flat.append(mx.array(np.asarray(entry.offset)) if hasattr(entry, "offset") else None)
    mx.eval([v for v in flat if v is not None])
    return flat


def _assert_bitwise(a, b):
    sa, sb = _state(a), _state(b)
    assert len(sa) == len(sb) > 0
    for x, y in zip(sa, sb):
        if x is None or y is None:
            assert x is None and y is None
            continue
        assert x.shape == y.shape and x.dtype == y.dtype
        np.testing.assert_array_equal(np.array(x), np.array(y))


def _family(model):
    return model.model.tail_family


def _compare(monkeypatch, model, chunks, make_cache, lengths=None):
    """Full (env 0) versus tail (env 1): returns the tail engagement count."""
    monkeypatch.setenv(pts.ENV, "0")
    full_cache = make_cache()
    full = _prefill(model, chunks, full_cache, lengths)
    monkeypatch.setenv(pts.ENV, "1")
    stat = f"{_family(model)}_tail_split"
    before = pts.STATS[stat]
    tail_cache = make_cache()
    tail = _prefill(model, chunks, tail_cache, lengths)
    engaged = pts.STATS[stat] - before
    for (want, _), (got, applied) in zip(full, tail):
        assert applied == 1
        assert got.shape == want.shape and got.shape[1] == 1
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-5)
    _assert_bitwise(full_cache, tail_cache)
    # The next S=1 decode step (never a tail) is bitwise identical.
    batch = len(chunks[0])
    step = mx.array([[5]] * batch, dtype=mx.int32)
    a, b = model(step, full_cache), model(step, tail_cache)
    mx.eval(a, b)
    np.testing.assert_array_equal(np.array(a), np.array(b))
    _assert_bitwise(full_cache, tail_cache)
    return engaged


def _chunks(seed, batch, widths):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, VOCAB, (batch, w)).tolist() for w in widths]


def test_dense_two_chunks_attention_last_layer(monkeypatch):
    model = _dense()
    assert not model.layers[-1].is_linear
    engaged = _compare(monkeypatch, model, _chunks(0, 1, (9, 7)), model.make_cache)
    assert engaged == 2


def test_dense_linear_last_layer_two_chunks(monkeypatch):
    model = _dense(layers=3, interval=2)
    assert model.layers[-1].is_linear
    engaged = _compare(monkeypatch, model, _chunks(1, 1, (8, 5)), model.make_cache)
    assert engaged == 2


def test_dense_left_padded_batch(monkeypatch):
    model = _dense()
    engaged = _compare(
        monkeypatch, model, _chunks(2, 2, (8, 6)), lambda: _batch_cache(model, (3, 0))
    )
    assert engaged == 2


def test_dense_quantized_kv(monkeypatch):
    model = _dense()
    engaged = _compare(monkeypatch, model, _chunks(3, 1, (9, 6)), lambda: _quantized_cache(model))
    assert engaged == 2


def test_sparse_two_chunks_and_left_padded_batch(monkeypatch):
    model = _sparse()
    lm = model.language_model
    assert type(lm.model).__name__ == "Qwen36TextModel"
    assert _compare(monkeypatch, model, _chunks(4, 1, (9, 7)), lm.make_cache) == 2
    assert _compare(
        monkeypatch, model, _chunks(5, 2, (8, 5)), lambda: _batch_cache(lm, (2, 0))
    ) == 2


def test_qwen35_122b_wrapper_reaches_the_tail(monkeypatch):
    from mlx2.runtime.models.qwen35_122b import Model as Qwen35Model

    model = _sparse()
    model.__class__ = Qwen35Model  # same trunk; only layout/pruning differ
    assert _compare(monkeypatch, model, _chunks(6, 1, (6,)), model.language_model.make_cache) == 1


# -- declines -----------------------------------------------------------------


def _engaged_delta(model, fn):
    stat = f"{_family(model)}_tail_split"
    before = pts.STATS[stat]
    fn()
    return pts.STATS[stat] - before


def _one_chunk(model, cache=None, lengths=None, width=6, batch=1):
    tokens = mx.array(_chunks(7, batch, (width,))[0], dtype=mx.int32)
    cache = model.make_cache() if cache is None else cache
    with prefill_checkpoint_rows(lengths or [width] * batch, width=width) as request:
        out = model(tokens, cache)
        mx.eval(out)
    return out, request.applied


@pytest.mark.parametrize("value", [None, "0", "head"])
def test_declines_when_switch_not_one(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(pts.ENV, raising=False)
    else:
        monkeypatch.setenv(pts.ENV, value)
    model = _dense()
    result = {}
    # Unset = family default on (2026-11-08); "0" and "head" decline.
    expected = 1 if value is None else 0
    assert _engaged_delta(model, lambda: result.update(r=_one_chunk(model))) == expected
    out, applied = result["r"]
    assert out.shape == (1, 1, VOCAB) and applied == 1  # ordinary head narrowing


def test_declines_without_open_request(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    model = _dense()
    tokens = mx.array(_chunks(8, 1, (6,))[0], dtype=mx.int32)
    holder = {}
    assert _engaged_delta(
        model, lambda: holder.update(out=model(tokens, model.make_cache()))
    ) == 0
    assert holder["out"].shape == (1, 6, VOCAB)


def test_declines_when_a_lane_checkpoint_is_not_last(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    model = _dense()
    holder = {}
    delta = _engaged_delta(model, lambda: holder.update(r=_one_chunk(
        model, cache=_batch_cache(model, (0, 0)), lengths=[6, 4], batch=2)))
    assert delta == 0
    out, applied = holder["r"]
    assert out.shape == (2, 1, VOCAB) and applied == 1


def test_decode_step_never_takes_the_tail(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    model = _dense()
    cache = model.make_cache()
    _one_chunk(model, cache=cache)
    step = mx.array([[3]], dtype=mx.int32)
    holder = {}

    def run():
        with prefill_checkpoint_rows([1], width=1):
            holder["out"] = model(step, cache)

    assert _engaged_delta(model, run) == 0
    assert holder["out"].shape == (1, 1, VOCAB)


def test_wrapped_last_layer_declines_and_matches_full(monkeypatch):
    model = _dense()
    monkeypatch.setenv(pts.ENV, "0")
    full, _ = _one_chunk(model)
    original = model.model.layers[-1]

    class Wrapped(type(original)):
        pass

    original.__class__ = Wrapped
    try:
        monkeypatch.setenv(pts.ENV, "1")
        before = pts.STATS["declined_qwen38_tail_wrapped_last_layer"]
        holder = {}
        assert _engaged_delta(model, lambda: holder.update(r=_one_chunk(model))) == 0
        assert pts.STATS["declined_qwen38_tail_wrapped_last_layer"] == before + 1
        np.testing.assert_array_equal(np.array(holder["r"][0]), np.array(full))
    finally:
        original.__class__ = Wrapped.__mro__[1]


def test_swapped_attention_class_declines(monkeypatch):
    """QSA sidecar / row-exact swap the attention class: no tail."""
    monkeypatch.setenv(pts.ENV, "1")
    model = _dense()
    attention = model.model.layers[-1].self_attn
    base = type(attention)

    class Swapped(base):
        pass

    attention.__class__ = Swapped
    try:
        before = pts.STATS["declined_qwen38_tail_wrapped_last_layer"]
        assert _engaged_delta(model, lambda: _one_chunk(model)) == 0
        assert pts.STATS["declined_qwen38_tail_wrapped_last_layer"] == before + 1
    finally:
        attention.__class__ = base


def test_row_exact_route_and_invariant_lane_decline(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    model = _dense()
    attention = model.model.layers[-1].self_attn
    object.__setattr__(attention, "_mlx2_qwen_hybrid_route", object())
    before = pts.STATS["declined_qwen38_tail_row_exact"]
    assert _engaged_delta(model, lambda: _one_chunk(model)) == 0
    object.__delattr__(attention, "_mlx2_qwen_hybrid_route")
    object.__setattr__(model.model, "_mlx2_invariant_prefill", SimpleNamespace(enabled=True))
    assert _engaged_delta(model, lambda: _one_chunk(model)) == 0
    object.__delattr__(model.model, "_mlx2_invariant_prefill")
    assert pts.STATS["declined_qwen38_tail_row_exact"] == before + 2
    assert _engaged_delta(model, lambda: _one_chunk(model)) == 1


def test_inputs_embeds_and_bucketed_cache_decline(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    model = _dense()
    tokens = mx.array(_chunks(9, 1, (6,))[0], dtype=mx.int32)
    embeds = model.model.embed_tokens(tokens)
    before = pts.STATS["declined_qwen38_tail_inputs_embeds"]

    def media():
        with prefill_checkpoint_rows([6], width=6):
            mx.eval(model(tokens, model.make_cache(), input_embeddings=embeds))

    assert _engaged_delta(model, media) == 0
    assert pts.STATS["declined_qwen38_tail_inputs_embeds"] == before + 1

    def bucketed():
        cache = _batch_cache(model, (0, 0))
        cache[-1] = BatchKVCache([0, 0], attention_backend="bucketed")
        _one_chunk(model, cache=cache, batch=2)

    before = pts.STATS["declined_qwen38_tail_cache_layout"]
    assert _engaged_delta(model, bucketed) == 0
    assert pts.STATS["declined_qwen38_tail_cache_layout"] == before + 1


def test_taps_prefill_body_and_mtp_seed_never_engage(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    model = _dense()
    tokens = mx.array(_chunks(10, 1, (6,))[0], dtype=mx.int32)
    holder = {}

    def taps():
        with prefill_checkpoint_rows([6], width=6):
            holder["taps"] = model.forward_with_taps(tokens, model.make_cache(), (1, 3))
            holder["body"] = model.prefill_body(tokens, model.make_cache(), (3,))
            # MTP backbone / mtp_seed readers call the trunk directly.
            holder["trunk"] = model.model(tokens, model.make_cache())

    assert _engaged_delta(model, taps) == 0
    logits, features = holder["taps"]
    assert logits.shape == (1, 6, VOCAB) and features.shape[1] == 6
    assert holder["body"].shape[1] == 6 and holder["trunk"].shape[1] == 6


def test_inpass_checkpoint_forward_never_engages(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    model = _dense()
    tokens = mx.array(_chunks(11, 1, (6,))[0], dtype=mx.int32)

    def run():
        with prefill_checkpoint_rows([6], width=6):
            try:
                model.prefill_checkpoint_forward(tokens, model.make_cache(), (2,))
            except Exception as exc:  # capture may refuse a tiny CPU shape
                from mlx2.runtime.inpass_checkpoints import CaptureRefused

                if not isinstance(exc, CaptureRefused):
                    raise

    assert _engaged_delta(model, run) == 0
