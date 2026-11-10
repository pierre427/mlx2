"""Prompt-chunk tier-2 tail (MLX2_PREFILL_TAIL_SKIP) for Flint, CPU.

The last layer writes K/V over every row and runs the query, sink attention,
output projection and MoE (routed plus shared expert) for the last row only.
Cache state and the next decode step must be bitwise identical to the full
forward; the checkpoint row's logits are close.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.runtime import prefill_tail_skip as pts  # noqa: E402
from mlx2.runtime.models import flint  # noqa: E402
from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache  # noqa: E402
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows  # noqa: E402

VOCAB = 59
WINDOW = 6
HIDDEN = 32


def _model(last="full_attention", *, last_rope=False):
    mx.random.seed(13)
    layer_types = ["full_attention", "sliding_attention", "sliding_attention", last]
    no_rope = [0] if last_rope else [0, 3]
    args = flint.ModelArgs(
        model_type="flint", vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=48,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rms_norm_eps=1e-6, max_position_embeddings=512, rope_theta=10_000.0,
        layer_types=layer_types, no_rope_layer_indices=no_rope, mlp_only_layers=[0],
        moe_intermediate_size=16, shared_expert_intermediate_size=24, num_experts=4,
        num_experts_per_tok=2, sliding_window=WINDOW, tie_word_embeddings=False,
    )
    model = flint.Model(args)
    mx.eval(model.parameters())
    # Learned sinks: non-zero so the sink path is exercised.
    for layer in model.layers:
        layer.self_attn.softmax_offset = mx.random.normal((4,)) * 0.5
    mx.eval(model.parameters())
    return model


def _batch_cache(model, padding):
    return [
        BatchKVCache(list(padding)) if kind == "full_attention"
        else BatchRotatingKVCache(WINDOW, list(padding))
        for kind in model.args.layer_types
    ]


def _state(cache):
    out = []
    for entry in cache:
        for name, value in sorted(vars(entry).items()):
            if isinstance(value, mx.array):
                out.append((name, np.array(value)))
            elif isinstance(value, (int, float)):
                out.append((name, value))
    return out


def _assert_same_state(a, b):
    sa, sb = _state(a), _state(b)
    assert [n for n, _ in sa] == [n for n, _ in sb]
    assert sa
    for (name, x), (_, y) in zip(sa, sb):
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y, err_msg=name)
        else:
            assert x == y, name


def _prefill(model, chunks, cache, lengths=None, *, prefill_forward=True):
    outs = []
    for chunk in chunks:
        tokens = mx.array(chunk, dtype=mx.int32)
        rows = lengths or [tokens.shape[1]] * tokens.shape[0]
        with prefill_checkpoint_rows(rows, width=tokens.shape[1]) as request:
            call = model.prefill_forward if prefill_forward else model
            logits = call(tokens, cache=cache)
            mx.eval(logits)
        outs.append((logits, request.applied))
    return outs


def _split_count():
    return pts.STATS["flint_tail_split"]


@pytest.mark.parametrize("last,last_rope", [
    ("full_attention", False),  # the released layout: last layer NoPE full
    ("full_attention", True),
    ("sliding_attention", True),
])
@pytest.mark.parametrize("prefill_forward", [True, False])
def test_two_chunks_bitwise_state_and_close_logits(monkeypatch, last, last_rope,
                                                   prefill_forward):
    rng = np.random.default_rng(0)
    chunks = [rng.integers(0, VOCAB, (1, 5)).tolist(), rng.integers(0, VOCAB, (1, 9)).tolist()]
    with mx.stream(mx.cpu):
        model = _model(last, last_rope=last_rope)
        monkeypatch.setenv(pts.ENV, "0")
        full_cache = model.make_cache()
        full = _prefill(model, chunks, full_cache, prefill_forward=prefill_forward)
        monkeypatch.setenv(pts.ENV, "1")
        before = _split_count()
        tail_cache = model.make_cache()
        tail = _prefill(model, chunks, tail_cache, prefill_forward=prefill_forward)
        assert _split_count() - before == 2
        for (want, _), (got, applied) in zip(full, tail):
            assert applied == 1 and got.shape == want.shape == (1, 1, VOCAB)
            np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-5)
        _assert_same_state(full_cache, tail_cache)
        step = mx.array([[5]], dtype=mx.int32)
        np.testing.assert_array_equal(np.array(model(step, full_cache)),
                                      np.array(model(step, tail_cache)))
        _assert_same_state(full_cache, tail_cache)


@pytest.mark.parametrize("last", ["full_attention", "sliding_attention"])
def test_left_padded_batch(monkeypatch, last):
    rng = np.random.default_rng(1)
    padding = (3, 0)
    chunks = [rng.integers(0, VOCAB, (2, 8)).tolist(), rng.integers(0, VOCAB, (2, 4)).tolist()]
    with mx.stream(mx.cpu):
        model = _model(last, last_rope=last == "sliding_attention")
        monkeypatch.setenv(pts.ENV, "0")
        full_cache = _batch_cache(model, padding)
        full = _prefill(model, chunks, full_cache)
        monkeypatch.setenv(pts.ENV, "1")
        before = _split_count()
        tail_cache = _batch_cache(model, padding)
        tail = _prefill(model, chunks, tail_cache)
        assert _split_count() - before == 2
        for (want, _), (got, applied) in zip(full, tail):
            assert applied == 1 and got.shape == want.shape == (2, 1, VOCAB)
            np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-5)
        _assert_same_state(full_cache, tail_cache)
        step = mx.array([[5], [7]], dtype=mx.int32)
        np.testing.assert_array_equal(np.array(model(step, full_cache)),
                                      np.array(model(step, tail_cache)))


@pytest.mark.parametrize("value", [None, "0", "head"])
def test_declines_when_switch_is_off(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(pts.ENV, raising=False)
    else:
        monkeypatch.setenv(pts.ENV, value)
    with mx.stream(mx.cpu):
        model = _model()
        before = _split_count()
        (logits, applied), = _prefill(model, [[[1, 2, 3, 4]]], model.make_cache())
        assert _split_count() == before + (value is None)  # family default on (2026-11-08)
        assert applied == 1 and logits.shape == (1, 1, VOCAB)


def test_declines_without_open_request(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        before = _split_count()
        logits = model(mx.array([[1, 2, 3, 4]], dtype=mx.int32), model.make_cache())
        assert _split_count() == before and logits.shape == (1, 4, VOCAB)


def test_declines_when_a_lane_checkpoint_is_not_last(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        before = _split_count()
        tokens = np.random.default_rng(2).integers(0, VOCAB, (2, 6)).tolist()
        (logits, applied), = _prefill(model, [tokens], _batch_cache(model, (0, 0)),
                                      lengths=[6, 4])
        assert _split_count() == before
        assert applied == 1 and logits.shape == (2, 1, VOCAB)


def test_decode_step_never_takes_the_tail(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        cache = model.make_cache()
        _prefill(model, [[[1, 2, 3]]], cache)
        before = _split_count()
        (logits, _), = _prefill(model, [[[4]]], cache)
        assert _split_count() == before and logits.shape == (1, 1, VOCAB)


def test_taps_and_prefill_body_keep_every_row(monkeypatch):
    rng = np.random.default_rng(3)
    tokens = mx.array(rng.integers(0, VOCAB, (1, 7)), dtype=mx.int32)
    with mx.stream(mx.cpu):
        model = _model()
        monkeypatch.setenv(pts.ENV, "0")
        ref_logits, ref_features = model.forward_with_taps(tokens, model.make_cache(), (1, 3))
        ref_body = model.prefill_body(tokens, model.make_cache(), (1, 3))
        monkeypatch.setenv(pts.ENV, "1")
        before = _split_count()
        with prefill_checkpoint_rows([7], width=7) as request:
            logits, features = model.forward_with_taps(tokens, model.make_cache(), (1, 3))
            body = model.prefill_body(tokens, model.make_cache(), (1, 3))
        assert _split_count() == before and request.applied == 0
        assert logits.shape == (1, 7, VOCAB)
        np.testing.assert_array_equal(np.array(logits), np.array(ref_logits))
        np.testing.assert_array_equal(np.array(features), np.array(ref_features))
        np.testing.assert_array_equal(np.array(body), np.array(ref_body))


def test_decoder_refuses_tail_with_taps():
    with mx.stream(mx.cpu):
        model = _model()
        tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        with pytest.raises(ValueError, match="no taps"):
            model.model(tokens, model.make_cache(), capture={3: None}, tail_row=True)


def test_declines_a_wrapped_last_layer(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        original = model.model.layers[-1]

        class Wrapped:
            layer_type = original.layer_type

            def __call__(self, *args, **kwargs):
                return original(*args, **kwargs)

        model.model.layers[-1] = Wrapped()
        before = _split_count()
        declined = pts.STATS["declined_flint_wrapped_last_layer"]
        (logits, applied), = _prefill(model, [[[1, 2, 3, 4, 5]]], model.make_cache())
        assert _split_count() == before
        assert pts.STATS["declined_flint_wrapped_last_layer"] == declined + 1
        assert applied == 1 and logits.shape == (1, 1, VOCAB)
