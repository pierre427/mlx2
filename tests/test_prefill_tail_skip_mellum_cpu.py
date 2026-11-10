"""Prompt-chunk tier-2 tail (MLX2_PREFILL_TAIL_SKIP) for Mellum, CPU.

The last layer writes K/V over every row and runs the query, attention
output and MoE for the last row only.  Cache state and the next decode step
must be bitwise identical to the full forward; the checkpoint row's logits
are close (the row count changes rounding).
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.runtime import prefill_tail_skip as pts  # noqa: E402
from mlx2.runtime.models import mellum  # noqa: E402
from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache  # noqa: E402
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows  # noqa: E402

VOCAB = 61
WINDOW = 6
HIDDEN = 32


def _model(last="full_attention", rope_full="yarn"):
    mx.random.seed(11)
    full = (
        {"rope_type": "yarn", "rope_theta": 10_000.0, "factor": 4.0,
         "original_max_position_embeddings": 64, "beta_fast": 32.0, "beta_slow": 1.0}
        if rope_full == "yarn"
        else {"rope_type": "default", "rope_theta": 10_000.0}
    )
    args = mellum.ModelArgs(
        hidden_size=HIDDEN, num_hidden_layers=4, intermediate_size=48,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        vocab_size=VOCAB, max_position_embeddings=512, sliding_window=WINDOW,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", last],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": full,
        },
    )
    model = mellum.Model(args)
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


def _prefill(model, chunks, cache, lengths=None):
    outs = []
    for chunk in chunks:
        tokens = mx.array(chunk, dtype=mx.int32)
        rows = lengths or [tokens.shape[1]] * tokens.shape[0]
        with prefill_checkpoint_rows(rows, width=tokens.shape[1]) as request:
            logits = model(tokens, cache)
            mx.eval(logits)
        outs.append((logits, request.applied))
    return outs


def _split_count():
    return pts.STATS["mellum_tail_split"]


@pytest.mark.parametrize("last", ["full_attention", "sliding_attention"])
def test_two_chunks_bitwise_state_and_close_logits(monkeypatch, last):
    rng = np.random.default_rng(0)
    # The second chunk outgrows the sliding window (array masks).
    chunks = [rng.integers(0, VOCAB, (1, 5)).tolist(), rng.integers(0, VOCAB, (1, 9)).tolist()]
    with mx.stream(mx.cpu):
        model = _model(last)
        monkeypatch.setenv(pts.ENV, "0")
        full_cache = model.make_cache()
        full = _prefill(model, chunks, full_cache)
        monkeypatch.setenv(pts.ENV, "1")
        before = _split_count()
        tail_cache = model.make_cache()
        tail = _prefill(model, chunks, tail_cache)
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
        model = _model(last, rope_full="default")
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
        assert _split_count() == before
        assert applied == 1 and logits.shape == (1, 1, VOCAB)  # ordinary head trim


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


def test_mtp_backbone_keeps_every_row(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        before = _split_count()
        tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        with prefill_checkpoint_rows([5], width=5) as request:
            hidden, seed = model.mtp_backbone(tokens, model.make_cache())
        assert _split_count() == before and request.applied == 0
        assert hidden.shape == seed.shape == (1, 5, HIDDEN)


def test_declines_a_wrapped_last_layer(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        original = model.model.layers[-1]

        class Wrapped:
            def __call__(self, *args, **kwargs):
                return original(*args, **kwargs)

        model.model.layers[-1] = Wrapped()
        before = _split_count()
        declined = pts.STATS["declined_mellum_wrapped_last_layer"]
        (logits, applied), = _prefill(model, [[[1, 2, 3, 4, 5]]], model.make_cache())
        assert _split_count() == before
        assert pts.STATS["declined_mellum_wrapped_last_layer"] == declined + 1
        assert applied == 1 and logits.shape == (1, 1, VOCAB)
