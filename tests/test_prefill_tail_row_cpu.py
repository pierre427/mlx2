"""Last-layer tail-row prefill for the standard decoder (TensorFold#2229), CPU."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.runtime.models import standard_decoder as sd  # noqa: E402
from mlx2.runtime.models.cache import BatchKVCache, KVCache  # noqa: E402
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows  # noqa: E402


def _model(model_type="qwen3"):
    mx.random.seed(7)
    args = sd.ModelArgs(model_type=model_type, hidden_size=64, num_hidden_layers=3,
                        intermediate_size=96, num_attention_heads=4, rms_norm_eps=1e-6,
                        vocab_size=101, num_key_value_heads=2, max_position_embeddings=512,
                        rope_theta=10000.0, tie_word_embeddings=True, head_dim=16)
    model = sd.Model(args)
    mx.eval(model.parameters())
    return model


def _prefill(model, chunks, cache, lengths=None):
    outs = []
    for chunk in chunks:
        tokens = mx.array(chunk, dtype=mx.int32)
        with prefill_checkpoint_rows(lengths or [tokens.shape[1]] * tokens.shape[0],
                                     width=tokens.shape[1]) as request:
            logits = model(tokens, cache)
            mx.eval(logits)
        outs.append((logits, request.applied))
    return outs


@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
def test_tail_row_keeps_cache_bitwise_and_last_row_logits(monkeypatch, model_type):
    rng = np.random.default_rng(0)
    chunks = [rng.integers(0, 101, (1, 9)).tolist(), rng.integers(0, 101, (1, 7)).tolist()]
    with mx.stream(mx.cpu):
        monkeypatch.setenv("MLX2_PREFILL_TAIL_ROW", "0")
        model = _model(model_type)
        full_cache = [KVCache() for _ in model.model.layers]
        full = _prefill(model, chunks, full_cache)
        monkeypatch.setenv("MLX2_PREFILL_TAIL_ROW", "1")
        before = sd.TAIL_ROW_STATS["engaged"]
        tail_cache = [KVCache() for _ in model.model.layers]
        tail = _prefill(model, chunks, tail_cache)
        assert sd.TAIL_ROW_STATS["engaged"] - before == 2
        for (want, _), (got, applied) in zip(full, tail):
            assert applied == 1 and got.shape == want.shape == (1, 1, 101)
            np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-5)
        for a, b in zip(full_cache, tail_cache):
            np.testing.assert_array_equal(np.array(a.keys[..., : a.offset, :]),
                                          np.array(b.keys[..., : b.offset, :]))
            np.testing.assert_array_equal(np.array(a.values[..., : a.offset, :]),
                                          np.array(b.values[..., : b.offset, :]))
        # The decode step after the prompt is unaffected and identical.
        step = mx.array([[5]], dtype=mx.int32)
        np.testing.assert_array_equal(np.array(model(step, full_cache)), np.array(model(step, tail_cache)))


def test_tail_row_with_left_padded_batch(monkeypatch):
    rng = np.random.default_rng(1)
    with mx.stream(mx.cpu):
        monkeypatch.setenv("MLX2_PREFILL_TAIL_ROW", "0")
        model = _model()
        tokens = rng.integers(0, 101, (2, 8)).tolist()
        full_cache = [BatchKVCache([3, 0]) for _ in model.model.layers]
        full = _prefill(model, [tokens], full_cache)
        monkeypatch.setenv("MLX2_PREFILL_TAIL_ROW", "1")
        tail_cache = [BatchKVCache([3, 0]) for _ in model.model.layers]
        tail = _prefill(model, [tokens], tail_cache)
        np.testing.assert_allclose(np.array(tail[0][0]), np.array(full[0][0]), rtol=1e-4, atol=1e-5)
        for a, b in zip(full_cache, tail_cache):
            np.testing.assert_array_equal(np.array(a.keys), np.array(b.keys))


def test_tail_row_declines_when_a_lane_checkpoint_is_not_last(monkeypatch):
    monkeypatch.setenv("MLX2_PREFILL_TAIL_ROW", "1")
    with mx.stream(mx.cpu):
        model = _model()
        before = sd.TAIL_ROW_STATS["engaged"]
        cache = [BatchKVCache([0, 0]) for _ in model.model.layers]
        tokens = mx.array(np.random.default_rng(2).integers(0, 101, (2, 6)), dtype=mx.int32)
        with prefill_checkpoint_rows([6, 4], width=6) as request:  # lane 1 right-padded
            mx.eval(model(tokens, cache))
        assert sd.TAIL_ROW_STATS["engaged"] == before and request.applied == 1


def test_tail_row_is_default_on(monkeypatch):
    monkeypatch.delenv("MLX2_PREFILL_TAIL_ROW", raising=False)
    with mx.stream(mx.cpu):
        model = _model()
        before = sd.TAIL_ROW_STATS["engaged"]
        _prefill(model, [[[1, 2, 3, 4]]], [KVCache() for _ in model.model.layers])
        assert sd.TAIL_ROW_STATS["engaged"] == before + 1


def test_tail_row_disabled_with_zero(monkeypatch):
    monkeypatch.setenv("MLX2_PREFILL_TAIL_ROW", "0")
    with mx.stream(mx.cpu):
        model = _model()
        before = sd.TAIL_ROW_STATS["engaged"]
        _prefill(model, [[[1, 2, 3, 4]]], [KVCache() for _ in model.model.layers])
        assert sd.TAIL_ROW_STATS["engaged"] == before


def test_tail_row_declines_a_wrapped_last_layer(monkeypatch):
    """Layer excision replaces Layer objects; the tail path must not engage."""
    monkeypatch.setenv("MLX2_PREFILL_TAIL_ROW", "1")
    with mx.stream(mx.cpu):
        model = _model()
        original = model.model.layers[-1]

        class Wrapped:
            def __call__(self, *args, **kwargs):
                return original(*args, **kwargs)

        model.model.layers[-1] = Wrapped()
        before = sd.TAIL_ROW_STATS["engaged"]
        outs = _prefill(model, [[[1, 2, 3, 4, 5]]], [KVCache() for _ in model.model.layers])
        assert sd.TAIL_ROW_STATS["engaged"] == before
        assert outs[0][1] == 1  # the ordinary head narrowing still applied
