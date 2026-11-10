"""Prompt-chunk tier-2 tail for gpt-oss, gpt-oss-puzzle and Laguna (CPU).

Tiny random-weight models; no checkpoint.  The tail must leave every cache
bitwise identical to the full forward (sliding last layers included), keep
the next decode step bitwise identical, and return checkpoint-row logits
close to the full forward's last row.  Opt-in via MLX2_PREFILL_TAIL_SKIP=1.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.runtime.models import gpt_oss, gpt_oss_puzzle, laguna  # noqa: E402
from mlx2.runtime.models.cache import (  # noqa: E402
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
)
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows  # noqa: E402
from mlx2.runtime.prefill_tail_skip import STATS  # noqa: E402

VOCAB = 64
WINDOW = 4


def _randomize_sinks(model):
    for layer in model.layers:
        attn = layer.self_attn
        for name in ("sinks", "sink"):
            value = getattr(attn, name, None)
            if value is not None:
                setattr(attn, name, mx.random.normal(value.shape) * 0.5)


def _gpt_oss(layer_types):
    mx.random.seed(11)
    args = gpt_oss.ModelArgs(
        num_hidden_layers=len(layer_types), num_local_experts=4, num_experts_per_tok=2,
        vocab_size=VOCAB, hidden_size=32, intermediate_size=32, head_dim=8,
        num_attention_heads=4, num_key_value_heads=2, sliding_window=WINDOW,
        layer_types=list(layer_types),
    )
    model = gpt_oss.Model(args)
    _randomize_sinks(model)
    mx.eval(model.parameters())
    return model, "gpt_oss_tail_split"


def _puzzle(windows):
    mx.random.seed(17)
    args = gpt_oss_puzzle.ModelArgs(
        num_hidden_layers=len(windows), num_experts_per_tok=2, vocab_size=VOCAB,
        hidden_size=32, intermediate_size=32, head_dim=8, num_attention_heads=4,
        num_key_value_heads=2,
        block_configs=[{"num_local_experts": 4 if i % 2 else 2, "sliding_window": w}
                       for i, w in enumerate(windows)],
    )
    model = gpt_oss_puzzle.Model(args)
    _randomize_sinks(model)
    mx.eval(model.parameters())
    return model, "gpt_oss_puzzle_tail_split"


def _laguna(layer_types, gating="per-head", sink=False):
    mx.random.seed(5)
    args = laguna.ModelArgs(
        vocab_size=VOCAB, hidden_size=16, intermediate_size=32,
        num_hidden_layers=len(layer_types), num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, sliding_window=WINDOW, layer_types=list(layer_types),
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=8,
        shared_expert_intermediate_size=8, max_position_embeddings=256,
        gating=gating, swa_attention_sink_enabled=sink,
    )
    model = laguna.Model(args)
    _randomize_sinks(model)
    mx.eval(model.parameters())
    return model, "laguna_tail_split"


# Last layer full and last layer sliding for every family (Laguna's real
# S/XS checkpoints end on a sliding layer; gpt-oss-20B ends on a full one).
FACTORIES = {
    "gpt_oss_last_full": lambda: _gpt_oss(["sliding_attention", "full_attention"] * 2),
    "gpt_oss_last_sliding": lambda: _gpt_oss(
        ["sliding_attention", "full_attention", "sliding_attention"]),
    "puzzle_last_full": lambda: _puzzle([WINDOW, None, 6, None]),
    "puzzle_last_sliding": lambda: _puzzle([None, 6, WINDOW]),
    "laguna_last_sliding": lambda: _laguna(
        ["full_attention", "sliding_attention", "sliding_attention"]),
    "laguna_last_full_sink_elementgate": lambda: _laguna(
        ["sliding_attention", "full_attention"], gating="per-element", sink=True),
    "laguna_last_sliding_sink": lambda: _laguna(
        ["full_attention", "sliding_attention"], sink=True),
}


def _batch_caches(model, left_padding):
    caches = []
    for entry in model.make_cache():
        if isinstance(entry, RotatingKVCache):
            caches.append(BatchRotatingKVCache(entry.max_size, list(left_padding)))
        else:
            assert type(entry) is KVCache
            caches.append(BatchKVCache(list(left_padding)))
    return caches


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


_STATE_FIELDS = ("keys", "values", "offset", "_idx", "_offset", "left_padding", "rotated")


def _assert_caches_bitwise(full_cache, tail_cache):
    for a, b in zip(full_cache, tail_cache):
        assert type(a) is type(b)
        for name in _STATE_FIELDS:
            if not hasattr(a, name):
                continue
            x, y = getattr(a, name), getattr(b, name)
            if isinstance(x, mx.array):
                np.testing.assert_array_equal(np.array(x), np.array(y), err_msg=name)
            else:
                assert x == y, name


def _compare(monkeypatch, factory, chunks, make_caches, decode_tokens):
    with mx.stream(mx.cpu):
        model, stat = factory()
        monkeypatch.setenv("MLX2_PREFILL_TAIL_SKIP", "0")
        full_cache = make_caches(model)
        full = _prefill(model, chunks, full_cache)
        monkeypatch.setenv("MLX2_PREFILL_TAIL_SKIP", "1")
        before = STATS[stat]
        tail_cache = make_caches(model)
        tail = _prefill(model, chunks, tail_cache)
        assert STATS[stat] - before == len(chunks)
        batch = len(chunks[0])
        for (want, _), (got, applied) in zip(full, tail):
            assert applied == 1
            assert got.shape == want.shape == (batch, 1, VOCAB)
            np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-5)
        _assert_caches_bitwise(full_cache, tail_cache)
        # The decode step after the prompt is bitwise identical, and leaves
        # the caches identical too.
        step = mx.array(decode_tokens, dtype=mx.int32)
        np.testing.assert_array_equal(
            np.array(model(step, full_cache)), np.array(model(step, tail_cache)))
        _assert_caches_bitwise(full_cache, tail_cache)


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_tail_two_chunks_single_lane_bitwise_state(monkeypatch, name):
    rng = np.random.default_rng(0)
    # Second chunk crosses the window, so sliding masks are arrays and the
    # rotating cache trims; the first fits (a "causal" mask on rotating).
    chunks = [rng.integers(0, VOCAB, (1, 3)).tolist(),
              rng.integers(0, VOCAB, (1, 7)).tolist(),
              rng.integers(0, VOCAB, (1, 5)).tolist()]
    _compare(monkeypatch, FACTORIES[name], chunks,
             lambda model: model.make_cache(), [[5]])


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_tail_left_padded_batch_bitwise_state(monkeypatch, name):
    rng = np.random.default_rng(1)
    chunks = [rng.integers(0, VOCAB, (2, 6)).tolist(),
              rng.integers(0, VOCAB, (2, 5)).tolist()]
    _compare(monkeypatch, FACTORIES[name], chunks,
             lambda model: _batch_caches(model, [3, 0]), [[5], [9]])


def test_tail_without_cache_matches_last_row(monkeypatch):
    with mx.stream(mx.cpu):
        for name in ("gpt_oss_last_sliding", "laguna_last_sliding", "puzzle_last_sliding"):
            model, _ = FACTORIES[name]()
            tokens = mx.array(np.random.default_rng(3).integers(0, VOCAB, (1, 9)), dtype=mx.int32)
            monkeypatch.setenv("MLX2_PREFILL_TAIL_SKIP", "0")
            want = model(tokens)[:, -1:]
            monkeypatch.setenv("MLX2_PREFILL_TAIL_SKIP", "1")
            with prefill_checkpoint_rows([9], width=9) as request:
                got = model(tokens)
            assert request.applied == 1 and got.shape == (1, 1, VOCAB)
            np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------ declines


def _engagements():
    return sum(STATS[k] for k in ("gpt_oss_tail_split", "gpt_oss_puzzle_tail_split",
                                  "laguna_tail_split"))


@pytest.mark.parametrize("value", [None, "0", "head"])
@pytest.mark.parametrize("name", ["gpt_oss_last_full", "puzzle_last_sliding",
                                  "laguna_last_sliding"])
def test_declines_when_switch_off(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv("MLX2_PREFILL_TAIL_SKIP", raising=False)
    else:
        monkeypatch.setenv("MLX2_PREFILL_TAIL_SKIP", value)
    with mx.stream(mx.cpu):
        model, _ = FACTORIES[name]()
        before = _engagements()
        outs = _prefill(model, [[[1, 2, 3, 4, 5, 6]]], model.make_cache())
        assert _engagements() == before
        # The ordinary head narrowing still applies.
        assert outs[0][1] == 1 and outs[0][0].shape == (1, 1, VOCAB)


@pytest.mark.parametrize("name", ["gpt_oss_last_full", "puzzle_last_sliding",
                                  "laguna_last_sliding"])
def test_declines_without_open_request_and_on_decode(monkeypatch, name):
    monkeypatch.setenv("MLX2_PREFILL_TAIL_SKIP", "1")
    with mx.stream(mx.cpu):
        model, _ = FACTORIES[name]()
        before = _engagements()
        cache = model.make_cache()
        out = model(mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32), cache)
        assert out.shape == (1, 6, VOCAB)  # verify-shaped forward: full width
        with prefill_checkpoint_rows([1], width=1):
            assert model(mx.array([[7]], dtype=mx.int32), cache).shape == (1, 1, VOCAB)
        assert _engagements() == before


@pytest.mark.parametrize("name", ["gpt_oss_last_sliding", "puzzle_last_full",
                                  "laguna_last_sliding"])
def test_declines_when_a_lane_checkpoint_is_not_last(monkeypatch, name):
    monkeypatch.setenv("MLX2_PREFILL_TAIL_SKIP", "1")
    with mx.stream(mx.cpu):
        model, _ = FACTORIES[name]()
        before = _engagements()
        cache = _batch_caches(model, [0, 0])
        tokens = mx.array(np.random.default_rng(2).integers(0, VOCAB, (2, 6)), dtype=mx.int32)
        with prefill_checkpoint_rows([6, 4], width=6) as request:  # lane 1 right-padded
            out = model(tokens, cache)
        assert _engagements() == before
        assert request.applied == 1 and out.shape == (2, 1, VOCAB)


@pytest.mark.parametrize("name", ["gpt_oss_last_full", "puzzle_last_sliding",
                                  "laguna_last_sliding"])
def test_taps_and_prefill_body_never_engage(monkeypatch, name):
    """Capture taps (DFlash / AST drafting) keep the full chunk even with a
    request open and the switch on, and their outputs are unchanged."""
    with mx.stream(mx.cpu):
        model, _ = FACTORIES[name]()
        tokens = mx.array([[3, 1, 4, 1, 5, 9, 2]], dtype=mx.int32)
        layers = (0, len(model.layers) - 1)
        monkeypatch.setenv("MLX2_PREFILL_TAIL_SKIP", "0")
        want_logits, want_feat = model.forward_with_taps(tokens, model.make_cache(), layers)
        want_body = model.prefill_body(tokens, model.make_cache(), layers)
        monkeypatch.setenv("MLX2_PREFILL_TAIL_SKIP", "1")
        before = _engagements()
        with prefill_checkpoint_rows([7], width=7) as request:
            logits, feat = model.forward_with_taps(tokens, model.make_cache(), layers)
            body = model.prefill_body(tokens, model.make_cache(), layers)
        assert _engagements() == before and request.applied == 0
        assert logits.shape == (1, 7, VOCAB)
        np.testing.assert_array_equal(np.array(logits), np.array(want_logits))
        np.testing.assert_array_equal(np.array(feat), np.array(want_feat))
        np.testing.assert_array_equal(np.array(body), np.array(want_body))


@pytest.mark.parametrize("name", ["gpt_oss_last_full", "puzzle_last_sliding",
                                  "laguna_last_sliding"])
def test_declines_a_wrapped_last_layer(monkeypatch, name):
    monkeypatch.setenv("MLX2_PREFILL_TAIL_SKIP", "1")
    with mx.stream(mx.cpu):
        model, _ = FACTORIES[name]()
        original = model.layers[-1]

        class Wrapped:
            def __getattr__(self, attr):  # e.g. Laguna's attention_type
                return getattr(original, attr)

            def __call__(self, *args, **kwargs):
                return original(*args, **kwargs)

            def tail(self, *args, **kwargs):  # pragma: no cover - must not run
                raise AssertionError("wrapped last layer took the tail")

        model.layers[-1] = Wrapped()
        before = _engagements()
        declined = STATS["declined_wrapped_last_layer"]
        outs = _prefill(model, [[[1, 2, 3, 4, 5]]], model.make_cache())
        assert _engagements() == before
        assert STATS["declined_wrapped_last_layer"] == declined + 1
        assert outs[0][1] == 1


@pytest.mark.parametrize("name", ["gpt_oss_last_full", "laguna_last_sliding"])
def test_declines_a_class_patched_attention(monkeypatch, name):
    """Residual-ablation style hooks patch the module class; decline."""
    monkeypatch.setenv("MLX2_PREFILL_TAIL_SKIP", "1")
    with mx.stream(mx.cpu):
        model, _ = FACTORIES[name]()
        attn = model.layers[-1].self_attn
        attn.__class__ = type("Patched", (type(attn),), {})
        before = _engagements()
        _prefill(model, [[[1, 2, 3, 4, 5]]], model.make_cache())
        assert _engagements() == before


@pytest.mark.parametrize("name", ["puzzle_last_sliding", "laguna_last_sliding"])
def test_declines_input_embeddings(monkeypatch, name):
    monkeypatch.setenv("MLX2_PREFILL_TAIL_SKIP", "1")
    with mx.stream(mx.cpu):
        model, _ = FACTORIES[name]()
        tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        embeddings = model.model.embed_tokens(tokens)
        before = _engagements()
        declined = STATS["declined_inputs_embeds"]
        with prefill_checkpoint_rows([5], width=5) as request:
            out = model(tokens, model.make_cache(), input_embeddings=embeddings)
        assert _engagements() == before
        assert STATS["declined_inputs_embeds"] == declined + 1
        assert request.applied == 1 and out.shape == (1, 1, VOCAB)


def test_tail_row_refuses_capture_in_body():
    with mx.stream(mx.cpu):
        for name in ("gpt_oss_last_full", "puzzle_last_full", "laguna_last_sliding"):
            model, _ = FACTORIES[name]()
            with pytest.raises(ValueError):
                model.model(mx.array([[1, 2, 3]]), None, capture={0: None}, tail_row=True)
