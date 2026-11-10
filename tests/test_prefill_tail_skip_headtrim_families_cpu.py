"""Tier-2 prompt-chunk tail (MLX2_PREFILL_TAIL_SKIP=1) for North, LFM2.5 MoE, Ling, CPU.

With the switch on and every lane's checkpoint row the chunk's last, the last
decoder layer writes its cache over every row exactly as the full forward
does, and computes everything else on the last row only.  Cache state and the
next decode step must be bitwise identical to the full forward; the
checkpoint row's logits close.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from mlx2.runtime import prefill_tail_skip as pts  # noqa: E402
from mlx2.runtime.models.cache import (  # noqa: E402
    ROWS_ATTENTION_STATS,
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
)
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows  # noqa: E402


def _north(layers=5):
    from mlx2.runtime.models.cohere2_moe import Model, ModelArgs

    mx.random.seed(21)
    model = Model(ModelArgs.from_dict(dict(
        model_type="cohere2_moe", hidden_size=64, head_dim=16,
        num_hidden_layers=layers, intermediate_size=32,
        prefix_dense_intermediate_size=96, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=96, rope_theta=50000.0,
        rms_norm_eps=1e-6, sliding_window=6, num_experts=8,
        num_experts_per_tok=2, first_k_dense_replace=1,
        layer_types=["full_attention" if i % 4 == 0 else "sliding_attention"
                     for i in range(layers)],
    )))
    mx.eval(model.parameters())
    return model


def _lfm(last="conv"):
    from mlx2.runtime.models.lfm25_moe import Model, ModelArgs

    mx.random.seed(22)
    model = Model(ModelArgs(
        model_type="lfm2_moe", vocab_size=64, hidden_size=16, intermediate_size=16,
        moe_intermediate_size=6, num_hidden_layers=4, num_experts=4,
        num_experts_per_tok=2, norm_topk_prob=True, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=128, use_expert_bias=True,
        num_dense_layers=1, norm_eps=1e-5, conv_bias=False, conv_L_cache=3,
        tie_word_embeddings=True, rope_parameters={"rope_theta": 10_000.0},
        layer_types=["conv", "full_attention", "conv", last],
    ))
    mx.eval(model.parameters())
    return model


def _ling(q_lora_rank=None):
    from mlx2.runtime.models.bailing_hybrid import Model, ModelArgs

    mx.random.seed(23)
    model = Model(ModelArgs.from_dict(dict(
        model_type="bailing_hybrid", vocab_size=128, hidden_size=128,
        intermediate_size=128, moe_intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=2, num_experts=4,
        num_experts_per_tok=2, num_shared_experts=1, n_group=1, topk_group=1,
        first_k_dense_replace=1, layer_group_size=2, max_position_embeddings=64,
        rms_norm_eps=1e-6, rope_theta=10000, routed_scaling_factor=1.0,
        head_dim=64, kv_lora_rank=64, qk_rope_head_dim=32,
        qk_nope_head_dim=64, v_head_dim=64, no_kda_lora=True,
        gated_attention_proj_granularity_type="head_wise",
        kda_safe_gate=True, kda_lower_bound=-5.0, q_lora_rank=q_lora_rank,
    )))
    mx.eval(model.parameters())
    return model


# (builder, stats key).  North: last layer full attention (5 layers) or
# sliding (6 layers, window 6 < chunk width).  LFM2.5: last layer ShortConv
# (the real 24B-A2B topology) or attention.  Ling: last layer MLA (always,
# by the layer-group rule), with and without a q LoRA.
VARIANTS = {
    "north_full": (lambda: _north(5), "north_tail_split"),
    "north_sliding": (lambda: _north(6), "north_tail_split"),
    "lfm25_conv": (lambda: _lfm("conv"), "lfm25_tail_split"),
    "lfm25_attention": (lambda: _lfm("full_attention"), "lfm25_tail_split"),
    "ling": (lambda: _ling(), "ling_tail_split"),
    "ling_q_lora": (lambda: _ling(32), "ling_tail_split"),
}


def _batch_cache(model, padding):
    caches = []
    for template in model.make_cache():
        name = type(template).__name__
        if name == "KVCache":
            caches.append(BatchKVCache(padding))
        elif name == "RotatingKVCache":
            caches.append(BatchRotatingKVCache(template.max_size, padding))
        elif name == "ArraysCache":
            caches.append(ArraysCache(len(template.cache), left_padding=padding))
        else:
            raise AssertionError(name)
    return caches


def _state(cache):
    return [np.array(v) for _, v in tree_flatten([c.state for c in cache])]


def _chunks(model, chunks, cache, lengths=None):
    outs = []
    for chunk in chunks:
        tokens = mx.array(chunk, dtype=mx.int32)
        rows = lengths or [tokens.shape[1]] * tokens.shape[0]
        with prefill_checkpoint_rows(rows, width=tokens.shape[1]) as request:
            logits = model(tokens, cache)
            mx.eval(logits)
        outs.append((logits, request.applied))
    return outs


def _compare(monkeypatch, model, key, chunks, make_cache, step):
    monkeypatch.setenv(pts.ENV, "0")
    full_cache = make_cache()
    full = _chunks(model, chunks, full_cache)
    monkeypatch.setenv(pts.ENV, "1")
    before = pts.STATS[key]
    tail_cache = make_cache()
    tail = _chunks(model, chunks, tail_cache)
    assert pts.STATS[key] - before == len(chunks)
    for (want, want_applied), (got, applied) in zip(full, tail):
        assert want_applied == applied == 1
        assert got.shape == want.shape == (want.shape[0], 1, model.args.vocab_size)
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-4)
    assert len(_state(full_cache)) == len(_state(tail_cache))
    for a, b in zip(_state(full_cache), _state(tail_cache)):
        np.testing.assert_array_equal(a, b)
    step = mx.array(step, dtype=mx.int32)
    np.testing.assert_array_equal(np.array(model(step, full_cache)),
                                  np.array(model(step, tail_cache)))


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_tail_two_chunks_state_bitwise_and_checkpoint_row(monkeypatch, variant):
    build, key = VARIANTS[variant]
    with mx.stream(mx.cpu):
        model = build()
        rng = np.random.default_rng(0)
        vocab = model.args.vocab_size
        chunks = [rng.integers(1, vocab, (1, 9)).tolist(),
                  rng.integers(1, vocab, (1, 7)).tolist()]
        _compare(monkeypatch, model, key, chunks, model.make_cache, [[5]])


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_tail_left_padded_batch(monkeypatch, variant):
    build, key = VARIANTS[variant]
    with mx.stream(mx.cpu):
        model = build()
        rng = np.random.default_rng(1)
        vocab = model.args.vocab_size
        chunks = [rng.integers(1, vocab, (2, 8)).tolist(),
                  rng.integers(1, vocab, (2, 5)).tolist()]
        _compare(monkeypatch, model, key, chunks,
                 lambda: _batch_cache(model, [3, 0]), [[5], [6]])


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_tail_declines_without_switch_request_or_final_row(monkeypatch, variant):
    build, key = VARIANTS[variant]
    with mx.stream(mx.cpu):
        model = build()
        vocab = model.args.vocab_size
        tokens = [[1, 2, 3, 4, 5, 6]]
        before = pts.STATS[key]
        for value in (None, "0", "head"):
            if value is None:
                monkeypatch.delenv(pts.ENV, raising=False)
            else:
                monkeypatch.setenv(pts.ENV, value)
            (logits, applied), = _chunks(model, [tokens], model.make_cache())
            # The ordinary head trim still applies.
            assert applied == 1 and logits.shape == (1, 1, vocab)
        monkeypatch.setenv(pts.ENV, "1")
        # No open request (decode / verify / drafting forwards): full width.
        assert model(mx.array(tokens, dtype=mx.int32), model.make_cache()).shape == (1, 6, vocab)
        # A lane whose checkpoint row is not the chunk's last: no claim.
        (logits, applied), = _chunks(model, [tokens * 2], _batch_cache(model, [0, 0]),
                                     lengths=[6, 4])
        assert applied == 1 and logits.shape == (2, 1, vocab)
        # Decode (S=1) inside a request: never the tail.
        _chunks(model, [[[1]]], model.make_cache())
        assert pts.STATS[key] == before


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_tail_never_engages_for_taps_or_prefill_body(monkeypatch, variant):
    build, key = VARIANTS[variant]
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = build()
        tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        before = pts.STATS[key]
        with prefill_checkpoint_rows([5], width=5):
            logits, features = model.forward_with_taps(tokens, model.make_cache(), (0, 1))
            assert logits.shape[1] == 5 and features.shape[1] == 5
        with prefill_checkpoint_rows([5], width=5):
            assert model.prefill_body(tokens, model.make_cache(), (0, 1)).shape[1] == 5
        assert pts.STATS[key] == before


def _last_layer_holder(model):
    return model.model.layers


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_tail_declines_a_wrapped_last_layer(monkeypatch, variant):
    build, key = VARIANTS[variant]
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = build()
        layers = _last_layer_holder(model)
        original = layers[-1]

        class Wrapped:
            def __init__(self):
                # Mirror the attributes the trunk reads off a layer.
                for name in ("attention_type", "is_attention_layer", "is_global"):
                    if hasattr(original, name):
                        setattr(self, name, getattr(original, name))

            def __call__(self, *args, **kwargs):
                return original(*args, **kwargs)

        layers[-1] = Wrapped()
        before = pts.STATS[key]
        declined = pts.STATS["declined_wrapped_last_layer"]
        (logits, applied), = _chunks(model, [[[1, 2, 3, 4, 5]]], model.make_cache())
        assert pts.STATS[key] == before
        assert pts.STATS["declined_wrapped_last_layer"] == declined + 1
        assert applied == 1 and logits.shape[1] == 1  # the head trim still applied


def test_north_declines_with_residual_taps(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _north(5)
        model.model.residual_taps.steer = (1, mx.zeros((1, 1, 64)))
        before = pts.STATS["north_tail_split"]
        declined = pts.STATS["declined_residual_taps"]
        try:
            (logits, applied), = _chunks(model, [[[1, 2, 3, 4, 5]]], model.make_cache())
        finally:
            model.model.residual_taps.steer = None
        assert pts.STATS["north_tail_split"] == before
        assert pts.STATS["declined_residual_taps"] == declined + 1
        assert applied == 1 and logits.shape == (1, 1, 96)


def test_north_trunk_refuses_tail_with_row_exact_or_taps():
    with mx.stream(mx.cpu):
        model = _north(5)
        model.model.residual_taps.capture = {1: None}
        try:
            with pytest.raises(RuntimeError, match="plain forward"):
                model.model(mx.array([[1, 2, 3]], dtype=mx.int32), model.make_cache(),
                            tail_row=True)
        finally:
            model.model.residual_taps.capture = None


def test_tail_with_a_rows_attention_batch_cache(monkeypatch):
    """A BatchKVCache decode-attention backend may serve the one tail query."""
    with mx.stream(mx.cpu):
        model = _north(5)
        rng = np.random.default_rng(3)

        def make():
            caches = _batch_cache(model, [2, 0])
            caches[-1] = BatchKVCache([2, 0], attention_backend="rows")
            return caches

        before = ROWS_ATTENTION_STATS["rows_calls"]
        _compare(monkeypatch, model, "north_tail_split",
                 [rng.integers(1, 96, (2, 7)).tolist()], make, [[5], [6]])
        # The tail query (and the decode steps) went through the rows backend.
        assert ROWS_ATTENTION_STATS["rows_calls"] > before


def test_lfm25_declines_embeddings_and_speculating_cache(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _lfm("conv")
        tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
        embeds = model.model.embed_tokens(tokens)
        before = dict(pts.STATS)
        with prefill_checkpoint_rows([3], width=3) as request:
            logits = model(tokens, model.make_cache(), input_embeddings=embeds)
            assert logits.shape == (1, 1, 64)  # head trim only
        assert request.applied == 1
        with prefill_checkpoint_rows([3], width=3) as request:
            assert not model._claim_tail(tokens, [SimpleNamespace(speculating=True)], None)
        assert request.applied == 0
        assert pts.STATS["declined_inputs_embeds"] == before.get("declined_inputs_embeds", 0) + 1
        assert pts.STATS["declined_speculating_cache"] == before.get("declined_speculating_cache", 0) + 1
        assert pts.STATS["lfm25_tail_split"] == before.get("lfm25_tail_split", 0)


def test_ling_declines_a_recurrent_last_layer(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _ling()
        last = model.model.layers[-1]
        tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
        declined = pts.STATS["declined_recurrent_last_layer"]
        last.is_global = False
        try:
            with prefill_checkpoint_rows([3], width=3) as request:
                assert not model._claim_tail(tokens)
        finally:
            last.is_global = True
        assert request.applied == 0
        assert pts.STATS["declined_recurrent_last_layer"] == declined + 1
