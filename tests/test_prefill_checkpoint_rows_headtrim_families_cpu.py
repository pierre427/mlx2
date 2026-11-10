"""Checkpoint-row head trim for North (cohere2_moe), LFM2.5 MoE and Ling, CPU.

Inside a prompt-chunk request the head returns ``[B, 1, V]`` (each lane's
checkpoint row), outside one the full ``[B, S, V]``; cache state is identical
either way, because only the head input is narrowed.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from mlx2.runtime.models.cache import ArraysCache, BatchKVCache, BatchRotatingKVCache  # noqa: E402
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows  # noqa: E402


def _north():
    from mlx2.runtime.models.cohere2_moe import Model, ModelArgs

    mx.random.seed(11)
    model = Model(ModelArgs.from_dict(dict(
        model_type="cohere2_moe", hidden_size=64, head_dim=16,
        num_hidden_layers=5, intermediate_size=32,
        prefix_dense_intermediate_size=96, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=96, rope_theta=50000.0,
        rms_norm_eps=1e-6, sliding_window=6, num_experts=8,
        num_experts_per_tok=2, first_k_dense_replace=1,
        layer_types=["full_attention" if i % 4 == 0 else "sliding_attention"
                     for i in range(5)],
    )))
    mx.eval(model.parameters())
    return model


def _lfm():
    from mlx2.runtime.models.lfm25_moe import Model, ModelArgs

    mx.random.seed(12)
    model = Model(ModelArgs(
        model_type="lfm2_moe", vocab_size=64, hidden_size=16, intermediate_size=16,
        moe_intermediate_size=6, num_hidden_layers=4, num_experts=4,
        num_experts_per_tok=2, norm_topk_prob=True, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=128, use_expert_bias=True,
        num_dense_layers=1, norm_eps=1e-5, conv_bias=False, conv_L_cache=3,
        tie_word_embeddings=True, rope_parameters={"rope_theta": 10_000.0},
        layer_types=["conv", "full_attention", "conv", "conv"],
    ))
    mx.eval(model.parameters())
    return model


def _ling():
    from mlx2.runtime.models.bailing_hybrid import Model, ModelArgs

    mx.random.seed(13)
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
        kda_safe_gate=True, kda_lower_bound=-5.0,
    )))
    mx.eval(model.parameters())
    return model


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


FAMILIES = {"north": _north, "lfm25": _lfm, "ling": _ling}


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("padding", [[0], [3, 0]])
def test_head_returns_checkpoint_rows_inside_a_request_only(family, padding):
    with mx.stream(mx.cpu):
        model = FAMILIES[family]()
        batch, width = len(padding), 8
        vocab = model.args.vocab_size
        tokens = mx.array(
            np.random.default_rng(len(padding)).integers(1, vocab, (batch, width)),
            dtype=mx.int32,
        )
        make = (lambda: model.make_cache()) if batch == 1 else (
            lambda: _batch_cache(model, padding))
        full_cache, narrow_cache = make(), make()
        full = model(tokens, full_cache)
        mx.eval(full)
        assert full.shape == (batch, width, vocab)
        with prefill_checkpoint_rows([width] * batch, width=width) as request:
            narrow = model(tokens, narrow_cache)
            mx.eval(narrow)
        assert request.applied == 1
        assert narrow.shape == (batch, 1, vocab)
        np.testing.assert_allclose(np.array(narrow), np.array(full[:, -1:]),
                                   rtol=1e-4, atol=1e-4)
        assert len(_state(full_cache)) == len(_state(narrow_cache))
        for a, b in zip(_state(full_cache), _state(narrow_cache)):
            np.testing.assert_array_equal(a, b)
        # Decode after the chunk: full width (S=1) and bitwise identical.
        step = mx.array([[5]] * batch, dtype=mx.int32)
        np.testing.assert_array_equal(np.array(model(step, full_cache)),
                                      np.array(model(step, narrow_cache)))


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_head_keeps_a_mid_chunk_checkpoint_row(family):
    """A right-padded lane's checkpoint row is not the last; select it exactly."""
    with mx.stream(mx.cpu):
        model = FAMILIES[family]()
        vocab = model.args.vocab_size
        tokens = mx.array(np.random.default_rng(5).integers(1, vocab, (2, 6)), dtype=mx.int32)
        full = model(tokens, _batch_cache(model, [0, 0]))
        with prefill_checkpoint_rows([6, 4], width=6) as request:
            narrow = model(tokens, _batch_cache(model, [0, 0]))
            mx.eval(narrow)
        assert request.applied == 1 and narrow.shape == (2, 1, vocab)
        np.testing.assert_allclose(np.array(narrow[0, 0]), np.array(full[0, 5]), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(np.array(narrow[1, 0]), np.array(full[1, 3]), rtol=1e-4, atol=1e-4)
