"""LiquidAI SGLang-format DSpark (Lfm2DSparkDraftModel) on tiny CPU models.

Covers the strict config contract and its interleaved RoPE, the format
dispatch, the LFM2.5 target's post-block taps, and exactness of verified
output against ordinary greedy decoding of a tiny ShortConv/MoE target.
No artifact loads except a header-only check of the pinned download.
"""
from itertools import pairwise
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_qwen38_dflash2_cpu import drain, generator, ordinary_greedy

from mlx2.adapters.dspark import (
    InterleavedRopeDSparkConfig,
    dspark_config,
    expected_weight_shapes,
    inspect_drafter,
    lfm_dspark_config,
)
from mlx2.adapters.external_drafter_formats import format_for_config
from mlx2.runtime.drafters.dspark import (
    BidirectionalDSparkDraftModel,
    dspark_draft_model,
)
from mlx2.runtime.models.lfm25_moe import Model, ModelArgs

ARTIFACT = Path.home() / "mlx-models/LFM2.5-24B-A2B-DSpark"
TARGET = Path.home() / "mlx-models/LFM2.5-24B-A2B-MLX-8bit"
VOCAB = 128
TAPS = [1, 6]

# LiquidAI/LFM2.5-24B-A2B-DSpark@e052e5b1 config.json, verbatim.
REAL_CONFIG = {
    "architectures": ["Lfm2DSparkDraftModel"], "model_type": "qwen3", "dtype": "bfloat16",
    "hidden_size": 2048, "num_hidden_layers": 5, "num_attention_heads": 32,
    "num_key_value_heads": 8, "head_dim": 64, "intermediate_size": 6144, "hidden_act": "silu",
    "rms_norm_eps": 1e-05, "vocab_size": 128000, "rope_theta": 10000000.0,
    "max_position_embeddings": 131072, "layer_types": ["full_attention"] * 5, "block_size": 9,
    "dflash_config": {"mask_token_id": 125017, "target_layer_ids": [2, 10, 18, 26, 34],
                      "num_target_layers": 40},
    "markov_rank": 256, "rope_is_neox_style": False, "enable_confidence_head": True,
    "markov_head_type": "vanilla",
}


def tiny_config(**overrides):
    config = {
        **REAL_CONFIG, "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
        "layer_types": ["full_attention"] * 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "vocab_size": VOCAB, "block_size": 7, "markov_rank": 4,
        "max_position_embeddings": 512,
        "dflash_config": {"mask_token_id": VOCAB - 1, "target_layer_ids": TAPS,
                          "num_target_layers": 8},
    }
    config.update(overrides)
    return config


def tiny_target(seed=7):
    mx.random.seed(seed)
    args = ModelArgs(
        model_type="lfm2_moe", vocab_size=VOCAB, hidden_size=32, intermediate_size=48,
        moe_intermediate_size=12, num_hidden_layers=8, num_experts=4, num_experts_per_tok=2,
        norm_topk_prob=True, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=512, use_expert_bias=True, num_dense_layers=1, norm_eps=1e-5,
        conv_bias=False, conv_L_cache=3, tie_word_embeddings=True, routed_scaling_factor=1.0,
        rope_parameters={"rope_theta": 10_000.0},
        layer_types=["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv"],
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def tiny_pair(seed=7, markov_scale=2.0):
    target = tiny_target(seed)
    mx.random.seed(seed + 11)
    draft = dspark_draft_model(lfm_dspark_config(tiny_config()))
    head = draft.markov_head
    # Random-init Markov rows are tiny; scale them so the bias moves choices.
    head.markov_w1.weight = mx.random.normal(head.markov_w1.weight.shape) * markov_scale
    head.markov_w2.weight = mx.random.normal(head.markov_w2.weight.shape) * markov_scale
    draft.bind(target)
    mx.eval(draft.parameters())
    return target, draft


# Motif prompts give the tiny target repeated n-grams, so drafts both land
# and miss (rejections exercise the ShortConv rollback).
PROMPTS = [[5, 9, 13, 21, 8, 3] * 3, [11, 4, 5, 9, 13], [70, 71, 72, 73, 74, 75, 76, 77]]


# -- contract ------------------------------------------------------------------


def test_real_config_parses_to_anchor_sampled_bidirectional_interleaved_block():
    args = lfm_dspark_config(REAL_CONFIG)
    assert type(args) is InterleavedRopeDSparkConfig
    assert dspark_config(REAL_CONFIG) == args
    assert args.block_size == 10  # anchor + 9 trained proposals
    assert args.target_layer_ids == [2, 10, 18, 26, 34] and args.num_target_layers == 40
    assert args.mask_token_id == 125017 and args.rope_is_neox_style is False
    assert args.proposal_attention == "bidirectional" and args.sample_from_anchor is True
    assert format_for_config(REAL_CONFIG).name == "dspark"
    assert expected_weight_shapes(args)["fc.weight"] == [2048, 5 * 2048]
    assert expected_weight_shapes(args)["confidence_head.proj.weight"] == [1, 2048 + 256]


def test_draft_rope_uses_interleaved_pairing():
    draft = dspark_draft_model(lfm_dspark_config(tiny_config()))
    assert isinstance(draft, BidirectionalDSparkDraftModel)
    assert draft.rope.traditional is True


@pytest.mark.parametrize("change", [
    {"architectures": ["Qwen3DSparkModel"]},
    {"model_type": "lfm2_moe"},
    {"rope_is_neox_style": True},
    {"rope_is_neox_style": None},
    {"confidence_head_with_markov": False},
    {"enable_confidence_head": False},
    {"markov_head_type": "gated"},
    {"sliding_window": 128},
    {"rope_scaling": {"rope_type": "yarn"}},
    {"quantization": {"bits": 4, "group_size": 64}},
    {"layer_types": ["full_attention", "sliding_attention"]},
    {"dflash_config": {"mask_token_id": VOCAB - 1, "target_layer_ids": TAPS}},
    {"dflash_config": {"mask_token_id": VOCAB - 1, "target_layer_ids": TAPS,
                       "num_target_layers": 8, "causal": True}},
])
def test_other_lfm_dspark_contracts_are_refused(change):
    with pytest.raises(ValueError):
        lfm_dspark_config(tiny_config(**change))


@pytest.mark.skipif(not (ARTIFACT / "model.safetensors").is_file() or not (TARGET / "config.json").is_file(),
                    reason="pinned LFM2.5 DSpark download or target not present")
def test_pinned_artifact_header_and_target_geometry_bind_without_tensor_load():
    record = inspect_drafter(ARTIFACT, TARGET)
    assert type(record["args"]) is InterleavedRopeDSparkConfig
    assert record["args"].num_target_layers == 40


# -- target taps -----------------------------------------------------------------


def test_target_taps_are_post_block_residuals_and_logits_unchanged():
    target = tiny_target()
    x = mx.array([[1, 2, 3, 4, 5]])
    logits, taps = target.forward_with_taps(x, target.make_cache(), TAPS)
    ordinary = target(x, cache=target.make_cache())
    np.testing.assert_allclose(np.asarray(logits), np.asarray(ordinary), atol=1e-5)
    assert taps.shape == (1, 5, 64)
    # Tap 1 is the residual stream after decoder layer 1 (two ShortConv layers).
    inner = target.model
    h = inner.embed_tokens(x)
    for layer in inner.layers[:2]:
        h = layer(h, mask=None, cache=None)
    np.testing.assert_allclose(np.asarray(taps[..., :32]), np.asarray(h), atol=1e-5)
    body = target.prefill_body(x, target.make_cache(), TAPS)
    np.testing.assert_allclose(np.asarray(body), np.asarray(taps), atol=1e-6)


@pytest.mark.parametrize("layers", [(), (6, 1), (1, 1), (8,), (-1,)])
def test_invalid_capture_layers_are_refused(layers):
    target = tiny_target()
    with pytest.raises(ValueError):
        target.forward_with_taps(mx.array([[1, 2]]), target.make_cache(), layers)


# -- verification exactness on the tiny ShortConv/MoE target ----------------------


@pytest.mark.parametrize("num_draft", [1, 3, 7])  # tiny block: 7 proposals
def test_greedy_chain_equals_ordinary_b1(num_draft):
    target, draft = tiny_pair()
    batch = generator(target, draft, num_draft=num_draft)
    uid = batch.insert([PROMPTS[0]], max_tokens=[14], sampling_configs=[{"sampling_temp": 0}])[0]
    got, final = drain(batch)
    assert got[uid] == ordinary_greedy(target, PROMPTS[0], 14)
    assert final[uid].speculative_receipt["kind"] == "external_dspark"
    assert batch.scheduler_stats["external_hybrid_transactions"] > 0
    assert batch.scheduler_stats["external_rounds"] > 0


def oracle_pair(prompts, count):
    """Draft whose Markov bias encodes the target's own greedy bigrams.

    Random heads never land a proposal on the tiny target, so this one makes
    accepts and rejects both happen (a bigram is wrong where context matters).
    """
    target = tiny_target()
    draft = dspark_draft_model(lfm_dspark_config(tiny_config(markov_rank=VOCAB)))
    successor = {}
    for prompt in prompts:
        sequence = list(prompt) + ordinary_greedy(target, prompt, count)
        successor.update(pairwise(sequence))
    w2 = np.zeros((VOCAB, VOCAB), np.float32)
    for previous, following in successor.items():
        w2[following, previous] = 50.0
    draft.markov_head.markov_w1.weight = mx.eye(VOCAB)
    draft.markov_head.markov_w2.weight = mx.array(w2)
    draft.bind(target)
    mx.eval(draft.parameters())
    return target, draft


def test_greedy_chain_with_accepts_and_rejects_equals_ordinary():
    target, draft = oracle_pair(PROMPTS, 16)
    batch = generator(target, draft, num_draft=5)
    uids = batch.insert(PROMPTS, max_tokens=[16] * 3, sampling_configs=[{"sampling_temp": 0}] * 3)
    got, _ = drain(batch)
    for uid, prompt in zip(uids, PROMPTS):
        assert got[uid] == ordinary_greedy(target, prompt, 16)
    stats = batch.scheduler_stats
    assert 0 < stats["accepted_proposals"] < stats["proposed_tokens"]


def test_greedy_chain_equals_ordinary_b3():
    target, draft = tiny_pair()
    batch = generator(target, draft, num_draft=5)
    uids = batch.insert(PROMPTS, max_tokens=[12] * 3, sampling_configs=[{"sampling_temp": 0}] * 3)
    got, _ = drain(batch)
    for uid, prompt in zip(uids, PROMPTS):
        assert got[uid] == ordinary_greedy(target, prompt, 12)


def test_memory_admission_reads_target_geometry():
    # Serving always passes a headroom callback; the KV estimate reads
    # ``head_dim`` from the target args (M5 smoke 2026-10-21 crashed here).
    target, draft = tiny_pair()
    assert target.args.head_dim == 16
    batch = generator(target, draft, num_draft=3, memory_headroom=lambda: 1 << 40)
    uid = batch.insert([PROMPTS[0]], max_tokens=[10], sampling_configs=[{"sampling_temp": 0}])[0]
    got, _ = drain(batch)
    assert got[uid] == ordinary_greedy(target, PROMPTS[0], 10)
