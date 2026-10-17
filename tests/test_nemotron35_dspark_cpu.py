import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.nemotron35_dspark import (
    DSparkConfig,
    expected_weight_shapes,
    post_layer_ids,
)
from mlx2.runtime.drafters.nemotron35_dspark import DSparkDraftModel, proposal_mask


def config():
    return DSparkConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                        vocab_size=32, block_size=5, mask_token_id=31, target_layer_ids=[1, 3],
                        num_target_layers=4, layer_types=["sliding_attention"] * 2,
                        sliding_window=3, markov_rank=2)


def test_causal_proposal_mask_preserves_whole_anchor_prefix():
    expected = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], bool)
    np.testing.assert_array_equal(np.asarray(proposal_mask(2, 3)), expected)


def test_shape_contract_has_only_trained_heads():
    shapes = expected_weight_shapes(config())
    assert len(shapes) == 29
    assert shapes["confidence_head.proj.weight"] == [1, 10]
    assert shapes["markov_head.markov_w2.weight"] == [32, 2]
    assert not any(name.startswith("xpress") for name in shapes)


def test_markov_chain_and_confidence_use_actual_predecessor(monkeypatch):
    cfg = config()
    model = DSparkDraftModel(cfg)
    w1, w2 = np.zeros((32, 2), np.float32), np.zeros((32, 2), np.float32)
    w1[0], w1[1], w1[2] = [1, 0], [0, 1], [1, 0]
    w2[1], w2[2] = [2, 0], [0, 2]
    model.markov_head.markov_w1.weight = mx.array(w1)
    model.markov_head.markov_w2.weight = mx.array(w2)
    model.confidence_head.proj.weight = mx.ones((1, 10))
    model.confidence_head.proj.bias = mx.array([.5])
    seen = []

    def backbone(inputs, hidden, cache):
        seen.append(inputs.tolist())
        return mx.zeros((1, 4, 8))

    monkeypatch.setattr(model, "_hidden", backbone)
    monkeypatch.setattr(model, "_logits", lambda hidden: mx.zeros((1, 4, 32)))
    tokens, laws = model.draft_distributions([0], mx.zeros((1, 2, 16)), model.make_cache(), 3, [None], [0.])
    assert seen == [[[0, 31, 31, 31]]]
    assert tokens == [[1, 2, 1]]
    assert [int(np.argmax(law)) for law in laws[0]] == tokens[0]
    assert all(law.sum() == 1 and law.max() == 1 for law in laws[0])
    assert model.adaptive_confidence_features == [[1.5, 1.5, 1.5]]
    assert model.receipt_settings["proposal_attention"] == "causal"


def test_shared_prefix_ordered_alternatives_keep_primary_and_diverge_late(monkeypatch):
    model = DSparkDraftModel(config())
    monkeypatch.setattr(
        model, "_hidden", lambda inputs, hidden, cache: mx.zeros((1, 4, 8))
    )
    logits = np.zeros((1, 4, 32), np.float32)
    logits[0, :, 1:5] = np.array([8.0, 7.0, 6.0, 5.0])
    monkeypatch.setattr(model, "_logits", lambda hidden: mx.array(logits))
    model.markov_head.markov_w1.weight = mx.zeros((32, 2))
    model.markov_head.markov_w2.weight = mx.zeros((32, 2))
    tokens, _ = model.draft_distributions(
        [0], mx.zeros((1, 2, 16)), model.make_cache(), 4, [None], [0.0]
    )
    paths = model.ordered_alternative_proposals(
        tokens,
        branch_count=4,
        strategy="shared_prefix",
        shared_prefix_tokens=2,
    )[0]
    assert paths[0] == tokens[0]
    assert all(path[:2] == tokens[0][:2] for path in paths)
    assert [path[2] for path in paths] == [1, 2, 3, 4]
    paired = model.ordered_alternative_proposals(
        tokens,
        branch_count=4,
        strategy="paired_continuations",
        shared_prefix_tokens=2,
    )[0]
    assert paired[0] == tokens[0]
    assert paired[0][2] == paired[2][2]
    assert paired[1][2] == paired[3][2]
    assert paired[0][3] != paired[2][3]
    assert paired[1][3] != paired[3][3]


def test_attention_future_proposals_cannot_change_first_slot():
    model = DSparkDraftModel(config())
    attention = model.layers[0].self_attn
    rope = model.rope
    mx.random.seed(13)
    inputs = mx.random.normal((1, 4, 8))
    future_changed = mx.concatenate([inputs[:, :1], mx.random.normal((1, 3, 8)) * 20], axis=1)
    context = mx.random.normal((1, 2, 8))
    first = attention(inputs, context, rope, model.make_cache()[0])
    second = attention(future_changed, context, rope, model.make_cache()[0])
    np.testing.assert_array_equal(np.asarray(first[:, 0]), np.asarray(second[:, 0]))
    assert not np.allclose(np.asarray(first[:, -1]), np.asarray(second[:, -1]))


def test_invalid_variant_is_refused():
    with pytest.raises(ValueError, match="Unsupported DSpark"):
        cfg = config()
        cfg.sample_from_anchor = False
        cfg.__post_init__()


def test_speculators_hidden_state_ids_match_independent_hybrid_layer_walk():
    from test_nemotron_external_taps_cpu import tiny_model

    from mlx2.runtime.models.base import create_attention_mask, create_ssm_mask

    model = tiny_model()
    tokens, caches = mx.array([[1, 2, 3]]), model.make_cache()
    hidden = model.backbone.embeddings(tokens)
    hf_states = [hidden]
    masks = {"*": create_attention_mask(hidden, caches[1]), "M": create_ssm_mask(hidden, caches[0])}
    cache_index = 0
    for layer in model.layers:
        entry = caches[cache_index] if layer.block_type in ("M", "*") else None
        cache_index += entry is not None
        hidden = layer(hidden, mask=masks.get(layer.block_type), cache=entry)
        hf_states.append(hidden)
    source_ids = [1, 3, 5]
    _, actual = model.forward_with_taps(tokens, model.make_cache(), post_layer_ids(source_ids, 5))
    expected = mx.concatenate([hf_states[index] for index in source_ids], axis=-1)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert post_layer_ids([17, 37, 59, 79, 87], 88) == [16, 36, 58, 78, 86]
    for invalid in ([0, 2], [1, 89], [True, 2], [2, 1], [1, 1]):
        with pytest.raises(ValueError, match="auxiliary"):
            post_layer_ids(invalid, 88)
