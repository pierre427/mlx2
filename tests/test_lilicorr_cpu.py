"""CPU parity against independently evaluated global lattice equations."""

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

mx.set_default_device(mx.cpu)

from mlx.utils import tree_flatten

from mlx2.adapters.lilicorr import LiLiCorrConfig
from mlx2.runtime.drafters.lilicorr import LiLiCorrDraftModel, LiLiCorrHead


def config(**changes):
    values = {
        "hidden_size": 4,
        "intermediate_size": 7,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 9,
        "mask_token_id": 8,
        "num_target_layers": 3,
        "target_layer_ids": [0, 2],
        "block_size": 4,
        "layer_types": ["full_attention"],
        "lilicorr_hidden_size": 6,
        "lilicorr_candidate_topk": 2,
        "lilicorr_num_layers": 2,
        "lilicorr_num_heads": 2,
        "lilicorr_mlp_ratio": 1.5,
        "lilicorr_factor_dim": 3,
        "lilicorr_logit_scale": 3.0,
    }
    values.update(changes)
    return LiLiCorrConfig(**values)


def initialized_head(args=None):
    args = args or config()
    head = LiLiCorrHead(args)
    rng = np.random.default_rng(5)
    head.load_weights(
        [
            (name, mx.array(rng.normal(0, 0.3, value.shape).astype(np.float32)))
            for name, value in tree_flatten(head.parameters())
        ],
        strict=True,
    )
    return head


def rms(x, weight, eps):
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * weight


def silu(x):
    return x / (1 + np.exp(-x))


def normalize(x, eps):
    return x / np.maximum(np.sqrt(np.sum(x * x, -1, keepdims=True)), eps)


def oracle(head, args, embeddings, log_probs, hidden, anchor, valid):
    w = {name: np.asarray(value) for name, value in tree_flatten(head.parameters())}

    def linear(name, x):
        return x @ w[name + ".weight"].T + w[name + ".bias"]

    batch, slots, k = log_probs.shape
    width = args.lilicorr_hidden_size or args.hidden_size
    ranks = np.broadcast_to(np.arange(k) / (k - 1 if k > 1 else 1), log_probs.shape)
    top1 = np.broadcast_to(np.arange(k) == 0, log_probs.shape)
    features = np.stack(
        [
            log_probs,
            np.exp(log_probs),
            log_probs - log_probs.max(-1, keepdims=True),
            ranks,
            top1,
        ],
        -1,
    )
    features = (features - features.mean(-1, keepdims=True)) / np.sqrt(
        features.var(-1, keepdims=True) + 1e-5
    )
    features = features * w["feature_norm.weight"] + w["feature_norm.bias"]
    token = (
        linear("token_proj", embeddings) if width != args.hidden_size else embeddings
    )
    x = token + linear("pass_hidden_proj", hidden)[:, :, None]
    x += linear("feature_mlp.down_proj", silu(linear("feature_mlp.up_proj", features)))
    x += w["slot_embedding"][:, 0, :slots] + w["rank_embedding"][:, 0]
    x = x.reshape(batch, slots * k, width)
    slotids = np.repeat(np.arange(slots), k)
    rel = slotids[:, None] - slotids[None, :]
    bias = (
        w["relative_slot_bias"][:, rel + args.block_size - 1]
        + w["same_slot_bias"][:, None, None] * (rel == 0)[None]
    )
    for i in range(args.lilicorr_num_layers):
        p = f"layers.{i}."
        normed = rms(x, w[p + "attn_norm.weight"], args.rms_norm_eps)
        qkv = normed @ w[p + "attn.in_proj_weight"].T + w[p + "attn.in_proj_bias"]
        q, kmat, v = [
            part.reshape(batch, slots * k, args.lilicorr_num_heads, -1).transpose(
                0, 2, 1, 3
            )
            for part in np.split(qkv, 3, -1)
        ]
        scores = (
            q @ kmat.swapaxes(-1, -2) / np.sqrt(width / args.lilicorr_num_heads)
            + bias[None]
        )
        weights = np.exp(scores - scores.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        attn = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, slots * k, width)
        x += linear(p + "attn.out_proj", attn)
        normed = rms(x, w[p + "mlp_norm.weight"], args.rms_norm_eps)
        x += linear(p + "mlp.down_proj", silu(linear(p + "mlp.up_proj", normed)))
    x = rms(x, w["output_norm.weight"], args.rms_norm_eps).reshape(
        batch, slots, k, width
    )
    anchor = linear("context_proj", anchor) * valid[:, None]
    anchor = rms(anchor, w["anchor_norm.weight"], args.rms_norm_eps)
    a = np.broadcast_to(anchor[:, None, None], x.shape)
    factor = silu(linear("factor_input_proj", np.concatenate([x, a, x * a], -1)))
    outs = normalize(linear("out_head", factor), args.lilicorr_vector_eps)
    ins = normalize(linear("in_head", factor), args.lilicorr_vector_eps)
    aout = normalize(linear("anchor_out_head", anchor), args.lilicorr_vector_eps)
    first = np.sum(aout[:, None] * ins[:, 0], -1)
    pairs = outs[:, :-1] @ ins[:, 1:].swapaxes(-1, -2)
    return first, pairs


@pytest.mark.parametrize("width,slots", [(4, 1), (6, 2), (6, 3)])
def test_learned_lattice_matches_numpy_reference(width, slots):
    args = config(lilicorr_hidden_size=width)
    head = initialized_head(args)
    rng = np.random.default_rng(8)
    embeddings = rng.normal(size=(2, slots, 2, 4)).astype(np.float32)
    probs = np.array([-0.4, -2.3], np.float32)[None, None] + np.zeros(
        (2, slots, 2), np.float32
    )
    hidden = rng.normal(size=(2, slots, 4)).astype(np.float32)
    anchor = rng.normal(size=(2, 4)).astype(np.float32)
    valid = np.array([True, False])
    expected = oracle(head, args, embeddings, probs, hidden, anchor, valid)
    actual = head.score(
        *[mx.array(v) for v in (embeddings, probs, hidden, anchor, valid)]
    )
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(result), reference, rtol=3e-5, atol=3e-6)
    table = np.asarray(
        head(*[mx.array(v) for v in (embeddings, probs, hidden, anchor, valid)])
    )
    np.testing.assert_allclose(table[:, 0, 0], expected[0] * 3, rtol=3e-5, atol=3e-6)
    np.testing.assert_array_equal(table[:, 0, 0], table[:, 0, 1])


def test_relative_bias_includes_distinct_same_slot_candidates_and_trained_offset():
    head = initialized_head()
    full = np.asarray(head.attention_bias(3))
    short = np.asarray(head.attention_bias(1))
    np.testing.assert_array_equal(short, full[:, :2, :2])
    assert np.array_equal(short[:, 0, 1], short[:, 0, 0])
    with pytest.raises(ValueError):
        head.attention_bias(4)


@pytest.mark.parametrize("conv", [False, True])
def test_real_tiny_backbone_fullblock_pointmass_confidence(conv):
    from types import SimpleNamespace

    model = LiLiCorrDraftModel(
        config(conv_kernel_size=2 if conv else 0, conv_group_size=2 if conv else 0)
    )
    model.bind(SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(9, 4))))
    seen = []
    original = model._hidden

    def traced(tokens, hidden, cache):
        seen.append(tuple(tokens.shape))
        return original(tokens, hidden, cache)

    model._hidden = traced
    hidden = mx.ones((1, 2, 8))
    tokens, laws = model.draft_distributions(
        [2], hidden, model.make_cache(), 1, [object()], [1.0], processor_histories=[[1]]
    )
    longer, _ = model.draft_distributions(
        [2], hidden, model.make_cache(), 3, [object()], [0.0], processor_histories=[[1]]
    )
    assert tokens[0] == longer[0][:1]
    assert seen == [(1, 4), (1, 4)]
    assert laws[0][0][tokens[0][0]] == 1 and np.count_nonzero(laws[0][0]) == 1
    assert np.isfinite(model.adaptive_confidence_features).all()
    assert len(model.adaptive_confidence_features[0]) == 3


def test_correlation_walk_uses_selected_predecessor_without_unary_addition():
    from types import SimpleNamespace

    model = LiLiCorrDraftModel(config())
    embedding = nn.Embedding(9, 4)
    model.bind(SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding)))
    model._hidden = lambda inputs, h, c: mx.zeros((1, 4, 4))
    model._logits = lambda h: mx.array(
        [[[10.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3]
    )
    # First selects column1; next must use row1, then column0's row at slot2.
    table = mx.array(
        [[[[0.0, 2.0], [0.0, 2.0]], [[5.0, 0.0], [0.0, 4.0]], [[0.0, 3.0], [5.0, 0.0]]]]
    )
    model.lilicorr = lambda *a: table
    tokens, laws = model.draft_distributions(
        [2], mx.zeros((1, 0, 8)), model.make_cache(), 3, [object()], [1.0]
    )
    assert tokens == [[1, 1, 0]]
    assert [q[t] for q, t in zip(laws[0], tokens[0])] == [1.0, 1.0, 1.0]
