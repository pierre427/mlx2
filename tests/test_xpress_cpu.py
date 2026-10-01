"""Independent CPU numerical oracles for the checkpoint's causal head."""

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.adapters.xpress import XPressConfig
from mlx2.runtime.drafters.xpress import (
    XPressDraftModel,
    XPressRefinerHead,
    fold_causal_mixer,
)


def tiny_config():
    return XPressConfig(
        hidden_size=4,
        intermediate_size=7,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        vocab_size=9,
        mask_token_id=8,
        num_target_layers=3,
        target_layer_ids=[0, 2],
        block_size=4,
        layer_types=["full_attention"],
        xpress_rank=3,
        xpress_mlp_hidden=5,
        xpress_num_passes=3,
    )


def assigned_head(seed=3):
    rng = np.random.default_rng(seed)
    head = XPressRefinerHead(9, 4, 4, rank=3, mlp_hidden=5)
    matrices = {}
    for name, module in [
        ("w1", head.w1),
        ("down_h", head.down_h),
        ("down_g", head.down_g),
        ("in_proj", head.in_proj),
        ("mlp_gate", head.mlp_gate),
        ("mlp_up", head.mlp_up),
        ("mlp_down", head.mlp_down),
        ("w2", head.w2),
    ]:
        value = rng.normal(0, 0.3, module.weight.shape).astype(np.float32)
        module.weight = mx.array(value)
        matrices[name] = value
    raw = rng.normal(0, 0.2, (3, 4, 4)).astype(np.float32)
    head.mix_L = fold_causal_mixer(mx.array(raw))
    matrices["mix"] = np.tril(raw) + np.eye(4, dtype=np.float32)[None]
    return head, matrices


def oracle_bias(weights, hidden, prev):
    global_h = np.broadcast_to(hidden.mean(1, keepdims=True), hidden.shape)
    hcache = np.concatenate(
        [hidden @ weights["down_h"].T, global_h @ weights["down_g"].T], -1
    )
    x = np.concatenate([hcache, weights["w1"][prev]], -1) @ weights["in_proj"].T
    mixed = np.empty_like(x)
    for row in range(x.shape[0]):
        for channel in range(x.shape[2]):
            for position in range(x.shape[1]):
                mixed[row, position, channel] = sum(
                    weights["mix"][channel, position, j] * x[row, j, channel]
                    for j in range(position + 1)
                )
    gate = mixed @ weights["mlp_gate"].T
    latent = (
        mixed
        + (gate / (1 + np.exp(-gate)) * (mixed @ weights["mlp_up"].T))
        @ weights["mlp_down"].T
    )
    return latent @ weights["w2"].T


def test_fold_masks_future_and_adds_identity_once():
    raw = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
    np.testing.assert_array_equal(
        np.asarray(fold_causal_mixer(mx.array(raw))), np.tril(raw) + np.eye(4)[None]
    )
    with pytest.raises(ValueError):
        fold_causal_mixer(mx.zeros((3, 4, 5)))


def test_head_matches_scalar_oracle_and_predecessor_causality():
    head, weights = assigned_head()
    hidden = np.random.default_rng(5).normal(size=(2, 4, 4)).astype(np.float32)
    prev = np.array([[1, 2, 3, 4], [3, 2, 5, 6]], dtype=np.int32)
    result = np.asarray(
        head.refine_bias(mx.array(prev), head.hidden_cache(mx.array(hidden)))
    )
    np.testing.assert_allclose(
        result, oracle_bias(weights, hidden, prev), rtol=2e-5, atol=2e-6
    )
    changed = prev.copy()
    changed[:, 3] = 8
    later = np.asarray(
        head.refine_bias(mx.array(changed), head.hidden_cache(mx.array(hidden)))
    )
    np.testing.assert_array_equal(result[:, :3], later[:, :3])
    changed = prev.copy()
    changed[:, 0] = 8
    predecessor = np.asarray(
        head.refine_bias(mx.array(changed), head.hidden_cache(mx.array(hidden)))
    )
    assert not np.allclose(result[:, 1:], predecessor[:, 1:])


def test_jacobi_simultaneous_oracle_preserves_anchor():
    head, weights = assigned_head()
    rng = np.random.default_rng(8)
    hidden = rng.normal(size=(2, 4, 4)).astype(np.float32)
    base = rng.normal(size=(2, 4, 9)).astype(np.float32)
    anchors = np.array([4, 6], dtype=np.int32)
    before = np.array([7, 1], dtype=np.int32)
    block = np.concatenate([anchors[:, None], base[:, 1:].argmax(-1)], 1)
    for _ in range(3):
        prev = np.concatenate([before[:, None], block[:, :-1]], 1)
        refined = base + oracle_bias(weights, hidden, prev)
        block = np.concatenate([anchors[:, None], refined[:, 1:].argmax(-1)], 1)
    actual, logits = head.jacobi_refine_greedy(
        mx.array(base), mx.array(hidden), mx.array(anchors), mx.array(before), 3
    )
    np.testing.assert_array_equal(np.asarray(actual), block[:, 1:])
    np.testing.assert_allclose(np.asarray(logits), refined[:, 1:], rtol=2e-5, atol=2e-6)


def fake_drafter():
    model = XPressDraftModel(tiny_config())
    head, _ = assigned_head()
    model.xpress_head = head
    model.seen = []

    def backbone(inputs, hidden, cache):
        model.seen.append(tuple(inputs.shape))
        return mx.array(np.arange(16, dtype=np.float32).reshape(1, 4, 4) / 10)

    model._hidden = backbone
    model._logits = lambda h: mx.concatenate([h, h, h[:, :, :1]], -1)
    return model


def test_full_block_truncation_and_exact_point_mass_for_sampled_target():
    model = fake_drafter()
    hidden = mx.zeros((1, 0, 8))
    cache = model.make_cache()
    short, q = model.draft_distributions(
        [2], hidden, cache, 1, [object()], [1.2], processor_histories=[[1]]
    )
    long, _ = model.draft_distributions(
        [2], hidden, cache, 3, [object()], [0.0], processor_histories=[[1]]
    )
    assert short[0] == long[0][:1]
    assert model.seen == [(1, 4), (1, 4)]
    assert q[0][0].sum() == 1 and q[0][0][short[0][0]] == 1
    assert np.count_nonzero(q[0][0]) == 1
    assert model.receipt_settings["proposal_distribution"] == "deterministic_point_mass"


def test_missing_predecessor_and_nonfinite_fail_closed():
    model = fake_drafter()
    hidden = mx.zeros((1, 0, 8))
    cache = model.make_cache()
    with pytest.raises(ValueError, match="predecessor histories"):
        model.draft_distributions([2], hidden, cache, 1, [object()], [0.0])
    model._logits = lambda h: mx.full((1, 4, 9), float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        model.draft_distributions(
            [2], hidden, cache, 1, [object()], [0.0], processor_histories=[[1]]
        )


def test_empty_committed_history_uses_explicit_anchor_convention():
    model = fake_drafter()
    hidden = mx.zeros((1, 0, 8))
    cache = model.make_cache()
    implicit, _ = model.draft_distributions(
        [2], hidden, cache, 2, [object()], [0.0], processor_histories=[[]]
    )
    explicit, _ = model.draft_distributions(
        [2], hidden, cache, 2, [object()], [0.0], predecessor_ids=[2]
    )
    assert implicit == explicit


def test_processor_uses_committed_history_anchor_and_chosen_prefix():
    model = fake_drafter()
    prefixes = []

    def processor(tokens, logits):
        prefixes.append(np.asarray(tokens).tolist())
        return mx.array([[0.0 if j == 5 else float("-inf") for j in range(9)]])

    processor.probe = processor
    proposals, laws = model.draft_distributions(
        [2],
        mx.zeros((1, 0, 8)),
        model.make_cache(),
        2,
        [object()],
        [1.0],
        logits_processors=[[processor]],
        processor_histories=[[7, 1]],
    )
    assert proposals == [[5, 5]]
    assert prefixes == [[7, 1, 2], [7, 1, 2, 5]]
    assert all(q[5] == 1 for q in laws[0])


def test_real_backbone_committed_kv_and_batched_rows_remain_finite():
    from types import SimpleNamespace

    from mlx import nn

    model = XPressDraftModel(tiny_config())
    embedding = nn.Embedding(9, 4)
    model.bind(SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding)))
    rows = [model.make_cache(), model.make_cache()]
    model.append_context(mx.ones((1, 2, 8)), rows[0])
    model.append_context(mx.ones((1, 1, 8)), rows[1])
    assert [row[0].offset for row in rows] == [2, 1]
    batch = model.batch_caches(rows)
    tokens, laws = model.draft_distributions(
        [2, 3],
        mx.ones((2, 1, 8)),
        batch,
        2,
        [object(), object()],
        [0.0, 1.0],
        processor_histories=[[1], [4]],
    )
    assert [row[0].offset for row in rows] == [3, 2]
    assert all(len(row) == 2 for row in tokens)
    assert all(np.isfinite(q).all() and q.sum() == 1 for row in laws for q in row)
