"""CPU-only component checks for the disabled Mellum one-layer MTP contract."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.mellum import Model, ModelArgs, MTPModule


def tiny_args() -> ModelArgs:
    return ModelArgs(
        hidden_size=8,
        num_hidden_layers=4,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        vocab_size=16,
        sliding_window=8,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        },
        num_mtp_layers=1,
        mtp_layer_types=["sliding_attention"],
        mtp_mlp_layer_types=["sparse"],
    )


def test_mellum_mtp_fusion_matches_independent_numpy_reference():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        args = tiny_args()
        module = MTPModule(args)
        embedding_weight = np.linspace(0.7, 1.3, args.hidden_size, dtype=np.float32)
        hidden_weight = np.linspace(1.2, 0.8, args.hidden_size, dtype=np.float32)
        projection = np.arange(
            args.hidden_size * args.hidden_size * 2, dtype=np.float32
        ).reshape(args.hidden_size, args.hidden_size * 2) / 100.0
        module.pre_fc_norm_embedding.weight = mx.array(embedding_weight)
        module.pre_fc_norm_hidden.weight = mx.array(hidden_weight)
        module.fc.weight = mx.array(projection)
        embeddings = np.arange(16, dtype=np.float32).reshape(1, 2, 8) / 10.0
        hidden = np.arange(16, 32, dtype=np.float32).reshape(1, 2, 8) / 10.0

        actual = np.asarray(module.fuse(mx.array(embeddings), mx.array(hidden)))

        def rms_norm(value, weight):
            return value * np.reciprocal(
                np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + args.rms_norm_eps)
            ) * weight

        expected = np.concatenate(
            [rms_norm(embeddings, embedding_weight), rms_norm(hidden, hidden_weight)],
            axis=-1,
        ) @ projection.T
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    finally:
        mx.set_default_device(previous)


def test_mellum_mtp_private_cache_rollback_matches_accepted_prefix():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(7)
        model = Model(tiny_args())
        baseline = model.make_mtp_cache()
        speculative = model.make_mtp_cache()
        first_hidden = mx.arange(8, dtype=mx.float32).reshape(1, 1, 8) / 10
        next_hidden = mx.arange(8, 16, dtype=mx.float32).reshape(1, 1, 8) / 10
        rejected_hidden = mx.arange(8, 24, dtype=mx.float32).reshape(1, 2, 8) / 10

        model.mtp_step(first_hidden, mx.array([[1]]), baseline)
        model.mtp_step(first_hidden, mx.array([[1]]), speculative)
        model.mtp_step(next_hidden, mx.array([[2]]), baseline)

        speculative[0].start_speculation()
        model.mtp_step(rejected_hidden, mx.array([[2, 15]]), speculative)
        assert speculative[0].trim(2) == 2
        speculative[0].stop_speculation()
        # Recompute the accepted token at the ordinary one-row shape.  Keeping
        # the first row of a width-two forward would import row-width numerical
        # drift into a state advertised as exact.
        model.mtp_step(next_hidden, mx.array([[2]]), speculative)
        assert baseline[0].offset == speculative[0].offset == 2
        base_k, base_v = baseline[0].keys_and_values()
        spec_k, spec_v = speculative[0].keys_and_values()
        mx.eval(base_k, base_v, spec_k, spec_v)
        np.testing.assert_array_equal(np.asarray(base_k), np.asarray(spec_k))
        np.testing.assert_array_equal(np.asarray(base_v), np.asarray(spec_v))

        follow_hidden = mx.arange(16, 24, dtype=mx.float32).reshape(1, 1, 8) / 10
        base_logits, _ = model.mtp_step(follow_hidden, mx.array([[3]]), baseline)
        spec_logits, _ = model.mtp_step(follow_hidden, mx.array([[3]]), speculative)
        mx.eval(base_logits, spec_logits)
        np.testing.assert_array_equal(np.asarray(base_logits), np.asarray(spec_logits))
    finally:
        mx.set_default_device(previous)


def test_mellum_mtp_model_args_fail_closed_on_nonpublished_geometry():
    args = tiny_args()
    args.num_mtp_layers = 2
    try:
        args.__post_init__()
    except ValueError as exc:
        assert "zero or one" in str(exc)
    else:
        raise AssertionError("two Mellum MTP layers must be refused")


def test_mellum_mtp_model_args_reject_float_count():
    args = tiny_args()
    args.num_mtp_layers = 1.0
    with pytest.raises(ValueError, match="zero or one"):
        args.__post_init__()


@pytest.mark.parametrize(
    ("layer_types", "mlp_types"),
    [(["sliding_attention"], []), ([], ["sparse"])],
)
def test_mellum_model_args_reject_ghost_mtp_geometry(layer_types, mlp_types):
    args = tiny_args()
    args.num_mtp_layers = 0
    args.mtp_layer_types = layer_types
    args.mtp_mlp_layer_types = mlp_types
    with pytest.raises(ValueError, match="requires one MTP layer"):
        args.__post_init__()


def test_mellum_mtp_weight_conversion_owns_source_layer_and_expert_packing():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        args = tiny_args()
        model = Model(args)
        weights = {
            "mtp.fc.weight": mx.ones((8, 16)),
            "mtp.pre_fc_norm_embedding.weight": mx.ones((8,)),
            "mtp.pre_fc_norm_hidden.weight": mx.ones((8,)),
            "mtp.norm.weight": mx.ones((8,)),
            "mtp.layers.28.self_attn.q_proj.weight": mx.ones((8, 8)),
            "mtp.layers.28.mlp.gate.weight": mx.ones((4, 8)),
        }
        for expert in range(args.num_experts):
            weights[f"mtp.layers.28.mlp.experts.{expert}.gate_proj.weight"] = mx.ones((4, 8))
            weights[f"mtp.layers.28.mlp.experts.{expert}.up_proj.weight"] = mx.ones((4, 8))
            weights[f"mtp.layers.28.mlp.experts.{expert}.down_proj.weight"] = mx.ones((8, 4))

        converted = model.sanitize(weights)

        assert not any(key.startswith("mtp.layers.28.") for key in converted)
        assert converted["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"].shape == (4, 4, 8)
        assert converted["mtp.layers.0.mlp.switch_mlp.up_proj.weight"].shape == (4, 4, 8)
        assert converted["mtp.layers.0.mlp.switch_mlp.down_proj.weight"].shape == (4, 8, 4)
    finally:
        mx.set_default_device(previous)
