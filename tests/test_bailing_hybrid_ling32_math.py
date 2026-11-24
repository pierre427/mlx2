import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.runtime.models import bailing_hybrid as bh

LIMITS = [0] * 35 + [4] * 7
SHARED = [0] * 34 + [5] * 6 + [7] * 2


def _args(**extra):
    base = dict(
        model_type="bailing_hybrid", vocab_size=64, hidden_size=64, intermediate_size=64,
        moe_intermediate_size=64, num_hidden_layers=42, num_attention_heads=2,
        num_key_value_heads=2, num_experts=8, num_experts_per_tok=2, n_group=2,
        topk_group=1, first_k_dense_replace=2, layer_group_size=6,
        max_position_embeddings=128, rms_norm_eps=1e-6, rope_theta=1e4,
        routed_scaling_factor=2.5, head_dim=32, kv_lora_rank=32, qk_rope_head_dim=16,
        qk_nope_head_dim=32, v_head_dim=32,
    )
    base.update(extra)
    return bh.ModelArgs(**base)


def _vendor(gate, up, limit):
    """transformers BailingMoeV3MLP.forward with ACT2FN['silu']."""
    silu = gate / (1 + np.exp(-gate))
    return np.minimum(silu, limit) * np.clip(up, -limit, limit)


def test_clamped_swiglu_matches_vendor_post_activation_order():
    rng = np.random.default_rng(0)
    gate = (rng.standard_normal((4, 64)) * 6).astype(np.float32)
    up = (rng.standard_normal((4, 64)) * 6).astype(np.float32)
    got = np.array(bh.clamped_swiglu(mx.array(gate), mx.array(up), 4.0))
    np.testing.assert_allclose(got, _vendor(gate, up, 4.0), rtol=1e-6, atol=1e-6)
    # The gpt-oss/GLM order (clamp the logit first) is a different law.
    pre = np.minimum(gate, 4.0) / (1 + np.exp(-np.minimum(gate, 4.0))) * np.clip(up, -4, 4)
    assert np.max(np.abs(pre - got)) > 1e-3


def test_limits_follow_vendor_lookup():
    assert bh.swiglu_limit(LIMITS, 34) is None
    assert bh.swiglu_limit(LIMITS, 35) == 4.0
    assert bh.swiglu_limit(LIMITS, 42) is None  # MTP index is out of range
    assert bh.swiglu_limit(None, 3) is None
    args = _args(expert_swiglu_limit_list=LIMITS, share_expert_swiglu_limit_list=SHARED,
                 vendor_v32_laws=True)
    late = bh.SparseMoeBlock(args, 40)
    assert isinstance(late.switch_mlp.activation, bh.ClampedSwiGLU)
    assert late.switch_mlp.activation.swiglu_limit == 4.0
    assert late.shared_experts.swiglu_limit == 7.0
    early = bh.SparseMoeBlock(args, 10)
    assert not isinstance(early.switch_mlp.activation, bh.ClampedSwiGLU)
    assert early.shared_experts.swiglu_limit is None
    assert bh.SparseMoeBlock(args, 34).shared_experts.swiglu_limit == 5.0


def test_clamped_mlp_forward():
    args = _args()
    mlp = bh.MLP(args, swiglu_limit=5.0)
    x = mx.random.normal((3, 64)) * 4
    expected = mlp.down_proj(bh.clamped_swiglu(mlp.gate_proj(x), mlp.up_proj(x), 5.0))
    assert mx.array_equal(mlp(x), expected)


def test_fp32_router_and_quantization_guard():
    args = _args(router_dtype="fp32", vendor_v32_laws=True)
    gate = bh.Gate(args)
    gate.gate_proj.weight = gate.gate_proj.weight.astype(mx.float32)
    x = mx.random.normal((5, 64)).astype(mx.bfloat16)
    indices, weights = gate(x)
    assert weights.dtype == mx.float32 and indices.shape == (5, 2)  # vendor fp32 topk_weight
    logits = np.array(x.astype(mx.float32)) @ np.array(gate.gate_proj.weight).T
    ref_idx, _ = bh.group_expert_select(mx.array(logits), gate.expert_bias, 2, 2, 1, 2.5,
                                        True, "sigmoid")
    assert mx.array_equal(mx.sort(indices, axis=-1), mx.sort(ref_idx, axis=-1))
    model_predicate = bh.Model.quant_predicate.fget(type("M", (), {"args": args})())
    assert model_predicate("model.layers.3.mlp.gate.gate_proj", None) is False
    nn.quantize(gate, group_size=64, bits=8)
    with pytest.raises(ValueError, match="must not be quantized"):
        gate(x)


def _tiny(mtp=True):
    args = _args(num_hidden_layers=6, num_nextn_predict_layers=1, router_dtype="fp32",
                 vendor_v32_laws=True,
                 no_kda_lora=True, kda_lower_bound=-5.0, kda_safe_gate=True,
                 gated_attention_proj_granularity_type="head_wise")
    mx.random.seed(0)
    model = bh.Model(args, mtp=mtp)
    mx.eval(model.parameters())
    return model


def test_mtp_head_math_and_seed_feature():
    model = _tiny()
    tokens = mx.array([[3, 7, 11, 2, 5]])
    hidden = model.model(tokens)  # post model.norm
    nxt = mx.array([[7, 11, 2, 5, 9]])
    logits, post = model.mtp_step(hidden, nxt, model.make_mtp_cache())
    layer = model.mtp.layers[0]
    e = layer.enorm(model.model.word_embeddings(nxt))
    x = layer.eh_proj(mx.concatenate([e, layer.hnorm(hidden)], axis=-1))
    mask = bh.create_attention_mask(x, None, return_array=True)
    x = x + layer.attention(layer.input_layernorm(x), mask, None)
    x = x + layer.mlp(layer.post_attention_layernorm(x))
    expected = layer.final_layernorm(x)
    assert mx.allclose(post, expected, atol=1e-5)
    assert mx.allclose(logits, model.lm_head(expected), atol=1e-4)
    # Incremental MTP decode over its own cache matches the batched call.
    cache = model.make_mtp_cache()
    rows = [model.mtp_step(hidden[:, i:i + 1], nxt[:, i:i + 1], cache)[1] for i in range(5)]
    assert mx.allclose(mx.concatenate(rows, axis=1), post, atol=1e-4)
    with pytest.raises(ValueError):
        model.select_external_feature_source("bad")
    model.select_external_feature_source("mtp_seed")
    assert model.external_feature_width == 64
    out_logits, seed = model.forward_with_taps(tokens, None, (5,))
    assert mx.allclose(seed, hidden, atol=1e-6)
    with pytest.raises(ValueError, match="post-norm"):
        model.forward_with_taps(tokens, None, (2,))


def test_sanitize_keeps_or_drops_mtp_and_splits_kv_b():
    model = _tiny()
    heads, nope, v, rank = 2, 32, 32, 32
    weights = {"mtp.layers.0.attention.kv_b_proj.weight": mx.zeros((heads * (nope + v), rank)),
               "mtp.layers.0.mlp.gate.weight": mx.zeros((8, 64)),
               "mtp.layers.0.eh_proj.weight": mx.zeros((64, 128))}
    out = model.sanitize(dict(weights))
    assert out["mtp.layers.0.attention.embed_q.weight"].shape == (heads, rank, nope)
    assert out["mtp.layers.0.attention.unembed_out.weight"].shape == (heads, v, rank)
    assert "mtp.layers.0.mlp.gate.gate_proj.weight" in out
    bare = _tiny(mtp=False)
    assert not any(k.startswith("mtp.") for k in bare.sanitize(dict(weights)))
    with pytest.raises(ValueError, match="no loaded MTP"):
        bare.make_mtp_cache()


def test_flash_config_with_v32_keys_keeps_its_served_laws():
    """Ling 3 Flash's oQ4 config also carries router_dtype and the limit lists,
    with a quantized router; without the adapter opt-in nothing changes."""
    args = _args(router_dtype="fp32", expert_swiglu_limit_list=LIMITS,
                 share_expert_swiglu_limit_list=SHARED)
    block = bh.SparseMoeBlock(args, 40)
    assert not isinstance(block.switch_mlp.activation, bh.ClampedSwiGLU)
    assert block.shared_experts.swiglu_limit is None
    assert block.gate.router_fp32 is False and block.gate.group_mask_neg_inf is False
    nn.quantize(block.gate, group_size=64, bits=8)  # Flash ships a q8 router
    x = mx.random.normal((3, 64)).astype(mx.bfloat16)
    indices, weights = block.gate(x)
    assert weights.dtype == mx.bfloat16
    predicate = bh.Model.quant_predicate.fget(type("M", (), {"args": args})())
    assert predicate("model.layers.3.mlp.gate.gate_proj", None) == {"group_size": 64, "bits": 8}


def test_v32_group_mask_uses_neg_inf():
    # Every biased score is negative: group 0 (-0.5, -0.5) is kept over group 1
    # (-0.9, -0.9). A 0.0 fill makes the dropped experts outrank the kept ones.
    gates = mx.zeros((1, 4))
    bias = mx.array([-1.0, -1.0, -1.4, -1.4])
    idx_v32, _ = bh.group_expert_select(gates, bias, 2, 2, 1, 1.0, True, "sigmoid", True)
    assert sorted(idx_v32[0].tolist()) == [0, 1]
    idx_old, _ = bh.group_expert_select(gates, bias, 2, 2, 1, 1.0, True, "sigmoid", False)
    assert sorted(idx_old[0].tolist()) == [2, 3]  # the pre-existing 0.0-fill law
