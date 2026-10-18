"""CPU checks for the Bolt GGML decoders, conversion plan and LiLiCorr wiring."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from test_nemotron35_bolt_cpu import target_config

from mlx2.conversion import ggml
from mlx2.conversion import nemotron35_bolt_plan as plan


def _fp16(value):
    return np.array([value], "<f2").view(np.uint8)


def test_iq2_xxs_tables_have_reference_structure():
    assert ggml.IQ2XXS_GRID.shape == (256, 8)
    assert set(np.unique(ggml.IQ2XXS_GRID)) == {8.0, 25.0, 43.0}
    # Each ksigns byte carries 7 explicit sign bits plus even parity.
    popcount = np.unpackbits(ggml.KSIGNS[:, None], axis=1).sum(axis=1)
    assert np.all(popcount % 2 == 0)
    assert np.array_equal(ggml.KSIGNS & 0x7F, np.arange(128, dtype=np.uint8))


def test_iq2_xxs_block_decodes_by_the_ggml_formula():
    rng = np.random.default_rng(0)
    block = np.zeros(66, np.uint8)
    block[:2] = _fp16(0.0123)
    words = np.zeros(16, "<u4")
    for sub in range(8):
        grid = rng.integers(0, 256, 4)
        signs = rng.integers(0, 128, 4)
        scale = int(rng.integers(0, 16))
        words[2 * sub] = int(np.sum(grid.astype(np.uint64) << (8 * np.arange(4, dtype=np.uint64))))
        words[2 * sub + 1] = int(
            np.sum(signs.astype(np.uint64) << (7 * np.arange(4, dtype=np.uint64)))
        ) | (scale << 28)
    block[2:] = words.view(np.uint8)
    decoded = ggml.dequantize_iq2_xxs(block[None, None, :])
    d = np.float32(np.array([0.0123], "<f2")[0])
    expected = []
    for sub in range(8):
        aux0, aux1 = int(words[2 * sub]), int(words[2 * sub + 1])
        db = d * (np.float32(0.5) + np.float32(aux1 >> 28)) * np.float32(0.25)
        for lane in range(4):
            grid = ggml.IQ2XXS_GRID[(aux0 >> (8 * lane)) & 0xFF]
            signs = ggml.KSIGNS[(aux1 >> (7 * lane)) & 0x7F]
            for j in range(8):
                expected.append(db * grid[j] * (np.float32(-1) if signs >> j & 1 else np.float32(1)))
    assert decoded.shape == (1, 256)
    assert np.array_equal(decoded[0].view(np.uint32), np.array(expected, np.float32).view(np.uint32))


def test_q8_0_decodes_scale_times_int8():
    block = np.zeros(34, np.uint8)
    block[:2] = _fp16(0.5)
    block[2:] = np.arange(-16, 16, dtype=np.int8).view(np.uint8)
    decoded = ggml.dequantize_q8_0(np.stack([block, block])[None])
    assert decoded.shape == (1, 64)
    assert np.array_equal(decoded[0, :32], np.arange(-16, 16, dtype=np.float32) * 0.5)


def test_ggml_decoders_reject_partial_blocks():
    with pytest.raises(ValueError, match="whole blocks"):
        ggml.dequantize_iq2_xxs(np.zeros((1, 65), np.uint8))
    with pytest.raises(ValueError, match="whole blocks"):
        ggml.dequantize_q8_0(np.zeros((1, 33), np.uint8))


def bolt_config():
    config = target_config()
    text = config["llm_config"]
    text.update(time_step_limit=[0.0, {"__float__": "Infinity"}], moe_latent_size=1024,
                moe_shared_expert_intermediate_size=5376, n_group=1, topk_group=1)
    return config


def test_plan_applies_super_law_with_group32_experts():
    config = bolt_config()
    result = plan.make_plan(config)
    modules = result["modules"]
    blocks = config["llm_config"]["block_configs"]
    moe = next(i for i, block in enumerate(blocks) if block["block_type"] == "moe")
    prefix = f"language_model.backbone.layers.{moe}.mixer."
    assert modules[prefix + "switch_mlp.fc1"] == {"bits": 4, "group_size": 32, "mode": "affine"}
    assert modules[prefix + "shared_experts.up_proj"]["bits"] == 8
    assert modules[prefix + "shared_experts.up_proj"]["group_size"] == 64
    assert prefix + "fc1_latent_proj" not in modules
    assert prefix + "gate" not in modules
    assert "language_model.lm_head" in modules
    assert "language_model.backbone.embeddings" not in modules
    assert not any(name.startswith(("vision_model.", "mlp1.")) for name in modules)
    fused = result["schema"][prefix + "switch_mlp.fc1.weight"]
    width = blocks[moe]["moe_intermediate_size"]
    assert fused == {"shape": [blocks[moe]["n_routed_experts"], width, 1024 * 4 // 32], "dtype": "U32"}
    assert result["schema"][prefix + "switch_mlp.fc1.scales"]["shape"][-1] == 1024 // 32
    assert not any(".mtp." in name for name in result["schema"])


def test_language_config_keeps_per_layer_moe_declarations():
    text = plan.language_config(bolt_config())
    assert text["num_hidden_layers"] == 88
    assert text["mtp_layers_block_type"] is None
    assert text["router_compute_dtype"] == "float32"
    broken = bolt_config()
    broken["llm_config"]["num_nextn_predict_layers"] = 1
    with pytest.raises(ValueError, match="language topology"):
        plan.language_config(broken)


def test_moe_layer_args_take_puzzle_expert_count():
    from mlx2.runtime.models.nemotron_h import _moe_layer_args

    args = SimpleNamespace(n_routed_experts=496, moe_intermediate_size=2560, num_experts_per_tok=9)
    layer = _moe_layer_args(args, {"n_routed_experts": 240, "moe_intermediate_size": 2048})
    assert (layer.n_routed_experts, layer.moe_intermediate_size) == (240, 2048)
    assert args.n_routed_experts == 496


def lilicorr_config():
    return {
        "architectures": ["LiLiCorrDraftModel"], "model_type": "qwen3", "block_size": 8,
        "dflash_config": {
            "mask_token_id": 990, "target_layer_ids": [1, 18, 35, 51, 68, 85],
            "use_swa": True, "swa_window_size": 1024, "causal": False,
            "attention_sink_bias": True, "projector_type": "lilicorr",
            "lilicorr_candidate_topk": 8, "lilicorr_hidden_size": 1024,
            "lilicorr_num_layers": 2, "lilicorr_num_heads": 8, "lilicorr_mlp_ratio": 2.0,
            "lilicorr_factor_dim": 1024, "lilicorr_vector_eps": 1e-4,
            "lilicorr_logit_scale": 8.0, "conv_kernel_size": 2, "conv_group_size": 16,
        },
        "hidden_size": 4096, "num_hidden_layers": 5, "num_attention_heads": 32,
        "num_key_value_heads": 2, "head_dim": 128, "intermediate_size": 12288,
        "rms_norm_eps": 1e-6, "vocab_size": 131072, "max_position_embeddings": 262144,
        "attention_bias": False, "rope_theta": 10000.0, "rope_scaling": None,
        "tie_word_embeddings": False, "num_target_layers": 0,
        "layer_types": ["sliding_attention"] * 5, "sliding_window": 1024,
        "attention_sink_bias": True, "use_sliding_window": True,
    }


def test_bolt_lilicorr_args_are_shared_head_sliding_with_sinks():
    from mlx2.adapters import nemotron35_bolt as bolt

    args = bolt.lilicorr_executable_config(lilicorr_config())
    assert args.has_own_lm_head is False and args.attention_sink_bias is True
    assert args.num_target_layers == 88 and args.sliding_window == 1024
    wrong = lilicorr_config()
    wrong["dflash_config"]["target_layer_ids"] = [1, 22, 43, 64, 85]
    with pytest.raises(ValueError, match="geometry"):
        bolt.lilicorr_executable_config(wrong)


def test_lilicorr_sinks_require_the_sliding_backbone():
    from mlx2.adapters import nemotron35_bolt as bolt
    from mlx2.adapters.lilicorr import LiLiCorrConfig

    args = bolt.lilicorr_executable_config(lilicorr_config())
    values = copy.deepcopy(vars(args))
    values.update(layer_types=["full_attention"] * 5, sliding_window=None)
    with pytest.raises(ValueError):
        LiLiCorrConfig(**values)


def test_draft_attention_sinks_follow_the_extra_softmax_logit_law():
    import mlx.core as mx

    from mlx2.runtime.drafters.base_config import DFlashConfig
    from mlx2.runtime.drafters.dflash_base import DFlashAttention

    config = DFlashConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=16, num_hidden_layers=1, layer_types=["full_attention"])
    config.attention_sink_bias = True
    with mx.stream(mx.cpu):
        attention = DFlashAttention(config, 0)
        attention.attention_sink_bias = mx.array([0.5, -1.0, 2.0, 0.0])
        q = mx.random.normal((1, 4, 3, 16), key=mx.random.key(1))
        k = mx.random.normal((1, 2, 5, 16), key=mx.random.key(2))
        v = mx.random.normal((1, 2, 5, 16), key=mx.random.key(3))
        fast = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=0.25, sinks=attention._sinks(q))
        kk, vv = mx.repeat(k, 2, axis=1), mx.repeat(v, 2, axis=1)
        logits = (q @ kk.swapaxes(-1, -2)) * 0.25
        sink = mx.broadcast_to(attention.attention_sink_bias[None, :, None, None], (1, 4, 3, 1))
        weights = mx.softmax(mx.concatenate([logits, sink], axis=-1), axis=-1)[..., :-1]
        reference = weights @ vv
        assert mx.allclose(fast, reference, atol=1e-5).item()


def test_shared_head_lilicorr_bind_validates_the_nemotron_contract():
    from mlx2.runtime.drafters.lilicorr import LiLiCorrDraftModel

    taps = (1, 18, 35, 51, 68, 85)
    contract = {
        "schema": "mlx2.nemotron-lilicorr-target.v1", "target_model_type": "nemotron_h_omni",
        "target_fingerprint": "f" * 64, "target_layer_ids": taps,
        "target_feature_width": 6 * 4096, "capture": "post-layer-pre-final-norm",
        "ordinary_reference_preserved": True, "speculative_rollback": "exact",
        "owned_lm_head": False, "owned_final_norm": True, "qualified": False,
    }
    draft = SimpleNamespace(config=SimpleNamespace(
        has_own_lm_head=False, target_layer_ids=list(taps), hidden_size=4096,
        vocab_size=131072, num_target_layers=88))
    target = SimpleNamespace(args=SimpleNamespace(hidden_size=4096, vocab_size=131072,
                                                  num_hidden_layers=88))
    with pytest.raises(ValueError, match="target interface"):
        LiLiCorrDraftModel.bind(draft, target, target_contract=contract)
    with pytest.raises(ValueError, match="contract differs"):
        LiLiCorrDraftModel.bind(draft, target, target_contract={**contract, "owned_lm_head": True})
