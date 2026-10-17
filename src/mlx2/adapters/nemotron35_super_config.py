"""CPU configuration contract for Nemotron 3.5 Super VL and its draft taps."""
import copy
import math

SOURCE_REVISION = "533d66230b6e75518bf32d2d7a7b95a24c3dd517"
SOURCE_CONFIG_SHA256 = "be121eafe4c174c8bc7204da2cd063baa0392c0adfd41dd82fc9e0bba5b0f9a0"
SOURCE_INDEX_SHA256 = "ce355e06784fc7e06d47e9c55d9a28125171fa18d1d1a9027ae50a5ea39dba2b"


def language_config(config):
    if (config.get("model_type") != "nemotron_h_omni"
            or config.get("architectures") != ["NemotronH_Omni_Reasoning_V3"]
            or not isinstance(config.get("llm_config"), dict)):
        raise ValueError("Expected Nemotron 3.5 Super VL composition")
    text = copy.deepcopy(config["llm_config"])
    expected = {"model_type": "nemotron_h", "hidden_size": 4096,
                "vocab_size": 131072, "num_attention_heads": 32,
                "num_key_value_heads": 2, "head_dim": 128,
                "mamba_num_heads": 128, "mamba_head_dim": 64,
                "ssm_state_size": 128, "n_groups": 8, "conv_kernel": 4,
                "n_routed_experts": 512, "num_experts_per_tok": 22,
                "moe_latent_size": 1024, "moe_intermediate_size": 2688,
                "moe_shared_expert_intermediate_size": 5376,
                "num_nextn_predict_layers": 1, "mtp_layers_block_type": ["attention", "moe"],
                "mamba_ssm_cache_dtype": "float32", "mlp_hidden_act": "relu2",
                "max_position_embeddings": 1048576}
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported Nemotron 3.5 Super language topology")
    layers = text.get("layers_block_type")
    if (not isinstance(layers, list) or len(layers) != 88
            or {k: layers.count(k) for k in set(layers)} != {"mamba": 40, "moe": 40, "attention": 8}):
        raise ValueError("Unsupported Nemotron 3.5 Super hybrid order")
    # This release declares its inference bound explicitly. time_step_min is
    # initialization metadata, and must not replace an explicit lower bound.
    limit = text.get("time_step_limit")
    if limit != [0.0, {"__float__": "Infinity"}]:
        raise ValueError("Unsupported Nemotron 3.5 Super time-step bound")
    text["time_step_limit"] = (0.0, math.inf)
    text["num_hidden_layers"] = len(layers)
    text["router_compute_dtype"] = "float32"
    return text


def vision_config(config):
    vision = config.get("vision_config", {})
    if (any(vision.get(k) != v for k, v in {
            "model_type": "radio", "hidden_size": 1280, "num_hidden_layers": 32,
            "num_attention_heads": 16, "patch_size": 16, "num_cls_tokens": 3,
            "num_registers": 7, "summary_idxs": [0, 1],
            "video_temporal_patch_size": 2}.items())
            or config.get("sound_config") is not None
            or config.get("downsample_ratio") != .5
            or config.get("img_context_token_id") != 18
            or config.get("projector_hidden_size") != 20480):
        raise ValueError("Unsupported Super VL RADIO/projector configuration")
    return copy.deepcopy(vision)


def tensor_schema(config, *, fused=False):
    """Exact published geometry, generated without constructing any tensors."""
    text = language_config(config)
    vision_config(config)
    schema = {}

    def add(name, shape, dtype="BF16"):
        schema[name] = {"shape": shape, "dtype": dtype}

    def language_layer(prefix, kind):
        add(prefix + ".norm.weight", [4096])
        p = prefix + ".mixer."
        if kind == "mamba":
            for name, shape in {
                "A_log": [128], "D": [128], "dt_bias": [128],
                "conv1d.weight": [10240, 1, 4], "conv1d.bias": [10240],
                "in_proj.weight": [18560, 4096], "norm.weight": [8192],
                "out_proj.weight": [4096, 8192],
            }.items():
                add(p + name, shape)
        elif kind == "attention":
            for name, outputs in (("q_proj", 4096), ("k_proj", 256),
                                  ("v_proj", 256), ("o_proj", 4096)):
                add(p + name + ".weight", [outputs, 4096])
        elif kind == "moe":
            add(p + "gate.weight", [512, 4096])
            add(p + "gate.e_score_correction_bias", [512], "F32")
            add(p + "fc1_latent_proj.weight", [1024, 4096])
            add(p + "fc2_latent_proj.weight", [4096, 1024])
            add(p + "shared_experts.up_proj.weight", [5376, 4096])
            add(p + "shared_experts.down_proj.weight", [4096, 5376])
            if fused:
                add(p + "switch_mlp.fc1.weight", [512, 2688, 1024])
                add(p + "switch_mlp.fc2.weight", [512, 1024, 2688])
            else:
                for expert in range(512):
                    add(p + f"experts.{expert}.up_proj.weight", [2688, 1024])
                    add(p + f"experts.{expert}.down_proj.weight", [1024, 2688])
        else:
            raise ValueError("Unsupported hybrid block")

    p = "language_model."
    add(p + "backbone.embeddings.weight", [131072, 4096])
    add(p + "backbone.norm_f.weight", [4096])
    add(p + "lm_head.weight", [131072, 4096])
    for i, kind in enumerate(text["layers_block_type"]):
        language_layer(p + f"backbone.layers.{i}", kind)
    for i, kind in enumerate(text["mtp_layers_block_type"]):
        language_layer(p + f"mtp.layers.{i}", kind)
    for name in ("enorm", "hnorm"):
        add(p + f"mtp.layers.0.{name}.weight", [4096])
    add(p + "mtp.layers.0.eh_proj.weight", [4096, 8192])
    add(p + "mtp.layers.1.final_layernorm.weight", [4096])
    p = "vision_model.radio_model."
    add(p + "input_conditioner.norm_mean", [3, 1, 1], "F32")
    add(p + "input_conditioner.norm_std", [3, 1, 1], "F32")
    add("vision_model.summary_idxs", [2], "I64")
    add(p + "model.patch_generator.cls_token.token", [10, 1280])
    add(p + "model.patch_generator.embedder.weight", [1280, 768])
    add(p + "model.patch_generator.video_embedder.weight", [1280, 1536])
    add(p + "model.patch_generator.pos_embed", [1, 16384, 1280])
    for i in range(32):
        q = p + f"model.blocks.{i}."
        for name in ("norm1", "norm2"):
            add(q + name + ".weight", [1280])
            add(q + name + ".bias", [1280])
        for name, inputs, outputs in (("attn.qkv", 1280, 3840),
                                     ("attn.proj", 1280, 1280),
                                     ("mlp.fc1", 1280, 5120),
                                     ("mlp.fc2", 5120, 1280)):
            add(q + name + ".weight", [outputs, inputs])
            add(q + name + ".bias", [outputs])
    add("mlp1.0.weight", [5120])
    add("mlp1.1.weight", [20480, 5120])
    add("mlp1.3.weight", [4096, 20480])
    for name in ("weight", "bias"):
        add("vision_projector.vision_final_layernorm." + name, [1280])
    return schema
