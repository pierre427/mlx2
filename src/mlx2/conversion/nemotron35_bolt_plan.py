"""Bolt geometry and mixed affine plan, independent of tensor payloads or MLX.

Bolt is a Puzzle-NAS pruning of the Super language stack: identical hybrid
order and widths, but each MoE layer declares its own routed-expert count and
expert width, and there is no MTP head.  The plan applies Super's
affine-experts4-protected8 law to the dequantized (BF16) tensors, except that
routed experts use group 32 so each affine group covers exactly one IQ2_XXS
sub-block (one source scale).  See provenance/nemotron35-bolt.json.
"""
from __future__ import annotations

import copy
import math

from .nvfp4 import DTYPE_BYTES

RECIPE = "nemotron35-bolt-affine-experts4g32-protected8-v1"
EXPERT_GROUP = 32
PROTECTED_GROUP = 64


def language_config(config):
    """Runtime language config for the mlx2 Nemotron-H model."""
    text = config.get("llm_config") if isinstance(config, dict) else None
    if (
        config.get("model_type") != "nemotron_h_omni"
        or not isinstance(text, dict)
        or text.get("model_type") != "nemotron_h_puzzle"
    ):
        raise ValueError("Expected a Nemotron 3.5 Bolt Puzzle composition")
    text = copy.deepcopy(text)
    blocks = text["block_configs"]
    layers = text["layers_block_type"]
    if (
        len(layers) != 88
        or [item.get("block_type") for item in blocks] != layers
        or text.get("num_nextn_predict_layers") != 0
        or text.get("mtp_layers_block_type") != []
        or text.get("time_step_limit") != [0.0, {"__float__": "Infinity"}]
        or text.get("hidden_size") != 4096
        or text.get("moe_latent_size") != 1024
        or text.get("moe_shared_expert_intermediate_size") != 5376
        or text.get("n_group") != 1
        or text.get("topk_group") != 1
    ):
        raise ValueError("Unsupported Nemotron 3.5 Bolt language topology")
    for block in blocks:
        if block["block_type"] != "moe":
            continue
        if (
            type(block.get("n_routed_experts")) is not int
            or type(block.get("moe_intermediate_size")) is not int
            or block.get("num_experts_per_tok") != 9
            or block["moe_intermediate_size"] % 256
        ):
            raise ValueError("Unsupported Nemotron 3.5 Bolt MoE block")
    text["time_step_limit"] = (0.0, math.inf)
    text["num_hidden_layers"] = len(layers)
    text["router_compute_dtype"] = "float32"
    # Uniform fields are placeholders; MoE layers take their per-block dims.
    text["n_routed_experts"] = max(b.get("n_routed_experts", 0) for b in blocks)
    text["moe_intermediate_size"] = max(b.get("moe_intermediate_size", 0) for b in blocks)
    text["num_experts_per_tok"] = 9
    text["mtp_layers_block_type"] = None
    text["mtp_block_configs"] = None
    return text


def tensor_schema(config, *, fused=False):
    """Logical BF16 geometry (dequantized), generated without tensors."""
    text = language_config(config)
    schema = {}

    def add(name, shape, dtype="BF16"):
        schema[name] = {"shape": shape, "dtype": dtype}

    p = "language_model."
    add(p + "backbone.embeddings.weight", [131072, 4096])
    add(p + "backbone.norm_f.weight", [4096])
    add(p + "lm_head.weight", [131072, 4096])
    for i, block in enumerate(text["block_configs"]):
        prefix = p + f"backbone.layers.{i}"
        add(prefix + ".norm.weight", [4096])
        m = prefix + ".mixer."
        kind = block["block_type"]
        if kind == "mamba":
            for name, shape in {
                "A_log": [128], "D": [128], "dt_bias": [128],
                "conv1d.weight": [10240, 1, 4], "conv1d.bias": [10240],
                "in_proj.weight": [18560, 4096], "norm.weight": [8192],
                "out_proj.weight": [4096, 8192],
            }.items():
                add(m + name, shape)
        elif kind == "attention":
            for name, outputs in (("q_proj", 4096), ("k_proj", 256),
                                  ("v_proj", 256), ("o_proj", 4096)):
                add(m + name + ".weight", [outputs, 4096])
        elif kind == "moe":
            experts, width = block["n_routed_experts"], block["moe_intermediate_size"]
            add(m + "gate.weight", [experts, 4096])
            add(m + "gate.e_score_correction_bias", [experts], "F32")
            add(m + "fc1_latent_proj.weight", [1024, 4096])
            add(m + "fc2_latent_proj.weight", [4096, 1024])
            add(m + "shared_experts.up_proj.weight", [5376, 4096])
            add(m + "shared_experts.down_proj.weight", [4096, 5376])
            if fused:
                add(m + "switch_mlp.fc1.weight", [experts, width, 1024])
                add(m + "switch_mlp.fc2.weight", [experts, 1024, width])
            else:
                for expert in range(experts):
                    add(m + f"experts.{expert}.up_proj.weight", [width, 1024])
                    add(m + f"experts.{expert}.down_proj.weight", [1024, width])
        else:
            raise ValueError("Unsupported hybrid block")
    v = "vision_model.radio_model."
    add(v + "input_conditioner.norm_mean", [3, 1, 1], "F32")
    add(v + "input_conditioner.norm_std", [3, 1, 1], "F32")
    add("vision_model.summary_idxs", [2], "I64")
    add(v + "model.patch_generator.cls_token.token", [10, 1280])
    add(v + "model.patch_generator.embedder.weight", [1280, 768])
    add(v + "model.patch_generator.video_embedder.weight", [1280, 1536])
    add(v + "model.patch_generator.pos_embed", [1, 16384, 1280])
    for i in range(32):
        q = v + f"model.blocks.{i}."
        for name in ("norm1", "norm2"):
            add(q + name + ".weight", [1280])
            add(q + name + ".bias", [1280])
        for name, inputs, outputs in (("attn.qkv", 1280, 3840), ("attn.proj", 1280, 1280),
                                      ("mlp.fc1", 1280, 5120), ("mlp.fc2", 5120, 1280)):
            add(q + name + ".weight", [outputs, inputs])
            add(q + name + ".bias", [outputs])
    add("mlp1.0.weight", [5120])
    add("mlp1.1.weight", [20480, 5120])
    add("mlp1.3.weight", [4096, 20480])
    return schema


def precision_for(name, entry):
    """Super's protected-8 law; routed experts at affine4 group 32."""
    if not name.startswith("language_model.") or not name.endswith(".weight"):
        return None
    if ".switch_mlp." in name:
        return {"bits": 4, "group_size": EXPERT_GROUP, "mode": "affine"}
    if (len(entry["shape"]) != 2 or ".conv1d." in name or ".gate." in name
            or "latent_proj." in name or ".embeddings." in name):
        return None
    return {"bits": 8, "group_size": PROTECTED_GROUP, "mode": "affine"}


def make_plan(config):
    output, modules, categories = {}, {}, {}
    for name, entry in tensor_schema(config, fused=True).items():
        quant = precision_for(name, entry)
        if quant is None:
            output[name] = dict(entry)
            category = "source_precision"
            names = [name]
        else:
            prefix, shape = name.removesuffix(".weight"), entry["shape"]
            bits, group = quant["bits"], quant["group_size"]
            if shape[-1] % group or shape[-1] * bits % 32:
                raise ValueError("Invalid mixed-quant group geometry")
            output[name] = {"shape": [*shape[:-1], shape[-1] * bits // 32], "dtype": "U32"}
            for suffix in (".scales", ".biases"):
                output[prefix + suffix] = {"shape": [*shape[:-1], shape[-1] // group],
                                           "dtype": "BF16"}
            modules[prefix] = quant
            category = "routed_experts" if ".switch_mlp." in name else "protected_8bit"
            names = [name, prefix + ".scales", prefix + ".biases"]
        categories[category] = categories.get(category, 0) + sum(
            math.prod(output[key]["shape"]) * DTYPE_BYTES[output[key]["dtype"]]
            for key in names
        )
    return {"recipe": RECIPE, "expert_bits": 4, "expert_group_size": EXPERT_GROUP,
            "schema": output, "modules": modules,
            "tensor_bytes": sum(categories.values()),
            "bytes_by_category": categories, "activation_aware": False,
            "source_law": "dequantize GGML IQ2_XXS/Q8_0 and ModelOpt NVFP4 to BF16, then refit",
            "qualified": False, "selected": False}
