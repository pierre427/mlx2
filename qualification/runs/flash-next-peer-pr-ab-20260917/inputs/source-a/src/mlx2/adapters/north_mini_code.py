"""CPU-safe North Mini Code artifact inspection and serving adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import struct
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane

CACHE_LAYOUT = "north-mini-code-layer-segments-v1"
_SAFETENSORS_HEADER_LIMIT = 64 << 20
_DTYPE_BYTES = {"BF16": 2, "U32": 4}

NORTH_MINI_CODE = ModelDescriptor(
    model_type="cohere2_moe",
    family="north-mini-code",
    variant="1.0-ordinary",
    state_planes=frozenset(
        {StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.STREAMING,
            Capability.TOOLS,
            Capability.REASONING,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.north_mini_code.NorthMiniCodeAdapter",
        "qualification": "pending",
        "scope": "text-only",
        "speculation": "none-qualified",
    },
)


def _load_json(path: Path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path.name}")
            value[key] = item
        return value

    return json.loads(path.read_text(), object_pairs_hook=unique)


def _safe_index(path: Path) -> dict:
    value = _load_json(path)
    index = value.get("weight_map") if isinstance(value, dict) else None
    if not isinstance(index, dict) or not index:
        raise ValueError("North artifact must have a nonempty weight index")
    return index


def _quantized_shapes(shape, *, group_size, bits):
    if shape[-1] % group_size or (shape[-1] * bits) % 32:
        raise ValueError("North quantization dimensions are not integral")
    return {
        "weight": [*shape[:-1], shape[-1] * bits // 32],
        "scales": [*shape[:-1], shape[-1] // group_size],
        "biases": [*shape[:-1], shape[-1] // group_size],
    }


def _expected_weight_headers(config):
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"]) * int(config["head_dim"])
    kv_heads = int(config["num_key_value_heads"]) * int(config["head_dim"])
    experts = int(config["num_experts"])
    intermediate = int(config["intermediate_size"])
    dense_intermediate = int(config["prefix_dense_intermediate_size"])
    quant = config.get("quantization", config.get("quantization_config"))
    if not isinstance(quant, dict):
        raise ValueError("North artifact must declare quantization metadata")
    expected = {}

    def add_quantized(name, shape):
        override = quant.get(name, quant)
        if not isinstance(override, dict):
            raise ValueError(f"Invalid North quantization override: {name}")
        group_size = override.get("group_size")
        bits = override.get("bits")
        if type(group_size) is not int or type(bits) is not int or bits not in {4, 8}:
            raise ValueError(f"Invalid North quantization parameters: {name}")
        for suffix, final_shape in _quantized_shapes(
            shape, group_size=group_size, bits=bits
        ).items():
            expected[f"{name}.{suffix}"] = (
                "U32" if suffix == "weight" else "BF16",
                final_shape,
            )

    add_quantized("model.embed_tokens", [int(config["vocab_size"]), hidden])
    for index in range(int(config["num_hidden_layers"])):
        prefix = f"model.layers.{index}."
        expected[prefix + "input_layernorm.weight"] = ("BF16", [hidden])
        for projection, shape in (
            ("q_proj", [heads, hidden]),
            ("k_proj", [kv_heads, hidden]),
            ("v_proj", [kv_heads, hidden]),
            ("o_proj", [hidden, heads]),
        ):
            add_quantized(prefix + "self_attn." + projection, shape)
        if index < int(config["first_k_dense_replace"]):
            for projection, shape in (
                ("gate_proj", [dense_intermediate, hidden]),
                ("up_proj", [dense_intermediate, hidden]),
                ("down_proj", [hidden, dense_intermediate]),
            ):
                add_quantized(prefix + "mlp." + projection, shape)
        else:
            add_quantized(prefix + "mlp.gate", [experts, hidden])
            for projection, shape in (
                ("gate_proj", [experts, intermediate, hidden]),
                ("up_proj", [experts, intermediate, hidden]),
                ("down_proj", [experts, hidden, intermediate]),
            ):
                add_quantized(prefix + "mlp.switch_mlp." + projection, shape)
    expected["model.norm.weight"] = ("BF16", [hidden])
    return expected


def _validate_weight_headers(path, index):
    expected = _expected_weight_headers(_load_json(path / "config.json"))
    observed = {}
    header_digests = []
    for name in sorted(set(index.values())):
        shard = (path / name).resolve()
        size = shard.stat().st_size
        with shard.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"Truncated safetensors file: {name}")
            length = struct.unpack("<Q", raw_length)[0]
            if not 0 < length <= min(_SAFETENSORS_HEADER_LIMIT, size - 8):
                raise ValueError(f"Invalid safetensors header length: {name}")
            raw_header = stream.read(length)
        header = json.loads(
            raw_header,
            object_pairs_hook=lambda pairs: _unique_pairs(pairs, name),
        )
        if not isinstance(header, dict):
            raise ValueError(f"Invalid safetensors header object: {name}")
        header_digests.append(hashlib.sha256(raw_header).hexdigest())
        payload_size = size - 8 - length
        ranges = []
        for tensor, record in header.items():
            if tensor == "__metadata__":
                continue
            if tensor in observed:
                raise ValueError(f"Duplicate North tensor across shards: {tensor}")
            if not isinstance(record, dict) or set(record) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise ValueError(f"Invalid North tensor metadata: {tensor}")
            dtype = record["dtype"]
            shape = record["shape"]
            offsets = record["data_offsets"]
            if dtype not in _DTYPE_BYTES or not isinstance(shape, list) or not all(
                type(value) is int and value >= 0 for value in shape
            ):
                raise ValueError(f"Invalid North tensor dtype/shape: {tensor}")
            if not (
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(type(value) is int for value in offsets)
                and 0 <= offsets[0] <= offsets[1] <= payload_size
            ):
                raise ValueError(f"Invalid North tensor offsets: {tensor}")
            elements = math.prod(shape)
            if offsets[1] - offsets[0] != elements * _DTYPE_BYTES[dtype]:
                raise ValueError(f"Invalid North tensor byte size: {tensor}")
            if index.get(tensor) != name:
                raise ValueError(f"North index/shard mismatch: {tensor}")
            observed[tensor] = (dtype, shape)
            ranges.append((offsets[0], offsets[1], tensor))
        for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
            if previous[1] > current[0]:
                raise ValueError(
                    f"Overlapping North tensor payloads: {previous[2]}, {current[2]}"
                )
    if set(index) != set(observed):
        raise ValueError("North weight index does not match shard headers")
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        wrong = sorted(
            name
            for name in set(expected) & set(observed)
            if expected[name] != observed[name]
        )
        raise ValueError(
            "North checkpoint schema mismatch "
            f"(missing={missing}, extra={extra}, wrong={wrong})"
        )
    return header_digests


def _unique_pairs(pairs, label):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r} in {label}")
        value[key] = item
    return value


def inspect_artifact(model_path: str | Path) -> dict:
    """Validate metadata and shard presence without importing MLX."""
    path = Path(model_path).expanduser().resolve()
    config = _load_json(path / "config.json")
    expected = {
        "model_type": "cohere2_moe",
        "hidden_size": 2048,
        "head_dim": 128,
        "num_hidden_layers": 49,
        "intermediate_size": 768,
        "prefix_dense_intermediate_size": 3072,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "vocab_size": 262144,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_shared_experts": 0,
        "first_k_dense_replace": 1,
        "expert_selection_fn": "sigmoid",
        "sliding_window": 4096,
        "rope_theta": 50000,
        "max_position_embeddings": 500000,
        "use_parallel_block": True,
        "use_qk_norm": False,
        "norm_topk_prob": False,
        "tie_word_embeddings": None,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("artifact topology does not match North Mini Code 1.0")
    layer_types = config.get("layer_types")
    required_layers = [
        "full_attention" if i % 4 == 0 else "sliding_attention" for i in range(49)
    ]
    if layer_types != required_layers:
        raise ValueError("artifact layer order does not match North Mini Code 1.0")
    if config.get("architectures") != ["Cohere2MoeForCausalLM"]:
        raise ValueError("artifact architecture does not match North Mini Code 1.0")
    index = _safe_index(path / "model.safetensors.index.json")
    names = sorted(set(index.values()))
    records = []
    for name in names:
        if not isinstance(name, str):
            raise TypeError("weight shard path must be a string")
        item = (path / name).resolve()
        if (
            not item.is_relative_to(path)
            or item.suffix != ".safetensors"
            or not item.is_file()
        ):
            raise ValueError("weight index must reference local safetensors files")
        stat = item.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
    if any(
        marker in key.lower() for key in index for marker in ("mtp.", "eagle", "draft")
    ):
        raise ValueError("embedded speculative tensors are not a North target artifact")
    if "model.embed_tokens.weight" not in index or "lm_head.weight" in index:
        raise ValueError("North artifact must use its tied embedding output head")
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    header_digests = _validate_weight_headers(path, index)
    for record in records:
        digest.update(json.dumps(record).encode())
    digest.update(json.dumps(header_digests).encode())
    return {
        "config": config,
        "weight_map": index,
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
            "header_sha256": header_digests,
        },
        "layers": 49,
        "sliding_layers": 36,
        "global_layers": 13,
        "sliding_window": 4096,
        "supports_native_mtp": False,
        "qualification": "pending",
    }


def configure_environment() -> dict[str, str]:
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_LM_SEGMENTED_SELF_MTP": "0",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "0",
        "MLX_LM_SHARED_QSA_SUFFIX": "0",
    }
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


def normalize_messages(messages: list[dict]) -> list[dict]:
    messages = copy.deepcopy(messages)
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise ValueError("North mlx2 port accepts text content only")
            message["content"] = "".join(part["text"] for part in content)
        for call in message.get("tool_calls", []):
            arguments = call["function"].get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise TypeError("North tool arguments must be a JSON object")
            call["function"]["arguments"] = arguments
    return messages


def reasoning_policy(request: dict) -> tuple[str, bool]:
    """Resolve North reasoning controls once for prompting, parsing, and receipts."""
    effort = request.get("reasoning_effort")
    if effort is None:
        effort = "high" if request.get("enable_thinking") is True else "none"
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
    if effort not in allowed:
        raise ValueError("Unsupported North reasoning_effort")
    thinking = bool(request.get("enable_thinking", effort != "none"))
    return effort, thinking


class NorthMiniCodeAdapter:
    descriptor = NORTH_MINI_CODE
    reasoning_effort_semantics = "thinking_toggle"

    @staticmethod
    def thinking_enabled(request: dict) -> bool:
        return reasoning_policy(request)[1]

    @staticmethod
    def profile_name(mtp):
        if mtp:
            raise ValueError("North Mini Code has no qualified native MTP route")
        return "north-mini-code-apcv2-ordinary"

    def execution_config(self, *, max_lanes, prefill_step):
        return {
            "persistent": True,
            "num_draft": 0,
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }

    def cache_budget(self, *, mtp):
        from .north_memory import NorthCacheBudget

        return NorthCacheBudget.from_config(self.config, mtp=mtp)

    def __init__(self, model_path: str, *, execution_policy=None):
        if execution_policy not in (None, {}):
            raise ValueError("North execution policy has no qualified overrides")
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        self.environment = configure_environment()
        self.layout = CACHE_LAYOUT
        path = Path(self.identity["path"])
        config = artifact["config"]
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.cohere2_moe import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import ubc_evict_paths

        self.model = Model(ModelArgs.from_dict(config))
        weights = {}
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        for file in files:
            weights.update(mx.load(str(file)))
        weights = self.model.sanitize(weights)
        quant = config.get("quantization", config.get("quantization_config"))
        if quant:
            def predicate(name, module):
                override = quant.get(name)
                if isinstance(override, dict):
                    return override
                return hasattr(module, "to_quantized") and f"{name}.scales" in weights

            nn.quantize(
                self.model,
                group_size=quant["group_size"],
                bits=quant["bits"],
                mode=quant.get("mode", "affine"),
                class_predicate=predicate,
            )
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        ubc_evict_paths([str(file) for file in files])
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        eos = config.get("eos_token_id")
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=[eos] if isinstance(eos, int) else list(eos or []),
        )
        self.max_context = int(config["max_position_embeddings"])

    def prompt_tokens(self, request: dict) -> list[int]:
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        effort, thinking = reasoning_policy(request)
        return self.tokenizer.apply_chat_template(
            normalize_messages(request["messages"]),
            add_generation_prompt=True,
            tokenize=True,
            reasoning=thinking,
            reasoning_effort=effort,
            skip_thinking=not thinking,
            tools=request.get("tools") if request.get("tool_choice") != "none" else None,
        )

    def output_parser(self, request):
        from .north_output import NorthOutputParser

        _, thinking = reasoning_policy(request)
        return NorthOutputParser(
            chat="messages" in request,
            thinking=thinking,
            tools=request.get("tools") if request.get("tool_choice") != "none" else None,
            stops=request.get("stop", ()),
        )

    def diagnostics(self):
        return {
            "architecture": "cohere2_moe",
            "cache_layout": self.layout,
            "sliding_layers": 36,
            "global_layers": 13,
            "sliding_window": 4096,
            "speculation": "none-qualified",
        }

    def close(self):
        """Release model ownership before the shared worker shuts down."""

        had_resources = any(
            getattr(self, name, None) is not None for name in ("model", "tokenizer")
        )
        self.model = None
        self.tokenizer = None
        if had_resources:
            import mlx.core as mx

            mx.clear_cache()
