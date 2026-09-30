"""CPU-safe GraniteMoeSWA 3B ordinary candidate adapter."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

from .artifact_paths import shard_within_artifact
from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .ordinary_text import OrdinaryTextAdapter


DESCRIPTOR = ModelDescriptor(
    model_type="granitemoe_swa", family="granite-swash", variant="3b-a600m-ordinary",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}),
    capabilities=frozenset({Capability.TEXT, Capability.STREAMING, Capability.CONTINUOUS_BATCH,
                            Capability.PREFIX_REUSE, Capability.APC_V2, Capability.LAYERED_CACHE}),
    cache_layout="granite-swash-full-swa-kv-v1",
    metadata={"execution": "mlx2.adapters.granite_swa.GraniteSWAAdapter",
              "qualification": "pending", "scope": "ordinary text only"},
)

# The local generation_config.json has do_sample=true and no sampling fields.
SAMPLING = VendorSampling.single(
    SamplingDefaults(temperature=1.0, source=GENERATION_CONFIG,
                     note="do_sample=true; temperature omitted"),
    model="granite-swash-3b-a600m local artifacts",
)


def _metadata(model_path):
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    if not isinstance(config, dict):
        raise ValueError("Granite config must be an object")
    quant = config.get("quantization")
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Granite weight index is empty")
    else:
        # The bf16 source checkpoint is a single safetensors file without an
        # index. Read only its bounded JSON header, never tensor payloads.
        if quant is not None:
            raise ValueError("quantized Granite artifact requires an index")
        with (path / "model.safetensors").open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError("short Granite safetensors header")
            size = struct.unpack("<Q", raw)[0]
            if not 0 < size <= 16 << 20:
                raise ValueError("invalid Granite safetensors header size")
            header = json.loads(stream.read(size))
        if not isinstance(header, dict):
            raise ValueError("invalid Granite safetensors header")
        weight_map = {key: "model.safetensors" for key in header if key != "__metadata__"}
    names = sorted(set(weight_map.values()))
    records = []
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or Path(name).suffix != ".safetensors":
            raise ValueError("unsafe Granite shard path")
        item = path / name
        if not item.is_file() or not shard_within_artifact(path, item.resolve()):
            raise ValueError(f"missing Granite shard: {name}")
        stat = item.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        item = path / name
        if item.is_file():
            digest.update(name.encode()); digest.update(item.read_bytes())
    for record in records:
        digest.update(json.dumps(record).encode())
    return path, config, weight_map, {"path": str(path), "fingerprint": digest.hexdigest(), "files": records}


def inspect_artifact(model_path: str | Path) -> dict:
    path, config, weights, identity = _metadata(model_path)
    expected = {
        "model_type": "granitemoe_swa", "architectures": ["GraniteMoeSWAForCausalLM"],
        "vocab_size": 100352, "hidden_size": 1280, "intermediate_size": 512,
        "num_hidden_layers": 28, "num_attention_heads": 20, "num_key_value_heads": 4,
        "num_local_experts": 48, "num_experts_per_tok": 4,
        "shared_intermediate_size": 1280, "sliding_window": 128,
        "max_position_embeddings": 8192, "tie_word_embeddings": True,
        "embedding_multiplier": 12, "attention_multiplier": 0.015625,
        "residual_multiplier": 0.26, "logits_scaling": 5,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Granite SWA artifact topology does not match local 3B target")
    # The port gives every layer the global rope and derives head_dim from
    # hidden_size / heads.  A same-shaped checkpoint carrying per-layer rope
    # (layer_rope_theta, 0 = no rope) or its own head_dim would load and serve
    # silently wrong logits, so both fail closed.
    if config.get("layer_rope_theta") is not None:
        raise ValueError("Granite SWA per-layer rope (layer_rope_theta) is not implemented")
    if config.get("head_dim") not in (None, expected["hidden_size"] // expected["num_attention_heads"]):
        raise ValueError("Granite SWA head_dim differs from hidden_size / num_attention_heads")
    layers = config.get("layer_types")
    if layers != ["full_attention" if i in (0, 3, 7, 11, 15, 19, 23, 27)
                  else "sliding_attention" for i in range(28)]:
        raise ValueError("Granite full/sliding layer order mismatch")
    quant = config.get("quantization")
    if quant is not None and (quant.get("group_size"), quant.get("bits"), quant.get("mode")) != (64, 4, "affine"):
        raise ValueError("unsupported Granite quantization")
    required = {"model.embed_tokens.weight", "model.norm.weight"}
    for layer in range(28):
        prefix = f"model.layers.{layer}."
        required.update(prefix + suffix for suffix in (
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "self_attn.sinks", "block_sparse_moe.router.weight",
            "shared_mlp.input_linear.weight", "shared_mlp.output_linear.weight",
        ))
        if quant:
            required.update({prefix + "block_sparse_moe.experts.gate_proj.weight",
                             prefix + "block_sparse_moe.experts.up_proj.weight"})
        else:
            required.add(prefix + "block_sparse_moe.experts.gate_up_proj")
    if not required <= weights.keys():
        raise ValueError("Granite indexed tensor topology is incomplete")
    return {"config": config, "weight_map": weights, "identity": identity,
            "quantized": bool(quant), "full_layers": 8, "sliding_layers": 20,
            "has_mtp": False}


class GraniteSWAAdapter(OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    profile = "granite-swash-apcv2-ordinary"

    def __init__(self, model_path: str, *, execution_policy=None):
        if execution_policy not in (None, {}):
            raise ValueError("Granite SWA supports ordinary execution only")
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        self.layout = DESCRIPTOR.cache_layout
        self.environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MLX_ENABLE_TF32": "0"}
        os.environ.update(self.environment)
        path = Path(self.identity["path"])
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer
        from ..runtime.models.granitemoe_swa import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting
        self.model = Model(ModelArgs.from_dict(self.config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = self.config.get("quantization")
        if quant:
            def predicate(name, module):
                override = quant.get(name)
                if isinstance(override, dict):
                    return override
                return hasattr(module, "to_quantized") and f"{name}.scales" in weights
            nn.quantize(self.model, group_size=quant["group_size"], bits=quant["bits"],
                        mode=quant["mode"], class_predicate=predicate)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval(); mx.eval(self.model.parameters())
        weights.clear(); mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        # Some tokenizer classes rebuild the pre-tokenizer instead of reading tokenizer.json.
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        eos = self.config.get("eos_token_id", tokenizer.eos_token_id)
        eos_ids = eos if isinstance(eos, list) else [eos]
        self.tokenizer = TokenizerWrapper(tokenizer, detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=eos_ids)
        self.max_context = int(self.config["max_position_embeddings"])

    def close(self):
        self.model = None
        self.tokenizer = None
