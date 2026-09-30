"""Separate, exact LLaDA denoising route; never an autoregressive adapter.

Metadata inspection is CPU safe.  Construction and direct generation require
an explicit later execution window.  The ordinary mlx2 scheduler must not
register this class: bidirectional denoising has no autoregressive KV state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from .artifact_paths import shard_within_artifact
from ..contracts import Capability, ModelDescriptor, StatePlane


LLADA = ModelDescriptor(
    model_type="llada",
    family="llada-8b",
    variant="diffusion-exact",
    state_planes=frozenset({StatePlane.RNG, StatePlane.TRANSCRIPT}),
    capabilities=frozenset({Capability.TEXT}),
    cache_layout=None,
    metadata={
        "execution": "mlx2.adapters.llada.LLaDADenoisingAdapter",
        "execution_kind": "bidirectional-denoising",
        "qualification": "pending",
        "scope": "direct synchronous generation, batch width one, exact full-forward schedule",
    },
)


def _json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path.name}")
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return value


def inspect_artifact(model_path: str | Path) -> dict:
    """Inspect the three local 8B checkpoint layouts without opening tensors."""
    path = Path(model_path).expanduser().resolve()
    config = _json(path / "config.json")
    expected = {
        "model_type": "llada", "architectures": ["LLaDAModelLM"],
        "d_model": 4096, "n_layers": 32, "n_heads": 32, "n_kv_heads": 32,
        "mlp_hidden_size": 12288, "vocab_size": 126464,
        "embedding_size": 126464, "mask_token_id": 126336,
        "eos_token_id": 126081, "rope_theta": 500000.0,
        "weight_tying": False, "use_cache": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("LLaDA 8B bidirectional topology does not match")
    quant = config.get("quantization")
    if quant is not None and quant not in (
        {"group_size": 64, "bits": 4, "mode": "affine"},
        {"group_size": 64, "bits": 6, "mode": "affine"},
    ):
        raise ValueError("unsupported LLaDA quantization layout")
    weight_map = _json(path / "model.safetensors.index.json").get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("LLaDA requires a nonempty weight index")
    if quant is None:
        required = {"model.transformer.wte.weight", "model.transformer.ln_f.weight", "model.transformer.ff_out.weight"}
        for layer in range(32):
            prefix = f"model.transformer.blocks.{layer}."
            required.update(prefix + suffix for suffix in (
                "q_proj.weight", "k_proj.weight", "v_proj.weight", "attn_out.weight",
                "ff_proj.weight", "up_proj.weight", "ff_out.weight",
            ))
    else:
        required = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
        for layer in range(32):
            prefix = f"model.layers.{layer}."
            required.update(prefix + suffix for suffix in (
                "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
            ))
    if not required <= weight_map.keys():
        raise ValueError("LLaDA indexed tensor topology is incomplete")
    names = sorted(set(weight_map.values()))
    records = []
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or Path(name).suffix != ".safetensors":
            raise ValueError("LLaDA index has an unsafe shard path")
        item = path / name
        if not item.is_file() or not shard_within_artifact(path, item.resolve()):
            raise ValueError(f"missing or foreign LLaDA shard: {name}")
        stat = item.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    for record in records:
        digest.update(json.dumps(record).encode())
    return {
        "config": config, "weight_map": weight_map, "quantized": quant is not None,
        "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": records},
        "execution_kind": "bidirectional-denoising", "has_autoregressive_cache": False,
    }


def validate_generation(*, prompt_length: int, gen_length: int, block_length: int, steps: int, temperature: float) -> None:
    """Pure metadata preflight before allocating a denoising canvas."""
    if any(type(value) is not int or value <= 0 for value in (prompt_length, gen_length, block_length, steps)):
        raise ValueError("LLaDA prompt length, generation length, block length, and steps must be positive integers")
    if gen_length % block_length or steps % (gen_length // block_length):
        raise ValueError("LLaDA generation geometry requires whole blocks and steps")
    if prompt_length + gen_length > 4096:
        raise ValueError("LLaDA canvas exceeds artifact sequence length")
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ValueError("LLaDA temperature must be between 0 and 2")


class LLaDADenoisingAdapter:
    """Direct exact full-forward sampler, outside normal serving dispatch."""

    descriptor = LLADA
    execution_kind = "bidirectional-denoising"

    def __init__(self, model_path: str):
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        path = Path(self.identity["path"])

        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer
        from ..runtime.models.llada import Model, ModelArgs
        from ..runtime.ubc_evict import load_shards_evicting

        self.model = Model(ModelArgs.from_dict(self.config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = self.config.get("quantization")
        if quant is not None:
            nn.quantize(
                self.model, group_size=quant["group_size"], bits=quant["bits"],
                mode=quant["mode"], class_predicate=lambda name, module:
                hasattr(module, "to_quantized") and f"{name}.scales" in weights,
            )
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)

    def generate(self, *, prompt: str | None = None, messages: list[dict] | None = None,
                 gen_length: int = 128, block_length: int = 128,
                 steps: int = 128, temperature: float = 0.0) -> dict:
        """Generate one response with the source model's exact denoising loop."""
        if (prompt is None) == (messages is None):
            raise ValueError("supply exactly one of prompt or messages")
        if messages is not None:
            if not isinstance(messages, list) or not messages or any(
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                raise ValueError("LLaDA messages must be plain text chat turns")
            tokens = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            # Current Transformers returns BatchEncoding for this tokenizer,
            # while older revisions returned the flat token list directly.
            if isinstance(tokens, Mapping):
                tokens = tokens["input_ids"]
        else:
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        validate_generation(prompt_length=len(tokens), gen_length=gen_length,
                            block_length=block_length, steps=steps, temperature=temperature)
        import mlx.core as mx
        from ..runtime.models.llada import generate
        output, text, stats = generate(
            self.model, mx.array([tokens]), steps=steps, gen_length=gen_length,
            block_length=block_length, temperature=float(temperature),
            cfg_scale=0.0, remasking="low_confidence",
            mask_id=int(self.config["mask_token_id"]), tokenizer=self.tokenizer,
            return_stats=True, kv_cache=False, dual_cache=False,
            incremental_cache=False, parallel_threshold=None,
        )
        canvas_ids = [int(token) for token in output[0].tolist()]
        terminal_ids = {int(self.config["eos_token_id"])}
        end_turn_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if type(end_turn_id) is int and end_turn_id >= 0:
            terminal_ids.add(end_turn_id)
        stop_index = next((i for i, token in enumerate(canvas_ids)
                           if token in terminal_ids), len(canvas_ids))
        visible_ids = canvas_ids[:stop_index]
        text = self.tokenizer.decode(visible_ids, skip_special_tokens=True)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("LLaDA denoising produced no client-visible text")
        return {
            "text": text, "token_ids": visible_ids, "canvas_token_ids": canvas_ids,
            "stop_index": stop_index, "stats": stats,
            "route": "denoising-exact", "artifact_fingerprint": self.identity["fingerprint"],
        }

    def close(self):
        self.model = None
        self.tokenizer = None
