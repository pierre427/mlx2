"""CPU-safe artifact inspection and ordinary Qwen3.5 9B text serving.

The checkpoint may contain an unimplemented vision tower and may advertise an
MTP layer in config.  This adapter deliberately loads only the language model,
and exposes neither capability unless the corresponding execution path is
implemented and qualified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .qwen38_27b import Qwen3827BAdapter


CACHE_LAYOUT = "qwen35-9b-hybrid-layer-segments-v1"


def descriptor_for(*, has_mtp: bool = False) -> ModelDescriptor:
    # ``has_mtp`` describes the selectable adapter route, not the config claim.
    # This initial adapter intentionally has no MTP implementation.
    if has_mtp:
        raise ValueError("Qwen3.5 9B MTP is not implemented")
    return ModelDescriptor(
        model_type="qwen3_5",
        family="qwen3.5-9b",
        variant="9b-ordinary",
        state_planes=frozenset(
            {
                StatePlane.ATTENTION_KV,
                StatePlane.RECURRENT,
                StatePlane.RNG,
                StatePlane.TRANSCRIPT,
            }
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
                Capability.PROMPT_LOOKUP,
                Capability.GRAMMAR,
            }
        ),
        cache_layout=CACHE_LAYOUT,
        metadata={
            "execution": "mlx2.adapters.qwen35_9b.Qwen359BAdapter",
            "qualification": "pending",
            "scope": "text-only",
            "context_profile": "32768-initial",
            "mtp": "not-implemented",
            "vision": "not-implemented",
        },
    )


QWEN35_9B = descriptor_for()


def inspect_artifact(model_path: str | Path) -> dict:
    """Validate the exact dense 9B topology without importing MLX."""
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config)
    expected = {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248320,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    }
    if config.get("model_type") != "qwen3_5" or text.get("num_experts", 0):
        raise ValueError("Qwen3.5 9B requires the dense qwen3_5 artifact layout")
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError("artifact topology does not match Qwen3.5 9B")
    expected_layers = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(32)
    ]
    if text.get("layer_types") not in (None, expected_layers):
        raise ValueError("artifact layer order does not match Qwen3.5 9B")

    index = json.loads((path / "model.safetensors.index.json").read_text()).get(
        "weight_map"
    )
    if not isinstance(index, dict) or not index:
        raise ValueError("artifact has no indexed weights")
    names = sorted(set(index.values()))
    for name in names:
        item = Path(name) if isinstance(name, str) else Path("/")
        if item.is_absolute() or ".." in item.parts:
            raise ValueError("weight shard paths must stay within the artifact")
        if not (path / item).is_file():
            raise ValueError(f"missing weight shard: {name}")

    mtp_keys = [key for key in index if key.startswith(("mtp.", "language_model.mtp."))]
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
    records = []
    for name in names:
        stat = (path / name).stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
    return {
        "config": config,
        "weight_map": index,
        # Config alone does not confer a route, and this artifact has no MTP tensors.
        "has_mtp": False,
        "advertised_mtp_layers": int(text.get("mtp_num_hidden_layers", 0)),
        "mtp_tensor_count": len(mtp_keys),
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
        },
    }


class Qwen359BAdapter(Qwen3827BAdapter):
    """Ordinary-decode, text-only adapter for the dense Qwen3.5 9B artifact."""

    default_route = "ordinary"
    descriptor = QWEN35_9B
    artifact_inspector = staticmethod(inspect_artifact)
    descriptor_builder = staticmethod(descriptor_for)

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("Qwen3.5 9B MTP is not implemented")
        return "qwen35-9b-apcv2-ordinary"

    def diagnostics(self):
        return {
            "architecture": "dense-hybrid-gdn-gqa",
            "layout": self.layout,
            "mtp_head_present": False,
            "scope": "text-only",
        }
