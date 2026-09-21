"""CPU-safe artifact inspection and ordinary Qwen3.5 9B text serving.

The checkpoint may contain an unimplemented vision tower and may advertise an
MTP layer in config.  This adapter deliberately loads only the language model,
and exposes neither capability unless the corresponding execution path is
implemented and qualified.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
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


def configure_environment() -> dict[str, str]:
    """Ordinary-only profile; disable inherited speculative toggles."""
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX_GDN_PACKED": "1",
        "MLX_GDN_CORE": "0",
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_LM_SEGMENTED_SELF_MTP": "0",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "0",
        "MLX_LM_SHARED_QSA_SUFFIX": "0",
        "MLX_LM_MTP_BOUNDARY_COW": "0",
    }
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


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
    environment_configurator = staticmethod(configure_environment)
    from .qwen import QWEN35_9B_SAMPLING as sampling_defaults

    def configure_neural_concept_bridge(self, artifact):
        """Bind the learned bridge to this exact dense hidden geometry."""
        text_config = self.model.args.text_config
        hidden_size = (
            text_config["hidden_size"]
            if isinstance(text_config, Mapping)
            else text_config.hidden_size
        )
        if artifact.hidden_dim != int(hidden_size):
            raise ValueError("neural concept bridge hidden dimension mismatch")
        self._neural_concept_artifact = artifact
        self._neural_concept_counts = {"prefills": 0, "concepts": 0, "tokens": 0}

    def neural_concept_prefill(self, tokens, payload, *, prefill_step):
        """Apply learned cross-attention to one request's prefill embeddings."""
        artifact = getattr(self, "_neural_concept_artifact", None)
        if artifact is None:
            raise ValueError("Qwen3.5 9B neural concept bridge is not configured")
        if payload.get("artifact_fingerprint") != artifact.fingerprint:
            raise ValueError("neural concept request artifact mismatch")
        concepts = payload.get("concepts")
        if not isinstance(concepts, list) or not 1 <= len(concepts) <= 32:
            raise ValueError("neural concept request requires 1..32 concepts")
        if not tokens or len(tokens) > int(prefill_step):
            raise ValueError(
                "neural concept bridge currently requires one bounded prefill chunk"
            )
        import mlx.core as mx

        key_state = mx.array(
            [item["key_state"] for item in concepts], dtype=mx.float32
        )
        value_state = mx.array(
            [item["value_state"] for item in concepts], dtype=mx.float32
        )
        if (
            key_state.shape != (len(concepts), artifact.state_dim)
            or value_state.shape != key_state.shape
        ):
            raise ValueError("neural concept request state geometry mismatch")
        arrays = artifact.arrays
        keys = key_state @ mx.array(arrays["key_projection"])
        values = value_state @ mx.array(arrays["value_projection"])
        keys = keys / mx.maximum(mx.linalg.norm(keys, axis=-1, keepdims=True), 1e-6)
        values = values / mx.maximum(
            mx.linalg.norm(values, axis=-1, keepdims=True), 1e-6
        )
        ids = mx.array([tokens], dtype=mx.uint32)
        embeddings = self.model.language_model.model.embed_tokens(ids)
        queries = embeddings.astype(mx.float32)
        queries = queries / mx.maximum(
            mx.linalg.norm(queries, axis=-1, keepdims=True), 1e-6
        )
        logits = mx.max(queries @ keys.T, axis=1, keepdims=True)
        weights = mx.softmax(
            logits / float(artifact.manifest["attention_temperature"]), axis=-1
        )
        gate = mx.sigmoid(mx.array(arrays["output_gate"], dtype=mx.float32))[0]
        residual = (gate * (weights @ values)).astype(embeddings.dtype)
        enhanced = mx.concatenate(
            [embeddings[:, :-1, :], embeddings[:, -1:, :] + residual], axis=1
        )
        self._neural_concept_counts["prefills"] += 1
        self._neural_concept_counts["concepts"] += len(concepts)
        self._neural_concept_counts["tokens"] += len(tokens)
        return {
            "input_embeddings": enhanced,
            "receipt": {
                "schema": "mlx2-neural-concept-prefill-v1",
                "engaged": True,
                "artifact_fingerprint": artifact.fingerprint,
                "concepts": len(concepts),
                "tokens": len(tokens),
                "bridge": "learned-recurrent-final-token-cross-attention",
                "steered_tokens": 1,
                "normalized_values": True,
                "gate": float(mx.array(gate).item()),
            },
        }

    def classifier_token_ids(self, labels):
        """Return one next-token id per label or fail closed.

        A leading-space token is preferred because classifier prompts end in a
        colon. Multi-token labels cannot participate in the forced-choice head.
        """
        result = {}
        for label in labels:
            if not isinstance(label, str) or not label:
                raise ValueError("classifier labels must be nonempty text")
            candidates = (
                list(self.tokenizer.encode(" " + label, add_special_tokens=False)),
                list(self.tokenizer.encode(label, add_special_tokens=False)),
            )
            ids = next((items for items in candidates if len(items) == 1), None)
            if ids is None:
                raise ValueError(f"classifier label {label!r} is not one token")
            result[label] = int(ids[0])
        if len(set(result.values())) != len(result):
            raise ValueError("classifier labels must map to distinct token ids")
        return result

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("Qwen3.5 9B MTP is not implemented")
        return "qwen35-9b-apcv2-ordinary"

    def cache_budget(self, *, mtp):
        if mtp:
            raise ValueError("Qwen3.5 9B MTP is not implemented")
        from .qwen38_memory import Qwen38CacheBudget

        budget = Qwen38CacheBudget.from_config(
            self.model.args.text_config, mtp=False
        )
        return _Qwen359BCacheBudget(budget)

    def diagnostics(self):
        result = {
            "architecture": "dense-hybrid-gdn-gqa",
            "layout": self.layout,
            "mtp_head_present": False,
            "scope": "text-only",
        }
        artifact = getattr(self, "_neural_concept_artifact", None)
        if artifact is not None:
            result["neural_concept_bridge"] = {
                "state": "configured-unqualified",
                "artifact_fingerprint": artifact.fingerprint,
                "counts": dict(self._neural_concept_counts),
            }
        return result


class _Qwen359BCacheBudget:
    """Naming wrapper around the shared dense Qwen3.5 geometry."""

    def __init__(self, budget):
        self._budget = budget

    def __getattr__(self, name):
        return getattr(self._budget, name)

    def as_dict(self):
        value = self._budget.as_dict()
        value["schema"] = "qwen35-9b-cache-geometry-v1"
        return value
