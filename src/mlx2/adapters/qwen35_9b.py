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
from ..process_env import (
    PROCESS_NUMERICS,
    clear_inherited_profile,
    require_process_numerics,
)
from ..runtime.activation_injection import (
    bind_activation_injection_bridge,
    compose_deep_concept_memory,
    prepare_activation_injection,
    prepare_loaded_activation_injection,
)
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
    # Refuse an explicit TF32 value before the profile overwrites it
    # (sweep 1002 review item 6).
    require_process_numerics("the Qwen3.5 9B profile")
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        **PROCESS_NUMERICS, "MLX_GDN_PACKED": "1",
        "MLX_GDN_CORE": "0",
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_LM_SEGMENTED_SELF_MTP": "0",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "0",
        "MLX_LM_SHARED_QSA_SUFFIX": "0",
        "MLX_LM_MTP_BOUNDARY_COW": "0",
    }
    clear_inherited_profile(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_"))
    os.environ.update(profile)
    return profile


def inspect_artifact(
    model_path: str | Path,
    *,
    expected_topology: Mapping[str, int] | None = None,
    family: str = "9B",
) -> dict:
    """Validate one dense Qwen3.5 topology without importing MLX."""
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    text = config.get("text_config", config)
    expected = dict(expected_topology) if expected_topology is not None else {
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
        "linear_conv_kernel_dim": 4,
    }
    if config.get("model_type") != "qwen3_5" or text.get("num_experts", 0):
        raise ValueError(f"Qwen3.5 {family} requires the dense qwen3_5 artifact layout")
    if any(text.get(key) != value for key, value in expected.items()):
        raise ValueError(f"artifact topology does not match Qwen3.5 {family}")
    expected_layers = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(expected["num_hidden_layers"])
    ]
    if text.get("layer_types") not in (None, expected_layers):
        raise ValueError(f"artifact layer order does not match Qwen3.5 {family}")

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
    fused_gdn_architecture = "qwen35"
    # The parent 27B decode-first default has only been measured on 27B.
    default_route_execution_policy = {}
    descriptor = QWEN35_9B
    artifact_inspector = staticmethod(inspect_artifact)
    descriptor_builder = staticmethod(descriptor_for)
    environment_configurator = staticmethod(configure_environment)
    from .qwen import QWEN35_9B_SAMPLING as sampling_defaults

    def prefill_step_default(self):
        """No 27B prefill preference inheritance without 9B evidence."""
        return None

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
        layer_count = len(self.model.language_model.model.layers)
        injection_layer = int(
            artifact.manifest.get("deep_injection_layer", (3 * layer_count) // 4 - 1)
        )
        if not 0 <= injection_layer < layer_count:
            raise ValueError("neural concept bridge injection layer is out of bounds")
        self._neural_concept_artifact = artifact
        self._neural_concept_injection_layer = injection_layer
        self._neural_concept_counts = {"prefills": 0, "concepts": 0, "tokens": 0}

    def configure_activation_capsule_bridge(self, artifact=None):
        """Bind activation injection to a pinned learned projector.

        The neural-concept artifact may be shared deliberately: both routes
        then use the same bound projection geometry while retaining distinct
        capsule payloads and receipts.  This does not authorize another model,
        route, layer, or artifact revision.
        """
        if artifact is None:
            artifact = getattr(self, "_neural_concept_artifact", None)
        if artifact is None:
            raise ValueError("activation capsule bridge requires a configured artifact")
        text_config = self.model.args.text_config
        hidden_size = (
            text_config["hidden_size"]
            if isinstance(text_config, Mapping)
            else text_config.hidden_size
        )
        layer_count = len(self.model.language_model.model.layers)
        self._activation_capsule_bridge = bind_activation_injection_bridge(
            artifact, hidden_dim=int(hidden_size), layer_count=layer_count
        )
        self._activation_capsule_counts = {
            "requests": 0,
            "prepared": 0,
            "states": 0,
            "decode_steps": 0,
        }

    def activation_capsule_prefill(
        self,
        tokens,
        snapshot,
        *,
        prefill_step,
        route="ordinary",
        batch_size=1,
    ):
        """Prepare one ordinary-B1 activation snapshot for model input."""
        bridge = getattr(self, "_activation_capsule_bridge", None)
        if bridge is None:
            raise ValueError("Qwen3.5 9B activation capsule bridge is not configured")
        if isinstance(snapshot, Mapping) and {
            "manifest",
            "tensors",
            "capsule_digest",
            "gate",
        } <= set(snapshot):
            prepared = prepare_loaded_activation_injection(
                snapshot["manifest"],
                snapshot["tensors"],
                bridge,
                capsule_digest=snapshot["capsule_digest"],
                gate=snapshot["gate"],
                tokens=tokens,
                prefill_step=prefill_step,
                route=route,
                batch_size=batch_size,
            )
        else:
            prepared = prepare_activation_injection(
                snapshot,
                bridge,
                tokens=tokens,
                prefill_step=prefill_step,
                route=route,
                batch_size=batch_size,
            )
        import mlx.core as mx

        memory = dict(prepared.memory)
        for name in ("keys", "values", "order_bias", "decode_values"):
            value = memory.get(name)
            if value is not None:
                memory[name] = mx.array(value, dtype=mx.float32)
        result = {"deep_concept_memory": memory, "receipt": dict(prepared.receipt)}
        if memory.get("decode_values") is not None:
            result["_mlx2_persistent_decode_inputs"] = {
                "deep_concept_memory": memory
            }
        counts = self._activation_capsule_counts
        counts["requests"] += 1
        # Preparation is not execution. Serving owns engagement/observed-use
        # and requires evaluated model-forward evidence for the exact UID.
        counts["prepared"] += 1
        counts["states"] += int(prepared.receipt["states"])
        counts["decode_steps"] += int(prepared.receipt["decode_steps"])
        return result

    @staticmethod
    def compose_semantic_prefill_inputs(*prepared_inputs):
        """Compose neural concepts with activation capsules in source order."""
        memories = []
        persistent = False
        receipts = []
        for prepared in prepared_inputs:
            if prepared is None:
                continue
            memory = prepared.get("deep_concept_memory")
            if memory is None:
                raise ValueError("semantic prefill input has no deep concept memory")
            memories.append(memory)
            persistent = persistent or prepared.get(
                "_mlx2_persistent_decode_inputs"
            ) is not None
            receipt = prepared.get("receipt")
            if receipt is not None:
                receipts.append(receipt)
        memory = compose_deep_concept_memory(*memories)
        result = {
            "deep_concept_memory": memory,
            "receipts": tuple(receipts),
        }
        if persistent:
            result["_mlx2_persistent_decode_inputs"] = {
                "deep_concept_memory": memory
            }
        return result

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
        candidate_concepts = len(concepts)
        selection = artifact.manifest.get("deep_selection", "latent_attention")
        if selection == "directory_top1":
            concepts = concepts[:1]
        elif selection != "latent_attention":
            raise ValueError("unsupported neural concept selection policy")
        if not tokens:
            raise ValueError("neural concept bridge requires a nonempty prompt")
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
        gate = mx.sigmoid(mx.array(arrays["output_gate"], dtype=mx.float32))[0]
        gate_value = float(mx.array(gate).item())
        memory = {
            "keys": keys,
            "values": values,
            "layer": self._neural_concept_injection_layer,
            "temperature": float(artifact.manifest["attention_temperature"]),
            "gate": gate_value,
        }
        decode_projection = arrays.get("decode_value_projections")
        persistent = None
        if decode_projection is not None:
            if (
                decode_projection.ndim != 3
                or decode_projection.shape[1:]
                != (artifact.state_dim, artifact.hidden_dim)
            ):
                raise ValueError("concept decode projection geometry mismatch")
            decode_length = concepts[0].get("decode_length")
            if (
                isinstance(decode_length, bool)
                or not isinstance(decode_length, int)
                or not 1 <= decode_length <= decode_projection.shape[0]
            ):
                raise ValueError(
                    "capsule concept requires a bounded decode_length"
                )
            decode_values = mx.stack(
                [
                    value_state[0] @ mx.array(projection)
                    for projection in decode_projection[:decode_length]
                ]
            )
            decode_values = decode_values / mx.maximum(
                mx.linalg.norm(decode_values, axis=-1, keepdims=True), 1e-6
            )
            memory["decode_values"] = decode_values
            persistent = {"deep_concept_memory": memory}
        self._neural_concept_counts["prefills"] += 1
        self._neural_concept_counts["concepts"] += len(concepts)
        self._neural_concept_counts["tokens"] += len(tokens)
        result = {
            "deep_concept_memory": memory,
            "receipt": {
                "schema": "mlx2-neural-concept-prefill-v2",
                "engaged": True,
                "artifact_fingerprint": artifact.fingerprint,
                "concepts": len(concepts),
                "candidate_concepts": candidate_concepts,
                "selection": selection,
                "tokens": len(tokens),
                "bridge": "learned-recurrent-deep-final-token-cross-attention",
                "injection_layer": self._neural_concept_injection_layer,
                # Serving applies the memory on the prompt-tail forward
                # (final prompt row); a schedule starts at that same step.
                "steered_tokens": (
                    "final-prompt-row-plus-capsule-schedule"
                    if persistent is not None
                    else 1
                ),
                "decode_policy": (
                    "capsule-scheduled-isolated-b1"
                    if persistent is not None
                    else "one-shot-final-prompt-row"
                ),
                "decode_steps": (
                    int(memory["decode_values"].shape[0])
                    if persistent is not None
                    else 0
                ),
                "normalized_values": True,
                "relative_gate": gate_value,
            },
        }
        if persistent is not None:
            # The runtime consumes this reserved key before calling the model.
            result["_mlx2_persistent_decode_inputs"] = persistent
        return result

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
        return _Qwen35CacheBudget(_dense_qwen35_budget(self), family="9b")

    def diagnostics(self):
        # The inherited 27B receipts: each mechanism the shared constructor
        # installs (fused GDN, fp16 GDN state, invariant prefill, ...) reports
        # itself only while selected, so the qualifier can observe it here.
        # The 27B's always-present speculation fields do not apply (no MTP
        # or draft route), so the default receipt is unchanged.
        result = {
            key: value
            for key, value in super().diagnostics().items()
            if key not in ("speculation", "segmented_mtp", "norm_convention")
        }
        result.update({
            "architecture": "dense-hybrid-gdn-gqa",
            "layout": self.layout,
            "mtp_head_present": False,
            "scope": "text-only",
        })
        artifact = getattr(self, "_neural_concept_artifact", None)
        if artifact is not None:
            result["neural_concept_bridge"] = {
                "state": "deep-bridge-implemented-unqualified",
                "artifact_fingerprint": artifact.fingerprint,
                "injection_layer": self._neural_concept_injection_layer,
                "counts": dict(self._neural_concept_counts),
            }
        bridge = getattr(self, "_activation_capsule_bridge", None)
        if bridge is not None:
            result["activation_capsule_bridge"] = {
                "state": "implemented-unqualified-default-off",
                "projector_fingerprint": bridge.artifact_fingerprint,
                "capsule_revision": bridge.capsule_revision,
                "injection_layer": bridge.injection_layer,
                "hidden_dim": bridge.hidden_dim,
                "ordinary_b1_only": True,
                "speculative_routes": "refused",
                "counts": dict(self._activation_capsule_counts),
            }
        return result


# The Qwen3.8 27B workspace measurements (qwen38_memory: 3.1 GiB per lane,
# 2.0 GiB per 2048-row prefill chunk) were taken at hidden 5120 /
# intermediate 17408.  No 4B/9B measurement exists yet, so they are scaled by
# the larger of the two width ratios (the forward's per-row activations and
# MLP intermediates are what the workspace holds), never above the measured
# 27B values.  Replace with a scripts/measure_lane_transient.py measurement
# (provenance/lane-transient-moe.json method) when one is taken on GPU.
_DENSE_27B_WIDTHS = {"hidden_size": 5120, "intermediate_size": 17408}


def _dense_qwen35_budget(adapter):
    """The shared dense hybrid cache bound at this model's own geometry."""
    from dataclasses import replace

    from .flash_next import gdn_state_bytes
    from .qwen38_memory import Qwen38CacheBudget

    text = adapter.model.args.text_config
    budget = Qwen38CacheBudget.from_config(
        text, mtp=False, recurrent_state_bytes=gdn_state_bytes(adapter)
    )
    # A width the config omits is charged at the 27B value.
    scale = min(1.0, max(
        int(text.get(name, width)) / width for name, width in _DENSE_27B_WIDTHS.items()
    ))
    return replace(
        budget,
        transient_gib_per_lane=round(budget.transient_gib_per_lane * scale, 2),
        prefill_chunk_transient_gib=round(budget.prefill_chunk_transient_gib * scale, 2),
    )


class _Qwen35CacheBudget:
    """Family-specific receipt schema around shared dense Qwen3.5 geometry."""

    def __init__(self, budget, *, family):
        self._budget = budget
        self._family = family

    def __getattr__(self, name):
        return getattr(self._budget, name)

    def as_dict(self):
        value = self._budget.as_dict()
        value["schema"] = f"qwen35-{self._family}-cache-geometry-v1"
        value["workspace_basis"] = "qwen38-27b-measurement-scaled-by-width-unmeasured"
        return value
