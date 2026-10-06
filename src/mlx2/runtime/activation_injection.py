"""Adapter-owned preparation for bounded activation-capsule injection.

This module is deliberately host-only.  It validates a revision-bound capsule
snapshot and projects its state with a configured model artifact before an
adapter converts the prepared arrays to its device representation.  Capsule
memory remains separate from chronological token KV and recurrent model state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

ACTIVATION_INJECTION_SCHEMA = "mlx2-activation-injection-v1"
ACTIVATION_CAPSULE_REVISION = "activation-capsule-v1"
MAX_CAPSULE_STATES = 32
_OPERATIONS = frozenset(
    {"directional_residual", "continuous_prefix", "cross_attention_memory"}
)
_CORE_SCHEMA = "mlx2-activation-capsule-v1"
QWEN35_TARGET_INJECTION = "qwen35-post-block-residual-v1"


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _required(value: Any, name: str) -> Any:
    result = _field(value, name)
    if result is None:
        raise ValueError(f"activation capsule is missing {name}")
    return result


def _finite_float(value: Any, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"activation capsule {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"activation capsule {name} is out of bounds")
    return result


def _states(value: Any, name: str, width: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or not 1 <= result.shape[0] <= MAX_CAPSULE_STATES:
        raise ValueError(f"activation capsule {name} must contain 1..32 states")
    if result.shape[1] != width:
        raise ValueError(f"activation capsule {name} geometry mismatch")
    if not np.isfinite(result).all():
        raise ValueError(f"activation capsule {name} must be finite")
    return result


def _hex_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"activation capsule {name} must be a lowercase sha256")
    return value


@dataclass(frozen=True, slots=True)
class ActivationInjectionBridge:
    """The adapter's immutable view of one learned capsule projector."""

    artifact_fingerprint: str
    bindings: Mapping[str, str]
    capsule_revision: str
    state_dim: int
    hidden_dim: int
    injection_layer: int
    key_projection: np.ndarray
    value_projection: np.ndarray
    direction_projection: np.ndarray


@dataclass(frozen=True, slots=True)
class PreparedActivationInjection:
    memory: Mapping[str, Any]
    receipt: Mapping[str, Any]


def bind_activation_injection_bridge(
    artifact: Any,
    *,
    hidden_dim: int,
    layer_count: int,
) -> ActivationInjectionBridge:
    """Bind an existing neural bridge artifact to the Qwen injection seam."""
    manifest = _required(artifact, "manifest")
    arrays = _required(artifact, "arrays")
    if not isinstance(manifest, Mapping) or not isinstance(arrays, Mapping):
        raise ValueError("activation bridge manifest and arrays must be mappings")
    artifact_hidden = int(_required(artifact, "hidden_dim"))
    state_dim = int(_required(artifact, "state_dim"))
    if artifact_hidden != int(hidden_dim):
        raise ValueError("activation bridge hidden dimension mismatch")
    layer = manifest.get("activation_injection_layer")
    if layer is None:
        layer = manifest.get("deep_injection_layer", (3 * int(layer_count)) // 4 - 1)
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < layer_count:
        raise ValueError("activation bridge injection layer is out of bounds")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "model",
        "tokenizer",
        "runtime",
    }:
        raise ValueError("activation bridge requires exact model/tokenizer/runtime bindings")
    bindings = {key: str(value) for key, value in bindings.items()}
    if any(not value for value in bindings.values()):
        raise ValueError("activation bridge bindings must be nonempty")
    expected = (state_dim, artifact_hidden)
    projections = {}
    for name, fallback in (
        ("key_projection", None),
        ("value_projection", None),
        ("direction_projection", "value_projection"),
    ):
        raw = arrays.get(name)
        if raw is None and fallback is not None:
            raw = arrays.get(fallback)
        projected = np.asarray(raw, dtype=np.float32)
        if projected.shape != expected or not np.isfinite(projected).all():
            raise ValueError(f"activation bridge {name} geometry mismatch")
        projections[name] = projected
    revision = manifest.get("activation_capsule_revision", ACTIVATION_CAPSULE_REVISION)
    if revision != ACTIVATION_CAPSULE_REVISION:
        raise ValueError("unsupported activation capsule revision")
    fingerprint = str(_required(artifact, "fingerprint"))
    if not fingerprint:
        raise ValueError("activation bridge fingerprint must be nonempty")
    return ActivationInjectionBridge(
        artifact_fingerprint=fingerprint,
        bindings=bindings,
        capsule_revision=revision,
        state_dim=state_dim,
        hidden_dim=artifact_hidden,
        injection_layer=layer,
        key_projection=projections["key_projection"],
        value_projection=projections["value_projection"],
        direction_projection=projections["direction_projection"],
    )


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-6)
    return (value / norms).astype(np.float32, copy=False)


def prepare_activation_injection(
    snapshot: Any,
    bridge: ActivationInjectionBridge,
    *,
    tokens: Sequence[int],
    prefill_step: int,
    route: str = "ordinary",
    batch_size: int = 1,
) -> PreparedActivationInjection:
    """Validate and compile one snapshot without constructing device state."""
    if route != "ordinary":
        raise ValueError(f"the {route} route cannot apply activation capsules")
    if batch_size != 1:
        raise ValueError("activation capsules require an isolated B=1 request")
    if not tokens or len(tokens) > int(prefill_step):
        raise ValueError("activation capsule requires one bounded prefill chunk")
    capsule_digest = _hex_digest(_required(snapshot, "capsule_digest"), "digest")
    revision = _required(snapshot, "capsule_revision")
    if revision != bridge.capsule_revision:
        raise ValueError("activation capsule revision mismatch")
    if _required(snapshot, "projector_fingerprint") != bridge.artifact_fingerprint:
        raise ValueError("activation capsule projector fingerprint mismatch")
    bindings = _required(snapshot, "bindings")
    if not isinstance(bindings, Mapping) or dict(bindings) != dict(bridge.bindings):
        raise ValueError("activation capsule model/tokenizer/runtime binding mismatch")
    if _required(snapshot, "hidden_dim") != bridge.hidden_dim:
        raise ValueError("activation capsule hidden geometry mismatch")
    if _required(snapshot, "injection_layer") != bridge.injection_layer:
        raise ValueError("activation capsule injection layer mismatch")
    operation = _required(snapshot, "operation")
    if operation not in _OPERATIONS:
        raise ValueError("unsupported activation capsule operation")
    ordered = _required(snapshot, "ordered")
    if ordered is not True:
        raise ValueError("activation capsule states must declare ordered=true")
    gate = _finite_float(_required(snapshot, "gate"), "gate", minimum=0.0, maximum=1.0)
    source = _states(_required(snapshot, "states"), "states", bridge.state_dim)
    memory: dict[str, Any] = {
        "schema": ACTIVATION_INJECTION_SCHEMA,
        "source": "activation_capsule",
        "capsule_digest": capsule_digest,
        "capsule_revision": revision,
        "projector_fingerprint": bridge.artifact_fingerprint,
        "layer": bridge.injection_layer,
        "operation": operation,
        "gate": gate,
        "ordered": True,
    }
    if operation == "directional_residual":
        signs = snapshot.get("direction_weights") if isinstance(snapshot, Mapping) else _field(snapshot, "direction_weights")
        if signs is None:
            signs = np.ones((source.shape[0],), dtype=np.float32)
        signs = np.asarray(signs, dtype=np.float32)
        if signs.shape != (source.shape[0],) or not np.isfinite(signs).all():
            raise ValueError("activation capsule direction weights mismatch")
        if np.max(np.abs(signs), initial=0.0) > 1.0 or not np.any(signs):
            raise ValueError("activation capsule direction weights are out of bounds")
        # Later deltas carry more of the trajectory endpoint.  This is
        # intentionally order-sensitive; permutation is a required control.
        ordinal = np.arange(1, source.shape[0] + 1, dtype=np.float32)
        weights = signs * ordinal
        denominator = max(float(np.sum(np.abs(weights))), 1e-6)
        direction = (source @ bridge.direction_projection) * weights[:, None]
        memory["values"] = _normalize_rows(np.sum(direction, axis=0, keepdims=True) / denominator)
    else:
        memory["keys"] = _normalize_rows(source @ bridge.key_projection)
        memory["values"] = _normalize_rows(source @ bridge.value_projection)
        # This is an external memory order bias, not a token position or RoPE
        # offset.  Reversing the trajectory changes which state receives which
        # bias without fabricating chronological KV.
        memory["order_bias"] = np.linspace(
            -0.5, 0.5, source.shape[0], dtype=np.float32
        )
        memory["temperature"] = _finite_float(
            _field(snapshot, "temperature", 0.5),
            "temperature",
            minimum=0.001,
            maximum=1.0,
        )
    decode_states = _field(snapshot, "decode_states")
    if decode_states is not None:
        if operation != "directional_residual":
            raise ValueError("scheduled capsule decode is directional-residual only")
        schedule = _states(decode_states, "decode_states", bridge.state_dim)
        memory["decode_values"] = _normalize_rows(
            schedule @ bridge.direction_projection
        )
        gates = _field(snapshot, "decode_gates")
        if gates is not None:
            if not isinstance(gates, Sequence) or len(gates) != schedule.shape[0]:
                raise ValueError("activation capsule decode gates mismatch")
            memory["decode_gates"] = tuple(
                _finite_float(value, "decode gate", minimum=0.0, maximum=1.0)
                for value in gates
            )
    decode_steps = int(memory.get("decode_values", np.empty((0,))).shape[0])
    engaged = gate > 0.0
    return PreparedActivationInjection(
        memory=memory,
        receipt={
            "schema": "mlx2-activation-capsule-injection-receipt-v1",
            "status": "prepared",
            "selected": True,
            "engaged": False,
            "observed_used": False,
            "preparation_outcome": "nonzero_gate" if engaged else "identity",
            "capsule_digest": capsule_digest,
            "capsule_revision": revision,
            "projector_fingerprint": bridge.artifact_fingerprint,
            "operation": operation,
            "ordered": True,
            "states": int(source.shape[0]),
            "injection_layer": bridge.injection_layer,
            "hidden_dim": bridge.hidden_dim,
            "relative_gate": gate,
            "decode_steps": decode_steps,
            "route": "ordinary",
            "batch_size": 1,
            "chronological_kv": False,
            "recurrent_state_edit": False,
        },
    )


def prepare_loaded_activation_injection(
    manifest: Mapping[str, Any],
    tensors: Mapping[str, Any],
    bridge: ActivationInjectionBridge,
    *,
    capsule_digest: str,
    gate: float,
    tokens: Sequence[int],
    prefill_step: int,
    route: str = "ordinary",
    batch_size: int = 1,
) -> PreparedActivationInjection:
    """Convert ``ActivationCapsuleBus.load`` output into Qwen memory.

    The bus has already checked content hashes, authority, and its load-time
    expectation.  The adapter repeats the target geometry, projector, tap,
    norm/gate, and route checks that protect model execution.
    """
    if route != "ordinary":
        raise ValueError(f"the {route} route cannot apply activation capsules")
    if batch_size != 1:
        raise ValueError("activation capsules require an isolated B=1 request")
    if not tokens or len(tokens) > int(prefill_step):
        raise ValueError("activation capsule requires one bounded prefill chunk")
    if not isinstance(manifest, Mapping) or manifest.get("schema") != _CORE_SCHEMA:
        raise ValueError("unsupported loaded activation capsule schema")
    if not isinstance(tensors, Mapping):
        raise ValueError("loaded activation capsule tensors must be a mapping")
    capsule_digest = _hex_digest(capsule_digest, "digest")
    bindings = manifest.get("bindings")
    expected_bindings = {
        "target_model": bridge.bindings["model"],
        "tokenizer": bridge.bindings["tokenizer"],
        "runtime": bridge.bindings["runtime"],
        "projector_revision": bridge.artifact_fingerprint,
    }
    if not isinstance(bindings, Mapping) or any(
        bindings.get(name) != expected for name, expected in expected_bindings.items()
    ):
        raise ValueError("loaded activation capsule model/projector binding mismatch")
    tap = manifest.get("tap")
    if (
        not isinstance(tap, Mapping)
        or tap.get("target_layer") != bridge.injection_layer
        or tap.get("target_injection") != QWEN35_TARGET_INJECTION
    ):
        raise ValueError("loaded activation capsule target tap mismatch")
    geometry = manifest.get("geometry")
    if not isinstance(geometry, Mapping) or geometry.get("hidden_width") != bridge.hidden_dim:
        raise ValueError("loaded activation capsule hidden geometry mismatch")
    bounds = manifest.get("bounds")
    if not isinstance(bounds, Mapping):
        raise ValueError("loaded activation capsule has no norm/gate bounds")
    gate = _finite_float(gate, "gate", minimum=0.0, maximum=1.0)
    minimum_gate = _finite_float(
        bounds.get("minimum_gate"), "minimum gate", minimum=0.0, maximum=1.0
    )
    maximum_gate = _finite_float(
        bounds.get("maximum_gate"), "maximum gate", minimum=0.0, maximum=1.0
    )
    maximum_norm = _finite_float(
        bounds.get("maximum_relative_norm"),
        "maximum relative norm",
        minimum=0.0,
        maximum=1.0,
    )
    if minimum_gate > maximum_gate or gate > maximum_gate or gate > maximum_norm:
        raise ValueError("loaded activation capsule gate exceeds adapter bounds")
    if gate != 0.0 and gate < minimum_gate:
        raise ValueError("loaded activation capsule gate is below adapter bounds")
    kind = manifest.get("payload_kind")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("loaded activation capsule representation metadata is missing")
    representation_space = metadata.get("representation_space")
    if kind == "cross_attention_bank":
        if representation_space != "target_projected":
            raise ValueError(
                "loaded cross-attention capsule is not target-projected"
            )
    elif (
        bindings.get("source_model") != bindings.get("target_model")
        or representation_space != "target_hidden"
    ):
        # These target-width tensors bypass the learned 96->hidden concept
        # projector. Authenticate its revision, but do not misrepresent that
        # authentication as cross-model latent translation.
        raise ValueError(
            "loaded activation capsule requires same-model target-hidden state"
        )
    operation: str
    state_count: int
    memory: dict[str, Any] = {
        "schema": ACTIVATION_INJECTION_SCHEMA,
        "source": "activation_capsule",
        "capsule_digest": capsule_digest,
        "capsule_revision": _CORE_SCHEMA,
        "projector_fingerprint": bridge.artifact_fingerprint,
        "layer": bridge.injection_layer,
        "gate": gate,
        "ordered": True,
    }
    if kind == "directional_residual":
        if set(tensors) != {"residual"}:
            raise ValueError("loaded directional capsule tensor set mismatch")
        residual = np.asarray(tensors["residual"], dtype=np.float32)
        if residual.shape != (bridge.hidden_dim,) or not np.isfinite(residual).all():
            raise ValueError("loaded directional capsule geometry mismatch")
        operation = "directional_residual"
        state_count = 1
        memory["values"] = _normalize_rows(residual[None, :])
    elif kind == "ordered_trajectory":
        if set(tensors) != {"anchor", "basis", "coefficients"}:
            raise ValueError("loaded trajectory capsule tensor set mismatch")
        anchor = np.asarray(tensors["anchor"], dtype=np.float32)
        basis = np.asarray(tensors["basis"], dtype=np.float32)
        coefficients = np.asarray(tensors["coefficients"], dtype=np.float32)
        if (
            anchor.shape != (bridge.hidden_dim,)
            or basis.ndim != 2
            or basis.shape[1] != bridge.hidden_dim
            or coefficients.ndim != 2
            or coefficients.shape[1] != basis.shape[0]
            or not 1 <= coefficients.shape[0] <= MAX_CAPSULE_STATES
            or not all(np.isfinite(value).all() for value in (anchor, basis, coefficients))
        ):
            raise ValueError("loaded trajectory capsule geometry mismatch")
        deltas = coefficients @ basis
        trajectory = anchor[None, :] + np.cumsum(deltas, axis=0)
        operation = "continuous_prefix"
        state_count = trajectory.shape[0]
        memory["keys"] = _normalize_rows(trajectory)
        memory["values"] = _normalize_rows(trajectory)
        memory["order_bias"] = np.linspace(-0.5, 0.5, state_count, dtype=np.float32)
        memory["temperature"] = 0.5
    elif kind == "continuous_prefix":
        if set(tensors) != {"prefix"}:
            raise ValueError("loaded continuous-prefix tensor set mismatch")
        prefix = np.asarray(tensors["prefix"], dtype=np.float32)
        if (
            prefix.ndim != 2
            or prefix.shape[1] != bridge.hidden_dim
            or not 1 <= prefix.shape[0] <= MAX_CAPSULE_STATES
            or not np.isfinite(prefix).all()
        ):
            raise ValueError("loaded continuous-prefix capsule geometry mismatch")
        operation = "continuous_prefix"
        state_count = prefix.shape[0]
        memory["keys"] = _normalize_rows(prefix)
        memory["values"] = _normalize_rows(prefix)
        memory["order_bias"] = np.linspace(-0.5, 0.5, state_count, dtype=np.float32)
        memory["temperature"] = 0.5
    elif kind == "cross_attention_bank":
        if set(tensors) != {"keys", "values"}:
            raise ValueError("loaded cross-attention tensor set mismatch")
        keys = np.asarray(tensors["keys"], dtype=np.float32)
        values = np.asarray(tensors["values"], dtype=np.float32)
        if (
            keys.ndim != 2
            or keys.shape != values.shape
            or keys.shape[1] != bridge.hidden_dim
            or not 1 <= keys.shape[0] <= MAX_CAPSULE_STATES
            or not np.isfinite(keys).all()
            or not np.isfinite(values).all()
        ):
            raise ValueError("loaded cross-attention capsule geometry mismatch")
        operation = "cross_attention_memory"
        state_count = keys.shape[0]
        memory["keys"] = _normalize_rows(keys)
        memory["values"] = _normalize_rows(values)
        memory["order_bias"] = np.linspace(-0.5, 0.5, state_count, dtype=np.float32)
        memory["temperature"] = 0.5
    else:
        raise ValueError("unsupported loaded activation capsule payload kind")
    memory["operation"] = operation
    engaged = gate > 0.0
    return PreparedActivationInjection(
        memory=memory,
        receipt={
            "schema": "mlx2-activation-capsule-injection-receipt-v1",
            "status": "prepared",
            "selected": True,
            "engaged": False,
            "observed_used": False,
            "preparation_outcome": "nonzero_gate" if engaged else "identity",
            "capsule_digest": capsule_digest,
            "capsule_revision": _CORE_SCHEMA,
            "projector_fingerprint": bridge.artifact_fingerprint,
            "operation": operation,
            "payload_kind": kind,
            "representation_space": representation_space,
            "ordered": True,
            "states": int(state_count),
            "injection_layer": bridge.injection_layer,
            "hidden_dim": bridge.hidden_dim,
            "relative_gate": gate,
            "maximum_relative_norm": maximum_norm,
            "decode_steps": 0,
            "route": "ordinary",
            "batch_size": 1,
            "chronological_kv": False,
            "recurrent_state_edit": False,
        },
    )


def compose_deep_concept_memory(*memories: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Compose neural-concept and activation components without overwriting either."""
    components: list[Mapping[str, Any]] = []
    for memory in memories:
        if memory is None:
            continue
        nested = memory.get("components")
        if nested is None:
            components.append(memory)
        else:
            if not isinstance(nested, (list, tuple)):
                raise ValueError("deep concept memory components must be ordered")
            components.extend(nested)
    if not components:
        raise ValueError("at least one semantic memory component is required")
    if len(components) == 1:
        return components[0]
    return {"components": tuple(components)}


def step_persistent_deep_memory(memory: Mapping[str, Any], step: int) -> Mapping[str, Any] | None:
    """Select one persistent decode step, including composed memory."""
    components = memory.get("components")
    if components is not None:
        active = [
            selected
            for component in components
            if (selected := step_persistent_deep_memory(component, step)) is not None
        ]
        if not active:
            return None
        return compose_deep_concept_memory(*active)
    schedule = memory.get("decode_values")
    if schedule is None:
        return memory
    shape = getattr(schedule, "shape", ())
    if len(shape) != 2:
        raise ValueError("concept decode_values must be a rank-2 array")
    if step >= shape[0]:
        return None
    selected = dict(memory)
    selected.pop("decode_values")
    gates = selected.pop("decode_gates", None)
    if gates is not None:
        if not isinstance(gates, (list, tuple)) or len(gates) != shape[0]:
            raise ValueError("concept decode_gates must match the capsule schedule")
        selected["gate"] = float(gates[step])
    selected["values"] = schedule[step : step + 1]
    return selected


def apply_activation_injection_numpy(
    hidden_states: np.ndarray, memory: Mapping[str, Any]
) -> np.ndarray:
    """Small host oracle for direction, order and strict zero-gate identity."""
    gate = float(memory["gate"])
    if gate == 0.0:
        return hidden_states
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("activation injection oracle requires B=1 hidden states")
    values = np.asarray(memory["values"], dtype=np.float32)
    query = hidden_states[:, -1:, :].astype(np.float32)
    query_norm = np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1e-6)
    operation = memory["operation"]
    if operation == "directional_residual":
        residual = values[None, :1, :]
    else:
        keys = np.asarray(memory["keys"], dtype=np.float32)
        normalized_query = query / query_norm
        logits = normalized_query @ keys.T / float(memory["temperature"])
        logits += np.asarray(memory["order_bias"], dtype=np.float32)[None, None, :]
        weights = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        weights /= np.sum(weights, axis=-1, keepdims=True)
        residual = weights @ values
    result = hidden_states.copy()
    result[:, -1:, :] += (gate * query_norm * residual).astype(result.dtype)
    return result
