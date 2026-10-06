"""Host-side builders for semantic activation capture and handoff.

This is an mlx2 inference-runtime contract.  It publishes approximate activation
capsules through :mod:`mlx2.runtime.activation_capsules`; it does not claim that
Codex or any other external agent transport can receive latent tensors.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .activation_capsules import (
    ActivationCapsuleBus,
    ActivationCapsuleError,
    ActivationCapsuleSpec,
    Authority,
    CapsuleProvenance,
    Geometry,
    ModelBindings,
    NormGateBounds,
    PayloadKind,
    PositionConvention,
    TapBinding,
    compile_pruned_trajectory,
)

CAPTURE_RECEIPT_SCHEMA = "mlx2-semantic-activation-capture-receipt-v1"
HANDOFF_DESCRIPTOR_SCHEMA = "mlx2-semantic-activation-handoff-v1"
HANDOFF_RECEIPT_SCHEMA = "mlx2-semantic-activation-handoff-receipt-v1"
RUNTIME_CONTRACT = "mlx2_inference_runtime"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class SemanticSourceKind(str):
    PRUNED_HIDDEN_STATE = "pruned_state"
    DOCUMENT = "document"
    SUBAGENT = "agent"
    SEMANTIC_MEMORY = "semantic_memory"


class SemanticPurpose(str):
    PRUNED_RECALL = "pruned_state_recall"
    DOCUMENT_RECALL = "document_recall"
    SUBAGENT_HANDOFF = "subagent_handoff"


_SOURCE_KINDS = frozenset(
    {
        SemanticSourceKind.PRUNED_HIDDEN_STATE,
        SemanticSourceKind.DOCUMENT,
        SemanticSourceKind.SUBAGENT,
        SemanticSourceKind.SEMANTIC_MEMORY,
    }
)
_PROMPT_INJECTION_LABELS = frozenset({"none", "suspected", "confirmed", "unknown"})


def _text(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 1024
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ActivationCapsuleError(f"{name} must be bounded nonempty text")
    return value


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ActivationCapsuleError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class CapturePolicy:
    bindings: ModelBindings
    tap: TapBinding
    dtype: str
    normalization: str
    position: PositionConvention
    authority: Authority
    bounds: NormGateBounds
    producer: str
    producer_revision: str

    def __post_init__(self) -> None:
        if self.dtype not in {"<f2", "<f4", "<f8"}:
            raise ActivationCapsuleError("capture dtype must be a canonical floating dtype")
        _text("normalization", self.normalization)
        _text("producer", self.producer)
        _text("producer_revision", self.producer_revision)


@dataclass(frozen=True, slots=True)
class SemanticSource:
    """Source metadata shared verbatim by manifest and recall tool handle."""

    source_id: str
    kind: str
    label: str
    uri: str
    provenance_digest: str
    prompt_injection: str = "none"

    def __post_init__(self) -> None:
        _text("source id", self.source_id)
        _text("source label", self.label)
        _text("source uri", self.uri)
        _digest("source provenance digest", self.provenance_digest)
        if self.kind not in _SOURCE_KINDS:
            raise ActivationCapsuleError("unsupported semantic source kind")
        if self.prompt_injection not in _PROMPT_INJECTION_LABELS:
            raise ActivationCapsuleError("unsupported prompt-injection label")

    def manifest_metadata(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "source_kind": self.kind,
            "source_label": self.label,
            "source_uri": self.uri,
            "prompt_injection": self.prompt_injection,
        }

    def tool_metadata(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "label": self.label,
            "uri": self.uri,
            "provenance_digest": self.provenance_digest,
            "prompt_injection": self.prompt_injection,
        }


@dataclass(frozen=True, slots=True)
class PreparedSemanticActivation:
    spec: ActivationCapsuleSpec
    tensors: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        frozen: dict[str, np.ndarray] = {}
        for name, value in self.tensors.items():
            array = np.ascontiguousarray(value).copy()
            array.setflags(write=False)
            frozen[name] = array
        object.__setattr__(self, "tensors", MappingProxyType(frozen))


@dataclass(frozen=True, slots=True)
class CaptureReceipt:
    capsule_digest: str
    source_kind: str
    purpose: str
    payload_kind: str
    status: str = "published"
    selected: bool = False
    engaged: bool = False
    observed_used: bool = False

    def __post_init__(self) -> None:
        _digest("capsule digest", self.capsule_digest)
        if self.status != "published" or self.selected or self.engaged or self.observed_used:
            raise ActivationCapsuleError("capture receipts describe publication only")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": CAPTURE_RECEIPT_SCHEMA,
            "capsule_digest": self.capsule_digest,
            "source_kind": self.source_kind,
            "purpose": self.purpose,
            "payload_kind": self.payload_kind,
            "status": self.status,
            "selected": False,
            "engaged": False,
            "observed_used": False,
        }


@dataclass(frozen=True, slots=True)
class HandoffDescriptor:
    capsule_digest: str
    source_agent: str
    target_agent: str
    payload_kind: str = PayloadKind.ORDERED_TRAJECTORY.value
    runtime_contract: str = RUNTIME_CONTRACT
    prompt_fallback: bool = False

    def __post_init__(self) -> None:
        _digest("capsule digest", self.capsule_digest)
        _text("source agent", self.source_agent)
        _text("target agent", self.target_agent)
        if self.payload_kind != PayloadKind.ORDERED_TRAJECTORY.value:
            raise ActivationCapsuleError("subagent handoff requires an ordered trajectory")
        if self.runtime_contract != RUNTIME_CONTRACT or self.prompt_fallback is not False:
            raise ActivationCapsuleError("handoff must remain an mlx2 latent-only runtime contract")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": HANDOFF_DESCRIPTOR_SCHEMA,
            "runtime_contract": self.runtime_contract,
            "capsule_digest": self.capsule_digest,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "payload_kind": self.payload_kind,
            "prompt_fallback": False,
        }


@dataclass(frozen=True, slots=True)
class HandoffReceipt:
    capsule_digest: str
    status: str = "published"
    selected: bool = False
    engaged: bool = False
    observed_used: bool = False

    def __post_init__(self) -> None:
        _digest("capsule digest", self.capsule_digest)
        if self.status != "published" or self.selected or self.engaged or self.observed_used:
            raise ActivationCapsuleError("handoff receipt must not claim runtime use")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": HANDOFF_RECEIPT_SCHEMA,
            "runtime_contract": RUNTIME_CONTRACT,
            "capsule_digest": self.capsule_digest,
            "status": self.status,
            "selected": False,
            "engaged": False,
            "observed_used": False,
        }


@dataclass(frozen=True, slots=True)
class PublishedHandoff:
    descriptor: HandoffDescriptor
    receipt: HandoffReceipt


def _provenance(
    policy: CapturePolicy,
    *,
    source: SemanticSource,
) -> CapsuleProvenance:
    return CapsuleProvenance(
        producer=policy.producer,
        producer_revision=policy.producer_revision,
        source_digests=(source.provenance_digest,),
        source_references=(source.uri,),
    )


def _spec(
    policy: CapturePolicy,
    *,
    kind: PayloadKind,
    geometry: Geometry,
    provenance: CapsuleProvenance,
    source: SemanticSource,
    purpose: str,
    transition_indices: tuple[int, ...] = (),
    metadata: Mapping[str, str | int | float | bool] | None = None,
) -> ActivationCapsuleSpec:
    values: dict[str, str | int | float | bool] = {
        **source.manifest_metadata(),
        "purpose": purpose,
        "representation_space": "target_hidden",
        "runtime_contract": RUNTIME_CONTRACT,
    }
    values.update(metadata or {})
    return ActivationCapsuleSpec(
        payload_kind=kind,
        bindings=policy.bindings,
        tap=policy.tap,
        geometry=geometry,
        dtype=policy.dtype,
        normalization=policy.normalization,
        position=policy.position,
        provenance=provenance,
        authority=policy.authority,
        bounds=policy.bounds,
        ordered_transition_indices=transition_indices,
        metadata=values,
    )


def _require_same_model_capture(policy: CapturePolicy) -> None:
    if policy.bindings.source_model != policy.bindings.target_model:
        raise ActivationCapsuleError(
            "raw semantic capture requires identical source and target models"
        )


def _ordered_states(value: object, policy: CapturePolicy, *, maximum_states: int) -> np.ndarray:
    if isinstance(maximum_states, bool) or not isinstance(maximum_states, int) or maximum_states <= 0:
        raise ActivationCapsuleError("maximum_states must be a positive integer")
    states = np.asarray(value)
    if states.ndim != 2 or not 1 <= states.shape[0] <= maximum_states or states.shape[1] <= 0:
        raise ActivationCapsuleError("ordered states exceed their sequence or geometry bound")
    if states.shape[1] > 65_536 or states.dtype.kind != "f" or not np.isfinite(states).all():
        raise ActivationCapsuleError("ordered states must be finite bounded floating-point tensors")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.ascontiguousarray(states, dtype=np.dtype(policy.dtype))
    if not np.isfinite(converted).all():
        raise ActivationCapsuleError("ordered states overflowed the capture dtype")
    return converted


def build_pruned_trajectory_capsule(
    policy: CapturePolicy,
    hidden_states: object,
    *,
    source: SemanticSource,
    maximum_transitions: int,
    maximum_rank: int,
) -> PreparedSemanticActivation:
    _require_same_model_capture(policy)
    if source.kind != SemanticSourceKind.PRUNED_HIDDEN_STATE:
        raise ActivationCapsuleError("pruned trajectories require a pruned_state source")
    compiled = compile_pruned_trajectory(
        hidden_states,
        maximum_transitions=maximum_transitions,
        maximum_rank=maximum_rank,
        dtype=policy.dtype,
    )
    steps, width = np.asarray(hidden_states).shape
    capsule_spec = _spec(
        policy,
        kind=PayloadKind.ORDERED_TRAJECTORY,
        geometry=Geometry(hidden_width=width, sequence_length=steps),
        provenance=_provenance(policy, source=source),
        source=source,
        purpose=SemanticPurpose.PRUNED_RECALL,
        transition_indices=compiled.transition_indices,
        metadata={
            "maximum_transitions": maximum_transitions,
            "maximum_rank": maximum_rank,
            "reconstruction_relative_error": compiled.reconstruction_relative_error,
        },
    )
    return PreparedSemanticActivation(capsule_spec, compiled.tensors())


def build_document_state_capsule(
    policy: CapturePolicy,
    ordered_states: object,
    *,
    source: SemanticSource,
    maximum_states: int,
) -> PreparedSemanticActivation:
    _require_same_model_capture(policy)
    if source.kind != SemanticSourceKind.DOCUMENT:
        raise ActivationCapsuleError("document states require a document source")
    states = _ordered_states(ordered_states, policy, maximum_states=maximum_states)
    capsule_spec = _spec(
        policy,
        kind=PayloadKind.CONTINUOUS_PREFIX,
        geometry=Geometry(hidden_width=states.shape[1], sequence_length=states.shape[0]),
        provenance=_provenance(policy, source=source),
        source=source,
        purpose=SemanticPurpose.DOCUMENT_RECALL,
        metadata={"ordered": True},
    )
    return PreparedSemanticActivation(capsule_spec, {"prefix": states})


def build_subagent_handoff_capsule(
    policy: CapturePolicy,
    hidden_states: object,
    *,
    source: SemanticSource,
    source_agent: str,
    target_agent: str,
    maximum_transitions: int,
    maximum_rank: int,
) -> PreparedSemanticActivation:
    _require_same_model_capture(policy)
    if source.kind != SemanticSourceKind.SUBAGENT:
        raise ActivationCapsuleError("subagent handoffs require an agent source")
    source_agent = _text("source agent", source_agent)
    target_agent = _text("target agent", target_agent)
    compiled = compile_pruned_trajectory(
        hidden_states,
        maximum_transitions=maximum_transitions,
        maximum_rank=maximum_rank,
        dtype=policy.dtype,
    )
    steps, width = np.asarray(hidden_states).shape
    capsule_spec = _spec(
        policy,
        kind=PayloadKind.ORDERED_TRAJECTORY,
        geometry=Geometry(hidden_width=width, sequence_length=steps),
        provenance=_provenance(policy, source=source),
        source=source,
        purpose=SemanticPurpose.SUBAGENT_HANDOFF,
        transition_indices=compiled.transition_indices,
        metadata={
            "source_agent": source_agent,
            "target_agent": target_agent,
            "maximum_transitions": maximum_transitions,
            "maximum_rank": maximum_rank,
            "reconstruction_relative_error": compiled.reconstruction_relative_error,
        },
    )
    return PreparedSemanticActivation(capsule_spec, compiled.tensors())


class SemanticActivationPublisher:
    """Publish prepared host activations without claiming inference engagement."""

    def __init__(self, bus: ActivationCapsuleBus):
        self.bus = bus

    def publish(
        self,
        prepared: PreparedSemanticActivation,
        *,
        parent_capsules: Sequence[str] = (),
    ) -> CaptureReceipt:
        metadata = prepared.spec.metadata
        if "source_kind" not in metadata or "purpose" not in metadata:
            raise ActivationCapsuleError("semantic activation metadata is incomplete")
        identity = self.bus.publish(
            prepared.spec, prepared.tensors, parent_capsules=parent_capsules
        )
        return CaptureReceipt(
            capsule_digest=identity.digest,
            source_kind=str(metadata["source_kind"]),
            purpose=str(metadata["purpose"]),
            payload_kind=prepared.spec.payload_kind.value,
        )

    def publish_handoff(
        self,
        prepared: PreparedSemanticActivation,
        *,
        source_agent: str,
        target_agent: str,
        parent_capsules: Sequence[str] = (),
    ) -> PublishedHandoff:
        if (
            prepared.spec.payload_kind is not PayloadKind.ORDERED_TRAJECTORY
            or prepared.spec.metadata.get("source_kind") != SemanticSourceKind.SUBAGENT
            or prepared.spec.metadata.get("purpose") != SemanticPurpose.SUBAGENT_HANDOFF
            or prepared.spec.metadata.get("source_agent") != source_agent
            or prepared.spec.metadata.get("target_agent") != target_agent
        ):
            raise ActivationCapsuleError("prepared activation is not this subagent handoff")
        receipt = self.publish(prepared, parent_capsules=parent_capsules)
        descriptor = HandoffDescriptor(receipt.capsule_digest, source_agent, target_agent)
        return PublishedHandoff(
            descriptor=descriptor,
            receipt=HandoffReceipt(receipt.capsule_digest),
        )


__all__ = [
    "CAPTURE_RECEIPT_SCHEMA",
    "HANDOFF_DESCRIPTOR_SCHEMA",
    "HANDOFF_RECEIPT_SCHEMA",
    "RUNTIME_CONTRACT",
    "CapturePolicy",
    "CaptureReceipt",
    "HandoffDescriptor",
    "HandoffReceipt",
    "PreparedSemanticActivation",
    "PublishedHandoff",
    "SemanticActivationPublisher",
    "SemanticPurpose",
    "SemanticSource",
    "SemanticSourceKind",
    "build_document_state_capsule",
    "build_pruned_trajectory_capsule",
    "build_subagent_handoff_capsule",
]
