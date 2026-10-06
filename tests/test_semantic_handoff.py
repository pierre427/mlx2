from __future__ import annotations

import hashlib
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mlx2.runtime.activation_capsules import (
    ActivationCapsuleBus,
    ActivationCapsuleError,
    ActivationExpectation,
    Authority,
    AuthorityScope,
    ModelBindings,
    NormGateBounds,
    PayloadKind,
    PositionConvention,
    TapBinding,
    TensorPayloadStore,
)
from mlx2.runtime.semantic_capsules import CapsuleStore
from mlx2.runtime.semantic_handoff import (
    CAPTURE_RECEIPT_SCHEMA,
    HANDOFF_DESCRIPTOR_SCHEMA,
    HANDOFF_RECEIPT_SCHEMA,
    RUNTIME_CONTRACT,
    CapturePolicy,
    SemanticActivationPublisher,
    SemanticPurpose,
    SemanticSource,
    SemanticSourceKind,
    build_document_state_capsule,
    build_pruned_trajectory_capsule,
    build_subagent_handoff_capsule,
)

SOURCE_DIGEST = hashlib.sha256(b"semantic source").hexdigest()


def policy(*, expiry: int | None = None) -> CapturePolicy:
    return CapturePolicy(
        bindings=ModelBindings(
            "target@revision-2",
            "target@revision-2",
            "tokenizer@revision-3",
            "mlx2@revision-4",
            "projector@revision-5",
        ),
        tap=TapBinding(2, "post_residual", 4, "gated_cross_attention"),
        dtype="<f4",
        normalization="source-rms-v1",
        position=PositionConvention("rope", "qwen-rope-v1", 1_000_000.0),
        authority=Authority("tenant-a", AuthorityScope.SESSION, "session-a", expiry),
        bounds=NormGateBounds(0.25, 0.0, 0.5),
        producer="semantic-capture",
        producer_revision="capture-revision-1",
    )


def semantic_source(kind: str, *, uri: str, source_id: str = "source-1") -> SemanticSource:
    return SemanticSource(
        source_id=source_id,
        kind=kind,
        label=f"Test {kind}",
        uri=uri,
        provenance_digest=SOURCE_DIGEST,
        prompt_injection="none",
    )


def expectation(kind: PayloadKind) -> ActivationExpectation:
    return ActivationExpectation(
        target_model="target@revision-2",
        tokenizer="tokenizer@revision-3",
        runtime="mlx2@revision-4",
        projector_revision="projector@revision-5",
        target_layer=4,
        target_injection="gated_cross_attention",
        tenant="tenant-a",
        scope=AuthorityScope.SESSION,
        scope_id="session-a",
        payload_kind=kind,
    )


@pytest.fixture
def publisher():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        capsules = CapsuleStore(root / "capsules")
        tensors = TensorPayloadStore(root / "tensors")
        bus = ActivationCapsuleBus(capsules, tensors)
        yield SemanticActivationPublisher(bus), bus


def trajectory_states() -> np.ndarray:
    return np.array(
        [[0, 0, 0], [2, -1, 0], [2.01, -0.99, 0], [-1, -1, 4]],
        dtype=np.float32,
    )


def test_pruned_capture_preserves_order_direction_provenance_and_authority(publisher):
    semantic, bus = publisher
    prepared = build_pruned_trajectory_capsule(
        policy(),
        trajectory_states(),
        source=semantic_source(
            SemanticSourceKind.PRUNED_HIDDEN_STATE,
            uri="session://session-a/turn-8#hidden",
        ),
        maximum_transitions=2,
        maximum_rank=2,
    )
    assert prepared.spec.payload_kind is PayloadKind.ORDERED_TRAJECTORY
    assert prepared.spec.ordered_transition_indices == (1, 3)
    assert prepared.spec.metadata["source_kind"] == SemanticSourceKind.PRUNED_HIDDEN_STATE
    assert prepared.spec.metadata["purpose"] == SemanticPurpose.PRUNED_RECALL
    assert prepared.spec.provenance.source_digests == (SOURCE_DIGEST,)
    assert prepared.spec.authority.tenant == "tenant-a"
    reconstructed = prepared.tensors["coefficients"] @ prepared.tensors["basis"]
    assert reconstructed[0, 1] < 0
    assert reconstructed[1, 0] < 0

    receipt = semantic.publish(prepared)
    assert receipt.to_dict()["schema"] == CAPTURE_RECEIPT_SCHEMA
    assert receipt.selected is receipt.engaged is receipt.observed_used is False
    manifest, loaded = bus.load(receipt.capsule_digest, expectation(PayloadKind.ORDERED_TRAJECTORY))
    assert manifest["metadata"]["runtime_contract"] == RUNTIME_CONTRACT
    assert manifest["provenance"]["source_references"] == [
        "session://session-a/turn-8#hidden"
    ]
    assert set(loaded) == {"anchor", "basis", "coefficients"}


def test_document_capture_keeps_order_and_uses_continuous_prefix(publisher):
    semantic, bus = publisher
    states = np.array([[1, 2, 3], [4, 5, 6], [-1, -2, -3]], dtype=np.float32)
    prepared = build_document_state_capsule(
        policy(),
        states,
        source=semantic_source(
            SemanticSourceKind.DOCUMENT,
            uri="document://decision-17#section-4",
        ),
        maximum_states=4,
    )
    assert prepared.spec.payload_kind is PayloadKind.CONTINUOUS_PREFIX
    assert prepared.spec.metadata == {
        "source_id": "source-1",
        "source_kind": SemanticSourceKind.DOCUMENT,
        "source_label": "Test document",
        "source_uri": "document://decision-17#section-4",
        "prompt_injection": "none",
        "purpose": SemanticPurpose.DOCUMENT_RECALL,
        "representation_space": "target_hidden",
        "runtime_contract": RUNTIME_CONTRACT,
        "ordered": True,
    }
    assert not prepared.tensors["prefix"].flags.writeable
    receipt = semantic.publish(prepared)
    _manifest, loaded = bus.load(receipt.capsule_digest, expectation(PayloadKind.CONTINUOUS_PREFIX))
    assert np.array_equal(loaded["prefix"], states)


def test_document_capture_rejects_nonfinite_or_excess_states():
    with pytest.raises(ActivationCapsuleError, match="finite"):
        build_document_state_capsule(
            policy(),
            np.array([[1.0, np.nan]], dtype=np.float32),
            source=semantic_source(SemanticSourceKind.DOCUMENT, uri="document://bad"),
            maximum_states=2,
        )
    with pytest.raises(ActivationCapsuleError, match="sequence"):
        build_document_state_capsule(
            policy(),
            np.ones((3, 2), dtype=np.float32),
            source=semantic_source(SemanticSourceKind.DOCUMENT, uri="document://too-long"),
            maximum_states=2,
        )


def test_subagent_handoff_is_digest_only_and_does_not_claim_external_support(publisher):
    semantic, bus = publisher
    prepared = build_subagent_handoff_capsule(
        policy(),
        trajectory_states(),
        source=semantic_source(
            SemanticSourceKind.SUBAGENT, uri="mlx2-agent://planner/turn-3"
        ),
        source_agent="planner",
        target_agent="solver",
        maximum_transitions=3,
        maximum_rank=2,
    )
    published = semantic.publish_handoff(
        prepared, source_agent="planner", target_agent="solver"
    )
    descriptor = published.descriptor.to_dict()
    receipt = published.receipt.to_dict()
    assert descriptor["schema"] == HANDOFF_DESCRIPTOR_SCHEMA
    assert descriptor["runtime_contract"] == RUNTIME_CONTRACT
    assert descriptor["prompt_fallback"] is False
    assert "tensors" not in descriptor and "states" not in descriptor and "prompt" not in descriptor
    assert set(descriptor) == {
        "schema", "runtime_contract", "capsule_digest", "source_agent",
        "target_agent", "payload_kind", "prompt_fallback",
    }
    assert receipt["schema"] == HANDOFF_RECEIPT_SCHEMA
    assert receipt["status"] == "published"
    assert receipt["selected"] is receipt["engaged"] is receipt["observed_used"] is False
    manifest, _loaded = bus.load(
        descriptor["capsule_digest"], expectation(PayloadKind.ORDERED_TRAJECTORY)
    )
    assert manifest["metadata"]["source_kind"] == SemanticSourceKind.SUBAGENT
    assert manifest["metadata"]["purpose"] == SemanticPurpose.SUBAGENT_HANDOFF


def test_handoff_refuses_mismatched_agents_and_non_handoff_capsules(publisher):
    semantic, _bus = publisher
    handoff = build_subagent_handoff_capsule(
        policy(), trajectory_states(),
        source=semantic_source(SemanticSourceKind.SUBAGENT, uri="mlx2-agent://planner/turn-3"),
        source_agent="planner",
        target_agent="solver", maximum_transitions=2, maximum_rank=2,
    )
    with pytest.raises(ActivationCapsuleError, match="not this subagent"):
        semantic.publish_handoff(handoff, source_agent="planner", target_agent="critic")
    document = build_document_state_capsule(
        policy(), np.ones((2, 3), np.float32),
        source=semantic_source(SemanticSourceKind.DOCUMENT, uri="document://17"),
        maximum_states=2,
    )
    with pytest.raises(ActivationCapsuleError, match="not this subagent"):
        semantic.publish_handoff(document, source_agent="planner", target_agent="solver")


def test_invalid_provenance_and_expired_authority_fail_closed(publisher):
    semantic, bus = publisher
    with pytest.raises(ActivationCapsuleError, match="provenance digest"):
        bad_source = replace(
            semantic_source(SemanticSourceKind.PRUNED_HIDDEN_STATE, uri="session://a"),
            provenance_digest="not-a-digest",
        )
        build_pruned_trajectory_capsule(
            policy(), trajectory_states(), source=bad_source,
            maximum_transitions=2, maximum_rank=2,
        )
    expired_policy = policy(expiry=time.time_ns() - 1)
    prepared = build_document_state_capsule(
        expired_policy, np.ones((2, 3), np.float32),
        source=semantic_source(SemanticSourceKind.DOCUMENT, uri="document://expired"),
        maximum_states=2,
    )
    receipt = semantic.publish(prepared)
    with pytest.raises(ActivationCapsuleError, match="expired"):
        bus.load(receipt.capsule_digest, expectation(PayloadKind.CONTINUOUS_PREFIX))


def test_prepared_tensors_are_detached_and_immutable():
    states = np.ones((2, 3), dtype=np.float32)
    prepared = build_document_state_capsule(
        policy(), states,
        source=semantic_source(
            SemanticSourceKind.DOCUMENT, uri="document://immutable"
        ),
        maximum_states=2,
    )
    states[0, 0] = 99
    assert prepared.tensors["prefix"][0, 0] == 1
    with pytest.raises(ValueError):
        prepared.tensors["prefix"][0, 0] = 7


def test_capture_policy_and_agent_identifiers_are_strict():
    with pytest.raises(ActivationCapsuleError, match="producer"):
        replace(policy(), producer="")
    with pytest.raises(ActivationCapsuleError, match="source agent"):
        build_subagent_handoff_capsule(
            policy(), trajectory_states(),
            source=semantic_source(SemanticSourceKind.SUBAGENT, uri="mlx2-agent://bad"),
            source_agent="", target_agent="solver",
            maximum_transitions=2, maximum_rank=2,
        )
    with pytest.raises(ActivationCapsuleError, match="dtype"):
        replace(policy(), dtype="float32")
    with pytest.raises(ActivationCapsuleError, match="overflowed"):
        build_document_state_capsule(
            replace(policy(), dtype="<f2"),
            np.array([[1e20, 0]], dtype=np.float64),
            source=semantic_source(SemanticSourceKind.DOCUMENT, uri="document://overflow"),
            maximum_states=2,
        )
    cross_model = replace(
        policy(),
        bindings=replace(policy().bindings, source_model="other-model@revision-1"),
    )
    with pytest.raises(ActivationCapsuleError, match="identical source and target"):
        build_document_state_capsule(
            cross_model,
            np.ones((2, 3), dtype=np.float32),
            source=semantic_source(SemanticSourceKind.DOCUMENT, uri="document://cross-model"),
            maximum_states=2,
        )
