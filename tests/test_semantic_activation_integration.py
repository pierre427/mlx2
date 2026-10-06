"""Static host integration for capture, recall mounting, and immutable loading."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mlx2.runtime.activation_capsules import (
    ActivationCapsuleBus,
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
from mlx2.runtime.capsule_bus import (
    TOOL_RESULT_SCHEMA,
    CapsuleBusConfig,
    CapsuleRequestContext,
    CapsuleRuntimeBindings,
    SemanticCapsuleBus,
    is_trusted_activation_request,
)
from mlx2.runtime.hyper_directory import DirectoryContext, HyperDirectory, Scope
from mlx2.runtime.semantic_capsules import CapsuleStore
from mlx2.runtime.semantic_handoff import (
    CapturePolicy,
    SemanticActivationPublisher,
    SemanticSource,
    SemanticSourceKind,
    build_document_state_capsule,
)

NOW_NS = 1_800_000_000_000_000_000
EXPIRY_NS = NOW_NS + 60_000_000_000


def test_capture_publish_directory_tool_mount_load_and_bind_is_one_contract(tmp_path):
    capsules = CapsuleStore(tmp_path / "capsules")
    tensors = TensorPayloadStore(tmp_path / "tensors")
    core = ActivationCapsuleBus(capsules, tensors)
    source_digest = hashlib.sha256(b"document-17 canonical bytes").hexdigest()
    source = SemanticSource(
        source_id="document-17#decision-4",
        kind=SemanticSourceKind.DOCUMENT,
        label="Decision record 17",
        uri="project://decision/17#4",
        provenance_digest=source_digest,
        prompt_injection="none",
    )
    model = "qwen35@revision-a"
    tokenizer = "qwen35-tokenizer@revision-a"
    runtime = "mlx2@revision-a"
    projector = "qwen35-capsule-projector@revision-a"
    policy = CapturePolicy(
        bindings=ModelBindings(model, model, tokenizer, runtime, projector),
        tap=TapBinding(12, "post_residual", 21, "gated_cross_attention"),
        dtype="<f4",
        normalization="source-rms-v1",
        position=PositionConvention("none", "none-v1"),
        authority=Authority(
            "tenant-alice", AuthorityScope.SESSION, "session-one", EXPIRY_NS
        ),
        bounds=NormGateBounds(0.2, 0.0, 0.75),
        producer="document-state-capture",
        producer_revision="capture-v1",
    )
    prepared = build_document_state_capsule(
        policy,
        np.asarray([[1.0, -1.0], [2.0, 3.0]], dtype=np.float32),
        source=source,
        maximum_states=4,
    )
    published = SemanticActivationPublisher(core).publish(prepared)

    directory = HyperDirectory(tmp_path / "directory", capsules)
    directory_context = DirectoryContext(
        model="model-scope", tenant="tenant-alice", session="session-one"
    )
    directory.update(
        Scope.SESSION,
        directory_context,
        expected_revision=0,
        handles={"document-recall": published.capsule_digest},
    )
    resolved = directory.resolve(directory_context)
    tool_result = {
        "schema": TOOL_RESULT_SCHEMA,
        "directory_revision": resolved.revision,
        "directory_fingerprint": resolved.fingerprint,
        "handles": [
            {
                "name": "document-recall",
                "digest": published.capsule_digest,
                "scope": "session",
                "scope_id": "session-one",
                "tenant": "tenant-alice",
                "expires_at_unix_ns": EXPIRY_NS,
                "source": source.tool_metadata(),
            }
        ],
    }
    runtime_bindings = CapsuleRuntimeBindings(
        model=model,
        tokenizer=tokenizer,
        runtime=runtime,
        projector=projector,
        target_layer=21,
        target_injection="gated_cross_attention",
    )
    control = SemanticCapsuleBus(
        capsules,
        directory,
        runtime_bindings,
        config=CapsuleBusConfig(enabled=True),
        now_unix_ns=lambda: NOW_NS,
    )
    mount, selected = control.mount(
        tool_result, CapsuleRequestContext(directory_context, "request-one")
    )
    assert selected.selected and not selected.engaged and not selected.observed_used
    assert mount.capsules[0].source.as_dict() == source.tool_metadata()

    manifest, loaded = core.load(
        published.capsule_digest,
        ActivationExpectation(
            target_model=model,
            tokenizer=tokenizer,
            runtime=runtime,
            projector_revision=projector,
            target_layer=21,
            target_injection="gated_cross_attention",
            tenant="tenant-alice",
            scope=AuthorityScope.SESSION,
            scope_id="session-one",
            payload_kind=PayloadKind.CONTINUOUS_PREFIX,
        ),
        now_unix_ns=NOW_NS,
    )
    assert loaded["prefix"].flags.writeable is False
    with pytest.raises(ValueError):
        loaded["prefix"][0, 0] = 99.0

    request = SemanticCapsuleBus.bind_request(
        {"messages": [{"role": "user", "content": "recall the decision"}]},
        mount,
        capsule_digest=published.capsule_digest,
        manifest=manifest,
        tensors=loaded,
        gate=0.1,
    )
    bound = request["_mlx2_activation_capsule"]
    assert is_trusted_activation_request(bound)
    assert bound.capsule_digest == published.capsule_digest
    assert request["_mlx2_semantic_fingerprint"] == mount.semantic_fingerprint
