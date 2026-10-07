"""Sweep 2026-10-06 (capsule lane, CAP-4/5/8): the bus refuses what serving
cannot apply, at mount/bind time instead of inside the scheduler loop."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from test_capsule_bus import capsule_env  # noqa: F401  (fixture re-export)

from mlx2.runtime.activation_capsules import (
    ActivationCapsuleBus,
    ActivationCapsuleError,
    Authority,
    AuthorityScope,
    ModelBindings,
    NormGateBounds,
    PositionConvention,
    TapBinding,
    TensorPayloadStore,
)
from mlx2.runtime.activation_injection import (
    MAX_CAPSULE_STATES,
    prepare_loaded_activation_injection,
)
from mlx2.runtime.capsule_bus import (
    TOOL_RESULT_SCHEMA,
    CapsuleBusConfig,
    CapsuleBusError,
    CapsuleRequestContext,
    CapsuleRuntimeBindings,
    SemanticCapsuleBus,
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


def _thaw(value):
    if hasattr(value, "items"):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def test_default_bus_selects_one_capsule_and_bind_refuses_a_multi_selection(capsule_env):  # noqa: F811
    _first, first_result = capsule_env["publish"](name="first", tensor_seed="first")
    _second, second_result = capsule_env["publish"](name="second", tensor_seed="second")
    result = {**second_result, "handles": [first_result["handles"][0], second_result["handles"][0]]}
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"]().mount(result, capsule_env["request"])
    assert error.value.code == "count_budget_exceeded"

    mount, _ = capsule_env["bus"](max_capsules=2).mount(result, capsule_env["request"])
    for item in mount.capsules:
        with pytest.raises(CapsuleBusError) as error:
            SemanticCapsuleBus.bind_request(
                {"messages": []},
                mount,
                capsule_digest=item.digest,
                manifest=_thaw(item.manifest),
                tensors={"residual": np.ones(4, dtype=np.float32)},
                gate=0.1,
            )
        assert error.value.code == "multi_capsule_unsupported"


def test_bind_refuses_a_gate_the_adapter_would_refuse(capsule_env):  # noqa: F811
    # Fixture capsule: maximum_relative_norm=0.2, maximum_gate=0.75.
    identity, result = capsule_env["publish"]()
    mount, _ = capsule_env["bus"]().mount(result, capsule_env["request"])
    item = mount.capsules[0]
    with pytest.raises(CapsuleBusError, match="exceeds stored bounds") as error:
        SemanticCapsuleBus.bind_request(
            {},
            mount,
            capsule_digest=identity.digest,
            manifest=_thaw(item.manifest),
            tensors={"residual": np.ones(4, dtype=np.float32)},
            gate=0.5,
        )
    assert error.value.code == "invalid_gate"
    bound = SemanticCapsuleBus.bind_request(
        {},
        mount,
        capsule_digest=identity.digest,
        manifest=_thaw(item.manifest),
        tensors={"residual": np.ones(4, dtype=np.float32)},
        gate=0.2,
    )
    assert bound["_mlx2_activation_capsule"].gate == 0.2


def test_spec_refuses_a_relative_norm_the_adapter_never_loads():
    with pytest.raises(ActivationCapsuleError):
        NormGateBounds(maximum_relative_norm=2.0)
    assert NormGateBounds(maximum_relative_norm=1.0).maximum_relative_norm == 1.0


def _mount_document_capsule(tmp_path, states):
    capsules = CapsuleStore(tmp_path / "capsules")
    core = ActivationCapsuleBus(capsules, TensorPayloadStore(tmp_path / "tensors"))
    source = SemanticSource(
        source_id="document-17#decision-4",
        kind=SemanticSourceKind.DOCUMENT,
        label="Decision record 17",
        uri="project://decision/17#4",
        provenance_digest=hashlib.sha256(b"document-17").hexdigest(),
        prompt_injection="none",
    )
    model, tokenizer, runtime, projector = "m@a", "t@a", "r@a", "p@a"
    policy = CapturePolicy(
        bindings=ModelBindings(model, model, tokenizer, runtime, projector),
        tap=TapBinding(12, "post_residual", 21, "gated_cross_attention"),
        dtype="<f4",
        normalization="source-rms-v1",
        position=PositionConvention("none", "none-v1"),
        authority=Authority("tenant-alice", AuthorityScope.SESSION, "session-one", EXPIRY_NS),
        bounds=NormGateBounds(0.2, 0.0, 0.75),
        producer="document-state-capture",
        producer_revision="capture-v1",
    )
    prepared = build_document_state_capsule(
        policy,
        np.arange(states * 2, dtype=np.float32).reshape(states, 2) + 1.0,
        source=source,
        maximum_states=states,
    )
    published = SemanticActivationPublisher(core).publish(prepared)
    directory = HyperDirectory(tmp_path / "directory", capsules)
    context = DirectoryContext(model="model-scope", tenant="tenant-alice", session="session-one")
    directory.update(
        Scope.SESSION, context, expected_revision=0,
        handles={"document-recall": published.capsule_digest},
    )
    resolved = directory.resolve(context)
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
    control = SemanticCapsuleBus(
        capsules,
        directory,
        CapsuleRuntimeBindings(
            model=model, tokenizer=tokenizer, runtime=runtime, projector=projector,
            target_layer=21, target_injection="gated_cross_attention",
        ),
        config=CapsuleBusConfig(enabled=True, max_bytes=1 << 20),
        now_unix_ns=lambda: NOW_NS,
    )
    return control.mount(tool_result, CapsuleRequestContext(context, "request-one"))


def test_mount_refuses_more_states_than_the_adapter_applies(tmp_path):
    mount, _ = _mount_document_capsule(tmp_path / "ok", MAX_CAPSULE_STATES)
    assert len(mount.capsules) == 1
    with pytest.raises(CapsuleBusError) as error:
        _mount_document_capsule(tmp_path / "big", MAX_CAPSULE_STATES + 1)
    assert error.value.code == "state_budget_exceeded"


def test_loaded_capsule_is_not_bounded_by_one_prefill_chunk():
    """The memory acts on the prompt-tail forward; prefill is ordinary chunks."""
    from test_activation_injection import _bridge

    manifest = {
        "schema": "mlx2-activation-capsule-v1",
        "payload_kind": "directional_residual",
        "bindings": {
            "source_model": "qwen35-9b@a",
            "target_model": "qwen35-9b@a",
            "tokenizer": "qwen35-tokenizer@a",
            "runtime": "mlx2@a",
            "projector_revision": "projector-v1",
        },
        "tap": {"target_layer": 2, "target_injection": "qwen35-post-block-residual-v1"},
        "geometry": {"hidden_width": 2, "sequence_length": 1},
        "metadata": {"representation_space": "target_hidden"},
        "bounds": {"maximum_relative_norm": 0.5, "minimum_gate": 0.0, "maximum_gate": 0.5},
    }
    bridge = _bridge()
    manifest["bindings"].update(
        target_model=bridge.bindings["model"],
        source_model=bridge.bindings["model"],
        tokenizer=bridge.bindings["tokenizer"],
        runtime=bridge.bindings["runtime"],
        projector_revision=bridge.artifact_fingerprint,
    )
    prepared = prepare_loaded_activation_injection(
        manifest,
        {"residual": np.asarray([1.0, 0.0], dtype=np.float32)},
        bridge,
        capsule_digest="a" * 64,
        gate=0.25,
        tokens=list(range(100)),
        prefill_step=8,
    )
    assert prepared.receipt["status"] == "prepared"
