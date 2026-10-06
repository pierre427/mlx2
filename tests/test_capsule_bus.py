"""Pure-host control-plane tests for the request-local semantic capsule bus."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from mlx2.runtime.activation_capsules import (
    ActivationCapsuleBus as CoreActivationCapsuleBus,
)
from mlx2.runtime.activation_capsules import (
    ActivationCapsuleSpec,
    ActivationExpectation,
    Authority,
    AuthorityScope,
    CapsuleProvenance,
    Geometry,
    ModelBindings,
    NormGateBounds,
    PayloadKind,
    PositionConvention,
    TapBinding,
    TensorPayloadStore,
)
from mlx2.runtime.capsule_bus import (
    ACTIVATION_CAPSULE_SCHEMA,
    TOOL_RESULT_SCHEMA,
    CapsuleBusCancelled,
    CapsuleBusConfig,
    CapsuleBusError,
    CapsuleRequestContext,
    CapsuleRuntimeBindings,
    SemanticCapsuleBus,
    TrustedActivationRequest,
    compose_semantic_fingerprint,
    is_trusted_activation_request,
    parse_tool_result,
)
from mlx2.runtime.hyper_directory import DirectoryContext, HyperDirectory, Scope
from mlx2.runtime.semantic_capsules import CapsuleStore, canonical_json

NOW_NS = 1_800_000_000_000_000_000
EXPIRY_NS = NOW_NS + 600_000_000_000


@pytest.fixture
def capsule_env(tmp_path):
    store = CapsuleStore(tmp_path / "capsules")
    directory = HyperDirectory(tmp_path / "directory", store)
    bindings = CapsuleRuntimeBindings(
        model="qwen35@revision-a",
        tokenizer="qwen35-tokenizer@revision-a",
        runtime="mlx2@revision-a",
        projector="qwen35-capsule-projector@revision-a",
        target_layer=21,
        target_injection="gated_cross_attention",
    )
    context = DirectoryContext(
        model="model-scope", tenant="tenant-alice", session="session-one"
    )
    request = CapsuleRequestContext(context, "request-one")

    def source(*, prompt_injection="none"):
        return {
            "source_id": "doc-17#decision-4",
            "kind": "document",
            "label": "Decision record 17",
            "uri": "project://decision/17#4",
            "provenance_digest": hashlib.sha256(b"provenance").hexdigest(),
            "prompt_injection": prompt_injection,
        }

    def publish(
        *,
        name="recall-1",
        source_metadata=None,
        scope="session",
        scope_id=None,
        tenant=None,
        expiry=EXPIRY_NS,
        payload_kind="directional_residual",
        tensor_bytes=16,
        tensor_seed="one",
        model_binding=None,
        projector_binding=None,
    ):
        source_metadata = source() if source_metadata is None else source_metadata
        tenant = context.tenant if tenant is None else tenant
        if scope_id is None:
            scope_id = context.session if scope == "session" else request.request_id
        projector_binding = (
            bindings.projector if projector_binding is None else projector_binding
        )
        tensor_digest = hashlib.sha256(tensor_seed.encode()).hexdigest()
        manifest = {
            "schema": ACTIVATION_CAPSULE_SCHEMA,
            "payload_kind": payload_kind,
            "approximation": "approximate_conditioning",
            "exact_state_schema": "mlx2-apcv2-exact-state",
            "bindings": {
                "source_model": "qwen-source@revision-a",
                "target_model": model_binding or bindings.model,
                "tokenizer": bindings.tokenizer,
                "runtime": bindings.runtime,
                "projector_revision": projector_binding,
            },
            "tap": {
                "source_layer": 12,
                "source_tap": "post_residual",
                "target_layer": bindings.target_layer,
                "target_injection": bindings.target_injection,
            },
            "geometry": {
                "hidden_width": 4,
                "sequence_length": 1,
                "key_width": None,
                "value_width": None,
            },
            "dtype": "<f4",
            "normalization": "source-rms-v1",
            "position": {"kind": "none", "revision": "none-v1", "base": None},
            "provenance": {
                "producer": "unit-test",
                "producer_revision": "test-v1",
                "source_digests": [source_metadata["provenance_digest"]],
                "source_references": [source_metadata["uri"]],
            },
            "authority": {
                "tenant": tenant,
                "scope": scope,
                "scope_id": scope_id,
                "expires_at_unix_ns": expiry,
            },
            "bounds": {
                "maximum_relative_norm": 0.2,
                "minimum_gate": 0.0,
                "maximum_gate": 0.75,
            },
            "ordered_transition_indices": [],
            "metadata": {
                "source_id": source_metadata["source_id"],
                "source_kind": source_metadata["kind"],
                "source_label": source_metadata["label"],
                "source_uri": source_metadata["uri"],
                "prompt_injection": source_metadata["prompt_injection"],
            },
        }
        identity = store.put(
            kind="activation_capsule",
            data={
                "activation": manifest,
                "payloads": {
                    "residual": {
                        "digest": tensor_digest,
                        "dtype": "<f4",
                        "shape": [4],
                        "nbytes": tensor_bytes,
                    }
                },
            },
            model_binding=model_binding or bindings.model,
            tokenizer_binding=bindings.tokenizer,
            runtime_binding=bindings.runtime,
            provenance={"source": "unit-test"},
        )
        current = directory.resolve(context).layers[-1]["revision"]
        directory.update(
            Scope.SESSION,
            context,
            expected_revision=current,
            handles={name: identity.digest},
        )
        resolved = directory.resolve(context)
        tool_result = {
            "schema": TOOL_RESULT_SCHEMA,
            "directory_revision": resolved.revision,
            "directory_fingerprint": resolved.fingerprint,
            "handles": [
                {
                    "name": name,
                    "digest": identity.digest,
                    "scope": scope,
                    "scope_id": scope_id,
                    "tenant": tenant,
                    "expires_at_unix_ns": expiry,
                    "source": source_metadata,
                }
            ],
        }
        return identity, tool_result

    def bus(**config):
        return SemanticCapsuleBus(
            store,
            directory,
            bindings,
            config=CapsuleBusConfig(enabled=True, **config),
            now_unix_ns=lambda: NOW_NS,
        )

    return locals()


def test_default_off_and_ordinary_only_fail_closed(capsule_env):
    _, tool_result = capsule_env["publish"]()
    disabled = SemanticCapsuleBus(
        capsule_env["store"],
        capsule_env["directory"],
        capsule_env["bindings"],
        now_unix_ns=lambda: NOW_NS,
    )
    with pytest.raises(CapsuleBusError) as error:
        disabled.mount(tool_result, capsule_env["request"])
    assert error.value.code == "capability_disabled"
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"]().mount(
            tool_result, replace(capsule_env["request"], route="speculative")
        )
    assert error.value.code == "ordinary_only"


def test_core_activation_envelope_mounts_without_duplicate_wrapper(capsule_env):
    identity, tool_result = capsule_env["publish"]()
    mount, selected = capsule_env["bus"]().mount(
        tool_result, capsule_env["request"], existing_semantic_fingerprint="sidecar"
    )
    item = mount.capsules[0]
    assert mount.digests == (identity.digest,)
    assert item.manifest["schema"] == ACTIVATION_CAPSULE_SCHEMA
    assert item.payload_references["residual"]["nbytes"] == 16
    assert "capsule_bus" not in item.manifest
    assert selected.as_dict()["selected"] is True
    assert selected.as_dict()["engaged"] is False
    with pytest.raises(TypeError):
        item.payload_references["residual"] = {}


def test_lifecycle_receipts_distinguish_selected_engaged_observed_used(capsule_env):
    identity, tool_result = capsule_env["publish"]()
    mount, _ = capsule_env["bus"]().mount(tool_result, capsule_env["request"])
    receipt = SemanticCapsuleBus.engagement_receipt(
        mount,
        engaged_handles=[identity.digest],
        observed_used_handles=[identity.digest],
    ).as_dict()
    assert receipt["selected"] and receipt["engaged"] and receipt["observed_used"]
    with pytest.raises(ValueError, match="not engaged"):
        SemanticCapsuleBus.engagement_receipt(
            mount, engaged_handles=[], observed_used_handles=[identity.digest]
        )


def test_tool_protocol_rejects_tensor_layer_and_gain_fields(capsule_env):
    _, result = capsule_env["publish"]()
    for field, value in (("gain", 1.0), ("injection_layer", 9), ("tensor", [[1.0]])):
        poisoned = json.loads(json.dumps(result))
        poisoned["handles"][0][field] = value
        with pytest.raises(CapsuleBusError, match="invalid shape"):
            parse_tool_result(poisoned)


def test_stale_directory_wrong_tenant_and_wrong_session_fail_closed(capsule_env):
    _, result = capsule_env["publish"]()
    resolved = capsule_env["directory"].resolve(capsule_env["context"])
    capsule_env["directory"].update(
        Scope.SESSION,
        capsule_env["context"],
        expected_revision=resolved.layers[-1]["revision"],
        policies={"changed": True},
    )
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"]().mount(result, capsule_env["request"])
    assert error.value.code == "stale_directory_revision"

    _, current = capsule_env["publish"](name="current")
    wrong = json.loads(json.dumps(current))
    wrong["handles"][0]["tenant"] = "tenant-bob"
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"]().mount(wrong, capsule_env["request"])
    assert error.value.code == "wrong_tenant"
    wrong = json.loads(json.dumps(current))
    wrong["handles"][0]["scope_id"] = "session-other"
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"]().mount(wrong, capsule_env["request"])
    assert error.value.code == "wrong_session"


def test_missing_and_corrupt_handles_have_distinct_receipts(capsule_env):
    identity, result = capsule_env["publish"]()
    path = capsule_env["store"].objects / f"{identity.digest}.json"
    original = path.read_bytes()
    path.unlink()
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"]().mount(result, capsule_env["request"])
    assert error.value.code == "missing_handle"
    path.write_bytes(original[:-2] + b"xx")
    path.chmod(0o600)
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"]().mount(result, capsule_env["request"])
    assert error.value.code == "corrupt_handle"
    assert list(capsule_env["store"].quarantine.glob(f"{identity.digest}.*.json"))


def test_cancellation_has_explicit_non_use_receipt(capsule_env):
    _, result = capsule_env["publish"]()
    with pytest.raises(CapsuleBusCancelled) as error:
        capsule_env["bus"]().mount(
            result, capsule_env["request"], cancelled=lambda: True
        )
    assert error.value.receipt(request_id="request-one") == {
        "schema": "mlx2-semantic-capsule-bus-receipt-v1",
        "status": "rejected",
        "reason": "cancelled",
        "request_id": "request-one",
        "requested_count": 0,
        "selected": False,
        "engaged": False,
        "observed_used": False,
    }


def test_prompt_injection_provenance_denied_or_preserved_never_rendered(capsule_env):
    source = capsule_env["source"](prompt_injection="suspected")
    identity, result = capsule_env["publish"](source_metadata=source)
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"]().mount(result, capsule_env["request"])
    assert error.value.code == "prompt_injection_source_denied"
    mount, receipt = capsule_env["bus"](allow_prompt_injection_sources=True).mount(
        result, capsule_env["request"]
    )
    assert receipt.prompt_injection_sources == (source["source_id"],)
    assert mount.digests == (identity.digest,)
    assert "messages" not in mount.capsules[0].manifest


def test_expiry_projector_and_model_bindings_fail_closed(capsule_env):
    _, result = capsule_env["publish"](name="expired", expiry=NOW_NS)
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"]().mount(result, capsule_env["request"])
    assert error.value.code == "expired_handle"
    _, result = capsule_env["publish"](name="projector", projector_binding="other")
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"]().mount(result, capsule_env["request"])
    assert error.value.code == "binding_mismatch"
    _, result = capsule_env["publish"](name="model", model_binding="other")
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"]().mount(result, capsule_env["request"])
    assert error.value.code == "binding_mismatch"


def test_count_and_declared_tensor_byte_budgets_are_enforced(capsule_env):
    _, result = capsule_env["publish"]()
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"](max_bytes=512).mount(result, capsule_env["request"])
    assert error.value.code == "byte_budget_exceeded"
    _, second = capsule_env["publish"](name="second", tensor_seed="two")
    combined = {**second, "handles": [result["handles"][0], second["handles"][0]]}
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"](max_capsules=1).mount(combined, capsule_env["request"])
    assert error.value.code == "count_budget_exceeded"


def test_document_scope_is_explicit_policy_and_directory_authorized(capsule_env):
    _, result = capsule_env["publish"](
        scope="document", scope_id="document-17", name="document"
    )
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"]().mount(result, capsule_env["request"])
    assert error.value.code == "scope_denied"
    document_bus = capsule_env["bus"](allowed_scopes=("session", "request", "document"))
    with pytest.raises(CapsuleBusError) as error:
        document_bus.mount(result, capsule_env["request"])
    assert error.value.code == "missing_scope_identity"
    with pytest.raises(CapsuleBusError) as error:
        document_bus.mount(
            result,
            replace(capsule_env["request"], document_id="document-other"),
        )
    assert error.value.code == "wrong_document"
    mount, _ = document_bus.mount(
        result,
        replace(capsule_env["request"], document_id="document-17"),
    )
    assert mount.capsules[0].scope_id == "document-17"


def test_project_scope_requires_authenticated_exact_project_identity(capsule_env):
    _, result = capsule_env["publish"](
        scope="project", scope_id="project-9", name="project"
    )
    project_bus = capsule_env["bus"](allowed_scopes=("session", "request", "project"))
    with pytest.raises(CapsuleBusError) as error:
        project_bus.mount(result, capsule_env["request"])
    assert error.value.code == "missing_scope_identity"
    with pytest.raises(CapsuleBusError) as error:
        project_bus.mount(
            result,
            replace(capsule_env["request"], project_id="project-other"),
        )
    assert error.value.code == "wrong_project"
    mount, _ = project_bus.mount(
        result,
        replace(capsule_env["request"], project_id="project-9"),
    )
    assert mount.capsules[0].scope_id == "project-9"


def test_fingerprints_are_deterministic_and_order_sensitive(capsule_env):
    first, first_result = capsule_env["publish"](name="first", tensor_seed="first")
    second, second_result = capsule_env["publish"](name="second", tensor_seed="second")
    result = {
        **second_result,
        "handles": [first_result["handles"][0], second_result["handles"][0]],
    }
    bus = capsule_env["bus"]()
    a, _ = bus.mount(
        result, capsule_env["request"], existing_semantic_fingerprint="sidecar"
    )
    b, _ = bus.mount(
        json.loads(json.dumps(result)),
        capsule_env["request"],
        existing_semantic_fingerprint="sidecar",
    )
    assert a.snapshot_fingerprint == b.snapshot_fingerprint
    assert a.semantic_fingerprint == compose_semantic_fingerprint(
        "sidecar", a.snapshot_fingerprint
    )
    reversed_mount, _ = bus.mount(
        {**result, "handles": list(reversed(result["handles"]))},
        capsule_env["request"],
        existing_semantic_fingerprint="sidecar",
    )
    assert reversed_mount.snapshot_fingerprint != a.snapshot_fingerprint
    assert set(reversed_mount.digests) == {first.digest, second.digest}
    first_item = a.capsules[0]

    def thaw(value):
        if hasattr(value, "items"):
            return {key: thaw(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [thaw(item) for item in value]
        return value

    bound = SemanticCapsuleBus.bind_request(
        {"messages": []},
        a,
        capsule_digest=first_item.digest,
        manifest=thaw(first_item.manifest),
        tensors={"residual": np.ones(4, dtype=np.float32)},
        gate=0.5,
    )
    assert len(a.digests) == 2
    assert bound["_mlx2_activation_capsule"].capsule_digest == first.digest
    lifecycle = SemanticCapsuleBus.engagement_receipt(
        a,
        engaged_handles=[first.digest],
        observed_used_handles=[first.digest],
    )
    assert len(lifecycle.selected_handles) == 2
    assert lifecycle.engaged_handles == (first.digest,)


def test_budget_counts_envelope_plus_tensor_references(capsule_env):
    identity, result = capsule_env["publish"]()
    mount, _ = capsule_env["bus"]().mount(result, capsule_env["request"])
    stored = capsule_env["store"].get(identity.digest)
    assert mount.total_bytes == len(canonical_json(stored)) + 16


@pytest.mark.parametrize(
    "mutation",
    [
        "exact_marker",
        "extra_manifest_field",
        "payload_name",
        "payload_dtype",
        "payload_shape",
        "payload_nbytes",
    ],
)
def test_mount_strictly_validates_manifest_and_tensor_reference_contract(
    capsule_env, mutation
):
    identity, result = capsule_env["publish"](name=f"base-{mutation}")
    envelope = capsule_env["store"].get(identity.digest)
    data = json.loads(json.dumps(envelope["data"]))
    if mutation == "exact_marker":
        data["activation"]["exact_state_schema"] = "forged-exact-state"
    elif mutation == "extra_manifest_field":
        data["activation"]["model_selected_gain"] = 1.0
    elif mutation == "payload_name":
        data["payloads"]["other"] = data["payloads"].pop("residual")
    elif mutation == "payload_dtype":
        data["payloads"]["residual"]["dtype"] = "<f2"
    elif mutation == "payload_shape":
        data["payloads"]["residual"]["shape"] = [5]
    else:
        data["payloads"]["residual"]["nbytes"] = 20
    replacement = capsule_env["store"].put(
        kind="activation_capsule",
        data=data,
        model_binding=envelope["bindings"]["model"],
        tokenizer_binding=envelope["bindings"]["tokenizer"],
        runtime_binding=envelope["bindings"]["runtime"],
        provenance=envelope["provenance"],
    )
    current = capsule_env["directory"].resolve(capsule_env["context"])
    name = f"tampered-{mutation}"
    capsule_env["directory"].update(
        Scope.SESSION,
        capsule_env["context"],
        expected_revision=current.layers[-1]["revision"],
        handles={name: replacement.digest},
    )
    resolved = capsule_env["directory"].resolve(capsule_env["context"])
    handle = json.loads(json.dumps(result["handles"][0]))
    handle.update(name=name, digest=replacement.digest)
    tampered_result = {
        "schema": TOOL_RESULT_SCHEMA,
        "directory_revision": resolved.revision,
        "directory_fingerprint": resolved.fingerprint,
        "handles": [handle],
    }
    with pytest.raises(CapsuleBusError) as error:
        capsule_env["bus"]().mount(tampered_result, capsule_env["request"])
    assert error.value.code == "invalid_capsule"


def test_real_activation_store_envelope_mounts_and_loads_without_translation(tmp_path):
    capsules = CapsuleStore(tmp_path / "capsules")
    tensors = TensorPayloadStore(tmp_path / "tensors")
    activation = CoreActivationCapsuleBus(capsules, tensors)
    source_digest = hashlib.sha256(b"source-document").hexdigest()
    source_uri = "project://decision/17#4"
    spec = ActivationCapsuleSpec(
        payload_kind=PayloadKind.DIRECTIONAL_RESIDUAL,
        bindings=ModelBindings(
            source_model="qwen-source@revision-a",
            target_model="qwen35@revision-a",
            tokenizer="qwen35-tokenizer@revision-a",
            runtime="mlx2@revision-a",
            projector_revision="qwen35-capsule-projector@revision-a",
        ),
        tap=TapBinding(12, "post_residual", 21, "gated_cross_attention"),
        geometry=Geometry(hidden_width=4, sequence_length=1),
        dtype="<f4",
        normalization="source-rms-v1",
        position=PositionConvention("none", "none-v1"),
        provenance=CapsuleProvenance(
            "unit-test", "test-v1", (source_digest,), (source_uri,)
        ),
        authority=Authority(
            "tenant-alice", AuthorityScope.SESSION, "session-one", EXPIRY_NS
        ),
        bounds=NormGateBounds(0.2, 0.25, 0.75),
        metadata={
            "source_id": "doc-17#decision-4",
            "source_kind": "document",
            "source_label": "Decision record 17",
            "source_uri": source_uri,
            "prompt_injection": "none",
        },
    )
    identity = activation.publish(spec, {"residual": np.ones(4, dtype=np.float32)})
    directory = HyperDirectory(tmp_path / "directory", capsules)
    directory_context = DirectoryContext(
        model="model-scope", tenant="tenant-alice", session="session-one"
    )
    directory.update(
        Scope.SESSION,
        directory_context,
        expected_revision=0,
        handles={"real-core": identity.digest},
    )
    resolved = directory.resolve(directory_context)
    source = {
        "source_id": "doc-17#decision-4",
        "kind": "document",
        "label": "Decision record 17",
        "uri": source_uri,
        "provenance_digest": source_digest,
        "prompt_injection": "none",
    }
    tool_result = {
        "schema": TOOL_RESULT_SCHEMA,
        "directory_revision": resolved.revision,
        "directory_fingerprint": resolved.fingerprint,
        "handles": [
            {
                "name": "real-core",
                "digest": identity.digest,
                "scope": "session",
                "scope_id": "session-one",
                "tenant": "tenant-alice",
                "expires_at_unix_ns": EXPIRY_NS,
                "source": source,
            }
        ],
    }
    bindings = CapsuleRuntimeBindings(
        model="qwen35@revision-a",
        tokenizer="qwen35-tokenizer@revision-a",
        runtime="mlx2@revision-a",
        projector="qwen35-capsule-projector@revision-a",
        target_layer=21,
        target_injection="gated_cross_attention",
    )
    bus = SemanticCapsuleBus(
        capsules,
        directory,
        bindings,
        config=CapsuleBusConfig(enabled=True),
        now_unix_ns=lambda: NOW_NS,
    )
    mount, _ = bus.mount(
        tool_result, CapsuleRequestContext(directory_context, "request-one")
    )
    tensor_digest = mount.capsules[0].payload_references["residual"]["digest"]
    assert tensor_digest == next(iter((tensors.objects).glob("*.tensor"))).stem

    expectation = ActivationExpectation(
        target_model=bindings.model,
        tokenizer=bindings.tokenizer,
        runtime=bindings.runtime,
        projector_revision=bindings.projector,
        target_layer=bindings.target_layer,
        target_injection=bindings.target_injection,
        tenant="tenant-alice",
        scope=AuthorityScope.SESSION,
        scope_id="session-one",
        payload_kind=PayloadKind.DIRECTIONAL_RESIDUAL,
    )
    loaded_manifest, loaded = activation.load(
        identity.digest, expectation, now_unix_ns=NOW_NS
    )

    def thaw(value):
        if hasattr(value, "items"):
            return {key: thaw(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [thaw(item) for item in value]
        return value

    assert loaded_manifest == thaw(mount.capsules[0].manifest)
    assert np.array_equal(loaded["residual"], np.ones(4, dtype=np.float32))
    request = {"messages": [{"role": "user", "content": "question"}]}
    prepared = SemanticCapsuleBus.bind_request(
        request,
        mount,
        capsule_digest=identity.digest,
        manifest=loaded_manifest,
        tensors=loaded,
        gate=0.5,
    )
    assert prepared["messages"] == request["messages"]
    assert prepared["_mlx2_semantic_fingerprint"] == mount.semantic_fingerprint
    payload = prepared["_mlx2_activation_capsule"]
    assert isinstance(payload, TrustedActivationRequest)
    assert is_trusted_activation_request(payload)
    assert payload.capsule_digest == identity.digest
    assert payload.semantic_fingerprint == mount.semantic_fingerprint
    assert not is_trusted_activation_request(
        {
            "capsule_digest": identity.digest,
            "manifest": loaded_manifest,
            "tensors": loaded,
            "gate": 0.5,
            "semantic_fingerprint": mount.semantic_fingerprint,
        }
    )
    forged = TrustedActivationRequest(
        identity.digest,
        loaded_manifest,
        loaded,
        0.5,
        mount.semantic_fingerprint,
        object(),
    )
    assert not is_trusted_activation_request(forged)
    zero = SemanticCapsuleBus.bind_request(
        request,
        mount,
        capsule_digest=identity.digest,
        manifest=loaded_manifest,
        tensors=loaded,
        gate=0.0,
    )
    assert zero["_mlx2_activation_capsule"].gate == 0.0
    with pytest.raises(CapsuleBusError) as error:
        SemanticCapsuleBus.bind_request(
            request,
            mount,
            capsule_digest=identity.digest,
            manifest=loaded_manifest,
            tensors=loaded,
            gate=0.1,
        )
    assert error.value.code == "invalid_gate"
    with pytest.raises(CapsuleBusError) as error:
        SemanticCapsuleBus.bind_request(
            request,
            mount,
            capsule_digest=identity.digest,
            manifest=loaded_manifest,
            tensors=loaded,
            gate=0.9,
        )
    assert error.value.code == "invalid_gate"
