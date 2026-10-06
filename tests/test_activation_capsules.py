from __future__ import annotations

import hashlib
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mlx2.runtime.activation_capsules import (
    ActivationCapsuleBus,
    ActivationCapsuleError,
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
    compile_pruned_trajectory,
)
from mlx2.runtime.semantic_capsules import CapsuleStore

SOURCE_DIGEST = hashlib.sha256(b"source transcript").hexdigest()


def spec(kind: PayloadKind, *, sequence_length: int = 3, width: int = 4, indices=()):
    return ActivationCapsuleSpec(
        payload_kind=kind,
        bindings=ModelBindings(
            source_model="qwen-source@sha256:111",
            target_model="qwen-target@sha256:222",
            tokenizer="tokenizer@sha256:333",
            runtime="mlx2@revision-444",
            projector_revision="projector@sha256:555",
        ),
        tap=TapBinding(3, "post_residual", 5, "gated_cross_attention"),
        geometry=Geometry(
            hidden_width=width,
            sequence_length=sequence_length,
            key_width=3 if kind is PayloadKind.CROSS_ATTENTION_BANK else None,
            value_width=5 if kind is PayloadKind.CROSS_ATTENTION_BANK else None,
        ),
        dtype="<f4",
        normalization="source-rms-v1",
        position=PositionConvention("rope", "qwen-rope-v1", 1_000_000.0),
        provenance=CapsuleProvenance(
            "trajectory-compiler", "compiler-revision-1", (SOURCE_DIGEST,), ("document:42#p3",)
        ),
        authority=Authority("alice", AuthorityScope.SESSION, "session-7"),
        bounds=NormGateBounds(0.2, 0.0, 0.75),
        ordered_transition_indices=tuple(indices),
        metadata={"purpose": "test"},
    )


def expectation(kind: PayloadKind) -> ActivationExpectation:
    return ActivationExpectation(
        target_model="qwen-target@sha256:222",
        tokenizer="tokenizer@sha256:333",
        runtime="mlx2@revision-444",
        projector_revision="projector@sha256:555",
        target_layer=5,
        target_injection="gated_cross_attention",
        tenant="alice",
        scope=AuthorityScope.SESSION,
        scope_id="session-7",
        payload_kind=kind,
    )


@pytest.fixture
def stores():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        capsules = CapsuleStore(root / "capsules")
        tensors = TensorPayloadStore(root / "tensors", maximum_payload_bytes=4096)
        yield ActivationCapsuleBus(capsules, tensors, maximum_total_bytes=8192), capsules, tensors


def test_trajectory_compiler_preserves_selected_order_and_signed_direction():
    states = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, -2.0, 0.0],       # retained: signed delta [1, -2, 0]
            [1.01, -1.99, 0.0],    # pruned: low norm
            [-2.0, -1.99, 4.0],    # retained: signed delta [-3.01, 0, 4]
            [-2.0, -1.98, 4.0],    # pruned: low norm
        ],
        dtype=np.float32,
    )
    first = compile_pruned_trajectory(states, maximum_transitions=2, maximum_rank=2)
    second = compile_pruned_trajectory(states, maximum_transitions=2, maximum_rank=2)
    assert first.transition_indices == (1, 3)
    assert np.array_equal(first.anchor, states[0])
    assert np.array_equal(first.basis, second.basis)
    assert np.array_equal(first.coefficients, second.coefficients)
    assert np.allclose(first.reconstructed_deltas(), np.diff(states, axis=0)[[0, 2]], atol=2e-6)
    assert first.reconstructed_deltas()[0, 1] < 0
    assert first.reconstructed_deltas()[1, 0] < 0


def test_compiler_low_rank_bound_is_real_and_zero_trajectory_is_finite():
    states = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0], [1, 2, 3]], dtype=np.float32)
    compiled = compile_pruned_trajectory(states, maximum_transitions=3, maximum_rank=1)
    assert compiled.basis.shape == (1, 3)
    assert compiled.coefficients.shape == (3, 1)
    assert 0 < compiled.reconstruction_relative_error < 1
    zero = compile_pruned_trajectory(np.zeros((4, 3), np.float32), maximum_transitions=2, maximum_rank=1)
    assert zero.reconstruction_relative_error == 0.0
    assert np.isfinite(zero.reconstructed_deltas()).all()


@pytest.mark.parametrize(
    ("states", "message"),
    [
        (np.array([[0.0], [np.nan]], dtype=np.float32), "finite"),
        (np.zeros((1, 2), dtype=np.float32), "steps"),
        (np.zeros((2, 5), dtype=np.float32), "width"),
    ],
)
def test_compiler_fails_closed(states, message):
    with pytest.raises(ActivationCapsuleError, match=message):
        compile_pruned_trajectory(
            states, maximum_transitions=1, maximum_rank=1, maximum_width=4
        )


def test_content_addressed_tensor_round_trip_is_deterministic(stores):
    _bus, _capsules, tensors = stores
    value = np.array([[1.5, -0.0], [2.5, 4.5]], dtype=np.float32)
    first = tensors.put(value)
    second = tensors.put(value.copy())
    assert first == second
    loaded, record = tensors.get(first.digest)
    assert record == first
    assert loaded.dtype == np.dtype("<f4")
    assert np.array_equal(loaded, value)
    assert (tensors.objects / f"{first.digest}.tensor").stat().st_mode & 0o077 == 0


def test_tensor_store_rejects_nonfinite_nonfloat_and_oversized(stores):
    _bus, _capsules, tensors = stores
    with pytest.raises(ActivationCapsuleError, match="finite"):
        tensors.put(np.array([np.inf], dtype=np.float32))
    with pytest.raises(ActivationCapsuleError, match="floating point"):
        tensors.put(np.array([1], dtype=np.int32))
    with pytest.raises(ActivationCapsuleError, match="byte limit"):
        tensors.put(np.zeros(2048, dtype=np.float32))


def test_corrupt_and_symlinked_payloads_are_quarantined(stores):
    _bus, _capsules, tensors = stores
    record = tensors.put(np.array([1.0, 2.0], dtype=np.float32))
    path = tensors.objects / f"{record.digest}.tensor"
    path.write_bytes(path.read_bytes()[:-1] + b"x")
    with pytest.raises(ActivationCapsuleError, match="quarantined"):
        tensors.get(record.digest)
    assert not path.exists()
    assert list(tensors.quarantine.glob(f"{record.digest}.*.tensor"))

    other = tensors.put(np.array([3.0, 4.0], dtype=np.float32))
    other_path = tensors.objects / f"{other.digest}.tensor"
    backup = tensors.root / "real.tensor"
    os.replace(other_path, backup)
    other_path.symlink_to(backup)
    with pytest.raises(ActivationCapsuleError, match="regular non-symlink"):
        tensors.get(other.digest)
    assert not other_path.exists()
    assert backup.exists()


def test_publish_and_load_ordered_trajectory_with_all_bindings(stores):
    bus, capsules, _tensors = stores
    states = np.array([[0, 0, 0, 0], [1, -2, 0, 0], [1, -2, 3, 0]], dtype=np.float32)
    compiled = compile_pruned_trajectory(states, maximum_transitions=2, maximum_rank=2)
    capsule_spec = spec(
        PayloadKind.ORDERED_TRAJECTORY,
        sequence_length=3,
        indices=compiled.transition_indices,
    )
    identity = bus.publish(capsule_spec, compiled.tensors())
    envelope = capsules.get(identity.digest)
    assert envelope["kind"] == "activation_capsule"
    assert envelope["bindings"] == {
        "model": capsule_spec.bindings.target_model,
        "tokenizer": capsule_spec.bindings.tokenizer,
        "runtime": capsule_spec.bindings.runtime,
    }
    manifest, loaded = bus.load(identity.digest, expectation(PayloadKind.ORDERED_TRAJECTORY))
    assert manifest["exact_state_schema"] == "mlx2-apcv2-exact-state"
    assert manifest["approximation"] == "approximate_conditioning"
    assert manifest["ordered_transition_indices"] == [1, 2]
    assert set(loaded) == {"anchor", "basis", "coefficients"}


@pytest.mark.parametrize(
    ("kind", "tensors", "sequence_length"),
    [
        (PayloadKind.DIRECTIONAL_RESIDUAL, {"residual": np.ones(4, np.float32)}, 1),
        (PayloadKind.CONTINUOUS_PREFIX, {"prefix": np.ones((2, 4), np.float32)}, 2),
        (
            PayloadKind.CROSS_ATTENTION_BANK,
            {"keys": np.ones((2, 3), np.float32), "values": np.ones((2, 5), np.float32)},
            2,
        ),
    ],
)
def test_all_payload_kinds_round_trip(stores, kind, tensors, sequence_length):
    bus, _capsules, _tensor_store = stores
    identity = bus.publish(spec(kind, sequence_length=sequence_length), tensors)
    _manifest, loaded = bus.load(identity.digest, expectation(kind))
    assert set(loaded) == set(tensors)


def test_load_fails_closed_on_every_compatibility_boundary(stores):
    bus, _capsules, _tensors = stores
    kind = PayloadKind.DIRECTIONAL_RESIDUAL
    identity = bus.publish(spec(kind, sequence_length=1), {"residual": np.ones(4, np.float32)})
    baseline = expectation(kind)
    mismatches = {
        "target_model": "another-model",
        "tokenizer": "another-tokenizer",
        "runtime": "another-runtime",
        "projector_revision": "another-projector",
        "target_layer": 6,
        "target_injection": "another-injection",
        "tenant": "mallory",
        "scope": AuthorityScope.PROJECT,
        "scope_id": "another-scope",
        "payload_kind": PayloadKind.CONTINUOUS_PREFIX,
    }
    for field, value in mismatches.items():
        with pytest.raises(ActivationCapsuleError, match="mismatch"):
            bus.load(identity.digest, replace(baseline, **{field: value}))


def test_expired_authority_and_exact_state_claim_are_rejected(stores):
    bus, _capsules, _tensors = stores
    expired = replace(
        spec(PayloadKind.DIRECTIONAL_RESIDUAL, sequence_length=1),
        authority=Authority(
            "alice", AuthorityScope.SESSION, "session-7", expires_at_unix_ns=time.time_ns() - 1
        ),
    )
    identity = bus.publish(expired, {"residual": np.ones(4, np.float32)})
    with pytest.raises(ActivationCapsuleError, match="expired"):
        bus.load(identity.digest, expectation(PayloadKind.DIRECTIONAL_RESIDUAL))
    with pytest.raises(ActivationCapsuleError, match="exact state"):
        replace(expired, approximation="exact")


def test_tensor_contract_rejects_wrong_names_shapes_and_dtype(stores):
    bus, _capsules, _tensors = stores
    trajectory = spec(PayloadKind.ORDERED_TRAJECTORY, sequence_length=3, indices=(1, 2))
    valid = {
        "anchor": np.zeros(4, np.float32),
        "basis": np.zeros((2, 4), np.float32),
        "coefficients": np.zeros((2, 2), np.float32),
    }
    with pytest.raises(ActivationCapsuleError, match="exactly"):
        bus.publish(trajectory, {"anchor": valid["anchor"]})
    with pytest.raises(ActivationCapsuleError, match="geometry"):
        bus.publish(trajectory, {**valid, "anchor": np.zeros(5, np.float32)})
    with pytest.raises(ActivationCapsuleError, match="declared dtype"):
        bus.publish(trajectory, {**valid, "anchor": np.zeros(4, np.float16)})
    with pytest.raises(ActivationCapsuleError, match="geometry"):
        bus.publish(
            spec(PayloadKind.DIRECTIONAL_RESIDUAL, sequence_length=2),
            {"residual": np.zeros(4, np.float32)},
        )


def test_payload_reference_metadata_tamper_is_detected(stores):
    bus, capsules, _tensors = stores
    kind = PayloadKind.DIRECTIONAL_RESIDUAL
    identity = bus.publish(spec(kind, sequence_length=1), {"residual": np.ones(4, np.float32)})
    envelope = capsules.get(identity.digest)
    reference = envelope["data"]["payloads"]["residual"]
    # The semantic envelope itself is immutable, so exercise a valid digest that
    # resolves to tensor bytes with a different declared shape.
    reference["shape"] = [2, 2]
    replacement = capsules.put(
        kind="activation_capsule",
        data=envelope["data"],
        parents=(),
        model_binding=envelope["bindings"]["model"],
        tokenizer_binding=envelope["bindings"]["tokenizer"],
        runtime_binding=envelope["bindings"]["runtime"],
        provenance=envelope["provenance"],
    )
    with pytest.raises(ActivationCapsuleError, match="metadata mismatch"):
        bus.load(replacement.digest, expectation(kind))


def test_direct_malformed_envelope_cannot_bypass_typed_manifest(stores):
    bus, capsules, _tensors = stores
    kind = PayloadKind.DIRECTIONAL_RESIDUAL
    identity = bus.publish(spec(kind, sequence_length=1), {"residual": np.ones(4, np.float32)})
    original = capsules.get(identity.digest)
    original["data"]["activation"]["bounds"]["maximum_gate"] = 8.0
    malformed = capsules.put(
        kind="activation_capsule",
        data=original["data"],
        model_binding=original["bindings"]["model"],
        tokenizer_binding=original["bindings"]["tokenizer"],
        runtime_binding=original["bindings"]["runtime"],
        provenance=original["provenance"],
    )
    with pytest.raises(ActivationCapsuleError, match="bounds"):
        bus.load(malformed.digest, expectation(kind))


def test_total_byte_limit_is_checked_before_any_tensor_write(tmp_path):
    capsules = CapsuleStore(tmp_path / "capsules")
    tensors = TensorPayloadStore(tmp_path / "tensors", maximum_payload_bytes=4096)
    bus = ActivationCapsuleBus(capsules, tensors, maximum_total_bytes=8)
    with pytest.raises(ActivationCapsuleError, match="total byte limit"):
        bus.publish(
            spec(PayloadKind.DIRECTIONAL_RESIDUAL, sequence_length=1),
            {"residual": np.ones(4, np.float32)},
        )
    assert list(tensors.objects.iterdir()) == []


def test_tensor_root_symlink_is_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ActivationCapsuleError, match="symlink"):
        TensorPayloadStore(link)
