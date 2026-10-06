from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pytest

from mlx2.runtime.activation_capsules import (
    ActivationCapsuleBus,
    ActivationCapsuleError,
    ActivationCapsuleSpec,
    ApproximationClass,
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
from mlx2.runtime.semantic_capsules import CapsuleIntegrityError, CapsuleStore


def _spec() -> ActivationCapsuleSpec:
    source_digest = hashlib.sha256(b"storage-hardening-source").hexdigest()
    return ActivationCapsuleSpec(
        payload_kind=PayloadKind.DIRECTIONAL_RESIDUAL,
        approximation=ApproximationClass.APPROXIMATE_CONDITIONING,
        bindings=ModelBindings(
            source_model="qwen@sha256:model",
            target_model="qwen@sha256:model",
            tokenizer="tokenizer@sha256:tokenizer",
            runtime="mlx2@storage-hardening",
            projector_revision="identity@storage-hardening",
        ),
        tap=TapBinding(2, "post_residual", 4, "post_block_residual"),
        geometry=Geometry(hidden_width=4, sequence_length=1),
        dtype="<f4",
        normalization="source-rms-v1",
        position=PositionConvention("none", "none-v1"),
        provenance=CapsuleProvenance(
            producer="storage-hardening-test",
            producer_revision="test-v1",
            source_digests=(source_digest,),
        ),
        authority=Authority("tenant-a", AuthorityScope.SESSION, "session-a"),
        bounds=NormGateBounds(0.25),
        metadata={"purpose": "storage-hardening"},
    )


def _bus(tmp_path):
    capsules = CapsuleStore(tmp_path / "capsules")
    tensors = TensorPayloadStore(tmp_path / "tensors", maximum_payload_bytes=4096)
    return ActivationCapsuleBus(capsules, tensors, maximum_total_bytes=8192)


def test_capsule_store_default_limit_preserves_normal_round_trip(tmp_path):
    store = CapsuleStore(tmp_path / "capsules")
    identity = store.put(
        kind="policy",
        data={"policy": "bounded"},
        provenance={"producer": "test"},
    )
    assert store.get(identity.digest)["data"] == {"policy": "bounded"}


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_capsule_store_rejects_invalid_envelope_limit(tmp_path, limit):
    with pytest.raises(CapsuleIntegrityError, match="positive integer"):
        CapsuleStore(tmp_path / str(limit), maximum_envelope_bytes=limit)


def test_capsule_store_refuses_oversized_write_before_object_creation(tmp_path):
    store = CapsuleStore(tmp_path / "capsules", maximum_envelope_bytes=256)
    with pytest.raises(CapsuleIntegrityError, match="byte limit"):
        store.put(
            kind="policy",
            data={"blob": "x" * 1024},
            provenance={"producer": "test"},
        )
    assert list(store.objects.iterdir()) == []


def test_capsule_store_quarantines_oversized_file_before_decode(tmp_path):
    root = tmp_path / "capsules"
    writer = CapsuleStore(root, maximum_envelope_bytes=4096)
    identity = writer.put(
        kind="policy",
        data={"blob": "x" * 1024},
        provenance={"producer": "test"},
    )
    path = writer.objects / f"{identity.digest}.json"
    assert path.stat().st_size > 512

    reader = CapsuleStore(root, maximum_envelope_bytes=512)
    with pytest.raises(CapsuleIntegrityError, match="quarantined oversized"):
        reader.get(identity.digest)
    assert not path.exists()
    assert list(reader.quarantine.glob(f"{identity.digest}.*.json"))


def test_tensor_gc_is_dry_run_by_default_and_deletes_only_old_orphans(tmp_path):
    bus = _bus(tmp_path)
    identity = bus.publish(_spec(), {"residual": np.ones(4, dtype=np.float32)})
    referenced_digest = bus.capsules.get(identity.digest)["data"]["payloads"][
        "residual"
    ]["digest"]
    old_orphan = bus.tensors.put(np.full(4, 2.0, dtype=np.float32))
    recent_orphan = bus.tensors.put(np.full(4, 3.0, dtype=np.float32))
    now = time.time_ns()
    old_time = now - 10_000_000_000
    os.utime(bus.tensors._path(old_orphan.digest), ns=(old_time, old_time))
    cutoff = now - 1_000_000_000

    dry = bus.collect_orphaned_tensors(retention_cutoff_unix_ns=cutoff)
    assert dry.dry_run is True
    assert dry.candidates == (old_orphan.digest,)
    assert dry.deleted == ()
    assert dry.referenced_tensor_count == 1
    assert dry.retained_recent_count == 1
    assert bus.tensors._path(old_orphan.digest).exists()

    applied = bus.collect_orphaned_tensors(
        retention_cutoff_unix_ns=cutoff,
        dry_run=False,
    )
    assert applied.candidates == (old_orphan.digest,)
    assert applied.deleted == (old_orphan.digest,)
    assert not bus.tensors._path(old_orphan.digest).exists()
    assert bus.tensors._path(recent_orphan.digest).exists()
    assert bus.tensors._path(referenced_digest).exists()


def test_tensor_gc_aborts_before_deletion_on_corrupt_capsule_inventory(tmp_path):
    bus = _bus(tmp_path)
    orphan = bus.tensors.put(np.full(4, 9.0, dtype=np.float32))
    old_time = time.time_ns() - 10_000_000_000
    os.utime(bus.tensors._path(orphan.digest), ns=(old_time, old_time))
    corrupt_digest = hashlib.sha256(b"corrupt-envelope").hexdigest()
    corrupt = bus.capsules.objects / f"{corrupt_digest}.json"
    corrupt.write_bytes(b"{")
    corrupt.chmod(0o600)

    with pytest.raises(CapsuleIntegrityError, match="quarantined corrupt"):
        bus.collect_orphaned_tensors(
            retention_cutoff_unix_ns=time.time_ns() - 1_000_000_000,
            dry_run=False,
        )
    assert bus.tensors._path(orphan.digest).exists()


def test_tensor_gc_refuses_symlink_inventory_without_following_it(tmp_path):
    bus = _bus(tmp_path)
    orphan = bus.tensors.put(np.full(4, 7.0, dtype=np.float32))
    old_time = time.time_ns() - 10_000_000_000
    os.utime(bus.tensors._path(orphan.digest), ns=(old_time, old_time))
    target = tmp_path / "outside.tensor"
    target.write_bytes(b"must-not-be-read-or-deleted")
    link_digest = hashlib.sha256(b"symlink-object").hexdigest()
    (bus.tensors.objects / f"{link_digest}.tensor").symlink_to(target)

    with pytest.raises(ActivationCapsuleError, match="nonregular"):
        bus.collect_orphaned_tensors(
            retention_cutoff_unix_ns=time.time_ns() - 1_000_000_000,
            dry_run=False,
        )
    assert target.read_bytes() == b"must-not-be-read-or-deleted"
    assert bus.tensors._path(orphan.digest).exists()


def test_tensor_gc_aborts_when_a_live_capsule_tensor_is_missing(tmp_path):
    bus = _bus(tmp_path)
    identity = bus.publish(_spec(), {"residual": np.ones(4, dtype=np.float32)})
    referenced_digest = bus.capsules.get(identity.digest)["data"]["payloads"][
        "residual"
    ]["digest"]
    bus.tensors._path(referenced_digest).unlink()
    orphan = bus.tensors.put(np.full(4, 8.0, dtype=np.float32))
    old_time = time.time_ns() - 10_000_000_000
    os.utime(bus.tensors._path(orphan.digest), ns=(old_time, old_time))

    with pytest.raises(ActivationCapsuleError, match="missing or corrupt"):
        bus.collect_orphaned_tensors(
            retention_cutoff_unix_ns=time.time_ns() - 1_000_000_000,
            dry_run=False,
        )
    assert bus.tensors._path(orphan.digest).exists()
