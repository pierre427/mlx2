"""CPU-only admission tests for the distinct Metal AST probe boundary."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import mlx.core as mx
import pytest
from test_standard_xpress_serving_cpu import tiny

from mlx2.adapters.dense_qwen3_flat import (
    DenseQwen3FlatSegmentTarget,
    DenseQwen3MetalFlatSegmentTarget,
)
from mlx2.adapters import dense_qwen3_metal_probe as metal_probe
from mlx2.adapters.dense_qwen3_metal_probe import (
    run_dense_qwen3_metal_flat_probe,
)
from mlx2.runtime.flat_segmented_ast import (
    DEVICE_ATTESTATION_SCHEMA,
    FlatSegmentDescriptor,
    query_metal_flat_segment_admission,
    query_real_flat_segment_admission,
)
from mlx2.runtime.flat_segmented_kv import (
    FlatSegmentedKVRows,
    MetalFlatSegmentedKVRows,
    canonical_kv_state,
)

SOURCE_REVISION = "61822df87a3cab91b1654f86b150ab6a950b5a76"
ARTIFACT = (
    "tiny-dense-qwen3-test-fixture@sha256:"
    "8c33d8410274bbef931109847ecdcd46c2694f7d60a8a5664a064731b9b5f617"
)
EXPECTED_OWNER_SEMANTIC_SHA256 = (
    "4bcee4100b641fd1eafbe4a3578ccf939687330ddd19cc62bc42c8fb258821b4"
)
EXPECTED_METAL_OWNER_SEMANTIC_SHA256 = (
    "d0532ae0fbad26c682324bb2a34fb52aa067f9127a65d07178755bcd6049d39e"
)
EXPECTED_STATE_FINGERPRINTS = (
    "5cce324cf8b59feb02ac4362c53872b62b19d06fc9e0ceab478d2a867ea88a2e",
)
OWNER_SOURCE_PATHS = (
    "src/mlx2/runtime/flat_segmented_kv.py",
    "src/mlx2/runtime/flat_segmented_ast.py",
    "src/mlx2/runtime/ast_execution_capability.py",
    "src/mlx2/runtime/cache_operations.py",
    "src/mlx2/runtime/models/cache.py",
)
TARGET_SOURCE_PATHS = (
    "src/mlx2/adapters/dense_qwen3_flat.py",
    "src/mlx2/runtime/models/standard_decoder.py",
    "src/mlx2/runtime/models/base.py",
    "src/mlx2/runtime/models/activations.py",
    "src/mlx2/runtime/models/rope_utils.py",
    "src/mlx2/runtime/flat_segmented_ast.py",
    "src/mlx2/runtime/flat_segmented_kv.py",
    "src/mlx2/adapters/dense_qwen3_metal_probe.py",
)
_ROOT = Path(__file__).resolve().parents[1]


def _expected_sources(paths):
    return {
        path: hashlib.sha256((_ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def _metal_identity():
    return {
        "source_revision": SOURCE_REVISION,
        "build": f"mlx-{mx.__version__}",
        "device": "mlx-metal:unavailable",
        "artifact": ARTIFACT,
    }


def _cpu_identity():
    return {
        "source_revision": SOURCE_REVISION,
        "build": f"mlx-{mx.__version__}",
        "device": "mlx-cpu",
        "artifact": ARTIFACT,
    }


def _rows(model, lanes):
    return [model.make_cache() for _ in range(lanes)]


def _expected(owner, target):
    target_paths = (
        TARGET_SOURCE_PATHS
        if type(target) is DenseQwen3MetalFlatSegmentTarget
        else TARGET_SOURCE_PATHS[:-1]
    )
    return {
        "expected_identity": dict(owner.identity),
        "expected_owner_semantic_contract_sha256": (
            EXPECTED_METAL_OWNER_SEMANTIC_SHA256
            if type(owner) is MetalFlatSegmentedKVRows
            else EXPECTED_OWNER_SEMANTIC_SHA256
        ),
        "expected_state_contract_fingerprints": EXPECTED_STATE_FINGERPRINTS,
        "expected_owner_source_hashes": _expected_sources(OWNER_SOURCE_PATHS),
        "expected_target_source_hashes": _expected_sources(target_paths),
    }


def _metal_pair(lengths=(1, 2)):
    model, _ = tiny(tied=False)
    descriptor = FlatSegmentDescriptor.from_query_lengths(lengths)
    identity = _metal_identity()
    owner = MetalFlatSegmentedKVRows(
        _rows(model, len(lengths)),
        identity=identity,
    )
    target = DenseQwen3MetalFlatSegmentTarget(
        model,
        capture_layers=(0, 2),
        identity=identity,
    )
    return descriptor, owner, target


def test_metal_contract_is_distinct_and_cpu_receipt_is_truthful():
    descriptor, owner, target = _metal_pair()
    admission = query_metal_flat_segment_admission(
        owner,
        target,
        descriptor,
        **_expected(owner, target),
    )
    assert admission.available is False
    assert admission.decline_reasons == ("metal_device_unavailable",)
    assert admission.device_attestation == {
        "backend": "metal",
        "build": f"mlx-{mx.__version__}",
        "device": _metal_identity()["device"],
        "ready": False,
        "schema": DEVICE_ATTESTATION_SCHEMA,
    }
    assert admission.receipt()["observed_used"] is False
    assert admission.receipt()["execution_scope"] == (
        "adapter_owned_metal_exact_artifact_probe"
    )
    assert target.receipt() == {
        "route": "dense-qwen3-flat-segment-metal-direct-probe",
        "execution_scope": "adapter_owned_metal_exact_artifact_probe",
        "implemented": True,
        "qualified": False,
        "selected": False,
        "observed_used": False,
        "target_invocations": 0,
        "logical_rows": 0,
        "packed_matmul_authority": False,
        "projection_fidelity": "ordinary_b1s1_row_exact",
        "serving_wired": False,
        "device_backend": "metal",
        "device_attestation": admission.device_attestation,
        "device_admitted": False,
        "direct_probe_only": True,
        "performance_claim": False,
        "default": False,
    }


def test_cpu_cannot_run_the_metal_probe_and_live_state_never_changes():
    descriptor, owner, target = _metal_pair()
    before = canonical_kv_state(owner.rows)
    with pytest.raises(RuntimeError, match="metal_device_unavailable"):
        run_dense_qwen3_metal_flat_probe(
            owner,
            target,
            descriptor,
            (1, 2, 3),
            (1, 2),
            (4, 5),
            **_expected(owner, target),
        )
    assert canonical_kv_state(owner.rows) == before
    assert owner._active is None
    assert owner.revision == 0
    assert target.invocations == 0


@pytest.mark.parametrize(
    "lengths",
    [
        (1, 1, 1, 1, 1),
        (5,),
        (4, 4, 4, 4),
    ],
)
def test_probe_hard_limits_decline_before_oracle_or_state_work(lengths, monkeypatch):
    descriptor, owner, target = _metal_pair(lengths)
    before = canonical_kv_state(owner.rows)
    oracle_calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal oracle_calls
        oracle_calls += 1
        raise AssertionError("over-limit probe must not run the serial oracle")

    monkeypatch.setattr(metal_probe, "_ordinary_serial_oracles", forbidden)
    with pytest.raises(RuntimeError, match="descriptor_exceeds_hard_limits"):
        run_dense_qwen3_metal_flat_probe(
            owner,
            target,
            descriptor,
            tuple(1 for _ in range(descriptor.logical_rows)),
            tuple(1 for _ in lengths),
            tuple(2 for _ in lengths),
            **_expected(owner, target),
        )
    assert oracle_calls == 0
    assert canonical_kv_state(owner.rows) == before
    assert owner._active is None and owner.revision == 0
    assert target.invocations == 0


def test_cpu_and_metal_protocols_cannot_authorize_each_other():
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((1,))
    cpu_identity = _cpu_identity()
    cpu_owner = FlatSegmentedKVRows(_rows(model, 1), identity=cpu_identity)
    cpu_target = DenseQwen3FlatSegmentTarget(
        model,
        capture_layers=(0, 2),
        identity=cpu_identity,
    )
    mismatch = query_metal_flat_segment_admission(
        cpu_owner,
        cpu_target,
        descriptor,
        **_expected(cpu_owner, cpu_target),
    )
    assert mismatch.available is False
    assert "owner_capability_invalid" in mismatch.decline_reasons
    assert "target_capability_invalid" in mismatch.decline_reasons
    assert "device_attestation_invalid" in mismatch.decline_reasons

    _, metal_owner, metal_target = _metal_pair((1,))
    mismatch = query_real_flat_segment_admission(
        metal_owner,
        metal_target,
        descriptor,
        **_expected(metal_owner, metal_target),
    )
    assert mismatch.available is False
    assert "owner_capability_invalid" in mismatch.decline_reasons
    assert "target_capability_invalid" in mismatch.decline_reasons
    assert "device_attestation_invalid" in mismatch.decline_reasons


def test_expected_device_and_build_must_match_live_attestation():
    descriptor, owner, target = _metal_pair((1,))
    expected = _expected(owner, target)
    expected["expected_identity"] = {
        **expected["expected_identity"],
        "device": "mlx-metal:not-this-device",
        "build": "mlx-not-this-build",
    }
    admission = query_metal_flat_segment_admission(
        owner,
        target,
        descriptor,
        **expected,
    )
    assert admission.available is False
    assert "target_identity_mismatch" in admission.decline_reasons
    assert "owner_identity_mismatch" in admission.decline_reasons
    assert "device_identity_mismatch" in admission.decline_reasons
    assert owner._active is None and target.invocations == 0


@pytest.mark.parametrize("mutation", ["omitted", "extra"])
def test_reviewed_metal_source_manifest_fails_closed_on_shape_drift(mutation):
    descriptor, owner, target = _metal_pair((1,))
    expected = _expected(owner, target)
    sources = dict(expected["expected_target_source_hashes"])
    if mutation == "omitted":
        sources.pop("src/mlx2/adapters/dense_qwen3_metal_probe.py")
    else:
        sources["src/mlx2/adapters/not-reviewed.py"] = "0" * 64
    expected["expected_target_source_hashes"] = sources
    admission = query_metal_flat_segment_admission(
        owner,
        target,
        descriptor,
        **expected,
    )
    assert admission.available is False
    assert "target_source_hashes_mismatch" in admission.decline_reasons
    assert owner._active is None and target.invocations == 0


def test_cpu_ticket_revalidates_the_backend_before_begin(monkeypatch):
    model, _ = tiny()
    descriptor = FlatSegmentDescriptor.from_query_lengths((1,))
    identity = _cpu_identity()
    owner = FlatSegmentedKVRows(_rows(model, 1), identity=identity)
    target = DenseQwen3FlatSegmentTarget(
        model,
        capture_layers=(0, 2),
        identity=identity,
    )
    admission = query_real_flat_segment_admission(
        owner,
        target,
        descriptor,
        **_expected(owner, target),
    )
    assert admission.available
    monkeypatch.setattr(
        target,
        "ast_flat_segment_device_attestation",
        lambda: {
            **admission.device_attestation,
            "ready": False,
        },
    )
    with pytest.raises(RuntimeError, match="device attestation changed"):
        owner.begin_flat(
            descriptor,
            target_executor=target,
            admission=admission,
        )
    assert owner._active is None and owner.revision == 0


def test_metal_ticket_binds_and_revalidates_exact_device_attestation(monkeypatch):
    descriptor, owner, target = _metal_pair((1,))
    attestation = target.ast_flat_segment_device_attestation()
    monkeypatch.setattr(
        target,
        "ast_flat_segment_device_attestation",
        lambda: {**attestation, "ready": True},
    )
    admission = query_metal_flat_segment_admission(
        owner,
        target,
        descriptor,
        **_expected(owner, target),
    )
    assert admission.available
    assert admission.device_attestation["ready"] is True
    monkeypatch.setattr(
        target,
        "ast_flat_segment_device_attestation",
        lambda: attestation,
    )
    before = canonical_kv_state(copy.deepcopy(owner.rows))
    with pytest.raises(RuntimeError, match="device attestation changed"):
        owner.begin_flat(
            descriptor,
            target_executor=target,
            admission=admission,
        )
    assert canonical_kv_state(owner.rows) == before
    assert owner._active is None and owner.revision == 0
    assert target.invocations == 0


def test_continuation_mismatch_aborts_before_publication(monkeypatch):
    descriptor, owner, target = _metal_pair((1, 2))
    attestation = target.ast_flat_segment_device_attestation()
    # This exercises only the host lifecycle guard. The forced attestation is
    # test-local and the injected mismatch prevents a successful probe receipt.
    monkeypatch.setattr(
        target,
        "ast_flat_segment_device_attestation",
        lambda: {**attestation, "ready": True},
    )
    before = canonical_kv_state(owner.rows)
    references = tuple(
        (id(cache.keys), id(cache.values), cache.offset)
        for row in owner.rows
        for cache in row
    )
    calls = 0
    original = metal_probe._outputs_exact

    def fail_continuation(actual, expected):
        nonlocal calls
        calls += 1
        return original(actual, expected) if calls == 1 else False

    monkeypatch.setattr(metal_probe, "_outputs_exact", fail_continuation)
    with pytest.raises(ValueError, match="continuation oracle differs"):
        run_dense_qwen3_metal_flat_probe(
            owner,
            target,
            descriptor,
            (1, 2, 3),
            (1, 2),
            (4, 5),
            **_expected(owner, target),
        )
    assert canonical_kv_state(owner.rows) == before
    assert references == tuple(
        (id(cache.keys), id(cache.values), cache.offset)
        for row in owner.rows
        for cache in row
    )
    assert owner._active is None
    assert owner.revision == 1  # abort closes and invalidates the one-shot ticket
