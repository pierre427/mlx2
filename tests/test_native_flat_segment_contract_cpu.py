"""CPU-only contract tests for native logical-row AST state staging."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.runtime.flat_segmented_ast import (
    ABORT_LAW,
    COMMIT_LAW,
    DESCRIPTOR_SCHEMA,
    DEVICE_ATTESTATION_SCHEMA,
    EXECUTION_LAW,
    FIDELITY_EXACT,
    LAYOUT_CLASS,
    METAL_EXECUTION_SCOPE,
    METAL_TARGET_PROTOCOL,
    NATIVE_EXECUTION_SCOPE,
    NATIVE_OUTER_INVOCATION_LAW,
    NATIVE_PHYSICAL_LAYOUT,
    NATIVE_PHYSICAL_ROWS_LAW,
    NATIVE_ROW_ARITHMETIC_LAW,
    NATIVE_STATEFUL_CORE_LAW,
    NATIVE_TARGET_CAPABILITY_SCHEMA,
    NATIVE_TARGET_PROTOCOL,
    STATE_WRITE_LAW,
    TARGET_CAPABILITY_SCHEMA,
    TERMINAL_LAW,
    FlatSegmentDescriptor,
    FlatSegmentExecution,
    NativeFlatSegmentExecution,
    query_metal_flat_segment_admission,
    query_native_flat_segment_admission,
    validate_native_flat_segment_execution,
)
from mlx2.runtime.flat_segmented_hybrid import (
    FlatSegmentedHybridTransaction,
    HybridLiveStateBoundary,
    MetalFlatSegmentedHybridRows,
    NativeFlatSegmentedHybridRows,
    NativeFlatSegmentedHybridTransaction,
    canonical_bound_hybrid_state,
    canonical_hybrid_state,
)
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows, HybridVerifyTransaction
from mlx2.runtime.models.cache import ArraysCache, KVCache

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "53f02d6b538e8ae10272fe8305bec4c83ef99483"
ARTIFACT = "synthetic-native-hybrid@sha256:" + "5" * 64
IDENTITY = {
    "source_revision": SOURCE_REVISION,
    "build": f"mlx-{mx.__version__}",
    "device": "mlx-cpu:test",
    "artifact": ARTIFACT,
}
TARGET_SOURCE_PATHS = (
    "src/mlx2/runtime/flat_segmented_ast.py",
    "tests/test_native_flat_segment_contract_cpu.py",
)
OWNER_NATIVE_SOURCE_PATHS = (
    "src/mlx2/runtime/flat_segmented_hybrid.py",
    "src/mlx2/runtime/flat_segmented_ast.py",
    "src/mlx2/runtime/ast_execution_capability.py",
    "src/mlx2/runtime/hybrid_verify_rows.py",
    "src/mlx2/runtime/hybrid_speculative.py",
    "src/mlx2/runtime/cache_operations.py",
    "src/mlx2/runtime/models/cache.py",
)
HARD_LIMITS = {"max_lanes": 4, "max_segment_rows": 4, "max_total_rows": 10}
STATE_FINGERPRINTS = (
    "5cce324cf8b59feb02ac4362c53872b62b19d06fc9e0ceab478d2a867ea88a2e",
    "8491ce6cf23f5a337f6ec7a052bdfb8cd036c01ecf2b6e51e773d6798a764cbe",
)
NATIVE_OWNER_SEMANTIC_SHA256 = (
    "8fab6249b89dcef830983801c23eb8bd98e2e4126cc7337b7cd04b474286c967"
)
LEGACY_OWNER_SEMANTIC_SHA256 = (
    "9647f7809789677a70de10d22115bd3514da531b6dbece01c71142cb7b26cb0e"
)


def _sources(paths):
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


class SyntheticNativeTarget:
    """State-only fixture; it is not a real model tensor executor."""

    def __init__(self):
        self.identity = dict(IDENTITY)
        self.hard_limits = dict(HARD_LIMITS)
        self.ready = True
        self.contract_physical_layout = NATIVE_PHYSICAL_LAYOUT
        self.execution_slots = (0, 1)
        self.source_override = None
        self.invocations = 0

    def ast_native_flat_device_attestation(self):
        return {
            "schema": DEVICE_ATTESTATION_SCHEMA,
            "backend": "cpu",
            "device": self.identity["device"],
            "build": self.identity["build"],
            "ready": self.ready,
        }

    def ast_flat_segment_device_attestation(self):
        return {
            **self.ast_native_flat_device_attestation(),
            "backend": "metal",
        }

    def ast_native_flat_execution_identity(self):
        return (id(self), self.execution_slots)

    def ast_flat_segment_execution_identity(self):
        return (id(self), self.execution_slots)

    def ast_native_flat_target_contract(self):
        return {
            "schema": NATIVE_TARGET_CAPABILITY_SCHEMA,
            "protocol": NATIVE_TARGET_PROTOCOL,
            "execution_scope": NATIVE_EXECUTION_SCOPE,
            "fidelity": FIDELITY_EXACT,
            "execution_law": EXECUTION_LAW,
            "descriptor_schema": DESCRIPTOR_SCHEMA,
            "layout_classes": [LAYOUT_CLASS],
            "input_storage": "flat_logical_rows",
            "output_storage": "flat_logical_rows_in_segment_order",
            "position_law": "per_segment_base_plus_local_offset",
            "causal_isolation": "strict_no_cross_segment_reads",
            "terminal_law": TERMINAL_LAW,
            "state_write_law": STATE_WRITE_LAW,
            "commit_law": COMMIT_LAW,
            "abort_law": ABORT_LAW,
            "hard_limits": dict(self.hard_limits),
            "source_hashes": (
                _sources(TARGET_SOURCE_PATHS)
                if self.source_override is None
                else dict(self.source_override)
            ),
            "identity": dict(self.identity),
            "physical_layout": self.contract_physical_layout,
            "physical_rows_law": NATIVE_PHYSICAL_ROWS_LAW,
            "outer_invocation_law": NATIVE_OUTER_INVOCATION_LAW,
            "row_arithmetic_law": NATIVE_ROW_ARITHMETIC_LAW,
            "stateful_core_law": NATIVE_STATEFUL_CORE_LAW,
        }

    def ast_flat_segment_target_contract(self):
        return {
            "schema": TARGET_CAPABILITY_SCHEMA,
            "protocol": METAL_TARGET_PROTOCOL,
            "execution_scope": METAL_EXECUTION_SCOPE,
            "fidelity": FIDELITY_EXACT,
            "execution_law": EXECUTION_LAW,
            "descriptor_schema": DESCRIPTOR_SCHEMA,
            "layout_classes": [LAYOUT_CLASS],
            "input_storage": "flat_logical_rows",
            "output_storage": "flat_logical_rows_in_segment_order",
            "position_law": "per_segment_base_plus_local_offset",
            "causal_isolation": "strict_no_cross_segment_reads",
            "terminal_law": TERMINAL_LAW,
            "state_write_law": STATE_WRITE_LAW,
            "commit_law": COMMIT_LAW,
            "abort_law": ABORT_LAW,
            "hard_limits": dict(self.hard_limits),
            "source_hashes": (
                _sources(TARGET_SOURCE_PATHS)
                if self.source_override is None
                else dict(self.source_override)
            ),
            "identity": dict(self.identity),
        }

    def validate_native_flat_owner(self, owner, descriptor):
        if type(owner) is not NativeFlatSegmentedHybridRows:
            raise TypeError("synthetic target requires exact hybrid owner")
        if owner._target_executor_id != id(self):
            raise ValueError("synthetic target issuer differs")
        if len(owner.rows) != descriptor.lane_count:
            raise ValueError("synthetic target lane count differs")

    def validate_flat_owner(self, owner, descriptor):
        if type(owner) is not MetalFlatSegmentedHybridRows:
            raise TypeError("synthetic padded target requires exact padded owner")
        if owner._target_executor_id != id(self):
            raise ValueError("synthetic padded target issuer differs")
        if len(owner.rows) != descriptor.lane_count:
            raise ValueError("synthetic padded target lane count differs")

    @staticmethod
    def _advance_recurrent(cache, length, lane):
        snapshot = list(cache.cache)

        def replay(prefix):
            return [
                snapshot[0] + (lane + 1) * prefix,
                snapshot[1] + (lane + 2) * prefix,
            ]

        cache.record_rollback(length, replay, snapshot)
        cache.cache = replay(length)

    def execute_native_flat_segments(self, values, descriptor, transaction):
        lane_caches = transaction.lane_caches
        self.invocations += 1
        for lane, (row, length) in enumerate(
            zip(lane_caches, descriptor.query_lengths, strict=True)
        ):
            kv, recurrent = row
            keys = mx.full((1, 1, length, 2), lane + 1, dtype=mx.float32)
            values_ = mx.full((1, 1, length, 2), lane + 2, dtype=mx.float32)
            kv.update_and_fetch(keys, values_)
            self._advance_recurrent(recurrent, length, lane)
        return NativeFlatSegmentExecution(
            outputs=tuple(values),
            geometry_sha256=descriptor.geometry_sha256,
            physical_rows=descriptor.logical_rows,
        )

    def execute_flat_segments(self, values, descriptor, transaction):
        return FlatSegmentExecution(
            outputs=tuple(values),
            geometry_sha256=descriptor.geometry_sha256,
        )


def _row():
    recurrent = ArraysCache(2)
    recurrent.cache = [mx.zeros((1, 2)), mx.ones((1, 2))]
    return [KVCache(), recurrent]


def _pair(lengths=(1, 2, 3, 4)):
    target = SyntheticNativeTarget()
    rows = tuple(_row() for _ in lengths)
    boundaries = tuple(
        HybridLiveStateBoundary._mint(
            revision=ARTIFACT,
            lane_id=lane,
            position=0,
            generation=0,
            row=row,
            issuer_id=id(target),
        )
        for lane, row in enumerate(rows)
    )
    owner = NativeFlatSegmentedHybridRows(
        rows,
        identity=IDENTITY,
        boundary_checkpoints=boundaries,
        target_executor=target,
        hard_limits=HARD_LIMITS,
    )
    descriptor = FlatSegmentDescriptor.from_query_lengths(lengths)
    return descriptor, owner, target


def _legacy_pair(lengths=(1, 2, 3, 4)):
    target = SyntheticNativeTarget()
    rows = tuple(_row() for _ in lengths)
    boundaries = tuple(
        HybridLiveStateBoundary._mint(
            revision=ARTIFACT,
            lane_id=lane,
            position=0,
            generation=0,
            row=row,
            issuer_id=id(target),
        )
        for lane, row in enumerate(rows)
    )
    owner = MetalFlatSegmentedHybridRows(
        rows,
        identity=IDENTITY,
        boundary_checkpoints=boundaries,
        target_executor=target,
    )
    return FlatSegmentDescriptor.from_query_lengths(lengths), owner, target


def _expected(owner):
    return {
        "expected_identity": IDENTITY,
        "expected_owner_semantic_contract_sha256": NATIVE_OWNER_SEMANTIC_SHA256,
        "expected_state_contract_fingerprints": STATE_FINGERPRINTS,
        "expected_owner_source_hashes": _sources(OWNER_NATIVE_SOURCE_PATHS),
        "expected_target_source_hashes": _sources(TARGET_SOURCE_PATHS),
        "expected_backend": "cpu",
    }


def _admission(owner, target, descriptor):
    return query_native_flat_segment_admission(
        owner, target, descriptor, **_expected(owner)
    )


def _execute(owner, target, descriptor):
    admission = _admission(owner, target, descriptor)
    assert admission.available, admission.decline_reasons
    transaction = owner.begin_native_flat(
        descriptor, target_executor=target, admission=admission
    )
    execution = target.execute_native_flat_segments(
        tuple(range(descriptor.logical_rows)), descriptor, transaction
    )
    validate_native_flat_segment_execution(execution, descriptor)
    return transaction, execution


def _serial_b1_accepted_prefix_oracle(owner, descriptor, accepted):
    """Independent ordinary B1 state oracle for the synthetic fixture math."""

    successors = []
    for lane, (live_row, length, keep) in enumerate(
        zip(owner.rows, descriptor.query_lengths, accepted, strict=True)
    ):
        lane_owner = HybridVerifyRows([copy.deepcopy(list(live_row))])
        transaction = lane_owner.begin([length])
        kv, recurrent = transaction.caches
        keys = mx.full((1, 1, length, 2), lane + 1, dtype=mx.float32)
        values = mx.full((1, 1, length, 2), lane + 2, dtype=mx.float32)
        kv.update_and_fetch(keys, values)
        snapshot = list(recurrent.cache)

        def replay(prefix, *, snapshot=snapshot, lane=lane):
            return [
                snapshot[0] + (lane + 1) * prefix,
                snapshot[1] + (lane + 2) * prefix,
            ]

        recurrent.record_rollback(length, replay, snapshot)
        recurrent.cache = replay(length)
        (successor,) = transaction.commit([keep])
        successors.append(successor)
    boundaries = tuple(
        HybridLiveStateBoundary._mint(
            revision=prior.revision,
            lane_id=prior.lane_id,
            position=prior.position + keep,
            generation=prior.generation + 1,
            row=row,
            issuer_id=prior.issuer_id,
        )
        for prior, row, keep in zip(
            owner.boundaries, successors, accepted, strict=True
        )
    )
    return tuple(tuple(row) for row in successors), boundaries


def _assert_raw_hybrid_rows_exact(actual_rows, expected_rows):
    assert canonical_hybrid_state(actual_rows) == canonical_hybrid_state(expected_rows)
    for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True):
        for actual, expected in zip(actual_row, expected_row, strict=True):
            assert type(actual) is type(expected)
            if type(actual) is KVCache:
                assert actual.offset == expected.offset
                assert (actual.keys is None) == (expected.keys is None)
                if actual.keys is not None:
                    assert bool(mx.array_equal(actual.keys, expected.keys))
                    assert bool(mx.array_equal(actual.values, expected.values))
            else:
                assert actual.speculating is expected.speculating is False
                assert not actual._rollbacks and not expected._rollbacks
                for actual_value, expected_value in zip(
                    actual.cache, expected.cache, strict=True
                ):
                    assert bool(mx.array_equal(actual_value, expected_value))
                for field in ("lengths", "left_padding"):
                    actual_value = getattr(actual, field)
                    expected_value = getattr(expected, field)
                    assert (actual_value is None) == (expected_value is None)
                    if actual_value is not None:
                        assert bool(mx.array_equal(actual_value, expected_value))


def test_b4_native_rows_use_one_invocation_without_padded_views(monkeypatch):
    descriptor, owner, target = _pair()
    before = canonical_bound_hybrid_state(owner.rows, owner.boundaries)
    accepted = (1, 1, 2, 3)
    oracle_rows, oracle_boundaries = _serial_b1_accepted_prefix_oracle(
        owner, descriptor, accepted
    )
    expected = canonical_bound_hybrid_state(oracle_rows, oracle_boundaries)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native-flat must not construct padded batch views")

    monkeypatch.setattr(HybridVerifyRows, "views", forbidden)
    transaction, execution = _execute(owner, target, descriptor)
    prepared = transaction.prepare_commit(accepted)
    assert prepared.state == expected
    committed = transaction.publish_commit(prepared)
    assert execution.physical_rows == descriptor.logical_rows == 10
    assert execution.invocation_count == target.invocations == 1
    assert committed == expected and committed != before
    assert canonical_bound_hybrid_state(owner.rows, owner.boundaries) == expected
    _assert_raw_hybrid_rows_exact(owner.rows, oracle_rows)
    assert [boundary.position for boundary in owner.boundaries] == [1, 1, 2, 3]
    assert owner.revision == 1 and owner._active is None


def test_missing_native_target_attestation_declines_to_existing_contract():
    descriptor, owner, target = _pair()
    target.ast_native_flat_target_contract = None
    target.execute_native_flat_segments = None
    admission = _admission(owner, target, descriptor)
    assert admission.available is False
    assert admission.decline_reasons == (
        "target_native_flat_capability_unavailable",
        "target_native_flat_executor_unavailable",
    )
    assert owner._active is None and owner.revision == 0


def test_padded_owner_does_not_advertise_native_and_v1_still_begins_and_aborts():
    descriptor, owner, target = _legacy_pair()
    assert not hasattr(owner, "ast_native_flat_transaction_contract")
    assert not hasattr(owner, "begin_native_flat")
    native = query_native_flat_segment_admission(
        owner, target, descriptor, **_expected(owner)
    )
    assert native.available is False
    assert "owner_native_flat_capability_unavailable" in native.decline_reasons
    assert "owner_begin_native_flat_unavailable" in native.decline_reasons

    padded = query_metal_flat_segment_admission(
        owner,
        target,
        descriptor,
        expected_identity=IDENTITY,
        expected_owner_semantic_contract_sha256=LEGACY_OWNER_SEMANTIC_SHA256,
        expected_state_contract_fingerprints=STATE_FINGERPRINTS,
        expected_owner_source_hashes=owner.source_hashes,
        expected_target_source_hashes=_sources(TARGET_SOURCE_PATHS),
    )
    assert padded.available, padded.decline_reasons
    transaction = owner.begin_flat(
        descriptor, target_executor=target, admission=padded
    )
    assert type(transaction) is FlatSegmentedHybridTransaction
    transaction.abort()
    assert owner._active is None and owner.revision == 1


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda _owner, target: setattr(target, "ready", False), "cpu_device_unavailable"),
        (
            lambda _owner, target: setattr(
                target, "contract_physical_layout", "padded_rectangular"
            ),
            "target_native_flat_capability_invalid",
        ),
        (
            lambda _owner, target: target.hard_limits.update(max_total_rows=9),
            "descriptor_exceeds_hard_limits",
        ),
        (
            lambda owner, _target: owner.identity.update(artifact="wrong@sha256:" + "6" * 64),
            "owner_identity_mismatch",
        ),
    ],
)
def test_device_layout_capacity_and_artifact_drift_decline_before_begin(
    mutation, reason
):
    descriptor, owner, target = _pair()
    before = canonical_bound_hybrid_state(owner.rows, owner.boundaries)
    mutation(owner, target)
    admission = _admission(owner, target, descriptor)
    assert admission.available is False and reason in admission.decline_reasons
    assert canonical_bound_hybrid_state(owner.rows, owner.boundaries) == before
    assert owner._active is None and owner.revision == 0 and target.invocations == 0


def test_state_drift_invalidates_ticket_before_begin():
    descriptor, owner, target = _pair()
    admission = _admission(owner, target, descriptor)
    owner.rows[0][0].update_and_fetch(
        mx.zeros((1, 1, 1, 2)), mx.zeros((1, 1, 1, 2))
    )
    with pytest.raises((ValueError, RuntimeError), match="state|boundary"):
        owner.begin_native_flat(
            descriptor, target_executor=target, admission=admission
        )
    assert owner._active is None and target.invocations == 0


@pytest.mark.parametrize(
    ("authority", "path"),
    [("owner", path) for path in OWNER_NATIVE_SOURCE_PATHS]
    + [("target", path) for path in TARGET_SOURCE_PATHS],
)
def test_each_implementation_source_drift_declines_before_begin(authority, path):
    descriptor, owner, target = _pair()
    expected = _expected(owner)
    field = f"expected_{authority}_source_hashes"
    expected[field] = dict(expected[field])
    expected[field][path] = "0" * 64
    admission = query_native_flat_segment_admission(
        owner, target, descriptor, **expected
    )
    assert admission.available is False
    assert f"{authority}_source_hashes_mismatch" in admission.decline_reasons
    assert owner._active is None and owner.revision == 0 and target.invocations == 0


@pytest.mark.parametrize("drift", ["device", "execution_identity", "capability", "source"])
def test_post_begin_target_drift_fails_before_private_lane_access(drift):
    descriptor, owner, target = _pair()
    before = canonical_bound_hybrid_state(owner.rows, owner.boundaries)
    admission = _admission(owner, target, descriptor)
    transaction = owner.begin_native_flat(
        descriptor, target_executor=target, admission=admission
    )
    if drift == "device":
        target.ready = False
    elif drift == "execution_identity":
        target.execution_slots = (0, 2)
    elif drift == "capability":
        target.contract_physical_layout = "padded_rectangular"
    else:
        target.source_override = _sources(TARGET_SOURCE_PATHS)
        target.source_override[TARGET_SOURCE_PATHS[-1]] = "0" * 64
    with pytest.raises(RuntimeError, match="native-flat"):
        try:
            target.execute_native_flat_segments(
                tuple(range(descriptor.logical_rows)), descriptor, transaction
            )
        finally:
            transaction.abort()
    assert target.invocations == 0
    assert canonical_bound_hybrid_state(owner.rows, owner.boundaries) == before
    assert owner._active is None and owner.revision == 1


def test_injected_begin_failure_aborts_all_private_lanes(monkeypatch):
    descriptor, owner, target = _pair()
    before = canonical_bound_hybrid_state(owner.rows, owner.boundaries)
    admission = _admission(owner, target, descriptor)
    original = HybridVerifyRows.begin
    calls = 0

    def fail_third(self, lengths, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected native lane begin failure")
        return original(self, lengths, **kwargs)

    monkeypatch.setattr(HybridVerifyRows, "begin", fail_third)
    with pytest.raises(RuntimeError, match="lane begin failure"):
        owner.begin_native_flat(
            descriptor, target_executor=target, admission=admission
        )
    assert canonical_bound_hybrid_state(owner.rows, owner.boundaries) == before
    assert owner._active is None and owner.revision == 0


def test_injected_lane_failure_aborts_without_live_publication(monkeypatch):
    descriptor, owner, target = _pair()
    before = canonical_bound_hybrid_state(owner.rows, owner.boundaries)
    admission = _admission(owner, target, descriptor)
    transaction = owner.begin_native_flat(
        descriptor, target_executor=target, admission=admission
    )
    original = target._advance_recurrent

    def fail_lane(cache, length, lane):
        if lane == 1:
            raise RuntimeError("injected native lane execution failure")
        return original(cache, length, lane)

    monkeypatch.setattr(target, "_advance_recurrent", fail_lane)
    with pytest.raises(RuntimeError, match="lane execution failure"):
        try:
            target.execute_native_flat_segments(
                tuple(range(descriptor.logical_rows)), descriptor, transaction
            )
        finally:
            transaction.abort()
    assert canonical_bound_hybrid_state(owner.rows, owner.boundaries) == before
    assert owner._active is None and owner.revision == 1


def test_injected_prepare_failure_discards_every_private_successor(monkeypatch):
    descriptor, owner, target = _pair()
    before = canonical_bound_hybrid_state(owner.rows, owner.boundaries)
    transaction, _ = _execute(owner, target, descriptor)
    original = HybridVerifyTransaction.commit
    calls = 0

    def fail_second(self, accepted):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected native prepare failure")
        return original(self, accepted)

    monkeypatch.setattr(HybridVerifyTransaction, "commit", fail_second)
    with pytest.raises(RuntimeError, match="native prepare failure"):
        transaction.prepare_commit((1, 1, 2, 3))
    assert canonical_bound_hybrid_state(owner.rows, owner.boundaries) == before
    assert owner._active is None and owner.revision == 1


def test_injected_publish_failure_restores_all_live_planes(monkeypatch):
    descriptor, owner, target = _pair()
    before = canonical_bound_hybrid_state(owner.rows, owner.boundaries)
    prior_boundaries = owner.boundaries
    transaction, _ = _execute(owner, target, descriptor)
    prepared = transaction.prepare_commit((1, 1, 2, 3))
    original = NativeFlatSegmentedHybridTransaction._assign
    calls = 0

    def fail_second(live, successor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected native publish failure")
        return original(live, successor)

    monkeypatch.setattr(
        NativeFlatSegmentedHybridTransaction,
        "_assign",
        staticmethod(fail_second),
    )
    with pytest.raises(RuntimeError, match="native publish failure"):
        transaction.publish_commit(prepared)
    assert canonical_bound_hybrid_state(owner.rows, owner.boundaries) == before
    assert owner.boundaries == prior_boundaries
    assert all(boundary.valid for boundary in prior_boundaries)
    assert owner._active is None and owner.revision == 1


def test_late_boundary_invalidation_failure_restores_atomic_publication(monkeypatch):
    descriptor, owner, target = _pair()
    prior_rows = owner.rows
    prior_member_ids = tuple(tuple(id(cache) for cache in row) for row in owner.rows)
    prior_boundaries = owner.boundaries
    before = canonical_bound_hybrid_state(owner.rows, prior_boundaries)
    transaction, _ = _execute(owner, target, descriptor)
    prepared = transaction.prepare_commit((1, 1, 2, 3))
    original = HybridLiveStateBoundary._invalidate
    calls = 0

    def fail_second_after_invalidation(self, bound_owner):
        nonlocal calls
        calls += 1
        original(self, bound_owner)
        if calls == 2:
            raise RuntimeError("injected late boundary invalidation failure")

    monkeypatch.setattr(HybridLiveStateBoundary, "_invalidate", fail_second_after_invalidation)
    with pytest.raises(RuntimeError, match="late boundary invalidation failure"):
        transaction.publish_commit(prepared)
    assert owner.rows is prior_rows
    assert tuple(tuple(id(cache) for cache in row) for row in owner.rows) == prior_member_ids
    assert owner.boundaries == prior_boundaries
    assert all(boundary.valid for boundary in prior_boundaries)
    assert canonical_bound_hybrid_state(owner.rows, owner.boundaries) == before
    assert owner._active is None and owner.revision == 1


@pytest.mark.parametrize(
    ("physical_rows", "invocations"),
    [(16, 1), (10, 2), (10, True)],
)
def test_execution_receipt_rejects_padding_or_multiple_outer_invocations(
    physical_rows, invocations
):
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2, 3, 4))
    execution = NativeFlatSegmentExecution(
        outputs=tuple(range(10)),
        geometry_sha256=descriptor.geometry_sha256,
        physical_rows=physical_rows,
        invocation_count=invocations,
    )
    with pytest.raises(ValueError, match="physical rows|one outer invocation"):
        validate_native_flat_segment_execution(execution, descriptor)
