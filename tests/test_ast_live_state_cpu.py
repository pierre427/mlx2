from __future__ import annotations

from dataclasses import replace

import mlx.core as mx
import pytest

from mlx2.runtime.ast_live_state import (
    ASTLiveLaneStateRegistry,
    ASTLiveStateAuthority,
    ASTLiveStateError,
    ASTPhysicalCommitCoordinator,
    ASTPhysicalResult,
    ASTPhysicalResultPlan,
)
from mlx2.runtime.cache_operations import KV_OPERATIONS
from mlx2.runtime.flat_segmented_ast import (
    FlatSegmentDescriptor,
    NativeFlatSegmentExecution,
)
from mlx2.runtime.models.cache import ArraysCache, KVCache

SHA = "a" * 64
IDENTITY = ASTLiveStateAuthority(
    artifact="Qwen3.8-27B-oQ4e-mtp@sha256:" + "b" * 64,
    session_scope_sha256="1" * 64,
    owner_identity_sha256="2" * 64,
    provider_manifest_sha256="3" * 64,
    source_manifest_sha256="4" * 64,
    build_manifest_sha256="5" * 64,
    device_manifest_sha256="6" * 64,
    max_lanes=4,
    max_caches_per_lane=8,
    max_segment_rows=4,
    max_total_rows=12,
    max_physical_rows=16,
    required_plane_kinds=("attention_kv", "gdn_recurrent"),
    publication_modes=("in_place_base", "transfer_successor"),
    result_storage_classes=(
        "flat_logical_rows",
        "flat_row_pairs",
        "rectangular_rows",
    ),
)


def _row(position: int, lane: int):
    kv = KVCache()
    if position:
        keys = mx.full((1, 1, position, 2), lane + 1, dtype=mx.float32)
        values = mx.full((1, 1, position, 2), lane + 2, dtype=mx.float32)
        kv.update_and_fetch(keys, values)
    recurrent = ArraysCache(2)
    recurrent.cache = [
        mx.full((1, 2), lane + 3, dtype=mx.float32),
        mx.full((1, 2), lane + 4, dtype=mx.float32),
    ]
    recurrent.left_padding = mx.array([0])
    recurrent.lengths = mx.array([position])
    return [kv, recurrent]


def _rows(positions=(2, 3)):
    return tuple(_row(position, lane) for lane, position in enumerate(positions))


def _result(descriptor, *, rectangular=False):
    if rectangular:
        width = descriptor.max_segment_rows
        logits = mx.arange(descriptor.lane_count * width * 3).reshape(
            descriptor.lane_count, width, 3
        )
        features = mx.arange(descriptor.lane_count * width * 2).reshape(
            descriptor.lane_count, width, 2
        )
        return ASTPhysicalResult(
            descriptor,
            logits,
            features,
            storage="rectangular_rows",
            invocation_count=1,
        )
    logits = mx.arange(descriptor.logical_rows * 3).reshape(descriptor.logical_rows, 3)
    features = mx.arange(descriptor.logical_rows * 2).reshape(
        descriptor.logical_rows, 2
    )
    return ASTPhysicalResult(
        descriptor,
        logits,
        features,
        storage="flat_logical_rows",
        invocation_count=1,
    )


def _result_plan(result):
    return ASTPhysicalResultPlan.from_descriptor(
        result.descriptor,
        storage=result.storage,
        rectangular_width=result.width,
    )


class _Delegate:
    def __init__(
        self,
        base_rows,
        successor_rows,
        descriptor,
        result,
        *,
        fail_prepare=False,
        fail_publish=False,
        mutate_base_during_prepare=False,
        publication_mode="in_place_base",
        publication_rows=None,
    ):
        self.base_rows = base_rows
        self.successor_rows = successor_rows
        self.descriptor = descriptor
        self.result = result
        self.fail_prepare = fail_prepare
        self.fail_publish = fail_publish
        self.mutate_base_during_prepare = mutate_base_during_prepare
        self.publication_mode = publication_mode
        self.publication_rows = (
            self.base_rows if publication_rows is None else publication_rows
        )
        self.prepared = 0
        self.published = 0
        self.aborted = 0
        self.executed = 0
        self.result_plan_seen = None
        self._backup = None

    def execute_private(self, *, descriptor, result_plan):
        self.executed += 1
        self.result_plan_seen = result_plan
        return self.result

    def prepare_commit(self, accepted, *, descriptor, result, prepare_authority):
        self.prepared += 1
        if self.fail_prepare:
            raise RuntimeError("prepare failed")
        if self.mutate_base_during_prepare:
            self.base_rows[0][0].offset += 1
        positions = tuple(row[0].offset for row in self.successor_rows)
        return prepare_authority.seal(
            successor_rows=self.successor_rows,
            publication_rows=self.publication_rows,
            publication_mode=self.publication_mode,
            accepted_lengths=accepted,
            successor_positions=positions,
            descriptor_geometry_sha256=descriptor.geometry_sha256,
            result_manifest_sha256=result.manifest_sha256,
            delegate_token=object(),
        )

    def publish_commit(self, prepared):
        self.published += 1
        self._backup = tuple(
            (
                row[0].keys,
                row[0].values,
                row[0].offset,
                row[1].cache,
                row[1].left_padding,
                row[1].lengths,
            )
            for row in self.base_rows
        )
        try:
            for live, successor in zip(self.base_rows, prepared.successor_rows):
                live[0].keys = successor[0].keys
                live[0].values = successor[0].values
                live[0].offset = successor[0].offset
                live[1].cache = successor[1].cache
                live[1].left_padding = successor[1].left_padding
                live[1].lengths = successor[1].lengths
            if self.fail_publish:
                raise RuntimeError("publish failed")
        except BaseException:
            self._restore()
            raise

    def abort(self):
        self.aborted += 1
        if self._backup is not None:
            self._restore()

    def _restore(self):
        for row, values in zip(self.base_rows, self._backup):
            (
                row[0].keys,
                row[0].values,
                row[0].offset,
                row[1].cache,
                row[1].left_padding,
                row[1].lengths,
            ) = values


def _coordinator(lengths=(2, 3), positions=(2, 3)):
    descriptor = FlatSegmentDescriptor.from_query_lengths(lengths)
    result = _result(descriptor)
    base = _rows(positions)
    registry = ASTLiveLaneStateRegistry(base, positions, authority=IDENTITY)
    accepted = tuple(min(length, lane + 1) for lane, length in enumerate(lengths))
    successor_positions = tuple(
        position + accepted[lane] for lane, position in enumerate(positions)
    )
    successor = _rows(successor_positions)
    delegate = _Delegate(base, successor, descriptor, result)
    coordinator = ASTPhysicalCommitCoordinator(
        registry,
        registry.handles,
        base,
        positions,
        descriptor=descriptor,
        result_plan=_result_plan(result),
        delegate=delegate,
        authority=IDENTITY,
    )
    coordinator.execute(authority=IDENTITY)
    return (
        descriptor,
        result,
        base,
        successor,
        registry,
        delegate,
        coordinator,
        accepted,
    )


def test_flat_and_rectangular_result_views_do_not_materialize_padding():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2, 4))
    flat = _result(descriptor)
    rectangular = _result(descriptor, rectangular=True)

    assert flat.physical_rows == descriptor.logical_rows == 7
    assert flat.padding_rows == 0
    assert rectangular.physical_rows == 12
    assert rectangular.padding_rows == 5
    assert flat.logits is not rectangular.logits
    assert flat.lane_slices == ((None, 0, 1), (None, 1, 3), (None, 3, 7))
    assert rectangular.lane_slices == ((0, 0, 1), (1, 0, 2), (2, 0, 4))
    assert tuple(flat.lane(1)[0].shape) == (2, 3)
    assert tuple(rectangular.lane(1)[0].shape) == (2, 3)
    assert tuple(flat.row(2, 3)[0].shape) == (3,)
    assert flat.receipt()["numerical_claim"] is False
    assert flat.receipt()["apcv2_serializable"] is False
    assert flat.receipt()["selected"] is False


def test_native_row_pair_execution_wraps_existing_storage_without_tensor_stack():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    rows = tuple(
        (mx.full((3,), row), mx.full((2,), row + 10))
        for row in range(descriptor.logical_rows)
    )
    execution = NativeFlatSegmentExecution(
        outputs=rows,
        geometry_sha256=descriptor.geometry_sha256,
        physical_rows=descriptor.logical_rows,
    )
    result = ASTPhysicalResult.from_native_execution(execution, descriptor)

    assert result.rows is rows
    assert result.logits is None
    assert result.row(1, 1)[0] is rows[2][0]
    lane_logits, lane_features = result.lane(1)
    assert lane_logits == (rows[1][0], rows[2][0])
    assert lane_features == (rows[1][1], rows[2][1])
    assert result.receipt()["tensor_values_serialized"] is False
    assert result.receipt()["logits_shape"] == (3, 3)
    assert result.receipt()["features_shape"] == (3, 2)
    assert result.receipt()["logits_dtype"] == str(mx.int32)
    with pytest.raises(AttributeError, match="immutable"):
        result.storage = "rectangular_rows"


def test_native_row_pair_storage_requires_immutable_consistent_metadata():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 1))
    rows = (
        (mx.zeros((3,), dtype=mx.float32), mx.zeros((2,), dtype=mx.float32)),
        (mx.zeros((4,), dtype=mx.float32), mx.zeros((2,), dtype=mx.float32)),
    )
    with pytest.raises(ASTLiveStateError, match="inconsistent"):
        ASTPhysicalResult(
            descriptor,
            rows,
            None,
            storage="flat_row_pairs",
            invocation_count=1,
        )
    with pytest.raises(ASTLiveStateError, match="immutable"):
        ASTPhysicalResult(
            descriptor,
            list(rows),
            None,
            storage="flat_row_pairs",
            invocation_count=1,
        )


@pytest.mark.parametrize(
    "storage,shape",
    [
        ("flat_logical_rows", (5, 3)),
        ("rectangular_rows", (2, 2, 3)),
    ],
)
def test_result_rejects_wrong_shapes(storage, shape):
    descriptor = FlatSegmentDescriptor.from_query_lengths((2, 4))
    with pytest.raises(ASTLiveStateError):
        ASTPhysicalResult(
            descriptor,
            mx.zeros(shape),
            None,
            storage=storage,
            invocation_count=1,
        )


def test_result_rejects_bool_invocation_and_invalid_access():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1,))
    with pytest.raises(ASTLiveStateError):
        ASTPhysicalResult(
            descriptor,
            mx.zeros((1, 2)),
            None,
            storage="flat_logical_rows",
            invocation_count=True,
        )
    result = _result(descriptor)
    with pytest.raises(ASTLiveStateError):
        result.row(0, 1)


@pytest.mark.parametrize(
    "storage,shape",
    [
        ("flat_logical_rows", (3,)),
        ("flat_logical_rows", (3, 0)),
        ("rectangular_rows", (2, 2)),
        ("rectangular_rows", (2, 2, 0)),
    ],
)
def test_result_rejects_missing_or_empty_payload_dimensions(storage, shape):
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    with pytest.raises(ASTLiveStateError, match="shape|ABI"):
        ASTPhysicalResult(
            descriptor,
            mx.zeros(shape),
            None,
            storage=storage,
            invocation_count=1,
        )

    rows = tuple(
        (mx.zeros((0,)), mx.zeros((2,))) for _ in range(descriptor.logical_rows)
    )
    with pytest.raises(ASTLiveStateError, match="positive"):
        ASTPhysicalResult(
            descriptor,
            rows,
            None,
            storage="flat_row_pairs",
            invocation_count=1,
        )


def test_result_plan_is_token_free_and_binds_exact_physical_geometry():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2, 4))
    flat = ASTPhysicalResultPlan.from_descriptor(
        descriptor,
        storage="flat_logical_rows",
    )
    rectangular = ASTPhysicalResultPlan.from_descriptor(
        descriptor,
        storage="rectangular_rows",
        rectangular_width=5,
    )

    assert flat.physical_rows == descriptor.logical_rows
    assert rectangular.physical_rows == 15
    assert rectangular.receipt()["padding_rows"] == 8
    assert rectangular.receipt()["tensor_values_serialized"] is False
    with pytest.raises(ASTLiveStateError, match="row law"):
        replace(rectangular, physical_rows=14)


def test_registry_receipt_is_membership_only_and_handles_hide_object_ids():
    rows = _rows()
    registry = ASTLiveLaneStateRegistry(rows, (2, 3), authority=IDENTITY)
    receipt = registry.receipt()
    assert receipt["identity_class"] == "exclusive_lifecycle_membership_only"
    assert receipt["numerical_state_identity"] is False
    assert receipt["selected"] is False
    assert receipt["authority"]["artifact"] == IDENTITY.artifact
    assert receipt["authority_manifest_sha256"] == IDENTITY.manifest_sha256
    assert "registry_id" not in registry.handles[0].receipt()
    contract = ASTPhysicalCommitCoordinator.contract_receipt()
    assert contract["publish_law"] == "delegate_atomic_publish_or_internal_rollback"
    assert contract["serving_wired"] is False
    semantics = registry._state.signatures[0][1].evaluation_semantics
    assert isinstance(semantics, tuple)
    assert all(isinstance(item, tuple) for item in semantics)


def test_registry_rejects_cache_alias_state_alias_lookalike_and_capacity():
    row = _row(2, 0)
    with pytest.raises(ASTLiveStateError, match="alias"):
        ASTLiveLaneStateRegistry((row, row), (2, 2), authority=IDENTITY)

    rows = _rows((2, 2))
    rows[1][1].cache[0] = rows[0][1].cache[0]
    with pytest.raises(ASTLiveStateError, match="members must not alias"):
        ASTLiveLaneStateRegistry(rows, (2, 2), authority=IDENTITY)

    rows = _rows((2, 2))
    shared_empty = mx.array([], dtype=mx.float32)
    rows[0][1].cache[0] = shared_empty
    rows[1][1].cache[0] = shared_empty
    registry = ASTLiveLaneStateRegistry(rows, (2, 2), authority=IDENTITY)
    assert len(registry.handles) == 2

    class Lookalike(KVCache):
        pass

    bad = _row(2, 0)
    bad[0] = Lookalike()
    with pytest.raises(Exception, match="explicit cache operations contract"):
        ASTLiveLaneStateRegistry((bad,), (0,), authority=IDENTITY)

    too_many = tuple(_row(1, lane) for lane in range(5))
    with pytest.raises(ASTLiveStateError, match="authority"):
        ASTLiveLaneStateRegistry(too_many, (1,) * 5, authority=IDENTITY)


def test_authority_declares_required_planes_and_each_lane_needs_position_authority():
    kv_only = replace(
        IDENTITY,
        required_plane_kinds=("attention_kv",),
        max_caches_per_lane=1,
    )
    kv_rows = tuple((row[0],) for row in _rows())
    registry = ASTLiveLaneStateRegistry(kv_rows, (2, 3), authority=kv_only)
    assert len(registry.handles) == 2

    recurrent_only = replace(
        IDENTITY,
        required_plane_kinds=("gdn_recurrent",),
        max_caches_per_lane=1,
    )
    recurrent_rows = tuple((row[1],) for row in _rows())
    with pytest.raises(ASTLiveStateError, match="positional"):
        ASTLiveLaneStateRegistry(
            recurrent_rows,
            (2, 3),
            authority=recurrent_only,
        )


def test_registry_rejects_position_order_stale_handle_and_double_lease():
    rows = _rows()
    registry = ASTLiveLaneStateRegistry(rows, (2, 3), authority=IDENTITY)
    handles = registry.handles
    with pytest.raises(ASTLiveStateError, match="position"):
        registry.begin(handles, rows, (2, 4), authority=IDENTITY)
    with pytest.raises(ASTLiveStateError, match="stale or reordered"):
        registry.begin(tuple(reversed(handles)), rows, (2, 3), authority=IDENTITY)
    forged = (replace(handles[0]), handles[1])
    with pytest.raises(ASTLiveStateError, match="stale or reordered"):
        registry.begin(forged, rows, (2, 3), authority=IDENTITY)
    lease = registry.begin(handles, rows, (2, 3), authority=IDENTITY)
    with pytest.raises(ASTLiveStateError, match="unavailable"):
        _ = registry.handles
    with pytest.raises(ASTLiveStateError, match="active lease"):
        registry.begin(handles, rows, (2, 3), authority=IDENTITY)
    lease.abort(authority=IDENTITY)
    assert registry.handles == handles


def test_registered_membership_uses_identity_not_cache_equality():
    class EqualDeclaredKV(KVCache):
        cache_operations_contract = KV_OPERATIONS

        def __eq__(self, other):
            return True

    rows = _rows()
    registry = ASTLiveLaneStateRegistry(rows, (2, 3), authority=IDENTITY)
    handles = registry.handles
    replacement = EqualDeclaredKV()
    replacement.keys = rows[0][0].keys
    replacement.values = rows[0][0].values
    replacement.offset = rows[0][0].offset
    lookalike_rows = ((replacement, rows[0][1]), rows[1])

    with pytest.raises(ASTLiveStateError, match="membership drifted"):
        registry.begin(handles, lookalike_rows, (2, 3), authority=IDENTITY)
    assert registry.handles == handles


def test_prepare_publish_advances_one_generation_and_invalidates_old_handles():
    (
        _descriptor,
        result,
        base,
        successor,
        registry,
        delegate,
        coordinator,
        accepted,
    ) = _coordinator()
    old_handles = tuple(registry._state.handles)

    prepared = coordinator.prepare(accepted, authority=IDENTITY)
    assert prepared.successor_rows == tuple(tuple(row) for row in successor)
    published = coordinator.publish(authority=IDENTITY)

    assert delegate.prepared == delegate.published == 1
    assert registry.epoch == 1
    assert published.result is result
    assert tuple(handle.generation for handle in published.handles) == (1, 1)
    assert tuple(handle.position for handle in published.handles) == (3, 5)
    with pytest.raises(ASTLiveStateError, match="stale"):
        registry.begin(old_handles, base, (3, 5), authority=IDENTITY)
    lease = registry.begin(published.handles, base, (3, 5), authority=IDENTITY)
    lease.abort(authority=IDENTITY)
    with pytest.raises(ASTLiveStateError, match="already closed"):
        coordinator.abort()


@pytest.mark.parametrize("publication_rows", ["foreign", "mixed"])
def test_publication_destination_rejects_foreign_or_mixed_rows(publication_rows):
    (
        _descriptor,
        _result,
        base,
        successor,
        registry,
        delegate,
        coordinator,
        accepted,
    ) = _coordinator()
    if publication_rows == "foreign":
        delegate.publication_rows = _rows((3, 5))
    else:
        delegate.publication_rows = (base[0], successor[1])

    with pytest.raises(ASTLiveStateError, match="authority-bound destination"):
        coordinator.prepare(accepted, authority=IDENTITY)
    assert delegate.published == 0
    assert registry.epoch == 0
    assert registry.handles


def test_successor_transfer_publication_uses_exact_successor_membership():
    descriptor = FlatSegmentDescriptor.from_query_lengths((2, 3))
    result = _result(descriptor)
    positions = (2, 3)
    accepted = (1, 2)
    successor_positions = (3, 5)
    base = _rows(positions)
    successor = _rows(successor_positions)
    registry = ASTLiveLaneStateRegistry(base, positions, authority=IDENTITY)
    delegate = _Delegate(
        base,
        successor,
        descriptor,
        result,
        publication_mode="transfer_successor",
        publication_rows=successor,
    )

    def transfer_publish(_prepared):
        delegate.published += 1

    delegate.publish_commit = transfer_publish
    coordinator = ASTPhysicalCommitCoordinator(
        registry,
        registry.handles,
        base,
        positions,
        descriptor=descriptor,
        result_plan=_result_plan(result),
        delegate=delegate,
        authority=IDENTITY,
    )
    coordinator.execute(authority=IDENTITY)
    coordinator.prepare(accepted, authority=IDENTITY)
    published = coordinator.publish(authority=IDENTITY)

    assert all(
        published.publication_rows[lane][index] is successor[lane][index]
        for lane in range(len(successor))
        for index in range(len(successor[lane]))
    )
    lease = registry.begin(
        published.handles,
        successor,
        successor_positions,
        authority=IDENTITY,
    )
    lease.abort(authority=IDENTITY)


def test_lease_is_acquired_before_private_execution():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    result = _result(descriptor)
    base = _rows((2, 3))
    successor = _rows((3, 5))
    registry = ASTLiveLaneStateRegistry(base, (2, 3), authority=IDENTITY)
    delegate = _Delegate(base, successor, descriptor, result)
    coordinator = ASTPhysicalCommitCoordinator(
        registry,
        registry.handles,
        base,
        (2, 3),
        descriptor=descriptor,
        result_plan=_result_plan(result),
        delegate=delegate,
        authority=IDENTITY,
    )
    with pytest.raises(ASTLiveStateError, match="unavailable"):
        _ = registry.handles
    assert delegate.executed == 0
    assert coordinator.execute(authority=IDENTITY) is result
    assert delegate.executed == 1
    assert delegate.result_plan_seen is coordinator._result_plan
    coordinator.abort()


def test_prepare_failure_aborts_without_generation_or_base_mutation():
    (
        _descriptor,
        _result,
        base,
        _successor,
        registry,
        delegate,
        coordinator,
        accepted,
    ) = _coordinator()
    original = tuple(registry._state.handles)
    delegate.fail_prepare = True
    with pytest.raises(RuntimeError, match="prepare failed"):
        coordinator.prepare(accepted, authority=IDENTITY)
    assert registry.epoch == 0
    assert registry.handles == original
    assert delegate.aborted == 1
    assert tuple(row[0].offset for row in base) == (2, 3)


def test_publish_failure_delegate_rolls_back_and_registry_remains_valid():
    (
        _descriptor,
        _result,
        base,
        _successor,
        registry,
        delegate,
        coordinator,
        accepted,
    ) = _coordinator()
    original = tuple(registry._state.handles)
    coordinator.prepare(accepted, authority=IDENTITY)
    delegate.fail_publish = True
    with pytest.raises(RuntimeError, match="publish failed"):
        coordinator.publish(authority=IDENTITY)
    assert registry.epoch == 0
    assert registry.handles == original
    assert tuple(row[0].offset for row in base) == (2, 3)
    lease = registry.begin(original, base, (2, 3), authority=IDENTITY)
    lease.abort(authority=IDENTITY)


def test_base_mutation_during_prepare_poison_registry_on_abort():
    (
        _descriptor,
        _result,
        _base,
        _successor,
        registry,
        delegate,
        coordinator,
        accepted,
    ) = _coordinator()
    delegate.mutate_base_during_prepare = True
    with pytest.raises(ASTLiveStateError, match="position|drifted"):
        coordinator.prepare(accepted, authority=IDENTITY)
    assert registry.receipt()["poisoned"] is True
    with pytest.raises(ASTLiveStateError, match="poisoned"):
        _ = registry.handles


def test_prepared_envelope_rejects_descriptor_result_and_position_drift():
    (
        _descriptor,
        _result,
        _base,
        _successor,
        registry,
        delegate,
        coordinator,
        accepted,
    ) = _coordinator()

    def bad_prepare(values, *, descriptor, result, prepare_authority):
        positions = tuple(row[0].offset for row in delegate.successor_rows)
        return prepare_authority.seal(
            successor_rows=delegate.successor_rows,
            publication_rows=delegate.base_rows,
            publication_mode="in_place_base",
            accepted_lengths=values,
            successor_positions=(999, *positions[1:]),
            descriptor_geometry_sha256=descriptor.geometry_sha256,
            result_manifest_sha256=result.manifest_sha256,
            delegate_token=object(),
        )

    delegate.prepare_commit = bad_prepare
    with pytest.raises(ASTLiveStateError, match="position"):
        coordinator.prepare(accepted, authority=IDENTITY)
    assert registry.epoch == 0
    assert registry.receipt()["poisoned"] is False


def test_copied_and_foreign_prepared_envelopes_fail_before_publication():
    (
        _descriptor,
        _result,
        _base,
        _successor,
        registry,
        delegate,
        coordinator,
        accepted,
    ) = _coordinator()
    original_prepare = delegate.prepare_commit

    def copied_prepare(values, **kwargs):
        return replace(original_prepare(values, **kwargs))

    delegate.prepare_commit = copied_prepare
    with pytest.raises(ASTLiveStateError, match="foreign, copied, or replayed"):
        coordinator.prepare(accepted, authority=IDENTITY)
    assert delegate.published == 0
    assert registry.handles

    *_, first_coordinator, first_accepted = _coordinator()
    foreign = first_coordinator.prepare(first_accepted, authority=IDENTITY)
    (
        _descriptor,
        _result,
        _base,
        _successor,
        second_registry,
        second_delegate,
        second_coordinator,
        second_accepted,
    ) = _coordinator()
    second_delegate.prepare_commit = lambda *args, **kwargs: foreign
    with pytest.raises(ASTLiveStateError, match="foreign, copied, or replayed"):
        second_coordinator.prepare(second_accepted, authority=IDENTITY)
    assert second_delegate.published == 0
    assert second_registry.handles
    first_coordinator.abort()


def test_successor_membership_drift_after_prepare_fails_before_publish():
    (
        _descriptor,
        _result,
        _base,
        successor,
        registry,
        delegate,
        coordinator,
        accepted,
    ) = _coordinator()
    coordinator.prepare(accepted, authority=IDENTITY)
    successor[0][0].keys = mx.array(successor[0][0].keys)
    with pytest.raises(ASTLiveStateError, match="membership drifted"):
        coordinator.publish(authority=IDENTITY)
    assert delegate.published == 0
    assert registry.handles


def test_public_successor_rejects_live_speculation_but_segmented_check_is_distinct():
    (
        _descriptor,
        _result,
        _base,
        successor,
        registry,
        _delegate,
        coordinator,
        accepted,
    ) = _coordinator()
    successor[0][1].start_speculation()
    # The segmented boundary admits an open exact rollback transaction.
    coordinator._lease.validate_segmented_successor(
        successor, (3, 5), authority=IDENTITY
    )
    # A final/public successor must not carry it across publication.
    with pytest.raises(Exception, match="speculation transaction is active"):
        coordinator.prepare(accepted, authority=IDENTITY)
    assert registry.epoch == 0


def test_host_semantic_and_member_drift_are_detected_under_lease():
    rows = _rows()
    registry = ASTLiveLaneStateRegistry(rows, (2, 3), authority=IDENTITY)
    handles = registry.handles
    active = registry.begin(handles, rows, (2, 3), authority=IDENTITY)
    rows[0][1].lengths = mx.array([99])
    with pytest.raises(ASTLiveStateError, match="drifted"):
        active.validate_base(authority=IDENTITY)
    with pytest.raises(ASTLiveStateError, match="drifted"):
        active.abort(authority=IDENTITY)
    assert registry.receipt()["poisoned"] is True


def test_manual_abort_is_idempotence_guarded_and_preserves_base():
    (
        _descriptor,
        _result,
        base,
        _successor,
        registry,
        _delegate,
        coordinator,
        _accepted,
    ) = _coordinator()
    coordinator.abort()
    assert registry.epoch == 0
    assert tuple(row[0].offset for row in base) == (2, 3)
    with pytest.raises(ASTLiveStateError, match="already closed"):
        coordinator.abort()


def test_bool_acceptance_and_double_prepare_fail_closed():
    (
        _descriptor,
        _result,
        _base,
        _successor,
        registry,
        _delegate,
        coordinator,
        accepted,
    ) = _coordinator()
    with pytest.raises(ASTLiveStateError, match="integer"):
        coordinator.prepare((True, accepted[1]), authority=IDENTITY)
    assert registry.epoch == 0

    *_, coordinator, accepted = _coordinator()
    coordinator.prepare(accepted, authority=IDENTITY)
    with pytest.raises(ASTLiveStateError, match="already prepared"):
        coordinator.prepare(accepted, authority=IDENTITY)
    coordinator.abort()


def test_non_atomic_delegate_failure_poison_is_explicit():
    (
        _descriptor,
        _result,
        base,
        _successor,
        registry,
        delegate,
        coordinator,
        accepted,
    ) = _coordinator()

    def broken_publish(prepared):
        base[0][0].offset += 1
        raise RuntimeError("irrevocable partial publish")

    def broken_abort():
        raise RuntimeError("delegate could not roll back")

    delegate.publish_commit = broken_publish
    delegate.abort = broken_abort
    coordinator.prepare(accepted, authority=IDENTITY)
    with pytest.raises(RuntimeError, match="irrevocable"):
        coordinator.publish(authority=IDENTITY)
    assert registry.receipt()["poisoned"] is True
    with pytest.raises(ASTLiveStateError, match="poisoned"):
        _ = registry.handles


def test_authority_drift_before_publish_prevents_delegate_publication():
    (
        _descriptor,
        _result,
        _base,
        _successor,
        registry,
        delegate,
        coordinator,
        accepted,
    ) = _coordinator()
    coordinator.prepare(accepted, authority=IDENTITY)
    drifted = replace(IDENTITY, device_manifest_sha256="8" * 64)
    with pytest.raises(ASTLiveStateError, match="authority drifted"):
        coordinator.publish(authority=drifted)
    assert delegate.published == 0
    assert registry.epoch == 0
    assert registry.handles


def test_authority_and_positions_reject_bool_or_unbound_values():
    with pytest.raises(ASTLiveStateError):
        ASTLiveStateAuthority(
            artifact="bad space",
            session_scope_sha256=SHA,
            owner_identity_sha256=SHA,
            provider_manifest_sha256=SHA,
            source_manifest_sha256=SHA,
            build_manifest_sha256=SHA,
            device_manifest_sha256=SHA,
            max_lanes=1,
            max_caches_per_lane=1,
            max_segment_rows=1,
            max_total_rows=1,
            max_physical_rows=1,
            required_plane_kinds=("attention_kv",),
            publication_modes=("in_place_base",),
            result_storage_classes=("flat_logical_rows",),
        )
    rows = _rows((1,))
    with pytest.raises(ASTLiveStateError, match="integer"):
        ASTLiveLaneStateRegistry(rows, (True,), authority=IDENTITY)


def test_authority_drift_fails_before_prepare_and_releases_lease():
    (
        _descriptor,
        _result,
        _base,
        _successor,
        registry,
        delegate,
        coordinator,
        accepted,
    ) = _coordinator()
    drifted = replace(IDENTITY, source_manifest_sha256="7" * 64)
    with pytest.raises(ASTLiveStateError, match="authority drifted"):
        coordinator.prepare(accepted, authority=drifted)
    assert delegate.prepared == 0
    assert delegate.aborted == 1
    assert registry.epoch == 0
    assert registry.handles


def test_successor_must_be_structurally_disjoint_from_registered_base():
    (
        _descriptor,
        _result,
        base,
        _successor,
        registry,
        delegate,
        coordinator,
        accepted,
    ) = _coordinator()
    delegate.successor_rows = base
    with pytest.raises(ASTLiveStateError, match="disjoint"):
        coordinator.prepare(accepted, authority=IDENTITY)
    assert registry.epoch == 0
    assert registry.handles


def test_descriptor_lane_mismatch_fails_before_lease():
    rows = _rows()
    registry = ASTLiveLaneStateRegistry(rows, (2, 3), authority=IDENTITY)
    handles = registry.handles
    descriptor = FlatSegmentDescriptor.from_query_lengths((1,))
    result = _result(descriptor)
    successor = (_row(3, 0),)
    delegate = _Delegate(rows, successor, descriptor, result)
    with pytest.raises(ASTLiveStateError, match="descriptor lanes"):
        ASTPhysicalCommitCoordinator(
            registry,
            handles,
            rows,
            (2, 3),
            descriptor=descriptor,
            result_plan=_result_plan(result),
            delegate=delegate,
            authority=IDENTITY,
        )
    assert registry.handles == handles


def test_descriptor_capacity_fails_before_lease_or_delegate_work():
    rows = _rows((2, 3))
    registry = ASTLiveLaneStateRegistry(rows, (2, 3), authority=IDENTITY)
    handles = registry.handles
    descriptor = FlatSegmentDescriptor.from_query_lengths((5, 5))
    result = _result(descriptor)
    delegate = _Delegate(rows, _rows((3, 5)), descriptor, result)
    with pytest.raises(ASTLiveStateError, match="exceeds"):
        ASTPhysicalCommitCoordinator(
            registry,
            handles,
            rows,
            (2, 3),
            descriptor=descriptor,
            result_plan=_result_plan(result),
            delegate=delegate,
            authority=IDENTITY,
        )
    assert delegate.prepared == delegate.published == 0
    assert registry.handles == handles


def test_result_storage_authority_fails_before_lease_or_delegate_work():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    result = _result(descriptor, rectangular=True)
    positions = (2, 3)
    base = _rows(positions)
    delegate = _Delegate(base, _rows((3, 4)), descriptor, result)
    flat_only = replace(
        IDENTITY,
        result_storage_classes=("flat_logical_rows", "flat_row_pairs"),
    )
    # Registry and coordinator authority must be identical, so construct the
    # owner under the restricted authority rather than weakening validation.
    registry = ASTLiveLaneStateRegistry(base, positions, authority=flat_only)
    handles = registry.handles

    with pytest.raises(ASTLiveStateError, match="storage class is not authorized"):
        ASTPhysicalCommitCoordinator(
            registry,
            handles,
            base,
            positions,
            descriptor=descriptor,
            result_plan=_result_plan(result),
            delegate=delegate,
            authority=flat_only,
        )
    assert delegate.executed == 0
    assert registry.handles == handles


def test_delegate_result_must_match_pre_execution_plan():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2))
    planned = _result(descriptor)
    returned = _result(descriptor, rectangular=True)
    positions = (2, 3)
    base = _rows(positions)
    successor = _rows((3, 4))
    registry = ASTLiveLaneStateRegistry(base, positions, authority=IDENTITY)
    delegate = _Delegate(base, successor, descriptor, returned)
    coordinator = ASTPhysicalCommitCoordinator(
        registry,
        registry.handles,
        base,
        positions,
        descriptor=descriptor,
        result_plan=_result_plan(planned),
        delegate=delegate,
        authority=IDENTITY,
    )

    with pytest.raises(ASTLiveStateError, match="differs from its plan"):
        coordinator.execute(authority=IDENTITY)
    assert delegate.executed == 1
    assert registry.handles


def test_rectangular_physical_capacity_fails_before_lease_or_execution():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2, 3, 4))
    logits = mx.zeros((4, 5, 3))
    features = mx.zeros((4, 5, 2))
    result = ASTPhysicalResult(
        descriptor,
        logits,
        features,
        storage="rectangular_rows",
        invocation_count=1,
    )
    positions = (1, 1, 1, 1)
    base = _rows(positions)
    successor = _rows((2, 2, 2, 2))
    registry = ASTLiveLaneStateRegistry(base, positions, authority=IDENTITY)
    delegate = _Delegate(base, successor, descriptor, result)
    handles = registry.handles
    with pytest.raises(ASTLiveStateError, match="planned physical rows exceed"):
        ASTPhysicalCommitCoordinator(
            registry,
            handles,
            base,
            positions,
            descriptor=descriptor,
            result_plan=_result_plan(result),
            delegate=delegate,
            authority=IDENTITY,
        )
    assert delegate.executed == 0
    assert delegate.prepared == delegate.published == 0
    assert registry.handles == handles


@pytest.mark.parametrize("field", ["geometry", "result"])
def test_prepared_identity_drift_aborts_before_publish(field):
    (
        _descriptor,
        _result,
        _base,
        _successor,
        registry,
        delegate,
        coordinator,
        accepted,
    ) = _coordinator()

    def drifted_prepare(values, *, descriptor, result, prepare_authority):
        key = (
            "descriptor_geometry_sha256"
            if field == "geometry"
            else "result_manifest_sha256"
        )
        identities = {
            "descriptor_geometry_sha256": descriptor.geometry_sha256,
            "result_manifest_sha256": result.manifest_sha256,
        }
        identities[key] = "f" * 64
        return prepare_authority.seal(
            successor_rows=delegate.successor_rows,
            publication_rows=delegate.base_rows,
            publication_mode="in_place_base",
            accepted_lengths=values,
            successor_positions=tuple(row[0].offset for row in delegate.successor_rows),
            delegate_token=object(),
            **identities,
        )

    delegate.prepare_commit = drifted_prepare
    with pytest.raises(ASTLiveStateError, match="geometry|identity"):
        coordinator.prepare(accepted, authority=IDENTITY)
    assert delegate.published == 0
    assert registry.epoch == 0
    assert registry.handles
