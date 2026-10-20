"""Private-until-publish hybrid ownership for flat AST execution.

The existing :mod:`hybrid_verify_rows` transaction is deliberately reused on
deep-copied rows.  Its padded compute views therefore never become
authoritative state: accepted KV and recurrent successors cross into the live
rows only in ``publish_commit`` after the external ordinary oracle passes.
The native physical v2 transaction instead opens one private B1 transaction
per lane and never constructs those padded views. This module is not wired
into serving and does not authorize a backend.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import mlx.core as mx
import numpy as np

from .ast_execution_capability import state_contract_from_rows
from .flat_segmented_ast import (
    ABORT_LAW,
    BEGIN_LAW,
    COMMIT_FAILURE_LAW,
    COMMIT_LAW,
    DESCRIPTOR_SCHEMA,
    LAYOUT_CLASS,
    METAL_EXECUTION_SCOPE,
    METAL_OWNER_PROTOCOL,
    NATIVE_AGGREGATE_COMMIT_LAW,
    NATIVE_EXECUTION_SCOPE,
    NATIVE_LANE_STAGING_LAW,
    NATIVE_OWNER_CAPABILITY_SCHEMA,
    NATIVE_OWNER_PROTOCOL,
    NATIVE_PHYSICAL_LAYOUT,
    NATIVE_PHYSICAL_ROWS_LAW,
    OWNER_CAPABILITY_SCHEMA,
    POSITION_BASE_AUTHORITY,
    FlatSegmentDescriptor,
    PreparedFlatSegmentCommit,
    validate_metal_flat_segment_admission,
    validate_native_flat_segment_admission,
)
from .flat_segmented_kv import METAL_PROBE_HARD_LIMITS
from .hybrid_verify_rows import HybridVerifyRows
from .models.cache import ArraysCache, KVCache

_PACKAGE = Path(__file__).resolve().parents[1]
_PACKAGE_PREFIX = "src/mlx2/"
_SOURCE_PATHS = (
    "src/mlx2/runtime/flat_segmented_hybrid.py",
    "src/mlx2/runtime/flat_segmented_ast.py",
    "src/mlx2/runtime/flat_segmented_kv.py",
    "src/mlx2/runtime/hybrid_verify_rows.py",
    "src/mlx2/runtime/hybrid_speculative.py",
    "src/mlx2/runtime/segmented_batch_cache.py",
    "src/mlx2/runtime/segmented_plain_kv.py",
    "src/mlx2/runtime/ast_execution_capability.py",
    "src/mlx2/runtime/cache_operations.py",
    "src/mlx2/runtime/models/cache.py",
    "src/mlx2/runtime/models/import_env.py",
    "src/mlx2/runtime/models/invariant_prefill.py",
    "src/mlx2/runtime/models/moe_nax_gather.py",
    "src/mlx2/runtime/models/qsdpa_verify_metal.py",
    "src/mlx2/runtime/models/qwen4_exp.py",
    "src/mlx2/runtime/models/qwen4_qsa_indexed.py",
    "src/mlx2/runtime/models/served_exp.py",
    "src/mlx2/runtime/paged_kv_cache.py",
    "src/mlx2/runtime/prefill_checkpoint_rows.py",
    "src/mlx2/runtime/round_levers.py",
)
_NATIVE_SOURCE_PATHS = (
    "src/mlx2/runtime/flat_segmented_hybrid.py",
    "src/mlx2/runtime/flat_segmented_ast.py",
    "src/mlx2/runtime/ast_execution_capability.py",
    "src/mlx2/runtime/hybrid_verify_rows.py",
    "src/mlx2/runtime/hybrid_speculative.py",
    "src/mlx2/runtime/cache_operations.py",
    "src/mlx2/runtime/models/cache.py",
)


_BOUNDARY_FACTORY = object()


class HybridLiveStateBoundary:
    """One target-issued KV/GDN boundary for stable live cache containers.

    Unlike :class:`GDNBoundaryCheckpoint`, this descriptor deliberately names
    cache containers that remain stable while their contents are atomically
    replaced.  A successful publication invalidates the prior descriptor and
    installs a new generation; failed or aborted publication leaves it valid.
    The artifact revision and target issuer are fixed at mint time.
    """

    __slots__ = (
        "revision",
        "lane_id",
        "position",
        "generation",
        "caches",
        "issuer_id",
        "state_sha256",
        "member_ids",
        "_owner_id",
        "_valid",
    )

    def __init__(
        self,
        *,
        revision,
        lane_id,
        position,
        generation,
        row,
        issuer_id,
        _authority=None,
    ):
        if _authority is not _BOUNDARY_FACTORY:
            raise TypeError("hybrid live boundaries are target-issued")
        if not isinstance(revision, str) or not revision:
            raise ValueError("hybrid boundary revision is unavailable")
        if any(
            type(value) is not int or value < 0
            for value in (lane_id, position, generation, issuer_id)
        ):
            raise ValueError("hybrid boundary integer fields differ")
        row = tuple(row)
        caches = tuple(cache for cache in row if type(cache) is ArraysCache)
        if not caches or any(type(cache) is not ArraysCache for cache in caches):
            raise TypeError("hybrid boundary requires exact recurrent caches")
        if len({id(cache) for cache in caches}) != len(caches):
            raise ValueError("hybrid boundary recurrent caches are aliased")
        self.revision = revision
        self.lane_id = lane_id
        self.position = position
        self.generation = generation
        self.caches = caches
        self.issuer_id = issuer_id
        self.state_sha256 = _lane_state_sha256(row)
        self.member_ids = tuple(id(cache) for cache in row)
        self._owner_id = None
        self._valid = True

    @classmethod
    def _mint(
        cls, *, revision, lane_id, position, generation, row, issuer_id
    ):
        return cls(
            revision=revision,
            lane_id=lane_id,
            position=position,
            generation=generation,
            row=row,
            issuer_id=issuer_id,
            _authority=_BOUNDARY_FACTORY,
        )

    @property
    def valid(self):
        return self._valid

    def _bind(self, owner):
        if not self._valid or self._owner_id is not None:
            raise RuntimeError("hybrid boundary is stale or already bound")
        self._owner_id = id(owner)

    def _successor(self, *, owner, row, position):
        if not self._valid or self._owner_id != id(owner):
            raise RuntimeError("hybrid boundary is stale or unbound")
        successor = type(self)._mint(
            revision=self.revision,
            lane_id=self.lane_id,
            position=position,
            generation=self.generation + 1,
            row=row,
            issuer_id=self.issuer_id,
        )
        return successor

    def _invalidate(self, owner):
        if self._owner_id != id(owner):
            raise RuntimeError("hybrid boundary owner differs")
        self._valid = False

    def _discard_successor(self, owner):
        if self._owner_id not in (None, id(owner)):
            raise RuntimeError("hybrid successor boundary owner differs")
        self._valid = False


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _source_hashes() -> dict[str, str]:
    result = {}
    for name in _SOURCE_PATHS:
        path = _PACKAGE / name[len(_PACKAGE_PREFIX) :]
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _native_source_hashes() -> dict[str, str]:
    result = {}
    for name in _NATIVE_SOURCE_PATHS:
        path = _PACKAGE / name[len(_PACKAGE_PREFIX) :]
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _array_receipt(value: mx.array | None):
    if value is None:
        return None
    if not isinstance(value, mx.array):
        raise TypeError("hybrid state contains a non-MLX array")
    mx.eval(value)
    host = np.asarray(value.reshape(-1).view(mx.uint8))
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(host.tobytes(order="C")).hexdigest(),
    }


def _optional_vector(value):
    if value is None:
        return None
    if not isinstance(value, mx.array):
        raise TypeError("hybrid cache metadata must be an MLX array")
    return _array_receipt(value)


def canonical_hybrid_state(rows) -> tuple[tuple[dict[str, object], ...], ...]:
    """Token-free exact KV plus recurrent-state receipt."""

    result = []
    for row in rows:
        layers = []
        for cache in row:
            if type(cache) is KVCache:
                if cache.keys is None:
                    keys = values = None
                else:
                    keys, values = cache.keys_and_values()
                    keys, values = _array_receipt(keys), _array_receipt(values)
                layers.append(
                    {
                        "kind": "kv",
                        "offset": int(cache.offset),
                        "keys": keys,
                        "values": values,
                    }
                )
            elif type(cache) is ArraysCache:
                if (
                    cache.speculating
                    or cache._rollbacks
                    or cache._rollback_position
                    or cache._rollback_positions is not None
                ):
                    raise RuntimeError("authoritative recurrent state is speculative")
                layers.append(
                    {
                        "kind": "gdn_recurrent",
                        "arrays": [_array_receipt(value) for value in cache.cache],
                        "lengths": _optional_vector(cache.lengths),
                        "left_padding": _optional_vector(cache.left_padding),
                    }
                )
            else:
                raise TypeError("hybrid flat state requires exact KVCache/ArraysCache")
        result.append(tuple(layers))
    return tuple(result)


def _lane_state_sha256(row) -> str:
    return hashlib.sha256(_canonical(canonical_hybrid_state((tuple(row),))[0])).hexdigest()


def _boundary_receipt(boundaries):
    return tuple(
        {
            "revision": boundary.revision,
            "lane_id": boundary.lane_id,
            "position": boundary.position,
            "generation": boundary.generation,
            "issuer_id": boundary.issuer_id,
            "state_sha256": boundary.state_sha256,
            "valid": boundary.valid,
        }
        for boundary in boundaries
    )


def canonical_bound_hybrid_state(rows, boundaries):
    """Exact state plus the common per-lane KV/GDN boundary identity."""

    return {
        "cache_state": canonical_hybrid_state(rows),
        "state_boundaries": _boundary_receipt(boundaries),
    }


def successor_bound_hybrid_state(rows, boundaries, accepted_lengths):
    """Expected cache state plus successor boundary metadata, without minting."""

    receipts = []
    for boundary, row, accepted in zip(
        boundaries, rows, accepted_lengths, strict=True
    ):
        receipts.append(
            {
                "revision": boundary.revision,
                "lane_id": boundary.lane_id,
                "position": boundary.position + int(accepted),
                "generation": boundary.generation + 1,
                "issuer_id": boundary.issuer_id,
                "state_sha256": _lane_state_sha256(row),
                "valid": True,
            }
        )
    return {
        "cache_state": canonical_hybrid_state(rows),
        "state_boundaries": tuple(receipts),
    }


def successor_hybrid_boundaries(boundaries, rows, accepted_lengths, *, owner):
    result = []
    for boundary, row, accepted in zip(
        boundaries, rows, accepted_lengths, strict=True
    ):
        result.append(
            boundary._successor(
                owner=owner,
                position=boundary.position + int(accepted),
                row=row,
            )
        )
    return tuple(result)


def _validate_identity(identity: Mapping[str, str]) -> dict[str, str]:
    fields = {"source_revision", "build", "device", "artifact"}
    if not isinstance(identity, Mapping) or set(identity) != fields or any(
        not isinstance(identity[name], str) or not identity[name] for name in fields
    ):
        raise ValueError("hybrid flat identity fields differ")
    return {name: identity[name] for name in sorted(fields)}


class MetalFlatSegmentedHybridRows:
    """Exclusive exact hybrid owner for one direct Qwen3.8 Metal probe."""

    execution_scope = METAL_EXECUTION_SCOPE
    owner_protocol = METAL_OWNER_PROTOCOL
    hard_limits = METAL_PROBE_HARD_LIMITS

    def __init__(
        self,
        rows,
        *,
        identity: Mapping[str, str],
        boundary_checkpoints,
        target_executor,
    ):
        self._rows = tuple(tuple(row) for row in rows)
        self._boundaries = tuple(boundary_checkpoints)
        self.identity = _validate_identity(identity)
        self._target_executor_id = id(target_executor)
        self.revision = 0
        self._active = None
        self._validate_rows(
            self._rows,
            self._boundaries,
            artifact=self.identity["artifact"],
            issuer_id=self._target_executor_id,
        )
        for boundary in self._boundaries:
            boundary._bind(self)
        state = state_contract_from_rows(self._rows)
        expected = {
            "state_plane_kinds": ["attention_kv", "gdn_recurrent"],
            "state_families": ["kv", "recurrent"],
            "state_contract_fingerprints": [
                "5cce324cf8b59feb02ac4362c53872b62b19d06fc9e0ceab478d2a867ea88a2e",
                "8491ce6cf23f5a337f6ec7a052bdfb8cd036c01ecf2b6e51e773d6798a764cbe",
            ],
            "state_ownership": "independent_request_private_b1_lanes",
        }
        if state != expected:
            raise ValueError("Qwen3.8 hybrid state contract differs")
        self._state_contract = state

    @property
    def rows(self):
        return self._rows

    @property
    def boundaries(self):
        return self._boundaries

    @staticmethod
    def _validate_rows(rows, boundaries, *, artifact, issuer_id) -> None:
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("hybrid flat rows require one common nonempty layout")
        if (
            len(boundaries) != len(rows)
            or any(type(boundary) is not HybridLiveStateBoundary for boundary in boundaries)
            or len({boundary.lane_id for boundary in boundaries}) != len(boundaries)
            or any(boundary.revision != artifact for boundary in boundaries)
            or any(boundary.issuer_id != issuer_id for boundary in boundaries)
            or any(not boundary.valid for boundary in boundaries)
        ):
            raise ValueError("hybrid rows require valid target-issued artifact boundaries")
        leaves = [cache for row in rows for cache in row]
        if len({id(cache) for cache in leaves}) != len(leaves):
            raise ValueError("hybrid flat rows require independent cache owners")
        topology = tuple(type(cache) for cache in rows[0])
        if KVCache not in topology or ArraysCache not in topology:
            raise ValueError("hybrid flat rows require KV and GDN planes")
        for row, boundary in zip(rows, boundaries, strict=True):
            if tuple(type(cache) for cache in row) != topology:
                raise ValueError("hybrid flat lane topology differs")
            recurrent = tuple(cache for cache in row if type(cache) is ArraysCache)
            if len(recurrent) != len(boundary.caches) or any(
                cache is not bound
                for cache, bound in zip(recurrent, boundary.caches, strict=True)
            ):
                raise ValueError("hybrid GDN lineage does not match its boundary")
            if tuple(id(cache) for cache in row) != boundary.member_ids:
                raise ValueError("hybrid plane membership differs from its boundary")
            if _lane_state_sha256(row) != boundary.state_sha256:
                raise ValueError("hybrid lane state differs from its boundary")
            kv_offsets = [cache.offset for cache in row if type(cache) is KVCache]
            if not kv_offsets or any(offset != boundary.position for offset in kv_offsets):
                raise ValueError("hybrid KV and GDN boundary positions differ")
            for cache in row:
                if type(cache) is KVCache:
                    if type(cache.offset) is not int or cache.offset < 0:
                        raise ValueError("hybrid KV offsets must be host integers")
                    if cache.keys is None:
                        if cache.values is not None or cache.offset:
                            raise ValueError("hybrid empty KV state is incomplete")
                    elif (
                        cache.values is None
                        or cache.keys.shape != cache.values.shape
                        or cache.keys.dtype != cache.values.dtype
                        or cache.keys.ndim != 4
                        or cache.keys.shape[0] != 1
                        or cache.offset > cache.keys.shape[2]
                    ):
                        raise ValueError("hybrid KV geometry differs")
                else:
                    if (
                        cache.speculating
                        or cache.batch_size != 1
                        or len(cache.cache) != 2
                        or any(not isinstance(value, mx.array) for value in cache.cache)
                        or cache._rollbacks
                        or cache._rollback_position
                        or cache._rollback_positions is not None
                    ):
                        raise ValueError("hybrid recurrent state is not clean B1")

    def ast_flat_segment_membership_identity(self):
        return tuple(
            tuple([*(id(cache) for cache in row), id(boundary)])
            for row, boundary in zip(self._rows, self._boundaries, strict=True)
        )

    def ast_flat_segment_state_identity_sha256(self) -> str:
        return hashlib.sha256(
            _canonical(canonical_bound_hybrid_state(self._rows, self._boundaries))
        ).hexdigest()

    @property
    def source_hashes(self):
        return _source_hashes()

    @property
    def state_contract_fingerprints(self):
        return tuple(self._state_contract["state_contract_fingerprints"])

    @property
    def semantic_contract_sha256(self) -> str:
        semantics = {
            **self._state_contract,
            "layout_classes": [LAYOUT_CLASS],
            "transaction_operations": [
                "begin_flat",
                "prepare_commit",
                "publish_commit",
                "abort",
            ],
            "begin_law": BEGIN_LAW,
            "commit_law": COMMIT_LAW,
            "commit_failure_law": COMMIT_FAILURE_LAW,
            "abort_law": ABORT_LAW,
            "rollback_law": "private_kv_and_recurrent_snapshot_restore",
            "position_base_authority": POSITION_BASE_AUTHORITY,
            "lane_state_boundary": (
                "target_issued_artifact_lane_generation_bound_gdn_at_every_kv_offset"
            ),
            "hard_limits": self.hard_limits,
        }
        return hashlib.sha256(_canonical(semantics)).hexdigest()

    def ast_flat_segment_transaction_contract(self) -> dict:
        return {
            "schema": OWNER_CAPABILITY_SCHEMA,
            "protocol": self.owner_protocol,
            "execution_scope": self.execution_scope,
            "descriptor_schema": DESCRIPTOR_SCHEMA,
            "layout_classes": [LAYOUT_CLASS],
            **self._state_contract,
            "owner_semantic_contract_sha256": self.semantic_contract_sha256,
            "transaction_operations": {
                "begin_flat": True,
                "prepare_commit": True,
                "publish_commit": True,
                "abort": True,
            },
            "begin_law": BEGIN_LAW,
            "commit_law": COMMIT_LAW,
            "commit_failure_law": COMMIT_FAILURE_LAW,
            "abort_law": ABORT_LAW,
            "rollback_law": "private_kv_and_recurrent_snapshot_restore",
            "position_base_authority": POSITION_BASE_AUTHORITY,
            "hard_limits": dict(self.hard_limits),
            "source_hashes": self.source_hashes,
            "identity": dict(self.identity),
        }

    def begin_flat(self, descriptor, *, target_executor, admission):
        if not isinstance(descriptor, FlatSegmentDescriptor):
            raise TypeError("hybrid begin_flat requires a validated descriptor")
        if self._active is not None:
            raise RuntimeError("hybrid flat membership is already leased")
        self._validate_rows(
            self.rows,
            self.boundaries,
            artifact=self.identity["artifact"],
            issuer_id=self._target_executor_id,
        )
        if id(target_executor) != self._target_executor_id:
            raise ValueError("hybrid owner target issuer differs")
        if descriptor.lane_count != len(self.rows):
            raise ValueError("hybrid descriptor lane count differs")
        validate_metal_flat_segment_admission(
            admission, self, target_executor, descriptor
        )
        validator = getattr(target_executor, "validate_flat_owner", None)
        if not callable(validator):
            raise TypeError("hybrid flat owner requires a compatible target")
        validator(self, descriptor)
        transaction = FlatSegmentedHybridTransaction(self, descriptor, admission)
        self._active = transaction
        return transaction

    def _install_boundaries(self, boundaries):
        boundaries = tuple(boundaries)
        if any(
            type(boundary) is not HybridLiveStateBoundary
            or not boundary.valid
            or boundary._owner_id is not None
            for boundary in boundaries
        ):
            raise RuntimeError("hybrid successor boundaries are not installable")
        for boundary in boundaries:
            boundary._bind(self)
        self._boundaries = boundaries


class NativeFlatSegmentedHybridRows(MetalFlatSegmentedHybridRows):
    """Model-neutral hybrid owner for the native physical v2 contract.

    The target still issues and binds every live state boundary.  This class
    adds no tensor implementation and is never selected by the scheduler.
    """

    def __init__(
        self,
        rows,
        *,
        identity: Mapping[str, str],
        boundary_checkpoints,
        target_executor,
        hard_limits: Mapping[str, int],
    ):
        fields = {"max_lanes", "max_segment_rows", "max_total_rows"}
        if (
            not isinstance(hard_limits, Mapping)
            or set(hard_limits) != fields
            or any(
                type(hard_limits[name]) is not int or hard_limits[name] < 1
                for name in fields
            )
        ):
            raise ValueError("native hybrid hard capacity differs")
        self.hard_limits = {name: hard_limits[name] for name in sorted(fields)}
        super().__init__(
            rows,
            identity=identity,
            boundary_checkpoints=boundary_checkpoints,
            target_executor=target_executor,
        )

    @property
    def native_semantic_contract_sha256(self) -> str:
        semantics = {
            **self._state_contract,
            "layout_classes": [LAYOUT_CLASS],
            "physical_layout": NATIVE_PHYSICAL_LAYOUT,
            "physical_rows_law": NATIVE_PHYSICAL_ROWS_LAW,
            "transaction_operations": [
                "begin_native_flat",
                "prepare_commit",
                "publish_commit",
                "abort",
            ],
            "lane_staging_law": NATIVE_LANE_STAGING_LAW,
            "aggregate_commit_law": NATIVE_AGGREGATE_COMMIT_LAW,
            "begin_law": BEGIN_LAW,
            "commit_law": COMMIT_LAW,
            "commit_failure_law": COMMIT_FAILURE_LAW,
            "abort_law": ABORT_LAW,
            "rollback_law": "private_kv_and_recurrent_snapshot_restore",
            "position_base_authority": POSITION_BASE_AUTHORITY,
            "lane_state_boundary": (
                "target_issued_artifact_lane_generation_bound_gdn_at_every_kv_offset"
            ),
            "hard_limits": self.hard_limits,
        }
        return hashlib.sha256(_canonical(semantics)).hexdigest()

    @property
    def native_source_hashes(self):
        return _native_source_hashes()

    def ast_native_flat_transaction_contract(self) -> dict:
        """Independent owner attestation for native logical-row execution."""

        return {
            "schema": NATIVE_OWNER_CAPABILITY_SCHEMA,
            "protocol": NATIVE_OWNER_PROTOCOL,
            "execution_scope": NATIVE_EXECUTION_SCOPE,
            "descriptor_schema": DESCRIPTOR_SCHEMA,
            "layout_classes": [LAYOUT_CLASS],
            **self._state_contract,
            "owner_semantic_contract_sha256": self.native_semantic_contract_sha256,
            "transaction_operations": {
                "begin_native_flat": True,
                "prepare_commit": True,
                "publish_commit": True,
                "abort": True,
            },
            "begin_law": BEGIN_LAW,
            "commit_law": COMMIT_LAW,
            "commit_failure_law": COMMIT_FAILURE_LAW,
            "abort_law": ABORT_LAW,
            "rollback_law": "private_kv_and_recurrent_snapshot_restore",
            "position_base_authority": POSITION_BASE_AUTHORITY,
            "hard_limits": dict(self.hard_limits),
            "source_hashes": self.native_source_hashes,
            "identity": dict(self.identity),
            "physical_layout": NATIVE_PHYSICAL_LAYOUT,
            "physical_rows_law": NATIVE_PHYSICAL_ROWS_LAW,
            "lane_staging_law": NATIVE_LANE_STAGING_LAW,
            "aggregate_commit_law": NATIVE_AGGREGATE_COMMIT_LAW,
        }

    def begin_native_flat(self, descriptor, *, target_executor, admission):
        """Open independent private B1 lane transactions without padded views."""

        if not isinstance(descriptor, FlatSegmentDescriptor):
            raise TypeError("hybrid begin_native_flat requires a validated descriptor")
        if self._active is not None:
            raise RuntimeError("hybrid flat membership is already leased")
        self._validate_rows(
            self.rows,
            self.boundaries,
            artifact=self.identity["artifact"],
            issuer_id=self._target_executor_id,
        )
        if id(target_executor) != self._target_executor_id:
            raise ValueError("native hybrid owner target issuer differs")
        if descriptor.lane_count != len(self.rows):
            raise ValueError("native hybrid descriptor lane count differs")
        backend = (
            admission.device_attestation.get("backend")
            if admission.device_attestation is not None
            else ""
        )
        validate_native_flat_segment_admission(
            admission,
            self,
            target_executor,
            descriptor,
            expected_backend=backend,
        )
        validator = getattr(target_executor, "validate_native_flat_owner", None)
        if not callable(validator):
            raise TypeError("native hybrid owner requires a compatible target")
        validator(self, descriptor)
        transaction = NativeFlatSegmentedHybridTransaction(
            self, descriptor, admission, target_executor=target_executor
        )
        self._active = transaction
        return transaction


class FlatSegmentedHybridTransaction:
    """Private hybrid verify successor with atomic publication."""

    def __init__(self, owner, descriptor, admission):
        self.owner = owner
        self.descriptor = descriptor
        self.admission = admission
        self.revision = owner.revision
        self._membership_identity = owner.ast_flat_segment_membership_identity()
        self._live_rows = [list(row) for row in owner.rows]
        self._base_boundaries = owner.boundaries
        self._base_receipt = canonical_bound_hybrid_state(
            self._live_rows, self._base_boundaries
        )
        self._private_owner = HybridVerifyRows(copy.deepcopy(self._live_rows))
        self._verify = self._private_owner.begin(descriptor.query_lengths)
        self._prepared = None
        self._original = [
            [self._snapshot(cache) for cache in row] for row in self._live_rows
        ]
        self.closed = False
        self.published = False
        self.aborted = False

    @staticmethod
    def _snapshot(cache):
        if type(cache) is KVCache:
            return ("kv", cache.keys, cache.values, cache.offset)
        return (
            "arrays",
            list(cache.cache),
            cache.lengths,
            cache.left_padding,
        )

    @property
    def caches(self):
        self._check_open()
        return self._verify.caches

    @property
    def private_rows(self):
        return self._private_owner.rows

    def _check_open(self):
        if (
            self.closed
            or self.owner is None
            or self.owner._active is not self
            or self.owner.revision != self.revision
        ):
            raise RuntimeError("stale or closed hybrid flat transaction")
        if self.owner.ast_flat_segment_membership_identity() != self._membership_identity:
            raise RuntimeError("authoritative hybrid membership changed")
        self.owner._validate_rows(
            self._live_rows,
            self.owner.boundaries,
            artifact=self.owner.identity["artifact"],
            issuer_id=self.owner._target_executor_id,
        )
        if any(
            boundary._owner_id != id(self.owner)
            for boundary in self.owner.boundaries
        ):
            raise RuntimeError("authoritative hybrid boundary owner changed")
        if canonical_bound_hybrid_state(
            self._live_rows, self.owner.boundaries
        ) != self._base_receipt:
            raise RuntimeError("authoritative hybrid state changed outside transaction")

    def prepare_commit(self, accepted_lengths):
        self._check_open()
        accepted = tuple(accepted_lengths)
        if (
            len(accepted) != self.descriptor.lane_count
            or any(type(value) is not int for value in accepted)
            or any(
                value < 1 or value > length
                for value, length in zip(
                    accepted, self.descriptor.query_lengths, strict=True
                )
            )
        ):
            raise ValueError("accepted hybrid lengths differ")
        successors = self._verify.commit(accepted)
        state = successor_bound_hybrid_state(
            successors, self._base_boundaries, accepted
        )
        prepared = PreparedFlatSegmentCommit(
            state=state,
            accepted_lengths=accepted,
            geometry_sha256=self.descriptor.geometry_sha256,
        )
        self._prepared = (prepared, successors)
        return prepared

    def _prepared_successor_rows_for_oracle(self, prepared):
        self._check_open()
        if self._prepared is None or prepared is not self._prepared[0]:
            raise ValueError("hybrid oracle requires its exact prepared successor")
        return tuple(tuple(row) for row in copy.deepcopy(self._prepared[1]))

    @staticmethod
    def _assign(live, successor):
        if type(live) is KVCache and type(successor) is KVCache:
            live.keys, live.values, live.offset = (
                successor.keys,
                successor.values,
                successor.offset,
            )
            return
        if type(live) is ArraysCache and type(successor) is ArraysCache:
            if successor.speculating or successor._rollbacks:
                raise RuntimeError("cannot publish speculative recurrent state")
            live.cache = list(successor.cache)
            live.lengths = successor.lengths
            live.left_padding = successor.left_padding
            live._host_lengths = None
            live._host_left_padding = None
            return
        raise TypeError("hybrid publication topology changed")

    def _restore_live(self):
        for row, snapshots in zip(self._live_rows, self._original, strict=True):
            for cache, snapshot in zip(row, snapshots, strict=True):
                if snapshot[0] == "kv":
                    _, cache.keys, cache.values, cache.offset = snapshot
                else:
                    _, state, lengths, padding = snapshot
                    cache.cache = list(state)
                    cache.lengths, cache.left_padding = lengths, padding
                    cache._host_lengths = None
                    cache._host_left_padding = None
        self.owner._boundaries = self._base_boundaries

    def _close(self):
        if self.owner is not None and self.owner._active is self:
            self.owner._active = None
            self.owner.revision += 1
        self.closed = True

    def publish_commit(self, prepared):
        self._check_open()
        if self._prepared is None or prepared is not self._prepared[0]:
            raise ValueError("hybrid publication requires exact prepared successor")
        live_successor_boundaries = ()
        try:
            for live_row, successor_row in zip(
                self._live_rows, self._prepared[1], strict=True
            ):
                for live, successor in zip(live_row, successor_row, strict=True):
                    self._assign(live, successor)
            live_successor_boundaries = successor_hybrid_boundaries(
                self._base_boundaries,
                self._live_rows,
                prepared.accepted_lengths,
                owner=self.owner,
            )
            result = canonical_bound_hybrid_state(self._live_rows, live_successor_boundaries)
            if result != prepared.state:
                raise RuntimeError("published hybrid state differs from prepared state")
            self.owner._install_boundaries(live_successor_boundaries)
            for boundary in self._base_boundaries:
                boundary._invalidate(self.owner)
        except BaseException:
            for boundary in live_successor_boundaries:
                # This is an internal successor minted by the owner-bound base
                # boundary.  Make rollback no-throw even when the failure was
                # injected into boundary lifecycle methods themselves.
                boundary._valid = False
            self._restore_live()
            for boundary in self._base_boundaries:
                boundary._valid = True
            self._close()
            raise
        self.published = True
        self._close()
        return result

    def abort(self):
        if self.aborted:
            return self._base_receipt
        if self.published:
            raise RuntimeError("published hybrid flat transaction is irrevocable")
        first = None
        if not self._verify.closed:
            try:
                self._verify.abort()
            except BaseException as error:  # noqa: BLE001
                first = error
        self._restore_live()
        self.aborted = True
        self._close()
        if first is not None:
            raise first
        return self._base_receipt


class NativeFlatSegmentedHybridTransaction(FlatSegmentedHybridTransaction):
    """One private B1 transaction per lane for native physical execution."""

    def __init__(self, owner, descriptor, admission, *, target_executor):
        self.owner = owner
        self.descriptor = descriptor
        self.admission = admission
        self._target_executor = target_executor
        self.revision = owner.revision
        self._membership_identity = owner.ast_flat_segment_membership_identity()
        self._live_rows = [list(row) for row in owner.rows]
        self._base_boundaries = owner.boundaries
        self._base_receipt = canonical_bound_hybrid_state(
            self._live_rows, self._base_boundaries
        )
        self._private_rows = copy.deepcopy(self._live_rows)
        self._lane_owners = []
        self._lane_transactions = []
        self._prepared = None
        self._original = [
            [self._snapshot(cache) for cache in row] for row in self._live_rows
        ]
        self.closed = False
        self.published = False
        self.aborted = False
        try:
            for row, length in zip(
                self._private_rows, descriptor.query_lengths, strict=True
            ):
                lane_owner = HybridVerifyRows([row])
                lane_transaction = lane_owner.begin([length])
                if lane_transaction.batched:
                    raise RuntimeError("native-flat lane unexpectedly built a batch")
                self._lane_owners.append(lane_owner)
                self._lane_transactions.append(lane_transaction)
        except BaseException:
            self._abort_private_lanes()
            raise

    @property
    def lane_caches(self):
        """One ordinary B1 cache row per lane, in descriptor order."""

        self._check_execution_authority()
        return tuple(tuple(transaction.caches) for transaction in self._lane_transactions)

    @property
    def private_rows(self):
        self._check_execution_authority()
        return tuple(tuple(row) for row in self._private_rows)

    def _check_execution_authority(self):
        self._check_open()
        backend = self.admission.device_attestation["backend"]
        validate_native_flat_segment_admission(
            self.admission,
            self.owner,
            self._target_executor,
            self.descriptor,
            expected_backend=backend,
        )
        validator = getattr(
            self._target_executor, "validate_native_flat_owner", None
        )
        if not callable(validator):
            raise TypeError("native hybrid target compatibility is unavailable")
        validator(self.owner, self.descriptor)

    def _abort_private_lanes(self):
        first = None
        for transaction in self._lane_transactions:
            if not transaction.closed:
                try:
                    transaction.abort()
                except BaseException as error:  # noqa: BLE001 - every lane is attempted
                    first = first or error
        return first

    def prepare_commit(self, accepted_lengths):
        self._check_execution_authority()
        accepted = tuple(accepted_lengths)
        if (
            len(accepted) != self.descriptor.lane_count
            or any(type(value) is not int for value in accepted)
            or any(
                value < 1 or value > length
                for value, length in zip(
                    accepted, self.descriptor.query_lengths, strict=True
                )
            )
        ):
            raise ValueError("accepted native hybrid lengths differ")
        successors = []
        try:
            for transaction, value in zip(
                self._lane_transactions, accepted, strict=True
            ):
                (successor,) = transaction.commit([value])
                successors.append(successor)
            state = successor_bound_hybrid_state(
                successors, self._base_boundaries, accepted
            )
            prepared = PreparedFlatSegmentCommit(
                state=state,
                accepted_lengths=accepted,
                geometry_sha256=self.descriptor.geometry_sha256,
            )
        except BaseException:
            self._abort_private_lanes()
            self._restore_live()
            self.aborted = True
            self._close()
            raise
        self._prepared = (prepared, successors)
        return prepared

    def abort(self):
        if self.aborted:
            return self._base_receipt
        if self.published:
            raise RuntimeError("published native hybrid transaction is irrevocable")
        first = self._abort_private_lanes()
        self._restore_live()
        self.aborted = True
        self._close()
        if first is not None:
            raise first
        return self._base_receipt


__all__ = [
    "FlatSegmentedHybridTransaction",
    "HybridLiveStateBoundary",
    "MetalFlatSegmentedHybridRows",
    "NativeFlatSegmentedHybridRows",
    "NativeFlatSegmentedHybridTransaction",
    "canonical_bound_hybrid_state",
    "canonical_hybrid_state",
    "successor_bound_hybrid_state",
    "successor_hybrid_boundaries",
]
