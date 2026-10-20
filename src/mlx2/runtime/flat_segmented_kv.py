"""Private-until-publish plain-KV ownership for real CPU flat AST probes.

This is deliberately narrower than :mod:`segmented_rotating_kv`: it accepts
only exact, independent ``KVCache`` rows and never exposes a padded batched
view.  Target execution mutates deep-copied private rows.  ``prepare_commit``
constructs an accepted-prefix successor privately, and ``publish_commit``
swaps that state into the authoritative rows only after the caller's ordinary
serial oracle has accepted it.

The module is not wired into serving and does not authorize a backend.
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
    OWNER_CAPABILITY_SCHEMA,
    POSITION_BASE_AUTHORITY,
    REAL_EXECUTION_SCOPE,
    REAL_OWNER_PROTOCOL,
    FlatSegmentDescriptor,
    PreparedFlatSegmentCommit,
    validate_real_flat_segment_admission,
    validate_metal_flat_segment_admission,
)
from .models.cache import KVCache

_PACKAGE = Path(__file__).resolve().parents[1]
_PACKAGE_PREFIX = "src/mlx2/"
_HARD_INDEX_LIMIT = (1 << 31) - 1
REAL_FLAT_HARD_LIMITS = {
    "max_lanes": _HARD_INDEX_LIMIT,
    "max_segment_rows": _HARD_INDEX_LIMIT,
    "max_total_rows": _HARD_INDEX_LIMIT,
}
METAL_PROBE_HARD_LIMITS = {
    "max_lanes": 4,
    "max_segment_rows": 4,
    "max_total_rows": 12,
}
_SOURCE_PATHS = (
    "src/mlx2/runtime/flat_segmented_kv.py",
    "src/mlx2/runtime/flat_segmented_ast.py",
    "src/mlx2/runtime/ast_execution_capability.py",
    "src/mlx2/runtime/cache_operations.py",
    "src/mlx2/runtime/models/cache.py",
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _source_hashes(paths=_SOURCE_PATHS) -> dict[str, str]:
    result = {}
    for name in paths:
        if not name.startswith(_PACKAGE_PREFIX):
            raise ValueError("flat KV source path must be package-relative")
        path = _PACKAGE / name[len(_PACKAGE_PREFIX) :]
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _array_receipt(value: mx.array) -> dict[str, object]:
    mx.eval(value)
    # NumPy has no native MLX BF16 PEP 3118 representation.  Hash the exact
    # storage bytes instead of converting values through a lossy float dtype.
    host = np.asarray(value.reshape(-1).view(mx.uint8))
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(host.tobytes(order="C")).hexdigest(),
    }


def canonical_kv_state(rows) -> tuple[tuple[dict[str, object], ...], ...]:
    """Token-free exact visible-state receipt used by the CPU oracle."""

    result = []
    for row in rows:
        layers = []
        for cache in row:
            if type(cache) is not KVCache:
                raise TypeError("real flat state receipts require exact KVCache")
            if cache.keys is None:
                keys = values = None
            else:
                keys, values = cache.keys_and_values()
                keys, values = _array_receipt(keys), _array_receipt(values)
            layers.append(
                {
                    "offset": int(cache.offset),
                    "keys": keys,
                    "values": values,
                }
            )
        result.append(tuple(layers))
    return tuple(result)


def _validate_identity(identity: Mapping[str, str]) -> dict[str, str]:
    fields = {"source_revision", "build", "device", "artifact"}
    if not isinstance(identity, Mapping) or set(identity) != fields or any(
        not isinstance(identity[name], str) or not identity[name] for name in fields
    ):
        raise ValueError("real flat KV identity fields differ")
    return {name: identity[name] for name in sorted(fields)}


class FlatSegmentedKVRows:
    """Exclusive exact-KV owner for one direct real flat CPU round."""

    execution_scope = REAL_EXECUTION_SCOPE
    owner_protocol = REAL_OWNER_PROTOCOL
    admission_validator = staticmethod(validate_real_flat_segment_admission)
    hard_limits = REAL_FLAT_HARD_LIMITS

    def __init__(self, rows, *, identity: Mapping[str, str]):
        self._rows = tuple(tuple(row) for row in rows)
        self.identity = _validate_identity(identity)
        self.revision = 0
        self._active = None
        self._validate_rows(self._rows)
        state = state_contract_from_rows(self._rows)
        if state != {
            "state_plane_kinds": ["attention_kv"],
            "state_families": ["kv"],
            "state_contract_fingerprints": [
                "5cce324cf8b59feb02ac4362c53872b62b19d06fc9e0ceab478d2a867ea88a2e"
            ],
            "state_ownership": "independent_request_private_b1_lanes",
        }:
            raise ValueError("plain KV state contract differs")
        self._state_contract = state

    @property
    def rows(self):
        """Read-only exact cache membership; cache state remains owner-managed."""

        return self._rows

    def ast_flat_segment_membership_identity(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(id(cache) for cache in row) for row in self._rows)

    def ast_flat_segment_state_identity_sha256(self) -> str:
        return hashlib.sha256(_canonical(canonical_kv_state(self._rows))).hexdigest()

    @staticmethod
    def _validate_rows(rows) -> None:
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("flat KV rows require one common nonempty layer layout")
        leaves = [cache for row in rows for cache in row]
        if len({id(cache) for cache in leaves}) != len(leaves):
            raise ValueError("flat KV rows must have independent cache owners")
        for row in rows:
            if any(type(cache) is not KVCache for cache in row):
                raise TypeError("real flat transactions support exact KVCache only")
            if len({cache.offset for cache in row}) != 1:
                raise ValueError("all flat KV layers in one lane need one offset")
            for cache in row:
                if type(cache.offset) is not int or cache.offset < 0:
                    raise ValueError("flat KV offsets must be host integers")
                if getattr(cache, "speculating", False):
                    raise ValueError("flat KV rows cannot have another speculation owner")
                if cache.keys is None:
                    if cache.values is not None or cache.offset:
                        raise ValueError("flat KV empty state is incomplete")
                elif (
                    cache.values is None
                    or not isinstance(cache.keys, mx.array)
                    or not isinstance(cache.values, mx.array)
                    or cache.keys.ndim != 4
                    or cache.values.ndim != 4
                    or cache.keys.shape != cache.values.shape
                    or cache.keys.dtype != cache.values.dtype
                    or cache.keys.shape[0] != 1
                    or cache.values.shape[0] != 1
                    or cache.offset > cache.keys.shape[2]
                    or cache.offset > cache.values.shape[2]
                ):
                    raise ValueError("flat KV storage geometry is unsupported")

    @property
    def source_hashes(self) -> dict[str, str]:
        return _source_hashes()

    @property
    def state_contract_fingerprints(self) -> tuple[str, ...]:
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
            "rollback_law": "private_kv_snapshot_restore",
            "position_base_authority": POSITION_BASE_AUTHORITY,
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
            "rollback_law": "private_kv_snapshot_restore",
            "position_base_authority": POSITION_BASE_AUTHORITY,
            "hard_limits": dict(self.hard_limits),
            "source_hashes": self.source_hashes,
            "identity": dict(self.identity),
        }

    def begin_flat(
        self,
        descriptor: FlatSegmentDescriptor,
        *,
        target_executor,
        admission,
    ):
        if not isinstance(descriptor, FlatSegmentDescriptor):
            raise TypeError("begin_flat requires a validated flat descriptor")
        if self._active is not None:
            raise RuntimeError("flat KV membership is already leased")
        self._validate_rows(self.rows)
        if descriptor.lane_count != len(self.rows):
            raise ValueError("flat descriptor lane count differs from KV ownership")
        self.admission_validator(
            admission,
            self,
            target_executor,
            descriptor,
        )
        validator = getattr(target_executor, "validate_flat_owner", None)
        if not callable(validator):
            raise TypeError("begin_flat requires a compatible real target executor")
        # This is deliberately the final operation before transaction creation.
        # A stale admission receipt cannot authorize a changed model/cache pair.
        validator(self, descriptor)
        transaction = FlatSegmentedKVTransaction(self, descriptor, admission)
        self._active = transaction
        return transaction


class MetalFlatSegmentedKVRows(FlatSegmentedKVRows):
    """Exact-KV owner whose admission is valid only for the Metal probe."""

    execution_scope = METAL_EXECUTION_SCOPE
    owner_protocol = METAL_OWNER_PROTOCOL
    admission_validator = staticmethod(validate_metal_flat_segment_admission)
    hard_limits = METAL_PROBE_HARD_LIMITS


class FlatSegmentedKVTransaction:
    """Private cache successor and exact prepare/publish lifecycle."""

    def __init__(
        self,
        owner: FlatSegmentedKVRows,
        descriptor: FlatSegmentDescriptor,
        admission,
    ):
        self.owner = owner
        self.descriptor = descriptor
        self.admission = admission
        self.revision = owner.revision
        self._membership_identity = owner.ast_flat_segment_membership_identity()
        self._live_rows = [list(row) for row in owner.rows]
        self._base_rows = copy.deepcopy(self._live_rows)
        self.private_rows = copy.deepcopy(self._live_rows)
        self._base_receipt = canonical_kv_state(self._live_rows)
        self._original = [
            [(cache.keys, cache.values, cache.offset) for cache in row]
            for row in self._live_rows
        ]
        self._completed_layers: set[int] = set()
        self._prepared = None
        self.closed = False
        self.published = False
        self.aborted = False

    def _check_open(self) -> None:
        if (
            self.closed
            or self.owner is None
            or self.owner._active is not self
            or self.owner.revision != self.revision
        ):
            raise RuntimeError("stale or closed real flat KV transaction")
        if (
            self.owner.ast_flat_segment_membership_identity()
            != self._membership_identity
        ):
            raise RuntimeError("authoritative flat KV membership changed")
        if canonical_kv_state(self._live_rows) != self._base_receipt:
            raise RuntimeError("authoritative KV state changed outside flat transaction")

    @property
    def layer_count(self) -> int:
        return len(self._live_rows[0])

    @property
    def lane_offsets(self) -> tuple[int, ...]:
        return tuple(row[0].offset for row in self._base_rows)

    def attend_flat_layer(self, layer, queries, keys, values, *, scale):
        """Append and attend private B1/S1 rows in descriptor order."""

        self._check_open()
        if type(layer) is not int or not 0 <= layer < self.layer_count:
            raise ValueError("flat KV layer index is invalid")
        if layer in self._completed_layers:
            raise RuntimeError("a flat KV layer can execute only once")
        rows = self.descriptor.logical_rows
        if (
            not all(isinstance(value, mx.array) for value in (queries, keys, values))
            or queries.ndim != 4
            or keys.ndim != 4
            or values.ndim != 4
            or queries.shape[:2] != (1, rows)
            or keys.shape[:2] != (1, rows)
            or values.shape[:2] != (1, rows)
            or keys.shape[2] != values.shape[2]
        ):
            raise ValueError("flat KV projected row geometry differs")
        outputs = []
        for lane in range(self.descriptor.lane_count):
            span = self.descriptor.lane_slice(lane)
            cache = self.private_rows[lane][layer]
            for position in range(span.start, span.stop):
                query = queries[:, position : position + 1].transpose(0, 2, 1, 3)
                key = keys[:, position : position + 1].transpose(0, 2, 1, 3)
                value = values[:, position : position + 1].transpose(0, 2, 1, 3)
                all_keys, all_values = cache.update_and_fetch(key, value)
                attended = mx.fast.scaled_dot_product_attention(
                    query,
                    all_keys,
                    all_values,
                    scale=scale,
                    mask=None,
                )
                outputs.append(attended.transpose(0, 2, 1, 3))
        self._completed_layers.add(layer)
        return mx.concatenate(outputs, axis=1)

    def _successors(self, accepted: tuple[int, ...]):
        successors = copy.deepcopy(self._base_rows)
        for lane, count in enumerate(accepted):
            for layer in range(self.layer_count):
                base = self._base_rows[lane][layer]
                candidate = self.private_rows[lane][layer]
                if candidate.offset != base.offset + self.descriptor.query_lengths[lane]:
                    raise RuntimeError("private flat KV row did not consume its complete path")
                if count:
                    keys, values = candidate.keys_and_values()
                    start = base.offset
                    for position in range(count):
                        successors[lane][layer].update_and_fetch(
                            keys[..., start + position : start + position + 1, :],
                            values[..., start + position : start + position + 1, :],
                        )
        return successors

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
            raise ValueError("accepted flat KV lengths must cover each K+1 path")
        if self._completed_layers != set(range(self.layer_count)):
            raise RuntimeError("every flat target layer must finish before prepare")
        successors = self._successors(accepted)
        state = canonical_kv_state(successors)
        prepared = PreparedFlatSegmentCommit(
            state=state,
            accepted_lengths=accepted,
            geometry_sha256=self.descriptor.geometry_sha256,
        )
        self._prepared = (prepared, successors)
        return prepared

    def _prepared_successor_rows_for_oracle(self, prepared):
        """Return an isolated prepared successor for prepublication checking."""

        self._check_open()
        if self._prepared is None or prepared is not self._prepared[0]:
            raise ValueError("flat KV oracle requires its exact prepared successor")
        return tuple(
            tuple(row) for row in copy.deepcopy(self._prepared[1])
        )

    @staticmethod
    def _assign_cache(live: KVCache, successor: KVCache) -> None:
        live.keys = successor.keys
        live.values = successor.values
        live.offset = successor.offset

    def _restore_live(self) -> None:
        for row, originals in zip(self._live_rows, self._original, strict=True):
            for cache, (keys, values, offset) in zip(row, originals, strict=True):
                cache.keys = keys
                cache.values = values
                cache.offset = offset

    def _close(self) -> None:
        if self.owner is not None and self.owner._active is self:
            self.owner._active = None
            self.owner.revision += 1
        self.closed = True

    def publish_commit(self, prepared):
        self._check_open()
        if self._prepared is None or prepared is not self._prepared[0]:
            raise ValueError("flat KV publication requires its exact prepared successor")
        successors = self._prepared[1]
        try:
            for live_row, successor_row in zip(
                self._live_rows, successors, strict=True
            ):
                for live, successor in zip(live_row, successor_row, strict=True):
                    self._assign_cache(live, successor)
            result = canonical_kv_state(self._live_rows)
            if result != prepared.state:
                raise RuntimeError(
                    "published flat KV state differs from prepared successor"
                )
        except BaseException:
            self._restore_live()
            self._close()
            raise
        self.published = True
        self._close()
        return result

    def abort(self):
        if self.aborted:
            return self._base_receipt
        if self.published:
            raise RuntimeError("published real flat KV transaction is irrevocable")
        if not self.closed:
            self._restore_live()
        self.aborted = True
        self._close()
        return self._base_receipt


__all__ = [
    "METAL_PROBE_HARD_LIMITS",
    "REAL_FLAT_HARD_LIMITS",
    "FlatSegmentedKVRows",
    "FlatSegmentedKVTransaction",
    "MetalFlatSegmentedKVRows",
    "canonical_kv_state",
]
