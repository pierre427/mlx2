"""Unwired lifecycle and result ABI for future physical AST targets.

This module does not select a target or alter serving.  A registry handle is a
session-local proof of exclusive cache *membership and lifecycle*, not a hash
of cache tensor values and not an exact numerical-state identity.  It is only
valid while the registry owns an exclusive lease and the registered base rows
remain structurally untouched.  Target execution must use disjoint private
successor rows until its delegate publishes them atomically.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .cache_operations import CacheOperationsContract, cache_operations
from .flat_segmented_ast import (
    FlatSegmentDescriptor,
    NativeFlatSegmentExecution,
    validate_native_flat_segment_execution,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+:/@-]{0,255}\Z")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_DTYPE = re.compile(r"[A-Za-z0-9_.]{1,64}\Z")
_PUBLICATION_MODES = {"in_place_base", "transfer_successor"}
_RESULT_STORAGE_CLASSES = {
    "flat_logical_rows",
    "flat_row_pairs",
    "rectangular_rows",
}
_HANDLE_SEAL = object()
_LEASE_SEAL = object()
_PREPARED_SEAL = object()
_PLAN_SEAL = object()


class ASTLiveStateError(RuntimeError):
    """A live-state or physical-result contract failed closed."""


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ASTLiveStateError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ASTLiveStateError(f"{label} must be an integer >= {minimum}")
    return value


def _shape(value: object, label: str) -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if not isinstance(raw, tuple):
        raise ASTLiveStateError(f"{label} must expose an immutable shape")
    if not raw or len(raw) > 16:
        raise ASTLiveStateError(f"{label} must have a row dimension")
    shape = tuple(_exact_int(item, f"{label} dimension") for item in raw)
    if any(item > 2**31 - 1 for item in shape):
        raise ASTLiveStateError(f"{label} dimensions exceed receipt bounds")
    return shape


def _dtype(value: object, label: str) -> str:
    dtype = str(getattr(value, "dtype", ""))
    if _DTYPE.fullmatch(dtype) is None:
        raise ASTLiveStateError(f"{label} dtype is invalid")
    return dtype


def _host_semantic(value: object) -> object:
    """Describe declared host semantics without reading tensor contents."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ASTLiveStateError("host semantic floats must be finite")
        return value
    if isinstance(value, tuple):
        # Provider properties commonly assemble an ephemeral tuple on every
        # read (for example ``state``).  Its members, not that wrapper, own the
        # declared state identity.
        return ("tuple", tuple(_host_semantic(item) for item in value))
    if isinstance(value, list):
        return (
            "list",
            id(value),
            tuple(_host_semantic(item) for item in value),
        )
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) and len(key) <= 128 for key in value):
            raise ASTLiveStateError("host semantic mapping keys are invalid")
        items = []
        for key in sorted(value):
            items.append((key, _host_semantic(value[key])))
        return ("mapping", id(value), tuple(items))
    return ("object", id(value))


def _identity_objects(value: object) -> tuple[int, ...]:
    if value is None or type(value) in {bool, int, float, str}:
        return ()
    shape = getattr(value, "shape", None)
    if (
        isinstance(shape, tuple)
        and shape
        and all(type(dimension) is int and dimension >= 0 for dimension in shape)
        and math.prod(shape) == 0
    ):
        # Empty tensor sentinels carry no mutable state and are legitimately
        # shared by independent Qwen GDN cache planes.
        return ()
    if isinstance(value, tuple):
        nested = []
        for item in value:
            nested.extend(_identity_objects(item))
        return tuple(nested)
    if isinstance(value, list):
        nested = [id(value)]
        for item in value:
            nested.extend(_identity_objects(item))
        return tuple(nested)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) and len(key) <= 128 for key in value):
            raise ASTLiveStateError("state member mapping keys are invalid")
        nested = [id(value)]
        for key in sorted(value):
            nested.extend(_identity_objects(value[key]))
        return tuple(nested)
    return (id(value),)


def _row_state_object_ids(rows: Sequence[Sequence[object]]) -> set[int]:
    identities: set[int] = set()
    for row in rows:
        for cache in row:
            identities.add(id(cache))
            contract = cache_operations(cache, capability="AST private-state isolation")
            for plane in contract.planes:
                for value in contract.projection(cache, plane):
                    identities.update(_identity_objects(value))
    return identities


def _same_row_membership(
    left: Sequence[Sequence[object]],
    right: Sequence[Sequence[object]],
) -> bool:
    """Compare nested lane/cache membership strictly by object identity."""

    if len(left) != len(right):
        return False
    return all(
        len(left_row) == len(right_row)
        and all(
            left_cache is right_cache
            for left_cache, right_cache in zip(left_row, right_row, strict=True)
        )
        for left_row, right_row in zip(left, right, strict=True)
    )


@dataclass(frozen=True, slots=True)
class ASTLiveStateAuthority:
    """Externally reviewed identity and hard capacity for one owner session."""

    artifact: str
    session_scope_sha256: str
    owner_identity_sha256: str
    provider_manifest_sha256: str
    source_manifest_sha256: str
    build_manifest_sha256: str
    device_manifest_sha256: str
    max_lanes: int
    max_caches_per_lane: int
    max_segment_rows: int
    max_total_rows: int
    max_physical_rows: int
    required_plane_kinds: tuple[str, ...]
    publication_modes: tuple[str, ...]
    result_storage_classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact, str)
            or _ARTIFACT.fullmatch(self.artifact) is None
        ):
            raise ASTLiveStateError("artifact identity is invalid")
        for name in (
            "session_scope_sha256",
            "owner_identity_sha256",
            "provider_manifest_sha256",
            "source_manifest_sha256",
            "build_manifest_sha256",
            "device_manifest_sha256",
        ):
            _digest(getattr(self, name), name)
        _exact_int(self.max_lanes, "max_lanes", minimum=1)
        _exact_int(self.max_caches_per_lane, "max_caches_per_lane", minimum=1)
        _exact_int(self.max_segment_rows, "max_segment_rows", minimum=1)
        _exact_int(self.max_total_rows, "max_total_rows", minimum=1)
        _exact_int(self.max_physical_rows, "max_physical_rows", minimum=1)
        if (
            not isinstance(self.required_plane_kinds, tuple)
            or not self.required_plane_kinds
            or len(self.required_plane_kinds) > 16
            or len(set(self.required_plane_kinds)) != len(self.required_plane_kinds)
            or any(
                not isinstance(kind, str) or _IDENTIFIER.fullmatch(kind) is None
                for kind in self.required_plane_kinds
            )
        ):
            raise ASTLiveStateError("required plane kinds are invalid")
        if (
            not isinstance(self.publication_modes, tuple)
            or not self.publication_modes
            or len(set(self.publication_modes)) != len(self.publication_modes)
            or set(self.publication_modes) - _PUBLICATION_MODES
        ):
            raise ASTLiveStateError("publication modes are invalid")
        if (
            not isinstance(self.result_storage_classes, tuple)
            or not self.result_storage_classes
            or len(set(self.result_storage_classes)) != len(self.result_storage_classes)
            or set(self.result_storage_classes) - _RESULT_STORAGE_CLASSES
        ):
            raise ASTLiveStateError("result storage classes are invalid")

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.receipt())

    def receipt(self) -> dict[str, object]:
        return {
            "schema": "mlx2.ast-live-state-authority.v1",
            "artifact": self.artifact,
            "session_scope_sha256": self.session_scope_sha256,
            "owner_identity_sha256": self.owner_identity_sha256,
            "provider_manifest_sha256": self.provider_manifest_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "build_manifest_sha256": self.build_manifest_sha256,
            "device_manifest_sha256": self.device_manifest_sha256,
            "max_lanes": self.max_lanes,
            "max_caches_per_lane": self.max_caches_per_lane,
            "max_segment_rows": self.max_segment_rows,
            "max_total_rows": self.max_total_rows,
            "max_physical_rows": self.max_physical_rows,
            "required_plane_kinds": self.required_plane_kinds,
            "publication_modes": self.publication_modes,
            "result_storage_classes": self.result_storage_classes,
        }


@dataclass(frozen=True, slots=True)
class ASTLiveLaneStateHandle:
    """Opaque membership/generation handle; never a numerical-state seal."""

    registry_id: int
    registry_epoch: int
    lane: int
    generation: int
    position: int
    authority_manifest_sha256: str
    membership_sha256: str
    _seal: object

    def receipt(self) -> dict[str, object]:
        return {
            "schema": "mlx2.ast-live-lane-state-handle.v1",
            "registry_epoch": self.registry_epoch,
            "lane": self.lane,
            "generation": self.generation,
            "position": self.position,
            "authority_manifest_sha256": self.authority_manifest_sha256,
            "membership_sha256": self.membership_sha256,
            "identity_class": "exclusive_lifecycle_membership_only",
            "numerical_state_identity": False,
        }


@dataclass(frozen=True, slots=True)
class _CacheSignature:
    cache_id: int
    contract: CacheOperationsContract
    fingerprint: str
    position: int | None
    evaluation_semantics: object
    condition_semantics: object
    compatibility: object

    def manifest(self) -> object:
        return {
            "cache_id": self.cache_id,
            "fingerprint": self.fingerprint,
            "position": self.position,
            "evaluation_semantics": self.evaluation_semantics,
            "condition_semantics": self.condition_semantics,
            "compatibility": self.compatibility,
        }


@dataclass(frozen=True, slots=True)
class _RegistrySnapshot:
    epoch: int
    rows: tuple[tuple[object, ...], ...]
    positions: tuple[int, ...]
    signatures: tuple[tuple[_CacheSignature, ...], ...]
    handles: tuple[ASTLiveLaneStateHandle, ...]
    active_lease: object | None
    poisoned: bool = False


def _cache_signature(
    cache: object,
    *,
    expected_position: int,
    publication: str,
    field_source: object | None = None,
) -> _CacheSignature:
    contract = cache_operations(cache, capability="AST live-state membership")
    if not contract.exact_prefix:
        raise ASTLiveStateError("cache operation contract is not exact-prefix")
    source = cache if field_source is None else field_source
    source_contract = cache_operations(source, capability="AST successor state")
    if source_contract != contract:
        raise ASTLiveStateError("publication and successor cache contracts differ")
    if publication == "public":
        source_contract.validate_publication(source)
    elif publication == "segmented":
        source_contract.validate_segmented_publication(source)
    else:
        raise ASTLiveStateError("unknown publication boundary")
    position = source_contract.position(source)
    if source_contract.offset_field is not None and position is None:
        raise ASTLiveStateError("declared cache position is not an exact integer")
    if position is not None and position != expected_position:
        raise ASTLiveStateError("cache position differs from lane position")
    evaluations = tuple(
        (
            plane.kind,
            tuple(
                _host_semantic(value)
                for value in source_contract.projection(source, plane)
            ),
        )
        for plane in source_contract.planes
    )
    conditions = tuple(
        dict.fromkeys(
            (
                condition.field,
                condition.active_when,
                condition.reason,
            )
            for condition in (
                *source_contract.publication_conditions,
                *source_contract.segmented_publication_conditions,
            )
        )
    )
    condition_values = tuple(
        (
            field,
            active_when,
            bool(getattr(source, field))
            if active_when == "truthy"
            else getattr(source, field) is not None,
        )
        for field, active_when, _reason in conditions
    )
    compatibility = tuple(
        (plane.kind, source_contract.compatibility(source, plane))
        for plane in source_contract.planes
    )
    return _CacheSignature(
        cache_id=id(cache),
        contract=contract,
        fingerprint=contract.fingerprint,
        position=position,
        evaluation_semantics=evaluations,
        condition_semantics=condition_values,
        compatibility=compatibility,
    )


def _sign_rows(
    rows: Sequence[Sequence[object]],
    positions: Sequence[int],
    *,
    publication: str,
    field_rows: Sequence[Sequence[object]] | None = None,
    max_lanes: int,
    max_caches_per_lane: int,
    required_plane_kinds: Sequence[str],
) -> tuple[tuple[tuple[object, ...], ...], tuple[tuple[_CacheSignature, ...], ...]]:
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ASTLiveStateError("live rows must be a non-empty sequence")
    if len(rows) > max_lanes or len(rows) != len(positions):
        raise ASTLiveStateError("live row count violates authority or positions")
    if field_rows is not None and len(field_rows) != len(rows):
        raise ASTLiveStateError("successor row count differs from publication rows")
    normalized_rows = []
    signatures = []
    cache_ids: set[int] = set()
    member_ids: set[int] = set()
    expected_row_fingerprints = None
    for lane, raw_row in enumerate(rows):
        position = _exact_int(positions[lane], "lane position")
        if not isinstance(raw_row, (tuple, list)) or not raw_row:
            raise ASTLiveStateError("each lane must own a non-empty cache row")
        if len(raw_row) > max_caches_per_lane:
            raise ASTLiveStateError("cache row exceeds reviewed capacity")
        source_row = raw_row if field_rows is None else field_rows[lane]
        if len(source_row) != len(raw_row):
            raise ASTLiveStateError("successor cache topology differs")
        normalized = tuple(raw_row)
        lane_signatures = []
        kinds = set()
        has_position_authority = False
        for index, cache in enumerate(normalized):
            if id(cache) in cache_ids:
                raise ASTLiveStateError("cache objects must not alias across rows")
            cache_ids.add(id(cache))
            source = source_row[index]
            signature = _cache_signature(
                cache,
                expected_position=position,
                publication=publication,
                field_source=source,
            )
            for plane in signature.contract.planes:
                for value in signature.contract.projection(source, plane):
                    for member_id in _identity_objects(value):
                        if member_id in member_ids:
                            raise ASTLiveStateError("cache state members must not alias")
                        member_ids.add(member_id)
            for plane in signature.contract.planes:
                kinds.add(plane.kind)
            has_position_authority = has_position_authority or (
                signature.contract.offset_field is not None
            )
            lane_signatures.append(signature)
        if not has_position_authority:
            raise ASTLiveStateError("lane lacks an exact positional cache authority")
        missing = set(required_plane_kinds) - kinds
        if missing:
            raise ASTLiveStateError("lane lacks an authority-required state plane")
        row_fingerprints = tuple(signature.fingerprint for signature in lane_signatures)
        if expected_row_fingerprints is None:
            expected_row_fingerprints = row_fingerprints
        elif row_fingerprints != expected_row_fingerprints:
            raise ASTLiveStateError("lane cache-operation topology differs")
        normalized_rows.append(normalized)
        signatures.append(tuple(lane_signatures))
    return tuple(normalized_rows), tuple(signatures)


def _membership_sha256(
    *,
    authority: ASTLiveStateAuthority,
    epoch: int,
    lane: int,
    generation: int,
    position: int,
    signatures: Sequence[_CacheSignature],
) -> str:
    return _sha256(
        {
            "authority": authority.manifest_sha256,
            "epoch": epoch,
            "lane": lane,
            "generation": generation,
            "position": position,
            "caches": [signature.manifest() for signature in signatures],
        }
    )


class ASTLiveLaneStateRegistry:
    """Exclusive owner-local lifecycle registry for private hybrid lanes."""

    def __init__(
        self,
        rows: Sequence[Sequence[object]],
        positions: Sequence[int],
        *,
        authority: ASTLiveStateAuthority,
    ) -> None:
        if type(authority) is not ASTLiveStateAuthority:
            raise ASTLiveStateError("live-state authority is unbound")
        normalized_positions = tuple(
            _exact_int(position, "lane position") for position in positions
        )
        normalized_rows, signatures = _sign_rows(
            rows,
            normalized_positions,
            publication="public",
            max_lanes=authority.max_lanes,
            max_caches_per_lane=authority.max_caches_per_lane,
            required_plane_kinds=authority.required_plane_kinds,
        )
        self._authority = authority
        self._registry_id = id(self)
        handles = self._make_handles(
            epoch=0,
            generations=(0,) * len(normalized_rows),
            positions=normalized_positions,
            signatures=signatures,
        )
        self._state = _RegistrySnapshot(
            0, normalized_rows, normalized_positions, signatures, handles, None
        )

    @property
    def handles(self) -> tuple[ASTLiveLaneStateHandle, ...]:
        if self._state.poisoned:
            raise ASTLiveStateError("live-state registry is poisoned")
        if self._state.active_lease is not None:
            raise ASTLiveStateError("live-state handles are unavailable")
        return self._state.handles

    @property
    def epoch(self) -> int:
        return self._state.epoch

    def receipt(self) -> dict[str, object]:
        return {
            "schema": "mlx2.ast-live-state-registry.v1",
            "authority": self._authority.receipt(),
            "authority_manifest_sha256": self._authority.manifest_sha256,
            "registry_epoch": self._state.epoch,
            "lane_count": len(self._state.rows),
            "active_lease": self._state.active_lease is not None,
            "poisoned": self._state.poisoned,
            "identity_class": "exclusive_lifecycle_membership_only",
            "numerical_state_identity": False,
            "implemented": True,
            "qualified": False,
            "selected": False,
            "default": False,
            "observed_used": False,
        }

    def _make_handles(
        self,
        *,
        epoch: int,
        generations: Sequence[int],
        positions: Sequence[int],
        signatures: Sequence[Sequence[_CacheSignature]],
    ) -> tuple[ASTLiveLaneStateHandle, ...]:
        return tuple(
            ASTLiveLaneStateHandle(
                self._registry_id,
                epoch,
                lane,
                generation,
                positions[lane],
                self._authority.manifest_sha256,
                _membership_sha256(
                    authority=self._authority,
                    epoch=epoch,
                    lane=lane,
                    generation=generation,
                    position=positions[lane],
                    signatures=signatures[lane],
                ),
                _HANDLE_SEAL,
            )
            for lane, generation in enumerate(generations)
        )

    def begin(
        self,
        handles: Sequence[ASTLiveLaneStateHandle],
        rows: Sequence[Sequence[object]],
        positions: Sequence[int],
        *,
        authority: ASTLiveStateAuthority,
    ) -> ASTLiveLaneStateLease:
        state = self._state
        self._validate_authority(authority)
        if state.poisoned:
            raise ASTLiveStateError("live-state registry is poisoned")
        if state.active_lease is not None:
            raise ASTLiveStateError("live-state registry already has an active lease")
        if (
            not isinstance(handles, (tuple, list))
            or len(handles) != len(state.handles)
            or any(
                candidate is not expected
                for candidate, expected in zip(handles, state.handles)
            )
        ):
            raise ASTLiveStateError("live-state handles are stale or reordered")
        self._validate_base(rows, positions, state)
        lease_id = object()
        unlocked = state
        leased = _RegistrySnapshot(
            state.epoch,
            state.rows,
            state.positions,
            state.signatures,
            state.handles,
            lease_id,
        )
        self._state = leased
        return ASTLiveLaneStateLease(self, lease_id, unlocked, _LEASE_SEAL)

    def _validate_authority(self, authority: ASTLiveStateAuthority) -> None:
        if type(authority) is not ASTLiveStateAuthority or authority != self._authority:
            raise ASTLiveStateError(
                "live artifact/provider/source/build/device authority drifted"
            )

    def _validate_base(
        self,
        rows: Sequence[Sequence[object]],
        positions: Sequence[int],
        expected: _RegistrySnapshot,
    ) -> None:
        normalized_positions = tuple(
            _exact_int(position, "lane position") for position in positions
        )
        normalized_rows, signatures = _sign_rows(
            rows,
            normalized_positions,
            publication="public",
            max_lanes=self._authority.max_lanes,
            max_caches_per_lane=self._authority.max_caches_per_lane,
            required_plane_kinds=self._authority.required_plane_kinds,
        )
        if (
            not _same_row_membership(normalized_rows, expected.rows)
            or normalized_positions != expected.positions
            or signatures != expected.signatures
        ):
            raise ASTLiveStateError("registered base state membership drifted")


@dataclass(frozen=True, slots=True)
class ASTPreparedPhysicalPublication:
    """Typed delegate result describing one still-private successor."""

    successor_rows: tuple[tuple[object, ...], ...]
    publication_rows: tuple[tuple[object, ...], ...]
    publication_mode: str
    accepted_lengths: tuple[int, ...]
    successor_positions: tuple[int, ...]
    descriptor_geometry_sha256: str
    result_manifest_sha256: str
    delegate_token: object
    _authority: object
    _sequence: int
    _seal: object


class _ASTPhysicalPrepareAuthority:
    __slots__ = ("_delegate_id", "_prepared", "_used")

    def __init__(self, delegate: object) -> None:
        self._delegate_id = id(delegate)
        self._prepared = None
        self._used = False

    def seal(
        self,
        *,
        successor_rows: Sequence[Sequence[object]],
        publication_rows: Sequence[Sequence[object]],
        publication_mode: str,
        accepted_lengths: Sequence[int],
        successor_positions: Sequence[int],
        descriptor_geometry_sha256: str,
        result_manifest_sha256: str,
        delegate_token: object,
    ) -> ASTPreparedPhysicalPublication:
        if self._used:
            raise ASTLiveStateError("prepare authority is single-use")
        if publication_mode not in _PUBLICATION_MODES:
            raise ASTLiveStateError("publication mode is invalid")
        prepared = ASTPreparedPhysicalPublication(
            tuple(tuple(row) for row in successor_rows),
            tuple(tuple(row) for row in publication_rows),
            publication_mode,
            tuple(accepted_lengths),
            tuple(successor_positions),
            descriptor_geometry_sha256,
            result_manifest_sha256,
            delegate_token,
            self,
            0,
            _PREPARED_SEAL,
        )
        self._prepared = prepared
        self._used = True
        return prepared

    def validate(self, prepared: object, delegate: object) -> None:
        if (
            type(prepared) is not ASTPreparedPhysicalPublication
            or prepared is not self._prepared
            or prepared._authority is not self
            or prepared._sequence != 0
            or prepared._seal is not _PREPARED_SEAL
            or id(delegate) != self._delegate_id
        ):
            raise ASTLiveStateError(
                "prepared publication is foreign, copied, or replayed"
            )


@dataclass(frozen=True, slots=True)
class ASTPublishedPhysicalRound:
    result: ASTPhysicalResult
    accepted_lengths: tuple[int, ...]
    publication_rows: tuple[tuple[object, ...], ...]
    handles: tuple[ASTLiveLaneStateHandle, ...]


@dataclass(frozen=True, slots=True)
class _PreparedRegistryPlan:
    prepared: ASTPreparedPhysicalPublication
    next_state: _RegistrySnapshot
    published: ASTPublishedPhysicalRound
    _seal: object


class ASTLiveLaneStateLease:
    def __init__(self, registry, lease_id, base, seal) -> None:
        if seal is not _LEASE_SEAL:
            raise ASTLiveStateError("live-state lease is unbound")
        self._registry = registry
        self._lease_id = lease_id
        self._base = base
        self._closed = False

    def _check(self) -> _RegistrySnapshot:
        state = self._registry._state
        if self._closed or state.active_lease is not self._lease_id:
            raise ASTLiveStateError("live-state lease is closed or stale")
        return state

    def validate_base(self, *, authority: ASTLiveStateAuthority) -> None:
        self._check()
        self._registry._validate_authority(authority)
        self._registry._validate_base(self._base.rows, self._base.positions, self._base)

    def validate_segmented_successor(
        self,
        rows: Sequence[Sequence[object]],
        positions: Sequence[int],
        *,
        authority: ASTLiveStateAuthority,
    ) -> None:
        self._check()
        self._registry._validate_authority(authority)
        if _row_state_object_ids(self._base.rows) & _row_state_object_ids(rows):
            raise ASTLiveStateError(
                "private successor state must be disjoint from registered base state"
            )
        _sign_rows(
            rows,
            tuple(_exact_int(position, "lane position") for position in positions),
            publication="segmented",
            max_lanes=self._registry._authority.max_lanes,
            max_caches_per_lane=self._registry._authority.max_caches_per_lane,
            required_plane_kinds=self._registry._authority.required_plane_kinds,
        )

    def _validate_publication_membership(
        self,
        prepared: ASTPreparedPhysicalPublication,
    ) -> None:
        authority = self._registry._authority
        if prepared.publication_mode not in authority.publication_modes:
            raise ASTLiveStateError("publication mode is not authorized")
        if prepared.publication_mode == "in_place_base":
            expected = self._base.rows
        elif prepared.publication_mode == "transfer_successor":
            expected = prepared.successor_rows
        else:  # Defensive even though the sealed envelope validates vocabulary.
            raise ASTLiveStateError("publication mode is invalid")
        if not _same_row_membership(prepared.publication_rows, expected):
            raise ASTLiveStateError(
                "publication rows differ from the authority-bound destination"
            )

    def prepare_plan(
        self,
        prepared: ASTPreparedPhysicalPublication,
        *,
        result: ASTPhysicalResult,
        authority: ASTLiveStateAuthority,
    ) -> _PreparedRegistryPlan:
        state = self._check()
        self.validate_base(authority=authority)
        if type(prepared) is not ASTPreparedPhysicalPublication:
            raise ASTLiveStateError("delegate returned an untyped prepared publication")
        if prepared.descriptor_geometry_sha256 != result.geometry_sha256:
            raise ASTLiveStateError("prepared descriptor geometry differs")
        if prepared.result_manifest_sha256 != result.manifest_sha256:
            raise ASTLiveStateError("prepared result identity differs")
        self._validate_publication_membership(prepared)
        accepted = tuple(
            _exact_int(value, "accepted length", minimum=1)
            for value in prepared.accepted_lengths
        )
        if len(accepted) != len(state.rows):
            raise ASTLiveStateError("accepted length count differs from lanes")
        positions = tuple(
            _exact_int(value, "successor position")
            for value in prepared.successor_positions
        )
        expected_positions = tuple(
            position + accepted[lane] for lane, position in enumerate(state.positions)
        )
        if positions != expected_positions:
            raise ASTLiveStateError("successor positions differ from accepted prefixes")
        publication_rows, signatures = _sign_rows(
            prepared.publication_rows,
            positions,
            publication="public",
            field_rows=prepared.successor_rows,
            max_lanes=self._registry._authority.max_lanes,
            max_caches_per_lane=self._registry._authority.max_caches_per_lane,
            required_plane_kinds=self._registry._authority.required_plane_kinds,
        )
        generations = tuple(handle.generation + 1 for handle in state.handles)
        next_epoch = state.epoch + 1
        handles = self._registry._make_handles(
            epoch=next_epoch,
            generations=generations,
            positions=positions,
            signatures=signatures,
        )
        next_state = _RegistrySnapshot(
            next_epoch,
            publication_rows,
            positions,
            signatures,
            handles,
            None,
        )
        published = ASTPublishedPhysicalRound(
            result, accepted, publication_rows, handles
        )
        return _PreparedRegistryPlan(prepared, next_state, published, _PLAN_SEAL)

    def validate_plan(
        self,
        plan: _PreparedRegistryPlan,
        *,
        authority: ASTLiveStateAuthority,
    ) -> None:
        self._check()
        self.validate_base(authority=authority)
        if type(plan) is not _PreparedRegistryPlan or plan._seal is not _PLAN_SEAL:
            raise ASTLiveStateError("prepared registry plan is unbound")
        if plan.next_state.epoch != self._base.epoch + 1:
            raise ASTLiveStateError("prepared registry epoch differs")
        self.validate_segmented_successor(
            plan.prepared.successor_rows,
            plan.prepared.successor_positions,
            authority=authority,
        )
        self._validate_publication_membership(plan.prepared)
        publication_rows, signatures = _sign_rows(
            plan.prepared.publication_rows,
            plan.prepared.successor_positions,
            publication="public",
            field_rows=plan.prepared.successor_rows,
            max_lanes=self._registry._authority.max_lanes,
            max_caches_per_lane=self._registry._authority.max_caches_per_lane,
            required_plane_kinds=self._registry._authority.required_plane_kinds,
        )
        if (
            not _same_row_membership(publication_rows, plan.next_state.rows)
            or signatures != plan.next_state.signatures
        ):
            raise ASTLiveStateError("prepared successor membership drifted")

    def publish_noexcept(
        self, plan: _PreparedRegistryPlan
    ) -> ASTPublishedPhysicalRound:
        """One preallocated pointer swap after the delegate returns successfully."""

        self._registry._state = plan.next_state
        self._closed = True
        return plan.published

    def abort(self, *, authority: ASTLiveStateAuthority) -> None:
        if self._closed:
            raise ASTLiveStateError("live-state lease is already closed")
        try:
            self.validate_base(authority=authority)
        except BaseException:
            current = self._registry._state
            self._registry._state = _RegistrySnapshot(
                current.epoch,
                current.rows,
                current.positions,
                current.signatures,
                current.handles,
                None,
                True,
            )
            self._closed = True
            raise
        self._registry._state = self._base
        self._closed = True


@dataclass(frozen=True, slots=True)
class ASTPhysicalResultPlan:
    """Token-free physical storage geometry fixed before target execution."""

    geometry_sha256: str
    storage: str
    lane_count: int
    logical_rows: int
    max_segment_rows: int
    physical_rows: int
    rectangular_width: int | None

    def __post_init__(self) -> None:
        _digest(self.geometry_sha256, "result plan geometry")
        if self.storage not in _RESULT_STORAGE_CLASSES:
            raise ASTLiveStateError("result plan storage class is unsupported")
        _exact_int(self.lane_count, "result plan lane count", minimum=1)
        _exact_int(self.logical_rows, "result plan logical rows", minimum=1)
        _exact_int(
            self.max_segment_rows,
            "result plan max segment rows",
            minimum=1,
        )
        _exact_int(self.physical_rows, "result plan physical rows", minimum=1)
        if self.storage == "rectangular_rows":
            width = _exact_int(
                self.rectangular_width,
                "result plan rectangular width",
                minimum=self.max_segment_rows,
            )
            if self.physical_rows != self.lane_count * width:
                raise ASTLiveStateError("rectangular result plan row law differs")
        elif (
            self.rectangular_width is not None
            or self.physical_rows != self.logical_rows
        ):
            raise ASTLiveStateError("flat result plan row law differs")
        if self.physical_rows < self.logical_rows:
            raise ASTLiveStateError("result plan truncates logical rows")

    @classmethod
    def from_descriptor(
        cls,
        descriptor: FlatSegmentDescriptor,
        *,
        storage: str,
        rectangular_width: int | None = None,
    ) -> ASTPhysicalResultPlan:
        if not isinstance(descriptor, FlatSegmentDescriptor):
            raise ASTLiveStateError("result plan descriptor is unbound")
        if storage == "rectangular_rows":
            width = _exact_int(
                rectangular_width,
                "result plan rectangular width",
                minimum=descriptor.max_segment_rows,
            )
            physical_rows = descriptor.lane_count * width
        else:
            width = rectangular_width
            physical_rows = descriptor.logical_rows
        return cls(
            descriptor.geometry_sha256,
            storage,
            descriptor.lane_count,
            descriptor.logical_rows,
            descriptor.max_segment_rows,
            physical_rows,
            width,
        )

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.receipt())

    def receipt(self) -> dict[str, object]:
        return {
            "schema": "mlx2.ast-physical-result-plan.v1",
            "geometry_sha256": self.geometry_sha256,
            "storage": self.storage,
            "lane_count": self.lane_count,
            "logical_rows": self.logical_rows,
            "max_segment_rows": self.max_segment_rows,
            "physical_rows": self.physical_rows,
            "padding_rows": self.physical_rows - self.logical_rows,
            "rectangular_width": self.rectangular_width,
            "tensor_values_serialized": False,
            "apcv2_serializable": False,
            "numerical_claim": False,
        }

    def validate_for(
        self,
        descriptor: FlatSegmentDescriptor,
        authority: ASTLiveStateAuthority,
    ) -> None:
        if type(self) is not ASTPhysicalResultPlan:
            raise ASTLiveStateError("physical result plan is unbound")
        if (
            self.geometry_sha256 != descriptor.geometry_sha256
            or self.lane_count != descriptor.lane_count
            or self.logical_rows != descriptor.logical_rows
            or self.max_segment_rows != descriptor.max_segment_rows
        ):
            raise ASTLiveStateError("physical result plan geometry differs")
        if self.storage not in authority.result_storage_classes:
            raise ASTLiveStateError("result storage class is not authorized")
        if self.physical_rows > authority.max_physical_rows:
            raise ASTLiveStateError("planned physical rows exceed live-state authority")

    def validate_result(self, result: ASTPhysicalResult) -> None:
        if type(result) is not ASTPhysicalResult or (
            result.geometry_sha256 != self.geometry_sha256
            or result.storage != self.storage
            or result.physical_rows != self.physical_rows
            or result.width != self.rectangular_width
        ):
            raise ASTLiveStateError("delegate physical result differs from its plan")


class ASTPhysicalResult:
    """Non-materializing view of existing flat or rectangular target output."""

    __slots__ = (
        "_frozen",
        "descriptor",
        "features",
        "features_dtype",
        "features_shape",
        "geometry_sha256",
        "lane_slices",
        "logits",
        "logits_dtype",
        "logits_shape",
        "manifest_sha256",
        "padding_rows",
        "physical_rows",
        "rows",
        "storage",
        "width",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("ASTPhysicalResult is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        descriptor: FlatSegmentDescriptor,
        logits: object,
        features: object | None,
        *,
        storage: str,
        invocation_count: int,
    ) -> None:
        if not isinstance(descriptor, FlatSegmentDescriptor):
            raise ASTLiveStateError("physical result requires a flat descriptor")
        if type(invocation_count) is not int or invocation_count != 1:
            raise ASTLiveStateError("physical result requires one outer invocation")
        rows = None
        if storage == "flat_row_pairs":
            if features is not None or not isinstance(logits, tuple):
                raise ASTLiveStateError(
                    "flat row-pair storage requires one immutable row sequence"
                )
            if len(logits) != descriptor.logical_rows:
                raise ASTLiveStateError("flat row-pair count differs from descriptor")
            row_logits_shape = row_features_shape = None
            row_logits_dtype = row_features_dtype = None
            for row in logits:
                if not isinstance(row, tuple) or len(row) != 2:
                    raise ASTLiveStateError("flat row-pair storage is malformed")
                current_logits_shape = _shape(row[0], "row logits")
                current_features_shape = _shape(row[1], "row features")
                if any(dimension == 0 for dimension in current_logits_shape) or any(
                    dimension == 0 for dimension in current_features_shape
                ):
                    raise ASTLiveStateError(
                        "flat row-pair payload dimensions must be positive"
                    )
                current_logits_dtype = _dtype(row[0], "row logits")
                current_features_dtype = _dtype(row[1], "row features")
                if row_logits_shape is None:
                    row_logits_shape = current_logits_shape
                    row_features_shape = current_features_shape
                    row_logits_dtype = current_logits_dtype
                    row_features_dtype = current_features_dtype
                elif (
                    current_logits_shape != row_logits_shape
                    or current_features_shape != row_features_shape
                    or current_logits_dtype != row_logits_dtype
                    or current_features_dtype != row_features_dtype
                ):
                    raise ASTLiveStateError(
                        "flat row-pair shapes or dtypes are inconsistent"
                    )
            rows = logits
            logits_shape = (descriptor.logical_rows, *row_logits_shape)
            feature_shape = (descriptor.logical_rows, *row_features_shape)
            logits_dtype = row_logits_dtype
            features_dtype = row_features_dtype
            width = None
            physical_rows = descriptor.logical_rows
            lane_slices = tuple(
                (
                    None,
                    descriptor.cu_query_lengths[lane],
                    descriptor.cu_query_lengths[lane + 1],
                )
                for lane in range(descriptor.lane_count)
            )
        else:
            logits_shape = _shape(logits, "logits")
            feature_shape = None if features is None else _shape(features, "features")
            logits_dtype = _dtype(logits, "logits")
            features_dtype = None if features is None else _dtype(features, "features")
            if storage == "flat_logical_rows":
                if (
                    len(logits_shape) < 2
                    or any(dimension == 0 for dimension in logits_shape[1:])
                    or logits_shape[0] != descriptor.logical_rows
                    or (
                        feature_shape is not None
                        and (
                            len(feature_shape) < 2
                            or any(dimension == 0 for dimension in feature_shape[1:])
                            or feature_shape[0] != descriptor.logical_rows
                        )
                    )
                ):
                    raise ASTLiveStateError(
                        "flat storage shape differs from the result ABI"
                    )
                width = None
                physical_rows = descriptor.logical_rows
                lane_slices = tuple(
                    (
                        None,
                        descriptor.cu_query_lengths[lane],
                        descriptor.cu_query_lengths[lane + 1],
                    )
                    for lane in range(descriptor.lane_count)
                )
            elif storage == "rectangular_rows":
                if (
                    len(logits_shape) < 3
                    or any(dimension == 0 for dimension in logits_shape[2:])
                    or logits_shape[0] != descriptor.lane_count
                ):
                    raise ASTLiveStateError(
                        "rectangular logits shape differs from the result ABI"
                    )
                width = logits_shape[1]
                if width < descriptor.max_segment_rows:
                    raise ASTLiveStateError("rectangular width truncates a segment")
                if feature_shape is not None and (
                    len(feature_shape) < 3
                    or any(dimension == 0 for dimension in feature_shape[2:])
                    or feature_shape[:2] != logits_shape[:2]
                ):
                    raise ASTLiveStateError(
                        "rectangular feature shape differs from the result ABI"
                    )
                physical_rows = descriptor.lane_count * width
                lane_slices = tuple(
                    (lane, 0, length)
                    for lane, length in enumerate(descriptor.query_lengths)
                )
            else:
                raise ASTLiveStateError("physical result storage class is unsupported")
        self.descriptor = descriptor
        self.logits = None if rows is not None else logits
        self.features = features
        self.rows = rows
        self.logits_shape = logits_shape
        self.logits_dtype = logits_dtype
        self.features_shape = feature_shape
        self.features_dtype = features_dtype
        self.storage = storage
        self.physical_rows = physical_rows
        self.padding_rows = physical_rows - descriptor.logical_rows
        self.width = width
        self.lane_slices = lane_slices
        self.geometry_sha256 = descriptor.geometry_sha256
        self.manifest_sha256 = _sha256(self.receipt())
        self._frozen = True

    @classmethod
    def from_native_execution(
        cls,
        execution: NativeFlatSegmentExecution,
        descriptor: FlatSegmentDescriptor,
    ) -> ASTPhysicalResult:
        execution = validate_native_flat_segment_execution(execution, descriptor)
        return cls(
            descriptor,
            execution.outputs,
            None,
            storage="flat_row_pairs",
            invocation_count=execution.invocation_count,
        )

    def lane(self, lane: int) -> tuple[object, object | None]:
        lane = _exact_int(lane, "lane")
        if lane >= self.descriptor.lane_count:
            raise ASTLiveStateError("lane is out of range")
        storage_lane, start, stop = self.lane_slices[lane]
        if self.storage == "flat_row_pairs":
            lane_rows = self.rows[start:stop]
            return (
                tuple(row[0] for row in lane_rows),
                tuple(row[1] for row in lane_rows),
            )
        if self.storage == "flat_logical_rows":
            return (
                self.logits[start:stop],
                None if self.features is None else self.features[start:stop],
            )
        return (
            self.logits[storage_lane, start:stop],
            None if self.features is None else self.features[storage_lane, start:stop],
        )

    def row(self, lane: int, local_position: int) -> tuple[object, object | None]:
        lane = _exact_int(lane, "lane")
        local_position = _exact_int(local_position, "local_position")
        if (
            lane >= self.descriptor.lane_count
            or local_position >= self.descriptor.query_lengths[lane]
        ):
            raise ASTLiveStateError("physical result row is out of range")
        if self.storage in {"flat_logical_rows", "flat_row_pairs"}:
            index = self.descriptor.cu_query_lengths[lane] + local_position
            if self.storage == "flat_row_pairs":
                row = self.rows[index]
                return row[0], row[1]
            return (
                self.logits[index],
                None if self.features is None else self.features[index],
            )
        return (
            self.logits[lane, local_position],
            None if self.features is None else self.features[lane, local_position],
        )

    def receipt(self) -> dict[str, object]:
        return {
            "schema": "mlx2.ast-physical-result.v1",
            "geometry_sha256": self.geometry_sha256,
            "storage": self.storage,
            "logical_rows": self.descriptor.logical_rows,
            "physical_rows": self.physical_rows,
            "padding_rows": self.padding_rows,
            "rectangular_width": self.width,
            "lane_slices": self.lane_slices,
            "logits_shape": self.logits_shape,
            "logits_dtype": self.logits_dtype,
            "features_shape": self.features_shape,
            "features_dtype": self.features_dtype,
            "invocation_count": 1,
            "tensor_values_serialized": False,
            "apcv2_serializable": False,
            "numerical_claim": False,
            "implemented": True,
            "qualified": False,
            "selected": False,
            "default": False,
            "observed_used": False,
        }


class ASTPhysicalCommitCoordinator:
    """Coordinate a private delegate with a preallocated registry transition.

    A normal return from ``delegate.publish_commit`` is the delegate's
    authoritative assertion that exactly the prepared successor was published.
    The delegate must roll back internally before raising from that method;
    generic code cannot repair a partially irrevocable publication.
    """

    def __init__(
        self,
        registry: ASTLiveLaneStateRegistry,
        handles: Sequence[ASTLiveLaneStateHandle],
        rows: Sequence[Sequence[object]],
        positions: Sequence[int],
        *,
        descriptor: FlatSegmentDescriptor,
        result_plan: ASTPhysicalResultPlan,
        delegate: object,
        authority: ASTLiveStateAuthority,
    ) -> None:
        if not isinstance(registry, ASTLiveLaneStateRegistry):
            raise ASTLiveStateError("coordinator registry is unbound")
        registry._validate_authority(authority)
        if not isinstance(descriptor, FlatSegmentDescriptor):
            raise ASTLiveStateError("coordinator descriptor is unbound")
        if descriptor.lane_count != len(registry._state.rows):
            raise ASTLiveStateError("descriptor lanes differ from live-state lanes")
        if (
            descriptor.lane_count > authority.max_lanes
            or descriptor.max_segment_rows > authority.max_segment_rows
            or descriptor.logical_rows > authority.max_total_rows
        ):
            raise ASTLiveStateError("descriptor geometry exceeds live-state authority")
        if type(result_plan) is not ASTPhysicalResultPlan:
            raise ASTLiveStateError("physical result plan is unbound")
        result_plan.validate_for(descriptor, authority)
        for name in (
            "execute_private",
            "prepare_commit",
            "publish_commit",
            "abort",
        ):
            if not callable(getattr(delegate, name, None)):
                raise ASTLiveStateError("publication delegate contract is incomplete")
        self._delegate = delegate
        self._result = None
        self._descriptor = descriptor
        self._result_plan = result_plan
        self._authority = authority
        self._prepare_authority = _ASTPhysicalPrepareAuthority(delegate)
        self._lease = registry.begin(handles, rows, positions, authority=authority)
        self._plan = None
        self._closed = False

    @staticmethod
    def contract_receipt() -> dict[str, object]:
        return {
            "schema": "mlx2.ast-physical-publication-contract.v1",
            "state_write_law": "disjoint_private_successor_until_publish",
            "execution_law": "lease_acquired_before_delegate_private_execution",
            "physical_capacity_law": (
                "token_free_result_plan_authorized_before_lease_and_execution"
            ),
            "prepare_law": "complete_successor_snapshot_preallocated_before_publish",
            "publish_law": "delegate_atomic_publish_or_internal_rollback",
            "finalize_law": "one_preallocated_registry_pointer_assignment",
            "abort_law": "base_membership_exact_or_registry_poisoned",
            "serving_wired": False,
            "selection_eligible": False,
            "implemented": True,
            "qualified": False,
            "selected": False,
            "default": False,
            "observed_used": False,
        }

    def execute(self, *, authority: ASTLiveStateAuthority) -> ASTPhysicalResult:
        if self._closed or self._result is not None:
            raise ASTLiveStateError("coordinator is closed or already executed")
        try:
            self._lease.validate_base(authority=authority)
            self._result_plan.validate_for(self._descriptor, authority)
            result = self._delegate.execute_private(
                descriptor=self._descriptor,
                result_plan=self._result_plan,
            )
            if type(result) is not ASTPhysicalResult or (
                result.descriptor != self._descriptor
                or result.geometry_sha256 != self._descriptor.geometry_sha256
            ):
                raise ASTLiveStateError("delegate physical result differs")
            self._result_plan.validate_result(result)
            self._lease.validate_base(authority=authority)
            self._result = result
        except BaseException:
            self._abort_preserving_error()
            raise
        return result

    def prepare(
        self,
        accepted_lengths: Sequence[int],
        *,
        authority: ASTLiveStateAuthority,
    ) -> ASTPreparedPhysicalPublication:
        if self._closed or self._result is None or self._plan is not None:
            raise ASTLiveStateError(
                "coordinator is closed, unexecuted, or already prepared"
            )
        accepted = tuple(
            _exact_int(value, "accepted length", minimum=1)
            for value in accepted_lengths
        )
        if len(accepted) != self._descriptor.lane_count or any(
            value > self._descriptor.query_lengths[lane]
            for lane, value in enumerate(accepted)
        ):
            raise ASTLiveStateError("accepted lengths differ from descriptor")
        try:
            self._lease.validate_base(authority=authority)
            prepared = self._delegate.prepare_commit(
                accepted,
                descriptor=self._descriptor,
                result=self._result,
                prepare_authority=self._prepare_authority,
            )
            self._prepare_authority.validate(prepared, self._delegate)
            if prepared.accepted_lengths != accepted:
                raise ASTLiveStateError("delegate prepared envelope differs")
            self._lease.validate_segmented_successor(
                prepared.successor_rows,
                prepared.successor_positions,
                authority=authority,
            )
            self._plan = self._lease.prepare_plan(
                prepared,
                result=self._result,
                authority=authority,
            )
        except BaseException:
            self._abort_preserving_error()
            raise
        return prepared

    def publish(self, *, authority: ASTLiveStateAuthority) -> ASTPublishedPhysicalRound:
        if self._closed or self._plan is None:
            raise ASTLiveStateError("coordinator is not prepared")
        plan = self._plan
        try:
            self._prepare_authority.validate(plan.prepared, self._delegate)
            self._lease.validate_plan(plan, authority=authority)
            self._delegate.publish_commit(plan.prepared)
        except BaseException:
            self._abort_preserving_error()
            raise
        published = self._lease.publish_noexcept(plan)
        self._closed = True
        return published

    def abort(self) -> None:
        if self._closed:
            raise ASTLiveStateError("coordinator is already closed")
        error = None
        try:
            self._delegate.abort()
        except BaseException as caught:  # noqa: BLE001 - lifecycle cleanup
            error = caught
        try:
            self._lease.abort(authority=self._authority)
        except BaseException as caught:  # noqa: BLE001 - lifecycle cleanup
            if error is None:
                error = caught
        self._closed = True
        if error is not None:
            raise error

    def _abort_preserving_error(self) -> None:
        try:
            self._delegate.abort()
        except BaseException:  # noqa: BLE001,S110 - preserve original failure
            pass
        try:
            self._lease.abort(authority=self._authority)
        except BaseException:  # noqa: BLE001,S110 - preserve original failure
            pass
        self._closed = True


__all__ = [
    "ASTLiveLaneStateHandle",
    "ASTLiveLaneStateRegistry",
    "ASTLiveStateAuthority",
    "ASTLiveStateError",
    "ASTPhysicalCommitCoordinator",
    "ASTPhysicalResult",
    "ASTPhysicalResultPlan",
    "ASTPreparedPhysicalPublication",
    "ASTPublishedPhysicalRound",
]
