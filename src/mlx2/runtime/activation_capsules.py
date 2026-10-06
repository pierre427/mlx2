"""Revision-bound approximate activation capsules.

This module deliberately does not persist exact continuation state.  APCv2 owns
exact KV/recurrent state.  Activation capsules are approximate conditioning
artifacts whose tensor bytes are content addressed separately from the existing
semantic-capsule envelope.
"""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
import re
import stat
import struct
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from .semantic_capsules import (
    CapsuleIdentity,
    CapsuleStore,
    canonical_json,
)

ACTIVATION_CAPSULE_SCHEMA = "mlx2-activation-capsule-v1"
TENSOR_PAYLOAD_SCHEMA = "mlx2-activation-tensor-v1"
EXACT_STATE_SCHEMA = "mlx2-apcv2-exact-state"
_MAGIC = b"MLX2ACT1\n"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ALLOWED_DTYPES = frozenset({"<f2", "<f4", "<f8"})
_MAX_HEADER_BYTES = 16 * 1024
_MAX_TEXT_BYTES = 1024
_CAPSULE_FILENAME = re.compile(r"([0-9a-f]{64})\.json\Z")
_TENSOR_FILENAME = re.compile(r"([0-9a-f]{64})\.tensor\Z")
TENSOR_GC_SCHEMA = "mlx2-activation-tensor-gc-v1"


class ActivationCapsuleError(ValueError):
    """A capsule is malformed, incompatible, corrupt, or unauthorized."""


class PayloadKind(str, Enum):
    ORDERED_TRAJECTORY = "ordered_trajectory"
    DIRECTIONAL_RESIDUAL = "directional_residual"
    CONTINUOUS_PREFIX = "continuous_prefix"
    CROSS_ATTENTION_BANK = "cross_attention_bank"


class ApproximationClass(str, Enum):
    APPROXIMATE_CONDITIONING = "approximate_conditioning"


class AuthorityScope(str, Enum):
    REQUEST = "request"
    SESSION = "session"
    PROJECT = "project"
    DOCUMENT = "document"


def _text(name: str, value: object, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
        raise ActivationCapsuleError(f"{name} must be a non-empty bounded string")
    if any(ord(character) < 0x20 for character in value):
        raise ActivationCapsuleError(f"{name} must not contain control characters")
    return value


def _finite_positive(name: str, value: object, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActivationCapsuleError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ActivationCapsuleError(f"{name} must be finite and positive")
    return result


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ActivationCapsuleError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _object(value: object, name: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ActivationCapsuleError(f"{name} must contain exactly {sorted(keys)!r}")
    return value


@dataclass(frozen=True, slots=True)
class ModelBindings:
    source_model: str
    target_model: str
    tokenizer: str
    runtime: str
    projector_revision: str

    def __post_init__(self) -> None:
        for name in ("source_model", "target_model", "tokenizer", "runtime", "projector_revision"):
            _text(name, getattr(self, name))

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class TapBinding:
    source_layer: int
    source_tap: str
    target_layer: int
    target_injection: str

    def __post_init__(self) -> None:
        if isinstance(self.source_layer, bool) or not isinstance(self.source_layer, int) or self.source_layer < 0:
            raise ActivationCapsuleError("source_layer must be a non-negative integer")
        if isinstance(self.target_layer, bool) or not isinstance(self.target_layer, int) or self.target_layer < 0:
            raise ActivationCapsuleError("target_layer must be a non-negative integer")
        _text("source_tap", self.source_tap)
        _text("target_injection", self.target_injection)

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Geometry:
    hidden_width: int
    sequence_length: int
    key_width: int | None = None
    value_width: int | None = None

    def __post_init__(self) -> None:
        for name in ("hidden_width", "sequence_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ActivationCapsuleError(f"{name} must be a positive integer")
        for name in ("key_width", "value_width"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ActivationCapsuleError(f"{name} must be a positive integer when present")

    def to_dict(self) -> dict[str, int | None]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class PositionConvention:
    kind: str
    revision: str
    base: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"none", "absolute", "relative", "rope"}:
            raise ActivationCapsuleError("unsupported position convention")
        _text("position revision", self.revision)
        if self.base is not None:
            _finite_positive("position base", self.base)

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "revision": self.revision, "base": self.base}


@dataclass(frozen=True, slots=True)
class NormGateBounds:
    maximum_relative_norm: float
    minimum_gate: float = 0.0
    maximum_gate: float = 1.0

    def __post_init__(self) -> None:
        maximum_norm = _finite_positive("maximum_relative_norm", self.maximum_relative_norm)
        minimum = _finite_positive("minimum_gate", self.minimum_gate, allow_zero=True)
        maximum = _finite_positive("maximum_gate", self.maximum_gate, allow_zero=True)
        if maximum_norm > 1_000 or minimum > maximum or maximum > 1:
            raise ActivationCapsuleError("invalid norm or gate bounds")

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Authority:
    tenant: str
    scope: AuthorityScope
    scope_id: str
    expires_at_unix_ns: int | None = None

    def __post_init__(self) -> None:
        _text("tenant", self.tenant)
        _text("scope_id", self.scope_id)
        if not isinstance(self.scope, AuthorityScope):
            raise ActivationCapsuleError("scope must be an AuthorityScope")
        if self.expires_at_unix_ns is not None and (
            isinstance(self.expires_at_unix_ns, bool)
            or not isinstance(self.expires_at_unix_ns, int)
            or self.expires_at_unix_ns <= 0
        ):
            raise ActivationCapsuleError("expiry must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "tenant": self.tenant,
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "expires_at_unix_ns": self.expires_at_unix_ns,
        }


@dataclass(frozen=True, slots=True)
class CapsuleProvenance:
    producer: str
    producer_revision: str
    source_digests: tuple[str, ...]
    source_references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text("producer", self.producer)
        _text("producer_revision", self.producer_revision)
        if not self.source_digests:
            raise ActivationCapsuleError("at least one source digest is required")
        for value in self.source_digests:
            _digest("source digest", value)
        if len(set(self.source_digests)) != len(self.source_digests):
            raise ActivationCapsuleError("source digests must be unique")
        for value in self.source_references:
            _text("source reference", value)

    def to_dict(self) -> dict[str, object]:
        return {
            "producer": self.producer,
            "producer_revision": self.producer_revision,
            "source_digests": list(self.source_digests),
            "source_references": list(self.source_references),
        }


@dataclass(frozen=True, slots=True)
class ActivationCapsuleSpec:
    payload_kind: PayloadKind
    bindings: ModelBindings
    tap: TapBinding
    geometry: Geometry
    dtype: str
    normalization: str
    position: PositionConvention
    provenance: CapsuleProvenance
    authority: Authority
    bounds: NormGateBounds
    approximation: ApproximationClass = ApproximationClass.APPROXIMATE_CONDITIONING
    ordered_transition_indices: tuple[int, ...] = ()
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.payload_kind, PayloadKind):
            raise ActivationCapsuleError("payload_kind must be a PayloadKind")
        if self.approximation is not ApproximationClass.APPROXIMATE_CONDITIONING:
            raise ActivationCapsuleError("exact state is APCv2-owned and cannot be an activation capsule")
        if self.dtype not in _ALLOWED_DTYPES:
            raise ActivationCapsuleError("unsupported canonical dtype")
        _text("normalization", self.normalization)
        indices = self.ordered_transition_indices
        if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in indices):
            raise ActivationCapsuleError("transition indices must be positive integers")
        if tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
            raise ActivationCapsuleError("transition indices must be unique and ordered")
        if self.payload_kind is PayloadKind.ORDERED_TRAJECTORY and not indices:
            raise ActivationCapsuleError("ordered trajectories require transition indices")
        if self.payload_kind is not PayloadKind.ORDERED_TRAJECTORY and indices:
            raise ActivationCapsuleError("transition indices are trajectory-only")
        if len(self.metadata) > 64:
            raise ActivationCapsuleError("too many metadata entries")
        for key, value in self.metadata.items():
            _text("metadata key", key)
            if not isinstance(value, (str, int, float, bool)) or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise ActivationCapsuleError("metadata values must be finite JSON scalars")
            if isinstance(value, str):
                _text("metadata value", value)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ACTIVATION_CAPSULE_SCHEMA,
            "payload_kind": self.payload_kind.value,
            "approximation": self.approximation.value,
            "exact_state_schema": EXACT_STATE_SCHEMA,
            "bindings": self.bindings.to_dict(),
            "tap": self.tap.to_dict(),
            "geometry": self.geometry.to_dict(),
            "dtype": self.dtype,
            "normalization": self.normalization,
            "position": self.position.to_dict(),
            "provenance": self.provenance.to_dict(),
            "authority": self.authority.to_dict(),
            "bounds": self.bounds.to_dict(),
            "ordered_transition_indices": list(self.ordered_transition_indices),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: object) -> ActivationCapsuleSpec:
        """Parse an untrusted persisted manifest with exact-key validation."""
        top = _object(
            value,
            "activation manifest",
            {
                "schema", "payload_kind", "approximation", "exact_state_schema",
                "bindings", "tap", "geometry", "dtype", "normalization",
                "position", "provenance", "authority", "bounds",
                "ordered_transition_indices", "metadata",
            },
        )
        if top["schema"] != ACTIVATION_CAPSULE_SCHEMA:
            raise ActivationCapsuleError("unsupported activation capsule schema")
        if top["approximation"] != ApproximationClass.APPROXIMATE_CONDITIONING.value:
            raise ActivationCapsuleError("activation capsules must be approximate conditioning")
        if top["exact_state_schema"] != EXACT_STATE_SCHEMA:
            raise ActivationCapsuleError("exact-state separation marker is missing")
        bindings = _object(
            top["bindings"], "model bindings",
            {"source_model", "target_model", "tokenizer", "runtime", "projector_revision"},
        )
        tap = _object(
            top["tap"], "tap binding",
            {"source_layer", "source_tap", "target_layer", "target_injection"},
        )
        geometry = _object(
            top["geometry"], "geometry",
            {"hidden_width", "sequence_length", "key_width", "value_width"},
        )
        position = _object(top["position"], "position", {"kind", "revision", "base"})
        provenance = _object(
            top["provenance"], "provenance",
            {"producer", "producer_revision", "source_digests", "source_references"},
        )
        authority = _object(
            top["authority"], "authority",
            {"tenant", "scope", "scope_id", "expires_at_unix_ns"},
        )
        bounds = _object(
            top["bounds"], "bounds",
            {"maximum_relative_norm", "minimum_gate", "maximum_gate"},
        )
        indices = top["ordered_transition_indices"]
        metadata = top["metadata"]
        if not isinstance(indices, list) or not isinstance(metadata, Mapping):
            raise ActivationCapsuleError("transition indices and metadata have invalid types")
        source_digests = provenance["source_digests"]
        source_references = provenance["source_references"]
        if not isinstance(source_digests, list) or not isinstance(source_references, list):
            raise ActivationCapsuleError("provenance digest and reference lists are required")
        try:
            return cls(
                payload_kind=PayloadKind(top["payload_kind"]),
                approximation=ApproximationClass(top["approximation"]),
                bindings=ModelBindings(**bindings),
                tap=TapBinding(**tap),
                geometry=Geometry(**geometry),
                dtype=top["dtype"],
                normalization=top["normalization"],
                position=PositionConvention(**position),
                provenance=CapsuleProvenance(
                    producer=provenance["producer"],
                    producer_revision=provenance["producer_revision"],
                    source_digests=tuple(source_digests),
                    source_references=tuple(source_references),
                ),
                authority=Authority(
                    tenant=authority["tenant"],
                    scope=AuthorityScope(authority["scope"]),
                    scope_id=authority["scope_id"],
                    expires_at_unix_ns=authority["expires_at_unix_ns"],
                ),
                bounds=NormGateBounds(**bounds),
                ordered_transition_indices=tuple(indices),
                metadata=dict(metadata),
            )
        except (TypeError, ValueError) as error:
            if isinstance(error, ActivationCapsuleError):
                raise
            raise ActivationCapsuleError("activation manifest contains invalid typed values") from error


@dataclass(frozen=True, slots=True)
class ActivationExpectation:
    target_model: str
    tokenizer: str
    runtime: str
    projector_revision: str
    target_layer: int
    target_injection: str
    tenant: str
    scope: AuthorityScope
    scope_id: str
    payload_kind: PayloadKind

    def __post_init__(self) -> None:
        for name in ("target_model", "tokenizer", "runtime", "projector_revision", "target_injection", "tenant", "scope_id"):
            _text(name, getattr(self, name))
        if isinstance(self.target_layer, bool) or not isinstance(self.target_layer, int) or self.target_layer < 0:
            raise ActivationCapsuleError("target_layer must be a non-negative integer")
        if not isinstance(self.scope, AuthorityScope):
            raise ActivationCapsuleError("expected scope must be an AuthorityScope")
        if not isinstance(self.payload_kind, PayloadKind):
            raise ActivationCapsuleError("expected payload kind must be a PayloadKind")


@dataclass(frozen=True, slots=True)
class StoredTensor:
    digest: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int

    def to_dict(self) -> dict[str, object]:
        return {"digest": self.digest, "dtype": self.dtype, "shape": list(self.shape), "nbytes": self.nbytes}


@dataclass(frozen=True, slots=True)
class TensorObjectInfo:
    """Filesystem identity used to prevent deletion after an inventory race."""

    digest: str
    file_bytes: int
    modified_at_unix_ns: int
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class TensorGCReceipt:
    """Auditable result of an explicit reachability collection request."""

    dry_run: bool
    retention_cutoff_unix_ns: int
    capsule_count: int
    activation_capsule_count: int
    referenced_tensor_count: int
    retained_recent_count: int
    candidates: tuple[str, ...]
    deleted: tuple[str, ...]
    schema: str = TENSOR_GC_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dry_run": self.dry_run,
            "retention_cutoff_unix_ns": self.retention_cutoff_unix_ns,
            "capsule_count": self.capsule_count,
            "activation_capsule_count": self.activation_capsule_count,
            "referenced_tensor_count": self.referenced_tensor_count,
            "retained_recent_count": self.retained_recent_count,
            "candidates": list(self.candidates),
            "deleted": list(self.deleted),
        }


class TensorPayloadStore:
    """Owner-private deterministic tensor store with atomic writes."""

    def __init__(self, root: str | Path, *, maximum_payload_bytes: int = 64 * 1024 * 1024):
        requested = Path(root).expanduser()
        if requested.exists() and requested.is_symlink():
            raise ActivationCapsuleError("tensor root must not be a symlink")
        if isinstance(maximum_payload_bytes, bool) or maximum_payload_bytes <= 0:
            raise ActivationCapsuleError("maximum_payload_bytes must be positive")
        self.root = requested.resolve()
        self.objects = self.root / "objects"
        self.quarantine = self.root / "quarantine"
        self.maximum_payload_bytes = int(maximum_payload_bytes)
        for directory in (self.root, self.objects, self.quarantine):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            if directory.is_symlink() or not directory.is_dir():
                raise ActivationCapsuleError("tensor directories must be real directories")
            os.chmod(directory, 0o700)

    def _path(self, digest: str) -> Path:
        return self.objects / f"{_digest('tensor digest', digest)}.tensor"

    def _canonical(self, value: object) -> tuple[np.ndarray, bytes, dict[str, object]]:
        array = np.asarray(value)
        if array.ndim < 1 or array.ndim > 4 or any(dimension <= 0 for dimension in array.shape):
            raise ActivationCapsuleError("tensor rank must be 1..4 with non-empty dimensions")
        if array.dtype.kind != "f":
            raise ActivationCapsuleError("activation tensors must be floating point")
        dtype = f"<f{array.dtype.itemsize}"
        if dtype not in _ALLOWED_DTYPES:
            raise ActivationCapsuleError("unsupported tensor dtype")
        if array.nbytes > self.maximum_payload_bytes:
            raise ActivationCapsuleError("tensor payload exceeds byte limit")
        canonical = np.ascontiguousarray(array.astype(np.dtype(dtype), copy=False))
        if not np.isfinite(canonical).all():
            raise ActivationCapsuleError("activation tensors must be finite")
        raw = canonical.tobytes(order="C")
        manifest = {
            "schema": TENSOR_PAYLOAD_SCHEMA,
            "dtype": dtype,
            "shape": list(canonical.shape),
            "order": "C",
            "nbytes": len(raw),
        }
        return canonical, raw, manifest

    def put(self, value: object) -> StoredTensor:
        _array, raw, manifest = self._canonical(value)
        header = canonical_json(manifest)
        payload = _MAGIC + struct.pack(">I", len(header)) + header + raw
        digest = hashlib.sha256(payload).hexdigest()
        destination = self._path(digest)
        if destination.exists() or destination.is_symlink():
            self.get(digest)
        else:
            fd, temporary = tempfile.mkstemp(prefix=".tensor-", suffix=".tmp", dir=self.objects)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
                os.chmod(destination, 0o600)
                directory_fd = os.open(self.objects, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return StoredTensor(digest, str(manifest["dtype"]), tuple(manifest["shape"]), len(raw))

    def _quarantine(self, path: Path, digest: str) -> None:
        target = self.quarantine / f"{digest}.{time.time_ns()}.tensor"
        try:
            os.replace(path, target)
        except FileNotFoundError:
            pass

    def get(self, digest: str) -> tuple[np.ndarray, StoredTensor]:
        path = self._path(digest)
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
        except FileNotFoundError:
            raise
        except OSError as error:
            self._quarantine(path, digest)
            raise ActivationCapsuleError("tensor payload must be a regular non-symlink file") from error
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
                raise ActivationCapsuleError("tensor payload must be owner-private and regular")
            maximum_file = self.maximum_payload_bytes + _MAX_HEADER_BYTES + len(_MAGIC) + 4
            if info.st_size > maximum_file:
                raise ActivationCapsuleError("tensor payload exceeds byte limit")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                payload = stream.read(maximum_file + 1)
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ActivationCapsuleError("tensor payload digest mismatch")
            if not payload.startswith(_MAGIC) or len(payload) < len(_MAGIC) + 4:
                raise ActivationCapsuleError("tensor payload header is invalid")
            header_size = struct.unpack(">I", payload[len(_MAGIC):len(_MAGIC) + 4])[0]
            if header_size <= 0 or header_size > _MAX_HEADER_BYTES:
                raise ActivationCapsuleError("tensor payload header exceeds limit")
            split = len(_MAGIC) + 4 + header_size
            import json
            manifest = json.loads(payload[len(_MAGIC) + 4:split])
            if not isinstance(manifest, dict) or manifest.get("schema") != TENSOR_PAYLOAD_SCHEMA:
                raise ActivationCapsuleError("unsupported tensor payload schema")
            dtype = manifest.get("dtype")
            shape_value = manifest.get("shape")
            nbytes = manifest.get("nbytes")
            if dtype not in _ALLOWED_DTYPES or manifest.get("order") != "C":
                raise ActivationCapsuleError("unsupported tensor encoding")
            if not isinstance(shape_value, list) or not 1 <= len(shape_value) <= 4:
                raise ActivationCapsuleError("invalid tensor shape")
            if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape_value):
                raise ActivationCapsuleError("invalid tensor dimension")
            expected = math.prod(shape_value) * np.dtype(dtype).itemsize
            raw = payload[split:]
            if nbytes != expected or len(raw) != expected or expected > self.maximum_payload_bytes:
                raise ActivationCapsuleError("tensor byte geometry mismatch")
            array = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape_value).copy()
            if not np.isfinite(array).all():
                raise ActivationCapsuleError("tensor payload contains nonfinite values")
            # The verified digest covers these exact bytes.  Returning a
            # writable array would create a TOCTOU gap between core load and
            # adapter preparation without changing the content handle.
            array.setflags(write=False)
            stored = StoredTensor(digest, dtype, tuple(shape_value), expected)
            return array, stored
        except (ActivationCapsuleError, UnicodeDecodeError, ValueError, TypeError, KeyError) as error:
            self._quarantine(path, digest)
            raise ActivationCapsuleError(f"quarantined corrupt tensor {digest}") from error
        finally:
            os.close(fd)

    def inventory(self) -> tuple[TensorObjectInfo, ...]:
        """Return a strict non-following inventory of tensor object files."""
        result: list[TensorObjectInfo] = []
        for path in sorted(self.objects.iterdir(), key=lambda item: item.name):
            match = _TENSOR_FILENAME.fullmatch(path.name)
            if match is None:
                raise ActivationCapsuleError(
                    f"unexpected entry in tensor object store: {path.name!r}"
                )
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
                raise ActivationCapsuleError(
                    "tensor inventory contains a nonregular or non-private object"
                )
            result.append(
                TensorObjectInfo(
                    digest=match.group(1),
                    file_bytes=info.st_size,
                    modified_at_unix_ns=info.st_mtime_ns,
                    device=info.st_dev,
                    inode=info.st_ino,
                )
            )
        return tuple(result)

    def delete_if_unchanged(self, expected: TensorObjectInfo) -> bool:
        """Delete one inventoried regular object only if its identity is unchanged."""
        if not isinstance(expected, TensorObjectInfo):
            raise TypeError("expected must be TensorObjectInfo")
        path = self._path(expected.digest)
        try:
            info = path.lstat()
        except FileNotFoundError:
            return False
        observed = (
            info.st_size,
            info.st_mtime_ns,
            info.st_dev,
            info.st_ino,
        )
        recorded = (
            expected.file_bytes,
            expected.modified_at_unix_ns,
            expected.device,
            expected.inode,
        )
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) & 0o077
            or observed != recorded
        ):
            raise ActivationCapsuleError(
                "tensor object changed after inventory; refusing deletion"
            )
        path.unlink()
        return True


def _expected_tensor_names(kind: PayloadKind) -> set[str]:
    return {
        PayloadKind.ORDERED_TRAJECTORY: {"anchor", "basis", "coefficients"},
        PayloadKind.DIRECTIONAL_RESIDUAL: {"residual"},
        PayloadKind.CONTINUOUS_PREFIX: {"prefix"},
        PayloadKind.CROSS_ATTENTION_BANK: {"keys", "values"},
    }[kind]


def _shape_contract(
    kind: PayloadKind,
    shapes: Mapping[str, tuple[int, ...]],
    spec: ActivationCapsuleSpec,
) -> None:
    expected_names = _expected_tensor_names(kind)
    if set(shapes) != expected_names:
        raise ActivationCapsuleError(f"{kind.value} tensors must be exactly {sorted(expected_names)!r}")
    width = spec.geometry.hidden_width
    if kind is PayloadKind.ORDERED_TRAJECTORY:
        count = len(spec.ordered_transition_indices)
        rank = shapes["basis"][0] if len(shapes["basis"]) == 2 else -1
        if shapes["anchor"] != (width,) or shapes["basis"] != (rank, width) or shapes["coefficients"] != (count, rank):
            raise ActivationCapsuleError("trajectory tensor geometry mismatch")
        if rank <= 0 or rank > min(count, width) or spec.geometry.sequence_length <= max(spec.ordered_transition_indices):
            raise ActivationCapsuleError("trajectory rank or transition index mismatch")
    elif kind is PayloadKind.DIRECTIONAL_RESIDUAL:
        if spec.geometry.sequence_length != 1 or shapes["residual"] != (width,):
            raise ActivationCapsuleError("directional residual geometry mismatch")
    elif kind is PayloadKind.CONTINUOUS_PREFIX:
        if shapes["prefix"] != (spec.geometry.sequence_length, width):
            raise ActivationCapsuleError("continuous prefix geometry mismatch")
    else:
        key_width = spec.geometry.key_width
        value_width = spec.geometry.value_width
        if key_width is None or value_width is None:
            raise ActivationCapsuleError("cross-attention geometry requires key and value widths")
        if shapes["keys"] != (spec.geometry.sequence_length, key_width) or shapes["values"] != (spec.geometry.sequence_length, value_width):
            raise ActivationCapsuleError("cross-attention bank geometry mismatch")


def _tensor_contract(kind: PayloadKind, tensors: Mapping[str, np.ndarray], spec: ActivationCapsuleSpec) -> None:
    expected_names = _expected_tensor_names(kind)
    if set(tensors) != expected_names:
        raise ActivationCapsuleError(f"{kind.value} tensors must be exactly {sorted(expected_names)!r}")
    shapes = {name: tuple(np.asarray(value).shape) for name, value in tensors.items()}
    _shape_contract(kind, shapes, spec)
    for name, value in tensors.items():
        array = np.asarray(value)
        if f"<f{array.dtype.itemsize}" != spec.dtype:
            raise ActivationCapsuleError(f"tensor {name!r} does not match declared dtype")


def _payload_reference_contract(
    spec: ActivationCapsuleSpec,
    references: object,
    *,
    maximum_tensor_bytes: int,
    maximum_total_bytes: int,
) -> dict[str, StoredTensor]:
    if not isinstance(references, Mapping):
        raise ActivationCapsuleError("activation capsule payload references must be a mapping")
    expected_names = _expected_tensor_names(spec.payload_kind)
    if set(references) != expected_names:
        raise ActivationCapsuleError("activation capsule tensor names do not match its payload kind")
    stored: dict[str, StoredTensor] = {}
    total = 0
    for name, reference in references.items():
        if not isinstance(name, str) or not isinstance(reference, Mapping) or set(reference) != {
            "digest", "dtype", "shape", "nbytes"
        }:
            raise ActivationCapsuleError("tensor reference is malformed")
        digest = _digest("tensor digest", reference["digest"])
        dtype = reference["dtype"]
        shape_value = reference["shape"]
        nbytes = reference["nbytes"]
        if dtype not in _ALLOWED_DTYPES or dtype != spec.dtype:
            raise ActivationCapsuleError("tensor reference dtype mismatch")
        if not isinstance(shape_value, list) or not 1 <= len(shape_value) <= 4:
            raise ActivationCapsuleError("tensor reference shape is invalid")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in shape_value
        ):
            raise ActivationCapsuleError("tensor reference dimension is invalid")
        expected_bytes = math.prod(shape_value) * np.dtype(dtype).itemsize
        if (
            isinstance(nbytes, bool)
            or not isinstance(nbytes, int)
            or nbytes != expected_bytes
            or nbytes > maximum_tensor_bytes
        ):
            raise ActivationCapsuleError("tensor reference byte geometry mismatch")
        item = StoredTensor(digest, dtype, tuple(shape_value), nbytes)
        stored[name] = item
        total += nbytes
    if total > maximum_total_bytes:
        raise ActivationCapsuleError("capsule tensor payloads exceed total byte limit")
    _shape_contract(
        spec.payload_kind,
        {name: item.shape for name, item in stored.items()},
        spec,
    )
    return stored


class ActivationCapsuleBus:
    """Publish and resolve typed activation capsules through semantic envelopes."""

    def __init__(self, capsules: CapsuleStore, tensors: TensorPayloadStore, *, maximum_total_bytes: int = 128 * 1024 * 1024):
        self.capsules = capsules
        self.tensors = tensors
        if isinstance(maximum_total_bytes, bool) or maximum_total_bytes <= 0:
            raise ActivationCapsuleError("maximum_total_bytes must be positive")
        self.maximum_total_bytes = maximum_total_bytes

    @contextmanager
    def _exclusive_store_lock(self):
        """Serialize publication and collection across bus instances/processes."""
        path = self.capsules.root / ".activation-capsule-bus.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags, 0o600)
        except OSError as error:
            raise ActivationCapsuleError("activation capsule store lock is unavailable") from error
        locked = False
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
                raise ActivationCapsuleError(
                    "activation capsule store lock must be owner-private and regular"
                )
            fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
            yield
        finally:
            if locked:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def publish(
        self,
        spec: ActivationCapsuleSpec,
        tensors: Mapping[str, object],
        *,
        parent_capsules: Sequence[str] = (),
    ) -> CapsuleIdentity:
        if not isinstance(spec, ActivationCapsuleSpec) or not isinstance(tensors, Mapping):
            raise TypeError("spec and tensors have invalid types")
        arrays = {name: np.asarray(value) for name, value in tensors.items()}
        _tensor_contract(spec.payload_kind, arrays, spec)
        if sum(array.nbytes for array in arrays.values()) > self.maximum_total_bytes:
            raise ActivationCapsuleError("capsule tensor payloads exceed total byte limit")
        with self._exclusive_store_lock():
            stored = {name: self.tensors.put(value) for name, value in arrays.items()}
            if sum(item.nbytes for item in stored.values()) > self.maximum_total_bytes:
                raise ActivationCapsuleError("capsule tensor payloads exceed total byte limit")
            data = {
                "activation": spec.to_dict(),
                "payloads": {name: item.to_dict() for name, item in sorted(stored.items())},
            }
            provenance = {
                "producer": spec.provenance.producer,
                "producer_revision": spec.provenance.producer_revision,
                "source_digests": list(spec.provenance.source_digests),
            }
            return self.capsules.put(
                kind="activation_capsule",
                data=data,
                parents=parent_capsules,
                model_binding=spec.bindings.target_model,
                tokenizer_binding=spec.bindings.tokenizer,
                runtime_binding=spec.bindings.runtime,
                provenance=provenance,
            )

    def collect_orphaned_tensors(
        self,
        *,
        retention_cutoff_unix_ns: int,
        dry_run: bool = True,
    ) -> TensorGCReceipt:
        """Find or delete old tensors unreachable from every activation envelope.

        Collection is explicit and dry-run by default.  Any malformed capsule
        inventory, invalid activation manifest/reference, missing referenced
        tensor, symlink, nonregular object, or filesystem race aborts the run.
        """
        if (
            isinstance(retention_cutoff_unix_ns, bool)
            or not isinstance(retention_cutoff_unix_ns, int)
            or retention_cutoff_unix_ns <= 0
        ):
            raise ActivationCapsuleError(
                "retention_cutoff_unix_ns must be a positive integer"
            )
        if not isinstance(dry_run, bool):
            raise ActivationCapsuleError("dry_run must be boolean")

        with self._exclusive_store_lock():
            capsule_count = 0
            activation_count = 0
            references_by_digest: dict[str, StoredTensor] = {}
            for path in sorted(self.capsules.objects.iterdir(), key=lambda item: item.name):
                match = _CAPSULE_FILENAME.fullmatch(path.name)
                if match is None:
                    raise ActivationCapsuleError(
                        f"unexpected entry in capsule object store: {path.name!r}"
                    )
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
                    raise ActivationCapsuleError(
                        "capsule inventory contains a nonregular or non-private object"
                    )
                envelope = self.capsules.get(match.group(1))
                capsule_count += 1
                if envelope.get("kind") != "activation_capsule":
                    continue
                activation_count += 1
                data = envelope.get("data")
                if not isinstance(data, Mapping) or set(data) != {"activation", "payloads"}:
                    raise ActivationCapsuleError("activation capsule data is malformed")
                spec = ActivationCapsuleSpec.from_dict(data["activation"])
                payloads = _payload_reference_contract(
                    spec,
                    data["payloads"],
                    maximum_tensor_bytes=self.tensors.maximum_payload_bytes,
                    maximum_total_bytes=self.maximum_total_bytes,
                )
                bindings = envelope.get("bindings")
                if not isinstance(bindings, Mapping) or bindings != {
                    "model": spec.bindings.target_model,
                    "tokenizer": spec.bindings.tokenizer,
                    "runtime": spec.bindings.runtime,
                }:
                    raise ActivationCapsuleError(
                        "activation envelope bindings do not match its manifest"
                    )
                for item in payloads.values():
                    prior = references_by_digest.setdefault(item.digest, item)
                    if prior != item:
                        raise ActivationCapsuleError(
                            "one tensor digest has conflicting reference metadata"
                        )

            for digest, expected in references_by_digest.items():
                try:
                    _tensor, observed = self.tensors.get(digest)
                except (FileNotFoundError, ActivationCapsuleError) as error:
                    raise ActivationCapsuleError(
                        "activation capsule references a missing or corrupt tensor"
                    ) from error
                if observed != expected:
                    raise ActivationCapsuleError(
                        "referenced tensor metadata does not match its capsule"
                    )

            inventory = self.tensors.inventory()
            referenced = set(references_by_digest)
            candidates = tuple(
                item
                for item in inventory
                if item.digest not in referenced
                and item.modified_at_unix_ns < retention_cutoff_unix_ns
            )
            retained_recent = sum(
                item.digest not in referenced
                and item.modified_at_unix_ns >= retention_cutoff_unix_ns
                for item in inventory
            )
            deleted: list[str] = []
            if not dry_run:
                # Preflight the complete tensor inventory a second time before
                # the first unlink. Publication uses the same exclusive lock.
                if self.tensors.inventory() != inventory:
                    raise ActivationCapsuleError(
                        "tensor inventory changed during collection; refusing deletion"
                    )
                for item in candidates:
                    if not self.tensors.delete_if_unchanged(item):
                        raise ActivationCapsuleError(
                            "tensor disappeared during collection; refusing partial result"
                        )
                    deleted.append(item.digest)
            return TensorGCReceipt(
                dry_run=dry_run,
                retention_cutoff_unix_ns=retention_cutoff_unix_ns,
                capsule_count=capsule_count,
                activation_capsule_count=activation_count,
                referenced_tensor_count=len(referenced),
                retained_recent_count=retained_recent,
                candidates=tuple(item.digest for item in candidates),
                deleted=tuple(deleted),
            )

    def load(
        self,
        digest: str,
        expectation: ActivationExpectation,
        *,
        now_unix_ns: int | None = None,
    ) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        envelope = self.capsules.get(digest)
        if envelope.get("kind") != "activation_capsule":
            raise ActivationCapsuleError("capsule is not an activation capsule")
        data = envelope.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("activation"), dict) or not isinstance(data.get("payloads"), dict):
            raise ActivationCapsuleError("activation capsule data is malformed")
        manifest = data["activation"]
        parsed = ActivationCapsuleSpec.from_dict(manifest)
        bindings = manifest["bindings"]
        tap = manifest["tap"]
        authority = manifest["authority"]
        checks = {
            "target_model": (bindings, expectation.target_model),
            "tokenizer": (bindings, expectation.tokenizer),
            "runtime": (bindings, expectation.runtime),
            "projector_revision": (bindings, expectation.projector_revision),
            "target_layer": (tap, expectation.target_layer),
            "target_injection": (tap, expectation.target_injection),
            "tenant": (authority, expectation.tenant),
            "scope": (authority, expectation.scope.value),
            "scope_id": (authority, expectation.scope_id),
        }
        for name, (container, expected) in checks.items():
            if not isinstance(container, dict) or container.get(name) != expected:
                raise ActivationCapsuleError(f"activation capsule {name} mismatch")
        if parsed.payload_kind is not expectation.payload_kind:
            raise ActivationCapsuleError("activation capsule payload kind mismatch")
        expiry = authority.get("expires_at_unix_ns")
        clock = time.time_ns() if now_unix_ns is None else now_unix_ns
        if expiry is not None and (not isinstance(expiry, int) or clock >= expiry):
            raise ActivationCapsuleError("activation capsule authority has expired")
        try:
            expected_payloads = _payload_reference_contract(
                parsed,
                data["payloads"],
                maximum_tensor_bytes=self.tensors.maximum_payload_bytes,
                maximum_total_bytes=self.maximum_total_bytes,
            )
        except ActivationCapsuleError as error:
            raise ActivationCapsuleError("tensor reference metadata mismatch") from error
        loaded: dict[str, np.ndarray] = {}
        total = 0
        for name, expected in expected_payloads.items():
            tensor, stored = self.tensors.get(expected.digest)
            if stored != expected:
                raise ActivationCapsuleError("tensor reference metadata mismatch")
            total += stored.nbytes
            loaded[name] = tensor
        if total > self.maximum_total_bytes:
            raise ActivationCapsuleError("capsule tensor payloads exceed total byte limit")
        _tensor_contract(parsed.payload_kind, loaded, parsed)
        return manifest, loaded


@dataclass(frozen=True, slots=True)
class CompiledTrajectory:
    anchor: np.ndarray
    basis: np.ndarray
    coefficients: np.ndarray
    transition_indices: tuple[int, ...]
    reconstruction_relative_error: float

    def tensors(self) -> dict[str, np.ndarray]:
        return {"anchor": self.anchor, "basis": self.basis, "coefficients": self.coefficients}

    def reconstructed_deltas(self) -> np.ndarray:
        return self.coefficients @ self.basis


def compile_pruned_trajectory(
    hidden_states: object,
    *,
    maximum_transitions: int,
    maximum_rank: int,
    maximum_width: int = 65_536,
    dtype: str = "<f4",
) -> CompiledTrajectory:
    """Compile ordered hidden-state deltas with deterministic salience and SVD signs."""
    states = np.asarray(hidden_states)
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] <= 0:
        raise ActivationCapsuleError("hidden states must have shape [steps>=2, width>0]")
    if states.shape[1] > maximum_width:
        raise ActivationCapsuleError("hidden-state width exceeds compiler bound")
    if dtype not in _ALLOWED_DTYPES:
        raise ActivationCapsuleError("unsupported compiler dtype")
    if isinstance(maximum_transitions, bool) or not isinstance(maximum_transitions, int) or maximum_transitions <= 0:
        raise ActivationCapsuleError("maximum_transitions must be positive")
    if isinstance(maximum_rank, bool) or not isinstance(maximum_rank, int) or maximum_rank <= 0:
        raise ActivationCapsuleError("maximum_rank must be positive")
    working = np.ascontiguousarray(states, dtype=np.float64)
    if not np.isfinite(working).all():
        raise ActivationCapsuleError("hidden states must be finite")
    all_deltas = np.diff(working, axis=0)
    scores = np.linalg.norm(all_deltas, axis=1)
    keep = min(maximum_transitions, len(all_deltas))
    ranked = sorted(range(len(all_deltas)), key=lambda index: (-float(scores[index]), index))[:keep]
    selected_zero_based = tuple(sorted(ranked))
    selected = all_deltas[list(selected_zero_based)]
    rank = min(maximum_rank, selected.shape[0], selected.shape[1])
    _u, _singular, vh = np.linalg.svd(selected, full_matrices=False)
    basis = vh[:rank].copy()
    coefficients = selected @ basis.T
    for component in range(rank):
        pivot = int(np.argmax(np.abs(basis[component])))
        if basis[component, pivot] < 0:
            basis[component] *= -1
            coefficients[:, component] *= -1
    reconstruction = coefficients @ basis
    denominator = float(np.linalg.norm(selected))
    error = float(np.linalg.norm(selected - reconstruction) / denominator) if denominator else 0.0
    output_dtype = np.dtype(dtype)
    return CompiledTrajectory(
        anchor=np.ascontiguousarray(working[0], dtype=output_dtype),
        basis=np.ascontiguousarray(basis, dtype=output_dtype),
        coefficients=np.ascontiguousarray(coefficients, dtype=output_dtype),
        transition_indices=tuple(index + 1 for index in selected_zero_based),
        reconstruction_relative_error=error,
    )


__all__ = [
    "ACTIVATION_CAPSULE_SCHEMA",
    "EXACT_STATE_SCHEMA",
    "TENSOR_GC_SCHEMA",
    "ActivationCapsuleBus",
    "ActivationCapsuleError",
    "ActivationCapsuleSpec",
    "ActivationExpectation",
    "ApproximationClass",
    "Authority",
    "AuthorityScope",
    "CapsuleProvenance",
    "CompiledTrajectory",
    "Geometry",
    "ModelBindings",
    "NormGateBounds",
    "PayloadKind",
    "PositionConvention",
    "StoredTensor",
    "TapBinding",
    "TensorGCReceipt",
    "TensorObjectInfo",
    "TensorPayloadStore",
    "compile_pruned_trajectory",
]
