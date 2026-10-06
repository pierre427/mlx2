"""Request-local control plane for semantic activation-capsule recall.

The public recall tool returns content-addressed handles and bounded source
metadata, never tensors. This module resolves those handles through the
existing semantic :class:`CapsuleStore`, authorizes them against one frozen
:class:`HyperDirectory` view, and mounts immutable references to tensor
payloads owned by ``runtime.activation_capsules``.

No second activation-envelope schema is introduced here. Tensor loading and
model-specific math remain with the activation-capsule store and adapter. The
model cannot select an injection layer or gain through this protocol.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .activation_capsules import (
    ActivationCapsuleError,
    ActivationCapsuleSpec,
    PayloadKind,
)
from .hyper_directory import DirectoryContext, HyperDirectory
from .semantic_capsules import (
    DIGEST_PATTERN,
    CapsuleIntegrityError,
    CapsuleStore,
    canonical_json,
)

TOOL_RESULT_SCHEMA = "mlx2-semantic-capsule-tool-result-v1"
ACTIVATION_CAPSULE_SCHEMA = "mlx2-activation-capsule-v1"
SNAPSHOT_SCHEMA = "mlx2-semantic-capsule-mount-v1"
RECEIPT_SCHEMA = "mlx2-semantic-capsule-bus-receipt-v1"
SOURCE_KINDS = frozenset({"document", "semantic_memory", "pruned_state", "agent"})
PROMPT_INJECTION_LABELS = frozenset({"none", "suspected", "confirmed", "unknown"})
CAPSULE_SCOPES = frozenset({"request", "session", "project", "document"})
PAYLOAD_KINDS = frozenset(
    {
        "ordered_trajectory",
        "directional_residual",
        "continuous_prefix",
        "cross_attention_bank",
    }
)
_TRUSTED_ACTIVATION_CAPABILITY = object()


class CapsuleBusError(RuntimeError):
    """Fail-closed capsule-bus refusal with a stable route-receipt code."""

    def __init__(self, code: str, message: str, *, requested_count: int = 0):
        super().__init__(message)
        self.code = code
        self.requested_count = requested_count

    def receipt(self, *, request_id: str | None = None) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "status": "rejected",
            "reason": self.code,
            "request_id": request_id,
            "requested_count": self.requested_count,
            "selected": False,
            "engaged": False,
            "observed_used": False,
        }


class CapsuleBusCancelled(CapsuleBusError):
    def __init__(self, *, requested_count: int = 0):
        super().__init__(
            "cancelled",
            "semantic capsule mount was cancelled",
            requested_count=requested_count,
        )


@dataclass(frozen=True, slots=True)
class CapsuleBusConfig:
    """Operator-owned policy; disabled and ordinary-only by default."""

    enabled: bool = False
    ordinary_only: bool = True
    max_capsules: int = 8
    max_bytes: int = 8 << 20
    allowed_scopes: tuple[str, ...] = ("session", "request")
    allowed_payload_kinds: tuple[str, ...] = tuple(sorted(PAYLOAD_KINDS))
    allow_prompt_injection_sources: bool = False

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool or type(self.ordinary_only) is not bool:
            raise TypeError("capsule bus enablement flags must be booleans")
        if type(self.max_capsules) is not int or not 1 <= self.max_capsules <= 64:
            raise ValueError("max_capsules must be between 1 and 64")
        if type(self.max_bytes) is not int or not 1 <= self.max_bytes <= 512 << 20:
            raise ValueError("max_bytes must be between 1 and 536870912")
        scopes = tuple(self.allowed_scopes)
        kinds = tuple(self.allowed_payload_kinds)
        if (
            not scopes
            or len(scopes) != len(set(scopes))
            or not set(scopes) <= CAPSULE_SCOPES
        ):
            raise ValueError("allowed capsule scopes are invalid")
        if (
            not kinds
            or len(kinds) != len(set(kinds))
            or not set(kinds) <= PAYLOAD_KINDS
        ):
            raise ValueError("allowed activation payload kinds are invalid")
        object.__setattr__(self, "allowed_scopes", scopes)
        object.__setattr__(self, "allowed_payload_kinds", kinds)


@dataclass(frozen=True, slots=True)
class CapsuleRuntimeBindings:
    """Operator/adapter-selected target contract, never model-selected."""

    model: str
    tokenizer: str
    runtime: str
    projector: str
    target_layer: int
    target_injection: str

    def __post_init__(self) -> None:
        for name in ("model", "tokenizer", "runtime", "projector", "target_injection"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 1024:
                raise ValueError(
                    f"capsule {name} binding must be bounded nonempty text"
                )
        if (
            isinstance(self.target_layer, bool)
            or not isinstance(self.target_layer, int)
            or self.target_layer < 0
        ):
            raise ValueError("capsule target layer must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class CapsuleRequestContext:
    directory: DirectoryContext
    request_id: str
    route: str = "ordinary"
    project_id: str | None = None
    document_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_id, str)
            or not self.request_id
            or len(self.request_id) > 128
        ):
            raise ValueError("capsule request_id must be bounded nonempty text")
        if self.route not in {"ordinary", "speculative", "draft", "batch"}:
            raise ValueError("unknown capsule route kind")
        for name in ("project_id", "document_id"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value or len(value) > 2048
            ):
                raise ValueError(
                    f"capsule {name} must be bounded nonempty text or null"
                )
        if (
            self.directory.model is None
            or self.directory.tenant is None
            or self.directory.session is None
        ):
            raise ValueError(
                "capsule requests require model, tenant and session directory scope"
            )


@dataclass(frozen=True, slots=True)
class CapsuleSource:
    source_id: str
    kind: str
    label: str
    uri: str | None
    provenance_digest: str
    prompt_injection: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "label": self.label,
            "uri": self.uri,
            "provenance_digest": self.provenance_digest,
            "prompt_injection": self.prompt_injection,
        }


@dataclass(frozen=True, slots=True)
class ToolCapsuleHandle:
    name: str
    digest: str
    scope: str
    scope_id: str
    tenant: str
    expires_at_unix_ns: int
    source: CapsuleSource


@dataclass(frozen=True, slots=True)
class MountedCapsule:
    """Validated immutable activation manifest and tensor references."""

    name: str
    digest: str
    scope: str
    scope_id: str
    byte_count: int
    source: CapsuleSource
    payload_kind: str
    manifest: Mapping[str, Any]
    payload_references: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CapsuleMount:
    request_id: str
    directory_revision: int
    directory_fingerprint: str
    snapshot_fingerprint: str
    semantic_fingerprint: str
    bindings: CapsuleRuntimeBindings
    capsules: tuple[MountedCapsule, ...]
    total_bytes: int

    @property
    def digests(self) -> tuple[str, ...]:
        return tuple(item.digest for item in self.capsules)


@dataclass(frozen=True, slots=True)
class CapsuleBusReceipt:
    request_id: str
    snapshot_fingerprint: str
    semantic_fingerprint: str
    selected_handles: tuple[str, ...]
    engaged_handles: tuple[str, ...] = ()
    observed_used_handles: tuple[str, ...] = ()
    prompt_injection_sources: tuple[str, ...] = ()

    @property
    def selected(self) -> bool:
        return bool(self.selected_handles)

    @property
    def engaged(self) -> bool:
        return bool(self.engaged_handles)

    @property
    def observed_used(self) -> bool:
        return bool(self.observed_used_handles)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "status": "mounted",
            "request_id": self.request_id,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "semantic_fingerprint": self.semantic_fingerprint,
            "selected": self.selected,
            "selected_handles": list(self.selected_handles),
            "engaged": self.engaged,
            "engaged_handles": list(self.engaged_handles),
            "observed_used": self.observed_used,
            "observed_used_handles": list(self.observed_used_handles),
            "prompt_injection_sources": list(self.prompt_injection_sources),
        }


@dataclass(frozen=True, slots=True)
class TrustedActivationRequest:
    """Process-local capability wrapping one core-verified activation payload."""

    capsule_digest: str
    manifest: Mapping[str, Any]
    tensors: Mapping[str, Any]
    gate: float
    semantic_fingerprint: str
    _capability: object = field(repr=False, compare=False)


def is_trusted_activation_request(value: object) -> bool:
    """Whether ``value`` was issued by this process's capsule bus."""
    return (
        isinstance(value, TrustedActivationRequest)
        and value._capability is _TRUSTED_ACTIVATION_CAPABILITY
    )


def _bounded_text(value: object, name: str, *, maximum: int = 2048) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise CapsuleBusError(
            "invalid_tool_result", f"{name} must be bounded nonempty text"
        )
    return value


def _source(value: object) -> CapsuleSource:
    fields = {
        "source_id",
        "kind",
        "label",
        "uri",
        "provenance_digest",
        "prompt_injection",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CapsuleBusError(
            "invalid_tool_result", "capsule source metadata has an invalid shape"
        )
    if value["kind"] not in SOURCE_KINDS:
        raise CapsuleBusError("invalid_tool_result", "unsupported capsule source kind")
    if value["prompt_injection"] not in PROMPT_INJECTION_LABELS:
        raise CapsuleBusError(
            "invalid_tool_result", "invalid prompt-injection provenance label"
        )
    uri = value["uri"]
    if uri is not None and (not isinstance(uri, str) or len(uri) > 4096):
        raise CapsuleBusError(
            "invalid_tool_result", "capsule source uri must be bounded text or null"
        )
    provenance = value["provenance_digest"]
    if not isinstance(provenance, str) or DIGEST_PATTERN.fullmatch(provenance) is None:
        raise CapsuleBusError(
            "invalid_tool_result", "capsule provenance must be a SHA-256 digest"
        )
    return CapsuleSource(
        source_id=_bounded_text(value["source_id"], "source_id"),
        kind=value["kind"],
        label=_bounded_text(value["label"], "source label"),
        uri=uri,
        provenance_digest=provenance,
        prompt_injection=value["prompt_injection"],
    )


def _tool_handle(value: object) -> ToolCapsuleHandle:
    fields = {
        "name",
        "digest",
        "scope",
        "scope_id",
        "tenant",
        "expires_at_unix_ns",
        "source",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        # Exact shape forbids tensor arrays, injection_layer, or gain in the
        # model/tool-controlled plane.
        raise CapsuleBusError(
            "invalid_tool_result", "capsule handle has an invalid shape"
        )
    digest = value["digest"]
    if not isinstance(digest, str) or DIGEST_PATTERN.fullmatch(digest) is None:
        raise CapsuleBusError(
            "invalid_tool_result", "capsule handle must be lowercase SHA-256"
        )
    if value["scope"] not in CAPSULE_SCOPES:
        raise CapsuleBusError("invalid_tool_result", "invalid capsule scope")
    expiry = value["expires_at_unix_ns"]
    if isinstance(expiry, bool) or not isinstance(expiry, int) or expiry <= 0:
        raise CapsuleBusError(
            "invalid_tool_result", "capsule expiry must be positive unix nanoseconds"
        )
    return ToolCapsuleHandle(
        name=_bounded_text(value["name"], "capsule name", maximum=128),
        digest=digest,
        scope=value["scope"],
        scope_id=_bounded_text(value["scope_id"], "capsule scope_id"),
        tenant=_bounded_text(value["tenant"], "capsule tenant", maximum=128),
        expires_at_unix_ns=expiry,
        source=_source(value["source"]),
    )


def parse_tool_result(value: object) -> tuple[int, str, tuple[ToolCapsuleHandle, ...]]:
    """Parse the metadata-only tool-to-runtime protocol."""
    fields = {"schema", "directory_revision", "directory_fingerprint", "handles"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CapsuleBusError(
            "invalid_tool_result", "capsule tool result has an invalid shape"
        )
    if value["schema"] != TOOL_RESULT_SCHEMA:
        raise CapsuleBusError(
            "invalid_tool_result", "unsupported capsule tool result schema"
        )
    revision = value["directory_revision"]
    fingerprint = value["directory_fingerprint"]
    handles = value["handles"]
    if type(revision) is not int or revision < 0:
        raise CapsuleBusError(
            "invalid_tool_result", "directory revision must be nonnegative"
        )
    if (
        not isinstance(fingerprint, str)
        or DIGEST_PATTERN.fullmatch(fingerprint) is None
    ):
        raise CapsuleBusError(
            "invalid_tool_result", "directory fingerprint must be SHA-256"
        )
    if not isinstance(handles, Sequence) or isinstance(
        handles, (str, bytes, bytearray)
    ):
        raise CapsuleBusError("invalid_tool_result", "capsule handles must be a list")
    parsed = tuple(_tool_handle(item) for item in handles)
    if len({item.digest for item in parsed}) != len(parsed):
        raise CapsuleBusError("invalid_tool_result", "capsule handles must be unique")
    return revision, fingerprint, parsed


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _cancelled(cancel: Callable[[], bool] | None) -> bool:
    return bool(cancel is not None and cancel())


def _activation_reference_bytes(
    spec: ActivationCapsuleSpec, payloads: Mapping[str, Any]
) -> int:
    expected_names = {
        PayloadKind.ORDERED_TRAJECTORY: {"anchor", "basis", "coefficients"},
        PayloadKind.DIRECTIONAL_RESIDUAL: {"residual"},
        PayloadKind.CONTINUOUS_PREFIX: {"prefix"},
        PayloadKind.CROSS_ATTENTION_BANK: {"keys", "values"},
    }[spec.payload_kind]
    if set(payloads) != expected_names:
        raise CapsuleBusError(
            "invalid_capsule", "activation tensor names do not match the payload kind"
        )
    shapes: dict[str, tuple[int, ...]] = {}
    total = 0
    itemsize = {"<f2": 2, "<f4": 4, "<f8": 8}[spec.dtype]
    for name, reference in payloads.items():
        if (
            not isinstance(name, str)
            or not isinstance(reference, Mapping)
            or set(reference) != {"digest", "dtype", "shape", "nbytes"}
        ):
            raise CapsuleBusError(
                "invalid_capsule", "activation tensor reference has an invalid shape"
            )
        digest = reference["digest"]
        dtype = reference["dtype"]
        shape_value = reference["shape"]
        nbytes = reference["nbytes"]
        if (
            not isinstance(digest, str)
            or DIGEST_PATTERN.fullmatch(digest) is None
            or dtype != spec.dtype
            or not isinstance(shape_value, list)
            or not 1 <= len(shape_value) <= 4
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in shape_value
            )
            or isinstance(nbytes, bool)
            or not isinstance(nbytes, int)
            or nbytes != math.prod(shape_value) * itemsize
        ):
            raise CapsuleBusError(
                "invalid_capsule", "activation tensor reference geometry is invalid"
            )
        shapes[name] = tuple(shape_value)
        total += nbytes

    width = spec.geometry.hidden_width
    length = spec.geometry.sequence_length
    if spec.payload_kind is PayloadKind.ORDERED_TRAJECTORY:
        count = len(spec.ordered_transition_indices)
        basis = shapes["basis"]
        rank = basis[0] if len(basis) == 2 else -1
        valid = (
            shapes["anchor"] == (width,)
            and basis == (rank, width)
            and shapes["coefficients"] == (count, rank)
            and 0 < rank <= min(count, width)
            and length > max(spec.ordered_transition_indices)
        )
    elif spec.payload_kind is PayloadKind.DIRECTIONAL_RESIDUAL:
        valid = shapes["residual"] == (width,)
    elif spec.payload_kind is PayloadKind.CONTINUOUS_PREFIX:
        valid = shapes["prefix"] == (length, width)
    else:
        valid = (
            spec.geometry.key_width is not None
            and spec.geometry.value_width is not None
            and shapes["keys"] == (length, spec.geometry.key_width)
            and shapes["values"] == (length, spec.geometry.value_width)
        )
    if not valid:
        raise CapsuleBusError(
            "invalid_capsule",
            "activation tensor references violate the payload contract",
        )
    return total


def compose_semantic_fingerprint(
    existing: object | None, snapshot_fingerprint: str
) -> str:
    """Bind a capsule snapshot into APCv2's existing semantic namespace."""
    if (
        not isinstance(snapshot_fingerprint, str)
        or DIGEST_PATTERN.fullmatch(snapshot_fingerprint) is None
    ):
        raise ValueError("capsule snapshot fingerprint must be lowercase SHA-256")
    payload = {
        "schema": "mlx2-apcv2-semantic-capsule-bus-v1",
        "existing": existing,
        "capsule_snapshot": snapshot_fingerprint,
    }
    try:
        return hashlib.sha256(canonical_json(payload)).hexdigest()
    except (TypeError, ValueError) as error:
        raise ValueError(
            "existing APC semantic fingerprint is not canonical JSON"
        ) from error


class SemanticCapsuleBus:
    """Resolve, authorize and freeze one request's capsule selection."""

    def __init__(
        self,
        capsules: CapsuleStore,
        directory: HyperDirectory,
        bindings: CapsuleRuntimeBindings,
        *,
        config: CapsuleBusConfig | None = None,
        now_unix_ns: Callable[[], int],
    ):
        if directory.capsules is not capsules:
            raise ValueError("capsule bus directory and store must share ownership")
        self.capsules = capsules
        self.directory = directory
        self.bindings = bindings
        self.config = config or CapsuleBusConfig()
        self.now_unix_ns = now_unix_ns

    def mount(
        self,
        tool_result: object,
        context: CapsuleRequestContext,
        *,
        existing_semantic_fingerprint: object | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[CapsuleMount, CapsuleBusReceipt]:
        if not self.config.enabled:
            raise CapsuleBusError(
                "capability_disabled", "semantic capsule bus is disabled"
            )
        if self.config.ordinary_only and context.route != "ordinary":
            raise CapsuleBusError(
                "ordinary_only", "semantic capsule bus supports ordinary inference only"
            )
        if _cancelled(cancelled):
            raise CapsuleBusCancelled()
        revision, directory_fingerprint, handles = parse_tool_result(tool_result)
        count = len(handles)
        if not handles:
            raise CapsuleBusError("empty_selection", "capsule tool selected no handles")
        if count > self.config.max_capsules:
            raise CapsuleBusError(
                "count_budget_exceeded",
                "semantic capsule count exceeds its request budget",
                requested_count=count,
            )
        if _cancelled(cancelled):
            raise CapsuleBusCancelled(requested_count=count)

        with self.directory.transaction():
            resolved = self.directory.resolve(context.directory)
            if (
                resolved.revision != revision
                or resolved.fingerprint != directory_fingerprint
            ):
                raise CapsuleBusError(
                    "stale_directory_revision",
                    "capsule selection does not match the current directory snapshot",
                    requested_count=count,
                )
            mounted: list[MountedCapsule] = []
            total_bytes = 0
            for handle in handles:
                if _cancelled(cancelled):
                    raise CapsuleBusCancelled(requested_count=count)
                self._authorize_handle(handle, context)
                if resolved.handles.get(handle.name) != handle.digest:
                    raise CapsuleBusError(
                        "unauthorized_handle",
                        "capsule handle is not published at this directory scope",
                        requested_count=count,
                    )
                try:
                    capsule = self.capsules.get(handle.digest)
                except FileNotFoundError as error:
                    raise CapsuleBusError(
                        "missing_handle",
                        "semantic capsule handle is missing",
                        requested_count=count,
                    ) from error
                except CapsuleIntegrityError as error:
                    raise CapsuleBusError(
                        "corrupt_handle",
                        "semantic capsule handle is corrupt",
                        requested_count=count,
                    ) from error
                item = self._validate_activation_capsule(capsule, handle)
                total_bytes += item.byte_count
                if total_bytes > self.config.max_bytes:
                    raise CapsuleBusError(
                        "byte_budget_exceeded",
                        "semantic capsule bytes exceed the request budget",
                        requested_count=count,
                    )
                mounted.append(item)

        if _cancelled(cancelled):
            raise CapsuleBusCancelled(requested_count=count)
        snapshot_payload = {
            "schema": SNAPSHOT_SCHEMA,
            "request_id": context.request_id,
            "directory_revision": revision,
            "directory_fingerprint": directory_fingerprint,
            "bindings": {
                "model": self.bindings.model,
                "tokenizer": self.bindings.tokenizer,
                "runtime": self.bindings.runtime,
                "projector": self.bindings.projector,
                "target_layer": self.bindings.target_layer,
                "target_injection": self.bindings.target_injection,
            },
            # Retrieval order is semantic state: a trajectory permutation is
            # not the same conditioner.
            "handles": [item.digest for item in mounted],
            "sources": [item.source.as_dict() for item in mounted],
            "total_bytes": total_bytes,
        }
        snapshot_fingerprint = hashlib.sha256(
            canonical_json(snapshot_payload)
        ).hexdigest()
        semantic_fingerprint = compose_semantic_fingerprint(
            existing_semantic_fingerprint, snapshot_fingerprint
        )
        mount = CapsuleMount(
            request_id=context.request_id,
            directory_revision=revision,
            directory_fingerprint=directory_fingerprint,
            snapshot_fingerprint=snapshot_fingerprint,
            semantic_fingerprint=semantic_fingerprint,
            bindings=self.bindings,
            capsules=tuple(mounted),
            total_bytes=total_bytes,
        )
        receipt = CapsuleBusReceipt(
            request_id=context.request_id,
            snapshot_fingerprint=snapshot_fingerprint,
            semantic_fingerprint=semantic_fingerprint,
            selected_handles=mount.digests,
            prompt_injection_sources=tuple(
                item.source.source_id
                for item in mounted
                if item.source.prompt_injection != "none"
            ),
        )
        return mount, receipt

    def _authorize_handle(
        self, handle: ToolCapsuleHandle, context: CapsuleRequestContext
    ) -> None:
        if handle.scope not in self.config.allowed_scopes:
            raise CapsuleBusError(
                "scope_denied", "capsule scope is not enabled by operator policy"
            )
        if handle.tenant != context.directory.tenant:
            raise CapsuleBusError(
                "wrong_tenant",
                "capsule tenant does not match the authenticated request",
            )
        expected_scope_id = {
            "session": context.directory.session,
            "request": context.request_id,
            "project": context.project_id,
            "document": context.document_id,
        }.get(handle.scope)
        if expected_scope_id is None:
            raise CapsuleBusError(
                "missing_scope_identity",
                f"request context has no authenticated {handle.scope} identity",
            )
        if handle.scope_id != expected_scope_id:
            code = {
                "session": "wrong_session",
                "request": "wrong_request",
                "project": "wrong_project",
                "document": "wrong_document",
            }[handle.scope]
            raise CapsuleBusError(
                code, "capsule authority scope does not match the request"
            )
        if handle.expires_at_unix_ns <= int(self.now_unix_ns()):
            raise CapsuleBusError(
                "expired_handle", "semantic capsule handle has expired"
            )
        if (
            handle.source.prompt_injection != "none"
            and not self.config.allow_prompt_injection_sources
        ):
            raise CapsuleBusError(
                "prompt_injection_source_denied",
                "capsule source carries prompt-injection provenance",
            )

    def _validate_activation_capsule(
        self, capsule: Mapping[str, Any], handle: ToolCapsuleHandle
    ) -> MountedCapsule:
        if capsule.get("kind") != "activation_capsule":
            raise CapsuleBusError(
                "invalid_capsule", "handle is not an activation capsule"
            )
        bindings = capsule.get("bindings")
        if not isinstance(bindings, Mapping) or dict(bindings) != {
            "model": self.bindings.model,
            "tokenizer": self.bindings.tokenizer,
            "runtime": self.bindings.runtime,
        }:
            raise CapsuleBusError(
                "binding_mismatch", "capsule model/tokenizer/runtime binding mismatch"
            )
        data = capsule.get("data")
        manifest = data.get("activation") if isinstance(data, Mapping) else None
        payloads = data.get("payloads") if isinstance(data, Mapping) else None
        if (
            not isinstance(data, Mapping)
            or set(data) != {"activation", "payloads"}
            or not isinstance(manifest, Mapping)
            or not isinstance(payloads, Mapping)
        ):
            raise CapsuleBusError(
                "invalid_capsule", "activation capsule data is malformed"
            )
        try:
            spec = ActivationCapsuleSpec.from_dict(manifest)
        except ActivationCapsuleError as error:
            raise CapsuleBusError(
                "invalid_capsule",
                "activation capsule manifest failed strict validation",
            ) from error
        declared = manifest.get("bindings")
        tap = manifest.get("tap")
        authority = manifest.get("authority")
        provenance = manifest.get("provenance")
        metadata = manifest.get("metadata")
        if not all(
            isinstance(item, Mapping)
            for item in (declared, tap, authority, provenance, metadata)
        ):
            raise CapsuleBusError(
                "invalid_capsule", "activation capsule manifest is incomplete"
            )
        expected_declared = {
            "source_model": declared.get("source_model"),
            "target_model": self.bindings.model,
            "tokenizer": self.bindings.tokenizer,
            "runtime": self.bindings.runtime,
            "projector_revision": self.bindings.projector,
        }
        if dict(declared) != expected_declared:
            raise CapsuleBusError(
                "binding_mismatch", "activation projector binding mismatch"
            )
        if (
            tap.get("target_layer") != self.bindings.target_layer
            or tap.get("target_injection") != self.bindings.target_injection
        ):
            raise CapsuleBusError(
                "binding_mismatch", "activation target seam binding mismatch"
            )
        if dict(authority) != {
            "tenant": handle.tenant,
            "scope": handle.scope,
            "scope_id": handle.scope_id,
            "expires_at_unix_ns": handle.expires_at_unix_ns,
        }:
            raise CapsuleBusError(
                "metadata_mismatch", "tool authority does not match the stored capsule"
            )
        if handle.expires_at_unix_ns <= int(self.now_unix_ns()):
            raise CapsuleBusError(
                "expired_handle", "semantic capsule expired while mounting"
            )
        payload_kind = spec.payload_kind.value
        if payload_kind not in self.config.allowed_payload_kinds:
            raise CapsuleBusError(
                "payload_kind_denied", "activation payload kind is not enabled"
            )
        expected_source = {
            "source_id": metadata.get("source_id"),
            "kind": metadata.get("source_kind"),
            "label": metadata.get("source_label"),
            "uri": metadata.get("source_uri"),
            "prompt_injection": metadata.get("prompt_injection"),
        }
        actual_source = handle.source.as_dict()
        if any(
            expected_source[name] != actual_source[name] for name in expected_source
        ):
            raise CapsuleBusError(
                "metadata_mismatch", "tool source does not match capsule provenance"
            )
        source_digests = provenance.get("source_digests")
        source_references = provenance.get("source_references")
        if (
            not isinstance(source_digests, list)
            or handle.source.provenance_digest not in source_digests
            or not isinstance(source_references, list)
            or (
                handle.source.uri is not None
                and handle.source.uri not in source_references
            )
        ):
            raise CapsuleBusError(
                "metadata_mismatch", "tool source lacks capsule provenance evidence"
            )
        referenced_bytes = _activation_reference_bytes(spec, payloads)
        byte_count = len(canonical_json(capsule)) + referenced_bytes
        return MountedCapsule(
            name=handle.name,
            digest=handle.digest,
            scope=handle.scope,
            scope_id=handle.scope_id,
            byte_count=byte_count,
            source=handle.source,
            payload_kind=payload_kind,
            manifest=_freeze(manifest),
            payload_references=_freeze(payloads),
        )

    @staticmethod
    def engagement_receipt(
        mount: CapsuleMount,
        *,
        engaged_handles: Sequence[str],
        observed_used_handles: Sequence[str] = (),
    ) -> CapsuleBusReceipt:
        selected = set(mount.digests)
        engaged = tuple(engaged_handles)
        observed = tuple(observed_used_handles)
        if len(set(engaged)) != len(engaged) or len(set(observed)) != len(observed):
            raise ValueError("capsule engagement handles must be unique")
        if not set(engaged) <= selected:
            raise ValueError("engaged capsule was not in the frozen selection")
        if not set(observed) <= set(engaged):
            raise ValueError("observed-used capsule was not engaged")
        return CapsuleBusReceipt(
            request_id=mount.request_id,
            snapshot_fingerprint=mount.snapshot_fingerprint,
            semantic_fingerprint=mount.semantic_fingerprint,
            selected_handles=mount.digests,
            engaged_handles=engaged,
            observed_used_handles=observed,
            prompt_injection_sources=tuple(
                item.source.source_id
                for item in mount.capsules
                if item.source.prompt_injection != "none"
            ),
        )

    @staticmethod
    def bind_request(
        request: Mapping[str, Any],
        mount: CapsuleMount,
        *,
        capsule_digest: str,
        manifest: Mapping[str, Any],
        tensors: Mapping[str, Any],
        gate: float,
    ) -> dict[str, Any]:
        """Attach one core-loaded, runtime-private activation payload.

        The semantic tool never returns these arrays. The inference stack
        first mounts an authorized handle, then the core activation store
        verifies and loads its tensor references, and only then calls this
        method. The gate is runtime-owned and must remain inside the stored
        capsule bounds.
        """
        if capsule_digest not in mount.digests:
            raise CapsuleBusError(
                "unauthorized_handle",
                "activation payload was not in the frozen selection",
            )
        selected = next(
            item for item in mount.capsules if item.digest == capsule_digest
        )
        if not isinstance(manifest, Mapping) or _thaw(selected.manifest) != _thaw(
            manifest
        ):
            raise CapsuleBusError(
                "metadata_mismatch",
                "loaded activation manifest changed after selection",
            )
        if not isinstance(tensors, Mapping) or set(tensors) != set(
            selected.payload_references
        ):
            raise CapsuleBusError(
                "metadata_mismatch",
                "loaded activation tensors do not match selected references",
            )
        for name, tensor in tensors.items():
            reference = selected.payload_references[name]
            dtype = getattr(getattr(tensor, "dtype", None), "str", None)
            shape = tuple(getattr(tensor, "shape", ()))
            nbytes = getattr(tensor, "nbytes", None)
            if (
                dtype != reference.get("dtype")
                or shape != tuple(reference.get("shape", ()))
                or nbytes != reference.get("nbytes")
            ):
                raise CapsuleBusError(
                    "metadata_mismatch",
                    "loaded activation tensor geometry changed after selection",
                )
        if (
            isinstance(gate, bool)
            or not isinstance(gate, (int, float))
            or not math.isfinite(gate)
        ):
            raise CapsuleBusError(
                "invalid_gate", "activation capsule gate must be finite"
            )
        bounds = selected.manifest.get("bounds")
        minimum = bounds.get("minimum_gate") if isinstance(bounds, Mapping) else None
        maximum = bounds.get("maximum_gate") if isinstance(bounds, Mapping) else None
        if (
            not isinstance(minimum, (int, float))
            or not isinstance(maximum, (int, float))
            or (
                float(gate) != 0.0
                and not float(minimum) <= float(gate) <= float(maximum)
            )
        ):
            raise CapsuleBusError(
                "invalid_gate", "activation capsule gate exceeds stored bounds"
            )
        if "_mlx2_activation_capsule" in request:
            raise CapsuleBusError(
                "duplicate_activation_payload",
                "request already contains an activation capsule payload",
            )
        prepared = dict(request)
        prepared["_mlx2_activation_capsule"] = TrustedActivationRequest(
            capsule_digest=capsule_digest,
            manifest=manifest,
            tensors=tensors,
            gate=float(gate),
            semantic_fingerprint=mount.semantic_fingerprint,
            _capability=_TRUSTED_ACTIVATION_CAPABILITY,
        )
        prepared["_mlx2_semantic_fingerprint"] = mount.semantic_fingerprint
        return prepared


__all__ = [
    "ACTIVATION_CAPSULE_SCHEMA",
    "CAPSULE_SCOPES",
    "PAYLOAD_KINDS",
    "RECEIPT_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "TOOL_RESULT_SCHEMA",
    "CapsuleBusCancelled",
    "CapsuleBusConfig",
    "CapsuleBusError",
    "CapsuleBusReceipt",
    "CapsuleMount",
    "CapsuleRequestContext",
    "CapsuleRuntimeBindings",
    "CapsuleSource",
    "MountedCapsule",
    "SemanticCapsuleBus",
    "ToolCapsuleHandle",
    "TrustedActivationRequest",
    "compose_semantic_fingerprint",
    "is_trusted_activation_request",
    "parse_tool_result",
]
