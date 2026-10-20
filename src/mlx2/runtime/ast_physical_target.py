"""Model-neutral discovery contract for optional AST physical targets.

Discovery is deliberately weaker than execution admission. It receives only
bounded token-free geometry and immutable route identity, may describe an
adapter-owned candidate, and can never authorize a backend or open a state
transaction. The ordinary padded executor remains authoritative until a later
integration owns the live-state, output, and publication contracts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .flat_segmented_ast import NATIVE_PHYSICAL_LAYOUT, FlatSegmentDescriptor

DISCOVERY_SCHEMA = "mlx2.ast-physical-target-discovery.v1"
CANDIDATE_SCHEMA = "mlx2.ast-physical-target-candidate.v1"
SERVING_ADMISSION_SCHEMA = "mlx2.ast-physical-target-serving-admission.v1"
PROVIDER_PROTOCOL = "ast-physical-target-provider-v1"
_IDENTITY_FIELDS = {"source_revision", "build", "device", "artifact"}
_LIMIT_FIELDS = {"max_lanes", "max_segment_rows", "max_total_rows"}
_PLAN_FIELDS = {
    "backend_mode", "logical_rows", "physical_rows", "padding_rows", "max_query_len"
}
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_ARTIFACT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SAFE_TEXT = re.compile(r"[ -~]+\Z")
_SOURCE_PATH = re.compile(r"src/mlx2/[A-Za-z0-9_./-]+\Z")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _git_revision(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a git SHA")
    return value


def _text(value: object, label: str, *, maximum: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= maximum
        or _SAFE_TEXT.fullmatch(value) is None
    ):
        raise ValueError(f"{label} must be bounded printable ASCII")
    return value


def _protocol(value: object, label: str) -> str:
    value = _text(value, label, maximum=128)
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be an identifier")
    return value


def _identity(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_FIELDS:
        raise ValueError("physical-target identity fields differ")
    result = {
        "source_revision": _git_revision(
            value["source_revision"], "physical-target source revision"
        ),
        "build": _text(value["build"], "physical-target build"),
        "device": _text(value["device"], "physical-target device"),
        "artifact": _text(value["artifact"], "physical-target artifact"),
    }
    if result["artifact"].count("@sha256:") != 1:
        raise ValueError("physical-target artifact identity must be fingerprinted")
    name, fingerprint = result["artifact"].rsplit("@sha256:", 1)
    if not name or _ARTIFACT_NAME.fullmatch(name) is None:
        raise ValueError("physical-target artifact name differs")
    _digest(fingerprint, "physical-target artifact")
    return tuple(sorted(result.items()))


def _source_hashes(value: object) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        pairs = tuple(value.items())
    elif type(value) is tuple:
        pairs = value
    else:
        raise ValueError("physical-target source hashes are unavailable")
    if not 1 <= len(pairs) <= 64:
        raise ValueError("physical-target source hash count differs")
    result = []
    for item in pairs:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("physical-target source identity differs")
        path, digest = item
        if (
            not isinstance(path, str)
            or not 1 <= len(path) <= 256
            or _SOURCE_PATH.fullmatch(path) is None
            or "/../" in f"/{path}/"
        ):
            raise ValueError("physical-target source path differs")
        result.append((path, _digest(digest, "physical-target source")))
    if len({path for path, _digest_value in result}) != len(result):
        raise ValueError("physical-target source paths must be unique")
    return tuple(sorted(result))


def _hard_limits(value: object) -> tuple[tuple[str, int], ...]:
    if (
        not isinstance(value, Mapping)
        or set(value) != _LIMIT_FIELDS
        or any(
            type(value[name]) is not int or not 1 <= value[name] <= 65536
            for name in value
        )
    ):
        raise ValueError("physical-target hard limits differ")
    return tuple(sorted((name, value[name]) for name in _LIMIT_FIELDS))


def _fallback_plan(value: object, descriptor: FlatSegmentDescriptor):
    pairs = tuple(value)
    if (
        len(pairs) != len(_PLAN_FIELDS)
        or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            for item in pairs
        )
        or len({item[0] for item in pairs}) != len(pairs)
    ):
        raise ValueError("physical-target fallback plan differs")
    plan = dict(pairs)
    if set(plan) != _PLAN_FIELDS or plan["backend_mode"] != "padded":
        raise ValueError("physical-target fallback plan schema differs")
    integer_fields = _PLAN_FIELDS - {"backend_mode"}
    if any(type(plan[name]) is not int or plan[name] < 0 for name in integer_fields):
        raise ValueError("physical-target fallback plan values differ")
    if (
        plan["logical_rows"] != descriptor.logical_rows
        or plan["max_query_len"] != descriptor.max_segment_rows
        or plan["physical_rows"] < plan["logical_rows"]
        or plan["padding_rows"] != plan["physical_rows"] - plan["logical_rows"]
    ):
        raise ValueError("physical-target fallback plan geometry differs")
    return tuple(sorted(plan.items()))


def _reasons(value: object) -> tuple[str, ...]:
    reasons = tuple(value)
    if not 1 <= len(reasons) <= 8 or any(
        not isinstance(reason, str)
        or not 1 <= len(reason) <= 64
        or _IDENTIFIER.fullmatch(reason) is None
        for reason in reasons
    ):
        raise ValueError("physical-target decline reasons differ")
    return reasons


def _receipt(body: Mapping[str, object]) -> dict[str, object]:
    body = dict(body)
    return {**body, "manifest_sha256": _sha256(body)}


def ast_serving_admission_receipt(
    *,
    discovery: Mapping[str, object],
    session_scope_sha256: str,
    source_manifest_sha256: str,
    build_manifest_sha256: str,
    device_manifest_sha256: str,
    owner_capability_manifest_sha256: str,
    target_capability_manifest_sha256: str,
) -> dict[str, object]:
    """Mint one exact live-serving authority after non-authoritative discovery."""

    if (
        not isinstance(discovery, Mapping)
        or discovery.get("schema") != CANDIDATE_SCHEMA
        or discovery.get("candidate_implemented") is not True
        or discovery.get("implemented") is not False
        or discovery.get("eligible") is not False
    ):
        raise ValueError("physical-target discovery cannot seed live admission")
    return _receipt(
        {
            "schema": SERVING_ADMISSION_SCHEMA,
            "discovery_manifest_sha256": _digest(
                discovery.get("manifest_sha256"), "physical-target discovery"
            ),
            "provider_binding_sha256": _digest(
                discovery.get("provider_binding_sha256"),
                "physical-target provider binding",
            ),
            "geometry_sha256": _digest(
                discovery.get("geometry_sha256"), "physical-target geometry"
            ),
            "session_scope_sha256": _digest(
                session_scope_sha256, "physical-target session scope"
            ),
            "source_manifest_sha256": _digest(
                source_manifest_sha256, "physical-target source manifest"
            ),
            "build_manifest_sha256": _digest(
                build_manifest_sha256, "physical-target build manifest"
            ),
            "device_manifest_sha256": _digest(
                device_manifest_sha256, "physical-target device manifest"
            ),
            "owner_capability_manifest_sha256": _digest(
                owner_capability_manifest_sha256,
                "physical-target owner capability",
            ),
            "target_capability_manifest_sha256": _digest(
                target_capability_manifest_sha256,
                "physical-target target capability",
            ),
            "implemented": True,
            "eligible": True,
            "serving_wired": True,
            "qualified": False,
            "selected": False,
            "default": False,
            "observed_used": False,
            "authority": "adapter_live_admission",
            "apcv2_authoritative": False,
        }
    )


def validate_ast_serving_admission(
    admission: object,
    discovery: Mapping[str, object],
    *,
    expected_session_scope_sha256: str,
) -> dict[str, object]:
    """Validate the adapter's live authority before a scheduler may select it."""

    if not isinstance(admission, Mapping):
        raise TypeError("physical-target live admission is unavailable")
    receipt = dict(admission)
    manifest = receipt.pop("manifest_sha256", None)
    required_true = ("implemented", "eligible", "serving_wired")
    required_false = (
        "qualified",
        "selected",
        "default",
        "observed_used",
        "apcv2_authoritative",
    )
    if (
        receipt.get("schema") != SERVING_ADMISSION_SCHEMA
        or manifest != _sha256(receipt)
        or receipt.get("discovery_manifest_sha256")
        != discovery.get("manifest_sha256")
        or receipt.get("provider_binding_sha256")
        != discovery.get("provider_binding_sha256")
        or receipt.get("geometry_sha256") != discovery.get("geometry_sha256")
        or receipt.get("session_scope_sha256")
        != _digest(
            expected_session_scope_sha256,
            "physical-target expected session scope",
        )
        or receipt.get("authority") != "adapter_live_admission"
        or any(receipt.get(name) is not True for name in required_true)
        or any(receipt.get(name) is not False for name in required_false)
    ):
        raise ValueError("physical-target live admission differs")
    for name in (
        "session_scope_sha256",
        "source_manifest_sha256",
        "build_manifest_sha256",
        "device_manifest_sha256",
        "owner_capability_manifest_sha256",
        "target_capability_manifest_sha256",
    ):
        _digest(receipt.get(name), f"physical-target admission {name}")
    return dict(admission)


@dataclass(frozen=True, slots=True)
class ASTPhysicalTargetDiscovery:
    """Token-free facts available before padded inputs are materialized."""

    descriptor: FlatSegmentDescriptor
    capture_layers: tuple[int, ...]
    fidelity: str
    fallback_target_protocol: str
    artifact_fingerprint: str
    fallback_plan: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, FlatSegmentDescriptor):
            raise TypeError("physical-target discovery requires flat geometry")
        layers = tuple(self.capture_layers)
        if (
            not 1 <= len(layers) <= 64
            or any(type(layer) is not int or not 0 <= layer <= 65535 for layer in layers)
            or layers != tuple(sorted(set(layers)))
        ):
            raise ValueError("physical-target capture layers differ")
        object.__setattr__(
            self, "fidelity", _protocol(self.fidelity, "physical-target fidelity")
        )
        object.__setattr__(
            self,
            "fallback_target_protocol",
            _protocol(
                self.fallback_target_protocol, "physical-target fallback protocol"
            ),
        )
        object.__setattr__(
            self,
            "artifact_fingerprint",
            _digest(self.artifact_fingerprint, "physical-target request artifact"),
        )
        object.__setattr__(self, "capture_layers", layers)
        object.__setattr__(
            self, "fallback_plan", _fallback_plan(self.fallback_plan, self.descriptor)
        )

    @property
    def capture_layers_sha256(self) -> str:
        return _sha256(list(self.capture_layers))

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.receipt())

    def receipt(self) -> dict[str, object]:
        return {
            "schema": DISCOVERY_SCHEMA,
            "descriptor": self.descriptor.receipt(),
            "capture_layers": list(self.capture_layers),
            "capture_layers_sha256": self.capture_layers_sha256,
            "fidelity": self.fidelity,
            "fallback_target_protocol": self.fallback_target_protocol,
            "artifact_fingerprint": self.artifact_fingerprint,
            "fallback_plan": dict(self.fallback_plan),
        }


@dataclass(frozen=True, slots=True)
class _ProviderBinding:
    protocol: str
    source_revision_token: str
    source_identity: tuple[tuple[str, str], ...]
    identity: tuple[tuple[str, str], ...]

    @property
    def manifest_sha256(self) -> str:
        return _sha256(
            {
                "protocol": self.protocol,
                "source_revision_token": self.source_revision_token,
                "source_identity": dict(self.source_identity),
                "identity": dict(self.identity),
            }
        )


def _provider_binding(provider) -> _ProviderBinding:
    protocol = _protocol(provider.protocol, "physical-target provider protocol")
    if protocol != PROVIDER_PROTOCOL:
        raise ValueError("physical-target provider protocol differs")
    revision = _git_revision(
        provider.source_revision_token, "physical-target provider revision"
    )
    sources = _source_hashes(provider.source_identity)
    identity = _identity(provider.identity)
    if dict(identity)["source_revision"] != revision:
        raise ValueError("physical-target provider revision identity differs")
    return _ProviderBinding(protocol, revision, sources, identity)


@dataclass(frozen=True, slots=True)
class ASTPhysicalTargetCandidate:
    """Strict non-authoritative adapter candidate discovered for one layout."""

    geometry_sha256: str
    capture_layers_sha256: str
    fidelity: str
    fallback_target_protocol: str
    identity: tuple[tuple[str, str], ...]
    source_hashes: tuple[tuple[str, str], ...]
    hard_limits: tuple[tuple[str, int], ...]
    owner_protocol: str
    native_target_protocol: str
    owner_capability_manifest_sha256: str
    target_capability_manifest_sha256: str
    decline_reasons: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        geometry_sha256: str,
        capture_layers_sha256: str,
        fidelity: str,
        fallback_target_protocol: str,
        identity: Mapping[str, str],
        source_hashes,
        hard_limits: Mapping[str, int],
        owner_protocol: str,
        native_target_protocol: str,
        owner_capability_manifest_sha256: str,
        target_capability_manifest_sha256: str,
        decline_reasons,
    ) -> ASTPhysicalTargetCandidate:
        return cls(
            geometry_sha256=_digest(geometry_sha256, "physical-target geometry"),
            capture_layers_sha256=_digest(
                capture_layers_sha256, "physical-target capture layers"
            ),
            fidelity=_protocol(fidelity, "physical-target candidate fidelity"),
            fallback_target_protocol=_protocol(
                fallback_target_protocol, "physical-target fallback protocol"
            ),
            identity=_identity(identity),
            source_hashes=_source_hashes(source_hashes),
            hard_limits=_hard_limits(hard_limits),
            owner_protocol=_protocol(owner_protocol, "physical-target owner protocol"),
            native_target_protocol=_protocol(
                native_target_protocol, "physical-target native target protocol"
            ),
            owner_capability_manifest_sha256=_digest(
                owner_capability_manifest_sha256,
                "physical-target owner capability",
            ),
            target_capability_manifest_sha256=_digest(
                target_capability_manifest_sha256,
                "physical-target target capability",
            ),
            decline_reasons=_reasons(decline_reasons),
        )

    def validate_for(
        self, discovery: ASTPhysicalTargetDiscovery, binding: _ProviderBinding
    ) -> None:
        if self.geometry_sha256 != discovery.descriptor.geometry_sha256:
            raise ValueError("physical-target candidate geometry drifted")
        if self.capture_layers_sha256 != discovery.capture_layers_sha256:
            raise ValueError("physical-target candidate capture schema drifted")
        if self.fidelity != discovery.fidelity:
            raise ValueError("physical-target candidate fidelity differs")
        if self.fallback_target_protocol != discovery.fallback_target_protocol:
            raise ValueError("physical-target fallback protocol differs")
        if self.identity != binding.identity or self.source_hashes != binding.source_identity:
            raise ValueError("physical-target candidate provider binding differs")
        artifact = dict(self.identity)["artifact"].rsplit("@sha256:", 1)[1]
        if artifact != discovery.artifact_fingerprint:
            raise ValueError("physical-target candidate artifact differs")
        limits = dict(self.hard_limits)
        descriptor = discovery.descriptor
        if (
            descriptor.lane_count > limits["max_lanes"]
            or descriptor.max_segment_rows > limits["max_segment_rows"]
            or descriptor.logical_rows > limits["max_total_rows"]
        ):
            raise ValueError("physical-target candidate exceeds hard capacity")

    def receipt(
        self, discovery: ASTPhysicalTargetDiscovery, binding: _ProviderBinding
    ) -> dict[str, object]:
        self.validate_for(discovery, binding)
        capability_intersection_sha256 = _sha256(
            {
                "owner": self.owner_capability_manifest_sha256,
                "target": self.target_capability_manifest_sha256,
                "geometry": self.geometry_sha256,
                "capture_layers": self.capture_layers_sha256,
                "fidelity": self.fidelity,
            }
        )
        return _receipt(
            {
                "schema": CANDIDATE_SCHEMA,
                "provider_protocol": PROVIDER_PROTOCOL,
                "provider_binding_sha256": binding.manifest_sha256,
                "discovery_manifest_sha256": discovery.manifest_sha256,
                "geometry_sha256": self.geometry_sha256,
                "capture_layers_sha256": self.capture_layers_sha256,
                "fidelity": self.fidelity,
                "fallback_target_protocol": self.fallback_target_protocol,
                "native_target_protocol": self.native_target_protocol,
                "physical_layout": NATIVE_PHYSICAL_LAYOUT,
                "logical_rows": discovery.descriptor.logical_rows,
                "physical_rows": discovery.descriptor.logical_rows,
                "identity": dict(self.identity),
                "source_hashes": dict(self.source_hashes),
                "hard_limits": dict(self.hard_limits),
                "owner_protocol": self.owner_protocol,
                "owner_capability_manifest_sha256": self.owner_capability_manifest_sha256,
                "target_capability_manifest_sha256": self.target_capability_manifest_sha256,
                "capability_intersection_sha256": capability_intersection_sha256,
                "admission_manifest_sha256": None,
                "decline_reasons": list(self.decline_reasons),
                "candidate_implemented": True,
                "serving_implemented": False,
                "implemented": False,
                "qualified": False,
                "eligible": False,
                "selected": False,
                "default": False,
                "observed_used": False,
                "authority": "live_admission_required_not_discovery",
                "apcv2_authoritative": False,
            }
        )


@runtime_checkable
class ASTPhysicalTargetProvider(Protocol):
    """Adapter-owned provider with an immutable, source-bound identity."""

    protocol: str
    source_revision_token: str
    source_identity: tuple[tuple[str, str], ...]
    identity: Mapping[str, str]

    def validate_revision(self) -> None: ...

    def discover(
        self, request: ASTPhysicalTargetDiscovery
    ) -> ASTPhysicalTargetCandidate | None: ...


def _decline_receipt(
    request: ASTPhysicalTargetDiscovery | None,
    *,
    reason: str,
    binding: _ProviderBinding | None = None,
) -> dict[str, object]:
    reason = _reasons((reason,))[0]
    descriptor = None if request is None else request.descriptor
    return _receipt(
        {
            "schema": CANDIDATE_SCHEMA,
            "provider_protocol": PROVIDER_PROTOCOL,
            "provider_binding_sha256": None if binding is None else binding.manifest_sha256,
            "discovery_manifest_sha256": None if request is None else request.manifest_sha256,
            "geometry_sha256": None if descriptor is None else descriptor.geometry_sha256,
            "capture_layers_sha256": (
                None if request is None else request.capture_layers_sha256
            ),
            "fidelity": None if request is None else request.fidelity,
            "fallback_target_protocol": (
                None if request is None else request.fallback_target_protocol
            ),
            "native_target_protocol": None,
            "physical_layout": NATIVE_PHYSICAL_LAYOUT,
            "logical_rows": None if descriptor is None else descriptor.logical_rows,
            "physical_rows": None if descriptor is None else descriptor.logical_rows,
            "admission_manifest_sha256": None,
            "decline_reasons": [reason],
            "candidate_implemented": False,
            "serving_implemented": False,
            "implemented": False,
            "qualified": False,
            "eligible": False,
            "selected": False,
            "default": False,
            "observed_used": False,
            "authority": "live_admission_required_not_discovery",
            "apcv2_authoritative": False,
        }
    )


def discover_ast_physical_target(
    provider,
    request: ASTPhysicalTargetDiscovery,
    *,
    provider_getter,
) -> dict:
    """Return a fail-closed receipt; never authorize or invoke a target."""

    binding = None
    try:
        if not callable(provider_getter) or provider is None or provider is not provider_getter():
            raise ValueError("physical-target provider ownership drifted")
        if not isinstance(provider, ASTPhysicalTargetProvider):
            raise TypeError("physical-target provider protocol is incomplete")
        binding = _provider_binding(provider)
        provider.validate_revision()
        candidate = provider.discover(request)
        provider.validate_revision()
        if provider is not provider_getter() or _provider_binding(provider) != binding:
            raise ValueError("physical-target provider binding drifted")
        if candidate is None:
            return _decline_receipt(request, reason="provider_declined", binding=binding)
        if not isinstance(candidate, ASTPhysicalTargetCandidate):
            raise TypeError("physical-target provider returned an invalid candidate")
        return candidate.receipt(request, binding)
    except Exception:  # noqa: BLE001 - diagnostics may never break padded serving
        return _decline_receipt(
            request, reason="provider_validation_failed", binding=binding
        )


def discover_ast_physical_target_from_layout(
    provider,
    *,
    provider_getter,
    query_lengths,
    capture_layers,
    fidelity,
    fallback_target_protocol,
    artifact_fingerprint,
    fallback_plan,
) -> dict:
    """Fail-soft construction plus discovery for the serving hot seam."""

    try:
        descriptor = FlatSegmentDescriptor.from_query_lengths(query_lengths)
        request = ASTPhysicalTargetDiscovery(
            descriptor=descriptor,
            capture_layers=tuple(capture_layers),
            fidelity=fidelity,
            fallback_target_protocol=fallback_target_protocol,
            artifact_fingerprint=artifact_fingerprint,
            fallback_plan=tuple(fallback_plan),
        )
    except Exception:  # noqa: BLE001 - diagnostics may never break padded serving
        return _decline_receipt(None, reason="discovery_request_invalid")
    return discover_ast_physical_target(
        provider, request, provider_getter=provider_getter
    )
