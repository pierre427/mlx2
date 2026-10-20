"""Offline numerical authority for optional AST physical targets.

The serving path must not discover numerical equivalence by executing a shadow
model.  A direct probe may instead mint this immutable, source/build/device/
artifact/profile-bound capability.  Serving only compares bounded scalar and
digest fields plus the requested layout against the qualified hard limits.

This module is model-neutral.  It does not select a route, open live state,
cache tensor results, or make an APCv2 hit authoritative.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

DIRECT_PROBE_SCHEMA = "mlx2.ast-whole-layer-direct-probe.v1"
CAPABILITY_SCHEMA = "mlx2.ast-qualified-physical-target.v1"
DISCOVERY_SCHEMA = "mlx2.ast-qualified-capability-discovery.v1"
QUALIFICATION_DOMAIN_SCHEMA = "mlx2.ast-whole-layer-qualified-domain.v1"

EXECUTION_SCOPE = "whole_transformer_layer_ragged_native_flat"
FIDELITY = "exact"
ORACLE = "ordinary_serial_b1s1"
NUMERICAL_EQUIVALENCE = "bit_exact"
AUTHORITY = "source_build_device_artifact_profile_direct_probe"

_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_SAFE_TEXT = re.compile(r"[ -~]+\Z")
_ARTIFACT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*@sha256:[0-9a-f]{64}\Z")
_LIMIT_FIELDS = ("max_lanes", "max_segment_rows", "max_total_rows")
_SESSION_FIELDS = (
    "source_revision",
    "source_manifest_sha256",
    "build",
    "device",
    "artifact",
    "profile_sha256",
    "provider_binding_sha256",
    "owner_capability_manifest_sha256",
    "target_capability_manifest_sha256",
    "executor_protocol",
)
_REQUIRED_GATES = (
    "outputs",
    "features",
    "prepared_state",
    "continuation",
    "zero_commit",
    "partial_commit",
    "full_commit",
    "abort_rollback",
    "lane_isolation",
    "ordinary_fallback",
)
_DIRECT_PROBE_FIELDS = {
    "schema",
    "execution_scope",
    "fidelity",
    "oracle",
    "numerical_equivalence",
    "session_binding",
    "layout_classes",
    "hard_limits",
    "qualification_domain",
    "coverage",
    "passed",
    "performance_claim",
}
_DISCOVERY_FIELDS = {
    "schema",
    "available",
    "decline_reason",
    "capability_manifest_sha256",
    "direct_probe_receipt_sha256",
    "authority",
    "qualified",
    "eligible",
    "selected",
    "default",
    "observed_used",
    "performance_claim",
    "apcv2_authoritative",
}
_UNAVAILABLE_DECLINE_REASONS = frozenset(
    {
        "qualification_unavailable_or_stale",
        "qualification_warmup_inputs_failed",
        "qualification_warmup_target_failed",
        "qualification_warmup_owner_failed",
        "qualification_warmup_authority_failed",
        "qualification_warmup_session_failed",
        "qualification_warmup_numerical_failed",
        "qualification_warmup_postcheck_failed",
    }
)
_REQUEST_CLASS_FIELDS = {
    "query_lengths",
    "base_offsets",
    "capacities",
    "allocation_steps",
    "capture_schema_sha256",
    "state_schema_sha256",
}
_CAPABILITY_SEAL = object()


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


def _revision(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("AST qualification source revision must be a git SHA")
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


def _limits(value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping) or set(value) != set(_LIMIT_FIELDS):
        raise ValueError("AST qualification hard-limit fields differ")
    checked = {}
    for name in _LIMIT_FIELDS:
        item = value[name]
        if type(item) is not int or not 1 <= item <= 65536:
            raise ValueError("AST qualification hard limits differ")
        checked[name] = item
    if checked["max_total_rows"] < checked["max_segment_rows"]:
        raise ValueError("AST qualification total-row limit is too small")
    return tuple(sorted(checked.items()))


def _layouts(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("AST qualification layouts are unavailable")
    layouts = tuple(value)
    if layouts != ("flat_segmented",):
        raise ValueError("AST qualification layouts differ")
    return layouts


def _session_binding(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or set(value) != set(_SESSION_FIELDS):
        raise ValueError("AST qualification session binding fields differ")
    checked = {
        "source_revision": _revision(value["source_revision"]),
        "source_manifest_sha256": _digest(
            value["source_manifest_sha256"], "AST qualification source manifest"
        ),
        "build": _text(value["build"], "AST qualification build"),
        "device": _text(value["device"], "AST qualification device"),
        "artifact": _text(value["artifact"], "AST qualification artifact"),
        "profile_sha256": _digest(value["profile_sha256"], "AST qualification profile"),
        "provider_binding_sha256": _digest(
            value["provider_binding_sha256"], "AST qualification provider binding"
        ),
        "owner_capability_manifest_sha256": _digest(
            value["owner_capability_manifest_sha256"],
            "AST qualification owner capability",
        ),
        "target_capability_manifest_sha256": _digest(
            value["target_capability_manifest_sha256"],
            "AST qualification target capability",
        ),
        "executor_protocol": _protocol(
            value["executor_protocol"], "AST qualification executor protocol"
        ),
    }
    if _ARTIFACT.fullmatch(checked["artifact"]) is None:
        raise ValueError("AST qualification artifact must be fingerprinted")
    return tuple(sorted(checked.items()))


def _request_class(value: object) -> str:
    if not isinstance(value, Mapping) or set(value) != _REQUEST_CLASS_FIELDS:
        raise ValueError("AST qualification request-class fields differ")
    arrays: dict[str, list[int]] = {}
    for name in ("query_lengths", "base_offsets", "capacities", "allocation_steps"):
        raw = value[name]
        if not isinstance(raw, (list, tuple)) or not 1 <= len(raw) <= 256:
            raise ValueError("AST qualification request-class geometry differs")
        if any(type(item) is not int for item in raw):
            raise ValueError("AST qualification request-class geometry differs")
        arrays[name] = list(raw)
    lane_count = len(arrays["query_lengths"])
    if any(len(items) != lane_count for items in arrays.values()):
        raise ValueError("AST qualification request-class lane counts differ")
    if any(not 1 <= item <= 65536 for item in arrays["query_lengths"]):
        raise ValueError("AST qualification query lengths differ")
    if any(item < 0 or item > 65536 for item in arrays["base_offsets"]):
        raise ValueError("AST qualification base offsets differ")
    if any(item < 0 or item > 65536 for item in arrays["capacities"]):
        raise ValueError("AST qualification capacities differ")
    if any(not 1 <= item <= 65536 for item in arrays["allocation_steps"]):
        raise ValueError("AST qualification allocation steps differ")
    checked = {
        **arrays,
        "capture_schema_sha256": _digest(
            value["capture_schema_sha256"], "AST qualification capture schema"
        ),
        "state_schema_sha256": _digest(
            value["state_schema_sha256"], "AST qualification state schema"
        ),
    }
    return _canonical(checked).decode()


def _qualification_domain(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "request_classes"}
        or value.get("schema") != QUALIFICATION_DOMAIN_SCHEMA
    ):
        raise ValueError("AST qualification domain differs")
    raw_classes = value.get("request_classes")
    if not isinstance(raw_classes, (list, tuple)) or not 1 <= len(raw_classes) <= 64:
        raise ValueError("AST qualification request classes are unavailable")
    classes = tuple(_request_class(item) for item in raw_classes)
    if len(set(classes)) != len(classes):
        raise ValueError("AST qualification request classes are duplicated")
    return tuple(sorted(classes))


@dataclass(frozen=True, slots=True)
class ASTQualifiedPhysicalTargetCapability:
    """Immutable result of an exact direct-probe campaign."""

    session_binding: tuple[tuple[str, str], ...]
    layout_classes: tuple[str, ...]
    hard_limits: tuple[tuple[str, int], ...]
    qualified_request_classes: tuple[str, ...]
    coverage_sha256: str
    direct_probe_receipt_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _CAPABILITY_SEAL:
            raise ValueError("AST qualified capability was not authority-minted")
        if self.session_binding != _session_binding(dict(self.session_binding)):
            raise ValueError("AST qualified capability session binding differs")
        if self.layout_classes != _layouts(self.layout_classes):
            raise ValueError("AST qualified capability layouts differ")
        if self.hard_limits != _limits(dict(self.hard_limits)):
            raise ValueError("AST qualified capability hard limits differ")
        if (
            not self.qualified_request_classes
            or len(set(self.qualified_request_classes))
            != len(self.qualified_request_classes)
            or tuple(sorted(self.qualified_request_classes))
            != self.qualified_request_classes
        ):
            raise ValueError("AST qualified capability request classes differ")
        for request_class in self.qualified_request_classes:
            parsed = json.loads(request_class)
            if _request_class(parsed) != request_class:
                raise ValueError("AST qualified capability request class differs")
        _digest(self.coverage_sha256, "AST qualified capability coverage")
        _digest(
            self.direct_probe_receipt_sha256,
            "AST qualified capability direct-probe receipt",
        )

    @property
    def manifest_sha256(self) -> str:
        return _sha256(self.receipt())

    def receipt(self) -> dict[str, object]:
        return {
            "schema": CAPABILITY_SCHEMA,
            "execution_scope": EXECUTION_SCOPE,
            "fidelity": FIDELITY,
            "oracle": ORACLE,
            "numerical_equivalence": NUMERICAL_EQUIVALENCE,
            "authority": AUTHORITY,
            "session_binding": dict(self.session_binding),
            "layout_classes": list(self.layout_classes),
            "hard_limits": dict(self.hard_limits),
            "qualification_domain": {
                "schema": QUALIFICATION_DOMAIN_SCHEMA,
                "request_classes": [
                    json.loads(item) for item in self.qualified_request_classes
                ],
            },
            "coverage_sha256": self.coverage_sha256,
            "direct_probe_receipt_sha256": self.direct_probe_receipt_sha256,
            "implemented": True,
            "qualified": True,
            "eligible": False,
            "selected": False,
            "default": False,
            "observed_used": False,
            "performance_claim": False,
            "apcv2_authoritative": False,
        }


def seal_ast_direct_probe_receipt(body: Mapping[str, object]) -> dict[str, object]:
    """Seal a direct-probe result after its runner has populated every gate."""

    if not isinstance(body, Mapping) or "receipt_sha256" in body:
        raise ValueError("AST direct-probe body is invalid")
    result = dict(body)
    result["receipt_sha256"] = _sha256(result)
    return result


def mint_ast_qualified_capability(
    direct_probe_receipt: Mapping[str, object],
) -> ASTQualifiedPhysicalTargetCapability:
    """Mint a capability from a complete, exact, integrity-bound direct probe."""

    if not isinstance(direct_probe_receipt, Mapping):
        raise TypeError("AST direct-probe receipt must be a mapping")
    body = dict(direct_probe_receipt)
    receipt_sha256 = body.pop("receipt_sha256", None)
    if receipt_sha256 != _sha256(body):
        raise ValueError("AST direct-probe receipt integrity differs")
    if set(body) != _DIRECT_PROBE_FIELDS:
        raise ValueError("AST direct-probe fields differ")
    if (
        body.get("schema") != DIRECT_PROBE_SCHEMA
        or body.get("execution_scope") != EXECUTION_SCOPE
        or body.get("fidelity") != FIDELITY
        or body.get("oracle") != ORACLE
        or body.get("numerical_equivalence") != NUMERICAL_EQUIVALENCE
        or body.get("passed") is not True
        or body.get("performance_claim") is not False
    ):
        raise ValueError("AST direct-probe authority differs")
    coverage = body.get("coverage")
    if (
        not isinstance(coverage, Mapping)
        or set(coverage) != set(_REQUIRED_GATES)
        or any(coverage[name] is not True for name in _REQUIRED_GATES)
    ):
        raise ValueError("AST direct-probe coverage is incomplete")
    session_binding = _session_binding(body.get("session_binding"))
    layouts = _layouts(body.get("layout_classes"))
    hard_limits = _limits(body.get("hard_limits"))
    qualification_domain = _qualification_domain(body.get("qualification_domain"))
    limits = dict(hard_limits)
    for encoded in qualification_domain:
        request = json.loads(encoded)
        lengths = request["query_lengths"]
        if (
            len(lengths) > limits["max_lanes"]
            or max(lengths) > limits["max_segment_rows"]
            or sum(lengths) > limits["max_total_rows"]
        ):
            raise ValueError("AST qualification domain exceeds hard limits")
    return ASTQualifiedPhysicalTargetCapability(
        session_binding=session_binding,
        layout_classes=layouts,
        hard_limits=hard_limits,
        qualified_request_classes=qualification_domain,
        coverage_sha256=_sha256(dict(coverage)),
        direct_probe_receipt_sha256=_digest(receipt_sha256, "AST direct-probe receipt"),
        _seal=_CAPABILITY_SEAL,
    )


def load_ast_qualified_capability_evidence(
    evidence: Mapping[str, object],
) -> ASTQualifiedPhysicalTargetCapability:
    """Load a sealed capability from a direct receipt or runner envelope.

    Runner evidence deliberately carries operational receipts outside the
    authority-minted direct-probe body.  Reconstruct exactly that closed body,
    verify its seal, and require the embedded capability to agree byte-for-byte
    with the independently minted result.  This performs no model work.
    """

    if not isinstance(evidence, Mapping):
        raise TypeError("AST qualification evidence must be a mapping")
    if set(evidence) == (_DIRECT_PROBE_FIELDS | {"receipt_sha256"}):
        return mint_ast_qualified_capability(evidence)
    if (
        evidence.get("verdict") != "pass_exact_direct_probe"
        or evidence.get("cleanup_completed") is not True
        or evidence.get("performance_claim") is not False
    ):
        raise ValueError("AST qualification runner evidence did not pass")
    missing = _DIRECT_PROBE_FIELDS - set(evidence)
    if missing:
        raise ValueError(
            "AST qualification runner evidence is incomplete: "
            + ",".join(sorted(missing))
        )
    direct = {name: evidence[name] for name in _DIRECT_PROBE_FIELDS}
    direct["receipt_sha256"] = evidence.get("receipt_sha256")
    capability = mint_ast_qualified_capability(direct)
    if evidence.get("qualified_capability") != capability.receipt():
        raise ValueError("AST qualification embedded capability differs")
    if (
        evidence.get("source_revision")
        != dict(capability.session_binding)["source_revision"]
    ):
        raise ValueError("AST qualification runner source revision differs")
    return capability


def unavailable_ast_qualified_capability_discovery(
    decline_reason: str = "qualification_unavailable_or_stale",
) -> dict[str, object]:
    """Canonical ordinary-fallback receipt for absent or malformed authority."""

    if decline_reason not in _UNAVAILABLE_DECLINE_REASONS:
        raise ValueError("AST qualification decline reason is unavailable")

    return {
        "schema": DISCOVERY_SCHEMA,
        "available": False,
        "decline_reason": decline_reason,
        "capability_manifest_sha256": None,
        "direct_probe_receipt_sha256": None,
        "authority": "none",
        "qualified": False,
        "eligible": False,
        "selected": False,
        "default": False,
        "observed_used": False,
        "performance_claim": False,
        "apcv2_authoritative": False,
    }


def normalize_ast_qualified_capability_discovery(value) -> dict[str, object]:
    """Admit only the exact positive discovery shape; otherwise fail closed."""

    if not isinstance(value, Mapping) or set(value) != _DISCOVERY_FIELDS:
        return unavailable_ast_qualified_capability_discovery()
    if value.get("available") is False:
        reason = value.get("decline_reason")
        if reason in _UNAVAILABLE_DECLINE_REASONS:
            expected = unavailable_ast_qualified_capability_discovery(reason)
            if dict(value) == expected:
                return expected
        return unavailable_ast_qualified_capability_discovery()
    if (
        value.get("schema") != DISCOVERY_SCHEMA
        or value.get("available") is not True
        or value.get("qualified") is not True
        or value.get("eligible") is not False
        or value.get("selected") is not False
        or value.get("default") is not False
        or value.get("observed_used") is not False
        or value.get("performance_claim") is not False
        or value.get("apcv2_authoritative") is not False
        or value.get("decline_reason") is not None
        or value.get("authority") != AUTHORITY
    ):
        return unavailable_ast_qualified_capability_discovery()
    try:
        _digest(
            value.get("capability_manifest_sha256"),
            "AST qualified capability manifest",
        )
        _digest(
            value.get("direct_probe_receipt_sha256"),
            "AST qualified direct-probe receipt",
        )
    except (TypeError, ValueError):
        return unavailable_ast_qualified_capability_discovery()
    return dict(value)


def validate_ast_qualified_session(
    capability: ASTQualifiedPhysicalTargetCapability,
    session_binding: Mapping[str, object],
    *,
    layout_class: str,
    query_lengths: tuple[int, ...],
    base_offsets: tuple[int, ...],
    capacities: tuple[int, ...],
    allocation_steps: tuple[int, ...],
    capture_schema_sha256: str,
    state_schema_sha256: str,
) -> None:
    """Cheap immutable serving check; performs no model or tensor operation."""

    if type(capability) is not ASTQualifiedPhysicalTargetCapability:
        raise TypeError("AST qualified capability is unavailable")
    capability.__post_init__()
    if capability.session_binding != _session_binding(session_binding):
        raise ValueError("AST qualified session binding differs")
    layout = _protocol(layout_class, "AST qualified request layout")
    if layout not in capability.layout_classes:
        raise ValueError("AST qualified request layout is unavailable")
    request_class = _request_class(
        {
            "query_lengths": query_lengths,
            "base_offsets": base_offsets,
            "capacities": capacities,
            "allocation_steps": allocation_steps,
            "capture_schema_sha256": capture_schema_sha256,
            "state_schema_sha256": state_schema_sha256,
        }
    )
    if request_class not in capability.qualified_request_classes:
        raise ValueError("AST qualified request class is unavailable")
    values = {
        "max_lanes": len(query_lengths),
        "max_segment_rows": max(query_lengths),
        "max_total_rows": sum(query_lengths),
    }
    limits = dict(capability.hard_limits)
    if any(
        type(value) is not int or value < 1 or value > limits[name]
        for name, value in values.items()
    ):
        raise ValueError("AST qualified request exceeds hard limits")


def discover_ast_qualified_capability(
    provider,
    *,
    provider_getter,
    session_binding: Mapping[str, object],
    layout_class: str,
    query_lengths: tuple[int, ...],
    base_offsets: tuple[int, ...],
    capacities: tuple[int, ...],
    allocation_steps: tuple[int, ...],
    capture_schema_sha256: str,
    state_schema_sha256: str,
) -> dict[str, object]:
    """Generic fail-closed discovery; never selects or invokes an executor."""

    try:
        if (
            provider is None
            or not callable(provider_getter)
            or provider is not provider_getter()
        ):
            raise ValueError("AST qualification provider ownership drifted")
        validate_revision = getattr(provider, "validate_revision", None)
        capability_provider = getattr(
            provider, "ast_qualified_physical_target_capability", None
        )
        if not callable(validate_revision) or not callable(capability_provider):
            raise TypeError("AST qualification provider protocol is incomplete")
        validate_revision()
        capability = capability_provider()
        validate_ast_qualified_session(
            capability,
            session_binding,
            layout_class=layout_class,
            query_lengths=query_lengths,
            base_offsets=base_offsets,
            capacities=capacities,
            allocation_steps=allocation_steps,
            capture_schema_sha256=capture_schema_sha256,
            state_schema_sha256=state_schema_sha256,
        )
        validate_revision()
        if provider is not provider_getter():
            raise ValueError("AST qualification provider changed during discovery")
        return {
            "schema": DISCOVERY_SCHEMA,
            "available": True,
            "decline_reason": None,
            "capability_manifest_sha256": capability.manifest_sha256,
            "direct_probe_receipt_sha256": capability.direct_probe_receipt_sha256,
            "authority": AUTHORITY,
            "qualified": True,
            "eligible": False,
            "selected": False,
            "default": False,
            "observed_used": False,
            "performance_claim": False,
            "apcv2_authoritative": False,
        }
    except Exception:  # noqa: BLE001 - discovery cannot disturb ordinary serving
        return unavailable_ast_qualified_capability_discovery()


__all__ = [
    "CAPABILITY_SCHEMA",
    "DIRECT_PROBE_SCHEMA",
    "DISCOVERY_SCHEMA",
    "QUALIFICATION_DOMAIN_SCHEMA",
    "ASTQualifiedPhysicalTargetCapability",
    "discover_ast_qualified_capability",
    "load_ast_qualified_capability_evidence",
    "mint_ast_qualified_capability",
    "normalize_ast_qualified_capability_discovery",
    "seal_ast_direct_probe_receipt",
    "unavailable_ast_qualified_capability_discovery",
    "validate_ast_qualified_session",
]
