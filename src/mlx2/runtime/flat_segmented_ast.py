"""Contract foundation for one-invocation flattened AST execution.

This module does not select or wire a serving backend.  It defines the semantic
boundary an adapter and transaction owner must both attest before flat
segments can execute. The generic executor helper remains restricted to the
synthetic CPU scope; real CPU and Metal execution use distinct adapter-owned
protocols and device-bound opaque tickets.

``flat_segmented`` means that every lane contributes its complete ``K + 1``
causal target path to one target-executor invocation.  It never means a
depth-wise loop over a shrinking live cohort.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Integral

DESCRIPTOR_SCHEMA = "mlx2.ast-flat-segment-layout.v1"
OWNER_CAPABILITY_SCHEMA = "mlx2.ast-flat-segment-owner-capability.v1"
TARGET_CAPABILITY_SCHEMA = "mlx2.ast-flat-segment-target-capability.v1"
ADMISSION_SCHEMA = "mlx2.ast-flat-segment-admission.v1"
EXECUTION_SCHEMA = "mlx2.ast-flat-segment-execution.v1"
NATIVE_OWNER_CAPABILITY_SCHEMA = "mlx2.ast-native-flat-owner-capability.v2"
NATIVE_TARGET_CAPABILITY_SCHEMA = "mlx2.ast-native-flat-target-capability.v2"
NATIVE_ADMISSION_SCHEMA = "mlx2.ast-native-flat-admission.v2"

LAYOUT_CLASS = "flat_segmented"
OWNER_PROTOCOL = "flat-segmented-transaction-v1"
TARGET_PROTOCOL = "flat-segmented-row-exact-v1"
EXECUTION_SCOPE = "synthetic_cpu_contract_only"
REAL_EXECUTION_SCOPE = "adapter_owned_real_cpu_contract"
REAL_OWNER_PROTOCOL = "flat-segmented-transaction-real-v1"
REAL_TARGET_PROTOCOL = "flat-segmented-row-exact-real-v1"
METAL_EXECUTION_SCOPE = "adapter_owned_metal_exact_artifact_probe"
METAL_OWNER_PROTOCOL = "flat-segmented-transaction-metal-probe-v1"
METAL_TARGET_PROTOCOL = "flat-segmented-row-exact-metal-probe-v1"
NATIVE_EXECUTION_SCOPE = "adapter_owned_native_flat_physical_contract_v2"
NATIVE_OWNER_PROTOCOL = "native-flat-private-hybrid-transaction-v2"
NATIVE_TARGET_PROTOCOL = "native-flat-logical-row-executor-v2"
NATIVE_PHYSICAL_LAYOUT = "native_flat_logical_rows"
DEVICE_ATTESTATION_SCHEMA = "mlx2.ast-flat-segment-device.v1"
FIDELITY_EXACT = "exact"

COMMIT_LAW = "accepted_verified_input_prefix_per_lane"
ABORT_LAW = "restore_exact_pre_begin_state"
TERMINAL_LAW = "k_proposals_require_k_plus_one_target_rows"
STATE_WRITE_LAW = "transaction_private_until_commit"
BEGIN_LAW = "atomic_lease_or_exact_pre_begin_state"
COMMIT_FAILURE_LAW = "restore_exact_pre_begin_state"
POSITION_BASE_AUTHORITY = "transaction_owner_state"
EXECUTION_LAW = "one_target_executor_invocation_for_all_complete_paths"
STATE_OWNERSHIP = "independent_request_private_b1_lanes"
NATIVE_PHYSICAL_ROWS_LAW = "physical_rows_equal_descriptor_logical_rows"
NATIVE_OUTER_INVOCATION_LAW = "exactly_one_outer_target_invocation"
NATIVE_ROW_ARITHMETIC_LAW = "ordinary_b1s1_exact_per_logical_row"
NATIVE_STATEFUL_CORE_LAW = "per_segment_order_no_cross_lane_reads"
NATIVE_LANE_STAGING_LAW = "one_private_b1_transaction_per_lane"
NATIVE_AGGREGATE_COMMIT_LAW = "prepare_only_after_every_lane_commit_succeeds"

_STATE_CONTRACTS = {
    (("attention_kv",), ("kv",)): "private_kv_snapshot_restore",
    (
        ("attention_kv", "gdn_recurrent"),
        ("kv", "recurrent"),
    ): "private_kv_and_recurrent_snapshot_restore",
}

_IDENTITY_FIELDS = ("source_revision", "build", "device", "artifact")
_HARD_LIMIT_FIELDS = ("max_lanes", "max_segment_rows", "max_total_rows")


class FlatSegmentError(ValueError):
    """A flat-segment descriptor or capability is malformed."""


class FlatSegmentUnsupported(RuntimeError):
    """The requested owner/executor pair did not attest the flat contract."""


_REAL_ADMISSION_SEAL = object()
_NATIVE_ADMISSION_SEAL = object()


@dataclass(frozen=True, slots=True)
class _BoundFlatAdmissionTicket:
    seal: object
    owner_id: int
    target_id: int
    owner_revision: int
    owner_membership_identity: tuple[tuple[int, ...], ...]
    owner_state_identity_sha256: str
    target_execution_identity: tuple[int, tuple[int, ...]]
    device_attestation: tuple[str, str, str, str, bool]
    admission_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _BoundNativeFlatAdmissionTicket:
    seal: object
    owner_id: int
    target_id: int
    owner_revision: int
    owner_membership_identity: tuple[tuple[int, ...], ...]
    owner_state_identity_sha256: str
    target_execution_identity: tuple[int, tuple[int, ...]]
    device_attestation: tuple[str, str, str, str, bool]
    admission_manifest_sha256: str


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _strict_device_attestation(raw: object, *, backend: str) -> dict:
    fields = {"schema", "backend", "device", "build", "ready"}
    if (
        not isinstance(raw, Mapping)
        or set(raw) != fields
        or raw.get("schema") != DEVICE_ATTESTATION_SCHEMA
        or raw.get("backend") != backend
        or not isinstance(raw.get("device"), str)
        or not raw.get("device")
        or not isinstance(raw.get("build"), str)
        or not raw.get("build")
        or type(raw.get("ready")) is not bool
    ):
        raise FlatSegmentError("flat device attestation differs")
    return {name: raw[name] for name in sorted(fields)}


def _device_attestation_identity(attestation: Mapping[str, object]) -> tuple:
    return (
        attestation["schema"],
        attestation["backend"],
        attestation["device"],
        attestation["build"],
        attestation["ready"],
    )


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise FlatSegmentError(f"{label} must be a positive integer")
    return int(value)


def _integer_tuple(
    values: Iterable[int], *, label: str, minimum: int, allow_empty: bool = False
) -> tuple[int, ...]:
    try:
        result = tuple(values)
    except TypeError as error:
        raise FlatSegmentError(f"{label} must be an integer sequence") from error
    if (not result and not allow_empty) or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < minimum
        for value in result
    ):
        suffix = " or an empty sequence" if allow_empty else ""
        raise FlatSegmentError(f"{label} requires integers >= {minimum}{suffix}")
    return tuple(int(value) for value in result)


@dataclass(frozen=True, slots=True)
class FlatSegmentDescriptor:
    """Immutable, token-free geometry for one complete flattened AST round."""

    query_lengths: tuple[int, ...]
    cu_query_lengths: tuple[int, ...]
    token_to_lane: tuple[int, ...]
    local_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        lengths = _integer_tuple(self.query_lengths, label="query_lengths", minimum=1)
        offsets = _integer_tuple(
            self.cu_query_lengths,
            label="cu_query_lengths",
            minimum=0,
        )
        token_to_lane = _integer_tuple(
            self.token_to_lane,
            label="token_to_lane",
            minimum=0,
        )
        positions = _integer_tuple(
            self.local_positions,
            label="local_positions",
            minimum=0,
        )
        expected_offsets = [0]
        expected_lanes = []
        expected_positions = []
        for lane, length in enumerate(lengths):
            expected_offsets.append(expected_offsets[-1] + length)
            expected_lanes.extend([lane] * length)
            expected_positions.extend(range(length))
        if offsets != tuple(expected_offsets):
            raise FlatSegmentError(
                "cu_query_lengths must be the exact cumulative query lengths"
            )
        if token_to_lane != tuple(expected_lanes):
            raise FlatSegmentError(
                "token_to_lane must preserve contiguous segment order"
            )
        if positions != tuple(expected_positions):
            raise FlatSegmentError(
                "local_positions must restart at zero for every segment"
            )
        object.__setattr__(self, "query_lengths", lengths)
        object.__setattr__(self, "cu_query_lengths", offsets)
        object.__setattr__(self, "token_to_lane", token_to_lane)
        object.__setattr__(self, "local_positions", positions)

    @classmethod
    def from_query_lengths(cls, query_lengths: Iterable[int]) -> FlatSegmentDescriptor:
        lengths = _integer_tuple(query_lengths, label="query_lengths", minimum=1)
        offsets = [0]
        lanes = []
        positions = []
        for lane, length in enumerate(lengths):
            offsets.append(offsets[-1] + length)
            lanes.extend([lane] * length)
            positions.extend(range(length))
        return cls(lengths, tuple(offsets), tuple(lanes), tuple(positions))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> FlatSegmentDescriptor:
        allowed = {
            "schema",
            "layout_class",
            "query_lengths",
            "cu_query_lengths",
            "token_to_lane",
            "local_positions",
            "geometry_sha256",
        }
        if not isinstance(raw, Mapping) or set(raw) != allowed:
            raise FlatSegmentError("flat descriptor fields differ from the schema")
        if raw.get("schema") != DESCRIPTOR_SCHEMA:
            raise FlatSegmentError("flat descriptor schema differs")
        if raw.get("layout_class") != LAYOUT_CLASS:
            raise FlatSegmentError("flat descriptor layout class differs")
        try:
            descriptor = cls(
                raw["query_lengths"],
                raw["cu_query_lengths"],
                raw["token_to_lane"],
                raw["local_positions"],
            )
        except TypeError as error:
            raise FlatSegmentError(
                "flat descriptor arrays must be sequences"
            ) from error
        if raw.get("geometry_sha256") != descriptor.geometry_sha256:
            raise FlatSegmentError("flat descriptor geometry digest differs")
        return descriptor

    @property
    def lane_count(self) -> int:
        return len(self.query_lengths)

    @property
    def logical_rows(self) -> int:
        return self.cu_query_lengths[-1]

    @property
    def max_segment_rows(self) -> int:
        return max(self.query_lengths)

    @property
    def draft_depths(self) -> tuple[int, ...]:
        return tuple(length - 1 for length in self.query_lengths)

    @property
    def geometry_sha256(self) -> str:
        return _sha256(
            {
                "query_lengths": self.query_lengths,
                "cu_query_lengths": self.cu_query_lengths,
                "token_to_lane": self.token_to_lane,
                "local_positions": self.local_positions,
            }
        )

    def lane_slice(self, lane: int) -> slice:
        if (
            isinstance(lane, bool)
            or not isinstance(lane, Integral)
            or not 0 <= lane < self.lane_count
        ):
            raise FlatSegmentError("lane index is outside the flat descriptor")
        return slice(self.cu_query_lengths[lane], self.cu_query_lengths[lane + 1])

    def validate_accepted_drafts(
        self, accepted_draft_lengths: Iterable[int]
    ) -> tuple[int, ...]:
        accepted = tuple(accepted_draft_lengths)
        if len(accepted) != self.lane_count or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in accepted
        ):
            raise FlatSegmentError(
                "accepted draft lengths must cover every flat segment"
            )
        accepted = tuple(int(value) for value in accepted)
        if any(
            value < 0 or value > depth
            for value, depth in zip(accepted, self.draft_depths, strict=True)
        ):
            raise FlatSegmentError("accepted draft prefix exceeds its proposal depth")
        return accepted

    def receipt(self) -> dict:
        return {
            "schema": DESCRIPTOR_SCHEMA,
            "layout_class": LAYOUT_CLASS,
            "query_lengths": list(self.query_lengths),
            "cu_query_lengths": list(self.cu_query_lengths),
            "token_to_lane": list(self.token_to_lane),
            "local_positions": list(self.local_positions),
            "geometry_sha256": self.geometry_sha256,
        }


def _strings(raw: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(raw, (list, tuple))
        or not raw
        or any(not isinstance(value, str) or not value for value in raw)
    ):
        raise FlatSegmentError(f"{label} must contain nonempty strings")
    return tuple(raw)


def _unique_bounded_strings(
    raw: object, label: str, *, allowed: frozenset[str] | None = None
) -> tuple[str, ...]:
    values = _strings(raw, label)
    if len(values) > 16 or values != tuple(sorted(set(values))):
        raise FlatSegmentError(f"{label} must be sorted, unique, and bounded")
    if allowed is not None and not set(values).issubset(allowed):
        raise FlatSegmentError(f"{label} contains an unsupported semantic value")
    return values


def _sha256_string(raw: object, label: str) -> str:
    if (
        not isinstance(raw, str)
        or len(raw) != 64
        or any(character not in "0123456789abcdef" for character in raw)
    ):
        raise FlatSegmentError(f"{label} must be a lowercase SHA-256")
    return raw


def _fingerprints(raw: object) -> tuple[str, ...]:
    values = _unique_bounded_strings(raw, "state_contract_fingerprints")
    return tuple(
        _sha256_string(value, "state contract fingerprint") for value in values
    )


def _hard_limits(raw: object) -> dict[str, int]:
    if not isinstance(raw, Mapping) or set(raw) != set(_HARD_LIMIT_FIELDS):
        raise FlatSegmentError("flat hard-limit fields differ")
    return {
        key: _positive_int(raw[key], f"hard_limits.{key}") for key in _HARD_LIMIT_FIELDS
    }


def _identity(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != set(_IDENTITY_FIELDS):
        raise FlatSegmentError("flat executor identity fields differ")
    if any(not isinstance(raw[key], str) or not raw[key] for key in _IDENTITY_FIELDS):
        raise FlatSegmentError("flat executor identity values must be nonempty strings")
    return {key: raw[key] for key in _IDENTITY_FIELDS}


def _source_hashes(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping) or not raw:
        raise FlatSegmentError("flat source hashes are required")
    if any(
        not isinstance(path, str)
        or not path
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for path, digest in raw.items()
    ):
        raise FlatSegmentError("flat source hash differs from SHA-256")
    return dict(sorted(raw.items()))


def _strict_owner_contract(
    raw: object,
    *,
    execution_scope: str = EXECUTION_SCOPE,
    protocol: str = OWNER_PROTOCOL,
    schema: str = OWNER_CAPABILITY_SCHEMA,
    begin_operation: str = "begin_flat",
) -> dict:
    fields = {
        "schema",
        "protocol",
        "execution_scope",
        "descriptor_schema",
        "layout_classes",
        "state_ownership",
        "state_plane_kinds",
        "state_families",
        "state_contract_fingerprints",
        "owner_semantic_contract_sha256",
        "transaction_operations",
        "begin_law",
        "commit_law",
        "commit_failure_law",
        "abort_law",
        "rollback_law",
        "position_base_authority",
        "hard_limits",
        "source_hashes",
        "identity",
    }
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise FlatSegmentError("flat owner capability fields differ")
    exact = {
        "schema": schema,
        "protocol": protocol,
        "execution_scope": execution_scope,
        "descriptor_schema": DESCRIPTOR_SCHEMA,
        "layout_classes": [LAYOUT_CLASS],
        "begin_law": BEGIN_LAW,
        "commit_law": COMMIT_LAW,
        "commit_failure_law": COMMIT_FAILURE_LAW,
        "abort_law": ABORT_LAW,
        "position_base_authority": POSITION_BASE_AUTHORITY,
    }
    if any(raw.get(key) != value for key, value in exact.items()):
        raise FlatSegmentError("flat owner capability semantics differ")
    operations = raw.get("transaction_operations")
    if (
        not isinstance(operations, Mapping)
        or set(operations)
        != {
            begin_operation,
            "prepare_commit",
            "publish_commit",
            "abort",
        }
        or any(operations[name] is not True for name in operations)
    ):
        raise FlatSegmentError("flat owner transaction operations differ")
    if raw.get("state_ownership") != STATE_OWNERSHIP:
        raise FlatSegmentError("flat owner state ownership differs")
    planes = _unique_bounded_strings(
        raw["state_plane_kinds"],
        "state_plane_kinds",
        allowed=frozenset({"attention_kv", "gdn_recurrent"}),
    )
    families = _unique_bounded_strings(
        raw["state_families"],
        "state_families",
        allowed=frozenset({"kv", "recurrent"}),
    )
    rollback_law = _STATE_CONTRACTS.get((planes, families))
    if rollback_law is None or raw.get("rollback_law") != rollback_law:
        raise FlatSegmentError("flat owner state planes or rollback law differ")
    return {
        **exact,
        "state_ownership": raw["state_ownership"],
        "state_plane_kinds": planes,
        "state_families": families,
        "state_contract_fingerprints": _fingerprints(
            raw["state_contract_fingerprints"]
        ),
        "owner_semantic_contract_sha256": _sha256_string(
            raw["owner_semantic_contract_sha256"],
            "owner semantic contract digest",
        ),
        "transaction_operations": {
            begin_operation: True,
            "prepare_commit": True,
            "publish_commit": True,
            "abort": True,
        },
        "rollback_law": rollback_law,
        "hard_limits": _hard_limits(raw["hard_limits"]),
        "source_hashes": _source_hashes(raw["source_hashes"]),
        "identity": _identity(raw["identity"]),
    }


def _strict_target_contract(
    raw: object,
    *,
    execution_scope: str = EXECUTION_SCOPE,
    protocol: str = TARGET_PROTOCOL,
    schema: str = TARGET_CAPABILITY_SCHEMA,
) -> dict:
    fields = {
        "schema",
        "protocol",
        "execution_scope",
        "fidelity",
        "execution_law",
        "descriptor_schema",
        "layout_classes",
        "input_storage",
        "output_storage",
        "position_law",
        "causal_isolation",
        "terminal_law",
        "state_write_law",
        "commit_law",
        "abort_law",
        "hard_limits",
        "source_hashes",
        "identity",
    }
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise FlatSegmentError("flat target capability fields differ")
    exact = {
        "schema": schema,
        "protocol": protocol,
        "execution_scope": execution_scope,
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
    }
    if any(raw.get(key) != value for key, value in exact.items()):
        raise FlatSegmentError("flat target capability semantics differ")
    return {
        **exact,
        "hard_limits": _hard_limits(raw["hard_limits"]),
        "source_hashes": _source_hashes(raw["source_hashes"]),
        "identity": _identity(raw["identity"]),
    }


@dataclass(frozen=True, slots=True)
class FlatSegmentAdmission:
    available: bool
    decline_reasons: tuple[str, ...]
    descriptor: FlatSegmentDescriptor
    owner_contract: dict | None
    target_contract: dict | None
    effective_hard_limits: dict[str, int] | None
    device_attestation: dict | None = None
    execution_scope: str = EXECUTION_SCOPE
    _authority_ticket: _BoundFlatAdmissionTicket | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def receipt(self) -> dict:
        body = {
            "schema": ADMISSION_SCHEMA,
            "available": self.available,
            "decline_reasons": list(self.decline_reasons),
            "layout_class": LAYOUT_CLASS,
            "execution_scope": self.execution_scope,
            "descriptor": self.descriptor.receipt(),
            "owner_contract": self.owner_contract,
            "target_contract": self.target_contract,
            "effective_hard_limits": self.effective_hard_limits,
            "one_target_executor_invocation_required": True,
            "depthwise_shrinking_loop_permitted": False,
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": False,
        }
        if self.device_attestation is not None:
            body["device_attestation"] = self.device_attestation
        body["manifest_sha256"] = _sha256(body)
        return body


def _strict_native_owner_contract(raw: object) -> dict:
    native_fields = {
        "physical_layout",
        "physical_rows_law",
        "lane_staging_law",
        "aggregate_commit_law",
    }
    if not isinstance(raw, Mapping) or not native_fields.issubset(raw):
        raise FlatSegmentError("native-flat owner capability fields differ")
    if (
        raw.get("physical_layout") != NATIVE_PHYSICAL_LAYOUT
        or raw.get("physical_rows_law") != NATIVE_PHYSICAL_ROWS_LAW
        or raw.get("lane_staging_law") != NATIVE_LANE_STAGING_LAW
        or raw.get("aggregate_commit_law") != NATIVE_AGGREGATE_COMMIT_LAW
    ):
        raise FlatSegmentError("native-flat owner physical semantics differ")
    base = {key: value for key, value in raw.items() if key not in native_fields}
    result = _strict_owner_contract(
        base,
        execution_scope=NATIVE_EXECUTION_SCOPE,
        protocol=NATIVE_OWNER_PROTOCOL,
        schema=NATIVE_OWNER_CAPABILITY_SCHEMA,
        begin_operation="begin_native_flat",
    )
    return {
        **result,
        "physical_layout": NATIVE_PHYSICAL_LAYOUT,
        "physical_rows_law": NATIVE_PHYSICAL_ROWS_LAW,
        "lane_staging_law": NATIVE_LANE_STAGING_LAW,
        "aggregate_commit_law": NATIVE_AGGREGATE_COMMIT_LAW,
    }


def _strict_native_target_contract(raw: object) -> dict:
    native_fields = {
        "physical_layout",
        "physical_rows_law",
        "outer_invocation_law",
        "row_arithmetic_law",
        "stateful_core_law",
    }
    if not isinstance(raw, Mapping) or not native_fields.issubset(raw):
        raise FlatSegmentError("native-flat target capability fields differ")
    if (
        raw.get("physical_layout") != NATIVE_PHYSICAL_LAYOUT
        or raw.get("physical_rows_law") != NATIVE_PHYSICAL_ROWS_LAW
        or raw.get("outer_invocation_law") != NATIVE_OUTER_INVOCATION_LAW
        or raw.get("row_arithmetic_law") != NATIVE_ROW_ARITHMETIC_LAW
        or raw.get("stateful_core_law") != NATIVE_STATEFUL_CORE_LAW
    ):
        raise FlatSegmentError("native-flat target physical semantics differ")
    base = {key: value for key, value in raw.items() if key not in native_fields}
    result = _strict_target_contract(
        base,
        execution_scope=NATIVE_EXECUTION_SCOPE,
        protocol=NATIVE_TARGET_PROTOCOL,
        schema=NATIVE_TARGET_CAPABILITY_SCHEMA,
    )
    return {
        **result,
        "physical_layout": NATIVE_PHYSICAL_LAYOUT,
        "physical_rows_law": NATIVE_PHYSICAL_ROWS_LAW,
        "outer_invocation_law": NATIVE_OUTER_INVOCATION_LAW,
        "row_arithmetic_law": NATIVE_ROW_ARITHMETIC_LAW,
        "stateful_core_law": NATIVE_STATEFUL_CORE_LAW,
    }


@dataclass(frozen=True, slots=True)
class NativeFlatSegmentAdmission:
    """Bound intersection for a native logical-row physical executor."""

    available: bool
    decline_reasons: tuple[str, ...]
    descriptor: FlatSegmentDescriptor
    owner_contract: dict | None
    target_contract: dict | None
    effective_hard_limits: dict[str, int] | None
    device_attestation: dict | None = None
    _authority_ticket: _BoundNativeFlatAdmissionTicket | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def receipt(self) -> dict:
        body = {
            "schema": NATIVE_ADMISSION_SCHEMA,
            "available": self.available,
            "decline_reasons": list(self.decline_reasons),
            "layout_class": LAYOUT_CLASS,
            "physical_layout": NATIVE_PHYSICAL_LAYOUT,
            "execution_scope": NATIVE_EXECUTION_SCOPE,
            "descriptor": self.descriptor.receipt(),
            "logical_rows": self.descriptor.logical_rows,
            "required_physical_rows": self.descriptor.logical_rows,
            "owner_contract": self.owner_contract,
            "target_contract": self.target_contract,
            "effective_hard_limits": self.effective_hard_limits,
            "one_outer_target_invocation_required": True,
            "depthwise_shrinking_loop_permitted": False,
            "padded_physical_execution_permitted": False,
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": False,
        }
        if self.device_attestation is not None:
            body["device_attestation"] = self.device_attestation
        body["manifest_sha256"] = _sha256(body)
        return body


def query_native_flat_segment_admission(
    owner,
    target_executor,
    descriptor: FlatSegmentDescriptor,
    *,
    expected_identity: Mapping[str, str],
    expected_owner_semantic_contract_sha256: str,
    expected_state_contract_fingerprints: Sequence[str],
    expected_owner_source_hashes: Mapping[str, str],
    expected_target_source_hashes: Mapping[str, str],
    expected_backend: str,
    fidelity: str = FIDELITY_EXACT,
) -> NativeFlatSegmentAdmission:
    """Intersect native owner and tensor authorities without selecting a route."""

    if not isinstance(descriptor, FlatSegmentDescriptor):
        raise FlatSegmentError("native-flat admission requires a validated descriptor")
    reasons: list[str] = []
    owner_contract = target_contract = None
    expected = expected_owner_digest = expected_fingerprints = None
    expected_owner_sources = expected_target_sources = None
    try:
        expected = _identity(expected_identity)
    except (FlatSegmentError, TypeError, ValueError):
        reasons.append("expected_identity_invalid")
    try:
        expected_owner_digest = _sha256_string(
            expected_owner_semantic_contract_sha256,
            "expected owner semantic contract digest",
        )
    except (FlatSegmentError, TypeError, ValueError):
        reasons.append("expected_owner_semantic_contract_invalid")
    try:
        expected_fingerprints = _fingerprints(expected_state_contract_fingerprints)
    except (FlatSegmentError, TypeError, ValueError):
        reasons.append("expected_state_contract_fingerprints_invalid")
    try:
        expected_owner_sources = _source_hashes(expected_owner_source_hashes)
    except (FlatSegmentError, TypeError, ValueError):
        reasons.append("expected_owner_source_hashes_invalid")
    try:
        expected_target_sources = _source_hashes(expected_target_source_hashes)
    except (FlatSegmentError, TypeError, ValueError):
        reasons.append("expected_target_source_hashes_invalid")
    owner_provider = getattr(owner, "ast_native_flat_transaction_contract", None)
    if not callable(owner_provider):
        reasons.append("owner_native_flat_capability_unavailable")
    else:
        try:
            owner_contract = _strict_native_owner_contract(owner_provider())
        except Exception:  # noqa: BLE001 - discovery fails closed
            reasons.append("owner_native_flat_capability_invalid")
    target_provider = getattr(target_executor, "ast_native_flat_target_contract", None)
    if not callable(target_provider):
        reasons.append("target_native_flat_capability_unavailable")
    else:
        try:
            target_contract = _strict_native_target_contract(target_provider())
        except Exception:  # noqa: BLE001 - discovery fails closed
            reasons.append("target_native_flat_capability_invalid")
    if not callable(getattr(owner, "begin_native_flat", None)):
        reasons.append("owner_begin_native_flat_unavailable")
    if not callable(getattr(target_executor, "execute_native_flat_segments", None)):
        reasons.append("target_native_flat_executor_unavailable")
    if fidelity != FIDELITY_EXACT:
        reasons.append("fidelity_not_exact")
    if not isinstance(expected_backend, str) or not expected_backend:
        reasons.append("expected_backend_invalid")
    if owner_contract is not None and expected is not None:
        if owner_contract["identity"] != expected:
            reasons.append("owner_identity_mismatch")
        if owner_contract["owner_semantic_contract_sha256"] != expected_owner_digest:
            reasons.append("owner_semantic_contract_mismatch")
        if owner_contract["state_contract_fingerprints"] != expected_fingerprints:
            reasons.append("owner_state_contract_fingerprints_mismatch")
        if owner_contract["source_hashes"] != expected_owner_sources:
            reasons.append("owner_source_hashes_mismatch")
    if target_contract is not None and expected is not None:
        if target_contract["identity"] != expected:
            reasons.append("target_identity_mismatch")
        if target_contract["source_hashes"] != expected_target_sources:
            reasons.append("target_source_hashes_mismatch")
    effective_limits = None
    if owner_contract is not None and target_contract is not None:
        effective_limits = {
            key: min(
                owner_contract["hard_limits"][key],
                target_contract["hard_limits"][key],
            )
            for key in _HARD_LIMIT_FIELDS
        }
        actuals = {
            "max_lanes": descriptor.lane_count,
            "max_segment_rows": descriptor.max_segment_rows,
            "max_total_rows": descriptor.logical_rows,
        }
        if any(actuals[key] > effective_limits[key] for key in _HARD_LIMIT_FIELDS):
            reasons.append("descriptor_exceeds_hard_limits")
    pair_validator = getattr(target_executor, "validate_native_flat_owner", None)
    if not callable(pair_validator):
        reasons.append("target_native_owner_compatibility_unavailable")
    else:
        try:
            pair_validator(owner, descriptor)
        except Exception:  # noqa: BLE001 - pair discovery fails closed
            reasons.append("target_native_owner_compatibility_invalid")
    provider = getattr(target_executor, "ast_native_flat_device_attestation", None)
    try:
        attestation = _strict_device_attestation(
            provider() if callable(provider) else None,
            backend=expected_backend,
        )
    except Exception:  # noqa: BLE001 - device discovery fails closed
        attestation = None
    if attestation is None:
        reasons.append("device_attestation_invalid")
    elif expected is not None:
        if (
            attestation["device"] != expected["device"]
            or attestation["build"] != expected["build"]
        ):
            reasons.append("device_identity_mismatch")
        if not attestation["ready"]:
            reasons.append(f"{expected_backend}_device_unavailable")
    if expected is not None and (
        len(expected["source_revision"]) != 40
        or any(c not in "0123456789abcdef" for c in expected["source_revision"])
        or "@sha256:" not in expected["artifact"]
        or len(expected["artifact"].rsplit("@sha256:", 1)[-1]) != 64
        or any(
            c not in "0123456789abcdef"
            for c in expected["artifact"].rsplit("@sha256:", 1)[-1]
        )
    ):
        reasons.append("exact_artifact_identity_invalid")
    unique_reasons = tuple(dict.fromkeys(reasons))
    admission = NativeFlatSegmentAdmission(
        available=not unique_reasons,
        decline_reasons=unique_reasons,
        descriptor=descriptor,
        owner_contract=owner_contract,
        target_contract=target_contract,
        effective_hard_limits=effective_limits,
        device_attestation=dict(attestation) if attestation is not None else None,
    )
    if not admission.available:
        return admission
    try:
        owner_revision = owner.revision
        membership = owner.ast_flat_segment_membership_identity()
        state_identity = _sha256_string(
            owner.ast_flat_segment_state_identity_sha256(),
            "owner state identity",
        )
        execution_identity = target_executor.ast_native_flat_execution_identity()
    except Exception:  # noqa: BLE001 - authority discovery fails closed
        return replace(
            admission,
            available=False,
            decline_reasons=("native_flat_authority_identity_invalid",),
        )
    if (
        type(owner_revision) is not int
        or owner_revision < 0
        or not isinstance(membership, tuple)
        or not membership
        or any(
            not isinstance(row, tuple)
            or not row
            or any(type(value) is not int or value < 1 for value in row)
            for row in membership
        )
        or not isinstance(execution_identity, tuple)
        or len(execution_identity) != 2
        or type(execution_identity[0]) is not int
        or execution_identity[0] < 1
        or not isinstance(execution_identity[1], tuple)
        or not execution_identity[1]
        or any(type(value) is not int or value < 0 for value in execution_identity[1])
    ):
        return replace(
            admission,
            available=False,
            decline_reasons=("native_flat_authority_identity_invalid",),
        )
    return replace(
        admission,
        _authority_ticket=_BoundNativeFlatAdmissionTicket(
            seal=_NATIVE_ADMISSION_SEAL,
            owner_id=id(owner),
            target_id=id(target_executor),
            owner_revision=owner_revision,
            owner_membership_identity=membership,
            owner_state_identity_sha256=state_identity,
            target_execution_identity=execution_identity,
            device_attestation=_device_attestation_identity(attestation),
            admission_manifest_sha256=admission.receipt()["manifest_sha256"],
        ),
    )


def validate_native_flat_segment_admission(
    admission,
    owner,
    target_executor,
    descriptor: FlatSegmentDescriptor,
    *,
    expected_backend: str,
) -> None:
    """Revalidate every native authority before opening owner state."""

    if (
        type(admission) is not NativeFlatSegmentAdmission
        or not admission.available
        or admission.descriptor != descriptor
    ):
        raise FlatSegmentUnsupported("native-flat admission ticket is unavailable")
    ticket = admission._authority_ticket
    try:
        live_attestation = _strict_device_attestation(
            target_executor.ast_native_flat_device_attestation(),
            backend=expected_backend,
        )
        owner_contract = _strict_native_owner_contract(
            owner.ast_native_flat_transaction_contract()
        )
        target_contract = _strict_native_target_contract(
            target_executor.ast_native_flat_target_contract()
        )
    except Exception as error:
        raise FlatSegmentUnsupported(
            "native-flat capability or device attestation changed after review"
        ) from error
    if (
        type(ticket) is not _BoundNativeFlatAdmissionTicket
        or ticket.seal is not _NATIVE_ADMISSION_SEAL
        or ticket.owner_id != id(owner)
        or ticket.target_id != id(target_executor)
        or ticket.owner_revision != getattr(owner, "revision", None)
        or ticket.owner_membership_identity
        != owner.ast_flat_segment_membership_identity()
        or ticket.owner_state_identity_sha256
        != owner.ast_flat_segment_state_identity_sha256()
        or ticket.target_execution_identity
        != target_executor.ast_native_flat_execution_identity()
        or ticket.admission_manifest_sha256
        != admission.receipt()["manifest_sha256"]
        or owner_contract != admission.owner_contract
        or target_contract != admission.target_contract
        or admission.device_attestation is None
        or ticket.device_attestation
        != _device_attestation_identity(admission.device_attestation)
        or ticket.device_attestation != _device_attestation_identity(live_attestation)
        or ticket.device_attestation[1] != expected_backend
        or ticket.device_attestation[4] is not True
    ):
        raise FlatSegmentUnsupported("native-flat admission ticket is stale or unbound")


@dataclass(frozen=True, slots=True)
class NativeFlatSegmentExecution:
    """One native physical invocation over exactly the descriptor's rows."""

    outputs: tuple[object, ...]
    geometry_sha256: str
    physical_rows: int
    invocation_count: int = 1


def validate_native_flat_segment_execution(
    execution: object,
    descriptor: FlatSegmentDescriptor,
) -> NativeFlatSegmentExecution:
    if type(execution) is not NativeFlatSegmentExecution:
        raise FlatSegmentError("native-flat target returned an unbound execution")
    if (
        type(execution.invocation_count) is not int
        or execution.invocation_count != 1
    ):
        raise FlatSegmentError("native-flat target must use one outer invocation")
    if type(execution.physical_rows) is not int or (
        execution.physical_rows != descriptor.logical_rows
    ):
        raise FlatSegmentError("native-flat physical rows differ from logical rows")
    if execution.geometry_sha256 != descriptor.geometry_sha256:
        raise FlatSegmentError("native-flat execution geometry differs")
    if len(execution.outputs) != descriptor.logical_rows:
        raise FlatSegmentError("native-flat output rows differ")
    return execution


def _query_flat_segment_admission(
    owner,
    target_executor,
    descriptor: FlatSegmentDescriptor,
    *,
    expected_identity: Mapping[str, str],
    expected_owner_semantic_contract_sha256: str,
    expected_state_contract_fingerprints: Sequence[str],
    expected_owner_source_hashes: Mapping[str, str],
    expected_target_source_hashes: Mapping[str, str],
    fidelity: str = FIDELITY_EXACT,
    execution_scope: str = EXECUTION_SCOPE,
    owner_protocol: str = OWNER_PROTOCOL,
    target_protocol: str = TARGET_PROTOCOL,
) -> FlatSegmentAdmission:
    """Intersect state-owner and target-tensor attestations, failing closed."""

    if not isinstance(descriptor, FlatSegmentDescriptor):
        raise FlatSegmentError("flat admission requires a validated descriptor")
    reasons = []
    owner_contract = target_contract = None
    expected = None
    expected_owner_digest = None
    expected_fingerprints = None
    expected_owner_sources = expected_target_sources = None
    try:
        expected = _identity(expected_identity)
    except (FlatSegmentError, TypeError, ValueError):
        reasons.append("expected_identity_invalid")
    try:
        expected_owner_digest = _sha256_string(
            expected_owner_semantic_contract_sha256,
            "expected owner semantic contract digest",
        )
    except (FlatSegmentError, TypeError, ValueError):
        reasons.append("expected_owner_semantic_contract_invalid")
    try:
        expected_fingerprints = _fingerprints(expected_state_contract_fingerprints)
    except (FlatSegmentError, TypeError, ValueError):
        reasons.append("expected_state_contract_fingerprints_invalid")
    try:
        expected_owner_sources = _source_hashes(expected_owner_source_hashes)
    except (FlatSegmentError, TypeError, ValueError):
        reasons.append("expected_owner_source_hashes_invalid")
    try:
        expected_target_sources = _source_hashes(expected_target_source_hashes)
    except (FlatSegmentError, TypeError, ValueError):
        reasons.append("expected_target_source_hashes_invalid")
    owner_provider = getattr(owner, "ast_flat_segment_transaction_contract", None)
    if not callable(owner_provider):
        reasons.append("owner_capability_unavailable")
    else:
        try:
            owner_contract = _strict_owner_contract(
                owner_provider(),
                execution_scope=execution_scope,
                protocol=owner_protocol,
            )
        except Exception:  # noqa: BLE001 - discovery must fail closed
            reasons.append("owner_capability_invalid")
    target_provider = getattr(target_executor, "ast_flat_segment_target_contract", None)
    if not callable(target_provider):
        reasons.append("target_capability_unavailable")
    else:
        try:
            target_contract = _strict_target_contract(
                target_provider(),
                execution_scope=execution_scope,
                protocol=target_protocol,
            )
        except Exception:  # noqa: BLE001 - discovery must fail closed
            reasons.append("target_capability_invalid")
    if not callable(getattr(owner, "begin_flat", None)):
        reasons.append("owner_begin_flat_unavailable")
    if not callable(getattr(target_executor, "execute_flat_segments", None)):
        reasons.append("target_executor_unavailable")
    if fidelity != FIDELITY_EXACT:
        reasons.append("fidelity_not_exact")
    if (
        target_contract is not None
        and expected is not None
        and target_contract["identity"] != expected
    ):
        reasons.append("target_identity_mismatch")
    if (
        owner_contract is not None
        and expected is not None
        and owner_contract["identity"] != expected
    ):
        reasons.append("owner_identity_mismatch")
    if (
        owner_contract is not None
        and expected_owner_digest is not None
        and owner_contract["owner_semantic_contract_sha256"] != expected_owner_digest
    ):
        reasons.append("owner_semantic_contract_mismatch")
    if (
        owner_contract is not None
        and expected_fingerprints is not None
        and owner_contract["state_contract_fingerprints"] != expected_fingerprints
    ):
        reasons.append("owner_state_contract_fingerprints_mismatch")
    if (
        owner_contract is not None
        and expected_owner_sources is not None
        and owner_contract["source_hashes"] != expected_owner_sources
    ):
        reasons.append("owner_source_hashes_mismatch")
    if (
        target_contract is not None
        and expected_target_sources is not None
        and target_contract["source_hashes"] != expected_target_sources
    ):
        reasons.append("target_source_hashes_mismatch")
    effective_limits = None
    if owner_contract is not None and target_contract is not None:
        effective_limits = {
            key: min(
                owner_contract["hard_limits"][key],
                target_contract["hard_limits"][key],
            )
            for key in _HARD_LIMIT_FIELDS
        }
        actuals = {
            "max_lanes": descriptor.lane_count,
            "max_segment_rows": descriptor.max_segment_rows,
            "max_total_rows": descriptor.logical_rows,
        }
        if any(actuals[key] > effective_limits[key] for key in _HARD_LIMIT_FIELDS):
            reasons.append("descriptor_exceeds_hard_limits")
    return FlatSegmentAdmission(
        available=not reasons,
        decline_reasons=tuple(dict.fromkeys(reasons)),
        descriptor=descriptor,
        owner_contract=owner_contract,
        target_contract=target_contract,
        effective_hard_limits=effective_limits,
        execution_scope=execution_scope,
    )


def query_flat_segment_admission(
    owner,
    target_executor,
    descriptor: FlatSegmentDescriptor,
    *,
    expected_identity: Mapping[str, str],
    expected_owner_semantic_contract_sha256: str,
    expected_state_contract_fingerprints: Sequence[str],
    expected_owner_source_hashes: Mapping[str, str],
    expected_target_source_hashes: Mapping[str, str],
    fidelity: str = FIDELITY_EXACT,
) -> FlatSegmentAdmission:
    """Intersect the original synthetic state and tensor attestations."""

    return _query_flat_segment_admission(
        owner,
        target_executor,
        descriptor,
        expected_identity=expected_identity,
        expected_owner_semantic_contract_sha256=(
            expected_owner_semantic_contract_sha256
        ),
        expected_state_contract_fingerprints=expected_state_contract_fingerprints,
        expected_owner_source_hashes=expected_owner_source_hashes,
        expected_target_source_hashes=expected_target_source_hashes,
        fidelity=fidelity,
    )


def _query_bound_flat_segment_admission(
    owner,
    target_executor,
    descriptor: FlatSegmentDescriptor,
    *,
    expected_identity: Mapping[str, str],
    expected_owner_semantic_contract_sha256: str,
    expected_state_contract_fingerprints: Sequence[str],
    expected_owner_source_hashes: Mapping[str, str],
    expected_target_source_hashes: Mapping[str, str],
    fidelity: str = FIDELITY_EXACT,
    execution_scope: str,
    owner_protocol: str,
    target_protocol: str,
    expected_backend: str,
) -> FlatSegmentAdmission:
    """Intersect one bound real adapter/owner contract."""

    admission = _query_flat_segment_admission(
        owner,
        target_executor,
        descriptor,
        expected_identity=expected_identity,
        expected_owner_semantic_contract_sha256=(
            expected_owner_semantic_contract_sha256
        ),
        expected_state_contract_fingerprints=expected_state_contract_fingerprints,
        expected_owner_source_hashes=expected_owner_source_hashes,
        expected_target_source_hashes=expected_target_source_hashes,
        fidelity=fidelity,
        execution_scope=execution_scope,
        owner_protocol=owner_protocol,
        target_protocol=target_protocol,
    )
    reasons = list(admission.decline_reasons)
    pair_validator = getattr(target_executor, "validate_flat_owner", None)
    if not callable(pair_validator):
        reasons.append("target_owner_compatibility_unavailable")
    else:
        try:
            pair_validator(owner, descriptor)
        except Exception:  # noqa: BLE001 - compatibility discovery fails closed
            reasons.append("target_owner_compatibility_invalid")
    expected = None
    try:
        expected = _identity(expected_identity)
    except (FlatSegmentError, TypeError, ValueError):
        pass
    if expected is not None and (
        len(expected["source_revision"]) != 40
        or any(
            character not in "0123456789abcdef"
            for character in expected["source_revision"]
        )
        or "@sha256:" not in expected["artifact"]
        or len(expected["artifact"].rsplit("@sha256:", 1)[-1]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected["artifact"].rsplit("@sha256:", 1)[-1]
        )
    ):
        reasons.append("exact_artifact_identity_invalid")
    provider = getattr(
        target_executor,
        "ast_flat_segment_device_attestation",
        None,
    )
    try:
        attestation = _strict_device_attestation(
            provider() if callable(provider) else None,
            backend=expected_backend,
        )
    except Exception:  # noqa: BLE001 - device discovery fails closed
        attestation = None
    if attestation is None:
        reasons.append("device_attestation_invalid")
    if attestation is not None and expected is not None and (
        attestation["device"] != expected["device"]
        or attestation["build"] != expected["build"]
    ):
        reasons.append("device_identity_mismatch")
    if attestation is not None and not attestation["ready"]:
        reasons.append(f"{expected_backend}_device_unavailable")
    unique_reasons = tuple(dict.fromkeys(reasons))
    if unique_reasons != admission.decline_reasons:
        admission = replace(
            admission,
            available=False,
            decline_reasons=unique_reasons,
            device_attestation=dict(attestation) if attestation is not None else None,
        )
    elif attestation is not None:
        admission = replace(admission, device_attestation=dict(attestation))
    if not admission.available:
        return admission
    owner_revision = getattr(owner, "revision", None)
    if type(owner_revision) is not int or owner_revision < 0:
        return replace(
            admission,
            available=False,
            decline_reasons=("owner_revision_invalid",),
        )
    membership_provider = getattr(owner, "ast_flat_segment_membership_identity", None)
    try:
        membership = membership_provider() if callable(membership_provider) else None
    except Exception:  # noqa: BLE001 - identity discovery fails closed
        membership = None
    if (
        not isinstance(membership, tuple)
        or not membership
        or any(
            not isinstance(row, tuple)
            or not row
            or any(type(value) is not int or value < 1 for value in row)
            for row in membership
        )
    ):
        return replace(
            admission,
            available=False,
            decline_reasons=("owner_membership_identity_invalid",),
        )
    state_provider = getattr(owner, "ast_flat_segment_state_identity_sha256", None)
    try:
        state_identity = state_provider() if callable(state_provider) else None
        state_identity = _sha256_string(
            state_identity,
            "owner state identity",
        )
    except Exception:  # noqa: BLE001 - identity discovery fails closed
        state_identity = None
    if state_identity is None:
        return replace(
            admission,
            available=False,
            decline_reasons=("owner_state_identity_invalid",),
        )
    execution_provider = getattr(
        target_executor,
        "ast_flat_segment_execution_identity",
        None,
    )
    try:
        execution_identity = (
            execution_provider() if callable(execution_provider) else None
        )
    except Exception:  # noqa: BLE001 - identity discovery fails closed
        execution_identity = None
    if (
        not isinstance(execution_identity, tuple)
        or len(execution_identity) != 2
        or type(execution_identity[0]) is not int
        or execution_identity[0] < 1
        or not isinstance(execution_identity[1], tuple)
        or not execution_identity[1]
        or any(type(value) is not int or value < 0 for value in execution_identity[1])
    ):
        return replace(
            admission,
            available=False,
            decline_reasons=("target_execution_identity_invalid",),
        )
    return replace(
        admission,
        _authority_ticket=_BoundFlatAdmissionTicket(
            seal=_REAL_ADMISSION_SEAL,
            owner_id=id(owner),
            target_id=id(target_executor),
            owner_revision=owner_revision,
            owner_membership_identity=membership,
            owner_state_identity_sha256=state_identity,
            target_execution_identity=execution_identity,
            device_attestation=_device_attestation_identity(attestation),
            admission_manifest_sha256=admission.receipt()["manifest_sha256"],
        ),
    )


def query_real_flat_segment_admission(
    owner,
    target_executor,
    descriptor: FlatSegmentDescriptor,
    *,
    expected_identity: Mapping[str, str],
    expected_owner_semantic_contract_sha256: str,
    expected_state_contract_fingerprints: Sequence[str],
    expected_owner_source_hashes: Mapping[str, str],
    expected_target_source_hashes: Mapping[str, str],
    fidelity: str = FIDELITY_EXACT,
) -> FlatSegmentAdmission:
    """Intersect the distinct direct CPU adapter/owner contract."""

    return _query_bound_flat_segment_admission(
        owner,
        target_executor,
        descriptor,
        expected_identity=expected_identity,
        expected_owner_semantic_contract_sha256=(
            expected_owner_semantic_contract_sha256
        ),
        expected_state_contract_fingerprints=expected_state_contract_fingerprints,
        expected_owner_source_hashes=expected_owner_source_hashes,
        expected_target_source_hashes=expected_target_source_hashes,
        fidelity=fidelity,
        execution_scope=REAL_EXECUTION_SCOPE,
        owner_protocol=REAL_OWNER_PROTOCOL,
        target_protocol=REAL_TARGET_PROTOCOL,
        expected_backend="cpu",
    )


def query_metal_flat_segment_admission(
    owner,
    target_executor,
    descriptor: FlatSegmentDescriptor,
    *,
    expected_identity: Mapping[str, str],
    expected_owner_semantic_contract_sha256: str,
    expected_state_contract_fingerprints: Sequence[str],
    expected_owner_source_hashes: Mapping[str, str],
    expected_target_source_hashes: Mapping[str, str],
    fidelity: str = FIDELITY_EXACT,
) -> FlatSegmentAdmission:
    """Intersect the direct exact-artifact Metal probe contract."""

    return _query_bound_flat_segment_admission(
        owner,
        target_executor,
        descriptor,
        expected_identity=expected_identity,
        expected_owner_semantic_contract_sha256=(
            expected_owner_semantic_contract_sha256
        ),
        expected_state_contract_fingerprints=expected_state_contract_fingerprints,
        expected_owner_source_hashes=expected_owner_source_hashes,
        expected_target_source_hashes=expected_target_source_hashes,
        fidelity=fidelity,
        execution_scope=METAL_EXECUTION_SCOPE,
        owner_protocol=METAL_OWNER_PROTOCOL,
        target_protocol=METAL_TARGET_PROTOCOL,
        expected_backend="metal",
    )


def _validate_bound_flat_segment_admission(
    admission,
    owner,
    target_executor,
    descriptor: FlatSegmentDescriptor,
    *,
    execution_scope: str,
    owner_protocol: str,
    target_protocol: str,
    expected_backend: str,
) -> None:
    """Require a live externally-bound admission before state begins."""

    if (
        type(admission) is not FlatSegmentAdmission
        or not admission.available
        or admission.execution_scope != execution_scope
        or admission.descriptor != descriptor
    ):
        raise FlatSegmentUnsupported("real flat admission ticket is unavailable")
    ticket = admission._authority_ticket
    try:
        live_attestation = _strict_device_attestation(
            target_executor.ast_flat_segment_device_attestation(),
            backend=expected_backend,
        )
    except Exception as error:
        raise FlatSegmentUnsupported(
            "real flat admission device attestation changed after review"
        ) from error
    if (
        type(ticket) is not _BoundFlatAdmissionTicket
        or ticket.seal is not _REAL_ADMISSION_SEAL
        or ticket.owner_id != id(owner)
        or ticket.target_id != id(target_executor)
        or ticket.owner_revision != getattr(owner, "revision", None)
        or not callable(
            getattr(owner, "ast_flat_segment_membership_identity", None)
        )
        or ticket.owner_membership_identity
        != owner.ast_flat_segment_membership_identity()
        or not callable(
            getattr(owner, "ast_flat_segment_state_identity_sha256", None)
        )
        or ticket.owner_state_identity_sha256
        != owner.ast_flat_segment_state_identity_sha256()
        or not callable(
            getattr(target_executor, "ast_flat_segment_execution_identity", None)
        )
        or ticket.target_execution_identity
        != target_executor.ast_flat_segment_execution_identity()
        or ticket.admission_manifest_sha256
        != admission.receipt()["manifest_sha256"]
    ):
        raise FlatSegmentUnsupported("real flat admission ticket is stale or unbound")
    if (
        admission.device_attestation is None
        or ticket.device_attestation
        != _device_attestation_identity(admission.device_attestation)
        or ticket.device_attestation != _device_attestation_identity(live_attestation)
        or ticket.device_attestation[1] != expected_backend
        or ticket.device_attestation[4] is not True
    ):
        raise FlatSegmentUnsupported(
            "real flat admission device attestation changed after review"
        )
    try:
        owner_contract = _strict_owner_contract(
            owner.ast_flat_segment_transaction_contract(),
            execution_scope=execution_scope,
            protocol=owner_protocol,
        )
        target_contract = _strict_target_contract(
            target_executor.ast_flat_segment_target_contract(),
            execution_scope=execution_scope,
            protocol=target_protocol,
        )
    except Exception as error:
        raise FlatSegmentUnsupported(
            "real flat admission capability changed after review"
        ) from error
    if (
        owner_contract != admission.owner_contract
        or target_contract != admission.target_contract
    ):
        raise FlatSegmentUnsupported(
            "real flat admission capability changed after review"
        )


def validate_real_flat_segment_admission(
    admission,
    owner,
    target_executor,
    descriptor: FlatSegmentDescriptor,
) -> None:
    _validate_bound_flat_segment_admission(
        admission,
        owner,
        target_executor,
        descriptor,
        execution_scope=REAL_EXECUTION_SCOPE,
        owner_protocol=REAL_OWNER_PROTOCOL,
        target_protocol=REAL_TARGET_PROTOCOL,
        expected_backend="cpu",
    )


def validate_metal_flat_segment_admission(
    admission,
    owner,
    target_executor,
    descriptor: FlatSegmentDescriptor,
) -> None:
    _validate_bound_flat_segment_admission(
        admission,
        owner,
        target_executor,
        descriptor,
        execution_scope=METAL_EXECUTION_SCOPE,
        owner_protocol=METAL_OWNER_PROTOCOL,
        target_protocol=METAL_TARGET_PROTOCOL,
        expected_backend="metal",
    )


@dataclass(frozen=True, slots=True)
class FlatSegmentExecution:
    """One target invocation's flat outputs in descriptor order."""

    outputs: tuple[object, ...]
    geometry_sha256: str
    invocation_count: int = 1


@dataclass(frozen=True, slots=True)
class PreparedFlatSegmentCommit:
    """Private candidate state that has not crossed the publication boundary."""

    state: object
    accepted_lengths: tuple[int, ...]
    geometry_sha256: str


@dataclass(frozen=True, slots=True)
class FlatSegmentRoundResult:
    outputs: tuple[object, ...]
    segment_outputs: tuple[tuple[object, ...], ...]
    consumed_lengths: tuple[int, ...]
    committed_state: object
    admission: dict
    target_invocations: int

    def receipt(self) -> dict:
        return {
            "schema": EXECUTION_SCHEMA,
            "layout_class": LAYOUT_CLASS,
            "execution_scope": EXECUTION_SCOPE,
            "logical_rows": len(self.outputs),
            "consumed_lengths": list(self.consumed_lengths),
            "target_invocations": self.target_invocations,
            "ordinary_serial_oracle": (
                "exact_outputs_prepublication_and_returned_published_state"
            ),
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": True,
            "admission_manifest_sha256": self.admission["manifest_sha256"],
        }


def execute_synthetic_flat_segment_round(
    owner,
    target_executor,
    descriptor: FlatSegmentDescriptor,
    flat_inputs: Sequence[object],
    accepted_draft_lengths: Iterable[int],
    ordinary_serial: Callable[[int, tuple[object, ...]], Sequence[object]],
    ordinary_committed_state: Callable[[tuple[int, ...]], object],
    *,
    expected_identity: Mapping[str, str],
    expected_owner_semantic_contract_sha256: str,
    expected_state_contract_fingerprints: Sequence[str],
    expected_owner_source_hashes: Mapping[str, str],
    expected_target_source_hashes: Mapping[str, str],
) -> FlatSegmentRoundResult:
    """Execute one synthetic flat target call and compare with serial B1.

    This helper is intentionally unusable by production capabilities: both
    attested contracts must declare ``synthetic_cpu_contract_only``.  A future
    real executor requires a new reviewed protocol and serving integration.
    """

    admission = query_flat_segment_admission(
        owner,
        target_executor,
        descriptor,
        expected_identity=expected_identity,
        expected_owner_semantic_contract_sha256=(
            expected_owner_semantic_contract_sha256
        ),
        expected_state_contract_fingerprints=expected_state_contract_fingerprints,
        expected_owner_source_hashes=expected_owner_source_hashes,
        expected_target_source_hashes=expected_target_source_hashes,
    )
    admission_receipt = admission.receipt()
    if not admission.available:
        raise FlatSegmentUnsupported(
            "flat segment admission declined: " + ",".join(admission.decline_reasons)
        )
    values = tuple(flat_inputs)
    if len(values) != descriptor.logical_rows:
        raise FlatSegmentError("flat inputs do not cover every logical row")
    accepted = descriptor.validate_accepted_drafts(accepted_draft_lengths)
    consumed = tuple(value + 1 for value in accepted)
    expected_segments = tuple(
        tuple(ordinary_serial(lane, values[descriptor.lane_slice(lane)]))
        for lane in range(descriptor.lane_count)
    )
    if any(
        len(outputs) != length
        for outputs, length in zip(
            expected_segments, descriptor.query_lengths, strict=True
        )
    ):
        raise FlatSegmentError("ordinary serial oracle returned the wrong row count")
    expected_outputs = tuple(
        value for segment in expected_segments for value in segment
    )
    expected_committed_state = ordinary_committed_state(consumed)

    transaction = None
    try:
        transaction = owner.begin_flat(descriptor)
        if any(
            not callable(getattr(transaction, name, None))
            for name in ("prepare_commit", "publish_commit", "abort")
        ):
            raise FlatSegmentError("flat transaction lacks prepare, publish, or abort")
        execution = target_executor.execute_flat_segments(
            values, descriptor, transaction
        )
        if not isinstance(execution, FlatSegmentExecution):
            raise FlatSegmentError("flat target returned an unbound execution result")
        if (
            type(execution.invocation_count) is not int
            or execution.invocation_count != 1
        ):
            raise FlatSegmentError("flat target must report exactly one invocation")
        if execution.geometry_sha256 != descriptor.geometry_sha256:
            raise FlatSegmentError("flat target result geometry differs")
        outputs = tuple(execution.outputs)
        if len(outputs) != descriptor.logical_rows:
            raise FlatSegmentError("flat target output row count differs")
        if outputs != expected_outputs:
            raise FlatSegmentError(
                "flat target differs from the ordinary serial oracle"
            )
        prepared = transaction.prepare_commit(accepted_lengths=consumed)
        if not isinstance(prepared, PreparedFlatSegmentCommit):
            raise FlatSegmentError(
                "flat transaction returned an unbound prepared commit"
            )
        if (
            prepared.accepted_lengths != consumed
            or prepared.geometry_sha256 != descriptor.geometry_sha256
        ):
            raise FlatSegmentError("flat prepared commit binding differs")
        if prepared.state != expected_committed_state:
            raise FlatSegmentError(
                "flat prepared state differs from the ordinary accepted-prefix oracle"
            )
        committed = transaction.publish_commit(prepared)
        if committed != expected_committed_state:
            raise FlatSegmentError(
                "flat published state differs from the ordinary accepted-prefix oracle"
            )
    except BaseException as error:
        if transaction is not None:
            try:
                transaction.abort()
            except BaseException as abort_error:  # noqa: BLE001 - preserve root cause
                try:
                    error.add_note(
                        f"flat transaction abort also failed: {abort_error!r}"
                    )
                except AttributeError:  # pragma: no cover - pre-3.11 compatibility
                    pass
        raise
    segments = tuple(
        outputs[descriptor.lane_slice(lane)] for lane in range(descriptor.lane_count)
    )
    return FlatSegmentRoundResult(
        outputs=outputs,
        segment_outputs=segments,
        consumed_lengths=consumed,
        committed_state=committed,
        admission=admission_receipt,
        target_invocations=execution.invocation_count,
    )
