"""Immutable declarations for exact cache operations.

Consumers must not infer state semantics from a cache class name or from the
presence of a ``state`` attribute.  A cache family declares its exact planes,
rollback mode and persistence identity here.  Existing built-in cache classes
are admitted by exact type; subclasses and new families must redeclare the
contract in their own class body.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


_MISSING = object()


class CacheOperationsUnsupported(RuntimeError):
    """A selected cache capability has no exact provider declaration."""


@dataclass(frozen=True, slots=True)
class CachePlaneDeclaration:
    kind: str
    fields: tuple[str, ...]
    compatibility_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.kind or not self.fields:
            raise ValueError("cache plane kind and fields must be non-empty")


@dataclass(frozen=True, slots=True)
class CheckpointPublicationCondition:
    """One provider-declared reason a cache is not a committed boundary."""

    field: str
    active_when: str
    reason: str

    def __post_init__(self) -> None:
        if not self.field or not self.reason:
            raise ValueError("checkpoint publication conditions must be non-empty")
        if self.active_when not in {"truthy", "not_none"}:
            raise ValueError(
                "checkpoint publication condition must use truthy or not_none"
            )

    def refusal(self, cache: Any) -> str | None:
        value = _field(cache, self.field)
        active = bool(value) if self.active_when == "truthy" else value is not None
        return self.reason if active else None


@dataclass(frozen=True, slots=True)
class CacheOperationsContract:
    """Exact state identity and operations for one cache family."""

    family: str
    topology: str
    state_revision: str
    cache_layout: str
    numerics: str
    planes: tuple[CachePlaneDeclaration, ...]
    exact_prefix: bool
    restores_at_checkpoints: bool
    rollback: str | None
    evaluation_fields: tuple[str, ...] = ("state",)
    offset_field: str | None = "offset"
    publication_conditions: tuple[CheckpointPublicationCondition, ...] = ()
    segmented_publication_conditions: tuple[CheckpointPublicationCondition, ...] = ()
    schema: str = "mlx2.cache-operations.v1"

    def __post_init__(self) -> None:
        for name in (
            self.family,
            self.topology,
            self.state_revision,
            self.cache_layout,
            self.numerics,
        ):
            if not isinstance(name, str) or not name:
                raise ValueError("cache operation identities must be non-empty")
        if not self.planes:
            raise ValueError("cache operations must declare at least one plane")
        if self.rollback not in {None, "kv_trim", "recurrent_record_trim"}:
            raise ValueError(f"unknown cache rollback operation: {self.rollback!r}")

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema": self.schema,
            "family": self.family,
            "topology": self.topology,
            "state_revision": self.state_revision,
            "cache_layout": self.cache_layout,
            "numerics": self.numerics,
            "planes": [
                [plane.kind, list(plane.fields), list(plane.compatibility_fields)]
                for plane in self.planes
            ],
            "exact_prefix": self.exact_prefix,
            "restores_at_checkpoints": self.restores_at_checkpoints,
            "rollback": self.rollback,
            "evaluation_fields": list(self.evaluation_fields),
            "offset_field": self.offset_field,
            "publication_conditions": [
                [condition.field, condition.active_when, condition.reason]
                for condition in self.publication_conditions
            ],
            "segmented_publication_conditions": [
                [condition.field, condition.active_when, condition.reason]
                for condition in self.segmented_publication_conditions
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def projection(self, cache: Any, plane: CachePlaneDeclaration) -> tuple[Any, ...]:
        if plane not in self.planes:
            raise CacheOperationsUnsupported("cache plane is not declared by provider")
        return tuple(_field(cache, field) for field in plane.fields)

    def compatibility(
        self, cache: Any, plane: CachePlaneDeclaration
    ) -> tuple[tuple[str, str], ...]:
        if plane not in self.planes:
            raise CacheOperationsUnsupported("cache plane is not declared by provider")
        return tuple(
            (field, repr(_field(cache, field)))
            for field in plane.compatibility_fields
        )

    def evaluation_values(self, cache: Any) -> tuple[Any, ...]:
        return tuple(_field(cache, field) for field in self.evaluation_fields)

    def position(self, cache: Any) -> int | None:
        if self.offset_field is None:
            return None
        value = _field(cache, self.offset_field)
        return value if type(value) is int else None

    def publication_refusal(self, cache: Any) -> str | None:
        for condition in self.publication_conditions:
            reason = condition.refusal(cache)
            if reason is not None:
                return reason
        return None

    def validate_publication(self, cache: Any) -> None:
        reason = self.publication_refusal(cache)
        if reason is not None:
            raise CacheOperationsUnsupported(reason)

    def validate_segmented_publication(self, cache: Any) -> None:
        for condition in self.segmented_publication_conditions:
            reason = condition.refusal(cache)
            if reason is not None:
                raise CacheOperationsUnsupported(reason)

    def capture_rollback(self, cache: Any, *, skip_rotating: bool = False) -> tuple:
        """Capture provider-declared PLD recovery state without class inference."""
        if self.rollback == "recurrent_record_trim":
            if not cache.is_trimmable():
                raise RuntimeError("recurrent rollback epoch is not active")
            payload = ("recurrent_record_trim", cache.rollback_marker())
        elif self.rollback != "kv_trim":
            raise CacheOperationsUnsupported(
                f"{self.family} does not declare speculative rollback"
            )
        elif self.family == "rotating_kv":
            if skip_rotating:
                payload = ("replay",)
            else:
                import mlx.core as mx

                payload = (
                    "rotating_kv",
                    None if cache.keys is None else mx.array(cache.keys),
                    None if cache.values is None else mx.array(cache.values),
                    cache.offset,
                    getattr(cache, "_idx", None),
                )
        else:
            payload = ("kv_trim", self.position(cache))
        # PLD rollback is an in-place transaction, not a portable checkpoint.
        # Keep the original owner alive and bind the payload to both that exact
        # object and this immutable provider revision/layout fingerprint.
        return ("mlx2.rollback.v1", self.fingerprint, cache, payload)

    def restore_rollback(self, cache: Any, snapshot: tuple) -> None:
        payload = self.validate_rollback_snapshot(snapshot, cache=cache)
        operation = payload[0]
        if operation == "recurrent_record_trim":
            cache.rewind_to_rollback_marker(payload[1])
            return
        if operation == "kv_trim":
            position = self.position(cache)
            if position is None or payload[1] is None:
                raise CacheOperationsUnsupported("KV rollback has no exact position")
            cache.trim(position - int(payload[1]))
            return
        if operation == "replay":
            return
        if operation == "rotating_kv":
            _, keys, values, offset, index = payload
            cache.keys, cache.values, cache.offset = keys, values, offset
            if index is not None:
                cache._idx = index
            return
        raise CacheOperationsUnsupported(
            f"rollback snapshot {operation!r} is not declared by provider"
        )

    def validate_rollback_snapshot(
        self, snapshot: tuple, *, cache: Any | None = None
    ) -> tuple:
        if (
            not isinstance(snapshot, tuple)
            or len(snapshot) != 4
            or snapshot[0] != "mlx2.rollback.v1"
        ):
            raise CacheOperationsUnsupported("rollback snapshot is malformed")
        _, fingerprint, owner, payload = snapshot
        if fingerprint != self.fingerprint:
            raise CacheOperationsUnsupported(
                "rollback snapshot provider revision or layout changed"
            )
        if cache is not None and owner is not cache:
            raise CacheOperationsUnsupported(
                "rollback snapshot cache owner changed before restore"
            )
        if not isinstance(payload, tuple) or not payload:
            raise CacheOperationsUnsupported("rollback snapshot payload is malformed")
        operation = payload[0]
        allowed = {
            "recurrent_record_trim"
            if self.rollback == "recurrent_record_trim"
            else "kv_trim"
            if self.rollback == "kv_trim" and self.family != "rotating_kv"
            else "rotating_kv"
            if self.rollback == "kv_trim"
            else None
        }
        if self.family == "rotating_kv":
            allowed.add("replay")
        if operation not in allowed:
            raise CacheOperationsUnsupported(
                f"rollback snapshot {operation!r} does not match {self.family}"
            )
        if operation == "recurrent_record_trim":
            if len(payload) != 2:
                raise CacheOperationsUnsupported(
                    "recurrent rollback snapshot must contain exactly a marker"
                )
        elif operation == "kv_trim":
            if len(payload) != 2 or type(payload[1]) is not int:
                raise CacheOperationsUnsupported(
                    "KV rollback snapshot must contain exactly an integer position"
                )
        elif operation == "replay":
            if len(payload) != 1:
                raise CacheOperationsUnsupported("replay snapshot must be empty")
        elif operation == "rotating_kv":
            if (
                len(payload) != 5
                or type(payload[3]) is not int
                or (payload[4] is not None and type(payload[4]) is not int)
                or ((payload[1] is None) != (payload[2] is None))
            ):
                raise CacheOperationsUnsupported(
                    "rotating KV rollback snapshot has invalid shape or field types"
                )
        return payload


SPECULATION_PUBLICATION_CONDITIONS = (
    CheckpointPublicationCondition(
        "speculating", "truthy", "speculation transaction is active"
    ),
)

ROLLBACK_PUBLICATION_CONDITIONS = SPECULATION_PUBLICATION_CONDITIONS + (
    CheckpointPublicationCondition(
        "_rollbacks", "truthy", "exact rollback records are live"
    ),
)

QSA_PUBLICATION_CONDITIONS = (
    CheckpointPublicationCondition(
        "_mtp_share_topk", "truthy", "a QSA/MTP transaction is active"
    ),
    CheckpointPublicationCondition(
        "_mtp_shared_topk", "not_none", "a QSA/MTP transaction is active"
    ),
)


KV_OPERATIONS = CacheOperationsContract(
    family="kv",
    topology="kv",
    state_revision="mlx2.kv.v1",
    cache_layout="bhsd",
    numerics="stored_dtype",
    planes=(CachePlaneDeclaration("attention_kv", ("keys", "values")),),
    exact_prefix=True,
    restores_at_checkpoints=False,
    rollback="kv_trim",
)

QUANTIZED_KV_OPERATIONS = CacheOperationsContract(
    family="quantized_kv",
    topology="kv",
    state_revision="mlx2.quantized-kv.v1",
    cache_layout="packed-bhsd-plus-channel-scales",
    numerics="declared_quantized_storage",
    planes=(CachePlaneDeclaration(
        "attention_kv", ("keys", "values", "key_scale", "value_scale"),
        ("group_size", "bits", "key_bits", "value_bits", "rotate", "normalize"),
    ),),
    exact_prefix=True,
    restores_at_checkpoints=False,
    rollback="kv_trim",
)

ROTATING_KV_OPERATIONS = CacheOperationsContract(
    family="rotating_kv",
    topology="rotating_kv",
    state_revision="mlx2.rotating-kv.v1",
    cache_layout="ring-bhsd",
    numerics="stored_dtype",
    planes=(CachePlaneDeclaration("attention_ring", ("keys", "values"), ("max_size",)),),
    exact_prefix=True,
    restores_at_checkpoints=True,
    rollback="kv_trim",
    publication_conditions=ROLLBACK_PUBLICATION_CONDITIONS,
)

ROTATING_QUANTIZED_KV_OPERATIONS = CacheOperationsContract(
    family="rotating_quantized_kv",
    topology="rotating_kv",
    state_revision="mlx2.rotating-quantized-kv.v1",
    cache_layout="ring-packed-bhsd",
    numerics="declared_quantized_storage",
    planes=(CachePlaneDeclaration(
        "attention_ring", ("keys", "values"), ("max_size", "group_size", "bits")
    ),),
    exact_prefix=True,
    restores_at_checkpoints=True,
    rollback="kv_trim",
    publication_conditions=ROLLBACK_PUBLICATION_CONDITIONS,
)

SINK_WINDOW_KV_OPERATIONS = CacheOperationsContract(
    family="sink_window_kv",
    topology="rotating_kv",
    state_revision="mlx2.sink-window-kv.v1",
    cache_layout="sink-plus-window-bhsd",
    numerics="stored_dtype",
    planes=(
        CachePlaneDeclaration(
            "attention_ring",
            ("keys", "values"),
            ("window_size", "sink_size", "rollback_window"),
        ),
    ),
    exact_prefix=True,
    restores_at_checkpoints=False,
    rollback="kv_trim",
    publication_conditions=SPECULATION_PUBLICATION_CONDITIONS,
)

# mlx-vlm's rotating cache exposes a bounded ring but no checkpoint ledger or
# speculative rollback transaction.  Keep that dependency surface distinct
# from mlx2's RotatingKVCache: sharing the mlx2 declaration would claim fields
# and restore operations the vendor object does not own.
VLM_ROTATING_KV_OPERATIONS = CacheOperationsContract(
    family="vlm_rotating_kv",
    topology="rotating_kv",
    state_revision="mlx-vlm.rotating-kv.v1",
    cache_layout="vendor-ring-bhsd",
    numerics="stored_dtype",
    planes=(
        CachePlaneDeclaration(
            "attention_ring", ("keys", "values"), ("max_size",)
        ),
    ),
    exact_prefix=True,
    restores_at_checkpoints=False,
    rollback=None,
)

RECURRENT_OPERATIONS = CacheOperationsContract(
    family="recurrent",
    topology="checkpointed_hybrid",
    state_revision="mlx2.recurrent-arrays.v1",
    cache_layout="ordered-state-arrays",
    numerics="stored_dtype",
    planes=(CachePlaneDeclaration("gdn_recurrent", ("cache", "left_padding", "lengths")),),
    exact_prefix=True,
    restores_at_checkpoints=True,
    rollback="recurrent_record_trim",
    offset_field=None,
    publication_conditions=ROLLBACK_PUBLICATION_CONDITIONS,
)

QSA_KV_OPERATIONS = CacheOperationsContract(
    family="indexed_kv",
    topology="checkpointed_hybrid",
    state_revision="mlx2.qsa-kv.v1",
    cache_layout="kv-plus-index-ledger",
    numerics="stored_dtype",
    planes=(
        CachePlaneDeclaration(
            "attention_kv",
            ("keys", "values"),
        ),
        CachePlaneDeclaration(
            "qsa_summary",
            (
                "index_keys", "_qsa_pooled_keys", "_qsa_pending_pooled",
                "_qsa_pooled_ratio", "_qsa_summary_identity", "_qsa_summary_restored",
                "_mtp_share_topk", "_mtp_shared_topk", "_mtp_shared_topk_n_blocks",
            ),
        ),
    ),
    exact_prefix=True,
    restores_at_checkpoints=True,
    rollback="kv_trim",
    publication_conditions=QSA_PUBLICATION_CONDITIONS,
    segmented_publication_conditions=QSA_PUBLICATION_CONDITIONS,
)

QSA_QUANTIZED_KV_OPERATIONS = CacheOperationsContract(
    family="indexed_quantized_kv",
    topology="checkpointed_hybrid",
    state_revision="mlx2.qsa-quantized-kv.v1",
    cache_layout="packed-kv-plus-index-ledger",
    numerics="declared_quantized_storage",
    planes=(
        CachePlaneDeclaration(
            "attention_kv", ("keys", "values", "key_scale", "value_scale"),
            ("group_size", "bits", "key_bits", "value_bits", "rotate", "normalize"),
        ),
        QSA_KV_OPERATIONS.planes[1],
    ),
    exact_prefix=True,
    restores_at_checkpoints=True,
    rollback="kv_trim",
    publication_conditions=QSA_PUBLICATION_CONDITIONS,
    segmented_publication_conditions=QSA_PUBLICATION_CONDITIONS,
)

SHARED_QSA_KV_OPERATIONS = CacheOperationsContract(
    family="indexed_kv",
    topology="checkpointed_hybrid",
    state_revision="mlx2.shared-qsa-kv.v1",
    cache_layout="immutable-base-plus-private-suffix",
    numerics="stored_dtype",
    planes=(
        CachePlaneDeclaration(
            "attention_kv",
            ("base.keys", "base.values", "_kv.keys", "_kv.values"),
        ),
        CachePlaneDeclaration(
            "qsa_summary",
            (
                "base.index_keys", "base.pooled_keys", "index_keys", "_suffix_pooled_keys",
                "_mtp_share_topk", "_mtp_shared_topk", "_mtp_shared_topk_n_blocks",
            ),
            ("base.layout_id", "base.pooled_ratio", "base.summary_identity"),
        ),
    ),
    exact_prefix=True,
    restores_at_checkpoints=True,
    rollback="kv_trim",
    publication_conditions=QSA_PUBLICATION_CONDITIONS,
    segmented_publication_conditions=QSA_PUBLICATION_CONDITIONS,
)


def _field(value: Any, path: str) -> Any:
    for name in path.split("."):
        value = getattr(value, name, _MISSING)
        if value is _MISSING:
            raise CacheOperationsUnsupported(
                f"declared cache field {path!r} is absent from provider value"
            )
        if value is None:
            break
    return value


def _declared_contract(cache: Any) -> CacheOperationsContract | None:
    """Read an own-class declaration; inherited declarations are not proof."""
    value = type(cache).__dict__.get("cache_operations_contract")
    if callable(value) and not isinstance(value, CacheOperationsContract):
        value = value(cache)
    return value if isinstance(value, CacheOperationsContract) else None


def _legacy_contract(cache: Any) -> CacheOperationsContract | None:
    """Exact-type declarations for current built-in cache families."""
    from .models.cache import (
        ArraysCache,
        KVCache,
        QuantizedKVCache,
        RotatingKVCache,
        RotatingQuantizedKVCache,
        SinkWindowKVCache,
    )

    contracts = {
        KVCache: KV_OPERATIONS,
        QuantizedKVCache: QUANTIZED_KV_OPERATIONS,
        RotatingKVCache: ROTATING_KV_OPERATIONS,
        RotatingQuantizedKVCache: ROTATING_QUANTIZED_KV_OPERATIONS,
        SinkWindowKVCache: SINK_WINDOW_KV_OPERATIONS,
        ArraysCache: RECURRENT_OPERATIONS,
    }
    # mlx-vlm caches are an explicitly supported dependency surface.  Import
    # them only when installed and admit exact types, never a class name or an
    # inherited lookalike.
    try:
        from mlx_vlm.models.cache import (
            KVCache as VLMKVCache,
            RotatingKVCache as VLMRotatingKVCache,
        )
    except ImportError:
        pass
    else:
        contracts[VLMKVCache] = KV_OPERATIONS
        contracts[VLMRotatingKVCache] = VLM_ROTATING_KV_OPERATIONS
    return contracts.get(type(cache))


def cache_operations(
    cache: Any, *, capability: str, required: bool = True
) -> CacheOperationsContract | None:
    contract = _declared_contract(cache) or _legacy_contract(cache)
    if contract is None and required:
        raise CacheOperationsUnsupported(
            f"{capability} requires an explicit cache operations contract for "
            f"{type(cache).__module__}.{type(cache).__qualname__}"
        )
    return contract


def cache_evaluation_values(cache: Any, *, capability: str) -> tuple[Any, ...]:
    return cache_operations(cache, capability=capability).evaluation_values(cache)


__all__ = [
    "CacheOperationsContract",
    "CacheOperationsUnsupported",
    "CachePlaneDeclaration",
    "CheckpointPublicationCondition",
    "KV_OPERATIONS",
    "QSA_KV_OPERATIONS",
    "QSA_QUANTIZED_KV_OPERATIONS",
    "QUANTIZED_KV_OPERATIONS",
    "RECURRENT_OPERATIONS",
    "ROTATING_KV_OPERATIONS",
    "ROTATING_QUANTIZED_KV_OPERATIONS",
    "SINK_WINDOW_KV_OPERATIONS",
    "SHARED_QSA_KV_OPERATIONS",
    "VLM_ROTATING_KV_OPERATIONS",
    "cache_evaluation_values",
    "cache_operations",
]
