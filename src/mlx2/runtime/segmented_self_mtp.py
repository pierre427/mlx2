# SPDX-License-Identifier: Apache-2.0
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
import hashlib
import os
import threading
from numbers import Integral
from dataclasses import dataclass, replace
from typing import Any, Iterable
import mlx.core as mx
from .cache_branch_transaction import (
    CacheDeltaBranch,
    CacheDeltaLineage,
    PlaneBase,
    PlaneDelta,
    create_cache_delta_lineage,
)
from .cache_planes import CachePlaneKind

_ZERO = {
    "requests": 0,
    "physical_join_flag_reconciled": 0,
    "width_lock_plain_fallbacks": 0,
    "engaged": 0,
    "declined": 0,
    "failures": 0,
    "fallback_physical_b2": 0,
    "physical_b2_formations": 0,
    "b1_target_forwards": 0,
    "b1_draft_forwards": 0,
    "batched_target_forwards": 0,
    "batched_draft_forwards": 0,
    "true_batched_requests": 0,
    "true_batched_engaged": 0,
    "true_batched_declined": 0,
    "live_width_change_deferrals": 0,
    "layer_local_materializations": 0,
    "recurrent_state_materializations": 0,
    "recurrent_state_materialized_bytes": 0,
    "row_state_splits": 0,
    "segmented_attention_calls": 0,
    "private_delta_requests": 0,
    "private_delta_declines": 0,
    "private_delta_preflight_declines": 0,
    "private_delta_late_gather_fallbacks": 0,
    "private_delta_attention_calls": 0,
    "private_delta_rows": 0,
    "private_delta_base_tokens_cumulative": 0,
    "private_delta_base_tokens_last": 0,
    "private_delta_base_tokens_min": 0,
    "private_delta_base_tokens_max": 0,
    "private_delta_duplicate_base_storage_bytes_not_formed_cumulative": 0,
    "shared_qsa_rows": 0,
    "shared_qsa_base_bytes": 0,
    "shared_qsa_private_bytes": 0,
    "shared_qsa_materializations": 0,
    "shared_qsa_materialized_bytes": 0,
    "shared_qsa_batched_selections": 0,
    "shared_qsa_policy_checks": 0,
    "shared_qsa_policy_admitted": 0,
    "shared_qsa_policy_declined": 0,
    "shared_qsa_policy_context_tokens_cumulative": 0,
    "shared_qsa_policy_remaining_tokens_cumulative": 0,
    "shared_qsa_policy_cutoff_tokens_cumulative": 0,
    "exact_set_fold_requests": 0,
    "exact_set_fold_declines": 0,
    "exact_set_fold_preflight_declines": 0,
    "exact_set_fold_private_fallbacks": 0,
    "exact_set_fold_attention_calls": 0,
    "exact_set_fold_rows": 0,
    "exact_set_fold_device_proofs": 0,
    "independent_lineages_consumed": 0,
    "full_prefix_materializations": 0,
    "full_prefix_materialized_bytes": 0,
    "transaction_branches": 0,
    "transaction_promotions": 0,
    "transaction_rejections": 0,
    "transaction_canonicalizations": 0,
    "committed_cycles": 0,
    "recovery_checkpoint_captures": 0,
    "recovery_checkpoint_restores": 0,
    "recovery_checkpoint_failures": 0,
    "accepted_zero": 0,
    "accepted_partial": 0,
    "accepted_all": 0,
    "async_qsa_promotion_requests": 0,
    "async_qsa_promotion_queued": 0,
    "async_qsa_promotion_engaged": 0,
    "async_qsa_promotion_declined": 0,
    "async_qsa_promotion_declined_shared_suffix": 0,
    "async_qsa_promotion_declined_memory": 0,
    "async_qsa_admission_settled": 0,
    "async_qsa_promotion_failures": 0,
    "async_qsa_promotion_reserved_bytes": 0,
    "async_qsa_promotion_patched_bytes": 0,
    "async_qsa_promotion_wait_ns": 0,
    "async_qsa_prequeue_requests": 0,
    "async_qsa_prequeue_queued": 0,
    "async_qsa_prequeue_bound": 0,
    "async_qsa_prequeue_declined": 0,
    "async_qsa_shared_prefix_fused_layers": 0,
    "async_qsa_budget_checks": 0,
    "async_qsa_budget_retained_segmented": 0,
    "async_qsa_budget_promotions": 0,
    "async_qsa_budget_remaining_tokens_cumulative": 0,
    "async_qsa_budget_cutoff_tokens_cumulative": 0,
    "device_synchronizations": 0,
    "proposal_ns": 0,
    "commit_ns": 0,
    "zero_depth_fast_rounds": 0,
    # Approximate-KV composition: segmented layers whose rows are quantized
    # (target-only kv_q8/kv_k8v4 under compose_mtp) and their attention calls.
    "quantized_kv_segmented_layers": 0,
    "quantized_kv_segmented_attention_calls": 0,
}
for _width in range(1, 10):
    for _event in ("requests", "engaged", "declined"):
        _ZERO[f"private_delta_width_{_width}_{_event}"] = 0
for _event in ("requests", "engaged", "declined"):
    _ZERO[f"private_delta_width_other_{_event}"] = 0
_STATS = dict(_ZERO)
_STATS_LOCK = threading.Lock()
_PRIVATE_DELTA_DECLINE_REASONS: dict[str, int] = {}
_EXACT_SET_FOLD_DECLINE_REASONS: dict[str, int] = {}


def segmented_self_mtp_enabled(value: bool | None = None) -> bool:
    if value is not None:
        return bool(value)
    return os.environ.get("MLX_LM_SEGMENTED_SELF_MTP", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def segmented_self_mtp_timing_enabled() -> bool:
    return os.environ.get("MLX_LM_SEGMENTED_SELF_MTP_TIMING", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def true_batched_segmented_self_mtp_enabled(value: bool | None = None) -> bool:
    """Use one batched forward over segmented rows.

    The consumer defaults on *inside* the separately default-off segmented
    experiment.  Setting this knob to zero retains the original serial B1
    consumer as an exact control/fallback.
    """
    if value is not None:
        return bool(value)
    return os.environ.get("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def qsa_private_delta_enabled(value: bool | None = None) -> bool:
    """Use the source-phased QSA consumer inside segmented self-MTP.

    This defaults on only inside the separately default-off segmented lane.
    The zero setting retains the serial per-row QSA reduction as its exact
    control without changing persistent cache ownership.
    """
    if value is not None:
        return bool(value)
    return os.environ.get("MLX_LM_QSA_PRIVATE_DELTA", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def qsa_private_delta_exact_set_fold_enabled() -> bool:
    """Return whether admitted private-delta B2 uses the exact-set fold."""
    return os.environ.get("MLX_LM_QSA_PRIVATE_DELTA_EXACT_SET_FOLD", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _shared_qsa_suffix_mode() -> str:
    value = os.environ.get("MLX_LM_SHARED_QSA_SUFFIX", "auto").lower()
    if value in {"1", "true", "yes", "on"}:
        return "on"
    if value in {"0", "false", "no", "off"}:
        return "off"
    if value == "auto":
        return "auto"
    raise ValueError("MLX_LM_SHARED_QSA_SUFFIX must be auto, on, or off")


def shared_qsa_suffix_admission(
    *, base_tokens: int, remaining_tokens: int
) -> tuple[bool, str, int]:
    """Apply the measured context/output-budget crossover for shared QSA."""
    mode = _shared_qsa_suffix_mode()
    base_tokens = max(0, int(base_tokens))
    remaining_tokens = max(0, int(remaining_tokens))
    if mode == "off":
        return (False, "forced_off", 0)
    if mode == "on":
        return (True, "forced_on", remaining_tokens)
    minimum_context = int(
        os.environ.get("MLX_LM_SHARED_QSA_SUFFIX_MIN_CONTEXT", str(16 * 1024 - 4))
    )
    maximum_remaining = int(
        os.environ.get("MLX_LM_SHARED_QSA_SUFFIX_MAX_REMAINING", "64")
    )
    if minimum_context < 0 or maximum_remaining < 0:
        raise ValueError("shared QSA auto-policy limits must be non-negative")
    cutoff = min(maximum_remaining, (base_tokens + 1023) // 1024)
    if base_tokens < minimum_context:
        return (False, "context_below_minimum", cutoff)
    if remaining_tokens > cutoff:
        return (False, "output_budget_above_cutoff", cutoff)
    return (True, "auto_admitted", cutoff)


def note_segmented_self_mtp(key: str, amount: int = 1) -> None:
    if key not in _STATS:
        raise KeyError(f"unknown segmented self-MTP counter: {key}")
    with _STATS_LOCK:
        _STATS[key] = min((1 << 63) - 1, _STATS[key] + int(amount))


def note_qsa_private_delta_event(
    event: str,
    *,
    width: int,
    base_tokens: int = 0,
    rows: int = 0,
    duplicate_base_storage_bytes_not_formed: int = 0,
    reason: str | None = None,
) -> None:
    """Record one source-phased admission outcome with unambiguous units."""
    if event not in {"request", "engaged", "declined"}:
        raise ValueError(f"unknown private-delta event {event!r}")
    width_key = str(int(width)) if 1 <= int(width) <= 9 else "other"
    with _STATS_LOCK:
        _STATS[
            f"private_delta_width_{width_key}_{(event if event != 'request' else 'requests')}"
        ] += 1
        if event == "request":
            _STATS["private_delta_requests"] += 1
            return
        if event == "declined":
            _STATS["private_delta_declines"] += 1
            key = str(reason or "unspecified")
            _PRIVATE_DELTA_DECLINE_REASONS[key] = (
                _PRIVATE_DELTA_DECLINE_REASONS.get(key, 0) + 1
            )
            return
        _STATS["private_delta_attention_calls"] += 1
        _STATS["private_delta_rows"] += int(rows)
        base_tokens = int(base_tokens)
        _STATS["private_delta_base_tokens_cumulative"] += base_tokens
        _STATS["private_delta_base_tokens_last"] = base_tokens
        if not _STATS["private_delta_base_tokens_min"]:
            _STATS["private_delta_base_tokens_min"] = base_tokens
        else:
            _STATS["private_delta_base_tokens_min"] = min(
                _STATS["private_delta_base_tokens_min"], base_tokens
            )
        _STATS["private_delta_base_tokens_max"] = max(
            _STATS["private_delta_base_tokens_max"], base_tokens
        )
        _STATS["private_delta_duplicate_base_storage_bytes_not_formed_cumulative"] += (
            int(duplicate_base_storage_bytes_not_formed)
        )


def note_qsa_exact_set_fold_event(
    event: str, *, rows: int = 0, reason: str | None = None
) -> None:
    """Record host-visible exact-set admission without reading its predicate."""
    if event not in {"request", "proof", "engaged", "declined", "private_fallback"}:
        raise ValueError(f"unknown exact-set fold event {event!r}")
    with _STATS_LOCK:
        if event == "request":
            _STATS["exact_set_fold_requests"] += 1
        elif event == "proof":
            _STATS["exact_set_fold_device_proofs"] += 1
        elif event == "engaged":
            _STATS["exact_set_fold_attention_calls"] += 1
            _STATS["exact_set_fold_rows"] += int(rows)
        else:
            _STATS["exact_set_fold_declines"] += 1
            if event == "private_fallback":
                _STATS["exact_set_fold_private_fallbacks"] += 1
            else:
                _STATS["exact_set_fold_preflight_declines"] += 1
            key = str(reason or "unspecified")
            _EXACT_SET_FOLD_DECLINE_REASONS[key] = (
                _EXACT_SET_FOLD_DECLINE_REASONS.get(key, 0) + 1
            )


@dataclass(frozen=True)
class _PlaneStamp:
    """Immutable host metadata; it is not a mutable cache backing."""

    fingerprint: tuple[Any, ...]
    logical_position: int
    component_state: tuple[Any, ...]
    physical_position: int | None = None
    pending_tokens: tuple[int, ...] = ()
    pending_hidden_geometry: tuple[Any, ...] | None = None
    seed_hidden_geometry: tuple[Any, ...] | None = None
    current_token: int | None = None


def _cache_type(cache: Any) -> str:
    cls = type(cache)
    return f"{cls.__module__}.{cls.__qualname__}"


def _tree_nbytes(value: Any, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    if isinstance(value, dict):
        return sum((_tree_nbytes(item, seen) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return sum((_tree_nbytes(item, seen) for item in value))
    return 0


def _plane_groups(pair: Any) -> dict[CachePlaneKind, tuple[Any, ...]]:
    groups: dict[CachePlaneKind, list[Any]] = {}
    for cache in pair.target:
        name = type(cache).__name__.lower()
        if "ring" in name or "rotating" in name:
            groups.setdefault(CachePlaneKind.ATTENTION_RING, []).append(cache)
        elif hasattr(cache, "keys") or "kv" in name or "qsa" in name:
            groups.setdefault(CachePlaneKind.ATTENTION_KV, []).append(cache)
        if "qsa" in name or hasattr(cache, "index_keys"):
            groups.setdefault(CachePlaneKind.QSA_SUMMARY, []).append(cache)
        if "arrayscache" in name or (
            hasattr(cache, "cache") and (not hasattr(cache, "keys"))
        ):
            groups.setdefault(CachePlaneKind.GDN_RECURRENT, []).append(cache)
    groups[CachePlaneKind.MTP_DRAFT] = list(pair.draft)
    if not groups.get(CachePlaneKind.ATTENTION_KV) and (
        not groups.get(CachePlaneKind.ATTENTION_RING)
    ):
        groups[CachePlaneKind.ATTENTION_KV] = list(pair.target)
    return {kind: tuple(caches) for (kind, caches) in groups.items() if caches}


def _ring_capacity(caches: tuple[Any, ...]) -> int | None:
    capacities = {
        int(getattr(cache, "max_size"))
        for cache in caches
        if getattr(cache, "max_size", None) is not None
    }
    if not capacities:
        return None
    if len(capacities) != 1:
        raise ValueError("segmented ring cache capacities disagree")
    return capacities.pop()


_QSA_ATTENTION_FIELDS = ("keys", "values", "key_scale", "value_scale")
_QSA_SUMMARY_FIELDS = (
    "index_keys",
    "_qsa_pooled_keys",
    "_qsa_pooled_ratio",
    "_qsa_summary_identity",
    "_qsa_summary_restored",
    "_qsa_pending_pooled",
    "_mtp_share_topk",
    "_mtp_shared_topk",
)
_GDN_FIELDS = ("cache", "left_padding", "lengths")
_FORMAT_FIELDS = ("group_size", "bits", "key_bits", "value_bits", "rotate", "normalize")


def _component_fields(kind: CachePlaneKind, cache: Any) -> tuple[str, ...]:
    is_qsa = "qsa" in type(cache).__name__.lower() or hasattr(cache, "index_keys")
    if getattr(cache, "supports_shared_qsa_suffix", False):
        return ("state",)
    if kind == CachePlaneKind.MTP_DRAFT and is_qsa:
        return _QSA_ATTENTION_FIELDS + _QSA_SUMMARY_FIELDS
    if is_qsa and kind == CachePlaneKind.ATTENTION_KV:
        return _QSA_ATTENTION_FIELDS
    if is_qsa and kind == CachePlaneKind.QSA_SUMMARY:
        return _QSA_SUMMARY_FIELDS
    if kind == CachePlaneKind.GDN_RECURRENT and hasattr(cache, "cache"):
        return _GDN_FIELDS
    if hasattr(cache, "keys"):
        return _QSA_ATTENTION_FIELDS
    return ("*",)


def _fingerprint(kind: CachePlaneKind, caches: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(
        (
            (
                _cache_type(cache),
                id(cache),
                _component_fields(kind, cache),
                tuple(
                    (
                        (name, repr(getattr(cache, name, None)))
                        for name in _FORMAT_FIELDS
                    )
                ),
            )
            for cache in caches
        )
    )


def _plane_logical_bytes(kind: CachePlaneKind, caches: tuple[Any, ...]) -> int:
    total = 0
    for cache in caches:
        fields = _component_fields(kind, cache)
        if fields == ("*",):
            total += int(getattr(cache, "nbytes", 0))
        else:
            total += _tree_nbytes(
                tuple((getattr(cache, name, None) for name in fields))
            )
    return total


def _state_identity(value: Any) -> Any:
    if isinstance(value, mx.array):
        return ("array", id(value), tuple(value.shape), str(value.dtype))
    if isinstance(value, (type(None), bool, int, float, str, bytes)):
        return value
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (
                    (repr(key), _state_identity(item))
                    for (key, item) in sorted(
                        value.items(), key=lambda item: repr(item[0])
                    )
                )
            ),
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple((_state_identity(item) for item in value)))
    return (type(value).__module__, type(value).__qualname__, id(value))


def _component_state(kind: CachePlaneKind, caches: tuple[Any, ...]) -> tuple[Any, ...]:
    result = []
    for cache in caches:
        fields = _component_fields(kind, cache)
        values = (
            getattr(cache, "state", None)
            if fields == ("*",)
            else tuple((getattr(cache, name, None) for name in fields))
        )
        result.append((_cache_type(cache), fields, _state_identity(values)))
    return tuple(result)


def _group_positions(caches: Iterable[Any]) -> tuple[int, ...]:
    positions = []
    for cache in caches:
        offset = getattr(cache, "offset", None)
        if isinstance(offset, Integral) and (not isinstance(offset, bool)):
            positions.append(int(offset))
            continue
        size = type(cache).__dict__.get("size")
        if callable(size):
            value = size(cache)
            if isinstance(value, Integral) and (not isinstance(value, bool)):
                positions.append(int(value))
    return tuple(positions)


def _require_group_position(
    kind: CachePlaneKind, caches: tuple[Any, ...], expected: int
) -> int | None:
    positions = _group_positions(caches)
    if positions and any((position != int(expected) for position in positions)):
        raise ValueError(
            f"segmented {kind.value} positions {positions} disagree with logical coordinate {expected}"
        )
    return None if not positions else positions[0]


def _hidden_geometry(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return (tuple(value.shape), str(value.dtype), id(value))


def _layout_id(kind: CachePlaneKind, caches: tuple[Any, ...]) -> str:
    geometry = (
        kind.value,
        tuple((_cache_type(cache) for cache in caches)),
        tuple((_component_fields(kind, cache) for cache in caches)),
        tuple(
            (
                tuple(
                    (
                        (name, repr(getattr(cache, name, None)))
                        for name in _FORMAT_FIELDS
                    )
                )
                for cache in caches
            )
        ),
        _ring_capacity(caches) if kind == CachePlaneKind.ATTENTION_RING else None,
    )
    return hashlib.sha256(repr(geometry).encode()).hexdigest()


def _stamp(
    kind: CachePlaneKind, caches: tuple[Any, ...], lane: Any, logical_position: int
) -> _PlaneStamp:
    for cache in caches:
        is_qsa = "qsa" in type(cache).__name__.lower() or hasattr(cache, "index_keys")
        if (
            is_qsa
            and kind in (CachePlaneKind.QSA_SUMMARY, CachePlaneKind.MTP_DRAFT)
            and (
                bool(getattr(cache, "_mtp_share_topk", False))
                or getattr(cache, "_mtp_shared_topk", None) is not None
            )
        ):
            raise ValueError("QSA shared-top-k cycle is not closed at publication")
    if kind != CachePlaneKind.MTP_DRAFT:
        _require_group_position(kind, caches, int(logical_position))
        return _PlaneStamp(
            _fingerprint(kind, caches),
            int(logical_position),
            _component_state(kind, caches),
        )
    pending = tuple((int(token) for token in lane.pending_ts))
    pending_hidden = lane.pending_hs
    if bool(pending) != (pending_hidden is not None) or (
        pending_hidden is not None and int(pending_hidden.shape[1]) != len(pending)
    ):
        raise ValueError("MTP pending token/hidden sidecar geometry disagrees")
    expected_physical = int(logical_position) - len(pending) - 1
    physical = _require_group_position(kind, caches, expected_physical)
    if physical is None:
        raise ValueError(
            "MTP B1 cache plus pending sidecar is not caught up to the target logical boundary"
        )
    return _PlaneStamp(
        _fingerprint(kind, caches),
        int(logical_position),
        _component_state(kind, caches),
        physical_position=physical,
        pending_tokens=pending,
        pending_hidden_geometry=_hidden_geometry(pending_hidden),
        seed_hidden_geometry=_hidden_geometry(lane.seed_h),
        current_token=int(lane.cur),
    )


class SegmentedLaneTransaction:
    """One lane's aligned cache-plane generation ledger."""

    def __init__(self, pair: Any, lane: Any, position: int) -> None:
        groups = _plane_groups(pair)
        bases = []
        self._groups = tuple(sorted(groups, key=lambda kind: kind.value))
        self._fingerprints = {
            kind: _fingerprint(kind, groups[kind]) for kind in self._groups
        }
        for kind in self._groups:
            caches = groups[kind]
            bases.append(
                PlaneBase(
                    kind=kind,
                    payload=_stamp(kind, caches, lane, position),
                    start=0,
                    length=int(position),
                    layout_id=_layout_id(kind, caches),
                    generation=0,
                    logical_bytes=_plane_logical_bytes(kind, caches),
                    payload_is_immutable=True,
                    ring_capacity=_ring_capacity(caches)
                    if kind == CachePlaneKind.ATTENTION_RING
                    else None,
                )
            )
        lineage = create_cache_delta_lineage(bases, enabled=True)
        if lineage is None:
            raise RuntimeError("cache-delta lineage unexpectedly declined")
        self.lineage: CacheDeltaLineage = lineage
        self.predecessor_lineage_id: str | None = None

    def validate(
        self, pair: Any, lane: Any, expected_position: int, *, continuity: bool = True
    ) -> None:
        groups = _plane_groups(pair)
        if set(groups) != set(self._groups):
            raise ValueError("segmented cache plane membership changed")
        for kind in self._groups:
            if _fingerprint(kind, groups[kind]) != self._fingerprints[kind]:
                raise ValueError(f"segmented {kind.value} fingerprint changed")
        current = {
            kind: _stamp(kind, groups[kind], lane, int(expected_position))
            for kind in self._groups
        }
        if continuity:
            view = self.lineage.current_view()
            canonical = (
                {delta.kind: delta.payload for delta in view.tip.deltas}
                if view.tip.deltas
                else {base.kind: base.payload for base in view.bases}
            )
            for kind in self._groups:
                if current[kind] != canonical[kind]:
                    raise ValueError(
                        f"segmented {kind.value} state changed outside its lineage"
                    )

    @property
    def closed(self) -> bool:
        return bool(self.lineage.stats()["closed"])

    @property
    def position(self) -> int:
        return self.lineage.position

    def fork(self, owner_id: str) -> CacheDeltaBranch:
        branch = self.lineage.fork(owner_id=owner_id)
        note_segmented_self_mtp("transaction_branches")
        return branch

    def canonicalize_live_tip(
        self, pair: Any, lane: Any, position: int
    ) -> "SegmentedLaneTransaction":
        """Rebase an exactly caught-up draft representation at one boundary."""
        self.validate(pair, lane, position, continuity=False)
        successor = SegmentedLaneTransaction(pair, lane, position)
        successor.predecessor_lineage_id = self.lineage.lineage_id
        self.close()
        note_segmented_self_mtp("transaction_canonicalizations")
        return successor

    def publish(
        self,
        branch: CacheDeltaBranch,
        pair: Any,
        lane: Any,
        advance: int,
        *,
        proposed: int,
        accepted: int,
        zero_rollback_attested: bool = False,
    ) -> "SegmentedLaneTransaction":
        advance = int(advance)
        if advance < 0:
            raise ValueError("segmented transaction advance must be non-negative")
        start = branch.position
        stop = start + advance
        if advance == 0:
            self.validate(pair, lane, stop, continuity=not zero_rollback_attested)
            branch.reject()
            note_segmented_self_mtp("transaction_rejections")
            note_segmented_self_mtp("accepted_zero")
            if zero_rollback_attested:
                return self.canonicalize_live_tip(pair, lane, stop)
            return self
        self.validate(pair, lane, stop, continuity=False)
        groups = _plane_groups(pair)
        bases = {base.kind: base for base in self.lineage.current_view().bases}
        deltas = {}
        for kind, base in bases.items():
            caches = groups[kind]
            ring_capacity = base.ring_capacity
            deltas[kind] = PlaneDelta(
                kind=kind,
                payload=_stamp(kind, caches, lane, stop),
                start=start,
                length=advance,
                layout_id=base.layout_id,
                generation=branch.generation,
                logical_bytes=0,
                payload_is_immutable=True,
                ring_capacity=ring_capacity,
                physical_start=start % ring_capacity if ring_capacity else None,
                physical_stop=stop % ring_capacity if ring_capacity else None,
                wrap_epoch=stop // ring_capacity if ring_capacity else None,
            )
        branch.append_checkpoint(deltas)
        branch.promote(start + advance)
        # Each stamp records the plane's whole state at its boundary, so the
        # promoted stamps become the bases. Keeping the chain would retain one
        # checkpoint per round for the lane's whole life.
        generation = self.lineage.generation
        self.lineage.rebase(
            {
                kind: replace(
                    base,
                    payload=deltas[kind].payload,
                    length=stop,
                    generation=generation,
                )
                for kind, base in bases.items()
            }
        )
        note_segmented_self_mtp("transaction_promotions")
        if accepted <= 0:
            note_segmented_self_mtp("accepted_zero")
        elif accepted < proposed:
            note_segmented_self_mtp("accepted_partial")
        else:
            note_segmented_self_mtp("accepted_all")
        return self

    def close(self) -> None:
        self.lineage.dispose()


def segmented_self_mtp_stats(*, reset: bool = False) -> dict[str, Any]:
    with _STATS_LOCK:
        result = dict(_STATS)
        result["private_delta_decline_reasons"] = dict(_PRIVATE_DELTA_DECLINE_REASONS)
        result["exact_set_fold_decline_reasons"] = dict(_EXACT_SET_FOLD_DECLINE_REASONS)
        if reset:
            _STATS.clear()
            _STATS.update(_ZERO)
            _PRIVATE_DELTA_DECLINE_REASONS.clear()
            _EXACT_SET_FOLD_DECLINE_REASONS.clear()
    result["environment_enabled"] = segmented_self_mtp_enabled()
    result["true_batched_environment_enabled"] = (
        true_batched_segmented_self_mtp_enabled()
    )
    result["qsa_private_delta_environment_enabled"] = qsa_private_delta_enabled()
    result["qsa_exact_set_fold_environment_enabled"] = (
        qsa_private_delta_exact_set_fold_enabled()
    )
    result["shared_qsa_suffix_mode"] = _shared_qsa_suffix_mode()
    result["shared_qsa_suffix_environment_enabled"] = result["shared_qsa_suffix_mode"] != "off"
    result["async_qsa_promotion_environment_enabled"] = os.environ.get(
        "MLX_LM_SEGMENTED_ASYNC_QSA_PROMOTION", "0"
    ).lower() in {"1", "true", "yes", "on"}
    result["timing_enabled"] = segmented_self_mtp_timing_enabled()
    result["counter_scope"] = "segmented_mechanism_only"
    result["array_readbacks"] = 0
    return result


def require_segmented_self_mtp_engagement(
    counters: dict[str, Any] | None = None,
) -> None:
    counters = segmented_self_mtp_stats() if counters is None else counters
    if int(counters.get("engaged", 0)) < 1:
        raise RuntimeError("segmented self-MTP arm never engaged")
    if (
        int(counters.get("b1_target_forwards", 0))
        + int(counters.get("batched_target_forwards", 0))
        < 1
    ):
        raise RuntimeError("segmented self-MTP arm ran no target forward")
    if int(counters.get("physical_b2_formations", 0)):
        raise RuntimeError("segmented self-MTP arm formed a physical B2 cache")
    if int(counters.get("transaction_branches", 0)) < 1:
        raise RuntimeError("segmented self-MTP arm opened no transaction branch")
    if int(counters.get("committed_cycles", 0)) < 1 or (
        int(counters.get("transaction_promotions", 0))
        + int(counters.get("transaction_rejections", 0))
        < 1
    ):
        raise RuntimeError("segmented self-MTP arm completed no transaction")


def require_true_batched_segmented_self_mtp_engagement(
    counters: dict[str, Any] | None = None,
) -> None:
    counters = segmented_self_mtp_stats() if counters is None else counters
    require_segmented_self_mtp_engagement(counters)
    if int(counters.get("true_batched_engaged", 0)) < 1:
        raise RuntimeError("true batched segmented consumer never engaged")
    if int(counters.get("batched_target_forwards", 0)) < 1:
        raise RuntimeError("true batched segmented consumer ran no target batch")
    if int(counters.get("b1_target_forwards", 0)):
        raise RuntimeError("true batched segmented consumer fell back to serial B1")
    if int(counters.get("segmented_attention_calls", 0)) < 1:
        raise RuntimeError(
            "true batched segmented consumer used no segmented attention"
        )
    if int(counters.get("full_prefix_materialized_bytes", 0)):
        raise RuntimeError(
            "true batched segmented consumer materialized full-prefix B2"
        )


def require_qsa_private_delta_engagement(
    counters: dict[str, Any] | None = None,
) -> None:
    counters = segmented_self_mtp_stats() if counters is None else counters
    require_true_batched_segmented_self_mtp_engagement(counters)
    if int(counters.get("private_delta_attention_calls", 0)) < 1:
        raise RuntimeError("source-phased QSA private-delta consumer never engaged")
    if int(counters.get("private_delta_rows", 0)) < 2:
        raise RuntimeError("QSA private-delta consumer saw no batched row cohort")
    requests = int(counters.get("private_delta_requests", 0))
    engaged = int(counters.get("private_delta_attention_calls", 0))
    declined = int(counters.get("private_delta_declines", 0))
    if requests != engaged + declined:
        raise RuntimeError(
            "QSA private-delta request accounting is incomplete: "
            f"{requests} requests != {engaged} engaged + {declined} declined"
        )
    if declined:
        raise RuntimeError(
            f"QSA private-delta qualification saw {declined} declined calls"
        )


def require_qsa_exact_set_fold_engagement(
    counters: dict[str, Any] | None = None,
) -> None:
    counters = segmented_self_mtp_stats() if counters is None else counters
    require_qsa_private_delta_engagement(counters)
    if int(counters.get("exact_set_fold_attention_calls", 0)) < 1:
        raise RuntimeError("QSA exact-set folded consumer never engaged")
    if int(counters.get("exact_set_fold_rows", 0)) < 2:
        raise RuntimeError("QSA exact-set folded consumer saw no B2 cohort")
    requests = int(counters.get("exact_set_fold_requests", 0))
    engaged = int(counters.get("exact_set_fold_attention_calls", 0))
    declined = int(counters.get("exact_set_fold_declines", 0))
    if requests != engaged + declined:
        raise RuntimeError(
            "QSA exact-set fold accounting is incomplete: "
            f"{requests} requests != {engaged} engaged + {declined} declined"
        )
    if declined:
        raise RuntimeError(
            f"QSA exact-set fold qualification saw {declined} declined calls"
        )
