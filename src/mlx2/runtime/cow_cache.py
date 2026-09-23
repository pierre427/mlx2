# SPDX-License-Identifier: Apache-2.0
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
import copy
import dataclasses
import hashlib
import os
import threading
import time
import uuid
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Optional
import mlx.core as mx
from .cache_planes import (
    CacheLayerSegment,
    CachePlaneFingerprint,
    CachePlaneKind,
    CachePlaneOwner,
    CacheSegmentKey,
    CompiledScheduleMetadata,
    LayeredCacheManifest,
    LayeredSegmentManifest,
    PLEResidencyHints,
    PromptHostPlane,
    TranscriptLedgerPlane,
)


class COWCacheError(RuntimeError):
    """Base error for a refused or stale COW cache operation."""


class COWCacheStale(COWCacheError):
    """The APC generation changed before a branch could be adopted."""


class COWCacheUnsupported(COWCacheError):
    """A cache graph cannot be cloned without speculative semantics."""


def cow_cache_enabled(value: Optional[bool] = None) -> bool:
    """Resolve the APC COW gate.  The absent environment value is off."""
    if value is not None:
        return bool(value)
    return os.environ.get("MLX_LM_APC_COW_BRANCH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def mtp_boundary_cow_enabled(value: Optional[bool] = None) -> bool:
    """Resolve the committed MTP-boundary snapshot gate. Default is on."""
    if value is not None:
        return bool(value)
    return os.environ.get("MLX_LM_MTP_BOUNDARY_COW", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def external_round_cow_enabled(value: Optional[bool] = None) -> bool:
    """Resolve descriptor snapshots for external/PLD boundaries. Default is off.

    ``MLX_LM_EXTERNAL_ROUND_COW=1`` replaces the deep copies of the external
    draft round checkpoint and the external/PLD prompt-boundary and finish
    caches.  ``mx.array.__deepcopy__`` already shares the immutable buffer,
    so the deep copy never duplicated KV bytes; descriptor COW adds
    stable-source validation at a small host cost and stays opt-in.
    """
    if value is not None:
        return bool(value)
    return os.environ.get("MLX_LM_EXTERNAL_ROUND_COW", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class COWPromptMetadata:
    """Host-only identity pinned to the immutable cache generation."""

    key: Hashable
    tokens: tuple[int, ...]
    cache_type: str
    prompt_host: PromptHostPlane | None = None
    transcript_ledger: TranscriptLedgerPlane | None = None
    ple_hints: PLEResidencyHints | None = None
    compiled_schedule: CompiledScheduleMetadata | None = None


@dataclass(frozen=True)
class COWBranchReceipt:
    """Per-lookup preparation attribution without device synchronization."""

    lineage_id: str
    generation: int
    branch_ns: int
    avoided_copy_bytes: int
    descriptor_aliases: int
    descriptor_bytes: int
    target_segments: int
    has_mtp_sidecar: bool
    physical_b2_formed: bool = False


@dataclass(frozen=True)
class COWDevicePlaneDescriptor:
    """Non-mutable manifest payload; live device arrays stay owner-private."""

    classes: tuple[str, ...]
    layout_digest: str
    logical_bytes: int


class COWCacheTelemetry:
    """Thread-safe host counters; no array readback is performed."""

    _ZERO = {
        "sources": 0,
        "source_bytes": 0,
        "freeze_failures": 0,
        "branches": 0,
        "branch_failures": 0,
        "stale_rejections": 0,
        "invalidations": 0,
        "active_leases": 0,
        "peak_leases": 0,
        "descriptor_aliases": 0,
        "descriptor_bytes": 0,
        "avoided_copy_bytes": 0,
        "materializations": 0,
        "materialized_bytes": 0,
        "fallback_deepcopies": 0,
        "fallback_bytes": 0,
        "freeze_ns": 0,
        "branch_ns": 0,
        "fallback_ns": 0,
        "releases": 0,
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values = dict(self._ZERO)
        self._planes: dict[str, dict[str, int]] = {}

    def add(self, key: str, amount: int = 1, *, plane: str | None = None) -> None:
        if key not in self._values:
            raise KeyError(f"unknown COW telemetry key: {key}")
        with self._lock:
            self._values[key] += int(amount)
            if plane is not None:
                values = self._planes.setdefault(
                    plane,
                    {
                        "descriptor_aliases": 0,
                        "descriptor_bytes": 0,
                        "materializations": 0,
                        "materialized_bytes": 0,
                    },
                )
                if key in values:
                    values[key] += int(amount)

    def lease_opened(self) -> None:
        with self._lock:
            self._values["active_leases"] += 1
            self._values["peak_leases"] = max(
                self._values["peak_leases"], self._values["active_leases"]
            )

    def lease_closed(self) -> None:
        with self._lock:
            if self._values["active_leases"] <= 0:
                raise RuntimeError("COW cache lease accounting underflow")
            self._values["active_leases"] -= 1
            self._values["releases"] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._values)
            result["planes"] = {
                name: dict(values) for (name, values) in self._planes.items()
            }
        result["materialized_bytes_are_estimated"] = True
        result["array_readbacks"] = 0
        return result


def _tree_nbytes(value: Any, seen: Optional[set[int]] = None) -> int:
    """Count logical array bytes without evaluating or copying arrays."""
    seen = set() if seen is None else seen
    ident = id(value)
    if ident in seen:
        return 0
    seen.add(ident)
    if isinstance(value, mx.array):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(
            (_tree_nbytes(k, seen) + _tree_nbytes(v, seen) for (k, v) in value.items())
        )
    if isinstance(value, (list, tuple, deque, set, frozenset)):
        return sum((_tree_nbytes(item, seen) for item in value))
    if hasattr(value, "__dict__"):
        return _tree_nbytes(vars(value), seen)
    return 0


def _tree_array_metrics(value: Any) -> tuple[int, int]:
    seen: set[int] = set()
    count = 0
    nbytes = 0

    def visit(item: Any) -> None:
        nonlocal count, nbytes
        ident = id(item)
        if ident in seen:
            return
        seen.add(ident)
        if isinstance(item, mx.array):
            count += 1
            nbytes += int(item.nbytes)
        elif isinstance(item, dict):
            for key, value in item.items():
                visit(key)
                visit(value)
        elif isinstance(item, (list, tuple, deque, set, frozenset)):
            for value in item:
                visit(value)
        elif hasattr(item, "__dict__") and (not callable(item)):
            visit(vars(item))

    visit(value)
    return (count, nbytes)


def _plane_for_cache(cache: Any, *, draft: bool = False) -> str:
    if draft:
        return "draft_mtp"
    name = type(cache).__name__.lower()
    if "qsa" in name:
        return "qsa_summary"
    if "arrayscache" in name or (
        hasattr(cache, "cache") and (not hasattr(cache, "keys"))
    ):
        return "gdn_recurrent"
    if "ring" in name or "rotating" in name:
        return "attention_ring"
    if hasattr(cache, "keys") or "kv" in name or "qsa" in name:
        return "attention_kv"
    return "cache_other"


def _plane_kind(name: str) -> CachePlaneKind | None:
    return {
        "attention_kv": CachePlaneKind.ATTENTION_KV,
        "attention_ring": CachePlaneKind.ATTENTION_RING,
        "qsa_summary": CachePlaneKind.QSA_SUMMARY,
        "gdn_recurrent": CachePlaneKind.GDN_RECURRENT,
        "draft_mtp": CachePlaneKind.MTP_DRAFT,
    }.get(name)


def _qsa_plane_projection(
    cache: Any, kind: CachePlaneKind
) -> tuple[tuple[Any, ...], tuple[tuple[str, str], ...]]:
    """Return non-overlapping QSA arrays plus format compatibility fields."""
    if kind == CachePlaneKind.ATTENTION_KV:
        values = (
            getattr(cache, "keys", None),
            getattr(cache, "values", None),
            getattr(cache, "key_scale", None),
            getattr(cache, "value_scale", None),
        )
        names = ("group_size", "bits", "key_bits", "value_bits", "rotate", "normalize")
    elif kind == CachePlaneKind.QSA_SUMMARY:
        values = (
            getattr(cache, "index_keys", None),
            getattr(cache, "_qsa_pooled_keys", None),
        )
        names = ("_qsa_pooled_ratio", "_qsa_summary_identity", "_qsa_summary_restored")
    else:
        raise ValueError(f"not a QSA plane: {kind.value}")
    compatibility = tuple(((name, repr(getattr(cache, name, None))) for name in names))
    return (values, compatibility)


def _layout_identity(values: Iterable[Any]) -> str:
    geometry = []
    seen: set[int] = set()
    for value in values:
        (count, nbytes) = _tree_array_metrics(value)
        shapes = []

        def visit(item: Any) -> None:
            ident = id(item)
            if ident in seen:
                return
            seen.add(ident)
            if isinstance(item, mx.array):
                shapes.append((tuple(item.shape), str(item.dtype)))
            elif isinstance(item, dict):
                for child in item.values():
                    visit(child)
            elif isinstance(item, (list, tuple, deque)):
                for child in item:
                    visit(child)
            elif hasattr(item, "__dict__") and (not callable(item)):
                visit(vars(item))

        visit(value)
        geometry.append(
            (type(value).__module__, type(value).__name__, count, nbytes, tuple(shapes))
        )
    return hashlib.sha256(repr(tuple(geometry)).encode()).hexdigest()


def _layer_segment_specs(
    cache: Any,
    *,
    layer_index: int,
    covered_tokens: int,
    tail_tokens: int = 256,
    draft: bool = False,
) -> list[tuple[CachePlaneKind, str, int, int, int, tuple[tuple[str, str], ...]]]:
    """Describe conservative APCv2 segments for one concrete cache layer."""
    plane_name = _plane_for_cache(cache, draft=draft)
    if draft:
        kinds = [(CachePlaneKind.MTP_DRAFT, cache, ())]
    elif plane_name == "qsa_summary":
        kinds = []
        for kind in (CachePlaneKind.ATTENTION_KV, CachePlaneKind.QSA_SUMMARY):
            (projection, compatibility) = _qsa_plane_projection(cache, kind)
            kinds.append((kind, projection, compatibility))
    else:
        kind = _plane_kind(plane_name)
        kinds = [] if kind is None else [(kind, cache, ())]
    offset = int(getattr(cache, "offset", covered_tokens) or covered_tokens)
    offset = max(offset, 0)
    specs = []
    for kind, projection, compatibility in kinds:
        logical_bytes = _tree_nbytes(projection)
        if kind == CachePlaneKind.GDN_RECURRENT:
            spans = [("state", 0, offset, logical_bytes)]
        elif kind == CachePlaneKind.ATTENTION_RING:
            retained = int(
                getattr(cache, "max_size", 0)
                or getattr(cache, "window_size", 0)
                or min(offset, tail_tokens)
            )
            spans = [("window", max(0, offset - retained), offset, logical_bytes)]
        else:
            split = max(0, offset - int(tail_tokens))
            spans = []
            if split > 0:
                prefix_bytes = int(logical_bytes * split / max(offset, 1))
                spans.append(("prefix", 0, split, prefix_bytes))
            tail_bytes = logical_bytes - sum((span[3] for span in spans))
            spans.append(("tail", split, offset, tail_bytes))
        for role, start, stop, nbytes in spans:
            specs.append((kind, role, start, stop, nbytes, compatibility))
    return specs


def _build_layer_segment_manifest(
    source_cache: Iterable[Any],
    source_sidecar: Any,
    *,
    covered_tokens: int,
    transcript_ledger: TranscriptLedgerPlane | None = None,
) -> LayeredSegmentManifest:
    segments = []
    for layer_index, cache in enumerate(source_cache):
        specs = _layer_segment_specs(
            cache, layer_index=layer_index, covered_tokens=covered_tokens
        )
        for segment_index, (
            kind,
            role,
            start,
            stop,
            nbytes,
            compatibility,
        ) in enumerate(specs):
            key = CacheSegmentKey(kind, layer_index, segment_index, start, stop, role)
            fingerprint = CachePlaneFingerprint.from_fields(
                kind,
                schema_version=2,
                layer=layer_index,
                segment=segment_index,
                role=role,
                token_start=start,
                token_stop=stop,
                cache_class=type(cache).__name__,
                compatibility=compatibility,
            )
            segments.append(CacheLayerSegment(key, fingerprint, nbytes))
    sidecar_state = getattr(source_sidecar, "state", source_sidecar)
    draft_cache = (
        sidecar_state[0]
        if isinstance(sidecar_state, (list, tuple)) and sidecar_state
        else None
    )
    if isinstance(draft_cache, (list, tuple)):
        for layer_index, cache in enumerate(draft_cache):
            specs = _layer_segment_specs(
                cache,
                layer_index=layer_index,
                covered_tokens=max(0, covered_tokens - 1),
                draft=True,
            )
            for segment_index, (
                kind,
                role,
                start,
                stop,
                nbytes,
                compatibility,
            ) in enumerate(specs):
                key = CacheSegmentKey(
                    kind, layer_index, segment_index, start, stop, role
                )
                fingerprint = CachePlaneFingerprint.from_fields(
                    kind,
                    schema_version=2,
                    layer=layer_index,
                    segment=segment_index,
                    role=role,
                    token_start=start,
                    token_stop=stop,
                    cache_class=type(cache).__name__,
                    compatibility=compatibility,
                )
                segments.append(CacheLayerSegment(key, fingerprint, nbytes))
    if transcript_ledger is not None:
        for segment_index, transcript in enumerate(transcript_ledger.segments):
            key = CacheSegmentKey(
                CachePlaneKind.TRANSCRIPT_LEDGER,
                0,
                segment_index,
                transcript.token_start,
                transcript.token_stop,
                f"transcript:{transcript.segment_id}",
            )
            fingerprint = CachePlaneFingerprint.from_fields(
                CachePlaneKind.TRANSCRIPT_LEDGER,
                schema_version=1,
                segment=transcript.segment_id,
                token_start=transcript.token_start,
                token_stop=transcript.token_stop,
                content=transcript.digest,
                ledger=transcript_ledger.fingerprint.digest,
            )
            segments.append(
                CacheLayerSegment(
                    key, fingerprint, len(transcript.token_ids) * 8, required=False
                )
            )
    return LayeredSegmentManifest(segments)


def _validate_stable_source(prompt_cache: Iterable[Any]) -> None:
    """Reject state whose temporary transaction semantics cannot be frozen."""
    stack = list(prompt_cache)
    while stack:
        cache = stack.pop()
        children = getattr(cache, "caches", None)
        if children is not None:
            stack.extend(children)
            continue
        if isinstance(cache, (list, tuple)):
            stack.extend(cache)
            continue
        if getattr(cache, "speculating", False):
            raise COWCacheUnsupported(
                f"{type(cache).__name__} is inside a speculation transaction"
            )
        if getattr(cache, "_rollbacks", None):
            raise COWCacheUnsupported(
                f"{type(cache).__name__} has live rollback records"
            )
        if getattr(cache, "_ple_rollback", None) is not None:
            raise COWCacheUnsupported(
                f"{type(cache).__name__} has a staged PLE rollback"
            )
        if (
            getattr(cache, "_mtp_share_topk", False)
            or getattr(cache, "_mtp_shared_topk", None) is not None
        ):
            raise COWCacheUnsupported(
                f"{type(cache).__name__} has a live QSA/MTP cycle"
            )


def _iter_cache_objects(prompt_cache: Iterable[Any]):
    stack = list(prompt_cache)
    while stack:
        cache = stack.pop()
        children = getattr(cache, "caches", None)
        if children is not None:
            stack.extend(children)
        elif isinstance(cache, (list, tuple)):
            stack.extend(cache)
        else:
            yield cache


class _SegmentToken:
    """Exactly-once logical first-write receipt for one cache object."""

    def __init__(self, telemetry: COWCacheTelemetry, plane: str, nbytes: int):
        self._telemetry = telemetry
        self.plane = plane
        self.nbytes = int(nbytes)
        self._lock = threading.Lock()
        self._materialized = False
        self.reason: str | None = None

    def note_mutation(self, reason: str) -> bool:
        if self._materialized:
            return False
        with self._lock:
            if self._materialized:
                return False
            self._materialized = True
            self.reason = str(reason)
        self._telemetry.add("materializations", plane=self.plane)
        self._telemetry.add("materialized_bytes", self.nbytes, plane=self.plane)
        return True


def _attach_mutation_hooks(cache: Any, tokens: dict[str, _SegmentToken]) -> None:
    """Instrument only COW instances, leaving the default hot path untouched.

    Python special methods such as ``__setitem__`` are resolved on the class,
    so callers that mutate recurrent slots directly should use the branch's
    ``note_mutation`` method around that operation. Normal cache update methods
    are covered automatically.
    """
    hooked_names = []
    for name in (
        "update_and_fetch",
        "update_index_keys",
        "advance",
        "filter",
        "extend",
    ):
        original = getattr(cache, name, None)
        if not callable(original):
            continue
        if name == "update_and_fetch" and "attention_kv" in tokens:
            token = tokens["attention_kv"]
        elif name == "update_index_keys" and "qsa_summary" in tokens:
            token = tokens["qsa_summary"]
        else:
            token = next(iter(tokens.values()))

        def hooked(*args, __name=name, __original=original, __token=token, **kwargs):
            __token.note_mutation(f"{type(cache).__name__}.{__name}")
            return __original(*args, **kwargs)

        setattr(cache, name, hooked)
        hooked_names.append(name)
    cache._cow_hooked_methods = tuple(hooked_names)
    state = getattr(cache, "cache", None)
    recurrent = tokens.get("gdn_recurrent")
    if recurrent is not None and isinstance(state, list):
        cache.cache = _COWTrackedList(state, recurrent)


class _COWTrackedList(list):
    """Branch-private recurrent slots with exactly-once write attribution."""

    def __init__(self, values: Iterable[Any], token: _SegmentToken) -> None:
        super().__init__(values)
        self._cow_token = token

    def _write(self, operation: str) -> None:
        self._cow_token.note_mutation(operation)

    def __setitem__(self, index, value) -> None:
        self._write("ArraysCache.__setitem__")
        super().__setitem__(index, value)

    def __delitem__(self, index) -> None:
        self._write("ArraysCache.__delitem__")
        super().__delitem__(index)

    def append(self, value) -> None:
        self._write("ArraysCache.append")
        super().append(value)

    def extend(self, values) -> None:
        self._write("ArraysCache.extend")
        super().extend(values)

    def insert(self, index, value) -> None:
        self._write("ArraysCache.insert")
        super().insert(index, value)

    def pop(self, index=-1):
        self._write("ArraysCache.pop")
        return super().pop(index)

    def remove(self, value) -> None:
        self._write("ArraysCache.remove")
        super().remove(value)

    def clear(self) -> None:
        self._write("ArraysCache.clear")
        super().clear()

    def reverse(self) -> None:
        self._write("ArraysCache.reverse")
        super().reverse()

    def sort(self, *args, **kwargs) -> None:
        self._write("ArraysCache.sort")
        super().sort(*args, **kwargs)

    def __iadd__(self, values):
        self._write("ArraysCache.__iadd__")
        return super().__iadd__(values)

    def __imul__(self, value):
        self._write("ArraysCache.__imul__")
        return super().__imul__(value)


_PRIMITIVES = (type(None), bool, int, float, complex, str, bytes)


def _clone_graph(
    value: Any,
    *,
    telemetry: COWCacheTelemetry,
    plane: str,
    attach_tokens: bool,
    memo: Optional[dict[int, Any]] = None,
) -> Any:
    """Clone containers/objects and alias MLX buffers descriptor-only."""
    memo = {} if memo is None else memo
    if isinstance(value, _PRIMITIVES) or callable(value):
        return value
    ident = id(value)
    if ident in memo:
        return memo[ident]
    if isinstance(value, mx.array):
        alias = mx.stop_gradient(value)
        memo[ident] = alias
        telemetry.add("descriptor_aliases", plane=plane)
        telemetry.add("descriptor_bytes", int(value.nbytes), plane=plane)
        return alias
    if isinstance(value, list):
        clone: list[Any] = []
        memo[ident] = clone
        clone.extend(
            (
                _clone_graph(
                    item,
                    telemetry=telemetry,
                    plane=plane,
                    attach_tokens=attach_tokens,
                    memo=memo,
                )
                for item in value
            )
        )
        return clone
    if isinstance(value, tuple) and (not hasattr(value, "_fields")):
        placeholder: list[Any] = []
        memo[ident] = placeholder
        clone = tuple(
            (
                _clone_graph(
                    item,
                    telemetry=telemetry,
                    plane=plane,
                    attach_tokens=attach_tokens,
                    memo=memo,
                )
                for item in value
            )
        )
        memo[ident] = clone
        return clone
    if isinstance(value, deque):
        clone = deque(maxlen=value.maxlen)
        memo[ident] = clone
        clone.extend(
            (
                _clone_graph(
                    item,
                    telemetry=telemetry,
                    plane=plane,
                    attach_tokens=attach_tokens,
                    memo=memo,
                )
                for item in value
            )
        )
        return clone
    if isinstance(value, dict):
        clone = {}
        memo[ident] = clone
        for key, item in value.items():
            clone[
                _clone_graph(
                    key,
                    telemetry=telemetry,
                    plane=plane,
                    attach_tokens=attach_tokens,
                    memo=memo,
                )
            ] = _clone_graph(
                item,
                telemetry=telemetry,
                plane=plane,
                attach_tokens=attach_tokens,
                memo=memo,
            )
        return clone
    if isinstance(value, set):
        clone = {
            _clone_graph(
                item,
                telemetry=telemetry,
                plane=plane,
                attach_tokens=attach_tokens,
                memo=memo,
            )
            for item in value
        }
        memo[ident] = clone
        return clone
    if isinstance(value, frozenset):
        clone = frozenset(
            (
                _clone_graph(
                    item,
                    telemetry=telemetry,
                    plane=plane,
                    attach_tokens=attach_tokens,
                    memo=memo,
                )
                for item in value
            )
        )
        memo[ident] = clone
        return clone
    if dataclasses.is_dataclass(value) and (not isinstance(value, type)):
        updates = {
            field.name: _clone_graph(
                getattr(value, field.name),
                telemetry=telemetry,
                plane=plane,
                attach_tokens=attach_tokens,
                memo=memo,
            )
            for field in dataclasses.fields(value)
        }
        clone = dataclasses.replace(value, **updates)
        memo[ident] = clone
        return clone
    if not hasattr(value, "__dict__"):
        raise COWCacheUnsupported(
            f"cannot safely COW-clone {type(value).__module__}.{type(value).__name__}"
        )
    try:
        clone = copy.copy(value)
    except Exception as error:
        raise COWCacheUnsupported(
            f"cannot copy {type(value).__module__}.{type(value).__name__}"
        ) from error
    memo[ident] = clone
    clone_plane = _plane_for_cache(value, draft=plane == "draft_mtp")
    ephemeral = {
        "_cow_segment_token",
        "_cow_segment_tokens",
        "_cow_hooked_methods",
        *getattr(value, "_cow_hooked_methods", ()),
    }
    for name in ephemeral:
        vars(clone).pop(name, None)
    for name, item in vars(value).items():
        if name in ephemeral:
            continue
        item_plane = clone_plane
        if clone_plane == "qsa_summary" and name in {"keys", "values"}:
            item_plane = "attention_kv"
        setattr(
            clone,
            name,
            _clone_graph(
                item,
                telemetry=telemetry,
                plane=item_plane,
                attach_tokens=attach_tokens,
                memo=memo,
            ),
        )
    if (
        attach_tokens
        and hasattr(clone, "nbytes")
        and (getattr(clone, "caches", None) is None)
    ):
        if clone_plane == "qsa_summary":
            attention_bytes = _tree_nbytes(
                (
                    getattr(clone, "keys", None),
                    getattr(clone, "values", None),
                    getattr(clone, "key_scale", None),
                    getattr(clone, "value_scale", None),
                )
            )
            total_bytes = _tree_nbytes(clone)
            tokens = {
                "attention_kv": _SegmentToken(
                    telemetry, "attention_kv", attention_bytes
                ),
                "qsa_summary": _SegmentToken(
                    telemetry, "qsa_summary", max(0, total_bytes - attention_bytes)
                ),
            }
        else:
            try:
                nbytes = int(clone.nbytes)
            except Exception:
                nbytes = _tree_nbytes(clone)
            tokens = {clone_plane: _SegmentToken(telemetry, clone_plane, nbytes)}
        clone._cow_segment_tokens = tokens
        clone._cow_segment_token = next(iter(tokens.values()))
        _attach_mutation_hooks(clone, tokens)
    return clone


class COWCacheOwner:
    """Immutable source and generation authority shared by live branches."""

    _TARGET_BLOCKING_PLANES = frozenset(
        {
            CachePlaneKind.ATTENTION_KV,
            CachePlaneKind.ATTENTION_RING,
            CachePlaneKind.QSA_SUMMARY,
            CachePlaneKind.GDN_RECURRENT,
        }
    )

    def __init__(
        self,
        source_cache: Iterable[Any],
        source_sidecar: Any,
        metadata: COWPromptMetadata,
        telemetry: COWCacheTelemetry,
        *,
        layer_segments: bool = False,
    ) -> None:
        self.lineage_id = uuid.uuid4().hex
        self.metadata = metadata
        self.telemetry = telemetry
        self._lock = threading.RLock()
        self._generation = 0
        self._valid = True
        self._pins = 0
        self._source_cache: tuple[Any, ...] | None = tuple(source_cache)
        self._source_sidecar = source_sidecar
        self._invalid_planes: dict[CachePlaneKind, str] = {}
        grouped: dict[
            CachePlaneKind, list[tuple[Any, Any, tuple[tuple[str, str], ...]]]
        ] = {}
        for cache in _iter_cache_objects(self._source_cache):
            plane = _plane_for_cache(cache)
            if plane == "qsa_summary":
                for kind in (CachePlaneKind.ATTENTION_KV, CachePlaneKind.QSA_SUMMARY):
                    (projection, compatibility) = _qsa_plane_projection(cache, kind)
                    grouped.setdefault(kind, []).append(
                        (cache, projection, compatibility)
                    )
            else:
                kind = _plane_kind(plane)
                if kind is not None:
                    grouped.setdefault(kind, []).append((cache, cache, ()))
        if source_sidecar is not None:
            grouped[CachePlaneKind.MTP_DRAFT] = [(source_sidecar, source_sidecar, ())]
        owners = []
        self._plane_descriptors: dict[CachePlaneKind, COWDevicePlaneDescriptor] = {}
        for kind, entries in grouped.items():
            caches = [entry[0] for entry in entries]
            projections = [entry[1] for entry in entries]
            compatibility = tuple((entry[2] for entry in entries))
            layout = _layout_identity(projections)
            fingerprint = CachePlaneFingerprint.from_fields(
                kind,
                layout=layout,
                classes=",".join((type(cache).__name__ for cache in caches)),
                compatibility=compatibility,
            )
            descriptor = COWDevicePlaneDescriptor(
                classes=tuple((type(cache).__name__ for cache in caches)),
                layout_digest=layout,
                logical_bytes=_tree_nbytes(projections),
            )
            self._plane_descriptors[kind] = descriptor
            owners.append(
                CachePlaneOwner(kind=kind, payload=descriptor, fingerprint=fingerprint)
            )
        if metadata.prompt_host is not None:
            owners.append(
                CachePlaneOwner(
                    kind=CachePlaneKind.PROMPT_HOST,
                    payload=metadata.prompt_host,
                    fingerprint=metadata.prompt_host.fingerprint,
                )
            )
        if metadata.transcript_ledger is not None:
            owners.append(
                CachePlaneOwner(
                    kind=CachePlaneKind.TRANSCRIPT_LEDGER,
                    payload=metadata.transcript_ledger,
                    fingerprint=metadata.transcript_ledger.fingerprint,
                )
            )
        if metadata.ple_hints is not None:
            owners.append(
                CachePlaneOwner(
                    kind=CachePlaneKind.PLE_HINTS,
                    payload=metadata.ple_hints,
                    fingerprint=CachePlaneFingerprint.from_fields(
                        CachePlaneKind.PLE_HINTS,
                        policy=metadata.ple_hints.policy_version,
                        backing=metadata.ple_hints.backing_identity,
                        layers=metadata.ple_hints.resident_layer_ids,
                    ),
                )
            )
        if metadata.compiled_schedule is not None:
            owners.append(
                CachePlaneOwner(
                    kind=CachePlaneKind.COMPILED_SCHEDULE,
                    payload=metadata.compiled_schedule,
                    fingerprint=CachePlaneFingerprint.from_fields(
                        CachePlaneKind.COMPILED_SCHEDULE,
                        implementation=metadata.compiled_schedule.implementation_digest,
                        schedule=metadata.compiled_schedule.schedule_digest,
                        runtime=metadata.compiled_schedule.runtime_fingerprint,
                        geometry=metadata.compiled_schedule.geometry,
                    ),
                )
            )
        self.plane_manifest = LayeredCacheManifest(owners)
        self.segment_manifest = (
            _build_layer_segment_manifest(
                self._source_cache,
                source_sidecar,
                covered_tokens=int(
                    getattr(source_sidecar, "covered_tokens", len(metadata.tokens))
                ),
                transcript_ledger=metadata.transcript_ledger,
            )
            if layer_segments
            else LayeredSegmentManifest()
        )

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def pin_count(self) -> int:
        with self._lock:
            return self._pins

    @property
    def source_released(self) -> bool:
        with self._lock:
            return self._source_cache is None

    def _release_source_locked(self) -> None:
        if self._source_cache is not None:
            self._source_cache = None
            self._source_sidecar = None

    def invalidate(self) -> int:
        with self._lock:
            if self._valid:
                self._valid = False
                self._generation += 1
                self.telemetry.add("invalidations")
                self.plane_manifest.invalidate_all("owner_invalidated")
                for segment in self.segment_manifest.segments:
                    self.segment_manifest.invalidate(segment.key, "owner_invalidated")
            if self._pins == 0:
                self._release_source_locked()
            return self._generation

    def invalidate_plane(self, kind: CachePlaneKind, reason: str) -> int | None:
        with self._lock:
            generation = self.plane_manifest.invalidate(kind, reason)
            if generation is not None:
                self._invalid_planes[kind] = reason
                self.segment_manifest.invalidate_plane(kind, reason)
        return generation

    def invalidate_segment(self, key: CacheSegmentKey, reason: str) -> bool:
        """Invalidate one APCv2 segment and conservatively block its plane."""
        with self._lock:
            changed = self.segment_manifest.invalidate(key, reason)
            if changed:
                self._invalid_planes.setdefault(key.kind, str(reason))
                self.plane_manifest.invalidate(key.kind, reason)
            return changed

    def _target_blocked_locked(self) -> bool:
        return self.segment_manifest.invalid_required(include_mtp=False) or any(
            (kind in self._TARGET_BLOCKING_PLANES for kind in self._invalid_planes)
        )

    def _branch_metadata_locked(self) -> COWPromptMetadata:
        return dataclasses.replace(
            self.metadata,
            prompt_host=None
            if CachePlaneKind.PROMPT_HOST in self._invalid_planes
            else self.metadata.prompt_host,
            transcript_ledger=None
            if CachePlaneKind.TRANSCRIPT_LEDGER in self._invalid_planes
            else self.metadata.transcript_ledger,
            ple_hints=None
            if CachePlaneKind.PLE_HINTS in self._invalid_planes
            else self.metadata.ple_hints,
            compiled_schedule=None
            if CachePlaneKind.COMPILED_SCHEDULE in self._invalid_planes
            else self.metadata.compiled_schedule,
        )

    def plane_stats(self) -> dict[str, dict[str, Any]]:
        stats = self.plane_manifest.stats()
        for kind, descriptor in self._plane_descriptors.items():
            stats[kind.value]["logical_bytes"] = descriptor.logical_bytes
            stats[kind.value]["classes"] = descriptor.classes
            stats[kind.value]["layout_digest"] = descriptor.layout_digest
        return stats

    def segment_stats(self) -> dict[str, Any]:
        return self.segment_manifest.summary()

    def branch(self, *, expected_generation: int) -> "COWPromptCacheBranch":
        started = time.perf_counter_ns()
        with self._lock:
            if (
                not self._valid
                or expected_generation != self._generation
                or self._source_cache is None
                or self._target_blocked_locked()
            ):
                self.telemetry.add("stale_rejections")
                raise COWCacheStale("APC COW source generation is stale")
            self._pins += 1
            self.telemetry.lease_opened()
            source_cache = self._source_cache
            source_sidecar = (
                None
                if CachePlaneKind.MTP_DRAFT in self._invalid_planes
                else self._source_sidecar
            )
        try:
            memo: dict[int, Any] = {}
            cache = [
                _clone_graph(
                    item,
                    telemetry=self.telemetry,
                    plane=_plane_for_cache(item),
                    attach_tokens=True,
                    memo=memo,
                )
                for item in source_cache
            ]
            sidecar = _clone_graph(
                source_sidecar,
                telemetry=self.telemetry,
                plane="draft_mtp",
                attach_tokens=True,
                memo=memo,
            )
            with self._lock:
                if (
                    not self._valid
                    or expected_generation != self._generation
                    or self._target_blocked_locked()
                ):
                    self.telemetry.add("stale_rejections")
                    raise COWCacheStale("APC COW source changed before branch adoption")
                if CachePlaneKind.MTP_DRAFT in self._invalid_planes:
                    sidecar = None
                metadata = self._branch_metadata_locked()
            branch_ns = time.perf_counter_ns() - started
            (descriptor_aliases, descriptor_bytes) = _tree_array_metrics(
                (source_cache, sidecar)
            )
            receipt = COWBranchReceipt(
                lineage_id=self.lineage_id,
                generation=expected_generation,
                branch_ns=branch_ns,
                avoided_copy_bytes=descriptor_bytes,
                descriptor_aliases=descriptor_aliases,
                descriptor_bytes=descriptor_bytes,
                target_segments=len(source_cache),
                has_mtp_sidecar=sidecar is not None,
            )
            branch = COWPromptCacheBranch(
                cache,
                owner=self,
                generation=expected_generation,
                sidecar=sidecar,
                receipt=receipt,
                metadata=metadata,
            )
        except Exception:
            self.telemetry.add("branch_failures")
            self._release_pin()
            raise
        self.telemetry.add("branches")
        self.telemetry.add("avoided_copy_bytes", receipt.avoided_copy_bytes)
        self.telemetry.add("branch_ns", receipt.branch_ns)
        return branch

    def _release_pin(self) -> None:
        with self._lock:
            if self._pins <= 0:
                raise RuntimeError("COW cache owner pin underflow")
            self._pins -= 1
            if not self._valid and self._pins == 0:
                self._release_source_locked()
        self.telemetry.lease_closed()

    def branch_rows(
        self, *, expected_generation: int, rows: int
    ) -> "COWCacheBranchGroup":
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise ValueError("COW cache row count must be a positive integer")
        branches = []
        try:
            for _ in range(rows):
                branches.append(self.branch(expected_generation=expected_generation))
        except Exception:
            for branch in branches:
                branch.close()
            raise
        return COWCacheBranchGroup(branches)

    def branch_live(
        self,
        source_cache: Iterable[Any],
        source_sidecar: Any,
        *,
        lineage_generation: int,
        memo: dict[int, Any] | None = None,
        metadata: COWPromptMetadata | None = None,
    ) -> "COWPromptCacheBranch":
        """Fork the current live tip, including post-APC catch-up state.

        This is the compatibility path for ``copy.deepcopy`` in self-MTP lane
        construction. It may run after APC eviction: an already-pinned live
        branch remains authoritative even though new branches from the frozen
        source are forbidden.
        """
        started = time.perf_counter_ns()
        source_cache = tuple(source_cache)
        with self._lock:
            if self._pins <= 0:
                raise COWCacheStale("live COW lineage is no longer pinned")
            self._pins += 1
            self.telemetry.lease_opened()
        try:
            memo = {} if memo is None else memo
            cache = [
                _clone_graph(
                    item,
                    telemetry=self.telemetry,
                    plane=_plane_for_cache(item),
                    attach_tokens=True,
                    memo=memo,
                )
                for item in source_cache
            ]
            sidecar = _clone_graph(
                source_sidecar,
                telemetry=self.telemetry,
                plane="draft_mtp",
                attach_tokens=True,
                memo=memo,
            )
            branch_ns = time.perf_counter_ns() - started
            (descriptor_aliases, descriptor_bytes) = _tree_array_metrics(
                (source_cache, source_sidecar)
            )
            receipt = COWBranchReceipt(
                lineage_id=self.lineage_id,
                generation=int(lineage_generation),
                branch_ns=branch_ns,
                avoided_copy_bytes=descriptor_bytes,
                descriptor_aliases=descriptor_aliases,
                descriptor_bytes=descriptor_bytes,
                target_segments=len(source_cache),
                has_mtp_sidecar=source_sidecar is not None,
            )
            branch = COWPromptCacheBranch(
                cache,
                owner=self,
                generation=lineage_generation,
                sidecar=sidecar,
                receipt=receipt,
                metadata=metadata,
            )
        except Exception:
            self.telemetry.add("branch_failures")
            self._release_pin()
            raise
        self.telemetry.add("branches")
        self.telemetry.add("avoided_copy_bytes", receipt.avoided_copy_bytes)
        self.telemetry.add("branch_ns", receipt.branch_ns)
        return branch


class COWCacheBranchGroup:
    """Independent B1 branches; no physical B=rows cache is formed."""

    def __init__(self, branches: Iterable["COWPromptCacheBranch"]) -> None:
        self.branches = tuple(branches)
        self.physical_batch_formed = False
        self._lock = threading.Lock()
        self._closed = False

    def __len__(self) -> int:
        return len(self.branches)

    def __iter__(self):
        return iter(self.branches)

    def __getitem__(self, index):
        return self.branches[index]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            branches = self.branches
            self.branches = ()
        for branch in branches:
            branch.close()


class COWPromptCacheBranch(list):
    """Native cache list plus explicit lineage/lease lifetime."""

    def __init__(
        self,
        values: Iterable[Any],
        *,
        owner: COWCacheOwner,
        generation: int,
        sidecar: Any,
        receipt: COWBranchReceipt,
        metadata: COWPromptMetadata | None = None,
    ) -> None:
        super().__init__(values)
        self.cow_owner = owner
        self.cow_generation = int(generation)
        self.cow_lineage_id = owner.lineage_id
        self.cow_metadata = owner.metadata if metadata is None else metadata
        self.cow_sidecar = sidecar
        self.cow_prep_telemetry = dataclasses.asdict(receipt)
        self.cow_plane_stats = owner.plane_stats()
        self.cow_segment_stats = owner.segment_stats()
        self._close_lock = threading.Lock()
        self._closed = False
        self._finalizer = weakref.finalize(self, owner._release_pin)

    @property
    def closed(self) -> bool:
        with self._close_lock:
            return self._closed

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            finalizer = self._finalizer
            self.clear()
            self.cow_sidecar = None
        finalizer()

    def __deepcopy__(self, memo: dict[int, Any]) -> "COWPromptCacheBranch":
        """Snapshot the current live tip for ordinary self-MTP clones.

        Copying instance-hook closures or the branch lock would either retain
        the original cache object or fail on ``thread.lock``.  A new owner
        live-tip branch preserves the existing ``copy.deepcopy(canonical)`` serving
        interface while keeping sibling mutations isolated.
        """
        with self._close_lock:
            if self._closed:
                raise COWCacheError("cannot deepcopy a closed COW cache branch")
            generation = self.cow_generation
            owner = self.cow_owner
            clone = owner.branch_live(
                tuple(self),
                self.cow_sidecar,
                lineage_generation=generation,
                memo=memo,
                metadata=self.cow_metadata,
            )
        memo[id(self)] = clone
        return clone

    def fork(self) -> "COWPromptCacheBranch":
        if self.closed:
            raise COWCacheError("cannot fork a closed COW cache branch")
        return self.cow_owner.branch(expected_generation=self.cow_generation)

    def fresh_rows(self, rows: int) -> COWCacheBranchGroup:
        """Fork independent rows from the frozen parent, never this live tip."""
        if self.closed:
            raise COWCacheError("cannot fork a closed COW cache branch")
        return self.cow_owner.branch_rows(
            expected_generation=self.cow_generation, rows=rows
        )

    def note_mutation(
        self, index: int, reason: str = "explicit", *, plane: str | None = None
    ) -> bool:
        if self.closed:
            raise COWCacheError("cannot mutate a closed COW cache branch")
        tokens = getattr(self[index], "_cow_segment_tokens", {})
        token = (
            tokens.get(plane)
            if plane is not None
            else getattr(self[index], "_cow_segment_token", None)
        )
        return False if token is None else token.note_mutation(reason)


class COWFrozenPromptCache(list):
    """APC-resident owner-private descriptor graph.

    The container is read-only. Its concrete cache objects are intentionally
    retained for native restore compatibility and must remain private to APC;
    callers receive mutable branches, never this source graph.
    """

    _cow_frozen = True

    def __init__(self, values: Iterable[Any], owner: COWCacheOwner) -> None:
        super().__init__(values)
        self.cow_owner = owner
        self.cow_generation = owner.generation

    @staticmethod
    def _immutable(*args, **kwargs):
        raise TypeError("COW APC source containers are read-only")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable

    def branch(self) -> COWPromptCacheBranch:
        return self.cow_owner.branch(expected_generation=self.cow_generation)

    def branch_rows(self, rows: int) -> COWCacheBranchGroup:
        return self.cow_owner.branch_rows(
            expected_generation=self.cow_generation, rows=rows
        )

    def close(self) -> None:
        self.cow_owner.invalidate()


def freeze_prompt_cache(
    prompt_cache: Iterable[Any],
    *,
    key: Hashable,
    tokens: Iterable[int],
    cache_type: str,
    sidecar: Any = None,
    prompt_host: PromptHostPlane | None = None,
    transcript_ledger: TranscriptLedgerPlane | None = None,
    ple_hints: PLEResidencyHints | None = None,
    compiled_schedule: CompiledScheduleMetadata | None = None,
    telemetry: Optional[COWCacheTelemetry] = None,
    layer_segments: bool = False,
) -> tuple[COWFrozenPromptCache, Any]:
    """Freeze target and optional MTP state without copying MLX buffers."""
    telemetry = telemetry or COWCacheTelemetry()
    started = time.perf_counter_ns()
    prompt_cache = list(prompt_cache)
    _validate_stable_source(prompt_cache)
    sidecar_state = getattr(sidecar, "state", None)
    if (
        sidecar_state is not None
        and isinstance(sidecar_state, (list, tuple))
        and sidecar_state
    ):
        draft_cache = sidecar_state[0]
    elif isinstance(sidecar, (list, tuple)):
        draft_cache = sidecar
    else:
        draft_cache = None
    if isinstance(draft_cache, (list, tuple)):
        _validate_stable_source(draft_cache)
    memo: dict[int, Any] = {}
    source_cache = [
        _clone_graph(
            item,
            telemetry=telemetry,
            plane=_plane_for_cache(item),
            attach_tokens=False,
            memo=memo,
        )
        for item in prompt_cache
    ]
    source_sidecar = _clone_graph(
        sidecar, telemetry=telemetry, plane="draft_mtp", attach_tokens=False, memo=memo
    )
    metadata = COWPromptMetadata(
        key=key,
        tokens=tuple((int(token) for token in tokens)),
        cache_type=str(cache_type),
        prompt_host=prompt_host,
        transcript_ledger=transcript_ledger,
        ple_hints=ple_hints,
        compiled_schedule=compiled_schedule,
    )
    owner = COWCacheOwner(
        source_cache, source_sidecar, metadata, telemetry, layer_segments=layer_segments
    )
    frozen = COWFrozenPromptCache(source_cache, owner)
    telemetry.add("sources")
    telemetry.add(
        "source_bytes", _tree_nbytes(source_cache) + _tree_nbytes(source_sidecar)
    )
    telemetry.add("freeze_ns", time.perf_counter_ns() - started)
    return (frozen, source_sidecar)


def snapshot_prompt_cache_descriptors(
    prompt_cache: Iterable[Any],
    sidecar: Any = None,
    *,
    memo: Optional[dict[int, Any]] = None,
) -> tuple[list[Any], Any, dict[str, int]]:
    """Capture one committed boundary as independent plane descriptors.

    Python/cache objects are cloned, while immutable MLX buffers remain
    descriptor aliases. Subsequent live-cache updates rebind only the live
    graph, so target, MTP draft, hidden seed, and RNG remain exact at capture.
    Callers snapshotting several rows pass one ``memo`` so objects the rows
    share, such as an immutable QSA base, stay shared in the snapshot.
    """
    telemetry = COWCacheTelemetry()
    started = time.perf_counter_ns()
    prompt_cache = list(prompt_cache)
    _validate_stable_source(prompt_cache)
    sidecar_state = getattr(sidecar, "state", sidecar)
    if isinstance(sidecar_state, (list, tuple)) and sidecar_state:
        draft_cache = sidecar_state[0]
        if isinstance(draft_cache, (list, tuple)):
            _validate_stable_source(draft_cache)
    memo = {} if memo is None else memo
    snapshot = [
        _clone_graph(
            item,
            telemetry=telemetry,
            plane=_plane_for_cache(item),
            attach_tokens=False,
            memo=memo,
        )
        for item in prompt_cache
    ]
    snapshot_sidecar = _clone_graph(
        sidecar, telemetry=telemetry, plane="draft_mtp", attach_tokens=False, memo=memo
    )
    return (
        snapshot,
        snapshot_sidecar,
        {
            "snapshot_ns": int(time.perf_counter_ns() - started),
            "snapshot_bytes": int(_tree_nbytes((snapshot, snapshot_sidecar))),
        },
    )


def _append_only_fields(cache: Any) -> tuple:
    """The in-place-appended buffers the cache's exact type declares.

    Only a class that names them in its own body qualifies, so a subclass
    that writes its buffers some other way is never treated as append-only
    by inheritance.
    """
    return type(cache).__dict__.get("_RECOVERY_APPEND_ONLY_FIELDS", ())


def _buffer_leaves(value: Any) -> tuple:
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def _buffer_geometry(value: Any, axis: int) -> tuple:
    """Each leaf's shape without the sequence axis, and its dtype."""
    return tuple(
        (
            tuple(size for i, size in enumerate(leaf.shape) if i != axis % leaf.ndim),
            str(leaf.dtype),
        )
        for leaf in _buffer_leaves(value)
    )


@dataclass(frozen=True)
class _BorrowedRecoveryPlane:
    """A live append-only cache whose buffers a recovery restore borrows."""

    live: Any
    clone: Any
    # (field, sequence axis, fill-level attribute, fill level, geometry)
    fields: tuple
    derived: tuple


def snapshot_recovery_descriptors(
    prompt_cache: Iterable[Any],
    sidecar: Any = None,
    *,
    memo: Optional[dict[int, Any]] = None,
) -> tuple[list[Any], Any, tuple]:
    """Capture a request-private recovery point without pinning KV buffers.

    ``snapshot_prompt_cache_descriptors`` aliases every MLX buffer. A live
    KV cache then appends into a buffer the alias still reads, which MLX
    cannot do in place, so every append until the checkpoint is dropped
    copies the whole buffer: bytes per token that grow with the context, on
    every decode cycle that captures a checkpoint first. A cache whose type
    declares ``_RECOVERY_APPEND_ONLY_FIELDS`` only ever writes those buffers
    at or past its fill level, so this snapshot keeps just the fill level
    for them and returns the live objects to borrow from on restore; the
    prefix below the level is still exactly the captured one then. Any
    other buffer (recurrent state replaced wholesale, a rotating window that
    overwrites old positions) keeps its descriptor alias.
    """
    memo = {} if memo is None else memo
    prompt_cache = list(prompt_cache)
    snapshot, snapshot_sidecar, _receipt = snapshot_prompt_cache_descriptors(
        prompt_cache, sidecar, memo=memo
    )
    roots = [prompt_cache]
    if isinstance(sidecar, (list, tuple)):
        roots.append(sidecar)
    borrowed = []
    seen = set()
    for live in _iter_cache_objects(roots):
        fields = _append_only_fields(live)
        clone = memo.get(id(live))
        if not fields or clone is None or clone is live or id(live) in seen:
            continue
        seen.add(id(live))
        record = []
        for name, axis, level_name in fields:
            value = getattr(live, name, None)
            if value is None:
                continue
            record.append(
                (
                    name,
                    axis,
                    level_name,
                    int(getattr(live, level_name)),
                    _buffer_geometry(value, axis),
                )
            )
            setattr(clone, name, None)
        derived = type(live).__dict__.get("_RECOVERY_DERIVED_FIELDS", ())
        for name in derived:
            setattr(clone, name, None)
        if record:
            borrowed.append(
                _BorrowedRecoveryPlane(live, clone, tuple(record), tuple(derived))
            )
    return snapshot, snapshot_sidecar, tuple(borrowed)


def restore_recovery_descriptors(
    snapshot: Iterable[Any],
    sidecar: Any,
    borrowed: tuple,
    *,
    memo: Optional[dict[int, Any]] = None,
) -> tuple[list[Any], Any]:
    """Rebuild a recovery point captured by ``snapshot_recovery_descriptors``.

    Borrowed buffers are re-read from the live caches the checkpoint named,
    as fresh descriptor aliases, after checking that each live cache still
    holds at least the captured fill level in the captured geometry. Anything
    else means the live prefix can no longer be proven to be the captured
    one, and the restore is refused rather than served.
    """
    memo = {} if memo is None else memo
    restored, restored_sidecar, _receipt = snapshot_prompt_cache_descriptors(
        snapshot, sidecar, memo=memo
    )
    pending = []
    for plane in borrowed:
        target = memo.get(id(plane.clone))
        if target is None:
            raise COWCacheError("a recovery restore lost a borrowed cache plane")
        who = type(plane.live).__name__
        for name, axis, level_name, level, geometry in plane.fields:
            value = getattr(plane.live, name, None)
            current = getattr(plane.live, level_name, None)
            if (
                value is None
                or not isinstance(current, int)
                or current < level
                or _buffer_geometry(value, axis) != geometry
                or any(leaf.shape[axis] < level for leaf in _buffer_leaves(value))
            ):
                raise COWCacheError(
                    f"{who}.{name} no longer holds its captured {level} positions; the recovery point cannot be restored exactly"
                )
            alias = tuple(mx.stop_gradient(leaf) for leaf in _buffer_leaves(value))
            pending.extend(alias)
            setattr(
                target, name, alias if isinstance(value, (tuple, list)) else alias[0]
            )
        for name in plane.derived:
            setattr(target, name, None)
    if pending:
        # Surface a live buffer whose graph cannot be evaluated here, where
        # the caller can still fail closed, not at the next forward.
        try:
            mx.eval(pending)
        except Exception as error:
            raise COWCacheError(
                "a borrowed recovery buffer could not be evaluated"
            ) from error
    return restored, restored_sidecar


def snapshot_committed_cache(
    prompt_cache: Iterable[Any], sidecar: Any = None, *, enabled: Optional[bool] = None
) -> tuple[list[Any], Any, str]:
    """Return an independent copy of one committed boundary.

    Descriptor COW is preferred (see :func:`snapshot_prompt_cache_descriptors`);
    a graph that cannot be frozen, e.g. one inside a speculation transaction,
    falls back to the historical deep copy.  The mode is ``descriptor_cow``,
    ``deepcopy_fallback`` or ``deepcopy_disabled``.

    A graph restored from an APC branch always takes the descriptor route.
    Its cache objects carry segment tokens holding locks, which a deep copy
    cannot pickle, and mutation hooks bound to the branch's own objects,
    which a deep copy would share, so the copy's writes would land in the
    branch.
    """
    prompt_cache = list(prompt_cache)
    cow_graph = _carries_cow_bookkeeping(prompt_cache, sidecar)
    if cow_graph or external_round_cow_enabled(enabled):
        try:
            cache, frozen_sidecar, _receipt = snapshot_prompt_cache_descriptors(
                prompt_cache, sidecar
            )
            return cache, frozen_sidecar, "descriptor_cow"
        except COWCacheUnsupported:
            mode = "deepcopy_fallback"
    else:
        mode = "deepcopy_disabled"
    if cow_graph:
        # Strip the bookkeeping so the deep copy sees only the cache state.
        memo: dict[int, Any] = {}
        telemetry = COWCacheTelemetry()
        prompt_cache = _clone_graph(
            prompt_cache,
            telemetry=telemetry,
            plane="attention_kv",
            attach_tokens=False,
            memo=memo,
        )
        sidecar = _clone_graph(
            sidecar,
            telemetry=telemetry,
            plane="draft_mtp",
            attach_tokens=False,
            memo=memo,
        )
    cache, frozen_sidecar = copy.deepcopy((prompt_cache, sidecar))
    return cache, frozen_sidecar, mode


def _carries_cow_bookkeeping(prompt_cache: Any, sidecar: Any = None) -> bool:
    """Whether any cache object was cloned with COW segment tokens or hooks."""
    sidecar_state = getattr(sidecar, "state", None)
    return any(
        hasattr(cache, "_cow_segment_tokens") or hasattr(cache, "_cow_hooked_methods")
        for cache in _iter_cache_objects([prompt_cache, sidecar, sidecar_state])
    )


def restore_prompt_cache(prompt_cache: Any) -> COWPromptCacheBranch:
    if not isinstance(prompt_cache, COWFrozenPromptCache):
        raise COWCacheUnsupported("prompt cache is not a COW APC source")
    return prompt_cache.branch()


def record_fallback_deepcopy(
    prompt_cache: Any, telemetry: COWCacheTelemetry | None, elapsed_ns: int
) -> None:
    if telemetry is None:
        return
    telemetry.add("fallback_deepcopies")
    telemetry.add("fallback_bytes", _tree_nbytes(prompt_cache))
    telemetry.add("fallback_ns", int(elapsed_ns))
