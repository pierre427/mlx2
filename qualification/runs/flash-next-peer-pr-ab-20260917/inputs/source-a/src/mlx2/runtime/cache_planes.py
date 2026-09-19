# SPDX-License-Identifier: Apache-2.0
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
import hashlib
import threading
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class CachePlaneKind(str, Enum):
    PROMPT_HOST = "prompt_host"
    TRANSCRIPT_LEDGER = "transcript_ledger"
    ATTENTION_KV = "attention_kv"
    ATTENTION_RING = "attention_ring"
    QSA_SUMMARY = "qsa_summary"
    GDN_RECURRENT = "gdn_recurrent"
    MTP_DRAFT = "mtp_draft"
    PLE_HINTS = "ple_hints"
    COMPILED_SCHEDULE = "compiled_schedule"


@dataclass(frozen=True, order=True)
class CacheSegmentKey:
    """Stable address of one layer-local token/state segment in APCv2."""

    kind: CachePlaneKind
    layer_index: int
    segment_index: int
    token_start: int
    token_stop: int
    role: str

    def __post_init__(self) -> None:
        if self.layer_index < 0 or self.segment_index < 0:
            raise ValueError("cache segment indices must be non-negative")
        if not 0 <= self.token_start <= self.token_stop:
            raise ValueError("invalid cache segment token span")
        if not self.role:
            raise ValueError("cache segment role is required")


@dataclass(frozen=True)
class CacheLayerSegment:
    """One independently accounted APCv2 layer segment."""

    key: CacheSegmentKey
    fingerprint: "CachePlaneFingerprint"
    logical_bytes: int
    required: bool = True

    def __post_init__(self) -> None:
        if self.fingerprint.kind != self.key.kind:
            raise ValueError("cache segment kind and fingerprint disagree")
        if self.logical_bytes < 0:
            raise ValueError("cache segment bytes must be non-negative")


@dataclass(frozen=True)
class CachePlaneFingerprint:
    """Exact compatibility key for one independently reusable plane."""

    kind: CachePlaneKind
    schema_version: int
    identity: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("cache-plane schema version must be positive")
        if tuple(sorted(self.identity)) != self.identity:
            raise ValueError("cache-plane identity fields must be sorted")

    @classmethod
    def from_fields(
        cls, kind: CachePlaneKind, *, schema_version: int = 1, **fields: Any
    ) -> "CachePlaneFingerprint":
        identity = tuple(
            sorted(((str(key), str(value)) for (key, value) in fields.items()))
        )
        return cls(kind, schema_version, identity)

    @property
    def digest(self) -> str:
        encoded = repr((self.kind.value, self.schema_version, self.identity)).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PromptPrefixSpan:
    name: str
    token_start: int
    token_stop: int
    character_start: int = 0
    character_stop: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("prompt prefix span needs a name")
        if not 0 <= self.token_start <= self.token_stop:
            raise ValueError("invalid prompt token span")
        if not 0 <= self.character_start <= self.character_stop:
            raise ValueError("invalid prompt character span")


@dataclass(frozen=True)
class PromptCacheKeyProvenance:
    """Stable cache-key inputs only; request/generation controls are absent."""

    model: str
    revision: str = ""
    adapter: str = ""
    semantic_fingerprint: str = ""
    cache_layout_fingerprint: str = ""


@dataclass(frozen=True)
class PromptHostPlane:
    """Canonical rendering/tokenization result, safe across device fallbacks.

    ``rendered_prompt`` can be empty when a fused chat-template call returns
    tokens without exposing its intermediate text.
    """

    input_fingerprint: str
    rendered_prompt: str
    token_ids: tuple[int, ...]
    prefix_spans: tuple[PromptPrefixSpan, ...]
    token_offsets: tuple[int, ...]
    tokenizer_identity: str
    tokenizer_version: str
    chat_template_identity: str
    chat_template_version: str
    cache_key_provenance: PromptCacheKeyProvenance
    initial_state: str = "normal"

    def __post_init__(self) -> None:
        if not self.input_fingerprint:
            raise ValueError("prompt input fingerprint is required")
        if not isinstance(self.initial_state, str) or not self.initial_state:
            raise ValueError("prompt initial state must be a non-empty string")
        if not self.tokenizer_identity or not self.chat_template_identity:
            raise ValueError("tokenizer and chat-template identities are required")
        if len(self.token_offsets) not in (0, len(self.token_ids)):
            raise ValueError("token offsets must be empty or match token IDs")
        if any((offset < 0 for offset in self.token_offsets)):
            raise ValueError("prompt token offsets must be non-negative")
        if tuple(sorted(self.token_offsets)) != self.token_offsets:
            raise ValueError("prompt token offsets must be monotone")
        if self.token_offsets and self.token_offsets[-1] > len(self.rendered_prompt):
            raise ValueError("prompt token offset exceeds rendered prompt")
        for token in self.token_ids:
            if isinstance(token, bool) or not isinstance(token, int) or token < 0:
                raise ValueError("prompt token IDs must be non-negative integers")
        for span in self.prefix_spans:
            if span.token_stop > len(self.token_ids):
                raise ValueError("prompt prefix span exceeds token count")
            if span.character_stop > len(self.rendered_prompt):
                raise ValueError("prompt prefix span exceeds rendered prompt")

    @property
    def fingerprint(self) -> CachePlaneFingerprint:
        provenance = self.cache_key_provenance
        return CachePlaneFingerprint.from_fields(
            CachePlaneKind.PROMPT_HOST,
            input=self.input_fingerprint,
            tokenizer=self.tokenizer_identity,
            tokenizer_version=self.tokenizer_version,
            chat_template=self.chat_template_identity,
            chat_template_version=self.chat_template_version,
            model=provenance.model,
            revision=provenance.revision,
            adapter=provenance.adapter,
            semantic=provenance.semantic_fingerprint,
        )


@dataclass(frozen=True)
class TranscriptLedgerSegment:
    """One immutable source-history span retained outside target attention."""

    segment_id: str
    token_start: int
    token_stop: int
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("transcript segment needs an identity")
        if not 0 <= self.token_start < self.token_stop:
            raise ValueError("invalid transcript segment token span")
        if self.token_stop - self.token_start != len(self.token_ids):
            raise ValueError("transcript segment span and token count disagree")
        if any(
            (
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in self.token_ids
            )
        ):
            raise ValueError("transcript token IDs must be non-negative integers")

    @property
    def digest(self) -> str:
        return hashlib.sha256(repr(self.token_ids).encode()).hexdigest()


@dataclass(frozen=True)
class TranscriptLedgerPlane:
    """Segmented uncompacted transcript for proposal/retrieval consumers only."""

    tokenizer_identity: str
    tokenizer_version: str
    revision: str
    segments: tuple[TranscriptLedgerSegment, ...]
    compaction_strategy: str = "oldest_contiguous"

    def __post_init__(self) -> None:
        if not self.tokenizer_identity or not self.revision:
            raise ValueError("transcript ledger identity and revision are required")
        if self.compaction_strategy not in (
            "oldest_contiguous",
            "largest_first",
            "lowest_importance",
        ):
            raise ValueError("unknown transcript compaction strategy")
        cursor = 0
        seen = set()
        for segment in self.segments:
            if segment.segment_id in seen:
                raise ValueError("duplicate transcript segment identity")
            if segment.token_start != cursor:
                raise ValueError("transcript segments must be contiguous and ordered")
            seen.add(segment.segment_id)
            cursor = segment.token_stop

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(
            (token for segment in self.segments for token in segment.token_ids)
        )

    @property
    def fingerprint(self) -> CachePlaneFingerprint:
        manifest = tuple(
            (
                (
                    segment.segment_id,
                    segment.token_start,
                    segment.token_stop,
                    segment.digest,
                )
                for segment in self.segments
            )
        )
        return CachePlaneFingerprint.from_fields(
            CachePlaneKind.TRANSCRIPT_LEDGER,
            schema_version=1,
            tokenizer=self.tokenizer_identity,
            tokenizer_version=self.tokenizer_version,
            revision=self.revision,
            compaction_strategy=self.compaction_strategy,
            segments=manifest,
        )


@dataclass(frozen=True)
class PLEResidencyHints:
    """Non-owning immutable hints; never an executable or a live buffer."""

    policy_version: str
    resident_layer_ids: tuple[int, ...]
    backing_identity: str


@dataclass(frozen=True)
class CompiledScheduleMetadata:
    """Validated identity only; functions/program handles are not cached here."""

    implementation_digest: str
    schedule_digest: str
    runtime_fingerprint: str
    geometry: tuple[int, ...]


@dataclass(frozen=True)
class CachePlaneFallback:
    kind: CachePlaneKind
    reason: str
    action: str


class CachePlaneLease:
    def __init__(self, owner: "CachePlaneOwner", generation: int, payload: Any):
        self.kind = owner.kind
        self.generation = int(generation)
        self.fingerprint = owner.fingerprint
        self.payload = payload
        self._owner = owner
        self._lock = threading.Lock()
        self._closed = False
        self._finalizer = weakref.finalize(self, owner._release)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            finalizer = self._finalizer
            self._owner = None
            self.payload = None
        finalizer()

    def __enter__(self) -> "CachePlaneLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class CachePlaneOwner:
    """One independently invalidatable immutable plane."""

    def __init__(
        self,
        *,
        kind: CachePlaneKind,
        payload: Any,
        fingerprint: CachePlaneFingerprint,
        eligible: bool = True,
        ineligible_reason: str | None = None,
    ) -> None:
        if fingerprint.kind != kind:
            raise ValueError("cache-plane kind and fingerprint disagree")
        if not eligible and (not ineligible_reason):
            raise ValueError("ineligible cache plane needs a reason")
        self.kind = kind
        self.fingerprint = fingerprint
        self._payload = payload
        self._eligible = bool(eligible)
        self._generation = 0
        self._pins = 0
        self._invalid_reason = ineligible_reason
        self._lock = threading.RLock()
        self._counters = {
            "lookups": 0,
            "hits": 0,
            "misses": 0,
            "materializations": 0,
            "materialized_bytes": 0,
            "invalidations": 0,
            "fallbacks": 0,
            "active_leases": 0,
        }

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def try_adopt(
        self, expected: CachePlaneFingerprint
    ) -> CachePlaneLease | CachePlaneFallback:
        with self._lock:
            self._counters["lookups"] += 1
            if not self._eligible:
                return self._fallback_locked(self._invalid_reason or "ineligible")
            if expected != self.fingerprint:
                return self._fallback_locked("fingerprint_mismatch")
            self._pins += 1
            self._counters["hits"] += 1
            self._counters["active_leases"] += 1
            return CachePlaneLease(self, self._generation, self._payload)

    def _fallback_locked(self, reason: str) -> CachePlaneFallback:
        self._counters["misses"] += 1
        self._counters["fallbacks"] += 1
        return CachePlaneFallback(self.kind, reason, f"rebuild_{self.kind.value}")

    def invalidate(self, reason: str) -> int:
        if not reason:
            raise ValueError("cache-plane invalidation needs a reason")
        with self._lock:
            if self._eligible:
                self._eligible = False
                self._generation += 1
                self._invalid_reason = str(reason)
                self._counters["invalidations"] += 1
                if self._pins == 0:
                    self._payload = None
            return self._generation

    def record_materialization(self, nbytes: int) -> None:
        if nbytes < 0:
            raise ValueError("materialized byte count must be non-negative")
        with self._lock:
            self._counters["materializations"] += 1
            self._counters["materialized_bytes"] += int(nbytes)

    def _release(self) -> None:
        with self._lock:
            if self._pins <= 0:
                raise RuntimeError("cache-plane lease underflow")
            self._pins -= 1
            self._counters["active_leases"] -= 1
            if not self._eligible and self._pins == 0:
                self._payload = None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._counters)
            result.update(
                {
                    "kind": self.kind.value,
                    "generation": self._generation,
                    "eligible": self._eligible,
                    "fingerprint": self.fingerprint.digest,
                    "invalidation_reason": self._invalid_reason,
                    "pinned": self._pins,
                    "payload_released": self._payload is None,
                }
            )
            return result


class LayeredCacheManifest:
    """Adopt or invalidate cache planes without cross-plane poisoning."""

    def __init__(self, owners: Iterable[CachePlaneOwner] = ()) -> None:
        self._owners = {}
        for owner in owners:
            self.add(owner)

    def add(self, owner: CachePlaneOwner) -> None:
        if owner.kind in self._owners:
            raise ValueError(f"duplicate cache plane: {owner.kind.value}")
        self._owners[owner.kind] = owner

    def try_adopt(
        self, kind: CachePlaneKind, expected: CachePlaneFingerprint
    ) -> CachePlaneLease | CachePlaneFallback:
        owner = self._owners.get(kind)
        if owner is None:
            return CachePlaneFallback(kind, "plane_absent", f"rebuild_{kind.value}")
        return owner.try_adopt(expected)

    def invalidate(self, kind: CachePlaneKind, reason: str) -> int | None:
        owner = self._owners.get(kind)
        return None if owner is None else owner.invalidate(reason)

    def invalidate_all(self, reason: str) -> None:
        for owner in self._owners.values():
            owner.invalidate(reason)

    def stats(self) -> dict[str, dict[str, Any]]:
        return {kind.value: owner.stats() for (kind, owner) in self._owners.items()}


class LayeredSegmentManifest:
    """APCv2 layer/segment ownership under one atomic cache generation.

    Segment invalidation is represented independently, but callers may still
    conservatively reject the whole correctness-bearing target boundary. This
    lets model integrations become more selective without changing the stored
    format or weakening today's restore contract.
    """

    def __init__(self, segments: Iterable[CacheLayerSegment] = ()) -> None:
        self._segments: dict[CacheSegmentKey, CacheLayerSegment] = {}
        self._invalid: dict[CacheSegmentKey, str] = {}
        for segment in segments:
            self.add(segment)

    def add(self, segment: CacheLayerSegment) -> None:
        if segment.key in self._segments:
            raise ValueError(f"duplicate cache segment: {segment.key!r}")
        self._segments[segment.key] = segment

    def invalidate(self, key: CacheSegmentKey, reason: str) -> bool:
        if key not in self._segments:
            return False
        if not reason:
            raise ValueError("cache segment invalidation needs a reason")
        self._invalid.setdefault(key, str(reason))
        return True

    def invalidate_plane(self, kind: CachePlaneKind, reason: str) -> int:
        changed = 0
        for key in self._segments:
            if key.kind == kind and key not in self._invalid:
                self._invalid[key] = str(reason)
                changed += 1
        return changed

    def invalid_required(self, *, include_mtp: bool = True) -> bool:
        return any(
            (
                segment.required
                and key in self._invalid
                and (include_mtp or key.kind != CachePlaneKind.MTP_DRAFT)
                for (key, segment) in self._segments.items()
            )
        )

    def summary(self) -> dict[str, Any]:
        by_plane: dict[str, dict[str, int]] = {}
        layers = set()
        for key, segment in self._segments.items():
            layers.add((key.kind.value, key.layer_index))
            values = by_plane.setdefault(
                key.kind.value,
                {"layers": 0, "segments": 0, "logical_bytes": 0, "invalid": 0},
            )
            values["segments"] += 1
            values["logical_bytes"] += int(segment.logical_bytes)
            values["invalid"] += int(key in self._invalid)
        for plane, values in by_plane.items():
            values["layers"] = sum((1 for (kind, _) in layers if kind == plane))
        return {
            "schema": "apcv2.layer-segments.v1",
            "layers": len({layer for (_, layer) in layers}),
            "plane_layers": len(layers),
            "segments": len(self._segments),
            "invalid_segments": len(self._invalid),
            "by_plane": by_plane,
        }

    @property
    def segments(self) -> tuple[CacheLayerSegment, ...]:
        return tuple(self._segments.values())
