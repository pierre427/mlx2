# SPDX-License-Identifier: MIT
# Standalone APCv2; provenance and retained notices in provenance/.
from __future__ import annotations
import copy
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Iterable, List, Optional
import mlx.core as mx
from .cache_capsule import CacheCapsuleGeneration
from .cache_planes import (
    CompiledScheduleMetadata,
    PLEResidencyHints,
    PromptHostPlane,
    TranscriptLedgerPlane,
)
from .cow_cache import (
    COWCacheStale,
    COWCacheTelemetry,
    COWFrozenPromptCache,
    COWPromptCacheBranch,
    cow_cache_enabled,
    freeze_prompt_cache,
)
from .models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    PrefixIndex,
    PromptTrie,
    RotatingKVCache,
    _copy_prompt_cache_for_restore,
    _mark_prompt_cache_restored,
    can_trim_prompt_cache,
    load_prompt_cache,
    save_prompt_cache,
)

try:
    from .models.cache import achievable_trim
except ImportError:
    achievable_trim = None


@dataclass(frozen=True)
class APCKey:
    """Fail-closed identity for a reusable prompt-cache namespace.

    ``semantic_fingerprint`` is where multimodal callers bind image/audio or
    other non-token inputs.  It must change whenever equal token IDs could
    produce different model state.
    """

    model: Hashable
    revision: Optional[Hashable] = None
    adapter: Optional[Hashable] = None
    tokenizer_fingerprint: Optional[Hashable] = None
    cache_layout_fingerprint: Optional[Hashable] = None
    semantic_fingerprint: Optional[Hashable] = None


@dataclass(frozen=True)
class APCCapabilities:
    topology: str
    exact_prefix: bool
    arbitrary_branch: bool
    reason: Optional[str] = None
    stored: Optional[bool] = None
    native: Any = None


@dataclass
class APCLookup:
    cache: Optional[List[Any]]
    remaining_tokens: List[int]
    cached_tokens: int
    hit: bool
    hit_kind: Optional[str]
    miss_reason: Optional[str]
    native: Any = None
    sidecar: Any = None
    prep_telemetry: Any = None
    prompt_host: Optional[PromptHostPlane] = None
    transcript_ledger: Optional[TranscriptLedgerPlane] = None
    capsule_generation: Optional[int] = None
    segment_manifest: Any = None


@dataclass
class MTPAPCSidecar:
    """Persistent draft state captured at an exact target-cache boundary.

    ``rng_key``/``rng_draws`` carry the decode lane's position in its own
    random stream, so a resumed request continues that stream instead of
    repeating draws it already made. They stay ``None``/``0`` for a greedy or
    keyless lane, which draws nothing.
    """

    state: Any
    covered_tokens: int
    rng_key: Optional[Any] = None
    rng_draws: int = 0

    @property
    def nbytes(self) -> int:
        (mtp_cache, tail_hidden) = self.state
        cache_bytes = sum(
            (
                int(getattr(entry, "nbytes", 0))
                for entry in _walk_cache_entries(mtp_cache)
            )
        )
        hidden_bytes = int(getattr(tail_hidden, "nbytes", 0))
        return cache_bytes + hidden_bytes


def _walk_cache_entries(prompt_cache: Iterable[Any]):
    for entry in prompt_cache:
        if isinstance(entry, CacheList):
            yield from _walk_cache_entries(entry.caches)
        elif isinstance(entry, (list, tuple)):
            yield from _walk_cache_entries(entry)
        else:
            yield entry


def _iter_trie_entries(trie: PromptTrie):
    """Yield every stored cache entry, whatever the LRU bookkeeping says."""
    stack = [trie._trie]
    while stack:
        node = stack.pop()
        for token, child in node.items():
            if token == "__value__":
                yield child
            else:
                stack.append(child)


def inspect_apc_capabilities(prompt_cache: List[Any]) -> APCCapabilities:
    """Describe which lossless APC operations a concrete cache supports."""
    leaves = list(_walk_cache_entries(prompt_cache))
    if not leaves:
        return APCCapabilities("empty", False, False, "empty_cache")
    has_rotating = any((isinstance(c, RotatingKVCache) for c in leaves))
    has_full = any((isinstance(c, KVCache) for c in leaves))
    has_state = any((isinstance(c, ArraysCache) for c in leaves))
    known = all(
        (
            isinstance(c, (KVCache, RotatingKVCache, ArraysCache))
            or hasattr(c, "state")
            for c in leaves
        )
    )
    if has_state:
        topology = "checkpointed_hybrid"
    elif has_rotating and has_full:
        topology = "mixed_rotating_kv"
    elif has_rotating:
        topology = "rotating_kv"
    elif has_full:
        topology = "kv"
    else:
        topology = "custom"
    arbitrary_branch = can_trim_prompt_cache(prompt_cache)
    if not arbitrary_branch and achievable_trim is not None:
        arbitrary_branch = achievable_trim(prompt_cache, 1) is not None
    return APCCapabilities(
        topology=topology,
        exact_prefix=known,
        arbitrary_branch=arbitrary_branch,
        reason=None if known else "unsupported_cache_entry",
    )


class APCv2(PrefixIndex):
    """APCv2: atomic segmented state ownership, prefix indexing and bounded residency."""

    _STAT_KEYS = ("lookups", "hits", "misses", "cached_tokens", "stores")
    _DISK_STAT_KEYS = (
        "idle_spills",
        "pressure_spills",
        "restores",
        "restore_failures",
        "spill_failures",
        "disk_evictions",
        "bytes_written",
        "bytes_read",
    )
    _RETENTION_DEFAULT = "default"
    _RETENTION_PROMPT_BOUNDARY = "committed_prompt_boundary"
    _RETENTION_ROLES = frozenset(
        {_RETENTION_DEFAULT, _RETENTION_PROMPT_BOUNDARY}
    )

    def __init__(
        self,
        max_size: int = 10,
        max_bytes: int = 1 << 63,
        max_tokens: Optional[int] = None,
        *,
        layout_name: str,
        idle_disk_seconds: float = 0.0,
        idle_disk_dir: Optional[str] = None,
        idle_disk_max_bytes: int = 1 << 63,
        now_fn=time.monotonic,
    ):
        if not layout_name:
            raise ValueError("APCv2 requires a model cache-layout declaration")
        super().__init__(max_size=max_size, max_bytes=max_bytes, max_tokens=max_tokens)
        self._apc_lock = threading.RLock()
        self._cow_branching = True
        self._cow_telemetry = COWCacheTelemetry()
        self._apc_stats = {key: 0 for key in self._STAT_KEYS}
        self._apc_lifetime = {key: 0 for key in self._STAT_KEYS}
        self._apc_clears = 0
        self._capsule_generation = CacheCapsuleGeneration()
        self._idle_disk_seconds = max(0.0, float(idle_disk_seconds))
        if self._idle_disk_seconds > 0 and (not idle_disk_dir):
            raise ValueError(
                "idle_disk_dir is required when idle_disk_seconds is enabled"
            )
        self._idle_disk_dir = (
            Path(idle_disk_dir).expanduser().resolve()
            if idle_disk_dir and self._idle_disk_seconds > 0
            else None
        )
        self._idle_disk_max_bytes = max(0, int(idle_disk_max_bytes))
        self._now = now_fn
        self._last_idle_scan = 0.0
        self._disk_stats = {key: 0 for key in self._DISK_STAT_KEYS}
        self._disk_bytes = 0
        if self._idle_disk_dir is not None:
            self._idle_disk_dir.mkdir(parents=True, exist_ok=True)
            for path in self._idle_disk_dir.glob("apc-idle-*.safetensors"):
                try:
                    path.unlink()
                except OSError:
                    pass
        self._layer_segments = True
        self.layout_name = str(layout_name)

    @property
    def capsule_generation(self) -> CacheCapsuleGeneration:
        """Generation authority for work captured at an APC lookup boundary."""
        return self._capsule_generation

    @staticmethod
    def key(
        model: Hashable,
        *,
        revision: Optional[Hashable] = None,
        adapter: Optional[Hashable] = None,
        tokenizer_fingerprint: Optional[Hashable] = None,
        cache_layout_fingerprint: Optional[Hashable] = None,
        semantic_fingerprint: Optional[Hashable] = None,
    ) -> APCKey:
        return APCKey(
            model=model,
            revision=revision,
            adapter=adapter,
            tokenizer_fingerprint=tokenizer_fingerprint,
            cache_layout_fingerprint=cache_layout_fingerprint,
            semantic_fingerprint=semantic_fingerprint,
        )

    def _entry_records_locked(self):
        seen = set()
        for cache_type in self._lru._ordering:
            for key, tokens in tuple(self._lru._lrus[cache_type]):
                entry = self._trie.get(key, tokens)
                if entry is not None and id(entry) not in seen:
                    seen.add(id(entry))
                    yield (key, list(tokens), entry)

    @staticmethod
    def _entry_pinned(entry) -> bool:
        cache = entry.prompt_cache
        if not isinstance(cache, COWFrozenPromptCache):
            return False
        return int(getattr(cache.cow_owner, "pin_count", 0) or 0) > 0

    @classmethod
    def _entry_retention_rank(cls, entry) -> int:
        return int(
            getattr(entry, "_apc_retention_role", cls._RETENTION_DEFAULT)
            == cls._RETENTION_PROMPT_BOUNDARY
        )

    def _legacy_order_records_locked(self):
        """Return entries in the exact order ``PrefixIndex.CacheOrder.pop`` uses."""
        order = PrefixIndex.CacheOrder(list(self._lru._ordering))
        for cache_type in order._ordering:
            order._lrus[cache_type].extend(self._lru._lrus[cache_type])
        records = []
        while len(order):
            try:
                key, tokens = order.pop()
            except IndexError:
                # CacheOrder historically assumes the final queue is non-empty.
                # Preserve its ordering preference, while making sparse custom
                # cache-type mixes total rather than failing limit enforcement.
                pair = next(
                    (
                        order._lrus[cache_type].popleft()
                        for cache_type in order._ordering
                        if order._lrus[cache_type]
                    ),
                    None,
                )
                if pair is None:
                    break
                key, tokens = pair
            entry = self._trie.get(key, tokens)
            if entry is not None:
                records.append((key, list(tokens), entry))
        return records

    def _retention_ordered_candidates_locked(
        self, *, resident_only: bool, exclude=None, include_exclude: bool = False
    ):
        """Prefer generic checkpoints, retaining prompt boundaries until last."""
        records = []
        excluded = []
        for key, tokens, entry in self._legacy_order_records_locked():
            if self._entry_pinned(entry):
                continue
            if resident_only and not entry.prompt_cache:
                continue
            record = (
                self._entry_retention_rank(entry),
                float(getattr(entry, "_apc_last_access_at", 0.0)),
                key,
                tokens,
                entry,
            )
            (excluded if entry is exclude else records).append(record)
        # Python's stable sort retains PrefixIndex's mixed cache-type eviction
        # order for entries without an explicit prompt-boundary role.
        records.sort(key=lambda record: record[0])
        if include_exclude:
            excluded.sort(key=lambda record: record[0])
            records.extend(excluded)
        return records

    def _pressure_candidates_locked(self, *, exclude=None, include_exclude=True):
        return self._retention_ordered_candidates_locked(
            resident_only=True,
            exclude=exclude,
            include_exclude=include_exclude,
        )

    @staticmethod
    def _atomic_save_cache(path: Path, cache: List[Any]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.safetensors")
        try:
            save_prompt_cache(str(temporary), list(cache))
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass

    @staticmethod
    def _atomic_save_arrays(path: Path, arrays: dict[str, mx.array]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.safetensors")
        try:
            mx.save_safetensors(str(temporary), arrays)
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass

    @staticmethod
    def _disk_paths(entry) -> tuple[Path, ...]:
        disk = getattr(entry, "_apc_disk", None) or {}
        return tuple(
            (
                Path(path)
                for path in (disk.get("target"), disk.get("draft"), disk.get("aux"))
                if path
            )
        )

    def _remove_disk_files_locked(self, entry) -> None:
        for path in self._disk_paths(entry):
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            try:
                path.unlink()
            except OSError:
                continue
            self._disk_bytes = max(0, self._disk_bytes - int(size))
        entry._apc_disk = None

    def _drop_entry_locked(self, key, tokens, entry) -> None:
        current = self._trie.pop(key, tokens)
        if current is None:
            return
        self._lru.remove(key, tokens)
        self._n_bytes = max(0, self._n_bytes - int(current.nbytes))
        self._n_bytes_by_type[current.cache_type] = max(
            0, self._n_bytes_by_type[current.cache_type] - int(current.nbytes)
        )
        if isinstance(current.prompt_cache, COWFrozenPromptCache):
            current.prompt_cache.close()
        self._remove_disk_files_locked(current)
        current.prompt_cache = []
        current.sidecar = None
        current.nbytes = 0
        self._capsule_generation.advance()

    def _spill_entry_locked(self, key, tokens, entry, *, reason: str) -> bool:
        if self._idle_disk_dir is None or not entry.prompt_cache:
            return False
        if self._entry_pinned(entry):
            return False
        disk = getattr(entry, "_apc_disk", None)
        if not disk:
            stem = f"apc-idle-{uuid.uuid4().hex}"
            target = self._idle_disk_dir / f"{stem}.target.safetensors"
            draft = self._idle_disk_dir / f"{stem}.draft.safetensors"
            aux = self._idle_disk_dir / f"{stem}.aux.safetensors"
            created = []
            try:
                self._atomic_save_cache(target, entry.prompt_cache)
                created.append(target)
                sidecar = entry.sidecar
                sidecar_info = None
                if sidecar is not None:
                    (draft_cache, tail_hidden) = sidecar.state
                    self._atomic_save_cache(draft, draft_cache)
                    created.append(draft)
                    arrays = {}
                    if tail_hidden is not None:
                        arrays["tail_hidden"] = tail_hidden
                    if sidecar.rng_key is not None:
                        arrays["rng_key"] = sidecar.rng_key
                    if arrays:
                        self._atomic_save_arrays(aux, arrays)
                        created.append(aux)
                    sidecar_info = {
                        "covered_tokens": int(sidecar.covered_tokens),
                        "rng_draws": int(sidecar.rng_draws),
                        "kind": getattr(sidecar, "kind", "self_mtp"),
                        "binding": getattr(sidecar, "binding", ""),
                    }
                metadata = getattr(
                    getattr(entry.prompt_cache, "cow_owner", None), "metadata", None
                )
                disk = {
                    "target": str(target),
                    "draft": str(draft) if draft in created else None,
                    "aux": str(aux) if aux in created else None,
                    "sidecar": sidecar_info,
                    "cow_metadata": metadata,
                    "resident_nbytes": int(entry.nbytes),
                }
                entry._apc_disk = disk
                written = sum((path.stat().st_size for path in created))
                self._disk_bytes += int(written)
                self._disk_stats["bytes_written"] += int(written)
            except Exception:
                self._disk_stats["spill_failures"] += 1
                for path in created:
                    try:
                        path.unlink()
                    except OSError:
                        pass
                return False
        resident_nbytes = int(entry.nbytes)
        if isinstance(entry.prompt_cache, COWFrozenPromptCache):
            entry.prompt_cache.close()
        entry.prompt_cache = []
        entry.sidecar = None
        entry.nbytes = 0
        self._n_bytes = max(0, self._n_bytes - resident_nbytes)
        self._n_bytes_by_type[entry.cache_type] = max(
            0, self._n_bytes_by_type[entry.cache_type] - resident_nbytes
        )
        self._disk_stats["idle_spills" if reason == "idle" else "pressure_spills"] += 1
        self._capsule_generation.advance()
        return True

    @staticmethod
    def _restored_resident_nbytes(cache, sidecar) -> int:
        return sum((int(getattr(item, "nbytes", 0)) for item in cache)) + int(
            getattr(sidecar, "nbytes", 0)
        )

    def _reserve_restore_bytes_locked(self, required: int, *, entry) -> bool:
        """Reserve a hard-capped resident budget before publishing a restore."""
        required = int(required)
        original_limit = int(self.max_bytes)
        if required < 0 or required > original_limit:
            return False
        temporary_limit = original_limit - required
        self.max_bytes = temporary_limit
        try:
            self._spill_resident_budget_locked(
                exclude=entry, include_exclude=False
            )
        finally:
            self.max_bytes = original_limit
        return self._n_bytes <= temporary_limit

    def _restore_entry_locked(self, key, tokens, entry) -> bool:
        disk = getattr(entry, "_apc_disk", None) or {}
        target = disk.get("target")
        if not target:
            return False
        cache = None
        try:
            expected = int(disk.get("resident_nbytes", 0) or 0)
            # The recorded size is the cold-allocation admission estimate.  An
            # impossible snapshot must fail before disk I/O; clamping the
            # temporary limit to zero used to let it publish over max_bytes.
            if not self._reserve_restore_bytes_locked(expected, entry=entry):
                raise ValueError("APCv2 disk snapshot exceeds resident byte cap")
            cache = load_prompt_cache(target)
            sidecar = None
            sidecar_info = disk.get("sidecar")
            if sidecar_info is not None:
                draft = load_prompt_cache(disk["draft"])
                arrays = mx.load(disk["aux"]) if disk.get("aux") else {}
                sidecar_type = MTPAPCSidecar
                sidecar_extra = {}
                if sidecar_info.get("kind") == "external_draft_v1":
                    from .external_speculative import ExternalDraftState
                    sidecar_type = ExternalDraftState
                    sidecar_extra = {"kind": sidecar_info["kind"], "binding": sidecar_info["binding"]}
                elif sidecar_info.get("kind", "self_mtp") != "self_mtp":
                    raise ValueError("Unknown APC speculative sidecar kind")
                sidecar = sidecar_type(
                    (draft, arrays.get("tail_hidden")),
                    covered_tokens=int(sidecar_info["covered_tokens"]),
                    rng_key=arrays.get("rng_key"),
                    rng_draws=int(sidecar_info.get("rng_draws", 0)),
                    **sidecar_extra,
                )
            mx.eval([item.state for item in cache])
            if sidecar is not None:
                mx.eval(
                    [item.state for item in sidecar.state[0]],
                    *([sidecar.state[1]] if sidecar.state[1] is not None else []),
                    *([sidecar.rng_key] if sidecar.rng_key is not None else []),
                )
            # Treat the recorded size as an estimate, never as authority.  A
            # serializer/runtime change can alter the restored footprint; use
            # the concrete arrays to reserve again before making them visible.
            actual = self._restored_resident_nbytes(cache, sidecar)
            if not self._reserve_restore_bytes_locked(actual, entry=entry):
                raise ValueError("restored APCv2 snapshot exceeds resident byte cap")
            metadata = disk.get("cow_metadata")
            if self._cow_branching:
                (cache, sidecar) = freeze_prompt_cache(
                    cache,
                    key=key,
                    tokens=tokens,
                    cache_type=entry.cache_type,
                    sidecar=sidecar,
                    prompt_host=getattr(metadata, "prompt_host", None),
                    transcript_ledger=getattr(metadata, "transcript_ledger", None),
                    ple_hints=getattr(metadata, "ple_hints", None),
                    compiled_schedule=getattr(metadata, "compiled_schedule", None),
                    telemetry=self._cow_telemetry,
                    layer_segments=True,
                )
            published_nbytes = self._restored_resident_nbytes(cache, sidecar)
            if not self._reserve_restore_bytes_locked(
                published_nbytes, entry=entry
            ):
                raise ValueError("frozen APCv2 snapshot exceeds resident byte cap")
            restored_disk_bytes = sum(
                (path.stat().st_size for path in self._disk_paths(entry))
            )
            entry.prompt_cache = cache
            entry.sidecar = sidecar
            entry.nbytes = published_nbytes
            self._n_bytes += int(entry.nbytes)
            self._n_bytes_by_type[entry.cache_type] += int(entry.nbytes)
            # Publication ends the temporary restore exclusion.  The entry is
            # now resident, so ordinary enforcement may safely discard its
            # retained disk snapshot to satisfy the disk cap.
            self._enforce_disk_limit_locked()
            self._disk_stats["restores"] += 1
            self._disk_stats["bytes_read"] += restored_disk_bytes
            entry._apc_last_access_at = self._now()
            return True
        except Exception:
            if isinstance(cache, COWFrozenPromptCache):
                cache.close()
            self._disk_stats["restore_failures"] += 1
            return False

    def _enforce_disk_limit_locked(self, *, exclude=None) -> None:
        if self._disk_bytes <= self._idle_disk_max_bytes:
            return
        records = self._retention_ordered_candidates_locked(
            resident_only=False,
            exclude=exclude,
            include_exclude=False,
        )
        for _rank, _last_access, key, tokens, entry in records:
            if self._disk_bytes <= self._idle_disk_max_bytes:
                break
            if not getattr(entry, "_apc_disk", None):
                continue
            if entry.prompt_cache:
                self._remove_disk_files_locked(entry)
            else:
                self._drop_entry_locked(key, tokens, entry)
            self._disk_stats["disk_evictions"] += 1

    def _spill_resident_budget_locked(
        self, *, exclude=None, include_exclude: bool = True
    ) -> int:
        if self._idle_disk_dir is None or self._n_bytes <= self.max_bytes:
            return 0
        spilled = 0
        while self._n_bytes > self.max_bytes:
            records = self._pressure_candidates_locked(
                exclude=exclude, include_exclude=include_exclude
            )
            if not records:
                break
            (_rank, _last_access, key, tokens, entry) = records[0]
            if key is None or tokens is None:
                break
            if not self._spill_entry_locked(key, tokens, entry, reason="pressure"):
                break
            spilled += 1
        if spilled:
            mx.clear_cache()
            # A restore can spill another resident entry while its own entry
            # is still a disk-only placeholder.  Keep that placeholder alive
            # until publication instead of letting disk-limit enforcement
            # remove the trie path underneath the restore.
            self._enforce_disk_limit_locked(exclude=exclude)
        return spilled

    def _enforce_entry_limits_locked(self) -> None:
        """Apply APC count/resident limits after the retention role is visible."""
        while len(self._lru) > self.max_size:
            records = self._retention_ordered_candidates_locked(
                resident_only=False,
                exclude=None,
                include_exclude=True,
            )
            if not records:
                break
            _rank, _last_access, key, tokens, entry = records[0]
            self._drop_entry_locked(key, tokens, entry)
        if self._idle_disk_dir is not None:
            self._spill_resident_budget_locked(
                exclude=None, include_exclude=True
            )
        # A failed/unavailable spill must not turn a retention preference into
        # permission to exceed the hard resident cap.
        while self._n_bytes > self.max_bytes:
            records = self._pressure_candidates_locked(
                exclude=None, include_exclude=True
            )
            if not records:
                break
            _rank, _last_access, key, tokens, entry = records[0]
            self._drop_entry_locked(key, tokens, entry)
        self._enforce_disk_limit_locked()

    def spill_idle_entries(self, *, now: Optional[float] = None) -> int:
        """Move unpinned APC entries idle past the configured age to disk."""
        if self._idle_disk_dir is None or self._idle_disk_seconds <= 0:
            return 0
        now = self._now() if now is None else float(now)
        scan_interval = 1.0
        with self._apc_lock:
            if now - self._last_idle_scan < scan_interval:
                return 0
            self._last_idle_scan = now
            spilled = 0
            for key, tokens, entry in tuple(self._entry_records_locked()):
                last_access = float(getattr(entry, "_apc_last_access_at", now))
                if (
                    entry.prompt_cache
                    and now - last_access >= self._idle_disk_seconds
                    and self._spill_entry_locked(key, tokens, entry, reason="idle")
                ):
                    spilled += 1
            if spilled:
                mx.clear_cache()
                self._enforce_disk_limit_locked()
            return spilled

    def lookup(self, key: Hashable, tokens: Iterable[int], *, allow_disk_restore: bool = True) -> APCLookup:
        with self._apc_lock:
            return self._lookup_locked(key, tokens, allow_disk_restore=allow_disk_restore)

    def _lookup_locked(self, key: Hashable, tokens: Iterable[int], *, allow_disk_restore: bool = True) -> APCLookup:
        tokens = [int(token) for token in tokens]
        while True:
            trie_result = self._trie.search(key, tokens)
            retry = False
            seen = set()
            for path in (trie_result.exact, trie_result.longer, trie_result.shorter):
                if path is None or tuple(path) in seen:
                    continue
                seen.add(tuple(path))
                try:
                    entry = self._trie.get(trie_result.model, path)
                except KeyError:
                    # Restoring an earlier candidate may have reclaimed this
                    # neighboring path while reserving its real footprint.
                    entry = None
                if entry is None:
                    continue
                if getattr(entry, "_apc_disk", None) and (not entry.prompt_cache):
                    if not allow_disk_restore:
                        # Admission may inspect resident state, but disk I/O and
                        # array restoration require the cold allocation gate.
                        return APCLookup(None, tokens, 0, False, None,
                                         "disk_restore_requires_admission")
                    if not self._restore_entry_locked(trie_result.model, path, entry):
                        self._drop_entry_locked(trie_result.model, path, entry)
                        retry = True
                        break
                entry._apc_last_access_at = self._now()
            if not retry:
                # Reservation may have removed a neighboring candidate.  Make
                # the selection phase consume a fresh, internally consistent
                # trie result without repeatedly restoring mutually exclusive
                # neighbors under a tight resident budget.
                trie_result = self._trie.search(key, tokens)
                break
        sidecar_candidates = []
        for path, common in (
            (trie_result.exact, len(tokens)),
            (trie_result.longer, trie_result.common_prefix),
            (
                trie_result.shorter,
                len(trie_result.shorter) if trie_result.shorter is not None else 0,
            ),
        ):
            if path is None:
                continue
            try:
                entry = self._trie.get(trie_result.model, path)
            except KeyError:
                continue
            sidecar = getattr(entry, "sidecar", None)
            covered = int(getattr(sidecar, "covered_tokens", 0))
            if (
                sidecar is not None
                and 0 < covered < len(tokens)
                and (common >= covered)
            ):
                cache_offset = max(
                    (
                        getattr(c, "offset", 0)
                        for c in _walk_cache_entries(entry.prompt_cache)
                    ),
                    default=0,
                )
                if cache_offset == covered:
                    sidecar_candidates.append((covered, entry, sidecar))
        if sidecar_candidates:
            (covered, entry, sidecar) = max(
                sidecar_candidates, key=lambda item: item[0]
            )
            try:
                restored_cache = _copy_prompt_cache_for_restore(entry.prompt_cache)
            except COWCacheStale:
                self._apc_stats["lookups"] += 1
                self._apc_stats["misses"] += 1
                return APCLookup(
                    None,
                    tokens,
                    0,
                    False,
                    None,
                    "stale_cow_generation",
                    capsule_generation=self._capsule_generation.current,
                )
            self._apc_stats["lookups"] += 1
            self._apc_stats["hits"] += 1
            self._apc_stats["cached_tokens"] += covered
            # Sidecar hits bypass PrefixIndex.fetch_nearest_cache; refresh the
            # selected checkpoint's recency so repeated reuse is not FIFO.
            selected_tokens = tokens[:covered]
            self._lru.remove(key, selected_tokens)
            self._lru.push(key, selected_tokens, entry.cache_type)
            restored_sidecar = getattr(restored_cache, "cow_sidecar", None)
            if (
                isinstance(restored_cache, COWPromptCacheBranch)
                and restored_sidecar is None
            ):
                restored_cache.close()
            else:
                if restored_sidecar is None:
                    restored_sidecar = copy.deepcopy(sidecar)
                _mark_prompt_cache_restored(restored_sidecar.state[0])
                return APCLookup(
                    restored_cache,
                    tokens[covered:],
                    covered,
                    True,
                    "external_draft_sidecar" if getattr(restored_sidecar, "kind", None) == "external_draft_v1" else "mtp_sidecar",
                    None,
                    sidecar=restored_sidecar,
                    prep_telemetry=getattr(restored_cache, "cow_prep_telemetry", None),
                    prompt_host=getattr(
                        getattr(restored_cache, "cow_metadata", None),
                        "prompt_host",
                        None,
                    ),
                    transcript_ledger=getattr(
                        getattr(restored_cache, "cow_metadata", None),
                        "transcript_ledger",
                        None,
                    ),
                    capsule_generation=self._capsule_generation.current,
                    segment_manifest=getattr(restored_cache, "cow_segment_stats", None),
                )
        try:
            (cache, remaining) = super().fetch_nearest_cache(key, tokens)
        except COWCacheStale:
            (cache, remaining) = (None, tokens)
            stale_generation = True
        else:
            stale_generation = False
        malformed_topology = False
        if cache is not None:
            prep_telemetry = getattr(cache, "cow_prep_telemetry", None)
            target_segments = (
                prep_telemetry.get("target_segments")
                if isinstance(prep_telemetry, dict)
                else None
            )
            malformed_topology = not cache or (
                target_segments is not None
                and (
                    not isinstance(target_segments, int)
                    or isinstance(target_segments, bool)
                    or target_segments < 1
                    or len(cache) != target_segments
                )
            )
            if malformed_topology:
                close = getattr(cache, "close", None)
                if callable(close):
                    close()
                cache = None
                remaining = tokens
        cached_tokens = len(tokens) - len(remaining) if cache is not None else 0
        hit = cache is not None and cached_tokens > 0
        self._apc_stats["lookups"] += 1
        self._apc_stats["hits" if hit else "misses"] += 1
        self._apc_stats["cached_tokens"] += cached_tokens
        if not hit:
            kind = None
            short_length = (
                len(trie_result.shorter) if trie_result.shorter is not None else 0
            )
            has_unusable_branch = trie_result.exact is not None or (
                trie_result.longer is not None
                and trie_result.common_prefix > short_length
            )
            if malformed_topology:
                reason = "malformed_cache_topology"
            elif stale_generation:
                reason = "stale_cow_generation"
            else:
                reason = (
                    "untrimmable_branch"
                    if has_unusable_branch
                    else "no_compatible_prefix"
                )
        elif trie_result.exact is not None:
            kind = "exact"
            reason = None
        else:
            kind = "prefix"
            reason = None
        return APCLookup(
            cache,
            remaining,
            cached_tokens,
            hit,
            kind,
            reason,
            prep_telemetry=getattr(cache, "cow_prep_telemetry", None),
            prompt_host=getattr(
                getattr(cache, "cow_metadata", None), "prompt_host", None
            ),
            transcript_ledger=getattr(
                getattr(cache, "cow_metadata", None), "transcript_ledger", None
            ),
            capsule_generation=self._capsule_generation.current,
            segment_manifest=getattr(cache, "cow_segment_stats", None),
        )

    def store(
        self,
        key: Hashable,
        tokens: Iterable[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
        sidecar: Any = None,
        prompt_host: Optional[PromptHostPlane] = None,
        transcript_ledger: Optional[TranscriptLedgerPlane] = None,
        ple_hints: Optional[PLEResidencyHints] = None,
        compiled_schedule: Optional[CompiledScheduleMetadata] = None,
        retention_role: str = _RETENTION_DEFAULT,
    ) -> APCCapabilities:
        if retention_role not in self._RETENTION_ROLES:
            raise ValueError(f"unknown APCv2 retention role: {retention_role!r}")
        with self._apc_lock:
            return self._store_locked(
                key,
                tokens,
                prompt_cache,
                cache_type=cache_type,
                sidecar=sidecar,
                prompt_host=prompt_host,
                transcript_ledger=transcript_ledger,
                ple_hints=ple_hints,
                compiled_schedule=compiled_schedule,
                retention_role=retention_role,
            )

    def _store_locked(
        self,
        key: Hashable,
        tokens: Iterable[int],
        prompt_cache: List[Any],
        *,
        cache_type: str = "assistant",
        sidecar: Any = None,
        prompt_host: Optional[PromptHostPlane] = None,
        transcript_ledger: Optional[TranscriptLedgerPlane] = None,
        ple_hints: Optional[PLEResidencyHints] = None,
        compiled_schedule: Optional[CompiledScheduleMetadata] = None,
        retention_role: str = _RETENTION_DEFAULT,
    ) -> APCCapabilities:
        tokens = [int(token) for token in tokens]
        capabilities = inspect_apc_capabilities(prompt_cache)
        if not capabilities.exact_prefix:
            return capabilities
        if self.max_tokens is not None and len(tokens) > self.max_tokens:
            self.overlength_rejections += 1
            return capabilities
        if self._cow_branching:
            try:
                (prompt_cache, sidecar) = freeze_prompt_cache(
                    prompt_cache,
                    key=key,
                    tokens=tokens,
                    cache_type=cache_type,
                    sidecar=sidecar,
                    prompt_host=prompt_host,
                    transcript_ledger=transcript_ledger,
                    ple_hints=ple_hints,
                    compiled_schedule=compiled_schedule,
                    telemetry=self._cow_telemetry,
                    layer_segments=True,
                )
            except Exception:
                self._cow_telemetry.add("freeze_failures")
                raise ValueError("APCv2 requires a valid segmented COW descriptor")
        cow_source = (
            prompt_cache if isinstance(prompt_cache, COWFrozenPromptCache) else None
        )
        before = (
            {id(entry): entry for entry in _iter_trie_entries(self._trie)}
            if self._cow_branching or self._idle_disk_dir is not None
            else {}
        )
        resident_limit = self.max_bytes
        sequence_limit = self.max_size
        # PrefixIndex cannot see APC retention roles.  Defer both of its hard
        # limits until the inserted entry has its role, then enforce them as
        # one role-aware operation below.
        self.max_bytes = 1 << 63
        self.max_size = 1 << 63
        try:
            super().insert_cache(
                key, tokens, prompt_cache, cache_type=cache_type, sidecar=sidecar
            )
        finally:
            self.max_bytes = resident_limit
            self.max_size = sequence_limit
        self._capsule_generation.advance()
        if before or cow_source is not None:
            live_entries = list(_iter_trie_entries(self._trie))
            live = {id(entry) for entry in live_entries}
            for ident, entry in before.items():
                if ident not in live:
                    if isinstance(entry.prompt_cache, COWFrozenPromptCache):
                        entry.prompt_cache.close()
                    self._remove_disk_files_locked(entry)
            if cow_source is not None and (
                not any((entry.prompt_cache is cow_source for entry in live_entries))
            ):
                cow_source.close()
        survivor = self._trie.search(key, tokens)
        if survivor.exact is not None:
            stored_entry = self._trie.get(key, survivor.exact)
            if retention_role == self._RETENTION_PROMPT_BOUNDARY:
                stored_entry._apc_retention_role = retention_role
            elif hasattr(stored_entry, "_apc_retention_role"):
                del stored_entry._apc_retention_role
            stored_entry._apc_last_access_at = self._now()
            self._enforce_entry_limits_locked()
        self._apc_stats["stores"] += 1
        return capabilities

    def clear(self, *, release_memory: bool = True) -> dict:
        """Drop every stored prefix, its MTP sidecar, and its bytes.

        A serving lever can change what a prefix *means*, not only how stale
        it is: the same tokens under two settings give different state, and
        some settings change the cache layout itself.  So entries are dropped,
        never marked stale, and the sidecars go with them.

        Safe to call on a live server.  The new trie and LRU are built first
        and then rebound, so a concurrent reader sees either the old cache or
        the empty one and never a half-emptied one.  It does not stop a
        request that is already generating from storing its own result
        afterwards; drain first when that matters.
        """
        with self._apc_lock:
            return self._clear_locked(release_memory=release_memory)

    def _clear_locked(self, *, release_memory: bool = True) -> dict:
        entries = list(_iter_trie_entries(self._trie))
        report = {
            "entries": len(entries),
            "sidecars": sum(
                (1 for entry in entries if getattr(entry, "sidecar", None) is not None)
            ),
            "bytes": int(self._n_bytes),
        }
        fresh_trie = PromptTrie()
        fresh_lru = PrefixIndex.CacheOrder(list(self._lru._ordering))
        self._trie = fresh_trie
        self._lru = fresh_lru
        self._n_bytes = 0
        self._n_bytes_by_type = {key: 0 for key in fresh_lru._ordering}
        for entry in entries:
            if isinstance(entry.prompt_cache, COWFrozenPromptCache):
                entry.prompt_cache.close()
            self._remove_disk_files_locked(entry)
        for entry in entries:
            entry.prompt_cache = []
            entry.sidecar = None
            entry.nbytes = 0
        entries.clear()
        if report["entries"]:
            self._capsule_generation.advance()
        for key in self._STAT_KEYS:
            self._apc_lifetime[key] += self._apc_stats[key]
            self._apc_stats[key] = 0
        self._apc_clears += 1
        if release_memory:
            mx.clear_cache()
        return report

    @property
    def _storage_stats(self):
        with self._apc_lock:
            stats = dict(self._apc_stats)
            stats["clears"] = self._apc_clears
            stats["lifetime"] = dict(self._apc_lifetime)
            for key in self._STAT_KEYS:
                stats["lifetime"][key] += self._apc_stats[key]
            stats["cow_enabled"] = self._cow_branching
            stats["cow"] = self._cow_telemetry.snapshot()
            stats["max_tokens"] = self.max_tokens
            stats["max_entry_tokens"] = self.max_entry_tokens
            stats["overlength_rejections"] = self.overlength_rejections
            stats["idle_disk"] = {
                "enabled": self._idle_disk_dir is not None,
                "idle_seconds": self._idle_disk_seconds,
                "resident_bytes": int(self._n_bytes),
                "disk_bytes": int(self._disk_bytes),
                "disk_max_bytes": int(self._idle_disk_max_bytes),
                "disk_entries": sum(
                    (
                        1
                        for entry in _iter_trie_entries(self._trie)
                        if getattr(entry, "_apc_disk", None)
                    )
                ),
                **dict(self._disk_stats),
            }
            return stats

    def evict_oldest_unleased(self) -> bool:
        """Reclaim one resident checkpoint without invalidating active branches.

        A configured disk tier keeps the checkpoint as a disk-only placeholder
        before releasing its resident arrays.  Disk-unavailable and failed/full
        spill cases retain the historical destructive eviction fallback so
        memory pressure can always make progress.
        """
        with self._apc_lock:
            records = self._pressure_candidates_locked()
            if records:
                (_rank, _last_access, key, tokens, entry) = records[0]
                if self._idle_disk_dir is not None and self._spill_entry_locked(
                    key, tokens, entry, reason="pressure"
                ):
                    # A zero/undersized disk budget may retire the new
                    # placeholder here.  Either outcome has released the
                    # resident arrays and therefore counts as progress.
                    self._enforce_disk_limit_locked()
                    return True
                self._drop_entry_locked(key, tokens, entry)
                return True
        return False

    def trim_to(
        self, *, n_sequences: Optional[int] = None, n_bytes: Optional[int] = None
    ):
        with self._apc_lock:
            return self._trim_to_locked(n_sequences=n_sequences, n_bytes=n_bytes)

    def _trim_to_locked(
        self, *, n_sequences: Optional[int] = None, n_bytes: Optional[int] = None
    ):
        before = {id(entry): entry for entry in _iter_trie_entries(self._trie)}
        super().trim_to(n_sequences=n_sequences, n_bytes=n_bytes)
        live = {id(entry) for entry in _iter_trie_entries(self._trie)}
        if live != set(before):
            self._capsule_generation.advance()
        for ident, entry in before.items():
            if ident not in live:
                if isinstance(entry.prompt_cache, COWFrozenPromptCache):
                    entry.prompt_cache.close()
                self._remove_disk_files_locked(entry)

    schema_version = 2

    @property
    def apc_stats(self):
        with self._apc_lock:
            stats = self._storage_stats
            aggregate = {
                "schema": "apcv2.layer-segments.v1",
                "entries": 0,
                "disk_entries": 0,
                "layers": 0,
                "plane_layers": 0,
                "segments": 0,
                "logical_bytes": 0,
                "by_plane": {},
            }
            for entry in _iter_trie_entries(self._trie):
                frozen = entry.prompt_cache
                if not isinstance(frozen, COWFrozenPromptCache):
                    aggregate["disk_entries"] += 1
                    continue
                summary = frozen.cow_owner.segment_stats()
                aggregate["entries"] += 1
                for key in ("layers", "plane_layers", "segments"):
                    aggregate[key] += int(summary.get(key, 0))
                for plane, values in summary.get("by_plane", {}).items():
                    combined = aggregate["by_plane"].setdefault(
                        plane,
                        {"layers": 0, "segments": 0, "logical_bytes": 0, "invalid": 0},
                    )
                    for key in combined:
                        combined[key] += int(values.get(key, 0))
                    aggregate["logical_bytes"] += int(values.get("logical_bytes", 0))
            stats["version"] = 2
            stats["layout_name"] = self.layout_name
            stats["layer_segments"] = aggregate
            return stats
