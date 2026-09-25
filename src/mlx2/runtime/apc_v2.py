# SPDX-License-Identifier: MIT
# Standalone APCv2; provenance and retained notices in provenance/.
from __future__ import annotations
import ast
import copy
from bisect import bisect_left
from contextlib import ExitStack
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Hashable, Iterable, List, Optional
import mlx.core as mx
from .cache_capsule import CacheCapsuleGeneration, CacheCapsulePool
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
from .persistent_blocks import (
    block_file_paths,
    encode_block_file,
    materialize_block_file,
    remove_block_file,
)

try:
    from .models.cache import achievable_trim
except ImportError:
    achievable_trim = None


log = logging.getLogger(__name__)


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

    @property
    def interior_checkpoint_target(self) -> bool:
        """Whether exact interior snapshots benefit this target topology.

        Recurrent/hybrid state cannot branch at an arbitrary token unless an
        exact state checkpoint already exists there.  A restored branch can
        therefore become trimmable without ceasing to be the target topology
        that needs interior checkpoints.
        """
        return self.exact_prefix and self.topology == "checkpointed_hybrid"


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
    retention_role: Optional[str] = None
    # Deepest position a stored path shares with this prompt when that is
    # beyond what was restored.  On a checkpointed hybrid the shared prefix
    # cannot be branched from the longer entry (recurrent state does not
    # trim), so admission may plan a junction snapshot here for the next
    # request that diverges at the same point.  0 when nothing is shared past
    # ``cached_tokens``.
    branch_tokens: int = 0
    target_only_plain_fallback: bool = False


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


class _RestoreBudgetUnavailable(Exception):
    """The resident byte cap could not make room for a healthy disk snapshot."""


class _RestoreDigestFailure(Exception):
    """A persisted snapshot no longer matches its manifest digest."""


class APCSessionNotFound(KeyError):
    """A tenant-scoped APC session tag is unknown."""


class APCSessionCapacityError(RuntimeError):
    """A session pin would exceed a configured tenant/global byte cap."""


class APCSessionUnavailable(RuntimeError):
    """The APC session service can no longer accept asynchronous work."""


class _PersistedPinCapacityError(RuntimeError):
    """Restart settings cannot safely retain all unexpired persisted pins."""


def _json_identity_value(value):
    """Return a deterministic JSON representation or fail closed."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_json_identity_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    raise TypeError(f"APCv2 persistence identity is not JSON-safe: {type(value).__name__}")


def _key_manifest(key: APCKey) -> dict:
    if not isinstance(key, APCKey):
        raise TypeError("persistent APCv2 entries require APCKey identities")
    return {
        field: _json_identity_value(getattr(key, field))
        for field in (
            "model",
            "revision",
            "adapter",
            "tokenizer_fingerprint",
            "cache_layout_fingerprint",
            "semantic_fingerprint",
        )
    }


def _key_from_manifest(value: dict) -> APCKey:
    if not isinstance(value, dict):
        raise ValueError("APCv2 manifest key must be an object")
    fields = {
        "model",
        "revision",
        "adapter",
        "tokenizer_fingerprint",
        "cache_layout_fingerprint",
        "semantic_fingerprint",
    }
    if set(value) != fields:
        raise ValueError("APCv2 manifest key fields are incomplete")

    def freeze(item):
        if isinstance(item, list):
            return tuple(freeze(part) for part in item)
        if isinstance(item, dict):
            return tuple((name, freeze(part)) for name, part in sorted(item.items()))
        return item

    return APCKey(**{name: freeze(value[name]) for name in fields})


# ``persistent_blocks.encode_block_file`` staging names for an APCv2 payload
# ``apc-idle-<uuid>.<plane>.safetensors``.
_BLOCK_STAGING_DIRECTORY = re.compile(
    r"\.apc-idle-[0-9a-f]{32}\.(?:target|draft|aux)\.safetensors\.blocks"
    r"\.[0-9a-f]{32}\.tmp"
)
_BLOCK_STAGING_MANIFEST = re.compile(
    r"\.apc-idle-[0-9a-f]{32}\.(?:target|draft|aux)\.safetensors"
    r"\.[0-9a-f]{32}\.manifest"
)


# Serving's base semantic namespaces (``serving.cache_semantic_fingerprint``)
# and the per-request scope suffix ``cache_key_for`` appends for media, LoRA and
# hyper-directory requests.
_SEMANTIC_TEXT_NAMESPACE = "text-token-v1"
_SEMANTIC_SCOPE_SEPARATOR = ":media:"


def _is_int8_semantic_wrapper(semantic) -> bool:
    """Whether ``semantic`` is ``int8_prefill.apc_semantic_fingerprint``'s wrapper."""
    return (
        isinstance(semantic, tuple)
        and len(semantic) == 3
        and semantic[1] == "int8-prefill"
    )


def _is_tenant_semantic(semantic) -> bool:
    return (
        isinstance(semantic, tuple)
        and len(semantic) == 3
        and semantic[:2] == (_SEMANTIC_TEXT_NAMESPACE, "tenant")
        and isinstance(semantic[2], str)
    )


def _serving_semantic_mode(semantic) -> Optional[str]:
    """Classify a serving semantic namespace as ``shared``, ``tenant`` or None.

    Scoped keys are ``f"{base}:media:{scope}"``.  A tenant base formats as the
    tuple's repr, and the tenant id itself may contain the separator, so every
    split point is tried until the head parses back to a tenant tuple.
    """
    if semantic == _SEMANTIC_TEXT_NAMESPACE:
        return "shared"
    if _is_tenant_semantic(semantic):
        return "tenant"
    if not isinstance(semantic, str):
        return None
    head, separator, scope = semantic.partition(_SEMANTIC_SCOPE_SEPARATOR)
    if separator and scope and head == _SEMANTIC_TEXT_NAMESPACE:
        return "shared"
    if not semantic.startswith(repr((_SEMANTIC_TEXT_NAMESPACE, "tenant"))[:-1]):
        return None
    start = 0
    while (index := semantic.find(_SEMANTIC_SCOPE_SEPARATOR, start)) >= 0:
        if index + len(_SEMANTIC_SCOPE_SEPARATOR) < len(semantic):
            try:
                value = ast.literal_eval(semantic[:index])
            except (SyntaxError, ValueError, MemoryError, RecursionError):
                value = None
            if _is_tenant_semantic(value):
                return "tenant"
        start = index + 1
    return None


class _FixedHistogram:
    """Small cumulative host histogram; O(1) work and no device interaction."""

    def __init__(self, buckets):
        self.buckets = tuple(float(value) for value in buckets)
        self.bin_counts = [0] * (len(self.buckets) + 1)
        self.count = 0
        self.total = 0.0

    def observe(self, value) -> None:
        value = max(0.0, float(value))
        self.bin_counts[bisect_left(self.buckets, value)] += 1
        self.count += 1
        self.total += value

    def snapshot(self) -> dict:
        cumulative = []
        running = 0
        for count in self.bin_counts[:-1]:
            running += count
            cumulative.append(running)
        return {
            "buckets": self.buckets,
            "bucket_counts": tuple(cumulative),
            "count": self.count,
            "sum": self.total,
        }


class _CapsuleCapacityReservation:
    """Pins a transient fanout allocation against APCv2's resident budget."""

    def __init__(self, owner: "APCv2", nbytes: int):
        self._owner = owner
        self.nbytes = int(nbytes)
        self._closed = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        with self._owner._apc_lock:
            self._owner._capsule_reserved_bytes = max(
                0, self._owner._capsule_reserved_bytes - self.nbytes
            )


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


def _foreign_stream_error(exc: BaseException) -> bool:
    """MLX refused to evaluate arrays owned by another thread's stream."""
    return isinstance(exc, RuntimeError) and "in current thread" in str(exc)


_AUX_ARRAY_NAMES = frozenset({"tail_hidden", "rng_key"})


def _aux_array_layout(arrays) -> dict:
    """JSON-safe ``name -> [shape, dtype]`` of a spilled speculative aux file."""
    return {
        name: [[int(n) for n in array.shape], str(array.dtype)]
        for name, array in arrays.items()
    }


def _check_aux_array_layout(arrays, recorded, *, required: bool) -> None:
    """Refuse restored aux arrays that differ from what the spill wrote.

    Persisted manifests written before the layout was recorded carry none;
    those are still limited to the two array names a spill ever writes.
    """
    if set(arrays) - _AUX_ARRAY_NAMES:
        raise ValueError("APCv2 aux sidecar holds unexpected arrays")
    if recorded is None:
        if required:
            raise ValueError("APCv2 aux sidecar layout was not recorded")
        return
    if _aux_array_layout(arrays) != recorded:
        raise ValueError("APCv2 aux sidecar shape or dtype mismatch")


class APCv2(PrefixIndex):
    """APCv2: atomic segmented state ownership, prefix indexing and bounded residency."""

    _STAT_KEYS = (
        "lookups", "hits", "misses", "queried_tokens", "cached_tokens", "stores",
        "interior_hits", "rolling_hits", "junction_hits",
    )
    _DISK_STAT_KEYS = (
        "idle_spills",
        "pressure_spills",
        "restores",
        "restore_failures",
        "restore_budget_deferrals",
        "spill_failures",
        "oversize_spills",
        "spill_capacity_skips",
        "park_deferred_to_worker",
        "disk_evictions",
        "bytes_written",
        "bytes_read",
        "parks",
        "resumes",
        "prefetch_restores_ok",
        "prefetch_restores_abandoned",
        "prefetch_restores_failed",
        "prefetch_restores_cancelled",
        "prefetch_hits",
        "prefetch_misses",
        "session_deletes",
        "pin_expiries",
        "pin_cap_rejections",
        "publication_rejections",
        "persistence_io_failures",
        "persisted_writes",
        "quarantined",
        "quarantine_evictions",
        "restore_digest_failures",
    )
    _PERSIST_SCHEMA = "mlx2.apcv2.persisted-entry.v1"
    _SESSION_TAG_LIMIT = 16
    # Longest wait before an idle scan retries a snapshot that failed to spill.
    _SPILL_RETRY_MAX_SECONDS = 300.0
    _RETENTION_DEFAULT = "default"
    _RETENTION_INTERIOR = "interior_checkpoint"
    _RETENTION_PROMPT_BOUNDARY = "committed_prompt_boundary"
    # Disposable prefill progress points (state_boundaries.ROLLING): evicted
    # first, retired by their publisher once a later boundary supersedes them.
    _RETENTION_ROLLING = "prefill_rolling"
    _RETENTION_JUNCTION = "junction"
    _RETENTION_ROLES = frozenset(
        {
            _RETENTION_DEFAULT,
            _RETENTION_INTERIOR,
            _RETENTION_PROMPT_BOUNDARY,
            _RETENTION_ROLLING,
            _RETENTION_JUNCTION,
        }
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
        persistent_block_bytes: int = 0,
        now_fn=time.monotonic,
        wall_time_fn=time.time,
        persist_dir: Optional[str] = None,
        persist_identity: Optional[APCKey] = None,
        persist_semantic_namespace: Optional[str] = None,
        persist_corruption_action: str = "quarantine",
        session_max_ttl_seconds: int = 3600,
        pinned_disk_bytes_per_tenant: int = 64 << 30,
        pinned_disk_bytes_global: Optional[int] = None,
        pinned_resident_bytes_per_tenant: int = 12 << 30,
        prefetch_ttl_seconds: int = 30,
        quarantine_max_entries: int = 128,
        quarantine_max_bytes: int = 1 << 30,
        max_interior_entries: Optional[int] = None,
        generation_prompt_suffixes: Iterable[Iterable[int]] = (),
    ):
        if not layout_name:
            raise ValueError("APCv2 requires a model cache-layout declaration")
        super().__init__(max_size=max_size, max_bytes=max_bytes, max_tokens=max_tokens)
        # Interior checkpoints get their own resident-entry allowance.  Sharing
        # ``max_size`` with prompt boundaries and finished lanes, whose count
        # grows with every request, meant that once the cache was full a fresh
        # interior checkpoint (lowest retention rank, never yet reused) was
        # evicted by its own publication and could never earn a hit.  Bytes
        # stay one shared budget, where interiors are still evicted first.
        if max_interior_entries is None:
            max_interior_entries = max_size
        if (
            isinstance(max_interior_entries, bool)
            or not isinstance(max_interior_entries, int)
            or max_interior_entries < 0
        ):
            raise ValueError("max_interior_entries must be a non-negative integer")
        self.max_interior_entries = int(max_interior_entries)
        # Serving's detected generation-prompt suffixes (see
        # interior_placement).  A session resume uses them to find the
        # boundary a template that drops the suffix from history resumes at.
        self._generation_prompt_suffixes = tuple(
            tuple(int(token) for token in suffix)
            for suffix in generation_prompt_suffixes
        )
        self._apc_lock = threading.RLock()
        self._cow_branching = True
        self._cow_telemetry = COWCacheTelemetry()
        self._apc_stats = {key: 0 for key in self._STAT_KEYS}
        self._apc_lifetime = {key: 0 for key in self._STAT_KEYS}
        self._apc_clears = 0
        self._interior_reused_entries = 0
        age_buckets = (1, 5, 30, 120, 600, 3600)
        self._reuse_histograms = {
            "hit_age_seconds": _FixedHistogram(age_buckets),
            "eviction_age_seconds": _FixedHistogram(age_buckets),
            "eviction_idle_seconds": _FixedHistogram(age_buckets),
            "eviction_hit_count": _FixedHistogram((0, 1, 2, 5, 10)),
        }
        self._capsule_generation = CacheCapsuleGeneration()
        self._capsule_reserved_bytes = 0
        self._capsule_capacity = {
            "reservations": 0,
            "reservation_rejections": 0,
            "reserved_bytes_peak": 0,
        }
        self._idle_disk_seconds = max(0.0, float(idle_disk_seconds))
        if persist_dir and idle_disk_dir and Path(persist_dir).expanduser().resolve() != Path(idle_disk_dir).expanduser().resolve():
            raise ValueError("persist_dir and idle_disk_dir must name the same disk tier")
        disk_dir = persist_dir or idle_disk_dir
        if self._idle_disk_seconds > 0 and (not disk_dir):
            raise ValueError(
                "idle_disk_dir is required when idle_disk_seconds is enabled"
            )
        self._idle_disk_dir = (
            Path(disk_dir).expanduser().resolve()
            if disk_dir and self._idle_disk_seconds > 0
            else None
        )
        self._persist_dir = (
            Path(persist_dir).expanduser().resolve() if persist_dir else None
        )
        if self._persist_dir is not None and self._idle_disk_dir is None:
            raise ValueError("persist_dir requires a positive idle_disk_seconds")
        if persist_corruption_action not in {"quarantine", "delete"}:
            raise ValueError("persist_corruption_action must be quarantine or delete")
        self._persist_corruption_action = persist_corruption_action
        self._persist_identity = persist_identity
        self._persist_semantic_namespace = persist_semantic_namespace
        if self._persist_dir is not None and not isinstance(persist_identity, APCKey):
            raise ValueError("persistent APCv2 requires a complete identity template")
        for name, value in (
            ("session_max_ttl_seconds", session_max_ttl_seconds),
            ("pinned_disk_bytes_per_tenant", pinned_disk_bytes_per_tenant),
            ("pinned_resident_bytes_per_tenant", pinned_resident_bytes_per_tenant),
            ("prefetch_ttl_seconds", prefetch_ttl_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._session_max_ttl_seconds = int(session_max_ttl_seconds)
        self._pinned_disk_bytes_per_tenant = int(pinned_disk_bytes_per_tenant)
        self._pinned_resident_bytes_per_tenant = int(pinned_resident_bytes_per_tenant)
        self._prefetch_ttl_seconds = min(
            int(prefetch_ttl_seconds), self._session_max_ttl_seconds
        )
        self._idle_disk_max_bytes = max(0, int(idle_disk_max_bytes))
        configured_global_pin_cap = pinned_disk_bytes_global
        if pinned_disk_bytes_global is None:
            pinned_disk_bytes_global = self._idle_disk_max_bytes
        if (
            isinstance(pinned_disk_bytes_global, bool)
            or not isinstance(pinned_disk_bytes_global, int)
            or pinned_disk_bytes_global < 0
        ):
            raise ValueError("pinned_disk_bytes_global must be a non-negative integer")
        if (
            configured_global_pin_cap is not None
            and pinned_disk_bytes_global > self._idle_disk_max_bytes
        ):
            raise ValueError(
                "pinned_disk_bytes_global must not exceed the disk tier byte cap"
            )
        self._pinned_disk_bytes_global = int(pinned_disk_bytes_global)
        for name, value in (
            ("quarantine_max_entries", quarantine_max_entries),
            ("quarantine_max_bytes", quarantine_max_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self._quarantine_max_entries = int(quarantine_max_entries)
        self._quarantine_max_bytes = int(quarantine_max_bytes)
        self._quarantine = {"entries": 0, "bytes": 0, "evictions": 0}
        self._persistent_block_bytes = max(0, int(persistent_block_bytes))
        self._now = now_fn
        self._wall_time = wall_time_fn
        self._last_idle_scan = 0.0
        self._disk_stats = {key: 0 for key in self._DISK_STAT_KEYS}
        self._disk_bytes = 0
        self._rescan = {
            "registered": 0,
            "registered_bytes": 0,
            "discarded": {},
            "elapsed_seconds": 0.0,
        }
        self._persist_lock_file = None
        self._prefetch_slot = threading.BoundedSemaphore(1)
        self._pending_prefetch = None
        # Pin-cap message of the most recent spill that wrote a snapshot and
        # then had to discard it; read by callers right after a failed spill.
        self._spill_capacity_violation = None
        # (key, tokens) -> role of retirements a live lease deferred.
        self._pending_retirements = {}
        self._closed = False
        self._layer_segments = True
        self.layout_name = str(layout_name)
        if self._idle_disk_dir is not None:
            self._idle_disk_dir.mkdir(parents=True, exist_ok=True)
            self._secure_owned_path(self._idle_disk_dir, 0o700, directory=True)
        if self._persist_dir is not None:
            self._acquire_persist_lock()
            try:
                self._secure_persist_tree()
                self._rescan_persisted_entries()
            except BaseException:
                self._release_persist_lock()
                raise
        elif self._idle_disk_dir is not None:
            # Snapshots are process-local scratch.  Also sweep the dot-prefixed
            # temporaries ``_atomic_save_*`` leave behind if the process died
            # mid-spill; they sit outside the disk-tier accounting otherwise.
            for pattern in (
                "apc-idle-*.safetensors",
                ".apc-idle-*.tmp.safetensors",
                ".apc-idle-*.restore.safetensors",
            ):
                for path in self._idle_disk_dir.glob(pattern):
                    try:
                        path.unlink()
                    except OSError:
                        pass
            for directory in self._idle_disk_dir.glob(
                "apc-idle-*.safetensors.blocks"
            ):
                if directory.is_dir() and directory.parent == self._idle_disk_dir:
                    shutil.rmtree(directory, ignore_errors=True)
            # ``encode_block_file`` stages its block directory and manifest
            # under generated names; a kill mid-encode leaves both behind.
            # Match only those exact names, never a foreign file.
            for path in self._idle_disk_dir.glob(".apc-idle-*"):
                if path.is_symlink():
                    continue
                if path.is_dir() and _BLOCK_STAGING_DIRECTORY.fullmatch(path.name):
                    shutil.rmtree(path, ignore_errors=True)
                elif path.is_file() and _BLOCK_STAGING_MANIFEST.fullmatch(path.name):
                    try:
                        path.unlink()
                    except OSError:
                        pass

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _secure_owned_path(path: Path, mode: int, *, directory: bool) -> None:
        """Refuse foreign/symlinked persistence objects and tighten their mode."""
        info = path.lstat()
        expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        if path.is_symlink() or not expected:
            raise RuntimeError("APCv2 persistence contains an unsupported file type")
        if info.st_uid != os.getuid():
            raise RuntimeError("APCv2 persistence must be owned by the process uid")
        os.chmod(path, mode, follow_symlinks=False)

    def _secure_persist_tree(self) -> None:
        self._secure_owned_path(self._persist_dir, 0o700, directory=True)
        for root, directories, files in os.walk(self._persist_dir, followlinks=False):
            root_path = Path(root)
            for name in directories:
                self._secure_owned_path(root_path / name, 0o700, directory=True)
            for name in files:
                self._secure_owned_path(root_path / name, 0o600, directory=False)

    def _acquire_persist_lock(self) -> None:
        lock_path = self._persist_dir / ".mlx2-apcv2.lock"
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                os.close(descriptor)
                raise RuntimeError(
                    "APCv2 persistence lock must be a process-owned regular file"
                )
            os.fchmod(descriptor, 0o600)
            handle = os.fdopen(descriptor, "a+b")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as error:
            try:
                handle.close()
            except (NameError, OSError):
                pass
            raise RuntimeError(
                "APCv2 persistence lock is already owned or is not a safe file"
            ) from error
        self._persist_lock_file = handle

    def _release_persist_lock(self) -> None:
        if self._persist_lock_file is None:
            return
        try:
            fcntl.flock(self._persist_lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            self._persist_lock_file.close()
            self._persist_lock_file = None

    def _identity_matches(self, key: APCKey) -> bool:
        expected = self._persist_identity
        for field in (
            "model",
            "revision",
            "adapter",
            "tokenizer_fingerprint",
            "cache_layout_fingerprint",
        ):
            if getattr(key, field) != getattr(expected, field):
                return False
        semantic = key.semantic_fingerprint
        namespace = self._persist_semantic_namespace
        if namespace not in {"tenant", "shared"}:
            return semantic == expected.semantic_fingerprint
        # Int8 prefill wraps every serving namespace in the same numerics
        # revision, so the wrapper must match the template exactly: exact and
        # int8 state, or two int8 revisions, never adopt each other's entries.
        template = expected.semantic_fingerprint
        if _is_int8_semantic_wrapper(template):
            if not _is_int8_semantic_wrapper(semantic) or semantic[1:] != template[1:]:
                return False
            semantic = semantic[0]
        elif _is_int8_semantic_wrapper(semantic):
            return False
        return _serving_semantic_mode(semantic) == namespace

    @staticmethod
    def _strict_child(directory: Path, name: object) -> Path:
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("APCv2 manifest path is not a strict basename")
        result = directory / name
        if result.resolve().parent != directory.resolve():
            raise ValueError("APCv2 manifest path escapes persistence directory")
        return result

    def _payload_file_record(self, path: Path) -> dict:
        """Capture an immutable payload record once, immediately after writing."""
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                size += len(chunk)
                digest.update(chunk)
        return {
            "name": path.name,
            "size": size,
            "sha256": digest.hexdigest(),
        }

    @staticmethod
    def _manifest_file_records(disk: dict) -> dict:
        records = disk.get("files")
        if not isinstance(records, dict) or "target" not in records:
            raise RuntimeError("APCv2 persisted payload records are unavailable")
        return records

    def _manifest_for_entry(self, key, tokens, entry) -> dict:
        disk = getattr(entry, "_apc_disk", None) or {}
        tags = sorted(
            getattr(entry, "_apc_session_tags", set()), key=lambda item: repr(item)
        )
        disk_pins = getattr(entry, "_apc_disk_pin_expiries", {})
        resident_pins = getattr(entry, "_apc_resident_pin_expiries", {})
        token_bytes = json.dumps(list(tokens), separators=(",", ":")).encode("utf-8")
        return {
            "schema": self._PERSIST_SCHEMA,
            "key": _key_manifest(key),
            "layout_name": self.layout_name,
            "tokens": list(tokens),
            "tokens_sha256": hashlib.sha256(token_bytes).hexdigest(),
            "covered_tokens": len(tokens),
            "cache_type": entry.cache_type,
            "retention_role": getattr(
                entry, "_apc_retention_role", self._RETENTION_DEFAULT
            ),
            "session_tags": [[tag[0], tag[1]] for tag in tags],
            "disk_pins": [
                {"tenant": tag[0], "session_id": tag[1], "expires_at": expiry}
                for tag, expiry in sorted(disk_pins.items(), key=lambda item: repr(item[0]))
            ],
            "resident_pins": [
                {"tenant": tag[0], "session_id": tag[1], "expires_at": expiry}
                for tag, expiry in sorted(resident_pins.items(), key=lambda item: repr(item[0]))
            ],
            "sidecar": disk.get("sidecar"),
            "resident_nbytes": int(disk.get("resident_nbytes", 0)),
            "target_signature": disk.get("target_signature"),
            "identity_sha256": disk.get("identity_sha256"),
            "draft_signature": disk.get("draft_signature"),
            "files": self._manifest_file_records(disk),
            "created_at": float(getattr(entry, "_apc_created_wall", self._wall_time())),
            "last_access_at": float(getattr(entry, "_apc_last_access_wall", self._wall_time())),
            "hit_count": int(getattr(entry, "_apc_hit_count", 0)),
        }

    def _write_manifest_locked(self, key, tokens, entry) -> bool:
        if self._persist_dir is None or not getattr(entry, "_apc_disk", None):
            return True
        disk = entry._apc_disk
        manifest = Path(disk.get("manifest") or Path(disk["target"]).with_suffix(".manifest.json"))
        disk["manifest"] = str(manifest)
        temporary = manifest.with_name(f".{manifest.name}.{uuid.uuid4().hex}.tmp")
        try:
            payload = json.dumps(
                self._manifest_for_entry(key, tokens, entry),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            with temporary.open("xb") as handle:
                os.chmod(temporary, 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, manifest)
            os.chmod(manifest, 0o600)
            self._fsync_directory(self._persist_dir)
            self._disk_stats["persisted_writes"] += 1
            return True
        except Exception:
            self._disk_stats["persistence_io_failures"] += 1
            log.exception("APCv2 manifest update failed")
            return False
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                self._disk_stats["persistence_io_failures"] += 1

    def _verify_persisted_files(self, disk: dict) -> None:
        if self._persist_dir is None or not disk.get("manifest"):
            return
        records = disk.get("files") or {}
        for kind, record in records.items():
            path = Path(disk.get(kind, ""))
            try:
                size = path.stat().st_size
            except OSError as error:
                raise _RestoreDigestFailure("persisted APCv2 file is missing") from error
            if size != int(record.get("size", -1)):
                raise _RestoreDigestFailure("persisted APCv2 file size changed")
            if self._sha256_file(path) != record.get("sha256"):
                raise _RestoreDigestFailure("persisted APCv2 file digest mismatch")

    def _remove_manifest_files(self, manifest: Path, value: Optional[dict]) -> None:
        files = (value or {}).get("files", {}) if isinstance(value, dict) else {}
        for record in files.values() if isinstance(files, dict) else ():
            try:
                self._strict_child(self._persist_dir, record.get("name")).unlink(missing_ok=True)
            except (OSError, ValueError, AttributeError):
                pass
        try:
            manifest.unlink(missing_ok=True)
        except OSError:
            self._disk_stats["persistence_io_failures"] += 1

    def _enforce_quarantine_limits(self) -> None:
        quarantine = self._persist_dir / "quarantine"
        if not quarantine.exists():
            self._quarantine.update(entries=0, bytes=0)
            return
        records = []
        for path in quarantine.iterdir():
            try:
                if path.is_file() and not path.is_symlink():
                    info = path.stat()
                    records.append((info.st_mtime_ns, path.name, path, info.st_size))
            except OSError:
                self._disk_stats["persistence_io_failures"] += 1
        records.sort(key=lambda record: record[:2])
        total = sum(record[3] for record in records)
        while records and (
            len(records) > self._quarantine_max_entries
            or total > self._quarantine_max_bytes
        ):
            _mtime, _name, path, size = records.pop(0)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                self._disk_stats["persistence_io_failures"] += 1
                break
            total -= size
            self._quarantine["evictions"] += 1
            self._disk_stats["quarantine_evictions"] += 1
        self._quarantine.update(entries=len(records), bytes=max(0, total))

    def _discard_manifest(self, manifest: Path, value: Optional[dict], reason: str) -> None:
        discarded = self._rescan["discarded"]
        discarded[reason] = int(discarded.get(reason, 0)) + 1
        corrupt = reason not in {"identity_mismatch", "unknown_schema"}
        if corrupt and self._persist_corruption_action == "quarantine":
            quarantine = self._persist_dir / "quarantine"
            quarantine.mkdir(mode=0o700, exist_ok=True)
            self._secure_owned_path(quarantine, 0o700, directory=True)
            paths = [manifest]
            files = (value or {}).get("files", {}) if isinstance(value, dict) else {}
            for record in files.values() if isinstance(files, dict) else ():
                try:
                    paths.append(self._strict_child(self._persist_dir, record.get("name")))
                except (ValueError, AttributeError):
                    pass
            for path in paths:
                if not path.exists():
                    continue
                destination = quarantine / path.name
                if destination.exists():
                    destination = quarantine / f"{uuid.uuid4().hex}-{path.name}"
                try:
                    os.replace(path, destination)
                    os.chmod(destination, 0o600, follow_symlinks=False)
                except OSError:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        self._disk_stats["persistence_io_failures"] += 1
            self._disk_stats["quarantined"] += 1
            self._enforce_quarantine_limits()
        else:
            self._remove_manifest_files(manifest, value)

    def _rescan_persisted_entries(self) -> None:
        started = time.monotonic()
        referenced = set()
        now = self._wall_time()
        persisted_pinned_bytes = 0
        persisted_tenant_pinned_bytes = {}
        for temporary in self._persist_dir.glob(".*.tmp*"):
            if temporary.name != ".mlx2-apcv2.lock":
                if temporary.is_dir():
                    shutil.rmtree(temporary, ignore_errors=True)
                else:
                    temporary.unlink(missing_ok=True)
        for temporary in self._persist_dir.glob(".apc-idle-*.restore.safetensors"):
            temporary.unlink(missing_ok=True)
        for directory in self._persist_dir.glob("apc-idle-*.safetensors.blocks"):
            if directory.is_dir() and directory.parent == self._persist_dir:
                shutil.rmtree(directory, ignore_errors=True)
        self._enforce_quarantine_limits()
        for manifest in sorted(self._persist_dir.glob("apc-idle-*.manifest.json")):
            value = None
            try:
                value = json.loads(manifest.read_text(encoding="utf-8"))
                if value.get("schema") != self._PERSIST_SCHEMA:
                    self._discard_manifest(manifest, value, "unknown_schema")
                    continue
                key = _key_from_manifest(value.get("key"))
                if value.get("layout_name") != self.layout_name or not self._identity_matches(key):
                    self._discard_manifest(manifest, value, "identity_mismatch")
                    continue
                tokens = value.get("tokens")
                if (
                    not isinstance(tokens, list)
                    or any(isinstance(token, bool) or not isinstance(token, int) for token in tokens)
                    or int(value.get("covered_tokens", -1)) != len(tokens)
                ):
                    raise ValueError("invalid token path")
                token_bytes = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
                if hashlib.sha256(token_bytes).hexdigest() != value.get("tokens_sha256"):
                    raise ValueError("token digest mismatch")
                files = value.get("files")
                if not isinstance(files, dict) or "target" not in files:
                    raise ValueError("missing target file record")
                disk = {}
                disk_bytes = 0
                for kind, record in files.items():
                    if kind not in {"target", "draft", "aux"} or not isinstance(record, dict):
                        raise ValueError("invalid file record")
                    path = self._strict_child(self._persist_dir, record.get("name"))
                    if not path.is_file() or path.is_symlink() or path.stat().st_size != int(record.get("size", -1)):
                        raise ValueError("missing or wrong-sized persisted file")
                    referenced.add(path.name)
                    disk[kind] = str(path)
                    disk_bytes += path.stat().st_size
                role = value.get("retention_role", self._RETENTION_DEFAULT)
                cache_type = value.get("cache_type", "assistant")
                if role not in self._RETENTION_ROLES or cache_type not in self._lru._lrus:
                    raise ValueError("invalid retention or cache type")
                disk.update(
                    sidecar=value.get("sidecar"),
                    resident_nbytes=int(value.get("resident_nbytes", 0)),
                    target_signature=value.get("target_signature"),
                    identity_sha256=value.get("identity_sha256"),
                    token_count=len(tokens),
                    draft_signature=value.get("draft_signature"),
                    manifest=str(manifest),
                    files=files,
                )
                entry = PrefixIndex.CacheEntry([], 0, cache_type, None)
                entry._apc_disk = disk
                entry._apc_retention_role = role
                entry._apc_inserted_at = self._now()
                entry._apc_last_access_at = self._now()
                entry._apc_created_wall = float(value.get("created_at", now))
                entry._apc_last_access_wall = float(value.get("last_access_at", now))
                entry._apc_hit_count = int(value.get("hit_count", 0))
                tags = set()
                for tag in value.get("session_tags", ()):
                    if isinstance(tag, list) and len(tag) == 2 and isinstance(tag[1], str):
                        tags.add((tag[0], tag[1]))
                entry._apc_session_tags = set(list(tags)[: self._SESSION_TAG_LIMIT])
                for attr, name in (
                    ("_apc_disk_pin_expiries", "disk_pins"),
                    ("_apc_resident_pin_expiries", "resident_pins"),
                ):
                    pins = {}
                    for pin in value.get(name, ()):
                        if not isinstance(pin, dict):
                            continue
                        tag = (pin.get("tenant"), pin.get("session_id"))
                        expiry = float(pin.get("expires_at", 0))
                        if tag in entry._apc_session_tags and expiry > now:
                            pins[tag] = expiry
                        elif expiry:
                            self._disk_stats["pin_expiries"] += 1
                    setattr(entry, attr, pins)
                disk_pins = getattr(entry, "_apc_disk_pin_expiries", {})
                if disk_pins:
                    if (
                        persisted_pinned_bytes + disk_bytes
                        > min(
                            self._pinned_disk_bytes_global,
                            self._idle_disk_max_bytes,
                        )
                    ):
                        raise _PersistedPinCapacityError(
                            "persisted APCv2 pins exceed the configured global disk cap"
                        )
                    for tenant in {tag[0] for tag in disk_pins}:
                        tenant_total = int(
                            persisted_tenant_pinned_bytes.get(tenant, 0)
                        ) + disk_bytes
                        if tenant_total > self._pinned_disk_bytes_per_tenant:
                            raise _PersistedPinCapacityError(
                                "persisted APCv2 pins exceed a configured tenant disk cap"
                            )
                        persisted_tenant_pinned_bytes[tenant] = tenant_total
                    persisted_pinned_bytes += disk_bytes
                self._trie.add(key, tokens, entry)
                self._lru.push(key, tokens, cache_type)
                self._disk_bytes += disk_bytes
                self._rescan["registered"] += 1
                self._rescan["registered_bytes"] += disk_bytes
            except _PersistedPinCapacityError:
                raise
            except Exception:
                self._discard_manifest(manifest, value, "corruption")
        # Final payload names without a manifest are interrupted writes.
        for path in self._persist_dir.glob("apc-idle-*.safetensors"):
            if path.name not in referenced:
                path.unlink(missing_ok=True)
        self._enforce_entry_limits_locked()
        self._rescan["elapsed_seconds"] = time.monotonic() - started
        log.info(
            "APCv2 persistence rescan registered=%d bytes=%d discarded=%s elapsed=%.6fs",
            self._rescan["registered"],
            self._rescan["registered_bytes"],
            self._rescan["discarded"],
            self._rescan["elapsed_seconds"],
        )

    @property
    def capsule_generation(self) -> CacheCapsuleGeneration:
        """Generation authority for work captured at an APC lookup boundary."""
        return self._capsule_generation

    @staticmethod
    def capsule_compatibility_signature(key: APCKey, layout_name: str) -> tuple:
        """Exact identity bound into every transient cache capsule."""
        if not isinstance(key, APCKey):
            raise TypeError("cache capsules require an APCKey")
        values = (
            key.model, key.revision, key.adapter, key.tokenizer_fingerprint,
            key.cache_layout_fingerprint, key.semantic_fingerprint, str(layout_name),
        )
        if any(value is None for value in values):
            raise ValueError("cache capsules require a complete APC compatibility identity")
        return values

    def reserve_capsule_bytes(self, nbytes: int):
        """Reserve fanout bytes before a capsule can become visible.

        The reservation shares APCv2's hard resident cap.  Reclaim is limited
        to unleased entries and occurs before publication; callers then repeat
        the generation check because reclaim can invalidate their source.
        """
        required = int(nbytes)
        with self._apc_lock:
            available_limit = int(self.max_bytes) - self._capsule_reserved_bytes
            if required < 0 or required > available_limit:
                self._capsule_capacity["reservation_rejections"] += 1
                return None
            original_limit = int(self.max_bytes)
            # Resident enforcement already subtracts existing reservations.
            self.max_bytes = original_limit - required
            try:
                self._spill_resident_budget_locked(
                    exclude=None, include_exclude=True, hard_cap=original_limit
                )
            finally:
                self.max_bytes = original_limit
            if self._n_bytes > available_limit - required:
                self._capsule_capacity["reservation_rejections"] += 1
                return None
            self._capsule_reserved_bytes += required
            self._capsule_capacity["reservations"] += 1
            self._capsule_capacity["reserved_bytes_peak"] = max(
                self._capsule_capacity["reserved_bytes_peak"],
                self._capsule_reserved_bytes,
            )
            return _CapsuleCapacityReservation(self, required)

    def new_cache_capsule_pool(self, **kwargs) -> CacheCapsulePool:
        """Create the default-off fanout lane under APCv2 capacity ownership."""
        return CacheCapsulePool(
            self._capsule_generation,
            reserve=self.reserve_capsule_bytes,
            **kwargs,
        )

    @property
    def capsule_capacity_stats(self) -> dict:
        with self._apc_lock:
            return {
                **self._capsule_capacity,
                "reserved_bytes": self._capsule_reserved_bytes,
            }

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

    def _expire_entry_pins_locked(self, key, tokens, entry, *, now=None) -> None:
        now = self._wall_time() if now is None else float(now)
        changed = False
        for attr in ("_apc_disk_pin_expiries", "_apc_resident_pin_expiries"):
            pins = getattr(entry, attr, {})
            expired = [tag for tag, expiry in pins.items() if float(expiry) <= now]
            for tag in expired:
                del pins[tag]
                self._disk_stats["pin_expiries"] += 1
                changed = True
            setattr(entry, attr, pins)
        if changed and getattr(entry, "_apc_disk", None):
            self._write_manifest_locked(key, tokens, entry)

    def _entry_disk_pinned_locked(self, key, tokens, entry) -> bool:
        self._expire_entry_pins_locked(key, tokens, entry)
        return bool(getattr(entry, "_apc_disk_pin_expiries", {}))

    def _entry_resident_pinned_locked(self, key, tokens, entry) -> bool:
        self._expire_entry_pins_locked(key, tokens, entry)
        return bool(getattr(entry, "_apc_resident_pin_expiries", {}))

    @classmethod
    def _entry_retention_rank(cls, entry) -> int:
        role = getattr(entry, "_apc_retention_role", cls._RETENTION_DEFAULT)
        if role == cls._RETENTION_INTERIOR and int(
            getattr(entry, "_apc_hit_count", 0)
        ) > 0:
            # A reused interior checkpoint (typically a preamble shared across
            # sessions) has proven its value; evict it like ordinary entries
            # instead of first.
            return 1
        return {
            cls._RETENTION_ROLLING: -1,
            cls._RETENTION_INTERIOR: 0,
            cls._RETENTION_DEFAULT: 1,
            # A junction is where two observed conversations diverged; it is
            # retained like an ordinary exact entry.
            cls._RETENTION_JUNCTION: 1,
            cls._RETENTION_PROMPT_BOUNDARY: 2,
        }.get(role, 1)

    def _record_entry_hit_locked(self, entry) -> None:
        now = self._now()
        inserted = float(getattr(entry, "_apc_inserted_at", now))
        entry._apc_last_access_at = now
        entry._apc_last_access_wall = self._wall_time()
        entry._apc_hit_count = int(getattr(entry, "_apc_hit_count", 0)) + 1
        self._reuse_histograms["hit_age_seconds"].observe(now - inserted)
        role = getattr(entry, "_apc_retention_role", None)
        if role == self._RETENTION_INTERIOR:
            self._apc_stats["interior_hits"] += 1
            if entry._apc_hit_count == 1:
                self._interior_reused_entries = (
                    getattr(self, "_interior_reused_entries", 0) + 1
                )
        elif role == self._RETENTION_ROLLING:
            self._apc_stats["rolling_hits"] += 1
        elif role == self._RETENTION_JUNCTION:
            self._apc_stats["junction_hits"] += 1

    def _record_entry_eviction_locked(self, entry) -> None:
        now = self._now()
        inserted = float(getattr(entry, "_apc_inserted_at", now))
        last_access = float(getattr(entry, "_apc_last_access_at", inserted))
        hits = int(getattr(entry, "_apc_hit_count", 0))
        self._reuse_histograms["eviction_age_seconds"].observe(now - inserted)
        self._reuse_histograms["eviction_idle_seconds"].observe(now - last_access)
        self._reuse_histograms["eviction_hit_count"].observe(hits)

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
        # Within a retention rank, evict least recently accessed first; every
        # lookup, store and restore stamps ``_apc_last_access_at``.  The stable
        # sort retains PrefixIndex's mixed cache-type order for ties (which is
        # what a lookup-free workload or a frozen clock produces).
        records.sort(key=lambda record: record[:2])
        if include_exclude:
            excluded.sort(key=lambda record: record[:2])
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
            os.chmod(temporary, 0o600)
            APCv2._fsync_file(temporary)
            os.replace(temporary, path)
            os.chmod(path, 0o600)
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
            os.chmod(temporary, 0o600)
            APCv2._fsync_file(temporary)
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass

    @staticmethod
    def _disk_paths(entry) -> tuple[Path, ...]:
        disk = getattr(entry, "_apc_disk", None) or {}
        paths = []
        for raw in (disk.get("target"), disk.get("draft"), disk.get("aux")):
            if raw:
                paths.extend(block_file_paths(Path(raw)))
        return tuple(paths)

    def _remove_disk_files_locked(self, entry) -> None:
        disk = getattr(entry, "_apc_disk", None) or {}
        for raw in (disk.get("target"), disk.get("draft"), disk.get("aux")):
            if raw:
                try:
                    removed = remove_block_file(Path(raw))
                except OSError:
                    self._disk_stats["persistence_io_failures"] += 1
                    log.exception("APCv2 payload removal failed")
                else:
                    self._disk_bytes = max(0, self._disk_bytes - removed)
        manifest = disk.get("manifest")
        if manifest:
            try:
                Path(manifest).unlink(missing_ok=True)
            except OSError:
                self._disk_stats["persistence_io_failures"] += 1
                log.exception("APCv2 manifest removal failed")
            if self._persist_dir is not None:
                try:
                    self._fsync_directory(self._persist_dir)
                except OSError:
                    self._disk_stats["persistence_io_failures"] += 1
        entry._apc_disk = None

    def _persistent_signature(self, key, tokens, cache_type, plane: str) -> str:
        identity = (
            "mlx2-apcv2-blocks-v1", self.layout_name, key, tuple(tokens),
            str(cache_type), str(plane),
        )
        return repr(identity)

    @staticmethod
    def _persistent_identity_digest(signature: str) -> str:
        return hashlib.sha256(signature.encode("utf-8")).hexdigest()

    def _drop_entry_locked(self, key, tokens, entry) -> None:
        current = self._trie.pop(key, tokens)
        if current is None:
            return
        self._record_entry_eviction_locked(current)
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

    def _spill_entry_locked(
        self, key, tokens, entry, *, reason: str, hard_cap: Optional[int] = None,
        keep_resident: bool = False,
    ) -> bool:
        self._spill_capacity_violation = None
        if self._idle_disk_dir is None or not entry.prompt_cache:
            return False
        if self._entry_pinned(entry):
            return False
        resident_cap = int(self.max_bytes if hard_cap is None else hard_cap)
        if not keep_resident and int(entry.nbytes) > resident_cap:
            # Disk-only checkpoints may become restorable after a cap change,
            # but they cannot be used under the current resident budget.  This
            # matters most for full-context snapshots whose spills are huge.
            self._disk_stats["oversize_spills"] += 1
            log.warning(
                "APCv2 spilling %d-token checkpoint (%d bytes) above "
                "current resident cap (%d bytes); reuse requires a larger cap",
                len(tokens), int(entry.nbytes), resident_cap,
            )
        disk = getattr(entry, "_apc_disk", None)
        if (
            not disk
            and self._disk_bytes >= self._idle_disk_max_bytes
            and not self._entry_disk_pinned_locked(key, tokens, entry)
        ):
            # With a full disk tier, a new checkpoint below every retained
            # disk entry's role is guaranteed to be its own first eviction.
            # Skip the expensive serialize/fsync/delete cycle; callers may
            # still evict this unpinned resident entry to make progress.
            disk_candidates = [
                record
                for record in self._retention_ordered_candidates_locked(
                    resident_only=False, exclude=entry
                )
                if getattr(record[4], "_apc_disk", None)
            ]
            if not disk_candidates or self._entry_retention_rank(entry) < min(
                record[0] for record in disk_candidates
            ):
                self._disk_stats["spill_capacity_skips"] += 1
                return False
        if not disk:
            stem = f"apc-idle-{uuid.uuid4().hex}"
            target = self._idle_disk_dir / f"{stem}.target.safetensors"
            draft = self._idle_disk_dir / f"{stem}.draft.safetensors"
            aux = self._idle_disk_dir / f"{stem}.aux.safetensors"
            created = []
            file_records = {}
            try:
                self._atomic_save_cache(target, entry.prompt_cache)
                created.append(target)
                if self._persist_dir is not None:
                    file_records["target"] = self._payload_file_record(target)
                target_signature = self._persistent_signature(
                    key, tokens, entry.cache_type, "target"
                )
                if self._persist_dir is None:
                    encode_block_file(
                        target,
                        block_bytes=self._persistent_block_bytes,
                        signature=target_signature,
                    )
                sidecar = entry.sidecar
                sidecar_info = None
                if sidecar is not None:
                    (draft_cache, tail_hidden) = sidecar.state
                    self._atomic_save_cache(draft, draft_cache)
                    created.append(draft)
                    if self._persist_dir is not None:
                        file_records["draft"] = self._payload_file_record(draft)
                    draft_signature = self._persistent_signature(
                        key, tokens, entry.cache_type, "draft"
                    )
                    if self._persist_dir is None:
                        encode_block_file(
                            draft,
                            block_bytes=self._persistent_block_bytes,
                            signature=draft_signature,
                        )
                    arrays = {}
                    if tail_hidden is not None:
                        arrays["tail_hidden"] = tail_hidden
                    if sidecar.rng_key is not None:
                        arrays["rng_key"] = sidecar.rng_key
                    if arrays:
                        self._atomic_save_arrays(aux, arrays)
                        created.append(aux)
                        if self._persist_dir is not None:
                            file_records["aux"] = self._payload_file_record(aux)
                        else:
                            # The aux file carries the seeded lane's RNG key,
                            # so it gets the same MAC'd manifest as target and
                            # draft; a plain file here could be swapped freely.
                            encode_block_file(
                                aux,
                                block_bytes=self._persistent_block_bytes,
                                signature=self._persistent_signature(
                                    key, tokens, entry.cache_type, "aux"
                                ),
                            )
                    sidecar_info = {
                        "covered_tokens": int(sidecar.covered_tokens),
                        "rng_draws": int(sidecar.rng_draws),
                        "kind": getattr(sidecar, "kind", "self_mtp"),
                        "binding": getattr(sidecar, "binding", ""),
                        "aux_arrays": _aux_array_layout(arrays),
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
                    "target_signature": target_signature,
                    "identity_sha256": self._persistent_identity_digest(
                        target_signature
                    ),
                    "token_count": len(tokens),
                    "draft_signature": (
                        draft_signature if draft in created else None
                    ),
                    "files": file_records if self._persist_dir is not None else None,
                    # Target, draft and aux were replaced by MAC'd block
                    # manifests; restore must refuse anything else there.
                    "block_encoded": (
                        self._persist_dir is None
                        and int(self._persistent_block_bytes) > 0
                    ),
                }
                entry._apc_disk = disk
                written = (
                    sum(int(record["size"]) for record in file_records.values())
                    if self._persist_dir is not None
                    else sum(
                        path.stat().st_size
                        for root in created
                        for path in block_file_paths(root)
                    )
                )
                self._disk_bytes += int(written)
                if self._persist_dir is not None:
                    if not self._write_manifest_locked(key, tokens, entry):
                        raise OSError("APCv2 manifest publication failed")
                violation = self._pin_limit_violation_locked()
                if violation is not None:
                    raise APCSessionCapacityError(violation)
                # Count only snapshots that stay published, never the ones a
                # failed publication step discards below.
                self._disk_stats["bytes_written"] += int(written)
            except Exception as exc:
                if isinstance(exc, APCSessionCapacityError):
                    self._spill_capacity_violation = str(exc)
                if reason == "park" and _foreign_stream_error(exc):
                    # MLX streams are thread-local: a park arriving on an HTTP
                    # thread cannot save arrays the generation worker's
                    # stream produced.  The entry stays park-pending and the
                    # worker's next idle scan (forced below) spills it on the
                    # owning thread.  Not a failure.
                    self._disk_stats["park_deferred_to_worker"] += 1
                else:
                    self._disk_stats["spill_failures"] += 1
                if getattr(entry, "_apc_disk", None):
                    self._remove_disk_files_locked(entry)
                else:
                    for path in created:
                        try:
                            remove_block_file(path)
                        except OSError:
                            self._disk_stats["persistence_io_failures"] += 1
                return False
        if keep_resident:
            # Persist only: the caller keeps serving the resident copy and
            # needs a crash-safe snapshot of exactly this content.
            return True
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
        if self._persist_dir is not None:
            self._write_manifest_locked(key, tokens, entry)
        if reason == "idle":
            self._disk_stats["idle_spills"] += 1
        elif reason == "pressure":
            self._disk_stats["pressure_spills"] += 1
        self._capsule_generation.advance()
        return True

    @staticmethod
    def _restored_resident_nbytes(cache, sidecar) -> int:
        return sum((int(getattr(item, "nbytes", 0)) for item in cache)) + int(
            getattr(sidecar, "nbytes", 0)
        )

    @property
    def _resident_entry_budget(self) -> int:
        """Bytes available to stored entries while fanout owners are live."""
        return max(0, int(self.max_bytes) - self._capsule_reserved_bytes)

    def _reserve_restore_bytes_locked(self, required: int, *, entry) -> bool:
        """Reserve a hard-capped resident budget before publishing a restore."""
        required = int(required)
        original_limit = int(self.max_bytes)
        if required < 0 or required > original_limit:
            # Impossible under any occupancy: the snapshot itself exceeds the
            # cap.  Callers drop it; a transient shortage returns False below.
            raise ValueError("APCv2 snapshot exceeds the resident byte cap")
        if required > self._resident_entry_budget:
            # Capsules are temporary consumers; keep the healthy disk entry
            # available for a later restore after their reservations release.
            return False
        temporary_limit = original_limit - required
        self.max_bytes = temporary_limit
        try:
            self._spill_resident_budget_locked(
                exclude=entry, include_exclude=False, hard_cap=original_limit
            )
        finally:
            self.max_bytes = original_limit
        return self._n_bytes <= temporary_limit - self._capsule_reserved_bytes

    def _restore_entry_locked(self, key, tokens, entry) -> bool:
        disk = getattr(entry, "_apc_disk", None) or {}
        target = disk.get("target")
        if not target:
            return False
        cache = None
        try:
            expected = int(disk.get("resident_nbytes", 0) or 0)
            if expected < 0 or expected > int(self.max_bytes):
                raise ValueError("APCv2 snapshot exceeds the resident byte cap")
            expected_signature = self._persistent_signature(
                key, tokens, entry.cache_type, "target"
            )
            if (
                disk.get("target_signature") != expected_signature
                or disk.get("identity_sha256")
                != self._persistent_identity_digest(expected_signature)
                or int(disk.get("token_count", -1)) != len(tokens)
            ):
                raise ValueError("APCv2 persistent identity or token count mismatch")
            self._verify_persisted_files(disk)
            # The recorded size is the cold-allocation admission estimate.  An
            # impossible snapshot must fail before disk I/O; clamping the
            # temporary limit to zero used to let it publish over max_bytes.
            if not self._reserve_restore_bytes_locked(expected, entry=entry):
                raise _RestoreBudgetUnavailable("APCv2 disk snapshot exceeds resident byte cap")
            with ExitStack() as stack:
                target_path = stack.enter_context(materialize_block_file(
                    Path(target),
                    expected_signature=disk.get("target_signature", ""),
                    require_manifest=bool(disk.get("block_encoded")),
                ))
                cache = load_prompt_cache(str(target_path))
                sidecar = None
                sidecar_info = disk.get("sidecar")
                if sidecar_info is not None:
                    draft_path = stack.enter_context(materialize_block_file(
                        Path(disk["draft"]),
                        expected_signature=disk.get("draft_signature", ""),
                        require_manifest=bool(disk.get("block_encoded")),
                    ))
                    draft = load_prompt_cache(str(draft_path))
                    arrays = {}
                    if disk.get("aux"):
                        aux_path = stack.enter_context(materialize_block_file(
                            Path(disk["aux"]),
                            expected_signature=self._persistent_signature(
                                key, tokens, entry.cache_type, "aux"
                            ),
                            require_manifest=bool(disk.get("block_encoded")),
                        ))
                        arrays = mx.load(str(aux_path))
                    _check_aux_array_layout(
                        arrays,
                        sidecar_info.get("aux_arrays"),
                        required=bool(disk.get("block_encoded")),
                    )
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
                raise _RestoreBudgetUnavailable("restored APCv2 snapshot exceeds resident byte cap")
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
                raise _RestoreBudgetUnavailable("frozen APCv2 snapshot exceeds resident byte cap")
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
        except _RestoreBudgetUnavailable:
            # Healthy snapshot, no room: leased residents pin the budget.  Keep
            # the placeholder and its files; the caller reports a miss.
            if isinstance(cache, COWFrozenPromptCache):
                cache.close()
            self._disk_stats["restore_budget_deferrals"] += 1
            return None
        except _RestoreDigestFailure:
            if isinstance(cache, COWFrozenPromptCache):
                cache.close()
            self._disk_stats["restore_digest_failures"] += 1
            self._disk_stats["restore_failures"] += 1
            return False
        except Exception:
            log.exception(
                "APCv2 disk restore failed for %d-token checkpoint",
                len(tokens),
            )
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
            if self._entry_disk_pinned_locked(key, tokens, entry):
                continue
            if entry.prompt_cache:
                self._remove_disk_files_locked(entry)
            else:
                self._drop_entry_locked(key, tokens, entry)
            self._disk_stats["disk_evictions"] += 1

    def _spill_resident_budget_locked(
        self, *, exclude=None, include_exclude: bool = True,
        hard_cap: Optional[int] = None,
    ) -> int:
        if self._idle_disk_dir is None or self._n_bytes <= self._resident_entry_budget:
            return 0
        spilled = 0
        while self._n_bytes > self._resident_entry_budget:
            records = self._pressure_candidates_locked(
                exclude=exclude, include_exclude=include_exclude
            )
            if not records:
                break
            (_rank, _last_access, key, tokens, entry) = records[0]
            if key is None or tokens is None:
                break
            if not self._spill_entry_locked(
                key, tokens, entry, reason="pressure", hard_cap=hard_cap
            ):
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

    def _resident_entry_count_locked(self, *, interior: Optional[bool] = None) -> int:
        """Resident entries; ``interior`` selects one count pool (None: all)."""
        return sum(
            bool(entry.prompt_cache)
            and (interior is None or self._is_interior_entry(entry) == interior)
            for _key, _tokens, entry in self._entry_records_locked()
        )

    @classmethod
    def _is_interior_entry(cls, entry) -> bool:
        return (
            getattr(entry, "_apc_retention_role", cls._RETENTION_DEFAULT)
            == cls._RETENTION_INTERIOR
        )

    def _count_pools_fit_locked(self) -> bool:
        return (
            self._resident_entry_count_locked(interior=False) <= self.max_size
            and self._resident_entry_count_locked(interior=True)
            <= self.max_interior_entries
        )

    def _enforce_count_pool_locked(self, *, interior: bool, limit: int) -> None:
        """Spill or drop within one count pool, in retention order."""
        while self._resident_entry_count_locked(interior=interior) > limit:
            records = [
                record
                for record in self._retention_ordered_candidates_locked(
                    resident_only=True,
                    exclude=None,
                    include_exclude=True,
                )
                if self._is_interior_entry(record[4]) == interior
            ]
            if not records:
                break
            progressed = False
            for _rank, _last_access, key, tokens, entry in records:
                if self._idle_disk_dir is not None and self._spill_entry_locked(
                    key, tokens, entry, reason="pressure"
                ):
                    progressed = True
                    break
                if self._entry_disk_pinned_locked(key, tokens, entry):
                    continue
                self._drop_entry_locked(key, tokens, entry)
                progressed = True
                break
            if not progressed:
                break

    def _enforce_entry_limits_locked(self, *, publication=None) -> bool:
        """Apply APC count/resident limits after the retention role is visible."""
        publication_rejected = False
        self._enforce_count_pool_locked(interior=False, limit=self.max_size)
        self._enforce_count_pool_locked(
            interior=True, limit=self.max_interior_entries
        )
        # Capsule reservations are hard-reserved against the same cap.
        original_limit = int(self.max_bytes)
        if self._idle_disk_dir is not None:
            self._spill_resident_budget_locked(
                exclude=None, include_exclude=True, hard_cap=original_limit
            )
        # A failed/unavailable spill must not turn a retention preference into
        # permission to exceed the hard resident cap.
        while self._n_bytes > self._resident_entry_budget:
            records = self._pressure_candidates_locked(
                exclude=None, include_exclude=True
            )
            if not records:
                break
            victim = next(
                (
                    record
                    for record in records
                    if not self._entry_disk_pinned_locked(
                        record[2], record[3], record[4]
                    )
                ),
                None,
            )
            if victim is None:
                break
            _rank, _last_access, key, tokens, entry = victim
            self._drop_entry_locked(key, tokens, entry)
        fits = self._count_pools_fit_locked() and self._n_bytes <= self._resident_entry_budget
        if not fits and publication is not None:
            key, tokens, entry = publication
            try:
                current = self._trie.get(key, tokens)
            except KeyError:
                current = None
            if current is entry:
                self._drop_entry_locked(key, tokens, entry)
                self._disk_stats["publication_rejections"] += 1
                publication_rejected = True
            fits = self._count_pools_fit_locked() and self._n_bytes <= self._resident_entry_budget
        self._enforce_disk_limit_locked()
        if publication is not None and not publication_rejected:
            key, tokens, entry = publication
            try:
                published = self._trie.get(key, tokens) is entry
            except KeyError:
                published = False
            if not published:
                self._disk_stats["publication_rejections"] += 1
                publication_rejected = True
        return fits and not publication_rejected

    def _validate_session_ttl(self, value, *, optional=False) -> int:
        if value is None and optional:
            return self._prefetch_ttl_seconds
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("ttl_seconds must be an integer")
        if not 1 <= value <= self._session_max_ttl_seconds:
            raise ValueError(
                f"ttl_seconds must be between 1 and {self._session_max_ttl_seconds}"
            )
        return int(value)

    def _session_entries_locked(self, tag: tuple):
        return [
            (key, tokens, entry)
            for key, tokens, entry in self._entry_records_locked()
            if tag in getattr(entry, "_apc_session_tags", set())
        ]

    def _entry_disk_bytes(self, entry) -> int:
        total = 0
        for path in self._disk_paths(entry):
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def _entry_projected_disk_bytes(self, entry) -> int:
        actual = self._entry_disk_bytes(entry)
        if actual:
            return actual
        disk = getattr(entry, "_apc_disk", None) or {}
        return int(disk.get("resident_nbytes", 0) or entry.nbytes)

    def _tenant_pin_bytes_locked(self, tenant, *, resident: bool) -> int:
        total = 0
        attr = (
            "_apc_resident_pin_expiries" if resident else "_apc_disk_pin_expiries"
        )
        seen = set()
        for key, tokens, entry in self._entry_records_locked():
            self._expire_entry_pins_locked(key, tokens, entry)
            if id(entry) in seen:
                continue
            pins = getattr(entry, attr, {})
            if any(tag[0] == tenant for tag in pins):
                seen.add(id(entry))
                total += (
                    int((getattr(entry, "_apc_disk", None) or {}).get("resident_nbytes", entry.nbytes))
                    if resident
                    else self._entry_projected_disk_bytes(entry)
                )
        return total

    def _global_pin_bytes_locked(self) -> int:
        total = 0
        seen = set()
        for key, tokens, entry in self._entry_records_locked():
            self._expire_entry_pins_locked(key, tokens, entry)
            if id(entry) in seen or not getattr(entry, "_apc_disk_pin_expiries", {}):
                continue
            seen.add(id(entry))
            total += self._entry_projected_disk_bytes(entry)
        return total

    def _pin_limit_violation_locked(self) -> Optional[str]:
        global_pinned = self._global_pin_bytes_locked()
        if global_pinned > self._pinned_disk_bytes_global:
            return "global pinned disk byte cap would be exceeded"
        if global_pinned > self._idle_disk_max_bytes:
            return "disk tier byte cap would be exceeded by pinned sessions"
        tenants = {
            tag[0]
            for _key, _tokens, entry in self._entry_records_locked()
            for tag in getattr(entry, "_apc_disk_pin_expiries", {})
        }
        if any(
            self._tenant_pin_bytes_locked(tenant, resident=False)
            > self._pinned_disk_bytes_per_tenant
            for tenant in tenants
        ):
            return "per-tenant pinned disk byte cap would be exceeded"
        return None

    def _session_state_locked(self, tag: tuple) -> dict:
        records = self._session_entries_locked(tag)
        if not records:
            raise APCSessionNotFound(tag[1])
        now = self._wall_time()
        for key, tokens, entry in records:
            self._expire_entry_pins_locked(key, tokens, entry, now=now)
        deepest = max(records, key=lambda record: len(record[1]))
        _key, deepest_tokens, deepest_entry = deepest
        resident_entries = sum(bool(entry.prompt_cache) for _, _, entry in records)
        disk_entries = sum(bool(getattr(entry, "_apc_disk", None)) for _, _, entry in records)
        disk_expiries = [
            getattr(entry, "_apc_disk_pin_expiries", {}).get(tag)
            for _, _, entry in records
        ]
        resident_expiries = [
            getattr(entry, "_apc_resident_pin_expiries", {}).get(tag)
            for _, _, entry in records
        ]
        return {
            "tenant": tag[0],
            "session_id": tag[1],
            "state": "resident" if deepest_entry.prompt_cache else (
                "disk" if getattr(deepest_entry, "_apc_disk", None) else "missing"
            ),
            "covered_tokens": len(deepest_tokens),
            "entries": len(records),
            "resident_entries": resident_entries,
            "disk_entries": disk_entries,
            "resident_bytes": sum(int(entry.nbytes) for _, _, entry in records),
            "disk_bytes": sum(self._entry_disk_bytes(entry) for _, _, entry in records),
            "disk_pin_expires_at": max((value for value in disk_expiries if value), default=None),
            "resident_pin_expires_at": max((value for value in resident_expiries if value), default=None),
            "park_pending": sum(bool(getattr(entry, "_apc_park_pending", False)) for _, _, entry in records),
            "delete_pending": sum(bool(getattr(entry, "_apc_delete_pending", False)) for _, _, entry in records),
        }

    def session_state(self, tenant, session_id: str) -> dict:
        with self._apc_lock:
            return self._session_state_locked((tenant, session_id))

    def list_sessions(self, tenant, *, limit: int = 50, cursor: int = 0) -> dict:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("cursor must be a non-negative integer")
        with self._apc_lock:
            session_ids = sorted(
                {
                    tag[1]
                    for _key, _tokens, entry in self._entry_records_locked()
                    for tag in getattr(entry, "_apc_session_tags", set())
                    if tag[0] == tenant
                }
            )
            page = session_ids[cursor : cursor + limit]
            return {
                "data": [self._session_state_locked((tenant, value)) for value in page],
                "next_cursor": (
                    cursor + len(page) if cursor + len(page) < len(session_ids) else None
                ),
            }

    def park_session(self, tenant, session_id: str, *, ttl_seconds: int) -> dict:
        ttl = self._validate_session_ttl(ttl_seconds)
        tag = (tenant, session_id)
        with self._apc_lock:
            records = self._session_entries_locked(tag)
            if not records:
                raise APCSessionNotFound(session_id)
            current = self._tenant_pin_bytes_locked(tenant, resident=False)
            incremental = sum(
                self._entry_projected_disk_bytes(entry)
                for _key, _tokens, entry in records
                if not any(
                    pin_tag[0] == tenant
                    for pin_tag in getattr(entry, "_apc_disk_pin_expiries", {})
                )
            )
            if current + incremental > self._pinned_disk_bytes_per_tenant:
                self._disk_stats["pin_cap_rejections"] += 1
                raise APCSessionCapacityError(
                    "per-tenant pinned disk byte cap would be exceeded"
                )
            global_current = self._global_pin_bytes_locked()
            global_incremental = sum(
                self._entry_projected_disk_bytes(entry)
                for _key, _tokens, entry in records
                if not getattr(entry, "_apc_disk_pin_expiries", {})
            )
            if global_current + global_incremental > self._pinned_disk_bytes_global:
                self._disk_stats["pin_cap_rejections"] += 1
                raise APCSessionCapacityError(
                    "global pinned disk byte cap would be exceeded"
                )
            if global_current + global_incremental > self._idle_disk_max_bytes:
                self._disk_stats["pin_cap_rejections"] += 1
                raise APCSessionCapacityError(
                    "disk tier byte cap would be exceeded by pinned sessions"
                )
            expiry = self._wall_time() + ttl
            spilled = 0
            spill_violation = None
            previous = []
            for key, tokens, entry in records:
                resident_pins = getattr(entry, "_apc_resident_pin_expiries", {})
                pins = getattr(entry, "_apc_disk_pin_expiries", {})
                previous.append(
                    (
                        pins.get(tag),
                        resident_pins.get(tag),
                        getattr(entry, "_apc_park_pending", False),
                    )
                )
                resident_pins.pop(tag, None)
                pins[tag] = expiry
                entry._apc_disk_pin_expiries = pins
                if entry.prompt_cache:
                    if self._entry_pinned(entry):
                        entry._apc_park_pending = True
                    elif self._spill_entry_locked(key, tokens, entry, reason="park"):
                        spilled += 1
                    elif self._spill_capacity_violation is not None:
                        # The written snapshot outgrew the projected cost the
                        # pin was admitted with; no later retry can fit it.
                        spill_violation = self._spill_capacity_violation
                        break
                    else:
                        entry._apc_park_pending = True
                if getattr(entry, "_apc_disk", None):
                    self._write_manifest_locked(key, tokens, entry)
            if spilled:
                mx.clear_cache()
            if any(getattr(entry, "_apc_park_pending", False) for _k, _t, entry in records):
                # Let the worker's next idle scan complete the park at once.
                self._last_idle_scan = float("-inf")
            violation = spill_violation or self._pin_limit_violation_locked()
            if violation is not None:
                # Fail closed and restore exactly what this park changed.
                for (key, tokens, entry), (disk_pin, resident_pin, pending) in zip(
                    records, previous
                ):
                    pins = getattr(entry, "_apc_disk_pin_expiries", {})
                    if disk_pin is None:
                        pins.pop(tag, None)
                    else:
                        pins[tag] = disk_pin
                    if resident_pin is not None:
                        entry._apc_resident_pin_expiries[tag] = resident_pin
                    entry._apc_park_pending = pending
                    if getattr(entry, "_apc_disk", None):
                        self._write_manifest_locked(key, tokens, entry)
                self._disk_stats["pin_cap_rejections"] += 1
                self._enforce_disk_limit_locked()
                raise APCSessionCapacityError(violation)
            self._disk_stats["parks"] += 1
            self._enforce_disk_limit_locked()
            return self._session_state_locked(tag)

    def _prefetch_entry_locked(self, tag, key, tokens, entry, expiry) -> bool:
        """Make one session entry resident and pin it until ``expiry``."""
        if entry.prompt_cache:
            restored = True
        else:
            # This method is serviced only by the generation worker at
            # its idle boundary. MLX documents cross-thread streams as
            # caller-serialized; APC locking alone cannot serialize
            # their graph evaluation against decode on another thread.
            restored = self._restore_entry_locked(key, tokens, entry)
        if restored is None:
            self._disk_stats["prefetch_restores_abandoned"] += 1
            return False
        if not restored:
            self._disk_stats["prefetch_restores_failed"] += 1
            self._drop_entry_locked(key, tokens, entry)
            return False
        current = self._tenant_pin_bytes_locked(tag[0], resident=True)
        if (
            not any(
                pin_tag[0] == tag[0]
                for pin_tag in getattr(entry, "_apc_resident_pin_expiries", {})
            )
            and current + int(entry.nbytes) > self._pinned_resident_bytes_per_tenant
        ):
            self._disk_stats["pin_cap_rejections"] += 1
            self._disk_stats["prefetch_restores_abandoned"] += 1
            return False
        pins = getattr(entry, "_apc_resident_pin_expiries", {})
        pins[tag] = expiry
        entry._apc_resident_pin_expiries = pins
        self._disk_stats["prefetch_restores_ok"] += 1
        if getattr(entry, "_apc_disk", None):
            self._write_manifest_locked(key, tokens, entry)
        return True

    def _session_turn_boundaries_locked(self, tag, key, tokens, tip):
        """Exact boundaries the session's latest turn published below its tip.

        ``tokens`` is the tip's path.  The deepest committed prompt boundary
        on it is that turn's ``P-1`` boundary, which serves a re-send of the
        prompt.  For each generation-prompt suffix the prompt ends with, the
        entry at ``P - len(suffix)`` is where the next turn resumes when the
        template re-renders the finished turn without that suffix.
        """
        on_path = {}
        for record_key, path, entry in self._session_entries_locked(tag):
            if (
                record_key == key
                and len(path) < len(tokens)
                and path == tokens[: len(path)]
            ):
                on_path[len(path)] = (record_key, path, entry)
        anchors = [
            depth
            for depth, (_key, _path, entry) in on_path.items()
            if getattr(entry, "_apc_retention_role", self._RETENTION_DEFAULT)
            == self._RETENTION_PROMPT_BOUNDARY
        ]
        if (
            getattr(tip, "_apc_retention_role", self._RETENTION_DEFAULT)
            == self._RETENTION_PROMPT_BOUNDARY
        ):
            anchors.append(len(tokens))
        if not anchors:
            return []
        boundary = max(anchors)
        prompt_end = boundary + 1
        depths = [boundary]
        for suffix in self._generation_prompt_suffixes:
            start = prompt_end - len(suffix)
            # A ``P-1`` tip lacks the prompt's last token; match what it has.
            window = tokens[start:prompt_end]
            if start > 0 and window == list(suffix[: len(window)]):
                depths.append(start)
        return [on_path[depth] for depth in depths if depth in on_path]

    @staticmethod
    def _tip_serves_depth(tip, tip_tokens, entry, depth) -> bool:
        """Whether a resident tip already serves what ``entry`` would.

        True only when the tip can land exactly at ``depth``.  A draft
        sidecar is state at its own entry's end: a sidecar-bearing tip cannot
        carry it back to ``depth``, and a sidecar-bearing boundary is what a
        speculative route selects over a trimmed target-only tip.
        """
        if (
            getattr(tip, "sidecar", None) is not None
            or getattr(entry, "sidecar", None) is not None
            or (getattr(entry, "_apc_disk", None) or {}).get("sidecar") is not None
        ):
            return False
        if can_trim_prompt_cache(tip.prompt_cache):
            return True
        if achievable_trim is None:
            return False
        landing = achievable_trim(tip.prompt_cache, len(tip_tokens) - depth)
        return landing is not None and landing[0] == depth

    def _prefetch_session(self, tag, key, tokens, entry, expiry) -> None:
        try:
            with self._apc_lock:
                if (
                    tag not in getattr(entry, "_apc_session_tags", set())
                    or entry is not self._trie.get(key, tokens)
                ):
                    self._disk_stats["prefetch_restores_abandoned"] += 1
                    return
                if not self._prefetch_entry_locked(tag, key, tokens, entry, expiry):
                    return
                # The deepest entry serves the next turn only when that turn
                # extends it.  A re-send of the prompt, or a next turn whose
                # template drops the generation prompt from history, resumes
                # at a boundary below it that the tip cannot land on (on a
                # speculative route, never: its draft sidecar is state at the
                # tip's own end).  Prefetch those too, so the resumed request
                # is served from resident state instead of restoring its
                # boundary from disk on its own critical path.
                for (
                    boundary_key,
                    boundary_tokens,
                    boundary,
                ) in self._session_turn_boundaries_locked(tag, key, tokens, entry):
                    try:
                        # An earlier restore may have reclaimed this path
                        # while reserving its own footprint.
                        live = self._trie.get(boundary_key, boundary_tokens)
                    except KeyError:
                        live = None
                    if live is not boundary or self._tip_serves_depth(
                        entry, tokens, boundary, len(boundary_tokens)
                    ):
                        continue
                    expected = getattr(boundary, "_apc_prefetch_expected", set())
                    expected.add(tag)
                    boundary._apc_prefetch_expected = expected
                    if not self._prefetch_entry_locked(
                        tag, boundary_key, boundary_tokens, boundary, expiry
                    ):
                        expected.discard(tag)
        except Exception:
            with self._apc_lock:
                self._disk_stats["prefetch_restores_failed"] += 1
        finally:
            self._prefetch_slot.release()

    def service_pending_prefetch(self) -> bool:
        """Run one queued restore on the model worker's idle boundary."""
        with self._apc_lock:
            pending = self._pending_prefetch
            self._pending_prefetch = None
        if pending is None:
            return False
        self._prefetch_session(*pending)
        return True

    def _cancel_pending_prefetch_locked(self) -> bool:
        pending = self._pending_prefetch
        if pending is None:
            return False
        self._pending_prefetch = None
        tag, _key, _tokens, entry, _expiry = pending
        getattr(entry, "_apc_prefetch_expected", set()).discard(tag)
        self._disk_stats["prefetch_restores_abandoned"] += 1
        self._disk_stats["prefetch_restores_cancelled"] += 1
        self._prefetch_slot.release()
        return True

    def cancel_pending_prefetch(self) -> bool:
        """Cancel one queued restore before it can touch MLX state."""
        with self._apc_lock:
            return self._cancel_pending_prefetch_locked()

    @property
    def has_pending_prefetch(self) -> bool:
        with self._apc_lock:
            return self._pending_prefetch is not None

    def resume_session(
        self, tenant, session_id: str, *, ttl_seconds: Optional[int] = None
    ) -> dict:
        ttl = self._validate_session_ttl(ttl_seconds, optional=True)
        tag = (tenant, session_id)
        with self._apc_lock:
            if self._closed:
                raise APCSessionUnavailable("APCv2 session service is closed")
            records = self._session_entries_locked(tag)
            if not records:
                raise APCSessionNotFound(session_id)
            key, tokens, entry = max(records, key=lambda record: len(record[1]))
            required = int(
                (getattr(entry, "_apc_disk", None) or {}).get(
                    "resident_nbytes", entry.nbytes
                )
            )
            current = self._tenant_pin_bytes_locked(tenant, resident=True)
            already = any(
                pin_tag[0] == tenant
                for pin_tag in getattr(entry, "_apc_resident_pin_expiries", {})
            )
            if not already and current + required > self._pinned_resident_bytes_per_tenant:
                self._disk_stats["pin_cap_rejections"] += 1
                raise APCSessionCapacityError(
                    "per-tenant pinned resident byte cap would be exceeded"
                )
            if not self._prefetch_slot.acquire(blocking=False):
                raise APCSessionCapacityError("APCv2 prefetch queue is full")
            expiry = self._wall_time() + ttl
            expected = getattr(entry, "_apc_prefetch_expected", set())
            expected.add(tag)
            entry._apc_prefetch_expected = expected
            self._disk_stats["resumes"] += 1
            self._pending_prefetch = (tag, key, list(tokens), entry, expiry)
            return self._session_state_locked(tag)

    def delete_session(self, tenant, session_id: str) -> dict:
        tag = (tenant, session_id)
        with self._apc_lock:
            records = self._session_entries_locked(tag)
            if not records:
                raise APCSessionNotFound(session_id)
            removed_entries = removed_files = removed_bytes = shared = deferred = 0
            for key, tokens, entry in tuple(records):
                getattr(entry, "_apc_session_tags", set()).discard(tag)
                getattr(entry, "_apc_disk_pin_expiries", {}).pop(tag, None)
                getattr(entry, "_apc_resident_pin_expiries", {}).pop(tag, None)
                getattr(entry, "_apc_prefetch_expected", set()).discard(tag)
                if getattr(entry, "_apc_session_tags", set()):
                    shared += 1
                    if getattr(entry, "_apc_disk", None):
                        self._write_manifest_locked(key, tokens, entry)
                    continue
                if self._entry_pinned(entry):
                    entry._apc_delete_pending = True
                    deferred += 1
                    continue
                before = self._entry_disk_bytes(entry)
                removed_files += len(self._disk_paths(entry))
                removed_bytes += before + int(entry.nbytes)
                self._drop_entry_locked(key, tokens, entry)
                removed_entries += 1
            self._disk_stats["session_deletes"] += 1
            return {
                "tenant": tenant,
                "session_id": session_id,
                "removed_entries": removed_entries,
                "removed_files": removed_files,
                "removed_bytes": removed_bytes,
                "shared_entries": shared,
                "deferred_leased_entries": deferred,
            }

    def park_all(self, *, time_budget_seconds: float) -> dict:
        deadline = time.monotonic() + max(0.0, float(time_budget_seconds))
        spilled = skipped = 0
        with self._apc_lock:
            for key, tokens, entry in tuple(self._entry_records_locked()):
                if time.monotonic() >= deadline:
                    skipped += 1
                    continue
                if not entry.prompt_cache:
                    continue
                if self._entry_pinned(entry):
                    skipped += 1
                    continue
                if self._spill_entry_locked(key, tokens, entry, reason="shutdown"):
                    spilled += 1
                else:
                    skipped += 1
        return {"spilled": spilled, "skipped": skipped}

    def suspend_resident(self) -> dict:
        """Spill every exact resident entry, dropping only individual failures.

        The serving worker calls this after admission has closed and every
        request lease has drained.  Unlike shutdown parking there is no time
        budget: a suspend is complete only after each resident checkpoint has
        become a disk placeholder or has been dropped.  Session tags, disk and
        resident pin expiries, retention roles, speculative sidecars, and
        interior checkpoints are entry metadata handled by the ordinary spill
        format.  Approximate state never reaches APCv2's exact store path.
        """
        started = time.monotonic()
        with self._apc_lock:
            if self._idle_disk_dir is None:
                raise ValueError("APCv2 suspension requires a configured disk tier")
            self._sweep_retirements_locked()
            records = tuple(self._entry_records_locked())
            active_leases = sum(
                int(getattr(entry.prompt_cache.cow_owner, "pin_count", 0) or 0)
                for _, _, entry in records
                if isinstance(entry.prompt_cache, COWFrozenPromptCache)
            )
            if active_leases:
                raise RuntimeError(
                    f"APCv2 suspension requires zero active leases; found {active_leases}"
                )
            resident = [
                (key, tokens, entry)
                for key, tokens, entry in records
                if entry.prompt_cache
            ]
            resident_bytes_before = sum(int(entry.nbytes) for _, _, entry in resident)
            spilled = failures = suspended_bytes = 0
            for key, tokens, entry in resident:
                entry_bytes = int(entry.nbytes)
                if self._spill_entry_locked(key, tokens, entry, reason="suspend"):
                    spilled += 1
                    suspended_bytes += entry_bytes
                    continue
                # One malformed/unserializable entry must not abort the
                # service-level transition.  No live generation lease should
                # remain here; if an invariant is violated, dropping the cache
                # entry is safer than leaving device state resident while the
                # service reports suspended.
                failures += 1
                self._drop_entry_locked(key, tokens, entry)
            self._enforce_disk_limit_locked()
            resident_entries_after = self._resident_entry_count_locked()
            resident_bytes_after = int(self._n_bytes)
        return {
            "entries": spilled,
            "bytes": suspended_bytes,
            "failures": failures,
            "resident_entries_before": len(resident),
            "resident_bytes_before": resident_bytes_before,
            "resident_entries_after": resident_entries_after,
            "resident_bytes_after": resident_bytes_after,
            "duration_seconds": max(0.0, time.monotonic() - started),
        }

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
                if getattr(entry, "_apc_delete_pending", False) and not self._entry_pinned(entry):
                    self._drop_entry_locked(key, tokens, entry)
                    continue
                last_access = float(getattr(entry, "_apc_last_access_at", now))
                parked = bool(getattr(entry, "_apc_park_pending", False)) and bool(
                    getattr(entry, "_apc_disk_pin_expiries", {})
                )
                if getattr(entry, "_apc_park_pending", False) and not parked:
                    entry._apc_park_pending = False
                if (
                    entry.prompt_cache
                    and not self._entry_resident_pinned_locked(key, tokens, entry)
                    and (parked or now - last_access >= self._idle_disk_seconds)
                    and now >= getattr(entry, "_apc_spill_retry_at", now)
                ):
                    failures = self._disk_stats["spill_failures"]
                    if self._spill_entry_locked(
                        key, tokens, entry, reason="park" if parked else "idle"
                    ):
                        entry._apc_park_pending = False
                        entry._apc_spill_failed_attempts = 0
                        spilled += 1
                    elif self._spill_capacity_violation is not None and getattr(
                        entry, "_apc_disk_pin_expiries", {}
                    ):
                        # This entry was never on disk, so none of its pins was
                        # ever honored, and its real snapshot does not fit
                        # them.  Fail those parks closed instead of rewriting
                        # and discarding the snapshot on every scan.
                        entry._apc_disk_pin_expiries = {}
                        entry._apc_park_pending = False
                        self._disk_stats["pin_cap_rejections"] += 1
                    elif self._disk_stats["spill_failures"] > failures:
                        # A snapshot that failed to serialize almost always
                        # fails the same way on the next scan.  Retry it with
                        # exponential backoff, so each failure is still
                        # counted but one bad entry is not re-serialized every
                        # second forever.
                        attempts = getattr(entry, "_apc_spill_failed_attempts", 0) + 1
                        entry._apc_spill_failed_attempts = attempts
                        entry._apc_spill_retry_at = now + min(
                            scan_interval * 2 ** min(attempts, 16),
                            self._SPILL_RETRY_MAX_SECONDS,
                        )
            if spilled:
                mx.clear_cache()
                self._enforce_disk_limit_locked()
            return spilled

    def lookup(
        self,
        key: Hashable,
        tokens: Iterable[int],
        *,
        allow_disk_restore: bool = True,
        session_tag: Optional[tuple] = None,
    ) -> APCLookup:
        with self._apc_lock:
            self._sweep_retirements_locked()
            return self._lookup_locked(
                key,
                tokens,
                allow_disk_restore=allow_disk_restore,
                session_tag=session_tag,
            )

    def _resolve_prefetch_locked(
        self, session_tag, expected, prefetched_ids, key, path, used_entry
    ) -> None:
        """Settle a resumed session's prefetch on its first tagged lookup.

        It is a hit only when this lookup served an entry the prefetch made
        resident.  Either way the expectation is cleared on every entry that
        carried it, so later lookups are not counted again and the entries
        rejoin ordinary prefix subsumption.
        """
        if session_tag is None or not expected:
            return
        self._disk_stats[
            "prefetch_hits"
            if used_entry is not None and id(used_entry) in prefetched_ids
            else "prefetch_misses"
        ] += 1
        for entry in expected:
            getattr(entry, "_apc_prefetch_expected", set()).discard(session_tag)
        if used_entry is not None:
            getattr(used_entry, "_apc_resident_pin_expiries", {}).pop(
                session_tag, None
            )
            if getattr(used_entry, "_apc_disk", None):
                self._write_manifest_locked(key, path, used_entry)

    def _fetch_view_locked(self, key, tokens, trie_result=None):
        """The trie result ``fetch_nearest_cache`` acts on for ``tokens``.

        A resident exact entry that cannot land inside its own prompt defers
        to the deepest shorter stored prefix.  A disk-only exact entry is
        reported as is, so callers restore or hide it first.
        """
        if trie_result is None:
            trie_result = self._trie.search(key, tokens)
        if trie_result.exact is None or not tokens:
            return trie_result
        try:
            entry = self._trie.get(trie_result.model, trie_result.exact)
        except KeyError:
            return trie_result
        if not entry.prompt_cache or self._exact_entry_serves(entry, tokens):
            return trie_result
        return self._proper_prefix_result(key, tokens)

    def _trie_result_placeholders_locked(self, trie_result):
        """Distinct non-resident entries a trie search result would select."""
        placeholders = []
        seen = set()
        for path in (trie_result.exact, trie_result.longer, trie_result.shorter):
            if path is None or tuple(path) in seen:
                continue
            seen.add(tuple(path))
            try:
                entry = self._trie.get(trie_result.model, path)
            except KeyError:
                continue
            if entry is not None and not entry.prompt_cache:
                placeholders.append((list(path), entry))
        return placeholders

    def _resident_trie_result_locked(self, key, tokens):
        """Select the best resident path while deferred disk entries stay indexed."""
        exact = shorter = longer = None
        common_prefix = 0
        for candidate_key, path, entry in self._entry_records_locked():
            if candidate_key != key or not entry.prompt_cache:
                continue
            shared = 0
            for left, right in zip(tokens, path):
                if left != right:
                    break
                shared += 1
            if shared == len(tokens) == len(path):
                exact = list(path)
                break
            if shared == len(path) < len(tokens):
                if len(path) > len(shorter or ()):
                    shorter = list(path)
                continue
            if shared and shared < len(path) and (
                shared > common_prefix
                or (
                    shared == common_prefix
                    and (longer is None or len(path) < len(longer))
                )
            ):
                longer = list(path)
                common_prefix = shared
        return type(self._trie.search(key, tokens))(
            key,
            exact,
            None if exact is not None else shorter,
            None if exact is not None else longer,
            0 if exact is not None else common_prefix,
        )

    def _lookup_locked(
        self,
        key: Hashable,
        tokens: Iterable[int],
        *,
        allow_disk_restore: bool = True,
        session_tag: Optional[tuple] = None,
    ) -> APCLookup:
        tokens = [int(token) for token in tokens]
        self._apc_stats["queried_tokens"] += len(tokens)
        # Entries a resume of this session asked to prefetch, and those the
        # prefetch actually made resident before this lookup.
        prefetch_expected = []
        prefetched_ids = set()
        if session_tag is not None:
            for _candidate_key, _candidate_tokens, candidate in self._entry_records_locked():
                if session_tag not in getattr(candidate, "_apc_session_tags", set()):
                    continue
                self._expire_entry_pins_locked(
                    _candidate_key, _candidate_tokens, candidate
                )
                if session_tag not in getattr(
                    candidate, "_apc_prefetch_expected", set()
                ):
                    continue
                prefetch_expected.append(candidate)
                if candidate.prompt_cache and session_tag in getattr(
                    candidate, "_apc_resident_pin_expiries", {}
                ):
                    prefetched_ids.add(id(candidate))
        restore_deferred = False
        restore_requires_admission = False
        deferred_entries = []

        def restore_candidates(trie_result):
            yield from (trie_result.exact, trie_result.longer, trie_result.shorter)
            if trie_result.exact is not None:
                # Evaluated after the exact entry was restored: if it cannot
                # serve its own prompt, fetch falls back to the deepest
                # shorter prefix, which must be resident as well.
                fallback = self._fetch_view_locked(key, tokens)
                if fallback.exact is None:
                    yield fallback.shorter

        while True:
            trie_result = self._trie.search(key, tokens)
            retry = False
            seen = set()
            for path in restore_candidates(trie_result):
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
                        # A disk neighbor must not hide an already-resident
                        # prefix, so defer only that entry during selection.
                        restore_requires_admission = True
                        deferred_entries.append(
                            (trie_result.model, list(path), entry)
                        )
                        continue
                    restored = self._restore_entry_locked(trie_result.model, path, entry)
                    if restored is None:
                        # One healthy disk neighbor being temporarily too large
                        # must not hide a resident prefix that another lane is
                        # already leasing.  Keep the snapshot indexed and let
                        # selection below consider only resident candidates.
                        restore_deferred = True
                        deferred_entries.append(
                            (trie_result.model, list(path), entry)
                        )
                        continue
                    if not restored:
                        self._drop_entry_locked(trie_result.model, path, entry)
                        retry = True
                        break
            if not retry:
                # Reservation may have removed a neighboring candidate.  Make
                # the selection phase consume a fresh, internally consistent
                # trie result without repeatedly restoring mutually exclusive
                # neighbors under a tight resident budget.
                trie_result = (
                    self._resident_trie_result_locked(key, tokens)
                    if restore_deferred or restore_requires_admission
                    else self._trie.search(key, tokens)
                )
                break
        # A longer stored path shares ``common_prefix`` tokens with this
        # prompt.  (An exact path reports 0: its only junction would be the
        # whole prompt, which the committed prompt boundary already covers.)
        shared_tokens = (
            int(trie_result.common_prefix) if trie_result.longer is not None else 0
        )

        def branch_beyond(cached):
            return shared_tokens if shared_tokens > int(cached) else 0

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
                    sidecar_candidates.append((covered, path, entry, sidecar))
        if sidecar_candidates:
            (covered, selected_tokens, entry, sidecar) = max(
                sidecar_candidates, key=lambda item: item[0]
            )
            try:
                restored_cache = _copy_prompt_cache_for_restore(entry.prompt_cache)
            except COWCacheStale:
                self._apc_stats["lookups"] += 1
                self._apc_stats["misses"] += 1
                self._resolve_prefetch_locked(
                    session_tag, prefetch_expected, prefetched_ids, None, None, None
                )
                return APCLookup(
                    None,
                    tokens,
                    0,
                    False,
                    None,
                    "stale_cow_generation",
                    capsule_generation=self._capsule_generation.current,
                    branch_tokens=branch_beyond(0),
                )
            self._apc_stats["lookups"] += 1
            self._apc_stats["hits"] += 1
            self._apc_stats["cached_tokens"] += covered
            self._record_entry_hit_locked(entry)
            self._resolve_prefetch_locked(
                session_tag, prefetch_expected, prefetched_ids,
                key, selected_tokens, entry,
            )
            prefetch_expected = []
            # Sidecar hits bypass PrefixIndex.fetch_nearest_cache; refresh the
            # selected checkpoint's recency so repeated reuse is not FIFO.
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
                    retention_role=getattr(
                        entry, "_apc_retention_role", self._RETENTION_DEFAULT
                    ),
                    branch_tokens=branch_beyond(covered),
                )
        hidden = []
        # Hit accounting must credit the entry fetch actually served.
        fetch_view = self._fetch_view_locked(key, tokens, trie_result)
        if restore_deferred or restore_requires_admission:
            for deferred_key, path, entry in deferred_entries:
                try:
                    if self._trie.get(deferred_key, path) is entry:
                        hidden.append(
                            (
                                deferred_key,
                                path,
                                self._trie.pop(deferred_key, path),
                            )
                        )
                except KeyError:
                    pass
        try:
            while restore_deferred or restore_requires_admission:
                # Selection above saw resident entries only, so the fetch must
                # too.  Hiding the deferred candidates is not enough: a
                # shallower disk-only snapshot that was never a candidate would
                # then match as the "shorter" prefix, hand back its empty cache
                # and turn the lookup into a malformed-topology miss that hides
                # both the resident prefix and the admission/budget reason.
                fetch_view = self._fetch_view_locked(key, tokens)
                placeholders = self._trie_result_placeholders_locked(fetch_view)
                if not placeholders:
                    break
                for path, _entry in placeholders:
                    hidden.append((key, path, self._trie.pop(key, path)))
            try:
                (cache, remaining) = super().fetch_nearest_cache(key, tokens)
            except COWCacheStale:
                (cache, remaining) = (None, tokens)
                stale_generation = True
            else:
                stale_generation = False
        finally:
            for deferred_key, path, entry in hidden:
                self._trie.add(deferred_key, path, entry)
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
        selected_entry = None
        if hit:
            short_length = (
                len(fetch_view.shorter) if fetch_view.shorter is not None else 0
            )
            selected_path = None
            if fetch_view.exact is not None:
                selected_path = fetch_view.exact
            elif (
                fetch_view.longer is not None
                and cached_tokens > short_length
            ):
                selected_path = fetch_view.longer
            elif fetch_view.shorter is not None:
                selected_path = fetch_view.shorter
            try:
                if selected_path is not None:
                    selected_entry = self._trie.get(key, selected_path)
            except KeyError:
                selected_entry = None
            if selected_entry is not None:
                self._record_entry_hit_locked(selected_entry)
        self._resolve_prefetch_locked(
            session_tag, prefetch_expected, prefetched_ids,
            key, selected_path if selected_entry is not None else None,
            selected_entry,
        )
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
            elif restore_requires_admission:
                reason = "disk_restore_requires_admission"
            elif restore_deferred:
                reason = "disk_restore_budget_unavailable"
            else:
                reason = (
                    "untrimmable_branch"
                    if has_unusable_branch
                    else "no_compatible_prefix"
                )
        elif fetch_view.exact is not None:
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
            retention_role=(
                getattr(
                    selected_entry,
                    "_apc_retention_role",
                    self._RETENTION_DEFAULT,
                )
                if selected_entry is not None
                else None
            ),
            branch_tokens=branch_beyond(cached_tokens),
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
        session_tag: Optional[tuple] = None,
    ) -> APCCapabilities:
        if retention_role not in self._RETENTION_ROLES:
            raise ValueError(f"unknown APCv2 retention role: {retention_role!r}")
        with self._apc_lock:
            self._sweep_retirements_locked()
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
                session_tag=session_tag,
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
        session_tag: Optional[tuple] = None,
    ) -> APCCapabilities:
        tokens = [int(token) for token in tokens]
        capabilities = inspect_apc_capabilities(prompt_cache)
        if not capabilities.exact_prefix:
            return replace(capabilities, stored=False)
        if self.max_tokens is not None and len(tokens) > self.max_tokens:
            self.overlength_rejections += 1
            return replace(capabilities, stored=False)
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
        try:
            replaced_entry = self._trie.get(key, tokens)
        except (KeyError, TypeError):
            replaced_entry = None
        existing_tags = set()
        existing_disk_pins = {}
        existing_resident_pins = {}
        existing_created_wall = self._wall_time()
        existing_hit_count = 0
        existing = self._trie.search(key, tokens)
        if existing.exact is not None:
            try:
                prior_entry = self._trie.get(key, existing.exact)
                existing_tags = set(
                    getattr(prior_entry, "_apc_session_tags", set())
                )
                existing_disk_pins = dict(
                    getattr(prior_entry, "_apc_disk_pin_expiries", {})
                )
                existing_resident_pins = dict(
                    getattr(prior_entry, "_apc_resident_pin_expiries", {})
                )
                existing_created_wall = float(
                    getattr(prior_entry, "_apc_created_wall", existing_created_wall)
                )
                existing_hit_count = int(
                    getattr(prior_entry, "_apc_hit_count", 0)
                )
            except KeyError:
                pass
        if session_tag is not None:
            if (
                not isinstance(session_tag, tuple)
                or len(session_tag) != 2
                or not isinstance(session_tag[1], str)
            ):
                raise ValueError("invalid APCv2 session tag")
            if len(existing_tags) < self._SESSION_TAG_LIMIT or session_tag in existing_tags:
                existing_tags.add(session_tag)
        removed_entries = []
        resident_limit = self.max_bytes
        sequence_limit = self.max_size
        # PrefixIndex cannot see APC retention roles, session pins, or live
        # leases.  Subsumption is safe only for disposable ordinary prefixes;
        # APC still owns size/byte eviction after publication.
        def can_prune_prefix(_length, entry):
            return retention_role in (
                self._RETENTION_DEFAULT,
                self._RETENTION_JUNCTION,
                self._RETENTION_PROMPT_BOUNDARY,
            ) and not (
                self._entry_pinned(entry)
                or getattr(entry, "_apc_session_tags", None)
                or getattr(entry, "_apc_disk_pin_expiries", None)
                or getattr(entry, "_apc_resident_pin_expiries", None)
                or getattr(entry, "_apc_prefetch_expected", None)
                or getattr(entry, "_apc_retention_role", self._RETENTION_DEFAULT)
                != self._RETENTION_DEFAULT
            )

        self.max_bytes = 1 << 63
        self.max_size = 1 << 63
        try:
            inserted = super().insert_cache(
                key, tokens, prompt_cache, cache_type=cache_type, sidecar=sidecar,
                prune_prefixes=can_prune_prefix,
                removed_entries=removed_entries,
            )
        finally:
            self.max_bytes = resident_limit
            self.max_size = sequence_limit
        if not inserted:
            if cow_source is not None:
                cow_source.close()
            return replace(capabilities, stored=False)
        self._capsule_generation.advance()
        superseded_parks = []
        for reason, _removed_key, _removed_tokens, entry in removed_entries:
            if reason != "replaced":
                self._record_entry_eviction_locked(entry)
            if isinstance(entry.prompt_cache, COWFrozenPromptCache):
                entry.prompt_cache.close()
            if (
                reason == "replaced"
                and existing_disk_pins
                and self._persist_dir is not None
                and getattr(entry, "_apc_disk", None)
            ):
                # A parked session's persisted snapshot is its only
                # crash-safe copy.  The replacement inherits the disk pin, so
                # unlink the old snapshot only once the replacement's own
                # content has been persisted in its place.
                superseded_parks.append(entry)
                continue
            self._remove_disk_files_locked(entry)
        survivor = self._trie.search(key, tokens)
        if survivor.exact is not None:
            stored_entry = self._trie.get(key, survivor.exact)
            now = self._now()
            if retention_role == self._RETENTION_ROLLING and replaced_entry is not None:
                # A disposable progress point never downgrades a published
                # state at the same tokens; a stronger role upgrades it.
                prior_role = getattr(
                    replaced_entry, "_apc_retention_role", self._RETENTION_DEFAULT
                )
                if prior_role != self._RETENTION_ROLLING:
                    retention_role = prior_role
            # This publication supersedes any deferred retirement of the path.
            self._pending_retirements.pop((key, tuple(survivor.exact)), None)
            stored_entry._apc_retention_role = retention_role
            stored_entry._apc_inserted_at = getattr(
                replaced_entry, "_apc_inserted_at", now
            )
            # A republish is a use: inheriting a spilled copy's old access
            # time made the fresh resident copy the LRU victim of this very
            # store, so it was spilled again before any reader could lease it.
            stored_entry._apc_last_access_at = now
            stored_entry._apc_hit_count = getattr(
                replaced_entry, "_apc_hit_count", existing_hit_count
            )
            stored_entry._apc_created_wall = existing_created_wall
            stored_entry._apc_last_access_wall = self._wall_time()
            stored_entry._apc_session_tags = set(
                list(existing_tags)[: self._SESSION_TAG_LIMIT]
            )
            stored_entry._apc_disk_pin_expiries = existing_disk_pins
            stored_entry._apc_resident_pin_expiries = existing_resident_pins
            if superseded_parks:
                self._spill_entry_locked(
                    key, list(survivor.exact), stored_entry,
                    reason="republish", keep_resident=True,
                )
        for entry in superseded_parks:
            self._remove_disk_files_locked(entry)
        if survivor.exact is not None:
            self._enforce_entry_limits_locked(
                publication=(key, list(survivor.exact), stored_entry)
            )
        self._apc_stats["stores"] += 1
        survivor = self._trie.search(key, tokens)
        return replace(capabilities, stored=survivor.exact is not None)

    def retire(self, key: Hashable, tokens: Iterable[int], *, role: str) -> bool:
        """Drop a disposable checkpoint its publisher no longer needs.

        Returns True when the postcondition already holds: the entry was
        dropped, or is absent, replaced, or upgraded to another role.  Returns
        False while a live lease pins it; the drop then happens at the first
        APCv2 operation after the lease is released.  Design reference:
        Splash ``StateCache::retireCheckpoint`` (rev f58d36dd).
        """
        if role not in self._RETENTION_ROLES:
            raise ValueError(f"unknown APCv2 retention role: {role!r}")
        tokens = tuple(int(token) for token in tokens)
        with self._apc_lock:
            self._sweep_retirements_locked()
            return self._retire_locked(key, tokens, role)

    def _retire_locked(self, key, tokens: tuple, role: str) -> bool:
        try:
            entry = self._trie.get(key, list(tokens))
        except (KeyError, TypeError):
            entry = None
        if (
            entry is None
            or getattr(entry, "_apc_retention_role", self._RETENTION_DEFAULT) != role
        ):
            self._pending_retirements.pop((key, tokens), None)
            return True
        if self._entry_pinned(entry):
            self._pending_retirements[(key, tokens)] = role
            return False
        self._pending_retirements.pop((key, tokens), None)
        self._drop_entry_locked(key, list(tokens), entry)
        return True

    def sweep_retirements(self) -> None:
        """Complete retirements whose deferring leases have been released."""
        with self._apc_lock:
            self._sweep_retirements_locked()

    def _sweep_retirements_locked(self) -> None:
        for (key, tokens), role in tuple(self._pending_retirements.items()):
            self._retire_locked(key, tokens, role)

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
        self._pending_retirements.clear()
        fresh_trie = PromptTrie()
        fresh_lru = PrefixIndex.CacheOrder(list(self._lru._ordering))
        self._trie = fresh_trie
        self._lru = fresh_lru
        self._n_bytes = 0
        self._n_bytes_by_type = {key: 0 for key in fresh_lru._ordering}
        for entry in entries:
            self._record_entry_eviction_locked(entry)
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

    def lifetime_stats(self) -> dict:
        """Live lifetime counters, without ``apc_stats``' whole-trie walk.

        ``apc_stats`` aggregates per-entry segment statistics, so serving
        snapshots it at most once a second.  The hit/lookup counters are plain
        integers and need no such cadence: exporting them from the snapshot
        made every ``mlx2_prefix_cache_*_hits_total`` series read 0 until the
        first refresh, and ``junction_hits`` reads 0 for a whole short run.
        """
        with self._apc_lock:
            lifetime = dict(self._apc_lifetime)
            for key in self._STAT_KEYS:
                lifetime[key] += self._apc_stats[key]
            return lifetime

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
            stats["resident_max_bytes"] = int(self.max_bytes)
            stats["max_entry_tokens"] = self.max_entry_tokens
            stats["overlength_rejections"] = self.overlength_rejections
            stats["idle_disk"] = {
                "enabled": self._idle_disk_dir is not None,
                "persistent": self._persist_dir is not None,
                "block_bytes": self._persistent_block_bytes,
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
            stats["persistence"] = {
                "enabled": self._persist_dir is not None,
                "rescan": {
                    **self._rescan,
                    "discarded": dict(self._rescan.get("discarded", {})),
                },
                "schema": self._PERSIST_SCHEMA if self._persist_dir else None,
                "quarantine": {
                    **self._quarantine,
                    "max_entries": self._quarantine_max_entries,
                    "max_bytes": self._quarantine_max_bytes,
                },
            }
            stats["sessions"] = {
                "enabled": self._idle_disk_dir is not None,
                "max_ttl_seconds": self._session_max_ttl_seconds,
                "pinned_disk_bytes_per_tenant": self._pinned_disk_bytes_per_tenant,
                "pinned_disk_bytes_global": self._pinned_disk_bytes_global,
                "pinned_disk_bytes_current": self._global_pin_bytes_locked(),
                "pinned_resident_bytes_per_tenant": self._pinned_resident_bytes_per_tenant,
                "prefetch_ttl_seconds": self._prefetch_ttl_seconds,
            }
            stats["cache_capsules"] = {
                **self._capsule_capacity,
                "reserved_bytes": self._capsule_reserved_bytes,
            }
            interior_entries = interior_bytes = interior_reused = 0
            for entry in _iter_trie_entries(self._trie):
                if (
                    getattr(entry, "_apc_retention_role", None)
                    != self._RETENTION_INTERIOR
                ):
                    continue
                interior_entries += 1
                if int(getattr(entry, "_apc_hit_count", 0)) > 0:
                    interior_reused += 1
                if not getattr(entry, "_apc_disk", None):
                    interior_bytes += int(getattr(entry, "nbytes", 0) or 0)
            stats["interior"] = {
                "entries": interior_entries,
                "max_entries": self.max_interior_entries,
                "resident_bytes": interior_bytes,
                "reused_entries": interior_reused,
                "lifetime_reused_entries": getattr(
                    self, "_interior_reused_entries", 0
                ),
            }
            stats["reuse_telemetry"] = {
                name: histogram.snapshot()
                for name, histogram in self._reuse_histograms.items()
            }
            return stats

    def close(
        self,
        *,
        persist_resident: bool = False,
        time_budget_seconds: float = 0.0,
        release_memory: bool = True,
    ) -> None:
        """Close APC ownership and release the persistent directory lock."""
        with self._apc_lock:
            if self._closed:
                return
            self._closed = True
            self._cancel_pending_prefetch_locked()
        if self._persist_dir is not None and persist_resident:
            self.park_all(time_budget_seconds=time_budget_seconds)
        with self._apc_lock:
            if self._persist_dir is None:
                self._clear_locked(release_memory=release_memory)
            else:
                entries = list(self._entry_records_locked())
                for key, tokens, entry in entries:
                    if getattr(entry, "_apc_disk", None):
                        self._write_manifest_locked(key, tokens, entry)
                    if isinstance(entry.prompt_cache, COWFrozenPromptCache):
                        entry.prompt_cache.close()
                    entry.prompt_cache = []
                    entry.sidecar = None
                    entry.nbytes = 0
                self._trie = PromptTrie()
                self._lru = PrefixIndex.CacheOrder()
                self._n_bytes = 0
                self._n_bytes_by_type = {
                    key: 0 for key in self._lru._ordering
                }
                if release_memory:
                    mx.clear_cache()
            self._release_persist_lock()

    def resident_nbytes(self) -> int:
        """Bytes of every resident checkpoint, leased or not."""
        with self._apc_lock:
            return int(self._n_bytes)

    def unleased_resident_nbytes(self) -> int:
        """Resident bytes that pressure eviction could actually reclaim."""
        with self._apc_lock:
            return sum(
                int(entry.nbytes)
                for (_rank, _last, _key, _tokens, entry) in self._pressure_candidates_locked()
            )

    def evict_oldest_unleased(self) -> bool:
        """Reclaim one resident checkpoint without invalidating active branches.

        A configured disk tier keeps the checkpoint as a disk-only placeholder
        before releasing its resident arrays. Disk pins protect the entry and
        its files through expiry; an unpinned spill failure may still fall back
        to destructive resident eviction so memory pressure can make progress.
        """
        with self._apc_lock:
            records = self._pressure_candidates_locked()
            for _rank, _last_access, key, tokens, entry in records:
                if self._idle_disk_dir is not None and self._spill_entry_locked(
                    key, tokens, entry, reason="pressure"
                ):
                    # A zero/undersized disk budget may retire the new
                    # placeholder here.  Either outcome has released the
                    # resident arrays and therefore counts as progress.
                    self._enforce_disk_limit_locked()
                    return True
                if self._entry_disk_pinned_locked(key, tokens, entry):
                    continue
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
        sequence_limit = max(0, n_sequences) if n_sequences is not None else 1 << 63
        byte_limit = max(0, n_bytes) if n_bytes is not None else 1 << 63
        while (
            self._resident_entry_count_locked() > sequence_limit
            or self._n_bytes > byte_limit
        ):
            progressed = False
            for _rank, _last_access, key, tokens, entry in self._pressure_candidates_locked():
                if self._idle_disk_dir is not None and self._spill_entry_locked(
                    key, tokens, entry, reason="pressure"
                ):
                    progressed = True
                    break
                if self._entry_disk_pinned_locked(key, tokens, entry):
                    continue
                self._drop_entry_locked(key, tokens, entry)
                progressed = True
                break
            if not progressed:
                break
        self._enforce_disk_limit_locked()

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
