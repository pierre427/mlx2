"""Bounded request ownership around the modern batch execution core."""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
from itertools import islice
import json
import math
import platform
import logging
from pathlib import Path
import queue
import secrets
import threading
import time
import uuid

from .logprobs import token_logprob, wants_logprobs
from .request_limits import (
    DEFAULT_OUTPUT_TOKENS,
    resolve_output_limit,
    validate_default_max_tokens,
)
from .sampling_defaults import (
    generation_config_drift,
    resolve_sampling,
    vendor_sampling,
)
from .batch_metrics import BatchFaultSpec, BatchRuntimeMetrics, HttpRuntimeMetrics

log = logging.getLogger(__name__)


def generation_stop_token_ids(adapter) -> tuple[int, ...]:
    """Return the adapter's canonical token-level generation terminators.

    Batch stop matching, minimum-token masking, and structured constraints must
    consume this same frozen value.  Reading the tokenizer independently at
    each site lets mutable wrappers or adapter-specific multi-EOS metadata
    drift between the generation loop and a request processor.
    """
    raw = getattr(getattr(adapter, "tokenizer", None), "eos_token_ids", ())
    values = () if raw is None else (raw,) if isinstance(raw, int) else raw
    try:
        token_ids = tuple(sorted({int(token) for token in values}))
    except (TypeError, ValueError) as error:
        raise ValueError("adapter generation stop token ids must be integers") from error
    if any(token < 0 for token in token_ids):
        raise ValueError("adapter generation stop token ids must be nonnegative")
    return token_ids


@dataclass
class IdleAdmissionCoalescer:
    """Bound the idle-to-active admission wait for ordinary submissions."""

    initial_seconds: float
    deadline: float | None = None

    def note_attachment(self, *, now: float) -> None:
        if self.deadline is None:
            self.deadline = now + self.initial_seconds

    def timeout(self, *, now: float, idle: bool, deferred: bool) -> float:
        if self.deadline is not None:
            return max(0.0, self.deadline - now)
        if idle:
            return 0.01 if deferred else 0.05
        return 0.0


def coalescing_window(initial_ms=5.0):
    """Return the bounded ordinary idle-admission window in seconds."""
    if isinstance(initial_ms, bool):
        raise ValueError("coalescing window must be numeric milliseconds")
    try:
        initial_ms = float(initial_ms)
    except (TypeError, ValueError) as error:
        raise ValueError("coalescing window must be numeric milliseconds") from error
    if not math.isfinite(initial_ms) or initial_ms < 0:
        raise ValueError("coalescing window must be finite and nonnegative")
    return initial_ms / 1000.0


def cache_capsule_policy(value) -> dict:
    """Validate the qualification-only warm-fanout policy."""
    defaults = {
        "enabled": False,
        "backend": "gpu",
        "fallback": "gpu",
        "deadline_ms": 50.0,
        "verify_raw_bits": False,
    }
    if value is None or value is False:
        return defaults
    if value is True:
        return {**defaults, "enabled": True}
    if not isinstance(value, dict):
        raise ValueError("cache_capsules must be a boolean or object")
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(f"unknown cache capsule settings: {sorted(unknown)}")
    policy = {**defaults, **value}
    if not isinstance(policy["enabled"], bool) or not isinstance(
        policy["verify_raw_bits"], bool
    ):
        raise ValueError("cache capsule enabled/verify_raw_bits must be booleans")
    if policy["backend"] not in {"cpu", "gpu"}:
        raise ValueError("cache capsule backend must be cpu or gpu")
    if policy["fallback"] not in {None, "cpu", "gpu"}:
        raise ValueError("cache capsule fallback must be cpu, gpu, or null")
    if isinstance(policy["deadline_ms"], bool):
        raise ValueError("cache capsule deadline must be numeric")
    policy["deadline_ms"] = float(policy["deadline_ms"])
    if not math.isfinite(policy["deadline_ms"]) or policy["deadline_ms"] <= 0:
        raise ValueError("cache capsule deadline must be finite and positive")
    return policy


def apc_interior_checkpoint_policy(value) -> dict:
    """Validate the default-off APCv2 hybrid checkpoint budget.

    Design reference: omlx#3456.  A small hard count bound keeps request-owned
    descriptor snapshots and metric cardinality bounded independently of prompt
    length.
    """
    defaults = {"count": 0, "min_stride": 1}
    if value is None:
        return defaults
    if not isinstance(value, dict):
        raise ValueError("apc_interior_checkpoints must be an object")
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(
            f"unknown APC interior checkpoint settings: {sorted(unknown)}"
        )
    policy = {**defaults, **value}
    count = policy["count"]
    stride = policy["min_stride"]
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 32:
        raise ValueError("APC interior checkpoint count must be an integer from 0 to 32")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("APC interior checkpoint min_stride must be a positive integer")
    return {"count": count, "min_stride": stride}


def budget_interior_checkpoint_positions(
    candidates, *, available_bytes, cache_projection
):
    """Keep the deepest optional checkpoints that fit measured headroom."""
    candidates = tuple(candidates)
    if (
        isinstance(available_bytes, bool)
        or not isinstance(available_bytes, (int, float))
        or not math.isfinite(available_bytes)
        or available_bytes < 0
    ):
        raise ValueError("available checkpoint bytes must be finite and nonnegative")
    if not candidates or not callable(cache_projection):
        return (), 0
    selected = []
    charged = 0
    for position in reversed(candidates):
        projected = cache_projection(position)
        if (
            isinstance(projected, bool)
            or not isinstance(projected, (int, float))
            or not math.isfinite(projected)
            or projected < 0
        ):
            raise ValueError("checkpoint cache projection must be finite and nonnegative")
        projected = int(projected)
        if charged + projected > available_bytes:
            break
        selected.append(position)
        charged += projected
    return tuple(reversed(selected)), charged


def decode_time_fairness_policy(*, external_draft: bool, prompt_lookup: bool) -> dict:
    """Return only the policy actually constructed by the selected route."""
    return {
        "enabled": not (external_draft or prompt_lookup),
        "fair_share": 0.5,
        "stall_target_ms": 500.0,
    }


class HostPromptCache:
    """Bounded, process-local cache for deterministic prompt tokenization.

    Keys are hashes of prompt-affecting request fields, so neither prompt text
    nor tool schemas appear in status output. Values are copied at both the
    insertion and lookup boundaries because admission mutates its own lists.
    """

    _NON_PROMPT_FIELDS = frozenset(
        {
            "model",
            "stream",
            "n",
            "max_tokens",
            "max_completion_tokens",
            "min_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "seed",
            "stop",
            "stream_options",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "sampling_profile",
            "context_limit",
            "response_format",
            "grammar",
            "thinking_budget",
            "thinking_budget_mode",
            "thinking_steer_alpha",
            "parallel_tool_calls",
            "batch_cohort",
            "mlx_fault",
            "skip_writing_prefix_cache",
            "session_id",
            "_mlx2_prefill_inputs",
            "_mlx2_multimodal_stats",
            "_mlx2_media_token_end",
        }
    )

    def __init__(self, *, max_entries: int = 128, max_tokens: int = 1 << 20):
        if type(max_entries) is not int or type(max_tokens) is not int:
            raise ValueError("host prompt cache bounds must be integers")
        if max_entries < 0 or max_tokens < 0:
            raise ValueError("host prompt cache bounds must be nonnegative")
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self._entries = OrderedDict()
        self._tokens = 0
        self._stats = Counter()
        self._lock = threading.Lock()

    @classmethod
    def key(cls, request: dict) -> str:
        # Default unknown fields to prompt-affecting. This makes adapter
        # extensions safe by construction; only the explicit generation-only
        # controls above are ignored for cache reuse.
        payload = {
            field: value
            for field, value in request.items()
            if field not in cls._NON_PROMPT_FIELDS
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def get(self, request: dict) -> list[int] | None:
        if not self.max_entries or not self.max_tokens:
            with self._lock:
                self._stats["disabled_misses"] += 1
            return None
        key = self.key(request)
        with self._lock:
            tokens = self._entries.pop(key, None)
            if tokens is None:
                self._stats["misses"] += 1
                return None
            self._entries[key] = tokens
            self._stats["hits"] += 1
            return list(tokens)

    def put(self, request: dict, tokens) -> None:
        if not self.max_entries or not self.max_tokens:
            with self._lock:
                self._stats["disabled_skips"] += 1
            return
        # Bound insertion work as well as retained storage.  ``tokens`` may be
        # a generator, so materialising it before checking its size can turn a
        # bounded cache into an unbounded allocation (or never return at all).
        tokens = tuple(
            int(token) for token in islice(tokens, self.max_tokens + 1)
        )
        if len(tokens) > self.max_tokens:
            with self._lock:
                self._stats["oversize_skips"] += 1
            return
        key = self.key(request)
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._tokens -= len(old)
            self._entries[key] = tokens
            self._tokens += len(tokens)
            while (
                len(self._entries) > self.max_entries
                or self._tokens > self.max_tokens
            ):
                _, evicted = self._entries.popitem(last=False)
                self._tokens -= len(evicted)
                self._stats["evictions"] += 1
            self._stats["stores"] += 1

    def status(self) -> dict:
        with self._lock:
            return {
                "schema": "mlx2.host-prompt-cache.v1",
                "entries": len(self._entries),
                "tokens": self._tokens,
                "max_entries": self.max_entries,
                "max_tokens": self.max_tokens,
                **dict(self._stats),
            }

    def clear(self) -> None:
        with self._lock:
            removed = len(self._entries)
            self._entries.clear()
            self._tokens = 0
            self._stats["clears"] += 1
            self._stats["cleared_entries"] += removed


def ordinary_compute_width(
    response, observed_width, *, mtp, external_draft, prompt_lookup=False
):
    if external_draft or prompt_lookup:
        receipt = getattr(response, "speculative_receipt", None) or {}
        return observed_width if receipt.get("execution") == "ordinary_target" else None
    return observed_width if not mtp or response.mtp_receipt is None else None


def minimum_tokens_processor(array_module, eos_token_ids, prompt_tokens, minimum):
    """Suppress model EOS tokens until the requested completion length.

    The processor is applied to ordinary, self-MTP, and external speculative
    lanes through their shared per-lane logits-processor contract. It leaves
    user stop strings active; callers that need an exact cap must omit them.
    """
    eos_token_ids = tuple(sorted(int(token) for token in eos_token_ids))
    prompt_tokens = int(prompt_tokens)
    minimum = int(minimum)
    if minimum <= 0 or not eos_token_ids:
        return None

    def processor(tokens, logits):
        generated = int(tokens.shape[-1]) - prompt_tokens
        if generated >= minimum:
            return logits
        vocabulary = array_module.arange(logits.shape[-1])
        eos_mask = vocabulary == eos_token_ids[0]
        for token in eos_token_ids[1:]:
            eos_mask = eos_mask | (vocabulary == token)
        return array_module.where(eos_mask, -float("inf"), logits)

    # Stateless: speculative draft probes can safely call the same function.
    processor.probe = processor
    return processor


def runtime_identity() -> dict:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    native = hashlib.sha256()
    dist = importlib.metadata.distribution("mlx")
    for file in sorted(dist.files or [], key=str):
        if str(file).endswith((".so", ".dylib", ".metallib")):
            path = Path(dist.locate_file(file))
            native.update(str(file).encode())
            native.update(path.read_bytes())
    return {
        "source_sha256": digest.hexdigest(),
        "mlx_native_sha256": native.hexdigest(),
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "mlx": importlib.metadata.version("mlx"),
        "transformers": importlib.metadata.version("transformers"),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "tokenizers",
                "jinja2",
                "psutil",
                "regex",
                "safetensors",
            )
        },
    }


def persistent_runtime_revision(identity: dict) -> tuple:
    """Bind persistent APC state to the complete qualification runtime identity."""
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return (
        "mlx2-qualified-runtime-v1",
        identity["source_sha256"],
        identity.get("mlx_native_sha256", "unavailable"),
        identity.get("mlx", "unavailable"),
        hashlib.sha256(payload).hexdigest(),
    )


def cache_semantic_fingerprint(tenant_scope):
    """APCv2 semantic namespace: shared, or bound to one tenant id."""
    if tenant_scope is None:
        return "text-token-v1"
    return ("text-token-v1", "tenant", str(tenant_scope))


def int8_prefill_status(engine):
    """Host-only int8 prefill status (policy, bound modules, live counters)."""
    from .runtime.int8_prefill import engine_status

    return engine_status(engine)


def thinking_enabled(adapter, request) -> bool:
    """Resolve the adapter's view of whether this request opens a think channel."""
    if hasattr(adapter, "thinking_enabled"):
        return bool(adapter.thinking_enabled(request))
    return bool(request.get("enable_thinking", False))


def thinking_close_token_ids(adapter):
    """The adapter's declared thinking-close marker ids, or None (fail closed)."""
    accessor = getattr(adapter, "thinking_close_token_ids", None)
    ids = accessor() if callable(accessor) else None
    return tuple(int(token) for token in ids) if ids else None


def structured_envelope_token_ids(adapter):
    """The adapter's answer-channel (open ids, close ids), or None."""
    accessor = getattr(adapter, "structured_envelope_token_ids", None)
    envelope = accessor() if callable(accessor) else None
    if not envelope:
        return None
    opens, closes = envelope
    return tuple(int(t) for t in opens), tuple(int(t) for t in closes)


def shared_prefix_attestation(hit):
    """Attest only sibling branches of one immutable APC generation plus one seed.

    Equal token text alone cannot attest tensor equality after different prefill
    chunking. The COW lineage and generation provide that stronger boundary.
    """
    branch = hit.cache
    lineage = getattr(branch, "cow_lineage_id", None)
    generation = getattr(branch, "cow_generation", None)
    if not lineage or generation is None or len(hit.remaining_tokens) != 1 or hit.sidecar is None:
        return None
    payload = (lineage, generation, hit.cached_tokens, int(hit.remaining_tokens[0]))
    return hashlib.sha256(repr(payload).encode()).hexdigest()


def warm_cache_copy_gib(hit, *, context_tokens, prefill_step, mtp):
    """Full-copy bound from complete resident state; never credit COW as free.

    A long uncached tail or absent draft state uses the cold model bound. Near
    a complete checkpoint, measured bytes encode the real dtype/capacity; scale
    them to the requested extent and let the controller add fp32 growth slack.
    """
    if not hit.cache or not hit.cached_tokens or len(hit.remaining_tokens) > prefill_step:
        return 0.0
    if mtp and hit.sidecar is None:
        return 0.0
    from .runtime.apc_v2 import _walk_cache_entries

    rows = list(_walk_cache_entries(hit.cache))
    size = sum(int(getattr(row, "nbytes", 0)) for row in rows)
    if mtp:
        size += int(hit.sidecar.nbytes)
    # Scale per *allocated* token.  A hit can be a longer checkpoint trimmed
    # to a short shared prefix (every chat prompt shares its template head):
    # its buffers still hold the whole entry, so dividing by the few cached
    # tokens projected hundreds of GiB and deferred the request until its
    # admission deadline (HTTP 429 on an idle server).
    capacity = hit.cached_tokens
    for row in rows:
        keys = getattr(row, "keys", None)
        if isinstance(keys, (tuple, list)) and keys:
            keys = keys[0]
        shape = getattr(keys, "shape", None)
        if shape is not None and len(shape) >= 2:
            capacity = max(capacity, int(shape[-2]))
    return size * max(1.0, context_tokens / capacity) / (1 << 30)


EVICTION_ESTIMATE_SLACK = 1.25


def prompt_lookup_verification_gib(controller, num_draft):
    """Incremental PLD target-forward transient beyond ordinary width one."""
    if isinstance(num_draft, bool) or not isinstance(num_draft, int) or num_draft < 1:
        raise ValueError("prompt-lookup num_draft must be a positive integer")
    return controller.transient_gib_per_lane * num_draft / 3.0


def ensure_admission_headroom(
    required_bytes, *, headroom, reclaim, evict, evictable=None
):
    """Observe completed reclamation before evicting or rejecting a request.

    ``evictable`` reports the logical bytes of unleased resident checkpoints.
    A request that eviction cannot plausibly satisfy (it must wait for active
    lanes to finish) is deferred without touching the warm checkpoints; each
    of its 250 ms retries would otherwise purge every unleased entry and
    force an allocator synchronization per eviction.  Logical bytes are not
    exactly the measured headroom eviction returns, so the test carries slack
    and only rules out eviction when it would fall clearly short.
    """
    if headroom() >= required_bytes:
        return True
    reclaim()
    free = headroom()
    if free >= required_bytes:
        return True
    if (
        evictable is not None
        and free + evictable() * EVICTION_ESTIMATE_SLACK < required_bytes
    ):
        return False
    while evict():
        reclaim()
        if headroom() >= required_bytes:
            return True
    return False


def reclaim_deferred_cache(apc, admission, clear_scratch):
    """An empty scheduler poll is not evidence of memory pressure.

    Only a fully memory-deferred decision permits eviction, and each retry
    retires at most one oldest entry. Width-change deferral and ordinary
    preparation must not destroy all warm prompt checkpoints.
    """
    clear_scratch()
    if admission.get("optional_reclaim_wait") or admission.get("stage") != "queue" or not len(apc):
        return False
    evicted = apc.evict_oldest_unleased()
    if evicted:
        clear_scratch()
    return evicted


class Overloaded(RuntimeError):
    pass


class AdmissionClosed(RuntimeError):
    """A model-executing request reached a service whose admission gate is shut."""

    def __init__(self, state, endpoint_class="generation"):
        self.state = str(state)
        self.endpoint_class = str(endpoint_class)
        self.status = 503
        self.code = "server_unavailable"
        super().__init__(f"server is {self.state}; retry after resume")


class SuspendUnavailable(RuntimeError):
    """Cache suspension was requested without an APCv2 disk tier."""


@dataclass
class Job:
    request: dict
    tenant_id: str = "default"
    fault: BatchFaultSpec | None = None
    id: str = field(default_factory=lambda: "chatcmpl-" + uuid.uuid4().hex)
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=256))
    cancelled: threading.Event = field(default_factory=threading.Event)
    created: float = field(default_factory=time.time)
    started: float = 0
    first_token: float | None = None
    prompt_tokens: int = 0
    effective_max_tokens: int | None = None
    max_tokens_defaulted: bool | None = None
    cached_tokens: int = 0
    cache_retention_role: str | None = None
    completion_tokens: int = 0
    cache_branch: object = None
    detokenizer: object = None
    output_parser: object = None
    pending_text: str = ""
    uid: int | None = None
    last_progress: float = 0
    observed_width: int = 1
    admission_hit: object = None
    admission_tokens: list | None = None
    admission_retry_at: float = 0
    admission_deadline: float = 0
    fault_fired: bool = False
    fanout_group: str | None = None
    fanout_role: str | None = None
    spomin_receipt: dict | None = None
    cache_capsule: object = None
    cache_capsule_rows: int = 0
    cache_capsule_receipt: dict | None = None
    fanout_boundary_tokens: int = 0
    fanout_one_prefill: bool = False
    fanout_reason: str | None = None
    structured: object = None
    thinking_guard: object = None
    thinking_budget: object = None
    thinking_budget_counted: bool = False
    thinking_budget_fired: bool = False
    thinking_budget_resolved: bool = False
    thinking_tokens: list[int] = field(default_factory=list)
    tool_parse_fallbacks_seen: int = 0
    tool_grammar_status: str = "disabled"
    approximate_kv_receipt: dict | None = None
    approximate_kv_applied: bool = False
    admission_final_reclaim_done: bool = False
    apc_interior_positions: tuple[int, ...] = ()
    effective_sampling: dict | None = None
    sampling_defaults: dict | None = None


@dataclass(frozen=True)
class PublishedCohort:
    """One queue item that makes every reserved cohort member visible at once.

    ``atomic`` cohorts are client-declared ``batch_cohort`` groups: they own
    the idle attachment boundary and attach or fail together.  Non-atomic
    cohorts only preserve publication order; APCv2 fanout siblings use this
    so they join the leader's live batch instead of waiting for it to finish.
    """

    jobs: tuple[Job, ...]
    atomic: bool = True


class ServingEngine:
    MEMORY_ADMISSION_TIMEOUT = 60.0
    MEMORY_ADMISSION_RETRY = 0.25

    def __init__(
        self,
        model_path,
        *,
        adapter_factory=None,
        max_inflight=8,
        max_lanes=4,
        max_context=16384,
        default_max_tokens=DEFAULT_OUTPUT_TOKENS,
        max_request_bytes=2 << 20,
        prefill_step=2048,
        cache_bytes=12 << 30,
        cache_dir=None,
        host_prompt_cache_entries=128,
        host_prompt_cache_tokens=1 << 20,
        coalesce_window_ms=5.0,
        batch_cohort_timeout_ms=1000.0,
        mtp=True,
        prompt_lookup=False,
        qualification_mode=False,
        qualification=None,
        execution_policy=None,
        tenant_scoped_cache=False,
        adaptive_mtp_depth=None,
        spomin_live_surgery=None,
        thinking_budget=None,
        thinking_steer_alpha=None,
        thinking_steer_hammer=None,
        thinking_auto_calibration=True,
        cache_capsules=None,
        persistent_block_bytes=0,
        approximate_kv=None,
        lora_root=None,
        reasoning_signing_key_file=None,
        reasoning_signing_key_env=None,
        apc_persist_dir=None,
        apc_persist_on_shutdown=False,
        apc_persist_shutdown_seconds=5,
        apc_persist_corruption="quarantine",
        apc_session_max_ttl_seconds=3600,
        apc_session_pinned_disk_bytes=64 << 30,
        apc_session_pinned_disk_bytes_global=64 << 30,
        apc_session_pinned_resident_bytes=12 << 30,
        apc_session_prefetch_ttl_seconds=30,
        apc_quarantine_max_entries=128,
        apc_quarantine_max_bytes=1 << 30,
        int8_prefill=None,
        _validate_only=False,
    ):
        self.default_max_tokens = validate_default_max_tokens(default_max_tokens)
        if min(
            max_inflight,
            max_lanes,
            max_context,
            max_request_bytes,
            prefill_step,
            cache_bytes,
        ) <= 0:
            raise ValueError("limits must be positive")
        if max_lanes > max_inflight:
            raise ValueError("max_lanes must not exceed max_inflight")
        self.started_at = time.monotonic()
        if execution_policy is not None and not isinstance(execution_policy, dict):
            raise ValueError("execution policy must be a JSON object")
        self.execution_policy = execution_policy
        for policy_name in ("constrained_tool_grammar", "tolerant_tool_markers"):
            policy_value = (execution_policy or {}).get(policy_name, False)
            if type(policy_value) is not bool:
                raise ValueError(f"{policy_name} must be boolean")
            setattr(self, policy_name, policy_value)
        self.apc_interior_checkpoint_policy = apc_interior_checkpoint_policy(
            (execution_policy or {}).get("apc_interior_checkpoints")
        )
        from .runtime.adaptive_policy import AdaptiveMTPDepthPolicy
        from .runtime.speculative_sampling import FLyVerificationPolicy
        from .runtime.spomin_live_surgery import ServingSpominPolicy

        self.adaptive_mtp_policy = AdaptiveMTPDepthPolicy.from_value(
            adaptive_mtp_depth
        )
        self.fly_verification_policy = FLyVerificationPolicy.from_value(
            (execution_policy or {}).get("fly_verification")
        )
        self.spomin_policy = ServingSpominPolicy.from_value(spomin_live_surgery)
        self.spomin_manager = None
        # None means "not configured": the adapter's own defaults then apply
        # (resolved once the adapter is loaded).  An explicit 0 disables.
        self._thinking_overrides = {
            "thinking_budget": thinking_budget,
            "thinking_steer_alpha": thinking_steer_alpha,
            "thinking_steer_hammer": thinking_steer_hammer,
        }
        self.thinking_budget = int(thinking_budget or 0)
        self.thinking_steer_alpha = float(thinking_steer_alpha or 0.0)
        self.thinking_steer_hammer = float(thinking_steer_hammer or 0.0)
        self.thinking_defaults_source = "server"
        self.thinking_auto_calibration = bool(thinking_auto_calibration)
        self._commit_direction = None
        self.thinking_steer_status = {"state": "off"}
        self.cache_capsule_policy = cache_capsule_policy(cache_capsules)
        if isinstance(persistent_block_bytes, bool) or not isinstance(
            persistent_block_bytes, int
        ) or persistent_block_bytes < 0:
            raise ValueError("persistent_block_bytes must be a nonnegative integer")
        self.persistent_block_bytes = persistent_block_bytes
        if self.cache_capsule_policy["enabled"] and not qualification_mode:
            raise ValueError("cache capsules are restricted to qualification mode")
        if self.persistent_block_bytes and not qualification_mode:
            raise ValueError("block persistence is restricted to qualification mode")
        if self.persistent_block_bytes and not (cache_dir or apc_persist_dir):
            raise ValueError("block persistence requires cache_dir")
        if cache_dir and apc_persist_dir and Path(cache_dir).expanduser().resolve() != Path(apc_persist_dir).expanduser().resolve():
            raise ValueError("cache_dir and apc_persist_dir must name the same directory")
        if not isinstance(apc_persist_on_shutdown, bool):
            raise ValueError("apc_persist_on_shutdown must be boolean")
        if apc_persist_on_shutdown and not apc_persist_dir:
            raise ValueError("apc_persist_on_shutdown requires apc_persist_dir")
        if apc_persist_corruption not in {"quarantine", "delete"}:
            raise ValueError("apc_persist_corruption must be quarantine or delete")
        if isinstance(apc_persist_shutdown_seconds, bool) or not isinstance(
            apc_persist_shutdown_seconds, (int, float)
        ) or not math.isfinite(apc_persist_shutdown_seconds) or apc_persist_shutdown_seconds <= 0:
            raise ValueError("apc_persist_shutdown_seconds must be finite and positive")
        for name, value in (
            ("apc_session_max_ttl_seconds", apc_session_max_ttl_seconds),
            ("apc_session_pinned_disk_bytes", apc_session_pinned_disk_bytes),
            ("apc_session_pinned_resident_bytes", apc_session_pinned_resident_bytes),
            ("apc_session_prefetch_ttl_seconds", apc_session_prefetch_ttl_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(apc_session_pinned_disk_bytes_global, bool)
            or not isinstance(apc_session_pinned_disk_bytes_global, int)
            or apc_session_pinned_disk_bytes_global < 0
        ):
            raise ValueError(
                "apc_session_pinned_disk_bytes_global must be a non-negative integer"
            )
        self.apc_persist_dir = apc_persist_dir
        self.apc_persist_on_shutdown = apc_persist_on_shutdown
        self.apc_persist_shutdown_seconds = float(apc_persist_shutdown_seconds)
        self.apc_persist_corruption = apc_persist_corruption
        self.apc_session_max_ttl_seconds = apc_session_max_ttl_seconds
        self.apc_session_pinned_disk_bytes = apc_session_pinned_disk_bytes
        if apc_session_pinned_disk_bytes_global > 64 << 30:
            raise ValueError(
                "apc_session_pinned_disk_bytes_global must not exceed the 64 GiB disk tier cap"
            )
        self.apc_session_pinned_disk_bytes_global = (
            apc_session_pinned_disk_bytes_global
        )
        self.apc_session_pinned_resident_bytes = apc_session_pinned_resident_bytes
        self.apc_session_prefetch_ttl_seconds = apc_session_prefetch_ttl_seconds
        for name, value in (
            ("apc_quarantine_max_entries", apc_quarantine_max_entries),
            ("apc_quarantine_max_bytes", apc_quarantine_max_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        self.apc_quarantine_max_entries = apc_quarantine_max_entries
        self.apc_quarantine_max_bytes = apc_quarantine_max_bytes
        if apc_session_prefetch_ttl_seconds > apc_session_max_ttl_seconds:
            raise ValueError(
                "apc_session_prefetch_ttl_seconds must not exceed the maximum TTL"
            )
        if self.cache_capsule_policy["enabled"] and (mtp or prompt_lookup):
            raise ValueError("cache capsules are qualified only for ordinary decode")
        if self.adaptive_mtp_policy.enabled:
            if not qualification_mode:
                raise ValueError(
                    "adaptive MTP depth is restricted to qualification mode"
                )
            if not mtp or prompt_lookup:
                raise ValueError("adaptive MTP depth requires the native self-MTP route")
        if self.spomin_policy.enabled:
            if not qualification_mode:
                raise ValueError(
                    "live Spomin surgery is approximate and restricted to qualification mode"
                )
            if mtp:
                raise ValueError("live Spomin surgery is incompatible with MTP")
            if prompt_lookup:
                raise ValueError("live Spomin surgery is incompatible with prompt lookup")
        if execution_policy and execution_policy.get("allow_unverified_indexed") and not qualification_mode:
            raise ValueError("unverified kernels are restricted to qualification mode")
        from .runtime.int8_prefill import Int8PrefillPolicy

        # W8A8 int8 NAX prefill: default off, approximate by construction.
        # The handle is bound in the worker once the adapter's model exists.
        self.int8_prefill_policy = Int8PrefillPolicy.from_value(int8_prefill)
        self.int8_prefill_handle = None
        from .runtime.approximate_kv import ServingApproximateKVPolicy

        self.approximate_kv_policy = ServingApproximateKVPolicy.from_value(
            approximate_kv
        )
        if self.approximate_kv_policy.enabled:
            if not qualification_mode and not qualification:
                raise ValueError(
                    "approximate KV is restricted to qualification mode unless a "
                    "qualification record carries its evidence"
                )
            if mtp:
                raise ValueError("approximate KV is incompatible with MTP")
            if prompt_lookup:
                raise ValueError("approximate KV is incompatible with prompt lookup")
            if self.spomin_policy.enabled:
                raise ValueError(
                    "approximate KV is incompatible with live Spomin surgery"
                )
            if self.cache_capsule_policy["enabled"]:
                raise ValueError("approximate KV is incompatible with cache capsules")
            if self.approximate_kv_policy.start_tokens and max_lanes != 1:
                # Exact and quantized lanes cannot merge into one batch.
                raise ValueError(
                    "approximate KV start_tokens > 0 requires max_lanes=1"
                )
        self.model_path = model_path
        self.lora_root = (
            Path(lora_root).expanduser().resolve() if lora_root is not None else None
        )
        self.lora_session = None
        self.max_inflight, self.max_lanes = max_inflight, max_lanes
        self.max_context, self.max_request_bytes = max_context, max_request_bytes
        self.prefill_step = prefill_step
        self.cache_bytes, self.cache_dir = cache_bytes, cache_dir
        self.host_prompt_cache = HostPromptCache(
            max_entries=host_prompt_cache_entries,
            max_tokens=host_prompt_cache_tokens,
        )
        self.coalesce_window_seconds = coalescing_window(coalesce_window_ms)
        if isinstance(batch_cohort_timeout_ms, bool):
            raise ValueError("batch cohort timeout must be numeric milliseconds")
        try:
            batch_cohort_timeout_ms = float(batch_cohort_timeout_ms)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "batch cohort timeout must be numeric milliseconds"
            ) from error
        if not math.isfinite(batch_cohort_timeout_ms) or batch_cohort_timeout_ms <= 0:
            raise ValueError("batch cohort timeout must be finite and positive")
        self.batch_cohort_timeout_seconds = batch_cohort_timeout_ms / 1000.0
        self.mtp = mtp
        self.prompt_lookup = bool(prompt_lookup)
        self.qualification_mode, self.qualification = qualification_mode, qualification
        # Off by default: one prefix cache shared by every client is the point
        # of the single-user lab deployment.  On, the APCv2 namespace carries
        # the (unauthenticated) X-Tenant-ID so a client cannot warm-hit, or
        # probe through cached_tokens, another tenant's prompts.
        self.tenant_scoped_cache = bool(tenant_scoped_cache)
        if _validate_only:
            return
        from .reasoning_signatures import ReasoningSigner

        self.reasoning_signer = ReasoningSigner.configured(
            key_file=reasoning_signing_key_file,
            key_env=reasoning_signing_key_env,
        )
        self.incoming = queue.Queue(maxsize=max_inflight)
        self.slots = threading.BoundedSemaphore(max_inflight)
        self.stop_event, self.ready = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self._quiesce_complete = threading.Event()
        self._quiesce_complete.set()
        now_wall = time.time()
        self._service_state = "serving"
        self._service_state_since = now_wall
        self._service_timestamps = {"serving": now_wall}
        self._last_service_transition = {
            "from": None,
            "to": "serving",
            "at": now_wall,
            "result": {"status": "initialized"},
        }
        self._drain_deadline = None
        self._drain_suspend = False
        self._drain_started_monotonic = None
        self._quiesce_worker_owned = False
        self._admin_prefetch_queue = deque()
        self._admin_prefetch_limit = 32
        self._admission_leases = set()
        # Tokenizer chat-template rendering is shared by generation admission
        # and the read-only Anthropic count_tokens endpoint.  Keep those calls
        # serialized: several tokenizer implementations maintain scratch state.
        self.prompt_lock = threading.Lock()
        self.adapter = None
        self.apc = None
        self.submission_lock = threading.Lock()
        self.jobs = {}
        self.pending_cohorts = {}
        self.fanout_waiting = {}
        self.fanout_capsules = {}
        self.queued_jobs = 0
        self.counts = Counter(
            {
                "thinking_budget_forced_closes": 0,
                "tool_call_parse_fallbacks": 0,
                "tool_call_constraint_failures": 0,
                "schema_ref_failures": 0,
                "structured_output_dead_ends": 0,
                "constrained_tool_grammar_engagements": 0,
                "constrained_tool_grammar_skips": 0,
                "tolerant_tool_marker_requests": 0,
                "quiesce_requests": 0,
                "drains_completed": 0,
                "drain_timeouts": 0,
                "jobs_drained": 0,
                "suspends": 0,
                "suspended_entries": 0,
                "suspended_bytes": 0,
                "suspend_failures": 0,
                "resumes": 0,
                "prefetches_queued": 0,
                "prefetches_cancelled": 0,
                **{
                    f"admissions_rejected_{endpoint}": 0
                    for endpoint in self._ADMISSION_CLASSES
                },
            }
        )
        self._memory_reclaim_lock = threading.Lock()
        self._memory_reclaim_last = 0.0
        self.apc_interior_route_supported = not self.prompt_lookup
        self.receipts = deque(maxlen=128)
        self.batch_metrics = BatchRuntimeMetrics()
        self.http_metrics = HttpRuntimeMetrics()
        self.snapshot = {"state": "loading"}
        self.adapter = None
        self.apc = None
        self.model_revision = 0
        self.operation_receipts = deque(maxlen=128)
        self.route_capabilities = frozenset()
        self.sampling_vendor = None
        self.error = None
        if adapter_factory is None:
            from .adapters.registry import resolve_adapter

            adapter_factory = resolve_adapter(model_path, mtp=mtp)
        self.adapter_factory = adapter_factory
        self.thread = threading.Thread(
            target=self._run, name="mlx2-generation", daemon=True
        )
        self.thread.start()

    @classmethod
    def validate_arguments(cls, model_path, **kwargs):
        """Run constructor argument validation without resolving or loading a model.

        The validation-only path is the same constructor prefix used by normal
        startup and returns before reading operational signing keys, resolving
        an adapter, or starting the generation thread.
        """
        cls(model_path, _validate_only=True, **kwargs)

    _ADMISSION_CLASSES = frozenset(
        {"generation", "embeddings", "rerank", "audio", "batch", "session_prefetch"}
    )

    def _transition_service_locked(self, state, result):
        previous = self._service_state
        now_wall = time.time()
        self._service_state = str(state)
        self._service_state_since = now_wall
        self._service_timestamps[self._service_state] = now_wall
        self._last_service_transition = {
            "from": previous,
            "to": self._service_state,
            "at": now_wall,
            "result": dict(result),
        }

    def _service_state_locked(self):
        deadline = self._drain_deadline
        return {
            "state": self._service_state,
            "since": self._service_state_since,
            "timestamps": dict(self._service_timestamps),
            "drain_deadline": (
                time.time() + max(0.0, deadline - time.monotonic())
                if deadline is not None and self._service_state == "draining"
                else None
            ),
            "suspend_requested": bool(
                self._drain_suspend and self._service_state == "draining"
            ),
            "last_transition": {
                **self._last_service_transition,
                "result": dict(self._last_service_transition.get("result") or {}),
            },
        }

    def service_state(self):
        with self.lock:
            return self._service_state_locked()

    def _ensure_admission(self, endpoint_class="generation", *, admitted=False):
        endpoint_class = str(endpoint_class)
        if endpoint_class not in self._ADMISSION_CLASSES:
            endpoint_class = "generation"
        with self.lock:
            # A few narrow unit fixtures construct the engine with __new__ to
            # exercise queue mechanics in isolation.  They predate the service
            # lifecycle and represent an ordinary serving engine.
            state = getattr(self, "_service_state", "serving")
            if state == "serving" or (
                admitted and state == "draining" and not self._quiesce_worker_owned
            ):
                return
            self.counts[f"admissions_rejected_{endpoint_class}"] += 1
        raise AdmissionClosed(state, endpoint_class)

    def quiesce(self, *, drain_timeout_seconds=600.0, suspend=True):
        if isinstance(drain_timeout_seconds, bool) or not isinstance(
            drain_timeout_seconds, (int, float)
        ):
            raise ValueError("drain_timeout_seconds must be numeric")
        drain_timeout_seconds = float(drain_timeout_seconds)
        if not math.isfinite(drain_timeout_seconds) or not 0.1 <= drain_timeout_seconds <= 3600:
            raise ValueError(
                "drain_timeout_seconds must be between 0.1 and 3600"
            )
        if not isinstance(suspend, bool):
            raise ValueError("suspend must be boolean")
        with self.submission_lock:
            with self.lock:
                self.counts["quiesce_requests"] += 1
                if self._service_state != "serving":
                    return self._service_state_locked()
                if suspend and not (self.apc_persist_dir or self.cache_dir):
                    raise SuspendUnavailable(
                        "cache suspension requires --apc-persist-dir or --cache-dir"
                    )
                now = time.monotonic()
                self._drain_deadline = now + drain_timeout_seconds
                self._drain_started_monotonic = now
                self._drain_suspend = suspend
                self._quiesce_worker_owned = False
                self._quiesce_complete.clear()
                self._transition_service_locked(
                    "draining",
                    {
                        "status": "accepted",
                        "drain_timeout_seconds": drain_timeout_seconds,
                        "suspend": suspend,
                    },
                )
                return self._service_state_locked()

    def resume(self, *, prefetch_sessions=()):
        prefetch_sessions = tuple(prefetch_sessions)
        if len(prefetch_sessions) > self._admin_prefetch_limit:
            raise ValueError(
                f"prefetch_sessions accepts at most {self._admin_prefetch_limit} entries"
            )
        # Holding submission_lock and lock makes drain cancellation atomic with
        # the worker's claim of the idle boundary. If the worker has already
        # claimed it (state is quiesced/suspended but completion is not set),
        # release both locks and wait until cache ownership is stable.
        while True:
            wait_for_worker = False
            with self.submission_lock:
                with self.lock:
                    if (
                        len(self._admin_prefetch_queue) + len(prefetch_sessions)
                        > self._admin_prefetch_limit
                    ):
                        raise ValueError("admin prefetch queue is full")
                    previous = self._service_state
                    if (
                        (previous != "draining" or self._quiesce_worker_owned)
                        and not self._quiesce_complete.is_set()
                    ):
                        wait_for_worker = True
                    else:
                        if previous != "serving":
                            self.counts["resumes"] += 1
                            self._drain_deadline = None
                            self._drain_started_monotonic = None
                            self._drain_suspend = False
                            self._quiesce_worker_owned = False
                            self._transition_service_locked(
                                "serving",
                                {
                                    "status": "drain_cancelled"
                                    if previous == "draining"
                                    else "resumed",
                                    "prefetches_queued": len(prefetch_sessions),
                                },
                            )
                            self._quiesce_complete.set()
                        self._admin_prefetch_queue.extend(prefetch_sessions)
                        self.counts["prefetches_queued"] += len(prefetch_sessions)
                        return self._service_state_locked()
            if wait_for_worker:
                self._quiesce_complete.wait()

    def wait_for_quiesce(self, timeout=None):
        return self._quiesce_complete.wait(timeout)

    def acquire_admission(self, endpoint_class="generation"):
        """Own one accepted multi-round HTTP lifecycle until its final reply."""
        with self.submission_lock:
            self._ensure_admission(endpoint_class)
            token = object()
            with self.lock:
                self._admission_leases.add(token)
            return token

    def release_admission(self, token):
        if token is None:
            return
        with self.lock:
            self._admission_leases.discard(token)

    def admit_batch_submission(self, callback):
        """Atomically admit one Batch API resource before its worker starts."""
        with self.submission_lock:
            self._ensure_admission("batch")
            return callback()

    def _active_batch_jobs(self):
        manager = getattr(self, "api_resources", {}).get("batches")
        if manager is None or not hasattr(manager, "status"):
            return 0
        states = manager.status().get("states") or {}
        return sum(int(states.get(name, 0)) for name in ("validating", "in_progress", "cancelling"))

    def _cancel_pending_prefetches(self, apc=None):
        """Cancel queued admin/APCv2 restores without executing MLX work."""
        with self.lock:
            cancelled = len(self._admin_prefetch_queue)
            self._admin_prefetch_queue.clear()
        apc = self.apc if apc is None else apc
        cancel_apc = getattr(apc, "cancel_pending_prefetch", None)
        if callable(cancel_apc) and cancel_apc():
            cancelled += 1
        if cancelled:
            with self.lock:
                self.counts["prefetches_cancelled"] += cancelled
        return cancelled

    def _worker_quiesce_action(self):
        with self.lock:
            if self._service_state != "draining":
                return None
            if self._quiesce_worker_owned:
                return None
        # Serving is the overwhelmingly common path. Only a real drain may
        # inspect BatchManager or APCv2, whose locks can cover disk persistence.
        active_batches = self._active_batch_jobs()
        apc = self.apc
        has_pending_prefetch = bool(
            apc is not None and getattr(apc, "has_pending_prefetch", False)
        )
        # Stabilize admission while claiming the boundary so resume cannot
        # reopen the service between the external snapshots and the claim.
        with self.submission_lock:
            with self.lock:
                if (
                    self._service_state != "draining"
                    or self._quiesce_worker_owned
                ):
                    return None
                timed_out = (
                    self._drain_deadline is not None
                    and time.monotonic() >= self._drain_deadline
                )
                if not timed_out and (
                    self.jobs
                    or active_batches
                    or self._admission_leases
                    or self._admin_prefetch_queue
                    or has_pending_prefetch
                ):
                    return None
                elapsed = max(
                    0.0,
                    time.monotonic()
                    - (self._drain_started_monotonic or time.monotonic()),
                )
                if timed_out:
                    self.counts["drain_timeouts"] += 1
                else:
                    self.counts["drains_completed"] += 1
                target = "suspended" if self._drain_suspend else "quiesced"
                self._quiesce_worker_owned = True
                action = {
                    "target": target,
                    "timed_out": timed_out,
                    "suspend": self._drain_suspend,
                    "drain_duration_seconds": elapsed,
                }
            if action["timed_out"]:
                self._cancel_pending_prefetches(apc)
            return action

    def _complete_worker_quiesce(self, action, suspend_report=None):
        with self.lock:
            result = {
                "status": "completed",
                "drain_timed_out": bool(action["timed_out"]),
                "drain_duration_seconds": float(action["drain_duration_seconds"]),
            }
            if suspend_report is not None:
                result["suspend"] = dict(suspend_report)
            self._transition_service_locked(action["target"], result)
            self._drain_deadline = None
            self._drain_started_monotonic = None
            self._drain_suspend = False
            self._quiesce_worker_owned = False
            self._quiesce_complete.set()

    def submit(
        self,
        request,
        *,
        tenant_id="default",
        admission_class="generation",
        admitted=False,
    ):
        if not self.ready.is_set() or self.error or not self.thread.is_alive():
            raise RuntimeError(self.error or "model is not ready")
        with self.submission_lock:
            self._ensure_admission(admission_class, admitted=admitted)
            if not self.slots.acquire(blocking=False):
                self.batch_metrics.rejected("maximum_inflight", self.queued_jobs)
                raise Overloaded("maximum inflight requests reached")
            try:
                job = self._prepare_job(request, tenant_id=tenant_id)
                self._publish_job(job)
                return job
            except BaseException:
                self.slots.release()
                raise

    def count_tokens(self, request):
        """Render a chat request through the loaded adapter without generating."""
        if not self.ready.is_set() or self.error or not self.thread.is_alive():
            raise RuntimeError(self.error or "model is not ready")
        with self.prompt_lock:
            adapter = self.adapter
            if adapter is None:
                raise RuntimeError("model adapter is not ready")
            return len(adapter.prompt_tokens(request))

    def _prepare_job(self, request, *, tenant_id):
        from .contracts import Capability
        from .output import constrained_tool_choice

        if (
            "grammar" in request or "response_format" in request
            or (
                getattr(self, "constrained_tool_grammar", False)
                and constrained_tool_choice(request)
            )
        ) and request.get("response_format") != {"type": "text"}:
            if Capability.GRAMMAR not in self.route_capabilities:
                raise ValueError("structured output is not available on this route")
        profile = request.get("sampling_profile")
        if profile is not None:
            # Fail an unknown profile before it reserves a lane; admission
            # repeats the check against the loaded adapter.
            vendor = getattr(self, "sampling_vendor", None)
            if vendor is None:
                raise ValueError("this model declares no sampling profiles")
            vendor.select(thinking=None, requested=profile)
        public_request = {key: value for key, value in request.items() if key != "mlx_fault"}
        has_media = any(
            isinstance(message.get("content"), list)
            for message in public_request.get("messages", ())
        )
        if has_media:
            prepare = getattr(self.adapter, "prepare_multimodal_request", None)
            if not callable(prepare):
                from .api_resources import CapabilityUnavailable

                raise CapabilityUnavailable(
                    "the loaded adapter has no qualified image, video, or audio encoder"
                )
            media_loader = getattr(self, "media_file_loader", None)
            file_loader = (
                (lambda file_id: media_loader(tenant_id, file_id))
                if media_loader is not None
                else None
            )
            public_request = prepare(public_request, file_loader=file_loader)
            if not isinstance(public_request, dict):
                raise RuntimeError(
                    "multimodal adapter hook must return a request object"
                )
            for name, value in public_request.get("_mlx2_multimodal_stats", {}).items():
                if (
                    name in {
                        "gemma3n_video_requests",
                        "gemma3n_video_frames",
                        "gemma3n_video_frame_batches",
                        "minicpmo_vision_batches",
                        "minicpmo_vision_slices",
                        "minicpmo_audio_chunks",
                    }
                    and isinstance(value, int)
                    and value >= 0
                ):
                    self.counts[name] += value
        fault = BatchFaultSpec.parse(
            request.get("mlx_fault"), enabled=self.qualification_mode
        )
        if request.get("skip_writing_prefix_cache", False):
            self.counts["apcv2_write_suppressed_requests"] += 1
        interior_policy = getattr(
            self,
            "apc_interior_checkpoint_policy",
            {"count": 0, "min_stride": 1},
        )
        if interior_policy["count"] and not getattr(
            self, "apc_interior_route_supported", False
        ):
            self.counts["apc_interior_checkpoints_skipped_route"] += 1
        return Job(
            public_request, tenant_id=str(tenant_id or "default"), fault=fault
        )

    def _clear_allocator_cache_before_reject(self, *, synchronize=False):
        """Rate-limited allocator reclaim shared by every admission seam."""
        now = time.monotonic()
        lock = getattr(self, "_memory_reclaim_lock", None)
        if lock is None:
            lock = self._memory_reclaim_lock = threading.Lock()
        with lock:
            last = float(getattr(self, "_memory_reclaim_last", 0.0))
            # Synchronized admission reclamation is the last non-destructive
            # step before eviction and must really run for this attempt.  Cheap
            # no-eviction remeasure paths remain coalesced by the shared gate.
            if (
                not synchronize
                and now - last < self.MEMORY_ADMISSION_RETRY
            ):
                return False
            self._memory_reclaim_last = now
        import mlx.core as mx

        if synchronize:
            mx.synchronize()
        mx.clear_cache()
        counts = getattr(self, "counts", None)
        if counts is not None:
            counts["memory_cache_reclaims_before_reject"] += 1
        return True

    def _permit_allocator_reclaim_after_eviction(self):
        with self._memory_reclaim_lock:
            self._memory_reclaim_last = 0.0

    def _publish_job(self, job):
        cohort = job.request.get("batch_cohort")
        if cohort is not None:
            cohort_id = cohort["id"]
            cohort_key = (job.tenant_id, cohort_id)
            target = cohort["size"]
            if target > self.max_lanes:
                raise Overloaded(
                    "batch cohort size exceeds configured lane capacity"
                )
            pending = self.pending_cohorts.get(cohort_key)
            if pending is None:
                pending = {
                    "size": target,
                    "created": time.monotonic(),
                    "jobs": [],
                }
                self.pending_cohorts[cohort_key] = pending
            elif pending["size"] != target:
                raise ValueError("batch cohort size changed before publication")
            if any(member.id == job.id for member in pending["jobs"]):
                raise ValueError("duplicate batch cohort request")
            with self.lock:
                self.jobs[job.id] = job
            pending["jobs"].append(job)
            self.counts["batch_cohort_jobs_staged"] += 1
            if len(pending["jobs"]) < target:
                return
            jobs = pending["jobs"]
            del self.pending_cohorts[cohort_key]
            self.counts["batch_cohort_releases"] += 1
            self.counts["batch_cohort_jobs_released"] += len(jobs)
            self._publish_jobs(jobs, already_registered=True)
            return
        self._publish_jobs([job])

    def _publish_jobs(self, jobs, *, already_registered=False, atomic=True):
        jobs = tuple(jobs)
        if not jobs:
            return
        with self.lock:
            if not already_registered:
                for job in jobs:
                    self.jobs[job.id] = job
            initial_depth = self.queued_jobs
            self.queued_jobs += len(jobs)
        item = jobs[0] if len(jobs) == 1 else PublishedCohort(jobs, atomic=atomic)
        self.incoming.put_nowait(item)
        for offset, job in enumerate(jobs, 1):
            self.batch_metrics.admitted(
                job.id, job.tenant_id, initial_depth + offset
            )

    def _expire_pending_cohorts(self):
        now = time.monotonic()
        expired = []
        with self.submission_lock:
            for cohort_key, pending in list(self.pending_cohorts.items()):
                if now - pending["created"] >= self.batch_cohort_timeout_seconds:
                    expired.append((cohort_key, pending))
                    del self.pending_cohorts[cohort_key]
        for (_, cohort_id), pending in expired:
            self.counts["batch_cohort_timeouts"] += 1
            self.counts["batch_cohort_jobs_timed_out"] += len(pending["jobs"])
            for job in pending["jobs"]:
                self._finish(
                    job,
                    {
                        "error": (
                            f"batch cohort {cohort_id!r} received "
                            f"{len(pending['jobs'])}/{pending['size']} requests "
                            "before its publication deadline"
                        ),
                        "status": 429,
                    },
                )

    def submit_many(
        self,
        requests,
        *,
        tenant_id="default",
        admission_class="generation",
        admitted=False,
    ):
        """Reserve parallel samples and publish one exact-prefill leader.

        Siblings become runnable only after the leader publishes its committed
        prompt boundary into APCv2. Their ordinary APC lookup then creates
        revision-bound COW branches from that one frozen generation.
        """
        requests = list(requests)
        if not requests:
            raise ValueError("parallel submission requires at least one request")
        if len(requests) > 1 and getattr(
            getattr(self, "spomin_policy", None), "enabled", False
        ):
            # Siblings attach to the leader's exact APCv2 boundary; a compacted
            # boundary is approximate and is never published.
            raise ValueError("parallel samples are unavailable with live Spomin surgery")
        if not self.ready.is_set() or self.error or not self.thread.is_alive():
            raise RuntimeError(self.error or "model is not ready")
        jobs = [
            self._prepare_job(request, tenant_id=tenant_id) for request in requests
        ]
        prompt_keys = {HostPromptCache.key(job.request) for job in jobs}
        if len(prompt_keys) != 1:
            raise ValueError("parallel samples must share one rendered prompt")
        # An approximate leader never publishes its prompt boundary, so there
        # is nothing for siblings to lease: every sample prefills on its own.
        write_suppressed = any(
            job.request.get("skip_writing_prefix_cache", False) for job in jobs
        )
        independent = bool(
            getattr(getattr(self, "approximate_kv_policy", None), "enabled", False)
            or write_suppressed
        )
        fanout_group = "fanout-" + uuid.uuid4().hex
        for index, job in enumerate(jobs):
            if independent:
                break
            job.fanout_group = fanout_group
            job.fanout_role = "prefill_leader" if index == 0 else "apcv2_sibling"
        reserved = 0
        with self.submission_lock:
            self._ensure_admission(admission_class, admitted=admitted)
            for _ in jobs:
                if not self.slots.acquire(blocking=False):
                    for _ in range(reserved):
                        self.slots.release()
                    self.batch_metrics.rejected(
                        "maximum_inflight", self.queued_jobs
                    )
                    raise Overloaded(
                        "parallel samples exceed available inflight capacity"
                    )
                reserved += 1
            try:
                with self.lock:
                    for job in jobs:
                        self.jobs[job.id] = job
                    if not independent:
                        self.fanout_waiting[fanout_group] = tuple(jobs[1:])
                if independent:
                    self._publish_jobs(jobs, already_registered=True, atomic=False)
                    if write_suppressed:
                        self.counts["apcv2_fanout_write_suppressed"] += 1
                    else:
                        self.counts["approximate_kv_fanout_bypassed"] += 1
                else:
                    self._publish_jobs([jobs[0]], already_registered=True)
                    self.counts["apcv2_fanout_groups"] += 1
                    self.counts["apcv2_fanout_lanes"] += len(jobs)
            except BaseException:
                with self.lock:
                    self.fanout_waiting.pop(fanout_group, None)
                    for job in jobs:
                        self.jobs.pop(job.id, None)
                for _ in range(reserved):
                    self.slots.release()
                raise
        return jobs

    def status(self):
        with self.lock:
            quiesce = self._service_state_locked()
            return {
                **self.snapshot,
                "healthy": self.ready.is_set()
                and self.thread.is_alive()
                and not self.error,
                "error": self.error,
                "inflight": len(self.jobs),
                "accepted_lifecycles": len(self._admission_leases),
                "queue_depth": self.queued_jobs,
                "counts": dict(self.counts),
                "quiesce": quiesce,
                "admissions_rejected": {
                    endpoint: self.counts[f"admissions_rejected_{endpoint}"]
                    for endpoint in sorted(self._ADMISSION_CLASSES)
                },
                "approximate_kv": {
                    **(self.snapshot.get("approximate_kv") or {}),
                    "applied": self.counts["approximate_kv_applied"],
                    "declined": self.counts["approximate_kv_declined"],
                    "requantized_prefix_hits": self.counts[
                        "approximate_kv_requantized_prefix_hits"
                    ],
                    "apcv2_store_skipped_approximate": self.counts[
                        "apcv2_store_skipped_approximate"
                    ],
                },
                "int8_prefill": int8_prefill_status(self),
                "recent_receipts": list(self.receipts),
                "recent_operation_receipts": list(self.operation_receipts),
                "model_revision": self.model_revision,
                "lora": {
                    "enabled": self.lora_root is not None,
                    "active": self.lora_session.name
                    if self.lora_session is not None
                    else None,
                },
                "host_prompt_cache": self.host_prompt_cache.status(),
                "reasoning_signing": {
                    "key_id": self.reasoning_signer.key_id,
                    "ephemeral": self.reasoning_signer.ephemeral,
                },
                "spomin_live_surgery": (
                    self.spomin_manager.snapshot()
                    if getattr(self, "spomin_manager", None) is not None
                    else {"enabled": False}
                ),
            }

    def _exclusive_adapter_operation(
        self,
        operation,
        callback,
        *,
        timeout=30.0,
        admission_class="generation",
        admitted=False,
    ):
        """Drain generation and run one revision-sensitive adapter operation."""
        if not self.ready.is_set() or self.error or not self.thread.is_alive():
            raise RuntimeError(self.error or "model is not ready")
        started = time.monotonic()
        with self.submission_lock:
            self._ensure_admission(admission_class, admitted=admitted)
            while True:
                with self.lock:
                    inflight = len(self.jobs)
                if not inflight:
                    break
                if time.monotonic() - started >= timeout:
                    self.counts[f"{operation}_drain_timeouts"] += 1
                    self.operation_receipts.append(
                        {
                            "operation": operation,
                            "status": "drain_timeout",
                            "drain_seconds": time.monotonic() - started,
                            "model_revision": self.model_revision,
                        }
                    )
                    raise TimeoutError(
                        f"{operation} could not drain inflight generation in time"
                    )
                time.sleep(0.01)
            adapter = self.adapter
            if adapter is None:
                raise RuntimeError("model adapter is not ready")
            try:
                result = callback(adapter)
            except BaseException:
                self.counts[f"{operation}_failed"] += 1
                self.operation_receipts.append(
                    {
                        "operation": operation,
                        "status": "failed",
                        "drain_seconds": time.monotonic() - started,
                        "model_revision": self.model_revision,
                    }
                )
                raise
            receipt = {
                "operation": operation,
                "status": "completed",
                "drain_seconds": time.monotonic() - started,
                "model_revision": getattr(self, "model_revision", 0),
            }
            self.operation_receipts.append(receipt)
            self.counts[f"{operation}_completed"] += 1
            return result

    def embed(
        self,
        inputs,
        *,
        dimensions=None,
        admission_class="embeddings",
        admitted=False,
    ):
        from .adapters.base import decoder_input_representations

        def execute(adapter):
            try:
                result = decoder_input_representations(
                    adapter, inputs, dimensions=dimensions
                )
            except NotImplementedError as error:
                from .api_resources import CapabilityUnavailable

                raise CapabilityUnavailable(str(error)) from error
            self.counts["embeddings_prompt_tokens"] += result[1]
            self.counts["embeddings_vectors"] += len(result[0])
            return result

        return self._exclusive_adapter_operation(
            "embeddings",
            execute,
            admission_class=admission_class,
            admitted=admitted,
        )

    def supports_multimodal(self):
        return callable(
            getattr(getattr(self, "adapter", None), "prepare_multimodal_request", None)
        )

    def supports_output_audio(self):
        from .contracts import Capability

        return (
            Capability.OUTPUT_AUDIO in self.route_capabilities
            and callable(
                getattr(getattr(self, "adapter", None), "synthesize_speech", None)
            )
        )

    def synthesize_speech(
        self, input_text, *, voice, instructions, response_format, speed
    ):
        from .adapters.base import AudioOutput
        from .api_resources import CapabilityUnavailable
        from .contracts import Capability

        def execute(adapter):
            synthesize = getattr(adapter, "synthesize_speech", None)
            if (
                Capability.OUTPUT_AUDIO not in self.route_capabilities
                or not callable(synthesize)
            ):
                raise CapabilityUnavailable(
                    "the selected route has no qualified output-audio implementation"
                )
            result = synthesize(
                input_text,
                voice=voice,
                instructions=instructions,
                response_format=response_format,
                speed=speed,
            )
            if not isinstance(result, AudioOutput):
                raise RuntimeError("audio adapter must return AudioOutput")
            self.counts["output_audio_requests"] += 1
            self.counts["output_audio_bytes"] += len(result.data)
            return result

        return self._exclusive_adapter_operation(
            "output_audio", execute, admission_class="audio"
        )

    def rerank(self, query, documents):
        def execute(adapter):
            from .adapters.base import decoder_input_representations

            try:
                vectors, prompt_tokens = decoder_input_representations(
                    adapter, [query, *documents]
                )
            except NotImplementedError as error:
                from .api_resources import CapabilityUnavailable

                raise CapabilityUnavailable(str(error)) from error
            query_vector = vectors[0]
            scores = [
                sum(left * right for left, right in zip(query_vector, vector))
                for vector in vectors[1:]
            ]
            self.counts["rerank_prompt_tokens"] += prompt_tokens
            self.counts["rerank_documents"] += len(documents)
            return scores

        return self._exclusive_adapter_operation(
            "rerank", execute, admission_class="rerank"
        )

    def _resolve_lora_path(self, path):
        if self.lora_root is None:
            from .api_resources import CapabilityUnavailable

            raise CapabilityUnavailable(
                "dynamic LoRA is disabled; configure --lora-dir"
            )
        candidate = Path(path).expanduser()
        candidate = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.lora_root / candidate).resolve()
        )
        if not candidate.is_relative_to(self.lora_root):
            raise ValueError("LoRA path must stay within the configured LoRA directory")
        return candidate

    def _invalidate_model_state(self):
        self.model_revision += 1
        self.host_prompt_cache.clear()
        invalidate_features = getattr(self.adapter, "invalidate_feature_cache", None)
        if callable(invalidate_features):
            invalidate_features()
        if self.apc is not None:
            self.apc.clear()
        try:
            import mlx.core as mx

            clear_compile_cache = getattr(mx, "clear_compile_cache", None)
            if callable(clear_compile_cache):
                clear_compile_cache()
            mx.clear_cache()
        except ImportError:
            pass

    def load_lora_adapter(self, name, path, *, base_model_name=None):
        candidate = self._resolve_lora_path(path)
        model_name = self.status().get("model")
        if base_model_name is not None and base_model_name not in {
            model_name,
            str(self.model_path),
            Path(self.model_path).name,
        }:
            raise ValueError("base_model_name does not match the loaded model")

        def install(adapter):
            if self.lora_session is not None:
                raise ValueError("unload the active LoRA adapter before loading another")
            from .runtime.lora import install_lora

            self.lora_session = install_lora(
                adapter.model, name=name, path=candidate
            )
            self._invalidate_model_state()
            return {
                "status": "loaded",
                "model_revision": self.model_revision,
                "base_model_name": model_name,
            }

        return self._exclusive_adapter_operation("lora_load", install)

    def unload_lora_adapter(self, name, *, lora_int_id=None):
        if lora_int_id is not None and (
            isinstance(lora_int_id, bool) or not isinstance(lora_int_id, int)
        ):
            raise ValueError("lora_int_id must be an integer")

        def remove(adapter):
            session = self.lora_session
            if session is None or session.name != name:
                raise ValueError("LoRA adapter is not loaded")
            session.restore(adapter.model)
            self.lora_session = None
            self._invalidate_model_state()
            return {"status": "unloaded", "model_revision": self.model_revision}

        return self._exclusive_adapter_operation("lora_unload", remove)

    def batching_status(self):
        with self.lock:
            memory = {
                key: self.snapshot.get(key)
                for key in (
                    "metal_active_bytes",
                    "metal_peak_bytes",
                    "process_physical_footprint_bytes",
                    "headroom_bytes",
                    "memory_waiting",
                )
            }
            return self.batch_metrics.snapshot(
                queue_depth=self.queued_jobs, memory=memory
            )

    def prometheus_metrics(self):
        """Return a non-destructive host-only Prometheus scrape."""

        from .prometheus import render_engine_metrics

        return render_engine_metrics(self)

    @property
    def apc_sessions_enabled(self):
        return bool(self.cache_dir or self.apc_persist_dir)

    def _session_scope(self, tenant_id):
        # Control ownership is always tenant-specific even when prompt state is
        # intentionally shared. X-Tenant-ID is an ownership partition, not auth.
        return str(tenant_id or "default")

    def _session_apc(self):
        apc = self.apc
        if apc is None and (self.stop_event.is_set() or self.error is not None):
            from .runtime.apc_v2 import APCSessionUnavailable

            raise APCSessionUnavailable("APCv2 session service is closed")
        if not self.apc_sessions_enabled or apc is None:
            raise LookupError("APCv2 session controls require a configured disk tier")
        return apc

    def apc_session_state(self, tenant_id, session_id):
        return self._session_apc().session_state(
            self._session_scope(tenant_id), session_id
        )

    def apc_sessions(self, tenant_id, *, limit=50, cursor=0):
        return self._session_apc().list_sessions(
            self._session_scope(tenant_id), limit=limit, cursor=cursor
        )

    def apc_session_park(self, tenant_id, session_id, *, ttl_seconds):
        return self._session_apc().park_session(
            self._session_scope(tenant_id), session_id, ttl_seconds=ttl_seconds
        )

    def apc_session_resume(
        self, tenant_id, session_id, *, ttl_seconds=None, admitted=False
    ):
        # Publish the worker-owned prefetch before quiesce can close admission,
        # so an accepted resume is visible to the drain's idle predicate.
        with self.submission_lock:
            self._ensure_admission("session_prefetch", admitted=admitted)
            return self._session_apc().resume_session(
                self._session_scope(tenant_id), session_id, ttl_seconds=ttl_seconds
            )

    def apc_session_delete(self, tenant_id, session_id):
        return self._session_apc().delete_session(
            self._session_scope(tenant_id), session_id
        )

    PARALLEL_SAMPLE_WAIT_SECONDS = 15.0

    def _resolve_commit_direction(self, adapter):
        """Bind steering to a direction calibrated for the loaded artifact, or turn it off.

        A direction measured on another artifact is never used.  When steering
        is wanted and none is bound, the server calibrates one itself; if that
        fails its gates steering stays off, and an operator who asked for it
        explicitly gets a startup error (fail closed) rather than a server that
        looks steered and is not.
        """
        from .thinking_calibration import resolve_direction, supports_calibration

        operator_asked = (self._thinking_overrides.get("thinking_steer_alpha") or 0) > 0
        wanted = self.thinking_steer_alpha > 0
        if not supports_calibration(adapter):
            self.thinking_steer_status = {"state": "unsupported"}
            if operator_asked:
                raise ValueError(
                    "--thinking-steer-alpha needs a model with residual taps and a single "
                    "thinking-close token; this model adapter has neither"
                )
            self.thinking_steer_alpha = 0.0
            return
        assets = getattr(adapter, "commit_direction_assets", None)
        shipped = assets() if callable(assets) else {}
        direction, status = resolve_direction(
            adapter,
            self.model_path,
            cache_dir=self.cache_dir,
            shipped=shipped.get("paths", ()),
            preferred_layer=shipped.get("layer"),
            allow_auto=wanted and self.thinking_auto_calibration,
        )
        self._commit_direction, self.thinking_steer_status = direction, status
        if direction is None and wanted:
            reason = (status.get("auto_calibration") or {}).get("failures") or [
                "automatic calibration is disabled" if not self.thinking_auto_calibration else "no calibration"
            ]
            if operator_asked:
                raise ValueError(
                    "steering was requested but no commit direction is calibrated for this "
                    "artifact: " + "; ".join(reason)
                )
            log.warning("thinking steering disabled for this artifact: %s", "; ".join(reason))
            self.thinking_steer_alpha = 0.0

    def admit_parallel_samples(self, count):
        """Guard opt-in fanout by lane count and measured physical headroom."""
        if type(count) is not int or count < 1:
            raise ValueError("parallel sample count must be positive")
        if count > self.max_lanes:
            raise Overloaded("parallel samples exceed configured lane capacity")
        from .memory import execution_headroom

        # Preserve the same 20 GiB hard reserve used by lane admission.  Each
        # sample is charged 2 GiB, the measured 1.76 GiB per-lane transient
        # rounded up (the earlier 4 GiB refused 70% of n=2 requests on
        # Flash-Next under a 20-lane load).  Headroom moves with every finished
        # lane, so wait briefly the way single requests queue on admission
        # instead of refusing on one unlucky reading.  This runs on the HTTP
        # thread at the request boundary, never on the token hot path.
        required = (20 + 2 * count) * (1 << 30)
        deadline = time.monotonic() + self.PARALLEL_SAMPLE_WAIT_SECONDS
        while execution_headroom() < required:
            if time.monotonic() >= deadline:
                self._clear_allocator_cache_before_reject()
                if execution_headroom() >= required:
                    break
                self.batch_metrics.rejected("parallel_sample_footprint", self.queued_jobs)
                raise Overloaded("parallel samples denied by physical footprint guard")
            time.sleep(0.25)
        return {
            "schema": "mlx2.parallel-sampling-admission.v1",
            "samples": count,
            "required_headroom_bytes": required,
            "guard": "physical_footprint",
        }

    def _emit(self, job, event):
        try:
            job.events.put_nowait(event)
        except queue.Full:
            job.cancelled.set()
            try:
                job.events.get_nowait()
                job.events.put_nowait(
                    {
                        "error": "client did not consume output fast enough",
                        "status": 429,
                    }
                )
            except (queue.Empty, queue.Full):
                pass

    def _finish(self, job, event):
        # A terminal event is an ownership boundary: once a client can observe
        # completion, the request must no longer pin APCv2 state or occupy an
        # inflight slot.  Publishing first exposed a small but real window in
        # which completed HTTP requests still reported an active COW lease.
        branch = job.cache_branch
        job.cache_branch = None
        job.admission_hit = job.admission_tokens = None
        if branch is not None and hasattr(branch, "close"):
            branch.close()
        waiting_siblings = ()
        with self.lock:
            fanout_role = getattr(job, "fanout_role", None)
            fanout_group = getattr(job, "fanout_group", None)
            fanout_waiting = getattr(self, "fanout_waiting", {})
            if fanout_role == "prefill_leader" and fanout_group:
                waiting_siblings = fanout_waiting.pop(fanout_group, ())
            elif fanout_role == "apcv2_sibling" and fanout_group:
                waiting = fanout_waiting.get(fanout_group)
                if waiting is not None:
                    remaining = tuple(item for item in waiting if item.id != job.id)
                    if remaining:
                        fanout_waiting[fanout_group] = remaining
                    else:
                        fanout_waiting.pop(fanout_group, None)
            if self.jobs.pop(job.id, None) is not None:
                if (
                    getattr(self, "_service_state", "serving") == "draining"
                    and event.get("error") != "drain timeout"
                ):
                    self.counts["jobs_drained"] += 1
                self.slots.release()
                status = (
                    "completed"
                    if "finish_reason" in event
                    else "cancelled"
                    if event.get("error") == "cancelled"
                    else "failed"
                )
                metrics = getattr(self, "batch_metrics", None)
                if metrics is not None:
                    finishing = getattr(metrics, "finishing", None)
                    if callable(finishing):
                        finishing(job.id, event.get("finish_reason"))
                    metrics.terminal(job.id, status)
        self._emit(job, event)
        if waiting_siblings:
            prepared = self.fanout_capsules.pop(fanout_group, None)
            if prepared is not None:
                prepared.close()
            sibling_event = {
                "error": "parallel prefill leader ended before APCv2 fanout",
                "status": event.get("status", 503),
            }
            for sibling in waiting_siblings:
                self._finish(sibling, sibling_event)

    def _observe_thinking_budget(self, job, *, final=False):
        """Commit one history-derived budget outcome from emitted lane tokens."""
        processor = job.thinking_budget
        if processor is None or job.thinking_budget_resolved:
            return
        decision_length = processor.budget + len(processor.close_token_ids)
        if not final and len(job.thinking_tokens) < decision_length:
            return
        job.thinking_budget_fired = processor.fired_for_generated(
            job.thinking_tokens
        )
        job.thinking_budget_resolved = True
        if job.thinking_budget_fired and not job.thinking_budget_counted:
            self.counts["thinking_budget_forced_closes"] += 1
            job.thinking_budget_counted = True

    def _cancel_pending_cache_capsule(self, batch, job, reason):
        """Decline a prepared fanout before every sibling is attached."""
        prepared = getattr(job, "cache_capsule", None)
        group = getattr(job, "fanout_group", None)
        if prepared is None or not group:
            return False
        batch.cancel_cache_capsule_group(group, reason)
        retained = self.fanout_capsules.pop(group, None)
        (retained or prepared).close()
        # Remaining queued siblings must take the ordinary APCv2 path.  The
        # prepared owner is all-or-nothing and cannot be partially published.
        with self.lock:
            for sibling in self.jobs.values():
                if sibling.fanout_group == group:
                    sibling.cache_capsule = None
                    sibling.cache_capsule_rows = 0
        return True

    def _bind_pending_cache_capsule(self, batch, job):
        """Attach one inserted lane, rolling it back on bind failure."""
        if job.cache_capsule is None:
            return False
        try:
            bound = batch.bind_cache_capsule(
                job.fanout_group,
                job.uid,
                job.cache_capsule,
                job.cache_capsule_rows,
            )
        except BaseException:
            batch.remove([job.uid])
            self._cancel_pending_cache_capsule(batch, job, "attachment_failed")
            raise
        if bound:
            # Ownership has moved to BatchGenerator and is held through the
            # last terminal lane.
            self.fanout_capsules.pop(job.fanout_group, None)
        return bound

    def _fail_deferred_admission_timeout(self, batch, job):
        self._cancel_pending_cache_capsule(
            batch, job, "memory_admission_timeout"
        )
        self._finish(job, {
            "error": "host memory admission did not recover before deadline",
            "status": 429,
        })
        self.counts["memory_admission_timeouts"] += 1

    def _cache_capsule_fanout_fits(self, active, siblings):
        """Require one scheduler boundary to attach the complete capsule."""
        if len(active) + len(siblings) <= self.max_lanes:
            return True
        self.counts["cache_capsule_width_fallbacks"] += 1
        receipt = {
            "schema": "mlx2.cache-capsule.v1",
            "status": "fallback",
            "reason": "scheduler_width_unavailable",
            "rows": len(siblings),
        }
        for sibling in siblings:
            sibling.cache_capsule_receipt = dict(receipt)
        return False

    def _publish_checkpoint(
        self, apc, key, tokens, prompt_cache, *, approximate=False, **kwargs
    ):
        """Store a committed checkpoint; a publish failure costs the checkpoint.

        ``apc.store`` raises when a cache cannot be frozen (a speculating or
        rollback-staged plane, an uncopyable object).  That is a lost reuse
        opportunity for later requests, not a reason to take down the
        generation worker and every inflight request with it.

        This is the only path into APCv2 (idle/disk spills, persistent blocks
        and cache capsules all derive from stored entries).  The exact prefix
        cache never receives approximate state: a lane flagged approximate is
        skipped, and so is any cache that structurally carries quantized
        planes even if its owner was not flagged.
        """
        from .runtime.approximate_kv import prompt_cache_is_approximate

        if approximate or prompt_cache_is_approximate(prompt_cache):
            self.counts["apcv2_store_skipped_approximate"] += 1
            return False
        try:
            capabilities = apc.store(key, tokens, prompt_cache, **kwargs)
            return getattr(capabilities, "stored", None) is not False
        except Exception:  # noqa: BLE001 - the worker must outlive one bad checkpoint
            self.counts["apcv2_store_failures"] += 1
            log.exception("APCv2 checkpoint publication failed")
            return False

    def _fail_attaching_cohort(self, batch, active, published, cohort, event):
        """Roll back a declared cohort before any member can decode.

        Members already inserted into ``BatchGenerator`` are removed together;
        members still in the local publication deque stop counting as queued.
        Every reserved request receives the same terminal failure and releases
        its slot.  The caller invokes this only while the cohort owns the idle
        attachment boundary, before ``batch.next``.
        """
        member_ids = {job.id for job in cohort.jobs}
        attached_uids = [
            uid for uid, job in active.items() if job.id in member_ids
        ]
        if attached_uids:
            batch.remove(attached_uids)
            for uid in attached_uids:
                active.pop(uid, None)
        remaining = [job for job in published if job.id in member_ids]
        if remaining:
            published_copy = [job for job in published if job.id not in member_ids]
            published.clear()
            published.extend(published_copy)
            with self.lock:
                self.queued_jobs -= len(remaining)
        self.counts["batch_cohort_attachment_failures"] += 1
        self.counts["batch_cohort_jobs_failed_closed"] += len(cohort.jobs)
        for member in cohort.jobs:
            self._finish(member, dict(event))

    def _fail_drain_timeout(
        self, batch, active, deferred, published, held_cohort, attaching_cohort
    ):
        """Fail every still-owned request from the model worker thread."""
        manager = getattr(self, "api_resources", {}).get("batches")
        abort_batches = getattr(manager, "abort_for_drain_timeout", None)
        if callable(abort_batches):
            abort_batches()
        self._cancel_pending_prefetches()
        for job in tuple(self.jobs.values()):
            job.cancelled.set()
        if active:
            try:
                batch.remove(tuple(active))
            except Exception:  # noqa: BLE001 - finish every request even if cleanup degrades
                log.exception("batch cleanup failed at drain timeout")
            active.clear()
        deferred.clear()
        published.clear()
        while True:
            try:
                self.incoming.get_nowait()
            except queue.Empty:
                break
        for cohort in (held_cohort, attaching_cohort):
            if cohort is not None:
                for member in cohort.jobs:
                    member.cancelled.set()
        with self.submission_lock:
            self.pending_cohorts.clear()
        for prepared in tuple(self.fanout_capsules.values()):
            try:
                prepared.close()
            except Exception:  # noqa: BLE001 - a capsule cannot abort cleanup
                log.exception("cache capsule release failed at drain timeout")
        self.fanout_capsules.clear()
        self.fanout_waiting.clear()
        with self.lock:
            remaining = list(self.jobs.values())
            self.queued_jobs = 0
        for job in remaining:
            self._finish(job, {"error": "drain timeout", "status": 503})

    def _service_admin_prefetch(self, apc):
        with self.submission_lock:
            with self.lock:
                if (
                    self._service_state not in {"serving", "draining"}
                    or self._quiesce_worker_owned
                    or not self._admin_prefetch_queue
                ):
                    return False
                tenant, session_id = self._admin_prefetch_queue.popleft()
            try:
                apc.resume_session(
                    self._session_scope(tenant),
                    session_id,
                    ttl_seconds=None,
                )
            except Exception:  # noqa: BLE001 - one requested prefetch is fail-soft
                self.counts["prefetch_failures"] += 1
                log.exception("admin APCv2 session prefetch failed")
                return False
        self.counts["prefetches_started"] += 1
        return True

    def _service_pending_prefetch(self, apc):
        """Restore only while admission is stably open; otherwise cancel it."""
        with self.submission_lock:
            with self.lock:
                may_restore = self._service_state in {"serving", "draining"} and not self._quiesce_worker_owned
            if not may_restore:
                return bool(self._cancel_pending_prefetches(apc))
            service = getattr(apc, "service_pending_prefetch", None)
            return bool(callable(service) and service())

    def _run(self):
        adapter = batch = apc = capsule_pool = None
        active = {}
        deferred = deque()
        published = deque()
        held_cohort = None
        attaching_cohort = None
        try:
            adapter_execution_policy = (
                {
                    key: value
                    for key, value in self.execution_policy.items()
                    if key not in {
                        "prompt_lookup",
                        "fly_verification",
                        "apc_interior_checkpoints",
                        "constrained_tool_grammar",
                        "tolerant_tool_markers",
                    }
                }
                if self.execution_policy is not None
                else None
            )
            if not adapter_execution_policy:
                adapter_execution_policy = None
            adapter = (
                self.adapter_factory(
                    self.model_path, execution_policy=adapter_execution_policy
                )
                if adapter_execution_policy is not None
                else self.adapter_factory(self.model_path)
            )
            self.adapter = adapter
            stop_token_ids = generation_stop_token_ids(adapter)
            # A model adapter may ship its own run-on reasoning defaults (North
            # does: its calibrated guard and steering).  Operator flags win,
            # including an explicit 0 to turn a lever off.
            accessor = getattr(adapter, "thinking_guard_defaults", None)
            adapter_defaults = dict(accessor() or {}) if callable(accessor) else {}
            resolved = {
                name: (override if override is not None else adapter_defaults.get(name, 0))
                for name, override in self._thinking_overrides.items()
            }
            self.thinking_budget = int(resolved["thinking_budget"] or 0)
            self.thinking_steer_alpha = float(resolved["thinking_steer_alpha"] or 0.0)
            self.thinking_steer_hammer = float(resolved["thinking_steer_hammer"] or 0.0)
            self.thinking_defaults_source = (
                "adapter"
                if adapter_defaults and all(v is None for v in self._thinking_overrides.values())
                else "operator" if any(v is not None for v in self._thinking_overrides.values())
                else "none"
            )
            self._resolve_commit_direction(adapter)
            import mlx.core as mx
            from .memory import execution_headroom
            from .runtime.apc_v2 import (
                APCLookup,
                APCv2,
                MTPAPCSidecar,
                inspect_apc_capabilities,
            )
            from .runtime.generate import BatchGenerator, interior_checkpoint_positions
            from .runtime.os_memory import physical_footprint_bytes
            from .runtime.sample_utils import (
                LaneRNG,
                make_transformed_logprobs,
                make_logits_processors,
                draw_key,
            )
            from .runtime.segmented_self_mtp import segmented_self_mtp_stats
            from .runtime.memory_policy import (
                SelfMTPLaneAdmissionController,
                _make_self_mtp_admission_callback,
            )

            identity = runtime_identity()
            self.max_context = min(self.max_context, adapter.max_context)
            settings = {
                "max_context": self.max_context,
                "default_max_tokens": self.default_max_tokens,
                "max_lanes": self.max_lanes,
                "max_inflight": self.max_inflight,
                "prefill_step": self.prefill_step,
                "cache_bytes": self.cache_bytes,
                "disk_cache": bool(self.cache_dir or self.apc_persist_dir),
                "apc_persistence": bool(self.apc_persist_dir),
                "apc_persist_on_shutdown": self.apc_persist_on_shutdown,
                "apc_persist_shutdown_seconds": self.apc_persist_shutdown_seconds,
                "apc_session_max_ttl_seconds": self.apc_session_max_ttl_seconds,
                "apc_session_pinned_disk_bytes": self.apc_session_pinned_disk_bytes,
                "apc_session_pinned_disk_bytes_global": (
                    self.apc_session_pinned_disk_bytes_global
                ),
                "apc_session_pinned_resident_bytes": self.apc_session_pinned_resident_bytes,
                "apc_session_prefetch_ttl_seconds": self.apc_session_prefetch_ttl_seconds,
                "apc_quarantine_max_entries": self.apc_quarantine_max_entries,
                "apc_quarantine_max_bytes": self.apc_quarantine_max_bytes,
                "host_prompt_cache_entries": self.host_prompt_cache.max_entries,
                "host_prompt_cache_tokens": self.host_prompt_cache.max_tokens,
                "coalesce_window_ms": self.coalesce_window_seconds * 1000,
                "batch_cohort_timeout_ms": self.batch_cohort_timeout_seconds * 1000,
                "mtp": self.mtp,
                "tenant_scoped_cache": self.tenant_scoped_cache,
                "environment": adapter.environment,
                "adaptive_mtp_depth": self.adaptive_mtp_policy.as_dict(),
                "fly_verification": self.fly_verification_policy.as_dict(),
                "spomin_live_surgery": asdict(self.spomin_policy),
                "thinking_budget": self.thinking_budget,
                "thinking_steer": {
                    "alpha": self.thinking_steer_alpha,
                    "hammer": self.thinking_steer_hammer,
                    "calibration": self.thinking_steer_status,
                },
                "thinking_defaults_source": self.thinking_defaults_source,
                "constrained_tool_grammar": self.constrained_tool_grammar,
                "tolerant_tool_markers": self.tolerant_tool_markers,
                "cache_capsules": dict(self.cache_capsule_policy),
                "persistent_block_bytes": self.persistent_block_bytes,
                "apc_interior_checkpoints": dict(
                    self.apc_interior_checkpoint_policy
                ),
            }
            # Bind the policy and the adapter-declared descriptor into the
            # settings a qualification receipt must match.  An adapter that
            # does not declare the selected operation fails closed here.
            approximate_operations = {}
            approximate_operation = None
            if self.approximate_kv_policy.enabled:
                from .runtime.approximate_kv import declared_operations

                approximate_operations = declared_operations(
                    adapter, adapter_fingerprint=adapter.identity["fingerprint"]
                )
                approximate_operation = approximate_operations.get(
                    self.approximate_kv_policy.operation
                )
                if approximate_operation is None:
                    raise ValueError(
                        "model adapter does not declare approximate KV operation "
                        f"{self.approximate_kv_policy.operation!r}"
                    )
            settings["approximate_kv"] = dict(
                self.approximate_kv_policy.as_dict(),
                descriptor=(
                    approximate_operation.descriptor.as_dict()
                    if approximate_operation is not None
                    else None
                ),
            )
            config = adapter.execution_config(
                max_lanes=self.max_lanes, prefill_step=self.prefill_step
            )
            settings["execution_policy"] = dict(config)
            external_draft = config.get("backend") == "external_draft"
            if self.spomin_policy.enabled and external_draft:
                raise ValueError(
                    "live Spomin surgery is incompatible with external draft"
                )
            if self.cache_capsule_policy["enabled"] and external_draft:
                raise ValueError("cache capsules are incompatible with external draft")
            if self.approximate_kv_policy.enabled and external_draft:
                raise ValueError("approximate KV is incompatible with external draft")
            prompt_lookup = self.prompt_lookup
            self.apc_interior_route_supported = not (
                external_draft or prompt_lookup or self.approximate_kv_policy.enabled
            )
            if (
                self.apc_interior_route_supported
                and self.apc_interior_checkpoint_policy["count"]
            ):
                from .runtime.models.cache import make_prompt_cache

                probe_cache = make_prompt_cache(adapter.model)
                self.apc_interior_route_supported = inspect_apc_capabilities(
                    probe_cache
                ).interior_checkpoint_target
                probe_cache = None
            if (
                self.apc_interior_checkpoint_policy["count"]
                and not self.apc_interior_route_supported
            ):
                route = (
                    "external draft"
                    if external_draft
                    else "prompt lookup"
                    if prompt_lookup
                    else "approximate KV"
                    if self.approximate_kv_policy.enabled
                    else "adapter cache"
                )
                raise ValueError(
                    "APCv2 interior checkpoints cannot capture on the selected "
                    f"{route} route"
                )
            if self.fly_verification_policy.enabled and (
                prompt_lookup or not (self.mtp or external_draft)
            ):
                raise ValueError(
                    "FLy verification requires native self-MTP or external draft"
                )
            prompt_lookup_policy = {}
            if "prompt_lookup" in (self.execution_policy or {}):
                # This block is carved out of the adapter's unknown-key check
                # above, so it needs its own: a policy that configures a route
                # the server did not select, or names a knob nothing reads,
                # must fail rather than be silently accepted.  Presence, not
                # truthiness: an explicit null is still a misplaced key.
                from .runtime.pld import PromptLookupBatchGenerator

                if not prompt_lookup:
                    raise ValueError(
                        "execution policy configures prompt_lookup but the "
                        "prompt-lookup route is not selected"
                    )
                prompt_lookup_policy = PromptLookupBatchGenerator.validate_policy(
                    self.execution_policy["prompt_lookup"]
                )
            settings["prompt_lookup"] = dict(prompt_lookup_policy)
            settings["decode_time_fairness"] = decode_time_fairness_policy(
                external_draft=external_draft,
                prompt_lookup=prompt_lookup,
            )
            settings["speculation"] = (
                "external_draft"
                if external_draft
                else "prompt_lookup"
                if prompt_lookup
                else "self_mtp"
                if self.mtp
                else "ordinary"
            )
            if self.int8_prefill_policy.enabled:
                # Fails closed (unsupported device, undeclared scope, or a
                # decode/verify block that could reach the row threshold)
                # before the route is qualified or published.  Recorded in
                # settings only when enabled so default-off records match.
                from .runtime.int8_prefill import bind_for_serving

                self.int8_prefill_handle, settings["int8_prefill"] = bind_for_serving(
                    adapter,
                    self.int8_prefill_policy,
                    max_lanes=self.max_lanes,
                    config=config,
                    speculation=settings["speculation"],
                    prompt_lookup_policy=prompt_lookup_policy,
                )
            def selected_profile_name():
                return (
                    adapter.profile_name(self.mtp)
                    + ("-pld" if prompt_lookup else "")
                    + (
                        f"-int8-prefill-{self.int8_prefill_policy.scope}"
                        if self.int8_prefill_policy.enabled
                        else ""
                    )
                    + (
                        "-adaptive-mtp-depth"
                        if self.adaptive_mtp_policy.enabled
                        else ""
                    )
                )

            profile_name = None
            if (external_draft or prompt_lookup) and self.mtp:
                raise ValueError("speculative routes are mutually exclusive")
            if external_draft and prompt_lookup:
                raise ValueError(
                    "external draft and prompt lookup are mutually exclusive"
                )
            cache_budget = adapter.cache_budget(mtp=self.mtp) if hasattr(adapter, "cache_budget") else None
            settings["cache_budget"] = cache_budget.as_dict() if cache_budget else None
            route_receipt = "candidate_validation"
            descriptor = getattr(adapter, "descriptor", None)
            implemented_capabilities = frozenset(
                descriptor.capabilities if descriptor is not None else frozenset()
            )
            route_capabilities = implemented_capabilities
            if prompt_lookup:
                from .contracts import Capability

                if Capability.PROMPT_LOOKUP not in route_capabilities:
                    raise ValueError(
                        "model adapter does not declare prompt-lookup execution"
                    )
            if self.qualification_mode:
                # Candidate mode advertises implemented capabilities, but a
                # speculation route the server did not select is not on this
                # route.  Mirror the qualified derivation so ``--ordinary``
                # candidates do not list ``mtp``/``segmented_mtp``.
                from .contracts import Capability

                unselected = set()
                if not self.mtp:
                    unselected |= {Capability.MTP, Capability.SEGMENTED_MTP}
                if not external_draft:
                    unselected.add(Capability.EXTERNAL_DRAFT)
                if not prompt_lookup:
                    unselected.add(Capability.PROMPT_LOOKUP)
                route_capabilities = frozenset(route_capabilities) - unselected
            else:
                if not self.qualification:
                    raise ValueError(
                        "A matching qualification receipt is required; use --qualification-mode for validation"
                    )
                from .qualification import load_qualified_route

                route = load_qualified_route(
                    self.qualification,
                    runtime=identity,
                    artifact=adapter.identity["fingerprint"],
                    settings=settings,
                    descriptor=adapter.descriptor,
                    name=selected_profile_name(),
                )
                profile_name = route.profile.name
                route_receipt = route.receipt
                route_capabilities = route.profile.capabilities
                approximate_evidence = tuple(route.profile.evidence)
            approximate_controller = None
            approximate_status = {
                "enabled": False,
                "state": "implemented",
                "fidelity": "exact",
            }
            if approximate_operation is not None:
                from .runtime.approximate_kv import LaneKVState, stage_lane_state
                from .runtime.approximate_state import (
                    ApproximateKVController,
                    ApproximateKVPolicy,
                )
                from .runtime.models.cache import make_prompt_cache

                # Outside qualification mode the matching record (settings and
                # ``feature_approximate_kv`` both verified above) is the
                # evidence; in candidate mode the policy stays unqualified and
                # every receipt says so.
                approximate_qualified = not self.qualification_mode
                approximate_controller = ApproximateKVController(
                    ApproximateKVPolicy(
                        self.approximate_kv_policy.operation,
                        enabled=True,
                        qualified=approximate_qualified,
                        evidence=(
                            approximate_evidence + self.approximate_kv_policy.evidence
                            if approximate_qualified
                            else self.approximate_kv_policy.evidence
                        ),
                    )
                )

                def apply_approximate_kv(request_id, planes):
                    return approximate_controller.apply(
                        request_id=request_id,
                        state_revision=approximate_operation.revision,
                        state=LaneKVState(
                            approximate_operation.revision, tuple(planes)
                        ),
                        adapters=approximate_operations,
                        candidate=self.qualification_mode,
                        stage=stage_lane_state,
                    )

                # Prove at construction that every attention plane this model
                # allocates accepts the operation and stays batchable.
                apply_approximate_kv(
                    "approximate-kv-construction-probe",
                    make_prompt_cache(adapter.model),
                )
                approximate_status = {
                    "enabled": True,
                    "state": "qualified" if approximate_qualified else "candidate",
                    "fidelity": "approximate",
                    "operation": approximate_operation.name,
                    "descriptor": approximate_operation.descriptor.as_dict(),
                    "revision": approximate_operation.revision,
                    "start_tokens": self.approximate_kv_policy.start_tokens,
                }
            # APCv2 must be able to retain at least one committed prompt
            # boundary per execution lane.  Otherwise a configured B20 server
            # with the historical 16-entry floor deterministically evicts four
            # warm prompts while the cohort is being primed, so those lanes
            # cannot compose batching with APCv2 reuse.
            from .runtime.int8_prefill import apc_semantic_fingerprint

            persistent_identity = APCv2.key(
                adapter.identity["fingerprint"],
                revision=persistent_runtime_revision(identity),
                adapter=adapter.identity["fingerprint"],
                tokenizer_fingerprint=adapter.identity["fingerprint"],
                cache_layout_fingerprint=adapter.layout,
                semantic_fingerprint=apc_semantic_fingerprint(
                    cache_semantic_fingerprint(
                        "__tenant_template__" if self.tenant_scoped_cache else None
                    ),
                    self.int8_prefill_policy,
                ),
            )
            disk_dir = self.apc_persist_dir or self.cache_dir
            apc = APCv2(
                layout_name=adapter.layout,
                max_size=max(16, self.max_lanes),
                max_bytes=self.cache_bytes,
                max_tokens=self.max_context,
                idle_disk_seconds=180 if disk_dir else 0,
                idle_disk_dir=disk_dir,
                idle_disk_max_bytes=64 << 30,
                persistent_block_bytes=self.persistent_block_bytes,
                persist_dir=self.apc_persist_dir,
                persist_identity=(persistent_identity if self.apc_persist_dir else None),
                persist_semantic_namespace=(
                    "tenant" if self.tenant_scoped_cache else "shared"
                ),
                persist_corruption_action=self.apc_persist_corruption,
                session_max_ttl_seconds=self.apc_session_max_ttl_seconds,
                pinned_disk_bytes_per_tenant=self.apc_session_pinned_disk_bytes,
                pinned_disk_bytes_global=self.apc_session_pinned_disk_bytes_global,
                pinned_resident_bytes_per_tenant=self.apc_session_pinned_resident_bytes,
                prefetch_ttl_seconds=self.apc_session_prefetch_ttl_seconds,
                quarantine_max_entries=self.apc_quarantine_max_entries,
                quarantine_max_bytes=self.apc_quarantine_max_bytes,
            )
            self.apc = apc
            cache_keys = {}

            def cache_key_for(tenant_id, media_fingerprint=None):
                """APCv2 namespace for a request: shared, or per tenant."""
                scope = tenant_id if self.tenant_scoped_cache else None
                identity_scope = (scope, media_fingerprint)
                cached = cache_keys.get(identity_scope)
                if cached is None:
                    semantic = cache_semantic_fingerprint(scope)
                    if media_fingerprint:
                        semantic = f"{semantic}:media:{media_fingerprint}"
                    cached = cache_keys[identity_scope] = apc.key(
                        adapter.identity["fingerprint"],
                        revision=persistent_runtime_revision(identity),
                        adapter=adapter.identity["fingerprint"],
                        tokenizer_fingerprint=adapter.identity["fingerprint"],
                        cache_layout_fingerprint=adapter.layout,
                        # Int8-prefill state lives in its own namespace
                        # (memory and disk); identity when disabled.
                        semantic_fingerprint=apc_semantic_fingerprint(
                            semantic,
                            self.int8_prefill_policy,
                        ),
                    )
                return cached

            def session_tag_for(job):
                if job is None or not job.request.get("session_id"):
                    return None
                return (
                    str(job.tenant_id or "default"),
                    job.request["session_id"],
                )

            key = cache_key_for(None)
            if self.cache_capsule_policy["enabled"]:
                capsule_pool = apc.new_cache_capsule_pool(
                    enabled=True,
                    verify_raw_bits=self.cache_capsule_policy["verify_raw_bits"],
                )
            controller = SelfMTPLaneAdmissionController(
                saturation_lane_cap=self.max_lanes, verification_row_cap=self.max_lanes * (config["num_draft"] + 1),
                cache_estimator=cache_budget.project if cache_budget else None,
                transient_gib_per_lane=getattr(
                    cache_budget,
                    "transient_gib_per_lane",
                    SelfMTPLaneAdmissionController.K2_TRANSIENT_GIB_PER_LANE,
                ),
            )
            admission = {}
            spomin_manager = None
            post_prefill_transform = None
            if self.spomin_policy.enabled:
                from .runtime.spomin_live_surgery import SpominLiveSurgeryManager

                # The surgical backend is adapter-owned tensor math. An adapter
                # without one yields a declined receipt, never a silent edit.
                backend_factory = getattr(adapter, "spomin_backend", None)
                spomin_manager = SpominLiveSurgeryManager(
                    enabled=True,
                    backend_factory=backend_factory
                    if callable(backend_factory)
                    else (lambda model, prompt_cache: None),
                )
                self.spomin_manager = spomin_manager

                def post_prefill_transform(
                    *, uid, model, prompt_cache, cached_token_ids
                ):
                    job = active.get(uid)
                    if job is None:
                        # Cancelled while prefilling: nothing to edit, and an
                        # exception here would take the worker down.
                        return None
                    # A warm APCv2 prefix does not make this state shared: the
                    # boundary hands over an extracted request-private cache and
                    # the backend only ever builds new arrays, so the frozen
                    # entry the prefix came from is never written.
                    transcript = self.spomin_policy.transcript(
                        cached_token_ids,
                        tokenizer_identity=adapter.identity["fingerprint"],
                        revision=f"request:{job.id}:prefill",
                    )
                    transaction = spomin_manager.prepare(
                        request_id=job.id,
                        prompt_token_ids=cached_token_ids,
                        transcript=transcript,
                        capacity_tokens=self.spomin_policy.capacity_tokens,
                        strategy=self.spomin_policy.strategy,
                        has_mtp_state=False,
                        has_recurrent_state=any(
                            getattr(layer, "is_linear", False)
                            for layer in getattr(
                                getattr(
                                    getattr(
                                        adapter.model, "language_model", adapter.model
                                    ),
                                    "model",
                                    adapter.model,
                                ),
                                "layers",
                                (),
                            )
                        ),
                        cache_is_request_private=True,
                        protected_segment_ids=tuple(
                            segment.segment_id
                            for segment in transcript.segments[
                                : self.spomin_policy.protect_prefix_segments
                            ]
                        ),
                    )
                    if transaction is None:
                        receipt = spomin_manager.snapshot()["recent"][-1]
                        job.spomin_receipt = receipt
                        return {"receipt": receipt}
                    # The callback runs after isolated B=1 prefill and before
                    # the lane is published to decode.  Drain the exact stream
                    # before mutating cache arrays, then publish only the
                    # transaction's revision-checked retained history.
                    mx.synchronize(batch.stream)
                    exact_boundary = None
                    try:
                        from .runtime.cow_cache import snapshot_prompt_cache_descriptors

                        exact_cache, _, snapshot = snapshot_prompt_cache_descriptors(
                            prompt_cache
                        )
                        exact_boundary = {
                            "tokens": list(cached_token_ids),
                            "target_cache": exact_cache,
                            "committed_only": True,
                            "pre_transform_exact": True,
                            "snapshot": snapshot,
                        }
                    except Exception:  # noqa: BLE001 - reuse optimization only
                        self.counts["spomin_exact_boundary_snapshot_failures"] += 1
                        log.exception("Could not preserve exact pre-Spomin boundary")
                    try:
                        receipt = transaction.apply(
                            model,
                            prompt_cache,
                            request_quiescent=True,
                            device_work_drained=True,
                        )
                    except Exception as error:  # noqa: BLE001 - one lane, not the worker
                        # Backends stage and evaluate every array before they
                        # touch a cache, so an unexpected failure leaves the
                        # exact prefill intact; decode it unedited.
                        log.exception("Spomin surgery failed; serving the exact prefill")
                        receipt = spomin_manager.decline(
                            job.id, "backend_error", detail=str(error)
                        )
                    job.spomin_receipt = dict(receipt)
                    if receipt.get("status") != "applied":
                        return {"receipt": receipt}
                    return {
                        "receipt": receipt,
                        "retained_token_ids": transaction.retained_token_ids,
                        "prompt_cache": prompt_cache,
                        "exact_prompt_boundary": exact_boundary,
                    }

            def reclaim_allocator():
                # Pressure-only synchronization: retired arrays can remain in
                # flight after Python releases them. Complete work before
                # clearing allocator pages and measuring admission headroom.
                return self._clear_allocator_cache_before_reject(synchronize=True)

            def observe_admission(decision):
                admission.update(asdict(decision))

            def evict_unused_checkpoint():
                evicted = apc.evict_oldest_unleased()
                if evicted:
                    self.counts["memory_pressure_evictions"] += 1
                    self._permit_allocator_reclaim_after_eviction()
                return evicted

            def route_usable_hit(hit, tokens):
                """Reject a warm hit this speculation route cannot consume.

                An ordinary-route checkpoint (for example a lane that finished
                after demotion to plain decode) carries no draft state.  The
                self-MTP lane cannot fabricate it, so for this route it is a
                miss.  Release the lease here, before admission accounting,
                so pressure eviction can reclaim the unusable checkpoint
                instead of deferring the request while it stays pinned.
                """
                if (
                    self.mtp
                    and not external_draft
                    and not prompt_lookup
                    and hit.cache is not None
                    and hit.cached_tokens
                    and hit.sidecar is None
                ):
                    branch = hit.cache
                    if hasattr(branch, "close"):
                        branch.close()
                    self.counts["mtp_sidecar_missing_misses"] += 1
                    return APCLookup(
                        None, list(tokens), 0, False, None, "mtp_sidecar_missing"
                    )
                return hit

            if external_draft:
                batch = adapter.create_external_batch(
                    completion_batch_size=self.max_lanes,
                    prefill_step_size=self.prefill_step,
                    memory_headroom=lambda: max(0, execution_headroom() - controller.hard_reserve_gib * (1 << 30)),
                    reclaim_memory=reclaim_allocator,
                    evict_checkpoint=evict_unused_checkpoint,
                    stop_tokens=[[token] for token in stop_token_ids],
                    fly_verification=self.fly_verification_policy,
                )
            elif prompt_lookup:
                from .runtime.pld import PromptLookupBatchGenerator

                batch = PromptLookupBatchGenerator(
                    adapter.model,
                    completion_batch_size=self.max_lanes,
                    prefill_step_size=self.prefill_step,
                    prompt_lookup=prompt_lookup_policy,
                    stop_tokens=[[token] for token in stop_token_ids],
                )
            else:
                batch = BatchGenerator(
                    adapter.model,
                    completion_batch_size=self.max_lanes,
                    # Surgery edits one request-private cache at an isolated
                    # boundary; decode lanes still batch normally afterwards.
                    prefill_batch_size=(
                        1 if self.spomin_policy.enabled else min(2, self.max_lanes)
                    ),
                    prefill_step_size=self.prefill_step,
                    prefill_batch_window=1,
                    adaptive_prefill=True,
                    decode_time_fairness=settings["decode_time_fairness"],
                    adaptive_mtp_depth=(
                        self.adaptive_mtp_policy.controller_kwargs()
                        if self.adaptive_mtp_policy.enabled
                        else None
                    ),
                    fly_verification=self.fly_verification_policy,
                    apc_interior_checkpoints=(
                        self.apc_interior_checkpoint_policy
                        if self.apc_interior_route_supported
                        else {"count": 0, "min_stride": 1}
                    ),
                    self_mtp=config if self.mtp else None,
                    mtp_admission=_make_self_mtp_admission_callback(
                        controller,
                        free_memory=lambda: execution_headroom() / (1 << 30),
                        observer=observe_admission,
                        reclaim_memory=reclaim_allocator,
                        evict_unused_cache=evict_unused_checkpoint,
                        max_draft=config["num_draft"],
                    )
                    if self.mtp
                    else None,
                    stop_tokens=[[token] for token in stop_token_ids],
                    post_prefill_transform=post_prefill_transform,
                )
            if profile_name is None:
                profile_name = selected_profile_name()
            with self.lock:
                self.route_capabilities = frozenset(route_capabilities)
                self.sampling_vendor = vendor_sampling(adapter)
                self.snapshot = {
                    "state": "ready",
                    "model": Path(self.model_path).name,
                    "runtime": identity,
                    "artifact": adapter.identity["fingerprint"],
                    "profile": profile_name,
                    "settings": settings,
                    "route_receipt": route_receipt,
                    "capabilities": sorted(
                        capability.value for capability in self.route_capabilities
                    ),
                    # Keep the historical ``capabilities`` field as the
                    # selected-route view, but publish each lifecycle set
                    # explicitly so telemetry never infers implementation or
                    # qualification from selection.
                    "implemented_capabilities": sorted(
                        capability.value for capability in implemented_capabilities
                    ),
                    "selected_capabilities": sorted(
                        capability.value for capability in self.route_capabilities
                    ),
                    "qualified_capabilities": sorted(
                        capability.value for capability in self.route_capabilities
                    )
                    if not self.qualification_mode
                    else [],
                    "qualification": "candidate"
                    if self.qualification_mode
                    else "qualified",
                    "max_context": self.max_context,
                    "max_lanes": self.max_lanes,
                    # Whether a request that says nothing about reasoning opens
                    # a think channel.  Harnesses size their token budgets from
                    # it: a thinking model spends tokens before it answers.
                    "thinking_default": thinking_enabled(adapter, {"messages": []}),
                    # Reasoning tokens a harness should allow on top of an
                    # answer budget; adapters of long reasoners declare more.
                    "thinking_allowance_tokens": getattr(adapter, "thinking_allowance_tokens", None),
                    "structured_output": {
                        "engines": ["automaton", "scanner"],
                        "thinking_deferral": thinking_close_token_ids(adapter) is not None,
                        "constrained_tools": self.constrained_tool_grammar and callable(
                            getattr(adapter, "tool_constraint", None)
                        ),
                        "constrained_tools_implemented": callable(
                            getattr(adapter, "tool_constraint", None)
                        ),
                    },
                    "http": {"max_request_bytes": self.max_request_bytes},
                    # Declared vendor sampling profiles, plus any drift
                    # between them and this artifact's generation config.
                    "sampling_defaults": (
                        {
                            **vendor_sampling(adapter).as_dict(),
                            "artifact": generation_config_drift(
                                self.model_path, vendor_sampling(adapter)
                            ),
                        }
                        if vendor_sampling(adapter) is not None
                        else None
                    ),
                    # Readiness is also the status contract boundary.  Publish
                    # every mechanism counter before exposing ``ready`` so an
                    # immediate qualification baseline cannot race the first
                    # periodic telemetry refresh below.
                    "apcv2": apc.apc_stats,
                    "cache_capsules": (
                        capsule_pool.counters if capsule_pool is not None else None
                    ),
                    "approximate_kv": approximate_status,
                    "scheduler": dict(batch.scheduler_stats),
                    "segmented_self_mtp": segmented_self_mtp_stats(),
                    "spomin_live_surgery": (
                        spomin_manager.snapshot()
                        if spomin_manager is not None
                        else {"enabled": False, "counts": {}}
                    ),
                    "metal_active_bytes": mx.get_active_memory(),
                    "metal_peak_bytes": mx.get_peak_memory(),
                    "process_physical_footprint_bytes": physical_footprint_bytes(),
                    "execution": adapter.diagnostics(),
                    "admission": dict(admission),
                    "memory_waiting": 0,
                    "headroom_bytes": execution_headroom(),
                }
            self.ready.set()
            last_snapshot = 0
            last_reclaim = 0
            while not self.stop_event.is_set():
                quiesce_action = self._worker_quiesce_action()
                if quiesce_action is not None:
                    if quiesce_action["timed_out"]:
                        self._fail_drain_timeout(
                            batch,
                            active,
                            deferred,
                            published,
                            held_cohort,
                            attaching_cohort,
                        )
                        held_cohort = attaching_cohort = None
                    suspend_report = None
                    if quiesce_action["suspend"]:
                        try:
                            for prepared in tuple(self.fanout_capsules.values()):
                                prepared.close()
                            self.fanout_capsules.clear()
                            release_capsules = getattr(
                                batch, "release_cache_capsules", None
                            )
                            if callable(release_capsules):
                                release_capsules()
                            get_cached = getattr(mx, "get_cache_memory", lambda: 0)
                            memory_before = {
                                "active_bytes": int(mx.get_active_memory()),
                                "cached_bytes": int(get_cached()),
                            }
                            suspend_report = apc.suspend_resident()
                            mx.clear_cache()
                            suspend_report["memory_before"] = memory_before
                            suspend_report["memory_after"] = {
                                "active_bytes": int(mx.get_active_memory()),
                                "cached_bytes": int(get_cached()),
                            }
                            self.counts["suspends"] += 1
                            self.counts["suspended_entries"] += int(
                                suspend_report["entries"]
                            )
                            self.counts["suspended_bytes"] += int(
                                suspend_report["bytes"]
                            )
                            self.counts["suspend_failures"] += int(
                                suspend_report["failures"]
                            )
                            with self.lock:
                                self.snapshot.update(
                                    {
                                        "apcv2": apc.apc_stats,
                                        "metal_active_bytes": mx.get_active_memory(),
                                        "metal_peak_bytes": mx.get_peak_memory(),
                                    }
                                )
                        except Exception as error:  # noqa: BLE001 - stay live, fail soft
                            log.exception("cache suspension failed")
                            self.counts["suspend_failures"] += 1
                            suspend_report = {
                                "status": "failed",
                                "error": f"{type(error).__name__}: {error}",
                            }
                            quiesce_action["target"] = "quiesced"
                    self._complete_worker_quiesce(
                        quiesce_action, suspend_report=suspend_report
                    )
                self._expire_pending_cohorts()
                # Pending requests own their warm leases, but no execution
                # lane. Cancellation/deadlines must progress even at full width.
                for _ in range(len(deferred)):
                    waiting = deferred.popleft()
                    if waiting.cancelled.is_set():
                        self._cancel_pending_cache_capsule(
                            batch, waiting, "queued_member_cancelled"
                        )
                        self._finish(waiting, {"error": "cancelled"})
                        self.counts["cancelled"] += 1
                    elif time.monotonic() >= waiting.admission_deadline:
                        if not waiting.admission_final_reclaim_done:
                            self._clear_allocator_cache_before_reject(
                                synchronize=True
                            )
                            waiting.admission_final_reclaim_done = True
                            waiting.admission_retry_at = time.monotonic()
                            waiting.admission_deadline = (
                                time.monotonic() + self.MEMORY_ADMISSION_RETRY
                            )
                            deferred.append(waiting)
                        else:
                            self._fail_deferred_admission_timeout(batch, waiting)
                    else:
                        deferred.append(waiting)
                waiting = None
                # Admission is bounded before prompt caches are allocated.
                coalescer = IdleAdmissionCoalescer(
                    self.coalesce_window_seconds,
                )
                retry_budget = len(deferred)
                while len(active) < self.max_lanes:
                    try:
                        queued_job = False
                        timeout = coalescer.timeout(
                            now=time.monotonic(),
                            idle=not active,
                            deferred=bool(deferred),
                        )
                        if held_cohort is not None:
                            # An explicit cohort never joins an in-flight
                            # physical batch.  Preserve queue order and wait
                            # until it can own all configured lanes.
                            if active:
                                break
                            attaching_cohort = held_cohort
                            held_cohort = None
                            published.extend(attaching_cohort.jobs)
                            job = published.popleft()
                            queued_job = True
                        elif published:
                            job = published.popleft()
                            queued_job = True
                        elif retry_budget and deferred and (
                            deferred[0].admission_retry_at <= time.monotonic()
                            or deferred[0].cancelled.is_set()
                        ):
                            job = deferred.popleft()
                            retry_budget -= 1
                            self.counts["memory_admission_retries"] += 1
                        else:
                            item = self.incoming.get(timeout=timeout)
                            if isinstance(item, PublishedCohort) and not item.atomic:
                                published.extend(item.jobs)
                                job = published.popleft()
                            elif isinstance(item, PublishedCohort):
                                if active:
                                    held_cohort = item
                                    break
                                attaching_cohort = item
                                published.extend(item.jobs)
                                job = published.popleft()
                            else:
                                job = item
                            queued_job = True
                        if queued_job:
                            with self.lock:
                                self.queued_jobs -= 1
                                queue_depth = self.queued_jobs
                        else:
                            with self.lock:
                                queue_depth = self.queued_jobs
                        self.batch_metrics.dequeued(job.id, queue_depth)
                    except queue.Empty:
                        break
                    if job.cancelled.is_set():
                        if attaching_cohort is not None:
                            self._fail_attaching_cohort(
                                batch,
                                active,
                                published,
                                attaching_cohort,
                                {
                                    "error": "declared batch cohort member cancelled before atomic attachment",
                                    "status": 429,
                                },
                            )
                            attaching_cohort = None
                        else:
                            self._cancel_pending_cache_capsule(
                                batch, job, "queued_member_cancelled"
                            )
                            self._finish(job, {"error": "cancelled"})
                        continue
                    try:
                        if not job.started:
                            job.started = time.monotonic()
                            job.admission_deadline = job.started + self.MEMORY_ADMISSION_TIMEOUT
                            if job.fault and not job.fault_fired and job.fault.kind in {
                                "cache_evict",
                                "cache_reallocate",
                            }:
                                if job.fault.kind == "cache_evict":
                                    apc.evict_oldest_unleased()
                                else:
                                    reclaim_allocator()
                                job.fault_fired = True
                                self.batch_metrics.fault(job.id, job.fault.kind)
                        if time.monotonic() >= job.admission_deadline:
                            if not job.admission_final_reclaim_done:
                                self._clear_allocator_cache_before_reject(
                                    synchronize=True
                                )
                                job.admission_final_reclaim_done = True
                                job.admission_deadline = (
                                    time.monotonic() + self.MEMORY_ADMISSION_RETRY
                                )
                            else:
                                raise Overloaded(
                                    "host memory admission did not recover before deadline"
                                )
                        tokens = job.admission_tokens
                        if tokens is None:
                            tokens = self.host_prompt_cache.get(job.request)
                            if tokens is None:
                                with self.prompt_lock:
                                    tokens = adapter.prompt_tokens(job.request)
                                self.host_prompt_cache.put(job.request, tokens)
                            job.admission_tokens = tokens
                        job.prompt_tokens = len(tokens)
                        context_limit = min(self.max_context, job.request.get("context_limit", self.max_context))
                        if not tokens:
                            raise ValueError(
                                f"prompt plus output must fit {context_limit} tokens"
                            )
                        if job.effective_max_tokens is None:
                            maximum, defaulted = resolve_output_limit(
                                job.request,
                                prompt_tokens=len(tokens),
                                effective_context=context_limit,
                                default_max_tokens=self.default_max_tokens,
                            )
                            job.effective_max_tokens = maximum
                            job.max_tokens_defaulted = defaulted
                            # Downstream generation owns one concrete cap; the
                            # separate Job flag preserves whether the client
                            # supplied it for the terminal receipt.
                            job.request["max_tokens"] = maximum
                        else:
                            maximum = job.effective_max_tokens
                        # Lease resident state before pressure eviction. Disk
                        # restoration remains behind the cold allocation gate.
                        hit = job.admission_hit
                        if hit is None:
                            key = cache_key_for(
                                job.tenant_id,
                                job.request.get("_mlx2_media_fingerprint"),
                            )
                            hit = route_usable_hit(
                                apc.lookup(
                                    key,
                                    tokens,
                                    allow_disk_restore=False,
                                    session_tag=session_tag_for(job),
                                ),
                                tokens,
                            )
                            media_end = job.request.get("_mlx2_media_token_end", 0)
                            if hit.cached_tokens and hit.cached_tokens < media_end:
                                if hit.cache is not None and hasattr(hit.cache, "close"):
                                    hit.cache.close()
                                hit = APCLookup(
                                    None,
                                    list(tokens),
                                    0,
                                    False,
                                    None,
                                    "media_boundary_not_cached",
                                )
                                self.counts["multimodal_apcv2_boundary_misses"] += 1
                            job.admission_hit = hit
                            job.cache_branch = hit.cache
                        cache_copy = warm_cache_copy_gib(
                            hit, context_tokens=len(tokens) + maximum,
                            prefill_step=self.prefill_step, mtp=self.mtp,
                        )
                        required = controller.hard_reserve_gib + controller.lane_gib(
                            len(tokens) + maximum, config["num_draft"] if self.mtp else 0,
                            cache_gib=cache_copy,
                        )
                        if prompt_lookup:
                            # PLD verifies the anchor plus as many as
                            # ``num_draft`` proposed tokens in one target
                            # forward. Ordinary admission charged only one row
                            # and could admit a verification step that did not
                            # fit. Scale the calibrated width-3 transient to
                            # the actual forward width; the ordinary one-row
                            # share is already included above.
                            required += prompt_lookup_verification_gib(
                                controller, batch.num_draft
                            )
                        if not ensure_admission_headroom(
                            required * (1 << 30), headroom=execution_headroom,
                            reclaim=reclaim_allocator, evict=evict_unused_checkpoint,
                            evictable=getattr(apc, "unleased_resident_nbytes", None),
                        ):
                            if attaching_cohort is not None:
                                raise Overloaded(
                                    "declared batch cohort could not atomically admit every member"
                                )
                            if not job.admission_retry_at:
                                self.counts["memory_admission_deferred"] += 1
                            job.admission_retry_at = time.monotonic() + self.MEMORY_ADMISSION_RETRY
                            deferred.append(job)
                            continue
                        checkpoint_candidates = ()
                        if (
                            self.apc_interior_route_supported
                            and self.apc_interior_checkpoint_policy["count"] > 0
                            and not job.request.get("skip_writing_prefix_cache", False)
                            and (
                                hit.cache is None
                                or inspect_apc_capabilities(
                                    hit.cache
                                ).interior_checkpoint_target
                            )
                        ):
                            checkpoint_candidates = tuple(
                                position
                                for position in interior_checkpoint_positions(
                                    len(tokens),
                                    count=self.apc_interior_checkpoint_policy["count"],
                                    min_stride=self.apc_interior_checkpoint_policy[
                                        "min_stride"
                                    ],
                                )
                                if position > int(hit.cached_tokens)
                            )
                        job.apc_interior_positions, _checkpoint_bytes = (
                            budget_interior_checkpoint_positions(
                                checkpoint_candidates,
                                available_bytes=max(
                                    0,
                                    int(execution_headroom())
                                    - int(required * (1 << 30)),
                                ),
                                cache_projection=(
                                    None if cache_budget is None else cache_budget.project
                                ),
                            )
                        )
                        self.counts["apc_interior_checkpoints_degraded"] += (
                            len(checkpoint_candidates)
                            - len(job.apc_interior_positions)
                        )
                        if hit.miss_reason == "disk_restore_requires_admission":
                            hit = route_usable_hit(
                                apc.lookup(
                                    cache_key_for(
                                        job.tenant_id,
                                        job.request.get("_mlx2_media_fingerprint"),
                                    ),
                                    tokens,
                                    session_tag=session_tag_for(job),
                                ), tokens
                            )
                            job.cache_branch = hit.cache
                        job.cached_tokens = hit.cached_tokens
                        job.cache_retention_role = getattr(
                            hit, "retention_role", None
                        )
                        self.batch_metrics.prompt(
                            job.id, job.prompt_tokens, job.cached_tokens
                        )
                        job.detokenizer = adapter.tokenizer.detokenizer
                        job.detokenizer.reset()
                        parser_request = job.request
                        if self.tolerant_tool_markers:
                            parser_request = {
                                **job.request,
                                "_tolerant_tool_markers": True,
                            }
                            if job.request.get("tools"):
                                self.counts["tolerant_tool_marker_requests"] += 1
                        job.output_parser = adapter.output_parser(parser_request)
                        rng = LaneRNG(job.request.get("seed", secrets.randbits(32)))
                        # Vendor defaults fill only fields the request left
                        # unset; the selected profile follows the adapter's
                        # view of the thinking mode (unknown for raw
                        # completions).  Every route below consumes these
                        # effective values, so ordinary and speculative lanes
                        # sample the same penalized distribution.
                        sampling, job.sampling_defaults = resolve_sampling(
                            job.request,
                            vendor_sampling(adapter),
                            thinking=(
                                thinking_enabled(adapter, job.request)
                                if "messages" in job.request
                                else None
                            ),
                        )
                        job.effective_sampling = sampling
                        temp = sampling["temperature"]
                        top_p, top_k = sampling["top_p"], sampling["top_k"]
                        min_p = sampling["min_p"]
                        vocab_size = adapter.tokenizer.vocab_size
                        # apply_top_k requires 0 < top_k < vocab_size; equality slipped
                        # through here and raised inside the batched decode step.
                        if top_k >= vocab_size or any(int(token) >= vocab_size for token in job.request.get("logit_bias", {})):
                            raise ValueError("sampling token IDs and top_k must fit the model vocabulary")
                        transform = (
                            make_transformed_logprobs(
                                temp, top_p=top_p, top_k=top_k, min_p=min_p
                            )
                            if temp
                            else None
                        )

                        def sampler(logprobs, rng=rng, transform=transform, temp=temp):
                            return (
                                mx.argmax(logprobs, axis=-1)
                                if temp == 0
                                else mx.random.categorical(
                                    transform(logprobs), key=draw_key(rng)
                                )
                            )

                        processors = make_logits_processors(
                            logit_bias={int(k): v for k, v in job.request.get("logit_bias", {}).items()},
                            # Repetition: prompt + generated (HF/vLLM).
                            # Presence/frequency: generated only (OpenAI/vLLM).
                            repetition_penalty=(sampling["repetition_penalty"] if sampling["repetition_penalty"] != 1.0 else None),
                            repetition_context_size=0,
                            presence_penalty=sampling["presence_penalty"],
                            presence_context_size=0,
                            frequency_penalty=sampling["frequency_penalty"],
                            frequency_context_size=0,
                            penalty_generation_start=len(tokens),
                        )
                        minimum_processor = minimum_tokens_processor(
                            mx,
                            stop_token_ids,
                            len(tokens),
                            job.request.get("min_tokens", 0),
                        )
                        if minimum_processor is not None:
                            processors.append(minimum_processor)
                        job.thinking_guard = None
                        from .thinking_guard import ThinkingGuard, resolve_thinking_budget

                        think_budget = (
                            resolve_thinking_budget(job.request, self.thinking_budget)
                            if "messages" in job.request
                            and thinking_enabled(adapter, job.request)
                            else 0
                        )
                        thinking_budget_mode = job.request.get(
                            "thinking_budget_mode", "state_aware"
                        )
                        close_ids = thinking_close_token_ids(adapter) if think_budget else None
                        if think_budget and close_ids is None:
                            if job.request.get("thinking_budget"):
                                raise ValueError(
                                    "thinking_budget needs an adapter-declared thinking-close token"
                                )
                            think_budget = 0  # a server default never fails a request
                        if (
                            think_budget
                            and thinking_budget_mode == "state_aware"
                            and len(close_ids or ()) != 1
                        ):
                            if job.request.get("thinking_budget"):
                                raise ValueError(
                                    "state-aware thinking_budget needs a single-token thinking-close marker"
                                )
                            think_budget = 0
                        steer_alpha = float(
                            self.thinking_steer_alpha
                            if job.request.get("thinking_steer_alpha") is None
                            else job.request["thinking_steer_alpha"]
                        )
                        direction = None
                        if steer_alpha > 0 and "messages" in job.request and thinking_enabled(adapter, job.request):
                            # Only ever a direction bound to the loaded artifact.
                            direction = self._commit_direction
                            if direction is None and job.request.get("thinking_steer_alpha"):
                                raise ValueError(
                                    "thinking_steer_alpha needs a commit direction calibrated for "
                                    "this exact artifact; none is available"
                                )
                            if direction is not None and close_ids is None:
                                close_ids = thinking_close_token_ids(adapter)
                        if (
                            len(close_ids or ()) == 1
                            and (think_budget or direction is not None)
                        ):
                            job.thinking_guard = ThinkingGuard(
                                len(tokens), close_ids,
                                budget=(
                                    think_budget
                                    if thinking_budget_mode == "state_aware"
                                    else None
                                ),
                                direction=direction, alpha=steer_alpha if direction else 0.0,
                                hammer=self.thinking_steer_hammer if direction else 0.0,
                            )
                            # Before any grammar: the guard only acts while the
                            # reasoning channel is open, the grammar only after.
                            processors.append(job.thinking_guard)
                        from .structured_output import (
                            ThinkingBudgetProcessor,
                            make_structured_processor,
                            structured_receipt,
                        )

                        # Chat requests with thinking enabled start inside the
                        # reasoning channel: the grammar is deferred until the
                        # adapter's thinking-close marker has been generated.
                        defer_until = (
                            thinking_close_token_ids(adapter)
                            if "messages" in job.request
                            and thinking_enabled(adapter, job.request)
                            else None
                        )
                        tool_grammar = None
                        from .output import constrained_tool_choice

                        strict_auto = (
                            job.request.get("tool_choice", "auto") == "auto"
                            and any(
                                tool["function"].get("strict", False)
                                for tool in job.request.get("tools", ())
                            )
                        )
                        if self.constrained_tool_grammar and strict_auto:
                            job.tool_grammar_status = "skipped_strict_auto"
                            self.counts["constrained_tool_grammar_skips"] += 1
                        elif (
                            self.constrained_tool_grammar
                            and constrained_tool_choice(job.request)
                        ):
                            incompatible = (
                                "response_format" in job.request
                                or "grammar" in job.request
                                or bool(job.request.get("min_tokens", 0))
                            )
                            accessor = getattr(adapter, "tool_constraint", None)
                            if incompatible:
                                job.tool_grammar_status = "skipped_request_combination"
                                self.counts["constrained_tool_grammar_skips"] += 1
                            elif not callable(accessor):
                                job.tool_grammar_status = "skipped_adapter_unsupported"
                                self.counts["constrained_tool_grammar_skips"] += 1
                            else:
                                tool_grammar = accessor(job.request)
                                if tool_grammar:
                                    job.tool_grammar_status = "engaged"
                                    self.counts[
                                        "constrained_tool_grammar_engagements"
                                    ] += 1
                                else:
                                    job.tool_grammar_status = "skipped_adapter_unsupported"
                                    self.counts["constrained_tool_grammar_skips"] += 1
                        budget = think_budget
                        budget_processor = None
                        if (
                            budget
                            and thinking_budget_mode == "history"
                            and thinking_enabled(adapter, job.request)
                        ):
                            if defer_until is None:
                                raise ValueError(
                                    "thinking_budget requires an adapter thinking-close marker"
                                )
                            budget_processor = ThinkingBudgetProcessor(
                                len(tokens), budget, defer_until
                            )
                            processors.append(budget_processor)
                        job.thinking_budget = budget_processor
                        structured = make_structured_processor(
                            adapter.tokenizer,
                            len(tokens),
                            response_format=job.request.get("response_format"),
                            grammar=tool_grammar or job.request.get("grammar"),
                            constraint_kind=(
                                "tool_grammar"
                                if tool_grammar
                                else (job.request.get("response_format") or {}).get(
                                    "type", "grammar"
                                )
                            ),
                            generation_stop_token_ids=stop_token_ids,
                            capture_failure_context=self.qualification_mode,
                            greedy=(temp == 0),
                            top_k=top_k,
                            top_p=top_p,
                            defer_until=defer_until,
                            # Chat only: a raw completion has no channel framing.
                            envelope=(
                                structured_envelope_token_ids(adapter)
                                if "messages" in job.request
                                else None
                            ),
                        )
                        job.structured = structured
                        if structured is not None:
                            if (
                                defer_until is None
                                and "messages" in job.request
                                and thinking_enabled(adapter, job.request)
                            ):
                                # The adapter declares no thinking-close
                                # marker, so the grammar would constrain from
                                # the first generated token while the output
                                # parser is inside the think channel: the
                                # constrained text would land in
                                # reasoning_content with an empty answer.
                                # Refuse rather than misreport.
                                raise ValueError(
                                    "structured output requires thinking to be disabled"
                                )
                            processors.append(structured)
                        sampling_config = {"sampling_temp": temp, "top_p": top_p, "top_k": top_k, "min_p": min_p, "shared_prefix_attestation": shared_prefix_attestation(hit)}
                        if job.request.get("batch_cohort") is not None:
                            sampling_config["batch_cohort"] = {
                                "tenant_id": job.tenant_id,
                                **job.request["batch_cohort"],
                            }
                        state_options = (
                            {"cache_states": [hit.sidecar], "sampling_configs": [sampling_config]}
                            if external_draft else
                            {"prompt_lookup_configs": [sampling_config]}
                            if prompt_lookup else
                            {"mtp_states": [hit.sidecar.state if hit.sidecar is not None else None], "self_mtp_configs": [sampling_config]}
                        )
                        lane_cache = hit.cache
                        if approximate_controller is not None:
                            if len(tokens) < self.approximate_kv_policy.start_tokens:
                                job.approximate_kv_receipt = {
                                    "schema": "mlx2.approximate-kv.v1",
                                    "request_id": job.id,
                                    "fidelity": "exact",
                                    "selected": False,
                                    "status": "declined",
                                    "reason": "below_start_tokens",
                                    "operation": approximate_operation.name,
                                    "start_tokens": self.approximate_kv_policy.start_tokens,
                                    "prompt_tokens": len(tokens),
                                    "quantized_layers": 0,
                                }
                                self.counts["approximate_kv_declined"] += 1
                            else:
                                # A warm hit is a per-lookup COW branch.  Its
                                # exact planes are only read: the staged tuple
                                # gets new quantized plane objects, and the
                                # branch itself is left for ``_finish`` to
                                # close.  Anything that is not such a private
                                # list fails the request closed.
                                source = hit.cache
                                requantized = bool(
                                    isinstance(source, list) and hit.cached_tokens
                                )
                                if not requantized:
                                    if source is not None:
                                        raise ValueError(
                                            "approximate KV cannot privately stage "
                                            "the restored prefix state"
                                        )
                                    source = make_prompt_cache(adapter.model)
                                updated, approximate_receipt = apply_approximate_kv(
                                    job.id, source
                                )
                                lane_cache = list(updated.planes)
                                job.approximate_kv_applied = True
                                job.approximate_kv_receipt = dict(
                                    approximate_receipt,
                                    descriptor=approximate_operation.descriptor.as_dict(),
                                    start_tokens=self.approximate_kv_policy.start_tokens,
                                    prompt_tokens=len(tokens),
                                    quantized_layers=updated.quantized_planes,
                                    requantized_prefix_tokens=(
                                        hit.cached_tokens if requantized else 0
                                    ),
                                    apcv2_publication="skipped",
                                )
                                self.counts["approximate_kv_applied"] += 1
                                self.counts[
                                    "approximate_kv_requantized_prefix_hits"
                                ] += int(requantized)
                        job.uid = batch.insert(
                            [hit.remaining_tokens], max_tokens=[maximum],
                            caches=[lane_cache], all_tokens=[tokens[:hit.cached_tokens]],
                            samplers=[sampler], logits_processors=[processors], lane_rngs=[rng],
                            apc_interior_positions=[job.apc_interior_positions],
                            **state_options,
                            prefill_inputs=[
                                job.request.get("_mlx2_prefill_inputs")
                                if not hit.cached_tokens
                                else None
                            ],
                        )[0]
                        if job.cache_capsule is not None:
                            self._bind_pending_cache_capsule(batch, job)
                        active[job.uid] = job
                        # Start the idle-to-active coalescing window at the
                        # lane-attachment seam.  Tokenization, APC lookup, and
                        # memory admission may take longer than the window;
                        # starting it at dequeue made the first decode cycle
                        # race later HTTP handlers and could split a requested
                        # B4 into fixed-width B2 segmented-MTP cohorts.
                        if len(active) == 1 and coalescer.deadline is None:
                            coalescer.note_attachment(now=time.monotonic())
                        job.last_progress = time.monotonic()
                        job.admission_hit = job.admission_tokens = None
                        self.counts["admitted"] += 1
                        self.batch_metrics.lane_attached(
                            job.id,
                            len(active),
                            "external_draft"
                            if external_draft
                            else "prompt_lookup"
                            if prompt_lookup
                            else "self_mtp"
                            if self.mtp
                            else "ordinary",
                        )
                        if attaching_cohort is not None and not published:
                            if any(
                                member.cancelled.is_set()
                                for member in attaching_cohort.jobs
                            ):
                                raise Overloaded(
                                    "declared batch cohort member cancelled before atomic attachment"
                                )
                            attaching_cohort = None
                    except Exception as exc:
                        if job.cache_capsule is not None:
                            if job.uid is not None and job.uid not in active:
                                batch.remove([job.uid])
                            self._cancel_pending_cache_capsule(
                                batch, job, "attachment_failed"
                            )
                        if isinstance(exc, Overloaded):
                            event = {"error": str(exc), "status": 429}
                        elif isinstance(exc, ValueError):
                            event = {"error": str(exc), "status": 400}
                        else:
                            log.exception("request attachment failed")
                            event = {
                                "error": "internal server error",
                                "status": 500,
                            }
                        if attaching_cohort is not None:
                            self._fail_attaching_cohort(
                                batch, active, published, attaching_cohort, event
                            )
                            attaching_cohort = None
                        else:
                            self._finish(job, event)
                    finally:
                        # The scheduler/APC own admitted cache state. Locals in
                        # this long-lived worker must not pin the last lookup
                        # across idle periods or a later pressure eviction.
                        hit = state_options = sampling_config = None
                        lane_cache = source = updated = None
                cancelled = [
                    uid for uid, job in active.items() if job.cancelled.is_set()
                ]
                if cancelled:
                    batch.remove(cancelled)
                    for uid in cancelled:
                        self._finish(active.pop(uid), {"error": "cancelled"})
                        self.counts["cancelled"] += 1
                # The watchdog below exists for lanes the memory controller
                # queued and that never recovered.  A lane the *scheduler* has
                # not reached yet (behind long prefills, or deferred by a live
                # width lock) is making no progress for a different reason and
                # must not be failed with a memory error.
                now = time.monotonic()
                for uid in getattr(batch, "scheduler_waiting_uids", lambda: ())():
                    waiting_job = active.get(uid)
                    if waiting_job is not None:
                        waiting_job.last_progress = now
                stalled = [
                    uid
                    for uid, job in active.items()
                    if now - job.last_progress > 60
                ]
                if stalled:
                    batch.remove(stalled)
                    for uid in stalled:
                        self._finish(
                            active.pop(uid),
                            {
                                "error": "memory admission did not permit progress",
                                "status": 429,
                            },
                        )
                if active:
                    self.counts["cycles"] += 1
                    self.counts["multi_request_cycles"] += int(len(active) > 1)
                    self.batch_metrics.batch_cycle(len(active), len(deferred))
                    prompts, responses = batch.next()
                    atomic_failures = getattr(
                        batch, "take_atomic_cohort_failures", lambda: ()
                    )()
                    for failure in atomic_failures:
                        failed_uids = tuple(failure["uids"])
                        batch.remove(failed_uids)
                        failed_jobs = [
                            active.pop(uid)
                            for uid in failed_uids
                            if uid in active
                        ]
                        self.counts["batch_cohort_scheduler_failures"] += 1
                        self.counts["batch_cohort_jobs_failed_closed"] += len(
                            failed_jobs
                        )
                        for failed_job in failed_jobs:
                            self._finish(
                                failed_job,
                                {"error": failure["reason"], "status": 429},
                            )
                    if not prompts and not responses:
                        # Reclaim idle scratch while memory admission is deferred.
                        now = time.monotonic()
                        if now - last_reclaim > 0.25:
                            if reclaim_deferred_cache(apc, admission, mx.clear_cache):
                                self.counts["memory_pressure_evictions"] += 1
                            last_reclaim = now
                        self.stop_event.wait(0.01)
                    for response in prompts:
                        if response.uid in active:
                            active[response.uid].last_progress = time.monotonic()
                        if response.end_of_prompt:
                            pop_surgery_receipt = getattr(
                                batch, "pop_post_prefill_receipt", None
                            )
                            surgery_receipt = (
                                pop_surgery_receipt(response.uid)
                                if callable(pop_surgery_receipt)
                                else None
                            )
                            owner = active.get(response.uid)
                            if owner is not None and surgery_receipt is not None:
                                owner.spomin_receipt = surgery_receipt
                                self.counts[
                                    "spomin_" + surgery_receipt.get("status", "unknown")
                                ] += 1
                            pop_interiors = getattr(
                                batch, "pop_interior_checkpoints", None
                            )
                            interiors = (
                                pop_interiors(response.uid)
                                if callable(pop_interiors)
                                else ()
                            )
                            self.counts["apc_interior_checkpoints_captured"] += len(
                                interiors
                            )
                            for checkpoint in interiors:
                                if owner is not None and owner.request.get(
                                    "skip_writing_prefix_cache", False
                                ):
                                    self.counts[
                                        "apc_interior_checkpoints_skipped_write_suppressed"
                                    ] += 1
                                    continue
                                if (
                                    owner is not None
                                    and owner.approximate_kv_applied
                                ) or (
                                    surgery_receipt is not None
                                    and surgery_receipt.get("status") == "applied"
                                ):
                                    self.counts[
                                        "apc_interior_checkpoints_skipped_approximate"
                                    ] += 1
                                    continue
                                checkpoint_sidecar = (
                                    MTPAPCSidecar(
                                        checkpoint["mtp_state"],
                                        checkpoint["covered_tokens"],
                                        rng_key=checkpoint.get("rng_key"),
                                        rng_draws=checkpoint.get("rng_draws", 0),
                                    )
                                    if checkpoint.get("mtp_state")
                                    else None
                                )
                                checkpoint_key = cache_key_for(
                                    owner.tenant_id if owner else None,
                                    (
                                        owner.request.get("_mlx2_media_fingerprint")
                                        if owner is not None
                                        else None
                                    ),
                                )
                                if self._publish_checkpoint(
                                    apc,
                                    checkpoint_key,
                                    checkpoint["tokens"],
                                    checkpoint["target_cache"],
                                    sidecar=checkpoint_sidecar,
                                    retention_role="interior_checkpoint",
                                    session_tag=session_tag_for(owner),
                                ):
                                    self.counts[
                                        "apc_interior_checkpoints_published"
                                    ] += 1
                                else:
                                    self.counts[
                                        "apc_interior_checkpoints_skipped_publish_failed"
                                    ] += 1
                            boundary = batch.pop_prompt_boundary(response.uid)
                            if (
                                boundary
                                and surgery_receipt is not None
                                and surgery_receipt.get("status") == "applied"
                                and not boundary.get("pre_transform_exact", False)
                            ):
                                # Compacted rows still carry the removed
                                # history's influence: this is approximate
                                # state and must never enter the exact prefix
                                # cache under the retained-token key.
                                self.counts["apcv2_store_skipped_approximate"] += 1
                                boundary = None
                            if (
                                boundary
                                and owner is not None
                                and owner.request.get(
                                    "skip_writing_prefix_cache", False
                                )
                            ):
                                boundary = None
                            if (
                                boundary
                                and boundary.get("committed_only")
                                and boundary["tokens"]
                            ):
                                sidecar = boundary.get("cache_sidecar") or (
                                    MTPAPCSidecar(
                                        boundary["mtp_state"],
                                        boundary["covered_tokens"],
                                    )
                                    if boundary.get("mtp_state")
                                    else None
                                )
                                boundary_job = active.get(response.uid)
                                key = cache_key_for(
                                    boundary_job.tenant_id if boundary_job else None,
                                    boundary_job.request.get("_mlx2_media_fingerprint")
                                    if boundary_job
                                    else None,
                                )
                                stored = self._publish_checkpoint(
                                    apc,
                                    key,
                                    boundary["tokens"],
                                    boundary["target_cache"],
                                    sidecar=sidecar,
                                    # The shadow is an ordinary exact committed
                                    # boundary; its origin is tracked by the
                                    # Spomin counters, not a new APCv2 role.
                                    retention_role="committed_prompt_boundary",
                                    approximate=bool(
                                        boundary_job is not None
                                        and boundary_job.approximate_kv_applied
                                    ),
                                    session_tag=session_tag_for(boundary_job),
                                )
                                if boundary.get("pre_transform_exact", False):
                                    self.counts[
                                        "spomin_exact_boundary_stores"
                                        if stored
                                        else "spomin_exact_boundary_store_failures"
                                    ] += 1
                                leader = active.get(response.uid)
                                if (
                                    leader is not None
                                    and leader.fanout_role == "prefill_leader"
                                ):
                                    with self.lock:
                                        siblings = self.fanout_waiting.pop(
                                            leader.fanout_group, ()
                                        )
                                    if siblings and stored:
                                        expected = len(boundary["tokens"])
                                        prepared = []
                                        reason = None
                                        for sibling in siblings:
                                            try:
                                                sibling_tokens = self.host_prompt_cache.get(
                                                    sibling.request
                                                )
                                                if sibling_tokens is None:
                                                    with self.prompt_lock:
                                                        sibling_tokens = adapter.prompt_tokens(
                                                            sibling.request
                                                        )
                                                    self.host_prompt_cache.put(
                                                        sibling.request, sibling_tokens
                                                    )
                                                sibling_hit = route_usable_hit(
                                                    apc.lookup(
                                                        key,
                                                        sibling_tokens,
                                                        allow_disk_restore=False,
                                                        session_tag=session_tag_for(sibling),
                                                    ),
                                                    sibling_tokens,
                                                )
                                            except Exception:  # noqa: BLE001 - fail the group, not the worker
                                                log.exception(
                                                    "APCv2 fanout boundary acquisition failed"
                                                )
                                                reason = "boundary_lookup_failed"
                                                break
                                            if (
                                                expected <= 0
                                                or sibling_hit.cache is None
                                                or sibling_hit.cached_tokens != expected
                                                or sibling_tokens[:expected]
                                                != boundary["tokens"][:expected]
                                            ):
                                                reason = "committed_boundary_miss"
                                                close = getattr(
                                                    sibling_hit.cache, "close", None
                                                )
                                                if callable(close):
                                                    close()
                                                break
                                            sibling.admission_tokens = list(
                                                sibling_tokens
                                            )
                                            sibling.admission_hit = sibling_hit
                                            sibling.cache_branch = sibling_hit.cache
                                            sibling.fanout_boundary_tokens = expected
                                            sibling.fanout_one_prefill = True
                                            sibling.fanout_reason = "boundary_reused"
                                            prepared.append(sibling)
                                        if reason is not None:
                                            for sibling in prepared:
                                                close = getattr(
                                                    sibling.cache_branch, "close", None
                                                )
                                                if callable(close):
                                                    close()
                                                sibling.cache_branch = None
                                                sibling.admission_hit = None
                                                sibling.admission_tokens = None
                                                sibling.fanout_one_prefill = False
                                                sibling.fanout_reason = reason
                                            leader.fanout_reason = reason
                                            self.counts[
                                                "apcv2_fanout_boundary_misses"
                                            ] += 1
                                            for sibling in siblings:
                                                self._finish(
                                                    sibling,
                                                    {
                                                        "error": "APCv2 fanout could not lease the committed prompt boundary",
                                                        "status": 503,
                                                    },
                                                )
                                            siblings = ()
                                    elif siblings:
                                        leader.fanout_reason = "checkpoint_not_stored"
                                        self.counts[
                                            "apcv2_fanout_store_failures"
                                        ] += 1
                                        for sibling in siblings:
                                            sibling.fanout_reason = "checkpoint_not_stored"
                                            self._finish(
                                                sibling,
                                                {
                                                    "error": "APCv2 fanout prompt boundary was not stored",
                                                    "status": 503,
                                                },
                                            )
                                        siblings = ()
                                    if siblings:
                                        if (
                                            capsule_pool is not None
                                            and len(siblings) >= 2
                                            and self._cache_capsule_fanout_fits(
                                                active, siblings
                                            )
                                        ):
                                            from .runtime.cache_capsule import (
                                                inspect_kv_cache_capsule,
                                                prepare_prompt_cache_capsules,
                                            )

                                            capsule_hit = apc.lookup(
                                                key, boundary["tokens"]
                                            )
                                            prepared = None
                                            started = time.monotonic()
                                            try:
                                                compatible = bool(
                                                    capsule_hit.hit
                                                    and capsule_hit.cache
                                                    and capsule_hit.capsule_generation
                                                    is not None
                                                    and any(
                                                        inspect_kv_cache_capsule(
                                                            plane, len(siblings)
                                                        )[0]
                                                        for plane in capsule_hit.cache
                                                    )
                                                )
                                                if compatible:
                                                    prepared = prepare_prompt_cache_capsules(
                                                        capsule_hit.cache,
                                                        target_batch=len(siblings),
                                                        generation=capsule_hit.capsule_generation,
                                                        pool=capsule_pool,
                                                        compatibility_signature=apc.capsule_compatibility_signature(
                                                            key, apc.layout_name
                                                        ),
                                                        backend=self.cache_capsule_policy["backend"],
                                                        fallback=self.cache_capsule_policy["fallback"],
                                                        source_prefix=leader.fanout_group,
                                                    )
                                                    elapsed_ms = (
                                                        time.monotonic() - started
                                                    ) * 1000
                                                    if elapsed_ms > self.cache_capsule_policy["deadline_ms"]:
                                                        prepared.close()
                                                        prepared = None
                                                        self.counts[
                                                            "cache_capsule_deadline_fallbacks"
                                                        ] += 1
                                                else:
                                                    self.counts[
                                                        "cache_capsule_incompatible_fallbacks"
                                                    ] += 1
                                            except Exception:
                                                if prepared is not None:
                                                    prepared.close()
                                                prepared = None
                                                self.counts[
                                                    "cache_capsule_prepare_fallbacks"
                                                ] += 1
                                                log.exception(
                                                    "cache capsule preparation declined"
                                                )
                                            finally:
                                                close = getattr(
                                                    capsule_hit.cache, "close", None
                                                )
                                                if callable(close):
                                                    close()
                                            if prepared is not None:
                                                self.fanout_capsules[
                                                    leader.fanout_group
                                                ] = prepared
                                                for sibling in siblings:
                                                    sibling.cache_capsule = prepared
                                                    sibling.cache_capsule_rows = len(
                                                        siblings
                                                    )
                                                self.counts[
                                                    "cache_capsule_prepared"
                                                ] += 1
                                        leader.fanout_boundary_tokens = expected
                                        leader.fanout_one_prefill = True
                                        leader.fanout_reason = "boundary_reused"
                                        # Siblings must join the leader's live
                                        # batch; an atomic cohort would be held
                                        # until every active lane drained.
                                        self._publish_jobs(
                                            siblings,
                                            already_registered=True,
                                            atomic=False,
                                        )
                                        self.counts[
                                            "apcv2_fanout_boundaries"
                                        ] += 1
                    for response in responses:
                        job = active.get(response.uid)
                        if job is None:
                            continue
                        if job.thinking_budget is not None:
                            job.thinking_tokens.append(int(response.token))
                            self._observe_thinking_budget(job)
                        job.last_progress = time.monotonic()
                        failure = getattr(job.structured, "failure", None)
                        if failure is not None:
                            # The grammar dead-ended or overran its budget; the
                            # processor stopped masking so the batch stayed
                            # consistent.  Fail this request closed here.
                            batch.remove([response.uid])
                            self.counts["structured_output_failures"] += 1
                            from .structured_automaton import NO_CONTINUATION

                            failure_context = getattr(
                                job.structured, "failure_context", None
                            )
                            if failure == NO_CONTINUATION:
                                self.counts["structured_output_dead_ends"] += 1
                            event = {
                                "error": f"structured output failed closed: {failure}",
                                "status": 502,
                            }
                            if self.qualification_mode and failure_context is not None:
                                receipt = {
                                    "structured_output_failure": failure_context
                                }
                                event["mlx2"] = receipt
                                log.error(
                                    "structured-output dead end receipt=%s",
                                    json.dumps(
                                        receipt,
                                        ensure_ascii=True,
                                        separators=(",", ":"),
                                    ),
                                )
                            self._finish(
                                job,
                                event,
                            )
                            del active[response.uid]
                            continue
                        job.observed_width = max(
                            job.observed_width, response.execution_width
                        )
                        if job.first_token is None:
                            job.first_token = time.monotonic()
                        job.completion_tokens += 1
                        self.batch_metrics.token(job.id)
                        if (
                            job.fault
                            and not job.fault_fired
                            and job.fault.kind == "lane_abort"
                            and job.completion_tokens >= max(1, job.fault.after_tokens)
                        ):
                            job.fault_fired = True
                            self.batch_metrics.fault(job.id, job.fault.kind)
                            batch.remove([response.uid])
                            self._finish(
                                job,
                                {
                                    "error": "qualification fault injected: lane_abort",
                                    "status": 503,
                                },
                            )
                            del active[response.uid]
                            continue
                        if wants_logprobs(job.request):
                            try:
                                probability = token_logprob(
                                    response.logprobs, response.token, adapter.tokenizer,
                                    top_n=job.request.get("top_logprobs", 0), array_module=mx,
                                )
                            except Exception as exc:  # noqa: BLE001 - one lane, not the worker
                                batch.remove([response.uid])
                                self._finish(job, {"error": str(exc), "status": 502})
                                del active[response.uid]
                                continue
                            self._emit(job, {"logprob": probability})
                        if response.finish_reason != "stop":
                            job.detokenizer.add_token(response.token)
                        if response.finish_reason:
                            job.detokenizer.finalize()
                        text = job.detokenizer.last_segment
                        try:
                            deltas = job.output_parser.push(
                                text, final=bool(response.finish_reason)
                            )
                        except Exception as exc:  # noqa: BLE001 - one lane, not the worker
                            batch.remove([response.uid])
                            if getattr(exc, "tool_call_constraint_error", False):
                                self.counts["tool_call_constraint_failures"] += 1
                            self._finish(job, {"error": str(exc), "status": 502})
                            del active[response.uid]
                            continue
                        for delta in deltas:
                            self._emit(job, {"delta": delta})
                        parse_fallbacks = int(
                            getattr(
                                job.output_parser,
                                "tool_call_parse_fallbacks",
                                0,
                            )
                        )
                        if parse_fallbacks > job.tool_parse_fallbacks_seen:
                            self.counts["tool_call_parse_fallbacks"] += (
                                parse_fallbacks - job.tool_parse_fallbacks_seen
                            )
                            job.tool_parse_fallbacks_seen = parse_fallbacks
                        stopped = job.output_parser.stopped
                        if stopped and not response.finish_reason:
                            batch.remove([response.uid])
                        if response.finish_reason or stopped:
                            self._observe_thinking_budget(job, final=True)
                            sidecar = getattr(response, "cache_sidecar", None) or (
                                MTPAPCSidecar(
                                    response.mtp_state, len(response.all_tokens)
                                )
                                if response.mtp_state
                                else None
                            )
                            # Residual steering changes the K/V written for the
                            # steered reasoning tokens, so the finished lane is
                            # not an exact prefix for an unsteered request.  The
                            # prompt boundary was stored before any steering.
                            approximate_state = (
                                job.spomin_receipt or {}
                            ).get("status") == "applied" or bool(
                                getattr(job.thinking_guard, "steered_steps", 0)
                            )
                            if response.finish_reason and approximate_state:
                                self.counts["apcv2_store_skipped_approximate"] += 1
                            elif response.finish_reason and not job.request.get(
                                "skip_writing_prefix_cache", False
                            ):
                                self._publish_checkpoint(
                                    apc,
                                    cache_key_for(
                                        job.tenant_id,
                                        job.request.get("_mlx2_media_fingerprint"),
                                    ),
                                    response.all_tokens,
                                    response.prompt_cache,
                                    sidecar=sidecar,
                                    approximate=job.approximate_kv_applied,
                                    session_tag=session_tag_for(job),
                                )
                            # Sanity telemetry: always-on host integers, no
                            # device sync.  Enough to tell from /v1/status alone
                            # how a batch of requests ended and how wide it ran.
                            reason = response.finish_reason or ("stop" if stopped else "unknown")
                            self.counts["finish_" + str(reason)] += 1
                            self.counts["completion_tokens"] += int(job.completion_tokens or 0)
                            self.counts["prompt_tokens"] += int(job.prompt_tokens or 0)
                            self.counts["cached_prompt_tokens"] += int(job.cached_tokens or 0)
                            self.counts["peak_observed_width"] = max(
                                self.counts["peak_observed_width"], int(job.observed_width or 0)
                            )
                            if job.structured is not None:
                                self.counts[
                                    "structured_engine_" + getattr(job.structured, "engine", "scanner")
                                ] += 1
                                if getattr(job.structured, "deferred", False):
                                    self.counts["structured_deferred"] += 1
                            budget_fired = bool(job.thinking_budget_fired)
                            receipt = {
                                "request_id": job.id,
                                "cache": "apcv2",
                                "cached_tokens": job.cached_tokens,
                                "cache_checkpoint_role": job.cache_retention_role,
                                "stop_sequence": getattr(
                                    job.output_parser, "stop_sequence", None
                                ),
                                "parallel_prefill": (
                                    {
                                        "schema": "mlx2.apcv2-fanout.v1",
                                        "group": job.fanout_group,
                                        "role": job.fanout_role,
                                        "one_prefill": job.fanout_one_prefill,
                                        "boundary_tokens": job.fanout_boundary_tokens,
                                        "reason": job.fanout_reason,
                                    }
                                    if job.fanout_group
                                    else None
                                ),
                                "profile": self.snapshot["profile"],
                                "qualification": self.snapshot["qualification"],
                                "route_receipt": route_receipt,
                                "request_controls": {
                                    "max_tokens": job.effective_max_tokens,
                                    "max_tokens_defaulted": job.max_tokens_defaulted,
                                    "default_max_tokens": self.default_max_tokens,
                                    "logprobs": wants_logprobs(job.request),
                                    "top_logprobs": job.request.get("top_logprobs", 0),
                                    "logprob_semantics": (
                                        ("execution_target: external=transformed_target_verifier" if external_draft else "execution_target: ordinary=post_processor_pre_sampler; mtp=transformed_target_verifier")
                                        if wants_logprobs(job.request) else None
                                    ),
                                    "thinking": thinking_enabled(adapter, job.request),
                                    "thinking_budget": job.request.get("thinking_budget"),
                                    "thinking_budget_mode": job.request.get(
                                        "thinking_budget_mode", "state_aware"
                                    ),
                                    "thinking_budget_fired": budget_fired,
                                    "thinking_mechanisms": {
                                        "state_aware_guard": job.thinking_guard is not None,
                                        "history_budget": job.thinking_budget is not None,
                                        "steering": bool(
                                            job.thinking_guard is not None
                                            and getattr(
                                                job.thinking_guard,
                                                "_direction",
                                                None,
                                            )
                                            is not None
                                        ),
                                    },
                                    "sampling": {name: job.request[name] for name in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty", "logit_bias", "seed") if name in job.request},
                                    # What the samplers actually used, and
                                    # which vendor defaults filled it in.
                                    "effective_sampling": job.effective_sampling,
                                    "sampling_defaults": job.sampling_defaults,
                                    "min_tokens": job.request.get("min_tokens", 0),
                                    "thinking_guard": (
                                        job.thinking_guard.receipt()
                                        if job.thinking_guard is not None
                                        else None
                                    ),
                                    "batch_cohort": job.request.get("batch_cohort"),
                                    "skip_writing_prefix_cache": job.request.get(
                                        "skip_writing_prefix_cache", False
                                    ),
                                    "session_id": job.request.get("session_id"),
                                    "reasoning_effort": job.request.get("reasoning_effort"),
                                    "effort_semantics": getattr(adapter, "reasoning_effort_semantics", "thinking_toggle"),
                                    "context_limit": min(self.max_context, job.request.get("context_limit", self.max_context)),
                                    "structured_output": (
                                        {
                                            "kind": "grammar",
                                            "enforced": True,
                                            **structured_receipt(job.structured, job.completion_tokens),
                                        }
                                        if "grammar" in job.request
                                        else {
                                            "kind": job.request["response_format"]["type"],
                                            "enforced": True,
                                            **structured_receipt(job.structured, job.completion_tokens),
                                        }
                                        if "response_format" in job.request
                                        else {
                                            "kind": "tool_choice",
                                            "enforced": True,
                                            **structured_receipt(
                                                job.structured,
                                                job.completion_tokens,
                                            ),
                                        }
                                        if (
                                            constrained_tool_choice(job.request)
                                            and job.structured is not None
                                        )
                                        else None
                                    ),
                                    "tool_choice": (
                                        {
                                            "mode": (
                                                "named"
                                                if isinstance(
                                                    job.request.get("tool_choice"),
                                                    dict,
                                                )
                                                else job.request.get(
                                                    "tool_choice", "auto"
                                                )
                                            ),
                                            "name": (
                                                job.request["tool_choice"]["function"]["name"]
                                                if isinstance(
                                                    job.request.get("tool_choice"), dict
                                                )
                                                else None
                                            ),
                                            "strict": any(
                                                tool["function"].get("strict", False)
                                                for tool in job.request.get("tools", ())
                                            ),
                                            "parallel": job.request.get(
                                                "parallel_tool_calls", True
                                            ),
                                            "enforced": constrained_tool_choice(
                                                job.request
                                            ),
                                            "decode_grammar": job.tool_grammar_status,
                                        }
                                        if "tools" in job.request
                                        else None
                                    ),
                                },
                                "ordinary_compute_width": ordinary_compute_width(response, job.observed_width, mtp=self.mtp, external_draft=external_draft, prompt_lookup=prompt_lookup),
                                "mtp": response.mtp_receipt,
                                "speculation": getattr(response, "speculative_receipt", None),
                                "spomin_live_surgery": job.spomin_receipt,
                                "approximate_kv": job.approximate_kv_receipt,
                                **(
                                    {"int8_prefill": self.int8_prefill_handle.receipt()}
                                    if self.int8_prefill_handle is not None
                                    else {}
                                ),
                                "cache_capsule": (
                                    getattr(
                                        batch,
                                        "pop_cache_capsule_receipt",
                                        lambda _uid: None,
                                    )(response.uid)
                                    or job.cache_capsule_receipt
                                ),
                                "prompt_tokens": job.prompt_tokens,
                                "completion_tokens": job.completion_tokens,
                                "ttft_seconds": job.first_token - job.started,
                                "elapsed_seconds": time.monotonic() - job.started,
                            }
                            self.receipts.append(receipt)
                            self.counts["completed"] += 1
                            if job.structured is not None:
                                self.counts["structured_output_completed"] += 1
                            reason = (
                                "tool_calls"
                                if job.output_parser.tool_count
                                else "stop"
                                if stopped
                                else response.finish_reason
                            )
                            self._finish(
                                job, {"finish_reason": reason, "receipt": receipt}
                            )
                            del active[response.uid]
                    # Responses and prompt boundaries carry entire target and
                    # draft caches. Drop the transfer references after APC and
                    # active lanes have taken ownership, before next admission.
                    response = boundary = sidecar = None
                    prompts = responses = ()
                    job = None
                else:
                    self._service_admin_prefetch(apc)
                    self._service_pending_prefetch(apc)
                    apc.spill_idle_entries()
                now = time.monotonic()
                if now - last_snapshot > 1:
                    with self.lock:
                        self.snapshot.update(
                            {
                                "apcv2": apc.apc_stats,
                                "cache_capsules": (
                                    capsule_pool.counters
                                    if capsule_pool is not None
                                    else None
                                ),
                                "scheduler": dict(batch.scheduler_stats),
                                "segmented_self_mtp": segmented_self_mtp_stats(),
                                "metal_active_bytes": mx.get_active_memory(),
                                "metal_peak_bytes": mx.get_peak_memory(),
                                "process_physical_footprint_bytes": physical_footprint_bytes(),
                                "execution": adapter.diagnostics(),
                                "admission": dict(admission),
                                "memory_waiting": len(deferred),
                                "headroom_bytes": execution_headroom(),
                                "spomin_live_surgery": (
                                    spomin_manager.snapshot()
                                    if spomin_manager is not None
                                    else {"enabled": False, "counts": {}}
                                ),
                            }
                        )
                    last_snapshot = now
        except BaseException as exc:
            log.exception("generation worker failed")
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self.lock:
                remaining = list(self.jobs.values())
            for job in remaining:
                self._finish(
                    job, {"error": self.error or "server stopped", "status": 503}
                )
            if batch is not None:
                batch.close()
            for prepared in tuple(self.fanout_capsules.values()):
                prepared.close()
            self.fanout_capsules.clear()
            if capsule_pool is not None:
                capsule_pool.close()
            if apc is not None:
                close_apc = getattr(apc, "close", None)
                if callable(close_apc):
                    close_apc(
                        persist_resident=self.apc_persist_on_shutdown,
                        time_budget_seconds=self.apc_persist_shutdown_seconds,
                    )
                else:
                    apc.clear()
                self.apc = None
            if self.int8_prefill_handle is not None:
                from .runtime.int8_prefill import remove as remove_int8_prefill

                remove_int8_prefill(self.int8_prefill_handle)
            if adapter is not None:
                with self.prompt_lock:
                    self.adapter = None
                    adapter.close()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=30)
        from .structured_output import shutdown_scanner_pools

        shutdown_scanner_pools()
