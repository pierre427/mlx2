"""Bounded request ownership around the modern batch execution core."""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
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
from .batch_metrics import BatchFaultSpec, BatchRuntimeMetrics

log = logging.getLogger(__name__)


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
            "min_tokens",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "seed",
            "stop",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "context_limit",
            "response_format",
            "grammar",
            "batch_cohort",
            "mlx_fault",
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
        tokens = tuple(int(token) for token in tokens)
        if (
            not self.max_entries
            or not self.max_tokens
            or len(tokens) > self.max_tokens
        ):
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
    size = sum(int(getattr(row, "nbytes", 0)) for row in _walk_cache_entries(hit.cache))
    if mtp:
        size += int(hit.sidecar.nbytes)
    return size * max(1.0, context_tokens / hit.cached_tokens) / (1 << 30)


def ensure_admission_headroom(required_bytes, *, headroom, reclaim, evict):
    """Observe completed reclamation before evicting or rejecting a request."""
    if headroom() >= required_bytes:
        return True
    reclaim()
    if headroom() >= required_bytes:
        return True
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
    cached_tokens: int = 0
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


@dataclass(frozen=True)
class PublishedCohort:
    """One queue item that makes every reserved cohort member visible at once."""

    jobs: tuple[Job, ...]


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
    ):
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
        self.execution_policy = execution_policy
        if execution_policy and execution_policy.get("allow_unverified_indexed") and not qualification_mode:
            raise ValueError("unverified kernels are restricted to qualification mode")
        self.model_path = model_path
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
        self.incoming = queue.Queue(maxsize=max_inflight)
        self.slots = threading.BoundedSemaphore(max_inflight)
        self.stop_event, self.ready = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.submission_lock = threading.Lock()
        self.jobs = {}
        self.pending_cohorts = {}
        self.fanout_waiting = {}
        self.queued_jobs = 0
        self.counts = Counter()
        self.receipts = deque(maxlen=128)
        self.batch_metrics = BatchRuntimeMetrics()
        self.snapshot = {"state": "loading"}
        self.route_capabilities = frozenset()
        self.error = None
        if adapter_factory is None:
            from .adapters.registry import resolve_adapter

            adapter_factory = resolve_adapter(model_path, mtp=mtp)
        self.adapter_factory = adapter_factory
        self.thread = threading.Thread(
            target=self._run, name="mlx2-generation", daemon=True
        )
        self.thread.start()

    def submit(self, request, *, tenant_id="default"):
        if not self.ready.is_set() or self.error or not self.thread.is_alive():
            raise RuntimeError(self.error or "model is not ready")
        with self.submission_lock:
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

    def _prepare_job(self, request, *, tenant_id):
        from .contracts import Capability

        if (
            "grammar" in request or "response_format" in request
        ) and request.get("response_format") != {"type": "text"}:
            if Capability.GRAMMAR not in self.route_capabilities:
                raise ValueError("structured output is not available on this route")
        fault = BatchFaultSpec.parse(
            request.get("mlx_fault"), enabled=self.qualification_mode
        )
        public_request = {key: value for key, value in request.items() if key != "mlx_fault"}
        return Job(
            public_request, tenant_id=str(tenant_id or "default"), fault=fault
        )

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

    def _publish_jobs(self, jobs, *, already_registered=False):
        jobs = tuple(jobs)
        if not jobs:
            return
        with self.lock:
            if not already_registered:
                for job in jobs:
                    self.jobs[job.id] = job
            initial_depth = self.queued_jobs
            self.queued_jobs += len(jobs)
        item = jobs[0] if len(jobs) == 1 else PublishedCohort(jobs)
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

    def submit_many(self, requests, *, tenant_id="default"):
        """Reserve parallel samples and publish one exact-prefill leader.

        Siblings become runnable only after the leader publishes its committed
        prompt boundary into APCv2. Their ordinary APC lookup then creates
        revision-bound COW branches from that one frozen generation.
        """
        requests = list(requests)
        if not requests:
            raise ValueError("parallel submission requires at least one request")
        if not self.ready.is_set() or self.error or not self.thread.is_alive():
            raise RuntimeError(self.error or "model is not ready")
        jobs = [
            self._prepare_job(request, tenant_id=tenant_id) for request in requests
        ]
        prompt_keys = {HostPromptCache.key(job.request) for job in jobs}
        if len(prompt_keys) != 1:
            raise ValueError("parallel samples must share one rendered prompt")
        fanout_group = "fanout-" + uuid.uuid4().hex
        for index, job in enumerate(jobs):
            job.fanout_group = fanout_group
            job.fanout_role = "prefill_leader" if index == 0 else "apcv2_sibling"
        reserved = 0
        with self.submission_lock:
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
                    self.fanout_waiting[fanout_group] = tuple(jobs[1:])
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
            return {
                **self.snapshot,
                "healthy": self.ready.is_set()
                and self.thread.is_alive()
                and not self.error,
                "error": self.error,
                "inflight": len(self.jobs),
                "queue_depth": self.queued_jobs,
                "counts": dict(self.counts),
                "recent_receipts": list(self.receipts),
                "host_prompt_cache": self.host_prompt_cache.status(),
            }

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

    def admit_parallel_samples(self, count):
        """Guard opt-in fanout by lane count and measured physical headroom."""
        if type(count) is not int or count < 1:
            raise ValueError("parallel sample count must be positive")
        if count > self.max_lanes:
            raise Overloaded("parallel samples exceed configured lane capacity")
        from .memory import execution_headroom

        # Preserve the same 20 GiB hard reserve used by admission and charge a
        # conservative 4 GiB transient envelope for every concurrently active
        # sample. This reads host accounting at the request boundary, not on the
        # token hot path.
        required = (20 + 4 * count) * (1 << 30)
        if execution_headroom() < required:
            self.batch_metrics.rejected("parallel_sample_footprint", self.queued_jobs)
            raise Overloaded("parallel samples denied by physical footprint guard")
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
                    metrics.terminal(job.id, status)
        self._emit(job, event)
        if waiting_siblings:
            sibling_event = {
                "error": "parallel prefill leader ended before APCv2 fanout",
                "status": event.get("status", 503),
            }
            for sibling in waiting_siblings:
                self._finish(sibling, sibling_event)

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

    def _run(self):
        adapter = batch = apc = None
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
                    if key != "prompt_lookup"
                }
                if self.execution_policy is not None
                else None
            )
            adapter = (
                self.adapter_factory(
                    self.model_path, execution_policy=adapter_execution_policy
                )
                if adapter_execution_policy is not None
                else self.adapter_factory(self.model_path)
            )
            import mlx.core as mx
            from .memory import execution_headroom
            from .runtime.apc_v2 import APCv2, MTPAPCSidecar
            from .runtime.generate import BatchGenerator
            from .runtime.os_memory import physical_footprint_bytes
            from .runtime.sample_utils import (
                LaneRNG,
                make_transformed_logprobs,
                make_logits_processors,
                draw_key,
            )
            from .runtime.memory_policy import (
                SelfMTPLaneAdmissionController,
                _make_self_mtp_admission_callback,
            )

            identity = runtime_identity()
            self.max_context = min(self.max_context, adapter.max_context)
            settings = {
                "max_context": self.max_context,
                "max_lanes": self.max_lanes,
                "max_inflight": self.max_inflight,
                "prefill_step": self.prefill_step,
                "cache_bytes": self.cache_bytes,
                "disk_cache": bool(self.cache_dir),
                "host_prompt_cache_entries": self.host_prompt_cache.max_entries,
                "host_prompt_cache_tokens": self.host_prompt_cache.max_tokens,
                "coalesce_window_ms": self.coalesce_window_seconds * 1000,
                "batch_cohort_timeout_ms": self.batch_cohort_timeout_seconds * 1000,
                "mtp": self.mtp,
                "environment": adapter.environment,
            }
            config = adapter.execution_config(
                max_lanes=self.max_lanes, prefill_step=self.prefill_step
            )
            settings["execution_policy"] = dict(config)
            settings["prompt_lookup"] = dict(
                (self.execution_policy or {}).get("prompt_lookup", {})
            )
            external_draft = config.get("backend") == "external_draft"
            prompt_lookup = self.prompt_lookup
            settings["speculation"] = (
                "external_draft"
                if external_draft
                else "prompt_lookup"
                if prompt_lookup
                else "self_mtp"
                if self.mtp
                else "ordinary"
            )
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
            route_capabilities = (
                descriptor.capabilities if descriptor is not None else frozenset()
            )
            if prompt_lookup:
                from .contracts import Capability

                if Capability.PROMPT_LOOKUP not in route_capabilities:
                    raise ValueError(
                        "model adapter does not declare prompt-lookup execution"
                    )
            if not self.qualification_mode:
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
                    name=adapter.profile_name(self.mtp)
                    + ("-pld" if prompt_lookup else ""),
                )
                route_receipt = route.receipt
                route_capabilities = route.profile.capabilities
            # APCv2 must be able to retain at least one committed prompt
            # boundary per execution lane.  Otherwise a configured B20 server
            # with the historical 16-entry floor deterministically evicts four
            # warm prompts while the cohort is being primed, so those lanes
            # cannot compose batching with APCv2 reuse.
            apc = APCv2(
                layout_name=adapter.layout,
                max_size=max(16, self.max_lanes),
                max_bytes=self.cache_bytes,
                max_tokens=self.max_context,
                idle_disk_seconds=180 if self.cache_dir else 0,
                idle_disk_dir=self.cache_dir,
                idle_disk_max_bytes=64 << 30,
            )
            key = apc.key(
                adapter.identity["fingerprint"],
                revision=identity["source_sha256"],
                tokenizer_fingerprint=adapter.identity["fingerprint"],
                cache_layout_fingerprint=adapter.layout,
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

            def reclaim_allocator():
                # Pressure-only synchronization: retired arrays can remain in
                # flight after Python releases them. Complete work before
                # clearing allocator pages and measuring admission headroom.
                mx.synchronize()
                mx.clear_cache()

            def observe_admission(decision):
                admission.update(asdict(decision))

            def evict_unused_checkpoint():
                evicted = apc.evict_oldest_unleased()
                if evicted:
                    self.counts["memory_pressure_evictions"] += 1
                return evicted

            if external_draft:
                batch = adapter.create_external_batch(
                    completion_batch_size=self.max_lanes,
                    prefill_step_size=self.prefill_step,
                    memory_headroom=lambda: max(0, execution_headroom() - controller.hard_reserve_gib * (1 << 30)),
                    reclaim_memory=reclaim_allocator,
                    evict_checkpoint=evict_unused_checkpoint,
                    stop_tokens=[[int(t)] for t in adapter.tokenizer.eos_token_ids],
                )
            elif prompt_lookup:
                from .runtime.pld import PromptLookupBatchGenerator

                batch = PromptLookupBatchGenerator(
                    adapter.model,
                    completion_batch_size=self.max_lanes,
                    prefill_step_size=self.prefill_step,
                    prompt_lookup=(self.execution_policy or {}).get(
                        "prompt_lookup", {}
                    ),
                    stop_tokens=[
                        [int(token)] for token in adapter.tokenizer.eos_token_ids
                    ],
                )
            else:
                batch = BatchGenerator(
                    adapter.model,
                    completion_batch_size=self.max_lanes,
                    prefill_batch_size=min(2, self.max_lanes),
                    prefill_step_size=self.prefill_step,
                    prefill_batch_window=1,
                    adaptive_prefill=True,
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
                    stop_tokens=[[int(t)] for t in adapter.tokenizer.eos_token_ids],
                )
            with self.lock:
                self.route_capabilities = frozenset(route_capabilities)
                self.snapshot = {
                    "state": "ready",
                    "model": Path(self.model_path).name,
                    "runtime": identity,
                    "artifact": adapter.identity["fingerprint"],
                    "profile": adapter.profile_name(self.mtp)
                    + ("-pld" if prompt_lookup else ""),
                    "settings": settings,
                    "route_receipt": route_receipt,
                    "capabilities": sorted(
                        capability.value for capability in self.route_capabilities
                    ),
                    "qualification": "candidate"
                    if self.qualification_mode
                    else "qualified",
                    "max_context": self.max_context,
                    "max_lanes": self.max_lanes,
                    "http": {"max_request_bytes": self.max_request_bytes},
                    # Readiness is also the status contract boundary.  Publish
                    # every mechanism counter before exposing ``ready`` so an
                    # immediate qualification baseline cannot race the first
                    # periodic telemetry refresh below.
                    "apcv2": apc.apc_stats,
                    "scheduler": dict(batch.scheduler_stats),
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
                self._expire_pending_cohorts()
                # Pending requests own their warm leases, but no execution
                # lane. Cancellation/deadlines must progress even at full width.
                for _ in range(len(deferred)):
                    waiting = deferred.popleft()
                    if waiting.cancelled.is_set():
                        self._finish(waiting, {"error": "cancelled"})
                        self.counts["cancelled"] += 1
                    elif time.monotonic() >= waiting.admission_deadline:
                        self._finish(waiting, {
                            "error": "host memory admission did not recover before deadline",
                            "status": 429,
                        })
                        self.counts["memory_admission_timeouts"] += 1
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
                            if isinstance(item, PublishedCohort):
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
                            raise Overloaded("host memory admission did not recover before deadline")
                        tokens = job.admission_tokens
                        if tokens is None:
                            tokens = self.host_prompt_cache.get(job.request)
                            if tokens is None:
                                tokens = adapter.prompt_tokens(job.request)
                                self.host_prompt_cache.put(job.request, tokens)
                            job.admission_tokens = tokens
                        job.prompt_tokens = len(tokens)
                        maximum = job.request.get("max_tokens", 512)
                        context_limit = min(self.max_context, job.request.get("context_limit", self.max_context))
                        if not tokens or len(tokens) + maximum > context_limit:
                            raise ValueError(
                                f"prompt plus output must fit {context_limit} tokens"
                            )
                        # Lease resident state before pressure eviction. Disk
                        # restoration remains behind the cold allocation gate.
                        hit = job.admission_hit
                        if hit is None:
                            hit = apc.lookup(key, tokens, allow_disk_restore=False)
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
                        if not ensure_admission_headroom(
                            required * (1 << 30), headroom=execution_headroom,
                            reclaim=reclaim_allocator, evict=apc.evict_oldest_unleased,
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
                        if hit.miss_reason == "disk_restore_requires_admission":
                            hit = apc.lookup(key, tokens)
                            job.cache_branch = hit.cache
                        job.cached_tokens = hit.cached_tokens
                        job.detokenizer = adapter.tokenizer.detokenizer
                        job.detokenizer.reset()
                        job.output_parser = adapter.output_parser(job.request)
                        rng = LaneRNG(job.request.get("seed", secrets.randbits(32)))
                        temp = job.request.get("temperature", 0.7)
                        top_p, top_k = (
                            job.request.get("top_p", 0.8),
                            job.request.get("top_k", 20),
                        )
                        vocab_size = adapter.tokenizer.vocab_size
                        if top_k > vocab_size or any(int(token) >= vocab_size for token in job.request.get("logit_bias", {})):
                            raise ValueError("sampling token IDs and top_k must fit the model vocabulary")
                        transform = (
                            make_transformed_logprobs(
                                temp, top_p=top_p, top_k=top_k, min_p=job.request.get("min_p", 0.0)
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
                            repetition_penalty=(job.request.get("repetition_penalty") if job.request.get("repetition_penalty", 1.0) != 1.0 else None),
                            repetition_context_size=0,
                            presence_penalty=job.request.get("presence_penalty", 0.0),
                            presence_context_size=0,
                            frequency_penalty=job.request.get("frequency_penalty", 0.0),
                            frequency_context_size=0,
                        )
                        minimum_processor = minimum_tokens_processor(
                            mx,
                            adapter.tokenizer.eos_token_ids,
                            len(tokens),
                            job.request.get("min_tokens", 0),
                        )
                        if minimum_processor is not None:
                            processors.append(minimum_processor)
                        from .structured_output import make_structured_processor

                        structured = make_structured_processor(
                            adapter.tokenizer,
                            len(tokens),
                            response_format=job.request.get("response_format"),
                            grammar=job.request.get("grammar"),
                        )
                        if structured is not None:
                            processors.append(structured)
                        sampling_config = {"sampling_temp": temp, "top_p": top_p, "top_k": top_k, "min_p": job.request.get("min_p", 0.0), "shared_prefix_attestation": shared_prefix_attestation(hit)}
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
                        job.uid = batch.insert(
                            [hit.remaining_tokens], max_tokens=[maximum],
                            caches=[hit.cache], all_tokens=[tokens[:hit.cached_tokens]],
                            samplers=[sampler], logits_processors=[processors], lane_rngs=[rng],
                            **state_options,
                        )[0]
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
                        event = {
                            "error": str(exc),
                            "status": 429 if isinstance(exc, Overloaded) else 400,
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
                cancelled = [
                    uid for uid, job in active.items() if job.cancelled.is_set()
                ]
                if cancelled:
                    batch.remove(cancelled)
                    for uid in cancelled:
                        self._finish(active.pop(uid), {"error": "cancelled"})
                        self.counts["cancelled"] += 1
                stalled = [
                    uid
                    for uid, job in active.items()
                    if time.monotonic() - job.last_progress > 60
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
                            boundary = batch.pop_prompt_boundary(response.uid)
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
                                apc.store(
                                    key,
                                    boundary["tokens"],
                                    boundary["target_cache"],
                                    sidecar=sidecar,
                                    retention_role="committed_prompt_boundary",
                                )
                                leader = active.get(response.uid)
                                if (
                                    leader is not None
                                    and leader.fanout_role == "prefill_leader"
                                ):
                                    with self.lock:
                                        siblings = self.fanout_waiting.pop(
                                            leader.fanout_group, ()
                                        )
                                    if siblings:
                                        self._publish_jobs(
                                            siblings, already_registered=True
                                        )
                                        self.counts[
                                            "apcv2_fanout_boundaries"
                                        ] += 1
                    for response in responses:
                        job = active.get(response.uid)
                        if job is None:
                            continue
                        job.last_progress = time.monotonic()
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
                            except ValueError as exc:
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
                        except ValueError as exc:
                            batch.remove([response.uid])
                            self._finish(job, {"error": str(exc), "status": 502})
                            del active[response.uid]
                            continue
                        for delta in deltas:
                            self._emit(job, {"delta": delta})
                        stopped = job.output_parser.stopped
                        if stopped and not response.finish_reason:
                            batch.remove([response.uid])
                        if response.finish_reason or stopped:
                            sidecar = getattr(response, "cache_sidecar", None) or (
                                MTPAPCSidecar(
                                    response.mtp_state, len(response.all_tokens)
                                )
                                if response.mtp_state
                                else None
                            )
                            if response.finish_reason:
                                apc.store(
                                    key,
                                    response.all_tokens,
                                    response.prompt_cache,
                                    sidecar=sidecar,
                                )
                            receipt = {
                                "request_id": job.id,
                                "cache": "apcv2",
                                "cached_tokens": job.cached_tokens,
                                "parallel_prefill": (
                                    {
                                        "schema": "mlx2.apcv2-fanout.v1",
                                        "group": job.fanout_group,
                                        "role": job.fanout_role,
                                        "one_prefill": True,
                                    }
                                    if job.fanout_group
                                    else None
                                ),
                                "profile": self.snapshot["profile"],
                                "qualification": self.snapshot["qualification"],
                                "route_receipt": route_receipt,
                                "request_controls": {
                                    "logprobs": wants_logprobs(job.request),
                                    "top_logprobs": job.request.get("top_logprobs", 0),
                                    "logprob_semantics": (
                                        ("execution_target: external=transformed_target_verifier" if external_draft else "execution_target: ordinary=post_processor_pre_sampler; mtp=transformed_target_verifier")
                                        if wants_logprobs(job.request) else None
                                    ),
                                    "thinking": (
                                        adapter.thinking_enabled(job.request)
                                        if hasattr(adapter, "thinking_enabled")
                                        else job.request.get("enable_thinking", False)
                                    ),
                                    "sampling": {name: job.request[name] for name in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty", "logit_bias", "seed") if name in job.request},
                                    "min_tokens": job.request.get("min_tokens", 0),
                                    "batch_cohort": job.request.get("batch_cohort"),
                                    "reasoning_effort": job.request.get("reasoning_effort"),
                                    "effort_semantics": getattr(adapter, "reasoning_effort_semantics", "thinking_toggle"),
                                    "context_limit": min(self.max_context, job.request.get("context_limit", self.max_context)),
                                    "structured_output": (
                                        {"kind": "grammar", "enforced": True}
                                        if "grammar" in job.request
                                        else {
                                            "kind": job.request["response_format"]["type"],
                                            "enforced": True,
                                        }
                                        if "response_format" in job.request
                                        else None
                                    ),
                                },
                                "ordinary_compute_width": ordinary_compute_width(response, job.observed_width, mtp=self.mtp, external_draft=external_draft, prompt_lookup=prompt_lookup),
                                "mtp": response.mtp_receipt,
                                "speculation": getattr(response, "speculative_receipt", None),
                                "prompt_tokens": job.prompt_tokens,
                                "completion_tokens": job.completion_tokens,
                                "ttft_seconds": job.first_token - job.started,
                                "elapsed_seconds": time.monotonic() - job.started,
                            }
                            self.receipts.append(receipt)
                            self.counts["completed"] += 1
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
                    apc.spill_idle_entries()
                now = time.monotonic()
                if now - last_snapshot > 1:
                    with self.lock:
                        self.snapshot.update(
                            {
                                "apcv2": apc.apc_stats,
                                "scheduler": dict(batch.scheduler_stats),
                                "metal_active_bytes": mx.get_active_memory(),
                                "metal_peak_bytes": mx.get_peak_memory(),
                                "process_physical_footprint_bytes": physical_footprint_bytes(),
                                "execution": adapter.diagnostics(),
                                "admission": dict(admission),
                                "memory_waiting": len(deferred),
                                "headroom_bytes": execution_headroom(),
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
            if apc is not None:
                apc.clear()
            if adapter is not None:
                adapter.close()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=30)
