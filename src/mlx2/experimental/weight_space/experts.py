"""CPU expert masks and bounded byte residency; no model loading or GPU work.

Logical pruning retains checkpoint expert IDs. Residency only changes which
verified bytes are retained, never routing, quantization, or accumulation.
An adapter must explicitly integrate either mechanism before it affects serving.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import wraps
from types import MappingProxyType

import numpy as np

from mlx2.runtime.weight_space_metrics import (
    register_expert_metrics,
    track_weight_space,
    update_expert_metrics,
)

from .programs import _identity

_STATUS = {"implemented": True, "qualified": False, "selected": False, "observed_used": False}
_MAX_LAYERS = 256
_MAX_EXPERTS = 8192
_MAX_ROUTE_ELEMENTS = 1024 * 1024


def _integer(value, name, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value or len(value) > 1024 or any(c.isspace() for c in value):
        raise ValueError(f"{name} must be a nonempty bounded identifier")
    return value


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be an exact sha256 digest")
    return value.removeprefix("sha256:")


def _canonical(value):
    result = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(result) > 4 * 1024 * 1024:
        raise ValueError("expert plan metadata exceeds size bound")
    return result


def _content_id(value):
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class LayerExpertMask:
    """Original expert indices, not a compacted checkpoint shape."""

    layer: int
    num_experts: int
    top_k: int
    enabled: tuple[int, ...]
    normalize: bool = True
    score_scale: float = 1.0

    def __post_init__(self):
        _integer(self.layer, "layer", maximum=_MAX_LAYERS - 1)
        _integer(self.num_experts, "num_experts", 1, _MAX_EXPERTS)
        _integer(self.top_k, "top_k", 1, self.num_experts)
        enabled = tuple(self.enabled)
        for expert in enabled:
            _integer(expert, "enabled expert", maximum=self.num_experts - 1)
        if len(set(enabled)) != len(enabled) or len(enabled) < self.top_k:
            raise ValueError("enabled experts must be unique and cover router top_k")
        if type(self.normalize) is not bool:
            raise ValueError("normalize must be boolean")
        if isinstance(self.score_scale, bool) or not isinstance(self.score_scale, (int, float)) or not math.isfinite(self.score_scale) or self.score_scale <= 0:
            raise ValueError("score_scale must be finite and positive")
        object.__setattr__(self, "enabled", tuple(sorted(enabled)))
        object.__setattr__(self, "score_scale", float(self.score_scale))


@dataclass(frozen=True)
class ExpertMaskPlan:
    checkpoint: str
    source_revision: str
    source_fingerprint: str
    layers: tuple[LayerExpertMask, ...]
    quantization_profile: str
    execution_profile: str
    calibration_digest: str
    overlay_fingerprint: str = "none"
    selection_method: str = "manual_original_ids"
    score_digest: str = "none"
    tie_policy: str = "stable_lowest_original_expert_id"

    def __post_init__(self):
        _identity(self.checkpoint, self.source_revision)
        for name in ("source_fingerprint", "calibration_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("overlay_fingerprint", "score_digest"):
            value = getattr(self, name)
            if value != "none":
                object.__setattr__(self, name, _digest(value, name))
        for name in ("quantization_profile", "execution_profile", "selection_method"):
            _text(getattr(self, name), name)
        if self.tie_policy not in {"stable_lowest_original_expert_id", "native_topk_after_mask"}:
            raise ValueError("expert tie_policy must identify stable CPU or native masked top-k")
        layers = tuple(self.layers)
        if not layers or len(layers) > _MAX_LAYERS or any(not isinstance(layer, LayerExpertMask) for layer in layers):
            raise ValueError("layers must contain bounded LayerExpertMask records")
        if len({layer.layer for layer in layers}) != len(layers):
            raise ValueError("duplicate layer")
        object.__setattr__(self, "layers", tuple(sorted(layers, key=lambda layer: layer.layer)))

    def layer(self, index):
        _integer(index, "layer", maximum=_MAX_LAYERS - 1)
        for layer in self.layers:
            if layer.layer == index:
                return layer
        raise ValueError("layer is absent from the explicit expert plan")

    @property
    def approximate(self):
        return any(len(layer.enabled) != layer.num_experts for layer in self.layers)

    def _content(self):
        return {
            "schema_version": 1, "kind": "logical_expert_mask",
            "checkpoint": self.checkpoint, "source_revision": self.source_revision,
            "source_fingerprint": self.source_fingerprint,
            "layers": [asdict(layer) for layer in self.layers],
            "quantization_profile": self.quantization_profile,
            "execution_profile": self.execution_profile,
            "calibration_digest": self.calibration_digest,
            "overlay_fingerprint": self.overlay_fingerprint,
            "selection_method": self.selection_method, "score_digest": self.score_digest,
            "approximate_routing": self.approximate,
            "expert_index_policy": "original_ids_no_shape_remapping",
            "tie_policy": self.tie_policy,
            "status": dict(_STATUS),
        }

    @property
    def plan_id(self):
        return _content_id(self._content())

    def to_dict(self):
        return json.loads(_canonical({**self._content(), "plan_id": self.plan_id}))

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise TypeError("expert plan must be an object")
        names = tuple(cls.__dataclass_fields__)
        try:
            values = {key: payload[key] for key in names}
            values["layers"] = tuple(LayerExpertMask(**layer) for layer in values["layers"])
            result = cls(**values)
            if result.to_dict() != payload:
                raise ValueError("expert plan content or identity mismatch")
            return result
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid expert plan fields") from exc


@track_weight_space("expert_mask_compile")
def select_expert_mask(*, importance, keep_counts, geometries, checkpoint, source_revision,
                       source_fingerprint, calibration_digest, quantization_profile,
                       execution_profile, overlay_fingerprint="none", tie_policy="stable_lowest_original_expert_id"):
    """Keep highest supplied calibration scores per layer; never use heldout data.

    The caller supplies and labels its importance metric. This function does not
    estimate REAP activation/router criteria or claim that pruning preserves quality.
    ``geometries`` maps original layer IDs to num_experts/top_k/normalize/score_scale.
    """
    if not isinstance(importance, dict) or not importance or set(importance) != set(keep_counts) or set(importance) != set(geometries):
        raise ValueError("importance, keep_counts and geometries must name exactly the same layers")
    if len(importance) > _MAX_LAYERS:
        raise ValueError("too many layers")
    layers, score_records = [], []
    for index in sorted(importance):
        geometry = geometries[index]
        if not isinstance(geometry, dict) or set(geometry) - {"num_experts", "top_k", "normalize", "score_scale"}:
            raise ValueError("invalid layer geometry")
        count = _integer(geometry.get("num_experts"), "num_experts", 1, _MAX_EXPERTS)
        keep = _integer(keep_counts[index], "keep_count", 1, count)
        scores = np.asarray(importance[index])
        if scores.shape != (count,) or scores.dtype.kind not in "fiu" or not np.isfinite(scores).all():
            raise ValueError("importance must be finite numeric scores for every original expert")
        with np.errstate(over="ignore"):
            scores = np.asarray(scores, dtype="<f8")
        if not np.isfinite(scores).all():
            raise ValueError("importance scores overflow float64")
        enabled = tuple(int(i) for i in np.argsort(-scores, kind="stable")[:keep])
        layers.append(LayerExpertMask(index, enabled=enabled, **geometry))
        score_records.append({"layer": index, "sha256": hashlib.sha256(scores.tobytes()).hexdigest()})
    return ExpertMaskPlan(checkpoint, source_revision, source_fingerprint, tuple(layers),
                          quantization_profile, execution_profile, calibration_digest,
                          overlay_fingerprint, "supplied_calibration_importance_descending",
                          _content_id(score_records), tie_policy)


def route_experts_numpy(plan, layer, combine_scores, *, ranking_scores=None):
    """Bounded CPU routing oracle; outputs sorted ORIGINAL IDs and combine weights.

    Ranking may include a model's correction bias. Combine weights always come
    from its unmodified nonnegative scores, then follow the explicit normalization
    and scale. This does not define a model's logits-to-scores transformation.
    Native kernel tie behavior and arithmetic still need an adapter A/B.
    """
    mask = plan.layer(layer)
    if plan.tie_policy != "stable_lowest_original_expert_id":
        raise ValueError("CPU sorted-ID oracle cannot emulate native top-k tie/order policy")
    scores = np.asarray(combine_scores)
    ranking = scores if ranking_scores is None else np.asarray(ranking_scores)
    if (scores.ndim not in (1, 2) or scores.shape[-1] != mask.num_experts
            or not 0 < scores.size <= _MAX_ROUTE_ELEMENTS or ranking.shape != scores.shape
            or scores.dtype.kind not in "fiu" or ranking.dtype.kind not in "fiu"
            or not np.isfinite(scores).all() or not np.isfinite(ranking).all() or (scores < 0).any()):
        raise ValueError("route scores must be finite, bounded and match original expert geometry")
    scores, ranking = np.asarray(scores, np.float64), np.asarray(ranking, np.float64)
    enabled = np.zeros(mask.num_experts, dtype=bool)
    enabled[list(mask.enabled)] = True
    ranking = np.where(enabled, ranking, -np.inf)
    indices = np.sort(np.argsort(-ranking, axis=-1, kind="stable")[..., :mask.top_k], axis=-1)
    weights = np.take_along_axis(scores, indices, axis=-1)
    if mask.normalize:
        total = np.sum(weights, axis=-1, keepdims=True)
        if (total <= 0).any():
            raise ValueError("selected experts have zero combine mass")
        weights = weights / (total + 1e-20)
    return indices, weights * mask.score_scale


@dataclass(frozen=True, order=True)
class ExpertKey:
    layer: int
    expert: int
    projection: str = "bundle"

    def __post_init__(self):
        _integer(self.layer, "layer", maximum=_MAX_LAYERS - 1)
        _integer(self.expert, "expert", maximum=_MAX_EXPERTS - 1)
        _text(self.projection, "projection")


@dataclass(frozen=True)
class ExpertRecord:
    """Host-indexed opaque byte slice. Packing/scales remain the adapter's job."""

    key: ExpertKey
    source_id: str
    offset: int
    nbytes: int
    sha256: str
    source_fingerprint: str
    storage_profile: str = "native"

    def __post_init__(self):
        if not isinstance(self.key, ExpertKey):
            raise TypeError("key must be an ExpertKey")
        _text(self.source_id, "source_id")
        _text(self.storage_profile, "storage_profile")
        _integer(self.offset, "offset")
        _integer(self.nbytes, "nbytes", 1)
        if self.offset + self.nbytes > 2**63 - 1:
            raise ValueError("expert byte range overflows")
        for name in ("sha256", "source_fingerprint"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))


class ExpertNotReady(RuntimeError):
    """The host must pump queued IO before acquiring this expert."""


class ExpertSourceChanged(RuntimeError):
    """The bound source became stale; construct a new source-bound controller."""


class ExpertBudgetError(RuntimeError):
    """Pinned payload or minimum expert geometry exceeds the configured budget."""


@track_weight_space("expert_budget_plan")
def plan_residency_budget(*, plan, records, memory_budget_bytes, nonexpert_reserved_bytes,
                          chunk_bytes, overlay_bytes=0, maximum_resident_bytes=None):
    """Plan an application allocation cap, not a physical-memory guarantee.

    ``nonexpert_reserved_bytes`` includes dense weights, KV, scratch, draft/vision
    buffers and headroom. The caller must count both base and overlay copies when
    both remain live. Supplied sizes must reflect actual packing plus its scales.
    A batch's routed expert union can exceed the single-token bound reported here.
    """
    if not isinstance(plan, ExpertMaskPlan):
        raise TypeError("plan must be an ExpertMaskPlan")
    total = _integer(memory_budget_bytes, "memory_budget_bytes", 1)
    fixed = _integer(nonexpert_reserved_bytes, "nonexpert_reserved_bytes")
    chunk = _integer(chunk_bytes, "chunk_bytes", 1, 64 * 1024 * 1024)
    overlays = _integer(overlay_bytes, "overlay_bytes")
    if overlays and plan.overlay_fingerprint == "none":
        raise ValueError("overlay bytes require an exact overlay fingerprint")
    available = total - fixed - overlays - chunk
    if maximum_resident_bytes is not None:
        available = min(available, _integer(maximum_resident_bytes, "maximum_resident_bytes", 1))
    records = tuple(records)
    if not records or len(records) > 65536 or any(not isinstance(record, ExpertRecord) for record in records):
        raise ValueError("bounded ExpertRecord source index required")
    if len({record.key for record in records}) != len(records):
        raise ValueError("duplicate expert record")
    bundles = {}
    for record in records:
        layer = plan.layer(record.key.layer)
        if record.source_fingerprint != plan.source_fingerprint or record.key.expert >= layer.num_experts:
            raise ValueError("expert record source or geometry mismatch")
        if record.key.expert in layer.enabled:
            key = (record.key.layer, record.key.expert)
            bundles[key] = bundles.get(key, 0) + record.nbytes
    if any((layer.layer, expert) not in bundles for layer in plan.layers for expert in layer.enabled):
        raise ValueError("every enabled expert needs a complete host-indexed bundle")
    largest = max(bundles.values())
    if available < largest:
        raise ExpertBudgetError("budget cannot fit one complete expert bundle plus nonexpert/overlay/IO reserves")
    single_token = max(sum(sorted((bundles[layer.layer, expert] for expert in layer.enabled),
                                  reverse=True)[:layer.top_k]) for layer in plan.layers)
    content = {
        "plan_id": plan.plan_id, "memory_budget_bytes": total,
        "nonexpert_reserved_bytes": fixed, "overlay_bytes": overlays,
        "chunk_bytes": chunk, "payload_budget_bytes": available,
        "controller_budget_bytes": available + overlays + chunk,
        "indexed_enabled_payload_bytes": sum(bundles.values()),
        "largest_indexed_expert_bundle_bytes": largest,
        "maximum_single_token_layer_union_bytes": single_token,
        "single_token_layer_union_fits": single_token <= available,
        "all_enabled_payload_fits": sum(bundles.values()) <= available,
        "batch_union_and_execution_workspace_validated": False,
        "source_index_completeness_verified": False,
        "physical_ram_or_os_swap_measured": False,
        "source_index_digest": _content_id([asdict(record) for record in sorted(records, key=lambda r: r.key)]),
        "status": dict(_STATUS),
    }
    return {**content, "budget_plan_id": _content_id(content)}


def _observe_experts(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            update_expert_metrics(self)
    return wrapped


class ExpertLease:
    """Pins complete verified bytes; returned bounded copies belong to the caller."""

    def __init__(self, controller, key):
        self._controller, self.key, self._active = controller, key, True

    def read(self, offset=0, length=None):
        """Copy at most one configured chunk; never expose a partial expert load.

        Callers must account for retained copies/device materializations in their
        own workspace budget. The controller does not claim a process RSS bound.
        """
        return self._controller._lease_read(self, offset, length)

    def release(self):
        self._controller._release(self)

    def __enter__(self):
        if not self._active:
            raise RuntimeError("lease is released")
        return self

    def __exit__(self, *_):
        self.release()


class ExpertResidencyController:
    """Synchronous, revision-bound byte cache with cooperative IO pacing.

    ``reader(record, relative_offset, length)`` is a trusted injected byte reader;
    it must return exactly ``length`` immutable bytes and allocate no unbounded
    scratch. ``source_validator()`` returns the currently authoritative source
    SHA256 before/after every chunk and before every lease read.

    ``step`` never sleeps or creates threads. The owner schedules its returned
    next_ready_time. Demand can evict unpinned LRU data; optional prefetch cannot.
    Full payload plus one IO chunk plus fixed overlay bytes must fit. This is a
    byte-storage controller, not permission to change native expert math geometry.
    """

    def __init__(self, *, plan, records, budget_bytes, chunk_bytes, reader,
                 source_validator, overlay_bytes=0, bytes_per_second=None,
                 clock=time.monotonic, maximum_pending=128):
        if not isinstance(plan, ExpertMaskPlan):
            raise TypeError("plan must be an ExpertMaskPlan")
        self.plan = plan
        self.budget_bytes = _integer(budget_bytes, "budget_bytes", 1)
        self.chunk_bytes = _integer(chunk_bytes, "chunk_bytes", 1, 64 * 1024 * 1024)
        self.overlay_bytes = _integer(overlay_bytes, "overlay_bytes")
        if self.overlay_bytes and plan.overlay_fingerprint == "none":
            raise ValueError("overlay bytes require an exact overlay fingerprint")
        self.maximum_pending = _integer(maximum_pending, "maximum_pending", 1, 4096)
        if bytes_per_second is not None and (isinstance(bytes_per_second, bool) or not isinstance(bytes_per_second, (int, float)) or not math.isfinite(bytes_per_second) or bytes_per_second <= 0):
            raise ValueError("bytes_per_second must be finite and positive")
        if not callable(reader) or not callable(source_validator) or not callable(clock):
            raise TypeError("reader, source_validator and clock must be callable")
        records = tuple(records)
        if not records or len(records) > 65536 or any(not isinstance(r, ExpertRecord) for r in records):
            raise ValueError("bounded expert records required")
        if len({r.key for r in records}) != len(records):
            raise ValueError("duplicate expert record")
        for record in records:
            layer = plan.layer(record.key.layer)
            if record.key.expert >= layer.num_experts or record.source_fingerprint != plan.source_fingerprint:
                raise ValueError("expert record source or original geometry mismatch")
        available = {(r.key.layer, r.key.expert) for r in records}
        if any((layer.layer, expert) not in available for layer in plan.layers for expert in layer.enabled):
            raise ValueError("every enabled expert requires an indexed source record")
        self.records = MappingProxyType({r.key: r for r in records})
        self.payload_capacity = self.budget_bytes - self.chunk_bytes - self.overlay_bytes
        if self.payload_capacity < max(r.nbytes for r in records if r.key.expert in plan.layer(r.key.layer).enabled):
            raise ExpertBudgetError("budget must fit largest enabled expert, one IO chunk and overlays")
        self.bytes_per_second = bytes_per_second
        self._reader, self._source_validator, self._clock = reader, source_validator, clock
        self._lock = threading.RLock()
        self._resident, self._pending = OrderedDict(), OrderedDict()
        self._pins = {}
        self._loading = None
        self._next_read_at = 0.0
        self._poisoned, self._closed = False, False
        self._counts = {name: 0 for name in (
            "requests", "hits", "faults", "evictions", "chunks", "streamed_bytes",
            "pacing_waits", "cancelled_bytes", "prefetch_requests", "prefetch_skips",
        )}
        self._peak_payload_bytes = 0
        self.controller_id = _content_id({
            "plan_id": plan.plan_id, "records": [asdict(r) for r in sorted(records, key=lambda r: r.key)],
            "budget_bytes": budget_bytes, "chunk_bytes": chunk_bytes,
            "overlay_bytes": overlay_bytes, "bytes_per_second": bytes_per_second,
            "maximum_pending": maximum_pending, "policy": "pinned_lru_demand_only_eviction_v1",
        })
        self._check_source()
        register_expert_metrics(self)

    def _check_source(self):
        if self._closed:
            raise RuntimeError("expert controller is closed")
        if self._poisoned:
            raise ExpertSourceChanged("expert source is stale")
        try:
            valid = _digest(self._source_validator(), "authoritative source") == self.plan.source_fingerprint
        except Exception as exc:
            self._invalidate()
            raise ExpertSourceChanged("could not validate expert source") from exc
        if not valid:
            self._invalidate()
            raise ExpertSourceChanged("expert source fingerprint changed")

    def _invalidate(self):
        self._poisoned = True
        self._pending.clear()
        if self._loading:
            self._counts["cancelled_bytes"] += self._loading["position"]
        self._loading = None
        for key in list(self._resident):
            if not self._pins.get(key):
                del self._resident[key]

    def _record(self, key):
        if not isinstance(key, ExpertKey) or key not in self.records:
            raise ValueError("expert is not indexed by this controller")
        if key.expert not in self.plan.layer(key.layer).enabled:
            raise ValueError("expert is disabled by logical mask")
        return self.records[key]

    def _resident_bytes(self):
        return sum(len(value) for value in self._resident.values())

    @_observe_experts
    def request(self, key, *, prefetch=False):
        with self._lock:
            self._check_source()
            self._record(key)
            if type(prefetch) is not bool:
                raise ValueError("prefetch must be boolean")
            self._counts["prefetch_requests" if prefetch else "requests"] += 1
            if key in self._resident:
                self._counts["hits"] += 1
                if not prefetch:
                    self._resident.move_to_end(key)
                return "ready"
            if self._loading and self._loading["key"] == key:
                if not prefetch:
                    self._loading["prefetch"] = False
                return "loading"
            if key in self._pending:
                self._pending[key] = self._pending[key] and prefetch
                return "queued"
            if len(self._pending) >= self.maximum_pending:
                raise ExpertBudgetError("pending expert queue bound exceeded")
            if not prefetch and self._loading and self._loading["prefetch"]:
                # A speculative partial read must not hold demand admission hostage.
                self._counts["cancelled_bytes"] += self._loading["position"]
                self._loading = None
            self._pending[key] = prefetch
            return "queued"

    def prefetch(self, key):
        return self.request(key, prefetch=True)

    def _start(self, key, prefetch):
        record = self.records[key]
        required = self._resident_bytes() + record.nbytes - self.payload_capacity
        victims = []
        if required > 0:
            if prefetch:
                return False
            for candidate in self._resident:
                if not self._pins.get(candidate):
                    victims.append(candidate)
                    required -= self.records[candidate].nbytes
                    if required <= 0:
                        break
            if required > 0:
                return False
        for victim in victims:
            del self._resident[victim]
            self._counts["evictions"] += 1
        # Allocation is private until EVERY source byte and its digest validate.
        self._loading = {"key": key, "prefetch": prefetch, "data": bytearray(record.nbytes),
                             "position": 0, "digest": hashlib.sha256()}
        self._peak_payload_bytes = max(self._peak_payload_bytes, self._resident_bytes() + record.nbytes)
        return True

    @_observe_experts
    def step(self, *, max_chunks=1):
        _integer(max_chunks, "max_chunks", 1, 1024)
        with self._lock:
            self._check_source()
            result = {"chunks": 0, "bytes": 0, "completed": [], "skipped": [], "blocked": None, "next_ready_time": None}
            for _ in range(max_chunks):
                if not self._loading:
                    if not self._pending:
                        break
                    key = min(self._pending, key=lambda candidate: self._pending[candidate])
                    prefetch = self._pending[key]
                    if not self._start(key, prefetch):
                        if prefetch:
                            del self._pending[key]
                            self._counts["prefetch_skips"] += 1
                            result["skipped"].append(asdict(key))
                            result["blocked"] = "prefetch_budget"
                            continue
                        result["blocked"] = "pinned_budget"
                        break
                    del self._pending[key]
                now = float(self._clock())
                if not math.isfinite(now):
                    raise ValueError("clock must return a finite monotonic time")
                if self.bytes_per_second and now < self._next_read_at:
                    self._counts["pacing_waits"] += 1
                    result.update(blocked="pacing", next_ready_time=self._next_read_at)
                    break
                load = self._loading
                record = self.records[load["key"]]
                position = load["position"]
                length = min(self.chunk_bytes, record.nbytes - position)
                try:
                    self._check_source()
                    chunk = self._reader(record, position, length)
                    if type(chunk) is not bytes or len(chunk) != length:
                        raise OSError("reader returned incomplete or non-bytes expert data")
                    self._check_source()
                    load["data"][position:position + length] = chunk
                    load["digest"].update(chunk)
                    del chunk
                    load["position"] += length
                    self._counts["chunks"] += 1
                    self._counts["streamed_bytes"] += length
                    result["chunks"] += 1
                    result["bytes"] += length
                    if self.bytes_per_second:
                        self._next_read_at = max(now, self._next_read_at) + length / self.bytes_per_second
                    if load["position"] == record.nbytes:
                        if load["digest"].hexdigest() != record.sha256:
                            raise OSError("expert byte content digest mismatch")
                        self._resident[record.key] = load["data"]
                        self._loading = None
                        result["completed"].append(asdict(record.key))
                except Exception:
                    self._counts["faults"] += 1
                    if self._loading:
                        self._counts["cancelled_bytes"] += self._loading["position"]
                    self._loading = None
                    raise
                del load
            return result

    @_observe_experts
    def acquire(self, key):
        with self._lock:
            self._check_source()
            self._record(key)
            if key not in self._resident:
                raise ExpertNotReady("expert is not fully read and verified; request then step")
            self._pins[key] = self._pins.get(key, 0) + 1
            self._resident.move_to_end(key)
            return ExpertLease(self, key)

    def _lease_read(self, lease, offset, length):
        with self._lock:
            self._check_source()
            if not lease._active or lease._controller is not self:
                raise RuntimeError("lease is released or foreign")
            data = self._resident[lease.key]
            _integer(offset, "offset", maximum=len(data) - 1)
            length = min(self.chunk_bytes, len(data) - offset) if length is None else length
            _integer(length, "length", 1, min(self.chunk_bytes, len(data) - offset))
            return bytes(memoryview(data)[offset:offset + length])

    @_observe_experts
    def _release(self, lease):
        with self._lock:
            if lease._controller is not self:
                raise RuntimeError("foreign lease")
            if not lease._active:
                return
            lease._active = False
            self._pins[lease.key] -= 1
            if self._pins[lease.key] == 0:
                del self._pins[lease.key]
                if self._poisoned:
                    self._resident.pop(lease.key, None)

    @_observe_experts
    def cancel(self, key):
        with self._lock:
            self._record(key)
            removed = self._pending.pop(key, None) is not None
            if self._loading and self._loading["key"] == key:
                self._counts["cancelled_bytes"] += self._loading["position"]
                self._loading = None
                removed = True
            return removed

    @_observe_experts
    def evict(self, key):
        with self._lock:
            self._check_source()
            self._record(key)
            if self._pins.get(key):
                raise ExpertBudgetError("cannot evict a pinned expert")
            if key in self._resident:
                del self._resident[key]
                self._counts["evictions"] += 1
                return True
            return False

    def metrics_snapshot(self):
        """Atomic low-cardinality counters/gauges; no artifact or expert labels."""
        with self._lock:
            resident = self._resident_bytes()
            loading = len(self._loading["data"]) if self._loading else 0
            return {
                **self._counts, "resident_bytes": resident, "loading_bytes": loading,
                "pinned_bytes": sum(len(self._resident[key]) for key in self._pins),
                "overlay_bytes": self.overlay_bytes,
                "reserved_bytes": resident + loading + self.overlay_bytes + self.chunk_bytes,
                "budget_bytes": self.budget_bytes, "resident_experts": len(self._resident),
                "pending_experts": len(self._pending) + bool(self._loading),
                "leases": sum(self._pins.values()), "peak_payload_bytes": self._peak_payload_bytes,
            }

    def receipt(self):
        with self._lock:
            return {
                "controller_id": self.controller_id, "plan_id": self.plan.plan_id,
                "checkpoint": self.plan.checkpoint, "source_revision": self.plan.source_revision,
                "source_fingerprint": self.plan.source_fingerprint,
                "quantization_profile": self.plan.quantization_profile,
                "overlay_fingerprint": self.plan.overlay_fingerprint,
                "status": dict(_STATUS), "source_stale": self._poisoned,
                "accounting_scope": "controller_owned_payload_plus_one_io_chunk_and_fixed_overlay_charge",
                "physical_ram_or_os_swap_measured": False,
                "native_arithmetic_or_gpu_qualified": False,
                "metrics": self.metrics_snapshot(),
            }

    @_observe_experts
    def close(self):
        with self._lock:
            if self._pins:
                raise ExpertBudgetError("release active expert leases before closing")
            self._resident.clear()
            self._pending.clear()
            if self._loading:
                self._counts["cancelled_bytes"] += self._loading["position"]
            self._loading = None
            self._closed = True
