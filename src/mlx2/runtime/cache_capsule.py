# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
"""Generation-safe APCv2 B1 to Bn cache materialization.

Capsules are transient physical fanout products, never a second cache engine.
They remain valid only while the exact APC compatibility signature and
generation captured at lookup remain current.
"""
from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from .models.cache import BatchKVCache, KVCache


class CacheCapsuleError(RuntimeError):
    pass


class CacheCapsuleDisabled(CacheCapsuleError):
    pass


class CacheCapsuleUnsupported(CacheCapsuleError):
    pass


class StaleCacheCapsule(CacheCapsuleError):
    pass


class CacheCapsuleDeadline(CacheCapsuleError):
    pass


class CacheCapsuleOwnerReleased(CacheCapsuleError):
    pass


def cache_capsules_enabled() -> bool:
    return os.environ.get("MLX2_CACHE_CAPSULE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


class CacheCapsuleGeneration:
    """Small generation authority shared by APC invalidation and workers."""

    def __init__(self, initial: int = 0):
        self._value = int(initial)
        self._lock = threading.Lock()

    @property
    def current(self) -> int:
        with self._lock:
            return self._value

    def advance(self) -> int:
        with self._lock:
            self._value += 1
            return self._value


@dataclass(frozen=True)
class KVCachePlaneSource:
    generation: int
    source_id: str
    compatibility_signature: Tuple[Any, ...]
    keys: Any
    values: Any
    offset: int
    target_batch: int
    layout_fingerprint: Tuple[Any, ...]
    creator_thread: int
    raw_digest: Optional[str] = None

    @property
    def required_bytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes) * self.target_batch


@dataclass(frozen=True)
class KVCacheCapsulePayload:
    keys: Any
    values: Any
    offset: int
    source_generation: int
    source_id: str
    compatibility_signature: Tuple[Any, ...]
    layout_fingerprint: Tuple[Any, ...]
    backend: str
    raw_digest: Optional[str] = None


@dataclass(frozen=True)
class CacheCapsuleProduct:
    payload: KVCacheCapsulePayload
    backing_owner: Any = None


@dataclass(frozen=True)
class CacheCapsuleReceipt:
    owner: "CacheCapsuleOwner"
    backend: str
    fallback_reason: Optional[str]


def inspect_kv_cache_capsule(cache: Any, target_batch: int = 2) -> tuple[bool, str | None]:
    if type(cache) is not KVCache:
        return False, "plain_kv_only"
    if cache.keys is None or cache.values is None:
        return False, "empty_cache"
    if len(cache.keys.shape) != 4 or cache.keys.shape != cache.values.shape:
        return False, "key_value_geometry_mismatch"
    if cache.keys.shape[0] != 1:
        return False, "source_batch_must_be_one"
    if int(target_batch) < 2:
        return False, "target_batch_must_exceed_one"
    if not 0 <= int(cache.offset) <= int(cache.keys.shape[2]):
        return False, "invalid_offset"
    if cache.keys.dtype != cache.values.dtype or cache.keys.dtype not in (mx.bfloat16, mx.float16):
        return False, "bf16_or_fp16_only"
    return True, None


def _raw_array(value):
    return np.asarray(value.view(mx.uint16) if value.dtype == mx.bfloat16 else value)


def _raw_digest(keys, values, repeats=1):
    digest = hashlib.sha256()
    for value in (keys, values):
        raw = _raw_array(value)
        if repeats != 1:
            raw = np.repeat(raw, repeats, axis=0)
        digest.update(str(raw.dtype).encode())
        digest.update(str(tuple(raw.shape)).encode())
        digest.update(raw.tobytes(order="C"))
    return digest.hexdigest()


def capture_kv_cache_plane(cache: KVCache, *, generation: int, source_id: str,
                           compatibility_signature: Sequence[Any], target_batch: int = 2,
                           verify_raw_bits: bool = False) -> KVCachePlaneSource:
    supported, reason = inspect_kv_cache_capsule(cache, target_batch)
    if not supported:
        raise CacheCapsuleUnsupported(reason or "unsupported")
    signature = tuple(compatibility_signature)
    if not signature or any(value is None for value in signature):
        raise CacheCapsuleError("exact compatibility signature is required")
    keys, values = mx.stop_gradient(cache.keys), mx.stop_gradient(cache.values)
    layout = (tuple(keys.shape), tuple(values.shape), str(keys.dtype), int(cache.offset),
              int(target_batch), signature)
    return KVCachePlaneSource(
        int(generation), str(source_id), signature, keys, values, int(cache.offset),
        int(target_batch), layout, threading.get_ident(),
        _raw_digest(keys, values) if verify_raw_bits else None,
    )


def _repeat_raw(value, repeats):
    raw = np.repeat(_raw_array(value), int(repeats), axis=0)
    result = mx.array(raw)
    return result.view(mx.bfloat16) if value.dtype == mx.bfloat16 else result.astype(value.dtype)


def _make_product(source, keys, values, backend):
    return CacheCapsuleProduct(KVCacheCapsulePayload(
        keys, values, source.offset, source.generation, source.source_id,
        source.compatibility_signature, source.layout_fingerprint, backend,
        _raw_digest(keys, values) if source.raw_digest is not None else None,
    ))


def build_kv_cache_capsule_cpu(source):
    return _make_product(source, _repeat_raw(source.keys, source.target_batch),
                         _repeat_raw(source.values, source.target_batch), "cpu")


def build_kv_cache_capsule_gpu(source):
    return _make_product(source, mx.concatenate([source.keys] * source.target_batch, axis=0),
                         mx.concatenate([source.values] * source.target_batch, axis=0), "gpu")


def _release(value):
    if value is None:
        return
    value = value.backing_owner if isinstance(value, CacheCapsuleProduct) else value
    callback = getattr(value, "release", None)
    if callable(callback):
        callback()


class _PreparedCapacityReservation:
    """Unforgeable-in-practice proof that a whole prepared batch is charged."""

    def __init__(self, pool, generation, required_bytes, backing):
        self.pool = pool
        self.generation = int(generation)
        self.required_bytes = int(required_bytes)
        self._backing = backing
        self._released = False
        self._remaining_bytes = int(required_bytes)
        self._lock = threading.Lock()

    def validate(self, pool, generation):
        with self._lock:
            if self._released or self.pool is not pool:
                raise CacheCapsuleError("prepared capacity reservation is invalid")
            if self.generation != int(generation):
                raise StaleCacheCapsule("prepared capacity generation changed")

    def claim(self, pool, generation, nbytes):
        nbytes = int(nbytes)
        with self._lock:
            if self._released or self.pool is not pool:
                raise CacheCapsuleError("prepared capacity reservation is invalid")
            if self.generation != int(generation):
                raise StaleCacheCapsule("prepared capacity generation changed")
            if nbytes < 0 or nbytes > self._remaining_bytes:
                raise CacheCapsuleError(
                    "prepared footprint exceeds capacity reservation"
                )
            self._remaining_bytes -= nbytes

    def require_fully_claimed(self):
        with self._lock:
            if self._remaining_bytes:
                raise CacheCapsuleError(
                    "prepared footprint does not match capacity reservation"
                )

    def release(self):
        with self._lock:
            if self._released:
                return
            self._released = True
            backing, self._backing = self._backing, None
        _release(backing)


def _contains_mlx_array(value, seen=None):
    """Return true unless ``value`` is a recursively host-only safe value.

    This is deliberately an allowlist. Treating an arbitrary wrapper as safe
    would let an adapter hide MLX state inside the worker input.
    """
    if isinstance(value, mx.array):
        return True
    if value is None or isinstance(
        value, (str, bytes, bytearray, memoryview, int, float, bool, np.generic)
    ):
        return False
    if isinstance(value, np.ndarray):
        if not value.dtype.hasobject:
            return False
        return any(_contains_mlx_array(item, seen) for item in value.flat)
    seen = set() if seen is None else seen
    if id(value) in seen:
        return False
    seen.add(id(value))
    if is_dataclass(value):
        return any(_contains_mlx_array(getattr(value, item.name), seen) for item in fields(value))
    if isinstance(value, dict):
        return any(_contains_mlx_array(item, seen) for pair in value.items() for item in pair)
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_contains_mlx_array(item, seen) for item in value)
    return True


class CacheCapsuleOwner:
    """Own backing and a capacity reservation through the final consumer lease."""

    def __init__(self, product, generation, source, reservation=None):
        self._product, self._generation, self._source = product, generation, source
        self._reservation, self._leases = reservation, 0
        self._release_requested = False
        self._lock = threading.Lock()

    def _retire(self):
        product, reservation = self._product, self._reservation
        self._product = self._reservation = None
        self._source = None
        _release(product); _release(reservation)

    def _assert_current(self):
        if self._source is None:
            raise CacheCapsuleOwnerReleased("capsule backing was released")
        if self._generation.current != self._source.generation:
            self._release_requested = True
            if self._leases == 0:
                self._retire()
            raise StaleCacheCapsule("capsule generation is stale")

    def lease(self):
        with self._lock:
            self._assert_current()
            if threading.get_ident() != self._source.creator_thread:
                raise CacheCapsuleError("capsule lease must use source thread")
            if self._product is None or self._release_requested:
                raise CacheCapsuleOwnerReleased("capsule owner was released")
            self._leases += 1
        return CacheCapsuleLease(self)

    def payload(self):
        with self._lock:
            self._assert_current()
            if self._product is None:
                raise CacheCapsuleOwnerReleased("capsule backing was released")
            return self._product.payload

    def release(self):
        with self._lock:
            self._release_requested = True
            if self._leases == 0:
                self._retire()

    def _close_lease(self):
        with self._lock:
            if self._leases <= 0:
                raise RuntimeError("capsule lease underflow")
            self._leases -= 1
            if self._release_requested and self._leases == 0:
                self._retire()

    @property
    def released(self):
        with self._lock:
            return self._product is None


class CacheCapsuleLease:
    def __init__(self, owner):
        self.owner, self.closed, self.synchronized = owner, False, False
        self._lock = threading.Lock()

    def payload_for_consumer(self, synchronize):
        with self._lock:
            if self.closed:
                raise CacheCapsuleOwnerReleased("capsule lease is closed")
            payload = self.owner.payload()
            if not self.synchronized:
                synchronize(payload); self.synchronized = True
            return payload

    def restore_batch_kv_cache(self, synchronize):
        payload = self.payload_for_consumer(synchronize)
        batch = int(payload.keys.shape[0])
        cache = BatchKVCache([0] * batch)
        cache.keys, cache.values = payload.keys, payload.values
        cache.offset = mx.array([payload.offset] * batch)
        cache.left_padding = mx.zeros((batch,), dtype=mx.int32)
        cache._idx = payload.offset
        return cache

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.closed = True
        self.owner._close_lease()


class CacheCapsuleTicket:
    """Off-thread external build; late results are always disposed."""

    def __init__(self, pool, source, staged, future, prepared_reservation=None):
        self.pool, self.source, self.staged, self.future = pool, source, staged, future
        self.prepared_reservation = prepared_reservation
        self.state, self.disposed = "pending", False
        self._lock = threading.Lock()

    def await_adopt(self, timeout_s=None, fallback="gpu"):
        return self.pool._await(self, timeout_s, fallback)

    def dispose_completed(self):
        with self._lock:
            if (
                self.state != "cancelled" or self.disposed
                or not self.future.done() or self.future.cancelled()
            ):
                return False
            try:
                product = self.future.result()
            except BaseException:
                return False
            self.disposed = True
        self.pool._discard(product)
        return True

    def cancel(self, reason="cancelled"):
        with self._lock:
            if self.state != "pending":
                return False
            self.state = "cancelled"
        self.dispose_completed()
        abort = getattr(self.pool.adapter, "abort", None)
        if callable(abort):
            threading.Thread(target=abort, args=(self.staged, reason), daemon=True).start()
        self.pool._terminal(self)
        return True


class CacheCapsulePool:
    """Bounded one-worker capsule lane with deadline fallback and receipts."""

    _COUNTERS = (
        "requests", "primary_successes", "fallbacks", "timeouts", "errors",
        "stale", "capacity_rejections", "late_disposals",
    )

    def __init__(self, generation, *, adapter=None, enabled=None,
                 verify_raw_bits=False, reserve=None):
        self.generation, self.adapter = generation, adapter
        self.enabled = cache_capsules_enabled() if enabled is None else bool(enabled)
        self.verify_raw_bits, self.reserve = bool(verify_raw_bits), reserve
        self._counts = {key: 0 for key in self._COUNTERS}
        self._count_lock, self._ticket_lock = threading.Lock(), threading.Lock()
        self._tickets, self._active, self._closed = set(), None, False

    @property
    def counters(self) -> Dict[str, int]:
        with self._count_lock:
            return dict(self._counts)

    def _count(self, key):
        with self._count_lock:
            self._counts[key] += 1

    def reserve_prepared(self, generation, required_bytes):
        """Charge one complete prepared batch before any materialization."""
        generation = int(generation)
        if generation != self.generation.current:
            self._count("stale")
            raise StaleCacheCapsule("source generation is stale")
        backing = self.reserve(int(required_bytes)) if self.reserve else None
        if self.reserve and backing is None:
            self._count("capacity_rejections")
            raise CacheCapsuleError("capacity reservation rejected")
        token = _PreparedCapacityReservation(
            self, generation, required_bytes, backing
        )
        if generation != self.generation.current:
            self._count("stale")
            token.release()
            raise StaleCacheCapsule("source generation is stale")
        return token

    def _check(self, source):
        if source.generation != self.generation.current:
            self._count("stale")
            raise StaleCacheCapsule("source generation is stale")

    def _build(self, backend, source):
        self._check(source)
        if backend == "cpu":
            return build_kv_cache_capsule_cpu(source)
        if backend == "gpu":
            return build_kv_cache_capsule_gpu(source)
        raise CacheCapsuleUnsupported(f"unknown_backend:{backend}")

    def _validate(self, product, source, backend):
        if not isinstance(product, CacheCapsuleProduct):
            raise CacheCapsuleError("backend returned invalid product")
        payload = product.payload
        if payload.backend != backend or payload.source_id != source.source_id:
            raise CacheCapsuleError("backend changed cache identity")
        if payload.source_generation != source.generation:
            raise StaleCacheCapsule("backend returned another generation")
        if payload.compatibility_signature != source.compatibility_signature:
            raise CacheCapsuleError("backend changed compatibility signature")
        if payload.layout_fingerprint != source.layout_fingerprint:
            raise CacheCapsuleError("backend changed layout fingerprint")
        if payload.offset != source.offset:
            raise CacheCapsuleError("backend changed cache offset")
        for output, original in ((payload.keys, source.keys), (payload.values, source.values)):
            if tuple(output.shape[1:]) != tuple(original.shape[1:]) or int(output.shape[0]) != source.target_batch:
                raise CacheCapsuleError("backend changed output geometry")
            if output.dtype != original.dtype:
                raise CacheCapsuleError("backend changed dtype")
        if self.verify_raw_bits:
            expected = _raw_digest(source.keys, source.values, source.target_batch)
            if (
                source.raw_digest is None or payload.raw_digest != expected
                or _raw_digest(payload.keys, payload.values) != expected
            ):
                raise CacheCapsuleError("raw-bit checksum mismatch")

    def _accept(
        self, product, source, backend, reason, *, prepared_reservation=None
    ):
        reservation = None
        try:
            self._check(source); self._validate(product, source, backend)
            if prepared_reservation is not None:
                prepared_reservation.validate(self, source.generation)
            reservation = (
                self.reserve(source.required_bytes)
                if self.reserve and prepared_reservation is None
                else None
            )
            if self.reserve and prepared_reservation is None and reservation is None:
                self._count("capacity_rejections")
                raise CacheCapsuleError("capacity reservation rejected")
            # Reservation may reclaim APC entries, so generation must be checked twice.
            self._check(source)
            # Closing may race with an external build, adoption, or a fallback
            # build after its ticket was cancelled. Serialize final ownership
            # publication with close, then let the caller own accepted leases.
            with self._ticket_lock:
                if self._closed:
                    raise CacheCapsuleError("capsule pool is closed")
                return CacheCapsuleReceipt(
                    CacheCapsuleOwner(product, self.generation, source, reservation),
                    backend, reason,
                )
        except BaseException:
            _release(reservation); _release(product)
            raise

    def prepare(
        self, source, *, primary, fallback="gpu", timeout_s=None,
        prepared_reservation=None,
    ):
        with self._ticket_lock:
            if self._closed:
                raise CacheCapsuleError("capsule pool is closed")
        if prepared_reservation is not None:
            prepared_reservation.claim(
                self, source.generation, source.required_bytes
            )
        if primary == "external":
            return self.submit(
                source, prepared_reservation=prepared_reservation
            ).await_adopt(timeout_s, fallback)
        if not self.enabled:
            raise CacheCapsuleDisabled("cache capsules are disabled")
        self._count("requests")
        receipt = self._accept(
            self._build(primary, source), source, primary, None,
            prepared_reservation=prepared_reservation,
        )
        self._count("primary_successes")
        return receipt

    def submit(self, source, *, prepared_reservation=None):
        if not self.enabled:
            raise CacheCapsuleDisabled("cache capsules are disabled")
        if threading.get_ident() != source.creator_thread:
            raise CacheCapsuleError("submission must use source thread")
        self._check(source)
        if self.adapter is None:
            raise CacheCapsuleUnsupported("external_adapter_unavailable")
        with self._ticket_lock:
            if self._closed:
                raise CacheCapsuleError("capsule pool is closed")
            if self._active is not None:
                raise CacheCapsuleUnsupported("capsule_circuit_busy")
            staged = self.adapter.stage(source)
            if _contains_mlx_array(staged):
                raise CacheCapsuleError(
                    "external stage must copy into host-only buffers before worker build"
                )
            future = Future()
            ticket = CacheCapsuleTicket(
                self, source, staged, future, prepared_reservation
            )
            self._active = ticket
            self._tickets.add(ticket)

        def run():
            try:
                future.set_result(self.adapter.build(staged))
            except BaseException as error:
                future.set_exception(error)

        def finished(_future):
            with ticket._lock:
                late = ticket.state == "cancelled"
            if late and ticket.dispose_completed():
                self._count("late_disposals")
            with self._ticket_lock:
                if self._active is ticket:
                    self._active = None

        future.add_done_callback(finished)
        threading.Thread(target=run, name="mlx2-cache-capsule", daemon=True).start()
        self._count("requests")
        return ticket

    def _await(self, ticket, timeout_s, fallback):
        with ticket._lock:
            if ticket.state != "pending":
                raise CacheCapsuleError("capsule ticket is no longer pending")
        if threading.get_ident() != ticket.source.creator_thread:
            ticket.cancel("wrong_thread")
            raise CacheCapsuleError("adoption must use source thread")
        try:
            self._check(ticket.source)
        except StaleCacheCapsule:
            ticket.cancel("stale_before_await")
            raise
        try:
            built = ticket.future.result(timeout=timeout_s)
        except TimeoutError as error:
            self._count("timeouts")
            if not ticket.cancel("deadline"):
                raise CacheCapsuleError("capsule ticket is no longer pending") from error
            if fallback is None:
                raise CacheCapsuleDeadline("external_timeout") from error
            self._count("fallbacks")
            return self._accept(self._build(fallback, ticket.source), ticket.source,
                                fallback, "external_timeout",
                                prepared_reservation=ticket.prepared_reservation)
        except Exception as error:
            self._count("errors")
            if not ticket.cancel("build_error"):
                raise CacheCapsuleError("capsule ticket is no longer pending") from error
            if fallback is None:
                raise CacheCapsuleError("external_error") from error
            self._count("fallbacks")
            return self._accept(self._build(fallback, ticket.source), ticket.source,
                                fallback, f"external_error:{type(error).__name__}",
                                prepared_reservation=ticket.prepared_reservation)
        try:
            self._check(ticket.source)
        except StaleCacheCapsule:
            # Dispose a completed host product now, or let the completion
            # callback dispose it later, and always terminalize the ticket.
            ticket.cancel("stale_before_adopt")
            raise
        # Transfer exclusive ownership before calling the adapter. Cancellation
        # may have disposed the host result while the future was being awaited;
        # after this point it must leave disposal to the adoption path.
        with ticket._lock:
            if ticket.state != "pending":
                raise CacheCapsuleError("capsule ticket is no longer pending")
            ticket.state = "adopting"
        try:
            product = self.adapter.adopt(built, ticket.source)
        except BaseException:
            with ticket._lock:
                ticket.state = "cancelled"
                ticket.disposed = True
            self._discard(built)
            self._terminal(ticket)
            self._count("errors")
            raise
        try:
            receipt = self._accept(
                product, ticket.source, "external", None,
                prepared_reservation=ticket.prepared_reservation,
            )
        except BaseException:
            # _accept owns and releases the adopted product on every failure.
            # Mark the ticket terminal without discarding ``built`` again: an
            # adapter may return the same external owner from adopt.
            with ticket._lock:
                ticket.state = "cancelled"
                ticket.disposed = True
            self._terminal(ticket)
            self._count("errors")
            raise
        with ticket._lock:
            ticket.state = "adopted"
            ticket.disposed = True
        self._terminal(ticket); self._count("primary_successes")
        return receipt

    def _discard(self, product):
        callback = getattr(self.adapter, "discard", None)
        callback(product) if callable(callback) else _release(product)

    def _terminal(self, ticket):
        with self._ticket_lock:
            self._tickets.discard(ticket)

    def close(self):
        with self._ticket_lock:
            self._closed = True
            tickets = tuple(self._tickets)
        for ticket in tickets:
            ticket.cancel("pool_close")


class PreparedPromptCacheCapsules:
    def __init__(
        self, prompt_cache, receipts, leases, ordinary_planes, reservation=None
    ):
        self.prompt_cache, self.receipts, self.leases = prompt_cache, tuple(receipts), tuple(leases)
        self.ordinary_planes = int(ordinary_planes)
        self._reservation, self._closed = reservation, False

    def close(self, synchronize=True):
        if self._closed:
            return
        self._closed = True
        error = None
        try:
            if synchronize:
                mx.synchronize()
        except BaseException as caught:
            error = caught
        finally:
            # Drop this owner's direct references before returning the capacity
            # reservation. Any surviving arrays must belong to a consumer
            # whose lease/lane has not yet reached this close boundary.
            self.prompt_cache = ()
            for lease in reversed(self.leases): lease.close()
            for receipt in reversed(self.receipts): receipt.owner.release()
            _release(self._reservation)
            self._reservation = None
        if error is not None:
            raise error


def prepare_prompt_cache_capsules(prompt_cache, *, target_batch, generation, pool,
                                  compatibility_signature, backend="gpu", fallback="gpu",
                                  timeout_s=None, source_prefix="apc", synchronize=None):
    capabilities = [inspect_kv_cache_capsule(cache, target_batch)[0] for cache in prompt_cache]
    if not any(capabilities):
        return None
    synchronize = synchronize or (lambda payload: mx.eval(payload.keys, payload.values))
    target_batch = int(target_batch)
    # Reserve the complete physical Bn result before any plane is built or
    # merged. Per-plane owners then skip their normal reservation so capsule
    # planes are not charged twice; the group reservation remains held until
    # the final consumer closes the prepared owner.
    required_bytes = 0
    for cache, supported in zip(prompt_cache, capabilities):
        if supported:
            # This is the exact payload built by both capsule backends.
            plane_bytes = int(cache.keys.nbytes + cache.values.nbytes)
        else:
            nbytes = getattr(cache, "nbytes", None)
            if nbytes is None:
                raise CacheCapsuleUnsupported(
                    f"unknown_plane_footprint:{type(cache).__name__}"
                )
            plane_bytes = int(nbytes)
        required_bytes += plane_bytes * target_batch
    reservation = pool.reserve_prepared(generation, required_bytes)
    batched, receipts, leases, ordinary = [], [], [], 0
    try:
        for index, (cache, supported) in enumerate(zip(prompt_cache, capabilities)):
            if supported:
                source = capture_kv_cache_plane(
                    cache, generation=generation, source_id=f"{source_prefix}:plane:{index}",
                    compatibility_signature=compatibility_signature, target_batch=target_batch,
                    verify_raw_bits=pool.verify_raw_bits,
                )
                receipt = pool.prepare(
                    source, primary=backend, fallback=fallback,
                    timeout_s=timeout_s, prepared_reservation=reservation,
                )
                lease = receipt.owner.lease()
                receipts.append(receipt); leases.append(lease)
                batched.append(lease.restore_batch_kv_cache(synchronize))
            else:
                merge = getattr(cache, "merge", None)
                if not callable(merge):
                    raise CacheCapsuleUnsupported(f"nonmergeable_plane:{type(cache).__name__}")
                reservation.claim(
                    pool, generation, int(cache.nbytes) * target_batch
                )
                batched.append(merge([cache] * target_batch)); ordinary += 1
        # Unsupported mergeable planes do not pass through ``_accept``, so
        # close their generation race before publishing the mixed result.
        reservation.validate(pool, generation)
        reservation.require_fully_claimed()
    except BaseException:
        for lease in reversed(leases): lease.close()
        for receipt in reversed(receipts): receipt.owner.release()
        _release(reservation)
        raise
    return PreparedPromptCacheCapsules(
        batched, receipts, leases, ordinary, reservation
    )
