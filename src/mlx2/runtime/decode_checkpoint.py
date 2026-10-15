# SPDX-License-Identifier: MIT
"""Same-process, executor-owned ordinary B=1 checkpoints; never prompt replay.

The small control plane stays owned in memory (including sampled pending
output, RNG, processors and stop matcher). Only KV backing storage is parked.
The serving caller admits the bounded serialization/restore staging window.
No pickle, global prefix publication, process restart, or route conversion.
"""
from __future__ import annotations

import hashlib
import io
import tempfile
from collections.abc import Mapping
from concurrent.futures import TimeoutError as FutureTimeout

import mlx.core as mx

from .io_ownership import ReadPool, ReadTimeout
from .models.cache import load_prompt_cache, save_prompt_cache

CHECKPOINT_IO_SECONDS = 60.0
HEADER_ALLOWANCE = 16 << 20


class _PrivateFile:
    def __init__(self, handle):
        self.handle = handle

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __del__(self):
        self.close()


def _write_payload(payload):
    # TemporaryFile is unlinked immediately on supported serving platforms;
    # ownership is its descriptor, so cancellation cannot orphan a pathname.
    owner = _PrivateFile(tempfile.TemporaryFile(mode="w+b"))  # noqa: SIM115 - ownership transfers to the checkpoint
    try:
        owner.handle.write(payload)
        owner.handle.flush()
        return owner
    except BaseException:
        owner.close()
        raise


def _read_payload(owner, size):
    owner.handle.seek(0)
    data = owner.handle.read(size + 1)
    if len(data) != size:
        raise ValueError("exact decode checkpoint length changed")
    return data


def checkpoint_block(batch, uid):
    """Capability checks, never a model-name exception."""
    if batch.uids != [uid] or batch._next_execution_widths != [1]:
        return "exact_checkpoint_requires_ordinary_width_one"
    if (not batch._next_logprobs or batch._next_tokens is None
            or batch._lane_failures or batch._mixed_segment is not None
            or any(value is not None for value in batch.persistent_inputs)):
        return "exact_checkpoint_boundary_unavailable"
    if not batch.prompt_cache or any(
        not callable(getattr(cache, "exact_decode_checkpoint_contract", None))
        or cache.exact_decode_checkpoint_contract() is None
        for cache in batch.prompt_cache
    ):
        return "exact_checkpoint_cache_unsupported"
    return None


def checkpoint_staging_bytes(batch):
    # BytesIO serialization and MLX restore may each coexist with a full
    # payload copy. Account both plus bounded metadata before attempting it.
    def arrays(value):
        if isinstance(value, mx.array):
            yield value
        elif isinstance(value, Mapping):
            for key in sorted(value):
                yield from arrays(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from arrays(item)
        elif value is not None:
            raise TypeError(
                f"unsupported exact checkpoint state leaf: {type(value).__name__}"
            )

    payload = sum(
        int(value.nbytes)
        for cache in batch.prompt_cache
        for value in arrays(cache.state)
    )
    return 2 * (payload + HEADER_ALLOWANCE)


def _state_geometry(value):
    if isinstance(value, mx.array):
        return ("array", tuple(value.shape), str(value.dtype))
    if isinstance(value, Mapping):
        return ("mapping", tuple(
            (key, _state_geometry(value[key])) for key in sorted(value)
        ))
    if isinstance(value, list):
        return ("list", tuple(_state_geometry(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_state_geometry(item) for item in value))
    if value is None:
        return None
    raise TypeError(f"unsupported exact checkpoint state leaf: {type(value).__name__}")


def _geometry(cache):
    return (
        type(cache).__name__,
        _state_geometry(cache.state),
        tuple(cache.meta_state),
    )


class OrdinaryDecodeCheckpoint:
    """Single-use owner, bound to one model identity, request and numerical law."""

    @classmethod
    def capture(cls, batch, uid, binding):
        reason = checkpoint_block(batch, uid)
        if reason:
            raise ValueError(reason)
        owner = cls()
        owner.control = None
        owner.file = None
        owner.pool = ReadPool(1, "ordinary-checkpoint", timeout=CHECKPOINT_IO_SECONDS,
                              max_pending=1)
        owner.binding = binding
        owner.model = batch.model
        owner.uid = uid
        owner.consumed = False
        owner.closed = False
        owner.staging_bytes = checkpoint_staging_bytes(batch)
        owner.geometry = tuple(_geometry(cache) for cache in batch.prompt_cache)
        owner.cache_contracts = tuple(
            cache.exact_decode_checkpoint_contract() for cache in batch.prompt_cache
        )
        try:
            # Drain the complete committed boundary and the already sampled
            # next output before releasing any KV backing arrays.
            mx.eval([cache.state for cache in batch.prompt_cache], batch._next_tokens,
                    batch._next_logprobs, [context.state for context in batch._token_context],
                    [rng.key for rng in batch.lane_rngs if rng is not None])
            output = io.BytesIO()
            save_prompt_cache(output, batch.prompt_cache)
            payload = output.getvalue()
            if len(payload) > owner.staging_bytes // 2:
                raise ValueError("exact decode checkpoint exceeds admitted staging")
            owner.size = len(payload)
            owner.digest = hashlib.sha256(payload).digest()
            future = owner.pool.submit(_write_payload, payload)
            try:
                owner.file = future.result(timeout=CHECKPOINT_IO_SECONDS)
            except FutureTimeout as error:
                failure = ReadTimeout("exact decode checkpoint write exceeded deadline")
                owner.pool.fail(failure)
                raise failure from error
            # Transfer only after private local ownership exists. A failed
            # capture leaves the original executor and its KV untouched.
            owner.control = batch
            batch.prompt_cache = []
            return owner
        except BaseException:
            owner.close()
            raise

    def restore(self, model, binding):
        if self.closed or self.consumed or self.control is None:
            raise ValueError("exact decode checkpoint is no longer owned")
        if self.model is not model or self.binding != binding:
            raise ValueError("exact decode checkpoint model/revision/request mismatch")
        future = self.pool.submit(_read_payload, self.file, self.size)
        try:
            payload = future.result(timeout=CHECKPOINT_IO_SECONDS)
        except FutureTimeout as error:
            failure = ReadTimeout("exact decode checkpoint read exceeded deadline")
            self.pool.fail(failure)
            raise failure from error
        if hashlib.sha256(payload).digest() != self.digest:
            raise ValueError("exact decode checkpoint digest mismatch")
        cache = load_prompt_cache(io.BytesIO(payload))
        mx.eval([item.state for item in cache])
        if tuple(_geometry(item) for item in cache) != self.geometry:
            raise ValueError("exact decode checkpoint geometry changed")
        if len(cache) != len(self.cache_contracts):
            raise ValueError("exact decode checkpoint cache contract count changed")
        for item, contract in zip(cache, self.cache_contracts):
            restore = getattr(item, "restore_exact_decode_checkpoint", None)
            if not callable(restore):
                raise TypeError("exact decode checkpoint cache capability changed")
            restore(contract)
        batch = self.control
        batch.prompt_cache = cache
        self.control = None
        self.consumed = True
        self.close()
        return batch

    def close(self):
        if self.closed:
            return
        self.control = None
        self.model = None
        self.owner_executor = None
        self.pool.close(self.file)
        # The pool owns the one final close, including an in-progress close
        # after its caller deadline. Never retry a possibly reused descriptor.
        self.file = None
        self.closed = True
