"""Exact one-way APCv2 Qwen3 KV restore into private native token pages.

APCv2 owns lookup and the ordinary cache branch. This module only copies an
already leased exact hit into a new request arena. Nothing becomes a public
native owner until every layer write reaches successful terminal completion.
"""

from __future__ import annotations

import mlx.core as mx
from threading import RLock

from .models.cache import KVCache


_RESTORE_ORPHANS = []
_RESTORE_ORPHANS_LOCK = RLock()


def reap_failed_native_restores() -> int:
    """Retry terminal retirement without ever releasing ambiguous owners."""
    from .paged_native_retirement import reap_native_request_owner

    with _RESTORE_ORPHANS_LOCK:
        pending = tuple(_RESTORE_ORPHANS)
    for owner, writer in pending:
        try:
            reap_native_request_owner(owner, writer)
            if getattr(owner, "fully_retired", False):
                with _RESTORE_ORPHANS_LOCK:
                    _RESTORE_ORPHANS.remove((owner, writer))
        except Exception:
            pass
    with _RESTORE_ORPHANS_LOCK:
        return len(_RESTORE_ORPHANS)


def retire_failed_native_restore(owner, writer) -> None:
    """Keep an ambiguous native restore reachable until terminal retirement."""
    with _RESTORE_ORPHANS_LOCK:
        _RESTORE_ORPHANS.append((owner, writer))
    try:
        owner.close()
        from .paged_native_retirement import reap_native_request_owner

        reap_native_request_owner(owner, writer)
    finally:
        if getattr(owner, "fully_retired", False):
            with _RESTORE_ORPHANS_LOCK:
                _RESTORE_ORPHANS.remove((owner, writer))


def exact_qwen3_apc_planes(cache, *, layers: int, tokens: int,
                           kv_heads: int, head_dim: int) -> tuple[tuple, ...]:
    """Validate a full-attention fp16 APCv2 branch before any native write."""
    if (not isinstance(cache, list) or len(cache) != layers or
            type(tokens) is not int or tokens < 1 or
            type(kv_heads) is not int or kv_heads < 1 or
            head_dim not in (128, 256)):
        raise ValueError("native APCv2 restore requires one exact KV leaf per layer")
    planes = []
    expected = (1, kv_heads, tokens, head_dim)
    for leaf in cache:
        if type(leaf) is not KVCache or leaf.offset != tokens:
            raise ValueError("native APCv2 restore requires exact full-attention KV")
        if leaf.keys is None or leaf.values is None:
            raise ValueError("native APCv2 KV plane is empty")
        keys, values = leaf.keys_and_values()
        if any(type(value) is not mx.array or value.dtype != mx.float16 or
               tuple(value.shape) != expected for value in (keys, values)):
            raise ValueError("native APCv2 KV shape or dtype mismatch")
        planes.append((keys, values))
    return tuple(planes)


def restore_exact_qwen3_apc_prefix(planes: tuple[tuple, ...],
                                   owners: tuple, backend, *, tokens: int) -> None:
    """Copy a validated branch to unpublished native owners and prove writes."""
    if len(planes) != len(owners) or not planes:
        raise ValueError("native APCv2 restore layer count mismatch")
    for (keys, values), owner in zip(planes, owners):
        # Ordinary KVCache is [B,H,T,D]; the paged writer takes [T,H,D].
        native_keys = mx.transpose(keys, (0, 2, 1, 3))[0]
        native_values = mx.transpose(values, (0, 2, 1, 3))[0]
        backend.append_completed((owner,), native_keys, native_values, (tokens,))
        if owner.offset != tokens:
            raise RuntimeError("native APCv2 restore returned before KV publication")


__all__ = ["exact_qwen3_apc_planes", "restore_exact_qwen3_apc_prefix"]
