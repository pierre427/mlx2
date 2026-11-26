"""Device-form lane caches for ``device_round.device_commit`` (spec-round DC-4).

Original mlx2 implementation; design in
docs/research/2026-11-25-gpu-resident-spec-rounds-design.md section 4.9.
Default off: only ``speculative_device_round`` calls into this module, and
only when the ``device_commit`` switch is on.

A one-lane device round with ``device_commit`` commits the lane's caches on
the device, with no host trim:

* target global layers: the exact ``KVCache`` is adopted, buffers and all,
  by a device-mode ``DeviceOffsetKVCache`` (DC-1);
* target recurrent layers (Kimi/Raon RDA ``ArraysCache``): unchanged objects,
  rolled back by ``device_rollback`` (DC-2);
* drafter sliding context: each ``RotatingKVCache(max_size=window)`` becomes
  a ``DeviceRingKVCache`` (DC-3) of slack ``num_draft + 1``; the ring takes
  every verify feature row and keeps the committed count (``pending_count``).

The round still reads every packet, so the host knows each committed count:
the KV caches are observed at their exact length after every commit (no
slack) and the lane remembers how many rows of ``lane.tail`` are committed
(``device_commit_tail_rows``).

The lane keeps that device form across its device rounds (``enter`` at its
first one, flag ``lane.device_commit_form``) and returns to the exact host
form (``leave``) wherever host form is needed: a declined or restored round
(the host path then runs), a finish (prompt-cache freeze and sidecar), an
ordinary or parked round, and ``remove`` with prompt caches. ``refuse`` is the
loud guard for any other reader: a ring has no host offset at all, and a
device-mode KV cache refuses host contracts.

Converting builds new cache objects (the ArraysCache rows stay the same
objects), so instance-level APCv2 copy-on-write telemetry (segment tokens and
mutation hooks of a lane restored from an APC hit) does not follow the KV and
ring caches into device form: their first-write materialization may go
uncounted. Tokens and cache values are unaffected.

Engines with a ``cost_per_commit_handoff`` keep the host commit (a parked
lane's batch route reads its caches before any round dispatch).

Fixed kernels this cannot serve (FOLD, a causal drafter, the fused DFlash2
layer and attention kernels, which read ``entry.offset``) keep the host
commit for that round (counted).
"""

from __future__ import annotations

VERSION = "speculative-device-commit-v1"


def drafter(owner):
    """The DFlash2 drafter under any proposal wrapper, or None."""
    from .drafters.dflash2 import DFlash2DraftModel

    model = getattr(owner, "draft", None)
    model = getattr(model, "backend", model)
    return model if isinstance(model, DFlash2DraftModel) else None


def slack(owner):
    return int(owner.num_draft) + 1


def in_form(lane):
    return bool(getattr(lane, "device_commit_form", False))


def lane_reason(owner, lane):
    """Why this lane cannot commit on the device this round, or None."""
    from .drafters.device_ring_kv import DeviceRingKVCache
    from .models.cache import ArraysCache, KVCache, RotatingKVCache
    from .models.device_offset_kv import DeviceOffsetKVCache

    from .hybrid_verify_rows import _recurrent

    form = in_form(lane)
    for cache in lane.cache:
        kind = type(cache)
        if kind is ArraysCache:
            # ``DeviceCommitTransaction``'s own checks: a recurrent B1 row
            # (device rollback) with no speculation owner yet.
            if (_recurrent(cache) and not cache.speculating
                    and (cache.batch_size == 1 or cache.empty())):
                continue
            return "target_cache"
        if kind is KVCache and not form:
            continue
        if kind is DeviceOffsetKVCache and form and cache.device_mode:
            continue
        return "target_cache"
    model = drafter(owner)
    if model is None:
        return "drafter"
    if len(lane.draft_cache) != len(model.layers):
        return "draft_cache"
    if getattr(model, "fold_policy", None) is not None:
        return "drafter_fold"
    for layer, cache in zip(model.layers, lane.draft_cache, strict=True):
        attn = layer.self_attn
        if (attn.is_causal or getattr(attn, "fold_policy", None) is not None
                or layer.__dict__.get("_fused_layer") is not None
                or attn.__dict__.get("_fused_forward") is not None):
            return "drafter_kernel"
        if form:
            if type(cache) is not DeviceRingKVCache:
                return "draft_cache"
            continue
        if (type(cache) is not RotatingKVCache or not attn.is_sliding or cache.keep
                or int(cache.max_size) != int(attn.sliding_window) - 1):
            return "draft_cache"
    if not form and int(lane.tail.shape[1]) > slack(owner):
        # The first round after a prefill slice: its tail is wider than a
        # ring append; this round commits on the host, the next one enters.
        return "long_tail"
    return None


def enter(owner, lane):
    """Convert the lane's caches to device form (no copy of target K/V)."""
    import mlx.core as mx

    from .drafters.device_ring_kv import DeviceRingKVCache
    from .models.cache import KVCache, RotatingKVCache
    from .models.device_offset_kv import DeviceOffsetKVCache

    if in_form(lane):
        return False
    width = slack(owner)
    cache = [
        DeviceOffsetKVCache.adopt(entry) if type(entry) is KVCache else entry
        for entry in lane.cache
    ]
    rings = [
        DeviceRingKVCache.from_host(entry, window=int(entry.max_size), slack=width)
        if type(entry) is RotatingKVCache else entry
        for entry in lane.draft_cache
    ]
    rows = int(lane.tail.shape[1])
    if rows:
        count = mx.array([rows], dtype=mx.int32)
        for ring in rings:
            ring.pending_count = count
    lane.cache, lane.draft_cache = cache, rings
    lane.device_commit_tail_rows = rows
    lane.device_commit_form = True
    return True


def stage_tail(lane, count, rows):
    """After a device commit: the whole verify tail is pending, ``rows`` committed."""
    from .drafters.device_ring_kv import DeviceRingKVCache

    for ring in lane.draft_cache:
        if type(ring) is not DeviceRingKVCache:
            raise RuntimeError("a device-commit lane's draft context must be device rings")
        ring.pending_count = count
    lane.device_commit_tail_rows = int(rows)


def leave(owner, lane):
    """Back to the exact host form: ``KVCache``, ``RotatingKVCache``, committed tail."""
    from .drafters.device_ring_kv import DeviceRingKVCache
    from .models.device_offset_kv import DeviceOffsetKVCache

    if not in_form(lane):
        return False
    rows = getattr(lane, "device_commit_tail_rows", None)
    if rows is None or not 0 <= int(rows) <= int(lane.tail.shape[1]):
        raise RuntimeError("a device-commit lane lost its committed tail length")
    cache = [
        entry.to_host_kv() if type(entry) is DeviceOffsetKVCache else entry
        for entry in lane.cache
    ]
    draft_cache = [
        entry.to_host() if type(entry) is DeviceRingKVCache else entry
        for entry in lane.draft_cache
    ]
    lane.cache, lane.draft_cache = cache, draft_cache
    lane.tail = lane.tail[:, : int(rows)]
    lane.device_commit_tail_rows = None
    lane.device_commit_form = False
    return True


def refuse(caches, what):
    """Loud guard: a host reader met device-form caches."""
    from .drafters.device_ring_kv import DeviceRingKVCache
    from .models.device_offset_kv import DeviceOffsetKVCache

    for entry in caches:
        if type(entry) is DeviceRingKVCache or (
                type(entry) is DeviceOffsetKVCache and entry.device_mode):
            raise RuntimeError(
                f"{what} needs host-form caches; a device-commit lane must leave its "
                "device form first (speculative_device_commit.leave)"
            )


__all__ = ["VERSION", "drafter", "enter", "in_form", "lane_reason", "leave", "refuse",
           "slack", "stage_tail"]
