"""Device-form lane caches for ``device_round.device_commit`` (spec-round DC-4).

Original mlx2 implementation; design in
docs/research/2026-11-25-gpu-resident-spec-rounds-design.md section 4.9.
Default off: only ``speculative_device_round`` calls into this module, and
only when the ``device_commit`` switch is on.

A one-lane device round with ``device_commit`` commits the lane's caches on
the device, with no host trim:

* target global layers: the exact ``KVCache`` is adopted, buffers and all,
  by a device-mode ``DeviceOffsetKVCache`` (DC-1);
* target recurrent layers (Kimi/Raon RDA, Qwen3.8 GDN ``ArraysCache``):
  unchanged objects, rolled back by ``device_rollback`` (DC-2);
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
    """Ring slack: one round's verify rows, or two under ``lagged_width``.

    The pipeline (DC-5, which needs ``lagged_width``) may undo a round and its
    follower together, so its rings take two rounds of rows. The serial
    lagged-width round gets the same rings: the drafter's attention runs over
    the whole ring, so a different capacity would change its reduction width
    and slot order, and the two rounds would no longer agree bitwise.
    """
    rows = int(owner.num_draft) + 1
    device_round = getattr(owner, "device_round", None)
    policy = getattr(device_round, "policy", None)
    return 2 * rows if getattr(policy, "lagged_width", False) else rows


def in_form(lane):
    return bool(getattr(lane, "device_commit_form", False))


def lane_reason(owner, lane):
    """Why this lane cannot commit on the device this round, or None."""
    from .drafters.device_ring_kv import DeviceRingKVCache
    from .models.cache import ArraysCache, KVCache, RotatingKVCache
    from .models.device_offset_kv import DeviceOffsetKVCache

    from .hybrid_verify_rows import _recurrent

    form = in_form(lane)
    if any(type(cache) is ArraysCache for cache in lane.cache) and getattr(
            getattr(owner, "model", None), "device_commit_recurrent_rollback", False) is not True:
        # Only a target whose recurrent layers stage a device rollback (Kimi
        # Linear's KDA, Raon's RDA, Qwen3.8's gated delta unless compact
        # replay is on) can commit them by a device count; any other keeps
        # the host commit instead of failing at ``device_rollback``.
        return "recurrent_rollback"
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


# Rows a fixed-shape entry reserves at most per K/V cache (``enter``), and the
# step a reserved cache grows by afterwards: one whole-K/V copy per 4,096 rows
# (Inkling 2's CAPACITY_STEP), at most 4,096 spare rows per layer per lane.
KV_RESERVE_CAP = 4096


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
    fixed = getattr(getattr(owner, "device_round", None), "fixed_shape", False)
    menu_rows = int(owner.num_draft) + 1
    ring_list = [r for r in rings if type(r) is DeviceRingKVCache]
    # An empty tail pads only onto allocated rings: an empty ring would take its
    # buffer dtype from the padding (an ``_empty_hidden`` tail is float32).
    if (fixed and rows < menu_rows and ring_list
            and (rows or all(r.keys is not None for r in ring_list))):
        # slice 1a-5: the first round appends the menu width, as every later one
        # does (an empty tail included); rows past the committed count are masked
        # out of the ring.
        tail = lane.tail
        lane.tail = mx.concatenate(
            [tail, mx.zeros((tail.shape[0], menu_rows - rows, *tail.shape[2:]), tail.dtype)],
            axis=1)
        count = mx.array([rows], dtype=mx.int32)
        for ring in rings:
            ring.pending_count = count
    elif rows:
        count = mx.array([rows], dtype=mx.int32)
        for ring in rings:
            ring.pending_count = count
    if fixed:
        # In-place K/V (slice 1a-5): reserve the request's remaining rows plus two
        # rounds of slack now, so no device round grows (concatenates) the K/V.
        # Capped: a longer request grows past the cap, counted at leave.
        remaining = max(0, int(lane.maximum) - int(getattr(lane, "generated", 0) or 0))
        reserve = min(remaining + 2 * width, KV_RESERVE_CAP)
        for entry in cache:
            if type(entry) is DeviceOffsetKVCache:
                entry.device_growths = 0
                if reserve > 0:
                    entry.reserve_rows(reserve, KV_RESERVE_CAP)
        # The conversion runs here, once per entry, not inside the first device
        # round: that round then builds the same graph as every later one.
        from .models.cache import ArraysCache

        mx.eval([lane.tail, *(a for ring in ring_list for a in ring.state if a is not None),
                 *(a for entry in cache if type(entry) is DeviceOffsetKVCache
                   for a in (entry.keys, entry.values, entry.pos) if a is not None),
                 *(a for entry in cache if type(entry) is ArraysCache
                   for a in entry.cache if a is not None)])
    lane.cache, lane.draft_cache = cache, rings
    lane.device_commit_tail_rows = rows
    # The target's exact committed length, kept on the host from the counts
    # it books (the pipelined round, DC-5, settles its KV bound with it).
    lane.device_commit_length = next(
        (int(entry.bound) for entry in cache if type(entry) is DeviceOffsetKVCache), 0)
    # The last booked round's exact start and verify rows: the shared read
    # base of ``device_round.lagged_width`` (``lagged_base``).
    lane.device_last_start = lane.device_commit_length
    lane.device_last_rows = 0
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
    growths = sum(int(entry.device_growths) for entry in lane.cache
                  if type(entry) is DeviceOffsetKVCache)
    cache = [
        entry.to_host_kv() if type(entry) is DeviceOffsetKVCache else entry
        for entry in lane.cache
    ]
    stats = getattr(owner, "scheduler_stats", None)
    if growths and stats is not None and getattr(getattr(owner, "device_round", None),
                                                 "fixed_shape", False):
        # Counted once the conversion succeeded (a refused leave is retried).
        key = "external_device_round_kv_capacity_growths"
        stats[key] = stats.get(key, 0) + growths
    draft_cache = [
        entry.to_host() if type(entry) is DeviceRingKVCache else entry
        for entry in lane.draft_cache
    ]
    lane.cache, lane.draft_cache = cache, draft_cache
    lane.tail = lane.tail[:, : int(rows)]
    lane.device_commit_tail_rows = None
    lane.device_commit_length = None
    lane.device_last_start = lane.device_last_rows = None
    lane.device_commit_form = False
    return True


def lagged_base(lane, inflight_rows=None):
    """The shared read base of round X (``device_round.lagged_width``, DC-5).

    ``start(X-1) + rows(X-1)``: the bound the KV caches carry when X is
    dispatched with X-1 still in flight, which the host also knows in the
    serial round. With X-1 in flight (``inflight_rows`` its verify rows) its
    start is the last booked length; otherwise X-1 is the last booked round.
    The serial and the pipelined round therefore read the same width.
    """
    if inflight_rows is not None:
        return int(lane.device_commit_length) + int(inflight_rows)
    return int(lane.device_last_start) + int(lane.device_last_rows)


def set_read_base(lane, base):
    from .models.device_offset_kv import DeviceOffsetKVCache

    for entry in lane.cache:
        if type(entry) is DeviceOffsetKVCache:
            entry.set_read_base(base)


def book_round(lane, rows, consumed):
    """Host bookkeeping of one committed device round (serial or pipelined)."""
    lane.device_last_start = int(lane.device_commit_length)
    lane.device_last_rows = int(rows)
    lane.device_commit_length = int(lane.device_commit_length) + int(consumed)


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


__all__ = ["VERSION", "book_round", "drafter", "enter", "in_form", "lagged_base", "lane_reason",
           "leave", "refuse", "set_read_base", "slack", "stage_tail"]
