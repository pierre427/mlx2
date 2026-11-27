"""Drafter context cache with a device-held fill (spec-round slice 1a-4, DC-3).

Design: docs/research/2026-11-25-gpu-resident-spec-rounds-design.md section 4.9.
Default off: nothing constructs this cache unless a caller opts in.

A DFlash drafter caches the committed target-feature context of its sliding
attention layers (``RotatingKVCache(max_size=window)``). After a device-side
commit the host no longer knows how many verify features were committed, so
this cache takes **every** verify feature row each round and keeps only the
first ``count`` (a device int32), as the host path's ``tail[:consumed]``
append does:

* storage is a ring of ``window + slack`` slots; each slot remembers the
  absolute position it holds (``slot_pos``, device int32, ``-1`` = empty);
* keys are stored already rotated at their absolute positions
  (``rope(ctx_keys, offset=pos)``), so slot order never matters to the law;
* ``append(keys, values, count)`` writes the ``S`` new rows at slots
  ``(pos + j) % capacity`` (``S <= slack`` keeps every still-visible slot
  intact), marks rows ``j >= count`` empty, and advances ``pos`` by
  ``count``;
* the context mask is ``pos - window <= slot_pos < pos``: the last
  ``window`` committed rows, exactly the rotating cache's visible window.

Attention over the whole ring under that mask equals the rotating cache's
attention up to column order and reduction width, so drafts match the host
path's up to near ties; drafts never change verified output, only
acceptance (Pierre's 2026-11-26 bar for device-commit rounds is near-tie).

**Recovery lending (DC-4).** A per-round recovery point
(``cow_cache.snapshot_recovery_descriptors``) borrows the ring's K/V buffers
instead of aliasing them (``recovery_lend``/``recovery_borrow``): an alias
would make every append copy the whole ring. The captured ``slot_pos`` and
``pos`` keep their aliases (small). This is exact while at most ``slack``
rows were written since the capture: those rows land in the slots of
absolute positions ``captured pos + j`` (``j < slack``), which, in the
captured state, hold positions below ``captured pos - window`` or nothing,
so they are invisible there. A restore after more rows than that refuses.
``to_host`` is the explicit conversion back to the host ``RotatingKVCache``
(one host read of ``pos``).
"""

from __future__ import annotations

import mlx.core as mx


class DeviceRingKVCache:
    def __init__(self, window: int, slack: int):
        if window < 1 or slack < 1:
            raise ValueError("device ring caches need a positive window and slack")
        self.window = int(window)
        self.slack = int(slack)
        self.capacity = self.window + self.slack
        self.keys = None
        self.values = None
        self.slot_pos = mx.full((self.capacity,), -1, dtype=mx.int32)
        self.pos = mx.array([0], dtype=mx.int32)
        # Device int32 [1]: committed rows of the next append (None: all).
        self.pending_count = None
        # Host upper bound on ``pos`` (rows ever written), for shape readers.
        self.bound = 0

    @property
    def offset(self):
        raise RuntimeError(
            "a device ring cache has no host offset; rotate with its device ``pos``"
        )

    @classmethod
    def from_host(cls, cache, *, window: int, slack: int):
        """Adopt a host sliding-layer ``RotatingKVCache(max_size=window)``.

        Only that law ("the last ``window`` committed rows") is a ring's; a
        full-attention ``KVCache`` or a rotating cache of another size would
        silently lose context, so both refuse.
        """
        from ..models.cache import RotatingKVCache

        if type(cache) is not RotatingKVCache or cache.keep or int(cache.max_size) != int(window):
            raise ValueError(
                "a device ring adopts only a sliding RotatingKVCache with keep=0 "
                "whose max_size is its window"
            )
        ring = cls(window, slack)
        length = int(cache.offset)
        if length == 0 or cache.keys is None:
            return ring
        keys, values = cache._temporal_order(cache.keys), cache._temporal_order(cache.values)
        held = min(int(keys.shape[2]), length, ring.window)
        keys, values = keys[..., -held:, :], values[..., -held:, :]
        positions = mx.arange(length - held, length, dtype=mx.int32)
        slots = positions % ring.capacity
        b, heads, _, k_dim = keys.shape
        ring.keys = mx.zeros((b, heads, ring.capacity, k_dim), keys.dtype)
        ring.values = mx.zeros((b, heads, ring.capacity, values.shape[-1]), values.dtype)
        ring.keys[:, :, slots, :] = keys
        ring.values[:, :, slots, :] = values
        ring.slot_pos[slots] = positions
        ring.pos = mx.array([length], dtype=mx.int32)
        ring.bound = length
        return ring

    def rope_offset(self):
        """The rotation offset of the next context row (device scalar)."""
        return self.pos.reshape(())

    def append(self, keys, values, count=None):
        """Write ``S`` rotated rows, keep the first ``count``; return the ring and its mask."""
        if keys.shape[0] != 1:
            raise ValueError("device ring caches hold one row (B=1)")
        rows = int(keys.shape[2])
        if rows > self.slack:
            raise ValueError("a device ring append is wider than its slack")
        if rows == 0:
            return self.keys, self.values, self.context_mask()  # keeps a pending count
        if count is None:
            count = self.pending_count
        if count is None:
            # A forgotten count would commit every verify row (rejected ones
            # included) with no error.
            raise ValueError("a device ring append needs a committed count (pending_count)")
        self.pending_count = None
        count = mx.minimum(mx.maximum(count.reshape(1).astype(mx.int32), 0), rows)
        if self.keys is None and rows:
            b, heads, _, k_dim = keys.shape
            self.keys = mx.zeros((b, heads, self.capacity, k_dim), keys.dtype)
            self.values = mx.zeros((b, heads, self.capacity, values.shape[-1]), values.dtype)
        offsets = mx.arange(rows, dtype=mx.int32)
        absolute = self.pos + offsets
        slots = absolute % self.capacity
        committed = offsets < count
        # A masked attention weight can still multiply a discarded NaN or
        # infinity. Keep rejected verify rows out of both operands.
        keep = committed[None, None, :, None]
        self.keys[:, :, slots, :] = mx.where(keep, keys, 0)
        self.values[:, :, slots, :] = mx.where(keep, values, 0)
        self.slot_pos[slots] = mx.where(committed, absolute, -1)
        self.pos = self.pos + count
        self.bound += rows
        return self.keys, self.values, self.context_mask()

    def context_mask(self):
        """``[capacity]`` bool: the last ``window`` committed rows."""
        end = self.pos.reshape(())
        return (self.slot_pos >= 0) & (self.slot_pos >= end - self.window) & (self.slot_pos < end)

    def empty(self):
        return self.keys is None

    @property
    def state(self):
        return (self.keys, self.values, self.slot_pos, self.pos)

    @property
    def nbytes(self):
        return sum(int(a.nbytes) for a in (self.keys, self.values, self.slot_pos, self.pos)
                   if a is not None)

    # -- host conversion (DC-4) ----------------------------------------------------
    def to_host(self):
        """The host ``RotatingKVCache(max_size=window)`` holding the committed window.

        Reads ``pos`` once (an explicit sync point). A pending count belongs to
        the caller's undrafted tail, which the caller slices on the host; the
        ring itself must hold every committed row of its window, else this
        refuses.
        """
        from ..models.cache import RotatingKVCache

        length = int(self.pos.reshape(()).item())
        cache = RotatingKVCache(max_size=self.window, keep=0)
        if length == 0:
            return cache
        if self.keys is None:
            raise RuntimeError("a device ring with committed rows has no buffers")
        held = min(length, self.window)
        positions = mx.arange(length - held, length, dtype=mx.int32)
        slots = positions % self.capacity
        if not bool(mx.all(self.slot_pos[slots] == positions).item()):
            raise RuntimeError("a device ring lost a committed row of its window")
        cache.keys = mx.contiguous(self.keys[:, :, slots, :])
        cache.values = mx.contiguous(self.values[:, :, slots, :])
        cache.offset = length
        cache._idx = held
        return cache

    # -- recovery lending (cow_cache) ----------------------------------------------
    def recovery_lend(self, clone):
        """Leave the clone without K/V; the token proves a later borrow exact."""
        if self.keys is None:
            return None
        clone.keys = clone.values = None
        return (int(self.bound), tuple(self.keys.shape), tuple(self.values.shape),
                str(self.keys.dtype), str(self.values.dtype))

    def recovery_borrow(self, target, token):
        """Give ``target`` (the restored clone) this live ring's K/V buffers."""
        from ..cow_cache import COWCacheError

        bound, k_shape, v_shape, k_dtype, v_dtype = token
        if (self.keys is None or self.values is None
                or tuple(self.keys.shape) != k_shape or tuple(self.values.shape) != v_shape
                or str(self.keys.dtype) != k_dtype or str(self.values.dtype) != v_dtype):
            raise COWCacheError("a device ring's buffers changed geometry since its recovery point")
        if not 0 <= self.bound - bound <= self.slack:
            raise COWCacheError(
                "a device ring took more than its slack since its recovery point; "
                "the captured window may be overwritten"
            )
        target.keys = mx.stop_gradient(self.keys)
        target.values = mx.stop_gradient(self.values)
        return (target.keys, target.values)


__all__ = ["DeviceRingKVCache"]
