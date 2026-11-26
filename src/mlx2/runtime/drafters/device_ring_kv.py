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

        if type(cache) is not RotatingKVCache or int(cache.max_size) != int(window):
            raise ValueError(
                "a device ring adopts only a sliding RotatingKVCache whose max_size is its window"
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
        self.keys[:, :, slots, :] = keys
        self.values[:, :, slots, :] = values
        self.slot_pos[slots] = mx.where(offsets < count, absolute, -1)
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


__all__ = ["DeviceRingKVCache"]
