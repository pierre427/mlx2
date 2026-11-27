"""KV cache with a device-held fill position (spec-round slice 1a-4, DC-1).

Design: docs/research/2026-11-25-gpu-resident-spec-rounds-design.md section 4.9.
Default off: nothing constructs this cache unless a caller opts in.

``DeviceOffsetKVCache`` is a ``KVCache`` in **host mode** (the default after
construction): its masks, offsets and attention outputs are the parent's, so a
forward is bitwise the parent's.  Features gated on the exact ``KVCache`` type
(planned-prompt growth, in-place B>1 flash buffers, K/V recovery lending) are
simply not taken, which changes allocation, not values.

In **device mode** (``enter_device_mode``) a speculative round can advance the
cache by an accepted count that stays on the device:

* ``pos`` is the true fill as an int32 ``mx.array`` of shape ``[1]``; the
  host keeps only ``bound``, an upper bound on it (the last exact length it
  read plus every row written since).
* ``update_and_fetch`` writes the new rows at ``pos`` (``mx.slice_update``)
  and returns the first ``bound + rows`` rows; ``offset`` mirrors that host
  width so shape-reading callers keep working.
* ``make_mask(N)`` returns the boolean ``j <= pos + i`` mask over that width:
  rows past the true fill (rejected rows of earlier rounds, the slack) are
  never attended.
* ``device_commit(count)`` sets ``pos = base + count`` for the rows of the
  last write; ``observe(length)`` tightens ``bound`` once the host has read
  the exact length (and leaves the slack nowhere).

``offset`` is the host *bound* in device mode, not the fill: it is only safe
for caches whose layer encodes no position from ``offset`` (Raon's NoPE MLA
latent).  Any position-encoding (RoPE) layer must take its position from
``pos``.  State and offset setters, serialization, quantization and merging refuse in
device mode instead of copying the slack rows as if they were filled.

Attention over the bounded width is equal to the exact-length attention only
up to reduction order (MLX's kernels group by key width): Pierre's
2026-11-26 bar for device-commit rounds is the near-tie contract, not bitwise.
With the bound exact (``observe`` after every round, as slice DC-4 does) the
width is the host width and the ``j <= pos + i`` mask selects exactly the
causal keys.

**Recovery lending (DC-4).** The engine's per-round recovery point
(``cow_cache.snapshot_recovery_descriptors``) borrows K/V instead of aliasing
them, as it does for ``KVCache``: a device write lands at ``pos``, never
below the committed fill, so the captured prefix stays intact and the write
needs no whole-buffer copy. The borrowed level is ``recovery_fill``: the host
offset in host mode and, in device mode, the bound only while it is exact
(observed, no write since). Otherwise it refuses, because rows in
``[pos, bound)`` are slack that the next write may overwrite. ``observe``
below a borrowed level marks that recovery point's rewind guard, as
``KVCache.trim`` does.
"""

from __future__ import annotations

import mlx.core as mx

from .cache import KVCache


class DeviceOffsetKVCache(KVCache):
    # Class defaults: ``from_state`` builds instances with ``__new__`` only.
    device_mode = False
    pos = None
    bound = 0
    _last_write = None
    # Device mode: whether ``bound`` is the exact fill (observed, no write since).
    _exact = True
    # Device mode: the host read base of the next forward (``read_base``), or
    # None for the bound. A shared read width (``device_round.lagged_width``,
    # DC-5): every device-commit round reads ``read_base + rows`` columns, the
    # same in the serial and the pipelined round, whatever the bound is.
    read_base = None
    # Recovery snapshots borrow K/V below ``recovery_fill`` (module note).
    _RECOVERY_APPEND_ONLY_FIELDS = (
        ("keys", -2, "recovery_fill"), ("values", -2, "recovery_fill"),
    )
    # The plane a device-mode request-private recovery clone is filed under
    # (``cow_cache``); committed, publishable captures still refuse device mode.
    _RECOVERY_PLANE = "attention_kv"
    # Device mode: buffer growths (a whole-K/V concatenate each) since entry;
    # ``reserve`` at entry keeps a fixed-shape lane at zero (slice 1a-5).
    device_growths = 0
    # ``reserve_rows``: the growth step of a reserved cache (rows), or 0 (``step``).
    # Its host form marks the spare rows as capacity (``_fixed_capacity_step``).
    reserve_step = 0

    def __init__(self):
        super().__init__()
        self.device_mode = False
        self.pos = None
        self.bound = 0
        self._last_write = None
        self._exact = True

    @classmethod
    def adopt(cls, cache):
        """A device-mode cache over an exact ``KVCache``'s own buffers (no copy)."""
        if type(cache) is not KVCache:
            raise TypeError("only an exact KVCache is adopted into device mode")
        adopted = cls()
        adopted.keys, adopted.values = cache.keys, cache.values
        adopted.offset = int(cache.offset)
        return adopted.enter_device_mode()

    def to_host_kv(self):
        """The exact ``KVCache`` at the observed fill, over the same buffers."""
        if self.device_mode and not self._exact:
            raise RuntimeError("observe() a device-mode KV cache's length before leaving it")
        host = KVCache()
        host.keys, host.values = self.keys, self.values
        host.offset = int(self.bound if self.device_mode else self.offset)
        if self.reserve_step and self.keys is not None and host.offset < int(self.keys.shape[2]):
            # The reserve's spare rows are capacity, not state (``KVCache.state``,
            # the capsule capture and batch merges expose only committed rows).
            host._fixed_capacity_step = int(self.reserve_step)
        return host

    @property
    def recovery_fill(self):
        if not self.device_mode:
            return int(self.offset)
        if not self._exact:
            raise RuntimeError(
                "a device-mode KV cache with unobserved rows has no exact recovery level"
            )
        return int(self.bound)

    # Host mode is KVCache's ABI exactly (same planes, fields and offset law).
    def cache_operations_contract(self):
        if self.device_mode:
            from ..cache_operations import CacheOperationsUnsupported

            raise CacheOperationsUnsupported(
                "a device-mode KV cache has no host offset; observe() its length first"
            )
        from ..cache_operations import KV_OPERATIONS

        return KV_OPERATIONS

    # -- mode changes ------------------------------------------------------------
    def enter_device_mode(self):
        if self.device_mode:
            return self
        self.pos = mx.array([int(self.offset)], dtype=mx.int32)
        self.bound = int(self.offset)
        self.device_mode = True
        self._exact = True
        return self

    def observe(self, length: int):
        """The host read the exact fill: tighten the bound (device mode stays)."""
        length = int(length)
        if not 0 <= length <= self.bound:
            raise ValueError("observed KV length exceeds the cache's host bound")
        self.bound = length
        self.offset = length
        self.pos = mx.array([length], dtype=mx.int32)
        self._last_write = None
        self._exact = True
        self.read_base = None
        from .cache import _note_recovery_rewind

        # Rows of a borrowed recovery prefix above ``length`` may be rewritten.
        _note_recovery_rewind(self)
        return self

    def settle(self, length: int, pending: int = 0):
        """The host read the exact committed fill ``length`` while ``pending``
        more rows (a dispatched, already device-committed round) may follow it.

        ``pending == 0`` is ``observe``. Otherwise the device ``pos`` stays (it
        includes the pending round's commit) and only the host bound tightens to
        ``length + pending``: the slack of a pipelined round (spec-round DC-5).
        """
        length, pending = int(length), int(pending)
        if pending == 0:
            return self.observe(length)
        if self._last_write is not None:
            raise RuntimeError("settle a device-mode KV cache only after its last write committed")
        if length < 0 or pending < 0 or length + pending > self.bound:
            raise ValueError("settled KV length exceeds the cache's host bound")
        self.bound = length + pending
        self.offset = self.bound
        self._exact = False
        return self

    def leave_device_mode(self, length: int):
        """Back to host mode at an exact host-read length."""
        self.observe(length)
        self.device_mode = False
        self.pos = None
        self._last_write = None
        return self

    # -- device-mode ops ----------------------------------------------------------
    def _base(self):
        """The column count before this forward's rows: the read base or the bound."""
        if self.read_base is None:
            return self.bound
        base = int(self.read_base)
        if base < self.bound:
            raise ValueError("a device KV read base below the host bound would hide written rows")
        return base

    def set_read_base(self, base):
        """Read ``base + rows`` columns in the next forward (None: the bound)."""
        if base is not None and int(base) < self.bound:
            raise ValueError("a device KV read base below the host bound would hide written rows")
        self.read_base = None if base is None else int(base)
        return self

    def reserve_rows(self, rows: int, step: int):
        """Grow the buffers once so ``rows`` more fit past the bound (slice 1a-5).

        ``device_round.fixed_shape`` reserves a request's remaining rows (up to
        ``step``) at device-commit entry, so no device-mode round grows (and
        copies) the whole K/V for the next ``step`` rows; a later growth takes
        ``step`` rows at once. Not ``KVCache.reserve`` (a planned total that
        batch merges carry). Capacity changes allocation only: every read
        slices ``[:base + rows]``. Returns whether it grew."""
        rows, step = int(rows), int(step)
        if step < 1:
            raise ValueError("a K/V reserve needs a positive growth step")
        self.reserve_step = step
        if self.keys is None or rows <= 0:
            return False
        need = self.bound + rows
        have = int(self.keys.shape[2])
        if need <= have:
            return False
        capacity = ((need + self.step - 1) // self.step) * self.step
        b, heads, _, k_dim = self.keys.shape
        v_dim = self.values.shape[3]
        self.keys = mx.concatenate(
            [self.keys, mx.zeros((b, heads, capacity - have, k_dim), self.keys.dtype)], axis=2)
        self.values = mx.concatenate(
            [self.values, mx.zeros((b, heads, capacity - have, v_dim), self.values.dtype)], axis=2)
        return True

    def _ensure_capacity(self, keys, values, rows: int):
        need = self._base() + rows
        if self.keys is not None and need <= self.keys.shape[2]:
            return
        if self.device_mode and self.keys is not None:
            self.device_growths = int(self.device_growths) + 1
            if self.reserve_step:
                # A reserved cache grows a whole reserve step at once (one copy
                # per ``reserve_step`` rows, not per ``step``).
                b, heads, _, k_dim = self.keys.shape
                have = int(self.keys.shape[2])
                extra = max(need - have, self.reserve_step)
                self.keys = mx.concatenate(
                    [self.keys, mx.zeros((b, heads, extra, k_dim), self.keys.dtype)], axis=2)
                self.values = mx.concatenate(
                    [self.values, mx.zeros((b, heads, extra, values.shape[3]), self.values.dtype)],
                    axis=2)
                return
        (b, heads, _, k_dim) = keys.shape
        v_dim = values.shape[3]
        have = 0 if self.keys is None else self.keys.shape[2]
        capacity = ((max(need, have + self.step) + self.step - 1) // self.step) * self.step
        extra = capacity - have
        new_k = mx.zeros((b, heads, extra, k_dim), keys.dtype)
        new_v = mx.zeros((b, heads, extra, v_dim), values.dtype)
        if self.keys is None:
            self.keys, self.values = new_k, new_v
        else:
            self.keys = mx.concatenate([self.keys, new_k], axis=2)
            self.values = mx.concatenate([self.values, new_v], axis=2)

    def update_and_fetch(self, keys, values):
        if not self.device_mode:
            return super().update_and_fetch(keys, values)
        if keys.shape[0] != 1:
            raise ValueError("device-mode KV caches hold one row (B=1)")
        rows = int(keys.shape[2])
        width = self._base() + rows
        self._ensure_capacity(keys, values, rows)
        self.keys = mx.slice_update(self.keys, keys, self.pos, axes=(2,))
        self.values = mx.slice_update(self.values, values, self.pos, axes=(2,))
        self._last_write = (self.pos, rows)
        self._exact = False
        self.bound += rows
        self.offset = self.bound
        return self.keys[..., :width, :], self.values[..., :width, :]

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        if not self.device_mode:
            return super().make_mask(N, return_array=return_array, window_size=window_size, **kwargs)
        if window_size is not None:
            raise ValueError("device-mode KV caches have no sliding window")
        width = self._base() + int(N)  # the fetch that follows returns this many rows
        columns = mx.arange(width, dtype=mx.int32)[None, :]
        rows = mx.arange(int(N), dtype=mx.int32)[:, None]
        return columns <= (self.pos + rows)

    def device_commit(self, count):
        """Keep the first ``count`` rows (device int32) of the last write."""
        if not self.device_mode or self._last_write is None:
            raise RuntimeError("device_commit needs a device-mode write to commit")
        base, rows = self._last_write
        count = mx.minimum(mx.maximum(count.reshape(1).astype(mx.int32), 0), rows)
        self.pos = base + count
        self._last_write = None
        self.read_base = None  # one forward's read width
        return self

    def trim(self, n):
        if self.device_mode:
            raise RuntimeError("a device-mode KV cache advances by device_commit, not trim")
        return super().trim(n)

    @property
    def meta_state(self):
        if self.device_mode:
            raise RuntimeError("a device-mode KV cache has no host offset to serialize")
        return KVCache.meta_state.fget(self)

    @meta_state.setter
    def meta_state(self, v):
        if self.device_mode:
            raise RuntimeError("a device-mode KV cache's offset is not host-settable")
        KVCache.meta_state.fset(self, v)

    @property
    def state(self):
        # Device mode: the bounded rows (fill plus slack) for evaluation only.
        return KVCache.state.fget(self)

    @state.setter
    def state(self, v):
        if getattr(self, "device_mode", False):
            raise RuntimeError("a device-mode KV cache's state is not host-settable")
        KVCache.state.fset(self, v)

    def to_quantized(self, *args, **kwargs):
        if self.device_mode:
            raise RuntimeError("quantize a device-mode KV cache only after leave_device_mode()")
        return super().to_quantized(*args, **kwargs)

    @classmethod
    def merge(cls, caches):
        if any(getattr(cache, "device_mode", False) for cache in caches):
            raise RuntimeError("merge device-mode KV caches only after leave_device_mode()")
        return KVCache.merge(caches)


__all__ = ["DeviceOffsetKVCache"]
