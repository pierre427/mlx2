# SPDX-License-Identifier: MIT
"""Candidate Xing MLA cache: 8-bit latent and BF16 positional key.

This is an approximate state format.  Callers must select it through a
revision-bound approximate operation and use a distinct APCv2 cache layout;
the ordinary Xing cache remains :class:`KVCache`.  The latent is quantized
per token in 64-channel affine groups.  The RoPE key is never quantized.
"""

from __future__ import annotations

import mlx.core as mx

from ..segmented_plain_kv import SegmentedBatchKVCache
from .cache import KVCache, _BaseCache, create_attention_mask

_COUNTERS = {
    "prefix_conversions": 0,
    "pack_calls": 0,
    "dequant_calls": 0,
    "max_cache_bytes": 0,
}


def latent_kv8_stats(*, reset=False):
    """Allocation-free mechanism counters; no device synchronization."""
    snapshot = dict(_COUNTERS)
    if reset:
        for name in _COUNTERS:
            _COUNTERS[name] = 0
    return snapshot


class XingLatentKV8Cache(_BaseCache):
    """B1 Xing MLA cache with packed latent and BF16 RoPE key.

    ``keys`` is MLX's (uint32, scale, bias) tuple for the 512-wide latent;
    ``values`` is the 64-wide positional key.  The names intentionally match
    the exact Xing KVCache planes, while the class and metadata distinguish
    approximate state from the exact cache.
    """

    step = 256
    bits = key_bits = 8
    value_bits = None  # The positional plane is BF16, never quantized.
    group_size = 64
    _RECOVERY_APPEND_ONLY_FIELDS = (("keys", -2, "offset"), ("values", -2, "offset"))

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    @staticmethod
    def _check_input(latent, rope):
        if (
            latent.ndim != 4
            or rope.ndim != 4
            or latent.shape[:3] != rope.shape[:3]
            or latent.shape[-1] != 512
            or rope.shape[-1] != 64
            or latent.shape[1] != 1
            or latent.dtype != mx.bfloat16
            or rope.dtype != mx.bfloat16
        ):
            raise ValueError("Xing latent KV8 requires BF16 [B, 1, S, 512] latent and BF16 [B, 1, S, 64] RoPE")

    @classmethod
    def from_exact(cls, source: KVCache) -> XingLatentKV8Cache:
        """Privately convert an exact prefix, leaving its APCv2 owner intact."""
        if type(source) is not KVCache:
            raise TypeError("Xing latent KV8 conversion requires an exact KVCache")
        result = cls()
        if source.offset:
            latent, rope = source.keys_and_values()
            cls._check_input(latent, rope)
            result.update_and_fetch(latent, rope)
        _COUNTERS["prefix_conversions"] += 1
        return result

    def update_and_fetch(self, latent, rope):
        self._check_input(latent, rope)
        steps = latent.shape[2]
        previous = self.offset
        if self.keys is not None and (
            self.keys[0].shape[:2] != latent.shape[:2]
            or self.values.shape[:2] != rope.shape[:2]
        ):
            raise ValueError("Xing latent KV8 batch/head geometry changed")
        # Reject mismatched state before changing the logical offset.  This
        # format is fixed; a quantized RoPE value is never a valid append.
        if self.values is not None and self.values.dtype != mx.bfloat16:
            raise ValueError("Xing latent KV8 positional key must be BF16")
        if self.keys is None or previous + steps > self.keys[0].shape[2]:
            grow = (self.step + steps - 1) // self.step * self.step
            shape = (*latent.shape[:2], grow)
            new_keys = (
                mx.zeros((*shape, 128), dtype=mx.uint32),
                mx.zeros((*shape, 8), dtype=latent.dtype),
                mx.zeros((*shape, 8), dtype=latent.dtype),
            )
            new_rope = mx.zeros((*shape, 64), dtype=mx.bfloat16)
            if self.keys is None:
                self.keys, self.values = new_keys, new_rope
            else:
                # Trim the unused tail first so growth cannot keep stale
                # speculative tokens or duplicate reserved capacity.
                self.keys = tuple(
                    mx.concatenate((old[..., :previous, :], new), axis=2)
                    for old, new in zip(self.keys, new_keys)
                )
                self.values = mx.concatenate(
                    (self.values[..., :previous, :], new_rope), axis=2
                )
        quantized = mx.quantize(latent, group_size=64, bits=8)
        _COUNTERS["pack_calls"] += 1
        end = previous + steps
        for target, part in zip(self.keys, quantized):
            target[..., previous:end, :] = part
        self.values[..., previous:end, :] = rope.astype(mx.bfloat16)
        self.offset = end
        _COUNTERS["max_cache_bytes"] = max(
            _COUNTERS["max_cache_bytes"], self.nbytes
        )
        return self.keys_and_values(dtype=latent.dtype)

    def keys_and_values(self, *, dtype=None):
        if self.keys is None:
            return None, None
        packed = tuple(part[..., : self.offset, :] for part in self.keys)
        latent = mx.dequantize(*packed, group_size=64, bits=8)
        _COUNTERS["dequant_calls"] += 1
        if dtype is not None:
            latent = latent.astype(dtype)
        return latent, self.values[..., : self.offset, :]

    @property
    def state(self):
        if self.keys is None:
            return None, None
        return (
            tuple(part[..., : self.offset, :] for part in self.keys),
            self.values[..., : self.offset, :],
        )

    @state.setter
    def state(self, value):
        self.keys, self.values = value
        self.offset = 0 if self.keys is None else self.keys[0].shape[2]

    @property
    def meta_state(self):
        return ("xing-latent-kv8-v1", str(self.offset))

    @meta_state.setter
    def meta_state(self, value):
        if len(value) != 2 or value[0] != "xing-latent-kv8-v1":
            raise ValueError("invalid Xing latent KV8 metadata version")
        offset = int(value[1])
        if offset < 0 or (self.keys is None) != (self.values is None):
            raise ValueError("invalid Xing latent KV8 state")
        if self.keys is None:
            if offset:
                raise ValueError("empty Xing latent KV8 state has nonzero offset")
        elif (
            len(self.keys) != 3
            or self.keys[0].dtype != mx.uint32
            or self.keys[1].shape != self.keys[2].shape
            or self.keys[0].shape[:3] != self.values.shape[:3]
            or self.keys[0].shape[-1] != 128
            or self.keys[1].shape[-1] != 8
            or self.values.shape[-1] != 64
            or self.values.dtype != mx.bfloat16
            or offset != self.keys[0].shape[2]
        ):
            raise ValueError("invalid Xing latent KV8 state geometry")
        self.offset = offset

    def is_trimmable(self):
        return True

    def trim(self, n):
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("trim count must be a nonnegative integer")
        removed = min(n, self.offset)
        self.offset -= removed
        return removed

    def size(self):
        return self.offset

    def empty(self):
        return self.keys is None

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return sum(part.nbytes for part in self.keys) + self.values.nbytes

    @classmethod
    def merge(cls, caches):
        # The ordinary batch cache expects dense, symmetric K/V planes.  A
        # dedicated segmented row adapter is needed before B>1 is selectable.
        raise NotImplementedError("Xing latent KV8 continuous batching needs a segmented MLA cache")

    @classmethod
    def segment_batch(cls, rows, *, note=None):
        """Return the model-owned segmented view through the generic cache hook."""
        return SegmentedBatchXingLatentKV8Cache(rows, note=note)


class SegmentedBatchXingLatentKV8Cache(SegmentedBatchKVCache):
    """Segmented Xing MLA view over independently owned approximate rows."""

    def __init__(self, rows, note=None):
        rows = list(rows)
        if not rows or any(type(row) is not XingLatentKV8Cache for row in rows):
            raise TypeError("segmented Xing latent KV8 requires homogeneous rows")
        if len({id(row) for row in rows}) != len(rows):
            raise ValueError("segmented Xing latent KV8 rows must have distinct owners")
        self.rows = rows
        self._note = note
        self.keys = self.values = None
        self._step_lengths = None
        self._right_padding = None
        self._updated = False
        self._value_dim = None
        self._refresh_geometry()
        self._bump("xing_latent_kv8_segmented_layers")

    def update_and_fetch(self, latent, rope):
        if self._step_lengths is None or self._updated:
            raise RuntimeError("segmented Xing KV8 append requires one prepared step")
        width = max(self._step_lengths, default=0)
        if (
            latent.ndim != 4
            or rope.ndim != 4
            or latent.shape[:3] != rope.shape[:3]
            or latent.shape[0] != self.batch_size
            or latent.shape[2] != width
            or latent.shape[1] != 1
            or latent.shape[-1] != 512
            or rope.shape[-1] != 64
        ):
            raise ValueError("segmented Xing latent KV8 geometry mismatch")
        if self._host_offsets() != self._base_lengths:
            raise RuntimeError("segmented Xing KV8 row changed after preparation")
        try:
            for index, (row, valid) in enumerate(zip(self.rows, self._step_lengths)):
                if valid:
                    row.update_and_fetch(
                        mx.contiguous(latent[index : index + 1, :, :valid]),
                        mx.contiguous(rope[index : index + 1, :, :valid]),
                    )
        except BaseException:
            for row, offset in zip(self.rows, self._base_lengths):
                if row.offset >= offset:
                    row.trim(row.offset - offset)
            raise
        self._updated = True
        self._value_dim = 64
        self.offset = mx.array(self._host_offsets())
        self._bump("row_state_splits", self.batch_size)
        return None, None

    def extract(self, index):
        """A request-private snapshot, retaining the approximate layout."""
        row = self.rows[index]
        result = XingLatentKV8Cache()
        if row.offset:
            packed, rope = row.state
            result.keys = tuple(mx.array(part) for part in packed)
            result.values = mx.array(rope)
            result.offset = row.offset
        return result
