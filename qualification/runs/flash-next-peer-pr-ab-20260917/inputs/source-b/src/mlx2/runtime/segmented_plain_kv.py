"""Segmented compute view over independently owned ordinary KV rows.

Batch projections and trunk execution do not require joining historical KV.
The generic attention seam calls ``bucketed_attention`` to reduce each row's
private history. This is an exact state mechanism; GPU qualification remains
separate from these implementation contracts.
"""

from __future__ import annotations

from numbers import Integral
import mlx.core as mx
from .models.cache import KVCache
from .models.base import create_causal_mask


class SegmentedBatchKVCache:
    """Batched attention ABI with B1 KVCache objects as authoritative storage."""

    def __init__(self, rows, note=None):
        if not rows or any(type(row) is not KVCache for row in rows):
            raise TypeError("segmented ordinary KV requires plain KVCache rows")
        if len({id(row) for row in rows}) != len(rows):
            raise ValueError("segmented ordinary KV rows must have distinct owners")
        self.rows = list(rows)
        self._note = note
        self.keys = self.values = None  # There is deliberately no joined KV slab.
        self._step_lengths = None
        self._right_padding = None
        self._updated = False
        self._value_dim = None
        self._refresh_geometry()

    def _bump(self, key, amount=1):
        if self._note is not None:
            self._note(key, amount)

    def _host_offsets(self):
        offsets = [row.offset for row in self.rows]
        if any(not isinstance(value, Integral) or value < 0 for value in offsets):
            raise ValueError("segmented ordinary KV requires nonnegative host offsets")
        return [int(value) for value in offsets]

    def _refresh_geometry(self):
        self._base_lengths = self._host_offsets()
        self._base_width = max(self._base_lengths, default=0)
        self._idx = self._base_width
        self.offset = mx.array(self._base_lengths)
        self.left_padding = mx.array([self._base_width - n for n in self._base_lengths])

    @property
    def batch_size(self):
        return len(self.rows)

    @property
    def state(self):
        return tuple(row.state for row in self.rows)

    @property
    def nbytes(self):
        # Authoritative arrays are accounted on the independent row caches.
        return 0

    def prepare(self, *, lengths=None, right_padding=None, **kwargs):
        if kwargs:
            raise ValueError(f"unsupported segmented KV preparation: {sorted(kwargs)}")
        if self._step_lengths is not None:
            raise RuntimeError("segmented KV step is already prepared")
        if lengths is None or len(lengths) != self.batch_size:
            raise ValueError("segmented KV lengths must cover every row")
        if any(not isinstance(n, Integral) or n < 0 for n in lengths):
            raise ValueError("segmented KV lengths must be nonnegative integers")
        lengths = [int(n) for n in lengths]
        width = max(lengths, default=0)
        expected = [width - n for n in lengths]
        if right_padding is not None and list(right_padding) != expected:
            raise ValueError("segmented KV lengths/right padding disagree")
        self._refresh_geometry()
        self._step_lengths = lengths
        self._right_padding = mx.array(expected)
        self._updated = False

    prepare_self_mtp_step = prepare

    def make_mask(self, N, return_array=False, **kwargs):
        del return_array
        if self._step_lengths is None or N != max(self._step_lengths, default=0):
            raise RuntimeError("segmented KV mask requires the prepared query width")
        return create_causal_mask(
            N,
            offset=self._base_width,
            left_padding=self.left_padding,
            right_padding=self._right_padding,
            **kwargs,
        )

    def update_and_fetch(self, keys, values):
        if self._step_lengths is None or self._updated:
            raise RuntimeError("segmented KV append requires one prepared step")
        width = max(self._step_lengths, default=0)
        if (
            keys.ndim != 4
            or values.ndim != 4
            or keys.shape[:3] != values.shape[:3]
            or keys.shape[0] != self.batch_size
            or keys.shape[2] != width
        ):
            raise ValueError("segmented KV append geometry mismatch")
        if self._host_offsets() != self._base_lengths:
            raise RuntimeError("segmented KV row changed after preparation")
        for row, valid in zip(self.rows, self._step_lengths):
            if valid and row.keys is not None:
                if (
                    row.keys.shape[:2] != (1, keys.shape[1])
                    or row.keys.shape[-1] != keys.shape[-1]
                    or row.values.shape[:2] != (1, values.shape[1])
                    or row.values.shape[-1] != values.shape[-1]
                ):
                    raise ValueError("segmented KV row geometry mismatch")
        try:
            for index, (row, valid) in enumerate(zip(self.rows, self._step_lengths)):
                if valid:
                    row.update_and_fetch(
                        mx.contiguous(keys[index : index + 1, :, :valid]),
                        mx.contiguous(values[index : index + 1, :, :valid]),
                    )
        except BaseException:
            # Recover logical positions if a later row rejected its append.
            for row, offset in zip(self.rows, self._base_lengths):
                if row.offset >= offset:
                    row.trim(row.offset - offset)
            raise
        self._updated = True
        self._value_dim = values.shape[-1]
        self.offset = mx.array(self._host_offsets())
        self._bump("row_state_splits", self.batch_size)
        return None, None

    def bucketed_attention(self, queries, scale, mask, *, sinks=None):
        if not self._updated or self._step_lengths is None:
            raise RuntimeError("segmented KV attention requires an appended step")
        width = max(self._step_lengths, default=0)
        if (
            queries.ndim != 4
            or queries.shape[0] != self.batch_size
            or queries.shape[2] != width
        ):
            raise ValueError("segmented KV query geometry mismatch")
        if isinstance(mask, str):
            raise ValueError(
                "segmented KV requires an explicit prepared attention mask"
            )
        outputs = []
        for index, (row, valid, base) in enumerate(
            zip(self.rows, self._step_lengths, self._base_lengths)
        ):
            if not valid:
                outputs.append(
                    mx.zeros(
                        (1, queries.shape[1], width, self._value_dim),
                        dtype=queries.dtype,
                    )
                )
                continue
            left = self._base_width - base
            row_mask = (
                create_causal_mask(valid, offset=base)
                if mask is None
                else mask[index : index + 1, :, :valid, left : left + base + valid]
            )
            keys, values = row.keys_and_values()
            output = mx.fast.scaled_dot_product_attention(
                queries[index : index + 1, :, :valid],
                keys,
                values,
                scale=scale,
                mask=row_mask,
                sinks=sinks,
            )
            if valid < width:
                output = mx.pad(output, [(0, 0), (0, 0), (0, width - valid), (0, 0)])
            outputs.append(output)
        self._bump("segmented_attention_calls")
        self._bump("independent_lineages_consumed", self.batch_size)
        return mx.concatenate(outputs, axis=0)

    def last_valid_query(self, values):
        if self._step_lengths is None or any(n == 0 for n in self._step_lengths):
            raise ValueError("last valid query requires a nonempty prepared row")
        rows = mx.arange(self.batch_size, dtype=mx.int32)
        return values[rows, mx.array(self._step_lengths, dtype=mx.int32) - 1]

    def max_left_padding(self):
        return max((self._base_width - n for n in self._base_lengths), default=0)

    def finalize(self):
        self._step_lengths = None
        self._right_padding = None
        self._updated = False
        self._refresh_geometry()

    finalize_self_mtp_step = finalize

    def supports_ragged_trim(self):
        return True

    def preflight_ragged_trim(self, counts, *, validate=True):
        del validate  # Cheap host bounds are always checked before any mutation.
        if len(counts) != self.batch_size:
            raise ValueError("segmented KV trim must cover every row")
        if any(not isinstance(n, Integral) or n < 0 for n in counts):
            raise ValueError("segmented KV trim counts must be nonnegative integers")
        counts = [int(n) for n in counts]
        if any(n > offset for n, offset in zip(counts, self._host_offsets())):
            raise ValueError("segmented KV trim exceeds row offset")
        return counts

    def trim_ragged(self, counts, *, validate=True):
        counts = self.preflight_ragged_trim(counts, validate=validate)
        for row, count in zip(self.rows, counts):
            if count:
                if row.trim(count) != count:
                    raise RuntimeError("segmented KV row trim diverged")
        self.finalize()
        return counts

    def trim(self, count):
        self.trim_ragged([count] * self.batch_size)
        return int(count)

    def extract(self, index):
        """Return a standalone snapshot; authoritative original ownership stays put."""
        row = self.rows[index]
        result = KVCache()
        if row.offset:
            keys, values = row.keys_and_values()
            result.update_and_fetch(mx.array(keys), mx.array(values))
        return result

    def filter(self, indices):
        if self._step_lengths is not None:
            raise RuntimeError("cannot change segmented KV membership during a step")
        indices = list(indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("segmented KV membership must be nonempty and unique")
        if any(
            not isinstance(i, Integral) or not 0 <= i < self.batch_size for i in indices
        ):
            raise ValueError("invalid segmented KV row index")
        self.rows = [self.rows[i] for i in indices]
        self._refresh_geometry()

    def is_trimmable(self):
        return True

    def empty(self):
        return all(row.empty() for row in self.rows)
