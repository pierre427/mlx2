"""Batched compute view over independently owned sliding-window ring rows.

Original mlx2 implementation (2026-11-26).  It is the ring counterpart of
``segmented_plain_kv.SegmentedBatchKVCache`` for ``HybridVerifyRows``: a
multi-lane verify round over lanes whose caches hold sliding-window rings
(``RotatingKVCache``) beside recurrent state (Inkling's local layers).

Nothing new is invented for the rollback.  Every authoritative row stays a
B1 ring that its owner put in a speculation epoch (``start_speculation``)
before the round, exactly as on the B1 hybrid path.  The view appends each
lane's valid verify rows to that lane's own ring (the ring records its exact
pre-forward window), and every rewind (``trim_ragged``, the owner's abort and
``release_row``) is the ring's own record-and-replay ``trim``, so a lane's
committed window is the B1 path's byte for byte, wrapped or not.  There is no
joined ring slab and no shared-batch offset rewind.

Attention reduces each lane's own history (``bucketed_attention`` with the
lane's B1 mask from its pre-round geometry, as ``SegmentedKVView`` does), or
an attention module that is not plain SDPA runs its own B1 code per lane
through ``row_attention``.  Lane outputs are right-padded to the round width.
No capability is qualified by this module; GPU qualification is separate.
"""

from __future__ import annotations

from numbers import Integral

import mlx.core as mx

from .models.cache import RotatingKVCache
from .segmented_rotating_kv import _mask_state


def _lane_outputs(rows, lengths, bases, fn, axis):
    """Per-lane outputs of ``fn(index, row, valid)``, padded and concatenated.

    Checks every lane appended exactly its ``valid`` rows and returned them
    on ``axis``; shared by the ring and plain KV batched views."""
    width = max(lengths, default=0)
    outputs, empty = [], []
    for index, (row, valid) in enumerate(zip(rows, lengths)):
        if not valid:
            empty.append(index)
            outputs.append(None)
            continue
        out = fn(index, row, valid)
        if out.ndim < 2 or out.shape[0] != 1 or out.shape[axis] != valid:
            raise ValueError("segmented lane attention returned the wrong shape")
        if row.offset != bases[index] + valid:
            raise RuntimeError("segmented lane attention did not append its rows")
        if valid < width:
            pad = [(0, 0)] * out.ndim
            pad[axis] = (0, width - valid)
            out = mx.pad(out, pad)
        outputs.append(out)
    if empty:
        template = next(o for o in outputs if o is not None)
        for index in empty:
            outputs[index] = mx.zeros_like(template)
    return mx.concatenate(outputs, axis=0)


class _RingRowMask:
    """Marker mask: each lane builds its own mask (``row_mask``).

    It carries only the window and the prepared lane lengths, so a model may
    build it once and pass it to every ring layer of the same step (each view
    rebuilds lane masks from its own pre-round geometry)."""

    def __init__(self, lengths, window_size):
        self.lengths = tuple(lengths)
        self.window_size = window_size


class SegmentedBatchRotatingKVCache:
    """Batched attention ABI with speculating B1 rings as authoritative storage."""

    def __init__(self, rows, note=None):
        rows = list(rows)
        if not rows or any(type(row) is not RotatingKVCache for row in rows):
            raise TypeError("segmented ring view requires RotatingKVCache rows")
        if len({id(row) for row in rows}) != len(rows):
            raise ValueError("segmented ring rows must have distinct owners")
        if len({(row.max_size, row.keep) for row in rows}) != 1:
            raise ValueError("segmented ring rows must share one window geometry")
        for row in rows:
            if row.keys is not None and row.keys.shape[0] != 1:
                raise ValueError("each authoritative ring must be a B1 row")
        self.rows = rows
        self.max_size = rows[0].max_size
        self.keep = rows[0].keep
        self._note = note
        self.keys = self.values = None  # There is deliberately no joined ring slab.
        self._step_lengths = None
        self._last_lengths = None
        self._before = None
        self._fetched = None
        self._deferred = None
        self._updated = False
        self._value_dim = None
        self._refresh_geometry()

    # ----- geometry -----

    def _bump(self, key, amount=1):
        if self._note is not None:
            self._note(key, amount)

    def _host_offsets(self):
        offsets = [row.offset for row in self.rows]
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in offsets):
            raise ValueError("segmented ring view requires nonnegative host offsets")
        return [int(v) for v in offsets]

    def _refresh_geometry(self):
        self._base_lengths = self._host_offsets()
        self.offset = mx.array(self._base_lengths)
        # Logical (unpadded) history per lane; rings hold at most a window of it.
        self.left_padding = mx.zeros((len(self.rows),), dtype=mx.int32)

    @property
    def batch_size(self):
        return len(self.rows)

    @property
    def state(self):
        return tuple(row.state for row in self.rows)

    @property
    def nbytes(self):
        # Rings, their rollback records and snapshots are charged on the rows.
        return 0

    @property
    def speculating(self):
        return all(row.speculating for row in self.rows)

    @property
    def step_width(self):
        return max(self._step_lengths or (), default=0)

    # ----- one prepared step -----

    def prepare(self, *, lengths=None, right_padding=None, **kwargs):
        if kwargs:
            raise ValueError(f"unsupported segmented ring preparation: {sorted(kwargs)}")
        if self._step_lengths is not None:
            raise RuntimeError("segmented ring step is already prepared")
        if lengths is None or len(lengths) != self.batch_size:
            raise ValueError("segmented ring lengths must cover every row")
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in lengths):
            raise ValueError("segmented ring lengths must be nonnegative integers")
        lengths = [int(n) for n in lengths]
        width = max(lengths, default=0)
        if right_padding is not None and list(right_padding) != [width - n for n in lengths]:
            raise ValueError("segmented ring lengths/right padding disagree")
        if not all(row.speculating for row in self.rows):
            # Without a speculation epoch a wrapped ring cannot be rewound.
            raise RuntimeError("segmented ring rows must be speculating before a step")
        self._refresh_geometry()
        self._step_lengths = lengths
        self._last_lengths = list(lengths)
        # Pre-round geometry only (no payload): each lane's B1 mask.
        self._before = [_mask_state(row) for row in self.rows]
        self._fetched = [None] * self.batch_size
        self._deferred = [False] * self.batch_size
        self._updated = False

    prepare_self_mtp_step = prepare

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        del return_array
        if kwargs:
            raise ValueError(f"unsupported segmented ring mask options: {sorted(kwargs)}")
        if self._step_lengths is None or N != self.step_width:
            raise RuntimeError("segmented ring mask requires the prepared query width")
        return _RingRowMask(self._step_lengths, window_size)

    def row_mask(self, index, mask=None):
        """Lane ``index``'s mask for its valid rows.

        The B1 mask from the lane's pre-round ring, except for a lane whose
        rows were appended deferred: its fetched K/V are chronological (the
        concatenation the rollback path would hold), so its mask is the
        chronological causal/window mask over them. (A B1 one-token mask is
        built for the ring's physical slot order, which a chronological view
        does not have once the attention window is smaller than the ring.)
        """
        if self._before is None:
            raise RuntimeError("segmented ring mask requires a prepared step")
        count = self._step_lengths[index]
        window = mask.window_size if isinstance(mask, _RingRowMask) else None
        if self._deferred[index]:
            from .models.base import create_causal_mask

            history = self._fetched[index][0].shape[2]
            return create_causal_mask(count, history - count,
                                      window_size=window or self.max_size)
        return self._before[index].make_mask(count, window_size=window, return_array=True)

    def _check_append(self, keys, values):
        if self._step_lengths is None or self._updated:
            raise RuntimeError("segmented ring append requires one prepared step")
        if (
            keys.ndim != 4
            or values.ndim != 4
            or keys.shape[:3] != values.shape[:3]
            or keys.shape[0] != self.batch_size
            or keys.shape[2] != self.step_width
        ):
            raise ValueError("segmented ring append geometry mismatch")
        if self._host_offsets() != self._base_lengths:
            raise RuntimeError("segmented ring row changed after preparation")
        for row, valid in zip(self.rows, self._step_lengths):
            if valid and row.keys is not None and (
                row.keys.shape[:2] != (1, keys.shape[1])
                or row.keys.shape[-1] != keys.shape[-1]
                or row.values.shape[:2] != (1, values.shape[1])
                or row.values.shape[-1] != values.shape[-1]
            ):
                raise ValueError("segmented ring row geometry mismatch")

    def _rewind_partial(self):
        """Undo this step's appends on every row (rings are speculating)."""
        for row, base in zip(self.rows, self._base_lengths):
            if row.offset > base:
                row.trim(row.offset - base)

    def _appended(self, value_dim):
        if self._host_offsets() != [b + n for b, n in zip(self._base_lengths, self._step_lengths)]:
            raise RuntimeError("segmented ring rows did not advance by their step lengths")
        self._updated = True
        self._value_dim = value_dim
        self.offset = mx.array(self._host_offsets())
        self._bump("ring_row_appends", self.batch_size)

    def update_and_fetch(self, keys, values, *, deferred=False):
        """Append each lane's valid rows to its own ring; no joined result.

        ``deferred`` passes through to every ring (it refuses what it cannot
        take). Attention then reads lanes through ``bucketed_attention`` or
        ``row_views``.
        """
        self._check_append(keys, values)
        try:
            for index, (row, valid) in enumerate(zip(self.rows, self._step_lengths)):
                if valid:
                    k = mx.contiguous(keys[index : index + 1, :, :valid])
                    v = mx.contiguous(values[index : index + 1, :, :valid])
                    # A ring already holding deferred rows this epoch stays
                    # deferred whatever the caller asks (the cache's rule).
                    lane_deferred = bool(deferred or row.deferred_rows)
                    fetched = (row.update_and_fetch(k, v, deferred=True) if deferred
                               else row.update_and_fetch(k, v))
                    self._fetched[index] = fetched
                    self._deferred[index] = lane_deferred
        except BaseException:
            self._rewind_partial()
            self._fetched = [None] * self.batch_size
            raise
        self._appended(values.shape[-1])
        return None, None

    def row_attention(self, fn, *, axis=2):
        """Run ``fn(index, ring, valid)`` per lane with rows to verify.

        ``fn`` appends exactly ``valid`` rows to ``ring`` itself (through its
        own B1 code path, deferred or not) and returns that lane's output with
        ``valid`` rows on ``axis`` (2 for ``[1, heads, rows, dim]``, 1 for
        ``[1, rows, hidden]``). Lanes are zero-padded to the round width on
        that axis and concatenated on axis 0. Any failure rewinds every
        lane's appends.
        """
        if self._step_lengths is None or self._updated:
            raise RuntimeError("segmented ring attention requires one prepared step")
        if self._host_offsets() != self._base_lengths:
            raise RuntimeError("segmented ring row changed after preparation")
        try:
            out = _lane_outputs(self.rows, self._step_lengths, self._base_lengths, fn, axis)
        except BaseException:
            self._rewind_partial()
            raise
        self._appended(None)
        self._bump("ring_row_attention_calls")
        return out

    def row_views(self, mask):
        """Yield ``(index, valid, keys, values, row_mask)`` per lane.

        Each lane's post-append K/V as its ring returned them and the lane's
        own B1 mask from its pre-round geometry. Lanes with nothing to verify
        yield ``valid == 0`` and ``None`` arrays.
        """
        if not self._updated or self._step_lengths is None:
            raise RuntimeError("segmented ring attention requires an appended step")
        if not isinstance(mask, _RingRowMask) or mask.lengths != tuple(self._step_lengths):
            raise ValueError("segmented ring attention needs a mask of this step's lane lengths")
        for index, valid in enumerate(self._step_lengths):
            if not valid or self._fetched[index] is None:
                yield index, valid, None, None, None
                continue
            keys, values = self._fetched[index]
            yield index, valid, keys, values, self.row_mask(index, mask)

    def note_attention(self):
        self._bump("segmented_attention_calls")
        self._bump("independent_lineages_consumed", self.batch_size)

    def bucketed_attention(self, queries, scale, mask, *, sinks=None):
        width = self.step_width
        if queries.ndim != 4 or queries.shape[0] != self.batch_size or queries.shape[2] != width:
            raise ValueError("segmented ring query geometry mismatch")
        outputs = []
        for index, valid, keys, values, row_mask in self.row_views(mask):
            if not valid:
                outputs.append(mx.zeros((1, queries.shape[1], width, self._value_dim),
                                        dtype=queries.dtype))
                continue
            if keys is None:
                raise RuntimeError("segmented ring lane was appended outside update_and_fetch")
            out = mx.fast.scaled_dot_product_attention(
                queries[index : index + 1, :, :valid], keys, values,
                scale=scale, mask=row_mask, sinks=sinks)
            if valid < width:
                out = mx.pad(out, [(0, 0), (0, 0), (0, width - valid), (0, 0)])
            outputs.append(out)
        self.note_attention()
        return mx.concatenate(outputs, axis=0)

    def last_valid_query(self, values):
        if self._step_lengths is None or any(n == 0 for n in self._step_lengths):
            raise ValueError("last valid query requires a nonempty prepared row")
        rows = mx.arange(self.batch_size, dtype=mx.int32)
        return values[rows, mx.array(self._step_lengths, dtype=mx.int32) - 1]

    def max_left_padding(self):
        return 0

    def finalize(self):
        """End the step. The rings keep their records for the owner's trim."""
        self._step_lengths = None
        self._before = None
        # Fetched slices pin the ring buffers; a replaying trim would copy them.
        self._fetched = None
        self._deferred = None
        self._updated = False
        self._refresh_geometry()

    finalize_self_mtp_step = finalize

    # ----- rewind: each ring's own record-and-replay trim -----

    def supports_ragged_trim(self):
        return True

    def is_trimmable(self):
        return all(row.is_trimmable() for row in self.rows)

    def preflight_ragged_trim(self, counts, *, validate=True):
        del validate  # Host bounds are always checked before any mutation.
        if len(counts) != self.batch_size:
            raise ValueError("segmented ring trim must cover every row")
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in counts):
            raise ValueError("segmented ring trim counts must be nonnegative integers")
        counts = [int(n) for n in counts]
        if self._step_lengths is not None:
            raise RuntimeError("segmented ring trim requires a finalized step")
        last = self._last_lengths or [0] * self.batch_size
        for row, count, appended in zip(self.rows, counts, last):
            if count > appended:
                raise ValueError("segmented ring trim exceeds the rows this step appended")
            if count and not row.speculating:
                raise RuntimeError("segmented ring trim requires a speculating ring")
        return counts

    def trim_ragged(self, counts, *, validate=True):
        counts = self.preflight_ragged_trim(counts, validate=validate)
        for row, count in zip(self.rows, counts):
            if count and int(row.trim(count)) != count:
                raise RuntimeError("segmented ring row trim diverged")
        self._last_lengths = [a - c for a, c in zip(self._last_lengths or counts, counts)]
        self._refresh_geometry()
        return counts

    def trim(self, count):
        self.trim_ragged([count] * self.batch_size)
        return int(count)

    def filter(self, indices):
        if self._step_lengths is not None:
            raise RuntimeError("cannot change segmented ring membership during a step")
        indices = list(indices)
        if not indices or len(set(indices)) != len(indices) or any(
            isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < self.batch_size
            for i in indices
        ):
            raise ValueError("invalid segmented ring row index")
        self.rows = [self.rows[i] for i in indices]
        if self._last_lengths is not None:
            self._last_lengths = [self._last_lengths[i] for i in indices]
        self._refresh_geometry()

    def empty(self):
        return all(row.empty() for row in self.rows)


__all__ = ["SegmentedBatchRotatingKVCache"]
