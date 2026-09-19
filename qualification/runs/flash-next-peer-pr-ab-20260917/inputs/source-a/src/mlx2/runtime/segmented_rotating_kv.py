"""Exact speculative KV transactions over independent global/sliding rows.

Verification may evict a ring token needed after rejection. A transaction owns
an immutable pre-forward snapshot and the appended K/V, then restores/replays
the accepted input prefix. Offset-only or shared-batch ring rewind is never used.
This CPU-qualified mechanism uses per-row SDPA; native GPU qualification is
separate. Snapshot memory is explicit and must be included in admission.
"""

from __future__ import annotations

import copy
from numbers import Integral
import mlx.core as mx

from .models.cache import KVCache, RotatingKVCache


def _counts(values, size, label):
    values = list(values)
    if len(values) != size or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 0 for n in values):
        raise ValueError(f"{label} requires one nonnegative integer per lane")
    return tuple(map(int, values))


def _stamp(cache):
    return (id(cache), id(cache.keys), id(cache.values), int(cache.offset),
            getattr(cache, "_idx", None), getattr(cache, "max_size", None),
            getattr(cache, "keep", None))


def _signature(cache):
    return (type(cache), getattr(cache, "max_size", None), getattr(cache, "keep", None))


class _RowMask:
    def __init__(self, geometry, window_size):
        self.geometry = geometry
        self.window_size = window_size


def clone_kv_row(caches):
    """Independent canonical snapshot for an idle target checkpoint/sidecar."""
    row = list(caches)
    SegmentedKVRows._validate([row])
    return copy.deepcopy(row)


class SegmentedKVRows:
    """Exclusive lane ownership and membership revision for ordinary KV caches."""

    def __init__(self, rows, note=None):
        self.rows = [list(row) for row in rows]
        self.note = note
        self.revision = 0
        self._active = None
        self._validate(self.rows)

    @staticmethod
    def _validate(rows):
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("KV lanes need the same nonzero layer count")
        leaves = [leaf for row in rows for leaf in row]
        if len({id(leaf) for leaf in leaves}) != len(leaves):
            raise ValueError("KV lanes/layers must have independent cache owners")
        for row in rows:
            if any(type(cache) not in (KVCache, RotatingKVCache) for cache in row):
                raise TypeError("segmented transactions support plain and rotating KV only")
            if len({cache.offset for cache in row}) != 1:
                raise ValueError("all layers in a lane must have equal logical offsets")
            if any(isinstance(cache.offset, bool) or not isinstance(cache.offset, Integral)
                   or cache.offset < 0 or getattr(cache, "speculating", False) for cache in row):
                raise ValueError("KV lanes require host offsets and no other speculation owner")
            if any(cache.keys is not None and cache.keys.shape[0] != 1 for cache in row):
                raise ValueError("each authoritative cache must be a B1 row")
            if any(type(cache) is RotatingKVCache and
                   (not isinstance(cache.max_size, Integral) or cache.max_size <= 0
                    or not isinstance(cache.keep, Integral) or not 0 <= cache.keep < cache.max_size)
                   for cache in row):
                raise ValueError("invalid rotating cache window/sink geometry")
        for layer in zip(*rows):
            if len({_signature(cache) for cache in layer}) != 1:
                raise ValueError("corresponding cache layers must have equal geometry")

    def _idle(self):
        if self._active is not None:
            raise RuntimeError("KV membership is leased by an active transaction")

    def begin(self, lengths):
        self._idle()
        self._validate(self.rows)
        lengths = _counts(lengths, len(self.rows), "verification lengths")
        if not any(lengths):
            raise ValueError("verification must contain at least one token")
        transaction = SegmentedKVTransaction(self, lengths)
        self._active = transaction
        return transaction

    def filter(self, indices):
        self._idle()
        indices = list(indices)
        if not indices or len(set(indices)) != len(indices) or any(
            isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < len(self.rows)
            for i in indices
        ):
            raise ValueError("membership requires unique valid lane indices")
        self.rows = [self.rows[i] for i in indices]
        self.revision += 1

    def extend(self, rows):
        self._idle()
        added = [list(row) for row in rows]
        self._validate(self.rows + added)
        self.rows.extend(added)
        self.revision += 1

    def extract(self, index):
        self._idle()
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self.rows):
            raise ValueError("invalid lane index")
        return copy.deepcopy(self.rows[index])

    @property
    def nbytes(self):
        current = sum(cache.nbytes for row in self.rows for cache in row)
        return current + (self._active.retained_nbytes if self._active is not None else 0)


class SegmentedKVTransaction:
    def __init__(self, owner, lengths):
        self.owner = owner
        self.revision = owner.revision
        self.lengths = lengths
        self.closed = False
        self._rows = [list(row) for row in owner.rows]
        self._snapshots = copy.deepcopy(self._rows)
        self.snapshot_nbytes = sum(cache.nbytes for row in self._snapshots for cache in row)
        self._expected = [[_stamp(cache) for cache in row] for row in self._rows]
        self.caches = [SegmentedKVView(self, layer) for layer in range(len(self._rows[0]))]
        if owner.note is not None:
            owner.note("snapshot_bytes", self.snapshot_nbytes)

    @property
    def retained_nbytes(self):
        return self.snapshot_nbytes + sum(
            array.nbytes for view in self.caches for pair in view._appends
            if pair is not None for array in pair)

    def _check(self):
        if self.closed or self.owner._active is not self or self.owner.revision != self.revision:
            raise RuntimeError("stale or closed KV transaction")
        if len(self.owner.rows) != len(self._rows) or any(
            len(live) != len(original) or any(a is not b for a, b in zip(live, original))
            for live, original in zip(self.owner.rows, self._rows)
        ):
            raise RuntimeError("KV lane membership changed during verification")
        if any(_stamp(cache) != expected for row, stamps in zip(self._rows, self._expected)
               for cache, expected in zip(row, stamps)):
            raise RuntimeError("KV state changed outside its transaction")

    def _restore(self, lane, layer):
        cache = self._rows[lane][layer]
        cache.__dict__.clear()
        cache.__dict__.update(copy.deepcopy(self._snapshots[lane][layer].__dict__))
        self._expected[lane][layer] = _stamp(cache)

    def _close(self):
        self.owner.revision += 1
        self.owner._active = None
        self.closed = True
        self._snapshots = []
        self.snapshot_nbytes = 0
        for cache in self.caches:
            cache._release()
        self._rows = []
        self.owner = None

    def commit(self, accepted_lengths):
        """Publish exact verified-input prefix counts, independently per lane."""
        self._check()
        accepted = _counts(accepted_lengths, len(self._rows), "accepted lengths")
        if any(a > length for a, length in zip(accepted, self.lengths)):
            raise ValueError("accepted prefix exceeds verified input length")
        if any(not cache._updated for cache in self.caches):
            raise RuntimeError("every target layer must finish verification before commit")
        try:
            for lane, count in enumerate(accepted):
                if count == self.lengths[lane]:
                    continue
                for layer, view in enumerate(self.caches):
                    self._restore(lane, layer)
                    if count:
                        keys, values = view._appends[lane]
                        self._rows[lane][layer].update_and_fetch(
                            keys[..., :count, :], values[..., :count, :])
                    self._expected[lane][layer] = _stamp(self._rows[lane][layer])
        except BaseException:
            for lane in range(len(self._rows)):
                for layer in range(len(self.caches)):
                    self._restore(lane, layer)
            self._close()
            raise
        if self.owner.note is not None:
            self.owner.note("committed_input_tokens", sum(accepted))
            self.owner.note("rejected_input_tokens", sum(self.lengths) - sum(accepted))
        rows = self.owner.rows
        self._close()
        return rows

    def abort(self):
        """Cancel a partial or complete forward and restore pre-round state."""
        self._check()
        for lane in range(len(self._rows)):
            for layer in range(len(self.caches)):
                self._restore(lane, layer)
        if self.owner.note is not None:
            self.owner.note("aborted_rounds", 1)
        rows = self.owner.rows
        self._close()
        return rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if not self.closed:
            self.abort()


class SegmentedKVView:
    """One layer's batched projection ABI; attention reduces independent rows."""

    def __init__(self, transaction, layer):
        self.transaction = transaction
        self.layer = layer
        self.rows = [row[layer] for row in transaction._rows]
        self._before = [row[layer] for row in transaction._snapshots]
        self.lengths = transaction.lengths
        self.width = max(self.lengths)
        self.offset = mx.array([cache.offset for cache in self.rows])
        self._geometry = (_signature(self.rows[0]), tuple(cache.offset for cache in self.rows), self.lengths)
        self._appends = [None] * len(self.rows)
        self._fetched = [None] * len(self.rows)
        self._updated = False
        self.keys = self.values = None

    @property
    def nbytes(self):
        return 0  # Authoritative row/snapshot memory is charged on the owner.

    def _check(self):
        if self.transaction is None:
            raise RuntimeError("closed KV compute view")
        self.transaction._check()

    def make_mask(self, N, return_array=False, window_size=None):
        del return_array
        self._check()
        if N != self.width:
            raise ValueError("mask width must match the verification block")
        return _RowMask(self._geometry, window_size)

    def update_and_fetch(self, keys, values):
        self._check()
        if self._updated:
            raise RuntimeError("a transaction layer can append only once")
        if keys.ndim != 4 or values.ndim != 4 or keys.shape[:3] != values.shape[:3] or keys.shape[0] != len(self.rows) or keys.shape[2] != self.width:
            raise ValueError("segmented KV append geometry mismatch")
        for row in self.rows:
            if row.keys is not None and (
                row.keys.shape[:2] != (1, keys.shape[1]) or row.keys.shape[-1] != keys.shape[-1]
                or row.values.shape[:2] != (1, values.shape[1]) or row.values.shape[-1] != values.shape[-1]
            ):
                raise ValueError("segmented KV existing row geometry mismatch")
        try:
            for index, (row, count) in enumerate(zip(self.rows, self.lengths)):
                if count:
                    pair = (mx.array(keys[index:index + 1, :, :count]),
                            mx.array(values[index:index + 1, :, :count]))
                    self._appends[index] = pair
                    self._fetched[index] = row.update_and_fetch(*pair)
                    self.transaction._expected[index][self.layer] = _stamp(row)
        except BaseException:
            # Restore every row of this layer, including a partially mutated row.
            for index in range(len(self.rows)):
                self.transaction._restore(index, self.layer)
            self._appends = [None] * len(self.rows)
            self._fetched = [None] * len(self.rows)
            raise
        self._updated = True
        self._value_dim = values.shape[-1]
        self.offset = mx.array([row.offset for row in self.rows])
        return None, None

    def bucketed_attention(self, queries, scale, mask, *, sinks=None):
        self._check()
        if not self._updated or queries.ndim != 4 or queries.shape[0] != len(self.rows) or queries.shape[2] != self.width:
            raise RuntimeError("segmented attention requires its verified append")
        if not isinstance(mask, _RowMask) or mask.geometry != self._geometry:
            raise ValueError("attention mask does not match this layer's lane geometry")
        outputs = []
        for index, count in enumerate(self.lengths):
            if not count:
                result = mx.zeros((1, queries.shape[1], self.width, self._value_dim), dtype=queries.dtype)
            else:
                row_mask = self._before[index].make_mask(count, window_size=mask.window_size, return_array=True)
                keys, values = self._fetched[index]
                result = mx.fast.scaled_dot_product_attention(
                    queries[index:index + 1, :, :count], keys, values,
                    scale=scale, mask=row_mask, sinks=sinks)
                if count < self.width:
                    result = mx.pad(result, [(0, 0), (0, 0), (0, self.width - count), (0, 0)])
            outputs.append(result)
        if self.transaction.owner.note is not None:
            self.transaction.owner.note("independent_attention_rows", len(self.rows))
        return mx.concatenate(outputs, axis=0)

    def _release(self):
        self._before = []
        self._appends = []
        self._fetched = []
        self.rows = []
        self.offset = None
        self.transaction = None
