"""One-token plain decode transaction over a B1 attention-only lane.

Original mlx2 implementation.  It gives a prompt-lookup lane whose caches are
plain ``KVCache`` and stock ``RotatingKVCache`` planes the same
``begin -> commit | release_row | abort`` seam that ``HybridVerifyRows``
gives a hybrid lane for a plain round (``plain_single_token``), so such a lane
can pipeline its proposal-free rounds: the forward runs on the lane's own
caches (the ordinary B1 decode step) and the next round is dispatched before
this one is read.

A one-token append is undone without a snapshot:

* ``KVCache`` rows are append-only; ``trim(1)`` restores the offset and the
  written row lies beyond it.
* A ``RotatingKVCache`` restores its offset and ring index.  Before the ring
  wraps the written slot lies beyond the offset.  Once it has wrapped, the
  append overwrote the ring's oldest row.  No later query reads that row: the
  next one-token append overwrites the same slot before attending, and a
  multi-token append keeps only the newest ``max_size - 1`` rows, dropping it.
  So every later append, mask and attention is bitwise what it would have been
  without the undone step; only that dead slot's bytes differ until the next
  append.  A ring that the append itself trimmed (a post-prefill buffer longer
  than the window) keeps its trimmed buffer, whose index restores to the
  window edge.

No rollback epoch, record or copy is taken, so the step costs exactly the
ordinary decode forward.  Anything else (multi-token rows, quantized or
speculating planes, nested cache lists) stays on ``SegmentedKVRows``.
"""

from __future__ import annotations

from numbers import Integral

from .models.cache import KVCache, RotatingKVCache


def is_plain_kv_row(cache) -> bool:
    """Whether a lane's planes are all plain/rotating KV a plain step can own."""
    if not isinstance(cache, list) or not cache:
        return False
    offsets = set()
    for entry in cache:
        if type(entry) not in (KVCache, RotatingKVCache):
            return False
        if getattr(entry, "speculating", False):
            return False
        offset = entry.offset
        if isinstance(offset, bool) or not isinstance(offset, Integral):
            return False
        if entry.keys is not None and entry.keys.shape[0] != 1:
            return False
        offsets.add(int(offset))
    return len(offsets) == 1


class PlainKVStep:
    """A plain one-token round on one attention-only lane's own caches."""

    plain = True
    batched = False

    def __init__(self, rows):
        rows = [list(row) for row in rows]
        if len(rows) != 1 or not is_plain_kv_row(rows[0]):
            raise TypeError(
                "a plain KV step owns one B1 lane of plain/rotating KV planes"
            )
        self.rows = rows
        self.caches = rows[0]
        self.closed = False
        self._released = set()
        self._base = [
            (int(cache.offset), getattr(cache, "_idx", None)) for cache in self.caches
        ]

    def _undo(self, cache, base):
        offset, index = base
        current = int(cache.offset)
        if current == offset:
            return
        if current != offset + 1:
            raise RuntimeError(
                f"plain KV step cannot rewind offset {current} to {offset}"
            )
        if type(cache) is KVCache:
            if int(cache.trim(1)) != 1:
                raise RuntimeError("plain KV step could not trim its appended row")
            return
        cache.offset = offset
        # A trimmed post-prefill buffer is a window now: its index restores
        # to the window edge, where the next append wraps.
        cache._idx = min(int(index), int(cache.keys.shape[2]))

    def release_row(self, index):
        """Rewind the lane to its pre-round boundary; commit then skips it."""
        if self.closed:
            raise RuntimeError("only an open plain round can release a row")
        index = int(index)
        if index != 0:
            raise ValueError("a plain KV step has exactly one row")
        if index in self._released:
            return
        for cache, base in zip(self.caches, self._base):
            self._undo(cache, base)
        self._released.add(index)

    def commit(self, accepted_lengths):
        if self.closed:
            raise RuntimeError("stale or closed plain KV step")
        accepted = [int(n) for n in accepted_lengths]
        if accepted != [1]:
            raise ValueError("a plain KV step commits exactly its one token")
        if 0 not in self._released and any(
            int(cache.offset) != offset + 1
            for cache, (offset, _index) in zip(self.caches, self._base)
        ):
            self.abort()
            raise RuntimeError(
                "every target layer must finish the plain step before commit"
            )
        rows = self.rows
        self._close()
        return rows

    def abort(self):
        """Rewind every plane that the (possibly partial) forward advanced."""
        if self.closed:
            return
        first = None
        if 0 not in self._released:
            for cache, base in zip(self.caches, self._base):
                try:
                    self._undo(cache, base)
                except BaseException as error:  # noqa: BLE001 - every plane gets its turn
                    first = first or error
        self._close()
        if first is not None:
            raise first

    def _close(self):
        self.closed = True
        self._base = []


__all__ = ["PlainKVStep", "is_plain_kv_row"]
