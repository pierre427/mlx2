"""Verify transactions over B1 lanes whose caches mix recurrent and KV state.

Original mlx2 implementation.  It gives ``ExternalDraftBatchGenerator`` the
same ``begin(lengths) -> transaction`` seam that ``SegmentedKVRows`` gives
attention-only targets, for targets whose layers hold gated-delta recurrent
state (``ArraysCache``) beside plain ``KVCache`` planes.

Nothing new is invented for the state math: rollback is the exact
record-and-replay contract the recurrent layers already stage while a cache
is ``speculating`` (``ArraysCache.record_rollback``/``trim``), and a
multi-lane verify uses the segmented batch views the self-MTP route runs on
the same models (``build_segmented_batch_cache_group``).  Selection is by
cache topology: a lane whose caches are only plain/rotating KV keeps the
``SegmentedKVRows`` owner.

Per round: every recurrent row starts a fresh rollback epoch, the verify
forward records one replay per recurrent layer, commit trims each row to its
consumed prefix and stops speculation (dropping the records and the
pre-forward state they pin).  No capability is qualified by this module.
"""
from __future__ import annotations

from numbers import Integral

from .models.cache import ArraysCache, KVCache


class HybridVerifyUnsupported(TypeError):
    """The lane caches are not recurrent + plain KV rows."""


def is_hybrid_rows(rows) -> bool:
    """True when at least one layer is recurrent and every layer is supported."""
    rows = [list(row) for row in rows]
    if not rows or not rows[0]:
        return False
    recurrent = False
    for row in rows:
        for cache in row:
            if isinstance(cache, ArraysCache):
                recurrent = True
            elif type(cache) is not KVCache:
                return False
    return recurrent


def _kv_offset(cache):
    offset = getattr(cache, "offset", None)
    if not isinstance(offset, Integral):
        raise HybridVerifyUnsupported(
            f"{type(cache).__name__} has no host-readable offset"
        )
    return int(offset)


class HybridVerifyRows:
    """Owner of independent B1 hybrid lanes; lends them to one transaction."""

    def __init__(self, rows, note=None):
        rows = [list(row) for row in rows]
        if not rows or not rows[0] or len({len(row) for row in rows}) != 1:
            raise ValueError("hybrid lanes need the same nonzero layer count")
        if len({id(cache) for row in rows for cache in row}) != sum(map(len, rows)):
            raise ValueError("hybrid lanes/layers must have independent cache owners")
        for layer in zip(*rows):
            kinds = {ArraysCache if isinstance(c, ArraysCache) else type(c) for c in layer}
            if len(kinds) != 1 or kinds.pop() not in (ArraysCache, KVCache):
                raise HybridVerifyUnsupported(
                    "hybrid verify supports ArraysCache and plain KVCache layers only"
                )
            for cache in layer:
                if isinstance(cache, ArraysCache):
                    if cache.batch_size != 1 and not cache.empty():
                        raise ValueError("each authoritative recurrent cache must be a B1 row")
                    if cache.speculating:
                        raise ValueError("recurrent row already has a speculation owner")
                else:
                    _kv_offset(cache)
        self.rows = rows
        self._note = note
        self._active = None

    def begin(self, lengths):
        if self._active is not None and not self._active.closed:
            raise RuntimeError("hybrid lanes are leased by an active transaction")
        self._active = HybridVerifyTransaction(self, lengths)
        return self._active


class HybridVerifyTransaction:
    def __init__(self, owner, lengths):
        from .hybrid_speculative import _prepare_self_mtp_cache_group

        rows = owner.rows
        if len(lengths) != len(rows) or any(
            not isinstance(n, Integral) or n < 1 for n in lengths
        ):
            raise ValueError("verification must contain at least one token per lane")
        self.owner = owner
        self.lengths = [int(n) for n in lengths]
        self.width = max(self.lengths)
        self.closed = False
        self._finalized = False
        self._kv_base = [
            [None if isinstance(c, ArraysCache) else _kv_offset(c) for c in row]
            for row in rows
        ]
        self._started = []
        try:
            for row in rows:
                for cache in row:
                    if isinstance(cache, ArraysCache):
                        cache.start_speculation()
                        self._started.append(cache)
            if len(rows) == 1:
                if self.lengths[0] != self.width:
                    raise AssertionError("single lane has no padding")
                self.batched = False
                self.caches = rows[0]
            else:
                from .segmented_batch_cache import build_segmented_batch_cache_group

                self.batched = True
                self.caches = build_segmented_batch_cache_group(rows, note=owner._note)
                _prepare_self_mtp_cache_group(
                    self.caches,
                    self.lengths,
                    [self.width - n for n in self.lengths],
                )
        except BaseException:
            self._stop()
            self.closed = True
            raise

    def _stop(self):
        first = None
        for cache in self._started:
            try:
                cache.stop_speculation()
            except BaseException as error:  # noqa: BLE001 - every row gets its turn
                if first is None:
                    first = error
        self._started = []
        if first is not None:
            raise first

    def _finalize(self):
        if self.batched and not self._finalized:
            from .hybrid_speculative import _finalize_self_mtp_cache_group

            self._finalized = True
            _finalize_self_mtp_cache_group(self.caches)

    def _trim(self, drops, *, validate):
        if not any(drops):
            return
        if self.batched:
            from .hybrid_speculative import _trim_self_mtp_cache_group

            _trim_self_mtp_cache_group(self.caches, drops, validate=validate)
            return
        (drop,) = drops
        applied = [int(cache.trim(drop)) for cache in self.caches]
        if any(value != drop for value in applied):
            raise RuntimeError(f"hybrid B1 trim diverged: expected {drop}, got {applied}")

    def _check_advanced(self):
        """Every KV row advanced by exactly its verified length."""
        for row, bases, length in zip(self.owner.rows, self._kv_base, self.lengths):
            for cache, base in zip(row, bases):
                if base is not None and _kv_offset(cache) != base + length:
                    raise RuntimeError("every target layer must finish verification before commit")

    def commit(self, accepted_lengths):
        if self.closed:
            raise RuntimeError("stale or closed hybrid transaction")
        accepted = [int(n) for n in accepted_lengths]
        if len(accepted) != len(self.lengths) or any(
            not 1 <= a <= n for a, n in zip(accepted, self.lengths)
        ):
            raise ValueError("accepted prefix exceeds verified input length")
        try:
            self._finalize()
            self._check_advanced()
            self._trim([n - a for n, a in zip(self.lengths, accepted)], validate=True)
        except BaseException:
            self.abort()
            raise
        self.closed = True
        self._stop()
        return [list(row) for row in self.owner.rows]

    def abort(self):
        """Best-effort rewind to the pre-round boundary; recovery snapshots win."""
        if self.closed:
            return
        self.closed = True
        try:
            self._finalize()
        except BaseException:  # noqa: BLE001, S110 - snapshots restore authoritatively
            pass
        try:
            for row, bases in zip(self.owner.rows, self._kv_base):
                for cache, base in zip(row, bases):
                    if base is not None and _kv_offset(cache) > base:
                        cache.trim(_kv_offset(cache) - base)
        except BaseException:  # noqa: BLE001, S110
            pass
        self._stop()


__all__ = [
    "HybridVerifyRows",
    "HybridVerifyTransaction",
    "HybridVerifyUnsupported",
    "is_hybrid_rows",
]
