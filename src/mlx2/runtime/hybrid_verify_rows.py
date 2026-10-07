"""Verify transactions over B1 lanes whose caches mix recurrent and KV state.

Original mlx2 implementation.  It gives ``ExternalDraftBatchGenerator`` the
same ``begin(lengths) -> transaction`` seam that ``SegmentedKVRows`` gives
attention-only targets, for targets whose layers hold gated-delta recurrent
state (``ArraysCache``) beside KV planes, or a KV layer with an explicitly
declared exact transaction contract (such as QSA's raw-key/pooled ledger).

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
pre-forward state they pin).  Abort returns every row to the pre-round
boundary: KV rows trim to their base offsets and recurrent rows take back
the state references held at ``begin`` (a forward replaces recurrent state,
never mutates it, which is what the rollback records' own snapshots rely
on), so an interrupted forward that wrote some layers and not others, or a
plain step without records, still rewinds exactly.  No capability is
qualified by this module.
"""

from __future__ import annotations

from numbers import Integral

from .models.cache import ArraysCache, KVCache


class HybridVerifyUnsupported(TypeError):
    """The lane caches do not declare the required exact transaction contract."""


def _supported_kv(cache):
    # A subclass can add state that ordinary KV trim does not settle.  Require
    # its own declaration: inheriting a parent's flag is not evidence that the
    # subclass's extra planes support the same exact rollback operation.
    return type(cache) is KVCache or (
        isinstance(cache, KVCache)
        and type(cache).__dict__.get("supports_external_verify_transaction") is True
    )


def is_hybrid_rows(rows) -> bool:
    """True for supported rows needing recurrent or declared KV transactions."""
    rows = [list(row) for row in rows]
    if not rows or not rows[0]:
        return False
    specialized = False
    for row in rows:
        for cache in row:
            if isinstance(cache, ArraysCache):
                specialized = True
            elif not _supported_kv(cache):
                return False
            elif type(cache) is not KVCache:
                specialized = True
    return specialized


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
            kinds = {
                ArraysCache if isinstance(c, ArraysCache) else type(c) for c in layer
            }
            if len(kinds) != 1 or not all(
                isinstance(cache, ArraysCache) or _supported_kv(cache)
                for cache in layer
            ):
                raise HybridVerifyUnsupported(
                    "hybrid verify needs ArraysCache, plain KVCache, or explicitly "
                    "declared exact KV transaction layers"
                )
            for cache in layer:
                if isinstance(cache, ArraysCache):
                    if cache.batch_size != 1 and not cache.empty():
                        raise ValueError(
                            "each authoritative recurrent cache must be a B1 row"
                        )
                    if cache.speculating:
                        raise ValueError(
                            "recurrent row already has a speculation owner"
                        )
                else:
                    _kv_offset(cache)
        self.rows = rows
        self._note = note
        self._active = None

    def begin(self, lengths, *, plain_single_token=False):
        """Lease the rows to one verify forward of ``lengths`` tokens per row.

        ``plain_single_token`` lets a round whose every row is one token skip
        the rollback epoch: commit then accepts exactly that token and trims
        nothing, so the forward is an ordinary decode step and takes the
        plain recurrent path instead of the verify kernel and its records.
        """
        if self._active is not None and not self._active.closed:
            raise RuntimeError("hybrid lanes are leased by an active transaction")
        self._active = HybridVerifyTransaction(
            self, lengths, plain_single_token=plain_single_token
        )
        return self._active


class HybridVerifyTransaction:
    def __init__(self, owner, lengths, *, plain_single_token=False):
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
        # The pre-round recurrent state, by reference (see the module note).
        self._recurrent_base = [
            (cache, list(cache.cache), cache.lengths, cache.left_padding)
            for row in rows
            for cache in row
            if isinstance(cache, ArraysCache)
        ]
        self.plain = bool(plain_single_token) and self.width == 1
        try:
            for row in rows:
                for cache in row:
                    if isinstance(cache, ArraysCache) and not self.plain:
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
                if self.plain:
                    # The batched recurrent view assumes a verify epoch; a
                    # plain step writes state through it without records.
                    for cache in self.caches:
                        if isinstance(cache, ArraysCache):
                            cache.speculating = False
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

    def _close(self):
        """Release the epoch lease and transaction-only cache references.

        The owner points at its active transaction and the transaction points
        back at the owner. Keeping that cycle after commit or abort retains
        every closed request-private branch until cyclic GC runs, which is an
        unbounded context-sized cost for staged continuation verification.
        """

        owner = self.owner
        if owner is not None and owner._active is self:
            owner._active = None
        self.closed = True
        self._kv_base = []
        self._recurrent_base = []
        self.caches = []
        self.owner = None

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
            raise RuntimeError(
                f"hybrid B1 trim diverged: expected {drop}, got {applied}"
            )

    def _check_advanced(self):
        """Every KV row advanced by exactly its verified length."""
        for row, bases, length in zip(self.owner.rows, self._kv_base, self.lengths):
            for cache, base in zip(row, bases):
                if base is not None and _kv_offset(cache) != base + length:
                    raise RuntimeError(
                        "every target layer must finish verification before commit"
                    )

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
        rows = [list(row) for row in self.owner.rows]
        try:
            self._stop()
        finally:
            self._close()
        return rows

    def abort(self):
        """Rewind every row to the pre-round boundary, KV and recurrent alike.

        Every row is attempted; the first failure is raised afterwards, so a
        caller never takes a partly rewound lane for an exact one.
        """
        if self.closed:
            return
        self.closed = True
        first = None
        try:
            self._finalize()
        except BaseException as error:  # noqa: BLE001 - every row gets its turn
            first = error
        for row, bases in zip(self.owner.rows, self._kv_base):
            for cache, base in zip(row, bases):
                try:
                    if base is not None and _kv_offset(cache) != base:
                        drop = _kv_offset(cache) - base
                        if drop < 0 or int(cache.trim(drop)) != drop:
                            raise RuntimeError(
                                f"hybrid abort could not rewind KV to offset {base}"
                            )
                except BaseException as error:  # noqa: BLE001
                    first = first or error
        for cache, state, lengths, left_padding in self._recurrent_base:
            cache.cache = list(state)
            cache.lengths, cache.left_padding = lengths, left_padding
        try:
            self._stop()
        except BaseException as error:  # noqa: BLE001
            first = first or error
        finally:
            self._close()
        if first is not None:
            raise first


__all__ = [
    "HybridVerifyRows",
    "HybridVerifyTransaction",
    "HybridVerifyUnsupported",
    "is_hybrid_rows",
]
