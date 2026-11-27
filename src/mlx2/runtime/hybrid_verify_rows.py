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

A sliding-window ``RotatingKVCache`` beside recurrent state (Inkling's
local layers) uses the ring's own exact record-and-replay rollback: it
speculates for every round, plain rounds included, so ``trim`` restores the
pre-forward window and re-appends the accepted prefix even once the ring has
wrapped.  Batched (B>1) verify over rings is opt-in per owner
(``batched_rings``, which the caller takes from the target's
``supports_batched_ring_verify``; default off, fail closed): each ring layer
gets a ``SegmentedBatchRotatingKVCache`` view whose authoritative rows are
the lanes' own speculating rings, so every rewind is still that B1 ring trim.

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

from .cache_operations import CacheOperationsUnsupported, cache_operations
from .models.cache import RotatingKVCache


class HybridVerifyUnsupported(TypeError):
    """The lane caches do not declare the required exact transaction contract."""


def _supported_kv(cache, *, allow_rings=True):
    contract = cache_operations(
        cache, capability="hybrid verify KV rollback", required=False
    )
    if contract is None or contract.rollback != "kv_trim":
        return False
    if contract.family == "rotating_kv":
        # Only rings that own an exact record/replay rollback (wrapped rings
        # cannot be offset-trimmed).
        return allow_rings and _rotating(cache)
    return contract.family in {"kv", "indexed_kv"}


def _rotating(cache):
    """A sliding-window ring: trimmable after wrap only while speculating."""
    contract = cache_operations(
        cache, capability="hybrid verify ring rollback", required=False
    )
    return (
        contract is not None
        and contract.family == "rotating_kv"
        and callable(getattr(cache, "start_speculation", None))
    )


def _recurrent(cache):
    contract = cache_operations(
        cache, capability="hybrid verify recurrent rollback", required=False
    )
    return contract is not None and contract.rollback == "recurrent_record_trim"


def is_hybrid_rows(rows, *, allow_rings=False) -> bool:
    """True for supported rows needing recurrent or declared KV transactions.

    ``allow_rings`` admits sliding-window rings, and only beside recurrent
    state (B1 unless the owner is built with ``batched_rings``; see the
    module note). It is opt-in per caller: the external
    draft route passes it; prompt lookup does not, so its ring+recurrent
    lanes keep their per-lane rounds."""
    rows = [list(row) for row in rows]
    if not rows or not rows[0]:
        return False
    specialized = False
    rings = recurrent = False
    for row in rows:
        for cache in row:
            if _recurrent(cache):
                specialized = recurrent = True
            elif not _supported_kv(cache, allow_rings=allow_rings):
                return False
            elif _rotating(cache):
                rings = True
            elif cache_operations(
                cache, capability="hybrid verify topology"
            ).family == "indexed_kv":
                specialized = True
    if rings and not recurrent:
        return False  # pure KV + rings keep the segmented owner
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

    def ast_execution_transaction_contract(self):
        """Semantic AST state contract; never a backend-selection hint."""

        from .ast_execution_capability import (
            OWNER_SCHEMA,
            PADDED_RECTANGULAR,
            state_contract_from_rows,
        )

        return {
            "schema": OWNER_SCHEMA,
            **state_contract_from_rows(self.rows),
            "supported_layout_classes": (PADDED_RECTANGULAR,),
            "transaction_operations": {
                "begin": True,
                "commit": True,
                "abort": True,
            },
            "commit_law": "accepted_verified_input_prefix_per_lane",
            "abort_law": "restore_exact_pre_begin_state",
            "rollback_law": "kv_trim_and_recurrent_record_restore",
        }

    def __init__(self, rows, note=None, *, deferred_commit=False, batched_rings=False):
        """``deferred_commit``: arm each ring that offers it
        (``supports_deferred_commit``) for deferred forwards every round; this
        owner rewinds rings only through ``trim``. Default off; the target
        model asks for it (``supports_deferred_verify_commit``).

        ``batched_rings``: admit more than one lane over ring layers (see the
        module note). Default off; the target declares that its attention
        reads ``SegmentedBatchRotatingKVCache`` views
        (``supports_batched_ring_verify``)."""
        rows = [list(row) for row in rows]
        if not rows or not rows[0] or len({len(row) for row in rows}) != 1:
            raise ValueError("hybrid lanes need the same nonzero layer count")
        if len({id(cache) for row in rows for cache in row}) != sum(map(len, rows)):
            raise ValueError("hybrid lanes/layers must have independent cache owners")
        for layer in zip(*rows):
            try:
                kinds = {
                    cache_operations(
                        c, capability="hybrid verify layer binding"
                    ).fingerprint
                    for c in layer
                }
            except CacheOperationsUnsupported as error:
                raise HybridVerifyUnsupported(
                    "hybrid verify needs an explicitly declared exact rollback contract"
                ) from error
            if len(kinds) != 1 or not all(
                _recurrent(cache) or _supported_kv(cache)
                for cache in layer
            ):
                details = [
                    {
                        "type": f"{type(cache).__module__}.{type(cache).__qualname__}",
                        "fingerprint": cache_operations(
                            cache, capability="hybrid verify diagnostic"
                        ).fingerprint,
                        "recurrent": _recurrent(cache),
                        "supported_kv": _supported_kv(cache),
                    }
                    for cache in layer
                ]
                raise HybridVerifyUnsupported(
                    "hybrid verify needs ArraysCache, plain KVCache, or explicitly "
                    f"declared exact KV transaction layers; rows={details!r}"
                )
            if len(rows) > 1 and any(_rotating(cache) for cache in layer):
                if not batched_rings:
                    raise HybridVerifyUnsupported(
                        "batched hybrid verify over rotating KV needs a target that "
                        "reads segmented ring views (supports_batched_ring_verify); B1 only"
                    )
                if any(type(cache) is not RotatingKVCache for cache in layer):
                    raise HybridVerifyUnsupported(
                        "batched hybrid verify over rings supports exact RotatingKVCache rows"
                    )
            for cache in layer:
                if _rotating(cache) and cache.speculating:
                    raise ValueError("rotating row already has a speculation owner")
                if _recurrent(cache):
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
        # Per row and layer: whether the plane is recurrent.  A plane's
        # rollback contract is fixed for its lifetime, and a kept owner opens
        # one transaction per round, so classify once.
        self._recurrent_rows = [[_recurrent(cache) for cache in row] for row in rows]
        self._rotating = [cache for row in rows for cache in row if _rotating(cache)]
        self._deferred_commit = bool(deferred_commit)
        self._batched_rings = bool(batched_rings)
        self._note = note
        self._active = None
        # Per-layer batched views, kept across this owner's rounds while
        # every round commits (``_views``); see ``views``.
        self._views = None

    def holds(self, rows):
        """Whether ``rows`` are exactly this owner's lanes and layers."""
        return len(rows) == len(self.rows) and all(
            len(row) == len(own) and all(a is b for a, b in zip(row, own))
            for row, own in zip(rows, self.rows)
        )

    def views(self, *, plain):
        """The batched compute views over the rows, built once per owner.

        A view kept from the previous round is resynchronized instead of
        rebuilt: KV views re-read their row offsets when prepared, and a
        recurrent view keeps its joined state while every row still holds
        the arrays it split into it (else it re-joins).  Only the plain KV
        and recurrent views are kept; any other layout is rebuilt per round.
        """
        from .segmented_batch_cache import (
            SegmentedBatchArraysCache,
            build_segmented_batch_cache_group,
        )
        from .segmented_plain_kv import SegmentedBatchKVCache

        group = self._views
        if group is None:
            group = self._build_views(build_segmented_batch_cache_group)
            if all(
                type(view) in (SegmentedBatchArraysCache, SegmentedBatchKVCache)
                for view in group
            ):
                self._views = group
        else:
            for view in group:
                if type(view) is SegmentedBatchArraysCache:
                    view.resync(speculating=not plain)
        if plain:
            # The batched recurrent view assumes a verify epoch; a plain
            # step writes state through it without records.
            for view in group:
                if _recurrent(view):
                    view.speculating = False
        return group

    def _build_views(self, build):
        """One batched view per layer: rings get the segmented ring view.

        Ring views are rebuilt every round (they capture the pre-round ring
        geometry at ``prepare``), so an owner with ring layers keeps no views.
        """
        if not any(_rotating(cache) for cache in self.rows[0]):
            return build(self.rows, note=self._note)
        from .segmented_batch_ring import SegmentedBatchRotatingKVCache

        group = []
        for index, layer in enumerate(zip(*self.rows)):
            if _rotating(layer[0]):
                group.append(SegmentedBatchRotatingKVCache(layer, note=self._note))
            else:
                group.extend(build([[row[index]] for row in self.rows], note=self._note))
        return group

    def begin(self, lengths, *, plain_single_token=False):
        if any(cache.speculating for cache in self._rotating):
            raise RuntimeError("rotating row already has a speculation owner")
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
        flags = getattr(owner, "_recurrent_rows", None)
        if flags is None:
            flags = [[_recurrent(cache) for cache in row] for row in rows]
        self._kv_base = [
            [None if recurrent else _kv_offset(c) for c, recurrent in zip(row, row_flags)]
            for row, row_flags in zip(rows, flags)
        ]
        self._started = []
        # The pre-round recurrent state, by reference (see the module note).
        self._recurrent_base = [
            (cache, list(cache.cache), cache.lengths, cache.left_padding)
            for row, row_flags in zip(rows, flags)
            for cache, recurrent in zip(row, row_flags)
            if recurrent
        ]
        self.plain = bool(plain_single_token) and self.width == 1
        # Rows of a plain round rewound and dropped before commit.
        self._released = set()
        try:
            # Rings record every round (plain too): after wrap only the
            # recorded window makes a trim or abort exact.
            rotating = getattr(owner, "_rotating", None)
            if rotating is None:
                rotating = [c for row in rows for c in row if _rotating(c)]
            deferred = getattr(owner, "_deferred_commit", False)
            for cache in rotating:
                if deferred and getattr(type(cache), "supports_deferred_commit", False):
                    cache.start_speculation(deferred_commit=True)
                else:
                    cache.start_speculation()
                self._started.append(cache)
            if not self.plain:
                for cache, *_state in self._recurrent_base:
                    cache.start_speculation()
                    self._started.append(cache)
            if len(rows) == 1:
                if self.lengths[0] != self.width:
                    raise AssertionError("single lane has no padding")
                self.batched = False
                self.caches = rows[0]
            else:
                self.batched = True
                self.caches = owner.views(plain=self.plain)
                # A plain step's recurrent views need no row lengths: every
                # row is one valid token, the ordinary decode geometry.
                self._prepared = [
                    cache for cache in self.caches
                    if not (self.plain and _recurrent(cache))
                ]
                _prepare_self_mtp_cache_group(
                    self._prepared,
                    self.lengths,
                    [self.width - n for n in self.lengths],
                )
        except BaseException:
            owner._views = None
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
            _finalize_self_mtp_cache_group(self._prepared)

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
        for index, (row, bases, length) in enumerate(
            zip(self.owner.rows, self._kv_base, self.lengths)
        ):
            if index in self._released:
                continue
            for cache, base in zip(row, bases):
                if base is not None and _kv_offset(cache) != base + length:
                    raise RuntimeError(
                        "every target layer must finish verification before commit"
                    )

    def release_row(self, index):
        """Rewind row ``index`` of a plain round to its pre-round boundary.

        The row leaves the transaction: commit no longer checks or trims it.
        Only a plain round may release a row, because nothing but the
        pre-round KV offsets and recurrent state references (both held since
        ``begin``) describe it; a pipelined plain round uses this for a lane
        that finished (or failed) on the round before, which it had already
        fed one more anchor.
        """
        if self.closed or not self.plain:
            raise RuntimeError("only an open plain round can release a row")
        index = int(index)
        if index in self._released:
            return
        row = self.owner.rows[index]
        for cache, base in zip(row, self._kv_base[index]):
            if base is not None and _kv_offset(cache) != base:
                drop = _kv_offset(cache) - base
                if drop < 0 or int(cache.trim(drop)) != drop:
                    raise RuntimeError(
                        f"hybrid release could not rewind KV to offset {base}"
                    )
        members = {id(cache) for cache in row}
        for cache, state, lengths, left_padding in self._recurrent_base:
            if id(cache) in members:
                cache.cache = list(state)
                cache.lengths, cache.left_padding = lengths, left_padding
        self._released.add(index)

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
        # Rewound rows no longer hold what the views split into them, and a
        # failed forward may have left a view mid-step: rebuild next round.
        self.owner._views = None
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


class DeviceCommitTransaction:
    """The device-commit mode of a hybrid verify transaction (spec-round DC-4).

    One B1 lane whose global layers are device-mode ``DeviceOffsetKVCache``
    and whose recurrent layers are ``ArraysCache`` rows with a staged device
    rollback (Kimi/Raon RDA), opened by ``speculative_device_round`` under
    ``device_round.device_commit`` (default off). There are no ``Integral``
    offset reads beyond each KV cache's exact host bound at ``begin`` and no
    trims:

    * ``commit`` keeps the first ``count`` verify rows of every layer on the
      device: KV ``device_commit(count)``, RDA ``device_rollback(count)``
      (which also closes that row's speculation window, so it is not trimmed
      again). ``count`` is the device int32 the round staged
      (``stage_counts``), else the host length; then, because this slice
      still reads every round's packet, each KV cache is ``observe``d at its
      exact length, leaving no slack.
    * ``abort`` returns every row to the pre-round boundary: KV caches are
      observed at their base (rows past it are stale slack, as after a host
      trim), and recurrent rows take back their pre-round state references.

    ``lagged=True`` is the pipelined form (``device_round.pipeline``, DC-5):
    the KV bound need not be exact at ``begin`` (a booked round may still
    have slack); ``commit_on_device(count)`` commits at dispatch with a device
    count and no host value; ``commit(accepted)``, called when the round is
    booked after its packet read, checks the host count and ``settle``s each
    KV cache at the lane's exact length plus the rows of a following round
    still in flight (``settle_length``/``settle_pending``, set by the booker).
    ``abort`` before the device commit puts back each KV cache's device
    ``pos`` and host bound as they were at ``begin``.
    """

    device_commit = True
    batched = False
    plain = False

    def __init__(self, caches, lengths, *, lagged=False):
        from .models.cache import ArraysCache
        from .models.device_offset_kv import DeviceOffsetKVCache

        caches = list(caches)
        if len(lengths) != 1 or not isinstance(lengths[0], Integral) or lengths[0] < 1:
            raise ValueError("a device-commit transaction verifies one lane of at least one row")
        self.lengths = [int(lengths[0])]
        self.width = self.lengths[0]
        self.caches = caches
        self.closed = False
        self._kv = []
        self._kv_saved = []
        self._recurrent = []
        self._counts = None
        self.pipelined = bool(lagged)
        self._device_committed = False
        self.expected_used = None
        self.settle_length = None
        self.settle_pending = 0
        for cache in caches:
            if type(cache) is DeviceOffsetKVCache:
                if not cache.device_mode or cache._last_write is not None or (
                        not lagged and not cache._exact):
                    raise HybridVerifyUnsupported(
                        "a device-commit lane needs every KV cache in device mode at an exact length"
                        if not lagged else
                        "a pipelined device-commit lane needs every KV cache in device mode, committed"
                    )
                self._kv.append((cache, int(cache.bound)))
                self._kv_saved.append(
                    (cache, cache.pos, int(cache.bound), int(cache.offset), cache._exact)
                )
            elif type(cache) is ArraysCache and _recurrent(cache):
                if cache.speculating:
                    raise ValueError("recurrent row already has a speculation owner")
                if cache.batch_size != 1 and not cache.empty():
                    raise ValueError("each authoritative recurrent cache must be a B1 row")
                self._recurrent.append(
                    (cache, list(cache.cache), cache.lengths, cache.left_padding)
                )
            else:
                raise HybridVerifyUnsupported(
                    f"a device-commit lane has no device form for {type(cache).__name__}"
                )
        started = []
        try:
            for cache, *_state in self._recurrent:
                cache.start_speculation()
                started.append(cache)
        except BaseException:
            for cache in started:
                cache.stop_speculation()
            self.closed = True
            raise

    def stage_counts(self, counts, expected):
        """The device int32 ``[1]`` count per lane and the host value it holds."""
        if len(counts) != 1 or len(expected) != 1:
            raise ValueError("a device-commit transaction stages one lane count")
        self._counts = (counts[0], int(expected[0]))

    def count_for(self, accepted):
        """The device count for ``accepted`` committed rows (``_commit`` stages
        the drafter rings with it)."""
        return self._count(int(accepted))

    def _count(self, accepted):
        import mlx.core as mx

        if self._counts is not None and self._counts[1] == accepted:
            return self._counts[0].reshape(1).astype(mx.int32)
        # A stop cut the round's delivery short of the device count.
        return mx.array([accepted], dtype=mx.int32)

    def _check_advanced(self):
        for cache, base in self._kv:
            if int(cache.bound) != base + self.lengths[0] or cache._last_write is None:
                raise RuntimeError("every target layer must finish verification before commit")

    def commit_on_device(self, count):
        """Pipelined form: keep ``count`` (device int32 [1]) rows at dispatch."""
        if not self.pipelined or self.closed or self._device_committed:
            raise RuntimeError("commit_on_device needs an open pipelined device-commit transaction")
        try:
            self._check_advanced()
            for cache, _base in self._kv:
                cache.device_commit(count)
            for cache, *_state in self._recurrent:
                cache.device_rollback(count)
        except BaseException:
            self.abort()
            raise
        self._device_committed = True

    def device_state(self):
        """The lazy arrays a device commit leaves: KV fills and rolled-back recurrent entries.

        A ``fixed_shape`` pipelined round schedules them with its own packet
        (slice 1a-5), so they are encoded in this round whether or not a
        following round is dispatched before it is read."""
        state = [cache.pos for cache, _base in self._kv]
        for cache, *_rest in self._recurrent:
            state.extend(entry for entry in cache.cache if entry is not None)
        return state

    def commit(self, accepted_lengths):
        if self.closed:
            raise RuntimeError("stale or closed device-commit transaction")
        accepted = [int(n) for n in accepted_lengths]
        if len(accepted) != 1 or not 1 <= accepted[0] <= self.lengths[0]:
            raise ValueError("accepted prefix exceeds verified input length")
        (used,) = accepted
        if self.pipelined:
            # Booking a round committed at dispatch: the host count must be the
            # device count, then the bound tightens (``settle``).
            if not self._device_committed or self.settle_length is None:
                raise RuntimeError("a pipelined device-commit round books after its device commit")
            if used != self.expected_used:
                raise RuntimeError(
                    f"pipelined device commit kept {self.expected_used} rows, the host books {used}"
                )
            for cache, _base in self._kv:
                cache.settle(self.settle_length, self.settle_pending)
            self.closed = True
            rows = [list(self.caches)]
            self._close()
            return rows
        try:
            self._check_advanced()
            count = self._count(used)
            for cache, base in self._kv:
                cache.device_commit(count)
                cache.observe(base + used)
            for cache, *_state in self._recurrent:
                cache.device_rollback(count)
        except BaseException:
            self.abort()
            raise
        self.closed = True
        rows = [list(self.caches)]
        self._close()
        return rows

    def commit_paths(self, paths):
        raise HybridVerifyUnsupported("a device-commit transaction commits chain prefixes only")

    def abort(self):
        if self.closed:
            return
        self.closed = True
        first = None
        if self._device_committed:
            # Past its device commit a pipelined round is undone by the
            # dispatcher's record (``speculative_device_pipeline``), not here.
            self._close()
            raise RuntimeError("a device-committed pipelined round cannot abort its transaction")
        if self.pipelined:
            for cache, pos, bound, offset, exact in self._kv_saved:
                cache.pos, cache.bound, cache.offset, cache._exact = pos, bound, offset, exact
                cache._last_write = None
                cache.read_base = None
        for cache, base in ([] if self.pipelined else self._kv):
            try:
                cache.observe(base)
            except BaseException as error:  # noqa: BLE001 - every row gets its turn
                first = first or error
        for cache, state, lengths, left_padding in self._recurrent:
            cache.cache = list(state)
            cache.lengths, cache.left_padding = lengths, left_padding
            try:
                if cache.speculating:
                    cache.stop_speculation()
            except BaseException as error:  # noqa: BLE001
                first = first or error
        self._close()
        if first is not None:
            raise first

    def _close(self):
        # The staged count stays: ``_commit`` stages the drafter rings with it
        # after this commit (``count_for``); it is one [1] int32.
        self._kv = []
        self._kv_saved = []
        self._recurrent = []


__all__ = [
    "DeviceCommitTransaction",
    "HybridVerifyRows",
    "HybridVerifyTransaction",
    "HybridVerifyUnsupported",
    "is_hybrid_rows",
]
