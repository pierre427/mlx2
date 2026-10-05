"""Default-off native request branch with one public state-pointer boundary.

The caller serializes all writer/pool operations and supplies genuine terminal
read events. This module never submits GPU work or attaches to serving.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Self

from .paged_attention_native import complete_packed_read_after_event
from .paged_attention_pack import PackedTokenRead
from .paged_attention_plan import PAGE_SIZE
from .paged_kv_token import PagedKVTokenOwner
from .paged_gdn_checkpoint import GDNBoundaryCheckpoint
from .paged_request_transaction import STATE_PLANES, CandidateRequest


class NativeAtomicError(RuntimeError):
    """A native request state boundary cannot be safely published."""


@dataclass(frozen=True)
class StagedReadTerminalProof:
    """One matched terminal read, reusable by lanes in the same packed use."""

    use: PackedTokenRead
    event: tuple[int, bool]


def complete_staged_read_after_event(use: PackedTokenRead,
                                     event: tuple[int, bool]) -> StagedReadTerminalProof:
    if not complete_packed_read_after_event(use, event):
        raise NativeAtomicError("staged native read terminal event failed")
    return StagedReadTerminalProof(use, event)


@dataclass(frozen=True)
class NativePublishedView:
    revision: str
    generation: int
    _layer_tables: tuple[tuple[object, ...], ...]
    offset: int
    _companions: tuple[tuple[str, tuple[Any, ...]], ...]
    _lease: _NativeReaderLease = field(repr=False, compare=False)

    @property
    def layer_tables(self) -> tuple[tuple[object, ...], ...]:
        self._lease.require_open()
        return self._layer_tables

    @property
    def companions(self) -> tuple[tuple[str, tuple[Any, ...]], ...]:
        self._lease.require_open()
        return self._companions

    @property
    def layer_owners(self) -> tuple[PagedKVTokenOwner, ...]:
        """Public owners for a read; the caller must hold this lease until terminal proof."""
        self._lease.require_open()
        return self._lease.state.layers

    def close(self) -> None:
        self._lease.close()

    def __enter__(self) -> Self:
        self._lease.require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class _NativeReaderLease:
    def __init__(self, owner: NativeAtomicRequestOwner, state: _State) -> None:
        self.owner, self.state = owner, state
        self.closed = False

    def require_open(self) -> None:
        if self.closed:
            raise NativeAtomicError("native reader lease is closed")
        if any(layer.writer.poisoned for layer in self.state.layers):
            raise NativeAtomicError("native reader arena is poisoned")

    def close(self) -> None:
        with self.owner._lock:
            if self.closed:
                return
            self.closed = True
            key = id(self.state)
            count = self.owner._readers[key]
            if count == 1:
                del self.owner._readers[key]
            else:
                self.owner._readers[key] = count - 1

    def __del__(self) -> None:
        # Legacy snapshot callers may omit close. Retain safety until the view
        # is no longer reachable; explicit close is required for prompt reap.
        if not getattr(self, "closed", True):
            self.close()


@dataclass(frozen=True)
class _State:
    revision: str
    generation: int
    layers: tuple[PagedKVTokenOwner, ...]
    companions: tuple[tuple[str, tuple[Any, ...]], ...]


class NativeAtomicRequestOwner:
    """Own a private per-layer fork and publish one immutable state pointer.

    Old public layers are retained after a swap. They may be reclaimed only
    after the serving reader lifecycle proves all old snapshots unused.
    """

    def __init__(self, revision: str, layers: tuple[PagedKVTokenOwner, ...],
                 companions: dict[str, Iterable[Any]], *,
                 supported_planes: tuple[str, ...] = STATE_PLANES,
                 enabled: bool = False,
                 reuse_private_tail: bool = False,
                 checkpoint_planes: tuple[str, ...] = (),
                 accepted_prefix_checkpoints: bool = False,
                 recurrent_clone: Callable[[tuple[Any, ...]], tuple[Any, ...]] | None = None,
                 lane_id: int | None = None) -> None:
        if not isinstance(revision, str) or not revision:
            raise ValueError("revision is required")
        if (type(layers) is not tuple or not layers or
                any(type(layer) is not PagedKVTokenOwner for layer in layers) or
                len({id(layer) for layer in layers}) != len(layers)):
            raise ValueError("distinct token owners are required for all layers")
        writer, profile, offset = layers[0].writer, layers[0].profile, layers[0].offset
        if any(layer.writer is not writer or layer.profile != profile or
               layer.offset != offset for layer in layers):
            raise ValueError("layer writer, geometry, and offset must agree")
        if (type(supported_planes) is not tuple or
                tuple(p for p in STATE_PLANES if p in supported_planes) != supported_planes or
                "kv" not in supported_planes or set(companions) != set(supported_planes) - {"kv"}):
            raise ValueError("complete ordered state-plane coverage is required")
        for layer in layers:
            layer.accepted_handles()
            if layer.sequence.kv_end != offset:
                raise ValueError("layer sequence is not accepted")
        if checkpoint_planes:
            if (checkpoint_planes != ("gdn",) or supported_planes != ("kv", "gdn") or
                    type(lane_id) is not int or lane_id < 0 or
                    not callable(recurrent_clone)):
                raise ValueError("GDN checkpoint requires an exact KV/GDN lane and clone")
            rows = tuple(companions["gdn"])
            if (len(rows) != 1 or type(rows[0]) is not GDNBoundaryCheckpoint or
                    rows[0].revision != revision or rows[0].lane_id != lane_id or
                    rows[0].offset != offset or rows[0].generation != 0):
                raise ValueError("initial GDN checkpoint disagrees with public KV")
            # Bootstrap still owns its ordinary cache containers. Keep the
            # same materialized tensor leaves but take distinct containers so
            # a caller retaining HybridBootstrap cannot mutate public state.
            private_initial = rows[0].private_successors(recurrent_clone)
            companions = {"gdn": (GDNBoundaryCheckpoint(
                revision, lane_id, offset, 0, private_initial),)}
        elif recurrent_clone is not None or lane_id is not None:
            raise ValueError("recurrent clone and lane require a GDN checkpoint")
        if type(accepted_prefix_checkpoints) is not bool or (
                accepted_prefix_checkpoints and not checkpoint_planes):
            raise ValueError("accepted-prefix checkpoints require a checkpoint plane")
        self._lock = threading.RLock()
        self.supported_planes = supported_planes
        self._checkpoint_planes = checkpoint_planes
        self._accepted_prefix_checkpoints = accepted_prefix_checkpoints
        self._recurrent_clone = recurrent_clone
        self._lane_id = lane_id
        self._enabled = enabled is True
        self._reuse_private_tail = reuse_private_tail is True
        self._tail_guard: NativeBranch | None = None
        self._public = _State(revision, 0, layers, self._copy_companions(companions))
        self._writers = frozenset(layer.writer for layer in layers)
        self._retired: list[_State] = []
        self._quarantine: list[NativeBranch] = []
        self._open_branches: set[NativeBranch] = set()
        self._readers: dict[int, int] = {}
        self._closed = False

    def _copy_companions(self, values: dict[str, Iterable[Any]]) -> tuple[tuple[str, tuple[Any, ...]], ...]:
        return tuple((p, tuple(rows) if p in self._checkpoint_planes else
                      tuple(copy.deepcopy(tuple(rows)))) for p, rows in values.items())

    @property
    def atomic_publish(self) -> bool:
        return self._enabled

    @property
    def fully_retired(self) -> bool:
        """Whether a closed continuation can release its final owner reference.

        A poisoned writer needs explicit one-way arena teardown even after
        every terminal callback has drained. This property never reaps state
        or advances a native completion watermark by itself.
        """
        with self._lock:
            return bool(
                self._closed
                and not self._retired
                and not self._quarantine
                and not self._open_branches
                and not self._readers
                and self._tail_guard is None
                and all(not writer.pending_epochs and not writer.ledger.pending_count
                        and (not writer.poisoned or writer.failed_arena_torn_down)
                        for writer in self._writers)
            )

    def snapshot(self) -> NativePublishedView:
        """Acquire a public reader lease; close it after the native read is terminal."""
        with self._lock:
            if self._closed:
                raise NativeAtomicError("native request owner is closed")
            if self._tail_guard is not None:
                raise NativeAtomicError("private tail transaction excludes public readers")
            state = self._public
            if any(layer.writer.poisoned for layer in state.layers):
                raise NativeAtomicError("native public arena is poisoned")
            tables = tuple(layer.accepted_handles() for layer in state.layers)
            companions = tuple((plane, rows if plane in self._checkpoint_planes else
                                copy.deepcopy(rows)) for plane, rows in state.companions)
            key = id(state)
            self._readers[key] = self._readers.get(key, 0) + 1
            return NativePublishedView(state.revision, state.generation, tables,
                                       state.layers[0].offset, companions,
                                       _NativeReaderLease(self, state))

    def reap_retired(self) -> int:
        """Close old public layers only after readers and all native epochs drain.

        The serving caller must serialize this with writer/pool operations. A
        poisoned writer has no trustworthy terminal proof, so it cannot reap.
        """
        with self._lock:
            remaining = []
            closed = 0
            for index, state in enumerate(self._retired):
                writer = state.layers[0].writer
                if (self._readers.get(id(state), 0) or writer.poisoned or
                        writer.pending_epochs or writer.ledger.pending_count):
                    remaining.append(state)
                    continue
                try:
                    for layer in state.layers:
                        layer.close()
                except BaseException:
                    self._retired = remaining + self._retired[index:]
                    raise
                closed += 1
            self._retired = remaining
            return closed

    def close(self) -> None:
        """Stop new readers/branches and retire the final public generation.

        Existing readers and native epochs keep its pages until reap_retired.
        An outstanding private branch must still be rolled back separately.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._retired.append(self._public)

    def begin(self, request: CandidateRequest) -> NativeBranch:
        if type(request) is not CandidateRequest:
            raise TypeError("request must be CandidateRequest")
        with self._lock:
            if self._closed:
                raise NativeAtomicError("native request owner is closed")
            if self._tail_guard is not None:
                raise NativeAtomicError("private tail transaction is active")
            if not self._enabled:
                raise NativeAtomicError("native atomic route is disabled")
            if request.planes != self.supported_planes:
                raise NativeAtomicError("request lacks complete state-plane coverage")
            if self._checkpoint_planes and (
                    request.lane_id != self._lane_id or
                    (request.proposed_rows != 1 and
                     not self._accepted_prefix_checkpoints)):
                raise NativeAtomicError(
                    "GDN checkpoint requires its bound lane and supported proposal width")
            origin = self._public
            if request.revision != origin.revision:
                raise NativeAtomicError("request revision drifted")
            if origin.layers[0].writer.poisoned or origin.layers[0].writer.pending_epochs:
                raise NativeAtomicError("native writer is pending or poisoned")
            recurrent_caches: tuple[Any, ...] = ()
            if self._checkpoint_planes:
                checkpoint = dict(origin.companions)["gdn"][0]
                if (type(checkpoint) is not GDNBoundaryCheckpoint or
                        checkpoint.revision != request.revision or
                        checkpoint.lane_id != request.lane_id or
                        checkpoint.offset != origin.layers[0].offset or
                        checkpoint.generation != origin.generation):
                    raise NativeAtomicError("public GDN and KV checkpoint drifted")
                recurrent_caches = checkpoint.private_successors(self._recurrent_clone)
            permit_tail = bool(self._reuse_private_tail and not self._readers and
                               not self._open_branches and not self._quarantine and
                               not origin.layers[0].writer.ledger.pending_count)
            children: list[PagedKVTokenOwner] = []
            try:
                for parent in origin.layers:
                    parent.accepted_handles()
                    child = PagedKVTokenOwner(parent.writer, parent.profile,
                                              permit_candidate=True)
                    child.sequence = parent.sequence.fork()
                    child._accepted_tokens = parent.offset
                    if (permit_tail and parent.offset % PAGE_SIZE and child.sequence.handles and
                            parent.writer.pool.references(child.sequence.handles[-1]) == 2):
                        child._reuse_shared_tail_refs = 2
                    children.append(child)
            except BaseException:
                for child in children:
                    child.close()
                raise
            branch = NativeBranch(self, request, origin, tuple(children),
                                  recurrent_caches=recurrent_caches)
            self._open_branches.add(branch)
            if permit_tail:
                self._tail_guard = branch
            return branch

    def reap_quarantine(self) -> int:
        """Release cancelled private forks only after all writer epochs drain."""
        with self._lock:
            remaining = []
            closed = 0
            for branch in self._quarantine:
                writer = branch.layers[0].writer
                if (writer.poisoned or writer.pending_epochs or writer.ledger.pending_count or
                        (self._tail_guard is branch and
                         any(layer._failed for layer in branch.layers))):
                    remaining.append(branch)
                else:
                    branch._close_layers()
                    if self._tail_guard is branch:
                        self._tail_guard = None
                    closed += 1
            self._quarantine = remaining
            return closed

    def reap_failed_after_teardown(self) -> int:
        """Drop failed host generations after one-way native arena teardown.

        This never reopens quarantined physical slots. It only balances host
        sequence references when the backend has synchronized and released its
        arena handle and no reader, branch, or native epoch can remain.
        """
        with self._lock:
            if not self._closed:
                raise NativeAtomicError("close the failed request owner first")
            if self._open_branches or self._readers:
                raise NativeAtomicError("failed arena still has reader or branch leases")
            states = self._retired
            branches = self._quarantine
            writers = {layer.writer for state in states for layer in state.layers}
            writers.update(layer.writer for branch in branches for layer in branch.layers)
            if any(not writer.failed_arena_torn_down for writer in writers):
                raise NativeAtomicError("native arena teardown is not complete")
            count = 0
            for state in states:
                for layer in state.layers:
                    layer.close()
                count += 1
            for branch in branches:
                branch._close_layers()
                count += 1
            self._retired = []
            self._quarantine = []
            self._tail_guard = None
            return count


class NativeBranch:
    def __init__(self, owner: NativeAtomicRequestOwner, request: CandidateRequest,
                 origin: _State, layers: tuple[PagedKVTokenOwner, ...], *,
                 recurrent_caches: tuple[Any, ...] = ()) -> None:
        self._owner, self._request, self._origin = owner, request, origin
        self.layers = layers
        self.recurrent_caches = recurrent_caches
        self._staged_recurrent: dict[int, GDNBoundaryCheckpoint] = {}
        self._companions: dict[str, tuple[Any, ...]] = {}
        self._proved: set[int] = set()
        self._proved_rows: dict[int, int] = {}
        self._executed_rows: int | None = None
        self._closed = False

    def stage(self, plane: str, rows: Iterable[Any]) -> None:
        if self._closed:
            raise NativeAtomicError("branch is closed")
        if plane in self._owner._checkpoint_planes:
            raise NativeAtomicError("recurrent checkpoint needs a boundary replacement")
        if plane == "kv" or plane not in self._request.planes or plane in self._companions:
            raise NativeAtomicError("companion plane is unrequested or already staged")
        copied = tuple(copy.deepcopy(tuple(rows)))
        if len(copied) != self._request.proposed_rows:
            raise NativeAtomicError("companion rows must cover the proposal")
        self._companions[plane] = copied

    def stage_recurrent_boundary(self, caches: tuple[Any, ...], *, offset: int) -> None:
        """Stage the complete proposal boundary (the ordinary Q1 API)."""
        self.stage_recurrent_prefix(
            caches, accepted_rows=self._request.proposed_rows, offset=offset)

    def stage_recurrent_prefix(self, caches: tuple[Any, ...], *,
                               accepted_rows: int, offset: int) -> None:
        """Snapshot one exact recurrent boundary inside the private proposal."""
        if (self._closed or self._owner._checkpoint_planes != ("gdn",) or
                type(accepted_rows) is not int or
                not 1 <= accepted_rows <= self._request.proposed_rows or
                self._staged_recurrent):
            raise NativeAtomicError(
                "GDN prefix is closed, unsupported, invalid or already staged")
        if (type(caches) is not tuple or len(caches) != len(self.recurrent_caches) or
                any(cache is not private for cache, private in
                    zip(caches, self.recurrent_caches))):
            raise NativeAtomicError("GDN boundary must use this branch's private caches")
        checkpoint = dict(self._origin.companions)["gdn"][0]
        if offset != self._origin.layers[0].offset + accepted_rows:
            raise NativeAtomicError("GDN boundary offset disagrees with accepted prefix")
        snapshot = self._owner._recurrent_clone(caches)
        self._staged_recurrent[accepted_rows] = checkpoint.successor(
            snapshot, offset=offset)

    def seal_executed_rows(self, rows: int) -> None:
        """Bind a short-circuited ragged proposal to its physical prefix.

        Online verification may stop a lane before its declared maximum depth.
        Sealing is allowed only after every layer has terminal proof for exactly
        that prefix and before publication. No later row can be submitted.
        """
        if (self._closed or type(rows) is not int or
                not 1 <= rows <= self._request.proposed_rows or
                self._executed_rows is not None or
                any(self._proved_rows.get(index, 0) != rows
                    for index in range(len(self.layers))) or
                any(layer.offset != self._origin.layers[0].offset + rows or
                    layer.sequence.kv_end != self._origin.layers[0].offset + rows
                    for layer in self.layers)):
            raise NativeAtomicError("executed proposal prefix lacks exact terminal proof")
        if (self._owner._checkpoint_planes and
                rows not in self._staged_recurrent):
            raise NativeAtomicError("executed proposal prefix lacks its GDN checkpoint")
        self._executed_rows = rows

    def prove_layer_read(self, layer_index: int, use: PackedTokenRead,
                         event: tuple[int, bool]) -> None:
        """Consume one matched, successful terminal callback for this fork."""
        if self._closed or type(layer_index) is not int or not 0 <= layer_index < len(self.layers):
            raise NativeAtomicError("unknown or closed layer")
        if layer_index in self._proved or type(use) is not PackedTokenRead:
            raise NativeAtomicError("duplicate or invalid layer proof")
        layer = self.layers[layer_index]
        if (use.writer is not layer.writer or use.state != "submitted" or
                tuple(use.plan.page_table) != layer.accepted_handles()):
            raise NativeAtomicError("read proof does not cover this private layer")
        if not complete_packed_read_after_event(use, event):
            raise NativeAtomicError("native read terminal event failed")
        self._proved.add(layer_index)
        # This API proves one complete request read.  Keep the row ledger in
        # sync with the staged-N API so checkpoint owners use one publication
        # rule for ordinary Q1 and online ragged execution.
        self._proved_rows[layer_index] = self._request.proposed_rows

    def prove_staged_layer_read(self, layer_index: int, lane_index: int,
                                proof: StagedReadTerminalProof) -> None:
        """Bind one completed packed read span to this private request fork."""
        if (self._closed or type(layer_index) is not int or
                not 0 <= layer_index < len(self.layers) or
                type(proof) is not StagedReadTerminalProof or
                type(lane_index) is not int or not 0 <= lane_index < len(proof.use.plan.spans)):
            raise NativeAtomicError("duplicate or invalid staged layer proof")
        use = proof.use
        if (use.state != "closed" or use.terminal_succeeded is not True or
                proof.event != (use.lease.epoch, True) or
                use.writer is not self.layers[layer_index].writer):
            raise NativeAtomicError("staged read has no matching successful terminal")
        span = use.plan.spans[lane_index]
        layer = self.layers[layer_index]
        proved = self._proved_rows.get(layer_index, 0)
        if (span.query_start != self._origin.layers[0].offset + proved or
                span.row_count < 1 or proved + span.row_count > self._request.proposed_rows or
                span.kv_end != span.query_start + span.row_count):
            raise NativeAtomicError("staged read rows do not match this request")
        table = use.plan.page_table[span.table_begin:span.table_begin + span.table_count]
        if (layer.offset != span.kv_end or layer._pending or
                tuple(table) != layer.accepted_handles()):
            raise NativeAtomicError("staged read table is not the completed private layer")
        proved += span.row_count
        self._proved_rows[layer_index] = proved
        if proved == self._request.proposed_rows:
            self._proved.add(layer_index)

    def prepare(self, accepted_rows: int) -> NativePrepared:
        if self._closed:
            raise NativeAtomicError("branch is closed")
        executed_rows = (self._request.proposed_rows if self._executed_rows is None
                         else self._executed_rows)
        if type(accepted_rows) is not int or not 0 <= accepted_rows <= executed_rows:
            raise ValueError("accepted rows must be an integer prefix")
        if self._owner._checkpoint_planes and accepted_rows == 0:
            raise NativeAtomicError("GDN boundary requires a positive accepted prefix or rollback")
        if set(self._companions) != set(self._request.planes) - {"kv"} - set(
                self._owner._checkpoint_planes):
            raise NativeAtomicError("all companion state planes must be staged")
        if (self._owner._checkpoint_planes and
                accepted_rows not in self._staged_recurrent):
            raise NativeAtomicError("accepted GDN successor checkpoint is not staged")
        if self._owner._checkpoint_planes:
            if any(self._proved_rows.get(index, 0) != executed_rows
                   for index in range(len(self.layers))):
                raise NativeAtomicError("every layer needs exact prefix terminal read proof")
        elif self._proved != set(range(len(self.layers))):
            raise NativeAtomicError("every layer needs successful terminal read proof")
        writer = self.layers[0].writer
        if writer.poisoned or writer.pending_epochs or writer.ledger.pending_count:
            raise NativeAtomicError("native work is pending or failed")
        expected = self._origin.layers[0].offset + executed_rows
        if any(layer.offset != expected or layer.sequence.kv_end != expected
               for layer in self.layers):
            raise NativeAtomicError("every layer needs a completed proposed suffix")
        accepted_end = self._origin.layers[0].offset + accepted_rows
        if any(layer.sequence.retained_start > accepted_end for layer in self.layers):
            raise NativeAtomicError("accepted prefix is outside retained KV")
        if accepted_rows != executed_rows:
            # All proposed writes and verification reads are terminal here.
            # The private layer tables can now release complete suffix pages.
            for layer in self.layers:
                layer.truncate_accepted(accepted_end)
        next_companions = dict(self._origin.companions)
        for plane, rows in self._companions.items():
            next_companions[plane] += rows[:accepted_rows]
        if self._owner._checkpoint_planes:
            # One token replaces exactly one recurrent boundary. A rejected
            # token must roll back instead of publishing a no-op generation.
            next_companions["gdn"] = (self._staged_recurrent[accepted_rows],)
        successor = _State(self._request.revision, self._origin.generation + 1,
                           self.layers, tuple((p, next_companions[p]) for p in
                                              self._owner.supported_planes if p != "kv"))
        self._closed = True
        self._companions.clear()
        self._staged_recurrent = {}
        self._executed_rows = None
        return NativePrepared(self._owner, self._request, self._origin, successor, self)

    def _close_layers(self) -> None:
        for layer in self.layers:
            layer.close()
        self.recurrent_caches = ()
        self._staged_recurrent = {}
        self._executed_rows = None
        self._origin = None

    def rollback(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._companions.clear()
        self._staged_recurrent = {}
        self._executed_rows = None
        with self._owner._lock:
            self._owner._open_branches.discard(self)
            writer = self.layers[0].writer
            if (writer.poisoned or writer.pending_epochs or writer.ledger.pending_count or
                    (self._owner._tail_guard is self and
                     any(layer._failed for layer in self.layers))):
                self._owner._quarantine.append(self)
            else:
                self._close_layers()
                if self._owner._tail_guard is self:
                    self._owner._tail_guard = None


class NativePrepared:
    def __init__(self, owner: NativeAtomicRequestOwner, request: CandidateRequest,
                 origin: _State, successor: _State, branch: NativeBranch) -> None:
        self._owner, self._origin, self._successor, self._branch = owner, origin, successor, branch
        self.planes, self.revision = request.planes, request.revision

    def publish(self) -> None:
        with self._owner._lock:
            self._validate_locked()
            self._publish_locked()

    def _validate_locked(self) -> None:
        if self._successor is None:
            raise NativeAtomicError("prepared native state is closed")
        writer = self._successor.layers[0].writer
        if (self._owner._closed or self._owner._public is not self._origin or
                writer.poisoned or writer.pending_epochs or
                writer.ledger.pending_count):
            raise NativeAtomicError("native state or public revision drifted")

    def _publish_locked(self) -> None:
        successor = self._successor
        self._owner._retired.append(self._origin)
        self._owner._public = successor  # Sole public pointer assignment.
        self._owner._open_branches.discard(self._branch)
        if self._owner._tail_guard is self._branch:
            self._owner._tail_guard = None
        self._successor = None
        if self._owner._checkpoint_planes:
            # The public pointer and reader-retained generations now own the
            # only recurrent roots; retained prepared/branch objects cannot
            # grow checkpoint history.
            self._branch.recurrent_caches = ()
            self._branch._staged_recurrent = {}
            self._branch._origin = None
            self._origin = None

    def rollback(self) -> None:
        if self._successor is not None:
            self._successor = None
            self._branch._closed = False
            self._branch.rollback()
            if self._owner._checkpoint_planes:
                self._origin = None


def publish_native_cohort(states: tuple[NativePrepared, ...]) -> None:
    """Preflight and publish distinct request owners as one host transaction.

    All owner locks are acquired in stable identity order. Every prepared
    successor is validated before the first public pointer changes, so a
    revision/terminal failure leaves the complete cohort unpublished.
    """
    if (type(states) is not tuple or not states or
            any(type(state) is not NativePrepared for state in states) or
            len({id(state._owner) for state in states}) != len(states)):
        raise NativeAtomicError("cohort publication requires distinct prepared owners")
    ordered = sorted(states, key=lambda state: id(state._owner))
    with ExitStack() as locks:
        for state in ordered:
            locks.enter_context(state._owner._lock)
        for state in states:
            state._validate_locked()
        for state in states:
            state._publish_locked()


__all__ = ["NativeAtomicError", "NativeAtomicRequestOwner", "NativePublishedView",
           "publish_native_cohort"]
