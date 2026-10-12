# SPDX-License-Identifier: Apache-2.0
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations

import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx

from .hybrid_speculative import (
    BatchedSelfMTPState,
    DetachedSelfMTPLane,
    SegmentedSelfMTPState,
    SelfMTPCachePair,
)
from .state_plane_operations import (
    SegmentedPromotionOperations,
    StatePlaneContractError,
    StatePlaneOperationRefused,
    StatePlaneOperations,
    require_state_plane_contracts,
)

_REQUIRED_PROMOTION_OPERATIONS = (
    "checkpoint",
    "private_successor",
    "rollback",
    "promotion",
)


def _require_promotion_operations(operations: Any) -> StatePlaneOperations:
    if not isinstance(operations, SegmentedPromotionOperations):
        raise SegmentedPhysicalPromotionDeclined(
            "segmented state has no promotion operations provider"
        )
    return operations


def _provider_call(callable_, *args, **kwargs):
    try:
        return callable_(*args, **kwargs)
    except StatePlaneOperationRefused as error:
        raise SegmentedPhysicalPromotionDeclined(str(error)) from error


def _promotion_contracts(provider, caches):
    try:
        return require_state_plane_contracts(
            provider,
            caches,
            operations=_REQUIRED_PROMOTION_OPERATIONS,
            fidelity="exact",
        )
    except StatePlaneContractError as error:
        raise SegmentedPhysicalPromotionDeclined(str(error)) from error


class SegmentedPhysicalPromotionDeclined(RuntimeError):
    """The state cannot be promoted by the suffix-only exact contract."""


@dataclass(frozen=True)
class SegmentedPhysicalPromotionReceipt:
    queued_ns: int
    finish_ns: int
    stream_wait_ns: int
    reserved_bytes: int
    patched_bytes: int
    recurrent_arrays_reused: int
    cleanup_error_count: int
    rows: int
    layers: int
    advance: int
    state_provider_revision: str = ""
    state_layouts: tuple[str, ...] = ()
    state_fidelity: str = ""


def _state_signature(state: SegmentedSelfMTPState):
    return (
        state.membership_epoch,
        tuple((lane.uid for lane in state.lanes)),
        tuple((id(pair) for pair in state.row_caches)),
        tuple((id(transaction) for transaction in state.transactions)),
        tuple(
            (
                (transaction.lineage.generation, transaction.position)
                for transaction in state.transactions
            )
        ),
    )


def _validate_segmented_view(state, view, operations):
    if not isinstance(state, SegmentedSelfMTPState):
        raise TypeError("physical promotion requires SegmentedSelfMTPState")
    if state.poisoned or state.proposal_open or (not state.lanes):
        raise SegmentedPhysicalPromotionDeclined(
            "promotion begins only on a healthy, proposal-closed cohort"
        )
    if not len(state.lanes) == len(state.row_caches) == len(state.transactions):
        raise SegmentedPhysicalPromotionDeclined(
            "segmented lane/cache/transaction ownership is misaligned"
        )
    if view is not state._segmented_caches:
        raise SegmentedPhysicalPromotionDeclined("segmented compute view was replaced")
    _provider_call(operations.validate_segmented_view, view, state.row_caches)


def ensure_segmented_compute_view(
    state: SegmentedSelfMTPState,
    *,
    operations: StatePlaneOperations,
    note: Callable[[str, int], None] | None = None,
) -> SelfMTPCachePair:
    """Return and retain the exact segmented compute view for ``state``."""
    if state._segmented_caches is None:
        if state.proposal_open:
            raise SegmentedPhysicalPromotionDeclined(
                "cannot build a compute view during an open proposal"
            )
        state._segmented_caches = _provider_call(
            operations.build_segmented_view,
            state.row_caches,
            note=note,
            shared_prefix=state.shared_state_identity is not None,
        )
    _validate_segmented_view(state, state._segmented_caches, operations)
    return state._segmented_caches


@dataclass
class SegmentedPhysicalPromotionTicket:
    state: SegmentedSelfMTPState
    view: SelfMTPCachePair
    operations: StatePlaneOperations
    state_provider_revision: str
    contracts: tuple
    stream: Any
    reserve_tail: int
    signature: tuple
    physical: SelfMTPCachePair
    target_bases: tuple[int | None, ...]
    draft_bases: tuple[int | None, ...]
    creator_thread: int
    queued_ns: int
    reserved_bytes: int
    _finished: bool = False
    _allocation_settled: bool = False

    @property
    def pending_bytes(self) -> int:
        # Once synchronized these arrays are included in measured live memory.
        return 0 if self._allocation_settled or self._finished else self.reserved_bytes

    def settle_for_admission(self) -> None:
        """Complete queued allocation without publishing or consuming state.

        Only the pressure path waits here. Ordinary promotion keeps its copy
        overlapped with the first segmented verification cycle.
        """
        if self._finished:
            raise RuntimeError("physical promotion ticket was already finished")
        if threading.get_ident() != self.creator_thread:
            raise SegmentedPhysicalPromotionDeclined(
                "promotion allocation must settle on its creator thread"
            )
        if not self._allocation_settled:
            mx.synchronize(self.stream)
            self._allocation_settled = True

    def cancel_and_drain(self) -> None:
        """Retire an unpublished candidate after all queued work completes."""
        if self._finished:
            return
        if threading.get_ident() != self.creator_thread:
            raise SegmentedPhysicalPromotionDeclined(
                "promotion must drain on the thread that queued its Metal work"
            )
        if self.stream is not None:
            mx.synchronize(self.stream)
        self._finished = True
        self.physical = None
        self.view = None
        self.state = None

    def finish(self) -> tuple[BatchedSelfMTPState, SegmentedPhysicalPromotionReceipt]:
        """Validate the committed state, patch it, and consume the B1 cohort."""
        started = time.perf_counter_ns()
        if self._finished:
            raise RuntimeError("physical promotion ticket was already finished")
        if threading.get_ident() != self.creator_thread:
            raise SegmentedPhysicalPromotionDeclined(
                "promotion must finish on the thread that queued its Metal work"
            )
        state = self.state
        if state.proposal_open or state._open_proposal is not None:
            raise SegmentedPhysicalPromotionDeclined(
                "first segmented proposal is not closed"
            )
        if state._batched_state is not None or state._transaction_branches:
            raise SegmentedPhysicalPromotionDeclined(
                "segmented proposal still owns compute or transaction branches"
            )
        (epoch, uids, pairs, transactions, generations) = self.signature
        if (
            state.membership_epoch != epoch
            or tuple(lane.uid for lane in state.lanes) != uids
        ):
            raise SegmentedPhysicalPromotionDeclined(
                "membership epoch or lane UID order changed"
            )
        if (
            tuple((id(p) for p in state.row_caches)) != pairs
            or tuple((id(t) for t in state.transactions)) != transactions
        ):
            raise SegmentedPhysicalPromotionDeclined("row ownership changed")
        if state._segmented_caches is not self.view:
            raise SegmentedPhysicalPromotionDeclined(
                "retained segmented compute view was replaced"
            )
        current_contracts = _promotion_contracts(
            self.operations, (*self.view.target, *self.view.draft)
        )
        if (
            str(self.operations.provider_revision) != self.state_provider_revision
            or current_contracts != self.contracts
        ):
            raise SegmentedPhysicalPromotionDeclined(
                "state-plane revision, layout, fidelity, or operations drifted"
            )
        advances = []
        for transaction, (generation, position) in zip(state.transactions, generations):
            stats = transaction.lineage.stats()
            advance = transaction.position - position
            if stats["closed"] or stats["generation"] != generation + 1 or advance <= 0:
                raise SegmentedPhysicalPromotionDeclined(
                    "transaction generation did not advance exactly once"
                )
            advances.append(advance)
        if len(set(advances)) != 1 or advances[0] > self.reserve_tail:
            raise SegmentedPhysicalPromotionDeclined(
                "first-commit target advances must be equal and fit the reserved tail"
            )
        advance = advances[0]
        patched_bytes = 0
        recurrent_reused = 0
        target = []
        draft = []
        evaluation_values = []
        context = mx.stream(self.stream) if self.stream is not None else nullcontext()
        with context:
            for source, candidate, base in zip(
                self.view.target, self.physical.target, self.target_bases
            ):
                materialized = _provider_call(
                    self.operations.materialize_promoted_layer,
                    source,
                    candidate,
                    base,
                    advance=advance,
                    target=True,
                )
                patched_bytes += materialized.patched_bytes
                recurrent_reused += materialized.recurrent_arrays_reused
                target.append(materialized.cache)
                evaluation_values.extend(materialized.evaluation_values)
            for source, candidate, base in zip(
                self.view.draft, self.physical.draft, self.draft_bases
            ):
                materialized = _provider_call(
                    self.operations.materialize_promoted_layer,
                    source,
                    candidate,
                    base,
                    advance=0,
                    target=False,
                )
                patched_bytes += materialized.patched_bytes
                recurrent_reused += materialized.recurrent_arrays_reused
                draft.append(materialized.cache)
                evaluation_values.extend(materialized.evaluation_values)
            if evaluation_values:
                mx.async_eval(*evaluation_values)
        wait_started = time.perf_counter_ns()
        if self.stream is not None:
            mx.synchronize(self.stream)
        stream_wait_ns = time.perf_counter_ns() - wait_started
        pair = SelfMTPCachePair(target=target, draft=draft)
        started_destinations = []
        try:
            for cache in pair.target:
                cache.start_speculation()
                started_destinations.append(cache)
        except BaseException:
            for cache in started_destinations:
                try:
                    cache.stop_speculation()
                except BaseException:
                    pass
            raise
        lanes = list(state.lanes)
        result = BatchedSelfMTPState(lanes, pair, state.membership_epoch)
        row_pairs_to_close = list(state.row_caches)
        transactions_to_close = list(state.transactions)
        state.lanes.clear()
        state.row_caches.clear()
        state.transactions.clear()
        state._segmented_caches = None
        state._batched_state = None
        state._transaction_branches.clear()
        state.poisoned = True
        state.poison_reason = "consumed by segmented physical promotion"
        self._finished = True
        cleanup_errors = []
        for row_pair in row_pairs_to_close:
            for cache in row_pair.target:
                try:
                    cache.stop_speculation()
                except BaseException as error:
                    cleanup_errors.append(error)
        for transaction in transactions_to_close:
            try:
                transaction.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            logging.warning(
                "physical promotion published with %d old-row cleanup errors; "
                "the replacement physical state remains authoritative",
                len(cleanup_errors),
            )
        receipt = SegmentedPhysicalPromotionReceipt(
            queued_ns=self.queued_ns,
            finish_ns=time.perf_counter_ns() - started,
            stream_wait_ns=stream_wait_ns,
            reserved_bytes=self.reserved_bytes,
            patched_bytes=patched_bytes,
            recurrent_arrays_reused=recurrent_reused,
            cleanup_error_count=len(cleanup_errors),
            rows=len(lanes),
            layers=len(target) + len(draft),
            advance=advance,
            state_provider_revision=self.state_provider_revision,
            state_layouts=tuple(contract.layout for contract in self.contracts),
            state_fidelity="exact",
        )
        self.physical = None
        self.view = None
        self.state = None
        return (result, receipt)


def begin_segmented_physical_promotion(
    state: SegmentedSelfMTPState,
    *,
    reserve_tail: int,
    stream: Any,
    note: Callable[[str, int], None] | None = None,
    operations: StatePlaneOperations | None = None,
) -> SegmentedPhysicalPromotionTicket:
    """Queue prefix-only physical formation without touching the B1 owners."""
    reserve_tail = int(reserve_tail)
    if reserve_tail <= 0:
        raise ValueError("reserve_tail must be positive")
    operations = _require_promotion_operations(
        operations or getattr(state, "state_plane_operations", None)
    )
    view = ensure_segmented_compute_view(state, operations=operations, note=note)
    contracts = _promotion_contracts(operations, (*view.target, *view.draft))
    signature = _state_signature(state)
    started = time.perf_counter_ns()
    (target, draft) = ([], [])
    (target_bases, draft_bases) = ([], [])
    reserved_bytes = 0
    evaluation_values = []
    with mx.stream(stream):
        for cache in view.target:
            reservation = _provider_call(
                operations.reserve_promotion_layer,
                cache,
                reserve_tail,
                shared_prefix=state.shared_state_identity is not None,
            )
            target.append(reservation.cache)
            target_bases.append(reservation.base)
            reserved_bytes += reservation.reserved_bytes
            evaluation_values.extend(reservation.evaluation_values)
            if (
                state.shared_state_identity is not None
                and reservation.cache is not None
                and note is not None
            ):
                note("async_shared_state_prefix_fused_layers", 1)
        for cache in view.draft:
            reservation = _provider_call(
                operations.reserve_promotion_layer,
                cache,
                0,
                shared_prefix=state.shared_state_identity is not None,
            )
            draft.append(reservation.cache)
            draft_bases.append(reservation.base)
            reserved_bytes += reservation.reserved_bytes
            evaluation_values.extend(reservation.evaluation_values)
            if (
                state.shared_state_identity is not None
                and reservation.cache is not None
                and note is not None
            ):
                note("async_shared_state_prefix_fused_layers", 1)
        if evaluation_values:
            mx.async_eval(*evaluation_values)
    return SegmentedPhysicalPromotionTicket(
        state=state,
        view=view,
        operations=operations,
        state_provider_revision=str(operations.provider_revision),
        contracts=contracts,
        stream=stream,
        reserve_tail=reserve_tail,
        signature=signature,
        physical=SelfMTPCachePair(target, draft),
        target_bases=tuple(target_bases),
        draft_bases=tuple(draft_bases),
        creator_thread=threading.get_ident(),
        queued_ns=time.perf_counter_ns() - started,
        reserved_bytes=reserved_bytes,
    )


@dataclass
class SharedPrefixPhysicalPromotionPrequeue:
    """Unpublished B2 shared-state base staged before a two-row fan-out exists.

    This narrow object is valid only when the eventual segmented cohort carries
    the same non-empty host attestation on every row.  Binding installs the
    ordinary membership/ownership signature; until then cancellation owns no
    live B1 state and can only retire the unpublished destination arrays.
    """

    prefix_id: str
    rows: int
    stream: Any
    reserve_tail: int
    physical: SelfMTPCachePair
    operations: StatePlaneOperations
    state_provider_revision: str
    contracts: tuple
    target_bases: tuple[int | None, ...]
    draft_bases: tuple[int | None, ...]
    creator_thread: int
    queued_ns: int
    reserved_bytes: int
    reserve_total_ns: int = 0
    validation_ns: int = 0
    allocation_graph_ns: int = 0
    copy_graph_ns: int = 0
    metadata_graph_ns: int = 0
    submit_ns: int = 0
    _finished: bool = False

    def cancel_and_drain(self) -> None:
        if self._finished:
            return
        if threading.get_ident() != self.creator_thread:
            raise SegmentedPhysicalPromotionDeclined(
                "promotion prequeue must drain on its creator thread"
            )
        if self.stream is not None:
            mx.synchronize(self.stream)
        self._finished = True
        self.physical = None

    def bind(
        self,
        state: SegmentedSelfMTPState,
        *,
        note: Callable[[str, int], None] | None = None,
    ) -> SegmentedPhysicalPromotionTicket:
        """Bind a staged shared-prefix candidate to the admitted cohort."""

        if self._finished:
            raise RuntimeError("physical promotion prequeue was already consumed")
        if threading.get_ident() != self.creator_thread:
            raise SegmentedPhysicalPromotionDeclined(
                "promotion prequeue must bind on its creator thread"
            )
        if not self.prefix_id or state.shared_state_identity != self.prefix_id:
            raise SegmentedPhysicalPromotionDeclined(
                "admitted cohort lacks the staged shared-prefix attestation"
            )
        if len(state.lanes) != self.rows:
            raise SegmentedPhysicalPromotionDeclined(
                "admitted cohort width differs from the staged candidate"
            )
        view = ensure_segmented_compute_view(
            state, operations=self.operations, note=note
        )
        view_contracts = _promotion_contracts(
            self.operations, (*view.target, *view.draft)
        )
        if (
            str(self.operations.provider_revision) != self.state_provider_revision
            or view_contracts != self.contracts
        ):
            raise SegmentedPhysicalPromotionDeclined(
                "admitted state-plane revision, layout, fidelity, or operations "
                "differ from the staged source"
            )
        if len(view.target) != len(self.physical.target) or len(view.draft) != len(
            self.physical.draft
        ):
            raise SegmentedPhysicalPromotionDeclined(
                "admitted cache layer width differs from the staged candidate"
            )
        for sources, candidates, bases in (
            (view.target, self.physical.target, self.target_bases),
            (view.draft, self.physical.draft, self.draft_bases),
        ):
            for source, candidate, base in zip(sources, candidates, bases):
                _provider_call(
                    self.operations.validate_staged_layer,
                    source,
                    candidate,
                    base,
                )
        ticket = SegmentedPhysicalPromotionTicket(
            state=state,
            view=view,
            operations=self.operations,
            state_provider_revision=self.state_provider_revision,
            contracts=view_contracts,
            stream=self.stream,
            reserve_tail=self.reserve_tail,
            signature=_state_signature(state),
            physical=self.physical,
            target_bases=self.target_bases,
            draft_bases=self.draft_bases,
            creator_thread=self.creator_thread,
            queued_ns=self.queued_ns,
            reserved_bytes=self.reserved_bytes,
        )
        self._finished = True
        self.physical = None
        return ticket


def begin_shared_prefix_physical_promotion(
    lane: DetachedSelfMTPLane,
    *,
    rows: int,
    reserve_tail: int,
    stream: Any,
    diagnostic_timing: bool = False,
    operations: StatePlaneOperations | None = None,
) -> SharedPrefixPhysicalPromotionPrequeue:
    """Stage an immutable replicated shared-state base before N=2 fan-out admission.

    The source lane remains authoritative and is only read.  Recurrent state is
    deliberately omitted: it is adopted from the real segmented rows after the
    first commit, exactly as in the ordinary promotion path.
    """

    rows = int(rows)
    reserve_tail = int(reserve_tail)
    if rows != 2:
        raise SegmentedPhysicalPromotionDeclined(
            "shared-prefix prequeue is qualified only for two rows"
        )
    if reserve_tail <= 0:
        raise ValueError("reserve_tail must be positive")
    prefix_id = lane.shared_state_identity
    if not prefix_id:
        raise SegmentedPhysicalPromotionDeclined(
            "shared-prefix prequeue requires a host prefix attestation"
        )
    operations = _require_promotion_operations(
        operations or getattr(lane, "state_plane_operations", None)
    )
    contracts = _promotion_contracts(
        operations, (*lane.caches.target, *lane.caches.draft)
    )
    started = time.perf_counter_ns()
    target, draft = [], []
    target_bases, draft_bases = [], []
    reserved_bytes = 0
    evaluation_values = []
    timing = (
        {
            "reserve_total_ns": 0,
            "validation_ns": 0,
            "allocation_graph_ns": 0,
            "copy_graph_ns": 0,
            "metadata_graph_ns": 0,
        }
        if diagnostic_timing
        else None
    )
    with mx.stream(stream):
        for cache in lane.caches.target:
            reservation = _provider_call(
                operations.reserve_promotion_layer,
                cache,
                reserve_tail,
                rows=rows,
                shared_prefix=True,
                timing=timing,
            )
            target.append(reservation.cache)
            target_bases.append(reservation.base)
            reserved_bytes += reservation.reserved_bytes
            evaluation_values.extend(reservation.evaluation_values)
        for cache in lane.caches.draft:
            reservation = _provider_call(
                operations.reserve_promotion_layer,
                cache,
                0,
                rows=rows,
                shared_prefix=True,
                timing=timing,
            )
            draft.append(reservation.cache)
            draft_bases.append(reservation.base)
            reserved_bytes += reservation.reserved_bytes
            evaluation_values.extend(reservation.evaluation_values)
        submit_started = time.perf_counter_ns() if diagnostic_timing else 0
        if evaluation_values:
            mx.async_eval(*evaluation_values)
        submit_ns = time.perf_counter_ns() - submit_started if diagnostic_timing else 0
    return SharedPrefixPhysicalPromotionPrequeue(
        prefix_id=str(prefix_id),
        rows=rows,
        stream=stream,
        reserve_tail=reserve_tail,
        physical=SelfMTPCachePair(target, draft),
        operations=operations,
        state_provider_revision=str(operations.provider_revision),
        contracts=contracts,
        target_bases=tuple(target_bases),
        draft_bases=tuple(draft_bases),
        creator_thread=threading.get_ident(),
        queued_ns=time.perf_counter_ns() - started,
        reserved_bytes=reserved_bytes,
        reserve_total_ns=(timing or {}).get("reserve_total_ns", 0),
        validation_ns=(timing or {}).get("validation_ns", 0),
        allocation_graph_ns=(timing or {}).get("allocation_graph_ns", 0),
        copy_graph_ns=(timing or {}).get("copy_graph_ns", 0),
        metadata_graph_ns=(timing or {}).get("metadata_graph_ns", 0),
        submit_ns=submit_ns,
    )
