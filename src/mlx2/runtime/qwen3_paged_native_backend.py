"""Completion-gated MLX backend for the default-off dense Qwen3 candidate.

The caller serializes all uses of this writer and its pool. A timeout or an
ambiguous submission poisons this backend; submitted leases remain pinned until
their native terminal callbacks are drained by the owner of the arena.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

from .paged_attention_native import (
    complete_packed_read_after_event,
    native_paged_attention_read_fp16,
    poll_native_paged_read_events,
    wait_native_paged_read_events,
)
from .paged_attention_pack import PackedTokenRead
from .paged_kv_token import PagedKVTokenOwner
from .paged_kv_write import (NativeWriteBackend, PagedKVWriteOwner, WriteSubmissionError,
                             WriteTicket)
from .paged_native_atomic_owner import (
    StagedReadTerminalProof,
    complete_staged_read_after_event,
)
from .paged_pack_diagnostics import describe_packed_read


def pack_head_spans(plane: object, spans: tuple, row_begin: int,
                    mx: object, stream: object) -> tuple:
    """Pack [row, KV head, channel] into exact head-local uint8 token spans."""
    result = []
    with mx.stream(stream):
        for span in spans:
            values = plane[
                row_begin + span.source_token_offset:
                row_begin + span.source_token_offset + span.token_count,
                span.kv_head,
                :,
            ]
            chunk = mx.contiguous(values, stream=stream).view(
                mx.uint8, stream=stream).reshape(-1, stream=stream)
            if (type(chunk) is not mx.array or chunk.dtype != mx.uint8 or
                    chunk.ndim != 1 or chunk.nbytes != span.byte_count):
                raise ValueError("packed head span has the wrong byte layout")
            result.append(chunk)
    return tuple(result)


class NativeQwen3PagedBackend:
    """A native implementation of ``PagedQwen3Backend`` for fp16 Qwen3 rows."""

    def __init__(self, writer: PagedKVWriteOwner, *, timeout_s: float = 10.0,
                 permit_candidate: bool = False, profile_host: bool = False) -> None:
        if not permit_candidate:
            raise RuntimeError("native Qwen3 paged backend requires explicit enablement")
        if (type(writer) is not PagedKVWriteOwner or
                type(writer.backend) is not NativeWriteBackend):
            raise TypeError("native Qwen3 backend requires an exact native arena writer")
        if not 0 < timeout_s <= 120:
            raise ValueError("completion timeout must be in (0, 120] seconds")
        self.writer = writer
        self.timeout_s = timeout_s
        self._dependency = None
        self._failed = False
        self.read_submissions = 0
        self.terminal_successes = 0
        self.staged_read_spans: list[int] = []
        self.direct_grouped_fence = False
        self.direct_grouped_fence_reads = 0
        self.profile_host = bool(profile_host)
        self.host_profile_ns = {name: 0 for name in (
            "span_pack", "write_submit", "read_enqueue", "terminal_drain",
            "plan_metadata", "graph_eval")}
        self.read_work: list[dict[str, int]] = []
        # Failed reads may finish after request teardown. Retain their leases
        # until a matching late native terminal callback arrives.
        self._orphaned_reads: dict[int, PackedTokenRead] = {}

    @property
    def profiling_enabled(self) -> bool:
        # CPU contract tests also construct a stub instance without __init__.
        return getattr(self, "profile_host", False) is True

    def profile_snapshot(self) -> dict:
        """Host wall spans and source-derived work; no GPU kernel timing."""
        result = self.profile_counters_snapshot()
        result["reads"] = tuple(dict(item) for item in self.read_work)
        return result

    def profile_counters_snapshot(self) -> dict:
        """Constant-size host/dispatch counters for the timed request loop."""
        result = {"host_ns": dict(self.host_profile_ns),
                  "read_count": len(self.read_work),
                  "direct_grouped_fence_reads": getattr(
                      self, "direct_grouped_fence_reads", 0)}
        if self.profiling_enabled:
            result["q1_stock_reduction_dispatches"] = getattr(
                self.writer.backend, "q1_stock_reduction_dispatch_count", lambda: 0)()
            result["q1_stock_singleton_dispatches"] = getattr(
                self.writer.backend, "q1_stock_singleton_dispatch_count", lambda: 0)()
            result["q1_stock_long_partial_dispatches"] = getattr(
                self.writer.backend, "q1_stock_long_partial_dispatch_count", lambda: 0)()
            result["q1_stock_long_reduce_dispatches"] = getattr(
                self.writer.backend, "q1_stock_long_reduce_dispatch_count", lambda: 0)()
            result["q1_stock_long_metadata_dispatches"] = getattr(
                self.writer.backend, "q1_stock_long_metadata_dispatch_count", lambda: 0)()
            result["q1_split_partial_dispatches"] = self.writer.backend.q1_split_partial_dispatch_count()
            result["q1_split_reduce_dispatches"] = self.writer.backend.q1_split_reduce_dispatch_count()
            result["q1_tile_dispatches"] = self.writer.backend.q1_tile_dispatch_count()
            result["q1_gather_dispatches"] = getattr(
                self.writer.backend, "q1_gather_dispatch_count", lambda: 0)()
            result["stock_sdpa_graph_calls"] = getattr(self.writer.backend, "stock_sdpa_graph_calls", 0)
            result["q1_metadata_dispatches"] = getattr(
                self.writer.backend, "q1_metadata_dispatch_count", lambda: 0)()
            for stripes in (8, 16, 32):
                result[f"q1_stripe_dispatches_{stripes}"] = getattr(
                    self.writer.backend, "q1_stripe_dispatch_count",
                    lambda _stripes: 0)(stripes)
            result["grouped_q1_writes"] = self.writer.backend.grouped_q1_write_count()
            result["grouped_multirow_writes"] = getattr(
                self.writer.backend, "grouped_multirow_write_count", lambda: 0)()
            result["grouped_multirow_rows"] = getattr(
                self.writer.backend, "grouped_multirow_row_count", lambda: 0)()
            for key, method in (
                ("grouped_n20_writes", "grouped_n20_write_count"),
                ("grouped_n20_rows", "grouped_n20_row_count"),
                ("prefill_long_n20_dispatches", "prefill_long_n20_dispatch_count"),
                ("q1_stock_long_n20_partial_dispatches", "q1_stock_long_n20_partial_dispatch_count"),
                ("q1_stock_long_n20_reduce_dispatches", "q1_stock_long_n20_reduce_dispatch_count"),
                ("q1_scalar_dispatches", "q1_scalar_dispatch_count"),
                ("q1_stock_long_n20_singleton_partial_dispatches", "q1_stock_long_n20_singleton_partial_dispatch_count"),
                ("q1_stock_long_n20_singleton_reduce_dispatches", "q1_stock_long_n20_singleton_reduce_dispatch_count"),
            ):
                result[key] = getattr(self.writer.backend, method, lambda: 0)()
            result["native_write_dispatches"] = self.writer.backend.write_dispatch_count()
            result["write_epoch"] = self.writer.ledger.completed_epoch
            result.update(getattr(self.writer.backend, "eval_submission_snapshot",
                                  lambda: {})())
        return result

    def _profile_read(self, use: PackedTokenRead) -> None:
        if self.profiling_enabled:
            self.host_profile_ns["plan_metadata"] += use.metadata_build_ns or 0
            self.read_work.append(describe_packed_read(use.plan).as_receipt())
    def drain_failed_read_events(self) -> int:
        """Retire late callbacks for reads abandoned by a failed request.

        Unknown or duplicate callbacks cannot justify releasing any lease.
        The caller retains this backend and arena until all epochs drain.
        """
        if getattr(self.writer.backend, "_closed", False):
            # Another owner may already have completed terminal-proved teardown.
            if self._orphaned_reads:
                raise RuntimeError("closed native arena still has orphaned read leases")
            return 0
        events = poll_native_paged_read_events(self.writer.backend)
        seen: set[int] = set()
        for epoch, _success in events:
            if epoch in seen or epoch not in self._orphaned_reads:
                self.writer.poisoned = True
                raise RuntimeError("unexpected failed native read callback")
            seen.add(epoch)
        for event in events:
            use = self._orphaned_reads[event[0]]
            complete_packed_read_after_event(use, event)
            del self._orphaned_reads[event[0]]
        return len(events)

    def _ready(self) -> None:
        if self._failed or self.writer.poisoned:
            raise RuntimeError("native Qwen3 paged backend is failed")

    def append_completed(self, owners: tuple[PagedKVTokenOwner, ...],
                         keys: object, values: object,
                         row_counts: tuple[int, ...]) -> None:
        self._ready()
        started = time.perf_counter_ns() if self.profiling_enabled else 0
        import mlx.core as mx

        if (type(owners) is not tuple or type(row_counts) is not tuple or
                not owners or len(owners) != len(row_counts) or
                len({id(owner) for owner in owners}) != len(owners) or
                any(type(owner) is not PagedKVTokenOwner or owner.writer is not self.writer or
                    owner.profile.dtype != getattr(self.writer.backend, "storage_dtype", "float16") for owner in owners) or
                any(type(count) is not int or count <= 0 for count in row_counts)):
            raise ValueError("native Qwen3 writes require distinct fp16 owners and positive rows")
        profile = owners[0].profile
        rows = sum(row_counts)
        expected = (rows, profile.kv_heads, profile.head_dim)
        if (any(owner.profile != profile for owner in owners) or
                any(type(plane) is not mx.array or plane.dtype != getattr(mx, getattr(self.writer.backend, "storage_dtype", "float16")) or
                    tuple(plane.shape) != expected for plane in (keys, values))):
            raise ValueError("projected K/V must be matching MLX fp16 token rows")
        # Preflight all spans and source arrays before the first page is reserved.
        prepared = []
        begin = 0
        for owner, count in zip(owners, row_counts):
            spans = owner.planned_spans(count)
            prepared.append((owner, count, owner.offset,
                             pack_head_spans(keys, spans, begin, mx, self.writer.backend.stream),
                             pack_head_spans(values, spans, begin, mx, self.writer.backend.stream)))
            begin += count
        if self.profiling_enabled:
            now = time.perf_counter_ns()
            self.host_profile_ns["span_pack"] += now - started
            started = now
        deadline = time.monotonic() + self.timeout_s
        try:
            for owner, count, old_offset, key_chunks, value_chunks in prepared:
                tickets = owner.append(key_chunks, value_chunks, token_count=count)
                if not tickets:
                    raise RuntimeError("native write accepted no tickets")
                # COW first returns a copy ticket. Polling may enqueue its
                # writes; capture their final dependency before publication.
                last_dependency = tickets[-1].dependency
                if self.profiling_enabled:
                    now = time.perf_counter_ns()
                    self.host_profile_ns["write_submit"] += now - started
                    started = now
                while owner.offset != old_offset + count:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("native Qwen3 KV publication timed out")
                    owner.poll_completions(wait_timeout_s=max(0.0, deadline - time.monotonic()))
                    if self.writer.poisoned:
                        raise RuntimeError("native Qwen3 KV write failed")
                    pending = owner._pending
                    if pending and pending[-1].dependency is not None:
                        last_dependency = pending[-1].dependency
                if last_dependency is None:
                    raise RuntimeError("native Qwen3 write has no dependency")
                self._dependency = last_dependency
                if self.profiling_enabled:
                    self.host_profile_ns["terminal_drain"] += time.perf_counter_ns() - started
                    started = time.perf_counter_ns()
        except BaseException:
            self._failed = True
            raise

    def read_completed(self, use: PackedTokenRead, queries: object,
                       *, scale: float,
                       terminal_proof: Callable[[PackedTokenRead, tuple[int, bool]], None] | None = None) -> object:
        self._ready()
        started = time.perf_counter_ns() if self.profiling_enabled else 0
        if (type(use) is not PackedTokenRead or use.writer is not self.writer or
                use.plan.dtype != getattr(self.writer.backend, "storage_dtype", "float16") or self._dependency is None):
            raise ValueError("native Qwen3 read requires completed fp16 writes")
        try:
            result = native_paged_attention_read_fp16(
                use, self.writer.backend, queries, self._dependency,
                scale=scale, permit_candidate=True)
        except BaseException:
            if use.state == "prepared":
                use.abort_before_submit()
            elif use.state == "submitted":
                self.read_submissions += 1
                self._wait_read(use)
            self._failed = True
            raise
        if use.state not in ("submitted", "closed"):
            raise RuntimeError("native Qwen3 read returned without submission")
        self.read_submissions += 1
        if self.profiling_enabled:
            now = time.perf_counter_ns()
            self.host_profile_ns["read_enqueue"] += now - started
            self._profile_read(use)
            started = now
        self._wait_read(use, terminal_proof=terminal_proof)
        if self.profiling_enabled:
            self.host_profile_ns["terminal_drain"] += time.perf_counter_ns() - started
        return result

    def append_packed_multirow(self, owners: tuple[PagedKVTokenOwner, PagedKVTokenOwner],
                               keys: object, values: object, counts: tuple[int, int],
                               *, permit_candidate: bool = False) -> tuple[WriteTicket]:
        """One explicit packed K/V scatter, shared lease, and terminal for two lanes.

        This never selects itself from append_staged. Shared-tail COW fails
        before submission; native evaluation checks source strides and backing
        bounds while the shared lease retains both owners.
        """
        self._ready()
        started = time.perf_counter_ns() if self.profiling_enabled else 0
        import mlx.core as mx

        arena = self.writer.backend
        native = getattr(arena, '_native', None)
        if (not permit_candidate or os.environ.get('MLX2_PAGED_GROUPED_MULTIROW_WRITE', '0') != '1' or
                not callable(getattr(native, 'grouped_multirow_write', None)) or
                not callable(getattr(native, 'grouped_multirow_write_count', None)) or
                not callable(getattr(native, 'grouped_multirow_row_count', None)) or
                type(owners) is not tuple or len(owners) != 2 or owners[0] is owners[1] or
                type(counts) is not tuple or len(counts) != 2 or
                any(type(n) is not int or not 1 <= n <= 8192 for n in counts) or
                any(type(owner) is not PagedKVTokenOwner or owner.writer is not self.writer
                    for owner in owners)):
            raise ValueError('packed multirow requires explicit two-owner native capability')
        profile = owners[0].profile
        total = sum(counts)
        shape = (total, profile.kv_heads, profile.head_dim)
        if (owners[1].profile != profile or
                any(type(plane) is not mx.array or tuple(plane.shape) != shape or
                    plane.dtype != getattr(mx, arena.storage_dtype)
                    for plane in (keys, values))):
            raise ValueError('packed multirow source shape/dtype differ')
        for owner, count in zip(owners, counts):
            owner.planned_spans(count, staged=True)
            if (owner.offset + count > 8192 or
                    (owner.sequence.handles and owner.offset % 64 and
                     self.writer.pool.references(owner.sequence.handles[-1]) >
                     owner._reuse_shared_tail_refs)):
                raise ValueError('packed multirow range or shared-tail COW differs')
        staged = []
        ticket = None
        try:
            plans = []
            for owner, count in zip(owners, counts):
                plans.append(owner.stage_packed_multirow(keys, values, count))
                staged.append(owner)
            touched = tuple(handle for item in plans for handle in item[0])
            if len(set(touched)) != len(touched) or len({h.page_id for h in touched}) != len(touched):
                raise ValueError('packed multirow destinations alias between lanes')
            page_ids = tuple(handle.page_id for item in plans for handle in item[1])
            table_begins = (0, len(plans[0][1]))
            ticket = self.writer.submit_packed_multirow_write(
                touched, keys=keys, values=values, counts=counts,
                starts=(plans[0][2], plans[1][2]),
                first_blocks=(plans[0][3], plans[1][3]),
                table_begins=table_begins, page_ids=page_ids,
                kv_heads=profile.kv_heads, dim=profile.head_dim)
            for owner in staged:
                owner.adopt_packed_multirow(ticket)
        except WriteSubmissionError as exc:
            provisional = self.writer._tickets[exc.epoch]
            for owner in staged:
                owner.retain_ambiguous_packed_multirow(provisional)
            self._failed = True
            raise
        except BaseException:
            if ticket is not None:
                self.writer.poisoned = True
                for owner in staged:
                    owner.fail_submitted_packed_multirow(ticket)
            else:
                for owner in staged:
                    if not owner._pending:
                        owner.abort_prepared_packed_multirow()
            self._failed = True
            raise
        if self.profiling_enabled:
            self.host_profile_ns['write_submit'] += time.perf_counter_ns() - started
        return (ticket,)

    def append_packed_multirow_n20(self, owners: tuple[PagedKVTokenOwner, ...],
                               keys: object, values: object, counts: tuple[int, ...],
                               *, permit_candidate: bool = False) -> tuple[WriteTicket]:
        """One explicit N20 K/V scatter, shared lease, and terminal for 1..20 lanes.

        This never selects itself from append_staged. Shared-tail COW fails
        before submission; native evaluation checks source strides and backing
        bounds while the shared lease retains both owners.
        """
        self._ready()
        started = time.perf_counter_ns() if self.profiling_enabled else 0
        import mlx.core as mx

        arena = self.writer.backend
        native = getattr(arena, '_native', None)
        if (not permit_candidate or os.environ.get('MLX2_PAGED_PACKED_N20', '0') != '1' or
                not callable(getattr(native, 'grouped_multirow_write_n20', None)) or
                not callable(getattr(native, 'grouped_n20_write_count', None)) or
                not callable(getattr(native, 'grouped_n20_row_count', None)) or
                type(owners) is not tuple or not 1 <= len(owners) <= 20 or len({id(owner) for owner in owners}) != len(owners) or
                type(counts) is not tuple or len(counts) != len(owners) or
                any(type(n) is not int or not 1 <= n <= 8192 for n in counts) or
                any(type(owner) is not PagedKVTokenOwner or owner.writer is not self.writer
                    for owner in owners)):
            raise ValueError('packed N20 requires explicit owner/native capability')
        profile = owners[0].profile
        total = sum(counts)
        shape = (total, profile.kv_heads, profile.head_dim)
        if (any(owner.profile != profile for owner in owners) or
                profile.kv_heads != 4 or profile.head_dim != 256 or
                profile.dtype != "bfloat16" or
                any(type(plane) is not mx.array or tuple(plane.shape) != shape or
                    plane.dtype != getattr(mx, arena.storage_dtype)
                    for plane in (keys, values))):
            raise ValueError('packed multirow source shape/dtype differ')
        for owner, count in zip(owners, counts):
            owner.planned_spans(count, staged=True)
            if (owner.offset + count > 8192 or owner.sequence.retained_start != 0 or
                    owner.sequence.first_block != 0 or
                    (owner.sequence.handles and owner.offset % 64 and
                     self.writer.pool.references(owner.sequence.handles[-1]) >
                     owner._reuse_shared_tail_refs)):
                raise ValueError('packed multirow range or shared-tail COW differs')
        staged = []
        ticket = None
        try:
            plans = []
            for owner, count in zip(owners, counts):
                plans.append(owner.stage_packed_multirow(keys, values, count))
                staged.append(owner)
            touched = tuple(handle for item in plans for handle in item[0])
            if len(set(touched)) != len(touched) or len({h.page_id for h in touched}) != len(touched):
                raise ValueError('packed multirow destinations alias between lanes')
            page_ids = tuple(handle.page_id for item in plans for handle in item[1])
            if len(set(page_ids)) != len(page_ids) or len(page_ids) > 2560:
                raise ValueError('packed N20 page tables alias or exceed capacity')
            table_begins = tuple(sum(len(item[1]) for item in plans[:index])
                                 for index in range(len(plans)))
            ticket = self.writer.submit_packed_multirow_write_n20(
                touched, keys=keys, values=values, counts=counts,
                starts=tuple(item[2] for item in plans),
                first_blocks=tuple(item[3] for item in plans),
                table_begins=table_begins, page_ids=page_ids,
                kv_heads=profile.kv_heads, dim=profile.head_dim)
            for owner in staged:
                owner.adopt_packed_multirow(ticket)
        except WriteSubmissionError as exc:
            provisional = self.writer._tickets[exc.epoch]
            for owner in staged:
                owner.retain_ambiguous_packed_multirow(provisional)
            self._failed = True
            raise
        except BaseException:
            if ticket is not None:
                self.writer.poisoned = True
                for owner in staged:
                    owner.fail_submitted_packed_multirow(ticket)
            else:
                for owner in staged:
                    if not owner._pending:
                        owner.abort_prepared_packed_multirow()
            self._failed = True
            raise
        if self.profiling_enabled:
            self.host_profile_ns['write_submit'] += time.perf_counter_ns() - started
        return (ticket,)

    def append_staged_grouped_q1_n20(self, owners: tuple[PagedKVTokenOwner, ...],
                                     keys: object, values: object,
                                     *, permit_candidate: bool = False) -> tuple[WriteTicket]:
        return self.append_packed_multirow_n20(
            owners, keys, values, (1,) * len(owners),
            permit_candidate=permit_candidate)

    def prepare_read_n20(self, owners: tuple[PagedKVTokenOwner, ...],
                         counts: tuple[int, ...], *, query_heads: int = 24,
                         permit_candidate: bool = False,
                         profile_host: bool = False) -> PackedTokenRead:
        self._ready()
        from .paged_attention_pack import prepare_staged_token_read_n20
        use = prepare_staged_token_read_n20(
            owners, counts, query_heads=query_heads,
            permit_candidate=permit_candidate, profile_host=profile_host)
        self._profile_read(use)
        return use

    def append_staged(self, owners: tuple[PagedKVTokenOwner, ...],
                      keys: object, values: object,
                      row_counts: tuple[int, ...]) -> tuple:
        """Enqueue private K/V proposals and return every native dependency."""
        self._ready()
        started = time.perf_counter_ns() if self.profiling_enabled else 0
        import mlx.core as mx

        if (type(owners) is not tuple or type(row_counts) is not tuple or
                not owners or len(owners) != len(row_counts) or
                len({id(owner) for owner in owners}) != len(owners) or
                any(type(owner) is not PagedKVTokenOwner or owner.writer is not self.writer or
                    owner.profile.dtype != getattr(self.writer.backend, "storage_dtype", "float16") for owner in owners) or
                any(type(count) is not int or count <= 0 for count in row_counts)):
            raise ValueError("staged native writes require distinct fp16 owners")
        profile = owners[0].profile
        rows = sum(row_counts)
        expected = (rows, profile.kv_heads, profile.head_dim)
        if (any(owner.profile != profile for owner in owners) or
                any(type(plane) is not mx.array or plane.dtype != getattr(mx, getattr(self.writer.backend, "storage_dtype", "float16")) or
                    tuple(plane.shape) != expected for plane in (keys, values))):
            raise ValueError("staged K/V must match the fp16 token rows")
        grouped_q1 = (os.environ.get("MLX2_PAGED_GROUPED_Q1_WRITE") == "1" and
                      len(owners) == 2 and row_counts == (1, 1) and
                      all(not (owner.sequence.handles and owner.offset % 64 and
                               self.writer.pool.references(owner.sequence.handles[-1]) >
                               owner._reuse_shared_tail_refs)
                          for owner in owners))
        if grouped_q1:
            staged = []
            ticket = None
            try:
                destinations = []
                for owner in owners:
                    destinations.append(owner.stage_grouped_q1(keys, values))
                    staged.append(owner)
                ticket = self.writer.submit_grouped_q1_write(
                    (destinations[0][0], destinations[1][0]),
                    (destinations[0][1], destinations[1][1]),
                    keys=keys, values=values,
                    kv_heads=profile.kv_heads, dim=profile.head_dim)
                for owner in staged:
                    owner.adopt_grouped_q1(ticket)
            except WriteSubmissionError as exc:
                provisional = self.writer._tickets[exc.epoch]
                for owner in staged:
                    owner.retain_ambiguous_grouped_q1(provisional)
                self._failed = True
                raise
            except BaseException:
                if ticket is not None:
                    self.writer.poisoned = True
                    for owner in staged:
                        owner.fail_submitted_grouped_q1(ticket)
                else:
                    for owner in staged:
                        if not owner._pending:
                            owner.abort_prepared_grouped_q1()
                self._failed = True
                raise
            if self.profiling_enabled:
                self.host_profile_ns["write_submit"] += time.perf_counter_ns() - started
            return (ticket,)
        prepared = []
        begin = 0
        for owner, count in zip(owners, row_counts):
            spans = owner.planned_spans(count, staged=True)
            prepared.append((owner, count,
                             pack_head_spans(keys, spans, begin, mx, self.writer.backend.stream),
                             pack_head_spans(values, spans, begin, mx, self.writer.backend.stream)))
            begin += count
        if self.profiling_enabled:
            now = time.perf_counter_ns()
            self.host_profile_ns["span_pack"] += now - started
            started = now
        tickets = []
        try:
            for owner, count, key_chunks, value_chunks in prepared:
                tickets.extend(owner.append_staged(key_chunks, value_chunks,
                                                   token_count=count))
            if not tickets or any(ticket.dependency is None for ticket in tickets):
                raise RuntimeError("staged native write lacks a dependency")
        except BaseException:
            self._failed = True
            self.writer.poisoned = True
            raise
        if self.profiling_enabled:
            self.host_profile_ns["write_submit"] += time.perf_counter_ns() - started
        return tuple(tickets)

    def read_staged(self, use: PackedTokenRead, queries: object,
                    tickets: tuple, *, scale: float) -> object:
        """Submit one read after every lane's copy/write dependency."""
        self._ready()
        started = time.perf_counter_ns() if self.profiling_enabled else 0
        import mlx.core as mx

        if (type(use) is not PackedTokenRead or use.writer is not self.writer or
                use.plan.dtype != getattr(self.writer.backend, "storage_dtype", "float16") or not tickets or
                any(ticket.dependency is None for ticket in tickets)):
            raise ValueError("staged read requires one exact native write set")
        direct_fence = bool(getattr(self, "direct_grouped_fence", False) and
                            len(tickets) == 1 and
                            type(tickets[0]) is WriteTicket and
                            tickets[0].grouped_q1 is True)
        try:
            if direct_fence:
                # The grouped native write itself emits the one-byte success
                # fence. Passing that exact output as the read input preserves
                # the graph edge without an extra ones/depends/contiguous chain.
                dependency = tickets[0].dependency
                if (type(dependency) is not mx.array or
                        dependency.dtype != mx.uint8 or
                        dependency.ndim != 1 or dependency.size != 1):
                    raise ValueError("grouped Q1 dependency is not one native byte")
            else:
                with mx.stream(self.writer.backend.stream):
                    dependency = mx.depends(
                        # The native attention kernel treats a zero dependency
                        # byte as a failed write fence and returns zero output.
                        mx.ones((1,), dtype=mx.uint8),
                        tuple(ticket.dependency for ticket in tickets),
                    )
                    dependency = mx.contiguous(dependency, stream=self.writer.backend.stream)
            result = native_paged_attention_read_fp16(
                use, self.writer.backend, queries, dependency,
                scale=scale, permit_candidate=True)
        except BaseException:
            if use.state == "submitted":
                self.read_submissions += 1
                self._orphaned_reads[use.lease.epoch] = use
            elif use.state == "prepared":
                use.abort_before_submit()
            self._failed = True
            self.writer.poisoned = True
            raise
        self.read_submissions += 1
        if direct_fence:
            self.direct_grouped_fence_reads = getattr(
                self, "direct_grouped_fence_reads", 0) + 1
        self.staged_read_spans.append(len(use.plan.spans))
        if self.profiling_enabled:
            self.host_profile_ns["read_enqueue"] += time.perf_counter_ns() - started
            self._profile_read(use)
        return result

    def abort_deferred_q1(self, owners: tuple[PagedKVTokenOwner, ...],
                          uses: tuple[PackedTokenRead, ...]) -> None:
        """Flush built roots before rolling back a partly built Q1 graph.

        A staged lease was submitted at graph construction. Without this
        flush an exception before the final logits evaluation could strand a
        lease that never reaches a native command buffer or terminal callback.
        An ambiguous evaluation keeps the writer poisoned and all unresolved
        leases pinned for late callback or failed-arena recovery.
        """
        import mlx.core as mx

        arena = self.writer.backend
        roots = arena.deferred_q1_roots()
        arena.deferred_q1_failure_flushes += 1
        try:
            for use in uses:
                if use.state == "prepared":
                    use.abort_before_submit()
            if roots:
                mx.eval(*roots)
            pending = {use.lease.epoch: use for use in uses
                       if use.state == "submitted"}
            deadline = time.monotonic() + self.timeout_s
            while pending or self.writer.pending_epochs:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("deferred Q1 failure terminal drain timed out")
                if self.writer.pending_epochs:
                    self.writer.poll_completions(wait_timeout_s=min(remaining, .01))
                events = poll_native_paged_read_events(arena)
                if not events and pending:
                    events = wait_native_paged_read_events(arena, min(remaining, .01))
                for event in events:
                    use = pending.pop(event[0], None)
                    if use is None:
                        raise RuntimeError("unknown deferred Q1 read terminal")
                    complete_staged_read_after_event(use, event)
                    self._orphaned_reads.pop(event[0], None)
                    self.terminal_successes += 1
            for owner in owners:
                owner.poll_completions()
        except BaseException:
            self._failed = True
            self.writer.poisoned = True
            for use in uses:
                if use.state == "submitted":
                    self._orphaned_reads[use.lease.epoch] = use
            raise
        self._failed = True
        self.writer.poisoned = True

    def retain_deferred_q1_roots(self, uses: tuple[PackedTokenRead, ...]) -> bool:
        """Keep MLX roots while any native write or read can still access pages."""
        return bool(self.writer.pending_epochs or self.writer.ledger.pending_count or
                    self._orphaned_reads or
                    any(use.state in ("prepared", "submitted") for use in uses))

    def drain_staged(self, owners: tuple[PagedKVTokenOwner, ...],
                     uses: tuple[PackedTokenRead, ...]) -> tuple[StagedReadTerminalProof, ...]:
        """Require every exact write/copy/read terminal before accepting KV."""
        started = time.perf_counter_ns() if self.profiling_enabled else 0
        deadline = time.monotonic() + self.timeout_s
        pending = {use.lease.epoch: use for use in uses}
        if len(pending) != len(uses) or not pending:
            raise ValueError("distinct staged read epochs are required")
        proofs: dict[int, StagedReadTerminalProof] = {}
        try:
            while pending or self.writer.pending_epochs:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("staged native graph terminal drain timed out")
                # Both queues have separate callback conditions. Bound each
                # wait so a terminal in the other queue is not starved.
                if self.writer.pending_epochs:
                    self.writer.poll_completions(wait_timeout_s=min(remaining, 0.01))
                events = poll_native_paged_read_events(self.writer.backend)
                if not events and pending:
                    events = wait_native_paged_read_events(
                        self.writer.backend, min(remaining, 0.01))
                for event in events:
                    epoch = event[0]
                    use = pending.pop(epoch, None)
                    if use is None:
                        raise RuntimeError("unknown or duplicate staged read terminal")
                    proof = complete_staged_read_after_event(use, event)
                    proofs[epoch] = proof
                    self.terminal_successes += 1
                if self.writer.poisoned:
                    raise RuntimeError("staged native write terminal failed")
            for owner in owners:
                owner.poll_completions()
            if any(owner._pending or owner._stage is not None for owner in owners):
                raise RuntimeError("staged native writes did not publish after terminals")
        except BaseException:
            self._failed = True
            self.writer.poisoned = True
            for use in uses:
                if use.state == "submitted":
                    self._orphaned_reads[use.lease.epoch] = use
            raise
        if self.profiling_enabled:
            self.host_profile_ns["terminal_drain"] += time.perf_counter_ns() - started
        return tuple(proofs[use.lease.epoch] for use in uses)

    def _wait_read(self, use: PackedTokenRead, *,
                   terminal_proof: Callable[[PackedTokenRead, tuple[int, bool]], None] | None = None) -> None:
        deadline = time.monotonic() + self.timeout_s
        try:
            while True:
                # Drain callbacks already queued before blocking. The same
                # native mutex protects the queue and its wake predicate.
                events = poll_native_paged_read_events(self.writer.backend)
                if not events:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("native Qwen3 attention read timed out")
                    events = wait_native_paged_read_events(self.writer.backend, remaining)
                matching = [event for event in events if event[0] == use.lease.epoch]
                if len(matching) > 1 or len(matching) != len(events):
                    raise RuntimeError("unexpected or duplicate native read callback")
                if matching:
                    if terminal_proof is None:
                        if not complete_packed_read_after_event(use, matching[0]):
                            raise RuntimeError("native Qwen3 attention read failed")
                    else:
                        # The atomic branch consumes this exact event and owns
                        # the one completion. Closed state alone proves nothing.
                        try:
                            terminal_proof(use, matching[0])
                        except BaseException:
                            # A branch can reject the layer/table before it
                            # consumes the callback. The native event has
                            # nonetheless proved the read terminal: release
                            # its pin once, then keep the proof failure.
                            if use.state == "submitted":
                                complete_packed_read_after_event(use, matching[0])
                            raise
                        if use.state != "closed":
                            raise RuntimeError("terminal proof did not close native read")
                    self.terminal_successes += 1
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError("native Qwen3 attention read timed out")
        except BaseException:
            self._failed = True
            if use.state == "submitted":
                self._orphaned_reads[use.lease.epoch] = use
            raise
