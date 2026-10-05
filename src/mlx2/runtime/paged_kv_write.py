"""Default-off host submission boundary for one native paged KV arena.

The injected backend must own both byte planes, enqueue a write on one
explicit stream, and return a dependency consumed by later reads. Its terminal
events must cover every possible arena access by the submitted command buffer.
This host contract does not claim that the current native scaffold exposes a
Python write binding or that any GPU behavior has been verified.

All methods, including completion polling, must run under the same host lock
as every mutation of the pool. A submission exception is ambiguous: the page
pin remains until a terminal event arrives, and the owner becomes poisoned.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Protocol

from .paged_attention_plan import U64_MAX, PageHandle, _positive, _uint
from .paged_kv_native import CompletionLease, PagedKVCompletionLedger
from .paged_kv_pool import PagedKVPool


class WriteBackend(Protocol):
    @property
    def plane_bytes(self) -> int: ...

    def write(
        self, key_bytes: object, value_bytes: object, offset: int, byte_count: int, epoch: int
    ) -> object: ...

    def copy_page(
        self, source_offset: int, destination_offset: int, byte_count: int, epoch: int
    ) -> object: ...

    def poll_completions(self) -> list[tuple[int, bool]]: ...

    def wait_completions(self, timeout_seconds: float) -> list[tuple[int, bool]]: ...

    def close_after_terminal(self) -> None: ...


class NativeWriteBackend:
    """Opt-in MLX byte-plane backend on one explicit GPU stream.

    ``write`` schedules the dependency through MLX's asynchronous evaluator.
    The caller must retain that dependency in every later reader graph and
    drain terminal events before releasing its host generation lease.
    """

    def __init__(self, plane_bytes: int, stream: object, *, permit_candidate: bool = False,
                 storage_dtype: str = "float16") -> None:
        if not permit_candidate:
            raise RuntimeError("native paged KV writes require explicit candidate enablement")
        import _paged_kv_native as native
        import mlx.core as mx

        if type(stream) is not mx.Stream or stream.device != mx.gpu:
            raise TypeError("an explicit MLX GPU Stream is required")
        _positive("plane bytes", plane_bytes, U64_MAX)
        self._mx = mx
        self._native = native
        self.stream = stream
        if storage_dtype not in ("float16", "bfloat16"):
            raise ValueError("native storage dtype must be float16 or bfloat16")
        self.storage_dtype = storage_dtype
        self._arena = (native.create_arena(plane_bytes) if storage_dtype == "float16" else
                       native.create_arena(plane_bytes, storage_dtype=storage_dtype))
        self.plane_bytes = native.plane_bytes(self._arena)
        self._closed = False
        self.defer_staged_q1_eval = False
        self.defer_staged_q1_writes = False
        self._deferred_q1_write_roots: list[object] = []
        self._deferred_q1_read_roots: list[object] = []
        self.grouped_write_async_evals = 0
        self.staged_read_async_evals = 0
        self.stock_sdpa_graph_calls = 0
        self.deferred_q1_write_roots = 0
        self.deferred_q1_read_roots = 0
        self.deferred_q1_final_evals = 0
        self.deferred_q1_failure_flushes = 0

    def eval_submission_snapshot(self) -> dict[str, int]:
        return {
            "grouped_write_async_evals": self.grouped_write_async_evals,
            "staged_read_async_evals": self.staged_read_async_evals,
            "deferred_q1_write_roots": self.deferred_q1_write_roots,
            "deferred_q1_read_roots": self.deferred_q1_read_roots,
            "deferred_q1_final_evals": self.deferred_q1_final_evals,
            "deferred_q1_failure_flushes": self.deferred_q1_failure_flushes,
        }

    def begin_deferred_q1(self, *, writes_only: bool = False) -> None:
        if type(writes_only) is not bool:
            raise TypeError("writes_only must be boolean")
        if (self._closed or self.defer_staged_q1_eval or
                getattr(self, "defer_staged_q1_writes", False) or
                self._deferred_q1_write_roots or self._deferred_q1_read_roots):
            raise RuntimeError("deferred Q1 graph state is not idle")
        self.defer_staged_q1_eval = not writes_only
        self.defer_staged_q1_writes = True

    def deferred_q1_roots(self) -> tuple[object, ...]:
        return tuple(self._deferred_q1_write_roots + self._deferred_q1_read_roots)

    def end_deferred_q1(self, *, retain_roots: bool = False) -> None:
        self.defer_staged_q1_eval = False
        self.defer_staged_q1_writes = False
        if not retain_roots:
            self._deferred_q1_write_roots.clear()
            self._deferred_q1_read_roots.clear()

    def q1_tile_dispatch_count(self) -> int:
        """Research-only count of encoded tile reads in this native arena."""
        if self._closed:
            raise RuntimeError("native arena is closed")
        return int(self._native.q1_tile_dispatch_count(self._arena))

    def q1_gather_dispatch_count(self) -> int:
        if self._closed:
            raise RuntimeError("native arena is closed")
        return int(self._native.q1_gather_dispatch_count(self._arena))

    def q1_stock_reduction_dispatch_count(self) -> int:
        if self._closed:
            raise RuntimeError("native arena is closed")
        counter = getattr(self._native, "q1_stock_reduction_dispatch_count", None)
        return 0 if counter is None else int(counter(self._arena))

    def q1_stock_singleton_dispatch_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter = getattr(self._native, "q1_stock_singleton_dispatch_count", None)
        return int(counter(self._arena)) if callable(counter) else 0

    def prefill_matrix_dispatch_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter = getattr(self._native, "prefill_matrix_dispatch_count", None)
        return int(counter(self._arena)) if callable(counter) else 0

    def q1_stock_long_partial_dispatch_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter = getattr(self._native, "q1_stock_long_partial_dispatch_count", None)
        return 0 if counter is None else int(counter(self._arena))

    def q1_stock_long_reduce_dispatch_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter = getattr(self._native, "q1_stock_long_reduce_dispatch_count", None)
        return 0 if counter is None else int(counter(self._arena))

    def q1_stock_long_metadata_dispatch_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter = getattr(self._native, "q1_stock_long_metadata_dispatch_count", None)
        return 0 if counter is None else int(counter(self._arena))

    def q1_split_partial_dispatch_count(self) -> int:
        return int(self._native.q1_split_partial_dispatch_count(self._arena))

    def q1_split_reduce_dispatch_count(self) -> int:
        return int(self._native.q1_split_reduce_dispatch_count(self._arena))

    def q1_stripe_dispatch_count(self, stripes: int) -> int:
        if self._closed:
            raise RuntimeError("native arena is closed")
        return int(self._native.q1_stripe_dispatch_count(self._arena, stripes))

    def q1_metadata_dispatch_count(self) -> int:
        if self._closed:
            raise RuntimeError("native arena is closed")
        return int(self._native.q1_metadata_dispatch_count(self._arena))

    def grouped_q1_write_count(self) -> int:
        if self._closed:
            raise RuntimeError("native arena is closed")
        return int(self._native.grouped_q1_write_count(self._arena))

    def grouped_multirow_write_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter = getattr(self._native, "grouped_multirow_write_count", None)
        return 0 if not callable(counter) else int(counter(self._arena))

    def grouped_multirow_row_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter = getattr(self._native, "grouped_multirow_row_count", None)
        return 0 if not callable(counter) else int(counter(self._arena))

    def grouped_n20_write_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter = getattr(self._native, "grouped_n20_write_count", None)
        return int(counter(self._arena)) if callable(counter) else 0

    def grouped_n20_row_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter = getattr(self._native, "grouped_n20_row_count", None)
        return int(counter(self._arena)) if callable(counter) else 0

    def prefill_long_n20_dispatch_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter = getattr(self._native, "prefill_long_n20_dispatch_count", None)
        return int(counter(self._arena)) if callable(counter) else 0

    def q1_stock_long_n20_partial_dispatch_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter = getattr(self._native, "q1_stock_long_n20_partial_dispatch_count", None)
        return int(counter(self._arena)) if callable(counter) else 0

    def q1_stock_long_n20_reduce_dispatch_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter = getattr(self._native, "q1_stock_long_n20_reduce_dispatch_count", None)
        return int(counter(self._arena)) if callable(counter) else 0

    def q1_scalar_dispatch_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter = getattr(self._native, "q1_scalar_dispatch_count", None)
        return int(counter(self._arena)) if callable(counter) else 0

    def q1_stock_long_n20_singleton_partial_dispatch_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter=getattr(self._native,"q1_stock_long_n20_singleton_partial_dispatch_count",None)
        return int(counter(self._arena)) if callable(counter) else 0

    def q1_stock_long_n20_singleton_reduce_dispatch_count(self) -> int:
        if self._closed: raise RuntimeError("native arena is closed")
        counter=getattr(self._native,"q1_stock_long_n20_singleton_reduce_dispatch_count",None)
        return int(counter(self._arena)) if callable(counter) else 0

    def grouped_multirow_write_n20(self, keys: object, values: object,
                                   counts: tuple[int, ...], starts: tuple[int, ...],
                                   first_blocks: tuple[int, ...], table_begins: tuple[int, ...],
                                   page_ids: tuple[int, ...], kv_heads: int, dim: int,
                                   epoch: int) -> object:
        if (self._closed or self.defer_staged_q1_eval or self.defer_staged_q1_writes or
                os.environ.get("MLX2_PAGED_PACKED_N20", "0") != "1"):
            raise RuntimeError("N20 write requires an open eager candidate arena")
        native = getattr(self._native, "grouped_multirow_write_n20", None)
        if not callable(native):
            raise ValueError("native N20 write ABI unavailable")
        dependency = native(self._arena, keys, values, list(counts), list(starts),
                            list(first_blocks), list(table_begins), list(page_ids),
                            kv_heads, dim, epoch, self.stream, permit_candidate=True)
        self._mx.async_eval(dependency)
        return dependency

    def grouped_multirow_write(self, keys: object, values: object,
                               counts: tuple[int, int], starts: tuple[int, int],
                               first_blocks: tuple[int, int], table_begins: tuple[int, int],
                               page_ids: tuple[int, ...], kv_heads: int, dim: int,
                               epoch: int) -> object:
        if self._closed or self.defer_staged_q1_eval or self.defer_staged_q1_writes:
            raise RuntimeError("grouped multirow write requires an open eager arena")
        native = getattr(self._native, "grouped_multirow_write", None)
        if not callable(native):
            raise ValueError("native grouped multirow write ABI unavailable")
        dependency = native(self._arena, keys, values, list(counts), list(starts),
                            list(first_blocks), list(table_begins), list(page_ids),
                            kv_heads, dim, epoch, self.stream, permit_candidate=True)
        self._mx.async_eval(dependency)
        return dependency

    def write_dispatch_count(self) -> int:
        if self._closed:
            raise RuntimeError("native arena is closed")
        return int(self._native.write_dispatch_count(self._arena))

    def grouped_q1_write(self, keys: object, values: object,
                         pages: tuple[int, int], slots: tuple[int, int],
                         kv_heads: int, dim: int, epoch: int) -> object:
        """One native dependency and terminal for two private Q1 rows."""
        if self._closed:
            raise RuntimeError("native paged arena has been torn down")
        dependency = self._native.grouped_q1_write(
            self._arena, keys, values, list(pages), list(slots), kv_heads,
            dim, epoch, self.stream, permit_candidate=True)
        if self.defer_staged_q1_eval or getattr(self, "defer_staged_q1_writes", False):
            self._deferred_q1_write_roots.append(dependency)
            self.deferred_q1_write_roots += 1
        else:
            self._mx.async_eval(dependency)
            self.grouped_write_async_evals += 1
        return dependency

    def write(
        self, key_bytes: object, value_bytes: object, offset: int,
        byte_count: int, epoch: int,
    ) -> object:
        if self._closed:
            raise RuntimeError("native paged arena has been torn down")
        dependency = self._native.write(
            self._arena, key_bytes, value_bytes, offset, byte_count, epoch, self.stream
        )
        self._mx.async_eval(dependency)
        return dependency

    def validate_sources(self, keys: object, values: object) -> bool:
        """Use the native binding's shape/stride check before reserving pages."""
        return bool(self._native.validate_source_types(keys, values, self.stream))

    def depend_source(self, source: object, dependency: object) -> object:
        """Order a COW destination write after its same-stream page copy."""
        if self._closed:
            raise RuntimeError("native paged arena has been torn down")
        result = self._mx.depends(source, dependency)
        return self._mx.contiguous(result, stream=self.stream)

    def copy_page(
        self, source_offset: int, destination_offset: int, byte_count: int, epoch: int
    ) -> object:
        if self._closed:
            raise RuntimeError("native paged arena has been torn down")
        dependency = self._native.copy_page(
            self._arena, source_offset, destination_offset, byte_count, epoch, self.stream
        )
        self._mx.async_eval(dependency)
        return dependency

    def poll_completions(self) -> list[tuple[int, bool]]:
        if self._closed:
            return []
        return self._native.poll_completions(self._arena)

    def wait_completions(self, timeout_seconds: float) -> list[tuple[int, bool]]:
        if self._closed:
            return []
        return self._native.wait_completions(self._arena, timeout_seconds)

    def close_after_terminal(self) -> None:
        """Synchronize the explicit stream and drop this backend's arena handle.

        The writer first proves every submitted read/write terminal. MLX graph
        nodes may retain their own arena references; no slot in the old pool is
        made reusable by this one-way operation.
        """
        if self._closed:
            return
        self._mx.synchronize(self.stream)
        self._arena = None
        self._closed = True

    def diagnostic_read(
        self, dependency: object, offset: int, byte_count: int,
        *, permit_diagnostic: bool = False,
    ) -> tuple[object, object]:
        """Return bounded copied K/V bytes; caller must pin and sync the read."""
        if self._closed:
            raise RuntimeError("native paged arena has been torn down")
        return tuple(self._native.diagnostic_read(
            self._arena, dependency, offset, byte_count, self.stream,
            permit_diagnostic,
        ))


@dataclass(frozen=True, eq=False)
class WriteTicket:
    lease: CompletionLease
    handle: PageHandle
    dependency: object | None
    grouped_q1: bool = False
    packed_multirow: bool = False

    @property
    def epoch(self) -> int:
        return self.lease.epoch


@dataclass(frozen=True, eq=False)
class CopyTicket:
    """An unpublished page copy with both generations pinned to terminal status."""

    lease: CompletionLease
    source: PageHandle
    destination: PageHandle
    dependency: object | None

    @property
    def epoch(self) -> int:
        return self.lease.epoch


@dataclass(frozen=True)
class WriteCompletion:
    ticket: WriteTicket | CopyTicket
    succeeded: bool


class WriteSubmissionError(RuntimeError):
    """Submission is ambiguous; ``epoch`` remains pinned until terminal proof."""

    def __init__(self, epoch: int) -> None:
        super().__init__(f"paged KV write submission failed at epoch {epoch}; drain required")
        self.epoch = epoch


class PagedKVWriteOwner:
    """Validate, pin, enqueue, and retire writes without publishing KV state."""

    def __init__(
        self, pool: PagedKVPool, backend: WriteBackend, *, page_bytes: int,
        permit_candidate: bool = False,
    ) -> None:
        if not permit_candidate:
            raise RuntimeError("paged KV writes require explicit candidate enablement")
        if type(pool) is not PagedKVPool:
            raise TypeError("pool must be a PagedKVPool")
        _positive("page bytes", page_bytes, U64_MAX)
        if type(backend.plane_bytes) is not int or backend.plane_bytes != pool.capacity * page_bytes:
            raise ValueError("native plane size does not match pool geometry")
        self.pool = pool
        self.backend = backend
        self.page_bytes = page_bytes
        self.ledger = PagedKVCompletionLedger(pool)
        self.poisoned = False
        self._tickets: dict[int, WriteTicket | CopyTicket] = {}
        self._finished: dict[int, bool] = {}
        self._failed_arena_torn_down = False

    @property
    def pending_epochs(self) -> tuple[int, ...]:
        return tuple(self._tickets)

    @property
    def failed_arena_torn_down(self) -> bool:
        return self._failed_arena_torn_down

    def teardown_failed_arena(self) -> None:
        """One-way failed-arena close after all native uses are terminal.

        Unknown or ambiguous in-flight epochs keep the arena and pins alive.
        The backend synchronizes its stream and releases its arena handle;
        quarantined pool slots remain unavailable for this pool's lifetime.
        """
        if self._failed_arena_torn_down:
            return
        if not self.poisoned:
            raise RuntimeError("only a poisoned native arena can use failed teardown")
        if self._tickets or self.ledger.pending_count:
            raise RuntimeError("native arena still has unproven in-flight epochs")
        self.backend.close_after_terminal()
        self._failed_arena_torn_down = True

    def submit_write(
        self, handle: PageHandle, *, within_page_offset: int, byte_count: int,
        key_bytes: object, value_bytes: object,
    ) -> WriteTicket:
        """Submit one page-local K/V span; retain the generation through completion.

        Sources are opaque MLX arrays to this layer. The native binding must
        validate dtype, contiguity, shape, byte length, and explicit stream.
        The returned dependency is not publication or a completion proof.
        """
        if self.poisoned:
            raise RuntimeError("paged KV arena is poisoned")
        _uint("within-page offset", within_page_offset, U64_MAX)
        _positive("write byte count", byte_count, U64_MAX)
        if within_page_offset >= self.page_bytes or byte_count > self.page_bytes - within_page_offset:
            raise ValueError("write crosses a physical page")
        # prepare validates the generation and pins it before native encoding.
        lease = self.ledger.prepare((handle,))
        self.ledger.submit(lease)  # An exception during native enqueue may be partial.
        ticket = WriteTicket(lease, handle, None)
        self._tickets[lease.epoch] = ticket
        try:
            dependency = self.backend.write(
                key_bytes, value_bytes,
                handle.page_id * self.page_bytes + within_page_offset,
                byte_count,
                lease.epoch,
            )
        except Exception as exc:
            self.poisoned = True
            raise WriteSubmissionError(lease.epoch) from exc
        ticket = WriteTicket(lease, handle, dependency)
        self._tickets[lease.epoch] = ticket
        return ticket

    def submit_grouped_q1_write(
        self, handles: tuple[PageHandle, PageHandle],
        slots: tuple[int, int], *, keys: object, values: object,
        kv_heads: int, dim: int,
    ) -> WriteTicket:
        """One terminal lease pins both owners' exact destination generations."""
        if self.poisoned:
            raise RuntimeError("paged KV arena is poisoned")
        if (type(handles) is not tuple or len(handles) != 2 or
                type(handles[0]) is not PageHandle or
                type(handles[1]) is not PageHandle or handles[0] == handles[1] or
                type(slots) is not tuple or len(slots) != 2 or
                any(type(slot) is not int or not 0 <= slot < 64 for slot in slots) or
                type(kv_heads) is not int or type(dim) is not int or
                self.page_bytes != kv_heads * 64 * dim * 2 or
                not callable(getattr(self.backend, "grouped_q1_write", None))):
            raise ValueError("grouped Q1 write geometry or backend differs")
        lease = self.ledger.prepare(handles)
        self.ledger.submit(lease)
        ticket = WriteTicket(lease, handles[0], None, grouped_q1=True)
        self._tickets[lease.epoch] = ticket
        try:
            dependency = self.backend.grouped_q1_write(
                keys, values, (handles[0].page_id, handles[1].page_id),
                slots, kv_heads, dim, lease.epoch)
        except Exception as exc:
            self.poisoned = True
            raise WriteSubmissionError(lease.epoch) from exc
        ticket = WriteTicket(lease, handles[0], dependency, grouped_q1=True)
        self._tickets[lease.epoch] = ticket
        return ticket

    def submit_packed_multirow_write(self, handles: tuple[PageHandle, ...], *,
                                     keys: object, values: object,
                                     counts: tuple[int, int], starts: tuple[int, int],
                                     first_blocks: tuple[int, int], table_begins: tuple[int, int],
                                     page_ids: tuple[int, ...], kv_heads: int, dim: int) -> WriteTicket:
        """Pin every touched generation under one shared write terminal."""
        if self.poisoned:
            raise RuntimeError("paged KV arena is poisoned")
        if (os.environ.get("MLX2_PAGED_GROUPED_MULTIROW_WRITE", "0") != "1" or
                type(handles) is not tuple or not handles or
                any(type(handle) is not PageHandle for handle in handles) or
                len(set(handles)) != len(handles) or
                type(counts) is not tuple or len(counts) != 2 or
                any(type(n) is not int or n <= 0 or n > 8192 for n in counts) or
                type(starts) is not tuple or len(starts) != 2 or
                type(first_blocks) is not tuple or len(first_blocks) != 2 or
                type(table_begins) is not tuple or len(table_begins) != 2 or
                type(page_ids) is not tuple or not page_ids or
                type(kv_heads) is not int or type(dim) is not int or
                self.page_bytes != kv_heads * 64 * dim * 2 or
                not callable(getattr(self.backend, "grouped_multirow_write", None)) or
                not callable(getattr(getattr(self.backend, "_native", None), "grouped_multirow_write", None))):
            raise ValueError("packed multirow writer geometry or native capability differs")
        lease = self.ledger.prepare(handles)
        self.ledger.submit(lease)
        ticket = WriteTicket(lease, handles[0], None, packed_multirow=True)
        self._tickets[lease.epoch] = ticket
        try:
            dependency = self.backend.grouped_multirow_write(
                keys, values, counts, starts, first_blocks, table_begins,
                page_ids, kv_heads, dim, lease.epoch)
        except Exception as exc:
            self.poisoned = True
            raise WriteSubmissionError(lease.epoch) from exc
        ticket = WriteTicket(lease, handles[0], dependency, packed_multirow=True)
        self._tickets[lease.epoch] = ticket
        return ticket

    def submit_packed_multirow_write_n20(self, handles: tuple[PageHandle, ...], *,
                                         keys: object, values: object,
                                         counts: tuple[int, ...], starts: tuple[int, ...],
                                         first_blocks: tuple[int, ...], table_begins: tuple[int, ...],
                                         page_ids: tuple[int, ...], kv_heads: int, dim: int) -> WriteTicket:
        if (self.poisoned or os.environ.get("MLX2_PAGED_PACKED_N20", "0") != "1" or
                type(handles) is not tuple or not handles or len(set(handles)) != len(handles) or
                any(type(h) is not PageHandle for h in handles) or
                type(counts) is not tuple or not 1 <= len(counts) <= 20 or
                any(type(n) is not int or not 1 <= n <= 8192 for n in counts) or
                any(type(v) is not tuple or len(v) != len(counts)
                    for v in (starts, first_blocks, table_begins)) or
                type(page_ids) is not tuple or not page_ids or len(page_ids) > 2560 or
                self.page_bytes != kv_heads * 64 * dim * 2 or
                not callable(getattr(self.backend, "grouped_multirow_write_n20", None))):
            raise ValueError("N20 writer geometry or native capability differs")
        lease = self.ledger.prepare(handles)
        self.ledger.submit(lease)
        ticket = WriteTicket(lease, handles[0], None, packed_multirow=True)
        self._tickets[lease.epoch] = ticket
        try:
            dependency = self.backend.grouped_multirow_write_n20(
                keys, values, counts, starts, first_blocks, table_begins,
                page_ids, kv_heads, dim, lease.epoch)
        except Exception as exc:
            self.poisoned = True
            raise WriteSubmissionError(lease.epoch) from exc
        ticket = WriteTicket(lease, handles[0], dependency, packed_multirow=True)
        self._tickets[lease.epoch] = ticket
        return ticket

    def cancel(self, ticket: WriteTicket | CopyTicket) -> None:
        """Cancel a request without treating cancellation as GPU completion."""
        self._known(ticket)
        self.ledger.cancel(ticket.lease)

    def submit_copy(
        self, source: PageHandle, destination: PageHandle, *, byte_count: int,
    ) -> CopyTicket:
        """Copy an accepted shared partial page into a private reserved page.

        The caller must have completed prior writes to ``source`` and must
        publish ``destination`` only after successful terminal completion.
        Both page generations remain pinned even if cancellation or ambiguous
        enqueue occurs. Same-stream ordering is owned by the backend.
        """
        if self.poisoned:
            raise RuntimeError("paged KV arena is poisoned")
        _positive("copy byte count", byte_count, U64_MAX)
        if byte_count > self.page_bytes:
            raise ValueError("copy crosses a physical page")
        if source == destination:
            raise ValueError("copy requires distinct page generations")
        # No source/destination ownership is transferred by this method.
        lease = self.ledger.prepare((source, destination))
        self.ledger.submit(lease)
        ticket = CopyTicket(lease, source, destination, None)
        self._tickets[lease.epoch] = ticket
        try:
            dependency = self.backend.copy_page(
                source.page_id * self.page_bytes,
                destination.page_id * self.page_bytes,
                byte_count,
                lease.epoch,
            )
        except Exception as exc:
            self.poisoned = True
            raise WriteSubmissionError(lease.epoch) from exc
        ticket = CopyTicket(lease, source, destination, dependency)
        self._tickets[lease.epoch] = ticket
        return ticket

    def poll_completions(self, *, wait_timeout_s: float | None = None) -> tuple[WriteCompletion, ...]:
        """Consume terminal native events and release their host generation pins."""
        if wait_timeout_s is not None and (not math.isfinite(wait_timeout_s) or
                                           not 0 <= wait_timeout_s <= 120):
            raise ValueError("write completion wait must be finite and in [0, 120] seconds")
        events = (self.backend.poll_completions() if wait_timeout_s is None else
                  self.backend.wait_completions(wait_timeout_s))
        # Reject an invalid batch before any lease can be released. A native
        # callback is terminal evidence only for its exact outstanding epoch.
        seen: dict[int, bool] = {}
        for event in events:
            if (type(event) is not tuple or len(event) != 2 or
                    type(event[0]) is not int or type(event[1]) is not bool or
                    event[0] not in self._tickets and event[0] not in self._finished):
                self.poisoned = True
                raise RuntimeError("invalid or unknown native completion event")
            epoch, succeeded = event
            if ((epoch in seen and seen[epoch] != succeeded) or
                    (epoch in self._finished and self._finished[epoch] != succeeded)):
                self.poisoned = True
                raise RuntimeError("conflicting native terminal status")
            seen[epoch] = succeeded
        completed = []
        for event in events:
            epoch, succeeded = event
            if epoch in self._finished:
                continue
            ticket = self._tickets.pop(epoch)
            self.ledger.complete(ticket.lease, succeeded=succeeded)
            self._finished[epoch] = succeeded
            if not succeeded:
                self.poisoned = True
            completed.append(WriteCompletion(ticket, succeeded))
        return tuple(completed)

    def _known(self, ticket: WriteTicket | CopyTicket) -> None:
        if type(ticket) not in (WriteTicket, CopyTicket) or self._tickets.get(ticket.epoch) is not ticket:
            raise ValueError("unknown write ticket")
