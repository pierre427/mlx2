"""Default-off token-profile owner for native byte-plane KV writes.

Each plane stores [page, kv_head, token, channel] 16-bit values. A
physical page contains exactly 64 tokens per head. This host owner never exposes a
mutable arena view or publishes a token before every write has completed.
The caller serializes all pool operations and supplies an explicit GPU stream
through ``NativeWriteBackend``. No serving adapter uses this owner yet.
"""

from __future__ import annotations

from dataclasses import dataclass

from .paged_attention_plan import PAGE_SIZE, U32_MAX, PageHandle, _positive
from .paged_kv_pool import AppendReservation, PagedKVSequence
from .paged_kv_write import CopyTicket, PagedKVWriteOwner, WriteTicket


@dataclass(frozen=True)
class TokenKVProfile:
    kv_heads: int
    head_dim: int
    dtype: str

    def __post_init__(self) -> None:
        _positive("kv_heads", self.kv_heads)
        if self.head_dim not in (128, 256) or self.dtype not in ("float16", "bfloat16"):
            raise ValueError("unsupported token KV profile")

    @property
    def token_bytes(self) -> int:
        return self.kv_heads * self.head_dim * 2

    @property
    def head_token_bytes(self) -> int:
        return self.head_dim * 2

    @property
    def page_bytes(self) -> int:
        return PAGE_SIZE * self.token_bytes


@dataclass(frozen=True)
class TokenWriteSpan:
    """One contiguous head-local region in the immutable reader's layout."""

    source_token_offset: int
    token_count: int
    kv_head: int
    block_index: int
    within_page_offset: int
    byte_count: int


class PagedKVTokenOwner:
    """Own a private token page table and completion-gated native appends.

    Inputs are one-dimensional contiguous MLX uint8 arrays, one K/V pair for
    each ``planned_spans`` entry, already in [token, channel] order for that
    entry's KV head. The adapter must pack them without changing its own
    KV-head/channel semantics. Every writer destination matches the immutable
    paged reader's [page, kv_head, token, channel] address formula. Shared
    partial tails use an unpublished reservation until copy and writes finish.
    """

    def __init__(self, writer: PagedKVWriteOwner, profile: TokenKVProfile,
                 *, permit_candidate: bool = False) -> None:
        if not permit_candidate:
            raise RuntimeError("token-profile paged KV requires explicit candidate enablement")
        if type(writer) is not PagedKVWriteOwner or type(profile) is not TokenKVProfile:
            raise TypeError("writer and profile must be exact paged KV types")
        if writer.page_bytes != profile.page_bytes:
            raise ValueError("writer page geometry does not match token profile")
        self.writer = writer
        self.profile = profile
        self.sequence = PagedKVSequence(writer.pool)
        self._accepted_tokens = 0
        self._pending: tuple[WriteTicket | CopyTicket, ...] = ()
        self._sources: tuple[tuple[object, ...], tuple[object, ...]] | None = None
        self._stage: AppendReservation | None = None
        self._graph_stage = False
        self._spans: tuple[TokenWriteSpan, ...] = ()
        self._cancelled = False
        self._failed = False
        self._closed = False
        # Set only by NativeAtomicRequestOwner while its public reader gate is held.
        self._reuse_shared_tail_refs = 1

    @property
    def offset(self) -> int:
        return self._accepted_tokens

    def _ready(self) -> None:
        if self._closed or self._failed or self.writer.poisoned:
            raise RuntimeError("token-profile paged KV is closed or failed")
        if self._pending:
            raise RuntimeError("token-profile paged KV append is pending")

    def planned_spans(self, token_count: int, *, staged: bool = False) -> tuple[TokenWriteSpan, ...]:
        """Describe exact head-local source chunks before allocating pages."""
        self._ready()
        _positive("token count", token_count)
        count = token_count
        old_end = self.sequence.kv_end
        if old_end + count > U32_MAX:
            raise ValueError("KV end exceeds uint32")
        shared = bool(self.sequence.handles and old_end % PAGE_SIZE and
                      self.writer.pool.references(self.sequence.handles[-1]) >
                      (self._reuse_shared_tail_refs if staged else 1))
        if shared and not callable(getattr(self.writer.backend, "copy_page", None)):
            raise ValueError("shared tail requires a native COW gate")
        needed = (old_end + count - 1) // PAGE_SIZE + 1 - len(self.sequence.handles) + int(shared)
        if needed > self.writer.pool.free_count:
            raise MemoryError("paged KV arena has insufficient retired pages")
        spans = []
        cursor = 0
        while cursor < count:
            logical = old_end + cursor
            take = min(count - cursor, PAGE_SIZE - logical % PAGE_SIZE)
            block = logical // PAGE_SIZE - self.sequence.first_block
            for head in range(self.profile.kv_heads):
                spans.append(TokenWriteSpan(
                    cursor, take, head, block,
                    (head * PAGE_SIZE + logical % PAGE_SIZE) * self.profile.head_token_bytes,
                    take * self.profile.head_token_bytes,
                ))
            cursor += take
        return tuple(spans)

    def append(self, key_chunks: tuple[object, ...], value_chunks: tuple[object, ...],
               *, token_count: int) -> tuple[WriteTicket | CopyTicket, ...]:
        """Submit head-local spans; return tickets, not accepted KV state."""
        spans = self.planned_spans(token_count)
        if type(key_chunks) is not tuple or type(value_chunks) is not tuple or (
                len(key_chunks) != len(spans) or len(value_chunks) != len(spans)):
            raise ValueError("one K/V chunk pair is required for every planned span")
        validator = getattr(self.writer.backend, "validate_sources", None)
        if validator is None:
            raise ValueError("native source validation is required")
        for span, key, value in zip(spans, key_chunks, value_chunks):
            if (not validator(key, value) or key.nbytes != span.byte_count or
                    value.nbytes != span.byte_count):
                raise ValueError("K/V chunk must be contiguous 1D MLX uint8 with exact span bytes")
        shared = bool(self.sequence.handles and self.sequence.kv_end % PAGE_SIZE and
                      self.writer.pool.references(self.sequence.handles[-1]) > 1)
        if shared:
            stage = self.sequence.stage_append(token_count)
            self._stage = stage
            self._spans = spans
            self._sources = (key_chunks, value_chunks)
            try:
                copy = self.writer.submit_copy(
                    stage.source, stage.destination, byte_count=self.profile.page_bytes,
                )
            except Exception:
                # An ambiguous enqueue still owns its ledger pins until a
                # terminal callback. The original sequence table is intact.
                self.sequence.discard_append(stage, after_epoch=max(
                    self.writer.pending_epochs, default=self.writer.ledger.completed_epoch))
                self._stage = None
                self._failed = True
                raise
            self._pending = (copy,)
            return self._pending
        self.sequence.append(token_count, after_epoch=self.writer.ledger.completed_epoch)
        tickets: list[WriteTicket] = []
        try:
            for span, key, value in zip(spans, key_chunks, value_chunks):
                handle = self.sequence.handles[span.block_index]
                tickets.append(self.writer.submit_write(
                    handle,
                    within_page_offset=span.within_page_offset,
                    byte_count=span.byte_count,
                    key_bytes=key, value_bytes=value,
                ))
        except Exception:
            # Enqueue may have reached a command buffer. Preserve leases and
            # source arrays until all terminal callbacks have been consumed.
            self._failed = True
            self._pending = tuple(tickets)
            self._sources = (key_chunks, value_chunks)
            raise
        self._pending = tuple(tickets)
        self._sources = (key_chunks, value_chunks)
        return self._pending

    def append_staged(self, key_chunks: tuple[object, ...], value_chunks: tuple[object, ...],
                      *, token_count: int) -> tuple[WriteTicket | CopyTicket, ...]:
        """Enqueue an unpublished suffix for a private dependency graph.

        The returned tickets are dependencies, not success proofs. The caller
        must retain this owner and drain every terminal event before publishing
        or sampling. A shared-tail copy precedes its writes on the same native
        stream; no accepted table changes until all epochs succeed.
        """
        spans = self.planned_spans(token_count, staged=True)
        if (type(key_chunks) is not tuple or type(value_chunks) is not tuple or
                len(key_chunks) != len(spans) or len(value_chunks) != len(spans)):
            raise ValueError("one K/V chunk pair is required for every planned span")
        validator = getattr(self.writer.backend, "validate_sources", None)
        if validator is None or any(
                not validator(key, value) or key.nbytes != span.byte_count or
                value.nbytes != span.byte_count
                for span, key, value in zip(spans, key_chunks, value_chunks)):
            raise ValueError("K/V chunk must be contiguous 1D MLX uint8 with exact span bytes")
        stage = self.sequence.stage_append(
            token_count, reuse_tail_if_references=self._reuse_shared_tail_refs)
        self._stage = stage
        self._graph_stage = True
        self._sources = (key_chunks, value_chunks)
        tickets: list[WriteTicket | CopyTicket] = []
        try:
            if stage.source is not None:
                copy = self.writer.submit_copy(
                    stage.source, stage.destination, byte_count=self.profile.page_bytes)
                tickets.append(copy)
                depend_source = getattr(self.writer.backend, "depend_source", None)
                if copy.dependency is None or not callable(depend_source):
                    raise RuntimeError("staged COW requires an explicit copy dependency")
                key_chunks = tuple(depend_source(key, copy.dependency) for key in key_chunks)
                value_chunks = tuple(depend_source(value, copy.dependency) for value in value_chunks)
                self._sources = (key_chunks, value_chunks)
                if any(not validator(key, value) for key, value in
                       zip(key_chunks, value_chunks)):
                    raise RuntimeError("staged COW dependency changed source layout")
            for span, key, value in zip(spans, key_chunks, value_chunks):
                tickets.append(self.writer.submit_write(
                    stage.handles[span.block_index],
                    within_page_offset=span.within_page_offset,
                    byte_count=span.byte_count, key_bytes=key, value_bytes=value))
        except BaseException:
            # The last enqueue may be ambiguous. Retain the reservation and
            # sources; rollback quarantines this owner until arena teardown.
            self._pending = tuple(tickets)
            self._failed = True
            raise
        self._pending = tuple(tickets)
        return self._pending

    def stage_grouped_q1(self, keys: object, values: object) -> tuple[PageHandle, int]:
        """Reserve one private token; a later shared ticket owns publication."""
        self.planned_spans(1, staged=True)
        old_end = self.sequence.kv_end
        if (self.sequence.handles and old_end % PAGE_SIZE and
                self.writer.pool.references(self.sequence.handles[-1]) >
                self._reuse_shared_tail_refs):
            raise ValueError("grouped Q1 does not admit a shared tail")
        stage = self.sequence.stage_append(
            1, reuse_tail_if_references=self._reuse_shared_tail_refs)
        if stage.source is not None:
            self.sequence.discard_append(stage, after_epoch=self.writer.ledger.completed_epoch)
            raise RuntimeError("grouped Q1 unexpectedly reserved a COW source")
        self._stage = stage
        self._graph_stage = True
        self._sources = ((keys,), (values,))
        return stage.handles[old_end // PAGE_SIZE], old_end % PAGE_SIZE

    def stage_packed_multirow(self, keys: object, values: object, token_count: int,
                              ) -> tuple[tuple[PageHandle, ...], tuple[PageHandle, ...], int, int]:
        """Reserve an unpublished contiguous source segment for one packed lane.

        Shared-tail COW is deliberately refused before reservation/submission;
        this one-command primitive has no ordered page-copy dependency.
        """
        self.planned_spans(token_count, staged=True)
        old_end = self.sequence.kv_end
        if (self.sequence.handles and old_end % PAGE_SIZE and
                self.writer.pool.references(self.sequence.handles[-1]) >
                self._reuse_shared_tail_refs):
            raise ValueError("packed multirow write does not admit shared-tail COW")
        stage = self.sequence.stage_append(
            token_count, reuse_tail_if_references=self._reuse_shared_tail_refs)
        if stage.source is not None:
            self.sequence.discard_append(stage, after_epoch=self.writer.ledger.completed_epoch)
            raise RuntimeError("packed multirow unexpectedly reserved a COW source")
        self._stage = stage
        self._graph_stage = True
        self._sources = ((keys,), (values,))
        try:
            touched = tuple(dict.fromkeys(stage.handles[
                logical // PAGE_SIZE - self.sequence.first_block]
                for logical in range(old_end, old_end + token_count)))
            return touched, stage.handles, old_end, self.sequence.first_block
        except BaseException:
            self.abort_prepared_packed_multirow()
            raise

    def _check_packed_multirow_ticket(self, ticket: WriteTicket, *, submitted: bool) -> None:
        stage = self._stage
        if (type(ticket) is not WriteTicket or not ticket.packed_multirow or ticket.grouped_q1 or
                (ticket.dependency is None) == submitted or stage is None or
                not self._graph_stage or self._closed or
                self.writer._tickets.get(ticket.epoch) is not ticket or
                self._pending not in ((), (ticket,))):
            raise RuntimeError("packed multirow ticket does not match private stage")
        pinned = set(self.writer.ledger.pinned_handles(ticket.lease))
        touched = {stage.handles[logical // PAGE_SIZE - self.sequence.first_block]
                   for logical in range(stage.old_end, stage.old_end + stage.token_count)}
        if not touched or not touched <= pinned:
            raise RuntimeError("packed multirow lease misses a destination generation")

    def adopt_packed_multirow(self, ticket: WriteTicket) -> None:
        self._check_packed_multirow_ticket(ticket, submitted=True)
        if self._failed or self._pending:
            raise RuntimeError("packed multirow stage cannot be adopted twice")
        self._pending = (ticket,)

    def retain_ambiguous_packed_multirow(self, ticket: WriteTicket) -> None:
        self._check_packed_multirow_ticket(ticket, submitted=False)
        self._pending = (ticket,)
        self._failed = True

    def fail_submitted_packed_multirow(self, ticket: WriteTicket) -> None:
        self._check_packed_multirow_ticket(ticket, submitted=True)
        self._pending = (ticket,)
        self._failed = True

    def abort_prepared_packed_multirow(self) -> None:
        if self._pending:
            raise RuntimeError("submitted packed multirow stage needs terminal proof")
        self._discard_stage()
        self._sources = None

    def adopt_grouped_q1(self, ticket: WriteTicket) -> None:
        if (type(ticket) is not WriteTicket or not ticket.grouped_q1 or
                ticket.dependency is None or
                self._stage is None or not self._graph_stage or self._pending or
                self._failed or self._closed or
                self.writer._tickets.get(ticket.epoch) is not ticket or
                self._stage.handles[self._stage.old_end // PAGE_SIZE] not in
                self.writer.ledger.pinned_handles(ticket.lease)):
            raise RuntimeError("grouped Q1 ticket does not match a private stage")
        self._pending = (ticket,)

    def retain_ambiguous_grouped_q1(self, ticket: WriteTicket) -> None:
        if (type(ticket) is not WriteTicket or not ticket.grouped_q1 or
                ticket.dependency is not None or
                self._stage is None or not self._graph_stage or self._pending or
                self.writer._tickets.get(ticket.epoch) is not ticket or
                self._stage.handles[self._stage.old_end // PAGE_SIZE] not in
                self.writer.ledger.pinned_handles(ticket.lease)):
            raise RuntimeError("ambiguous grouped Q1 ticket does not match a stage")
        self._pending = (ticket,)
        self._failed = True

    def fail_submitted_grouped_q1(self, ticket: WriteTicket) -> None:
        """Keep a submitted shared lease pinned after partial adoption."""
        if (type(ticket) is not WriteTicket or not ticket.grouped_q1 or
                ticket.dependency is None or
                self._stage is None or not self._graph_stage or
                self._pending not in ((), (ticket,)) or
                self.writer._tickets.get(ticket.epoch) is not ticket or
                self._stage.handles[self._stage.old_end // PAGE_SIZE] not in
                self.writer.ledger.pinned_handles(ticket.lease)):
            raise RuntimeError("submitted grouped Q1 ticket does not match a stage")
        self._pending = (ticket,)
        self._failed = True

    def abort_prepared_grouped_q1(self) -> None:
        """Release a reservation only before any native ticket was submitted."""
        if self._pending:
            raise RuntimeError("submitted grouped Q1 stage needs terminal proof")
        self._discard_stage()
        self._sources = None

    def staged_handles(self, token_count: int) -> tuple[PageHandle, ...]:
        """Expose only this private proposal to a pinned staged-read builder."""
        stage = self._stage
        if (not self._graph_stage or stage is None or self._failed or self._closed or
                stage.old_end != self._accepted_tokens or stage.token_count != token_count or
                not self._pending):
            raise RuntimeError("no valid private staged KV proposal")
        return stage.handles

    def cancel_pending(self) -> None:
        """Cancel a staged COW without accepting a copy or releasing its pins."""
        if self._stage is None:
            raise RuntimeError("no staged COW append")
        self._cancelled = True
        self._failed = True
        for ticket in self._pending:
            if ticket.epoch in self.writer.pending_epochs:
                self.writer.cancel(ticket)

    def _discard_stage(self) -> None:
        stage = self._stage
        if stage is not None:
            after = max((ticket.epoch for ticket in self._pending),
                        default=self.writer.ledger.completed_epoch)
            self.sequence.discard_append(stage, after_epoch=after)
            self.writer.pool.retire(self.writer.ledger.completed_epoch)
            self._stage = None
            self._graph_stage = False

    def _submit_staged_writes(self) -> None:
        stage = self._stage
        assert stage is not None and self._sources is not None
        tickets: list[WriteTicket] = []
        try:
            for span, key, value in zip(self._spans, *self._sources):
                tickets.append(self.writer.submit_write(
                    stage.handles[span.block_index],
                    within_page_offset=span.within_page_offset,
                    byte_count=span.byte_count,
                    key_bytes=key, value_bytes=value,
                ))
        except Exception:
            self._pending = tuple(tickets)
            self._failed = True
            self._discard_stage()
            raise
        self._pending = tuple(tickets)

    def poll_completions(self, *, wait_timeout_s: float | None = None) -> bool:
        """Publish the entire suffix only after every terminal success."""
        events = self.writer.poll_completions(wait_timeout_s=wait_timeout_s)
        if self.writer.poisoned or any(not event.succeeded for event in events):
            self._failed = True
        if self._stage is not None:
            if self._failed or self._cancelled or self._closed:
                self._discard_stage()
            elif not self._graph_stage and self._pending and type(self._pending[0]) is CopyTicket and (
                    self._pending[0].epoch not in self.writer.pending_epochs):
                self._pending = ()
                self._submit_staged_writes()
                return False
        if not self._pending:
            if not self.writer.pending_epochs:
                self._sources = None
            return False
        if any(ticket.epoch in self.writer.pending_epochs for ticket in self._pending):
            return False
        tickets, self._pending = self._pending, ()
        if not self.writer.pending_epochs:
            self._sources = None
        if self._failed or self._closed:
            return False
        if self._stage is not None:
            self.sequence.commit_append(self._stage, after_epoch=max(
                ticket.epoch for ticket in tickets))
            self.writer.pool.retire(self.writer.ledger.completed_epoch)
            self._stage = None
            self._graph_stage = False
        self._accepted_tokens = self.sequence.kv_end
        return bool(tickets)

    def accepted_handles(self) -> tuple[PageHandle, ...]:
        """Return only a completed table; readers still need their own lease."""
        self._ready()
        return self.sequence.handles

    def truncate_accepted(self, new_end: int) -> None:
        """Keep an exact completed prefix after all arena uses are terminal."""
        self._ready()
        if self.writer.pending_epochs or self.writer.ledger.pending_count:
            raise RuntimeError("native arena use is not terminal")
        if self.sequence.kv_end != self._accepted_tokens:
            raise RuntimeError("token-profile sequence is not completed")
        self.sequence.truncate(new_end, after_epoch=self.writer.ledger.completed_epoch)
        self.writer.pool.retire(self.writer.ledger.completed_epoch)
        self._accepted_tokens = new_end

    def close(self) -> None:
        if self._closed:
            return
        if self._stage is not None:
            self._cancelled = True
            for ticket in self._pending:
                if ticket.epoch in self.writer.pending_epochs:
                    self.writer.cancel(ticket)
            self._discard_stage()
        after = max(self.writer.pending_epochs, default=self.writer.ledger.completed_epoch)
        self.sequence.abort(after_epoch=after)
        self.writer.pool.retire(self.writer.ledger.completed_epoch)
        self._closed = True
