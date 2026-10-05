"""Default-off private paged KV leaf; host contract, not a serving route.

The byte payload here is an exact host staging copy for later export. Native
write/read ordering and the attention kernel still need an integration gate.
All methods touching the pool must be serialized by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from .paged_attention_plan import (
    PAGE_SIZE,
    U32_MAX,
    PagedAttentionPlan,
    SequenceSpan,
    _positive,
)
from .paged_kv_native import CompletionLease
from .paged_kv_pool import PagedKVSequence
from .paged_kv_write import NativeWriteBackend, PagedKVWriteOwner, WriteTicket


@dataclass(frozen=True)
class ExactKVExport:
    """Accepted logical tokens in token-major, KV-head-major byte order."""

    key_bytes: bytes
    value_bytes: bytes
    tokens: int
    kv_heads: int
    head_dim: int
    dtype: str


class FrozenPagedKV:
    """Immutable prefix bytes and a retained physical page-table owner.

    This object never enters a persistence manifest. Its page references are
    process-local and are released only by ``close``.
    """

    def __init__(self, source: PagedKVPrivateCache) -> None:
        source._ready()
        self.writer = source.writer
        self.sequence = source.sequence.fork()
        self.export = source.export_exact()
        self._closed = False

    def branch(self, *, permit_candidate: bool = False,
               staging_headroom_pages: int = 0) -> PagedKVPrivateCache:
        if not permit_candidate:
            raise RuntimeError("paged branch requires explicit candidate enablement")
        if self._closed:
            raise RuntimeError("frozen paged prefix is closed")
        if type(staging_headroom_pages) is not int or staging_headroom_pages < 0:
            raise ValueError("staging headroom must be a nonnegative page count")
        if self.writer.pool.free_count < staging_headroom_pages:
            raise MemoryError("paged branch lacks staging headroom")
        reserved = self.writer.pool.reserve(staging_headroom_pages) if staging_headroom_pages else ()
        try:
            child = PagedKVPrivateCache(
                self.writer, kv_heads=self.export.kv_heads,
                head_dim=self.export.head_dim, dtype=self.export.dtype,
                permit_candidate=True,
            )
            child.sequence = self.sequence.fork()
            child._keys = bytearray(self.export.key_bytes)
            child._values = bytearray(self.export.value_bytes)
            child._staging_reservation = reserved
            return child
        except Exception:
            if reserved:
                self.writer.pool.release(reserved, after_epoch=self.writer.ledger.completed_epoch)
                self.writer.pool.retire(self.writer.ledger.completed_epoch)
            raise

    def close(self) -> None:
        if self._closed:
            return
        self.sequence.abort(after_epoch=self.writer.ledger.completed_epoch)
        self.writer.pool.retire(self.writer.ledger.completed_epoch)
        self._closed = True


class AttentionUse:
    """A prepared page-generation pin; completion is an external proof."""

    def __init__(self, owner: PagedKVPrivateCache, plan: PagedAttentionPlan,
                 lease: CompletionLease) -> None:
        self.owner, self.plan, self.lease = owner, plan, lease
        self.state = "prepared"
        self.completed_successfully = False
        self.receipt_recorded = False

    def mark_submitted(self) -> None:
        if self.state != "prepared":
            raise ValueError("attention use is not prepared")
        self.owner.writer.ledger.submit(self.lease)
        self.state = "submitted"

    def abort_before_submit(self) -> None:
        if self.state != "prepared":
            raise ValueError("submitted attention needs terminal proof")
        self.owner.writer.ledger.abort_before_submit(self.lease)
        self.state = "closed"

    def complete_after_proof(self, *, succeeded: bool = True) -> None:
        """Call only after a terminal event covers every reader of this plan."""
        if self.state != "submitted":
            raise ValueError("attention use is not submitted")
        if type(succeeded) is not bool:
            raise TypeError("succeeded must be bool")
        self.owner.writer.ledger.complete(self.lease)
        self.completed_successfully = succeeded
        self.state = "closed"


class PagedKVPrivateCache:
    """One private dense KV sequence, with explicit append/plan/export APIs.

    The leaf deliberately has no ``update_and_fetch`` implementation. It is
    not interchangeable with an ordinary KVCache until an adapter explicitly
    consumes the paged plan and retains ``AttentionUse`` to GPU completion.
    """

    def __init__(self, writer: PagedKVWriteOwner, *, kv_heads: int,
                 head_dim: int, dtype: str, permit_candidate: bool = False) -> None:
        if not permit_candidate:
            raise RuntimeError("private paged KV requires explicit candidate enablement")
        if type(writer) is not PagedKVWriteOwner:
            raise TypeError("writer must be a PagedKVWriteOwner")
        _positive("kv_heads", kv_heads)
        if dtype not in ("float16", "bfloat16") or head_dim not in (128, 256):
            raise ValueError("unsupported private paged KV profile")
        self.token_bytes = kv_heads * head_dim * 2
        if writer.page_bytes != PAGE_SIZE * self.token_bytes:
            raise ValueError("writer page geometry does not match KV profile")
        self.writer = writer
        self.sequence = PagedKVSequence(writer.pool)
        self.kv_heads, self.head_dim, self.dtype = kv_heads, head_dim, dtype
        self._keys = bytearray()
        self._values = bytearray()
        self._pending: tuple[WriteTicket, ...] = ()
        self._staged: tuple[bytes, bytes] | None = None
        self._failed = False
        self._closed = False
        self._staging_reservation = ()

    @property
    def offset(self) -> int:
        """Accepted tokens only; an in-flight append is not published."""
        return len(self._keys) // self.token_bytes

    def _ready(self) -> None:
        if self._closed or self._failed or self.writer.poisoned:
            raise RuntimeError("private paged KV is closed or failed")
        if self._pending:
            raise RuntimeError("private paged KV append is pending")

    def append(self, key_bytes: bytes, value_bytes: bytes) -> tuple[WriteTicket, ...]:
        """Stage one token-major suffix; publish only after terminal success.

        This CPU byte contract is for fake-backend tests. Native MLX array
        conversion and stream ordering are separate integration work.
        """
        self._ready()
        if type(self.writer.backend) is NativeWriteBackend:
            raise ValueError("token-major private cache staging cannot write the native head-major arena")
        if type(key_bytes) is not bytes or type(value_bytes) is not bytes:
            raise TypeError("append requires exact immutable byte payloads")
        if not key_bytes or len(key_bytes) != len(value_bytes) or len(key_bytes) % self.token_bytes:
            raise ValueError("K/V payloads must contain equal whole tokens")
        count = len(key_bytes) // self.token_bytes
        old_end = self.sequence.kv_end
        if old_end + count > U32_MAX:
            raise ValueError("KV end exceeds uint32")
        shared_tail = bool(old_end % PAGE_SIZE and self.sequence.handles and
                           self.writer.pool.references(self.sequence.handles[-1]) > 1)
        if shared_tail:
            # Sequence.append would publish the replacement handle before a
            # native page copy completed. The arena is KV-head-major, so a
            # token-major byte count would also leave head data uncopied.
            raise ValueError("shared partial tail requires staged native COW")
        needed = (old_end + count - 1) // PAGE_SIZE + 1 - len(self.sequence.handles)
        if needed > self.writer.pool.free_count:
            raise MemoryError("paged KV arena has insufficient retired pages")
        # Every rejectable profile/capacity condition precedes metadata mutation.
        self.sequence.append(count, after_epoch=self.writer.ledger.completed_epoch)
        tickets: list[WriteTicket] = []
        try:
            cursor = 0
            while cursor < count:
                token = old_end + cursor
                take = min(count - cursor, PAGE_SIZE - token % PAGE_SIZE)
                byte_start, byte_end = cursor * self.token_bytes, (cursor + take) * self.token_bytes
                handle = self.sequence.handles[token // PAGE_SIZE - self.sequence.first_block]
                tickets.append(self.writer.submit_write(
                    handle, within_page_offset=token % PAGE_SIZE * self.token_bytes,
                    byte_count=byte_end - byte_start,
                    key_bytes=key_bytes[byte_start:byte_end],
                    value_bytes=value_bytes[byte_start:byte_end],
                ))
                cursor += take
        except Exception:
            # A submission may have reached the device. Keep every submitted
            # pin and refuse further use; close only after terminal events.
            self._failed = True
            self._pending = tuple(tickets)
            raise
        self._pending = tuple(tickets)
        self._staged = (key_bytes, value_bytes)
        return self._pending

    def poll_completions(self) -> bool:
        """Return true once a whole staged append has been accepted."""
        completions = self.writer.poll_completions()
        if any(not item.succeeded for item in completions):
            self._failed = True
        if not self._pending:
            return False
        pending_epochs = set(self.writer.pending_epochs)
        if any(ticket.epoch in pending_epochs for ticket in self._pending):
            return False
        self._pending = ()
        if self._failed:
            self._staged = None
            return False
        assert self._staged is not None
        key_bytes, value_bytes = self._staged
        self._keys.extend(key_bytes)
        self._values.extend(value_bytes)
        self._staged = None
        return True

    def attention(self, *, row_count: int, query_heads: int,
                  mask_kind: str = "causal", window: int | None = None) -> AttentionUse:
        """Build a validated suffix plan and pin generations before submission."""
        self._ready()
        _positive("row_count", row_count)
        if row_count > self.offset:
            raise ValueError("query rows exceed accepted KV")
        handles = self.sequence.handles
        lease = self.writer.ledger.prepare(handles)
        try:
            plan = PagedAttentionPlan(
                spans=(SequenceSpan(0, row_count, self.offset - row_count,
                                    self.offset, 0, 0, 0, len(handles), 1,
                                    mask_kind, window),),
                page_table=handles, total_rows=row_count,
                query_heads=query_heads, kv_heads=self.kv_heads,
                head_dim=self.head_dim, dtype=self.dtype,
                pool_capacity=self.writer.pool.capacity,
                live_generations=self.writer.pool.live_generations(),
            )
        except Exception:
            self.writer.ledger.abort_before_submit(lease)
            raise
        return AttentionUse(self, plan, lease)

    def export_exact(self) -> ExactKVExport:
        self._ready()
        return ExactKVExport(bytes(self._keys), bytes(self._values), self.offset,
                             self.kv_heads, self.head_dim, self.dtype)

    def freeze_exact(self) -> FrozenPagedKV:
        """Retain accepted prefix pages across independent branch lifetimes."""
        return FrozenPagedKV(self)

    def close(self) -> None:
        if self._closed:
            return
        after = max(self.writer.pending_epochs, default=self.writer.ledger.completed_epoch)
        self.sequence.abort(after_epoch=after)
        if self._staging_reservation:
            self.writer.pool.release(self._staging_reservation, after_epoch=after)
            self._staging_reservation = ()
        self.writer.pool.retire(self.writer.ledger.completed_epoch)
        self._closed = True
