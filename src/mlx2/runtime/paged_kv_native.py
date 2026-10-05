"""Host-side completion lease contract for a future native paged KV owner.

This module owns no KV bytes and submits no GPU commands. A native backend
must pin pages with ``prepare`` before encoding work, call ``submit`` once
encoding may have reached a command buffer, and call ``complete`` only from a
proof that all reads and writes for that submission have finished. Cancelling
a request is not such a proof. All calls touching the pool must be serialized
with other users of that pool by the eventual owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from weakref import WeakSet

from .paged_attention_plan import U64_MAX, PageHandle
from .paged_kv_pool import PagedKVPool


@dataclass(frozen=True, eq=False)
class CompletionLease:
    """Identity-bound host token; an epoch is not itself a completion proof."""

    epoch: int


@dataclass
class _Use:
    lease: CompletionLease
    handles: tuple[PageHandle, ...]
    state: str = "prepared"
    cancelled: bool = False


class PagedKVCompletionLedger:
    """Pins page generations across submission and contiguous completion.

    An aborted prepared use can retire without GPU evidence because it was
    never submitted. Once submitted, even cancellation retains every pin
    until ``complete``. Out-of-order completion releases pins individually,
    while the pool retirement watermark advances only through a contiguous
    prefix of completed epochs. This is a CPU contract, not a GPU fence.
    """

    def __init__(self, pool: PagedKVPool) -> None:
        if type(pool) is not PagedKVPool:
            raise TypeError("pool must be a PagedKVPool")
        if pool.completed_epoch != 0:
            raise ValueError("completion ledger requires a fresh pool")
        self.pool = pool
        self._next_epoch = 1
        self._completed_epoch = 0
        self._uses: dict[int, _Use] = {}
        self._completed_leases: WeakSet[CompletionLease] = WeakSet()

    @property
    def completed_epoch(self) -> int:
        return self._completed_epoch

    @property
    def pending_count(self) -> int:
        return sum(use.state != "completed" for use in self._uses.values())

    def prepare(self, handles: tuple[PageHandle, ...]) -> CompletionLease:
        """Pin a nonempty set of live pages before any native encoding."""
        if not handles:
            raise ValueError("a completion lease requires pages")
        if self._next_epoch > U64_MAX:
            raise OverflowError("completion epoch exhausted")
        self.pool.retain(handles)  # Validates all handles before mutating any.
        lease = CompletionLease(self._next_epoch)
        self._uses[lease.epoch] = _Use(lease, handles)
        self._next_epoch += 1
        return lease

    def submit(self, lease: CompletionLease) -> None:
        """Mark work potentially queued; ambiguous partial submission counts."""
        use = self._lookup(lease)
        if use.state != "prepared":
            raise ValueError("lease is not prepared")
        use.state = "submitted"

    def pinned_handles(self, lease: CompletionLease) -> tuple[PageHandle, ...]:
        """Return the exact generations retained by this live lease."""
        use = self._lookup(lease)
        if use.state == "completed":
            raise ValueError("completed lease no longer pins pages")
        return use.handles

    def abort_before_submit(self, lease: CompletionLease) -> None:
        """Release a prepared lease only when no native work was submitted."""
        use = self._lookup(lease)
        if use.state != "prepared":
            raise ValueError("submitted work requires completion proof")
        self._finish(use)

    def cancel(self, lease: CompletionLease) -> None:
        """Record cancellation; submitted pins stay live until completion."""
        if type(lease) is not CompletionLease:
            raise ValueError("unknown completion lease")
        if lease in self._completed_leases:
            return
        use = self._lookup(lease)
        if use.state == "completed":
            return
        use.cancelled = True

    def complete(self, lease: CompletionLease, *, succeeded: bool = True) -> None:
        """Accept a native completion proof, including failed command buffers.

        The caller must establish that the command buffer cannot access the
        arena again. This method cannot independently authenticate that proof.
        """
        if type(lease) is not CompletionLease:
            raise ValueError("unknown completion lease")
        if lease in self._completed_leases:
            return
        use = self._lookup(lease)
        if use.state == "completed":
            return
        if use.state != "submitted":
            raise ValueError("unsubmitted work has no native completion")
        if type(succeeded) is not bool:
            raise TypeError("terminal status must be a bool")
        if not succeeded:
            self.pool.quarantine(use.handles)
        self._finish(use)

    def _lookup(self, lease: CompletionLease) -> _Use:
        if type(lease) is not CompletionLease:
            raise ValueError("unknown completion lease")
        use = self._uses.get(lease.epoch)
        if use is None or use.lease is not lease:
            raise ValueError("unknown completion lease")
        return use

    def _finish(self, use: _Use) -> None:
        self.pool.release(use.handles, after_epoch=use.lease.epoch)
        use.state = "completed"
        self._completed_leases.add(use.lease)
        watermark = self._completed_epoch
        while (next_use := self._uses.get(watermark + 1)) is not None and next_use.state == "completed":
            watermark += 1
        if watermark != self._completed_epoch:
            self.pool.retire(watermark)
            previous = self._completed_epoch
            self._completed_epoch = watermark
            for epoch in range(previous + 1, watermark + 1):
                self._uses.pop(epoch, None)
