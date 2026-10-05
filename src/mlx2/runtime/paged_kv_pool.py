"""CPU-only ownership model for a future, default-off paged KV arena.

This module manages *identities and leases*, not KV bytes.  A native owner
must prove completion of every use before advancing ``retire``.  Neither a
Python release nor a cancelled request implies GPU completion.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .paged_attention_plan import PAGE_SIZE, U32_MAX, PageHandle, _positive, _uint


@dataclass
class _Slot:
    generation: int = 0
    references: int = 0
    retired_after: int | None = None
    quarantined_generation: int | None = None


@dataclass(frozen=True, eq=False)
class AppendReservation:
    """Unpublished append slots; the sequence retains its original leases."""

    source: PageHandle | None
    destination: PageHandle | None
    handles: tuple[PageHandle, ...]
    reserved: tuple[PageHandle, ...]
    token_count: int
    old_end: int


class PagedKVPool:
    """Fixed-capacity host lease ledger; one instance represents one arena.

    ``reserve`` creates one owner reference per page. ``retain`` creates one
    more reference per handle. ``release`` drops one reference per handle and
    records the last possible use epoch. A zero-reference slot remains
    unavailable until ``retire(completed_epoch)`` proves that epoch complete.
    All methods are host-serial; callers must synchronize concurrent access.
    """

    def __init__(self, capacity: int) -> None:
        _positive("capacity", capacity)
        self.capacity = capacity
        self._slots = [_Slot() for _ in range(capacity)]
        self._free = list(reversed(range(capacity)))
        self._pending: deque[tuple[int, PageHandle]] = deque()
        self._completed_epoch = 0

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def quarantined_count(self) -> int:
        """Physical slots permanently unavailable until the arena is destroyed."""
        return sum(slot.quarantined_generation is not None for slot in self._slots)

    @property
    def allocated_count(self) -> int:
        return sum(slot.references > 0 for slot in self._slots)

    @property
    def completed_epoch(self) -> int:
        return self._completed_epoch

    def live_generations(self) -> dict[int, int]:
        """Generations of referenced pages eligible for a new host plan."""
        return {
            page_id: slot.generation
            for page_id, slot in enumerate(self._slots)
            if slot.references > 0 and slot.quarantined_generation is None
        }

    def references(self, handle: PageHandle) -> int:
        return self._active_slot(handle).references

    def reserve(self, count: int = 1) -> tuple[PageHandle, ...]:
        _positive("page count", count)
        if count > len(self._free):
            raise MemoryError("paged KV arena has insufficient retired pages")
        # Exhaustion fails atomically, before taking any reusable slot.
        if any(self._slots[page_id].generation == U32_MAX for page_id in self._free[-count:]):
            raise OverflowError("page generation exhausted")
        handles = []
        for _ in range(count):
            page_id = self._free.pop()
            slot = self._slots[page_id]
            slot.generation += 1
            slot.references = 1
            slot.retired_after = None
            if slot.quarantined_generation is not None:
                raise RuntimeError("quarantined page entered the free list")
            handles.append(PageHandle(page_id, slot.generation))
        return tuple(handles)

    def quarantine(self, handles: tuple[PageHandle, ...]) -> None:
        """Exclude failed native-use generations from every future arena plan.

        A terminal failure proves the command buffer stopped, but cannot prove
        its destination bytes or referenced pages are suitable for reuse. The
        slots remain unavailable for this arena's entire lifetime, including
        after the last host reference and completion-watermark advancement.
        """
        slots = self._validate_unique_active(handles)
        for slot in slots:
            slot.quarantined_generation = slot.generation

    def retain(self, handles: tuple[PageHandle, ...]) -> None:
        slots = self._validate_unique_active(handles)
        if any(slot.quarantined_generation == slot.generation for slot in slots):
            raise ValueError("quarantined page cannot gain a new lease")
        for slot in slots:
            slot.references += 1

    def release(self, handles: tuple[PageHandle, ...], *, after_epoch: int) -> None:
        _uint("last use epoch", after_epoch, (1 << 64) - 1)
        slots = self._validate_unique_active(handles)
        for handle, slot in zip(handles, slots):
            slot.references -= 1
            if slot.references == 0:
                slot.retired_after = max(after_epoch, self._completed_epoch)
                self._pending.append((slot.retired_after, handle))

    def copy_on_write(self, handle: PageHandle, *, after_epoch: int) -> PageHandle:
        """Transfer one owner lease to a fresh slot when a tail is shared.

        The caller must copy old KV bytes into the new slot before publishing
        the returned handle. This method does not copy data or submit work.
        """
        _uint("last use epoch", after_epoch, (1 << 64) - 1)
        if self.references(handle) == 1:
            return handle
        new_handle = self.reserve()[0]
        self.release((handle,), after_epoch=after_epoch)
        return new_handle

    def retire(self, completed_epoch: int) -> None:
        """Reclaim only zero-reference slots whose last use has completed."""
        _uint("completed epoch", completed_epoch, (1 << 64) - 1)
        if completed_epoch < self._completed_epoch:
            raise ValueError("completed epoch cannot move backward")
        self._completed_epoch = completed_epoch
        # Releases can arrive out of epoch order, so scan the bounded queue.
        pending = deque()
        while self._pending:
            required, handle = self._pending.popleft()
            if required > completed_epoch:
                pending.append((required, handle))
                continue
            slot = self._slots[handle.page_id]
            if slot.generation != handle.generation or slot.references != 0:
                raise RuntimeError("retirement ledger is inconsistent")
            slot.retired_after = None
            if slot.quarantined_generation != handle.generation:
                self._free.append(handle.page_id)
        self._pending = pending

    def _active_slot(self, handle: PageHandle) -> _Slot:
        if type(handle) is not PageHandle or handle.page_id >= self.capacity:
            raise ValueError("page handle is outside this pool")
        slot = self._slots[handle.page_id]
        if slot.generation != handle.generation or slot.references == 0:
            raise ValueError("stale or unreferenced page handle")
        return slot

    def _validate_unique_active(self, handles: tuple[PageHandle, ...]) -> list[_Slot]:
        if type(handles) is not tuple or len(handles) != len(set(handles)):
            raise ValueError("handles must be a tuple of unique page leases")
        return [self._active_slot(handle) for handle in handles]


class PagedKVSequence:
    """Private logical page table; branch and append remain metadata only.

    ``stage_append`` reserves a prospective table while preserving the old
    one; a byte-plane owner commits it only after successful terminal work.
    The original ``append`` is a metadata-only helper and cannot prove a
    shared-tail byte copy. ``trim`` may retain a partial first page, but
    consumers must respect ``retained_start``.
    """

    def __init__(self, pool: PagedKVPool) -> None:
        self.pool = pool
        self.retained_start = 0
        self.kv_end = 0
        self._first_block = 0
        self._pages: list[PageHandle] = []
        self._closed = False
        self._staged: AppendReservation | None = None

    def stage_append(self, token_count: int, *, reuse_tail_if_references: int = 1) -> AppendReservation:
        """Reserve a complete prospective table without publishing or releasing it.

        A shared partial tail needs a byte copy before the caller may commit.
        The caller must retain all submitted native uses before discarding.
        """
        self._check_open()
        if self._staged is not None:
            raise RuntimeError("sequence append reservation is pending")
        _positive("token count", token_count)
        if self.kv_end + token_count > U32_MAX:
            raise ValueError("KV end exceeds uint32 token coordinates")
        if type(reuse_tail_if_references) is not int or reuse_tail_if_references not in (1, 2):
            raise ValueError("invalid private tail reference permit")
        # Two references are admissible only under the request owner's reader
        # exclusion guard. A third reference (APCv2, another fork, or a pinned
        # reader) always restores ordinary COW.
        source = (self._pages[-1] if self._pages and self.kv_end % PAGE_SIZE and
                  self.pool.references(self._pages[-1]) > reuse_tail_if_references else None)
        needed_blocks = (self.kv_end + token_count - 1) // PAGE_SIZE - self._first_block + 1
        extra = needed_blocks - len(self._pages)
        reserved = self.pool.reserve(extra + int(source is not None)) if extra or source else ()
        destination = reserved[0] if source is not None else None
        handles = tuple(self._pages[:-1] + [destination] if destination else self._pages) + reserved[int(source is not None):]
        stage = AppendReservation(source, destination, handles, reserved, token_count, self.kv_end)
        self._staged = stage
        return stage

    def commit_append(self, stage: AppendReservation, *, after_epoch: int) -> None:
        """Publish only after the owner has observed terminal success."""
        self._check_stage(stage)
        _uint("last use epoch", after_epoch, (1 << 64) - 1)
        if stage.source is not None:
            self.pool.release((stage.source,), after_epoch=after_epoch)
        self._pages = list(stage.handles)
        self.kv_end = stage.old_end + stage.token_count
        self._staged = None

    def discard_append(self, stage: AppendReservation, *, after_epoch: int) -> None:
        """Keep accepted metadata; retire unpublished slots after terminal use."""
        self._check_stage(stage)
        _uint("last use epoch", after_epoch, (1 << 64) - 1)
        if stage.reserved:
            self.pool.release(stage.reserved, after_epoch=after_epoch)
        self._staged = None

    def _check_stage(self, stage: AppendReservation) -> None:
        self._check_open()
        if type(stage) is not AppendReservation or self._staged is not stage:
            raise ValueError("unknown append reservation")

    @property
    def handles(self) -> tuple[PageHandle, ...]:
        self._check_open()
        return tuple(self._pages)

    @property
    def first_block(self) -> int:
        self._check_open()
        return self._first_block

    def fork(self) -> PagedKVSequence:
        self._check_open()
        if self._staged is not None:
            raise RuntimeError("sequence append reservation is pending")
        self.pool.retain(tuple(self._pages))
        child = PagedKVSequence(self.pool)
        child.retained_start = self.retained_start
        child.kv_end = self.kv_end
        child._first_block = self._first_block
        child._pages = self._pages.copy()
        return child

    def append(self, token_count: int, *, after_epoch: int) -> tuple[tuple[PageHandle, PageHandle], ...]:
        """Reserve pages and return (source, destination) COW copy pairs.

        The owner must copy before writing the new suffix. If capacity fails,
        metadata and leases are unchanged. No KV bytes are written here.
        """
        self._check_open()
        if self._staged is not None:
            raise RuntimeError("sequence append reservation is pending")
        _positive("token count", token_count)
        _uint("last use epoch", after_epoch, (1 << 64) - 1)
        if self.kv_end + token_count > U32_MAX:
            raise ValueError("KV end exceeds uint32 token coordinates")
        new_end = self.kv_end + token_count
        tail_is_shared = bool(self._pages and self.kv_end % PAGE_SIZE and
                              self.pool.references(self._pages[-1]) > 1)
        needed_blocks = (new_end - 1) // PAGE_SIZE - self._first_block + 1
        new_count = needed_blocks - len(self._pages)
        # Reserve all new slots before changing the table or old reference.
        reserved = self.pool.reserve(new_count + int(tail_is_shared)) if new_count or tail_is_shared else ()
        copied: tuple[tuple[PageHandle, PageHandle], ...] = ()
        if tail_is_shared:
            old_tail = self._pages[-1]
            self._pages[-1] = reserved[0]
            self.pool.release((old_tail,), after_epoch=after_epoch)
            copied = ((old_tail, reserved[0]),)
            reserved = reserved[1:]
        self._pages.extend(reserved)
        self.kv_end = new_end
        return copied

    def trim(self, retained_start: int, *, after_epoch: int) -> None:
        self._check_open()
        if self._staged is not None:
            raise RuntimeError("sequence append reservation is pending")
        _uint("retained start", retained_start)
        _uint("last use epoch", after_epoch, (1 << 64) - 1)
        if not self.retained_start <= retained_start <= self.kv_end:
            raise ValueError("retained start must move forward within KV")
        new_first_block = retained_start // PAGE_SIZE
        drop = len(self._pages) if retained_start == self.kv_end else new_first_block - self._first_block
        if drop:
            self.pool.release(tuple(self._pages[:drop]), after_epoch=after_epoch)
            del self._pages[:drop]
        self.retained_start = retained_start
        self._first_block = new_first_block

    def truncate(self, new_end: int, *, after_epoch: int) -> None:
        """Drop a completed rejected suffix without exposing its former tokens.

        Native callers must first prove every read and write that could touch
        the dropped pages terminal. A retained partial tail remains allocated;
        later appends overwrite its rejected bytes or COW it if shared.
        """
        self._check_open()
        if self._staged is not None:
            raise RuntimeError("sequence append reservation is pending")
        _uint("new KV end", new_end)
        _uint("last use epoch", after_epoch, (1 << 64) - 1)
        if not self.retained_start <= new_end <= self.kv_end:
            raise ValueError("new KV end must be within retained KV")
        keep = ((new_end - 1) // PAGE_SIZE - self._first_block + 1
                if new_end > self.retained_start else 0)
        if keep < len(self._pages):
            self.pool.release(tuple(self._pages[keep:]), after_epoch=after_epoch)
            del self._pages[keep:]
        self.kv_end = new_end
        if not keep:
            self._first_block = new_end // PAGE_SIZE

    def abort(self, *, after_epoch: int) -> None:
        self._check_open()
        if self._staged is not None:
            raise RuntimeError("sequence append reservation is pending")
        self.pool.release(tuple(self._pages), after_epoch=after_epoch)
        self._pages.clear()
        self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("sequence owner is closed")
