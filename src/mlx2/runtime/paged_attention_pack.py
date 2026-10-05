"""Default-off packed read plans for completed private token-page owners.

This host-only seam snapshots several sequences against one arena and pins
their page generations. It does not expose native arena bytes or submit a GPU
read. All calls touching an owner and its pool must be serialized by the caller.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter_ns

from .paged_attention_metal import PagedReadMetadata, build_paged_read_metadata
from .paged_attention_plan import PagedAttentionPlan, SequenceSpan, _positive
from .paged_kv_native import CompletionLease
from .paged_kv_token import PagedKVTokenOwner
from .paged_kv_write import PagedKVWriteOwner


@dataclass
class PackedTokenRead:
    """A pinned plan and immutable Metal metadata; completion is external."""

    writer: PagedKVWriteOwner
    plan: PagedAttentionPlan
    metadata: PagedReadMetadata
    lease: CompletionLease
    state: str = "prepared"
    terminal_succeeded: bool | None = None
    metadata_build_ns: int | None = None

    def mark_submitted(self) -> None:
        if self.state != "prepared":
            raise ValueError("packed read is not prepared")
        self.writer.ledger.submit(self.lease)
        self.state = "submitted"

    def abort_before_submit(self) -> None:
        if self.state != "prepared":
            raise ValueError("submitted packed read requires terminal proof")
        self.writer.ledger.abort_before_submit(self.lease)
        self.state = "closed"

    def complete_after_proof(self, *, succeeded: bool = True) -> None:
        """Caller must prove the native read cannot access the arena again."""
        if self.state != "submitted":
            raise ValueError("packed read is not submitted")
        self.writer.ledger.complete(self.lease, succeeded=succeeded)
        self.terminal_succeeded = succeeded
        if not succeeded:
            self.writer.poisoned = True
        self.state = "closed"


def prepare_packed_token_read(
    owners: tuple[PagedKVTokenOwner, ...],
    row_counts: tuple[int, ...],
    *,
    query_heads: int,
    masks: tuple[tuple[str, int | None], ...] | None = None,
    permit_candidate: bool = False,
    profile_host: bool = False,
) -> PackedTokenRead:
    """Pin one dense fp16/bf16 GQA plan for completed lanes in a shared arena.

    Each query range is the accepted KV suffix of its lane. Partial retained
    first pages and sliding windows are preserved. Unknown codecs, sinks, QSA,
    MTP branches and non-native sources have no admission through this seam.
    """
    if not permit_candidate:
        raise RuntimeError("packed paged attention requires explicit candidate enablement")
    started = perf_counter_ns() if profile_host else 0
    if (type(owners) is not tuple or type(row_counts) is not tuple or not owners or
            len(owners) != len(row_counts) or len({id(owner) for owner in owners}) != len(owners)):
        raise ValueError("one unique token-page owner and row count are required per lane")
    if masks is None:
        masks = (("causal", None),) * len(owners)
    if type(masks) is not tuple or len(masks) != len(owners):
        raise ValueError("one mask is required per lane")
    _positive("query_heads", query_heads)
    first = owners[0]
    if type(first) is not PagedKVTokenOwner:
        raise TypeError("packed lanes require exact token-page owners")
    writer, profile = first.writer, first.profile
    spans = []
    table = []
    row_begin = 0
    for owner, rows, mask in zip(owners, row_counts, masks):
        if type(owner) is not PagedKVTokenOwner or owner.writer is not writer or owner.profile != profile:
            raise ValueError("packed lanes must share the native arena and KV geometry")
        _positive("row count", rows)
        if rows > owner.offset:
            raise ValueError("query rows exceed completed KV")
        if type(mask) is not tuple or len(mask) != 2:
            raise ValueError("mask must be a (kind, window) pair")
        handles = owner.accepted_handles()  # Refuses pending, failed or closed owners.
        sequence = owner.sequence
        if sequence.kv_end != owner.offset or not handles:
            raise ValueError("sequence metadata is not a completed KV snapshot")
        spans.append(SequenceSpan(
            row_begin, rows, owner.offset - rows, owner.offset,
            sequence.retained_start, sequence.first_block, len(table), len(handles),
            max(handle.generation for handle in handles), mask[0], mask[1],
        ))
        table.extend(handles)
        row_begin += rows
    # Shared prefix pages are pinned once per use, even if several lanes map
    # them. Preserve the logical table duplicates for attention addressing.
    unique_handles = tuple(dict.fromkeys(table))
    lease = writer.ledger.prepare(unique_handles)
    try:
        # prepare validated and pinned every generation. Snapshot only those
        # handles rather than scanning every slot in a potentially large arena.
        pinned_generations = {handle.page_id: handle.generation
                              for handle in unique_handles}
        plan = PagedAttentionPlan(
            spans=tuple(spans), page_table=tuple(table), total_rows=row_begin,
            query_heads=query_heads, kv_heads=profile.kv_heads,
            head_dim=profile.head_dim, dtype=profile.dtype,
            pool_capacity=writer.pool.capacity,
            live_generations=pinned_generations,
        )
        metadata = build_paged_read_metadata(plan)
    except Exception:
        writer.ledger.abort_before_submit(lease)
        raise
    return PackedTokenRead(writer, plan, metadata, lease,
                           metadata_build_ns=perf_counter_ns() - started if profile_host else None)


def prepare_staged_token_read(
    owners: tuple[PagedKVTokenOwner, ...], row_counts: tuple[int, ...],
    *, query_heads: int, permit_candidate: bool = False,
    profile_host: bool = False, long_fused: bool = False,
    n20: bool = False,
) -> PackedTokenRead:
    """Pin an unpublished, exact private suffix after all writes are enqueued.

    The returned plan cannot become public merely by being submitted. Its
    caller must prove every write and read terminal before atomic publication.
    """
    if not permit_candidate:
        raise RuntimeError("staged paged attention requires explicit candidate enablement")
    started = perf_counter_ns() if profile_host else 0
    if type(long_fused) is not bool or type(n20) is not bool:
        raise ValueError('long staged read selectors must be boolean')
    if n20 and (not long_fused or os.environ.get('MLX2_PAGED_PACKED_N20', '0') != '1' or
                type(owners) is not tuple or not 1 <= len(owners) <= 20 or
                type(row_counts) is not tuple or len(row_counts) != len(owners) or
                not (all(type(n) is int and 256 <= n <= 8192 for n in row_counts) or
                     all(type(n) is int and n == 1 for n in row_counts))):
        raise ValueError('N20 staged read requires exact cold or Q1 geometry')
    if (type(owners) is not tuple or type(row_counts) is not tuple or not owners or
            len(owners) != len(row_counts) or
            len({id(owner) for owner in owners}) != len(owners)):
        raise ValueError("one distinct staged owner and row count are required per lane")
    _positive("query_heads", query_heads)
    first = owners[0]
    if type(first) is not PagedKVTokenOwner:
        raise TypeError("staged lanes require exact token-page owners")
    writer, profile = first.writer, first.profile
    spans = []
    table = []
    row_begin = 0
    for owner, rows in zip(owners, row_counts):
        if (type(owner) is not PagedKVTokenOwner or owner.writer is not writer or
                owner.profile != profile):
            raise ValueError("staged lanes must share the native arena and KV geometry")
        _positive("row count", rows)
        handles = owner.staged_handles(rows)
        sequence = owner.sequence
        end = owner.offset + rows
        spans.append(SequenceSpan(
            row_begin, rows, owner.offset, end, sequence.retained_start,
            sequence.first_block, len(table), len(handles),
            max(handle.generation for handle in handles), "causal", None,
        ))
        table.extend(handles)
        row_begin += rows
    unique_handles = tuple(dict.fromkeys(table))
    lease = writer.ledger.prepare(unique_handles)
    try:
        pinned_generations = {handle.page_id: handle.generation
                              for handle in unique_handles}
        plan = PagedAttentionPlan(
            spans=tuple(spans), page_table=tuple(table), total_rows=row_begin,
            query_heads=query_heads, kv_heads=profile.kv_heads,
            head_dim=profile.head_dim, dtype=profile.dtype,
            pool_capacity=writer.pool.capacity,
            live_generations=pinned_generations,
            **({'profile':('q1_long_n20_v1' if n20 and all(n == 1 for n in row_counts)
                            else 'prefill_long_n20_v1' if n20 else 'prefill_long_nax_v1'),
                'max_work_items':(20 if n20 else 2)*8192*24,
                'max_scratch_bytes':64*1024*1024 if n20 and all(n == 1 for n in row_counts) else 0}
               if long_fused else {}),
        )
        metadata = build_paged_read_metadata(
            plan, omit_long_row_span=long_fused and not all(n == 1 for n in row_counts))
    except BaseException:
        writer.ledger.abort_before_submit(lease)
        raise
    return PackedTokenRead(writer, plan, metadata, lease,
                           metadata_build_ns=perf_counter_ns() - started if profile_host else None)


def prepare_staged_token_read_n20(
    owners: tuple[PagedKVTokenOwner, ...], row_counts: tuple[int, ...],
    *, query_heads: int, permit_candidate: bool = False,
    profile_host: bool = False,
) -> PackedTokenRead:
    return prepare_staged_token_read(
        owners, row_counts, query_heads=query_heads,
        permit_candidate=permit_candidate, profile_host=profile_host,
        long_fused=True, n20=True)
