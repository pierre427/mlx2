"""CPU-only, default-off geometry contract for a future paged attention route.

This module neither allocates pages nor executes attention. An owner must
provide current page generations before a plan can be constructed; callers
must still retain those pages for the entire eventual GPU epoch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

U32_MAX = (1 << 32) - 1
U64_MAX = (1 << 64) - 1
PAGE_SIZE = 64
KV_CHUNK = 512


def _uint(name: str, value: int, limit: int = U32_MAX) -> None:
    if type(value) is not int or not 0 <= value <= limit:
        raise ValueError(f"{name} must be an integer in [0, {limit}]")


def _positive(name: str, value: int, limit: int = U32_MAX) -> None:
    _uint(name, value, limit)
    if value == 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class PageHandle:
    """A physical slot and its ABA-protecting generation, not an ownership lease."""

    page_id: int
    generation: int

    def __post_init__(self) -> None:
        _uint("page_id", self.page_id)
        _positive("page generation", self.generation)


@dataclass(frozen=True)
class SequenceSpan:
    """One packed query interval and its absolute, retained KV interval.

    The page table maps every logical block intersecting
    [retained_start, kv_end). A partial first page does not expose tokens
    before retained_start. ``query_start + row_count == kv_end`` means the
    planned queries are the new suffix of this KV state.
    """

    row_begin: int
    row_count: int
    query_start: int
    kv_end: int
    retained_start: int
    first_block: int
    table_begin: int
    table_count: int
    owner_generation: int
    mask_kind: str = "causal"
    window: int | None = None

    @property
    def row_end(self) -> int:
        return self.row_begin + self.row_count

    def visible_bounds(self, local_row: int) -> tuple[int, int]:
        """Return the half-open visible key interval for one local query row."""
        _uint("local_row", local_row)
        if local_row >= self.row_count:
            raise ValueError("local_row is outside the span")
        upper = self.query_start + local_row + 1
        lower = self.retained_start
        if self.window is not None:
            lower = max(lower, upper - self.window)
        return lower, upper


@dataclass(frozen=True)
class PagedAttentionPlan:
    """Validated immutable host descriptor for the initial dense vector profile.

    Deliberately excludes q8, QSA, sinks, arbitrary masks, writes, speculative
    branches and non-64-token pages until separate profiles provide them.
    ``live_generations`` is owner-supplied and is checked only at construction;
    the eventual owner must pin the pages and revalidate before submission.
    """

    spans: tuple[SequenceSpan, ...]
    page_table: tuple[PageHandle, ...]
    total_rows: int
    query_heads: int
    kv_heads: int
    head_dim: int
    dtype: str
    pool_capacity: int
    live_generations: Mapping[int, int]
    max_work_items: int = 65536
    max_scratch_bytes: int = 1 << 30
    profile: str = "dense_vector_v1"
    page_size: int = PAGE_SIZE

    def __post_init__(self) -> None:
        _uint("total_rows", self.total_rows)
        _positive("query_heads", self.query_heads)
        _positive("kv_heads", self.kv_heads)
        _positive("head_dim", self.head_dim)
        _positive("pool_capacity", self.pool_capacity)
        _positive("max_work_items", self.max_work_items)
        _uint("max_scratch_bytes", self.max_scratch_bytes, U64_MAX)
        if self.profile not in ("dense_vector_v1","prefill_long_nax_v1","prefill_long_n20_v1","q1_long_n20_v1") or self.page_size != PAGE_SIZE:
            raise ValueError("unsupported paged attention profile or page size")
        if self.dtype not in ("float16", "bfloat16") or self.head_dim not in (128, 256):
            raise ValueError("unsupported dtype or head dimension")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        if type(self.spans) is not tuple or type(self.page_table) is not tuple:
            raise ValueError("spans and page_table must be immutable tuples")
        if not isinstance(self.live_generations, Mapping):
            raise TypeError("live_generations must be a mapping")
        object.__setattr__(self, "live_generations", MappingProxyType(dict(self.live_generations)))
        if self.total_rows == 0 and (self.spans or self.page_table):
            raise ValueError("zero-work plan must have no spans or pages")
        if self.total_rows and not self.spans:
            raise ValueError("nonempty plan requires spans")

        # Bound every product before a descriptor is handed to a native layer.
        max_address = ((self.pool_capacity * self.kv_heads * PAGE_SIZE * self.head_dim) - 1) * 2
        if max_address > U64_MAX:
            raise ValueError("pool element address exceeds uint64")
        long_prefill=self.profile in ("prefill_long_nax_v1", "prefill_long_n20_v1")
        if long_prefill and (self.dtype!='bfloat16' or self.head_dim!=256 or self.query_heads!=24 or self.kv_heads!=4 or
                not (1 <= len(self.spans) <= 20 if self.profile == "prefill_long_n20_v1" else len(self.spans) == 2) or
                self.max_scratch_bytes!=0 or self.max_work_items!=(20 if self.profile == "prefill_long_n20_v1" else 2)*8192*24):
            raise ValueError('long fused plan requires exact bounded BF16 head geometry and zero score scratch')
        n20_q1 = self.profile == "q1_long_n20_v1"
        if n20_q1 and (self.dtype != 'bfloat16' or self.head_dim != 256 or
                self.query_heads != 24 or self.kv_heads != 4 or
                not 1 <= len(self.spans) <= 20 or self.max_work_items != 20*8192*24 or
                self.max_scratch_bytes != 64*1024*1024):
            raise ValueError('N20 Q1 plan geometry or scratch bound differs')
        expected_row = expected_table = scratch_bytes = work_items = 0
        seen_generations: dict[int, int] = {}
        for span in self.spans:
            if type(span) is not SequenceSpan:
                raise ValueError("spans must contain SequenceSpan values")
            for name in (
                "row_begin", "row_count", "query_start", "kv_end",
                "retained_start", "first_block", "table_begin", "table_count",
                "owner_generation",
            ):
                _uint(name, getattr(span, name))
            if span.row_count == 0 or span.owner_generation == 0:
                raise ValueError("each span needs rows and a positive owner generation")
            if span.row_begin != expected_row or span.table_begin != expected_table:
                raise ValueError("row offsets and table offsets must be contiguous and ordered")
            if span.row_end > self.total_rows or span.row_end > U32_MAX:
                raise ValueError("row interval exceeds packed rows")
            if not (span.retained_start <= span.query_start < span.kv_end <= U32_MAX):
                raise ValueError("invalid retained or causal positions")
            if span.query_start + span.row_count != span.kv_end:
                raise ValueError("query rows must be the new KV suffix")
            if span.first_block != span.retained_start // PAGE_SIZE:
                raise ValueError("first block does not match retained start")
            required = (span.kv_end - 1) // PAGE_SIZE - span.first_block + 1
            if span.table_count != required or span.table_begin + required > len(self.page_table):
                raise ValueError("page table does not exactly cover retained KV")
            if span.mask_kind == "causal":
                if span.window is not None:
                    raise ValueError("causal mask cannot specify a window")
            elif span.mask_kind == "sliding":
                _positive("window", span.window)  # type: ignore[arg-type]
            else:
                raise ValueError("unsupported mask kind")
            for handle in self.page_table[span.table_begin : span.table_begin + required]:
                if type(handle) is not PageHandle or handle.page_id >= self.pool_capacity:
                    raise ValueError("page handle is outside the pool")
                if self.live_generations.get(handle.page_id) != handle.generation:
                    raise ValueError("stale or unknown page generation")
                old = seen_generations.setdefault(handle.page_id, handle.generation)
                if old != handle.generation:
                    raise ValueError("one page slot has conflicting generations")
            if long_prefill and (not 256<=span.row_count<=8192 or span.kv_end>8192 or span.retained_start!=0 or
                    span.first_block!=0 or span.mask_kind!='causal' or span.window is not None):
                raise ValueError('long fused plan span geometry differs')
            if n20_q1 and (span.row_count != 1 or span.kv_end > 8192 or
                    span.retained_start != 0 or span.first_block != 0 or
                    span.mask_kind != 'causal' or span.window is not None):
                raise ValueError('N20 Q1 requires one full-prefix unwindowed row')
            if long_prefill:
                # Exact causal, unwindowed spans have affine upper bounds:
                # upper(r)=query_start+r+1, lower(r)=retained_start=0.
                # The checked suffix equality proves the last upper is kv_end;
                # the first upper is positive. Every intervening row is visible.
                # One logical work item per head/row; no global score scratch.
                work_items += self.query_heads * span.row_count
                if work_items > self.max_work_items:
                    raise ValueError("work item or scratch capacity exceeded")
            for local_row in range(0 if long_prefill else span.row_count):
                lower, upper = span.visible_bounds(local_row)
                if lower >= upper:
                    raise ValueError("query has no visible retained key")
                # The explicit fused kernel owns no global score scratch.
                # Bound logical head/query metadata separately from scalar
                # vector split work; preserve every visibility/page check.
                splits = 1 if long_prefill else (upper - lower + KV_CHUNK - 1) // KV_CHUNK
                work_items += self.query_heads * splits
                if not long_prefill:scratch_bytes += 4 * self.query_heads * splits * (self.head_dim + 2)
                if work_items > self.max_work_items or scratch_bytes > self.max_scratch_bytes:
                    raise ValueError("work item or scratch capacity exceeded")
            expected_row = span.row_end
            expected_table += required
        if expected_row != self.total_rows or expected_table != len(self.page_table):
            raise ValueError("row or page table has unused entries")

    def page_for_token(self, sequence_index: int, token: int) -> tuple[PageHandle, int]:
        """Resolve a retained absolute token; never expose stale partial-page data."""
        _uint("sequence_index", sequence_index)
        if sequence_index >= len(self.spans):
            raise ValueError("sequence index is outside the plan")
        span = self.spans[sequence_index]
        _uint("token", token)
        if not span.retained_start <= token < span.kv_end:
            raise ValueError("token is outside retained KV")
        index = span.table_begin + token // PAGE_SIZE - span.first_block
        return self.page_table[index], token % PAGE_SIZE
