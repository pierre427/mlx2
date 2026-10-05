"""CPU-only work counters for the default-off native vector reader.

These count the *current* one-threadgroup-per-(row, query-head) Metal source,
not measured GPU occupancy or kernel duration. They make B2/B4/mixed receipts
comparable without interpreting a host callback wait as device time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .paged_attention_plan import PagedAttentionPlan


@dataclass(frozen=True)
class PackedReadWork:
    spans: int
    rows: int
    query_heads: int
    physical_threadgroups: int
    page_table_entries: int
    unique_physical_pages: int
    serial_kv_steps: int
    longest_serial_kv_span: int
    kv_64_tile_slots: int
    useful_kv_64_tile_slots: int

    def as_receipt(self) -> dict[str, int]:
        return asdict(self)


def describe_packed_read(plan: PagedAttentionPlan) -> PackedReadWork:
    """Count validated row/head launches and serial KV work exactly."""
    if type(plan) is not PagedAttentionPlan:
        raise TypeError("packed work requires a validated PagedAttentionPlan")
    serial = longest = slots = 0
    for span in plan.spans:
        if plan.profile in ("prefill_long_nax_v1", "prefill_long_n20_v1"):
            # Validated unwindowed origin-zero spans have visible lengths
            # query_start+1 .. kv_end. Preserve the exact scalar-equivalent
            # diagnostics without walking every real query row.
            first, last = span.query_start + 1, span.kv_end
            serial += span.row_count * (first + last) // 2 * plan.query_heads
            longest = max(longest, last)
            def tile_prefix(end: int) -> int:
                full, tail = divmod(end, 64)
                return 64 * full * (full + 1) // 2 + tail * (full + 1)
            slots += (tile_prefix(last) - tile_prefix(first - 1)) * 64 * plan.query_heads
            continue
        for local in range(span.row_count):
            lower, upper = span.visible_bounds(local)
            visible = upper - lower
            serial += visible * plan.query_heads
            longest = max(longest, visible)
            slots += ((visible + 63) // 64) * 64 * plan.query_heads
    return PackedReadWork(
        spans=len(plan.spans), rows=plan.total_rows,
        query_heads=plan.query_heads,
        physical_threadgroups=plan.total_rows * plan.query_heads,
        page_table_entries=len(plan.page_table),
        unique_physical_pages=len(set(plan.page_table)),
        serial_kv_steps=serial, longest_serial_kv_span=longest,
        kv_64_tile_slots=slots, useful_kv_64_tile_slots=serial,
    )
