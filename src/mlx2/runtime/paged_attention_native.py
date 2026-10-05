"""Default-off native fp16 attention read over the opaque paged KV arena.

The C++ primitive consumes a validated packed host plan and the same arena
that received K/V writes. Read completion events are distinct from writer
events. No adapter or serving route imports this experimental module.
"""

from __future__ import annotations

import math
import os

from .paged_attention_pack import PackedTokenRead
from .paged_kv_write import NativeWriteBackend


class NativePagedReadSubmissionError(RuntimeError):
    """A read may have reached the stream; keep its page lease until terminal proof."""


def _validate_pinned_read_handles(packed: PackedTokenRead) -> None:
    """Revalidate only the exact generations already pinned by the read lease."""
    try:
        for handle in dict.fromkeys(packed.plan.page_table):
            packed.writer.pool.references(handle)
    except ValueError as exc:
        raise ValueError("paged read handle generation changed before submission") from exc


def _stock_q1_sdpa(packed, backend, query, dependency, spans, factor, native, mx):
    """Build a complete lazy owned-K/V reference graph before lease submission."""
    plan = packed.plan
    visible = tuple(upper - lower for span in plan.spans
                    for lower, upper in (span.visible_bounds(0),))
    if (plan.total_rows != 2 or len(plan.spans) != 2 or
            any(span.row_count != 1 for span in plan.spans) or
            any(not 32 <= count <= 128 for count in visible) or
            plan.kv_heads > 32 or len(plan.page_table) > 6):
        raise ValueError("stock SDPA diagnostic requires two bounded short Q1 lanes")
    # Gather returns fresh owned arrays: SDPA never observes an arena view.
    # Both primitives stay on the writer stream and retain the exact fence.
    with mx.stream(backend.stream):
        keys, values = native.gather_q1_fp16(
            backend._arena, query, dependency, spans,
            [handle.page_id for handle in plan.page_table], plan.kv_heads,
            factor, packed.lease.epoch, backend.stream, True)
        mask = (mx.arange(128)[None, :] < mx.array(visible)[:, None]).reshape(2, 1, 1, 128)
        result = mx.fast.scaled_dot_product_attention(
            query.reshape(2, plan.query_heads, 1, plan.head_dim), keys, values,
            scale=factor, mask=mask).reshape(2, plan.query_heads, plan.head_dim)
    backend.stock_sdpa_graph_calls += 1
    return result


def native_paged_attention_read_fp16(
    packed: PackedTokenRead,
    backend: NativeWriteBackend,
    query: object,
    dependency: object,
    *,
    scale: float | None = None,
    permit_candidate: bool = False,
) -> object:
    """Enqueue a short-query dense read; caller later drains native read events.

    The caller serializes pool operations, retains ``packed`` and the result,
    and calls ``complete_packed_read_after_event`` for its terminal epoch.
    Cancellation is never a completion proof. The plan and arena must declare the same 16-bit storage dtype.
    """
    if not permit_candidate:
        raise RuntimeError("native paged attention requires explicit candidate enablement")
    if type(packed) is not PackedTokenRead or type(backend) is not NativeWriteBackend:
        raise TypeError("a packed token read and its exact native backend are required")
    if packed.state != "prepared" or packed.writer.backend is not backend:
        raise ValueError("packed read is not prepared for this native arena")
    if packed.writer.poisoned:
        raise RuntimeError("native paged arena is poisoned")
    plan = packed.plan
    if plan.dtype not in ("float16", "bfloat16") or plan.dtype != getattr(backend, "storage_dtype", "float16"):
        raise ValueError("native paged attention plan and arena storage dtype differ")
    if plan.profile=='prefill_long_nax_v1' and (os.environ.get('MLX2_PAGED_PREFILL_NAX_LONG_FUSED')!='1' or
            os.environ.get('MLX2_PAGED_PREFILL_MATRIX')!='1' or os.environ.get('MLX2_PAGED_PREFILL_NAX_EXACT','0')!='0'):
        raise ValueError('long fused native plan requires its exact explicit selector')
    if plan.profile in ('prefill_long_n20_v1', 'q1_long_n20_v1') and (
            os.environ.get('MLX2_PAGED_PACKED_N20') != '1' or
            os.environ.get('MLX2_PAGED_PREFILL_NAX_LONG_FUSED') != '1' or
            os.environ.get('MLX2_PAGED_PREFILL_MATRIX') != '1' or
            os.environ.get('MLX2_PAGED_PREFILL_NAX_EXACT', '0') != '0' or
            (plan.profile == 'q1_long_n20_v1' and
             os.environ.get('MLX2_PAGED_Q1_STOCK_LONG') != '1')):
        raise ValueError('N20 native read requires its exact explicit selectors')
    # The prepared read lease pins each unique handle. Revalidate those exact
    # generations here without rescanning the entire arena once *per handle*.
    _validate_pinned_read_handles(packed)
    factor = float(scale) if scale is not None else plan.head_dim ** -0.5
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("attention scale must be finite and positive")
    import _paged_kv_native as native
    import mlx.core as mx

    if plan.profile in ('prefill_long_n20_v1', 'q1_long_n20_v1'):
        capability = getattr(native, 'packed_n20_capability', None)
        if not callable(capability) or capability().get('max_spans') != 20:
            raise ValueError('native N20 read capability unavailable')

    if (type(query) is not mx.array or tuple(query.shape) !=
            (plan.total_rows, plan.query_heads, plan.head_dim) or
            query.dtype != getattr(mx, plan.dtype)):
        raise ValueError("query shape, dtype or layout disagrees with packed plan")
    if (type(dependency) is not mx.array or dependency.dtype != mx.uint8 or
            dependency.ndim != 1 or dependency.size != 1):
        raise ValueError("one native uint8 write dependency is required")
    spans = [
        [span.row_count, span.query_start, span.kv_end, span.retained_start,
         span.first_block, span.table_begin, span.table_count, span.window or 0]
        for span in plan.spans
    ]
    try:
        stock_q1 = (os.environ.get("MLX2_PAGED_Q1_STOCK_SDPA") == "1" and
                    plan.total_rows == 2 and len(plan.spans) == 2 and
                    all(span.row_count == 1 for span in plan.spans))
        if stock_q1:
            result = _stock_q1_sdpa(packed, backend, query, dependency, spans,
                                    factor, native, mx)
        else:
            result = native.attention_read_fp16(
                backend._arena, query, dependency, spans,
                [handle.page_id for handle in plan.page_table], plan.kv_heads,
                factor, packed.lease.epoch, backend.stream, True,
            )
    except Exception:
        packed.abort_before_submit()
        raise
    packed.mark_submitted()
    if backend.defer_staged_q1_eval or getattr(backend, "defer_staged_q1_writes", False):
        # The staged graph's final logits evaluation consumes this read.
        # Retain a root so construction failure can still submit the native
        # primitive and obtain a terminal callback for this submitted lease.
        backend._deferred_q1_read_roots.append(result)
        backend.deferred_q1_read_roots += 1
    if not backend.defer_staged_q1_eval:
        try:
            mx.async_eval(result)
            backend.staged_read_async_evals += 1
        except Exception as error:
            # Async evaluation may have encoded work. Hold the lease and require
            # a terminal native callback or explicit arena teardown proof.
            raise NativePagedReadSubmissionError(
                f"native paged read submission ambiguous at epoch {packed.lease.epoch}"
            ) from error
    return result


def poll_native_paged_read_events(backend: NativeWriteBackend) -> tuple[tuple[int, bool], ...]:
    """Drain only attention-read callbacks; writer/COW callbacks stay separate."""
    if type(backend) is not NativeWriteBackend:
        raise TypeError("native read events require the exact arena backend")
    import _paged_kv_native as native

    events = native.poll_read_completions(backend._arena)
    if any(type(event) is not tuple or len(event) != 2 or
           type(event[0]) is not int or type(event[1]) is not bool for event in events):
        raise RuntimeError("invalid native paged read completion event")
    return tuple(events)


def wait_native_paged_read_events(
    backend: NativeWriteBackend, timeout_seconds: float,
) -> tuple[tuple[int, bool], ...]:
    """Wait for a native callback, then drain the read queue exactly once."""
    if type(backend) is not NativeWriteBackend:
        raise TypeError("native read events require the exact arena backend")
    if not math.isfinite(timeout_seconds) or not 0 <= timeout_seconds <= 120:
        raise ValueError("native read wait must be finite and in [0, 120] seconds")
    import _paged_kv_native as native

    events = native.wait_read_completions(backend._arena, timeout_seconds)
    if any(type(event) is not tuple or len(event) != 2 or
           type(event[0]) is not int or type(event[1]) is not bool for event in events):
        raise RuntimeError("invalid native paged read completion event")
    return tuple(events)


def complete_packed_read_after_event(packed: PackedTokenRead, event: tuple[int, bool]) -> bool:
    """Release one matching reader pin after its native terminal callback."""
    if type(packed) is not PackedTokenRead or packed.state != "submitted":
        raise ValueError("packed read has no submitted native use")
    if (type(event) is not tuple or len(event) != 2 or
            type(event[0]) is not int or type(event[1]) is not bool or
            event[0] != packed.lease.epoch):
        raise ValueError("native read completion does not match the packed lease")
    packed.complete_after_proof(succeeded=event[1])
    return event[1]
