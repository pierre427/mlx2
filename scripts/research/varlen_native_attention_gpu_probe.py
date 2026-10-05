"""Default-dry-run native arena write-to-packed-attention read gate.

--execute-gpu requires a separately coordinated shared GPU lease and both
locks. This is a short candidate cell, not a model-serving qualification.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def dry_run() -> dict:
    return {
        "schema": "mlx2.varlen-native-attention-probe.v1",
        "mode": "dry-run", "gpu_executed": False,
        "planned_cell": "two fp16 d128 GQA lanes, 65 and 63 accepted tokens, four packed query rows",
        "native_arena_read_planned": True,
        "requires": ["fresh gpuq FIFO ownership", "both GPU lock guards", "no foreign service owner",
                     "extension rebuilt from this exact source against pinned MLX wheel"],
        "not_proven": ["native kernel compilation", "native read callback", "model serving",
                       "qualification", "performance"],
    }


def _wait_writes(owner, tickets, expected_offset: int, *, timeout_s: float = 10.0) -> None:
    expected_epochs = {ticket.epoch for ticket in tickets}
    if len(expected_epochs) != len(tickets) or not expected_epochs:
        raise AssertionError("each planned write needs a distinct terminal epoch")
    deadline = time.monotonic() + timeout_s
    remaining = expected_epochs
    while time.monotonic() < deadline:
        published = owner.poll_completions()
        remaining = expected_epochs.intersection(owner.writer.pending_epochs)
        if owner.writer.poisoned:
            raise AssertionError("at least one native K/V write failed")
        if published:
            # PagedKVTokenOwner publishes only after every one of its tickets
            # has received a successful terminal event. Explicitly check the
            # epoch set as well; an early single callback is insufficient.
            if remaining or owner.offset != expected_offset:
                raise AssertionError("native K/V suffix published before all terminal writes")
            return
        time.sleep(0.02)
    raise TimeoutError(f"native K/V write callbacks missing: {sorted(remaining)}")


def gpu_cell() -> dict:
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.paged_attention_metal import paged_attention_cpu_reference
    from mlx2.runtime.paged_attention_native import (
        complete_packed_read_after_event, native_paged_attention_read_fp16,
        poll_native_paged_read_events,
    )
    from mlx2.runtime.paged_attention_pack import prepare_packed_token_read
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("native paged attention probe requires an MLX GPU")
    stream = mx.default_stream(mx.gpu)
    rng = np.random.default_rng(10031)
    profile = TokenKVProfile(2, 128, "float16")
    pool = PagedKVPool(4)
    backend = NativeWriteBackend(pool.capacity * profile.page_bytes, stream,
                                 permit_candidate=True)
    writer = PagedKVWriteOwner(pool, backend, page_bytes=profile.page_bytes,
                               permit_candidate=True)
    owners = []
    logical = []
    final_dependency = None
    for tokens in (65, 63):
        owner = PagedKVTokenOwner(writer, profile, permit_candidate=True)
        keys = rng.normal(0, 0.25, (tokens, 2, 128)).astype(np.float16)
        values = rng.normal(0, 0.25, keys.shape).astype(np.float16)
        old_end = owner.offset
        spans = owner.planned_spans(tokens)
        def chunks(array):
            return tuple(mx.array(np.frombuffer(
                array[span.source_token_offset:span.source_token_offset + span.token_count,
                      span.kv_head].tobytes(), dtype=np.uint8).copy())
                for span in spans)
        tickets = owner.append(chunks(keys), chunks(values), token_count=tokens)
        for span, ticket in zip(spans, tickets):
            if (ticket.handle != owner.sequence.handles[span.block_index] or
                    span.within_page_offset !=
                    (span.kv_head * 64 + (old_end + span.source_token_offset) % 64) *
                    profile.head_token_bytes or
                    span.byte_count != span.token_count * profile.head_token_bytes):
                raise AssertionError("token writer layout disagrees with native attention address formula")
        final_dependency = tickets[-1].dependency
        _wait_writes(owner, tickets, tokens)
        owners.append(owner)
        logical.append((keys, values))
    mx.eval(final_dependency)
    if np.array(final_dependency).tolist() != [1]:
        raise AssertionError("last native write dependency was not satisfied")
    packed = prepare_packed_token_read(tuple(owners), (3, 1), query_heads=4,
                                       permit_candidate=True)
    plan = packed.plan
    k_arena = np.zeros((pool.capacity, 2, 64, 128), dtype=np.float16)
    v_arena = np.zeros_like(k_arena)
    for sequence, (keys, values) in enumerate(logical):
        for token in range(len(keys)):
            handle, slot = plan.page_for_token(sequence, token)
            k_arena[handle.page_id, :, slot] = keys[token]
            v_arena[handle.page_id, :, slot] = values[token]
    query = rng.normal(0, 0.25, (4, 4, 128)).astype(np.float16)
    expected = paged_attention_cpu_reference(plan, query, k_arena, v_arena)
    output = native_paged_attention_read_fp16(
        packed, backend, mx.array(query), final_dependency, permit_candidate=True)
    mx.eval(output)
    mx.synchronize(stream)
    actual = np.array(output)
    events = poll_native_paged_read_events(backend)
    matching = [event for event in events if event[0] == packed.lease.epoch]
    if len(matching) != 1 or len(events) != 1:
        raise AssertionError(f"native read terminal event missing or unexpected: {events}")
    succeeded = complete_packed_read_after_event(packed, matching[0])
    if not succeeded:
        raise AssertionError("native attention command buffer failed")
    error = actual.astype(np.float64) - expected.astype(np.float64)
    normalized_rms = float(np.sqrt(np.mean(error * error)) /
                           max(np.sqrt(np.mean(expected.astype(np.float64) ** 2)), 1e-12))
    for owner in owners:
        owner.close()
    assert pool.free_count == pool.capacity
    return {
        "schema": "mlx2.varlen-native-attention-probe.v1",
        "mode": "gpu-native", "gpu_executed": True,
        "native_arena_read_observed": True,
        "accepted_tokens": [65, 63], "packed_rows": 4,
        "physical_pages": len(plan.page_table), "read_epoch": packed.lease.epoch,
        "all_write_epochs_terminal_success": True,
        "last_write_dependency_one": True,
        "token_writer_layout_preflight": True,
        "terminal_success": succeeded, "normalized_rms": normalized_rms,
        "max_abs": float(np.max(np.abs(error))),
        "passed_screen": normalized_rms <= 2e-3,
        "serving_route_selected": False, "qualification": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = gpu_cell() if args.execute_gpu else dry_run()
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result.get("passed_screen") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
