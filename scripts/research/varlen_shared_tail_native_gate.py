"""Default-dry-run, bounded native shared-tail COW publication gate."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path


def _wait(owner, expected_offset: int, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        owner.poll_completions()
        if owner.writer.poisoned:
            raise AssertionError("native writer was poisoned")
        if owner.offset == expected_offset and not owner._pending:
            return
        time.sleep(0.02)
    raise TimeoutError("native shared-tail callbacks did not publish the suffix")


def gpu_cell() -> dict:
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("native shared-tail gate requires an MLX GPU")
    stream = mx.default_stream(mx.gpu)
    profile = TokenKVProfile(2, 128, "float16")
    pool = PagedKVPool(3)
    backend = NativeWriteBackend(pool.capacity * profile.page_bytes, stream,
                                 permit_candidate=True)
    writer = PagedKVWriteOwner(pool, backend, page_bytes=profile.page_bytes,
                               permit_candidate=True)
    owner = PagedKVTokenOwner(writer, profile, permit_candidate=True)

    def fixture(begin: int, count: int):
        base = np.arange(begin, begin + count, dtype=np.float32)[:, None, None]
        heads = np.arange(2, dtype=np.float32)[None, :, None]
        channels = np.arange(128, dtype=np.float32)[None, None, :]
        keys = (base * 0.01 + heads * 0.1 + channels * 0.001).astype(np.float16)
        values = (base * 0.02 - heads * 0.1 + channels * 0.001).astype(np.float16)
        return keys, values

    def submit(keys, values):
        spans = owner.planned_spans(len(keys))
        def chunks(array):
            return tuple(mx.array(np.frombuffer(
                array[span.source_token_offset:span.source_token_offset + span.token_count,
                      span.kv_head].tobytes(), dtype=np.uint8).copy())
                for span in spans)
        return owner.append(chunks(keys), chunks(values), token_count=len(keys))

    old_k, old_v = fixture(0, 63)
    first_tickets = submit(old_k, old_v)
    mx.eval(*(ticket.dependency for ticket in first_tickets))
    mx.synchronize(stream)
    _wait(owner, 63)
    sibling = owner.sequence.fork()
    old_handle = sibling.handles[0]
    suffix_k, suffix_v = fixture(63, 2)
    copy_ticket, = submit(suffix_k, suffix_v)
    if owner.offset != 63 or owner.sequence.handles != sibling.handles:
        raise AssertionError("shared tail was published before copy terminal")
    mx.eval(copy_ticket.dependency)
    mx.synchronize(stream)
    owner.poll_completions()  # Successful copy queues the staged suffix writes.
    if owner.offset != 63 or len(owner._pending) != 4:
        raise AssertionError("shared tail was published before staged write terminals")
    suffix_tickets = owner._pending
    mx.eval(*(ticket.dependency for ticket in suffix_tickets))
    mx.synchronize(stream)
    _wait(owner, 65)
    if sibling.handles != (old_handle,) or owner.sequence.handles[0] == old_handle:
        raise AssertionError("COW changed the frozen sibling or failed to replace tail")

    read_handles = tuple(dict.fromkeys((*sibling.handles, *owner.sequence.handles)))
    lease = writer.ledger.prepare(read_handles)
    writer.ledger.submit(lease)
    expected_k = np.concatenate((old_k, suffix_k))
    expected_v = np.concatenate((old_v, suffix_v))
    copied = []
    for handle in read_handles:
        pair = backend.diagnostic_read(
            suffix_tickets[-1].dependency, handle.page_id * profile.page_bytes,
            profile.page_bytes, permit_diagnostic=True)
        copied.append(pair)
    mx.eval(*(array for pair in copied for array in pair))
    mx.synchronize(stream)
    writer.ledger.complete(lease)
    by_page = {
        handle: tuple(np.array(array).view(np.float16).reshape(2, 64, 128)
                      for array in pair)
        for handle, pair in zip(read_handles, copied)
    }
    old_page = by_page[old_handle]
    new_page = by_page[owner.sequence.handles[0]]
    next_page = by_page[owner.sequence.handles[1]]
    for plane, expected in ((0, expected_k), (1, expected_v)):
        if (not np.array_equal(old_page[plane][:, :63].transpose(1, 0, 2), expected[:63]) or
                not np.array_equal(new_page[plane][:, :64].transpose(1, 0, 2), expected[:64]) or
                not np.array_equal(next_page[plane][:, :1].transpose(1, 0, 2), expected[64:])):
            raise AssertionError("native shared-tail K/V byte parity failed")
    owner.close()
    sibling.abort(after_epoch=writer.ledger.completed_epoch)
    pool.retire(writer.ledger.completed_epoch)
    if pool.free_count != 3:
        raise AssertionError("shared-tail pages did not retire")
    return {
        "schema": "mlx2.varlen-shared-tail-native-gate.v1",
        "mode": "gpu-native", "gpu_executed": True,
        "old_tokens": 63, "accepted_tokens": 65, "physical_pages": 3,
        "copy_epoch": copy_ticket.epoch,
        "copy_terminal_before_suffix_publication": True,
        "all_suffix_writes_terminal_before_publication": True,
        "frozen_sibling_unchanged": True, "exact_kv_bytes": True,
        "all_pages_retired": True,
        "failed_command_buffer_tested": False,
        "serving_route_selected": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    signal.alarm(90) if args.execute_gpu else None
    result = gpu_cell() if args.execute_gpu else {
        "schema": "mlx2.varlen-shared-tail-native-gate.v1",
        "mode": "dry-run", "gpu_executed": False,
        "planned_cell": "63-token frozen tail, 2-token COW append over page boundary",
    }
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
