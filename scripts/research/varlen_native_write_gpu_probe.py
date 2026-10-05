"""One bounded opt-in native paged KV GPU proof; never a serving route."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def _terminal_write(owner, ticket, *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        completions = owner.poll_completions()
        if completions:
            if (len(completions) != 1 or completions[0].ticket is not ticket
                    or not completions[0].succeeded):
                raise AssertionError("unexpected native write completion")
            return
        time.sleep(0.05)
    raise TimeoutError("native terminal write event did not arrive")


def run_probe() -> dict:
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_write import (
        NativeWriteBackend,
        PagedKVWriteOwner,
        WriteSubmissionError,
    )

    page_bytes = 64
    count = 8
    stream = mx.default_stream(mx.gpu)
    pool = PagedKVPool(1)
    backend = NativeWriteBackend(page_bytes, stream, permit_candidate=True)
    owner = PagedKVWriteOwner(pool, backend, page_bytes=page_bytes,
                              permit_candidate=True)
    handle = pool.reserve()[0]
    expected_k = np.arange(1, count + 1, dtype=np.uint8)
    expected_v = np.arange(101, 101 + count, dtype=np.uint8)
    ticket = owner.submit_write(
        handle, within_page_offset=0, byte_count=count,
        key_bytes=mx.array(expected_k), value_bytes=mx.array(expected_v),
    )
    read_lease = owner.ledger.prepare((handle,))
    owner.ledger.submit(read_lease)
    actual_k, actual_v = backend.diagnostic_read(
        ticket.dependency, 0, count, permit_diagnostic=True,
    )
    mx.eval(actual_k, actual_v)
    mx.synchronize(stream)
    observed_k = np.array(actual_k)
    observed_v = np.array(actual_v)
    if not np.array_equal(observed_k, expected_k) or not np.array_equal(observed_v, expected_v):
        raise AssertionError("native K/V read did not match written bytes")

    pool.release((handle,), after_epoch=read_lease.epoch)
    if pool.free_count != 0:
        raise AssertionError("page recycled before terminal proof")
    _terminal_write(owner, ticket)
    if pool.free_count != 0:
        raise AssertionError("page recycled while diagnostic read lease was pinned")
    # The synchronized evaluator is the read's completion proof. The native
    # write callback alone cannot retire this separate reader lease.
    owner.ledger.complete(read_lease)
    if pool.free_count != 1:
        raise AssertionError("page not recycled after both leases retired")
    reused = pool.reserve()[0]
    if reused.page_id != handle.page_id or reused.generation != handle.generation + 1:
        raise AssertionError("generation-tagged page reuse failed")

    # A malformed source is rejected before native enqueue. The host owner
    # poisons and retains its pin; this is not a simulated GPU command failure.
    try:
        owner.submit_write(
            reused, within_page_offset=0, byte_count=count,
            key_bytes=mx.array(expected_k.astype(np.int8)),
            value_bytes=mx.array(expected_v),
        )
    except WriteSubmissionError as error:
        rejected_epoch = error.epoch
    else:
        raise AssertionError("invalid dtype unexpectedly submitted")
    pool.release((reused,), after_epoch=rejected_epoch)
    if not owner.poisoned or pool.free_count != 0 or pool.references(reused) != 1:
        raise AssertionError("rejected write did not poison and retain its pin")
    return {
        "schema": "mlx2.varlen-native-write-probe.v1",
        "host": platform.node(),
        "mlx_version": mx.__version__,
        "page_bytes": page_bytes,
        "byte_count": count,
        "write_epoch": ticket.epoch,
        "read_epoch": read_lease.epoch,
        "reused_generation": reused.generation,
        "key_bytes": observed_k.tolist(),
        "value_bytes": observed_v.tolist(),
        "terminal_write_succeeded": True,
        "synchronized_read_completed": True,
        "pre_dispatch_rejection_poisoned_and_pinned": True,
        "gpu_command_failure_tested": False,
        "serving_route_selected": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-gpu", action="store_true", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe()
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
