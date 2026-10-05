"""Short default-off native K/V copy proof; run only inside an owned GPU lease."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def _terminal(owner, ticket, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        matches = [event for event in owner.poll_completions() if event.ticket is ticket]
        if matches:
            if len(matches) != 1 or not matches[0].succeeded:
                raise AssertionError("native command did not terminate successfully")
            return
        time.sleep(0.05)
    raise TimeoutError("native terminal event did not arrive")


def run_probe() -> dict:
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner

    page_bytes, copied_bytes = 64, 17
    stream = mx.default_stream(mx.gpu)
    pool = PagedKVPool(2)
    backend = NativeWriteBackend(2 * page_bytes, stream, permit_candidate=True)
    owner = PagedKVWriteOwner(pool, backend, page_bytes=page_bytes, permit_candidate=True)
    source, destination = pool.reserve(2)
    pool.retain((source,))  # Model a frozen prefix held by another branch.
    expected_k = np.arange(1, copied_bytes + 1, dtype=np.uint8)
    expected_v = np.arange(101, 101 + copied_bytes, dtype=np.uint8)
    write = owner.submit_write(source, within_page_offset=0, byte_count=copied_bytes,
                               key_bytes=mx.array(expected_k), value_bytes=mx.array(expected_v))
    mx.eval(write.dependency)
    mx.synchronize(stream)
    _terminal(owner, write)

    copy = owner.submit_copy(source, destination, byte_count=copied_bytes)
    read_lease = owner.ledger.prepare((source, destination))
    owner.ledger.submit(read_lease)
    copied_k, copied_v = backend.diagnostic_read(
        copy.dependency, page_bytes, copied_bytes, permit_diagnostic=True
    )
    source_k, source_v = backend.diagnostic_read(
        copy.dependency, 0, copied_bytes, permit_diagnostic=True
    )
    mx.eval(copied_k, copied_v, source_k, source_v)
    mx.synchronize(stream)
    for actual, expected in ((copied_k, expected_k), (copied_v, expected_v),
                             (source_k, expected_k), (source_v, expected_v)):
        if not np.array_equal(np.array(actual), expected):
            raise AssertionError("COW changed source or destination bytes")
    pool.release((destination,), after_epoch=read_lease.epoch)
    if pool.free_count:
        raise AssertionError("destination recycled before terminal proof")
    _terminal(owner, copy)
    if pool.free_count:
        raise AssertionError("destination recycled while reader pin is live")
    owner.ledger.complete(read_lease)  # Synchronized read is the proof.
    if pool.free_count != 1:
        raise AssertionError("destination did not retire after copy and reader")
    recycled = pool.reserve()[0]
    if recycled.page_id != destination.page_id or recycled.generation != destination.generation + 1:
        raise AssertionError("destination generation did not advance on reuse")
    return {
        "schema": "mlx2.varlen-native-cow-probe.v1",
        "host": platform.node(), "mlx_version": mx.__version__,
        "page_bytes": page_bytes, "copied_bytes": copied_bytes,
        "write_epoch": write.epoch, "copy_epoch": copy.epoch,
        "read_epoch": read_lease.epoch,
        "source_unchanged": True, "destination_exact": True,
        "destination_generation_reused": recycled.generation,
        "terminal_copy_succeeded": True,
        "gpu_command_failure_tested": False,
        "model_serving_tested": False,
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
