"""One source-bound, default-off grouped Q1 Metal write and ownership oracle."""

from __future__ import annotations

import argparse
import json
import resource
import signal
import time
from pathlib import Path

from varlen_pack_price_bench import MAX_RESIDENT_BYTES, MAX_SECONDS, _gpuq_owner
from varlen_staged_graph_price import preflight_cpu

KV_HEADS = 8
DIM = 128
PAGE_TOKENS = 64
PAGE_BYTES = KV_HEADS * PAGE_TOKENS * DIM * 2
SLOTS = (3, 61)


def _source_planes():
    """The sliced channel axis exercises non-unit native source strides."""
    import mlx.core as mx
    import numpy as np

    shape = (2, KV_HEADS, 2 * DIM)
    numbers = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    key_base = (numbers / 1024).astype(np.float16)
    value_base = ((numbers + 137) / 2048).astype(np.float16)
    return (mx.array(key_base)[:, :, ::2], mx.array(value_base)[:, :, ::2],
            key_base[:, :, ::2], value_base[:, :, ::2])


def _terminal_write(owner, ticket, *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        completions = owner.poll_completions()
        if completions:
            if (len(completions) != 1 or completions[0].ticket is not ticket or
                    completions[0].succeeded is not True):
                raise AssertionError("grouped write terminal or ticket differs")
            return
        time.sleep(0.01)
    raise TimeoutError("grouped write terminal missing")


def run(manifest: dict) -> dict:
    preflight_cpu(manifest)
    import _paged_kv_native
    import mlx.core as mx
    import numpy as np
    from varlen_pack_price_bench import sha256

    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner

    if (mx.__version__ != manifest["identity"]["mlx_wheel_version"] or
            mx.default_device() != mx.gpu or not mx.metal.is_available() or
            Path(_paged_kv_native.__file__).resolve() !=
            Path(manifest["paths"]["kernel"]).resolve() or
            sha256(Path(_paged_kv_native.__file__)) != manifest["identity"]["kernel_sha256"]):
        raise RuntimeError("live MLX/native identity or GPU differs")
    stream = mx.default_stream(mx.gpu)
    pool = PagedKVPool(2)
    backend = NativeWriteBackend(2 * PAGE_BYTES, stream, permit_candidate=True)
    owner = PagedKVWriteOwner(pool, backend, page_bytes=PAGE_BYTES,
                              permit_candidate=True)
    handles = tuple(pool.reserve(2))
    if tuple(handle.page_id for handle in handles) != (0, 1):
        raise AssertionError("private page IDs differ")
    keys, values, expected_keys, expected_values = _source_planes()
    if keys.shape != (2, KV_HEADS, DIM) or values.shape != keys.shape:
        raise AssertionError("strided source shape differs")
    ticket = owner.submit_grouped_q1_write(
        handles, SLOTS, keys=keys, values=values, kv_heads=KV_HEADS, dim=DIM)
    read_lease = owner.ledger.prepare(handles)
    owner.ledger.submit(read_lease)
    readbacks = [backend.diagnostic_read(ticket.dependency,
                 handle.page_id * PAGE_BYTES, PAGE_BYTES, permit_diagnostic=True)
                 for handle in handles]
    mx.eval(*(plane for pair in readbacks for plane in pair))
    mx.synchronize(stream)
    for row, (observed_k, observed_v) in enumerate(readbacks):
        actual_k = np.asarray(observed_k).view(np.float16).reshape(KV_HEADS, PAGE_TOKENS, DIM)
        actual_v = np.asarray(observed_v).view(np.float16).reshape(KV_HEADS, PAGE_TOKENS, DIM)
        if (not np.array_equal(actual_k[:, SLOTS[row], :], expected_keys[row]) or
                not np.array_equal(actual_v[:, SLOTS[row], :], expected_values[row])):
            raise AssertionError(f"grouped Q1 K/V destination bytes differ for lane {row}")
    pool.release(handles, after_epoch=read_lease.epoch)
    if pool.free_count != 0:
        raise AssertionError("pages recycled before grouped terminal")
    _terminal_write(owner, ticket)
    if pool.free_count != 0:
        raise AssertionError("pages recycled while synchronized read lease is pinned")
    owner.ledger.complete(read_lease)
    if (pool.free_count != 2 or owner.pending_epochs or
            owner.ledger.pending_count != 0 or owner.poisoned or
            backend.grouped_q1_write_count() != 1 or
            backend.write_dispatch_count() != 0):
        raise AssertionError("grouped primitive count or final ownership differs")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak <= 0 or peak > manifest["max_resident_bytes"]:
        raise RuntimeError("grouped Q1 oracle resident ceiling exceeded")
    return {"schema": "mlx2.grouped-q1-native-oracle.v1", "status": "passed",
            "gpu_executed": True, "source_commit": manifest["identity"]["source_commit"],
            "source_tree_sha256": manifest["identity"]["source_tree_sha256"],
            "kernel_sha256": manifest["identity"]["kernel_sha256"],
            "gpuq_owner": _gpuq_owner(), "slots": list(SLOTS),
            "source_shape": [2, KV_HEADS, DIM], "source_channel_slice_step": 2,
            "private_page_ids": [handle.page_id for handle in handles],
            "grouped_dispatches": 1, "ordinary_write_dispatches": 0,
            "terminal_successes": 1, "synchronized_diagnostic_reads": 2,
            "pending_epochs": 0, "retained_pages": 0, "byte_exact": True,
            "peak_resident_bytes": peak, "max_resident_bytes": MAX_RESIDENT_BYTES,
            "hard_seconds": MAX_SECONDS,
            "qualified": False, "selected": False, "serving_selected": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not args.preflight_only:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
            TimeoutError("grouped Q1 oracle exceeded 180 seconds")))
        signal.alarm(MAX_SECONDS)
    try:
        result = (preflight_cpu(manifest) if args.preflight_only else run(manifest))
    finally:
        signal.alarm(0)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
