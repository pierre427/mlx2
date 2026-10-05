"""Source-bound direct Metal oracle for the default-off packed multirow K/V writer.

Run only under an owned GPUQ lease. This proves writer bytes and lifecycle; it
does not qualify a model route or measure serving performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

from varlen_pack_price_bench import _gpuq_owner, source_tree_sha256

ROOT = Path(__file__).resolve().parents[2]
CASES = ((32, 96), (63, 129))
KV_HEADS, DIM, PAGE_TOKENS = 4, 256, 64
PAGE_BYTES = KV_HEADS * DIM * PAGE_TOKENS * 2
MAX_SECONDS = 60
MAX_RESIDENT_BYTES = 4 * 1024**3
CASE_CLEANUP: list[dict] = []
FAILURE_ROOTS: list[object] = []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(manifest: dict) -> dict:
    required = {"source_commit", "source_tree_sha256", "native_path",
                "native_sha256", "mlx_version"}
    if type(manifest) is not dict or set(manifest) != required:
        raise ValueError("exact packed multirow source/native manifest required")
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                text=True).strip() != manifest["source_commit"] or
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip() or
            source_tree_sha256() != manifest["source_tree_sha256"]):
        raise RuntimeError("source revision, cleanliness, or tracked bytes differ")
    native = Path(manifest["native_path"]).resolve(strict=True)
    if sha256(native) != manifest["native_sha256"]:
        raise RuntimeError("native binary bytes differ")
    return {"schema": "mlx2.packed-multirow-write-oracle.v1", "status": "preflight_passed",
            "gpu_executed": False, "source_commit": manifest["source_commit"],
            "native_sha256": manifest["native_sha256"], "cases": [list(x) for x in CASES],
            "hard_seconds": MAX_SECONDS, "max_resident_bytes": MAX_RESIDENT_BYTES}


def _case(mx, np, counts: tuple[int, int], dtype: str) -> dict:
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
    from mlx2.runtime.paged_kv_write import NativeWriteBackend, PagedKVWriteOwner
    from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend

    pages = 8
    profile = TokenKVProfile(KV_HEADS, DIM, dtype)
    pool = PagedKVPool(pages)
    stream = mx.default_stream(mx.gpu)
    arena = NativeWriteBackend(pages * PAGE_BYTES, stream, permit_candidate=True,
                               storage_dtype=dtype)
    writer = PagedKVWriteOwner(pool, arena, page_bytes=PAGE_BYTES,
                               permit_candidate=True)
    backend = NativeQwen3PagedBackend(writer, permit_candidate=True, timeout_s=5)
    owners = tuple(PagedKVTokenOwner(writer, profile, permit_candidate=True)
                   for _ in range(2))
    state: dict = {"read_lease": None, "roots": ()}
    try:
        return _case_body(mx, np, counts, dtype, pages, pool, stream, arena,
                          writer, backend, owners, state)
    except BaseException:
        cleanup = {"counts": list(counts), "dtype": dtype,
                   "synchronized": False, "read_lease_terminal": False,
                   "owners_closed": False, "arena_closed": False}
        # A failed raw-byte assertion still needs the already-submitted
        # terminal and diagnostic read lease retired before arena teardown.
        try:
            mx.synchronize(stream)
            cleanup["synchronized"] = True
            deadline = time.monotonic() + 5
            while writer.pending_epochs and time.monotonic() < deadline:
                writer.poll_completions(wait_timeout_s=0.1)
            for owner in owners:
                owner.poll_completions()
            if state["read_lease"] is not None:
                writer.ledger.complete(state["read_lease"])
            cleanup["read_lease_terminal"] = writer.ledger.pending_count == 0
            if not writer.pending_epochs and writer.ledger.pending_count == 0:
                for owner in owners:
                    owner.close()
                writer.pool.retire(writer.ledger.completed_epoch)
                cleanup["owners_closed"] = True
                arena.close_after_terminal()
                cleanup["arena_closed"] = True
        except BaseException as cleanup_error:
            cleanup["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
        cleanup["pending_epochs"] = len(writer.pending_epochs)
        cleanup["pending_leases"] = writer.ledger.pending_count
        cleanup["retained_pages"] = pages - pool.free_count
        CASE_CLEANUP.append(cleanup)
        if not cleanup["arena_closed"]:
            FAILURE_ROOTS.append((arena, writer, backend, owners, state))
        raise


def _case_body(mx, np, counts, dtype, pages, pool, stream, arena, writer,
               backend, owners, state: dict) -> dict:
    total = sum(counts)
    source = (np.arange(total * KV_HEADS * DIM, dtype=np.float32)
              .reshape(total, KV_HEADS, DIM) % 4096) / 4096
    keys = mx.array(source).astype(getattr(mx, dtype))
    values = mx.array(source * 0.5 + 0.125).astype(getattr(mx, dtype))
    state["roots"] = (keys, values)
    ticket, = backend.append_packed_multirow(owners, keys, values, counts,
                                             permit_candidate=True)
    if (not ticket.packed_multirow or ticket.grouped_q1 or
            writer.ledger.pending_count != 1 or
            tuple(owner.offset for owner in owners) != (0, 0)):
        raise AssertionError("one unpublished shared packed-write lease differs")
    touched = tuple(handle for owner, n in zip(owners, counts)
                    for handle in owner.staged_handles(n))
    read_lease = writer.ledger.prepare(touched)
    writer.ledger.submit(read_lease)
    state["read_lease"] = read_lease
    readbacks = tuple(arena.diagnostic_read(ticket.dependency,
                      handle.page_id * PAGE_BYTES, PAGE_BYTES,
                      permit_diagnostic=True) for handle in touched)
    state["roots"] = (keys, values, readbacks)
    mx.eval(*(plane for pair in readbacks for plane in pair))
    mx.synchronize(stream)
    expected_k = np.asarray(keys.view(mx.uint16))
    expected_v = np.asarray(values.view(mx.uint16))
    tables = (owners[0].staged_handles(counts[0]), owners[1].staged_handles(counts[1]))
    physical = {handle: tuple(np.asarray(plane).view(np.uint16).reshape(
        KV_HEADS, PAGE_TOKENS, DIM) for plane in pair)
        for handle, pair in zip(touched, readbacks)}
    for lane in range(2):
        for local in range(counts[lane]):
            row = local + (counts[0] if lane else 0)
            k, v = physical[tables[lane][local // PAGE_TOKENS]]
            slot = local % PAGE_TOKENS
            if (not np.array_equal(k[:, slot, :], expected_k[row]) or
                    not np.array_equal(v[:, slot, :], expected_v[row])):
                raise AssertionError(
                    f"raw K/V bytes differ at lane {lane}, row {local}; "
                    f"K actual={k[0,slot,:8].tolist()} expected={expected_k[row,0,:8].tolist()}, "
                    f"V actual={v[0,slot,:8].tolist()} expected={expected_v[row,0,:8].tolist()}")
    deadline = time.monotonic() + 5
    while writer.pending_epochs and time.monotonic() < deadline:
        writer.poll_completions(wait_timeout_s=0.1)
    if writer.pending_epochs or not all(owner.poll_completions() for owner in owners):
        raise AssertionError("one shared write terminal did not publish both lanes")
    if tuple(owner.offset for owner in owners) != counts:
        raise AssertionError("accepted logical lengths differ")
    writer.ledger.complete(read_lease)
    for owner in owners:
        owner.close()
    writer.pool.retire(writer.ledger.completed_epoch)
    if (writer.ledger.pending_count or writer.pending_epochs or writer.poisoned or
            arena.grouped_multirow_write_count() != 1 or
            arena.grouped_multirow_row_count() != total or
            arena.grouped_q1_write_count() != 0 or
            arena.write_dispatch_count() != 0 or pool.free_count != pages):
        raise AssertionError("dispatch counters, read lease, or page retirement differ")
    arena.close_after_terminal()
    return {"counts": list(counts), "dtype": dtype, "raw_bytes_exact": True,
            "packed_dispatches": 1, "packed_rows": total,
            "direct_dispatches": 0, "grouped_q1_dispatches": 0,
            "write_terminal_successes": 1, "pending_epochs": 0,
            "retained_pages": 0}


def run(manifest: dict) -> dict:
    preflight(manifest)
    owner = _gpuq_owner()
    import _paged_kv_native
    import mlx.core as mx
    import numpy as np
    if (Path(_paged_kv_native.__file__).resolve() !=
            Path(manifest["native_path"]).resolve() or
            mx.__version__ != manifest["mlx_version"] or
            mx.default_device() != mx.gpu or not mx.metal.is_available()):
        raise RuntimeError("loaded MLX/native/GPU identity differs")
    if os.environ.get("MLX2_PAGED_GROUPED_MULTIROW_WRITE") != "1":
        raise RuntimeError("explicit packed multirow selector required")
    cases = [_case(mx, np, counts, dtype) for dtype in ("float16", "bfloat16")
             for counts in CASES]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if peak > MAX_RESIDENT_BYTES:
        raise MemoryError("packed multirow oracle RSS ceiling exceeded")
    return {"schema": "mlx2.packed-multirow-write-oracle.v1", "status": "passed",
            "gpu_executed": True, "source_commit": manifest["source_commit"],
            "source_tree_sha256": manifest["source_tree_sha256"],
            "native_sha256": manifest["native_sha256"], "gpuq_owner": owner,
            "cases": cases, "peak_resident_bytes": peak,
            "max_resident_bytes": MAX_RESIDENT_BYTES, "hard_seconds": MAX_SECONDS,
            "qualified": False, "selected": False, "price_usable": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not args.preflight_only:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
            TimeoutError("packed multirow oracle exceeded 60 seconds")))
        signal.alarm(MAX_SECONDS)
    try:
        result = preflight(manifest) if args.preflight_only else run(manifest)
    except BaseException as exc:
        result = {"schema": "mlx2.packed-multirow-write-oracle.v1",
                  "status": "failed", "gpu_executed": not args.preflight_only,
                  "source_commit": manifest.get("source_commit"),
                  "native_sha256": manifest.get("native_sha256"),
                  "error_type": type(exc).__name__, "error": str(exc),
                  "case_cleanup": list(CASE_CLEANUP),
                  "unresolved_failure_roots": len(FAILURE_ROOTS),
                  "qualified": False, "selected": False, "price_usable": False}
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        raise
    finally:
        signal.alarm(0)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
