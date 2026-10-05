"""Short double-opt-in native terminal-failure retirement probe.

The Metal command buffer actually completes; the native test hook reports a
failed terminal event. This does not simulate a device fault or prove handling
of CommandBufferStatusError. Run only under the shared two-lock gpuq lease.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "native/paged_kv/arena.cpp", "native/paged_kv/arena.h",
    "native/paged_kv/binding.cpp", "src/mlx2/runtime/paged_kv_pool.py",
    "src/mlx2/runtime/paged_kv_native.py", "src/mlx2/runtime/paged_kv_write.py",
    "scripts/research/varlen_failed_terminal_gpu_probe.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(frozen: dict) -> dict:
    binary = Path(frozen["binary"])
    hashes = {name: digest(ROOT / name) for name in SOURCES}
    assert hashes == frozen["source_hashes"], "source hash drift"
    assert binary.is_file() and digest(binary) == frozen["binary_sha256"], "native binary drift"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   text=True).strip() == frozen["source_head"], "HEAD drift"
    session, lease = frozen["session"], frozen["lease_id"]
    assert os.environ.get("GPUQ_SESSION") == session, "GPUQ session drift"
    assert os.environ.get("GPUQ_LEASE") == lease, "GPUQ lease drift"
    owners = []
    for lock in (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock")):
        owner = json.loads((lock / "owner.json").read_text())
        assert owner["session"] == session and owner["lease_id"] == lease, "lock owner drift"
        owners.append(owner)
    with urllib.request.urlopen("http://127.0.0.1:8600/health", timeout=2) as response:
        health = json.load(response)
    assert not health.get("loaded") and not health.get("busy"), "Music3 is active"
    return {"source_hashes": hashes, "binary_sha256": digest(binary),
            "owners": owners, "music3": health}


def run() -> dict:
    import mlx.core as mx
    import _paged_kv_native as native
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_write import PagedKVWriteOwner

    class InjectedBackend:
        def __init__(self) -> None:
            self.stream = mx.new_stream(mx.gpu)
            self.arena = native.create_arena(64)
            self.plane_bytes = 64
            self.closed = False

        def write(self, key, value, offset, count, epoch):
            dependency = native.write(self.arena, key, value, offset, count,
                                      epoch, self.stream, True, True)
            mx.async_eval(dependency)
            return dependency

        def copy_page(self, *_args):
            raise AssertionError("copy not used by failure probe")

        def poll_completions(self):
            return native.poll_completions(self.arena)

        def close_after_terminal(self):
            mx.synchronize(self.stream)
            self.arena = None
            self.closed = True

    pool = PagedKVPool(1)
    backend = InjectedBackend()
    writer = PagedKVWriteOwner(pool, backend, page_bytes=64, permit_candidate=True)
    handle = pool.reserve()[0]
    key = mx.array([7] * 64, dtype=mx.uint8)
    value = mx.array([9] * 64, dtype=mx.uint8)
    ticket = writer.submit_write(handle, within_page_offset=0, byte_count=64,
                                 key_bytes=key, value_bytes=value)
    pool.release((handle,), after_epoch=ticket.epoch)
    mx.synchronize(backend.stream)
    events = writer.poll_completions()
    assert len(events) == 1 and events[0].ticket is ticket and not events[0].succeeded
    assert writer.poisoned and writer.pending_epochs == ()
    assert pool.quarantined_count == 1 and pool.free_count == 0
    assert pool.live_generations() == {}
    try:
        pool.reserve()
    except MemoryError:
        no_reuse = True
    else:
        no_reuse = False
    assert no_reuse
    writer.teardown_failed_arena()
    writer.teardown_failed_arena()
    assert writer.failed_arena_torn_down and backend.closed
    return {"epoch": ticket.epoch, "terminal_event": False,
            "writer_poisoned": writer.poisoned, "quarantined_count": pool.quarantined_count,
            "free_count": pool.free_count, "no_reuse": no_reuse,
            "teardown": writer.failed_arena_torn_down}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.preflight.read_text())
    evidence = preflight(frozen)
    if not args.execute_gpu:
        print(json.dumps({"preflight": "pass", **evidence}, sort_keys=True))
        return
    signal.alarm(60)
    start = time.monotonic()
    receipt = {"kind": "injected_terminal_failure_not_device_fault",
               "source_head": frozen["source_head"], "preflight": evidence}
    try:
        receipt["result"] = run()
        receipt["status"] = "pass"
    except BaseException as error:
        receipt["status"] = "fail"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["duration_s"] = time.monotonic() - start
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], **receipt["result"]}, sort_keys=True))


if __name__ == "__main__":
    main()
