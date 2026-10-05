"""Default-dry-run 63/64/65-token native paged KV address/ownership gate.

--cpu-fake verifies fixture and host state only. --execute-gpu is an opt-in
same-stream native copy/read cell under an externally acquired gpuq lease.
Neither path executes attention or selects a serving route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import time
from pathlib import Path

SCHEMA = "mlx2.varlen-token-native-gate.v1"
BOUNDARIES = (63, 64, 65)
KV_HEADS = 2
HEAD_DIM = 128
PAGE_CAPACITY = 2
MAX_GPU_SECONDS = 300
EXPECTED_MLX_VERSION = "0.32.2.dev20260919+39400a0d4"
EXPECTED_BINARY_SHA256 = "bdc164d220a68e71f93c54479d1a7cde81c44d25b2a51b20e69dd91bd8270a0a"
EXPECTED_NATIVE_SOURCE_SHA256 = {
    "native/paged_kv/arena.cpp": "1e95adbc4a932d681f44336a367072d8205c88165554e2819197d3477c2fea33",
    "native/paged_kv/arena.h": "6e723b3604f7dcfd5e5e0375daf1a10d2fd94e8c308b0da23745b98b9b08b535",
    "native/paged_kv/binding.cpp": "bc836c93e88ea2038c244cee3f77298391a5017691b753fb1c1932dc4c8079c1",
}
ROOT = Path(__file__).resolve().parents[2]
LOCKS = (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock"))
BINARY = Path("/tmp/mlx2-varlen-binding-build-215/_paged_kv_native.cpython-312-darwin.so")
STAGE = "not-started"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _available_hash(path: Path) -> str | None:
    try:
        return sha256(path)
    except OSError:
        return None


def preflight(*, locks=LOCKS, binary=BINARY, source_root=ROOT,
              expected_binary=EXPECTED_BINARY_SHA256,
              expected_sources=EXPECTED_NATIVE_SOURCE_SHA256,
              session=None) -> dict:
    """Validate the *held* gpuq lease and pinned native bytes before MLX import."""
    session = os.environ.get("GPUQ_SESSION") if session is None else session
    if not session:
        raise RuntimeError("GPUQ_SESSION is required for native gate")
    lease = os.environ.get("GPUQ_LEASE") or f"{session}-token63-65"
    owners = []
    for lock in locks:
        if not lock.is_dir() or lock.is_symlink():
            raise RuntimeError(f"gpuq lock is not an owned directory: {lock}")
        owner = json.loads((lock / "owner.json").read_text())
        if (owner.get("session") != session or owner.get("lease_id") != lease or
                owner.get("label") != "token63-65" or
                type(owner.get("pid")) is not int or owner["pid"] <= 0):
            raise RuntimeError(f"gpuq owner mismatch: {lock}")
        try:
            os.kill(owner["pid"], 0)
        except OSError as exc:
            raise RuntimeError(f"gpuq owner process is not live: {lock}") from exc
        owners.append(owner)
    if owners[0] != owners[1]:
        raise RuntimeError("gpuq lock owner receipts disagree")
    source_hashes = {name: sha256(source_root / name) for name in expected_sources}
    if source_hashes != dict(expected_sources):
        raise RuntimeError("native source hashes differ from reviewed build")
    binary_hash = sha256(binary)
    if binary_hash != expected_binary:
        raise RuntimeError("native binding binary differs from reviewed build")
    return {"gpuq_owner": owners[0], "source_sha256": source_hashes,
            "binary_path": str(binary), "binary_sha256": binary_hash,
            "runner_sha256": sha256(Path(__file__).resolve())}


def _alarm(_signum, _frame):
    raise TimeoutError(f"native token gate exceeded {MAX_GPU_SECONDS}s hard cap")


def dry_run() -> dict:
    return {
        "schema": SCHEMA, "mode": "dry-run", "gpu_executed": False,
        "accepted_token_boundaries": list(BOUNDARIES),
        "profile": {"kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
                    "dtype": "float16", "physical_page_tokens": 64},
        "arena_bytes_total": 2 * PAGE_CAPACITY * KV_HEADS * 64 * HEAD_DIM * 2,
        "scope": ["head-local native K/V writes", "same-stream diagnostic copies",
                  "terminal callback publication", "generation reuse after read retirement"],
        "not_proven": ["attention on opaque arena", "native COW",
                       "real failed command-buffer retirement", "model serving"],
    }


def pattern(kind: str, first_token: int, count: int, head: int, head_bytes: int):
    """Distinct deterministic bytes for every head, token and K/V plane."""
    import numpy as np

    token = np.arange(first_token, first_token + count, dtype=np.uint32)[:, None]
    channel = np.arange(head_bytes, dtype=np.uint32)[None, :]
    salt = 17 if kind == "key" else 113
    return ((token * 37 + channel * 13 + head * 53 + salt) & 255).astype(np.uint8).reshape(-1)


def chunks(owner, count: int, make_array):
    spans = owner.planned_spans(count)
    start = owner.offset
    keys = tuple(make_array(pattern("key", start + span.source_token_offset,
                                    span.token_count, span.kv_head,
                                    owner.profile.head_token_bytes)) for span in spans)
    values = tuple(make_array(pattern("value", start + span.source_token_offset,
                                      span.token_count, span.kv_head,
                                      owner.profile.head_token_bytes)) for span in spans)
    return spans, keys, values


def expected_regions(owner, accepted: int):
    """Return reader-compatible physical offsets and exact logical payloads."""
    for block, handle in enumerate(owner.sequence.handles):
        first = block * 64
        count = min(64, accepted - first)
        for head in range(owner.profile.kv_heads):
            offset = handle.page_id * owner.profile.page_bytes + (
                head * 64 * owner.profile.head_token_bytes)
            yield (offset, count * owner.profile.head_token_bytes,
                   pattern("key", first, count, head, owner.profile.head_token_bytes),
                   pattern("value", first, count, head, owner.profile.head_token_bytes))


class FakeBackend:
    """CPU-only byte planes and injected callbacks, never a native proof."""

    def __init__(self, plane_bytes: int) -> None:
        self.plane_bytes = plane_bytes
        self.keys = bytearray(plane_bytes)
        self.values = bytearray(plane_bytes)
        self.events = []

    def validate_sources(self, keys, values):
        import mlx.core as mx
        return (type(keys) is mx.array and type(values) is mx.array and
                keys.dtype == values.dtype == mx.uint8 and
                keys.ndim == values.ndim == 1)

    def write(self, keys, values, offset, byte_count, epoch):
        import numpy as np
        self.keys[offset:offset + byte_count] = np.array(keys).tobytes()
        self.values[offset:offset + byte_count] = np.array(values).tobytes()
        self.events.append((epoch, True))
        return object()

    def poll_completions(self):
        events, self.events = self.events, []
        return events


def _owner(backend):
    from mlx2.runtime.paged_kv_pool import PagedKVPool
    from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
    from mlx2.runtime.paged_kv_write import PagedKVWriteOwner

    profile = TokenKVProfile(KV_HEADS, HEAD_DIM, "float16")
    pool = PagedKVPool(PAGE_CAPACITY)
    writer = PagedKVWriteOwner(pool, backend, page_bytes=profile.page_bytes,
                               permit_candidate=True)
    return PagedKVTokenOwner(writer, profile, permit_candidate=True)


def cpu_fake() -> dict:
    import mlx.core as mx
    import numpy as np

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        profile_bytes = PAGE_CAPACITY * KV_HEADS * 64 * HEAD_DIM * 2
        backend = FakeBackend(profile_bytes)
        owner = _owner(backend)
        results = []
        for boundary in BOUNDARIES:
            count = boundary - owner.offset
            _, keys, values = chunks(owner, count, mx.array)
            tickets = owner.append(keys, values, token_count=count)
            assert owner.offset == boundary - count
            assert owner.poll_completions() and owner.offset == boundary
            regions = list(expected_regions(owner, boundary))
            for offset, length, expected_k, expected_v in regions:
                assert np.array_equal(np.frombuffer(backend.keys[offset:offset + length],
                                                    dtype=np.uint8), expected_k)
                assert np.array_equal(np.frombuffer(backend.values[offset:offset + length],
                                                    dtype=np.uint8), expected_v)
            results.append({"tokens": boundary, "pages": len(owner.sequence.handles),
                            "writes": len(tickets), "regions": len(regions)})
        old = owner.sequence.handles
        owner.close()
        assert owner.writer.pool.free_count == PAGE_CAPACITY
        reused = owner.writer.pool.reserve(PAGE_CAPACITY)
        assert {h.page_id: h.generation for h in reused} == {
            h.page_id: h.generation + 1 for h in old}
        return {"schema": SCHEMA, "mode": "cpu-fake", "gpu_executed": False,
                "boundary_cases": results, "exact_fake_bytes": True,
                "generation_reuse": True, "native_proof": False}
    finally:
        mx.set_default_device(previous)


def _wait_published(owner, boundary: int, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        owner.poll_completions()
        if owner.writer.poisoned:
            raise AssertionError("native writer failed")
        if owner.offset == boundary:
            return
        time.sleep(0.05)
    raise TimeoutError(f"terminal callbacks did not publish {boundary} tokens")


def gpu_native(verified: dict) -> dict:
    global STAGE
    STAGE = "import-mlx"
    import mlx.core as mx
    import numpy as np
    from mlx2.runtime.paged_kv_write import NativeWriteBackend

    if mx.__version__ != EXPECTED_MLX_VERSION:
        raise RuntimeError("MLX wheel version differs from reviewed native build")
    STAGE = "create-native-arena"
    stream = mx.default_stream(mx.gpu)
    plane_bytes = PAGE_CAPACITY * KV_HEADS * 64 * HEAD_DIM * 2
    backend = NativeWriteBackend(plane_bytes, stream, permit_candidate=True)
    owner = _owner(backend)
    results = []
    final_reader = None
    for boundary in BOUNDARIES:
        STAGE = f"boundary-{boundary}-append"
        count = boundary - owner.offset
        spans, keys, values = chunks(owner, count, mx.array)
        tickets = owner.append(keys, values, token_count=count)
        handles = owner.sequence.handles
        reader = owner.writer.ledger.prepare(handles)
        owner.writer.ledger.submit(reader)
        regions = list(expected_regions(owner, boundary))
        STAGE = f"boundary-{boundary}-diagnostic-read"
        reads = [backend.diagnostic_read(tickets[-1].dependency, offset, length,
                                         permit_diagnostic=True)
                 for offset, length, _, _ in regions]
        mx.eval(*(array for pair in reads for array in pair))
        mx.synchronize(stream)
        observed = []
        for (offset, length, expected_k, expected_v), pair in zip(regions, reads):
            actual_k, actual_v = np.array(pair[0]), np.array(pair[1])
            if (not np.array_equal(actual_k, expected_k) or
                    not np.array_equal(actual_v, expected_v)):
                raise AssertionError(f"native token K/V mismatch at {boundary}, offset {offset}")
            observed.append(actual_k.tobytes() + actual_v.tobytes())
        STAGE = f"boundary-{boundary}-terminal-callbacks"
        _wait_published(owner, boundary)
        if boundary == BOUNDARIES[-1]:
            final_reader = reader
        else:
            owner.writer.ledger.complete(reader)  # synchronized read proof
        digest = hashlib.sha256(b"".join(observed)).hexdigest()
        results.append({"tokens": boundary, "pages": len(handles),
                        "write_epochs": [t.epoch for t in tickets],
                        "reader_epoch": reader.epoch, "regions": len(regions),
                        "exact_kv_sha256": digest})
    assert final_reader is not None
    STAGE = "final-reader-retirement"
    old = owner.sequence.handles
    owner.close()
    if owner.writer.pool.free_count != 0:
        raise AssertionError("page recycled with diagnostic reader pin")
    owner.writer.ledger.complete(final_reader)  # synchronized final read proof
    if owner.writer.pool.free_count != PAGE_CAPACITY:
        raise AssertionError("pages not retired after final reader proof")
    reused = owner.writer.pool.reserve(PAGE_CAPACITY)
    if {h.page_id: h.generation for h in reused} != {
            h.page_id: h.generation + 1 for h in old}:
        raise AssertionError("page generations did not advance after retirement")
    STAGE = "complete"
    return {"schema": SCHEMA, "mode": "gpu-native", "outcome": "passed",
            "gpu_executed": True, "preflight": verified,
            "host": platform.node(), "mlx_version": mx.__version__,
            "profile": dry_run()["profile"], "boundary_cases": results,
            "exact_same_stream_diagnostic_copies": True,
            "terminal_callbacks_and_generation_reuse": True,
            "attention_on_opaque_arena_tested": False,
            "native_command_failure_tested": False, "native_cow_tested": False,
            "serving_route_selected": False}


def main() -> None:
    global STAGE
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cpu-fake", action="store_true")
    mode.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.execute_gpu and args.receipt is None:
        parser.error("--execute-gpu requires --receipt for failure evidence")
    if args.execute_gpu:
        STAGE = "gpuq-and-build-preflight"
        previous_handler = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, MAX_GPU_SECONDS)
        try:
            verified = preflight()
            result = gpu_native(verified)
        except BaseException as exc:
            failure = {"schema": SCHEMA, "mode": "gpu-native", "outcome": "failed",
                       "gpu_executed": STAGE not in ("gpuq-and-build-preflight", "import-mlx"),
                       "failure_stage": STAGE, "error_type": type(exc).__name__,
                       "error": str(exc), "host": platform.node(),
                       "runner_sha256": _available_hash(Path(__file__).resolve()),
                       "source_sha256": {name: _available_hash(ROOT / name)
                                         for name in EXPECTED_NATIVE_SOURCE_SHA256},
                       "binary_path": str(BINARY),
                       "binary_sha256": _available_hash(BINARY)}
            if "verified" in locals():
                failure["preflight"] = verified
            args.receipt.write_text(json.dumps(failure, indent=2) + "\n")
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
    else:
        result = cpu_fake() if args.cpu_fake else dry_run()
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
