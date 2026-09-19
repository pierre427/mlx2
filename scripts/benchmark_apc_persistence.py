#!/usr/bin/env python3
"""Measure APCv2 persistence without touching Metal.

The synthetic entries preserve the production topology: ordinary KV planes,
recurrent ``ArraysCache`` planes, and a target-bound MTP sidecar.  Persistence
still goes through APCv2's real safetensors, fsync, digest, manifest, rescan,
restore, park, and resume code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from unittest import mock

import mlx.core as mx

# This benchmark is intentionally incapable of allocating on Metal.
mx.set_default_device(mx.cpu)

from mlx2.runtime.apc_v2 import APCKey, APCv2, MTPAPCSidecar
from mlx2.runtime.models.cache import ArraysCache, KVCache


GIB = 1 << 30
MIN_FREE_BYTES = 24 * GIB
LAYOUT = "apcv2-persistence-benchmark-hybrid-v1"
SCHEMA = "mlx2.apcv2-persistence-benchmark.v1"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def validate_scratch_root(value: Path, *, repo: Path | None = None) -> Path:
    root = value.expanduser().resolve()
    source = (repo or repository_root()).resolve()
    if root == source or source in root.parents:
        raise ValueError(f"--dir must be outside the repository: {source}")
    return root


def parse_vm_stat(text: str) -> dict[str, int]:
    first = text.splitlines()[0]
    marker = "page size of "
    if marker not in first:
        raise RuntimeError("vm_stat did not report its page size")
    page_size = int(first.split(marker, 1)[1].split()[0])
    pages: dict[str, int] = {}
    for line in text.splitlines()[1:]:
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        raw = raw.strip().rstrip(".").replace(".", "")
        if raw.isdigit():
            pages[name.strip()] = int(raw)
    return {"page_size": page_size, **pages}


def free_memory_bytes() -> int:
    try:
        output = subprocess.run(
            ["vm_stat"], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("vm_stat is required for the 24 GiB allocation guard") from error
    stats = parse_vm_stat(output)
    # Use only immediately free and speculative pages. Inactive/purgeable pages
    # are reclaimable in theory but are not counted as permission to squeeze a
    # concurrently running model.
    return stats["page_size"] * (
        stats.get("Pages free", 0) + stats.get("Pages speculative", 0)
    )


def require_allocation_headroom(*, allocation_bytes: int) -> int:
    free = free_memory_bytes()
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            "refusing synthetic APC allocation: vm_stat free+speculative "
            f"memory is {free / GIB:.2f} GiB, below 24 GiB"
        )
    if free - allocation_bytes < MIN_FREE_BYTES:
        raise RuntimeError(
            "refusing synthetic APC allocation: the next allocation would "
            f"leave less than 24 GiB free ({allocation_bytes / GIB:.3f} GiB requested)"
        )
    return free


def guarded_zeros(shape: tuple[int, ...], *, dtype=mx.float16):
    elements = 1
    for extent in shape:
        elements *= int(extent)
    allocation_bytes = elements * int(dtype.size)
    require_allocation_headroom(allocation_bytes=allocation_bytes)
    value = mx.zeros(shape, dtype=dtype)
    mx.eval(value)
    return value


def guarded_array(values, *, dtype):
    allocation_bytes = len(values) * int(dtype.size)
    require_allocation_headroom(allocation_bytes=allocation_bytes)
    value = mx.array(values, dtype=dtype)
    mx.eval(value)
    return value


def _kv_cache(array_bytes: int, *, token_count: int) -> tuple[KVCache, int]:
    unit = token_count * int(mx.float16.size)
    per_array = max(unit, (array_bytes // 2 // unit) * unit)
    width = per_array // unit
    cache = KVCache()
    cache.keys = guarded_zeros((1, 1, token_count, width))
    cache.values = guarded_zeros((1, 1, token_count, width))
    cache.offset = token_count
    return cache, per_array * 2


def _recurrent_cache(array_bytes: int, *, token_count: int) -> tuple[ArraysCache, int]:
    per_array = max(2, (array_bytes // 2) // 2 * 2)
    elements = per_array // int(mx.float16.size)
    cache = ArraysCache(2)
    cache[0] = guarded_zeros((1, elements))
    cache[1] = guarded_zeros((1, elements))
    cache.lengths = guarded_array([token_count], dtype=mx.int32)
    cache._host_lengths = (cache.lengths, [token_count])
    return cache, per_array * 2


def synthetic_hybrid_entry(entry_bytes: int):
    """Return a materialized hybrid target and MTP sidecar near ``entry_bytes``."""
    token_count = 256
    draft_tokens = token_count - 1
    target = []
    draft = []
    allocated = 0

    target_kv_budget = int(entry_bytes * 0.55)
    recurrent_budget = int(entry_bytes * 0.20)
    draft_budget = int(entry_bytes * 0.20)
    for _ in range(4):
        cache, used = _kv_cache(target_kv_budget // 4, token_count=token_count)
        target.append(cache)
        allocated += used
    for _ in range(4):
        cache, used = _recurrent_cache(
            recurrent_budget // 4, token_count=token_count
        )
        target.append(cache)
        allocated += used
    for _ in range(2):
        cache, used = _kv_cache(draft_budget // 2, token_count=draft_tokens)
        draft.append(cache)
        allocated += used

    tail_bytes = max(2, (entry_bytes - allocated) // 2 * 2)
    tail_hidden = guarded_zeros(
        (1, 1, tail_bytes // int(mx.float16.size)), dtype=mx.float16
    )
    allocated += tail_bytes
    sidecar = MTPAPCSidecar(
        (draft, tail_hidden),
        covered_tokens=token_count,
        rng_key=guarded_array([7, 11], dtype=mx.uint32),
        rng_draws=5,
    )
    return target, sidecar, token_count, allocated


def identity() -> APCKey:
    return APCKey(
        "synthetic-hybrid",
        revision="benchmark-source-271b21e",
        adapter="synthetic-hybrid",
        tokenizer_fingerprint="synthetic-tokenizer-v1",
        cache_layout_fingerprint=LAYOUT,
        semantic_fingerprint="text-token-v1",
    )


def make_apc(directory: Path, *, entries: int, total_bytes: int) -> APCv2:
    budget = max(total_bytes * 2, GIB)
    return APCv2(
        max_size=max(entries + 4, 68),
        max_bytes=budget,
        layout_name=LAYOUT,
        idle_disk_seconds=180,
        idle_disk_dir=str(directory),
        idle_disk_max_bytes=budget,
        persist_dir=str(directory),
        persist_identity=identity(),
        persist_semantic_namespace="shared",
        pinned_disk_bytes_per_tenant=budget,
        pinned_disk_bytes_global=budget,
        pinned_resident_bytes_per_tenant=budget,
    )


def token_path(index: int, token_count: int = 256) -> list[int]:
    return [index + 1] + [10_000 + position for position in range(token_count - 1)]


def session_tag(index: int) -> tuple[str, str]:
    return ("benchmark", f"entry-{index:04d}")


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": 0.0, "mean": 0.0, "p50": 0.0, "max": 0.0}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": mean(ordered),
        "p50": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def machine_info(scratch_root: Path) -> dict:
    memory = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    disk = shutil.disk_usage(scratch_root)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "hw_memsize_bytes": int(memory),
        "scratch_filesystem": {
            "path": str(scratch_root),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }


class SpillInstrumentation:
    def __init__(self, apc: APCv2):
        self.payload_write_seconds = 0.0
        self.digest_seconds = 0.0
        self.manifest_seconds = 0.0
        self.digest_calls = 0
        self.manifest_calls = 0
        self.payload_files = 0

        save_cache = apc._atomic_save_cache
        save_arrays = apc._atomic_save_arrays
        payload_record = apc._payload_file_record
        write_manifest = apc._write_manifest_locked

        def timed_save_cache(*args, **kwargs):
            started = time.perf_counter()
            try:
                return save_cache(*args, **kwargs)
            finally:
                self.payload_write_seconds += time.perf_counter() - started
                self.payload_files += 1

        def timed_save_arrays(*args, **kwargs):
            started = time.perf_counter()
            try:
                return save_arrays(*args, **kwargs)
            finally:
                self.payload_write_seconds += time.perf_counter() - started
                self.payload_files += 1

        def timed_payload_record(*args, **kwargs):
            started = time.perf_counter()
            try:
                return payload_record(*args, **kwargs)
            finally:
                self.digest_seconds += time.perf_counter() - started
                self.digest_calls += 1

        def timed_manifest(*args, **kwargs):
            started = time.perf_counter()
            try:
                return write_manifest(*args, **kwargs)
            finally:
                self.manifest_seconds += time.perf_counter() - started
                self.manifest_calls += 1

        apc._atomic_save_cache = timed_save_cache
        apc._atomic_save_arrays = timed_save_arrays
        apc._payload_file_record = timed_payload_record
        apc._write_manifest_locked = timed_manifest


@contextmanager
def forbid_payload_reads():
    original_open = Path.open
    observed = {"calls": 0, "bytes": 0}

    def guarded_open(path, mode="r", *args, **kwargs):
        if path.name.endswith(".safetensors") and "r" in mode:
            observed["calls"] += 1
            observed["bytes"] += path.stat().st_size
            raise AssertionError(f"startup rescan read payload {path.name}")
        return original_open(path, mode, *args, **kwargs)

    with mock.patch.object(Path, "open", guarded_open):
        yield observed


def measured_rescan(directory: Path, *, entries: int, total_bytes: int):
    with forbid_payload_reads() as reads:
        started = time.perf_counter()
        apc = make_apc(directory, entries=entries, total_bytes=total_bytes)
        elapsed = time.perf_counter() - started
    stats = apc.apc_stats["persistence"]["rescan"]
    return apc, {
        "wall_seconds": elapsed,
        "internal_seconds": stats["elapsed_seconds"],
        "entries": stats["registered"],
        "total_payload_bytes": stats["registered_bytes"],
        "payload_read_calls": reads["calls"],
        "payload_read_bytes": reads["bytes"],
        "metadata_only_confirmed": reads["calls"] == 0,
    }


def restore_arm(
    directory: Path,
    *,
    entries: int,
    total_bytes: int,
    verify_sha256: bool,
) -> dict:
    apc = make_apc(directory, entries=entries, total_bytes=total_bytes)
    timings = []
    hashed = {"calls": 0, "bytes": 0, "seconds": 0.0}
    if verify_sha256:
        original_sha = apc._sha256_file

        def timed_sha(path):
            started = time.perf_counter()
            try:
                hashed["calls"] += 1
                hashed["bytes"] += Path(path).stat().st_size
                return original_sha(path)
            finally:
                hashed["seconds"] += time.perf_counter() - started

        apc._sha256_file = timed_sha
    else:
        # Qualification-only counterfactual: production always verifies. This
        # bypass isolates the extra full-file SHA-256 pass from deserialize.
        apc._verify_persisted_files = lambda _disk: None
    try:
        for index in range(entries):
            tokens = token_path(index)
            started = time.perf_counter()
            hit = apc.lookup(identity(), tokens + [999_999])
            elapsed = time.perf_counter() - started
            if not hit.hit or hit.cached_tokens != len(tokens):
                raise AssertionError(f"entry {index} did not restore exactly")
            timings.append(elapsed)
            hit.cache.close()
            apc.park_session(*session_tag(index), ttl_seconds=600)
            mx.clear_cache()
    finally:
        stats = apc.apc_stats["idle_disk"]
        apc.close()
    return {
        "sha256_verification": verify_sha256,
        "timing_seconds": summarize(timings),
        "per_entry_seconds": timings,
        "throughput_gib_per_second": (
            (total_bytes / GIB) / sum(timings) if sum(timings) else 0.0
        ),
        "sha256_calls": hashed["calls"],
        "sha256_bytes": hashed["bytes"],
        "sha256_seconds": hashed["seconds"],
        "restore_failures": stats["restore_failures"],
        "restore_digest_failures": stats["restore_digest_failures"],
    }


def measure_prefetch(directory: Path, *, entries: int, total_bytes: int) -> dict:
    apc = make_apc(directory, entries=entries, total_bytes=total_bytes)
    try:
        before = apc.apc_stats["idle_disk"]["prefetch_restores_ok"]
        started = time.perf_counter()
        accepted = apc.resume_session(*session_tag(0), ttl_seconds=60)
        accepted_at = time.perf_counter()
        serviced = apc.service_pending_prefetch()
        resident_at = time.perf_counter()
        state = apc.session_state(*session_tag(0))
        hit_started = time.perf_counter()
        hit = apc.lookup(
            identity(), token_path(0) + [999_999], session_tag=session_tag(0)
        )
        hit_elapsed = time.perf_counter() - hit_started
        if not serviced or state["state"] != "resident" or not hit.hit:
            raise AssertionError("resume/prefetch did not restore an exact resident hit")
        hit.cache.close()
        after = apc.apc_stats["idle_disk"]
        return {
            "accepted_state": accepted["state"],
            "resume_enqueue_seconds": accepted_at - started,
            "prefetch_service_seconds": resident_at - accepted_at,
            "resume_to_resident_seconds": resident_at - started,
            "prefetched_lookup_seconds": hit_elapsed,
            "prefetch_restores_ok_delta": after["prefetch_restores_ok"] - before,
            "prefetch_hits": after["prefetch_hits"],
        }
    finally:
        apc.close()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True, help="scratch root outside the repository")
    parser.add_argument("--entry-gib", type=float, required=True)
    parser.add_argument("--entries", type=int, required=True)
    parser.add_argument("--max-total-gib", type=float, default=8.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--rescan-only",
        action="store_true",
        help="build/spill and measure startup rescan, skipping restore/prefetch arms",
    )
    return parser


def validate_args(args) -> tuple[Path, int, int]:
    scratch = validate_scratch_root(args.dir)
    if (
        not math.isfinite(args.entry_gib)
        or not math.isfinite(args.max_total_gib)
        or not args.entry_gib > 0
        or not args.max_total_gib > 0
    ):
        raise ValueError("--entry-gib and --max-total-gib must be positive")
    if args.entries < 1:
        raise ValueError("--entries must be positive")
    requested = args.entry_gib * args.entries
    if requested > args.max_total_gib + 1e-12:
        raise ValueError(
            f"requested {requested:g} GiB exceeds --max-total-gib {args.max_total_gib:g}"
        )
    entry_bytes = int(args.entry_gib * GIB)
    if entry_bytes < 1 << 20:
        raise ValueError("--entry-gib must be at least 1 MiB")
    return scratch, entry_bytes, entry_bytes * args.entries


def run(args) -> dict:
    scratch, entry_bytes, requested_total = validate_args(args)
    scratch.mkdir(parents=True, exist_ok=True)
    work = scratch / f"mlx2-apc-persistence-{os.getpid()}-{uuid.uuid4().hex}"
    work.mkdir(mode=0o700)
    apc = None
    try:
        report = {
            "schema": SCHEMA,
            "source": {
                "requested_base_revision": "271b21e",
                "git_head": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=repository_root(),
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            "device": "cpu",
            "arguments": {
                "entry_gib": args.entry_gib,
                "entries": args.entries,
                "max_total_gib": args.max_total_gib,
                "rescan_only": args.rescan_only,
                "scratch_root": str(scratch),
            },
            "guardrails": {
                "minimum_free_memory_bytes": MIN_FREE_BYTES,
                "requested_resident_bytes_per_entry": entry_bytes,
                "requested_total_bytes": requested_total,
                "cleanup_on_exit": True,
            },
            "machine": machine_info(scratch),
        }
        apc = make_apc(work, entries=args.entries, total_bytes=requested_total)
        instrumentation = SpillInstrumentation(apc)
        park_times = []
        park_breakdown = []
        resident_sizes = []
        for index in range(args.entries):
            target, sidecar, token_count, allocated = synthetic_hybrid_entry(entry_bytes)
            resident_sizes.append(allocated)
            tokens = token_path(index, token_count)
            stored = apc.store(
                identity(),
                tokens,
                target,
                sidecar=sidecar,
                session_tag=session_tag(index),
            )
            if not stored.stored:
                raise AssertionError(f"synthetic entry {index} was not stored")
            before_components = (
                instrumentation.payload_write_seconds,
                instrumentation.digest_seconds,
                instrumentation.manifest_seconds,
            )
            original_clear_cache = mx.clear_cache
            clear_cache_seconds = 0.0

            def timed_clear_cache():
                nonlocal clear_cache_seconds
                clear_started = time.perf_counter()
                try:
                    return original_clear_cache()
                finally:
                    clear_cache_seconds += time.perf_counter() - clear_started

            mx.clear_cache = timed_clear_cache
            started = time.perf_counter()
            try:
                parked = apc.park_session(*session_tag(index), ttl_seconds=600)
            finally:
                elapsed = time.perf_counter() - started
                mx.clear_cache = original_clear_cache
            park_times.append(elapsed)
            component_deltas = (
                instrumentation.payload_write_seconds - before_components[0],
                instrumentation.digest_seconds - before_components[1],
                instrumentation.manifest_seconds - before_components[2],
            )
            park_breakdown.append(
                {
                    "total_seconds": elapsed,
                    "payload_write_and_fsync_seconds": component_deltas[0],
                    "digest_seconds": component_deltas[1],
                    "manifest_write_seconds": component_deltas[2],
                    "mlx_clear_cache_seconds": clear_cache_seconds,
                    "other_seconds": max(
                        0.0,
                        elapsed
                        - sum(component_deltas)
                        - clear_cache_seconds,
                    ),
                }
            )
            if parked["state"] != "disk":
                raise AssertionError(f"synthetic entry {index} did not park to disk")
            del target, sidecar
            mx.clear_cache()
        disk_stats = apc.apc_stats["idle_disk"]
        disk_bytes = int(disk_stats["disk_bytes"])
        apc.close()
        apc = None

        expected_payload_files = args.entries * 3
        report["synthetic"] = {
            "topology": {
                "target_kv_planes": 4,
                "target_recurrent_planes": 4,
                "recurrent_arrays_per_plane": 2,
                "mtp_draft_kv_planes": 2,
                "mtp_tail_hidden": True,
            },
            "requested_entry_bytes": entry_bytes,
            "resident_entry_bytes": resident_sizes,
            "persisted_total_bytes": disk_bytes,
        }
        total_park = sum(park_times)
        report["spill"] = {
            "park_latency_seconds": summarize(park_times),
            "per_entry_park_seconds": park_times,
            "per_entry_breakdown": park_breakdown,
            "mlx_clear_cache_seconds": sum(
                row["mlx_clear_cache_seconds"] for row in park_breakdown
            ),
            "total_seconds_including_fsync_digest_manifest": total_park,
            "throughput_gib_per_second": (
                (disk_bytes / GIB) / total_park if total_park else 0.0
            ),
            "payload_write_and_fsync_seconds": instrumentation.payload_write_seconds,
            "digest_seconds": instrumentation.digest_seconds,
            "manifest_write_seconds": instrumentation.manifest_seconds,
            "payload_files": instrumentation.payload_files,
            "payload_digest_calls": instrumentation.digest_calls,
            "manifest_write_calls": instrumentation.manifest_calls,
            "digest_computed_once_per_payload": (
                instrumentation.digest_calls == expected_payload_files
                and instrumentation.payload_files == expected_payload_files
            ),
        }

        rescanned, rescan = measured_rescan(
            work, entries=args.entries, total_bytes=requested_total
        )
        rescanned.close()
        report["startup_rescan"] = rescan
        if rescan["entries"] != args.entries or not rescan["metadata_only_confirmed"]:
            raise AssertionError("startup rescan was incomplete or read payload bytes")

        if not args.rescan_only:
            report["restore_without_sha256"] = restore_arm(
                work,
                entries=args.entries,
                total_bytes=requested_total,
                verify_sha256=False,
            )
            report["restore_with_sha256"] = restore_arm(
                work,
                entries=args.entries,
                total_bytes=requested_total,
                verify_sha256=True,
            )
            report["prefetch"] = measure_prefetch(
                work, entries=args.entries, total_bytes=requested_total
            )
        report["peak_rss_bytes"] = peak_rss_bytes()
        report["completed_at_unix"] = time.time()
        return report
    finally:
        if apc is not None:
            apc.close()
        shutil.rmtree(work, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if args.output:
        atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
