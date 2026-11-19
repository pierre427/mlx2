#!/usr/bin/env python3
"""CPU-only I/O probe: cold PLE decode-row reads under five read arms.

Recon 2026-11-13, probe 1 (omlx #5225 MADV_WILLNEED). One decode token of
Flash-Next reads ``rows_per_token`` 100-byte rows of ``ple_rows.bin``, one
in each of that many random shards. This probe times one token's worth of
row reads, page-cache cold, under:

  serial          production ``_pread_rows_fixed``, one worker (the arm the
                  adaptive default selects once the table is warmed)
  serial_advise   the same with the opt-in ``ple_read_advise`` hint
  pooled16        production ``_pread_rows_fixed``, 16 workers ("pooled")
  (a pooled16 + hint arm measured slower and the hint is now applied only
  to one-worker reads; see the README)
  direct_rdadvise F_RDADVISE then pread on the calling thread, no pool hop
                  (reference: the hint without mlx2's I/O-ownership hop)
  mmap_fault  mmap the sidecar and copy each row: one page fault at a time
              (what omlx ``main`` does for gathers of <= 8 rows)
  mmap_will   mmap + madvise(MADV_WILLNEED) on every row before any copy
              (omlx #5225)

"Cold" is checked, not assumed: before each sample, mincore() must report
every page the sample touches as non-resident, otherwise the sample is
skipped and counted. Each sample uses fresh random rows, so no arm warms
another's. The arms are interleaved round-robin. ``purge`` is not needed and
not used. No GPU work.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import fcntl
import json
import mmap
import os
import platform
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PAGE = os.sysconf("SC_PAGE_SIZE")
F_RDADVISE = 44  # <sys/fcntl.h>: struct radvisory { off_t ra_offset; int ra_count; }
MADV_WILLNEED = 3

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                      ctypes.c_int, ctypes.c_long]
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]


def pages_of(offsets, row_bytes):
    out = set()
    for off in offsets:
        out.add(off // PAGE)
        out.add((off + row_bytes - 1) // PAGE)
    return sorted(out)


def resident(base, page):
    vec = ctypes.create_string_buffer(1)
    if libc.mincore(ctypes.c_void_p(base + page * PAGE), PAGE, vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore")
    return bool(vec.raw[0] & 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows-per-token", type=int, default=16)
    ap.add_argument("--samples", type=int, default=200, help="cold samples per arm")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--max-draws", type=int, default=100,
                    help="draws per cold sample before giving up (raise when the sidecar is partly resident)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["MLX_QWEN4_PLE_NVME_READ_ADVISE"] = "1"  # backend only; arms pick advise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from mlx2.adapters.flash_next import artifact_identity
    from mlx2.runtime.models.qwen4_ple_nvme import FileBackedShardedEmbedding, load_manifest

    model = Path(args.model)
    sidecar = model / "ple_rows.bin"
    manifest = load_manifest(str(sidecar))
    row_bytes = manifest["row_bytes"]
    num_shards = manifest["num_shards"]
    rows_per_shard = manifest["rows_per_shard"]
    table = FileBackedShardedEmbedding(
        str(sidecar), manifest["total_rows"], manifest["dims"], num_shards,
        data_offset=manifest["data_offset"],
    )
    assert table.row_bytes == row_bytes
    size = os.path.getsize(sidecar)
    fd = os.open(sidecar, os.O_RDONLY)
    base = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    if base in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), "mmap")
    rng = np.random.default_rng(args.seed)

    def draw():
        shards = rng.choice(num_shards, size=args.rows_per_token, replace=False)
        local = rng.integers(0, rows_per_shard, size=args.rows_per_token)
        ids = np.sort(shards * rows_per_shard + local).astype(np.int64)
        offs = [manifest["data_offset"] + int(i) * row_bytes for i in ids]
        return ids, offs

    def arm_serial(ids, offs):
        return table._pread_rows_fixed(ids, 1, inline=True)

    def arm_pooled(ids, offs):
        return table._pread_rows_fixed(ids, 16)

    def arm_serial_advise(ids, offs):
        return table._pread_rows_fixed(ids, 1, inline=True, advise=True)

    def arm_rdadvise(ids, offs):
        for off in offs:
            fcntl.fcntl(fd, F_RDADVISE, struct.pack("qi4x", off, row_bytes))
        out = np.empty((len(offs), row_bytes), np.uint8)
        for k, off in enumerate(offs):
            out[k] = np.frombuffer(os.pread(fd, row_bytes, off), np.uint8)
        return out

    def _copy(offs):
        out = np.empty((len(offs), row_bytes), np.uint8)
        for k, off in enumerate(offs):
            ctypes.memmove(out[k].ctypes.data, base + off, row_bytes)
        return out

    def arm_mmap_fault(ids, offs):
        return _copy(offs)

    def arm_mmap_will(ids, offs):
        for off in offs:
            start = (off // PAGE) * PAGE
            end = off + row_bytes
            libc.madvise(ctypes.c_void_p(base + start), end - start, MADV_WILLNEED)
        return _copy(offs)

    arms = {"serial": arm_serial, "serial_advise": arm_serial_advise,
            "pooled16": arm_pooled,
            "direct_rdadvise": arm_rdadvise,
            "mmap_fault": arm_mmap_fault, "mmap_will": arm_mmap_will}
    cold = {a: [] for a in arms}
    warm = {a: [] for a in arms}
    skipped = {a: 0 for a in arms}
    mismatches = 0
    # Warm-up: thread pool start, first syscalls.
    ids, offs = draw()
    for fn in arms.values():
        fn(ids, offs)
    while min(len(v) for v in cold.values()) < args.samples:
        for name, fn in arms.items():
            if len(cold[name]) >= args.samples:
                continue
            for _ in range(args.max_draws):
                ids, offs = draw()
                if not any(resident(base, p) for p in pages_of(offs, row_bytes)):
                    break
                skipped[name] += 1
            else:
                raise RuntimeError(f"could not find a cold sample in {args.max_draws} draws")
            t0 = time.perf_counter_ns()
            got = fn(ids, offs)
            t1 = time.perf_counter_ns()
            again = fn(ids, offs)
            t2 = time.perf_counter_ns()
            ref = np.stack([np.frombuffer(os.pread(fd, row_bytes, o), np.uint8) for o in offs])
            if not (np.array_equal(got, ref) and np.array_equal(again, ref)):
                mismatches += 1
            cold[name].append((t1 - t0) / 1e3)
            warm[name].append((t2 - t1) / 1e3)

    def summary(xs):
        xs = sorted(xs)
        q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
        return {"n": len(xs), "median_us": round(statistics.median(xs), 1),
                "mean_us": round(statistics.fmean(xs), 1),
                "p10_us": round(q(0.10), 1), "p90_us": round(q(0.90), 1)}

    root = Path(__file__).resolve().parents[1]
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=root).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts/probe_ple_cold_read_arms.py"],
                      capture_output=True, text=True, cwd=root).stdout.strip():
        rev += " (dirty)"
    result = {
        "probe": "ple_cold_read_arms",
        "checkpoint": f"{model.name}@sha256:{artifact_identity(model)['fingerprint']}",
        "sidecar_sha256": manifest["file_sha256"],
        "source_revision": rev,
        "host": platform.node(), "machine": platform.machine(),
        "page_size": PAGE, "row_bytes": row_bytes, "rows_per_token": args.rows_per_token,
        "sidecar_bytes": size, "seed": args.seed,
        "cold": {a: summary(v) for a, v in cold.items()},
        "warm_repeat": {a: summary(v) for a, v in warm.items()},
        "cold_skipped_resident_draws": skipped,
        "row_mismatches": mismatches,
        "note": "one decode token of rows; mincore-verified cold; arms interleaved; CPU only",
    }
    table.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(out)
    print(json.dumps({"cold": result["cold"], "warm": result["warm_repeat"],
                      "skipped": skipped, "mismatches": mismatches}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
