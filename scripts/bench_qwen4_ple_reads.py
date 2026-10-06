#!/usr/bin/env python3
"""CPU-only serial/parallel PLE row-read viability benchmark.

This measures the exact ``FileBackedShardedEmbedding._pread_rows`` primitive
already used by mlx2.  It does not claim a cold-storage result: unless the
operator supplies independently cloned/evicted files, the OS page cache is an
uncontrolled variable.  Disjoint row sets and first/repeat phases are reported
separately, and every arm must return identical bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np


def _run(table, ids: np.ndarray, workers: int) -> tuple[float, str]:
    started = time.perf_counter_ns()
    rows = table._pread_rows(ids, workers)  # intentional: benchmark current primitive
    elapsed = (time.perf_counter_ns() - started) / 1e6
    return elapsed, hashlib.sha256(rows.tobytes()).hexdigest()


def benchmark(
    sidecar: Path,
    manifest: dict,
    *,
    rows: int,
    reps: int,
    workers: list[int],
    seed: int,
) -> dict:
    from mlx2.runtime.models.qwen4_ple_nvme import FileBackedShardedEmbedding

    total = int(manifest["total_rows"])
    if rows <= 0 or rows * reps > total:
        raise ValueError("rows * reps must fit inside total_rows")
    table = FileBackedShardedEmbedding(
        str(sidecar),
        vocab_size=total,
        dims=int(manifest["dims"]),
        num_shards=int(manifest["num_shards"]),
        data_offset=int(manifest.get("data_offset", 0)),
    )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(total, size=rows * reps, replace=False).reshape(reps, rows)
    result = {
        "schema": "mlx2.qwen4-ple-read-benchmark.v1",
        "sidecar": str(sidecar.resolve()),
        "page_cache_controlled": False,
        "warning": "first-touch is not proof of cold storage without external residency control",
        "rows_per_sample": rows,
        "reps": reps,
        "workers": {},
    }
    try:
        reference: dict[tuple[int, int], str] = {}
        for phase, phase_name in enumerate(("first_pass", "repeat")):
            # The second phase intentionally revisits exactly the first phase's
            # rows. Within a phase ABBA order only balances which arm touches a
            # row first; it does not provide OS-cache control.
            phase_ids = chosen
            for rep in range(reps):
                ids = np.asarray(phase_ids[rep], dtype=np.int64)
                order = workers if rep % 2 == 0 else list(reversed(workers))
                for count in order:
                    ms, digest = _run(table, ids, count)
                    key = (phase, rep)
                    if key in reference and reference[key] != digest:
                        raise AssertionError(
                            "serial and parallel PLE reads returned different bytes"
                        )
                    reference[key] = digest
                    bucket = result["workers"].setdefault(
                        str(count), {"first_pass_ms": [], "repeat_ms": []}
                    )
                    bucket[f"{phase_name}_ms"].append(ms)
        for bucket in result["workers"].values():
            bucket["median_first_pass_ms"] = statistics.median(bucket["first_pass_ms"])
            bucket["median_repeat_ms"] = statistics.median(bucket["repeat_ms"])
    finally:
        table.close()
    if "1" in result["workers"]:
        baseline = result["workers"]["1"]
        for key, bucket in result["workers"].items():
            bucket["first_pass_delta_vs_serial_pct"] = 100 * (
                bucket["median_first_pass_ms"] / baseline["median_first_pass_ms"] - 1
            )
            bucket["repeat_delta_vs_serial_pct"] = 100 * (
                bucket["median_repeat_ms"] / baseline["median_repeat_ms"] - 1
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--seed", type=int, default=687)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest or Path(str(args.sidecar) + ".manifest.json")
    report = benchmark(
        args.sidecar.expanduser().resolve(),
        json.loads(manifest_path.read_text()),
        rows=args.rows,
        reps=args.reps,
        workers=list(dict.fromkeys(args.workers)),
        seed=args.seed,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
