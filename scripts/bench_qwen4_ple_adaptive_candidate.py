#!/usr/bin/env python3
"""CPU-only request-level probe for the opt-in adaptive Qwen4 PLE reader.

The probe uses an existing ``ple_rows.bin`` and never flushes system caches,
creates memory pressure, starts the whole-table warmer, or constructs Metal
work.  It records measured page residency, exact route receipts, serial-byte
parity and the warm-transition latch.  ``--run`` is required for file reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bench_qwen4_ple_read_policy import PageResidency, residency_label


@contextmanager
def _environment(**changes):
    previous = {key: os.environ.get(key) for key in changes}
    try:
        for key, value in changes.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _request(table, ids: np.ndarray, residency: PageResidency) -> dict:
    pages = residency.pages_for_rows(
        ids, data_offset=table.data_offset, row_bytes=table.row_bytes
    )
    before = residency.fraction(pages)
    workers = table._workers_for(int(ids.size))
    started = time.perf_counter_ns()
    candidate = table._pread_rows(ids, workers)
    elapsed_ns = time.perf_counter_ns() - started
    reference = table._pread_rows_fixed(ids, 1, inline=True)
    exact = bool(np.array_equal(candidate, reference))
    if not exact:
        raise AssertionError("adaptive PLE read bytes differ from serial reference")
    after = residency.fraction(pages)
    return {
        "rows": int(ids.size),
        "elapsed_ns": elapsed_ns,
        "residency_before": residency_label(before),
        "resident_fraction_before": before,
        "residency_after": residency_label(after),
        "resident_fraction_after": after,
        "sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
        "serial_exact": exact,
        "route_receipt": table.read_policy_status["last_receipt"],
    }


def run_probe(sidecar: Path, manifest: dict, *, rows: int, seed: int) -> dict:
    from mlx2.runtime.models.qwen4_ple_nvme import (
        PREFILL_ID_THRESHOLD,
        FileBackedShardedEmbedding,
    )

    if rows < PREFILL_ID_THRESHOLD:
        raise ValueError(
            f"rows must be at least {PREFILL_ID_THRESHOLD} to exercise warm refresh"
        )
    total = int(manifest["total_rows"])
    if rows > total:
        raise ValueError("rows exceeds the sidecar vocabulary")
    rng = np.random.default_rng(seed)
    request_ids = rng.choice(total, size=rows, replace=False).astype(np.int64)

    with _environment(
        MLX_QWEN4_PLE_NVME_READ_POLICY="adaptive",
        MLX_QWEN4_PLE_NVME_ADAPTIVE_WARM="1",
    ):
        table = FileBackedShardedEmbedding(
            str(sidecar),
            vocab_size=total,
            dims=int(manifest["dims"]),
            num_shards=int(manifest["num_shards"]),
            data_offset=int(manifest.get("data_offset", 0)),
        )
        try:
            with PageResidency(sidecar) as residency:
                load_status = table.read_policy_status
                before_refresh = _request(table, request_ids, residency)

                # Warm only the deterministic pages used by the next policy
                # calibration. This is a controlled transition probe, not a
                # claim that the complete table is resident.
                serial_ids, pooled_ids = table._calibration_row_ids("warmed")
                calibration_ids = np.concatenate((serial_ids, pooled_ids))
                calibration_pages = residency.pages_for_rows(
                    calibration_ids,
                    data_offset=table.data_offset,
                    row_bytes=table.row_bytes,
                )
                table._pread_rows_fixed(calibration_ids, 1)
                controlled_fraction = residency.fraction(calibration_pages)
                if not table.notify_adaptive_warm_complete():
                    raise RuntimeError("adaptive warm-completion signal was refused")
                after_refresh = _request(table, request_ids, residency)
                final_status = table.read_policy_status
                return {
                    "schema": "mlx2.qwen4-ple-adaptive-candidate-benchmark.v1",
                    "source_candidate": (
                        "ddalcu/mlx-serve#687@"
                        "62a8fb569c3c572077366d87d430bafc768bd0bf"
                    ),
                    "sidecar": str(sidecar.resolve()),
                    "sidecar_stat": {
                        "size": sidecar.stat().st_size,
                        "mtime_ns": sidecar.stat().st_mtime_ns,
                    },
                    "source_index_sha256": manifest.get("source_index_sha256"),
                    "rows": rows,
                    "seed": seed,
                    "load_status": load_status,
                    "before_refresh": before_refresh,
                    "controlled_calibration_pages": {
                        "pages": len(calibration_pages),
                        "resident_fraction": controlled_fraction,
                        "label": residency_label(controlled_fraction),
                    },
                    "after_refresh": after_refresh,
                    "final_status": final_status,
                    "claims": {
                        "whole_table_warmed": False,
                        "cold": False,
                        "request_performance_qualification": False,
                        "route_and_byte_parity_only": True,
                    },
                }
        finally:
            table.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x6871)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    sidecar = args.sidecar.expanduser().resolve()
    manifest_path = args.manifest or Path(str(sidecar) + ".manifest.json")
    if args.run:
        report = run_probe(
            sidecar,
            json.loads(manifest_path.read_text()),
            rows=args.rows,
            seed=args.seed,
        )
    else:
        report = {
            "schema": "mlx2.qwen4-ple-adaptive-candidate-plan.v1",
            "sidecar": str(sidecar),
            "manifest": str(manifest_path),
            "rows": args.rows,
            "seed": args.seed,
            "will_run_gpu": False,
            "will_start_whole_table_warmer": False,
            "command": "rerun with --run to perform bounded existing-sidecar reads",
        }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
