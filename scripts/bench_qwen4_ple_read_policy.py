#!/usr/bin/env python3
"""CPU-only dynamic PLE read-policy probe with page-residency evidence.

This reproduces mlx-serve #687's policy question against mlx2's existing
``FileBackedShardedEmbedding._pread_rows`` primitive without flushing global
caches or creating memory pressure:

1. measure serial and parallel readers on disjoint deterministic row sets;
2. record the exact file pages' residency with ``mincore`` before each arm;
3. explicitly read a second pair of row sets, verify their residency, and
   recalibrate in that controlled-resident state; and
4. compare a latched load decision, the merged warm-refresh policy, and the
   documented warm-disabled policy.

The initial state is labelled from measured residency, never called "cold"
unless every sampled page is observed nonresident.  A lack of ``mincore`` is
reported as unknown, not inferred from timing.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import mmap
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CALIBRATION_ROWS = 128
POOL_MARGIN_PERCENT = 20


def prefer_parallel(serial_ns: int, parallel_ns: int) -> bool:
    """mlx-serve #687's overflow-safe >20% win rule."""
    if serial_ns <= 0 or parallel_ns <= 0:
        return False
    return parallel_ns * 100 < serial_ns * (100 - POOL_MARGIN_PERCENT)


def residency_label(fraction: float | None) -> str:
    if fraction is None:
        return "unknown"
    if fraction == 0:
        return "verified_nonresident"
    if fraction == 1:
        return "verified_resident"
    return "mixed_residency"


class PageResidency:
    """Read-only residency queries for selected pages of one file mapping."""

    def __init__(self, path: Path):
        self.path = path
        self.page_size = int(os.sysconf("SC_PAGE_SIZE"))
        self._fd = os.open(path, os.O_RDONLY)
        self._map = None
        self._base = None
        self._mincore = None
        self.error = None
        try:
            size = os.fstat(self._fd).st_size
            if size <= 0:
                raise ValueError("cannot inspect an empty file")
            # ACCESS_COPY makes the mapping buffer-addressable without changing
            # the file. Mapping the 30 GiB sidecar reserves virtual address
            # space only; mincore does not fault pages in.
            self._map = mmap.mmap(self._fd, 0, access=mmap.ACCESS_COPY)
            self._base = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
            fn = ctypes.CDLL(None, use_errno=True).mincore
            fn.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_ubyte),
            ]
            fn.restype = ctypes.c_int
            self._mincore = fn
        except Exception as exc:  # noqa: BLE001 - capability is reported, never assumed
            self.error = f"{type(exc).__name__}: {exc}"
            self.close()

    @property
    def supported(self) -> bool:
        return self._mincore is not None

    def pages_for_rows(
        self, row_ids: np.ndarray, *, data_offset: int, row_bytes: int
    ) -> list[int]:
        pages = set()
        for row_id in np.asarray(row_ids, dtype=np.int64).tolist():
            start = data_offset + int(row_id) * row_bytes
            stop = start + row_bytes - 1
            pages.update(range(start // self.page_size, stop // self.page_size + 1))
        return sorted(pages)

    def fraction(self, pages: list[int]) -> float | None:
        if not self.supported or not pages:
            return None
        resident = 0
        byte = (ctypes.c_ubyte * 1)()
        for page in pages:
            address = self._base + page * self.page_size
            if self._mincore(address, self.page_size, byte) != 0:
                error = ctypes.get_errno()
                self.error = f"mincore errno {error}: {os.strerror(error)}"
                return None
            resident += int(bool(byte[0] & 1))
        return resident / len(pages)

    def close(self) -> None:
        if self._map is not None:
            self._map.close()
            self._map = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


@dataclass
class ArmResult:
    elapsed_ns: int
    rows: int
    pages: int
    resident_fraction_before: float | None
    residency_before: str
    resident_fraction_after: float | None
    residency_after: str
    sha256: str


def _timed_read(
    table, ids: np.ndarray, workers: int, residency: PageResidency
) -> ArmResult:
    pages = residency.pages_for_rows(
        ids, data_offset=table.data_offset, row_bytes=table.row_bytes
    )
    before = residency.fraction(pages)
    started = time.perf_counter_ns()
    values = table._pread_rows(ids, workers)  # benchmark the current mlx2 primitive
    elapsed = time.perf_counter_ns() - started
    after = residency.fraction(pages)
    return ArmResult(
        elapsed_ns=elapsed,
        rows=int(ids.size),
        pages=len(pages),
        resident_fraction_before=before,
        residency_before=residency_label(before),
        resident_fraction_after=after,
        residency_after=residency_label(after),
        sha256=hashlib.sha256(values.tobytes()).hexdigest(),
    )


def _row_sets(total: int, rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if total < 2 * rows:
        raise ValueError("table needs two disjoint calibration row sets")
    rng = np.random.default_rng(seed)
    half = total // 2
    serial = rng.choice(half, size=rows, replace=False)
    parallel = half + rng.choice(total - half, size=rows, replace=False)
    return serial.astype(np.int64), parallel.astype(np.int64)


def _calibrate(
    table, residency: PageResidency, *, rows: int, workers: int, seed: int
) -> dict:
    serial_ids, parallel_ids = _row_sets(table.vocab_size, rows, seed)
    serial = _timed_read(table, serial_ids, 1, residency)
    parallel = _timed_read(table, parallel_ids, workers, residency)
    # Untimed cross-check catches a parallel assembly error without crediting
    # the warmed serial reread to either timing arm.
    serial_check = table._pread_rows(parallel_ids, 1)
    if hashlib.sha256(serial_check.tobytes()).hexdigest() != parallel.sha256:
        raise AssertionError("parallel PLE read differs from serial bytes")
    selected = prefer_parallel(serial.elapsed_ns, parallel.elapsed_ns)
    return {
        "serial": asdict(serial),
        "parallel": asdict(parallel),
        "selected": "parallel" if selected else "serial",
        "prefer_parallel": selected,
    }


def policy_outcomes(load: dict, warmed: dict, *, warming_enabled: bool) -> dict:
    initial = bool(load["prefer_parallel"])
    refreshed = bool(warmed["prefer_parallel"])
    warm_serial = int(warmed["serial"]["elapsed_ns"])
    warm_parallel = int(warmed["parallel"]["elapsed_ns"])

    def selected_ns(parallel: bool) -> int:
        return warm_parallel if parallel else warm_serial

    dynamic = refreshed if warming_enabled else initial
    optimum = min(warm_serial, warm_parallel)
    return {
        "load_selection": "parallel" if initial else "serial",
        "warmed_measurement_selection": "parallel" if refreshed else "serial",
        "latched_bug_effective_after_warm": "parallel" if initial else "serial",
        "dynamic_effective_after_warm": "parallel" if dynamic else "serial",
        "warm_disabled_effective_after_demand_warm": "parallel"
        if initial
        else "serial",
        "refresh_consumed": bool(warming_enabled),
        "latched_bug_manifested": initial != refreshed,
        "latched_warm_regret_pct": 100 * (selected_ns(initial) / optimum - 1),
        "dynamic_warm_regret_pct": 100 * (selected_ns(dynamic) / optimum - 1),
        "warm_disabled_regret_pct": 100 * (selected_ns(initial) / optimum - 1),
        "explicit_serial_effective": "serial",
        "explicit_parallel_effective": "parallel",
    }


def benchmark(
    sidecar: Path,
    manifest: dict,
    *,
    rows: int,
    workers: int,
    reps: int,
    warming_enabled: bool,
    seed: int = 0x6870,
) -> dict:
    from mlx2.runtime.models.qwen4_ple_nvme import FileBackedShardedEmbedding

    table = FileBackedShardedEmbedding(
        str(sidecar),
        vocab_size=int(manifest["total_rows"]),
        dims=int(manifest["dims"]),
        num_shards=int(manifest["num_shards"]),
        data_offset=int(manifest.get("data_offset", 0)),
    )
    load_runs, warm_runs = [], []
    try:
        with PageResidency(sidecar) as residency:
            for rep in range(reps):
                load = _calibrate(
                    table,
                    residency,
                    rows=rows,
                    workers=workers,
                    seed=seed + rep,
                )
                warm_serial, warm_parallel = _row_sets(
                    table.vocab_size, rows, seed * 0x100 + rep
                )
                # Warm only these exact calibration pages. This neither flushes
                # other cache state nor creates broad memory pressure.
                table._pread_rows(np.concatenate((warm_serial, warm_parallel)), 1)
                warm = _calibrate(
                    table,
                    residency,
                    rows=rows,
                    workers=workers,
                    seed=seed * 0x100 + rep,
                )
                load_runs.append(load)
                warm_runs.append(warm)

            load_serial = statistics.median(
                run["serial"]["elapsed_ns"] for run in load_runs
            )
            load_parallel = statistics.median(
                run["parallel"]["elapsed_ns"] for run in load_runs
            )
            warm_serial = statistics.median(
                run["serial"]["elapsed_ns"] for run in warm_runs
            )
            warm_parallel = statistics.median(
                run["parallel"]["elapsed_ns"] for run in warm_runs
            )
            load_summary = {
                "serial": {"elapsed_ns": load_serial},
                "parallel": {"elapsed_ns": load_parallel},
                "prefer_parallel": prefer_parallel(load_serial, load_parallel),
            }
            warm_summary = {
                "serial": {"elapsed_ns": warm_serial},
                "parallel": {"elapsed_ns": warm_parallel},
                "prefer_parallel": prefer_parallel(warm_serial, warm_parallel),
            }
            return {
                "schema": "mlx2.qwen4-ple-read-policy-benchmark.v1",
                "source_candidate": "ddalcu/mlx-serve#687@62a8fb569c3c572077366d87d430bafc768bd0bf",
                "sidecar": str(sidecar.resolve()),
                "rows_per_arm": rows,
                "parallel_workers": workers,
                "reps": reps,
                "seed": seed,
                "residency_probe": {
                    "supported": residency.supported,
                    "error": residency.error,
                    "page_size": residency.page_size,
                },
                "load_runs": load_runs,
                "controlled_resident_runs": warm_runs,
                "median": {"load": load_summary, "controlled_resident": warm_summary},
                "policies": policy_outcomes(
                    load_summary, warm_summary, warming_enabled=warming_enabled
                ),
                "warming_enabled": warming_enabled,
                "claims": {
                    "cold": False,
                    "reason": "No cache eviction was performed; initial state is labelled per arm from mincore.",
                    "sampled_load_pages_all_nonresident": all(
                        arm["resident_fraction_before"] == 0
                        for run in load_runs
                        for arm in (run["serial"], run["parallel"])
                    ),
                    "sampled_load_pages_any_mixed": any(
                        arm["residency_before"] == "mixed_residency"
                        for run in load_runs
                        for arm in (run["serial"], run["parallel"])
                    ),
                    "controlled_resident": all(
                        arm["resident_fraction_before"] == 1
                        for run in warm_runs
                        for arm in (run["serial"], run["parallel"])
                    ),
                    "performance_qualification": False,
                },
            }
    finally:
        table.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rows", type=int, default=CALIBRATION_ROWS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=lambda value: int(value, 0),
        default=0x6870,
        help="base row-set seed (decimal or 0x-prefixed)",
    )
    parser.add_argument("--warming", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    sidecar = args.sidecar.expanduser().resolve()
    manifest_path = args.manifest or Path(str(sidecar) + ".manifest.json")
    report = benchmark(
        sidecar,
        json.loads(manifest_path.read_text()),
        rows=args.rows,
        workers=args.workers,
        reps=args.reps,
        warming_enabled=args.warming == "enabled",
        seed=args.seed,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
