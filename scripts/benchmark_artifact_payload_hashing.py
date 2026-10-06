#!/usr/bin/env python3
"""Measure host-only DFlash2 target/draft payload inspection cost.

The benchmark performs the same complete SHA-256 scans used by the adapters.
It never loads tensors or constructs an MLX device. Results describe startup
identity-check cost only; they are not serving-throughput measurements.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from mlx2.adapters.dflash2 import inspect_drafter
from mlx2.adapters.qwen38_27b import _inspect_target_content, inspect_artifact


def _weight_paths(root: Path) -> list[Path]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        mapping = json.loads(index_path.read_text()).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"invalid weight index: {index_path}")
        names = sorted(set(mapping.values()))
    else:
        names = ["model.safetensors"]
    paths = []
    for name in names:
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not name.endswith(".safetensors")
        ):
            raise ValueError(f"unsafe weight path: {name!r}")
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing weight shard: {path}")
        paths.append(path)
    return paths


def _measure(label: str, byte_count: int, action, iterations: int) -> dict:
    durations = []
    identities = []
    for _ in range(iterations):
        started = time.perf_counter()
        identities.append(action())
        durations.append(time.perf_counter() - started)
    if len(set(identities)) != 1:
        raise ValueError(f"{label} identity changed between benchmark iterations")
    median = statistics.median(durations)
    return {
        "label": label,
        "iterations": iterations,
        "bytes_scanned_per_iteration": byte_count,
        "seconds": durations,
        "median_seconds": median,
        "median_gib_per_second": (byte_count / (1 << 30)) / median,
        "identity": identities[0],
    }


def benchmark(target: Path, draft: Path, iterations: int) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    target = target.expanduser().resolve()
    draft = draft.expanduser().resolve()
    target_paths = _weight_paths(target)
    draft_paths = _weight_paths(draft)

    metadata_started = time.perf_counter()
    metadata_identity = inspect_artifact(target)["identity"]["fingerprint"]
    metadata_seconds = time.perf_counter() - metadata_started

    target_result = _measure(
        "target-full-payload-sha256",
        sum(path.stat().st_size for path in target_paths),
        lambda: _inspect_target_content(target)["revision"],
        iterations,
    )

    def inspect_draft() -> str:
        record = inspect_drafter(draft, target)
        return record["fingerprint"]

    draft_result = _measure(
        "draft-full-payload-sha256",
        sum(path.stat().st_size for path in draft_paths),
        inspect_draft,
        iterations,
    )
    return {
        "schema": "mlx2.payload-hashing-benchmark.v1",
        "host_only": True,
        "clock": "time.perf_counter",
        "target": str(target),
        "draft": str(draft),
        "target_metadata_inspection": {
            "seconds": metadata_seconds,
            "identity": metadata_identity,
        },
        "target_payload": target_result,
        "draft_payload": draft_result,
        "combined_median_seconds": (
            target_result["median_seconds"] + draft_result["median_seconds"]
        ),
        "note": (
            "Host file-identity cost only. Cache warmth is uncontrolled; "
            "this is not route qualification or a serving performance result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args.target, args.draft, args.iterations)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".part")
        temporary.write_text(encoded)
        os.replace(temporary, output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
