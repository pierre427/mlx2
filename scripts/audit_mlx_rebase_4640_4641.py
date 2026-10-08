#!/usr/bin/env python3
"""Offline reachability audit for the MLX #4640/#4641 rebase.

This script reads source text and safetensors headers only.  It deliberately
does not import MLX, construct a model, or submit Metal work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


DENSE_DTYPES = frozenset({"BF16", "F16", "F32"})
THIN_N_DTYPES = frozenset({"BF16", "F16"})


def _tensor_headers(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        size_bytes = handle.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        header_size = struct.unpack("<Q", size_bytes)[0]
        header = json.loads(handle.read(header_size))
    header.pop("__metadata__", None)
    return header


def read_model_headers(model_dir: Path) -> dict[str, dict[str, Any]]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        shard_names = [path.name for path in sorted(model_dir.glob("*.safetensors"))]
    if not shard_names:
        raise FileNotFoundError(f"no safetensors shards found under {model_dir}")
    tensors: dict[str, dict[str, Any]] = {}
    for shard_name in shard_names:
        tensors.update(_tensor_headers(model_dir / shard_name))
    return tensors


def audit_model(label: str, model_dir: Path) -> dict[str, Any]:
    tensors = read_model_headers(model_dir)
    names = set(tensors)
    dense_2d: list[dict[str, Any]] = []
    quantized_2d: list[dict[str, Any]] = []
    for name, metadata in tensors.items():
        shape = metadata.get("shape", [])
        if not name.endswith(".weight") or len(shape) != 2:
            continue
        base = name[: -len(".weight")]
        dtype = metadata.get("dtype")
        entry = {"name": name, "dtype": dtype, "shape": shape}
        if f"{base}.scales" in names or dtype not in DENSE_DTYPES:
            quantized_2d.append(entry)
        else:
            dense_2d.append(entry)

    thin = [
        entry
        for entry in dense_2d
        if entry["dtype"] in THIN_N_DTYPES and 2 <= int(entry["shape"][0]) <= 64
    ]
    return {
        "label": label,
        "path": str(model_dir),
        "tensor_count": len(tensors),
        "dtype_counts": dict(sorted(Counter(v.get("dtype") for v in tensors.values()).items())),
        "dense_2d_count": len(dense_2d),
        "quantized_2d_count": len(quantized_2d),
        "thin_n_dense_count": len(thin),
        "thin_n_dense": sorted(thin, key=lambda entry: entry["name"]),
        "interpretation": (
            "candidate #4640 artifact surface; runtime use still requires NAX, "
            "single-output-batch, bf16/fp16, non-transposed lhs, N in [2,64], "
            "a row count not intercepted by gemv_wide (normally M >= 16), and an executed matmul"
        ),
    }


def audit_source(mlx_source: Path, invariant_file: Path) -> dict[str, Any]:
    matmul = (mlx_source / "mlx/backend/metal/matmul.cpp").read_text()
    quantized = (mlx_source / "mlx/backend/metal/quantized.cpp").read_text()
    invariant = invariant_file.read_text()
    thin_start = matmul.index("// Case 2: Few output columns")
    thin_end = matmul.index("// Case 3: Large K", thin_start)
    thin_block = matmul[thin_start:thin_end]
    compact_thin = " ".join(thin_block.split())
    checks = {
        "thin_n_kernel_present": "steel_gemm_thin_nax" in thin_block,
        "thin_n_complete_gate": (
            "if (use_nax && batch_size_out == 1 && !transpose_a && N <= 64 && "
            "out.dtype() != float32 && (transpose_b || int64_t(K) * ldb <= INT_MAX))"
            in compact_thin
        ),
        "thin_n_m_boundary": "int sn = M < 2048 ? 1 : 2, ks = 4 / sn" in thin_block,
        "gemv_wide_precedes_thin_n": matmul.index("gemv_wide(", thin_end) < matmul.index("steel_matmul(", thin_end),
        "qmm_splitk_fp32_partials": "array intermediate({split_k, M, N}, float32" in quantized,
        "qvm_splitk_old_rounding": "array intermediate(temp_shape, x.dtype()" in quantized,
        "invariant_lane_uses_two_batches": "LINEAR_BATCHES = 2" in invariant,
        "invariant_dense_stack_has_real_batch_stride": "mx.contiguous(mx.stack([wt, wt]))" in invariant,
    }
    conclusions = None
    if all(checks.values()):
        conclusions = {
            "ordinary_dense_single_batch": "eligible for #4640 only after earlier gemv routes decline",
            "invariant_prefill_dense_lane": "suppresses #4640 because it uses a materialized two-copy stack",
            "qmm_splitk": "numerical baseline moves under #4641; fp32 partials are intentional",
            "qvm_splitk": "unchanged old rounding; do not generalize #4641 expectations to this path",
            "qmv_nax_splitk": "separate input-dtype partial path; receipts/probes must distinguish it from qmm_splitk",
        }
    return {
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "route_conclusions": conclusions,
    }


def audit_build(
    wheel: Path,
    build_dir: Path,
    *,
    revision: str,
    expected_revision: str,
    source_tracked_clean: bool,
) -> dict[str, Any]:
    cache = (build_dir / "CMakeCache.txt").read_text()
    flags_files = list(build_dir.glob("CMakeFiles/mlx.dir/flags.make"))
    build_make = build_dir / "mlx/backend/metal/kernels/CMakeFiles/mlx-metallib.dir/build.make"
    checks = {
        "deployment_target_26_2": bool(
            re.search(r"^CMAKE_OSX_DEPLOYMENT_TARGET(?::[^=]+)?=26\.2$", cache, re.MULTILINE)
        ),
        "mlx_metal_no_nax_absent": bool(flags_files)
        and all("MLX_METAL_NO_NAX" not in path.read_text() for path in flags_files),
        "thin_n_air_built": (build_dir / "mlx/backend/metal/kernels/steel_gemm_thin_nax.air").is_file(),
        "thin_n_air_linked": build_make.is_file()
        and "steel_gemm_thin_nax.air" in build_make.read_text(),
        "revision_matches_expected": revision == expected_revision,
        "source_tracked_clean": source_tracked_clean,
        "wheel_version_contains_revision": revision[:9] in wheel.name,
    }
    return {
        "wheel": str(wheel),
        "bytes": wheel.stat().st_size,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "build_dir": str(build_dir),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def _model_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-source", type=Path, required=True)
    parser.add_argument("--invariant-file", type=Path, required=True)
    parser.add_argument("--model", action="append", type=_model_arg, default=[])
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--expected-revision")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    wheel = args.wheel.resolve() if args.wheel else None
    if wheel and not wheel.is_file():
        parser.error(f"wheel does not exist: {wheel}")
    build_args = (wheel, args.build_dir, args.expected_revision)
    if any(build_args) and not all(build_args):
        parser.error("--wheel, --build-dir, and --expected-revision must be provided together")
    mlx_source = args.mlx_source.resolve()
    revision = subprocess.run(
        ["git", "-C", str(mlx_source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "-C", str(mlx_source), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    build = None
    if wheel:
        build = audit_build(
            wheel,
            args.build_dir.resolve(),
            revision=revision,
            expected_revision=args.expected_revision,
            source_tracked_clean=not tracked_status.strip(),
        )

    result = {
        "schema": "mlx2.mlx-rebase-4640-4641-audit.v1",
        "execution": "offline-source-and-safetensors-headers-only",
        "gpu_used": False,
        "source_identity": {"path": str(mlx_source), "revision": revision},
        "source": audit_source(mlx_source, args.invariant_file.resolve()),
        "models": [audit_model(label, path) for label, path in args.model],
        "build": build,
        "status": {
            "implemented": True,
            "built": bool(build and build["all_checks_pass"]),
            "qualified": False,
            "selected": False,
            "observed_used": False,
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    passed = result["source"]["all_checks_pass"] and (not build or build["all_checks_pass"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
