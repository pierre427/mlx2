#!/usr/bin/env python3
"""Bounded native GGUF kernel fixture unit; GPU requires live external ownership.

Default prepares CPU fixture/oracle checks only. This script does not acquire a
lease or load a full model. The caller owns one bounded CPG/paired-lock window.
All numerical/performance evidence is exploratory, never qualification.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mlx2.runtime.gguf.blocks import decode_blocks
from mlx2.runtime.gguf.reader import FORMATS, GGUFFile, GGUFTensor

FORMATS_TO_TEST = (8, 10, 16)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_dequant(packed, kind, shape):
    """Installed gguf 0.19 reference, independent of mlx2's decoder/kernel."""
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize

    return (
        dequantize(np.ascontiguousarray(packed), GGMLQuantizationType(kind))
        .reshape(shape)
        .astype(np.float32)
    )


def synthetic(shape, kind, seed=20261024):
    block, size = FORMATS[kind][1:]
    if shape[-1] % block:
        raise ValueError("fixture row width must be block aligned")
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, (math.prod(shape) // block, size), dtype=np.uint8)
    d = rng.uniform(0.0005, 0.005, len(raw)).astype("<f2").view(np.uint8).reshape(-1, 2)
    if kind == 10:
        raw[:, 80:82] = d
        raw[:, 82:84] = (
            rng.uniform(0.0003, 0.003, len(raw))
            .astype("<f2")
            .view(np.uint8)
            .reshape(-1, 2)
        )
    else:
        raw[:, :2] = d
    return raw.reshape(*shape[:-1], shape[-1] // block * size)


def compare(actual, expected, *, atol=0.002, rtol=0.002):
    a, b = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    error = np.abs(a - b)
    return {
        "passed": finite and bool(np.allclose(a, b, atol=atol, rtol=rtol)),
        "finite": finite,
        "shape": list(a.shape),
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": float(error.max(initial=0)) if finite else None,
        "rmse": float(np.sqrt(np.mean(error * error))) if finite else None,
        "max_relative_error": float(
            (error / np.maximum(np.abs(b), 1e-8)).max(initial=0)
        )
        if finite
        else None,
    }


def verify_ownership(window, owner_paths):
    """Validate active receipts; never manufacture permission from a CLI flag."""
    lease = json.loads((window / "desktop-lease.json").read_text())
    owners = [json.loads(Path(path).read_text()) for path in owner_paths]
    owner = owners[0]
    if owners[0] != owners[1] or not owner.get("lease_id"):
        raise RuntimeError("matching active paired-lock owner receipts required")
    if lease.get("finished_at") or lease.get("expires", 0) <= time.time():
        raise RuntimeError("desktop CPG lease finished or expired")
    for key in ("cpg_task", "cpg_session", "cpg_generation", "cpg_worker"):
        if not lease.get(key):
            raise RuntimeError(f"missing desktop lease identity {key}")
        if key in owner and owner[key] != lease[key]:
            raise RuntimeError(f"paired-lock/CPG identity mismatch: {key}")
    if owner.get("session") != lease["cpg_session"]:
        raise RuntimeError("paired-lock session does not match desktop CPG lease")
    os.kill(int(owner["pid"]), 0)
    os.kill(int(lease["pid"]), 0)
    return {"window": str(window), "desktop_lease": lease, "paired_owner": owner}


def source_receipt():
    import gguf.quants

    revision = os.environ.get("MLX2_SOURCE_REVISION")
    snapshot = os.environ.get("MLX2_SOURCE_SNAPSHOT_SHA256")
    if revision or snapshot or not (ROOT / ".git").exists():
        if not revision or not snapshot:
            raise RuntimeError(
                "frozen source requires both MLX2_SOURCE_REVISION and MLX2_SOURCE_SNAPSHOT_SHA256"
            )
        descriptor_path = Path(
            os.environ.get("MLX2_SOURCE_MANIFEST", str(ROOT / "desktop-source.json"))
        )
        descriptor = json.loads(descriptor_path.read_text())
        expected_revision = descriptor.get("source_revision")
        expected_hash = descriptor.get(
            "source_sha256", descriptor.get("source_snapshot_sha256")
        )
        if expected_revision != revision or expected_hash != snapshot:
            raise RuntimeError(
                "environment identities do not match frozen source descriptor"
            )
        from mlx2.desktop_runner import digest, source_manifest

        actual_manifest = source_manifest(ROOT)
        actual_hash = digest(actual_manifest)
        if actual_hash != snapshot:
            raise RuntimeError(
                f"frozen source manifest mismatch: expected {snapshot}, actual {actual_hash}"
            )
        if "files" in descriptor and descriptor["files"] != actual_manifest:
            raise RuntimeError("frozen per-file manifest differs from source bytes")
        source = {
            "commit": revision,
            "source_snapshot_sha256": snapshot,
            "source_descriptor": str(descriptor_path),
            "source_descriptor_sha256": sha(descriptor_path),
            "verified_source_files": len(actual_manifest),
            "mode": "verified_frozen_source",
        }
    else:
        source = {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "uncommitted_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
            ).hexdigest(),
            "mode": "git_worktree",
        }
    source.update(
        {
            "files": {
                str(p.relative_to(ROOT)): sha(p)
                for p in sorted((ROOT / "src/mlx2/runtime/gguf").glob("*.py"))
            },
            "probe_sha256": sha(__file__),
            "oracle": {
                "package": "gguf",
                "version": importlib.metadata.version("gguf"),
                "quants_source_sha256": sha(gguf.quants.__file__),
            },
        }
    )
    return source


def atomic(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def measure(fn, repetitions):
    import mlx.core as mx

    start = time.perf_counter()
    value = fn()
    mx.eval(value)
    first = time.perf_counter() - start
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        value = fn()
        mx.eval(value)
        times.append(time.perf_counter() - start)
    return np.array(value), {
        "first_evaluation_seconds": first,
        "first_evaluation_scope": "includes lazy compile/evaluation; persistent compiler cache not proven cold",
        "warm_seconds": times,
        "warm_median_seconds": statistics.median(times),
    }


def upload(packed, kind, shape, name):
    import mlx.core as mx

    from mlx2.runtime.gguf.packed_metal import PackedMatrix

    tensor = GGUFTensor(name, tuple(shape[::-1]), kind, 0, packed.nbytes)
    return PackedMatrix(mx.array(packed.reshape(-1)), tensor)


def dense_case(
    raw,
    kind,
    shape,
    batch,
    *,
    gpu,
    repetitions,
    tiled=False,
    cooperative_rows=(),
    nax=False,
):
    n, k = shape
    rng = np.random.default_rng(17)
    x = rng.normal(0, 0.1, (batch, k)).astype(np.float32)
    ref_w = independent_dequant(raw, kind, shape)
    expected = x @ ref_w.T
    format_check = compare(
        decode_blocks(raw.tobytes(), kind, max_elements=n * k).reshape(shape),
        ref_w,
        atol=0,
        rtol=0,
    )
    result = {
        "kind": "dense",
        "format": FORMATS[kind][0],
        "shape": list(shape),
        "batch": batch,
        "independent_block_decoder_check": format_check,
        "route": "packed_simd_fp32",
        "dense_reference_peak_bytes": ref_w.nbytes,
        "packed_bytes": raw.nbytes,
    }
    if not format_check["passed"]:
        raise RuntimeError(f"independent format decoder mismatch {result}")
    if gpu:
        import mlx.core as mx

        weight = upload(raw, kind, shape, "fixture.dense")
        xm = mx.array(x)
        actual, timing = measure(lambda: weight(xm), repetitions)
        result.update(numerical=compare(actual, expected), timing=timing)
        if kind in (8, 16) and cooperative_rows:
            from mlx2.runtime.gguf.cooperative_metal import cooperative_dense

            result["cooperative_candidates"] = []
            for rows in cooperative_rows:
                actual, candidate_timing = measure(
                    lambda rows=rows: cooperative_dense(xm, weight, rows=rows),
                    repetitions,
                )
                result["cooperative_candidates"].append(
                    {
                        "route": "packed_block_reuse_fp32",
                        "rows_per_simdgroup": rows,
                        "selected": False,
                        "reduction_order_changed": True,
                        "numerical": compare(actual, expected),
                        "timing": candidate_timing,
                    }
                )
        if tiled or nax:
            from mlx2.runtime.gguf.prefill_metal import packed_prefill

            tile_ref = (
                x.astype(np.float16).astype(np.float32)
                @ ref_w.astype(np.float16).astype(np.float32).T
            )
            candidates = []
            if tiled:
                candidates.append(("tile_candidate", packed_prefill))
            if nax:
                from mlx2.runtime.gguf.nax_metal import nax_prefill

                candidates.append(("nax_candidate", nax_prefill))
            for name, function in candidates:
                actual, timing = measure(
                    lambda function=function: function(xm, weight), repetitions
                )
                result[name] = {
                    "route": "packed_simdgroup_fp16_tiles",
                    "selected": False,
                    "different_arithmetic": True,
                    "numerical_vs_fp16_operand_reference": compare(actual, tile_ref),
                    "numerical_vs_fp32_ordinary_reference": compare(
                        actual, expected, atol=0.02, rtol=0.02
                    ),
                    "timing": timing,
                }
                if name == "nax_candidate":
                    result[name]["route"] = "packed_mpp_nax_fp16_tiles"
    return result


def grouped_case(k, n, *, gpu, repetitions, cooperative_rows=()):
    from mlx2.runtime.gguf.packed_metal import grouped_linear

    groups = 8
    shape = (groups, n, k)
    raw = synthetic(shape, 8, seed=3)
    x = np.random.default_rng(4).normal(0, 0.1, (2, groups, k)).astype(np.float32)
    expected = np.empty((2, groups, n), np.float32)
    for group in range(groups):
        expected[:, group] = x[:, group] @ independent_dequant(raw[group], 8, (n, k)).T
    result = {
        "kind": "grouped_wo_a",
        "shape": list(shape),
        "batch": 2,
        "route": "packed_simd_fp32",
        "packed_bytes": raw.nbytes,
    }
    if gpu:
        import mlx.core as mx

        weight = upload(raw, 8, shape, "fixture.grouped")
        xm = mx.array(x)
        actual, timing = measure(lambda: grouped_linear(xm, weight), repetitions)
        result.update(numerical=compare(actual, expected), timing=timing)
        if cooperative_rows:
            from mlx2.runtime.gguf.cooperative_metal import cooperative_grouped

            result["cooperative_candidates"] = []
            for rows in cooperative_rows:
                actual, timing = measure(
                    lambda rows=rows: cooperative_grouped(xm, weight, rows=rows),
                    repetitions,
                )
                result["cooperative_candidates"].append(
                    {
                        "rows": rows,
                        "numerical": compare(actual, expected),
                        "timing": timing,
                    }
                )
    return result


def moe_case(k, m, batch, *, gpu, repetitions, cooperative_rows=()):
    from mlx2.runtime.gguf.packed_metal import packed_moe

    experts = 6
    gate, up = (
        synthetic((experts, m, k), 16, seed=5),
        synthetic((experts, m, k), 16, seed=6),
    )
    down = synthetic((experts, k, m), 10, seed=7)
    rng = np.random.default_rng(8)
    x = rng.normal(0, 0.1, (batch, k)).astype(np.float32)
    ids = np.stack([rng.permutation(experts) for _ in range(batch)]).astype(np.uint32)
    scores = rng.random((batch, experts), dtype=np.float32)
    scores = 1.5 * scores / scores.sum(axis=-1, keepdims=True)
    expected = np.zeros((batch, k), np.float32)
    # One expert decoded at a time: no giant 256-expert floating reference.
    for expert in range(experts):
        g = x @ independent_dequant(gate[expert], 16, (m, k)).T
        u = x @ independent_dequant(up[expert], 16, (m, k)).T
        g = np.minimum(g, 10)
        u = np.clip(u, -10, 10)
        mid = (g / (1 + np.exp(-g))) * u
        y = mid @ independent_dequant(down[expert], 10, (k, m)).T
        slots = np.argmax(ids == expert, axis=-1)
        expected += scores[np.arange(batch), slots, None] * y
    result = {
        "kind": "top6_fused_moe",
        "input": k,
        "hidden": m,
        "batch": batch,
        "experts_allocated": 6,
        "routed_topk": 6,
        "route": "packed_fused_clipped_gate_up_down_fp32",
        "packed_bytes": gate.nbytes + up.nbytes + down.nbytes,
        "reference": "independent gguf dequant, float32 products, clipped gate/up, unbiased caller scores",
    }
    if gpu:
        import mlx.core as mx

        gm = upload(gate, 16, (experts, m, k), "gate")
        um = upload(up, 16, (experts, m, k), "up")
        dm = upload(down, 10, (experts, k, m), "down")
        xm, im, sm = mx.array(x), mx.array(ids), mx.array(scores)
        actual, timing = measure(
            lambda: packed_moe(xm, im, sm, gm, um, dm), repetitions
        )
        result.update(
            numerical=compare(actual, expected, atol=0.01, rtol=0.005), timing=timing
        )
        if cooperative_rows:
            from mlx2.runtime.gguf.cooperative_metal import cooperative_moe

            result["cooperative_candidates"] = []
            for rows in cooperative_rows:
                actual, candidate_timing = measure(
                    lambda rows=rows: cooperative_moe(
                        xm, im, sm, gm, um, dm, rows=rows
                    ),
                    repetitions,
                )
                result["cooperative_candidates"].append(
                    {
                        "route": "packed_block_reuse_parallel_down_fp32",
                        "rows_per_simdgroup": rows,
                        "selected": False,
                        "reduction_order_changed": True,
                        "numerical": compare(actual, expected, atol=0.01, rtol=0.005),
                        "timing": candidate_timing,
                    }
                )
    return result


def actual_cases(path, *, allow_partial=False):
    if path.name.endswith(".partial") and not allow_partial:
        raise RuntimeError(
            "partial download refused; use --allow-unverified-partial explicitly for block probes only"
        )
    with GGUFFile(path) as f:
        for kind in FORMATS_TO_TEST:
            tensor = next(
                t
                for t in f.tensors.values()
                if t.ggml_type == kind and len(t.shape) >= 2
            )
            k = tensor.shape[0]
            n = min(128, tensor.shape[1])
            rowbytes = k // FORMATS[kind][1] * FORMATS[kind][2]
            view = f.packed_view(tensor.name)
            try:
                raw = (
                    np.frombuffer(view[: n * rowbytes], dtype=np.uint8)
                    .copy()
                    .reshape(n, rowbytes)
                )
            finally:
                view.release()
                f.release_mapping()
            yield (
                raw,
                kind,
                (n, k),
                {
                    "path": str(path),
                    "tensor": tensor.name,
                    "expert": 0 if len(tensor.shape) == 3 else None,
                    "rows": [0, n],
                    "payload_verified_by_this_probe": False,
                    "partial_payload": path.name.endswith(".partial"),
                    "byte_offset": f.data_offset + tensor.offset,
                    "slice_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                },
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("--ownership-window", type=Path)
    parser.add_argument("--owner-path", type=Path, action="append")
    parser.add_argument(
        "--full-shapes",
        action="store_true",
        help="K4096/N2048, grouped8x1024x4096 and six full experts",
    )
    parser.add_argument("--tiled-candidate", action="store_true")
    parser.add_argument("--nax-candidate", action="store_true")
    parser.add_argument("--dense-batches", type=int, nargs="+", default=[1, 32])
    parser.add_argument("--cooperative-candidate", action="store_true")
    parser.add_argument(
        "--cooperative-rows", type=int, nargs="+", choices=(1, 2, 4), default=[4]
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-working-bytes", type=int, default=1 << 30)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--artifact-fingerprint", default="synthetic-fixtures-v1")
    parser.add_argument("--allow-unverified-partial", action="store_true")
    args = parser.parse_args(argv)
    if any(not 1 <= value <= 512 for value in args.dense_batches):
        parser.error("dense fixture batches must be in 1..512")
    if not 0 < args.max_working_bytes < 2 << 30 or not 1 <= args.repetitions <= 100:
        parser.error(
            "working-memory cap must be positive and below 2GiB; repetitions 1..100"
        )
    if args.output.exists():
        parser.error("refusing existing evidence directory")
    ownership = None
    owner_paths = args.owner_path or [
        Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
        Path("/tmp/gpu.lock/owner.json"),
    ]
    if args.run_gpu:
        if args.ownership_window is None or len(owner_paths) != 2:
            parser.error(
                "GPU execution requires ownership window and two matching owner receipts"
            )
        ownership = verify_ownership(args.ownership_window, owner_paths)
    k, n, m = (4096, 2048, 2048) if args.full_shapes else (256, 32, 256)
    # Upper bound includes oracle temporaries, six packed expert projections,
    # device copies, matmul reference, and allocator/compiler allowance.
    estimated = 512 * (1 << 20) + 6 * k * m * 4 + 8 * 1024 * k * 4
    if estimated > args.max_working_bytes:
        parser.error(f"fixture conservative bound {estimated} exceeds memory cap")
    args.output.mkdir(parents=True)
    receipt = {
        "status": "running",
        "qualification": False,
        "performance_claim": False,
        "source": source_receipt(),
        "artifact_fingerprint": args.artifact_fingerprint,
        "ownership": ownership,
        "run_gpu": args.run_gpu,
        "synthetic_recipe": "v2_nonzero_q2_minima_finite_random_scales",
        "working_memory_cap_bytes": args.max_working_bytes,
        "conservative_working_memory_bound_bytes": estimated,
        "cases": [],
        "timing_scope": "observational kernel fixtures; no thermally controlled model throughput claim",
    }
    atomic(args.output / "results.json", receipt)
    if args.run_gpu:
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(args.max_working_bytes)
        mx.set_cache_limit(64 << 20)
        receipt["device_info"] = mx.metal.device_info()

    def record(case):
        if args.run_gpu:
            verify_ownership(args.ownership_window, owner_paths)
        if args.run_gpu:
            case["allocator_peak_bytes"] = mx.get_peak_memory()
            case["allocator_active_bytes"] = mx.get_active_memory()
        receipt["cases"].append(case)
        atomic(args.output / "results.json", receipt)
        print(
            json.dumps(
                {
                    "case": case["kind"],
                    "batch": case.get("batch"),
                    "format": case.get("format"),
                    "numerical": case.get("numerical"),
                    "timing": case.get("timing"),
                }
            ),
            flush=True,
        )
        gc.collect()
        if args.run_gpu:
            mx.clear_cache()

    try:
        for kind in FORMATS_TO_TEST:
            raw = synthetic((n, k), kind)
            for batch in args.dense_batches:
                record(
                    dense_case(
                        raw,
                        kind,
                        (n, k),
                        batch,
                        gpu=args.run_gpu,
                        repetitions=args.repetitions,
                        tiled=args.tiled_candidate,
                        nax=args.nax_candidate,
                        cooperative_rows=args.cooperative_rows
                        if args.cooperative_candidate
                        else (),
                    )
                )
        record(
            grouped_case(
                k,
                1024 if args.full_shapes else 32,
                gpu=args.run_gpu,
                repetitions=args.repetitions,
                cooperative_rows=args.cooperative_rows
                if args.cooperative_candidate
                else (),
            )
        )
        for batch in (1, 2):
            record(
                moe_case(
                    k,
                    m,
                    batch,
                    gpu=args.run_gpu,
                    repetitions=args.repetitions,
                    cooperative_rows=args.cooperative_rows
                    if args.cooperative_candidate
                    else (),
                )
            )
        if args.gguf:
            for raw, kind, shape, identity in actual_cases(
                args.gguf, allow_partial=args.allow_unverified_partial
            ):
                case = dense_case(
                    raw, kind, shape, 1, gpu=args.run_gpu, repetitions=args.repetitions
                )
                case["actual_tensor_slice"] = identity
                record(case)
        receipt["status"] = "completed"
        receipt["all_numerical_checks_passed"] = all(
            c.get("numerical", {}).get("passed", True)
            and c.get("independent_block_decoder_check", {}).get("passed", True)
            and all(
                v["numerical"]["passed"] for v in c.get("cooperative_candidates", [])
            )
            and c.get("tile_candidate", {})
            .get("numerical_vs_fp16_operand_reference", {})
            .get("passed", True)
            and c.get("nax_candidate", {})
            .get("numerical_vs_fp16_operand_reference", {})
            .get("passed", True)
            for c in receipt["cases"]
        )
    except BaseException as exc:
        receipt.update(
            status="failed", error={"type": type(exc).__name__, "message": str(exc)}
        )
        raise
    finally:
        atomic(args.output / "results.json", receipt)
    return 0 if receipt["all_numerical_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
