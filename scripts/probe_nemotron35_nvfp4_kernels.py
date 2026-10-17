#!/usr/bin/env python3
"""Prepare real head matrices on CPU, or probe them after an owned server unit.

All timings describe isolated kernels with synthetic BF16 activations. The
dense baseline decodes published quantized weights; it is not original BF16.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import nemotron35_probe_common as common
import numpy as np


def validate_unit(receipt, *, owner, source_sha, inputs):
    if (receipt.get("schema") != "mlx2.nemotron35-kernel-server-unit.v1"
            or receipt.get("source_sha256") != source_sha
            or receipt.get("kernel_inputs") != inputs
            or receipt.get("ordinary_reference_passed") is not True
            or receipt.get("ordinary_server_passed") is not True
            or receipt.get("server_group_stopped") is not True
            or receipt.get("max_lanes") != 1 or receipt.get("max_inflight") != 1
            or receipt.get("cpg_owner") != owner["cpg_owner"]
            or receipt.get("paired_owner") != owner["paired_gpu_owners"]["shared"]
            or receipt.get("dependencies") != common.runtime_dependencies()
            or type(receipt.get("server_pgid")) is not int or receipt["server_pgid"] < 2):
        raise ValueError("Kernel probe lacks the matching ordinary/server/input/ownership unit")
    if common.group_running(receipt["server_pgid"]):
        raise ValueError("Kernel probe cannot run while the unit server is live")


def run(args):
    from mlx2.conversion.kernel_probe import error_metrics, verify

    owner = common.require_owner(args.cpg_owner_lock)
    source = common.source_identity()
    dependencies = common.runtime_dependencies()
    manifest = verify(args.inputs)
    validate_unit(json.loads(args.unit_receipt.read_text()), owner=owner,
                  source_sha=source["sha256"], inputs=manifest["fingerprint"])
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite kernel evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "mlx2.nemotron35-nvfp4-kernel-probe.v1", "state": "running",
               "inputs": manifest, "source": source, "dependencies": dependencies,
               "owner": owner, "cells": [], "qualified": False,
               "scope": "Isolated real-weight kernels, synthetic activations; no full-model speed or quality claim"}
    previous = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda signum, frame: sys.exit(128 + signum))
        before = common.host_snapshot()
        errors = common.admission_errors(before, minimum_available_bytes=8 * common.GIB, performance=True)
        if errors:
            raise RuntimeError("Kernel admission failed: " + "; ".join(errors))
        receipt["host_before"] = before
        # Import the accelerator only after the live ownership and host gates.
        import mlx.core as mx

        from mlx2.runtime.models.modelopt_nvfp4 import ModelOptNVFP4Linear

        started = time.monotonic()
        for matrix in manifest["matrices"]:
            with np.load(args.inputs / matrix["file"], allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            rows, columns = matrix["shape"]
            native = ModelOptNVFP4Linear(columns, rows)
            native.load_weights([("weight", mx.array(arrays["native_weight"])),
                                 ("scales", mx.array(arrays["native_scales"])),
                                 ("global_scale", mx.array(arrays["global_scale"]))], strict=True)
            dense = mx.array(arrays["dense_fp32"]).astype(mx.bfloat16)
            affine = {bits: (mx.array(arrays[f"affine{bits}_weight"]),
                             mx.array(arrays[f"affine{bits}_scales"]).view(mx.bfloat16),
                             mx.array(arrays[f"affine{bits}_biases"]).view(mx.bfloat16)) for bits in (4, 8)}
            mx.eval(dense, native.parameters(), *[value for group in affine.values() for value in group])

            def affine_call(x, bits, matrices=affine):
                weight, scales, biases = matrices[bits]
                return mx.quantized_matmul(x, weight, scales=scales, biases=biases,
                                           group_size=64, bits=bits, mode="affine", transpose=True)

            functions = {"dense_bf16_decoded": lambda x, weight=dense: x @ weight.T, "native_nvfp4": native,
                         "affine4": lambda x, call=affine_call: call(x, 4),
                         "affine8": lambda x, call=affine_call: call(x, 8)}
            for token_rows in manifest["rows"]:
                common.require_owner(args.cpg_owner_lock)
                pre = common.host_snapshot()
                errors = common.admission_errors(pre, minimum_available_bytes=8 * common.GIB, performance=True)
                if errors or time.monotonic() - started > 180:
                    raise RuntimeError("Kernel cell admission/time bound failed: " + "; ".join(errors))
                rng = np.random.default_rng(manifest["seed"] + token_rows)
                x = mx.array(rng.normal(size=(token_rows, columns)).astype(np.float32)).astype(mx.bfloat16)
                mx.eval(x)
                # Independent NumPy FP32 decode reference, bounded to 8 rows
                # and 64 evenly spaced output channels. No reference is timed.
                selected = np.linspace(0, rows - 1, min(64, rows), dtype=np.int32)
                expected = np.asarray(x.astype(mx.float32))[:8] @ arrays["dense_fp32"][selected].T
                results, timings = {}, {name: [] for name in functions}
                for name, function in functions.items():
                    for _ in range(3):
                        output = function(x)
                        mx.eval(output)
                    results[name] = error_metrics(np.asarray(output.astype(mx.float32))[:8, selected], expected)
                    if not bool(mx.all(mx.isfinite(output)).item()):
                        raise RuntimeError("Kernel output is nonfinite")
                if results["native_nvfp4"]["relative_rmse"] > .01 or results["dense_bf16_decoded"]["relative_rmse"] > .02:
                    raise RuntimeError("Native or decoded BF16 kernel disagrees with the independent reference")
                names = list(functions)
                # Rotate order to reduce systematic hot/cold ordering bias.
                for repetition in range(12):
                    for index in range(len(names)):
                        name = names[(index + repetition) % len(names)]
                        tick = time.perf_counter_ns()
                        output = functions[name](x)
                        mx.eval(output)
                        timings[name].append((time.perf_counter_ns() - tick) * 1e-6)
                post = common.host_snapshot()
                contamination = common.contamination(pre, post)
                contamination.extend(common.admission_errors(post, minimum_available_bytes=8 * common.GIB, performance=True))
                baseline = float(np.median(timings["dense_bf16_decoded"]))
                cell = {"matrix": matrix["prefix"], "shape": matrix["shape"], "token_rows": token_rows,
                        "active_sessions": 1, "rows_are": "Token/verification rows within one conversation",
                        "accuracy_sample": {"rows": min(8, token_rows), "output_channels": selected.tolist()},
                        "modes": {name: {"milliseconds": values, "median_ms": float(np.median(values)),
                                          "speedup_vs_dense_decoded": baseline / float(np.median(values)),
                                          "reference_error": results[name]} for name, values in timings.items()},
                        "host_before": pre, "host_after": post, "contamination": contamination,
                        "performance_reportable": not contamination}
                receipt["cells"].append(cell)
                common.atomic_json(args.out_dir / "kernels.json", receipt)
                if contamination:
                    raise RuntimeError("Kernel cell contaminated; release before retry")
            del arrays, dense, native, affine, affine_call, functions, output, x
            mx.clear_cache()
        receipt["state"] = "completed"
        receipt["peak_active_bytes"] = mx.get_peak_memory()
    except BaseException as error:
        receipt.update(state="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        reasons = []
        try:
            if common.source_identity()["sha256"] != source["sha256"]:
                reasons.append("source changed")
            if common.runtime_dependencies() != dependencies:
                reasons.append("dependencies changed")
            common.require_owner(args.cpg_owner_lock)
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            reasons.append(str(error))
        receipt["finalization_errors"] = reasons
        if reasons:
            for cell in receipt["cells"]:
                cell["performance_reportable"] = False
        common.atomic_json(args.out_dir / "kernels.json", receipt)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, help="Original pinned LiLiCorr directory; CPU only")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--unit-receipt", type=Path)
    parser.add_argument("--cpg-owner-lock", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.inputs or args.unit_receipt or args.cpg_owner_lock:
            parser.error("CPU preparation has no GPU unit options")
        from mlx2.conversion.kernel_probe import prepare
        prepare(args.prepare, args.out_dir, progress=lambda value: print(json.dumps(value), flush=True))
    else:
        if not args.inputs or not args.unit_receipt or not args.cpg_owner_lock:
            parser.error("GPU probe requires inputs, same-window server unit and CPG ownership")
        run(args)


if __name__ == "__main__":
    main()
