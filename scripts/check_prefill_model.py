#!/usr/bin/env python3
"""Locked Qwen3.8 full-model prefill A/B; writes candidate evidence, not qualification.

Uses identical weights, prompts and forced continuation tokens for every arm.
Measures cold prefill and a warm continuation, plus logit and recurrent-state
drift against ordinary decode. Run only with an explicit model and --gpu.
"""

import argparse
import fcntl
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["MLX_ENABLE_TF32"] = "0"


def thermal():
    return json.loads(
        subprocess.check_output(
            ["swift", str(ROOT / "scripts/thermal_probe.swift")], text=True
        )
    )


def run(args, receipt):
    import mlx.core as mx
    from check_tensorfold_prefill import comparison

    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.runtime.models.cache import make_prompt_cache
    from mlx2.runtime.models.gated_delta import install_prefill_scan
    from mlx2.runtime.models.tensorfold_prefill import install

    mx.set_default_device(mx.gpu)
    adapter = Qwen3827BAdapter(args.model)
    print("Model loaded; installing prefill candidates", flush=True)
    model = adapter.model
    projection = install(model, backend="native")
    scan = install_prefill_scan(model, 8)
    modules = [m for _, m in model.named_modules()]
    sentence = adapter.tokenizer.encode(
        "Explain why causal attention and recurrent state must agree when a cached prompt is resumed. "
    )
    forced = adapter.tokenizer.encode(
        "The answer depends on the state of the preceding tokens."
    )[:8]
    receipt.update(
        {
            "model": Path(args.model).name,
            "artifact": adapter.identity["fingerprint"],
            "device": dict(mx.device_info()),
            "projection": {k: v for k, v in projection.items() if k != "counters"},
            "scan": {k: v for k, v in scan.items() if k != "counters"},
            "cells": [],
        }
    )

    def select(groups, tile):
        for m in modules:
            if hasattr(m, "_prefill_counts"):
                object.__setattr__(m, "_prefill_enabled", groups)
            if hasattr(m, "_prefill_scan_chunk"):
                object.__setattr__(m, "_prefill_scan_chunk", tile)
        projection["counters"].clear()
        scan["counters"].clear()

    def forward(ids, cache):
        text = model.language_model
        hidden = text.model(ids, cache=cache)
        logits = text.lm_head(hidden[:, -1:])
        mx.eval(logits, [c.state for c in cache])
        return logits

    for batch, length, chunk in ((1, 512, 512), (2, 512, 256), (1, 1536, 512)):
        ids = mx.array([(sentence * ((length // len(sentence)) + 1))[:length]] * batch)
        baseline = None
        for name, groups, tile in (
            ("reference", False, 0),
            ("packed", True, 0),
            ("scan8", False, 8),
            ("scan16", False, 16),
            ("packed_scan16", True, 16),
        ):
            if name not in args.arms.split(","):
                continue
            select(groups, tile)
            samples = []
            for rep in range(args.reps + 1):
                cache = make_prompt_cache(model)
                started = time.perf_counter()
                for offset in range(0, length, chunk):
                    logits = forward(ids[:, offset : offset + chunk], cache)
                elapsed = (time.perf_counter() - started) * 1000
                if rep:
                    samples.append(elapsed)
            first = logits
            continuations = []
            warm_started = time.perf_counter()
            for token in forced:
                continuations.append(
                    forward(mx.full((batch, 1), token, dtype=mx.int32), cache)
                )
            warm_ms = (time.perf_counter() - warm_started) * 1000
            recurrent = [
                mx.array(c[1]) for c in cache if c.__class__.__name__ == "ArraysCache"
            ]
            mx.eval(recurrent)
            current = (first, continuations, recurrent)
            if baseline is None:
                baseline = current
            checks = [comparison(mx, first, baseline[0])]
            checks += [comparison(mx, a, b) for a, b in zip(continuations, baseline[1])]
            state_checks = [
                comparison(mx, a, b) for a, b in zip(recurrent, baseline[2])
            ]
            lp = first.astype(mx.float32)
            lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
            lr = baseline[0].astype(mx.float32)
            lr = lr - mx.logsumexp(lr, axis=-1, keepdims=True)
            kl = float(mx.mean(mx.sum(mx.exp(lr) * (lr - lp), axis=-1)).item())
            top1 = bool(
                mx.array_equal(
                    mx.argmax(first, axis=-1), mx.argmax(baseline[0], axis=-1)
                ).item()
            )
            cell = {
                "arm": name,
                "batch": batch,
                "tokens": length,
                "chunk": chunk,
                "prefill_ms": samples,
                "median_prefill_ms": statistics.median(samples),
                "continuation_ms": warm_ms,
                "logits": checks,
                "state": state_checks,
                "next_token_equal": top1,
                "next_token_kl": kl,
                "projection_counters": dict(projection["counters"]),
                "scan_counters": dict(scan["counters"]),
                "thermal": thermal(),
            }
            receipt["cells"].append(cell)
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: cell[k]
                        for k in (
                            "arm",
                            "batch",
                            "tokens",
                            "median_prefill_ms",
                            "next_token_equal",
                            "next_token_kl",
                            "thermal",
                        )
                    }
                ),
                flush=True,
            )
            if cell["thermal"]["thermal_state"] != 0:
                raise RuntimeError("thermal state left nominal; stopped benchmark")
            del cache, current, recurrent, continuations, logits, first
            mx.clear_cache()
    receipt["peak_memory_bytes"] = mx.get_peak_memory()
    candidates = [c for c in receipt["cells"] if c["arm"] != "reference"]
    receipt["bit_exact_gate"] = all(
        check["finite"] and check["bit_equal"]
        for cell in candidates
        for check in cell["logits"] + cell["state"]
    )
    receipt["qualification"] = "unqualified"
    receipt["status"] = "complete-candidate-evidence"
    receipt["timing_limitations"] = (
        "Three repetitions per arm by default, fixed arm order; no serving throughput qualification."
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", action="store_true", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--arms", default="reference,packed,scan8,scan16,packed_scan16")
    args = p.parse_args()
    if args.reps < 1:
        p.error("--reps must be positive")
    arms = set(args.arms.split(","))
    if "reference" not in arms or not arms <= {
        "reference",
        "packed",
        "scan8",
        "scan16",
        "packed_scan16",
    }:
        p.error("--arms must include reference and use known arm names")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "mlx2.prefill-model-ab.v1",
        "status": "running",
        "initial_thermal": thermal(),
    }
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/mlx2").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    receipt["source_sha256"] = digest.hexdigest()
    try:
        with ExitStack() as locks:
            for path in ("/Users/Shared/mlxuag/gpu.lock", "/tmp/gpu.lock"):
                handle = locks.enter_context(open(path, "a"))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if receipt["initial_thermal"]["thermal_state"] != 0:
                raise RuntimeError("GPU must cool to nominal thermal state")
            processes = subprocess.check_output(["ps", "-axo", "command"], text=True)
            if any(
                "-m mlx2.server" in line
                or "tensorfold serve" in line
                or "run_perf.py" in line
                for line in processes.splitlines()
            ):
                raise RuntimeError(
                    "another serving or qualification process is running"
                )
            run(args, receipt)
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
