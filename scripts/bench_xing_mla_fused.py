#!/usr/bin/env python3
"""Source-bound Xing MLA model-path A/B under the shared GPU lease.

Run this only inside the CPG GPU job wrapper with both lock receipts present.
The candidate is compared with ordinary attention at the same cache state;
each measured step is rolled back before the next arm. This is a development
gate, not a qualification or an HTTP serving benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST_LOCK = Path("/Users/Shared/mlxuag/gpu.lock/owner.json")
FCNTL_LOCK = Path("/tmp/gpu.lock")


def require_locks() -> list[dict]:
    if not HOST_LOCK.is_file():
        raise SystemExit(f"GPU benchmark requires ownership receipt {HOST_LOCK}")
    if not FCNTL_LOCK.is_file() or os.environ.get("MLX2_GPU_FCNTL_LOCKED") != "1":
        raise SystemExit("GPU benchmark requires the inherited /tmp/gpu.lock fcntl wrapper")
    owner = json.loads(HOST_LOCK.read_text())
    if not isinstance(owner, dict) or not owner.get("pid") or not owner.get("cpg_generation"):
        raise SystemExit(f"GPU ownership receipt is incomplete: {HOST_LOCK}")
    return [owner]


def trim_to(cache, length: int) -> None:
    for layer in cache:
        current = layer.offset
        if current < length or layer.trim(current - length) != current - length:
            raise RuntimeError("Xing MLA benchmark could not restore cache boundary")


def host_signals() -> dict[str, str]:
    result = {}
    for key, command in {
        "swapusage": ["sysctl", "vm.swapusage"],
        "vm_stat": ["vm_stat"],
        "thermal": ["pmset", "-g", "therm"],
    }.items():
        try:
            result[key] = subprocess.check_output(command, text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError) as error:
            result[key] = f"unavailable: {type(error).__name__}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.batch <= 16 and 1 <= args.width <= 4 and 1 <= args.context <= 131072):
        parser.error("batch, width, or context is outside the bounded MLA screen")
    # Forty Xing layers keep one BF16 512-wide latent and 64-wide RoPE key.
    # Bound persistent cache storage before loading the model; this leaves
    # room for weights, Metal scratch, and the pinned service on the M5 Max.
    estimated_cache_bytes = args.batch * args.context * 40 * (512 + 64) * 2
    if estimated_cache_bytes > 24 * 1024**3:
        parser.error("estimated exact MLA cache exceeds the 24 GiB probe limit")
    if args.repeats < 3:
        parser.error("at least three repeats are required")
    owners = require_locks()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise SystemExit("commit candidate source before the fused MLA GPU probe")
    host_before = host_signals()

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    from mlx2.adapters.xing import XingAdapter
    from mlx2.runtime.models import xing4_0

    started = time.time()
    adapter = XingAdapter(str(args.model), execution_policy={"num_draft": 1})
    model = adapter.model
    mx.eval(model.parameters())
    xing4_0.set_fused_mla(False)
    vocab = int(model.args.vocab_size)
    tokens = mx.random.randint(100, vocab, (args.batch, args.context + args.width), dtype=mx.int32)
    mx.eval(tokens)
    cache = model.make_cache()
    for start in range(0, args.context, 512):
        stop = min(start + 512, args.context)
        mx.eval(model(tokens[:, start:stop], cache=cache))
    base = cache[0].offset
    if base != args.context or any(c.offset != base for c in cache):
        raise RuntimeError("prefill produced inconsistent Xing cache offsets")
    query = tokens[:, args.context:]

    def run(candidate: bool):
        trim_to(cache, base)
        xing4_0.set_fused_mla(candidate)
        before = mx.get_active_memory()
        mx.reset_peak_memory()
        tick = time.perf_counter_ns()
        logits = model(query, cache=cache)
        mx.eval(logits)
        elapsed = (time.perf_counter_ns() - tick) / 1e6
        peak = max(0, mx.get_peak_memory() - before)
        return logits, elapsed, peak

    # Gate numerical agreement before timing. The candidate counter proves
    # that this comparison did not silently measure the reference twice.
    ordinary, _, _ = run(False)
    candidate, _, _ = run(True)
    max_abs = float(mx.max(mx.abs(ordinary.astype(mx.float32) - candidate.astype(mx.float32))).item())
    same_argmax = bool(mx.all(mx.argmax(ordinary, axis=-1) == mx.argmax(candidate, axis=-1)).item())
    engagement = xing4_0.fused_mla_stats()
    if engagement.get("fused_calls", 0) <= 0:
        raise RuntimeError("the fused MLA candidate was not observed used")
    if not same_argmax or max_abs > 0.5:
        failure = {
            "schema": "mlx2.xing-mla-fused-model-gate.v1",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "model": str(args.model.resolve()),
            "model_config_sha256": hashlib.sha256(
                (args.model / "config.json").read_bytes()
            ).hexdigest(),
            "lock_owners": owners,
            "context": args.context,
            "batch": args.batch,
            "width": args.width,
            "max_abs_logits": max_abs,
            "same_argmax": same_argmax,
            "mechanism": engagement,
            "numerical_gate_passed": False,
            "host_before": host_before,
            "host_after": host_signals(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(f"fused MLA numerical screen failed: max_abs={max_abs}, same_argmax={same_argmax}")

    samples = {"ordinary": [], "fused": []}
    peaks = {"ordinary": [], "fused": []}
    for repeat in range(args.repeats):
        order = (False, True) if repeat % 2 == 0 else (True, False)
        for fused in order:
            _, elapsed, peak = run(fused)
            key = "fused" if fused else "ordinary"
            samples[key].append(elapsed)
            peaks[key].append(peak)
    trim_to(cache, base)
    xing4_0.set_fused_mla(False)
    receipt = {
        "schema": "mlx2.xing-mla-fused-model-gate.v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        "model": str(args.model.resolve()),
        "host": platform.node(),
        "mlx_version": __import__("importlib.metadata", fromlist=["version"]).version("mlx"),
        "lock_owners": owners,
        "context": args.context,
        "batch": args.batch,
        "width": args.width,
        "repeats": args.repeats,
        "loaded_at_unix": started,
        "max_abs_logits": max_abs,
        "same_argmax": same_argmax,
        "numerical_gate_passed": True,
        "mechanism": xing4_0.fused_mla_stats(),
        "samples_ms": samples,
        "median_ms": {key: statistics.median(values) for key, values in samples.items()},
        "peak_delta_bytes": peaks,
        "host_before": host_before,
        "host_after": host_signals(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in ("context", "batch", "width", "max_abs_logits", "same_argmax", "median_ms", "mechanism")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
