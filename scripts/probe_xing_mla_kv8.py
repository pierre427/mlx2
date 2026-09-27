#!/usr/bin/env python3
"""Bounded real-model Xing MLA latent-KV8 screen under the shared GPU lease.

Run through ``scripts/run_with_gpu_fcntl.py`` after the coordinator owns the
CPG GPU lease and host lock.  This tests a research cache format; it does not
publish an approximate serving route or calibrate admission memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST_LOCK = Path("/Users/Shared/mlxuag/gpu.lock/owner.json")
FCNTL_LOCK = Path("/tmp/gpu.lock")


def require_gpu_lease() -> dict:
    """Fail before importing MLX or loading weights without both receipts."""
    if os.environ.get("MLX2_GPU_FCNTL_LOCKED") != "1" or not FCNTL_LOCK.is_file():
        raise SystemExit("Xing KV8 probe requires inherited /tmp/gpu.lock fcntl lock")
    if not HOST_LOCK.is_file():
        raise SystemExit(f"Xing KV8 probe requires host ownership receipt {HOST_LOCK}")
    owner = json.loads(HOST_LOCK.read_text())
    if not isinstance(owner, dict) or not owner.get("pid") or not owner.get("cpg_generation"):
        raise SystemExit("Xing KV8 probe has an incomplete GPU ownership receipt")
    return owner


def git_commit() -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise SystemExit("commit candidate source before the source-bound KV8 GPU probe")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def trim_to(cache, length: int) -> None:
    for layer in cache:
        removed = layer.offset - length
        if removed < 0 or layer.trim(removed) != removed or layer.offset != length:
            raise RuntimeError("Xing KV8 probe could not restore cache boundary")


def cache_bytes(cache) -> int:
    return sum(int(layer.nbytes) for layer in cache)


def host_signals() -> dict[str, str]:
    signals = {}
    for name, command in {
        "swapusage": ["sysctl", "vm.swapusage"],
        "thermal": ["pmset", "-g", "therm"],
    }.items():
        try:
            signals[name] = subprocess.check_output(
                command, text=True, timeout=5
            ).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            signals[name] = f"unavailable: {type(exc).__name__}"
    return signals


def compare_logits(mx, exact, candidate) -> dict:
    delta = mx.abs(exact.astype(mx.float32) - candidate.astype(mx.float32))
    return {
        "max_abs": float(mx.max(delta).item()),
        "mean_abs": float(mx.mean(delta).item()),
        "greedy_equal": bool(mx.all(mx.argmax(exact, axis=-1) == mx.argmax(candidate, axis=-1)).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-abs-limit", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.context <= 2048:
        parser.error("context must be in [1, 2048]")
    if not 3 <= args.repeats <= 5:
        parser.error("repeats must be in [3, 5]")
    if not 0 < args.max_abs_limit <= 2:
        parser.error("max-abs-limit must be in (0, 2]")
    owner = require_gpu_lease()
    commit = git_commit()
    host_before = host_signals()

    import mlx.core as mx

    from mlx2.adapters.xing import XingAdapter
    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache
    from mlx2.runtime.models.xing4_0 import set_fused_mla
    from mlx2.runtime.models.xing_latent_kv8 import (
        XingLatentKV8Cache,
        latent_kv8_stats,
    )

    mx.set_default_device(mx.gpu)
    set_fused_mla(False)  # The fused candidate is not qualified with KV8.
    loaded_at = time.time()
    adapter = XingAdapter(str(args.model), execution_policy={"num_draft": 1})
    try:
        model = adapter.model
        quant = adapter.config.get("quantization", adapter.config.get("quantization_config"))
        if not isinstance(quant, dict) or quant.get("bits") != 6:
            raise RuntimeError("KV8 probe requires the Xing 6-bit weight artifact")
        mx.eval(model.parameters())
        # Deterministic, valid token IDs.  This is a numerical and mechanism
        # screen, not a language-quality qualification corpus.
        vocab = int(model.args.vocab_size)
        tokens = mx.array(
            [[100 + (i * 37) % (vocab - 101) for i in range(args.context + 2)]],
            dtype=mx.int32,
        )
        exact_cache = model.make_cache()
        approx_cache = model.make_cache(cache_factory=XingLatentKV8Cache)
        latent_kv8_stats(reset=True)

        def prefill(cache):
            for start in range(0, args.context, 256):
                stop = min(start + 256, args.context)
                mx.eval(model(tokens[:, start:stop], cache=cache))
            if any(layer.offset != args.context for layer in cache):
                raise RuntimeError("Xing KV8 prefill cache offsets disagree")

        prefill(exact_cache)
        prefill(approx_cache)  # BF16-only cache rejects a mismatched activation.
        exact_storage = cache_bytes(exact_cache)
        approx_storage = cache_bytes(approx_cache)
        if approx_storage >= exact_storage:
            raise RuntimeError("Xing KV8 candidate did not reduce cache storage")

        with tempfile.TemporaryDirectory(prefix="mlx2-xing-kv8-") as scratch:
            exact_file = str(Path(scratch) / "exact.safetensors")
            approx_file = str(Path(scratch) / "latent-kv8.safetensors")
            save_prompt_cache(exact_file, exact_cache)
            save_prompt_cache(approx_file, approx_cache)
            restored_exact = load_prompt_cache(exact_file)
            restored_approx = load_prompt_cache(approx_file)
            if any(type(layer) is not type(exact_cache[0]) for layer in restored_exact):
                raise RuntimeError("exact prompt-cache restore changed cache layout")
            if any(type(layer) is not XingLatentKV8Cache for layer in restored_approx):
                raise RuntimeError("KV8 prompt-cache restore changed cache layout")
            if any(layer.offset != args.context for layer in restored_exact + restored_approx):
                raise RuntimeError("prompt-cache restore changed offset")

            # Restore is verified before benchmark loops; both restored caches
            # append the same one-token query and roll back independently.
            probe = tokens[:, args.context : args.context + 1]
            original_exact = model(probe, cache=exact_cache)
            original_approx = model(probe, cache=approx_cache)
            restored_exact_logits = model(probe, cache=restored_exact)
            restored_approx_logits = model(probe, cache=restored_approx)
            mx.eval(original_exact, original_approx, restored_exact_logits, restored_approx_logits)
            exact_restore = compare_logits(mx, original_exact, restored_exact_logits)
            approx_restore = compare_logits(mx, original_approx, restored_approx_logits)
            for cache in (exact_cache, approx_cache, restored_exact, restored_approx):
                trim_to(cache, args.context)
            if exact_restore["max_abs"] != 0 or approx_restore["max_abs"] != 0:
                raise RuntimeError("prompt-cache restore changed next-token logits")

        widths = {}
        for width in (1, 2):
            query = tokens[:, args.context : args.context + width]
            samples = {"exact": [], "latent_kv8": []}
            peaks = {"exact": [], "latent_kv8": []}
            logits = {}
            for repeat in range(args.repeats):
                for name, cache in (
                    (("exact", exact_cache), ("latent_kv8", approx_cache))
                    if repeat % 2 == 0
                    else (("latent_kv8", approx_cache), ("exact", exact_cache))
                ):
                    trim_to(cache, args.context)
                    active = mx.get_active_memory()
                    mx.reset_peak_memory()
                    started = time.perf_counter_ns()
                    result = model(query, cache=cache)
                    mx.eval(result)
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    logits[name] = result
                    samples[name].append(elapsed)
                    peaks[name].append(max(0, mx.get_peak_memory() - active))
            comparison = compare_logits(mx, logits["exact"], logits["latent_kv8"])
            widths[str(width)] = {
                "comparison": comparison,
                "samples_ms": samples,
                "median_ms": {name: statistics.median(values) for name, values in samples.items()},
                "peak_delta_bytes": peaks,
            }
            trim_to(exact_cache, args.context)
            trim_to(approx_cache, args.context)

        mechanism = latent_kv8_stats()
        if mechanism["pack_calls"] <= 0 or mechanism["dequant_calls"] <= 0 or mechanism["max_cache_bytes"] <= 0:
            raise RuntimeError("KV8 candidate did not engage")
        quality_pass = all(
            entry["comparison"]["greedy_equal"]
            and entry["comparison"]["max_abs"] <= args.max_abs_limit
            for entry in widths.values()
        )
        receipt = {
            "schema": "mlx2.xing-mla-latent-kv8-model-probe.v1",
            "source_commit": commit,
            "model": str(args.model.resolve()),
            "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
            "host": platform.node(),
            "mlx_version": __import__("importlib.metadata", fromlist=["version"]).version("mlx"),
            "gpu_owner": owner,
            "host_before": host_before,
            "host_after": host_signals(),
            "loaded_at_unix": loaded_at,
            "context": args.context,
            "batch": 1,
            "repeats": args.repeats,
            "exact_cache_bytes": exact_storage,
            "latent_kv8_cache_bytes": approx_storage,
            "cache_ratio": approx_storage / exact_storage,
            "restore": {"exact": exact_restore, "latent_kv8": approx_restore},
            "widths": widths,
            "mechanism": mechanism,
            "quality_limit_max_abs": args.max_abs_limit,
            "quality_pass": quality_pass,
            "qualification": "research model-path screen only; no APCv2 or serving selection",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({key: receipt[key] for key in (
            "context", "cache_ratio", "restore", "widths", "mechanism", "quality_pass"
        )}, sort_keys=True), flush=True)
        if not quality_pass:
            raise SystemExit("Xing latent KV8 model-path quality screen failed; see receipt")
    finally:
        set_fused_mla(False)
        adapter.close()


if __name__ == "__main__":
    main()
