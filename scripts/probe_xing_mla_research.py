#!/usr/bin/env python3
"""GPU-lease-only Xing MLA hybrid and shifted-reuse research probes.

The hybrid screen uses a loaded Xing model's cache and attention weights. Its
query is projected from a real post-trunk hidden state, so this is an
attention-path comparison, not an end-to-end serving throughput claim. The
shifted-reuse screen recomputes target context and reports layer-state,
continuation-hidden, and logit error; it never installs a serving cache route.

Run through ``scripts/run_with_gpu_fcntl.py`` after obtaining the CPG GPU
lease and host ownership receipt. This script cannot acquire a lease itself.
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
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOST_LOCK = Path("/Users/Shared/mlxuag/gpu.lock/owner.json")
FCNTL_LOCK = Path("/tmp/gpu.lock")
TINY = ROOT / "tests" / "fixtures" / "xing4_0_tiny"


def require_locks() -> dict:
    if not HOST_LOCK.is_file():
        raise SystemExit(f"GPU probe requires ownership receipt {HOST_LOCK}")
    if not FCNTL_LOCK.is_file() or os.environ.get("MLX2_GPU_FCNTL_LOCKED") != "1":
        raise SystemExit("GPU probe requires the inherited /tmp/gpu.lock fcntl wrapper")
    owner = json.loads(HOST_LOCK.read_text())
    if not isinstance(owner, dict) or not owner.get("pid") or not owner.get("cpg_generation"):
        raise SystemExit(f"GPU ownership receipt is incomplete: {HOST_LOCK}")
    return owner


def host_signals() -> dict[str, str]:
    signals = {}
    for name, command in {
        "swapusage": ["sysctl", "vm.swapusage"],
        "thermal": ["pmset", "-g", "therm"],
    }.items():
        try:
            signals[name] = subprocess.check_output(command, text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            signals[name] = f"unavailable: {type(exc).__name__}"
    return signals


def load_model(path: Path, *, tiny: bool):
    import mlx.core as mx

    if not tiny:
        from mlx2.adapters.xing import XingAdapter

        return XingAdapter(str(path), execution_policy={"num_draft": 1}).model
    from mlx2.runtime.models.xing4_0 import Model, ModelArgs

    config = json.loads((path / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    model.load_weights(list(model.sanitize(mx.load(str(path / "weights.safetensors"))).items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def dense_multilinear_weight(module):
    import mlx.core as mx

    if not hasattr(module, "scales"):
        return module.weight
    return mx.dequantize(
        module.weight,
        module.scales,
        getattr(module, "biases", None),
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
    )


def hybrid_probe(model, *, prefix: int, suffix: int, batches: tuple[int, ...], repeats: int,
                 tile_size: int, max_abs_allowed: float, seed: int) -> dict:
    import mlx.core as mx

    from mlx2.runtime.models import xing4_0
    from mlx2.runtime.models.xing_mla_hybrid_mlx import (
        hybrid_shared_prefix_attention_mlx,
    )

    xing4_0.set_fused_mla(False)
    attention = model.layers[0].self_attn
    embed_weight = dense_multilinear_weight(attention.embed_q)
    unembed_weight = dense_multilinear_weight(attention.unembed_out)
    mx.eval(embed_weight, unembed_weight)
    vocab = int(model.args.vocab_size)
    if vocab <= 16:
        raise ValueError("Xing probe needs at least 17 vocabulary tokens")
    rng = np.random.default_rng(seed)
    outcomes = {}
    for batch in batches:
        shared_ids = rng.integers(5, vocab, size=(1, prefix), dtype=np.int32)
        private_ids = rng.integers(5, vocab, size=(batch, suffix + 1), dtype=np.int32)
        ids = mx.array(np.concatenate((np.repeat(shared_ids, batch, axis=0), private_ids), axis=1))
        cache = model.make_cache()
        hidden = model.model(ids, cache=cache)
        mx.eval(hidden)
        keys, rope_keys = cache[0].keys_and_values()
        context = prefix + suffix + 1
        if keys.shape != (batch, 1, context, attention.kv_lora_rank):
            raise RuntimeError("Xing hybrid probe received an unexpected latent cache")

        # A real model hidden state drives the layer's own Q projection. The
        # cache and folded weights are also from the loaded checkpoint.
        h = hidden[:, -1:, :]
        if attention.q_lora_rank is None:
            projected = attention.q_proj(h)
        else:
            projected = attention.q_b_proj(attention.q_a_layernorm(attention.q_a_proj(h)))
        projected = projected.reshape(batch, 1, attention.num_heads, attention.q_head_dim)
        projected = projected.transpose(0, 2, 1, 3)
        query_nope, query_rope = mx.split(projected, [attention.qk_nope_head_dim], axis=-1)
        query_rope = attention.rope(query_rope, context - 1)
        query_positions = mx.full((batch, 1), context - 1, dtype=mx.int32)
        shared_positions = mx.arange(prefix, dtype=mx.int32)
        suffix_positions = mx.broadcast_to(
            mx.arange(prefix, context, dtype=mx.int32)[None, :], (batch, suffix + 1)
        )
        shared_latent = keys[0, 0, :prefix]
        shared_rope = rope_keys[0, 0, :prefix]
        private_latent = keys[:, 0, prefix:]
        private_rope = rope_keys[:, 0, prefix:]

        # Identical shared token prefixes must yield identical cached state.
        prefix_delta = float(mx.max(mx.abs(keys[:, 0, :prefix].astype(mx.float32)
                                           - shared_latent[None].astype(mx.float32))).item())
        if prefix_delta > 1e-3:
            raise RuntimeError(f"shared prefix states diverged across lanes: {prefix_delta}")

        candidate_calls = 0

        def ordinary(inputs=(query_nope, query_rope, keys, rope_keys)):
            value = attention._attend(*inputs, None)
            mx.eval(value)
            return value

        candidate_inputs = {
            "query_nope": query_nope,
            "query_rope": query_rope,
            "shared_latent": shared_latent,
            "shared_rope": shared_rope,
            "suffix_latent": private_latent,
            "suffix_rope": private_rope,
            "embed_weight": embed_weight,
            "unembed_weight": unembed_weight,
            "query_positions": query_positions,
            "shared_positions": shared_positions,
            "suffix_positions": suffix_positions,
            "scale": attention.scale,
            "tile_size": tile_size,
        }

        def candidate(inputs=candidate_inputs):
            nonlocal candidate_calls
            value = hybrid_shared_prefix_attention_mlx(**inputs)
            mx.eval(value)
            candidate_calls += 1
            return value

        ordinary_value = ordinary()
        candidate_value = candidate()
        discrepancy = ordinary_value.astype(mx.float32) - candidate_value.astype(mx.float32)
        max_abs = float(mx.max(mx.abs(discrepancy)).item())
        rms = float(mx.sqrt(mx.mean(discrepancy * discrepancy)).item())
        finite = bool(mx.all(mx.isfinite(candidate_value)).item())
        engaged = candidate_calls > 0
        passed = finite and engaged and max_abs <= max_abs_allowed
        timings = {"ordinary": [], "hybrid": []}
        peaks = {"ordinary": [], "hybrid": []}
        if passed:
            ordinary()
            candidate()
            for repeat in range(repeats):
                order = ("ordinary", "hybrid") if repeat % 2 == 0 else ("hybrid", "ordinary")
                for arm in order:
                    before = mx.get_active_memory()
                    mx.reset_peak_memory()
                    start = time.perf_counter_ns()
                    ordinary() if arm == "ordinary" else candidate()
                    timings[arm].append((time.perf_counter_ns() - start) / 1e6)
                    peaks[arm].append(max(0, mx.get_peak_memory() - before))
        outcomes[str(batch)] = {
            "context_tokens": context,
            "shared_prefix_tokens": prefix,
            "private_suffix_tokens": suffix + 1,
            "shared_prefix_max_abs_between_lanes": prefix_delta,
            "max_abs_attention_output": max_abs,
            "rms_attention_output": rms,
            "candidate_finite": finite,
            "candidate_calls": candidate_calls,
            "candidate_engaged": engaged,
            "numerical_gate_passed": passed,
            "samples_ms": timings,
            "median_ms": {arm: statistics.median(values) for arm, values in timings.items() if values},
            "peak_delta_bytes": peaks,
        }
        if not engaged:
            raise RuntimeError("hybrid candidate was not observed used")
        del cache, hidden, ids
        mx.clear_cache()
    xing4_0.set_fused_mla(False)
    return outcomes


def reuse_probe(model, *, seed: int, model_revision: str) -> dict:
    from mlx2.runtime.models.xing_mla_reuse_probe import probe_shifted_reuse_on_model

    vocab = int(model.args.vocab_size)
    rng = np.random.default_rng(seed + 1)
    source_prefix = rng.integers(5, vocab, size=16, dtype=np.int32)
    target_prefix = rng.integers(5, vocab, size=24, dtype=np.int32)
    chunk = rng.integers(5, vocab, size=8, dtype=np.int32)
    continuation = rng.integers(5, vocab, size=1, dtype=np.int32)
    result = probe_shifted_reuse_on_model(
        model,
        source_prefix_ids=source_prefix,
        target_prefix_ids=target_prefix,
        shared_chunk_ids=chunk,
        continuation_ids=continuation,
        model_revision=model_revision,
    )
    return asdict(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=Path, help="converted Xing model directory")
    model_group.add_argument("--tiny", action="store_true", help="load the checked-in tiny Xing fixture")
    parser.add_argument("--mode", choices=("hybrid", "reuse", "both"), default="both")
    parser.add_argument("--prefix", type=int, default=128)
    parser.add_argument("--suffix", type=int, default=128)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--max-abs", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.prefix < 1 or args.suffix < 0 or args.prefix + args.suffix + 1 > 4096:
        parser.error("hybrid prefix/suffix must have 1..4096 total cached tokens")
    if not args.batches or any(batch not in (1, 4) for batch in args.batches) or len(set(args.batches)) != len(args.batches):
        parser.error("--batches must be distinct values from 1 and 4")
    if args.repeats < 3 or args.repeats > 20 or args.tile_size < 1 or args.tile_size > 4096:
        parser.error("repeats must be 3..20 and tile size 1..4096")
    if not np.isfinite(args.max_abs) or args.max_abs <= 0:
        parser.error("--max-abs must be finite and positive")
    owner = require_locks()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True
    )
    if dirty:
        raise SystemExit("GPU probe requires a clean committed source tree")
    before = host_signals()

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    path = TINY if args.tiny else args.model.resolve()
    model = load_model(path, tiny=args.tiny)
    mx.eval(model.parameters())
    config_hash = hashlib.sha256((path / "config.json").read_bytes()).hexdigest()
    receipt = {
        "schema": "mlx2.xing-mla-research-gpu-probe.v1",
        "scope": "attention-path hybrid and offline approximate-reuse model probe",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_dirty": False,
        "model": str(path),
        "model_config_sha256": config_hash,
        "model_revision": config_hash,
        "tiny_fixture": args.tiny,
        "host": platform.node(),
        "mlx_version": __import__("importlib.metadata", fromlist=["version"]).version("mlx"),
        "gpu_owner": owner,
        "settings": {"mode": args.mode, "prefix": args.prefix, "suffix": args.suffix,
                     "batches": args.batches, "repeats": args.repeats, "tile_size": args.tile_size,
                     "max_abs": args.max_abs, "seed": args.seed},
        "host_before": before,
        "started_unix": time.time(),
    }
    error = None
    try:
        if args.mode in ("hybrid", "both"):
            receipt["hybrid"] = hybrid_probe(
                model, prefix=args.prefix, suffix=args.suffix, batches=tuple(args.batches),
                repeats=args.repeats, tile_size=args.tile_size,
                max_abs_allowed=args.max_abs, seed=args.seed,
            )
        if args.mode in ("reuse", "both"):
            receipt["shifted_reuse"] = reuse_probe(model, seed=args.seed, model_revision=config_hash)
    except Exception as exc:  # noqa: BLE001 - preserve an evidence receipt for any probe failure
        error = f"{type(exc).__name__}: {exc}"
        receipt["error"] = error
    finally:
        receipt["finished_unix"] = time.time()
        receipt["host_after"] = host_signals()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"receipt": str(args.out), "hybrid": receipt.get("hybrid"),
                          "shifted_reuse": receipt.get("shifted_reuse"), "error": error}, sort_keys=True), flush=True)
    if error:
        raise SystemExit(error)
    if args.mode in ("hybrid", "both") and not all(
        cell["numerical_gate_passed"] for cell in receipt["hybrid"].values()
    ):
        raise SystemExit("hybrid attention numerical gate failed; see receipt")


if __name__ == "__main__":
    main()
