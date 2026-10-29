"""Bounded synthetic Metal component diagnostic, never model qualification.

One fixture artifact, one CPG/paired-lock window; no autoregressive server,
KV, APCv2, streaming or performance claims. Refuses occupied evidence paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def environment():
    from run_single_model_smoke_ladder import thermal_sample

    foreign = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        command = " ".join(process.info["cmdline"] or [])
        if any(marker in command for marker in (
            "mlx2.server", "mlx2 serve", "mlx_lm.server", "omlx serve",
            "vllm serve", "llama-server", "desktop_campaign_worker artifact",
        )):
            foreign.append(process.info["pid"])
    waiters = Path("/Users/Shared/mlxuag/gpu.lock.waiters")
    queued = sorted(p.name for p in waiters.iterdir()) if waiters.exists() else []
    sample = {"thermal": thermal_sample(), "swap": psutil.swap_memory()._asdict(),
              "foreign_model_pids": foreign, "waiters": queued,
              "time": time.time()}
    if foreign or sample["thermal"]["thermal_state"] > 1:
        raise RuntimeError("foreign model process or unsuitable thermals")
    return sample


def child(spec):
    from mlx2.desktop_campaign_worker import atomic, check_ownership

    out = Path(spec["out"])
    manifest = json.loads((out / "source-files.json").read_text())
    if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
           for name, digest in manifest.items()):
        raise RuntimeError("source changed after component fixture was sealed")
    ownership = check_ownership(spec)  # Before importing MLX.
    before = environment()
    import mlx.core as mx
    from mlx import nn

    from mlx2.runtime.models import lfm25_moe_decode as fusion
    from mlx2.runtime.models import qwen4_ple_window as window
    from mlx2.runtime.models.lfm25_moe import SparseMoE
    from mlx2.runtime.models.qwen4_exp import (
        NGramEmbedding,
        ShardedEmbedding,
        TextModelArgs,
    )
    from mlx2.runtime.models.row_exact_qmv import (
        TileGeometry,
        per_row,
        quantized_linear,
    )
    from mlx2.runtime.row_exact.canary import bit_mismatches, build_identity

    results = []
    report = {**spec, "ownership": ownership, "before": before,
              "execution_kind": "synthetic_component_reference",
              "qualified": False, "selected": False,
              "not_applicable": ["model serving", "KV", "APCv2", "streaming"],
              "cells": results, "runtime_build": build_identity(),
              "device": mx.device_info()}
    try:
        mx.set_default_device(mx.cpu)
        # Ordinary CPU reference smoke before any candidate dispatch.
        a = mx.arange(128, dtype=mx.float32).reshape(1, 1, 128)
        smoke = mx.sum(a * a).item()
        if smoke != 690880:
            raise RuntimeError("ordinary synthetic reference smoke failed")
        report["ordinary_reference_smoke"] = {"passed": True, "sum_squares": smoke}
        mx.set_default_device(mx.gpu)
        mx.random.seed(20261027)
        for experts in (32, 64, 256):
            for top_k in (1, 4, 8):
                for bias in (False, True):
                    for norm in (False, True):
                        block = SimpleNamespace(num_experts=experts, top_k=top_k,
                            use_expert_bias=bias, norm_topk_prob=norm,
                            routed_scaling_factor=1.5,
                            expert_bias=mx.arange(experts, dtype=mx.float32) / 100)
                        for dtype in (mx.bfloat16, mx.float32):
                            for seed in (0, 1, 2, 3):
                                p = (mx.full((1, 1, experts), 0.5) if seed == 0 else
                                     mx.sigmoid(mx.random.normal((1, 1, experts),
                                         key=mx.random.key(seed)) * (seed + 1))).astype(dtype)
                                actual = fusion.route(block, p)
                                expected = fusion.reference_route(block, p)
                                values = mx.random.normal((1, 1, top_k, 128),
                                    key=mx.random.key(100 + seed)).astype(mx.bfloat16)
                                combined = fusion.combine(values, expected[1])
                                reference = mx.sum(values * expected[1][..., None], axis=-2)
                                mx.eval(actual, expected, combined, reference)
                                mismatches = [bit_mismatches(x, y) for x, y in zip(actual, expected)]
                                mismatches.append(bit_mismatches(combined, reference))
                                results.append({"component": "lfm", "experts": experts,
                                    "top_k": top_k, "bias": bias, "normalize": norm,
                                    "probability_dtype": str(dtype), "seed": seed,
                                    "bit_mismatches": mismatches})
        for biased in (False, True):
            for normalized in (False, True):
                block = SparseMoE(SimpleNamespace(num_experts=32, num_experts_per_tok=4,
                    norm_topk_prob=normalized, use_expert_bias=biased,
                    routed_scaling_factor=1.5, hidden_size=128, moe_intermediate_size=128))
                block.set_dtype(mx.bfloat16)
                block.gate.weight = block.gate.weight.astype(mx.float32)
                if biased:
                    block.expert_bias = mx.arange(32, dtype=mx.float32) / 100
                block.eval()
                nn.quantize(block.switch_mlp, group_size=64, bits=8)
                xs = [mx.random.normal((1, 1, 128), key=mx.random.key(seed)).astype(mx.bfloat16)
                      for seed in range(4)]
                references = [block(x) for x in xs]
                mx.eval(references)
                handle = fusion.install(SimpleNamespace(named_modules=lambda bound=block: [("block", bound)]))
                actuals = [block(x) for x in xs]
                mx.eval(actuals)
                results.append({"component": "lfm_block", "bias": biased,
                    "normalize": normalized, "expert_bits": 8,
                    "bit_mismatches": [bit_mismatches(a, b) for a, b in zip(actuals, references)],
                    "diagnostic_dispatch": handle.status()})
        for k in (256, 512):
            linear = nn.QuantizedLinear(k, 128, bias=False, group_size=64, bits=8)
            linear.scales = linear.scales.astype(mx.bfloat16)
            linear.biases = linear.biases.astype(mx.bfloat16)
            for rows in (6, 7, 8, 11, 16):
                x = mx.random.normal((rows, k), key=mx.random.key(rows)).astype(mx.bfloat16)
                reference = per_row(linear, x)
                for group in (6, 7, 8):
                    if group > rows:
                        continue
                    actual, route = quantized_linear(linear, x,
                        geometry=TileGeometry(group, row_value_budget=64))
                    mx.eval(actual, reference)
                    results.append({"component": "head", "k": k, "rows": rows,
                        "group": group, "route": route,
                        "bit_mismatches": [bit_mismatches(actual, reference)]})
        for bits in (None, 4, 8):
            table = ShardedEmbedding(128, 32, 8)
            for shard in range(8):
                getattr(table, f"shard_{shard}").weight = (
                    mx.arange(512).reshape(16, 32) / 100 + shard)
            if bits:
                nn.quantize(table, bits=bits, group_size=32)
            ranges = ((0, 33), (33, 80), (80, 128))
            ids = mx.array([[[0, 33, 80], [15, 47, 95], [16, 64, 96], [32, 79, 127]]])
            reference = table(ids)
            actual = window.lookup(table, ids, ranges)
            mx.eval(actual, reference)
            results.append({"component": "ple", "bits": bits,
                "bit_mismatches": [bit_mismatches(actual, reference)]})
        # Full producer: identical masks/history, ordinary CPU hash/gather
        # versus device hash/window gather, then one continuation token.
        args = TextModelArgs(hidden_size=128, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, vocab_size=64,
            ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=17,
            make_ngram_vocab_size_divisible_by=4, split_ngram_parts=4,
            eos_token_id=63)
        module = NGramEmbedding(args, 128, 0, 0)
        mx.eval(module.ngram_embedding.parameters())  # Only the tiny synthetic table.
        module.prepare_windowed_gather(resident_shards_prepared=True)
        left, right = [None] * 4, [None] * 4
        for ids, mask in ((mx.array([[3, 5, 63, 7], [11, 13, 0, 0]]),
                           mx.array([[True] * 4, [True, True, False, False]])),
                          (mx.array([[17], [19]]), None)):
            module.hash_backend, module.windowed_gather = "cpu", False
            reference = module(ids, left, mask)
            mx.eval(reference, left[3])
            module.hash_backend, module.windowed_gather = "metal", True
            if not module._window_eligible(ids):
                raise RuntimeError("synthetic resident producer did not admit window gather")
            actual = module(ids, right, mask)
            mx.eval(actual, right[3])
            results.append({"component": "ple_producer", "shape": list(ids.shape),
                "bit_mismatches": [bit_mismatches(actual, reference),
                                   bit_mismatches(right[3], left[3])]})
        report["passed"] = all(not any(row["bit_mismatches"]) for row in results)
        if not report["passed"]:
            raise RuntimeError("synthetic component parity failed; inspect results")
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        cleanup_errors = []
        for name, operation in (("synchronize", mx.synchronize),
                                ("after", environment),
                                ("ownership_after", lambda: check_ownership(spec))):
            try:
                report[name] = operation()
            except Exception as error:  # noqa: BLE001 -- seal failures before releasing ownership
                cleanup_errors.append(f"{name}: {type(error).__name__}: {error}")
        report["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            report["passed"] = False
        atomic(out / "components.json", report)
        if cleanup_errors:
            raise RuntimeError("component cleanup validation failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--child-spec", type=Path)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    if args.child_spec:
        child(json.loads(args.child_spec.read_text()))
        return
    if args.out is None:
        parser.error("--out required")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    from mlx2.desktop_campaign_process import spawn
    from mlx2.desktop_campaign_worker import atomic, stop

    before = environment()
    if before["waiters"]:
        raise RuntimeError("yield to queued GPU work before opening a component window")
    atomic(out / "environment-before.json", before)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = subprocess.check_output(["git", "ls-files", "-m", "-o", "--exclude-standard"],
                                   cwd=ROOT, text=True).splitlines()
    manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in files if (ROOT / name).is_file()}
    atomic(out / "source-files.json", manifest)
    fixture = {"schema": "mlx2.pr-adoption-component-fixture.v1", "seed_range": [0, 3], "global_initialization_seed": 20261027,
               "source_files": manifest}
    fingerprint = hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest()
    spec = {"out": str(out), "job_id": "pr-adoption-" + out.name,
            "checkpoint": "synthetic-pr-adoption-v1@recipe-sha256:" + fingerprint,
            "source_revision": revision, "source_files_sha256": fingerprint,
            "mcp_url": "http://127.0.0.1:8766/mcp"}
    atomic(out / "spec.json", spec)
    process = None
    with DesktopCPGLease(out, spec["job_id"], checkpoint=spec["checkpoint"],
                         source_revision=revision) as lease:
        try:
            command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                "--session", lease.owner["cpg_session"], "--label", spec["job_id"],
                "--desktop-cpg-lease", str(lease.path), "--mcp-url", spec["mcp_url"],
                "--receipt", str(out / "paired-locks.json"), "--", sys.executable,
                "-u", str(Path(__file__).resolve()), "--child-spec", str(out / "spec.json")]
            process = spawn(command, start_new_session=True)
            deadline = time.monotonic() + 300
            while process.poll() is None:
                lease.check()
                if time.monotonic() > deadline:
                    raise TimeoutError("component window exceeded five minutes")
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise RuntimeError(f"component child failed: {process.returncode}")
            lease.success = True
        finally:
            stop(process)


if __name__ == "__main__":
    main()
