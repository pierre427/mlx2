#!/usr/bin/env python3
"""Owned actual-weight Inkling MoE A/B, not generated-token throughput."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
from mlx import nn
from probe_inkling import atomic, host, ownership, source_revision

from mlx2.adapters.inkling import inspect_artifact
from mlx2.runtime.models import moe_nax_gather as nax
from mlx2.runtime.models.inkling import ModelArgs
from mlx2.runtime.models.inkling import experts as expert_tiles
from mlx2.runtime.models.inkling.language import (
    InklingSparseMoE,
    shared_experts_to_dense,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    artifact = inspect_artifact(args.model)
    revision = source_revision()
    report = {
        "checkpoint": args.model.name + "@" + artifact["identity"]["fingerprint"],
        "source_revision": revision,
        "ownership": ownership(),
        "before": host(),
        "scope": "actual layer 2 weights; random BF16 activations; interleaved operator A/B; no full-model throughput or qualification",
        "cases": [],
    }
    mx.set_default_device(mx.gpu)
    config = ModelArgs.from_dict(artifact["text_config"])
    m = InklingSparseMoE(config)
    m.set_dtype(mx.bfloat16)
    nn.quantize(m.switch_mlp, group_size=128, bits=2, mode="affine")
    prefix = "language_model.model.layers.2.mlp."
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    wanted = {key for key in index if key.startswith(prefix)}
    weights = {}
    for shard in sorted({index[k] for k in wanted}):
        loaded = mx.load(str(args.model / shard))
        weights.update({k: v for k, v in loaded.items() if k in wanted})
        del loaded
    weights = shared_experts_to_dense(weights)
    weights = {k[len(prefix) :]: v for k, v in weights.items()}
    router = weights.pop("gate_weight")
    pad = (-router.shape[0]) % 4
    weights["gate.weight"] = mx.pad(router, [(0, pad), (0, 0)]) if pad else router
    m.load_weights(list(weights.items()), strict=False)
    m.eval()
    mx.eval(m.parameters())
    weights.clear()
    gc.collect()
    mx.clear_cache()
    nax.set_mode("fused")
    nax.set_affine_bits((2,))
    mx.random.seed(3321)
    original_routed = expert_tiles.routed
    for rows in (1, 4, 8, 16, 32, 128):
        x = (mx.random.normal((1, rows, 4096)) * 0.7).astype(mx.bfloat16)
        mx.eval(x)
        outputs = {}
        modes = ["stock", "fused", "grouped"]
        if rows <= 64:
            modes += [
                "tree_fused",
                "tree_grouped",
                "tree_grouped_c1",
                "tree_grouped_c2",
            ]
        timings = {key: [] for key in modes}
        counters = {}

        def call(mode, x=x, rows=rows):
            nax.set_mode("off" if mode == "stock" else "fused")
            m.fused = mode != "stock"
            m.down_combine = mode != "stock"
            m.grouped = "grouped" in mode
            chunk = 1 if mode.endswith("_c1") else 2 if mode.endswith("_c2") else 4
            expert_tiles.routed = partial(original_routed, chunk=chunk)
            with nax.prefill_scope(rows > 64):
                return m.tree(x) if mode.startswith("tree_") else m(x)

        for mode in timings:
            before = (m.grouped_calls, m.down_combine_calls)
            for _ in range(2):
                outputs[mode] = call(mode)
                mx.eval(outputs[mode])
            counters[mode] = {
                "grouped_q2": m.grouped_calls - before[0],
                "down_combine": m.down_combine_calls - before[1],
                "calls": 2,
            }
        for repeat in range(6):
            order = list(timings) if repeat % 2 == 0 else list(reversed(timings))
            for mode in order:
                start = time.perf_counter()
                y = call(mode)
                mx.eval(y)
                timings[mode].append(time.perf_counter() - start)
        ref = outputs["stock"].astype(mx.float32)
        case = {
            "rows": rows,
            "warmup_engagement": counters,
            "seconds": timings,
            "medians": {k: statistics.median(v) for k, v in timings.items()},
            "max_abs_error": {
                k: mx.max(mx.abs(v.astype(mx.float32) - ref)).item()
                for k, v in outputs.items()
            },
            "grouped_tile_exact": all(
                bool(mx.array_equal(outputs["grouped"], output).item())
                for mode, output in outputs.items()
                if "grouped" in mode
            ),
        }
        report["cases"].append(case)
        atomic(args.out / "report.json", report)
        print(json.dumps(case), flush=True)
    report["after"] = host()
    report["nax"] = nax.status()
    atomic(args.out / "report.json", report)


if __name__ == "__main__":
    main()
