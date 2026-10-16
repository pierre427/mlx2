#!/usr/bin/env python3
"""Bounded streamed-MoE trace plus router-reuse diagnostic.

This is a correctness/policy probe, not a benchmark or qualification run.
Router forecasts deliberately add gate projections and synchronization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def corpus_tokens(tokenizer, count: int) -> tuple[list[int], dict]:
    text = "\n\n".join(path.read_text(encoding="utf-8") for path in sorted((ROOT / "docs").glob("*.md")))
    ids = list(tokenizer.encode(text))
    if not ids:
        raise ValueError("probe corpus tokenized empty")
    return (ids * ((count + len(ids) - 1) // len(ids)))[:count], {
        "source": "docs/*.md in lexical order",
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def router_summary(records: list[dict], leads: tuple[int, ...]) -> dict:
    actual = {(row["step"], row["layer"]): set(row["actual"]) for row in records}
    result = {}
    for lead in leads:
        rows = []
        for row in records:
            forecast = row.get("forecast", {}).get(str(lead))
            target = actual.get((row["step"], row["layer"] + lead))
            if forecast is not None and target is not None:
                predicted = set(forecast)
                rows.append((len(predicted & target), predicted == target, len(target)))
        denominator = sum(width for _, _, width in rows)
        result[str(lead)] = {
            "comparisons": len(rows),
            "topk_recall": sum(hit for hit, _, _ in rows) / denominator if denominator else None,
            "exact_set_rate": sum(exact for _, exact, _ in rows) / len(rows) if rows else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget-gib", type=float, default=6.0)
    parser.add_argument("--read-workers", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--prefill-chunk", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--router-leads", default="1,2,4")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty evidence directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    leads = tuple(sorted({int(value) for value in args.router_leads.split(",") if value}))

    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.runtime.expert_atlas import AtlasCollector
    from mlx2.runtime.streamed_load import WeightStreamRequest

    trace_path = out / "expert-access.trace"
    collector = AtlasCollector(args.model, trace_path=trace_path, checkpoint_every=0)
    request = WeightStreamRequest(
        mode="moe_experts", budget_bytes=int(args.budget_gib * (1 << 30)),
        read_workers=args.read_workers, collector=collector,
    )
    adapter = None
    original_call = None
    moe = None
    records: list[dict] = []
    started = time.time()
    try:
        adapter = Qwen3635BA3BAdapter(args.model, execution_policy={"eager_dispatch_stride": 0}, weight_streaming=request)
        from mlx2.runtime.models import qwen36_moe_decode as moe_module
        from mlx2.runtime.models.cache import make_prompt_cache

        moe = moe_module
        original_call = moe.Qwen36SparseMoeBlock.__call__
        serving_start = int(collector.counters()["atlas_calls_total"])
        model = adapter.model
        manager = adapter.weight_stream
        blocks = [module for _, module in model.named_modules() if isinstance(module, moe.Qwen36SparseMoeBlock)]
        layer_of = {id(block): index for index, block in enumerate(blocks)}
        step = {"value": -1}

        def instrumented(block, x):
            index = layer_of[id(block)]
            rows = x.size // x.shape[-1]
            if rows == 1:
                if index == 0:
                    step["value"] += 1
                logits = block.gate(x)
                top_k = int(block.top_k)
                actual = mx.argpartition(logits, kth=-top_k, axis=-1)[..., -top_k:]
                forecasts = {}
                arrays = [actual]
                for lead in leads:
                    if index + lead < len(blocks):
                        forecast_logits = blocks[index + lead].gate(x)
                        selected = mx.argpartition(forecast_logits, kth=-top_k, axis=-1)[..., -top_k:]
                        forecasts[str(lead)] = selected
                        arrays.append(selected)
                mx.eval(*arrays)
                records.append({
                    "step": step["value"], "layer": index,
                    "actual": sorted(np.array(actual).reshape(-1).astype(int).tolist()),
                    "forecast": {key: sorted(np.array(value).reshape(-1).astype(int).tolist()) for key, value in forecasts.items()},
                })
            return original_call(block, x)

        moe.Qwen36SparseMoeBlock.__call__ = instrumented
        ids, corpus = corpus_tokens(adapter.tokenizer, args.prompt_tokens)
        cache = make_prompt_cache(model)
        cuts = []
        logits = None
        for start in range(0, len(ids), args.prefill_chunk):
            chunk = mx.array(ids[start:start + args.prefill_chunk], dtype=mx.int32)[None]
            logits = model(chunk, cache=cache)
            mx.eval(logits, *[entry.state for entry in cache])
            cuts.append(int(collector.counters()["atlas_calls_total"]))
        decode_start = cuts[-1]
        terminal_start = cuts[-2] if len(cuts) > 1 else 0
        generated = []
        for _ in range(args.decode_tokens):
            token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
            mx.eval(token)
            generated.append(int(np.array(token).reshape(-1)[0]))
            logits = model(token.reshape(1, 1), cache=cache)
            mx.eval(logits, *[entry.state for entry in cache])
        projections = [
            {
                "trace_layer": index, "path": path,
                "capacity": int(cache.capacity),
                "expert_bytes": int(cache.reader.expert_bytes),
                "group": str(cache.group),
            }
            for index, (path, cache) in enumerate(manager.caches.items())
        ]
        report = {
            "schema": "mlx2.streamed-moe-residency-probe.v1",
            "state": "experimental_unqualified_not_performance_evidence",
            "timing_eligible": False,
            "timing_ineligibility": "router diagnostic adds gate projections and host synchronization",
            "source_revision": os.environ.get("MLX2_SOURCE_REVISION") or subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_dirty": os.environ.get("MLX2_SOURCE_DIRTY", "unknown"),
            "artifact": adapter.identity,
            "platform": platform.platform(),
            "mlx_version": mx.__version__,
            "device": mx.device_info(),
            "parameters": vars(args),
            "corpus": corpus,
            "prefill_call_cuts": cuts,
            "serving_start_call": serving_start,
            "terminal_prefill_start_call": terminal_start,
            "decode_start_call": decode_start,
            "generated_token_ids": generated,
            "stream_plan": manager.plan.as_dict(),
            "stream_receipt": manager.receipt(),
            "stream_counters": manager.counters(),
            "projections": projections,
            "router_reuse": router_summary(records, leads),
            "router_records": len(records),
            "elapsed_diagnostic_seconds": time.time() - started,
        }
        atomic_json(out / "capture.json", report)
    finally:
        if moe is not None and original_call is not None:
            moe.Qwen36SparseMoeBlock.__call__ = original_call
        if adapter is not None:
            adapter.close()
        elif collector is not None:
            collector.close()


if __name__ == "__main__":
    main()
