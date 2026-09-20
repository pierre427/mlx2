#!/usr/bin/env python3
"""End-to-end concurrent multi-LoRA throughput and parity (GPU-gated).

Loads one model on the ordinary route with ``max_loras`` slots, writes random
rank-R adapters over the attention projections, and runs interleaved arms:

* ``base``  -- every row base (manager attached, delta skipped);
* ``mixN``  -- rows round-robin over N adapters (N in --adapters, N > 0).

Each arm submits ``--lanes`` concurrent greedy requests and records completion
tokens per wall second.  Arms are interleaved ``--interleave`` times
(A B C A B C ...) because back-to-back server A/Bs drift ~10%.

Mechanism gates (an arm whose counter did not move is refused):
* ``mixN``: ``rows_adapter`` and ``delta_applications`` advanced;
* ``base``: ``base_only_forwards`` advanced and ``delta_applications`` did not.

Parity: every adapter's mixed-batch row is compared with the same request run
alone (B=1); the first-divergence index is recorded.

Go/no-go (see the rm05 plan): mix4 decode tok/s >= 0.85 x base at the chosen
lanes and full greedy parity on the first 32 tokens.

``--tiny-cpu-smoke`` runs the whole harness on the CPU test fixture model so
the script logic is verified without Metal.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

PROMPTS = [
    "Write a SQL query that lists the ten most recent orders.",
    "Summarize the plot of a heist movie in three sentences.",
    "Explain what a radix tree is to a new engineer.",
    "Give three tips for writing clear commit messages.",
    "Describe how a bicycle gear system works.",
    "What is the difference between a process and a thread?",
    "Draft a polite reminder email about an overdue invoice.",
    "List five uses for a paperclip besides holding paper.",
]


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model")
    p.add_argument("--adapters", default="0,1,2,4", help="arms: adapter counts (0 = base)")
    p.add_argument("--lanes", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--targets", default="self_attn.q_proj,self_attn.v_proj,self_attn.o_proj")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--parity-tokens", type=int, default=32)
    p.add_argument("--interleave", type=int, default=3)
    p.add_argument("--out")
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tiny-cpu-smoke", action="store_true")
    return p.parse_args()


def build_engine(args, lora_root, max_loras):
    from mlx2.serving import ServingEngine

    kwargs = dict(
        qualification_mode=True,
        mtp=False,
        lora_root=str(lora_root),
        max_loras=max_loras,
        max_lora_rank=args.rank,
        max_lanes=args.lanes,
        max_inflight=max(args.lanes * 2, 8),
        coalesce_window_ms=50.0,
    )
    if args.tiny_cpu_smoke:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from test_approximate_kv_serving import make_adapter, tiny_model

        from mlx2 import memory, serving
        from mlx2.runtime import os_memory

        serving.runtime_identity = lambda: {"source_sha256": "smoke"}
        memory.execution_headroom = lambda: 100 * 2**30
        os_memory.physical_footprint_bytes = lambda: 0
        engine = ServingEngine("tiny", adapter_factory=make_adapter(tiny_model(), operations=None), **kwargs)
    else:
        engine = ServingEngine(args.model, **kwargs)
    if not engine.ready.wait(1800):
        raise SystemExit(f"engine not ready: {engine.error}")
    if engine.error:
        raise SystemExit(f"engine failed: {engine.error}")
    return engine


def request(args, index, model):
    if args.tiny_cpu_smoke:
        body = {"tokens": [(7 * index + k) % 120 + 1 for k in range(12)]}
    else:
        body = {"messages": [{"role": "user", "content": PROMPTS[index % len(PROMPTS)]}]}
    body.update(max_tokens=args.max_tokens, temperature=0)
    if model:
        body["model"] = model
    return body


def collect(job):
    pieces = []
    while True:
        event = job.events.get(timeout=1800)
        if "error" in event:
            raise RuntimeError(event)
        if "delta" in event:
            # Thinking models stream into reasoning_content first; a 128-token
            # budget can end before any content, so parity must see both
            # channels or it compares two empty strings.
            delta = event["delta"]
            pieces.append((delta.get("reasoning_content", "") or "") + (delta.get("content", "") or ""))
        if "finish_reason" in event:
            return pieces, event


def run_arm(engine, args, names):
    before = dict(engine.multi_lora.status()["counts"])
    rows = [names[i % len(names)] if names else None for i in range(args.lanes)]
    start = time.perf_counter()
    jobs = [engine.submit(request(args, i, rows[i])) for i in range(args.lanes)]
    outputs = [collect(job) for job in jobs]
    wall = time.perf_counter() - start
    after = engine.multi_lora.status()["counts"]
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in set(after) | set(before)}
    tokens = sum(event["receipt"]["completion_tokens"] for _, event in outputs)
    if names:
        if delta.get("rows_adapter", 0) <= 0 or delta.get("delta_applications", 0) <= 0:
            raise SystemExit(f"mechanism gate: adapter arm {names} did not run LoRA deltas: {delta}")
    else:
        if delta.get("base_only_forwards", 0) <= 0 or delta.get("delta_applications", 0) != 0:
            raise SystemExit(f"mechanism gate: base arm ran deltas or no forwards: {delta}")
    return {
        "rows": rows,
        "wall_s": wall,
        "completion_tokens": tokens,
        "tok_s": tokens / wall,
        "mixed_forwards": delta.get("mixed_forwards", 0),
        "delta_applications": delta.get("delta_applications", 0),
        "outputs": [pieces for pieces, _ in outputs],
    }


def main():
    args = parse()
    arms = [int(v) for v in args.adapters.split(",") if v != ""]
    if args.dry_run:
        print(json.dumps({"arms": arms, "lanes": args.lanes, "rank": args.rank,
                          "targets": args.targets.split(","), "interleave": args.interleave,
                          "model": args.model}, indent=1))
        return 0
    if not args.tiny_cpu_smoke:
        if not args.i_own_the_gpu:
            print("refusing to touch Metal without --i-own-the-gpu", file=sys.stderr)
            return 2
        if not args.model:
            print("--model is required", file=sys.stderr)
            return 2
    import mlx.core as mx

    if args.tiny_cpu_smoke:
        mx.set_default_device(mx.cpu)
    from mlx import nn

    from mlx2.runtime.multi_lora import write_adapter

    max_adapters = max(arms) if arms else 0
    if max_adapters < 1:
        raise SystemExit("at least one adapter arm is required")
    lora_root = Path(tempfile.mkdtemp(prefix="rm05-loras-"))
    engine = build_engine(args, lora_root, max_adapters)
    report = {"schema": "mlx2.multi-lora-e2e.v1", "args": vars(args), "arms": {}, "parity": {}}
    try:
        targets = tuple(args.targets.split(","))
        if args.tiny_cpu_smoke:
            targets = ("self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj")
        modules = dict(engine.adapter.model.named_modules())
        keys = sorted(
            name for name, module in modules.items()
            if name.endswith(targets) and isinstance(module, (nn.Linear, nn.QuantizedLinear))
        )
        if not keys:
            raise SystemExit(f"no Linear modules match {targets}")
        dims = {}
        for key in keys:
            module = modules[key]
            out_dims, packed = module.weight.shape
            in_dims = packed * 32 // module.bits if isinstance(module, nn.QuantizedLinear) else packed
            dims[key] = (in_dims, out_dims)
        names = [f"lora{i}" for i in range(max_adapters)]
        for i, name in enumerate(names):
            write_adapter(lora_root / name, keys=keys, dims=dims, rank=args.rank,
                          scale=2.0, seed=100 + i, dtype=mx.bfloat16 if not args.tiny_cpu_smoke else None)
            report.setdefault("registrations", []).append(engine.load_lora_adapter(name, name))
        report["keys"] = len(keys)
        order = [n for n in arms]
        for round_index in range(args.interleave):
            for count in order:
                result = run_arm(engine, args, names[:count])
                label = "base" if count == 0 else f"mix{count}"
                entry = report["arms"].setdefault(label, {"runs": []})
                entry["runs"].append({k: v for k, v in result.items() if k != "outputs"})
                if round_index == 0:
                    entry["first_outputs"] = result["outputs"]
                print(json.dumps({"round": round_index, "arm": label, "tok_s": round(result["tok_s"], 2),
                                  "mixed_forwards": result["mixed_forwards"]}), flush=True)
        for label, entry in report["arms"].items():
            entry["median_tok_s"] = statistics.median(run["tok_s"] for run in entry["runs"])
        # Parity: each adapter row of the widest mixed arm vs the same request alone.
        widest = f"mix{max_adapters}"
        mixed_rows = report["arms"][widest]["runs"][0]["rows"]
        mixed_outputs = report["arms"][widest].pop("first_outputs")
        for index, name in enumerate(mixed_rows):
            if name is None or name in report["parity"]:
                continue
            alone, _ = collect(engine.submit(request(args, index, name)))
            a, b = "".join(mixed_outputs[index]), "".join(alone)
            if not a or not b:
                raise SystemExit(f"parity gate: empty generated text for {name} (mixed={len(a)}, alone={len(b)})")
            first = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
            report["parity"][name] = {
                "identical": a == b,
                "first_divergence_char": first,
                "mixed_prefix": a[:120],
                "alone_prefix": b[:120],
            }
        # Control: a base row is also compared B=8 vs B=1, so a divergence
        # caused by bf16 reduction order at batch 8 is not read as a LoRA
        # parity failure.
        base_entry = report["arms"].get("base")
        if base_entry is not None and base_entry.get("first_outputs"):
            alone, _ = collect(engine.submit(request(args, 0, None)))
            a, b = "".join(base_entry["first_outputs"][0]), "".join(alone)
            first = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
            report["base_batch_control"] = {
                "identical": a == b,
                "first_divergence_char": first,
                "mixed_prefix": a[:120],
                "alone_prefix": b[:120],
            }
        for entry in report["arms"].values():
            entry.pop("first_outputs", None)
        base = report["arms"].get("base", {}).get("median_tok_s")
        top = report["arms"].get(widest, {}).get("median_tok_s")
        report["verdict"] = {
            "ratio_widest_over_base": (top / base) if base and top else None,
            "throughput_go": bool(base and top and top / base >= 0.85),
            "throughput_nogo": bool(base and top and top / base < 0.70),
            "parity_go": all(p["identical"] or (p["first_divergence_char"] or 0) >= args.parity_tokens * 3
                             for p in report["parity"].values()),
        }
        report["multi_lora_status"] = engine.multi_lora.status()
    finally:
        engine.close()
    text = json.dumps(report, indent=1, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(json.dumps(report["verdict"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
