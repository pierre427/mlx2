#!/usr/bin/env python3
"""GPU validation driver for MoE expert disk streaming.

One process = one arm. Loads a real MoE adapter, optionally installs expert
streaming, runs a fixed greedy decode, and writes every step's pre-sampling
logit vector plus the streaming counters.

Arms are compared offline by ``--compare``: argmax equality can hide a cache
bug that perturbs a logit below the decision margin, so the gate is the max
absolute logit delta across all steps, not token equality.

Wall-clock numbers printed here are NOT benchmarks. A streamed run's timing
is diagnostic only and must never reach a perf receipt.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run_arm(args):
    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.runtime.weight_stream import install_expert_streaming

    t0 = time.perf_counter()
    adapter = Qwen3635BA3BAdapter(args.model)
    load_s = time.perf_counter() - t0
    after_load = mx.get_active_memory()

    manager = collector = None
    if args.mode == "streamed":
        collector = None
        if args.trace:
            from mlx2.runtime.expert_atlas import AtlasCollector

            collector = AtlasCollector(
                args.model, sink=args.atlas, trace_path=args.trace, checkpoint_every=0
            )
        t1 = time.perf_counter()
        manager = install_expert_streaming(
            adapter.model,
            args.model,
            ceiling_bytes=int(args.cache_gib * (1 << 30)),
            top_k=args.top_k,
            read_workers=args.read_workers,
            collector=collector,
        )
        install_s = time.perf_counter() - t1
        print(f"[install] {install_s:.2f}s  plan={json.dumps(manager.plan.as_dict())}")
    mx.clear_cache()
    after_install = mx.get_active_memory()

    tokenizer = adapter.tokenizer
    ids = tokenizer.encode(args.prompt)
    model = adapter.model
    cache = model.make_cache()

    step_logits = []
    tokens = []
    t2 = time.perf_counter()
    out = model(mx.array([ids]), cache=cache)
    logits = out[:, -1, :].astype(mx.float32)
    mx.eval(logits)
    for _ in range(args.tokens):
        step_logits.append(np.array(logits)[0])
        nxt = int(mx.argmax(logits, axis=-1).item())
        tokens.append(nxt)
        out = model(mx.array([[nxt]]), cache=cache)
        logits = out[:, -1, :].astype(mx.float32)
        mx.eval(logits)
    decode_s = time.perf_counter() - t2

    record = {
        "mode": args.mode,
        "model": str(args.model),
        "prompt": args.prompt,
        "tokens": tokens,
        "text": tokenizer.decode(tokens),
        "load_seconds": round(load_s, 3),
        # Diagnostic only. A streamed run's timing is not a benchmark.
        "decode_seconds_NOT_A_BENCHMARK": round(decode_s, 3),
        "active_bytes_after_load": int(after_load),
        "active_bytes_after_install": int(after_install),
        "peak_bytes": int(mx.get_peak_memory()),
        "cache_gib": args.cache_gib if args.mode == "streamed" else None,
    }
    if manager is not None:
        record["counters"] = manager.counters()
        record["plan"] = manager.plan.as_dict()
        if collector is not None:
            collector.close()
        manager.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path.with_suffix(".logits.npy"), np.stack(step_logits))
    out_path.write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k != "text"}, indent=2))
    return 0


def compare(args):
    import numpy as np

    arms = {}
    for spec in args.compare:
        (name, path) = spec.split("=", 1)
        meta = json.loads(Path(path).read_text())
        logits = np.load(Path(path).with_suffix(".logits.npy"))
        arms[name] = (meta, logits)

    (ref_name, (ref_meta, ref_logits)) = next(iter(arms.items()))
    report = {
        "schema": "mlx2.expert-streaming-bit-identity.v1",
        "reference": ref_name,
        "steps": int(ref_logits.shape[0]),
        "vocab": int(ref_logits.shape[1]),
        "arms": {},
        "note": (
            "Gate is max absolute logit delta, not token equality: argmax "
            "agreement can hide a cache bug that moves a logit below the "
            "decision margin. Timings are diagnostic, never benchmarks."
        ),
    }
    ok = True
    for (name, (meta, logits)) in arms.items():
        delta = float(np.max(np.abs(logits - ref_logits))) if logits.shape == ref_logits.shape else float("nan")
        identical = bool(logits.shape == ref_logits.shape and np.array_equal(logits, ref_logits))
        same_tokens = meta["tokens"] == ref_meta["tokens"]
        counters = meta.get("counters") or {}
        report["arms"][name] = {
            "mode": meta["mode"],
            "cache_gib": meta.get("cache_gib"),
            "max_abs_logit_delta": delta,
            "bit_identical": identical,
            "tokens_match": same_tokens,
            "page_ins": counters.get("stream_page_ins_total"),
            "evictions": counters.get("stream_evictions_total"),
            "hits": counters.get("stream_expert_hits_total"),
            "misses": counters.get("stream_expert_misses_total"),
            "resident_bytes": counters.get("stream_resident_bytes"),
            "decode_seconds_NOT_A_BENCHMARK": meta.get(
                "decode_seconds_NOT_A_BENCHMARK"
            ),
        }
        if not (identical and same_tokens):
            ok = False

    evictions = {
        name: row["evictions"]
        for (name, row) in report["arms"].items()
        if row["evictions"] is not None
    }
    # Check this BEFORE reading anything into bit-identity: if the cache
    # never evicted, the streamed arms were effectively resident and the
    # identity result proves nothing.
    report["cache_actually_evicted"] = bool(evictions and max(evictions.values()) > 0)
    report["eviction_counts_differ"] = bool(len(set(evictions.values())) > 1)
    report["verdict"] = (
        "PASS: bit-identical across arms, and the cache demonstrably evicted"
        if ok and report["cache_actually_evicted"] and report["eviction_counts_differ"]
        else "INCONCLUSIVE: cache did not evict differently across arms"
        if ok
        else "FAIL: output diverged"
    )
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model")
    p.add_argument("--mode", choices=("resident", "streamed"), default="resident")
    p.add_argument("--cache-gib", type=float, default=4.0)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--read-workers", type=int, default=16)
    p.add_argument("--tokens", type=int, default=24)
    p.add_argument("--prompt", default="Explain in one paragraph why unified memory changes how large models are served.")
    p.add_argument("--atlas", default=None)
    p.add_argument("--trace", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--compare", nargs="*", default=None,
                   help="name=path.json pairs; first is the reference arm")
    args = p.parse_args(argv)
    if args.compare:
        return compare(args)
    if not args.model or not args.out:
        p.error("--model and --out are required for an arm run")
    return run_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())
