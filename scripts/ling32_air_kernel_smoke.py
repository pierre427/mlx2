#!/usr/bin/env python3
"""Ling-3.2-air kernel smoke: kda_fused and routed_moe_decode on the real checkpoint.

Sanity/attribution only (no thermal hold, not qualification, not publishable
statistics). One resident artifact; arms flip the adapter-bound per-module
switches in place (``KimiDeltaAttention.ling_kda_fused``,
``SparseMoeBlock.ling_routed_decode``), so the arms share weights and differ
only in the decode kernels:

    off, kda_fused, routed_moe_decode, both

Per arm and prompt: greedy decode of ``--decode-tokens`` tokens; every decode
step's logit row is compared bitwise with the ``off`` arm (prefill rows are
not compared: both kernels are B1 one-token decode paths). Decode tok/s is the
median of ``--reps`` timed decode loops after one warm-up. Engagement comes
from the kernels' own counters.

Pass (per kernel arm): all tokens and all decode logit rows bitwise equal to
``off``; the kernel engaged on decode steps; decode no slower than ``off`` by
more than ``--max-slowdown`` (default 2 %). Writes ``<out>/smoke.json``
atomically; refuses a nonempty ``<out>``. Exit 0 only if every kernel arm passes.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

PROMPTS = (
    "Explain in two sentences why the sky looks blue during the day.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the trip in minutes?",
)
ARMS = {
    "off": (False, False),
    "kda_fused": (True, False),
    "routed_moe_decode": (False, True),
    "both": (True, True),
}


def atomic_json(path: Path, value) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-slowdown", type=float, default=0.02)
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is nonempty; refuse")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from mlx2.adapters.ling32_air import Ling32AirAdapter
    from mlx2.runtime.models import ling_routed_decode
    from mlx2.runtime.models.bailing_hybrid import KimiDeltaAttention, SparseMoeBlock

    started = time.time()
    adapter = Ling32AirAdapter(str(args.model))
    model = adapter.model
    kda = [m for _, m in model.named_modules() if isinstance(m, KimiDeltaAttention)]
    moe = [m for _, m in model.named_modules() if isinstance(m, SparseMoeBlock)]
    # The shards are memory-mapped: weights page in on first use. Read every
    # parameter once so all paging (and the idle-page eviction it causes)
    # happens inside the load phase; then mark it for the caller's swap guard,
    # which applies its strict limit from here on.
    from mlx.utils import tree_flatten

    t0 = time.time()
    touched = 0
    for _name, value in tree_flatten(model.parameters()):
        mx.eval(mx.sum(value.view(mx.uint8) if value.dtype != mx.bool_ else value))
        touched += value.nbytes
    page_in = {"seconds": round(time.time() - t0, 1), "bytes": touched}
    (args.out / ".loaded").write_text(f"{time.time():.3f}\n")
    receipt = {"checkpoint": f"{args.model.name}@sha256:{adapter.identity['fingerprint']}",
               "load_seconds": round(time.time() - started, 1), "page_in": page_in,
               "kda_layers": len(kda),
               "moe_layers": len(moe), "arms": {}}

    def set_arm(kda_on, routed_on):
        if kda_on:
            from mlx2.runtime.models.ling_kda_fused import served_spellings

            adapter.kda_fused_spellings = served_spellings()
        for module in kda:
            module.ling_kda_fused = kda_on
            module.ling_calls.clear()
        for module in moe:
            module.ling_routed_decode = routed_on
        ling_routed_decode.status(reset=True)

    def run_prompt(tokens):
        cache = model.make_cache()
        logits = model(mx.array([tokens]), cache)
        mx.eval(logits)
        out, rows = [], []
        t0 = time.perf_counter()
        for _ in range(args.decode_tokens):
            token = int(mx.argmax(logits[0, -1]).item())
            out.append(token)
            logits = model(mx.array([[token]]), cache)
            row = logits[0, -1]
            mx.eval(row)
            rows.append(row)
        return out, rows, time.perf_counter() - t0

    prompts = []
    for prompt in PROMPTS:
        request = {"messages": [{"role": "user", "content": prompt}], "enable_thinking": False}
        prompts.append(list(adapter.prompt_tokens(request)))
    reference = {}
    try:
        for arm, (kda_on, routed_on) in ARMS.items():
            set_arm(kda_on, routed_on)
            run_prompt(prompts[0][:16])  # warm-up (kernel compile, probes)
            set_arm(kda_on, routed_on)
            rows_out, seconds = [], []
            for index, tokens in enumerate(prompts):
                out, rows, elapsed = run_prompt(tokens)
                times = [elapsed] + [run_prompt(tokens)[2] for _ in range(args.reps - 1)]
                seconds.append(statistics.median(times))
                if arm == "off":
                    reference[index] = (out, rows)
                    tokens_equal, bitwise = True, len(rows)
                else:
                    ref_out, ref_rows = reference[index]
                    tokens_equal = out == ref_out
                    bitwise = sum(bool(mx.array_equal(a, b).item()) for a, b in zip(rows, ref_rows))
                rows_out.append({"prompt": index, "tokens_equal": tokens_equal,
                                 "logits_bitwise_steps": bitwise, "steps": len(rows),
                                 "decode_tps": round(len(rows) / seconds[-1], 2),
                                 "text": adapter.tokenizer.decode(out) if arm == "off" else None})
            calls = Counter()
            for module in kda:
                calls.update(module.ling_calls)
            receipt["arms"][arm] = {
                "rows": rows_out,
                "decode_tps_median": round(statistics.median(r["decode_tps"] for r in rows_out), 2),
                "kda_calls": dict(calls),
                "routed_status": ling_routed_decode.status(),
            }
        base = receipt["arms"]["off"]["decode_tps_median"]
        verdicts = {}
        for arm in ("kda_fused", "routed_moe_decode", "both"):
            row = receipt["arms"][arm]
            exact = all(r["tokens_equal"] and r["logits_bitwise_steps"] == r["steps"] for r in row["rows"])
            kda_engaged = row["kda_calls"].get("kda_fused", 0) > 0
            routed_engaged = (row["routed_status"] or {}).get("engaged", 0) > 0
            engaged = {"kda_fused": kda_engaged, "routed_moe_decode": routed_engaged,
                       "both": kda_engaged and routed_engaged}[arm]
            speed = row["decode_tps_median"] / base - 1 if base else None
            verdicts[arm] = {"exact": exact, "engaged": engaged, "decode_speedup": speed,
                             "passed": bool(exact and engaged and speed is not None
                                            and speed >= -args.max_slowdown)}
        receipt["verdicts"] = verdicts
        receipt["peak_memory_gib"] = round(mx.get_peak_memory() / 2**30, 2)
    finally:
        receipt["seconds"] = round(time.time() - started, 1)
        atomic_json(args.out / "smoke.json", receipt)
        adapter.close()
    print(json.dumps(receipt.get("verdicts"), indent=1))
    return 0 if all(v["passed"] for v in receipt.get("verdicts", {}).values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
