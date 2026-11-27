#!/usr/bin/env python3
"""Multi-tenant multi-LoRA serving test: startup, adapter count, long tail, churn.

Asked by Pierre on 2026-11-26 after reading a Reddit post (GCMP, a VeRA-sized
LoRA replacement) whose serving claims were really about vLLM's multi-LoRA
slot design: slot memory eating the KV cache, serial adapter loading, a
throughput drop from 1 to 64 to 256 adapters, p99 TTFT under a long-tail
tenant mix, and slow tenant swaps. This harness asks the same questions of
mlx2's concurrent multi-LoRA (``mlx2.runtime.multi_lora``: registry unbounded,
``max_loras`` device slots with pin + LRU residency, ordinary route only).

One process = one engine configuration (rank, registered adapters, slots).
Phases, in order:

1. ``startup``    -- write N random adapters (7 projections where present:
                     q/k/v/o + gate/up/down), register all, time it; record
                     slot bytes, Metal memory and the engine's memory status.
2. ``reference``  -- B=1 greedy outputs for base and four adapters.
3. ``throughput`` -- closed loop: C concurrent requests, each to a random
                     adapter among the first K registered (K = 0 base, 1, 16,
                     64, 256 capped at N). Interleaved twice.
4. ``longtail``   -- open loop: Poisson arrivals at ``--load`` x the measured
                     K=min(64,N) request rate, Zipf(1.1) tenant popularity over
                     all N adapters, for ``--phase-s`` seconds.
5. ``churn``      -- open loop over a set of min(64, N) tenants; every
                     ``--churn-s`` seconds the oldest idle tenant is unloaded
                     and a never-seen adapter is loaded under a new name, then
                     sent a request at once (swap latency + first TTFT).
6. ``reload``     -- the reference outputs again, after all that eviction:
                     each must be byte-identical (width one, greedy).

Mechanism gates: adapter phases must move ``rows_adapter`` and
``delta_applications``; the base arm must not apply deltas; K > slots must
show ``slot_evictions``. A gate that fails is recorded and the phase marked
failed (the run continues so the night is not wasted).

Sanity/attribution evidence, no thermal hold; not publishable statistics.

``--tiny-cpu-smoke`` runs every phase on the CPU fixture model, seconds long.
``--summarize DIR`` writes ``REPORT.md`` from the per-config JSON files.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
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
    "Why does bread go stale faster in the fridge?",
    "Outline a three-day itinerary for Lisbon on a budget.",
    "How would you explain recursion using a set of nesting dolls?",
    "Compare TCP and UDP for a multiplayer game.",
    "Suggest names for a bakery that only sells sourdough.",
    "What should a beginner check before buying a used car?",
    "Explain the Monty Hall problem without equations.",
    "Write a haiku about a server room at night.",
]
TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model")
    p.add_argument("--checkpoint", help="<artifact>@<fingerprint> label for receipts")
    p.add_argument("--label", default="cfg")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--adapters", type=int, default=256, help="registered adapters N")
    p.add_argument("--slots", type=int, default=64, help="max_loras device slots")
    p.add_argument("--lanes", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--k-arms", default="0,1,16,64,256")
    p.add_argument("--interleave", type=int, default=2)
    p.add_argument("--phase-s", type=float, default=900.0)
    p.add_argument("--churn-s", type=float, default=20.0)
    p.add_argument("--load", type=float, default=0.6, help="open-loop rate as a fraction of closed-loop capacity")
    p.add_argument("--zipf", type=float, default=1.1)
    p.add_argument("--deadline-epoch", type=float, default=0.0, help="skip phases that would end after this")
    p.add_argument("--out", help="evidence directory (must be empty or absent)")
    p.add_argument("--lora-dir", help="where adapters are written (default: temp, removed at the end)")
    p.add_argument("--delta-precision", choices=("fp32", "adapter"),
                   help="multi-LoRA slot/delta precision (default: engine default, fp32)")
    p.add_argument("--delta-kernel", choices=("gather", "take", "auto"),
                   help="sets MLX2_MULTI_LORA_DELTA_KERNEL for this process (default: unset = gather)")
    p.add_argument("--take-max-rows", type=int, help="sets MLX2_MULTI_LORA_TAKE_MAX_ROWS (auto only)")
    p.add_argument("--phases", default="throughput,longtail,churn",
                   help="comma list of throughput,longtail,churn (startup/reference/reload always run)")
    p.add_argument("--i-own-the-gpu", action="store_true")
    p.add_argument("--tiny-cpu-smoke", action="store_true")
    p.add_argument("--summarize")
    return p.parse_args()


# ------------------------------------------------------------------ engine
def build_engine(args, lora_root):
    import os

    from mlx2.serving import ServingEngine

    if args.delta_kernel:
        os.environ["MLX2_MULTI_LORA_DELTA_KERNEL"] = args.delta_kernel
    if args.take_max_rows is not None:
        os.environ["MLX2_MULTI_LORA_TAKE_MAX_ROWS"] = str(args.take_max_rows)

    kwargs = dict(
        qualification_mode=True, mtp=False, lora_root=str(lora_root),
        max_loras=args.slots, max_lora_rank=args.rank, max_lanes=args.lanes,
        max_inflight=max(args.concurrency * 2, 64), coalesce_window_ms=50.0,
    )
    if args.delta_precision:
        kwargs["multi_lora_delta_precision"] = args.delta_precision
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


def body_for(args, index, model, max_tokens=None):
    if args.tiny_cpu_smoke:
        body = {"tokens": [(7 * index + k) % 120 + 1 for k in range(12)]}
    else:
        body = {"messages": [{"role": "user", "content": PROMPTS[index % len(PROMPTS)]}]}
    body.update(max_tokens=max_tokens or args.max_tokens, temperature=0)
    if model:
        body["model"] = model
    return body


def run_one(engine, body):
    """Submit and collect; return timing + text. Never raises."""
    t0 = time.perf_counter()
    rec = {"t_submit": t0, "ttft_s": None, "e2e_s": None, "tokens": 0, "error": None, "text": ""}
    try:
        job = engine.submit(body)
        pieces = []
        while True:
            event = job.events.get(timeout=1800)
            if "error" in event:
                rec["error"] = str(event["error"])[:300]
                break
            if "delta" in event:
                d = event["delta"]
                piece = (d.get("reasoning_content", "") or "") + (d.get("content", "") or "")
                if piece and rec["ttft_s"] is None:
                    rec["ttft_s"] = time.perf_counter() - t0
                pieces.append(piece)
            if "finish_reason" in event:
                rec["tokens"] = int((event.get("receipt") or {}).get("completion_tokens", 0))
                break
        rec["text"] = "".join(pieces)
    except Exception as exc:  # admission refusal, timeout, ...
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    rec["e2e_s"] = time.perf_counter() - t0
    return rec


def counts(engine):
    return dict(engine.multi_lora.status()["counts"])


def delta(before, after):
    return {k: after.get(k, 0) - before.get(k, 0) for k in set(after) | set(before) if after.get(k, 0) != before.get(k, 0)}


def pct(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values) - 1, max(0, math.ceil(q / 100 * len(values)) - 1))]


def latency_summary(recs):
    ok = [r for r in recs if not r["error"]]
    ttft = [r["ttft_s"] for r in ok]
    e2e = [r["e2e_s"] for r in ok]
    return {
        "requests": len(recs), "failed": len(recs) - len(ok),
        "errors": sorted({r["error"] for r in recs if r["error"]})[:5],
        "ttft_p50_s": pct(ttft, 50), "ttft_p95_s": pct(ttft, 95), "ttft_p99_s": pct(ttft, 99),
        "e2e_p50_s": pct(e2e, 50), "e2e_p99_s": pct(e2e, 99),
        "tokens": sum(r["tokens"] for r in ok),
    }


def memory_snapshot(engine):
    import mlx.core as mx

    snap = {"active_gib": mx.get_active_memory() / 2**30, "peak_gib": mx.get_peak_memory() / 2**30,
            "slot_gib": engine.multi_lora.slot_nbytes() / 2**30}
    status = engine.status()
    for key in ("memory", "kv", "apc", "cache_budget", "kv_budget"):
        if key in status:
            snap[f"status_{key}"] = status[key]
    return snap


# ------------------------------------------------------------------ phases
def phase_throughput(engine, args, names, report, rng):
    arms = sorted({min(int(k), len(names)) for k in args.k_arms.split(",") if k != ""})
    out = {}
    for rnd in range(args.interleave):
        for k in arms:
            before = counts(engine)
            pool = names[:k]
            bodies = [body_for(args, i + 97 * rnd, rng.choice(pool) if pool else None) for i in range(args.concurrency)]
            recs = [None] * len(bodies)
            t0 = time.perf_counter()

            def go(i):
                recs[i] = run_one(engine, bodies[i])

            threads = [threading.Thread(target=go, args=(i,)) for i in range(len(bodies))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall = time.perf_counter() - t0
            d = delta(before, counts(engine))
            summ = latency_summary(recs)
            gate = None
            if k and (d.get("rows_adapter", 0) <= 0):
                gate = f"adapter arm K={k} moved no adapter rows: {d}"
            if not k and d.get("rows_adapter", 0):
                gate = f"base arm ran adapter rows: {d}"
            # An eviction is only forced when the arm itself asks for more
            # distinct adapters than there are slots; 32 requests over 256
            # tenants touch at most 32, which fit in 64 slots (2026-11-27 night
            # run: 24 loads, 0 evictions, correct). Eviction under pressure is
            # exercised by the long-tail phase.
            distinct = len({b.get("model") for b in bodies if b.get("model")})
            if distinct > args.slots and not d.get("slot_evictions", 0):
                gate = f"{distinct} distinct adapters > slots={args.slots} but no slot_evictions: {d}"
            entry = out.setdefault(str(k), {"runs": []})
            entry["runs"].append({**summ, "distinct_adapters": distinct, "wall_s": wall, "tok_s": summ["tokens"] / wall,
                                  "req_s": (summ["requests"] - summ["failed"]) / wall, "counters": d, "gate": gate})
            print(json.dumps({"phase": "throughput", "K": k, "round": rnd, "tok_s": round(summ["tokens"] / wall, 2),
                              "failed": summ["failed"], "gate": gate}), flush=True)
    for entry in out.values():
        entry["median_tok_s"] = statistics.median(r["tok_s"] for r in entry["runs"])
        entry["median_req_s"] = statistics.median(r["req_s"] for r in entry["runs"])
        entry["gate_failures"] = [r["gate"] for r in entry["runs"] if r["gate"]]
    report["phases"]["throughput"] = out
    return out


def open_loop(engine, args, duration, rate, pick, rng, on_tick=None, tick_s=None):
    """Poisson arrivals for ``duration`` s; ``pick()`` -> adapter name. Returns records."""
    recs, threads, lock = [], [], threading.Lock()
    t_end = time.perf_counter() + duration
    next_tick = time.perf_counter() + tick_s if tick_s else None
    i = 0
    while True:
        now = time.perf_counter()
        if now >= t_end:
            break
        if next_tick and now >= next_tick:
            on_tick()
            next_tick += tick_s
        name = pick()
        body = body_for(args, i, name)
        i += 1

        def go(b=body, n=name):
            r = run_one(engine, b)
            r["adapter"] = n
            with lock:
                recs.append(r)

        t = threading.Thread(target=go, daemon=True)
        t.start()
        threads.append(t)
        wait = rng.expovariate(rate)
        stop = min(t_end, time.perf_counter() + wait)
        if next_tick:
            stop = min(stop, next_tick)
        time.sleep(max(0.0, stop - time.perf_counter()))
    for t in threads:
        t.join(timeout=1800)
    return recs


def phase_longtail(engine, args, names, rate, report, rng):
    weights = [1.0 / (i + 1) ** args.zipf for i in range(len(names))]
    total = sum(weights)
    cum, acc = [], 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)

    def pick():
        x = rng.random()
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return names[lo]

    before = counts(engine)
    mem0 = memory_snapshot(engine)
    recs = open_loop(engine, args, args.phase_s, rate, pick, rng)
    d = delta(before, counts(engine))
    head = set(names[:4])
    out = {
        "rate_req_s": rate, "duration_s": args.phase_s, "zipf": args.zipf,
        "all": latency_summary(recs),
        "busy_top4": latency_summary([r for r in recs if r["adapter"] in head]),
        "rare_rest": latency_summary([r for r in recs if r["adapter"] not in head]),
        "distinct_tenants_hit": len({r["adapter"] for r in recs}),
        "counters": d, "memory_before": mem0, "memory_after": memory_snapshot(engine),
        "gate": None if d.get("rows_adapter", 0) > 0 else f"no adapter rows: {d}",
    }
    report["phases"]["longtail"] = out
    print(json.dumps({"phase": "longtail", **{k: out["all"][k] for k in ("requests", "failed", "ttft_p50_s", "ttft_p99_s")}}), flush=True)


def phase_churn(engine, args, names, spares, rate, report, rng):
    live = list(names[: min(64, len(names))])
    keep = set(names[:4])  # reference adapters stay registered for the reload-parity check
    swaps, firsts, refused = [], [], 0
    first_threads = []

    def tick():
        nonlocal refused
        if not spares:
            return
        new_name, new_dir = spares.pop(0)
        for victim in [n for n in live if n not in keep]:
            try:
                t0 = time.perf_counter()
                engine.unload_lora_adapter(victim)
                t_unload = time.perf_counter() - t0
            except ValueError:
                refused += 1
                continue
            live.remove(victim)
            t1 = time.perf_counter()
            engine.load_lora_adapter(new_name, str(new_dir))
            t_load = time.perf_counter() - t1
            live.append(new_name)
            swaps.append({"victim": victim, "new": new_name, "unload_s": t_unload, "load_s": t_load,
                          "swap_s": t_unload + t_load})

            def first(n=new_name):
                r = run_one(engine, body_for(args, 7, n))
                r["adapter"] = n
                firsts.append(r)

            th = threading.Thread(target=first, daemon=True)
            th.start()
            first_threads.append(th)
            return

    before = counts(engine)
    recs = open_loop(engine, args, args.phase_s, rate, lambda: rng.choice(live), rng, on_tick=tick, tick_s=args.churn_s)
    for th in first_threads:
        th.join(timeout=1800)
    d = delta(before, counts(engine))
    out = {
        "rate_req_s": rate, "duration_s": args.phase_s, "churn_s": args.churn_s, "tenants": len(live),
        "traffic": latency_summary(recs), "new_tenant_first_request": latency_summary(firsts),
        "swaps": len(swaps), "unload_refused_in_use": refused,
        "swap_p50_s": pct([s["swap_s"] for s in swaps], 50), "swap_p95_s": pct([s["swap_s"] for s in swaps], 95),
        "load_p50_s": pct([s["load_s"] for s in swaps], 50),
        "counters": d, "memory_after": memory_snapshot(engine),
        "gate": None if swaps and d.get("registrations", 0) >= len(swaps) else f"swaps={len(swaps)} counters={d}",
    }
    report["phases"]["churn"] = out
    print(json.dumps({"phase": "churn", "swaps": len(swaps), "swap_p50_s": out["swap_p50_s"],
                      "first_ttft_p95_s": out["new_tenant_first_request"]["ttft_p95_s"]}), flush=True)


def reference(engine, args, names):
    out = {}
    for name in [None] + list(names[:4]):
        r = run_one(engine, body_for(args, 3, name, max_tokens=64))
        out[name or "<base>"] = {"text": r["text"], "error": r["error"]}
    return out


# ------------------------------------------------------------------ main
def find_fingerprint(obj, depth=0):
    """First artifact ``fingerprint`` string in the engine status (resolver shapes differ)."""
    if depth > 4:
        return None
    if isinstance(obj, dict):
        for key in ("artifact", "identity"):
            if key in obj:
                found = find_fingerprint(obj[key], depth + 1)
                if found:
                    return found
        value = obj.get("fingerprint")
        if isinstance(value, str) and value:
            return value
    return None


def git_head():
    try:
        return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def main():
    args = parse()
    if args.summarize:
        return summarize(Path(args.summarize))
    if not args.tiny_cpu_smoke:
        if not args.i_own_the_gpu:
            print("refusing to touch Metal without --i-own-the-gpu", file=sys.stderr)
            return 2
        if not (args.model and args.out):
            print("--model and --out are required", file=sys.stderr)
            return 2
    out_dir = Path(args.out) if args.out else None
    if out_dir and out_dir.exists() and any(out_dir.iterdir()):
        print(f"refusing to overwrite nonempty {out_dir}", file=sys.stderr)
        return 2
    import mlx.core as mx

    if args.tiny_cpu_smoke:
        mx.set_default_device(mx.cpu)
    from mlx import nn

    from mlx2.runtime.multi_lora import write_adapter

    rng = random.Random(20261126)
    lora_root = Path(args.lora_dir) if args.lora_dir else Path(tempfile.mkdtemp(prefix="tenancy-loras-"))
    lora_root.mkdir(parents=True, exist_ok=True)
    report = {"schema": "mlx2.multi-lora-tenancy.v1", "label": args.label, "args": vars(args),
              "checkpoint": args.checkpoint or "tiny-fixture", "source_revision": git_head(),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phases": {}, "skipped": []}

    def save():
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_dir / f"{args.label}.json.tmp"
            tmp.write_text(json.dumps(report, indent=1, default=str))
            tmp.replace(out_dir / f"{args.label}.json")

    def time_left(need_s):
        if args.deadline_epoch and time.time() + need_s > args.deadline_epoch:
            report["skipped"].append({"at": time.strftime("%H:%M:%S"), "need_s": need_s})
            return False
        return True

    t_engine = time.perf_counter()
    engine = build_engine(args, lora_root)
    report["engine_load_s"] = time.perf_counter() - t_engine
    try:
        targets = TARGETS if not args.tiny_cpu_smoke else ("self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj")
        modules = dict(engine.adapter.model.named_modules())
        keys = sorted(n for n, m in modules.items()
                      if n.endswith(targets) and isinstance(m, (nn.Linear, nn.QuantizedLinear)))
        if not keys:
            raise SystemExit(f"no Linear modules match {targets}")
        dims = {}
        for key in keys:
            m = modules[key]
            out_dims, packed = m.weight.shape
            dims[key] = (packed * 32 // m.bits if isinstance(m, nn.QuantizedLinear) else packed, out_dims)
        # A random rank-r delta grows ~sqrt(r): scale 2.0 at rank 8 gave coherent
        # text, but rank 32 produced gibberish (2026-11-27), which makes output
        # comparisons meaningless. Keep every rank at rank 8's delta strength.
        adapter_scale = 2.0 * (8 / args.rank) ** 0.5
        report["adapter_scale"] = adapter_scale
        n_spares = int(args.phase_s // args.churn_s) + 2 if "churn" in args.phases.split(",") else 0
        dtype = None if args.tiny_cpu_smoke else mx.bfloat16
        t0 = time.perf_counter()
        names = []
        for i in range(args.adapters + n_spares):
            write_adapter(lora_root / f"t{i:04d}", keys=keys, dims=dims, rank=args.rank, scale=adapter_scale,
                          seed=1000 + i, dtype=dtype)
        write_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        for i in range(args.adapters):
            engine.load_lora_adapter(f"t{i:04d}", f"t{i:04d}")
            names.append(f"t{i:04d}")
        reg_s = time.perf_counter() - t0
        spares = [(f"t{i:04d}", f"t{i:04d}") for i in range(args.adapters, args.adapters + n_spares)]
        one = engine.multi_lora.describe(names[0])
        report["phases"]["startup"] = {
            "keys": len(keys), "adapter_params": sum(dims[k][0] * args.rank + args.rank * dims[k][1] for k in keys),
            "adapter_nbytes": one.get("nbytes"), "write_s": write_s, "register_s": reg_s,
            "register_per_adapter_ms": 1000 * reg_s / max(1, args.adapters),
            "memory": memory_snapshot(engine),
        }
        status = engine.status()
        report["engine_identity"] = {k: status.get(k) for k in ("model", "model_revision", "artifact", "identity",
                                                                "route", "fingerprint") if k in status}
        fp = find_fingerprint(status)
        if fp and not args.tiny_cpu_smoke:
            observed = f"{Path(args.model).name}@{fp}"
            if args.checkpoint and args.checkpoint != observed:
                report["checkpoint_label_mismatch"] = args.checkpoint
            report["checkpoint"] = observed
        print(json.dumps({"phase": "startup", **{k: report["phases"]["startup"][k] for k in ("keys", "register_s")}}), flush=True)
        save()

        ref = reference(engine, args, names)
        report["phases"]["reference"] = ref
        tp = phase_throughput(engine, args, names, report, rng)
        save()
        k_cal = str(min(64, len(names)))
        cap = tp.get(k_cal, {}).get("median_req_s") or 0.1
        rate = max(0.02, args.load * cap)
        report["open_loop_rate_req_s"] = rate
        phases = set(args.phases.split(","))
        if "longtail" in phases and time_left(args.phase_s + 120):
            phase_longtail(engine, args, names, rate, report, rng)
            save()
        if "churn" in phases and time_left(args.phase_s + 120):
            phase_churn(engine, args, names, spares, rate, report, rng)
            save()
        # Reload parity: the reference adapters were evicted and reloaded many times by now.
        still = [n for n in names[:4] if engine.multi_lora.lookup(n)]
        again = reference(engine, args, still)
        parity = {}
        for key, val in again.items():
            a, b = ref.get(key, {}).get("text"), val["text"]
            first = None if a == b else next((i for i, (x, y) in enumerate(zip(a or "", b or "")) if x != y),
                                             min(len(a or ""), len(b or "")))
            parity[key] = {"identical": a == b and bool(a), "first_divergence_char": first,
                           "before": (a or "")[:100], "after": (b or "")[:100]}
        report["phases"]["reload_parity"] = parity
        report["multi_lora_status"] = engine.multi_lora.status()
        report["multi_lora_settings"] = engine.multi_lora.settings()
        report["slot_dtype"] = str(next(iter(engine.multi_lora.wrapped.values())).lora_a.dtype)
    except SystemExit as exc:
        report["fatal"] = str(exc)
        raise
    except Exception as exc:
        report["fatal"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save()
        engine.close()
        if not args.lora_dir:
            shutil.rmtree(lora_root, ignore_errors=True)
    print(json.dumps({"done": args.label, "parity": {k: v["identical"] for k, v in parity.items()}}))
    return 0


# ------------------------------------------------------------------ report
def fmt(v, nd=2):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def summarize(d: Path):
    # One subdirectory per config (<label>/<label>.json); the root also holds
    # queue receipts (gpu-receipt-*.json), which are not ours.
    files = []
    for p in sorted(d.glob("*/*.json")):
        try:
            if json.loads(p.read_text()).get("schema") == "mlx2.multi-lora-tenancy.v1":
                files.append(p)
        except (OSError, ValueError):
            continue
    lines = ["# Multi-LoRA tenancy test", ""]
    for p in files:
        r = json.loads(p.read_text())
        a = r["args"]
        lines += [f"## {r['label']}: rank {a['rank']}, {a['adapters']} adapters, {a['slots']} slots", "",
                  f"- checkpoint `{r['checkpoint']}`, mlx2 `{r['source_revision']}`, {r['started']} → {r.get('finished')}",
                  f"- delta precision `{(r.get('multi_lora_settings') or {}).get('delta_precision', a.get('delta_precision') or 'adapter')}`, slot dtype `{r.get('slot_dtype')}`, "
                  f"delta kernel `{(r.get('multi_lora_settings') or {}).get('delta_kernel', 'gather')}`, "
                  f"adapter scale {fmt(r.get('adapter_scale', 2.0))}, "
                  f"take applications {((r.get('multi_lora_status') or {}).get('counts') or {}).get('take_applications', 0)}"]
        if r.get("fatal"):
            lines.append(f"- **FATAL:** {r['fatal']}")
        if r.get("skipped"):
            lines.append(f"- skipped phases (deadline): {len(r['skipped'])}")
        st = r["phases"].get("startup")
        if st:
            lines += [f"- engine load {fmt(r.get('engine_load_s'),1)} s; {st['keys']} wrapped projections; "
                      f"{st['adapter_params']/1e6:.1f}M params/adapter; register {a['adapters']} adapters "
                      f"{fmt(st['register_s'],1)} s ({fmt(st['register_per_adapter_ms'],1)} ms each); "
                      f"slots {fmt(st['memory']['slot_gib'])} GiB, Metal active {fmt(st['memory']['active_gib'])} GiB", ""]
        tp = r["phases"].get("throughput")
        if tp:
            base = tp.get("0", {}).get("median_tok_s")
            lines += ["| adapters K | tok/s (median) | vs base | TTFT p95 s | failed | gate |", "|---|---|---|---|---|---|"]
            for k in sorted(tp, key=int):
                e = tp[k]
                ttft = statistics.median([x["ttft_p95_s"] for x in e["runs"] if x["ttft_p95_s"] is not None] or [0])
                lines.append(f"| {k} | {fmt(e['median_tok_s'],1)} | {fmt(e['median_tok_s']/base if base else None)} | "
                             f"{fmt(ttft)} | {sum(x['failed'] for x in e['runs'])} | {'; '.join(e['gate_failures']) or 'ok'} |")
            lines.append("")
        lt = r["phases"].get("longtail")
        if lt:
            lines += [f"**Long tail** ({fmt(lt['rate_req_s'],3)} req/s, {int(lt['duration_s'])} s, Zipf {lt['zipf']}, "
                      f"{lt['distinct_tenants_hit']} tenants hit): evictions {lt['counters'].get('slot_evictions',0)}, "
                      f"slot loads {lt['counters'].get('slot_loads',0)}, deferred {lt['counters'].get('slot_deferred',0)}", "",
                      "| traffic | requests | failed | TTFT p50 | TTFT p99 | e2e p99 |", "|---|---|---|---|---|---|"]
            for key in ("all", "busy_top4", "rare_rest"):
                s = lt[key]
                lines.append(f"| {key} | {s['requests']} | {s['failed']} | {fmt(s['ttft_p50_s'])} | {fmt(s['ttft_p99_s'])} | {fmt(s['e2e_p99_s'])} |")
            lines.append("")
        ch = r["phases"].get("churn")
        if ch:
            t, f = ch["traffic"], ch["new_tenant_first_request"]
            lines += [f"**Churn** (one tenant replaced every {ch['churn_s']} s, {ch['tenants']} live): {ch['swaps']} swaps, "
                      f"swap p50/p95 {fmt(ch['swap_p50_s'],3)}/{fmt(ch['swap_p95_s'],3)} s, unload refused (in use) {ch['unload_refused_in_use']}; "
                      f"new tenant first TTFT p50/p95 {fmt(f['ttft_p50_s'])}/{fmt(f['ttft_p95_s'])} s; "
                      f"traffic TTFT p50/p99 {fmt(t['ttft_p50_s'])}/{fmt(t['ttft_p99_s'])} s, failed {t['failed'] + f['failed']}", ""]
        rp = r["phases"].get("reload_parity")
        if rp:
            bad = [k for k, v in rp.items() if not v["identical"]]
            lines += ["**Reload parity** (B=1 greedy, before vs after eviction): "
                      + ("all identical" if not bad else f"DIVERGED: {', '.join(bad)}"), ""]
    # Cross-config output identity: the same B=1 greedy reference requests
    # (base + first four adapters, same seeds) in every config.
    refs = [(json.loads(p.read_text())["label"], json.loads(p.read_text())["phases"].get("reference") or {}) for p in files]
    if len(refs) > 1:
        lines += ["## Reference outputs across configs (B=1 greedy, vs the first config)", "",
                  "| config | " + " | ".join(refs[0][1]) + " |", "|---|" + "---|" * len(refs[0][1])]
        for label, ref in refs:
            cells = []
            for key, val in refs[0][1].items():
                a_text, b_text = val.get("text") or "", (ref.get(key) or {}).get("text") or ""
                first = None if a_text == b_text else next(
                    (i for i, (x, y) in enumerate(zip(a_text, b_text)) if x != y), min(len(a_text), len(b_text)))
                cells.append("same" if first is None else f"differs @{first}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    (d / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(d / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
