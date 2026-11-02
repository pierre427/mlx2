"""GLM-5.3 direct B1 decode, per-step timing at full context, serial and pipelined.

No server: the model and its own caches only, so a serving scheduler, APC or
transport cannot add time. For each context, prefill a realistic prompt in
2048-token chunks, then decode greedily ``--steps`` tokens twice from fresh
copies of nothing but the same prefill: ``serial`` (build, eval, read the token
each step) and ``pipelined`` (mlx-lm style: build and async_eval step N+1
before reading token N). Per-step wall gaps are reported for the first 64 and
the last 256 steps, so a cost that grows within one request (accumulating
graph, thermal ramp) separates from one set by context. Research only.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_ordinary_profile import CHECKPOINT, MODEL, prompt_ids  # noqa: E402


def die_temp():
    # Same sampler the ladder's cooldown uses, when available (best effort).
    try:
        out = subprocess.run(["/usr/bin/pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
        return out.strip().splitlines()[-1] if out else None
    except Exception:  # noqa: BLE001
        return None


def summary(gaps):
    def p50(values):
        values = sorted(values)
        return values[len(values) // 2] * 1e3 if values else None
    return {"first_64_p50_ms": p50(gaps[:64]), "last_256_p50_ms": p50(gaps[-256:]),
            "all_p50_ms": p50(gaps), "tok_s": len(gaps) / sum(gaps) if gaps else None}


def phases(gaps, start):
    """Median step gap by token-position phase: pool completion (mod 4), KV
    growth block (mod 256, first 8 positions after a boundary vs the rest),
    and step parity, to expose a periodic cost hidden in the p50."""
    def p50(values):
        values = sorted(values)
        return values[len(values) // 2] * 1e3 if values else None
    pos = [start + i for i in range(len(gaps))]
    return {
        "mod4": {r: p50([g for g, p in zip(gaps, pos) if p % 4 == r]) for r in range(4)},
        "after_256_boundary_8": p50([g for g, p in zip(gaps, pos) if p % 256 < 8]),
        "rest_of_256_block": p50([g for g, p in zip(gaps, pos) if p % 256 >= 8]),
        "parity": {r: p50([g for g, p in zip(gaps, pos) if p % 2 == r]) for r in range(2)},
        "over_50ms": [(p, round(g * 1e3, 1)) for g, p in zip(gaps, pos) if g > 0.05],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 32768])
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    info = mx.device_info()
    if info.get("max_recommended_working_set_size"):
        mx.set_wired_limit(info["max_recommended_working_set_size"])
    policy = json.loads(args.policy.read_text())
    adapter = Glm53GGUFAdapter(MODEL, execution_policy=policy)
    model = adapter.model
    record = {"schema": "mlx2.glm53-decode-steady.v1", "checkpoint": CHECKPOINT, "source_revision": args.source_revision,
              "host": platform.node(), "mlx": mx.__version__, "execution_policy": policy, "steps": args.steps,
              "cells": []}

    def prefill(ids):
        cache = model.make_cache()
        for begin in range(0, len(ids), 2048):
            piece = mx.array([ids[begin:begin + 2048]])
            with model._prompt_scope(piece.shape[1]):
                logits = model(piece, cache, logits_to_keep=1)
            mx.eval(logits, [c.state for c in cache])
        return cache, int(mx.argmax(logits[0, -1]).item())

    for tokens in args.tokens:
        ids = prompt_ids(adapter, tokens)
        cell = {"tokens": len(ids)}
        for mode in ("serial", "pipelined", "pipelined_nogc"):
            cache, token = prefill(ids)
            temp_before = die_temp()
            gaps, out, builds, waits, gcs = [], [], [], [], []
            gc_events = []
            gc_hook = lambda phase, info: gc_events.append((phase, time.perf_counter()))  # noqa: E731
            gc.callbacks.append(gc_hook)
            if mode == "pipelined_nogc":
                gc.collect()
                gc.disable()
            if mode == "serial":
                last = time.perf_counter()
                for _ in range(args.steps):
                    logits = model(mx.array([[token]]), cache)
                    token = int(mx.argmax(logits[0, -1]).item())
                    now = time.perf_counter()
                    gaps.append(now - last)
                    last = now
                    out.append(token)
            else:
                y = mx.argmax(model(mx.array([[token]]), cache)[0, -1])
                mx.async_eval(y)
                last = time.perf_counter()
                for _ in range(args.steps):
                    seen = len(gc_events)
                    nxt = mx.argmax(model(y.reshape(1, 1), cache)[0, -1])
                    mx.async_eval(nxt)
                    built = time.perf_counter()
                    out.append(int(y.item()))
                    now = time.perf_counter()
                    gaps.append(now - last)
                    builds.append(built - last)
                    waits.append(now - built)
                    gcs.append(len(gc_events) > seen)
                    last = now
                    y = nxt
            gc.enable()
            gc.callbacks.remove(gc_hook)
            start = len(ids) + 1  # absolute position of the first timed step's input token
            cell[mode] = {**summary(gaps), "temp_before": temp_before, "temp_after": die_temp(),
                          "output_head": out[:16], "phase": phases(gaps, start),
                          "gaps_ms": [round(g * 1e3, 3) for g in gaps]}
            if builds:
                # Host (graph build + async_eval) vs exposed GPU wait per step,
                # and whether a Python GC pass ran inside the step.
                slow = [i for i, g in enumerate(gaps) if g > 0.05]
                cell[mode].update(
                    build_p50_ms=sorted(builds)[len(builds) // 2] * 1e3,
                    wait_p50_ms=sorted(waits)[len(waits) // 2] * 1e3,
                    slow_steps=[{"pos": start + i, "gap_ms": round(gaps[i] * 1e3, 1),
                                 "build_ms": round(builds[i] * 1e3, 1), "wait_ms": round(waits[i] * 1e3, 1),
                                 "gc": gcs[i]} for i in slow],
                    gc_steps=sum(gcs),
                    builds_ms=[round(b * 1e3, 3) for b in builds],
                    waits_ms=[round(w * 1e3, 3) for w in waits])
            del cache
            mx.clear_cache()
            print(json.dumps({"tokens": len(ids), "mode": mode,
                              **{k: (round(v, 2) if isinstance(v, float) else v)
                                 for k, v in cell[mode].items()
                                 if k not in ("output_head", "gaps_ms", "builds_ms", "waits_ms")}}), flush=True)
        cell["same_tokens"] = (cell["serial"]["output_head"] == cell["pipelined"]["output_head"]
                               == cell["pipelined_nogc"]["output_head"])
        record["cells"].append(cell)
    (args.out / "decode-steady.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
