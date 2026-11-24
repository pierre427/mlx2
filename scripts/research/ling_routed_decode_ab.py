#!/usr/bin/env python3
"""``ling_optimizations.routed_moe_decode`` on a real Ling checkpoint (GPU).

Original mlx2 script (2026-11-23). Sanity/attribution, never qualification and
never publishable statistics (no thermal hold). One process, one resident
artifact, run under the caller's GPU lease:

1. ``parity``: every SparseMoeBlock input of one real decode step (after a
   repository-text prefix) is captured; ``ling_routed_decode.self_check``
   compares the routed kernels with ``SwitchGLU`` bitwise per layer.
2. ``tokens``: per prompt, two caches prefilled identically (prefill is
   multi-row, the routed route declines), then ``steps`` greedy steps per arm;
   tokens and logits compared bitwise per step; at the first argmax
   divergence, the top-8 log probabilities of both arms at that position only.
3. ``ab``: one cache per arm per context, arms interleaved ABAB in 16-step
   blocks for ``rounds`` rounds, median ms/step per arm (whole-model decode,
   ``mx.set_wired_limit`` at the recommended working set).
4. ``premise``: per decode step, wall time vs the time the host waits in
   ``mx.eval`` (a whole-model view of whether decode waits on anything but the
   GPU), plus the routed engagement counters.

The adapter loads once with ``routed_moe_decode`` selected (identity/layout
bound); the off arm flips every SparseMoeBlock's flag in-process. Receipt:
``<run dir>/result.json`` (atomic); a nonempty run dir is refused.

Usage: ling_routed_decode_ab.py --model DIR --out DIR [--adapter ling32_air|ling3_flash]
       [--contexts 1024,8192] [--steps 64] [--rounds 3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 512
BLOCK = 16
TOPN = 8
PROMPTS = (
    "Explain in two sentences why the sky looks blue during the day.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the trip in minutes?",
)


def atomic_json(path: Path, value) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, path)


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          check=False).stdout.strip()


def swap():
    return subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True,
                          check=False).stdout.strip()


def corpus(tokenizer, n):
    parts = []
    for pattern in ("docs/**/*.md", "src/mlx2/**/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            parts.append(path.read_text(errors="replace"))
            if len(parts) % 25 == 0:
                ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
                if len(ids) >= n:
                    return list(ids[:n])
    raise SystemExit("corpus too small")


def blocks(model):
    from mlx2.runtime.models.bailing_hybrid import SparseMoeBlock

    return [(name, m) for name, m in model.named_modules() if isinstance(m, SparseMoeBlock)]


def set_arm(model, routed: bool):
    for _, block in blocks(model):
        block.ling_routed_decode = routed


def prefill(model, ids, mx):
    cache = model.make_cache()
    logits = None
    for s in range(0, len(ids), CHUNK):
        logits = model(mx.array([ids[s:s + CHUNK]]), cache=cache)
        mx.eval(logits)
    return cache, logits


def parity_phase(model, ids, mx):
    from mlx2.runtime.models import bailing_hybrid as bh
    from mlx2.runtime.models import ling_routed_decode as routed

    set_arm(model, False)
    cache, logits = prefill(model, ids, mx)
    token = int(mx.argmax(logits[0, -1]).item())
    captured = []
    original = bh.SparseMoeBlock.__call__

    def hook(self, x):
        indices, _ = self.gate(x)
        captured.append((self, x, indices))
        return original(self, x)

    bh.SparseMoeBlock.__call__ = hook
    try:
        mx.eval(model(mx.array([[token]]), cache=cache))
    finally:
        bh.SparseMoeBlock.__call__ = original
    names = {id(m): name for name, m in blocks(model)}
    rows = []
    for block, x, indices in captured:
        mx.eval(x, indices)
        check = routed.self_check(block.switch_mlp, x, indices)
        rows.append({"block": names.get(id(block)), "clamp": block.expert_swiglu_limit, **check,
                     **({} if check["engaged"] else {"refusal": routed.refusal(block.switch_mlp, x, indices)})})
    return {"visits": len(rows), "engaged": sum(r["engaged"] for r in rows),
            "exact": sum(bool(r.get("exact")) for r in rows), "rows": rows}


def tokens_phase(model, adapter, steps, mx):
    out = []
    for prompt in PROMPTS:
        request = {"messages": [{"role": "user", "content": prompt}], "enable_thinking": False}
        ids = list(adapter.prompt_tokens(request))
        arms = {}
        for name, flag in (("off", False), ("routed", True)):
            set_arm(model, flag)
            cache, logits = prefill(model, ids, mx)
            toks, rows = [], []
            for _ in range(steps):
                row = logits[0, -1].astype(mx.float32)
                token = int(mx.argmax(row).item())
                toks.append(token)
                rows.append(row)
                logits = model(mx.array([[token]]), cache=cache)
            mx.eval(rows)
            arms[name] = (toks, rows)
        (t_off, r_off), (t_on, r_on) = arms["off"], arms["routed"]
        bitwise = [bool(mx.array_equal(a.view(mx.uint32), b.view(mx.uint32)).item())
                   for a, b in zip(r_off, r_on)]
        first = next((i for i, (a, b) in enumerate(zip(t_off, t_on)) if a != b), None)
        record = {"prompt": prompt, "prompt_tokens": len(ids), "steps": steps,
                  "tokens_equal": t_off == t_on, "logits_bitwise_steps": sum(bitwise),
                  "first_logit_mismatch": next((i for i, b in enumerate(bitwise) if not b), None),
                  "first_token_divergence": first, "text_off": adapter.tokenizer.decode(t_off)}
        if first is not None:
            window = {}
            for name, rows in (("off", r_off), ("routed", r_on)):
                lp = rows[first] - mx.logsumexp(rows[first])
                top = mx.argsort(-lp)[:TOPN]
                window[name] = [[int(t), float(lp[t].item())] for t in top.tolist()]
            record["divergence_window_top8_logprobs"] = window
            record["text_routed"] = adapter.tokenizer.decode(t_on)
        out.append(record)
    return out


def ab_phase(model, ids_by_ctx, rounds, mx):
    result = {}
    for ctx, ids in ids_by_ctx.items():
        caches, logits = {}, {}
        for name, flag in (("off", False), ("routed", True)):
            set_arm(model, flag)
            caches[name], last = prefill(model, ids, mx)
            logits[name] = int(mx.argmax(last[0, -1]).item())
        times = {"off": [], "routed": []}
        waits = {"off": [], "routed": []}
        order = ["off", "routed"]
        for r in range(rounds * 2):
            name = order[r % 2] if (r // 2) % 2 == 0 else order[1 - r % 2]  # ABBA rotation
            set_arm(model, name == "routed")
            token = logits[name]
            for _ in range(BLOCK):
                t0 = time.perf_counter()
                out = model(mx.array([[token]]), cache=caches[name])
                t1 = time.perf_counter()
                token = int(mx.argmax(out[0, -1]).item())  # the one host sync per step
                t2 = time.perf_counter()
                times[name].append((t2 - t0) * 1e3)
                waits[name].append((t2 - t1) * 1e3)
            logits[name] = token
        result[str(ctx)] = {
            arm: {"steps": len(times[arm]), "median_ms_per_step": statistics.median(times[arm]),
                  "median_graph_build_ms": statistics.median(
                      [t - w for t, w in zip(times[arm], waits[arm])]),
                  "median_eval_wait_ms": statistics.median(waits[arm])}
            for arm in times
        }
        off, on = (result[str(ctx)][a]["median_ms_per_step"] for a in ("off", "routed"))
        result[str(ctx)]["routed_speedup"] = off / on
        del caches
        mx.clear_cache()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", choices=("ling32_air", "ling3_flash"), default="ling32_air")
    parser.add_argument("--contexts", default="1024,8192")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is nonempty; refuse")
    args.out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from mlx2.runtime.models import ling_routed_decode as routed

    mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))
    policy = {"ling_optimizations": {"routed_moe_decode": True}}
    started = time.time()
    if args.adapter == "ling32_air":
        from mlx2.adapters.ling32_air import Ling32AirAdapter as Adapter
    else:
        from mlx2.adapters.ling3_flash import Ling3FlashAdapter as Adapter
    adapter = Adapter(str(args.model), execution_policy=policy)
    model = adapter.model
    conversion = args.model / "mlx2-conversion-receipt.json"
    receipt = {
        "checkpoint": f"{args.model.name}@sha256:{adapter.identity['fingerprint']}",
        "conversion_receipt_sha256": (hashlib.sha256(conversion.read_bytes()).hexdigest()
                                      if conversion.exists() else None),
        "source_revision": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "mlx": mx.__version__, "adapter": args.adapter, "policy": policy,
        "implementation_identity": routed.implementation_identity(),
        "moe_blocks": len(blocks(model)), "load_seconds": round(time.time() - started, 1),
        "swap_before": swap(), "thermal_hold": False, "qualification": "unqualified",
        "statistics": "sanity, not publishable",
    }
    try:
        contexts = [int(c) for c in args.contexts.split(",")]
        ids = corpus(adapter.tokenizer, max(contexts))
        routed.status(reset=True)
        receipt["parity"] = parity_phase(model, ids[:contexts[0]], mx)
        receipt["tokens"] = tokens_phase(model, adapter, args.steps, mx)
        receipt["ab"] = ab_phase(model, {c: ids[:c] for c in contexts}, args.rounds, mx)
        receipt["counters"] = routed.status()
        receipt["verdict"] = {
            "parity_all_exact": receipt["parity"]["exact"] == receipt["parity"]["visits"] > 0,
            "tokens_all_equal": all(t["tokens_equal"] for t in receipt["tokens"]),
            "engaged": receipt["counters"].get("engaged", 0) > 0,
        }
    finally:
        receipt["swap_after"] = swap()
        receipt["wall_seconds"] = round(time.time() - started, 1)
        atomic_json(args.out / "result.json", receipt)
        adapter.close()


if __name__ == "__main__":
    main()
