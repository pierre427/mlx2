"""Inkling 2 ``inkling2_fuse_*``: exactness and decode/prefill rate on the GPU.

The adapter loads with the current defaults; only the named fuse options are
toggled (``Model.configure_kernels``, everything else kept). For each
context one model carries two caches: A is prefilled with the options off and
B with them on. Decode then alternates in blocks (A off, B on, ...), 32
greedy tokens per block. Arm B is fed arm A's tokens, so every step's logits
are compared bitwise even for the near-exact kernel options; B's own argmax
is recorded too. At the first step whose logits differ, the top-5 log
probabilities of both arms are kept (for near-tie classification). This
records:

- prefill: seconds per arm, last-row logits bitwise equal;
- decode: per-step logits bitwise, tokens, argmax agreement, tok/s per block;
- memory after each block, the kernel receipt (fuse counters) and the
  launch-count script's tiny-model numbers are separate
  (``scripts/research/inkling2_launch_count.py``).

This is a sanity A/B (no thermal hold), not publishable statistics.

Usage: python scripts/research/inkling2_fuse_ab.py <model dir> <run dir>
       [--contexts 1024,16384,65536] [--options fuse_controllers,...] [--blocks 2]
       [--base reference|defaults]
Default options: the four exact ones. Add ``fuse_sconv_kernel`` and/or
``fuse_residual_kernel`` for the near-exact arms (run them separately).

``io:`` options toggle the bit-identical attention I/O module switches
instead of a kernel option (2026-11-22; on by default, so run them with
``--base defaults``): ``io:sink_slot_spare`` (``language.SINK_SLOT_SPARE``),
``io:ring_direct`` (``splitk.RING_DIRECT``) and ``io:vector_loads``
(``splitk.VECTOR_LOADS``). Arm A runs with the named switches off, arm B on.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 512
BLOCK = 32
GIB = 2**30
EXACT = ("fuse_controllers", "fuse_residual", "fuse_sconv", "fuse_attention")


def corpus_tokens(tokenizer, n):
    parts = []
    for pattern in ("docs/**/*.md", "src/mlx2/**/*.py", "scripts/**/*.py", "tests/**/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            parts.append(path.read_text(errors="replace"))
            if len(parts) % 25 == 0:
                ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
                if len(ids) >= n:
                    return ids[:n]
    raise SystemExit("corpus too short")


def memory():
    import mlx.core as mx

    return {"active_gib": round(mx.get_active_memory() / GIB, 2),
            "cache_gib": round(mx.get_cache_memory() / GIB, 2),
            "peak_gib": round(mx.get_peak_memory() / GIB, 2)}


IO_SWITCHES = {"io:sink_slot_spare": ("language", "SINK_SLOT_SPARE"),
               "io:ring_direct": ("splitk", "RING_DIRECT"),
               "io:vector_loads": ("splitk", "VECTOR_LOADS")}


class Toggle:
    def __init__(self, model, options):
        unknown = [o for o in options if o.startswith("io:") and o not in IO_SWITCHES]
        if unknown:
            raise SystemExit(f"unknown io switch {unknown[0]}; known: {sorted(IO_SWITCHES)}")
        self.model = model
        self.options = [o for o in options if not o.startswith("io:")]
        self.switches = [IO_SWITCHES[o] for o in options if o.startswith("io:")]
        self.base = {k: v for k, v in model.kernel_policy.items()}

    def __call__(self, on):
        from mlx2.runtime.models.inkling2 import language, splitk

        modules = {"language": language, "splitk": splitk}
        for module, name in self.switches:
            setattr(modules[module], name, bool(on))
        self.model.configure_kernels(
            **{**self.base, **{name: bool(on) for name in self.options}})


def prefill(model, toggle, tokens, on):
    import mlx.core as mx

    toggle(on)
    cache = model.make_cache()
    mx.synchronize()
    t0 = time.perf_counter()
    for a in range(0, len(tokens), CHUNK):
        logits = model(mx.array([tokens[a:a + CHUNK]]), cache=cache)
        mx.eval(logits)
        mx.clear_cache()  # ordinary serving prefill clears per chunk
    mx.synchronize()
    return cache, logits[0, -1], time.perf_counter() - t0


def decode_block(model, toggle, cache, last, on, forced=None):
    """Greedy block; with ``forced`` the inputs are those tokens instead."""
    import mlx.core as mx

    toggle(on)
    token = int(mx.argmax(last).item()) if forced is None else forced[0]
    rows, tokens, own = [], [], []
    mx.synchronize()
    t0 = time.perf_counter()
    for i in range(BLOCK):
        tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)[0, -1]
        mx.eval(logits)
        rows.append(logits)
        own.append(int(mx.argmax(logits).item()))
        token = own[-1] if forced is None or i + 1 >= len(forced) else forced[i + 1]
    mx.synchronize()
    seconds = time.perf_counter() - t0
    return rows, tokens, own, rows[-1], round(BLOCK / seconds, 2)


def top5(row):
    import mlx.core as mx

    logp = row - mx.logsumexp(row)
    idx = mx.argsort(-logp)[:5]
    return [[int(i), round(float(logp[i].item()), 5)] for i in idx.tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("run_dir")
    parser.add_argument("--contexts", default="1024,16384,65536")
    parser.add_argument("--options", default=",".join(EXACT))
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--base", choices=("reference", "defaults"), default="reference",
                        help="reference: every default-on option off (the historical A/B); "
                             "defaults: the shipped defaults, so the near-exact kernel "
                             "options are measured on top of the exact four")
    args = parser.parse_args()
    model_dir, run_dir = Path(args.model_dir), Path(args.run_dir)
    contexts = [int(c) for c in args.contexts.split(",")]
    options = [o for o in args.options.split(",") if o]
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import REFERENCE_POLICY, Inkling2Adapter

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "src"],
                                capture_output=True, text=True).stdout.strip())
    receipt = {"script": "scripts/research/inkling2_fuse_ab.py", "mlx2": head,
               "mlx2_src_dirty": dirty, "mlx_version": mx.__version__,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "options": options,
               "kind": "exactness + sanity A/B (no thermal hold; not publishable statistics)",
               "chunk": CHUNK, "block": BLOCK, "cells": []}
    out = run_dir / "fuse_ab.json"

    def save():
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out)

    policy = dict(REFERENCE_POLICY) if args.base == "reference" else {}
    receipt["base"] = args.base
    adapter = Inkling2Adapter(str(model_dir), execution_policy=policy)  # only the named options toggle
    try:
        model = adapter.model
        receipt["checkpoint"] = adapter.checkpoint
        receipt["base_kernel_policy"] = dict(model.kernel_policy)
        toggle = Toggle(model, options)
        corpus = corpus_tokens(adapter.tokenizer, max(contexts))
        for n in contexts:
            tokens = corpus[:n]
            mx.reset_peak_memory()
            cache_a, last_a, sec_a = prefill(model, toggle, tokens, False)
            cache_b, last_b, sec_b = prefill(model, toggle, tokens, True)
            cell = {"context": n, "prefill_s": {"off": round(sec_a, 2), "on": round(sec_b, 2)},
                    "prefill_tok_s": {"off": round(n / sec_a, 1), "on": round(n / sec_b, 1)},
                    "prefill_last_logits_bitwise": bool(mx.array_equal(last_a, last_b)),
                    "prefill_last_max_abs": float(mx.abs(last_a - last_b).max().item()),
                    "after_prefill": memory(), "blocks": []}
            equal, agree, total, first = True, 0, 0, None
            for i in range(args.blocks):
                rows_a, toks_a, own_a, last_a, rate_a = decode_block(
                    model, toggle, cache_a, last_a, False)
                mem_a = memory()
                rows_b, toks_b, own_b, last_b, rate_b = decode_block(
                    model, toggle, cache_b, last_b, True, forced=toks_a)
                mem_b = memory()
                same = [bool(mx.array_equal(x, y)) for x, y in zip(rows_a, rows_b)]
                equal &= all(same)
                agree += sum(a == b for a, b in zip(own_a, own_b))
                total += len(own_a)
                if first is None and not all(same):
                    j = same.index(False)
                    first = {"block": i, "step": j, "token_in": toks_a[j],
                             "off_top5": top5(rows_a[j]), "on_top5": top5(rows_b[j]),
                             "max_abs": float(mx.abs(rows_a[j] - rows_b[j]).max().item())}
                cell["blocks"].append({"block": i, "off_tok_s": rate_a, "on_tok_s": rate_b,
                                       "logits_bitwise_steps": sum(same), "steps": len(same),
                                       "tokens": toks_a, "on_argmax_equal": sum(
                                           a == b for a, b in zip(own_a, own_b)),
                                       "mem_after_off": mem_a, "mem_after_on": mem_b})
                print(n, {k: v for k, v in cell["blocks"][-1].items() if k != "tokens"},
                      flush=True)
            cell["decode_logits_bitwise"] = equal
            cell["decode_argmax_agreement"] = f"{agree}/{total}"
            cell["first_divergence"] = first
            receipt["cells"].append(cell)
            del cache_a, cache_b
            mx.clear_cache()
            save()
        receipt["kernel_receipt"] = model.kernel_receipt()
        receipt["peak_memory_gb"] = round(mx.get_peak_memory() / 2**30, 2)
    finally:
        adapter.close()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()


if __name__ == "__main__":
    main()
