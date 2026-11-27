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

Prefill options (the 2026-11-22 prefill/head set: ``prefill_tiled_attention``,
``prefill_mask_with_sink``, ``prefill_mask_bf16_budget``) change every
prefill row, so the prefill is compared row by row as well: each arm keeps
the top-5 log probabilities of every prefill row (computed outside the timed
forward), the cell records the argmax agreement over all rows and, at the
first row whose argmax differs, both arms' top-5 and top-1/top-2 margins for
near-tie classification. ``prefill_tok_s`` times the forwards only.
Engagement counters (the kernel receipt's call counts) are recorded per arm
per cell.

This is a sanity A/B (no thermal hold), not publishable statistics.

Usage: python scripts/research/inkling2_fuse_ab.py <model dir> <run dir>
       [--contexts 1024,16384,65536] [--options fuse_controllers,...] [--blocks 2]
       [--base reference|defaults] [--base-extra head_kernel,...] [--base-off name,...]
Default options: the four exact ones. Add ``fuse_sconv_kernel`` and/or
``fuse_residual_kernel`` for the near-exact arms (run them separately).

``io:`` options toggle the bit-identical attention I/O module switches
instead of a kernel option (2026-11-22; on by default, so run them with
``--base defaults``): ``io:sink_slot_spare`` (``language.SINK_SLOT_SPARE``),
``io:ring_direct`` (``splitk.RING_DIRECT``) and ``io:vector_loads``
(``splitk.VECTOR_LOADS``). Arm A runs with the named switches off, arm B on.
Option names may carry the policy prefix (``inkling2_prefill_tiled_attention``).
``--base-extra`` turns options on in both arms (``head_kernel`` for an
``head_kernel_wide`` A/B, whose arms then differ only in the loads).
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


def option_name(name):
    """``inkling2_<name>`` policy keys or bare ``configure_kernels`` names."""
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.inkling2 import KERNEL_OPTIONS

    if name.startswith("io:"):  # module switches, checked by Toggle
        return name
    name = KERNEL_OPTIONS.get(name, name)
    if name not in KERNEL_OPTIONS.values():
        raise SystemExit(f"unknown Inkling 2 kernel option {name!r}")
    return name


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
               "io:vector_loads": ("splitk", "VECTOR_LOADS"),
               # Near-exact, not bit-identical (oMLX #5620 split count A/B).
               "io:split_fixed256": ("splitk", "SPLIT_FIXED256"),
               # Candidate bitwise residual kernel (MLX 0.32.5 reduction orders).
               "io:residual_mlx_order": ("fuse_kernels", "RESIDUAL_MLX_ORDER")}


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
        from mlx2.runtime.models.inkling2 import fuse_kernels, language, splitk

        modules = {"language": language, "splitk": splitk, "fuse_kernels": fuse_kernels}
        for module, name in self.switches:
            setattr(modules[module], name, bool(on))
        self.model.configure_kernels(
            **{**self.base, **{name: bool(on) for name in self.options}})


def row_top5(logits):
    """Top-5 (indices, log probabilities) of every row, best first."""
    import mlx.core as mx

    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    idx = mx.argpartition(-logp, 4, axis=-1)[..., :5]
    vals = mx.take_along_axis(logp, idx, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    return mx.take_along_axis(idx, order, axis=-1), mx.take_along_axis(vals, order, axis=-1)


def prefill(model, toggle, tokens, on):
    """(cache, last logits, forward seconds, per-row top-5 of every row)."""
    import mlx.core as mx

    toggle(on)
    cache = model.make_cache()
    tops = []
    seconds = 0.0
    for a in range(0, len(tokens), CHUNK):
        mx.synchronize()
        t0 = time.perf_counter()
        logits = model(mx.array([tokens[a:a + CHUNK]]), cache=cache)
        mx.eval(logits)
        mx.synchronize()
        seconds += time.perf_counter() - t0
        top = row_top5(logits[0].astype(mx.float32))
        mx.eval(top)
        tops.append(top)
        last = logits[0, -1]
        del logits
        mx.clear_cache()  # ordinary serving prefill clears per chunk
    idx = mx.concatenate([t[0] for t in tops], axis=0)
    vals = mx.concatenate([t[1] for t in tops], axis=0)
    return cache, last, seconds, (idx, vals)


def prefill_divergence(top_a, top_b):
    """Argmax agreement over all prefill rows and the first differing row."""
    import mlx.core as mx

    (ia, va), (ib, vb) = top_a, top_b
    same = ia[:, 0] == ib[:, 0]
    agree = int(same.sum().item())
    first = None
    if agree < same.size:
        row = int(mx.argmin(same.astype(mx.int32)).item())

        def five(i, v):
            return [[int(t), round(float(x), 5)] for t, x in zip(i[row].tolist(), v[row].tolist())]

        first = {"row": row, "off_top5": five(ia, va), "on_top5": five(ib, vb),
                 "off_margin": round(float((va[row, 0] - va[row, 1]).item()), 5),
                 "on_margin": round(float((vb[row, 0] - vb[row, 1]).item()), 5)}
    return f"{agree}/{same.size}", first


COUNTERS = ("prefill_tiled_calls", "prefill_tiled_declines", "prefill_mask_with_sink_calls",
            "prefill_mask_bf16_budget_calls", "head_kernel_calls", "head_kernel_wide_calls",
            "compiled_moe_calls", "compiled_moe_traces", "flash_prefill_calls",
            "verify_splitk_calls", "fuse_counts",
            "prefill_handover_calls",
            # 2026-11-23 MTPLX launch set (calls / proved / refused / declined:*).
            "pass_messages", "stage_a", "gate_merge", "attn_output", "embed_kernel",
            # inkling2_prefill_onelaunch_attention (calls; declines by reason).
            "prefill_onelaunch_calls", "prefill_onelaunch_declines")


def counters(model):
    receipt = model.kernel_receipt()
    return {k: receipt.get(k) for k in COUNTERS}


def counter_delta(before, after):
    out = {}
    for key, value in after.items():
        old = before.get(key)
        if isinstance(value, dict):
            delta = {k: v - (old or {}).get(k, 0) for k, v in value.items()}
            delta = {k: v for k, v in delta.items() if v}
            if delta:
                out[key] = delta
        elif isinstance(value, int) and value - (old or 0):
            out[key] = value - (old or 0)
    return out


def decode_block(model, toggle, cache, last, on, forced=None, untimed=0):
    """Greedy block; with ``forced`` the inputs are those tokens instead.

    The first ``untimed`` steps run (and are compared) but are left out of
    the rate: the switches that prove a launch shape on first use
    (library_kernels) compile and prove inside their first decode step, a
    one-time cost per process that is not decode throughput."""
    import mlx.core as mx

    toggle(on)
    token = int(mx.argmax(last).item()) if forced is None else forced[0]
    rows, tokens, own = [], [], []
    untimed = max(0, min(int(untimed), BLOCK - 1))
    mx.synchronize()
    t0 = time.perf_counter()
    for i in range(BLOCK):
        if i == untimed and untimed:
            mx.synchronize()
            t0 = time.perf_counter()
        tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)[0, -1]
        mx.eval(logits)
        rows.append(logits)
        own.append(int(mx.argmax(logits).item()))
        token = own[-1] if forced is None or i + 1 >= len(forced) else forced[i + 1]
    mx.synchronize()
    seconds = time.perf_counter() - t0
    return rows, tokens, own, rows[-1], round((BLOCK - untimed) / seconds, 2)


def top5(row):
    import mlx.core as mx

    logp = row - mx.logsumexp(row)
    idx = mx.argsort(-logp)[:5]
    return [[int(i), round(float(logp[i].item()), 5)] for i in idx.tolist()]


CEILING = 0.5  # AGENTS.md near-tie ceiling (nats)


def classify_flip(off5, on5):
    """Margin classification of one teacher-forced argmax flip (same prefix by
    construction).  Each arm's top-5 is ``[[id, logp], ...]``.  ``top2`` is the
    two-arm contract (same unordered top two, both selected-vs-alternate
    margins <= CEILING); ``multiway`` the multi-way clause (each arm's selected
    token within CEILING of the other arm's top log-probability, both read from
    the windows).  The functional-oracle part of the contract is not run here
    (no continuation), so this is margin evidence only."""
    a = {i: v for i, v in off5}
    b = {i: v for i, v in on5}
    sel_a, sel_b = off5[0][0], on5[0][0]
    margin_a = off5[0][1] - off5[1][1]
    margin_b = on5[0][1] - on5[1][1]
    top2 = ({off5[0][0], off5[1][0]} == {on5[0][0], on5[1][0]}
            and margin_a <= CEILING and margin_b <= CEILING)
    multiway = None
    if sel_b in a and sel_a in b:
        multiway = (off5[0][1] - a[sel_b] <= CEILING) and (on5[0][1] - b[sel_a] <= CEILING)
    return {"off_selected": sel_a, "on_selected": sel_b, "off_margin": round(margin_a, 5),
            "on_margin": round(margin_b, 5), "top2_near_tie": top2, "multiway_near_tie": multiway,
            "near_tie_margins": bool(top2 or multiway)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("run_dir")
    parser.add_argument("--contexts", default="1024,16384,65536")
    parser.add_argument("--options", default=",".join(EXACT))
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--untimed-steps", type=int, default=0,
                        help="leading decode steps of the first block left out of both "
                             "arms' rates (first-use compile + proof of library_kernels "
                             "launches); they are still compared bitwise")
    parser.add_argument("--base", choices=("reference", "defaults"), default="reference",
                        help="reference: every default-on option off (the historical A/B); "
                             "defaults: the shipped defaults, so the near-exact kernel "
                             "options are measured on top of the exact four")
    parser.add_argument("--base-off", default="",
                        help="comma list of options turned OFF in both arms (e.g. a default "
                             "that is an alternative to the option under test)")
    parser.add_argument("--base-extra", default="",
                        help="options turned on in both arms (e.g. head_kernel)")
    args = parser.parse_args()
    model_dir, run_dir = Path(args.model_dir), Path(args.run_dir)
    contexts = [int(c) for c in args.contexts.split(",")]
    options = [option_name(o) for o in args.options.split(",") if o]
    extra = [option_name(o) for o in args.base_extra.split(",") if o]
    base_off = [option_name(o) for o in args.base_off.split(",") if o]
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
               "base_extra": extra, "base_off": base_off,
               "kind": "exactness + sanity A/B (no thermal hold; not publishable statistics)",
               "chunk": CHUNK, "block": BLOCK, "untimed_steps": args.untimed_steps,
               "cells": []}
    out = run_dir / "fuse_ab.json"

    def save():
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out)

    policy = dict(REFERENCE_POLICY) if args.base == "reference" else {}
    # --base-off: off in both arms, through the adapter policy so its
    # validation sees the choice (a default that is an alternative to the
    # option under test, e.g. prefill_tiled_attention vs onelaunch).
    policy.update({"inkling2_" + name: False for name in base_off})
    receipt["base"] = args.base
    adapter = Inkling2Adapter(str(model_dir), execution_policy=policy)  # only the named options toggle
    try:
        model = adapter.model
        receipt["checkpoint"] = adapter.checkpoint
        if extra:
            model.configure_kernels(**{**model.kernel_policy, **{n: True for n in extra}})
        receipt["base_kernel_policy"] = dict(model.kernel_policy)
        toggle = Toggle(model, options)
        corpus = corpus_tokens(adapter.tokenizer, max(contexts))
        for n in contexts:
            tokens = corpus[:n]
            mx.reset_peak_memory()
            c0 = counters(model)
            cache_a, last_a, sec_a, top_a = prefill(model, toggle, tokens, False)
            c1 = counters(model)
            cache_b, last_b, sec_b, top_b = prefill(model, toggle, tokens, True)
            c2 = counters(model)
            agreement, first_prefill = prefill_divergence(top_a, top_b)
            cell = {"context": n, "prefill_s": {"off": round(sec_a, 2), "on": round(sec_b, 2)},
                    "prefill_tok_s": {"off": round(n / sec_a, 1), "on": round(n / sec_b, 1)},
                    "prefill_last_logits_bitwise": bool(mx.array_equal(last_a, last_b)),
                    "prefill_last_max_abs": float(mx.abs(last_a - last_b).max().item()),
                    "prefill_argmax_agreement": agreement,
                    "prefill_first_divergence": first_prefill,
                    "prefill_counters": {"off": counter_delta(c0, c1),
                                         "on": counter_delta(c1, c2)},
                    "after_prefill": memory(), "blocks": []}
            del top_a, top_b
            print(n, {k: cell[k] for k in ("prefill_tok_s", "prefill_argmax_agreement",
                                           "prefill_first_divergence")}, flush=True)
            c_decode = counters(model)
            equal, agree, total, first = True, 0, 0, None
            flips = []
            for i in range(args.blocks):
                rows_a, toks_a, own_a, last_a, rate_a = decode_block(
                    model, toggle, cache_a, last_a, False,
                    untimed=args.untimed_steps if i == 0 else 0)
                mem_a = memory()
                rows_b, toks_b, own_b, last_b, rate_b = decode_block(
                    model, toggle, cache_b, last_b, True, forced=toks_a,
                    untimed=args.untimed_steps if i == 0 else 0)
                mem_b = memory()
                same = [bool(mx.array_equal(x, y)) for x, y in zip(rows_a, rows_b)]
                equal &= all(same)
                agree += sum(a == b for a, b in zip(own_a, own_b))
                total += len(own_a)
                for j, (x, y) in enumerate(zip(own_a, own_b)):
                    if x != y:
                        o5, n5 = top5(rows_a[j]), top5(rows_b[j])
                        flips.append({"block": i, "step": j, "token_in": toks_a[j],
                                      "off_top5": o5, "on_top5": n5, **classify_flip(o5, n5)})
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
            cell["decode_counters_both_arms"] = counter_delta(c_decode, counters(model))
            cell["decode_logits_bitwise"] = equal
            cell["decode_argmax_agreement"] = f"{agree}/{total}"
            cell["first_divergence"] = first
            cell["argmax_flips"] = flips
            cell["argmax_flips_near_tie_margins"] = (
                f"{sum(f['near_tie_margins'] for f in flips)}/{len(flips)}")
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
