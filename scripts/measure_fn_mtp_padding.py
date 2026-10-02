"""Pad-row census and row-cost curve for Flash-Next batched self-MTP rounds.

Measures, on the served artifact and the served batched self-MTP route
(``adapter.execution_config``, the route's copy-draft policy, greedy, real
chat prompts), how many rows of every batched forward are right-padding:

- ``verify``: the target trunk forward over the (B, width) verify rectangle
  (every lane's current token plus its drafts, right-padded to the widest
  lane), and the depth-zero target round (all lengths 1).
- ``draft``: the MTP head forwards; the first depth teacher-forces each
  lane's pending accepted tokens, so its width varies per lane.

It records every forward's per-lane valid lengths and right padding (by
wrapping ``hybrid_speculative._prepare_self_mtp_cache_group`` and the two
forward entry points), and with ``--timed`` synchronises around each forward
to attribute wall time to verify and draft forwards (the syncs serialise the
pipeline; timed numbers attribute cost, they are not throughput).

``--row-cost`` then times the trunk forward (``model.mtp_backbone`` + head)
on fresh caches at shapes (1, M) for small M, decode shapes (B, 1), verify
shapes (B, 3), and prefill slices P with and without D extra rows packed in
the same stream (the cost side of folding decode rows into a prefill slice).
Fresh caches mean no attention history: these isolate projection/MoE/head
cost per row, which is what pad rows and folded rows add.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/measure_fn_mtp_padding.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP \\
      --configs 2:2:route 3:2:route 4:2:route --timed --row-cost --out pad.json

A config is ``B:num_draft:copy`` with copy one of ``route`` (the served
policy: cohorts never copy), ``cohort`` (``batched_max_span: null``, the
head-depth cap inside cohorts) or ``off``.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

_DOC = ROOT / "docs" / "QUALIFICATION.md"


def pad_census(records):
    """Rows and pad rows per forward kind, grouped by lanes in the forward."""
    out = collections.defaultdict(lambda: {"forwards": 0, "rows": 0, "valid": 0, "pad": 0,
                                           "widths": collections.Counter(),
                                           "ms": [], "ms_by_rows": collections.defaultdict(list)})
    for rec in records:
        if rec["lengths"] is None:
            continue
        lanes = len(rec["lengths"])
        width = rec["shape"][1]
        key = f"{rec['kind']}@B{lanes}"
        g = out[key]
        g["forwards"] += 1
        g["rows"] += lanes * width
        g["valid"] += sum(rec["lengths"])
        g["pad"] += lanes * width - sum(rec["lengths"])
        g["widths"][width] += 1
        if rec["ms"] is not None:
            g["ms"].append(rec["ms"])
            g["ms_by_rows"][lanes * width].append(rec["ms"])
    result = {}
    for key, g in sorted(out.items()):
        entry = {
            "forwards": g["forwards"], "rows": g["rows"], "valid_rows": g["valid"],
            "pad_rows": g["pad"], "pad_fraction": g["pad"] / max(1, g["rows"]),
            "width_histogram": dict(sorted(g["widths"].items())),
        }
        if g["ms"]:
            entry["ms_total"] = sum(g["ms"])
            entry["ms_median"] = statistics.median(g["ms"])
            entry["ms_median_by_rows"] = {r: statistics.median(v) for r, v in sorted(g["ms_by_rows"].items())}
            # Linear (upper-bound) attribution: pad share of each forward's time.
            entry["pad_ms_linear_upper_bound"] = sum(
                rec["ms"] * (1 - sum(rec["lengths"]) / (len(rec["lengths"]) * rec["shape"][1]))
                for rec in records
                if rec["ms"] is not None and rec["lengths"] is not None
                and f"{rec['kind']}@B{len(rec['lengths'])}" == key)
        result[key] = entry
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", nargs="*", default=["2:2:route", "3:2:route", "4:2:route"])
    ap.add_argument("--gen", type=int, default=192)
    ap.add_argument("--with-document", action="store_true")
    ap.add_argument("--timed", action="store_true")
    ap.add_argument("--row-cost", action="store_true")
    ap.add_argument("--row-cost-reps", type=int, default=7)
    ap.add_argument("--max-swapout-pages", type=int, default=20000)
    ap.add_argument("--cache-limit-gib", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    # Construct the adapter before any mlx2.runtime model module is imported.
    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    import mlx.core as mx

    from bench_fn_batch_verify import TASKS
    from mlx2.runtime import generate as G
    from mlx2.runtime import hybrid_speculative as H
    from mlx2.runtime.sample_utils import LaneRNG

    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(a.cache_limit_gib << 30)

    def swapouts():
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        line = next(l for l in out.splitlines() if l.startswith("Swapouts"))
        return int(line.split(":")[1].strip().rstrip("."))

    swap0 = swapouts()

    def check_swap():
        grew = swapouts() - swap0
        if grew > a.max_swapout_pages:
            raise SystemExit(f"aborting: Swapouts grew by {grew} pages since load")

    print(f"LOADED active={mx.get_active_memory() / 2**30:.1f}GiB", flush=True)
    route_copy = dict(adapter.default_route_execution_policy["native_mtp"]["self_mtp_copy_draft"])
    document = _DOC.read_text()[:6000] if a.with_document else None

    def prompt(i):
        task = TASKS[i % len(TASKS)]
        if document is not None and i % 2:
            task = f"Read this document:\n\n{document}\n\nThen: {task}"
        return list(adapter.prompt_tokens({"messages": [{"role": "user", "content": task}]}))

    # ---- instrumentation -------------------------------------------------
    state = {"last": None, "timed": False, "records": None}
    orig_prepare = H._prepare_self_mtp_cache_group
    orig_backbone = H._mtp_backbone
    orig_proposal = H._mtp_proposal_step

    def prepare(caches, lengths, right_padding, *args, **kwargs):
        state["last"] = (tuple(int(x) for x in lengths), tuple(int(x) for x in right_padding))
        return orig_prepare(caches, lengths, right_padding, *args, **kwargs)

    def timed_call(kind, fn, rows_shape, *args):
        last, state["last"] = state["last"], None
        if state["timed"]:
            mx.synchronize()
            t0 = time.perf_counter()
        out = fn(*args)
        ms = None
        if state["timed"]:
            mx.eval(out)
            ms = 1e3 * (time.perf_counter() - t0)
        if state["records"] is not None:
            state["records"].append({"kind": kind, "shape": rows_shape, "lengths": last and last[0],
                                     "padding": last and last[1], "ms": ms})
        return out

    def backbone(model_, tokens, cache):
        return timed_call("verify", orig_backbone, tuple(tokens.shape), model_, tokens, cache)

    def proposal(model_, hidden, tokens, cache, lanes):
        return timed_call("draft", orig_proposal, tuple(tokens.shape), model_, hidden, tokens, cache, lanes)

    H._prepare_self_mtp_cache_group = prepare
    H._mtp_backbone = backbone
    H._mtp_proposal_step = proposal

    def run(batch, num_draft, copy, timed):
        prompts = [prompt(i) for i in range(batch)]
        cfg = adapter.execution_config(max_lanes=batch, prefill_step=adapter.prefill_step_default())
        cfg["num_draft"] = int(num_draft)
        kwargs = {"self_mtp": cfg}
        if copy == "route":
            kwargs["copy_draft"] = dict(route_copy)
        elif copy == "cohort":
            kwargs["copy_draft"] = {**route_copy, "batched_max_span": None}
        stats = {}
        gen = G.BatchGenerator(model, completion_batch_size=batch, prefill_batch_size=1,
                               prefill_step_size=adapter.prefill_step_default(),
                               scheduler_stats=stats, **kwargs)
        insert = {"max_tokens": [a.gen] * batch,
                  "lane_rngs": [LaneRNG(1 + i) for i in range(batch)],
                  "self_mtp_configs": [{"sampling_temp": 0.0}] * batch}
        gen.insert(prompts, **insert)
        records = []
        state["records"], state["timed"] = records, timed
        steps = []
        done = set()
        try:
            while len(done) < batch:
                n0 = len(records)
                t0 = time.perf_counter()
                _p, responses = gen.next()
                ms = 1e3 * (time.perf_counter() - t0)
                if len(records) > n0:
                    steps.append({"ms": ms, "first": n0, "end": len(records),
                                  "emitted": len(responses)})
                for r in responses:
                    if r.finish_reason:
                        done.add(r.uid)
        finally:
            state["records"], state["timed"] = None, False
            gen.close()
        mx.clear_cache()
        return records, steps, [len(p) for p in prompts]

    results = {}
    run(2, 2, "route", False)  # warm-up (kernel compilation), discarded
    for config in a.configs:
        b, nd, copy = config.split(":")
        records, steps, plens = run(int(b), int(nd), copy, False)
        check_swap()
        entry = {"census": pad_census(records), "prompt_tokens": plens}
        if a.timed:
            trecords, tsteps, _ = run(int(b), int(nd), copy, True)
            check_swap()
            entry["timed_census"] = pad_census(trecords)
            full = [s for s in tsteps if all(
                trecords[i]["lengths"] is None or len(trecords[i]["lengths"]) == int(b)
                for i in range(s["first"], s["end"]))]
            entry["timed_rounds_full_width"] = {
                "rounds": len(full),
                "round_ms_median": statistics.median(s["ms"] for s in full) if full else None,
                "verify_ms_median": statistics.median(
                    sum(trecords[i]["ms"] for i in range(s["first"], s["end"]) if trecords[i]["kind"] == "verify")
                    for s in full) if full else None,
                "draft_ms_median": statistics.median(
                    sum(trecords[i]["ms"] for i in range(s["first"], s["end"]) if trecords[i]["kind"] == "draft")
                    for s in full) if full else None,
            }
        results[config] = entry
        print(config, json.dumps({k: {kk: vv for kk, vv in v.items() if kk in (
            "forwards", "rows", "pad_rows", "pad_fraction", "width_histogram", "ms_median",
            "pad_ms_linear_upper_bound", "ms_total")} for k, v in entry["census"].items()}), flush=True)
        if a.timed:
            print(config, "timed", json.dumps(entry["timed_rounds_full_width"]),
                  json.dumps({k: {kk: v.get(kk) for kk in ("ms_total", "pad_ms_linear_upper_bound", "pad_fraction")}
                              for k, v in entry["timed_census"].items()}), flush=True)

    H._prepare_self_mtp_cache_group = orig_prepare
    H._mtp_backbone = orig_backbone
    H._mtp_proposal_step = orig_proposal

    row_cost = None
    if a.row_cost:
        source = list(adapter.prompt_tokens({"messages": [{"role": "user", "content":
                      (ROOT / "docs" / "QUALIFICATION.md").read_text()[:20000]}]}))
        shapes = [(1, m) for m in (1, 2, 3, 4, 6, 8, 9, 10, 12, 16, 24, 32, 48, 64)]
        shapes += [(b, 1) for b in (2, 3, 4, 8, 16)]
        shapes += [(b, 3) for b in (2, 3, 4)]
        for p in (128, 256, 448, 1024, 2048):
            shapes += [(1, p), (1, p - 4), (1, p - 1), (1, p + 1), (1, p + 4)]
        shapes = sorted(set(shapes), key=lambda s: (s[0] * s[1], s))

        def forward(shape):
            b, m = shape
            ids = mx.array([source[i * 97: i * 97 + m] for i in range(b)], mx.uint32)
            cache = model.make_cache()
            mx.synchronize()
            t0 = time.perf_counter()
            hidden, _hyper = model.mtp_backbone(ids, cache=cache)
            logits = model.logits(hidden[:, -1:, :] if m > 16 else hidden)
            mx.eval(logits)
            ms = 1e3 * (time.perf_counter() - t0)
            del cache, hidden, logits
            return ms

        for shape in shapes:  # warm-up, two passes
            forward(shape)
            forward(shape)
        mx.clear_cache()
        times = {s: [] for s in shapes}
        for rep in range(a.row_cost_reps):
            order = shapes if rep % 2 == 0 else shapes[::-1]
            for s in order:
                times[s].append(forward(s))
            mx.clear_cache()
            check_swap()
        row_cost = {f"{b}x{m}": {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v)}
                    for (b, m), v in times.items()}
        print("ROWCOST", json.dumps({k: round(v["median_ms"], 2) for k, v in row_cost.items()}), flush=True)

    json.dump({"model": a.model, "policy": adapter.policy.as_dict(), "gen": a.gen,
               "route_copy_draft": route_copy, "results": results, "row_cost": row_cost,
               "swapouts_delta_pages": swapouts() - swap0,
               "peak_gib": mx.get_peak_memory() / 2**30, "mlx": mx.__version__},
              open(a.out, "w"), indent=1, default=lambda o: dict(o) if isinstance(o, collections.Counter) else str(o))


if __name__ == "__main__":
    main()
