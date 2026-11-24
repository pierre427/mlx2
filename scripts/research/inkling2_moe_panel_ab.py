"""Inkling 2 ``inkling2_moe_panel_kernel`` and ``inkling2_moe_nax_mxfp8``: exactness
and rate on the GPU (real checkpoint).

Original mlx2 script (2026-11-23, MTPLX sweep candidates 1 and 10).  The
adapter loads with the current defaults; only the named option is toggled
(``Model.configure_kernels``, everything else kept).

1. **Verify panels** (``inkling2_moe_panel_kernel``), per context: one cache
   prefilled with the option off (512-token chunks), then rounds of an R-row
   verify forward through ``HybridVerifyRows`` (begin, forward, abort, as a
   rejected DSpark round) for R in ``--rows`` (default 2, 4, 8), alternating
   off and on on the same cache.  Per R: median forward seconds per arm, the
   first round's per-row logits compared (``bitwise`` when every row is equal;
   else max |diff|, argmax agreement and top-two margins where argmax differs,
   for near-tie classification), and the kernel-receipt delta per arm
   (``moe_panel_kernel_calls`` / formats / declines prove engagement: 48
   visits per verify forward, 44 NVFP4 + 4 MXFP8).
2. **NAX MXFP8 prefill** (``--nax-mxfp8``; needs ``inkling2_moe_nax_gather``
   resolved on, i.e. an M5): per context, prefill off then on (separate
   caches), last-row logits bitwise, the last row of every chunk compared,
   seconds per arm, and the NAX ``calls_by_format`` delta
   (``gather/mxfp8-8`` > 0 on the on arm only).

A sanity A/B (no thermal hold), not publishable statistics.  Refuses a
nonempty run directory.

Usage: python scripts/research/inkling2_moe_panel_ab.py <model dir> <run dir>
       [--contexts 1024,4096,16384] [--rows 2,4,8] [--rounds 8] [--nax-mxfp8]
       [--skip-panel] [--panel-over-weighted-sum]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))
from inkling2_splitk_ab import (  # noqa: E402
    CHUNK, GIB, compare_rows, corpus_tokens, device_name, memory, verify_round)


def panel_counts(model):
    r = model.kernel_receipt()
    return {"moe_panel_kernel_calls": r["moe_panel_kernel_calls"],
            "moe_panel_kernel_formats": dict(r["moe_panel_kernel_formats"]),
            "moe_panel_kernel_declines": dict(r["moe_panel_kernel_declines"]),
            "moe_fused_calls": r["moe_fused_calls"]}


def nax_counts(model):
    """Cumulative NAX counters (differenced per arm)."""
    nax = model.kernel_receipt().get("moe_nax_gather") or {}
    return {"calls_by_format": dict(nax.get("calls_by_format") or {}),
            "fallbacks": dict(nax.get("fallbacks") or {})}


def mxfp8_owned_tables(model):
    """A gauge, not a counter: the MXFP8 tables owned on the live route now.
    (The first smoke, run w1122-3ee59f64e-mtplx-smoke-moe-a1, differenced it
    like the counters and so recorded 0 on the on arm, where it was 8.)"""
    return int(model.kernel_receipt()["moe_nax_mxfp8"]["owned_live_tables"])


def delta(after, before):
    out = {}
    for key, value in after.items():
        if isinstance(value, dict):
            out[key] = {k: v - before[key].get(k, 0) for k, v in value.items()
                        if v - before[key].get(k, 0)}
        else:
            out[key] = value - before[key]
    return out


def set_arm(model, base, name, on):
    model.configure_kernels(**{**base, name: on})


def prefill(model, tokens, keep_rows=False):
    import mlx.core as mx

    cache = model.make_cache()
    rows = []
    mx.synchronize()
    t0 = time.perf_counter()
    for a in range(0, len(tokens), CHUNK):
        logits = model(mx.array([tokens[a:a + CHUNK]]), cache=cache)
        mx.eval(logits)
        if keep_rows:
            rows.append(logits[0, -1])
        mx.clear_cache()
    mx.synchronize()
    return cache, logits[0, -1], time.perf_counter() - t0, rows


def panel_cell(model, base, tokens, verify, rows_list, rounds):
    import mlx.core as mx

    set_arm(model, base, "moe_panel_kernel", False)
    cache, _, seconds, _ = prefill(model, tokens)
    cell = {"context": len(tokens), "prefill_s": round(seconds, 2), "after_prefill": memory(),
            "panels": []}
    for rows in rows_list:
        times = {"off": [], "on": []}
        first, deltas = {}, {}
        for i in range(rounds + 1):  # round 0 is warm-up (times dropped)
            for arm in ("off", "on"):
                set_arm(model, base, "moe_panel_kernel", arm == "on")
                before = panel_counts(model)
                out, sec = verify_round(model, cache, verify[:rows])
                if arm not in deltas:
                    deltas[arm] = delta(panel_counts(model), before)
                if i == 0:
                    first[arm] = [out[j] for j in range(out.shape[0])]
                else:
                    times[arm].append(sec)
        cmp = compare_rows(first["off"], first["on"])
        bitwise = all(bool(mx.array_equal(a, b)) for a, b in zip(first["off"], first["on"]))
        panel = {"rows": rows, "rounds": rounds, "bitwise": bitwise,
                 "median_forward_s": {a: round(statistics.median(t), 5) for a, t in times.items()},
                 "forward_s": {a: [round(s, 5) for s in t] for a, t in times.items()},
                 "receipt_delta_first_round": deltas, "logits": cmp}
        cell["panels"].append(panel)
        print(len(tokens), "rows", rows, "bitwise", bitwise, panel["median_forward_s"],
              deltas.get("on", {}).get("moe_panel_kernel_calls"), flush=True)
    del cache
    mx.clear_cache()
    return cell


def nax_cell(model, base, tokens):
    import mlx.core as mx

    out = {"context": len(tokens)}
    lasts, chunk_rows = {}, {}
    for arm in ("off", "on"):
        set_arm(model, base, "moe_nax_mxfp8", arm == "on")
        before = nax_counts(model)
        cache, last, seconds, rows = prefill(model, tokens, keep_rows=True)
        out[arm] = {"prefill_s": round(seconds, 2),
                    "prefill_tok_s": round(len(tokens) / seconds, 1),
                    "nax_delta": delta(nax_counts(model), before),
                    "mxfp8_owned_tables": mxfp8_owned_tables(model)}
        lasts[arm], chunk_rows[arm] = last, rows
        del cache
        mx.clear_cache()
    out["last_logits_bitwise"] = bool(mx.array_equal(lasts["off"], lasts["on"]))
    out["chunk_last_rows"] = compare_rows(chunk_rows["off"], chunk_rows["on"])
    out["chunk_last_rows_bitwise"] = all(
        bool(mx.array_equal(a, b)) for a, b in zip(chunk_rows["off"], chunk_rows["on"]))
    print(len(tokens), "nax_mxfp8", {a: out[a]["prefill_tok_s"] for a in ("off", "on")},
          "bitwise", out["last_logits_bitwise"], out["on"]["nax_delta"], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--contexts", default="1024,4096,16384")
    ap.add_argument("--rows", default="2,4,8")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--nax-mxfp8", action="store_true")
    ap.add_argument("--skip-panel", action="store_true")
    ap.add_argument("--panel-over-weighted-sum", action="store_true",
                    help="set moe_panel_kernel.PANEL_OVER_WEIGHTED_SUM: with the default "
                         "weighted sum on, sorted verify panels (>= 3 rows) otherwise decline")
    args = ap.parse_args()
    contexts = [int(c) for c in args.contexts.split(",")]
    rows_list = [int(r) for r in args.rows.split(",")]
    if any(not 2 <= r <= 32 for r in rows_list):
        raise SystemExit("--rows must be 2..32")
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
                           capture_output=True, text=True).stdout.strip()
    receipt = {"script": "scripts/research/inkling2_moe_panel_ab.py", "mlx2": head,
               "src_dirty": bool(dirty), "mlx_version": mx.__version__,
               "device": device_name(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "kind": "exactness + sanity A/B (no thermal hold; not publishable statistics)",
               "chunk": CHUNK, "contexts": contexts, "rows": rows_list, "panel": [], "nax": []}
    from mlx2.runtime.models.inkling2 import moe_panel_kernel

    moe_panel_kernel.PANEL_OVER_WEIGHTED_SUM = bool(args.panel_over_weighted_sum)
    receipt["panel_over_weighted_sum"] = moe_panel_kernel.PANEL_OVER_WEIGHTED_SUM
    out = run_dir / "moe_panel_ab.json"

    def save():
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(receipt, indent=1, default=str))
        os.replace(tmp, out)

    from mlx2.adapters.inkling2 import Inkling2Adapter

    adapter = Inkling2Adapter(str(args.model_dir))  # current defaults
    try:
        model = adapter.model
        receipt["checkpoint"] = adapter.checkpoint
        base = dict(model.kernel_policy)
        receipt["base_policy"] = base
        corpus = corpus_tokens(adapter.tokenizer, max(contexts) + max(rows_list))
        with mx.stream(mx.gpu):
            if not args.skip_panel:
                for n in contexts:
                    receipt["panel"].append(panel_cell(model, base, corpus[:n],
                                                       corpus[n:n + max(rows_list)],
                                                       rows_list, args.rounds))
                    save()
            if args.nax_mxfp8:
                if not base.get("moe_nax_gather"):
                    receipt["nax_refused"] = "inkling2_moe_nax_gather resolved off on this host"
                else:
                    for n in contexts:
                        receipt["nax"].append(nax_cell(model, base, corpus[:n]))
                        save()
        receipt["kernel_receipt"] = model.kernel_receipt()
        receipt["peak_memory_gb"] = round(mx.get_peak_memory() / GIB, 2)
    finally:
        adapter.close()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()


if __name__ == "__main__":
    main()
