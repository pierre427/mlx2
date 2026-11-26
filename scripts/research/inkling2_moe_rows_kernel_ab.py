"""``inkling2_moe_rows_kernel`` on the real checkpoint (GPU): parity, expert
time and whole-model verify A/B, in one process.

Original mlx2 script (2026-11-26).  The adapter loads with its shipped
defaults; the arms toggle only ``moe_rows_kernel`` through
``Model.configure_kernels`` (everything else kept).

1. ``parity``: for each block of R rows (``--rows``, default 2,4,8,16), the
   MoE input of every visit of one real R-row verify forward after a
   ``--parity-context`` prefix of repository text is captured.  On each NVFP4
   visit the kernels are compared with the shipped multi-row
   ``moe_fast.fused_call`` (the route today) and with each row's one-token
   shipped call (``moe_rows_kernel.law``): bitwise counts and max |diff|.
   Each pair of the experts' time (shipped vs kernel, all visits of the
   block chained in one evaluation) is timed too.
2. ``verify``: per context (``--contexts``), one B1 cache; per R, arms
   interleaved (off, on, off, on, ...) for ``--rounds`` rounds, each a
   ``HybridVerifyRows`` round of R real tokens that is aborted afterwards, so
   every forward sees the same cache.  Median ms per forward per arm and the
   logits compared bitwise between arms.

Sanity numbers, no thermal hold; the caller owns the GPU.
Usage: python scripts/research/inkling2_moe_rows_kernel_ab.py <model dir> <run dir>
       [--rows 2,4,8,16] [--contexts 1024,8192] [--rounds 8] [--parity-context 1024]
The run directory must not exist or be empty; ``result.json`` is written atomically.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 512


def corpus(tokenizer, n):
    parts = []
    for pattern in ("docs/**/*.md", "src/mlx2/**/*.py"):
        for path in sorted(ROOT.glob(pattern)):
            parts.append(path.read_text(errors="replace"))
            if len(parts) % 25 == 0:
                ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
                if len(ids) >= n:
                    return ids[:n]
    raise SystemExit("corpus too small")


def git_head():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False).stdout.strip()


def prefill(model, ids):
    import mlx.core as mx

    cache = model.make_cache()
    for s in range(0, len(ids), CHUNK):
        mx.eval(model(mx.array([ids[s:s + CHUNK]]), cache=cache))
    return cache


def capture(model, cache, block):
    """Every SparseMoE input of one R-row forward (inside a verify round that
    is aborted, so ``cache`` is unchanged)."""
    import mlx.core as mx

    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
    from mlx2.runtime.models.inkling2 import language

    captured = []
    original = language.SparseMoE.__call__

    def hook(self, x):
        captured.append((self, x))
        return original(self, x)

    txn = HybridVerifyRows([cache]).begin([len(block)])
    language.SparseMoE.__call__ = hook
    try:
        mx.eval(model(mx.array([block]), cache=txn.caches))
    finally:
        language.SparseMoE.__call__ = original
        txn.abort()
    for _, x in captured:
        mx.eval(x)
    return captured


def maxdiff(a, b):
    import mlx.core as mx

    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item())


def timed_chain(fn, reps=7):
    import mlx.core as mx

    out = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        out.append(time.perf_counter() - t0)
    return statistics.median(out[1:])


def parity_phase(model, ids, rows_list):
    import mlx.core as mx

    from mlx2.runtime.models.inkling2 import moe_decode_kernel as mdk
    from mlx2.runtime.models.inkling2 import moe_fast
    from mlx2.runtime.models.inkling2 import moe_rows_kernel as mrk
    from mlx2.runtime.models.inkling2.language import _fp32

    cache = prefill(model, ids)
    out = {}
    for rows in rows_list:
        block = ids[-rows:]  # real tokens (re-read text; routing is realistic)
        captured = capture(model, cache, block)
        row = {"visits": 0, "engaged": 0, "declines": {}, "bitwise_vs_shipped": 0,
               "bitwise_vs_one_token": 0, "max_diff_vs_shipped": 0.0,
               "max_diff_vs_one_token": 0.0, "distinct_experts_mean": 0.0}
        pairs = []
        distinct = []
        for moe, x in captured:
            experts, refusal = moe._fast_experts()
            row["visits"] += 1
            reason = refusal or mrk.decline_reason(moe, experts, x)
            if reason is not None:
                row["declines"][reason] = row["declines"].get(reason, 0) + 1
                continue
            row["engaged"] += 1
            order, scores = mdk.route_compiled(x, _fp32(moe, "gate_weight"),
                                               moe.e_score_correction_bias, moe.top_k)
            got = mrk.launch(moe, experts, x, order, scores)
            armed = moe.__dict__.get("_moe_rows_kernel", False)
            object.__setattr__(moe, "_moe_rows_kernel", False)
            try:
                shipped = moe_fast.fused_call(moe, experts, x)
            finally:
                object.__setattr__(moe, "_moe_rows_kernel", armed)
            one = mrk.law(moe, experts, x)
            mx.eval(got, shipped, one)
            row["bitwise_vs_shipped"] += int(mx.array_equal(got, shipped).item())
            row["bitwise_vs_one_token"] += int(mx.array_equal(got, one).item())
            row["max_diff_vs_shipped"] = max(row["max_diff_vs_shipped"], maxdiff(got, shipped))
            row["max_diff_vs_one_token"] = max(row["max_diff_vs_one_token"], maxdiff(got, one))
            distinct.append(len(set(order.reshape(rows, -1)[:, :moe.top_k].reshape(-1).tolist())))
            pairs.append((moe, experts, x, order, scores))
        row["distinct_experts_mean"] = statistics.mean(distinct) if distinct else 0.0

        def shipped_all(pairs=pairs):
            outs = []
            for moe, experts, x, _o, _s in pairs:
                object.__setattr__(moe, "_moe_rows_kernel", False)
                outs.append(moe_fast.fused_call(moe, experts, x))
            return outs

        def kernel_all(pairs=pairs):
            return [mrk.launch(moe, experts, x, o, s) for moe, experts, x, o, s in pairs]

        if pairs:
            row["experts_ms_shipped"] = 1e3 * timed_chain(shipped_all)
            row["experts_ms_kernel"] = 1e3 * timed_chain(kernel_all)
        out[str(rows)] = row
        print(json.dumps({"parity_rows": rows, **row}), flush=True)
    return out


def verify_phase(model, ids, contexts, rows_list, rounds):
    import mlx.core as mx

    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows

    policy = dict(model.kernel_policy)
    arms = {"off": {**policy, "moe_rows_kernel": False}, "on": {**policy, "moe_rows_kernel": True}}
    out = {}
    for context in contexts:
        cache = prefill(model, ids[:context])
        cells = {}
        for rows in rows_list:
            block = mx.array([ids[context:context + rows]])
            times = {"off": [], "on": []}
            logits = {}
            for r in range(rounds + 1):
                order = ("off", "on") if r % 2 == 0 else ("on", "off")
                for arm in order:
                    model.configure_kernels(**arms[arm])
                    txn = HybridVerifyRows([cache]).begin([rows])
                    mx.synchronize()
                    t0 = time.perf_counter()
                    y = model(block, cache=txn.caches)
                    mx.eval(y)
                    mx.synchronize()
                    dt = time.perf_counter() - t0
                    txn.abort()
                    if r:  # round 0 warms both arms (kernel compiles)
                        times[arm].append(dt)
                    logits.setdefault(arm, y)
            receipt = model.kernel_receipt()
            cells[str(rows)] = {
                "ms_off": 1e3 * statistics.median(times["off"]),
                "ms_on": 1e3 * statistics.median(times["on"]),
                "logits_bitwise": bool(mx.array_equal(logits["off"], logits["on"]).item()),
                "argmax_equal": bool(mx.array_equal(mx.argmax(logits["off"], -1),
                                                    mx.argmax(logits["on"], -1)).item()),
                "max_logit_diff": maxdiff(logits["off"], logits["on"]),
                "rows_kernel_calls": receipt.get("moe_rows_kernel_calls"),
                "rows_kernel_declines": receipt.get("moe_rows_kernel_declines"),
            }
            print(json.dumps({"context": context, "rows": rows, **cells[str(rows)]}), flush=True)
        out[str(context)] = cells
        model.configure_kernels(**policy)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("run", type=Path)
    ap.add_argument("--rows", default="2,4,8,16")
    ap.add_argument("--contexts", default="1024,8192")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--parity-context", type=int, default=1024)
    a = ap.parse_args()
    if a.run.exists() and any(a.run.iterdir()):
        raise SystemExit(f"{a.run} is not empty")
    a.run.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import Inkling2Adapter

    adapter = Inkling2Adapter(str(a.model))
    model = adapter.model
    rows_list = [int(r) for r in a.rows.split(",")]
    contexts = [int(c) for c in a.contexts.split(",")]
    ids = corpus(adapter.tokenizer, max(contexts + [a.parity_context]) + max(rows_list) + 8)
    report = {"script": "scripts/research/inkling2_moe_rows_kernel_ab.py",
              "kind": "sanity (no thermal hold; not publishable)",
              "checkpoint": adapter.checkpoint, "source_revision": git_head(),
              "mlx": mx.__version__, "kernel_policy": {k: v for k, v in model.kernel_policy.items() if v}}
    report["parity"] = parity_phase(model, ids[:a.parity_context + max(rows_list)], rows_list)
    report["verify"] = verify_phase(model, ids, contexts, rows_list, a.rounds)
    report["peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    tmp = a.run / "result.json.tmp"
    tmp.write_text(json.dumps(report, indent=1, default=str))
    tmp.replace(a.run / "result.json")


if __name__ == "__main__":
    main()
