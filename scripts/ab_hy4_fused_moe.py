#!/usr/bin/env python3
"""A/B of the Hy4-flash opt-in MoE routes: fused [gate|up] banks and NAX prefill.

One arm per process (the fused banks replace the split ones at load, and a
second resident copy of a Hy4 checkpoint does not fit), so each run writes
bit hashes and timings and ``compare`` checks them across arms:

    run      load the adapter with the arm's policy, then
             1. prefill a fixed prompt at each chunk size: sha256 of the
                final hidden of every row and of the last-row logits, median
                seconds over --reps;
             2. B=1 greedy decode: tokens, sha256 of every step's logits, tok/s;
             3. --lanes concurrent greedy requests (BatchGenerator): tokens, tok/s;
             4. AST verify: --lanes lanes x --verify-rows rows as one
                chain-packed joint forward (Hy4ChainTreeExecutor), aborted:
                sha256 of the logits, median seconds;
             5. receipts: moe_fused_gate_up / moe_nax_gather diagnostics,
                checkpoint, mlx2 commit, mlx version, swap delta.
    compare  arm JSONs, first is the reference: bit identity per cell and
             timing ratios.  A non-reference arm whose selected mechanism
             has observed_used false is reported as refused.

GPU (one model per process, the GPU owned by this run):

    PYTHONPATH=src .venv/bin/python scripts/ab_hy4_fused_moe.py run --i-own-the-gpu \\
        --model <hy4 artifact> --arm split --out split.json
    PYTHONPATH=src .venv/bin/python scripts/ab_hy4_fused_moe.py run --i-own-the-gpu \\
        --model <hy4 artifact> --arm fused --out fused.json
    PYTHONPATH=src .venv/bin/python scripts/ab_hy4_fused_moe.py compare split.json fused.json

``--cpu`` runs on the CPU device instead (smoke test on tiny fixtures).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = {
    "split": {},
    "fused": {"moe_fused_gate_up": True},
    "split+nax": {"moe_nax": "gather"},
    "fused+nax": {"moe_fused_gate_up": True, "moe_nax": "gather"},
}


def _swapouts():
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
        return int(next(l for l in out.splitlines() if l.startswith("Swapouts")).split(":")[1].strip(" ."))
    except Exception:  # noqa: BLE001
        return None


def _commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _sha(arrays):
    import mlx.core as mx
    import numpy as np

    h = hashlib.sha256()
    for x in arrays:
        mx.eval(x)
        a = np.array(x.view(mx.uint16) if x.dtype == mx.bfloat16 else x)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _prompt(adapter, n):
    tok = adapter.tokenizer
    text = "\n\n".join((ROOT / p).read_text() for p in ("docs/SERVING.md", "docs/QUALIFICATION.md"))
    ids = list(tok.encode(text, add_special_tokens=False))
    vocab = adapter.model.args.vocab_size
    ids = [i for i in ids if 3 < i < vocab]
    if not ids:  # a tokenizer that maps the text to specials only (tiny fixtures)
        ids = [4 + (7 * i) % (vocab - 4) for i in range(n)]
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def run(a):
    import mlx.core as mx

    if a.cpu:
        mx.set_default_device(mx.cpu)
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor
    from mlx2.runtime.generate import BatchGenerator

    swap0 = _swapouts()
    adapter = HYV4FlashAdapter(a.model, execution_policy=ARMS[a.arm])
    try:
        model = adapter.model
        prompt = _prompt(adapter, a.prompt_tokens)
        out = {"arm": a.arm, "policy": ARMS[a.arm], "mlx": mx.__version__, "mlx2_commit": _commit(),
               "device": str(mx.default_device()),
               "checkpoint": adapter.diagnostics()["checkpoint"], "prompt_tokens": len(prompt),
               "prefill": {}, "cells": {}}

        def prefill(chunk, keep=True):
            cache = model.make_cache()
            hs = []
            mx.synchronize()
            t0 = time.perf_counter()
            for s in range(0, len(prompt), chunk):
                h = model.hidden(mx.array([prompt[s:s + chunk]]), cache)
                mx.eval(h)
                if keep:
                    hs.append(h)
            logits = model.logits(h[:, -1:])
            mx.eval(logits)
            mx.synchronize()
            return time.perf_counter() - t0, hs, logits, cache

        for chunk in a.chunks:
            times = []
            for rep in range(a.reps + 1):
                dt, hs, logits, cache = prefill(chunk, keep=rep == 0)
                if rep == 0:
                    out["cells"][f"prefill:{chunk}"] = {"hidden_sha256": _sha(hs), "logits_sha256": _sha([logits])}
                else:
                    times.append(dt)
                del hs, cache
                mx.clear_cache()
            out["prefill"][str(chunk)] = {"median_s": statistics.median(times) if times else None,
                                          "tok_s": len(prompt) / statistics.median(times) if times else None}

        # B=1 greedy decode.
        cache = model.make_cache()
        logits = model(mx.array([prompt[: a.decode_prompt]]), cache)[:, -1]
        tokens, rows = [], []
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.decode):
            t = int(mx.argmax(logits, axis=-1).item())
            tokens.append(t)
            logits = model(mx.array([[t]]), cache)[:, -1]
            rows.append(logits)
        mx.synchronize()
        dt = time.perf_counter() - t0
        out["cells"]["decode_b1"] = {"tokens": tokens, "logits_sha256": _sha(rows)}
        out["decode_b1"] = {"tok_s": a.decode / dt}
        del cache, rows
        mx.clear_cache()

        # Concurrent lanes.
        prompts = [prompt[i * 37: i * 37 + a.decode_prompt - 3 * i] for i in range(a.lanes)]
        batch = BatchGenerator(model, max_tokens=a.decode, completion_batch_size=a.lanes,
                               prefill_batch_size=a.lanes)
        uids = batch.insert(prompts, max_tokens=[a.decode] * a.lanes)
        got = {u: [] for u in uids}
        t0 = time.perf_counter()
        while not all(len(v) >= a.decode for v in got.values()):
            result = batch.next()
            for r in (result[1] if isinstance(result, tuple) else result):
                got[r.uid].append(int(r.token))
        dt = time.perf_counter() - t0
        out["cells"][f"batch_{a.lanes}"] = {"tokens": [got[u][: a.decode] for u in uids]}
        out[f"batch_{a.lanes}"] = {"tok_s_incl_prefill": a.lanes * a.decode / dt}
        del batch
        mx.clear_cache()

        # AST verify: lanes x verify_rows, chain-packed joint forward, aborted.
        trees = [([prompt[100 + 7 * i + j] for j in range(a.verify_rows)],
                  [-1] + list(range(a.verify_rows - 1))) for i in range(a.lanes)]
        times, sha = [], None
        for rep in range(a.reps + 1):
            caches = []
            for p in prompts:
                c = model.make_cache()
                mx.eval(model(mx.array([p]), c))
                caches.append(c)
            executor = Hy4ChainTreeExecutor.create()
            mx.synchronize()
            t0 = time.perf_counter()
            logits, features, tx = executor.forward_many(
                model, [t for t, _ in trees], [p for _, p in trees], caches,
                (model.args.num_hidden_layers - 1,), cached=False, configured_limit=a.lanes)
            mx.eval(logits, features)
            mx.synchronize()
            if rep == 0:
                sha = _sha([logits, features])
            else:
                times.append(time.perf_counter() - t0)
            tx.abort()
            del caches
            mx.clear_cache()
        out["cells"][f"verify_{a.lanes}x{a.verify_rows}"] = {"logits_sha256": sha}
        out[f"verify_{a.lanes}x{a.verify_rows}"] = {"median_s": statistics.median(times) if times else None}

        d = adapter.diagnostics()
        out["receipts"] = {"moe_fused_gate_up": d["moe_fused_gate_up"],
                           "moe_nax_gather": d["moe_nax_gather"], "layout": adapter.layout}
        swap1 = _swapouts()
        out["swapouts_delta"] = None if swap0 is None or swap1 is None else swap1 - swap0
    finally:
        adapter.close()
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1))


def compare(paths):
    arms = [json.loads(Path(p).read_text()) for p in paths]
    ref = arms[0]
    report = {"reference": ref["arm"], "checkpoint": ref["checkpoint"], "arms": {}}
    for arm in arms[1:]:
        if arm["checkpoint"] != ref["checkpoint"]:
            raise SystemExit("arms ran different checkpoints")
        cells = {k: arm["cells"].get(k) == v for k, v in ref["cells"].items()}
        row = {"bit_identical": all(cells.values()), "cells": cells}
        timing = {}
        for chunk, v in ref["prefill"].items():
            w = arm["prefill"].get(chunk, {})
            if v.get("median_s") and w.get("median_s"):
                timing[f"prefill:{chunk}"] = v["median_s"] / w["median_s"]
        for key in ref:
            if key.startswith(("decode_b1", "batch_")) and key in arm:
                k = next(iter(ref[key]))
                timing[key] = arm[key][k] / ref[key][k]
            if key.startswith("verify_") and key in arm and ref[key]["median_s"] and arm[key]["median_s"]:
                timing[key] = ref[key]["median_s"] / arm[key]["median_s"]
        row["speedup_vs_reference"] = timing
        refused = []
        for name in ("moe_fused_gate_up", "moe_nax_gather"):
            r = arm["receipts"].get(name) or {}
            if r.get("selected") and not r.get("observed_used"):
                refused.append(f"{name} selected but not observed_used")
        row["refused"] = refused
        report["arms"][arm["arm"]] = row
    print(json.dumps(report, indent=1))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--arm", choices=sorted(ARMS), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--prompt-tokens", type=int, default=4096)
    r.add_argument("--chunks", nargs="+", type=int, default=[512, 2048])
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--decode", type=int, default=64)
    r.add_argument("--decode-prompt", type=int, default=256)
    r.add_argument("--lanes", type=int, default=4)
    r.add_argument("--verify-rows", type=int, default=6)
    r.add_argument("--i-own-the-gpu", action="store_true")
    r.add_argument("--cpu", action="store_true", help="run on the CPU device (smoke test)")
    c = sub.add_parser("compare")
    c.add_argument("paths", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        return compare(a.paths)
    if not (a.cpu or a.i_own_the_gpu):
        ap.error("refusing Metal execution without --i-own-the-gpu (or pass --cpu)")
    if Path(a.out).exists() and Path(a.out).stat().st_size:
        ap.error(f"{a.out} exists; refusing to overwrite evidence")
    return run(a)


if __name__ == "__main__":
    main()
