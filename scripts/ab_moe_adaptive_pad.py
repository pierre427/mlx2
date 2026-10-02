"""A/B the sorted-MoE pad policies on a Flash-Next checkpoint (Metal).

Arms are ``switch_layers._RHS_PAD_POLICY`` values: ``floor`` (today: pad at
>= 3 rows/expert), ``adaptive`` (calibrated per-table cost model) and
``always`` (pad every sorted gather to the streaming kernel).

1. Row-cost curve: one trunk forward + head on fresh caches at each row
   count, arms interleaved (ABBA) per rep after a two-pass warm-up; median ms.
2. Exactness vs ``floor``: last-row logits and every MoE block output.
3. Slice invariance per arm: an ~8K-token real prompt prefilled in slices of
   each ``--slices`` size; last-row logits compared with the first schedule
   (bytes / max|diff|) and B1 greedy tokens.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/ab_moe_adaptive_pad.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --out ab.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ROWS = [32, 48, 64, 80, 96, 112, 128, 144, 153, 154, 176, 204, 205, 256, 384, 512, 768, 1024]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", nargs="+", default=["floor", "adaptive", "always"])
    ap.add_argument("--rows", nargs="+", type=int, default=ROWS)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--exact-rows", nargs="*", type=int, default=[64, 128, 176, 512])
    ap.add_argument("--slices", nargs="*", default=["512", "448", "1024", "2048"])
    ap.add_argument("--decode", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    # Construct the adapter before any mlx2.runtime model module is imported.
    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    import mlx.core as mx

    from mlx2.runtime.models import switch_layers as SL

    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(4 << 30)

    def swapouts():
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
        return int(next(l for l in out.splitlines() if l.startswith("Swapouts")).split(":")[1].strip(" ."))

    swap0 = swapouts()
    tok = adapter.tokenizer
    text = (ROOT / "docs" / "SERVING.md").read_text()
    ids = tok.encode(text, add_special_tokens=False)[: 8192 - 64]
    prompt = list(adapter.prompt_tokens({"messages": [{"role": "user", "content":
                  "Summarise the following excerpt in three sentences.\n\n" + tok.decode(ids)}]}))
    assert len(prompt) > max(a.rows)

    def set_arm(arm):
        SL._RHS_PAD_POLICY = arm

    def forward(m, capture=None):
        cache = model.make_cache()
        mx.synchronize()
        t0 = time.perf_counter()
        hidden, _ = model.mtp_backbone(mx.array([prompt[:m]], mx.uint32), cache=cache)
        logits = model.logits(hidden[:, -1:, :])
        mx.eval(logits)
        ms = 1e3 * (time.perf_counter() - t0)
        return ms, logits

    out = {"model": a.model, "mlx": mx.__version__, "arms": a.arms,
           "cost_models": {str(k): v for k, v in SL._PAD_COST_MODELS.items()}}

    # 1. curve
    times = {(arm, m): [] for arm in a.arms for m in a.rows}
    for arm in a.arms:
        set_arm(arm)
        for m in a.rows:
            forward(m)
            forward(m)
    mx.clear_cache()
    choices = {}
    for rep in range(a.reps):
        arms = a.arms if rep % 2 == 0 else a.arms[::-1]
        for m in (a.rows if rep % 2 == 0 else a.rows[::-1]):
            for arm in arms:
                set_arm(arm)
                SL.moe_pad_status(reset=True)
                times[(arm, m)].append(forward(m)[0])
                choices[f"{arm}:{m}"] = SL.moe_pad_status()["choices"]
        mx.clear_cache()
        if swapouts() - swap0 > 20000:
            raise SystemExit("aborting: swap")
        print(f"rep {rep} done", flush=True)
    curve = {arm: {str(m): round(statistics.median(times[(arm, m)]), 1) for m in a.rows} for arm in a.arms}
    out["curve"] = curve
    out["curve_spread"] = {arm: {str(m): [round(min(times[(arm, m)]), 1), round(max(times[(arm, m)]), 1)]
                                 for m in a.rows} for arm in a.arms}
    out["choices"] = choices
    print("CURVE rows      " + " ".join(f"{m:>6}" for m in a.rows), flush=True)
    for arm in a.arms:
        print(f"CURVE {arm:9s} " + " ".join(f"{curve[arm][str(m)]:6.0f}" for m in a.rows), flush=True)

    # 2. exactness vs the first arm
    moe_cls = type(model.language_model.model.layers[0].mlp)
    orig = moe_cls.__call__
    captured = []

    def cap(self, *args, **kwargs):
        y = orig(self, *args, **kwargs)
        captured.append(y)
        return y

    exact = {}
    moe_cls.__call__ = cap
    try:
        for m in a.exact_rows:
            ref = None
            for arm in a.arms:
                set_arm(arm)
                captured.clear()
                _, logits = forward(m)
                mx.eval(captured)
                outs = [logits.astype(mx.float32)] + list(captured)
                if ref is None:
                    ref = (arm, outs)
                    continue
                exact[f"{m}:{ref[0]}vs{arm}"] = {
                    "logits_identical": bool(mx.array_equal(outs[0], ref[1][0]).item()),
                    "logits_max_abs_diff": float(mx.max(mx.abs(outs[0] - ref[1][0])).item()),
                    "argmax_equal": bool((mx.argmax(outs[0], -1) == mx.argmax(ref[1][0], -1)).all().item()),
                    "moe_layers_identical": sum(bool(mx.array_equal(x, y).item())
                                                for x, y in zip(outs[1:], ref[1][1:])),
                    "moe_layers": len(outs) - 1,
                }
                print(f"EXACT {m}:{ref[0]}vs{arm} {json.dumps(exact[f'{m}:{ref[0]}vs{arm}'])}", flush=True)
            mx.clear_cache()
    finally:
        moe_cls.__call__ = orig
    out["exactness"] = exact

    # 3. slice invariance per arm
    def prefill(step):
        n = len(prompt)
        step = n if step == "all" else int(step)
        cache = model.make_cache()
        pos = 0
        while pos < n:
            end = min(n, pos + step)
            hidden = model.language_model.model(mx.array([prompt[pos:end]], mx.uint32), cache)
            logits = model.logits(hidden[:, -1:, :])
            mx.eval(logits)
            pos = end
        last = logits[:, -1, :].astype(mx.float32)
        toks, nxt = [], int(mx.argmax(last, -1).item())
        for _ in range(a.decode):
            toks.append(nxt)
            nxt = int(mx.argmax(model(mx.array([[nxt]], mx.uint32), cache=cache)[:, -1, :], -1).item())
        del cache
        mx.clear_cache()
        return last, toks

    inv = {}
    for arm in a.arms:
        set_arm(arm)
        ref = None
        for step in a.slices:
            last, toks = prefill(step)
            if ref is None:
                ref = (step, last, toks)
                inv[f"{arm}:{step}"] = {"tokens": toks}
                continue
            inv[f"{arm}:{step}"] = {
                "vs": ref[0],
                "logits_identical": bool(mx.array_equal(last, ref[1]).item()),
                "logits_max_abs_diff": float(mx.max(mx.abs(last - ref[1])).item()),
                "tokens_identical": toks == ref[2],
                "tokens": toks,
            }
            print(f"SLICE {arm}:{step} vs {ref[0]} " + json.dumps(
                {k: v for k, v in inv[f'{arm}:{step}'].items() if k != 'tokens'}), flush=True)
    out["slice_invariance"] = inv
    out["swapouts_delta_pages"] = swapouts() - swap0
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
