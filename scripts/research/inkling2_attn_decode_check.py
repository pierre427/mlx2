"""GPU check of ``inkling2_attn_decode_kernel`` on the real checkpoint.

Original mlx2 script (2026-11-22). Loads the artifact with its shipped kernel
defaults, then for each context (default 1K and 16K of repository text,
prefilled in 512-token chunks):

1. ``parity``: during one decode step with the option on, every visit's
   Metal front end (``attn_decode.prep``) is compared on the same live
   tensors with the composed path's own intermediates
   (``attn_decode.parity_probe`` / ``composed_reference``: the modules and
   helpers ``Attention.__call__`` runs) and with ``prep_reference`` (the law
   in plain MLX ops): bitwise flags and max |diff| per output, per visit kind.
2. ``tokens``: greedy decode of ``steps`` tokens with the option off, off
   again (the determinism floor: the same arm twice from its own prefill)
   and on; max |dlogit| per step against the first off arm. Exact, or the first divergent step is classified
   with the log-probability half of ``mlx2.near_tie.near_tie_equivalent`` (0.5 nat; a row is near_tie_equivalent only when both continuations also pass a functional oracle, AGENTS.md) from the top-8
   log-probability windows of both arms at that step only (the logits each
   arm already computed; nothing extra runs during the decode).
3. ``counters``: the ``kernel_receipt()["attn_decode"]`` delta of every on
   step (48 calls expected) and the declines seen.
4. ``ab``: decode tok/s, arms interleaved in blocks on one cache (off, on,
   on + inkling2_fused_projections, repeated), one warm-up block per arm
   dropped. Sanity numbers under whatever contention the host has, no
   thermal hold; not publishable statistics.

Usage: python scripts/research/inkling2_attn_decode_check.py <model dir> <out.json>
       [--contexts 1024 16384] [--steps 64] [--blocks 6] [--block-steps 16]
The caller owns the GPU (scripts/run_with_gpu_locks.py).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WINDOW = 8


def _corpus():
    spec = importlib.util.spec_from_file_location(
        "stage_price", ROOT / "scripts/research/inkling2_decode_stage_price.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.corpus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--contexts", type=int, nargs="+", default=[1024, 16384])
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--block-steps", type=int, default=16)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import Inkling2Adapter
    from mlx2.near_tie import near_tie_equivalent
    from mlx2.runtime.models.inkling2 import attn_decode

    adapter = Inkling2Adapter(str(a.model))
    model = adapter.model
    base_policy = dict(model.kernel_policy)
    corpus = _corpus()

    def configure(on, fused_projections=False):
        model.configure_kernels(**{**base_policy, "attn_decode_kernel": on,
                                   "fused_projections": fused_projections})

    def receipt():
        return model.kernel_receipt()["attn_decode"]

    def prefill(ids):
        cache = model.make_cache()
        for s in range(0, len(ids), 512):
            logits = model(mx.array([ids[s:s + 512]]), cache=cache)
            mx.eval(logits)
        return cache, logits[0, -1]

    def window(logits):
        logp = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32))
        top = mx.argsort(-logp)[:WINDOW]
        return {int(t): float(v) for t, v in zip(top.tolist(), logp[top].tolist())}

    def decode(cache, last, steps, on, per_step=None):
        token = int(mx.argmax(last).item())
        tokens, rows, times = [], [], []
        for _ in range(steps):
            before = receipt() if per_step is not None else None
            mx.synchronize()
            t0 = time.perf_counter()
            logits = model(mx.array([[token]]), cache=cache)[0, -1]
            token = int(mx.argmax(logits).item())
            mx.synchronize()
            times.append(time.perf_counter() - t0)
            tokens.append(token)
            rows.append(logits)
            if per_step is not None:
                after = receipt()
                per_step.append({"calls": after["calls"] - before["calls"],
                                 "declines": {k: v - before["declines"].get(k, 0)
                                              for k, v in after["declines"].items()
                                              if v != before["declines"].get(k, 0)}})
        return tokens, rows, times, token

    result = {"script": "scripts/research/inkling2_attn_decode_check.py",
              "checkpoint": adapter.checkpoint,
              "mlx2": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
              "mlx_version": mx.__version__, "kernel_policy_base": base_policy,
              "kind": "sanity, one process, no thermal hold; A/B arms interleaved",
              "contexts": {}}
    try:
        for context in a.contexts:
            ids = corpus(adapter.tokenizer, context)
            row = {}
            # 1. parity on live tensors (one decode step, option on).
            configure(True)
            cache, last = prefill(ids)
            diffs = {}
            original = attn_decode.prep

            def checked(attn, *args):
                out = original(attn, *args)
                ref = attn_decode.prep_reference(attn, *args)
                kind = "sliding" if attn.is_sliding else "global"
                held = diffs.setdefault(kind, {"visits": 0, "outputs": {}})
                held["visits"] += 1
                for name, x, y in zip(("q", "k", "v", "rel", "sink", "k_state", "v_state"),
                                      out, ref):
                    d = mx.abs(x.astype(mx.float32) - y.astype(mx.float32))
                    entry = held["outputs"].setdefault(
                        name, {"bitwise_visits": 0, "max_abs": 0.0, "mismatch_elems": 0,
                               "elems": 0})
                    exact = bool(mx.array_equal(x, y).item())
                    entry["bitwise_visits"] += int(exact)
                    entry["max_abs"] = max(entry["max_abs"], float(d.max().item()))
                    entry["mismatch_elems"] += int((d > 0).sum().item())
                    entry["elems"] += int(d.size)
                return out

            attn_decode.prep = checked
            records = []
            try:
                with attn_decode.parity_probe(records):
                    logits = model(mx.array([[int(mx.argmax(last).item())]]), cache=cache)
                    mx.eval(logits)
            finally:
                attn_decode.prep = original
            row["parity_vs_prep_reference"] = diffs
            composed = {}
            for record in records:
                held = composed.setdefault(record["kind"], {"visits": 0, "outputs": {}})
                held["visits"] += 1
                for name in attn_decode.OUTPUTS:
                    x, y = record["front"][name], record["composed"][name]
                    d = mx.abs(x.astype(mx.float32) - y.astype(mx.float32))
                    entry = held["outputs"].setdefault(
                        name, {"bitwise_visits": 0, "max_abs": 0.0, "mismatch_elems": 0,
                               "elems": 0})
                    entry["bitwise_visits"] += int(mx.array_equal(x, y).item())
                    entry["max_abs"] = max(entry["max_abs"], float(d.max().item()))
                    entry["mismatch_elems"] += int((d > 0).sum().item())
                    entry["elems"] += int(d.size)
            row["parity"] = composed
            diffs = composed
            del cache
            mx.clear_cache()
            # 2-3. greedy tokens off vs on from the same prefill.
            arms = {}
            for name, on in (("off", False), ("off_control", False), ("on", True)):
                configure(on)
                cache, last = prefill(ids)
                per_step = [] if on else None
                tokens, rows, times, _ = decode(cache, last, a.steps, on, per_step)
                arms[name] = (tokens, rows, times, per_step)
                del cache
                mx.clear_cache()

            def per_step_dlogit(other):
                return [float(mx.abs(x - y).max().item())
                        for x, y in zip(arms["off"][1], arms[other][1])]

            control = per_step_dlogit("off_control")
            row["control"] = {"tokens_equal": arms["off"][0] == arms["off_control"][0],
                              "max_abs_dlogit_per_step": control,
                              "bitwise_steps": sum(1 for d in control if d == 0.0)}
            arms[False], arms[True] = arms["off"], arms["on"]
            off_t, on_t = arms[False][0], arms[True][0]
            first = next((i for i, (x, y) in enumerate(zip(off_t, on_t)) if x != y), None)
            tok = {"off": off_t, "on": on_t, "exact": first is None, "first_divergence": first}
            if first is not None:
                left, right = window(arms[False][1][first]), window(arms[True][1][first])
                tok["window_off"], tok["window_on"] = left, right
                tok["near_tie_logprob_test"] = near_tie_equivalent(off_t[first], left,
                                                                 on_t[first], right)
                tok["logit_max_abs_before_divergence"] = max(
                    (float(mx.abs(x - y).max().item())
                     for x, y in zip(arms[False][1][:first], arms[True][1][:first])),
                    default=0.0)
            else:
                tok["logit_max_abs"] = max(float(mx.abs(x - y).max().item())
                                           for x, y in zip(arms[False][1], arms[True][1]))
                tok["logits_bitwise_steps"] = sum(
                    int(mx.array_equal(x, y).item())
                    for x, y in zip(arms[False][1], arms[True][1]))
            tok["max_abs_dlogit_per_step"] = per_step_dlogit("on")
            row["tokens"] = tok
            steps = arms[True][3]
            row["counters"] = {"steps": len(steps),
                               "steps_with_48_calls": sum(1 for s in steps if s["calls"] == 48),
                               "declines_during_decode": sum(sum(s["declines"].values())
                                                             for s in steps),
                               "receipt_after": receipt()}
            row["sequential_step_ms_median"] = {
                "off": 1e3 * statistics.median(arms[False][2][4:]),
                "off_control": 1e3 * statistics.median(arms["off_control"][2][4:]),
                "on": 1e3 * statistics.median(arms[True][2][4:])}
            del arms
            # 4. interleaved A/B on one cache.
            plan = [("off", False, False), ("on", True, False), ("on_fp", True, True)]
            configure(False)
            cache, last = prefill(ids)
            token_logits = last
            times = {name: [] for name, _, _ in plan}
            for block in range(a.blocks + 1):
                for name, on, fp in plan:
                    configure(on, fp)
                    _, rows, ts, token = decode(cache, token_logits, a.block_steps, on)
                    token_logits = rows[-1]
                    if block:
                        times[name].extend(ts)
            row["ab"] = {name: {"step_ms_median": 1e3 * statistics.median(ts),
                                "tok_s": len(ts) / sum(ts), "steps": len(ts)}
                         for name, ts in times.items()}
            del cache
            mx.clear_cache()
            result["contexts"][str(context)] = row
            print(json.dumps({"context": context, "parity": {
                k: {n: (e["bitwise_visits"], round(e["max_abs"], 6)) for n, e in v["outputs"].items()}
                for k, v in diffs.items()}, "exact": tok["exact"],
                "first_divergence": first, "near_tie_logprob_test": tok.get("near_tie_logprob_test"),
                "dlogit_step0": tok["max_abs_dlogit_per_step"][0],
                "control_bitwise_steps": row["control"]["bitwise_steps"],
                "counters": row["counters"]["steps_with_48_calls"],
                "ab": {k: round(v["tok_s"], 2) for k, v in row["ab"].items()}}), flush=True)
        configure(False)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=1, default=str))
        tmp.rename(a.out)
    finally:
        adapter.close()
    ok = all(r["counters"]["steps_with_48_calls"] == r["counters"]["steps"]
             for r in result["contexts"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
