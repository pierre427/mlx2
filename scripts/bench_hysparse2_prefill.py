"""Time HySparse2 prefill paths at paper scale with random weights (GPU).

Arms, interleaved per round (back-to-back runs drift ~10% on this machine):

* ``serving_loop`` – mlx2's batch prefill loop: ``model(chunk, cache)`` on
  every chunk of tokens[:-1] with only cache state evaluated (MLX laziness
  prunes the cross-decoder), then the last token as a decode step (the YOCO
  split).
* ``paper_exit`` – ``Model.prefill(suffix_bound=False)``: the paper's early
  exit, the full self-decoder on every row.
* ``suffix_bound`` – ``Model.prefill(suffix_bound=True)``.
* ``no_exit`` (``--no-exit-max``) – the serving loop but forcing each chunk's
  logits, i.e. what running the cross-decoder on every prompt row costs.

Random weights measure time and memory only. Last-token logits of the arms are
compared so a silently different computation shows up.
"""

import argparse
import json
import time

import mlx.core as mx
import mlx.nn as nn

from mlx2.runtime.models.hysparse2 import Model, ModelArgs


def build(intermediate: int, dtype) -> Model:
    mx.random.seed(0)
    model = Model(ModelArgs(vocab_size=32000, hidden_size=2048, intermediate_size=intermediate))
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


def serving_loop(model, toks, chunk, force_logits=False):
    cache = model.make_cache()
    body = toks[:, :-1]
    for c0 in range(0, body.shape[1], chunk):
        out = model(body[:, c0 : c0 + chunk], cache=cache)
        mx.eval(out if force_logits else [c.state for c in cache])
        mx.clear_cache()
    logits = model(toks[:, -1:], cache=cache)[:, -1]
    mx.eval(logits)
    return logits


def prefill(model, toks, chunk, suffix_bound):
    cache = model.make_cache()
    logits = model.prefill(toks, cache, chunk_size=chunk, suffix_bound=suffix_bound)[:, -1]
    mx.eval(logits)
    mx.clear_cache()
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384, 65536, 131072])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--intermediate", type=int, default=4096)
    ap.add_argument("--no-exit-max", type=int, default=16384)
    ap.add_argument("--no-exit-chunk", type=int, default=256)
    ap.add_argument("--output")
    args = ap.parse_args()

    model = build(args.intermediate, mx.bfloat16)
    n_params = sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))
    arms = {
        "serving_loop": lambda t: serving_loop(model, t, args.chunk),
        "paper_exit": lambda t: prefill(model, t, args.chunk, False),
        "suffix_bound": lambda t: prefill(model, t, args.chunk, True),
        "no_exit": lambda t: serving_loop(model, t, args.no_exit_chunk, True),
    }
    prefill(model, mx.random.randint(0, 32000, (1, 3000)), args.chunk, True)  # warmup
    serving_loop(model, mx.random.randint(0, 32000, (1, 600)), args.chunk)
    rows = []
    for T in args.lengths:
        toks = mx.random.randint(0, 32000, (1, T))
        names = [n for n in arms if n != "no_exit" or T <= args.no_exit_max]
        times = {n: [] for n in names}
        peaks = {}
        logits = {}
        for r in range(args.rounds):
            order = names if r % 2 == 0 else names[::-1]
            for name in order:
                mx.reset_peak_memory()
                t0 = time.perf_counter()
                logits[name] = arms[name](toks)
                times[name].append(time.perf_counter() - t0)
                peaks[name] = mx.get_peak_memory() / 2**30
        ref = logits["serving_loop"].astype(mx.float32)
        row = {"tokens": T}
        for name in names:
            best = min(times[name])
            diff = float(mx.abs(logits[name].astype(mx.float32) - ref).max())
            row[name] = {
                "seconds": [round(x, 3) for x in times[name]],
                "best_s": round(best, 3),
                "tok_per_s": round(T / best, 1),
                "peak_gib": round(peaks[name], 2),
                "max_abs_logit_diff_vs_serving_loop": diff,
                "argmax_agrees": bool(
                    mx.argmax(logits[name], -1).item() == mx.argmax(ref, -1).item()
                ),
            }
        base = row["paper_exit"]["best_s"]
        row["speedup_suffix_vs_paper_exit"] = round(base / row["suffix_bound"]["best_s"], 2)
        if "no_exit" in row:
            row["speedup_paper_exit_vs_no_exit"] = round(row["no_exit"]["best_s"] / base, 2)
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "device": str(mx.default_device()),
        "mlx": mx.__version__,
        "params": n_params,
        "config": {"hidden": 2048, "layers": 49, "heads": "64q/1kv x 256", "intermediate": args.intermediate, "dtype": "bfloat16", "chunk": args.chunk},
        "rows": rows,
    }
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
