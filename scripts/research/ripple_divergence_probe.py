#!/usr/bin/env python3
"""Bounded first-divergence log-probability windows for Ripple A/B mismatches.

For each recorded divergence (prompt, shared prefix, base token, arm token),
recompute the logits at exactly that position under three execution laws:

* ``ordinary``: prompt prefilled in ``--ordinary-step`` rows, the prefix
  decoded one row at a time (the base arm's arithmetic);
* ``chunk_<n>``: prompt prefilled in ``n`` rows, prefix decoded serially;
* ``verify_<k>``: as ordinary, except the last ``k`` prefix tokens run as one
  multi-row forward (a verify-block approximation; the exact PLD block
  composition is not recorded).

Only the already-selected position is captured (AGENTS.md cold/warm rule);
the receipt keeps the top-5 window and both tokens' log-probabilities.
Requires the caller's GPU ownership.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx


def logprob_window(logits, tokens):
    logprobs = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
    order = mx.argsort(-logprobs)[:5]
    mx.eval(logprobs, order)
    top = [(int(t), float(logprobs[t])) for t in order.tolist()]
    return {"top5": top, **{f"token_{t}": float(logprobs[t]) for t in tokens}}


def run(model, prompt, prefix, *, step, verify=1):
    cache = model.make_cache()
    for start in range(0, len(prompt), step):
        mx.eval(model(mx.array([prompt[start:start + step]]), cache=cache))
    serial = prefix[: len(prefix) - verify] if verify > 1 else prefix
    logits = None
    for token in serial:
        logits = model(mx.array([[token]]), cache=cache)
        mx.eval(logits)
    if verify > 1:
        logits = model(mx.array([prefix[len(prefix) - verify:]]), cache=cache)
    return logits[0, -1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--divergences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ordinary-step", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ripple_ab import COPY_SOURCES, copy_prompt

    from mlx2.adapters.ripple import RippleAdapter

    adapter = RippleAdapter(str(args.model))
    model = adapter.model
    prompt = adapter.prompt_tokens({"messages": [{"role": "user", "content": copy_prompt(COPY_SOURCES[0])}]})
    receipt = {"schema": "mlx2.ripple-divergence-probe.v1",
               "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "checkpoint": args.model.name + "@sha256:" + adapter.identity["fingerprint"],
               "prompt_tokens": len(prompt), "rows": {}}
    for name, row in json.loads(args.divergences.read_text()).items():
        prefix, tokens = row["shared_prefix"], (row["base_token"], row["arm_token"])
        laws = {"ordinary": run(model, prompt, prefix, step=args.ordinary_step),
                "chunk_2048": run(model, prompt, prefix, step=2048),
                "verify_5": run(model, prompt, prefix, step=args.ordinary_step, verify=5)}
        windows = {law: logprob_window(logits, tokens) for law, logits in laws.items()}
        for window in windows.values():
            window["selected"] = window["top5"][0][0]
            window["base_minus_arm_nats"] = window[f"token_{tokens[0]}"] - window[f"token_{tokens[1]}"]
        receipt["rows"][name] = {"position": row["first_divergence"], "base_token": tokens[0],
                                 "arm_token": tokens[1], "windows": windows}
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({n: {law: (w["selected"], round(w["base_minus_arm_nats"], 4)) for law, w in r["windows"].items()}
                      for n, r in receipt["rows"].items()}))


if __name__ == "__main__":
    main()
