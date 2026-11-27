"""Dump a donor model's top-k next-token log-probs over the fluids box (forward only).

For each split (fit / val / held) we take whole documents in id order until a token budget,
tokenize with the donor's own tokenizer, run causal forward passes in windows of --window
tokens (each window is independent: no cache), and save per position:
  ids[t]        token at t
  top_ids[t]    donor top-k candidates for position t+1 (int32)
  top_lp[t]     their log-probs (float16)
  lp_next[t]    donor log-prob of the actual token t+1 (float32; nan at window ends)
Runs under mlx-uag venv (mlx_lm unified). One model per process.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load


def run_plan(plan):
    """Run the remaining donors in order inside the same GPU lease, each in its own process."""
    for step in plan["steps"]:
        left = plan["deadline"] - time.time() - 90
        if left < 120:
            print(f"skip {step['out']}: window exhausted", flush=True)
            continue
        cap = min(step["cap"], left)
        cmd = [sys.executable, __file__, "--model", step["model"], "--corpus", "fluids.jsonl",
               "--out", step["out"], "--budget", step["budget"], "--max-seconds", str(int(cap))]
        print(time.strftime("%T"), "plan start", step["out"], f"cap {cap:.0f}s", flush=True)
        with open(step["out"] + ".log", "a") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT,
                                 env={**os.environ, "PLAN2_CHILD": "1"})
        print(time.strftime("%T"), "plan done", step["out"], "rc", rc, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", default="fit=800000,val=60000,held=150000")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--max-seconds", type=float, default=900)
    args = ap.parse_args()
    if Path("plan2.json").exists() and args.out == "donor_nem30" and not os.environ.get("PLAN2_CHILD"):
        return run_plan(json.load(open("plan2.json")))
    t0 = time.time()
    model, tok = load(args.model)
    print(f"loaded {args.model} in {time.time() - t0:.1f}s", flush=True)

    docs = {}
    for line in open(args.corpus):
        r = json.loads(line)
        docs.setdefault(r["split"], []).append((r["id"], r["text"]))
    budget = dict(kv.split("=") for kv in args.budget.split(","))
    out = {"model": args.model, "window": args.window, "topk": args.topk, "packed": True}
    for split in [s for s in ("held", "val", "fit") if s in budget]:
        b = int(budget[split])
        seqs, n = [], 0
        for _, text in sorted(docs[split]):
            ids = tok.encode(text)
            if n + len(ids) > b:
                break
            seqs.append(ids)
            n += len(ids)
        doc_start = np.cumsum([0] + [len(x) for x in seqs[:-1]]).astype(np.int64)
        arr = np.concatenate([np.array(x, np.int32) for x in seqs])
        ti = np.zeros((len(arr), args.topk), np.int32)
        tl = np.full((len(arr), args.topk), -30, np.float16)
        ln = np.full(len(arr), np.nan, np.float32)
        ts = time.time()
        stopped, done = False, len(arr)
        for s in range(0, len(arr), args.window):
            chunk = mx.array(arr[s:s + args.window])[None]
            logits = model(chunk)[0].astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            k = mx.argpartition(-lp, args.topk - 1, axis=-1)[:, :args.topk]
            kv = mx.take_along_axis(lp, k, axis=-1)
            L = chunk.shape[1]
            nxt = mx.array(arr[s + 1:s + L])[:, None]
            lnx = mx.take_along_axis(lp[:L - 1], nxt, axis=-1)[:, 0] if L > 1 else mx.zeros((0,))
            mx.eval(k, kv, lnx)
            ti[s:s + L] = np.array(k)
            tl[s:s + L] = np.array(kv).astype(np.float16)
            ln[s:s + L - 1] = np.array(lnx)
            if time.time() - t0 > args.max_seconds:
                stopped, done = True, s + L
                break
        # keep whole docs only
        keep_docs = int(np.searchsorted(doc_start, done, side="right"))
        if stopped:
            keep_docs = max(keep_docs - 1, 1)
        end = int(doc_start[keep_docs]) if keep_docs < len(doc_start) else len(arr)
        tokens = end
        dt = time.time() - ts
        print(f"{split}: {keep_docs} docs {tokens} tokens {dt:.1f}s ({done / max(dt, 1e-9):.0f} tok/s)"
              f"{' STOPPED at time cap' if stopped else ''}", flush=True)
        np.savez(f"{args.out}.{split}.npz", ids=arr[:end], top_ids=ti[:end], top_lp=tl[:end],
                 lp_next=ln[:end], doc_start=doc_start[:keep_docs])
        out[split] = {"docs": keep_docs, "tokens": tokens, "seconds": dt, "stopped": stopped}
    out["peak_gib"] = mx.get_peak_memory() / 2 ** 30
    out["total_seconds"] = time.time() - t0
    json.dump(out, open(f"{args.out}.json", "w"), indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
