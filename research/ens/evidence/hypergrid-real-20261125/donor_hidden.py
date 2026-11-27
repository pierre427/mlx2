"""Qwen3 donor dump with hidden states (forward only, packed 2048-token windows).

Saves per split <out>.<split>.npz: ids, doc_start, top_ids/top_lp (top-k next-token), lp_next,
and <out>.<split>.h<L>.npy (float16 [tokens, hidden]) for each captured layer L.
L = number of blocks applied; L == num_layers means the final *normed* state (what the
LM head reads). Run under mlx-uag venv.
"""

import argparse
import json
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", default="held=150000,val=60000,fit=800000")
    ap.add_argument("--layers", default="14,28")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--max-seconds", type=float, default=840)
    args = ap.parse_args()
    t0 = time.time()
    model, tok = load(args.model)
    inner = model.model
    nL = len(inner.layers)
    caps = [int(x) for x in args.layers.split(",")]
    D = inner.args.hidden_size if hasattr(inner, "args") else model.args.hidden_size
    print(f"loaded in {time.time() - t0:.1f}s layers={nL} hidden={D}", flush=True)

    docs = {}
    for line in open(args.corpus):
        r = json.loads(line)
        docs.setdefault(r["split"], []).append((r["id"], r["text"]))
    budget = dict(kv.split("=") for kv in args.budget.split(","))
    out = {"model": args.model, "layers": caps, "window": args.window, "packed": True}
    for split in [s for s in ("held", "val", "fit") if s in budget]:
        seqs, n = [], 0
        for _, text in sorted(docs[split]):
            ids = tok.encode(text)
            if n + len(ids) > int(budget[split]):
                break
            seqs.append(ids)
            n += len(ids)
        doc_start = np.cumsum([0] + [len(x) for x in seqs[:-1]]).astype(np.int64)
        arr = np.concatenate([np.array(x, np.int32) for x in seqs])
        N = len(arr)
        ti = np.zeros((N, args.topk), np.int32)
        tl = np.full((N, args.topk), -30, np.float16)
        ln = np.full(N, np.nan, np.float32)
        hs = {L: np.lib.format.open_memmap(f"{args.out}.{split}.h{L}.npy", mode="w+",
                                            dtype=np.float16, shape=(N, D)) for L in caps}
        ts, stopped, done = time.time(), False, N
        for s in range(0, N, args.window):
            x = mx.array(arr[s:s + args.window])[None]
            h = inner.embed_tokens(x)
            mask = create_attention_mask(h, None)
            grabbed = {}
            for i, layer in enumerate(inner.layers):
                h = layer(h, mask, None)
                if (i + 1) in caps and (i + 1) != nL:
                    grabbed[i + 1] = h[0].astype(mx.float16)
            hn = inner.norm(h)
            if nL in caps:
                grabbed[nL] = hn[0].astype(mx.float16)
            logits = (inner.embed_tokens.as_linear(hn) if model.args.tie_word_embeddings
                      else model.lm_head(hn))[0].astype(mx.float32)
            if s == 0 and split == "held":   # manual forward must equal model(x)
                ref = model(x)[0].astype(mx.float32)
                diff = mx.max(mx.abs(ref - logits)).item()
                print(f"self-check max |logit diff| = {diff:.4g}", flush=True)
                assert diff < 0.05, "manual forward disagrees with model()"
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            k = mx.argpartition(-lp, args.topk - 1, axis=-1)[:, :args.topk]
            kv = mx.take_along_axis(lp, k, axis=-1)
            L = x.shape[1]
            lnx = (mx.take_along_axis(lp[:L - 1], mx.array(arr[s + 1:s + L])[:, None], axis=-1)[:, 0]
                   if L > 1 else mx.zeros((0,)))
            mx.eval(k, kv, lnx, *grabbed.values())
            ti[s:s + L] = np.array(k)
            tl[s:s + L] = np.array(kv).astype(np.float16)
            ln[s:s + L - 1] = np.array(lnx)
            for Lc, g in grabbed.items():
                hs[Lc][s:s + L] = np.array(g)
            if time.time() - t0 > args.max_seconds:
                stopped, done = True, s + L
                break
        keep = int(np.searchsorted(doc_start, done, side="right"))
        if stopped:
            keep = max(keep - 1, 1)
        end = int(doc_start[keep]) if keep < len(doc_start) else N
        for m in hs.values():
            m.flush()
        dt = time.time() - ts
        print(f"{split}: {keep} docs {end} tokens {dt:.1f}s ({done / max(dt, 1e-9):.0f} tok/s)"
              f"{' STOPPED' if stopped else ''}", flush=True)
        np.savez(f"{args.out}.{split}.npz", ids=arr[:end], top_ids=ti[:end], top_lp=tl[:end],
                 lp_next=ln[:end], doc_start=doc_start[:keep], n_hidden_rows=N)
        out[split] = {"docs": keep, "tokens": end, "seconds": dt, "stopped": stopped}
    out["peak_gib"] = mx.get_peak_memory() / 2 ** 30
    out["total_seconds"] = time.time() - t0
    json.dump(out, open(f"{args.out}.json", "w"), indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
