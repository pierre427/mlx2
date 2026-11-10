#!/usr/bin/env python3
"""Build a compact token-aligned routing trace from the capture (ids/scores/timestamps only, no x).

Output trace.npz:
  ids    int16 [T, 40, 6]  routed experts per token per layer (T = 17,915 rows, x_by_layer order)
  rec_*  per capture record (layer call): layer, rows, t_ns, first token row, forward-step index
  tok_step, tok_req, tok_split (0 train / 1 heldout / -1 none), tok_decode (1 if rows==1 call)
  req_ids / req_cat as JSON in meta
"""
import json

import numpy as np

from common import CAPTURE, LAYERS, OUT, atomic_json, atomic_npz, provenance
from cache_sim import read_capture


def main():
    replies = [json.loads(l) for l in open(f"{CAPTURE}/replies.jsonl")]
    win = [(r["t0_ns"], r["t1_ns"], i) for i, r in enumerate(replies)]
    recs, pos, step, last = [], [0] * LAYERS, 0, None
    ids_l = [[] for _ in range(LAYERS)]
    for layer, rows, t, ids, _s, _x in read_capture(f"{CAPTURE}/capture.bin"):
        if last is not None and layer <= last:
            step += 1
        last = layer
        recs.append((layer, rows, t, pos[layer], step))
        ids_l[layer].append(ids.astype(np.int16))
        pos[layer] += rows
    assert len(set(pos)) == 1, pos
    T = pos[0]
    ids = np.stack([np.concatenate(ids_l[L]) for L in range(LAYERS)], 1)  # [T,40,6]
    rec = np.array(recs, np.int64)
    tok_step = np.zeros(T, np.int32); tok_req = np.full(T, -1, np.int32); tok_dec = np.zeros(T, np.int8)
    tok_t = np.zeros(T, np.int64)
    for layer, rows, t, p, s in recs:
        if layer == 0:
            tok_step[p:p + rows] = s; tok_dec[p:p + rows] = rows == 1; tok_t[p:p + rows] = t
            w = next((w for w in win if w[0] <= t <= w[1]), None)
            if w is not None:
                tok_req[p:p + rows] = w[2]
    tok_split = np.array([-1 if r < 0 else (1 if replies[r]["split"] == "heldout" else 0) for r in tok_req], np.int8)
    atomic_npz(f"{OUT}/trace.npz", ids=ids, rec_layer=rec[:, 0], rec_rows=rec[:, 1], rec_t=rec[:, 2],
               rec_pos=rec[:, 3], rec_step=rec[:, 4], tok_step=tok_step, tok_req=tok_req,
               tok_split=tok_split, tok_decode=tok_dec, tok_t=tok_t)
    meta = provenance(tokens=T, records=len(recs), steps=int(step) + 1,
                      decode_tokens=int(tok_dec.sum()),
                      decode_train=int(((tok_dec == 1) & (tok_split == 0)).sum()),
                      decode_heldout=int(((tok_dec == 1) & (tok_split == 1)).sum()),
                      requests=[{"id": r["id"], "cat": r["cat"], "split": r["split"],
                                 "prompt_tokens": r["usage"]["prompt_tokens"],
                                 "completion_tokens": r["usage"]["completion_tokens"],
                                 "wall_s": (r["t1_ns"] - r["t0_ns"]) / 1e9} for r in replies])
    atomic_json(f"{OUT}/trace.meta.json", meta)
    print({k: v for k, v in meta["params"].items() if k != "requests"})


if __name__ == "__main__":
    main()
