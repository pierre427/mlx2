#!/usr/bin/env python3
"""Router-level token map for all 384 experts of every DS4.1 layer (embedding level, no expert weights).

For every real token t: x_t = rms_norm(embed[t]) * ffn_norm, scored by the layer router
(sqrt(softplus(logits)) + correction bias, top-6). Per expert: how many tokens would select it, and the
top-N tokens by router score. Writes OUT/router/layer_XX.npz.  usage: atlas_router.py OUT
"""
import os, sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

REAL, CH, TOPN, EPS, E, TOPK = 128000, 16384, 50, 1e-6, 384, 6


def main():
    out = sys.argv[1]
    mx.set_default_device(mx.gpu); mx.set_cache_limit(2 << 30)
    os.makedirs(f"{out}/router", exist_ok=True)
    emb = mx.load(f"{out}/lite/embed.safetensors")["embed.weight"]
    for L in range(40):
        p = f"{out}/router/layer_{L:02d}.npz"
        if os.path.exists(p):
            continue
        t0 = time.time()
        lite = mx.load(f"{out}/lite/layer_{L:02d}.safetensors")
        w = lite["ffn_norm.weight"].astype(mx.float32); gate = lite["gate.weight"]; bias = lite["gate.bias"]
        counts = mx.zeros((E,), mx.int32); best_v = best_i = None
        for c0 in range(0, REAL, CH):
            e = emb[c0:min(REAL, c0 + CH)].astype(mx.float32)
            xn = (e * mx.rsqrt(mx.mean(e * e, -1, keepdims=True) + EPS) * w).astype(mx.bfloat16)
            sc = mx.sqrt(nn.softplus((xn @ gate.T).astype(mx.float32)))
            top = mx.argpartition(-(sc + bias), TOPK - 1, axis=-1)[:, :TOPK]
            counts = counts.at[top.reshape(-1)].add(1)
            scT = sc.T
            ids = mx.broadcast_to(mx.arange(c0, c0 + scT.shape[1])[None], scT.shape)
            if best_v is not None:
                scT = mx.concatenate([best_v, scT], 1); ids = mx.concatenate([best_i, ids], 1)
            sel = mx.argpartition(-scT, TOPN - 1, axis=-1)[:, :TOPN]
            best_v, best_i = mx.take_along_axis(scT, sel, -1), mx.take_along_axis(ids, sel, -1)
            mx.eval(counts, best_v, best_i)
        o = mx.argsort(-best_v, axis=-1)
        np.savez_compressed(p, counts=np.array(counts), top_score=np.array(mx.take_along_axis(best_v, o, -1)),
                            top_id=np.array(mx.take_along_axis(best_i, o, -1)).astype(np.int32))
        print(f"router L{L:02d} {time.time() - t0:.1f}s", flush=True)
    print("ROUTER MAP DONE", flush=True)


if __name__ == "__main__":
    main()
