#!/usr/bin/env python3
"""Embedding-level token atlas for DS4.1 Source routed experts (one unit = one layer, some experts).

Method (as the Kimi-K3.1 expert-slice probe, docs memory expert-token-signatures-method):
  x_t = rms_norm(embed[t]) * ffn_norm                    for every real token t (< 128000)
  a   = limited_swiglu(x @ w1^T, x @ w3^T)                per expert, 2304 neurons (official MXFP4 codes)
Per neuron: top/bottom K tokens, mean/std/kurtosis over the vocabulary. One row-permuted copy of the
unit's first expert is the structure control. Needs only mlx + numpy (runs on the M3/M2 too).

usage: atlas_worker.py LAYER_LITE.safetensors EMBED.safetensors UNIT.safetensors OUT.npz
UNIT holds e{E}.w1.q (uint32), e{E}.w1.s (uint8 E8M0), e{E}.w3.q, e{E}.w3.s.
"""
import os, sys, time
import numpy as np
import mlx.core as mx

K = 20
EPS = 1e-6
REAL = 128000
CH = 16384
SEED = 20261108
LIMIT = 10.0


def act(g, u):
    g, u = g.astype(mx.float32), u.astype(mx.float32)
    g = mx.minimum(g, LIMIT)
    return g * mx.sigmoid(g) * mx.clip(u, -LIMIT, LIMIT)


def normed(lite, emb):
    w = lite["ffn_norm.weight"].astype(mx.float32)
    out = []
    for c0 in range(0, REAL, CH):
        e = emb["embed.weight"][c0:min(REAL, c0 + CH)].astype(mx.float32)
        xn = (e * mx.rsqrt(mx.mean(e * e, -1, keepdims=True) + EPS) * w).astype(mx.bfloat16)
        mx.eval(xn); out.append(xn)
    return mx.concatenate(out)


def qmm(x, wq):
    return mx.quantized_matmul(x, wq[0], wq[1], None, transpose=True, group_size=32, bits=4, mode="mxfp4")


def topk2(at, k, largest):
    n, v = at.shape
    c = 8192
    pad = (-v) % c
    fill = -mx.inf if largest else mx.inf
    if pad:
        at = mx.concatenate([at, mx.full((n, pad), fill, at.dtype)], axis=1)
    key = -at if largest else at
    kc = key.reshape(n, -1, c)
    i1 = mx.argpartition(kc, k - 1, axis=-1)[..., :k]
    base = (mx.arange(kc.shape[1]) * c)[None, :, None]
    v1 = mx.take_along_axis(kc, i1, -1).reshape(n, -1)
    g1 = (i1 + base).reshape(n, -1)
    i2 = mx.argpartition(v1, k - 1, axis=-1)[:, :k]
    vals = -mx.take_along_axis(v1, i2, -1) if largest else mx.take_along_axis(v1, i2, -1)
    ids = mx.take_along_axis(g1, i2, -1)
    o = mx.argsort(-vals if largest else vals, axis=-1)
    return mx.take_along_axis(vals, o, -1), mx.take_along_axis(ids, o, -1)


def analyse(x, w1, w3, rows=16384):
    n = x.shape[0]
    s1 = s2 = s3 = s4 = 0.0
    cand = {True: ([], []), False: ([], [])}
    for c0 in range(0, n, rows):
        a = act(qmm(x[c0:c0 + rows], w1), qmm(x[c0:c0 + rows], w3))
        a2 = a * a
        s1 = s1 + mx.sum(a, 0); s2 = s2 + mx.sum(a2, 0); s3 = s3 + mx.sum(a2 * a, 0); s4 = s4 + mx.sum(a2 * a2, 0)
        at = mx.contiguous(a.T.astype(mx.bfloat16))
        for largest in (True, False):
            v, i = topk2(at, K, largest)
            cand[largest][0].append(v); cand[largest][1].append(i + c0)
        mx.eval(s1, s2, s3, s4, *[x_[-1] for lst in cand.values() for x_ in lst])
    m1, m2, m3, m4 = s1 / n, s2 / n, s3 / n, s4 / n
    var = m2 - m1 * m1
    kurt = (m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 ** 4) / mx.maximum(var * var, 1e-30)
    out = {}
    for largest, tag in ((True, "pos"), (False, "neg")):
        v = mx.concatenate(cand[largest][0], axis=1); i = mx.concatenate(cand[largest][1], axis=1)
        o = mx.argsort(-v if largest else v, axis=-1)[:, :K]
        out[f"{tag}_val"] = np.array(mx.take_along_axis(v, o, -1).astype(mx.float16))
        out[f"{tag}_id"] = np.array(mx.take_along_axis(i, o, -1)).astype(np.int32)
    mx.eval(kurt, m1, var)
    out.update(kurt=np.array(kurt), mean=np.array(m1), std=np.sqrt(np.maximum(np.array(var), 0)))
    return out


def main():
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(2 << 30)
    lite_p, emb_p, unit_p, out_p = sys.argv[1:5]
    t0 = time.time()
    lite, emb = mx.load(lite_p), mx.load(emb_p)
    x = normed(lite, emb)
    del emb
    unit = mx.load(unit_p)
    experts = sorted({int(k.split(".")[0][1:]) for k in unit})
    res = {}
    for j, e in enumerate(experts):
        w1 = (unit[f"e{e}.w1.q"], unit[f"e{e}.w1.s"]); w3 = (unit[f"e{e}.w3.q"], unit[f"e{e}.w3.s"])
        for k, v in analyse(x, w1, w3).items():
            res[f"e{e}_{k}"] = v
        if j == 0:   # structure control: entries permuted within each row (same per-row values)
            rng = np.random.default_rng(SEED + e)
            def perm(wq):
                w = np.array(mx.dequantize(wq[0], wq[1], None, group_size=32, bits=4, mode="mxfp4").astype(mx.float32))
                w = np.take_along_axis(w, np.argsort(rng.random(w.shape), axis=1), axis=1)
                return tuple(mx.quantize(mx.array(w).astype(mx.bfloat16), group_size=32, bits=4, mode="mxfp4"))
            for k, v in analyse(x, perm(w1), perm(w3)).items():
                res[f"ctrl{e}_{k}"] = v
    np.savez_compressed(out_p + ".tmp.npz", experts=np.array(experts), **res)
    os.replace(out_p + ".tmp.npz", out_p)
    print(f"unit done: {len(experts)} experts in {time.time() - t0:.0f}s ({(time.time() - t0) / max(1, len(experts)):.2f}s/expert)", flush=True)


if __name__ == "__main__":
    main()
