#!/usr/bin/env python3
"""Amoeba hub/deformation pilot, phase 0 (CPU): align a donor's captured residuals with the host's.

Uses a finished probe_amoeba_unit_box.py run (box-<tag>/traces.jsonl + hidden/trace*.npz) as the donor
side, so the donor model is never loaded (only its tokenizer). The host runs on the CPU, teacher-forced
on each donor trace's generated text under the host's own chat prompt.

1. Pairing: donor and host tokens are paired where their running character offsets into the shared
   generated text coincide (shared token boundaries); only generated-text positions are used.
2. Depth: linear CKA between every captured donor layer and every host layer; a monotone warp
   (dynamic time warping on 1 - CKA) picks the donor layer for each host layer.
3. Arms (leave-one-trace-out, on the warp-matched pairs of a few host layers):
     procrustes   PCA(k) on both sides, orthogonal map donor -> host PCA space
     ridge        PCA(k) donor -> full host residual, ridge (the linear hub D_h . E_d; CCA spectrum reported)
     relrep       relative representations: cosine to shared anchor positions (geometric only)
     gw           entropic Gromov-Wasserstein on within-model distances (no shared coordinates)
   controls: shuffled pairing (same maps, mismatched rows), host-PCA(k) ceiling (host -> host).
   Geometric scores: held-out R^2 in host space, retrieval top-1 (nearest held-out host row), GW mass on
   the true pair.
4. Stitching (the score that counts): on the held-out trace, replace the host residual at that layer, on
   paired positions, with the mapped donor vector; teacher-forced KL(host || stitched) and top-1 agreement
   over generated positions, against shuffled-map and host-mean replacement.

Writes OUT/{pairs.json,cka.npy,warp.json,arms.jsonl,stitch.jsonl,run.json}. Refuses a non-empty OUT.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12))


def dtw(cost):
    """Monotone path over a (donor x host) cost matrix; returns donor index per host column."""
    n, m = cost.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, best = n, m, {}
    while i > 0 and j > 0:
        best.setdefault(j - 1, []).append(i - 1)
        k = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return [min(best.get(j, [0]), key=lambda d: cost[d, j]) for j in range(m)]


class PCA:
    def __init__(self, X, k):
        self.mu = X.mean(0)
        _, _, vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.V = vt[:k].T

    def __call__(self, X):
        return (X - self.mu) @ self.V

    def inv(self, Z):
        return Z @ self.V.T + self.mu


def ridge(X, Y, lam):
    Xm, Ym = X.mean(0), Y.mean(0)
    Xc = X - Xm
    # Penalty relative to the mean feature variance, so the map is scale-invariant (2026-11-14: an absolute
    # lam*n*I swamped Flash Next, whose residual variances are ~1e-2..1e-7, giving ridge R2 0.1 vs Procrustes 0.6).
    G = Xc.T @ Xc
    W = np.linalg.solve(G + lam * np.trace(G) / X.shape[1] * np.eye(X.shape[1]), Xc.T @ (Y - Ym))
    return lambda Z: (Z - Xm) @ W + Ym


def procrustes(A, B):
    u, _, vt = np.linalg.svd(A.T @ B)
    return u @ vt


def cca_spectrum(A, B, k, eps=1e-3):
    def white(M):
        M = M - M.mean(0)
        c = M.T @ M / len(M) + eps * np.eye(M.shape[1])
        w, v = np.linalg.eigh(c)
        return M @ v @ np.diag(w ** -0.5) @ v.T
    s = np.linalg.svd(white(A).T @ white(B) / len(A), compute_uv=False)
    return [float(x) for x in s[:k]]


def r2(P, Y):
    return float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def retrieval(P, Y):
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    return float((np.argmax(Pn @ Yn.T, 1) == np.arange(len(P))).mean())


def gw_diag(A, B, eps, iters=40):
    """Entropic GW (square loss) between point clouds A, B (same n, true pairing = identity)."""
    import torch
    a, b = torch.tensor(A, dtype=torch.float64), torch.tensor(B, dtype=torch.float64)
    C1, C2 = torch.cdist(a, a), torch.cdist(b, b)
    C1, C2 = C1 / C1.max(), C2 / C2.max()
    n = len(a)
    p = torch.full((n,), 1.0 / n, dtype=torch.float64)
    T = p[:, None] * p[None, :]
    const = (C1 ** 2 @ p)[:, None] + (C2 ** 2 @ p)[None, :]
    for _ in range(iters):
        G = const - 2 * C1 @ T @ C2.T
        K = torch.exp(-(G - G.min()) / eps)
        u = torch.ones(n, dtype=torch.float64)
        for _ in range(100):
            v = p / (K.T @ u)
            u = p / (K @ v)
        T = u[:, None] * K * v[None, :]
    T = T.numpy() * n
    return {"diag_mass": float(np.trace(T) / n), "argmax_top1": float((T.argmax(1) == np.arange(n)).mean()),
            "chance": 1.0 / n}


def char_ends_from_ids(tok, ids):
    ends, text = [], ""
    for i in range(len(ids)):
        text = tok.decode(ids[: i + 1])
        ends.append(len(text))
    return ends, text


def admission(cap_gb, margin_gb, min_swap_free_gb=4.0):
    """Start only if available memory (free+inactive+speculative+purgeable) covers the cap plus a margin and
    swap is not nearly exhausted. Runs beside the :8282 service must fit beside it (2026-11-14)."""
    import re
    vm = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", vm).group(1))
    get = lambda k: int(m.group(1)) if (m := re.search(rf"{k}:\s+(\d+)", vm)) else 0
    avail = sum(get(k) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")) * page / 1e9
    sw = subprocess.run(["/usr/sbin/sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"free = ([\d.]+)M", sw)
    swap_free = float(m.group(1)) / 1e3 if m else 0.0
    ok = avail >= cap_gb + margin_gb and swap_free >= min_swap_free_gb
    return {"ok": ok, "avail_gb": round(avail, 1), "swap_free_gb": round(swap_free, 1), "cap_gb": cap_gb,
            "margin_gb": margin_gb}


def dequantize_model(mx, model, dtype):
    """Replace quantized linears/embeddings by dense ones: MLX's CPU quantized matmul is single-threaded
    (a 4B prefill of ~900 tokens took >20 min); dense CPU matmul goes through multithreaded BLAS."""
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    def deq(m):
        kw = {"mode": m.mode} if hasattr(m, "mode") else {}
        return mx.dequantize(m.weight, m.scales, getattr(m, "biases", None), m.group_size, m.bits, **kw).astype(dtype)
    new = []
    for name, m in model.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            w = deq(m)
            lin = nn.Linear(w.shape[1], w.shape[0], bias="bias" in m)
            lin.weight = w
            if "bias" in m:
                lin.bias = m.bias.astype(dtype)
            new.append((name, lin))
        elif isinstance(m, nn.QuantizedEmbedding):
            w = deq(m)
            emb = nn.Embedding(w.shape[0], w.shape[1])
            emb.weight = w
            new.append((name, emb))
    if new:
        model.update_modules(tree_unflatten(new))
    mx.eval(model.parameters())
    return len(new)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--donor-run", required=True, help="box-<tag> dir of probe_amoeba_unit_box.py")
    ap.add_argument("--donor-model", required=True, help="donor model dir (tokenizer only)")
    ap.add_argument("--host-model", default=str(Path.home() / "mlx-models" / "Qwen3.5-4B-MLX-4bit"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--host-layers", default="3,8,16,24", help="host layers for arms and stitching")
    ap.add_argument("--k", type=int, default=256, help="PCA width for the maps")
    ap.add_argument("--lam", type=float, default=1e-2, help="ridge penalty as a fraction of the mean feature variance")
    ap.add_argument("--gw-points", type=int, default=600)
    ap.add_argument("--gw-eps", type=float, default=5e-3)
    ap.add_argument("--anchors", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--no-stitch", action="store_true")
    ap.add_argument("--chunk", type=int, default=512, help="host prefill chunk (tokens)")
    ap.add_argument("--mem-cap-gb", type=float, default=24.0, help="hard MLX memory limit for this process")
    ap.add_argument("--admit-margin-gb", type=float, default=12.0,
                    help="refuse to start unless available memory >= mem cap + this margin")
    ap.add_argument("--keep-quantized", action="store_true", help="skip the CPU dequantization (slow)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    adm = admission(args.mem_cap_gb, args.admit_margin_gb)
    print(f"admission {adm}", flush=True)
    if not adm["ok"]:
        sys.exit(3)
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    mx.set_memory_limit(int(args.mem_cap_gb * 1e9))
    mx.set_cache_limit(int(1e9))
    from transformers import AutoTokenizer
    from mlx2.adapters.registry import inspect_model
    from mlx2.experimental.weight_space import residual_ablation as ra
    from mlx2.serving import render_prompt_tokens

    drun = Path(args.donor_run)
    drj = json.loads((drun / "run.json").read_text())
    traces = [json.loads(l) for l in (drun / "traces.jsonl").read_text().splitlines() if l.strip()]
    dtok = AutoTokenizer.from_pretrained(args.donor_model)

    t0 = time.time()
    adapter = inspect_model(args.host_model).adapter_type(str(args.host_model))
    model, htok = adapter.model, adapter.tokenizer
    if not args.keep_quantized:
        nq = dequantize_model(mx, model, mx.float32)
        print(f"dequantized {nq} modules to float32", flush=True)
    layers = list(getattr(model, "language_model", model).model.layers)
    HL = len(layers)
    host_ck = f"{Path(args.host_model).name}@{adapter.identity['fingerprint']}"
    rev = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    cand = drj["candidate"]  # the probe stores the full candidate (id, prompt, answer)
    hprompt = [int(t) for t in render_prompt_tokens(
        adapter, {"messages": [{"role": "user", "content": cand["prompt"]}], "enable_thinking": False})]
    print(f"host {host_ck} layers {HL} donor {drj['checkpoint']} rev {rev} dirty {dirty} load {time.time() - t0:.0f}s",
          flush=True)

    def host_forward(seq, gen0, patch=None, ref=None, k=256):
        """Teacher-forced host forward in --chunk-token pieces through the model cache, so activations and
        logits never exist for the whole sequence at once (2026-11-14: a whole-sequence float32 forward with
        full-vocab logits reached a 69 GB footprint beside the 72 GB :8282 service). Returns
        (summary-or-score, residuals); logits are reduced per chunk to a top-k summary of generated rows."""
        keep = {i: [] for i in range(HL)}
        cur = {"c0": 0, "c1": 0}

        def hook(i):
            def h(o):
                x = o[0] if isinstance(o, tuple) else o
                if patch is not None and i in patch:
                    rows, vecs = patch[i]
                    sel = (rows >= cur["c0"]) & (rows < cur["c1"])
                    if sel.any():
                        x0 = np.array(x[0].astype(mx.float32))
                        x0[rows[sel] - cur["c0"]] = vecs[sel]
                        x = mx.array(x0)[None].astype(x.dtype)
                        o = (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
                if patch is None:
                    xi = x[0].astype(mx.float32)
                    mx.eval(xi)
                    keep[i].append(np.array(xi))
                return o
            return h

        inp, cache, parts = seq[:-1], model.make_cache(), []
        with ra._hooked([(layers[i], hook(i)) for i in range(HL)]):
            for c0 in range(0, len(inp), args.chunk):
                c1 = min(c0 + args.chunk, len(inp))
                cur.update(c0=c0, c1=c1)
                lg = model(mx.array([inp[c0:c1]]), cache=cache)[0]
                r0 = max(gen0 - 1, c0)  # rows of generated text in this chunk
                if r0 < c1:
                    lg = lg[r0 - c0:].astype(mx.float32)
                    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
                    if ref is None:
                        top = mx.argpartition(-lp, kth=k, axis=-1)[:, :k]
                        part = {"top": top, "top_lp": mx.take_along_axis(lp, top, axis=-1), "argmax": mx.argmax(lp, -1)}
                    else:
                        g0, g1 = r0 - (gen0 - 1), c1 - (gen0 - 1)
                        top, rlp = mx.array(ref["top"][g0:g1]), mx.array(ref["top_lp"][g0:g1])
                        lq_top = mx.take_along_axis(lp, top, axis=-1)
                        p_top = mx.exp(rlp)
                        p_tail = mx.maximum(1 - p_top.sum(-1), 1e-12)
                        q_tail = mx.maximum(1 - mx.exp(lq_top).sum(-1), 1e-12)
                        part = {"kl": (p_top * (rlp - lq_top)).sum(-1) + p_tail * (mx.log(p_tail) - mx.log(q_tail)),
                                "agree": mx.argmax(lp, -1) == mx.array(ref["argmax"][g0:g1])}
                    mx.eval(part)
                    parts.append({a: np.array(b) for a, b in part.items()})
                    del lp
                del lg
                mx.clear_cache()
        cat = {a: np.concatenate([p[a] for p in parts]) for a in parts[0]}
        if ref is None:
            res = cat
        else:
            kl, agree = cat["kl"], cat["agree"]
            res = {"kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)), "top1_agree": float(agree.mean())}
        del cache
        mx.clear_cache()
        return res, ({i: np.concatenate(v) for i, v in keep.items()} if patch is None else {})

    # 1. pairs per trace
    per = []
    for ti, tr in enumerate(traces):
        ids = tr["ids"]
        T0 = len(ids) - tr["tokens"]
        gen = ids[T0:]
        dends, gtext = char_ends_from_ids(dtok, gen)
        enc = htok.encode(gtext, add_special_tokens=False) if hasattr(htok, "encode") else None
        hends, _ = char_ends_from_ids(htok, enc)
        hseq = hprompt + enc
        hpos = {e: j for j, e in enumerate(hends)}
        pairs = [(T0 + i, len(hprompt) + hpos[e]) for i, e in enumerate(dends) if e in hpos]
        # hidden row r is the output at input token r (forward on seq[:-1]); drop the final token
        pairs = [(d, h) for d, h in pairs if d < len(ids) - 1 and h < len(hseq) - 1]
        hz = np.load(drun / "hidden" / f"trace{ti}.npz")
        dh = {int(k[1:]): hz[k].astype(np.float32) for k in hz.files}
        tf = time.time()
        href, hh = host_forward(hseq, len(hprompt))
        print(f"{time.strftime('%H:%M:%S')} trace {ti}: donor gen {len(gen)} host gen {len(enc)} pairs {len(pairs)} "
              f"host forward {time.time() - tf:.0f}s mlx peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
        per.append({"pairs": pairs, "dh": dh, "hh": hh, "href": href, "hseq": hseq, "hgen0": len(hprompt)})
    dlayers = sorted(per[0]["dh"])
    (out / "pairs.json").write_text(json.dumps({"donor_layers": dlayers, "host_layers": HL,
                                                "pairs_per_trace": [len(p["pairs"]) for p in per]}))

    def mats(dl, hl, traces_idx):
        X = np.concatenate([per[t]["dh"][dl][[d for d, _ in per[t]["pairs"]]] for t in traces_idx])
        Y = np.concatenate([per[t]["hh"][hl][[h for _, h in per[t]["pairs"]]] for t in traces_idx])
        return X, Y

    # 2. CKA + warp
    allt = list(range(len(per)))
    C = np.zeros((len(dlayers), HL))
    for a, dl in enumerate(dlayers):
        for hl in range(HL):
            X, Y = mats(dl, hl, allt)
            C[a, hl] = cka(X, Y)
    np.save(out / "cka.npy", C)
    warp = dtw(1 - C)
    (out / "warp.json").write_text(json.dumps({"donor_layers": dlayers, "donor_for_host": [dlayers[w] for w in warp],
                                               "cka_on_warp": [float(C[w, j]) for j, w in enumerate(warp)]}))
    print("warp (host -> donor layer):", [dlayers[w] for w in warp], flush=True)

    # 3/4. arms and stitching, leave one trace out
    arms_f, st_f = open(out / "arms.jsonl", "w"), open(out / "stitch.jsonl", "w")
    for hl in [int(x) for x in args.host_layers.split(",")]:
        dl = dlayers[warp[hl]]
        for ho in allt:
            tr_idx = [t for t in allt if t != ho]
            Xtr, Ytr = mats(dl, hl, tr_idx)
            Xte, Yte = mats(dl, hl, [ho])
            k = min(args.k, len(Xtr) - 1, Xtr.shape[1])
            pd, ph = PCA(Xtr, k), PCA(Ytr, k)
            Zd, Zh, Zdte, Zhte = pd(Xtr), ph(Ytr), pd(Xte), ph(Yte)
            perm = rng.permutation(len(Xtr))
            rec = {"host_layer": hl, "donor_layer": dl, "heldout": ho, "n_train": len(Xtr), "n_test": len(Xte), "k": k,
                   "cka": float(C[warp[hl], hl])}
            # A procrustes (scale-normalised PCA spaces)
            sd, sh = np.linalg.norm(Zd), np.linalg.norm(Zh)
            R = procrustes(Zd / sd, Zh / sh)
            P_proc = ph.inv((Zdte / sd) @ R * sh)
            R_sh = procrustes(Zd[perm] / sd, Zh / sh)
            # E-lin ridge
            f = ridge(Zd, Ytr, args.lam)
            f_sh = ridge(Zd[perm], Ytr, args.lam)
            g_ceiling = ridge(Zh, Ytr, args.lam)
            P_r, P_rsh, P_ceil = f(Zdte), f_sh(Zdte), g_ceiling(Zhte)
            rec.update({
                "procrustes": {"r2": r2(P_proc, Yte), "top1": retrieval(P_proc, Yte)},
                "procrustes_shuffled": {"r2": r2(ph.inv((Zdte / sd) @ R_sh * sh), Yte),
                                        "top1": retrieval(ph.inv((Zdte / sd) @ R_sh * sh), Yte)},
                "ridge": {"r2": r2(P_r, Yte), "top1": retrieval(P_r, Yte)},
                "ridge_shuffled": {"r2": r2(P_rsh, Yte), "top1": retrieval(P_rsh, Yte)},
                "host_pca_ceiling": {"r2": r2(P_ceil, Yte), "top1": retrieval(P_ceil, Yte)},
                "cca_top": cca_spectrum(Zd[:, :64], Zh[:, :64], 16),
                "chance_top1": 1.0 / len(Xte),
            })
            # B relative representations
            anc = rng.choice(len(Xtr), size=min(args.anchors, len(Xtr)), replace=False)

            def rel(Z, A):
                Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
                An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
                return Zn @ An.T
            rd, rh = rel(Zdte, Zd[anc]), rel(Zhte, Zh[anc])
            rec["relrep"] = {"top1": retrieval(rd - rd.mean(0), rh - rh.mean(0))}
            # C gromov-wasserstein on a held-out subsample
            sub = rng.choice(len(Xte), size=min(args.gw_points, len(Xte)), replace=False)
            rec["gw"] = gw_diag(Zdte[sub], Zhte[sub], args.gw_eps)
            arms_f.write(json.dumps(rec) + "\n")
            arms_f.flush()
            print(f"{time.strftime('%H:%M:%S')} L{hl}<-D{dl} ho{ho}: ridge r2 {rec['ridge']['r2']:.3f} top1 "
                  f"{rec['ridge']['top1']:.3f} (shuf {rec['ridge_shuffled']['top1']:.3f}, ceil "
                  f"{rec['host_pca_ceiling']['top1']:.3f}) procr top1 {rec['procrustes']['top1']:.3f} relrep "
                  f"{rec['relrep']['top1']:.3f} gw top1 {rec['gw']['argmax_top1']:.3f} (chance {rec['gw']['chance']:.4f})",
                  flush=True)
            if args.no_stitch:
                continue
            # 4. stitching on the held-out trace
            p = per[ho]
            rows = np.array([h for _, h in p["pairs"]])

            def score(vecs):
                return host_forward(p["hseq"], p["hgen0"], patch={hl: (rows, vecs)}, ref=p["href"])[0]
            srec = {"host_layer": hl, "donor_layer": dl, "heldout": ho, "patched_rows": int(len(rows)),
                    "ridge": score(P_r), "procrustes": score(P_proc), "ridge_shuffled": score(P_rsh),
                    "host_mean": score(np.repeat(Ytr.mean(0, keepdims=True), len(rows), 0)),
                    "host_pca_ceiling": score(P_ceil)}
            st_f.write(json.dumps(srec) + "\n")
            st_f.flush()
            print(f"   stitch KL: ridge {srec['ridge']['kl_mean']:.3f} procr {srec['procrustes']['kl_mean']:.3f} "
                  f"shuf {srec['ridge_shuffled']['kl_mean']:.3f} mean {srec['host_mean']['kl_mean']:.3f} "
                  f"ceil {srec['host_pca_ceiling']['kl_mean']:.3f}", flush=True)
    (out / "run.json").write_text(json.dumps({
        "schema": "amoeba-hub-align/v1", "host_checkpoint": host_ck, "donor_checkpoint": drj["checkpoint"],
        "donor_run": str(drun), "donor_run_revision": drj.get("mlx2_source_revision"), "mlx2_source_revision": rev,
        "tree_dirty": dirty, "candidate": cand["id"], "args": vars(args), "wall_s": round(time.time() - t0),
        "admission": adm, "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2)}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
