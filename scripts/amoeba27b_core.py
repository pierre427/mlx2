#!/usr/bin/env python3
"""Amoeba on Qwen3.8-27B, window 1: checks, English core rankings, bisection, sink scan, atlas capture.

MLX on the GPU; run only inside an owned GPU window (the 27B needs the :8282 service paused).
Phases (--phases, comma list, in this order):
  checks   wrapper identity (core = ones gives the base logits exactly), train-mode (GDN ops path)
           vs eval-mode (kernel) logits, and a first-order gradient check on the core loss.
  mean     mean MLP activation per neuron on the mean split.
  rank     one straight-through core per ablation in --ablations (zero, mean): Z0 = 3, lam, steps,
           lr, KL(full || core) over positions >= 1, position 0 exact in every layer (the 0.6B
           recipe). Saves ranking_<ablation>.npz.
  bisect   per ranking and target: bisect the kept fraction to the target mean held-out d_CE;
           control 1 a random core with the same per-layer counts; control 2 a random global
           ranking bisected to the same cost; capitals accuracy at each matched core. The export
           core (zero ablation) is also cut on the fit-eval split ("spare") at --export-target.
  sink     on the export core: English cost with no position-0 protection, the true rule at depths
           k, dense stand-ins (layers 0..k-1 whole) and add-back neurons (largest position-0 |act|
           among dropped neurons of layers < k). Chosen on the spare split, reported on held.
  atlas    stream per-neuron sequence peaks of |act - mean| (positions >= 1) to per-layer fp16
           .npy, domain profiles, top tokens, and position-0 magnitudes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import amoeba27b_lib as A


def load_jsonl_corpus(host, path, seq_len, n_mean, n_fit, n_held, seed, log):
    """As research.amoeba.carve.corpus._jsonl (main 72f0ac6ed): rows {"text", "domain", "source", "split"?}.
    A row's split (held / mean / fit / spare; fresh -> held) is honoured; other rows are ordered by a hash
    of their source and fill held, mean and fit per domain, the rest is spare. Measurement sets refused."""
    import hashlib
    import random as _r
    markers = ("gsm8k-test", "gsm8k/test", "gsm8k_test", "six-problem", "mmlu-pro", "mmlu_pro", "gpqa")
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    bad = [r.get("source", "") for r in rows if any(m in str(r.get("source", "")).lower() for m in markers)]
    if bad:
        raise SystemExit(f"corpus {path}: refusing measurement-set rows: {sorted(set(bad))[:5]}")
    by = {}
    for r in rows:
        by.setdefault(r.get("domain", "text"), []).append(r)
    doms = sorted(by)
    rng = _r.Random(seed)
    split = {}
    for dn in doms:
        out = {k: [] for k in ("held", "mean", "fit", "spare")}
        free = []
        for r in by[dn]:
            ids = host.encode(r["text"])
            if len(ids) < seq_len:
                continue
            ids = ids[:seq_len]
            sp = "held" if r.get("split") == "fresh" else r.get("split")
            if sp in out:
                out[sp].append(ids)
            else:
                h = int(hashlib.sha256(f"{seed}:{r.get('source', r['text'][:64])}".encode()).hexdigest()[:12], 16)
                free.append((h, ids))
        free.sort(key=lambda t: t[0])
        i = 0
        for name, k in (("held", n_held), ("mean", n_mean // len(doms)), ("fit", n_fit // len(doms))):
            take = max(0, k - len(out[name]))
            out[name] += [ids for _, ids in free[i:i + take]]
            i += take
        out["spare"] += [ids for _, ids in free[i:]]
        for k in out:
            rng.shuffle(out[k])
        out["held"] = out["held"][:n_held]          # explicit held rows are capped too (eval cost per bisect step)
        split[dn] = out
        log(f"corpus {dn}: " + ", ".join(f"{k} {len(v)}" for k, v in out.items()))
    return split, doms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=str(A.MODEL_27B))
    ap.add_argument("--phases", default="checks,mean,rank,bisect,sink,atlas")
    ap.add_argument("--resume-from", default="", help="earlier evidence dir: reuse mean.npz / ranking_*.npz")
    ap.add_argument("--ablations", default="zero,mean")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=6, help="sequences per optimizer step (one per micro-batch)")
    ap.add_argument("--train-peak-cap-gb", type=float, default=52)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--targets", default="0.18,0.05")
    ap.add_argument("--export-target", type=float, default=0.18)
    ap.add_argument("--bisect-iters", type=int, default=9)
    ap.add_argument("--atlas-seqs", type=int, default=2048)
    ap.add_argument("--atlas-dir", default="", help="where the per-layer captures go (several GB; keep out of git)")
    ap.add_argument("--memory-gb", type=float, default=56)
    ap.add_argument("--cache-gb", type=float, default=4)
    ap.add_argument("--watchdog-gb", type=float, default=80)
    ap.add_argument("--need-gb", type=float, default=45)
    ap.add_argument("--mem-policy", default="strict")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default="legacy", help="legacy | jsonl:PATH ({text, domain, source, split?} rows; "
                    "graded cores, 2026-11-15)")
    ap.add_argument("--n-mean", type=int, default=240)
    ap.add_argument("--n-fit", type=int, default=900)
    ap.add_argument("--n-held", type=int, default=60)
    ap.add_argument("--fixed", default="", help="NPZ[:key] mask kept fixed (a lower tier): carve only the free units")
    ap.add_argument("--cpu", action="store_true", help="smoke tests only")
    ap.add_argument("--smoke", action="store_true", help="tiny splits for a code smoke test")
    args = ap.parse_args()
    out = A.fresh_dir(args.out)
    log = A.make_log(out)
    mem = A.admit(args.need_gb, args.mem_policy, log)
    A.footprint_watchdog(args.watchdog_gb, log)
    A.setup_mlx(args.memory_gb, args.cache_gb)
    import mlx.core as mx
    if args.cpu:
        mx.set_default_device(mx.cpu)
    mx.random.seed(args.seed)
    phases = args.phases.split(",")
    t_start = time.time()
    host = A.Host(args.model, log=log)
    L, N = host.L, host.N
    rep = {"schema": "mlx2.amoeba27b-core.v1", "checkpoint": host.checkpoint, "fingerprint": host.fingerprint,
           "mlx2_source_revision": A.git_rev(), "args": vars(args), "memcheck": mem,
           "shape": {"layers": L, "intermediate": N, "hidden": host.d, "neurons": L * N},
           "corpus_sha256_12": A.sha12(A.CORPUS)}

    def save():
        rep["wall_s"] = round(time.time() - t_start, 1)
        rep["mlx_peak_gb"] = A.peak_gb()
        rep["watch"] = dict(A.WATCH)
        A.write_json(out / "core27b.json", rep)

    if args.corpus.startswith("jsonl:"):
        split, doms = load_jsonl_corpus(host, Path(args.corpus[len("jsonl:"):]).expanduser(), args.seq_len,
                                        args.n_mean, args.n_fit, args.n_held, args.seed, log)
        rep["corpus"] = {"kind": "jsonl", "path": args.corpus[len("jsonl:"):],
                         "sha256_12": A.sha12(Path(args.corpus[len("jsonl:"):]).expanduser()),
                         "sizes": {d: {k: len(v) for k, v in split[d].items()} for d in doms}}
    elif args.smoke:
        split = A.load_corpus(host, seq_len=args.seq_len, n_mean=12, n_fit=12, n_held=4, seed=args.seed, log=log)
        doms = list(A.DOMAINS)
    else:
        split = A.load_corpus(host, seq_len=args.seq_len, seed=args.seed, log=log)
        doms = list(A.DOMAINS)
    fixed = None
    if args.fixed:
        fp, _, fk = args.fixed.partition(":")
        fixed = mx.array((np.load(fp)[fk or "core"] > 0).astype(np.float32))
        assert fixed.shape == (L, N), fixed.shape
        rep["fixed"] = {"path": args.fixed, "kept_frac": float(fixed.mean())}
        log(f"fixed lower tier {args.fixed}: {float(fixed.mean()):.4f} of neurons; carving the free units only")
    tasks, tsplits = A.build_tasks(host, seed=args.seed)
    rep["tasks"] = {k: {"n": len(v), "fit": len(tsplits[k]["fit"]), "held": len(tsplits[k]["held"])} for k, v in tasks.items()}
    caps, caps_idx = tasks["capitals"], tsplits["capitals"]["held"]
    fit = [q for dn in doms for q in split[dn]["fit"]]
    import random
    random.Random(args.seed).shuffle(fit)
    spare = {dn: {"spare": split[dn]["spare"][:(4 if args.smoke else 60)]} for dn in doms}
    if args.smoke:
        caps_idx = caps_idx[:8]
    clean_held, clean_spare = {}, {}
    resume = Path(args.resume_from) if args.resume_from else None

    def cfg(core, mean=None, sink=0, dense=()):
        def f():
            host.ctl.reset()
            host.ctl.core, host.ctl.mean, host.ctl.sink_layers, host.ctl.dense_layers = core, mean, sink, tuple(dense)
        return f

    def held(core, mean=None, sink=L, dense=()):
        return A.english_cost(host, split, clean_held, cfg(core, mean, sink, dense),
                              key_domains=[d for d in doms if split[d]["held"]])

    def fit_eval(core, mean=None, sink=L, dense=()):
        return A.english_cost(host, spare, clean_spare, cfg(core, mean, sink, dense),
                              key_domains=[d for d in doms if spare[d]["spare"]], part="spare")

    def capitals(core, mean=None, sink=L, dense=()):
        return A.task_eval(host, caps, caps_idx, cfg(core, mean, sink, dense))[0]

    # ------------------------------------------------------------------ checks
    if "checks" in phases:
        ids = mx.array(fit[:2])
        host.ctl.reset()
        base = host.logits_at(ids)
        host.ctl.reset(); host.ctl.core = mx.ones((L, N)); host.ctl.sink_layers = 0
        ones = host.logits_at(ids)
        host.ctl.reset()
        host.train_mode(True)
        trn = host.logits_at(ids)
        ck = host.train_logits_at(ids, mx.ones((L, N)))
        host.train_mode(False)
        mx.eval(base, ones, trn, ck)
        c = {"ones_vs_base_max_abs": float(mx.abs(ones - base).max()),
             "train_vs_eval_max_abs": float(mx.abs(trn - base).max()),
             "train_vs_eval_top1_agree": float((trn.argmax(-1) == base.argmax(-1)).mean()),
             "train_vs_eval_kl": float(A.kl_rows(base, trn).mean()),
             "ckpt_train_vs_train_max_abs": float(mx.abs(ck - trn).max())}
        # first-order check: d loss / d m at m = 1; drop the 512 neurons the gradient says hurt most
        # and 512 random ones; the predicted increase should order them and track the real one.
        host.train_mode(True)
        lc = mx.stop_gradient(base[:, 1:])

        def apply(l, sl):
            host.ctl.reset(); host.ctl.core = {l: sl["core"]}
        _, gd = A.layerwise_value_and_grad(host, ids, {"core": mx.ones((L, N))}, apply,
                                           lambda hid: A.kl_rows(lc, host.head(hid[:, 1:]).astype(mx.float32)).mean())
        g = gd["core"]
        mx.eval(g)
        host.train_mode(False)
        flat = -g.reshape(-1)                       # predicted increase when m goes 1 -> 0
        top = mx.argsort(flat)[-512:]
        rnd = mx.array(np.random.default_rng(1).choice(L * N, 512, replace=False))
        res = {}
        for name, sel in (("top_grad", top), ("random", rnd)):
            m = mx.ones((L * N,))
            m[sel] = 0.0
            m = m.reshape(L, N)
            host.ctl.reset(); host.ctl.core = m
            lm = host.logits_at(ids)
            host.ctl.reset()
            actual = float(A.kl_rows(base[:, 1:], lm[:, 1:]).mean())
            res[name] = {"predicted": float(flat[sel].sum()), "actual_kl": actual}
            mx.clear_cache()
        c["grad_check"] = res
        rep["checks"] = c
        log(f"checks: {json.dumps(c)}")
        save()

    # ------------------------------------------------------------------ mean
    if "mean" in phases:
        if resume is not None and (resume / "mean.npz").exists():
            mean = mx.array(np.load(resume / "mean.npz")["mean"])
            log(f"mean from {resume}")
        else:
            t0 = time.time()
            mean = A.mean_acts(host, [q for dn in doms for q in split[dn]["mean"]])
            A.save_npz(out / "mean.npz", mean=mean)
            log(f"mean activations in {time.time() - t0:.0f}s")
    elif resume is not None and (resume / "mean.npz").exists():
        mean = mx.array(np.load(resume / "mean.npz")["mean"])
    else:
        mean = None                                  # checks-only run: nothing below needs it
        if set(phases) - {"checks"}:
            raise SystemExit("phases after checks need --phases mean or --resume-from with mean.npz")

    def mean_for(abl):
        return mean if abl == "mean" else None

    # ------------------------------------------------------------------ rank
    rankings = {}
    for abl in args.ablations.split(","):
        p = out / f"ranking_{abl}.npz"
        if "rank" in phases and not (resume is not None and (resume / p.name).exists()):
            log(f"training the {abl}-ablation core: lam={args.lam} steps={args.steps} lr={args.lr} bs={args.bs}")
            host.train_mode(True)
            mu = mean_for(abl)

            def core_grad(m, ids):
                host.train_mode(False)
                host.ctl.reset()
                pos = list(range(1, ids.shape[1]))
                lc = host.logits_at(ids, pos)
                mx.eval(lc)
                host.train_mode(True)

                def apply(l, sl):
                    host.ctl.reset(); host.ctl.mean = mu; host.ctl.sink_layers = L
                    host.ctl.core = {l: sl["core"]}
                loss, g = A.layerwise_value_and_grad(
                    host, ids, {"core": m}, apply,
                    lambda hid: A.kl_rows(lc, host.head(hid[:, mx.array(pos)]).astype(mx.float32)).mean())
                return loss, g["core"]

            def batches(step):
                i = (step * args.bs) % len(fit)
                return [mx.array((fit + fit)[i:i + args.bs])]   # one micro-batch of bs sequences
            t0 = time.time()
            Z = A.train_mask(mx.full((L, N), 3.0), mx.zeros((L, N)) if fixed is None else fixed, args.lam, args.steps, args.lr,
                             core_grad, batches, log, f"core-{abl}", peak_cap_gb=args.train_peak_cap_gb)
            host.train_mode(False)
            A.save_npz(p, Z=Z, mean=mean)
            rep.setdefault("train_s", {})[abl] = round(time.time() - t0, 1)
            save()
        elif resume is not None and (resume / p.name).exists():
            p = resume / p.name
        if p.exists():
            rankings[abl] = mx.array(np.load(p)["Z"])

    # ------------------------------------------------------------------ bisect
    g_seed = args.seed + 1234
    rnd_score = mx.array(np.random.default_rng(g_seed).random((L, N)).astype(np.float32))

    def with_fixed(score):
        """Fixed (lower-tier) units rank above every free unit, so every cut keeps the lower tier."""
        return score if fixed is None else mx.where(fixed > 0, mx.array(1e9, dtype=score.dtype), score)

    def random_free(M, seed):
        """Random control: the fixed tier plus random free units with M's per-layer free counts."""
        if fixed is None:
            return A.random_same_counts(M, seed)
        rng = np.random.default_rng(seed)
        Fx, Mn = np.array(fixed), np.array(M)
        R = Fx.copy()
        for l in range(L):
            free_idx = np.where(Fx[l] == 0)[0]
            k = int((Mn[l] * (1 - Fx[l])).sum())
            R[l, rng.choice(free_idx, k, replace=False)] = 1.0
        return mx.array(R)
    rnd_score = with_fixed(rnd_score)
    rankings = {k: with_fixed(v) for k, v in rankings.items()}
    fixed_frac = 0.0 if fixed is None else float(fixed.mean())
    if "bisect" in phases:
        rep["full_capitals_acc"] = capitals(None)
        rep["bisect"] = {}

        def bisect(score, target, mu, evalf, tag):
            lo, hi, evals = max(0.02, fixed_frac + 1e-4), 1.0, []
            for _ in range(args.bisect_iters):
                mid = (lo + hi) / 2
                d = evalf(A.topk_mask(score, mid), mu)["mean_d_ce"]
                evals.append((round(mid, 5), round(d, 4)))
                if d > target:
                    lo = mid
                else:
                    hi = mid
            log(f"  [{tag} +{target}] kept in ({lo:.4f}, {hi:.4f}]; evals {evals}")
            return hi, evals

        for abl, score in rankings.items():
            mu = mean_for(abl)
            trained = (score > 0).astype(mx.float32)          # with_fixed: fixed units score 1e9 > 0
            r = {"trained_core": {"kept_frac": float(trained.mean()), "held": held(trained, mu),
                                  "capitals_acc": capitals(trained, mu)}}
            log(f"[{abl}] trained core Z>0 kept {r['trained_core']['kept_frac']:.4f} "
                f"dCE {r['trained_core']['held']['mean_d_ce']:.3f} capitals {r['trained_core']['capitals_acc']:.2f}")
            for t in [float(x) for x in args.targets.split(",")]:
                frac, ev = bisect(score, t, mu, held, f"{abl} learned")
                M = A.topk_mask(score, frac)
                R = random_free(M, g_seed + int(t * 1000))
                rfrac, rev = bisect(rnd_score, t, mu, held, f"{abl} random ranking")
                row = {"learned_kept_frac": frac, "learned_held": held(M, mu), "learned_capitals_acc": capitals(M, mu),
                       "learned_per_layer_kept": [float(x) for x in M.mean(1).tolist()], "bisect_evals": ev,
                       "random_same_counts_held": held(R, mu), "random_same_counts_capitals_acc": capitals(R, mu),
                       "random_ranking_kept_frac_at_target": rfrac, "random_bisect_evals": rev}
                r[str(t)] = row
                log(f"[{abl} matched +{t}] learned kept {frac:.4f} (dCE {row['learned_held']['mean_d_ce']:.3f}, capitals "
                    f"{row['learned_capitals_acc']:.2f}) | random same counts dCE {row['random_same_counts_held']['mean_d_ce']:.3f}"
                    f" capitals {row['random_same_counts_capitals_acc']:.2f} | random ranking needs {rfrac:.4f}")
                rep["bisect"][abl] = r
                save()
        if "zero" in rankings:
            frac, ev = bisect(rankings["zero"], args.export_target, None, fit_eval, "export core on spare (fit-eval)")
            M = A.topk_mask(rankings["zero"], frac)
            A.save_npz(out / "export_core.npz", core=M, Z=rankings["zero"], frac=np.array(frac))
            rep["export_core"] = {"kept_frac": frac, "chosen_on": "spare (fit-eval) split", "evals": ev,
                                  "held": held(M), "spare": fit_eval(M), "capitals_acc": capitals(M)}
            log(f"export core: kept {frac:.4f} held dCE {rep['export_core']['held']['mean_d_ce']:.3f} "
                f"capitals {rep['export_core']['capitals_acc']:.2f}")
            save()

    # ------------------------------------------------------------------ sink
    if "sink" in phases:
        src = out / "export_core.npz" if (out / "export_core.npz").exists() else resume / "export_core.npz"
        M = mx.array(np.load(src)["core"])
        # position-0 |act| per neuron, for add-back choice (on fit sequences)
        p0 = [mx.zeros((N,)) for _ in range(L)]

        def st(l, act, x):
            p0[l] = p0[l] + mx.abs(act[:, 0].astype(mx.float32)).sum(0)
        n0 = min(48, len(fit))
        for i in range(0, n0, 8):
            host.ctl.reset(); host.ctl.stats = st
            mx.eval(host.hidden(mx.array(fit[i:min(i + 8, n0)])), *p0)
        host.ctl.reset()
        P0 = mx.stack(p0) / float(n0)
        A.save_npz(out / "pos0_abs_act.npz", p0=P0)
        dropped_p0 = P0 * (1 - M)
        arms = {"none": dict(sink=0), "full_rule_all_layers": dict(sink=L)}
        for k in (1, 2, 3, 4, 6, 8, 16):
            arms[f"rule_k{k}"] = dict(sink=k)
        for k in (1, 2, 3, 4, 6, 8):
            arms[f"dense_k{k}"] = dict(sink=0, dense=tuple(range(k)))
        addback = {}
        for k in (2, 3, 4):
            for n in (1, 3, 8):
                Mb = np.array(M)
                for l in range(k):
                    top = np.argsort(-np.array(dropped_p0[l]))[:n]
                    Mb[l, top] = 1.0
                addback[f"addback_k{k}_n{n}"] = mx.array(Mb)
        res = {}
        for name, a in arms.items():
            res[name] = {"spare": fit_eval(M, None, a["sink"], a.get("dense", ()))["mean_d_ce"],
                         "held": held(M, None, a["sink"], a.get("dense", ()))}
            log(f"sink {name}: spare {res[name]['spare']:.3f} held {res[name]['held']['mean_d_ce']:.3f}")
        for name, Mb in addback.items():
            res[name] = {"spare": fit_eval(Mb, None, 0)["mean_d_ce"], "held": held(Mb, None, 0),
                         "kept_frac": float(Mb.mean())}
            log(f"sink {name}: spare {res[name]['spare']:.3f} held {res[name]['held']['mean_d_ce']:.3f}")
        stand_ins = {k: v for k, v in res.items() if k.startswith("dense_") or k.startswith("addback_")}
        best = min(stand_ins, key=lambda k: stand_ins[k]["spare"])
        rep["sink"] = {"arms": res, "chosen_stand_in": best, "chosen_on": "spare (fit-eval) split",
                       "note": "core = export core (zero ablation); rule_kK = position 0 exact in layers < K"}
        log(f"sink: chosen dense stand-in {best} (spare {res[best]['spare']:.3f}, held {res[best]['held']['mean_d_ce']:.3f})")
        save()

    # ------------------------------------------------------------------ atlas
    if "atlas" in phases:
        adir = Path(args.atlas_dir) if args.atlas_dir else out / "atlas"
        adir.mkdir(parents=True, exist_ok=False)
        seqs, src = [], []
        per = args.atlas_seqs // 4
        for dn in A.DOMAINS:
            pool = split[dn]["fit"] + split[dn]["spare"]
            seqs += pool[:per]
            src += [dn] * len(pool[:per])
        # task prompts as plain text (no labels used to form clusters)
        tl = []
        for name in ("capitals", "arithmetic", "french", "recall"):
            for it in tasks[name]:
                tl.append((it["ids"], "task_" + name))
        random.Random(args.seed).shuffle(tl)
        tl = tl[:args.atlas_seqs - len(seqs)]
        S = len(seqs) + len(tl)
        mm = [np.lib.format.open_memmap(adir / f"peak_L{l:02d}.npy", mode="w+", dtype=np.float16, shape=(S, N))
              for l in range(L)]
        names = sorted(set(src + [s for _, s in tl]))
        dsum = np.zeros((len(names), L, N), dtype=np.float64)
        dcnt = np.zeros(len(names))
        topk = 8
        topv = [mx.full((topk, N), -1.0) for _ in range(L)]
        topt = [mx.zeros((topk, N), dtype=mx.int32) for _ in range(L)]
        state = {}

        def st(l, act, x):
            dev = mx.abs(act[:, 1:].astype(mx.float32) - mean[l])              # B, T-1, N
            state["peak"][l] = dev.max(1)
            state["dsum"][l] = dev.sum(1)                                     # B, N
            flat = dev.reshape(-1, N)
            k = min(topk, flat.shape[0])
            idx = mx.argpartition(-flat, k - 1, axis=0)[:k]                    # k, N
            v = mx.take_along_axis(flat, idx, axis=0)
            t = state["tok"][idx]
            cv, ct = mx.concatenate([topv[l], v]), mx.concatenate([topt[l], t])
            o = mx.argsort(-cv, axis=0)[:topk]
            topv[l], topt[l] = mx.take_along_axis(cv, o, axis=0), mx.take_along_axis(ct, o, axis=0)

        order = [(q, s) for q, s in zip(seqs, src)] + tl
        by_len = {}
        for j, (q, s) in enumerate(order):
            by_len.setdefault(len(q), []).append(j)
        t0, done = time.time(), 0
        for ln, js in by_len.items():
            for b in range(0, len(js), 8):
                grp = js[b:b + 8]
                ids = mx.array([order[j][0] for j in grp])
                state.update(peak=[None] * L, dsum=[None] * L, tok=ids[:, 1:].reshape(-1).astype(mx.int32))
                host.ctl.reset(); host.ctl.stats = st
                h = host.hidden(ids)
                mx.eval(h, *state["peak"], *state["dsum"], *topv, *topt)
                host.ctl.reset()
                di = [names.index(order[j][1]) for j in grp]
                for l in range(L):
                    mm[l][grp] = np.array(state["peak"][l]).astype(np.float16)
                for l in range(L):
                    rows = np.array(state["dsum"][l])
                    for r in range(len(grp)):
                        dsum[di[r], l] += rows[r]
                for r in range(len(grp)):
                    dcnt[di[r]] += ln - 1
                done += len(grp)
                mx.clear_cache()
                if done % 256 < 8:
                    log(f"  atlas capture {done}/{S} ({time.time() - t0:.0f}s, peak {A.peak_gb()} GB)")
        for m in mm:
            m.flush()
        prof = dsum / np.maximum(dcnt, 1)[:, None, None]
        np.savez(adir / "atlas_stats.npz", domain_names=np.array(names), domain_profile=prof.astype(np.float32),
                 top_vals=np.stack([np.array(v) for v in topv]), top_toks=np.stack([np.array(t) for t in topt]),
                 seq_source=np.array([s for _, s in order]), mean=np.array(mean))
        A.write_json(adir / "meta.json", {"checkpoint": host.checkpoint, "sequences": S, "layers": L, "neurons": N,
                                          "seq_sources": {n: int(sum(1 for _, s in order if s == n)) for n in names},
                                          "unit": "per-sequence peak |act - mean| over positions >= 1, fp16",
                                          "mlx2_source_revision": A.git_rev()})
        rep["atlas"] = {"dir": str(adir), "sequences": S, "capture_s": round(time.time() - t0, 1)}
        log(f"atlas capture done: {S} sequences in {time.time() - t0:.0f}s")
        save()
    save()
    log(f"done in {time.time() - t_start:.0f}s, MLX peak {A.peak_gb()} GB")


if __name__ == "__main__":
    main()
