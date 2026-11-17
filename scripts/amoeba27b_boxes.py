#!/usr/bin/env python3
"""Amoeba on Qwen3.8-27B, window 2: cluster functions, boxes, gates, router, pruning, box-spec proxy.

MLX on the GPU, inside an owned GPU window. Inputs: the window-1 evidence (export_core.npz,
core27b.json with the chosen sink stand-in) and the CPU atlas (pool_clusters.npz/json).

Every configuration uses the serving form of the core: zero ablation, no position-dependent rule,
the dense sink stand-in chosen in window 1 (layers kept whole, or add-back neurons).

Phases (--phases, in order):
  base      full model and core on every task's held split, and English d_CE (held).
  clusters  each pool cluster: add to the core (sufficiency) and remove from the full model
            (specificity), each beside a random twin with the same per-layer counts from the pool.
            Task accuracy on held splits, English d_CE. Writes the functional catalogue.
  boxes     straight-through box per skill (capitals, arithmetic, french) plus refined boxes for the
            --refine-clusters most useful clusters (Z initialised from the cluster). Lambda ladder,
            size chosen on the fit split (smallest within 0.05 of the best fit accuracy), reported on
            held / fresh / new phrasing, with a random box of the same per-layer counts.
  gates     per-token gates for all boxes trained jointly (task fit splits + recall + English KL +
            openness penalty); core + all boxes ungated vs gated (hard gates at 0.5).
  router    linear classifier on the core's state (mean over prompt tokens) at candidate layers;
            layer chosen on a validation part of the fit split; reported on held, the new
            phrasing, and hard negatives; routed end to end vs oracle.
  prune     per box: keep the top fraction by straight-through logit Z, against a random cut of
            the same size, at --prune-ratios.
  spec      box-speculation proxy: the core drafts, core + gated boxes verify (greedy); mean
            tokens per verify step at k = 8 from teacher-forced agreement on the verifier's own
            greedy continuation; control verifier = full model.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

import amoeba27b_lib as A

SKILLS = ["capitals", "arithmetic", "french"]
REPORT = ["capitals", "capitals_fresh", "capitals_phrasing", "arithmetic", "french", "recall"]
# Added 2026-11-14 for cluster and bucket labelling (opt-in so earlier runs stay comparable): --probes wide
WIDE = REPORT + ["multiplication", "logic_entail"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--core-run", required=True, help="window-1 evidence dir")
    ap.add_argument("--clusters", required=True, help="amoeba27b_atlas.py output dir")
    ap.add_argument("--model", default=str(A.MODEL_27B))
    ap.add_argument("--phases", default="base,clusters,boxes,gates,router,prune,spec")
    ap.add_argument("--resume-from", default="", help="earlier window-2 dir: reuse boxes.npz / gates.npz")
    ap.add_argument("--cluster-top", type=int, default=64, help="clusters to test (all by default)")
    ap.add_argument("--stand-in", default="", help="override the window-1 sink stand-in (e.g. none)")
    ap.add_argument("--refine-clusters", type=int, default=2)
    ap.add_argument("--probes", default="report", choices=["report", "wide"])
    ap.add_argument("--cluster-items", type=int, default=32, help="held items per task in the cluster tests")
    ap.add_argument("--box-lambdas", default="1,4,16")
    ap.add_argument("--box-steps", type=int, default=40, help="optimizer steps per lambda")
    ap.add_argument("--micro", type=int, default=12, help="prompts per micro-batch (gradient accumulation)")
    ap.add_argument("--box-items-per-step", type=int, default=12)
    ap.add_argument("--train-peak-cap-gb", type=float, default=52)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--gate-steps", type=int, default=60)
    ap.add_argument("--gate-lr", type=float, default=0.01)
    ap.add_argument("--gate-penalty", type=float, default=0.05)
    ap.add_argument("--router-layers", default="8,16,24,32,40")
    ap.add_argument("--prune-ratios", default="0.5,0.25,0.1")
    ap.add_argument("--n-english", type=int, default=20, help="held English seqs per domain")
    ap.add_argument("--memory-gb", type=float, default=56)
    ap.add_argument("--cache-gb", type=float, default=4)
    ap.add_argument("--watchdog-gb", type=float, default=80)
    ap.add_argument("--need-gb", type=float, default=45)
    ap.add_argument("--mem-policy", default="strict")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = A.fresh_dir(args.out)
    log = A.make_log(out)
    mem = A.admit(args.need_gb, args.mem_policy, log)
    A.footprint_watchdog(args.watchdog_gb, log)
    A.setup_mlx(args.memory_gb, args.cache_gb)
    import mlx.core as mx
    import mlx.optimizers as optim
    if args.cpu:
        mx.set_default_device(mx.cpu)
    mx.random.seed(args.seed)
    phases = args.phases.split(",")
    t_start = time.time()
    host = A.Host(args.model, log=log)
    L, N, d = host.L, host.N, host.d
    crun = Path(args.core_run)
    core_rep = json.loads((crun / "core27b.json").read_text())
    core0 = mx.array(np.load(crun / "export_core.npz")["core"])
    stand_in = args.stand_in or core_rep["sink"]["chosen_stand_in"]
    dense = ()
    core = core0
    if stand_in.startswith("dense_k"):
        dense = tuple(range(int(stand_in[len("dense_k"):])))
    elif stand_in.startswith("addback_"):
        k, n = stand_in.split("_")[1:]
        k, n = int(k[1:]), int(n[1:])
        P0 = np.load(crun / "pos0_abs_act.npz")["p0"]
        Mb = np.array(core0)
        for l in range(k):
            Mb[l, np.argsort(-(P0[l] * (1 - Mb[l])))[:n]] = 1.0
        core = mx.array(Mb)
    free = 1.0 - core
    if dense:
        fl = np.array(free)
        fl[list(dense)] = 0.0                     # neurons of whole layers are never box neurons
        free = mx.array(fl)
    rep = {"schema": "mlx2.amoeba27b-boxes.v1", "checkpoint": host.checkpoint, "fingerprint": host.fingerprint,
           "mlx2_source_revision": A.git_rev(), "args": vars(args), "memcheck": mem,
           "core": {"run": str(crun), "kept_frac": float(core.mean()), "stand_in": stand_in, "dense_layers": list(dense)}}

    def save():
        rep["wall_s"] = round(time.time() - t_start, 1)
        rep["mlx_peak_gb"] = A.peak_gb()
        rep["watch"] = dict(A.WATCH)
        A.write_json(out / "boxes27b.json", rep)

    split = A.load_corpus(host, seed=args.seed, log=log, **({"n_mean": 12, "n_fit": 12, "n_held": 4} if args.smoke else {}))
    eng = {dn: {"held": split[dn]["held"][:args.n_english]} for dn in A.DOMAINS}
    eng_fit = [q for dn in A.DOMAINS for q in split[dn]["fit"][:16]]
    tasks, ts = A.build_tasks(host, seed=args.seed)
    if args.smoke:
        ts = {k: {"fit": v["fit"][:6], "held": v["held"][:6]} for k, v in ts.items()}
    clean_eng = {}

    def cfg(core_m=None, boxes=(), dense_l=dense):
        def f():
            host.ctl.reset()
            host.ctl.core, host.ctl.boxes, host.ctl.dense_layers = core_m, list(boxes), tuple(dense_l)
        return f

    eng_small = {dn: {"held": split[dn]["held"][:8]} for dn in A.DOMAINS}
    clean_small = {}

    def evaluate(tag, core_m=None, boxes=(), names=None, english=True, part="held", limit=None, dense_l=None):
        r = {}
        dl = dense if dense_l is None else dense_l
        names = names if names is not None else (WIDE if args.probes == "wide" else REPORT)
        for n in names:
            idx = ts[n][part] if limit is None else ts[n][part][:limit]
            r[n] = A.task_eval(host, tasks[n], idx, cfg(core_m, boxes, dl))[0]
        if english and core_m is not None:
            if limit is None:
                r["english_d_ce"] = A.english_cost(host, eng, clean_eng, cfg(core_m, boxes, dl))["mean_d_ce"]
            else:
                r["english_d_ce"] = A.english_cost(host, eng_small, clean_small, cfg(core_m, boxes, dl))["mean_d_ce"]
        elif english:
            r["english_d_ce"] = 0.0
        log(f"[eval] {tag:28s} " + " ".join(f"{k}={v:.3f}" for k, v in r.items()))
        return r

    resume = Path(args.resume_from) if args.resume_from else None
    rep["eval"] = {}
    if "base" in phases:
        rep["eval"]["full"] = evaluate("full")
        rep["eval"]["core"] = evaluate("core", core)
        save()

    # ------------------------------------------------------------------ clusters
    cblob = np.load(Path(args.clusters) / "pool_clusters.npz")
    cinfo = json.loads((Path(args.clusters) / "pool_clusters.json").read_text())
    pool, labels = cblob["pool"], cblob["labels"]
    # the pool is the export core's complement; drop neurons the stand-in put back into the core
    in_core = np.array(core)[pool[:, 0], pool[:, 1]] > 0
    rng = np.random.default_rng(args.seed + 99)

    def cluster_mask(j):
        m = np.zeros((L, N), np.float32)
        sel = pool[(labels == j) & ~in_core]
        m[sel[:, 0], sel[:, 1]] = 1.0
        return m

    def twin(m):
        """Random pool neurons with the same per-layer counts."""
        R = np.zeros_like(m)
        fr = np.array(free)
        for l in range(L):
            k = int(m[l].sum())
            if k:
                cand = np.flatnonzero(fr[l] > 0)
                R[l, rng.choice(cand, k, replace=False)] = 1.0
        return R

    if "clusters" in phases:
        cat = []
        lim = args.cluster_items
        full = evaluate("full (cluster sets)", limit=lim)
        cor = evaluate("core (cluster sets)", core, limit=lim)
        rep["eval"]["full_cluster_sets"], rep["eval"]["core_cluster_sets"] = full, cor
        kk = min(args.cluster_top, cinfo["k"])
        order = sorted(range(cinfo["k"]), key=lambda j: -cinfo["clusters"][j]["selectivity"])[:kk]
        for j in order:
            m = cluster_mask(j)
            if m.sum() == 0:
                continue
            R = twin(m)
            row = {"id": cinfo["clusters"][j]["id"], "name": cinfo["clusters"][j].get("name"), "neurons": int(m.sum()),
                   "descriptive": {k: cinfo["clusters"][j][k] for k in ("most_selective_source", "selectivity",
                                                                         "top_tokens", "layers")}}
            for kind, mm in (("cluster", m), ("random_twin", R)):
                add = evaluate(f"core+{row['id']} {kind}", mx.minimum(core + mx.array(mm), 1.0), limit=lim)
                rem = evaluate(f"full-{row['id']} {kind}", 1.0 - mx.array(mm), dense_l=(), limit=lim)
                row[kind] = {"add_to_core": {k: round(add[k] - cor[k], 4) for k in add},
                             "remove_from_full": {k: round(rem[k] - full[k], 4) for k in rem}}
            # function label: the task with the largest add gain over the twin, if clearly above it
            probe_names = WIDE if args.probes == "wide" else REPORT
            gains = {t: row["cluster"]["add_to_core"][t] - row["random_twin"]["add_to_core"][t] for t in probe_names}
            losses = {t: row["random_twin"]["remove_from_full"][t] - row["cluster"]["remove_from_full"][t] for t in probe_names}
            best = max(gains, key=gains.get)
            row["function"] = {"best_add_task": best, "add_gain_over_twin": round(gains[best], 4),
                               "remove_loss_over_twin": {t: round(v, 4) for t, v in losses.items()},
                               "english_add_vs_twin": round(row["cluster"]["add_to_core"]["english_d_ce"]
                                                             - row["random_twin"]["add_to_core"]["english_d_ce"], 4),
                               "label": (best if gains[best] >= 0.1 else
                                         "english" if (row["random_twin"]["add_to_core"]["english_d_ce"]
                                                       - row["cluster"]["add_to_core"]["english_d_ce"]) > 0.02 else
                                         "unclassified")}
            cat.append(row)
            rep["clusters"] = cat
            save()
        rep["clusters"] = cat

    # ------------------------------------------------------------------ boxes
    def train_box(items, fit_idx, Z0, tag):
        """Straight-through box over the free neurons, lambda ladder warm-started; returns rows."""
        groups = A.groups_of(items, fit_idx, cap=args.micro)    # micro-batches: bounded GDN tape memory
        clean = {}
        A.cache_clean(host, items, fit_idx, clean, tag, cap=args.micro)
        Z = Z0
        rows = []
        for lam in [float(x) for x in args.box_lambdas.split(",")]:
            def grad_fn(m, gi):
                grp = groups[gi]
                ids, pos = A.batch_of(items, grp)
                lc = clean[(tag, tuple(grp))]

                def apply(l, sl):
                    host.ctl.reset(); host.ctl.core, host.ctl.dense_layers = core, dense
                    host.ctl.boxes = [{"mask": {l: sl["box"]}}]
                loss, g = A.layerwise_value_and_grad(
                    host, ids, {"box": m * free}, apply,
                    lambda hid: A.kl_rows(lc, host.head(mx.take_along_axis(hid, pos[..., None], axis=1)).astype(mx.float32)).mean())
                return loss, g["box"] * free
            host.train_mode(True)
            per = max(1, args.box_items_per_step // args.micro)
            Z = A.train_mask(Z, 1.0 - free, lam, args.box_steps, args.lr, grad_fn,
                             lambda s: [(s * per + j) % len(groups) for j in range(per)], log, f"box {tag}",
                             peak_cap_gb=args.train_peak_cap_gb)
            host.train_mode(False)
            B = (Z > 0).astype(mx.float32) * free
            fit_acc = A.task_eval(host, items, fit_idx, cfg(core, [{"mask": B}]))[0]
            rows.append({"lam": lam, "neurons": int(B.sum()), "fit_acc": fit_acc, "B": B, "Z": Z})
            log(f"[box {tag}] lam={lam} neurons={int(B.sum())} fit={fit_acc:.3f}")
        return rows

    boxes = {}
    if "boxes" in phases and not (resume and (resume / "boxes.npz").exists()):
        rep["boxes"] = {}
        targets = [(s, s, None) for s in SKILLS]
        if rep.get("clusters"):
            useful = sorted([c for c in rep["clusters"] if c["function"]["label"] not in ("unclassified", "english")],
                            key=lambda c: -c["function"]["add_gain_over_twin"])[:args.refine_clusters]
            for c in useful:
                j = int(c["id"].split("C")[-1])
                targets.append((f"refined_{c['id'].replace(':', '_')}", c["function"]["best_add_task"], j))
        for name, task, j in targets:
            items = tasks[task]
            Z0 = mx.full((L, N), -0.5)
            if j is not None:
                Z0 = mx.where(mx.array(cluster_mask(j)) > 0, 0.5, -3.0)
            rows = train_box(items, ts[task]["fit"], Z0, name)
            best = max(r["fit_acc"] for r in rows)
            pick = min([r for r in rows if r["fit_acc"] >= best - 0.05], key=lambda r: r["neurons"])
            boxes[name] = {"mask": pick["B"], "Z": pick["Z"], "task": task}
            R = mx.array(twin(np.array(pick["B"])))
            reps = [t for t in REPORT if t.startswith(task.split("_")[0])] or [task]
            rep["boxes"][name] = {
                "task": task, "from_cluster": None if j is None else int(j),
                "ladder": [{k: v for k, v in r.items() if k not in ("B", "Z")} for r in rows],
                "picked": {"lam": pick["lam"], "neurons": pick["neurons"], "chosen_on": "fit split"},
                "core+box": evaluate(f"core+{name}", core, [{"mask": pick["B"]}], names=reps + ["recall"]),
                "core+random_box": evaluate(f"core+random({name})", core, [{"mask": R}], names=reps)}
            save()
        A.save_npz(out / "boxes.npz", **{f"{k}__mask": v["mask"] for k, v in boxes.items()},
                   **{f"{k}__Z": v["Z"] for k, v in boxes.items()})
        A.write_json(out / "boxes_index.json", {k: v["task"] for k, v in boxes.items()})
    elif resume and (resume / "boxes.npz").exists():
        z = np.load(resume / "boxes.npz")
        idx = json.loads((resume / "boxes_index.json").read_text())
        for k, t in idx.items():
            boxes[k] = {"mask": mx.array(z[f"{k}__mask"]), "Z": mx.array(z[f"{k}__Z"]), "task": t}
    names = list(boxes)
    if names and "boxes" in phases:
        ov = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ov[f"{a}&{b}"] = int((boxes[a]["mask"] * boxes[b]["mask"]).sum())
        rep["box_overlap_neurons"] = ov
        rep["eval"]["core+all_ungated"] = evaluate("core+all (ungated)", core, [{"mask": boxes[n]["mask"]} for n in names])
        save()

    # ------------------------------------------------------------------ gates
    gates = {}
    if "gates" in phases and names:
        if resume and (resume / "gates.npz").exists():
            z = np.load(resume / "gates.npz")
            gates = {n: (mx.array(z[f"{n}__W"]), mx.array(z[f"{n}__b"])) for n in names}
        else:
            params = {n: {"W": mx.zeros((L, d)), "b": mx.full((L,), 2.0)} for n in names}
            opt = optim.Adam(learning_rate=args.gate_lr)
            mix = [(t, boxes[n]["task"]) for n in names for t in [n]] + [("recall", "recall")]
            mixes = []
            for _, task in mix:
                for grp in A.groups_of(tasks[task], ts[task]["fit"], cap=args.micro):
                    mixes.append((task, grp))
            clean = {}
            for task in set(t for _, t in mix):
                A.cache_clean(host, tasks[task], ts[task]["fit"], clean, task, cap=args.micro)
            eng_pos = list(range(1, 64, 4))
            eng_b = [mx.array(eng_fit[i:i + 4]) for i in range(0, len(eng_fit), 4)]   # layer-wise backward: B=4 fits
            eng_clean = []
            for ids in eng_b:
                host.ctl.reset()
                eng_clean.append(host.logits_at(ids, eng_pos))
            mx.eval(eng_clean)

            def box_list(p):
                return [{"mask": boxes[n]["mask"], "gate": (p[n]["W"], p[n]["b"]), "name": n} for n in names]

            box_layers = {n: [l for l in range(L) if float(boxes[n]["mask"][l].sum()) > 0] for n in names}

            def capture_x(ids, p):
                """MLP inputs on English under the current gates (no gradient), per box layer."""
                cap = {}

                def st(l, act, x):
                    cap[l] = mx.stop_gradient(x.astype(mx.float32))
                host.ctl.reset(); host.ctl.core, host.ctl.dense_layers = core, dense
                host.ctl.boxes = box_list(p)
                host.ctl.stats = st
                mx.eval(host.hidden(ids), *[])
                host.ctl.reset()
                mx.eval(list(cap.values()))
                return cap

            def lw(p, ids, pos, lc):
                """Layer-wise loss and gradients for every gate parameter (masks constant)."""
                tr = {}
                for n in names:
                    tr[f"W_{n}"], tr[f"b_{n}"] = p[n]["W"], p[n]["b"]

                def apply(l, sl):
                    host.ctl.reset(); host.ctl.core, host.ctl.dense_layers = core, dense
                    host.ctl.boxes = [{"mask": boxes[n]["mask"], "gate": ({l: sl[f"W_{n}"]}, {l: sl[f"b_{n}"]}), "name": n}
                                      for n in names]
                loss, g = A.layerwise_value_and_grad(
                    host, ids, tr, apply,
                    lambda hid: A.kl_rows(lc, host.head(mx.take_along_axis(hid, pos[..., None], axis=1)
                                                        if isinstance(pos, mx.array) else hid[:, mx.array(pos)]).astype(mx.float32)).mean())
                return loss, {n: {"W": g[f"W_{n}"], "b": g[f"b_{n}"]} for n in names}

            def openness(p, xcap):
                opens = [mx.sigmoid(xcap[l] @ p[n]["W"][l] + p[n]["b"][l]).mean() for n in names for l in box_layers[n]]
                return mx.stack(opens).mean() if opens else mx.array(0.0)
            vg_open = mx.value_and_grad(openness)
            from mlx.utils import tree_map
            t0 = time.time()
            for step in range(args.gate_steps):
                host.train_mode(False)
                xcap = capture_x(eng_b[step % len(eng_b)], params)
                host.train_mode(True)
                task, grp = mixes[step % len(mixes)]
                ids, pos = A.batch_of(tasks[task], grp)
                kt, g1 = lw(params, ids, pos, clean[(task, tuple(grp))])
                e = step % len(eng_b)
                ke, g2 = lw(params, eng_b[e], eng_pos, eng_clean[e])
                oe, g3 = vg_open(params, xcap)
                g = tree_map(lambda a, b, c: a + b + args.gate_penalty * c, g1, g2, g3)
                params = opt.apply_gradients(g, params)
                mx.eval(params, opt.state, kt, ke, oe)
                mx.clear_cache()
                if step == 0:
                    log(f"  gates: first step peak {A.peak_gb()} GB")
                    if A.peak_gb() > args.train_peak_cap_gb:
                        raise SystemExit(f"gates: MLX peak {A.peak_gb()} GB over the cap")
                if step % 10 == 0 or step == args.gate_steps - 1:
                    log(f"  gates step {step} task_kl={float(kt):.4f} eng_kl={float(ke):.4f} open_eng={float(oe):.3f} "
                        f"({(time.time() - t0) / (step + 1):.1f} s/step)")
            host.train_mode(False)
            gates = {n: (params[n]["W"], params[n]["b"]) for n in names}
            A.save_npz(out / "gates.npz", **{f"{n}__W": w for n, (w, _) in gates.items()},
                       **{f"{n}__b": b for n, (_, b) in gates.items()})
        gated = [{"mask": boxes[n]["mask"], "gate": gates[n], "hard": True, "name": n} for n in names]
        rep["eval"]["core+all_gated_hard"] = evaluate("core+all (gated, hard)", core, gated)
        rep["eval"]["core+all_gated_soft"] = evaluate("core+all (gated, soft)", core,
                                                      [{**b, "hard": False} for b in gated], english=False)
        # openness on English tokens (hard gates), per box
        opens = {n: [] for n in names}
        for ids in [mx.array(q[:8]) for q in [eng["tinystories_train"]["held"], eng["repo_markdown"]["held"],
                                               eng["donor_generated"]["held"]]]:
            host.ctl.reset(); host.ctl.core, host.ctl.boxes, host.ctl.dense_layers = core, gated, dense
            host.ctl.gate_log = []
            mx.eval(host.hidden(ids))
            for (n, l, g) in host.ctl.gate_log:
                opens[n].append(float(g.mean()))
            host.ctl.reset()
        rep["gate_open_on_english"] = {n: (float(np.mean(v)) if v else None) for n, v in opens.items()}
        log(f"gate openness on English (hard, mean over box layers): {rep['gate_open_on_english']}")
        save()

    # ------------------------------------------------------------------ router
    if "router" in phases and names:
        classes = sorted(set(boxes[n]["task"] for n in names)) + ["none"]
        box_of = {}
        for n in names:
            box_of.setdefault(boxes[n]["task"], []).append(n)

        def prompt_ids(it):
            return it["ids"][:len(it["ids"]) - len(it["tgt"])]
        neg_texts = ["The meeting was moved to 3 pm on Friday.", "Prices rose 12% in 2023 across the region.",
                     "She lived at 42 Elm Street for ten years.", "The train leaves at 7:45 from platform 9.",
                     "Paris is a lovely city in the spring.", "He scored 105 points over three games.",
                     "The recipe calls for 250 grams of flour.", "Our capital budget for next year is fixed.",
                     "In French class we read a short poem.", "Add the totals at the bottom of the page."]
        neg = [{"ids": host.encode(t), "cls": "none"} for t in neg_texts]
        data = {"fit": [], "held": []}
        for task in classes[:-1]:
            for part in ("fit", "held"):
                data[part] += [{"ids": prompt_ids(tasks[task][i]), "cls": task} for i in ts[task][part]]
        for part in ("fit", "held"):
            data[part] += [{"ids": prompt_ids(tasks["recall"][i]), "cls": "none"} for i in ts["recall"][part]]
            data[part] += [{"ids": q[:24], "cls": "none"} for dn in A.DOMAINS for q in split[dn][part][:20]]
        data["phrasing"] = [{"ids": prompt_ids(tasks["capitals_phrasing"][i]), "cls": "capitals"}
                            for i in ts["capitals_phrasing"]["held"]] if "capitals" in classes else []
        data["hard_negatives"] = neg
        rng2 = random.Random(args.seed)
        fit_all = data["fit"][:]
        rng2.shuffle(fit_all)
        nv = len(fit_all) // 4
        data["val"], data["train"] = fit_all[:nv], fit_all[nv:]
        layers = [int(x) for x in args.router_layers.split(",")]

        def feats(rows):
            F = {l: [] for l in layers}
            for r in rows:
                cap = {}

                def st(l, act, x, cap=cap):
                    if l in layers:
                        cap[l] = x.astype(mx.float32).mean(1)
                host.ctl.reset(); host.ctl.core, host.ctl.dense_layers = core, dense
                host.ctl.stats = st
                h = host.hidden(mx.array([r["ids"]]))
                mx.eval(h, *cap.values())
                host.ctl.reset()
                for l in layers:
                    F[l].append(np.array(cap[l][0]))
            return {l: np.stack(v) for l, v in F.items()}

        feat = {p: feats(v) for p, v in data.items() if v}
        y = {p: np.array([classes.index(r["cls"]) for r in v]) for p, v in data.items() if v}

        def fit_lr(X, yy, C=len(classes), steps=300, lr=0.1, wd=1e-3):
            mu, sd = X.mean(0), X.std(0) + 1e-6
            Xn = (X - mu) / sd
            W = np.zeros((X.shape[1], C)); b = np.zeros(C)
            Y = np.eye(C)[yy]
            for _ in range(steps):
                z = Xn @ W + b
                z -= z.max(1, keepdims=True)
                p = np.exp(z); p /= p.sum(1, keepdims=True)
                gW = Xn.T @ (p - Y) / len(yy) + wd * W
                W -= lr * gW; b -= lr * (p - Y).mean(0)
            return mu, sd, W, b

        def pred(m, X):
            mu, sd, W, b = m
            return ((X - mu) / sd @ W + b).argmax(1)
        res, models = {}, {}
        for l in layers:
            models[l] = fit_lr(feat["train"][l], y["train"])
            res[l] = {p: float((pred(models[l], feat[p][l]) == y[p]).mean()) for p in feat if p != "train"}
        best_l = max(layers, key=lambda l: res[l]["val"])
        rep["router"] = {"classes": classes, "per_layer": res, "chosen_layer": best_l, "chosen_on": "val (part of fit)",
                         "held_confusion": {}}
        P = pred(models[best_l], feat["held"][best_l])
        for ci, c in enumerate(classes):
            sel = y["held"] == ci
            rep["router"]["held_confusion"][c] = {classes[k]: int((P[sel] == k).sum()) for k in range(len(classes))}
        log(f"router: chosen layer {best_l} {res[best_l]}")
        # end to end on held: routed vs oracle vs all boxes ungated
        if gates:
            e2e = {}
            for task in classes[:-1]:
                items, idx = tasks[task], ts[task]["held"]
                rows = [{"ids": prompt_ids(items[i]), "cls": task} for i in idx]
                Fh = feats(rows)[best_l]
                routed = pred(models[best_l], Fh)
                corr = {"oracle": 0, "routed": 0}
                for k, i in enumerate(idx):
                    for arm in ("oracle", "routed"):
                        c = task if arm == "oracle" else classes[int(routed[k])]
                        bl = [{"mask": boxes[n]["mask"], "gate": gates[n], "hard": True} for n in box_of.get(c, [])]
                        corr[arm] += A.task_eval(host, items, [i], cfg(core, bl))[0]
                e2e[task] = {k: v / len(idx) for k, v in corr.items()}
                log(f"router e2e {task}: {e2e[task]}")
            rep["router"]["end_to_end_held"] = e2e
        save()

    # ------------------------------------------------------------------ prune
    if "prune" in phases and names:
        rep["prune"] = {}
        for n in names:
            task = boxes[n]["task"]
            B, Z = np.array(boxes[n]["mask"]), np.array(boxes[n]["Z"])
            idx = np.argwhere(B > 0)
            score = Z[B > 0]
            rows = []
            for ratio in [float(x) for x in args.prune_ratios.split(",")]:
                k = max(1, int(round(ratio * len(idx))))
                keep = idx[np.argsort(-score)[:k]]
                Kp = np.zeros_like(B); Kp[keep[:, 0], keep[:, 1]] = 1
                rk = idx[rng.choice(len(idx), k, replace=False)]
                Kr = np.zeros_like(B); Kr[rk[:, 0], rk[:, 1]] = 1
                g = gates.get(n)
                mk = (lambda M: [{"mask": mx.array(M), "gate": g, "hard": True}] if g is not None else [{"mask": mx.array(M)}])
                acc_imp = A.task_eval(host, tasks[task], ts[task]["held"], cfg(core, mk(Kp)))[0]
                acc_rnd = A.task_eval(host, tasks[task], ts[task]["held"], cfg(core, mk(Kr)))[0]
                rows.append({"ratio": ratio, "neurons": k, "importance_acc": acc_imp, "random_cut_acc": acc_rnd})
                log(f"[prune {n}] keep {ratio} ({k}): importance {acc_imp:.3f} random cut {acc_rnd:.3f}")
            rep["prune"][n] = {"unpruned_neurons": int(B.sum()), "rows": rows, "gated": n in gates}
            save()

    # ------------------------------------------------------------------ spec proxy
    if "spec" in phases and names and gates:
        gated = [{"mask": boxes[n]["mask"], "gate": gates[n], "hard": True} for n in names]
        prompts = []
        for task in ("capitals", "arithmetic", "french", "recall"):
            for i in ts[task]["held"][:6]:
                it = tasks[task][i]
                prompts.append((task, it["ids"][:len(it["ids"]) - len(it["tgt"])]))
        for dn in A.DOMAINS:
            prompts += [("english_" + dn, q[:24]) for q in split[dn]["held"][:4]]
        k, gen = 8, 24
        rep["spec"] = {}
        for vname, vcfg in (("core+boxes", cfg(core, gated)), ("full", cfg(None))):
            per = {}
            for task, p in prompts:
                seq = list(p)
                for _ in range(gen):                         # verifier's greedy continuation (no cache)
                    vcfg()
                    nt = int(host.logits_at(mx.array([seq]), [len(seq) - 1])[0, 0].argmax())
                    host.ctl.reset()
                    seq.append(nt)
                cfg(core)()                                  # drafter = core alone, teacher-forced
                dl = host.logits_at(mx.array([seq]), list(range(len(p) - 1, len(seq) - 1)))[0].argmax(-1)
                host.ctl.reset()
                agree = (np.array(dl) == np.array(seq[len(p):])).tolist()
                steps, acc, i = 0, 0, 0
                while i < len(agree):
                    run = 0
                    while run < k and i + run < len(agree) and agree[i + run]:
                        run += 1
                    acc += run + 1                           # accepted drafts + the verifier's own token
                    i += run + 1
                    steps += 1
                per.setdefault(task.split("_")[0], []).append(acc / steps)
                mx.clear_cache()
            rep["spec"][vname] = {t: float(np.mean(v)) for t, v in per.items()}
            log(f"spec proxy (verifier {vname}): tokens per verify step {rep['spec'][vname]}")
        save()
    save()
    log(f"done in {time.time() - t_start:.0f}s, MLX peak {A.peak_gb()} GB")


if __name__ == "__main__":
    main()
