#!/usr/bin/env python3
"""Puzzle stage: place unlabelled neurons by what they co-activate with (Pierre, 2026-11-14). CPU only.

"Cluster neurons that can't be classified either stay in the core or are placed like a puzzle, by what
they co-activate with; the gaps get clearer as more gets classified."

Inputs: a topic capture, its labelling run (labels.json + per-neuron d arrays), the core, the clusters.
  anchors      neurons of clusters/subclusters that passed the selectivity gates for a topic (a label's
               functional check comes later, in the GPU window), plus the core as one anchor ("core")
  candidates   every neuron of an unlabelled cluster, taken one neuron at a time
Each round, per candidate neuron n and anchor topic t (fit texts only):
  c_t(n)   Pearson correlation of n's per-text response (rel) with anchor t's score (mean rel of its neurons)
  d_t(n)   n's own selectivity for t (fit d from the labelling run)
  n joins t when c_t >= --corr, d_t >= --d, c_t > c_core(n) (it follows t more than the core), with
  confidence c_t * min(1, d_t / 2). Several topics passing within --margin of the best: weighted
  multi-membership (weights proportional to confidence); the neuron is "shared".
  New members become anchors for the next round. Held confirmation per round: the fraction of the round's
  new members with held d_t >= 0.5. Stop when a round adds < --min-new of the candidates, its held
  confirmation falls below --min-confirm, or after --rounds.
Explain away: each leftover neuron's fit response is regressed on [all anchor scores, core score]; R^2
says how much known activity explains it. The texts with the largest summed positive residual over the
leftovers are listed as passages nobody probed (hints for new topics).
Leftover tags (they stay in the core, with evidence): general (R^2 from the core score alone >= 0.5),
shared (multi-membership), unexplained (the rest), each with its top co-activation partners. Structural
(early/late-layer) neurons get no category: Pierre 2026-11-14, "untagged until the puzzle is solved";
only their layer histogram is reported.
Outputs: OUT/puzzle.json (rounds, fractions, per-topic propagated sets, passages), OUT/puzzle.npz
(per-topic pairs and weights; leftover tags), and OUT/units.npz + OUT/plan_extra.json: one "puzzle:<topic>"
unit per topic for the group-level twin-controlled removal check (amoeba_topics_plan.py --extra).
A propagated label counts only after that check (registry), else it is cluster_unverified.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import amoeba_topics_lib as T


def zcols(X):
    X = X - X.mean(0, keepdims=True)
    return X / np.maximum(np.linalg.norm(X, axis=0, keepdims=True), 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--arrays", required=True)
    ap.add_argument("--core", required=True)
    ap.add_argument("--clusters", default="", help="pool_clusters.npz (default: labels/formed_clusters.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corr", type=float, default=0.5)
    ap.add_argument("--d", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--min-new", type=float, default=0.01)
    ap.add_argument("--min-confirm", type=float, default=0.7)
    ap.add_argument("--passages", type=int, default=40)
    ap.add_argument("--min-unit", type=int, default=16, help="smallest propagated set sent to the GPU check")
    args = ap.parse_args()
    out = T.fresh_dir(args.out)
    log = T.make_log(out)
    cap = Path(args.capture)
    meta = json.loads((cap / "meta.json").read_text())
    L, N, S = meta["layers"], meta["neurons"], meta["sequences"]
    items = json.loads((cap / "items.json").read_text())
    split = np.array([it["split"] for it in items])
    fit = split == "fit"
    lab = json.loads((Path(args.labels) / "labels.json").read_text())
    core = np.load(args.core)["core"] > 0
    cz = np.load(args.clusters) if args.clusters else np.load(Path(args.labels) / "formed_clusters.npz")
    cpool, clab = cz["pool"], cz["labels"].astype(int)
    subs = np.load(Path(args.labels) / "subclusters.npz") if (Path(args.labels) / "subclusters.npz").exists() else {}
    node_ids = json.loads((Path(args.arrays) / "neuron_d_nodes.json").read_text())
    nd_fit = np.load(Path(args.arrays) / "neuron_d_fit.npy", mmap_mode="r")
    nd_held = np.load(Path(args.arrays) / "neuron_d_held.npy", mmap_mode="r")

    # ---------------------------------------------------------------- anchors and candidates
    owner = np.full((L, N), -1, np.int32)            # anchor topic index per neuron (-1 none)
    topics = []
    unl = []
    for c in lab["clusters"]:
        j = int(c["id"].split("C")[1])
        members = cpool[clab == j]
        if c["labels"]:
            for l in c["labels"][:1]:                 # the cluster's strongest label anchors it
                t = l["topic"]
                if t not in topics:
                    topics.append(t)
                owner[members[:, 0], members[:, 1]] = topics.index(t)
        else:
            unl.append(members)
        for s in c.get("subclusters", {}).get("items", []):
            if s["labels"]:
                t = s["labels"][0]["topic"]
                if t not in topics:
                    topics.append(t)
                p = np.asarray(subs[s["id"].split(":")[1]])
                owner[p[:, 0], p[:, 1]] = topics.index(t)
    cand = np.concatenate(unl) if unl else np.zeros((0, 2), int)
    cand = cand[owner[cand[:, 0], cand[:, 1]] < 0]
    log(f"{meta['checkpoint']}: {len(topics)} anchor topics, {int((owner >= 0).sum())} anchor neurons, "
        f"{len(cand)} candidate neurons from {len(unl)} unlabelled clusters")
    T_ = len(topics)
    tix = [node_ids.index(t) if t in node_ids else -1 for t in topics]
    member_w = {}                                    # (l, n) -> {topic: weight}
    cand_by_l = T.by_layer(cand) if len(cand) else {}

    def layer_R(l):
        R = np.asarray(np.load(cap / f"resp_L{l:02d}.npy", mmap_mode="r"), dtype=np.float32)
        R /= np.maximum(R[fit].mean(0, keepdims=True), 1e-6)
        return R

    def anchor_scores():
        acc = np.zeros((S, T_ + 1))
        cnt = np.zeros(T_ + 1)
        for l in range(L):
            R = layer_R(l)
            o = owner[l]
            for k in np.unique(o[o >= 0]):
                cols = np.where(o == k)[0]
                acc[:, k] += R[:, cols].sum(1)
                cnt[k] += len(cols)
            cc = np.where(core[l])[0]
            acc[:, T_] += R[:, cc].sum(1)
            cnt[T_] += len(cc)
        return acc / np.maximum(cnt, 1)

    rounds = []
    t0 = time.time()
    for rd in range(1, args.rounds + 1):
        A = anchor_scores()                              # [S, T+1], last column = core
        Az = zcols(A[fit])
        new, confirm = [], []
        for l, cols in cand_by_l.items():
            cols = np.array([c for c in cols if owner[l, c] < 0 and (l, c) not in member_w])
            if not len(cols):
                continue
            R = layer_R(l)
            C = zcols(R[fit][:, cols]).T @ Az            # [n, T+1] correlations
            dfit = np.stack([np.asarray(nd_fit[k, l, cols], np.float32) if k >= 0 else np.zeros(len(cols))
                             for k in tix], 1)            # [n, T]
            dheld = np.stack([np.asarray(nd_held[k, l, cols], np.float32) if k >= 0 else np.zeros(len(cols))
                              for k in tix], 1)
            ok = (C[:, :T_] >= args.corr) & (dfit >= args.d) & (C[:, :T_] > C[:, T_:T_ + 1])
            conf = np.where(ok, C[:, :T_] * np.minimum(1.0, dfit / 2.0), 0.0)
            for i in np.where(ok.any(1))[0]:
                best = conf[i].max()
                win = np.where(conf[i] >= best - args.margin)[0]
                w = conf[i, win] / conf[i, win].sum()
                key = (int(l), int(cols[i]))
                member_w[key] = {topics[k]: round(float(x), 3) for k, x in zip(win, w)}
                k0 = int(win[np.argmax(conf[i, win])])
                new.append((key, k0, float(best)))
                confirm.append(bool(dheld[i, k0] >= 0.5))
        frac = len(new) / max(1, len(cand))
        conf_rate = float(np.mean(confirm)) if confirm else 1.0
        rounds.append({"round": rd, "new_members": len(new), "fraction_of_candidates": round(frac, 4),
                       "held_confirm_rate": round(conf_rate, 3), "elapsed_s": round(time.time() - t0, 1),
                       "by_topic": {t: sum(1 for _, k, _ in new if topics[k] == t) for t in topics if any(topics[k] == t for _, k, _ in new)}})
        log(f"round {rd}: +{len(new)} ({frac:.4f} of candidates), held confirm {conf_rate:.2f}")
        if conf_rate < args.min_confirm:                    # roll the round back: its members are not kept
            for key, _, _ in new:
                member_w.pop(key, None)
            rounds[-1]["rolled_back"] = len(new)
            rounds[-1]["stop"] = "held confirmation fell (round rolled back)"
            break
        for (l, n), k0, _ in new:
            owner[l, n] = k0                                # new members anchor the next round
        if frac < args.min_new:
            rounds[-1]["stop"] = "few new members"
            break

    # ---------------------------------------------------------------- explain away the leftovers
    A = anchor_scores()
    X = np.concatenate([A[fit], np.ones((int(fit.sum()), 1))], 1)
    Xc = np.concatenate([A[fit][:, -1:], np.ones((int(fit.sum()), 1))], 1)
    tags = {"general": [], "shared": [], "unexplained": []}
    partners = {}
    resid_sum = np.zeros(int(fit.sum()))
    fit_idx = np.where(fit)[0]
    r2_all = []
    Az = zcols(A[fit])
    for l, cols in cand_by_l.items():
        left = np.array([c for c in cols if (l, c) not in member_w])
        shared = [c for c in cols if (l, c) in member_w and len(member_w[(l, c)]) > 1]
        tags["shared"] += [(int(l), int(c)) for c in shared]
        if not len(left):
            continue
        R = layer_R(l)
        Y = R[fit][:, left]
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        r2 = 1 - (res ** 2).sum(0) / np.maximum(((Y - Y.mean(0)) ** 2).sum(0), 1e-12)
        bc, *_ = np.linalg.lstsq(Xc, Y, rcond=None)
        rc = Y - Xc @ bc
        r2c = 1 - (rc ** 2).sum(0) / np.maximum(((Y - Y.mean(0)) ** 2).sum(0), 1e-12)
        C = zcols(Y).T @ Az
        for i, c in enumerate(left):
            key = (int(l), int(c))
            top = np.argsort(-C[i])[:5]
            partners[key] = [(("core" if k == T_ else topics[k]), round(float(C[i, k]), 3)) for k in top]
            if r2c[i] >= 0.5:
                tags["general"].append(key)
            else:
                tags["unexplained"].append(key)
                resid_sum += np.maximum(res[:, i], 0) / max(float(res[:, i].std()), 1e-6)
        r2_all.append(r2)
    order = np.argsort(-resid_sum)[:args.passages]
    passages = [{"item": items[fit_idx[i]]["id"], "topics": items[fit_idx[i]]["topics"],
                 "residual_score": round(float(resid_sum[i]), 2)} for i in order]
    r2_all = np.concatenate(r2_all) if r2_all else np.zeros(0)
    left_layers = np.bincount([k[0] for k in tags["unexplained"] + tags["general"]], minlength=L).tolist()

    # ---------------------------------------------------------------- outputs
    per_topic = {}
    for key, ws in member_w.items():
        for t, w in ws.items():
            per_topic.setdefault(t, []).append((key[0], key[1], w))
    arrays, units = {}, []
    for t, rows in sorted(per_topic.items()):
        a = np.array([(l, n) for l, n, _ in rows], np.int32)
        arrays[f"{t}|pairs"] = a
        arrays[f"{t}|weights"] = np.array([w for _, _, w in rows], np.float32)
        if len(a) >= args.min_unit:
            key = f"puzzle_{t}"
            units.append({"unit": f"puzzle:{t}", "kind": "puzzle", "key": key, "labelled_topics": [t],
                          "candidate_topic": None, "tests": [t], "add_arm": True, "bands": False, "twin": "pool",
                          "neurons": int(len(a)), "note": "propagated by co-activation; counts only if the group-level removal check passes"})
    for k, v in tags.items():
        arrays[f"tag|{k}"] = np.array(v, np.int32).reshape(-1, 2)
    np.savez_compressed(out / "puzzle.npz", **arrays)
    np.savez_compressed(out / "units.npz", **{u["key"]: arrays[f"{u['unit'].split(':', 1)[1]}|pairs"] for u in units})
    T.write_json(out / "plan_extra.json", {"units": units})
    total = L * N
    n_anchor0 = int(sum(1 for c in lab["clusters"] if c["labels"]))
    classified = int((owner >= 0).sum())
    rep = {"schema": "mlx2.amoeba-topic-puzzle.v1", "checkpoint": meta["checkpoint"], "mlx2_source_revision": T.git_rev(),
           "labels": str(args.labels), "capture": str(cap), "args": vars(args), "anchor_topics": topics, "rounds": rounds,
           "fractions_of_all_neurons": {
               "core": round(float(core.mean()), 4),
               "classified_pool (anchor clusters + propagated)": round(classified / total, 4),
               "propagated": round(len(member_w) / total, 4),
               "shared": round(len(tags["shared"]) / total, 4),
               "core_by_evidence_general": round(len(tags["general"]) / total, 4),
               "unexplained": round(len(tags["unexplained"]) / total, 4)},
           "fractions_of_candidates": {"propagated": round(len(member_w) / max(1, len(cand)), 4),
                                       "general": round(len(tags["general"]) / max(1, len(cand)), 4),
                                       "unexplained": round(len(tags["unexplained"]) / max(1, len(cand)), 4)},
           "explain_away": {"r2_median": round(float(np.median(r2_all)), 3) if len(r2_all) else None,
                            "r2_p90": round(float(np.percentile(r2_all, 90)), 3) if len(r2_all) else None},
           "unexplained_passages": passages,
           "leftover_layer_histogram": left_layers,
           "per_topic_propagated": {t: len(v) for t, v in sorted(per_topic.items())},
           "units_for_gpu_check": len(units), "labelled_clusters": n_anchor0,
           "partners_sample": {f"L{k[0]}:N{k[1]}": v for k, v in list(partners.items())[:50]}}
    T.write_json(out / "puzzle.json", rep)
    log(f"done: {rep['fractions_of_all_neurons']}")


if __name__ == "__main__":
    main()
