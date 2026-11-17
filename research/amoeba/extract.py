"""Amoeba M0 window 2: find the geo.capitals box in the donor and test it.

Units are transcoder features at layers 6-18 (plan section 4.1). Interventions
never touch the donor's weights; they edit the MLP output at runtime by the
difference a feature makes:

    y_new = y + sy * W_d[:, S] (a'_S - a_S)

so the transcoder's reconstruction error never enters an ablation.

* attribution  attribution patching: grad of the restricted answer log-prob
               w.r.t. a per-feature multiplier (= a_f * dmetric/da_f), summed
               over positions. The transcoder error and skip terms get their own
               multipliers, which gives the error-node share.
* necessity    accuracy with the box resample-ablated (box activations replaced
               by those of random off-topic tokens), on the extract and the
               held-out countries. Control: a random box with the same per-layer
               sizes.
* specificity  topic accuracy drop / off-topic NLL increase; the language topic
               is reported as a neighbouring control.
* sufficiency  attention frozen to the clean run; at layers 6-18 the MLP output
               is the box's features plus the off-topic mean of everything else.
               Control: the empty box (everything mean-ablated).
* absorb loop  re-trace on the donor with the box ablated, add the newly
               exposed features, repeat (Hydra measurement).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from research.amoeba import evals, transcoders as T
from research.amoeba.donor import Donor
from research.amoeba.m0 import DATA, LAYERS, atomic, source_rev, tokenizer


class Lab:
    def __init__(self, tc_dir: Path, seed=0):
        self.donor = Donor()
        self.tok = tokenizer(self.donor)
        self.tcs = T.load(Path(tc_dir))
        self.F = next(iter(self.tcs.values())).F
        self.spec = evals.load_topic()
        self.prompts = {s: evals.topic_prompts(self.spec, s, self.tok) for s in ("extract", "heldout")}
        self.control = evals.topic_prompts(self.spec, None, self.tok, control=True)
        self.off = evals.offtopic_texts(DATA, self.tok)
        self.rng = np.random.default_rng(seed)
        self._bank()

    # ---- off-topic statistics: activation bank (for resample) and MLP-output means (for sufficiency)
    def _bank(self, n_tok=2048):
        acts = {i: [] for i in LAYERS}; ys = {i: [] for i in LAYERS}; xs = {i: [] for i in LAYERS}
        for k in range(0, len(self.off), 8):
            b = np.array(self.off[k:k + 8])[:, :-1]
            _, caps, _ = self.donor.forward(mx.array(b), record=set(LAYERS), stop_after=max(LAYERS))
            for i in LAYERS:
                x, y = (c.reshape(-1, self.donor.d) for c in caps[i])
                acts[i].append(self.tcs[i].encode(x)); ys[i].append(y); xs[i].append(x)
            mx.eval(acts, ys)
        self.bank = {i: mx.concatenate(acts[i])[:n_tok] for i in LAYERS}
        self.bank_n = self.bank[LAYERS[0]].shape[0]
        self.off_y = {i: mx.concatenate(ys[i]) for i in LAYERS}
        self.off_a = {i: mx.concatenate(acts[i]) for i in LAYERS}
        self.off_x = {i: mx.concatenate(xs[i]) for i in LAYERS}
        self.off_fire = {i: np.array(mx.mean(self.off_a[i] > 0, axis=0)) for i in LAYERS}
        mx.eval(self.bank, self.off_y, self.off_x)

    # ---- hooks
    def resample_hook(self, box: dict[int, np.ndarray], seed=0):
        rng = np.random.default_rng(seed)
        ids = {i: mx.array(v.astype(np.int32)) for i, v in box.items() if len(v)}
        Wd = {i: self.tcs[i].feature_write(ids[i]) for i in ids}

        def hook(i, x, y):
            if i not in ids:
                return None
            tc = self.tcs[i]
            a = tc.encode(x)[..., ids[i]]
            rows = mx.array(rng.integers(0, self.bank_n, size=x.shape[:-1]).astype(np.int32))
            a_new = self.bank[i][rows][..., ids[i]]
            return y + (a_new - a) @ Wd[i].T
        return hook

    def sufficiency_hook(self, box, keep_skip=False, all_features=False):
        """MLP output at layers 6-18 = kept part + off-topic mean of the rest.

        kept part: the box's features (all features if all_features), plus the
        transcoder's linear skip path if keep_skip. all_features + keep_skip is the
        method's ceiling: the transcoder reconstruction with only its error mean-ablated."""
        ids = {i: (None if all_features else mx.array(box.get(i, np.zeros(0)).astype(np.int32))) for i in LAYERS}

        def kept(i, x):
            tc = self.tcs[i]
            a = tc.encode(x)
            if ids[i] is None:
                part = tc.sy * (a @ tc.W_d.T)
            elif ids[i].size:
                part = a[..., ids[i]] @ tc.feature_write(ids[i]).T
            else:
                part = mx.zeros(x.shape[:-1] + (self.donor.d,))
            if keep_skip:
                part = part + tc.sy * ((x / tc.sx) @ tc.W_s.T + tc.b_d)
            return part
        self._suff_cache = getattr(self, "_suff_cache", {})

        def mean_rest(i):
            key = (i, keep_skip, all_features, None if ids[i] is None else tuple(np.array(ids[i]).tolist()))
            if key not in self._suff_cache:
                self._suff_cache[key] = mx.mean(self.off_y[i] - kept(i, self.off_x[i]), axis=0)
            return self._suff_cache[key]

        def hook(i, x, y):
            if i not in ids:
                return None
            return kept(i, x) + mean_rest(i)
        return hook

    # ---- evaluations
    def acc(self, split, hook=None, control=False):
        p = self.control if control else self.prompts[split]
        return evals.topic_accuracy(self.donor, p, hook=hook)

    def offnll(self, hook=None):
        return evals.offtopic_loss(self.donor, self.off, hook=hook)

    def sufficiency(self, box, split, **kw):
        """Topic accuracy with attention frozen to the clean run and only the kept part at layers 6-18."""
        hook = self.sufficiency_hook(box, **kw)
        correct, n = 0, 0
        for batch in evals._batches(self.prompts[split], 64):
            ids = mx.array([b[0] for b in batch])
            _, _, probs = self.donor.forward(ids, keep_probs=True)
            logits, _, _ = self.donor.forward(ids, mlp_hook=hook, attn_probs=probs)
            cands = list(batch[0][2])
            sub = np.array(logits[:, -1][:, mx.array(cands)])
            for j, b in enumerate(batch):
                correct += int(sub[j].argmax() == cands.index(b[1])); n += 1
        return correct / n

    # ---- attribution
    def attribute(self, split, box=None, seed=0):
        """Per-prompt feature attributions (summed over positions) on the donor, optionally with a box ablated."""
        ablate = self.resample_hook(box, seed) if box else None
        per_prompt, err_share = [], []
        for batch in evals._batches(self.prompts[split], 32):
            ids = mx.array([b[0] for b in batch])
            cands = mx.array(list(batch[0][2]))
            ans = mx.array([list(batch[0][2]).index(b[1]) for b in batch])
            B, Tn = ids.shape
            mult = {i: mx.ones((B, Tn, self.F)) for i in LAYERS}
            emult = {i: mx.ones((B, Tn, 1)) for i in LAYERS}

            def metric(mult, emult):
                def hook(i, x, y):
                    if ablate is not None:
                        y = ablate(i, x, y) if i in (box or {}) else y
                    if i not in mult:
                        return y
                    tc = self.tcs[i]
                    a = tc.encode(x)
                    y_hat = tc.decode(a, x)
                    err = y - y_hat
                    feat = (a * mult[i]) @ tc.W_d.T * tc.sy
                    skip = tc.sy * ((x / tc.sx) @ tc.W_s.T + tc.b_d)
                    return mx.stop_gradient(y - y_hat - err) + feat + skip + err * emult[i]
                logits, _, _ = self.donor.forward(ids, mlp_hook=hook)
                sub = logits[:, -1][:, cands]
                lp = sub - mx.logsumexp(sub, axis=-1, keepdims=True)
                return mx.sum(mx.take_along_axis(lp, ans[:, None], axis=-1))

            g_mult, g_err = mx.grad(metric, argnums=(0, 1))(mult, emult)
            # attribution = a * dM/da  ==  dM/dmult at mult = 1 (mult multiplies a)
            attr = {i: mx.sum(g_mult[i], axis=1) for i in LAYERS}           # (B, F)
            eattr = {i: mx.sum(g_err[i], axis=(1, 2)) for i in LAYERS}       # (B,)
            mx.eval(attr, eattr)
            A = np.stack([np.array(attr[i]) for i in LAYERS], axis=1)        # (B, L, F)
            E = np.stack([np.array(eattr[i]) for i in LAYERS], axis=1)       # (B, L)
            for j in range(B):
                per_prompt.append(A[j])
                err_share.append(np.abs(E[j]).sum() / (np.abs(E[j]).sum() + np.abs(A[j]).sum() + 1e-12))
        return np.stack(per_prompt), float(np.mean(err_share))

    def topic_fire(self, split="extract", last=2):
        acc = {i: [] for i in LAYERS}
        for batch in evals._batches(self.prompts[split], 64):
            _, caps, _ = self.donor.forward(mx.array([b[0] for b in batch]), record=set(LAYERS),
                                            stop_after=max(LAYERS))
            for i in LAYERS:
                acc[i].append(np.array(mx.mean(self.tcs[i].encode(caps[i][0][:, -last:]) > 0, axis=(0, 1))))
        return {i: np.mean(acc[i], axis=0) for i in LAYERS}


def select(A, topic_fire, off_fire, *, top=40, p=0.10, q=0.25, exclude=None):
    """Features in the top-`top` positive attributions of >= p of prompts, firing on off-topic text < q x topic rate."""
    n, L, F = A.shape
    flat = A.reshape(n, -1)
    if exclude is not None:
        flat = flat.copy(); flat[:, exclude] = -np.inf
    top_idx = np.argpartition(-flat, top, axis=1)[:, :top]
    counts = np.zeros(L * F)
    for j in range(n):
        idx = top_idx[j][flat[j, top_idx[j]] > 0]
        counts[idx] += 1
    frac = counts / n
    tf = np.concatenate([topic_fire[i] for i in LAYERS]); of = np.concatenate([off_fire[i] for i in LAYERS])
    ratio = of / np.maximum(tf, 1e-9)
    keep = np.where((frac >= p) & (ratio < q))[0]
    return keep, frac, ratio


def to_box(flat_ids, F):
    box = {}
    for f in flat_ids:
        box.setdefault(LAYERS[f // F], []).append(f % F)
    return {i: np.array(sorted(v)) for i, v in box.items()}


def random_box_like(box, F, rng):
    return {i: np.sort(rng.choice(F, size=len(v), replace=False)) for i, v in box.items()}


def window2(args):
    out = Path(args.out)
    t0 = time.time()
    lab = Lab(Path(args.tc))
    if args.smoke:
        lab.prompts = {k: v[:24] for k, v in lab.prompts.items()}
        lab.control = lab.control[:16]; lab.off = lab.off[:8]
    max_rounds = 2 if args.smoke else 8
    chance = lab.spec["chance"]
    rec = {"schema": "amoeba.m0.window2.v1", "donor": lab.donor.artifact, "transcoders": args.tc,
           "source_revision": source_rev(), "started": t0, "chance": chance}
    clean = {"extract": lab.acc("extract"), "heldout": lab.acc("heldout"),
             "control_language": lab.acc(None, control=True), "offtopic_nll": lab.offnll()}
    rec["clean"] = clean
    atomic(out / "window2.json", rec)
    print("clean", json.dumps(clean), flush=True)
    topic_fire = lab.topic_fire()
    box_flat = np.zeros(0, dtype=np.int64)
    rounds = []
    cap = int(0.10 * len(LAYERS) * lab.F)
    for r in range(max_rounds):
        box = to_box(box_flat, lab.F) if len(box_flat) else None
        A, err_share = lab.attribute("extract", box=box, seed=r)
        new, frac, ratio = select(A, topic_fire, lab.off_fire, exclude=box_flat if len(box_flat) else None)
        new = np.setdiff1d(new, box_flat)
        box_flat = np.union1d(box_flat, new)
        box = to_box(box_flat, lab.F)
        hook = lab.resample_hook(box, seed=100 + r)
        ab = {"extract": lab.acc("extract", hook), "heldout": lab.acc("heldout", hook),
              "control_language": lab.acc(None, hook, control=True), "offtopic_nll": lab.offnll(hook)}
        drop = clean["heldout"]["acc"] - ab["heldout"]["acc"]
        d_off = ab["offtopic_nll"] - clean["offtopic_nll"]
        spec_ratio = drop / max(d_off, 1e-4)
        row = {"round": r, "added": int(len(new)), "box_size": int(len(box_flat)),
               "per_layer": {int(i): int(len(v)) for i, v in box.items()},
               "error_share_trace": err_share, "ablated": ab, "heldout_drop": drop,
               "offtopic_dnll": d_off, "specificity": spec_ratio, "sec": time.time() - t0}
        rounds.append(row)
        rec["absorb_rounds"] = rounds
        atomic(out / "window2.json", rec)
        print("round", r, json.dumps({k: row[k] for k in ("added", "box_size", "heldout_drop", "offtopic_dnll",
                                                           "specificity", "error_share_trace")}),
              "heldout_acc", ab["heldout"]["acc"], flush=True)
        if ab["heldout"]["acc"] <= chance + 0.05:
            rec["stop"] = "absorbed: held-out accuracy at chance + 5 points"; break
        if r > 0 and spec_ratio < 2:
            rec["stop"] = "specificity below 2: ablation is hurting everything"; break
        if len(box_flat) > cap:
            rec["stop"] = "box exceeds 10% of donor transcoder features: topic not localised"; break
        if len(new) == 0:
            rec["stop"] = "no new features exposed"; break
    else:
        rec["stop"] = f"round limit ({max_rounds})"
    box = to_box(box_flat, lab.F)
    np.savez(out / "box.npz", **{f"L{i}": v for i, v in box.items()})
    # controls and final tests
    rng = np.random.default_rng(5)
    rbox = random_box_like(box, lab.F, rng)
    rh = lab.resample_hook(rbox, seed=7)
    rec["random_box_control"] = {"heldout": lab.acc("heldout", rh), "offtopic_nll": lab.offnll(rh),
                                 "control_language": lab.acc(None, rh, control=True)}
    rec["sufficiency"] = {
        "box_heldout_acc": lab.sufficiency(box, "heldout"),
        "box_extract_acc": lab.sufficiency(box, "extract"),
        "empty_box_heldout_acc": lab.sufficiency({}, "heldout"),
        "random_box_heldout_acc": lab.sufficiency(rbox, "heldout"),
        "box_plus_skip_heldout_acc": lab.sufficiency(box, "heldout", keep_skip=True),
        "skip_only_heldout_acc": lab.sufficiency({}, "heldout", keep_skip=True),
        "ceiling_all_features_plus_skip_heldout_acc": lab.sufficiency({}, "heldout", keep_skip=True, all_features=True),
        "clean_heldout_acc": clean["heldout"]["acc"],
    }
    s = rec["sufficiency"]; s["frac_of_clean"] = s["box_heldout_acc"] / max(clean["heldout"]["acc"], 1e-9)
    last = rounds[-1]
    rec["tests"] = {
        "necessity": {"heldout_acc": last["ablated"]["heldout"]["acc"], "pass_threshold": chance + 0.05,
                      "pass": last["ablated"]["heldout"]["acc"] <= chance + 0.05},
        "specificity": {"value": last["specificity"], "offtopic_dnll": last["offtopic_dnll"],
                        "pass": last["specificity"] >= 5 and last["offtopic_dnll"] <= 0.05},
        "sufficiency": {"frac_of_clean": s["frac_of_clean"], "pass": s["frac_of_clean"] >= 0.8},
        "error_node_share": last["error_share_trace"],
    }
    rec["finished"] = time.time()
    atomic(out / "window2.json", rec)
    print("tests", json.dumps(rec["tests"]), flush=True)
    print("controls", json.dumps(rec["random_box_control"]), json.dumps(rec["sufficiency"]), flush=True)
