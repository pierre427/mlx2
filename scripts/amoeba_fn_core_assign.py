#!/usr/bin/env python3
"""Core re-probe: propose moving clusters of core neurons into existing tiles (Pierre 2026-11-19: "After that do
further probing of the core to see what other clusters can be split into an existing expert tile"). CPU only; reads an
scripts/amoeba_fn_partition.py run (tiles.npz + partition.json and the compare maps it names). Writes a proposal and
a candidate partition; the GPU battery (stage 03-coreassign) decides.

Profiles (box space): for each unit, the vector over the partition's B boxes of the maps' mean_target (the per-token
mean contribution |h| * w on each box's own corpus, the drive the partition uses); its Hellinger form
sqrt(p / sum p) is the profile. Only the movable core is profiled: core units outside T2 (the English map is outside
the core when the partition has an English box; without one its units stay put too).
Clusters: spherical k-means (scripts/amoeba_fn_profile_analyse.py) on a seeded sample of the movable core, then every
movable unit assigned to its nearest centroid.
Tile signatures: the mean Hellinger profile of a seeded sample of each candidate tile's units (box, pair, family,
joint tiles; the English box and the core excluded), normalised.
Match per cluster: cosine of its centroid to each signature (best, margin to the second best), and member mass = the
share of the centroid's profile mass (p) on the tile's member boxes and their dotted-name lineage. A cluster is
proposed for its best tile when cos >= --min-cos, margin >= --min-margin, member mass >= --min-mass and it has >=
--min-units units; then each of its units moves only if its own cosine to that signature is >= --min-cos (so a
cluster's stragglers stay in the core).
Writes OUT/proposals.json (every cluster: size, top boxes by mass, best / second tile, scores, proposed or why not),
OUT/partition/ (tiles.npz + partition.json: the input partition with the moves applied, an exact cover; partition.json
"core_assign" lists the moves per tile; every other key unchanged, so the battery runs on it as on the input) and
OUT/moves.npz (per target tile the moved units).

Usage: amoeba_fn_core_assign.py --partition DIR --out DIR [--clusters 128] [--sample 400000] [--min-cos 0.9]
       [--min-margin 0.02] [--min-mass 0.5] [--min-units 2000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from amoeba_fn_partition import save_npz  # noqa: E402
from amoeba_fn_profile_analyse import spherical_kmeans  # noqa: E402


def tile_members(key, prep):
    if key.startswith("box:"):
        return [key[4:]]
    if key.startswith("pair:"):
        return key[5:].split("__")
    if key.startswith("joint:"):
        return key[6:].split("+")
    if key.startswith("group:"):
        return list(((prep.get("groups") or {}).get("kept") or {}).get(key[6:], {}).get("members") or [])
    return []


def lineage_cols(mem, names):
    out = set()
    for m in mem:
        for i, x in enumerate(names):
            if x == m or x.startswith(m + ".") or m.startswith(x + "."):
                out.add(i)
    return sorted(out)


def hell(X):
    """In place: X [U, B] >= 0 -> its Hellinger rows sqrt(p), p = x / sum x (unit norm; H**2 = p). A zero row stays 0."""
    np.maximum(X, 0, out=X)
    for s0 in range(0, X.shape[0], 500000):
        x = X[s0:s0 + 500000]
        s = x.sum(1, keepdims=True)
        np.divide(x, np.maximum(s, 1e-30), out=x)
        np.sqrt(x, out=x)
    return X


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clusters", type=int, default=128)
    ap.add_argument("--sample", type=int, default=400000)
    ap.add_argument("--sig-sample", type=int, default=20000)
    ap.add_argument("--min-cos", type=float, default=0.9)
    ap.add_argument("--min-margin", type=float, default=0.02)
    ap.add_argument("--min-mass", type=float, default=0.5)
    ap.add_argument("--min-units", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--en-ratio", type=float, default=0.25, help="a unit whose English drive is > this x its "
                    "strongest box drive stays in the core (the partition's exclusivity ratio)")
    a = ap.parse_args(argv)
    pd, out = Path(a.partition), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    prep = json.loads((pd / "partition.json").read_text())
    tz = np.load(pd / "tiles.npz")
    keys = [k for k in tz.files if k != "layers"]
    layers = [int(x) for x in tz["layers"]]
    core = tz["core"]
    L, n = core.shape
    t2 = np.load(prep["inputs"]["t2"])["core"] > 0
    t2 = t2[layers]
    if t2.shape[1] != n:
        t2 = np.repeat(t2, n // t2.shape[1], axis=1)
    eb = (prep.get("english_box") or {}).get("name")
    movable = core & ~t2
    if not eb:                                         # no English box: the English map's units stay in the core
        cz = np.load(Path(prep["inputs"]["core_map"]) / "map.npz")
        movable &= ~np.stack([cz[f"target_{s}"] for s in range(L)])
    names = sorted(b for b, v in prep["boxes"].items() if not v.get("english_box"))
    B = len(names)
    pos = np.flatnonzero(movable)
    U = pos.size
    print(f"movable core units {U:,} of core {int(core.sum()):,}; {B} boxes", flush=True)
    X = np.zeros((U, B + 1), np.float32)              # last column: English (the maps' English baseline mean)
    en_n = 0
    for j, b in enumerate(names):                      # one map resident at a time; [U, B+1] fp32 (~2 GB at 5 M x 110)
        z = np.load(Path(prep["boxes"][b]["map"]) / "map.npz")
        X[:, j] = np.concatenate([z[f"mean_target_{s}"] for s in range(L)])[pos]
        X[:, B] += np.concatenate([z[f"mean_baseline_{s}"] for s in range(L)])[pos]
        en_n += 1
    X[:, B] /= max(en_n, 1)
    # review 2026-11-19: a unit English drives at > --en-ratio x its strongest box drive is no topic's; and a unit
    # no corpus drives has no profile: both stay in the core (counted)
    en_keep = X[:, B] <= a.en_ratio * X[:, :B].max(1)
    nz = X.sum(1) > 0
    keep_rows = en_keep & nz
    excluded = {"zero_profile": int((~nz).sum()), "english_dominated": int((nz & ~en_keep).sum())}
    pos = pos[keep_rows]
    X = X[keep_rows]
    U = pos.size
    print(f"profiled {U:,} (excluded {excluded})", flush=True)
    if U == 0:
        raise SystemExit("core assign: no movable core unit has a topic profile")
    H = hell(X)
    rng = np.random.default_rng(a.seed)
    samp = rng.choice(U, min(a.sample, U), replace=False)
    C, _ = spherical_kmeans(H[samp], a.clusters, a.seed)
    lab = np.empty(U, np.int32)
    for s0 in range(0, U, 500000):
        lab[s0:s0 + 500000] = np.argmax(H[s0:s0 + 500000] @ C.T, axis=1)
    cand = [k for k in keys if k != "core" and k != f"box:{eb}" and k.startswith(("box:", "pair:", "group:", "joint:"))]
    sigs, sig_keys = [], []
    for k in cand:                                     # tile signatures from a sample of the tile's own units
        tp = np.flatnonzero(tz[k])
        if tp.size == 0:
            continue
        tp = np.sort(rng.choice(tp, min(a.sig_sample, tp.size), replace=False))
        sig_keys.append((k, tp))
    # signatures need the maps again: one pass per map over every tile sample
    allp = np.concatenate([tp for _, tp in sig_keys]) if sig_keys else np.zeros(0, np.int64)
    XS = np.zeros((allp.size, B + 1), np.float32)
    for j, b in enumerate(names):
        z = np.load(Path(prep["boxes"][b]["map"]) / "map.npz")
        XS[:, j] = np.concatenate([z[f"mean_target_{s}"] for s in range(L)])[allp]
        XS[:, B] += np.concatenate([z[f"mean_baseline_{s}"] for s in range(L)])[allp]
    XS[:, B] /= max(en_n, 1)
    HS = hell(XS)
    o = 0
    for k, tp in sig_keys:
        v = HS[o:o + tp.size].mean(0)
        o += tp.size
        sigs.append(v / max(np.linalg.norm(v), 1e-30))
    if not sigs:
        raise SystemExit("core assign: no candidate tile with units")
    S = np.stack(sigs)
    sk = [k for k, _ in sig_keys]
    cs = C @ S.T                                       # [K, T]
    props, moves = [], {}
    pmass = np.zeros((len(C), B + 1), np.float64)     # H**2 = p: the cluster's mean profile mass per box (chunked)
    for s0 in range(0, U, 500000):
        np.add.at(pmass, lab[s0:s0 + 500000], H[s0:s0 + 500000].astype(np.float64) ** 2)
    pmass /= np.maximum(np.bincount(lab, minlength=len(C)), 1)[:, None]
    for c in range(len(C)):
        size = int((lab == c).sum())
        order = np.argsort(-cs[c])
        if (lab == c).sum() == 0:
            props.append({"cluster": c, "units": 0, "not_proposed": ["empty"]})
            continue
        best, second = order[0], order[1] if len(order) > 1 else order[0]
        mem = tile_members(sk[best], prep)
        mass = float(pmass[c, lineage_cols(mem, names)].sum())
        row = {"cluster": c, "units": size, "best": sk[best], "cos": round(float(cs[c, best]), 4),
               "second": sk[second], "margin": round(float(cs[c, best] - cs[c, second]), 4),
               "member_mass": round(mass, 4),
               "english_mass": round(float(pmass[c, B]), 4),
               "top_boxes": {(names + ["english"])[i]: round(float(pmass[c, i]), 4) for i in np.argsort(-pmass[c])[:6]}}
        why = [w for w, ok in (("size", size >= a.min_units), ("cos", row["cos"] >= a.min_cos),
                               ("margin", row["margin"] >= a.min_margin), ("mass", mass >= a.min_mass)) if not ok]
        if why:
            row["not_proposed"] = why
        else:
            m = lab == c
            ucos = H[m] @ S[best]
            mv = pos[m][ucos >= a.min_cos]
            row["proposed"] = True
            row["units_moved"] = int(mv.size)
            moves.setdefault(sk[best], []).append(mv)
        props.append(row)
    moves = {k: np.sort(np.concatenate(v)) for k, v in moves.items()}
    moved_total = int(sum(v.size for v in moves.values()))
    newcore = core.copy()
    for k, v in moves.items():
        newcore.ravel()[v] = False

    def get(k):
        if k == "core":
            return newcore
        m = tz[k]
        if k in moves:
            m = m.copy()
            m.ravel()[moves[k]] = True
        return m
    cover = np.zeros((L, n), np.int16)
    for k in keys:
        cover += get(k)
    if not (cover == 1).all():
        raise SystemExit("core assign: not an exact cover (internal error)")
    del cover
    (out / "partition").mkdir()
    save_npz(out / "partition" / "tiles.npz", [("layers", np.array(layers))] + [(k, None) for k in keys], get)
    np.savez_compressed(out / "moves.npz", layers=np.array(layers),
                        **{k.replace(":", "__"): v for k, v in moves.items()})
    nrep = json.loads(json.dumps(prep))
    nrep["core"]["units_before_core_assign"] = prep["core"]["units"]
    nrep["core"]["units"] = int(newcore.sum())
    nrep["core"]["share"] = float(newcore.mean())
    for k, v in moves.items():
        if k.startswith("box:") and k[4:] in nrep["boxes"]:
            nrep["boxes"][k[4:]]["units"] = nrep["boxes"][k[4:]].get("units", 0) + int(v.size)
            nrep["boxes"][k[4:]]["core_assign_received"] = int(v.size)
        elif k.startswith("pair:") and k[5:] in nrep["pairs"]:
            nrep["pairs"][k[5:]] += int(v.size)
        elif k.startswith("group:") and k[6:] in ((nrep.get("groups") or {}).get("kept") or {}):
            nrep["groups"]["kept"][k[6:]]["units"] += int(v.size)
        elif k.startswith("joint:") and k[6:] in ((nrep.get("joint") or {}).get("tiles") or {}):
            nrep["joint"]["tiles"][k[6:]]["units"] += int(v.size)
    params = {k: getattr(a, k) for k in ("clusters", "sample", "sig_sample", "min_cos", "min_margin", "min_mass",
                                          "min_units", "seed", "en_ratio")}
    nrep["core_assign"] = {"from": str(pd), "params": params, "movable_units": U, "excluded": excluded,
                           "units_moved": moved_total, "stale_fields": "share / whole_experts / expert_equivalents "
                           "and core.from_* / unassigned are the input partition's (only units counts are updated)",
                           "moves": {k: int(v.size) for k, v in sorted(moves.items(), key=lambda x: -x[1].size)},
                           "rule": "core clusters (box-space mean_target profiles) moved to their best-matching "
                                   "existing tile; see proposals.json"}
    nrep["rule"] = prep.get("rule", "") + "; core assign: proposed core clusters moved into existing tiles"
    (out / "partition" / "partition.json").write_text(json.dumps(nrep, indent=1))
    for f in ("unassigned.npz",):                      # the unclaimed arm's mask stays a subset of the (new) core
        if (pd / f).exists():
            uz = np.load(pd / f)
            np.savez_compressed(out / "partition" / f, layers=uz["layers"], unassigned=uz["unassigned"] & newcore)
    rep = {"schema": "mlx2.amoeba-fn-core-assign.v1", "partition": str(pd), "checkpoint": prep.get("checkpoint"),
           "fingerprint": prep.get("fingerprint"), "params": params, "movable_units": U, "excluded": excluded,
           "units_moved": moved_total,
           "core_share": {"before": prep["core"]["share"], "after": float(newcore.mean())},
           "proposed_clusters": sum(1 for r in props if r.get("proposed")), "clusters": props}
    (out / "proposals.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("movable_units", "units_moved", "core_share", "proposed_clusters")}))


if __name__ == "__main__":
    main()
