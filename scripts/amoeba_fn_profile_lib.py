"""Shared pieces of the FN neuron-profile capture (scripts/amoeba_fn_profile_capture.py) and its two CPU analyses
(scripts/amoeba_fn_profile_analyse.py, scripts/amoeba_fn_untangle_analyse.py). Pure numpy except ExpertWeights (MLX
on the CPU stream). No GPU work here.

Corpus specs: ``NAME=PATH[?k=v&k=v...]``. Keys: ``group=FIELD`` (default ``source``; a list value uses its first
element, so ``group=topics`` is topics[0]), ``depth=N`` (keep the first N dot components: prog.web.css -> prog.web),
``label=G`` (every row in group G), ``max_docs=N`` (per group); any other key filters rows (``split=fit``,
``box=ml.inference``: str(row[key]) == value).

Second-moment sketch (Tool B): per (layer, expert, group) Y = sum_t x_t (Omega^T x_t), Omega a fixed seeded Gaussian
[D, r] shared by every expert and group. With M = sum_t x_t x_t^T (uncentred), Y = M Omega and C = Omega^T Y; the
Nystrom approximation M ~ Y C^+ Y^T is exact when rank(M) <= r (Tropp et al. 2017, fixed-rank PSD approximation from
streaming data) and otherwise a rank-r approximation of M's top range. The diagonal of M is captured exactly, so the
model used downstream is M ~ Y C^+ Y^T + diag(max(0, diag(M) - diag(Y C^+ Y^T))).
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------------------- corpus specs


def parse_spec(spec: str):
    name, eq, rest = spec.partition("=")
    if not eq or not name or "/" in name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise SystemExit(f"bad corpus spec {spec!r} (NAME=PATH[?k=v&...])")
    path, _, q = rest.partition("?")
    opts = {"group": "source", "depth": None, "label": None, "max_docs": None}
    filt = {}
    for kv in [x for x in q.split("&") if x]:
        k, eq2, v = kv.partition("=")
        if not eq2:
            raise SystemExit(f"bad option {kv!r} in {spec!r}")
        if k in ("group", "label"):
            opts[k] = v
        elif k in ("depth", "max_docs"):
            opts[k] = int(v)
        else:
            filt[k] = v
    return name, os.path.expanduser(path), filt, opts


def row_group(row: dict, opts: dict):
    if opts.get("label"):
        return opts["label"]
    v = row.get(opts.get("group") or "source")
    if isinstance(v, list):
        v = v[0] if v else None
    if v is None or str(v) == "":
        return None
    v = str(v)
    if opts.get("depth"):
        v = ".".join(v.split(".")[: opts["depth"]])
    return v


def row_kept(row: dict, filt: dict) -> bool:
    return all(str(row.get(k)) == v for k, v in filt.items())


def x_group_rules(specs, box_names=(), box_alias_prefix="lang."):
    """[(xname, pattern)] from --x-group specs (``PATTERN``: every matching group its own x group; ``NAME=PATTERN``:
    one pooled x group) plus the partition's box names (a box name that is not itself a group id maps to
    ``<prefix><name>``, e.g. arb -> lang.arb; resolved against the groups present by resolve_x_groups)."""
    rules = []
    for s in specs:
        nm, eq, pat = s.partition("=")
        rules.append((nm, pat) if eq else (None, s))
    return rules, list(box_names), box_alias_prefix


def resolve_x_groups(groups, rules, box_names, prefix, rest=True):
    """Map each profile group to its x group (or None). Returns (xmap {group: xgroup|None}, xorder [xgroups],
    box_alias {box: group id or None})."""
    gset = set(groups)
    xmap = {g: None for g in groups}
    alias = {}
    for b in box_names:
        gid = b if b in gset else (prefix + b if prefix + b in gset else None)
        alias[b] = gid
        if gid is not None:
            xmap[gid] = gid
    for nm, pat in rules:
        for g in groups:
            if xmap[g] is None and fnmatch.fnmatchcase(g, pat):
                xmap[g] = nm or g
    if rest:
        for g in groups:
            if xmap[g] is None:
                xmap[g] = "rest"
    named = sorted({x for x in xmap.values() if x is not None and x != "rest"})
    xorder = named + (["rest"] if "rest" in set(xmap.values()) else [])
    return xmap, xorder, alias


def order_groups(groups, xmap, xorder):
    """Profile groups ordered so every x group is one contiguous run (x groups in xorder, then groups without one)."""
    rank = {x: i for i, x in enumerate(xorder)}
    return sorted(groups, key=lambda g: (rank.get(xmap[g], len(xorder)), g))


# ----------------------------------------------------------------------------------------------------- npz writing


def save_npz_lazy(path, items):
    """Write an uncompressed .npz from [(name, callable -> ndarray)], materialising one array at a time (a 3 GB
    checkpoint never holds a second full copy in host memory). Atomic: written to <path>.tmp then renamed."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for name, fn in items:
            arr = np.asanyarray(fn())
            with zf.open(name + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, arr, allow_pickle=False)
            del arr
    os.replace(tmp, path)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False, default=str))
    os.replace(tmp, path)


def scaled_f16(a, axes):
    """(fp16 a / s, fp32 s) with s = max |a| over ``axes`` (1 where all zero), so a large-range statistic keeps
    fp16 relative precision without overflow."""
    a = np.asarray(a, np.float32)
    s = np.max(np.abs(a), axis=axes, keepdims=True)
    s = np.where(s > 0, s, 1.0).astype(np.float32)
    return (a / s).astype(np.float16), np.squeeze(s, axis=axes)


def unscale(q, s, axes):
    return q.astype(np.float32) * np.expand_dims(np.asarray(s, np.float32), axes)


# ----------------------------------------------------------------------------------------------------- capture output


class Capture:
    """Read side of one capture directory (report.json present = complete)."""

    def __init__(self, d):
        self.d = Path(d)
        if not (self.d / "report.json").exists():
            raise SystemExit(f"{self.d}: no report.json (capture incomplete)")
        self.rep = json.loads((self.d / "report.json").read_text())
        self.groups = [g["id"] for g in self.rep["groups"]]
        self.tokens = np.array([g["tokens"] for g in self.rep["groups"]], np.float64)
        self.layers = [s["layer"] for s in self.rep["sites"]]
        self.E = int(self.rep["sites"][0]["E"])
        self.F = int(self.rep["sites"][0]["F"])
        self.D = int(self.rep["d_model"])
        self.n = self.E * self.F

    def profile(self, stat):
        """Raw [G, L, E*F] fp16 memmap of stat 'mean' (mean |h| * w per token) or 'strong' (fraction of tokens over
        theta), stored divided by a per (group, layer, expert) scale (profile_scale): read through profile_layer."""
        return np.load(self.d / "profiles" / f"{stat}.npy", mmap_mode="r")

    def profile_scale(self, stat):
        return np.load(self.d / "profiles" / f"{stat}_scale.npy", mmap_mode="r")     # [G, L, E] fp32

    def profile_layer(self, stat, li, cols=None):
        """float32 [G, n] (or [G, len(cols)]) of stat at layer index li, unscaled."""
        q = self.profile(stat)[:, li, :]
        s = np.asarray(self.profile_scale(stat)[:, li, :], np.float32)
        if cols is not None:
            return np.asarray(q[:, cols], np.float32) * s[:, np.asarray(cols) // self.F]
        v = np.asarray(q, np.float32).reshape(q.shape[0], self.E, self.F)
        v *= s[:, :, None]
        return v.reshape(q.shape[0], -1)

    def profile_value(self, stat, gi, li, unit):
        return float(self.profile(stat)[gi, li, unit]) * float(self.profile_scale(stat)[gi, li, unit // self.F])

    def routed(self):
        return np.load(self.d / "profiles" / "routed.npy", mmap_mode="r")         # [G, L, E] tokens routed

    def wsum(self):
        return np.load(self.d / "profiles" / "wsum.npy", mmap_mode="r")           # [G, L, E] routing weight sum

    def totals(self):
        return np.load(self.d / "totals.npz")

    def xgroups(self):
        return list(self.rep.get("x_groups") or [])

    def xstats(self, xg):
        i = self.xgroups().index(xg)
        return XStats(self.d / "xstats" / f"{i:03d}.npz", self)


class XStats:
    """Per x group: n [L, E], mean / m2 (diagonal second moment) [L, E, D], sketch Y [L, E, D, r] (Y / n, i.e. the
    normalised second moment times Omega). Arrays are stored as scaled fp16 per (layer, expert)."""

    def __init__(self, path, cap):
        self.z = np.load(path)
        self.name = str(self.z["name"])
        self.n = self.z["n"].astype(np.float64)                                    # [L, E]
        self.r = int(self.z["omega"].shape[1]) if "omega" in self.z.files else 0
        self.omega = self.z["omega"].astype(np.float64) if self.r else None

    def layer(self, li):
        z = self.z
        mu = unscale(z[f"mean_{li}"], z[f"mean_s_{li}"], (1,))
        m2 = unscale(z[f"m2_{li}"], z[f"m2_s_{li}"], (1,))
        y = unscale(z[f"sk_{li}"], z[f"sk_s_{li}"], (1, 2)) if self.r else None
        return mu, m2, y

    def h_moments(self, li):
        """Exact E[h] and E[h^2] [E, F] over the tokens routed to each expert (h = act(gate.x) * (up.x), unweighted):
        the reference the Gaussian model of x is checked against."""
        z = self.z
        return unscale(z[f"h1_{li}"], z[f"h1_s_{li}"], (1,)), unscale(z[f"h2_{li}"], z[f"h2_s_{li}"], (1,))


# ----------------------------------------------------------------------------------------------------- moment model


def nystrom_factor(Y, omega, rcond=1e-6):
    """B [D, r] with M ~ B B^T from the sketch Y = M Omega (M PSD): C = Omega^T Y, B = Y C^{+1/2}."""
    Y = np.asarray(Y, np.float64)
    C = omega.T @ Y
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    keep = w > rcond * max(1e-300, float(np.abs(w).max()) if w.size else 0.0)
    if not keep.any():
        return np.zeros((Y.shape[0], 0))
    return Y @ (V[:, keep] / np.sqrt(w[keep]))


class Moments:
    """Second-moment model of one (layer, expert, group): mean mu [D], uncentred M ~ B B^T + diag(dres)."""

    def __init__(self, mu, m2diag, B=None):
        self.mu = np.asarray(mu, np.float64)
        self.B = np.zeros((self.mu.size, 0)) if B is None else np.asarray(B, np.float64)
        lr = (self.B * self.B).sum(1)
        self.dres = np.maximum(0.0, np.asarray(m2diag, np.float64) - lr)

    @classmethod
    def pooled(cls, parts):
        """Token-weighted pool of [(n, mu, m2diag, Y)] sharing one Omega -> (n, mu, m2diag, Y)."""
        n = sum(p[0] for p in parts)
        if n <= 0:
            return 0.0, None, None, None
        mu = sum(p[0] * np.asarray(p[1], np.float64) for p in parts) / n
        m2 = sum(p[0] * np.asarray(p[2], np.float64) for p in parts) / n
        Y = None if parts[0][3] is None else sum(p[0] * np.asarray(p[3], np.float64) for p in parts) / n
        return n, mu, m2, Y

    def cov(self, A, Bv=None):
        """Centred covariance A^T Sigma Bv for column-vector sets A [D, a], Bv [D, b]."""
        Bv = A if Bv is None else Bv
        lr = (A.T @ self.B) @ (self.B.T @ Bv)
        dg = (A * self.dres[:, None]).T @ Bv
        m = (A.T @ self.mu)[:, None] * (self.mu @ Bv)[None, :]
        return lr + dg - m

    def mean(self, A):
        return A.T @ self.mu


def silu(z):
    return z / (1.0 + np.exp(-np.clip(z, -60, 60)))


def psd(c):
    c = 0.5 * (c + c.T)
    w, V = np.linalg.eigh(c)
    return (V * np.maximum(w, 0.0)) @ V.T


def gaussian_split(mom: Moments, g, u, mask, z):
    """Split moments of one neuron under one group, Gaussian model of x: pre-activations (p_O, p_S, q_O, q_S) with
    g_S = g * mask, g_O = g - g_S (same for u) are jointly Gaussian with mean V^T mu and covariance V^T Sigma V;
    z [n, 4] standard normal draws (common random numbers). Returns E[a^2], E[(a - a_O)^2], E[(a - a_S)^2],
    E[(a - a_O - a_S)^2], E[a_O^2] with a = silu(g.x)(u.x), a_O = silu(g_O.x)(u_O.x), a_S = silu(g_S.x)(u_S.x)."""
    gS, uS = g * mask, u * mask
    V = np.stack([g - gS, gS, u - uS, uS], axis=1)
    m = mom.mean(V)
    c = psd(mom.cov(V))
    w, Q = np.linalg.eigh(c)
    L = Q * np.sqrt(np.maximum(w, 0.0))
    s = m[None, :] + z @ L.T
    pO, pS, qO, qS = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    a = silu(pO + pS) * (qO + qS)
    aO = silu(pO) * qO
    aS = silu(pS) * qS
    return {"a2": float((a * a).mean()), "keep": float(((a - aO) ** 2).mean()), "copy": float(((a - aS) ** 2).mean()),
            "sum": float(((a - aO - aS) ** 2).mean()), "leak": float((aO * aO).mean()), "ea": float(a.mean()),
            "eabs": float(np.abs(a).mean())}


def concentration(c, ks):
    """Spread of a per-dimension contribution vector c (c_d = g_d * delta_d, sum = g . delta): L1 share of the top-k
    |c_d| for each k, the dims needed for 90% of the L1 mass, the participation ratio (sum|c|)^2 / sum c^2 (the
    effective number of dims; D when uniform, 1 when one dim carries everything) and the signed share of g.delta the
    top-k (by |c|) carry."""
    a = np.abs(c)
    tot = a.sum()
    if tot <= 0:
        return {"l1_share": {k: 0.0 for k in ks}, "signed_share": {k: 0.0 for k in ks}, "dims90": 0, "pr": 0.0,
                "total": 0.0}
    o = np.argsort(-a)
    cs = np.cumsum(a[o])
    s = c.sum()
    sc = np.cumsum(c[o])
    return {"l1_share": {k: float(cs[min(k, a.size) - 1] / tot) for k in ks},
            "signed_share": {k: float(sc[min(k, a.size) - 1] / s) if s != 0 else 0.0 for k in ks},
            "dims90": int(np.searchsorted(cs, 0.9 * tot) + 1), "pr": float(tot * tot / max(1e-300, (c * c).sum())),
            "total": float(s)}


def mean_diff_se(stA, stB):
    """Per-dim standard error of mu_A - mu_B from the captured n, mean and E[x^2] (tokens treated as independent, so
    a lower bound: tokens of one document are correlated)."""
    nA, muA, m2A = stA[0], stA[1], stA[2]
    nB, muB, m2B = stB[0], stB[1], stB[2]
    vA = np.maximum(m2A - muA * muA, 0.0)
    vB = np.maximum(m2B - muB * muB, 0.0)
    return np.sqrt(vA / max(nA, 1.0) + vB / max(nB, 1.0))


def noise_null(w, se, ks, rng, draws=8):
    """The concentration a pure-noise contrast would show: mean top-k L1 share of w * se * z (z standard normal) over
    draws, and the signal-to-noise denominator sqrt(sum (w_d se_d)^2) (diagonal approximation)."""
    ws = w * se
    sh = {k: 0.0 for k in ks}
    for _ in range(draws):
        c = concentration(ws * rng.standard_normal(ws.size), ks)["l1_share"]
        for k in ks:
            sh[k] += c[k] / draws
    return sh, float(np.sqrt((ws * ws).sum()))


# ----------------------------------------------------------------------------------------------------- weights


class ExpertWeights:
    """Dequantized switch-MLP rows of one model directory (safetensors), on the MLX CPU stream: rows(l, e) ->
    (gate [F, D], up [F, D], down [D, F]) float32. Quantization parameters are read off the shapes (bits =
    32 * packed / D, group = D / scale columns), so per-module overrides in config.json need no parsing; a weight
    without scales is a float weight."""

    # the decoder's own layers only: FN also has language_model.mtp.layers.0.mlp.switch_mlp.* (the MTP head), same
    # shapes, which a bare "layers.N" pattern would take for decoder layer 0 (review 2026-11-18)
    PAT = re.compile(r"^(.*\.)?model\.layers\.(\d+)\.mlp\.switch_mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)\.weight$")

    def __init__(self, model_dir, D, F):
        import mlx.core as mx
        self.mx = mx
        self.dir = Path(model_dir)
        self.D, self.F = int(D), int(F)
        idx = self.dir / "model.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
        else:
            wm = {}
            for f in sorted(self.dir.glob("*.safetensors")):
                for k in mx.load(str(f)):
                    wm[k] = f.name
        self.wm = wm
        self.keys = {}
        for k in wm:
            m = self.PAT.match(k)
            if m and ".mtp." not in f".{k}":
                key = (int(m.group(2)), m.group(3))
                if key in self.keys:
                    raise SystemExit(f"two switch-MLP weights for layer {key[0]} {key[1]}: {self.keys[key]}, {k}")
                self.keys[key] = k[: -len(".weight")]
        self._files = {}
        self._cache = (None, None)

    def _get(self, key):
        f = self.wm[key]
        if f not in self._files:
            self._files[f] = self.mx.load(str(self.dir / f))
        return self._files[f][key]

    def _deq(self, base, e, out_dim, in_dim):
        mx = self.mx
        w = self._get(base + ".weight")[e]
        if base + ".scales" not in self.wm:
            a = w.astype(mx.float32)
        else:
            s = self._get(base + ".scales")[e]
            b = self._get(base + ".biases")[e] if base + ".biases" in self.wm else None
            bits = 32 * int(w.shape[-1]) // in_dim
            gs = in_dim // int(s.shape[-1])
            a = mx.dequantize(w, s, b, group_size=gs, bits=bits).astype(mx.float32)
        with mx.stream(mx.cpu):
            out = np.array(a)
        if out.shape != (out_dim, in_dim):
            raise SystemExit(f"{base}: expert {e} weight {out.shape}, expected {(out_dim, in_dim)}")
        return out

    def rows(self, layer, e):
        """Callers go layer by layer: evaluating one expert's slice materialises the layer's whole stacked tensor
        (MLX loads a safetensors array whole), so the loaded files are dropped when the layer changes (FN: ~1.3 GB
        resident per layer instead of the whole switch MLP)."""
        if self._cache[0] == (layer, e):
            return self._cache[1]
        if self._cache[0] is not None and self._cache[0][0] != layer:
            self._files = {}
        D, F = self.D, self.F
        if (layer, "gate_up_proj") in self.keys:
            gu = self._deq(self.keys[(layer, "gate_up_proj")], e, 2 * F, D)
            g, u = gu[:F], gu[F:]             # units._switch_masked: activation(gu[..., half:], gu[..., :half])
        else:
            g = self._deq(self.keys[(layer, "gate_proj")], e, F, D)
            u = self._deq(self.keys[(layer, "up_proj")], e, F, D)
        d = self._deq(self.keys[(layer, "down_proj")], e, D, F)
        self._cache = ((layer, e), (g, u, d))
        return g, u, d


# ----------------------------------------------------------------------------------------------------- partition


def partition_categories(pdir, log=print):
    """Per-unit categories of an amoeba_fn_partition.py run, recomputed from its inputs (the same rule) and checked
    against its partition.json counts and tiles.npz masks. Returns dict: layers, names (boxes, sorted), cat [L, n]
    int8 (CATS), claims [L, n] int64 bitmask over names, partition json."""
    pdir = Path(pdir)
    pj = json.loads((pdir / "partition.json").read_text())
    names = sorted(pj["boxes"])
    if len(names) > 62:
        raise SystemExit("more than 62 boxes: the claims bitmask is int64")
    r = float(pj["ratio"])
    cdir = Path(pj["inputs"]["core_map"])
    crep = json.loads((cdir / "report.json").read_text())
    cz = np.load(cdir / "map.npz")
    S = len(crep["per_site"])
    layers = [s["layer"] for s in crep["per_site"]]
    n = crep["per_site"][0]["units"]
    core_map = np.stack([cz[f"target_{s}"] for s in range(S)])
    t2 = np.load(pj["inputs"]["t2"])["core"] > 0
    t2 = t2[layers]
    if t2.shape[1] != n:
        t2 = np.repeat(t2, n // t2.shape[1], axis=1)
    claims = np.zeros((S, n), np.int64)
    count = np.zeros((S, n), np.int16)
    for i, nm in enumerate(names):
        d = Path(pj["boxes"][nm]["map"])
        z = np.load(d / "map.npz")
        for s in range(S):
            sel = z[f"target_{s}"] & (z[f"mean_baseline_{s}"].astype(np.float64)
                                      <= r * z[f"mean_target_{s}"].astype(np.float64))
            claims[s] |= sel.astype(np.int64) << i
            count[s] += sel
        log(f"partition: box {nm} read")
    eng = core_map & ~t2
    cat = np.full((S, n), CATS["unassigned"], np.int8)
    cat[count == 1] = CATS["box"]
    cat[count == 2] = CATS["pair"]
    cat[count >= 3] = CATS["shared3"]
    cat[eng] = CATS["eng"]
    cat[t2] = CATS["t2"]
    core = pj["core"]
    got = {"from_t2": int((cat == CATS["t2"]).sum()), "from_english_map_outside_t2": int((cat == CATS["eng"]).sum()),
           "from_3plus_boxes": int((cat == CATS["shared3"]).sum()), "unassigned": int((cat == CATS["unassigned"]).sum())}
    bad = {k: (v, core.get(k)) for k, v in got.items() if core.get(k) != v}
    if core.get("small_pairs_spilled"):
        bad["small_pairs_spilled"] = core["small_pairs_spilled"]
    tiles = np.load(pdir / "tiles.npz")
    for i, nm in enumerate(names):
        want = (cat == CATS["box"]) & ((claims >> i) & 1).astype(bool)
        if not np.array_equal(tiles[f"box:{nm}"], want):
            bad[f"box:{nm}"] = "tile differs from the recomputed rule"
    if int((cat == CATS["pair"]).sum()) != sum(pj["pairs"].values()):
        bad["pairs"] = (int((cat == CATS["pair"]).sum()), sum(pj["pairs"].values()))
    if bad:
        raise SystemExit(f"partition categories do not reproduce {pdir}: {bad}")
    return {"layers": layers, "names": names, "cat": cat, "claims": claims, "partition": pj, "n": n}


CATS = {"t2": 0, "eng": 1, "shared3": 2, "pair": 3, "box": 4, "unassigned": 5}


def members(mask_bits, names):
    return [names[i] for i in range(len(names)) if (int(mask_bits) >> i) & 1]
