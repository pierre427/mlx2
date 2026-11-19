#!/usr/bin/env python3
"""FN neuron topic profiles + routed-input moments in one forward pass (Pierre 2026-11-18; aMoEba). Inside a GPU window.

Tool A (profiles): per SOURCE GROUP g and MoE neuron unit (expert e, neuron f; unit id e * F + f, as the firing maps),
with the firing capture's activation definition (scripts/amoeba_firing_capture.py: contribution c = |h| * routing
weight, 0 when the token is not routed to e; h = act(gate . x) * (up . x)):
  mean[g]    = sum over the group's tokens of c / tokens(g)            (the firing maps' per-token mean)
  strong[g]  = #tokens with c > theta_layer / tokens(g)                (fraction of tokens strongly active)
  routed[g]  = tokens routed to each expert, wsum[g] = their routing-weight sum   ([L, E], fp32)
theta_layer is the --strong-q quantile of the routed contributions of that layer on a seeded calibration sample
(written to thresholds.npz; pass --thresholds to reuse one run's thresholds so later corpora compare).
Exact fp32 totals over every group (sum c, #strong, tokens) go to totals.npz (dead / rare neurons).

Tool B (x moments, Pierre's "un-entangle the dims of shared neurons"): per X GROUP and (layer, expert), over the tokens
routed to that expert: n, the mean of the expert input x (the MoE block input), the diagonal second moment E[x_d^2]
and a rank-r sketch Y = E[x (Omega^T x)] of the second moment (scripts/amoeba_fn_profile_lib.py: Nystrom model), plus
per neuron the exact E[h] and E[h^2] over those tokens (the check of the Gaussian model of x the analysis uses).
X groups are the partition's box groups (--x-boxes; a language box name maps to lang.<name>), --x-group patterns
(NAME=PATTERN pools the matching groups into one), and "rest" (every other group, unless --no-x-rest).

Memory layout (why not 128 groups at once): documents are processed group by group (every x group is one contiguous run
of profile groups), so only ONE profile group's fp32 accumulators (2 x 63 MB on FN) and ONE x group's (mean + m2 +
r sketch columns: (2 + r) x 252 MB, + 2 x 63 MB h moments; r = 8: 2.6 GB) live at a time, whatever the group count.
A finished profile group is written as fp16 rows of profiles/{mean,strong}.npy ([G, L, E*F], divided by
{mean,strong}_scale.npy [G, L, E]) and a finished x group as xstats/NNN.npz (fp16 scaled per (layer, expert)).
Extra memory ~ 0.25 GB (+ exact fp32 totals 0.13 GB) + 2.6 GB + per-layer transients (~0.7 GB at 4096 tokens).

Documents: each corpus spec NAME=PATH?group=FIELD&depth=N&label=G&max_docs=N&<filter>=V (scripts/amoeba_fn_profile_lib
.parse_spec); the first --tokens tokens of each row with >= --min-tokens tokens, right-padded to --tokens (pad positions
are masked out of every statistic; the forward is causal so they never touch real positions).
Stop / resume / yield as scripts/amoeba_firing_capture.py: every --ckpt-seconds the current accumulators, the totals and
the cursor go to ckpt.npz (atomic) under an identity (model fingerprint, corpus sha, groups, arguments, thresholds,
this script's and the tree's revision); a rerun with the same identity resumes, any other identity refuses (use a new
--out). At each checkpoint (and before / after the load) a live --gates pid, a --waiters entry, a --foreign-gb process,
the --hours window or --yield-cmd exit 0 writes the checkpoint and exits 75. report.json is written last (done).

Usage: amoeba_fn_profile_capture.py --model M --out DIR --corpus atlas=C.jsonl?group=topics&depth=2 \
         --corpus sess=P.jsonl?group=source --x-boxes PARTITION/partition.json --x-group sess=sess.* --assert-ple
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import amoeba_fn_profile_lib as PL  # noqa: E402

YIELD = 75
SCHEMA = "mlx2.amoeba-fn-profiles.v1"


def script_rev():
    r = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h", "--", "scripts/amoeba_fn_profile_capture.py",
                        "scripts/amoeba_fn_profile_lib.py"], capture_output=True, text=True).stdout.strip()
    return r or "unknown"


def tree_rev():
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def sha12(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


PEND = {}


def others_pending(*_):
    import amoeba_filler_lib as FL
    out = FL.pending(PEND.get("gates", ()), PEND.get("waiters", ""), PEND.get("foreign_gb", 0.0), PEND.get("hours", ""))
    if PEND.get("cmd") and subprocess.run(["/bin/zsh", "-c", PEND["cmd"]], capture_output=True).returncode == 0:
        out.append("yield-cmd")
    return out


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", action="append", required=True, help="NAME=PATH[?group=F&depth=N&label=G&max_docs=N&k=v]")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--max-docs-per-group", type=int, default=0, help="0: every document")
    ap.add_argument("--max-groups", type=int, default=512, help="refuse more profile groups (disk: G x 63 MB on FN)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strong-q", type=float, default=0.99, help="theta = this quantile of a layer's routed contributions")
    ap.add_argument("--calib-docs", type=int, default=64)
    ap.add_argument("--calib-samples", type=int, default=200_000, help="contribution samples per layer per batch")
    ap.add_argument("--thresholds", default="", help="reuse a thresholds.npz (later corpora compare with this run)")
    ap.add_argument("--x-boxes", default="", help="partition.json: its box names are x groups (arb -> lang.arb)")
    ap.add_argument("--x-alias-prefix", default="lang.")
    ap.add_argument("--x-group", action="append", default=[], help="PATTERN (each match its own x group) or NAME=PATTERN")
    ap.add_argument("--no-x-rest", action="store_true", help="no 'rest' x group (other groups get no x moments)")
    ap.add_argument("--sketch-rank", type=int, default=8, help="second-moment sketch columns (0: mean + diagonal only)")
    ap.add_argument("--ckpt-seconds", type=float, default=600)
    ap.add_argument("--gates", action="append", default=[])
    ap.add_argument("--waiters", default="")
    ap.add_argument("--foreign-gb", type=float, default=0.0)
    ap.add_argument("--hours", default="")
    ap.add_argument("--yield-cmd", default="")
    ap.add_argument("--min-avail-gb", type=float, default=15.0)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--waiver-note", default="")
    ap.add_argument("--watchdog-gb", type=float, default=92)
    ap.add_argument("--memory-gb", type=float, default=88)
    ap.add_argument("--cache-gb", type=float, default=2)
    ap.add_argument("--assert-ple", action="store_true", help="FN rule: refuse unless the PLE NVMe offload is active")
    ap.add_argument("--no-watchdog", action="store_true", help="tests only")
    return ap.parse_args(argv)


def plan_documents(a, host, log):
    """[(group, tokens)] per corpus spec; group -> [token lists]; group -> corpora; skipped counts."""
    T = a.tokens
    per_group, corpora_of, skipped, specs = {}, {}, {}, []
    for spec in a.corpus:
        name, path, filt, opts = PL.parse_spec(spec)
        if not Path(path).is_file():
            raise SystemExit(f"{name}: no file {path}")
        rows = [json.loads(x) for x in Path(path).read_text().split("\n") if x.strip()]
        n_rows = len(rows)
        rows = [r for r in rows if PL.row_kept(r, filt)]
        by = {}
        for r in rows:
            g = PL.row_group(r, opts)
            if g is not None and r.get("text"):
                by.setdefault(g, []).append(r["text"])
        cap = min([c for c in (opts.get("max_docs"), a.max_docs_per_group or None) if c] or [0])
        sk = 0
        for g in sorted(by):
            texts = by[g]
            rng = np.random.default_rng([a.seed, int(hashlib.sha256(f"{name}/{g}".encode()).hexdigest()[:8], 16)])
            rng.shuffle(texts)
            took = 0
            for t in texts:
                if cap and took >= cap:          # the cap is per (corpus, group)
                    break
                ids = host.encode(t[: T * 32])[:T]
                if len(ids) < a.min_tokens:
                    sk += 1
                    continue
                per_group.setdefault(g, []).append(np.asarray(ids, np.int32))
                took += 1
                if name not in corpora_of.setdefault(g, []):
                    corpora_of[g].append(name)
        skipped[name] = sk
        specs.append({"name": name, "path": path, "sha256_12": sha12(path), "filter": filt, "opts": opts,
                      "rows": n_rows, "kept_rows": len(rows), "groups": len(by)})
        log(f"corpus {name}: {len(rows)}/{n_rows} rows kept, {len(by)} groups, {sk} under {a.min_tokens} tokens")
    return per_group, corpora_of, skipped, specs


def main(argv=None, host=None):
    a = build_args(argv)
    PEND.update(gates=a.gates, waiters=a.waiters, foreign_gb=a.foreign_gb, hours=a.hours, cmd=a.yield_cmd)
    import mlx.core as mx
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)    # noqa: E731
    out = Path(a.out)
    if (out / "report.json").exists():
        log("already complete")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    pend = others_pending()
    if pend:
        log(f"yield before load: {pend}")
        return YIELD
    if host is None:
        from research.amoeba.carve import ops as O
        if not a.no_watchdog:
            O.footprint_watchdog(a.watchdog_gb, log, min_avail_gb=a.min_avail_gb, swap_growth_gb=a.swap_growth_gb)
        O.setup_mlx(a.memory_gb, a.cache_gb)
        from research.amoeba.carve.host import Host
        host = Host(a.model, log=log, moe_units="neuron")
        if a.assert_ple:
            tables = getattr(getattr(host, "adapter", None), "_tables", None)
            if not tables or Path(os.environ.get("MLX_QWEN4_PLE_NVME", "")).expanduser().resolve() != \
                    Path(a.model).expanduser().resolve() / "ple_rows.bin":
                raise SystemExit("PLE offload is not active (FN rule: assert it)")
        peak = O.peak_gb
        pend = others_pending()
        if pend:
            log(f"yield after load: {pend}")
            return YIELD
    else:
        peak = lambda: round(mx.get_peak_memory() / 2 ** 30, 2)    # noqa: E731
    sites, skipped_sites = [], []
    for s in host.sites:
        if s.kind not in (None, "dense") and getattr(s, "units", "") == "neuron" and s.E:
            sites.append({"layer": s.layer, "kind": s.kind, "E": int(s.E), "F": int(s.F), "n": int(s.n)})
        elif s.kind is not None:
            skipped_sites.append({"layer": s.layer, "kind": s.kind})
    if not sites:
        raise SystemExit("no MoE neuron sites")
    if len({(s["E"], s["F"]) for s in sites}) != 1:
        raise SystemExit("uneven MoE site shapes")
    E, F = sites[0]["E"], sites[0]["F"]
    D = int(host.d)
    S = len(sites)
    sidx = {s["layer"]: i for i, s in enumerate(sites)}
    T, r = a.tokens, a.sketch_rank

    per_group, corpora_of, skipped, specs = plan_documents(a, host, log)
    if not per_group:
        raise SystemExit("no documents")
    if len(per_group) > a.max_groups:
        raise SystemExit(f"{len(per_group)} profile groups > --max-groups {a.max_groups}")
    box_names = []
    if a.x_boxes:
        box_names = sorted(json.loads(Path(a.x_boxes).read_text())["boxes"])
    rules, _, _ = PL.x_group_rules(a.x_group)
    xmap, xorder, alias = PL.resolve_x_groups(sorted(per_group), rules, box_names, a.x_alias_prefix,
                                              rest=not a.no_x_rest)
    missing = [b for b, g in alias.items() if g is None]
    if missing:
        log(f"x boxes without a group in these corpora (no x moments): {missing}")
    groups = PL.order_groups(sorted(per_group), xmap, xorder)
    G = len(groups)
    xgi = {x: i for i, x in enumerate(xorder)}
    pad = getattr(host.tok, "pad_token_id", None)
    pad = int(pad if pad is not None else (getattr(host.tok, "eos_token_id", None) or 0))

    # ---------------------------------------------------------------- thresholds (calibration on a seeded sample)
    thr_path = out / "thresholds.npz"
    if a.thresholds:
        z = np.load(a.thresholds)
        theta = z["theta"].astype(np.float32)
        if list(z["layers"]) != [s["layer"] for s in sites] or str(z["fingerprint"]) != host.fingerprint:
            raise SystemExit(f"{a.thresholds}: thresholds of another model or layout")
        if not thr_path.exists():
            PL.save_npz_lazy(thr_path, [(k, (lambda k=k: z[k])) for k in z.files])
    elif thr_path.exists():
        theta = np.load(thr_path)["theta"].astype(np.float32)
    else:
        rng = np.random.default_rng(a.seed + 11)
        alld = [d for g in groups for d in per_group[g]]
        pick = rng.choice(len(alld), size=min(a.calib_docs, len(alld)), replace=False)
        cal = [alld[i] for i in sorted(pick)]
        samples = [[] for _ in range(S)]
        cur = {}

        def calib(l, kind, p):
            si = sidx.get(l)
            if si is None or p.get("h") is None:
                return
            h, w = p["h"], p["w"]
            v = cur["valid"]
            c = (mx.abs(h.astype(mx.float32)) * w.astype(mx.float32)[..., None]).reshape(-1)
            vk = mx.broadcast_to(v[..., None, None], h.shape).reshape(-1)
            m = int(c.size)
            ix = mx.random.randint(0, m, (min(a.calib_samples, m),), key=mx.random.key(len(samples[si])))
            cv, vv = np.array(c[ix]), np.array(vk[ix])
            samples[si].append(cv[vv > 0])

        host.ctl.reset()
        host.ctl.stats = calib
        host.ctl.stats_fast = True
        try:
            for b0 in range(0, len(cal), a.batch):
                chunk = cal[b0:b0 + a.batch]
                ids = np.full((len(chunk), T), pad, np.int32)
                val = np.zeros((len(chunk), T), np.float32)
                for i, t in enumerate(chunk):
                    ids[i, :len(t)] = t
                    val[i, :len(t)] = 1
                cur["valid"] = mx.array(val)
                mx.eval(host.hidden(mx.array(ids)))
                mx.clear_cache()
        finally:
            host.ctl.reset()
        allv = [np.concatenate(x) if x else np.zeros(1, np.float32) for x in samples]
        theta = np.array([np.quantile(v, a.strong_q) for v in allv], np.float32)
        PL.save_npz_lazy(thr_path, [
            ("theta", lambda: theta), ("layers", lambda: np.array([s["layer"] for s in sites])),
            ("q", lambda: np.array(a.strong_q)), ("fingerprint", lambda: np.array(host.fingerprint)),
            ("calib_docs", lambda: np.array(len(cal))), ("samples", lambda: np.array([v.size for v in allv])),
            ("median", lambda: np.array([np.median(v) for v in allv], np.float32))])
        log(f"thresholds (q {a.strong_q}, {len(cal)} docs): layer theta {np.round(theta[:4], 4)}...")
    theta_sha = hashlib.sha256(theta.tobytes()).hexdigest()[:12]

    omega = None
    if r:
        omega = np.random.default_rng(a.seed + 101).standard_normal((D, r)).astype(np.float32)
    om = mx.array(omega) if r else None

    ident = {"fingerprint": host.fingerprint, "corpora": [{k: v for k, v in s.items() if k != "kept_rows"} for s in specs],
             "groups": [[g, len(per_group[g]), xmap[g]] for g in groups], "x_groups": xorder, "tokens": T,
             "min_tokens": a.min_tokens, "max_docs_per_group": a.max_docs_per_group, "batch": a.batch, "seed": a.seed,
             "strong_q": a.strong_q, "theta_sha12": theta_sha, "sketch_rank": r, "capture_rev": script_rev(),
             "source_rev": tree_rev(), "pad": pad, "sites": sites}
    ident_s = json.dumps(ident, sort_keys=True, default=str)
    (out / "profiles").mkdir(exist_ok=True)
    (out / "xstats").mkdir(exist_ok=True)
    plan_p = out / "plan.json"
    if plan_p.exists():
        old = json.loads(plan_p.read_text())
        if json.dumps(old["identity"], sort_keys=True, default=str) != ident_s:
            diff = [k for k in ident if json.dumps(old["identity"].get(k), sort_keys=True, default=str)
                    != json.dumps(ident[k], sort_keys=True, default=str)]
            raise SystemExit(f"{out}: an unfinished capture with another identity ({diff}); use a new --out")
    else:
        PL.write_json(plan_p, {"schema": SCHEMA + ".plan", "identity": ident, "x_box_alias": alias,
                               "corpora_of": corpora_of, "skipped_short": skipped, "skipped_sites": skipped_sites})
    prof = {}
    for stat, dt, shape in (("mean", np.float16, (G, S, E * F)), ("strong", np.float16, (G, S, E * F)),
                            ("mean_scale", np.float32, (G, S, E)), ("strong_scale", np.float32, (G, S, E)),
                            ("routed", np.float32, (G, S, E)), ("wsum", np.float32, (G, S, E))):
        p = out / "profiles" / f"{stat}.npy"
        if not p.exists():
            m = np.lib.format.open_memmap(p.with_suffix(".tmp.npy"), mode="w+", dtype=dt, shape=shape)
            del m
            os.replace(p.with_suffix(".tmp.npy"), p)
        prof[stat] = p

    def zeros(*shape):
        return mx.zeros(shape, mx.float32)

    # accumulators: per site lists (one per layer), current profile group, current x group, totals
    acc = {"A_sum": [zeros(E * F) for _ in range(S)], "A_str": [zeros(E * F) for _ in range(S)],
           "A_rout": [zeros(E) for _ in range(S)], "A_w": [zeros(E) for _ in range(S)],
           "T_sum": [zeros(E * F) for _ in range(S)], "T_str": [zeros(E * F) for _ in range(S)],
           "X_n": [zeros(E) for _ in range(S)], "X_s": [zeros(E, D) for _ in range(S)],
           "X_2": [zeros(E, D) for _ in range(S)], "X_h1": [zeros(E * F) for _ in range(S)],
           "X_h2": [zeros(E * F) for _ in range(S)]}
    if r:
        acc["X_y"] = [zeros(E, D, r) for _ in range(S)]
    scal = {"A_tok": 0.0, "X_tok": 0.0, "T_tok": 0.0}
    tokens_g = np.zeros(G, np.float64)
    gi0, bi0 = 0, 0
    ck = out / "ckpt.npz"
    if ck.exists():
        z = np.load(ck)
        if str(z["ident"]) != ident_s:
            raise SystemExit(f"{ck}: checkpoint identity differs; use a new --out")
        for k in acc:
            acc[k] = [mx.array(z[f"{k}_{si}"]) for si in range(S)]
        for k in scal:
            scal[k] = float(z[k])
        tokens_g = z["tokens_g"].astype(np.float64)
        gi0, bi0 = int(z["cursor"][0]), int(z["cursor"][1])
        log(f"resumed at group {gi0} ({groups[gi0] if gi0 < G else 'end'}) batch {bi0}")

    def save_ckpt(gi, bi):
        items = [("ident", lambda: np.array(ident_s)), ("cursor", lambda: np.array([gi, bi])),
                 ("tokens_g", lambda: tokens_g.copy())]
        items += [(k, (lambda k=k: np.array(scal[k]))) for k in scal]
        for k in acc:
            for si in range(S):
                items.append((f"{k}_{si}", (lambda k=k, si=si: np.array(acc[k][si]))))
        PL.save_npz_lazy(ck, items)

    def flush_profile(gi):
        """fp16 rows divided by a per (group, layer, expert) scale = the expert's max over its neurons: the per-token
        means sit around 3e-5..9e-5 on FN (below fp16's normal range, review 2026-11-18), so unscaled fp16 would
        lose precision and round small ones to zero."""
        tok = max(1.0, scal["A_tok"])
        for stat, key, div in (("mean", "A_sum", tok), ("strong", "A_str", tok)):
            m = np.load(prof[stat], mmap_mode="r+")
            ms = np.load(prof[stat + "_scale"], mmap_mode="r+")
            for si in range(S):
                v = np.array(acc[key][si]) / div
                if not np.isfinite(v).all():          # a nonfinite batch must stop the run, never write NaN / zeros
                    raise SystemExit(f"group {groups[gi]} {stat} is not finite at layer {sites[si]['layer']}")
                q, s = PL.scaled_f16(v.reshape(E, F), (1,))
                m[gi, si] = q.reshape(-1)
                ms[gi, si] = s
            m.flush()
            ms.flush()
            del m, ms
        for stat, key in (("routed", "A_rout"), ("wsum", "A_w")):
            m = np.load(prof[stat], mmap_mode="r+")
            for si in range(S):
                m[gi, si] = np.array(acc[key][si])
            m.flush()
            del m
        tokens_g[gi] = scal["A_tok"]
        for k in ("A_sum", "A_str", "A_rout", "A_w"):
            acc[k] = [mx.zeros_like(x) for x in acc[k]]
        scal["A_tok"] = 0.0

    def flush_x(xg):
        xi = xgi[xg]
        n = [np.array(acc["X_n"][si]) for si in range(S)]

        def norm(key, si):
            v = np.array(acc[key][si])
            if key in ("X_h1", "X_h2"):
                v = v.reshape(E, F)
            d = np.maximum(n[si], 1.0).reshape((-1,) + (1,) * (v.ndim - 1))
            if not np.isfinite(v).all():
                raise SystemExit(f"x group {xg} {key} is not finite at layer {sites[si]['layer']}")
            return v / d
        items = [("name", lambda: np.array(xg)), ("n", lambda: np.stack(n)), ("tokens", lambda: np.array(scal["X_tok"])),
                 ("layers", lambda: np.array([s["layer"] for s in sites]))]
        if r:
            items.append(("omega", lambda: omega))
        cache = {}

        def part(key, si, axes, which):
            ck_ = (key, si)
            if ck_ not in cache:
                cache.clear()
                cache[ck_] = PL.scaled_f16(norm(key, si), axes)
            return cache[ck_][which]
        for si in range(S):
            for key, nm, axes in (("X_s", "mean", (1,)), ("X_2", "m2", (1,)), ("X_h1", "h1", (1,)),
                                  ("X_h2", "h2", (1,))) + ((("X_y", "sk", (1, 2)),) if r else ()):
                items.append((f"{nm}_{si}", (lambda key=key, si=si, axes=axes: part(key, si, axes, 0))))
                items.append((f"{nm}_s_{si}", (lambda key=key, si=si, axes=axes: part(key, si, axes, 1))))
        PL.save_npz_lazy(out / "xstats" / f"{xi:03d}.npz", items)
        for k in [k for k in acc if k.startswith("X_")]:
            acc[k] = [mx.zeros_like(x) for x in acc[k]]
        scal["X_tok"] = 0.0

    cur = {"x": False}

    def stats(l, kind, p):
        si = sidx.get(l)
        if si is None:
            return
        h = p.get("h")
        if h is None:
            return
        inds, w, x = p["inds"], p["w"], p["x"]
        v = cur["valid"]                                                   # [B, T]
        k = inds.shape[-1]
        flat = inds.reshape(-1).astype(mx.int32)
        c = (mx.abs(h.astype(mx.float32)) * (w.astype(mx.float32) * v[..., None])[..., None]).reshape(-1, F)
        vk = mx.broadcast_to(v[..., None], inds.shape).reshape(-1)
        strong = (c > float(theta[si])).astype(mx.float32) * vk[:, None]
        wk = (w.astype(mx.float32) * v[..., None]).reshape(-1)
        s1 = mx.zeros((E, F), mx.float32).at[flat].add(c).reshape(-1)
        s2 = mx.zeros((E, F), mx.float32).at[flat].add(strong).reshape(-1)
        acc["A_sum"][si] = acc["A_sum"][si] + s1
        acc["A_str"][si] = acc["A_str"][si] + s2
        acc["T_sum"][si] = acc["T_sum"][si] + s1
        acc["T_str"][si] = acc["T_str"][si] + s2
        rc = mx.zeros((E,), mx.float32).at[flat].add(vk)
        acc["A_rout"][si] = acc["A_rout"][si] + rc
        acc["A_w"][si] = acc["A_w"][si] + mx.zeros((E,), mx.float32).at[flat].add(wk)
        upd = [acc[q][si] for q in ("A_sum", "A_str", "T_sum", "T_str", "A_rout", "A_w")]
        if cur["x"]:
            N = int(np.prod(inds.shape[:-1]))
            rows = (mx.arange(N, dtype=mx.int32)[:, None] * E + inds.reshape(N, k).astype(mx.int32)).reshape(-1)
            A = mx.zeros((N * E,), mx.float32).at[rows].add(vk).reshape(N, E)     # routed-token indicator (valid)
            xf = x.reshape(N, D).astype(mx.float32)
            At = A.T
            acc["X_n"][si] = acc["X_n"][si] + rc
            acc["X_s"][si] = acc["X_s"][si] + At @ xf
            acc["X_2"][si] = acc["X_2"][si] + At @ (xf * xf)
            hv = h.astype(mx.float32).reshape(-1, F) * vk[:, None]              # exact h moments (model check)
            acc["X_h1"][si] = acc["X_h1"][si] + mx.zeros((E, F), mx.float32).at[flat].add(hv).reshape(-1)
            acc["X_h2"][si] = acc["X_h2"][si] + mx.zeros((E, F), mx.float32).at[flat].add(hv * hv).reshape(-1)
            upd += [acc[q][si] for q in ("X_n", "X_s", "X_2", "X_h1", "X_h2")]
            if r:
                zz = xf @ om                                                     # [N, r]
                P = (xf[:, :, None] * zz[:, None, :]).reshape(N, D * r)
                acc["X_y"][si] = acc["X_y"][si] + (At @ P).reshape(E, D, r)
                upd.append(acc["X_y"][si])
        mx.eval(*upd)

    host.ctl.reset()
    host.ctl.stats = stats
    host.ctl.stats_fast = True
    t0, ntok, last_ck = time.time(), 0, time.time()
    log(f"{host.checkpoint}: {S} MoE neuron sites, E {E}, F {F}, d {D}; {G} profile groups, {len(xorder)} x groups "
        f"(sketch rank {r}); {sum(len(per_group[g]) for g in groups)} documents x {T} tokens; theta {theta_sha}")
    try:
        for gi in range(gi0, G):
            g = groups[gi]
            xg = xmap[g]
            cur["x"] = xg is not None
            docs = per_group[g]
            nb = (len(docs) + a.batch - 1) // a.batch
            for bi in range(bi0 if gi == gi0 else 0, nb):
                chunk = docs[bi * a.batch:(bi + 1) * a.batch]
                ids = np.full((len(chunk), T), pad, np.int32)
                val = np.zeros((len(chunk), T), np.float32)
                for i, t in enumerate(chunk):
                    ids[i, :len(t)] = t
                    val[i, :len(t)] = 1
                cur["valid"] = mx.array(val)
                mx.eval(host.hidden(mx.array(ids)))
                mx.clear_cache()
                nv = float(val.sum())
                scal["A_tok"] += nv
                scal["T_tok"] += nv
                if cur["x"]:
                    scal["X_tok"] += nv
                ntok += len(chunk) * T
                if time.time() - last_ck >= a.ckpt_seconds and bi + 1 < nb:
                    save_ckpt(gi, bi + 1)
                    last_ck = time.time()
                    log(f"checkpoint at group {gi} ({g}) batch {bi + 1}/{nb}; {ntok / (time.time() - t0):.0f} tok/s, "
                        f"peak {peak()} GB")
                    pend = others_pending()
                    if pend:
                        log(f"yield to {pend}")
                        return YIELD
            # Outputs written after a checkpoint are rewritten identically by a resume from it (the checkpoint is a
            # consistent snapshot of every accumulator), so a group end needs no checkpoint of its own unless the
            # interval passed or we yield: at ~2.8 GB a checkpoint per group would write ~1 TB over 363 groups.
            flush_profile(gi)
            if xg is not None and (gi + 1 == G or xmap[groups[gi + 1]] != xg):
                flush_x(xg)
            log(f"group {gi + 1}/{G} {g} done ({len(docs)} docs, x {xg}); {ntok / max(1e-9, time.time() - t0):.0f} tok/s, "
                f"peak {peak()} GB")
            pend = others_pending() if gi + 1 < G else []
            if pend or time.time() - last_ck >= a.ckpt_seconds:
                save_ckpt(gi + 1, 0)
                last_ck = time.time()
            if pend:
                log(f"yield to {pend}")
                return YIELD
    finally:
        host.ctl.reset()
    PL.save_npz_lazy(out / "totals.npz", [("sum", lambda: np.stack([np.array(x) for x in acc["T_sum"]])),
                                          ("strong", lambda: np.stack([np.array(x) for x in acc["T_str"]])),
                                          ("tokens", lambda: np.array(scal["T_tok"])), ("theta", lambda: theta),
                                          ("layers", lambda: np.array([s["layer"] for s in sites]))])
    rep = {"schema": SCHEMA, "checkpoint": host.checkpoint, "fingerprint": host.fingerprint, "model": str(a.model),
           "mlx2_source_revision": tree_rev(), "capture_rev": script_rev(), "identity": ident, "d_model": D,
           "sites": sites, "skipped_sites": skipped_sites,
           "groups": [{"id": g, "docs": len(per_group[g]), "tokens": float(tokens_g[i]), "x_group": xmap[g],
                       "corpora": corpora_of[g]} for i, g in enumerate(groups)],
           "x_groups": xorder, "x_box_alias": alias, "thresholds": {"q": a.strong_q, "sha12": theta_sha,
                                                                   "theta": [float(t) for t in theta]},
           "activation": "c = |h| * routing weight (0 when not routed), h = act(gate.x) * (up.x) "
                         "(scripts/amoeba_firing_capture.py); mean = sum c / valid tokens; strong = #(c > theta) / tokens",
           "x_moments": "over tokens routed to the expert (valid positions): mean x, E[x_d^2], sketch E[x (Omega^T x)], "
                        "and per neuron the exact E[h], E[h^2]",
           "storage": "profiles mean / strong: fp16 / <stat>_scale [G, L, E] (per group, layer, expert max); xstats: "
                      "fp16 / <name>_s per (layer, expert)",
           "comparability": f"same per-token contribution as the firing maps, averaged over each document's first "
                            f"{T} tokens (documents of >= {a.min_tokens} tokens, token-weighted); the maps used fixed "
                            "256-token documents (192 for the inference boxes) and a per-document mean, so absolute "
                            "values are not interchangeable with last night's map means",
           "watchdog": {"min_avail_gb": a.min_avail_gb, "swap_growth_gb": a.swap_growth_gb,
                        "waiver_note": a.waiver_note or None},
           "skipped_short": skipped, "tokens_this_run": ntok, "wall_s_this_run": round(time.time() - t0, 1),
           "mlx_peak_gb": peak()}
    PL.write_json(out / "report.json", rep)
    if ck.exists():
        ck.unlink()
    log(f"done: {G} groups, {len(xorder)} x groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
