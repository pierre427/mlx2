#!/usr/bin/env python3
"""Resumable firing capture: one model, several corpora, per-document neuron statistics (Pierre 2026-11-16: expert maps
for the remaining models from our corpus, "easily stopped and resumed", low priority so daytime work goes first).
Inside a GPU window. Forward only; any model the carve driver's sites cover: dense MLP sites (units = MLP neurons,
contribution |act|) and MoE sites in neuron units (unit = expert * F + neuron, contribution |h| * routing weight).

Per corpus (--corpus NAME=PATH, repeatable), split --split (fit), up to --max-docs documents of >= --tokens tokens
(their first --tokens tokens), seeded order, groups = source x half (seeded halves): per group and site the sum and
sum of squares of the per-document mean contribution, the per-document routed share per expert (MoE), and n. A
target/baseline map is a CPU comparison of two corpora's sums (scripts/amoeba_firing_compare.py), so a baseline is
captured once per model and reused by every target.

Stop / resume: every --ckpt-batches batches the sums and the cursor are written atomically to OUT/NAME/ckpt.npz with
an identity (model fingerprint, corpus sha and filter, the capture arguments, this script's and the tree's revision);
a rerun resumes from it; a mismatched identity moves the checkpoint aside (ckpt.stale-*) and restarts that corpus.
Yield (scripts/amoeba_filler_lib.py): before the model load, at each checkpoint and between corpora, if a --gates glob
names a live pid, the --waiters ledger has a live entry not ours, a --foreign-gb process runs, the hour is outside
--hours, or --yield-cmd exits 0, the checkpoint is written and the process exits 75. A finished corpus has
OUT/NAME/report.json then OUT/NAME/sums.npz (written last: its presence means done) and is skipped on rerun.

Usage: amoeba_firing_capture.py --model M --out DIR --corpus eng=E.jsonl --corpus cmn=C.jsonl [--gates GLOB]
"""
from __future__ import annotations

import argparse
import glob
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
YIELD = 75


def script_rev():
    r = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h", "--", "scripts/amoeba_firing_capture.py"],
                       capture_output=True, text=True).stdout.strip()
    if r:
        return r
    f = ROOT / "CAPTURE_REV"          # a git-less snapshot (the M3 lane): written by amoeba_map_filler_m3_sync.sh
    return f.read_text().strip() if f.exists() else "unknown"


def source_rev(O):
    r = O.git_rev()
    if r and r != "unknown":
        return r
    f = ROOT / "REVISION"
    return f.read_text().strip() if f.exists() else r


def sha12(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


PEND = {}


def others_pending(*_):
    """The filler's yield rule (scripts/amoeba_filler_lib.py): live gates, live waiter-ledger entries, optional
    foreign model processes and an hours window; plus --yield-cmd exiting 0."""
    import amoeba_filler_lib as FL
    out = FL.pending(PEND.get("gates", ()), PEND.get("waiters", ""), PEND.get("foreign_gb", 0.0), PEND.get("hours", ""))
    if PEND.get("cmd") and subprocess.run(["/bin/zsh", "-c", PEND["cmd"]], capture_output=True).returncode == 0:
        out.append("yield-cmd")
    return out


def quant_of(model):
    try:
        c = json.loads((Path(model) / "config.json").read_text())
        return c.get("quantization") or c.get("quantization_config") or "none (bf16/fp16)"
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", action="append", required=True,
                    help="NAME=corpus.jsonl, or NAME=corpus.jsonl?FIELD=VALUE to keep only rows with that field value "
                         "(e.g. one box of a shared subject corpus: ?box=life.cooking)")
    ap.add_argument("--group-field", default="source", help="the row field the per-source groups use (subject boxes: "
                    "register); a value's prefix before the first ':' is used")
    ap.add_argument("--split", default="fit")
    ap.add_argument("--max-docs", type=int, default=1500)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-batches", type=int, default=20)
    ap.add_argument("--ckpt-seconds", type=float, default=150, help="also checkpoint (and check for a yield) this often")
    ap.add_argument("--gates", action="append", default=[], help="gate-file glob (a live pid in one: yield)")
    ap.add_argument("--waiters", default="", help="GPU lock waiter ledger dir (a live entry not ours: yield)")
    ap.add_argument("--foreign-gb", type=float, default=0.0, help="yield to another process over this RSS")
    ap.add_argument("--hours", default="", help="allowed local hours HH-HH (night-only models: 23-07)")
    ap.add_argument("--yield-cmd", default="", help="shell command; exit 0 means another job wants the GPU")
    ap.add_argument("--min-avail-gb", type=float, default=15.0, help="watchdog: minimum available memory")
    ap.add_argument("--swap-growth-gb", type=float, default=2.0, help="watchdog: allowed swap growth")
    ap.add_argument("--waiver-note", default="", help="why the watchdog limits differ from the defaults (recorded)")
    ap.add_argument("--watchdog-gb", type=float, default=92)
    ap.add_argument("--memory-gb", type=float, default=88)
    ap.add_argument("--cache-gb", type=float, default=2)
    ap.add_argument("--assert-ple", action="store_true", help="FN rule: refuse unless the PLE NVMe offload is active")
    a = ap.parse_args()
    PEND.update(gates=a.gates, waiters=a.waiters, foreign_gb=a.foreign_gb, hours=a.hours, cmd=a.yield_cmd)
    import mlx.core as mx
    from research.amoeba.carve import ops as O
    from research.amoeba.carve.host import Host
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)    # noqa: E731
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    corpora = [c.partition("=")[::2] for c in a.corpus]
    for nm, _ in corpora:
        if not nm or "/" in nm:
            raise SystemExit(f"bad corpus name {nm!r}")
    todo = [(n, p) for n, p in corpora if not (out / n / "sums.npz").exists()]
    if not todo:
        log("every corpus already captured")
        return 0
    pend = others_pending()
    if pend:
        log(f"yield before load: {pend}")
        return YIELD
    O.footprint_watchdog(a.watchdog_gb, log, min_avail_gb=a.min_avail_gb, swap_growth_gb=a.swap_growth_gb,
                         watcher=True)   # SIGKILL watcher: the same FN model as the battery (2026-11-19 panic)
    O.setup_mlx(a.memory_gb, a.cache_gb)
    host = Host(a.model, log=log, moe_units="neuron")
    if a.assert_ple:
        tables = getattr(getattr(host, "adapter", None), "_tables", None)
        if not tables or Path(os.environ.get("MLX_QWEN4_PLE_NVME", "")).expanduser().resolve() != \
                Path(a.model).expanduser().resolve() / "ple_rows.bin":
            raise SystemExit("PLE offload is not active (FN rule: assert it)")
    pend = others_pending()
    if pend:                               # a large model can take minutes to load: check again before any work
        log(f"yield after load: {pend}")
        return YIELD
    sites = []
    for s in host.sites:
        cls = "dense" if s.kind == "dense" else "moe"
        if cls == "moe" and not getattr(s, "E", None):
            continue
        sites.append({"layer": s.layer, "class": cls, "kind": s.kind, "n": int(s.n),
                      "E": int(getattr(s, "E", 0) or 0), "F": int(getattr(s, "F", 0) or 0)})
    if not sites:
        raise SystemExit("no dense or MoE sites in this model")
    sidx = {(s["layer"], s["class"]): i for i, s in enumerate(sites)}
    rev = script_rev()
    srev = source_rev(O)
    T = a.tokens
    log(f"{host.checkpoint}: {len(sites)} sites ({sum(s['class'] == 'moe' for s in sites)} MoE), "
        f"{sum(s['n'] for s in sites):,} units; capture rev {rev}")
    for name, path in todo:
        d = out / name
        d.mkdir(exist_ok=True)
        fpath, _, filt = path.partition("?")
        rows = [json.loads(x) for x in Path(fpath).read_text().split("\n") if x.strip()]
        rows = [r for r in rows if r.get("split") == a.split]
        if filt:
            fk, _, fv = filt.partition("=")
            rows = [r for r in rows if str(r.get(fk)) == fv]
        rng = np.random.default_rng(a.seed)
        rng.shuffle(rows)
        docs, skipped = [], 0
        for r in rows:
            if len(docs) >= a.max_docs:
                break
            t = host.encode(r["text"])
            if len(t) < T:
                skipped += 1
                continue
            docs.append((t[:T], str(r.get(a.group_field) or "all").split(":", 1)[0]))
        half = np.random.default_rng(a.seed + 7).permutation(len(docs)) % 2
        srcs = sorted({s for _, s in docs})
        groups = [f"{s}:{h}" for s in srcs for h in (0, 1)]
        members = {g: [docs[j][0] for j in range(len(docs)) if f"{docs[j][1]}:{half[j]}" == g] for g in groups}
        ident = {"fingerprint": host.fingerprint, "corpus_sha256_12": sha12(fpath), "filter": filt or None,
                 "group_field": a.group_field, "split": a.split, "source_rev": srev,
                 "max_docs": a.max_docs, "tokens": T, "batch": a.batch, "seed": a.seed, "capture_rev": rev,
                 "groups": groups, "n_docs": len(docs)}
        G, S = len(groups), len(sites)
        S1 = [[mx.zeros((s["n"],), mx.float32) for s in sites] for _ in range(G)]
        S2 = [[mx.zeros((s["n"],), mx.float32) for s in sites] for _ in range(G)]
        R = [[mx.zeros((max(1, s["E"]),), mx.float32) for s in sites] for _ in range(G)]
        n = [0] * G
        start = (0, 0)
        ck = d / "ckpt.npz"
        if ck.exists():
            z = np.load(ck, allow_pickle=False)
            old = json.loads(str(z["ident"]))
            if old != ident:                   # code or data changed: never mix; restart this corpus (kept for audit)
                stale = d / f"ckpt.stale-{time.strftime('%Y%m%d-%H%M%S')}.npz"
                os.replace(ck, stale)
                log(f"{name}: checkpoint identity differs {[k for k in ident if old.get(k) != ident[k]]}; moved to "
                    f"{stale.name}, restarting this corpus")
        if ck.exists():
            for gi in range(G):
                for si in range(S):
                    S1[gi][si] = mx.array(z[f"S1_{gi}_{si}"])
                    S2[gi][si] = mx.array(z[f"S2_{gi}_{si}"])
                    R[gi][si] = mx.array(z[f"R_{gi}_{si}"])
            n = [int(x) for x in z["n"]]
            start = (int(z["cursor"][0]), int(z["cursor"][1]))
            log(f"{name}: resumed at group {start[0]} batch {start[1]} ({sum(n)} documents done)")
        cur = {}

        def stats(l, kind, p):
            cls = "dense" if kind == "dense" else "moe"
            si = sidx.get((l, cls))
            if si is None:
                return
            gi = cur["g"]
            if cls == "dense":
                act = p["act"].astype(mx.float32)
                m = mx.abs(act).mean(axis=1)                       # [B, n]
                rr = None
            else:
                h = p.get("h")
                if h is None:
                    return
                st = sites[si]
                E, F = st["E"], st["F"]
                inds, w = p["inds"], p["w"]
                B, k = inds.shape[0], inds.shape[-1]
                flat = (mx.arange(B, dtype=mx.int32)[:, None, None] * E + inds.astype(mx.int32)).reshape(-1)
                val = (mx.abs(h.astype(mx.float32)) * w.astype(mx.float32)[..., None]).reshape(-1, F)
                m = mx.zeros((B * E, F), mx.float32).at[flat].add(val).reshape(B, E * F) / T
                rr = mx.zeros((B * E,), mx.float32).at[flat].add(1.0).reshape(B, E).sum(0) / (T * k)
            S1[gi][si] = S1[gi][si] + m.sum(0)
            S2[gi][si] = S2[gi][si] + (m * m).sum(0)
            if rr is not None:
                R[gi][si] = R[gi][si] + rr
            mx.eval(S1[gi][si], S2[gi][si], R[gi][si])

        def save_ckpt(gi, bi):
            tmp = d / "ckpt.tmp.npz"
            arr = {"ident": np.array(json.dumps(ident, sort_keys=True)), "n": np.array(n), "cursor": np.array([gi, bi])}
            for g in range(G):
                for si in range(S):
                    arr[f"S1_{g}_{si}"] = np.array(S1[g][si])
                    arr[f"S2_{g}_{si}"] = np.array(S2[g][si])
                    arr[f"R_{g}_{si}"] = np.array(R[g][si])
            np.savez(tmp, **arr)
            os.replace(tmp, ck)

        host.ctl.reset()
        host.ctl.stats = stats
        host.ctl.stats_fast = True
        t0, ntok, since = time.time(), 0, 0
        last_ck = time.time()
        log(f"{name}: {len(docs)} documents ({skipped} shorter skipped), groups {groups}")
        try:
            for gi in range(start[0], G):
                sel = members[groups[gi]]
                b0 = start[1] if gi == start[0] else 0
                for bi in range(b0, (len(sel) + a.batch - 1) // a.batch):
                    chunk = sel[bi * a.batch:(bi + 1) * a.batch]
                    cur["g"] = gi
                    mx.eval(host.hidden(mx.array(chunk, dtype=mx.int32)))
                    mx.clear_cache()
                    n[gi] += len(chunk)
                    ntok += len(chunk) * T
                    since += 1
                    if since >= a.ckpt_batches or time.time() - last_ck >= a.ckpt_seconds:
                        since = 0
                        last_ck = time.time()
                        save_ckpt(gi, bi + 1)
                        log(f"{name}: checkpoint at group {gi} batch {bi + 1}; {ntok / (time.time() - t0):.0f} tok/s, "
                            f"peak {O.peak_gb()} GB")
                        pend = others_pending()
                        if pend:
                            log(f"{name}: yield to {pend}")
                            return YIELD
        finally:
            host.ctl.reset()
        res = {"schema": "mlx2.amoeba-firing-sums.v1", "checkpoint": host.checkpoint, "fingerprint": host.fingerprint,
               "model": str(a.model), "quant": quant_of(a.model), "mlx2_source_revision": source_rev(O),
               "corpus": str(path), "identity": ident,
               "precision": ("not at served precision (unquantized artifact; core q8 / boxes q4 rule)"
                             if not isinstance(quant_of(a.model), dict) else "artifact precision"),
               "watchdog": {"min_avail_gb": a.min_avail_gb, "swap_growth_gb": a.swap_growth_gb,
                            "waiver_note": a.waiver_note or None},
               "coverage_note": ("gemma4_experts layers: only the experts block is a site; their parallel dense MLP "
                                 "is not captured" if any(x["kind"] == "gemma4_experts" for x in sites) else None), "sites": sites, "groups": groups, "n": n,
               "contribution": {"dense": "|act| (down_proj input), mean over the document's first T tokens",
                                "moe": "|h| * routing weight (0 when not routed), mean over the first T tokens"},
               "skipped_short": skipped, "tokens_this_run": ntok, "wall_s_this_run": round(time.time() - t0, 1),
               "mlx_peak_gb": O.peak_gb()}
        arr = {"n": np.array(n), "groups": np.array(groups)}
        for g in range(G):
            for si in range(S):
                arr[f"S1_{g}_{si}"] = np.array(S1[g][si])
                arr[f"S2_{g}_{si}"] = np.array(S2[g][si])
                arr[f"R_{g}_{si}"] = np.array(R[g][si])
        (d / "report.tmp.json").write_text(json.dumps(res, indent=1, default=str))
        os.replace(d / "report.tmp.json", d / "report.json")
        np.savez(d / "sums.tmp.npz", **arr)
        os.replace(d / "sums.tmp.npz", d / "sums.npz")          # last: sums.npz present = corpus done
        if ck.exists():
            ck.unlink()
        log(f"{name}: done, {sum(n)} documents")
        if others_pending():
            log("yield between corpora")
            return YIELD
    return 0


if __name__ == "__main__":
    sys.exit(main())
