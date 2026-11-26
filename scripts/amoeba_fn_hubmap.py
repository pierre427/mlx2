#!/usr/bin/env python3
"""FN hub-unit map from existing firing maps and captures (Pierre 2026-11-20: "use our existing data, so no GPU use ...
map this out on the CPU without causing memory contention with ongoing tests"). CPU only: numpy, never mlx.

Question: which MoE neuron units are heavily used across very different contexts (topics, families, modes,
languages), as opposed to topic-specific units and generic always-on units?

Inputs (read-only): a partition (scripts/amoeba_fn_partition.py output: partition.json lists every box -> map dir,
tiles.npz, unassigned.npz), the maps (scripts/amoeba_firing_compare.py: per site target / every_source / d / q /
mean_target / mean_baseline / routed shares), the captures those maps came from (scripts/amoeba_firing_capture.py
sums.npz: per group = source x half, the sum and sum of squares of the per-document mean contribution), the English
map (--core-map of the partition: English vs the 10-language mix; its mean_target is the English drive), T2, and the
joint-tile partition. A context is one box of the partition except the English box (its baseline is not English).

Stages (each resumable; a finished stage leaves its outputs and is skipped on rerun):
  profiles  read each map once; write per context the Cohen's d row (float16) and a flag row (bit0 target =
            q <= 0.01 & d >= 0.5, bit1 every_source, bit2 selected = target & mean_baseline <= ratio x mean_target),
            the routed shares, the English drive and per-layer rates.
  sources   read each capture once; per context and source, the compare test of that source alone against the
            context's whole baseline (Welch t, pooled-sd d, BH over every unit, q <= 0.01 & d >= --min-d); write a
            source-type bitmask row per context. Also re-runs the pooled test and checks it reproduces the map.
  metrics   per layer, per unit: breadth, family diversity, load, evenness, participation, language and source-type
            breakdowns.
  select    family threshold from an independence null, categories, hub scores (three variants).
  hubrows   per layer again: hub profiles, context x context hub overlap, expert counts.
  tiles     hub overlap with T2, partition core / boxes / pairs / groups / unassigned and joint tiles.
  report    summary.json, hubs.npz, report.md.

Memory: every heavy read goes through an F_NOCACHE descriptor (no page-cache growth), one map / capture / layer at a
time; intermediates are row files written with pwrite. A guard samples vm_stat and swap every few seconds and pauses
before reading more when free pages < --min-free-pages, swap used grows > --max-swap-growth-gb from the start, free
pages dropped by more than --drop-gb since the last sample (a window loading), or --pause-file exists.

Usage: nice -n 19 taskpolicy -b amoeba_fn_hubmap.py --partition DIR --joint-partition DIR --out DIR
       [--map-root DIR] [--stage all] [--breadth-d 1.0]
       Exit code 75 = yielded after a long guard pause; rerun (e.g. in a loop with a sleep) to resume.
"""
from __future__ import annotations

import argparse
import fcntl
import io
import json
import math
import os
import re
import resource
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

FAMILIES = ["sci", "hum", "soc", "agent", "dialog", "prog", "swe", "sec", "math", "cs", "eng", "life", "ml", "lang"]
LANGS = {"arb", "cmn", "deu", "fra", "hin", "jpn", "rus", "spa", "swh", "vie"}
# source type of each capture group prefix (the corpus "source" / "register" field). Unknown sources fail closed.
SOURCE_TYPES = ["forum", "academic", "web", "wiki", "book", "code", "synth", "chat"]
assert len(SOURCE_TYPES) <= 8, "source types are one bit each of a uint8 mask"
SOURCE_RULES = [
    ("forum", r"^(qa|stackexchange|issues|github_pr|prindex|ubuntu_irc)$"),
    ("academic", r"^(abstract|academic|arxiv)$"),
    ("web", r"^(fineweb-edu|fineweb2-.*|web|news|guide|reports|rules|vuln)$"),
    ("wiki", r"^(wikipedia.*|wiki)$"),
    ("book", r"^book$"),
    ("code", r"^(code|rosetta)$"),
    ("synth", r"^(nemotron_agentic|toucan|swe_smith|toolace|glaive|when2call|llama_nemotron|openr1|openthoughts|"
              r"opencodereasoning|numina|human_solutions|trace|tuning)$"),
    ("chat", r"^(hh_rlhf|smoltalk|wildchat|ultrachat|autoif|conifer|ifeval_like|multifaceted|slimorca)$"),
]
F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
LOG = None


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(line + "\n")


def family_of(name):
    if name in LANGS:
        return "lang"
    f = name.split(".", 1)[0]
    if f not in FAMILIES:
        raise SystemExit(f"context {name}: unknown family {f!r}")
    return f


def source_type(src):
    for t, rx in SOURCE_RULES:
        if re.match(rx, src):
            return t
    raise SystemExit(f"source {src!r} has no source type (add it to SOURCE_RULES)")


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30      # bytes on macOS


# ---------------------------------------------------------------------------------------------------------- guard
def vm_free_pages():
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"Pages free:\s+(\d+)", out)
        ps = re.search(r"page size of (\d+) bytes", out)
        return int(m.group(1)), int(ps.group(1)) if ps else 16384
    except Exception:
        return None, 16384


def swap_used_bytes():
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"used = ([\d.]+)([MG])", out)
        return float(m.group(1)) * (2 ** 20 if m.group(2) == "M" else 2 ** 30)
    except Exception:
        return None


class Yield(Exception):
    """A pause outlasted --yield-after-s: exit (code 75) so the working set is released, and rerun later."""


class Guard:
    def __init__(self, a, state):
        self.min_free = a.min_free_pages
        self.swap_growth = a.max_swap_growth_gb * 2 ** 30
        self.drop = a.drop_gb * 2 ** 30
        self.pause_file = Path(a.pause_file).expanduser() if a.pause_file else None
        self.interval = a.guard_interval
        self.recheck_s = a.drop_recheck_s
        self.disabled = a.no_guard
        self.yield_after = a.yield_after_s
        self.state = state
        g = state.setdefault("guard", {"pauses": []})
        now_swap = swap_used_bytes()
        # the swap baseline is the first run's start and survives restarts; only --reset-swap-baseline moves it
        if a.reset_swap_baseline or g.get("swap_start_bytes") is None:
            g["swap_start_bytes"] = now_swap
            g.setdefault("swap_baseline_resets", []).append({"when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                             "bytes": now_swap})
        self.swap0 = g["swap_start_bytes"]
        self.last_t = 0.0
        self.last_free = None

    def reasons(self):
        free, ps = vm_free_pages()
        swap = swap_used_bytes()
        r = []
        if free is None:
            r.append("vm_stat unreadable (fail closed)")
        if swap is None:
            r.append("vm.swapusage unreadable (fail closed)")
        if self.pause_file and self.pause_file.exists():
            r.append(f"pause file {self.pause_file}")
        if free is not None and free < self.min_free:
            r.append(f"free pages {free} < {self.min_free}")
        if swap is not None and self.swap0 is not None and swap - self.swap0 > self.swap_growth:
            r.append(f"swap grew {(swap - self.swap0) / 2 ** 30:.2f} GB")
        if free is not None and self.last_free is not None and (self.last_free - free) * ps > self.drop:
            # a window loading keeps free pages down; allocator churn of another job recovers within seconds
            # (observed 2026-11-20: 15-30 GB swings every 3-5 s from a CPU job). Re-sample before calling it a load.
            time.sleep(self.recheck_s)
            free2, _ = vm_free_pages()
            if free2 is None or (self.last_free - free2) * ps > self.drop:
                r.append(f"free pages fell {(self.last_free - free) * ps / 2 ** 30:.1f} GB since the last sample "
                         f"and stayed down {self.recheck_s:.0f} s later")
            else:
                self.state["guard"]["churn_ignored"] = self.state["guard"].get("churn_ignored", 0) + 1
                free = max(free, free2)
        self.last_free = free
        return r, free, swap

    def check(self, where=""):
        if self.disabled:
            return
        now = time.time()
        if now - self.last_t < self.interval:
            return
        self.last_t = now
        r, free, swap = self.reasons()
        if not r:
            return
        t0 = time.time()
        log(f"guard: pause at {where}: {'; '.join(r)}")
        quiet = 0
        last_note = time.time()
        while True:
            time.sleep(15)
            r2, free, swap = self.reasons()
            if r2 and self.yield_after and time.time() - t0 > self.yield_after:
                self.state["guard"]["pauses"].append({"at": where, "why": r2, "seconds": round(time.time() - t0, 1),
                                                      "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "yielded": True})
                log(f"guard: paused {time.time() - t0:.0f} s at {where} ({'; '.join(r2)}); yielding (exit 75) to "
                    f"release memory")
                raise Yield()
            if r2 and time.time() - last_note > 300:
                log(f"guard: still paused ({time.time() - t0:.0f} s): {'; '.join(r2)}")
                last_note = time.time()
            # after a sharp drop (a window loading) wait for two quiet samples before reading again
            quiet = quiet + 1 if not r2 else 0
            if quiet >= 2:
                break
        dt = time.time() - t0
        self.state["guard"]["pauses"].append({"at": where, "why": r, "seconds": round(dt, 1),
                                              "when": time.strftime("%Y-%m-%dT%H:%M:%S")})
        log(f"guard: resume after {dt:.0f} s (free pages {free})")
        self.last_t = time.time()


# ------------------------------------------------------------------------------------------------------ io helpers
def open_nocache(path, mode="rb"):
    flags = {"rb": os.O_RDONLY, "r+b": os.O_RDWR, "wb": os.O_RDWR | os.O_CREAT}[mode]
    fd = os.open(str(path), flags, 0o644)
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    except OSError:
        pass
    return os.fdopen(fd, mode if mode != "wb" else "r+b", buffering=0)


class Npz:
    """One npz member at a time through an F_NOCACHE descriptor (stored or deflated members)."""

    def __init__(self, path):
        self.f = open_nocache(path)
        self.z = zipfile.ZipFile(self.f)
        self.names = {i.filename[:-4] if i.filename.endswith(".npy") else i.filename for i in self.z.infolist()}

    def __getitem__(self, k):
        with self.z.open(k + ".npy") as m:
            return np.lib.format.read_array(io.BufferedReader(m, 1 << 22), allow_pickle=False)

    def __contains__(self, k):
        return k in self.names

    def close(self):
        self.z.close()
        self.f.close()


class Rows:
    """[C, U] row file of one dtype, written / read by row slices with pwrite / pread (no page cache)."""

    def __init__(self, path, C, U, dtype, create=False):
        self.C, self.U, self.dt = C, U, np.dtype(dtype)
        self.path = Path(path)
        if create and not self.path.exists():
            with open(self.path, "wb") as f:
                f.truncate(C * U * self.dt.itemsize)
        if self.path.stat().st_size != C * U * self.dt.itemsize:
            raise SystemExit(f"{self.path}: {self.path.stat().st_size} bytes, expected {C} x {U} x {self.dt.itemsize}")
        self.f = open_nocache(self.path, "r+b")

    def write_row(self, c, arr):
        arr = np.ascontiguousarray(arr, self.dt)
        assert arr.size == self.U
        os.pwrite(self.f.fileno(), arr.tobytes(), c * self.U * self.dt.itemsize)

    def read_block(self, lo, hi, guard=None):
        out = np.empty((self.C, hi - lo), self.dt)
        n = (hi - lo) * self.dt.itemsize
        for c in range(self.C):
            if guard:
                guard.check(f"read {self.path.name} row {c}")
            b = os.pread(self.f.fileno(), n, (c * self.U + lo) * self.dt.itemsize)
            out[c] = np.frombuffer(b, self.dt)
        return out

    def flush(self):
        os.fsync(self.f.fileno())


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, np.ndarray):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None                                           # strict JSON: no NaN / Infinity
    return o


def save_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(_clean(obj), indent=1, allow_nan=False, default=str))
    os.replace(tmp, path)


def save_npz(path, **arr):
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, **arr)
    os.replace(tmp, path)


def stamp_of(**kw):
    """A stage's identity: its parameters and its upstream stamps (sha256 of the sorted JSON)."""
    import hashlib
    return hashlib.sha256(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:16]


def refuse_stamp(what, have, want):
    if have != want:
        raise SystemExit(f"{what} was computed with other parameters or inputs (stamp {have} != {want}); "
                         f"use a new --out or delete the stage's files")


# ------------------------------------------------------------------------------------------------------ statistics
def bh(p):
    n = p.size
    o = np.argsort(p, kind="stable")
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1.0)
    return out


def erfc_p(t):
    """Two-sided normal p, exactly as scripts/amoeba_firing_compare.py (Numerical Recipes erfcc)."""
    z = np.abs(t) / math.sqrt(2)
    tt = 1.0 / (1.0 + 0.5 * z)
    y = tt * np.exp(-z * z - 1.26551223 + tt * (1.00002368 + tt * (0.37409196 + tt * (0.09678418 + tt * (
        -0.18628806 + tt * (0.27886807 + tt * (-1.13520398 + tt * (1.48851587 + tt * (-0.82215223 + tt * 0.17087277)))))))))
    return np.minimum(1.0, y)


def p_chunked(t, chunk=1 << 20):
    p = np.empty(t.size, np.float64)
    for i in range(0, t.size, chunk):
        p[i:i + chunk] = erfc_p(t[i:i + chunk].astype(np.float64))
    return p


def q_from_t(t):
    """BH q over the whole t vector (compare's test: two-sided normal p on Welch t, BH over every unit)."""
    return bh(p_chunked(t))


# -------------------------------------------------------------------------------------------------------- contexts
def resolve_map(box, info, map_root):
    if map_root:
        p = Path(map_root) / box
        if (p / "map.npz").exists():
            return p
    return Path(info["map"])


def load_contexts(a):
    part = json.loads((Path(a.partition) / "partition.json").read_text())
    eng_box = (part.get("english_box") or {}).get("name")
    names = sorted(b for b in part["boxes"] if b != eng_box)
    ctx = []
    for b in names:
        md = resolve_map(b, part["boxes"][b], a.map_root)
        rep = json.loads((md / "report.json").read_text())
        ctx.append({"name": b, "family": family_of(b), "map": str(md), "map_real": os.path.realpath(md / "map.npz"),
                    "capture": rep["target"]["dir"], "baseline_capture": rep["baseline"]["dir"],
                    "groups": rep["target"]["groups"], "docs": rep["target"]["docs"], "test": rep["test"],
                    "map_target_units": rep["target_units"], "map_target_every_source": rep["target_in_every_source"],
                    "map_by_target_source": rep.get("by_target_source", {}),
                    "checkpoint": rep["checkpoint"], "fingerprint": rep["fingerprint"]})
    fps = {c["fingerprint"] for c in ctx}
    if len(fps) != 1:
        raise SystemExit(f"maps of different models: {fps}")
    core_map = Path(part["inputs"]["core_map"])
    return part, ctx, core_map


def sites_of(map_dir):
    rep = json.loads((Path(map_dir) / "report.json").read_text())
    return [{"layer": s["layer"], "n": s["units"], "class": s["class"]} for s in rep["per_site"]]


# ---------------------------------------------------------------------------------------------------- stage: profiles
def stage_profiles(a, out, ctx, core_map, guard, state):
    done_p = out / "profiles.done.json"
    stamp = stamp_of(stage="profiles", ratio=a.ratio, grid=a.rate_d_grid, english_map=str(core_map),
                     contexts=[(c["name"], c["map_real"]) for c in ctx])
    if done_p.exists():
        res = json.loads(done_p.read_text())
        refuse_stamp("profiles", res.get("stamp"), stamp)
        return res
    sites = sites_of(ctx[0]["map"])
    S = len(sites)
    U = sum(s["n"] for s in sites)
    off = np.cumsum([0] + [s["n"] for s in sites])
    C = len(ctx)
    prog_p = out / "profiles.progress.json"
    prog = json.loads(prog_p.read_text()) if prog_p.exists() else {"done": [], "rates": {}, "stamp": stamp}
    refuse_stamp("profiles progress", prog.get("stamp"), stamp)
    D = Rows(out / "d16.rows", C, U, np.float16, create=True)
    FL = Rows(out / "flags.rows", C, U, np.uint8, create=True)
    routed_p = out / "routed.npz"
    rt = np.zeros((C, S, 512), np.float32)
    rb = np.zeros((C, S, 512), np.float32)
    if routed_p.exists():
        z = np.load(routed_p)
        rt, rb = z["rt"], z["rb"]
    dgrid = [float(x) for x in a.rate_d_grid.split(",")]
    for ci, c in enumerate(ctx):
        if c["name"] in prog["done"]:
            continue
        guard.check(f"profiles {c['name']}")
        t0 = time.time()
        z = Npz(Path(c["map"]) / "map.npz")
        drow = np.empty(U, np.float16)
        frow = np.empty(U, np.uint8)
        rates = {"target": [], "every_source": [], "selected": [], **{f"d>={x}": [] for x in dgrid}}
        for s in range(S):
            guard.check(f"profiles {c['name']} site {s}")
            tm = z[f"target_{s}"]
            ev = z[f"every_source_{s}"]
            d = z[f"d_{s}"]
            mt = z[f"mean_target_{s}"]
            mb = z[f"mean_baseline_{s}"]
            sel = tm & (mb <= a.ratio * mt)
            drow[off[s]:off[s + 1]] = np.clip(np.nan_to_num(d, nan=0.0, posinf=6e4, neginf=-6e4), -6e4, 6e4)
            frow[off[s]:off[s + 1]] = tm.astype(np.uint8) | (ev.astype(np.uint8) << 1) | (sel.astype(np.uint8) << 2)
            n = float(tm.size)
            rates["target"].append(float(tm.sum()) / n)
            rates["every_source"].append(float((tm & ev).sum()) / n)
            rates["selected"].append(float(sel.sum()) / n)
            for x in dgrid:
                rates[f"d>={x}"].append(float((tm & (d >= x)).sum()) / n)
            if f"routed_target_{s}" in z:
                rt[ci, s, :] = z[f"routed_target_{s}"][:512]
                rb[ci, s, :] = z[f"routed_baseline_{s}"][:512]
            del tm, ev, d, mt, mb, sel
        z.close()
        got = int((frow & 1).sum())
        if got != c["map_target_units"]:
            raise SystemExit(f"{c['name']}: map has {got} target units, its report says {c['map_target_units']}")
        D.write_row(ci, drow)
        FL.write_row(ci, frow)
        D.flush()
        FL.flush()
        save_npz(routed_p, rt=rt, rb=rb)
        prog["done"].append(c["name"])
        prog["rates"][c["name"]] = rates
        save_json(prog_p, prog)
        log(f"profiles {ci + 1}/{C} {c['name']}: {got} target units, {time.time() - t0:.1f} s, peak RSS {rss_gb():.2f} GB")
    # English drive and English-specific units from the English map (English vs the 10-language mix)
    guard.check("profiles english map")
    z = Npz(core_map / "map.npz")
    eng = np.empty(U, np.float32)
    engsel = np.empty(U, bool)
    mix = np.empty(U, np.float32)
    for s in range(S):
        mt, mb, tm = z[f"mean_target_{s}"], z[f"mean_baseline_{s}"], z[f"target_{s}"]
        eng[off[s]:off[s + 1]] = mt
        mix[off[s]:off[s + 1]] = mb
        engsel[off[s]:off[s + 1]] = tm & (mb <= a.ratio * mt)
    z.close()
    save_npz(out / "english.npz", english_drive=eng, mix10_drive=mix, english_selected=engsel)
    res = {"C": C, "U": U, "S": S, "site_units": [s["n"] for s in sites], "layers": [s["layer"] for s in sites],
           "contexts": [c["name"] for c in ctx], "rates": prog["rates"], "stamp": stamp, "rev": git_rev(),
           "english_drive_source": f"{core_map}/map.npz mean_target (the English capture the old maps used as "
                                   f"baseline; the new maps' English recapture has the same documents)"}
    save_json(done_p, res)
    return res


# ----------------------------------------------------------------------------------------------------- stage: sources
def stage_sources(a, out, ctx, prof, guard, state):
    done_p = out / "sources.done.json"
    stamp = stamp_of(stage="sources", breadth_d=a.breadth_d, profiles=prof["stamp"], rules=SOURCE_RULES)
    if done_p.exists():
        res = json.loads(done_p.read_text())
        refuse_stamp("sources", res.get("stamp"), stamp)
        return res
    C, U, S = prof["C"], prof["U"], prof["S"]
    off = np.cumsum([0] + prof["site_units"])
    prog_p = out / "sources.progress.json"
    prog = json.loads(prog_p.read_text()) if prog_p.exists() else {"done": [], "per_context": {}, "stamp": stamp}
    refuse_stamp("sources progress", prog.get("stamp"), stamp)
    SM = Rows(out / "srcmask.rows", C, U, np.uint8, create=True)
    order = sorted(range(C), key=lambda i: (ctx[i]["baseline_capture"], ctx[i]["name"]))
    base_cache = {}
    for ci in order:
        c = ctx[ci]
        if c["name"] in prog["done"]:
            continue
        guard.check(f"sources {c['name']}")
        t0 = time.time()
        bdir = c["baseline_capture"]
        if bdir not in base_cache:
            base_cache.clear()
            brep = json.loads((Path(bdir) / "report.json").read_text())
            bn = [int(x) for x in brep["n"]]
            bz = Npz(Path(bdir) / "sums.npz")
            N0 = sum(bn)
            m0 = np.empty(U, np.float64)
            v0 = np.empty(U, np.float64)
            for s in range(S):
                guard.check(f"baseline {Path(bdir).name} site {s}")
                s1 = sum(bz[f"S1_{g}_{s}"].astype(np.float64) for g in range(len(bn)))
                s2 = sum(bz[f"S2_{g}_{s}"].astype(np.float64) for g in range(len(bn)))
                m = s1 / N0
                m0[off[s]:off[s + 1]] = m
                v0[off[s]:off[s + 1]] = np.maximum(s2 / N0 - m * m, 0.0) * N0 / max(1, N0 - 1)
            bz.close()
            base_cache[bdir] = (N0, m0, v0, brep)
        N0, m0, v0, brep = base_cache[bdir]
        crep = json.loads((Path(c["capture"]) / "report.json").read_text())
        if crep["fingerprint"] != brep["fingerprint"]:
            raise SystemExit(f"{c['name']}: capture and baseline of different models")
        groups, n = crep["groups"], [int(x) for x in crep["n"]]
        if groups != c["groups"]:
            raise SystemExit(f"{c['name']}: capture groups {groups} != map report groups {c['groups']}")
        srcs = sorted({g.rsplit(":", 1)[0] for g in groups})
        stype = {s: source_type(s) for s in srcs}
        members = {s: [i for i, g in enumerate(groups) if g.rsplit(":", 1)[0] == s] for s in srcs}
        members["__pooled__"] = list(range(len(groups)))
        tv = {s: np.empty(U, np.float32) for s in members}
        dv = {s: np.empty(U, np.float32) for s in members}
        z = Npz(Path(c["capture"]) / "sums.npz")
        for s in range(S):
            guard.check(f"sources {c['name']} site {s}")
            s1 = [z[f"S1_{g}_{s}"].astype(np.float64) for g in range(len(groups))]
            s2 = [z[f"S2_{g}_{s}"].astype(np.float64) for g in range(len(groups))]
            mb, vb = m0[off[s]:off[s + 1]], v0[off[s]:off[s + 1]]
            for src, gs in members.items():
                N1 = sum(n[g] for g in gs)
                m1 = sum(s1[g] for g in gs) / N1
                v1 = np.maximum(sum(s2[g] for g in gs) / N1 - m1 ** 2, 0.0) * N1 / max(1, N1 - 1)
                se = np.sqrt(v1 / N1 + vb / N0)
                sd = np.sqrt(((N1 - 1) * v1 + (N0 - 1) * vb) / max(1, N1 + N0 - 2))
                with np.errstate(divide="ignore", invalid="ignore"):
                    tv[src][off[s]:off[s + 1]] = np.where(se > 0, (m1 - mb) / se, 0.0)
                    dv[src][off[s]:off[s + 1]] = np.where(sd > 0, (m1 - mb) / sd, 0.0)
            del s1, s2
        z.close()
        guard.check(f"sources {c['name']} bh")
        qd = c["test"]["q"]
        md0 = c["test"]["min_d"]
        tp, dp = tv.pop("__pooled__"), dv.pop("__pooled__")
        npool = int(((q_from_t(tp) <= qd) & (dp >= md0)).sum())
        del tp, dp
        mask = np.zeros(U, np.uint8)
        per_src = {}
        for src in srcs:
            guard.check(f"sources {c['name']} bh {src}")
            qq = q_from_t(tv.pop(src))
            dd = dv.pop(src)
            f = (qq <= qd) & (dd >= a.breadth_d)
            n_map_rule = int(((qq <= qd) & (dd >= md0)).sum())       # compare's by_target_source, as a check
            del qq, dd
            want = c["map_by_target_source"].get(src)
            per_src[src] = {"type": stype[src], "units": int(f.sum()), "docs": sum(n[g] for g in members[src]),
                            "at_map_min_d": n_map_rule, "map_by_target_source": want,
                            "match": want is None or abs(n_map_rule - want) <= max(10, 1e-4 * U)}
            mask |= (f.astype(np.uint8) << SOURCE_TYPES.index(stype[src]))
            del f
        SM.write_row(ci, mask)
        SM.flush()
        prog["done"].append(c["name"])
        prog["per_context"][c["name"]] = {"sources": per_src, "pooled_reproduced": npool,
                                          "map_target_units": c["map_target_units"],
                                          "pooled_match": abs(npool - c["map_target_units"]) <= max(10, 1e-4 * U)
                                          and all(v["match"] for v in per_src.values())}
        save_json(prog_p, prog)
        log(f"sources {len(prog['done'])}/{C} {c['name']}: pooled {npool} vs map {c['map_target_units']}; "
            f"{ {k: v['units'] for k, v in per_src.items()} } {time.time() - t0:.0f} s, peak RSS {rss_gb():.2f} GB")
        del mask
    bad = [k for k, v in prog["per_context"].items() if not v["pooled_match"]]
    res = {"breadth_d": a.breadth_d, "per_context": prog["per_context"], "pooled_mismatch": bad, "stamp": stamp,
           "rev": git_rev()}
    save_json(done_p, res)
    return res


# ----------------------------------------------------------------------------------------------------- stage: metrics
def fam_index(ctx):
    fam = np.array([FAMILIES.index(c["family"]) for c in ctx])
    return fam


def layer_metrics(d, fl, sm, fam, lang_c, eng_drive, breadth_d):
    """Per-unit metrics for one layer: d [C, n] float32, fl / sm [C, n] uint8."""
    C, n = d.shape
    K = len(FAMILIES)
    fb = ((fl & 1) > 0) & (d >= breadth_d)                 # breadth flag: map target and d >= breadth_d
    ev = fb & ((fl & 2) > 0)
    dpos = np.where(fb, d, 0.0).astype(np.float32)
    famflag = np.zeros((K, n), bool)
    famev = np.zeros((K, n), bool)
    fammean = np.zeros((K, n), np.float32)
    forum_bit = 1 << SOURCE_TYPES.index("forum")
    chat_bit = 1 << SOURCE_TYPES.index("chat")
    D_src = {}
    any_src = (sm > 0) & fb
    nonforum = ((sm & ~np.uint8(forum_bit)) > 0) & fb
    nonconv = ((sm & ~np.uint8(forum_bit | chat_bit)) > 0) & fb
    forum_only = fb & ((sm & forum_bit) > 0) & ~((sm & ~np.uint8(forum_bit)) > 0)
    fam_any_src = np.zeros((K, n), bool)
    fam_nonforum = np.zeros((K, n), bool)
    fam_nonconv = np.zeros((K, n), bool)
    fam_type = np.zeros((len(SOURCE_TYPES), K, n), bool)
    for k in range(K):
        cs = np.where(fam == k)[0]
        if cs.size == 0:
            continue
        famflag[k] = fb[cs].any(0)
        famev[k] = ev[cs].any(0)
        fammean[k] = dpos[cs].sum(0) / cs.size
        fam_any_src[k] = any_src[cs].any(0)
        fam_nonforum[k] = nonforum[cs].any(0)
        fam_nonconv[k] = nonconv[cs].any(0)
        for ti in range(len(SOURCE_TYPES)):
            fam_type[ti, k] = (((sm[cs] >> ti) & 1) > 0).__and__(fb[cs]).any(0)
    lam = fammean.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(lam > 0, fammean / lam, 0.0)
        H = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)).sum(0)
        Dn = famflag.sum(0)
        even = np.where(Dn > 1, H / np.log(np.maximum(Dn, 2)), 0.0)
        lraw = dpos.sum(0)
        pr = np.where(lraw > 0, lraw ** 2 / np.maximum((dpos ** 2).sum(0), 1e-30), 0.0)
    for ti, t in enumerate(SOURCE_TYPES):
        D_src[t] = fam_type[ti].sum(0).astype(np.uint8)
    famkey = (famflag.astype(np.uint16) << np.arange(K, dtype=np.uint16)[:, None]).sum(0).astype(np.uint16)
    rank = np.argsort(np.argsort(eng_drive, kind="stable"), kind="stable")
    out = {
        "B": fb.sum(0).astype(np.uint8), "D": Dn.astype(np.uint8), "B_every": ev.sum(0).astype(np.uint8),
        "D_every": famev.sum(0).astype(np.uint8), "n_lang": fb[lang_c].sum(0).astype(np.uint8),
        "L_fam": lam.astype(np.float32), "L_raw": lraw.astype(np.float32), "even": even.astype(np.float32),
        "PR": pr.astype(np.float32), "famkey": famkey,
        "D_srcany": fam_any_src.sum(0).astype(np.uint8), "D_nonforum": fam_nonforum.sum(0).astype(np.uint8),
        "D_nonconv": fam_nonconv.sum(0).astype(np.uint8), "B_forum_only": forum_only.sum(0).astype(np.uint8),
        "B_srcany": any_src.sum(0).astype(np.uint8),
        "D_topic": (famflag & (np.arange(K) != FAMILIES.index("lang"))[:, None]).sum(0).astype(np.uint8),
        "eng_pct": ((rank + 0.5) / n).astype(np.float32), "eng_drive": eng_drive.astype(np.float32),
        "sel_count": ((fl & 4) > 0).sum(0).astype(np.uint8),
        "max_d": np.max(np.where((fl & 1) > 0, d, -np.inf), axis=0).astype(np.float32),
    }
    for t in SOURCE_TYPES:
        out[f"D_type_{t}"] = D_src[t]
    for alt in (0.5, 1.5):                                     # breadth-threshold sensitivity (pooled flags only)
        fa = ((fl & 1) > 0) & (d >= alt)
        out[f"D_at_d{alt}"] = np.stack([fa[fam == k].any(0) if (fam == k).any() else np.zeros(n, bool)
                                        for k in range(K)]).sum(0).astype(np.uint8)
    return out, fb


def stage_metrics(a, out, ctx, prof, src, guard, state):
    md = out / "metrics"
    md.mkdir(exist_ok=True)
    stamp = stamp_of(stage="metrics", breadth_d=a.breadth_d, sources=src["stamp"], families=FAMILIES)
    sp = md / "stamp.json"
    if not sp.exists() or json.loads(sp.read_text()).get("stamp") != stamp:
        for f in md.glob("layer_*.npz"):
            f.unlink()
        save_json(sp, {"stamp": stamp, "rev": git_rev()})
    C, U, S = prof["C"], prof["U"], prof["S"]
    off = np.cumsum([0] + prof["site_units"])
    fam = fam_index(ctx)
    lang_c = np.array([c["family"] == "lang" for c in ctx])
    eng = np.load(out / "english.npz")["english_drive"]
    D = Rows(out / "d16.rows", C, U, np.float16)
    FL = Rows(out / "flags.rows", C, U, np.uint8)
    SM = Rows(out / "srcmask.rows", C, U, np.uint8)
    for s in range(S):
        p = md / f"layer_{s:02d}.npz"
        if p.exists():
            continue
        guard.check(f"metrics site {s}")
        lo, hi = int(off[s]), int(off[s + 1])
        d = D.read_block(lo, hi, guard).astype(np.float32)
        fl = FL.read_block(lo, hi, guard)
        sm = SM.read_block(lo, hi, guard)
        m, _ = layer_metrics(d, fl, sm, fam, lang_c, eng[lo:hi], a.breadth_d)
        save_npz(p, **m)
        del d, fl, sm, m
        log(f"metrics site {s}: peak RSS {rss_gb():.2f} GB")


# ------------------------------------------------------------------------------------------------------ stage: select
def poisson_binomial_tail(ps):
    """P(X >= k) for k = 0..len(ps) of a sum of independent Bernoulli(ps)."""
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return dist[::-1].cumsum()[::-1]


def family_null(prof, ctx, breadth_d, met_rates):
    """Expected count of units with D >= k if each context's breadth flag were independent across contexts at its
    per-layer rate (rates from the metrics pass)."""
    K = len(FAMILIES)
    exp = np.zeros(K + 1)
    fam = fam_index(ctx)
    for s, n in enumerate(prof["site_units"]):
        r = met_rates[:, s]
        pf = [1 - np.prod(1 - r[fam == k]) if (fam == k).any() else 0.0 for k in range(K)]
        exp += n * poisson_binomial_tail(pf)
    return exp


def load_metric(out, S, key):
    return np.concatenate([np.load(out / "metrics" / f"layer_{s:02d}.npz")[key] for s in range(S)])


def ranks01(x):
    r = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(np.float64)
    return (r + 0.5) / max(1, x.size)


def stage_select(a, out, ctx, prof, guard, state):
    p = out / "select.npz"
    pj = out / "select.json"
    stamp = stamp_of(stage="select", metrics=json.loads((out / "metrics" / "stamp.json").read_text())["stamp"],
                     generic_pct=a.generic_pct, min_families=a.min_families, floor=a.min_families_floor,
                     min_breadth=a.min_breadth, null_alpha=a.null_alpha)
    if p.exists() and pj.exists() and json.loads(pj.read_text()).get("stamp") == stamp:
        return json.loads(pj.read_text()), dict(np.load(p))
    S, U = prof["S"], prof["U"]
    K = len(FAMILIES)
    # per-context per-layer breadth-flag rates, for the independence null
    rates = np.array([[prof["rates"][c["name"]][f"d>={a.breadth_d}"][s] for s in range(S)] for c in ctx]) \
        if f"d>={a.breadth_d}" in prof["rates"][ctx[0]["name"]] else None
    if rates is None:
        raise SystemExit(f"--breadth-d {a.breadth_d} is not in the profiles rate grid; rerun profiles with it")
    Dv = load_metric(out, S, "D")
    obs = np.array([(Dv >= k).sum() for k in range(K + 1)], np.float64)
    exp = family_null(prof, ctx, a.breadth_d, rates)
    ratio = np.where(obs > 0, exp / np.maximum(obs, 1), np.inf)
    null_failed = False
    floor_binding = False
    if a.min_families:
        kmin = a.min_families
        why = "given (--min-families)"
    else:
        ok = [k for k in range(1, K + 1) if ratio[k] <= a.null_alpha and k >= a.min_families_floor]
        null_failed = not ok
        if null_failed:
            raise SystemExit(f"no family threshold k >= {a.min_families_floor} passes the independence null at "
                             f"alpha {a.null_alpha}; pass --min-families")
        kmin = ok[0]
        floor_binding = kmin == a.min_families_floor and a.min_families_floor > 1 and \
            ratio[a.min_families_floor - 1] <= a.null_alpha
        why = (f"smallest k >= {a.min_families_floor} with independence-null expected(D>=k) <= {a.null_alpha} x "
               f"observed(D>=k)" + ("; the floor binds (the null already passes below it)" if floor_binding else "")
               + ". The null treats contexts as independent; they share one English baseline and nest, so it is a "
                 "lower bound on chance co-occurrence, not a calibrated test")
    engp = load_metric(out, S, "eng_pct")
    B = load_metric(out, S, "B")
    generic = engp >= a.generic_pct
    hub = (~generic) & (Dv >= kmin) & (B >= max(kmin, a.min_breadth))
    broad_generic = generic & (Dv >= kmin) & (B >= max(kmin, a.min_breadth))
    topic = (~generic) & (B > 0) & ((Dv <= 1) | (B <= 2)) & ~hub
    none = (~generic) & (B == 0)
    inter = ~(generic | hub | topic | none)
    thr_sens = {}
    for alt in (0.5, 1.5):
        Da = load_metric(out, S, f"D_at_d{alt}")
        ha = (~generic) & (Da >= kmin)
        thr_sens[f"d>={alt}"] = {"hubs_D_only": int(ha.sum()),
                                 "jaccard_with_main": float((ha & hub).sum()) / max(1, int((ha | hub).sum()))}
        del Da, ha
    cat = np.zeros(U, np.uint8)              # 0 none, 1 topic, 2 intermediate, 3 hub, 4 generic
    cat[topic] = 1
    cat[inter] = 2
    cat[hub] = 3
    cat[generic] = 4
    del engp, topic, none, inter
    idx = np.where(hub)[0]
    L = load_metric(out, S, "L_fam")[idx].astype(np.float64)
    E = load_metric(out, S, "even")[idx].astype(np.float64)
    Lr = load_metric(out, S, "L_raw")[idx].astype(np.float64)
    PR = load_metric(out, S, "PR")[idx].astype(np.float64)
    Dh = Dv[idx].astype(np.float64)
    scores = {"main": Dh * E * L, "rank": (ranks01(Dh) + ranks01(E) + ranks01(L)) / 3, "raw": Dh * Lr,
              "pr": PR * Lr}
    order = np.argsort(-scores["main"], kind="stable")
    sens = {}
    for N in (100, 1000, 10000):
        tops = {k: set(idx[np.argsort(-v, kind="stable")[:N]].tolist()) for k, v in scores.items()}
        sens[str(N)] = {f"{x}~{y}": len(tops[x] & tops[y]) / max(1, len(tops[x] | tops[y]))
                        for i, x in enumerate(tops) for y in list(tops)[i + 1:]}
    from_spear = {}
    rm = ranks01(scores["main"])
    for k, v in scores.items():
        if k != "main":
            from_spear[k] = float(np.corrcoef(rm, ranks01(v))[0, 1]) if idx.size > 2 else None
    info = {"stamp": stamp, "rev": git_rev(), "breadth_d": a.breadth_d, "min_families": kmin,
            "min_families_rule": why, "floor_binding": floor_binding, "null_failed": null_failed,
            "min_breadth": max(kmin, a.min_breadth), "generic_pct": a.generic_pct,
            "null": {"observed_D_ge_k": obs.tolist(), "expected_D_ge_k": exp.tolist(), "ratio": ratio.tolist()},
            "counts": {"none": int((cat == 0).sum()), "topic_specific": int((cat == 1).sum()),
                       "intermediate": int((cat == 2).sum()), "hub": int((cat == 3).sum()),
                       "generic_always_on": int((cat == 4).sum()), "generic_but_broad": int(broad_generic.sum())},
            "score_sensitivity_jaccard_topN": sens, "score_spearman_vs_main": from_spear,
            "breadth_threshold_sensitivity": thr_sens}
    save_npz(p, cat=cat, hub_idx=idx[order], **{f"score_{k}": v[order].astype(np.float32) for k, v in scores.items()})
    save_json(pj, info)
    log(f"select: D >= {kmin} ({why}); {info['counts']}")
    return info, dict(np.load(p))


# ----------------------------------------------------------------------------------------------------- stage: hubrows
def stage_hubrows(a, out, ctx, prof, sel_info, sel, guard, state):
    p = out / "hubrows.npz"
    stamp = stamp_of(stage="hubrows", select=sel_info["stamp"], profile_top=a.profile_top)
    if p.exists():
        z = dict(np.load(p))
        if str(z.get("stamp")) == stamp:
            return z
    C, U, S = prof["C"], prof["U"], prof["S"]
    off = np.cumsum([0] + prof["site_units"])
    hub_idx = np.sort(sel["hub_idx"])
    cat = sel["cat"]
    D = Rows(out / "d16.rows", C, U, np.float16)
    FL = Rows(out / "flags.rows", C, U, np.uint8)
    ov = np.zeros((C, C), np.int64)
    ctx_hub = np.zeros(C, np.int64)
    exp_counts = np.zeros((S, 512), np.int32)
    exp_units = np.zeros((S, 512), np.int32)
    cat_layer = np.zeros((S, 5), np.int64)
    top_n = min(a.profile_top, hub_idx.size)
    top_set = np.sort(sel["hub_idx"][:top_n])
    prof_bits = np.zeros((top_set.size, C), bool)
    prof_d = np.zeros((top_set.size, C), np.float16)
    pos = 0
    for s in range(S):
        guard.check(f"hubrows site {s}")
        lo, hi = int(off[s]), int(off[s + 1])
        n = hi - lo
        cat_layer[s] = np.bincount(cat[lo:hi], minlength=5)
        h = hub_idx[(hub_idx >= lo) & (hub_idx < hi)] - lo
        F = n // 512 if n % 512 == 0 else None
        if F:
            exp_units[s] = F
            exp_counts[s] = np.bincount(h // F, minlength=512)[:512]
        if h.size == 0:
            continue
        d = D.read_block(lo, hi, guard)[:, h].astype(np.float32)
        fl = FL.read_block(lo, hi, guard)[:, h]
        fb = ((fl & 1) > 0) & (d >= a.breadth_d)
        X = fb.astype(np.float32)
        ov += (X @ X.T).astype(np.int64)
        ctx_hub += fb.sum(1)
        t = top_set[(top_set >= lo) & (top_set < hi)] - lo
        if t.size:
            j = np.searchsorted(h, t)
            prof_bits[pos:pos + t.size] = fb[:, j].T
            prof_d[pos:pos + t.size] = d[:, j].T.astype(np.float16)
            pos += t.size
        del d, fl, fb, X
        log(f"hubrows site {s}: {h.size} hubs, peak RSS {rss_gb():.2f} GB")
    res = {"overlap": ov, "ctx_hub": ctx_hub, "exp_counts": exp_counts, "exp_units": exp_units,
           "cat_layer": cat_layer, "prof_idx": top_set, "prof_bits": prof_bits, "prof_d": prof_d,
           "stamp": np.array(stamp)}
    save_npz(p, **res)
    return res


# ------------------------------------------------------------------------------------------------------- stage: tiles
def stage_tiles(a, out, ctx, prof, sel_info, sel, part, guard, state):
    p = out / "tiles.json"
    stamp = stamp_of(stage="tiles", select=sel_info["stamp"], partition=str(a.partition), joint=str(a.joint_partition))
    if p.exists() and json.loads(p.read_text()).get("stamp") == stamp:
        return json.loads(p.read_text())
    U, S = prof["U"], prof["S"]
    cat = sel["cat"]
    hub = cat == 3
    gen = cat == 4
    nhub = int(hub.sum())
    res = {"hub_units": nhub, "all_units": U, "stamp": stamp}

    def frac(m):
        m = m.reshape(-1)[:U]
        return {"units": int(m.sum()), "hubs_in": int((m & hub).sum()), "generic_in": int((m & gen).sum()),
                "share_of_hubs": float((m & hub).sum()) / max(1, nhub), "share_of_all": float(m.sum()) / U}
    # T2: [L, E] expert units expanded to neurons
    t2 = np.load(part["inputs"]["t2"])["core"] > 0
    rows = []
    for s, n in enumerate(prof["site_units"]):
        F = n // 512
        rows.append(np.repeat(t2[prof["layers"][s]][:512], F))
    res["t2"] = frac(np.concatenate(rows))
    del rows, t2
    pdir = Path(a.partition)
    un = Npz(pdir / part.get("unassigned_file", "unassigned.npz"))
    res["unassigned"] = frac(un["unassigned"].astype(bool))
    un.close()
    lang_boxes = {k for k in (f"box:{b}" for b in part["boxes"]) if family_of(k[4:]) == "lang"} | {"box:lang.eng"}
    z = Npz(pdir / "tiles.npz")
    if [int(x) for x in z["layers"]] != [int(x) for x in prof["layers"]]:
        raise SystemExit("tiles.npz layers differ from the maps' sites")
    agg = {"core": None, "box_topic": np.zeros(U, bool), "box_lang": np.zeros(U, bool), "box_english": None,
           "pair": np.zeros(U, bool), "group": np.zeros(U, bool)}
    per_box = {}
    per_group = {}
    names = sorted(z.names)
    for i, k in enumerate(names):
        if k == "layers":
            continue
        if i % 50 == 0:
            guard.check(f"tiles {k}")
        m = z[k].astype(bool).reshape(-1)[:U]
        if k == "core":
            agg["core"] = m
        elif k == "box:lang.eng":
            agg["box_english"] = m
            agg["box_lang"] |= m
        elif k in lang_boxes:
            agg["box_lang"] |= m
            per_box[k[4:]] = int((m & hub).sum())
        elif k.startswith("box:"):
            agg["box_topic"] |= m
            per_box[k[4:]] = int((m & hub).sum())
        elif k.startswith("pair:"):
            agg["pair"] |= m
        elif k.startswith("group:"):
            agg["group"] |= m
            per_group[k[6:]] = int((m & hub).sum())
    z.close()
    for k, m in agg.items():
        if m is not None:
            res[f"partition_{k}"] = frac(m)
    # the partition's kept 3+-claimed set: core = T2 | English map outside T2 | 3+ kept | unassigned
    if agg["core"] is not None:
        t2m = np.concatenate([np.repeat((np.load(part["inputs"]["t2"])["core"] > 0)[prof["layers"][s]][:512],
                                        n // 512) for s, n in enumerate(prof["site_units"])])
        um = Npz(pdir / part.get("unassigned_file", "unassigned.npz"))
        res["partition_3plus_kept"] = frac(agg["core"] & ~t2m & ~um["unassigned"].astype(bool).reshape(-1)[:U])
        um.close()
        res["partition_3plus_kept"]["note"] = ("core minus T2 minus unassigned; equals the kept 3+ set when the "
                                               "English map adds nothing outside T2 (partition core."
                                               "from_english_map_outside_t2 = "
                                               f"{part.get('core', {}).get('from_english_map_outside_t2')})")
        del t2m
    res["hubs_per_box"] = dict(sorted(per_box.items(), key=lambda x: -x[1]))
    res["hubs_per_group"] = per_group
    del agg
    if a.joint_partition:
        jz = Npz(Path(a.joint_partition) / "tiles.npz")
        jm = np.zeros(U, bool)
        jn = sorted(k for k in jz.names if k.startswith("joint:"))
        per_joint = {}
        for i, k in enumerate(jn):
            if i % 50 == 0:
                guard.check(f"joint tiles {i}/{len(jn)}")
            m = jz[k].astype(bool).reshape(-1)[:U]
            jm |= m
            c = int((m & hub).sum())
            if c:
                per_joint[k[6:]] = c
        jz.close()
        res["joint_tiles"] = frac(jm)
        res["joint_tiles"]["tiles"] = len(jn)
        res["hubs_per_joint_top"] = dict(sorted(per_joint.items(), key=lambda x: -x[1])[:25])
    sc = load_metric(out, S, "sel_count")
    res["raw_selected_3plus_before_detach_and_nesting"] = frac(sc >= 3)
    res["raw_selected_0_before_detach_and_nesting"] = frac(sc == 0)
    eng = np.load(out / "english.npz")["english_selected"]
    res["english_map_selected"] = frac(eng)
    save_json(p, res)
    return res


# ------------------------------------------------------------------------------------------------------ stage: report
def kmeans(X, k, seed=0, iters=60, restarts=4):
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        c = [X[rng.integers(len(X))]]
        for _ in range(1, k):                                   # k-means++
            d2 = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            c.append(X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(len(X))])
        c = np.array(c, np.float64)
        for _ in range(iters):
            dist = (X ** 2).sum(1)[:, None] - 2 * X @ c.T + (c ** 2).sum(1)[None]
            lab = dist.argmin(1)
            nc = np.array([X[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(nc, c):
                break
            c = nc
        inertia = float(dist[np.arange(len(X)), lab].sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab, c)
    return best


def unit_str(u, prof):
    off = np.cumsum([0] + prof["site_units"])
    s = int(np.searchsorted(off, u, side="right") - 1)
    r = u - off[s]
    F = prof["site_units"][s] // 512
    return prof["layers"][s], int(r // F), int(r % F)


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def stage_report(a, out, ctx, prof, src, sel_info, sel, hr, tiles, state, meta):
    S, C = prof["S"], len(ctx)
    K = len(FAMILIES)
    names = [c["name"] for c in ctx]
    fam = fam_index(ctx)
    hub_idx = sel["hub_idx"]
    nh = hub_idx.size
    met = {k: load_metric(out, S, k)[hub_idx] for k in
           ["B", "D", "L_fam", "L_raw", "even", "PR", "n_lang", "D_topic", "D_every", "B_every", "D_srcany",
            "D_nonforum", "D_nonconv", "B_forum_only", "B_srcany", "eng_pct", "famkey", "sel_count", "max_d"]
           + [f"D_type_{t}" for t in SOURCE_TYPES]}
    kmin = sel_info["min_families"]
    # register labels (per hub)
    # families whose contexts have no non-forum source (or nothing outside forum + chat) cannot be tested without
    # those sources; count them separately instead of reading source availability as a property of the unit
    ftypes = {f: set() for f in FAMILIES}
    for c in ctx:
        ftypes[c["family"]] |= {source_type(g.rsplit(":", 1)[0]) for g in c["groups"]}
    forum_only = [f for f in FAMILIES if ftypes[f] and not (ftypes[f] - {"forum"})]
    conv_only = [f for f in FAMILIES if ftypes[f] and not (ftypes[f] - {"forum", "chat"})]
    fo_mask = sum(1 << FAMILIES.index(f) for f in forum_only)
    co_mask = sum(1 << FAMILIES.index(f) for f in conv_only)
    popc = np.vectorize(lambda x: bin(int(x)).count("1"), otypes=[np.int64])
    fk = met["famkey"].astype(np.int64)
    U_nf = popc(fk & fo_mask) if nh else np.zeros(0, np.int64)
    U_nc = popc(fk & co_mask) if nh else np.zeros(0, np.int64)
    concept = met["D_nonforum"] >= kmin
    untestable = ~concept & (met["D_nonforum"].astype(np.int64) + U_nf >= kmin)
    srcany = met["D_srcany"] >= kmin
    reg = np.where(concept, 0, np.where(untestable, 1, np.where(srcany, 2, 3)))
    reg_names = ["concept (D >= kmin without forum sources)",
                 f"untestable (reaches kmin only with forum-only families: {', '.join(forum_only) or 'none'})",
                 "forum register (breadth needs forum/QA sources)", "underpowered (single-source tests too weak)"]
    strict = met["D_nonconv"] >= kmin
    strict_untestable = ~strict & (met["D_nonconv"].astype(np.int64) + U_nc >= kmin)
    best_single = np.max(np.stack([met[f"D_type_{t}"] for t in SOURCE_TYPES]), axis=0)
    single_ok = best_single >= kmin
    # layer bands
    layers = np.array(prof["layers"])
    off = np.cumsum([0] + prof["site_units"])
    site_of = np.searchsorted(off, hub_idx, side="right") - 1
    lay = layers[site_of]
    L = int(layers.max()) + 1
    band = np.where(lay < L / 3, 0, np.where(lay < 2 * L / 3, 1, 2))
    band_names = [f"early (0-{int(L / 3) - 1 if L % 3 == 0 else int(L / 3)})", "middle", "late"]
    # language check
    langspan = (met["n_lang"] >= a.lang_min) & (met["D_topic"] >= a.lang_topic_min)
    nl_all = load_metric(out, S, "n_lang")
    dt_all = load_metric(out, S, "D_topic")
    lang_units = nl_all >= a.lang_min
    per_layer_lang = []
    for s in range(S):
        lo, hi = off[s], off[s + 1]
        lu = lang_units[lo:hi]
        per_layer_lang.append({"layer": int(layers[s]), "lang_units": int(lu.sum()),
                               "also_topic": int((lu & (dt_all[lo:hi] >= a.lang_topic_min)).sum()),
                               "hubs": int(((sel["cat"][lo:hi] == 3)).sum()),
                               "hubs_langspan": int(langspan[site_of == s].sum())})
    del nl_all, dt_all, lang_units
    # experts
    ec, eu = hr["exp_counts"], hr["exp_units"]
    share = np.where(eu > 0, ec / np.maximum(eu, 1), 0.0)
    hub_exp = [(int(layers[s]), int(e), float(share[s, e]), int(ec[s, e])) for s, e in zip(*np.where(share >= 0.5))]
    hub_exp.sort(key=lambda x: -x[2])
    frac_in_hubexp = float(sum(x[3] for x in hub_exp)) / max(1, nh)
    # router breadth per expert: contexts whose routed share is >= 2x the baseline's
    rz = np.load(out / "routed.npz")
    rt, rb = rz["rt"], rz["rb"]
    with np.errstate(divide="ignore", invalid="ignore"):
        up = (rt >= a.route_ratio * rb) & (rt > 0)
    fam_up = np.zeros((K, S, 512), bool)
    for k in range(K):
        if (fam == k).any():
            fam_up[k] = up[fam == k].any(0)
    route_D = fam_up.sum(0)                                     # [S, 512]
    hub_share_e = share
    corr = float(np.corrcoef(route_D.reshape(-1), hub_share_e.reshape(-1))[0, 1]) if nh else None
    corr = corr if corr is not None and np.isfinite(corr) else None
    # clusters on top-N binarized context profiles
    pb = hr["prof_bits"].astype(np.float64)
    clusters = []
    stability = None
    if pb.shape[0] >= a.k:
        # rows scaled to unit length: Euclidean k-means on them is cosine k-means, so clusters follow the pattern of
        # contexts rather than how many contexts a unit fires in
        pn = pb / np.maximum(np.linalg.norm(pb, axis=1, keepdims=True), 1e-12)
        runs = [kmeans(pn, a.k, seed=sd) for sd in range(3)]
        inertia, lab, cent = runs[0]
        rng = np.random.default_rng(0)
        sub = rng.choice(len(pn), size=min(3000, len(pn)), replace=False)

        def rand_index(x, y):
            same_x = x[:, None] == x[None]
            same_y = y[:, None] == y[None]
            iu = np.triu_indices(len(x), 1)
            return float((same_x == same_y)[iu].mean())
        stability = {"rand_index_seed0_vs_seed1": rand_index(runs[0][1][sub], runs[1][1][sub]),
                     "rand_index_seed0_vs_seed2": rand_index(runs[0][1][sub], runs[2][1][sub]),
                     "distance": "cosine (unit-length rows)", "k": a.k, "restarts_per_seed": 4}
        for j in range(a.k):
            m = lab == j
            fr = pb[m].mean(0)
            top = np.argsort(-fr)[:8]
            ff = np.zeros(K)
            for k in range(K):
                if (fam == k).any():
                    ff[k] = fr[fam == k].mean()
            pl = np.array([unit_str(int(u), prof)[0] for u in hr["prof_idx"][m]])
            clusters.append({"cluster": j, "units": int(m.sum()), "mean_contexts": float(pb[m].sum(1).mean()),
                             "top_contexts": [(names[t], round(float(fr[t]), 2)) for t in top],
                             "family_rate": {FAMILIES[k]: round(float(ff[k]), 2) for k in np.argsort(-ff)[:6]},
                             "layer_median": float(np.median(pl)) if pl.size else None,
                             "layer_iqr": [float(np.percentile(pl, 25)), float(np.percentile(pl, 75))] if pl.size else None})
        clusters.sort(key=lambda x: -x["units"])
    # context overlap
    ov = hr["overlap"].astype(np.float64)
    diag = np.diag(ov).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = ov / (diag[:, None] + diag[None] - ov)
    jac[~np.isfinite(jac)] = 0
    pairs = []
    for i in range(C):
        for j in range(i + 1, C):
            if fam[i] != fam[j]:
                pairs.append((jac[i, j], names[i], names[j], int(ov[i, j])))
    pairs.sort(reverse=True)
    tot = max(1, int(nh))
    lift_pairs = []
    for i in range(C):
        for j in range(i + 1, C):
            if fam[i] != fam[j] and diag[i] > 0 and diag[j] > 0:
                lift_pairs.append((ov[i, j] * tot / (diag[i] * diag[j]), names[i], names[j], int(ov[i, j])))
    lift_pairs = [x for x in lift_pairs if x[3] >= 100]
    lift_pairs.sort(reverse=True)
    famov = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            ci, cj = np.where(fam == i)[0], np.where(fam == j)[0]
            if ci.size and cj.size:
                sub = jac[np.ix_(ci, cj)]
                if i == j:
                    sub = sub[~np.eye(ci.size, dtype=bool)] if ci.size > 1 else np.array([np.nan])
                famov[i, j] = np.nanmean(sub)
    # top 50 table
    rows = []
    for r in range(min(50, nh)):
        u = int(hub_idx[r])
        ly, e, nn = unit_str(u, prof)
        fk = int(met["famkey"][r])
        fams = [FAMILIES[k] for k in range(K) if (fk >> k) & 1]
        rows.append([r + 1, f"L{ly} E{e} n{nn}", f"{sel['score_main'][r]:.1f}", int(met["D"][r]), int(met["B"][r]),
                     f"{met['even'][r]:.2f}", f"{met['L_fam'][r]:.2f}", int(met["n_lang"][r]),
                     int(met["D_nonforum"][r]), int(met["D_every"][r]), f"{met['eng_pct'][r]:.2f}",
                     ["concept", "untestable", "forum", "weak"][int(reg[r])], ",".join(fams)])
    counts = sel_info["counts"]
    summ = {
        "schema": "mlx2.amoeba-fn-hubmap.v1", "meta": meta, "contexts": C,
        "families": {f: [c["name"] for c in ctx if c["family"] == f] for f in FAMILIES},
        "selection": sel_info, "sources_pooled_mismatch": src.get("pooled_mismatch"),
        "hubs": int(nh),
        "hub_layers": {"per_layer": np.bincount(lay, minlength=L).tolist(),
                       "bands": {band_names[b]: int((band == b).sum()) for b in range(3)}},
        "hub_experts": {"experts_with_hub_share_ge_0.5": len(hub_exp),
                        "experts_with_hub_share_ge_0.9": int((share >= 0.9).sum()),
                        "share_of_hubs_in_those_experts": frac_in_hubexp, "top": hub_exp[:40],
                        "experts_with_any_hub": int((ec > 0).sum()),
                        "router_breadth_vs_hub_share_corr": corr,
                        "router_breadth_definition": f"families with a context routing the expert >= "
                                                     f"{a.route_ratio}x the English baseline's share"},
        "register": {reg_names[i]: int((reg == i).sum()) for i in range(4)},
        "register_strict": {"D >= kmin without forum and chat sources": int(strict.sum()),
                            f"reaches kmin only with forum/chat-only families ({', '.join(conv_only)})":
                                int(strict_untestable.sum())},
        "register_families": {"forum_only": forum_only, "forum_or_chat_only": conv_only,
                              "source_types": {f: sorted(v) for f, v in ftypes.items()}},
        "register_single_source_type_ok": int(single_ok.sum()),
        "register_best_single_type": {t: int(((met[f"D_type_{t}"] >= kmin)).sum()) for t in SOURCE_TYPES},
        "language": {"hubs_spanning_langs_and_topics": int(langspan.sum()),
                     "rule": f">= {a.lang_min} language contexts and >= {a.lang_topic_min} non-language families",
                     "by_band": {band_names[b]: int((langspan & (band == b)).sum()) for b in range(3)},
                     "per_layer": per_layer_lang},
        "clusters": clusters,
        "top_cross_family_pairs": [{"a": x[1], "b": x[2], "jaccard": round(float(x[0]), 3), "hubs": x[3]}
                                   for x in pairs[:30]],
        "top_cross_family_pairs_by_lift": [{"a": x[1], "b": x[2], "lift": round(float(x[0]), 2), "hubs": x[3]}
                                           for x in lift_pairs[:30]],
        "cluster_stability": stability,
        "family_overlap_mean_jaccard": {FAMILIES[i]: {FAMILIES[j]: (round(float(famov[i, j]), 3)
                                                                    if np.isfinite(famov[i, j]) else None)
                                                         for j in range(K)} for i in range(K)},
        "hubs_per_context": {names[i]: int(hr["ctx_hub"][i]) for i in range(C)},
        "tiles": tiles, "guard": state.get("guard"), "peak_rss_gb": rss_gb(),
    }
    save_json(out / "summary.json", summ)
    save_npz(out / "hubs.npz", unit=hub_idx, score_main=sel["score_main"], score_rank=sel["score_rank"],
             score_raw=sel["score_raw"], score_pr=sel["score_pr"], register=reg.astype(np.uint8),
             layer=lay.astype(np.int16), **{k: v for k, v in met.items()},
             profile_unit=hr["prof_idx"], profile_bits=hr["prof_bits"], profile_d=hr["prof_d"],
             contexts=np.array(names), families=np.array(FAMILIES))
    # ---- report.md
    R = []
    R.append(f"# FN hub units (CPU, existing maps)\n\nModel `{meta['checkpoint']}` (fingerprint "
             f"`{meta['fingerprint'][:16]}`); report from mlx2 `{meta['mlx2_rev']}` (stage revisions: profiles "
             f"`{prof.get('rev')}`, sources `{src.get('rev')}`, select `{sel_info.get('rev')}`); "
             f"partition `{meta['partition']}`; {C} contexts (English box excluded). Firing statistics only, "
             f"no causal claim.\n")
    R.append("## Method\n")
    tests = sorted({(c["test"]["q"], c["test"]["min_d"]) for c in ctx})
    bases = sorted({Path(c["baseline_capture"]).parent.name + "/" + Path(c["baseline_capture"]).name for c in ctx})
    R.append(f"- Profile per unit: the map's Cohen's d against its English baseline capture ({', '.join(bases)}), "
             f"per context; d is (target mean - English mean) / pooled per-unit sd.\n"
             f"- A context counts for a unit (breadth flag) when the map's own target test holds (BH over every "
             f"unit; (q, min d) = {tests}) and d >= {a.breadth_d}.\n"
             f"- Diversity D = number of the 14 families ({', '.join(FAMILIES)}) with a flagged context; "
             f"subtopics merge into their family, the 10 language boxes form `lang`.\n"
             f"- Load L = sum over families of the family's mean flagged d (so families with many subtopics do not "
             f"dominate); evenness = Pielou entropy of those family loads; PR = participation ratio of flagged d "
             f"over contexts.\n"
             f"- Generic always-on = English drive at or above the {a.generic_pct:.0%} quantile of its layer.\n"
             f"- Hub = not generic, D >= {kmin}, B >= {sel_info['min_breadth']}. Rule: {sel_info['min_families_rule']}.\n"
             f"- Topic-specific = flagged in <= 2 contexts or a single family. Intermediate = the rest.\n"
             f"- Hub score (main) = D x evenness x L; variants: mean percentile rank of (D, evenness, L), D x raw load, "
             f"PR x raw load.\n")
    R.append("## Categories\n")
    R.append(md_table(["category", "units"], [[k, f"{v:,}"] for k, v in counts.items()]))
    nul = sel_info["null"]
    R.append("\n\nIndependence null for D (each context's flag independent at its per-layer rate):\n")
    R.append(md_table(["D >= k", "observed", "null expected", "ratio"],
                      [[k, f"{int(nul['observed_D_ge_k'][k]):,}", f"{nul['expected_D_ge_k'][k]:,.1f}",
                        (f"{nul['ratio'][k]:.3g}" if nul['ratio'][k] is not None else "n/a")] for k in range(1, K + 1)]))
    R.append("\n\nScore sensitivity (Jaccard of top-N hub sets between score variants):\n")
    sens = sel_info["score_sensitivity_jaccard_topN"]
    keys = list(next(iter(sens.values())).keys()) if sens else []
    R.append(md_table(["N"] + keys, [[N] + [f"{v[k]:.2f}" for k in keys] for N, v in sens.items()]))
    R.append(f"\n\nSpearman of variant ranks vs main over all hubs: {sel_info['score_spearman_vs_main']}\n")
    R.append(f"\nBreadth-threshold sensitivity (hub = not generic and D >= {kmin} with the context flag at another d; "
             f"Jaccard with the main hub set): {sel_info.get('breadth_threshold_sensitivity')}\n")
    R.append("## Top 50 hubs (main score)\n")
    R.append(md_table(["#", "unit", "score", "D", "B", "even", "L", "langs", "D no-forum", "D every-src",
                       "Eng pct", "register", "families"], rows))
    R.append("\n\n## Where hubs sit\n")
    R.append(md_table(["band", "hubs"], [[b, v] for b, v in summ["hub_layers"]["bands"].items()]))
    R.append("\n\nPer layer (hub units / layer): " + ", ".join(f"L{i}:{v}" for i, v in
                                                            enumerate(summ["hub_layers"]["per_layer"])) + "\n")
    he = summ["hub_experts"]
    R.append(f"\nExperts with >= 50% hub units: {he['experts_with_hub_share_ge_0.5']} "
             f"(>= 90%: {he['experts_with_hub_share_ge_0.9']}); they hold {he['share_of_hubs_in_those_experts']:.1%} "
             f"of hubs. Experts with any hub: {he['experts_with_any_hub']} of {S * 512}. "
             f"Correlation of router breadth ({he['router_breadth_definition']}) with hub share per expert: "
             f"{he['router_breadth_vs_hub_share_corr']}.\n")
    R.append(md_table(["layer", "expert", "hub share", "hub units"],
                      [[x[0], x[1], f"{x[2]:.2f}", x[3]] for x in hub_exp[:25]]))
    R.append(f"\n\n## Co-activation clusters (k-means, k={a.k}, on the binarized context profiles of the top "
             f"{hr['prof_bits'].shape[0]:,} hubs)\n")
    R.append(md_table(["cluster", "units", "mean contexts", "layer median [IQR]", "families (rate)", "top contexts"],
                      [[c["cluster"], c["units"], f"{c['mean_contexts']:.1f}",
                        f"{c['layer_median']} {c['layer_iqr']}",
                        ", ".join(f"{k} {v}" for k, v in c["family_rate"].items()),
                        ", ".join(f"{n} {v}" for n, v in c["top_contexts"][:5])] for c in clusters]))
    R.append(f"\n\nCluster stability across k-means seeds (Rand index on a 3,000-unit subsample): {stability}\n")
    R.append("\n\n## Hub overlap between contexts\n\nJaccard grows with how many hubs each context has; lift = "
             "shared hubs x all hubs / (hubs of a x hubs of b) corrects for that (1 = as expected by size).\n\n"
             "Top cross-family pairs by lift (at least 100 shared hubs):\n")
    R.append(md_table(["context a", "context b", "lift", "shared hubs"],
                      [[x["a"], x["b"], x["lift"], f"{x['hubs']:,}"] for x in summ["top_cross_family_pairs_by_lift"][:20]]))
    R.append("\n\nTop cross-family context pairs by Jaccard of their hub sets:\n")
    R.append(md_table(["context a", "context b", "Jaccard", "shared hubs"],
                      [[x["a"], x["b"], x["jaccard"], f"{x['hubs']:,}"] for x in summ["top_cross_family_pairs"][:20]]))
    R.append("\n\nFamily x family mean Jaccard of hub sets (diagonal: within family, off-diagonal: across):\n")
    R.append(md_table([""] + FAMILIES, [[FAMILIES[i]] + [f"{famov[i, j]:.2f}" for j in range(K)] for i in range(K)]))
    R.append("\n\n## Overlap with the partition\n")
    trows = []
    for k, v in tiles.items():
        if isinstance(v, dict) and "hubs_in" in v:
            trows.append([k, f"{v['units']:,}", f"{v['hubs_in']:,}", f"{v['share_of_hubs']:.1%}", f"{v['share_of_all']:.1%}"])
    R.append(md_table(["set", "units", "hubs in it", "share of hubs", "share of all units"], trows))
    R.append("\n\nHubs per box tile (top 15): " + ", ".join(f"{k} {v}" for k, v in list(tiles.get("hubs_per_box", {}).items())[:15]))
    R.append("\n\n## Register / source check\n")
    R.append(f"Each context's sources were re-tested alone against the English baseline (compare's test, BH over "
             f"every unit, q <= 0.01, d >= {a.breadth_d}). Source types: "
             + "; ".join(f"{t}: {rx[2:-2]}" for t, rx in SOURCE_RULES) + ".\n\n")
    R.append(f"Families with only forum sources: {forum_only or 'none'}; with only forum or chat sources: "
             f"{conv_only or 'none'}. A hub that needs such a family to reach D >= {kmin} is 'untestable', not "
             f"'forum register'. `agent` contexts are mostly synthetic traces, itself a register.\n\n")
    R.append(md_table(["register label", "hubs"], [[k, f"{v:,}"] for k, v in summ["register"].items()]))
    R.append("\n\nStricter (without forum and chat sources): " + ", ".join(f"{k}: {v:,}" for k, v in
                                                                          summ["register_strict"].items()) + ".\n")
    R.append(f"\n\nHubs whose D >= {kmin} holds within one single source type: {int(single_ok.sum()):,}. "
             "Per type: " + ", ".join(f"{t} {v:,}" for t, v in summ["register_best_single_type"].items()) + ".\n")
    R.append(f"\nMap every_source check: median D over hubs using only every-source flags: "
             f"{float(np.median(met['D_every'])):.0f} (all flags: {float(np.median(met['D'])):.0f}).\n")
    R.append("\n## Language check\n")
    lg = summ["language"]
    R.append(f"Hubs spanning languages and English topics ({lg['rule']}): {lg['hubs_spanning_langs_and_topics']:,}. "
             f"By band: {lg['by_band']}.\n\n")
    R.append(md_table(["layer", "units in >= %d langs" % a.lang_min, "of those also >= %d topic families" % a.lang_topic_min,
                       "hubs", "hubs spanning langs+topics"],
                      [[x["layer"], x["lang_units"], x["also_topic"], x["hubs"], x["hubs_langspan"]]
                       for x in lg["per_layer"]]))
    R.append("\n\n## Ablation candidates (not run)\n\nTop concept-labelled hubs and the hub-dense experts are the "
             "candidates for a future ablation check; nothing here shows they are needed.\n")
    cand = [i for i in range(nh) if reg[i] == 0][:20]
    R.append(md_table(["unit", "D", "families"],
                      [["L%d E%d n%d" % unit_str(int(hub_idx[i]), prof), int(met["D"][i]),
                        ",".join(FAMILIES[k] for k in range(K) if (int(met["famkey"][i]) >> k) & 1)] for i in cand]))
    R.append("\n")
    (out / "report.md").write_text("\n".join(R))
    return summ


# ---------------------------------------------------------------------------------------------------------- main
def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--joint-partition", default="")
    ap.add_argument("--map-root", default="", help="prefer DIR/<box>/map.npz over the partition's map path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all",
                    choices=["all", "profiles", "sources", "metrics", "select", "hubrows", "tiles", "report"])
    ap.add_argument("--ratio", type=float, default=None, help="exclusivity ratio (default and check: partition.json)")
    ap.add_argument("--breadth-d", type=float, default=1.0)
    ap.add_argument("--rate-d-grid", default="0.5,0.8,1.0,1.5,2.0")
    ap.add_argument("--min-families", type=int, default=0, help="fix the hub family threshold (0: from the null)")
    ap.add_argument("--min-families-floor", type=int, default=4)
    ap.add_argument("--min-breadth", type=int, default=0)
    ap.add_argument("--null-alpha", type=float, default=0.01)
    ap.add_argument("--generic-pct", type=float, default=0.95)
    ap.add_argument("--profile-top", type=int, default=20000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lang-min", type=int, default=3)
    ap.add_argument("--lang-topic-min", type=int, default=3)
    ap.add_argument("--route-ratio", type=float, default=2.0)
    ap.add_argument("--min-free-pages", type=int, default=300000)
    ap.add_argument("--max-swap-growth-gb", type=float, default=1.0)
    ap.add_argument("--drop-gb", type=float, default=2.0)
    ap.add_argument("--guard-interval", type=float, default=3.0)
    ap.add_argument("--drop-recheck-s", type=float, default=6.0,
                    help="after a sharp free-page drop, re-sample this many seconds later; pause only if it stayed down")
    ap.add_argument("--pause-file", default=str(Path.home() / "amoeba-runs/HUBMAP_PAUSE"))
    ap.add_argument("--no-guard", action="store_true", help="tests only")
    ap.add_argument("--yield-after-s", type=float, default=300.0,
                    help="exit with code 75 when a pause lasts longer (releases memory; rerun resumes); 0: never")
    ap.add_argument("--reset-swap-baseline", action="store_true",
                    help="measure swap growth from now instead of from the first run's start")
    a = ap.parse_args()
    import signal
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit("SIGTERM")))
    if "mlx" in sys.modules:
        raise SystemExit("mlx is imported: this tool is CPU only")
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    LOG = out / "hubmap.log"
    state_p = out / "state.json"
    state = json.loads(state_p.read_text()) if state_p.exists() else {}
    guard = Guard(a, state)
    part, ctx, core_map = load_contexts(a)
    if a.ratio is None:
        a.ratio = part.get("ratio", 0.25)
    elif part.get("ratio") is not None and abs(a.ratio - part["ratio"]) > 1e-12:
        raise SystemExit(f"--ratio {a.ratio} differs from the partition's {part['ratio']}")
    meta = {"checkpoint": ctx[0]["checkpoint"], "fingerprint": ctx[0]["fingerprint"], "mlx2_rev": git_rev(),
            "partition": str(Path(a.partition).resolve()), "joint_partition": a.joint_partition,
            "english_map": str(core_map), "args": vars(a),
            "contexts": [{k: c[k] for k in ("name", "family", "map_real", "capture", "baseline_capture")} for c in ctx]}
    hist_p = out / "runs.json"
    hist = json.loads(hist_p.read_text()) if hist_p.exists() else []
    hist.append({"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "rev": meta["mlx2_rev"], "stage": a.stage,
                 "args": vars(a)})
    save_json(hist_p, hist)
    save_json(out / "meta.json", meta)
    log(f"start: {len(ctx)} contexts, stage {a.stage}, swap at start {guard.swap0}")
    st = a.stage
    try:
        prof = stage_profiles(a, out, ctx, core_map, guard, state)
        if st == "profiles":
            return
        src = stage_sources(a, out, ctx, prof, guard, state)
        if src.get("breadth_d") != a.breadth_d:
            raise SystemExit(f"sources were computed at breadth d {src.get('breadth_d')}, not {a.breadth_d}")
        if st == "sources":
            return
        stage_metrics(a, out, ctx, prof, src, guard, state)
        if st == "metrics":
            return
        sel_info, sel = stage_select(a, out, ctx, prof, guard, state)
        if st == "select":
            return
        hr = stage_hubrows(a, out, ctx, prof, sel_info, sel, guard, state)
        if st == "hubrows":
            return
        tiles = stage_tiles(a, out, ctx, prof, sel_info, sel, part, guard, state)
        if st == "tiles":
            return
        stage_report(a, out, ctx, prof, src, sel_info, sel, hr, tiles, state, meta)
        log(f"report written: {out / 'report.md'}; peak RSS {rss_gb():.2f} GB")
    except Yield:
        state["peak_rss_gb"] = max(state.get("peak_rss_gb", 0.0), rss_gb())
        save_json(state_p, state)
        sys.exit(75)
    finally:
        state["peak_rss_gb"] = max(state.get("peak_rss_gb", 0.0), rss_gb())
        save_json(state_p, state)


if __name__ == "__main__":
    main()
