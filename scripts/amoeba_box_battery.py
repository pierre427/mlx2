#!/usr/bin/env python3
"""Box battery: test every box of a neuron partition (scripts/amoeba_fn_partition.py) in one model load (Pierre
2026-11-16/17: removing a box must not damage any other topic; boxes must reach +0.05 on held as served). Inside a GPU
window, neuron units, forward only. Per box B (its tile only; pair tiles stay in):
  without      the full model minus B: held dCE on B's own topic and on --unrelated topics (English plus two others
               of another kind), to show B is its topic's and nobody else's;
  random       the same per-layer counts removed at random among all units outside B (seeded): own topic and English;
  core_plus    only the core, B and the pair tiles that contain B kept (every other box and pair removed): held dCE on
               B's own topic (the +0.05 target as served) and on English.
Verdict per box (thresholds recorded): exclusive = own_without >= --min-own and >= --own-x x random_own, and every
unrelated_without <= --max-unrelated; served = core_plus_own <= --target. A path-offset arm (nothing removed) is
scored once. Results go to OUT/battery.jsonl (one row per box, written as it finishes: a rerun skips finished boxes)
and OUT/battery.md.

Held sets: --held NAME=CORPUS[?field=value][@TOKENS] (split "held", the first TOKENS tokens of up to --docs documents of
at least that length, seeded order). Each box needs a held set of the same name; --unrelated-map lists, per box, the
unrelated held set names (default: eng plus two chosen from the other kind: a language for a subject or inference
box, a subject for a language box). Unrelated sets per box: English, one far set of the other kind and one far set
of the same kind (FAR below: e.g. Mandarin -> French, a subject -> cooking / astronomy), never the box's own.

Opt-in arms (--arms, default "boxes" = the battery above, unchanged; Pierre 2026-11-18 "have we been too strict?"):
  unclaimed    the unclaimed units (the partition's --emit-unassigned file, a subset of the core) removed: the full
               model minus them on every held set; the core (group tiles folded back in, i.e. the core without
               family tiles) with and without them on English and every --global-held set (GSM8K); and per box the
               core minus them plus the box and its pair tiles on its held set and English (OUT/unclaimed.jsonl);
  groups       per family tile group:<name> of a v2 partition (scripts/amoeba_fn_partition.py --groups): removed vs
               the same per-layer counts removed at random outside it, on every member's held set and English; and
               per member box, served = core + every group tile containing it + the box + its pair tiles, on its held
               set and English (OUT/groups.jsonl). Verdicts: removal_beats_random = every member's net hurt minus the
               random hurt >= --group-min-hurt; served_+0.05 = every member and English <= --target.
Served compositions (core_plus in the boxes arm, served in the groups arm): core + every family tile containing the
box + the box + its pair tiles. Nested partitions (amoeba_fn_partition.py --nested, partition.json "nested") add the
box's ancestors' box tiles (a sci.physics.quantum document is a sci.physics and a sci document) and, for a parent,
its whole subtree (each descendant's box, family and pair tiles: a parent's held set is mostly its subtopics' docs),
recorded per row as "ancestors" / "descendants"; the unclaimed arm's per-box rows use the same composition on its
core. Unrelated sets are never in the box's lineage (an ancestor or descendant), whatever the partition. A held spec
NAME=CORPUS?split=val reads that split instead of held (a box whose build put no rows in held). --english NAME names the English held set (default eng; the category
partitions use "english", since eng is the engineering box there).

A held spec NAME=gsm8k:PATH[@TOKENS] reads GSM8K rows (question + answer, every row: the test split is never used to
carve). --held-note NAME=TEXT is printed with every row and table that uses NAME (e.g. a held split that is not a
separate held build). --reference-battery OLD/battery.jsonl adds last run's core+box values beside the unclaimed arm.
--joint-partition DIR (runs whatever --arms says; the joint-tile variant of --partition: amoeba_fn_partition.py --joint-floor, the same partition
with joint:<a>+<b>+... tiles cut from the core; checked) adds the joint arm on its --joint-top (10) largest joint
tiles: removal vs a random same-count removal on its members' held sets and English, and per member served = the
variant's core + its family tiles + ancestors / subtree + box + pairs + every joint tile naming the member, one of
its ancestors or one of its descendants (the joint gate is the OR of its members' gates; parent gates are always on) (OUT/joint.jsonl; skipped members recorded, not scored).

English box (a partition made with amoeba_fn_partition.py --english-box, partition.json "english_box"; Pierre
2026-11-19): its gate is "the text is English", so every served composition (core_plus, the groups and joint arms'
served, the unclaimed arm's per-box rows) adds the English box unless the box is in english_box.closed_for (the
language boxes): an English-text box is served with it, a language box without it. The English box is no topic box:
it is left out of the boxes arm (its own held set is --english) and tested by the english arm (required then):
  without        the full model minus the English box, on --english, every closed_for box's held set and
                 --global-held: removal must hurt English (net of the random removal, >= --group-min-hurt);
  random         the same per-layer counts removed at random outside it, on the same sets;
  core_open      the core + the English box (every other tile closed) on --english: <= --target (it serves English);
  core_closed    the core alone on --english: closing the English box must hurt English (>= --min-own over core_open);
  per closed_for box L: served_open = L's served composition + the English box, on L's held set; served_closed is
                 L's boxes-arm core_plus (English closed, reused when the boxes arm ran): closing English on non-English
                 text must cost little (served_closed - served_open <= --max-unrelated).
Rows: OUT/english.jsonl; verdicts in battery.md. A detach partition (--detach-share, partition.json "detach") changes
no arm: the boxes, groups and joint arms read the post-detach tiles, so a box that lost a claim it needed fails its
served arm; boxes-arm rows carry the box's claims_detached.

Usage: amoeba_box_battery.py --model M --partition DIR --held cmn=C.jsonl --held eng=E.jsonl ... --out DIR
       [--arms boxes,unclaimed,groups] [--unclaimed U.npz] [--global-held gsm8k] [--held-note NAME=TEXT]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LANGS = {"cmn", "fra", "deu", "spa", "jpn", "rus", "arb", "hin", "swh", "vie", "eng"}
LANG10 = LANGS - {"eng"}                                 # language boxes (English is the reference set, never a box)
FAR_LANG = {"cmn": "fra", "jpn": "fra"}                  # default far language: cmn (Han-script boxes: French)
FAR_SUBJ = ("life.cooking", "sci.astronomy")            # far subjects (the first that is not the box itself)


ARMS = ("boxes", "unclaimed", "groups", "english")


class Tiles:
    """A partition's tiles.npz held sparsely: per tile its flat unit indices (int32), read one tile at a time and
    never kept as dense masks (2026-11-19: a floor-50 joint partition has 13,557 tiles of one (48, 327680) bool mask
    each, 213 GB dense). tz[key] builds a fresh dense mask (as np.load(...)[key] did); mask(*keys) builds the union in
    one buffer. The file format is unchanged (any tiles.npz reads). Index memory is the partition's unit count x 4
    bytes (~63 MB for FN's 15.7M units), whatever the tile count."""

    def __init__(self, path):
        with np.load(path) as z:
            self.layers = [int(x) for x in z["layers"]]
            self.keys = [k for k in z.files if k != "layers"]
            self.files = ["layers"] + self.keys
            self.shape, self.idx = None, {}
            for k in self.keys:
                m = z[k]
                if self.shape is None:
                    self.shape = m.shape
                elif m.shape != self.shape:
                    raise SystemExit(f"{path}: tile {k} has shape {m.shape}, not {self.shape}")
                self.idx[k] = np.flatnonzero(m).astype(np.int32 if m.size < 2**31 else np.int64)
                del m
        self.size = int(np.prod(self.shape)) if self.shape else 0

    def mask(self, *keys):
        """The union of the named tiles as a fresh dense bool mask."""
        m = np.zeros(self.size, bool)
        for k in keys:
            m[self.idx[k]] = True
        return m.reshape(self.shape)

    def __getitem__(self, key):
        if key == "layers":
            return np.array(self.layers)
        return self.mask(key)

    def units(self, key):
        return int(self.idx[key].size)


def random_like(mask, g):
    """The same per-layer counts drawn at random among the units outside `mask` (rng g)."""
    rnd = np.zeros_like(mask)
    for l in range(mask.shape[0]):
        k = int(mask[l].sum())
        if k:
            cand = np.flatnonzero(~mask[l])
            rnd[l, g.choice(cand, min(k, cand.size), replace=False)] = True
    return rnd


def lineage(a, b):
    """a and b are the same box, or one is a dotted-name ancestor of the other."""
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def ancestors_of(prep, b):
    """The box's ancestors in a nested partition (else none)."""
    return list(((prep.get("nested") or {}).get("ancestors") or {}).get(b, []))


def descendants_of(prep, b):
    """The box's descendants in a nested partition (else none)."""
    return list(((prep.get("nested") or {}).get("descendants") or {}).get(b, []))


def english_box(prep):
    """(name, closed_for) of the partition's English box, or (None, [])."""
    e = prep.get("english_box") or {}
    return e.get("name"), list(e.get("closed_for") or [])


_ETILES = {}


def english_tiles(get, keys, prep):
    """The English box, and with an English-shared partition (amoeba_fn_partition.py --english-shared) every pair and
    family tile naming it (their gate is the OR of their members', English is open). Cached per partition."""
    key = (id(prep), len(keys))
    if key not in _ETILES:
        en, _ = english_box(prep)
        keep = get(f"box:{en}").copy()
        if (prep.get("english_box") or {}).get("shared"):
            for k in pair_keys(keys, en) + group_keys_of(prep, en):
                keep |= get(k)
        _ETILES.clear() if len(_ETILES) > 4 else None
        _ETILES[key] = (prep, keep)                  # prep kept alive, so its id is not reused while cached
    return _ETILES[key][1]


def subtree_keep(get, keys, prep, b, base, groups=True):
    """base + the ancestors' box tiles + for b and each descendant: its box, its pair tiles and (groups) every family
    tile containing it. Without nesting: base + box b + b's pairs (+ b's family tiles). With an English box (gate:
    the text is English): + the English box unless b is one of its closed_for boxes."""
    keep = base
    en, closed = english_box(prep)
    if en and b not in closed and b != en:
        keep = keep | english_tiles(get, keys, prep)
    for x in ancestors_of(prep, b):
        keep = keep | get(f"box:{x}")
    for x in [b] + descendants_of(prep, b):
        keep = keep | get(f"box:{x}")
        for k in (group_keys_of(prep, x) if groups else []) + pair_keys(keys, x):
            keep = keep | get(k)
    return keep


def pair_keys(keys, b):
    return [k for k in keys if k.startswith("pair:") and b in k[5:].split("__")]


def group_keys_of(prep, b):
    """The kept family tiles (v2 partition) whose members include box b."""
    kept = (prep.get("groups") or {}).get("kept") or {}
    return [f"group:{g}" for g in sorted(kept) if b in kept[g]["members"]]


def served_keep(get, keys, prep, b, core):
    """Core + every group tile containing b + box b + b's pair tiles (+ ancestors and subtree when nested)."""
    return subtree_keep(get, keys, prep, b, core)


def folded_core(get, keys):
    """The core with every group tile folded back in (= the partition's core without family tiles). An English box
    stays out (subtree_keep opens it for English-text boxes)."""
    core = get("core").copy()
    for k in keys:
        if k.startswith("group:"):
            core |= get(k)
    return core


def read_held_rows(path, kind, filt):
    if kind == "gsm8k":
        if filt:
            raise SystemExit(f"a gsm8k held spec takes no ?field=value filter ({path}?{filt})")
        rows = [json.loads(x) for x in Path(path).read_text().split("\n") if x.strip()]
        return [{"text": f"Question: {r['question']}\nAnswer: {r['answer']}"} for r in rows]
    rows = [json.loads(x) for x in Path(path).read_text().split("\n") if x.strip()]
    split = "held"
    if filt.startswith("split="):                       # another split in place of held (?split=val)
        split, filt = filt.partition("=")[2], ""
    rows = [r for r in rows if r.get("split") == split]
    if filt:
        fk, _, fv = filt.partition("=")
        rows = [r for r in rows if str(r.get(fk)) == fv]
    return rows


def _wired_arg(spec):
    """--wired-gb: a bad value fails at parse time, before the model loads (ops.wired_arg, imported lazily)."""
    from research.amoeba.carve.ops import wired_arg
    return wired_arg(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--partition", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--held", action="append", required=True)
    ap.add_argument("--boxes", default="", help="comma list (default: every box of the partition)")
    ap.add_argument("--docs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-own", type=float, default=0.10)
    ap.add_argument("--own-x", type=float, default=2.0)
    ap.add_argument("--max-unrelated", type=float, default=0.02)
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--assert-ple", action="store_true")
    ap.add_argument("--memory-gb", type=float, default=88)
    ap.add_argument("--watchdog-gb", type=float, default=92)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--watchdog-hard-gb", type=float, default=None,
                    help="footprint at which the watcher process SIGKILLs the run (default --watchdog-gb + 6)")
    ap.add_argument("--watchdog-every-s", type=float, default=1.0)
    ap.add_argument("--min-avail-gb", type=float, default=15.0,
                    help="stop when free + inactive + speculative + purgeable falls below this")
    ap.add_argument("--wired-gb", default=os.environ.get("AMOEBA_WIRED_GB", "").strip() or "auto", type=_wired_arg,
                    help="wire the model weights after load (MLX set_wired_limit; research/amoeba/carve/ops.py "
                         "wire_weights): auto = MLX active memory after load + 10 GiB (FN: ~80), a number of GiB, or "
                         "0 = off; capped 8 GiB under min(iogpu.wired_limit_mb, the recommended working set) and at "
                         "--memory-gb; env AMOEBA_WIRED_GB (2026-11-19 19:19: macOS compressed the battery's weights)")
    ap.add_argument("--compressor-growth-gb", type=float, default=12.0,
                    help="stop when the memory compressor grows this much over its level at start")
    ap.add_argument("--waiver-note", default="")
    ap.add_argument("--arms", default="boxes", help="comma list of " + ", ".join(ARMS))
    ap.add_argument("--unclaimed", default="", help="unclaimed units npz (default PARTITION/unassigned.npz)")
    ap.add_argument("--global-held", default="", help="comma list of held sets for the global unclaimed rows")
    ap.add_argument("--group-min-hurt", type=float, default=0.01)
    ap.add_argument("--held-note", action="append", default=[], help="NAME=TEXT")
    ap.add_argument("--reference-battery", default="")
    ap.add_argument("--english", default="eng", help="the English held set's name (category partitions: english)")
    ap.add_argument("--joint-partition", default="", help="the joint-tile variant partition (joint arm)")
    ap.add_argument("--joint-top", type=int, default=10)
    ap.add_argument("--skipped", action="append", default=[], help="NAME=REASON: a partition box left out of every "
                    "arm (recorded in the meta and battery.md)")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    if not arms or any(x not in ARMS for x in arms):
        raise SystemExit(f"--arms: one or more of {ARMS}")
    notes = {}
    for x in a.held_note:
        k, _, v = x.partition("=")
        notes[k] = v
    from research.amoeba.carve import ops as O
    O.prepare_wiring(a.wired_gb)                       # before any Metal call (MLX reads the refresh env then)
    from research.amoeba.carve import __main__ as D
    from research.amoeba.carve import evals as E
    from research.amoeba.carve.host import Host
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)    # noqa: E731
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd = Path(a.partition)
    prep = json.loads((pd / "partition.json").read_text())
    tz = Tiles(pd / "tiles.npz")                               # sparse: masks are built per composition
    layers = tz.layers
    keys = list(tz.keys)
    core_t = tz["core"]
    boxes = [b for b in (a.boxes.split(",") if a.boxes else sorted(k[4:] for k in keys if k.startswith("box:"))) if b]
    EBOX, ECLOSED = english_box(prep)
    if EBOX and f"box:{EBOX}" not in keys:
        raise SystemExit(f"partition.json names English box {EBOX} but tiles.npz has no box:{EBOX}")
    if EBOX and EBOX in boxes and "english" not in arms:
        raise SystemExit(f"the partition has an English box ({EBOX}): add the english arm (--arms ...,english) or "
                         "leave it out of --boxes")
    if "english" in arms and not EBOX:
        raise SystemExit("--arms english needs a partition with an English box (amoeba_fn_partition.py --english-box)")
    run_english = bool(EBOX) and "english" in arms and (not a.boxes or EBOX in boxes)
    boxes = [b for b in boxes if b != EBOX]                # the English box is the english arm's, not a topic box
    skipped = dict(x.partition("=")[::2] for x in a.skipped)        # checked before the model loads
    bad = [k for k in skipped if k in boxes]
    if bad:
        raise SystemExit(f"--skipped boxes still in --boxes: {bad}")
    unknown = [k for k in skipped if f"box:{k}" not in keys]
    if unknown:
        raise SystemExit(f"--skipped boxes not in the partition: {unknown}")
    jz = jprep = None
    if a.joint_partition:                              # checked before the model loads
        jd = Path(a.joint_partition)
        jprep = json.loads((jd / "partition.json").read_text())
        jz = Tiles(jd / "tiles.npz")
        jkeys = list(jz.keys)
        if jprep.get("fingerprint") != prep.get("fingerprint") or jz.layers != layers:
            raise SystemExit("--joint-partition: another model or layer list")
        if not (jprep.get("joint") or {}).get("tiles"):
            raise SystemExit("--joint-partition has no joint tiles")
        if sorted(k for k in jkeys if not k.startswith("joint:")) != sorted(keys):
            raise SystemExit("--joint-partition: its non-joint tiles differ from --partition's keys")
        folded = jz.mask("core", *[k for k in jkeys if k.startswith("joint:")])
        if not np.array_equal(folded, core_t):
            raise SystemExit("--joint-partition: core + joint tiles != --partition's core")
        if ({k: v.get("units") for k, v in (jprep.get("boxes") or {}).items()} !=
                {k: v.get("units") for k, v in (prep.get("boxes") or {}).items()} or
                jprep.get("pairs") != prep.get("pairs") or
                {g: v.get("units") for g, v in ((jprep.get("groups") or {}).get("kept") or {}).items()} !=
                {g: v.get("units") for g, v in ((prep.get("groups") or {}).get("kept") or {}).items()}):
            raise SystemExit("--joint-partition: box, pair or family tile counts differ from --partition's")
        del folded
    if "groups" in arms and not (prep.get("groups") or {}).get("kept"):
        raise SystemExit("--arms groups needs a partition with kept family tiles (amoeba_fn_partition.py --groups)")
    upath = Path(a.unclaimed) if a.unclaimed else pd / "unassigned.npz"
    if "unclaimed" in arms:
        if not upath.exists():
            raise SystemExit(f"--arms unclaimed: {upath} missing (amoeba_fn_partition.py --emit-unassigned)")
        uz = np.load(upath)
        if [int(x) for x in uz["layers"]] != layers or uz["unassigned"].shape != core_t.shape:
            raise SystemExit("unclaimed mask of another layout")
        U = uz["unassigned"].astype(bool)
        if (U & ~folded_core(lambda k: tz[k], keys)).any():
            raise SystemExit("unclaimed units outside the core: not this partition's")
    hspecs = {x.partition("=")[0]: x for x in a.held}

    def write_meta(m):
        tmp = out / "battery.meta.json.tmp"
        tmp.write_text(json.dumps(m, indent=1))
        tmp.replace(out / "battery.meta.json")

    def start_meta(status, docs_loaded=None):
        """battery.meta.json while the run is going (2026-11-19: a battery killed mid-run left battery/ empty, and the
        interaction arm, scripts/amoeba_fn_interactions.py, reads its held sets, notes and skips from this file).
        The end-of-run write replaces it with the full meta of what was scored."""
        hn = dict(notes)
        for k, x in hspecs.items():                     # the gsm8k default note, as the held loading adds it
            if x.partition("=")[2].startswith("gsm8k:"):
                hn.setdefault(k, "GSM8K test rows (question + answer) as held text; dCE, not accuracy")
        m = {"status": status, "fingerprint": prep.get("fingerprint"), "partition": str(pd), "rule": prep.get("rule"),
             "docs": a.docs, "held": {k: {"spec": x, "docs_loaded": (docs_loaded or {}).get(k)}
                                      for k, x in hspecs.items()},
             "held_notes": hn, "skipped": skipped, "arms": arms, "boxes": boxes,
             "watchdog": {"limit_gb": a.watchdog_gb, "hard_gb": a.watchdog_hard_gb, "every_s": a.watchdog_every_s,
                          "min_avail_gb": a.min_avail_gb, "compressor_growth_gb": a.compressor_growth_gb,
                          "swap_growth_gb": a.swap_growth_gb, "waiver_note": a.waiver_note or None},
             "wired": wired[0] or {"requested": a.wired_gb, "status": "not yet (wired after the model loads)"},
             **({"joint_partition": a.joint_partition, "joint_top": a.joint_top} if a.joint_partition else {})}
        write_meta(m)
    wired = [None]
    mp = out / "battery.meta.json"
    try:                                               # a resumed run: keep the last finished meta beside it
        if json.loads(mp.read_text()).get("status", "finished") == "finished":
            mp.replace(out / "battery.meta.prev.json")
    except (OSError, ValueError):
        pass
    start_meta("started")
    O.footprint_watchdog(a.watchdog_gb, log, swap_growth_gb=a.swap_growth_gb, every_s=a.watchdog_every_s,
                         min_avail_gb=a.min_avail_gb, hard_gb=a.watchdog_hard_gb,
                         compressor_growth_gb=a.compressor_growth_gb, receipt=out / "watchdog-stop.json",
                         watcher=True)
    O.setup_mlx(a.memory_gb, 2)
    host = Host(a.model, log=log, moe_units="neuron")
    if host.fingerprint != prep["fingerprint"]:
        raise SystemExit(f"partition of {prep['fingerprint']}, model is {host.fingerprint}")
    if a.assert_ple:
        tables = getattr(getattr(host, "adapter", None), "_tables", None)
        if not tables or Path(os.environ.get("MLX_QWEN4_PLE_NVME", "")).expanduser().resolve() != \
                Path(a.model).expanduser().resolve() / "ple_rows.bin":
            raise SystemExit("PLE offload is not active (FN rule: assert it)")
    wired[0] = O.wire_weights(a.wired_gb, log, memory_gb=a.memory_gb)     # after load: Host evaluated the weights
    W = max(host.widths)
    if core_t.shape != (len(layers), W):
        raise SystemExit(f"tile shape {core_t.shape} != [moe layers, {W}]")
    held, htok = {}, {}
    for spec in a.held:
        name, _, rest = spec.partition("=")
        rest, _, tok = rest.partition("@")
        T = int(tok or 256)
        kind = "gsm8k" if rest.startswith("gsm8k:") else "corpus"
        path, _, filt = rest.removeprefix("gsm8k:").partition("?")
        rows = read_held_rows(path, kind, filt)
        if kind == "gsm8k":
            notes.setdefault(name, "GSM8K test rows (question + answer) as held text; dCE, not accuracy")
        np.random.default_rng(a.seed).shuffle(rows)
        seqs = []
        for r in rows:
            t = host.encode(r["text"])
            if len(t) >= T:
                seqs.append(t[:T])
            if len(seqs) >= a.docs:
                break
        if not seqs:
            raise SystemExit(f"held {name}: no document of {T} tokens")
        held[name], htok[name] = {"held": seqs}, T
        log(f"held {name}: {len(seqs)} x {T} tokens")
    start_meta("scoring", {k: len(v["held"]) for k, v in held.items()})
    missing = [b for b in boxes if b not in held] + [x for x in a.global_held.split(",") if x and x not in held]
    EN = a.english
    if EN in boxes:
        raise SystemExit(f"--english {EN} is also a box name: name the English held set differently")
    if missing or EN not in held:
        raise SystemExit(f"held sets missing for {missing + ([] if EN in held else [EN])}")
    clean = {}

    def score(keep_rows, names):
        m = np.ones((host.L, W), np.float32)
        m[layers] = keep_rows.astype(np.float32)
        cfg = E.configure(host, D.from_np(m, host), None, host.L)
        r = E.english_cost(host, {k: held[k] for k in names}, clean, cfg, pooled=False)
        return {k: v for k, v in r.items() if k != "mean_d_ce"}

    allones = np.ones_like(core_t)
    settings = {"docs": a.docs, "seed": a.seed, "partition": str(pd), "held": sorted(a.held)}
    if a.english != "eng":                                  # keys only off the defaults (golden rows unchanged)
        settings["english"] = a.english
    if prep.get("nested"):
        settings["nested"] = True
    done = set()
    bj = out / "battery.jsonl"
    if bj.exists():                         # reuse only complete rows made with these settings
        for x in bj.read_text().split("\n"):
            try:
                r = json.loads(x)
            except ValueError:
                continue
            if r.get("settings") == settings:
                done.add(r["box"])
    po = out / "path_offset.json"
    if po.exists() and set(held) - set(json.loads(po.read_text())):     # a held set added on resume: rescore all
        po.unlink()
    if not po.exists():
        po.write_text(json.dumps(score(allones, list(held)), indent=1))
    off = json.loads(po.read_text())
    log("path offset:", off)
    for i, b in enumerate(boxes if "boxes" in arms else []):
        if b in done:
            continue
        t0 = time.time()
        box = tz[f"box:{b}"]
        if b in LANG10:
            far_other = next((s for s in FAR_SUBJ if s in held), None)
            far_same = FAR_LANG.get(b, "cmn")
        else:
            far_other = "cmn"
            far_same = next((s for s in FAR_SUBJ if s in held and not lineage(s, b)), None)
        unrelated = [EN] + [x for x in (far_other, far_same) if x and x in held and not lineage(x, b)]
        without = score(~box, [b] + unrelated)
        rnd = random_like(box, np.random.default_rng(a.seed + 1000 + i))
        random_ = score(~rnd, [b, EN])
        pairs = [k for k in keys if k.startswith("pair:") and b in k[5:].split("__")]
        anc, desc, grp = ancestors_of(prep, b), descendants_of(prep, b), group_keys_of(prep, b)
        keep = served_keep(lambda k: tz[k], keys, prep, b, core_t)
        core_plus = score(keep, [b, EN])
        net = lambda d: {k: v - off.get(k, 0.0) for k, v in d.items()}      # noqa: E731  (path offset removed)
        nw, nr, nc = net(without), net(random_), net(core_plus)
        own, ro = nw[b], nr[b]
        verdict = {"exclusive": bool(own >= a.min_own and own >= a.own_x * max(ro, 1e-9)
                                     and all(nw[u] <= a.max_unrelated for u in unrelated)),
                   "served_+0.05": bool(nc[b] <= a.target), "on": "dCE net of the path offset"}
        row = {"box": b, "units": int(box.sum()), "pairs": pairs, **({"ancestors": anc} if anc else {}),
               **({"served_composition": "core + family tiles containing the box + every ancestor box (parent tiles "
                   "are always gated in for their subtiles) + the box + its pair tiles"
                   + (" + its whole subtree (every descendant's box, pair and family tiles)" if desc else "")}
                  if prep.get("nested") else {}),
               **({"descendants": desc} if desc else {}), **({"groups": grp} if grp else {}),
               **({"claims_detached": (prep["boxes"].get(b) or {}).get("claims_detached", 0)}
                  if (prep.get("detach") or {}).get("share") else {}),
               **({"english_box_open": b not in ECLOSED} if EBOX else {}),
               "unrelated": unrelated, "without": without,
               "random": random_, "core_plus": core_plus, "path_offset": off, "verdict": verdict,
               "settings": settings, "wall_s": round(time.time() - t0, 1)}
        with bj.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        log(f"{b}: units {row['units']:,}; without own {own:+.3f} " +
            " ".join(f"{u} {without[u]:+.3f}" for u in unrelated) +
            f"; random own {ro:+.3f} {EN} {random_[EN]:+.3f}; core+box own {core_plus[b]:+.3f} {EN} "
            f"{core_plus[EN]:+.3f}; {verdict}")
    net = lambda d: {k: v - off.get(k, 0.0) for k, v in d.items()}          # noqa: E731
    get = lambda k: tz[k]                                                     # noqa: E731
    boxes_rows = {}                                    # this run's boxes-arm rows (same settings): core_plus reuse
    for x in bj.read_text().split("\n") if bj.exists() else []:
        try:
            r = json.loads(x)
        except ValueError:
            continue
        if r.get("settings") == settings:
            boxes_rows[r["box"]] = r
    asettings = {**settings, "target": a.target, "group_min_hurt": a.group_min_hurt, "global_held": a.global_held,
                 "unclaimed": str(upath)}
    mark = lambda k: f"{k}\u2020" if k in notes else k                       # noqa: E731  (a held-set note below)

    def arm_rows(path):
        got = {}
        if path.exists():
            for x in path.read_text().split("\n"):
                try:
                    r = json.loads(x)
                except ValueError:
                    continue
                if r.get("settings") == asettings:
                    got[r["key"]] = r
        return got

    def put(path, row):
        row["settings"] = asettings
        row["held_notes"] = {k: notes[k] for k in row.get("sets", []) if k in notes}
        with path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        return row

    extra_md = []
    if "unclaimed" in arms:
        uj = out / "unclaimed.jsonl"
        got = arm_rows(uj)
        C0 = folded_core(get, keys)
        EOPEN = english_tiles(get, keys, prep) if EBOX else np.zeros_like(C0)    # English text: English box open
        gl = [x for x in a.global_held.split(",") if x]
        jobs = [("global:full_minus_unclaimed", lambda: ~U, list(held)),
                ("global:core", lambda: C0 | EOPEN, [EN] + gl),
                ("global:core_minus_unclaimed", lambda: (C0 & ~U) | EOPEN, [EN] + gl)]
        for b in boxes:
            jobs.append((f"box:{b}", lambda b=b: subtree_keep(get, keys, prep, b, C0 & ~U, groups=False), [b, EN]))
        for key, mk, sets in jobs:
            if key in got:
                continue
            t0 = time.time()
            keep = mk()
            d = score(keep, sets)
            nd = net(d)
            row = {"arm": "unclaimed", "key": key, "sets": sets, "kept_units": int(keep.sum()),
                   "unclaimed_units": int(U.sum()), "dce": d, "net": nd,
                   "served_+0.05": bool(all(v <= a.target for v in nd.values())), "wall_s": round(time.time() - t0, 1)}
            got[key] = put(uj, row)
            log(f"unclaimed {key}: kept {row['kept_units']:,}; " + " ".join(f"{k} {v:+.3f}" for k, v in nd.items()))
        ref = {}
        if a.reference_battery and Path(a.reference_battery).exists():    # last run's core+box, net of its offset
            for x in Path(a.reference_battery).read_text().split("\n"):
                try:
                    r = json.loads(x)
                except ValueError:
                    continue
                rs = r.get("settings") or {}
                if rs.get("docs") == a.docs and rs.get("seed") == a.seed:
                    po_ = r.get("path_offset") or {}
                    ref[r["box"]] = {k: v - po_.get(k, 0.0) for k, v in r["core_plus"].items()}
        extra_md += [f"\nUnclaimed arm: {int(U.sum()):,} unclaimed units removed from the core "
                     f"(core without family tiles {int(C0.sum()):,}); held dCE net of the path offset.\n",
                     "| row | kept units | held dCE | reference core+box (last run) | +0.05 |", "|---|---|---|---|---|"]
        for key, _, _ in jobs:
            r = got[key]
            b = key[4:] if key.startswith("box:") else None
            rf = ref.get(b) if b else None
            extra_md.append(f"| {key} | {r['kept_units']:,} | " + ", ".join(f"{mark(k)} {v:+.3f}" for k, v in r["net"].items())
                            + " | " + (f"{b} {rf[b]:+.3f}, {EN} {rf[EN]:+.3f}" if rf and b in rf and EN in rf else "-")
                            + f" | {'yes' if r['served_+0.05'] else 'NO'} |")
    if "groups" in arms:
        gj = out / "groups.jsonl"
        got = arm_rows(gj)
        kept = prep["groups"]["kept"]
        core = tz["core"]
        served = {}
        for k, r in got.items():
            if k.startswith("served:"):
                served[k[7:]] = r
        for gi, gname in enumerate(sorted(kept)):
            mem = kept[gname]["members"]
            have = [m for m in mem if m in held]
            for m in have:
                if m in served:
                    continue
                t0 = time.time()
                keep = served_keep(get, keys, prep, m, core)
                bx = boxes_rows.get(m) if "boxes" in arms else None     # the boxes arm scored this same mask
                d = dict(bx["core_plus"]) if bx else score(keep, [m, EN])
                nd = net(d)
                served[m] = put(gj, {"arm": "groups", "key": f"served:{m}", "sets": [m, EN], "dce": d, "net": nd,
                                     "kept_units": int(keep.sum()), "groups": group_keys_of(prep, m),
                                     "pairs": pair_keys(keys, m), "wall_s": round(time.time() - t0, 1),
                                     **({"reused": "boxes arm core_plus (same mask)"} if bx else {})})
                log(f"served {m}: {nd[m]:+.3f} {EN} {nd[EN]:+.3f} (groups {served[m]['groups']})")
            key = f"group:{gname}"
            if key in got:
                continue
            t0 = time.time()
            G = tz[key]
            sets = have + [EN]
            without = score(~G, sets)
            rnd = random_like(G, np.random.default_rng(a.seed + 5000 + gi))
            random_ = score(~rnd, sets)
            nw, nr = net(without), net(random_)
            hurt = {m: nw[m] - nr[m] for m in have}
            verdict = {"removal_beats_random": bool(have and all(h >= a.group_min_hurt for h in hurt.values())),
                       "served_+0.05": bool(have and all(served[m]["net"][m] <= a.target and
                                                         served[m]["net"][EN] <= a.target for m in have)),
                       "members_without_held": [m for m in mem if m not in held], "on": "dCE net of the path offset",
                       **({"members_skipped": {m: skipped[m] for m in mem if m in skipped}}
                          if any(m in skipped for m in mem) else {})}
            row = put(gj, {"arm": "groups", "key": key, "sets": sets, "members": mem, "units": int(G.sum()),
                           "parent": kept[gname].get("parent"), "without": without, "random": random_,
                           "hurt_minus_random": hurt, "served": {m: served[m]["net"] for m in have},
                           "verdict": verdict, "wall_s": round(time.time() - t0, 1)})
            got[key] = row
            log(f"{key}: units {row['units']:,}; " + " ".join(f"{m} {nw[m]:+.3f}/{nr[m]:+.3f}" for m in have)
                + f" {EN} {nw[EN]:+.3f}/{nr[EN]:+.3f}; {verdict}")
        extra_md += ["\nFamily tiles: removal (group tile removed) vs a random same-count removal, and served = core "
                     "+ every group tile containing the member + its box + its pair tiles; held dCE net of the path "
                     "offset.\n",
                     "| group | units | members: without / random | eng without / random | served member / eng | "
                     "beats random | +0.05 served |", "|---|---|---|---|---|---|---|"]
        for gname in sorted(kept):
            r = got[f"group:{gname}"]
            have = [m for m in r["members"] if m in held]
            nw, nr = net(r["without"]), net(r["random"])
            ns = len(r["verdict"].get("members_skipped") or {})
            extra_md.append(f"| {gname}{f' ({ns} skipped)' if ns else ''} | {r['units']:,} | "
                            + ", ".join(f"{mark(m)} {nw[m]:+.3f} / {nr[m]:+.3f}" for m in have)
                            + f" | {nw[EN]:+.3f} / {nr[EN]:+.3f} | "
                            + ", ".join(f"{mark(m)} {r['served'][m][m]:+.3f} / {r['served'][m][EN]:+.3f}" for m in have)
                            + f" | {'yes' if r['verdict']['removal_beats_random'] else 'NO'} | "
                              f"{'yes' if r['verdict']['served_+0.05'] else 'NO'} |")
    if run_english:
        ej = out / "english.jsonl"
        got = arm_rows(ej)
        Ebox = english_tiles(get, keys, prep)            # the box (+ English-naming tiles when shared)
        core = tz["core"]
        langs = [x for x in ECLOSED if x in held and x not in skipped]
        gl = [x for x in a.global_held.split(",") if x and x != EN]
        wsets = [EN] + [x for x in langs + gl if x != EN]
        jobs = [("without", lambda: ~Ebox, wsets),
                ("random", lambda: ~random_like(Ebox, np.random.default_rng(a.seed + 7000)), wsets),
                ("core_open", lambda: core | Ebox, [EN]),
                ("core_closed", lambda: core, [EN])]
        for L_ in langs:
            jobs.append((f"served_open:{L_}", lambda L_=L_: served_keep(get, keys, prep, L_, core) | Ebox, [L_]))
            if L_ not in boxes_rows:                      # English closed = L's boxes-arm core_plus mask
                jobs.append((f"served_closed:{L_}", lambda L_=L_: served_keep(get, keys, prep, L_, core), [L_]))
        for key, mk, sets in jobs:
            if key in got:
                continue
            t0 = time.time()
            keep = mk()
            d = score(keep, sets)
            got[key] = put(ej, {"arm": "english", "key": key, "sets": sets, "kept_units": int(keep.sum()),
                                "english_box_units": int(Ebox.sum()), "dce": d, "net": net(d),
                                "wall_s": round(time.time() - t0, 1)})
            log(f"english {key}: kept {int(keep.sum()):,}; " + " ".join(f"{k} {v:+.3f}" for k, v in net(d).items()))

        def closed_net(L_):
            r = boxes_rows.get(L_)
            return net(r["core_plus"])[L_] if r else got[f"served_closed:{L_}"]["net"][L_]
        nw, nr = got["without"]["net"], got["random"]["net"]
        co, cc = got["core_open"]["net"][EN], got["core_closed"]["net"][EN]
        cost = {L_: closed_net(L_) - got[f"served_open:{L_}"]["net"][L_] for L_ in langs}
        verdict = {"removal_beats_random": bool(nw[EN] - nr[EN] >= a.group_min_hurt),
                   "served_+0.05": bool(co <= a.target),
                   "closed_hurts_english": bool(cc - co >= a.min_own),
                   "closed_costs_little_off_english": (bool(all(v <= a.max_unrelated for v in cost.values()))
                                                       if cost else None),
                   "on": "dCE net of the path offset"}
        if (got.get("verdict") or {}).get("verdict") != verdict:      # a verdict row per change (resume-safe)
            got["verdict"] = put(ej, {"arm": "english", "key": "verdict", "sets": [], "box": EBOX, "verdict": verdict,
                                      "closing_cost_off_english": cost, "core_open": co, "core_closed": cc,
                                      "removal_hurt_minus_random": nw[EN] - nr[EN], "closed_for": ECLOSED,
                                      "languages_scored": langs})
        extra_md += [f"\nEnglish box {EBOX} ({int(Ebox.sum()):,} units; gate: the text is English, closed for "
                     f"{', '.join(ECLOSED) or 'none'}); held dCE net of the path offset.\n",
                     "| row | value |", "|---|---|",
                     f"| removed: {EN} / random | {nw[EN]:+.3f} / {nr[EN]:+.3f} |",
                     "| removed: other sets | " + ", ".join(f"{mark(k)} {nw[k]:+.3f} / {nr[k]:+.3f}" for k in wsets
                                                      if k != EN) + " |",
                     f"| core + English box on {EN} (open) | {co:+.3f} |",
                     f"| core alone on {EN} (closed) | {cc:+.3f} |",
                     "| closing cost off English (served closed - open) | "
                     + ", ".join(f"{mark(k)} {v:+.3f}" for k, v in cost.items()) + " |",
                     "| verdict | " + ", ".join(f"{k} {'-' if v is None else 'yes' if v else 'NO'}"
                                                for k, v in verdict.items() if k != "on")
                     + " |"]
    if jz is not None:
        jj = out / "joint.jsonl"
        with open(Path(a.joint_partition) / "tiles.npz", "rb") as fh:       # chunked: no spike beside the model
            jsha = hashlib.file_digest(fh, "sha256").hexdigest()[:12]
        asettings = {**asettings, "joint_partition": a.joint_partition, "joint_floor": jprep["joint"].get("floor"),
                     "joint_tiles_sha256_12": jsha, "joint_top": a.joint_top}  # resume only on the same variant
        got = arm_rows(jj)
        jget = lambda k: jz[k]                                             # noqa: E731
        jt = jprep["joint"]["tiles"]
        top = sorted(jt, key=lambda k: -jt[k]["units"])[:a.joint_top]
        jkeys_all = list(jz.keys)
        jcore = jz["core"]
        jserved = {k[7:]: r for k, r in got.items() if k.startswith("served:")}

        def lineage_set(m):                            # the member, its ancestors (gates always on) and descendants
            return {m} | set(ancestors_of(jprep, m)) | set(descendants_of(jprep, m))
        for ji, name in enumerate(top):
            mem = jt[name]["members"]
            have = [m for m in mem if m in held and m not in skipped]
            for m in have:
                if m in jserved:
                    continue
                t0 = time.time()
                keep = served_keep(jget, jkeys_all, jprep, m, jcore)
                lin = lineage_set(m)                   # OR gates: open for m, its ancestors and its subtree
                en_, closed_ = english_box(jprep)
                if en_ and m not in closed_:           # English text: English's joint tiles are open too
                    lin.add(en_)
                opened = [k for k in jkeys_all if k.startswith("joint:") and lin & set(k[6:].split("+"))]
                if opened:                             # one union buffer, not one dense mask per opened tile
                    keep = keep | jz.mask(*opened)
                d = score(keep, [m, EN])
                jserved[m] = put(jj, {"arm": "joint", "key": f"served:{m}", "sets": [m, EN], "dce": d, "net": net(d),
                                      "kept_units": int(keep.sum()), "joint_tiles_open": opened,
                                      "wall_s": round(time.time() - t0, 1)})
                log(f"joint served {m}: {jserved[m]['net'][m]:+.3f} {EN} {jserved[m]['net'][EN]:+.3f}")
            key = f"joint:{name}"
            if key in got:
                continue
            t0 = time.time()
            J = jz[key]
            sets = have + [EN]
            without = score(~J, sets)
            rnd = random_like(J, np.random.default_rng(a.seed + 9000 + ji))
            random_ = score(~rnd, sets)
            nw, nr = net(without), net(random_)
            hurt = {m: nw[m] - nr[m] for m in have}
            verdict = {"removal_beats_random": bool(have and all(h >= a.group_min_hurt for h in hurt.values())),
                       "served_+0.05": bool(have and all(jserved[m]["net"][m] <= a.target and
                                                         jserved[m]["net"][EN] <= a.target for m in have)),
                       "members_without_held": [m for m in mem if m not in held],
                       **({"members_skipped": {m: skipped[m] for m in mem if m in skipped}}
                          if any(m in skipped for m in mem) else {}), "on": "dCE net of the path offset"}
            got[key] = put(jj, {"arm": "joint", "key": key, "sets": sets, "members": mem, "units": int(J.sum()),
                                "without": without, "random": random_, "hurt_minus_random": hurt,
                                "served": {m: jserved[m]["net"] for m in have}, "verdict": verdict,
                                "wall_s": round(time.time() - t0, 1)})
            log(f"{key}: units {got[key]['units']:,}; {verdict}")
        extra_md += [f"\nJoint tiles ({a.joint_partition}, floor {jprep['joint'].get('floor')}; the {len(top)} "
                     "largest): removal vs a random same-count removal, and served = the variant's core + family "
                     "tiles + ancestors / subtree + box + pairs + every joint tile naming the member, an ancestor or a "
                     "descendant.\n",
                     "| joint tile | units | members: without / random | served member / eng | beats random | "
                     "+0.05 served |", "|---|---|---|---|---|---|"]
        for name in top:
            r = got[f"joint:{name}"]
            have = [m for m in r["members"] if m in r["served"]]
            nw, nr = net(r["without"]), net(r["random"])
            ns = len(r["verdict"].get("members_skipped") or {})
            extra_md.append(f"| {name}{f' ({ns} skipped)' if ns else ''} | {r['units']:,} | "
                            + ", ".join(f"{mark(m)} {nw[m]:+.3f} / {nr[m]:+.3f}" for m in have) + " | "
                            + ", ".join(f"{mark(m)} {r['served'][m][m]:+.3f} / {r['served'][m][EN]:+.3f}" for m in have)
                            + f" | {'yes' if r['verdict']['removal_beats_random'] else 'NO'} | "
                              f"{'yes' if r['verdict']['served_+0.05'] else 'NO'} |")
    if notes:
        extra_md += ["\nHeld-set notes (\u2020 in the tables):"] + [f"- {k}: {v}" for k, v in sorted(notes.items())]
    rows = []
    for x in bj.read_text().split("\n") if bj.exists() else []:
        try:
            r = json.loads(x)
        except ValueError:
            continue
        if r.get("settings") == settings:
            rows.append(r)
    meta = {"checkpoint": host.checkpoint, "fingerprint": host.fingerprint, "mlx2_source_revision": O.git_rev(),
            "partition": str(pd), "rule": prep.get("rule"), "docs": a.docs, "held_tokens": htok,
            "held": {k: {"spec": s, "docs_loaded": len(held[k]["held"])} for k, s in
                     ((x.partition("=")[0], x) for x in a.held)}, "path_offset": off,
            "thresholds": {"min_own": a.min_own, "own_x": a.own_x, "max_unrelated": a.max_unrelated, "target": a.target},
            "watchdog": {"swap_growth_gb": a.swap_growth_gb, "waiver_note": a.waiver_note or None,
                         "limit_gb": a.watchdog_gb, "every_s": a.watchdog_every_s, "min_avail_gb": a.min_avail_gb,
                         "compressor_growth_gb": a.compressor_growth_gb,
                         **{k: v for k, v in getattr(O, "WATCH", {}).items() if k.startswith(("peak_", "observed_"))}},
            "wired": wired[0],
            "mlx_peak_gb": O.peak_gb(), "status": "finished"}
    if skipped:
        meta["skipped"] = skipped
    if EBOX:
        meta["english_box"] = {"name": EBOX, "closed_for": ECLOSED, "arm_run": run_english}
    if (prep.get("detach") or {}).get("share"):
        meta["detach_share"] = prep["detach"]["share"]
    if jz is not None:
        meta.update(joint_partition=a.joint_partition, joint_top=a.joint_top, joint_floor=jprep["joint"].get("floor"))
    if arms != ["boxes"] or notes:
        meta.update(arms=arms, held_notes=notes, arm_thresholds={"target": a.target, "group_min_hurt":
                                                                  a.group_min_hurt},
                    unclaimed=str(upath) if "unclaimed" in arms else None)
    write_meta(meta)
    lines = [f"Battery on {host.checkpoint}, partition {pd.name} ({prep.get('rule')}); held dCE, {a.docs} docs per set.\n",
             "| box | units | own without | unrelated without | random own / eng | core+box own / eng | exclusive | +0.05 served |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['box']} | {r['units']:,} | {r['without'][r['box']]:+.3f} | "
                     + ", ".join(f"{u} {r['without'][u]:+.3f}" for u in r["unrelated"])
                     + f" | {r['random'][r['box']]:+.3f} / {r['random'][EN]:+.3f} | {r['core_plus'][r['box']]:+.3f} / "
                       f"{r['core_plus'][EN]:+.3f} | {'yes' if r['verdict']['exclusive'] else 'NO'} | "
                       f"{'yes' if r['verdict']['served_+0.05'] else 'NO'} |")
    if "boxes" not in arms:
        lines = lines[:1]
    lines += extra_md
    if skipped:
        lines += ["\nSkipped boxes (in the partition, left out of every arm):"] + \
            [f"- {k}: {v}" for k, v in sorted(skipped.items())]
    (out / "battery.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
