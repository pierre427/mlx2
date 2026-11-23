"""Removal check: the stripping bar, full model minus the box (mapping-workflow s10; child process, one model load).

    python -m mlx2.experimental.amoeba.child_removal --model M --core CORE.npz[:core] \
        --cut BOX_CUT.npz[:core] [--cut ...] --nontarget-val F --nontarget-held F --target-val F --target-held F \
        --out DIR [--moe-units neuron] [--cpu]

A box is the units a cut keeps beyond the core (``cut - core``: the carve's ``tier_<ranking>_size<f>.npz`` files
from ``--size-targets``/``--box-fractions``, or ``export_core.npz``). Removing it serves every other unit (the core
plus the remainder): the zero-ablation mask ``keep = all units - box``, the same law as the serving overlay
``amoeba_mask``. For every cut, on VALIDATION rows only:

  non-target   KL(full || full - box) on non-target text (English prose, code, math), per class and pooled over rows
  target       the same KL on the box's own text (zh/ja/hi): the box's effect on what it holds
  random       a same-count random removal (per layer, from the units outside the core; seeded): its non-target and
               target KL, the specificity control

The export cut is the LARGEST cut with non-target val KL <= --val-margin (default +0.03: +0.05 minus the measured
val->held gap; pass the measured margin), target val KL >= --specificity x the random removal's target KL and at least
--floor above it. Held is then scored exactly ONCE, on that cut only (``held.json``; a ledger keyed by the box mask's
and the held rows' digests, in the run root, refuses a second held scoring of the same box from any out dir; val
and held must be disjoint by document id and text hash): it passes when the pooled non-target held KL <= --bar (+0.05, Pierre 2026-11-16). A miss is reported
as a miss; the fix is better data and more neurons, never a wider bar or a held re-cut.

Rows (jsonl): ``text`` and ``domain`` (the class: e.g. ``eng``, ``code``, ``math``, ``lang.cmn``), optional ``prompt``
(an on-policy continuation row: its last --seq-len tokens, scored from --keep-prompt tokens before the continuation),
``doc_id``. A document row gives --windows windows of --seq-len tokens at seeded random offsets across the whole
document; a ``code``-class document is cut at its first line of code. Rows too short for a window are counted and
skipped. Every scored position of a window counts; a row's KL is the mean over its scored positions; a class's KL is
the mean over its rows; pooled = the row-weighted mean over classes (the carve's pooling).

Outputs in DIR: ``val.json`` (every cut), ``chosen.json``, ``held.json``, ``box_mask.npz`` (key ``box``: the chosen
cut's box, ``[L, max width]`` uint8, the input of ``python -m mlx2.runtime.amoeba_neuron_mask export``).
Experimental, unqualified; every receipt names ``<artifact>@<fingerprint>`` and the mlx2 commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

SCHEMA = "mlx2.amoeba-removal.v1"


def _read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def _mask(spec, key_default="core"):
    path, _, key = str(spec).partition(":")
    with np.load(Path(path).expanduser()) as z:
        return np.asarray(z[key or (key_default if key_default in z.files else z.files[0])]).astype(np.float32) > 0


def windows(host, rows, seq_len, nwin, seed, keep_prompt, code_prefixes=("code",)):
    """[(class, ids, loss_from)] with the carve's data rules (module docstring); and the skipped-row counts."""
    from research.amoeba.carve.corpus import cut_at_first_code
    out, skipped = [], {}
    for r in rows:
        cls = str(r.get("domain", "text"))
        if r.get("prompt") is not None:
            ids = host.encode(r["text"])
            n_prompt = len(host.encode(r["prompt"]))
            if len(ids) - n_prompt < 8 or len(ids) < seq_len:
                skipped[cls] = skipped.get(cls, 0) + 1
                continue
            cut = len(ids) - seq_len
            out.append((cls, ids[cut:], max(1, n_prompt - keep_prompt - cut)))
            continue
        text = cut_at_first_code(r["text"]) if cls.startswith(tuple(code_prefixes)) else r["text"]
        ids = host.encode(text)
        if len(ids) < seq_len:
            skipped[cls] = skipped.get(cls, 0) + 1
            continue
        span = len(ids) - seq_len
        doc = r.get("doc_id") or hashlib.sha256(r["text"].encode()).hexdigest()[:16]
        h = int.from_bytes(hashlib.sha256(f"{seed}:{cls}:{doc}".encode()).digest()[:8], "big")
        offs = sorted({int(o) for o in np.random.default_rng(h).integers(0, span + 1, size=min(nwin, span + 1))})
        out += [(cls, ids[o:o + seq_len], 1) for o in offs]
    return out, skipped


class Scorer:
    """KL(full || keep-masked) per window on one loaded carve host, streamed: memory is O(batch), not O(windows).

    Per batch of windows the full model's float32 logits are computed once and every requested arm (each a keep
    mask) is scored against them; nothing is kept between batches (the full model is recomputed for each batch, so
    the KL is exact: no top-k or bf16 storage). A batch holds bs x (T - 1) x V float32 for the full model plus one
    arm at a time (Hy4: 4 x 127 x 120,832 x 4 B = 0.23 GiB each)."""

    def __init__(self, host, bs=4):
        self.host, self.bs = host, bs
        self.peak_live_batches = 0

    def _layout(self, keep):
        import mlx.core as mx
        return [mx.array(keep[l, :n].astype(np.float32)) for l, n in enumerate(self.host.widths)]

    def kl_many(self, wins, keeps):
        """{name: per-window mean KL over scored positions (numpy)} for every keep mask in ``keeps``."""
        import mlx.core as mx
        from research.amoeba.carve.carving import kl_rows
        h = self.host
        cores = {name: self._layout(k) for name, k in keeps.items()}
        out = {name: np.zeros(len(wins)) for name in keeps}
        for i in range(0, len(wins), self.bs):
            by_len = {}
            for j, (_, ids, lf) in enumerate(wins[i:i + self.bs]):
                by_len.setdefault(len(ids), []).append((i + j, ids, lf))
            for T, items in by_len.items():
                ids = mx.array([x[1] for x in items])
                pos = list(range(T - 1))
                h.ctl.reset()
                full = h.logits_at(ids, pos)
                mx.eval(full)
                for name, core in cores.items():
                    h.ctl.reset()
                    h.ctl.core, h.ctl.sink_layers = core, 0
                    lm = h.logits_at(ids, pos)
                    h.ctl.reset()
                    k = np.array(kl_rows(full, lm))                                  # [B, T-1]
                    del lm
                    for b, (idx, _, lf) in enumerate(items):
                        out[name][idx] = float(k[b, max(0, lf - 1):].mean())
                del full
            mx.clear_cache()
        return out

    def kl(self, wins, keep, tag=None):
        return self.kl_many(wins, {"x": keep})["x"]


def _doc_keys(rows):
    """Document ids and content hashes of a row set (for the val/held disjointness check)."""
    ids, shas = set(), set()
    for r in rows:
        if r.get("doc_id"):
            ids.add(str(r["doc_id"]))
        ids.add(str(r.get("group"))) if r.get("group") else None
        shas.add(hashlib.sha256(r["text"].encode()).hexdigest())
    return ids, shas


def check_disjoint(val_rows, held_rows):
    """Refuse when a held document is also a val document (same doc id / group, or the same text)."""
    vi, vs = _doc_keys(val_rows)
    hi, hs = _doc_keys(held_rows)
    both_ids, both_text = vi & hi, vs & hs
    if both_ids or both_text:
        raise SystemExit(f"removal: val and held share {len(both_ids)} document ids and {len(both_text)} texts; held "
                         "must be documents val never saw")


def _digest_rows(rows):
    h = hashlib.sha256()
    for t in sorted(hashlib.sha256(r["text"].encode()).hexdigest() for r in rows):
        h.update(t.encode())
    return h.hexdigest()


LEDGER_NAME = "removal-held-ledger.jsonl"


def held_doc_ids(rows) -> list:
    """Per-row document keys of the held rows (doc id, else group, else the text hash), hashed, sorted."""
    out = set()
    for r in rows:
        k = r.get("doc_id") or r.get("group") or hashlib.sha256(r["text"].encode()).hexdigest()
        out.add(hashlib.sha256(str(k).encode()).hexdigest()[:16])
    return sorted(out)


def default_ledger(held_paths) -> Path:
    """The held data's own ledger: next to the held files (their common directory), absolute."""
    import os
    dirs = {str(Path(p).expanduser().resolve().parent) for p in held_paths}
    return Path(os.path.commonpath(sorted(dirs))) / LEDGER_NAME


def _ledger_check(ledger: Path, key: dict):
    """Held is scored once per box on its held data, across every out dir: a ledger entry with the same box digest
    and the same held text digest, or ANY shared held document, refuses (dropping or adding a held row does not
    reopen held)."""
    if ledger.exists():
        docs = set(key["held_doc_ids"])
        for line in ledger.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("box_sha256") != key["box_sha256"]:
                    continue
                shared = docs & set(r.get("held_doc_ids") or ())
                if r.get("held_sha256") == key["held_sha256"] or shared:
                    raise SystemExit(f"removal: held was already scored for this box ({r.get('out')} at "
                                     f"{r.get('time')}; {len(shared)} held documents in common; ledger {ledger}); "
                                     "held is scored exactly once")


def summarize(wins, kl):
    by = {}
    for (cls, _, _), v in zip(wins, kl):
        by.setdefault(cls, []).append(float(v))
    per = {c: float(np.mean(v)) for c, v in sorted(by.items())}
    rows = {c: len(v) for c, v in sorted(by.items())}
    tot = sum(rows.values())
    return {"per_class": per, "rows": rows, "pooled": (sum(per[c] * rows[c] for c in per) / tot) if tot else None}


def random_like(box, core, widths, seed):
    """A same-count random removal: per layer, as many units as the box, drawn from the units outside the core."""
    rng = np.random.default_rng(seed)
    R = np.zeros_like(box)
    for l, n in enumerate(widths):
        free = np.flatnonzero(~core[l, :n])
        k = min(len(free), int(box[l, :n].sum()))
        if k:
            R[l, rng.choice(free, k, replace=False)] = True
    return R


def keep_without(box, widths):
    valid = np.zeros_like(box)
    for l, n in enumerate(widths):
        valid[l, :n] = True
    return valid & ~box


def run(host, *, core, cuts, nontarget_val, nontarget_held, target_val, target_held, out, seq_len=128, windows_per_doc=2,
        window_seed=0, keep_prompt=32, val_margin=0.03, bar=0.05, specificity=2.0, floor=0.01, seed=0, bs=4,
        identity=None, log=print, ledger=None, held_paths=None, ledger_override=None):
    """The check on a loaded host (module docstring). ``cuts``: {name: [L, W] bool keep mask of core + box}.
    ``ledger``: the held ledger. With ``held_paths`` (the held files) the ledger is the one next to them
    (``default_ledger``); another ledger refuses unless ``ledger_override`` gives the reason, which is recorded.
    Without ``held_paths`` (direct calls), ``ledger`` is required."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "held.json").exists():
        raise SystemExit(f"{out}/held.json exists: held is scored exactly once (use a fresh --out for a new round)")
    if held_paths:
        home = default_ledger(held_paths)
        if ledger and Path(ledger).expanduser().resolve() != home and not ledger_override:
            raise SystemExit(f"removal: --ledger {ledger} is not the held data's ledger {home}; pass "
                             "--ledger-override REASON to use another one")
        ledger = Path(ledger).expanduser().resolve() if ledger else home
    elif not ledger:
        raise SystemExit("removal: no held ledger (pass held_paths or ledger)")
    ledger = Path(ledger).expanduser().resolve()
    check_disjoint(list(nontarget_val) + list(target_val), list(nontarget_held) + list(target_held))
    widths = list(host.widths)
    L, W = len(widths), max(widths)
    if core.shape != (L, W):
        raise SystemExit(f"core mask shape {core.shape} != the host's [L, max width] {(L, W)}")
    total = int(sum(widths))
    sc = Scorer(host, bs=bs)
    data = {}
    for name, rows in (("nontarget_val", nontarget_val), ("target_val", target_val)):
        data[name] = windows(host, rows, seq_len, windows_per_doc, window_seed, keep_prompt)
        if not data[name][0]:
            raise SystemExit(f"removal: no {name} window reaches {seq_len} tokens; nothing to score")
    res = []
    for i, (name, cut) in enumerate(sorted(cuts.items(), key=lambda kv: int(kv[1].sum()))):
        if cut.shape != (L, W):
            raise SystemExit(f"cut {name}: shape {cut.shape} != {(L, W)}")
        if (core & ~cut).any():
            raise SystemExit(f"cut {name} drops core units: a box cut must contain the core")
        box = cut & ~core
        rnd = random_like(box, core, widths, seed + i)
        rec = {"cut": name, "box_units": int(box.sum()), "box_fraction": box.sum() / total,
               "box_fraction_of_noncore": box.sum() / max(1, total - int(core.sum()))}
        for part in ("nontarget_val", "target_val"):       # one full-model pass per batch scores both arms
            wins = data[part][0]
            k = sc.kl_many(wins, {"box": keep_without(box, widths), "random": keep_without(rnd, widths)})
            for arm in ("box", "random"):
                rec[f"{arm}_{part}"] = summarize(wins, k[arm])
        b_nt, b_t = rec["box_nontarget_val"]["pooled"], rec["box_target_val"]["pooled"]
        r_t = rec["random_target_val"]["pooled"]
        rec["meets_val_margin"] = b_nt <= val_margin
        rec["specific"] = b_t >= specificity * r_t and (b_t - r_t) >= floor
        rec["eligible"] = rec["meets_val_margin"] and rec["specific"]
        log(f"removal {name}: box {rec['box_fraction']:.4f} of units, non-target val KL {b_nt:.4f}, target val KL "
            f"{b_t:.4f} (random {r_t:.4f}), eligible {rec['eligible']}")
        res.append(rec)
    base = {"schema": SCHEMA, **(identity or {}), "units": total, "core_units": int(core.sum()),
            "rule": {"val_margin_nats": val_margin, "bar_nats": bar, "specificity_x_random": specificity,
                     "floor_nats": floor, "metric": "KL(full || full - box), mean over scored positions per row, "
                     "row-weighted over classes"},
            "data": {k: {"windows": len(v[0]), "skipped_rows": v[1]} for k, v in data.items()},
            "seq_len": seq_len, "windows_per_doc": windows_per_doc, "window_seed": window_seed}
    (out / "val.json").write_text(json.dumps({**base, "cuts": res}, indent=1))
    ok = [r for r in res if r["eligible"]]
    if not ok:
        chosen = {**base, "chosen": None, "reason": "no cut meets the val margin with specificity: better data and "
                  "more neurons next round (the bar is never relaxed)"}
        (out / "chosen.json").write_text(json.dumps(chosen, indent=1))
        return chosen
    pick = max(ok, key=lambda r: r["box_units"])
    box = cuts[pick["cut"]] & ~core
    np.savez_compressed(out / "box_mask.npz", box=box.astype(np.uint8))
    chosen = {**base, "chosen": pick["cut"], "box_units": pick["box_units"], "box_fraction": pick["box_fraction"],
              "largest_eligible": True, "val": pick}
    (out / "chosen.json").write_text(json.dumps(chosen, indent=1))
    # held, exactly once per (box, held data) across every out dir (the ledger), on the chosen cut only
    key = {"box_sha256": hashlib.sha256(np.ascontiguousarray(box.astype(np.uint8)).tobytes()).hexdigest(),
           "held_sha256": _digest_rows(list(nontarget_held) + list(target_held)),
           "held_doc_ids": held_doc_ids(list(nontarget_held) + list(target_held))}
    _ledger_check(ledger, key)
    with open(ledger, "a") as f:                       # recorded before scoring: a crash mid-held still counts
        f.write(json.dumps({**key, "out": str(out.resolve()), "cut": pick["cut"],
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ledger_override": ledger_override,
                            **(identity or {})}) + "\n")
    held = {}
    rnd = random_like(box, core, widths, seed + 1000)
    for name, rows in (("nontarget_held", nontarget_held), ("target_held", target_held)):
        wins, skipped = windows(host, rows, seq_len, windows_per_doc, window_seed, keep_prompt)
        k = sc.kl_many(wins, {"box": keep_without(box, widths), "random": keep_without(rnd, widths)})
        held[name] = {"box": summarize(wins, k["box"]), "random": summarize(wins, k["random"]),
                      "windows": len(wins), "skipped_rows": skipped}
    nt = held["nontarget_held"]["box"]["pooled"]
    rec = {**base, "chosen": pick["cut"], "box_units": pick["box_units"], "box_fraction": pick["box_fraction"],
           "held": held, "nontarget_held_kl": nt, "target_held_kl": held["target_held"]["box"]["pooled"],
           "passes_bar": nt is not None and nt <= bar, "scored_once": True,
           "val_held_gap": None if nt is None else nt - pick["box_nontarget_val"]["pooled"]}
    (out / "held.json").write_text(json.dumps(rec, indent=1))
    log(f"removal held (once): {pick['cut']} non-target KL {nt:.4f} -> {'PASS' if rec['passes_bar'] else 'MISS'} "
        f"(bar {bar}); target KL {rec['target_held_kl']:.4f}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mlx2.experimental.amoeba.child_removal")
    ap.add_argument("--model", required=True)
    ap.add_argument("--core", required=True, metavar="NPZ[:KEY]")
    ap.add_argument("--cut", action="append", required=True, metavar="NPZ[:KEY]",
                    help="a box cut (core + box keep mask); repeatable (tier_*_size*.npz, export_core.npz)")
    for k in ("nontarget-val", "nontarget-held", "target-val", "target-held"):
        ap.add_argument(f"--{k}", required=True, action="append", metavar="JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--moe-units", choices=["expert", "neuron", "block64"], default="neuron")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--window-seed", type=int, default=0)
    ap.add_argument("--keep-prompt", type=int, default=32)
    ap.add_argument("--val-margin", type=float, default=0.03)
    ap.add_argument("--bar", type=float, default=0.05)
    ap.add_argument("--specificity", type=float, default=2.0)
    ap.add_argument("--floor", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--ledger", default="", help="held ledger (default: next to the held files; another one "
                    "needs --ledger-override)")
    ap.add_argument("--ledger-override", default="", metavar="REASON", help="why another ledger is used (recorded)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    if a.bar > 0.05:
        raise SystemExit("--bar over +0.05 nats: the bar is never relaxed (Pierre 2026-11-16)")
    import mlx.core as mx
    if a.cpu:
        mx.set_default_device(mx.cpu)
    from research.amoeba.carve.host import Host
    host = Host(a.model, moe_units=a.moe_units)
    rows = {k: [r for f in getattr(a, k.replace("-", "_")) for r in _read_jsonl(f)]
            for k in ("nontarget-val", "nontarget-held", "target-val", "target-held")}
    cuts = {Path(c.partition(":")[0]).stem: _mask(c) for c in a.cut}
    ident = {"checkpoint": host.checkpoint, "fingerprint": host.fingerprint, "mlx2_commit": _commit(),
             "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    run(host, core=_mask(a.core), cuts=cuts, nontarget_val=rows["nontarget-val"],
        nontarget_held=rows["nontarget-held"], target_val=rows["target-val"], target_held=rows["target-held"],
        out=a.out, seq_len=a.seq_len, windows_per_doc=a.windows, window_seed=a.window_seed,
        keep_prompt=a.keep_prompt, val_margin=a.val_margin, bar=a.bar, specificity=a.specificity, floor=a.floor,
        bs=a.bs, identity=ident, ledger=a.ledger or None, held_paths=list(a.nontarget_held) + list(a.target_held),
        ledger_override=a.ledger_override or None)


def _commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=9", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parent).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
