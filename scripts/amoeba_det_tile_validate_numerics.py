#!/usr/bin/env python3
"""Validate a Hy4-flash calculator det tile under the opt-in fusions (research tooling, GPU).

Original mlx2 script.  One process loads the checkpoint once, through the
ordinary-route adapter with the tile's fit policy and the tile itself, then
installs every fusion (``hy_v4_flash_fusions.install``) and switches them per
arm in place (``set_active``).  Arms: ``off`` (the reference: every fusion
off, exactly the unfused ops), each fusion alone, ``all``.  For every arm:

probe       the tile's own probe (``read_layer`` capture -> ``rowwise_logit``,
            serving's fp32 arithmetic; decision ``logit > logit(threshold)``)
            over the fit corpus's held splits (``research/amoeba/det_calc.py``
            positive / hard-negative templates and the English corpus, built
            exactly as ``scripts/amoeba_det_calc_cpu.py`` builds them), as
            prefill forwards (the fit's shape), plus a decode-shaped subset:
            each sequence after an English prefix of ``--long-prefix-tokens``
            (past the indexer's 2048-token dense limit, so the indexer and the
            one-row fused GEMMs engage), one token per forward.
served      a fixed set of 3x2-digit multiplications and word problems,
            chat-rendered, greedy, calculator on (the adapter's own
            processors), ``--max-new`` tokens: tokens, text, plans, events.

Per non-off arm against ``off`` (``assess``): probe logit deltas (max/mean
abs), decision flips each way, the minimum margin to the threshold in both
arms, and the pass rule: zero flips, and every row whose logit changed keeps
a margin (in both arms) larger than the largest logit change.  An arm whose
fusion never engaged (its counters stayed 0) fails as ``not_exercised``.
Served differences (plans, events, text) are recorded, not gated: a fused
route may change generated text by near-ties, and then a different expression
is a different plan.

Writes ``<out>/evidence.json`` (atomic; refuses a nonempty ``--out``) naming
the checkpoint ``<dir>@<fingerprint>`` and the mlx2 commit.
``--write-manifest DIR`` copies the tile to DIR with ``validated_numerics``
for every fit-numerics key whose single-key arm passed (and, when ``all`` ran,
only if it passed too) and ``validated_numerics_evidence``.  The source
manifest is never changed unless ``--write-manifest`` names its own directory
and ``--overwrite-source`` is given.

Usage (one bounded GPU window; the model is ~91 GB resident):
  PYTHONPATH=$PWD/src:$PWD ~/Desktop/mlx2/.venv/bin/python scripts/amoeba_det_tile_validate_numerics.py \
      --out <runs>/hy4-calc-fusions-<rev> --write-manifest <runs>/hy4-calc-fusions-<rev>/tile
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCHEMA = "mlx2.amoeba-det-tile-validate-numerics.v1"
MODEL = Path.home() / "mlx-models/Hy4-flash-MLX-4.6bit-mlx2"
CORPUS = Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba/corpus-2026-11-13/corpus.json"
FUSION_ARMS = {
    "off": {},
    "hc_seam_fused": {"hc_seam": True},
    "mla_fused_input_proj": {"mla_inputs": True},
    "engram_fused_proj": {"engram_inputs": True},
    "indexer_fused_scores": {"indexer_scores": True},
    "engram_gate_rounding": {"engram_gate_rounding": "torch"},
    "all": {"hc_seam": True, "mla_inputs": True, "engram_inputs": True, "indexer_scores": True,
            "engram_gate_rounding": "torch"},
}
# The policy key and serving value each single-fusion arm validates (None: not a fit key).
ARM_POLICY = {"hc_seam_fused": ("hc_seam_fused", True), "engram_fused_proj": ("engram_fused_proj", True),
              "indexer_fused_scores": ("indexer_fused_scores", True),
              "engram_gate_rounding": ("engram_gate_rounding", "torch"),
              "mla_fused_input_proj": (None, True)}
WORD_PROBLEMS = [
    ("A shop packs 24 pens in each box. How many pens are in 37 boxes?", 888),
    ("A train travels 86 km every hour. How far does it go in 14 hours?", 1204),
    ("Each crate holds 125 apples. How many apples are in 48 crates?", 6000),
    ("A school orders 312 notebooks and 29 more. How many notebooks is that?", 341),
    ("A farmer plants 17 rows of 63 trees. How many trees does he plant?", 1071),
    ("A theatre has 452 seats and sells 389 tickets. How many seats are empty?", 63),
    ("There are 7 days in a week. How many days are in 156 weeks?", 1092),
    ("A recipe needs 45 grams of sugar per cake. How much sugar do 26 cakes need?", 1170),
]
# Fusion counters (``counters``) that must move in each arm.
ENGAGEMENT = {"off": (), "hc_seam_fused": ("hc_seam_metal",), "mla_fused_input_proj": ("mla_fused_calls",),
              "engram_fused_proj": ("engram_fused_calls",), "indexer_fused_scores": ("indexer_fused_calls",),
              "engram_gate_rounding": ("engram_gate_torch_calls",),
              "all": ("hc_seam_metal", "mla_fused_calls", "engram_fused_calls", "indexer_fused_calls",
                      "engram_gate_torch_calls")}


# ------------------------------------------------------------------ assessment (pure) --

def assess(ref, arm, threshold_logit):
    """Compare one arm's probe logits with the reference arm's, row for row.

    ``ref``/``arm``: 1-D float arrays (same rows).  Decision: ``logit > threshold_logit``
    (serving's rule).  Pass: no flips, and every changed row's margin to the threshold, in
    both arms, exceeds the largest absolute change.
    """
    import numpy as np

    ref = np.asarray(ref, dtype=np.float64)
    arm = np.asarray(arm, dtype=np.float64)
    if ref.shape != arm.shape or ref.ndim != 1:
        raise ValueError("assess needs two 1-D score arrays of the same length")
    if ref.size == 0:
        return {"rows": 0, "passed": False, "reason": "no rows"}
    delta = np.abs(arm - ref)
    open_ref, open_arm = ref > threshold_logit, arm > threshold_logit
    margin_ref, margin_arm = np.abs(ref - threshold_logit), np.abs(arm - threshold_logit)
    changed = delta > 0
    max_delta = float(delta.max())
    flips_on = int((~open_ref & open_arm).sum())
    flips_off = int((open_ref & ~open_arm).sum())
    min_changed = float(np.minimum(margin_ref, margin_arm)[changed].min()) if changed.any() else None
    reason = None
    if flips_on or flips_off:
        reason = f"{flips_on + flips_off} decision flips"
    elif changed.any() and not min_changed > max_delta:
        reason = f"a changed row's margin {min_changed:.3g} does not exceed the max delta {max_delta:.3g}"
    return {"rows": int(ref.size), "changed_rows": int(changed.sum()), "max_abs_delta": max_delta,
            "mean_abs_delta": float(delta.mean()), "flips_closed_to_open": flips_on,
            "flips_open_to_closed": flips_off, "fires_ref": int(open_ref.sum()), "fires_arm": int(open_arm.sum()),
            "min_margin_ref": float(margin_ref.min()), "min_margin_arm": float(margin_arm.min()),
            "min_margin_changed_rows": min_changed, "passed": reason is None, "reason": reason}


def validated_entries(results):
    """``{key: [value]}`` for each fit-numerics arm that passed (gated on ``all`` when it ran)."""
    arms = results.get("arms", {})
    if "all" in arms and not arms["all"].get("passed"):
        return {}
    out = {}
    for arm, (key, value) in ARM_POLICY.items():
        if key is not None and arms.get(arm, {}).get("passed"):
            out[key] = [value]
    return out


def write_manifest_copy(source_dir, dest_dir, validated, evidence, *, overwrite_source=False):
    """Copy the tile (manifest + files) to ``dest_dir`` with ``validated_numerics`` added."""
    source_dir, dest_dir = Path(source_dir).resolve(), Path(dest_dir).resolve()
    if source_dir.is_file():
        source_dir = source_dir.parent
    manifest = json.loads((source_dir / "manifest.json").read_text())
    if dest_dir == source_dir:
        if not overwrite_source:
            raise SystemExit("refusing to rewrite the source manifest in place without --overwrite-source")
    else:
        if dest_dir.exists() and any(dest_dir.iterdir()):
            raise SystemExit(f"refusing to overwrite nonempty {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        for name in manifest["files"]:
            shutil.copy2(source_dir / name, dest_dir / name)
    manifest["validated_numerics"] = {k: list(v) for k, v in sorted(validated.items())}
    manifest["validated_numerics_evidence"] = str(evidence)
    tmp = dest_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    os.replace(tmp, dest_dir / "manifest.json")
    return dest_dir / "manifest.json"


# --------------------------------------------------------------------------- receipts --

def git_rev():
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "src", "research", "scripts"],
                           cwd=ROOT).returncode != 0
    return rev + ("-dirty" if dirty else "")


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


# ------------------------------------------------------------------------------ corpus --

def held_corpus(donor, tok_encode, splits, n_pos, n_neg, max_sequences=None):
    """Sequences of the fit's held splits, built as scripts/amoeba_det_calc_cpu.py builds them."""
    from research.amoeba import det_calc as C

    seeds = {"train": 1, "val": 2, "test": 3}
    ranges = {"train": (0, 40), "val": (40, 70), "test": (70, 130)}
    blob = json.loads(CORPUS.read_text())["seqs"] if CORPUS.is_file() else None
    seqs = []
    for split in splits:
        seed = seeds[split]
        for p, a in C.positive_rows(split, n_pos if split == "train" else n_pos // 2, seed):
            ids, lab = C.encode_positive(donor._tok, p, a)
            seqs.append((f"{split}:arith", ids, lab))
        for kind, rows in C.negative_rows(split, n_neg, seed + 1).items():
            for r in rows:
                ids = tok_encode(r)
                seqs.append((f"{split}:{kind}", ids, [0] * len(ids)))
        if blob is not None:
            lo, hi = ranges[split]
            for dom, rows in blob.items():
                for ids in rows[lo:hi]:
                    ids = donor.reencode(ids)
                    seqs.append((f"{split}:english:{dom}", ids, [0] * len(ids)))
    if max_sequences:
        step = max(1, len(seqs) // max_sequences)
        seqs = seqs[::step][:max_sequences]
    return seqs, blob is not None


def english_prefix(donor, n_tokens):
    blob = json.loads(CORPUS.read_text())["seqs"] if CORPUS.is_file() else None
    ids = []
    if blob is not None:
        for rows in blob.values():
            for row in rows[:40]:
                ids += donor.reencode(row)
                if len(ids) >= n_tokens:
                    return ids[:n_tokens]
    filler = donor.encode(" ".join(["The harbour was quiet in the early morning light."] * 400))
    while len(ids) < n_tokens:
        ids += filler
    return ids[:n_tokens]


def arith_items(n, seed=20261121):
    rng = random.Random(seed)
    items = []
    while len(items) < n:
        a, b = rng.randint(100, 999), rng.randint(10, 99)
        items.append((f"Compute {a} * {b}. Show the multiplication, then give the result.", a * b))
    return items


# ------------------------------------------------------------------------------ arms --

class Runner:
    def __init__(self, donor, tile, gate, log):
        import mlx.core as mx

        self.mx = mx
        self.donor = donor
        self.model = donor.model
        self.layer = int(tile["read_layer"])
        self.gate = gate
        self.log = log

    def _without_tile_reader(self):
        """The served tile's own reader on ``read_layer`` (it stashes only while a calculator
        processor is live) steps aside while the probe capture installs its reader."""
        import contextlib

        readers = self.model.layer_output_readers

        @contextlib.contextmanager
        def scope():
            saved = readers.pop(self.layer, None)
            try:
                yield
            finally:
                readers.pop(self.layer, None)
                if saved is not None:
                    readers[self.layer] = saved
        return scope()

    def _logits_of(self, capture):
        from mlx2.runtime.amoeba_det_tiles import rowwise_logit

        return rowwise_logit(capture, self.gate.w, self.gate.b)

    def probe_prefill(self, seqs):
        import numpy as np

        mx = self.mx
        out = []
        with self._without_tile_reader():
            for _, ids, lab in seqs:
                _, caps, _ = self.donor.forward(mx.array([ids]), record={self.layer}, stop_after=self.layer)
                z = np.array(self._logits_of(caps[self.layer][0])[0].astype(mx.float32))
                keep = np.asarray(lab) >= 0
                out.append(z[keep])
        return np.concatenate(out) if out else np.zeros((0,))

    def probe_decode(self, seqs, prefix):
        """Each sequence after ``prefix``, one token per forward (decode-shaped rows)."""
        import numpy as np

        mx = self.mx
        from mlx2.adapters.hy_v4_flash import det_tile_capture

        readers = self.model.layer_output_readers
        got = []
        out = []
        with self._without_tile_reader():
            readers[self.layer] = lambda layer, streams, _ids, _p: got.append(det_tile_capture(streams))
            for _, ids, lab in seqs:
                cache = self.model.make_cache()
                if prefix:
                    for a in range(0, len(prefix), 2048):
                        mx.eval(self.model.hidden(mx.array([prefix[a:a + 2048]]), cache))
                    got.clear()
                for t in ids:
                    mx.eval(self.model.hidden(mx.array([[int(t)]]), cache))
                scores = [float(self._logits_of(c)[0, -1].item()) for c in got]
                got.clear()
                keep = np.asarray(lab) >= 0
                out.append(np.asarray(scores)[keep])
        return np.concatenate(out) if out else np.zeros((0,))

    def served(self, items, max_new):
        mx = self.mx
        adapter = self.donor.adapter
        tok = adapter.tokenizer
        eos = {t for t in (getattr(tok, "eos_token_id", None), *(getattr(adapter, "eos_ids", None) or ()))
               if isinstance(t, int)}
        rows = []
        for question, expected in items:
            text = self.donor.chat_format.format(q=question)
            prompt = tok.encode(text, add_special_tokens=False)
            procs = adapter.request_logits_processors({"prompt": text}, prompt_length=len(prompt))
            cache = self.model.make_cache()
            ids = list(prompt)
            logits = self.model(mx.array([prompt]), cache)[:, -1]
            new = []
            for _ in range(max_new):
                for p in procs:
                    if hasattr(p, "bind_row"):
                        p.bind_row(0)
                    logits = p(mx.array(ids), logits)
                t = int(mx.argmax(logits[0]).item())
                new.append(t)
                ids.append(t)
                if t in eos:
                    break
                logits = self.model(mx.array([[t]]), cache)[:, -1]
            out_text = tok.decode(new)
            rows.append({"question": question, "expected": expected, "tokens": new, "text": out_text,
                         "correct": str(expected) in out_text.replace(",", ""),
                         "plans": [repr(getattr(p, "plan", None)) for p in procs],
                         "events": [json.loads(json.dumps(getattr(p, "events", []), default=str)) for p in procs]})
        return rows


def counters(model):
    from mlx2.runtime.models import hy_v4_flash_fusions as F

    st = F.status(model)
    eng = [layer.engram_layer for layer in model.model.layers if "engram_layer" in layer]
    return {"hc_seam_metal": st["hc_seam_fused"]["calls"]["metal"],
            "mla_fused_calls": st["mla_fused_input_proj"]["fused_calls"],
            "engram_fused_calls": st["engram_fused_proj"]["fused_calls"],
            "indexer_fused_calls": st["indexer_fused_scores"]["fused_score_calls"],
            "engram_gate_torch_calls": sum(getattr(e, "gate_calls", 0) for e in eng)}


def compare_served(ref_rows, rows):
    diffs = []
    for i, (a, b) in enumerate(zip(ref_rows, rows)):
        d = {k: a[k] != b[k] for k in ("tokens", "plans", "events", "correct")}
        if any(d.values()):
            diffs.append({"item": i, **d})
    return {"items": len(rows), "differing_items": diffs,
            "tokens_equal": all(not d["tokens"] for d in diffs),
            "plans_equal": all(not d["plans"] and not d["events"] for d in diffs),
            "correct_ref": sum(r["correct"] for r in ref_rows), "correct_arm": sum(r["correct"] for r in rows)}


# ------------------------------------------------------------------------------ main --

def main(argv=None):
    global CORPUS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="evidence directory (must be empty or absent)")
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--manifest", default=None, help="tile manifest (default: the checkpoint's packaged tile)")
    ap.add_argument("--base-policy", type=json.loads, default=None,
                    help='fit execution policy JSON (default {"engram_table_format": "iq4_nl"})')
    ap.add_argument("--arms", default=",".join(FUSION_ARMS))
    ap.add_argument("--splits", default="val,test", help="fit-corpus splits (held: val,test)")
    ap.add_argument("--corpus", type=Path, default=CORPUS, help="the fit's English corpus.json")
    ap.add_argument("--n-pos", type=int, default=600)
    ap.add_argument("--n-neg", type=int, default=60)
    ap.add_argument("--max-sequences", type=int, default=None, help="subsample the prefill corpus (smoke)")
    ap.add_argument("--decode-sequences", type=int, default=24)
    ap.add_argument("--long-prefix-tokens", type=int, default=2304)
    ap.add_argument("--arith-items", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--write-manifest", default=None, help="directory for a validated copy of the tile")
    ap.add_argument("--overwrite-source", action="store_true",
                    help="allow --write-manifest to name the source tile's own directory")
    ap.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--min-available-gb", type=float, default=15.0)
    ap.add_argument("--footprint-limit-gb", type=float, default=104.0)
    ap.add_argument("--no-wire", action="store_true")
    ap.add_argument("--no-watchdog", action="store_true", help="tests only: no memory watchdog thread")
    ap.add_argument("--chat-format", default=None,
                    help="str.format template with {q} for a checkpoint without a chat template (tests)")
    args = ap.parse_args(argv)
    CORPUS = args.corpus
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty {out}")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if "off" not in arms:
        arms.insert(0, "off")
    unknown = set(arms) - set(FUSION_ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}; choose from {sorted(FUSION_ARMS)}")
    arms = ["off"] + [a for a in arms if a != "off"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import mlx.core as mx
    import numpy as np

    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    from mlx2.adapters import hy_v4_flash as A
    from mlx2.runtime import amoeba_det_tiles as D
    from mlx2.runtime.models import hy_v4_flash_fusions as F
    from research.amoeba import det_calc as C
    from research.amoeba.det_backend_hy4 import BASE_POLICY, BackendHy4

    log = lambda m: print(f"[validate-numerics] {m}", flush=True)  # noqa: E731
    base = dict(BASE_POLICY if args.base_policy is None else args.base_policy)
    for key in A.FUSION_KEYS:
        if base.get(key) not in (None, A.FUSION_DEFAULTS[key]):
            raise SystemExit(f"the base policy must leave fusions at their defaults ({key})")
    inspected = A.inspect_artifact(str(args.model))
    manifest_path = args.manifest or A.default_det_tile_manifest(inspected["identity"]["fingerprint"])
    if manifest_path is None:
        raise SystemExit("no --manifest and no packaged tile for this checkpoint")
    manifest = D.load_manifest(manifest_path)
    (tile,) = [t for t in manifest["tiles"] if t["kind"] == "calculator"]
    # The tile loads under its fit numerics (fusions off); fusions install after the load.
    donor = BackendHy4(path=args.model, policy={**base, "amoeba_det_tiles": str(manifest_path)},
                       wire=not args.no_wire and args.device == "gpu", log=log, watchdog=not args.no_watchdog,
                       min_available_gb=args.min_available_gb, footprint_limit_gb=args.footprint_limit_gb)
    if donor.adapter.amoeba_det_tiles is None:
        raise SystemExit(f"the tile did not load: {donor.adapter._det_tiles_receipt()}")
    if args.chat_format is not None:
        donor.chat_format = args.chat_format
    if donor.chat_format is None:
        raise SystemExit("the checkpoint renders no chat template (pass --chat-format)")
    C.CHAT = donor.chat_format
    record = F.install(donor.model, hc_seam=True, mla_inputs=True, engram_inputs=True, indexer_scores=True)
    tile_dir = Path(manifest_path).resolve()
    tile_dir = tile_dir if tile_dir.is_dir() else tile_dir.parent
    tensors = mx.load(str(tile_dir / tile["file"]))
    gate = D.ProbeGate(tensors[f"{tile['id']}.probe_w"], tile["gate"]["bias"], tile["gate"]["threshold"])
    runner = Runner(donor, tile, gate, log)
    seqs, english_ok = held_corpus(donor, donor.encode, [s.strip() for s in args.splits.split(",")],
                                   args.n_pos, args.n_neg, args.max_sequences)
    step = max(1, len(seqs) // max(1, args.decode_sequences))
    decode_seqs = seqs[::step][:args.decode_sequences]
    prefix = english_prefix(donor, args.long_prefix_tokens) if args.long_prefix_tokens else []
    items = arith_items(args.arith_items) + list(WORD_PROBLEMS)
    ev = {"schema": SCHEMA, "checkpoint": f"{donor.path.name}@{donor.source_fingerprint}",
          "mlx2_source_revision": git_rev(), "mlx": mx.__version__, "device": args.device,
          "manifest": str(manifest_path), "manifest_sha256": manifest.get("_digest"),
          "tile": {"id": tile["id"], "read_layer": tile["read_layer"], "threshold": gate.threshold,
                   "logit_threshold": gate.logit_threshold},
          "base_policy": base, "row_bounds": record.get("row_bounds"), "qualification": "unqualified",
          "corpus": {"splits": args.splits, "prefill_sequences": len(seqs), "decode_sequences": len(decode_seqs),
                     "english_corpus": str(CORPUS) if english_ok else "missing (arith and hard negatives only)",
                     "long_prefix_tokens": len(prefix)},
          "pass_rule": ("zero probe decision flips (prefill and decode rows), every changed row's margin in "
                        "both arms above the max logit change, and the arm's fusion engaged"),
          "args": vars(args), "arms": {}, "receipt": donor.receipt}
    evidence_path = out / "evidence.json"
    ref = None
    for arm in arms:
        a0 = time.time()
        active = F.set_active(donor.model, **FUSION_ARMS[arm])
        before = counters(donor.model)
        prefill = runner.probe_prefill(seqs)
        decode = runner.probe_decode(decode_seqs, prefix)
        served = runner.served(items, args.max_new)
        after = counters(donor.model)
        engaged = {k: after[k] - before[k] for k in after}
        row = {"active": active, "engaged": engaged, "wall_s": round(time.time() - a0, 1)}
        if arm == "off":
            ref = {"prefill": prefill, "decode": decode, "served": served}
            row.update(prefill_rows=int(prefill.size), decode_rows=int(decode.size),
                       fires_prefill=int((prefill > gate.logit_threshold).sum()),
                       fires_decode=int((decode > gate.logit_threshold).sum()),
                       min_margin_prefill=float(np.abs(prefill - gate.logit_threshold).min()) if prefill.size else None,
                       served=served)
        else:
            row["prefill"] = assess(ref["prefill"], prefill, gate.logit_threshold)
            row["decode"] = assess(ref["decode"], decode, gate.logit_threshold)
            row["served"] = compare_served(ref["served"], served)
            idle = [c for c in ENGAGEMENT[arm] if engaged[c] == 0]
            row["not_exercised"] = idle
            row["passed"] = row["prefill"]["passed"] and row["decode"]["passed"] and not idle
        ev["arms"][arm] = row
        ev["wall_s"] = round(time.time() - t0, 1)
        write_json(evidence_path, ev)
        log(f"arm {arm}: {'reference' if arm == 'off' else ('PASS' if row['passed'] else 'FAIL')} "
            f"({row['wall_s']} s)")
    F.set_active(donor.model)
    ev["validated_numerics"] = validated_entries(ev)
    if args.write_manifest:
        ev["validated_manifest"] = str(write_manifest_copy(
            Path(manifest_path), Path(args.write_manifest), ev["validated_numerics"], evidence_path.resolve(),
            overwrite_source=args.overwrite_source))
    ev["watchdog"] = getattr(donor, "watchdog", None)
    write_json(evidence_path, ev)
    log(f"done: validated {ev['validated_numerics']} -> {evidence_path}")
    return ev


if __name__ == "__main__":
    main()
