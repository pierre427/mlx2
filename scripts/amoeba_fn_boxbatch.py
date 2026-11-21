#!/usr/bin/env python3
"""Carve a batch of FN boxes in ONE process (one model load; coordinator 2026-11-16, lever (b)). Inside a GPU window.

For each box of --boxset (amoeba_fn_boxset_build.py output, with the GPU stage's continuations merged into
corpus.jsonl / val_full.jsonl / held_full.jsonl), runs research.amoeba.carve's pipeline with the loaded Host:
inspect (gates) + mean + rank + bisect, T2 fixed, the fixed data defaults (128-token windows, 2 per document), the
export point chosen on the box's val documents at --export-target (+0.03 until the val->held gap is measured), the
held check once (Pierre: +0.05 on held is the target; a miss is reported as a miss, never re-cut against held).
Per box: <out>/box-<name>/ (carve receipt), one row in <out>/boxes.md, and <out>/verdicts.jsonl
({box, experts, export_val, export_held, held_meets_0.05, val_held_gap, gate, wall_s}). Boxes already bisected are
skipped (resumable); a box whose carve raises is recorded and the batch goes on, then the batch exits 1 (the stage's
return code must not read as success when carves failed).

Usage: amoeba_fn_boxbatch.py --model M --boxset DIR --t2 T2.npz --out DIR [--boxes a,b] [carve args after --]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--boxset", required=True)
    ap.add_argument("--t2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boxes", default="")
    ap.add_argument("--export-target", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--val-windows", type=int, default=0, help="windows per val document (0 = the carve windows, 2)")
    ap.add_argument("--corpus-name", default="corpus.jsonl",
                    help="the carve corpus file in each box dir (ablation arms: corpus_text.jsonl = probe text only, "
                         "corpus_greedy.jsonl = text + greedy continuations; corpus.jsonl = text + 1 greedy + 3 T=0.7)")
    ap.add_argument("--moe-units", default="expert", choices=["expert", "neuron"],
                    help="neuron: boxes carved as neurons within experts (Pierre step 6, after the neuron smoke); the "
                         "expert-unit T2 is expanded exactly (--fixed-expert-units)")
    ap.add_argument("--stop-at", type=float, default=None, help="clean stop (epoch s): no box starts after it, a box in "
                    "rank checkpoints and stops (resume the batch later); exit 75")
    a, extra = ap.parse_known_args()
    extra = [x for x in extra if x != "--"]
    from research.amoeba.carve import __main__ as D
    from research.amoeba.carve import ops as O
    from research.amoeba.carve.host import Host
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bs = Path(a.boxset)
    spec = json.loads((bs / "boxes.json").read_text())
    names = [b for b in a.boxes.split(",") if b] or [b["box"] for b in spec["boxes"]]
    log_main = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)    # noqa: E731
    O.footprint_watchdog(92, log_main)
    O.setup_mlx(88, 2)
    host = Host(a.model, log=log_main, moe_units=a.moe_units)
    import os
    tables = getattr(getattr(host, "adapter", None), "_tables", None)
    if not tables or Path(os.environ.get("MLX_QWEN4_PLE_NVME", "")).expanduser().resolve() != \
            Path(a.model).expanduser().resolve() / "ple_rows.bin":
        raise SystemExit("PLE offload is not active (FN rule: assert it)")
    t2 = np.load(a.t2)["core"] > 0
    if a.moe_units == "neuron" and t2.shape != (host.L, max(host.widths)):
        t2 = D.expand_expert_mask(t2.astype(np.float32), host) > 0      # T2 in this host's (neuron) layout
    table = out / "boxes.md"
    if not table.exists():
        table.write_text(f"Units: {a.moe_units}. T2-alone held is measured in this unit mode and is not comparable "
                         f"across unit modes (neuron mode recomputes kept experts piecewise on Metal).\n\n"
                         f"Verdict (coordinator 2026-11-16): **pass** = held <= +0.05 AND the random same-count box "
                         f"is informatively worse (random held >= box held + 0.05, or >= 2x a positive box held); "
                         f"held <= +0.05 without that = **uninformative** (a fail: the box is about as good as random).\n\n"
                         f"| box | kind | kept {'neurons' if a.moe_units == 'neuron' else 'experts'} | val rows (text windows + continuations) | T2 alone held | export val dCE | export held dCE | +0.05 on held | val→held gap | "
                         f"held text / greedy / t0.7 | random same-count (export size; {a.moe_units}-level) | gate | wall s |\n"
                         "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    kinds = {b["box"]: b["kind"] for b in spec["boxes"]}
    errors = []
    bflags = {b["box"]: [f for f in ("held_synthetic", "held_lab") if b.get(f)] for b in spec["boxes"]}
    for name in names:
        bd = bs / name
        d = out / f"box-{name}"
        try:
            prev = json.loads((d / "carve.json").read_text())
        except FileNotFoundError:
            prev = {}
        if prev and (prev.get("identity") or {}).get("moe_units", "expert") != a.moe_units:
            raise SystemExit(f"box {name}: {d} was carved in {(prev.get('identity') or {}).get('moe_units', 'expert')} "
                             f"units, this batch is {a.moe_units}: use a fresh --out (never mix layouts)")
        done = prev.get("stages_done", [])
        if "bisect" in done:
            log_main(f"box {name}: done, skipped")
            continue
        argv = ["--model", a.model, "--out", str(d), "--no-admit", "--stages", "inspect,mean,rank,bisect",
                "--ablations", "zero", "--steps", str(a.steps), "--bs", "2", "--lr", "0.1",
                "--corpus", f"jsonl:{bd / a.corpus_name}", "--held-file", str(bd / "held_full.jsonl"),
                "--val-file", str(bd / "val_full.jsonl"), "--val-windows", str(a.val_windows), "--held-len", "128", "--seq-len", "128", "--windows", "2",
                # no held bisect (held is the test set, review 2026-11-16): the only cut is the export cut on val (full
                # set), the random same-count box is taken at the export count and both are scored on held once
                "--n-mean", "12", "--n-fit", "144", "--n-held", "0", "--targets", "",
                "--export-target", str(a.export_target), "--bisect-iters", "12", "--bisect-subsample", "1.0",
                "--bisect-tol", "0.005", "--fixed", f"{a.t2}:core", "--assert-ple-offload", "--watchdog-gb", "92",
                "--memory-gb", "88", "--cache-gb", "2", "--train-peak-cap-gb", "86", "--moe-units", a.moe_units,
                "--pool-domains"] + (["--fixed-expert-units"] if a.moe_units == "neuron" else []) + extra        # pooled held/val verdicts (coordinator 2026-11-16)
        if d.exists():
            argv.append("--resume")
        if a.stop_at is not None:
            if time.time() >= a.stop_at:
                log_main(f"--stop-at reached before box {name}: clean stop")
                sys.exit(75)
            argv += ["--stop-at", str(a.stop_at)]
        t0 = time.time()
        log_main(f"box {name} start")
        rec = {"box": name, "kind": kinds.get(name), "flags": bflags.get(name, [])}
        from research.amoeba.carve.carving import CleanStop
        try:
            D.main(argv, host_factory=lambda log: host)
        except (CleanStop, KeyboardInterrupt):
            log_main(f"box {name}: clean stop / interrupt; the batch resumes from this box")
            raise
        except (Exception, SystemExit) as e:             # gate refusals (SystemExit) and errors: record, go on
            rec["error"] = f"{type(e).__name__}: {e}"[:400]
            if not isinstance(e, SystemExit):
                rec["traceback"] = traceback.format_exc()[-1500:]
            try:                                         # a failed box must not leave the host in training state
                import mlx.core as mx
                host.train_mode(False)
                host.ctl.reset()
                mx.clear_cache()
            except Exception:
                pass
        rec["wall_s"] = round(time.time() - t0, 1)
        try:
            r = json.loads((d / "carve.json").read_text())
        except FileNotFoundError:
            r = {}
        ec = r.get("export_core") or {}
        g = ((r.get("inspect") or {}).get("paths") or {}).get("layerwise_grad")
        rec["gate"] = g.get("ok") if isinstance(g, dict) else g
        if ec:
            m = (np.load(d / "export_core.npz")["core"] > 0) & ~t2
            hv, vv = ec["held"]["mean_d_ce"], ec["spare"]["mean_d_ce"]
            rec.update({"kept_units": int(m.sum()), "units": a.moe_units, "export_val": vv, "export_held": hv,
                        "held_meets_0.05": hv is not None and hv <= 0.05,
                        "val_held_gap": None if hv is None or vv is None else hv - vv,
                        "held_split": {k: v for k, v in ec["held"].items() if k not in ("mean_d_ce", "domain_rows")},
                        "held_as_exported_sink0": (ec.get("held_as_exported_sink0") or {}).get("mean_d_ce"),
                        "random_same_count_held": (ec.get("random_same_count_held") or {}).get("mean_d_ce"),
                        "t2_alone_held": (r.get("fixed_only_held") or {}).get("mean_d_ce"),
                        "held_weighting": "pooled over held rows (probe windows + 1 greedy + 3 T=0.7 per prompt)",
                        "val_windows": sum(1 for x in (d / "inputs.jsonl").read_text().split("\n")
                                           if x.strip() and json.loads(x).get("split") == "val")
                        if (d / "inputs.jsonl").exists() else None,
                        "domain_over_0.10": sorted(k for k, v in ec["held"].items()
                                                   if k not in ("mean_d_ce", "domain_rows") and v > 0.10)})
        if ec:
            rr, hv = rec.get("random_same_count_held"), rec.get("export_held")
            informative = (rr is not None and hv is not None and
                           (rr >= hv + 0.05 or (hv > 0 and rr >= 2 * hv)))
            rec["informative"] = informative
            rec["verdict"] = ("pass" if rec["held_meets_0.05"] and informative else
                              "uninformative" if rec["held_meets_0.05"] else "miss")
        with (out / "verdicts.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        f = lambda x: "—" if x is None else f"{x:.3f}"   # noqa: E731
        hs = rec.get("held_split", {})
        gr = {k.rsplit(":", 1)[-1]: v for k, v in hs.items() if k.startswith("gen:")}
        with table.open("a") as fh:
            fh.write(f"| {name}{' (' + ', '.join(rec['flags']) + ')' if rec['flags'] else ''} | {rec['kind']} | {rec.get('kept_units', '—')} | {rec.get('val_windows', '—')} | {f(rec.get('t2_alone_held'))} | {f(rec.get('export_val'))} | "
                     f"{f(rec.get('export_held'))} | {({'pass': '**pass**', 'uninformative': 'UNINFORMATIVE (≈ random)', 'miss': 'MISSES'}[rec['verdict']] if ec else rec.get('error', '—')[:60]) + (' ⚠ domain > 0.10: ' + ','.join(rec['domain_over_0.10']) if rec.get('domain_over_0.10') else '')} | "
                     f"{f(rec.get('val_held_gap'))} | {f(hs.get('box'))} / {f(gr.get('greedy'))} / {f(gr.get('t0.7'))} | "
                     f"{f(rec.get('random_same_count_held'))} | {rec['gate']} | {rec['wall_s']} |\n")
        if "error" in rec:
            errors.append(name)
        log_main(f"box {name} done in {rec['wall_s']} s: {json.dumps({k: rec.get(k) for k in ('kept_units', 'export_held', 'held_meets_0.05', 'error')})}")
    if errors:
        log_main(f"{len(errors)} box(es) failed: {', '.join(errors)} (see verdicts.jsonl)")
        sys.exit(1)


if __name__ == "__main__":
    main()
