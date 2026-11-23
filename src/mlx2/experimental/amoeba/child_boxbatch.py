"""Carve every box of a box set in ONE process (one model load): mapping-workflow step 3.3-3.6 (child process).

Generalised from scripts/amoeba_fn_boxbatch.py (claude/amoeba-27b-80bc6a@52fd0d6c9): the same carve call
(research.amoeba.carve: inspect + mean + rank + bisect, the core fixed, the export cut chosen on the box's
validation documents at the val target, the held check once, the random same-count box scored on held)
with the Flash-Next constants replaced by the build's resource plan and data rules (no model branches).

Per box: <out>/box-<name>/ (the carve run), one line in <out>/verdicts.jsonl, events box_start / box_end.
Verdict (pooled held dCE of core + box vs the full model, probe windows + 1 greedy : 3 T=0.7 by rows):
  pass     held dCE <= the held target AND <= ratio x the random same-count box's held dCE
  miss     otherwise (reported with its numbers; the fix is better data next round, never a held re-cut)
  covered  the carve kept no units beyond the core
The core-alone dCE on the box's val documents is recorded for every box (core_alone_val) as a diagnostic; it never
skips a carve (more boxes, never fewer: mapping-workflow at 77d3442e9).
  failed   the carve raised or a code gate refused
Boxes already bisected are skipped (resumable).

    python -m mlx2.experimental.amoeba.child_boxbatch --spec SPEC.json
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from .progress import child_event


def carve_argv(spec: dict, bd: Path, d: Path) -> list[str]:
    c, data, plan = spec["compute"], spec["data"], spec["plan"]
    argv = ["--model", spec["model"], "--out", str(d), "--no-admit", "--stages", "inspect,mean,rank,bisect",
            "--ablations", "zero", "--steps", str(c["box_steps"]), "--bs", str(plan["batches"]["carve_box"]["bs"]),
            "--lr", str(c["box_lr"]),
            "--corpus", f"jsonl:{bd / 'corpus.jsonl'}", "--held-file", str(bd / "held_full.jsonl"),
            "--val-file", str(bd / "val_full.jsonl"),
            "--seq-len", str(data["window_tokens"]), "--held-len", str(data["window_tokens"]),
            "--windows", str(data["windows_per_doc"]), "--window-seed", str(data["window_seed"]),
            "--strip-prefix", str(data["strip_prefix"]), "--keep-prompt", str(data["keep_prompt"]),
            "--n-mean", str(c["box_n_mean"]), "--n-fit", str(c["box_n_fit"]), "--n-held", "0", "--targets", "",
            "--export-target", str(spec["verdict"]["val_target_nats"]),
            "--bisect-iters", str(c["box_bisect_iters"]), "--bisect-subsample", "1.0",
            "--bisect-tol", str(c["box_bisect_tol"]), "--fixed", f"{spec['core']}:core",
            "--watchdog-gb", str(plan["watchdog_gb"]), "--memory-gb", str(plan["mlx_memory_gb"]), "--cache-gb", "2",
            "--train-peak-cap-gb", str(plan["watchdog_gb"] - 2), "--moe-units", plan["moe_units"],
            "--pool-domains"] + [str(x) for x in spec["grad_gate"]]
    if data["code_cut"]:
        argv.append("--code-cut")
    if plan.get("assert_ple_offload"):
        argv.append("--assert-ple-offload")
    if spec.get("cpu"):
        argv.append("--cpu")
    if d.exists():
        argv.append("--resume")
    if spec.get("stop_at") is not None:
        argv += ["--stop-at", str(spec["stop_at"])]
    if c.get("box_fractions"):            # extra box cuts at fractions of ALL units beyond the core (tier_*_size*.npz)
        argv += ["--size-targets", str(c["box_fractions"])]
    return argv + list(spec.get("extra_carve_args") or [])


def verdict(rec: dict, ec: dict, units: int, v: dict) -> dict:
    hv = (ec.get("held") or {}).get("mean_d_ce")
    vv = (ec.get("spare") or {}).get("mean_d_ce")
    rnd = (ec.get("random_same_count_held") or {}).get("mean_d_ce")
    rec.update({"units": units, "export_val": vv, "export_held": hv,
                "val_held_gap": None if hv is None or vv is None else hv - vv,
                "random_same_count_held": rnd,
                "random_margin": None if hv is None or rnd is None else rnd - hv,
                "held_split": {k: x for k, x in (ec.get("held") or {}).items() if k not in ("mean_d_ce", "domain_rows")},
                "held_rows": (ec.get("held") or {}).get("domain_rows"),
                "held_weighting": v["pooling"]})
    meets = hv is not None and hv <= v["held_target_nats"]
    # a clear margin over the random box: both <= ratio x random AND random worse by at least the margin floor, so a
    # box that took (nearly) the whole free pool, whose same-count random box is the same units (both ~0), cannot pass
    floor = v.get("random_margin_floor_nats", 0.01)
    beats = (hv is not None and rnd is not None and hv <= v["random_control_ratio"] * rnd
             and (rnd - hv) >= floor)
    pool = rec.get("pool_units")
    if pool and units >= 0.95 * pool:
        rec["takes_whole_pool"] = True
    rec["held_meets_target"] = meets
    rec["beats_random"] = beats
    if units == 0:
        rec["verdict"] = "covered"
    elif meets and beats:
        rec["verdict"] = "pass"
    else:
        rec["verdict"] = "miss"
        rec["miss_reasons"] = ([] if meets else [f"held dCE {hv} > {v['held_target_nats']}"]) + \
                              ([] if beats else [f"no clear margin over the random same-count box ({rnd}): needs <= "
                                                 f"{v['random_control_ratio']} x random and a margin >= {floor} nats"]) + \
                              (["the box took the whole free pool (the core leaves too little out)"]
                               if rec.get("takes_whole_pool") else [])
    return rec


GATE_REFUSAL = "layer-wise gradients fail the directional-derivative check"   # research/amoeba/carve/__main__.py


def classify_failure(e) -> str:
    """'gate-refused': the s7 per-box gradient direction check (a diagnostic: that box only); 'refused': any other
    carve refusal (ones parity, the fused-vs-reference layer gate, replay paths: code faults that stop the stage);
    'crash': an exception."""
    if isinstance(e, SystemExit):
        return "gate-refused" if GATE_REFUSAL in str(e) else "refused"
    return "crash"


def core_alone_val(host, core_mx, val_path: Path, spec: dict) -> dict:
    """Section 9 check, on the box's VALIDATION documents (never held): mean dCE of the core alone vs the full model
    over one seeded window per document (rule 1 / 4; rule 3 for code)."""
    from research.amoeba.carve import evals as E
    from .child_probe import read_jsonl, window
    rows = read_jsonl(val_path) if val_path.exists() else []
    data = spec["data"]
    seqs = [s for s in (window(host, r["text"], data["window_tokens"], r.get("probe_id") or r.get("doc_id") or "",
                               data["window_seed"], r.get("domain") == "code" and data["code_cut"]) for r in rows)
            if s is not None]
    if not seqs:
        return {"d_ce": None, "windows": 0}
    full, n = E.text_ce(host, seqs)
    cut, _ = E.text_ce(host, seqs, E.configure(host, core=core_mx))      # configure() returns the setter
    return {"d_ce": round(cut - full, 5), "windows": len(seqs), "tokens": n}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text())
    from research.amoeba.carve import __main__ as D
    from research.amoeba.carve import ops as O
    from research.amoeba.carve.carving import CleanStop
    from research.amoeba.carve.host import Host
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    bs = Path(spec["boxset"])
    boxes = spec["boxes"]
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)    # noqa: E731
    plan = spec["plan"]
    O.footprint_watchdog(plan["watchdog_gb"], log)
    O.setup_mlx(plan["mlx_memory_gb"], 2)
    import mlx.core as mx
    if spec.get("cpu"):
        mx.set_default_device(mx.cpu)
    host = Host(spec["model"], log=log, moe_units=plan["moe_units"])
    core = np.load(spec["core"])["core"] > 0
    from research.amoeba.carve.__main__ import load_mask
    core_mx = load_mask(f"{spec['core']}:core", host)
    child_event("boxes_queued", boxes=[b["box"] for b in boxes])
    for b in boxes:
        name = b["box"]
        d = out / f"box-{name}"
        try:
            done = json.loads((d / "carve.json").read_text()).get("stages_done", [])
        except FileNotFoundError:
            done = []
        prior = None
        if (out / "verdicts.jsonl").exists():
            for x in (out / "verdicts.jsonl").read_text().splitlines():
                v = json.loads(x) if x.strip() else {}
                if v.get("box") == name and "bisect" in done and v.get("verdict") in ("pass", "miss", "covered"):
                    prior = v
        if prior is not None:
            log(f"box {name}: done, skipped")
            child_event("box_end", box=name, verdict=prior.get("verdict"), held_dce=prior.get("export_held"),
                        skipped=True)
            continue
        if spec.get("stop_at") is not None and time.time() >= spec["stop_at"]:
            log(f"--stop-at reached before box {name}: clean stop")
            raise SystemExit(75)
        child_event("box_start", box=name, kind=b.get("kind"), steps=spec["compute"]["box_steps"])
        t0 = time.time()
        rec = {"box": name, "kind": b.get("kind"), "node": b.get("node"), "topics": b.get("topics"),
               "flags": [f for f in ("held_synthetic", "held_lab") if b.get(f)],
               "continuations": b.get("continuations")}
        if b.get("continuations") and not b["continuations"].get("held"):
            rec["flags"].append("no_held_continuations")
        # diagnostic only (mapping-workflow at 77d3442e9: more boxes, never skip one because the core covers it)
        cov = core_alone_val(host, core_mx, bs / name / "val.jsonl", spec)
        cov["core_within_target"] = cov.get("d_ce") is not None and cov["d_ce"] <= spec["verdict"]["core_covers_nats"]
        rec["core_alone_val"] = cov
        try:
            D.main(carve_argv(spec, bs / name, d), host_factory=lambda _log: host)
        except (CleanStop, KeyboardInterrupt):
            log(f"box {name}: clean stop / interrupt; the batch resumes from this box")
            raise SystemExit(75)
        except (Exception, SystemExit) as e:             # gate refusals (SystemExit) and errors: record, go on
            rec["error"] = f"{type(e).__name__}: {e}"[:600]
            # s7: the per-box gradient check is a diagnostic: a refusal fails THIS box only (gate-refused); anything
            # else (a crash, a refusal outside the gate) is an error the stage reports (resume must not skip it)
            rec["failure"] = classify_failure(e)
            if not isinstance(e, SystemExit):
                rec["traceback"] = traceback.format_exc()[-2000:]
            try:
                host.train_mode(False)
                host.ctl.reset()
                mx.clear_cache()
            except Exception:  # noqa: BLE001
                pass
        rec["wall_s"] = round(time.time() - t0, 1)
        try:
            r = json.loads((d / "carve.json").read_text())
        except FileNotFoundError:
            r = {}
        ec = r.get("export_core") or {}
        g = ((r.get("inspect") or {}).get("paths") or {}).get("layerwise_grad")
        rec["grad_gate"] = g.get("ok") if isinstance(g, dict) else g
        if rec.get("failure") == "gate-refused" and isinstance(g, dict):
            rec["gate_cells"] = {k: g.get(k) for k in ("fails", "eps", "gate_eps") if k in g}
        if ec and "error" not in rec:
            m = (np.load(d / "export_core.npz")["core"] > 0) & ~core
            rec["pool_units"] = int((~core[:, :]).sum() - sum(core.shape[1] - w for w in host.widths))
            np.savez_compressed(d / "box_mask.npz", box=m.astype(np.uint8))
            verdict(rec, ec, int(m.sum()), spec["verdict"])
        else:
            rec["verdict"] = "failed"
            rec.setdefault("error", "no export core in the carve receipt")
            rec.setdefault("failure", "crash")
        with (out / "verdicts.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        child_event("box_end", box=name, verdict=rec["verdict"], held_dce=rec.get("export_held"),
                    units=rec.get("units"), wall_s=rec["wall_s"])
        log(f"box {name} {rec['verdict']} in {rec['wall_s']} s: "
            f"{json.dumps({k: rec.get(k) for k in ('units', 'export_val', 'export_held', 'random_same_count_held', 'error')})}")


if __name__ == "__main__":
    main()
