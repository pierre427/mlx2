"""Install an extracted box into another model (mapping-workflow s12.2-12.3).

    mlx2 amoeba install --box <extract out>/box_bundle --donor <donor checkpoint> --host <model> --out DIR \
        [--routes side,tile,lora,bridge]

Stages (receipts, resume and progress as in ``build``; kind ``install``):
  pair          the bundle against its donor (the checkpoint fingerprint must match the manifest), the host, whether
                their tokenizers are byte-identical (required: every route pairs tokens by id), the joint memory plan
                (both models resident: the routes run donor and host in one process, a declared exception to one
                model per process), the label (models below 8B are smoke tests)
  topo          the donor's box layers (the layers holding the most box units, at least 2 apart) and the layer warp
                to the host by relative depth (s12.2 item 2; a CKA-fitted warp is not built). The host stub is
                chosen inside the bridge route, contrastively on the topic (a host map is not wired yet)
  route_side    the bundle's core + box beside the host, blended at the output through the bundle's gate detector;
                controls: host, core + same-count random units, gate always open, the whole donor
  route_tile    gated tiles in the host, trained by full-vocabulary KL to the box teacher on the topic's fit probes;
                controls: host, untrained tiles, tiles trained on the core + random-units teacher
  route_lora    the same with a gated low-rank linear adapter
  route_bridge  a learned low-rank gated map from the box units' activations to host writes (ridge-initialised onto
                -dKL/dh); controls: host, random donor units, the host's own stub units (no donor), another text's
                box features, noise
  report        per route: the main arm against each control (paired bootstrap over held texts), English drift
                against the cap, gate open rates, cost; the install bundle (schema mlx2.amoeba-install.v1)

Every write and tile is capped per token at --norm-budget x the host layer's own increment, and every trained arm
is held to the English drift cap. Held texts only score: they never choose an epoch, a unit or a threshold.
Ported from scripts/amoeba_bridge_lab.py (claude/amoeba-sidestream@138312ddb). Experimental and unqualified.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import paths
from .runner import run_child
from .stages import _py
from .state import file_outputs, stage_commits, write_json

ROUTES = ("side", "tile", "lora", "bridge")
INSTALL_STAGES = ("pair", "topo") + tuple(f"route_{r}" for r in ROUTES) + ("report",)
INSTALL_SCHEMA = "mlx2.amoeba-install.v1"
DEFAULTS = {
    "teacher": "box", "norm_budget": 0.25, "drift_cap": 0.05, "drift_adapt": 4.0, "drift_weight": 1.0,
    "drift_weight_max": 256.0, "epochs": 3, "batch": 4, "lr": {"tile": 1e-3, "lora": 1e-3, "bridge": 1e-3},
    "tile_units": 32, "lora_rank": 8, "tile_layers": "all", "bridge_rank": 32, "ridge_lam": 0.1,
    "box_units_per_layer": 64, "n_box_layers": 4, "n_stub_layers": 4, "seed": 20261116, "gate_min_tokens": 8,
    "data": {"topic_tokens": 64, "n_train": 120, "n_val": 40, "n_test": 100, "eng_retain": 12, "eng_val": 4,
             "eng_test": 8, "eng_tokens": 96},
}
QUICK = {"epochs": 1, "data": {"topic_tokens": 48, "n_train": 8, "n_val": 4, "n_test": 12, "eng_retain": 2,
                               "eng_val": 1, "eng_test": 2, "eng_tokens": 32},
         "tile_units": 4, "lora_rank": 2, "bridge_rank": 4, "box_units_per_layer": 8, "n_box_layers": 2,
         "n_stub_layers": 2}


def settings(profile: str, overrides: dict) -> dict:
    s = json.loads(json.dumps(DEFAULTS))
    if profile == "quick":
        for k, v in QUICK.items():
            if isinstance(v, dict):
                s[k].update(v)
            else:
                s[k] = v
    for k, v in overrides.items():               # KEY=VALUE, data.KEY=VALUE, lr.ROUTE=VALUE
        if k == "teacher":
            raise SystemExit("use --teacher, not --set teacher=")
        if "." in k:
            a, b = k.split(".", 1)
            if not isinstance(s.get(a), dict) or b not in s[a]:
                raise SystemExit(f"--set {k}: unknown setting (have {sorted(s)} and data.*, lr.*)")
            s[a][b] = type(s[a][b])(v)
        else:
            if k not in s or isinstance(s[k], dict):
                raise SystemExit(f"--set {k}: unknown setting (have {sorted(x for x in s if not isinstance(s[x], dict))})")
            s[k] = type(s[k])(v)
    return s


def params_estimate(info: dict) -> float:
    """Logical parameters from the safetensors headers: packed (uint32) weights count x 32 / bits, quantization
    scales and biases are not parameters; falls back to the stored bytes / width when headers are unreadable."""
    import struct
    bits = ((info.get("quantization") or {}).get("bits") or 16)
    gsize = int((info.get("quantization") or {}).get("group_size") or 64)
    total, ok = 0, False
    for f in sorted(Path(info["path"]).glob("*.safetensors")):
        try:
            with f.open("rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
        except (OSError, ValueError, struct.error):
            ok = False
            break
        ok = True
        for k, v in hdr.items():
            if k == "__metadata__" or k.endswith((".scales", ".biases")):
                continue
            size = int(np.prod(v["shape"])) if v["shape"] else 1
            sc = hdr.get(k[:-len("weight")] + "scales") if k.endswith("weight") else None
            if v["dtype"] in ("U32", "I32") and sc is not None and len(v["shape"]) >= 2:
                # packed: rows x (packed columns x 32 / bits) = rows x (scale groups x group size); the bits per
                # row follow from the two shapes, so mixed and 3/5/6-bit quantization count right
                cols_packed, groups = v["shape"][-1], sc["shape"][-1]
                b_eff = 32 * cols_packed / max(groups * gsize, 1)
                size = int(round(size * 32 / b_eff))
            elif v["dtype"] in ("U32", "I32") and bits < 16:
                size = int(size * 32 / bits)
            total += size
    return float(total) if ok and total else info["weight_bytes"] * 8.0 / bits


# ------------------------------------------------------------------ stages

def run_pair(b, ctx):
    d = b.stage_dir("pair")
    x = b.rec["install"]
    man = json.loads(Path(x["bundle"], "manifest.json").read_text())
    donor_fp = b.rec["donor_inspection"]["fingerprint"]
    if man["donor"]["fingerprint"] != donor_fp:
        raise SystemExit(f"the bundle's donor is {man['donor']['checkpoint']} ({man['donor']['fingerprint'][:12]}), "
                         f"not --donor {b.rec['donor_inspection']['checkpoint']} ({donor_fp[:12]}): refusing")
    if man.get("verdict") in ("covered", "failed") or not (Path(x["bundle"]) / "box_mask.npz").exists():
        raise SystemExit(f"the bundle holds no box ({man.get('verdict')}: {man.get('note')}): nothing to install")
    vd, vh = b.rec["donor_inspection"].get("vocab"), b.info.get("vocab")
    if vd and vh and vd != vh:
        raise SystemExit(f"donor vocabulary {vd} != host vocabulary {vh}: token ids pair, the distributions do not")
    same = x["same_tokenizer"]
    if not same:
        raise SystemExit("donor and host tokenizers are not byte-identical: every install route pairs tokens by id "
                         "(a string-id map for the side stream is not built)")
    res = {"bundle": x["bundle"], "topic": man["topic"], "box": man["box"], "donor": man["donor"],
           "donor_quality": man.get("donor_quality"), "gap_screen": man.get("gap_screen"),
           "host": {"checkpoint": b.checkpoint, "fingerprint": b.info["fingerprint"]}, "same_tokenizer": same,
           "joint_plan": b.rec["joint_plan"], "label": b.rec["label"]}
    write_json(d / "pair.json", res)
    return file_outputs({"pair": d / "pair.json"}), {"topic": man["topic"], "box": man["box"], "label": b.rec["label"]}


def run_topo(b, ctx):
    d = b.stage_dir("topo")
    x = b.rec["install"]
    box = np.load(Path(x["bundle"]) / "box_mask.npz")["box"] > 0
    core = np.load(Path(x["bundle"]) / "core/core.npz")["core"] > 0
    box &= ~core
    LD, LH = box.shape[0], int(b.info["layers"])
    counts = box.sum(1)
    sel = []
    for l in np.argsort(-counts):
        if counts[l] == 0:
            break
        if 0 < l < LD - 1 and all(abs(int(l) - m) >= 2 for m in sel):
            sel.append(int(l))
        if len(sel) == x["settings"]["n_box_layers"]:
            break
    if not sel:
        raise SystemExit("the box has no units outside the core on an interior layer")
    sel = sorted(sel)
    warp = {str(l): int(round(l * (LH - 1) / max(LD - 1, 1))) for l in sel}
    if len(set(warp.values())) < len(warp):
        raise SystemExit(f"box layers {sel} collide under the relative-depth warp {warp}")
    res = {"box_layers": sel, "box_units": {str(l): int(counts[l]) for l in sel}, "warp": warp,
           "warp_method": "relative depth (a CKA-fitted monotone warp, s12.2 item 2, is not built)",
           "donor_layers": LD, "host_layers": LH, "box_units_total": int(box.sum()), "core_units": int(core.sum())}
    write_json(d / "topo.json", res)
    return file_outputs({"topo": d / "topo.json"}), {"box_layers": sel, "warp": warp}


def _route_runner(route):
    def run(b, ctx):
        d = b.stage_dir(f"route_{route}")
        x = b.rec["install"]
        if route not in x["routes"]:
            write_json(d / "result.json", {"route": route, "skipped": "not in --routes"})
            return file_outputs({"result": d / "result.json"}), {"skipped": True}
        bundle = Path(x["bundle"])
        gates = json.loads((bundle / "gate/gates.json").read_text()).get("boxes", {})
        man = json.loads((bundle / "manifest.json").read_text())
        g = gates.get(man["box"])
        s = x["settings"]
        spec = {**{k: v for k, v in s.items() if k != "data"}, "data": s["data"], "route": route,
                "donor": b.rec["donor_inspection"]["path"], "host": b.info["path"], "topic": man["topic"],
                "core_mask": str(bundle / "core/core.npz"), "box_mask": str(bundle / "box_mask.npz"),
                "gate": ({"npz": str(bundle / "gate/gates.npz"), "box": man["box"], "layer": g["layer"],
                          "threshold": g["threshold"], "status": g.get("status")}
                         if g and g.get("layer") is not None and g.get("status", "trained") == "trained" else
                         {"status": (g or {}).get("status", "absent")}),
                "probe_corpus": b.rec["inputs"]["probe_corpus"]["path"],
                "english": [c["path"] for c in b.rec["inputs"]["core_corpora"]],
                "topo": json.loads((b.out / "topo/topo.json").read_text()), "out": str(d / "run"), "cpu": ctx.cpu,
                "host_hidden": s["host_hidden"], "watchdog_gb": b.rec["joint_plan"]["cap_gb"],
                "host_checkpoint": f"{b.info['artifact']}@{b.info['fingerprint']}",
                "donor_checkpoint": f"{b.rec['donor_inspection']['artifact']}@{b.rec['donor_inspection']['fingerprint']}",
                "mlx2_commit": ctx.commit, "bundle_verdict": (man.get("donor_quality") or {}).get("verdict"),
                "memory_gb": b.rec["joint_plan"]["mlx_memory_gb"], "label": b.rec["label"]}
        if (d / "run").exists():              # a partial run: archived, so retries stay distinguishable
            dst = b.out / ".redo" / f"route_{route}-run-{int(time.time())}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            (d / "run").rename(dst)
        sp = d / "spec.json"
        write_json(sp, spec)
        from .runner import ChildFailed, CleanStop, StageFailedContinue
        try:
            run_child(_py("mlx2.experimental.amoeba.child_install_route") + ["--spec", sp], stage=f"route_{route}",
                      progress=ctx.progress, log_path=b.log_path(f"route_{route}"), what=f"install route {route}")
        except CleanStop:
            raise
        except ChildFailed as e:              # this route failed: recorded, the other routes and the report go on
            tail = b.log_path(f"route_{route}").read_text().splitlines()[-12:]
            write_json(d / "result.json", {"route": route, "failed": str(e), "log_tail": tail})
            # not a ``done`` stage: the receipt says failed (a resume retries the route), the run continues
            raise StageFailedContinue(str(e), file_outputs({"result": d / "result.json"}), {"failed": str(e)})
        r = json.loads((d / "run/result.json").read_text())
        (d / "result.json").unlink(missing_ok=True)  # supersede any earlier failure marker
        main = {"side": "side_box", "tile": "tile_box", "lora": "lora_box", "bridge": "bridge_box"}[route]
        m = r["arms"].get(main, {})
        return file_outputs({"result": d / "run/result.json", "rows": d / "run/rows.jsonl"}), \
            {"arm": main, "gap_closed": m.get("gap_closed"), "gap_closed_ci95": m.get("gap_closed_ci95"),
             "english_drift_test": m.get("english_drift_test")}
    return run


def _paired(rows, a, c, n=2000, seed=5):
    """paired bootstrap over held texts of (control - arm) nats/byte: > 0 means the arm is better"""
    arm = [r for r in rows if r["arm"] == a]
    control = [r for r in rows if r["arm"] == c]
    x = {r["probe_id"]: r["nats_per_byte"] for r in arm}
    y = {r["probe_id"]: r["nats_per_byte"] for r in control}
    if x.keys() != y.keys() or len(x) != len(arm) or len(y) != len(control):
        return None
    ids = sorted(x)
    dv = np.array([y[i] - x[i] for i in ids])
    if not len(dv) or not np.isfinite(dv).all():
        return None
    g = np.random.default_rng(seed)
    bs = np.sort([dv[g.integers(0, len(dv), len(dv))].mean() for _ in range(n)])
    return {"mean": float(dv.mean()), "ci95": [float(bs[int(0.025 * n)]), float(bs[int(0.975 * n) - 1])], "n": len(ids)}


def run_report(b, ctx):
    d = b.out / "report"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    x = b.rec["install"]
    cap = x["settings"]["drift_cap"]
    verdicts = []
    completed = []
    for route in ROUTES:
        rd = b.out / f"route_{route}/run"
        if route not in x["routes"]:
            continue
        failed = b.out / f"route_{route}/result.json"
        if failed.exists() or not (rd / "result.json").is_file():
            fr = json.loads(failed.read_text()) if failed.exists() else {"failed": "no result"}
            verdicts.append({"route": route, "verdict": "failed", "error": fr.get("failed"), "label": b.rec["label"]})
            continue
        r = json.loads((rd / "result.json").read_text())
        with (rd / "rows.jsonl").open() as f:
            rows = [json.loads(line) for line in f]
        main = {"side": "side_box", "tile": "tile_box", "lora": "lora_box", "bridge": "bridge_box"}[route]
        m = r["arms"][main]
        vs = {}
        for c in m.get("controls") or []:
            if c == "host":
                vs["host"] = {"mean": m["delta_vs_host"], "ci95": m["delta_vs_host_ci95"]}
            else:
                vs[c] = _paired(rows, main, c) if c in r["arms"] else None
        drift = m.get("english_drift_test")
        expected_n = (r.get("n") or {}).get("test")
        complete = ("host" in vs and len(vs) >= 2 and
                    all(v and np.isfinite(v["ci95"]).all() and
                        (name == "host" or expected_n is None or v["n"] == expected_n) for name, v in vs.items()))
        if not complete or drift is None or not np.isfinite(drift) or not np.isfinite(r["teacher_gap_ci95"]).all():
            verdicts.append({"route": route, "verdict": "failed", "error": "incomplete control or drift evidence",
                             "label": b.rec["label"]})
            continue
        completed.append(route)
        drift_ok = drift <= cap
        beats = {c: bool(v and v["ci95"][0] > 0) for c, v in vs.items()}
        no_gap = r["teacher_gap_ci95"][0] <= 0
        verdict = ("no teacher gap" if no_gap else
                   "transfers" if beats.get("host") and all(beats.values()) and drift_ok else
                   "helps, not specific" if beats.get("host") and drift_ok else
                   "drifts" if beats.get("host") else "no gain")
        verdicts.append({"route": route, "arm": main, "verdict": verdict, "label": b.rec["label"],
                         "bundle": b.rec["bundle_quality"],
                         "rule": "no teacher gap: the teacher is not better than the host on held texts (CI); "
                                 "transfers: the arm beats the host AND every control (paired 95% CI over held "
                                 "texts above 0) with English drift <= the cap; helps, not specific: beats the host "
                                 "only; drifts: beats the host over the drift cap",
                         "n": r.get("n"), "result_gate": r.get("gate"), "ceiling": m.get("ceiling"),
                         "vs_controls": vs, "beats": beats, "gap_closed": m.get("gap_closed"),
                         "gap_closed_ci95": m.get("gap_closed_ci95"), "english_drift_test": m.get("english_drift_test"),
                         "drift_cap": cap, "teacher": r["teacher"], "teacher_gap": r["teacher_gap"],
                         "teacher_gap_ci95": r["teacher_gap_ci95"], "trainable_params": m.get("trainable_params"),
                         "teacher_forced_tok_s": m.get("teacher_forced_tok_s"),
                         "gate": {k: m.get(k) for k in ("held_topic_open_rate", "english_open_rate")}})
    with (d / "verdicts.jsonl").open("w") as f:
        for v in verdicts:
            f.write(json.dumps(v) + "\n")
    for name in ("pair/pair.json", "topo/topo.json"):
        shutil.copy2(b.out / name, d / Path(name).name)
    for route in completed:
        rd = b.out / f"route_{route}/run"
        if (rd / "result.json").exists():
            (d / route).mkdir()
            for f in ("result.json", "rows.jsonl", "curves.jsonl"):
                if (rd / f).exists():
                    shutil.copy2(rd / f, d / route / f)
            if (rd / "params").exists():
                shutil.copytree(rd / "params", d / route / "params")
    files = {str(f.relative_to(d)): {"sha256": paths.sha256_file(f), "bytes": f.stat().st_size}
             for f in sorted(d.rglob("*")) if f.is_file()}
    man = {"schema": INSTALL_SCHEMA, "label": b.rec["label"], "topic": json.loads((d / "pair.json").read_text())["topic"],
           "donor": b.rec["donor_inspection"]["checkpoint"], "host": b.checkpoint, "bundle": x["bundle"],
           # mlx2_commit: the commit this report was written at; each route's verdict was made at its own stage's
           "mlx2_commit": ctx.commit, "bundled_at_commit": ctx.commit, "stage_commits": stage_commits(b),
           "profile": b.cfg["profile"], "settings": x["settings"], "routes": x["routes"],
           "verdicts": [{k: v.get(k) for k in ("route", "arm", "verdict", "gap_closed", "english_drift_test")}
                        for v in verdicts],
           "host_checkpoint": f"{b.info['artifact']}@{b.info['fingerprint']}",
           "donor_checkpoint": (f"{b.rec['donor_inspection']['artifact']}@"
                                f"{b.rec['donor_inspection']['fingerprint']}"),
           "bundle_quality": b.rec["bundle_quality"],
           "processes": b.rec["joint_plan"]["processes"],
           "status": "experimental and unqualified (research install)", "files": files,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write_json(d / "manifest.json", man)
    return file_outputs({"manifest": d / "manifest.json", "verdicts": d / "verdicts.jsonl"}), \
        {v["route"]: v["verdict"] for v in verdicts}


INSTALL_RUNNERS = {"pair": run_pair, "topo": run_topo, "report": run_report,
                   **{f"route_{r}": _route_runner(r) for r in ROUTES}}
MODEL_STAGES = {f"route_{r}" for r in ROUTES}


def joint_plan(donor_info: dict, host_info: dict, cap_gb: float | None, cap_reason: str, cpu: bool,
               s: dict | None = None, routes=ROUTES) -> dict:
    """Both models resident in one process (the routes' declared exception): their weights (float32 copies on the
    CPU) plus a working-set allowance must fit under the per-process cap."""
    from .plan import GB, physical_gb
    cap = 80.0 if cap_gb is None else float(cap_gb)
    if cap > 80.0 and not cap_reason:
        raise SystemExit(f"--cap-gb {cap} is over the 80 GB per-process cap: pass --cap-reason (who approved it)")
    phys = physical_gb()
    if phys:
        cap = min(cap, phys - 15.0)

    def resident(info):
        w = info["weight_bytes"] / GB
        if cpu:                                        # dequantized to float32 on the CPU
            w = params_estimate(info) * 4 / GB
        return w
    wd, wh = resident(donor_info), resident(host_info)
    work = 8.0 + 0.1 * (wd + wh)
    lp = 0.0                                           # teacher / host log-probs held as [T, V] float16 (host RAM)
    if s is not None:
        dd = s["data"]
        V = int(max(host_info.get("vocab") or 0, donor_info.get("vocab") or 0) or 262144)
        T = dd["topic_tokens"] + 1
        copies = 2 if any(r in ("tile", "lora") for r in routes) else 1
        lp = ((dd["n_train"] + dd["n_val"]) * T * V * 2 * copies + dd["n_val"] * T * V * 2 +
              (dd["eng_retain"] + dd["eng_val"] + dd["eng_test"]) * (dd["eng_tokens"] + 1) * V * 2) / GB
    need = wd + wh + work + lp
    return {"cap_gb": round(cap, 1), "donor_gb": round(wd, 2), "host_gb": round(wh, 2), "work_gb": round(work, 1),
            "logprob_store_gb": round(lp, 2),
            "need_gb": round(need, 1), "fits": need <= cap, "mlx_memory_gb": round(min(cap, need + 4), 1),
            "processes": {f"route:{r}": "donor+host in one process (lockstep / teacher)" for r in ROUTES}}


def add_parser(sub, common_plan_args, run_args) -> None:
    p = sub.add_parser("install", help="install an extracted box into another model (s12.2-12.3)")
    p.add_argument("--box", required=True, help="an extract's box_bundle directory")
    p.add_argument("--donor", required=True, help="the bundle's donor checkpoint (its fingerprint must match)")
    p.add_argument("--host", required=True, help="the model that receives the box")
    p.add_argument("--out", required=True)
    p.add_argument("--routes", default="side", help=f"comma list of {','.join(ROUTES)} (default side, as s12.3: the "
                   "others are experimental options)")
    p.add_argument("--teacher", choices=["box", "full"], default="box",
                   help="box: the bundle's core + box (the extracted expert); full: the whole donor")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="route setting (e.g. epochs=2, data.n_test=200, lr.bridge=5e-4)")
    p.add_argument("--probe-corpus", default=str(paths.DEFAULT_PROBE_CORPUS),
                   help="the probe corpus the bundle was made from (its sha256 must match the bundle's manifest)")
    p.add_argument("--core-corpus", action="append", default=[], help="English prose jsonl (repeatable)")
    common_plan_args(p)
    run_args(p)


def cmd_install(a, execute, check_commit) -> int:
    from .inspection import inspect_checkpoint
    from . import config as K
    from .state import Build
    bundle = Path(a.box).expanduser().resolve()
    man = json.loads((bundle / "manifest.json").read_text())
    if man.get("schema") != "mlx2.amoeba-box-bundle.v1":
        raise SystemExit(f"{bundle}: not a box bundle (schema {man.get('schema')!r})")
    routes = [r for r in a.routes.split(",") if r]
    bad = sorted(set(routes) - set(ROUTES))
    if bad:
        raise SystemExit(f"unknown routes {bad} (have {', '.join(ROUTES)})")
    dinfo, hinfo = inspect_checkpoint(a.donor), inspect_checkpoint(a.host)
    if man["donor"]["fingerprint"] != dinfo["fingerprint"]:
        raise SystemExit(f"--donor {dinfo['checkpoint']} is not the bundle's donor {man['donor']['checkpoint']}: refusing")
    s = settings(a.profile, K.parse_overrides(a.set))
    s["teacher"] = a.teacher
    s["host_hidden"] = int(hinfo["hidden"])
    if dinfo.get("kind") == "moe":
        raise SystemExit("the donor is MoE: install reads dense MLP units only (MoE expert units in a box are not "
                         "supported yet)")
    _verify_bundle(bundle, man)
    jp = joint_plan(dinfo, hinfo, a.cap_gb, a.cap_reason, a.cpu, s, routes)
    print(json.dumps({"joint_plan": jp}, indent=1))
    if not jp["fits"]:
        raise SystemExit(f"donor {jp['donor_gb']} GB + host {jp['host_gb']} GB + work {jp['work_gb']} GB = "
                         f"{jp['need_gb']} GB do not fit together under the {jp['cap_gb']} GB cap: refusing")
    pc = Path(a.probe_corpus).expanduser().resolve()
    pman = bundle / "taxonomy/probe-corpus-manifest.json"
    psha = paths.sha256_file(pc)
    if json.loads(pman.read_text()).get("corpus_sha256") != psha:
        raise SystemExit(f"--probe-corpus {pc} is not the corpus the bundle was made from: refusing")
    eff = json.loads((bundle / "manifest.json").read_text()).get("probe_corpus_effective")
    if eff:     # the extraction excluded texts (a reserved final test): the routes read its effective corpus only
        ep = Path(eff["path"])
        if not ep.exists() or paths.sha256_file(ep) != eff["sha256"]:
            raise SystemExit(f"the bundle was carved on an effective probe corpus ({ep}) that is missing or changed: "
                             "refusing (its excluded texts must stay out of every route)")
        pc, psha = ep, eff["sha256"]
    tax = sorted((bundle / "taxonomy").glob("taxonomy*.json"))
    cores = [Path(x).expanduser().resolve() for x in a.core_corpus] or list(paths.DEFAULT_CORE_CORPORA)
    inputs = {"probe_corpus": {"path": str(pc), "sha256": psha},
              "probe_manifest": {"path": str(pman), "sha256": paths.sha256_file(pman)},
              "taxonomy": {"path": str(tax[0]) if tax else str(pman), "sha256": paths.sha256_file(tax[0] if tax else pman)},
              "core_corpora": [{"path": str(c), "sha256": paths.sha256_file(c)} for c in cores],
              "bundle_manifest": {"path": str(bundle / "manifest.json"), "sha256": paths.sha256_file(bundle / "manifest.json")}}
    cfg = {"profile": a.profile, "install": s}
    from .cli import _same_tokenizer                 # one tokenizer rule for extract and install
    same = _same_tokenizer(a.donor, a.host)
    commit = check_commit(a.allow_dirty)
    stages = list(INSTALL_STAGES)
    rec = {"kind": "install", "model": hinfo["path"], "checkpoint": hinfo["checkpoint"], "inspection": hinfo,
           "donor_inspection": dinfo, "config": cfg, "plan": jp, "joint_plan": jp, "inputs": inputs,
           "install": {"bundle": str(bundle), "routes": routes, "settings": s, "same_tokenizer": same},
           "label": label_for(dinfo, hinfo, man), "bundle_quality": _bundle_quality(man), "options": {"cpu": a.cpu, "gen_backend": "local", "gen_route": "",
                                                         "boxes": "", "max_boxes": 0, "max_cluster_boxes": 0},
           "stage_options": {st: {"device": "cpu" if a.cpu else "gpu"} for st in stages}, "stages": stages,
           "mlx2_commit": commit, "argv": __import__("sys").argv,
           "status": "implemented (research install; unqualified)"}
    b = Build.create(Path(a.out), rec)
    return execute(b, stages, a, commit, redo=[])


def _verify_bundle(bundle: Path, man: dict) -> None:
    """every file the routes read matches the sha256 the bundle's manifest records (when it records files)"""
    files = man.get("files") or {}
    for rel in ("box_mask.npz", "core/core.npz", "gate/gates.npz", "gate/gates.json"):
        if rel in files and paths.sha256_file(bundle / rel) != files[rel]["sha256"]:
            raise SystemExit(f"bundle file {rel} does not match its manifest sha256: refusing")


def check_install_inputs(b) -> None:
    """resume / stage: the donor checkpoint and the bundle manifest are what the install started from"""
    from .inspection import inspect_checkpoint
    d = b.rec["donor_inspection"]
    if inspect_checkpoint(d["path"])["fingerprint"] != d["fingerprint"]:
        raise SystemExit(f"donor {d['path']} changed since the install started")
    bm = b.rec["inputs"]["bundle_manifest"]
    if paths.sha256_file(bm["path"]) != bm["sha256"]:
        raise SystemExit(f"bundle manifest {bm['path']} changed since the install started")
    _verify_bundle(Path(b.rec["install"]["bundle"]), json.loads(Path(bm["path"]).read_text()))


def _bundle_quality(man: dict) -> dict:
    q = man.get("donor_quality") or {}
    return {"verdict": q.get("verdict") or man.get("verdict"), "profile": (man.get("config") or {}).get("profile"),
            "status": man.get("status"), "held_d_ce": q.get("export_held")}


def label_for(donor_info: dict, host_info: dict, man: dict | None = None) -> str:
    """smoke test below 8B; a quick-profile or missed box bundle says so too"""
    pd, ph = params_estimate(donor_info), params_estimate(host_info)
    parts = []
    if min(pd, ph) < 8e9:
        parts.append(f"smoke test ({ph / 1e9:.1f}B host / {pd / 1e9:.1f}B donor), not a result for the >8B goal")
    q = _bundle_quality(man or {})
    if q["profile"] == "quick":
        parts.append("the box bundle is a quick-profile pipeline check, not a box-quality box")
    if q["verdict"] == "miss":
        parts.append("the box missed its held target on the donor")
    return "; ".join(parts)
