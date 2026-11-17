"""The six stages of mapping-workflow section 5, as orchestration over the research code.

Each ``run_<stage>(b, ctx)`` writes its files under ``<out>/<stage>/``, runs model work in child processes
(one model per process), and returns ``(outputs, summary)`` for the stage receipt. Every stage is
resumable: a finished sub-step is detected from its own output file and not redone.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import paths
from .runner import Background, free_port, run_child
from .state import file_outputs, write_json


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().split("\n") if x.strip()]


def write_jsonl(p, rows):
    """Atomic: a resume never reads a file a killed run left half-written."""
    p = Path(p)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(p)


def fit_tags(c: dict) -> set:
    """A cluster's topics by the labeller's FIT-split gate only (fit_ok: d >= 0.8, >= twin95 + 0.5 and the shuffled-
    label test, Bonferroni p < 0.01, all on fit): held text checks the tagger, it never chooses which clusters seed
    a box or become cluster boxes. Label files from before fit_ok fall back to the d and twin parts."""
    out = set()
    for x in c.get("top_nodes_fit") or []:
        if "fit_ok" in x and x["fit_ok"] is not None:
            ok = x["fit_ok"]
        else:
            ok = x.get("d_fit") is not None and x["d_fit"] >= 0.8 and x["d_fit"] >= (x.get("twin95_fit") or 0) + 0.5
        if ok:
            out.add(x["node"])
    return out


def h(s, seed=0):
    return int(hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()[:12], 16)


class Ctx:
    """What every stage needs besides the build: progress, device, commit, generation backend."""

    def __init__(self, progress, *, cpu: bool, commit: dict, gen_backend: str, stop_at: float | None,
                 boxes: str = "", max_boxes: int = 0, max_cluster_boxes: int = -1, gen_route: str = ""):
        self.progress = progress
        self.cpu = cpu
        self.commit = commit
        self.gen_backend = gen_backend
        self.stop_at = stop_at
        self.boxes = boxes
        self.max_boxes = max_boxes
        self.max_cluster_boxes = max_cluster_boxes
        self.gen_route = gen_route
        self.redo = set()          # stages being redone in this run (extract: bypass the fingerprint cache)


def _mem_args(b, job: str) -> list:
    p = b.plan
    return ["--watchdog-gb", p["watchdog_gb"], "--memory-gb", p["mlx_memory_gb"]]


def _py(module_or_script: str) -> list:
    if module_or_script.endswith(".py"):
        return [paths.python(), "-u", str(paths.repo_root() / "scripts" / module_or_script)]
    return [paths.python(), "-u", "-m", module_or_script]


def _reset_partial(d: Path, done_marker: str) -> None:
    """A sub-step whose output dir exists without its completion marker is redone from scratch (fresh dirs only)."""
    if d.exists() and not (d / done_marker).exists():
        shutil.rmtree(d)


# ====================================================================== generation (on-policy continuations)

def generate(b, ctx, stage: str, corpora: list, out: Path, *, held_per_domain: int = 0, free_per_domain: int = 0) -> dict:
    """1 greedy + 3 T=0.7 continuations (rule 5) for every row of ``corpora`` through
    scripts/amoeba_onpolicy_gen.py (resumable: gen.partial.jsonl), against a local ordinary-decode server
    (child_gen_server) or an mlx2 dev server on the model's best route (scripts/amoeba_gen_stage.py)."""
    if (out / "gen.jsonl").exists():
        return json.loads((out / "manifest.json").read_text())
    data = b.cfg["data"]
    gen_args = ["--out", out, "--window-tokens", data["gen_prompt_tokens"], "--tokenizer", b.info["path"],
                "--max-tokens", data["gen_new_tokens"], "--samples", data["gen_samples"],
                "--temperature", data["gen_temperature"], "--seed0", data["gen_seed0"],
                "--window-seed", data["window_seed"], "--strip-prefix", data["strip_prefix"],
                "--held-per-domain", held_per_domain, "--free-per-domain", free_per_domain]
    if data["code_cut"]:
        gen_args.append("--code-cut")
    for c in corpora:
        gen_args += ["--corpus", c]
    log = b.log_path(stage)
    if ctx.gen_backend == "server":
        if ctx.cpu:
            raise SystemExit("--gen-backend server needs the GPU (an mlx2 dev server); use --gen-backend local on CPU")
        run_child(_py("amoeba_gen_stage.py") + ["--model", b.info["path"], "--route", ctx.gen_route, "--port", 8395,
                                                "--workers", 4] + gen_args,
                  stage=stage, progress=ctx.progress, log_path=log, what="gen (mlx2 dev server)")
        route = f"mlx2 dev server :8395, route {ctx.gen_route or 'adapter default best route'}"
    else:
        port = free_port()
        srv = Background(_py("mlx2.experimental.amoeba.child_gen_server") + ["--model", b.info["path"], "--port", port]
                         + (["--cpu"] if ctx.cpu else []), stage=stage, progress=ctx.progress, log_path=log)
        try:
            srv.wait_ready()
            run_child(_py("amoeba_onpolicy_gen.py") + ["--base-url", f"http://127.0.0.1:{port}",
                                                       "--checkpoint", b.checkpoint, "--workers", 1] + gen_args,
                      stage=stage, progress=ctx.progress, log_path=log, what="gen (local ordinary decode)")
        finally:
            srv.stop()
        route = "local ordinary-reference decode (mlx2.experimental.amoeba.child_gen_server)"
    man = json.loads((out / "manifest.json").read_text())
    man["route"] = route
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    ctx.progress.emit("gen", stage=stage, tok_s=man.get("tok_s_aggregate"), tokens_total=man.get("completion_tokens"))
    return man


# ====================================================================== step 1: map and tag

def _subsample_probes(src: Path, dst: Path, per_node: int, seed: int) -> dict:
    rows = read_jsonl(src)
    by = defaultdict(list)
    for r in rows:
        t = r["topics"][0]
        node = t if t.startswith("lang.") else ".".join(t.split(".")[:2])
        by[(node, r["split"])].append(r)
    keep = []
    for k in sorted(by):
        keep += sorted(by[k], key=lambda r: h(r["id"], seed))[:per_node]
    write_jsonl(dst, keep)
    return {"rows": len(keep), "of": len(rows), "per_node_and_split": per_node, "nodes": len({k[0] for k in by})}


def carve_widths(b) -> list:
    """Per-layer unit counts of the carve (the core carve's receipt): the layout of every mask in this build."""
    return json.loads((b.out / "core/carve/carve.json").read_text())["shape"]["widths"]


def map_core_mask(b, map_widths) -> np.ndarray:
    """The core in the map's unit layout [L, max map width]. Where the map unit is a whole expert and the carve unit
    a neuron of it, the expert is core only when every one of its neurons is."""
    core = np.load(b.out / "core/core.npz")["core"] > 0
    cw = carve_widths(b)
    L, N = len(map_widths), max(map_widths)
    out = np.zeros((L, N), bool)
    for l, (mw, w) in enumerate(zip(map_widths, cw)):
        if not mw:
            continue
        if w == mw:
            out[l, :mw] = core[l, :mw]
        elif mw and w % mw == 0:
            out[l, :mw] = core[l, :w].reshape(mw, w // mw).all(1)
        else:
            raise SystemExit(f"layer {l}: map width {mw} and carve width {w} do not nest")
    return out


def _check_capture_disk(b, corpus: Path, d: Path) -> None:
    """The capture writes [texts, units] fp16 per layer (neuron-unit MoE maps are large): refuse before starting when
    it would not fit in 80% of the free disk."""
    n = sum(1 for _ in Path(corpus).open("rb"))
    u = b.info["units"]
    moe_u = u["moe_neuron_units"] if b.plan["moe_units"] == "neuron" else u["moe_expert_units"]
    rolling = (b.plan.get("stream") or {}).get("mode") == "rolling"
    # fp16 responses for every unit; rolling dense units also keep a float32 max and min per text until the finish
    need = n * (u["dense_neurons"] * (2 + (8 if rolling else 0)) + moe_u * 2)
    free = shutil.disk_usage(d).free
    if need > 0.8 * free:
        raise SystemExit(f"map capture needs ~{need / 2**30:.0f} GB on disk ({n} texts), over 80% of the "
                         f"{free / 2**30:.0f} GB free at {d}; subsample (--compute map_probes_per_node=N) or free space")


def run_map(b, ctx):
    d = b.stage_dir("map")
    comp = b.cfg["compute"]
    probes = Path(b.rec["inputs"]["probe_corpus"]["path"])
    sub = None
    if comp["map_probes_per_node"]:
        corpus = d / "probes.jsonl"
        if not corpus.exists():
            sub = _subsample_probes(probes, corpus, comp["map_probes_per_node"], b.cfg["data"]["window_seed"])
            write_json(d / "probes.subsample.json", sub)
        if not (d / "probes.subsample.json").exists():        # killed between the two writes, or an older run
            corpus.unlink()
            sub = _subsample_probes(probes, corpus, comp["map_probes_per_node"], b.cfg["data"]["window_seed"])
            write_json(d / "probes.subsample.json", sub)
        sub = json.loads((d / "probes.subsample.json").read_text())
    else:
        corpus = probes
    cap = d / "capture"
    _reset_partial(cap, "meta.json")
    rolling = (b.plan.get("stream") or {}).get("mode") == "rolling"
    if not cap.exists():
        _check_capture_disk(b, corpus, d)
        bs = (b.plan["stream"].get("bs") or 8) if rolling else b.plan["batches"]["capture"]["bs"]
        common = ["--model", b.info["path"], "--corpus", corpus, "--out", cap,
                  "--seq-len", b.cfg["data"]["window_tokens"], "--bs", bs,
                  "--moe-units", b.plan["moe_units"]] + _mem_args(b, "capture") + (["--cpu"] if ctx.cpu else [])
        st = b.plan.get("stream") or {}
        if rolling:              # s8: the model does not fit (or --stream rolling); a sweep family or the host family
            fam = st.get("family") or "host"
            extra = (["--stream-mode", st["stream_mode"], "--ceiling-gb", st["ceiling_gb"]] if fam == "host" else [])
            if fam != "host" and st.get("expert_budget_gb"):
                extra += ["--family-opt", f"expert_budget_gb={st['expert_budget_gb']}"]
            run_child(_py("mlx2.experimental.amoeba.child_rolling") + common +
                      ["--family", fam, "--window", st.get("window") or 0, "--chunks", st.get("chunks") or 1,
                       "--carry", st.get("carry") or "resident",
                       # host family: full depth (two extra weight reads); a sweep family's reference is a B1 whole-
                       # model forward, so its gate checks the first 2 blocks (its own parity run covers the rest)
                       "--parity-layers", 0 if fam == "host" else 2] + extra,
                      stage="map", progress=ctx.progress, log_path=b.log_path("map"),
                      what=f"map capture (rolling, {fam} family)")
        else:
            run_child(_py("mlx2.experimental.amoeba.child_capture") + common,
                      stage="map", progress=ctx.progress, log_path=b.log_path("map"), what="map capture")
    meta = json.loads((cap / "meta.json").read_text())
    if meta.get("fingerprint") != b.info["fingerprint"]:
        raise SystemExit(f"map capture fingerprint {meta.get('fingerprint')} != build {b.info['fingerprint']}")
    units = int(sum(meta["widths"]))
    core_map = d / "core_units.npz"
    with_core = (b.receipt("core") or {}).get("status") == "done"
    excl = map_core_mask(b, meta["widths"]) if with_core else \
        np.zeros((len(meta["widths"]), max(meta["widths"])), bool)
    np.savez_compressed(core_map, core=excl.astype(np.uint8))
    from .config import cluster_k
    k = comp.get("cluster_k") or cluster_k(units - int(excl.sum()))
    cl = d / "clusters"
    _reset_partial(cl, "clusters.json")
    if not cl.exists():
        run_child(_py("amoeba_map_cluster.py") + ["--capture", cap, "--out", cl, "--k", k,
                                                   "--ve-per-layer", comp["ve_per_layer"]] +
                  (["--exclude", f"{core_map}:core"] if with_core else []),
                  stage="map", progress=ctx.progress, log_path=b.log_path("map"), what="map clusters",
                  env_extra={"MLX2_PREFLIGHT_CPU_ONLY": "1"})
    lab = d / "labels"
    _reset_partial(lab, "labels.json")
    if not lab.exists():
        # The labeller creates two fresh directories; an interrupted run leaves both behind.
        shutil.rmtree(d / "label-arrays", ignore_errors=True)
        run_child(_py("amoeba_topics_label.py") + ["--capture", cap, "--corpus", corpus, "--out", lab,
                                                    "--arrays", d / "label-arrays", "--clusters", cl / "clusters.npz",
                                                    ] + (["--core", core_map] if with_core else ["--no-core"]) +
                  ["--taxonomy", b.rec["inputs"]["taxonomy"]["path"]],
                  stage="map", progress=ctx.progress, log_path=b.log_path("map"), what="map labels",
                  env_extra={"MLX2_PREFLIGHT_CPU_ONLY": "1"})
    labels = json.loads((lab / "labels.json").read_text())
    summ = labels.get("summary") or {}
    m = {"schema": "mlx2.amoeba-map.v1", "checkpoint": b.checkpoint, "fingerprint": b.info["fingerprint"],
         "mlx2_commit": ctx.commit, "probe_corpus": b.rec["inputs"]["probe_corpus"], "capture_corpus": str(corpus),
         "subsample": sub, "units": units, "core_units_excluded": int(excl.sum()),
         "clustered": ("the non-core units (s5.1.2); the core is carved first" if with_core else
                       "every unit: a map-only build (no core yet), the maps-campaign mode"),
         "capture": "rolling layer-window (s8)" if rolling else "resident",
         "rolling": meta.get("rolling") and {k_: v for k_, v in meta["rolling"].items() if k_ != "per_layer"},
         "rolling_parity": meta.get("parity"),
         "unit_kind": f"MLP neurons; MoE layers in {meta.get('moe_units')} units", "widths": meta["widths"],
         "signal": "per-text activation peak over positions >= 1",
         "sequences": meta["sequences"], "k": k, "clusters": summ.get("clusters"), "labelled": summ.get("labelled"),
         "labelled_topics": sorted({x["topic"] for c in labels["clusters"] for x in c.get("labels", [])}),
         "ones_parity": meta.get("ones_parity"), "capture_mlx_peak_gb": meta.get("mlx_peak_gb"),
         "status": "implemented (research map; unqualified)"}
    write_json(d / "map.json", m)
    outs = file_outputs({"map": d / "map.json", "clusters": cl / "clusters.npz", "clusters_json": cl / "clusters.json",
                         "virtual_experts": cl / "virtual_experts.npz", "labels": lab / "labels.json",
                         "capture_meta": cap / "meta.json"})
    return outs, {k_: m[k_] for k_ in ("units", "sequences", "k", "clusters", "labelled")} | \
        {"labelled_topics": len(m["labelled_topics"])}


# ====================================================================== step 2: core

def _english_corpus(b, dst: Path) -> dict:
    rows = []
    for p in b.rec["inputs"]["core_corpora"]:
        for r in read_jsonl(p["path"]):
            rows.append({**r, "source": r.get("source") or Path(p["path"]).name, "core_corpus": Path(p["path"]).name})
    write_jsonl(dst, rows)
    return {"rows": len(rows), "domains": dict(Counter(r.get("domain") for r in rows)),
            "held": sum(r.get("split") == "held" for r in rows)}


def run_core(b, ctx):
    d = b.stage_dir("core")
    comp, data, plan = b.cfg["compute"], b.cfg["data"], b.plan
    eng = d / "english.jsonl"
    ec = _english_corpus(b, eng) if not eng.exists() else None
    gen = generate(b, ctx, "core", [eng], d / "gen", held_per_domain=comp["core_gen_held_per_domain"],
                   free_per_domain=comp["core_gen_free_per_domain"])
    mixed = d / "corpus.jsonl"
    if not mixed.exists():
        write_jsonl(mixed, read_jsonl(eng) + read_jsonl(d / "gen/gen.jsonl"))
    run = d / "carve"
    argv = ["--model", b.info["path"], "--out", run, "--stages", "inspect,mean,rank,bisect", "--ablations", "zero",
            "--steps", comp["core_steps"], "--bs", plan["batches"]["carve_core"]["bs"], "--lr", comp["core_lr"],
            "--corpus", f"jsonl:{mixed}", "--n-mean", comp["core_n_mean"], "--n-fit", comp["core_n_fit"],
            "--n-held", comp["core_n_held"], "--targets", comp["core_targets"],
            "--export-target", comp["core_export_target"],
            "--seq-len", data["window_tokens"], "--windows", data["windows_per_doc"], "--window-seed",
            data["window_seed"], "--strip-prefix", data["strip_prefix"], "--keep-prompt", data["keep_prompt"],
            "--need-gb", plan["need_gb"], "--train-peak-cap-gb", plan["watchdog_gb"] - 2,
            "--moe-units", plan["moe_units"], "--cache-gb", 2] + list(b.cfg["grad_gate"]) + _mem_args(b, "core")
    if data["code_cut"]:
        argv.append("--code-cut")
    if plan.get("assert_ple_offload"):
        argv.append("--assert-ple-offload")
    if ctx.cpu:
        argv += ["--cpu", "--no-admit"]
    if ctx.stop_at:
        argv += ["--stop-at", ctx.stop_at]
    if not (run / "export_core.npz").exists():
        if run.exists():
            argv.append("--resume")
        run_child(_py("research.amoeba.carve") + argv, stage="core", progress=ctx.progress,
                  log_path=b.log_path("core"), what="core carve")
    rep = json.loads((run / "carve.json").read_text())
    if rep.get("fingerprint") != b.info["fingerprint"]:
        raise SystemExit("core carve fingerprint does not match the build")
    core = np.load(run / "export_core.npz")["core"] > 0
    np.savez_compressed(d / "core.npz", core=core.astype(np.uint8))
    chk = d / "corecheck.json"
    if not chk.exists():
        run_child(_py("mlx2.experimental.amoeba.child_probe") +
                  ["corecheck", "--model", b.info["path"], "--core", d / "core.npz",
                   "--probes", b.rec["inputs"]["probe_corpus"]["path"], "--english", eng, "--out", chk,
                   "--window-tokens", data["window_tokens"], "--moe-units", plan["moe_units"],
                   "--per-language", 16 if b.cfg["profile"] == "full" else 4,
                   "--english-rows", 48 if b.cfg["profile"] == "full" else 12,
                   "--english-max", comp["core_export_target"] + 0.05] + _mem_args(b, "core") +
                  (["--cpu"] if ctx.cpu else []),
                  stage="core", progress=ctx.progress, log_path=b.log_path("core"), what="English-only check")
    cc = json.loads(chk.read_text())
    e = rep.get("export_core") or {}
    widths = rep["shape"]["widths"]
    valid = np.arange(core.shape[1])[None, :] < np.asarray(widths)[:, None]
    insp = (rep.get("inspect") or {}).get("paths") or {}
    summ = {"units": int(core.sum()), "pool_units": int(valid.sum()), "share": round(float(core.sum() / valid.sum()), 4),
            "export_held_d_ce": (e.get("held") or {}).get("mean_d_ce"), "english_only": cc.get("english_only"),
            "english_d_ce": cc.get("english_d_ce"), "non_english_median_d_ce": cc.get("non_english_median_d_ce"),
            "ones_parity": insp.get("ones_parity"), "layerwise_grad": insp.get("layerwise_grad"),
            "gen_rows": gen.get("rows"), "gen_route": gen.get("route"), "english_corpus": ec}
    outs = file_outputs({"core": d / "core.npz", "carve": run / "carve.json", "corecheck": chk,
                         "english": eng, "gen": d / "gen/gen.jsonl"})
    return outs, summ


# ====================================================================== step 3: boxes

def _code_box(node: str, prefixes) -> bool:
    return any(node.startswith(p) or node + "." == p for p in prefixes)


def _rename_domain(bd: Path, new: str) -> None:
    for f in bd.glob("*.jsonl"):
        rows = read_jsonl(f)
        write_jsonl(f, [{**r, "domain": new} if r.get("domain") == "box" else r for r in rows])


def _cluster_boxes(b, ctx, bs: Path, existing: list) -> tuple[list, list]:
    """Step 3.1: one box per untagged map cluster.

    Selection never reads held responses: the cluster's fit response picks its top fit probes (carve and val
    documents), and its topics are the primary topics of those fit probes. Its held documents are the held
    probes of those topics (natural text only, a seeded sample), so the held set is defined by topic labels, not
    by the units under test. The same minimum counts as the box-set builder apply; a cluster that falls short is
    listed for probe extension."""
    paths.research_imports()
    import amoeba_topics_lib as T
    from research.amoeba.carve.corpus import _doc_id
    comp, data = b.cfg["compute"], b.cfg["data"]
    md = b.out / "map"
    labels = json.loads((md / "labels/labels.json").read_text())
    z = np.load(md / "clusters/clusters.npz")
    pool, lab = z["pool"], z["labels"].astype(int)
    cap = md / "capture"
    items = json.loads((cap / "items.json").read_text())
    meta = json.loads((cap / "meta.json").read_text())
    fit = np.array([it["split"] == "fit" for it in items])
    fit_idx = np.nonzero(fit)[0]
    unl = [int(c["id"].split("C")[-1]) for c in labels["clusters"] if not fit_tags(c)]   # untagged on fit
    sizes = Counter(lab.tolist())
    unl.sort(key=lambda j: -sizes[j])
    if ctx.max_cluster_boxes >= 0:
        unl = unl[:ctx.max_cluster_boxes]
    if not unl:
        return [], []
    score = np.zeros((len(unl), len(fit_idx)), np.float64)          # fit texts only
    for l in range(meta["layers"]):
        sel = pool[:, 0] == l
        if not sel.any():
            continue
        R = T.load_resp(cap, l)[fit_idx]
        rel = R / (R.mean(0) + 1e-6)
        for i, j in enumerate(unl):
            u = pool[sel & (lab == j), 1]
            if len(u):
                score[i] += rel[:, u].sum(1)
    for i, j in enumerate(unl):
        score[i] /= max(1, sizes[j])
    probes = read_jsonl(b.rec["inputs"]["probe_corpus"]["path"])
    by_id = {r["id"]: r for r in probes}
    build = Path(b.rec["inputs"]["probe_corpus"]["path"]).parent.name

    def row(r):
        return {"text": r["text"], "domain": "box", "doc_id": _doc_id(r["text"]), "probe_id": r["id"],
                "topics": r["topics"], "source": r.get("group", ""), "origin": r.get("origin"),
                "license": r.get("license"), "probe_source": r.get("source"), "method": r.get("method"),
                "path_key": r.get("path_key"), "probe_sha256": r.get("sha256"), "corpus_build": build}
    minlen = 4 * (data["gen_prompt_tokens"] + 16)                    # characters: a prompt-length document
    out, short = [], []
    names = {x["box"] for x in existing}
    for i, j in enumerate(unl):
        name = f"cluster_c{j}"
        if name in names:
            continue
        order = np.argsort(-score[i])
        fit_ids = [items[fit_idx[s]]["id"] for s in order][:comp["cluster_box_docs"]]
        fit_rows = [by_id[x] for x in fit_ids if x in by_id]
        top = [t for t, _ in Counter(r["topics"][0] for r in fit_rows).most_common(3)]
        held_pool = [r for r in probes if r["split"] == "held" and r["topics"][0] in top and r.get("origin") != "synthetic"]
        held_rows = sorted(held_pool, key=lambda r: h(r["id"], 13))[:comp["cluster_box_held"]]
        val = [r for r in fit_rows if h(r.get("group", r["id"]), 7) % 5 == 0 and r.get("origin") != "synthetic"]
        vid = {r["id"] for r in val}
        carve = [r for r in fit_rows if r["id"] not in vid]
        counts = {"carve_windows": 2 * len(carve), "val_windows": len(val), "held_windows": len(held_rows)}
        need = {"carve_windows": comp["box_min_carve"], "val_windows": max(1, comp["box_min_val"] // 4),
                "held_windows": max(2, comp["box_min_held"] // 4)}
        miss = [k for k in need if counts[k] < need[k]]
        if miss:
            short.append({"node": f"cluster:C{j}", **counts, "short": miss, "topics": top})
            continue
        dom = "code" if all(_code_box(t, data["code_topic_prefixes"]) for t in top) else "box"
        bd = bs / name
        bd.mkdir(parents=True, exist_ok=True)
        for k, rs in (("docs", carve), ("val", val), ("held", held_rows)):
            write_jsonl(bd / f"{k}.jsonl", [{**row(r), "domain": dom} for r in rs])
        for k, rs, n in (("carve", carve, comp["box_carve_prompts"]), ("val", val, comp["box_val_prompts"]),
                         ("held", held_rows, comp["box_held_prompts"])):
            pick = sorted((r for r in rs if len(r["text"]) >= minlen), key=lambda r: h(r["id"], 11))[:n]
            write_jsonl(bd / f"prompts_{k}.jsonl", [{**row(r), "domain": dom, "box": name, "role": k} for r in pick])
        out.append({"box": name, "node": f"cluster:C{j}", "kind": "cluster", "cluster": j, "units": sizes[j],
                    "domain": dom, "topics": top, "carve_docs": len(carve), "val_docs": len(val),
                    "held_docs": len(held_rows), **counts,
                    "selection": "carve/val: top fit probes by the cluster's mean relative fit response; held: held "
                                 "probes of the top fit probes' primary topics (never chosen by held responses)"})
    return out, short


def _build_boxset(b, ctx, bs: Path) -> None:
    """The box set, built in ``<bs>.partial`` and renamed when complete, so a failure never leaves a half-made set
    that a resume would take as finished."""
    comp, data = b.cfg["compute"], b.cfg["data"]
    tmp = bs.with_name(bs.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    run_child(_py("amoeba_fn_boxset_build.py") +
              ["--corpus", b.rec["inputs"]["probe_corpus"]["path"], "--out", tmp, "--boxes", ctx.boxes,
               "--min-fit", comp["box_min_fit"], "--carve-prompts", comp["box_carve_prompts"],
               "--val-prompts", comp["box_val_prompts"], "--held-prompts", comp["box_held_prompts"],
               "--model", b.info["path"], "--seq-len", data["window_tokens"],
               "--prompt-min", data["gen_prompt_tokens"] + 16, "--min-carve", comp["box_min_carve"],
               "--min-val", comp["box_min_val"], "--min-held", comp["box_min_held"],
               "--manifest", b.rec["inputs"]["probe_manifest"]["path"]],
              stage="boxes", progress=ctx.progress, log_path=b.log_path("boxes"), what="box set")
    spec = json.loads((tmp / "boxes.json").read_text())
    # step 3.2 order: branch level (the builder's depth-2 branches and languages); leaf splits are a later round
    boxes = sorted(spec["boxes"], key=lambda x: (x["kind"] != "language", -x["carve_docs"]))
    dropped = []
    if ctx.max_boxes:
        dropped = [x["box"] for x in boxes[ctx.max_boxes:]]
        boxes = boxes[:ctx.max_boxes]
        for n in dropped:
            shutil.rmtree(tmp / n, ignore_errors=True)
    for x in boxes:                           # rule 3: code boxes' documents start at their first line of code
        x["domain"] = "code" if _code_box(x["node"], data["code_topic_prefixes"]) else "box"
        if x["domain"] == "code":
            _rename_domain(tmp / x["box"], "code")
    short = []
    if ctx.max_cluster_boxes != 0 and not ctx.boxes:
        cl, short = _cluster_boxes(b, ctx, tmp, boxes)
        boxes += cl
    spec.update(boxes=boxes, dropped_by_max_boxes=dropped, cluster_boxes_too_few_probes=short,
                order="languages first, then subject branches by carve documents; untagged clusters last")
    (tmp / "boxes.json").write_text(json.dumps(spec, indent=1))
    ext = json.loads((tmp / "extend.json").read_text()) if (tmp / "extend.json").exists() else []
    (tmp / "extend.json").write_text(json.dumps(ext + short, indent=1))
    tmp.rename(bs)


def _assign_gen(boxes, bs: Path, gen_path: Path) -> dict:
    """Continuation rows -> every (box, role) that owns the prompt's document. A document several boxes chose (a
    cluster box and a taxonomy box) gives its continuations to all of them; duplicate generations are dropped."""
    owner = defaultdict(set)
    for x in boxes:
        for role in ("carve", "val", "held"):
            for r in read_jsonl(bs / x["box"] / f"prompts_{role}.jsonl"):
                owner[r["doc_id"]].add((x["box"], role))
    g, seen = defaultdict(list), set()
    for r in read_jsonl(gen_path):
        k = (r["doc_id"], r["mode"], r.get("seed"))
        if k in seen:
            continue
        seen.add(k)
        for o in sorted(owner.get(r["doc_id"], ())):
            g[o].append(r)
    return g


def run_boxes(b, ctx):
    d = b.stage_dir("boxes")
    comp, data = b.cfg["compute"], b.cfg["data"]
    bs = d / "boxset"
    if not (bs / "boxes.json").exists():
        shutil.rmtree(bs, ignore_errors=True)
        _build_boxset(b, ctx, bs)
    spec = json.loads((bs / "boxes.json").read_text())
    boxes = spec["boxes"]
    if not boxes:
        raise SystemExit("no box passed the box-set minimum counts (see boxes/boxset/extend.json)")
    ctx.progress.emit("boxes_queued", stage="boxes", boxes=[x["box"] for x in boxes])
    # generation prompts get one domain per box, so rule 2 (shared-template stripping) is judged per topic
    gp = d / "gen-prompts"
    gp.mkdir(exist_ok=True)
    prompts = []
    for x in boxes:
        for role in ("carve", "val", "held"):
            f = gp / f"{x['box']}__{role}.jsonl"
            if not f.exists():
                write_jsonl(f, [{**r, "domain": f"{x.get('domain', 'box')}@{x['box']}"}
                                for r in read_jsonl(bs / x["box"] / f"prompts_{role}.jsonl")])
            prompts.append(f)
    gen = generate(b, ctx, "boxes", prompts, d / "gen")
    g = _assign_gen(boxes, bs, d / "gen/gen.jsonl")
    for x in boxes:
        n = x["box"]
        dom = x.get("domain", "box")
        for outf, base, role in (("corpus.jsonl", "docs.jsonl", "carve"), ("val_full.jsonl", "val.jsonl", "val"),
                                 ("held_full.jsonl", "held.jsonl", "held")):
            if not (bs / n / outf).exists():
                rows = read_jsonl(bs / n / base) + [{**r, "domain": f"gen:{dom}:{r['mode']}"} for r in g.get((n, role), [])]
                write_jsonl(bs / n / outf, rows)
        x["continuations"] = {role: len(g.get((n, role), [])) for role in ("carve", "val", "held")}
    carves = d / "carves"
    spec_child = {"model": b.info["path"], "out": str(carves), "boxset": str(bs), "boxes": boxes,
                  "core": str(b.out / "core/core.npz"), "compute": comp, "data": data, "plan": b.plan,
                  "verdict": b.cfg["verdict"], "grad_gate": b.cfg["grad_gate"], "cpu": ctx.cpu, "stop_at": ctx.stop_at}
    sp = d / "batch-spec.json"
    sp.write_text(json.dumps(spec_child, indent=1))
    run_child(_py("mlx2.experimental.amoeba.child_boxbatch") + ["--spec", sp], stage="boxes", progress=ctx.progress,
              log_path=b.log_path("boxes"), what="box batch")
    ver = {}
    for x in read_jsonl(carves / "verdicts.jsonl"):
        ver[x["box"]] = x                          # the last line per box wins (a rerun after a failure)
    missing = [x["box"] for x in boxes if x["box"] not in ver]
    if missing:
        raise SystemExit(f"box batch ended without verdicts for {missing}")
    _boxes_table(d / "boxes.md", boxes, ver)
    failed = box_failure_policy(boxes, ver)
    extend = json.loads((bs / "extend.json").read_text()) if (bs / "extend.json").exists() else []
    misses = [{"node": ver[n]["node"], "box": n, "export_held": ver[n].get("export_held"),
               "reasons": ver[n].get("miss_reasons")} for n in ver if ver[n]["verdict"] == "miss"]
    gaps = [v["val_held_gap"] for v in ver.values() if v.get("val_held_gap") is not None]
    gap = float(np.median(gaps)) if len(gaps) >= 3 else None
    write_json(d / "next-round.json", {"extend_probes": extend, "misses": misses,
                                       "val_held_gap_median": gap, "val_held_gap_boxes": len(gaps),
                                       "val_target_next_round": None if gap is None else
                                       max(0.0, round(b.cfg["verdict"]["held_target_nats"] - gap, 4)),
                                       "val_target_note": None if gap is None or gap < b.cfg["verdict"]["held_target_nats"]
                                       else "the val->held gap reaches the held target: boxes need better val data",
                                       "val_target_rule": "s5.3.4: aim at +0.05 minus the measured val->held gap "
                                                          "(this round used " + str(b.cfg["verdict"]["val_target_nats"])
                                                          + "); pass --compute val_target_nats=X next round",
                                       "rule": "a miss is fixed with more varied, deeper licensed probes for that topic "
                                               "and a re-carve; never a wider target or a held re-cut"})
    cnt = Counter(v["verdict"] for v in ver.values())
    summ = {"boxes": len(ver), **{k: cnt.get(k, 0) for k in ("pass", "miss", "covered", "failed")},
            "gate_refused": sorted(n for n in failed if ver[n].get("failure") == "gate-refused"),
            "gen_rows": gen.get("rows"), "gen_route": gen.get("route"), "gen_tok_s": gen.get("tok_s_aggregate"),
            "too_few_probes": len(extend)}
    outs = file_outputs({"boxset": bs / "boxes.json", "verdicts": carves / "verdicts.jsonl", "table": d / "boxes.md",
                         "next_round": d / "next-round.json", "gen": d / "gen/gen.jsonl"})
    return outs, summ


def box_failure_policy(boxes, ver) -> list:
    """s7: a per-box gate refusal fails that box only (its units stay unclaimed: remainder tiles) and the build goes on.
    A crash (anything but a gate refusal) stops the stage so resume retries it (5dc586507); so does every box refused.
    Returns the failed boxes."""
    failed = [x["box"] for x in boxes if ver[x["box"]]["verdict"] == "failed"]
    crashed = [n for n in failed if ver[n].get("failure", "crash") != "gate-refused"]
    if crashed:
        raise SystemExit(f"box carves failed: {', '.join(crashed)}; see boxes.md and resume to retry failed carves")
    if failed and len(failed) == len(boxes):
        raise SystemExit(f"every box was refused by the s7 gradient gate ({', '.join(failed)}): nothing to partition")
    return failed


def _boxes_table(path: Path, boxes, ver) -> None:
    f = lambda x: "-" if x is None else f"{x:.3f}"   # noqa: E731
    lines = ["| box | taxonomy / cluster | kind | units | val dCE | held dCE | random same-count | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for x in boxes:
        v = ver[x["box"]]
        lines.append(f"| {x['box']} | {x['node']} | {x['kind']} | {v.get('units', '-')} | {f(v.get('export_val'))} | "
                     f"{f(v.get('export_held'))} | {f(v.get('random_same_count_held'))} | {v['verdict']}"
                     f"{' (' + '; '.join(v.get('miss_reasons') or []) + ')' if v.get('miss_reasons') else ''}"
                     f"{' ' + v['error'][:80] if v.get('error') else ''} |")
    path.write_text("\n".join(lines) + "\n")


# ====================================================================== step 4: partition

def unit_clusters(b, shape) -> np.ndarray:
    """[L, N] map cluster per carve unit (-1 none, e.g. core). The map's unit and the carve's unit agree layer by
    layer, or the map unit is a whole expert holding ``carve width / map width`` carve neurons (e*F .. e*F+F-1)."""
    L, N = shape
    z = np.load(b.out / "map/clusters/clusters.npz")
    mw = json.loads((b.out / "map/capture/meta.json").read_text())["widths"]
    cw = carve_widths(b)
    out = np.full((L, N), -1, np.int32)
    for (l, u), c in zip(z["pool"], z["labels"]):
        if cw[l] == mw[l]:
            out[l, u] = c
        elif mw[l] and cw[l] % mw[l] == 0:
            F = cw[l] // mw[l]
            out[l, u * F:(u + 1) * F] = c
        else:
            raise SystemExit(f"layer {l}: map width {mw[l]} and carve width {cw[l]} do not nest")
    return out


def run_partition(b, ctx):
    d = b.stage_dir("partition")
    core = np.load(b.out / "core/core.npz")["core"] > 0
    rep = json.loads((b.out / "core/carve/carve.json").read_text())
    widths = rep["shape"]["widths"]
    L, N = core.shape
    valid = np.arange(N)[None, :] < np.asarray(widths)[:, None]
    ver = {x["box"]: x for x in read_jsonl(b.out / "boxes/carves/verdicts.jsonl")}
    spec = {x["box"]: x for x in json.loads((b.out / "boxes/boxset/boxes.json").read_text())["boxes"]}
    masks = {}
    for n, v in sorted(ver.items()):
        if v["verdict"] in ("pass", "miss"):
            m = np.load(b.out / f"boxes/carves/box-{n}/box_mask.npz")["box"] > 0
            masks[n] = m & valid & ~core
    names = sorted(masks)
    claims = np.zeros((L, N), np.int32)
    for n in names:
        claims += masks[n]
    to_core = (core | (claims >= 3)) & valid
    tiles = {"core": to_core}
    meta = {"core": {"kind": "core", "boxes": [], "opens_on": "always"}}
    for n in names:
        own = masks[n] & (claims == 1) & ~to_core
        tiles[f"box:{n}"] = own
        meta[f"box:{n}"] = {"kind": "box", "boxes": [n], "opens_on": [n]}
    pairs = {}
    two = (claims == 2) & ~to_core
    if two.any():
        stack = np.stack([masks[n] for n in names])            # [B, L, N]
        first = stack.argmax(0)                                 # lowest claiming box index
        rest = stack.copy()
        np.put_along_axis(rest, first[None], False, axis=0)
        second = rest.argmax(0)
        code = np.where(two, first * len(names) + second, -1)
        for c in np.unique(code[two]):
            pairs[(names[c // len(names)], names[c % len(names)])] = code == c
    for (a, c), m in sorted(pairs.items()):
        tiles[f"pair:{a}+{c}"] = m
        meta[f"pair:{a}+{c}"] = {"kind": "cross-box", "boxes": [a, c], "opens_on": [a, c]}
    rem = valid & ~to_core & (claims == 0)
    uc = unit_clusters(b, (L, N))
    for c in sorted(set(uc[rem].tolist())):
        m = rem & (uc == c)
        key = f"remainder:C{c}" if c >= 0 else "remainder:unclustered"
        tiles[key] = m
        meta[key] = {"kind": "remainder", "boxes": [], "cluster": int(c) if c >= 0 else None, "opens_on": "never"}
    cover = np.zeros((L, N), np.int32)
    for m in tiles.values():
        cover += m
    exact = bool(np.array_equal(cover, valid.astype(np.int32)))
    if not exact:
        raise SystemExit(f"partition not exact: {int((cover > 1).sum())} units twice, "
                         f"{int(((cover == 0) & valid).sum())} units nowhere")
    np.savez_compressed(d / "partition.npz", **{re.sub(r"[^A-Za-z0-9_]", "_", k) + "__mask": v.astype(np.uint8)
                                                 for k, v in tiles.items()})
    keys = {k: re.sub(r"[^A-Za-z0-9_]", "_", k) + "__mask" for k in tiles}
    pool = int(valid.sum())
    rows = []
    for k, m in tiles.items():
        rows.append({"tile": k, "npz_key": keys[k], **meta[k], "units": int(m.sum()),
                     "share": round(float(m.sum() / pool), 6)})
    per_box = {}
    for n in names:
        x = spec.get(n, {})
        shared = [r["tile"] for r in rows if r["kind"] == "cross-box" and n in r["boxes"]]
        per_box[n] = {"taxonomy_id": x.get("node"), "kind": x.get("kind"), "carved_units": int(masks[n].sum()),
                      "own_units": int(tiles[f"box:{n}"].sum()), "units_to_core": int((masks[n] & to_core).sum()),
                      "cross_box_tiles": shared, "verdict": ver[n]["verdict"]}
    jac = {}
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            inter = int((masks[a] & masks[c]).sum())
            if inter:
                jac[f"{a}+{c}"] = round(inter / max(1, int((masks[a] | masks[c]).sum())), 4)
    parity = ((rep.get("inspect") or {}).get("paths") or {}).get("ones_parity")
    out = {"schema": "mlx2.amoeba-partition.v1", "checkpoint": b.checkpoint, "rules": b.cfg["partition_rules"],
           "boxes_in": names, "boxes_left_out": {n: v["verdict"] for n, v in ver.items() if n not in masks},
           "pool_units": pool, "core_units_carved": int(core.sum()), "core_units": int(to_core.sum()),
           "units_to_core_by_3plus": int(((claims >= 3) & ~core & valid).sum()),
           "tiles": rows, "boxes": per_box, "jaccard": jac,
           "check": {"exact_cover": exact, "nothing_pruned": True,
                     "everything_open_equals_full": "the tiles cover every unit exactly once, so all tiles open = the "
                                                    "all-ones mask, whose parity with the stock model the core carve's "
                                                    "inspect stage measured (ones_parity)",
                     "ones_parity": parity}}
    write_json(d / "partition.json", out)
    kinds = Counter(r["kind"] for r in rows)
    return file_outputs({"partition": d / "partition.npz", "partition_json": d / "partition.json"}), \
        {"tiles": len(rows), **{f"{k}_tiles": v for k, v in kinds.items()}, "core_units": out["core_units"],
         "exact_cover": exact}


# ====================================================================== step 5: gating

def run_gates(b, ctx):
    d = b.stage_dir("gates")
    comp, data = b.cfg["compute"], b.cfg["data"]
    part = json.loads((b.out / "partition/partition.json").read_text()) if (b.out / "partition/partition.json").exists() \
        else None
    ver = {x["box"]: x for x in read_jsonl(b.out / "boxes/carves/verdicts.jsonl")}
    names = sorted(part["boxes_in"]) if part else sorted(n for n, v in ver.items() if v["verdict"] in ("pass", "miss"))
    if not names and not (d / "gates.json").exists():      # every box failed or was covered: nothing to gate
        np.savez_compressed(d / "gates.npz")
        write_json(d / "gates.json", {"schema": "mlx2.amoeba-gates.v1", "checkpoint": b.checkpoint, "layer": None,
                                      "boxes": {}, "note": "no box in the partition"})
    if not (d / "gates.json").exists():
        run_child(_py("mlx2.experimental.amoeba.child_probe") +
                  ["gates", "--model", b.info["path"], "--boxset", b.out / "boxes/boxset", "--boxes", ",".join(names),
                   "--bs", b.plan["batches"]["gates"]["bs"],
                   "--english", b.out / "core/english.jsonl", "--out", d, "--window-tokens", data["window_tokens"],
                   "--tokens", comp["gate_tokens_per_window"], "--windows", comp["gate_windows_per_box"],
                   "--negatives", comp["gate_negatives"], "--moe-units", b.plan["moe_units"],
                   "--code-prefixes", ",".join(data["code_topic_prefixes"])] + _mem_args(b, "gates") +
                  (["--cpu"] if ctx.cpu else []),
                  stage="gates", progress=ctx.progress, log_path=b.log_path("gates"), what="gate detectors")
    g = json.loads((d / "gates.json").read_text())
    tr = {n: v for n, v in g["boxes"].items() if v.get("status") == "trained"}
    fo = [v["held_false_open_rate"] for v in tr.values() if v.get("held_false_open_rate") is not None]
    mi = [v["held_miss_rate"] for v in tr.values() if v.get("held_miss_rate") is not None]
    summ = {"layer": g["layer"], "trained": len(tr), "untrained": len(g["boxes"]) - len(tr),
            "held_false_open_max": max(fo) if fo else None, "held_false_open_mean": round(float(np.mean(fo)), 4) if fo else None,
            "held_miss_mean": round(float(np.mean(mi)), 4) if mi else None}
    return file_outputs({"gates": d / "gates.npz", "gates_json": d / "gates.json"}), summ


# ====================================================================== step 6 + section 10: report, register, bundle

def run_bundle(b, ctx):
    d = b.out / "bundle"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    inp = b.rec["inputs"]
    copies = {
        "tiles/partition.npz": b.out / "partition/partition.npz",
        "tiles/partition.json": b.out / "partition/partition.json",
        "tiles/core.npz": b.out / "core/core.npz",
        "gates/gates.npz": b.out / "gates/gates.npz",
        "gates/gates.json": b.out / "gates/gates.json",
        "map/map.json": b.out / "map/map.json",
        "map/clusters.npz": b.out / "map/clusters/clusters.npz",
        "map/clusters.json": b.out / "map/clusters/clusters.json",
        "map/virtual_experts.npz": b.out / "map/clusters/virtual_experts.npz",
        "map/labels.json": b.out / "map/labels/labels.json",
        "taxonomy/" + Path(inp["taxonomy"]["path"]).name: Path(inp["taxonomy"]["path"]),
        "taxonomy/probe-corpus-manifest.json": Path(inp["probe_manifest"]["path"]),
        "report/boxes.md": b.out / "boxes/boxes.md",
        "report/verdicts.jsonl": b.out / "boxes/carves/verdicts.jsonl",
        "report/next-round.json": b.out / "boxes/next-round.json",
        "report/corecheck.json": b.out / "core/corecheck.json",
    }
    for dst, src in copies.items():
        (d / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d / dst)
    # prompt set: every text used in mapping and carving (section 10), Parquet + manifest
    ap = paths.arrow_python()
    if ap is None:
        raise SystemExit("bundle: no interpreter with pyarrow for the prompt-set Parquet (set MLX2_AMOEBA_ARROW_PYTHON)")
    from .inspection import slug
    name = f"prompt-set-{slug(b.checkpoint)}-{b.info['fingerprint'][:8]}"
    runs = ["--carve-run", b.out / "core/carve", "--stage", "core"]
    ver = {x["box"]: x for x in read_jsonl(b.out / "boxes/carves/verdicts.jsonl")}
    for n in sorted(ver):
        rd = b.out / f"boxes/carves/box-{n}"
        if ver[n]["verdict"] == "failed":       # never carved: its planned inputs are not texts the tile set used
            continue
        if (rd / "inputs.jsonl").exists():
            runs += ["--carve-run", rd, "--stage", f"box:{n}"]
    runs += ["--capture-run", b.out / "map/capture", "--stage", "map"]
    run_child([ap, "-u", paths.repo_root() / "scripts/amoeba_prompt_set.py", "--out", d / "prompt-set", "--name", name,
               "--probe-corpus", inp["probe_corpus"]["path"], "--checkpoint", b.checkpoint,
               "--taxonomy-version", inp["taxonomy"]["version"]] + runs,
              stage="bundle", progress=ctx.progress, log_path=b.log_path("bundle"), what="prompt set")
    ps_man = json.loads((d / f"prompt-set/{name}.manifest.json").read_text())
    # registry entries (step 6)
    part = json.loads((b.out / "partition/partition.json").read_text())
    gates = json.loads((b.out / "gates/gates.json").read_text())["boxes"]
    reg = []
    for n, v in sorted(ver.items()):              # every box, step 6 (failed and covered boxes hold no tiles)
        if n not in part["boxes"]:
            reg.append({"schema": "mlx2.amoeba-box-registry.v1", "taxonomy_id": v.get("node"), "box": n,
                        "kind": v.get("kind"), "units": 0, "verdict": v["verdict"], "error": v.get("error"),
                        "failure": v.get("failure"), "gate_cells": v.get("gate_cells"),
                        "unclaimed_units": "remainder tiles" if v["verdict"] == "failed" else None,
                        "core_alone_val": v.get("core_alone_val"), "checkpoint": b.checkpoint,
                        "fingerprint": b.info["fingerprint"], "mlx2_commit": ctx.commit, "profile": b.cfg["profile"]})
            continue
        pb = part["boxes"][n]
        reg.append({"schema": "mlx2.amoeba-box-registry.v1", "taxonomy_id": pb["taxonomy_id"], "box": n,
                    "kind": pb["kind"], "units": pb["own_units"], "carved_units": pb["carved_units"],
                    "share_of_pool": round(pb["own_units"] / part["pool_units"], 6),
                    "held_d_ce": {"pooled": v.get("export_held"), **(v.get("held_split") or {})},
                    "val_d_ce": v.get("export_val"), "random_same_count_held": v.get("random_same_count_held"),
                    "random_margin": v.get("random_margin"), "verdict": v["verdict"],
                    "gate": {k: (gates.get(n) or {}).get(k) for k in ("status", "layer", "threshold", "held_false_open_rate",
                                                                       "held_miss_rate", "held_false_open_english")},
                    "cross_box_tiles": pb["cross_box_tiles"], "checkpoint": b.checkpoint,
                    "fingerprint": b.info["fingerprint"], "mlx2_commit": ctx.commit, "profile": b.cfg["profile"]})
    write_jsonl(d / "registry.jsonl", reg)
    files = {}
    for f in sorted(d.rglob("*")):
        if f.is_file():
            files[str(f.relative_to(d))] = {"sha256": paths.sha256_file(f), "bytes": f.stat().st_size}
    status = "quick profile: a pipeline check, never a box-quality claim" if b.cfg["profile"] != "full" else \
        "implemented (research tile set; unqualified)"
    man = {"schema": "mlx2.amoeba-bundle.v1", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "checkpoint": b.checkpoint, "fingerprint": b.info["fingerprint"], "artifact": b.info["artifact"],
           "mlx2_commit": ctx.commit, "profile": b.cfg["profile"], "status": status,
           "taxonomy": {**inp["taxonomy"], "file": "taxonomy/" + Path(inp["taxonomy"]["path"]).name},
           "probe_corpus": inp["probe_corpus"], "probe_manifest": inp["probe_manifest"],
           "map": {"file": "map/map.json", "sha256": files["map/map.json"]["sha256"],
                   "clusters_sha256": files["map/clusters.npz"]["sha256"]},
           "prompt_set": {"file": f"prompt-set/{name}.parquet", "manifest": f"prompt-set/{name}.manifest.json",
                          "sha256": ps_man["sha256"], "rows": ps_man["rows"], "counts": ps_man["counts"]},
           "tiles": {"file": "tiles/partition.npz", "index": "tiles/partition.json",
                     "sha256": files["tiles/partition.npz"]["sha256"], "count": len(part["tiles"]),
                     "exact_cover": part["check"]["exact_cover"], "unit_layout": "[layers, max width] masks, uint8"},
           "gates": {"file": "gates/gates.npz", "index": "gates/gates.json"},
           "config": b.cfg, "files": files,
           "serving": "phase 3 (bundle loader, mlx2 serve --amoeba-bundle) is not built yet"}
    write_json(d / "manifest.json", man)
    return file_outputs({"manifest": d / "manifest.json", "registry": d / "registry.jsonl"}), \
        {"files": len(files), "prompt_set_rows": ps_man["rows"], "registry_entries": len(reg),
         "bundle": str(d)}


RUNNERS = {"map": run_map, "core": run_core, "boxes": run_boxes, "partition": run_partition, "gates": run_gates,
           "bundle": run_bundle}
