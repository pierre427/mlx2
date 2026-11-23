"""``mlx2 amoeba``: carve a model into an English core, boxes, cross-box tiles and gates, and bundle them.

    mlx2 amoeba build  --model DIR --out DIR [--profile full|quick] [--cpu] [--metrics-port 9466]
    mlx2 amoeba resume --out DIR [--redo STAGE]
    mlx2 amoeba status --out DIR [--json]
    mlx2 amoeba inspect --model DIR          model identity and layout, no weights loaded
    mlx2 amoeba plan --model DIR             the resource plan (refuses when it will not fit)
    mlx2 amoeba {map,core,boxes,partition,gates,bundle} --out DIR [--redo]
    mlx2 amoeba rolling-parity --model DIR --out DIR   rolling vs resident capture (models that fit)

Experimental and unqualified (mlx2.experimental.amoeba). The procedure, the fixed data defaults and the
verdict rules are docs/research/2026-11-13-amoeba-guide/mapping-workflow.md. GPU stages run only inside
an owned GPU window (scripts/run_with_gpu_locks.py); --cpu runs everything on the CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import STAGES, config as K, paths
from .inspection import inspect_checkpoint
from .plan import format_plan, make_plan
from .progress import MetricsServer, Progress
from .runner import ChildFailed, CleanStop, StageFailedContinue, require_gpu_window
from .state import Build, now
from .stages import RUNNERS, Ctx

MODEL_STAGES = {"map", "core", "boxes", "gates", "gap", "box", "box_gate", "box_bundle",
                "route_side", "route_tile", "route_lora", "route_bridge"}


def kind_stages(b) -> tuple:
    if b.rec.get("kind") == "extract":
        from .extract import EXTRACT_STAGES
        return EXTRACT_STAGES
    if b.rec.get("kind") == "install":
        from .install import INSTALL_STAGES
        return INSTALL_STAGES
    return STAGES


def kind_runners(b) -> dict:
    if b.rec.get("kind") == "extract":
        from .extract import EXTRACT_RUNNERS
        return EXTRACT_RUNNERS
    if b.rec.get("kind") == "install":
        from .install import INSTALL_RUNNERS
        return INSTALL_RUNNERS
    return RUNNERS


def _common_plan_args(p):
    p.add_argument("--profile", default="full", choices=sorted(K.PROFILES))
    p.add_argument("--cpu", action="store_true", help="run every model stage on the CPU")
    p.add_argument("--cap-gb", type=float, default=None, help="per-process memory cap (default 80 GB)")
    p.add_argument("--cap-reason", default="", help="who approved a cap over 80 GB")
    p.add_argument("--min-free-gb", type=float, default=15.0,
                   help="core carve's free+inactive floor in GiB (default 15); a resident model near the "
                        "cap needs less, e.g. Hy4-flash at 104 GB")
    p.add_argument("--moe-units", choices=["expert", "neuron", "block64"], default=None,
                   help="MoE unit granularity (default: neuron, the workflow's choice for MoE models)")
    p.add_argument("--compute", action="append", default=[], metavar="KEY=VALUE", help="override a compute setting")
    p.add_argument("--stream", choices=["auto", "rolling", "resident"], default="auto",
                   help="auto (default): rolling layer-window capture exactly when the resident load does not fit "
                        "under the cap (mapping-workflow s8); rolling: force it (parity runs); resident: never stream")
    p.add_argument("--window", type=int, default=None, help="rolling: layers read ahead (default: from free memory)")


def _run_args(p):
    p.add_argument("--metrics-port", type=int, default=9466, help="loopback Prometheus exporter (0 = off)")
    p.add_argument("--stop-at", default="", help="clean stop: HH:MM (next occurrence) or epoch seconds")
    p.add_argument("--quiet", action="store_true", help="no terminal progress line")
    p.add_argument("--gpu-window-owner", default="", help="the job that owns the GPU window for this run")
    p.add_argument("--ignore-gpu-queue", default="", metavar="REASON", help="bypass only GPU-window admission; recorded with lock holder")
    p.add_argument("--allow-dirty", action="store_true", help="run from a tree with uncommitted tracked changes")
    p.add_argument("--swap-growth-gb", type=float, default=0.0,
                   help="stop when system swap grows more than this over the run's start (0 = no guard); needs "
                        "--swap-waiver")
    p.add_argument("--swap-waiver", default="", help="who approved the --swap-growth-gb allowance (recorded)")


def parse(argv):
    ap = argparse.ArgumentParser(prog="mlx2 amoeba", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect", help="model identity and layout (no weights loaded)")
    p.add_argument("--model", required=True)
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("plan", help="resource plan for a build")
    p.add_argument("--model", required=True)
    p.add_argument("--json", action="store_true")
    _common_plan_args(p)
    p = sub.add_parser("build", help="run every stage (or --stages)")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stages", default="", help=f"comma list (default: {','.join(STAGES)}; a model that needs "
                   "rolling capture runs map only)")
    _common_plan_args(p)
    _run_args(p)
    p.add_argument("--probe-corpus", default=str(paths.DEFAULT_PROBE_CORPUS))
    p.add_argument("--probe-manifest", default="", help="default: the manifest at the pinned commit "
                   f"{paths.DEFAULT_PROBE_MANIFEST_REV}")
    p.add_argument("--taxonomy", default="")
    p.add_argument("--core-corpus", action="append", default=[], help="English prose jsonl (repeatable)")
    p.add_argument("--boxes", default="", help="only these box-set boxes (comma list; no cluster boxes)")
    p.add_argument("--max-boxes", type=int, default=0, help="at most this many taxonomy boxes (0 = all)")
    p.add_argument("--max-cluster-boxes", type=int, default=-1, help="at most this many untagged-cluster boxes "
                   "(-1 = all, 0 = none)")
    p.add_argument("--gen-backend", choices=["local", "server"], default=None,
                   help="local: ordinary decode in a child (default on CPU); server: an mlx2 dev server on the model's "
                        "best route (default on GPU)")
    p.add_argument("--gen-route", default="", help="server backend: route flags (empty = the adapter's best route)")
    p.add_argument("--data-override", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--override-reason", default="")
    brain_default = str(Path.home() / "mlx-models/Qwen3.5-4B-MLX-4bit")
    p = sub.add_parser("auto", help="bounded 4B skill-ingestion harness (experimental smoke test)")
    p.add_argument("--goal", default="", help="what to ingest (required unless --self)")
    p.add_argument("--host", default="", help="the model that should gain the skill (--self: the brain)")
    p.add_argument("--self", dest="self_mode", action="store_true",
                   help="ingest into the brain itself: host = --brain, the skills the harness uses, donors that share "
                        "the brain's tokenizer; a passing box goes on to install (measured, never adopted)")
    p.add_argument("--candidates", nargs="+", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--brain", default=brain_default)
    p.add_argument("--tracker", default="", help="tracker sqlite (default: research-artifacts/amoeba/agent/)")
    p.add_argument("--no-tracker", action="store_true", help="no cross-run memory: no reuse, no lessons, no labels")
    p.add_argument("--minutes", type=float, default=0, help="time budget: no action or child starts after it; running "
                   "children get --stop-at it and are stopped 2 min later (0 = none)")
    _run_args(p)
    p = sub.add_parser("auto-status", help="the auto harness's tracker: competence, lessons, open jobs")
    p.add_argument("--tracker", default="")
    p.add_argument("--reset", action="append", default=[], metavar="JOB_KEY",
                   help="send a job back to the queue (a stale reuse or a fixed bug); repeatable")
    p = sub.add_parser("auto-eval", help="replay oracle-labelled auto decisions through another brain checkpoint")
    p.add_argument("--brain", default=brain_default)
    p.add_argument("--tracker", default="")
    p.add_argument("--limit", type=int, default=0, help="only the newest N labelled decisions")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--gpu-window-owner", default="")
    p.add_argument("--ignore-gpu-queue", default="", metavar="REASON")
    p = sub.add_parser("extract", help="one topic's box from a donor, as a portable box bundle (s11.1)")
    p.add_argument("--model", required=True, help="the donor")
    p.add_argument("--box-fractions", default="", metavar="F1,F2,...",
                   help="also cut the box at these fractions of ALL units (beyond the core), e.g. 0.05,0.10,0.15; each "
                        "cut is saved for the removal check (child_removal), which picks the export size")
    p.add_argument("--topic", required=True, help="a taxonomy.v1.json node id, or several joined by '+' for one box "
                   "carved from all of them (lang.cmn+lang.jpn+lang.hin) (e.g. sci.biology, sci.physics, prog.python, "
                   "prog.rust, lang.fra); unknown ids refuse")
    p.add_argument("--out", required=True)
    p.add_argument("--against", default="", help="the host the donor must beat on the topic (the donor-gap screen)")
    p.add_argument("--gap-only", action="store_true", help="stop after the donor-gap screen; resume the remaining stages later")
    p.add_argument("--skip-gap-screen", default="", metavar="REASON", help="no screen (recorded in the bundle)")
    p.add_argument("--force-extract", default="", metavar="REASON", help="carve even if the donor does not lead")
    p.add_argument("--attention-units", choices=["none", "heads"], default="none")
    _common_plan_args(p)
    _run_args(p)
    p.add_argument("--probe-corpus", default=str(paths.DEFAULT_PROBE_CORPUS))
    p.add_argument("--probe-manifest", default="")
    p.add_argument("--taxonomy", default="")
    p.add_argument("--core-corpus", action="append", default=[])
    p.add_argument("--gen-backend", choices=["local", "server"], default=None)
    p.add_argument("--gen-route", default="")
    p.add_argument("--data-override", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--override-reason", default="")
    p.add_argument("--core-from", default="", metavar="NPZ[:KEY]",
                   help="use this core mask instead of carving one (dense models; key default 'core')")
    p.add_argument("--core-from-reason", default="", help="who chose the --core-from core (recorded)")
    p.add_argument("--no-map", default="", metavar="REASON", help="no map stage: the box carve starts unseeded")
    p.add_argument("--exclude-texts", action="append", default=[], metavar="FILE",
                   help="JSON {probe_ids, sha256, groups}: texts no stage may see (repeatable)")
    p.add_argument("--extra-topic-corpus", action="append", default=[], metavar="FILE",
                   help="more probe-format rows for the effective probe corpus (repeatable)")
    from .install import INSTALL_STAGES, add_parser as add_install_parser
    add_install_parser(sub, _common_plan_args, _run_args)
    p = sub.add_parser("rolling-parity", help="rolling capture vs the ordinary resident capture on the same texts "
                       "(mapping-workflow s8 item 7: before trusting rolling on a model family)")
    p.add_argument("--model", required=True, help="a model that FITS (the resident side needs it)")
    p.add_argument("--out", required=True)
    p.add_argument("--texts", type=int, default=64)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--chunks", type=int, default=2, help="> 1 also exercises the corpus split")
    p.add_argument("--carry", choices=["resident", "memmap"], default="resident")
    p.add_argument("--rolling-resident", action="store_true",
                   help="the rolling side loads resident: checks the layer-major schedule only (an MoE model no sweep "
                        "family streams)")
    p.add_argument("--streamed-only", action="store_true",
                   help="a model that does not fit: run only the streamed capture and its full-depth gate (schedule vs "
                        "the family's own streamed text-major forward); no resident side")
    p.add_argument("--probe-corpus", default=str(paths.DEFAULT_PROBE_CORPUS))
    p.add_argument("--moe-units", choices=["expert", "neuron", "block64"], default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--gpu-window-owner", default="")
    p.add_argument("--swap-growth-gb", type=float, default=0.0,
                   help="stop when system swap grows more than this over the run's start (0 = no guard); needs "
                        "--swap-waiver")
    p.add_argument("--swap-waiver", default="", help="who approved the --swap-growth-gb allowance (recorded)")
    p = sub.add_parser("resume", help="continue a build: done stages are skipped, the rest run")
    p.add_argument("--out", required=True)
    p.add_argument("--stages", default="")
    p.add_argument("--redo", action="append", default=[], help="rerun this stage (and what follows) from scratch")
    _run_args(p)
    p = sub.add_parser("status", help="stage receipts and box verdicts of a build")
    p.add_argument("--out", required=True)
    p.add_argument("--json", action="store_true")
    from .extract import EXTRACT_STAGES
    for st in list(STAGES) + [x for x in EXTRACT_STAGES if x not in STAGES] + \
            [x for x in INSTALL_STAGES if x not in STAGES and x not in EXTRACT_STAGES]:
        p = sub.add_parser(st, help=f"run stage {st} of an existing build" if st in STAGES
                           else f"run stage {st} of an existing extraction" if st in EXTRACT_STAGES
                           else f"run stage {st} of an existing install")
        p.add_argument("--out", required=True)
        p.add_argument("--redo", action="store_true", help="discard this stage's receipt and rerun it")
        _run_args(p)
    return ap.parse_args(argv)


def stop_at(s: str) -> float | None:
    if not s:
        return None
    if ":" in s:
        hh, mm = (int(x) for x in s.split(":"))
        t = time.localtime()
        ts = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, hh, mm, 0, 0, 0, -1))
        return ts if ts > time.time() else ts + 86400
    return float(s)


def resolve_inputs(a, out_tmp: Path) -> dict:
    pc = Path(a.probe_corpus).expanduser().resolve()
    if not pc.exists():
        raise SystemExit(f"probe corpus {pc} does not exist")
    sha = paths.sha256_file(pc)
    man = Path(a.probe_manifest).expanduser().resolve() if a.probe_manifest else \
        paths.git_show(paths.DEFAULT_PROBE_MANIFEST_REV, "docs/research/amoeba-topics/probe-corpus-manifest.json",
                       out_tmp / "inputs/probe-corpus-manifest.json")
    m = json.loads(man.read_text())
    if m.get("corpus_sha256") != sha:
        raise SystemExit(f"probe manifest {man} is for corpus {m.get('corpus_sha256')}, not {pc} ({sha}): refusing")
    tax = Path(a.taxonomy).expanduser().resolve() if a.taxonomy else paths.default_taxonomy()
    t = json.loads(tax.read_text())
    cores = [Path(x).expanduser().resolve() for x in a.core_corpus] or list(paths.DEFAULT_CORE_CORPORA)
    for c in cores:
        if not c.exists():
            raise SystemExit(f"core corpus {c} does not exist")
    return {"probe_corpus": {"path": str(pc), "sha256": sha, "build": pc.parent.name},
            "probe_manifest": {"path": str(man), "sha256": paths.sha256_file(man), "taxonomy_version": m.get("taxonomy_version"),
                               "commit": None if a.probe_manifest else paths.DEFAULT_PROBE_MANIFEST_REV},
            "taxonomy": {"path": str(tax), "sha256": paths.sha256_file(tax), "version": str(t.get("version")),
                         "probe_tagging_version": m.get("taxonomy_version"), "nodes": len(t.get("nodes") or [])},
            "core_corpora": [{"path": str(c), "sha256": paths.sha256_file(c)} for c in cores]}


def check_commit(allow_dirty: bool) -> dict:
    c = paths.git_rev()
    if c["dirty"] and not allow_dirty:
        raise SystemExit(f"mlx2 tree at {c['commit']} has uncommitted tracked changes: every receipt names the commit "
                         "it ran from, so commit first (or --allow-dirty for a code test)")
    return c


def map_texts(probe_corpus: str, cfg: dict) -> int:
    """Texts the map capture will hold state for: the probe corpus rows, or the per-node subsample's bound."""
    p = Path(probe_corpus).expanduser()
    if not p.exists():
        return 46000
    with p.open("rb") as f:
        n = sum(1 for _ in f)
    per = cfg["compute"].get("map_probes_per_node") or 0
    return min(n, per * 2 * 1500) if per else n


def _same_tokenizer(m1, m2):
    f1, f2 = Path(m1).expanduser() / "tokenizer.json", Path(m2).expanduser() / "tokenizer.json"
    if not (f1.exists() and f2.exists()):
        return None
    return paths.sha256_file(f1) == paths.sha256_file(f2)


def topic_nodes(topic: str) -> list[str]:
    """``--topic a+b+c``: one box carved from several taxonomy nodes together (each with every node below it)."""
    nodes = [t.strip() for t in str(topic).split("+") if t.strip()]
    if not nodes or len(set(nodes)) != len(nodes):
        raise SystemExit(f"--topic {topic!r}: expected one node id or distinct ids joined by '+'")
    return nodes


def topic_matches(topic: str, t: str) -> bool:
    """Whether probe topic ``t`` is at or below any node of ``topic`` (one id or ids joined by '+')."""
    return any(t == n or t.startswith(n + ".") for n in topic_nodes(topic))


def check_topic(topic: str, taxonomy: Path) -> None:
    """--topic must be a node of the taxonomy (ids are permanent; a deprecated node names its replacement); several
    nodes joined by '+' are each checked."""
    if "+" in topic:
        for t in topic_nodes(topic):
            check_topic(t, taxonomy)
        return
    nodes = json.loads(Path(taxonomy).read_text()).get("nodes", [])
    by = {n.get("id"): n for n in nodes}
    if topic in by:
        n = by[topic]
        if n.get("deprecated") or n.get("status") == "deprecated":
            raise SystemExit(f"--topic {topic} is deprecated in {taxonomy}: use {n.get('replaced_by') or n.get('see')}")
        return
    import difflib
    near = difflib.get_close_matches(topic, list(by), n=6, cutoff=0.6)
    kids = sorted(i for i in by if i.startswith(topic.split(".")[0] + "."))[:12] if not near else []
    raise SystemExit(f"--topic {topic}: not a node of {taxonomy}" +
                     (f"; did you mean {', '.join(near)}?" if near else (f"; nodes under {topic.split('.')[0]}: "
                                                                         f"{', '.join(kids)}" if kids else "")))


def cmd_build(a) -> int:
    kind = "extract" if a.cmd == "extract" else "build"
    extract = None
    if kind == "extract":
        check_topic(a.topic, Path(a.taxonomy).expanduser() if a.taxonomy else paths.default_taxonomy())
        if a.attention_units != "none":
            raise SystemExit("--attention-units heads: attention heads are not carvable units in the carve host yet "
                             "(research/amoeba/carve/units.py has MLP and MoE sites only); not built")
        if not a.against and not a.skip_gap_screen:
            raise SystemExit("extract needs --against HOST for the donor-gap screen, or --skip-gap-screen REASON")
        if "+" in a.topic and not a.skip_gap_screen:
            raise SystemExit("a multi-topic box (--topic a+b) has no donor-gap screen: pass --skip-gap-screen REASON")
        if a.box_fractions:
            fr = [float(x) for x in a.box_fractions.split(",") if x.strip()]
            if not fr or any(not 0 < x < 1 for x in fr) or fr != sorted(set(fr)):
                raise SystemExit("--box-fractions: ascending distinct fractions of ALL units in (0, 1)")
            a.compute = list(a.compute) + [f"box_fractions={','.join(f'{x:g}' for x in fr)}"]
        from .extract import EXTRACT_STAGES
        a.stages = "gap" if a.gap_only else ",".join(EXTRACT_STAGES)
        a.boxes, a.max_boxes, a.max_cluster_boxes = "", 0, 0
        if a.core_from_reason and not a.core_from:
            raise SystemExit("--core-from-reason without --core-from")
        extract = {"topic": a.topic, "skip_gap_screen": a.skip_gap_screen or None,
                   "force_extract": a.force_extract or None, "attention_units": a.attention_units}
        if a.against:
            hi = inspect_checkpoint(a.against)
            hcfg = K.resolve(a.profile, K.parse_overrides(a.compute), K.parse_overrides(a.data_override),
                             a.override_reason)
            hplan = make_plan(hi, hcfg, cap_gb=a.cap_gb, cap_reason=a.cap_reason, cpu=a.cpu, stream="resident")
            if not a.skip_gap_screen and not hplan["fits"]:
                print(format_plan(hi, hplan), file=sys.stderr)
                raise SystemExit("--against: the host does not fit the memory plan for the gap screen")
            extract["against"] = {"path": hi["path"], "checkpoint": hi["checkpoint"], "fingerprint": hi["fingerprint"],
                                  "plan": hplan, "same_tokenizer": _same_tokenizer(a.model, a.against)}
    info = inspect_checkpoint(a.model)
    cfg = K.resolve(a.profile, K.parse_overrides(a.compute), K.parse_overrides(a.data_override), a.override_reason)
    plan = make_plan(info, cfg, cap_gb=a.cap_gb, cap_reason=a.cap_reason, moe_units=a.moe_units, cpu=a.cpu,
                     stream=a.stream, window=a.window, texts=map_texts(a.probe_corpus, cfg))
    plan["min_free_gb"] = round(float(a.min_free_gb), 1)
    print(format_plan(info, plan), file=sys.stderr)
    if not plan["fits"]:
        return 2
    core_from = None
    if kind == "extract" and a.core_from:
        from .extract import external_core
        core_from = external_core(a.core_from, a.core_from_reason, info)
    check_swap_args(a)
    commit = check_commit(a.allow_dirty)
    out = Path(a.out).expanduser().resolve()
    if out.exists() and any(p.name != ".DS_Store" for p in out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: use `mlx2 amoeba resume --out {out}` or a fresh --out")
    stages = [s for s in a.stages.split(",") if s] or list(STAGES)
    if plan["stream"]["mode"] == "rolling":
        if kind == "extract":
            raise SystemExit("extract needs the donor resident (the gap screen, carve and gate are not streamed): the "
                             "plan chose rolling capture for this donor; the streamed backward is not built")
        carve = [s for s in stages if s != "map"]
        if a.stages and carve:
            raise SystemExit(f"rolling capture: stages {carve} need a carve or a resident model (the streamed backward "
                             "is not built); run --stages map")
        stages = ["map"]
        print("rolling capture: this build runs the map only (all units: no core to exclude)", file=sys.stderr)
    from .extract import EXTRACT_STAGES
    bad = set(stages) - set(EXTRACT_STAGES if kind == "extract" else STAGES)
    if bad:
        raise SystemExit(f"unknown stages {sorted(bad)}")
    if out.exists() and any(p.name != ".DS_Store" for p in out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty: use `mlx2 amoeba resume --out {out}` or a fresh --out")
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="amoeba-inputs-"))
    try:
        inputs = resolve_inputs(a, tmp)
        want_eff = kind == "extract" and (a.exclude_texts or a.extra_topic_corpus)
        if want_eff:
            # built before the build exists and recorded in its first build.json (at its final paths), so no build
            # can run or resume without it: an interrupted creation leaves a record whose file check fails
            from .extract import build_effective_corpus
            eff = build_effective_corpus(Path(inputs["probe_corpus"]["path"]), tmp / "dry", a.exclude_texts,
                                         a.extra_topic_corpus)
            for k in ("path", "accounting"):
                eff[k] = str(out / Path(eff[k]).relative_to(tmp / "dry"))
            inputs["probe_corpus_effective"] = eff
        gen_backend = a.gen_backend or ("local" if a.cpu else "server")
        device = "cpu" if a.cpu else "gpu"
        opts = {"cpu": a.cpu, "gen_backend": gen_backend, "gen_route": a.gen_route, "boxes": a.boxes,
                "max_boxes": a.max_boxes, "max_cluster_boxes": a.max_cluster_boxes,
                "swap_guard": {"growth_gb": a.swap_growth_gb or None, "waiver": a.swap_waiver or None}}
        stage_options = {"map": {"device": device}, "core": {"device": device, "gen_backend": gen_backend},
                         "boxes": {"device": device, "gen_backend": gen_backend, "boxes": a.boxes,
                                   "max_boxes": a.max_boxes, "max_cluster_boxes": a.max_cluster_boxes},
                         "gates": {"device": device}}
        if kind == "extract":
            stage_options = {"gap": {"device": device, "extract": extract}, "map": stage_options["map"],
                             "core": stage_options["core"], "box": {"device": device, "gen_backend": gen_backend,
                                                                    "topic": a.topic},
                             "box_gate": {"device": device}, "box_bundle": {"device": device}}
            if core_from:
                stage_options["core"] = {**stage_options["core"], "core_from": core_from}
            if a.no_map:
                stage_options["map"] = {**stage_options["map"], "skip": a.no_map}
        if "core" not in stages and kind != "extract":
            stage_options["map"]["without_core"] = True
        full_stages = stages
        if kind == "extract" and getattr(a, "gap_only", False):
            from .extract import EXTRACT_STAGES
            full_stages = list(EXTRACT_STAGES)
        rec = {"kind": kind, "extract": extract,
               "model": info["path"], "checkpoint": info["checkpoint"], "inspection": info, "config": cfg,
               "plan": plan, "inputs": inputs, "options": opts, "stage_options": stage_options,
               # --gap-only stops this run after the screen; the build's stages stay the whole extraction, so a
               # plain resume continues it
               "stages": full_stages,
               "mlx2_commit": commit, "argv": sys.argv, "status": "implemented (research pipeline; unqualified)"}
        b = Build.create(out, rec)
        if (tmp / "inputs").exists():             # the pinned default manifest, extracted from git
            shutil.copytree(tmp / "inputs", out / "inputs")
            b.rec["inputs"]["probe_manifest"]["path"] = str(out / "inputs/probe-corpus-manifest.json")
            b.save()
        if want_eff:
            shutil.copytree(tmp / "dry/inputs", out / "inputs", dirs_exist_ok=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return execute(b, stages, a, commit, redo=[])


def swap_settings(b, a) -> tuple:
    """This command's --swap-growth-gb / --swap-waiver, else the build's recorded guard (a resume keeps it)."""
    if getattr(a, "swap_growth_gb", 0.0):
        return a.swap_growth_gb, a.swap_waiver
    g = (b.rec.get("options") or {}).get("swap_guard") or {}
    return g.get("growth_gb") or 0.0, g.get("waiver")


def check_swap_args(a) -> None:
    g = getattr(a, "swap_growth_gb", 0.0) or 0.0
    if g < 0:
        raise SystemExit("--swap-growth-gb must be >= 0")
    if g and not getattr(a, "swap_waiver", ""):
        raise SystemExit("--swap-growth-gb needs --swap-waiver (who approved the allowance)")


def ctx_for(b, a, commit, progress) -> Ctx:
    o = b.rec["options"]
    c = Ctx(progress, cpu=o["cpu"], commit=commit, gen_backend=o["gen_backend"], stop_at=stop_at(a.stop_at),
            boxes=o["boxes"], max_boxes=o["max_boxes"], max_cluster_boxes=o["max_cluster_boxes"],
            gen_route=o.get("gen_route", ""))
    return c


def execute(b: Build, stages: list, a, commit: dict, redo: list) -> int:
    """Run ``stages`` under the build directory's lock (one run per build: a second one refuses)."""
    from . import runner as _runner
    b.acquire_lock()
    _runner.CHILD_GROUP_HOOK["fn"] = b.note_child_group      # the lock names every child's process group
    try:
        b.refuse_if_running_elsewhere()
        return _execute(b, stages, a, commit, redo)
    finally:
        _runner.CHILD_GROUP_HOOK["fn"] = None
        b.release_lock()


def commit_notice(b: Build, commit: dict) -> str | None:
    """The line a resume prints when earlier stages ran at another mlx2 commit (each receipt, and every manifest a
    later stage writes, keeps the commit its own stage ran at)."""
    from .state import stage_commits
    other = {st: (c or {}).get("commit") for st, c in stage_commits(b).items()
             if (c or {}).get("commit") != (commit or {}).get("commit")}
    if not other:
        return None
    return (f"notice: this run is at mlx2 commit {(commit or {}).get('commit')}; earlier stages ran at "
            + ", ".join(f"{st}@{c}" for st, c in other.items())
            + " (receipts and manifests name each stage's own commit)")


def _execute(b: Build, stages: list, a, commit: dict, redo: list) -> int:
    if (b.plan.get("stream") or {}).get("mode") == "rolling" and any(s != "map" for s in stages):
        raise SystemExit("rolling capture supports the map stage only; a carve needs a resident model")
    extract = b.rec.get("extract") or {}
    host_plan = (extract.get("against") or {}).get("plan") or {}
    if "gap" in stages and not extract.get("skip_gap_screen") and (host_plan.get("stream") or {}).get("mode") == "rolling":
        raise SystemExit("the gap screen needs a resident host; this build planned rolling capture for it")
    # Every child's footprint watchdog uses the plan's free floor (--min-free-gb, default 15).
    if b.plan.get("min_free_gb") is not None:
        os.environ["AMOEBA_MIN_FREE_GB"] = f"{float(b.plan['min_free_gb']):g}"
    progress = Progress(b.out / "events.jsonl", quiet=getattr(a, "quiet", False))
    note = commit_notice(b, commit)
    if note:
        print(note, file=sys.stderr)
    srv = None
    if getattr(a, "metrics_port", 0):
        try:
            srv = MetricsServer(progress, a.metrics_port)
            print(f"metrics: http://127.0.0.1:{srv.port}/metrics", file=sys.stderr)
        except OSError as e:
            print(f"metrics exporter not started ({e}); continuing without it", file=sys.stderr)
    ctx = ctx_for(b, a, commit, progress)
    from . import runner as _runner
    _runner.DEADLINE["t"] = ctx.stop_at
    check_swap_args(a)
    guard = _runner.arm_swap_guard(*swap_settings(b, a))
    if getattr(a, "ignore_gpu_queue", "") and "MLX2_AMOEBA_AUTO_SWAP_BASELINE" not in os.environ:
        # the bypass skips only the GPU-window guard: a run beside another job still gets the harness's memory
        # admission (20 GiB available, +1 GiB swap) before every child, from this run's swap baseline
        from .agent import memory_snapshot
        os.environ["MLX2_AMOEBA_AUTO_SWAP_BASELINE"] = str(memory_snapshot()["swap_gb"])
    import signal

    def _term(signum, frame):                  # kill / hangup: stop the children like Ctrl-C does
        raise KeyboardInterrupt(f"signal {signum}")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _term)
    progress.emit("build_start", checkpoint=b.checkpoint, commit=commit["commit"], profile=b.cfg["profile"],
                  out=str(b.out), stages=list(kind_stages(b)), device="cpu" if ctx.cpu else "gpu", swap_guard=guard)
    runners = kind_runners(b)
    for st in kind_stages(b):                      # stages already done show as done
        r = b.receipt(st)
        if r:
            progress.state["stages"][st] = r.get("status", "pending")
    rc = 0
    try:
        redoing = False
        for st in stages:
            if st in redo:
                redoing = True
            if redoing:
                b.forget(st)
                ctx.redo.add(st)
            elif b.read_a_failure_since_retried(st):
                # it read a failed attempt's result (the install report) and that stage has been retried since, in
                # this run or an earlier one (the receipt records it): make it again
                b.forget(st)
            if b.is_done(st):
                progress.emit("stage_skip", stage=st)
                continue
            gpu = None
            reason = getattr(a, "ignore_gpu_queue", "")
            if reason:
                from .runner import gpu_window
                gpu = {**gpu_window(), "ignore_gpu_queue": reason}
            if st in MODEL_STAGES:
                gpu = require_gpu_window(ctx.cpu, getattr(a, "gpu_window_owner", ""),
                                         ignore_queue=reason)
            r = b.begin(st, commit)
            r["gpu_window"] = gpu
            r["swap_guard"] = guard
            progress.emit("stage_start", stage=st)
            try:
                outs, summ = runners[st](b, ctx)
            except CleanStop as e:
                b.end(r, "stopped", error=str(e))
                progress.emit("stage_end", stage=st, status="pending", wall_s=None)
                progress.newline()
                print(str(e), file=sys.stderr)
                return 75
            except KeyboardInterrupt:
                b.end(r, "interrupted", error="KeyboardInterrupt")
                progress.emit("stage_end", stage=st, status="failed", wall_s=None, error="interrupted")
                progress.newline()
                return 130
            except StageFailedContinue as e:       # truthful status; the stages after it still run
                r2 = b.end({**r, "continue_past": True}, "failed", outputs=e.outputs, summary=e.summary,
                           error=e.error)
                progress.emit("stage_end", stage=st, status="failed", wall_s=r2["wall_s"], error=e.error[:400])
                progress.newline()
                print(f"stage {st} failed: {e.error}; continuing (a resume retries it)", file=sys.stderr)
                continue
            except (ChildFailed, SystemExit, Exception) as e:
                import traceback
                err = f"{type(e).__name__}: {e}" + ("" if isinstance(e, SystemExit) else "\n" + traceback.format_exc())
                r2 = b.end(r, "failed", error=err)
                progress.emit("stage_end", stage=st, status="failed", wall_s=r2["wall_s"], error=str(e)[:400])
                progress.newline()
                print(f"stage {st} failed: {e}", file=sys.stderr)
                return 1
            r2 = b.end(r, "done", outputs=outs, summary=summ)
            if st in ("core", "map") and b.rec.get("kind", "build") == "build":
                try:                               # the fingerprint cache that `extract` reuses
                    from .extract import publish
                    publish(b, st, replace=st in ctx.redo)
                except Exception as e:             # noqa: BLE001  (a cache copy never fails a stage)
                    print(f"cache publish of {st} skipped: {e}", file=sys.stderr)
            progress.emit("stage_end", stage=st, status="done", wall_s=r2["wall_s"], summary=summ)
            progress.newline()
            print(f"stage {st} done in {r2['wall_s']} s: {json.dumps(summ, default=str)}", file=sys.stderr)
        b.rec["last_run"] = {"finished": now(), "stages": stages, "commit": commit}
        b.save()
    finally:
        if srv:
            srv.close()
    return rc


def check_inputs(b) -> None:
    """Every input file still has the sha256 the build recorded, and the --against host its fingerprint."""
    inp = b.rec["inputs"]
    files = [inp["probe_corpus"], inp["probe_manifest"], inp["taxonomy"]] + list(inp["core_corpora"])
    eff = inp.get("probe_corpus_effective")
    if eff:
        files += [eff] + list(eff.get("exclude") or []) + list(eff.get("extra") or [])
    cf = ((b.rec.get("stage_options") or {}).get("core") or {}).get("core_from")
    if cf:
        files.append(cf)
    for f in files:
        if paths.sha256_file(f["path"]) != f["sha256"]:
            raise SystemExit(f"input {f['path']} changed since the build started (sha256); start a fresh build")
    ag = (b.rec.get("extract") or {}).get("against")
    if ag and inspect_checkpoint(ag["path"])["fingerprint"] != ag["fingerprint"]:
        raise SystemExit(f"--against {ag['path']} changed since the extraction started")
    if b.rec.get("kind") == "install":
        from .install import check_install_inputs
        check_install_inputs(b)


def cmd_rolling_parity(a) -> int:
    """Resident child_capture and streamed child_rolling on the same seeded probe texts; every resp_L, mu and the
    NLL compared (max-abs). Writes <out>/parity.json naming the checkpoint and the commit."""
    import numpy as np
    from .progress import Progress
    from .runner import run_child
    from .stages import read_jsonl, write_jsonl
    from . import runner as _runner
    from .plan import _host_unit_families
    check_swap_args(a)
    commit = check_commit(a.allow_dirty)
    require_gpu_window(a.cpu, a.gpu_window_owner)
    info = inspect_checkpoint(a.model)
    cfg = K.resolve("quick", {}, {}, "")
    plan = make_plan(info, cfg, moe_units=a.moe_units, cpu=a.cpu, stream="rolling", window=a.window, texts=a.texts)
    st = plan["stream"]
    fam = "host" if a.rolling_resident else (st.get("family") or "host")   # resident weights: the host family
    if fam != "host" and fam not in _host_unit_families():
        raise SystemExit(f"rolling parity: the {fam} sweep family runs its own parity (sweep capture --parity)")
    if a.streamed_only:
        # a model that does not fit: no resident side, only the streamed capture and its own full-depth gate (the
        # layer-major schedule against the family's ordinary text-major forward, both streamed)
        if a.rolling_resident:
            raise SystemExit("--streamed-only and --rolling-resident exclude each other")
        bs = st.get("bs") or 8
    else:
        reference_plan = make_plan(info, cfg, moe_units=a.moe_units, cpu=a.cpu, stream="resident", texts=a.texts)
        bs = min(8, reference_plan["batches"]["capture"]["bs"])
        if not bs or info.get("no_adapter"):
            raise SystemExit(f"rolling parity requires a resident reference: {reference_plan['refusals']} (a model "
                             "that does not fit: --streamed-only)")
    if info["kind"] == "moe" and not a.rolling_resident and fam == "host":
        raise SystemExit("rolling parity on an MoE model: no sweep family streams this model's whole layers (mlx2's "
                         "loaders have no whole-layer expert streaming); pass --rolling-resident to check the "
                         "layer-major schedule with resident weights")
    if a.rolling_resident:
        st, fam = {"fits": True, "window": 1, "stream_mode": "dense_layers", "ceiling_gb": 0.0}, "host"
    if not st.get("fits"):
        raise SystemExit(f"rolling parity: {st.get('refusal')}")
    guard = _runner.arm_swap_guard(a.swap_growth_gb, a.swap_waiver)
    out = Path(a.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_jsonl(a.probe_corpus) if r.get("origin") != "synthetic"]
    import hashlib
    rows = sorted(rows, key=lambda r: hashlib.sha256(f"parity:{r['id']}".encode()).hexdigest())[:a.texts]
    write_jsonl(out / "texts.jsonl", rows)
    prog = Progress(out / "events.jsonl", quiet=False)
    common = ["--model", info["path"], "--corpus", out / "texts.jsonl", "--seq-len", cfg["data"]["window_tokens"],
              "--bs", bs, "--moe-units", plan["moe_units"], "--memory-gb", plan["mlx_memory_gb"],
              "--watchdog-gb", plan["watchdog_gb"]] + (["--cpu"] if a.cpu else [])
    t0 = time.time()
    if not a.streamed_only:
        run_child([paths.python(), "-u", "-m", "mlx2.experimental.amoeba.child_capture", "--out", out / "resident"]
                  + common, stage="parity", progress=prog, log_path=out / "resident.log", what="resident capture")
    t1 = time.time()
    stream_args = (["--stream-mode", st["stream_mode"], "--ceiling-gb", st["ceiling_gb"]] if fam == "host" else [])
    run_child([paths.python(), "-u", "-m", "mlx2.experimental.amoeba.child_rolling", "--out", out / "rolling",
               "--family", fam, "--window", st.get("window") or 0, "--chunks", a.chunks, "--carry", a.carry,
               "--parity-layers", 0] + stream_args + (["--resident"] if a.rolling_resident else []) + common,
              stage="parity", progress=prog, log_path=out / "rolling.log", what=f"rolling capture ({fam} family)")
    t2 = time.time()
    if a.streamed_only:
        return _streamed_only_result(a, out, info, commit, plan, st, fam, guard, t2 - t1)
    ra, rb = out / "resident", out / "rolling"
    ma, mb = json.loads((ra / "meta.json").read_text()), json.loads((rb / "meta.json").read_text())
    if ma["fingerprint"] != info["fingerprint"] or any(ma[k] != mb[k] for k in ("fingerprint", "sequences", "layers", "widths")):
        raise SystemExit("rolling parity: capture identity or declared shape differs")
    items = json.loads((ra / "items.json").read_text())
    if not items or len(items) != ma["sequences"] or items != json.loads((rb / "items.json").read_text()):
        raise SystemExit("rolling parity: capture items differ or are empty")

    def pair(name, shape):
        x, y = np.load(ra / name), np.load(rb / name)
        if x.shape != shape or y.shape != shape or x.dtype != y.dtype:
            raise SystemExit(f"rolling parity: {name} expected shape {shape} and matching dtype, "
                             f"got {x.shape}/{x.dtype} and {y.shape}/{y.dtype}")
        return x, y

    def finite_max(x):
        value = float(x.max(initial=0))
        return value if np.isfinite(value) else None

    per = []
    mu_a, mu_b = pair("mu.npy", (ma["layers"], max(ma["widths"])))
    nll_a, nll_b = pair("nll_full.npy", (len(items),))
    for l in range(ma["layers"]):
        x, y = pair(f"resp_L{l:02d}.npy", (len(items), ma["widths"][l]))
        bitwise = np.array_equal(x.view(np.uint8), y.view(np.uint8))
        x, y = x.astype(np.float32), y.astype(np.float32)
        d = np.abs(np.subtract(x, y, where=x != y, out=np.zeros_like(x)))
        # element-wise, one bf16 ulp (2^-7) of the ACTIVATION behind each response (|act| <= resp + |mu|, so a
        # last-bit change of act is not an error on a small |act - mu|), plus an fp16 floor; a non-finite value on
        # either side that the other does not share is always over the tolerance
        act = np.maximum(np.abs(x), np.abs(y)) + np.abs(mu_a[l, :x.shape[1]])[None]
        tol = act * 2 ** -7 + 2 ** -14
        bad = (d > tol) | (np.isfinite(x) != np.isfinite(y)) | (~np.isfinite(x) & (x != y))
        per.append({"layer": l, "kind": ma["unit_kinds"][l], "max_abs": finite_max(d), "bitwise": bitwise,
                    "scale": finite_max(np.abs(x)),
                    "over_tol": int(bad.sum()) if x.size else 0, "elements": int(x.size),
                    "nonfinite": int((~np.isfinite(x)).sum() + (~np.isfinite(y)).sum()) if x.size else 0})
    with np.errstate(invalid="ignore"):
        mu_delta = np.abs(mu_a - mu_b)
        mu_rel = mu_delta / (np.abs(mu_a) + 1e-3)
        nll_delta = np.abs(nll_a - nll_b)
    res = {"schema": "mlx2.amoeba-rolling-parity.v1", "checkpoint": info["checkpoint"], "fingerprint": info["fingerprint"],
           "mlx2_commit": commit, "texts": len(rows), "window": st["window"], "chunks": a.chunks,
           "stream_mode": "resident (schedule check)" if a.rolling_resident else st["stream_mode"],
           "moe_units": plan["moe_units"],
           "resp_max_abs": max(x["max_abs"] for x in per) if all(x["max_abs"] is not None for x in per) else None,
           "per_layer": per, "mu_max_abs": finite_max(mu_delta), "nll_max_abs": finite_max(nll_delta),
           "schedule_parity": mb.get("parity"), "rolling": {k: v for k, v in mb["rolling"].items() if k != "per_layer"},
           "stream_counters": mb.get("stream_counters"),
           "seconds": {"resident": round(t1 - t0, 1), "rolling": round(t2 - t1, 1)},
           "peak_gb": {"resident": ma.get("mlx_peak_gb"), "rolling": mb.get("mlx_peak_gb")},
           "note": "resident = the ordinary text-major capture (child_capture); rolling = layer-major through the "
                   "adapter's weight streaming; fp16 resp rows, so a max-abs of one fp16 ulp is rounding"}
    # fp16 rows: one ulp of the row's own scale is rounding (batch composition differs between the two captures)
    res["over_tol_elements"] = sum(x["over_tol"] for x in per)
    res["bitwise"] = all(x["bitwise"] for x in per)
    res["mu_rel_max"] = finite_max(mu_rel)
    res["mu_over_tol"] = int((~np.isfinite(mu_delta) | (mu_delta > 1e-3 * np.abs(mu_a) + 1e-5)).sum())
    res["observed_reads"] = mb.get("weight_reads_observed")
    res["tolerance"] = ("every response within 2^-7 x (|resp| + |mu|) + 2^-14 (one bf16 ulp of the activation), no "
                        "unshared non-finite value; mu within 1e-3 relative + 1e-5; NLL within 1e-4; each layer read "
                        "once per chunk (measured)")
    res["batch_composition"] = (f"resident: {len(items)} retained texts in length batches of {bs}; "
                                f"rolling: whole batches allocated to at most {a.chunks} token-balanced chunks")
    reads_ok = a.rolling_resident or (mb.get("weight_reads_observed") or {}).get("once_per_chunk") is True
    res["exact"] = (res["over_tol_elements"] == 0 and res["mu_over_tol"] == 0 and res["nll_max_abs"] is not None
                    and res["nll_max_abs"] <= 1e-4 and mb["parity"]["ok"] and reads_ok)
    res.update(family=fam, stream_receipt=(mb.get("load") or {}).get("stream_receipt"), swap_guard=guard)
    (out / "parity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("checkpoint", "exact", "bitwise", "over_tol_elements", "resp_max_abs",
                                          "mu_rel_max", "nll_max_abs", "observed_reads", "seconds")}))
    return 0 if res["exact"] else 3


def _streamed_only_result(a, out, info, commit, plan, st, fam, guard, seconds) -> int:
    """--streamed-only: the streamed capture's own receipt (no resident side). Passes when its full-depth gate (the
    layer-major schedule against the family's ordinary text-major forward) passed and each layer was read once per
    chunk (measured); the responses are not compared with anything."""
    mb = json.loads((out / "rolling" / "meta.json").read_text())
    if mb.get("fingerprint") != info["fingerprint"]:
        raise SystemExit(f"rolling parity: the capture names {mb.get('fingerprint')}, the checkpoint is "
                         f"{info['fingerprint']}")
    reads = mb.get("weight_reads_observed") or {}
    res = {"schema": "mlx2.amoeba-rolling-parity.v1", "mode": "streamed-only", "checkpoint": info["checkpoint"],
           "fingerprint": info["fingerprint"], "mlx2_commit": commit, "family": fam, "texts": mb.get("sequences"),
           "window": st.get("window"), "chunks": a.chunks, "moe_units": plan["moe_units"],
           "schedule_parity": mb.get("parity"), "rolling": {k: v for k, v in mb["rolling"].items() if k != "per_layer"},
           "stream_counters": mb.get("stream_counters"), "observed_reads": reads or None,
           "stream_receipt": (mb.get("load") or {}).get("stream_receipt"), "swap_guard": guard,
           "seconds": {"rolling": round(seconds, 1)}, "peak_gb": {"rolling": mb.get("mlx_peak_gb")},
           "note": "no resident reference (the model does not fit): the gate is the streamed schedule against the "
                   "family's own streamed text-major forward over every layer; compare streaming with a resident load "
                   "on a sibling that fits (rolling-parity without --streamed-only)"}
    # both sides run the same streamed module tree, so anything short of bitwise equality is a finding, not rounding
    texts = (mb.get("parity") or {}).get("texts") or []
    res["bitwise"] = bool(texts) and all(t.get("bitwise_equal") for t in texts)
    res["exact"] = bool((mb.get("parity") or {}).get("ok")) and res["bitwise"] and reads.get("once_per_chunk") is True
    (out / "parity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("checkpoint", "mode", "exact", "bitwise", "observed_reads", "seconds", "peak_gb")}))
    return 0 if res["exact"] else 3


def cmd_resume(a) -> int:
    b = Build(a.out).require()
    info = inspect_checkpoint(b.rec["model"])
    if info["fingerprint"] != b.info["fingerprint"]:
        raise SystemExit(f"checkpoint changed: {b.info['fingerprint'][:12]} -> {info['fingerprint'][:12]}; start a "
                         "fresh build")
    commit = check_commit(a.allow_dirty)
    check_inputs(b)
    stages = [s for s in (a.stages.split(",") if a.stages else b.rec["stages"]) if s]
    unknown = set(stages) - set(kind_stages(b))
    if unknown or not stages:
        raise SystemExit(f"invalid resume stages {a.stages!r}; choose from {', '.join(kind_stages(b))}")
    for r in a.redo:
        if r not in kind_stages(b):
            raise SystemExit(f"unknown stage {r}")
        if r not in stages:
            stages.append(r)
    stages = [s for s in kind_stages(b) if s in stages]
    return execute(b, stages, a, commit, redo=a.redo)


def cmd_stage(a) -> int:
    b = Build(a.out).require()
    if a.cmd not in kind_stages(b):
        raise SystemExit(f"{a.cmd} is not a stage of this {b.rec.get('kind', 'build')} ({', '.join(kind_stages(b))})")
    info = inspect_checkpoint(b.rec["model"])
    if info["fingerprint"] != b.info["fingerprint"]:
        raise SystemExit("checkpoint changed since the build started; start a fresh build")
    commit = check_commit(a.allow_dirty)
    check_inputs(b)
    return execute(b, [a.cmd], a, commit, redo=[a.cmd] if a.redo else [])


def status(b: Build) -> dict:
    st = {}
    for s in kind_stages(b):
        r = b.receipt(s) or {}
        st[s] = {k: r.get(k) for k in ("status", "started", "finished", "wall_s", "attempts", "summary", "error")}
    boxes = []
    vp = b.out / ("box/carves/verdicts.jsonl" if b.rec.get("kind") == "extract" else "boxes/carves/verdicts.jsonl")
    if vp.exists():
        seen = {}
        for x in vp.read_text().splitlines():
            if x.strip():
                v = json.loads(x)
                seen[v["box"]] = v
        boxes = [{k: v.get(k) for k in ("box", "node", "kind", "units", "export_val", "export_held",
                                         "random_same_count_held", "verdict", "error")} for v in seen.values()]
    last = None
    ev = b.out / "events.jsonl"
    if ev.exists():
        lines = ev.read_text().splitlines()
        last = json.loads(lines[-1]) if lines else None
    return {"out": str(b.out), "checkpoint": b.checkpoint, "profile": b.cfg["profile"],
            "mlx2_commit": b.rec.get("mlx2_commit"), "stages": st, "boxes": boxes, "last_event": last}


def cmd_status(a) -> int:
    b = Build(a.out).require()
    s = status(b)
    if a.json:
        print(json.dumps(s, indent=1, default=str))
        return 0
    print(f"{s['checkpoint']}  profile {s['profile']}  commit {s['mlx2_commit'].get('commit')}  {s['out']}")
    for k, v in s["stages"].items():
        summ = json.dumps(v["summary"], default=str)[:110] if v.get("summary") else (v.get("error") or "")[:110]
        print(f"  {k:10s} {str(v['status'] or 'pending'):8s} {str(v.get('wall_s') or ''):>8s}  {summ}")
    if s["boxes"]:
        f = lambda x: "-" if x is None else f"{x:.3f}"   # noqa: E731
        print("  boxes:")
        for x in s["boxes"]:
            print(f"    {x['box']:28s} {str(x['kind']):9s} {str(x.get('units') or '-'):>8s} val {f(x['export_val'])} "
                  f"held {f(x['export_held'])} rand {f(x['random_same_count_held'])}  {x['verdict']}")
    if s["last_event"]:
        print(f"  last event: {json.dumps(s['last_event'], default=str)[:160]}")
    return 0


def main(argv=None) -> int:
    a = parse(list(sys.argv[1:] if argv is None else argv))
    if a.cmd == "auto":
        from .agent import run_auto
        if not a.self_mode and not (a.goal and a.host):
            raise SystemExit("auto needs --goal and --host, or --self")
        return run_auto(a)
    if a.cmd == "auto-status":
        from .agent import run_status
        return run_status(a)
    if a.cmd == "auto-eval":
        from .agent import run_eval
        return run_eval(a)
    if a.cmd == "inspect":
        info = inspect_checkpoint(a.model)
        print(json.dumps(info, indent=1) if a.json else
              f"{info['checkpoint']}  {info['kind']}  adapter {info['adapter']}  layers {info['layers']}  "
              f"hidden {info['hidden']}  units {info['units']}  weights {info['weight_bytes'] / 2**30:.2f} GB")
        return 0
    if a.cmd == "plan":
        info = inspect_checkpoint(a.model)
        cfg = K.resolve(a.profile, K.parse_overrides(a.compute), {}, "")
        plan = make_plan(info, cfg, cap_gb=a.cap_gb, cap_reason=a.cap_reason, moe_units=a.moe_units, cpu=a.cpu,
                         stream=a.stream, window=a.window, texts=map_texts(str(paths.DEFAULT_PROBE_CORPUS), cfg))
        print(json.dumps(plan, indent=1) if a.json else format_plan(info, plan))
        return 0 if plan["fits"] else 2
    if a.cmd in ("build", "extract"):
        return cmd_build(a)
    if a.cmd == "install":
        from .install import cmd_install
        return cmd_install(a, execute, check_commit)
    if a.cmd == "resume":
        return cmd_resume(a)
    if a.cmd == "rolling-parity":
        return cmd_rolling_parity(a)
    if a.cmd == "status":
        return cmd_status(a)
    return cmd_stage(a)


if __name__ == "__main__":
    raise SystemExit(main())
