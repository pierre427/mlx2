"""Targeted extraction (mapping-workflow s11.1): one taxonomy topic's box from a donor, as a portable bundle.

    mlx2 amoeba extract --model DONOR --topic sci.biology --out DIR --against HOST

Stages (receipts, resume and progress as in ``build``; kind ``extract``):
  gap        the donor-gap screen: the topic's held probes (every held text at or below the topic, plus a fixed
             per-topic sample of the rest for the offset), per-byte NLL bare and with a neutral prefix, on the
             donor and on --against, centred on each condition's whole-corpus gap. Extract only if the donor
             leads in both conditions at 95% (CI lower bound above the margin); otherwise the run stops,
             unless --force-extract REASON; --skip-gap-screen REASON runs no screen (both recorded)
             Both models score the same characters of each text; the offset leaves the topic out and its
             bootstrap enters the CI.                             scripts/amoeba_topics_donor_gap.py score/compare
  core       the donor's English core, reused from the fingerprint cache when one was made with the same settings,
             else carved (stages.run_core)
  map        the donor's map (non-core units), reused from the cache likewise
  box        one box: the topic's probes + continuations (1 greedy + 3 T=0.7), the core fixed, the export cut on
             val at +0.03, held checked once at +0.05 pooled; the mask is seeded with the map clusters tagged
             with the topic (carve --init-from SEED --init-raw); a miss lists the topic for probe extension
  box_gate   its detector against English and off-topic probes (fit only), held false-open / miss
  box_bundle box mask + packed box weights, gate, the donor's map, taxonomy and prompt set, the donor fingerprint,
             the donor-side quality, the gap-screen result (schema mlx2.amoeba-box-bundle.v1)

Attention heads as carvable units (--attention-units heads) need head sites in the carve host, which are not
built: the option refuses. Experimental and unqualified.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from . import paths
from .runner import run_child
from .stages import (_py, _reset_partial, fit_tags, generate, read_jsonl, run_core, run_map, unit_clusters,
                     write_jsonl)
from .state import digest, file_outputs, write_json

EXTRACT_STAGES = ("gap", "core", "map", "box", "box_gate", "box_bundle")
CACHE = paths.ARTIFACTS / "cache"


# ------------------------------------------------------------------ fingerprint cache (core, map)

def _cache_key(b, what: str) -> str:
    """Core: its settings, inputs and device. Map: its settings and the CONTENT of the core it clusters around
    (sha256 of this build's core.npz), so a map is never paired with another core."""
    c, d = b.cfg["compute"], b.cfg["data"]
    device = "cpu" if (b.rec.get("options") or {}).get("cpu") else "gpu"
    if what == "core":
        keys = {k: v for k, v in c.items() if k.startswith("core_")}
        extra = {"corpora": [x["sha256"] for x in b.rec["inputs"]["core_corpora"]], "moe_units": b.plan["moe_units"],
                 "probe_corpus": b.rec["inputs"]["probe_corpus"]["sha256"], "device": device}
        opts = b.rec.get("options") or {}
        extra.update(gen_backend=opts.get("gen_backend"), gen_route=opts.get("gen_route", ""),
                     grad_gate=b.cfg.get("grad_gate"))
    else:
        keys = {k: v for k, v in c.items() if k.startswith("map_") or k in ("ve_per_layer", "cluster_k")}
        extra = {"probe_corpus": b.rec["inputs"]["probe_corpus"]["sha256"], "moe_units": b.plan["moe_units"],
                 "core_sha256": paths.sha256_file(b.out / "core/core.npz"), "device": device,
                 "taxonomy": (b.rec["inputs"].get("taxonomy") or {}).get("sha256")}
    return digest({"data": d, "compute": keys, **extra})


def cache_dir(b, what: str) -> Path:
    return CACHE / b.info["fingerprint"][:16] / f"{what}-{_cache_key(b, what)}"


def _relocate(carve_json: Path, old: str, new: str) -> None:
    """Point a copied carve receipt's corpus / held / val paths at the copy (the prompt set reads them)."""
    r = json.loads(carve_json.read_text())
    a = r.get("args") or {}
    for k in ("corpus", "held_file", "val_file", "fixed", "out"):
        if isinstance(a.get(k), str) and old in a[k]:
            a[k] = a[k].replace(old, new)
    carve_json.write_text(json.dumps(r, indent=1))


def publish(b, what: str, replace: bool = False) -> Path | None:
    """Copy a finished core or map stage into the fingerprint cache (compact parts only for the map). A run from a
    dirty tree never publishes. ``replace`` (a --redo) moves an existing entry aside and publishes this one."""
    commit = (b.receipt(what) or {}).get("mlx2_commit") or b.rec.get("mlx2_commit") or {}
    if commit.get("dirty") is not False:
        return None
    src = b.out / what
    dst = cache_dir(b, what)
    if dst.exists():
        if not replace:
            return dst
        import time
        dst.rename(dst.with_name(f"{dst.name}.replaced-{int(time.time())}"))
    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    if what == "core":
        for f in ("core.npz", "corecheck.json", "english.jsonl", "corpus.jsonl"):
            shutil.copy2(src / f, tmp / f)
        shutil.copytree(src / "carve", tmp / "carve", ignore=shutil.ignore_patterns("*.ckpt*", "rank_ckpt*"))
        shutil.copytree(src / "gen", tmp / "gen")
        _relocate(tmp / "carve/carve.json", str(src), str(dst))
    else:
        for f in ("map.json", "probes.jsonl", "core_units.npz"):
            if (src / f).exists():
                shutil.copy2(src / f, tmp / f)
        shutil.copytree(src / "clusters", tmp / "clusters")
        shutil.copytree(src / "labels", tmp / "labels")
        (tmp / "capture").mkdir()
        for f in ("meta.json", "items.json"):
            shutil.copy2(src / "capture" / f, tmp / "capture" / f)
    write_json(tmp / "cache.json", {"checkpoint": b.checkpoint, "fingerprint": b.info["fingerprint"], "what": what,
                                    "key": _cache_key(b, what), "from_build": str(b.out),
                                    "mlx2_commit": commit,
                                    "device": "cpu" if (b.rec.get("options") or {}).get("cpu") else "gpu",
                                    "receipt": b.receipt(what) if b.receipt(what) else None})
    tmp.rename(dst)
    return dst


def _reuse(b, what: str) -> Path | None:
    if what == "map" and not (b.out / "core/core.npz").exists():
        return None
    d = cache_dir(b, what)
    if not (d / "cache.json").exists():
        return None
    dst = b.out / what
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(d, dst)
    if what == "core":
        _relocate(dst / "carve/carve.json", str(d), str(dst))
    return d


# ------------------------------------------------------------------ stages

def run_gap(b, ctx):
    d = b.stage_dir("gap")
    x = b.rec["extract"]
    if x.get("skip_gap_screen"):
        res = {"screen": "skipped", "reason": x["skip_gap_screen"], "donor_leads": None,
               "force_extract": x.get("force_extract")}
        write_json(d / "gap.json", res)
        return file_outputs({"gap": d / "gap.json"}), res
    probes = b.rec["inputs"]["probe_corpus"]["path"]
    per = 24 if b.cfg["profile"] == "full" else 4
    chars = 4 * b.cfg["data"]["window_tokens"]          # the same characters for both tokenizers (~128 tokens)
    for role, model, plan in (("donor", b.info["path"], b.plan), ("host", x["against"]["path"], x["against"]["plan"])):
        od = d / role
        _reset_partial(od, "score.json")
        if not od.exists():
            run_child(_py("amoeba_topics_donor_gap.py") +
                      ["score", "--backend", "mlx2", "--model", model, "--corpus", probes, "--out", od,
                       "--seq-chars", chars, "--per-topic", per, "--include-topic", x["topic"],
                       "--need-gb", plan["need_gb"], "--watchdog-gb", plan["watchdog_gb"],
                       "--memory-gb", plan["mlx_memory_gb"],
                       # the admission the carves use (strict when it passes, else the coordinator's current waiver)
                       "--mem-policy", "auto:waived:60:80:15"] + (["--cpu"] if ctx.cpu else []),
                      stage="gap", progress=ctx.progress, log_path=b.log_path("gap"), what=f"gap score ({role})")
    cmp_ = d / "compare"
    _reset_partial(cmp_, "donor_gap.json")
    if not cmp_.exists():
        run_child(_py("amoeba_topics_donor_gap.py") + ["compare", "--host", d / "host", "--donor", d / "donor",
                                                        "--out", cmp_, "--offset-exclude-topic", x["topic"]],
                  stage="gap", progress=ctx.progress, log_path=b.log_path("gap"), what="gap compare")
    g = json.loads((cmp_ / "donor_gap.json").read_text())
    t = x["topic"]
    row = (g.get("topics") or {}).get(t)
    margin = g.get("margin_nats_per_byte", 0.005)
    leads = bool(row) and all(row.get(c, {}).get("centred_ci95", [-1])[0] > margin for c in ("bare", "prefix"))
    res = {"screen": "run", "topic": t, "against": x["against"]["checkpoint"], "row": row, "margin": margin,
           "scored_span": f"the first {chars} characters of each held text (cut at a space), for both models",
           "same_tokenizer": x["against"].get("same_tokenizer"), "force_extract": x.get("force_extract"),
           "offset": "whole-corpus gap without the topic; its bootstrap is in the centred CI",
           "donor_leads": leads, "offset_nats_per_byte": g.get("overall_gap_nats_per_byte"),
           "in_donor_better_centred_bh": t in (g.get("donor_better_centred") or []),
           "rule": "centred (host - donor) nats/byte CI95 lower bound > margin in both bare and prefix",
           "untested": (g.get("untested_few_groups") or {}).get(t)}
    write_json(d / "gap.json", res)
    if not leads and not x.get("force_extract"):
        raise SystemExit(f"gap screen: the donor does not lead on {t} in both conditions at 95% ({row or 'untested'}); "
                         "not extracting (pass --force-extract REASON to carve it anyway)")
    return file_outputs({"gap": d / "gap.json", "compare": cmp_ / "donor_gap.json"}), \
        {k: res[k] for k in ("topic", "donor_leads", "margin")}


def run_xmap(b, ctx):
    hit = None if "map" in ctx.redo else _reuse(b, "map")
    if hit:
        m = json.loads((b.out / "map/map.json").read_text())
        outs = file_outputs({"map": b.out / "map/map.json", "clusters": b.out / "map/clusters/clusters.npz",
                             "labels": b.out / "map/labels/labels.json"})
        src = json.loads((b.out / "map/cache.json").read_text())
        return outs, {"reused_from_cache": str(hit), "map_commit": src.get("mlx2_commit"),
                      "clusters": m.get("clusters"), "labelled": m.get("labelled")}
    outs, summ = run_map(b, ctx)
    summ["cached_to"] = str(publish(b, "map", replace="map" in ctx.redo))
    return outs, summ


def run_xcore(b, ctx):
    hit = None if "core" in ctx.redo else _reuse(b, "core")
    if hit:
        cc = json.loads((b.out / "core/corecheck.json").read_text())
        outs = file_outputs({"core": b.out / "core/core.npz", "carve": b.out / "core/carve/carve.json",
                             "corecheck": b.out / "core/corecheck.json", "english": b.out / "core/english.jsonl"})
        src = json.loads((b.out / "core/cache.json").read_text())
        return outs, {"reused_from_cache": str(hit), "core_commit": src.get("mlx2_commit"), "core_device": src.get("device"),
                      "core_from_build": src.get("from_build"), "english_only": cc.get("english_only"),
                      "units": int((np.load(b.out / "core/core.npz")["core"] > 0).sum())}
    outs, summ = run_core(b, ctx)
    summ["cached_to"] = str(publish(b, "core", replace="core" in ctx.redo))
    return outs, summ


def _topic_seed(b, topic: str, shape) -> tuple[np.ndarray | None, dict]:
    """Z seed: units of map clusters whose fit-split tags reach the topic (it or a node below it) at +3, all others
    at +1."""
    labels = json.loads((b.out / "map/labels/labels.json").read_text())
    hit = []
    for c in labels["clusters"]:
        if any(t == topic or t.startswith(topic + ".") for t in fit_tags(c)):
            hit.append(int(c["id"].split("C")[-1]))
    if not hit:
        return None, {"seed": "none: no map cluster carries the topic", "clusters": []}
    uc = unit_clusters(b, shape)
    Z = np.ones(shape, np.float32)
    on = np.isin(uc, hit)
    Z[on] = 3.0
    return Z, {"seed": "map clusters whose fit-split tags reach the topic at +3, every other unit at +1 (--init-raw)",
               "clusters": hit, "units": int(on.sum())}


def run_box(b, ctx):
    d = b.stage_dir("box")
    comp, data = b.cfg["compute"], b.cfg["data"]
    topic = b.rec["extract"]["topic"]
    bs = d / "boxset"
    _reset_partial(bs, "boxes.json")
    if not bs.exists():
        run_child(_py("amoeba_fn_boxset_build.py") +
                  ["--corpus", b.rec["inputs"]["probe_corpus"]["path"], "--out", bs, "--topic", topic,
                   "--min-fit", comp["box_min_fit"], "--carve-prompts", comp["box_carve_prompts"],
                   "--val-prompts", comp["box_val_prompts"], "--held-prompts", comp["box_held_prompts"],
                   "--model", b.info["path"], "--seq-len", data["window_tokens"],
                   "--prompt-min", data["gen_prompt_tokens"] + 16, "--min-carve", comp["box_min_carve"],
                   "--min-val", comp["box_min_val"], "--min-held", comp["box_min_held"],
                   "--manifest", b.rec["inputs"]["probe_manifest"]["path"]],
                  stage="box", progress=ctx.progress, log_path=b.log_path("box"), what="box set (one topic)")
    spec = json.loads((bs / "boxes.json").read_text())
    if not spec["boxes"]:
        ext = json.loads((bs / "extend.json").read_text())
        write_json(d / "next-round.json", {"extend_probes": ext, "misses": []})
        raise SystemExit(f"{topic}: too few probes for a box ({ext}); extend its probes (licensed text) and rerun")
    box = spec["boxes"][0]
    name = box["box"]
    if any(box["node"].startswith(p) or box["node"] + "." == p for p in data["code_topic_prefixes"]):
        box["domain"] = "code"
        for f in (bs / name).glob("*.jsonl"):
            write_jsonl(f, [{**r, "domain": "code"} if r.get("domain") == "box" else r for r in read_jsonl(f)])
    else:
        box["domain"] = "box"
    prompts = sorted((bs / name).glob("prompts_*.jsonl"))
    gen = generate(b, ctx, "box", prompts, d / "gen")
    owner = {}
    for role in ("carve", "val", "held"):
        for r in read_jsonl(bs / name / f"prompts_{role}.jsonl"):
            owner[r["doc_id"]] = role
    g = {}
    for r in read_jsonl(d / "gen/gen.jsonl"):
        g.setdefault(owner[r["doc_id"]], []).append({**r, "domain": f"gen:{box['domain']}:{r['mode']}"})
    for outf, base, role in (("corpus.jsonl", "docs.jsonl", "carve"), ("val_full.jsonl", "val.jsonl", "val"),
                             ("held_full.jsonl", "held.jsonl", "held")):
        if not (bs / name / outf).exists():
            write_jsonl(bs / name / outf, read_jsonl(bs / name / base) + g.get(role, []))
    rep = json.loads((b.out / "core/carve/carve.json").read_text())
    widths = rep["shape"]["widths"]
    shape = (len(widths), max(widths))
    Z, seed = _topic_seed(b, topic, shape)
    extra = []
    if Z is not None:
        np.savez_compressed(d / "seed.npz", Z=Z)
        extra = ["--init-from", str(d / "seed.npz"), "--init-raw"]
    spec_child = {"model": b.info["path"], "out": str(d / "carves"), "boxset": str(bs), "boxes": [box],
                  "core": str(b.out / "core/core.npz"), "compute": comp, "data": data, "plan": b.plan,
                  "verdict": b.cfg["verdict"], "grad_gate": b.cfg["grad_gate"], "cpu": ctx.cpu, "stop_at": ctx.stop_at,
                  "extra_carve_args": extra}
    sp = d / "batch-spec.json"
    sp.write_text(json.dumps(spec_child, indent=1))
    run_child(_py("mlx2.experimental.amoeba.child_boxbatch") + ["--spec", sp], stage="box", progress=ctx.progress,
              log_path=b.log_path("box"), what="box carve")
    ver = {v["box"]: v for v in read_jsonl(d / "carves/verdicts.jsonl")}[name]
    if ver["verdict"] == "failed":
        if ver.get("failure") == "gate-refused":
            raise SystemExit(f"{name}: the s7 gradient check refused this donor's box ({ver.get('gate_cells')}); a retry "
                             "would refuse again: there is no box to extract on this checkpoint and commit")
        raise SystemExit(f"box carve failed: {name}: {ver.get('error')}; resume to retry the failed carve")
    write_json(d / "next-round.json", {
        "extend_probes": json.loads((bs / "extend.json").read_text()) if (bs / "extend.json").exists() else [],
        "misses": [{"node": topic, "box": name, "export_held": ver.get("export_held"),
                    "reasons": ver.get("miss_reasons")}] if ver["verdict"] == "miss" else [],
        "rule": "a miss: extend the topic's probes with more varied, deeper licensed text and re-carve; never a wider "
                "target or a held re-cut"})
    write_json(d / "seed.json", seed)
    summ = {"box": name, "verdict": ver["verdict"], "units": ver.get("units"), "held_d_ce": ver.get("export_held"),
            "val_d_ce": ver.get("export_val"), "random_same_count_held": ver.get("random_same_count_held"),
            "seed_clusters": len(seed["clusters"]), "gen_rows": gen.get("rows"), "gen_route": gen.get("route")}
    return file_outputs({"verdict": d / "carves/verdicts.jsonl", "boxset": bs / "boxes.json",
                         "next_round": d / "next-round.json"}), summ


def _box_name(b) -> str:
    return json.loads((b.out / "box/boxset/boxes.json").read_text())["boxes"][0]["box"]


def _verdict(b) -> dict:
    name = _box_name(b)
    return {v["box"]: v for v in read_jsonl(b.out / "box/carves/verdicts.jsonl")}[name]


def run_box_gate(b, ctx):
    d = b.stage_dir("box_gate")
    comp, data = b.cfg["compute"], b.cfg["data"]
    name = _box_name(b)
    if _verdict(b)["verdict"] not in ("pass", "miss"):
        write_json(d / "gates.json", {"schema": "mlx2.amoeba-gates.v1", "boxes": {}, "note": f"box {name} is "
                                      f"{_verdict(b)['verdict']}: no gate"})
        np.savez_compressed(d / "gates.npz")
        return file_outputs({"gates_json": d / "gates.json"}), {"status": f"skipped ({_verdict(b)['verdict']})"}
    if not (d / "gates.json").exists():
        run_child(_py("mlx2.experimental.amoeba.child_probe") +
                  ["gates", "--model", b.info["path"], "--boxset", b.out / "box/boxset", "--boxes", name,
                   "--bs", b.plan["batches"]["gates"]["bs"],
                   "--english", b.out / "core/english.jsonl", "--out", d, "--window-tokens", data["window_tokens"],
                   "--tokens", comp["gate_tokens_per_window"], "--windows", comp["gate_windows_per_box"],
                   "--negatives", comp["gate_negatives"], "--moe-units", b.plan["moe_units"],
                   "--code-prefixes", ",".join(data["code_topic_prefixes"]),
                   "--negative-probes", b.rec["inputs"]["probe_corpus"]["path"],
                   "--exclude-topic", b.rec["extract"]["topic"],
                   "--watchdog-gb", b.plan["watchdog_gb"], "--memory-gb", b.plan["mlx_memory_gb"]] +
                  (["--cpu"] if ctx.cpu else []),
                  stage="box_gate", progress=ctx.progress, log_path=b.log_path("box_gate"), what="gate detector")
    g = json.loads((d / "gates.json").read_text())["boxes"][name]
    return file_outputs({"gates": d / "gates.npz", "gates_json": d / "gates.json"}), \
        {k: g.get(k) for k in ("status", "layer", "threshold", "held_false_open_rate", "held_false_open_english",
                               "held_false_open_off_topic", "held_miss_rate")}


def run_box_bundle(b, ctx):
    d = b.out / "box_bundle"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    name = _box_name(b)
    ver = _verdict(b)
    if ver["verdict"] not in ("pass", "miss"):
        man = {"schema": "mlx2.amoeba-box-bundle.v1", "topic": b.rec["extract"]["topic"], "box": name,
               "verdict": ver["verdict"], "donor": {"checkpoint": b.checkpoint, "fingerprint": b.info["fingerprint"]},
               "mlx2_commit": ctx.commit, "extract": b.rec["extract"], "donor_quality": ver,
               "gap_screen": json.loads((b.out / "gap/gap.json").read_text()),
               "note": ("the carve kept no units beyond the core: no box to carry"
                        if ver["verdict"] == "covered" else
                        f"the box carve failed ({ver.get('failure', 'crash')}): no box to carry"),
               "gate_cells": ver.get("gate_cells")}
        write_json(d / "manifest.json", man)
        return file_outputs({"manifest": d / "manifest.json"}), {"box": name, "verdict": ver["verdict"], "files": 1}
    mask = b.out / f"box/carves/box-{name}/box_mask.npz"
    shutil.copy2(mask, d / "box_mask.npz")
    run_child(_py("mlx2.experimental.amoeba.child_pack") + ["--model", b.info["path"], "--mask", f"{mask}:box",
                                                            "--out", d / "weights", "--moe-units", b.plan["moe_units"],
                                                            "--memory-gb", b.plan["mlx_memory_gb"],
                                                            "--watchdog-gb", b.plan["watchdog_gb"]] +
              (["--cpu"] if ctx.cpu else []),
              stage="box_bundle", progress=ctx.progress, log_path=b.log_path("box_bundle"), what="pack box weights")
    inp = b.rec["inputs"]
    copies = {"core/core.npz": b.out / "core/core.npz", "core/corecheck.json": b.out / "core/corecheck.json",
              "gate/gates.npz": b.out / "box_gate/gates.npz", "gate/gates.json": b.out / "box_gate/gates.json",
              "map/map.json": b.out / "map/map.json", "map/clusters.npz": b.out / "map/clusters/clusters.npz",
              "map/clusters.json": b.out / "map/clusters/clusters.json", "map/labels.json": b.out / "map/labels/labels.json",
              "taxonomy/" + Path(inp["taxonomy"]["path"]).name: Path(inp["taxonomy"]["path"]),
              "taxonomy/probe-corpus-manifest.json": Path(inp["probe_manifest"]["path"]),
              "report/verdict.json": None, "report/gap.json": b.out / "gap/gap.json",
              "report/next-round.json": b.out / "box/next-round.json", "report/seed.json": b.out / "box/seed.json"}
    for dst, src in copies.items():
        (d / dst).parent.mkdir(parents=True, exist_ok=True)
        if src is None:
            write_json(d / dst, ver)
        else:
            shutil.copy2(src, d / dst)
    ap = paths.arrow_python()
    if ap is None:
        raise SystemExit("box bundle: no interpreter with pyarrow for the prompt-set Parquet (MLX2_AMOEBA_ARROW_PYTHON)")
    from .inspection import slug
    pname = f"prompt-set-{slug(b.checkpoint)}-{b.info['fingerprint'][:8]}-{name}"
    run_child([ap, "-u", paths.repo_root() / "scripts/amoeba_prompt_set.py", "--out", d / "prompt-set", "--name", pname,
               "--probe-corpus", inp["probe_corpus"]["path"], "--checkpoint", b.checkpoint,
               "--taxonomy-version", inp["taxonomy"]["version"],
               "--carve-run", b.out / "core/carve", "--stage", "core",
               "--carve-run", b.out / f"box/carves/box-{name}", "--stage", f"box:{name}",
               "--capture-run", b.out / "map/capture", "--stage", "map"],
              stage="box_bundle", progress=ctx.progress, log_path=b.log_path("box_bundle"), what="prompt set")
    ps = json.loads((d / f"prompt-set/{pname}.manifest.json").read_text())
    gate = json.loads((b.out / "box_gate/gates.json").read_text())["boxes"].get(name, {})
    files = {str(f.relative_to(d)): {"sha256": paths.sha256_file(f), "bytes": f.stat().st_size}
             for f in sorted(d.rglob("*")) if f.is_file()}
    man = {"schema": "mlx2.amoeba-box-bundle.v1", "topic": b.rec["extract"]["topic"], "box": name,
           "donor": {"checkpoint": b.checkpoint, "fingerprint": b.info["fingerprint"], "artifact": b.info["artifact"],
                     "adapter": b.info["adapter"], "kind": b.info["kind"], "moe_units": b.plan["moe_units"]},
           "mlx2_commit": ctx.commit, "profile": b.cfg["profile"], "extract": b.rec["extract"],
           "core_source": _source(b, "core"), "map_source": _source(b, "map"),
           "status": ("quick profile: a pipeline check, never a box-quality claim" if b.cfg["profile"] != "full"
                      else "implemented (research box; unqualified)"),
           "donor_quality": {k: ver.get(k) for k in ("verdict", "units", "export_held", "export_val", "held_split",
                                                     "random_same_count_held", "random_margin", "val_held_gap")},
           "gap_screen": json.loads((b.out / "gap/gap.json").read_text()),
           "gate": {k: gate.get(k) for k in ("status", "layer", "threshold", "held_false_open_rate", "held_miss_rate",
                                             "held_false_open_english", "held_false_open_off_topic")},
           "cross_box_tiles": [], "cross_box_note": "one extracted box: no other box on this donor to share units with",
           "taxonomy": {**inp["taxonomy"], "file": "taxonomy/" + Path(inp["taxonomy"]["path"]).name},
           "map": {"file": "map/map.json", "sha256": files["map/map.json"]["sha256"]},
           "prompt_set": {"file": f"prompt-set/{pname}.parquet", "sha256": ps["sha256"], "rows": ps["rows"]},
           "weights": {"file": "weights/box_weights.npz", "index": "weights/box_weights.json",
                       "not_copied": "attention, norms, embeddings and the core run from the donor checkpoint"},
           "routes": {"side_stream": "the donor core + this box beside the host, blended at the output by the gate "
                                     "(mapping-workflow s11.3; `mlx2 amoeba install`, not built yet)",
                      "mapped_injection": "open for later (needs the per-pair warp and ridge maps)",
                      "distilled_tile": "open for later"},
           "config": b.cfg, "files": files}
    write_json(d / "manifest.json", man)
    return file_outputs({"manifest": d / "manifest.json"}), \
        {"box": name, "verdict": ver["verdict"], "files": len(files), "prompt_set_rows": ps["rows"]}


def _source(b, what):
    p = b.out / what / "cache.json"
    if p.exists():
        c = json.loads(p.read_text())
        return {"reused_from_cache": True, **{k: c.get(k) for k in ("from_build", "mlx2_commit", "device", "key")}}
    return {"reused_from_cache": False, "carved_in_this_run": True}


EXTRACT_RUNNERS = {"gap": run_gap, "map": run_xmap, "core": run_xcore, "box": run_box, "box_gate": run_box_gate,
                   "box_bundle": run_box_bundle}
