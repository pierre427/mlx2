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

Options for a donor whose core already exists or whose map is too slow to capture (2026-11-18):
  --core-from NPZ[:KEY] --core-from-reason TEXT   the core stage takes this mask (a dense model's [layers,
             intermediate] core, e.g. the 27B T2 cut) instead of carving one: no cache, its sha256 and the receipts
             beside it are recorded; the English-only check still runs on it
  --no-map REASON   no map stage: the box carve starts unseeded (every free unit at the same init), the bundle
             carries no map, and the reason is recorded
  --exclude-texts FILE   (repeatable) probe texts no stage may see (a reserved final-test set): a JSON object with
             any of "probe_ids", "sha256" (of the text), "groups"; matched rows leave the effective probe corpus
  --extra-topic-corpus FILE   (repeatable) more probe-format rows (id, topics, split, text, group, license) for the
             effective corpus; duplicate and excluded texts drop, and a row whose group sits in the other split
             refuses, as recorded in inputs/effective-probes.json
The bar (box/bar.json, the box receipt and the bundle manifest): pass or fail on BOTH conditions, core + box held
dCE <= +0.05 nats of the full donor, and held dCE <= 0.5 x a same-count random box's.

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
from .stages import (
    _english_corpus,
    _py,
    _reset_partial,
    carve_widths,
    corecheck,
    fit_tags,
    generate,
    probe_corpus,
    probe_corpus_sha,
    read_jsonl,
    run_core,
    run_map,
    unit_clusters,
    write_jsonl,
)
from .state import digest, file_outputs, stage_commits, write_json

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
                 "probe_corpus": probe_corpus_sha(b), "device": device}
        opts = b.rec.get("options") or {}
        extra.update(gen_backend=opts.get("gen_backend"), gen_route=opts.get("gen_route", ""),
                     grad_gate=b.cfg.get("grad_gate"))
    else:
        keys = {k: v for k, v in c.items() if k.startswith("map_") or k in ("ve_per_layer", "cluster_k")}
        extra = {"probe_corpus": probe_corpus_sha(b), "moe_units": b.plan["moe_units"],
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


# ------------------------------------------------------------------ effective probe corpus (exclusions, extra rows)

EXTRA_FIELDS = ("id", "topics", "split", "text", "group", "license")
EXCLUDE_KEYS = ("probe_ids", "sha256", "groups")


def _load_exclusions(paths_: list) -> tuple[dict, list]:
    keys = {k: set() for k in EXCLUDE_KEYS}
    files = []
    for f in paths_:
        f = Path(f).expanduser().resolve()
        try:
            obj = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(f"--exclude-texts {f}: not a JSON object ({e})")
        if isinstance(obj, dict) and obj.get("schema") == "mlx2.amoeba-bridge-split-manifest.v1":
            # a bridge run's split manifest (scripts/amoeba_bridge_split_manifest.py): its reserved final test
            if not obj.get("probes_sha256"):
                raise SystemExit(f"--exclude-texts {f}: a split manifest without probes_sha256 (its corpus is unknown)")
            obj ={**(obj.get("reserved_final_test") or {}), "probes": {"sha256": obj.get("probes_sha256")}}
        if not isinstance(obj, dict) or not any(obj.get(k) for k in keys):
            raise SystemExit(f"--exclude-texts {f}: needs a JSON object with a non-empty list under any of "
                             f"{list(EXCLUDE_KEYS)}")
        n = {}
        for k in keys:
            v = obj.get(k) or []
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise SystemExit(f"--exclude-texts {f}: {k} must be a list of strings")
            keys[k] |= set(v)
            n[k] = len(v)
        files.append({"path": str(f), "sha256": paths.sha256_file(f), "counts": n,
                      "for_probes_sha256": (obj.get("probes") or {}).get("sha256")})
    return keys, files


def build_effective_corpus(src: Path, out: Path, excludes: list, extras: list) -> dict:
    """Write <out>/inputs/<probe build>/corpus.jsonl: the probe corpus minus every excluded text, plus the extra
    topic rows (the directory keeps the probe build's name, so row provenance still names it). Returns the record
    for build.json inputs.probe_corpus_effective; the per-rule accounting goes to <out>/inputs/effective-probes.json."""
    import hashlib
    from collections import Counter
    src = Path(src)
    keys, xfiles = _load_exclusions(excludes)
    src_sha = paths.sha256_file(src)
    for x in xfiles:
        if x["for_probes_sha256"] and x["for_probes_sha256"] != src_sha:
            raise SystemExit(f"--exclude-texts {x['path']} was made for probe corpus {x['for_probes_sha256'][:12]}, "
                             f"not {src} ({src_sha[:12]})")
    tsha = lambda r: hashlib.sha256(r["text"].encode()).hexdigest()       # noqa: E731
    rows = read_jsonl(src)

    def why(r):
        if r.get("id") in keys["probe_ids"]:
            return "probe_id"
        if tsha(r) in keys["sha256"]:
            return "sha256"
        if r.get("group") is not None and str(r.get("group")) in keys["groups"]:
            return "group"
        return None
    kept, dropped = [], []
    for r in rows:
        w = why(r)
        if w:
            dropped.append((w, r))
        else:
            kept.append(r)
    if excludes and not dropped:
        raise SystemExit(f"--exclude-texts matched no row of {src}: an exclusion list for another corpus?")
    seen_ids = {r["id"] for r in rows}
    seen_text = {tsha(r) for r in kept}
    group_split = {}
    for r in rows:
        group_split.setdefault(str(r.get("group")), set()).add(r["split"])
    added, xrec = [], []
    for f in extras:
        f = Path(f).expanduser().resolve()
        n = Counter()
        for i, r in enumerate(read_jsonl(f)):
            miss = [k for k in EXTRA_FIELDS if r.get(k) in (None, "", [])]
            if miss or r["split"] not in ("fit", "held") or not isinstance(r["topics"], list):
                raise SystemExit(f"--extra-topic-corpus {f} row {i}: needs {EXTRA_FIELDS} (split fit|held, topics a "
                                 f"list); missing or bad: {miss or ['split/topics']}")
            if r["id"] in seen_ids:
                raise SystemExit(f"--extra-topic-corpus {f} row {i}: id {r['id']} is already a probe id")
            other = group_split.get(str(r["group"]), set()) - {r["split"]}
            if other:
                raise SystemExit(f"--extra-topic-corpus {f} row {i}: group {r['group']!r} is in the {sorted(other)} "
                                 f"split already; a group never spans splits")
            w = why(r)
            if w:
                n[f"excluded_{w}"] += 1
                continue
            if tsha(r) in seen_text:
                n["duplicate_text"] += 1
                continue
            seen_ids.add(r["id"])
            seen_text.add(tsha(r))
            group_split.setdefault(str(r["group"]), set()).add(r["split"])
            added.append({**r, "sha256": tsha(r), "corpus_build": r.get("corpus_build") or f.parent.name,
                          "extra_corpus": str(f)})
            n[f"added_{r['split']}"] += 1
        xrec.append({"path": str(f), "sha256": paths.sha256_file(f), "counts": dict(n)})
    d = out / "inputs" / src.parent.name
    d.mkdir(parents=True, exist_ok=True)
    dst = d / "corpus.jsonl"
    write_jsonl(dst, kept + added)
    acct = {"source": str(src), "source_sha256": src_sha, "rows_in": len(rows),
            "rows_out": len(kept) + len(added), "exclude_files": xfiles, "extra_files": xrec,
            "excluded": {"rows": len(dropped), "by_rule": dict(Counter(w for w, _ in dropped)),
                         "by_split": dict(Counter(r["split"] for _, r in dropped)),
                         "by_primary_topic": dict(Counter(r["topics"][0] for _, r in dropped).most_common()),
                         "groups": len({str(r.get("group")) for _, r in dropped}),
                         "ids": sorted(r["id"] for _, r in dropped)},
            "rule": "an excluded text leaves every stage: box carve/val/held selection, gate negatives, the English "
                    "check's probes and the prompt set"}
    write_json(out / "inputs/effective-probes.json", acct)
    return {"path": str(dst), "sha256": paths.sha256_file(dst), "accounting": str(out / "inputs/effective-probes.json"),
            "exclude": [{"path": x["path"], "sha256": x["sha256"]} for x in xfiles],
            "extra": [{"path": x["path"], "sha256": x["sha256"]} for x in xrec],
            "excluded_rows": len(dropped), "added_rows": len(added)}


def external_core(spec: str, reason: str, info: dict) -> dict:
    """--core-from NPZ[:KEY]: a dense model's [layers, intermediate] core mask, checked against the donor's layout."""
    if not reason:
        raise SystemExit("--core-from needs --core-from-reason (who chose this core, and why it is the donor's core)")
    if info.get("kind") != "dense" or info.get("moe"):
        raise SystemExit("--core-from: dense models only (an MoE core's unit layout is the carve host's; carve it)")
    path, _, key = spec.partition(":")
    f = Path(path).expanduser().resolve()
    if not f.exists():
        raise SystemExit(f"--core-from {f}: no such file")
    key = key or "core"
    z = np.load(f)
    if key not in z.files:
        raise SystemExit(f"--core-from {f}: no array {key!r} (has {z.files})")
    m = np.asarray(z[key]) > 0
    want = (int(info["layers"]), int(info["intermediate"]))
    if m.shape != want:
        raise SystemExit(f"--core-from {f}:{key}: shape {m.shape}, the donor's MLP layout is {want}")
    if not m.any() or m.all():
        raise SystemExit(f"--core-from {f}:{key}: an empty or full core leaves no box to carve")
    side = sorted(p for p in f.parent.glob("*.json"))
    for p in side:
        try:
            fp = json.loads(p.read_text()).get("fingerprint")
        except (OSError, json.JSONDecodeError, AttributeError):
            fp = None
        if isinstance(fp, str) and len(fp) >= 8 and info.get("fingerprint") and fp != info["fingerprint"]:
            raise SystemExit(f"--core-from {f}: {p.name} beside it names fingerprint {fp[:12]}, not the donor's "
                             f"{info['fingerprint'][:12]}: a core carved on another checkpoint")
    return {"path": str(f), "key": key, "sha256": paths.sha256_file(f), "reason": reason, "shape": list(m.shape),
            "units": int(m.sum()), "share": round(float(m.mean()), 6), "widths": [want[1]] * want[0],
            "receipts_beside": [{"path": str(p), "sha256": paths.sha256_file(p)} for p in side]}


# ------------------------------------------------------------------ stages

def run_gap(b, ctx):
    d = b.stage_dir("gap")
    x = b.rec["extract"]
    if x.get("skip_gap_screen"):
        res = {"screen": "skipped", "reason": x["skip_gap_screen"], "donor_leads": None,
               "force_extract": x.get("force_extract")}
        write_json(d / "gap.json", res)
        return file_outputs({"gap": d / "gap.json"}), res
    probes = probe_corpus(b)
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


def _map_skip(b) -> str | None:
    return ((b.rec.get("stage_options") or {}).get("map") or {}).get("skip")


def run_xmap(b, ctx):
    skip = _map_skip(b)
    if skip:
        d = b.stage_dir("map")
        write_json(d / "skipped.json", {"map": "none", "reason": skip,
                                        "effect": "the box carve starts unseeded (every free unit at the same init); "
                                                  "the bundle carries no map"})
        return file_outputs({"skipped": d / "skipped.json"}), {"map": "skipped", "reason": skip}
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


def _core_from(b) -> dict | None:
    return ((b.rec.get("stage_options") or {}).get("core") or {}).get("core_from")


def run_given_core(b, ctx, cf: dict):
    """The core stage on a given mask: core.npz from it (sha256 checked), core_source.json (widths, provenance),
    the English corpus, and the English-only check. Never published to the fingerprint cache."""
    d = b.stage_dir("core")
    if paths.sha256_file(cf["path"]) != cf["sha256"]:
        raise SystemExit(f"--core-from {cf['path']} changed since the extraction started (sha256)")
    m = np.load(cf["path"])[cf["key"]] > 0
    np.savez_compressed(d / "core.npz", core=m.astype(np.uint8))
    write_json(d / "core_source.json", {**cf, "carved_in_this_run": False})
    eng = d / "english.jsonl"
    ec = _english_corpus(b, eng) if not eng.exists() else None
    chk = corecheck(b, ctx, d, eng)
    cc = json.loads(chk.read_text())
    outs = file_outputs({"core": d / "core.npz", "core_source": d / "core_source.json", "corecheck": chk,
                         "english": eng})
    return outs, {"core_from": cf["path"], "core_sha256": cf["sha256"], "reason": cf["reason"], "units": cf["units"],
                  "share": cf["share"], "english_only": cc.get("english_only"), "english_d_ce": cc.get("english_d_ce"),
                  "non_english_median_d_ce": cc.get("non_english_median_d_ce"), "english_corpus": ec}


def run_xcore(b, ctx):
    cf = _core_from(b)
    if cf:
        return run_given_core(b, ctx, cf)
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
    lp = b.out / "map/labels/labels.json"
    if not lp.exists():
        return None, {"seed": f"none: no map ({_map_skip(b) or 'no labels'})", "clusters": []}
    labels = json.loads(lp.read_text())
    hit = []
    from .cli import topic_matches
    for c in labels["clusters"]:
        if any(topic_matches(topic, t) for t in fit_tags(c)):
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
                  ["--corpus", probe_corpus(b), "--out", bs, "--topic", topic,
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
    widths = carve_widths(b)
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
    bar = bar_record(b, ctx, ver)
    write_json(d / "bar.json", bar)
    summ = {"box": name, "verdict": ver["verdict"], "bar": bar["conditions"], "units": ver.get("units"),
            "held_d_ce": ver.get("export_held"),
            "val_d_ce": ver.get("export_val"), "random_same_count_held": ver.get("random_same_count_held"),
            "seed_clusters": len(seed["clusters"]), "gen_rows": gen.get("rows"), "gen_route": gen.get("route")}
    return file_outputs({"verdict": d / "carves/verdicts.jsonl", "boxset": bs / "boxes.json",
                         "next_round": d / "next-round.json", "bar": d / "bar.json"}), summ


def bar_record(b, ctx, ver: dict) -> dict:
    """The aMoEba carving bar, stated per condition (cli.md / mapping-workflow): (1) core + box held dCE within the
    held target of the full donor; (2) held dCE <= the ratio x a same-count random box's (and the random box worse by
    the margin floor). The size was chosen on validation documents at the val target; held only confirms."""
    v = b.cfg["verdict"]
    hv, rnd = ver.get("export_held"), ver.get("random_same_count_held")
    ratio = None if hv is None or not rnd else round(hv / rnd, 4)
    return {"schema": "mlx2.amoeba-bar.v1",
            "checkpoint": f"{b.info['artifact']}@{b.info['fingerprint']}", "mlx2_commit": ctx.commit,
            "box": ver.get("box"), "topic": b.rec["extract"]["topic"], "verdict": ver.get("verdict"),
            "units": ver.get("units"), "val_d_ce": ver.get("export_val"),
            "size_chosen_on": f"validation documents at +{v['val_target_nats']} nats (held only confirms)",
            "conditions": {
                "held_within_target": {"pass": bool(ver.get("held_meets_target")), "held_d_ce": hv,
                                       "target_nats": v["held_target_nats"]},
                "beats_random": {"pass": bool(ver.get("beats_random")), "held_d_ce": hv,
                                 "random_same_count_held_d_ce": rnd, "held_over_random": ratio,
                                 "max_ratio": v["random_control_ratio"],
                                 "margin_floor_nats": v.get("random_margin_floor_nats")}},
            "pass": ver.get("verdict") == "pass", "miss_reasons": ver.get("miss_reasons"),
            "partition": "carve is a partition: the box is units outside the core; nothing is pruned"}


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
                   "--negative-probes", probe_corpus(b),
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
               "mlx2_commit": ctx.commit, "bundled_at_commit": ctx.commit, "stage_commits": stage_commits(b),
               "extract": b.rec["extract"], "donor_quality": ver,
               "gap_screen": json.loads((b.out / "gap/gap.json").read_text()),
               "note": ("the carve kept no units beyond the core: no box to carry"
                        if ver["verdict"] == "covered" else
                        f"the box carve failed ({ver.get('failure', 'crash')}): no box to carry"),
               "gate_cells": ver.get("gate_cells"),
               "bar": json.loads((b.out / "box/bar.json").read_text()) if (b.out / "box/bar.json").exists() else None}
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
    has_map = not _map_skip(b)
    maps = ({"map/map.json": b.out / "map/map.json", "map/clusters.npz": b.out / "map/clusters/clusters.npz",
             "map/clusters.json": b.out / "map/clusters/clusters.json",
             "map/labels.json": b.out / "map/labels/labels.json"} if has_map
            else {"map/skipped.json": b.out / "map/skipped.json"})
    extra_core = {"core/core_source.json": b.out / "core/core_source.json"} if _core_from(b) else {}
    eff = inp.get("probe_corpus_effective")
    copies = {"core/core.npz": b.out / "core/core.npz", "core/corecheck.json": b.out / "core/corecheck.json",
              **extra_core,
              "gate/gates.npz": b.out / "box_gate/gates.npz", "gate/gates.json": b.out / "box_gate/gates.json",
              **maps, "report/bar.json": b.out / "box/bar.json",
              **({"taxonomy/effective-probes.json": Path(eff["accounting"])} if eff else {}),
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
    runs = ([] if _core_from(b) else ["--carve-run", b.out / "core/carve", "--stage", "core"]) + \
        ["--carve-run", b.out / f"box/carves/box-{name}", "--stage", f"box:{name}"] + \
        (["--capture-run", b.out / "map/capture", "--stage", "map"] if has_map else [])
    run_child([ap, "-u", paths.repo_root() / "scripts/amoeba_prompt_set.py", "--out", d / "prompt-set", "--name", pname,
               "--probe-corpus", probe_corpus(b), "--checkpoint", b.checkpoint,
               "--taxonomy-version", inp["taxonomy"]["version"]] + runs,
              stage="box_bundle", progress=ctx.progress, log_path=b.log_path("box_bundle"), what="prompt set")
    ps = json.loads((d / f"prompt-set/{pname}.manifest.json").read_text())
    gate = json.loads((b.out / "box_gate/gates.json").read_text())["boxes"].get(name, {})
    files = {str(f.relative_to(d)): {"sha256": paths.sha256_file(f), "bytes": f.stat().st_size}
             for f in sorted(d.rglob("*")) if f.is_file()}
    man = {"schema": "mlx2.amoeba-box-bundle.v1", "topic": b.rec["extract"]["topic"], "box": name,
           "donor": {"checkpoint": b.checkpoint, "fingerprint": b.info["fingerprint"], "artifact": b.info["artifact"],
                     "adapter": b.info["adapter"], "kind": b.info["kind"], "moe_units": b.plan["moe_units"]},
           # mlx2_commit: the commit this bundle was written at; the verdict, gate and gap screen it carries were
           # made at their own stages' commits (stage_commits, from the stage receipts)
           "mlx2_commit": ctx.commit, "bundled_at_commit": ctx.commit, "stage_commits": stage_commits(b),
           "profile": b.cfg["profile"], "extract": b.rec["extract"],
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
           "map": ({"file": "map/map.json", "sha256": files["map/map.json"]["sha256"]} if has_map
                   else {"skipped": _map_skip(b), "file": "map/skipped.json"}),
           "bar": json.loads((b.out / "box/bar.json").read_text()),
           "probe_corpus_effective": eff,
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
    if what == "core" and _core_from(b):
        return {"reused_from_cache": False, "carved_in_this_run": False, "core_from": _core_from(b)}
    if what == "map" and _map_skip(b):
        return {"reused_from_cache": False, "skipped": _map_skip(b)}
    p = b.out / what / "cache.json"
    if p.exists():
        c = json.loads(p.read_text())
        return {"reused_from_cache": True, **{k: c.get(k) for k in ("from_build", "mlx2_commit", "device", "key")}}
    return {"reused_from_cache": False, "carved_in_this_run": True}


EXTRACT_RUNNERS = {"gap": run_gap, "map": run_xmap, "core": run_xcore, "box": run_box, "box_gate": run_box_gate,
                   "box_bundle": run_box_bundle}
