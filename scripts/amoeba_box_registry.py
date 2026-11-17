#!/usr/bin/env python3
"""Amoeba box registry: provenance and an append-only changelog for every box.

One JSON file per box under docs/research/amoeba-box-registry/<box_id>.json (schema
mlx2.amoeba-box.v1). A box records:
  identity   : box_id, kind (core | box | head-set | table), topic tags
  origin     : donor model name, snapshot revision, weight-file sha256 values, source dtype /
               quantization, and the exact units copied: per layer, expert id (null for dense
               donors) and neuron ids; attention heads as layer/head
  versions   : content hash of the box tensors at each version (v1, v2, ...)
  changelog  : append-only entries -- what happened, when, by which tool and mlx2 commit,
               with parameters, input/output hashes and metrics. Training after boxing
               (gates, weight tuning, merges, quantization, exports) is always a new entry,
               never an edit of an old one.

CLI:
  ingest-skill-boxes  register the core + every box in a skill_boxes.pt (M0b-2 output)
  ingest-mask-box     register one box from a mask tensor file (key path into a .pt)
  log                 append a changelog entry to a box (optionally bumping its version)
  show                print a box record
  verify              recompute the latest version's content hash from its tensor file
  annotate            regroup units by a neuron atlas's virtual experts (dense donors) and tag them
  family              list a family's variants with status and A/B scoreboard
  status              set a variant's implemented / qualified / observed_used flags (with evidence)
  select              point a family's default at a variant (records who approved the change)
  path-add            add a path to a box: alternative content built from the donor's runs at one
                      effort level (planned until it has a tensor)
  path-status         set a path's implemented / qualified / observed_used flags (with evidence)
  path-evidence       append a test result to a path (checkpoint, metrics, evidence path)
  path-select         make one path the box default (needs --approved-by)
  paths               list a box's paths with status and last result

Families and variants: every box belongs to a family (a slot such as "capitals"); boxes in a
family are alternative chains (variants). Status flags stay distinct, as the repo requires:
implemented, qualified, selected, observed_used. Only one variant per family is selected.

Paths: inside one box, paths are alternatives built from the donor run at different effort levels.
A request that names no path runs the selected path (resolve_path fails closed when none is selected
or the named path is not implemented), so a host with no effort control can still run, say, a path
built from an xhigh donor.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "docs" / "research" / "amoeba-box-registry"
SCHEMA = "mlx2.amoeba-box.v1"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha_tensor(t):
    t = t.detach().to("cpu").contiguous()
    return hashlib.sha256(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes()).hexdigest()


def donor_record(model, dtype_note, fingerprint=None):
    """HF hub snapshot (by repo id), or a local artifact directory (named by its fingerprint)."""
    if Path(model).expanduser().is_dir():
        snap = Path(model).expanduser().resolve()
        rev = fingerprint or snap.name
        name = snap.name
    else:
        from huggingface_hub import snapshot_download
        snap = Path(snapshot_download(model, local_files_only=True))
        rev, name = snap.name, model
    cfg = json.loads((snap / "config.json").read_text())
    cfg = {**cfg, **(cfg.get("text_config") or {})}
    return {"model": name, "snapshot_revision": rev, "artifact": f"{name}@{rev[:8]}", "path": str(snap),
            "weights_sha256": {Path(w).name: sha_file(w) for w in sorted(glob.glob(str(snap / "*.safetensors")))},
            "source_dtype": cfg.get("torch_dtype") or cfg.get("dtype"), "quantization": cfg.get("quantization"),
            "handling": dtype_note, "architecture": (cfg.get("architectures") or [None])[0],
            "layers": cfg.get("num_hidden_layers"), "hidden_size": cfg.get("hidden_size"),
            "intermediate_size": cfg.get("intermediate_size")}


def units_from_mask(mask, expert=None):
    """Per-layer neuron ids of a [layers, neurons] 0/1 mask (expert id for MoE donors)."""
    out = []
    for l in range(mask.shape[0]):
        ids = mask[l].nonzero().flatten().tolist()
        if ids:
            out.append({"layer": l, "expert": expert, "neurons": ids})
    return out


def path_of(box_id):
    return REG / f"{box_id}.json"


def load_box(box_id):
    p = path_of(box_id)
    return json.loads(p.read_text()) if p.exists() else None


def save_box(rec):
    REG.mkdir(parents=True, exist_ok=True)
    p = path_of(rec["box_id"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.rename(p)


def entry(action, *, params=None, inputs=None, outputs=None, metrics=None, note=None, session=None):
    return {"at": now(), "action": action, "tool": Path(sys.argv[0]).name if sys.argv else None,
            "mlx2_source_revision": git_rev(), "session": session or os.environ.get("AMOEBA_SESSION"),
            "params": params or {}, "inputs": inputs or {}, "outputs": outputs or {}, "metrics": metrics or {},
            "note": note}


def register(box_id, kind, tags, donor, units, content_hash, tensor_ref, first_entry, heads=None):
    rec = load_box(box_id)
    if rec is not None:
        if rec["versions"][-1]["content_sha256"] == content_hash:
            rec["changelog"].append(dict(first_entry, action="re-ingested (unchanged)"))
            save_box(rec)
            return rec
        v = len(rec["versions"]) + 1
        rec["versions"].append({"version": f"v{v}", "content_sha256": content_hash, "tensor": tensor_ref, "at": now(),
                                "units_count": sum(len(u["neurons"]) for u in units)})
        rec["origin"]["units"] = units
        rec["changelog"].append(dict(first_entry, version=f"v{v}"))
        save_box(rec)
        return rec
    rec = {"schema": SCHEMA, "box_id": box_id, "kind": kind, "tags": tags,
           "family": (tags[0] if tags else box_id), "variant_of": None,
           "status": {"implemented": True, "qualified": False, "selected": False, "observed_used": False,
                      "evidence": {}},
           "ab": [],
           "origin": {"donor": donor, "units": units, "heads": heads or [],
                      "units_count": sum(len(u["neurons"]) for u in units)},
           "versions": [{"version": "v1", "content_sha256": content_hash, "tensor": tensor_ref, "at": now(),
                         "units_count": sum(len(u["neurons"]) for u in units)}],
           "changelog": [dict(first_entry, version="v1")]}
    save_box(rec)
    return rec


def cmd_ingest_skill_boxes(a):
    sk = torch.load(a.skill_boxes, map_location="cpu")
    run = json.loads(Path(a.run_json).read_text()) if a.run_json else {}
    donor = donor_record(a.donor, a.dtype_note, getattr(a, "donor_fingerprint", "") or None)
    src = {"file": a.skill_boxes, "sha256": sha_file(a.skill_boxes)}
    produced_by = {"script": a.produced_by, "mlx2_source_revision": run.get("mlx2_source_revision"),
                   "checkpoint": run.get("checkpoint"), "device": run.get("device")}
    core = sk["core"]
    ablation = "zero" if float(sk["mean"].abs().max()) == 0 else "mean"
    core_id = a.prefix + "english-core"
    rc = register(core_id, "core", ["english", "core"], donor, units_from_mask(core), sha_tensor(core),
                  {"file": a.skill_boxes, "key": "core"},
                  entry("selected", params={"method": "shared sigmoid mask over MLP neurons, KL to full model",
                                            "ablation": ablation, "source_run": a.run_json,
                                            "core_masks": run.get("core_masks"), "produced_by": produced_by},
                        inputs={"skill_boxes": src}, metrics={"kept_fraction": float(core.mean()),
                                                              "eval": run.get("eval", {}).get("core")}))
    print(f"{rc['box_id']}: {rc['versions'][-1]['version']} ({rc['origin']['units_count']} neurons)")
    for name, B in sk["boxes"].items():
        picked = run.get("boxes", {}).get(f"{name}_picked", {})
        rows = run.get("boxes", {}).get(name, [])
        rec = register(a.prefix + f"box-{name}", "box", [name], donor, units_from_mask(B), sha_tensor(B),
                       {"file": a.skill_boxes, "key": f"boxes.{name}"},
                       entry("selected", params={"method": "straight-through mask over non-core neurons, KL to full model",
                                                 "core": core_id, "lambda": picked.get("lam"), "sweep": rows,
                                                 "ablation": ablation, "produced_by": produced_by},
                             inputs={"skill_boxes": src},
                             metrics={"neurons": int(B.sum()), "eval_core_plus_box": run.get("eval", {}).get(f"core+{name}")}))
        if name in sk.get("gates", {}):
            w, b = sk["gates"][name]
            rec["changelog"].append(entry("gate trained", params={"gate": "per-layer sigmoid(x.w_l + b_l)",
                                                                  "produced_by": produced_by,
                                                                  "trained_with": ["capitals", "arithmetic", "french", "recall", "english"],
                                                                  "objective": "KL to full model + openness penalty on English"},
                                          outputs={"gate_sha256": sha_tensor(torch.cat([w.flatten(), b.flatten()]))},
                                          metrics={"eval_all_gated": run.get("eval", {}).get("core+all_gated")}))
            save_box(rec)
        print(f"{rec['box_id']}: {rec['versions'][-1]['version']} ({rec['origin']['units_count']} neurons)")


def cmd_ingest_mask_box(a):
    obj = torch.load(a.file, map_location="cpu")
    for k in a.key.split("."):
        obj = obj[k]
    donor = donor_record(a.donor, a.dtype_note)
    heads = [{"layer": int(x.split("H")[0][1:]), "head": int(x.split("H")[1])} for x in a.heads.split(",") if x] if a.heads else []
    rec = register(a.box_id, a.kind, a.tags.split(","), donor, units_from_mask(obj, a.expert), sha_tensor(obj),
                   {"file": a.file, "key": a.key},
                   entry(a.action, params={**json.loads(a.params or "{}"), "produced_by": a.produced_by or None},
                         metrics=json.loads(a.metrics or "{}"),
                         inputs={"file": {"path": a.file, "sha256": sha_file(a.file)}}, note=a.note), heads=heads)
    print(f"{rec['box_id']}: {rec['versions'][-1]['version']} ({rec['origin']['units_count']} neurons)")


def cmd_log(a):
    rec = load_box(a.box)
    if rec is None:
        sys.exit(f"no such box {a.box}")
    e = entry(a.action, params=json.loads(a.params or "{}"), metrics=json.loads(a.metrics or "{}"),
              inputs={p: sha_file(p) for p in (a.inputs or [])}, outputs={p: sha_file(p) for p in (a.outputs or [])},
              note=a.note)
    if a.new_version_from:
        f, key = a.new_version_from.split("#")
        obj = torch.load(f, map_location="cpu")
        for k in key.split("."):
            obj = obj[k]
        v = f"v{len(rec['versions']) + 1}"
        rec["versions"].append({"version": v, "content_sha256": sha_tensor(obj), "tensor": {"file": f, "key": key}, "at": now()})
        e["version"] = v
    rec["changelog"].append(e)
    save_box(rec)
    print(f"{a.box}: {len(rec['changelog'])} changelog entries, latest {rec['versions'][-1]['version']}")


def cmd_annotate(a):
    """Group each box's units by the neuron atlas's virtual experts (dense donors) and tag them."""
    atlas = torch.load(Path(a.atlas) / "atlas.pt", map_location="cpu")
    summ = json.loads((Path(a.atlas) / "atlas_summary.json").read_text())
    ve = atlas["virtual_expert"]
    for f in sorted(REG.glob("*.json")):
        rec = json.loads(f.read_text())
        if rec["origin"]["donor"]["artifact"] != atlas["artifact"]:
            continue
        units = []
        ves = {}
        for u in rec["origin"]["units"]:
            groups = {}
            for n in u["neurons"]:
                groups.setdefault(int(ve[u["layer"], n]), []).append(n)
            for k, ns in sorted(groups.items()):
                vid = f"L{u['layer']}:VE{k:02d}"
                units.append({"layer": u["layer"], "expert": u.get("expert"), "virtual_expert": vid,
                              "virtual": True, "neurons": ns})
                ves[vid] = ves.get(vid, 0) + len(ns)
        top = sorted(ves.items(), key=lambda x: -x[1])[:10]
        rec["origin"]["units"] = units
        rec["origin"]["neuron_id_format"] = "<artifact>:L<layer>:N<neuron>"
        rec["origin"]["atlas"] = {"path": str(a.atlas), "schema": summ["schema"],
                                  "experts_per_layer": summ["experts_per_layer"]}
        rec["origin"]["top_virtual_experts"] = [
            {"id": v, "neurons_in_box": c, "expert_size": summ["virtual_experts"][v]["neurons"],
             "most_selective_domain": summ["virtual_experts"][v]["most_selective_domain"],
             "common_top_tokens": summ["virtual_experts"][v]["common_top_tokens"][:5]} for v, c in top]
        rec["changelog"].append(entry("annotated with neuron atlas",
                                      params={"atlas": str(a.atlas), "experts_per_layer": summ["experts_per_layer"]},
                                      metrics={"virtual_experts_touched": len(ves)},
                                      note="units regrouped by virtual expert; tensors unchanged (same version)"))
        save_box(rec)
        print(f"{rec['box_id']}: {len(ves)} virtual experts; top {top[:3]}")


def _family(name):
    out = []
    for f in sorted(REG.glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("family", (r["tags"] or [r["box_id"]])[0]) == name:
            out.append(r)
    return out


def cmd_family(a):
    for r in _family(a.name):
        st = r.get("status", {})
        flags = "".join(k[0].upper() if st.get(k) else "." for k in ("implemented", "qualified", "selected", "observed_used"))
        last = (r.get("ab") or [{}])[-1]
        print(f"{flags} {r['box_id']:36s} v{len(r['versions'])} units={r['origin']['units_count']:6d} "
              f"last A/B: {json.dumps(last.get('metrics', {}))[:110]}")


def cmd_status(a):
    r = load_box(a.box)
    r.setdefault("status", {"implemented": True, "qualified": False, "selected": False, "observed_used": False, "evidence": {}})
    for k in ("implemented", "qualified", "observed_used"):
        v = getattr(a, k)
        if v is not None:
            r["status"][k] = v == "yes"
            r["status"]["evidence"][k] = a.evidence
    r["changelog"].append(entry("status changed", params={k: getattr(a, k) for k in ("implemented", "qualified", "observed_used")},
                                note=a.evidence))
    save_box(r)
    print(r["box_id"], r["status"])


def cmd_select(a):
    target = load_box(a.box)
    fam = target.get("family", target["tags"][0])
    if not a.approved_by:
        sys.exit("selecting a family default needs --approved-by (defaults change only with explicit approval)")
    for r in _family(fam):
        was = r.get("status", {}).get("selected", False)
        now_sel = r["box_id"] == a.box
        if was != now_sel:
            r.setdefault("status", {})["selected"] = now_sel
            r["changelog"].append(entry("selected as family default" if now_sel else "deselected as family default",
                                        params={"family": fam, "approved_by": a.approved_by}, note=a.note))
            save_box(r)
    print(f"{fam}: default -> {a.box} (approved by {a.approved_by})")


# --------------------------------------------------------------------- paths
# A path is one alternative content of a box, built from the donor's runs at one effort level
# (completion, off, low, medium, xhigh, ...). The host never needs an effort control: a request that
# names no path runs the box's selected path. Each path keeps its own four status flags; one path
# per box is selected, and only with explicit approval.

PATH_FLAGS = ("implemented", "qualified", "selected", "observed_used")


def _path(rec, name):
    p = (rec.get("paths") or {}).get(name)
    if p is None:
        sys.exit(f"{rec['box_id']} has no path {name!r}; paths: {sorted(rec.get('paths') or {})}")
    return p


def resolve_path(rec, requested=None):
    """Return (path name, record) to run, failing closed.

    A named path must exist and be implemented. With no name, the selected path runs; a box with
    paths but none selected refuses rather than guessing."""
    paths = rec.get("paths") or {}
    if requested is not None:
        p = paths.get(requested)
        if p is None or not p["status"]["implemented"]:
            raise LookupError(f"{rec['box_id']}: path {requested!r} is not an implemented path")
        return requested, p
    chosen = [n for n, p in paths.items() if p["status"]["selected"]]
    if not paths:
        return None, None
    if len(chosen) != 1:
        raise LookupError(f"{rec['box_id']}: {len(chosen)} selected paths; select one with path-select")
    return chosen[0], paths[chosen[0]]


def cmd_path_add(a):
    rec = load_box(a.box)
    if rec is None:
        sys.exit(f"no such box {a.box}")
    paths = rec.setdefault("paths", {})
    if a.path in paths:
        sys.exit(f"{a.box} already has path {a.path!r}; use path-log to record a new version")
    tensor, digest = None, None
    if a.tensor:
        file, _, key = a.tensor.partition("#")
        digest = sha_file(file)
        tensor = {"file": file, "key": key or None}
    paths[a.path] = {"donor_effort": a.donor_effort, "donor": a.donor, "traces": a.traces,
                     "params": json.loads(a.params) if a.params else {}, "tensor": tensor, "content_sha256": digest,
                     "created": now(), "mlx2_source_revision": git_rev(),
                     "status": {"implemented": tensor is not None, "qualified": False, "selected": False,
                                "observed_used": False, "evidence": {}},
                     "ab": []}
    rec["changelog"].append(entry("path added", params={"path": a.path, "donor_effort": a.donor_effort, "donor": a.donor},
                                  inputs=[a.traces] if a.traces else None, outputs=[a.tensor] if a.tensor else None,
                                  note=a.note))
    save_box(rec)
    print(f"{a.box}: path {a.path} (donor effort {a.donor_effort}) implemented={tensor is not None}")


def cmd_path_status(a):
    rec = load_box(a.box)
    p = _path(rec, a.path)
    for k in ("implemented", "qualified", "observed_used"):
        v = getattr(a, k)
        if v is not None:
            if k == "implemented" and v == "yes" and not p.get("tensor"):
                sys.exit("a path without a tensor cannot be implemented")
            p["status"][k] = v == "yes"
            p["status"]["evidence"][k] = a.evidence
    rec["changelog"].append(entry("path status changed", params={"path": a.path, **{k: getattr(a, k) for k in ("implemented", "qualified", "observed_used")}},
                                  note=a.evidence))
    save_box(rec)
    print(a.box, a.path, p["status"])


def cmd_path_evidence(a):
    rec = load_box(a.box)
    p = _path(rec, a.path)
    item = {"at": now(), "mlx2_source_revision": git_rev(), "checkpoint": a.checkpoint,
            "metrics": json.loads(a.metrics), "evidence": a.evidence}
    p["ab"].append(item)
    rec["changelog"].append(entry("path evidence", params={"path": a.path, "checkpoint": a.checkpoint},
                                  metrics=item["metrics"], note=a.evidence))
    save_box(rec)
    print(a.box, a.path, item["metrics"])


def cmd_path_select(a):
    if not a.approved_by:
        sys.exit("selecting a default path needs --approved-by (defaults change only with explicit approval)")
    rec = load_box(a.box)
    target = _path(rec, a.path)
    if not target["status"]["implemented"]:
        sys.exit(f"path {a.path!r} is not implemented")
    for name, p in rec["paths"].items():
        p["status"]["selected"] = name == a.path
    rec["changelog"].append(entry("path selected as box default", params={"path": a.path, "approved_by": a.approved_by},
                                  note=a.note))
    save_box(rec)
    print(f"{a.box}: default path -> {a.path} (approved by {a.approved_by})")


def cmd_paths(a):
    rec = load_box(a.box)
    for name, p in sorted((rec.get("paths") or {}).items()):
        flags = "".join(k[0].upper() if p["status"].get(k) else "." for k in PATH_FLAGS)
        last = (p.get("ab") or [{}])[-1]
        print(f"{flags} {name:16s} effort={p['donor_effort']:8s} donor={p['donor']} "
              f"last: {json.dumps(last.get('metrics', {}))[:100]}")


def cmd_show(a):
    rec = load_box(a.box)
    if rec is None:
        sys.exit(f"no such box {a.box}")
    if a.full:
        print(json.dumps(rec, indent=1))
        return
    o = rec["origin"]
    print(f"{rec['box_id']} [{rec['kind']}] tags={rec['tags']} donor={o['donor']['artifact']} "
          f"({o['donor']['source_dtype']}, {o['donor']['handling']}) units={o['units_count']} "
          f"layers={[u['layer'] for u in o['units']]} heads={o['heads']}")
    for v in rec["versions"]:
        print(f"  {v['version']} {v['content_sha256'][:16]} {v['tensor']}")
    for e in rec["changelog"]:
        print(f"  {e['at']} {e.get('version', ''):3s} {e['action']} ({e['tool']} @ {e['mlx2_source_revision']}) "
              f"{json.dumps(e['metrics'])[:120]}")


def cmd_verify(a):
    rec = load_box(a.box)
    v = rec["versions"][-1]
    if v.get("hash_type") == "file" or v.get("has_deltas"):     # whole payload (mask + deltas + gate)
        ok = sha_file(v["tensor"]["file"]) == v["content_sha256"]
    else:
        obj = torch.load(v["tensor"]["file"], map_location="cpu")
        for k in v["tensor"]["key"].split("."):
            obj = obj[k]
        ok = sha_tensor(obj) == v["content_sha256"]
    print(f"{a.box} {v['version']}: {'OK' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("ingest-skill-boxes")
    s.add_argument("--skill-boxes", required=True)
    s.add_argument("--run-json", default="")
    s.add_argument("--donor", default="Qwen/Qwen3-0.6B")
    s.add_argument("--dtype-note", default="bf16 weights, fp32 compute in the probes")
    s.add_argument("--prefix", default="qwen3-0.6b-")
    s.add_argument("--donor-fingerprint", default="", help="for a local artifact directory")
    s.add_argument("--produced-by", default="scripts/probe_amoeba_skill_boxes_cpu.py")
    s.set_defaults(fn=cmd_ingest_skill_boxes)
    s = sp.add_parser("ingest-mask-box")
    for k in ("--file", "--key", "--box-id"):
        s.add_argument(k, required=True)
    s.add_argument("--kind", default="box")
    s.add_argument("--tags", default="")
    s.add_argument("--donor", default="Qwen/Qwen3-0.6B")
    s.add_argument("--dtype-note", default="bf16 weights, fp32 compute in the probes")
    s.add_argument("--expert", type=int, default=None)
    s.add_argument("--heads", default="", help="comma list like L14H5,L17H12")
    s.add_argument("--action", default="selected")
    s.add_argument("--params", default="")
    s.add_argument("--metrics", default="")
    s.add_argument("--note", default="")
    s.add_argument("--produced-by", default="")
    s.set_defaults(fn=cmd_ingest_mask_box)
    s = sp.add_parser("log")
    s.add_argument("--box", required=True)
    s.add_argument("--action", required=True)
    s.add_argument("--params", default="")
    s.add_argument("--metrics", default="")
    s.add_argument("--inputs", nargs="*")
    s.add_argument("--outputs", nargs="*")
    s.add_argument("--note", default="")
    s.add_argument("--new-version-from", default="", help="file.pt#key: record a new content version")
    s.set_defaults(fn=cmd_log)
    s = sp.add_parser("family")
    s.add_argument("--name", required=True)
    s.set_defaults(fn=cmd_family)
    s = sp.add_parser("status")
    s.add_argument("--box", required=True)
    for k in ("--implemented", "--qualified", "--observed-used"):
        s.add_argument(k, choices=["yes", "no"], default=None)
    s.add_argument("--evidence", required=True)
    s.set_defaults(fn=cmd_status)
    s = sp.add_parser("select")
    s.add_argument("--box", required=True)
    s.add_argument("--approved-by", default="")
    s.add_argument("--note", default="")
    s.set_defaults(fn=cmd_select)
    s = sp.add_parser("path-add")
    s.add_argument("--box", required=True)
    s.add_argument("--path", required=True, help="path name, e.g. xhigh, low, off")
    s.add_argument("--donor-effort", required=True, help="the donor effort level this path was built from")
    s.add_argument("--donor", required=True, help="<artifact>@<fingerprint> of the donor that produced the runs")
    s.add_argument("--traces", default="", help="donor-run manifest the path was trained on")
    s.add_argument("--tensor", default="", help="file[#key] of the path payload; without it the path is planned, not implemented")
    s.add_argument("--params", default="")
    s.add_argument("--note", default="")
    s.set_defaults(fn=cmd_path_add)
    s = sp.add_parser("path-status")
    s.add_argument("--box", required=True)
    s.add_argument("--path", required=True)
    for k in ("--implemented", "--qualified", "--observed-used"):
        s.add_argument(k, choices=["yes", "no"], default=None)
    s.add_argument("--evidence", required=True)
    s.set_defaults(fn=cmd_path_status)
    s = sp.add_parser("path-evidence")
    s.add_argument("--box", required=True)
    s.add_argument("--path", required=True)
    s.add_argument("--checkpoint", required=True, help="<artifact>@<fingerprint> the test ran on")
    s.add_argument("--metrics", required=True, help="JSON")
    s.add_argument("--evidence", required=True, help="evidence path")
    s.set_defaults(fn=cmd_path_evidence)
    s = sp.add_parser("path-select")
    s.add_argument("--box", required=True)
    s.add_argument("--path", required=True)
    s.add_argument("--approved-by", default="")
    s.add_argument("--note", default="")
    s.set_defaults(fn=cmd_path_select)
    s = sp.add_parser("paths")
    s.add_argument("--box", required=True)
    s.set_defaults(fn=cmd_paths)
    s = sp.add_parser("annotate")
    s.add_argument("--atlas", required=True, help="directory written by amoeba_neuron_atlas.py")
    s.set_defaults(fn=cmd_annotate)
    for n, fn in (("show", cmd_show), ("verify", cmd_verify)):
        s = sp.add_parser(n)
        s.add_argument("--box", required=True)
        if n == "show":
            s.add_argument("--full", action="store_true")
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
