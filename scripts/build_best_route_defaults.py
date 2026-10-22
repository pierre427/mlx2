#!/usr/bin/env python3
"""Build ``src/mlx2/best_route_defaults.json`` from measured h2h evidence.

GPU-free.  For every checkpoint measured by the best-route-defaults campaign
(``qualification/runs/best-route-defaults-20261021/h2h/<slug>__<arm>.json``)
this reads the plain-decode arm (``ordinary``), every candidate arm and the
frozen tree's no-flag arm (``default``, today's route), applies the selection
rule in ``mlx2.best_route_defaults.rule_verdict`` and writes one table entry
keyed by the artifact fingerprint the adapter registry computes.  The served
route, drafter path and drafter identity pin of each arm come from the
server's own ``/v1/status`` snapshot recorded in the h2h receipt.

Family-level and pending entries that the campaign did not measure are
declared in ``STATIC_ENTRIES`` below with their evidence.  The summary of
every verdict is written beside the receipts (``summary.json``,
``SUMMARY.md``).

usage: build_best_route_defaults.py [--check]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.adapters.drafter_pairing import display_name
from mlx2.best_route_defaults import (
    RULE,
    SCHEMA,
    TABLE_PATH,
    rule_cells,
    rule_verdict,
    validate_entry,
)

RUN = ROOT / "qualification/runs/best-route-defaults-20261021"
# Campaign windows: the first ran on frozen main 7ad53cc86 until the 15:17
# session loss; -r2 resumed on the branch's frozen tree (with the APCv2
# warm-hit hang fix and the q4 g64 lane DFlash2 drafter default).  A slug is
# read from the first directory that has it.
RUNS = (RUN, ROOT / "qualification/runs/best-route-defaults-20261021-r2")
PROMPTS = {"code": "code", "chat": "chat", "spomin7k": "long", "rewrite": "copy"}
MODEL_MATRIX = "qualification/runs/model-matrix-20261021/SUMMARY.md"
RETRO = "docs/research/RETRO-2026-10-18-to-21.md"

# Families the campaign did not change, with the evidence for keeping them.
STATIC_ENTRIES = [
    {
        "family": "FlashNextAdapter",
        "status": "kept_current_default",
        "default": {"kind": "adapter_default"},
        "reason": (
            "native MTP stays: DFlash2 external lost to native MTP on every B1 smoke cell "
            "(Flash-Next 4bit code 80 vs 113, explain 65 vs 90) and Flash-Next packed AST "
            "lost to ordinary (23.7 vs 63.1); B4 rule cells not measured"
        ),
        "evidence": {"files": [MODEL_MATRIX, RETRO],
                     "source_revision": "ca786630c6c48cd07cc9d384a2747cb2331d3acd"},
    },
    {
        "family": "GptOssAdapter",
        "status": "kept_current_default",
        "default": {"kind": "adapter_default"},
        "reason": (
            "latched adaptive prompt lookup stays (capability default): pipelined parked PLD "
            "runs at ordinary speed with nothing to copy and ~2x on copy; PLD tree tiles "
            "lost (75-89 vs 116 tok/s), so --ast auto routes here to PLD too"
        ),
        "evidence": {"files": [RETRO, "docs/SERVING.md", MODEL_MATRIX],
                     "source_revision": "5020f4f9d937005dbce22f7c8277486ab164ad97"},
    },
    *[
        {
            "family": name,
            "status": "kept_current_default",
            "default": {"kind": "adapter_default"},
            "reason": reason,
            "evidence": {"files": [MODEL_MATRIX],
                         "source_revision": "67dbd6fc6e140917273deba4a792c2e0780c164d"},
        }
        for name, reason in (
            ("Nemotron35SuperAdapter",
             ("native MTP stays (1.2-1.4x ordinary on B1 smoke cells); DSpark and LiLiCoRR "
              "drafters ran below native MTP and below ordinary on prose")),
            ("Nemotron35SuperVLAdapter", "adapter default kept; no drafter route measured above it"),
            ("Nemotron35BoltAdapter",
             ("ordinary stays: LiLiCoRR external ran below ordinary on every B1 smoke cell "
              "except tool calls (code 27 vs 47)")),
            ("Nemotron35LightningAdapter",
             "ordinary stays: native MTP ran below ordinary on every B1 smoke cell (70-94 vs 115-119)"),
            ("Nemotron3SuperAdapter",
             "ordinary stays: native MTP ran below ordinary on every B1 smoke cell (28-37 vs 42-44)"),
        )
    ],
    {
        "family": "MuseGlimmerAdapter",
        "status": "pending",
        "default": {"kind": "adapter_default"},
        "reason": (
            "pending: the Muse assistant drafter accepts ~75% on code but 0 of ~450 proposals "
            "on the 4K transcript and has an open sampled-acceptance defect (being fixed on "
            "another branch); re-measure against the rule once that fix lands on main"
        ),
        "evidence": {"files": [MODEL_MATRIX, RETRO],
                     "source_revision": "67dbd6fc6e140917273deba4a792c2e0780c164d"},
    },
]


def compress(raw: Path) -> Path:
    """The committed form of an h2h receipt: deterministic gzip beside it.

    The raw receipts (status snapshots included) are 3-10 MB each; the
    ``.json.gz`` is what the table cites and git carries.  The raw file stays
    untracked (``.gitignore``)."""
    target = raw.with_name(raw.name + ".gz")
    if not target.exists():
        with open(raw, "rb") as source, open(target, "wb") as sink:
            with gzip.GzipFile(filename="", mode="wb", fileobj=sink, mtime=0, compresslevel=9) as stream:
                stream.write(source.read())
    return target


def load_receipt(path: Path) -> dict:
    with gzip.open(path, "rb") as stream:
        return json.loads(stream.read())


def arm_label(path: Path) -> str:
    return path.name.removesuffix(".json.gz")


def arm_files() -> dict[str, dict[str, Path]]:
    found: dict[str, dict[str, Path]] = {}
    owner: dict[str, Path] = {}
    for run in RUNS:
        for raw in sorted((run / "h2h").glob("*__*.json")):
            compress(raw)
        for path in sorted((run / "h2h").glob("*__*.json.gz")):
            slug, arm = arm_label(path).split("__", 1)
            if owner.setdefault(slug, run) != run:
                continue
            found.setdefault(slug, {})[arm] = path
    return found


def arm_record(path: Path) -> dict:
    data = load_receipt(path)
    cells, statuses = {}, []
    rows = [row for row in data["rows"] if not row.get("warmup") and row.get("sampling") == "greedy"]
    for name, cell in data["cells"].items():
        prompt, sampling, width = name.split("/")
        if sampling != "greedy" or prompt not in PROMPTS:
            continue
        if width == "B1":
            value = cell.get("median_decode_tps")
        else:
            # Decode throughput at width 4: per rep, the sum of the lanes'
            # own decode tok/s (each lane's first-to-last token span), then
            # the median over reps.  Cold-prefix prefill is excluded, as at
            # B1; the h2h aggregate (tokens / makespan) would mostly measure
            # 4 x 7K prefill on the long prompt.
            reps = {}
            for row in rows:
                if row.get("prompt") == prompt and f"B{row.get('width')}" == width:
                    reps.setdefault(row["rep"], []).append(row.get("decode_tps"))
            sums = sorted(sum(values) for values in reps.values()
                          if values and all(isinstance(v, (int, float)) for v in values))
            value = statistics.median(sums) if sums else None
        if cell.get("errors"):
            value = None
        cells[f"{PROMPTS[prompt]}/{width}"] = None if value is None else round(value, 3)
        status = ((cell.get("snapshot_before") or {}).get("/v1/status") or {}).get("json")
        if status:
            statuses.append(status)
    settings = (statuses[0] if statuses else {}).get("settings", {})
    policy = settings.get("execution_policy") or {}
    # The served drafter comes from the window's arm plan (the exact server
    # argv); its identity is verified the way auto-pairing verifies it.
    slug = arm_label(path).split("__", 1)[0]
    plan = path.parent.parent / "logs" / f"{slug}.arms.json"
    draft_model = draft_identity = None
    if plan.exists():
        for item in json.loads(plan.read_text()):
            serve = item["serve"]
            if item["label"] == arm_label(path) and "--ast-drafter" in serve:
                draft_model = serve[serve.index("--ast-drafter") + 1]
    if draft_model and draft_model not in ("mtp", "pld", "auto"):
        draft_identity = verified_identity(data["meta"]["model_path"], draft_model)
    return {
        "file": str(path.relative_to(ROOT)),
        "model_path": data["meta"]["model_path"],
        "mlx2_commit": data["meta"]["mlx2_commit"],
        "cells": cells,
        "route": settings.get("route"),
        "route_selection_source": settings.get("route_selection_source"),
        "draft_model": draft_model,
        "draft_revision": draft_identity,
        "tree": policy.get("batch_size_route"),
        # The drafter arithmetic the arm ran (the Qwen3.8 27B --ast default
        # became the q4 g64 lane drafter on main f24146e24).
        "drafter_kernel": None if not draft_model else drafter_kernel(data),
    }


def drafter_kernel(data: dict) -> dict | None:
    """The drafter arithmetic a served receipt reports (``draft_settings``)."""
    for cell in data["cells"].values():
        status = ((cell.get("snapshot_after") or {}).get("/v1/status") or {}).get("json") or {}
        for receipt in status.get("recent_receipts") or ():
            settings = (receipt.get("speculation") or {}).get("draft_settings")
            if isinstance(settings, dict):
                kernel = settings.get("projection_kernel") or {}
                return {"projection_kernel": kernel.get("kernel", "stock"), "law_id": kernel.get("law_id"),
                        "runtime_quantization": settings.get("runtime_quantization")}
    return None


def verified_identity(model_path: str, draft_path: str) -> str:
    """The drafter identity the target adapter's own inspector verifies."""
    from mlx2.adapters.drafter_pairing import drafter_row, pair_drafter
    from mlx2.adapters.registry import inspect_model

    target = {"adapter": inspect_model(model_path).adapter_type.__name__, "path": model_path}
    row = drafter_row(Path(draft_path))
    if row is None:
        raise SystemExit(f"{draft_path}: not a complete drafter artifact")
    return pair_drafter(target, row, target_revision=False)["identity"]


def checkpoint(model_path: str) -> tuple[str, str, str]:
    from mlx2.adapters.registry import inspect_model

    resolution = inspect_model(model_path)
    fingerprint = resolution.artifact_fingerprint
    return (f"{display_name(Path(model_path))}@sha256:{fingerprint}", fingerprint,
            resolution.adapter_type.__name__)


def drafter_spec(arm: dict) -> dict:
    path = Path(arm["draft_model"])
    config = json.loads((path / "config.json").read_text())
    return {
        "name": display_name(path),
        "architectures": list(config.get("architectures") or []),
        "identities": [arm["draft_revision"]],
        "config_sha256": [hashlib.sha256((path / "config.json").read_bytes()).hexdigest()],
    }


def route_spec(name: str, arm: dict) -> dict:
    if name.startswith("ast"):
        spec = {"name": name, "kind": "ast"}
        if arm.get("draft_model") and arm.get("draft_revision"):
            spec["drafter"] = drafter_spec(arm)
        return spec
    route = arm["route"]
    if route in ("native_mtp", "prompt_lookup", "ordinary"):
        return {"name": route, "kind": route}
    raise ValueError(f"{arm['file']}: no route spec for served route {route}")


def build_entry(slug: str, arms: dict[str, Path]) -> tuple[dict | None, dict]:
    records = {name: arm_record(path) for name, path in arms.items()}
    baselines = [rec for name, rec in records.items() if name.startswith("ordinary")]
    if not baselines:
        return None, {"slug": slug, "status": "incomplete", "arms": sorted(records)}
    # Several plain-decode arms (taken in different windows) are averaged.
    base = dict(baselines[0])
    base["cells"] = {
        cell: (None if any(rec["cells"].get(cell) is None for rec in baselines)
               else round(statistics.fmean(rec["cells"][cell] for rec in baselines), 3))
        for cell in baselines[0]["cells"]
    }
    current = records.get("default")
    current_route = current["route"] if current else None
    candidates = {name: rec["cells"] for name, rec in records.items()
                  if not name.startswith("ordinary")
                  and not (name == "default" and current_route == "ordinary")}
    verdict = rule_verdict(base["cells"], candidates, "default")
    ckpt, fingerprint, adapter = checkpoint(base["model_path"])
    commits = sorted({rec["mlx2_commit"] for rec in records.values()})
    evidence = {
        "files": [rec["file"] for rec in records.values()],
        "source_revision": ";".join(commits),
        "baseline": "ordinary",
        "current_default_route": current_route,
        "scores": verdict["scores"],
        "cells_tok_s": {name: rec["cells"] for name, rec in records.items()},
        "drafter_kernel": {name: rec["drafter_kernel"] for name, rec in records.items()
                           if rec["drafter_kernel"]},
    }
    summary = {"slug": slug, "checkpoint": ckpt, "adapter": adapter, "current_default_route": current_route,
               "winner": verdict["winner"], "decision": verdict["decision"], "scores": verdict["scores"],
               "routes": {name: {"route": rec["route"], "draft_revision": rec["draft_revision"],
                                 "tree": rec["tree"], "drafter_kernel": rec["drafter_kernel"],
                                 "mlx2_commit": rec["mlx2_commit"]}
                          for name, rec in records.items()}}
    score = verdict["scores"].get(verdict["winner"]) or {}
    entry = {"checkpoint": ckpt, "fingerprint": fingerprint, "adapter": adapter}
    if verdict["decision"] == "selected" and verdict["winner"] != "default":
        winner = records[verdict["winner"]]
        entry.update(
            status="selected_unqualified",
            default=route_spec(verdict["winner"], winner),
            reason=(f"{verdict['winner']}: geomean {score['geomean']:.2f}x plain decode, "
                    f"minimum cell {score['min_cell']:.2f}x ({score['min_cell_name']}) over "
                    f"{len(rule_cells())} cells; rule >= {RULE['min_geomean']} and floor "
                    f">= {RULE['min_cell']}"),
        )
        others = sorted(
            (name for name, s in verdict["scores"].items() if s["qualifies"] and name != verdict["winner"]),
            key=lambda name: -verdict["scores"][name]["geomean"],
        )
        # The next-best route by the same rule; when that is today's default
        # (or nothing else qualifies) the adapter default is the failover.
        if others and others[0] != "default":
            entry["fallback"] = [route_spec(others[0], records[others[0]])]
    else:
        detail = "; ".join(
            f"{name} geomean {s['geomean']}x min {s['min_cell']}x" if s["geomean"] is not None
            else f"{name} missing {s['missing']}"
            for name, s in verdict["scores"].items()
        )
        entry.update(
            status="kept_current_default",
            default={"kind": "adapter_default"},
            reason=f"current default ({current_route}) kept: {detail}",
        )
    entry["evidence"] = evidence
    validate_entry(entry)
    return entry, summary


FAMILY_MIN_CHECKPOINTS = 2


def family_entries(entries: list[dict]) -> list[dict]:
    """Family fallbacks derived from measured checkpoints.

    An adapter class gets a family entry when at least
    ``FAMILY_MIN_CHECKPOINTS`` of its checkpoints were measured and every one
    of them selected the same route with the same drafter identities.  The
    entry carries the worst measured geometric mean and minimum cell.  It
    applies only to checkpoints of that class with no entry of their own,
    and an external drafter must still verify on the served checkpoint.
    """
    by_adapter: dict[str, list[dict]] = {}
    for entry in entries:
        if "fingerprint" in entry and not entry["evidence"].get("decision_row"):
            by_adapter.setdefault(entry["adapter"], []).append(entry)
    derived = []
    for adapter, members in sorted(by_adapter.items()):
        if len(members) < FAMILY_MIN_CHECKPOINTS:
            continue
        specs = {json.dumps(m["default"], sort_keys=True) for m in members}
        statuses = {m["status"] for m in members}
        if len(specs) != 1 or len(statuses) != 1:
            continue
        if statuses == {"kept_current_default"}:
            family = {
                "family": adapter, "status": "kept_current_default", "default": {"kind": "adapter_default"},
                "reason": (f"family: all {len(members)} measured {adapter} checkpoints kept their current "
                           "default under the rule"),
                "evidence": {"files": sorted({f for m in members for f in m["evidence"]["files"]}),
                             "source_revision": ";".join(sorted({r for m in members for r in
                                                                 m["evidence"]["source_revision"].split(";")})),
                             "checkpoints": [m["checkpoint"] for m in members]},
            }
            validate_entry(family)
            derived.append(family)
            continue
        if statuses != {"selected_unqualified"}:
            continue
        name = members[0]["default"]["name"]
        scores = [m["evidence"]["scores"][name] for m in members]
        worst = min(scores, key=lambda score: score["min_cell"])
        family = {
            "family": adapter,
            "status": "selected_unqualified",
            "default": members[0]["default"],
            "reason": (f"family fallback: all {len(members)} measured {adapter} checkpoints selected {name} "
                       f"(geomean {min(sc['geomean'] for sc in scores):.2f}-{max(sc['geomean'] for sc in scores):.2f}x, "
                       f"worst minimum cell {worst['min_cell']:.2f}x); unmeasured checkpoints of the class take it, "
                       f"with the drafter verified on the served checkpoint"),
            "evidence": {
                "files": sorted({f for m in members for f in m["evidence"]["files"]}),
                "source_revision": ";".join(sorted({r for m in members
                                                    for r in m["evidence"]["source_revision"].split(";")})),
                "checkpoints": [m["checkpoint"] for m in members],
                "scores": {name: {"qualifies": True, "missing": [], "ratios": {},
                                  "geomean": min(sc["geomean"] for sc in scores),
                                  "min_cell": worst["min_cell"], "min_cell_name": worst["min_cell_name"]}},
            },
        }
        validate_entry(family)
        derived.append(family)
    return derived


DECISIONS = ROOT / "docs/research/model-feature-decisions.tsv"
TABLE_REF = "src/mlx2/best_route_defaults.json"
ROW_ROUTE = re.compile(
    r"Route: (?P<kind>ast|native_mtp|prompt_lookup|ordinary)"
    r"(?: drafter=(?P<drafter>[^@\s]+)@sha256:(?P<identity>[0-9a-f]{64}))?"
)
ROW_ADAPTER = re.compile(r"adapter=(?P<adapter>[A-Za-z0-9_]+)")


def lane_rows() -> tuple[list[dict], list[dict]]:
    """Best-route winners other lanes recorded as decision rows.

    Convention (docs/SERVING.md): ``feature_or_flag=best_route_default``,
    ``selected=yes_default``, an exact ``<name>@sha256:<fingerprint>``
    checkpoint, and a reason containing ``Route: <kind>[ drafter=<name>@sha256:
    <identity>]`` and ``adapter=<AdapterClass>``.  Such a row becomes a table
    entry unless this campaign measured the checkpoint itself.  The drafter
    named in the row is looked up on disk by name only to read its
    architecture and config digest; the served drafter must still verify to
    the row's identity at startup.  Rows this script wrote (they cite the
    table) and rows it cannot parse are returned separately and reported.
    """
    lines = DECISIONS.read_text().splitlines()
    header = lines[0].split("\t")
    entries, skipped = [], []
    drafters = None
    for line in lines[1:]:
        row = dict(zip(header, line.split("\t")))
        if row.get("feature_or_flag") != "best_route_default" or row.get("selected") != "yes_default":
            continue
        if TABLE_REF in row.get("evidence", ""):
            continue
        route, adapter = ROW_ROUTE.search(row.get("reason", "")), ROW_ADAPTER.search(row.get("reason", ""))
        checkpoint_id = row.get("checkpoint", "")
        if not route or not adapter or "@sha256:" not in checkpoint_id:
            skipped.append({"checkpoint": checkpoint_id, "why": "no parseable Route:/adapter= in reason"})
            continue
        spec = {"name": route["kind"] if route["kind"] != "ast" else f"ast_{route['drafter'] or 'auto'}",
                "kind": route["kind"]}
        if route["drafter"]:
            if drafters is None:
                from mlx2.adapters.drafter_pairing import (
                    DEFAULT_ROOTS,
                    discover_drafters,
                )

                drafters = discover_drafters(list(DEFAULT_ROOTS), include_hf=True)
            match = [d for d in drafters if display_name(Path(d["path"])) == route["drafter"]]
            if not match:
                skipped.append({"checkpoint": checkpoint_id, "why": f"drafter {route['drafter']} not on disk"})
                continue
            spec["drafter"] = {
                "name": route["drafter"],
                "architectures": list(match[0].get("architectures") or []),
                "identities": [route["identity"]],
                "config_sha256": sorted({hashlib.sha256((Path(d["path"]) / "config.json").read_bytes()).hexdigest()
                                         for d in match if (Path(d["path"]) / "config.json").exists()}),
            }
        fingerprint = checkpoint_id.rsplit("@sha256:", 1)[1]
        entry = {
            "checkpoint": checkpoint_id, "fingerprint": fingerprint, "adapter": adapter["adapter"],
            "status": "selected_unqualified", "default": spec,
            "reason": f"decision row from another lane: {row.get('reason', '')[:600]}",
            "evidence": {"files": [f for f in row.get("evidence", "").split(";") if f][:8] or [str(DECISIONS.relative_to(ROOT))],
                         "source_revision": row.get("source_revision") or "unknown",
                         "decision_row": True},
        }
        try:
            validate_entry(entry)
        except (TypeError, ValueError) as error:
            skipped.append({"checkpoint": checkpoint_id, "why": str(error)})
            continue
        entries.append(entry)
    return entries, skipped


OPERATOR_DECISION = "Operator decision:"


def merge_lane_entries(entries: list[dict], imported: list[dict], unimported: list[dict]) -> list[dict]:
    """Add other lanes' rows; a measured checkpoint keeps its campaign entry.

    The one exception is a row whose reason starts with ``Operator decision:``
    (Pierre's call overriding the rule for that checkpoint).  It replaces the
    campaign entry, and the campaign's own evidence files stay listed with it.
    """
    by_fingerprint = {entry["fingerprint"]: index for index, entry in enumerate(entries)}
    entries = list(entries)
    for entry in imported:
        index = by_fingerprint.get(entry["fingerprint"])
        if index is None:
            by_fingerprint[entry["fingerprint"]] = len(entries)
            entries.append(entry)
        elif OPERATOR_DECISION in entry["reason"]:
            measured = entries[index]
            entry = dict(entry, evidence=dict(
                entry["evidence"],
                files=list(dict.fromkeys(entry["evidence"]["files"] + measured["evidence"]["files"]))[:8],
                measured_scores=measured["evidence"].get("scores"),
            ))
            validate_entry(entry)
            entries[index] = entry
        else:
            unimported.append({"checkpoint": entry["checkpoint"],
                               "why": "measured by this campaign; campaign entry kept"})
    return entries


def decision_rows(table: dict, source_revision: str) -> list[str]:
    """One ``best_route_default`` decision row per checkpoint entry."""
    lines = DECISIONS.read_text().splitlines()
    header = lines[0].split("\t")
    model_of = {}
    for line in lines[1:]:
        fields = dict(zip(header, line.split("\t")))
        model_of.setdefault(fields.get("checkpoint"), fields.get("model"))
    rows = []
    for entry in table["entries"]:
        if "fingerprint" not in entry or entry["evidence"].get("decision_row"):
            continue
        model = model_of.get(entry["checkpoint"])
        if model is None:
            raise SystemExit(f"no ledger model key for {entry['checkpoint']}; add one by hand")
        selected = {"selected_unqualified": "yes_default",
                    "kept_current_default": "no_kept_current_default",
                    "pending": "no_pending"}[entry["status"]]
        score = entry["evidence"]["scores"].get(entry["default"].get("name", ""), {})
        drafter = entry["default"].get("drafter")
        route = entry["default"]["kind"] + (
            f" drafter={drafter['name']}@sha256:{drafter['identities'][0]}" if isinstance(drafter, dict) else "")
        reason = (f"best-route default (Pierre 2026-10-21 decision; selected unqualified). "
                  f"{entry['reason']}. Route: {route}. Cells tok/s by route: "
                  + json.dumps(entry["evidence"]["cells_tok_s"], sort_keys=True))
        if score:
            reason += f" Geomean {score.get('geomean')}x, min cell {score.get('min_cell')}x."
        row = {
            "model": model, "checkpoint": entry["checkpoint"], "scope": "feature",
            "feature_or_flag": "best_route_default", "implemented": "yes", "qualified": "no",
            "selected": selected, "observed_used": "no",
            "decision": "selected_unqualified" if selected == "yes_default" else entry["status"],
            "reason": reason.replace("\t", " ").replace("\n", " "),
            "evidence": ";".join(entry["evidence"]["files"] + ["src/mlx2/best_route_defaults.json"]),
            "source_revision": source_revision,
            "artifact_profile": ("best-route-defaults-20261021; M5 Max 128GB, MLX 0.32.4; frozen tree "
                                 f"{entry['evidence']['source_revision'][:9]}; h2h greedy 256 tokens, "
                                 "1 warmup + 2 reps, cold prefixes, B1 decode / B4 lane-sum decode; prompts "
                                 "code chat spomin7k rewrite"),
        }
        rows.append("\t".join(row.get(name, "") for name in header))
    return rows


def route_text(spec: dict) -> str:
    kind = spec["kind"]
    if kind == "adapter_default":
        return "adapter default (unchanged)"
    drafter = spec.get("drafter")
    if isinstance(drafter, dict):
        return f"`--ast`, drafter {drafter['name']}@sha256:{drafter['identities'][0][:12]}"
    return {"ast": "`--ast`", "native_mtp": "`--native-mtp`", "prompt_lookup": "`--prompt-lookup`",
            "ordinary": "`--ordinary`"}[kind] + (f" drafter={drafter}" if drafter else "")


def docs_table(table: dict) -> str:
    rows = ["| checkpoint / family | status | default route | geomean / min cell vs plain decode | evidence (commit) |",
            "|---|---|---|---|---|"]
    for entry in table["entries"]:
        name = entry.get("checkpoint") or f"family `{entry['family']}`"
        if "@sha256:" in name:
            label, fingerprint = name.split("@sha256:", 1)
            name = f"{label}@sha256:{fingerprint[:12]}"
        scores = entry["evidence"].get("scores") or {}
        chosen = entry["default"].get("name") if entry["status"] == "selected_unqualified" else "default"
        score = scores.get(chosen) or {}
        numbers = (f"{score['geomean']:.2f}x / {score['min_cell']:.2f}x ({score['min_cell_name']})"
                   if score.get("geomean") else "see reason")
        commit = entry["evidence"]["source_revision"]
        commit = ";".join(part[:9] for part in commit.split(";"))
        rows.append(f"| {name} | {entry['status']} | {route_text(entry['default'])} | {numbers} | "
                    f"{len(entry['evidence']['files'])} files ({commit}) |")
    return "\n".join(rows)


def fill_marker(path: Path, marker: str, body: str) -> None:
    text = path.read_text()
    start, end = f"<!-- {marker}:start -->", f"<!-- {marker}:end -->"
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n{body}\n{end}{tail}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the table is stale")
    parser.add_argument("--decision-rows", metavar="COMMIT",
                        help="append best_route_default decision rows bound to COMMIT")
    args = parser.parse_args(argv)
    if args.decision_rows:
        from mlx2.best_route_defaults import load_table

        rows = decision_rows(load_table(), args.decision_rows)
        with DECISIONS.open("a") as stream:
            for row in rows:
                stream.write(row + "\n")
        print(f"appended {len(rows)} rows")
        return 0
    entries, summaries = [], []
    for slug, arms in arm_files().items():
        entry, summary = build_entry(slug, arms)
        summaries.append(summary)
        if entry is not None:
            entries.append(entry)
    imported, unimported = lane_rows()
    entries = merge_lane_entries(entries, imported, unimported)
    for entry in STATIC_ENTRIES:
        validate_entry(entry)
    table = {
        "schema": SCHEMA,
        "decision": (
            "Pierre 2026-10-21: build the per-artifact best-route defaults now (auto-pair "
            "verified on-disk drafters, --ast where it wins), labelled unqualified in "
            "receipts; queue qualification afterwards"
        ),
        "rule": dict(RULE, prompts=list(RULE["prompts"]), widths=list(RULE["widths"])),
        "campaign": "qualification/runs/best-route-defaults-20261021",
        "entries": (sorted(entries, key=lambda e: e["checkpoint"]) + family_entries(entries)
                    + STATIC_ENTRIES),
    }
    text = json.dumps(table, indent=1, sort_keys=False) + "\n"
    if args.check:
        if TABLE_PATH.read_text() != text:
            print("best_route_defaults.json is stale; rerun scripts/build_best_route_defaults.py")
            return 1
        return 0
    TABLE_PATH.write_text(text)
    fill_marker(ROOT / "docs/SERVING.md", "best-route-table", docs_table(table))
    selected = [e for e in table["entries"] if e["status"] == "selected_unqualified"]
    plan = ["| checkpoint | model path (evidence) | explicit equivalent | paired drafter | evidence commit |",
            "|---|---|---|---|---|"]
    for e in selected:
        files = e["evidence"]["files"]
        model_path = next((load_receipt(ROOT / f).get("meta", {}).get("model_path")
                           for f in files if f.endswith(".json.gz") and (ROOT / f).exists()), "?")
        drafter = e["default"].get("drafter")
        flags = {"ast": "--ast --ast-drafter <paired path>" if isinstance(drafter, dict) else "--ast",
                 "native_mtp": "--native-mtp", "prompt_lookup": "--prompt-lookup",
                 "ordinary": "--ordinary"}[e["default"]["kind"]]
        pin = (f"{drafter['name']}@sha256:{drafter['identities'][0]}" if isinstance(drafter, dict) else "none")
        if "family" in e:
            model_path = "each unmeasured checkpoint of the class (qualify per checkpoint)"
        plan.append(f"| {e.get('checkpoint') or 'family ' + e['family']} | `{model_path}` | `{flags}` | {pin} | "
                    f"{e['evidence']['source_revision'][:12]} |")
    fill_marker(ROOT / "qualification/plans/best-route-defaults-qualification-20261021.md", "plan-rows",
                "\n".join(plan) if selected else "No checkpoint selected a new default.")
    ast_rows = [e for e in table["entries"] if e["status"] == "selected_unqualified" and e["default"]["kind"] == "ast"]
    fill_marker(ROOT / "docs/AST.md", "best-route-ast",
                ("Checkpoints whose no-flag default is currently `--ast` (generated):\n\n"
                 + "\n".join(f"- {e['checkpoint'].split('@sha256:')[0]}@sha256:{e['fingerprint'][:12]}: "
                              f"{route_text(e['default'])}" if "checkpoint" in e else
                              f"- family `{e['family']}` (checkpoints without their own entry): "
                              f"{route_text(e['default'])}" for e in ast_rows))
                if ast_rows else "No checkpoint selects `--ast` as its no-flag default yet.")
    (RUN / "summary.json").write_text(json.dumps(summaries, indent=1) + "\n")
    lines = [
        "| checkpoint | current default | winner | route geomean / min cell (min cell name) |",
        "|---|---|---|---|",
    ]
    for summary in summaries:
        if "scores" not in summary:
            lines.append(f"| {summary['slug']} | incomplete: {summary.get('arms')} | | |")
            continue
        scores = "; ".join(
            f"{name} {s['geomean']} / {s['min_cell']} ({s.get('min_cell_name')})" if s["geomean"]
            else f"{name} missing {len(s['missing'])} cells"
            for name, s in summary["scores"].items()
        )
        lines.append(f"| {summary['checkpoint'][:90]} | {summary['current_default_route']} | "
                     f"{summary['winner']} | {scores} |")
    for item in imported:
        lines.append(f"| {item['checkpoint'][:90]} | (decision row) | {item['default']['kind']} | "
                     f"imported from {item['evidence']['source_revision'][:9]} |")
    for item in unimported:
        lines.append(f"| {item['checkpoint'][:90]} | (decision row) | not imported | {item['why']} |")
    (RUN / "verdicts.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
