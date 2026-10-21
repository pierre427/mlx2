#!/usr/bin/env python3
"""Turn model-matrix cell receipts into model-feature-decisions.tsv rows.

One row per (checkpoint, server flag): ``--ordinary``, ``--prompt-lookup``
(default and advanced PLD cells summarized together), ``--native-mtp`` and
``--external-draft`` (every drafter cell summarized together), plus a
``tools`` feature row when a tool-call request ran.  These are smoke results:
``qualified`` is always ``no``; ``observed_used`` is ``yes`` only when the
route receipt showed the route serving and every request was sane.

Rows whose (model, checkpoint, scope, feature) key already exists in the TSV
are not written (existing reviewed decisions are never overwritten); they are
listed on stderr instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_matrix import effective_verdict  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TSV = ROOT / "docs/research/model-feature-decisions.tsv"
COLUMNS = (
    "model", "checkpoint", "scope", "feature_or_flag", "implemented", "qualified",
    "selected", "observed_used", "decision", "reason", "evidence",
    "source_revision", "artifact_profile",
)
FAMILY_KEYS = {
    "StandardDecoderAdapter": "qwen3:base",
    "Qwen359BAdapter": "qwen3_5:9b",
    "Qwen3827BAdapter": "qwen3_5:27b",
    "Qwen3627BAdapter": "qwen3_5:27b",
    "Qwen3635BA3BAdapter": "qwen3_5_moe:35b_a3b",
    "GptOssAdapter": "gpt_oss:base",
    "MiniCPMOAdapter": "minicpmo:base",
    "Mellum21ThinkingAdapter": "mellum:base",
    "XingAdapter": "xing4_0:base",
    "LFM25MoEAdapter": "lfm2_moe:base",
    "Agnes3FlashAdapter": "agnes:base",
    "Gemma4A4BAdapter": "gemma4:a4b",
    "Gemma431BAdapter": "gemma4:31b",
    "NorthMiniCodeAdapter": "cohere2_moe:base",
    "Nemotron35LightningAdapter": "nemotron_h:lightning",
    "Nemotron3SuperAdapter": "nemotron_h:super",
    "Nemotron35SuperAdapter": "nemotron_h_omni:base",
    "Nemotron35BoltAdapter": "nemotron_h_omni:base",
    "MuseGlimmerAdapter": "muse_glimmer:base",
    "LagunaXS21Adapter": "laguna:xs21",
    "LagunaS21Adapter": "laguna:s21",
    "FlashNextAdapter": "qwen4_exp:flash-next",
    "HYV3Adapter": "hy_v3:base",
}
FLAG_FOR_ROUTE = {
    "ordinary": "--ordinary",
    "pld_default": "--prompt-lookup",
    "pld_advanced": "--prompt-lookup",
    "native_mtp": "--native-mtp",
}


def flag_for(route: str) -> tuple[str, str]:
    """(scope, feature_or_flag) for a cell route name (suffixes ignored)."""
    if route.startswith("no_flag_default"):
        return "feature", "prompt_lookup"
    for prefix, flag in (("ordinary", "--ordinary"), ("pld_", "--prompt-lookup"),
                         ("native_mtp", "--native-mtp"), ("ast_", "--ast"),
                         ("external_", "--external-draft")):
        if route.startswith(prefix):
            return "server_cli", flag
    raise ValueError(f"unmapped route {route!r}")


def current_cells(cells: list) -> tuple[list, list]:
    """Cells from the newest source revision decide; older ones are history."""
    def stamp(item):
        return item[1].get("finished_at") or ""
    # A rerun of the same route (evidence dir suffix -r2/-fix-*) supersedes it.
    latest_by_route = {}
    for item in sorted(cells, key=stamp):
        latest_by_route[item[1]["route"]] = item
    ordered = sorted(latest_by_route.values(), key=stamp)
    latest = ordered[-1][1].get("source_revision")
    current = [item for item in ordered if item[1].get("source_revision") == latest]
    keep = {id(item) for item in current}
    return current, [item for item in sorted(cells, key=stamp) if id(item) not in keep]


def cell_summary(cell: dict, path: Path) -> str:
    reqs = cell.get("requests") or []
    tps = [r["decode_tokens_per_second"] for r in reqs if r.get("decode_tokens_per_second")]
    accepted = proposed = 0
    for r in reqs:
        spec = (r.get("mlx2") or {}).get("speculation") or (r.get("mlx2") or {}).get("mtp") or {}
        if isinstance(spec, dict):
            accepted += spec.get("accepted") or 0
            proposed += spec.get("proposed") or 0
    oracle = [r["functional_oracle"]["all_found"] for r in reqs if r.get("functional_oracle")]
    parts = [f"{cell['route']}={cell.get('verdict')}"]
    if cell["route"].startswith("ast_") and cell.get("routes_seen"):
        parts.append("--ast served " + "/".join(cell["routes_seen"]))
    if cell.get("draft"):
        parts.append(f"draft {cell['draft']}")
    if cell.get("load_seconds") is not None:
        parts.append(f"load {cell['load_seconds']}s")
    if tps:
        parts.append(f"decode {min(tps):.1f}-{max(tps):.1f} tok/s")
    if proposed:
        parts.append(f"accepted {accepted}/{proposed}")
    if oracle:
        parts.append(f"spomin4K oracle {'pass' if all(oracle) else 'miss'}")
    if cell.get("peak_phys_footprint_bytes"):
        parts.append(f"peak {cell['peak_phys_footprint_bytes'] / 2**30:.1f}GiB")
    failures = cell.get("failures") or []
    if failures:
        parts.append("failures: " + " | ".join(str(f)[:160] for f in failures[:3]))
    if cell.get("reason"):
        parts.append("skipped: " + str(cell["reason"])[:160])
    return "; ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--profile", required=True, help="artifact_profile text (host, MLX, serving shape)")
    parser.add_argument("--host", help="only cells from this host label")
    parser.add_argument("--write", action="store_true", help="append to the TSV (default: print)")
    args = parser.parse_args(argv)
    groups: dict[tuple, list] = defaultdict(list)
    for path in sorted(args.evidence_root.resolve().glob("*/*/cell.json")):
        cell = json.loads(path.read_text())
        cell["verdict"] = effective_verdict(cell)
        if cell["verdict"] == "artifact_offline":
            # Offline artifacts get no decision row (coordinator, 2026-10-21).
            continue
        if args.host and cell.get("host") != args.host:
            continue
        key = FAMILY_KEYS.get(cell.get("adapter") or "")
        if not key:
            print(f"no family key for {cell.get('adapter')}: {path}", file=sys.stderr)
            continue
        scope, flag = flag_for(cell["route"])
        groups[(key, cell["checkpoint"], scope, flag)].append((path, cell))
        if any(r.get("kind") == "tool" for r in cell.get("requests") or ()):
            groups[(key, cell["checkpoint"], "feature", "tools")].append((path, cell))
    with TSV.open(newline="") as handle:
        existing = {(r["model"], r["checkpoint"], r["scope"], r["feature_or_flag"])
                    for r in csv.DictReader(handle, delimiter="\t")}
    rows = []
    for (model, checkpoint, scope, flag), all_cells in sorted(groups.items()):
        if (model, checkpoint, scope, flag) in existing:
            print(f"existing decision kept, not overwritten: {model} {checkpoint} {flag}", file=sys.stderr)
            continue
        cells, history = current_cells(all_cells)
        if flag == "tools":
            tool_ok = [all((r.get("sanity") or {}).get("checks", {}).get("tool_called") for r in c.get("requests") or ()
                           if r.get("kind") == "tool") for _, c in cells]
            verdicts = ["pass" if ok else "fail" for ok in tool_ok]
            summary = "; ".join(
                f"{c['route']}: tool call {'emitted and parsed' if ok else 'missing or unparsed'}"
                for (_, c), ok in zip(cells, tool_ok))
        else:
            verdicts = [c.get("verdict") for _, c in cells]
            summary = " || ".join(cell_summary(c, p) for p, c in cells)
        if history:
            summary += " || earlier or superseded cells: " + "; ".join(
                f"{c['route']}={c.get('verdict')}@{str(c.get('source_revision'))[:10]}" for _, c in history)
        cells = all_cells
        passed = [v in {"pass", "pass_with_warnings"} for v in verdicts]
        fail_closed = all(v == "fail_closed_expected" for v in verdicts)
        skipped = all(v == "skipped" for v in verdicts)
        if fail_closed:
            decision, observed = "smoke_fail_closed_capability_not_declared", "no"
        elif skipped:
            decision, observed = "smoke_skipped", "no"
        elif all(passed):
            decision, observed = "smoke_pass", "yes"
        elif any(passed):
            decision, observed = "smoke_partial", "yes"
        else:
            decision, observed = "smoke_fail", "no"
        revisions = sorted({c.get("source_revision") or "" for _, c in cells} - {""})
        evidence = ";".join(str(p.relative_to(ROOT)) for p, _ in cells)
        rows.append({
            "model": model,
            "checkpoint": checkpoint,
            "scope": scope,
            "feature_or_flag": flag,
            "implemented": "no" if fail_closed else "yes",
            "qualified": "no",
            "selected": "no" if fail_closed else "yes_explicit",
            "observed_used": observed,
            "decision": decision,
            "reason": ("model-matrix-20261021 smoke (code, explain, multi-turn, frozen-Spomin 4K, "
                       "default sampling, tool call when declared; temperature 0, max_tokens 200). "
                       "Not qualification. " + summary).replace("\t", " ").replace("\n", " "),
            "evidence": evidence,
            "source_revision": ";".join(revisions) or "not_recorded",
            "artifact_profile": args.profile + "; host " + ",".join(sorted({c.get('host') or '?' for _, c in cells})),
        })
    if args.write:
        with TSV.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
            for row in rows:
                writer.writerow(row)
        print(f"appended {len(rows)} rows to {TSV}")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        for row in rows:
            writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
