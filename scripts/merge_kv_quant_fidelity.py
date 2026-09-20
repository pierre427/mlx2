#!/usr/bin/env python3
"""Merge per-context KV-quantization fidelity bundles into one gated bundle.

Long-context sweeps are run as several GPU jobs (one context each, so no job
holds the shared GPU for more than ~40 minutes).  This joins those bundles
into the single ``mlx2.kv-quant-fidelity-bundle.v1`` that the qualifier's
``--kv-fidelity-report`` consumes, and re-evaluates every verdict on the
merged contexts.

It fails closed: every part must be a bundle from the same harness hash, the
same adapter fingerprint, operation revision, descriptor and corpus; a
context measured twice is refused.  CPU only; loads no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BINDING_KEYS = ("device", "adapter_fingerprint", "operation_revision", "descriptor",
                "corpus_sha256")


def merge(bundles, sources=None):
    from mlx2.runtime.kv_quant_fidelity import (
        BUNDLE_SCHEMA,
        REPORT_SCHEMA,
        evaluate_fidelity_report,
    )

    if not bundles:
        raise ValueError("nothing to merge")
    sources = list(sources or [None] * len(bundles))
    for index, bundle in enumerate(bundles):
        if bundle.get("schema") != BUNDLE_SCHEMA:
            raise ValueError(f"part {index} is not a fidelity bundle")
    harness = {json.dumps(b.get("harness"), sort_keys=True) for b in bundles}
    if len(harness) != 1:
        raise ValueError("parts were produced by different harness revisions")
    # A part may measure a subset of operations (a long context split per
    # operation); each operation merges the parts that measured it.
    names = set()
    for bundle in bundles:
        names |= set(bundle.get("reports") or {})
    merged_reports, verdicts = {}, {}
    for name in sorted(names):
        parts = [b["reports"][name] for b in bundles if name in (b.get("reports") or {})]
        head = parts[0]
        for part in parts:
            if part.get("schema") != REPORT_SCHEMA or part.get("operation") != name:
                raise ValueError(f"{name}: part is not a report for this operation")
            for key in BINDING_KEYS:
                if part.get(key) != head.get(key):
                    raise ValueError(f"{name}: parts disagree on {key}")
        entries, seen = [], set()
        for part in parts:
            for entry in part.get("contexts") or ():
                context = int(entry.get("context", 0))
                if context in seen:
                    raise ValueError(f"{name}: context {context} measured twice")
                seen.add(context)
                entries.append(entry)
        report = dict(head)
        report["contexts"] = sorted(entries, key=lambda e: int(e.get("context", 0)))
        merged_reports[name] = report
        verdicts[name] = evaluate_fidelity_report(
            report, operation=name, adapter_fingerprint=head.get("adapter_fingerprint")
        )
    return {
        "schema": BUNDLE_SCHEMA,
        "harness": bundles[0].get("harness"),
        "merged_from": [
            {"source": src, "plan": b.get("plan"), "platform": b.get("platform"),
             "mlx_version": b.get("mlx_version")}
            for src, b in zip(sources, bundles)
        ],
        "plan": {"merged": True,
                 "contexts": sorted({int(e["context"]) for r in merged_reports.values()
                                     for e in r["contexts"]})},
        "platform": bundles[0].get("platform"),
        "mlx_version": bundles[0].get("mlx_version"),
        "reports": merged_reports,
        "verdicts": verdicts,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("parts", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    bundles = [json.loads(path.read_text()) for path in args.parts]
    merged = merge(bundles, [str(path) for path in args.parts])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2, default=str) + "\n")
    print(json.dumps({k: {"passed": v["passed"], "failures": v["failures"]}
                      for k, v in merged["verdicts"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
