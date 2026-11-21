#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Pierre-Lamy-Calculator-NC-ND-1.0
# Copyright (c) 2026 Pierre Lamy
# Scope and terms: provenance/LICENSE-calculator-scope.json; provenance/LICENSE-calculator.txt
"""Derive a calculator manifest that differs from its source only in the trigger (unqualified).

``--parser v1|v2|v4|marker`` rewrites every calculator tile's parser. ``v1``/``v2`` keep the source's
probe gate; ``v4`` keeps the source tile as it is and requires it to be v4 already (a copy, so each
run owns an immutable artifact); ``marker`` replaces the probe gate with the ``explicit_marker`` gate
(the probe tensor stays in the copied file, unused) and drops device prefilters. Tensors are copied
byte for byte and the derived manifest records the source manifest's digest. Refuses a nonempty
output directory.

  python scripts/research/calc_trigger_manifests.py --source M/manifest.json --parser marker --out DIR
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mlx2.runtime import amoeba_det_tiles as D  # noqa: E402

PARSERS = {"v1": D.PARSER_ID, "v2": D.PARSER_V2, "v4": D.PARSER_V4, "marker": D.PARSER_MARKER}


def derive(source: Path, parser: str, out: Path) -> Path:
    src = D.load_manifest(source)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty {out}")
    out.mkdir(parents=True, exist_ok=True)
    for name in src["files"]:
        shutil.copy2(Path(src["_dir"]) / name, out / name)
    m = {k: v for k, v in src.items() if not k.startswith("_")}
    m["derived_from"] = {"manifest_sha256": src["_digest"], "path": str(Path(src["_dir"]) / "manifest.json"),
                         "mlx2_source_revision": src.get("mlx2_source_revision"), "trigger": parser}
    m["mlx2_source_revision"] = subprocess.check_output(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT,
                                                        text=True).strip()
    for t in m["tiles"]:
        if t["kind"] != "calculator":
            continue
        if parser == "v4":
            if t.get("parser") != D.PARSER_V4:
                raise SystemExit("--parser v4 copies a v4 tile; the source tile is " + str(t.get("parser")))
            continue
        for key in ("trigger_prefilter", "complete_literal_requests", "context_tokens", "arithmetic_backend"):
            t.pop(key, None)                    # the v1/v2/marker paths: CPU arithmetic, host text only
        t["parser"] = PARSERS[parser]
        if parser == "marker":
            t["gate"] = {"kind": D.MARKER_GATE, "unused_probe": t.get("gate")}
    (out / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    D.load_manifest(out)                        # the derived manifest must validate
    return out / "manifest.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--parser", choices=sorted(PARSERS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    print(derive(a.source, a.parser, a.out))


if __name__ == "__main__":
    main()
