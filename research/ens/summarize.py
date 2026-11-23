"""Tabulate A0 eval receipts under an evidence directory.

  python -m research.ens.summarize <evidence dir> [--json]

One row per eval receipt: core checkpoint, arm, family, module kind, on, the bar on must beat
(max of bypass and z-blind), the margin, both, the codebook ceiling and ceiling_ok. Receipts from code
before the z-blind baseline (pilot 3 and earlier) show zblind as missing, and the bar is then bypass only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(root: Path):
    for p in sorted(root.rglob("receipt.json")):
        r = json.loads(p.read_text())
        if r.get("schema") != "ens.a0.eval_receipt.v1":
            continue
        x = r["results"]
        bar = max(x.get("bypass", 0.0), x.get("zblind") or 0.0)
        yield {"dir": p.parent.name, "core": r["core_checkpoint"], "arm": r["arm"], "family": r["family"],
               "kind": r.get("module_kind"), "quant": r.get("quant", "none"), "on": x.get("on"), "bypass": x.get("bypass"), "zblind": x.get("zblind"),
               "bar": bar, "margin": None if x.get("on") is None else x["on"] - bar, "both": x.get("both"),
               "ceiling": x.get("oracle_codebook", x.get("oracle_typed")), "ceiling_ok": x.get("ceiling_ok"),
               "n": x.get("n"), "n_ops": x.get("n_ops"), "source_revision": r.get("source_revision")}


def _f(v):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rs = list(rows(Path(a.root)))
    if a.json:
        print(json.dumps(rs, indent=1))
        return
    cols = ["dir", "family", "kind", "quant", "on", "bar", "margin", "both", "ceiling", "ceiling_ok", "n_ops"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rs:
        print("| " + " | ".join(_f(r[c]) for c in cols) + " |")


if __name__ == "__main__":
    main()
