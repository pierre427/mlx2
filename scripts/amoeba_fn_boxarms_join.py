#!/usr/bin/env python3
"""Join the continuation-ablation arms with the pilot (CPU): per box and arm (text / greedy / full = pilot), kept units,
pooled held dCE and the +0.05 verdict, held text / greedy / T=0.7, val, and the identity checks (same held and val
files, same source revision, identities differing only in corpus / n_mean / n_fit, T2-alone identical).
Usage: amoeba_fn_boxarms_join.py OUT_DIR UNITS box1,box2,..."""
import json
import sys
from pathlib import Path


def main():
    out, units, boxes = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
    arms = {"text": out / f"carves-{units}-arm-text", "greedy": out / f"carves-{units}-arm-greedy",
            "full": out / f"carves-{units}"}
    f = lambda x: "—" if x is None else f"{x:.3f}"   # noqa: E731
    print(f"Continuation ablation ({units} units): same held / val rows; probe-text mean/fit documents held at the pilot's.\n")
    print("| box | arm | kept units | held pooled | +0.05 | held text / greedy / t0.7 | val | T2 alone | checks |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b in boxes:
        ref = None
        for arm, d in arms.items():
            v = {}
            for x in (d / "verdicts.jsonl").read_text().split("\n") if (d / "verdicts.jsonl").exists() else []:
                if x.strip():
                    r = json.loads(x)
                    if r["box"] == b:
                        v = r
            try:
                c = json.loads((d / f"box-{b}" / "carve.json").read_text())
            except FileNotFoundError:
                print(f"| {b} | {arm} | missing | | | | | | |")
                continue
            ident = dict(c.get("identity") or {})
            key = (c["corpus"].get("held_sha256_12"), c["corpus"].get("val_sha256_12"), c.get("mlx2_source_revision"))
            checks = "ok"
            if ref is None:
                ref = (key, {k: v_ for k, v_ in ident.items() if k not in ("corpus", "n_mean", "n_fit")})
            else:
                if key != ref[0]:
                    checks = "held/val/rev DIFFER"
                elif {k: v_ for k, v_ in ident.items() if k not in ("corpus", "n_mean", "n_fit")} != ref[1]:
                    checks = "identity DIFFERS"
            hs = v.get("held_split") or {}
            g = {k.rsplit(":", 1)[-1]: x for k, x in hs.items() if k.startswith("gen:")}
            print(f"| {b} | {arm} | {v.get('kept_units', '—')} | {f(v.get('export_held'))} | "
                  f"{'meets' if v.get('held_meets_0.05') else 'misses'} | {f(hs.get('box'))} / {f(g.get('greedy'))} / "
                  f"{f(g.get('t0.7'))} | {f(v.get('export_val'))} | {f(v.get('t2_alone_held'))} | {checks} |")


if __name__ == "__main__":
    main()
