#!/usr/bin/env python3
"""CPU-only comparison of D3 judge variants on saved judgecheck rows.

Ground truth: harmless answers = comply, unmodified-model refusals = refuse,
probe answers = blind labels (C complies / R refuses / D redirects) made
without seeing any judge output.  No GPU window: runs on the MLX CPU device.

Usage: residual_ablation_judge_eval.py JUDGECHECK_DIR LABELS.json D3_PATH OUT.json
"""

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    import residual_ablation as ra

    judgecheck, labels_path, d3_path, out = map(Path, sys.argv[1:5])
    rows = json.loads((judgecheck / "judge_rows.json").read_text())
    labels = {row["id"]: row for row in json.loads(labels_path.read_text())}
    saved = json.loads((judgecheck / "judge_exemplars.json").read_text())
    truth = {
        "harmless": ["C"] * len(rows["harmless"]),
        "baseline_refusing": ["R"] * len(rows["baseline_refusing"]),
        "probe_measured": [labels[i]["label"] for i in range(len(rows["probe_measured"]))],
    }
    base = ra.D3Judge(d3_path)
    variants = {
        "two": ra.D3Judge(engine=base.engine, options="two"),
        "two_x": ra.D3Judge(engine=base.engine, options="two", exemplars=saved["hard"] + saved["soft"]),
        "three": ra.D3Judge(engine=base.engine, options="three"),
        "three_x": ra.D3Judge(engine=base.engine, options="three", exemplars=saved["hard"] + saved["soft"]),
    }
    report = {"labels": str(labels_path), "exemplars": saved,
              "truth_counts": {k: {c: v.count(c) for c in "CRD"} for k, v in truth.items()},
              "variants": {}}
    for name, judge in variants.items():
        stats = {}
        t0 = time.perf_counter()
        for split, split_rows in rows.items():
            judged = judge.judge(copy.deepcopy(split_rows), force=True)
            said = ["C" if r["judge"] == "A" else "X" for r in judged]
            gold = truth[split]
            stats[split] = {
                "accuracy": sum((s == "C") == (g == "C") for s, g in zip(said, gold)) / len(gold),
                # judge says comply where the truth is a refusal/redirect:
                # refusals the search would NOT see (undercount).
                "false_comply": sum(s == "C" and g != "C" for s, g in zip(said, gold)),
                "false_refuse": sum(s != "C" and g == "C" for s, g in zip(said, gold)),
                "redirects_caught": sum(s != "C" and g == "D" for s, g in zip(said, gold)),
                "redirects_total": gold.count("D"),
            }
            if split == "probe_measured":
                stats[split]["misses"] = [
                    {"id": i, "gold": g, "label": r.get("judge_label"), "margin": r["margin"]}
                    for i, (r, s, g) in enumerate(zip(judged, said, gold)) if (s == "C") != (g == "C")
                ]
        stats["seconds"] = time.perf_counter() - t0
        report["variants"][name] = stats
        p = stats["probe_measured"]
        print(f"{name:8s} probe acc={p['accuracy']:.2f} false_comply={p['false_comply']} "
              f"false_refuse={p['false_refuse']} redirects {p['redirects_caught']}/{p['redirects_total']} | "
              f"harmless acc={stats['harmless']['accuracy']:.2f} baseline acc="
              f"{stats['baseline_refusing']['accuracy']:.2f} | {stats['seconds']:.0f}s", flush=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
