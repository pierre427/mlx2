#!/usr/bin/env python3
"""Train and evaluate a self-MTP draft-confidence head from acceptance logs (CPU).

Input: one or more JSONL files written by ``--mtp-acceptance-log`` (schema
``mlx2.mtp_acceptance_log.v1``).  Rows are split by uid parity-hash into train
and holdout sets, so one request never lands on both sides.

Output:

* ``--out``: the logistic model, calibrated per position by STS;
* ``--out-top1``: the zero-training top-1 baseline with STS temperatures only;
* ``--report``: holdout ECE/AUC per position for raw top-1, top-1+STS and
  logistic, plus the expected-accepted-length bias and MAE.

Pure numpy; no MLX or GPU use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx2.runtime.mtp_confidence import (  # noqa: E402
    Top1ProbConfidence,
    evaluate_confidence,
    fit_logistic_confidence,
    fit_sequential_temperatures,
    load_acceptance_log,
)


def _split(keyed_rows, holdout: float, salt: str):
    train, test = [], []
    for key, row in keyed_rows:
        digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
        (test if digest[0] / 256.0 < holdout else train).append(row)
    return train, test


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, nargs="+", required=True)
    parser.add_argument("--positions", type=int, default=0, help="default: max drafted positions in the log")
    parser.add_argument("--buckets", type=int, default=1024)
    parser.add_argument("--l2", type=float, default=1e-2)
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--out-top1", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if not 0.0 < args.holdout < 1.0:
        parser.error("--holdout must be in (0, 1)")
    keyed = []
    for path in args.log:
        # uids are per-server; key by file so the split stays per-request.
        keyed.extend(
            (f"{path}:{row.uid}", row)
            for row in load_acceptance_log(path)
            if row.verify_depth > 0 and row.features
        )
    rows = [row for _, row in keyed]
    if not rows:
        parser.error("no drafted rows in the acceptance log(s)")
    positions = args.positions or max(len(row.features) for row in rows)
    verify_positions = max(row.verify_depth for row in rows)
    train, test = _split(keyed, args.holdout, "rm10")
    if not train or not test:
        parser.error("train/holdout split is empty; collect more requests")
    top1_sts = Top1ProbConfidence(
        temperatures=fit_sequential_temperatures(Top1ProbConfidence(), train, verify_positions)
    )
    logistic = fit_logistic_confidence(
        train, positions=positions, buckets=args.buckets, l2=args.l2
    )
    report = {
        "rows": {"train": len(train), "holdout": len(test)},
        "positions": positions,
        "verify_positions": verify_positions,
        "holdout": {
            "top1_raw": evaluate_confidence(Top1ProbConfidence(), test, verify_positions),
            "top1_sts": evaluate_confidence(top1_sts, test, verify_positions),
            "logistic": evaluate_confidence(logistic, test, verify_positions),
        },
    }
    args.out.write_text(json.dumps(logistic.to_dict(), indent=1))
    if args.out_top1:
        args.out_top1.write_text(json.dumps(top1_sts.to_dict(), indent=1))
    if args.report:
        args.report.write_text(json.dumps(report, indent=1, default=float))
    for name, item in report["holdout"].items():
        ece = [None if p["ece"] is None else round(p["ece"], 3) for p in item["positions"]]
        auc = [None if p["auc"] is None else round(p["auc"], 3) for p in item["positions"]]
        print(f"{name:9s} ece={ece} auc={auc} len_bias={item['expected_length_bias']:.3f} len_mae={item['expected_length_mae']:.3f}")
    return report


if __name__ == "__main__":
    main()
