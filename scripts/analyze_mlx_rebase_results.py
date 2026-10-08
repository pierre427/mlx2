#!/usr/bin/env python3
"""Reduce the MLX #4596/#4640/#4641 GPU evidence into one receipt."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


BIT_KEYS = (
    "last_hidden_sha256",
    "logits_sha256",
    "rows_sha256",
    "state_sha256",
)


def load(path: Path, name: str) -> dict:
    return json.loads((path / name).read_text())


def pct(candidate: float, baseline: float) -> float:
    return round(100 * (candidate / baseline - 1), 2)


def timing_summary(path: Path, prefix: str, section: str) -> list[dict]:
    baseline = [load(path, f"steady-baseline-r{i}.json") for i in (1, 2)]
    candidate = [load(path, f"steady-candidate-r{i}.json") for i in (1, 2)]
    rows = []
    for index, reference in enumerate(baseline[0][section]):
        b = [run[section][index]["timing"]["median_ms"] for run in baseline]
        c = [run[section][index]["timing"]["median_ms"] for run in candidate]
        rows.append(
            {
                "case": prefix,
                "mode": reference.get("mode"),
                "shape": reference["shape"],
                "baseline_medians_ms": b,
                "candidate_medians_ms": c,
                "aggregate_speedup": round(statistics.mean(b) / statistics.mean(c), 3),
                "output_hash_equal": reference["output_sha256_f32"]
                == candidate[0][section][index]["output_sha256_f32"],
            }
        )
    return rows


def bit_reference(path: Path, stem: str) -> dict:
    baseline = load(path, f"bitref-{stem}-baseline.json")
    candidate = load(path, f"bitref-{stem}-candidate.json")
    out = {}
    for arm in ("on", "off"):
        before = baseline["arms"][arm]["512"]
        after = candidate["arms"][arm]["512"]
        out[arm] = {
            "hashes_equal": {key: before[key] == after[key] for key in BIT_KEYS},
            "tokens_equal": before["tokens"] == after["tokens"],
        }
    return out


def ttft_pair(path: Path, baseline_name: str, candidate_name: str) -> dict:
    baseline = load(path, baseline_name)["summary"]
    candidate = load(path, candidate_name)["summary"]
    out = {"baseline": baseline, "candidate": candidate, "candidate_vs_baseline_pct": {}}
    for arm in ("stock", "lane"):
        out["candidate_vs_baseline_pct"][arm] = {
            key: pct(candidate[arm][key], baseline[arm][key])
            for key in baseline[arm]
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = args.run_dir

    baseline = load(path, "steady-baseline-r1.json")
    candidate = load(path, "steady-candidate-r1.json")
    qmm = []
    for before, after in zip(baseline["qmm_splitk"], candidate["qmm_splitk"]):
        qmm.append(
            {
                "mode": before["mode"],
                "shape": before["shape"],
                "baseline_error_over_single_rounding": before["error_over_single_rounding"],
                "candidate_error_over_single_rounding": after["error_over_single_rounding"],
                "mean_abs_error_change_pct": pct(after["mean_abs_error"], before["mean_abs_error"]),
                "output_hash_equal": before["output_sha256_f32"]
                == after["output_sha256_f32"],
            }
        )

    fn = ttft_pair(path, "ttft-fn-baseline.json", "ttft-fn-candidate.json")
    q27 = ttft_pair(path, "ttft-27b-baseline.json", "ttft-27b-candidate.json")
    full_rerun_doc = load(path, "ttft-fn-candidate-rerun.json")
    ablation_doc = load(path, "ttft-fn-ablation.json")
    full_rerun = full_rerun_doc["summary"]
    ablation = ablation_doc["summary"]
    ablation_kernel = load(path, "steady-ablation.json")
    fn["full_candidate_rerun_vs_without_4640_pct"] = {
        arm: {key: pct(full_rerun[arm][key], ablation[arm][key]) for key in ablation[arm]}
        for arm in ("stock", "lane")
    }
    fn["without_4640"] = ablation
    fn["full_candidate_rerun"] = full_rerun
    full_reps = len([row for row in full_rerun_doc["runs"] if row["arm"] == "stock"])
    ablation_rows = [row for row in ablation_doc["runs"] if row["arm"] == "stock"]
    outlier = max(row["prefill_512_s"] for row in ablation_rows)
    fn["ablation_comparison_warning"] = (
        f"inconclusive separate-process comparison: full candidate rerun has {full_reps} "
        f"reps, ablation has {len(ablation_rows)} reps including a {outlier:.2f} s outlier, "
        "order was not counterbalanced, and observed between-run drift exceeds the apparent effect"
    )

    result = {
        "schema": "mlx2.mlx-rebase-gpu-summary.v1",
        "baseline_core": baseline["mlx"],
        "candidate_core": candidate["mlx"],
        "ablation_core": ablation_kernel["mlx"],
        "device": candidate["device"],
        "qmm_splitk_precision": qmm,
        "steady_kernel_timings": {
            "thin_n": timing_summary(path, "thin_n", "thin_n"),
            "sdpa_vector": timing_summary(path, "sdpa_vector", "sdpa_vector"),
            "ablation": {
                "mlx": ablation_kernel["mlx"],
                "thin_n": [
                    {
                        "shape": row["shape"],
                        "median_ms": row["timing"]["median_ms"],
                        "output_sha256_f32": row["output_sha256_f32"],
                    }
                    for row in ablation_kernel["thin_n"]
                ],
                "qmm_error_over_single_rounding": [
                    {
                        "mode": row["mode"],
                        "shape": row["shape"],
                        "value": row["error_over_single_rounding"],
                    }
                    for row in ablation_kernel["qmm_splitk"]
                ],
            },
            "warning": "separate-process sub-millisecond timings are power-state sensitive; no speedup or regression is established",
        },
        "bit_references": {
            "flash_next": bit_reference(path, "fn"),
            "qwen38_27b": bit_reference(path, "27b"),
        },
        "ttft": {"flash_next": fn, "qwen38_27b": q27},
        "captures": load(path, "captures.json"),
        "assessment": {
            "4596": "output-exact with no demonstrated regression; direct timing results were inconsistent and the in-process short-context decode did not establish a gain",
            "4640": "engaged and numerically safe, but has no demonstrated benefit on this M5 Max; direct timing and the separate-process ablation are inconclusive because drift exceeds the apparent effect",
            "4641": "helpful precision change: every exercised split-K case reached one-rounding error; the non-split affine control stayed unchanged",
            "selection": "carry #4596 on safety grounds and #4641 for precision; do not select #4640 on applegpu_g17s without a controlled demonstration of benefit",
        },
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
