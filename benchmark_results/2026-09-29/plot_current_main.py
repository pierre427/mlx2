#!/usr/bin/env python3
"""Plot source-bound M5 single-prompt ladders from local private receipts.

Example: python plot_current_main.py --mlx2 mlx2.json --tensorfold tf.json
    --omlx omlx.json --flash flash.json --output current-main-ladders.png
The input receipts stay local; the output contains aggregate measurements only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def series(receipt: dict, *, flash: bool, metric: str):
    points = []
    for cell in receipt["cells"]:
        runs = cell.get("runs", [])
        if len(runs) != 3:
            continue
        values = []
        for run in runs:
            value = (run.get("summary", {}).get(metric) if flash else run.get(metric))
            if not isinstance(value, (int, float)):
                break
            values.append(float(value))
        if len(values) == 3:
            points.append((cell.get("effective_target_tokens", cell["requested_tokens"] if flash
                           else cell["target_tokens"]) / 1024,
                           statistics.median(values), min(values), max(values)))
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlx2", "tensorfold", "omlx", "flash", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    dense = [("mlx2 27B", load(args.mlx2), "#1965b0"),
             ("TensorFold 27B", load(args.tensorfold), "#e28b1d"),
             ("omlx 27B", load(args.omlx), "#23875b")]
    flash = load(args.flash)
    config_hashes = {item["artifact_config_sha256"] for _, item, _ in dense}
    harness_hashes = {item["harness_sha256"] for _, item, _ in dense}
    if len(config_hashes) != 1 or len(harness_hashes) != 1:
        raise ValueError("dense receipts must use the same checkpoint config and runner")
    if any(item["status"] != "passed" for _, item, _ in dense):
        raise ValueError("one of the dense ladder receipts did not pass")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})
    for ax, metric, ylabel in zip(
        axes, ("ttft_seconds", "decode_tokens_per_second"),
        ("Time to first token (seconds)", "Decode (tokens/second)"),
    ):
        for label, receipt, color in dense:
            points = series(receipt, flash=False, metric=metric)
            x, median, low, high = zip(*points)
            ax.plot(x, median, marker="o", linewidth=2.2, markersize=6,
                    color=color, label=label)
            ax.fill_between(x, low, high, color=color, alpha=0.13)
        points = series(flash, flash=True, metric=metric)
        x, median, low, high = zip(*points)
        ax.plot(x, median, marker="D", linestyle="--", linewidth=2.3,
                markersize=5.6, color="#8247a9", label="mlx2 Flash-Next candidate")
        ax.fill_between(x, low, high, color="#8247a9", alpha=0.1)
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylabel(ylabel)
        ax.axvline(256, color="#9b9b9b", linewidth=1, linestyle=":")
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper left", ncol=2, frameon=False)
    axes[0].annotate("Flash 262K:\nadmission refused\n(no samples)",
                     xy=(256, 104), xytext=(155, 15),
                     arrowprops={"arrowstyle": "->", "color": "#666666"},
                     fontsize=9, color="#666666")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlim(0.85, 275)
    axes[1].set_xticks([1, 4, 16, 32, 64, 128, 256],
                       labels=["1K", "4K", "16K", "32K", "65K", "131K", "262K"])
    axes[1].set_xlabel("Prompt context rung (tokens; 32K dense rung uses 32,512)")
    fig.suptitle("M5 Max · thermally admitted single-prompt ladders · 3 reps per point",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.018,
             "Lines show medians; bands show 3-rep range. Solid lines share one 27B checkpoint and identical outputs. "
             "Dashed Flash-Next uses a different model and speculative candidate settings.",
             ha="center", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=[0.025, 0.055, 0.995, 0.95])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
