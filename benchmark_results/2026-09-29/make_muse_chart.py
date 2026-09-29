#!/usr/bin/env python3
"""Plot the public Muse-Glimmer qualification tables as one figure.

Run from this directory with matplotlib available. The markdown report is the
single source for plotted values, so the chart can be regenerated after edits.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
REPORT = HERE / "muse.md"


def rows_in(section: str) -> list[list[str]]:
    text = REPORT.read_text()
    block = text.split(f"## {section}\n", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in block.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
        if cells[0].startswith("Host and route"):
            continue
        rows.append(cells)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "muse-performance.png")
    args = parser.parse_args()

    stress = rows_in("20×20 batching and APCv2")
    ladder = rows_in("Thermally controlled single-prompt performance")
    wide_ladder = rows_in("Thermally controlled four-stream performance")
    if not stress or not ladder:
        raise SystemExit("report tables are missing")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(15.5, 10.4), facecolor="#f7f9fc")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.16, 1], hspace=0.42, wspace=0.24,
                           left=0.18, right=0.96, top=0.88, bottom=0.18)
    ax_stress = fig.add_subplot(grid[0, :])
    ax_m5 = fig.add_subplot(grid[1, 0])
    ax_m3 = fig.add_subplot(grid[1, 1])
    for ax in (ax_stress, ax_m5, ax_m3):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y" if ax is not ax_stress else "x", color="#e7ecf2", zorder=0)

    palette = {"ordinary": "#1f77b4", "current": "#087e6b", "lane": "#e69f00",
               "dflash": "#ad5f9e", "pld": "#d35454", "prior": "#667b91",
               "wide": "#344c61"}
    stress_labels = []
    stress_values = []
    stress_colors = []
    stress_hatch = []
    for host, source, graded, errors, rate, apc, batching, swap in stress:
        label = host.replace("M5 ", "M5  ").replace("M3 ", "M3  ")
        stress_labels.append(label)
        stress_values.append(float(rate))
        key = "current" if "current default" in host else (
            "lane" if "lane auto" in host or "lane exact" in host else (
                "dflash" if "DFlash2" in host else ("pld" if "PLD" in host else "ordinary")))
        stress_colors.append(palette[key])
        stress_hatch.append("///" if batching == "not observed" else "")
        if graded != "400/400" or errors != "0" or swap != "0" or apc != "passed":
            raise SystemExit(f"unexpected stress quality gate: {host}")
    y = list(range(len(stress)))[::-1]
    bars = ax_stress.barh(y, stress_values, height=0.72, color=stress_colors,
                          edgecolor="#27313a", linewidth=0.45, zorder=2)
    for bar, hatch, value in zip(bars, stress_hatch, stress_values):
        bar.set_hatch(hatch)
        ax_stress.text(value + 0.8, bar.get_y() + bar.get_height()/2,
                       f"{value:g}", va="center", fontweight="bold", fontsize=10)
    ax_stress.set_yticks(y, stress_labels)
    ax_stress.set_xlim(0, max(stress_values) * 1.16)
    ax_stress.set_xlabel("Aggregate generated tokens/s, median of 20 mixed rounds")
    ax_stress.set_title("20×20 domain workload  ·  every row 400/400, APCv2 pass, zero HTTP errors and swap",
                        loc="left", fontsize=12, fontweight="bold", pad=11)
    ax_stress.text(1.0, 1.025, "Hatching = cross-request batching not observed",
                   transform=ax_stress.transAxes, ha="right", fontsize=9, color="#536575")

    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for host, source, tokens, ttft, decode, needles, swap in ladder:
        machine = host.split()[0]
        label = host[len(machine):].strip()
        if label == "ordinary, lane off":
            label = "ordinary earlier"
        elif label == "ordinary, current default":
            label = "ordinary final default"
        elif label == "ordinary, lane auto":
            label = "lane auto"
        elif label == "ordinary, prior long ladder":
            label = "prior long, width 1"
        elif label == "ordinary, completed max context":
            label = "max-context follow-up"
        grouped[(machine, label)].append((int(tokens.replace(",", "")), float(decode)))
        if needles != "6/6" or swap != "0":
            raise SystemExit(f"unexpected ladder quality gate: {host} {tokens}")
    for host, source, tokens, width, ttft, decode, needles, swap in wide_ladder:
        if not host.startswith("M5 ") or width != "4" or needles != "24/24" or swap != "0":
            raise SystemExit(f"unexpected wide ladder quality gate: {host} {tokens}")
        grouped[("M5", "prior long, width 4")].append((int(tokens.replace(",", "")), float(decode)))
    for ax, machine in ((ax_m5, "M5"), (ax_m3, "M3")):
        for (host, label), points in grouped.items():
            if host != machine:
                continue
            points.sort()
            key = "current" if "final" in label or "follow-up" in label else (
                "wide" if "width 4" in label else (
                    "prior" if "prior long" in label else (
                        "lane" if "lane" in label else (
                            "dflash" if "DFlash2" in label else ("pld" if "PLD" in label else "ordinary")))))
            ax.plot([x for x, _ in points], [value for _, value in points],
                    color=palette[key], marker="s" if label == "ordinary earlier" or "width 4" in label else "o",
                    linestyle="--" if label == "ordinary earlier" or "prior long" in label else "-",
                    markersize=6, linewidth=2.2, label=label,
                    zorder=2 if label == "ordinary earlier" else 3)
        ax.set_xscale("log", base=2)
        if machine == "M5":
            ax.set_xticks([1024, 4096, 16384, 65536, 131072], ["1K", "4K", "16K", "65K", "131K"])
            ax.set_xlim(800, 150000)
        else:
            ax.set_xticks([1024, 4096], ["1K", "4K"])
            ax.set_xlim(850, 5000)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Prompt context tokens")
        ax.set_ylabel("Single-stream decode tokens/s")
        ax.set_title(f"{machine} thermal ladder  ·  3 reps/cell, all needles pass, zero swap",
                     loc="left", fontsize=11, fontweight="bold", pad=10)
        ax.legend(loc="upper right", fontsize=7.5, frameon=False)

    fig.suptitle("Muse-Glimmer 30B (MLX 4-bit): qualification throughput",
                 x=0.18, ha="left", y=0.965, fontsize=20, fontweight="bold", color="#172a3a")
    fig.text(0.18, 0.918,
             "M5 Max 128 GB and M3 Pro 36 GB  |  current default = ordinary decode, lane wrapper off",
             color="#536575", fontsize=11)
    fig.text(0.18, 0.082,
             "Ordinary smoke passed on both hosts  |  applicable feature gates: M5 10/10; M3 9/9 safe plus Fly isolated",
             color="#273e4c", fontsize=9.5, fontweight="bold")
    fig.text(0.18, 0.055,
             "Mixed-load aggregate and single-stream decode are different measurements. "
             "PLD ladder prompts favor n-gram reuse; speculative 20×20 rows lack batching evidence.",
             color="#536575", fontsize=9.5)
    fig.text(0.18, 0.031, "Source: benchmark_results/2026-09-29/muse.md (revision-bound rows).",
             color="#687a89", fontsize=8.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190, facecolor=fig.get_facecolor())
    if args.output.suffix.lower() != ".svg":
        svg_path = args.output.with_suffix(".svg")
        fig.savefig(svg_path, facecolor=fig.get_facecolor())
        # Matplotlib indents multiline path data with trailing spaces.
        svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
