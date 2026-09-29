#!/usr/bin/env python3
"""Rebuild the public-safe model pages from the private campaign receipts.

Only a fixed set of aggregate fields is emitted. Raw prompts, completions,
local paths, process details, and credentials never leave the receipt tree.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "qualification/runs/requal-20260928"
HOSTS = (("m5-max-128gb", "M5 Max, 128 GB"), ("m3-pro-36gb", "M3 Pro, 36 GB"))

NOTES = {
    "flash-next": "File-backed PLE was observed in clean smoke, but the full 20×20 run swapped during load; the stress gate remains contaminated. The harness sampled width one while scheduler counters reported multi-request cycles; that batching discrepancy needs review.",
    "flash-next-uncensored": "File-backed PLE passed a clean longer probe; a current-source full 20×20 stress pass remains open.",
    "laguna": "Default reasoning exhausted the answer allowance in many requests. A guarded candidate scored 400/400 but leaked `</think>` and was not promoted.",
    "xing": "Fifteen empty final answers kept the default-route quality gate open.",
    "xing-bf16": "Eleven empty answers and swap growth kept the quality and memory gates open.",
    "gemma3n": "Sixty grammar/tool requests were correctly refused as unsupported; supported text tasks also missed quality checks. Media needs its own workload.",
    "muse": "The M5 ladder was interrupted by the user's pause after nine passing cells. M3 external DFlash2 load swap was fixed, but Fly repeated-prefix checks still returned 429.",
    "qwen38": "The M5 32K four-stream cell missed warm APCv2 reuse; measured 262K was refused by memory admission after a successful unmeasured warmup.",
    "qwen38-crack-4bit": "The M3 gdn_core candidate differed from its base on one open-ended prompt; it is not selected.",
    "qwen36-27b-heretic-4bit": "The standard-cap M3 width-one ladder was partial. A separate 4 GiB cache / 8K context retry passed three measured 4K single-stream runs; it does not turn the standard-cap ladder into a pass.",
}


def read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def fmt(value: object) -> str:
    return "—" if value is None else str(value)


def stress_lines(directory: Path) -> list[str]:
    receipt = read(directory / "stress.json")
    result = read(directory / "20x20.json")
    if receipt is None and result is None:
        return ["20×20: not run in this queue. The smoke pass below does not imply a stress pass."]
    lines = [f"20×20 gate: **{fmt(receipt.get('status') if receipt else 'receipt unavailable')}**."]
    if result:
        speeds = [row.get("tokens_per_second") for row in result.get("rounds_detail", [])]
        speeds = [float(speed) for speed in speeds if isinstance(speed, (int, float))]
        lines += [
            f"Graded correct: **{result.get('graded_correct', 0)}/{result.get('requests', 0)}**; HTTP errors: **{result.get('http_errors', 0)}**; observed peak batch width: **{fmt(result.get('peak_observed_width'))}**.",
            f"Median aggregate generated rate across rounds: **{statistics.median(speeds):.1f} tokens/s**. This is mixed-workload throughput, not single-stream decode speed." if speeds else "Round throughput: unavailable.",
        ]
        issues = result.get("issue_totals") or {}
        if issues:
            lines.append("Issue counts: " + ", ".join(f"{key}={value}" for key, value in sorted(issues.items())) + ".")
    if receipt:
        swap = (receipt.get("swapouts") or {}).get("delta")
        lines.append(f"Owned-run swap-out delta: **{fmt(swap)} pages**; APCv2 repeated-prefix probe: **{fmt((receipt.get('apc_probe') or {}).get('passed'))}**; batching engaged: **{fmt(receipt.get('batching_engaged'))}**.")
        lines.append(f"Source commit: `{receipt.get('source_head', 'unknown')}`; artifact config SHA-256: `{receipt.get('artifact_config_sha256', 'unknown')}`.")
    return lines


def ladder_lines(directory: Path, host: str) -> list[str]:
    pattern = "ladder-1024-262144-r3.json" if host.startswith("m5") else "ladder-1024-8192-r3-w1.json"
    path = directory / pattern
    ladder = read(path)
    if ladder is None:
        return ["No three-repetition performance ladder in this campaign."]
    rows = []
    for cell in ladder.get("cells", []):
        runs = cell.get("runs", [])
        if not runs:
            continue
        decode = [r.get("summary", {}).get("decode_tokens_per_second") for r in runs]
        decode = [float(v) for v in decode if isinstance(v, (int, float))]
        ttft = [r.get("summary", {}).get("ttft_seconds") for r in runs]
        ttft = [float(v) for v in ttft if isinstance(v, (int, float))]
        if decode:
            rows.append(f"| {cell['requested_tokens']:,} | {cell['width']} | {len(runs)} | {fmt(cell.get('passed'))} | {statistics.median(ttft):.2f} | {statistics.median(decode):.1f} |")
    if not rows:
        return ["Ladder attempted, with no complete measured cell."]
    state = "passed" if ladder.get("passed") is True else "partial or interrupted"
    lines = [f"Three-repetition ladder: **{state}**. Only completed measured cells appear below.",
             "", "| Prompt tokens | Width | Measured runs | Cell passed | Median cold TTFT (s) | Median decode (tokens/s/stream) |",
             "|---:|---:|---:|:---:|---:|---:|", *rows]
    run = read(path.with_name(path.stem + "-run.json"))
    if run:
        lines.append(f"Source commit: `{run.get('source_head', 'unknown')}`; owned-run swap-out delta: {fmt((run.get('swapouts') or {}).get('delta'))} pages.")
    if host.startswith("m5") and directory.name == "muse":
        lines.append("The user paused the queue before the 131K run; the harness return code `-15` records interruption, not a model failure.")
    if host.startswith("m5") and directory.name == "qwen38":
        lines.append("The unmeasured 262K warmup is excluded from the table; the first measured request received HTTP 429.")
    if host.startswith("m3") and directory.name == "qwen36-27b-heretic-4bit":
        retry = read(directory / "ladder-4096-4096-r3-w1.json")
        if retry and retry.get("passed"):
            cell = retry["cells"][0]
            speeds = [float(r["summary"]["decode_tokens_per_second"]) for r in cell["runs"]]
            lines.append(f"Separate capped 4K retry: **passed** {len(speeds)} measured runs at {statistics.median(speeds):.2f} median decode tokens/s/stream; 4 GiB APCv2 cache and 8K serving context.")
    return lines


def feature_lines(directory: Path) -> list[str]:
    receipt = read(directory / "features/qualification.json")
    if not receipt:
        return ["Feature qualification was not run on this model and host."]
    summary = receipt.get("summary") or {}
    status = receipt.get("status", "unknown")
    detail = summary.get("detail") or {}
    passed = [name for name, item in detail.items() if item.get("applicable") and item.get("engaged") and item.get("ok")]
    failed = [name for name, item in detail.items() if item.get("applicable") and not item.get("ok")]
    lines = [f"Feature-run status: **{status}**; applicable operations: {fmt(summary.get('applicable'))}; engaged and passing in this run: {len(passed)}."]
    if passed:
        qualifier = "Qualified exercised operations" if status == "pass" else "Passing observations in a partial/contaminated run; not promoted by this report"
        lines.append(f"{qualifier}: " + ", ".join(sorted(passed)) + ".")
    if failed:
        lines.append("Open or inconclusive operations: " + ", ".join(sorted(failed)) + ".")
    lines.append("Feature engagement and qualification do not select a production route or establish production use.")
    return lines


def model_page(name: str) -> str:
    smoke = read(RUNS / HOSTS[0][0] / name / "smoke.json")
    if smoke is None:
        raise ValueError(f"missing M5 smoke: {name}")
    artifact_name = Path(smoke.get("artifact", "unknown")).name
    lines = [f"# {name}", "", f"Artifact: `{artifact_name}`. This page summarizes the 2026-09-28–29 campaign; model files and raw request content are not included.",
             "", "Statuses below are per workload and host. A smoke or 20×20 pass does not by itself qualify every route or feature."]
    for host, label in HOSTS:
        directory = RUNS / host / name
        s = read(directory / "smoke.json")
        if s is None:
            if host.startswith("m3"):
                lines += ["", f"## {label}", "", "Model artifact not staged on this host; no load or performance verdict."]
            continue
        lines += ["", f"## {label}", "", f"Served smoke: **{s.get('status', 'unknown')}**; default route: `{s.get('route', 'unknown')}`; source commit: `{s.get('source_head', 'unknown')}`; artifact config SHA-256: `{s.get('artifact_config_sha256', 'unknown')}`.",
                  "", "### 20×20 domain and batching", "", *stress_lines(directory),
                  "", "### Context performance", "", *ladder_lines(directory, host),
                  "", "### Feature qualification", "", *feature_lines(directory)]
    if name in NOTES:
        lines += ["", "## Interpretation", "", NOTES[name]]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    global RUNS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-root", type=Path, default=RUNS)
    args = parser.parse_args()
    RUNS = args.receipt_root.resolve()
    names = sorted(p.name for p in (RUNS / HOSTS[0][0]).iterdir() if p.is_dir() and (p / "smoke.json").is_file())
    if len(names) != 34:
        raise RuntimeError(f"expected 34 staged M5 smoke models, found {len(names)}")
    for name in names:
        (HERE / f"{name}.md").write_text(model_page(name))
    index = [
        "# mlx2 qualification results, 2026-09-29", "",
        "The 2026-09-28–29 campaign was paused at the user's request. These are public summaries of source-bound private receipts, one page per M5-staged model. No raw prompts, model files, local paths, process details, or credentials are published.", "",
        "## Scope", "",
        "- M5 Max 128 GB: 34/34 staged models passed ordinary served smoke with expected defaults. The eligible 20×20 queue recorded 22 passes and four failures; eight roster entries were outside that queue.",
        "- Flash-Next also has a separately retained contaminated 20×20 attempt, although it was excluded from the later eligible queue. This is why five model pages show a nonpassing 20×20 receipt while the queue itself reports four failures.",
        "- M3 Pro 36 GB: six locally staged models passed smoke and 400/400 in 20×20. The other artifacts were absent on that host, not load failures.",
        "- Qwen3.6 passed the complete M5 three-repetition context ladder. Qwen3.8 was partial. Muse's M5 ladder was deliberately interrupted after nine passing cells. Five M3 models passed a separate single-stream 1K/4K basic ladder.",
        "- M3 North qualified eight applicable exercised feature operations. Other M3 feature attempts were partial or contaminated; no M5 per-model feature run began.", "",
        "The 20×20 rate is median aggregate generated tokens per second across mixed domain rounds. Context-ladder decode is per stream, from thermally admitted measured runs. They are different measurements. A functional smoke, stress pass, feature implementation, feature qualification, route selection, and observed production use are distinct states.", "",
        "## Model pages", "", "| Model | M5 smoke | M5 20×20 | M3 staged |", "|---|---|---|---|",
    ]
    for name in names:
        m5 = RUNS / HOSTS[0][0] / name
        m3 = RUNS / HOSTS[1][0] / name
        stress = read(m5 / "stress.json")
        index.append(f"| [{name}]({name}.md) | passed | {stress.get('status', 'not run') if stress else 'not run'} | {'yes' if (m3 / 'smoke.json').exists() else 'no'} |")
    index += [
        "", "## Scripts and provenance", "",
        "The `scripts/` directory snapshots the smoke, 20×20, concurrency, thermal-ladder, feature, queue, and memory-probe scripts plus their execution policies. `scripts/manifest.json` records each original and published SHA-256. Two Python snapshots replace the run host's home-directory prefix with `Path.home()`; two policies use a literal `${HOME}` placeholder for the external draft path. Other algorithm and test settings are unchanged. Restore the original qualification layout and set local model paths before running these historical snapshots; they also require the local MLX environment. The generated pages can be rebuilt from the private source-bound receipts with `build_model_pages.py --receipt-root <campaign-receipt-directory>`; `package_scripts.py` rebuilds the script snapshot from the private qualification source.", "",
        "Private source receipt paths are recorded only as source commit IDs and artifact configuration hashes on the model pages. The private pause record is `qualification/runs/requal-20260928/PAUSED-20260929.md` at Forgejo commit `df0120ca`; it is intentionally not copied here.", "",
        "## Open gates at pause", "",
        "Flash-Next full-stress load swap; Laguna final-answer quality and thinking-marker leakage; Xing final-answer quality and BF16 swap; Gemma 3n supported-surface quality; Qwen3.8 M5 warm APCv2 misses and 262K admission; M3 multi-stream APCv2/admission limits; M3 Muse Fly repeated-prefix admission; remaining M5 performance and per-model feature qualification.", "",
    ]
    (HERE / "README.md").write_text("\n".join(index))
    print(f"Wrote {len(names)} model pages")


if __name__ == "__main__":
    main()
