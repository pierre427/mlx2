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
    "flash-next": "The earlier contaminated 20×20 remains historical. A source-bound MoE isolation run measured zero swap-out pages for the split control and expert-only dispatch, versus 389,404 pages during fused gate/up load plus 33,784 while active. This identifies the triggering option, not the underlying allocator mechanism. The current Flash default retains split gate/up projections and file-backed PLE; the current-source stress and focused feature receipts above are separate gates.",
    "flash-next-uncensored": "The current Flash default retains split gate/up projections and file-backed PLE. Its focused memory and feature checks and the full 20×20 stress receipt are independent gates.",
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
    for tag in ("split-mtp-counter", "split-default"):
        focused = directory / f"stress-{tag}"
        if focused.joinpath("stress.json").is_file():
            directory = focused
            break
    receipt = read(directory / "stress.json")
    result = read(directory / "20x20.json")
    if receipt is None and result is None:
        return ["20×20: not run in this queue. The smoke pass below does not imply a stress pass."]
    lines = [f"20×20 gate: **{fmt(receipt.get('status') if receipt else 'receipt unavailable')}**."]
    if result:
        mtp = result.get("segmented_mtp_delta") or {}
        width_label = "ordinary-reply peak-width field" if mtp else "observed peak batch width"
        speeds = [row.get("tokens_per_second") for row in result.get("rounds_detail", [])]
        speeds = [float(speed) for speed in speeds if isinstance(speed, (int, float))]
        lines += [
            f"Graded correct: **{result.get('graded_correct', 0)}/{result.get('requests', 0)}**; HTTP errors: **{result.get('http_errors', 0)}**; {width_label}: **{fmt(result.get('peak_observed_width'))}**.",
            f"Median aggregate generated rate across rounds: **{statistics.median(speeds):.1f} tokens/s**. This is mixed-workload throughput, not single-stream decode speed." if speeds else "Round throughput: unavailable.",
        ]
        issues = result.get("issue_totals") or {}
        if issues:
            lines.append("Issue counts: " + ", ".join(f"{key}={value}" for key, value in sorted(issues.items())) + ".")
        if mtp:
            lines.append(f"Native-MTP batched target forwards during 20×20: **{mtp.get('batched_target_forwards', 0)}**; true-batched requests: **{mtp.get('true_batched_requests', 0)}**. Ordinary reply width is not the native-MTP batching metric.")
    if receipt:
        swap = (receipt.get("swapouts") or {}).get("delta")
        lines.append(f"Owned-run swap-out delta: **{fmt(swap)} pages**; APCv2 repeated-prefix probe: **{fmt((receipt.get('apc_probe') or {}).get('passed'))}**; batching engaged: **{fmt(receipt.get('batching_engaged'))}**.")
        lines.append(f"Source commit: `{receipt.get('source_head', 'unknown')}`; artifact config SHA-256: `{receipt.get('artifact_config_sha256', 'unknown')}`.")
    return lines


def ladder_lines(directory: Path, host: str) -> list[str]:
    pattern = "ladder-1024-262144-r3.json" if host.startswith("m5") else "ladder-1024-8192-r3-w1.json"
    single_flash = False
    if host.startswith("m5") and directory.name == "flash-next":
        single_prompt = directory / "ladder-1024-262144-r3-w1.json"
        if single_prompt.is_file() and read(single_prompt).get("finished_at"):
            pattern = single_prompt.name
            single_flash = True
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
            if single_flash:
                prefill = [float(v) for r in runs if isinstance((v := r.get("summary", {}).get("prefill_tokens_per_second")), (int, float))]
                warm_ttft = [float(v) for r in runs if isinstance((v := r.get("summary", {}).get("warm_ttft_seconds")), (int, float))]
                decode_spread = (((cell.get("stats") or {}).get("decode_tokens_per_second") or {}).get("spread_pct"))
                rows.append(f"| {cell['requested_tokens']:,} | {len(runs)} | {fmt(cell.get('passed'))} | {statistics.median(ttft):.2f} | {statistics.median(prefill):.0f} | {statistics.median(decode):.1f} | {decode_spread:.0f}% | {statistics.median(warm_ttft):.2f} |")
            else:
                rows.append(f"| {cell['requested_tokens']:,} | {cell['width']} | {len(runs)} | {fmt(cell.get('passed'))} | {statistics.median(ttft):.2f} | {statistics.median(decode):.1f} |")
    if not rows:
        return ["Ladder attempted, with no complete measured cell."]
    state = "passed" if ladder.get("passed") is True else "partial or interrupted"
    if single_flash:
        lines = [f"Thermally controlled single-prompt ladder: **{state}**. Only completed measured cells appear below.",
                 "", "| Prompt tokens | Measured runs | Cell passed | Median cold TTFT (s) | Median prefill (tokens/s) | Median decode (tokens/s) | Decode spread | Median warm TTFT (s) |",
                 "|---:|---:|:---:|---:|---:|---:|---:|---:|", *rows]
    else:
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


def flash_candidate_lines(directory: Path) -> list[str]:
    stem = "ladder-1024-262144-r3-w1-tensorfold-qmv-mtp-pld-latched-reset"
    ladder = read(directory / f"{stem}.json")
    run = read(directory / f"{stem}-run.json")
    if not ladder or not run or not ladder.get("finished_at"):
        return []
    lines = [
        "### TensorFold row kernel and gated speculation candidate",
        "",
        "Opt-in profile: adapted TensorFold q4/group-64 row matvec for eligible dense projections; "
        "native MTP with a single-stream acceptance/goodput latch; prompt-copy proposals "
        "inside MTP with match-strength and yield latches. This prompt-copy path is "
        "PLD-style, not the separate `prompt_lookup` route. The full TensorFold fused "
        "Flash executor is not integrated or qualified.",
        "",
    ]
    reset_gate = read(directory / "ladder-1024-1024-r3-w1-tensorfold-latched-reset-gate.json")
    fixed_control = read(directory / "ladder-1024-1024-r3-w1-tensorfold-fixed-depth-control.json")
    if reset_gate and fixed_control:
        reset_runs = [r for c in reset_gate.get("cells", []) for r in c.get("runs", [])]
        fixed_runs = [r for c in fixed_control.get("cells", []) for r in c.get("runs", [])]
        lines += [
            "The first latched ladder was interrupted after cold/warm output mismatches. "
            "An empty scheduler batch had retained the prior request's adaptive "
            "depth and goodput state. The same-source fixed-depth control matched "
            f"{sum(r.get('summary', {}).get('warm_equals_cold') == 1 for r in fixed_runs)}/{len(fixed_runs)} "
            "at 1K; after the request-boundary reset, the latched 1K gate matched "
            f"{sum(r.get('summary', {}).get('warm_equals_cold') == 1 for r in reset_runs)}/{len(reset_runs)}. "
            "The interrupted cells are excluded from this ladder.",
            "",
        ]
    lines += [
        f"Thermal ladder: **{fmt(run.get('status'))}**; source `{run.get('source_head')}`; "
        f"execution policy SHA-256 `{run.get('execution_policy_sha256')}`; "
        f"owned-run swap-out delta **{fmt((run.get('swapouts') or {}).get('delta'))} pages**.",
        "",
        "| Prompt tokens | Runs | Cell passed | Cold/warm equal | Median cold TTFT (s) | Median prefill (tokens/s) | Median decode (tokens/s) | Decode spread |",
        "|---:|---:|:---:|:---:|---:|---:|---:|---:|",
    ]
    for cell in ladder.get("cells", []):
        runs = cell.get("runs") or []
        if len(runs) != 3 or not cell.get("passed"):
            continue
        stats = cell.get("stats") or {}
        equal = sum(r.get("summary", {}).get("warm_equals_cold") == 1 for r in runs)
        lines.append(
            f"| {cell['requested_tokens']:,} | 3 | yes | {equal}/3 | "
            f"{stats['ttft_seconds']['median']:.2f} | "
            f"{stats['prefill_tokens_per_second']['median']:.0f} | "
            f"{stats['decode_tokens_per_second']['median']:.1f} | "
            f"{stats['decode_tokens_per_second']['spread_pct']:.0f}% |"
        )
    all_runs = [r for cell in ladder.get("cells", []) for r in cell.get("runs", [])]
    equal_runs = sum(r.get("summary", {}).get("warm_equals_cold") == 1 for r in all_runs)
    failed_cells = [c for c in ladder.get("cells", []) if not c.get("passed")]
    lines += [
        "",
        f"Temperature-zero cold/warm equality: **{equal_runs}/{len(all_runs)}** "
        "measured repeats. Any mismatch keeps this candidate unqualified.",
    ]
    for cell in failed_cells:
        lines.append(
            f"The {cell['requested_tokens']:,}-token cell did not pass: "
            f"{cell.get('error') or 'incomplete'}; {len(cell.get('runs') or [])} measured runs. "
            "This cell is excluded from performance medians."
        )
    scheduler = (run.get("status_after") or {}).get("scheduler") or {}
    if scheduler:
        lines += [
            "",
            "Observed latch counters across the owned run: "
            f"MTP depth changes {scheduler.get('adaptive_mtp_depth_changes', 0)}, "
            f"MTP cost probes {scheduler.get('adaptive_mtp_cost_probes', 0)}, "
            f"prompt-copy rounds {scheduler.get('self_mtp_copy_rounds', 0)}, "
            f"copy gate declines {scheduler.get('self_mtp_copy_gate_declines', 0)}. "
            "These show exercised candidate behavior, not a production default.",
        ]
    candidate_stress = directory / "stress-tensorfold-qmv-mtp-pld-latched-reset"
    stress_receipt = read(candidate_stress / "stress.json")
    if stress_receipt and stress_receipt.get("finished_at"):
        lines += ["", "#### Candidate 20×20, APCv2, and batching", "",
                  *stress_lines(candidate_stress)]
    return lines


def feature_lines(directory: Path) -> list[str]:
    if directory.name in {"flash-next", "flash-next-uncensored"}:
        initial_core = read(directory / "features-core/qualification.json")
        cache16_core = read(directory / "features-core-cache16/qualification.json")
        core = cache16_core if cache16_core and cache16_core.get("status") != "running" else initial_core
        if core:
            rolling = read(directory / "features-apc_rolling_checkpoints/qualification.json")
            kernels = read(directory / "features-kernels-safe/qualification.json")
            memory_probes = [read(p) for p in directory.glob("memory-probe-split-projections-*.json")]
            memory_probes = [p for p in memory_probes if p]
            latest_probe = max(memory_probes, key=lambda p: p.get("finished_at", 0), default=None)
            summary = core.get("summary") or {}
            cap = (core.get("host_caps") or {}).get("cache_gib")
            lines = [f"Core feature sweep: **{core.get('status')}**, {summary.get('ok')}/{summary.get('applicable')} applicable checks in one combined run, APCv2 budget {cap or 'default'} GiB; source `{core.get('source_head')}`; swap-out delta {(core.get('swapouts') or {}).get('delta')} pages."]
            if cache16_core and initial_core and core is cache16_core:
                earlier = initial_core.get("summary") or {}
                earlier_rolling = (((earlier.get("detail") or {}).get("apc_rolling_checkpoints") or {}).get("evidence") or {})
                lines.append(f"Earlier 8 GiB combined sweep: **{initial_core.get('status')}**, {earlier.get('ok')}/{earlier.get('applicable')} checks; rolling recovery missed as APCv2 recorded {(earlier_rolling.get('memory') or {}).get('apc_pressure_spills')} pressure spills. Its isolated retry passed separately below.")
                rolling_evidence = (((summary.get("detail") or {}).get("apc_rolling_checkpoints") or {}).get("evidence") or {})
                if core.get("status") != "pass":
                    lines.append(f"The 16 GiB rerun still recorded {(rolling_evidence.get('memory') or {}).get('apc_pressure_spills')} APCv2 pressure spills and only {rolling_evidence.get('retry_cached_tokens')} cached retry tokens; the combined interaction remains open.")
            if summary.get("failed"):
                lines.append("Combined-run open checks: " + ", ".join(summary["failed"]) + ".")
            if rolling:
                evidence = (((rolling.get("summary") or {}).get("detail") or {}).get("apc_rolling_checkpoints") or {}).get("evidence") or {}
                lines.append(f"Isolated rolling recovery: **{rolling.get('status')}**, {evidence.get('rolling_hits')} APCv2 rolling hit(s), {evidence.get('retry_cached_tokens')} retry cached tokens, swap-out delta {(rolling.get('swapouts') or {}).get('delta')} pages; source `{rolling.get('source_head')}`.")
            fly = (summary.get("detail") or {}).get("fly_verification") or {}
            fly_evidence = fly.get("evidence") or {}
            lines.append(f"FLy greedy route selected with sampled exact fallback; relaxed accepts observed: {fly_evidence.get('relaxed_accepts')}. Approximate relaxation has no observed-use claim from this probe and remains default-off.")
            lines.append("Cache capsules are inapplicable: this hybrid artifact has no plain KVCache plane eligible for capsule fanout.")
            if latest_probe:
                ple = latest_probe.get("ple_offload") or {}
                values = [row.get("swapouts") for row in latest_probe.get("phases") or []]
                delta = values[-1] - values[0] if len(values) > 1 and None not in values else None
                lines.append(f"Current-default PLE and smoke probe: **{latest_probe.get('status')}**, {ple.get('rows_read')} sidecar rows read, {delta} swap-out pages from before load through shutdown; source `{latest_probe.get('source_head')}`.")
            if kernels:
                rows = (kernels.get("summary") or {}).get("kernels") or []
                passed = [row["kernel"] for row in rows if row.get("ok")]
                lines.append(f"Safe kernel sweep: **{kernels.get('status')}**, {len(passed)}/{len(rows)} output-equal engaged arms, swap-out delta {(kernels.get('swapouts') or {}).get('delta')} pages; source `{kernels.get('source_head')}`. Passing arms: " + ", ".join(passed) + ". Gate/up fusion was excluded after its separate swap failure.")
            lines.append("Kernel rates in this sweep use one quick repetition; they do not establish a thermal performance gain or change route selection.")
            return lines
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
        if host.startswith("m5") and name == "flash-next":
            candidate = flash_candidate_lines(directory)
            if candidate:
                lines += ["", *candidate]
    if name in NOTES:
        lines += ["", "## Interpretation", "", NOTES[name]]
    if name == "qwen38-mlx-4bit":
        lines += ["", "## Current-main control", "", "This checkpoint was also used for the M5 single-prompt [mlx2, TensorFold, and omlx comparison](upstream-main-comparison.md). That later control has its own source revisions, settings, and thermal receipts; it is separate from this campaign's M3 ladder and 20×20 workload."]
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
        "- At the campaign pause, M3 North qualified eight applicable exercised feature operations. Other M3 feature attempts were partial or contaminated; no M5 per-model feature run had begun. The Flash follow-up below is later work.", "",
        "## Flash-Next follow-up", "",
        "Both staged Flash variants received focused M5 memory and feature checks after the pause. Their model pages distinguish combined feature observations, isolated APCv2 rolling recovery, optional kernel checks, and any 20×20 result. The new default avoids gate/up fusion's observed swap while retaining file-backed PLE. The M3 does not hold either large Flash artifact.", "",
        "The Flash-Next page also reports the later opt-in TensorFold row-kernel and gated MTP/prompt-copy candidate when its thermal ladder has completed. It is a candidate measurement, not a default-route promotion or a full TensorFold executor qualification.", "",
        "A later [current-main comparison](upstream-main-comparison.md) records why the staged Flash-Next checkpoint did not load in TensorFold or omlx, plus a thermally controlled three-engine speed control using the shared Qwen3.8 27B 4-bit checkpoint.", "",
        "The 20×20 rate is median aggregate generated tokens per second across mixed domain rounds. Context-ladder decode is per stream, from thermally admitted measured runs. They are different measurements. A functional smoke, stress pass, feature implementation, feature qualification, route selection, and observed production use are distinct states.", "",
        "## Model pages", "", "| Model | M5 smoke | M5 20×20 | M3 staged |", "|---|---|---|---|",
    ]
    for name in names:
        m5 = RUNS / HOSTS[0][0] / name
        m3 = RUNS / HOSTS[1][0] / name
        stress = (read(m5 / "stress-split-mtp-counter/stress.json")
                  or read(m5 / "stress-split-default/stress.json")
                  or read(m5 / "stress.json"))
        index.append(f"| [{name}]({name}.md) | passed | {stress.get('status', 'not run') if stress else 'not run'} | {'yes' if (m3 / 'smoke.json').exists() else 'no'} |")
    index += [
        "", "## Scripts and provenance", "",
        "The `scripts/` directory snapshots the smoke, 20×20, concurrency, thermal-ladder, feature, queue, and memory-probe scripts plus their execution policies. `scripts/manifest.json` records each original and published SHA-256. Two Python snapshots replace the run host's home-directory prefix with `Path.home()`; two policies use a literal `${HOME}` placeholder for the external draft path, and one local LaunchAgent label is anonymized. Other algorithm and test settings are unchanged. Restore the original qualification layout and set local model paths before running these historical snapshots; they also require the local MLX environment. The generated pages can be rebuilt from the private source-bound receipts with `build_model_pages.py --receipt-root <campaign-receipt-directory>`; `package_scripts.py` rebuilds the script snapshot from the private qualification source.", "",
        "Private source receipt paths are recorded only as source commit IDs and artifact configuration hashes on the model pages. The private pause record is `qualification/runs/requal-20260928/PAUSED-20260929.md` at Forgejo commit `df0120ca`; it is intentionally not copied here.", "",
        "## Open gates at pause", "",
        "Flash-Next full-stress load swap; Laguna final-answer quality and thinking-marker leakage; Xing final-answer quality and BF16 swap; Gemma 3n supported-surface quality; Qwen3.8 M5 warm APCv2 misses and 262K admission; M3 multi-stream APCv2/admission limits; M3 Muse Fly repeated-prefix admission; remaining M5 performance and per-model feature qualification.", "",
    ]
    (HERE / "README.md").write_text("\n".join(index))
    print(f"Wrote {len(names)} model pages")


if __name__ == "__main__":
    main()
