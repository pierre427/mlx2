#!/usr/bin/env python3
"""Export aggregate Lightning qualification evidence without local paths or replies."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "qualification/runs/requal-20260928/m5-max-128gb/nemotron35-lightning"
OUTPUT = HERE / "nemotron35-lightning-evidence.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stress(tag: str) -> dict | None:
    directory = RUN / f"stress-{tag}"
    receipt_path = directory / "stress.json"
    if not receipt_path.is_file():
        return None
    workload_path = directory / "20x20.json"
    receipt = read(receipt_path)
    if receipt.get("status") == "running":
        return None
    result = read(workload_path) if workload_path.is_file() else None
    log_path = directory / "20x20.log"
    partial_rounds = []
    if result is None and log_path.is_file():
        for seconds, tokens, rate in re.findall(
            r"round \d+\s+([\d.]+)s\s+(\d+) tok\s+([\d.]+) tok/s",
            log_path.read_text(),
        ):
            partial_rounds.append({"seconds": float(seconds),
                                   "completion_tokens": int(tokens),
                                   "aggregate_generated_tokens_per_second": float(rate)})
    rounds = (result or {}).get("rounds_detail", [])
    rates = [row.get("tokens_per_second") for row in rounds]
    rates = [float(rate) for rate in rates if isinstance(rate, (float, int))]
    total_tokens = sum(row.get("completion_tokens", 0) for row in rounds)
    total_seconds = sum(row.get("seconds", 0) for row in rounds)
    final_status = receipt.get("status_after") or {}
    mtp = final_status.get("segmented_self_mtp") or {}
    scheduler = final_status.get("scheduler") or {}
    return {
        "tag": tag,
        "source_head": receipt.get("source_head"),
        "route": receipt.get("route"),
        "status": receipt.get("status"),
        "max_lanes": receipt.get("max_lanes"),
        "cache_cap_gib": receipt.get("cache_cap_gib"),
        "thinking_allowance": receipt.get("thinking_allowance"),
        "execution_policy_sha256": receipt.get("execution_policy_sha256"),
        "partial_rounds": partial_rounds,
        "workload_log_sha256": digest(log_path) if log_path.is_file() else None,
        "swapout_pages": (receipt.get("swapouts") or {}).get("delta"),
        "apcv2_probe_passed": (receipt.get("apc_probe") or {}).get("passed"),
        "batching_engaged": receipt.get("batching_engaged"),
        "batching_evidence": receipt.get("batching_evidence"),
        "mtp_status_counters": {key: mtp.get(key) for key in (
            "batched_target_forwards", "true_batched_engaged", "true_batched_declined"
        )} if mtp else None,
        "mtp_ordinary_handoff_events": scheduler.get("mtp_ordinary_handoff_events"),
        "mtp_ordinary_handoff_lanes": scheduler.get("mtp_ordinary_handoff_lanes"),
        "requests": (result or {}).get("requests"),
        "graded_correct": (result or {}).get("graded_correct"),
        "http_errors": (result or {}).get("http_errors"),
        "issues": (result or {}).get("issue_totals"),
        "observed_widths": (result or {}).get("observed_widths"),
        "completion_tokens": total_tokens,
        "round_seconds": round(total_seconds, 2),
        "weighted_generated_tokens_per_second": (
            round(total_tokens / total_seconds, 2) if total_seconds else None),
        "aggregate_generated_tokens_per_second": (
            {"median": statistics.median(rates), "min": min(rates), "max": max(rates)}
            if rates else None
        ),
        "receipt_sha256": digest(receipt_path),
        "workload_sha256": digest(workload_path) if result is not None else None,
    }


def smoke(name: str) -> dict | None:
    path = RUN / name
    if not path.is_file():
        return None
    receipt = read(path)
    if receipt.get("status") == "running":
        return None
    def case_summary(row: dict) -> dict:
        summary = {"case": row.get("case"), "passed": row.get("passed"),
                   "finish_reason": row.get("finish_reason")}
        mtp = (row.get("route_receipt") or {}).get("mtp") or {}
        if mtp:
            stats = mtp.get("stats") or {}
            summary["mtp"] = {
                "route": mtp.get("route"), "num_draft": mtp.get("num_draft"),
                "verification": mtp.get("verification"),
                "draft_proposed": stats.get("draft_proposed"),
                "draft_accepted": stats.get("draft_accepted"),
                "draft_acceptance": stats.get("draft_acceptance"),
                "plain_cycles": stats.get("plain_cycles"),
            }
        return summary

    return {
        "source_head": receipt.get("source_head"),
        "status": receipt.get("status"),
        "route": receipt.get("route"),
        "artifact_config_sha256": receipt.get("artifact_config_sha256"),
        "default_checks": receipt.get("default_checks"),
        "sampling_drift": receipt.get("sampling_drift"),
        "cases": [case_summary(row) for row in (receipt.get("smoke") or {}).get("cases", [])],
        "receipt_sha256": digest(path),
    }


def ladder(stem: str) -> dict | None:
    receipt_path = RUN / f"{stem}-run.json"
    result_path = RUN / f"{stem}.json"
    if not receipt_path.is_file() or not result_path.is_file():
        return None
    receipt, result = read(receipt_path), read(result_path)
    if receipt.get("status") == "running":
        return None
    cells = []
    for cell in result.get("cells", []):
        stats = cell.get("stats") or {}
        cells.append({
            "context_tokens": cell.get("requested_tokens"),
            "width": cell.get("width"),
            "passed": cell.get("passed"),
            "repetitions": len(cell.get("runs") or []),
            "needle_quality": cell.get("quality"),
            "cold_ttft_median_seconds": (stats.get("ttft_seconds") or {}).get("median"),
            "prefill_median_tokens_per_second": (
                stats.get("prefill_tokens_per_second") or {}).get("median"),
            "decode_median_tokens_per_second_per_stream": (
                stats.get("decode_tokens_per_second") or {}).get("median"),
        })
    return {
        "tag": stem,
        "source_head": receipt.get("source_head"),
        "status": receipt.get("status"),
        "route": receipt.get("route"),
        "host_caps": receipt.get("host_caps"),
        "runs_per_cell": receipt.get("runs_per_cell"),
        "swapout_pages": (receipt.get("swapouts") or {}).get("delta"),
        "passed": result.get("passed"),
        "cells": cells,
        "receipt_sha256": digest(receipt_path),
        "ladder_sha256": digest(result_path),
    }


def feature_run(stem: str) -> dict | None:
    path = RUN / stem / "qualification.json"
    if not path.is_file():
        return None
    receipt = read(path)
    if receipt.get("status") == "running":
        return None
    details = (receipt.get("summary") or {}).get("detail") or {}
    selected_evidence = {
        "apc_rolling_checkpoints": ("planned", "published", "retired", "rolling_hits",
                                    "cancel_published", "retry_cached_tokens", "retry_status"),
        "fly_verification": ("verification", "fly_disabled", "fly_disabled_reason",
                             "relaxed_accepts"),
    }
    return {
        "tag": stem,
        "source_head": receipt.get("source_head"),
        "status": receipt.get("status"),
        "only_feature": receipt.get("only_feature"),
        "host_caps": receipt.get("host_caps"),
        "rolling_units": receipt.get("rolling_units"),
        "rolling_prefill_step": receipt.get("rolling_prefill_step"),
        "applicable": (receipt.get("summary") or {}).get("applicable"),
        "okay": sum(bool(row.get("applicable") and row.get("engaged") and row.get("ok"))
                    for row in details.values()),
        "swapout_pages": (receipt.get("swapouts") or {}).get("delta"),
        "features": {
            name: {
                "applicable": row.get("applicable"),
                "engaged": row.get("engaged"),
                "okay": row.get("ok"),
                "notes": row.get("notes"),
                "evidence": {key: (row.get("evidence") or {}).get(key) for key in selected_evidence.get(name, ())},
            }
            for name, row in details.items()
        },
        "receipt_sha256": digest(path),
    }


def main() -> None:
    tags = (
        "lightning-ordinary-20260929",
        "lightning-ordinary-l8c4-20260929",
        "lightning-ordinary-l8c4-think4096-20260929",
        "lightning-ordinary-latest-d174-20260929",
        "lightning-ordinary-current-mamba-clamp-20260929",
        "lightning-ordinary-current-apc-20260929",
        "lightning-mtp2-current-apc-20260929",
        "lightning-mtp2-width8-current-20260929",
    )
    report = {
        "schema": "mlx2.public-lightning-evidence.v1",
        "model": "nemotron35-lightning",
        "host": "m5-max-128gb",
        "smokes": [row for name in (
            "smoke.json", "smoke-latest-d174-20260929.json",
            "smoke-current-mamba-clamp-20260929.json",
            "smoke-current-apc-20260929.json",
            "smoke-mtp2-current-apc-mtp2-20260929.json",
        ) if (row := smoke(name)) is not None],
        "stress_attempts": [row for tag in tags if (row := stress(tag)) is not None],
        "ladders": [row for stem in (
            "ladder-1024-32768-r3-lightning-ordinary-short-c4-20260929",
            "ladder-65536-262144-r3-w1-lightning-ordinary-long-c8-20260929",
            "ladder-1024-32768-r3-lightning-ordinary-short-latest-20260929",
            "ladder-16384-16384-r3-lightning-ordinary-isolated-latest-20260929",
            "ladder-1024-32768-r3-lightning-ordinary-post-clamp-cooled-20260929",
            "ladder-65536-262144-r3-w1-lightning-ordinary-post-clamp-long-20260929",
        ) if (row := ladder(stem)) is not None],
        "feature_runs": [row for stem in (
            "features-core-lightning-final-20260929",
            "features-fly_verification-lightning-isolated-fly-20260929",
            "features-apc_rolling_checkpoints-lightning-isolated-rolling-u300-s64-20260929",
        ) if (row := feature_run(stem)) is not None],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
