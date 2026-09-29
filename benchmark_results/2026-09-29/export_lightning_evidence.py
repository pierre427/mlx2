#!/usr/bin/env python3
"""Export aggregate Lightning qualification evidence without local paths or replies."""

from __future__ import annotations

import hashlib
import json
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


def stress(tag: str) -> dict:
    directory = RUN / f"stress-{tag}"
    receipt_path = directory / "stress.json"
    workload_path = directory / "20x20.json"
    receipt = read(receipt_path)
    result = read(workload_path) if workload_path.is_file() else None
    rates = [row.get("tokens_per_second") for row in (result or {}).get("rounds_detail", [])]
    rates = [float(rate) for rate in rates if isinstance(rate, (float, int))]
    return {
        "tag": tag,
        "source_head": receipt.get("source_head"),
        "route": receipt.get("route"),
        "status": receipt.get("status"),
        "max_lanes": receipt.get("max_lanes"),
        "cache_cap_gib": receipt.get("cache_cap_gib"),
        "thinking_allowance": receipt.get("thinking_allowance"),
        "swapout_pages": (receipt.get("swapouts") or {}).get("delta"),
        "apcv2_probe_passed": (receipt.get("apc_probe") or {}).get("passed"),
        "batching_engaged": receipt.get("batching_engaged"),
        "requests": (result or {}).get("requests"),
        "graded_correct": (result or {}).get("graded_correct"),
        "http_errors": (result or {}).get("http_errors"),
        "issues": (result or {}).get("issue_totals"),
        "observed_widths": (result or {}).get("observed_widths"),
        "aggregate_generated_tokens_per_second": (
            {"median": statistics.median(rates), "min": min(rates), "max": max(rates)}
            if rates else None
        ),
        "receipt_sha256": digest(receipt_path),
        "workload_sha256": digest(workload_path) if result is not None else None,
    }


def main() -> None:
    smoke_path = RUN / "smoke.json"
    smoke = read(smoke_path)
    safe_smoke = {
        "source_head": smoke.get("source_head"),
        "status": smoke.get("status"),
        "route": smoke.get("route"),
        "artifact_config_sha256": smoke.get("artifact_config_sha256"),
        "default_checks": smoke.get("default_checks"),
        "sampling_drift": smoke.get("sampling_drift"),
        "cases": [
            {"case": row.get("case"), "passed": row.get("passed"),
             "finish_reason": row.get("finish_reason")}
            for row in (smoke.get("smoke") or {}).get("cases", [])
        ],
        "receipt_sha256": digest(smoke_path),
    }
    tags = (
        "lightning-ordinary-20260929",
        "lightning-ordinary-l8c4-20260929",
        "lightning-ordinary-l8c4-think4096-20260929",
    )
    report = {
        "schema": "mlx2.public-lightning-evidence.v1",
        "model": "nemotron35-lightning",
        "host": "m5-max-128gb",
        "smoke": safe_smoke,
        "stress_attempts": [stress(tag) for tag in tags],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
