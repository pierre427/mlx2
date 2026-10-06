#!/usr/bin/env python3
"""Run the North row-exact q4 candidate at matched B1/B4 context rungs.

This is a bounded candidate gate, not qualification or performance evidence.
The caller must already own the CPG GPU lease and both filesystem locks.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
THERMAL_DIR = ROOT / "benchmark_results/2026-09-29/scripts/series"
sys.path.insert(0, str(THERMAL_DIR))
# thermal_ladder resolves its checked-in policy through this source root.
_prior_campaign_root = os.environ.get("MLX2_CAMPAIGN_ROOT")
os.environ["MLX2_CAMPAIGN_ROOT"] = str(ROOT)

import thermal_ladder as thermal
from run_north_feature_smoke import (
    atomic_json,
    batch_row_exact_q4_engaged,
    chat,
    content,
    prompt,
    receipt,
    request_json,
    stop_server,
    token_ids,
    validate_ownership,
    wait_health,
)

if _prior_campaign_root is None:
    os.environ.pop("MLX2_CAMPAIGN_ROOT", None)
else:
    os.environ["MLX2_CAMPAIGN_ROOT"] = _prior_campaign_root

REQUESTED_CONTEXTS = (1024, 4096, 16384)
WIDTH = 4
MAX_CONTEXT = 16384
LAYOUT_SUFFIX = ":north-batch-row-exact-q4-v1"
POLICY = {"batch_row_exact_q4": True}
DEFAULT_FOREIGN_CPU_THRESHOLD = 2.0
DEFAULT_SWAPOUT_TOLERANCE_PAGES = 0


def normalized_identifier(value: str) -> str:
    """Normalize dash punctuation only; keep all other bytes case-sensitive."""
    return value.translate(str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-"}))


def identifier_present(text: str, identifier: str) -> bool:
    return identifier in normalized_identifier(text)


def count_tokens(base: str, model: str, text: str, timeout: float) -> int:
    response = request_json(
        base,
        "/v1/messages/count_tokens",
        {"model": model, "messages": [{"role": "user", "content": text}]},
        timeout=timeout,
    )
    return int(response["input_tokens"])


def calibrate_prompt(
    base: str,
    model: str,
    target_tokens: int,
    identifier: str,
    nonce: str,
    timeout: float,
) -> tuple[str, int]:
    """Return the largest server-tokenized prompt at or below the target."""
    tail = (
        f"\nThe exact identifier is {identifier}.\n"
        f"Campaign nonce: {nonce}. Reply with {identifier} first, then one short sentence."
    )
    unit = " archival evidence"
    if count_tokens(base, model, tail, timeout) > target_tokens:
        raise ValueError(f"target {target_tokens} is smaller than request framing")
    low, high = 0, target_tokens
    while low < high:
        middle = (low + high + 1) // 2
        candidate = unit * middle + tail
        if count_tokens(base, model, candidate, timeout) <= target_tokens:
            low = middle
        else:
            high = middle - 1
    text = unit * low + tail
    return text, count_tokens(base, model, text, timeout)


def counter_delta(initial: dict, final: dict) -> dict[str, int]:
    initial_counts = initial.get("counts") or {}
    final_counts = final.get("counts") or {}
    keys = set(initial_counts) | set(final_counts)
    return {
        key: int(final_counts.get(key, 0) or 0)
        - int(initial_counts.get(key, 0) or 0)
        for key in sorted(keys)
    }


def candidate_delta_engaged(initial: dict, final: dict) -> bool:
    delta = counter_delta(initial, final)
    return (
        batch_row_exact_q4_engaged(final)
        and delta.get("kernel", 0) > 0
        and delta.get("group_kernel", 0) > 0
        and delta.get("complete_forwards", 0) > 0
        and delta.get("started_forwards", 0) == delta.get("complete_forwards", 0)
        and delta.get("per_row", 0) == 0
        and delta.get("refusals", 0) == 0
        and all(value >= 0 for value in delta.values())
    )


def response_record(response: dict) -> dict:
    value = content(response)
    return {
        "content": value,
        "content_sha256": hashlib.sha256(value.encode()).hexdigest(),
        "token_ids": token_ids(response),
        "receipt": receipt(response),
    }


def evaluate_cell(
    *,
    identifiers: list[str],
    prime: list[dict],
    b1: list[dict],
    b4: list[dict],
    max_tokens: int,
) -> dict:
    if not all(len(rows) == WIDTH for rows in (prime, b1, b4)):
        return {"passed": False, "failure": "expected exactly four rows per phase"}
    b1_ids = [token_ids(row) for row in b1]
    b4_ids = [token_ids(row) for row in b4]
    b1_content = [content(row) for row in b1]
    b4_content = [content(row) for row in b4]
    b1_receipts = [receipt(row) for row in b1]
    b4_receipts = [receipt(row) for row in b4]
    checks = {
        "identifier_parity": all(
            identifier_present(left, identifier)
            and identifier_present(right, identifier)
            for identifier, left, right in zip(identifiers, b1_content, b4_content)
        ),
        "token_parity": b1_ids == b4_ids
        and all(len(ids) == max_tokens for ids in b1_ids + b4_ids),
        "text_parity": b1_content == b4_content,
        "b1_width": all(
            int(row.get("ordinary_compute_width") or 0) == 1 for row in b1_receipts
        ),
        "b4_width": all(
            int(row.get("ordinary_compute_width") or 0) == WIDTH
            for row in b4_receipts
        ),
        "apcv2": all(
            row.get("cache") == "apcv2" and int(row.get("cached_tokens") or 0) > 0
            for row in b1_receipts + b4_receipts
        ),
        "candidate_receipts": all(
            row.get("qualification") == "candidate"
            for row in b1_receipts + b4_receipts
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "prime": [response_record(row) for row in prime],
        "b1": [response_record(row) for row in b1],
        "b4": [response_record(row) for row in b4],
    }


def wait_quiescent(base: str, timeout: float = 10) -> dict:
    deadline = time.monotonic() + timeout
    status = request_json(base, "/v1/status", timeout=10)
    while (
        status.get("inflight") != 0
        or status.get("apcv2", {}).get("cow", {}).get("active_leases") != 0
    ) and time.monotonic() < deadline:
        time.sleep(0.1)
        status = request_json(base, "/v1/status", timeout=10)
    return status


def thermal_admission() -> dict:
    """Apply the same fair-or-nominal settled admission as thermal_ladder."""
    try:
        samples = thermal.matrix.stabilize_thermal(thermal.ADMISSION_POLICY)
    except TimeoutError as error:
        return {
            "samples": [thermal.matrix.sample_thermal()],
            "stable": False,
            "error": str(error),
            "admission_policy": thermal.ADMISSION_POLICY,
        }
    return {
        "samples": samples,
        "stable": True,
        "admission_policy": thermal.ADMISSION_POLICY,
        "interpretation": "ProcessInfo fair (1) or nominal (0); no throttling",
    }


def owned_process_tree(server_pid: int) -> set[int]:
    """Exclude the server, runner, their children, and the runner's ancestors."""
    return (
        thermal.process_tree({server_pid})
        | thermal.process_tree({os.getpid()})
        | thermal.ancestors(os.getpid())
    )


def begin_contamination_window(server_pid: int) -> dict:
    owned = owned_process_tree(server_pid)
    return {
        "owned_pids_before": sorted(owned),
        "foreign_before": thermal.foreign_snapshot(owned),
        "swapouts_before": thermal.swapouts(),
    }


def contamination_reasons(
    *,
    thermal_pre: dict,
    thermal_post: dict,
    swapouts_before: int,
    swapouts_after: int,
    foreign_activity: list[dict],
    swapout_tolerance_pages: int,
) -> list[str]:
    reasons = []
    if not thermal_pre.get("stable"):
        reasons.append("thermal admission failed before the cell")
    if thermal_post.get("breached"):
        reasons.append("two consecutive post-cell samples show throttling")
    if swapouts_before < 0 or swapouts_after < 0:
        reasons.append("swapout counter unavailable")
    elif swapouts_after - swapouts_before > swapout_tolerance_pages:
        reasons.append(
            f"swapouts rose by {swapouts_after - swapouts_before} pages"
        )
    if foreign_activity:
        reasons.append(f"foreign GPU-capable process active: {foreign_activity}")
    return reasons


def finish_contamination_window(
    start: dict,
    *,
    server_pid: int,
    foreign_cpu_threshold: float,
    swapout_tolerance_pages: int,
    thermal_pre: dict,
) -> dict:
    swapouts_after = thermal.swapouts()
    owned_after = owned_process_tree(server_pid)
    foreign_after = thermal.foreign_snapshot(owned_after)
    foreign = thermal.foreign_activity(
        start["foreign_before"], foreign_after, foreign_cpu_threshold
    )
    thermal_post = thermal.post_run_thermal()
    reasons = contamination_reasons(
        thermal_pre=thermal_pre,
        thermal_post=thermal_post,
        swapouts_before=start["swapouts_before"],
        swapouts_after=swapouts_after,
        foreign_activity=foreign,
        swapout_tolerance_pages=swapout_tolerance_pages,
    )
    return {
        "thermal_pre": thermal_pre,
        "thermal_post": thermal_post,
        "swapouts": {
            "before": start["swapouts_before"],
            "after": swapouts_after,
            "delta": swapouts_after - start["swapouts_before"],
            "tolerance_pages": swapout_tolerance_pages,
        },
        "process_contamination": {
            "foreign_cpu_threshold_seconds": foreign_cpu_threshold,
            "owned_pids_before": start["owned_pids_before"],
            "owned_pids_after": sorted(owned_after),
            "foreign_before": start["foreign_before"],
            "foreign_after": foreign_after,
            "foreign_activity": foreign,
        },
        "contamination": reasons,
        "contaminated": bool(reasons),
        "passed": not reasons,
    }


def run_gate(
    base: str,
    initial: dict,
    *,
    model: str,
    label: str,
    max_tokens: int,
    request_timeout: float,
    server_pid: int,
    foreign_cpu_threshold: float,
    swapout_tolerance_pages: int,
) -> dict:
    initial_route = initial.get("execution", {}).get("batch_row_exact_q4", {})
    startup_checks = {
        "candidate_selected": initial_route.get("selected") is True,
        "candidate_not_yet_observed": initial_route.get("observed_used") is False,
        "candidate_unqualified": initial_route.get("qualified") is False,
        "qualification_candidate": initial.get("qualification") == "candidate",
        "apcv2_layout_revision": str(
            initial.get("apcv2", {}).get("layout_name") or ""
        ).endswith(LAYOUT_SUFFIX),
        "ordinary_route": initial.get("settings", {}).get("route") == "ordinary",
        "policy_exact": initial.get("settings", {})
        .get("execution_policy", {})
        .get("batch_row_exact_q4", {})
        .get("algorithm")
        == "north-batch-row-exact-q4-v1",
    }
    if not all(startup_checks.values()):
        return {
            "schema": "mlx2.north-row-exact-context-gate.v1",
            "passed": False,
            "startup_checks": startup_checks,
            "initial_status": initial,
            "failure": "candidate startup contract not satisfied",
        }

    cells = []
    for requested in REQUESTED_CONTEXTS:
        headroom = max(max_tokens + 128, 256)
        effective = min(requested, MAX_CONTEXT - headroom)
        prompts: list[str] = []
        identifiers: list[str] = []
        prompt_records = []
        for lane in range(WIDTH):
            digest = hashlib.sha256(
                f"{label}:{requested}:{lane}".encode()
            ).hexdigest()[:10].upper()
            identifier = f"NEEDLE-{digest}-{lane}"
            text, actual = calibrate_prompt(
                base,
                model,
                effective,
                identifier,
                f"{label}-{requested}-{lane}",
                request_timeout,
            )
            prompts.append(text)
            identifiers.append(identifier)
            prompt_records.append({
                "lane": lane,
                "identifier": identifier,
                "requested_tokens": requested,
                "effective_target_tokens": effective,
                "actual_tokens": actual,
                "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            })
        bodies = [
            prompt(
                text,
                reasoning_effort="none",
                think=False,
                logprobs=True,
                min_tokens=max_tokens,
                max_tokens=max_tokens,
            )
            for text in prompts
        ]
        thermal_pre = thermal_admission()
        if not thermal_pre["stable"]:
            cells.append({
                "prompts": prompt_records,
                "passed": False,
                "failure": "thermal admission failed before the cell",
                "environment": {
                    "thermal_pre": thermal_pre,
                    "contamination": ["thermal admission failed before the cell"],
                    "contaminated": True,
                    "passed": False,
                },
            })
            continue
        environment_start = begin_contamination_window(server_pid)
        prime = [chat(base, body, timeout=request_timeout) for body in bodies]
        b1 = [chat(base, body, timeout=request_timeout) for body in bodies]
        with ThreadPoolExecutor(max_workers=WIDTH) as pool:
            b4 = list(pool.map(lambda body: chat(base, body, timeout=request_timeout), bodies))
        environment = finish_contamination_window(
            environment_start,
            server_pid=server_pid,
            foreign_cpu_threshold=foreign_cpu_threshold,
            swapout_tolerance_pages=swapout_tolerance_pages,
            thermal_pre=thermal_pre,
        )
        evaluated = evaluate_cell(
            identifiers=identifiers,
            prime=prime,
            b1=b1,
            b4=b4,
            max_tokens=max_tokens,
        )
        evaluated["passed"] = evaluated["passed"] and environment["passed"]
        cells.append({
            "prompts": prompt_records,
            "environment": environment,
            **evaluated,
        })

    final = wait_quiescent(base)
    final_route = final.get("execution", {}).get("batch_row_exact_q4", {})
    route_check = candidate_delta_engaged(initial_route, final_route)
    quiescent = (
        final.get("inflight") == 0
        and final.get("apcv2", {}).get("cow", {}).get("active_leases") == 0
    )
    return {
        "schema": "mlx2.north-row-exact-context-gate.v1",
        "semantics": "bounded candidate gate; not qualification or performance evidence",
        "startup_checks": startup_checks,
        "contexts": list(REQUESTED_CONTEXTS),
        "widths": {"reference": 1, "candidate": WIDTH},
        "thermal_policy": thermal.THERMAL_POLICY,
        "thermal_admission_policy": thermal.ADMISSION_POLICY,
        "foreign_cpu_threshold_seconds": foreign_cpu_threshold,
        "swapout_tolerance_pages": swapout_tolerance_pages,
        "cells": cells,
        "route_counter_delta": counter_delta(initial_route, final_route),
        "route_engaged": route_check,
        "quiescent": quiescent,
        "initial_status": initial,
        "final_status": final,
        "passed": all(startup_checks.values())
        and all(cell["passed"] for cell in cells)
        and route_check
        and quiescent,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--port", type=int, default=18958)
    parser.add_argument("--load-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--foreign-cpu-threshold",
        type=float,
        default=DEFAULT_FOREIGN_CPU_THRESHOLD,
    )
    parser.add_argument(
        "--swapout-tolerance-pages",
        type=int,
        default=DEFAULT_SWAPOUT_TOLERANCE_PAGES,
    )
    args = parser.parse_args(argv)
    if not 2 <= args.max_tokens <= 32:
        parser.error("--max-tokens must be 2..32")
    if args.foreign_cpu_threshold < 0 or args.swapout_tolerance_pages < 0:
        parser.error("contamination tolerances must be non-negative")

    model_path = args.model.expanduser().resolve(strict=True)
    out = args.out_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"refusing to overwrite nonempty evidence directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    owner = validate_ownership(args.session, args.label)
    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            parser.error(f"port {args.port} is already occupied")

    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked_dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
    )
    campaign = {
        "schema": "mlx2.north-row-exact-context-campaign.v1",
        "status": "running",
        "semantics": "bounded candidate gate; not qualification or performance evidence",
        "source_revision": revision,
        "source_tracked_dirty": tracked_dirty,
        "model": str(model_path),
        "owner": owner,
        "contexts": list(REQUESTED_CONTEXTS),
        "reference_width": 1,
        "candidate_width": WIDTH,
        "started_at": time.time(),
    }
    atomic_json(out / "campaign.json", campaign)
    atomic_json(out / "execution-policy.json", POLICY)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX2_CAMPAIGN_ROOT": str(ROOT),
    }
    server_command = [
        sys.executable,
        "-u",
        "-m",
        "mlx2.server",
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--ordinary",
        "--qualification-mode",
        "--max-context",
        str(MAX_CONTEXT),
        "--max-lanes",
        str(WIDTH),
        "--max-inflight",
        str(WIDTH),
        "--cache-dir",
        str(out / "apcv2-cache"),
        "--execution-policy",
        str(out / "execution-policy.json"),
    ]
    server = None
    report = None
    try:
        with (out / "server.log").open("w") as log:
            server = subprocess.Popen(
                server_command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            base = f"http://127.0.0.1:{args.port}"
            initial = wait_health(base, server, args.load_timeout)
            atomic_json(out / "server-startup.json", {
                "command": server_command,
                "pid": server.pid,
                "status": initial,
            })
            report = run_gate(
                base,
                initial,
                model=model_path.name,
                label=args.label,
                max_tokens=args.max_tokens,
                request_timeout=args.request_timeout,
                server_pid=server.pid,
                foreign_cpu_threshold=args.foreign_cpu_threshold,
                swapout_tolerance_pages=args.swapout_tolerance_pages,
            )
            atomic_json(out / "context-gate.json", report)
            campaign["status"] = "passed" if report["passed"] else "failed"
    except BaseException as error:
        campaign["status"] = "failed"
        campaign["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop_server(server)
        campaign["server_stopped"] = server is None or server.poll() is not None
        campaign["finished_at"] = time.time()
        campaign["locks_still_owned_before_return"] = (
            validate_ownership(args.session, args.label) == owner
        )
        atomic_json(out / "campaign.json", campaign)
    return 0 if report is not None and report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
