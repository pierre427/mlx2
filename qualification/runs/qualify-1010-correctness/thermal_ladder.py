#!/usr/bin/env python3
"""Thermally controlled context ladder: 3 measured runs per cell, warmed first.

Combines the campaign ladder (server-tokenizer calibration, needle recall,
cold then warm APCv2 request, width 1 and 4) with the thermal admission rule
of ``scripts/run_qualification_matrix.py`` (``stabilize_thermal`` before every
measured run, ``post_thermal_samples`` after it, the same policy block as
the portable ``thermal-policy.json`` beside this producer).

Differences from ``ladder.py`` (deliberate, recorded in every report):

* Every run's prompts start with a unique session header, so the "cold"
  request cannot reuse an earlier run's APCv2 prefix.  The old ladder shared
  one filler body across rungs and lanes, so its cold rows were partly warm.
  A cold request whose ``cached_tokens`` exceeds max(256 tokens, 1% of its prompt) is recorded
  as not cold and the run is repeated.
* One unmeasured warm-up run precedes the three measured runs of each cell.
* Requests set ``min_tokens == max_tokens`` so decode speed is measured over a
  fixed number of tokens instead of a ten-token needle answer; the needle
  must still appear in the content.  If the server rejects ``min_tokens`` the
  ladder falls back to plain requests and records it.
* Swap-outs (``vm_stat``) and CPU time of every foreign model/GPU process are
  sampled around each measured run.  A run during which swap-outs rose or a
  foreign GPU process was active is marked contaminated and repeated (at most
  ``--max-retries`` times); the contaminated attempt is kept in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ladder import Client, extract_peak_memory

ROOT = Path(os.environ.get("MLX2_CAMPAIGN_ROOT", str(HERE.parents[2]))).resolve()
sys.path.insert(0, str(ROOT / "scripts"))
import run_qualification_matrix as matrix

LENGTHS = (1024, 4096, 16384, 32768, 65536, 131072, 262144)
THERMAL_POLICY = json.loads((HERE / "thermal-policy.json").read_text())
# Safe execution admission accepts nominal/fair (0/1) with the configured
# temperature, settling, and pmset checks. Keep the measured state unmodified;
# nominal-only performance eligibility is evaluated separately below.
ADMISSION_POLICY = {
    **THERMAL_POLICY,
    "required_thermal_state": 0,
    "accepted_thermal_states": [0, 1],
}
_matrix_thermally_stable = matrix.thermally_stable


def _admit(sample, policy):
    if policy is ADMISSION_POLICY:
        state = int(sample.get("thermal_state", 3))
        if state not in policy["accepted_thermal_states"]:
            return False
        return _matrix_thermally_stable(
            sample, {**policy, "required_thermal_state": state}
        )
    return _matrix_thermally_stable(sample, policy)


matrix.thermally_stable = _admit
FOREIGN = re.compile(
    r"(mlx2\.(?:server|decisions\.server)(?:\s|$)|mlx_lm|mlx_vlm|mlx-lm|rapid-mlx|ltx-2-mlx|llama-server|ollama|"
    r"--i-own-the-gpu|cpg_job|scripts/(probe|bench|qualify|gpu_|run_))",
    re.IGNORECASE,
)


def ancestors(pid: int) -> set[int]:
    found = set()
    while pid > 1 and pid not in found:
        found.add(pid)
        out = subprocess.run(
            ["/bin/ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        pid = int(out) if out else 0
    return found


def swapouts() -> int:
    text = subprocess.run(
        ["/usr/bin/vm_stat"], capture_output=True, text=True, check=False
    ).stdout
    match = re.search(r"Swapouts:\s+(\d+)", text)
    return int(match.group(1)) if match else -1


def cpu_seconds(value: str) -> float:
    days = 0
    if "-" in value:
        day, value = value.split("-", 1)
        days = int(day)
    parts = [float(part) for part in value.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def process_tree(root_pids: set[int]) -> set[int]:
    rows = subprocess.run(
        ["/bin/ps", "-Ao", "pid=,ppid="], capture_output=True, text=True, check=False
    ).stdout.split("\n")
    children: dict[int, list[int]] = {}
    for row in rows:
        fields = row.split()
        if len(fields) == 2:
            children.setdefault(int(fields[1]), []).append(int(fields[0]))
    seen, stack = set(), list(root_pids)
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
    return seen


def foreign_snapshot(own: set[int]) -> dict[int, dict]:
    rows = subprocess.run(
        ["/bin/ps", "-Ao", "pid=,time=,command="],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    found = {}
    for row in rows.split("\n"):
        fields = row.strip().split(None, 2)
        if len(fields) < 3:
            continue
        pid = int(fields[0])
        if pid in own or not FOREIGN.search(fields[2]):
            continue
        found[pid] = {"cpu_seconds": cpu_seconds(fields[1]), "command": fields[2][:240]}
    return found


def foreign_activity(
    before: dict[int, dict], after: dict[int, dict], threshold: float
) -> list[dict]:
    active = []
    for pid, row in after.items():
        start = (
            before.get(pid, {"cpu_seconds": 0.0})["cpu_seconds"]
            if pid in before
            else 0.0
        )
        delta = row["cpu_seconds"] - start
        if delta > threshold:
            active.append(
                {
                    "pid": pid,
                    "cpu_seconds_delta": round(delta, 2),
                    "new": pid not in before,
                    "command": row["command"],
                }
            )
    return active


def lock_owner() -> dict | None:
    """Return the paired-lock receipt, or fail closed on missing/drifted locks."""
    receipts = []
    for lock in (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock")):
        try:
            receipts.append(json.loads((lock / "owner.json").read_text()))
        except (OSError, json.JSONDecodeError):
            return None
    return receipts[0] if receipts[0] == receipts[1] else None


def throttled(sample: dict) -> bool:
    """Throttling evidence, as opposed to the heat the measured work produced.

    The pre-run admission keeps the matrix policy (thermal state 0, settled
    temperatures).  After a run, ProcessInfo "fair" (1) is the expected
    consequence of a long prefill; only "serious"/"critical" (>= 2), a pmset
    thermal/performance/CPU-power warning, or a CPU speed limit below 100
    means the run was throttled.
    """
    return (
        int(sample.get("thermal_state", 3)) >= 2
        or sample.get("pmset_no_thermal_warning") is not True
        or sample.get("pmset_no_performance_warning") is not True
        or sample.get("pmset_no_cpu_power_warning") is not True
        or int(sample.get("cpu_speed_limit", 100)) < 100
    )


def post_run_thermal() -> dict:
    """Two consecutive throttled samples invalidate the run (overnight rule)."""
    samples = [matrix.sample_thermal()]
    if throttled(samples[0]):
        time.sleep(15)
        samples.append(matrix.sample_thermal())
    for row in samples:
        row.pop("raw_evidence", None)
    return {
        "samples": samples,
        "breached": len(samples) == 2 and all(throttled(row) for row in samples),
        "max_state": max(int(row.get("thermal_state", 0)) for row in samples),
        "rule": "invalid only if two consecutive samples show throttling (state >= 2 or pmset warning)",
    }


def build_prompt(filler_units: int, head: str, needle: str) -> str:
    return (
        f"{head}\n"
        + " archival evidence" * filler_units
        + f"\nThe unique fact is: the launch code is {needle}.\n"
        f"What is the launch code? Reply with {needle} first."
    )


def calibrate(
    client: Client, target: int, head: str, needle: str, guess: int | None
) -> tuple[str, int, int]:
    """Largest filler count whose rendered prompt is <= target tokens."""
    low, high = 0, target
    if guess is not None:
        low, high = max(0, guess - 96), min(target, guess + 96)
        if client.count(build_prompt(low, head, needle)) > target:
            low = 0
        if client.count(build_prompt(high, head, needle)) <= target:
            high = target
    while low < high:
        middle = (low + high + 1) // 2
        if client.count(build_prompt(middle, head, needle)) <= target:
            low = middle
        else:
            high = middle - 1
    text = build_prompt(low, head, needle)
    return text, client.count(text), low


class Stream(Client):
    """ladder.Client with min_tokens and server receipts."""

    min_tokens_supported = True

    def request(self, text: str, max_tokens: int) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "enable_thinking": False,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.min_tokens_supported:
            body["min_tokens"] = max_tokens
        import urllib.request

        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started, first = time.monotonic(), None
        content, usage, receipt, done = [], {}, {}, False
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                event = json.loads(payload)
                usage = event.get("usage") or usage
                receipt = event.get("mlx2") or receipt
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    piece = (delta.get("reasoning_content") or "") + (
                        delta.get("content") or ""
                    )
                    if piece and first is None:
                        first = time.monotonic()
                    content.append(delta.get("content") or "")
        wall = time.monotonic() - started
        client_ttft = (first - started) if first else wall
        prompt_tokens = int(
            usage.get("prompt_tokens") or receipt.get("prompt_tokens") or 0
        )
        completion = int(
            usage.get("completion_tokens") or receipt.get("completion_tokens") or 0
        )
        cached = int(receipt.get("cached_tokens") or 0)
        server_ttft = receipt.get("ttft_seconds")
        elapsed = receipt.get("elapsed_seconds")
        ttft = server_ttft if isinstance(server_ttft, (int, float)) else client_ttft
        decode_window = (
            (elapsed - server_ttft)
            if isinstance(elapsed, (int, float))
            and isinstance(server_ttft, (int, float))
            else (wall - client_ttft)
        )
        return {
            "done": done,
            "content": "".join(content),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "cached_tokens": cached,
            "client_ttft_seconds": client_ttft,
            "server_ttft_seconds": server_ttft,
            "server_elapsed_seconds": elapsed,
            "wall_seconds": wall,
            "prefill_tokens_per_second": ((prompt_tokens - cached) / ttft)
            if ttft and ttft > 0
            else None,
            "decode_tokens_per_second": ((completion - 1) / decode_window)
            if completion > 1 and decode_window > 0
            else None,
            "receipt": {
                key: receipt.get(key)
                for key in (
                    "cache",
                    "profile",
                    "route",
                    "cached_tokens",
                    "ordinary_compute_width",
                    "speculation",
                    "qualification",
                )
            }
            | {
                "mtp": {
                    key: (receipt.get("mtp") or {}).get(key)
                    for key in (
                        "route",
                        "num_draft",
                        "acceptance_rate",
                        "observed_compute_widths",
                    )
                }
                if receipt.get("mtp")
                else None
            },
        }

    def batch(self, prompts: list[str], max_tokens: int) -> dict:
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            rows = list(
                pool.map(
                    lambda text: self._guarded(lambda: self.request(text, max_tokens)),
                    prompts,
                )
            )
        elapsed = time.monotonic() - started
        total_completion = sum(row["completion_tokens"] for row in rows)
        return {
            "rows": rows,
            "elapsed_seconds": elapsed,
            "aggregate_completion_tokens_per_second": total_completion / elapsed
            if elapsed > 0
            else None,
        }


# A warm request is an APCv2 hit only when it reused (nearly) its whole prompt.
# ``cached_tokens > 0`` also counted a warm request that reused only the chat
# template's preamble (~54 of 32,000 tokens) as a hit.  Served warm rows reuse
# all but the last token (0.999 at 1K, 1.000 at 64K).
WARM_HIT_MIN_FRACTION = 0.9


def warm_hit(row) -> bool:
    return row["cached_tokens"] >= WARM_HIT_MIN_FRACTION * max(row["prompt_tokens"], 1)


def median(values):
    values = [
        value
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    return statistics.median(values) if values else None


def summarize_run(cold: dict, warm: dict, needles: list[str]) -> dict:
    lanes = len(needles)
    cold_rows, warm_rows = cold["rows"], warm["rows"]
    return {
        "ttft_seconds": median(
            [
                row["server_ttft_seconds"] or row["client_ttft_seconds"]
                for row in cold_rows
            ]
        ),
        "warm_ttft_seconds": median(
            [
                row["server_ttft_seconds"] or row["client_ttft_seconds"]
                for row in warm_rows
            ]
        ),
        "prefill_tokens_per_second": median(
            [row["prefill_tokens_per_second"] for row in cold_rows]
        ),
        "aggregate_prefill_tokens_per_second": (
            sum(row["prompt_tokens"] - row["cached_tokens"] for row in cold_rows)
            / max(
                row["server_ttft_seconds"] or row["client_ttft_seconds"]
                for row in cold_rows
            )
        )
        if lanes > 1
        else None,
        "decode_tokens_per_second": median(
            [row["decode_tokens_per_second"] for row in cold_rows]
        ),
        "warm_decode_tokens_per_second": median(
            [row["decode_tokens_per_second"] for row in warm_rows]
        ),
        "aggregate_decode_tokens_per_second": cold[
            "aggregate_completion_tokens_per_second"
        ],
        "needle_correct": sum(
            needle in row["content"] for row, needle in zip(cold_rows, needles)
        )
        + sum(needle in row["content"] for row, needle in zip(warm_rows, needles)),
        "needle_total": 2 * lanes,
        "streams_done": all(row["done"] for row in cold_rows + warm_rows),
        "cold_cached_max_tokens": max(row["cached_tokens"] for row in cold_rows),
        "cold_cached_max_fraction": max(
            row["cached_tokens"] / max(row["prompt_tokens"], 1) for row in cold_rows
        ),
        "warm_apc_hits": sum(warm_hit(row) for row in warm_rows),
        "warm_hit_min_fraction": WARM_HIT_MIN_FRACTION,
        "warm_cached_min_fraction": min(
            (row["cached_tokens"] / max(row["prompt_tokens"], 1) for row in warm_rows),
            default=None,
        ),
        "warm_equals_cold": sum(
            a["content"] == b["content"] for a, b in zip(cold_rows, warm_rows)
        ),
        "prompt_tokens": [row["prompt_tokens"] for row in cold_rows],
        "completion_tokens": [row["completion_tokens"] for row in cold_rows],
    }


def spread(values):
    values = [
        value
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    if not values:
        return None
    mid = statistics.median(values)
    return {
        "median": mid,
        "min": min(values),
        "max": max(values),
        "spread_pct": (100.0 * (max(values) - min(values)) / mid) if mid else None,
    }


def validate_serving_profile(
    status: dict,
    *,
    expected_route: str,
    expected_artifact: str,
    max_context: int,
    cache_bytes: int,
    max_lanes: int,
    max_inflight: int,
    prefill_step: int,
    apc_persistence: bool,
    apc_persist_on_shutdown: bool,
    mtp_policy,
    draft_loop_policy: dict,
    prefill_policy: dict,
):
    """Require the live server to match the declared, bounded cell profile."""
    settings = status.get("settings") if isinstance(status, dict) else None
    if not isinstance(settings, dict):
        raise TypeError("/v1/status is missing serving settings")
    if not isinstance(draft_loop_policy, dict) or not {
        "draft_loop",
        "draft_loop_threshold",
        "draft_loop_widths",
    } <= set(draft_loop_policy):
        raise ValueError("draft-loop policy must declare mode, threshold, and widths")
    if not isinstance(prefill_policy, dict) or set(prefill_policy) != {
        "prefill_depth_budget",
        "prefill_step_autoscale",
    }:
        raise ValueError(
            "prefill policy must declare prefill_depth_budget and prefill_step_autoscale"
        )
    route_receipt = status.get("route_receipt")
    actual_route = settings.get("route")
    if actual_route is None and isinstance(route_receipt, dict):
        actual_route = route_receipt.get("route")
    identity_mismatches = {}
    if actual_route != expected_route:
        identity_mismatches["route"] = {
            "expected": expected_route,
            "actual": actual_route,
        }
    if status.get("artifact") != expected_artifact:
        identity_mismatches["artifact"] = {
            "expected": expected_artifact,
            "actual": status.get("artifact"),
        }
    if status.get("max_context") != max_context:
        identity_mismatches["max_context"] = {
            "expected": max_context,
            "actual": status.get("max_context"),
        }
    if identity_mismatches:
        raise ValueError(f"live route/artifact/context mismatch: {identity_mismatches}")
    expected = {
        "cache_bytes": cache_bytes,
        "max_lanes": max_lanes,
        "max_inflight": max_inflight,
        "prefill_step": prefill_step,
        "apc_persistence": apc_persistence,
        "apc_persist_on_shutdown": apc_persist_on_shutdown,
        "mtp": mtp_policy,
        **draft_loop_policy,
        **prefill_policy,
    }
    mismatches = {
        key: {"expected": value, "actual": settings.get(key)}
        for key, value in expected.items()
        if settings.get(key) != value
    }
    if mismatches:
        raise ValueError(f"live serving profile mismatch: {mismatches}")
    if max_lanes < 1 or max_inflight < max_lanes or prefill_step < 1 or cache_bytes < 1:
        raise ValueError(
            "serving profile bounds must be positive and inflight >= lanes"
        )
    return {
        **expected,
        "host": status.get("host"),
        "model": status.get("model"),
        "route": actual_route,
        "artifact": status.get("artifact"),
        "max_context": max_context,
    }


def json_arg(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"expected JSON: {error}") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--artifact-identity", required=True)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--cache-bytes", type=int, required=True)
    parser.add_argument("--max-lanes", type=int, required=True)
    parser.add_argument("--max-inflight", type=int, required=True)
    parser.add_argument("--prefill-step", type=int, required=True)
    parser.add_argument("--apc-persistence", choices=("on", "off"), required=True)
    parser.add_argument(
        "--apc-persist-on-shutdown", choices=("on", "off"), required=True
    )
    parser.add_argument(
        "--mtp-policy",
        type=json_arg,
        required=True,
        help="JSON value expected at /v1/status settings.mtp",
    )
    parser.add_argument(
        "--draft-loop-policy",
        type=json_arg,
        required=True,
        help="JSON object with draft_loop, draft_loop_threshold, draft_loop_widths expected in live settings",
    )
    parser.add_argument(
        "--prefill-policy",
        type=json_arg,
        required=True,
        help="JSON object with prefill_depth_budget and prefill_step_autoscale",
    )
    parser.add_argument("--server-pid", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--performance-mode",
        action="store_true",
        help="require nominal thermal state before every measured run",
    )
    parser.add_argument("--wide", type=int, default=4)
    parser.add_argument("--wide-max-context", type=int, default=32768)
    parser.add_argument("--max-length", type=int, default=262144)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--foreign-cpu-threshold", type=float, default=2.0)
    parser.add_argument("--swapout-tolerance-pages", type=int, default=0)
    args = parser.parse_args(argv)
    if args.runs != 3:
        parser.error(
            "qualification ladders require exactly three measured repetitions per cell"
        )
    ADMISSION_POLICY["accepted_thermal_states"] = (
        [0] if args.performance_mode else [0, 1]
    )
    if (
        args.cache_bytes < 1
        or args.max_lanes < 1
        or args.max_inflight < args.max_lanes
        or args.prefill_step < 1
        or not 2 <= args.wide <= args.max_lanes
    ):
        parser.error(
            "profile bounds must be positive, inflight >= lanes, and 2 <= wide <= max-lanes"
        )
    client = Stream(args.url, args.timeout, args.model_id)
    if not args.server_pid:
        from urllib.parse import urlparse

        port = urlparse(args.url).port
        listing = subprocess.run(
            ["/usr/sbin/lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        args.server_pid = int(listing[0]) if listing else 0
    try:
        report_status = client.status()
        serving_profile = validate_serving_profile(
            report_status,
            expected_route=args.route,
            expected_artifact=args.artifact_identity,
            max_context=args.max_context,
            cache_bytes=args.cache_bytes,
            max_lanes=args.max_lanes,
            max_inflight=args.max_inflight,
            prefill_step=args.prefill_step,
            apc_persistence=args.apc_persistence == "on",
            mtp_policy=args.mtp_policy,
            draft_loop_policy=args.draft_loop_policy,
            prefill_policy=args.prefill_policy,
        )
    except ValueError as error:
        parser.error(str(error))
    if not report_status.get("healthy") or not report_status.get("apcv2"):
        parser.error(
            "server must be healthy and publish APCv2 status before the ladder"
        )
    own_roots = {args.server_pid} if args.server_pid else set()
    own_fixed = ancestors(os.getpid())
    expected_cells = [
        {"requested_tokens": requested, "width": width}
        for requested in LENGTHS
        if args.min_length <= requested <= min(args.max_context, args.max_length)
        for width in (1, args.wide)
        if width == 1 or requested <= args.wide_max_context
    ]
    report = {
        "schema": "mlx2.thermal-context-ladder.v1",
        "model": args.model,
        "route": args.route,
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "max_context": args.max_context,
        "host": platform.node(),
        "server_pid": args.server_pid,
        "runs_per_cell": args.runs,
        "max_tokens": args.max_tokens,
        "thermal_policy": THERMAL_POLICY,
        "thermal_policy_sha256": hashlib.sha256(
            (HERE / "thermal-policy.json").read_bytes()
        ).hexdigest(),
        "admission_policy": ADMISSION_POLICY,
        "performance_mode": args.performance_mode,
        "expected_cells": expected_cells,
        "serving_profile": serving_profile,
        "settings": report_status.get("settings"),
        "root": str(ROOT),
        "design": __doc__,
        "started_at": time.time(),
        "initial": client.status(),
        "cells": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(args.output)

    guesses: dict[int, int] = {}

    def make_prompts(effective: int, width: int, tag: str):
        prompts, needles, counts = [], [], []
        for lane in range(width):
            nonce = hashlib.sha256(
                f"{args.model}-{args.route}-{tag}-{lane}-{time.time_ns()}".encode()
            ).hexdigest()[:12]
            head = f"Session {nonce} lane {lane}. Read the archive and answer the question at the end."
            needle = f"NEEDLE-{nonce[:6].upper()}-{lane}"
            text, actual, units = calibrate(
                client, effective, head, needle, guesses.get(effective)
            )
            guesses[effective] = units
            prompts.append(text)
            needles.append(needle)
            counts.append(actual)
        return prompts, needles, counts

    def one_run(effective: int, width: int, tag: str, measured: bool) -> dict:
        thermal_pre = None
        if measured:
            try:
                thermal_pre = {
                    "samples": matrix.stabilize_thermal(ADMISSION_POLICY),
                    "stable": True,
                    "safe_admitted": True,
                    "admission_policy": ADMISSION_POLICY,
                }
            except TimeoutError as error:
                thermal_pre = {
                    "samples": [matrix.sample_thermal()],
                    "stable": False,
                    "safe_admitted": False,
                    "error": str(error),
                }
                return {
                    "tag": tag,
                    "measured": True,
                    "started_at": time.time(),
                    "finished_at": time.time(),
                    "width": width,
                    "thermal_pre": thermal_pre,
                    "pending_admission": True,
                    "contaminated": False,
                    "contamination": [],
                }
        prompts, needles, counts = make_prompts(effective, width, tag)
        own = process_tree(own_roots) | own_fixed | process_tree({os.getpid()})
        foreign_before = foreign_snapshot(own)
        swap_before, swap_used_before = swapouts(), matrix.sample_swap()
        owner_before = lock_owner()
        started = time.time()
        refusals = []

        def batch_with_refusal_retry(label):
            # A 429 is the server's memory-admission refusal.  Record its body,
            # wait past the APCv2 idle-disk window (180 s) once, and retry.
            import urllib.error

            for attempt in range(2):
                try:
                    return client.batch(prompts, args.max_tokens)
                except urllib.error.HTTPError as error:
                    if error.code != 429 or attempt:
                        raise
                    try:
                        body = error.read().decode(errors="replace")[:2000]
                    except Exception:  # noqa: BLE001
                        body = None
                    refusals.append(
                        {
                            "phase": label,
                            "t": time.time(),
                            "body": body,
                            "status": client.status(),
                        }
                    )
                    print(f"  429 on {label}: {body}", flush=True)
                    time.sleep(200)

        cold = batch_with_refusal_retry("cold")
        warm = batch_with_refusal_retry("warm")
        finished = time.time()
        swap_after, swap_used_after = swapouts(), matrix.sample_swap()
        own = process_tree(own_roots) | own_fixed | process_tree({os.getpid()})
        foreign = foreign_activity(
            foreign_before, foreign_snapshot(own), args.foreign_cpu_threshold
        )
        owner_after = lock_owner()
        status = client.status()
        peak_path, peak_bytes = extract_peak_memory(status)
        thermal_post = None
        if measured:
            thermal_post = post_run_thermal()
        summary = summarize_run(cold, warm, needles)
        contamination = []
        functional_failures = []
        if swap_after - swap_before > args.swapout_tolerance_pages:
            contamination.append(f"swapouts rose by {swap_after - swap_before} pages")
        if foreign:
            contamination.append(f"foreign GPU-capable process active: {foreign}")
        # A chat template's fixed preamble (Muse: ~54 tokens) is legitimately
        # shared by every prompt; only reuse beyond it means "not cold".
        if summary["cold_cached_max_tokens"] > max(
            256, 0.01 * min(summary["prompt_tokens"] or [0])
        ):
            message = f"cold request reused {summary['cold_cached_max_fraction']:.3f} of its prompt"
            contamination.append(message)
            functional_failures.append(message)
        if measured and thermal_post and thermal_post["breached"]:
            contamination.append(
                "throttling observed in two consecutive post-run samples"
            )
        return {
            "tag": tag,
            "measured": measured,
            "started_at": started,
            "finished_at": finished,
            "effective_target_tokens": effective,
            "width": width,
            "calibrated_tokens": counts,
            "needles": needles,
            "summary": summary,
            "metal_peak_bytes": status.get("metal_peak_bytes", peak_bytes),
            "peak_path": peak_path,
            "metal_active_bytes": status.get("metal_active_bytes"),
            "process_physical_footprint_bytes": status.get(
                "process_physical_footprint_bytes"
            ),
            "thermal_pre": thermal_pre,
            "thermal_post": thermal_post,
            "swapouts": {
                "before": swap_before,
                "after": swap_after,
                "delta": swap_after - swap_before,
            },
            "swap_used_bytes": {
                "before": swap_used_before["used_bytes"],
                "after": swap_used_after["used_bytes"],
            },
            "gpu_lock_owner": {"before": owner_before, "after": owner_after},
            "refusals_429": refusals,
            "foreign_activity": foreign,
            "loadavg_after": os.getloadavg(),
            "contamination": contamination,
            "contaminated": bool(contamination),
            "functional_failures": functional_failures,
            "requests": {"cold": cold, "warm": warm},
        }

    # Probe min_tokens support once with a tiny request.
    try:
        client.request("Say OK.", 4)
    except Exception as error:  # noqa: BLE001 - recorded, fall back
        report["min_tokens_probe_error"] = f"{type(error).__name__}: {error}"[:500]
        Stream.min_tokens_supported = False
    report["min_tokens_supported"] = Stream.min_tokens_supported
    save()

    headroom = max(args.max_tokens + 128, 256)
    for requested in LENGTHS:
        if (
            requested > min(args.max_context, args.max_length)
            or requested < args.min_length
        ):
            continue
        effective = min(requested, args.max_context - headroom)
        for width in (1, args.wide):
            if width > 1 and requested > args.wide_max_context:
                continue
            cell = {
                "requested_tokens": requested,
                "effective_target_tokens": effective,
                "width": width,
                "warmup": None,
                "runs": [],
                "attempts": [],
                "error": None,
            }
            report["cells"].append(cell)
            print(f"CELL {requested} w{width} start {time.strftime('%T')}", flush=True)
            try:
                cell["warmup"] = one_run(
                    effective, width, f"{requested}-w{width}-warmup", False
                )
                save()
                for run in range(args.runs):
                    for attempt in range(args.max_retries + 1):
                        row = one_run(
                            effective,
                            width,
                            f"{requested}-w{width}-run{run}-a{attempt}",
                            True,
                        )
                        if row.get("pending_admission"):
                            row["run"] = run
                            row["attempt"] = attempt
                            cell["attempts"].append(row)
                            report["pending_admission"] = {
                                "cell": {"requested_tokens": requested, "width": width},
                                "run": run,
                                "attempt": attempt,
                                "reason": row.get("thermal_pre", {}).get("error"),
                            }
                            save()
                            break
                        if not row["contaminated"] or attempt == args.max_retries:
                            row["run"] = run
                            row["attempt"] = attempt
                            cell["runs"].append(row)
                            break
                        cell["attempts"].append(row)
                        print(
                            f"  run {run} attempt {attempt} contaminated: {row['contamination']}",
                            flush=True,
                        )
                        save()
                    save()
                    if report.get("pending_admission"):
                        break
                    summary = cell["runs"][-1]["summary"]
                    print(
                        f"  run {run}: ttft={summary['ttft_seconds']:.3f}s prefill={summary['prefill_tokens_per_second'] or 0:.0f} "
                        f"decode={summary['decode_tokens_per_second'] or 0:.1f} needle={summary['needle_correct']}/{summary['needle_total']} "
                        f"contaminated={cell['runs'][-1]['contaminated']}",
                        flush=True,
                    )
            except Exception as error:  # noqa: BLE001 - one cell fails, the ladder continues
                cell["error"] = f"{type(error).__name__}: {error}"[:2000]
                print(f"  CELL ERROR {cell['error']}", flush=True)
            runs = cell["runs"]
            cell["stats"] = {
                key: spread([row["summary"][key] for row in runs])
                for key in (
                    "ttft_seconds",
                    "warm_ttft_seconds",
                    "prefill_tokens_per_second",
                    "aggregate_prefill_tokens_per_second",
                    "decode_tokens_per_second",
                    "warm_decode_tokens_per_second",
                    "aggregate_decode_tokens_per_second",
                )
            }
            cell["stats"]["metal_peak_bytes"] = spread(
                [row["metal_peak_bytes"] for row in runs]
            )
            cell["passed"] = (
                cell["error"] is None
                and len(runs) == args.runs
                and all(row["summary"]["streams_done"] for row in runs)
                and all(
                    row["summary"]["needle_correct"] == row["summary"]["needle_total"]
                    for row in runs
                )
                and all(row["summary"]["warm_apc_hits"] == width for row in runs)
                and all(row["summary"]["warm_equals_cold"] == width for row in runs)
                and all(not row.get("functional_failures") for row in runs)
            )
            cell["quality"] = (
                f"{sum(row['summary']['needle_correct'] for row in runs)}/"
                f"{sum(row['summary']['needle_total'] for row in runs)} needles"
            )
            save()
            if report.get("pending_admission"):
                break
        if report.get("pending_admission"):
            break
    report["final"] = client.status()
    report["finished_at"] = time.time()
    if report.get("pending_admission"):
        report["status"] = "pending_admission"
        report["passed"] = False
        report["performance_eligible"] = False
        save()
        print(
            "SUMMARY status=pending_admission; no qualification failure recorded",
            flush=True,
        )
        return 2
    report["passed"] = bool(report["cells"]) and all(
        cell["passed"] for cell in report["cells"]
    )
    all_rows = [
        row
        for cell in report["cells"]
        for row in (cell.get("attempts", []) + cell.get("runs", []))
    ]
    report["performance_eligible"] = (
        report["passed"]
        and all(not row.get("contaminated") for row in all_rows)
        and all(
            row.get("measured") is not True
            or (
                (row.get("thermal_pre") or {}).get("safe_admitted") is True
                and bool((row.get("thermal_pre") or {}).get("samples"))
                and all(
                    int(sample.get("thermal_state", 3)) == 0
                    for sample in (row.get("thermal_pre") or {}).get("samples", [])
                )
            )
            for row in all_rows
        )
        and all(
            not ((row.get("thermal_post") or {}).get("breached")) for row in all_rows
        )
    )
    save()
    print(f"SUMMARY cells={len(report['cells'])} passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
