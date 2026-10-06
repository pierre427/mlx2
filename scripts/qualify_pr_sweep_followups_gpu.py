#!/usr/bin/env python3
"""Focused GPU evidence for the 2026-10-06 peer-PR follow-up slice.

This is a functional qualification harness, not a performance benchmark.  It
owns every server process group it starts but deliberately leaves GPU/CPG lock
ownership to the enclosing lab wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmark_adaptive_mtp import (
    _json_get,
    _stream_request,
    _wait_ready,
    stop_process_group,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mlx2.pr-sweep-followups-gpu.v1"


def _delta(after: dict, before: dict, key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _server_command(args, *, route: str, policy: Path | None) -> list[str]:
    command = [
        args.python,
        "-m",
        "mlx2.server",
        "--model",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-context",
        str(args.max_context),
        "--max-lanes",
        "1" if args.template_compatibility_smoke else "2",
        "--max-inflight",
        "2" if args.template_compatibility_smoke else "4",
        "--cache-bytes",
        str(args.cache_gib << 30),
        "--qualification-mode",
        "--admin-token-file",
        str(args.run_dir / "admin-token"),
    ]
    command.append("--native-mtp" if route == "native_mtp" else "--ordinary")
    if policy is not None:
        command.extend(("--execution-policy", str(policy)))
    if route == "native_mtp":
        command.append("--adaptive-mtp-depth")
    return command


def _run_server(args, *, route: str, callback):
    policy = None
    if route == "native_mtp":
        policy = args.run_dir / "adaptive-policy.json"
        policy.write_text(
            json.dumps(
                {
                    "num_draft": args.depth,
                    "mtp_ordinary_handoff": False,
                    "adaptive_mtp_depth": {"enabled": True},
                },
                indent=2,
            )
            + "\n"
        )
    command = _server_command(args, route=route, policy=policy)
    log_path = args.run_dir / f"{route}-server.log"
    inherited = os.environ.get("PYTHONPATH")
    pythonpath = [str(ROOT / "src")]
    if inherited:
        pythonpath.append(inherited)
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(pythonpath))
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    url = f"http://127.0.0.1:{args.port}"
    lifecycle = {"pid": process.pid, "command": command, "log": str(log_path)}
    try:
        initial = _wait_ready(process, url, args.startup_timeout)
        actual = (initial.get("settings") or {}).get("route")
        if actual != route:
            raise RuntimeError(f"server selected {actual!r}, expected {route!r}")
        return callback(url, initial), lifecycle
    finally:
        lifecycle["shutdown"] = stop_process_group(process, url)


def _ordinary_checks(args, url: str, initial: dict) -> dict:
    before = _json_get(url, "/v1/status")
    budget = _stream_request(
        url,
        "Reply with one deterministic token.",
        args.request_timeout,
        1,
    )

    shared = "cache-coherent deterministic prefix " * args.prefix_repetitions
    warm_prompt = shared + "Finish with the word alpha."
    warm_first = _stream_request(url, warm_prompt, args.request_timeout, 2)
    warm_second = _stream_request(url, warm_prompt, args.request_timeout, 2)
    variants = [
        shared + "Explain suffix branch A in one sentence.",
        shared + "Explain suffix branch B in one sentence.",
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        joined = list(
            pool.map(
                lambda prompt: _stream_request(
                    url, prompt, args.request_timeout, 2
                ),
                variants,
            )
        )
    time.sleep(1.25)
    after = _json_get(url, "/v1/status")
    before_apc = before.get("apcv2") or {}
    after_apc = after.get("apcv2") or {}
    before_cow = before_apc.get("cow") or {}
    after_cow = after_apc.get("cow") or {}
    before_host = before.get("host_prompt_cache") or {}
    after_host = after.get("host_prompt_cache") or {}

    budget_passed = (
        int((budget.get("usage") or {}).get("completion_tokens", -1)) == 1
        and bool(budget.get("receipt"))
    )
    host_delta = {
        "hits": _delta(after_host, before_host, "hits"),
        "misses": _delta(after_host, before_host, "misses"),
    }
    cow_delta = {
        key: _delta(after_cow, before_cow, key)
        for key in (
            "branches",
            "releases",
            "peak_leases",
            "avoided_copy_bytes",
            "materializations",
            "fallback_deepcopies",
        )
    }
    apc_delta = {
        key: _delta(after_apc, before_apc, key)
        for key in ("lookups", "hits", "misses", "stores", "cached_tokens")
    }
    apc_join_passed = (
        cow_delta["branches"] >= 3
        and cow_delta["releases"] >= cow_delta["branches"]
        and int(after_cow.get("active_leases", -1)) == 0
        and cow_delta["avoided_copy_bytes"] > 0
        and apc_delta["hits"] >= 2
    )
    prompt_cache_passed = host_delta["hits"] >= 1 and host_delta["misses"] >= 3
    return {
        "initial_status_identity": {
            "artifact": initial.get("artifact"),
            "profile": initial.get("profile"),
            "qualification": initial.get("qualification"),
            "route": (initial.get("settings") or {}).get("route"),
        },
        "terminal_output_budget": {
            "passed": budget_passed,
            "scope": "gpu functional smoke; exact forward-count falsifier is CPU-instrumented",
            "usage": budget.get("usage"),
            "receipt": budget.get("receipt"),
        },
        "apcv2_restored_join": {
            "passed": apc_join_passed,
            "apcv2_delta": apc_delta,
            "cow_delta": cow_delta,
            "active_leases_after_idle": after_cow.get("active_leases"),
            "joined_request_receipts": [row.get("receipt") for row in joined],
        },
        "host_prompt_tokenization": {
            "passed": prompt_cache_passed,
            "scope": "real tokenizer warm-cache reuse; retry single-encode is CPU-instrumented",
            "host_prompt_cache_delta": host_delta,
            "first_tokens": warm_first.get("usage", {}).get("prompt_tokens"),
            "warm_tokens": warm_second.get("usage", {}).get("prompt_tokens"),
            "output_exact": warm_first.get("tokens") == warm_second.get("tokens"),
        },
    }


def _adaptive_checks(args, url: str, initial: dict) -> dict:
    prompt = "Give a deterministic four-point explanation of prefix caching."
    repetitions = [
        _stream_request(url, prompt, args.request_timeout, 16) for _ in range(2)
    ]
    concurrent_prompts = [
        "Give a deterministic explanation of speculative decoding.",
        "Give a deterministic explanation of continuous batching.",
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(
            pool.map(
                lambda value: _stream_request(
                    url, value, args.request_timeout, 16
                ),
                concurrent_prompts,
            )
        )
    rows = repetitions + concurrent
    adaptive = [
        ((row.get("receipt") or {}).get("mtp") or {}).get("adaptive_depth") or {}
        for row in rows
    ]
    pinned = all(
        value.get("selected") is True
        and value.get("reproducible_greedy") is True
        and int(value.get("current", -1)) == args.depth
        and int(value.get("max_depth", -1)) == args.depth
        for value in adaptive
    )
    stable = repetitions[0].get("tokens") == repetitions[1].get("tokens")
    after = _json_get(url, "/v1/status")
    return {
        "passed": pinned and stable,
        "initial_status_identity": {
            "artifact": initial.get("artifact"),
            "profile": initial.get("profile"),
            "qualification": initial.get("qualification"),
            "route": (initial.get("settings") or {}).get("route"),
        },
        "qualified_depth": args.depth,
        "all_greedy_receipts_pinned": pinned,
        "repeat_token_exact": stable,
        "repeat_token_hashes": [_stable_hash(row.get("tokens")) for row in repetitions],
        "adaptive_receipts": adaptive,
        "scheduler": after.get("scheduler"),
    }


def _template_compatibility_checks(args, url: str, initial: dict) -> dict:
    prompt = "Answer with exactly TEMPLATE_GPU_OK."
    rows = [
        _stream_request(url, prompt, args.request_timeout, 8) for _ in range(2)
    ]
    final = _json_get(url, "/v1/status")
    exact = rows[0].get("tokens") == rows[1].get("tokens")
    return {
        "passed": bool(final.get("healthy")) and exact,
        "scope": "GPU adapter/template compatibility; not vulnerable-renderer security qualification",
        "route": (initial.get("settings") or {}).get("route"),
        "qualification": initial.get("qualification"),
        "repeat_token_exact": exact,
        "repeat_token_hashes": [_stable_hash(row.get("tokens")) for row in rows],
        "receipts": [row.get("receipt") for row in rows],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=8436)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--cache-gib", type=int, default=8)
    parser.add_argument("--prefix-repetitions", type=int, default=768)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--template-compatibility-smoke", action="store_true")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.run_dir = args.output.parent / f"{args.output.stem}-artifacts"
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    token = args.run_dir / "admin-token"
    token.write_text(secrets.token_urlsafe(32) + "\n")
    token.chmod(0o600)
    report = {
        "schema": SCHEMA,
        "model": args.model,
        "source_root": str(ROOT),
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "performance_claim": False,
        "checks": {},
        "lifecycles": {},
    }
    if args.template_compatibility_smoke:
        compatibility, lifecycle = _run_server(
            args,
            route="ordinary",
            callback=lambda url, status: _template_compatibility_checks(
                args, url, status
            ),
        )
        report["checks"]["chat_template_gpu_compatibility"] = compatibility
        report["lifecycles"]["ordinary"] = lifecycle
    else:
        ordinary, ordinary_lifecycle = _run_server(
            args,
            route="ordinary",
            callback=lambda url, status: _ordinary_checks(args, url, status),
        )
        report["checks"].update(ordinary)
        report["lifecycles"]["ordinary"] = ordinary_lifecycle
        adaptive, adaptive_lifecycle = _run_server(
            args,
            route="native_mtp",
            callback=lambda url, status: _adaptive_checks(args, url, status),
        )
        report["checks"]["adaptive_mtp_greedy"] = adaptive
        report["lifecycles"]["native_mtp"] = adaptive_lifecycle
    verdicts = [
        check["passed"]
        for check in report["checks"].values()
        if "passed" in check
    ]
    report["passed"] = bool(verdicts) and all(verdict is True for verdict in verdicts)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "passed": report["passed"],
        "checks": {
            name: check.get("passed") for name, check in report["checks"].items()
        },
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
