#!/usr/bin/env python3
"""Run bounded real-artifact North serving feature and B1/BN parity smoke.

The caller must hold the CPG GPU lease and both filesystem locks.  This is a
component smoke, not qualification: it deliberately skips the full unit suite
and near-limit context matrix handled by the qualification campaign.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock"))
GREEDY = {
    "temperature": 0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_owner(path: Path) -> dict:
    owner = path / "owner.json"
    if not path.is_dir() or not owner.is_file():
        raise RuntimeError(f"GPU lock is not owned: {path}")
    return json.loads(owner.read_text())


def validate_ownership(session: str, label: str) -> dict:
    owners = [read_owner(path) for path in LOCKS]
    if owners[0] != owners[1]:
        raise RuntimeError("shared GPU lock owner receipts differ")
    owner = owners[0]
    if owner.get("session") != session or owner.get("label") != label:
        raise RuntimeError(
            f"GPU owner mismatch: expected {session}/{label}, got "
            f"{owner.get('session')}/{owner.get('label')}"
        )
    return owner


def request_json(base: str, path: str, body: dict | None = None, timeout=120):
    request = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={} if body is None else {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def chat(base: str, body: dict, timeout=300):
    return request_json(
        base,
        "/v1/chat/completions",
        {**GREEDY, "max_tokens": 64, **body},
        timeout=timeout,
    )


def prompt(text: str, **options) -> dict:
    return {"messages": [{"role": "user", "content": text}], **options}


def content(response: dict) -> str:
    return (response["choices"][0]["message"].get("content") or "").strip()


def token_ids(response: dict) -> list[int]:
    entries = response["choices"][0].get("logprobs", {}).get("content", [])
    return [int(entry["id"]) for entry in entries]


def receipt(response: dict) -> dict:
    return response.get("mlx2") or {}


def wait_health(base: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during load with rc={process.returncode}")
        try:
            health = request_json(base, "/health", timeout=3)
            if health.get("status") == "ok":
                return request_json(base, "/v1/status", timeout=10)
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"server did not become healthy within {timeout:g}s")


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def execution_policy(*, expert_gather_sort: str, batch_row_exact_q4: bool) -> dict:
    policy = {}
    if expert_gather_sort != "auto":
        policy["expert_gather_sort"] = expert_gather_sort
    if batch_row_exact_q4:
        policy["batch_row_exact_q4"] = True
    return policy


def batch_row_exact_q4_engaged(route: dict) -> bool:
    counts = route.get("counts") or {}
    complete = int(counts.get("complete_forwards", 0) or 0)
    started = int(counts.get("started_forwards", 0) or 0)
    return (
        route.get("selected") is True
        and route.get("observed_used") is True
        and int(counts.get("kernel", 0) or 0) > 0
        and int(counts.get("group_kernel", 0) or 0) > 0
        and complete > 0
        and started == complete
        and int(counts.get("per_row", 0) or 0) == 0
        and int(counts.get("refusals", 0) or 0) == 0
    )


def build_parity_prompts(nonce_prefix: str = "north-parity") -> list[dict]:
    return [
        prompt(
            f"Explain {subject} in numbered steps. nonce {nonce_prefix}-{index}",
            reasoning_effort="none",
            think=False,
            logprobs=True,
            min_tokens=32,
            max_tokens=32,
        )
        for index, subject in enumerate(
            ("a compiler", "a database", "a refrigerator", "photosynthesis")
        )
    ]


def run_features(
    base: str,
    initial: dict,
    *,
    expert_gather_sort: str,
    batch_row_exact_q4: bool,
    schema: str = "mlx2.north-feature-smoke.v1",
    marker: str = "NORTH_APC_READY",
    nonce_prefix: str = "north-parity",
) -> dict:
    checks: dict[str, dict] = {}

    def check(name: str, passed: bool, evidence) -> None:
        checks[name] = {"passed": bool(passed), "evidence": evidence}

    if batch_row_exact_q4:
        initial_route = initial.get("execution", {}).get("batch_row_exact_q4", {})
        check(
            "batch_row_exact_q4_selected",
            initial_route.get("selected") is True,
            initial_route,
        )

    apc_request = prompt(
        f"Reply with exactly {marker}",
        reasoning_effort="none",
        think=False,
        logprobs=True,
        max_tokens=32,
    )
    cold = chat(base, apc_request)
    warm = chat(base, apc_request)
    check(
        "apcv2_exact_replay",
        content(cold) == content(warm)
        and token_ids(cold) == token_ids(warm)
        and receipt(warm).get("cached_tokens", 0) > 0
        and receipt(warm).get("cache") == "apcv2",
        {"cold": cold, "warm": warm},
    )

    parity_prompts = build_parity_prompts(nonce_prefix)
    # First responses prime exact APCv2 entries; second responses are the B1
    # warm reference.  The same four requests then enter one concurrent cohort.
    primed = [chat(base, item) for item in parity_prompts]
    b1 = [chat(base, item) for item in parity_prompts]
    with ThreadPoolExecutor(max_workers=4) as pool:
        bn = list(pool.map(lambda item: chat(base, item), parity_prompts))
    b1_ids = [token_ids(response) for response in b1]
    bn_ids = [token_ids(response) for response in bn]
    widths = [int(receipt(response).get("ordinary_compute_width") or 0) for response in bn]
    check(
        "b1_bn_token_parity",
        all(len(ids) == 32 for ids in b1_ids + bn_ids)
        and b1_ids == bn_ids
        and max(widths, default=0) >= 2
        and all(receipt(response).get("cached_tokens", 0) > 0 for response in b1 + bn),
        {
            "primed": [receipt(response) for response in primed],
            "b1": b1,
            "bn": bn,
            "b1_token_ids": b1_ids,
            "bn_token_ids": bn_ids,
            "bn_observed_widths": widths,
        },
    )

    structured_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"const": "yes"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }
    structured = chat(
        base,
        prompt(
            "Return a JSON object whose answer is yes.",
            response_format=structured_format,
            reasoning_effort="none",
            think=False,
        ),
    )
    try:
        structured_value = json.loads(content(structured))
    except (TypeError, ValueError, json.JSONDecodeError):
        structured_value = None
    structured_receipt = (
        receipt(structured).get("request_controls", {}).get("structured_output") or {}
    )
    check(
        "grammar",
        structured_value == {"answer": "yes"}
        and structured_receipt.get("enforced") is True
        and structured_receipt.get("tail_mass_bound") == 0.0,
        structured,
    )

    tools = [{
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    tool_request = prompt(
        "Use the weather tool to get the weather in Toronto.",
        tools=tools,
        max_tokens=256,
    )
    tool_response = chat(base, tool_request)
    choice = tool_response["choices"][0]
    calls = choice["message"].get("tool_calls", [])
    tool_ok = False
    if len(calls) == 1:
        try:
            arguments = json.loads(calls[0]["function"]["arguments"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            arguments = {}
        tool_ok = (
            choice.get("finish_reason") == "tool_calls"
            and calls[0]["function"].get("name") == "weather"
            and arguments.get("city") == "Toronto"
        )
    check("tools", tool_ok, tool_response)
    if tool_ok:
        followup = chat(base, {
            "tools": tools,
            "messages": tool_request["messages"] + [
                choice["message"],
                {
                    "role": "tool",
                    "tool_call_id": calls[0]["id"],
                    "name": "weather",
                    "content": '{"temperature_c": 21, "condition": "sunny"}',
                },
            ],
        })
        check("tool_roundtrip", "21" in content(followup), followup)
    else:
        check("tool_roundtrip", False, {"blocked_by": "tools"})

    reasoning = chat(
        base,
        prompt(
            "What is 17 plus 25? Give a short final answer.",
            reasoning_effort="medium",
            think=True,
            max_tokens=512,
        ),
        timeout=600,
    )
    message = reasoning["choices"][0]["message"]
    check(
        "reasoning",
        bool((message.get("reasoning_content") or "").strip())
        and bool((message.get("content") or "").strip())
        and "42" in message.get("content", ""),
        reasoning,
    )

    before = request_json(base, "/v1/status")["counts"].get("cancelled", 0)
    cancel_body = {
        **GREEDY,
        **prompt(
            "Write a very long guide to compiler optimization.",
            reasoning_effort="none",
            think=False,
            max_tokens=8192,
        ),
        "stream": True,
    }
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(cancel_body).encode(),
        headers={"Content-Type": "application/json"},
    )
    response = urllib.request.urlopen(request, timeout=120)
    first_line = response.readline().decode(errors="replace")
    response.close()
    deadline = time.monotonic() + 30
    state = request_json(base, "/v1/status")
    while state.get("inflight") != 0 and time.monotonic() < deadline:
        time.sleep(0.2)
        state = request_json(base, "/v1/status")
    check(
        "cancellation",
        state.get("inflight") == 0 and state["counts"].get("cancelled", 0) > before,
        {"first_line": first_line, "before": before, "final_status": state},
    )
    recovered = chat(base, apc_request)
    check(
        "recovery",
        content(recovered) == content(cold)
        and token_ids(recovered) == token_ids(cold)
        and request_json(base, "/health").get("status") == "ok",
        recovered,
    )

    final = request_json(base, "/v1/status")
    quiescence_deadline = time.monotonic() + 5
    while (
        final.get("inflight") != 0
        or final.get("apcv2", {}).get("cow", {}).get("active_leases") != 0
    ) and time.monotonic() < quiescence_deadline:
        time.sleep(0.1)
        final = request_json(base, "/v1/status")
    if expert_gather_sort != "auto":
        route = final.get("execution", {}).get("expert_gather_sort", {})
        counters = route.get("counts", {})
        observed = int(counters.get("unsorted_decode", 0) or 0)
        check(
            "expert_gather_route",
            route.get("selected") is True
            and route.get("observed_used") is True
            and route.get("policy") == expert_gather_sort
            and observed > 0,
            route,
        )
    if batch_row_exact_q4:
        route = final.get("execution", {}).get("batch_row_exact_q4", {})
        check("batch_row_exact_q4_route", batch_row_exact_q4_engaged(route), route)
    check(
        "quiescence",
        final.get("inflight") == 0
        and final.get("apcv2", {}).get("cow", {}).get("active_leases") == 0,
        final,
    )
    return {
        "schema": schema,
        "semantics": "bounded component smoke; not qualification or performance evidence",
        "initial_status": initial,
        "checks": checks,
        "final_status": final,
        "passed": all(row["passed"] for row in checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--port", type=int, default=18953)
    parser.add_argument("--load-timeout", type=float, default=600)
    parser.add_argument(
        "--expert-gather-sort",
        choices=("auto", "unsorted_decode"),
        default="auto",
    )
    parser.add_argument(
        "--batch-row-exact-q4",
        action="store_true",
        help="select the default-off North q4 row-exact ordinary decode candidate",
    )
    args = parser.parse_args(argv)

    if args.batch_row_exact_q4 and args.expert_gather_sort != "auto":
        parser.error(
            "--batch-row-exact-q4 cannot be combined with --expert-gather-sort"
        )

    model = args.model.expanduser().resolve(strict=True)
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
    tracked_dirty = bool(subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip())
    campaign = {
        "schema": "mlx2.north-feature-smoke-campaign.v1",
        "status": "running",
        "semantics": "bounded component smoke; not qualification or performance evidence",
        "source_revision": revision,
        "source_tracked_dirty": tracked_dirty,
        "model": str(model),
        "owner": owner,
        "expert_gather_sort": args.expert_gather_sort,
        "batch_row_exact_q4": args.batch_row_exact_q4,
        "started_at": time.time(),
    }
    atomic_json(out / "campaign.json", campaign)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX2_CAMPAIGN_ROOT": str(ROOT),
    }
    smoke_command = [
        sys.executable,
        str(ROOT / "scripts/smoke_local_model.py"),
        "--model", str(model),
        "--out", str(out / "ordinary-smoke.json"),
        "--max-tokens", "16",
        "--i-own-gpu",
        "--cpg-lease", f"cpg:{args.session}:{args.label}",
    ]
    server_command = [
        sys.executable, "-u", "-m", "mlx2.server",
        "--model", str(model),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--ordinary",
        "--qualification-mode",
        "--max-context", "16384",
        "--max-lanes", "4",
        "--max-inflight", "4",
        "--cache-dir", str(out / "apcv2-cache"),
    ]
    policy = execution_policy(
        expert_gather_sort=args.expert_gather_sort,
        batch_row_exact_q4=args.batch_row_exact_q4,
    )
    if policy:
        policy_path = out / "execution-policy.json"
        atomic_json(policy_path, policy)
        server_command += ["--execution-policy", str(policy_path)]
    server = None
    report = None
    try:
        with (out / "ordinary-smoke.log").open("w") as log:
            subprocess.run(
                smoke_command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=args.load_timeout,
            )
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
            report = run_features(
                base,
                initial,
                expert_gather_sort=args.expert_gather_sort,
                batch_row_exact_q4=args.batch_row_exact_q4,
            )
            atomic_json(out / "feature-smoke.json", report)
            campaign["status"] = "passed" if report["passed"] else "failed"
    except BaseException as error:
        campaign["status"] = "failed"
        campaign["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop_server(server)
        campaign["finished_at"] = time.time()
        campaign["locks_still_owned_before_return"] = (
            validate_ownership(args.session, args.label) == owner
        )
        atomic_json(out / "campaign.json", campaign)
    return 0 if report is not None and report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
