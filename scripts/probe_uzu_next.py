#!/usr/bin/env python3
"""Owned research component tests followed by an ordinary production-route sanity server.

Research kernels are never installed in this server. HTTP validates the preserved
baseline route and APCv2, not the experimental arithmetic.
"""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

from probe_q4_gate_rows import policy_for
from probe_uzu_kernels import atomic, swapouts
from probe_uzu_next_worker import direct
from run_single_model_smoke_ladder import stop_server, thermal_sample, wait_health

ROOT = Path(__file__).resolve().parents[1]


def server_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("functional", "performance"), default="performance"
    )
    parser.add_argument("--reps", type=int, default=6)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--only", choices=("all", "gdn"), default="all")
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="ordinary reference plus HTTP follow-up for an already completed direct gate",
    )
    parser.add_argument("--candidate", choices=("on", "off"), default="off")
    args = parser.parse_args()
    if args.worker:
        direct(args)
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError("refusing a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)
    from qualification_gpu_ownership import require_qualification_lease
    from run_with_gpu_locks import foreign_model_processes

    owner = json.loads((args.cpg_owner_lock / "owner.json").read_text())
    ownership = require_qualification_lease(
        task_id=owner["cpg_task"],
        generation=owner["cpg_generation"],
        cpg_owner_lock=args.cpg_owner_lock,
    )
    if foreign_model_processes():
        raise RuntimeError("foreign model process contamination")
    start_thermal = thermal_sample()
    if start_thermal["thermal_state"] not in (0, 1) or (
        args.phase == "performance" and start_thermal["thermal_state"] != 0
    ):
        raise RuntimeError("thermal admission refused")
    files = [
        "scripts/probe_uzu_next.py",
        "scripts/probe_uzu_next_worker.py",
        "scripts/uzu_next_kernels.py",
        "src/mlx2/runtime/models/qsdpa_verify_metal.py",
        "src/mlx2/runtime/models/qwen38_tree_gdn.py",
        "src/mlx2/runtime/models/qwen4_moe_window.py",
        "src/mlx2/runtime/models/qwen4_routed_decode.py",
        "src/mlx2/runtime/models/qwen3_next.py",
        "src/mlx2/runtime/models/qwen3_5.py",
        "src/mlx2/runtime/models/gated_delta.py",
        "src/mlx2/adapters/qwen38_27b.py",
        "src/mlx2/adapters/flash_next.py",
        "scripts/probe_q4_gate_rows.py",
    ]
    result = {
        "schema": "mlx2.uzu-next-candidates.v1",
        "phase": args.phase,
        "qualified": False,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_files": {},
        "ownership": ownership,
        "thermal_start": start_thermal,
        "status": "running",
    }
    for path in files:
        data = (ROOT / path).read_bytes()
        result["source_files"][path] = hashlib.sha256(data).hexdigest()
        target = args.out / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    policy = policy_for(args.model, args.candidate == "on")
    result["candidate"] = args.candidate
    atomic(args.out / "policy.json", policy)
    atomic(args.out / "result.json", result)
    worker = server = None
    swap0 = swapouts()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(str(signum))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
            "--worker",
        ]
        with (args.out / "direct.log").open("w") as log:
            worker = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=server_env(), start_new_session=True
            )
            if worker.wait() != 0:
                raise RuntimeError("direct worker failed; see direct.json/direct.log")
        result["direct"] = json.loads((args.out / "direct.json").read_text())
        result["checkpoint"] = result["direct"]["checkpoint"]
        atomic(args.out / "result.json", result)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        command = [
            sys.executable,
            "-m",
            "mlx2.server",
            "--model",
            str(args.model),
            "--execution-policy",
            str(args.out / "policy.json"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--qualification-mode",
            "--max-lanes",
            "1",
            "--max-inflight",
            "1",
            "--max-context",
            "4096",
            "--cache-bytes",
            str(1 << 30),
        ]
        command.append(
            "--external-draft" if "draft_model" in policy else "--native-mtp"
        )
        with (args.out / "server.log").open("w") as log:
            server = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=server_env(), start_new_session=True
            )
            result["server_startup_status"] = wait_health(base, server, 300)
            with urllib.request.urlopen(base + "/v1/models", timeout=20) as response:
                served = json.load(response)["data"][0]["id"]
            responses = []
            for arm in ("cold", "warm"):
                request = {
                    "model": served,
                    "messages": [
                        {
                            "role": "user",
                            "content": "The test identifier is GATE-7492. " * 24
                            + "Reply only with the test identifier.",
                        }
                    ],
                    "enable_thinking": False,
                    "temperature": 0,
                    "max_tokens": 32,
                    "return_token_trace": True,
                }
                req = urllib.request.Request(
                    base + "/v1/chat/completions",
                    data=json.dumps(request).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=240) as response:
                    body = json.load(response)
                atomic(args.out / f"http-{arm}.json", body)
                responses.append(body)
            from mlx2.generation_trace import read_token_trace

            traces = [read_token_trace(body["mlx2"]) for body in responses]
            if traces[0]["token_ids"] != traces[1]["token_ids"]:
                raise AssertionError("HTTP B1 cold/warm token mismatch")
            if not all(
                b["choices"][0]["message"]["content"].strip() == "GATE-7492"
                for b in responses
            ):
                raise AssertionError("HTTP identifier oracle failed")
            if not responses[1]["mlx2"]["cached_tokens"]:
                raise AssertionError("HTTP APCv2 reuse not observed")
            with urllib.request.urlopen(base + "/v1/status", timeout=20) as response:
                result["server_final_status"] = json.load(response)
            observed = (
                result["server_final_status"]
                .get("execution", {})
                .get("q4_gate_three_rows", {})
            )
            result["http"] = {
                "exact_tokens": True,
                "functional_oracle": True,
                "warm_reused_tokens": responses[1]["mlx2"]["cached_tokens"],
                "policy_observed": observed,
            }
        result["status"] = "passed"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop_server(worker)
        stop_server(server)
        result["thermal_end"] = thermal_sample()
        result["swapouts_delta"] = swapouts() - swap0
        result["process_groups_stopped"] = all(
            p is None or p.poll() is not None for p in (worker, server)
        )
        if (args.out / "direct.json").exists():
            result["direct"] = json.loads((args.out / "direct.json").read_text())
            result["checkpoint"] = result["direct"].get("checkpoint")
        atomic(args.out / "result.json", result)


if __name__ == "__main__":
    main()
