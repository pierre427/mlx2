#!/usr/bin/env python3
"""Served confirmation of a Super VL prefill step: TTFT and greedy identity.

``owned OUT`` takes a desktop CPG lease and the paired GPU locks and runs
``unit OUT``: for each ``--steps`` value one loopback server on the default
route with ``--prefill-step`` set; three distinct prompts at each length
(1-token completions, TTFT from the server's own receipt), then one 96-token
greedy completion on a fixed 4K prompt whose text and token ids are compared
across steps.  Attribution only, no thermal hold.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts/model_matrix"))
import nemotron_tree_check as check  # noqa: E402


def _prompt(tok, tokens, seed):
    words = " ".join(f"Record {seed}-{i}: shipment {i * 31 % 997} left depot {i % 41}"
                     f" carrying {i * 7 % 113} crates." for i in range(tokens // 12))
    return tok.decode(tok.encode(words)[:tokens])


def unit(out, args):
    import run_matrix as rm
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(check.TARGET, local_files_only=True)
    report = {"schema": "mlx2.nemotron-prefill-step-confirm.v1", "checkpoint": check.checkpoint(),
              "source": check.source(), "qualified": False,
              "evidence_kind": "served attribution, no thermal hold", "steps": {}}
    for step in args.steps:
        command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(check.TARGET),
                   "--host", "127.0.0.1", "--port", str(args.port), "--fabric", "off",
                   "--max-context", "40960", "--cache-bytes", str(2 << 30),
                   "--prefill-step", str(step)]
        log = (out / f"server-{step}.log").open("w")
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                  start_new_session=True,
                                  env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        base = f"http://127.0.0.1:{args.port}"
        row = {"command": command, "ttft": [], "greedy": None}
        try:
            row["load_seconds"] = rm.wait_health(base, server, 1800, lambda: None)
            for length in args.lengths:
                for rep in range(3):
                    body = {"model": check.TARGET.name,
                            "messages": [{"role": "user", "content": _prompt(tok, length, f"{length}-{rep}")}],
                            "max_tokens": 1, "temperature": 0, "stream": False,
                            "enable_thinking": False}
                    code, response, _wall = rm.http_json(base + "/v1/chat/completions", body, timeout=1800)
                    block = response.get("mlx2", {}) if isinstance(response, dict) else {}
                    prompt = (response.get("usage") or {}).get("prompt_tokens") if code == 200 else None
                    ttft = block.get("ttft_seconds")
                    item = {"length": length, "rep": rep, "status": code, "prompt_tokens": prompt,
                            "ttft_seconds": ttft, "route": block.get("route"),
                            "prefill_tok_s": round(prompt / ttft, 1) if prompt and ttft else None}
                    row["ttft"].append(item)
                    print(json.dumps({"step": step, **item}), flush=True)
            body = {"model": check.TARGET.name,
                    "messages": [{"role": "user", "content": _prompt(tok, 4096, "greedy") +
                                  "\nSummarize the three largest shipments."}],
                    "max_tokens": 96, "temperature": 0, "stream": False, "enable_thinking": False,
                    "return_token_trace": True}
            code, response, _wall = rm.http_json(base + "/v1/chat/completions", body, timeout=1800)
            choice = (response.get("choices") or [{}])[0] if code == 200 else {}
            block = (response.get("mlx2") or {}) if code == 200 else {}
            row["greedy"] = {"status": code, "content": (choice.get("message") or {}).get("content"),
                             "completion_tokens": (response.get("usage") or {}).get("completion_tokens")}
            row["greedy_trace"] = {k: v for k, v in block.items() if "token" in k}
        finally:
            row["server_stop"] = rm.stop_server(server)
            log.close()
        report["steps"][str(step)] = row
        check.atomic(out / "report.json", report)
    greedy = [row["greedy"] for row in report["steps"].values()]
    report["greedy_identical"] = all(g == greedy[0] for g in greedy) if greedy else None
    check.atomic(out / "report.json", report)
    print(json.dumps({"greedy_identical": report["greedy_identical"]}), flush=True)
    return 0


def owned(out, args):
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    job_id = "nemotron-prefill-step-" + out.name
    with DesktopCPGLease(out, job_id, checkpoint=check.checkpoint(),
                         source_revision=check.source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out),
                   "--steps", *map(str, args.steps), "--lengths", *map(str, args.lengths),
                   "--port", str(args.port)]
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                                 env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32768])
    parser.add_argument("--port", type=int, default=8796)
    args = parser.parse_args()
    if args.mode == "unit":
        args.out.mkdir(parents=True, exist_ok=True)
    return {"owned": owned, "unit": unit}[args.mode](args.out, args)


if __name__ == "__main__":
    raise SystemExit(main())
