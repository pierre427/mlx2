#!/usr/bin/env python3
"""Interleaved server A/B for the draft-loop default (one server per arm visit).

Each arm is a list of extra ``mlx2.server`` arguments.  For every pair the
arms run in order: start the server, read ``/v1/status`` (the selected
verify topology and execution policy), send the mixed prompt set one request
at a time (greedy, thinking off), stop the server.  Decode time per request
is the receipt's ``elapsed_seconds - ttft_seconds``.

An arm named with ``loop`` must show ``draft_loop.observed_used`` on every
request; any other arm must show no draft loop.  A missing mechanism refuses
the run.  Refuses to start servers without ``--i-own-the-gpu``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mtp_confidence_gpu import PROMPTS  # noqa: E402


def _get(port, path):
    with urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as response:
        return json.load(response)


def _post(port, body):
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=3600) as response:
        return json.load(response)


def run_arm(args, name, extra, log_path):
    cmd = [sys.executable, "-m", "mlx2.server", "--model", args.model, "--port", str(args.port),
           "--max-lanes", str(args.concurrency), "--max-inflight", str(2 * args.concurrency),
           "--max-context", "32768",
           "--qualification-mode", "--native-mtp", *extra]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    log = open(log_path, "w")
    started = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    try:
        while True:
            if proc.poll() is not None:
                raise SystemExit(f"server for {name} exited early; see {log_path}")
            try:
                status = _get(args.port, "/v1/status")
                if status.get("state") == "ready":
                    break
            except OSError:
                pass
            if time.monotonic() - started > args.startup_timeout:
                raise SystemExit(f"server for {name} not ready; see {log_path}")
            time.sleep(2)
        ready_seconds = time.monotonic() - started
        settings = status.get("settings", {})
        rows = []
        groups = []

        def one(index):
            result = _post(args.port, {
                "messages": [{"role": "user", "content": PROMPTS[index]}],
                "temperature": 0, "max_tokens": args.max_tokens, "enable_thinking": False,
            })
            receipt = result["mlx2"]
            mtp = receipt.get("mtp") or {}
            return {
                "prompt": index,
                "completion_tokens": receipt["completion_tokens"],
                "decode_seconds": receipt["elapsed_seconds"] - receipt["ttft_seconds"],
                "text": result["choices"][0]["message"]["content"],
                "draft_loop": mtp.get("draft_loop"),
                "route": mtp.get("route"),
                "observed_compute_widths": mtp.get("observed_compute_widths"),
            }

        # Prompts go out ``concurrency`` at a time; a group's wall time gives
        # aggregate throughput, each request's receipt its own decode time.
        with ThreadPoolExecutor(args.concurrency) as pool:
            for start in range(0, len(PROMPTS), args.concurrency):
                indices = range(start, min(start + args.concurrency, len(PROMPTS)))
                began = time.monotonic()
                batch = list(pool.map(one, indices))
                groups.append({"prompts": list(indices), "wall_seconds": time.monotonic() - began,
                               "tokens": sum(r["completion_tokens"] for r in batch)})
                rows.extend(batch)
        final = _get(args.port, "/v1/status")
        mechanism = {
            "segmented_self_mtp": final.get("segmented_self_mtp")
            or (final.get("telemetry") or {}).get("segmented_self_mtp"),
            "scheduler": final.get("scheduler") or (final.get("telemetry") or {}).get("scheduler"),
        }
        return {
            "arm": name, "extra": extra, "ready_seconds": ready_seconds,
            "mechanism": mechanism,
            "verify_topology": settings.get("verify_topology"),
            "execution_policy": settings.get("execution_policy"),
            "lane_matmul": settings.get("lane_matmul"),
            "concurrency": args.concurrency,
            "groups": groups,
            "rows": rows,
        }
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        time.sleep(args.cooldown)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", action="append", required=True,
                        help="name=extra server args (shell-quoted)")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--port", type=int, default=8397)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="requests in flight at once (server --max-lanes)")
    parser.add_argument("--cooldown", type=float, default=5,
                        help="seconds idle after each server visit (thermal settling)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.i_own_the_gpu:
        raise SystemExit("refusing to start servers without --i-own-the-gpu")
    arms = []
    for spec in args.arm:
        name, _, extra = spec.partition("=")
        arms.append((name, shlex.split(extra)))
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for pair in range(args.pairs):
        for name, extra in arms:
            result = run_arm(args, name, extra, args.out / f"{name}-{pair}.server.log")
            looped = "loop" in name
            used = [bool(r["draft_loop"] and r["draft_loop"]["observed_used"]) for r in result["rows"]]
            # Every request must carry the loop's receipt; a request that never
            # ran at a selected width legitimately made no decision.
            if looped and (not all(r["draft_loop"] for r in result["rows"]) or not any(used)):
                raise SystemExit(f"REFUSED arm {name}: draft loop not selected or never used")
            if not looped and any(r["draft_loop"] for r in result["rows"]):
                raise SystemExit(f"REFUSED arm {name}: unexpected draft loop")
            result["pair"] = pair
            results.append(result)
            tokens = sum(r["completion_tokens"] for r in result["rows"])
            seconds = sum(r["decode_seconds"] for r in result["rows"])
            wall = sum(g["wall_seconds"] for g in result["groups"])
            print(json.dumps({"pair": pair, "arm": name, "ready_seconds": round(result["ready_seconds"], 1),
                              "topology": (result["verify_topology"] or {}).get("selected"),
                              "ms_per_token": round(1e3 * seconds / tokens, 3),
                              "aggregate_tok_s": round(tokens / wall, 2)}), flush=True)
            (args.out / "results.json").write_text(json.dumps(results, indent=1))
    summary = {}
    base = None
    base_tps = None
    for name, _ in arms:
        visits = [x for x in results if x["arm"] == name]
        values = [1e3 * sum(r["decode_seconds"] for r in x["rows"]) / sum(r["completion_tokens"] for r in x["rows"])
                  for x in visits]
        tps = [sum(g["tokens"] for g in x["groups"]) / sum(g["wall_seconds"] for g in x["groups"])
               for x in visits]
        summary[name] = {"ms_per_token_pairs": values, "median": statistics.median(values),
                         "aggregate_tok_s_pairs": tps, "aggregate_tok_s": statistics.median(tps)}
        base = base or summary[name]["median"]
        base_tps = base_tps or summary[name]["aggregate_tok_s"]
        summary[name]["speedup_vs_first"] = base / summary[name]["median"]
        summary[name]["aggregate_speedup_vs_first"] = summary[name]["aggregate_tok_s"] / base_tps
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
