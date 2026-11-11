#!/usr/bin/env python3
"""Super VL NVFP4: served decode at short and long context, per route arm.

``owned OUT`` takes a desktop CPG lease and the paired GPU locks and runs
``unit OUT``.  Arms: ``ordinary`` (--ordinary), ``tree`` (the no-flag default
with ``MLX2_SUPER_TREE_COST_HANDOFF=off``) and ``guarded`` (the same with
``guarded``).  Per arm, per length, two distinct record-list prompts end with
a question and request exactly 512 tokens (``min_tokens == max_tokens``, the
ladder's decode shape), greedy.  Recorded: decode tok/s from the server's
own TTFT and elapsed, prompt tokens, and the speculation receipt (rounds,
accepted, proposed, plain/park counters).  Attribution only, no thermal hold.
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
sys.path.insert(0, str(ROOT / "scripts/research"))
sys.path.insert(0, str(ROOT / "scripts/model_matrix"))
import nemotron_prefill_step_confirm as confirm  # noqa: E402
import nemotron_tree_check as check  # noqa: E402

ARMS = {
    "ordinary": (["--ordinary"], {}),
    "tree": ([], {"MLX2_SUPER_TREE_COST_HANDOFF": "off"}),
    "guarded": ([], {"MLX2_SUPER_TREE_COST_HANDOFF": "guarded"}),
}


def unit(out, args):
    import run_matrix as rm
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(check.TARGET, local_files_only=True)
    report = {"schema": "mlx2.super-vl-long-decode.v1", "checkpoint": check.checkpoint(),
              "source": check.source(), "qualified": False,
              "evidence_kind": "served attribution, no thermal hold", "arms": {}}
    for name in args.arms:
        flags, arm_env = ARMS[name]
        command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(check.TARGET),
                   "--host", "127.0.0.1", "--port", str(args.port), "--fabric", "off",
                   "--max-context", "40960", "--cache-bytes", str(2 << 30), *flags]
        log = (out / f"server-{name}.log").open("w")
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                  start_new_session=True,
                                  env={**os.environ, "PYTHONPATH": str(ROOT / "src"), **arm_env})
        base = f"http://127.0.0.1:{args.port}"
        row = {"command": command, "env": arm_env, "requests": []}
        try:
            row["load_seconds"] = rm.wait_health(base, server, 1800, lambda: None)
            for length in args.lengths:
                for rep in range(args.reps):
                    text = confirm._prompt(tok, length, f"long-{length}-{rep}")
                    body = {"model": check.TARGET.name,
                            "messages": [{"role": "user", "content": text +
                                          "\nList every shipment that carried more than 100 crates, "
                                          "with its depot, then summarize the pattern."}],
                            "max_tokens": 512, "min_tokens": 512, "temperature": 0,
                            "stream": False, "enable_thinking": False}
                    code, response, _wall = rm.http_json(base + "/v1/chat/completions", body,
                                                         timeout=1800)
                    block = (response.get("mlx2") or {}) if code == 200 else {}
                    usage = (response.get("usage") or {}) if code == 200 else {}
                    ttft, elapsed = block.get("ttft_seconds"), block.get("elapsed_seconds")
                    tokens = usage.get("completion_tokens")
                    spec = block.get("speculation") if isinstance(block.get("speculation"), dict) else {}
                    item = {"length": length, "rep": rep, "status": code,
                            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": tokens,
                            "ttft_seconds": ttft, "elapsed_seconds": elapsed,
                            "decode_tok_s": (round((tokens - 1) / (elapsed - ttft), 2)
                                             if tokens and ttft and elapsed and elapsed > ttft else None),
                            "route": block.get("route"),
                            "speculation": {k: v for k, v in spec.items()
                                            if isinstance(v, (int, float, str, bool))}}
                    row["requests"].append(item)
                    print(json.dumps({"arm": name, **{k: item[k] for k in (
                        "length", "rep", "status", "decode_tok_s", "prompt_tokens")},
                        "rounds": spec.get("external_rounds"), "accepted": spec.get("accepted")}),
                        flush=True)
            row["status"] = rm.http_json(base + "/v1/status", None, timeout=30)[1]
        finally:
            row["server_stop"] = rm.stop_server(server)
            log.close()
        report["arms"][name] = row
        check.atomic(out / "report.json", report)
    return 0


def owned(out, args):
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    job_id = "super-vl-long-decode-" + out.name
    with DesktopCPGLease(out, job_id, checkpoint=check.checkpoint(),
                         source_revision=check.source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out),
                   "--arms", *args.arms, "--lengths", *map(str, args.lengths),
                   "--reps", str(args.reps), "--port", str(args.port)]
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
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--port", type=int, default=8796)
    args = parser.parse_args()
    if args.mode == "unit":
        args.out.mkdir(parents=True, exist_ok=True)
    return {"owned": owned, "unit": unit}[args.mode](args.out, args)


if __name__ == "__main__":
    raise SystemExit(main())
