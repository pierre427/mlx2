#!/usr/bin/env python3
"""End-to-end check that greedy 4-lane output equals 1-lane output in bit-exact mode.

GPU only: it starts real mlx2 servers. Run it under the lock wrapper:

  cpg_job.py run --label rm09-e2e --out <f> --lock -- \\
    .venv/bin/python scripts/gpu_check_verify_bitexact.py --i-own-the-gpu \\
      --overlay <scratch>/rm09/overlay --out <scratch>/rm09/e2e.jsonl

The ``bitexact`` arm (``--verify-bitexact``) and the ``control`` arm (no flag)
run on the same overlay build and are interleaved (bitexact, control,
bitexact, control, ...). In each arm the server:
1. runs every prompt alone (1 lane, sequential);
2. runs all four prompts together (4 lanes, concurrent);
3. compares the output text of each prompt between the two.

Greedy, temperature 0, fixed max_tokens.

Gates, which refuse the arm:
- Bit-exact arm: every receipt must say ``verify_bitexact: true``, and the
  server's ``verify_bitexact.dispatches`` must have advanced.
- Both arms: the 4-lane round must show observed compute width 4, so the
  mechanism under test (batched verify) actually ran.

**Go:** the bit-exact arm matches on 4/4 prompts in every repetition. The
control arm's divergence count is reported as context: the NAX evidence showed
2/4 on the 27B, but this is not required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [
    "Explain how a compiler works, in numbered sections.",
    "Explain how a database transaction works, in numbered sections.",
    "Explain how a CPU cache works, in numbered sections.",
    "Explain how a network router works, in numbered sections.",
]


def server_cmd(args, arm):
    cmd = [
        sys.executable,
        "-m",
        "mlx2.server",
        "--model",
        args.model,
        "--port",
        str(args.port),
        "--max-context",
        "32768",
        "--max-lanes",
        "4",
        "--max-inflight",
        "8",
        "--qualification-mode",
    ]
    if arm == "bitexact":
        cmd.append("--verify-bitexact")
    return cmd


def env_for(args):
    env = dict(os.environ)
    paths = [p for p in (args.overlay, str(ROOT / "src"), env.get("PYTHONPATH")) if p]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def http(url, body=None, timeout=1800):
    data = None if body is None else json.dumps(body).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def chat(args, text, arm):
    body = {
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "enable_thinking": False,
    }
    if arm == "bitexact":
        body["verify_bitexact"] = True
    result = http(f"http://127.0.0.1:{args.port}/v1/chat/completions", body)
    output = result["choices"][0]["message"]["content"] or ""
    receipt = result.get("mlx2") or {}
    widths = (receipt.get("mtp") or {}).get("observed_compute_widths") or [
        receipt.get("ordinary_compute_width", 1)
    ]
    return {
        "sha256": hashlib.sha256(output.encode()).hexdigest(),
        "text": output,
        "completion_tokens": result["usage"]["completion_tokens"],
        "verify_bitexact": receipt.get("verify_bitexact"),
        "verify_bitexact_reason": (receipt.get("verify_bitexact_detail") or {}).get("reason"),
        "widths": sorted(set(int(w) for w in widths)),
    }


def run_arm(args, arm, rep, log_dir):
    log = open(log_dir / f"server-{arm}-{rep}.log", "w")
    proc = subprocess.Popen(
        server_cmd(args, arm), env=env_for(args), stdout=log, stderr=subprocess.STDOUT, cwd=ROOT
    )
    try:
        deadline = time.monotonic() + args.startup_timeout
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"{arm} server exited during startup; see {log.name}")
            try:
                status = http(f"http://127.0.0.1:{args.port}/v1/status", timeout=10)
                if status.get("state") == "ready":
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"{arm} server not ready in {args.startup_timeout}s")
            time.sleep(2)
        before = (status.get("verify_bitexact") or {}).get("dispatches", 0)
        t0 = time.monotonic()
        single = [chat(args, text, arm) for text in PROMPTS]
        t1 = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            batched = list(pool.map(lambda text: chat(args, text, arm), PROMPTS))
        t2 = time.monotonic()
        after_status = http(f"http://127.0.0.1:{args.port}/v1/status", timeout=30)
        after = (after_status.get("verify_bitexact") or {}).get("dispatches", 0)
        if max(max(row["widths"]) for row in batched) < 4:
            raise RuntimeError(f"{arm}: the 4-lane round never reached compute width 4")
        if arm == "bitexact":
            if after <= before:
                raise RuntimeError("bitexact arm: route counter did not advance; refusing arm")
            bad = [row for row in single + batched if row["verify_bitexact"] is not True]
            if bad:
                raise RuntimeError(
                    f"bitexact arm: receipts without verify_bitexact=true: "
                    f"{[row['verify_bitexact_reason'] for row in bad]}"
                )
        matches = [a["sha256"] == b["sha256"] for a, b in zip(single, batched)]
        tokens_1 = sum(row["completion_tokens"] for row in single)
        tokens_4 = sum(row["completion_tokens"] for row in batched)
        return {
            "arm": arm,
            "rep": rep,
            "matches": matches,
            "identical": sum(matches),
            "prompts": len(PROMPTS),
            "tok_s_1lane": round(tokens_1 / (t1 - t0), 2),
            "tok_s_4lane": round(tokens_4 / (t2 - t1), 2),
            "bitexact_dispatch_delta": after - before,
            "first_divergence": [
                next(
                    (i for i, (x, y) in enumerate(zip(a["text"], b["text"])) if x != y),
                    None,
                )
                for a, b in zip(single, batched)
            ],
            "single_sha": [row["sha256"] for row in single],
            "batched_sha": [row["sha256"] for row in batched],
        }
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default=str(Path.home() / "mlx-models/Qwen3.8-27B-oQ4e-mtp"))
    parser.add_argument("--overlay", help="PYTHONPATH dir with the rm09 mlx build (mlx + dist-info)")
    parser.add_argument("--out")
    parser.add_argument("--port", type=int, default=8285)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--arms", nargs="+", default=["bitexact", "control"])
    parser.add_argument("--startup-timeout", type=float, default=600)
    args = parser.parse_args()
    if set(args.arms) - {"bitexact", "control"}:
        parser.error("arms must be bitexact and/or control")
    if args.dry_run:
        for rep in range(args.reps):
            for arm in args.arms:
                print(json.dumps({"rep": rep, "arm": arm, "cmd": server_cmd(args, arm),
                                  "PYTHONPATH": env_for(args)["PYTHONPATH"]}))
        return 0
    if not args.i_own_the_gpu:
        parser.error("refusing to start GPU servers without --i-own-the-gpu")
    if not args.out:
        parser.error("--out is required")
    out = Path(args.out)
    log_dir = out.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("a") as sink:
        for rep in range(args.reps):
            for arm in args.arms:
                row = run_arm(args, arm, rep, log_dir)
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if not k.endswith("_sha")}), flush=True)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
    bitexact = [row for row in rows if row["arm"] == "bitexact"]
    go = bool(bitexact) and all(row["identical"] == row["prompts"] for row in bitexact)
    verdict = {"verdict": "GO" if go else "NO_GO", "bitexact_reps": len(bitexact)}
    print(json.dumps(verdict))
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
