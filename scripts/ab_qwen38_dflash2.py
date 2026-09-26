#!/usr/bin/env python3
"""Served A/B on Qwen3.8 27B: ordinary vs self-MTP K=2 vs DFlash2 external draft.

GPU-only (starts real ``mlx2.server`` processes, one model per process).
Refuses to run without ``--i-own-the-gpu``; ``--dry-run`` prints the plan.

Arms (same target artifact, same server flags except the route):

* ``ord``  ``--ordinary``
* ``mtp2`` adapter default route (native self-MTP, num_draft 2)
* ``dkN``  ``--external-draft`` with the pinned DFlash2 policy
  ``policy-kN.json`` (block size N+1)

Per arm process: load, a discarded warm-up (every workload, both
temperatures, B1 and B4, so first-use Metal shape compiles land outside the
timed cells), then every cell: workload x width x temperature.  Arms run in
the order given; a campaign alternates the order between reps (ABBA over
processes) so drift lands on both sides.  Every request carries a fresh
nonce so no arm reuses another's prefix cache.

Swap guard: ``vm_stat`` pageouts/swapouts are sampled every 2 s.  A rise of
1.5 GiB from the pre-launch baseline aborts during load/warm-up; once the
timed cells start, a rise of 256 MiB aborts.

Per request it records TTFT, decode tok/s ((tokens-1)/(elapsed-ttft)), the
route receipt and the full output text (for the greedy equality gate).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "qualification/runs/sp-dflash2-27b-20260925"
MODEL = "~/mlx-models/Qwen3.8-27B-oQ4e-mtp"

CODE = [
    "Write a Python module implementing an LRU cache class with get, put and "
    "delete, O(1) operations, type hints, docstrings and five pytest tests. "
    "Output only the code.",
    "Implement Dijkstra's shortest path in TypeScript over an adjacency list "
    "with a binary-heap priority queue, plus a small usage example. Output "
    "only the code.",
    "Write a Rust function that parses a CSV line with quoted fields and "
    "escaped quotes into a Vec<String>, with unit tests. Output only the code.",
    "Write a bash script that rotates log files in a directory: gzip files "
    "older than 1 day, delete archives older than 30 days, and print a "
    "summary. Output only the code.",
]
PROSE = [
    "Write a 500-word essay on why cities build public libraries.",
    "Explain, for a general audience, how vaccines train the immune system.",
    "Write a short story about a lighthouse keeper who finds a message in a "
    "bottle written in her own handwriting.",
    "Describe the history of the printing press and its effect on Europe.",
]
CHAT = [
    [
        {"role": "user", "content": "I'm planning a 3-day trip to Kyoto in November."},
        {"role": "assistant", "content": "Great choice: autumn leaves peak in mid to late November. What are your interests?"},
        {"role": "user", "content": "Temples, food, and a bit of hiking. Give me a day-by-day plan."},
    ],
    [
        {"role": "user", "content": "My Python script says 'list index out of range' in a loop."},
        {"role": "assistant", "content": "Can you share the loop?"},
        {"role": "user", "content": "for i in range(len(xs)+1): print(xs[i])\nWhy does it fail and how do I fix it? Also explain enumerate."},
    ],
    [
        {"role": "user", "content": "What's the difference between a Roth IRA and a traditional IRA, in general terms?"},
        {"role": "assistant", "content": "The main difference is when you pay tax. Want a comparison table?"},
        {"role": "user", "content": "Yes, a table, then a short explanation of who each tends to suit."},
    ],
    [
        {"role": "user", "content": "Help me write SQL: tables orders(id, customer_id, total, created_at) and customers(id, name)."},
        {"role": "assistant", "content": "Sure. What should the query return?"},
        {"role": "user", "content": "Top 5 customers by total spend in 2025, with their order count. Then explain the query."},
    ],
]
WORKLOADS = {"code": CODE, "prose": PROSE, "chat": CHAT}
# Vendor defaults from the artifact's generation_config.json (temperature
# 1.0, top_p 0.95, top_k 20); see adapters/qwen.py.
SAMPLED = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def arm_route_args(arm):
    if arm == "ord":
        return ["--ordinary"]
    if arm == "mtp2":
        return []
    match = re.fullmatch(r"dk(\d)", arm)
    if match:
        return ["--external-draft", "--execution-policy",
                str(RUN / f"policy-k{match.group(1)}.json")]
    raise ValueError(f"unknown arm {arm}")


def _post(url, body, timeout):
    request = Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _status(url):
    with urlopen(url + "/v1/status", timeout=30) as response:
        return json.load(response)


class SwapGuard:
    """Sample vm_stat swapouts; flag a breach instead of killing from a thread."""

    PAGE = 16384

    def __init__(self):
        self.base = self.read()
        self.limit = int(1.5 * (1 << 30))
        self.breach = None
        self.peak = 0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    @classmethod
    def read(cls):
        text = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        match = re.search(r"Swapouts:\s+(\d+)", text)
        return int(match.group(1)) * cls.PAGE if match else 0

    def arm_timed(self):
        self.base = self.read()
        self.limit = 256 << 20

    def _loop(self):
        while not self._stop.wait(2.0):
            rise = self.read() - self.base
            self.peak = max(self.peak, rise)
            if rise >= self.limit and self.breach is None:
                self.breach = rise

    def check(self):
        if self.breach is not None:
            raise RuntimeError(f"swap guard: swapouts rose {self.breach >> 20} MiB")

    def close(self):
        self._stop.set()


def _wait_ready(url, process, timeout, guard):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        guard.check()
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}")
        try:
            status = _status(url)
            if status.get("healthy") and status.get("ready", True):
                return status
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError("server did not become ready")


def _messages(item, nonce):
    if isinstance(item, str):
        return [{"role": "user", "content": f"[{nonce}] {item}"}]
    messages = [dict(message) for message in item]
    messages[0]["content"] = f"[{nonce}] {messages[0]['content']}"
    return messages


def _request(url, item, *, arm, max_tokens, temperature, timeout, nonce, seed):
    body = {"messages": _messages(item, nonce), "max_tokens": max_tokens,
            "enable_thinking": False}
    if temperature == 0:
        body["temperature"] = 0.0
    else:
        body.update(SAMPLED)
        body["seed"] = seed
    started = time.perf_counter()
    result = _post(url, body, timeout)
    wall = time.perf_counter() - started
    receipt = result["mlx2"]
    speculation = receipt.get("speculation") or {}
    mtp = receipt.get("mtp")
    if arm.startswith("dk") and speculation.get("kind") != "external_dflash2":
        raise RuntimeError(f"refusing arm {arm}: receipt is not the DFlash2 route: {speculation}")
    if arm == "mtp2" and not mtp:
        raise RuntimeError("refusing arm mtp2: receipt is not native self-MTP")
    if arm == "ord" and (mtp or speculation):
        raise RuntimeError("refusing arm ord: speculative receipt on the ordinary arm")
    tokens = int(receipt["completion_tokens"])
    ttft = float(receipt["ttft_seconds"])
    decode_s = float(receipt["elapsed_seconds"]) - ttft
    content = result["choices"][0]["message"].get("content") or ""
    reasoning = result["choices"][0]["message"].get("reasoning_content") or ""
    return {
        "completion_tokens": tokens,
        "prompt_tokens": receipt.get("prompt_tokens"),
        "ttft_s": ttft,
        "decode_tok_s": (tokens - 1) / decode_s if decode_s > 0 and tokens > 1 else None,
        "wall_s": wall,
        "finish_reason": result["choices"][0].get("finish_reason"),
        "output_sha256": hashlib.sha256((reasoning + "\x00" + content).encode()).hexdigest(),
        "output": content,
        "reasoning": reasoning,
        "speculation": speculation or None,
        "mtp": mtp,
        "route": receipt.get("route"),
        "profile": receipt.get("profile"),
    }


def run_cell(url, arm, workload, width, temperature, args, seed_base, rep):
    items = WORKLOADS[workload]
    batch = items[:width] if width > 1 else items[: args.b1_prompts]
    # Identical across arms (so greedy outputs are comparable), distinct per
    # cell (so no cell reuses another's prefix cache in the same process).
    # APCv2 disk persistence is off, so nothing crosses server processes.
    nonce = hashlib.sha256(
        f"{args.nonce_salt}:{rep}:{workload}:{width}:{temperature}".encode()
    ).hexdigest()[:16]

    def one(pair):
        index, item = pair
        return _request(url, item, arm=arm, max_tokens=args.max_tokens,
                        temperature=temperature, timeout=args.timeout,
                        nonce=nonce, seed=seed_base + index)

    started = time.perf_counter()
    if width > 1:
        with ThreadPoolExecutor(max_workers=width) as pool:
            rows = list(pool.map(one, enumerate(batch)))
    else:
        rows = [one(pair) for pair in enumerate(batch)]
    wall = time.perf_counter() - started
    for index, row in enumerate(rows):
        row["prompt_index"] = index
    return {"rows": rows, "cell_wall_s": wall,
            "aggregate_tok_s": sum(r["completion_tokens"] for r in rows) / wall}


def run_arm(args, arm, rep):
    url = f"http://127.0.0.1:{args.port}"
    command = [
        sys.executable, "-m", "mlx2.server", "--model", args.model,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--max-context", str(args.max_context), "--max-lanes", "4",
        "--max-inflight", "8", "--qualification-mode", *arm_route_args(arm),
    ]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    out = Path(args.out)
    log_path = out.with_suffix(f".{arm}.r{rep}.server.log")
    guard = SwapGuard()
    log = open(log_path, "w")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    cells = []
    try:
        status = _wait_ready(url, process, args.startup_timeout, guard)
        route = {k: status.get(k) for k in ("route", "profile", "execution_policy", "adapter")}
        # Discarded warm-up: every workload, both temperatures, B1 and B4.
        for temperature in args.temperatures:
            for workload in WORKLOADS:
                for width in args.widths:
                    guard.check()
                    items = WORKLOADS[workload][:width]
                    with ThreadPoolExecutor(max_workers=width) as pool:
                        list(pool.map(lambda item: _request(
                            url, item, arm=arm, max_tokens=args.warmup_tokens,
                            temperature=temperature, timeout=args.timeout,
                            nonce=uuid.uuid4().hex, seed=7), items))
        guard.check()
        guard.arm_timed()
        seed_base = 1000
        for temperature in args.temperatures:
            for workload in WORKLOADS:
                for width in args.widths:
                    cell = run_cell(url, arm, workload, width, temperature, args, seed_base, rep)
                    guard.check()
                    status_after = _status(url)
                    cell.update({"arm": arm, "rep": rep, "workload": workload, "width": width,
                                 "temperature": temperature,
                                 "metal_peak_bytes": status_after.get("metal_peak_bytes"),
                                 "route_status": route})
                    cells.append(cell)
                    rates = [r["decode_tok_s"] for r in cell["rows"] if r["decode_tok_s"]]
                    print(json.dumps({"arm": arm, "rep": rep, "workload": workload,
                                      "width": width, "temperature": temperature,
                                      "decode_tok_s": [round(x, 2) for x in rates],
                                      "aggregate_tok_s": round(cell["aggregate_tok_s"], 2),
                                      "swap_peak_mib": guard.peak >> 20}), flush=True)
        final_status = _status(url)
    finally:
        process.terminate()
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
        guard.close()
    return cells, {"arm": arm, "rep": rep, "swap_peak_mib": guard.peak >> 20,
                   "scheduler": final_status.get("scheduler") if cells else None,
                   "metal_peak_bytes": final_status.get("metal_peak_bytes") if cells else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", required=True, help="comma list: ord,mtp2,dk3,dk5,dk6,dk7")
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--port", type=int, default=18731)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--warmup-tokens", type=int, default=48)
    parser.add_argument("--b1-prompts", type=int, default=3)
    parser.add_argument("--widths", default="1,4")
    parser.add_argument("--temperatures", default="0,1")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--nonce-salt", default="sp-dflash2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    args.widths = [int(v) for v in args.widths.split(",")]
    args.temperatures = [float(v) for v in args.temperatures.split(",")]
    arms = [a for a in args.arms.split(",") if a]
    for arm in arms:
        arm_route_args(arm)
    if args.dry_run:
        print(json.dumps({"arms": arms, "rep": args.rep, "widths": args.widths,
                          "temperatures": args.temperatures, "max_tokens": args.max_tokens}))
        return 0
    if not args.i_own_the_gpu:
        raise SystemExit("refusing: pass --i-own-the-gpu inside the GPU lock")
    out = Path(args.out)
    with open(out, "a") as stream:
        for arm in arms:
            cells, summary = run_arm(args, arm, args.rep)
            for cell in cells:
                stream.write(json.dumps(cell) + "\n")
            stream.write(json.dumps({"summary": summary}) + "\n")
            stream.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
