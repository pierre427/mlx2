#!/usr/bin/env python3
"""F5 A/B: time-credit speculative recovery (``reprobe_credit_share``).

Feature 5 (d5593d878) lets the reversible cost-per-commit controller earn
re-probe credit from clean ordinary elapsed time instead of a token backoff
clock. This driver serves one checkpoint per arm, one server at a time, and
for each arm runs

* ``scripts/spec_acceptance_gsm8k.py --base-url`` (project standard:
  acceptance length, per-position acceptance, accuracy), and
* a fixed prose set (chat, thinking off, greedy) recording tokens, decode
  rate and the controller receipt (``cost_per_commit_handoff``).

Arms come from a JSON list ``[{"name", "args"}]``. Prose tokens of every arm
are compared with the first arm (exact verification should be lossless at
width one); any divergence is recorded with its position for near-tie
classification. Sanity and attribution numbers, no thermal hold.

Usage: ``ab_time_credit_recovery.py owned OUT --model PATH --arms ARMS``
(CPG lease + paired GPU locks) or ``unit OUT ...`` inside a lock.
Original mlx2 code.
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
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
PORT = 8392

PROSE = [
    ("Write a short story about a lighthouse keeper who finds a message in a bottle "
     "written in a language nobody in the village can read."),
    "Explain to a curious twelve-year-old why the sky looks blue during the day and red at sunset.",
    "Describe a bustling night market in a coastal city, focusing on smells, sounds and the people.",
    "Write a persuasive letter to a city council asking them to plant more trees along a busy road.",
    "Compare the experience of reading a paper book with reading on a tablet, with personal anecdotes.",
    "Tell the story of a stubborn mule that refuses to cross a bridge, from the mule's point of view.",
]


def checkout_env() -> dict:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT / "src"), environment.get("PYTHONPATH", "")) if p)
    return environment


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def source() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {"commit": git("rev-parse", "HEAD"),
            "dirty_tracked": bool(git("status", "--porcelain", "--untracked-files=no"))}


def host_sample() -> dict:
    return {"thermal": subprocess.check_output(["pmset", "-g", "therm"], text=True),
            "swap": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()}


def _find(tree, key):
    if isinstance(tree, dict):
        if key in tree:
            return tree[key]
        for value in tree.values():
            found = _find(value, key)
            if found is not None:
                return found
    elif isinstance(tree, list):
        for value in tree:
            found = _find(value, key)
            if found is not None:
                return found
    return None


def call(path, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
    try:
        return json.loads(body)
    except ValueError:
        return body.decode()


def prose_cell(model_name: str, prompt: str) -> dict:
    payload = {"model": model_name, "temperature": 0, "stream": False, "max_tokens": 384,
               "enable_thinking": False, "return_token_trace": True,
               "messages": [{"role": "user", "content": prompt}]}
    start = time.monotonic()
    result = call("/v1/chat/completions", payload)
    elapsed = time.monotonic() - start
    usage = result.get("usage") or {}
    extra = result.get("mlx2") or {}
    ttft = _find(result, "ttft_seconds")
    completion = usage.get("completion_tokens") or 0
    return {"completion_tokens": completion, "elapsed_seconds": elapsed, "ttft_seconds": ttft,
            "decode_tps": ((completion - 1) / (elapsed - ttft)
                           if ttft and completion > 1 and elapsed > ttft else None),
            "token_ids": (extra.get("token_trace") or {}).get("token_ids"),
            "route": extra.get("route"),
            "cost_per_commit_handoff": _find(extra, "cost_per_commit_handoff"),
            "accept_hist": _find(extra, "verify_accept_hist"),
            "text_head": (result["choices"][0]["message"].get("content") or "")[:160]}


def first_divergence(a, b):
    a, b = a or [], b or []
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"position": index, "tokens": [x, y]}
    return None if len(a) == len(b) else {"position": min(len(a), len(b)), "length_only": True}


def run_arm(out: Path, model: Path, checkpoint: str, arm: dict, questions: int) -> dict:
    arm_dir = out / arm["name"]
    arm_dir.mkdir()
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(model),
               "--host", "127.0.0.1", "--port", str(PORT), "--max-lanes", "1", "--max-inflight", "1",
               "--max-context", "8192", "--cache-bytes", str(2 << 30),
               "--cache-dir", str(arm_dir / "cache"), *arm.get("args", [])]
    row = {"name": arm["name"], "command": command, "host_before": host_sample()}
    log = (arm_dir / "server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              env=checkout_env(), cwd=ROOT)
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"{arm['name']}: server exited before readiness")
            try:
                status = call("/v1/status", timeout=10)
                if isinstance(status, dict) and status.get("healthy"):
                    break
            except (URLError, TimeoutError, ConnectionError, OSError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"{arm['name']}: readiness timeout")
            time.sleep(0.5)
        row["startup_route"] = _find(status, "route_receipt")
        # Fail fast if the arm's controller intent did not reach the route.
        probe = prose_cell(model.name, PROSE[0])
        row["controller_probe"] = probe["cost_per_commit_handoff"]
        gsm = arm_dir / "gsm8k"
        subprocess.run([sys.executable, str(ROOT / "scripts/spec_acceptance_gsm8k.py"),
                        "--base-url", f"http://127.0.0.1:{PORT}", "--checkpoint", checkpoint,
                        "--output", str(gsm), "--num-questions", str(questions)],
                       check=True, cwd=ROOT, env=checkout_env(), timeout=7200)
        gsm_receipt = json.loads((gsm / "receipt.json").read_text())
        row["gsm8k"] = gsm_receipt["arms"][0]["summary"]
        row["prose"] = [prose_cell(model.name, prompt) for prompt in PROSE]
        row["final_status_speculation"] = _find(call("/v1/status"), "cost_per_commit_handoff")
        metrics = call("/metrics")
        (arm_dir / "metrics.prom").write_text(metrics if isinstance(metrics, str) else json.dumps(metrics))
        row["passed"] = True
    except BaseException as exc:
        row.update(passed=False, error=repr(exc))
        raise
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=30)
        log.close()
        row["server_stopped"] = server.poll() is not None
        row["host_after"] = host_sample()
    return row


def unit(out: Path, model: Path, arms_path: Path, questions: int) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.registry import inspect_model

    checkpoint = f"{model.name}@sha256:{inspect_model(model).artifact_fingerprint}"
    arms = json.loads(arms_path.read_text())
    receipt = {"schema": "mlx2.f5-time-credit-ab.v1", "source": source(), "checkpoint": checkpoint,
               "arms_file": arms, "questions": questions, "qualified": False,
               "kind": "sanity/attribution A/B, no thermal hold", "arms": []}
    try:
        for arm in arms:
            try:
                receipt["arms"].append(run_arm(out, model, checkpoint, arm, questions))
            finally:
                atomic_json(out / "receipt.json", receipt)
        base = receipt["arms"][0]
        for arm in receipt["arms"][1:]:
            arm["prose_vs_first_arm"] = [first_divergence(x["token_ids"], y["token_ids"])
                                         for x, y in zip(base["prose"], arm["prose"])]
        receipt["passed"] = all(a.get("passed") for a in receipt["arms"])
    except BaseException as exc:
        receipt.update(passed=False, error=repr(exc))
        raise
    finally:
        atomic_json(out / "receipt.json", receipt)
    summary = [{"arm": a["name"], "acc": (a.get("gsm8k") or {}).get("accuracy"),
                "mal": ((a.get("gsm8k") or {}).get("acceptance") or {}).get("mean_acceptance_length"),
                "gsm_tps": (a.get("gsm8k") or {}).get("decode_tps_median")} for a in receipt["arms"]]
    print(json.dumps({"passed": receipt["passed"], "summary": summary}), flush=True)
    return 0 if receipt["passed"] else 1


def owned(out: Path, model: Path, arms: Path, questions: int, job_id: str) -> int:
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    with DesktopCPGLease(out, job_id, checkpoint=model.name, source_revision=source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out), "--model", str(model),
                   "--arms", str(arms), "--questions", str(questions)]
        child = subprocess.Popen(command, cwd=ROOT, env=checkout_env(), start_new_session=True)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=100)
    parser.add_argument("--job-id", default="f5-time-credit-ab-20261105")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode == "unit":
        return unit(out, args.model, args.arms.resolve(), args.questions)
    return owned(out, args.model, args.arms.resolve(), args.questions, args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
