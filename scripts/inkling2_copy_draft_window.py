#!/usr/bin/env python3
"""Inkling 2 copy-draft A/B: GSM8K acceptance gate plus a file-rewrite cell, per arm.

Original mlx2 script (recon 2026-11-22 items 1 and 3).

Inkling 2's DSpark route already composes prompt-lookup copies with the DSpark
draft by default: ``ExternalDraftAdapterMixin._bind_external_drafter`` wraps a
point-mass drafter in ``ComposedDraftModel`` with
``DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION`` unless the policy names
``proposal_composition``.  Every ladder and the 2026-11-21 GSM8K attempt pinned
``"proposal_composition": false``, so the copy arm was never measured.  This
window measures it.  Copies are capped at ``num_draft`` (7) here; omlx #5528
lets them run to 15 in a 16-row verify, which mlx2 does not do yet.

Arms (one loopback server each, never :8282, ``--max-lanes 1``):

  ordinary        --ordinary (shipped default kernels)
  dspark_nocopy   DSpark, num_draft 7, proposal_composition false (the ladder route)
  dspark_copy     DSpark, num_draft 7, composition at the shipped default

Per DSpark arm: ``spec_acceptance_gsm8k.py --base-url`` (the AGENTS.md
acceptance standard; 40 questions, max-tokens 1024, effort low, the shape of
the Inkling 2 session's 2026-11-22 run, so its arms compare).  Every arm then
runs the rewrite cell: greedy requests that ask the model to repeat
a source file with one rename, where verbatim copies dominate.  The rewrite
outputs of every arm are compared token for token with ``ordinary`` (first
divergence; this is the DSpark on/off identity check, item 3).

The caller owns the GPU (``cpg_job.py`` lease -> ``run_with_gpu_locks.py``).
Sanity and attribution numbers, no thermal hold: not publishable statistics.
Refuses a nonempty output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from spec_acceptance_gsm8k import (  # noqa: E402
    Client, find, revision, speculation_receipt, write_atomic,
)

SCHEMA = "mlx2.inkling2-copy-draft-window.v1"
DSPARK = {"draft_model": "embedded_dspark", "num_draft": 7}
ARMS = [
    # GSM8K accuracy reference for ordinary decode comes from the Inkling 2
    # session's same-day run (40 questions, effort low, 39/40), so this arm
    # serves only the rewrite cell and its token-identity reference.
    {"name": "ordinary", "args": ["--ordinary"], "gsm8k": False},
    {"name": "dspark_nocopy", "policy": {**DSPARK, "proposal_composition": False},
     "draft_depth": 7},
    {"name": "dspark_copy", "policy": dict(DSPARK), "draft_depth": 7},
]
REWRITES = [
    ("src/mlx2/runtime/models/inkling2/config.py", "LoopConfig", "PassLoopConfig"),
    ("src/mlx2/runtime/drafters/inkling2_dspark_external.py", "LastFeatureCache",
     "FeatureRowCache"),
]


def rewrite_prompt(text, old, new):
    return ("Below is a Python file. Rewrite the whole file with exactly one change: rename "
            f"`{old}` to `{new}` everywhere it appears. Keep every other character the same, "
            "including comments and blank lines. Output only the full file in one ```python "
            "code block, nothing else.\n\n```python\n" + text + "```")


def start_server(args, arm, out, token_file):
    command = [args.python, "-u", "-m", "mlx2.server", "--model", str(args.model),
               "--host", "127.0.0.1", "--port", str(args.port), "--max-inflight", "1",
               "--max-context", "16384", "--cache-bytes", str(2 << 30),
               "--cache-dir", str(out / "cache"), "--admin-token-file", str(token_file),
               "--max-lanes", "1", *arm.get("args", [])]
    if "policy" in arm:
        policy = out / "policy.json"
        policy.write_text(json.dumps(arm["policy"]) + "\n")
        command += ["--external-draft", "--execution-policy", str(policy)]
    log = (out / "server.log").open("w")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True, env=env, cwd=ROOT)
    return server, log, command


def wait_ready(server, client):
    ready_by = time.monotonic() + 900
    while True:
        if server.poll() is not None:
            raise RuntimeError("server exited before readiness")
        try:
            status = client.call("/v1/status", timeout=10)
            if status.get("healthy"):
                return status
        except (URLError, TimeoutError, ConnectionError, OSError):
            pass
        if time.monotonic() > ready_by:
            raise RuntimeError("server readiness timed out")
        time.sleep(.5)


def rewrite_cell(client, model_name, max_tokens):
    rows = []
    for path, old, new in REWRITES:
        text = (ROOT / path).read_text()
        payload = {"model": model_name, "temperature": 0, "stream": False,
                   "max_tokens": max_tokens, "reasoning_effort": "none", "logprobs": True,
                   "messages": [{"role": "user", "content": rewrite_prompt(text, old, new)}]}
        start = time.monotonic()
        result = client.call("/v1/chat/completions", payload)
        elapsed = time.monotonic() - start
        choice = result["choices"][0]
        content = choice["message"].get("content") or ""
        usage = result.get("usage") or {}
        extra = result.get("mlx2") or {}
        ttft = find(result, "ttft_seconds")
        completion = usage.get("completion_tokens") or 0
        expected = text.replace(old, new)
        body = content.split("```python\n", 1)[-1].rsplit("```", 1)[0]
        token_ids = [int(item["id"]) for item in ((choice.get("logprobs") or {}).get("content") or [])]
        if not token_ids:
            raise ValueError(f"missing rewrite token IDs for {path}")
        rows.append({
            "source": path, "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "rename": [old, new], "finish_reason": choice.get("finish_reason"),
            "completion_tokens": completion, "prompt_tokens": usage.get("prompt_tokens"),
            "elapsed_seconds": elapsed, "ttft_seconds": ttft,
            "decode_tps": ((completion - 1) / (elapsed - ttft)
                           if ttft and completion > 1 and elapsed > ttft else None),
            "oracle_exact_rewrite": body == expected,
            "route": extra.get("route"), "speculation": speculation_receipt(extra),
            "proposal_composition": find(extra, "proposal_composition"),
            "composition_stats": find(extra, "composition_stats"),
            "text_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "token_ids": token_ids, "text": content,
            "reasoning_chars": len(choice["message"].get("reasoning_content") or ""),
        })
    return rows


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--port", type=int, default=8397)
    ap.add_argument("--num-questions", type=int, default=40)
    ap.add_argument("--gsm8k-max-tokens", type=int, default=1024)
    ap.add_argument("--rewrite-max-tokens", type=int, default=4096)
    ap.add_argument("--arms", default=",".join(a["name"] for a in ARMS))
    args = ap.parse_args(argv)
    if args.port == 8282:
        raise SystemExit("port 8282 is reserved for release trees")
    wanted = args.arms.split(",")
    if len(set(wanted)) != len(wanted) or set(wanted) - {a["name"] for a in ARMS}:
        ap.error("--arms must list unique known arm names")
    if args.num_questions < 1:
        ap.error("--num-questions must be positive")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit(f"port {args.port} is occupied")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("refusing nonempty evidence directory")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.registry import inspect_model

    resolution = inspect_model(args.model)
    checkpoint = f"{args.model.name}@sha256:{resolution.artifact_fingerprint}"
    receipt = {"schema": SCHEMA, "qualified": False, "checkpoint": checkpoint,
               "driver_revision": revision(ROOT), "kind": "greedy, B1, no thermal hold",
               "swap_before": subprocess.check_output(["sysctl", "-n", "vm.swapusage"],
                                                      text=True).strip(),
               "arms": []}
    path = args.output / "receipt.json"
    token_file = args.output / "admin-token"
    token_file.write_text(os.urandom(24).hex())
    token_file.chmod(0o600)
    client = Client(f"http://127.0.0.1:{args.port}", token_file.read_text())
    for arm in [a for a in ARMS if a["name"] in wanted]:
        out = args.output / arm["name"]
        out.mkdir()
        server, log, command = start_server(args, arm, out, token_file)
        row = {"name": arm["name"], "command": command, "policy": arm.get("policy")}
        try:
            row["startup_status"] = wait_ready(server, client)
            gsm = None if arm.get("gsm8k") is False else subprocess.run(
                [args.python, "-u", str(ROOT / "scripts/spec_acceptance_gsm8k.py"),
                 "--base-url", f"http://127.0.0.1:{args.port}", "--checkpoint", checkpoint,
                 "--output", str(out / "gsm8k"), "--num-questions", str(args.num_questions),
                 "--max-tokens", str(args.gsm8k_max_tokens),
                 "--request", json.dumps({"reasoning_effort": "low"}),
                 *(["--draft-depth", str(arm["draft_depth"])] if "draft_depth" in arm else [])],
                cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                capture_output=True, text=True)
            if gsm is not None:
                (out / "gsm8k.log").write_text(gsm.stdout + gsm.stderr)
                row["gsm8k_returncode"] = gsm.returncode
            if gsm is not None and gsm.returncode == 0:
                g = json.loads((out / "gsm8k" / "receipt.json").read_text())
                row["gsm8k_summary"] = g["arms"][0]["summary"]
            row["rewrite"] = rewrite_cell(client, row["startup_status"].get("model") or args.model.name,
                                          args.rewrite_max_tokens)
            row["final_status"] = client.call("/v1/status")
            (out / "metrics.prom").write_text(client.call("/metrics"))
            row["passed"] = gsm is None or (
                gsm.returncode == 0
                and row["gsm8k_summary"].get("questions") == args.num_questions
                and (row["gsm8k_summary"].get("acceptance") or {}).get("verify_rounds", 0) > 0
            )
        except Exception as error:  # continue to the next arm; the receipt says why
            row["passed"] = False
            row["error"] = repr(error)
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)
            log.close()
            row["server_stopped"] = server.poll() is not None
            receipt["arms"].append(row)
            receipt["swap_after"] = subprocess.check_output(["sysctl", "-n", "vm.swapusage"],
                                                            text=True).strip()
            write_atomic(path, receipt)
        summary = row.get("gsm8k_summary") or {}
        print(json.dumps({"arm": arm["name"], "passed": row["passed"],
                          "gsm8k": {"accuracy": summary.get("accuracy"),
                                    "mean_acceptance_length": (summary.get("acceptance") or {}).get("mean_acceptance_length"),
                                    "decode_tps_median": summary.get("decode_tps_median")},
                          "rewrite_tps": [r["decode_tps"] for r in row.get("rewrite", [])],
                          "error": row.get("error")}), flush=True)
    reference = next((a for a in receipt["arms"] if a["name"] == "ordinary" and a.get("rewrite")),
                     None)
    receipt["rewrite_identity_vs_ordinary"] = [
        {"arm": a["name"], "source": r["source"],
         "first_divergent_token": first_divergence(ref["token_ids"], r["token_ids"]),
         "identical_text": ref["text_sha256"] == r["text_sha256"]}
        for a in receipt["arms"] if reference and a is not reference
        for r, ref in zip(a.get("rewrite", []), reference["rewrite"])]
    receipt["rewrite_identity_passed"] = (
        all(r["first_divergent_token"] is None and r["identical_text"]
            for r in receipt["rewrite_identity_vs_ordinary"])
        if receipt["rewrite_identity_vs_ordinary"] else None
    )
    write_atomic(path, receipt)
    print(json.dumps(receipt["rewrite_identity_vs_ordinary"]), flush=True)
    return 0 if all(a.get("passed") for a in receipt["arms"]) and receipt["rewrite_identity_passed"] is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
