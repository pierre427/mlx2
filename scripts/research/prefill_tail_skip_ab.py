#!/usr/bin/env python3
"""Cold-TTFT A/B for the prompt-chunk tail skip (MLX2_PREFILL_TAIL_SKIP).

One ``--ordinary`` server per arm, arms counterbalanced (default 0,1,1,0),
B=1, APCv2 cleared before every request.  Each cell is a distinct-document
prompt sized to a target token count; every rep records cold TTFT, decode
token stamps, prompt tokens and the full greedy text, and each arm records
the server's ``prefill_tail_skip`` receipt so engagement is checked, not
assumed.  Texts are compared across arms per cell.

Usage (under the shared GPU queue):
  PYTHONPATH=$PWD/src python scripts/research/prefill_tail_skip_ab.py \
      --model ~/mlx-models/<model> --out qualification/runs/<dir> --arms 0,1,1,0
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_rows_attention_ab as h

# Repository documents run at roughly this many characters per token for the
# tested tokenizers; the measured prompt_tokens are recorded per rep.
CHARS_PER_TOKEN = 3.6


def _stream(base, prompt, max_tokens, record, speculation):
    """Greedy streamed chat; ``speculation`` None leaves the route's default."""
    import time
    import urllib.request

    body = {"model": h.MODEL_ID[0], "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "enable_thinking": False, "stream_options": {"include_usage": True}}
    if speculation is not None:
        body["speculation"] = speculation
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
    record["sent"] = time.perf_counter()
    stamps, text = [], []
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                record["usage"] = chunk["usage"]
            for choice in chunk.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    stamps.append(time.perf_counter())
                    text.append(piece)
    record["stamps"] = stamps
    record["text"] = "".join(text)


def _serve(model, port, token_file, workdir, max_context, extra, route="--ordinary"):
    log = open(workdir / f"server-{port}.log", "w")
    cmd = [sys.executable, "-m", "mlx2.server", route, "--model", str(model),
           "--host", "127.0.0.1", "--port", str(port), "--max-lanes", "1",
           "--max-inflight", "1", "--max-context", str(max_context),
           "--admin-token-file", str(token_file), "--cache-dir", str(workdir / f"cache-{port}"),
           *extra]
    proc = subprocess.Popen(cmd, env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    import time

    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode}; see {log.name}")
        try:
            if h._get(base, "/health"):
                h.MODEL_ID[0] = h._get(base, "/v1/models")["data"][0]["id"]
                return proc, base
        except OSError:
            time.sleep(1)
    h._stop(proc)
    raise RuntimeError("server did not become healthy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default="0,1,1,0")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--tokens", default="2048,16384,32768")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--max-context", type=int, default=40960)
    parser.add_argument("--port", type=int, default=8431)
    parser.add_argument("--route", default="--ordinary",
                        help="server route flag (--ordinary, or --native-mtp for the reader check)")
    parser.add_argument("--server-arg", action="append", default=[],
                        help="extra mlx2.server argument (repeatable)")
    parser.add_argument("--abort-swapouts", type=int, default=20000,
                        help="abort when vm_stat Swapouts rises by more than this")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty evidence dir {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="tailskip-ab-"))
    token = secrets.token_hex(16)
    token_file = workdir / "admin.token"
    token_file.write_text(token + "\n")
    token_file.chmod(0o600)
    docs = sorted((h.ROOT / "docs").rglob("*.md"))
    targets = [int(t) for t in args.tokens.split(",")]
    prompts = {}
    for index, target in enumerate(targets):
        # A different starting document per cell keeps cells distinct.
        start = (index * len(docs)) // max(1, len(targets))
        prompts[target] = h._document(docs[start:] + docs[:start],
                                      int(target * CHARS_PER_TOKEN)) + h.QUESTION
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=h.ROOT, capture_output=True,
                              text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=h.ROOT,
                           capture_output=True, text=True).stdout.strip()
    results = {"model": str(args.model), "source_revision": revision, "dirty": bool(dirty),
               "arms_order": args.arms, "route": args.route, "server_args": args.server_arg, "arms": [],
               "swapouts_start": h._swapouts()}
    # The ordinary route pins speculation off; another route keeps its own.
    speculation = "off" if args.route == "--ordinary" else None
    try:
        for index, flag in enumerate(args.arms.split(",")):
            os.environ["MLX2_PREFILL_TAIL_SKIP"] = flag
            proc, base = _serve(args.model, args.port + index, token_file, workdir,
                                args.max_context, args.server_arg, args.route)
            arm = {"tail_skip": flag, "cells": {}}
            # Swap is watched from server-ready (measurement), not across the
            # load, which can coincide with unrelated system paging.
            arm_swap0 = h._swapouts()
            arm["swapouts_at_ready"] = arm_swap0
            try:
                # Ordinary-reference smoke: a short factual answer per arm.
                smoke = {}
                _stream(base, "In one sentence: what is the capital of France, and on "
                          "which river does it stand?", 40, smoke, speculation)
                arm["smoke_text"] = smoke["text"]
                for target, prompt in prompts.items():
                    cell = []
                    for _ in range(args.reps):
                        h._post(base, "/v1/admin/apc/clear", {"reason": "tailskip-ab"}, token)
                        record = {}
                        _stream(base, prompt, args.max_tokens, record, speculation)
                        stamps = record["stamps"]
                        cell.append({
                            "ttft_s": round(stamps[0] - record["sent"], 4) if stamps else None,
                            "decode_tok_s": (round((len(stamps) - 1) / (stamps[-1] - stamps[0]), 2)
                                             if len(stamps) > 2 else None),
                            "tokens": len(stamps), "usage": record.get("usage"),
                            "text": record["text"]})
                        rise = h._swapouts() - arm_swap0
                        if rise > args.abort_swapouts:
                            raise RuntimeError(f"swapouts rose by {rise}; aborting")
                    arm["cells"][str(target)] = cell
                final = h._get(base, "/v1/status")
                arm["prefill_tail_skip"] = h._find(final, "prefill_tail_skip")
                arm["prefill_tail_row"] = h._find(final, "prefill_tail_row")
                arm["checkpoint"] = h._find(final, "checkpoint")
            finally:
                h._stop(proc)
            arm["ttft_median_s"] = {c: statistics.median(r["ttft_s"] for r in cell)
                                    for c, cell in arm["cells"].items()}
            results["arms"].append(arm)
            print(json.dumps({"tail_skip": flag, "ttft_median_s": arm["ttft_median_s"],
                              "receipt": arm["prefill_tail_skip"]}), flush=True)
    finally:
        results["swapouts_end"] = h._swapouts()
        (args.out / "results.json").write_text(json.dumps(results, indent=1))
    # Summary: per cell, pooled TTFT by flag and text equality against arm 0.
    summary = {}
    for target in prompts:
        key = str(target)
        by_flag = {}
        for arm in results["arms"]:
            by_flag.setdefault(arm["tail_skip"], []).extend(r["ttft_s"] for r in arm["cells"][key])
        texts = {(arm["tail_skip"], r["text"]) for arm in results["arms"] for r in arm["cells"][key]}
        summary[key] = {
            "prompt_tokens": results["arms"][0]["cells"][key][0]["usage"],
            "ttft_median_s": {f: statistics.median(v) for f, v in by_flag.items()},
            "ttft_spread_s": {f: [min(v), max(v)] for f, v in by_flag.items()},
            "distinct_texts": len({t for _, t in texts}),
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
