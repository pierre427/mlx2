#!/usr/bin/env python3
"""TTFT A/B for last-layer tail-row prefill (MLX2_PREFILL_TAIL_ROW, TensorFold#2229).

One server per arm (counterbalanced off,on,on,off), B=1, APCv2 cleared before
every request; each rep streams a long distinct-document prompt and records
TTFT, decode tokens and the full greedy text (compared across arms).
"""

from __future__ import annotations

import argparse
import json
import secrets
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_rows_attention_ab as h  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default="0,1,1,0")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--chars", default="60000,120000")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-context", type=int, default=40960)
    parser.add_argument("--port", type=int, default=8411)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty evidence dir {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="tail-ab-"))
    token = secrets.token_hex(16)
    token_file = workdir / "admin.token"
    token_file.write_text(token + "\n")
    token_file.chmod(0o600)
    docs = sorted((h.ROOT / "docs").rglob("*.md"))
    prompts = {int(c): h._document(docs, int(c)) + h.QUESTION for c in args.chars.split(",")}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=h.ROOT, capture_output=True,
                              text=True).stdout.strip()
    results = {"model": str(args.model), "source_revision": revision, "arms": [],
               "swapouts_start": h._swapouts()}
    import os

    for index, flag in enumerate(args.arms.split(",")):
        os.environ["MLX2_PREFILL_TAIL_ROW"] = flag
        proc, base = h._serve(args.model, "sdpa", args.port + index, token_file, workdir, args.max_context)
        arm = {"tail_row": flag, "cells": {}}
        try:
            for chars, prompt in prompts.items():
                cell = []
                for _ in range(args.reps):
                    h._post(base, "/v1/admin/apc/clear", {"reason": "tail-ab"}, token)
                    record = {}
                    h._stream(base, prompt, args.max_tokens, record)
                    stamps = record["stamps"]
                    cell.append({"ttft_s": round(stamps[0] - record["sent"], 4) if stamps else None,
                                 "tokens": len(stamps), "usage": record.get("usage"),
                                 "text": record["text"]})
                arm["cells"][str(chars)] = cell
            final = h._get(base, "/v1/status")
            arm["prefill_tail_row"] = h._find(final, "prefill_tail_row")
            arm["checkpoint"] = h._find(final, "checkpoint")
        finally:
            h._stop(proc)
        summary = {c: statistics.median(r["ttft_s"] for r in cell) for c, cell in arm["cells"].items()}
        arm["ttft_median_s"] = summary
        results["arms"].append(arm)
        print(json.dumps({"tail_row": flag, "ttft_median_s": summary,
                          "receipt": arm["prefill_tail_row"]}), flush=True)
    results["swapouts_end"] = h._swapouts()
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
