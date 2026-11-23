#!/usr/bin/env python3
"""Speculative-route A/B on copy-heavy and long-prose cells (complements GSM8K).

GSM8K answers rarely repeat their prompt, so prompt lookup parks on them; this
harness asks for outputs that copy long spans of the prompt (whole-file edits,
document rewrites, verbatim-quote summaries) and for long original prose, and
records per arm: mean acceptance length, per-position acceptance, decode tok/s,
completion tokens and greedy text parity against the first arm.

Cells are built from this repository's own files (no external text).  One
loopback server per arm, one at a time (``--max-lanes 1``), under the caller's
CPG lease and paired GPU locks; each server's process group is stopped in
``finally``.  Greedy, single rep, no thermal hold: sanity numbers, unqualified.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from spec_acceptance_gsm8k import (  # noqa: E402
    Client, acceptance_summary, find, median, revision, speculation_receipt, write_atomic)

COPY_SOURCES = {
    "code_rename": ("scripts/amoeba_staged_window.py",
                    "Return the complete file below with the function `in_window` renamed to `inside_window` "
                    "everywhere it appears (definition and every call). Change nothing else. Output only the code, "
                    "with no explanation and no code fence."),
    "code_docstrings": ("scripts/research/hy4_expert_kernel_bench.py",
                        "Return the complete file below unchanged except that every function without a docstring "
                        "gets a one-line docstring. Output only the code, with no explanation and no code fence."),
    "doc_rewrite": ("docs/hy4-flash/sparse-prefill.md",
                    "Rewrite the document below, replacing the word 'kernel' with 'shader' everywhere and keeping "
                    "every other word, line and character identical. Output only the rewritten document."),
    "summary_quotes": ("docs/hy4-flash/expert-read-once.md",
                       "Summarize the document below in about 300 words. Include the six most important sentences "
                       "quoted verbatim, each on its own line in quotation marks."),
}
PROSE = {
    "story": "Write an original short story of about 1,200 words about a lighthouse keeper who receives a letter "
             "that was posted forty years ago. Use dialogue and a clear ending.",
    "essay": "Explain, in about 1,000 words of plain prose without lists or headings, how a CPU cache hierarchy "
             "works and why it matters for program performance.",
}


def cells(max_chars: int) -> list[dict]:
    out = []
    for name, (path, instruction) in COPY_SOURCES.items():
        text = (ROOT / path).read_text()[:max_chars]
        out.append({"name": name, "kind": "copy", "source": path,
                    "content": f"{instruction}\n\n{text}"})
    for name, prompt in PROSE.items():
        out.append({"name": name, "kind": "prose", "source": None, "content": prompt})
    return out


def run_cells(client, model_name, items, max_tokens, request_extra, log):
    rows = []
    for item in items:
        start = time.monotonic()
        payload = {"model": model_name, "temperature": 0, "stream": False, "max_tokens": max_tokens,
                   **request_extra, "messages": [{"role": "user", "content": item["content"]}]}
        result = client.call("/v1/chat/completions", payload)
        elapsed = time.monotonic() - start
        choice = result["choices"][0]
        text = choice["message"].get("content") or ""
        usage = result.get("usage") or {}
        extra = result.get("mlx2") or {}
        ttft = find(result, "ttft_seconds")
        completion = usage.get("completion_tokens") or 0
        row = {"name": item["name"], "kind": item["kind"], "source": item["source"],
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": completion,
               "finish_reason": choice.get("finish_reason"), "elapsed_seconds": elapsed, "ttft_seconds": ttft,
               "decode_tps": ((completion - 1) / (elapsed - ttft)
                              if ttft and completion > 1 and elapsed > ttft else None),
               "route": extra.get("route"), "speculation": speculation_receipt(extra), "text": text}
        rows.append(row)
        log({k: row[k] for k in ("name", "kind", "prompt_tokens", "completion_tokens", "finish_reason",
                                 "decode_tps", "route")})
    return rows


def first_divergence(a: str, b: str):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--arms", type=Path, required=True, help='[{"name", "tree", "args", "env", "request"}]')
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8285)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-chars", type=int, default=9000, help="cap on each copy source, in characters")
    ap.add_argument("--max-context", type=int, default=16384)
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    if a.output.exists() and any(a.output.iterdir()):
        raise SystemExit(f"{a.output} is not empty")
    a.output.mkdir(parents=True, exist_ok=True)
    items = cells(a.max_chars)
    receipt = {"schema": "mlx2.spec-copy-prose-ab.v1", "qualification": "unqualified",
               "model": str(a.model), "driver_revision": revision(ROOT), "max_tokens": a.max_tokens,
               "cells": [{k: c[k] for k in ("name", "kind", "source")} for c in items], "arms": []}
    path = a.output / "receipt.json"
    client = Client(f"http://127.0.0.1:{a.port}")
    for arm in json.loads(a.arms.read_text()):
        out = a.output / arm["name"]
        out.mkdir()
        tree = Path(arm["tree"]).resolve()
        command = [a.python, "-u", "-m", "mlx2.server", "--model", str(a.model), "--host", "127.0.0.1",
                   "--port", str(a.port), "--max-inflight", "1", "--max-lanes", "1",
                   "--max-context", str(a.max_context), "--no-cache-dir", *arm.get("args", [])]
        log = (out / "server.log").open("w")
        env = {**os.environ, "PYTHONPATH": str(tree / "src"), **arm.get("env", {})}
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                  env=env, cwd=tree)
        row = {"name": arm["name"], "revision": revision(tree), "command": command}
        try:
            ready_by = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError("server exited before readiness")
                try:
                    status = client.call("/v1/status", timeout=10)
                    if status.get("healthy"):
                        break
                except (URLError, TimeoutError, ConnectionError, OSError):
                    pass
                if time.monotonic() > ready_by:
                    raise RuntimeError("server readiness timed out")
                time.sleep(.5)
            row["checkpoint"] = status.get("artifact") or find(status, "checkpoint")
            rows = run_cells(client, a.model.name, items, a.max_tokens, arm.get("request", {}),
                             lambda r: print(json.dumps({"arm": arm["name"], **r}), flush=True))
            row["rows"] = rows
            for kind in ("copy", "prose", None):
                sel = [r for r in rows if kind is None or r["kind"] == kind]
                row[f"summary_{kind or 'all'}"] = {
                    "acceptance": acceptance_summary([r["speculation"] for r in sel]),
                    "decode_tps_median": median([r["decode_tps"] for r in sel]),
                    "completion_tokens": sum(r["completion_tokens"] for r in sel)}
        except Exception as error:  # noqa: BLE001 - record the failed arm and go on
            row["error"] = repr(error)
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(60)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            log.close()
        receipt["arms"].append(row)
        write_atomic(path, receipt)
        print(json.dumps({"arm": arm["name"], "summary": {k: row.get(k) for k in row if k.startswith("summary")},
                          "error": row.get("error")}, default=str), flush=True)
    base = next((r for r in receipt["arms"] if r.get("rows")), None)
    if base:
        for arm in receipt["arms"]:
            if arm is base or not arm.get("rows"):
                continue
            arm["parity_vs_" + base["name"]] = {
                r["name"]: {"identical": r["text"] == b["text"], "first_divergence_char": first_divergence(r["text"], b["text"])}
                for r, b in zip(arm["rows"], base["rows"])}
    write_atomic(path, receipt)
    return 0 if all(not r.get("error") for r in receipt["arms"]) else 1


if __name__ == "__main__":
    sys.exit(main())
