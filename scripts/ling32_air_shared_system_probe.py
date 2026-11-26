#!/usr/bin/env python3
"""Shared-system-prompt APCv2 probe for one running loopback server.

Agent clients (Hermes) open every new conversation with the same long system
prompt and a different first user turn.  This sends ``--conversations`` such
requests (system prompt of roughly ``--system-tokens`` built from distinct
GSM8K test questions, no repeated strings; user turn a different GSM8K
question each time), greedy with thinking off, after one APCv2 clear, and
records per request: prompt/cached tokens, TTFT, elapsed, the server's mlx2
receipt (hit kind, miss reason, checkpoint counters) and the text.  ``compare``
reports text identity of a candidate arm against a reference arm per
conversation.  No GPU work here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_acceptance_gsm8k as gsm  # noqa: E402
from ling32_air_prefill_ab import find, prompt_text  # noqa: E402


def run(args):
    _, files = gsm.dataset_files()
    test = gsm.load_split(files["test"])
    client = gsm.Client(args.base_url, Path(args.admin_token).read_text().strip())
    model = gsm.served_model_name(client.call("/v1/status", timeout=30), args.checkpoint)
    system = ("You are a careful assistant. Reference problems follow; use them only as context.\n\n"
              + prompt_text(test, args.system_tokens).rsplit("\n\n", 1)[0])
    client.call("/v1/admin/apc/clear", {"reason": "shared-system-probe"}, admin=True)
    rows = []
    for index in range(args.conversations):
        question = test[len(test) - 1 - index]["question"]
        start = time.monotonic()
        result = client.call("/v1/chat/completions", {
            "model": model, "temperature": 0, "stream": False, "max_tokens": args.max_tokens,
            "enable_thinking": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": question + gsm.CHAT_INSTRUCTION}]})
        elapsed = time.monotonic() - start
        usage = result.get("usage") or {}
        receipt = result.get("mlx2") or {}
        rows.append({
            "conversation": index, "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "ttft_seconds": find(result, "ttft_seconds"), "elapsed_seconds": elapsed,
            "apcv2_miss_reason": find(receipt, "apcv2_miss_reason"),
            "apcv2_hit_kind": find(receipt, "apcv2_hit_kind"),
            "mlx2": receipt, "text": result["choices"][0]["message"].get("content")})
        print(json.dumps({k: rows[-1][k] for k in ("conversation", "prompt_tokens", "cached_tokens",
                                                  "ttft_seconds", "apcv2_miss_reason")}), flush=True)
    status = client.call("/v1/status", timeout=30)
    out = {"checkpoint": args.checkpoint, "arm": args.arm, "system_tokens_target": args.system_tokens,
           "rows": rows, "counts": status.get("counts"), "apcv2": status.get("apcv2"),
           "status_settings": status.get("settings")}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")


def compare(args):
    a, b = (json.loads(Path(p).read_text()) for p in (args.reference, args.candidate))
    if a["checkpoint"] != b["checkpoint"] or len(a["rows"]) != len(b["rows"]):
        raise ValueError("arms must cover the same checkpoint and conversations")
    cells = []
    for ra, rb in zip(a["rows"], b["rows"]):
        ta, tb = ra["text"] or "", rb["text"] or ""
        first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y),
                     None if len(ta) == len(tb) else min(len(ta), len(tb)))
        cells.append({"conversation": ra["conversation"], "identical": ta == tb,
                      "first_char_divergence": first,
                      "reference_cached": ra["cached_tokens"], "candidate_cached": rb["cached_tokens"],
                      "reference_ttft": ra["ttft_seconds"], "candidate_ttft": rb["ttft_seconds"],
                      "reference_answer": gsm.extract_answer(ta), "candidate_answer": gsm.extract_answer(tb)})
    out = {"reference": a["arm"], "candidate": b["arm"], "cells": cells}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", required=True)
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--admin-token", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--system-tokens", type=int, default=16384)
    r.add_argument("--conversations", type=int, default=4)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--output", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--reference", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    {"run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
