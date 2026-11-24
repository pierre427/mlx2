#!/usr/bin/env python3
"""Cold-prefill throughput + output identity probe for one running loopback server.

Builds deterministic long prompts from distinct GSM8K train questions (no repeated
strings) at roughly the requested token sizes, clears APCv2 before every request
(admin token), and records prompt tokens, TTFT, prefill tok/s, the greedy output
token trace, and the adapter's NAX engagement counters from /v1/status.
Run it against an arm with and one without ``ling_optimizations.moe_nax``; compare
the outputs with ``compare``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_acceptance_gsm8k as gsm  # noqa: E402


def find(document, key):
    stack = [document]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if key in item:
                return item[key]
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return None


def prompt_text(train, target_tokens):
    parts, chars = [], 0
    for index, item in enumerate(train):
        parts.append(f"Q{index + 1}. {item['question']}")
        chars += len(parts[-1]) + 1
        if chars >= target_tokens * 4:
            break
    return "\n".join(parts) + "\n\nHow many numbered questions are listed above? Reply with the number only."


def run(args):
    _, files = gsm.dataset_files()
    # The few-shot train file holds only 8 questions; the 1,319-question test split
    # gives ~4.4K / ~17.7K-token prompts for 4096 / 16384 (Ling tokenizer).
    train = gsm.load_split(files["test"])
    token = Path(args.admin_token).read_text().strip()
    client = gsm.Client(args.base_url, token)
    status = client.call("/v1/status", timeout=30)
    model = gsm.served_model_name(status, args.checkpoint)
    # Warm-up (not recorded): first-request graph and allocator setup.
    client.call("/v1/chat/completions", {"model": model, "temperature": 0, "max_tokens": 4,
                "enable_thinking": False, "messages": [{"role": "user", "content": prompt_text(train, 512)}]})
    rows = []
    for size in [int(v) for v in args.sizes.split(",")]:
        text = prompt_text(train, size)
        for repeat in range(args.repeats):
            client.call("/v1/admin/apc/clear", {"reason": "prefill-ab-cold"}, admin=True)
            result = client.call("/v1/chat/completions", {
                "model": model, "temperature": 0, "stream": False, "max_tokens": 24,
                "enable_thinking": False, "return_token_trace": True,
                "messages": [{"role": "user", "content": text}]})
            usage = result.get("usage") or {}
            ttft = find(result, "ttft_seconds")
            prompt = usage.get("prompt_tokens") or 0
            trace = find(result, "token_trace") or {}
            rows.append({"target_tokens": size, "repeat": repeat, "prompt_tokens": prompt,
                         "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                         "ttft_seconds": ttft,
                         "prefill_tok_s": (prompt / ttft) if ttft else None,
                         "text": result["choices"][0]["message"].get("content"),
                         "token_ids": trace.get("token_ids")})
            print(json.dumps({k: rows[-1][k] for k in ("target_tokens", "repeat", "prompt_tokens",
                                                      "ttft_seconds", "prefill_tok_s")}), flush=True)
    import time
    time.sleep(3)  # the status snapshot can lag one request
    status = client.call("/v1/status", timeout=30)
    out = {"checkpoint": args.checkpoint, "rows": rows, "moe_nax_gather": find(status, "moe_nax_gather")}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")


def compare(args):
    a = json.loads(Path(args.reference).read_text())
    b = json.loads(Path(args.candidate).read_text())
    rows = []
    for x, y in zip(a["rows"], b["rows"]):
        rows.append({"target_tokens": x["target_tokens"], "repeat": x["repeat"],
                     "identical_tokens": x["token_ids"] == y["token_ids"] and x["token_ids"] is not None,
                     "reference_tok_s": x["prefill_tok_s"], "candidate_tok_s": y["prefill_tok_s"],
                     "speedup": (y["prefill_tok_s"] / x["prefill_tok_s"])
                     if x["prefill_tok_s"] and y["prefill_tok_s"] else None})
    out = {"rows": rows, "candidate_nax": b.get("moe_nax_gather"), "reference_nax": a.get("moe_nax_gather")}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", required=True)
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--admin-token", required=True)
    r.add_argument("--sizes", default="4096,16384")
    r.add_argument("--repeats", type=int, default=2)
    r.add_argument("--output", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--reference", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    {"run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
