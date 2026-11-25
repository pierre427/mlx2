#!/usr/bin/env python3
"""Long-context decode probe for one running loopback server (B1).

Builds deterministic prompts of roughly the requested token sizes from distinct
GSM8K test questions (no repeated strings), asks for a fixed-length answer, and
records per request: prompt/cached tokens, TTFT, decode tok/s
((completion - 1) / (elapsed - ttft)), the greedy token ids, and the native-MTP
segmented acceptance counters from /v1/status (before/after deltas).  The first
request per size is cold (APCv2 cleared); the repeats are warm, so decode is
measured without a long prefill in the way.  ``compare`` reports decode speedups
and token identity of candidate arms against a reference arm.  No GPU work here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_acceptance_gsm8k as gsm  # noqa: E402
from ling32_air_prefill_ab import find, prompt_text  # noqa: E402

COUNTERS = ("batched_draft_forwards", "accepted_zero", "accepted_partial", "accepted_all")


def counters(status):
    block = find(status, "segmented_self_mtp") or {}
    return {key: int(block.get(key) or 0) for key in COUNTERS}


def run(args):
    _, files = gsm.dataset_files()
    test = gsm.load_split(files["test"])
    client = gsm.Client(args.base_url, Path(args.admin_token).read_text().strip())
    model = gsm.served_model_name(client.call("/v1/status", timeout=30), args.checkpoint)
    instruction = ("\n\nSummarise the kinds of problems listed above in about 150 words, "
                   "as plain prose.")
    rows = []
    for size in [int(v) for v in args.sizes.split(",")]:
        text = prompt_text(test, size).rsplit("\n\n", 1)[0] + instruction
        for repeat in range(args.repeats):
            if repeat == 0:
                client.call("/v1/admin/apc/clear", {"reason": "long-decode-cold"}, admin=True)
            before = counters(client.call("/v1/status", timeout=30))
            start = time.monotonic()
            result = client.call("/v1/chat/completions", {
                "model": model, "temperature": 0, "stream": False, "max_tokens": args.max_tokens,
                "enable_thinking": False, "return_token_trace": True,
                "messages": [{"role": "user", "content": text}]})
            elapsed = time.monotonic() - start
            time.sleep(1)  # the status snapshot can lag one request
            after = counters(client.call("/v1/status", timeout=30))
            usage = result.get("usage") or {}
            completion = usage.get("completion_tokens") or 0
            ttft = find(result, "ttft_seconds")
            delta = {key: after[key] - before[key] for key in COUNTERS}
            rounds = delta["accepted_zero"] + delta["accepted_partial"] + delta["accepted_all"]
            rows.append({
                "target_tokens": size, "repeat": repeat, "cold": repeat == 0,
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "completion_tokens": completion, "elapsed_seconds": elapsed, "ttft_seconds": ttft,
                "decode_tok_s": ((completion - 1) / (elapsed - ttft)
                                 if ttft and completion > 1 and elapsed > ttft else None),
                "mtp_rounds": rounds,
                "tokens_per_round": ((rounds + delta["accepted_partial"] + 2 * delta["accepted_all"]) / rounds
                                     if rounds else None),
                "mtp_counters": delta,
                "route": (result.get("mlx2") or {}).get("route"),
                "token_ids": (find(result, "token_trace") or {}).get("token_ids"),
                "text": result["choices"][0]["message"].get("content")})
            print(json.dumps({k: rows[-1][k] for k in ("target_tokens", "repeat", "prompt_tokens",
                                                      "decode_tok_s", "tokens_per_round")}), flush=True)
    status = client.call("/v1/status", timeout=30)
    summary = {}
    for size in sorted({r["target_tokens"] for r in rows}):
        warm = [r["decode_tok_s"] for r in rows if r["target_tokens"] == size and not r["cold"] and r["decode_tok_s"]]
        summary[str(size)] = {"warm_decode_tok_s_median": statistics.median(warm) if warm else None}
    out = {"checkpoint": args.checkpoint, "arm": args.arm, "rows": rows, "summary": summary,
           "optimizations": find(status, "optimizations"), "status_settings": status.get("settings")}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")


def compare(args):
    reference = json.loads(Path(args.reference).read_text())
    out = {"reference": reference["arm"], "candidates": {}}
    for path in args.candidates:
        candidate = json.loads(Path(path).read_text())
        cells = {}
        for size, ref in reference["summary"].items():
            cand = candidate["summary"].get(size) or {}
            r, c = ref["warm_decode_tok_s_median"], cand.get("warm_decode_tok_s_median")
            ref_ids = [x["token_ids"] for x in reference["rows"] if str(x["target_tokens"]) == size and x["cold"]]
            cand_ids = [x["token_ids"] for x in candidate["rows"] if str(x["target_tokens"]) == size and x["cold"]]
            first = None
            if ref_ids and cand_ids and ref_ids[0] and cand_ids[0]:
                first = next((i for i, (a, b) in enumerate(zip(ref_ids[0], cand_ids[0])) if a != b),
                             None if len(ref_ids[0]) == len(cand_ids[0]) else min(len(ref_ids[0]), len(cand_ids[0])))
            cells[size] = {"reference_tok_s": r, "candidate_tok_s": c,
                           "speedup": (c / r) if r and c else None,
                           "cold_first_divergence": first}
        out["candidates"][candidate["arm"]] = cells
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
    r.add_argument("--sizes", default="4096,16384,32768")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--output", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--reference", required=True)
    c.add_argument("--candidates", nargs="+", required=True)
    c.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    {"run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
