#!/usr/bin/env python3
"""Concurrent GSM8K cell for one running loopback server (B-width route check).

Sends GSM8K test questions (chat-fewshot, greedy, thinking off, the project
standard prompt from ``spec_acceptance_gsm8k``) in batches of ``--width``
concurrent requests, so a server with ``--max-lanes N`` decodes N lanes at
once.  Per batch it records the wall time, the aggregate completion tok/s
(sum of completion tokens / batch wall) and the sum of per-lane decode tok/s
(the best-route rule's B4 metric); per request the answer, oracle verdict,
route, text and timing.  ``compare`` reports accuracy, answer agreement and
text identity of a candidate against a reference arm.  No GPU work here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_acceptance_gsm8k as gsm  # noqa: E402


def one(client, model, shots, item, max_tokens):
    payload = {"model": model, "temperature": 0, "stream": False, "max_tokens": max_tokens,
               "enable_thinking": False,
               "messages": list(shots) + [{"role": "user", "content": item["question"] + gsm.CHAT_INSTRUCTION}]}
    start = time.monotonic()
    result = client.call("/v1/chat/completions", payload)
    elapsed = time.monotonic() - start
    choice = result["choices"][0]
    text = choice["message"].get("content") or ""
    completion = (result.get("usage") or {}).get("completion_tokens") or 0
    ttft = gsm.find(result, "ttft_seconds")
    predicted, expected = gsm.extract_answer(text), gsm.reference_answer(item["answer"])
    return {"test_index": item["test_index"], "expected": expected, "predicted": predicted,
            "correct": predicted is not None and predicted == expected,
            "finish_reason": choice.get("finish_reason"), "completion_tokens": completion,
            "elapsed_seconds": elapsed, "ttft_seconds": ttft,
            "decode_tps": ((completion - 1) / (elapsed - ttft) if ttft and completion > 1 and elapsed > ttft else None),
            "route": (result.get("mlx2") or {}).get("route"), "text": text}


def run(args):
    _, files = gsm.dataset_files()
    train, test = gsm.load_split(files["train"]), gsm.load_split(files["test"])
    shots = gsm.few_shot_turns(train, 5)
    items = [dict(test[i], test_index=i) for i in range(args.offset, args.offset + args.questions)]
    client = gsm.Client(args.base_url, None)
    model = gsm.served_model_name(client.call("/v1/status", timeout=30), args.checkpoint)
    one(client, model, shots, items[0], 8)  # warm-up, not recorded
    batches = []
    with ThreadPoolExecutor(max_workers=args.width) as pool:
        for start in range(0, len(items), args.width):
            chunk = items[start:start + args.width]
            t0 = time.monotonic()
            rows = list(pool.map(lambda item: one(client, model, shots, item, args.max_tokens), chunk))
            wall = time.monotonic() - t0
            tokens = sum(r["completion_tokens"] for r in rows)
            lanes = [r["decode_tps"] for r in rows if r["decode_tps"]]
            batches.append({"wall_seconds": wall, "completion_tokens": tokens,
                            "aggregate_tok_s": tokens / wall if wall else None,
                            "lane_decode_tps_sum": sum(lanes) if len(lanes) == len(rows) else None,
                            "rows": rows})
            print(json.dumps({"batch": len(batches), "aggregate_tok_s": round(batches[-1]["aggregate_tok_s"], 2),
                              "correct": sum(r["correct"] for r in rows)}), flush=True)
    every = [r for b in batches for r in b["rows"]]
    out = {"checkpoint": args.checkpoint, "width": args.width, "questions": len(every),
           "prompt": "GSM8K chat 5-shot, greedy, enable_thinking false", "max_tokens": args.max_tokens,
           "accuracy": sum(r["correct"] for r in every) / len(every),
           "aggregate_tok_s_median": statistics.median(b["aggregate_tok_s"] for b in batches),
           "lane_decode_tps_sum_median": statistics.median(
               [b["lane_decode_tps_sum"] for b in batches if b["lane_decode_tps_sum"]] or [0]),
           "routes": sorted({str(r["route"]) for r in every}), "batches": batches,
           "status": client.call("/v1/status", timeout=30)}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")


def compare(args):
    a, b = (json.loads(Path(p).read_text()) for p in (args.reference, args.candidate))
    ra = {r["test_index"]: r for x in a["batches"] for r in x["rows"]}
    rb = {r["test_index"]: r for x in b["batches"] for r in x["rows"]}
    if (a["checkpoint"] != b["checkpoint"] or a["width"] != b["width"]
            or a["max_tokens"] != b["max_tokens"] or not ra or ra.keys() != rb.keys()
            or len(ra) != a["questions"] or len(rb) != b["questions"]):
        raise ValueError("batch arms must cover the same checkpoint, serving shape, and questions")
    keys = sorted(ra)
    out = {"reference_accuracy": a["accuracy"], "candidate_accuracy": b["accuracy"],
           "answers_differ": [k for k in keys if ra[k]["predicted"] != rb[k]["predicted"]],
           "texts_identical": sum(ra[k]["text"] == rb[k]["text"] for k in keys), "questions": len(keys),
           "reference_aggregate_tok_s": a["aggregate_tok_s_median"],
           "candidate_aggregate_tok_s": b["aggregate_tok_s_median"],
           "aggregate_speedup": b["aggregate_tok_s_median"] / a["aggregate_tok_s_median"],
           "reference_lane_sum": a["lane_decode_tps_sum_median"],
           "candidate_lane_sum": b["lane_decode_tps_sum_median"],
           "lane_sum_speedup": (b["lane_decode_tps_sum_median"] / a["lane_decode_tps_sum_median"]
                                if a["lane_decode_tps_sum_median"] else None)}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", required=True)
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--width", type=int, default=4)
    r.add_argument("--questions", type=int, default=16)
    r.add_argument("--offset", type=int, default=0)
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--output", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--reference", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    {"run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
