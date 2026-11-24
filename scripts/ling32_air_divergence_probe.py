#!/usr/bin/env python3
"""First-divergence probe for a speculative route against ordinary (GSM8K prompts).

Drives an already-running mlx2 server (``--base-url``) with the exact chat-fewshot
greedy payload of ``spec_acceptance_gsm8k.py`` and records per question:

  trace   -> the generated token ids (``return_token_trace``), content and receipt
  window  -> the top-N logprobs at one completion position per question
             (``logprobs_start == logprobs_end``; diagnostic only, no sampling change)

Compare two trace files offline (``compare``) to find each question's first
divergent position, then ask both servers for a window at that position and
``classify`` it against the AGENTS.md near-tie contract (0.5-nat ceiling).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_acceptance_gsm8k as gsm  # noqa: E402

CEILING = 0.5


def payload(model, train, item, *, max_tokens, shots, extra):
    return {"model": model, "temperature": 0, "stream": False, "max_tokens": max_tokens,
            "messages": list(gsm.few_shot_turns(train, shots))
                        + [{"role": "user", "content": item["question"] + gsm.CHAT_INSTRUCTION}],
            "enable_thinking": False, **extra}


def run(args):
    _, files = gsm.dataset_files()
    train, test = gsm.load_split(files["train"]), gsm.load_split(files["test"])
    client = gsm.Client(args.base_url)
    status = client.call("/v1/status", timeout=30)
    model = gsm.served_model_name(status, args.checkpoint)
    positions = json.loads(Path(args.positions).read_text()) if args.positions else {}
    out = {"checkpoint": args.checkpoint, "mode": args.mode, "route": status.get("route_receipt"),
           "rows": {}}
    for index in [int(i) for i in args.indices.split(",")]:
        extra = {}
        if args.mode == "trace":
            extra = {"return_token_trace": True}
        else:
            position = positions.get(str(index))
            if position is None:
                continue
            extra = {"logprobs": True, "top_logprobs": args.top, "logprobs_start": position,
                     "logprobs_end": position}
        result = client.call("/v1/chat/completions",
                             payload(model, train, test[index], max_tokens=args.max_tokens,
                                     shots=args.shots, extra=extra))
        choice = result["choices"][0]
        mlx2 = result.get("mlx2") or {}
        row = {"content": choice["message"].get("content") or "",
               "reasoning": choice["message"].get("reasoning_content") or "",
               "finish_reason": choice.get("finish_reason"),
               "completion_tokens": (result.get("usage") or {}).get("completion_tokens"),
               "route": mlx2.get("route")}
        if args.mode == "trace":
            row["token_trace"] = gsm.find(result, "token_trace")
        else:
            row["position"] = positions[str(index)]
            row["logprobs"] = choice.get("logprobs")
        out["rows"][str(index)] = row
        print(json.dumps({"index": index, "finish": row["finish_reason"],
                          "tokens": row["completion_tokens"]}), flush=True)
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")


def compare(args):
    a = json.loads(Path(args.reference).read_text())["rows"]
    b = json.loads(Path(args.candidate).read_text())["rows"]
    positions, report = {}, {}
    for index in sorted(set(a) & set(b), key=int):
        x, y = a[index]["token_trace"]["token_ids"], b[index]["token_trace"]["token_ids"]
        first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
        if first is None and len(x) != len(y):
            first = min(len(x), len(y))
        report[index] = {"first_divergence_completion_token": None if first is None else first + 1,
                         "reference_len": len(x), "candidate_len": len(y),
                         "reference_token": x[first] if first is not None and first < len(x) else None,
                         "candidate_token": y[first] if first is not None and first < len(y) else None}
        if first is not None:
            positions[index] = first + 1  # one-based completion position
    Path(args.output).write_text(json.dumps(positions) + "\n")
    print(json.dumps(report, indent=1))


def _top(row):
    content = (row.get("logprobs") or {}).get("content") or []
    if not content:
        return None, None
    entry = content[-1]
    tops = {str(t["id"]): float(t["logprob"])
            for t in entry.get("top_logprobs") or []}
    return str(entry["id"]), tops


def classify(args):
    ref = json.loads(Path(args.reference).read_text())["rows"]
    cand = json.loads(Path(args.candidate).read_text())["rows"]
    verdicts = {}
    for index in sorted(set(ref) & set(cand), key=int):
        rs, rt = _top(ref[index])
        cs, ct = _top(cand[index])
        if not rt or not ct:
            verdicts[index] = {"verdict": "unclassified", "why": "missing window"}
            continue
        r_top, c_top = max(rt.values()), max(ct.values())
        # Each arm's selected token must sit within the ceiling of the other arm's top.
        c_in_r = rt.get(cs)
        r_in_c = ct.get(rs)
        margins = {"ref_selected_vs_cand_top": None if r_in_c is None else c_top - r_in_c,
                   "cand_selected_vs_ref_top": None if c_in_r is None else r_top - c_in_r}
        known = all(v is not None for v in margins.values())
        near = known and all(v <= CEILING for v in margins.values())
        verdicts[index] = {"verdict": "near_tie_candidate" if near else
                           ("high_margin" if known else "unclassified"),
                           "position": ref[index].get("position"), "ref_selected": rs,
                           "cand_selected": cs, "margins_nats": margins,
                           "ref_top5": rt, "cand_top5": ct,
                           "note": "functional oracle and shared-prefix checks are separate"}
    Path(args.output).write_text(json.dumps(verdicts, indent=1) + "\n")
    print(json.dumps({k: (v["verdict"], v.get("margins_nats")) for k, v in verdicts.items()}, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", required=True)
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--mode", choices=("trace", "window"), required=True)
    r.add_argument("--indices", required=True)
    r.add_argument("--positions")
    r.add_argument("--top", type=int, default=5)
    r.add_argument("--shots", type=int, default=5)
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--output", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--reference", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--output", required=True)
    k = sub.add_parser("classify")
    k.add_argument("--reference", required=True)
    k.add_argument("--candidate", required=True)
    k.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    {"run": run, "compare": compare, "classify": classify}[args.command](args)


if __name__ == "__main__":
    main()
