#!/usr/bin/env python3
"""First-divergence probe for the Phase 0 16K prompt (design doc section 4.6 note (d)).

Drives an already-running mlx2 server with the exact deterministic request of
``spec_round_phase0.py`` (16384-size prompt, temperature 0, ``enable_thinking`` false) and
records, per request:

  trace   -> generated token ids (``return_token_trace``), content, its sha256, finish
             reason, route and cached tokens; ``--repeats`` requests (the first is cold)
  window  -> the top-N logprobs at ONE completion position (``logprobs_start ==
             logprobs_end``), warm; diagnostic only, no sampling change

``compare`` finds the first divergent completion position between two trace files.
``classify`` applies the AGENTS.md near-tie contract (0.5-nat ceiling, exact-tie and
multi-way clauses) to two windows taken at that position, plus a functional oracle on both
continuations (the prompt asks for a ~150-word plain-prose summary of the listed problems).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CEILING = 0.5
SIZE = 16384


def _prompt():
    import spec_round_phase0 as p0
    return p0.build_prompts([SIZE])[SIZE]


def _payload(model, text, max_tokens, extra):
    return {"model": model, "temperature": 0, "stream": False, "max_tokens": max_tokens,
            "enable_thinking": False, "messages": [{"role": "user", "content": text}], **extra}


def _client(args):
    import spec_acceptance_gsm8k as gsm
    token = Path(args.admin_token).read_text().strip() if args.admin_token else None
    client = gsm.Client(args.base_url, token)
    status = client.call("/v1/status", timeout=30)
    return client, status, gsm.served_model_name(status, args.checkpoint)


def _find(document, key):
    from ling32_air_prefill_ab import find
    return find(document, key)


def _row(result, elapsed):
    choice = (result.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    usage = result.get("usage") or {}
    mlx2 = result.get("mlx2") or {}
    return {"content": content or "",
            "text_sha256": hashlib.sha256(str(content).encode()).hexdigest(),
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "route": mlx2.get("route"), "elapsed_s": round(elapsed, 3),
            "mlx2_receipt": {k: v for k, v in mlx2.items() if k not in ("round_probe", "token_trace")}}


def run(args):
    client, status, model = _client(args)
    text = _prompt()
    out = {"checkpoint": args.checkpoint, "source_revision": args.source_revision,
           "arm": args.arm, "mode": args.mode, "size": SIZE, "max_tokens": args.max_tokens,
           "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "runtime": status.get("runtime") or _find(status, "runtime_identity"),
           "route_receipt": status.get("route_receipt"), "rows": []}
    if args.mode == "trace":
        for repeat in range(args.repeats):
            started = time.monotonic()
            result = client.call("/v1/chat/completions",
                                 _payload(model, text, args.max_tokens, {"return_token_trace": True}))
            row = _row(result, time.monotonic() - started)
            row["token_trace"] = _find(result, "token_trace")
            row["repeat"] = repeat
            out["rows"].append(row)
            print(json.dumps({k: row[k] for k in ("repeat", "cached_tokens", "completion_tokens",
                                                   "finish_reason", "route")}
                             | {"sha": row["text_sha256"][:8]}), flush=True)
    else:
        position = args.position
        if position is None:
            position = json.loads(Path(args.positions).read_text())["position"]
        if position is None:
            out["note"] = "no divergence: no window requested"
        else:
            extra = {"logprobs": True, "top_logprobs": args.top,
                     "logprobs_start": int(position), "logprobs_end": int(position)}
            if args.warm_first:
                client.call("/v1/chat/completions", _payload(model, text, 1, {}))
            started = time.monotonic()
            result = client.call("/v1/chat/completions",
                                 _payload(model, text, args.max_tokens, extra))
            row = _row(result, time.monotonic() - started)
            row["position"] = int(position)
            row["logprobs"] = (result.get("choices") or [{}])[0].get("logprobs")
            out["rows"].append(row)
            print(json.dumps({"position": position, "route": row["route"],
                              "cached_tokens": row["cached_tokens"],
                              "sha": row["text_sha256"][:8]}), flush=True)
    out["status_final"] = {"route_receipt": client.call("/v1/status", timeout=30).get("route_receipt")}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")


def _ids(path, repeat):
    doc = json.loads(Path(path).read_text())
    rows = doc["rows"]
    row = rows[repeat if repeat < len(rows) else -1]
    trace = row.get("token_trace") or {}
    ids = trace.get("token_ids") if isinstance(trace, dict) else None
    if not ids:
        raise ValueError(f"{path}: no token trace in row {repeat}")
    return doc, row, [int(t) for t in ids]


def first_divergence(x, y):
    """Zero-based index of the first differing token, or None when identical."""
    first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
    if first is None and len(x) != len(y):
        first = min(len(x), len(y))
    return first


def compare(args):
    rdoc, rrow, x = _ids(args.reference, args.repeat)
    cdoc, crow, y = _ids(args.candidate, args.repeat)
    if rdoc.get("prompt_sha256") != cdoc.get("prompt_sha256"):
        raise ValueError("the two traces did not send the same prompt")
    first = first_divergence(x, y)
    report = {"reference": {"arm": rdoc.get("arm"), "source_revision": rdoc.get("source_revision"),
                            "sha": rrow["text_sha256"], "tokens": len(x), "finish": rrow["finish_reason"]},
              "candidate": {"arm": cdoc.get("arm"), "source_revision": cdoc.get("source_revision"),
                            "sha": crow["text_sha256"], "tokens": len(y), "finish": crow["finish_reason"]},
              "repeat": args.repeat, "shared_prefix_tokens": len(x) if first is None else first,
              # One-based completion position, the unit of logprobs_start/logprobs_end.
              "position": None if first is None else first + 1,
              "reference_token": x[first] if first is not None and first < len(x) else None,
              "candidate_token": y[first] if first is not None and first < len(y) else None,
              "token_exact": first is None}
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report), flush=True)


def _window(path):
    doc = json.loads(Path(path).read_text())
    row = (doc.get("rows") or [{}])[0]
    content = (row.get("logprobs") or {}).get("content") or []
    if not content:
        return doc, row, None, None
    entry = content[-1]
    tops = {str(t["id"]): float(t["logprob"]) for t in entry.get("top_logprobs") or []}
    return doc, row, str(entry["id"]), tops


_KEYWORDS = ("money", "cost", "price", "time", "hour", "age", "percent", "rate", "distance",
             "speed", "fraction", "ratio", "arithmetic", "multiplication", "division", "word problem",
             "math", "earn", "sell", "buy", "total", "proportion", "measurement")


def oracle(text, finish_reason):
    """Functional oracle for the prompt: a plain-prose summary of the problems (~150 words)."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    grams = [" ".join(words[i:i + 8]).lower() for i in range(max(0, len(words) - 7))]
    repeated = max((grams.count(g) for g in set(grams)), default=0)
    hits = sorted({k for k in _KEYWORDS if k in text.lower()})
    checks = {"nonempty": bool(text.strip()),
              "length_40_to_260_words": 40 <= len(words) <= 260,
              "no_8gram_loop": repeated < 3,
              "on_topic_keywords_ge_3": len(hits) >= 3,
              "ascii_prose": sum(ord(c) > 0x2FFF for c in text) == 0}
    return {"pass": all(checks.values()), "checks": checks, "words": len(words),
            "keywords": hits, "finish_reason": finish_reason}


def classify(args):
    rdoc, rrow, rs, rt = _window(args.reference)
    cdoc, crow, cs, ct = _window(args.candidate)
    div = json.loads(Path(args.divergence).read_text())
    pick = lambda rows: rows[min(int(div.get("repeat", 0)), len(rows) - 1)]
    ref_trace = pick(json.loads(Path(args.reference_trace).read_text())["rows"])
    cand_trace = pick(json.loads(Path(args.candidate_trace).read_text())["rows"])
    verdict = {"position": div.get("position"), "shared_prefix_tokens": div.get("shared_prefix_tokens"),
               "reference_arm": rdoc.get("arm"), "candidate_arm": cdoc.get("arm"),
               "window_routes": [rrow.get("route"), crow.get("route")],
               "oracle": {"reference": oracle(ref_trace["content"], ref_trace["finish_reason"]),
                          "candidate": oracle(cand_trace["content"], cand_trace["finish_reason"])}}
    if div.get("token_exact"):
        verdict.update(verdict="token_exact")
    elif not rt or not ct:
        verdict.update(verdict="unclassified", why="missing window")
    elif (rrow.get("position") != div["position"] or crow.get("position") != div["position"]
          or rs != str(div["reference_token"]) or cs != str(div["candidate_token"])):
        verdict.update(verdict="unclassified",
                       why="windows do not reproduce the traced divergence",
                       window_selected=[rs, cs], traced=[div["reference_token"], div["candidate_token"]])
    else:
        r_top, c_top = max(rt.values()), max(ct.values())
        r_sorted, c_sorted = sorted(rt.values(), reverse=True), sorted(ct.values(), reverse=True)
        margins = {"reference_selected_vs_alternate": r_sorted[0] - r_sorted[1] if len(r_sorted) > 1 else None,
                   "candidate_selected_vs_alternate": c_sorted[0] - c_sorted[1] if len(c_sorted) > 1 else None,
                   # Multi-way clause: each arm's selected token within the ceiling of the other's top.
                   "candidate_selected_below_reference_top": None if cs not in rt else r_top - rt[cs],
                   "reference_selected_below_candidate_top": None if rs not in ct else c_top - ct[rs]}
        top2 = lambda t: set(sorted(t, key=t.get, reverse=True)[:2])
        same_top2 = top2(rt) == top2(ct)
        both_oracle = verdict["oracle"]["reference"]["pass"] and verdict["oracle"]["candidate"]["pass"]
        cross = [margins["candidate_selected_below_reference_top"],
                 margins["reference_selected_below_candidate_top"]]
        missing_far = ((cs not in rt and r_top - min(rt.values()) > CEILING)
                       or (rs not in ct and c_top - min(ct.values()) > CEILING))
        base = both_oracle
        two_way = (same_top2 and all(m is not None and m <= CEILING for m in
                                     (margins["reference_selected_vs_alternate"],
                                      margins["candidate_selected_vs_alternate"])))
        multi_way = all(m is not None and m <= CEILING for m in cross)
        exact_tie = any(m == 0.0 for m in (margins["reference_selected_vs_alternate"],
                                            margins["candidate_selected_vs_alternate"])) and multi_way
        if base and (two_way or multi_way or exact_tie):
            clause = "two_way" if two_way else ("exact_tie" if exact_tie else "multi_way")
            verdict.update(verdict="near_tie_equivalent", clause=clause)
        elif missing_far or any(m is not None and m > CEILING for m in cross):
            verdict.update(verdict="high_margin_divergence")
        elif not both_oracle:
            verdict.update(verdict="unclassified", why="a continuation fails the functional oracle")
        else:
            verdict.update(verdict="unclassified", why="a selected token is missing from the window")
        verdict.update(reference_selected=rs, candidate_selected=cs, margins_nats=margins,
                       same_top2=same_top2, reference_top=rt, candidate_top=ct, ceiling=CEILING)
    Path(args.output).write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps({k: verdict.get(k) for k in ("verdict", "clause", "position", "margins_nats", "why")}),
          flush=True)


def gsm8k_rows(path):
    arm = json.loads(Path(path).read_text())["arms"][0]
    return arm, {int(r["test_index"]): r for r in arm["rows"]}


def widths(args):
    """GSM8K receipts (base-url mode): ordinary B1, route B1, route B4 on the same questions."""
    ref_arm, ref = gsm8k_rows(args.ordinary)
    b1_arm, b1 = gsm8k_rows(args.b1)
    b4_arm, b4 = gsm8k_rows(args.b4)
    if ref.keys() != b1.keys() or b1.keys() != b4.keys():
        raise ValueError("the three GSM8K cells must cover the same questions")
    keys = sorted(b1)
    differ = [k for k in keys if b1[k]["text"] != b4[k]["text"]]
    routes = lambda rows: sorted({str(r.get("route")) for r in rows.values()})
    report = {"questions": len(keys),
              "b1_equals_b4": not differ, "b1_b4_texts_differ": differ,
              "b1_b4_answers_differ": [k for k in keys if b1[k]["predicted"] != b4[k]["predicted"]],
              "routes": {"ordinary": routes(ref), "b1": routes(b1), "b4": routes(b4)},
              "b4_concurrency": b4_arm.get("concurrency"),
              "accuracy": {"ordinary": ref_arm["summary"]["accuracy"], "b1": b1_arm["summary"]["accuracy"],
                           "b4": b4_arm["summary"]["accuracy"]},
              "summaries": {"ordinary": ref_arm["summary"], "b1": b1_arm["summary"], "b4": b4_arm["summary"]},
              "route_vs_ordinary_texts_differ": sum(ref[k]["text"] != b1[k]["text"] for k in keys),
              "route_correct_ordinary_wrong": [k for k in keys if b1[k]["correct"] and not ref[k]["correct"]],
              "ordinary_correct_route_wrong": [k for k in keys if ref[k]["correct"] and not b1[k]["correct"]]}
    Path(args.output).write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({k: report[k] for k in ("questions", "b1_equals_b4", "accuracy", "routes",
                                              "route_vs_ordinary_texts_differ")}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", required=True)
    r.add_argument("--admin-token")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--source-revision", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--mode", choices=("trace", "window"), required=True)
    r.add_argument("--repeats", type=int, default=2, help="trace requests (the first is cold)")
    r.add_argument("--position", type=int)
    r.add_argument("--positions", help="a compare output; its position is used")
    r.add_argument("--warm-first", action="store_true", help="a 1-token request first (cold server)")
    r.add_argument("--top", type=int, default=8)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--output", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--reference", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--repeat", type=int, default=1, help="trace row (1 = first warm request)")
    c.add_argument("--output", required=True)
    k = sub.add_parser("classify")
    k.add_argument("--reference", required=True)
    k.add_argument("--candidate", required=True)
    k.add_argument("--reference-trace", required=True)
    k.add_argument("--candidate-trace", required=True)
    k.add_argument("--divergence", required=True)
    k.add_argument("--output", required=True)
    w = sub.add_parser("widths")
    w.add_argument("--ordinary", required=True)
    w.add_argument("--b1", required=True)
    w.add_argument("--b4", required=True)
    w.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    {"run": run, "compare": compare, "classify": classify, "widths": widths}[args.command](args)


if __name__ == "__main__":
    main()
