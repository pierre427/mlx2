#!/usr/bin/env python3
"""Localize the Raon output change between MLX 0.32.4 and 0.32.5 (probe doc, P0 side finding).

Same mlx2 commit, same prompts (the Phase 0 1K/16K prose prompts), greedy; two servers that
differ only in MLX (the two production venvs, read-only). Per AGENTS.md: localize the first
divergent output token from host token traces, then capture a bounded logprob window at that
already-selected position only, and classify it against the near-tie contract (0.5-nat ceiling).

    trace    --base-url URL --output FILE            token ids + text per prompt (no logprobs)
    compare  A.json B.json --output FILE             first divergent index per prompt
    window   --base-url URL --compare FILE --output FILE
                                                      top-8 logprobs at each first-divergence position
    classify --compare FILE --window-a FILE --window-b FILE --output FILE

The prose prompts have no functional oracle, so a row that meets every numerical near-tie
condition is reported ``near_tie_numerics_met`` (oracle n/a), never ``near_tie_equivalent``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_round_phase0 as phase0  # noqa: E402

SIZES = (1024, 16384)
CEILING = 0.5
TOP = 8


def call(base, path, body=None, timeout=1800):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method="POST" if body is not None else "GET")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def ask(base, model, text, extra):
    result = call(base, "/v1/chat/completions", {
        "model": model, "temperature": 0, "stream": False, "max_tokens": 256, "enable_thinking": False,
        "messages": [{"role": "user", "content": text}], **extra})
    choice = (result.get("choices") or [{}])[0]
    receipt = result.get("mlx2") or {}
    return result, choice, receipt


def trace(args):
    base = args.base_url
    model = call(base, "/v1/models")["data"][0]["id"]
    prompts = phase0.build_prompts(list(SIZES))
    status = call(base, "/v1/status")
    out = {"mlx": (status.get("runtime") or {}).get("mlx") or status.get("mlx_version"), "rows": {}}
    for size in SIZES:
        ask(base, model, prompts[size], {"max_tokens": 8})  # warm: cold prefill + cache entry
        _, choice, receipt = ask(base, model, prompts[size], {"return_token_trace": True})
        tr = receipt.get("token_trace") or {}
        out["rows"][str(size)] = {"token_ids": tr.get("token_ids"), "text": (choice.get("message") or {}).get("content"),
                                  "finish_reason": choice.get("finish_reason"), "route": receipt.get("route"),
                                  "cached_tokens": receipt.get("cached_tokens")}
    out["status_versions"] = {k: status.get(k) for k in ("version", "mlx_version", "mlx") if k in status}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")


def compare(args):
    a, b = json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text())
    rows = {}
    for size in a["rows"]:
        ta, tb = a["rows"][size]["token_ids"] or [], b["rows"][size]["token_ids"] or []
        first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if first is None and len(ta) != len(tb):
            first = min(len(ta), len(tb))
        rows[size] = {"identical": first is None, "first_index": first,
                      "position": None if first is None else first + 1,  # one-based completion token
                      "a_token": None if first is None or first >= len(ta) else ta[first],
                      "b_token": None if first is None or first >= len(tb) else tb[first],
                      "a_len": len(ta), "b_len": len(tb)}
    Path(args.output).write_text(json.dumps(rows, indent=1) + "\n")
    print(json.dumps(rows))


def window(args):
    base = args.base_url
    model = call(base, "/v1/models")["data"][0]["id"]
    prompts = phase0.build_prompts(list(SIZES))
    cmp = json.loads(Path(args.compare).read_text())
    out = {}
    for size, row in cmp.items():
        if row["position"] is None:
            continue
        pos = row["position"]
        ask(base, model, prompts[int(size)], {"max_tokens": 8})
        result, choice, receipt = ask(base, model, prompts[int(size)], {
            "return_token_trace": True, "logprobs": True, "top_logprobs": TOP,
            "logprobs_start": pos, "logprobs_end": pos, "max_tokens": pos + 2})
        content = ((choice.get("logprobs") or {}).get("content") or [])
        out[size] = {"position": pos, "window": content, "token_ids": (receipt.get("token_trace") or {}).get("token_ids"),
                     "route": receipt.get("route")}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")


def _entry(window_row):
    rows = window_row.get("window") or []
    if not rows:
        return None
    row = rows[-1]
    top = [(t.get("id"), t.get("logprob")) for t in (row.get("top_logprobs") or [])]
    return {"selected": row.get("id"), "selected_text": row.get("token"), "logprob": row.get("logprob"),
            "top": top, "top_text": [t.get("token") for t in (row.get("top_logprobs") or [])]}


def classify(args):
    cmp = json.loads(Path(args.compare).read_text())
    wa, wb = json.loads(Path(args.window_a).read_text()), json.loads(Path(args.window_b).read_text())
    out = {}
    for size, row in cmp.items():
        if row["identical"]:
            out[size] = {"verdict": "identical"}
            continue
        ea, eb = _entry(wa.get(size, {})), _entry(wb.get(size, {}))
        if ea is None or eb is None:
            out[size] = {"verdict": "unclassified", "reason": "missing window"}
            continue
        def margin(e):
            if len(e["top"]) < 2:
                return None
            return e["top"][0][1] - e["top"][1][1]
        def lp(e, token):
            return next((v for t, v in e["top"] if t == token), None)
        ma, mb = margin(ea), margin(eb)
        same_top2 = {t for t, _ in ea["top"][:2]} == {t for t, _ in eb["top"][:2]}
        # Multi-way clause: each arm's selection within 0.5 nat of the other arm's top.
        cross_a = lp(eb, ea["selected"]); cross_b = lp(ea, eb["selected"])
        multiway = (cross_a is not None and cross_b is not None
                    and eb["top"][0][1] - cross_a <= CEILING and ea["top"][0][1] - cross_b <= CEILING)
        twoway = same_top2 and ma is not None and mb is not None and ma <= CEILING and mb <= CEILING
        numerics = twoway or multiway
        out[size] = {"position": row["position"], "a": ea, "b": eb, "margin_a": ma, "margin_b": mb,
                     "same_top2": same_top2, "multiway_within_ceiling": multiway,
                     "verdict": "near_tie_numerics_met (no functional oracle for prose)" if numerics
                                else "high_margin_divergence"}
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v["verdict"] for k, v in out.items()}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trace"); t.add_argument("--base-url", required=True); t.add_argument("--output", required=True)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b"); c.add_argument("--output", required=True)
    w = sub.add_parser("window"); w.add_argument("--base-url", required=True); w.add_argument("--compare", required=True)
    w.add_argument("--output", required=True)
    k = sub.add_parser("classify"); k.add_argument("--compare", required=True); k.add_argument("--window-a", required=True)
    k.add_argument("--window-b", required=True); k.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    {"trace": trace, "compare": compare, "window": window, "classify": classify}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
