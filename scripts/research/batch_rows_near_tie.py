#!/usr/bin/env python3
"""Near-tie classification for the per-row decode attention backend (omlx#5090).

For each arm (MLX_LM_BATCH_ATTENTION_BACKEND = sdpa, rows; one fresh
--ordinary server each, max-lanes 2), with distinct documentation prompts that
each carry a planted fact and a question (functional oracle):

1. B1 ordinary reference: each prompt alone, streamed with top logprobs.
2. Paired pass: the long prompt streams first; once it has emitted 4 tokens
   the short prompt starts, so both lanes decode in one left-padded
   BatchKVCache (the A/B's overlap shape).  Both stream top logprobs.

Every paired output is classified against the same arm's B1 reference and
against the sdpa arm's paired output with the AGENTS.md first-divergence
near-tie contract (``kimi_linear_speculative.classify``).  APCv2 is cleared
before every request.  Writes results.json (refuses a nonempty dir).
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_rows_attention_ab as h  # noqa: E402
from kimi_linear_speculative import classify  # noqa: E402

ROOT = h.ROOT
SOURCES = ("docs/SERVING.md", "docs/METRICS.md", "docs/QUALIFICATION.md", "docs/PROVENANCE.md",
           "README.md", "docs/KERNEL-RELOAD.md")
FACTS = (("release codename", "ORCHID-7431"), ("backup site", "Kestrel Ridge"))


def prompts(long_chars, short_chars):
    text = "\n\n".join((ROOT / path).read_text() for path in SOURCES if (ROOT / path).is_file())
    cases, start = [], 0
    for name, (topic, value), size in zip(("long", "short"), FACTS, (long_chars, short_chars)):
        body = text[start:start + size]
        start += size
        cut = len(body) // 3
        body = body[:cut] + f"\n\nNote for this deployment: the {topic} is {value}.\n\n" + body[cut:]
        content = (f"Here is part of the project documentation.\n\n{body}\n\n"
                   f"Question: according to the note above, what is the {topic}? "
                   "Answer in one short sentence, then summarize what this part of the "
                   "documentation covers in about 150 words.")
        cases.append((name, content, value))
    return cases


def stream(base, content, max_tokens, top, record, started=None):
    body = {"model": h.MODEL_ID[0], "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True, "enable_thinking": False,
            "speculation": "off", "logprobs": True, "top_logprobs": top}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
    rows, text = [], []
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[5:])
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        text.append(piece)
                    for row in ((choice.get("logprobs") or {}).get("content") or ()):
                        rows.append({"token": row.get("token"), "logprob": row.get("logprob"),
                                     "top": {i["token"]: i["logprob"] for i in row.get("top_logprobs") or ()}})
                        if started is not None and len(rows) == 4:
                            started.set()
    finally:
        if started is not None:
            started.set()
        record["rows"] = rows
        record["text"] = "".join(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default="sdpa,rows")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--long-chars", type=int, default=110_000)
    parser.add_argument("--short-chars", type=int, default=16_000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--port", type=int, default=8431)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty evidence dir {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="rows-nt-"))
    token = secrets.token_hex(16)
    token_file = workdir / "admin.token"
    token_file.write_text(token + "\n")
    token_file.chmod(0o600)
    cases = prompts(args.long_chars, args.short_chars)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    results = {"model": str(args.model), "source_revision": revision, "arms": {},
               "swapouts_start": h._swapouts()}
    for index, backend in enumerate(args.arms.split(",")):
        proc, base = h._serve(args.model, backend, args.port + index, token_file, workdir, 40960)
        arm = {"b1": {}, "pairs": []}
        try:
            for name, content, value in cases:
                h._post(base, "/v1/admin/apc/clear", {"reason": "rows-near-tie"}, token)
                record = {}
                stream(base, content, args.max_tokens, args.top_logprobs, record)
                record["oracle"] = value in record["text"]
                arm["b1"][name] = record
            for _ in range(args.reps):
                h._post(base, "/v1/admin/apc/clear", {"reason": "rows-near-tie"}, token)
                long_rec, short_rec, started = {}, {}, threading.Event()
                (_, long_text, long_value), (_, short_text, short_value) = cases
                t_long = threading.Thread(target=stream, args=(base, long_text, args.max_tokens,
                                                               args.top_logprobs, long_rec, started))
                t_long.start()
                started.wait(900)
                t_short = threading.Thread(target=stream, args=(base, short_text, args.max_tokens,
                                                                args.top_logprobs, short_rec))
                t_short.start()
                t_long.join()
                t_short.join()
                long_rec["oracle"] = long_value in long_rec["text"]
                short_rec["oracle"] = short_value in short_rec["text"]
                arm["pairs"].append({"long": long_rec, "short": short_rec})
            arm["receipt"] = h._find(h._get(base, "/v1/status"), "batch_rows_attention")
        finally:
            h._stop(proc)
        results["arms"][backend] = arm
    # Classification.
    verdicts = []
    for backend, arm in results["arms"].items():
        for rep, pair in enumerate(arm["pairs"]):
            for lane in ("long", "short"):
                ref = arm["b1"][lane]
                verdicts.append({"arm": backend, "rep": rep, "lane": lane, "against": "own_b1",
                                 **classify(ref["rows"], pair[lane]["rows"], ref["oracle"], pair[lane]["oracle"])})
                if backend != "sdpa" and "sdpa" in results["arms"]:
                    base_pair = results["arms"]["sdpa"]["pairs"][min(rep, len(results["arms"]["sdpa"]["pairs"]) - 1)]
                    verdicts.append({"arm": backend, "rep": rep, "lane": lane, "against": "sdpa_pair",
                                     **classify(base_pair[lane]["rows"], pair[lane]["rows"],
                                                base_pair[lane]["oracle"], pair[lane]["oracle"])})
    results["verdicts"] = verdicts
    results["swapouts_end"] = h._swapouts()
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    for v in verdicts:
        print(json.dumps({k: v.get(k) for k in ("arm", "rep", "lane", "against", "verdict", "first_divergence",
                                                 "reference_selected", "candidate_selected", "reference_margin",
                                                 "candidate_margin", "same_top_two", "clause",
                                                 "both_pass_functional_oracle")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
