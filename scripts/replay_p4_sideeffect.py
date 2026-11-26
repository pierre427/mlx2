#!/usr/bin/env python3
"""Localize the P4 side effect (docs/research/2026-11-26-metal-replay-feasibility-probes.md,
"Open defect: the in-server P4 pass changes later outputs").

Drives one loopback server started with ``MLX2_ROUND_PROBE=1 MLX2_ROUND_PROBE_GPU=1
MLX2_ROUND_PROBE_RECORD=2 MLX2_ROUND_PROBE_RECORD_MAX=2 MLX2_ROUND_PROBE_RECORD_GAP_S=5``
(plus the P4 variant's ``MLX2_ROUND_PROBE_SUBSTITUTE*``), greedy, the Phase 0 prompts:

1. 16K: APCv2 clear, cold 8-token warm-up, then a warm 256-token reference (``ref16``);
2. 1K: the same (``ref1k``);
3. a pause longer than the recording gap, then the trigger: one warm 1K request whose rounds
   after the gap are recorded (and, in a P4 variant, substitution-checked);
4. 1K again (``post1k``), then 16K: clear, cold warm-up, warm (``post16``) and again
   (``post16b``).

Each request returns its token ids (``return_token_trace``, host metadata only). The
receipt reports the first index at which each post request differs from its reference.
Every recording and the substitution JSON land in ``MLX2_ROUND_PROBE_RECORD_DIR``.

    scripts/replay_p4_sideeffect.py --base-url URL --admin-token FILE --output DIR --variant NAME
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_round_phase0 as phase0  # noqa: E402


def call(base, path, body=None, token=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method="POST" if body is not None else "GET")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def ask(base, model, text, max_tokens):
    started = time.monotonic()
    result = call(base, "/v1/chat/completions", {
        "model": model, "temperature": 0, "stream": False, "max_tokens": max_tokens,
        "enable_thinking": False, "return_token_trace": True,
        "messages": [{"role": "user", "content": text}]})
    receipt = result.get("mlx2") or {}
    trace = receipt.get("token_trace") or {}
    usage = result.get("usage") or {}
    return {
        "elapsed_s": round(time.monotonic() - started, 3),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": (result.get("choices") or [{}])[0].get("finish_reason"),
        "text": ((result.get("choices") or [{}])[0].get("message") or {}).get("content"),
        "token_ids": trace.get("token_ids"),
        "truncated": trace.get("truncated"),
        "route": receipt.get("route"),
    }


def first_divergence(a, b):
    if a is None or b is None:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"index": i, "ref": x, "post": y, "shared_prefix": i}
    if len(a) != len(b):
        return {"index": min(len(a), len(b)), "ref_len": len(a), "post_len": len(b)}
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--admin-token", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--gap-s", type=float, default=7.0)
    args = ap.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    token = Path(args.admin_token).read_text().strip()
    model = call(args.base_url, "/v1/models")["data"][0]["id"]
    prompts = phase0.build_prompts([1024, 16384])
    log = {"variant": args.variant, "steps": []}

    def step(name, size, max_tokens, clear=False):
        if clear:
            call(args.base_url, "/v1/admin/apc/clear", {"reason": f"p4-sideeffect-{name}"}, token=token)
        row = ask(args.base_url, model, prompts[size], max_tokens)
        row.update(name=name, size=size)
        log["steps"].append(row)
        print(json.dumps({k: row[k] for k in ("name", "completion_tokens", "cached_tokens", "elapsed_s")}),
              flush=True)
        return row

    step("warm16", 16384, 8, clear=True)
    ref16 = step("ref16", 16384, 256)
    step("warm1k", 1024, 8, clear=True)
    ref1k = step("ref1k", 1024, 256)
    time.sleep(args.gap_s)
    step("trigger", 1024, 256)
    post1k = step("post1k", 1024, 256)
    step("warm16b", 16384, 8, clear=True)
    post16 = step("post16", 16384, 256)
    post16b = step("post16b", 16384, 256)
    log["divergence"] = {
        "post1k_vs_ref1k": first_divergence(ref1k["token_ids"], post1k["token_ids"]),
        "post16_vs_ref16": first_divergence(ref16["token_ids"], post16["token_ids"]),
        "post16b_vs_ref16": first_divergence(ref16["token_ids"], post16b["token_ids"]),
    }
    log["identical"] = {k: v is None for k, v in log["divergence"].items()}
    status = call(args.base_url, "/v1/status")
    log["gpu_timeline"] = (status.get("round_probe") or {}).get("gpu_timeline")
    (out / "sideeffect.json").write_text(json.dumps(log, indent=1) + "\n")
    print(json.dumps({"variant": args.variant, "identical": log["identical"], "divergence": log["divergence"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
