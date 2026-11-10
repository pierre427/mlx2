#!/usr/bin/env python3
"""Gemma 4 tensor backend A/B: native mlx2 port vs pinned mlx-vlm.

One ``--ordinary`` loopback server per arm (``MLX2_GEMMA4_BACKEND``), arms
counterbalanced (default mlx_vlm,native,native,mlx_vlm), B=1, no thermal
holds.  The first arm of each backend runs the correctness cells:

- ordinary-reference smoke (short factual chat);
- greedy parity prompts (short, ~4K, ~16K tokens of repository docs) with
  per-token logprobs (top 5) so the first divergent token, if any, can be
  classified by the AGENTS.md near-tie rules from that position's window;
- one image request.

Every arm runs the perf cells: cold TTFT and decode tok/s per prompt size,
APCv2 cleared before every request.  Each arm records the server's profile
receipt (``gemma4-native-apcv2-ordinary`` vs ``mlx-vlm-apcv2-ordinary``) so
the backend under test is checked, not assumed.

Usage (under the shared GPU queue):
  PYTHONPATH=$PWD/src python scripts/research/gemma4_native_ab.py \
      --model ~/mlx-models/<artifact> --out qualification/runs/<dir>
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import secrets
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_rows_attention_ab as h  # noqa: E402

CHARS_PER_TOKEN = 3.6
SMOKE = "In one sentence: what is the capital of France, and on which river does it stand?"
SHORT = (
    "Explain, in three short numbered steps, how a binary search finds a value "
    "in a sorted list of integers."
)
IMAGE_QUESTION = "Describe this image: list each shape you see with its colour and position."


def _image_data_url():
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (448, 336), (245, 245, 240))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 40, 170, 160), fill=(200, 30, 30))
    draw.ellipse((250, 50, 400, 200), fill=(30, 80, 200))
    draw.polygon([(120, 300), (210, 190), (300, 300)], fill=(40, 160, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _chat(base, content, max_tokens, *, logprobs=False):
    body = {
        "model": h.MODEL_ID[0],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "speculation": "off",
    }
    if logprobs:
        body.update(logprobs=True, top_logprobs=5)
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    sent = time.perf_counter()
    with urllib.request.urlopen(request, timeout=3600) as response:
        payload = json.loads(response.read())
    payload["_wall_s"] = round(time.perf_counter() - sent, 3)
    return payload


def _stream(base, prompt, max_tokens):
    body = {
        "model": h.MODEL_ID[0],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "enable_thinking": False,
        "speculation": "off",
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    record = {}
    sent = time.perf_counter()
    stamps, text = [], []
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                record["usage"] = chunk["usage"]
            for choice in chunk.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    stamps.append(time.perf_counter())
                    text.append(piece)
    record["ttft_s"] = round(stamps[0] - sent, 4) if stamps else None
    record["decode_tok_s"] = (
        round((len(stamps) - 1) / (stamps[-1] - stamps[0]), 2) if len(stamps) > 2 else None
    )
    record["chunks"] = len(stamps)
    record["text"] = "".join(text)
    return record


def _serve(model, backend, port, token_file, workdir, max_context):
    log = open(workdir / f"server-{port}.log", "w")
    cmd = [
        sys.executable, "-m", "mlx2.server", "--ordinary", "--model", str(model),
        "--host", "127.0.0.1", "--port", str(port), "--max-lanes", "1",
        "--max-inflight", "1", "--max-context", str(max_context),
        "--admin-token-file", str(token_file), "--cache-dir", str(workdir / f"cache-{port}"),
    ]
    env = dict(os.environ, MLX2_GEMMA4_BACKEND=backend)
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 1200
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited rc={proc.returncode}; see {log.name}")
            try:
                if h._get(base, "/health"):
                    h.MODEL_ID[0] = h._get(base, "/v1/models")["data"][0]["id"]
                    return proc, base
            except OSError:
                time.sleep(2)
        raise RuntimeError("server did not become healthy")
    except BaseException:
        h._stop(proc)
        raise


def _tokens(payload):
    content = (((payload.get("choices") or [{}])[0].get("logprobs")) or {}).get("content") or []
    return content


def _first_divergence(a, b):
    ta, tb = _tokens(a), _tokens(b)
    for index, (x, y) in enumerate(zip(ta, tb)):
        if x.get("token") != y.get("token"):
            return index, x, y
    if len(ta) != len(tb):
        return min(len(ta), len(tb)), None, None
    return None


def _classify(divergence):
    """AGENTS.md near-tie contract from the first divergent position only."""
    if divergence is None:
        return {"verdict": "exact_token_parity"}
    index, x, y = divergence
    if x is None or y is None:
        return {"verdict": "length_mismatch", "index": index}

    def window(entry):
        return {t["token"]: t["logprob"] for t in entry.get("top_logprobs") or []}

    wa, wb = window(x), window(y)
    sel_a, sel_b = x["token"], y["token"]
    top_a, top_b = max(wa.values()), max(wb.values())
    out = {"index": index, "selected": [sel_a, sel_b], "window_a": wa, "window_b": wb}
    if sel_b not in wa or sel_a not in wb:
        out["verdict"] = "unclassified_missing_token"
        return out
    margin_a = wa[sel_a] - wa[sel_b]
    margin_b = wb[sel_b] - wb[sel_a]
    out["margins"] = [round(margin_a, 4), round(margin_b, 4)]
    two_a = set(sorted(wa, key=wa.get, reverse=True)[:2])
    two_b = set(sorted(wb, key=wb.get, reverse=True)[:2])
    within = top_a - wa[sel_b] <= 0.5 and top_b - wb[sel_a] <= 0.5
    if (two_a == two_b and margin_a <= 0.5 and margin_b <= 0.5) or within:
        out["verdict"] = "near_tie_candidate"  # functional oracle checked by hand
    else:
        out["verdict"] = "high_margin_divergence"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default="mlx_vlm,native,native,mlx_vlm")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--perf-tokens", default="4096,16384")
    parser.add_argument("--parity-tokens", default="4096,16384")
    parser.add_argument("--parity-max-tokens", type=int, default=64)
    parser.add_argument("--perf-max-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=20480)
    parser.add_argument("--port", type=int, default=8471)
    parser.add_argument("--abort-swapouts", type=int, default=200000)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty evidence dir {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="gemma4-native-ab-"))
    token = secrets.token_hex(16)
    token_file = workdir / "admin.token"
    token_file.write_text(token + "\n")
    token_file.chmod(0o600)
    docs = sorted((h.ROOT / "docs").rglob("*.md"))

    def doc_prompt(target, salt):
        start = (salt * 7919) % len(docs)
        return h._document(docs[start:] + docs[:start], int(target * CHARS_PER_TOKEN)) + h.QUESTION

    parity = {"short": SHORT}
    for i, t in enumerate(int(x) for x in args.parity_tokens.split(",")):
        parity[f"doc{t}"] = doc_prompt(t, i + 1)
    perf = {int(t): doc_prompt(int(t), 100 + i) for i, t in enumerate(args.perf_tokens.split(","))}
    image = [{"type": "text", "text": IMAGE_QUESTION},
             {"type": "image_url", "image_url": {"url": _image_data_url()}}]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=h.ROOT, capture_output=True,
                              text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=h.ROOT,
                           capture_output=True, text=True).stdout.strip()
    results = {"model": str(args.model), "source_revision": revision, "dirty": bool(dirty),
               "arms_order": args.arms, "route": "--ordinary", "arms": [],
               "swapouts_start": h._swapouts(), "started": time.strftime("%FT%T%z")}
    seen = set()
    try:
        for index, backend in enumerate(args.arms.split(",")):
            proc, base = _serve(args.model, backend, args.port + index, token_file, workdir,
                                args.max_context)
            arm = {"backend": backend, "cells": {}}
            swap0 = h._swapouts()
            arm["swapouts_at_ready"] = swap0
            try:
                status = h._get(base, "/v1/status", token)
                # The serving profile name carries the backend (adapter
                # profile_name): gemma4-native-* for native, mlx-vlm-* else.
                text = json.dumps(status)
                profiles = {name for name in ("gemma4-native-apcv2-ordinary",
                                              "mlx-vlm-apcv2-ordinary") if name in text}
                arm["profile_receipt"] = sorted(profiles)
                wanted = ("gemma4-native-apcv2-ordinary" if backend == "native"
                          else "mlx-vlm-apcv2-ordinary")
                if profiles != {wanted}:
                    raise RuntimeError(f"server profile {sorted(profiles)} is not {wanted}")
                if backend not in seen:
                    seen.add(backend)
                    h._post(base, "/v1/admin/apc/clear", {"reason": "gemma4-native-ab"}, token)
                    arm["smoke"] = _chat(base, SMOKE, 48)
                    arm["parity"] = {}
                    for name, prompt in parity.items():
                        h._post(base, "/v1/admin/apc/clear", {"reason": "gemma4-native-ab"}, token)
                        arm["parity"][name] = _chat(base, prompt, args.parity_max_tokens,
                                                    logprobs=True)
                    h._post(base, "/v1/admin/apc/clear", {"reason": "gemma4-native-ab"}, token)
                    arm["image"] = _chat(base, image, 96)
                for target, prompt in perf.items():
                    cell = []
                    for _ in range(args.reps):
                        h._post(base, "/v1/admin/apc/clear", {"reason": "gemma4-native-ab"}, token)
                        cell.append(_stream(base, prompt, args.perf_max_tokens))
                        rise = h._swapouts() - swap0
                        if rise > args.abort_swapouts:
                            raise RuntimeError(f"swapouts rose by {rise}; aborting")
                    arm["cells"][str(target)] = cell
                arm["swapouts_end"] = h._swapouts()
                arm["final_status"] = h._get(base, "/v1/status", token)
            finally:
                h._stop(proc)
            results["arms"].append(arm)
            print(json.dumps({"backend": backend, "ttft": {
                c: [r["ttft_s"] for r in cell] for c, cell in arm["cells"].items()}}), flush=True)
    finally:
        results["swapouts_end"] = h._swapouts()
        results["finished"] = time.strftime("%FT%T%z")
        (args.out / "results.json").write_text(json.dumps(results, indent=1))

    first = {}
    for arm in results["arms"]:
        first.setdefault(arm["backend"], arm)
    summary = {"model": results["model"], "source_revision": revision, "parity": {}, "perf": {}}
    if {"mlx_vlm", "native"} <= first.keys():
        ref, nat = first["mlx_vlm"], first["native"]
        for name in parity:
            a, b = ref["parity"][name], nat["parity"][name]
            summary["parity"][name] = {
                "prompt_tokens": (a.get("usage") or {}).get("prompt_tokens"),
                "texts_equal": a["choices"][0]["message"]["content"]
                == b["choices"][0]["message"]["content"],
                **_classify(_first_divergence(a, b)),
            }
        summary["parity"]["smoke_equal"] = (ref["smoke"]["choices"][0]["message"]["content"]
                                            == nat["smoke"]["choices"][0]["message"]["content"])
        summary["parity"]["image_equal"] = (ref["image"]["choices"][0]["message"]["content"]
                                            == nat["image"]["choices"][0]["message"]["content"])
        summary["image_text"] = {k: first[k]["image"]["choices"][0]["message"]["content"]
                                 for k in first}
    for target in perf:
        key = str(target)
        by = {}
        for arm in results["arms"]:
            by.setdefault(arm["backend"], []).extend(arm["cells"][key])
        summary["perf"][key] = {
            backend: {
                "ttft_median_s": statistics.median(r["ttft_s"] for r in rows),
                "ttft_spread_s": [min(r["ttft_s"] for r in rows), max(r["ttft_s"] for r in rows)],
                "decode_median_tok_s": statistics.median(r["decode_tok_s"] for r in rows),
                "decode_spread_tok_s": [min(r["decode_tok_s"] for r in rows),
                                        max(r["decode_tok_s"] for r in rows)],
                "prompt_tokens": (rows[0].get("usage") or {}).get("prompt_tokens"),
                "n": len(rows),
            }
            for backend, rows in by.items()
        }
        summary["perf"][key]["distinct_texts"] = len(
            {r["text"] for rows in by.values() for r in rows})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
