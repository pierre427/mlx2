#!/usr/bin/env python3
"""A/B the per-row decode attention backend on a real 2-lane serve (omlx#5090).

One server process per arm (MLX_LM_BATCH_ATTENTION_BACKEND = sdpa | rows |
rows_masked), counterbalanced.  Each rep streams a long and a short request
concurrently (the short one is sent once the long one is decoding), so the two
lanes decode in one left-padded batch; the measured window is the overlap in
which both are decoding.  The sdpa arm also runs each prompt alone (B=1
ordinary reference).  Prompts are distinct repository documents, never a
repeated string.  Writes JSON evidence to --out (refuses a nonempty dir).

Usage (under the shared GPU lock):
  PYTHONPATH=$PWD/src python scripts/research/batch_rows_attention_ab.py \
      --model ~/mlx-models/<dense model> --out qualification/runs/<dir>
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = ["m"]  # set from /v1/models once a server is up
# Both probed options live on the ordinary route; a speculative default route
# (prompt lookup) owns its own prefill and verify-row decode.
QUESTION = "\n\nWrite a careful, detailed summary of the document above, section by section."


def _find(obj, key):
    """First value under ``key`` anywhere in a nested status document."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        values = obj.values()
    elif isinstance(obj, list):
        values = obj
    else:
        return None
    for value in values:
        found = _find(value, key)
        if found is not None:
            return found
    return None


def _swapouts() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return int([line for line in out.splitlines() if "Swapouts" in line][0].split()[-1].rstrip("."))


def _document(paths, chars):
    text, used = [], 0
    for path in paths:
        body = path.read_text(errors="replace")
        text.append(f"# {path.relative_to(ROOT)}\n\n{body}")
        used += len(body)
        if used >= chars:
            break
    return "\n\n".join(text)[:chars]


def _prompts(long_chars, short_chars):
    docs = sorted((ROOT / "docs").rglob("*.md"))
    half = len(docs) // 2
    return _document(docs[:half], long_chars) + QUESTION, _document(docs[half:], short_chars) + QUESTION


def _get(base, path, token=None):
    request = urllib.request.Request(base + path, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def _post(base, path, body, token):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def _stream(base, prompt, max_tokens, record):
    body = {"model": MODEL_ID[0], "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "enable_thinking": False,
            "stream_options": {"include_usage": True}, "speculation": "off"}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
    record["sent"] = time.perf_counter()
    stamps, text = [], []
    with urllib.request.urlopen(request, timeout=900) as response:
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
    record["stamps"] = stamps
    record["text"] = "".join(text)


def _overlap(long_rec, short_rec):
    """Tokens/s of both lanes while both are past their first token."""
    ls, ss = long_rec["stamps"], short_rec["stamps"]
    if len(ls) < 2 or len(ss) < 2:
        return None
    start, stop = max(ls[0], ss[0]), min(ls[-1], ss[-1])
    if stop - start <= 0.5:
        return None
    n = sum(start < t <= stop for t in ls) + sum(start < t <= stop for t in ss)
    return {"window_s": round(stop - start, 3), "tokens": n, "tok_per_s": round(n / (stop - start), 2)}


def _run_pair(base, long_prompt, short_prompt, max_tokens):
    long_rec, short_rec = {}, {}
    first = threading.Event()

    def long_job():
        _stream_with_first(base, long_prompt, max_tokens, long_rec, first)

    t_long = threading.Thread(target=long_job)
    t_long.start()
    first.wait(900)
    t_short = threading.Thread(target=_stream, args=(base, short_prompt, max_tokens, short_rec))
    t_short.start()
    t_long.join()
    t_short.join()
    return long_rec, short_rec


def _stream_with_first(base, prompt, max_tokens, record, first):
    body = {"model": MODEL_ID[0], "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "enable_thinking": False,
            "stream_options": {"include_usage": True}, "speculation": "off"}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     method="POST", headers={"Content-Type": "application/json"})
    stamps, text = [], []
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
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
                        if len(stamps) == 4:  # past prefill and the first decode steps
                            first.set()
    finally:
        first.set()
        record["stamps"] = stamps
        record["text"] = "".join(text)


def _serve(model, backend, port, token_file, workdir, max_context):
    env = {**os.environ, "MLX_LM_BATCH_ATTENTION_BACKEND": backend}
    log = open(workdir / f"server-{backend}-{port}.log", "w")
    # --ordinary: the ordinary BatchGenerator (merged, left-padded BatchKVCache
    # lanes and checkpoint-row prompt chunks).  A per-request "speculation":
    # "off" alone still decodes inside the startup route's generator.
    cmd = [sys.executable, "-m", "mlx2.server", "--ordinary", "--model", str(model), "--host", "127.0.0.1",
           "--port", str(port), "--max-lanes", "2", "--max-inflight", "2",
           "--max-context", str(max_context), "--admin-token-file", str(token_file),
           "--cache-dir", str(workdir / f"cache-{backend}-{port}")]
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode}; see {log.name}")
        try:
            if _get(base, "/health"):
                MODEL_ID[0] = _get(base, "/v1/models")["data"][0]["id"]
                return proc, base
        except OSError:
            time.sleep(1)
    _stop(proc)
    raise RuntimeError("server did not become healthy")


def _stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default="sdpa,rows,rows_masked,rows,sdpa,rows_masked")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--long-chars", type=int, default=110_000)
    parser.add_argument("--short-chars", type=int, default=14_000)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--max-context", type=int, default=40960)
    parser.add_argument("--port", type=int, default=8391)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty evidence dir {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="rows-ab-"))
    token = secrets.token_hex(16)
    token_file = workdir / "admin.token"
    token_file.write_text(token + "\n")
    token_file.chmod(0o600)
    long_prompt, short_prompt = _prompts(args.long_chars, args.short_chars)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    results = {"model": str(args.model), "source_revision": revision, "src_dirty": dirty,
               "arms": [], "swapouts_start": _swapouts()}
    for index, backend in enumerate(args.arms.split(",")):
        proc, base = _serve(args.model, backend, args.port + index, token_file, workdir, args.max_context)
        arm = {"backend": backend, "reps": []}
        try:
            status = _get(base, "/v1/status")
            arm["checkpoint"] = _find(status, "checkpoint")
            arm["status_model"] = {k: status.get(k) for k in ("model", "model_path", "fingerprint",
                                                              "model_revision") if k in status}
            if backend == "sdpa" and not any(a["backend"] == "sdpa" for a in results["arms"]):
                refs = {}
                for name, prompt in (("long", long_prompt), ("short", short_prompt)):
                    _post(base, "/v1/admin/apc/clear", {"reason": "rows-ab"}, token)
                    record = {}
                    _stream(base, prompt, args.max_tokens, record)
                    refs[name] = {"text": record["text"], "usage": record.get("usage")}
                results["b1_reference"] = refs
            for _ in range(args.reps):
                _post(base, "/v1/admin/apc/clear", {"reason": "rows-ab"}, token)
                long_rec, short_rec = _run_pair(base, long_prompt, short_prompt, args.max_tokens)
                arm["reps"].append({
                    "overlap": _overlap(long_rec, short_rec),
                    "long": {"text": long_rec["text"], "usage": long_rec.get("usage"), "tokens": len(long_rec["stamps"])},
                    "short": {"text": short_rec["text"], "usage": short_rec.get("usage"), "tokens": len(short_rec["stamps"])},
                })
            final = _get(base, "/v1/status")
            arm["batch_rows_attention"] = _find(final, "batch_rows_attention")
            arm["checkpoint"] = arm.get("checkpoint") or _find(final, "checkpoint")
        finally:
            _stop(proc)
        rates = [r["overlap"]["tok_per_s"] for r in arm["reps"] if r["overlap"]]
        arm["overlap_tok_per_s_median"] = statistics.median(rates) if rates else None
        results["arms"].append(arm)
        print(json.dumps({"backend": backend, "median": arm["overlap_tok_per_s_median"],
                          "rates": rates, "receipt": arm["batch_rows_attention"]}), flush=True)
    results["swapouts_end"] = _swapouts()
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
