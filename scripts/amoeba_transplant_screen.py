#!/usr/bin/env python3
"""Find one prompt the host cannot do and the donor can (amoeba transplant, step 0).

Run inside an owned GPU window. Starts one mlx2 server for --model on a non-production port
from this tree and answers every candidate in chat thinking mode at the model's default
reasoning effort: one greedy sample plus --samples samples at --temperature (seeded).
Candidates are procedural skills with an exact Python oracle, not bare facts: the answer
must be computed, not recalled. Each reply must end with a line "FINAL: <answer>"; the
oracle compares it after stripping whitespace, quotes and backticks (case-sensitive,
Unicode dashes folded to '-').

Writes rows.jsonl (one row per candidate and sample, with the full reply), run.json
(checkpoint fingerprint, mlx2 revision, server settings, per-candidate pass counts), and
stops the server's process group in `finally`. --only limits the run to candidate ids
(for the donor pass on the host's failures).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import signal
import struct
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _vigenere(text, key):
    out = []
    for i, ch in enumerate(text):
        out.append(chr((ord(ch) - 65 + ord(key[i % len(key)]) - 65) % 26 + 65))
    return "".join(out)


def _ieee(x):
    return struct.pack(">f", x).hex().upper()


def _lattice():
    # 6x6 grid of points (0..5), move right/up, blocked points
    blocked = {(1, 1), (2, 3), (3, 1), (4, 4), (1, 4)}
    n = 6
    dp = [[0] * n for _ in range(n)]
    for x in range(n):
        for y in range(n):
            if (x, y) in blocked:
                continue
            if x == 0 and y == 0:
                dp[x][y] = 1
                continue
            dp[x][y] = (dp[x - 1][y] if x else 0) + (dp[x][y - 1] if y else 0)
    return str(dp[n - 1][n - 1])


def _logic():
    people = ["Ana", "Ben", "Cyd", "Dov", "Eli"]
    sols = []
    for p in permutations(people):
        pos = {name: i for i, name in enumerate(p)}
        if not (pos["Ana"] < pos["Ben"]):
            continue
        if abs(pos["Cyd"] - pos["Dov"]) != 1:
            continue
        if pos["Eli"] in (0, 4):
            continue
        if pos["Ben"] != pos["Eli"] + 2:
            continue
        if pos["Dov"] == 0:
            continue
        if pos["Cyd"] > pos["Ana"]:
            continue
        sols.append(",".join(p))
    assert len(sols) == 1, sols
    return sols[0]


def _python_snippet():
    src = (
        "def f(x, acc=[]):\n"
        "    acc.append(x)\n"
        "    return len(acc)\n"
        "g = (f(i) * i for i in range(1, 5))\n"
        "a = f(10)\n"
        "b = sum(g)\n"
        "c = f(20)\n"
        "print(a, b, c)\n"
    )
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(src, {})
    return src, buf.getvalue().strip()


def _toposort():
    deps = {"B": ["A"], "C": ["A"], "D": ["B", "C"], "E": ["C"], "F": ["D", "E"], "G": ["F"], "H": ["E"]}
    nodes = sorted(set(deps) | {d for v in deps.values() for d in v})
    indeg = {n: 0 for n in nodes}
    for n, ds in deps.items():
        indeg[n] = len(ds)
    order, ready = [], sorted(n for n in nodes if indeg[n] == 0)
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m, ds in deps.items():
            if n in ds:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
                    ready.sort()
    return deps, ",".join(order)


def candidates():
    c = []
    s = "Amoeba boxes: 42!"
    c.append(("base64_encode", f"Encode the exact ASCII string `{s}` in standard Base64 (with padding).",
              base64.b64encode(s.encode()).decode()))
    t = "TRANSPLANTED"
    c.append(("base64_decode", f"Decode this standard Base64 string to ASCII: `{base64.b64encode(t.encode()).decode()}`", t))
    c.append(("vigenere", "Encrypt the plaintext TRANSPLANTBOXES with the Vigenere cipher, key QWEN "
              "(A=0, ..., Z=25; the key repeats; no spaces). Give the ciphertext in capitals.",
              _vigenere("TRANSPLANTBOXES", "QWEN")))
    c.append(("multiply", "Compute 48763 * 9127 exactly. Give the integer without separators.", str(48763 * 9127)))
    c.append(("ieee754", "Give the IEEE-754 single-precision bit pattern of -13.375 as 8 uppercase hexadecimal digits "
              "(big-endian, no 0x prefix).", _ieee(-13.375)))
    c.append(("modpow", "Compute 7^222 mod 1001. Give the integer.", str(pow(7, 222, 1001))))
    c.append(("lattice", "On the integer grid with points (x, y) for x, y in 0..5, count the monotone paths from (0,0) "
              "to (5,5) that move one step right (+x) or one step up (+y) at a time and never visit the blocked points "
              "(1,1), (2,3), (3,1), (4,4), (1,4). Give the count.", _lattice()))
    c.append(("logic", "Five people Ana, Ben, Cyd, Dov, Eli stand in a row, positions 1 to 5 from the left. "
              "Ana is somewhere left of Ben. Cyd and Dov are adjacent. Eli is not at either end. Ben is exactly two "
              "places right of Eli. Dov is not in position 1. Cyd is somewhere left of Ana. Give the order from left to "
              "right as comma-separated names with no spaces.", _logic()))
    src, out = _python_snippet()
    c.append(("python_eval", "What does this Python 3 program print? Give the printed line exactly.\n```python\n" + src + "```", out))
    deps, order = _toposort()
    desc = "; ".join(f"{k} needs {', '.join(v)}" for k, v in deps.items())
    c.append(("toposort", f"Tasks A to H have these prerequisites: {desc}. Give the topological order that always picks "
              "the alphabetically smallest available task next, as comma-separated letters with no spaces.", order))
    c.append(("reverse_words", "Reverse the letters of each word, keep the word order, and uppercase everything: "
              "`slow purple lanterns drift`. Give the result with single spaces.",
              " ".join(w[::-1] for w in "slow purple lanterns drift".split()).upper()))
    c.append(("decimal_17", "Give the first 16 digits after the decimal point of 1/17, as one digit string.",
              f"{(10**16) // 17:016d}"))
    return [{"id": i, "prompt": p + "\nEnd your reply with a line of the form FINAL: <answer>.", "answer": a} for i, p, a in c]


DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")


def extract_final(text):
    m = re.findall(r"FINAL:\s*(.+)", text or "")
    if not m:
        return None
    v = m[-1].strip().translate(DASHES)
    v = v.strip("`*\"' .").replace("\\boxed{", "").rstrip("}")
    return v.strip()


def post(base, path, payload, timeout=3600):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()[:2000]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8396)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--only", default="", help="comma-separated candidate ids")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--server-arg", action="append", default=[])
    ap.add_argument("--list", action="store_true", help="print candidates and oracle answers, then exit")
    ap.add_argument("--device", default="mps", help="accepted for the queue driver; the server owns Metal")
    args = ap.parse_args()
    cands = candidates()
    if args.only:
        keep = set(args.only.split(","))
        cands = [c for c in cands if c["id"] in keep]
    if args.list:
        for c in cands:
            print(json.dumps(c))
        return
    if args.port == 8282:
        sys.exit("8282 is reserved for release trees")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty dir {out}")
    out.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    cmd = [sys.executable, "-m", "mlx2.server", "--model", args.model, "--port", str(args.port), *args.server_arg]
    log = open(out / "server.log", "w")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rev = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    rows, status, thinking = [], {}, None
    try:
        end = time.time() + 1800
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:
                pass
            if time.time() > end:
                raise RuntimeError("server not healthy")
            time.sleep(3)
        with urllib.request.urlopen(base + "/v1/status", timeout=30) as r:
            status = json.loads(r.read())
        load_s = time.time() - t0
        model_name = status.get("model") or Path(args.model).name
        jobs = [(c, -1) for c in cands] + [(c, s) for c in cands for s in range(args.samples)]
        # A route without reasoning fails closed (400) on enable_thinking; probe once and
        # answer in plain chat on such routes, recorded as thinking=false in run.json.
        thinking = True
        probe = post(base, "/v1/chat/completions", {"model": model_name, "stream": False, "max_tokens": 1,
                     "enable_thinking": True, "temperature": 0, "messages": [{"role": "user", "content": "Hi"}]})
        if "error" in probe and "reasoning is not available" in json.dumps(probe):
            thinking = False
            print("route has no reasoning: screening in plain chat (thinking=false)", flush=True)

        def one(job):
            c, s = job
            payload = {"model": model_name, "stream": False, "max_tokens": args.max_tokens, "enable_thinking": thinking,
                       "temperature": 0 if s < 0 else args.temperature,
                       "messages": [{"role": "user", "content": c["prompt"]}]}
            if s >= 0:
                payload["seed"] = 1000 + s
            ts = time.time()
            res = post(base, "/v1/chat/completions", payload)
            lat = round(time.time() - ts, 1)
            if "error" in res:
                return {"id": c["id"], "sample": s, "error": res, "correct": False, "latency_s": lat}
            ch = res["choices"][0]
            text = ch["message"].get("content") or ""
            pred = extract_final(text)
            return {"id": c["id"], "sample": s, "temperature": payload["temperature"], "answer": c["answer"], "pred": pred,
                    "correct": pred == c["answer"], "finish_reason": ch.get("finish_reason"),
                    "completion_tokens": res.get("usage", {}).get("completion_tokens"), "latency_s": lat,
                    "content": text, "reasoning": ch["message"].get("reasoning_content") or "",
                    "route": res.get("mlx2_route") or res.get("route")}

        with ThreadPoolExecutor(args.concurrency) as ex:
            for r in ex.map(one, jobs):
                rows.append(r)
                print(f"{time.strftime('%H:%M:%S')} {r['id']} s{r['sample']} correct={r['correct']} "
                      f"pred={r.get('pred')!r} tok={r.get('completion_tokens')} {r.get('finish_reason')}", flush=True)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        log.close()
        with open(out / "rows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    from mlx2.adapters.registry import inspect_model
    fp = inspect_model(args.model).artifact_fingerprint
    per = {c["id"]: {"greedy": next((r["correct"] for r in rows if r["id"] == c["id"] and r["sample"] < 0), None),
                     "sampled": sum(r["correct"] for r in rows if r["id"] == c["id"] and r["sample"] >= 0)} for c in cands}
    rep = {"schema": "mlx2.amoeba-transplant-screen.v1", "model": args.model, "checkpoint": f"{Path(args.model).name}@{fp}",
           "mlx2_source_revision": rev, "server_cmd": cmd, "server_settings": status.get("settings"),
           "qualification": status.get("qualification"), "args": vars(args), "load_s": round(load_s, 1) if status else None,
           "per_candidate": per, "thinking": thinking, "wall_s": round(time.time() - t0, 1)}
    (out / "run.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(per))


if __name__ == "__main__":
    main()
