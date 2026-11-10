"""One-artifact DS4.1 Source smoke: ordinary-reference serving with original_lfu experts.

Starts one loopback server in its own process group, sends a few fixed
greedy requests, writes receipts atomically, and always stops the group.
Run under the shared GPU queue. Refuses a nonempty evidence directory.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    ("count", "Count from 1 to 12, separated by commas. Output only the numbers.", lambda t: "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12" in t.replace(",  ", ", ")),
    ("capital", "What is the capital of Australia? Answer with one word.", lambda t: "canberra" in t.lower()),
    ("gsm8k", "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
              "How many clips did Natalia sell altogether in April and May? End with 'Answer: <number>'.",
     lambda t: "72" in t.split("Answer")[-1] if "Answer" in t else "72" in t[-40:]),
]


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), dict(r.headers)


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8291)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--repeat", type=int, default=1, help="run the prompt list N times (1st cold, later warm)")
    ap.add_argument("--max-context", type=int, default=16384)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        sys.exit("refusing nonempty evidence directory " + str(a.out))
    a.out.mkdir(parents=True, exist_ok=True)
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    seal = Path(a.model, "mlx2-source-integrity.json")
    cmd = [sys.executable, "-m", "mlx2.server", "--model", a.model, "--port", str(a.port),
           "--execution-policy", str(a.policy), "--max-context", str(a.max_context), "--ordinary", *a.extra]
    source = {"mlx2_commit": rev, "tracked_dirty": bool(dirty), "command": cmd,
              "policy": json.loads(a.policy.read_text()), "seal_present": seal.exists(),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    write(a.out / "source.json", source)
    log = open(a.out / "server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")})
    results, base = [], f"http://127.0.0.1:{a.port}"
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode} during load")
            try:
                if get(base + "/health", 5):
                    break
            except Exception:
                time.sleep(5)
            if time.time() - t0 > 3600:
                raise RuntimeError("server load timeout")
        load_s = time.time() - t0
        status = get(base + "/v1/status", 30)
        write(a.out / "status-before.json", status)
        model = get(base + "/v1/models")["data"][0]["id"]
        for name, prompt, oracle in [(f"{n}#{r}", p_, o) for r in range(a.repeat) for n, p_, o in PROMPTS]:
            t = time.time()
            body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": a.max_tokens, "temperature": 0}
            resp, headers = post(base + "/v1/chat/completions", body)
            msg = resp["choices"][0]["message"]
            text = msg.get("content") or ""
            m = resp.get("mlx2") or {}
            dec = (m.get("completion_tokens", 0) - 1) / max(1e-9, m.get("elapsed_seconds", 0) - m.get("ttft_seconds", 0))
            results.append({"name": name, "elapsed_s": round(time.time() - t, 3), "pass": bool(oracle(text)),
                            "ttft_s": m.get("ttft_seconds"), "decode_tok_s": round(dec, 3),
                            "content": text, "reasoning": msg.get("reasoning_content"),
                            "finish_reason": resp["choices"][0].get("finish_reason"),
                            "usage": resp.get("usage"), "headers": {k: v for k, v in headers.items() if "mlx2" in k.lower() or "route" in k.lower()},
                            "receipt": {k: v for k, v in resp.items() if k not in ("choices",)}})
            write(a.out / "results.json", {"load_s": load_s, "results": results})
        write(a.out / "status-after.json", get(base + "/v1/status", 30))
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(60)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        source["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write(a.out / "source.json", source)
    print(json.dumps([{k: r[k] for k in ("name", "pass", "elapsed_s", "ttft_s", "decode_tok_s", "finish_reason")} for r in results]))


if __name__ == "__main__":
    main()
