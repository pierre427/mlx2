"""GPU smoke: does ``--int8-prefill auto`` bind on a real server, and serve?

Starts ``python -m mlx2.server`` on one checkpoint in qualification mode,
saves /v1/status before and after, sends one long (>= 2K-token) prompt and one
short prompt (greedy), and records whether int8 prefill resolved on, its
census, whether ``q8_calls`` grew on the long prompt, lane matmul coverage, and
the generated text.  Stops the server it started.  Must run under the GPU lock.
"""

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
        return json.loads(r.read())


def chat(port, content, max_tokens):
    body = json.dumps({"messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    out["_wall_s"] = time.perf_counter() - start
    return out


def swapouts():
    for line in subprocess.run(["vm_stat"], capture_output=True, text=True).stdout.splitlines():
        if line.startswith("Swapouts"):
            return int(line.split()[-1].rstrip("."))
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--extra", default="", help="extra server arguments, one shell-quoted string")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    cmd = [sys.executable, "-m", "mlx2.server", "--model", a.model, "--qualification-mode",
           "--port", str(port), "--max-context", "16384", "--max-lanes", "4",
           "--cache-bytes", "2147483648", *shlex.split(a.extra)]
    s0 = swapouts()
    log = open(out / "server.log", "w")
    # Pin the child to this checkout's src, ahead of any inherited PYTHONPATH,
    # so a run queued from a worktree smokes that worktree's engine.
    source = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [source] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != source])
    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=a.repo, env=env)
    rec = {"model": a.model, "command": cmd[1:], "swapouts_start": s0}
    try:
        deadline = time.monotonic() + 900
        status = None
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise SystemExit(f"server exited rc={server.returncode}")
            try:
                status = get(port, "/v1/status")
                if status.get("healthy") or status.get("error"):
                    break
            except Exception:
                pass
            time.sleep(3)
        if not (status or {}).get("healthy"):
            raise SystemExit(f"server not healthy: {(status or {}).get('error')}")
        (out / "status-before.json").write_text(json.dumps(status, indent=1))
        doc = "\n\n".join(p.read_text() for p in sorted((Path(a.repo) / "docs").glob("*.md"))[:6])
        doc = doc[:14000]
        long_prompt = ("Read the following project documentation, then answer: what is "
                       "this project, and name three of its mechanisms, in three short "
                       "bullet points.\n\n" + doc)
        long = chat(port, long_prompt, 160)
        mid = get(port, "/v1/status")
        short = chat(port, "In two sentences, explain what a hash table is.", 96)
        after = get(port, "/v1/status")
        (out / "status-after.json").write_text(json.dumps(after, indent=1))

        def q8(st):
            return ((st.get("int8_prefill") or {}).get("counts") or {}).get("q8_calls", 0)

        i8 = after.get("int8_prefill") or {}
        lane = after.get("lane_matmul") or {}
        rec.update({
            "int8_auto": i8.get("auto"),
            "int8_active": i8.get("active"),
            "int8_modules": i8.get("modules"),
            "int8_module_kinds": i8.get("module_kinds"),
            "int8_skipped": i8.get("skipped"),
            "int8_revision": i8.get("revision"),
            "q8_calls": {"before": q8(status), "after_long": q8(mid), "after_short": q8(after)},
            "int8_counts_after": i8.get("counts"),
            "lane_matmul": {k: lane.get(k) for k in
                            ("requested", "mode", "installed", "backend", "covered",
                             "refused", "counts")},
            "long": {"prompt_tokens": long.get("usage", {}).get("prompt_tokens"),
                     "completion_tokens": long.get("usage", {}).get("completion_tokens"),
                     "wall_s": long["_wall_s"],
                     "text": long["choices"][0]["message"].get("content")},
            "short": {"prompt_tokens": short.get("usage", {}).get("prompt_tokens"),
                      "text": short["choices"][0]["message"].get("content")},
        })
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log.close()
        rec["swapouts_delta"] = swapouts() - s0
        (out / "smoke.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: rec.get(k) for k in ("int8_auto", "int8_modules", "q8_calls",
                                              "lane_matmul", "swapouts_delta")},
                     indent=1, default=str))
    print("LONG:", rec.get("long", {}).get("prompt_tokens"), rec.get("long", {}).get("text"))
    print("SHORT:", rec.get("short", {}).get("text"))


if __name__ == "__main__":
    sys.exit(main())
