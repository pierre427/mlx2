"""Where a loaded Flash-Next server's memory goes (diagnostic, one server).

Starts one loopback mlx2 server on the Flash-Next artifact, and records at
three points (before launch, after ready, after one short chat request) the
host terms admission reads (psutil available, the Mach estimate that counts
file-backed and purgeable pages as free, wired/active/inactive/compressor),
the server's physical footprint, RSS, and a ``vmmap -summary`` of the server
process (region totals: MALLOC, IOKit / Metal, mapped files). The server's
admission log lines are copied. The process group is stopped in ``finally``.
Run only through the GPU lease.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def host_terms():
    import psutil
    from mlx2.runtime.os_memory import host_memory_snapshot

    g = 1 << 30
    s = host_memory_snapshot()
    vm = psutil.virtual_memory()
    out = {"psutil_available_gib": vm.available / g}
    if s is not None:
        p = s.page_size
        out.update({"mach_available_gib": s.available_bytes / g,
                    **{f"{k}_gib": getattr(s, k) * p / g for k in
                       ("free", "active", "inactive", "speculative", "wired", "compressor",
                        "file_backed", "purgeable")}})
    return {k: round(v, 2) for k, v in out.items()}


def process_terms(pid):
    import psutil
    from mlx2.runtime.os_memory import physical_footprint_bytes

    g = 1 << 30
    proc = psutil.Process(pid)
    out = {"footprint_gib": (physical_footprint_bytes(pid) or 0) / g, "rss_gib": proc.memory_info().rss / g}
    try:
        summary = subprocess.run(["vmmap", "-summary", str(pid)], capture_output=True, text=True, timeout=180).stdout
    except subprocess.TimeoutExpired:
        summary = "vmmap timed out"
    return {k: round(v, 2) for k, v in out.items()}, summary


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18321)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.root, text=True).strip()
    if head != args.source:
        raise RuntimeError(f"source HEAD {head} != {args.source}")
    if args.out.exists():
        raise RuntimeError("refusing an existing evidence directory")
    args.out.mkdir(parents=True)
    report = {"schema": "mlx2.flash-next-server-memory.v1", "source_revision": head,
              "checkpoint": f"{args.model.name}@sha256:{args.fingerprint}", "status": "running", "stages": {}}
    save = lambda: (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    report["stages"]["before_launch"] = {"host": host_terms()}
    save()
    log = (args.out / "server.log").open("w")
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(args.model), "--host", "127.0.0.1",
               "--port", str(args.port), "--max-inflight", "1", "--max-lanes", "1", "--max-context", "8192",
               "--cache-bytes", str(2 << 30), "--cache-dir", str(args.out / "cache")]
    report["command"] = command
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "PYTHONPATH": str(args.root / "src")})
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("server exited before readiness")
            try:
                get(f"http://127.0.0.1:{args.port}/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("server not ready in 600 s")
        time.sleep(5)
        terms, summary = process_terms(proc.pid)
        (args.out / "vmmap-after-ready.txt").write_text(summary)
        try:
            status = get(f"http://127.0.0.1:{args.port}/v1/status", timeout=10)
        except Exception as exc:
            status = {"error": repr(exc)}
        (args.out / "status-after-ready.json").write_text(json.dumps(status, indent=1) + "\n")
        report["stages"]["after_ready"] = {"host": host_terms(), "server": terms}
        save()
        model_id = get(f"http://127.0.0.1:{args.port}/v1/models", timeout=10)["data"][0]["id"]
        report["model_id"] = model_id
        long_prompt = " ".join(f"Item {i}: a short note about databases and indexes." for i in range(300))
        body = json.dumps({"model": model_id, "max_tokens": 32, "temperature": 0,
                           "messages": [{"role": "user", "content": long_prompt + " Summarize in one line."}]}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as r:
                report["request"] = {"http": r.status, "body": json.loads(r.read().decode())}
        except urllib.error.HTTPError as exc:
            report["request"] = {"http": exc.code, "body": exc.read().decode()[:2000]}
        terms, summary = process_terms(proc.pid)
        (args.out / "vmmap-after-request.txt").write_text(summary)
        report["stages"]["after_request"] = {"host": host_terms(), "server": terms}
        report["status"] = "passed"
    except BaseException as exc:
        report["status"], report["error"] = "failed", repr(exc)
        raise
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        log.close()
        report["stages"]["after_stop"] = {"host": host_terms()}
        save()


if __name__ == "__main__":
    main()
