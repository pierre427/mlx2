"""Watchdog for one bounded HySparse2 context cell; child owns GPU admission."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-revision", required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--decode-tokens", type=int, default=8)
    p.add_argument("--memory-limit-gib", type=float, default=20)
    p.add_argument("--timeout-seconds", type=int, default=3600)
    args = p.parse_args()
    status = args.output.with_name(args.output.stem + "-process.json")
    log_path = args.output.with_suffix(".log")
    if args.output.exists() or status.exists() or args.timeout_seconds < 1:
        p.error("use a fresh output and positive timeout")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "scripts/bench_hysparse2_context.py",
               "--checkpoint", args.checkpoint, "--tokens", args.tokens,
               "--output", str(args.output), "--source-revision", args.source_revision,
               "--lengths", str(args.length), "--decode-tokens", str(args.decode_tokens),
               "--memory-limit-gib", str(args.memory_limit_gib)]
    with log_path.open("x") as log:
        child = subprocess.Popen(command, env=dict(os.environ, PYTHONPATH="src"),
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"pid": child.pid, "launcher_pid": os.getpid(), "command": command,
                  "started_unix": time.time(), "timeout_seconds": args.timeout_seconds,
                  "status": "running"}

        def save():
            tmp = status.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=2) + "\n")
            tmp.replace(status)

        def interrupted(signum, _frame):
            raise InterruptedError(f"launcher received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            save()
            code = child.wait(timeout=args.timeout_seconds)
            record.update(status="completed" if code == 0 else "failed", exit_code=code)
        except (subprocess.TimeoutExpired, InterruptedError) as exc:
            record.update(status="timed_out" if isinstance(exc, subprocess.TimeoutExpired)
                          else "interrupted", reason=str(exc))
        finally:
            # TERM/timeout/other exceptions must not leave a child owning the GPU.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            record.update(exit_code=child.returncode, finished_unix=time.time())
            save()


if __name__ == "__main__":
    main()
