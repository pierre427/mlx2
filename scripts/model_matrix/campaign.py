#!/usr/bin/env python3
"""Drive the model matrix one artifact per GPU-lock window, smallest first.

M5: every artifact window runs through ``scripts/run_with_gpu_locks.py``.  The
driver waits (without holding anything) while another session owns the GPU,
so the lock is never held across idle time.  Progress lines are appended to
``PROGRESS.md`` in the evidence root.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCKS = (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock"))
WAITERS = Path("/Users/Shared/mlxuag/gpu.lock.waiters")


def lock_busy() -> str | None:
    for path in LOCKS:
        if path.is_dir():
            owner = path / "owner.json"
            try:
                data = json.loads(owner.read_text())
                return f"{data.get('session')}/{data.get('label')}"
            except (OSError, json.JSONDecodeError):
                return f"{path} held"
    return None


def progress(root: Path, line: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with (root / "PROGRESS.md").open("a") as stream:
        stream.write(f"- {stamp} {line}\n")
    print(f"[{stamp}] {line}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--session", default="claude-model-matrix-20261021")
    parser.add_argument("--out-root", type=Path, default=ROOT / "qualification/runs/model-matrix-20261021")
    parser.add_argument("--targets", help="comma-separated checkpoints/names; default all wirable targets")
    parser.add_argument("--routes", default="all")
    parser.add_argument("--max-weight-gb", type=float, default=200.0)
    parser.add_argument("--min-weight-gb", type=float, default=0.0)
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--window-timeout", type=float, default=5400)
    parser.add_argument("--yield-seconds", type=float, default=120.0,
                        help="M5: idle this long after each window before queueing again")
    parser.add_argument("--m3-queue", help="M3 two-lock queue script (gpuq2-tf.sh) instead of run_with_gpu_locks.py")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--wait-pid", type=int, help="start after this process exits, then yield once")
    args = parser.parse_args(argv)
    if args.wait_pid:
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(5)
        time.sleep(args.yield_seconds)
    inventory = json.loads(args.inventory.read_text())
    targets = [t for t in inventory["targets"] if t.get("status") == "target"]
    if args.targets:
        wanted = args.targets.split(",")
        targets = [t for t in targets if {t["checkpoint"], t["name"], Path(t["path"]).name} & set(wanted)]
    targets = [t for t in targets
               if args.min_weight_gb * 1e9 <= (t.get("weight_bytes") or 0) <= args.max_weight_gb * 1e9]
    if args.routes != "all":
        wanted = args.routes.split(",")
        exact = {w for w in wanted if not w.endswith("*")}
        prefixes = tuple(w[:-1] for w in wanted if w.endswith("*"))

        def has_route(target):
            return any(r["route"] in exact or (prefixes and r["route"].startswith(prefixes))
                       for r in target.get("routes", ()))

        # Never take a GPU window for a target with nothing to run.
        targets = [t for t in targets if has_route(t)]
    targets.sort(key=lambda t: t.get("weight_bytes") or 0)
    args.out_root.mkdir(parents=True, exist_ok=True)
    receipts = args.out_root / "lock-receipts"
    receipts.mkdir(exist_ok=True)
    for target in targets:
        label = "mm-" + "".join(c if c.isalnum() else "-" for c in target["name"])[:48] + args.suffix
        command = [
            sys.executable, str(ROOT / "scripts/model_matrix/run_matrix.py"),
            "--inventory", str(args.inventory), "--target", target["checkpoint"],
            "--routes", args.routes, "--out-root", str(args.out_root), "--port", str(args.port),
            *([f"--suffix={args.suffix}"] if args.suffix else []), *args.extra,
        ]
        attempt = 0
        while True:
            waited = 0.0
            while not args.m3_queue and (holder := lock_busy()) is not None:
                if waited == 0:
                    progress(args.out_root, f"{target['name']}: waiting for GPU held by {holder}")
                time.sleep(30)
                waited += 30
            receipt = receipts / f"{label}-{int(time.time())}.json"
            progress(args.out_root, f"{target['name']}: start window ({(target.get('weight_bytes') or 0) / 1e9:.1f} GB)")
            started = time.time()
            if args.m3_queue:
                # The M3 two-lock queue waits for the lock itself and holds it
                # only for the command's lifetime.
                env = {**os.environ, "GPUQ_LEASE": f"{args.session}-{label}"}
                completed = subprocess.run([args.m3_queue, label, *command], env=env,
                                           timeout=args.window_timeout + 600)
                lock_receipt = {"status": "m3_queue", "returncode": completed.returncode}
                receipt.write_text(json.dumps({**lock_receipt, "command": command}, indent=2) + "\n")
            else:
                completed = subprocess.run([
                    sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                    "--session", args.session, "--label", label, "--receipt", str(receipt), "--", *command,
                ], timeout=args.window_timeout + 600)
                try:
                    lock_receipt = json.loads(receipt.read_text())
                except (OSError, json.JSONDecodeError):
                    lock_receipt = {}
            error = lock_receipt.get("error") or ""
            if "already held" in error or "flock-held" in error or "foreign model process" in error:
                attempt += 1
                progress(args.out_root, f"{target['name']}: lock contention ({error[:160]}); retry {attempt}")
                time.sleep(60)
                if attempt < 120:
                    continue
            progress(args.out_root, f"{target['name']}: window done rc={completed.returncode} "
                                    f"in {time.time() - started:.0f}s status={lock_receipt.get('status')}")
            if not args.m3_queue and args.yield_seconds:
                # Let other sessions' retrying wrappers take the shared GPU
                # before this campaign queues its next artifact.
                time.sleep(args.yield_seconds)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
