#!/usr/bin/env python3
"""Super VL NVFP4 prefill sweep: NAX gather tile plans x prefill chunk sizes.

``owned OUT`` takes a desktop CPG lease and the paired GPU locks and runs
``unit OUT`` under them; ``unit`` runs one child process per NAX plan (the
plan pin ``MLX2_MOE_NAX_GATHER_PLAN`` is read at import), each loading the
served adapter once and prefilling non-repeating prompts of each length in
chunks of each size, the serving shape: hidden-only trunk chunks, the
final chunk through the checkpoint logits.  Per (plan, chunk, length):
prefill tok/s (median of reps, fresh cache each) and the sha256 of the final
logits, so bit-identity against the default (auto plan, 2048 chunks) and
against stock (NAX off) is visible.  Attribution only, no thermal hold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nemotron_tree_check as check  # noqa: E402

PLANS = {
    "auto": None,
    "db64": "db,64,64,32,0",
    "db96": "db,96,64,32,0",
    "db96_plain": "db,96,64,0,0",
    "seg96_bk128": "seg,96,128,32,0",
    "seg128_bk128": "seg,128,128,32,8192",
    "seg128_nopad": "seg,128,128,32,0",
    "nax_off": "off",
}


def child(args):
    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models import moe_nax_gather as nax

    policy = {} if args.plan != "nax_off" else {"moe_nax_gather": "off"}
    adapter = resolve_adapter(check.TARGET)(str(check.TARGET),
                                            execution_policy=policy or None)
    model = adapter.model
    rows = []
    for length in args.lengths:
        ids = check._prompt(adapter.tokenizer, length)
        for chunk in args.chunks:
            def prefill():
                cache = model.make_cache()
                for start in range(0, length - chunk, chunk):
                    hidden = model.backbone(mx.array([ids[start:start + chunk]]), cache=cache)
                    mx.eval(hidden, [entry.state for entry in cache])
                start = ((length - 1) // chunk) * chunk
                logits = model._checkpoint_logits(mx.array([ids[start:]]), cache)
                mx.eval(logits)
                return logits

            logits = prefill()  # warm (compiles, canaries)
            times = []
            for _ in range(args.reps):
                mx.synchronize()
                started = time.perf_counter()
                logits = prefill()
                mx.synchronize()
                times.append(time.perf_counter() - started)
            last = logits[0, -1].astype(mx.float32)
            digest = hashlib.sha256(np.asarray(last).tobytes()).hexdigest()[:16]
            row = {"plan": args.plan, "length": length, "chunk": chunk,
                   "seconds": round(statistics.median(times), 3),
                   "tok_s": round(length / statistics.median(times), 1),
                   "logits_sha": digest, "argmax": int(mx.argmax(last).item())}
            rows.append(row)
            print(json.dumps(row), flush=True)
    status = nax.status() if hasattr(nax, "status") else {}
    print(json.dumps({"plan": args.plan, "nax_status": {k: status.get(k) for k in
                                                         ("mode", "calls", "fallbacks")}},
                     default=str), flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1) + "\n")
    adapter.close()
    return 0


def unit(out, args):
    report = {"schema": "mlx2.nemotron-prefill-sweep.v1", "checkpoint": check.checkpoint(),
              "source": check.source(), "qualified": False,
              "evidence_kind": "attribution sweep, no thermal hold", "plans": {}, "rows": []}
    for name, pin in PLANS.items():
        if args.plans and name not in args.plans:
            continue
        env = {k: v for k, v in os.environ.items() if k != "MLX2_MOE_NAX_GATHER_PLAN"}
        if pin not in (None, "off"):
            env["MLX2_MOE_NAX_GATHER_PLAN"] = pin
        env["PYTHONPATH"] = str(ROOT / "src")
        target = out / f"{name}.json"
        command = [sys.executable, "-u", str(Path(__file__)), "child", str(target),
                   "--plan", name, "--lengths", *map(str, args.lengths),
                   "--chunks", *map(str, args.chunks), "--reps", str(args.reps)]
        with (out / f"{name}.log").open("w") as log:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        report["plans"][name] = {"pin": pin, "returncode": code}
        if target.exists():
            report["rows"].extend(json.loads(target.read_text()))
        check.atomic(out / "report.json", report)
        print(json.dumps({"plan": name, "returncode": code}), flush=True)
    return 0


def owned(out, args):
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    job_id = "nemotron-prefill-sweep-" + out.name
    with DesktopCPGLease(out, job_id, checkpoint=check.checkpoint(),
                         source_revision=check.source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out),
                   "--lengths", *map(str, args.lengths), "--chunks", *map(str, args.chunks),
                   "--reps", str(args.reps), *(["--plans", *args.plans] if args.plans else [])]
        child_process = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        try:
            while child_process.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child_process.returncode == 0
            return child_process.returncode
        finally:
            if child_process.poll() is None:
                os.killpg(child_process.pid, signal.SIGTERM)
                child_process.wait(timeout=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("owned", "unit", "child"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--plan", default="auto")
    parser.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--chunks", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--plans", nargs="*", default=[])
    args = parser.parse_args()
    if args.mode == "child":
        return child(args)
    if args.mode == "unit":
        args.out.mkdir(parents=True, exist_ok=True)
    return {"owned": owned, "unit": unit}[args.mode](args.out, args)


if __name__ == "__main__":
    raise SystemExit(main())
