#!/usr/bin/env python3
"""One qualification-ladder cell on the Super VL default route, this tree's code.

``owned OUT`` takes a desktop CPG lease and the paired GPU locks and runs
``unit OUT``: the qualification ladder producer
(``qualification/runs/qualify-1010-correctness/run_profile.py``) with the
default-route stage's server argv and options from ``--from`` (a stage
directory of qualify-super-vl-nvfp4-20261110), restricted to one context
length and a run-local cache, under the extra environment ``--env K=V``.
Then the producer's correctness verdict.  Reproduces the qualification's own
prompt content (acceptance there fell to 2.7 tokens/round at 32K) for an A/B
of a route switch.  Attribution only; not a qualification.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/research"))
import nemotron_tree_check as check  # noqa: E402

PRODUCERS = ROOT / "qualification/runs/qualify-1010-correctness"


def unit(out, args):
    argv = json.loads((args.source / "server-argv.json").read_text())
    argv[argv.index("--cache-dir") + 1] = str(out / "cache")
    options = json.loads((args.source / "options.json").read_text())
    options["min-length"] = options["max-length"] = args.length
    (out / "server-argv.json").write_text(json.dumps(argv, indent=1) + "\n")
    (out / "options.json").write_text(json.dumps(options, indent=1) + "\n")
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
                   **dict(item.split("=", 1) for item in args.env)}
    rc = subprocess.run([sys.executable, str(PRODUCERS / "run_profile.py"),
                         "--server-argv-json", str(out / "server-argv.json"),
                         "--ladder-options-json", str(out / "options.json"),
                         "--output", str(out / "ladder.json"),
                         "--server-log", str(out / "server.log"),
                         "--startup-timeout", "1200", "--job-timeout", "7200"],
                        cwd=ROOT, env=environment).returncode
    if (out / "ladder.json").is_file():
        with (out / "verdict.json").open("w") as handle:
            subprocess.run([sys.executable, str(PRODUCERS / "qualification_verdict.py"),
                            str(out / "ladder.json")], cwd=ROOT, env=environment,
                           stdout=handle, check=False)
    return rc


def owned(out, args):
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    job_id = "super-vl-ladder-cell-" + out.name
    with DesktopCPGLease(out, job_id, checkpoint=check.checkpoint(),
                         source_revision=check.source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out),
                   "--from", str(args.source), "--length", str(args.length),
                   *[f"--env={item}" for item in args.env]]
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                                 env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--from", dest="source", type=Path, required=True)
    parser.add_argument("--length", type=int, default=32768)
    parser.add_argument("--env", action="append", default=[])
    args = parser.parse_args()
    if args.mode == "unit":
        args.out.mkdir(parents=True, exist_ok=True)
    return {"owned": owned, "unit": unit}[args.mode](args.out, args)


if __name__ == "__main__":
    raise SystemExit(main())
