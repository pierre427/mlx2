#!/usr/bin/env python3
"""Run ``python -m MODULE ...`` with the amoeba memory watchdog in that process (and its SIGKILL watcher).

For harnesses that start a model server as a child (``spec_acceptance_gsm8k.py --python``): the
watchdog watches the process that holds the model, not the harness. Use through the executable
wrapper ``scripts/research/memwatched_python``. Limits come from the environment:
MEMWATCH_LIMIT_GB (footprint, default 60), MEMWATCH_SWAP_GROWTH_GB (default 2),
MEMWATCH_RECEIPT (JSON written when it stops the process). Only ``[-u] -m MODULE args...`` is
accepted.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv):
    args = list(argv)
    if args[:1] == ["-u"]:
        args = args[1:]
        os.environ["PYTHONUNBUFFERED"] = "1"
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    if len(args) < 2 or args[0] != "-m":
        raise SystemExit("memwatched_python: usage [-u] -m MODULE [args...]")
    module, rest = args[1], args[2:]
    sys.path.insert(0, str(ROOT))
    from research.amoeba.carve import ops as CO

    def log(msg):
        print(f"memwatch: {msg}", file=sys.stderr, flush=True)

    CO.footprint_watchdog(float(os.environ.get("MEMWATCH_LIMIT_GB", "60")), log,
                          swap_growth_gb=float(os.environ.get("MEMWATCH_SWAP_GROWTH_GB", "2")),
                          receipt=os.environ.get("MEMWATCH_RECEIPT") or None, watcher=True)
    sys.path.remove(str(ROOT))
    sys.argv = [module] + rest
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
