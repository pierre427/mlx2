#!/usr/bin/env python3
"""P5b (address patch cost) and P5c (Metal 4 placement sparse buffers) of
docs/research/2026-11-26-metal-replay-feasibility-probes.md. P5a is
scripts/replay_p5a_addresses.py over round recordings.

    scripts/replay_probe_p5.py build              # CPU: both native binaries
    scripts/replay_probe_p5.py run --output DIR   # GPU, under the lease

Synthetic (no model): ``checkpoint`` is ``none (synthetic)``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "replay_probe"
SOURCES = {"replay_probe": ROOT / "native/replay_probe/replay_probe.mm",
           "sparse_probe": ROOT / "native/replay_probe/sparse_probe.mm",
           "gpu_plan_probe": ROOT / "native/replay_probe/gpu_plan_probe.mm"}


def build():
    BUILD.mkdir(parents=True, exist_ok=True)
    for name, src in SOURCES.items():
        subprocess.run(["clang++", "-std=c++20", "-fobjc-arc", "-O2", "-arch", "arm64", str(src),
                        "-framework", "Metal", "-framework", "Foundation", "-o", str(BUILD / name)], check=True)
        print(BUILD / name)


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    doc = {"meta": {
        "source_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=9", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
        "checkpoint": "none (synthetic)", "macos": platform.mac_ver()[0],
        "macos_build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip()}}
    for key, cmd in (("p5b", [str(BUILD / "replay_probe"), "rebind"]), ("p5c", [str(BUILD / "sparse_probe")])):
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        (out / f"{key}.stdout").write_text(proc.stdout + proc.stderr)
        doc[key] = json.loads(proc.stdout) if proc.returncode == 0 and proc.stdout.strip() else {
            "error": f"rc {proc.returncode}", "stderr": proc.stderr[-2000:]}
    (out / "p5bc.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    build() if args.cmd == "build" else run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
