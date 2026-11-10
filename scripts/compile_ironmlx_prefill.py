#!/usr/bin/env python3
"""Compile the experimental shader offline; never create a Metal device."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# Compile this checkout's shader: the receipt records this checkout's
# revision, while an editable install may resolve mlx2 to another worktree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx2.experimental.ironmlx_tiled_prefill import metal_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.output.resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error("output directory must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    compiler = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "--version"],
                              capture_output=True, text=True, check=False)
    results = []
    for ntcol in (32, 64):
        source = directory / f"fused-nt{ntcol}.metal"
        output = directory / f"fused-nt{ntcol}.air"
        source.write_text(metal_source(ntcol))
        command = ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-Werror",
                   "-c", str(source), "-o", str(output)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        results.append({"ntcol": ntcol, "command": command, "returncode": result.returncode,
                        "diagnostics": result.stdout + result.stderr,
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "air_sha256": hashlib.sha256(output.read_bytes()).hexdigest()
                        if result.returncode == 0 else None})
    receipt = {"source_revision": revision, "checkpoint": "none; compile-only",
               "gpu_used": False, "ane_used": False, "device_created": False,
               "qualified": False, "compiler": compiler.stdout + compiler.stderr,
               "results": results}
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return int(any(row["returncode"] for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
