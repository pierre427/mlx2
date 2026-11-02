"""Run Metal pytest selections (MLX2_RUN_GPU_TESTS=1) inside an owned GPU window.

Usage under bench.sh: glm53_gpu_pytest.py --out DIR -- <pytest args>.
Writes the pytest log to DIR/pytest.log and exits with pytest's code.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    extra = [a for a in args.pytest_args if a != "--"]
    env = {**os.environ, "MLX2_RUN_GPU_TESTS": "1"}
    with (args.out / "pytest.log").open("w") as log:
        log.write(f"source_revision {args.source_revision}\n")
        log.flush()
        rc = subprocess.call([sys.executable, "-m", "pytest", "-q", *extra], stdout=log,
                             stderr=subprocess.STDOUT, env=env)
    print((args.out / "pytest.log").read_text()[-3000:], flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
