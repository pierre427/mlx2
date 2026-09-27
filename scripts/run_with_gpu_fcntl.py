#!/usr/bin/env python3
"""Hold the lab's /tmp/gpu.lock fcntl lock for one child process.

The shared CPG job wrapper must take the CPG lease and host lock first. This
helper never removes or truncates another owner's lock file.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_with_gpu_fcntl.py COMMAND [ARGS...]")
    if not Path("/Users/Shared/mlxuag/gpu.lock/owner.json").is_file():
        raise SystemExit("CPG host GPU lock is required before taking fcntl lock")
    descriptor = os.open("/tmp/gpu.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("/tmp/gpu.lock is held by another process") from exc
        env = dict(os.environ, MLX2_GPU_FCNTL_LOCKED="1")
        return subprocess.call(sys.argv[1:], env=env)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
