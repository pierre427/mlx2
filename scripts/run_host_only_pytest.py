"""Run explicit pytest targets with tensor-framework imports blocked.

This deliberately skips tests/conftest.py, which imports MLX even for CPU tests.
The inherited startup guard also covers Python subprocesses using this
environment. It is an accidental-import guard, not a sandbox for hostile code.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GUARD = """import importlib.abc
import sys
BLOCKED = {
    "mlx", "mlx_lm", "mlx_vlm", "torch", "tensorflow", "jax",
    "jaxlib", "cupy", "pyopencl", "_paged_kv_native",
}
if any(name.split(".", 1)[0] in BLOCKED for name in sys.modules):
    raise RuntimeError("Framework was already imported before host-only guard")
class HostOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in BLOCKED:
            raise RuntimeError("Framework import forbidden in host-only validation: " + fullname)
sys.meta_path.insert(0, HostOnly())
sys._mlx2_host_only_guard = True
"""

# Python prints sitecustomize failures and continues. Check installation before
# pytest can import a test, so a missing or failed startup guard fails closed.
BOOTSTRAP = """import sys
if not getattr(sys, "_mlx2_host_only_guard", False):
    raise SystemExit("Host-only startup guard is missing; refusing test execution")
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("targets", nargs="+")
    args = parser.parse_args()
    if (
        not math.isfinite(args.timeout)
        or args.timeout <= 0
        or any(x.startswith("-") for x in args.targets)
    ):
        parser.error("positive timeout and explicit test targets required")
    if args.output.exists():
        parser.error("evidence output already exists; use a fresh path")
    root = Path(__file__).resolve().parents[1]
    for target in args.targets:
        path = (root / target.split("::", 1)[0]).resolve()
        if not path.is_relative_to(root / "tests") or not path.is_file():
            parser.error("targets must be existing files beneath tests/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlx2-host-only-") as directory:
        guard = Path(directory)
        (guard / "sitecustomize.py").write_text(GUARD)
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(
                (str(guard), str(root / "src"), str(root / "tests"))
            ),
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            PYTHONDONTWRITEBYTECODE="1",
            MLX2_STRUCTURED_WORKERS="0",
        )
        command = [
            sys.executable,
            "-c",
            BOOTSTRAP,
            "--noconftest",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "-q",
            *args.targets,
        ]
        started = time.monotonic()
        try:
            result = subprocess.run(
                command,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            code, output = result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired as exc:
            code = 124
            output = (
                "host-only test timeout\n"
                + str(exc.stdout or "")
                + str(exc.stderr or "")
            )
        proof = {
            "schema": "mlx2.host-only-tests.v1",
            "command": command,
            "framework_imports_blocked": True,
            "returncode": code,
            "seconds": time.monotonic() - started,
            "output": output,
        }
        args.output.write_text(json.dumps(proof, indent=2) + "\n")
        print(output, end="")
        return code


if __name__ == "__main__":
    raise SystemExit(main())
