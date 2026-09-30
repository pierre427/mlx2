"""Reproduce the audit's two CPU validation groups without enabling Metal.

Run with the project's development Python. Default output: /tmp/mlx2-cpu-sweep.
The runtime group imports MLX with CPU as default and refuses GPU streams;
the metadata group rejects MLX imports and omits the tensor conftest.
"""

import importlib.abc
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]


def run_group(kind, args):
    if kind == "runtime":
        import mlx.core as mx

        mx.set_default_device(mx.cpu)
        original_device, original_stream = mx.set_default_device, mx.new_stream

        def cpu_device(device):
            if device != mx.cpu:
                raise RuntimeError("CPU audit refused GPU device")
            return original_device(device)

        def cpu_stream(device):
            if device != mx.cpu:
                raise RuntimeError("CPU audit refused GPU stream")
            return original_stream(device)

        mx.set_default_device = cpu_device
        mx.new_stream = cpu_stream
        mx.metal.is_available = lambda: False
        for key in tuple(os.environ):
            if ("TEST" in key or "RUN" in key) and any(
                flag in key for flag in ("METAL", "GPU", "NAX")
            ):
                os.environ.pop(key)
    else:
        class NoMLX(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "mlx" or fullname.startswith("mlx."):
                    raise AssertionError("metadata validation refuses MLX import")

        sys.meta_path.insert(0, NoMLX())
        args = ["--noconftest", *args]
    import pytest

    return pytest.main(args)


def main():
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    os.environ["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
    if len(sys.argv) > 1 and sys.argv[1] in ("runtime", "metadata"):
        return run_group(sys.argv[1], sys.argv[2:])
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mlx2-cpu-sweep")
    output.mkdir(parents=True, exist_ok=True)
    modules = Path(__file__).with_name("import-free-modules.txt").read_text().splitlines()
    groups = {
        "runtime": ["tests", *[f"--ignore={path}" for path in modules]],
        "metadata": modules,
    }
    status = 0
    for kind, args in groups.items():
        command = [
            sys.executable, str(Path(__file__).resolve()), kind, *args,
            "-o", "addopts=", "-q", "--tb=short",
            f"--junitxml={output / (kind + '.xml')}",
        ]
        with (output / (kind + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        print(f"{kind}: exit {result.returncode}; {output / (kind + '.log')}", flush=True)
        status = max(status, int(result.returncode != 0))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
