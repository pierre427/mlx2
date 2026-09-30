"""TF32 is decided once per process, before any MLX dispatch, in one place."""

import re
import subprocess
import sys
from pathlib import Path

import mlx2
from mlx2.process_env import PROCESS_NUMERICS, apply_process_numerics

SRC = Path(mlx2.__file__).parent


def test_default_applied_at_import_and_explicit_value_wins():
    env = {}
    assert apply_process_numerics(env) == PROCESS_NUMERICS
    env = {"MLX_ENABLE_TF32": "1"}
    assert apply_process_numerics(env) == {}
    assert env["MLX_ENABLE_TF32"] == "1"


def test_importing_the_package_sets_the_default_before_mlx():
    script = (
        "import os, sys; os.environ.pop('MLX_ENABLE_TF32', None); "
        "import mlx2; assert 'mlx' not in sys.modules; "
        "print(os.environ['MLX_ENABLE_TF32'])"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env={"PYTHONPATH": str(SRC.parent), "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0"


def test_no_adapter_carries_its_own_tf32_literal():
    copies = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if path.name not in {"process_env.py", "import_env.py"}
        and "MLX_ENABLE_TF32" in path.read_text()
    ]
    assert not copies, f"TF32 pinned outside process_env: {copies}"
