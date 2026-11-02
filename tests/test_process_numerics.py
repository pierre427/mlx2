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


def test_runtime_sync_default_applied_at_import_and_explicit_value_wins():
    from mlx2.process_env import PROCESS_RUNTIME_DEFAULTS, apply_process_runtime_defaults

    env = {}
    assert apply_process_runtime_defaults(env) == PROCESS_RUNTIME_DEFAULTS
    assert env["MLX_METAL_FAST_SYNCH"] == "1"
    env = {"MLX_METAL_FAST_SYNCH": "0"}
    assert apply_process_runtime_defaults(env) == {}
    assert env["MLX_METAL_FAST_SYNCH"] == "0"
    script = (
        "import os, sys; os.environ.pop('MLX_METAL_FAST_SYNCH', None); "
        "import mlx2; assert 'mlx' not in sys.modules; "
        "print(os.environ['MLX_METAL_FAST_SYNCH'])"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         env={"PYTHONPATH": str(SRC.parent), "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "1"


def test_command_buffer_policy_is_opt_in_and_explicit_values_win():
    import pytest

    from mlx2.process_env import COMMAND_BUFFER_POLICIES, apply_process_runtime_defaults

    env = {}
    applied = apply_process_runtime_defaults(env)
    assert "MLX_MAX_OPS_PER_BUFFER" not in env and "MLX_MAX_MB_PER_BUFFER" not in env
    assert "MLX_MAX_OPS_PER_BUFFER" not in applied
    env = {"MLX2_COMMAND_BUFFERS": "ops100_v1"}
    applied = apply_process_runtime_defaults(env)
    assert env["MLX_MAX_OPS_PER_BUFFER"] == "100" and env["MLX_MAX_MB_PER_BUFFER"] == "1000000"
    assert COMMAND_BUFFER_POLICIES["ops100_v1"].items() <= applied.items()
    env = {"MLX2_COMMAND_BUFFERS": "ops100_v1", "MLX_MAX_OPS_PER_BUFFER": "40"}
    apply_process_runtime_defaults(env)
    assert env["MLX_MAX_OPS_PER_BUFFER"] == "40" and env["MLX_MAX_MB_PER_BUFFER"] == "1000000"
    with pytest.raises(ValueError, match="MLX2_COMMAND_BUFFERS"):
        apply_process_runtime_defaults({"MLX2_COMMAND_BUFFERS": "one_buffer"})


def test_raised_byte_cap_is_detected_however_it_was_set():
    from mlx2.process_env import raised_command_buffer_byte_cap

    assert not raised_command_buffer_byte_cap({})
    assert not raised_command_buffer_byte_cap({"MLX_MAX_MB_PER_BUFFER": "50"})
    assert raised_command_buffer_byte_cap({"MLX_MAX_MB_PER_BUFFER": "1000000"})
    assert raised_command_buffer_byte_cap({"MLX_MAX_MB_PER_BUFFER": "lots"})
