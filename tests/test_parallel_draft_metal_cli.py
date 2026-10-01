"""Metal smoke host preflight must never allocate/import tensor libraries."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/smoke_parallel_draft_metal.py"
spec = importlib.util.spec_from_file_location("parallel_smoke_cli", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def args(*extra):
    return smoke.build_parser().parse_args(
        [
            "--model",
            "/missing-target",
            "--draft",
            "/missing-draft",
            "--out",
            "/tmp/not-written",
            *extra,
        ]
    )


def test_default_gpu_run_refuses_without_ownership():
    with pytest.raises(ValueError, match="i-own-the-gpu"):
        smoke.preflight(args())


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--max-tokens", "17"),
        ("--max-tokens", "1"),
        ("--prompt-tokens", "129"),
        ("--prompt-tokens", "15"),
        ("--num-draft", "16"),
        ("--xpress-num-passes", "0"),
    ],
)
def test_smoke_is_bounded(flag, value):
    with pytest.raises(ValueError):
        smoke.preflight(args("--dry-run", flag, value))


def test_dry_run_under_hard_tensor_import_guard():
    code = """
import sys, runpy
class Guard:
    def find_spec(self, fullname, *args, **kwargs):
        if fullname.startswith(('mlx', 'transformers', 'numpy')):
            raise RuntimeError('tensor library import forbidden: ' + fullname)
sys.meta_path.insert(0, Guard())
sys.argv = [sys.argv[1], '--model', '/missing-target', '--draft', '/missing-draft',
            '--out', '/missing-output', '--dry-run']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["will_execute"] is False and plan["performance_claim"] is False
    assert (
        plan["policy"]["num_draft"] == 15 and plan["policy"]["xpress_num_passes"] == 6
    )
    assert plan["max_tokens"] == 16


def test_host_float32_accepts_real_cpu_bfloat16_buffer():
    import mlx.core as mx
    import numpy as np

    mx.set_default_device(mx.cpu)
    result = smoke.host_float32(mx.array([1.5, -2.25, 0.0], dtype=mx.bfloat16))
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [1.5, -2.25, 0.0])
