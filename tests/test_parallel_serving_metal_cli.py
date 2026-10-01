"""The HTTP Metal harness cannot start inference in a dry run or unowned run."""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/validate_parallel_serving_metal.py"
)


def test_http_validation_dry_run_cannot_import_tensor_or_spawn_server():
    code = """
import sys, runpy, subprocess
class Guard:
    def find_spec(self, fullname, *args, **kwargs):
        if fullname.startswith(('mlx', 'transformers', 'numpy')):
            raise RuntimeError('tensor import forbidden: ' + fullname)
sys.meta_path.insert(0, Guard())
def forbidden(*args, **kwargs):
    raise RuntimeError('subprocess launch forbidden')
subprocess.Popen = forbidden
sys.argv = [sys.argv[1], '--model', '/missing', '--draft', '/missing',
            '--out', '/missing', '--dry-run', '--target-verify-row-exact']
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
    assert plan["will_execute"] is False
    assert plan["target_verify_row_exact"] is True


def test_http_validation_refuses_unowned_gpu_before_reading_artifacts():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            "/missing",
            "--draft",
            "/missing",
            "--out",
            "/missing",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "i-own-the-gpu" in result.stderr
