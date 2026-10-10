"""Qualification auditors and launch helpers import without MLX or a device."""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "qualification/runs/qualify-1010-correctness"


def test_qualification_auditors_import_with_mlx_blocked():
    files = [
        ROOT / "scripts/dloop_ab.py",
        RUN / "dloop_qualification.py",
        RUN / "ladder.py",
        RUN / "performance_assessment.py",
        RUN / "qualification_verdict.py",
        RUN / "run_profile.py",
        RUN / "thermal_ladder.py",
    ]
    child = r'''
import importlib.util
import sys

class BlockMLX:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx.") or fullname == "mlx_lm" or fullname.startswith("mlx_lm."):
            raise AssertionError("CPU qualification import tried to load " + fullname)

sys.meta_path.insert(0, BlockMLX())
for index, path in enumerate(sys.argv[1:]):
    name = "qualification_import_" + str(index)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
assert not any(name == "mlx" or name.startswith(("mlx.", "mlx_lm", "mlx2.runtime")) for name in sys.modules)
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in (ROOT / "src", ROOT / "scripts")
    )
    result = subprocess.run(
        [sys.executable, "-c", child, *(str(path) for path in files)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
