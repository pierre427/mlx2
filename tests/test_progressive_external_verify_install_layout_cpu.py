"""Progressive source binding must work from a non-editable install.

L3-4 (sweep 2026-10-12): the binding hashed ``src/mlx2/...`` relative to the
repository root, so a wheel/site-packages install raised FileNotFoundError
for every progressive fidelity, including ``exact``.  The files are now
resolved inside the imported ``mlx2`` package; the identity keys and the
hashed bytes are the same in either layout.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx2
from mlx2.runtime import progressive_external_verify as progressive

_DECLARED = (
    "src/mlx2/adapters/qwen38_27b.py",
    "src/mlx2/runtime/models/qwen38_27b.py",
    "src/mlx2/runtime/models/qwen3_next.py",
)
_SCRIPT = """
import json, sys
from types import SimpleNamespace
import mlx2
from mlx2.runtime import progressive_external_verify as p
declared = tuple(json.loads(sys.argv[1]))
model = SimpleNamespace(
    progressive_external_verify_protocol=p.PROGRESSIVE_TARGET_PROTOCOL_VERSION,
    progressive_external_approximate_protocols={
        p.PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION: {
            "fidelity": p.PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED,
            "source_paths": declared,
        }
    },
)
print(json.dumps({
    "package": mlx2.__file__,
    "exact": p.progressive_target_protocol_binding(
        model, p.PROGRESSIVE_FIDELITY_EXACT)["source_hashes"],
    "near_tie": p.progressive_target_protocol_binding(
        model, p.PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED)["source_hashes"],
}))
"""


def _model():
    return SimpleNamespace(
        progressive_external_verify_protocol=(
            progressive.PROGRESSIVE_TARGET_PROTOCOL_VERSION
        ),
        progressive_external_approximate_protocols={
            progressive.PROGRESSIVE_APPROXIMATE_TARGET_PROTOCOL_VERSION: {
                "fidelity": progressive.PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED,
                "source_paths": _DECLARED,
            }
        },
    )


def test_binding_hashes_package_files_from_a_site_packages_layout(tmp_path):
    site = tmp_path / "lib" / "python3" / "site-packages"
    package = Path(mlx2.__file__).resolve().parent
    shutil.copytree(
        package, site / "mlx2", ignore=shutil.ignore_patterns("__pycache__")
    )
    env = {
        key: value for key, value in os.environ.items() if key != "PYTHONPATH"
    }
    env["PYTHONPATH"] = str(site)
    env["MLX2_PREFLIGHT_CPU_ONLY"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", _SCRIPT, json.dumps(_DECLARED)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    installed = json.loads(completed.stdout.strip().splitlines()[-1])
    assert Path(installed["package"]).resolve().parent == (site / "mlx2").resolve()
    model = _model()
    exact = progressive.progressive_target_protocol_binding(
        model, progressive.PROGRESSIVE_FIDELITY_EXACT
    )["source_hashes"]
    near_tie = progressive.progressive_target_protocol_binding(
        model, progressive.PROGRESSIVE_FIDELITY_NEAR_TIE_BOUNDED
    )["source_hashes"]
    # Same identity keys and the same bytes in either layout.
    assert installed["exact"] == exact
    assert installed["near_tie"] == near_tie
    assert set(_DECLARED) <= set(near_tie)
    assert "src/mlx2/runtime/progressive_external_verify.py" in exact
