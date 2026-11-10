"""Sweep 2026-11-06, experimental lane regressions (CPU only, no checkpoint)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fake_upstream_artifact(model_path):
    return {"config": {}, "weight_map": {}, "exl3_upstream": {"modules": {}}}


@pytest.mark.parametrize("execution", ["reference", "rows", "fused", "fused_w8a8"])
@pytest.mark.parametrize("resident", ["lm_head", "verify"])
def test_qwen36_resident_dense_refuses_laws_without_fused_nax_linears(
    monkeypatch, execution, resident
):
    """EXP-1: these laws install no fused_nax dense linear, so an explicit
    resident selection made nothing resident yet loaded and was reported."""
    from mlx2.adapters import qwen36_35b

    monkeypatch.setattr(qwen36_35b, "inspect_artifact", _fake_upstream_artifact)
    with pytest.raises(ValueError, match="exl3_resident_dense requires exl3_execution"):
        qwen36_35b.Qwen3635BA3BAdapter(
            "/nonexistent/qwen36-exl3",
            execution_policy={"exl3_execution": execution, "exl3_resident_dense": resident},
        )


def test_qwen36_resident_executions_match_installed_fused_nax_law():
    """The admitted executions are exactly those install_upstream maps to the
    fused_nax dense law (the only law make_resident extends)."""
    from mlx2.adapters.qwen36_35b import EXL3_EXECUTIONS, EXL3_RESIDENT_EXECUTIONS

    source = (ROOT / "src/mlx2/runtime/trellis/upstream.py").read_text()
    assert 'law = "fused_nax" if execution in ("nax", "nax_fused") else execution' in source
    fused_nax = {e for e in EXL3_EXECUTIONS if e in ("nax", "nax_fused") or e == "fused_nax"}
    assert set(EXL3_RESIDENT_EXECUTIONS) == fused_nax


def test_compile_ironmlx_prefill_imports_its_own_checkout():
    """EXP-2: without PYTHONPATH the script imported mlx2 through the shared
    venv's editable install (the primary checkout) while its receipt named
    this checkout's revision."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    probe = (
        "import runpy, sys; sys.argv=['x','--help'];\n"
        "try:\n"
        f"    runpy.run_path({str(ROOT / 'scripts/compile_ironmlx_prefill.py')!r}, run_name='probe')\n"
        "finally:\n"
        "    import mlx2.experimental.ironmlx_tiled_prefill as m; print(m.__file__)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], env=env, cwd="/", capture_output=True, text=True,
        check=True,
    ).stdout.strip().splitlines()[-1]
    assert Path(out).resolve() == (
        ROOT / "src/mlx2/experimental/ironmlx_tiled_prefill.py"
    ).resolve()
