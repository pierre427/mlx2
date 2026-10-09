"""Run every direct-only source-contract module so its failures reach pytest.

paged-packed#1 (sweep 2026-10-08): several tests/ modules raise
``unittest.SkipTest`` at import unless run as ``__main__`` (tests/conftest.py
imports MLX; they forbid it).  Under pytest they were only ever reported as
skipped, so four of them went red on stale fakes (the native read's
``plan.profile``, the grouped sampler's receipt import, the hybrid lifecycle's
relative imports) without any suite noticing.  Run each one in a fresh
interpreter (no conftest, no MLX, no GPU device) and require a clean unittest
exit.
"""
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _is_direct_only(path):
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return False
    for node in tree.body:
        if (isinstance(node, ast.If)
                and ast.unparse(node.test).replace('"', "'") == "__name__ != '__main__'"
                and any(isinstance(item, ast.Raise) and "SkipTest" in ast.unparse(item)
                        for item in node.body)):
            return True
    return False


DIRECT_ONLY = sorted(
    str(path.relative_to(ROOT)) for path in (ROOT / "tests").glob("test_*.py")
    if _is_direct_only(path))


def test_direct_only_manifest_is_not_empty():
    # Guard against this wrapper passing vacuously if discovery breaks.
    for name in ("tests/test_paged_q1_stock_sdpa_source_cpu.py",
                 "tests/test_paged_grouped_sampling_source_cpu.py",
                 "tests/test_varlen_write_only_defer_source_cpu.py",
                 "tests/test_hybrid_serving_lifecycle_source_cpu.py"):
        assert name in DIRECT_ONLY


@pytest.mark.parametrize("module", DIRECT_ONLY)
def test_direct_only_source_contract_passes(module):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    completed = subprocess.run([sys.executable, module], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=180)
    tail = (completed.stdout + completed.stderr)[-4000:]
    assert completed.returncode == 0, f"{module} failed when run directly:\n{tail}"
    assert "\nOK" in tail, tail
