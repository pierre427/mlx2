"""Every import-time flag a tensor module reads must be visible to the guard.

The 09-25 triage lost fused GDN because a module read its flag before the
adapter pinned it and nothing noticed; ``MLX_QWEN36_FUSED_GDN_DECODE`` had the
same gap until 09-30.  This test walks the modules so the allowlist cannot
drift again.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType

import pytest

from mlx2.runtime.models import import_env

MODELS = Path(import_env.__file__).parent


def _import_time_calls(node):
    """Calls evaluated when the module is imported: module and class bodies,
    not function bodies."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return
    if isinstance(node, ast.Call):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from _import_time_calls(child)


def _import_time_env_reads(source: str):
    tree = ast.parse(source)
    names = []
    for node in tree.body:
        for call in _import_time_calls(node):
            # ``os.environ.get("MLX_X")`` and helper calls such as
            # ``_env_int("MLX_X", 4)`` alike: any flag-shaped literal argument
            # evaluated at import is a flag read at import.
            for arg in call.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith(("MLX_", "MLX2_")):
                        names.append(arg.value)
    snapshots = any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func).endswith("snapshot")
        for node in tree.body
    )
    return names, snapshots


def test_every_import_time_flag_is_covered_and_snapshotted():
    uncovered, unsnapshotted = [], []
    for path in sorted(MODELS.glob("*.py")):
        names, snapshots = _import_time_env_reads(path.read_text())
        flags = [n for n in names if n.startswith(("MLX_", "MLX2_"))]
        if not flags:
            continue
        for name in flags:
            if not name.startswith(import_env.PREFIXES):
                uncovered.append(f"{path.name}: {name}")
        if not snapshots:
            unsnapshotted.append(path.name)
    assert not uncovered, f"flags outside import_env.PREFIXES: {uncovered}"
    assert not unsnapshotted, f"modules reading flags without snapshot(): {unsnapshotted}"


def test_late_qwen36_pin_fails_closed(monkeypatch):
    name = "mlx2.runtime.models._fake_qwen36_for_test"
    monkeypatch.delenv("MLX_QWEN36_FUSED_GDN_DECODE", raising=False)
    monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(import_env._SNAPSHOTS, name, {})
    import_env.snapshot(name)
    monkeypatch.setenv("MLX_QWEN36_FUSED_GDN_DECODE", "1")
    with pytest.raises(import_env.ImportOrderError, match="MLX_QWEN36_FUSED_GDN_DECODE"):
        import_env.assert_profile_applied("the Qwen3.6 adapter")
