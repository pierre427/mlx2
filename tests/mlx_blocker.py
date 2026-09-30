"""Make real MLX unimportable for one test, inside the shared CPU process.

``conftest.py`` imports ``mlx.core`` to force the CPU device, so ``mlx`` is
always in ``sys.modules`` and a test cannot ask that it was never imported:
every such assertion failed on every full run (87 of them on 2026-09-30) and
told nobody anything.  What a CPU-only test can ask is that MLX is not
*importable* while it runs: every ``mlx`` module is unlinked from
``sys.modules`` for the test's duration and a finder refuses new imports, so
any ``import mlx`` in the code under test raises.  Already-loaded modules keep
their references; ``sys.modules`` is restored afterwards.  A module-level
``sys.meta_path.insert`` must not be used for this: it outlives the test and
breaks every later test that imports an MLX submodule for the first time.
"""

import importlib.abc
import sys


class BlockMLX(importlib.abc.MetaPathFinder):
    def __init__(self, label="a CPU-only test"):
        self.label = label

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError(f"real MLX import ({fullname}) during {self.label}")
        return None


def mlx_module_names():
    return [name for name in sys.modules if name == "mlx" or name.startswith("mlx.")]


def block_mlx_imports(monkeypatch, label="a CPU-only test"):
    """pytest form: unlink MLX and refuse imports until the test ends."""
    for name in mlx_module_names():
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [BlockMLX(label), *sys.meta_path])


class unimportable_mlx:
    """unittest form: ``self.addCleanup`` restores what ``__enter__`` unlinked."""

    def __init__(self, label="a CPU-only test"):
        self._label = label

    def __enter__(self):
        self._saved = {name: sys.modules.pop(name) for name in mlx_module_names()}
        self._finder = BlockMLX(self._label)
        sys.meta_path.insert(0, self._finder)
        return self

    def __exit__(self, *exc):
        sys.meta_path.remove(self._finder)
        for name in mlx_module_names():
            sys.modules.pop(name, None)
        sys.modules.update(self._saved)
        return False


def install_for_test_case(case):
    """Call from ``setUp``: MLX is unimportable until the test's cleanup."""
    guard = unimportable_mlx(case.id())
    guard.__enter__()
    case.addCleanup(guard.__exit__, None, None, None)
