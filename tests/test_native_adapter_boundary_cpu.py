"""A missing backend fails closed; lifecycle code does not select model math."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2.adapters.native_cohort import backend_for
from mlx2.adapters.native_hybrid import HybridNativeCohort
from mlx2.adapters.native_qwen3 import Qwen3NativeCohort
from mlx2.runtime import native_admission_retirement as retirement


@pytest.mark.parametrize(
    "backend", [Qwen3NativeCohort(), HybridNativeCohort("hybrid_pair")]
)
def test_adapter_owns_profile_and_allocation(backend):
    adapter = SimpleNamespace(native_cohort_backend=lambda kind: backend)
    assert backend_for(adapter, "requested") is backend
    with pytest.raises(ValueError, match="cancelled before allocation"):
        backend.allocate(adapter, (), None, cancelled=lambda: True)


@pytest.mark.parametrize(
    "adapter",
    [
        object(),
        SimpleNamespace(native_cohort_backend=lambda kind: object()),
    ],
)
def test_undeclared_or_incomplete_backend_fails_closed(adapter):
    with pytest.raises(ValueError, match="capability"):
        backend_for(adapter, "requested")


def test_retirement_polls_all_registered_owners_after_one_raises(monkeypatch):
    monkeypatch.setattr(retirement, "_REAPERS", {})
    calls = []

    def failed():
        calls.append("failed")
        raise RuntimeError("not terminal yet")

    def healthy():
        calls.append("healthy")

    retirement.register_reaper("a", failed)
    retirement.register_reaper("b", healthy)
    retirement.register_reaper("a", failed)
    with pytest.raises(ValueError, match="already registered"):
        retirement.register_reaper("a", healthy)
    retirement.reap_registered()
    assert calls == ["failed", "healthy"]


def test_serving_cohort_installers_do_not_import_model_factories_or_profiles():
    path = Path(__file__).resolve().parents[1] / "src/mlx2/serving.py"
    tree = ast.parse(path.read_text())
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and (
            fn.name.endswith("_cohort") or fn.name == "_reap_native_admission_orphans"
        ):
            imports = [
                n.module or "" for n in ast.walk(fn) if isinstance(n, ast.ImportFrom)
            ]
            assert not any("qwen3" in name and "factory" in name for name in imports)
            assert not any("research_profile" in name for name in imports)
