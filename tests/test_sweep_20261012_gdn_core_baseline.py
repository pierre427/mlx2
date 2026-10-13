"""L5-05 (sweep 2026-10-12): Qwen3.6-35B and Qwen3.5-122B inherit Qwen3.8's
``diagnostics()``, which reports MLX GDN core calls since ``_gdn_core_initial``
and zero without it.  Only ``_init_qwen38`` set the baseline (e3c19168e), so
a selected and running core on these models always reported ``calls = 0``.
"""

import importlib
import inspect
from types import SimpleNamespace

import pytest

from mlx2.runtime.models import gated_delta


def _loader_source(klass):
    return "".join(
        inspect.getsource(member)
        for name, member in vars(klass).items()
        if (name == "__init__" or name.startswith("_init")) and inspect.isfunction(member)
    )


@pytest.mark.parametrize(
    "module,name",
    [
        ("mlx2.adapters.qwen36_35b", "Qwen3635BA3BAdapter"),
        ("mlx2.adapters.qwen35_122b", "Qwen35122BA10BAdapter"),
    ],
)
def test_qwen38_diagnostics_heirs_set_the_gdn_core_baseline(module, name):
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    cls = getattr(importlib.import_module(module), name)
    assert issubclass(cls, Qwen3827BAdapter)
    init = next(klass for klass in cls.__mro__ if "__init__" in vars(klass))
    # Qwen3.8 diagnostics report calls since this baseline, and zero without it.
    assert "_gdn_core_initial" in _loader_source(init), init.__name__


def test_qwen36_reports_selected_core_calls_since_its_baseline(monkeypatch):
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    monkeypatch.setattr(gated_delta, "_ENABLE_GDN_CORE", True)
    monkeypatch.setattr(gated_delta, "_GDN_CORE_CALLS", 10)
    adapter = object.__new__(Qwen3635BA3BAdapter)
    adapter.layout = "qwen36"
    adapter.environment = {"MLX_GDN_CORE": "1"}
    adapter.model = None
    adapter._gdn_core_initial = gated_delta.gdn_core_status()  # as the loader does
    monkeypatch.setattr(gated_delta, "_GDN_CORE_CALLS", 74)
    adapter.descriptor = SimpleNamespace(capabilities=frozenset())
    core = Qwen3827BAdapter.diagnostics(adapter)["gdn_core"]
    assert core["calls"] == 64
