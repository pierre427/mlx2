"""L5-04 / L8-3 (sweep 2026-10-12): Flash-Next reports the MLX GDN core.

e3c19168e reduced serving's ``_execution_diagnostics`` to the adapter's own
route diagnostics; serving used to add the process-wide ``gdn_core`` counter.
Flash-Next's policy can select the core (``gdn_core`` -> ``MLX_GDN_CORE=1``)
but its ``diagnostics()`` never reported it, so a selected and running core
was absent from status and receipts and qualification (which requires
``gdn_core`` whenever it is selected) could never observe it.
"""

import importlib.util
import pathlib

from mlx import nn

import mlx2
from mlx2 import qualification, serving
from mlx2.adapters.flash_next import FlashNextAdapter
from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.models import gated_delta

ROOT = pathlib.Path(mlx2.__file__).resolve().parents[2]


def _adapter(environment, *, baseline):
    adapter = object.__new__(FlashNextAdapter)
    adapter.model = nn.Module()
    adapter._tables = []
    adapter._diagnostic_modules = ()
    adapter.policy = FlashNextPolicy()
    adapter.environment = environment
    if baseline:
        adapter._gdn_core_initial = gated_delta.gdn_core_status()  # as the loader does
    return adapter


def test_selected_core_reaches_serving_status_since_load(monkeypatch):
    monkeypatch.setattr(gated_delta, "_ENABLE_GDN_CORE", True)
    monkeypatch.setattr(gated_delta, "_GDN_CORE_CALLS", 5)
    adapter = _adapter({"MLX_GDN_CORE": "1"}, baseline=True)
    monkeypatch.setattr(gated_delta, "_GDN_CORE_CALLS", 12)
    core = serving._execution_diagnostics(adapter)["gdn_core"]
    assert core["enabled"] is True and core["calls"] == 7


def test_unselected_idle_core_is_absent(monkeypatch):
    monkeypatch.setattr(gated_delta, "_ENABLE_GDN_CORE", False)
    monkeypatch.setattr(gated_delta, "_GDN_CORE_CALLS", 9)
    adapter = _adapter({"MLX_GDN_CORE": "0"}, baseline=True)
    assert "gdn_core" not in serving._execution_diagnostics(adapter)


def test_partial_adapter_claims_no_process_calls(monkeypatch):
    monkeypatch.setattr(gated_delta, "_ENABLE_GDN_CORE", True)
    monkeypatch.setattr(gated_delta, "_GDN_CORE_CALLS", 9)
    core = _adapter({"MLX_GDN_CORE": "1"}, baseline=False).diagnostics()["gdn_core"]
    assert core["calls"] == 0


def test_loader_sets_the_baseline():
    import inspect

    assert "_gdn_core_initial" in inspect.getsource(FlashNextAdapter._init_flash_next)


def test_selected_core_is_observed_by_qualification(monkeypatch):
    environment = {"MLX_GDN_CORE": "1"}
    settings = {"environment": environment, "max_lanes": 1}
    assert "gdn_core" in qualification._default_on_mechanisms(settings)[0]
    spec = importlib.util.spec_from_file_location(
        "qualify_serving_g2_flash", ROOT / "scripts" / "qualify_serving.py"
    )
    qs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qs)
    monkeypatch.setattr(gated_delta, "_ENABLE_GDN_CORE", True)
    monkeypatch.setattr(gated_delta, "_GDN_CORE_CALLS", 3)
    adapter = _adapter(environment, baseline=True)
    initial = {"execution": serving._execution_diagnostics(adapter), "environment": environment}
    monkeypatch.setattr(gated_delta, "_GDN_CORE_CALLS", 43)
    final = {"execution": serving._execution_diagnostics(adapter), "environment": environment}
    assert qs.default_on_observations(final, initial)["gdn_core"] == 40
