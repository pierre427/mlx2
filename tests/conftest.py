"""Routine tests are CPU-only; explicit Metal oracle tests opt in themselves."""
import mlx.core as mx

# Apply before test-module imports create streams or tensor fixtures.
mx.set_default_device(mx.cpu)

# The structured-output scanner pool spawns processes; unit tests exercise
# it explicitly where they need it.
import os

os.environ.setdefault("MLX2_STRUCTURED_WORKERS", "0")

import pytest


@pytest.fixture(autouse=True)
def _restore_process_environment():
    """Adapters' configure_environment() writes os.environ directly.  Restore
    it after every test so one test's serving profile (e.g. Flash-Next's
    segmented self-MTP layout) cannot change a later module's behaviour
    (test_qualify_gdn_retirement after test_execution_policy, 2026-10-01)."""
    saved = dict(os.environ)
    yield
    if os.environ != saved:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture
def served_exp_forms_match(monkeypatch):
    """Every served-exp gate accepts its kernel's spelling (the probe needs
    Metal; tests/test_served_exp_gates.py covers the gates themselves)."""
    from mlx2.runtime.models.served_exp import ServedExpGate

    monkeypatch.setattr(ServedExpGate, "refusal", lambda self, dtype=None: None)


# Test modules that read private material at import time and whose own bytes
# are frozen by a reviewed hash (scripts/qualify_segmented_moe_prefill_research.py
# FROZEN), so they cannot guard themselves.  The public mirror does not carry
# provenance/; there the module is reported as skipped instead of erroring.
_MODULE_PRIVATE_MATERIAL = {
    "test_segmented_moe_prefill_research.py": "provenance/segmented-moe-prefill-research.json",
}
_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


class _PrivateMaterialAbsentModule(pytest.Module):
    skip_reason = ""

    def collect(self):
        pytest.skip(self.skip_reason, allow_module_level=True)


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, parent):
    needed = _MODULE_PRIVATE_MATERIAL.get(module_path.name)
    if needed is None or (_ROOT / needed).is_file():
        return None
    module = _PrivateMaterialAbsentModule.from_parent(parent, path=module_path)
    module.skip_reason = (f"private material absent: {needed} "
                          "(not exported to the public mirror)")
    return module
