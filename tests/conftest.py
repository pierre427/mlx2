"""Routine tests are CPU-only; explicit Metal oracle tests opt in themselves."""
import mlx.core as mx

# Apply before test-module imports create streams or tensor fixtures.
mx.set_default_device(mx.cpu)

# The structured-output scanner pool spawns processes; unit tests exercise
# it explicitly where they need it.
import ast
import os
import subprocess
import sys

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

# Use the qualification runner's reviewed isolation manifest for ordinary
# pytest too: collection-time import blockers and package stand-ins must not
# contaminate CPU tensor tests. Read the literal without importing the harness.
_guard_tree = ast.parse((_ROOT / "scripts" / "qualify_serving.py").read_text())
_guard_manifest = next(
    node.value for node in _guard_tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "PREFLIGHT_IMPORT_GUARD_MODULES"
            for target in node.targets)
)
_ISOLATED_HOST_MODULES = {
    __import__("pathlib").Path(path).name for path in ast.literal_eval(_guard_manifest)
} | {
    # This module executes two guarded helper modules via runpy at collection.
    "test_n20_gdn_wave_profile_cpu.py",
    "test_spomin400_bootstrap_progress_cpu.py",
}


class _IsolatedHostItem(pytest.Item):
    def runtest(self):
        path = self.parent.path
        command = [sys.executable, "-m", "pytest", "--noconftest", "-p", "no:cacheprovider",
                   "-o", "addopts=", "-q", str(path)]
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(_ROOT / "src"), env.get("PYTHONPATH"),
        )))
        result = subprocess.run(
            command, cwd=_ROOT, env=env, capture_output=True, text=True, timeout=120,
            check=False,
        )
        self.add_report_section("call", "isolated host suite", result.stdout + result.stderr)
        if result.returncode:
            pytest.fail(
                f"isolated host suite exited {result.returncode}\n"
                f"{result.stdout}{result.stderr}", pytrace=False,
            )


class _IsolatedHostModule(pytest.File):
    def collect(self):
        yield _IsolatedHostItem.from_parent(self, name="isolated_host_suite")


class _PrivateMaterialAbsentModule(pytest.Module):
    skip_reason = ""

    def collect(self):
        pytest.skip(self.skip_reason, allow_module_level=True)


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, parent):
    if module_path.name in _ISOLATED_HOST_MODULES:
        return _IsolatedHostModule.from_parent(parent, path=module_path)
    needed = _MODULE_PRIVATE_MATERIAL.get(module_path.name)
    if needed is None or (_ROOT / needed).is_file():
        return None
    module = _PrivateMaterialAbsentModule.from_parent(parent, path=module_path)
    module.skip_reason = (f"private material absent: {needed} "
                          "(not exported to the public mirror)")
    return module
