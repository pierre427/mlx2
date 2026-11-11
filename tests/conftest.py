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

if os.environ.get("MLX2_PREFLIGHT_CPU_ONLY") == "1":
    # A CPU-only publication gate must not let availability-only Metal tests
    # opt themselves in on an Apple host. Explicit GPU selection fails before
    # a tensor can be dispatched; unit tests may still mock hardware probes.
    mx.metal.is_available = lambda: False
    _set_default_device = mx.set_default_device

    def _cpu_only_default_device(device):
        if device == mx.gpu:
            raise RuntimeError("GPU execution forbidden by CPU-only preflight")
        return _set_default_device(device)

    mx.set_default_device = _cpu_only_default_device

os.environ.setdefault("MLX2_STRUCTURED_WORKERS", "0")

import pytest


@pytest.fixture(autouse=True)
def _restore_process_environment():
    """Adapters' configure_environment() writes os.environ directly.  Restore
    it after every test so one test's serving profile (e.g. Flash-Next's
    segmented self-MTP layout) cannot change a later module's behaviour
    (test_qualify_gdn_retirement after test_execution_policy, 2026-10-01)."""
    from mlx2 import process_env

    # Each test starts as a process whose first adapter profile has not run:
    # the operator selections latched by one test must not become another
    # test's "operator" environment (sweep 2026-10-12 L1-F2).
    saved_operator = process_env._OPERATOR_SELECTORS
    process_env._OPERATOR_SELECTORS = None
    saved = dict(os.environ)
    yield
    process_env._OPERATOR_SELECTORS = saved_operator
    if os.environ != saved:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture
def served_exp_forms_match(monkeypatch):
    """Every served-exp gate accepts its kernel's spelling (the probe needs
    Metal; tests/test_served_exp_gates.py covers the gates themselves)."""
    from mlx2.runtime.models.served_exp import ServedExpGate

    monkeypatch.setattr(ServedExpGate, "refusal", lambda self, dtype=None: None)
    monkeypatch.setattr(
        ServedExpGate, "adaptation_refusal", lambda self, dtype=None: None
    )
    monkeypatch.setattr(
        ServedExpGate, "spelling", lambda self, dtype=None: self.kernel_exp
    )


# Test modules that read private material at import time and whose own bytes
# are frozen by a reviewed hash (scripts/qualify_segmented_moe_prefill_research.py
# FROZEN), so they cannot guard themselves.  The public mirror does not carry
# provenance/; there the module is reported as skipped instead of erroring.
_MODULE_PRIVATE_MATERIAL = {
    "test_segmented_moe_prefill_research.py": "provenance/segmented-moe-prefill-research.json",
    # Modules that load a private harness, policy set or package at import or
    # in every test (not exported to the public mirror).
    "test_ast_smoke_host.py": "qualification/runs/ast-smokes-20261018/ast_smoke.py",
    "test_qwen38_whole_layer_refresh_runner_host.py": (
        "qualification/runs/ast-whole-layer-qwen38-20261020/"
        "qwen38-27b-whole-layer-final-head-refresh-runner.py"),
    "test_flash_next_ast_perf_policy_cpu.py": (
        "qualification/runs/ast-smokes-20261018/policies/flash-next-native-mtp-ast.json"),
    "test_qwen36_ast_perf_policies_cpu.py": (
        "qualification/runs/ast-smokes-20261018/policies/qwen36-35b-ast-q8-moe-attn-k2-primary.json"),
    "test_decision_benchmark_v1_cpu.py": "benchmarks/decisions/v1/run.py",
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


# The public source export ships no feature-decision record
# (docs/research/model-feature-decisions.tsv), while a private checkout always
# has it (AGENTS.md requires it). Only in the export, a test that needs
# private evidence (run receipts, policies, docs) or a never-published module
# is reported as skipped instead of failing; in a private checkout these
# hooks do nothing, so missing evidence there still fails.
_PUBLIC_EXPORT = not (_ROOT / "docs" / "research" / "model-feature-decisions.tsv").is_file()
_PRIVATE_ROOTS = {"qualification", "docs", "provenance", "benchmark_results", "benchmarks",
                  "artifacts", "output"}


def _unexported_reason(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ModuleNotFoundError) and error.name and "mcp" in error.name.lower():
            return f"module {error.name} is not part of the public source export"
        if isinstance(error, FileNotFoundError) and error.filename:
            path = __import__("pathlib").Path(os.fsdecode(error.filename))
            try:
                rel = path.resolve().relative_to(_ROOT.resolve())
            except ValueError:
                rel = None
            if rel is not None and rel.parts and (rel.parts[0] in _PRIVATE_ROOTS
                                                  or rel.parts[0].startswith(".local")) \
                    and not path.exists():
                return f"private material absent: {rel} (not exported to the public mirror)"
        error = error.__cause__ or error.__context__
    return None


# Tests whose inputs exist on the export only as older frozen copies (they
# pass in a private checkout, which has the current files).
_PUBLIC_EXPORT_SKIP_MODULES = {
    "test_export_public_guard.py":
        "needs scripts/public_export_withheld.txt, which the exporter never exports",
    "test_segmented_moe_prefill_native_qualifier.py":
        "binds the reviewed hash of a private provenance record",
    "test_segmented_moe_native_identity_cpu.py":
        "binds the reviewed hash of a private provenance record",
    "test_packed_prefill_long_contract_cpu.py": "reads private run receipts in an isolated host suite",
    "test_thermal_ladder_identifier_oracle.py":
        "tests the current benchmark_results ladder script; the export carries an older frozen copy",
}
_PUBLIC_EXPORT_SKIP_TESTS = {
    "test_qualification_matrix.py::test_canonical_manifests_name_real_services_selected_arms_and_new_arms":
        "reads the private canonical qualification manifests",
    "test_server_subprocess_pythonpath.py::test_manifest_arms_declare_the_tree_they_import":
        "reads the private qualification manifests",
    "test_qwen_wide_row_diagnostic_cpu.py::test_default_planner_imports_no_mlx_and_claims_no_execution":
        "runs a private qualification planner",
    "test_smoke_tokenizers_v1_serving_ab.py::test_live_retained_manifest_and_artifact_identity_are_exact":
        "reads a private retained worker manifest",
    "test_qualify_serving_receipts.py::test_preflight_tree_binds_script_fixtures_and_provenance":
        "binds private provenance and fixture files into the preflight tree",
}


def pytest_collection_modifyitems(config, items):
    if not _PUBLIC_EXPORT:
        return
    for item in items:
        key = f"{item.path.name}::{item.originalname if hasattr(item, 'originalname') else item.name}"
        reason = _PUBLIC_EXPORT_SKIP_TESTS.get(key)
        if reason:
            item.add_marker(pytest.mark.skip(reason=f"{reason} (not exported to the public mirror)"))


class _PublicExportModule(pytest.Module):
    def _getobj(self):
        try:
            return super()._getobj()
        except (FileNotFoundError, pytest.Collector.CollectError) as error:
            reason = _unexported_reason(error)
            if reason:
                pytest.skip(reason, allow_module_level=True)
            raise


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if not _PUBLIC_EXPORT or call.excinfo is None or call.when not in ("setup", "call"):
        return
    report = outcome.get_result()
    reason = _unexported_reason(call.excinfo.value) if report.failed else None
    if reason:
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: {reason}")


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, parent):
    if _PUBLIC_EXPORT and module_path.name in _PUBLIC_EXPORT_SKIP_MODULES:
        module = _PrivateMaterialAbsentModule.from_parent(parent, path=module_path)
        module.skip_reason = (f"{_PUBLIC_EXPORT_SKIP_MODULES[module_path.name]} "
                              "(not exported to the public mirror)")
        return module
    if module_path.name in _ISOLATED_HOST_MODULES:
        return _IsolatedHostModule.from_parent(parent, path=module_path)
    needed = _MODULE_PRIVATE_MATERIAL.get(module_path.name)
    if needed is None or (_ROOT / needed).is_file():
        return _PublicExportModule.from_parent(parent, path=module_path) if _PUBLIC_EXPORT else None
    module = _PrivateMaterialAbsentModule.from_parent(parent, path=module_path)
    module.skip_reason = (f"private material absent: {needed} "
                          "(not exported to the public mirror)")
    return module


# Tests should write only to temporary directories. Report (do not fail) any
# file a test run created or changed in the checkout, so the writer can be
# found: such files leak between runs and into exports.
def _checkout_state():
    try:
        out = subprocess.run(["git", "--no-optional-locks", "-C", str(_ROOT), "status", "--porcelain",
                              "--untracked-files=all"],
                             capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    state = {}
    for line in out.splitlines():
        path = line[3:]
        target = _ROOT / path
        try:
            stamp = target.stat().st_mtime_ns if target.is_file() else None
        except OSError:
            stamp = None
        state[path] = (line[:2], stamp)
    return state


def pytest_sessionstart(session):
    # Not in xdist workers, and not in a pytest that a test launched (its output may be parsed).
    if (os.environ.get("MLX2_TEST_WRITE_REPORT", "1") != "0" and not hasattr(session.config, "workerinput")
            and "PYTEST_CURRENT_TEST" not in os.environ):
        session.config._mlx2_checkout_before = _checkout_state()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    before = getattr(config, "_mlx2_checkout_before", None)
    if before is None:
        return
    after = _checkout_state() or {}
    written = sorted(path for path, value in after.items() if before.get(path) != value)
    if written:
        terminalreporter.write_sep(
            "-", f"{len(written)} checkout file(s) changed during this test session "
                 "(a test, or another session working in this checkout)")
        for path in written[:40]:
            terminalreporter.write_line(f"  {path}")
        if len(written) > 40:
            terminalreporter.write_line(f"  ... and {len(written) - 40} more")
