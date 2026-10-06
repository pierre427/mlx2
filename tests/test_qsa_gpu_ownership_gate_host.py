"""Host-only fail-closed checks for the QSA GPU scripts' paired-lock gate."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "probe": (ROOT / "scripts/probe_qsa_stage1_selectors.py", RuntimeError),
    "qualify": (
        ROOT / "scripts/qualify_qsa_stage1_direct_selector.py",
        SystemExit,
    ),
    "bench": (
        ROOT / "scripts/bench_qsa_stage1_direct_selector_model.py",
        SystemExit,
    ),
}
SHARED = "/Users/Shared/mlxuag/gpu.lock"
TEMPORARY = "/tmp/gpu.lock"


def _ownership_gate(script: Path, environment: dict[str, str], owners: dict):
    tree = ast.parse(script.read_text())
    method = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_prove_gpu_ownership"
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {
        "os": SimpleNamespace(environ=environment),
        "_lock_owner": owners.__getitem__,
    }
    exec(compile(module, str(script), "exec"), namespace)  # noqa: S102
    return namespace["_prove_gpu_ownership"]


def _owners():
    owner = {"lease_id": "lease-1", "session": "session-1", "pid": 123}
    return {SHARED: dict(owner), TEMPORARY: dict(owner)}


@pytest.mark.parametrize("script,error", SCRIPTS.values(), ids=SCRIPTS)
@pytest.mark.parametrize(
    "environment,message",
    [
        ({}, "GPUQ_LEASE"),
        ({"GPUQ_LEASE": "lease-1"}, "GPUQ_SESSION"),
    ],
)
def test_qsa_gpu_gate_requires_explicit_environment_ownership(
    script, error, environment, message
):
    gate = _ownership_gate(script, environment, _owners())
    with pytest.raises(error, match=message):
        gate()


@pytest.mark.parametrize("script,error", SCRIPTS.values(), ids=SCRIPTS)
@pytest.mark.parametrize(
    "field,value,message",
    [
        ("lease_id", "foreign-lease", "GPUQ_LEASE"),
        ("session", "foreign-session", "GPUQ_SESSION"),
    ],
)
def test_qsa_gpu_gate_checks_both_owner_files(script, error, field, value, message):
    owners = _owners()
    owners[TEMPORARY][field] = value
    gate = _ownership_gate(
        script,
        {"GPUQ_LEASE": "lease-1", "GPUQ_SESSION": "session-1"},
        owners,
    )
    with pytest.raises(error, match=message):
        gate()


@pytest.mark.parametrize("script,_error", SCRIPTS.values(), ids=SCRIPTS)
def test_qsa_gpu_gate_accepts_only_the_matching_pair(script, _error):
    owners = _owners()
    gate = _ownership_gate(
        script,
        {"GPUQ_LEASE": "lease-1", "GPUQ_SESSION": "session-1"},
        owners,
    )
    assert gate() == {"shared": owners[SHARED], "temporary": owners[TEMPORARY]}
