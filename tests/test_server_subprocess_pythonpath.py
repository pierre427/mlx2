"""Every harness that spawns ``python -m mlx2.server`` must pin the child to
its own checkout's ``src``.

Regression (2026-09-19): a run queued from a lane worktree inherited the
caller's ``PYTHONPATH``, so the server imported mlx2 from the MAIN checkout
and the harness measured main instead of the lane.  A silently wrong
measurement is worse than a refusal, so this is a structural test over the
live ``scripts/`` harnesses, not one script's unit test.
"""

import importlib.util
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Harnesses that spawn the server as a subprocess from scripts/.
SPAWNERS = sorted(
    path.name
    for path in SCRIPTS.glob("*.py")
    if re.search(r'"-m",\s*"mlx2\.server"', path.read_text())
)


def _module(name):
    spec = importlib.util.spec_from_file_location(
        f"probe_{name.removesuffix('.py')}", SCRIPTS / name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_spawner_inventory_is_not_empty():
    assert SPAWNERS, "no scripts/ harness spawns mlx2.server -- did the glob break?"


@pytest.mark.parametrize("name", SPAWNERS)
def test_spawner_never_inherits_the_ambient_pythonpath(name):
    """No spawner may hand the child a bare ``dict(os.environ)``/``os.environ``."""
    source = (SCRIPTS / name).read_text()
    for bad in ('env=dict(os.environ)', 'env=os.environ'):
        assert bad not in source, (
            f"{name} spawns a subprocess with the inherited environment; the "
            f"server would import mlx2 from whatever checkout is on PYTHONPATH"
        )
    assert "PYTHONPATH" in source, f"{name} does not pin the child's PYTHONPATH"


def test_external_route_smoke_puts_this_checkout_first(monkeypatch):
    module = _module("smoke_external_route_serving.py")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else/src")
    parts = module.server_env()["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(ROOT / "src")
    assert "/somewhere/else/src" in parts


# --------------------------------------------------------------------------
# config-declared runs (qualification manifests)
# --------------------------------------------------------------------------

import json  # noqa: E402

MANIFESTS = sorted(
    path
    for path in (ROOT / "qualification").glob("*.json")
    if "models" in json.loads(path.read_text())
)


def test_the_manifest_inventory_is_not_empty():
    assert MANIFESTS, "no qualification manifest found -- did the glob break?"


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.name)
def test_manifest_arms_declare_the_tree_they_import(manifest):
    """A manifest arm must name the tree its server imports, absolutely.

    Regression: every arm ran `--cwd ~/Desktop/mlx2` with
    `PYTHONPATH=src`, so the relative entry resolved against the MAIN
    checkout and the arm could never measure the worktree under test -- while
    its receipts still looked healthy.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe_matrix", SCRIPTS / "run_qualification_matrix.py"
    )
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)

    document = json.loads(manifest.read_text())
    seen = 0
    for model in document["models"]:
        for arm in model.get("arms", []):
            command = arm["activate_command"]
            label = f"{manifest.name}:{model['name']}/{arm['name']}"
            matrix.validate_arm_source_tree(label, command)
            declared = [a for a in command if a.startswith("PYTHONPATH=")]
            assert declared, label
            value = declared[0].split("=", 1)[1]
            cwd = command[command.index("--cwd") + 1]
            # The imported tree and the declared working tree must agree, or a
            # half-done rewrite to a worktree measures two different trees.
            assert value == str(Path(cwd) / "src"), (
                f"{label}: PYTHONPATH {value} does not match --cwd {cwd}"
            )
            seen += 1
    assert seen, f"{manifest.name} declares no arms"


def test_validator_refuses_a_relative_pythonpath():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "probe_matrix2", SCRIPTS / "run_qualification_matrix.py"
    )
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)

    bad = [
        "python", "activate_qualification_arm.py", "--cwd", "/repo", "--",
        "/usr/bin/env", "PYTHONPATH=src", "/repo/.venv/bin/python", "-m", "mlx2.server",
    ]
    with pytest.raises(ValueError, match="relative PYTHONPATH"):
        matrix.validate_arm_source_tree("probe/arm", bad)
    good = list(bad)
    good[good.index("PYTHONPATH=src")] = "PYTHONPATH=/repo/src"
    matrix.validate_arm_source_tree("probe/arm", good)
    # An arm that does not spawn the server is none of this check's business.
    matrix.validate_arm_source_tree("probe/arm", ["python", "other.py"])
