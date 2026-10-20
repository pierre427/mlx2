from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

RUNNER = (
    Path(__file__).parents[1] / "qualification/runs/ast-whole-layer-qwen38-20261020/"
    "qwen38-27b-whole-layer-final-head-refresh-runner.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("whole_layer_refresh_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def _repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.check_call(["git", "init", "-q"], cwd=root)
    subprocess.check_call(["git", "config", "user.name", "mlx2 test"], cwd=root)
    subprocess.check_call(
        ["git", "config", "user.email", "mlx2-test@example.invalid"], cwd=root
    )
    source = root / "src/reviewed.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n")
    subprocess.check_call(["git", "add", "src/reviewed.py"], cwd=root)
    subprocess.check_call(["git", "commit", "-qm", "fixture"], cwd=root)
    return root, source, _git(root, "rev-parse", "HEAD")


def test_refresh_source_contract_accepts_only_clean_live_head(tmp_path):
    runner = _module()
    root, source, head = _repository(tmp_path)

    hashes = runner.validate_live_source_revision(root, head, ("src/reviewed.py",))
    assert set(hashes) == {"src/reviewed.py"}

    with pytest.raises(RuntimeError, match="revision differs"):
        runner.validate_live_source_revision(root, "0" * 40, ("src/reviewed.py",))

    source.write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="tracked source is dirty"):
        runner.validate_live_source_revision(root, head, ("src/reviewed.py",))


def test_refresh_source_contract_rejects_ambiguous_closure(tmp_path):
    runner = _module()
    root, _source, head = _repository(tmp_path)
    with pytest.raises(ValueError, match="empty or duplicated"):
        runner.validate_live_source_revision(
            root, head, ("src/reviewed.py", "src/reviewed.py")
        )
