from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_with_gpu_locks.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_with_gpu_locks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exclusive_receipt_refuses_to_overwrite(tmp_path: Path) -> None:
    module = _load_module()
    receipt = tmp_path / "attempt.json"

    module._write_json_exclusive(receipt, {"attempt_id": "first"})

    with pytest.raises(FileExistsError):
        module._write_json_exclusive(receipt, {"attempt_id": "second"})
    assert json.loads(receipt.read_text()) == {"attempt_id": "first"}


def test_terminal_receipt_is_written_before_legacy_receipt(monkeypatch) -> None:
    module = _load_module()
    calls = []
    monkeypatch.setattr(
        module,
        "_write_json_exclusive",
        lambda path, payload: calls.append(("immutable", path, payload.copy())),
    )
    monkeypatch.setattr(
        module,
        "_atomic_write_json",
        lambda path, payload: calls.append(("legacy", path, payload.copy())),
    )

    module._save_terminal(Path("latest.json"), Path("attempt.json"), {"status": "done"})

    assert [kind for kind, _, _ in calls] == ["immutable", "legacy"]


def test_retries_keep_immutable_attempts_and_update_legacy(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_module()
    waiters = tmp_path / "waiters"
    locks = (tmp_path / "shared.lock", tmp_path / "tmp.lock")
    for lock in locks:
        lock.touch()
    monkeypatch.setattr(module, "WAITERS", waiters)
    monkeypatch.setattr(module, "LOCKS", locks)
    monkeypatch.setattr(module, "foreign_model_processes", list)

    class FakeChild:
        pid = 4242
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: FakeChild())
    attempt_ids = iter(("a" * 32, "b" * 32))
    monkeypatch.setattr(
        module.uuid, "uuid4", lambda: SimpleNamespace(hex=next(attempt_ids))
    )
    legacy = tmp_path / "run.json"
    argv = [
        "--session",
        "session-1",
        "--label",
        "cell-1",
        "--receipt",
        str(legacy),
        "--",
        "ignored-command",
    ]

    assert module.main(argv) == 0
    first = json.loads(legacy.read_text())
    assert module.main(argv) == 0
    second = json.loads(legacy.read_text())

    assert first["logical_run_id"] == second["logical_run_id"] == "session-1:cell-1"
    assert first["attempt_id"] == "a" * 32
    assert second["attempt_id"] == "b" * 32
    first_immutable = Path(first["immutable_receipt"])
    second_immutable = Path(second["immutable_receipt"])
    assert first_immutable != second_immutable
    assert json.loads(first_immutable.read_text()) == first
    assert json.loads(second_immutable.read_text()) == second
    assert json.loads(legacy.read_text()) == second
    assert not list(tmp_path.glob(".*.tmp.*"))
