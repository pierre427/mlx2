"""Advisory helper contracts; fake providers only, no model or external CLI calls."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2 import desktop_helpers as helpers


def controller(tmp_path):
    preferences = {"helper_enabled": True, "helper_codex_enabled": True, "helper_claude_enabled": True,
                   "adapter_helper": "local", "adapter_model_path": "/models/advisor", "helper_timeout": 30}
    model = tmp_path / "candidate"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"synthetic"}')
    class Library:
        def checked_path(self, path, identity):
            if path != str(model) or identity != "synthetic-identity":
                raise ValueError("Stale identity")
            return model
    return SimpleNamespace(home=tmp_path, preferences=preferences, primary=SimpleNamespace(library=Library()),
                           snapshot=lambda: {"servers": [{"endpoint": {"url": "http://127.0.0.1:9999", "pid": 10, "model_path": "/models/advisor"}}]},
                           dispatch=lambda request: {"url": "http://127.0.0.1:9999", "api_key": "private-key"})


def test_fixed_helper_commands_have_no_permission_bypass(tmp_path):
    codex = helpers.helper_command("codex", "/bin/codex", tmp_path)
    assert codex[codex.index("--sandbox") + 1] == "read-only"
    assert "--ignore-user-config" in codex
    claude = helpers.helper_command("claude", "/bin/claude", tmp_path)
    assert claude[claude.index("--tools") + 1] == ""
    assert "--restricted" in claude and "--strict-mcp-config" in claude
    for command in (codex, claude):
        assert not any("bypass" in item or "dangerously" in item for item in command)


def test_helper_disabled_and_unknown_provider(tmp_path):
    control = controller(tmp_path)
    manager = helpers.Helpers(control)
    control.preferences["helper_enabled"] = False
    with pytest.raises(ValueError, match="Enable"):
        manager.dispatch({"command": "helper_start", "prompt": "hello"})
    control.preferences["helper_enabled"] = True
    with pytest.raises(ValueError, match="Unknown helper"):
        manager.dispatch({"command": "helper_start", "provider": "shell", "prompt": "hello"})
    assert not manager.home.exists()


def test_local_adapter_receipt_and_review_only(tmp_path, monkeypatch):
    control = controller(tmp_path)
    manager = helpers.Helpers(control)
    monkeypatch.setattr(helpers, "source_identity", lambda: {"source_revision": "synthetic-commit"})
    monkeypatch.setattr(manager, "_run_local", lambda row: "Proposed adapter, requiring review and tests.")
    result = manager.dispatch({"command": "adapter_draft", "path": str(tmp_path / "candidate"), "identity": "synthetic-identity"})
    row = manager.jobs[result["job_id"]]
    row["thread"].join(2)
    assert row["status"] == "completed"
    proposal = manager.dispatch({"command": "helper_result", "job_id": row["id"]})
    assert "REVIEW ONLY" in proposal["proposal"]
    receipt = json.loads((Path(row["workspace"]) / "receipt.json").read_text())
    assert receipt["context"]["source"]["source_revision"] == "synthetic-commit"
    assert receipt["context"]["checkpoint"] == "synthetic-identity"
    assert "api_key" not in json.dumps(receipt)
    assert "private-key" not in json.dumps(manager.snapshot())
    assert "target" not in row and "prompt" not in row
    assert (tmp_path / "candidate" / "config.json").read_text() == '{"model_type":"synthetic"}'


def test_local_helper_requires_already_running_designated_model(tmp_path):
    control = controller(tmp_path)
    manager = helpers.Helpers(control)
    control.preferences["adapter_model_path"] = "/models/not-running"
    with pytest.raises(ValueError, match="not running"):
        manager.dispatch({"command": "helper_start", "provider": "local", "prompt": "help"})
    assert not manager.home.exists()


def test_fake_cli_subprocess_and_cancel(tmp_path, monkeypatch):
    control = controller(tmp_path)
    manager = helpers.Helpers(control)
    monkeypatch.setattr(helpers, "executable", lambda name: sys.executable)
    monkeypatch.setattr(helpers, "helper_command", lambda provider, binary, workspace: [binary, "-c", "import sys; print('advice: ' + sys.stdin.read())"])
    result = manager.dispatch({"command": "helper_start", "provider": "claude", "prompt": "literal $(no shell)"})
    row = manager.jobs[result["job_id"]]
    row["thread"].join(3)
    assert row["status"] == "completed"
    assert "$(no shell)" in manager.dispatch({"command": "helper_result", "job_id": row["id"]})["proposal"]
    monkeypatch.setattr(helpers, "helper_command", lambda *args: [sys.executable, "-c", "import time; time.sleep(30)"])
    result = manager.dispatch({"command": "helper_start", "provider": "claude", "prompt": "cancel"})
    row = manager.jobs[result["job_id"]]
    manager.dispatch({"command": "helper_cancel", "job_id": row["id"]})
    row["thread"].join(3)
    assert row["status"] == "cancelled"
    assert row["child"].poll() is not None
    manager.close()


def test_helper_failure_does_not_expose_credentials(tmp_path, monkeypatch):
    manager = helpers.Helpers(controller(tmp_path))
    def fail(row):
        raise ValueError("sensitive private-key body")
    monkeypatch.setattr(manager, "_run_local", fail)
    result = manager.dispatch({"command": "helper_start", "provider": "local", "prompt": "hello"})
    row = manager.jobs[result["job_id"]]
    row["thread"].join(2)
    assert row["status"] == "failed"
    assert "private-key" not in json.dumps(manager.snapshot())
