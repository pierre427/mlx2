import importlib.util
import json
import signal
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "activate_qualification_arm", ROOT / "scripts" / "activate_qualification_arm.py"
)
activate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(activate)


def state_file(tmp_path, **updates):
    path = tmp_path / "server.json"
    state = {"pid": 42, "pgid": 42, "sid": 42, "command": "server"}
    state.update(updates)
    path.write_text(json.dumps(state))
    return path


def test_exec_transition_command_accepts_only_exact_env_launcher_shape():
    assert activate.exec_transition_command([
        "/usr/bin/env", "PYTHONPATH=src", "/repo/.venv/bin/python", "-m", "mlx2.server"
    ]) == "/repo/.venv/bin/python -m mlx2.server"
    assert activate.exec_transition_command(
        ["/usr/bin/env", "--", "/repo/.venv/bin/python"]
    ) is None
    assert activate.exec_transition_command(
        ["/usr/bin/env", "PYTHONPATH=src", "python", "-m", "mlx2.server"]
    ) is None


def test_exec_transition_command_accepts_existing_relative_target_with_cwd(tmp_path):
    executable = tmp_path / ".venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_text("")
    assert activate.exec_transition_command(
        ["/usr/bin/env", "PYTHONPATH=src", ".venv/bin/python", "-m", "mlx2.server"],
        cwd=tmp_path,
    ) == ".venv/bin/python -m mlx2.server"
    assert activate.exec_transition_command(
        ["/usr/bin/env", "PYTHONPATH=src", "missing/python", "-m", "mlx2.server"],
        cwd=tmp_path,
    ) is None


def test_stop_owned_terms_exact_process_group_and_removes_state(tmp_path):
    path = state_file(tmp_path)
    members = [{"pid": 42, "pgid": 42, "sid": 42},
               {"pid": 43, "pgid": 42, "sid": 42}]
    with patch.object(activate, "process_command", return_value="server"), \
         patch.object(activate, "process_group_members", side_effect=[members, []]), \
         patch.object(activate.os, "killpg") as killpg:
        activate.stop_owned(path)
    killpg.assert_called_once_with(42, signal.SIGTERM)
    assert not path.exists()


def test_stop_owned_refuses_pid_or_session_mismatch(tmp_path):
    path = state_file(tmp_path)
    with patch.object(activate, "process_command", return_value="other"), \
         patch.object(activate, "process_group_members", return_value=[
             {"pid": 42, "pgid": 42, "sid": 42}
         ]), patch.object(activate.os, "killpg") as killpg:
        with pytest.raises(RuntimeError, match="reused/unowned"):
            activate.stop_owned(path)
    killpg.assert_not_called()
    assert path.exists()


def test_stop_owned_accepts_exact_recorded_env_exec_transition(tmp_path):
    path = state_file(
        tmp_path,
        command="/usr/bin/env PYTHONPATH=src /repo/.venv/bin/python -m mlx2.server",
        exec_command="/repo/.venv/bin/python -m mlx2.server",
        argv=["/usr/bin/env", "PYTHONPATH=src", "/repo/.venv/bin/python", "-m", "mlx2.server"],
    )
    members = [{"pid": 42, "pgid": 42, "sid": 42}]
    with patch.object(
        activate, "process_command", return_value="/repo/.venv/bin/python -m mlx2.server"
    ), patch.object(activate, "process_group_members", side_effect=[members, []]), \
         patch.object(activate.os, "killpg") as killpg:
        activate.stop_owned(path)
    killpg.assert_called_once_with(42, signal.SIGTERM)
    assert not path.exists()


def test_stop_owned_refuses_different_exec_target_even_with_same_launcher(tmp_path):
    path = state_file(
        tmp_path,
        command="/usr/bin/env PYTHONPATH=src /repo/.venv/bin/python -m mlx2.server",
        exec_command="/repo/.venv/bin/python -m mlx2.server",
    )
    members = [{"pid": 42, "pgid": 42, "sid": 42}]
    with patch.object(
        activate, "process_command", return_value="/repo/.venv/bin/python -m other.server"
    ), patch.object(activate, "process_group_members", return_value=members), \
         patch.object(activate.os, "killpg") as killpg:
        with pytest.raises(RuntimeError, match="reused/unowned"):
            activate.stop_owned(path)
    killpg.assert_not_called()
    assert path.exists()


def test_stop_owned_derives_exact_transition_for_legacy_recorded_argv(tmp_path):
    path = state_file(
        tmp_path,
        command="/usr/bin/env PYTHONPATH=src /repo/.venv/bin/python -m mlx2.server",
        argv=["/usr/bin/env", "PYTHONPATH=src", "/repo/.venv/bin/python", "-m", "mlx2.server"],
    )
    members = [{"pid": 42, "pgid": 42, "sid": 42}]
    with patch.object(
        activate, "process_command", return_value="/repo/.venv/bin/python -m mlx2.server"
    ), patch.object(activate, "process_group_members", side_effect=[members, []]), \
         patch.object(activate.os, "killpg") as killpg:
        activate.stop_owned(path)
    killpg.assert_called_once_with(42, signal.SIGTERM)


def test_stop_owned_derives_relative_transition_from_recorded_cwd(tmp_path):
    executable = tmp_path / ".venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.write_text("")
    path = state_file(
        tmp_path,
        command="/usr/bin/env PYTHONPATH=src .venv/bin/python -m mlx2.server",
        cwd=str(tmp_path),
        argv=["/usr/bin/env", "PYTHONPATH=src", ".venv/bin/python", "-m", "mlx2.server"],
    )
    members = [{"pid": 42, "pgid": 42, "sid": 42}]
    with patch.object(
        activate, "process_command", return_value=".venv/bin/python -m mlx2.server"
    ), patch.object(activate, "process_group_members", side_effect=[members, []]), \
         patch.object(activate.os, "killpg") as killpg:
        activate.stop_owned(path)
    killpg.assert_called_once_with(42, signal.SIGTERM)


def test_clear_fresh_cache_removes_only_directory_below_cwd(tmp_path):
    cache = tmp_path / "cache" / "cell-a"
    cache.mkdir(parents=True)
    (cache / "entry").write_text("cached")
    activate.clear_fresh_cache(cache, tmp_path)
    assert not cache.exists()
    with pytest.raises(ValueError, match="strictly below cwd"):
        activate.clear_fresh_cache(tmp_path, tmp_path)
    with pytest.raises(ValueError, match="strictly below cwd"):
        activate.clear_fresh_cache(tmp_path.parent / "outside", tmp_path)


def test_stop_owned_escalates_exact_group_to_sigkill(tmp_path):
    path = state_file(tmp_path)
    members = [{"pid": 42, "pgid": 42, "sid": 42}]
    # Initial ownership, one TERM loop, escalation check, then group exits.
    with patch.object(activate, "process_command", return_value="server"), \
         patch.object(activate, "process_group_members", side_effect=[members, []]), \
         patch.object(activate.time, "monotonic", side_effect=[0, 31, 31, 31]), \
         patch.object(activate.time, "sleep"), \
         patch.object(activate.os, "killpg") as killpg:
        activate.stop_owned(path)
    assert [call.args for call in killpg.call_args_list] == [
        (42, signal.SIGTERM), (42, signal.SIGKILL)
    ]
    assert not path.exists()
