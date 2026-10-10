from __future__ import annotations

import json
import os
import subprocess
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_vlm_qualification_unit as unit

ROOT = Path(__file__).resolve().parents[1]


def test_profiles_are_read_from_exact_family_producers_and_map_to_server_flags(
    tmp_path,
):
    smol = unit.producer_profile(ROOT, "smolvlm")
    assert smol["max_lanes"] == 1
    assert smol["max_inflight"] == 1
    assert smol["max_context"] == 4096
    assert smol["cache_bytes"] == 1 << 30
    assert smol["route_selection_source"] == "explicit_flag"
    smol_args = unit.server_argv(
        python="/native/python",
        root=ROOT,
        artifact="/models/smol",
        profile=smol,
        host="127.0.0.1",
        port=8123,
        policy_path=tmp_path / "smol-policy.json",
    )
    assert smol_args[smol_args.index("--max-lanes") + 1] == "1"
    assert smol_args[smol_args.index("--max-inflight") + 1] == "1"
    assert smol_args[smol_args.index("--max-context") + 1] == "4096"
    assert smol_args[smol_args.index("--cache-bytes") + 1] == str(1 << 30)
    assert "--ordinary" in smol_args

    native = unit.producer_profile(ROOT, "gemma3n")
    native_args = unit.server_argv(
        python="/native/python",
        root=ROOT,
        artifact="/models/gemma3n",
        profile=native,
        host="127.0.0.1",
        port=8124,
        policy_path=tmp_path / "native-policy.json",
    )
    assert native["max_context"] == 8192
    assert native["max_lanes"] == 2
    assert native["max_inflight"] == 4
    assert native["cache_bytes"] == 2 << 30
    assert "--ordinary" not in native_args


def test_lfm_execution_policy_is_preserved_exactly(tmp_path):
    profile = unit.producer_profile(ROOT, "lfm2_vl")
    policy = tmp_path / "policy.json"
    command = unit.server_argv(
        python="/native/python",
        root=ROOT,
        artifact="/models/lfm",
        profile=profile,
        host="127.0.0.1",
        port=8125,
        policy_path=policy,
    )
    assert json.loads(policy.read_text()) == {"lfm_media_checkpoint": "candidate_v1"}
    assert command[command.index("--execution-policy") + 1] == str(policy)
    with pytest.raises(FileExistsError):
        unit.server_argv(
            python="/native/python",
            root=ROOT,
            artifact="/models/lfm",
            profile=profile,
            host="127.0.0.1",
            port=8125,
            policy_path=policy,
        )


def test_profile_parser_rejects_executable_expressions():
    import ast

    with pytest.raises(ValueError, match="unsupported producer profile"):
        unit._safe_profile_value(
            ast.parse("__import__('os').system('true')").body[0].value
        )


def test_unit_orchestration_runs_media_then_matching_profile_and_generic(
    tmp_path, monkeypatch
):
    root = ROOT
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "config.json").write_text(json.dumps({"model_type": "smolvlm"}))
    (artifact / "model.safetensors").write_bytes(b"weights")
    preflight = tmp_path / "preflight.json"
    preflight.write_text("{}")
    args = SimpleNamespace(
        root=root,
        artifact=artifact,
        output_dir=tmp_path / "out",
        preflight_receipt=preflight,
        family="smolvlm",
        python=Path("/native/python"),
        legacy_source_root=tmp_path / "mlx-vlm",
        cpg_owner_lock="/tmp/cpg-owner",
        cpg_task="task-id",
        generation=17,
        host="127.0.0.1",
        media_timeout=90,
        server_start_timeout=10,
        generic_timeout=90,
    )
    calls = []
    server_calls = []

    class FakeProbe:
        def __init__(self, command):
            self.command = command
            self.returncode = None

        def communicate(self, timeout):
            if self.command[1].endswith("qualify_vlm_routes.py"):
                assert self.command[self.command.index("--generation") + 1] == "17"
                stdout = json.dumps({"passed": True})
            else:
                assert self.command[1].endswith("qualify_serving.py")
                assert self.command[
                    self.command.index("--adapter-qualification") + 1
                ].endswith("media-companion.json")
                output = Path(self.command[self.command.index("--output") + 1])
                output.write_text(
                    json.dumps({"passed": True, "checks": {"profile": True}})
                )
                stdout = ""
            self.returncode = 0
            return stdout, ""

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout):
            return self.returncode

        def kill(self):
            self.returncode = -9

    class FakeServer:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, _signal):
            self.returncode = 0

        def wait(self, timeout):
            return 0

        def kill(self):
            self.returncode = -9

    def fake_popen(command, **kwargs):
        if command[1].endswith(("qualify_vlm_routes.py", "qualify_serving.py")):
            calls.append(command)
            return FakeProbe(command)
        server_calls.append((command, kwargs))
        return FakeServer()

    monkeypatch.setattr(unit.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(unit, "_wait_ready", lambda *args: None)

    report = unit.run_unit(args)
    assert report["passed"] is True
    assert report["media_passed"] is True
    assert report["generic_passed"] is True
    assert [Path(call[1]).name for call in calls] == [
        "qualify_vlm_routes.py",
        "qualify_serving.py",
    ]
    server_command, server_kwargs = server_calls[0]
    assert server_command[1:3] == ["-m", "mlx2.server"]
    assert server_command[server_command.index("--max-context") + 1] == "4096"
    assert server_command[server_command.index("--cache-bytes") + 1] == str(1 << 30)
    assert server_kwargs["cwd"] == root
    assert server_kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(
        args.legacy_source_root.resolve()
    )
    assert report["media_report"].endswith("media-companion.json")


def test_run_unit_rejects_non_loopback_and_unbounded_timeouts():
    args = SimpleNamespace(
        generation=1,
        host="0.0.0.0",
        media_timeout=60,
        server_start_timeout=60,
        generic_timeout=60,
    )
    with pytest.raises(ValueError, match="loopback"):
        unit._validate_run_args(args)
    args.host = "127.0.0.1"
    args.generic_timeout = float("inf")
    with pytest.raises(ValueError, match="finite"):
        unit._validate_run_args(args)


def test_owned_probe_is_reaped_when_cancelled(monkeypatch):
    events = []

    class CancelledProbe:
        def communicate(self, timeout):
            raise KeyboardInterrupt

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout):
            events.append("wait")
            return -15

        def kill(self):
            events.append("kill")

    monkeypatch.setattr(
        unit.subprocess, "Popen", lambda *args, **kwargs: CancelledProbe()
    )
    with pytest.raises(KeyboardInterrupt):
        unit._run_owned(["probe"], cwd=ROOT, env={}, timeout=10)
    assert events == ["terminate", "wait"]


def test_python_interpreter_path_preserves_venv_and_runtime_prefix(tmp_path):
    environment = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    selected = environment / "bin" / "python"
    python = unit.python_interpreter_path(selected)

    assert python == Path(os.path.abspath(selected))
    result = subprocess.run(
        [
            str(python),
            "-c",
            "import json, site, sys; print(json.dumps(dict(prefix=sys.prefix, site=site.getsitepackages())))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = json.loads(result.stdout)
    assert Path(runtime["prefix"]) == environment
    assert any(str(environment) in path for path in runtime["site"])
