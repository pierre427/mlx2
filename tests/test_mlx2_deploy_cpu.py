"""scripts/mlx2_deploy.py and mlx2.production: temp repos, temp prod roots, a fake launchctl."""
import importlib.util
import json
import os
import plistlib
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mlx2_deploy_cpu", ROOT / "scripts/mlx2_deploy.py")
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)

from mlx2 import production  # noqa: E402


class FakeLaunchd:
    def __init__(self, loaded=(), disabled=()):
        self._loaded = set(loaded)
        self._disabled = set(disabled)
        self.calls = []

    def info(self, label):
        return {"pid": "4242", "runs": "1"} if label in self._loaded else None

    def loaded(self, label):
        return label in self._loaded

    def disabled(self):
        return set(self._disabled)

    def bootout(self, label):
        self.calls.append(("bootout", label))
        self._loaded.discard(label)

    def bootstrap(self, plist):
        label = plistlib.load(open(plist, "rb"))["Label"]
        self.calls.append(("bootstrap", label))
        self._loaded.add(label)

    def enable(self, label):
        self.calls.append(("enable", label))
        self._disabled.discard(label)

    def disable(self, label):
        self.calls.append(("disable", label))
        self._disabled.add(label)


@pytest.fixture
def env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    files = {
        "src/mlx2/__init__.py": "",
        "src/mlx2/server.py": "VALUE = 1\n",
        "src/mlx2/production.py": (ROOT / "src/mlx2/production.py").read_text(),
        "scripts/tool.py": "x = 1\n",
        "qualification/runs/evidence.json": "{}\n",
        "README.md": "readme\n",
    }
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=T", "-c", "user.email=t@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "base"], check=True)
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    prod, agents, services = tmp_path / "prod", tmp_path / "LaunchAgents", repo / "ops" / "services"
    agents.mkdir(); services.mkdir(parents=True)
    for name, value in (("REPO", repo), ("PROD", prod), ("LAUNCH_AGENTS", agents), ("SERVICES", services)):
        monkeypatch.setattr(deploy, name, value)
    venv = prod / "venvs" / "lockhash"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    (venv / "VENV.json").write_text("{}")
    return {"repo": repo, "sha": sha, "prod": prod, "agents": agents, "services": services, "venv": venv}


def service(env, name="svc", port=8282, commit=True, **extra):
    spec = {"label": f"com.example.{name}", "port": port, "model": "/models/m",
            "args": ["--max-context", "4096"], **extra}
    (env["services"] / f"{name}.json").write_text(json.dumps(spec))
    if commit:
        subprocess.run(["git", "-C", str(env["repo"]), "add", "ops"], check=True)
        subprocess.run(["git", "-C", str(env["repo"]), "-c", "user.name=T", "-c", "user.email=t@example.invalid",
                        "-c", "commit.gpgsign=false", "commit", "-qm", f"service {name}"], check=True)
    return spec


OK = dict(health=lambda *a, **k: True, listeners=lambda port: set(), check_model=False)
BAD = dict(health=lambda *a, **k: False, listeners=lambda port: set(), check_model=False)


def foreign_plist(env, label, port, run_at_load=True):
    path = env["agents"] / f"{label}.plist"
    with path.open("wb") as fh:
        plistlib.dump({"Label": label, "RunAtLoad": run_at_load,
                       "ProgramArguments": ["/dev/python", "-m", "mlx2.server", "--port", str(port)]}, fh)
    return path


def test_release_is_read_only_minimal_and_idempotent(env):
    release = deploy.build_release(env["sha"], venv=env["venv"], repo=env["repo"])
    info = json.loads((release / "RELEASE.json").read_text())
    assert info["sha"] == env["sha"] and release.name == env["sha"]
    assert (release / "src/mlx2/server.py").is_file() and (release / "scripts/tool.py").is_file()
    assert not (release / "qualification").exists()
    assert not os.access(release / "src/mlx2/server.py", os.W_OK)
    assert not (release.stat().st_mode & stat.S_IWUSR)
    assert production.release_info(release) == info
    assert deploy.build_release(env["sha"], repo=env["repo"]) == release


def test_release_import_smoke_failure_leaves_no_release(env):
    (env["repo"] / "src/mlx2/server.py").write_text("raise ImportError('broken')\n")
    subprocess.run(["git", "-C", str(env["repo"]), "-c", "user.name=T", "-c", "user.email=t@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-qam", "break"], check=True)
    with pytest.raises(SystemExit, match="import smoke failed"):
        deploy.build_release("HEAD", venv=env["venv"], repo=env["repo"])
    assert not any((env["prod"] / "releases").glob("[0-9a-f]*"))


def test_lock_hash_ignores_comments_and_order(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("# lock\nmlx==0.32.4\nnumpy==2.5.1\n")
    b.write_text("numpy==2.5.1\n\nmlx==0.32.4\n# other comment\n")
    assert deploy.lock_hash(a) == deploy.lock_hash(b)
    assert deploy.parse_lock("mlx==0.32.4\nmlx-vlm @ git+https://x/y@abc\n") == {
        "mlx": "0.32.4", "mlx-vlm": "git+https://x/y@abc"}


def test_service_definition_is_validated(env):
    (env["services"] / "bad.json").write_text(json.dumps(
        {"label": "x", "port": 1, "model": "/m", "args": ["--port", "9"]}))
    with pytest.raises(SystemExit, match="set by the deploy"):
        deploy.load_service("bad")
    (env["services"] / "bad.json").write_text(json.dumps({"label": "x", "port": 1, "model": "/m", "typo": 1}))
    with pytest.raises(SystemExit, match="unknown keys"):
        deploy.load_service("bad")


def test_plist_runs_the_release_with_the_production_venv(env):
    spec = service(env)
    release = deploy.build_release(env["sha"], repo=env["repo"])
    data = deploy.plist_for(deploy.load_service("svc"), release, env["venv"])
    assert data["ProgramArguments"][0] == str(env["venv"] / "bin/python")
    assert data["ProgramArguments"][-2:] == ["--max-context", "4096"]
    assert deploy.plist_port(data) == spec["port"]
    assert data["EnvironmentVariables"]["PYTHONPATH"] == str(release / "src")
    assert data["WorkingDirectory"] == str(release)
    assert "Desktop" not in json.dumps(data)


def test_port_owners_counts_loaded_and_login_jobs_only(env):
    foreign_plist(env, "com.example.loaded", 8282, run_at_load=False)
    foreign_plist(env, "com.example.at-login", 8282)
    foreign_plist(env, "com.example.disabled", 8282)
    foreign_plist(env, "com.example.other-port", 9000)
    launchd = FakeLaunchd(loaded={"com.example.loaded"}, disabled={"com.example.disabled"})
    labels = {o["label"] for o in deploy.port_owners(8282, launchd)}
    assert labels == {"com.example.loaded", "com.example.at-login"}


def test_promote_dry_run_changes_nothing(env):
    service(env)
    deploy.build_release(env["sha"], repo=env["repo"])
    foreign_plist(env, "com.example.old", 8282)
    launchd = FakeLaunchd(loaded={"com.example.old"})
    plan = deploy.promote("svc", env["sha"], "lockhash", yes=False, launchd=launchd, **OK)
    assert plan["dry_run"] and [o["label"] for o in plan["stop_and_disable"]] == ["com.example.old"]
    assert launchd.calls == [] and not (env["agents"] / "com.example.svc.plist").exists()


def test_promote_takes_the_port_and_records_the_deployment(env):
    service(env)
    deploy.build_release(env["sha"], repo=env["repo"])
    foreign_plist(env, "com.example.old", 8282)
    launchd = FakeLaunchd(loaded={"com.example.old"})
    result = deploy.promote("svc", env["sha"], "lockhash", yes=True, launchd=launchd, **OK)
    assert result["ok"]
    assert ("bootout", "com.example.old") in launchd.calls and ("disable", "com.example.old") in launchd.calls
    assert launchd.loaded("com.example.svc") and not launchd.loaded("com.example.old")
    assert "com.example.old" in launchd.disabled()
    written = plistlib.load(open(env["agents"] / "com.example.svc.plist", "rb"))
    assert written["EnvironmentVariables"]["MLX2_RELEASE"] == env["sha"]
    assert json.loads((env["prod"] / "services/svc/deployed.json").read_text())["release"] == env["sha"]


def test_failed_health_check_rolls_back_plist_and_port_owner(env):
    service(env)
    deploy.build_release(env["sha"], repo=env["repo"])
    old_owner = foreign_plist(env, "com.example.old", 8282)
    previous = env["agents"] / "com.example.svc.plist"
    with previous.open("wb") as fh:
        plistlib.dump({"Label": "com.example.svc", "ProgramArguments": ["/old/python"], "RunAtLoad": False}, fh)
    launchd = FakeLaunchd(loaded={"com.example.old"})
    result = deploy.promote("svc", env["sha"], "lockhash", yes=True, launchd=launchd, **BAD)
    assert not result["ok"] and result["rolled_back"]
    assert plistlib.load(open(previous, "rb"))["ProgramArguments"] == ["/old/python"]
    assert launchd.loaded("com.example.old") and not launchd.loaded("com.example.svc")
    assert "com.example.old" not in launchd.disabled()
    assert old_owner.is_file()
    assert not (env["prod"] / "services/svc/deployed.json").exists()


def test_gc_keeps_referenced_and_newest(env):
    service(env)
    used = deploy.build_release(env["sha"], repo=env["repo"])
    deploy.promote("svc", env["sha"], "lockhash", yes=True, launchd=FakeLaunchd(), **OK)
    for name in ("a" * 40, "b" * 40):
        (env["prod"] / "releases" / name).mkdir()
    plan = deploy.gc(keep=0, yes=False)
    assert str(used) in plan["keep"] and str(env["venv"]) in plan["keep"]
    assert {str(env["prod"] / "releases" / n) for n in ("a" * 40, "b" * 40)} <= set(plan["remove"])
    assert (env["prod"] / "releases" / ("a" * 40)).exists()
    deploy.gc(keep=0, yes=True)
    assert used.exists() and not (env["prod"] / "releases" / ("a" * 40)).exists()


def test_production_port_is_reserved_for_release_trees(tmp_path, monkeypatch):
    monkeypatch.delenv(production.ALLOW_ENV, raising=False)
    assert production.port_refusal(8282, root=tmp_path)
    assert production.port_refusal(8300, root=tmp_path) is None
    assert production.port_refusal(8282, allow=True, root=tmp_path) is None
    monkeypatch.setenv(production.ALLOW_ENV, "1")
    assert production.port_refusal(8282, root=tmp_path) is None
    monkeypatch.delenv(production.ALLOW_ENV)
    (tmp_path / "RELEASE.json").write_text(json.dumps({"sha": "0" * 40}))
    assert production.port_refusal(8282, root=tmp_path)  # a RELEASE.json outside releases/<sha>
    release = tmp_path / ("0" * 40)
    release.mkdir()
    (release / "RELEASE.json").write_text(json.dumps({"sha": "0" * 40}))
    assert production.port_refusal(8282, root=release) is None
    (release / "RELEASE.json").write_text(json.dumps({"sha": "short"}))
    assert production.port_refusal(8282, root=release)


def test_server_refuses_the_production_port_from_a_development_tree(monkeypatch, capsys):
    monkeypatch.delenv(production.ALLOW_ENV, raising=False)
    from mlx2 import server

    monkeypatch.setattr(sys, "argv", ["mlx2.server", "--model", "/nonexistent", "--port", "8282"])
    with pytest.raises(SystemExit) as caught:
        server.main()
    assert caught.value.code == 2 and "reserved for production" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["mlx2.server", "--model", "/nonexistent", "--port", "8282",
                                      "--allow-production-port"])
    with pytest.raises(SystemExit):
        server.main()
    assert "reserved for production" not in capsys.readouterr().err


def test_service_definitions_in_ops_are_valid():
    for path in sorted((ROOT / "ops/services").glob("*.json")):
        spec = json.loads(path.read_text())
        assert set(spec) <= deploy.SERVICE_KEYS, path
        assert spec["port"] == production.PRODUCTION_PORT
        assert not {"--port", "--host", "--model"} & set(spec.get("args", [])), path


def test_install_writes_a_disabled_plist_without_starting(env):
    service(env, "idle")
    deploy.build_release(env["sha"], repo=env["repo"])
    launchd = FakeLaunchd(disabled={"com.example.idle"})
    record = deploy.install("idle", env["sha"], "lockhash", yes=True, launchd=launchd)
    assert record["active"] is False
    assert [c for c in launchd.calls if c[0] != "disable"] == []
    assert (env["agents"] / "com.example.idle.plist").is_file()
    assert deploy.port_owners(8282, launchd) == []  # will not start at login


def test_install_refuses_a_running_service(env):
    service(env, "live")
    deploy.build_release(env["sha"], repo=env["repo"])
    with pytest.raises(SystemExit, match="use promote"):
        deploy.install("live", env["sha"], "lockhash", yes=True, launchd=FakeLaunchd(loaded={"com.example.live"}))


def test_promote_marks_the_previous_active_service_inactive(env):
    service(env, "one"); service(env, "two")
    deploy.build_release(env["sha"], repo=env["repo"])
    launchd = FakeLaunchd()
    deploy.promote("one", env["sha"], "lockhash", yes=True, launchd=launchd, **OK)
    deploy.promote("two", env["sha"], "lockhash", yes=True, launchd=launchd, **OK)
    one = json.loads((env["prod"] / "services/one/deployed.json").read_text())
    two = json.loads((env["prod"] / "services/two/deployed.json").read_text())
    assert two["active"] and not one["active"]
    assert "com.example.one" in launchd.disabled() and not launchd.loaded("com.example.one")


def test_port_owner_detection_reads_launcher_scripts_and_definitions(env, tmp_path):
    script = tmp_path / "serve.sh"
    script.write_text('#!/bin/zsh\nPORT="${PORT:-8282}"\nexec python -m mlx2.server --port "${PORT}"\n')
    with (env["agents"] / "com.example.script.plist").open("wb") as fh:
        plistlib.dump({"Label": "com.example.script", "ProgramArguments": [str(script)]}, fh)
    with (env["agents"] / "com.example.eq.plist").open("wb") as fh:
        plistlib.dump({"Label": "com.example.eq", "ProgramArguments": ["/py", "--port=8282"]}, fh)
    with (env["agents"] / "com.example.shc.plist").open("wb") as fh:
        plistlib.dump({"Label": "com.example.shc", "ProgramArguments": ["/bin/sh", "-c", "x --port 8282"]}, fh)
    service(env, "defined")
    with (env["agents"] / "com.example.defined.plist").open("wb") as fh:
        plistlib.dump({"Label": "com.example.defined", "ProgramArguments": ["/opaque"]}, fh)
    labels = {o["label"] for o in deploy.port_owners(8282, FakeLaunchd())}
    assert labels == {"com.example.script", "com.example.eq", "com.example.shc", "com.example.defined"}


def test_bootstrap_failure_rolls_back_owner_enable_state_and_history(env):
    service(env)
    deploy.build_release(env["sha"], repo=env["repo"])
    foreign_plist(env, "com.example.old", 8282)

    class Failing(FakeLaunchd):
        def bootstrap(self, plist):
            if plistlib.load(open(plist, "rb"))["Label"] == "com.example.svc":
                raise subprocess.CalledProcessError(5, "launchctl bootstrap")
            super().bootstrap(plist)

    launchd = Failing(loaded={"com.example.old"}, disabled={"com.example.svc"})
    with pytest.raises(subprocess.CalledProcessError):
        deploy.promote("svc", env["sha"], "lockhash", yes=True, launchd=launchd, **OK)
    assert launchd.loaded("com.example.old") and "com.example.old" not in launchd.disabled()
    assert "com.example.svc" in launchd.disabled()  # prior disabled state restored
    assert not (env["agents"] / "com.example.svc.plist").exists()
    history = (env["prod"] / "services/svc/history.jsonl").read_text().splitlines()
    assert json.loads(history[-1])["rolled_back"] and "CalledProcessError" in json.loads(history[-1])["error"]


def test_install_refuses_an_enabled_label_unless_forced(env):
    service(env, "paused")
    deploy.build_release(env["sha"], repo=env["repo"])
    with pytest.raises(SystemExit, match="enabled at login"):
        deploy.install("paused", env["sha"], "lockhash", yes=True, launchd=FakeLaunchd())
    record = deploy.install("paused", env["sha"], "lockhash", yes=True, launchd=FakeLaunchd(), force=True)
    assert record["ok"] and record["active"] is False


def test_definitions_may_not_override_owned_flags_or_env(env):
    for spec in ({"args": ["--port=9000"]}, {"args": ["--allow-production-port"]},
                 {"env": {"PYTHONPATH": "/dev/src"}}, {"launchd": {"Program": "/bin/sh"}}):
        (env["services"] / "bad.json").write_text(json.dumps({"label": "x", "port": 1, "model": "/m", **spec}))
        with pytest.raises(SystemExit):
            deploy.load_service("bad")


def test_promote_refuses_an_uncommitted_definition(env):
    service(env, "draft", commit=False)
    deploy.build_release(env["sha"], repo=env["repo"])
    with pytest.raises(SystemExit, match="commit the service definition"):
        deploy.promote("draft", env["sha"], "lockhash", yes=True, launchd=FakeLaunchd(), **OK)
