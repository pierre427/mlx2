"""Environment publication/identity tests; installers and model work are mocked."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2 import desktop_runner as runner


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "source checkout"
    package = source / "src/mlx2"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# synthetic runtime fixture\n")
    (source / "pyproject.toml").write_text('[project]\nname="mlx2"\nversion="0.0.0"\n')
    home = tmp_path / "Application Support/MLX2"
    home.mkdir(parents=True)
    model = tmp_path / "synthetic model fixture"
    model.mkdir()
    monkeypatch.setattr(
        runner.importlib.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": "mlx"}, version="0.32.4")],
    )
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/usr/bin/true")
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *args, **kwargs: "synthetic-source-revision" if kwargs.get("text") else b"synthetic-commit-object",
    )
    calls = []

    class Installer:
        def __init__(self, args, **kwargs):
            calls.append(args)
            if args[1] == "venv":
                python = Path(args[-1]) / "bin/python"
                python.parent.mkdir(parents=True)
                python.touch()
            self.returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", Installer)
    manager = runner.RunnerManager(home, source)
    options = {
        "multimodal": False,
        "cancel": threading.Event(),
        "progress": lambda message: None,
        "log_path": home / "setup.log",
    }
    return manager, model, calls, options


def test_runner_uses_encoded_paths_pins_and_reuses_only_completed_environment(setup):
    manager, model, calls, options = setup
    record = manager.ensure(model, "synthetic-fingerprint", **options)
    assert Path(record["python"]).is_file()
    assert (
        record["binding"]["checkpoint"]
        == "synthetic model fixture@synthetic-fingerprint"
    )
    install = next(args for args in calls if "install" in args)
    assert "%20" in install[install.index("--constraint") + 1]
    assert install[-2].startswith("mlx2[serve,observability] @ file:")
    assert "--offline" in install
    count = len(calls)
    assert manager.ensure(model, "synthetic-fingerprint", **options) == record
    assert len(calls) == count


def test_changed_source_gets_new_runner_and_corrupt_snapshot_is_refused(setup):
    manager, model, _, options = setup
    first = manager.ensure(model, "synthetic", **options)
    source = manager.source / "src/mlx2/__init__.py"
    source.write_text("# revised synthetic source\n")
    second = manager.ensure(model, "synthetic", **options)
    assert first["python"] != second["python"]
    (Path(second["source"]) / "src/mlx2/__init__.py").write_text("# corrupted\n")
    with pytest.raises(RuntimeError, match="identity"):
        manager.ensure(model, "synthetic", **options)


def test_failed_install_and_cancellation_never_publish_ready_record(setup, monkeypatch):
    manager, model, _, options = setup

    class Failure:
        def __init__(self, *args, **kwargs):
            self.returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(runner.subprocess, "Popen", Failure)
    with pytest.raises(RuntimeError, match="setup failed"):
        manager.ensure(model, "synthetic", **options)
    assert not list(manager.home.rglob("ready.json"))
    options["cancel"].set()
    with pytest.raises(runner.PreparationCancelled):
        manager.ensure(model, "synthetic", **options)
    assert not list(manager.home.rglob("ready.json"))


def test_snapshot_excludes_editable_install_metadata_and_external_links(
    setup, tmp_path
):
    manager, _, _, _ = setup
    egg = manager.source / "src/mlx2.egg-info"
    egg.mkdir()
    (egg / "PKG-INFO").write_text("generated editable metadata")
    assert not any(
        "egg-info" in path for path in runner.source_manifest(manager.source)
    )
    outside = tmp_path / "outside.py"
    outside.write_text("# not a runtime source\n")
    (manager.source / "src/external.py").symlink_to(outside)
    with pytest.raises(ValueError, match="leaves"):
        runner.source_manifest(manager.source)
