"""Canonical media supervisor tests: no process, service or model is started."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlx2.desktop_media_canonical_service import CanonicalService
from mlx2.desktop_records import RecordStore


def controller(tmp_path):
    value = SimpleNamespace(home=tmp_path, preferences={}, store=RecordStore(tmp_path), event=Mock(),
        primary=SimpleNamespace(library=SimpleNamespace(root=tmp_path / "models")))
    value.configure_preferences = lambda patch: value.preferences.update(patch)
    return value


def test_canonical_source_configuration_validates_capability_and_does_not_edit_source(tmp_path):
    ctl = controller(tmp_path)
    service = CanonicalService(ctl)
    with pytest.raises(ValueError, match="containing"):
        service.configure({"source": str(tmp_path)})
    source = tmp_path / "external"
    (source / "src/mlx2/media").mkdir(parents=True)
    (source / "src/mlx2/media/server.py").write_text("original")
    (source / "pyproject.toml").write_text("original")
    service.configure({"source": str(source)})
    assert service.settings()["source"] == str(source)
    assert (source / "src/mlx2/media/server.py").read_text() == "original"
    with pytest.raises(ValueError, match="one source"):
        service.configure({"source": str(source), "extras": "arbitrary"})


def test_existing_canonical_service_is_reused_and_close_does_not_stop_it(tmp_path, monkeypatch):
    ctl = controller(tmp_path)
    client = Mock()
    client.request.return_value = {"service": "mlx2-media"}
    monkeypatch.setattr("mlx2.desktop_media_canonical_service.CanonicalMediaClient", Mock(return_value=client))
    client.base = "http://127.0.0.1:8777"
    service = CanonicalService(ctl)
    assert service.start()["state"] == "reused"
    assert {p["capability"] for p in ctl.preferences["media_services"]} == {"image", "video", "voice", "voice_training"}
    service.close()
    assert len(ctl.store.jobs(10)) == 0
    client.request.assert_called_once_with("GET", "/v1/media/status")


def test_stop_requires_explicit_confirmation_and_proven_idle_queue(tmp_path, monkeypatch):
    service = CanonicalService(controller(tmp_path))
    with pytest.raises(ValueError, match="Confirm"):
        service.stop()
    client = Mock()
    monkeypatch.setattr("mlx2.desktop_media_canonical_service.CanonicalMediaClient", Mock(return_value=client))
    for status in [{"service": "mlx2-media"}, {"service": "mlx2-media", "queue": {"jobs": [{"status": "running"}]}}]:
        client.request.return_value = status
        with pytest.raises(ValueError, match="active or unverified"):
            service.stop(force_shared=True)


@pytest.mark.parametrize("cancel_during_readiness", [False, True])
def test_preparation_uses_only_fixed_media_extras_and_preserves_shared_child(tmp_path, monkeypatch, cancel_during_readiness):
    ctl = controller(tmp_path)
    service = CanonicalService(ctl)
    identity = ctl.store.enqueue("canonical_media_prepare", {})
    source = tmp_path / "source"
    source.mkdir()
    ensure = Mock(return_value={"source": str(source), "python": "/fixed/python"})
    monkeypatch.setattr("mlx2.desktop_runner.RunnerManager", Mock(return_value=SimpleNamespace(ensure=ensure)))
    child = Mock(pid=12345)
    child.poll.return_value = None
    popen = Mock(return_value=child)
    monkeypatch.setattr("mlx2.desktop_media_canonical_service.subprocess.Popen", popen)
    monkeypatch.setattr("psutil.Process", Mock(return_value=SimpleNamespace(create_time=lambda: 1.0)))
    client = Mock()
    def status(*args):
        if cancel_during_readiness:
            service.cancel.set()
        return {"service": "mlx2-media"}
    client.request.side_effect = status
    monkeypatch.setattr("mlx2.desktop_media_canonical_service.CanonicalMediaClient", Mock(return_value=client))
    stop = Mock()
    monkeypatch.setattr("mlx2.desktop.stop_group", stop)
    service.run(identity, source)
    assert ensure.call_args.kwargs["extras_override"] == "media,observability"
    assert ensure.call_args.kwargs["purpose"] == "canonical-media"
    assert popen.call_args.args[0] == ["/fixed/python", "-m", "mlx2.media.server", "--host", "127.0.0.1", "--port", "8777"]
    assert ctl.store.jobs(1)[0]["state"] == ("cancelled" if cancel_during_readiness else "completed")
    service.close()
    if cancel_during_readiness:
        stop.assert_called_once_with(child)
        assert not (service.home / "process.json").exists()
    else:
        stop.assert_not_called()


def test_canonical_controller_refuses_restart_after_close_without_contacting_service(tmp_path, monkeypatch):
    service = CanonicalService(controller(tmp_path))
    service.close()
    client = Mock()
    monkeypatch.setattr("mlx2.desktop_media_canonical_service.CanonicalMediaClient", client)
    with pytest.raises(ValueError, match="closing"):
        service.start()
    client.assert_not_called()
