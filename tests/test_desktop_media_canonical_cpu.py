from __future__ import annotations

from unittest.mock import Mock

import pytest

from mlx2.desktop_media_canonical import CanonicalMediaClient, canonical_profile


def catalog():
    return {"object": "list", "data": [{"route_id": "speech.indextts25", "label": "IndexTTS 2.5",
        "operation": "tts", "capability": "audio.speak", "implemented": True, "ready": True,
        "inputs": [{"role": "speaker_reference", "kind": "audio"}],
        "params": [{"name": "text", "type": "text"}],
        "artifact": {"kind": "indextts25", "available": True},
        "qualification": {"qualified": False, "selected": False, "observed_used": False}}]}


def fixture_client(capability="voice"):
    client = CanonicalMediaClient(canonical_profile(capability))
    calls = []
    def request(method, path, body=None):
        calls.append((method, path, body))
        if path == "/v1/media/status":
            return {"service": "mlx2-media"}
        if path == "/v1/media/routes":
            return catalog()
        if path == "/v1/media/jobs" and method == "POST":
            return {"job": {"id": "media_one", "status": "queued"}}
        return {"id": "media_one", "status": "completed"}
    client.request = request
    return client, calls


def test_canonical_voice_uses_real_durable_job_contract_and_reference_not_training():
    client, calls = fixture_client()
    tools = {row["name"] for row in client.list_tools()}
    assert "generate_voice" in tools
    model = client.models()["models"][0]
    assert model["reference_conditioning"] is True
    assert model["training"] is False
    assert model["qualification"]["qualified"] is False
    reply = client.call_tool("generate_voice", {"model": "speech.indextts25", "prompt": "Hello", "speaker_reference": "/Users/Shared/reference.wav"})
    assert reply["structuredContent"]["id"] == "media_one"
    submitted = calls[-1][2]
    assert submitted["route"] == "speech.indextts25"
    assert submitted["operation"] == "tts"
    assert submitted["params"] == {"text": "Hello"}
    assert submitted["inputs"] == [{"role": "speaker_reference", "path": "/Users/Shared/reference.wav"}]
    assert submitted["client"]["reference_conditioning_is_training"] is False
    assert submitted["idempotency_key"]


def test_training_is_unavailable_without_actual_advertised_training_route():
    client, _ = fixture_client("voice_training")
    assert client.models()["models"] == []
    assert "generate_voice_training" not in {row["name"] for row in client.list_tools()}
    with pytest.raises(ValueError, match="unavailable"):
        client.call_tool("generate_voice_training", {"prompt": "train"})


def test_canonical_client_is_loopback_only_and_never_follows_userinfo_redirects():
    for url in ["http://evil.example:8777", "http://127.0.0.1:80@evil.example", "https://127.0.0.1:8777", "http://127.0.0.1:8777/?key=x"]:
        with pytest.raises(ValueError, match="loopback"):
            CanonicalMediaClient({**canonical_profile("voice"), "endpoint": url})


def test_unrelated_endpoint_is_not_treated_as_mlx2_media():
    client = CanonicalMediaClient(canonical_profile("voice"))
    client.request = Mock(return_value={"service": "unrelated"})
    with pytest.raises(ValueError, match="canonical"):
        client.models()


def test_poll_and_cancel_retain_the_exact_submitted_remote_id():
    client, calls = fixture_client()
    client.call_tool("get_voice_job", {"job_id": "media_one"})
    assert calls[-1][:2] == ("GET", "/v1/media/jobs/media_one")
    client.call_tool("cancel_voice_job", {"job_id": "media_one"})
    assert calls[-1][:2] == ("POST", "/v1/media/jobs/media_one/cancel")


def test_voice_reference_preparation_is_bounded_local_preprocessing_with_hashes(tmp_path, monkeypatch):
    import json
    import wave
    from pathlib import Path

    from mlx2.desktop_media_canonical import prepare_reference
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"selected video")
    original_path = Path
    monkeypatch.setattr("mlx2.desktop_media_canonical.Path", lambda value: tmp_path / "staged" if str(value).startswith("/Users/Shared/") else original_path(value))
    monkeypatch.setattr("shutil.which", lambda name: "/fixed/ffmpeg")
    def fake_run(args, **kwargs):
        assert args[args.index("-protocol_whitelist") + 1] == "file,pipe"
        assert args[args.index("-t") + 1] == "15"
        with wave.open(args[-1], "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(22050)
            stream.writeframes(b"\0\0" * 22050)
    monkeypatch.setattr("mlx2.desktop_media_canonical.subprocess.run", fake_run)
    result = prepare_reference(None, {"path": str(source)})
    assert result["duration_seconds"] == 1
    assert result["reference_conditioning"] is True and result["training"] is False
    assert len(result["source_sha256"]) == len(result["sha256"]) == 64
    assert json.loads(original_path(result["path"]).with_suffix(".receipt.json").read_text()) == result


def test_canonical_job_catalog_normalizes_real_data_for_shared_idle_checks():
    client, _ = fixture_client()
    client.request = Mock(return_value={"object": "list", "data": [{"id": "media_other", "status": "running"}]})
    reply = client.call_tool("list_voice_jobs", {"limit": 100})["structuredContent"]
    assert reply["jobs"] == [{"id": "media_other", "status": "running"}]
    client.request.return_value = {"object": "list", "data": None}
    with pytest.raises(ValueError, match="invalid job catalog"):
        client.call_tool("list_voice_jobs", {})
