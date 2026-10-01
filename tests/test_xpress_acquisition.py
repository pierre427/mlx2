"""Pinned-byte acquisition reconciliation; tiny mocked network, no MLX imports."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


@pytest.fixture
def acquisition(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/acquire_xpress.py"
    spec = importlib.util.spec_from_file_location("tiny_xpress_acquisition", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    blobs = {"config.json": b'{"tiny":1}', "model.safetensors": b"abcdefgh"}
    manifest = {
        name: {"size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
        for name, blob in blobs.items()
    }
    manifest["config.json"]["git_blob"] = hashlib.sha1(
        f"blob {len(blobs['config.json'])}\0".encode() + blobs["config.json"]
    ).hexdigest()
    monkeypatch.setattr(module, "FILES", manifest)
    return module, blobs


def _existing(destination, blobs):
    for name, blob in blobs.items():
        (destination / name).write_bytes(blob)


class Response(io.BytesIO):
    def __init__(self, data, status=200, content_range=None):
        super().__init__(data)
        self.status = status
        self.headers = {} if content_range is None else {"Content-Range": content_range}


def test_existing_verified_files_reconcile_without_network_or_tensor_loads(
    acquisition, tmp_path, monkeypatch
):
    module, blobs = acquisition
    _existing(tmp_path, blobs)
    (tmp_path / ".mlx2-acquisition.json.partial").write_text("interrupted receipt")

    def forbidden(*a, **k):
        raise AssertionError("existing verified artifact should not download")

    monkeypatch.setattr(module.urllib.request, "urlopen", forbidden)
    receipt = module.acquire(tmp_path)
    assert receipt["hash_verified"] and receipt["partials"] == 0
    assert not receipt["model_loaded"] and not receipt["qualified"]
    assert receipt["revision"] == module.REVISION
    assert not list(tmp_path.glob("*.partial"))
    assert json.loads((tmp_path / ".mlx2-acquisition.json").read_text()) == receipt


@pytest.mark.parametrize("corrupt", ["size", "sha256", "git_blob"])
def test_corrupt_existing_file_invalidates_previous_completion_receipt(
    acquisition, tmp_path, corrupt
):
    module, blobs = acquisition
    _existing(tmp_path, blobs)
    receipt = tmp_path / ".mlx2-acquisition.json"
    receipt.write_text('{"hash_verified":true}')
    file = tmp_path / "config.json"
    if corrupt == "size":
        file.write_bytes(b"x")
    elif corrupt == "sha256":
        file.write_bytes(b"x" * len(blobs["config.json"]))
    else:
        module.FILES["config.json"]["git_blob"] = "0" * 40
    with pytest.raises(
        ValueError,
        match={"size": "Size", "sha256": "SHA256", "git_blob": "Git blob"}[corrupt],
    ):
        module.acquire(tmp_path)
    assert not receipt.exists()


def test_resume_is_revision_pinned_and_hash_verified(
    acquisition, tmp_path, monkeypatch
):
    module, blobs = acquisition
    (tmp_path / "config.json").write_bytes(blobs["config.json"])
    (tmp_path / "model.safetensors.partial").write_bytes(blobs["model.safetensors"][:3])
    requests = []

    def network(request, **kwargs):
        requests.append(request)
        return Response(blobs["model.safetensors"][3:], 206, "bytes 3-7/8")

    monkeypatch.setattr(module.urllib.request, "urlopen", network)
    module.acquire(tmp_path)
    assert len(requests) == 1
    assert f"/resolve/{module.REVISION}/model.safetensors" in requests[0].full_url
    assert requests[0].get_header("Range") == "bytes=3-"
    assert (tmp_path / "model.safetensors").read_bytes() == blobs["model.safetensors"]
    assert not (tmp_path / "model.safetensors.partial").exists()


@pytest.mark.parametrize("status,content_range", [(200, None), (206, "bytes 0-7/8")])
def test_bad_resume_responses_never_publish_completion(
    acquisition, tmp_path, monkeypatch, status, content_range
):
    module, blobs = acquisition
    (tmp_path / "config.json").write_bytes(blobs["config.json"])
    partial = tmp_path / "model.safetensors.partial"
    partial.write_bytes(b"abc")
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *a, **k: Response(blobs["model.safetensors"][3:], status, content_range),
    )
    with pytest.raises(ValueError, match="range"):
        module.acquire(tmp_path)
    assert partial.read_bytes() == b"abc"
    assert not (tmp_path / ".mlx2-acquisition.json").exists()


@pytest.mark.parametrize(
    "payload,error", [(b"abcdefghi", "exceeded pinned size"), (b"zzzzzzzz", "SHA256")]
)
def test_oversized_or_wrong_download_never_becomes_final(
    acquisition, tmp_path, monkeypatch, payload, error
):
    module, blobs = acquisition
    (tmp_path / "config.json").write_bytes(blobs["config.json"])
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *a, **k: Response(payload)
    )
    with pytest.raises(ValueError, match=error):
        module.acquire(tmp_path)
    assert not (tmp_path / "model.safetensors").exists()
    assert not (tmp_path / ".mlx2-acquisition.json").exists()


def test_unknown_partial_blocks_receipt_without_erasing_partial(acquisition, tmp_path):
    module, blobs = acquisition
    _existing(tmp_path, blobs)
    unknown = tmp_path / "another-model.partial"
    unknown.write_bytes(b"preserve")
    with pytest.raises(ValueError, match="Unreconciled partial"):
        module.acquire(tmp_path)
    assert unknown.read_bytes() == b"preserve"
    assert not (tmp_path / ".mlx2-acquisition.json").exists()
