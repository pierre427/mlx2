"""CPU HTTP checks for restart-safe, content-pinned Ripple acquisition."""
import hashlib
import importlib.util
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "ripple_download", Path(__file__).parents[1] / "scripts/download_ripple.py")
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


@pytest.mark.parametrize("ignore_range", [False, True])
def test_adopt_largest_hub_partial_and_resume_http(tmp_path, ignore_range):
    payload = bytes(range(256)) * 256
    digest = hashlib.sha256(payload).hexdigest()
    entry = {"path": "model.safetensors", "sha256": digest, "git_blob": None, "size": len(payload)}
    cache = tmp_path / ".cache/huggingface/download"
    cache.mkdir(parents=True)
    (cache / f"old.{digest}.first.incomplete").write_bytes(payload[:12345])
    (cache / f"new.{digest}.second.incomplete").write_bytes(payload[:2345])
    ranges = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            requested = self.headers.get("Range")
            ranges.append(requested)
            offset = 0 if ignore_range else int(requested.removeprefix("bytes=").removesuffix("-"))
            self.send_response(200 if ignore_range else 206)
            self.send_header("Content-Length", str(len(payload) - offset))
            if not ignore_range:
                self.send_header("Content-Range", f"bytes {offset}-{len(payload)-1}/{len(payload)}")
            self.end_headers()
            self.wfile.write(payload[offset:])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        download.resume_file(entry, tmp_path, url=f"http://127.0.0.1:{server.server_port}/weight")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert ranges == ["bytes=12345-"]
    assert (tmp_path / entry["path"]).read_bytes() == payload
    assert not list(cache.glob("*.incomplete"))


def test_failure_preserves_stable_partial(tmp_path, monkeypatch):
    from huggingface_hub import file_download

    entry = {"path": "weight", "sha256": "content-hash", "git_blob": None, "size": 20}

    def interrupted(_url, stream, **kwargs):
        assert kwargs["resume_size"] == 0
        stream.write(b"prefix")
        raise OSError("interrupted transfer")

    monkeypatch.setattr(file_download, "http_get", interrupted)
    with pytest.raises(OSError, match="interrupted transfer"):
        download.resume_file(entry, tmp_path, url="https://example.invalid/weight")
    assert (tmp_path / ".cache/mlx2-ripple-download/content-hash.incomplete").read_bytes() == b"prefix"
    assert not (tmp_path / "weight").exists()


def test_oversized_partial_refused_before_network(tmp_path):
    cache = tmp_path / ".cache/mlx2-ripple-download"
    cache.mkdir(parents=True)
    (cache / "hash.incomplete").write_bytes(b"too long")
    entry = {"path": "weight", "sha256": "hash", "git_blob": None, "size": 2}
    with pytest.raises(ValueError, match="oversized"):
        download.resume_file(entry, tmp_path, url="https://example.invalid/weight")


def test_full_size_corrupt_file_is_quarantined_then_refetched(tmp_path, monkeypatch):
    """A right-size, wrong-hash file used to fail every rerun with no refetch."""
    if not (Path(__file__).resolve().parents[1] / "provenance/ripple-65b-snapshot.json").is_file():
        pytest.skip("private snapshot manifest not present (public mirror)")
    import huggingface_hub
    from huggingface_hub import file_download

    payload = b"A" * 100
    entry = {"path": "weight", "sha256": hashlib.sha256(payload).hexdigest(),
             "git_blob": None, "size": len(payload)}
    manifest = {"repo": "r/m", "revision": "rev", "files": [entry]}
    real_loads = download.json.loads
    monkeypatch.setattr(download.json, "loads",
                        lambda text: manifest if "ricursiveAI" in text else real_loads(text))
    token = tmp_path / "token"
    token.write_text("secret\n")
    output = tmp_path / "out"
    output.mkdir()
    (output / "weight").write_bytes(b"B" * 100)

    class Metadata:
        commit_hash, size, etag = "rev", len(payload), entry["sha256"]
        location = "https://cdn.example.invalid/weight"

    calls = []
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata",
                        lambda url, token: calls.append(url) or Metadata)

    def fetch(_url, stream, **kwargs):
        assert kwargs["headers"] is None  # PAT never sent to a foreign origin
        stream.write(payload[kwargs["resume_size"]:])

    monkeypatch.setattr(file_download, "http_get", fetch)
    with pytest.raises(ValueError, match="content hash differs"):
        download.download(output, token_file=token)
    assert not (output / "weight").exists()
    assert (output / ".cache/mlx2-ripple-download/rejected/weight").read_bytes() == b"B" * 100
    assert not calls
    receipt = download.download(output, token_file=token)
    assert len(calls) == 1
    assert (output / "weight").read_bytes() == payload
    assert receipt["verified"] == [{"path": "weight", "bytes": 100, "content_hash": entry["sha256"]}]


def test_legacy_hub_partial_name_is_adopted(tmp_path, monkeypatch):
    from huggingface_hub import file_download

    payload = b"0123456789" * 10
    digest = hashlib.sha256(payload).hexdigest()
    entry = {"path": "weight", "sha256": digest, "git_blob": None, "size": len(payload)}
    legacy = tmp_path / ".cache/huggingface/download/sub"
    legacy.mkdir(parents=True)
    (legacy / f"weight.{digest}.incomplete").write_bytes(payload[:40])

    def fetch(_url, stream, **kwargs):
        assert kwargs["resume_size"] == 40
        stream.write(payload[40:])

    monkeypatch.setattr(file_download, "http_get", fetch)
    download.resume_file(entry, tmp_path, url="https://example.invalid/weight")
    assert (tmp_path / "weight").read_bytes() == payload
    assert not list(legacy.glob("*.incomplete"))
