"""CPU-only byte-binding checks for live price identity."""

import hashlib
import json
import sys
import types
import zipfile

import pytest

from mlx2.runtime import paged_price_identity as identity


def digest(data):
    return hashlib.sha256(data).hexdigest()


def test_artifact_identity_binds_exact_loaded_root_and_every_byte(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    files = {"config.json": b"{}", "model.safetensors": b"weights",
             "tokenizer.json": b"{}"}
    for name, value in files.items():
        (root / name).write_bytes(value)
    manifest = tmp_path / "artifact.json"
    manifest.write_text(json.dumps({"root": str(root),
                                    "files": {name: digest(value)
                                              for name, value in files.items()}}))
    assert identity._artifact_sha256(manifest, root) == identity._sha256(manifest)
    with pytest.raises(RuntimeError, match="loaded adapter"):
        identity._artifact_sha256(manifest, tmp_path)
    (root / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="bytes differ"):
        identity._artifact_sha256(manifest, root)


def test_wheel_must_contain_loaded_core_and_library_bytes(tmp_path, monkeypatch):
    pkg = tmp_path / "installed/mlx"
    (pkg / "lib").mkdir(parents=True)
    core = pkg / "core.fake.so"
    library = pkg / "lib/libmlx.dylib"
    core.write_bytes(b"core")
    library.write_bytes(b"library")
    package = types.ModuleType("mlx"); package.__path__ = []
    mx = types.ModuleType("mlx.core"); mx.__file__ = str(core); mx.__version__ = "test"
    package.core = mx
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    wheel = tmp_path / "mlx.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("mlx/core.fake.so", b"core")
        archive.writestr("mlx/lib/libmlx.dylib", b"library")
    assert identity._wheel_identity(wheel) == ("test", identity._sha256(wheel))
    core.write_bytes(b"different")
    with pytest.raises(RuntimeError, match="wheel differs"):
        identity._wheel_identity(wheel)


def test_wrong_loaded_native_module_fails_before_source_scan(tmp_path, monkeypatch):
    expected = tmp_path / "expected.so"
    actual = tmp_path / "actual.so"
    expected.write_bytes(b"expected")
    actual.write_bytes(b"actual")
    fake = types.ModuleType("_paged_kv_native")
    fake.__file__ = str(actual)
    monkeypatch.setitem(sys.modules, "_paged_kv_native", fake)
    with pytest.raises(RuntimeError, match="loaded native"):
        identity.compute_live_price_identity(
            tmp_path / "artifact.json", tmp_path / "mlx.whl", expected,
            adapter_artifact_root=tmp_path)


def test_process_identity_memo_refuses_mutable_file_and_source_drift(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    artifact = tmp_path / "artifact.json"
    weight = model / "model.safetensors"
    wheel = tmp_path / "mlx.whl"
    kernel = tmp_path / "native.so"
    for path in (artifact, weight, wheel, kernel):
        path.write_bytes(b"initial")
    paths = (artifact, weight, wheel, kernel)
    calls = []
    source_clean = [True]
    monkeypatch.setattr(identity.subprocess, "check_output", lambda *_, **__: "source-a\n")
    monkeypatch.setattr(identity, "_watched_paths", lambda *_: paths)
    monkeypatch.setattr(identity, "_source_unchanged", lambda _: source_clean[0])
    monkeypatch.setattr(identity, "compute_live_price_identity",
                        lambda *_, **__: (calls.append(1) or
                                          {"source_commit": "source-a", "host": "test"}))

    def attest():
        return identity.cached_live_price_identity(
            artifact, wheel, kernel, adapter_artifact_root=model)

    first = attest()
    first["host"] = "mutated caller copy"
    assert attest()["host"] == "test" and calls == [1]
    weight.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="changed after attestation"):
        attest()
    weight.write_bytes(b"initial")
    source_clean[0] = False
    with pytest.raises(RuntimeError, match="changed after attestation"):
        attest()
