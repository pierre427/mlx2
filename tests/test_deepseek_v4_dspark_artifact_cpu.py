"""Stat-bound hash sealing; tiny fixture avoids touching actual model weights."""

import hashlib
import json

import pytest

from mlx2.adapters import deepseek_v4_dspark_artifact as artifact


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    payload = b"exact sidecar test fixture"
    path = tmp_path / artifact.FILENAME
    path.write_bytes(payload)
    monkeypatch.setattr(artifact, "SIZE", len(payload))
    monkeypatch.setattr(artifact, "SIDECAR_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        artifact,
        "_metadata",
        lambda _: ({"stages": 3, "target_layers": [40, 41, 42]}, 81),
    )
    return path


def test_complete_actual_bytes_sealed_then_inspected(sidecar):
    with pytest.raises(FileNotFoundError):
        artifact.inspect_sidecar(sidecar)
    result = artifact.prepare_sidecar(sidecar)
    assert result["verification"] == "actual_full_file_stream_sha256_before_after_stat"
    assert result["tensor_count"] == 81
    assert artifact.inspect_sidecar(sidecar) == result


def test_hash_is_actual_bytes_not_download_receipt(sidecar):
    sidecar.write_bytes(b"x" * artifact.SIZE)
    with pytest.raises(ValueError, match="actual DSpark bytes"):
        artifact.prepare_sidecar(sidecar)
    assert not artifact.seal_path(sidecar).exists()


def test_changed_file_or_index_invalidates_seal(sidecar, monkeypatch):
    artifact.prepare_sidecar(sidecar)
    monkeypatch.setattr(artifact, "_metadata", lambda _: ({"stages": 4}, 81))
    with pytest.raises(ValueError, match="index/config"):
        artifact.inspect_sidecar(sidecar)
    sidecar.write_bytes(b"y" * artifact.SIZE)
    with pytest.raises(ValueError, match="stale"):
        artifact.inspect_sidecar(sidecar)


def test_changed_stat_during_hash_refuses_seal(sidecar, monkeypatch):
    real_stat = artifact.file_stat
    count = 0

    def changing(path):
        nonlocal count
        value = real_stat(path)
        count += 1
        if count > 1:
            value["mtime_ns"] += 1
        return value

    monkeypatch.setattr(artifact, "file_stat", changing)
    with pytest.raises(ValueError, match="during complete"):
        artifact.prepare_sidecar(sidecar)


def test_foreign_seal_cannot_authorize_payload(sidecar):
    artifact.prepare_sidecar(sidecar)
    path = artifact.seal_path(sidecar)
    seal = json.loads(path.read_text())
    seal["sha256"] = "0" * 64
    path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match="incompatible"):
        artifact.inspect_sidecar(sidecar)


def test_source_binding_checks_actual_bytes_and_descriptor(tmp_path, monkeypatch):
    from mlx2 import desktop_runner

    manifest = {"src/example.py": "b" * 64}
    snapshot = desktop_runner.digest(manifest)
    monkeypatch.setattr(desktop_runner, "source_manifest", lambda root: manifest)
    monkeypatch.setenv("MLX2_SOURCE_REVISION", "c" * 40)
    monkeypatch.setenv("MLX2_SOURCE_SNAPSHOT_SHA256", snapshot)
    monkeypatch.delenv("MLX2_SOURCE_MANIFEST", raising=False)
    descriptor = tmp_path / "desktop-source.json"
    descriptor.write_text(
        json.dumps(
            {"source_revision": "c" * 40, "source_sha256": snapshot, "files": manifest}
        )
    )
    assert (
        artifact.verified_source_identity(tmp_path)["source_snapshot_sha256"]
        == snapshot
    )
    monkeypatch.setenv("MLX2_SOURCE_SNAPSHOT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="actual|loaded"):
        artifact.verified_source_identity(tmp_path)


def test_source_environment_alone_is_not_authority(tmp_path, monkeypatch):
    from mlx2 import desktop_runner

    manifest = {"src/example.py": "b" * 64}
    snapshot = desktop_runner.digest(manifest)
    monkeypatch.setattr(desktop_runner, "source_manifest", lambda root: manifest)
    monkeypatch.setenv("MLX2_SOURCE_REVISION", "c" * 40)
    monkeypatch.setenv("MLX2_SOURCE_SNAPSHOT_SHA256", snapshot)
    monkeypatch.delenv("MLX2_SOURCE_MANIFEST", raising=False)
    (tmp_path / "desktop-source.json").write_text(
        json.dumps({"source_revision": "wrong", "source_sha256": snapshot})
    )
    with pytest.raises(ValueError, match="descriptor"):
        artifact.verified_source_identity(tmp_path)


def test_symlinked_seal_is_rejected_for_read_and_publication(sidecar, tmp_path):
    artifact.prepare_sidecar(sidecar)
    seal=artifact.seal_path(sidecar)
    real=tmp_path/'foreign-seal.json'
    seal.rename(real)
    before=real.read_bytes()
    seal.symlink_to(real)
    for operation in (artifact.inspect_sidecar,artifact.prepare_sidecar):
        with pytest.raises(ValueError,match='nonsymlink'):
            operation(sidecar)
    assert real.read_bytes()==before


def test_preexisting_temporary_symlink_is_never_followed(sidecar,tmp_path):
    victim=tmp_path/'unrelated.txt';victim.write_text('preserve')
    seal=artifact.seal_path(sidecar)
    old_temp=seal.with_suffix(seal.suffix+'.tmp')
    old_temp.symlink_to(victim)
    assert artifact.prepare_sidecar(sidecar)['tensor_count']==81
    assert victim.read_text()=='preserve'
    assert old_temp.is_symlink()
    assert list(tmp_path.glob(seal.name+'.*.tmp'))==[]
