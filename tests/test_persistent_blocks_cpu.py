"""Persistent scratch file integrity and cleanup without a device runtime."""

import json
from pathlib import Path

import pytest

from mlx2.runtime import persistent_blocks


def snapshot(tmp_path):
    path = tmp_path / "snapshot.safetensors"
    raw = b"0123456789abcdef"
    path.write_bytes(raw)
    persistent_blocks.encode_block_file(path, block_bytes=4, signature="identity")
    manifest = json.loads(path.read_text())
    directory = path.with_suffix(path.suffix + ".blocks")
    return path, raw, manifest, directory


def test_block_snapshot_round_trip_and_deletion(tmp_path):
    path, raw, _manifest, _directory = snapshot(tmp_path)
    with persistent_blocks.materialize_block_file(path, expected_signature="identity") as restored:
        assert restored.read_bytes() == raw
    assert not restored.exists()
    assert persistent_blocks.remove_block_file(path) > len(raw)
    assert not list(tmp_path.iterdir())


def test_symlinked_block_directory_cannot_authorize_external_deletion(tmp_path):
    path, _raw, manifest, directory = snapshot(tmp_path)
    outside = tmp_path / "other-owner"
    directory.rename(outside)
    directory.symlink_to(outside, target_is_directory=True)
    original = outside / manifest["blocks"][0]["name"]
    with pytest.raises(ValueError, match="symlink"):
        persistent_blocks.block_file_paths(path)
    persistent_blocks.remove_block_file(path)
    assert original.is_file()


def test_symlinked_block_is_rejected_even_within_its_directory(tmp_path):
    path, _raw, manifest, directory = snapshot(tmp_path)
    block = directory / manifest["blocks"][0]["name"]
    replacement = directory / "another-owner.block"
    block.rename(replacement)
    block.symlink_to(replacement)
    with pytest.raises(ValueError, match="symlink"):
        persistent_blocks.block_file_paths(path)
    persistent_blocks.remove_block_file(path)
    assert replacement.is_file()


def test_failed_manifest_publication_cleans_blocks_and_can_retry(tmp_path, monkeypatch):
    path = tmp_path / "snapshot.safetensors"
    raw = b"0123456789abcdef"
    path.write_bytes(raw)
    original_replace = persistent_blocks.os.replace

    def fail_manifest(source, destination):
        if Path(destination) == path:
            raise OSError("injected manifest publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(persistent_blocks.os, "replace", fail_manifest)
    with pytest.raises(OSError, match="publication failure"):
        persistent_blocks.encode_block_file(path, block_bytes=4, signature="identity")
    assert path.read_bytes() == raw
    assert list(tmp_path.iterdir()) == [path]
    monkeypatch.setattr(persistent_blocks.os, "replace", original_replace)
    persistent_blocks.encode_block_file(path, block_bytes=4, signature="identity")
    with persistent_blocks.materialize_block_file(path, expected_signature="identity") as restored:
        assert restored.read_bytes() == raw
