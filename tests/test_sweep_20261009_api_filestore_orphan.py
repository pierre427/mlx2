"""A durable ``FileStore.create`` that publishes the content
``.bin`` and then fails to publish the metadata ``.json`` leaves the content
on disk, untracked (``files: 0, bytes: 0``), unreachable through the API and
unreclaimed on restart because ``_restore`` only discovers ``*/*.json``.

Style follows tests/test_api_resource_restore.py (``resources`` alias,
``tmp_path``, ``monkeypatch``).
"""

from pathlib import Path

import pytest

import mlx2.api_resources as resources
from mlx2.api_resources import FileStore


def _upload(store, size=900):
    return store.create(
        "tenant-a",
        filename="upload.txt",
        purpose="user_data",
        content_type="text/plain",
        content=b"x" * size,
    )


def test_failed_metadata_publication_rolls_back_the_content(tmp_path, monkeypatch):
    store = FileStore(root=tmp_path, max_bytes=1024, max_file_bytes=1024)
    write = resources._atomic_write

    def fail_metadata(path, data):
        if Path(path).suffix == ".json":
            raise OSError(28, "no space left on device")
        return write(path, data)

    monkeypatch.setattr(resources, "_atomic_write", fail_metadata)
    with pytest.raises(OSError):
        _upload(store)
    monkeypatch.setattr(resources, "_atomic_write", write)

    assert store.status()["files"] == 0 and store.status()["bytes"] == 0
    # The failed upload must not leave its payload behind.
    assert list(tmp_path.glob("*/*.bin")) == []
    assert list(tmp_path.glob("*/*.json")) == []
    # ...and the store is still usable for the next upload.
    created = _upload(store)
    assert store.status()["files"] == 1
    restored = FileStore(root=tmp_path, max_bytes=1024, max_file_bytes=1024)
    assert restored.status()["files"] == 1
    assert restored.get("tenant-a", created["id"])["id"] == created["id"]


def test_restart_reclaims_content_that_has_no_metadata(tmp_path):
    # The crash window between the two publications has the same shape: a
    # ``.bin`` with no sibling ``.json`` is unreachable and must not survive.
    store = FileStore(root=tmp_path, max_bytes=1024, max_file_bytes=1024)
    created = _upload(store, 300)
    metadata_path, content_path = store._paths("tenant-a", created["id"])
    orphan = content_path.with_name("file-orphan0000000000000000000000000.bin")
    orphan.write_bytes(b"o" * 500)

    restored = FileStore(root=tmp_path, max_bytes=1024, max_file_bytes=1024)
    assert restored.status()["files"] == 1
    assert restored.status()["bytes"] == 300
    assert content_path.exists() and metadata_path.exists()
    assert not orphan.exists(), "orphan content survived restart"


def test_restart_leaves_symlinked_and_metadata_backed_content_alone(tmp_path):
    store = FileStore(root=tmp_path, max_bytes=1024, max_file_bytes=1024)
    created = _upload(store, 300)
    metadata_path, content_path = store._paths("tenant-a", created["id"])
    # Bad metadata keeps both paths (the restore tests' rule): not an orphan.
    metadata_path.write_text("{not json")
    # A symlinked payload is never followed, so it is never unlinked either.
    external = tmp_path / "external.bin"
    external.write_bytes(b"e" * 10)
    link = content_path.with_name("file-link00000000000000000000000000.bin")
    link.symlink_to(external)

    restored = FileStore(root=tmp_path, max_bytes=1024, max_file_bytes=1024)
    assert restored.status()["files"] == 0
    assert content_path.exists() and metadata_path.exists()
    assert link.is_symlink() and external.exists()
    assert restored.status()["counts"].get("orphans_reclaimed", 0) == 0


def test_restart_keeps_content_behind_a_dangling_metadata_symlink(tmp_path):
    # Review round 1: ``exists()`` follows symlinks, so a dangling ``.json``
    # symlink looked like "no metadata" and its ``.bin`` was swept.  The
    # restore reader rejects and keeps such a pair; so must the sweep.
    store = FileStore(root=tmp_path, max_bytes=1024, max_file_bytes=1024)
    created = _upload(store, 300)
    metadata_path, content_path = store._paths("tenant-a", created["id"])
    metadata_path.unlink()
    metadata_path.symlink_to(tmp_path / "gone.json")
    assert metadata_path.is_symlink() and not metadata_path.exists()

    restored = FileStore(root=tmp_path, max_bytes=1024, max_file_bytes=1024)
    assert restored.status()["files"] == 0
    assert restored.status()["counts"]["restore_failures"] == 1
    assert content_path.exists() and metadata_path.is_symlink()
    assert restored.status()["counts"].get("orphans_reclaimed", 0) == 0
