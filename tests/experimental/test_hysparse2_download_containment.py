"""Pinned download namespace cannot follow links outside its explicit root."""

import hashlib
import io
import json

import pytest

from mlx2.experimental.hysparse2.download import fetch_file


@pytest.mark.parametrize("level", ["repo", "revision", "nested", "file"])
def test_symlink_escape_refused_before_writes_or_network(tmp_path, level):
    root, foreign = tmp_path / "downloads", tmp_path / "foreign"
    root.mkdir()
    foreign.mkdir()
    revision = "a" * 40
    repo = root / "owner--dataset"
    destination = repo / revision / "nested/file.bin"
    selected = {
        "repo": repo,
        "revision": repo / revision,
        "nested": destination.parent,
        "file": destination,
    }[level]
    selected.parent.mkdir(parents=True, exist_ok=True)
    if level == "file":
        foreign_target = foreign / "file.bin"
        foreign_target.write_bytes(b"x")
    else:
        foreign_target = foreign
    selected.symlink_to(foreign_target, target_is_directory=level != "file")
    before = sorted(
        (str(p.relative_to(foreign)), p.read_bytes() if p.is_file() else None)
        for p in foreign.rglob("*")
    )
    item = {
        "path": "nested/file.bin",
        "size": 1,
        "sha256": hashlib.sha256(b"x").hexdigest(),
    }
    with pytest.raises(ValueError, match="escapes output root"):
        fetch_file(
            "owner/dataset",
            revision,
            item,
            root,
            open_url=lambda *a, **k: pytest.fail("escape reached network"),
        )
    assert before == sorted(
        (str(p.relative_to(foreign)), p.read_bytes() if p.is_file() else None)
        for p in foreign.rglob("*")
    )
    assert not list(root.rglob("*.partial"))


def test_valid_pinned_download_and_reuse_remain_inside_root(tmp_path):
    data = b"bounded corpus"
    item = {
        "path": "nested/file.bin",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    result = fetch_file(
        "owner/dataset",
        "a" * 40,
        item,
        tmp_path,
        open_url=lambda *a, **k: io.BytesIO(data),
    )
    from pathlib import Path

    target = Path(result["path"])
    assert target.is_relative_to(tmp_path.resolve())
    assert target.read_bytes() == data
    reused = fetch_file(
        "owner/dataset",
        "a" * 40,
        item,
        tmp_path,
        open_url=lambda *a, **k: pytest.fail("reuse reached network"),
    )
    assert reused["status"] == "verified-existing"


def test_receipt_escape_refused_before_any_download(tmp_path, monkeypatch):
    from mlx2.experimental.hysparse2 import download

    root = tmp_path / "downloads"
    directory = root / "owner--dataset" / ("a" * 40)
    directory.mkdir(parents=True)
    foreign = tmp_path / "foreign.json"
    foreign.write_text("preserve")
    (directory / "receipt.json").symlink_to(foreign)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "id": "owner/dataset",
                        "revision": "a" * 40,
                        "files": [
                            {
                                "path": "data.bin",
                                "size": 1,
                                "sha256": hashlib.sha256(b"x").hexdigest(),
                            }
                        ],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        download,
        "fetch_file",
        lambda *a, **k: pytest.fail("receipt escape reached download"),
    )
    with pytest.raises(ValueError, match="escapes output root"):
        download.main(["--manifest", str(manifest), "--output", str(root)])
    assert foreign.read_text() == "preserve"
    assert not (directory / "data.bin").exists()


def test_receipt_destination_rechecked_after_download(tmp_path, monkeypatch):
    from mlx2.experimental.hysparse2 import download

    root = tmp_path / "downloads"
    directory = root / "owner--dataset" / ("a" * 40)
    directory.mkdir(parents=True)
    foreign = tmp_path / "foreign.json"
    foreign.write_text("preserve")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"sources": [{"id": "owner/dataset", "revision": "a" * 40, "files": [{}]}]}
        )
    )

    def changed(*args, **kwargs):
        (directory / "receipt.json").symlink_to(foreign)
        return {"status": "synthetic-test"}

    monkeypatch.setattr(download, "fetch_file", changed)
    with pytest.raises(ValueError, match="escapes output root"):
        download.main(["--manifest", str(manifest), "--output", str(root)])
    assert foreign.read_text() == "preserve"
