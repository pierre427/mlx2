"""CPU-only integrity checks for the historical source restoration tool."""

import hashlib
import json
import subprocess

import pytest

from scripts.research.uzu.restore_snapshot import restore


def checkout(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    subprocess.run(["git", "init", "-q", str(target)], check=True)
    (target / "base").write_text("unchanged\n")
    subprocess.run(["git", "-C", str(target), "add", "base"], check=True)
    subprocess.run(["git", "-C", str(target), "-c", "user.name=Fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    return target, revision


def snapshot(tmp_path, revision, digest=None):
    source = tmp_path / "snapshot"
    file = source / "source/src/example.py"
    file.parent.mkdir(parents=True)
    data = b"historical bytes\n"
    file.write_bytes(data)
    (source / "result.json").write_text(json.dumps({"source_revision": revision,
        "source_files": {"src/example.py": digest or hashlib.sha256(data).hexdigest()}}))
    return source


def test_restoration_refuses_bad_hash_before_writing(tmp_path):
    target, revision = checkout(tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        restore(snapshot(tmp_path, revision, "0" * 64), target)
    assert not (target / "src").exists()
    assert (target / "base").read_text() == "unchanged\n"


def test_restoration_refuses_dirty_destination(tmp_path):
    target, revision = checkout(tmp_path)
    (target / "base").write_text("unrelated edits\n")
    with pytest.raises(ValueError, match="clean"):
        restore(snapshot(tmp_path, revision), target)
    assert (target / "base").read_text() == "unrelated edits\n"


def test_restoration_refuses_wrong_base(tmp_path):
    target, _ = checkout(tmp_path)
    with pytest.raises(ValueError, match="experiment base"):
        restore(snapshot(tmp_path, "0" * 40), target)
    assert not (target / "src").exists()


def test_restoration_preserves_exact_source_bytes(tmp_path):
    target, revision = checkout(tmp_path)
    receipt = restore(snapshot(tmp_path, revision), target)
    assert receipt["source_revision"] == revision
    assert receipt["gpu_execution"] is False
    assert (target / "src/example.py").read_bytes() == b"historical bytes\n"
    assert (target / "scripts/probe_uzu_checkpoint.py").is_file()
