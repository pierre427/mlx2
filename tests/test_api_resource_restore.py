"""CPU-only regression gates for bounded durable resource restore."""

import json
import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest

import mlx2.api_resources as resources
from mlx2.api_resources import FileStore, ResourceNotFound, ResponseStore


def context(text):
    return [{"role": "user", "content": text}, {"role": "assistant", "content": "answer"}]


def ordered_paths(root):
    paths = sorted(root.glob("*/*.json"))
    for index, path in enumerate(paths):
        os.utime(path, ns=(1000 + index, 1000 + index))
    return paths


def track_resident_high_water(monkeypatch):
    peaks = {"bytes": 0, "entries": 0}

    class TrackedEntries(OrderedDict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            size = sum(
                item[0] if isinstance(item, tuple) else len(item.content)
                for item in self.values()
            )
            peaks["bytes"] = max(peaks["bytes"], size)
            peaks["entries"] = max(peaks["entries"], len(self))

    monkeypatch.setattr(resources, "OrderedDict", TrackedEntries)
    return peaks


@pytest.mark.parametrize("entry_limit,byte_limit", [(2, 100_000), (100, 5000)])
def test_response_restore_evicts_during_admission(tmp_path, monkeypatch, entry_limit, byte_limit):
    writer = ResponseStore(root=tmp_path, max_entries=100, max_bytes=100_000)
    for index in range(24):
        writer.put("tenant", {"id": f"resp_{index:02}"}, context("x" * 2000))
    paths = ordered_paths(tmp_path)
    sizes = [path.stat().st_size for path in paths]
    expected = list(zip(paths, sizes))
    while len(expected) > entry_limit or sum(size for _, size in expected) > byte_limit:
        expected.pop(0)
    peaks = track_resident_high_water(monkeypatch)

    restored = ResponseStore(root=tmp_path, max_entries=entry_limit, max_bytes=byte_limit)

    assert peaks["bytes"] <= byte_limit + max(sizes)
    assert peaks["entries"] <= entry_limit + 1
    assert restored.status()["bytes"] == sum(size for _, size in expected)
    assert restored.status()["counts"]["evictions"] == len(paths) - len(expected)
    assert set(tmp_path.glob("*/*.json")) == {path for path, _ in expected}
    assert [key[1] for key in restored._entries] == [path.stem for path, _ in expected]


@pytest.mark.parametrize("entry_limit,byte_limit", [(2, 100_000), (100, 5000)])
def test_file_restore_evicts_during_admission(tmp_path, monkeypatch, entry_limit, byte_limit):
    writer = FileStore(root=tmp_path, max_files=100, max_bytes=100_000)
    for index in range(24):
        writer.create("tenant", filename=f"{index}.txt", purpose="user_data",
                      content_type="text/plain", content=b"x" * 2000)
    paths = ordered_paths(tmp_path)
    count = min(entry_limit, byte_limit // 2000)
    peaks = track_resident_high_water(monkeypatch)

    restored = FileStore(root=tmp_path, max_files=entry_limit, max_bytes=byte_limit)

    assert peaks["bytes"] <= byte_limit + 2000
    assert peaks["entries"] <= entry_limit + 1
    assert restored.status()["bytes"] == count * 2000
    assert [key[1] for key in restored._files] == [path.stem for path in paths[-count:]]
    assert set(tmp_path.glob("*/*.json")) == set(paths[-count:])
    assert set(tmp_path.glob("*/*.bin")) == {path.with_suffix(".bin") for path in paths[-count:]}


@pytest.mark.parametrize("change", [
    lambda record: [],
    lambda record: {**record, "payload": []},
    lambda record: {**record, "context_messages": {}},
    lambda record: {**record, "context_messages": [None]},
    lambda record: {**record, "context_messages": [{"role": 2}]},
    lambda record: {**record, "tenant_id": "other"},
    lambda record: {**record, "payload": {"id": "../victim"}},
    lambda record: {**record, "payload": {"id": "resp_other"}},
    lambda record: {**record, "payload": {"id": "resp_bad", "model": []}},
    lambda record: {**record, "payload": {"id": "resp_bad", "mlx2": []}},
    lambda record: {**record, "payload": {"id": "resp_bad", "mlx2": {"agent_compat": []}}},
])
def test_bad_response_records_are_skipped_without_deleting_paths(tmp_path, change):
    writer = ResponseStore(root=tmp_path)
    writer.put("tenant", {"id": "resp_bad"}, context("bad"))
    writer.put("tenant", {"id": "resp_good"}, context("good"))
    bad = next(tmp_path.glob("*/resp_bad.json"))
    bad.write_text(json.dumps(change(json.loads(bad.read_text()))))

    restored = ResponseStore(root=tmp_path)

    assert restored.status()["entries"] == 1
    assert restored.status()["counts"]["restore_failures"] == 1
    assert restored.context("tenant", "resp_good") == context("good")
    assert bad.exists()
    assert restored.status()["bytes"] == next(tmp_path.glob("*/resp_good.json")).stat().st_size


@pytest.mark.parametrize("field,value", [
    ("bytes", True), ("bytes", -1), ("bytes", 0), ("bytes", 100_000),
    ("created_at", "1"), ("created_at", True), ("created_at", -1),
    ("filename", []), ("filename", ""), ("filename", "x" * 256),
    ("purpose", []), ("purpose", "other"), ("content_type", None),
    ("tenant_id", "other"), ("id", "../victim"),
])
def test_bad_file_metadata_is_rejected_before_content_read(tmp_path, monkeypatch, field, value):
    writer = FileStore(root=tmp_path)
    writer.create("tenant", filename="a.txt", purpose="user_data",
                  content_type="text/plain", content=b"hello")
    metadata_path = next(tmp_path.glob("*/*.json"))
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))
    read = resources._read_restore_file
    seen = []

    def record_read(path, limit):
        seen.append(path)
        return read(path, limit)

    monkeypatch.setattr(resources, "_read_restore_file", record_read)
    restored = FileStore(root=tmp_path, max_file_bytes=20, max_bytes=100)

    assert restored.status()["files"] == 0
    assert restored.status()["counts"]["restore_failures"] == 1
    assert seen == [metadata_path]
    assert metadata_path.exists()
    assert metadata_path.with_suffix(".bin").read_bytes() == b"hello"


def test_oversized_response_is_skipped_without_discarding_valid_older_records(tmp_path):
    writer = ResponseStore(root=tmp_path)
    writer.put("tenant", {"id": "resp_a"}, context("small"))
    writer.put("tenant", {"id": "resp_z"}, context("x" * 5000))
    paths = ordered_paths(tmp_path)
    restored = ResponseStore(root=tmp_path, max_bytes=1000)
    assert restored.context("tenant", "resp_a") == context("small")
    assert restored.status()["counts"]["restore_failures"] == 1
    assert all(path.exists() for path in paths)


def test_bounded_reader_rejects_growth_and_large_file_before_read(tmp_path, monkeypatch):
    path = tmp_path / "record"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="bound"):
        resources._read_restore_file(path, 2)
    real_fstat = resources.os.fstat

    def old_size(fd):
        info = real_fstat(fd)
        return SimpleNamespace(st_mode=info.st_mode, st_size=1)

    monkeypatch.setattr(resources.os, "fstat", old_size)
    with pytest.raises(ValueError, match="grew"):
        resources._read_restore_file(path, 10)


def test_restore_skips_disappearing_candidate(tmp_path, monkeypatch):
    writer = ResponseStore(root=tmp_path)
    writer.put("tenant", {"id": "resp_bad"}, context("bad"))
    writer.put("tenant", {"id": "resp_good"}, context("good"))
    original = Path.lstat

    def vanished(path):
        if path.name == "resp_bad.json":
            raise FileNotFoundError(path)
        return original(path)

    monkeypatch.setattr(Path, "lstat", vanished)
    restored = ResponseStore(root=tmp_path)
    assert restored.status()["entries"] == 1
    assert restored.status()["counts"]["restore_failures"] == 1


@pytest.mark.parametrize("kind", ["metadata", "tenant", "content"])
def test_restore_does_not_follow_symlinks(tmp_path, kind):
    root = tmp_path / "store"
    writer = FileStore(root=root)
    writer.create("tenant", filename="a.txt", purpose="user_data",
                  content_type="text/plain", content=b"hello")
    path = next(root.glob("*/*.json"))
    target = path.parent if kind == "tenant" else path if kind == "metadata" else path.with_suffix(".bin")
    external = tmp_path / "external"
    target.rename(external)
    target.symlink_to(external, target_is_directory=kind == "tenant")
    restored = FileStore(root=root)
    assert restored.status()["files"] == 0
    assert restored.status()["counts"]["restore_failures"] == 1
    assert external.exists()


def test_cleanup_failure_does_not_defeat_restore_bounds(tmp_path, monkeypatch):
    writer = ResponseStore(root=tmp_path)
    for index in range(4):
        writer.put("tenant", {"id": f"resp_{index}"}, context(str(index)))
    ordered_paths(tmp_path)
    original = Path.unlink

    def denied(path, *args, **kwargs):
        if path.suffix == ".json":
            raise PermissionError(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", denied)
    restored = ResponseStore(root=tmp_path, max_entries=1)
    assert restored.status()["entries"] == 1
    assert restored.get("tenant", "resp_3")["id"] == "resp_3"
    assert restored.status()["counts"]["restore_cleanup_failures"] == 3


@pytest.mark.parametrize("remove_parent", ["delete", "evict"])
def test_child_continuation_survives_parent_removal_and_restart(tmp_path, remove_parent):
    writer = ResponseStore(root=tmp_path, max_entries=2)
    parent = context("parent")
    child = parent + context("child")
    writer.put("tenant", {"id": "resp_parent"}, parent)
    captured = writer.context("tenant", "resp_parent")
    if remove_parent == "delete":
        writer.delete("tenant", "resp_parent")
    else:
        writer.put("tenant", {"id": "resp_other"}, context("other"))
    # Complete a request using context captured before the parent went away.
    writer.put("tenant", {"id": "resp_child", "previous_response_id": "resp_parent"},
               captured + context("child"))
    restored = ResponseStore(root=tmp_path, max_entries=2)
    assert restored.context("tenant", "resp_child") == child
    with pytest.raises(ResourceNotFound):
        restored.get("tenant", "resp_parent")
    with pytest.raises(ResourceNotFound):
        restored.context("other-tenant", "resp_child")
    result = restored.context("tenant", "resp_child")
    result[0]["content"] = "changed"
    assert restored.context("tenant", "resp_child") == child


def test_input_record_copies_only_context_and_preserves_tenant_isolation(monkeypatch):
    store = ResponseStore()
    payload = {"id": "resp_test", "model": "fixture", "output": [{"large": "x" * 10000}]}
    store.put("tenant", payload, context("hello"))
    stored_payload = store._entries[("tenant", "resp_test")][1]["payload"]
    original = resources.deepcopy

    def checked_copy(value):
        assert value is not stored_payload
        assert value is not stored_payload["output"]
        assert not (isinstance(value, dict) and "payload" in value)
        return original(value)

    monkeypatch.setattr(resources, "deepcopy", checked_copy)
    record = store.input_record("tenant", "resp_test")
    assert record == {"payload": {"model": "fixture"}, "context_messages": context("hello")}
    record["context_messages"][0]["content"] = "changed"
    assert store.context("tenant", "resp_test") == context("hello")
    with pytest.raises(ResourceNotFound):
        store.input_record("other-tenant", "resp_test")


@pytest.mark.parametrize("raw", [b"{", b'{"payload":NaN}', b"[]"])
def test_invalid_response_json_is_counted(tmp_path, raw):
    writer = ResponseStore(root=tmp_path)
    writer.put("tenant", {"id": "resp_bad"}, context("bad"))
    path = next(tmp_path.glob("*/*.json"))
    path.write_bytes(raw)
    restored = ResponseStore(root=tmp_path)
    assert restored.status()["entries"] == 0
    assert restored.status()["counts"]["restore_failures"] == 1
    assert path.exists()


@pytest.mark.parametrize("binary", [b"he", b"hello-too-long"])
def test_file_binary_must_match_declared_size(tmp_path, binary):
    writer = FileStore(root=tmp_path)
    writer.create("tenant", filename="a.txt", purpose="user_data",
                  content_type="text/plain", content=b"hello")
    path = next(tmp_path.glob("*/*.bin"))
    path.write_bytes(binary)
    restored = FileStore(root=tmp_path)
    assert restored.status()["files"] == 0
    assert restored.status()["counts"]["restore_failures"] == 1
    assert path.read_bytes() == binary


def test_file_restore_respects_smaller_per_file_limit_before_binary_read(tmp_path, monkeypatch):
    writer = FileStore(root=tmp_path)
    writer.create("tenant", filename="a.txt", purpose="user_data",
                  content_type="text/plain", content=b"hello")
    original = resources._read_restore_file

    def metadata_only(path, limit):
        assert path.suffix == ".json"
        return original(path, limit)

    monkeypatch.setattr(resources, "_read_restore_file", metadata_only)
    restored = FileStore(root=tmp_path, max_file_bytes=4)
    assert restored.status()["files"] == 0
    assert restored.status()["counts"]["restore_failures"] == 1


def test_oversized_file_metadata_is_not_decoded(tmp_path, monkeypatch):
    writer = FileStore(root=tmp_path)
    writer.create("tenant", filename="a.txt", purpose="user_data",
                  content_type="text/plain", content=b"hello")
    path = next(tmp_path.glob("*/*.json"))
    path.write_bytes(b" " * (64 * 1024 + 1))

    def no_decode(raw):
        raise AssertionError("oversized metadata must not reach JSON decoder")

    monkeypatch.setattr(resources, "_restore_json", no_decode)
    restored = FileStore(root=tmp_path)
    assert restored.status()["files"] == 0
    assert restored.status()["counts"]["restore_failures"] == 1


def test_restore_rejects_copied_identity_in_wrong_tenant_directory(tmp_path):
    writer = ResponseStore(root=tmp_path)
    writer.put("tenant", {"id": "resp_same"}, context("original"))
    original = next(tmp_path.glob("*/*.json"))
    wrong = tmp_path / resources._tenant_name("other") / original.name
    wrong.parent.mkdir()
    wrong.write_bytes(original.read_bytes())
    restored = ResponseStore(root=tmp_path, max_entries=1)
    assert restored.status()["entries"] == 1
    assert restored.status()["counts"]["restore_failures"] == 1
    assert restored.context("tenant", "resp_same") == context("original")
    assert restored.status()["bytes"] == original.stat().st_size
    assert original.exists() and wrong.exists()


def test_equal_mtimes_have_deterministic_retention(tmp_path):
    writer = ResponseStore(root=tmp_path)
    for index in reversed(range(4)):
        writer.put("tenant", {"id": f"resp_{index}"}, context(str(index)))
    for path in tmp_path.glob("*/*.json"):
        os.utime(path, ns=(1000, 1000))
    restored = ResponseStore(root=tmp_path, max_entries=2)
    assert [key[1] for key in restored._entries] == ["resp_2", "resp_3"]
