"""Exact source I/O and seal lifetime; synthetic files, no model or GPU."""
import os

import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_source_artifact as source
from mlx2.runtime.immutable_weight_pages import file_stamp


def _artifact(tmp_path):
    tensors, seals, expected = {}, {}, {}
    for shard in ("a.bin", "b.bin"):
        payload = bytearray(b"unread-prefix")
        for i in range(6):
            name = f"{shard}.{i}"
            data = np.arange(32, dtype=np.uint8) + i
            tensors[name] = dict(file=shard, offset=len(payload), size=data.nbytes,
                                 dtype="I8", shape=(2, 16))
            payload.extend(data.tobytes())
            payload.extend(b"unread-gap")
            expected[name] = data.reshape(2, 16)
        path = tmp_path / shard
        path.write_bytes(payload)
        seals[shard] = file_stamp(path.stat())
    return dict(root=tmp_path, tensors=tensors, seals=seals), expected


def _track_descriptors(monkeypatch):
    opened, closed = [], []
    real_open, real_close = os.open, os.close

    def open_file(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def close_file(fd):
        closed.append(fd)
        return real_close(fd)

    monkeypatch.setattr(source.os, "open", open_file)
    monkeypatch.setattr(source.os, "close", close_file)
    return opened, closed


def test_six_expert_extents_share_descriptor_without_reading_gaps(tmp_path, monkeypatch):
    artifact, expected = _artifact(tmp_path)
    names = [f"a.bin.{i}" for i in range(6)]
    opened, closed = _track_descriptors(monkeypatch)
    calls, real_read = [], source.read_page

    def read(fd, offset, size, chunk):
        calls.append((fd, offset, size, chunk))
        return real_read(fd, offset, size, chunk)

    monkeypatch.setattr(source, "read_page", read)
    actual = source.read_tensors(artifact, names)
    assert len(opened) == 1 and closed == opened
    assert [(offset, size) for _, offset, size, _ in calls] == [
        (artifact["tensors"][n]["offset"], artifact["tensors"][n]["size"]) for n in names]
    assert all(chunk == 16 << 20 for _, _, _, chunk in calls)
    assert sum(v.nbytes for v in actual.values()) == 6 * 32
    for name in names:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_interleaved_shards_preserve_requested_order_and_codes(tmp_path, monkeypatch):
    artifact, expected = _artifact(tmp_path)
    names = ["b.bin.3", "a.bin.5", "b.bin.1", "a.bin.0"]
    opened, closed = _track_descriptors(monkeypatch)
    actual = source.read_tensors(artifact, iter(names))
    assert list(actual) == names
    assert len(opened) == 2 and closed == opened
    for name in names:
        assert actual[name].tobytes() == expected[name].tobytes()


@pytest.mark.parametrize("mutation", ["append", "replace", "unlink"])
def test_changed_shard_during_batch_is_rejected_and_closed(tmp_path, monkeypatch, mutation):
    artifact, _ = _artifact(tmp_path)
    opened, closed = _track_descriptors(monkeypatch)
    real_read = source.read_page
    changed = False

    def read(*args):
        nonlocal changed
        result = real_read(*args)
        if not changed:
            changed = True
            path = tmp_path / "a.bin"
            if mutation == "append":
                with path.open("ab") as out:
                    out.write(b"x")
            elif mutation == "replace":
                replacement = tmp_path / "replacement"
                replacement.write_bytes(path.read_bytes())
                os.replace(replacement, path)
            else:
                path.unlink()
        return result

    monkeypatch.setattr(source, "read_page", read)
    with pytest.raises((ValueError, FileNotFoundError)):
        source.read_tensors(artifact, ["a.bin.0", "a.bin.1"])
    assert len(opened) == 1 and closed == opened


def test_short_extent_closes_shared_descriptor(tmp_path, monkeypatch):
    artifact, _ = _artifact(tmp_path)
    artifact["tensors"]["a.bin.1"]["offset"] = artifact["seals"]["a.bin"]["size"]
    opened, closed = _track_descriptors(monkeypatch)
    with pytest.raises(OSError, match="short"):
        source.read_tensors(artifact, ["a.bin.0", "a.bin.1"])
    assert len(opened) == 1 and closed == opened


def test_empty_read_does_not_open_shards(tmp_path, monkeypatch):
    artifact, _ = _artifact(tmp_path)
    opened, closed = _track_descriptors(monkeypatch)
    assert source.read_tensors(artifact, []) == {}
    assert opened == closed == []
