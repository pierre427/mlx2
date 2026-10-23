"""CPU-only index and restart checks with fake native completion events."""

import json
from pathlib import Path

import pytest

from mlx2.runtime.apc_v2 import APCKey
from mlx2.runtime.paged_apc_bridge import ExactPagedAPCCheckpoint
from mlx2.runtime.paged_apc_index import PagedAPCIndex
from mlx2.runtime.paged_kv_cache import PagedKVPrivateCache
from test_paged_apc_bridge_cpu import TOKEN_BYTES, accept, writer


def saved_source(tokens):
    native, backend = writer(2)
    source = PagedKVPrivateCache(native, kv_heads=2, head_dim=128,
                                 dtype="float16", permit_candidate=True)
    count = len(tokens)
    accept(source, backend, source.append(b"k" * (count * TOKEN_BYTES),
                                          b"v" * (count * TOKEN_BYTES)))
    return source


def test_longest_prefix_restart_and_pending_restore(tmp_path):
    key = APCKey(model="test", revision="weight-v1", tokenizer_fingerprint="tok-v1")
    source = saved_source((1, 2, 3))
    index = PagedAPCIndex(tmp_path, max_entries=2, max_tokens_per_entry=8,
                          max_payload_bytes=10 * TOKEN_BYTES, permit_candidate=True)
    index.store(key=key, tokens=(1, 2), source=saved_source((1, 2)))
    index.store(key=key, tokens=(1, 2, 3), source=source)
    restarted = PagedAPCIndex(tmp_path, max_entries=2, max_tokens_per_entry=8,
                              max_payload_bytes=10 * TOKEN_BYTES, permit_candidate=True)
    assert restarted.lookup(key=key, tokens=(1, 2, 3, 4))[1] == 3
    assert restarted.lookup(key=key, tokens=(1, 2, 9))[1] == 2
    assert restarted.lookup(key=APCKey(model="test", revision="weight-v2"),
                            tokens=(1, 2, 3)) is None
    assert restarted.lookup(key=key, tokens=(9,)) is None
    native, backend = writer(2)
    owner, tickets, count = restarted.restore_longest(
        key=key, tokens=(1, 2, 3, 4), writer=native)
    assert count == 3
    with pytest.raises(RuntimeError, match="pending"):
        owner.export_exact()
    accept(owner, backend, tickets)
    assert owner.export_exact().key_bytes == b"k" * (3 * TOKEN_BYTES)
    owner.close()


def test_bounds_version_and_tampered_payload_fail_closed(tmp_path):
    key = APCKey(model="test", revision="weight-v1")
    source = saved_source((1,))
    with pytest.raises(RuntimeError, match="enablement"):
        PagedAPCIndex(tmp_path)
    index = PagedAPCIndex(tmp_path, max_entries=1, max_tokens_per_entry=1,
                          max_payload_bytes=2 * TOKEN_BYTES, permit_candidate=True)
    path = index.store(key=key, tokens=(1,), source=source)
    with pytest.raises(MemoryError, match="capacity"):
        index.store(key=key, tokens=(2,), source=source)
    with pytest.raises(ValueError, match="token bound"):
        index.store(key=key, tokens=(1, 2), source=source)
    (path / "keys.bin").write_bytes(b"x" * TOKEN_BYTES)
    native, _ = writer(1)
    with pytest.raises(ValueError, match="payload"):
        index.restore_longest(key=key, tokens=(1,), writer=native)
    assert native.pool.free_count == 1
    index_file = tmp_path / PagedAPCIndex.FILE
    document = json.loads(index_file.read_text())
    document["schema"] = "unknown-v2"
    index_file.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="schema"):
        PagedAPCIndex(tmp_path, max_entries=1, max_tokens_per_entry=1,
                      max_payload_bytes=2 * TOKEN_BYTES, permit_candidate=True)


def test_unindexed_checkpoint_rejected_instead_of_escaping_capacity(tmp_path):
    key = APCKey(model="test", revision="weight-v1")
    ExactPagedAPCCheckpoint.save(tmp_path, key=key, tokens=(1,),
                                 source=saved_source((1,)), permit_candidate=True)
    with pytest.raises(ValueError, match="unindexed"):
        PagedAPCIndex(tmp_path, max_entries=1, max_tokens_per_entry=1,
                      max_payload_bytes=2 * TOKEN_BYTES, permit_candidate=True)


def test_postpublication_index_error_keeps_memory_and_next_store_consistent(tmp_path, monkeypatch):
    key = APCKey(model="test", revision="weight-v1")
    index = PagedAPCIndex(tmp_path, permit_candidate=True)
    original_write = index._write

    def published_then_failed(entries):
        original_write(entries)
        raise OSError("directory fsync failed after index publication")

    monkeypatch.setattr(index, "_write", published_then_failed)
    with pytest.raises(OSError, match="fsync"):
        index.store(key=key, tokens=(1,), source=saved_source((1,)))
    assert index.lookup(key=key, tokens=(1,))[1] == 1
    monkeypatch.setattr(index, "_write", original_write)
    index.store(key=key, tokens=(2,), source=saved_source((2,)))
    restarted = PagedAPCIndex(tmp_path, permit_candidate=True)
    assert restarted.lookup(key=key, tokens=(1,))[1] == 1
    assert restarted.lookup(key=key, tokens=(2,))[1] == 1


def test_unknown_index_publication_preserves_payload_and_gates_until_reload(tmp_path, monkeypatch):
    key = APCKey(model="test", revision="weight-v1")
    index = PagedAPCIndex(tmp_path, permit_candidate=True)
    original_write = index._write
    original_read = Path.read_text

    def published_then_unreadable(entries):
        original_write(entries)

        def unreadable(path, *args, **kwargs):
            if path.name == PagedAPCIndex.FILE:
                raise OSError("transient index read failure")
            return original_read(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", unreadable)
        raise OSError("directory fsync failed after index publication")

    monkeypatch.setattr(index, "_write", published_then_unreadable)
    with pytest.raises(OSError, match="fsync"):
        index.store(key=key, tokens=(1,), source=saved_source((1,)))
    with pytest.raises(RuntimeError, match="reload"):
        index.lookup(key=key, tokens=(1,))
    with pytest.raises(RuntimeError, match="reload"):
        index.store(key=key, tokens=(2,), source=saved_source((2,)))
    monkeypatch.setattr(Path, "read_text", original_read)
    restarted = PagedAPCIndex(tmp_path, permit_candidate=True)
    assert restarted.lookup(key=key, tokens=(1,))[1] == 1
