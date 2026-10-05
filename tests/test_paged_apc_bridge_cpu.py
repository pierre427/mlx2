"""Exact APCv2 paged checkpoint tests; fake completions, no GPU."""

import json

import pytest

from mlx2.runtime.apc_v2 import APCKey
from mlx2.runtime.paged_apc_bridge import ExactPagedAPCCheckpoint
from mlx2.runtime.paged_kv_cache import PagedKVPrivateCache
from mlx2.runtime.paged_kv_pool import PagedKVPool
from mlx2.runtime.paged_kv_write import PagedKVWriteOwner

TOKEN_BYTES = 2 * 128 * 2
PAGE_BYTES = 64 * TOKEN_BYTES


class Backend:
    def __init__(self, pages):
        self.plane_bytes = pages * PAGE_BYTES
        self.events = []
        self.fail_write = False

    def write(self, key, value, offset, byte_count, epoch):
        if self.fail_write:
            raise RuntimeError("ambiguous native submission")
        return object()

    def copy_page(self, source, destination, byte_count, epoch):
        return object()

    def poll_completions(self):
        events, self.events = self.events, []
        return events


def writer(pages):
    backend = Backend(pages)
    return PagedKVWriteOwner(PagedKVPool(pages), backend,
                             page_bytes=PAGE_BYTES, permit_candidate=True), backend


def accept(owner, backend, tickets):
    backend.events.extend((ticket.epoch, True) for ticket in tickets)
    assert owner.poll_completions()


def test_cold_restart_exact_continuation_and_no_physical_refs(tmp_path):
    key = APCKey(model="test", revision="weights-abc", tokenizer_fingerprint="tok")
    tokens = tuple(range(65))
    first, backend = writer(3)
    hot = PagedKVPrivateCache(first, kv_heads=2, head_dim=128,
                              dtype="float16", permit_candidate=True)
    keys, values = b"k" * (65 * TOKEN_BYTES), b"v" * (65 * TOKEN_BYTES)
    accept(hot, backend, hot.append(keys, values))
    path = ExactPagedAPCCheckpoint.save(tmp_path, key=key, tokens=tokens,
                                         source=hot, permit_candidate=True)
    manifest = json.loads((path / "manifest.json").read_text())
    assert "page_id" not in json.dumps(manifest)
    hot.close()

    fresh, fresh_backend = writer(4)
    cold, tickets = ExactPagedAPCCheckpoint.restore(
        path, key=key, tokens=tokens, writer=fresh,
        staging_headroom_pages=1, permit_candidate=True)
    assert fresh.pool.free_count == 1  # Two data pages and one staging page held.
    with pytest.raises(RuntimeError, match="pending"):
        cold.export_exact()
    accept(cold, fresh_backend, tickets)
    assert cold.export_exact().key_bytes == keys
    assert cold.export_exact().value_bytes == values
    accept(cold, fresh_backend, cold.append(b"a" * TOKEN_BYTES, b"b" * TOKEN_BYTES))
    assert cold.export_exact().key_bytes == keys + b"a" * TOKEN_BYTES
    cold.close()


def test_revision_integrity_and_staging_budget_fail_closed(tmp_path):
    key = APCKey(model="test", revision="one")
    source_writer, backend = writer(1)
    source = PagedKVPrivateCache(source_writer, kv_heads=2, head_dim=128,
                                 dtype="float16", permit_candidate=True)
    accept(source, backend, source.append(b"k" * TOKEN_BYTES, b"v" * TOKEN_BYTES))
    path = ExactPagedAPCCheckpoint.save(tmp_path, key=key, tokens=(7,),
                                         source=source, permit_candidate=True)
    fresh, _ = writer(1)
    with pytest.raises(ValueError, match="identity"):
        ExactPagedAPCCheckpoint.restore(path, key=APCKey(model="test", revision="two"),
                                        tokens=(7,), writer=fresh, permit_candidate=True)
    with pytest.raises(MemoryError, match="headroom"):
        ExactPagedAPCCheckpoint.restore(path, key=key, tokens=(7,), writer=fresh,
                                        staging_headroom_pages=1, permit_candidate=True)
    assert fresh.pool.free_count == 1
    (path / "keys.bin").write_bytes(b"x" * TOKEN_BYTES)
    with pytest.raises(ValueError, match="payload"):
        ExactPagedAPCCheckpoint.restore(path, key=key, tokens=(7,),
                                        writer=fresh, permit_candidate=True)
    assert fresh.pool.free_count == 1


def test_malformed_manifest_and_failed_submission_release_staging(tmp_path):
    key = APCKey(model="test", revision="one")
    source_writer, backend = writer(1)
    source = PagedKVPrivateCache(source_writer, kv_heads=2, head_dim=128,
                                 dtype="float16", permit_candidate=True)
    accept(source, backend, source.append(b"k" * TOKEN_BYTES, b"v" * TOKEN_BYTES))
    path = ExactPagedAPCCheckpoint.save(tmp_path, key=key, tokens=(7,),
                                         source=source, permit_candidate=True)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["profile"]["kv_heads"] = True
    manifest_path.write_text(json.dumps(manifest))
    fresh, fresh_backend = writer(2)
    with pytest.raises(ValueError, match="geometry"):
        ExactPagedAPCCheckpoint.restore(path, key=key, tokens=(7,),
                                        writer=fresh, staging_headroom_pages=1,
                                        permit_candidate=True)
    assert fresh.pool.free_count == 2
    manifest["profile"]["kv_heads"] = 2
    manifest["identity"]["tokens"] = [True]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        ExactPagedAPCCheckpoint.restore(path, key=key, tokens=(1,),
                                        writer=fresh, permit_candidate=True)
    manifest["identity"]["tokens"] = [7]
    manifest_path.write_text(json.dumps(manifest))
    fresh_backend.fail_write = True
    with pytest.raises(RuntimeError, match="failed at epoch"):
        ExactPagedAPCCheckpoint.restore(path, key=key, tokens=(7,),
                                        writer=fresh, staging_headroom_pages=1,
                                        permit_candidate=True)
    assert fresh.pool.free_count == 0  # Both releases wait for ambiguous epoch.
    assert fresh.pool.pending_count == 1  # Staging has no submitted use.
    fresh_backend.events.append((1, False))
    fresh.poll_completions()
    assert fresh.pool.free_count == 1
    assert fresh.pool.quarantined_count == 1
