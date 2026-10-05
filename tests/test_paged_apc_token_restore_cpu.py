"""Logical APCv2 bytes restored into native head-major token pages; CPU only."""

import mlx.core as mx
import numpy as np
import pytest
from test_paged_apc_bridge_cpu import TOKEN_BYTES, accept, writer
from test_paged_kv_token_cpu import PROFILE, FakeBackend

from mlx2.runtime.apc_v2 import APCKey
from mlx2.runtime.paged_apc_bridge import ExactPagedAPCCheckpoint
from mlx2.runtime.paged_apc_index import PagedAPCIndex
from mlx2.runtime.paged_kv_cache import PagedKVPrivateCache
from mlx2.runtime.paged_kv_pool import PagedKVPool
from mlx2.runtime.paged_kv_write import PagedKVWriteOwner


@pytest.fixture(autouse=True)
def cpu_default():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def checkpoint(tmp_path, tokens):
    key = APCKey(model="qwen3", revision="exact-weight-revision",
                 tokenizer_fingerprint="exact-tokenizer")
    source_writer, source_backend = writer(3)
    source = PagedKVPrivateCache(source_writer, kv_heads=2, head_dim=128,
                                 dtype="float16", permit_candidate=True)
    key_bytes = bytes((row * 7 + head * 31 + channel) % 256
                      for row in range(len(tokens)) for head in range(2)
                      for channel in range(PROFILE.head_token_bytes))
    value_bytes = bytes((row * 11 + head * 17 + channel) % 256
                        for row in range(len(tokens)) for head in range(2)
                        for channel in range(PROFILE.head_token_bytes))
    accept(source, source_backend, source.append(key_bytes, value_bytes))
    path = ExactPagedAPCCheckpoint.save(tmp_path, key=key, tokens=tokens,
                                         source=source, permit_candidate=True)
    source.close()
    return key, path, key_bytes, value_bytes


def token_writer(pages):
    backend = FakeBackend(pages)
    native = PagedKVWriteOwner(PagedKVPool(pages), backend,
                               page_bytes=PROFILE.page_bytes, permit_candidate=True)
    return native, backend


def test_exact_ragged_restore_is_pending_then_head_major_and_unshared(tmp_path):
    tokens = tuple(range(65))
    key, path, keys, values = checkpoint(tmp_path, tokens)
    native, backend = token_writer(2)
    owner, tickets = ExactPagedAPCCheckpoint.restore_token_owner(
        path, key=key, tokens=tokens, writer=native, permit_candidate=True)
    assert owner.offset == 0 and len(tickets) == len(backend.writes) == 4
    with pytest.raises(RuntimeError, match="pending"):
        owner.accepted_handles()
    # Planned spans are captured before append because pending writes refuse
    # another plan. Their offsets and source bytes must match the writer log.
    expected = ((0, 64, 0, 0), (0, 64, 1, 0),
                (64, 1, 0, 1), (64, 1, 1, 1))
    for write, (row_start, count, head, block) in zip(backend.writes, expected):
        begin = (head * 64) * PROFILE.head_token_bytes + block * PROFILE.page_bytes
        assert write[2] == begin
        assert write[3] == count * PROFILE.head_token_bytes
        logical = b"".join(
            keys[row * TOKEN_BYTES + head * PROFILE.head_token_bytes:
                 row * TOKEN_BYTES + (head + 1) * PROFILE.head_token_bytes]
            for row in range(row_start, row_start + count))
        assert np.asarray(write[0]).tobytes() == logical
        assert np.asarray(write[1]).tobytes() == b"".join(
            values[row * TOKEN_BYTES + head * PROFILE.head_token_bytes:
                   row * TOKEN_BYTES + (head + 1) * PROFILE.head_token_bytes]
            for row in range(row_start, row_start + count))
    backend.events.extend((ticket.epoch, True) for ticket in tickets)
    assert owner.poll_completions()
    assert owner.offset == 65
    assert all(native.pool.references(handle) == 1 for handle in owner.accepted_handles())
    owner.close()
    assert native.pool.free_count == 2


def test_index_restore_refuses_identity_corruption_and_capacity(tmp_path):
    tokens = (1, 2, 3)
    key, path, _, _ = checkpoint(tmp_path / "direct", tokens)
    native, _ = token_writer(1)
    wrong = APCKey(model="qwen3", revision="other")
    with pytest.raises(ValueError, match="identity"):
        ExactPagedAPCCheckpoint.restore_token_owner(
            path, key=wrong, tokens=tokens, writer=native, permit_candidate=True)
    with pytest.raises(ValueError, match="retain staging headroom"):
        ExactPagedAPCCheckpoint.restore_token_owner(
            path, key=key, tokens=tokens, writer=native,
            staging_headroom_pages=1, permit_candidate=True)
    long_tokens = tuple(range(65))
    long_key, long_path, _, _ = checkpoint(tmp_path / "long", long_tokens)
    with pytest.raises(MemoryError, match="headroom"):
        ExactPagedAPCCheckpoint.restore_token_owner(
            long_path, key=long_key, tokens=long_tokens, writer=native,
            permit_candidate=True)
    assert native.pool.free_count == 1
    (path / "keys.bin").write_bytes(b"x" * (3 * TOKEN_BYTES))
    with pytest.raises(ValueError, match="payload"):
        ExactPagedAPCCheckpoint.restore_token_owner(
            path, key=key, tokens=tokens, writer=native, permit_candidate=True)
    assert native.pool.free_count == 1


def test_index_longest_prefix_restores_native_owner(tmp_path):
    key = APCKey(model="qwen3", revision="exact-weight-revision")
    source_writer, source_backend = writer(1)
    source = PagedKVPrivateCache(source_writer, kv_heads=2, head_dim=128,
                                 dtype="float16", permit_candidate=True)
    accept(source, source_backend,
           source.append(b"k" * (2 * TOKEN_BYTES), b"v" * (2 * TOKEN_BYTES)))
    index = PagedAPCIndex(tmp_path, permit_candidate=True)
    index.store(key=key, tokens=(8, 9), source=source)
    native, backend = token_writer(1)
    owner, tickets, count = index.restore_longest_token_owner(
        key=key, tokens=(8, 9, 10), writer=native)
    assert count == 2 and owner.offset == 0
    backend.events.extend((ticket.epoch, True) for ticket in tickets)
    assert owner.poll_completions() and owner.offset == 2
    owner.close()
    assert index.restore_longest_token_owner(
        key=APCKey(model="qwen3", revision="other"),
        tokens=(8, 9, 10), writer=native) is None


def test_failed_terminal_write_never_accepts_restored_prefix(tmp_path):
    key, path, _, _ = checkpoint(tmp_path, (3, 4))
    native, backend = token_writer(1)
    owner, tickets = ExactPagedAPCCheckpoint.restore_token_owner(
        path, key=key, tokens=(3, 4), writer=native, permit_candidate=True)
    backend.events.extend((ticket.epoch, False) for ticket in tickets)
    assert not owner.poll_completions()
    assert owner.offset == 0
    with pytest.raises(RuntimeError, match="failed"):
        owner.accepted_handles()
    owner.close()
    assert native.pool.free_count == 0
    assert native.pool.quarantined_count == 1


def test_direct_restore_refuses_symlinked_path_or_manifest_before_writes(tmp_path):
    tokens = (3, 4)
    key, path, _, _ = checkpoint(tmp_path / "source", tokens)
    native, backend = token_writer(1)
    linked_path = tmp_path / "linked-checkpoint"
    linked_path.symlink_to(path, target_is_directory=True)
    with pytest.raises(ValueError, match="path is missing or unsafe"):
        ExactPagedAPCCheckpoint.restore_token_owner(
            linked_path, key=key, tokens=tokens, writer=native,
            permit_candidate=True)
    manifest = path / "manifest.json"
    original = manifest.read_bytes()
    manifest.unlink()
    target = tmp_path / "external-manifest.json"
    target.write_bytes(original)
    manifest.symlink_to(target)
    with pytest.raises(ValueError, match="manifest is missing, unsafe or oversized"):
        ExactPagedAPCCheckpoint.restore_token_owner(
            path, key=key, tokens=tokens, writer=native,
            permit_candidate=True)
    assert native.pool.free_count == 1 and backend.writes == []


def test_direct_restore_refuses_oversized_manifest_before_writes(tmp_path):
    tokens = (3, 4)
    key, path, _, _ = checkpoint(tmp_path, tokens)
    (path / "manifest.json").write_bytes(b" " * 1_000_001)
    native, backend = token_writer(1)
    with pytest.raises(ValueError, match="manifest is missing, unsafe or oversized"):
        ExactPagedAPCCheckpoint.restore_token_owner(
            path, key=key, tokens=tokens, writer=native,
            permit_candidate=True)
    assert native.pool.free_count == 1 and backend.writes == []
