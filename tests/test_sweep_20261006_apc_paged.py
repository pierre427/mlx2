"""Paged APCv2 index/bridge/private-cache hardening (2026-10-06 sweep).

PAGED-1: a store whose index write failed left its checkpoint directory
behind, so a retry raised FileExistsError and every restart refused the
directory.  PAGED-3: a token restore checked page bytes only, so a bf16
checkpoint restored into an fp16 arena.  PAGED-4: a cache whose failure event
a sibling consumed accepted its failed append.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.apc_v2 import APCKey
from mlx2.runtime.paged_apc_bridge import ExactPagedAPCCheckpoint
from mlx2.runtime.paged_apc_index import PagedAPCIndex
from mlx2.runtime.paged_kv_cache import PagedKVPrivateCache
from mlx2.runtime.paged_kv_pool import PagedKVPool
from mlx2.runtime.paged_kv_token import PagedKVTokenOwner, TokenKVProfile
from mlx2.runtime.paged_kv_write import PagedKVWriteOwner

PROFILE = TokenKVProfile(2, 128, "float16")
TOKEN_BYTES = PROFILE.token_bytes
HEAD = PROFILE.head_token_bytes
PAGE = PROFILE.page_bytes


@pytest.fixture(autouse=True)
def cpu_default():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


class SimArena:
    """Byte-accurate fake arena: writes/copies land immediately, events queued."""

    def __init__(self, pages, storage_dtype="float16"):
        self.plane_bytes = pages * PAGE
        self.k = bytearray(self.plane_bytes)
        self.v = bytearray(self.plane_bytes)
        self.events = []
        self.storage_dtype = storage_dtype
        self.auto = True
        self.fail_epochs = set()

    def validate_sources(self, key, value):
        return all(type(x) is mx.array and x.dtype == mx.uint8 and x.ndim == 1
                   for x in (key, value))

    def _queue(self, epoch):
        self.events.append((epoch, epoch not in self.fail_epochs))

    def write(self, key, value, offset, byte_count, epoch):
        if isinstance(key, (bytes, bytearray)):
            kb, vb = bytes(key), bytes(value)
        else:
            kb, vb = np.asarray(key).tobytes(), np.asarray(value).tobytes()
        self.k[offset:offset + byte_count] = kb
        self.v[offset:offset + byte_count] = vb
        self._queue(epoch)
        return object()

    def copy_page(self, source, destination, byte_count, epoch):
        self.k[destination:destination + byte_count] = self.k[source:source + byte_count]
        self.v[destination:destination + byte_count] = self.v[source:source + byte_count]
        self._queue(epoch)
        return object()

    def depend_source(self, source, _dep):
        return source

    def poll_completions(self):
        events, self.events = self.events, []
        return events


def logical(tokens, salt):
    rng = np.random.default_rng(salt)
    return rng.integers(0, 256, size=tokens * TOKEN_BYTES, dtype=np.uint8).tobytes()


def read_back(arena, plane, handles, tokens, first_block=0):
    """Independent oracle of the [page, head, token, channel] address formula."""
    out = bytearray()
    for t in range(tokens):
        page = handles[t // 64 - first_block].page_id
        for h in range(PROFILE.kv_heads):
            off = page * PAGE + (h * 64 + t % 64) * HEAD
            out += plane[off:off + HEAD]
    return bytes(out)


def drain(owner, arena):
    for _ in range(4):
        owner.poll_completions()
        if not owner._pending:
            return


def fake_source(tokens, keys, values, tmp_path, key, dtype="float16"):
    backend = SimArena(4)
    w = PagedKVWriteOwner(PagedKVPool(4), backend, page_bytes=PAGE, permit_candidate=True)
    src = PagedKVPrivateCache(w, kv_heads=2, head_dim=128, dtype=dtype, permit_candidate=True)
    src.append(keys, values)
    assert src.poll_completions()
    path = ExactPagedAPCCheckpoint.save(tmp_path, key=key, tokens=tokens, source=src,
                                         permit_candidate=True)
    src.close()
    return path


KEY = APCKey(model="qwen3", revision="rev-1", tokenizer_fingerprint="tok")


def test_failed_index_write_leaves_the_store_retryable(tmp_path, monkeypatch):
    path_dir = tmp_path / "idx"
    index = PagedAPCIndex(path_dir, permit_candidate=True)
    backend = SimArena(4)
    w = PagedKVWriteOwner(PagedKVPool(4), backend, page_bytes=PAGE, permit_candidate=True)
    src = PagedKVPrivateCache(w, kv_heads=2, head_dim=128, dtype="float16", permit_candidate=True)
    src.append(logical(3, 1), logical(3, 2))
    assert src.poll_completions()
    calls = {"n": 0}
    real = PagedAPCIndex._write

    def flaky(self, entries):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(28, "No space left on device")
        return real(self, entries)

    monkeypatch.setattr(PagedAPCIndex, "_write", flaky)
    with pytest.raises(OSError):
        index.store(key=KEY, tokens=(1, 2, 3), source=src)
    # A failed store leaves no orphan, so a retry in the same process works.
    index.store(key=KEY, tokens=(1, 2, 3), source=src)


def test_index_restart_after_failed_store_is_usable(tmp_path, monkeypatch):
    path_dir = tmp_path / "idx"
    index = PagedAPCIndex(path_dir, permit_candidate=True)
    backend = SimArena(4)
    w = PagedKVWriteOwner(PagedKVPool(4), backend, page_bytes=PAGE, permit_candidate=True)
    src = PagedKVPrivateCache(w, kv_heads=2, head_dim=128, dtype="float16", permit_candidate=True)
    src.append(logical(3, 1), logical(3, 2))
    assert src.poll_completions()
    monkeypatch.setattr(PagedAPCIndex, "_write",
                        lambda self, entries: (_ for _ in ()).throw(OSError(28, "ENOSPC")))
    with pytest.raises(OSError):
        index.store(key=KEY, tokens=(1, 2, 3), source=src)
    monkeypatch.undo()
    PagedAPCIndex(path_dir, permit_candidate=True)


def test_index_restart_after_failed_store_is_usable(tmp_path, monkeypatch):
    path_dir = tmp_path / "idx"
    index = PagedAPCIndex(path_dir, permit_candidate=True)
    backend = SimArena(4)
    w = PagedKVWriteOwner(PagedKVPool(4), backend, page_bytes=PAGE, permit_candidate=True)
    src = PagedKVPrivateCache(w, kv_heads=2, head_dim=128, dtype="float16", permit_candidate=True)
    src.append(logical(3, 1), logical(3, 2))
    assert src.poll_completions()
    monkeypatch.setattr(PagedAPCIndex, "_write",
                        lambda self, entries: (_ for _ in ()).throw(OSError(28, "ENOSPC")))
    with pytest.raises(OSError):
        index.store(key=KEY, tokens=(1, 2, 3), source=src)
    monkeypatch.undo()
    PagedAPCIndex(path_dir, permit_candidate=True)


def test_token_restore_refuses_dtype_mismatch_with_arena(tmp_path):
    tokens = (1, 2, 3)
    path = fake_source(tokens, logical(3, 3), logical(3, 4), tmp_path, KEY, dtype="bfloat16")
    arena = SimArena(2, storage_dtype="float16")
    w = PagedKVWriteOwner(PagedKVPool(2), arena, page_bytes=PAGE, permit_candidate=True)
    # bf16 checkpoint bytes must not land in an fp16-storage arena.
    with pytest.raises(ValueError):
        ExactPagedAPCCheckpoint.restore_token_owner(
            path, key=KEY, tokens=tokens, writer=w, permit_candidate=True)


def test_private_cache_refuses_its_failed_append_after_a_sibling_polled_it():
    arena = SimArena(4)
    w = PagedKVWriteOwner(PagedKVPool(4), arena, page_bytes=PAGE, permit_candidate=True)
    a = PagedKVPrivateCache(w, kv_heads=2, head_dim=128, dtype="float16", permit_candidate=True)
    b = PagedKVPrivateCache(w, kv_heads=2, head_dim=128, dtype="float16", permit_candidate=True)
    arena.fail_epochs = {1}
    a.append(logical(1, 1), logical(1, 2))  # epoch 1 -> fails
    b.append(logical(1, 3), logical(1, 4))  # epoch 2 -> succeeds
    b.poll_completions()                    # B consumes A's failure event
    accepted = a.poll_completions()
    # A's failed write is never accepted, whoever polled its failure event.
    assert not accepted and a.offset == 0

