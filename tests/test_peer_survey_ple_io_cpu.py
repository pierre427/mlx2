"""PLE foreground, optional prefetch and warm ownership on real file reads."""

import os
import threading
import time

import numpy as np
import pytest

from mlx2.runtime.io_ownership import ReadPool, ReadTimeout
from mlx2.runtime.models import qwen4_ple_nvme as ple


def _table(tmp_path, monkeypatch, policy):
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_READ_POLICY", policy)
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_DECODE_WORKERS", "2")
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_PREFILL_WORKERS", "2")
    monkeypatch.setattr(ple, "READ_TIMEOUT_SECONDS", 0.03)
    path = tmp_path / "rows.bin"
    path.write_bytes(bytes(1024 * 20))
    if policy == "adaptive":
        monkeypatch.setattr(ple.FileBackedShardedEmbedding, "_measure_read_arms",
                            lambda *_args: (100, 200))
    return ple.FileBackedShardedEmbedding(str(path), 1024, 32, 2)


def _block_reads(monkeypatch):
    gate, started, finished = threading.Event(), threading.Event(), threading.Event()
    real = os.pread
    fds = []

    def blocked(fd, length, offset):
        fds.append(fd)
        started.set()
        assert gate.wait(5)
        result = real(fd, length, offset)  # fails if close reused the descriptor
        finished.set()
        return result

    monkeypatch.setattr(ple.os, "pread", blocked)
    return gate, started, finished, fds


def _eventually(predicate):
    end = time.monotonic() + 2
    while not predicate() and time.monotonic() < end:
        threading.Event().wait(0.002)
    assert predicate()


@pytest.mark.parametrize("policy", ["pooled", "serial"])
def test_ple_foreground_timeout_retains_descriptor_through_close(tmp_path, monkeypatch, policy):
    table = _table(tmp_path, monkeypatch, policy)
    gate, started, finished, fds = _block_reads(monkeypatch)
    try:
        with pytest.raises(ReadTimeout):
            table._pread_rows(np.array([0], dtype=np.int64), 1)
        assert started.is_set()
        began = time.monotonic()
        table.close()
        table.close()
        assert time.monotonic() - began < 0.5
        os.fstat(fds[0])
        with pytest.raises(ReadTimeout, match="not drained"):
            ReadPool(1, "replacement")
        gate.set()
        assert finished.wait(2)
    finally:
        gate.set()
        table.close()
    _eventually(lambda: not any(t.is_alive() for t in threading.enumerate()
                               if t.name.startswith("ple-nvme-")))


def test_optional_prefetch_has_a_deadline_without_a_waiting_caller(tmp_path, monkeypatch):
    table = _table(tmp_path, monkeypatch, "pooled")
    gate, started, finished, fds = _block_reads(monkeypatch)
    pool = table._prefetch_pool
    try:
        futures = table.prefetch_rows(np.array([0], dtype=np.int64))
        assert started.wait(1)
        _eventually(lambda: pool.failure is not None)
        assert table.prefetch_rows(np.array([1], dtype=np.int64)) == []
        table.close()
        os.fstat(fds[0])
        gate.set()
        assert finished.wait(2)
        _eventually(lambda: all(f.done() for f in futures))
    finally:
        gate.set()
        table.close()


def test_adaptive_warm_timeout_does_not_trap_close_or_publish_completion(tmp_path, monkeypatch):
    table = _table(tmp_path, monkeypatch, "adaptive")
    gate, started, finished, fds = _block_reads(monkeypatch)
    try:
        assert table.start_adaptive_warm()
        assert started.wait(1)
        _eventually(lambda: table._warm_pool.failure is not None)
        assert not table.wait_for_adaptive_warm(0.001)
        began = time.monotonic()
        table.close()
        assert time.monotonic() - began < 0.5
        os.fstat(fds[0])
        gate.set()
        assert finished.wait(2)
        _eventually(lambda: table._warm_future.done())
        assert not table._warm_refresh_pending
        assert table._warm_state == "stopped"
    finally:
        gate.set()
        table.close()


def test_prefetch_hash_callbacks_never_construct_or_evaluate_device_arrays(monkeypatch):
    from types import SimpleNamespace

    import mlx.core as mx

    from mlx2.runtime.models.qwen4_exp import NGramEmbedding

    args = SimpleNamespace(ngram_size=3, heads_per_ngram=1, eos_token_id=2,
                           ngram_vocab_size_base=11, make_ngram_vocab_size_divisible_by=2,
                           vocab_size=64, seed=7, split_ngram_parts=2)
    embedding = NGramEmbedding(args, 8, 0, 0)
    tokens = np.array([[4, 5, 2, 6]], dtype=np.int64)
    previous = np.array([[8, 9]], dtype=np.int64)
    expected = embedding._ngram_ids_numpy(mx.array(tokens), previous=previous)
    callbacks, observed = [], []
    embedding.ngram_embedding = SimpleNamespace(
        is_file_backed=True, submit_prefetch=callbacks.append,
        prefetch_rows=lambda rows: observed.append(rows),
        prefetch_dequant_rows=lambda rows: observed.append(rows),
    )
    embedding.prefetch_prompt_chunk(tokens, previous)
    assert embedding.prefetch_positions(previous, tokens)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("prefetch worker touched MLX")

    monkeypatch.setattr(mx, "array", forbidden)
    monkeypatch.setattr(mx, "eval", forbidden)
    pool = ReadPool(1, "host-prefetch")
    try:
        for callback in callbacks:
            pool.submit(callback).result(timeout=2)
        assert len(observed) == 2
        for rows in observed:
            assert np.array_equal(rows, expected)
    finally:
        pool.close()


@pytest.mark.parametrize("dequant", [False, True])
def test_late_prefetch_without_close_cannot_publish_packed_or_dequant_rows(tmp_path, monkeypatch, dequant):
    monkeypatch.setenv("MLX_QWEN4_PLE_NVME_LRU_MB", "0.01")
    table = _table(tmp_path, monkeypatch, "pooled")
    gate, started, finished, _fds = _block_reads(monkeypatch)
    pool = table._prefetch_pool
    try:
        if dequant:
            assert table.prefetch_dequant_rows(np.array([0, 1], dtype=np.int64)) == 2
        else:
            assert table.prefetch_rows(np.array([0, 1], dtype=np.int64))
        assert started.wait(1)
        _eventually(lambda: pool.failure is not None)
        gate.set()
        assert finished.wait(2)
        _eventually(lambda: pool.counters()["weight_read_inflight"] == 0)
        assert not table._closed
        assert not table._lru and not table._dq_cache
        assert table.prefetch_dequant_rows(np.array([0], dtype=np.int64)) == 0
    finally:
        gate.set()
        table.close()
