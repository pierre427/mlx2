"""Plain concurrent requests may share one primed APCv2 checkpoint."""

from concurrent.futures import ThreadPoolExecutor
import time

import pytest

from mlx2 import memory, serving
from mlx2.runtime import os_memory
from mlx2.serving import ServingEngine
from mlx2.server import collect_nonstream_job

from test_apc_hits_hybrid_gdn_self_mtp import (
    make_adapter,
    run,
    tiny_qwen4_mtp,
)


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def _engine(model, vocab, *, mtp, cache_dir=None):
    adapter = make_adapter(model, vocab)
    adapter.max_context = 2048
    engine = ServingEngine(
        "tiny",
        adapter_factory=adapter,
        qualification_mode=True,
        mtp=mtp,
        max_lanes=2,
        max_inflight=4,
        max_context=2048,
        cache_dir=cache_dir,
        prefill_step=16,
        coalesce_window_ms=50,
    )
    assert engine.ready.wait(60), engine.error
    return engine


def _wait_attached(job):
    deadline = time.monotonic() + 30
    while job.uid is None and time.monotonic() < deadline:
        time.sleep(0.001)
    assert job.uid is not None and job.completion_tokens < 256


def _collect_pair(first, second, body):
    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(
            pool.map(
                lambda job: collect_nonstream_job(job, body, chat=False),
                (first, second),
            )
        )


def test_plain_concurrent_ordinary_requests_ignore_blocked_neighbor(
    host, monkeypatch
):
    """A disk-only neighbor must not hide the leased resident warm prefix."""
    model, vocab = tiny_qwen4_mtp()
    prompt = [(7 * index + 3) % (vocab - 2) + 1 for index in range(256)]
    engine = _engine(model, vocab, mtp=False)
    try:
        run(engine, prompt, max_tokens=8)
        body = {"tokens": list(prompt), "max_tokens": 256, "temperature": 0}
        first = engine.submit(body)
        _wait_attached(first)

        apc = engine.apc
        with apc._apc_lock:
            records = list(apc._entry_records_locked())
            resident = next(
                entry for _key, _tokens, entry in records
                if apc._entry_pinned(entry)
            )
            neighbor_key, _neighbor_tokens, neighbor = next(
                record for record in records
                if not apc._entry_pinned(record[2])
            )
            assert neighbor is not resident and neighbor.sidecar is None
            neighbor_bytes = int(neighbor.nbytes)
            neighbor.prompt_cache.close()
            neighbor.prompt_cache = []
            neighbor.nbytes = 0
            neighbor._apc_disk = {"target": "budget-deferred"}
            apc._n_bytes -= neighbor_bytes
        original_restore = apc._restore_entry_locked

        def restore(key, tokens, entry):
            if entry is neighbor:
                apc._disk_stats["restore_budget_deferrals"] += 1
                return None
            return original_restore(key, tokens, entry)

        monkeypatch.setattr(apc, "_restore_entry_locked", restore)
        shared = apc.lookup(neighbor_key, prompt)
        assert shared.hit and shared.cached_tokens == 255
        shared.cache.close()

        second = engine.submit(body)
        _collect_pair(first, second, body)
        assert [first.cached_tokens, second.cached_tokens] == [255, 255]
        assert engine.apc.apc_stats["idle_disk"]["restore_budget_deferrals"] >= 1
        assert not engine.error
    finally:
        engine.close()


def test_plain_concurrent_mtp_requests_reuse_sidecarless_target(host):
    """A target-only warm checkpoint falls back per lane, without cold prefill."""
    model, vocab = tiny_qwen4_mtp()
    prompt = [(11 * index + 5) % (vocab - 2) + 1 for index in range(256)]
    ordinary = _engine(model, vocab, mtp=False)
    try:
        expected, _receipt, _job = run(ordinary, prompt, max_tokens=64)
    finally:
        ordinary.close()
    engine = _engine(model, vocab, mtp=True)
    try:
        run(engine, prompt, max_tokens=8)
        apc = engine.apc
        with apc._apc_lock:
            warm = next(
                entry
                for _key, tokens, entry in apc._entry_records_locked()
                if len(tokens) == len(prompt) - 1
            )
            assert warm.sidecar is not None
            warm.sidecar = None

        body = {"tokens": list(prompt), "max_tokens": 64, "temperature": 0}
        first = engine.submit(body)
        second = engine.submit(body)
        results = _collect_pair(first, second, body)
        assert [first.cached_tokens, second.cached_tokens] == [255, 255]
        outputs = [
            [int(token) for token in result[0]["text"].split()]
            for result in results
        ]
        assert outputs == [expected, expected]
        assert engine.counts["mtp_sidecar_missing_plain_fallbacks"] == 2
        assert engine.counts["mtp_sidecar_missing_misses"] == 0
        assert not engine.error
    finally:
        engine.close()
