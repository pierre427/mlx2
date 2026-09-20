"""APCv2 fanout (``n>1``) end-to-end through a fake batch runtime.

Regressions covered:

* siblings published after the leader's prompt boundary must join the
  leader's live batch, not be held as an atomic declared cohort;
* every sample's bounded event queue must be drained while the others decode,
  otherwise long completions cancel the siblings as slow consumers;
* a failing sample keeps its engine status instead of collapsing to 503.
"""

import time
from collections import Counter
from types import SimpleNamespace as NS

import pytest

from mlx2 import memory, serving
from mlx2.runtime import apc_v2, generate, os_memory
from mlx2.server import (
    SampleFailed,
    collect_nonstream_job,
    collect_parallel_samples,
)


def _engine(monkeypatch, log, *, tokens, store_fails=False, fanout_lookup_misses=False):
    state = {
        "stored": False,
        "stored_tokens": [],
        "fanout_lookups": 0,
        "branches": [],
    }

    class Branch(list):
        def __init__(self):
            super().__init__()
            self.closed = False
            state["branches"].append(self)

        def close(self):
            self.closed = True

    class APC:
        def __init__(self, **_kw):
            self.apc_stats = {}

        def key(self, *_a, **_kw):
            return "key"

        def lookup(self, *_a, **_kw):
            requested = list(_a[1])
            if state["stored"]:
                state["fanout_lookups"] += 1
                if fanout_lookup_misses == "raise":
                    raise ValueError("injected APC fanout lookup failure")
            miss = fanout_lookup_misses is True or (
                type(fanout_lookup_misses) is int
                and state["fanout_lookups"] >= fanout_lookup_misses
            )
            if state["stored"] and not miss:
                coverage = len(state["stored_tokens"])
                if fanout_lookup_misses == "short":
                    coverage = max(coverage - 1, 0)
                return NS(
                    cache=Branch(),
                    cached_tokens=coverage,
                    remaining_tokens=requested[coverage:],
                    miss_reason=None,
                    sidecar=None,
                )
            return NS(
                cache=None,
                cached_tokens=0,
                remaining_tokens=[1, 2],
                miss_reason=None,
                sidecar=None,
            )

        def store(self, *_a, **_kw):
            if store_fails:
                raise ValueError("injected APC store failure")
            state["stored"] = True
            state["stored_tokens"] = list(_a[1])
            return NS(stored=True)

        def spill_idle_entries(self):
            pass

        def evict_oldest_unleased(self):
            return False

        def clear(self):
            pass

    class Batch:
        scheduler_stats = {}

        def __init__(self, *_a, **_kw):
            self.lanes = {}
            self.uid = 0
            self.boundaries = {}

        def insert(self, *_a, **_kw):
            uid = self.uid
            self.uid += 1
            self.lanes[uid] = {"emitted": 0, "prompted": False}
            return [uid]

        def pop_prompt_boundary(self, uid):
            return self.boundaries.pop(uid, None)

        def next(self):
            time.sleep(0.001)
            prompts, responses = [], []
            log.append(sorted(self.lanes))
            for uid, lane in list(self.lanes.items()):
                if not lane["prompted"]:
                    lane["prompted"] = True
                    self.boundaries[uid] = {
                        "committed_only": True,
                        "tokens": [1, 2],
                        "target_cache": [],
                        "mtp_state": None,
                    }
                    prompts.append(NS(uid=uid, end_of_prompt=True))
                    continue
                lane["emitted"] += 1
                done = lane["emitted"] >= tokens
                responses.append(
                    NS(
                        uid=uid,
                        execution_width=len(self.lanes),
                        finish_reason="length" if done else None,
                        token=3,
                        mtp_state=None,
                        all_tokens=[1, 2, 3],
                        prompt_cache=[],
                        mtp_receipt=None,
                    )
                )
                if done:
                    del self.lanes[uid]
            return prompts, responses

        def remove(self, uids):
            for uid in uids:
                self.lanes.pop(uid, None)

        def close(self):
            pass

    class Detok:
        last_segment = "x"

        def reset(self):
            pass

        def add_token(self, _t):
            pass

        def finalize(self):
            pass

    class Parser:
        stopped = False
        tool_count = 0

        def push(self, text, final=False):
            return [{"content": text}]

    class Adapter:
        max_context = 100000
        identity = {"fingerprint": "fake"}
        environment = {}
        layout = "fake"
        model = None
        tokenizer = NS(vocab_size=100, eos_token_ids=[], detokenizer=Detok())

        def __init__(self, _path):
            pass

        def profile_name(self, _mtp):
            return "fake"

        def execution_config(self, **_kw):
            return {"num_draft": 0}

        def prompt_tokens(self, _request):
            return [1, 2]

        def output_parser(self, _request):
            return Parser()

        def diagnostics(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    engine = serving.ServingEngine(
        "fake",
        adapter_factory=Adapter,
        qualification_mode=True,
        mtp=False,
        max_lanes=4,
        max_inflight=8,
    )
    assert engine.ready.wait(5)
    engine._test_fanout_state = state
    return engine


@pytest.mark.parametrize("samples", [2, 3, 4])
def test_fanout_siblings_join_the_leaders_live_batch(monkeypatch, samples):
    log = []
    engine = _engine(monkeypatch, log, tokens=20)
    try:
        body = {"max_tokens": 20, "n": 1}
        jobs = engine.submit_many([dict(body, seed=i) for i in range(samples)])
        results = collect_parallel_samples(jobs, body, chat=True)
    finally:
        engine.close()
    assert len(results) == samples
    # Every sibling decoded alongside the leader at some point.
    assert max(len(lanes) for lanes in log) == samples
    assert engine.counts.get("batch_cohort_attachment_failures", 0) == 0
    assert engine.counts["apcv2_fanout_boundaries"] == 1
    for _, _, receipt in results:
        fanout = receipt["parallel_prefill"]
        assert fanout["one_prefill"] is True
        assert fanout["boundary_tokens"] == 2
        assert fanout["reason"] == "boundary_reused"


def test_write_suppressed_parallel_samples_prefill_independently(monkeypatch):
    log = []
    engine = _engine(monkeypatch, log, tokens=2)
    try:
        body = {
            "max_tokens": 2,
            "n": 1,
            "skip_writing_prefix_cache": True,
        }
        jobs = engine.submit_many([dict(body, seed=i) for i in range(3)])
        results = collect_parallel_samples(jobs, body, chat=True)
        status = engine.status()
    finally:
        engine.close()
    assert len(results) == 3
    assert engine._test_fanout_state["stored"] is False
    assert status["counts"]["apcv2_write_suppressed_requests"] == 3
    assert status["counts"]["apcv2_fanout_write_suppressed"] == 1
    assert all(
        receipt["request_controls"]["skip_writing_prefix_cache"]
        for _, _, receipt in results
    )


@pytest.mark.parametrize(
    ("store_fails", "fanout_lookup_misses", "counter"),
    [
        (True, False, "apcv2_fanout_store_failures"),
        (False, True, "apcv2_fanout_boundary_misses"),
        (False, "raise", "apcv2_fanout_boundary_misses"),
        (False, "short", "apcv2_fanout_boundary_misses"),
    ],
)
def test_fanout_fails_closed_without_a_leased_committed_boundary(
    monkeypatch, store_fails, fanout_lookup_misses, counter
):
    log = []
    engine = _engine(
        monkeypatch,
        log,
        tokens=4,
        store_fails=store_fails,
        fanout_lookup_misses=fanout_lookup_misses,
    )
    try:
        body = {"max_tokens": 4, "n": 1}
        jobs = engine.submit_many([dict(body, seed=i) for i in range(2)])
        with pytest.raises(SampleFailed, match="APCv2 fanout"):
            collect_parallel_samples(jobs, body, chat=True)
    finally:
        engine.close()
    assert engine.counts[counter] == 1
    assert engine.counts.get("apcv2_fanout_boundaries", 0) == 0


def test_fanout_partial_sibling_acquisition_is_all_or_none(monkeypatch):
    log = []
    engine = _engine(
        monkeypatch, log, tokens=4, fanout_lookup_misses=2
    )
    try:
        body = {"max_tokens": 4, "n": 1}
        jobs = engine.submit_many([dict(body, seed=i) for i in range(3)])
        with pytest.raises(SampleFailed, match="APCv2 fanout"):
            collect_parallel_samples(jobs, body, chat=True)
    finally:
        engine.close()
    assert engine.counts["apcv2_fanout_boundary_misses"] == 1
    assert engine.counts.get("apcv2_fanout_boundaries", 0) == 0
    assert engine._test_fanout_state["branches"]
    assert all(branch.closed for branch in engine._test_fanout_state["branches"])


def test_fanout_samples_longer_than_the_event_queue_complete(monkeypatch):
    log = []
    tokens = serving.Job({}).events.maxsize + 64
    engine = _engine(monkeypatch, log, tokens=tokens)
    try:
        body = {"max_tokens": tokens, "n": 1}
        jobs = engine.submit_many([dict(body, seed=i) for i in range(2)])
        results = collect_parallel_samples(jobs, body, chat=True)
    finally:
        engine.close()
    assert [usage["completion_tokens"] for _, usage, _ in results] == [tokens, tokens]
    assert engine.counts.get("cancelled", 0) == 0


def test_fanout_sequential_collection_would_overflow_the_sibling(monkeypatch):
    """Documents why the samples are drained concurrently."""
    log = []
    tokens = serving.Job({}).events.maxsize + 64
    engine = _engine(monkeypatch, log, tokens=tokens)
    try:
        body = {"max_tokens": tokens, "n": 1}
        jobs = engine.submit_many([dict(body, seed=i) for i in range(2)])
        collect_nonstream_job(jobs[0], body, chat=True)
        with pytest.raises(SampleFailed, match="did not consume output"):
            collect_nonstream_job(jobs[1], body, chat=True)
    finally:
        engine.close()


def test_fanout_failure_keeps_the_engine_status(monkeypatch):
    log = []
    engine = _engine(monkeypatch, log, tokens=20)
    try:
        # prompt (2 tokens) + max_tokens exceeds the request's own ceiling.
        body = {"max_tokens": 20, "n": 1, "context_limit": 8}
        jobs = engine.submit_many([dict(body, seed=i) for i in range(2)])
        with pytest.raises(SampleFailed) as failure:
            collect_parallel_samples(jobs, body, chat=True)
    finally:
        engine.close()
    assert failure.value.status == 400
    assert failure.value.code == "context_length_exceeded"
    assert "maximum context length is 8 tokens" in str(failure.value)


def test_top_k_equal_to_vocab_size_is_a_request_error_not_a_worker_death(monkeypatch):
    log = []
    engine = _engine(monkeypatch, log, tokens=4)
    try:
        body = {"prompt": "hi", "max_tokens": 4, "top_k": 100, "temperature": 0.7}
        job = engine.submit(body)
        with pytest.raises(SampleFailed) as failure:
            collect_nonstream_job(job, body, chat=False)
        assert failure.value.status == 400
        assert "must fit the model vocabulary" in str(failure.value)
        # The worker is still alive and serves the next request.
        ok = engine.submit({"prompt": "hi", "max_tokens": 4, "top_k": 99})
        collect_nonstream_job(ok, {"max_tokens": 4}, chat=False)
        assert engine.error is None
    finally:
        engine.close()


def test_parallel_sample_guard_waits_for_headroom_before_refusing(monkeypatch):
    from mlx2 import memory, serving
    import mlx.core as mx

    engine = serving.ServingEngine.__new__(serving.ServingEngine)
    engine.max_lanes = 4
    engine.queued_jobs = 0
    engine.counts = Counter()
    # The guard charges the host-scaled service/driver reserve; pin it to the
    # 128 GiB calibration value so this test does not depend on the test host.
    engine._hard_reserve_gib = 20.0
    rejected = []
    engine.batch_metrics = type("M", (), {"rejected": lambda self, *a: rejected.append(a)})()
    readings = iter([21, 22, 25])  # GiB: two short readings, then enough for n=2
    monkeypatch.setattr(memory, "execution_headroom", lambda: next(readings) << 30)
    monkeypatch.setattr(serving.time, "sleep", lambda _s: None)
    receipt = engine.admit_parallel_samples(2)
    assert receipt["required_headroom_bytes"] == 24 << 30 and not rejected
    monkeypatch.setattr(serving.ServingEngine, "PARALLEL_SAMPLE_WAIT_SECONDS", 0.0)

    state = {"free": 23, "reclaims": 0, "recover": True}
    monkeypatch.setattr(
        memory, "execution_headroom", lambda: state["free"] << 30
    )

    def clear_cache():
        state["reclaims"] += 1
        if state["recover"]:
            state["free"] = 25

    monkeypatch.setattr(mx, "clear_cache", clear_cache)
    receipt = engine.admit_parallel_samples(2)
    assert receipt["required_headroom_bytes"] == 24 << 30
    assert state["reclaims"] == 1
    assert engine.counts["memory_cache_reclaims_before_reject"] == 1

    state.update(free=23, recover=False)
    engine._memory_reclaim_last = 0.0
    with pytest.raises(serving.Overloaded, match="physical footprint guard"):
        engine.admit_parallel_samples(2)
    assert state["reclaims"] == 2
    assert engine.counts["memory_cache_reclaims_before_reject"] == 2
    assert len(rejected) == 1
    with pytest.raises(serving.Overloaded, match="lane capacity"):
        engine.admit_parallel_samples(5)
