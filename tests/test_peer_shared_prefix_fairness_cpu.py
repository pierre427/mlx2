"""Real serving progress under cold shared-prefix and unrelated tenant traffic."""

from collections import defaultdict
from itertools import pairwise
from threading import Event

import pytest
from test_06_junction_snapshots import _adapter
from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen38_mtp
from test_apcv2_inflight_prefix_wait import collect, conversation

from mlx2 import memory, serving
from mlx2.runtime import os_memory
from mlx2.runtime.generate import BatchGenerator


@pytest.mark.parametrize("mtp", [False, True], ids=["ordinary", "native"])
@pytest.mark.parametrize("srpt", [False, True], ids=["default", "srpt"])
def test_shared_prefix_waiters_leave_unrelated_tenant_and_leader_progress(monkeypatch, mtp, srpt):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "cpu-fairness"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    model, vocab = tiny_qwen38_mtp()
    submitted = Event()
    round_number = 0
    generated_rounds = defaultdict(list)
    prompt_rounds = defaultdict(list)
    original_next = BatchGenerator.next

    def next_round(batch):
        nonlocal round_number
        # Queue all competitors before allowing execution. The production
        # admission, prefix wait, generator and publication paths remain live.
        assert submitted.wait(10)
        round_number += 1
        prompts, generated = original_next(batch)
        for row in prompts:
            prompt_rounds[row.uid].append(round_number)
        for row in generated:
            generated_rounds[row.uid].append(round_number)
        return prompts, generated

    monkeypatch.setattr(BatchGenerator, "next", next_round)
    policy = {"apc_inflight_prefix_wait": {"enabled": True, "min_shared_tokens": 64}}
    if srpt:
        policy["prefill_scheduling"] = {"order": "srpt", "max_bypass": 3}
    engine = serving.ServingEngine(
        "tiny", adapter_factory=_adapter(model, vocab), qualification_mode=True,
        mtp=mtp, max_lanes=4, max_inflight=8, prefill_step=16,
        coalesce_window_ms=100, tenant_scoped_cache=True, execution_policy=policy,
    )
    try:
        assert engine.ready.wait(60), engine.error
        family = [
            engine.submit(
                {"tokens": conversation(vocab, seed, shared_tokens=384),
                 "max_tokens": 6, "temperature": 0}, tenant_id="alice",
            )
            for seed in (5, 9, 13, 17)
        ]
        outsider = engine.submit(
            {"tokens": [3, 5, 7, 9, 11, 13, 15], "max_tokens": 20,
             "temperature": 0}, tenant_id="bob",
        )
        competitors = [
            engine.submit(
                {"tokens": [seed, 7, 9, 11, 13, 15], "max_tokens": 6,
                 "temperature": 0}, tenant_id=f"short-{seed}",
            )
            for seed in (19, 23, 29)
        ]
        submitted.set()
        jobs = family + [outsider] + competitors
        results = [collect(job) for job in jobs]
        assert all("error" not in receipt for _tokens, receipt in results), results
        assert [len(tokens) for tokens, _receipt in results] == [6] * 4 + [20] + [6] * 3
        assert not outsider.apc_prefix_wait_done and outsider.cached_tokens == 0
        assert all(not job.apc_prefix_wait_done for job in competitors)
        followers = [job for job in family if job.apc_prefix_wait_done]
        assert len(followers) == 3
        assert all(job.cached_tokens == 384 for job in followers)
        assert all(job.apc_prefix_wait_reuse == "checkpoint_reused" for job in followers)
        leader = next(job for job in family if not job.apc_prefix_wait_done)
        bob_rounds = sorted(set(generated_rounds[outsider.uid]))
        assert bob_rounds and bob_rounds[0] <= 8
        assert bob_rounds[0] < min(generated_rounds[leader.uid])
        # Competing prefill and checkpoint release must not stop the already
        # decoding tenant. Use scheduler rounds, not wall-clock timing claims.
        assert max(b - a for a, b in pairwise(bob_rounds)) <= 4
        assert prompt_rounds[leader.uid]
        assert all(generated_rounds[job.uid] for job in jobs)
        assert not engine.apc_inflight_waiters
        assert not engine.apc_inflight_resolutions
    finally:
        submitted.set()
        engine.close()
