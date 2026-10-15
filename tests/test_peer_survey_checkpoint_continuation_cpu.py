"""Execute cold/warm continuations across hybrid checkpoint boundaries.

Real tiny GDN+KV target and optional native MTP, real APCv2 leases, CPU only.
Token parity and full-vocabulary first-row KL are checked separately. These
tests do not establish artifact qualification or physical-width equivalence.
"""

import mlx.core as mx
import numpy as np
import pytest
from route_harness import tiny_qwen38_mtp

from mlx2.runtime.apc_v2 import APCv2
from mlx2.runtime.generate import BatchGenerator


def _generator(model, *, native=False):
    return BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                          prefill_step_size=16,
                          self_mtp={"num_draft": 2, "persistent": True} if native else None)


def _drive(batch, prompt, *, count=4, **kwargs):
    uid = batch.insert([prompt], max_tokens=[count], **kwargs)[0]
    rows, boundary = [], None
    for _ in range(100):
        prompts, responses = batch.next()
        for response in prompts:
            if response.end_of_prompt:
                boundary = batch.pop_prompt_boundary(uid)
        for response in responses:
            rows.append(response)
            if response.finish_reason:
                return rows, boundary
    raise AssertionError("bounded CPU request stalled")


def _law(row):
    value = np.asarray(row.astype(mx.float32)).astype(np.float64)
    value -= value.max()
    law = np.exp(value)
    return law / law.sum()


def _assert_continuation(warm, cold):
    assert [row.token for row in warm] == [row.token for row in cold]
    p, q = _law(cold[0].logprobs), _law(warm[0].logprobs)
    assert np.sum(p * np.log(p / q)) < 1e-9
    assert np.max(np.abs(p - q)) < 1e-6


@pytest.mark.parametrize("shared", [15, 16, 17, 31, 32, 33])
def test_hybrid_restore_lands_at_reachable_boundary_and_executes_suffix(monkeypatch, shared):
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_STRIDE", "16")
    model, _ = tiny_qwen38_mtp()
    source = [(index * 7) % 100 + 1 for index in range(40)]
    query = source[:shared] + [109, 110, 111]
    apc = APCv2(max_size=8, layout_name="peer-survey-hybrid")
    producer, cold_batch, warm_batch = (_generator(model) for _ in range(3))
    hit = None
    try:
        _rows, boundary = _drive(producer, source)
        apc.store("revision-a", boundary["tokens"], boundary["target_cache"])
        hit = apc.lookup("revision-a", query)
        if shared < 16:
            assert not hit.hit  # no exact recurrent checkpoint before stride
            return
        assert hit.hit and hit.cached_tokens == (shared // 16) * 16
        assert not apc.lookup("revision-b", query).hit
        cold, _ = _drive(cold_batch, query)
        warm, _ = _drive(warm_batch, hit.remaining_tokens, caches=[hit.cache],
                         all_tokens=[query[:hit.cached_tokens]])
        _assert_continuation(warm, cold)
    finally:
        for batch in (producer, cold_batch, warm_batch):
            batch.close()
        if hit is not None and hit.cache is not None:
            hit.cache.close()
        assert apc.apc_stats["cow"]["active_leases"] == 0
        apc.close()


@pytest.mark.parametrize("budget", [1, 2, 3, 4])
@pytest.mark.parametrize("native", [False, True])
def test_terminal_partial_verify_cache_executes_as_ordinary_reference(budget, native):
    model, _ = tiny_qwen38_mtp()
    prompt = [3, 8, 21, 34, 55, 89]
    producer = _generator(model, native=native)
    cold_batch, warm_batch = (_generator(model) for _ in range(2))
    apc = APCv2(max_size=8, layout_name="peer-survey-terminal")
    hit = None
    try:
        rows, _ = _drive(producer, prompt, count=budget)
        end = rows[-1]
        assert end.finish_reason == "length"
        if native:
            assert end.mtp_receipt["route"] == "continuous_batched_self_mtp"
            if budget >= 3:
                assert end.mtp_receipt["stats"]["draft_proposed"] > 0
        # The final emitted token is pending, not materialized in this cache.
        query = end.all_tokens + [end.token, 117, 118]
        apc.store("revision-a", end.all_tokens, end.prompt_cache)
        hit = apc.lookup("revision-a", query)
        assert hit.hit and hit.cached_tokens == len(end.all_tokens)
        cold, _ = _drive(cold_batch, query)
        warm, _ = _drive(warm_batch, hit.remaining_tokens, caches=[hit.cache],
                         all_tokens=[query[:hit.cached_tokens]])
        _assert_continuation(warm, cold)
    finally:
        for batch in (producer, cold_batch, warm_batch):
            batch.close()
        if hit is not None and hit.cache is not None:
            hit.cache.close()
        assert apc.apc_stats["cow"]["active_leases"] == 0
        apc.close()
