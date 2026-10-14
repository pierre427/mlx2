"""An APCv2 hit continues from the exact recurrent state a cold run passes through.

sglang #44195: prefix-cache checkpoints snapshotted a bf16 copy of the GDN
state, so the first decoded token changed on every cache hit.  mlx2 records
its prefill checkpoints from the live cache (``ArraysCache.state_checkpoint``
copies the slot, keeping its dtype) and restores by deep copy, so a hit whose
prompt chunks line up with the cold run's must reproduce the cold run's
log-probabilities bit for bit.

The prompt is one token longer than a multiple of ``prefill_step``: the
stored boundary is ``len(prompt) - 1``, so the warm request's remaining
chunks start on the cold run's chunk grid and only the stored state can make
the two differ.  The planted arm rounds every restored rank-4 recurrent leaf
through bf16 (the sglang defect) and must be caught. CPU only.
"""

import mlx.core as mx
import pytest

from mlx2.runtime import apc_v2
from mlx2.runtime.models import cache as cache_module
from test_apc_hits_hybrid_gdn_self_mtp import (
    host,  # noqa: F401 - fixture
    make_adapter,
    tiny_qwen38_mtp,
    tiny_qwen4_mtp,
)

PREFILL_STEP = 16


def _bf16_restore(original):
    def restore(prompt_cache):
        restored = original(prompt_cache)
        stack = list(restored)
        while stack:
            entry = stack.pop()
            children = getattr(entry, "caches", None)
            if children is not None:
                stack.extend(children)
                continue
            slots = getattr(entry, "cache", None)
            if isinstance(slots, list):
                for index, leaf in enumerate(slots):
                    if isinstance(leaf, mx.array) and leaf.ndim == 4:
                        slots[index] = leaf.astype(mx.bfloat16).astype(leaf.dtype)
        return restored

    return restore


def _engine(model, vocab):
    from mlx2.serving import ServingEngine

    adapter = make_adapter(model, vocab)
    tokenizer = type(adapter.tokenizer)
    tokenizer.convert_ids_to_tokens = lambda self, ids: [f"t{int(i)}" for i in ids]
    engine = ServingEngine(
        "tiny", adapter_factory=adapter, qualification_mode=True, mtp=False,
        max_lanes=1, prefill_step=PREFILL_STEP,
    )
    assert engine.ready.wait(60), engine.error
    return engine


def _run(engine, tokens):
    job = engine.submit({
        "tokens": list(tokens), "max_tokens": 4, "temperature": 0,
        "logprobs": True, "top_logprobs": 5,
    })
    rows = []
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            raise AssertionError(event)
        if "logprob" in event:
            row = event["logprob"]
            rows.append(
                (row["id"], row["logprob"],
                 tuple((t["id"], t["logprob"]) for t in row["top_logprobs"]))
            )
        if "finish_reason" in event:
            return rows, int(job.cached_tokens or 0)


def _hit_and_cold(factory):
    model, vocab = factory()
    prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(6 * PREFILL_STEP + 1)]
    extended = prompt + [(5 * i + 1) % (vocab - 2) + 1 for i in range(24)]
    warm = _engine(model, vocab)
    try:
        _run(warm, prompt)
        hit, cached = _run(warm, extended)
    finally:
        warm.close()
    cold_engine = _engine(model, vocab)
    try:
        cold, cold_cached = _run(cold_engine, extended)
    finally:
        cold_engine.close()
    assert cached == len(prompt) - 1 and cached % PREFILL_STEP == 0
    assert cold_cached == 0
    assert len(hit) == len(cold) == 4
    return hit, cold


@pytest.mark.parametrize(
    "factory", [tiny_qwen38_mtp, tiny_qwen4_mtp], ids=["qwen38", "qwen4_exp"]
)
def test_apcv2_hit_logprobs_are_bitwise_the_cold_run(host, factory):  # noqa: F811
    hit, cold = _hit_and_cold(factory)
    assert hit == cold


def test_bf16_rounded_restore_is_caught(host, monkeypatch):  # noqa: F811
    planted = _bf16_restore(cache_module._copy_prompt_cache_for_restore)
    monkeypatch.setattr(cache_module, "_copy_prompt_cache_for_restore", planted)
    monkeypatch.setattr(apc_v2, "_copy_prompt_cache_for_restore", planted)
    hit, cold = _hit_and_cold(tiny_qwen4_mtp)
    assert hit != cold
