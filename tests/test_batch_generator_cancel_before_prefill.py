"""omlx#5104 analogue on the real BatchGenerator: a request removed before
its prefill finished must leave nothing behind in the generator's queues.

omlx left an aborted request in its prefilling queue, so the scheduler never
went idle. Here a tiny qwen4_exp model runs chunked prefill (4 tokens a step),
and a lane is removed (a) before the first ``next()`` and (b) partway through
its chunked prefill. The other lane must still finish.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.generate import BatchGenerator

from test_batched_mtp import _tiny_qwen4_model


def _held(gen, uid):
    where = []
    if any(entry[0] == uid for entry in gen._unprocessed_sequences):
        where.append("unprocessed")
    if uid in getattr(gen._prompt_batch, "uids", ()):
        where.append("prompt_batch")
    if any(getattr(x, "uid", x[0] if isinstance(x, tuple) else x) == uid
           for x in gen._currently_processing):
        where.append("currently_processing")
    if uid in getattr(gen._generation_batch, "uids", ()):
        where.append("generation_batch")
    if uid in getattr(gen, "_prompt_boundaries", {}):
        where.append("prompt_boundaries")
    memory_queued = getattr(gen, "_memory_queued_prefill", None)
    if memory_queued and uid in memory_queued:
        where.append("memory_queued_prefill")
    return where


def _finish(gen, uid, steps=400):
    for _ in range(steps):
        _prompts, responses = gen.next()
        if any(r.uid == uid and r.finish_reason for r in responses):
            return
    raise AssertionError("surviving lane never finished")


@pytest.mark.parametrize("prefill_steps_before_cancel", [0, 2])
def test_remove_before_prefill_completes_leaves_no_orphan(prefill_steps_before_cancel):
    mx.random.seed(5)
    model = _tiny_qwen4_model()
    gen = BatchGenerator(model, prefill_step_size=4)
    try:
        long_prompt = [(7 * i + 3) % 60 + 1 for i in range(40)]
        short_prompt = [(5 * i + 2) % 60 + 1 for i in range(9)]
        victim, survivor = gen.insert([long_prompt, short_prompt], max_tokens=[4, 4])
        for _ in range(prefill_steps_before_cancel):
            gen.next()
        before = _held(gen, victim)
        # (a) still queued; (b) inside the chunked prompt batch.
        expected = "unprocessed" if prefill_steps_before_cancel == 0 else "prompt_batch"
        assert expected in before and "generation_batch" not in before, before
        gen.remove([victim])
        assert _held(gen, victim) == [], _held(gen, victim)
        _finish(gen, survivor)
        # Drained: nothing queued, nothing prefilling, so the caller sees idle.
        assert len(gen._unprocessed_sequences) == 0
        assert len(gen._prompt_batch) == 0
        # And the generator still admits and finishes new work.
        fresh = gen.insert([short_prompt], max_tokens=[2])[0]
        _finish(gen, fresh)
    finally:
        gen.close()
