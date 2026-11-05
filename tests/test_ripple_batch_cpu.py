"""Continuous batching on Ripple: batched greedy equals per-request greedy.

Left-padded lanes of different lengths with a four-token window, so batched
rotating caches wrap at different offsets and RoPE takes per-lane offsets.
FP32 activations keep the comparison exact; BF16 row-shape rounding is a
serving-level near-tie question, not this structural check.
"""
import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.generate import BatchGenerator
from test_ripple_pld_cpu import greedy, tiny


@pytest.mark.parametrize("quantize", [False, True])
@pytest.mark.parametrize("prefill_step", [3, 8, 2048])
def test_batched_lanes_match_per_request_greedy(prefill_step, quantize):
    model = tiny(quantize=quantize)
    prompts = [[3, 7, 11, 5, 9, 2, 8, 1, 4, 6, 10, 12, 13], [5, 1], [9, 9, 4, 2, 7, 3, 8]]
    count = 14
    expected = [greedy(model, prompt, count) for prompt in prompts]
    generator = BatchGenerator(model, max_tokens=count, prefill_step_size=prefill_step,
                               completion_batch_size=4, prefill_batch_size=4)
    uids = generator.insert(prompts, max_tokens=[count] * len(prompts))
    emitted = {uid: [] for uid in uids}
    for _ in range(400):
        responses = generator.next()
        responses = responses[1] if isinstance(responses, tuple) else responses
        for response in responses:
            emitted[response.uid].append(response.token)
        if all(len(tokens) >= count for tokens in emitted.values()):
            break
    assert [emitted[uid][:count] for uid in uids] == expected
