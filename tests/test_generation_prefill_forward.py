"""Only the last prompt token takes ``prefill_forward``; generated tokens decode.

Gemma 3n's adapter model exposes ``prefill_forward`` (outer-model input
embeddings) for prompt tokens and ``__call__`` (decode embedding scaling) for
generated ones; the wrong path changes greedy answers.  GenerationBatch keyed
the choice on its step counter, and the BatchGenerator's long-lived batch is
built empty and extended with lanes after their own first step, so its first
decode step fed the first generated token through ``prefill_forward``.
"""

import mlx.core as mx

from mlx2.runtime.generate import BatchGenerator
from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen38_mtp


def test_generated_tokens_never_take_the_prefill_forward():
    mx.set_default_device(mx.cpu)
    model, vocab = tiny_qwen38_mtp()
    single_token_prefills = []
    real_call = type(model).__call__

    def prefill_forward(inputs, cache=None, **kwargs):
        if inputs.shape[1] == 1:
            single_token_prefills.append(int(inputs[0, -1].item()))
        return real_call(model, inputs, cache=cache, **kwargs)

    object.__setattr__(model, "prefill_forward", prefill_forward)
    prompt = [(5 * i + 2) % (vocab - 2) + 1 for i in range(40)]
    generator = BatchGenerator(model, prefill_step_size=16)
    try:
        for _request in range(2):
            single_token_prefills.clear()
            uid = generator.insert([prompt], max_tokens=[4])[0]
            for _ in range(100):
                _prompts, responses = generator.next()
                if any(r.uid == uid and r.finish_reason for r in responses):
                    break
            # Just the last prompt token, on a fresh generator and after.
            assert single_token_prefills == [prompt[-1]]
    finally:
        generator.close()
