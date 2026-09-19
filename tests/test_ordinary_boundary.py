import mlx.core as mx
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.apc_v2 import APCv2
from test_batched_mtp import _tiny_qwen4_model


def test_ordinary_prompt_checkpoint_reuses_exact_hybrid_state():
    mx.random.seed(73)
    model = _tiny_qwen4_model()
    apc = APCv2(layout_name=model.apc_v2_layout)
    key = apc.key("tiny", revision="test")
    batch = BatchGenerator(model, prefill_step_size=4, max_tokens=6)
    tokens = [1, 2, 3, 4, 5, 6, 7]

    def run(cache=None, history=None, prompt=tokens):
        uid = batch.insert(
            [prompt], max_tokens=[6], caches=[cache], all_tokens=[history or []]
        )[0]
        emitted = []
        for _ in range(32):
            prompts, responses = batch.next()
            for response in prompts:
                if response.end_of_prompt:
                    boundary = batch.pop_prompt_boundary(uid)
                    assert len(boundary["tokens"]) == len(tokens) - 1
                    apc.store(key, boundary["tokens"], boundary["target_cache"])
            for response in responses:
                emitted.append(response.token)
                if response.finish_reason:
                    return emitted
        raise AssertionError("no terminal response")

    try:
        reference = run()
        hit = apc.lookup(key, tokens)
        assert hit.cached_tokens == len(tokens) - 1
        assert (
            run(hit.cache, tokens[: hit.cached_tokens], hit.remaining_tokens)
            == reference
        )
        hit.cache.close()
    finally:
        batch.close()
        apc.clear()
