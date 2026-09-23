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


def _plain_greedy(model, prompt, count):
    from mlx2.runtime.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt]), cache=cache)
    tokens = []
    for _ in range(count):
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
        logits = model(mx.array([tokens[-1:]]), cache=cache)
    return tokens


def test_one_token_prompt_serves_without_killing_the_worker(monkeypatch):
    # A one-token prompt feeds its only token to the first decode step, so the
    # prompt boundary is captured from caches nothing was written to yet.
    from mlx2 import memory, serving
    from mlx2.runtime import os_memory
    from mlx2.serving import ServingEngine
    from test_approximate_kv_serving import make_adapter, run, tiny_model

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    model = tiny_model()
    engine = ServingEngine(
        "tiny",
        adapter_factory=make_adapter(model, operations=None),
        qualification_mode=True,
        mtp=False,
    )
    assert engine.ready.wait(30), engine.error
    try:
        single, _ = run(engine, [5], max_tokens=4)
        after, _ = run(engine, [7, 8, 9], max_tokens=4)
        alive, error = engine.thread.is_alive(), engine.error
    finally:
        engine.close()
    assert alive and error is None
    assert single == _plain_greedy(model, [5], 4)
    assert after == _plain_greedy(model, [7, 8, 9], 4)


def test_empty_batch_caches_extract_as_empty_single_caches():
    from mlx2.runtime.models.cache import (
        BatchKVCache,
        BatchRotatingQuantizedKVCache,
    )

    for batch in (BatchKVCache([0, 0]), BatchRotatingQuantizedKVCache(8, [0, 0])):
        extracted = batch.extract(1)
        assert extracted.keys is None and extracted.offset == 0
