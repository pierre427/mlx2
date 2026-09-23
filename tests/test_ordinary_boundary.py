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


def _tiny_kv_heavy_qwen38():
    """Hybrid model whose full-attention K/V dominates its per-row state."""
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
        head_dim=128, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=2,
        mtp_num_hidden_layers=1, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=1 << 16,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    return model


def test_published_prompt_boundary_holds_only_its_own_row():
    """A boundary must not pin the whole ready batch it was extracted from.

    A short and a long prompt prefill together; the short one's boundary is
    published and everything else is dropped. The APC entry is accounted as
    one short row, so that is all it may keep alive (it used to hold the
    long row's padded K/V as well, ~36x its accounted bytes).
    """
    import gc

    from mlx2.runtime.apc_v2 import APCKey

    model = _tiny_kv_heavy_qwen38()
    lengths = (120, 1900)
    batch = BatchGenerator(
        model, completion_batch_size=4, prefill_batch_size=2,
        prefill_step_size=2048, prefill_batch_window=1,
    )
    uids = batch.insert(
        [[(i * (k + 3)) % 97 + 2 for i in range(n)] for k, n in enumerate(lengths)],
        max_tokens=[8] * len(lengths),
    )
    ended = 0
    while ended < len(lengths):
        (prompts, _responses) = batch.next()
        ended += sum(1 for p in prompts if p.end_of_prompt)
    boundaries = [batch.pop_prompt_boundary(uid) for uid in uids]
    batch.close()
    del batch
    short = boundaries[0]
    del boundaries
    accounted = sum(cache.nbytes for cache in short["target_cache"])
    apc = APCv2(
        max_size=1, layout_name=getattr(model, "apc_v2_layout", "qwen38-tiny")
    )
    apc.store(
        APCKey("model"), short["tokens"], short["target_cache"],
        retention_role="committed_prompt_boundary",
    )
    del short
    gc.collect()
    held_with_entry = mx.get_active_memory()
    apc.clear()
    del apc
    gc.collect()
    held = held_with_entry - mx.get_active_memory()
    assert accounted > 0
    assert held <= 1.25 * accounted, (held, accounted)


def test_state_checkpoint_does_not_pin_the_batched_state():
    """A surviving lane's checkpoint must keep only its own row alive.

    Checkpoints are cut per row from the batched recurrent state at a chunk
    boundary. Once the other lanes are gone, the survivor's snapshot is
    accounted as one row and must not still hold every row's state.
    """
    import gc

    from mlx2.runtime.models.cache import ArraysCache

    rows = 4
    cache = ArraysCache(2)
    cache.cache = [
        mx.random.normal((rows, 64, 1024), key=mx.random.key(k)) for k in range(2)
    ]
    mx.eval(cache.cache)
    cache.state_checkpoint([256] * rows, force=True)
    survivor = cache._checkpoints[0]
    accounted = sum(
        array.nbytes for (_position, snapshot) in survivor for array in snapshot
    )
    del cache
    gc.collect()
    mx.synchronize()
    held_with_survivor = mx.get_active_memory()
    del survivor
    gc.collect()
    held = held_with_survivor - mx.get_active_memory()
    assert accounted == 2 * 64 * 1024 * 4
    assert held <= 1.25 * accounted, (held, accounted)


def test_empty_caches_of_every_kind_report_empty_state():
    # _promote_ready_prompts evaluates the state of every extracted boundary
    # row; a one-token prompt prefills nothing, so each row is still empty.
    from mlx2.runtime.models.cache import (
        KVCache,
        QuantizedKVCache,
        RotatingKVCache,
        RotatingQuantizedKVCache,
    )

    for cache in (
        KVCache(),
        RotatingKVCache(max_size=8),
        QuantizedKVCache(),
        RotatingQuantizedKVCache(max_size=8),
    ):
        assert cache.state == (None, None), type(cache).__name__
        mx.async_eval([cache.state])


def test_one_token_prompt_on_a_sliding_window_model_keeps_the_generator_alive():
    # Sliding-window models (North, Muse) keep RotatingKVCache layers; the
    # plain-KV fix above did not cover them and the boundary capture raised.
    from test_pld_batched_verify import _north, _greedy

    model = _north()
    generator = BatchGenerator(model, max_tokens=4, completion_batch_size=2)
    (uid,) = generator.insert([[5]], max_tokens=[4])
    produced = []
    for _ in range(50):
        _prompts, responses = generator.next()
        produced += [r.token for r in responses if r.uid == uid]
        if any(r.finish_reason for r in responses if r.uid == uid):
            break
    assert produced == _greedy(model, [5], 4)
