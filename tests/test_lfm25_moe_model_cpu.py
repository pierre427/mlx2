"""Small CPU numerical checks for the LFM2.5 MoE reference model."""

import mlx.core as mx

from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.models.lfm25_moe import Model, ModelArgs, SparseMoE

mx.set_default_device(mx.cpu)


def _args():
    return ModelArgs(
        model_type="lfm2_moe",
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=6,
        num_hidden_layers=4,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=128,
        use_expert_bias=True,
        num_dense_layers=1,
        norm_eps=1e-5,
        conv_bias=False,
        conv_L_cache=3,
        tie_word_embeddings=True,
        routed_scaling_factor=1.0,
        rope_parameters={"rope_theta": 10_000.0},
        layer_types=["conv", "full_attention", "conv", "full_attention"],
    )


def test_hybrid_cache_and_ordinary_decode_are_finite():
    model = Model(_args())
    cache = model.make_cache()
    assert [type(item) for item in cache] == [
        ArraysCache,
        KVCache,
        ArraysCache,
        KVCache,
    ]
    logits = model(mx.array([[1, 2, 3]]), cache=cache)
    mx.eval(logits, [item.state for item in cache])
    assert logits.shape == (1, 3, 32)
    assert bool(mx.all(mx.isfinite(logits)).item())
    assert cache[0][0].shape == (1, 2, 8)
    next_logits = model(mx.array([[4]]), cache=cache)
    mx.eval(next_logits, [item.state for item in cache])
    assert next_logits.shape == (1, 1, 32)
    assert cache[1].offset == 4


def test_expert_bias_changes_selection_not_mixture_weights():
    moe = SparseMoE(_args())
    moe.gate.weight = mx.zeros_like(moe.gate.weight)
    moe.expert_bias = mx.array([0.0, 0.0, 10.0, 20.0])
    indices, scores = moe.routing(mx.zeros((1, 1, 8)))
    mx.eval(indices, scores)
    assert set(indices.reshape(-1).tolist()) == {2, 3}
    assert mx.allclose(scores, mx.array([[[0.4999995, 0.4999995]]]), atol=1e-6)


def test_layer_order_and_attention_index_validation():
    values = vars(_args()).copy()
    values["full_attn_idxs"] = [0, 1]
    try:
        ModelArgs(**values)
    except ValueError as error:
        assert "attention indices" in str(error)
    else:
        raise AssertionError("inconsistent attention indices were accepted")


def _tiny_model(seed=7):
    mx.random.seed(seed)
    args = _args()
    args.vocab_size = 64
    args.hidden_size = 16
    args.num_hidden_layers = 6
    args.layer_types = ["conv", "conv", "full_attention", "conv", "full_attention", "conv"]
    args.full_attn_idxs = None
    args.__post_init__()
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _drain(gen, prompts, max_tokens, **extra):
    uids = gen.insert(prompts, max_tokens=[max_tokens] * len(prompts), **extra)
    tokens = {uid: [] for uid in uids}
    finished = 0
    try:
        for _ in range(4 * max_tokens * len(prompts) + 50):
            for response in gen.next()[1]:
                tokens[response.uid].append(int(response.token))
                finished += bool(response.finish_reason)
            if finished == len(prompts):
                break
    finally:
        gen.close()
    assert finished == len(prompts)
    return [tokens[uid] for uid in uids]


def _prompts():
    motif = [5, 9, 13, 21, 8, 3]
    return [
        (motif * 6)[:31],
        [(7 * i + 2) % 60 + 2 for i in range(12)] + motif * 3,
        [11, 4] + motif * 4 + [17],
    ]


def test_shortconv_rollback_restores_the_shorter_forward_exactly():
    model = _tiny_model()
    prefix, proposal = [3, 7, 11, 2, 9], [4, 15, 6, 22]
    for accepted in range(len(proposal) + 1):
        cache = model.make_cache()
        model(mx.array([prefix]), cache=cache)
        for item in cache:
            item.start_speculation()
        model(mx.array([proposal]), cache=cache)
        for item in cache:
            item.trim(len(proposal) - accepted)
            item.stop_speculation()
        reference = model.make_cache()
        model(mx.array([prefix + proposal[:accepted]]), cache=reference)
        for got, want in zip(cache, reference):
            if isinstance(got, ArraysCache):
                # Equal up to matmul row-count variance between a 4-row and an
                # ``accepted``-row forward; a wrong slice is off by O(1).
                assert mx.allclose(got[0], want[0], atol=1e-6, rtol=1e-6).item()
            else:
                assert got.offset == want.offset
        nxt = mx.array([[30]])
        assert mx.allclose(
            model(nxt, cache=cache), model(nxt, cache=reference), atol=1e-5
        ).item()


def test_shortconv_ragged_rollback_matches_each_row():
    model = _tiny_model()
    layer = next(item for item in model.layers if not item.is_attention_layer)
    conv = layer.conv
    x = mx.random.normal((2, 3, 16))
    step = mx.random.normal((2, 4, 16))
    cache = ArraysCache(size=1)
    conv(x, cache=cache)
    cache.start_speculation()
    conv(step, cache=cache)
    cache.trim_ragged([3, 1])
    for row, kept in enumerate((1, 3)):
        single = ArraysCache(size=1)
        conv(x[row : row + 1], cache=single)
        conv(step[row : row + 1, :kept], cache=single)
        assert mx.allclose(
            cache[0][row : row + 1], single[0], atol=1e-6, rtol=1e-6
        ).item()


def test_mixed_length_lanes_match_single_lane_greedy():
    from mlx2.runtime.generate import BatchGenerator

    model = _tiny_model()
    prompts = _prompts()
    batched = _drain(
        BatchGenerator(model, prefill_step_size=8, completion_batch_size=3,
                       prefill_batch_size=3),
        prompts, 12,
    )
    single = [
        _drain(
            BatchGenerator(model, prefill_step_size=8, completion_batch_size=1,
                           prefill_batch_size=1),
            [prompt], 12,
        )[0]
        for prompt in prompts
    ]
    assert batched == single


def _pld_prompts():
    # The tiny model degenerates to repeating one token; these n-grams make
    # lookup propose continuations it then rejects, so rollback must run.
    return [
        [5, 9, 13] + [54, 54, 7, 1, 2, 3, 54, 54, 54, 9, 9] * 2,
        [11, 4] + [54, 54, 7, 1, 2, 3, 54, 54, 54, 9, 9] * 2 + [54],
        [54, 54, 7, 1, 2, 3, 54, 54, 54, 9, 9] * 2 + [6, 54, 54],
    ]


def test_prompt_lookup_matches_greedy_with_rejections():
    from mlx2.runtime import pld
    from mlx2.runtime.generate import BatchGenerator

    model = _tiny_model()
    prompts = _pld_prompts()
    greedy = [
        _drain(
            BatchGenerator(model, prefill_step_size=8, completion_batch_size=1,
                           prefill_batch_size=1),
            [prompt], 16,
        )[0]
        for prompt in prompts
    ]

    def run(lanes, batch):
        gen = pld.PromptLookupBatchGenerator(
            model, completion_batch_size=lanes, prefill_step_size=8,
            prompt_lookup={}, stop_tokens=[],
        )
        tokens = _drain(gen, batch, 16)
        return tokens, gen.scheduler_stats

    single = [run(1, [prompt]) for prompt in prompts]
    assert [tokens[0] for tokens, _ in single] == greedy
    batched, stats = run(3, prompts)
    assert batched == greedy
    # One verify forward for the hybrid cohort: rejected tails are trimmed
    # from the ShortConv rollback records, not replayed.
    assert stats["pld_batched_max_width"] == 3
    for item in [stats] + [stats for _, stats in single]:
        assert item["pld_fallbacks"] == 0
        assert item["pld_failed_round_rebuilds"] == 0
    total = lambda key: sum(item[key] for item in [stats] + [s for _, s in single])
    assert total("pld_rollbacks") > 0
    assert total("pld_proposed") > total("pld_accepted")
