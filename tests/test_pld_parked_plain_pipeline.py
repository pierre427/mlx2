"""Parked PLD lanes on attention-only targets pipeline their plain rounds.

GPU evidence (gpt-oss-20b, M5 Max, B1 greedy, 2026-10-21): ``--ordinary``
decoded 116/116/105 tok/s on code/chat/7K prompts, the default adaptive
``--prompt-lookup`` 102/102/94, with zero proposals.  Every round was a
cost-latched parked "direct" round (``pld_parked_direct_rounds`` 1774,
``pld_pipelined_rounds`` 0): the pipelined plain round was open only to
hybrid lanes without a cost latch, so a parked attention-only lane built its
forward, waited for it, and only then built the next, while ordinary decode
builds step k+1 while the device runs step k.

A lone lane whose cost latch is parked now pipelines its plain rounds on its
own caches (``PlainKVStep`` for plain/rotating KV planes); the latch keeps
timing them (between reads), shadow-probes on the host, and drains the
pipeline when it leaves ``parked``.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.generate import StopSequenceMatcher
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.plain_kv_step import PlainKVStep, is_plain_kv_row
from mlx2.runtime.pld import PromptLookupBatchGenerator

# server.DEFAULT_PROMPT_LOOKUP_POLICY: the policy a serving route with no
# drafter gets.
DEFAULT_POLICY = {
    "cost_aware_admission": True,
    "cost_shadow_span": 4,
    "cost_probe_stride": 4,
    "cost_shadow_window": 2,
    "cost_shadow_gate": 0.75,
    "cost_plain_rounds": 8,
    "cost_explore_rounds": 2,
    "cost_margin": 0.05,
    "cost_reprobe_interval": 32,
    "cost_park_rounds": 4,
}

WINDOW = 8


def _tiny_gpt_oss(window=WINDOW):
    from mlx.utils import tree_map

    from mlx2.runtime.models.gpt_oss import Model, ModelArgs

    args = ModelArgs(
        num_hidden_layers=4, num_local_experts=4, num_experts_per_tok=2,
        vocab_size=64, hidden_size=64, intermediate_size=64, head_dim=8,
        num_attention_heads=4, num_key_value_heads=2, sliding_window=window,
    )
    mx.random.seed(3)
    model = Model(args)
    model.eval()
    for layer in model.model.layers:
        layer.self_attn.sinks = mx.random.normal((args.num_attention_heads,))
    # Scaled weights keep the tiny model's greedy output from collapsing
    # into a one-token loop (which the lookup would copy).
    model.update(tree_map(lambda x: x * 3.0, model.parameters()))
    mx.eval(model.parameters())
    return model


# No repeated n-gram: the lookup has nothing to copy.
NOVEL = [[(13 * i + 29 * lane + 5) % 60 + 1 for i in range(21)] for lane in range(3)]


def _greedy(model, prompt, count):
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    tokens = []
    for _ in range(count):
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)
    return tokens


def _ordinary(model, prompt, count):
    from mlx2.runtime.generate import BatchGenerator

    batch = BatchGenerator(
        model, completion_batch_size=1, prefill_batch_size=1, prefill_step_size=64
    )
    uid = batch.insert([prompt], max_tokens=[count])[0]
    tokens = []
    try:
        for _ in range(4 * count + 16):
            result = batch.next()
            responses = result[1] if isinstance(result, tuple) else result
            for response in responses:
                if response.uid == uid:
                    tokens.append(response.token)
            if len(tokens) >= count:
                break
    finally:
        batch.close()
    return tokens


class _Forwards:
    def __init__(self, model):
        self.model = model
        self.widths = []

    def __call__(self, tokens, cache=None, **kwargs):
        self.widths.append(tuple(tokens.shape))
        return self.model(tokens, cache=cache, **kwargs)

    def __getattr__(self, name):
        return getattr(self.model, name)


def _run(model, prompts, max_tokens, *, stop=None, insert_later=None, **policy):
    forwards = _Forwards(model)
    generator = PromptLookupBatchGenerator(
        forwards, completion_batch_size=4, prefill_step_size=64,
        prompt_lookup={**DEFAULT_POLICY, **policy},
    )
    uids = generator.insert(
        prompts, max_tokens=list(max_tokens),
        caches=[model.make_cache() for _ in prompts],
        stop_matchers=None if stop is None else [StopSequenceMatcher(s) for s in stop],
    )
    tokens = {uid: [] for uid in uids}
    finals = {}
    try:
        for poll in range(2000):
            if insert_later is not None and poll == insert_later[0]:
                more = generator.insert(
                    [insert_later[1]], max_tokens=[insert_later[2]],
                    caches=[model.make_cache()],
                )
                uids.extend(more)
                tokens.update({uid: [] for uid in more})
            _prompts, responses = generator.next()
            for response in responses:
                tokens[response.uid].append(response.token)
                if response.finish_reason:
                    finals[response.uid] = response
            if not generator.lanes:
                break
        stats = dict(generator.scheduler_stats)
    finally:
        generator.close()
    return [tokens[uid] for uid in uids], [finals.get(uid) for uid in uids], stats, forwards


# ------------------------------------------------------- the plain KV step
def _kv(shape_len, seed):
    mx.random.seed(seed)
    return (
        mx.random.normal((1, 2, shape_len, 4)),
        mx.random.normal((1, 2, shape_len, 4)),
    )


def _build(kind, history):
    cache = KVCache() if kind == "kv" else RotatingKVCache(
        max_size=WINDOW, keep=1 if kind == "sink" else 0
    )
    for index, length in enumerate(history):
        cache.update_and_fetch(*_kv(length, 100 + index))
    return cache


RING_PHASES = {
    "kv-mid-buffer": ("kv", [5, 1, 1]),
    "kv-buffer-edge": ("kv", [255]),  # the step grows the buffer
    "ring-filling": ("ring", [3, 1]),
    "ring-reaches-window": ("ring", [WINDOW - 1]),
    "ring-wrapped-mid": ("ring", [WINDOW, 1, 1, 1]),
    "ring-wraps-on-step": ("ring", [WINDOW - 1, 1]),
    "ring-post-prefill-trim": ("ring", [WINDOW + 5]),  # 13-row buffer
    "sink-wrapped": ("sink", [WINDOW + 3, 1, 1]),
}


@pytest.mark.parametrize("phase", sorted(RING_PHASES))
def test_released_step_leaves_every_later_append_bitwise_unchanged(phase):
    kind, history = RING_PHASES[phase]
    stepped, reference = _build(kind, history), _build(kind, history)
    step = PlainKVStep([[stepped]])
    stepped.update_and_fetch(*_kv(1, 7))  # the pipelined, later released, row
    step.release_row(0)
    step.abort()  # a released row is not rewound twice
    assert stepped.offset == reference.offset
    if phase != "ring-post-prefill-trim":  # that buffer is a window now
        assert getattr(stepped, "_idx", None) == getattr(reference, "_idx", None)
    # Single-token, multi-token, then single-token appends: what attention
    # reads after each must be bitwise what it reads without the undone step.
    for index, length in enumerate((1, 3, 1, 1)):
        update = _kv(length, 200 + index)
        got = stepped.update_and_fetch(*update)
        want = reference.update_and_fetch(*update)
        assert stepped.offset == reference.offset
        for a, b in zip(got, want):
            assert a.shape == b.shape and bool(mx.array_equal(a, b))


def test_commit_requires_every_plane_advanced_and_abort_rewinds_a_partial_forward():
    planes = [_build("kv", [WINDOW + 2]), _build("ring", [WINDOW + 2])]
    step = PlainKVStep([planes])
    planes[0].update_and_fetch(*_kv(1, 9))  # the forward failed before plane 1
    with pytest.raises(RuntimeError, match="finish the plain step"):
        step.commit([1])
    assert step.closed and planes[0].offset == planes[1].offset == WINDOW + 2


def test_plain_step_owns_only_b1_plain_and_rotating_kv_rows():
    from mlx2.runtime.models.cache import ArraysCache, CacheList, QuantizedKVCache

    assert is_plain_kv_row([KVCache(), RotatingKVCache(max_size=4)])
    assert not is_plain_kv_row([KVCache(), ArraysCache(size=2)])
    assert not is_plain_kv_row([CacheList(KVCache())])
    assert not is_plain_kv_row([QuantizedKVCache()])
    ring = RotatingKVCache(max_size=4)
    ring.start_speculation()
    assert not is_plain_kv_row([ring])
    with pytest.raises(TypeError):
        PlainKVStep([[KVCache()], [KVCache()]])


# ---------------------------------------------- live prompt-lookup lanes
def test_parked_attention_only_lane_pipelines_and_matches_ordinary_decode():
    model = _tiny_gpt_oss()
    count = 40
    outputs, finals, stats, forwards = _run(model, NOVEL[:1], [count])
    assert outputs[0] == _greedy(model, NOVEL[0], count)
    assert outputs[0] == _ordinary(model, NOVEL[0], count)
    # Every decode round was pipelined: none ran the serial parked driver.
    assert stats["pld_parked_direct_rounds"] == 0
    assert stats["pld_pipelined_rounds"] == stats["pld_cycles"] == count
    assert stats["pld_proposed"] == 0
    # One prefill forward, then exactly one B1 one-token forward per round.
    assert forwards.widths[1:] == [(1, 1)] * count
    receipt = finals[0].speculative_receipt
    assert receipt["pipelined_plain_rounds"] == count
    assert receipt["current_target_path"] == "plain_decode"
    # The latch still timed its plain rounds and shadow-probed on the host.
    latch = receipt["cost_latch"]
    assert latch["state"] == "parked" and latch["plain_tokens"] == count
    assert latch["plain_ns"] > 0 and latch["shadow_lookups"] > 0
    # The frozen boundary is exactly the committed history.
    cache = finals[0].prompt_cache
    assert all(c.offset == len(finals[0].all_tokens) for c in cache)


def test_disabled_pipeline_keeps_the_serial_parked_driver():
    model = _tiny_gpt_oss()
    outputs, _finals, stats, _forwards = _run(
        model, NOVEL[:1], [24], pipelined_plain=False
    )
    assert outputs[0] == _greedy(model, NOVEL[0], 24)
    assert stats["pld_pipelined_rounds"] == 0
    assert stats["pld_parked_direct_rounds"] == 24


def test_stop_on_a_read_round_rewinds_the_wrapped_ring_exactly():
    """The lane stops on a round whose successor was already dispatched: that
    row is released from the wrapped sliding-window ring before the lane's
    cache is frozen, and decoding on from the frozen cache is bitwise what
    decoding on from an ordinary cache at that boundary gives."""
    model = _tiny_gpt_oss()
    reference = _greedy(model, NOVEL[1], 40)
    stop_at = next(i for i in range(14, 40) if reference[i] not in reference[:i])
    outputs, finals, stats, _forwards = _run(
        model, NOVEL[1:2], [40], stop=[[[reference[stop_at]]]]
    )
    assert outputs[0] == reference[: stop_at + 1]
    assert stats["pld_pipeline_released_rows"] == 1
    frozen = finals[0].prompt_cache
    history = finals[0].all_tokens
    assert history == NOVEL[1] + reference[:stop_at]
    assert any(type(c) is RotatingKVCache and c.offset > c.max_size for c in frozen)
    # The lane's own schedule without the released step: the prompt but its
    # last token in one prefill forward, then one-token decode steps.
    ordinary = model.make_cache()
    model(mx.array([NOVEL[1][:-1]]), cache=ordinary)
    for token in history[len(NOVEL[1]) - 1 :]:
        model(mx.array([[token]]), cache=ordinary)
    assert [c.offset for c in frozen] == [c.offset for c in ordinary]
    for inputs in ([reference[stop_at]], [3], [5, 9, 11], [17], [42]):
        got = model(mx.array([inputs]), cache=frozen)
        want = model(mx.array([inputs]), cache=ordinary)
        assert bool(mx.array_equal(got, want))


class _CycleAfter:
    """KV + sliding ring; novel output until ``start``, then a copyable cycle."""

    def __init__(self, start):
        self.start = start

    def make_cache(self):
        return [KVCache(), RotatingKVCache(max_size=WINDOW)]

    def __call__(self, tokens, *, cache):
        first = int(cache[0].offset)
        values = tokens.astype(mx.float32)[:, None, :, None]
        cache[0].update_and_fetch(values, values)
        cache[1].update_and_fetch(values + 100.0, values + 200.0)
        positions = mx.arange(first, first + tokens.shape[1])[None]
        cyclic = tokens % 3 + 1
        novel = (positions * 7 + 11) % 50 + 10
        predicted = mx.where(positions >= self.start, cyclic, novel)
        return mx.where(
            mx.arange(64)[None, None, :] == predicted[..., None], 20.0, -20.0
        )


def _reference(model, prompt, count):
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    tokens = []
    for _ in range(count):
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        logits = model(mx.array([[token]]), cache=cache)
    return tokens


def test_parked_lane_still_reprobes_and_readmits_when_copying_becomes_useful():
    prompt = [1, 2, 3] * 6 + [40, 41, 42]
    model = _CycleAfter(start=len(prompt) + 24)
    count = 90
    outputs, finals, stats, _forwards = _run(model, [prompt], [count])
    assert outputs[0] == _reference(model, prompt, count)
    receipt = finals[0].speculative_receipt
    latch = receipt["cost_latch"]
    # Parked rounds before the cycle were pipelined ...
    assert receipt["pipelined_plain_rounds"] >= 20
    # ... shadows kept probing, and once copying paid the lane re-admitted
    # proposals and committed accepted drafts.
    assert latch["activations"] >= 1
    assert receipt["accepted"] > 0 and stats["pld_accepted"] > 0
    assert stats["pld_pipeline_drains"] >= 1


def test_pipelined_round_sharing_a_prefill_slice_is_not_timed():
    model = _tiny_gpt_oss()
    later = [(11 * i + 3) % 60 + 1 for i in range(30)]
    outputs, finals, stats, _forwards = _run(
        model, NOVEL[:1], [40], insert_later=(8, later, 12)
    )
    assert outputs[0] == _greedy(model, NOVEL[0], 40)
    assert outputs[1] == _greedy(model, later, 12)
    assert stats["pld_pipeline_untimed_rounds"] >= 1
    latch = finals[0].speculative_receipt["cost_latch"]
    # Untimed rounds are not booked as plain cost.
    assert latch["plain_tokens"] < 40


def test_parked_hybrid_lane_with_a_cost_latch_pipelines_too():
    from test_segmented_mtp import _tiny_hybrid_mtp_model
    from test_sweep_20261006_spec_pld_hybrid import _greedy as hybrid_greedy

    model = _tiny_hybrid_mtp_model()
    prompt = [(17 * i + 7) % 120 + 1 for i in range(40)]
    outputs, finals, stats, _forwards = _run(model, [prompt], [30])
    assert outputs[0] == hybrid_greedy(model, prompt, 30)
    assert stats["pld_pipelined_rounds"] == 30
    assert stats["pld_parked_direct_rounds"] == 0
    assert finals[0].speculative_receipt["cost_latch"]["state"] == "parked"


def test_concurrent_cost_lanes_decode_in_one_batched_cohort():
    model = _tiny_gpt_oss()
    outputs, _finals, stats, forwards = _run(model, NOVEL[:2], [20, 20])
    for prompt, output in zip(NOVEL[:2], outputs):
        assert output == _greedy(model, prompt, 20)
    assert stats["pld_batched_max_width"] == 2
    assert stats.get("pld_cost_width1_cohort_splits", 0) == 0
    # Both lanes decode in shared two-row forwards, not one row at a time.
    assert forwards.widths.count((2, 1)) >= 19


def test_concurrent_cost_latched_hybrid_lanes_pipeline_as_one_cohort():
    from test_segmented_mtp import _tiny_hybrid_mtp_model
    from test_sweep_20261006_spec_pld_hybrid import _greedy as hybrid_greedy

    model = _tiny_hybrid_mtp_model()
    prompts = [[(17 * i + 7 + 5 * lane) % 120 + 1 for i in range(40)] for lane in range(3)]
    outputs, finals, stats, _forwards = _run(model, prompts, [24, 24, 24])
    for prompt, output in zip(prompts, outputs):
        assert output == hybrid_greedy(model, prompt, 24)
    assert stats["pld_batched_max_width"] == 3
    assert stats["pld_pipelined_rounds"] >= 20
    assert stats.get("pld_cost_width1_cohort_splits", 0) == 0
    assert all(final.speculative_receipt["cost_latch"]["state"] == "parked" for final in finals)
