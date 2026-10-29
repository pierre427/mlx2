"""Synthetic multi-request equations/state@cpu; no actual checkpoint claim."""

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_deepseek_v41_cpu import Weights, config

from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.runtime.models.deepseek_v41_batch import (
    DeepSeekV41BatchedModel,
    DeepSeekV41BatchLayerCache,
)


def pair(width=4):
    c = config()
    w = Weights(c)
    base = DeepSeekV41Model(c, w, identity="synthetic-batch@cpu")
    return base, DeepSeekV41BatchedModel(base, width=width)


def merge(caches):
    return [
        DeepSeekV41BatchLayerCache.merge([r[i] for r in caches])
        for i in range(len(caches[0]))
    ]


def assert_state(actual, expected):
    assert actual.meta_state == expected.meta_state
    for a, e in zip(actual.state, expected.state):
        if a is None or e is None:
            assert a is e
        else:
            np.testing.assert_allclose(np.array(a), np.array(e), rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("width", [2, 4])
def test_real_joint_ffn_head_ragged_prefill_and_decode_match_independent(width):
    base, batch = pair(width)
    prompts = [[1, 2, 3, 4, 5], [6, 7, 8], [9, 10, 11, 12], [13, 14]][:width]
    singles = [base.make_cache() for _ in prompts]
    controls = []
    for p, c in zip(prompts, singles):
        controls.append(base(mx.array([p], mx.int32), c))
    caches = merge([batch.make_cache() for _ in prompts])
    size = max(map(len, prompts))
    lengths = list(map(len, prompts))
    for c in caches:
        c.prepare(lengths=lengths, right_padding=[size - l for l in lengths])
    tokens = mx.array([p + [0] * (size - len(p)) for p in prompts], mx.int32)
    actual = batch(tokens, caches)
    for c in caches:
        c.finalize()
    for i, p in enumerate(prompts):
        np.testing.assert_allclose(
            np.array(actual[i, : len(p)]),
            np.array(controls[i][0]),
            rtol=4e-5,
            atol=2e-6,
        )
        for layer, expected in zip(caches, singles[i]):
            assert_state(layer.rows[i], expected)
    next_ids = mx.array([[1 + i] for i in range(width)], mx.int32)
    result = batch(next_ids, caches)
    for i in range(width):
        expected = base(next_ids[i : i + 1], singles[i])
        np.testing.assert_allclose(
            np.array(result[i]), np.array(expected[0]), rtol=4e-5, atol=2e-6
        )
    assert batch.batch_counters["maximum_live_requests"] == width
    assert batch.batch_counters["joint_ffn_calls"] >= len(base.layers)
    assert batch.batch_execution_profile["attention_physical_width"] == 1


def test_warm_cold_merge_filter_cancel_and_extract_preserve_independent_frontiers():
    _base, batch = pair()
    warm = batch.make_cache()
    batch(mx.array([[1, 2, 3]], mx.int32), warm)
    cold = batch.make_cache()
    both = merge([warm, cold])
    batch(mx.array([[4], [5]], mx.int32), both)
    assert (
        both[0].host_lengths == [4, 1] and warm[0].offset == 3 and cold[0].offset == 0
    )
    fork = [c.fork() for c in both]
    extracted = [c.extract(1) for c in both]
    for c in both:
        c.filter([0])
    batch(mx.array([[6]], mx.int32), both)
    assert (
        both[0].offset == 5
        and fork[0].host_lengths == [4, 1]
        and extracted[0].offset == 1
    )
    for c in both:
        c.extend(extracted[both.index(c)])
    batch(mx.array([[7], [8]], mx.int32), both)
    assert both[0].host_lengths == [6, 2]


def test_failure_discards_all_private_lane_frontiers(monkeypatch):
    base, batch = pair(2)
    caches = merge([batch.make_cache(), batch.make_cache()])
    batch(mx.array([[1, 2], [3, 4]], mx.int32), caches)
    snapshots = [c.snapshot() for c in caches]
    original = base.layers[3].shared_expert
    monkeypatch.setattr(
        base.layers[3],
        "shared_expert",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected joint failure")),
    )
    with pytest.raises(RuntimeError, match="injected"):
        batch(mx.array([[5], [6]], mx.int32), caches)
    for c, s in zip(caches, snapshots):
        restored = type(c).from_state(s.arrays, s.metadata)
        assert c.meta_state == restored.meta_state and not c.pending
        for a, e in zip(c.rows, restored.rows):
            assert_state(a, e)
    monkeypatch.setattr(base.layers[3], "shared_expert", original)
    assert batch(mx.array([[5], [6]], mx.int32), caches).shape == (2, 1, 19)


def test_single_request_cache_persistence_apc_fork_and_batch_identity(tmp_path):
    from mlx2.runtime.cow_cache import freeze_prompt_cache, restore_prompt_cache
    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache

    _, batch = pair()
    cache = batch.make_cache()
    batch(mx.array([[1, 2, 3]], mx.int32), cache)
    path = tmp_path / "batch-private.safetensors"
    save_prompt_cache(str(path), cache)
    restored = load_prompt_cache(str(path))
    assert [r.meta_state for r in restored] == [r.meta_state for r in cache]
    frozen, _ = freeze_prompt_cache(
        cache, key="synthetic-batch", tokens=(1, 2, 3), cache_type="deepseek_v41"
    )
    branch = restore_prompt_cache(frozen)
    try:
        logits = batch(mx.array([[4]], mx.int32), branch)
        assert (
            logits.shape == (1, 1, 19)
            and branch[0].offset == 4
            and cache[0].offset == 3
        )
    finally:
        branch.close()
        frozen.close()
    changed = DeepSeekV41BatchedModel(batch.base, width=2)
    with pytest.raises(ValueError, match="execution"):
        changed(mx.array([[5]], mx.int32), cache)


def test_real_batch_generator_interleaves_ragged_prompts_decode_and_cancellation():
    from mlx2.runtime.generate import BatchGenerator

    base, batch = pair()
    prompts = [[1, 2, 3, 4], [5, 6], [7, 8, 9], [10, 11, 12, 13, 14]]
    expected = []
    for prompt in prompts:
        cache = base.make_cache()
        logits = base(mx.array([prompt], mx.int32), cache)
        out = []
        for _ in range(5):
            token = int(mx.argmax(logits[0, -1]).item())
            out.append(token)
            logits = base(mx.array([[token]], mx.int32), cache)
        expected.append(out)
    generator = BatchGenerator(
        batch,
        completion_batch_size=4,
        prefill_batch_size=4,
        prefill_step_size=2,
        stop_tokens=set(),
        sampler=lambda x: mx.argmax(x, -1),
    )
    ids = generator.insert(prompts, max_tokens=[5] * 4)
    actual = {uid: [] for uid in ids}
    for _ in range(100):
        _progress, rows = generator.next()
        for row in rows:
            actual[row.uid].append(int(row.token))
        if all(len(out) == 5 for out in actual.values()):
            break
    assert [actual[uid] for uid in ids] == expected
    assert batch.batch_counters["maximum_live_requests"] >= 2


def test_left_padding_and_zero_length_prefill_rows_do_not_advance_state():
    base, batch = pair(2)
    caches = merge([batch.make_cache(), batch.make_cache()])
    for c in caches:
        c.prepare(left_padding=[2, 0])
    result = batch(mx.array([[0, 0, 3], [4, 5, 6]], mx.int32), caches)
    for c in caches:
        c.finalize()
    assert caches[0].host_lengths == [1, 3]
    for i, prompt in enumerate(([3], [4, 5, 6])):
        expected = base(mx.array([prompt], mx.int32), base.make_cache())
        np.testing.assert_allclose(
            np.array(result[i, -len(prompt) :]),
            np.array(expected[0]),
            rtol=4e-5,
            atol=2e-6,
        )
    for c in caches:
        c.prepare(lengths=[0, 2], right_padding=[2, 0])
    before = [c.extract(0) for c in caches]
    batch(mx.array([[0, 0], [7, 8]], mx.int32), caches)
    for c in caches:
        c.finalize()
    for a, e in zip(caches, before):
        assert_state(a.rows[0], e.rows[0])


def test_actual_generator_cancel_filters_private_rows_without_affecting_survivor():
    from mlx2.runtime.generate import BatchGenerator

    base, batch = pair(2)
    p = [1, 2, 3, 4]
    cache = base.make_cache()
    logits = base(mx.array([p], mx.int32), cache)
    expected = []
    for _ in range(8):
        token = int(mx.argmax(logits[0, -1]).item())
        expected.append(token)
        logits = base(mx.array([[token]], mx.int32), cache)
    generator = BatchGenerator(
        batch,
        completion_batch_size=2,
        prefill_batch_size=2,
        prefill_step_size=2,
        stop_tokens=set(),
        sampler=lambda x: mx.argmax(x, -1),
    )
    first, second = generator.insert([p, [5, 6]], max_tokens=[8, 8])
    actual = []
    cancelled = False
    for _ in range(30):
        _, rows = generator.next()
        actual.extend(int(r.token) for r in rows if r.uid == first)
        if rows and not cancelled:
            generator.remove([second])
            cancelled = True
        if len(actual) == 8:
            break
    assert cancelled and actual == expected


def test_pure_cpu_batch_policy_refuses_unsupported_before_allocation():
    from mlx2.adapters.deepseek_v41_batch import validate_physical_batch_policy

    assert (
        validate_physical_batch_policy(
            {
                "deepseek_source_experts": {"mode": "original_lfu"},
                "deepseek_source_expert_math": "reference",
            },
            width=4,
        )
        == 4
    )
    for value in (
        {"draft_model": "embedded_dspark"},
        {"draft_model": "prompt_lookup"},
        {"ast_prompt_lookup": True},
        {"deepseek_vision": "enabled"},
        {"deepseek_ced_prefill": "dependency_exact"},
        {"deepseek_source_completion": "bounded_2"},
        {"deepseek_source_expert_math": "routed_fused_fp32"},
    ):
        with pytest.raises(ValueError):
            validate_physical_batch_policy(value, width=2)
    for width in (True, 1, 3, "4"):
        with pytest.raises(ValueError):
            validate_physical_batch_policy({}, width=width)


def test_packed_failure_after_appends_preserves_visible_prefix_and_retries(
    monkeypatch, tmp_path
):
    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache

    c = config(
        head_dim=32,
        index_head_dim=32,
        simulate_quantization=True,
        activation_dtype="bfloat16",
        sliding_window=3,
    )
    base = DeepSeekV41Model(
        c,
        Weights(c),
        identity="synthetic-packed-batch@cpu",
        kv_storage_policy="native_packed",
    )
    batch = DeepSeekV41BatchedModel(base, width=2)
    caches = merge([batch.make_cache(), batch.make_cache()])
    batch(mx.array([[1, 2, 3, 4], [5, 6, 7, 8]], mx.int32), caches)
    snapshots = [r.snapshot() for r in caches]
    control = [
        type(r).from_state(s.arrays, s.metadata) for r, s in zip(caches, snapshots)
    ]
    original = base.layers[4].shared_expert

    def fail(*args, **kwargs):
        # Both ratio2 and ratio1 owner appends have occurred before this layer.
        raise RuntimeError("after packed append")

    monkeypatch.setattr(base.layers[4], "shared_expert", fail)
    with pytest.raises(RuntimeError, match="after packed append"):
        batch(mx.array([[9, 10], [11, 12]], mx.int32), caches)
    assert base._weight_executions == 0
    for actual, expected in zip(caches, control):
        assert actual.meta_state == expected.meta_state
        for a, e in zip(actual.rows, expected.rows):
            assert_state(a, e)
    monkeypatch.setattr(base.layers[4], "shared_expert", original)
    expected = batch(mx.array([[9, 10], [11, 12]], mx.int32), control)
    actual = batch(mx.array([[9, 10], [11, 12]], mx.int32), caches)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))
    # Persistence stores the physical uint8 code/scales rather than dequantized views.
    selected = [r.extract(1) for r in caches]
    path = tmp_path / "packed-private.safetensors"
    save_prompt_cache(str(path), selected)
    loaded = load_prompt_cache(str(path))
    assert loaded[0].rows[0].state[0].dtype == mx.uint8
    out = batch(mx.array([[13]], mx.int32), loaded)
    np.testing.assert_array_equal(
        np.array(out), np.array(batch(mx.array([[13]], mx.int32), selected))
    )


def test_engram_prefetch_is_consumed_per_lane_and_hash_tails_remain_private():
    from test_deepseek_v41_cpu import hash_metadata

    from mlx2.runtime.models.deepseek_v41_engram import EngramHasher

    c = config(
        engram_layer_ids=(1, 3),
        engram_num_embeddings=(48, 72),
        engram_compressed_vocab_size=5,
    )
    w = Weights(c)
    rng = np.random.default_rng(1989)
    for layer in c.engram_layer_ids:
        w.data[f"blk.{layer}.engram_q_norm.weight"] = mx.ones((4, 8))
        w.data[f"blk.{layer}.engram_k_norm.weight"] = mx.ones((4, 8))
        w.data[f"blk.{layer}.engram_kv.weight"] = mx.array(
            rng.normal(0, 0.0004, (40, 6144)).astype(np.float32)
        )

    class ExclusiveRows:
        pending = None
        consumed = 0

        def prefetch(self, ids):
            assert self.pending is None, "another lane overwrote Engram prefetch"
            self.pending = np.asarray(ids).copy()

        def read(self, ids):
            np.testing.assert_array_equal(self.pending, np.asarray(ids))
            self.pending = None
            self.consumed += 1
            return (
                np.broadcast_to(np.asarray(ids)[..., None] / 100, (*ids.shape, 256))
                .astype(np.float32)
                .copy()
            )

    metadata = hash_metadata()
    metadata["deepseek41.engram.token_map"] = [i % 5 for i in range(19)]
    tables = {i: ExclusiveRows() for i in c.engram_layer_ids}
    base = DeepSeekV41Model(
        c,
        w,
        identity="synthetic-engram-batch@cpu",
        hasher=EngramHasher(metadata),
        engram_tables=tables,
    )
    batch = DeepSeekV41BatchedModel(base, width=2)
    prompts = ([1, 2, 3, 4, 5], [10, 11])
    originals = [base.make_cache(), base.make_cache()]
    expected = [
        base(mx.array([p], mx.int32), cache) for p, cache in zip(prompts, originals)
    ]
    caches = merge([batch.make_cache(), batch.make_cache()])
    for cache in caches:
        cache.prepare(lengths=[5, 2], right_padding=[0, 3])
    actual = batch(mx.array([prompts[0], prompts[1] + [0, 0, 0]], mx.int32), caches)
    for cache in caches:
        cache.finalize()
    for i, prompt in enumerate(prompts):
        np.testing.assert_allclose(
            np.array(actual[i, : len(prompt)]),
            np.array(expected[i][0]),
            rtol=4e-5,
            atol=2e-6,
        )
        np.testing.assert_array_equal(
            np.array(caches[0].rows[i].hash_tail), np.array(originals[i][0].hash_tail)
        )
    selected = [cache.extract(1) for cache in caches]
    for cache in caches:
        cache.filter([0])
    batch(mx.array([[6]], mx.int32), caches)
    assert caches[0].rows[0].hash_tail.tolist() == [1, 0, 4]
    assert selected[0].rows[0].hash_tail.tolist() == [1, 0, -1]
    assert all(
        table.pending is None and table.consumed == 5 for table in tables.values()
    )


def test_park_drops_batch_layer_references_and_active_execution_blocks_park(
    monkeypatch,
):
    base, batch = pair(2)

    def fail_park(*args, **kwargs):
        with pytest.raises(ValueError, match="active forward"):
            batch.release_weight_references()
        return original(*args, **kwargs)

    original = base.layers[0].shared_expert
    monkeypatch.setattr(base.layers[0], "shared_expert", fail_park)
    batch(mx.array([[1, 2]], mx.int32), batch.make_cache())
    batch.release_weight_references()
    assert batch.layers == [] and base._weights_parked
    with pytest.raises(ValueError):
        batch(mx.array([[3]], mx.int32))
    assert base._weight_executions == 0


@pytest.mark.parametrize("width", [2, 4])
def test_seeded_generator_staggered_cohorts_match_independent_request_rngs(width):
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG, draw_key

    base, batch = pair(width)
    prompts = [[1, 2, 3, 4, 5], [6, 7], [8, 9, 10], [11, 12, 13, 14]]
    counts = [7, 4, 6, 5]
    seeds = [57, 91, 24, 103]

    def sampler(rng):
        def sample(logprobs):
            return mx.random.categorical(logprobs[0] / 0.8, key=draw_key(rng))[None]

        sample.batch_groupable = False
        return sample

    expected = []
    for prompt, count, seed in zip(prompts, counts, seeds):
        rng = LaneRNG(seed)
        draw = sampler(rng)
        cache = base.make_cache()
        logits = base(mx.array([prompt], mx.int32), cache)
        output = []
        for _ in range(count):
            values = logits[0, -1][None]
            token = int(draw(values - mx.logsumexp(values, -1, keepdims=True)).item())
            output.append(token)
            logits = base(mx.array([[token]], mx.int32), cache)
        expected.append(output)
    generator = BatchGenerator(
        batch,
        completion_batch_size=width,
        prefill_batch_size=width,
        prefill_step_size=2,
        stop_tokens=set(),
    )
    uids = []
    rngs = [LaneRNG(seed) for seed in seeds]

    def insert(first, last):
        uids.extend(
            generator.insert(
                prompts[first:last],
                max_tokens=counts[first:last],
                samplers=[sampler(r) for r in rngs[first:last]],
                lane_rngs=rngs[first:last],
            )
        )

    insert(0, 2)
    actual = {}
    for step in range(100):
        _, rows = generator.next()
        for row in rows:
            actual.setdefault(row.uid, []).append(int(row.token))
        if step == 2:
            insert(2, 4)
        if len(uids) == 4 and all(
            len(actual.get(uid, [])) == count for uid, count in zip(uids, counts)
        ):
            break
    assert [actual[uid] for uid in uids] == expected
    assert batch.batch_counters["maximum_live_requests"] >= 2
    generator.close()


def test_random_chunked_left_and_right_ragged_layouts_preserve_private_state():
    """Padding can cross chunk boundaries, and a row can have no work at all."""
    base, batch = pair(4)
    rng = np.random.default_rng(846)
    for _ in range(12):
        lengths = rng.integers(0, 24, 4).tolist()
        # Every case contains a completely inactive lane; choose its padding
        # independently so the lane can remain inactive through several chunks.
        lengths[0] = 0
        lengths[1] = max(1, lengths[1])
        left = rng.integers(0, 8, 4).tolist()
        width = max(a + b for a, b in zip(left, lengths))
        prompts = [rng.integers(0, 19, n).tolist() for n in lengths]
        controls = [base.make_cache() for _ in prompts]
        expected = [
            base(mx.array([prompt], mx.int32), cache) if prompt else None
            for prompt, cache in zip(prompts, controls)
        ]
        caches = merge([batch.make_cache() for _ in prompts])
        for cache in caches:
            cache.prepare(lengths=lengths, left_padding=left)
        tokens = mx.array([
            [0] * padding + prompt + [0] * (width - padding - len(prompt))
            for padding, prompt in zip(left, prompts)
        ], mx.int32)
        actual = mx.concatenate([
            batch(tokens[:, first:first + 3], caches)
            for first in range(0, width, 3)
        ], 1)
        for cache in caches:
            cache.finalize()
        for i, prompt in enumerate(prompts):
            if prompt:
                np.testing.assert_allclose(
                    np.asarray(actual[i, left[i]:left[i] + len(prompt)]),
                    np.asarray(expected[i][0]), rtol=5e-5, atol=3e-6,
                )
            for actual_layer, control in zip(caches, controls[i]):
                assert_state(actual_layer.rows[i], control)
