"""CPU equations/contracts only; no checkpoint load, GPU or qualification."""

import dataclasses
import struct
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_deepseek_v4_forward_cpu import SmallWeights

from mlx2.runtime.gguf import decode_blocks
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model, candidate_blocks
from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache
from mlx2.runtime.models.deepseek_v41_config import DeepSeekV41Config
from mlx2.runtime.models.deepseek_v41_engram import (
    EngramHasher,
    EngramRows,
    decode_engram_rows,
    file_stamp,
)
from mlx2.runtime.models.deepseek_v41_quant import fp4_round, quantize


def config(**kwargs):
    values = dict(  # noqa: C408 - readable fixture defaults
        vocab_size=19,
        hidden_size=8,
        num_hidden_layers=5,
        num_attention_heads=2,
        head_dim=8,
        q_lora_rank=4,
        qk_rope_head_dim=4,
        o_groups=2,
        o_lora_rank=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=6,
        rms_norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        compress_ratios=(0, 2, 2, 1, 1),
        kv_source_layer_ids=(1, 3),
        index_source_layer_ids=(1, 3, 4),
        candidate_source_layer_id=3,
        candidate_topk_blocks=1,
        candidate_block_size=2,
        engram_layer_ids=(),
        engram_num_embeddings=(),
        simulate_quantization=False,
        activation_dtype="float32",
    )
    values.update(kwargs)
    return DeepSeekV41Config(**values)


class Weights(SmallWeights):
    dense_prefill_policy = "reference"

    def __init__(self, c):
        data = dataclasses.asdict(c)
        data.update(num_hash_layers=0, compress_ratios=(0,) * c.num_hidden_layers)
        super().__init__(SimpleNamespace(**data))
        rng = np.random.default_rng(567)

        def put(name, shape, ones=False):
            self.data[name] = mx.array(
                np.ones(shape, np.float32)
                if ones
                else rng.normal(0, 0.08, shape).astype(np.float32)
            )

        for i in c.kv_source_layer_ids:
            p = f"blk.{i}."
            for name, shape in [
                ("attn_compressor_kv", (c.head_dim, c.hidden_size)),
                ("attn_compressor_gate", (c.head_dim, c.hidden_size)),
                ("indexer.attn_k", (c.index_head_dim, c.head_dim)),
            ]:
                put(p + name + ".weight", shape)
            for name, dim in [
                ("attn_compressor_norm", c.head_dim),
                ("indexer.k_norm", c.index_head_dim),
            ]:
                put(p + name + ".weight", (dim,), True)
        for i in c.index_source_layer_ids:
            put(
                f"blk.{i}.indexer.attn_q_b.weight",
                (c.index_n_heads * c.index_head_dim, c.q_lora_rank),
            )
            put(f"blk.{i}.indexer.proj.weight", (c.index_n_heads, c.hidden_size))

    def gate_up(self, x, g, u):
        return self.linear(g)(x), self.linear(u)(x)


def test_whole_chunked_and_tokenwise_forward_match():
    c = config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@v41-cpu")
    ids = mx.array((np.arange(17) % 19)[None], mx.int32)
    whole = model(ids, model.make_cache())
    cache = model.make_cache()
    chunk = mx.concatenate(
        [
            model(ids[:, :3], cache),
            model(ids[:, 3:8], cache),
            model(ids[:, 8:9], cache),
            model(ids[:, 9:], cache),
        ],
        1,
    )
    one = model.make_cache()
    decode = mx.concatenate([model(ids[:, i : i + 1], one) for i in range(17)], 1)
    np.testing.assert_allclose(np.array(whole), np.array(chunk), atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(np.array(whole), np.array(decode), atol=3e-6, rtol=3e-5)
    assert cache[1].pooled.shape == (1, 8, 8)
    assert cache[3].pooled.shape == (1, 17, 8)
    assert cache[2].pooled is None and cache[4].pooled is None
    assert all(s.offset == 17 and not s.pending for s in cache)


def test_bf16_and_all_qat_laws_preserve_snapshot_replay():
    c = config(
        hidden_size=32,
        head_dim=32,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        index_head_dim=32,
        o_lora_rank=16,
        moe_intermediate_size=32,
        simulate_quantization=True,
        activation_dtype="bfloat16",
    )
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@v41-bf16-qat-cpu")
    cache = model.make_cache()
    first = model(mx.array([[1, 2, 3, 4, 5]], mx.int32), cache)
    assert np.isfinite(np.array(first)).all()
    snapshots = [state.snapshot() for state in cache]
    before = np.array(model(mx.array([[6]], mx.int32), cache))
    for state, snapshot in zip(cache, snapshots):
        state.restore(snapshot)
    after = np.array(model(mx.array([[6]], mx.int32), cache))
    np.testing.assert_array_equal(before.view(np.uint32), after.view(np.uint32))
    assert all(state.offset == 6 and not state.pending for state in cache)


@pytest.mark.parametrize("dtype", [mx.int32, mx.int64, mx.uint32])
def test_public_integer_tokens_reach_native_embedding_as_uint32(dtype):
    c = config()
    weights = Weights(c)
    reference = weights.embedding
    calls = []

    def native_embedding(ids):
        assert ids.dtype == mx.uint32
        calls.append(ids.dtype)
        return reference(ids)

    weights.embedding = native_embedding
    model = DeepSeekV41Model(c, weights, identity="synthetic@native-token-boundary")
    result = model(mx.array([[1, 2, 3]], dtype), model.make_cache())
    assert np.isfinite(np.array(result)).all() and calls == [mx.uint32]


def test_snapshot_replay_restores_partial_group_and_all_owners():
    c = config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@v41-cpu")
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]], mx.int32), cache)
    snapshots = [s.snapshot() for s in cache]
    expected = model(mx.array([[4]], mx.int32), cache)
    model(mx.array([[5, 6]], mx.int32), cache)
    for s, snap in zip(cache, snapshots):
        s.restore(snap)
    actual = model(mx.array([[4]], mx.int32), cache)
    np.testing.assert_array_equal(np.array(expected), np.array(actual))
    assert cache[1].buf_kv.shape == (1, 0, 8)


def test_cache_identity_and_pending_fail_closed():
    state = DeepSeekV41LayerCache(xp=np, identity="one", history=True)
    snap = state.snapshot()
    state._in_forward = True
    with pytest.raises(Exception, match="incomplete"):
        state.snapshot()
    state.restore(snap)
    assert not state.pending
    other = DeepSeekV41LayerCache(xp=np, identity="two", history=True)
    with pytest.raises(ValueError, match="identity"):
        other.restore(snap)
    with pytest.raises(ValueError, match="revision"):
        DeepSeekV41LayerCache.from_state(
            snap.arrays, ("wrong", *snap.metadata[1:]), xp=np
        )


def test_hash_tail_is_exact_cache_state():
    state = DeepSeekV41LayerCache(xp=np, identity="one", history=True)
    state.update_local(np.ones((1, 1, 1, 512), np.float32))
    state.hash_tail = np.array([5, -1, -1], np.int32)
    snap = state.snapshot()
    state.hash_tail[0] = 8
    state.restore(snap)
    assert state.hash_tail.tolist() == [5, -1, -1]
    restored = DeepSeekV41LayerCache.from_state(state.state, state.meta_state, xp=np)
    state.hash_tail[0] = 9
    assert restored.hash_tail[0] == 5


def hash_metadata():
    return {
        "deepseek41.engram.token_map": [0, 1, 2, 3, 3, 4],
        "deepseek41.engram.rows": [48, 72],
        "deepseek41.engram.primes": [2] * 24 + [3] * 24,
        "deepseek41.engram.multipliers": [3, 5, 7, 9, 11, 13, 15, 17],
        "deepseek41.engram.compressed_vocab_size": 5,
        "deepseek41.engram.pad_id": 2,
    }


def test_hash_chunking_mask_breaks_and_normalized_token_map():
    hasher = EngramHasher(hash_metadata())
    ids = np.array([3, 4, 1, 5, 2, 3])
    tail = np.full(3, -1, np.int32)
    full, end = hasher(ids, tail)
    a, ta = hasher(ids[:2], tail)
    b, tb = hasher(ids[2:], ta)
    np.testing.assert_array_equal(full, np.concatenate((a, b)))
    np.testing.assert_array_equal(end, tb)
    mask = np.array([True, False, True, True, True, True])
    masked, _mt = hasher(ids, tail, mask)
    chunks = []
    t = tail
    for i in range(len(ids)):
        h, t = hasher(ids[i : i + 1], t, mask[i : i + 1])
        chunks.append(h)
    np.testing.assert_array_equal(masked, np.concatenate(chunks))
    assert np.any(full != masked)
    np.testing.assert_array_equal(
        hasher(np.array([3]), tail)[0], hasher(np.array([4]), tail)[0]
    )


def test_row_decode_and_mixed_cache_reads(tmp_path):
    raw = np.zeros((5, 264), np.uint8)
    raw[:, :256] = np.arange(5)[:, None] + 0x38
    raw[:, 256:] = 127
    path = tmp_path / "rows"
    path.write_bytes(raw.tobytes())
    stamp = file_stamp(path.stat())
    table = EngramRows(
        path, offset=0, rows=5, expected_stat=stamp, cache_bytes=3 * 1024
    )
    try:
        a = table.read(np.array([[2, 1, 2]], np.int32))
        expected = decode_engram_rows(raw[[2, 1, 2]])
        np.testing.assert_array_equal(a[0], expected)
        assert table.stats["bytes_read"] == 528
        table.prefetch(np.array([2, 3], np.int32))
        b = table.read(np.array([2, 3], np.int32))
        np.testing.assert_array_equal(b, decode_engram_rows(raw[[2, 3]]))
        assert table.stats["bytes_read"] == 792 and table.stats["cache_hits"] == 1
        assert table.diagnostics()["cached_bytes"] <= 3072
        with pytest.raises(ValueError, match="budget"):
            table.read(np.array([-1]))
        with path.open("r+b") as f:
            f.write(b"x")
        with pytest.raises(ValueError, match="changed"):
            table.read(np.array([1]))
    finally:
        table.close()
    table.close()
    with pytest.raises(ValueError, match="closed"):
        table.read(np.array([1]))


@pytest.mark.parametrize("bad", [(0, 127), (256, 255)])
def test_nonfinite_rows_refuse(bad):
    raw = np.zeros((1, 264), np.uint8)
    raw[:, 256:] = 127
    raw[0, bad[0]] = bad[1]
    with pytest.raises(ValueError, match="nonfinite"):
        decode_engram_rows(raw)


def test_fp4_midpoint_ties_and_distinct_kv_laws():
    x = mx.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75])
    np.testing.assert_array_equal(np.array(fp4_round(x)), [0, 1, 1, 2, 2, 4, 4, 0, -1])
    values = np.tile(np.linspace(-0.2, 2.7, 32, dtype=np.float32), (2, 1))
    a = np.array(quantize(mx.array(values), "index_fp4"))
    b = np.array(quantize(mx.array(values), "compressed_fp4"))
    assert np.isfinite(a).all() and np.isfinite(b).all() and np.any(a != b)


def test_candidate_pins_newest_and_never_includes_future():
    score = mx.array([[[10.0, 9.0, 2.0, 1.0, -mx.inf, -mx.inf]]])
    mask = np.array(candidate_blocks(score, mx.array([[[3]]]), 1, 2))
    np.testing.assert_array_equal(mask, [[[False, False, True, True, False, False]]])


def test_iq1_block_uses_signed_grid_delta_and_scales():
    from mlx2.runtime.gguf.iq1_codebook import IQ1_GRID

    raw = bytearray(50)
    struct.pack_into("<e", raw, 0, 0.5)
    raw[2:34] = bytes(range(32))
    for group in range(8):
        struct.pack_into(
            "<H", raw, 34 + 2 * group, (group << 12) | (0x8000 if group % 2 else 0)
        )
    decoded = decode_blocks(raw, 19)
    expected = []
    for group in range(8):
        for j in range(4):
            grid = IQ1_GRID[4 * group + j]
            for lane in range(8):
                v = (grid >> (8 * lane)) & 255
                v = v - 256 if v > 127 else v
                expected.append(
                    0.5 * (2 * group + 1) * (v + (-0.125 if group % 2 else 0.125))
                )
    np.testing.assert_array_equal(decoded, np.array(expected, np.float32))
    with pytest.raises(ValueError, match="budget"):
        decode_blocks(raw, 19, max_elements=1)


def test_model_engram_history_chunk_and_rejected_replay():
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
    metadata = hash_metadata()
    metadata["deepseek41.engram.token_map"] = [i % 5 for i in range(19)]

    class Rows:
        def prefetch(self, ids):
            pass

        def read(self, ids):
            return (
                np.broadcast_to(np.asarray(ids)[..., None] / 100, (*ids.shape, 256))
                .astype(np.float32)
                .copy()
            )

    target = DeepSeekV41Model(
        c,
        w,
        identity="synthetic-engram@cpu",
        hasher=EngramHasher(metadata),
        engram_tables={i: Rows() for i in c.engram_layer_ids},
    )
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7]], mx.int32)
    whole = target(ids, target.make_cache())
    cache = target.make_cache()
    a = target(ids[:, :3], cache)
    snapshots = [s.snapshot() for s in cache]
    expected = target(ids[:, 3:], cache)
    for s, snap in zip(cache, snapshots):
        s.restore(snap)
    replay = target(ids[:, 3:], cache)
    np.testing.assert_array_equal(np.array(expected), np.array(replay))
    np.testing.assert_allclose(
        np.array(whole),
        np.array(mx.concatenate([a, expected], 1)),
        rtol=3e-5,
        atol=3e-6,
    )
    assert cache[0].hash_tail.tolist() == [2, 1, 0]


def test_cache_capacity_and_owned_lru_do_not_hide_unbounded_buffers(tmp_path):
    c = config()
    target = DeepSeekV41Model(c, Weights(c), identity="capacity@cpu")
    cache = target.make_cache()
    target(mx.array([[1, 2, 3]], mx.int32), cache)
    assert cache[3]._capacity == 256 and cache[3].pooled.shape[1] == 3
    before = cache[3].nbytes
    target(mx.array([[4]], mx.int32), cache)
    assert cache[3]._capacity == 256 and cache[3].nbytes >= before
    raw = np.zeros((8, 264), np.uint8)
    raw[:, 256:] = 127
    p = tmp_path / "rows"
    p.write_bytes(raw.tobytes())
    table = EngramRows(
        p, offset=0, rows=8, expected_stat=file_stamp(p.stat()), cache_bytes=1024
    )
    try:
        table.read(np.arange(8))
        assert len(table._cache) == 1
        row = next(iter(table._cache.values()))
        assert row.base is None and row.nbytes == 1024
    finally:
        table.close()


def test_prefill_candidates_are_scoped_away_from_decode_and_verification():
    from mlx2.runtime.gguf.segmented_prefill import segmented_prefill_active

    c = config()
    w = Weights(c)
    w.expert_prefill_policy = "segmented_nax_fp16"
    target = DeepSeekV41Model(c, w, identity="scoped-prefill@cpu")
    calls = []
    ordinary = w.moe

    def record(x, *args):
        calls.append(segmented_prefill_active(x.shape[0]))
        return ordinary(x, *args)

    w.moe = record
    ids = mx.array(np.arange(16)[None] % 19, mx.int32)
    target.prefill_forward(ids, target.make_cache())
    assert calls and all(calls)
    calls.clear()
    target(ids, target.make_cache())
    assert calls and not any(calls)
    calls.clear()
    target.verify_rows(ids[:, :3], target.make_cache())
    assert calls and not any(calls)


def test_v41_segmented_capacity_covers_all_384_experts():
    from mlx2.runtime.gguf.v41_prefill import SCAN_SOURCE, plan_capacity, scratch_bytes

    for rows in (1, 383, 384, 1024 * 6):
        capacity = plan_capacity(rows, 384)
        assert min(rows, 384) <= capacity <= rows
    assert "scan[512]" in SCAN_SOURCE and "scan[511]" in SCAN_SOURCE
    assert scratch_bytes(1024, 6, 2304, 5120, 384) < 512 << 20
    with pytest.raises(ValueError):
        plan_capacity(1, 513)


def test_fused_qat_does_not_dispatch_on_cpu():
    from mlx2.runtime.models.deepseek_v41_quant import fused_quantize

    with pytest.raises(ValueError, match="owned Metal"):
        fused_quantize(mx.ones((1, 128), mx.float32), "index_fp4")


def test_execution_identity_binds_dtype_qat_and_prefill_profiles():
    a = config()
    b = dataclasses.replace(a, activation_dtype="bfloat16")
    first = DeepSeekV41Model(a, Weights(a), identity="same-artifact")
    second = DeepSeekV41Model(b, Weights(b), identity="same-artifact")
    assert first.identity != second.identity
    with pytest.raises(ValueError, match="identity"):
        second(mx.array([[1]], mx.int32), first.make_cache())
