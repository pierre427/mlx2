"""Synthetic CPU contracts only; no checkpoint, GPU or qualification evidence."""


import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41_cpu import Weights, config
from test_deepseek_v41_kv_fastpaths_cpu import _selector

from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.runtime.models.deepseek_v41_ced import dependency_starts
from mlx2.runtime.models.deepseek_v41_packed_cache import DeepSeekV41PackedLayerCache
from mlx2.runtime.models.deepseek_v41_packed_qat import decode, produce
from mlx2.runtime.models.deepseek_v41_quant import quantize


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic CPU test attempted Metal")
    monkeypatch.setattr(mx.fast, "metal_kernel", forbidden)
    yield
    mx.set_default_device(previous)


def _bits(x):
    return np.array(x).view(np.uint32)


@pytest.mark.parametrize("kind,bytes_per_row", [("fp8", 528), ("index_fp4", 272), ("compressed_fp4", 288)])
def test_original_code_scales_decode_reference_exactly(kind, bytes_per_row):
    rng = np.random.default_rng(815)
    values = rng.normal(size=(9, 512)).astype(np.float32)
    values[0, :16] = [0., -0., .25, -.25, .75, -.75, 1.25, -1.25, 1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5., -5.]
    for scale in (2.**-40, 1., 2.**40):
        x = mx.array(values * scale)
        packed = produce(x, kind)
        assert packed.codes.dtype == packed.scales.dtype == mx.uint8
        assert packed.nbytes == 9 * bytes_per_row
        expected = quantize(x, kind).astype(mx.bfloat16).astype(mx.float32)
        np.testing.assert_array_equal(_bits(decode(packed)), _bits(expected))


def test_original_code_storage_avoids_nonidempotent_requantization():
    # Search deterministically for the documented scale-boundary counterexample.
    x = mx.array(np.random.default_rng(4).normal(size=(4096, 32)).astype(np.float32))
    first = quantize(x, "compressed_fp4").astype(mx.bfloat16).astype(mx.float32)
    second = quantize(first, "compressed_fp4").astype(mx.bfloat16).astype(mx.float32)
    assert np.any(_bits(first) != _bits(second))
    np.testing.assert_array_equal(_bits(decode(produce(x, "compressed_fp4"))), _bits(first))


def _packed_config(**kwargs):
    return config(head_dim=32, index_head_dim=32, simulate_quantization=True,
                  activation_dtype="bfloat16", sliding_window=3, **kwargs)


@pytest.mark.parametrize("with_engram", [False, True])
def test_wired_packed_model_state_snapshot_and_replay_match_reference(with_engram):
    from test_deepseek_v41_cpu import hash_metadata

    from mlx2.runtime.models.deepseek_v41_engram import EngramHasher

    c = _packed_config(**({"engram_layer_ids": (1, 3), "engram_num_embeddings": (48, 72),
                           "engram_compressed_vocab_size": 5} if with_engram else {}))
    metadata = hash_metadata()
    metadata["deepseek41.engram.token_map"] = [i % 5 for i in range(c.vocab_size)]
    class Rows:
        def prefetch(self, ids):
            pass
        def read(self, ids):
            return np.broadcast_to(np.asarray(ids)[..., None] / 100, (*ids.shape, 256)).astype(np.float32).copy()
    def make_model(**policies):
        weights = Weights(c)
        rng = np.random.default_rng(1989)
        for layer in c.engram_layer_ids:
            weights.data[f"blk.{layer}.engram_q_norm.weight"] = mx.ones((4, 8))
            weights.data[f"blk.{layer}.engram_k_norm.weight"] = mx.ones((4, 8))
            weights.data[f"blk.{layer}.engram_kv.weight"] = mx.array(rng.normal(0, .0004, (40, 6144)).astype(np.float32))
        return DeepSeekV41Model(c, weights, identity="synthetic@native-qat",
                                hasher=EngramHasher(metadata) if with_engram else None,
                                engram_tables={i: Rows() for i in c.engram_layer_ids}, **policies)
    ref = make_model()
    model = make_model(kv_storage_policy="native_packed")
    rc, pc = ref.make_cache(), model.make_cache()
    for values in ([1], [2, 3, 4, 5], [6], [7, 8]):
        tokens = mx.array([values])
        np.testing.assert_array_equal(_bits(ref(tokens, rc, logits_to_keep=0)), _bits(model(tokens, pc, logits_to_keep=0)))
        for a, b in zip(rc, pc):
            for av, bv in zip(a.state, (b.keys, b.buf_kv, b.buf_gate, b.pooled, b.index_keys, b.hash_tail)):
                if av is not None:
                    np.testing.assert_array_equal(np.array(av), np.array(bv))
                else:
                    assert bv is None
    assert sum(c.nbytes for c in pc) < sum(c.nbytes for c in rc)
    restored = [DeepSeekV41PackedLayerCache.from_state(c.state, c.meta_state, xp=mx) for c in pc]
    saved = [c.snapshot() for c in restored]
    expected = model(mx.array([[9, 10]]), pc, logits_to_keep=0)
    actual = model(mx.array([[9, 10]]), restored, logits_to_keep=0)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))
    for c, snapshot in zip(restored, saved):
        c.restore(snapshot)
    np.testing.assert_array_equal(_bits(model(mx.array([[9, 10]]), restored, logits_to_keep=0)), _bits(expected))
    assert model.scheduled_kv_calls["native_packed_local_rows"] > 0
    with pytest.raises(ValueError, match="identity"):
        ref(mx.array([[1]]), restored)


def test_packed_publication_does_not_dequantize_and_rejects_decoded_write(monkeypatch):
    c = _packed_config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@raw-publication", kv_storage_policy="native_packed")
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]]), cache)
    def forbidden(*args):
        raise AssertionError("publication dequantized persistent planes")
    monkeypatch.setattr("mlx2.runtime.models.deepseek_v41_packed_cache.decode", forbidden)
    for state in cache:
        state.validate_state()
        state.snapshot()
        assert state.nbytes > 0
        with pytest.raises(ValueError, match="producer"):
            state.update_local(mx.zeros((1, 1, 1, 32)))
    malformed = list(cache[1].state)
    malformed[5] = malformed[5].astype(mx.float32)
    with pytest.raises(ValueError, match="geometry"):
        DeepSeekV41PackedLayerCache.from_state(malformed, cache[1].meta_state, xp=mx)


@pytest.mark.parametrize("policy,plane", [("reference", 3), ("reference", 4),
                                         *(('native_packed', p) for p in (4, 5, 6, 7))])
def test_restore_rejects_mismatched_serialized_pool_extents(policy, plane):
    c = _packed_config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@source-plane-extents",
                             kv_storage_policy=policy)
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]]), cache)
    owner = cache[1]
    # Live allocations have capacity slack; serialized planes must contain
    # exactly the published rows, including both original scale planes.
    assert owner._capacity > owner._pool_count == 1
    restored = type(owner).from_state(owner.state, owner.meta_state, xp=mx)
    assert restored._capacity == restored._pool_count == 1
    malformed = list(owner.state)
    malformed[plane] = mx.concatenate((malformed[plane], malformed[plane]), axis=1)
    with pytest.raises(ValueError, match="geometry|frontier|extent"):
        type(owner).from_state(malformed, owner.meta_state, xp=mx)


def test_packed_private_ast_profile_cannot_be_laundered_through_disk(tmp_path):
    from mlx2.runtime.cache_operations import CacheOperationsUnsupported
    from mlx2.runtime.cow_cache import (
        COWCheckpointPublicationUnsafe,
        snapshot_prompt_cache_descriptors,
    )
    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache

    c = _packed_config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@packed-private-profile",
                             kv_storage_policy="native_packed")
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]]), cache)
    for row in cache:
        row.ast_private_profile = "synthetic-private-ast"
        row.ast_private_revision = "synthetic-source-revision"
        with pytest.raises(CacheOperationsUnsupported, match="private"):
            row.snapshot()
    with pytest.raises(COWCheckpointPublicationUnsafe, match="private"):
        snapshot_prompt_cache_descriptors(cache)
    private, _, _ = snapshot_prompt_cache_descriptors(cache, require_committed=False)
    assert all(row.ast_private_profile == "synthetic-private-ast" and
               row.ast_private_revision == "synthetic-source-revision" for row in private)
    path = tmp_path / "private-packed.safetensors"
    save_prompt_cache(str(path), cache)
    with pytest.raises(ValueError, match="revision"):
        load_prompt_cache(str(path))


@pytest.mark.parametrize("entry", ["provider", "tree", "batched_tree", "bank", "model_bank"])
def test_packed_cache_refuses_reference_only_ast_and_bank_before_execution(entry, monkeypatch):
    from types import SimpleNamespace

    from mlx2.adapters.deepseek_v41_ast_batched_target import execute_batched_tree
    from mlx2.adapters.deepseek_v41_ast_target import (
        SourceASTTargetTreeExecutor,
        execute_tree,
    )
    from mlx2.adapters.deepseek_v41_source_bank_session import PreparedBankSession

    c = _packed_config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@packed-transaction-refusal",
                             kv_storage_policy="native_packed")
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]]), cache)
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported transaction touched native packed values or acquired resources")
    monkeypatch.setattr(DeepSeekV41PackedLayerCache, "keys", property(forbidden))
    paths = ((4, 5, 6, 7, 8), (9, 10, 11, 12, 13), (14, 15, 16, 17, 18))
    with pytest.raises(ValueError, match="reference.*cache|packed.*bank"):
        if entry == "provider":
            SourceASTTargetTreeExecutor(model, cpu_reference=True)
        elif entry == "tree":
            execute_tree(model, cache, 3, paths, (4,), cpu_reference=True)
        elif entry == "batched_tree":
            execute_batched_tree(model, cache, 3, paths, (4,), cpu_reference=True,
                                 expert_law="ordinary_m16_v1")
        elif entry == "bank":
            session = object.__new__(PreparedBankSession)
            session.poisoned = session.active = session.replaying = False
            session.cancelled = session.ownercheck = forbidden
            session.forward(model, mx.array([[4]]), cache, forbidden)
        else:
            model.weights.source_resident_bank_session = SimpleNamespace(active=False, replaying=False,
                                                                          selected=forbidden)
            model(mx.array([[4]]), cache)
    assert all(row.offset == 3 and not row.pending for row in cache)
    assert model._weight_executions == 0


def test_packed_cache_declines_reference_only_checkpoint_resource_hints():
    from types import SimpleNamespace

    from mlx2.adapters.deepseek_v41_source_checkpoint_resources import snapshot

    c = _packed_config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@packed-advisory-resources",
                             kv_storage_policy="native_packed")
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]]), cache)
    # Hints are optional, non-numerical resources. Decline before touching an
    # expert-store inventory: this model intentionally has no such store.
    assert snapshot(SimpleNamespace(model=model), cache, (1, 2, 3)) is None


@pytest.mark.parametrize("start", [0, 200, 509])
@pytest.mark.parametrize("all_masked", [False, True])
def test_candidate_only_scores_match_global_ties_and_causal_masks(start, all_masked):
    r, rs, ref, x, qr, _, _ = _selector(513, layer=24, cap=2, length=4)
    f, fs, fast, _, _, _, _ = _selector(513, layer=24, cap=2, length=4)
    f.model.candidate_score_policy = "candidate_only"
    mask = np.zeros((1, 4, 513), bool)
    if not all_masked:
        mask[:, :, 32:40] = True
        mask[:, :, 504:512] = True
    ref["candidates"] = fast["candidates"] = mx.array(mask)
    r.select(x, qr, rs, start, ref)
    f.select(x, qr, fs, start, fast)
    for key in ("indices", "selected_mask"):
        np.testing.assert_array_equal(np.array(ref[key]), np.array(fast[key]))
    assert f.model.scheduled_kv_calls["candidate_only_score_rows"] == 4 * 16
    assert f.model.scheduled_kv_calls["candidate_only_unscored_rows"] == 4 * (513 - 16)


def _ced_config():
    return config(num_hidden_layers=8, compress_ratios=(0, 2, 2, 1, 1, 1, 1, 1),
                  kv_source_layer_ids=(1, 3), index_source_layer_ids=(1, 3, 5, 7),
                  sliding_window=3)


def test_dependency_plan_preserves_source_pools_and_every_future_local_window():
    c = _ced_config()
    starts = dependency_starts(c, 24)
    assert starts[:4] == (0, 0, 0, 0)
    assert starts[4:] == (15, 17, 19, 21)
    assert all(24 - s >= c.sliding_window for s in starts)
    assert dependency_starts(c, 2) == (0,) * 8


def test_dependency_prefill_resume_full_state_and_next_decode_match_reference():
    c = _ced_config()
    ref = DeepSeekV41Model(c, Weights(c), identity="synthetic@ced")
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@ced", ced_policy="dependency_exact")
    rc, pc = ref.make_cache(), model.make_cache()
    for length in (2, 16, 9):
        tokens = mx.array([[i % c.vocab_size for i in range(length)]])
        expected = ref(tokens, rc, logits_to_keep=1)
        actual = model.prefill_forward(tokens, pc)
        np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=1e-5, atol=1e-6)
        for a, b in zip(rc, pc):
            assert a.offset == b.offset
            for av, bv in zip(a.state, b.state):
                if av is not None:
                    np.testing.assert_allclose(np.array(av), np.array(bv), rtol=1e-5, atol=1e-6)
                else:
                    assert bv is None
        np.testing.assert_allclose(np.array(ref(mx.array([[1]]), rc)),
                                   np.array(model(mx.array([[1]]), pc)), rtol=1e-5, atol=1e-6)
    assert model.scheduled_kv_calls["ced_skipped_layer_rows"] > 0
    snapshot = [c.snapshot() for c in pc]
    output = model(mx.array([[2]]), pc)
    for state, saved in zip(pc, snapshot):
        state.restore(saved)
    np.testing.assert_array_equal(_bits(model(mx.array([[2]]), pc)), _bits(output))


def test_taps_keep_full_depth_and_verifier_preserves_token_order():
    c = _ced_config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@taps", ced_policy="dependency_exact")
    cache = model.make_cache()
    inputs = mx.array([[1, 2, 3]])
    logits, features = model.verify_rows(inputs, cache, (5, 6, 7))
    assert logits.shape == (1, 3, c.vocab_size)
    assert features.shape == (1, 3, c.hidden_size * 3)
    assert "ced_prefill_calls" not in model.scheduled_kv_calls
    other = model.make_cache()
    values = [model.forward_with_taps(inputs[:, i:i+1], other, (5, 6, 7)) for i in range(3)]
    np.testing.assert_array_equal(_bits(logits), _bits(mx.concatenate([v[0] for v in values], 1)))
    np.testing.assert_array_equal(_bits(features), _bits(mx.concatenate([v[1] for v in values], 1)))


def test_dependency_prefill_marks_exact_checkpoint_guard():
    from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows

    c = _ced_config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@ced-checkpoint", ced_policy="dependency_exact")
    with prefill_checkpoint_rows([16], width=16) as request:
        output = model.prefill_forward(mx.array([[1] * 16]), model.make_cache())
        assert request.applied == 1
        assert output.shape == (1, 1, c.vocab_size)
    with prefill_checkpoint_rows([10], width=16), pytest.raises(ValueError, match="exact final-row"):
        model.prefill_forward(mx.array([[1] * 16]), model.make_cache())


def test_packed_disk_apc_fork_and_partial_verification_commit(tmp_path):
    from mlx2.adapters.deepseek_v4_external import DeepSeekV4VerifyRows
    from mlx2.runtime.cow_cache import freeze_prompt_cache, restore_prompt_cache
    from mlx2.runtime.models.cache import (
        _resolve_cache_class,
        load_prompt_cache,
        save_prompt_cache,
    )

    c = _packed_config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@packed-apc", kv_storage_policy="native_packed")
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]]), cache)
    expected = [c.snapshot() for c in cache]
    path = tmp_path / "packed.safetensors"
    save_prompt_cache(str(path), cache)
    loaded = load_prompt_cache(str(path))
    assert _resolve_cache_class("DeepSeekV41PackedLayerCache") is DeepSeekV41PackedLayerCache
    assert [c.meta_state for c in loaded] == [c.meta_state for c in cache]
    frozen, _ = freeze_prompt_cache(cache, key="native-qat", tokens=(1, 2, 3), cache_type="deepseek_v41")
    branch = restore_prompt_cache(frozen)
    try:
        model(mx.array([[4, 5]]), branch)
        assert all(c.offset == 5 for c in branch)
        for state, saved in zip(cache, expected):
            for actual, wanted in zip(state.state, saved.arrays):
                if actual is not None:
                    np.testing.assert_array_equal(np.array(actual), np.array(wanted))
    finally:
        branch.close()
        frozen.close()
    owner = DeepSeekV4VerifyRows(model, [loaded], (3, 4))
    tx = owner.begin([3])
    tx.forward(mx.array([[4, 5, 6]]))
    committed = tx.commit([1])[0]
    model(mx.array([[4]]), cache)
    for a, b in zip(cache, committed):
        assert a.meta_state == b.meta_state
        for av, bv in zip(a.state, b.state):
            if av is not None:
                np.testing.assert_array_equal(np.array(av), np.array(bv))


def test_packed_budget_counts_codes_scales_and_keeps_decode_scratch():
    from mlx2.adapters.deepseek_v41_memory import DeepSeekV41CacheBudget
    from mlx2.runtime.models.deepseek_v41_config import DeepSeekV41Config

    c = DeepSeekV41Config()
    budget = DeepSeekV41CacheBudget.from_config(c, kv_storage_policy="native_packed")
    reference = DeepSeekV41CacheBudget.from_config(c)
    assert (budget.local_row_bytes, budget.pooled_row_bytes, budget.index_row_bytes) == (528, 288, 68)
    assert budget.local_live_bytes == 40 * 128 * 528
    assert budget.pooled_bytes(16384) == sum(budget._pool_capacity(16384, r) * 356 for r in (2, 2, 2, 1))
    assert budget.compression_fixed_bytes == reference.compression_fixed_bytes
    assert budget.project(16384) < reference.project(16384)
    assert budget.prefill_transient_bytes(16384, 128) > reference.prefill_transient_bytes(16384, 128)
    assert "native-FP8" in budget.as_dict()["physical_storage"]
    # The decoded workspace charge does not grow as forty full FP32 pools.
    assert budget.prefill_transient_bytes(1048576, 1) - budget.prefill_transient_bytes(65536, 1) < 2 << 30


def test_visual_router_bias_is_selected_per_row():
    c = config()
    weights = Weights(c)
    model = DeepSeekV41Model(c, weights, identity="synthetic@visual-router")
    block = model.layers[0]
    block.correction = mx.array([100., 100., 0., 0.])
    weights.data["blk.0.exp_probs_b_vl.bias"] = mx.array([0., 0., 100., 100.])
    ids, _ = block.router(mx.zeros((1, 2, c.hidden_size)), image_mask=np.array([False, True]))
    assert set(np.array(ids)[0, 0]) == {0, 1}
    assert set(np.array(ids)[0, 1]) == {2, 3}


def test_visual_prompt_chunks_once_and_preserves_global_checkpoint_row():
    from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows

    c = config(num_hidden_layers=2, compress_ratios=(0, 0), kv_source_layer_ids=(),
               index_source_layer_ids=(), candidate_source_layer_id=-1, sliding_window=3)
    weights = Weights(c)
    for i in range(c.num_hidden_layers):
        weights.data[f"blk.{i}.exp_probs_b_vl.bias"] = mx.zeros((c.n_routed_experts,))
    model = DeepSeekV41Model(c, weights, identity="synthetic@visual-chunks")
    inputs = mx.array([[1] * 1027])
    cache = model.make_cache()
    calls, chunks = [], []
    def prepare(tokens, cache, images):
        calls.append((tokens.shape, images))
        return weights.embedding(tokens), np.arange(tokens.shape[1]) < 1000
    model.vision_inputs = prepare
    forward = model._forward
    def capture(tokens, cache, **kwargs):
        chunks.append((tokens.shape[1], kwargs["image_mask"].sum()))
        return forward(tokens, cache, **kwargs)
    model._forward = capture
    with prefill_checkpoint_rows([1027], width=1027) as request:
        output = model.prefill_forward(inputs, cache, image_inputs="synthetic-image")
        assert request.applied == 1
        assert output.shape == (1, 1, c.vocab_size)
    assert calls == [((1, 1027), "synthetic-image")]
    assert chunks == [(1024, 1000), (3, 0)]
    assert all(state.offset == 1027 and not state.pending for state in cache)


def test_production_head_width_qat_exact_future_state_and_decode_at_odd_seams():
    # Production D512/indexD128/W128/QAT boundaries, synthetic small FFN and
    # eight layers/two heads. This is not a production-checkpoint result.
    c = config(num_hidden_layers=8, compress_ratios=(0, 2, 2, 1, 1, 1, 1, 1),
               kv_source_layer_ids=(1, 3), index_source_layer_ids=(1, 3, 5, 7),
               head_dim=512, index_head_dim=128, sliding_window=128,
               simulate_quantization=True, activation_dtype="bfloat16")
    ref = DeepSeekV41Model(c, Weights(c), identity="synthetic@production-width-qat")
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@production-width-qat",
                             kv_storage_policy="native_packed", candidate_score_policy="candidate_only",
                             ced_policy="dependency_exact")
    a, b = ref.make_cache(), model.make_cache()
    for length in (129, 131, 257, 3):
        ids = mx.array([[i % c.vocab_size for i in range(length)]])
        np.testing.assert_array_equal(_bits(ref(ids, a, logits_to_keep=1)), _bits(model.prefill_forward(ids, b)))
        for u, v in zip(a, b):
            decoded = (v.keys, v.buf_kv, v.buf_gate, v.pooled, v.index_keys, v.hash_tail)
            for left, right in zip(u.state, decoded):
                if left is not None:
                    np.testing.assert_array_equal(_bits(left), _bits(right))
                else:
                    assert right is None
        np.testing.assert_array_equal(_bits(ref(mx.array([[1]]), a)), _bits(model(mx.array([[1]]), b)))
    assert model.scheduled_kv_calls["ced_skipped_layer_rows"] > 0
    assert model.scheduled_kv_calls["candidate_only_unscored_rows"] > 0


@pytest.mark.parametrize("limit", [True, 0, -1, 65536.0, 1048577])
def test_context_geometry_rejects_invalid_bounds(limit):
    c = _packed_config()
    with pytest.raises(ValueError, match="context limit"):
        DeepSeekV41Model(c, Weights(c), identity="synthetic@context", context_limit=limit)


def test_long_context_admission_requires_packed_candidates_and_binds_identity():
    c = _packed_config()
    for options in ({}, {"kv_storage_policy": "native_packed"}, {"candidate_score_policy": "candidate_only"}):
        with pytest.raises(ValueError, match="requires native packed"):
            DeepSeekV41Model(c, Weights(c), identity="synthetic@context", context_limit=65537, **options)
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@context", context_limit=1048576,
                             kv_storage_policy="native_packed", candidate_score_policy="candidate_only")
    assert model.context_limit == 1048576
    assert ":context_limit=1048576" in model.identity
    assert DeepSeekV41Model(c, Weights(c), identity="synthetic@context").context_limit == 65536


@pytest.mark.parametrize("length", [1, 3])
def test_large_packed_pool_attention_decodes_only_selected_rows(monkeypatch, length):
    from mlx2.runtime.models import deepseek_v41_packed_qat as codec
    from mlx2.runtime.models.deepseek_v41 import selected_attention

    rng = np.random.default_rng(200)
    pool = produce(mx.array(rng.normal(size=(1, 10003, 512)).astype(np.float32)), "compressed_fp4")
    full = decode(pool)
    q = mx.array(rng.normal(size=(1, 2, length, 512)).astype(np.float32))
    local = mx.array(rng.normal(size=(1, 1, 128, 512)).astype(np.float32))
    ids = mx.array(np.sort(rng.choice(10003, size=(length, 512), replace=False), axis=-1)[None].astype(np.uint32))
    valid = mx.array(rng.random((1, length, 512)) > .25)
    mask = mx.ones((1, 1, length, 128), mx.bool_)
    sinks = mx.array([.2, -.1])
    expected = selected_attention(q, local, mask, full, ids, valid, sinks, workspace_bytes=1)
    staged = []
    ordinary_decode = codec.decode
    def bounded(packed):
        rows = int(np.prod(packed.shape[:-1]))
        staged.append(rows)
        assert rows <= 512
        return ordinary_decode(packed)
    monkeypatch.setattr(codec, "decode", bounded)
    counters = {}
    actual = selected_attention(q, local, mask, None, ids, valid, sinks,
                                 packed_pooled=pool, counters=counters, workspace_bytes=1)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))
    assert staged == [512] * length
    assert counters["packed_selected_staging_max_rows"] == 512


def test_wired_native_model_never_materializes_full_float_pools(monkeypatch):
    c = _packed_config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@bounded-pool",
                             kv_storage_policy="native_packed", candidate_score_policy="candidate_only",
                             selected_kv_policy="decode_reuse")
    def forbidden(self):
        raise AssertionError("native serving materialized a full floating pool")
    monkeypatch.setattr(DeepSeekV41PackedLayerCache, "pooled", property(forbidden))
    monkeypatch.setattr(DeepSeekV41PackedLayerCache, "index_keys", property(forbidden))
    cache = model.make_cache()
    model(mx.array([[1, 2, 3, 4, 5]]), cache)
    model(mx.array([[6]]), cache)
    for state in cache:
        state.snapshot()
    assert model.scheduled_kv_calls["packed_selected_staging_max_rows"] <= 5 * c.index_topk
    assert model.scheduled_kv_calls["decode_gather_hit"] > 0


def test_full_source_indexer_tiles_native_index_decode(monkeypatch):
    from types import SimpleNamespace

    from mlx2.runtime.models import deepseek_v41_packed_qat as codec
    from mlx2.runtime.models.deepseek_v41 import Attention

    rng = np.random.default_rng(44)
    count, length = 8201, 2
    packed = produce(mx.array(rng.normal(size=(1, count, 32)).astype(np.float32)), "index_fp4")
    full = decode(packed)
    query = mx.array(rng.normal(size=(1, length, 64)).astype(np.float32))
    c = SimpleNamespace(index_topk=512, index_n_heads=2, index_head_dim=32,
                        candidate_source_layer_id=20, candidate_block_size=8, candidate_topk_blocks=2)
    def attention():
        a = object.__new__(Attention)
        a.c, a.model, a.layer, a.ratio = c, SimpleNamespace(index_policy="reference", scheduled_kv_calls={}), 20, 1
        a.linear = lambda x, name: query if name == "indexer.attn_q_b" else mx.ones((1, length, 2))
        a.rotate_quantized = lambda x, *args, **kwargs: x
        return a
    reference, model = attention(), attention()
    x, qr = mx.zeros((1, length, 8)), mx.zeros((1, length, 4))
    ref, actual = {}, {}
    reference.select(x, qr, SimpleNamespace(index_keys=full), count-length, ref)
    class Source:
        _pool_count = count
        packed_index_keys = packed
        @property
        def index_keys(self):
            raise AssertionError("full index decode")
    staged = []
    ordinary_decode = codec.decode
    def bounded(value):
        staged.append(value.shape[1])
        assert value.shape[1] <= 4096
        return ordinary_decode(value)
    monkeypatch.setattr(codec, "decode", bounded)
    model.select(x, qr, Source(), count-length, actual)
    for key in ("indices", "selected_mask", "candidates"):
        np.testing.assert_array_equal(np.array(actual[key]), np.array(ref[key]))
    assert staged == [4096, 4096, 9]


def test_packed_storage_refuses_unimplemented_source_attention_math():
    c = _packed_config()
    for policy in ("all_retained_broadcast", "omlx_nax_bf16_online_v1"):
        with pytest.raises(ValueError, match="reference selected attention"):
            DeepSeekV41Model(c, Weights(c), identity="synthetic@packed-incompatible",
                             kv_storage_policy="native_packed", source_attention_math_policy=policy)


@pytest.mark.parametrize("count", [511, 512, 513, 16383, 16384, 16385])
def test_production_index_candidate_boundaries_with_packed_keys(count):
    from types import SimpleNamespace

    from mlx2.runtime.models.deepseek_v41 import Attention, candidate_blocks
    from mlx2.runtime.models.deepseek_v41_packed_qat import gather_decode

    rng = np.random.default_rng(16385)
    length = 3
    packed = produce(mx.array(rng.normal(size=(1, count, 128)).astype(np.float32)), "index_fp4")
    full = decode(packed)
    assert packed.nbytes == count * 68
    query = rng.normal(size=(1, length, 32 * 128)).astype(np.float32)
    query[:, 0] = 0  # Exact score ties must keep chronological global IDs.
    query = mx.array(query)
    weights = mx.array(rng.uniform(.1, 1., size=(1, length, 32)).astype(np.float32))
    c = SimpleNamespace(index_topk=512, index_n_heads=32, index_head_dim=128,
                        candidate_source_layer_id=20, candidate_block_size=8, candidate_topk_blocks=2048)
    lens = mx.array([count - 2, count - 1, count])[None, :, None]
    scores = mx.array(rng.normal(size=(1, length, count)).astype(np.float32))
    scores = mx.where(mx.arange(count)[None, None] < lens, scores, -mx.inf)
    candidates = candidate_blocks(scores, lens, 2048, 8)
    def attention(policy):
        a = object.__new__(Attention)
        a.c, a.model, a.layer, a.ratio = c, SimpleNamespace(index_policy="reference", candidate_score_policy=policy, scheduled_kv_calls={}), 24, 1
        a.linear = lambda x, name: query if name == "indexer.attn_q_b" else weights
        a.rotate_quantized = lambda x, *args, **kwargs: quantize(x, "index_fp4").astype(mx.bfloat16).astype(mx.float32)
        return a
    class Source:
        _pool_count = count
        packed_index_keys = packed
        def gather_index_keys(self, indices):
            return gather_decode(packed, indices)
        @property
        def index_keys(self):
            raise AssertionError("candidate scorer decoded complete native history")
    ref, actual = {"candidates": candidates}, {"candidates": candidates}
    x, qr = mx.zeros((1, length, 8)), mx.zeros((1, length, 4))
    attention("reference").select(x, qr, SimpleNamespace(index_keys=full), count-length, ref)
    model = attention("candidate_only")
    model.select(x, qr, Source(), count-length, actual)
    for key in ("indices", "selected_mask"):
        np.testing.assert_array_equal(np.array(actual[key]), np.array(ref[key]))
    assert actual["indices"].shape == (1, length, min(512, count))
    assert model.model.scheduled_kv_calls["candidate_only_unscored_rows"] == length * max(0, count - 16384)


@pytest.mark.parametrize("first_length", [1023, 1024])
def test_ratio_two_1023_1024_1025_frontiers_preserve_production_width_state_bits(first_length):
    c = config(num_hidden_layers=8, compress_ratios=(0, 2, 2, 1, 1, 1, 1, 1),
               kv_source_layer_ids=(1, 3), index_source_layer_ids=(1, 3, 5, 7),
               head_dim=512, index_head_dim=128, sliding_window=128, index_topk=512,
               candidate_topk_blocks=2048, candidate_block_size=8,
               simulate_quantization=True, activation_dtype="bfloat16")
    ref = DeepSeekV41Model(c, Weights(c), identity="synthetic@ratio2-production-boundaries")
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@ratio2-production-boundaries",
                             kv_storage_policy="native_packed", candidate_score_policy="candidate_only",
                             ced_policy="dependency_exact")
    a, b = ref.make_cache(), model.make_cache()
    frames = ((first_length, first_length), *((1, n) for n in range(first_length + 1, 1026)))
    for length, frontier in frames:
        ids = mx.array([[i % c.vocab_size for i in range(length)]])
        np.testing.assert_array_equal(_bits(ref(ids, a, logits_to_keep=1)), _bits(model.prefill_forward(ids, b)))
        for u, v in zip(a, b):
            assert u.offset == v.offset == frontier
            decoded = (v.keys, v.buf_kv, v.buf_gate, v.pooled, v.index_keys, v.hash_tail)
            for left, right in zip(u.state, decoded):
                if left is not None:
                    np.testing.assert_array_equal(_bits(left), _bits(right))
                else:
                    assert right is None
        assert b[1]._pool_count == frontier // 2
        assert b[1].buf_kv.shape[1] == frontier % 2
        saved = [[state.snapshot() for state in cache] for cache in (a, b)]
        np.testing.assert_array_equal(_bits(ref(mx.array([[1]]), a)), _bits(model(mx.array([[1]]), b)))
        for cache, snapshots in zip((a, b), saved):
            for state, snapshot in zip(cache, snapshots):
                state.restore(snapshot)
