"""Synthetic CPU equality for the opt-in V4.1 index tiling/selection laws.

Never model qualification or GPU evidence: Metal GEMM shape choice is not
exercised here.
"""
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import deepseek_v41 as v41
from mlx2.runtime.models import deepseek_v41_index_select as sel
from mlx2.runtime.models.deepseek_v4 import stable_pool_topk
from mlx2.runtime.models.deepseek_v41_packed_qat import PackedQAT, decode, produce


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old = mx.default_device()
    mx.set_default_device(mx.cpu)

    def forbidden(*args, **kwargs):
        raise AssertionError("index CPU oracle reached a Metal kernel")
    monkeypatch.setattr(mx.fast, "metal_kernel", forbidden)
    yield
    mx.set_default_device(old)


HEADS, DIM = 4, 32


def _scores(rows, width, seed, *, ties=False, neg_inf=0.0):
    rng = np.random.default_rng(seed)
    value = rng.normal(size=(1, rows, width)).astype(np.float32)
    if ties:
        value = np.maximum(np.round(value * 2) / 2, 0)  # many exact ReLU zeros and equal values
    if neg_inf:
        value = np.where(rng.random(value.shape) < neg_inf, -np.inf, value)
    return mx.array(value)


@pytest.mark.parametrize("width,k", [(1, 1), (7, 3), (64, 64), (513, 512), (2000, 17), (4096, 512)])
@pytest.mark.parametrize("ties,neg_inf", [(False, 0.0), (True, 0.0), (True, 0.5), (False, 0.97)])
def test_compact_topk_equals_stable_pool_topk(width, k, ties, neg_inf):
    scores = _scores(3, width, width + k, ties=ties, neg_inf=neg_inf)
    ref = stable_pool_topk(scores, k)
    new = sel.stable_topk_compact(scores, k)
    assert new.dtype == ref.dtype == mx.uint32
    np.testing.assert_array_equal(np.array(new), np.array(ref))


@pytest.mark.parametrize("width", [1, 7, 8, 9, 100, 511, 4097])
@pytest.mark.parametrize("count", [1, 3, 64])
def test_compact_candidate_blocks_equals_reference(width, count):
    lens = np.array([1, max(1, width // 2), width], np.int32)[None, :, None]
    score = np.array(_scores(3, width, width * 7 + count, ties=True))
    score = np.where(np.arange(width)[None, None] < lens, score, -np.inf)
    args = (mx.array(score), mx.array(lens), count, 8)
    np.testing.assert_array_equal(np.array(sel.candidate_blocks_compact(*args)),
                                  np.array(v41.candidate_blocks(*args)))


@pytest.mark.parametrize("count,capacity,density", [(10, 4, .2), (10, 4, .9), (100, 40, .3), (100, 100, .5),
                                                    (257, 16, 0.0), (257, 16, 1.0), (300, 64, .15)])
def test_compact_candidate_ids_equals_argsort_law(count, capacity, density):
    rng = np.random.default_rng(count + capacity)
    prior = mx.array(rng.random((1, 4, count)) < density)
    positions = mx.arange(count, dtype=mx.uint32)[None, None]
    ref = mx.sort(mx.argsort(mx.where(prior, positions, count + positions), axis=-1)[..., :capacity], axis=-1)
    new = sel.candidate_ids_compact(prior, capacity)
    assert new.dtype == ref.dtype
    np.testing.assert_array_equal(np.array(new), np.array(ref))


def test_plan_bounds_head_scratch_and_reduced_row():
    rows, chunk = sel.plan_key_chunked(1024, 65536, 32, head_scratch_bytes=16 << 20)
    assert rows == 64 and rows * 65536 * 4 <= sel.INDEX_REDUCED_SCORE_BYTES
    assert chunk == 2048 and rows * 32 * chunk * 4 <= 16 << 20
    assert sel.plan_key_chunked(1, 1000, 32, head_scratch_bytes=16 << 20) == (1, 1000)
    with pytest.raises(ValueError):
        sel.plan_key_chunked(0, 1, 1, head_scratch_bytes=1)


def _selector(count, *, layer, ratio, length, tiling="reference", select="reference", policy="reference",
              candidate_only=False, packed=False, cap_blocks=3, topk=7):
    c = SimpleNamespace(index_topk=topk, index_n_heads=HEADS, index_head_dim=DIM,
                        candidate_source_layer_id=20, candidate_block_size=8,
                        candidate_topk_blocks=cap_blocks)
    m = SimpleNamespace(index_policy=policy, scheduled_kv_calls={},
                        source_index_tiling_policy=tiling, source_index_select_policy=select,
                        candidate_score_policy="candidate_only" if candidate_only else "reference")
    a = object.__new__(v41.Attention)
    a.c, a.model, a.layer, a.ratio = c, m, layer, ratio
    rng = np.random.default_rng(count * 31 + length)
    query = mx.array(rng.normal(size=(1, length, HEADS * DIM)).astype(np.float32))
    weight = mx.array(rng.normal(size=(1, length, HEADS)).astype(np.float32))

    def linear(x, name):
        return query if name == "indexer.attn_q_b" else weight
    a.linear = linear
    a.rotate_quantized = lambda x, *args, **kwargs: x
    raw = mx.array(rng.normal(size=(1, count, DIM)).astype(np.float32))
    if packed:
        planes = produce(raw[:, None], "index_fp4")
        packed_keys = PackedQAT(planes.codes.squeeze(1), planes.scales.squeeze(1), planes.kind)
        source = SimpleNamespace(packed_index_keys=packed_keys, _pool_count=count,
                                 gather_index_keys=lambda ids: decode(PackedQAT(
                                     packed_keys.codes[0, ids[0]][None], packed_keys.scales[0, ids[0]][None], "index_fp4")))
    else:
        source = SimpleNamespace(index_keys=raw)
    shared = {}
    start = max(0, count * ratio - length)
    if layer > 20:
        lens = (np.arange(start, start + length) + 1) // ratio
        prior = (rng.random((1, length, count)) < .4) & (np.arange(count)[None, None] < lens[None, :, None])
        shared["candidates"] = mx.array(prior)
    x = mx.zeros((1, length, 8), mx.float32)
    qr = mx.zeros((1, length, 4), mx.float32)
    return a, source, shared, x, qr, start


CASES = [(2, 2), (20, 1), (24, 1)]


@pytest.mark.parametrize("layer,ratio", CASES)
@pytest.mark.parametrize("count,length", [(5, 3), (40, 17), (200, 33), (513, 9), (1000, 70)])
@pytest.mark.parametrize("tiling,select", [("key_chunked_v1", "reference"), ("reference", "compact_v1"),
                                           ("key_chunked_v1", "compact_v1")])
@pytest.mark.parametrize("policy", ["reference", "finite_all_retained"])
@pytest.mark.parametrize("packed", [False, True])
def test_select_laws_match_reference_bitwise(monkeypatch, layer, ratio, count, length, tiling, select, policy, packed):
    # Force the reference to take tiny row tiles and the chunked law to split keys.
    monkeypatch.setattr(v41, "INDEX_SCORE_SCRATCH_BYTES", HEADS * 4 * 64)
    monkeypatch.setattr(sel, "INDEX_REDUCED_SCORE_BYTES", 4 * 4096)
    monkeypatch.setattr(v41, "PACKED_INDEX_KEY_CHUNK", 48)
    args = dict(layer=layer, ratio=ratio, length=length, policy=policy, packed=packed)
    r, rs, ref, x, qr, start = _selector(count, **args)
    f, fs, fast, _, _, _ = _selector(count, tiling=tiling, select=select, **args)
    if layer > 20:
        fast["candidates"] = ref["candidates"]
    r.select(x, qr, rs, start, ref)
    f.select(x, qr, fs, start, fast)
    for key in ("indices", "selected_mask", "candidates"):
        assert (key in ref) == (key in fast)
        if key in ref:
            assert ref[key].dtype == fast[key].dtype and ref[key].shape == fast[key].shape
            np.testing.assert_array_equal(np.array(ref[key]), np.array(fast[key]))
    counters = f.model.scheduled_kv_calls
    scored = "all_retained_index" not in counters
    if scored and tiling == "key_chunked_v1":
        assert counters["index_key_chunked_calls"] == 1 and counters["index_key_chunked_tiles"] >= 1
        assert counters["index_key_chunked_key_chunks"] >= counters["index_key_chunked_tiles"]
    if scored and select == "compact_v1":
        assert counters["index_compact_select"] >= 1
    if tiling == "reference" and select == "reference":
        assert not any(k.startswith("index_") for k in counters)


def test_key_chunked_actually_widens_tiles_and_splits_keys(monkeypatch):
    monkeypatch.setattr(v41, "INDEX_SCORE_SCRATCH_BYTES", HEADS * 4 * 64)
    monkeypatch.setattr(sel, "INDEX_REDUCED_SCORE_BYTES", 4 * 4096)
    a, source, shared, x, qr, start = _selector(1000, layer=20, ratio=1, length=70, tiling="key_chunked_v1")
    a.select(x, qr, source, start, shared)
    counters = a.model.scheduled_kv_calls
    # Reference: 64-element head scratch / (4 heads x 1000 keys) -> 1-row tiles (70 tiles).
    # Chunked: 4 rows per tile (16 KiB / 4 KB rows) -> 18 tiles; 4 x 4 x chunk <= 1 KiB -> chunk 16.
    assert counters["index_key_chunked_tiles"] == 18
    assert counters["index_key_chunked_tiles_avoided"] == 70 - 18
    assert counters["index_key_chunked_key_chunks"] == 18 * 63


@pytest.mark.parametrize("count,length", [(40, 1), (300, 5)])
def test_candidate_only_compact_ids_match_and_tiling_declines(monkeypatch, count, length):
    monkeypatch.setattr(v41, "INDEX_SCORE_SCRATCH_BYTES", HEADS * 4 * 64)
    args = dict(layer=24, ratio=1, length=length, candidate_only=True)
    r, rs, ref, x, qr, start = _selector(count, **args)
    f, fs, fast, _, _, _ = _selector(count, tiling="key_chunked_v1", select="compact_v1", **args)
    fast["candidates"] = ref["candidates"]
    rs.gather_index_keys = lambda ids: rs.index_keys[0, ids[0]][None]
    fs.gather_index_keys = lambda ids: fs.index_keys[0, ids[0]][None]
    r.select(x, qr, rs, start, ref)
    f.select(x, qr, fs, start, fast)
    for key in ("indices", "selected_mask"):
        np.testing.assert_array_equal(np.array(ref[key]), np.array(fast[key]))
    counters = f.model.scheduled_kv_calls
    assert counters["index_key_chunked_declined_candidate_only"] == 1
    assert counters["index_compact_candidate_ids"] >= 1 and "index_key_chunked_tiles" not in counters


def test_source_policy_admits_opt_in_laws_and_keeps_defaults_off():
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter, source_policy
    defaults = DeepSeekV41SourceAdapter.default_route_execution_policy["ordinary"]
    assert "deepseek_source_index_tiling" not in defaults and "deepseek_source_index_select" not in defaults
    owned = DeepSeekV41SourceAdapter.adapter_owned_default_policy_keys
    assert {"deepseek_source_index_tiling", "deepseek_source_index_select"} <= owned
    result = source_policy({"deepseek_source_index_tiling": "key_chunked_v1",
                            "deepseek_source_index_select": "compact_v1"})
    assert result["deepseek_source_index_tiling"] == "key_chunked_v1"
    assert result["deepseek_source_index_select"] == "compact_v1"
    assert "deepseek_source_index_tiling" not in source_policy({})
    for bad in ({"deepseek_source_index_tiling": "on"}, {"deepseek_source_index_select": True}):
        with pytest.raises(ValueError, match="invalid Source policy"):
            source_policy(bad)


def test_source_policy_refuses_ast_preparation_combination():
    from mlx2.adapters.deepseek_v41_source import source_policy
    with pytest.raises(ValueError, match="ordinary indexer only"):
        source_policy({"deepseek_source_index_select": "compact_v1",
                       "deepseek_source_dspark_prepare": "oracle_v1",
                       "deepseek_source_dspark_target_forward": "fully_batched_m1_v1",
                       "deepseek_source_dspark_target_expert_law": "source_ast_m1_rows_v1",
                       "deepseek_source_expert_math": "routed_fused_fp32"})


def test_identity_binding_only_when_selected():
    from mlx2.adapters.deepseek_v41_source import DeepSeekV41SourceAdapter
    adapter = object.__new__(DeepSeekV41SourceAdapter)
    adapter.kernel_policy, adapter.numerics_revision, adapter.profile = {}, "r", "p"
    adapter._bind_index_select_identity()
    assert (adapter.numerics_revision, adapter.profile) == ("r", "p")
    adapter.kernel_policy = {"deepseek_source_index_tiling": "key_chunked_v1"}
    adapter._bind_index_select_identity()
    assert adapter.numerics_revision.startswith("r:deepseek_source_index_tiling=key_chunked_v1:implementation=")
    assert adapter.profile.startswith("p:indexsel-")
