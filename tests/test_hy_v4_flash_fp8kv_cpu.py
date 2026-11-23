"""Hy4-flash fp8 latent KV (opt-in, approximate) CPU tests on the tiny checkpoint.

CPU only: these check the dequantize-before-SDPA path, which is the only
path implemented (there is no fp8-MLA Metal kernel).  No real weights.
"""

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from mlx2.runtime.models import hy_v4_flash_fp8kv as F  # noqa: E402
from test_hy_v4_flash_cpu import (  # noqa: E402
    LONG, TOKENS, _drain, _greedy, _tiny_artifact, build, checkpoint, tiny_config)


@pytest.fixture(scope="module")
def pair():
    """The same tiny weights twice: native rows and fp8 rows."""
    config = tiny_config()
    W = checkpoint(config)
    native, drafter = build(config, W)
    fp8, fp8_drafter = build(config, W)
    fp8.args.latent_kv = "fp8"
    return config, W, native, fp8, drafter, fp8_drafter


# ------------------------------------------------------------ NumPy oracle --

def e4m3_ref(x):
    """Round-to-nearest-even E4M3 (bias 7, subnormals below 2**-6), |x| <= 448."""
    x = np.asarray(x, np.float64)
    a = np.abs(x)
    e = np.maximum(np.floor(np.log2(np.where(a > 0, a, 1))), -6)
    step = 2.0 ** (e - 3)
    return np.sign(x) * np.minimum(np.round(a / step) * step, 448)


def bf16_ref(x):
    """Round-to-nearest-even fp32 -> bf16, back to fp32."""
    bits = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def fp8_row_ref(rows, rank):
    rows = np.asarray(rows, np.float32)
    lead = rows.shape[:-1]
    width = min(128, rank)
    lat = rows[..., :rank].reshape(*lead, rank // width, width).astype(np.float64)
    amax = np.maximum(np.abs(lat).max(-1, keepdims=True), 1e-30)
    scale = 2.0 ** np.ceil(np.log2(amax / 448.0))
    deq = (e4m3_ref(np.clip(lat / scale, -448, 448)) * scale).reshape(*lead, rank)
    return np.concatenate([deq, bf16_ref(rows[..., rank:])], axis=-1)


@pytest.mark.parametrize("rank,rope", [(512, 64), (12, 4)])
def test_round_trip_matches_numpy_e4m3_oracle(rank, rope):
    rng = np.random.default_rng(11)
    x = rng.standard_normal((3, 9, rank + rope)).astype(np.float32)
    x[..., :rank] *= np.exp(rng.uniform(-12, 8, size=(3, 9, 1))).astype(np.float32)
    x[0, 0, : min(128, rank)] = 0.0  # an all-zero tile
    x[1, 2, 5] = 3e4  # a single large outlier in a tile
    packed = F.pack_rows(mx.array(x), rank)
    assert packed.dtype == mx.uint8 and packed.shape[-1] == F.row_bytes(rank, rope)
    got = np.array(F.unpack_rows(packed, rank, mx.float32))
    np.testing.assert_array_equal(got, fp8_row_ref(x, rank).astype(np.float32))
    # Scales are exact powers of two and no code saturates.
    tiles = rank // min(128, rank)
    scales = np.array(mx.view(mx.contiguous(packed[..., rank:rank + 4 * tiles]), mx.float32))
    assert np.all(np.log2(scales) == np.round(np.log2(scales)))
    # bf16 activations dequantize to the same values (codes * 2**k is exact).
    bf = np.array(F.unpack_rows(packed, rank, mx.bfloat16).astype(mx.float32))
    np.testing.assert_array_equal(bf, got)


def test_scale_boundary_and_fp16_range():
    # amax exactly 448 * 2**k: scale 2**k, top code 448 (no clip, no bump).
    x = np.zeros((1, 16), np.float32)
    x[0, :4] = [448 * 4.0, -3.0, 1.0, 0.0]
    got = np.array(F.unpack_rows(F.pack_rows(mx.array(x), 12), 12, mx.float32))
    np.testing.assert_array_equal(got, fp8_row_ref(x, 12).astype(np.float32))
    assert got[0, 0] == 448 * 4.0
    big = np.full((1, 16), 65504.0, np.float32)
    half = np.array(F.unpack_rows(F.pack_rows(mx.array(big), 12), 12, mx.float16).astype(mx.float32))
    assert np.isfinite(half).all()


def test_fp8_cache_declares_append_only_buffers_for_recovery():
    from mlx2.runtime.cow_cache import _append_only_fields

    fp8 = H.HY4FlashFP8LayerCache(H.INDEXED, rank=12, rope=4)
    assert _append_only_fields(fp8) == _append_only_fields(H.HY4FlashLayerCache(H.INDEXED))
    assert _append_only_fields(fp8)


def test_hy4_row_is_656_bytes_and_43_percent_smaller():
    assert F.row_bytes(512, 64) == 656
    assert F.native_row_bytes(512, 64) == 1152
    assert 1 - 656 / 1152 == pytest.approx(0.4306, abs=1e-4)


def test_nbytes_reflects_packed_rows_at_hy4_geometry():
    a = H.ModelArgs()
    T = 1024
    rows = mx.random.normal((1, T, a.kv_lora_rank + a.qk_rope_head_dim)).astype(mx.bfloat16)
    native = H.HY4FlashLayerCache(H.SLIDING, window=a.sliding_window)
    fp8 = H.HY4FlashFP8LayerCache(H.SLIDING, window=a.sliding_window, rank=a.kv_lora_rank,
                                  rope=a.qk_rope_head_dim)
    for cache in (native, fp8):
        cache.append(rows[:, :1])  # first growth step allocates 256 rows
    assert native.nbytes == 256 * 1152 and fp8.nbytes == 256 * 656
    ind_native = H.HY4FlashLayerCache(H.INDEXED)
    ind_fp8 = H.HY4FlashFP8LayerCache(H.INDEXED, rank=a.kv_lora_rank, rope=a.qk_rope_head_dim)
    for cache in (ind_native, ind_fp8):
        cache.append(rows)
    assert ind_native.kv_data.nbytes == T * 1152 and ind_fp8.kv_data.nbytes == T * 656
    assert ind_fp8.kv_data.dtype == mx.uint8


# --------------------------------------------------------------- default --

def test_default_is_native_and_reads_rows_in_place(pair):
    config, W, native, *_ = pair
    assert H.ModelArgs.from_hf_config(config).latent_kv == "native"
    caches = native.make_cache()
    assert all(type(c) is H.HY4FlashLayerCache for c in caches)
    assert all(c.cache_operations_contract.exact_prefix for c in caches)
    rows = mx.zeros((1, 3, 16))
    assert caches[0].latent_rows(rows, mx.bfloat16) is rows  # no copy, no cast
    with pytest.raises(ValueError):
        H.ModelArgs.from_hf_config(tiny_config(latent_kv="fp4"))


# ---------------------------------------------------------------- fp8 math --

def _rel_rms(a, b):
    return float(np.sqrt(((a - b) ** 2).mean() / (a ** 2).mean()))


def test_fp8_model_equals_native_model_on_round_tripped_keys(pair, monkeypatch):
    """The whole fp8 path is exactly "store each key row's fp8 round trip"."""
    _, _, native, fp8, _, _ = pair
    assert all(type(c) is H.HY4FlashFP8LayerCache for c in fp8.make_cache())
    got = np.array(fp8(mx.array([LONG])))
    project = H.Attention.project

    def round_tripped(self, x, offset):
        q, q_resid, keys = project(self, x, offset)
        rows = F.unpack_rows(F.pack_rows(keys, self.kv_lora_rank), self.kv_lora_rank, keys.dtype)
        return q, q_resid, rows

    monkeypatch.setattr(H.Attention, "project", round_tripped)
    np.testing.assert_array_equal(got, np.array(native(mx.array([LONG]))))


@pytest.mark.parametrize("layer", [0, 2])  # indexed, sliding
def test_fp8_single_attention_error_is_e4m3_sized(pair, layer):
    _, _, native, _, _, _ = pair
    attn = native.model.layers[layer].self_attn
    x = mx.random.normal((1, 40, 32), key=mx.random.key(0))
    exact = H.HY4FlashLayerCache(attn.kind, window=8)
    fp8 = H.HY4FlashFP8LayerCache(attn.kind, window=8, rank=12, rope=4)
    err = _rel_rms(np.array(attn._attend(x, exact)), np.array(attn._attend(x, fp8)))
    print(f"layer {layer} ({attn.kind}) latent output rel RMS error {err:.4f}")
    assert err < 0.06  # E4M3 has 3 mantissa bits: ~3% RMS per element


def test_fp8_logits_vs_native_report(pair, monkeypatch):
    """Whole-model drift on the tiny random checkpoint (reported, loosely bounded).

    The tiny random model is chaotic: 3% Gaussian noise on the native key
    rows moves its logits about as much as fp8 rows do, so this is a smoke
    bound, not a quality claim.  Real quality needs the GSM8K A/B on real
    weights (not run: CPU only).
    """
    _, _, native, fp8, _, _ = pair
    rows = {}
    for tokens in (TOKENS, LONG):
        a = np.array(native(mx.array([tokens]))[0])
        b = np.array(fp8(mx.array([tokens]))[0])
        rows[len(tokens)] = (float(np.abs(a - b).max()), float(np.mean(a.argmax(-1) == b.argmax(-1))),
                             _rel_rms(a, b))
        print(f"fp8 vs native, {len(tokens)} tokens: max |dlogit| {rows[len(tokens)][0]:.3f} "
              f"(max |logit| {np.abs(a).max():.2f}), top-1 {rows[len(tokens)][1]:.3f}, "
              f"rel RMS {rows[len(tokens)][2]:.3f}")
    project = H.Attention.project

    def noisy(self, x, offset):
        q, q_resid, keys = project(self, x, offset)
        return q, q_resid, keys * (1 + 0.03 * mx.random.normal(keys.shape, key=mx.random.key(1)))

    a = np.array(native(mx.array([LONG]))[0])
    monkeypatch.setattr(H.Attention, "project", noisy)
    control = _rel_rms(a, np.array(native(mx.array([LONG]))[0]))
    print(f"control: native + 3% key noise, {len(LONG)} tokens: rel RMS {control:.3f}")
    assert rows[len(LONG)][2] < 1.5 * control
    assert all(np.isfinite(v).all() for v in rows.values())


def test_fp8_chunked_stepwise_and_mask_paths_match_full(pair, monkeypatch):
    _, _, _, fp8, _, _ = pair
    full = np.array(fp8(mx.array([TOKENS])))
    cache = fp8.make_cache()
    parts = [fp8(mx.array([TOKENS[:9]]), cache), fp8(mx.array([TOKENS[9:15]]), cache)]
    parts += [fp8(mx.array([[t]]), cache) for t in TOKENS[15:]]
    np.testing.assert_allclose(np.array(mx.concatenate(parts, axis=1)), full, rtol=1e-4, atol=1e-4)
    monkeypatch.setattr(H, "GATHER_MAX_QUERY", 0)
    np.testing.assert_allclose(np.array(fp8(mx.array([TOKENS]))), full, rtol=1e-4, atol=1e-4)


def test_fp8_trim_rollback_is_exact_on_packed_buffer(pair):
    _, _, _, fp8, _, _ = pair
    cache = fp8.make_cache()
    fp8(mx.array([TOKENS[:18]]), cache)
    before = [np.array(H._buf_view(c, c._kv)) for c in cache]
    fp8(mx.array([[40, 41, 42, 43, 44]]), cache)  # a rejected block
    for c in cache:
        c.trim(5)
    for c, rows in zip(cache, before):
        np.testing.assert_array_equal(np.array(H._buf_view(c, c._kv)), rows)
    after = np.array(fp8(mx.array([TOKENS[18:]]), cache))
    fresh = fp8.make_cache()
    fp8(mx.array([TOKENS[:18]]), fresh)
    np.testing.assert_array_equal(after, np.array(fp8(mx.array([TOKENS[18:]]), fresh)))


def test_fp8_state_round_trip_and_apc_refuses_approximate_rows(pair):
    from mlx2.runtime.apc_v2 import inspect_apc_capabilities

    _, _, native, fp8, _, _ = pair
    cache = fp8.make_cache()
    fp8(mx.array([TOKENS[:13]]), cache)
    restored = [type(c).from_state(c.state, c.meta_state) for c in cache]
    assert all(type(r) is H.HY4FlashFP8LayerCache for r in restored)
    a = np.array(fp8(mx.array([[44]]), cache))
    b = np.array(fp8(mx.array([[44]]), restored))
    np.testing.assert_array_equal(a, b)
    assert not inspect_apc_capabilities(cache).exact_prefix
    assert not cache[0].cache_operations_contract.exact_prefix
    exact = native.make_cache()
    native(mx.array([TOKENS[:13]]), exact)
    assert inspect_apc_capabilities(exact).exact_prefix


# ------------------------------------------------------- batching and AST --

def test_fp8_batch_generator_ragged_lanes_equal_single_lane(pair):
    from mlx2.runtime.generate import BatchGenerator

    _, _, _, fp8, _, _ = pair
    prompts = [TOKENS[:10], TOKENS[:17], TOKENS[3:7]]
    batch = BatchGenerator(fp8, max_tokens=6, completion_batch_size=3, prefill_batch_size=2)
    uids = batch.insert(prompts, max_tokens=[6] * 3)
    out = _drain(batch, uids, 6)
    for uid, prompt in zip(uids, prompts):
        assert out[uid][:6] == _greedy(fp8, prompt, 6)


def test_fp8_batch_cache_merge_extract_and_serialize(pair):
    _, _, native, fp8, _, _ = pair
    a, b = fp8.make_cache(), fp8.make_cache()
    fp8(mx.array([TOKENS[:9]]), a)
    fp8(mx.array([TOKENS[:12]]), b)
    merged = [type(x).merge([x, y]) for x, y in zip(a, b)]
    assert not merged[0].cache_operations_contract().exact_prefix
    fp8(mx.array([[40, 41], [42, 43]]), merged)
    for layer in merged:
        layer.trim_ragged([2, 1])
    restored = [H.HY4FlashBatchLayerCache.from_state(m.state, m.meta_state) for m in merged]
    one = [layer.extract(1) for layer in restored]
    ref = fp8.make_cache()
    fp8(mx.array([TOKENS[:12] + [42]]), ref)
    np.testing.assert_allclose(np.array(fp8(mx.array([[44]]), one)),
                               np.array(fp8(mx.array([[44]]), ref)), rtol=1e-5, atol=1e-5)
    mixed = native.make_cache()
    with pytest.raises(ValueError):
        H.HY4FlashBatchLayerCache([a[0].clone(), mixed[0]])
    exact = H.HY4FlashBatchLayerCache([mixed[0].clone()])
    with pytest.raises(ValueError):
        exact.extend(a[0])


def test_fp8_dfly2_history_and_propose_match_round_tripped_native(pair, monkeypatch):
    """The drafter's fp8 reads equal a native drafter over round-tripped rows
    (greedy-equality tests cannot see a drafter-side dequantize error)."""
    _, _, native, fp8, head, fp8_head = pair
    project = H.Attention.project

    def round_tripped(self, x, offset):
        q, q_resid, keys = project(self, x, offset)
        rows = F.unpack_rows(F.pack_rows(keys, self.kv_lora_rank), self.kv_lora_rank, keys.dtype)
        return q, q_resid, rows

    prompt = LONG[:30]
    for model in (native, fp8):
        model.record_layers = frozenset(head.index_layers)
    try:
        cf = fp8.make_cache()
        hf = fp8.hidden(mx.array([prompt]), cf)
        with monkeypatch.context() as m:
            m.setattr(H.Attention, "project", round_tripped)
            cn = native.make_cache()
            hn = native.hidden(mx.array([prompt]), cn)
        np.testing.assert_array_equal(np.array(hf), np.array(hn))
        for layer in range(len(head.layers)):
            rf, vf = fp8_head.history(cf, layer, len(prompt), dtype=mx.float32)
            rn, vn = head.history(cn, layer, len(prompt), dtype=mx.float32)
            assert rf.dtype == mx.float32
            np.testing.assert_array_equal(np.array(rf), np.array(rn))
            np.testing.assert_array_equal(np.array(vf), np.array(vn))
        _, lf = fp8_head.propose(fp8, cf, mx.array([5]), hf[:, -1], anchor=len(prompt), num_draft=3)
        _, ln = head.propose(native, cn, mx.array([5]), hn[:, -1], anchor=len(prompt), num_draft=3)
        np.testing.assert_array_equal(np.array(lf), np.array(ln))
    finally:
        native.record_layers = fp8.record_layers = frozenset()


def test_fp8_dfly2_greedy_equals_fp8_ordinary_greedy(pair):
    _, _, _, fp8, _, head = pair
    prompt = TOKENS[:13]
    tokens, receipt = H.dfly_greedy_generate(fp8, head, prompt, max_tokens=12, num_draft=3)
    assert tokens == _greedy(fp8, prompt, 12)
    assert receipt["latent_kv"] == "fp8" and receipt["approximate_state"]


@pytest.mark.parametrize("tree", [False, True])
def test_fp8_external_engine_dfly2_equals_fp8_greedy(pair, tree):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor
    from mlx2.runtime.drafters.hy4_dfly import Hy4DFlyDraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    _, _, _, fp8, _, head = pair
    fp8.record_layers = frozenset(head.index_layers)
    try:
        draft = Hy4DFlyDraftModel(fp8, head, num_draft=3, revision="test")
        executor = Hy4ChainTreeExecutor.create()
        options = dict(dynamic_singleton_tree=True, dynamic_tree_max_width=1,
                       target_tree_executor_provider=executor,
                       tree_node_budget_by_lanes={1: 7}) if tree else {}
        batch = ExternalDraftBatchGenerator(fp8, draft_model=draft, binding="test", num_draft=3,
                                            prefill_step_size=4, **options)
        prompt = TOKENS[:13]
        uids = batch.insert([prompt], max_tokens=[16], sampling_configs=[{"sampling_temp": 0}])
        out = _drain(batch, uids, 16)
        assert out[uids[0]][:16] == _greedy(fp8, prompt, 16)
        assert draft.stats["block_drafts"] > 0
        if tree:
            assert draft.stats["tree_lattices"] > 0 and executor.receipt()["chain_forwards"] > 0
    finally:
        fp8.record_layers = frozenset()


# ----------------------------------------------------------------- adapter --

def test_adapter_policy_and_receipts_label_fp8_approximate_unqualified(tmp_path, pair):
    from mlx2.adapters import hy_v4_flash as A

    config, W, *_ = pair
    path = _tiny_artifact(tmp_path, config, W)
    adapter = A.HYV4FlashAdapter(str(path), execution_policy={
        "latent_kv": "fp8", "draft_model": "embedded_dfly", "num_draft": 3})
    try:
        assert adapter.model.args.latent_kv == "fp8"
        assert adapter.layout.startswith(A.FP8_CACHE_LAYOUT) and A.FP8_CACHE_LAYOUT != A.CACHE_LAYOUT
        # The Hy4 band default (2026-11-21) adds its tag before the route tag.
        assert adapter.descriptor.cache_layout.startswith(A.FP8_CACHE_LAYOUT + "+swa-band512-v1+sparse-prefill-v1:")
        assert "approximate" in adapter.descriptor.metadata["latent_kv"]
        assert adapter.descriptor.variant.endswith("+fp8-latent-kv")
        before = adapter.diagnostics()["latent_kv"]
        assert before["selected"] and before["approximate"] and before["qualified"] is False
        batch = adapter.create_external_batch(completion_batch_size=1, prefill_step_size=4)
        prompt = TOKENS[:13]
        uids = batch.insert([prompt], max_tokens=[10], sampling_configs=[{"sampling_temp": 0}])
        out = _drain(batch, uids, 10)
        assert out[uids[0]][:10] == _greedy(adapter.model, prompt, 10)
        d = adapter.diagnostics()
        assert d["approximate_state"] and d["ast"]["selected"]
        lk = d["latent_kv"]
        assert lk["format"] == "fp8" and lk["observed_used"] and lk["qualified"] is False
        assert lk["state"] == "approximate; unqualified"
    finally:
        adapter.close()


def test_adapter_default_route_is_native_and_bad_format_refused(tmp_path, pair):
    from mlx2.adapters import hy_v4_flash as A

    config, W, *_ = pair
    path = _tiny_artifact(tmp_path, config, W)
    adapter = A.HYV4FlashAdapter(str(path))
    try:
        assert adapter.model.args.latent_kv == "native"
        assert adapter.layout == A.CACHE_LAYOUT + "+swa-band512-v1+sparse-prefill-v1" == adapter.descriptor.cache_layout
        assert adapter.descriptor.variant == "ordinary"
        d = adapter.diagnostics()
        assert not d["approximate_state"] and not d["latent_kv"]["selected"]
        assert d["latent_kv"]["state"] == "exact"
    finally:
        adapter.close()
    with pytest.raises(ValueError, match="latent_kv"):
        A._parse_policy({"latent_kv": "int4"})
