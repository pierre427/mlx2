"""Hy4-flash opt-in performance repairs (2026-11-22 audit items H1-H6), CPU checks.

CPU only, on the tiny random checkpoint of tests/test_hy_v4_flash_cpu.py and
synthetic tensors at the real Hy4 widths.  Every option is default off; each
section checks that the off path is unchanged, that the on path is within
its stated tolerance (or bitwise, where it is claimed bitwise), and that the
engagement counters move.  The Metal kernels are covered by
tests/test_hy4_nonbitwise_metal.py (not run here).
"""

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_cpu import LONG, TOKENS, _greedy, build, checkpoint, tiny_config  # noqa: E402

from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from mlx2.runtime.models import hy_v4_flash_fp8kv as F8  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    config = tiny_config()
    W = checkpoint(config)
    model, _ = build(config, W)
    return config, W, model


# ------------------------------------------------ H3: fp8 latent unpack --

def _unpack_fp32_route(packed, rank, dtype):
    """``unpack_rows`` as it was before H3: fp32 codes, fp32 product, one cast."""
    lead = packed.shape[:-1]
    width = F8.tile_width(rank)
    tiles = rank // width
    end = rank + 4 * tiles
    scales = mx.view(mx.contiguous(packed[..., rank:end]), mx.float32)
    tail = mx.view(mx.contiguous(packed[..., end:]), mx.bfloat16)
    latent = mx.from_fp8(packed[..., :rank], dtype=mx.float32).reshape(*lead, tiles, width)
    latent = (latent * scales[..., None]).reshape(*lead, rank)
    return mx.concatenate([latent, tail.astype(mx.float32)], axis=-1).astype(dtype)


def _every_code_rows(rank=512, rope=64):
    """Packed rows holding every finite E4M3 code under extreme and ordinary scales."""
    codes = np.array([c for c in range(256) if c not in (0x7F, 0xFF)], np.uint8)
    tiles = rank // 128
    exps = [-108, -60, -9, -1, 0, 3, 17, 60, 100]
    rows = []
    rng = np.random.default_rng(0)
    for e in exps:
        lat = np.resize(rng.permutation(codes), rank).astype(np.uint8)
        sc = np.full(tiles, 2.0 ** e, np.float32).view(np.uint8)
        tail = rng.standard_normal(rope).astype(np.float32)
        tail = np.array(mx.view(mx.array(tail).astype(mx.bfloat16), mx.uint8))
        rows.append(np.concatenate([lat, sc, tail]))
    return mx.array(np.stack(rows)[None])


def test_h3_bf16_unpack_is_bitwise_the_fp32_route():
    packed = _every_code_rows()
    got = F8.unpack_rows(packed, 512, mx.bfloat16)
    want = _unpack_fp32_route(packed, 512, mx.bfloat16)
    assert got.dtype == mx.bfloat16
    assert mx.array_equal(mx.view(got, mx.uint16), mx.view(want, mx.uint16)).item()
    # Packed from real rows too (every tile scale the packer chooses).
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 33, 576)).astype(np.float32)
    x[..., :512] *= np.exp(rng.uniform(-30, 30, size=(2, 33, 1))).astype(np.float32)
    packed = F8.pack_rows(mx.array(x).astype(mx.bfloat16), 512)
    got = F8.unpack_rows(packed, 512, mx.bfloat16)
    assert mx.array_equal(mx.view(got, mx.uint16),
                          mx.view(_unpack_fp32_route(packed, 512, mx.bfloat16), mx.uint16)).item()


def test_h3_other_dtypes_keep_the_fp32_route():
    packed = _every_code_rows()
    for dtype in (mx.float32, mx.float16):
        before = F8.latent_fp8_stats()["bf16_direct_unpacks"]
        got = F8.unpack_rows(packed, 512, dtype)
        assert F8.latent_fp8_stats()["bf16_direct_unpacks"] == before
        assert mx.array_equal(got, _unpack_fp32_route(packed, 512, dtype) if dtype == mx.float32 else
                              mx.clip(_unpack_fp32_route(packed, 512, mx.float32), -65504, 65504)
                              .astype(mx.float16), equal_nan=True).item()


def _kernel_stage_emulation(packed, rank, f16=False):
    """NumPy transcription of the fp8 kernel variant's staging decode
    (``mlx2_e4m3`` x tile scale, bf16 tail bits), so its math is checked on CPU."""
    p = np.array(packed)
    lead = p.shape[:-1]
    tiles = rank // 128
    codes = p[..., :rank].astype(np.uint32)
    e = (codes >> 3) & 0xF
    m = (codes & 7).astype(np.float32)
    normal = (((e + 120) << 23) | ((codes & 7) << 20)).astype(np.uint32).view(np.float32)
    v = np.where(e == 0, m * np.float32(2.0 ** -9), normal).astype(np.float32)
    v = np.where(codes & 0x80, -v, v)
    scale = p[..., rank:rank + 4 * tiles].copy().view(np.float32).reshape(*lead, tiles)
    lat = (v.reshape(*lead, tiles, 128) * scale[..., None]).reshape(*lead, rank)
    tail_bits = p[..., rank + 4 * tiles:].copy().view(np.uint16).astype(np.uint32) << 16
    rows = np.concatenate([lat, tail_bits.view(np.float32)], axis=-1)
    if f16:
        rows = np.clip(rows, -65504, 65504)
    return rows


def test_h3_kernel_staging_math_is_bitwise_unpack_rows():
    packed = _every_code_rows()
    emu = mx.array(_kernel_stage_emulation(packed, 512))
    assert mx.array_equal(emu.astype(mx.bfloat16), F8.unpack_rows(packed, 512, mx.bfloat16)).item()
    emu16 = mx.array(_kernel_stage_emulation(packed, 512, f16=True)).astype(mx.float16)
    assert mx.array_equal(emu16, F8.unpack_rows(packed, 512, mx.float16)).item()


def test_h3_fp8_variant_geometry_gate_and_source():
    from mlx2.runtime.models import hy_v4_flash_sparse_prefill as S

    assert S.fp8_row_geometry(512, 576) == (656, 128)
    src = S._fp8_source()
    assert "kvb8" in src and "ktw" in src and "MLX2_FP8_CLAMP" in src
    # CPU: never engages (a counted reason, not an error).
    assert S.fp8_unsupported_reason(32, 576, 512, mx.bfloat16, row_bytes=656) == "metal_unavailable"


def _fp8_model(config, W, dtype=None):
    model, _ = build(config, W)
    model.args.latent_kv = "fp8"
    return model


def test_h3_model_routes_packed_rows_to_the_fp8_kernel(tiny, monkeypatch):
    """With the kernel gates forced open on CPU and the kernel emulated by
    unpack + the per-row gather law, the fp8 forward hands the packed rows to
    the variant (no whole-history unpack) and equals the gather law."""
    config, W, _ = tiny
    monkeypatch.setattr(H, "GATHER_MAX_QUERY", 4)
    monkeypatch.setattr(H, "SPARSE_PREFILL_MIN_KEYS", 0)
    seen = []

    def emulated(q, packed, recalled, sinks, scale, rank, *, start, first):
        assert packed.dtype == mx.uint8
        seen.append(int(packed.shape[1]))
        rank_rows = lambda g: F8.unpack_rows(g, rank, q.dtype)  # noqa: E731
        return H.gathered_indexed_attend(q, packed, recalled, sinks, scale, rank, rows=rank_rows)

    ref = _fp8_model(config, W)
    tokens = mx.array([LONG[:40]])
    want = ref(tokens)  # gather law for <= 4 rows, dense for longer chunks

    monkeypatch.setattr(H, "GATHER_MAX_QUERY", 64)
    gathered = _fp8_model(config, W)
    want_gather = gathered(tokens)
    model = _fp8_model(config, W)
    model.set_sparse_prefill_attention(True)
    monkeypatch.setattr(H, "GATHER_MAX_QUERY", 4)
    monkeypatch.setattr(H, "sparse_unsupported_reason", lambda *a, **k: None)
    monkeypatch.setattr(H, "_sparse_fp8_reason", lambda *a, **k: None)
    monkeypatch.setattr(H, "_sparse_prefill_attend_fp8", emulated)
    whole = []
    original = H.HY4FlashFP8LayerCache.latent_rows

    def spy(self, rows, dtype):
        if self.kind == H.INDEXED and rows.ndim == 3:
            whole.append(int(rows.shape[1]))  # a [B, T, row] whole-history unpack
        return original(self, rows, dtype)

    monkeypatch.setattr(H.HY4FlashFP8LayerCache, "latent_rows", spy)
    got = model(tokens)
    indexed = [l.self_attn for l in model.model.layers if l.self_attn.kind == H.INDEXED]
    assert seen and all(a.sparse_prefill_fp8_forwards == 1 for a in indexed)
    assert all(a.sparse_prefill_fp8_declines == {} for a in indexed)
    assert whole == []  # only recalled rows were dequantized
    np.testing.assert_allclose(np.array(got), np.array(want_gather), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-4)


def test_h3_fp8_variant_decline_keeps_native_kernel(tiny, monkeypatch):
    config, W, _ = tiny
    monkeypatch.setattr(H, "GATHER_MAX_QUERY", 4)
    monkeypatch.setattr(H, "SPARSE_PREFILL_MIN_KEYS", 0)
    native_calls = []

    def native(q, kv, recalled, sinks, scale, rank, *, start, first):
        assert kv.dtype != mx.uint8
        native_calls.append(1)
        return H.gathered_indexed_attend(q, kv, recalled, sinks, scale, rank)

    model = _fp8_model(config, W)
    model.set_sparse_prefill_attention(True)
    monkeypatch.setattr(H, "sparse_unsupported_reason", lambda *a, **k: None)
    monkeypatch.setattr(H, "_sparse_fp8_reason", lambda *a, **k: "fp8_row_alignment")
    monkeypatch.setattr(H, "sparse_prefill_attend", native)
    model(mx.array([LONG[:40]]))
    indexed = [l.self_attn for l in model.model.layers if l.self_attn.kind == H.INDEXED]
    assert native_calls and all(a.sparse_prefill_fp8_declines == {"fp8_row_alignment": 1} for a in indexed)
    assert all(a.sparse_prefill_declines == {} for a in indexed)


def test_h3_native_cache_never_takes_the_fp8_variant(tiny, monkeypatch):
    config, W, ref = tiny
    model, _ = build(config, W)
    model.set_sparse_prefill_attention(True)
    monkeypatch.setattr(H, "_sparse_fp8_reason", lambda *a, **k: pytest.fail("fp8 gate on native rows"))
    tokens = mx.array([LONG[:40]])
    assert mx.array_equal(model(tokens), ref(tokens)).item()


# ------------------------------------------ H1 / H6: indexer fused scores --

def _capture_selections(monkeypatch):
    seen = []
    original = H.CompressedIndexer.select

    def spy(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        seen.append(np.array(out))
        return out

    monkeypatch.setattr(H.CompressedIndexer, "select", spy)
    return seen


def _sets(selections):
    return [np.sort(s, axis=-1) for s in selections]


def test_h1_policy_selects_v1_and_simd_and_namespaces():
    from mlx2.adapters import hy_v4_flash as A

    for value in (True, "simd"):
        model, _, _ = A._parse_policy({"indexer_fused_scores": value})
        assert model == {"indexer_fused_scores": value}
    for bad in ("v2", 1, "true"):
        with pytest.raises(ValueError, match="indexer_fused_scores"):
            A._parse_policy({"indexer_fused_scores": bad})
    assert A.numerics_layout("L", fusions=A.fusion_policy({})) == "L"
    assert A.numerics_layout("L", fusions=A.fusion_policy({"indexer_fused_scores": True})) == "L+indexer-fused-v1"
    assert (A.numerics_layout("L", fusions=A.fusion_policy({"indexer_fused_scores": "simd"}))
            == "L+indexer-fused-v1-simd")


@pytest.mark.parametrize("variant", [True, "simd"])
def test_h1_fused_branch_picks_the_default_topk_on_cpu_fallback(tiny, monkeypatch, variant):
    """The fused branch (negated logits -> argpartition -> causal mask -> own
    group), driven on CPU by the reference logits as its kernel, picks the
    default route's top-k sets on every indexed forward (prefill and decode)."""
    from mlx2.runtime.models import hy_v4_flash_fusions as F

    config, W, ref = tiny
    seen = _capture_selections(monkeypatch)
    out_ref = _greedy(ref, LONG[:40], 6)
    want = _sets(seen)
    seen.clear()

    model, _ = build(config, W)
    F.install(model, indexer_scores=variant)
    calls = {"v1": 0, "simd": 0}

    def kernel(name):
        def run(q, gk, w, own, negate=False):
            calls[name] += 1
            out = F.indexer_logits_reference(q, gk.astype(mx.float32), w, own)
            return -out if negate else out
        return run

    monkeypatch.setattr(F, "on_gpu", lambda: True)
    monkeypatch.setattr(F, "indexer_logits_metal", kernel("v1"))
    monkeypatch.setattr(F, "indexer_logits_simd", kernel("simd"))
    # The tiny indexer (2 heads x 8 dims) fits the simd geometry.
    assert F.indexer_simd_unsupported_reason(2, 8) is None
    assert _greedy(model, LONG[:40], 6) == out_ref
    got = _sets(seen)
    assert len(got) == len(want) and all(np.array_equal(a, b) for a, b in zip(got, want))
    used = "simd" if variant == "simd" else "v1"
    assert calls[used] > 0 and calls["v1" if used == "simd" else "simd"] == 0
    st = F.status(model)["indexer_fused_scores"]
    assert st["selected"] and st["observed_used"] and st["variant"] == used
    assert st["simd_calls"] == (calls["simd"])


def test_h1_cpu_stream_runs_the_reference_bitwise(tiny):
    from mlx2.runtime.models import hy_v4_flash_fusions as F

    config, W, ref = tiny
    for variant in (True, "simd"):
        model, _ = build(config, W)
        F.install(model, indexer_scores=variant)
        tokens = mx.array([LONG[:40]])
        assert mx.array_equal(model(tokens), ref(tokens)).item()
        assert F.status(model)["indexer_fused_scores"]["fused_score_calls"] == 0  # CPU: reference ops


def test_h6_simd_geometry_gate():
    from mlx2.runtime.models import hy_v4_flash_fusions as F

    assert F.indexer_simd_rows(16) == 8          # Hy4: 8 query rows x 16 heads per key read
    assert F.indexer_simd_rows(2) == 32 and F.indexer_simd_rows(32) == 8
    assert F.indexer_simd_unsupported_reason(16, 128) is None
    assert F.indexer_simd_unsupported_reason(16, 132) == "head_dim_132"
    assert F.indexer_simd_unsupported_reason(64, 128).startswith("registers_")
    for rb, h in ((F.indexer_simd_rows(16), 16), (F.indexer_simd_rows(2), 2)):
        assert (rb * h) % 64 == 0


def test_h6_simd_decline_falls_back_to_v1(tiny, monkeypatch):
    from mlx2.runtime.models import hy_v4_flash_fusions as F

    config, W, _ = tiny
    model, _ = build(config, W)
    F.install(model, indexer_scores="simd")
    monkeypatch.setattr(F, "on_gpu", lambda: True)
    monkeypatch.setattr(F, "indexer_simd_unsupported_reason", lambda h, d: "forced")
    monkeypatch.setattr(F, "indexer_logits_simd", lambda *a, **k: pytest.fail("declined variant ran"))
    monkeypatch.setattr(F, "indexer_logits_metal",
                        lambda q, gk, w, own, negate=False: -F.indexer_logits_reference(q, gk, w, own))
    model(mx.array([LONG[:40]]))
    st = F.status(model)["indexer_fused_scores"]
    assert st["simd_calls"] == 0 and st["simd_declines"].get("forced", 0) > 0 and st["fused_score_calls"] > 0


# ---------------------------------------------------- H4: union prefill --

def _union_case(seed, *, H=4, L=70, T=300, K=24, start=230, D=40, rank=32, empty=0.2):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((1, H, L, D)).astype(np.float32))
    kv = mx.array(rng.standard_normal((1, T, D)).astype(np.float32))
    rec = np.full((1, L, K), -1, np.int32)
    for l in range(L):
        pos = start + l
        n = min(int(K * (1 - empty)), pos + 1)
        # Neighbouring rows share most picks (as the indexer's do).
        pool = rng.choice(pos + 1, size=min(pos + 1, 2 * K), replace=False)
        rec[0, l, :n] = rng.choice(pool, size=n, replace=False)
        if pos + 1 < T:
            rec[0, l, -1] = rng.integers(pos + 1, T)  # a future id: masked
    rec[0, 3, :] = rec[0, 3, 0]  # a repeated id: counted once, as in the dense mask
    sinks = mx.array(rng.standard_normal(H).astype(np.float32))
    return q, kv, mx.array(rec), sinks, start, rank


@pytest.mark.parametrize("rows", [1, 7, 64, 1000])
def test_h4_union_attend_matches_dense_reference(rows):
    from mlx2.runtime.models import hy_v4_flash_union_prefill as U

    q, kv, rec, sinks, start, rank = _union_case(rows)
    L, T = q.shape[2], kv.shape[1]
    qpos, kpos = mx.arange(start, start + L), mx.arange(0, T)
    dense = H.dense_indexed_attend(q, kv, rec, sinks, 0.2, rank, qpos, kpos)
    before = U.union_stats()
    got = U.union_indexed_attend(q, kv, rec, sinks, 0.2, rank, qpos, rows, attend=H.mla_attend)
    after = U.union_stats()
    np.testing.assert_allclose(np.array(got), np.array(dense), rtol=2e-5, atol=2e-6)
    assert after["union_forwards"] == before["union_forwards"] + 1
    assert after["union_subchunks"] - before["union_subchunks"] == -(-L // min(rows, L))
    # Reads: the union of a sub-chunk is never more than its rows' recalled slots.
    assert (after["union_rows"] - before["union_rows"]) <= (after["recalled_rows"] - before["recalled_rows"])


def test_h4_union_bounded_blocks_are_bitwise_unbounded():
    from mlx2.runtime.models import hy_v4_flash_union_prefill as U

    q, kv, rec, sinks, start, rank = _union_case(9)
    qpos = mx.arange(start, start + q.shape[2])
    a = U.union_indexed_attend(q, kv, rec, sinks, 0.2, rank, qpos, 8, attend=H.mla_attend, bounded=False)
    b = U.union_indexed_attend(q, kv, rec, sinks, 0.2, rank, qpos, 8, attend=H.mla_attend, budget=1)
    assert mx.array_equal(a, b).item()


def _union_model(config, W, rows, latent_kv="native"):
    model, _ = build(config, W)
    model.args.latent_kv = latent_kv
    model.set_sparse_prefill_union_rows(rows)
    return model


@pytest.mark.parametrize("latent_kv", ["native", "fp8"])
def test_h4_model_union_matches_dense_and_counts(tiny, monkeypatch, latent_kv):
    config, W, _ = tiny
    monkeypatch.setattr(H, "GATHER_MAX_QUERY", 4)
    monkeypatch.setattr(H, "SPARSE_PREFILL_MIN_KEYS", 0)
    ref = _union_model(config, W, 0, latent_kv)
    tokens = mx.array([LONG[:40]])
    want = ref(tokens)
    for rows in (3, 16):
        model = _union_model(config, W, rows, latent_kv)
        got = model(tokens)
        indexed = [l.self_attn for l in model.model.layers if l.self_attn.kind == H.INDEXED]
        assert all(a.sparse_prefill_union_forwards == 1 and a.sparse_prefill_union_declines == {}
                   for a in indexed)
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-4)
        assert _greedy(model, LONG[:40], 6) == _greedy(ref, LONG[:40], 6)


def test_h4_off_is_unchanged_and_short_history_declines(tiny, monkeypatch):
    config, W, ref = tiny
    tokens = mx.array([LONG[:40]])
    off = _union_model(config, W, 0)
    assert mx.array_equal(off(tokens), ref(tokens)).item()
    monkeypatch.setattr(H, "GATHER_MAX_QUERY", 4)
    on = _union_model(config, W, 8)  # SPARSE_PREFILL_MIN_KEYS (4096) is above the tiny history
    dense = _union_model(config, W, 0)
    assert mx.array_equal(on(tokens), dense(tokens)).item()
    indexed = [l.self_attn for l in on.model.layers if l.self_attn.kind == H.INDEXED]
    assert all(a.sparse_prefill_union_forwards == 0 and a.sparse_prefill_union_declines == {"short_context": 1}
               for a in indexed)


def test_h4_policy_key_namespace_receipt_and_load(tmp_path, tiny):
    from test_hy_v4_flash_cpu import _write_artifact

    from mlx2.adapters import hy_v4_flash as A

    assert "sparse_prefill_union_rows" in A.MODEL_KEYS
    assert "sparse_prefill_union_rows" in A.HYV4FlashAdapter.route_neutral_policy_keys
    assert "sparse_prefill_union_rows" in A.DET_TILE_NUMERICS_KEYS
    model, _, _ = A._parse_policy({"sparse_prefill_union_rows": 64})
    assert model == {"sparse_prefill_union_rows": 64}
    for bad in (-1, 1.5, "64", True):
        with pytest.raises(ValueError, match="sparse_prefill_union_rows"):
            A._parse_policy({"sparse_prefill_union_rows": bad})
    assert A.numerics_layout("L") == "L"
    assert A.numerics_layout("L", sparse_prefill_union_rows=64) == "L+union-prefill64-v1"
    stub = A.HYV4FlashAdapter.__new__(A.HYV4FlashAdapter)
    union = stub._sparse_prefill_receipt()["union"]
    assert union["selected"] is False and union["observed_used"] is False and union["qualified"] is False


def test_h4_adapter_installs_union_rows(tmp_path, tiny):
    from test_hy_v4_flash_cpu import _tiny_artifact

    from mlx2.adapters import hy_v4_flash as A

    config, W, _ = tiny
    adapter = A.HYV4FlashAdapter(str(_tiny_artifact(tmp_path, config, W)),
                                 execution_policy={"sparse_prefill_union_rows": 32, "vision": False})
    try:
        indexed = [l.self_attn for l in adapter.model.model.layers if l.self_attn.kind == H.INDEXED]
        assert all(a.sparse_prefill_union_rows == 32 for a in indexed)
        assert "+union-prefill32-v1" in adapter.layout
        union = adapter.diagnostics()["sparse_prefill_attention"]["union"]
        assert union["selected"] and union["rows"] == 32
    finally:
        adapter.close()


# ------------------------------------------- H5: BF16-resident fp32 planes --

def _bf16_valued(W):
    """The tiny checkpoint rounded to bf16 values (kept fp32), as a bf16 checkpoint loads."""
    return {k: (np.array(mx.array(v).astype(mx.bfloat16).astype(mx.float32))
                if v.dtype == np.float32 else v) for k, v in W.items()}


@pytest.fixture(scope="module")
def tiny_bf16_valued():
    config = tiny_config()
    W = _bf16_valued(checkpoint(config))
    model, _ = build(config, W)
    return config, W, model


def test_h5_parts_parse_and_agree_with_the_adapter():
    from mlx2.adapters import hy_v4_flash as A
    from mlx2.runtime.models import hy_v4_flash_resident as R

    assert A.RESIDENT_BF16_PARTS == R.PARTS
    for value in (False, True, ["head"], ["q2_scales", "router"]):
        assert A.resident_bf16_parts(value) == R.parse(value)
    for bad in ([], ["lm_head"], "router", 1):
        with pytest.raises(ValueError, match="resident_bf16"):
            A._parse_policy({"resident_bf16": bad})
    assert A._parse_policy({"resident_bf16": ["router"]})[0] == {"resident_bf16": ["router"]}
    assert A.numerics_layout("L", resident_bf16=False) == "L"
    assert A.numerics_layout("L", resident_bf16=["head", "router"]) == "L+resident-bf16-router-head-v1"
    assert A.numerics_layout("L", resident_bf16=True) == \
        "L+resident-bf16-router-head-indexerkeys-q2scales-v1"
    assert {"resident_bf16"} <= A.MODEL_KEYS & A.HYV4FlashAdapter.route_neutral_policy_keys
    assert "resident_bf16" in A.DET_TILE_NUMERICS_KEYS


def test_h5_router_and_head_narrow_exactly_and_forward_is_bitwise(tiny_bf16_valued):
    from mlx2.runtime.models import hy_v4_flash_resident as R

    config, W, ref = tiny_bf16_valued
    model, _ = build(config, W)
    record = R.install(model, ("router", "head"))
    t = config["text_config"]
    moe_layers = sum(1 for k in t["mlp_layer_types"] if k == "sparse")
    E, D, V = t["n_routed_experts"], t["hidden_size"], t["vocab_size"]
    assert record["saved_bytes"] == {"router": moe_layers * E * D * 2, "head": V * D * 2}
    assert record["kept_fp32"] == {"router": [], "head": []}
    assert all(l.mlp.gate.weight.dtype == mx.bfloat16 for l in model.model.layers if isinstance(l.mlp, H.MoE))
    assert model.lm_head.weight.dtype == mx.bfloat16
    before = dict(R.STATS)
    tokens = mx.array([LONG[:40]])
    assert mx.array_equal(model(tokens), ref(tokens)).item()        # CPU: widened at use, bitwise
    assert _greedy(model, TOKENS[:12], 6) == _greedy(ref, TOKENS[:12], 6)
    assert R.STATS["widen_calls"] > before["widen_calls"] and R.STATS["gemv_calls"] == before["gemv_calls"]
    st = R.status(model)
    assert st["selected"] and st["observed_used"] and st["saved_bytes_total"] == sum(record["saved_bytes"].values())


def test_h5_inexact_planes_stay_fp32(tiny):
    from mlx2.runtime.models import hy_v4_flash_resident as R

    config, W, ref = tiny  # fp32 random values: not bf16-exact
    model, _ = build(config, W)
    record = R.install(model, ("router", "head"))
    assert record["saved_bytes_total"] == 0
    assert record["kept_fp32"]["head"] == ["lm_head.weight"] and len(record["kept_fp32"]["router"]) == 5
    tokens = mx.array([LONG[:40]])
    assert mx.array_equal(model(tokens), ref(tokens)).item()


def test_h5_default_router_never_takes_the_gemv(tiny_bf16_valued, monkeypatch):
    """Without the option a bf16 router weight (e.g. ``set_dtype``) keeps the reference expression."""
    from mlx2.runtime.models import hy_v4_flash_resident as R

    monkeypatch.setattr(R, "gemv_f32", lambda *a: pytest.fail("GEMV without the opt-in"))
    x = mx.array(np.random.default_rng(0).standard_normal((1, 1, 32)).astype(np.float32))
    w = mx.array(np.random.default_rng(1).standard_normal((8, 32)).astype(np.float32)).astype(mx.bfloat16)
    assert mx.array_equal(R.f32_matmul(x, w), x @ w.astype(mx.float32).T).item()


def test_h5_indexer_keys_bf16_are_exact_and_select_bitwise(tiny_bf16_valued):
    """bf16 activations (the served dtype; the tiny CPU model cannot run bf16
    expert gathers, so the indexer runs alone): pooled keys cached bf16 hold
    the fp32 plane's values, and the reference selection is bitwise."""
    from mlx2.runtime.models import hy_v4_flash_resident as R

    config, W, _ = tiny_bf16_valued
    model, _ = build(config, W)
    layer = next(l for l in model.model.layers if l.self_attn.kind == H.INDEXED)
    ix = layer.self_attn.compressed_indexer
    rng = np.random.default_rng(6)
    D, ratio = config["text_config"]["hidden_size"], ix.ratio
    x = mx.array(rng.standard_normal((1, 40, D)).astype(np.float32)).astype(mx.bfloat16)
    q_resid = mx.array(rng.standard_normal((1, 40, 16)).astype(np.float32)).astype(mx.bfloat16)
    token_keys = ix.token_key_norm(ix.token_key_projection(x)).astype(mx.bfloat16)
    want = ix.pool(token_keys, 0)
    R.install(model, ("indexer_keys",))
    got = ix.pool(token_keys, 0)
    assert want.dtype == mx.float32 and got.dtype == mx.bfloat16 and got.nbytes * 2 == want.nbytes
    assert mx.array_equal(got.astype(mx.float32), want).item()
    assert ix.lowp_key_groups == 40 // ratio
    a = ix.select(x, q_resid, 40, want)
    b = ix.select(x, q_resid, 40, got)
    assert mx.array_equal(a, b).item()
    assert R.status(model)["indexer_key_groups_bf16"] > 0


def test_h5_fp32_activations_keep_fp32_indexer_keys(tiny):
    from mlx2.runtime.models import hy_v4_flash_resident as R

    config, W, ref = tiny
    model, _ = build(config, W)
    R.install(model, ("indexer_keys",))
    cache = model.make_cache()
    model(mx.array([LONG[:40]]), cache)
    assert all(c.gk_data.dtype == mx.float32 for c in cache if c.kind == H.INDEXED)


def test_h5_q2_scales_f16_resident_gather_is_bitwise():
    from mlx2.runtime.models import hy_v4_flash_resident as R

    rng = np.random.default_rng(4)
    bank = H.Q2ZeroSwitchLinear(128, 16, 4, group_size=64)
    codes = mx.array(rng.integers(0, 2 ** 32, size=(4, 16, 8), dtype=np.uint64).astype(np.uint32))
    scales = mx.array((rng.standard_normal((4, 16, 2)) * 0.02).astype(np.float16)).astype(mx.float32)
    bank.weight, bank.scales, bank.biases = codes, scales, -scales
    x = mx.array(rng.standard_normal((3, 1, 128)).astype(np.float32)).astype(mx.bfloat16)
    idx = mx.array([[0], [2], [3]], dtype=mx.uint32)
    want = bank(x, idx)

    class Holder:  # the install walk: layers[i].mlp.experts.<proj>
        pass

    moe = H.MoE.__new__(H.MoE)
    import mlx.nn as nn

    nn.Module.__init__(moe)
    experts = nn.Module()
    experts.gate_proj = bank
    moe.experts = experts
    layer = nn.Module()
    layer.mlp = moe
    layer.self_attn = nn.Module()
    model = Holder()
    model.model = Holder()
    model.model.layers = [layer]
    model.args = Holder()
    model.args.enable_lm_head_fp32 = False
    record = R.install(model, ("q2_scales",))
    assert bank.scales.dtype == mx.float16 and bank.biases.dtype == mx.float16
    assert record["saved_bytes"]["q2_scales"] == 2 * 4 * 16 * 2 * 2
    assert mx.array_equal(bank(x, idx), want).item()


def test_h5_load_model_installs_and_receipts(tmp_path, tiny_bf16_valued):
    from test_hy_v4_flash_cpu import _tiny_artifact

    from mlx2.adapters import hy_v4_flash as A

    config, W, ref = tiny_bf16_valued
    adapter = A.HYV4FlashAdapter(str(_tiny_artifact(tmp_path, config, W)),
                                 execution_policy={"resident_bf16": ["router", "head"], "vision": False})
    try:
        assert "+resident-bf16-router-head-v1" in adapter.layout
        model = adapter.model
        assert model.lm_head.weight.dtype == mx.bfloat16
        tokens = mx.array([LONG[:30]])
        assert mx.array_equal(model(tokens), ref(tokens)).item()
        receipt = adapter.diagnostics()["resident_bf16"]
        assert receipt["selected"] and receipt["observed_used"] and receipt["qualified"] is False
        assert receipt["saved_bytes_total"] > 0
    finally:
        adapter.close()
    (tmp_path / "plain").mkdir()
    plain = A.HYV4FlashAdapter(str(_tiny_artifact(tmp_path / "plain", config, W)),
                               execution_policy={"vision": False})
    try:
        assert plain.model.lm_head.weight.dtype == mx.float32
        assert plain.diagnostics()["resident_bf16"]["selected"] is False
    finally:
        plain.close()


# ---------------------------------------------------------- A/B driver --

def test_ab_driver_runs_and_compares_on_cpu(tmp_path, tiny_bf16_valued):
    import importlib.util
    from pathlib import Path

    from test_hy_v4_flash_cpu import _tiny_artifact

    spec = importlib.util.spec_from_file_location(
        "ab_hy4_nonbitwise", Path(__file__).resolve().parents[1] / "scripts" / "ab_hy4_nonbitwise.py")
    ab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ab)
    config, W, _ = tiny_bf16_valued
    model = str(_tiny_artifact(tmp_path, config, W))
    common = ["--model", model, "--cpu", "--contexts", "40", "--chunk", "16", "--decode", "3", "--select-rows", "2"]
    outs = {}
    for arm in ("reference", "resident-router", "union16"):
        out = tmp_path / f"{arm}.json"
        ab.main(["run", "--arm", arm, "--out", str(out), *common])
        outs[arm] = out
    with pytest.raises(SystemExit):  # evidence is never overwritten
        ab.main(["run", "--arm", "reference", "--out", str(outs["reference"]), *common])
    report = ab.compare([str(outs["reference"]), str(outs["resident-router"]), str(outs["union16"])])
    res = report["arms"]["resident-router"]
    cell = res["contexts"]["40"]
    assert cell["bitwise"] and cell["max_abs_logit_gap_before_divergence"] == 0.0
    assert cell["indexer_topk_jaccard_mean"] == 1.0 and res["refused"] == []
    # The tiny history is below the union key floor: selected, never engaged, reported.
    assert any("sparse_prefill_union_rows" in r for r in report["arms"]["union16"]["refused"])
