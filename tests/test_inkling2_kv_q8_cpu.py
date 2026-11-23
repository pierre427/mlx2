"""``inkling2_kv_q8``: int8 K/V for Inkling 2's global visits, CPU only.

On the CPU every global visit takes the fallback (dequantize the fetched
K/V, then the composed path), which is also the oracle of the Metal kernels
(tests/test_inkling2_kv_q8_metal.py). These tests pin the storage format,
the closeness of the Q8 model to the BF16 one, the cache semantics
(trim rollback, batching, APCv2 freeze/restore), the fail-closed gates and
the numerics/layout identity.
"""

import mlx.core as mx
import numpy as np
import pytest
from test_inkling2_cpu import TINY, _run, _tiny_model

from mlx2.runtime.models.inkling2 import kv_q8


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


# ----- format -----


def _numpy_pack(x):
    """Independent reference: symmetric int8 per group, FP16 scale."""
    *lead, dim = x.shape
    group = min(64, dim)
    xg = x.astype(np.float32).reshape(*lead, dim // group, group)
    scale = (np.abs(xg).max(-1, keepdims=True) / np.float32(127)).astype(np.float16)
    sf = scale.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        codes = np.where(sf > 0, np.clip(np.rint(xg / np.where(sf > 0, sf, 1)), -127, 127), 0)
    return codes.astype(np.int8).reshape(*lead, dim), scale.reshape(*lead, -1)


@pytest.mark.parametrize("dim", [32, 64, 128])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_pack_round_trip_within_int8_error(dim, dtype):
    mx.random.seed(dim)
    x = (mx.random.normal((2, 3, 17, dim)) * 1.7).astype(dtype)
    x = mx.concatenate([x, mx.zeros((2, 3, 1, dim), dtype)], axis=2)  # an all-zero row
    packed = kv_q8.pack(x)
    assert packed.dtype == mx.int8 and packed.shape == (2, 3, 18, kv_q8.packed_width(dim))
    group = kv_q8.group_size(dim)
    codes, scales = kv_q8.unpack(packed, dim)
    assert scales.dtype == mx.float16 and scales.shape[-1] == dim // group
    assert mx.abs(codes.astype(mx.int32)).max().item() <= 127
    back = kv_q8.dequantize(packed, dim, mx.float32)
    xf = x.astype(mx.float32)
    err = mx.abs(back - xf).reshape(2, 3, 18, dim // group, group).max(-1)
    # Half a step, plus the FP16 rounding of the scale over the code range.
    step = scales.astype(mx.float32)
    assert (err <= step * (0.5 + 127 * 2 ** -11) + 1e-7).all().item()
    assert mx.array_equal(back[:, :, -1], mx.zeros_like(back[:, :, -1]))
    # Codes and scales equal an independent NumPy quantizer.
    ref_codes, ref_scales = _numpy_pack(np.array(xf))
    assert np.array_equal(np.array(codes), ref_codes)
    assert np.array_equal(np.array(scales), ref_scales)
    # The stored value is a fixed point of the format.
    assert mx.array_equal(kv_q8.pack(back), packed)


def _pack_v1(x):
    """The v1 row (codes, then FP16 scale bytes, no padding; width 132)."""
    codes, scales = _numpy_pack(np.array(x.astype(mx.float32)))
    return np.concatenate([codes, scales.view(np.int8)], axis=-1)


def _dequantize_v1(raw, dim):
    group = min(64, dim)
    codes = raw[..., :dim].astype(np.float32)
    scales = np.ascontiguousarray(raw[..., dim:]).view(np.float16).astype(np.float32)
    return (codes.reshape(*codes.shape[:-1], dim // group, group) * scales[..., None]).reshape(codes.shape)


@pytest.mark.parametrize("dim", [32, 64, 128])
def test_v2_rows_are_aligned_and_dequantize_like_v1_bitwise(dim):
    """v2 pads each row to 16 bytes (codes at 0, scales at D, zeros after);
    the dequantized values equal the v1 format's bitwise."""
    mx.random.seed(dim + 1)
    x = (mx.random.normal((2, 3, 9, dim)) * 1.3).astype(mx.bfloat16)
    packed = kv_q8.pack(x)
    width = kv_q8.packed_width(dim)
    assert width % 16 == 0 and packed.shape[-1] == width
    n = dim // kv_q8.group_size(dim)
    raw = np.array(packed)
    assert not raw[..., dim + 2 * n:].any()  # zero padding
    v1 = _pack_v1(x)
    assert v1.shape[-1] == dim + 2 * n
    assert np.array_equal(raw[..., :dim + 2 * n], v1)
    got = np.array(kv_q8.dequantize(packed, dim, mx.float32))
    assert np.array_equal(got, _dequantize_v1(v1, dim))


def test_format_geometry_and_refusals():
    assert kv_q8.packed_width(128) == 144 and kv_q8.packed_width(32) == 48
    assert kv_q8.packed_width(64) == 80
    assert kv_q8.geometry_reason(128) is None and kv_q8.geometry_reason(32) is None
    assert kv_q8.geometry_reason(96) == "group" and kv_q8.geometry_reason(33) == "head_dim"
    with pytest.raises(ValueError, match="int8 of width"):
        kv_q8.unpack(mx.zeros((1, 1, 1, 128), mx.bfloat16), 128)
    with pytest.raises(ValueError, match="int8 of width"):
        kv_q8.unpack(mx.zeros((1, 1, 1, 130), mx.int8), 128)


def test_decline_reasons():
    q = mx.zeros((1, 24, 8, 128), mx.bfloat16)
    k = mx.zeros((1, 8, 64, 144), mx.int8)
    geo = dict(n_heads=24, n_kv=8, head_dim=128)
    assert kv_q8.decline_reason(q, k, k, **geo, device=mx.cpu) == "cpu"
    assert kv_q8.decline_reason(q, k, k, **geo, device=mx.gpu) is None
    assert kv_q8.decline_reason(q[:, :, :1], k, k, **geo, device=mx.gpu) is None
    assert kv_q8.decline_reason(mx.zeros((1, 24, 32, 128), mx.bfloat16), k, k, **geo,
                                device=mx.gpu) == "rows"
    assert kv_q8.decline_reason(q.astype(mx.float16), k, k, **geo, device=mx.gpu) == "dtype"
    bf = mx.zeros((1, 8, 64, 128), mx.bfloat16)
    assert kv_q8.decline_reason(q, bf, bf, **geo, device=mx.gpu) == "cache_format"
    assert kv_q8.decline_reason(q, k[..., :130], k, **geo, device=mx.gpu) == "cache_format"
    assert kv_q8.decline_reason(q, k, k, n_heads=20, n_kv=8, head_dim=128,
                                device=mx.gpu) == "geometry"
    wide = mx.zeros((1, 8, 64, kv_q8.packed_width(256)), mx.int8)
    assert kv_q8.decline_reason(mx.zeros((1, 24, 8, 256), mx.bfloat16), wide, wide,
                                n_heads=24, n_kv=8, head_dim=256, device=mx.gpu) == "geometry"


# ----- model -----


PLAN = [(0, 20)] + [(i, i + 1) for i in range(20, 40)]  # ring (window 8) wrapped


def _logp(x):
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


@pytest.mark.parametrize("rows", [1, 2])
def test_q8_model_stays_close_to_bf16(rows):
    """Prefill (20 rows) plus 20 decode steps with the ring wrapped and log
    scaling engaged: a bounded logit gap and high greedy agreement.

    The bound is on the median position and on KL: on this random tiny MoE a
    position occasionally jumps (0.25 logits at seed 5 position 0, from the V
    codes; BF16 rounding of the same K/V moves it 0.0014) when the
    perturbation flips a routed expert downstream."""
    model = _tiny_model()
    mx.random.seed(5)
    tokens = mx.random.randint(0, 500, (rows, 40))
    base = _run(model, tokens, PLAN)
    model.configure_kernels(kv_q8=True)
    got = _run(model, tokens, PLAN)
    receipt = model.kernel_receipt()["kv_q8"]
    model.configure_kernels()
    assert receipt["selected"] and receipt["fidelity"] == "approximate"
    assert receipt["qualified"] is False and receipt["global_layers"] == 4
    assert receipt["calls"] == {"blockwise:cpu": 8 * len(PLAN)}
    assert receipt["kv_q8_full_prefix_dequant"] == 0
    assert receipt["kv_q8_block_dequant_calls"] == 8 * len(PLAN)  # one key block each
    gaps = np.array(mx.abs(got - base).max(-1).reshape(-1))
    kl = np.array((mx.exp(_logp(base)) * (_logp(base) - _logp(got))).sum(-1).reshape(-1))
    agree = (mx.argmax(got, -1) == mx.argmax(base, -1)).mean().item()
    assert gaps.max() > 0  # it is not the BF16 model
    assert np.median(gaps) < 0.01, np.median(gaps)
    assert gaps.max() < 1.0, gaps.max()
    assert kl.mean() < 1e-3 and kl.max() < 0.05, (kl.mean(), kl.max())
    assert agree >= 0.95, agree


def test_q8_off_is_the_reference_bitwise():
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 30))
    plan = [(0, 12)] + [(i, i + 1) for i in range(12, 30)]
    base = _run(model, tokens, plan)
    model.configure_kernels(kv_q8=True)
    _run(model, tokens, plan)
    model.configure_kernels()
    assert model.apc_v2_layout == "inkling2-kv-four-shortconv-v1"
    assert mx.array_equal(_run(model, tokens, plan), base)


def test_q8_chunking_and_decode_agree_with_one_prefill():
    """Quantization is per row, so the Q8 model is chunk-invariant up to the
    FP32 chunking noise of the BF16 model (bounded by 1e-4 in
    tests/test_inkling2_cpu.py), which can move a code across a rounding
    boundary: one int8 step on one channel."""
    model = _tiny_model()
    model.configure_kernels(kv_q8=True)
    tokens = mx.random.randint(0, 500, (1, 21))
    base = _run(model, tokens, [(0, 21)])
    decode = _run(model, tokens, [(0, 16)] + [(i, i + 1) for i in range(16, 21)])
    chunks = _run(model, tokens, [(0, 7), (7, 14), (14, 21)])
    assert mx.abs(decode - base).max().item() < 1e-3
    assert mx.abs(chunks - base).max().item() < 1e-3


def test_q8_planes_hold_int8_rows_and_rings_stay_float():
    model = _tiny_model()
    model.configure_kernels(kv_q8=True)
    cache = model.make_cache()
    model(mx.random.randint(0, 500, (1, 20)), cache=cache)
    kinds = {}
    for visit, plane in enumerate(cache[0::2]):
        kinds.setdefault(type(plane).__name__, set()).add(plane.keys.dtype)
        if type(plane).__name__ == "KVCache":
            assert plane.keys.shape[-1] == plane.values.shape[-1] == 48
    assert kinds == {"KVCache": {mx.int8}, "RotatingKVCache": {mx.float32}}
    held = kv_q8.kv_bytes(cache[0::2])
    assert held["q8"] > 0 and held["other"] > 0


def test_no_cache_forward_reads_the_stored_values():
    """Without a cache the global visits pack this call's K/V and attend over
    the codes, so a cacheless forward equals a cached prefill."""
    model = _tiny_model()
    model.configure_kernels(kv_q8=True)
    tokens = mx.random.randint(0, 500, (1, 12))
    cached = _run(model, tokens, [(0, 12)])
    assert mx.abs(model(tokens) - cached).max().item() < 1e-5
    assert model.kernel_receipt()["kv_q8"]["calls"]["uncached"] == 8


def _engage_reference(monkeypatch):
    seen = []

    def fake(q, k, v, rel, sinks, **kw):
        assert k.dtype == v.dtype == mx.int8
        seen.append((q.shape[2], kw["window"], kw["left_padding"] is not None))
        return kv_q8.reference_attention(q, k, v, rel, sinks, **kw)

    monkeypatch.setattr(kv_q8, "decline_reason", lambda *a, **kw: None)
    monkeypatch.setattr(kv_q8, "q8_attention", fake)
    return seen


@pytest.mark.parametrize("rows", [1, 2])
def test_kernel_route_wiring_matches_the_fallback(monkeypatch, rows):
    """With the kernel route engaged on the CPU (the oracle in place of the
    Metal kernel), every block's arguments (packed K/V, offset, tau-scaled q
    and bias rows, sinks) reproduce the blockwise fallback (FP32 online
    softmax vs one SDPA: reduction order only)."""
    model = _tiny_model()
    model.configure_kernels(kv_q8=True)
    tokens = mx.random.randint(0, 500, (rows, 30))
    plan = [(0, 9), (9, 17)] + [(i, i + 1) for i in range(17, 30)]
    base = _run(model, tokens, plan)
    seen = _engage_reference(monkeypatch)
    got = _run(model, tokens, plan)
    calls = model.kernel_receipt()["kv_q8"]["calls"]
    assert calls["kernel_decode"] == 8 * 13 and calls["kernel_rows"] == 8 * 2
    assert {w for _, w, _ in seen} == {0}
    assert mx.abs(got - base).max().item() < 1e-3  # FP32 order noise through the loop


def test_batched_left_padding_reaches_the_kernel_route(monkeypatch):
    from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache, KVCache

    model = _tiny_model()
    model.configure_kernels(kv_q8=True)
    tokens = mx.random.randint(0, 500, (2, 24))
    plan = [(0, 12), (12, 20)] + [(i, i + 1) for i in range(20, 24)]

    def run():
        cache = model.make_cache()
        for i in range(0, len(cache), 2):
            cache[i] = (BatchKVCache([0, 3]) if type(cache[i]) is KVCache
                        else BatchRotatingKVCache(8, [0, 3]))
        out = mx.concatenate([model(tokens[:, a:b], cache=cache) for a, b in plan], axis=1)
        return out, cache

    base, cache = run()
    assert cache[10].keys.dtype == mx.int8 and type(cache[10]) is BatchKVCache
    seen = _engage_reference(monkeypatch)
    got, _ = run()
    assert seen and all(padded for _, _, padded in seen)
    assert mx.abs(got - base).max().item() < 1e-3  # FP32 order noise through the loop


def test_merged_lanes_decode_like_single_lanes():
    """Two B1 Q8 lanes of different lengths merged by the batch generator's
    merge (BatchKVCache over int8 rows, left padding) decode as they do
    alone."""
    from mlx2.runtime.generate import _merge_caches

    model = _tiny_model()
    model.configure_kernels(kv_q8=True)
    mx.random.seed(3)
    prompts = [mx.random.randint(0, 500, (1, n)) for n in (14, 9)]
    lanes, alone = [], []
    for p in prompts:
        cache = model.make_cache()
        model(p, cache=cache)
        lanes.append(cache)
    nxt = mx.array([[7], [11]])
    for i, p in enumerate(prompts):
        cache = model.make_cache()
        model(p, cache=cache)
        alone.append(model(nxt[i:i + 1], cache=cache)[0, -1])
    merged = _merge_caches(lanes)
    assert merged[10].keys.dtype == mx.int8
    out = model(nxt, cache=merged)[:, -1]
    for i in range(2):
        assert mx.abs(out[i] - alone[i]).max().item() < 1e-4


def test_trim_rollback_restores_the_pre_verify_state():
    """A rejected verify block (8 rows) on the global planes rolls back by
    trim, exactly as KVCache: decoding after it equals decoding without it."""
    from mlx2.runtime.models.cache import KVCache

    model = _tiny_model()
    model.configure_kernels(kv_q8=True)
    tokens = mx.random.randint(0, 500, (1, 6))
    prefix = mx.random.randint(0, 500, (1, 16))

    def fresh():
        cache = model.make_cache()
        model(prefix, cache=cache)
        return cache

    clean = fresh()
    want = model(tokens[:, :1], cache=clean)
    cache = fresh()
    snapshot = [(c.keys, c.offset) for c in cache[0::2] if type(c) is KVCache]
    for plane in cache:
        plane.start_speculation(8)
    model(mx.random.randint(0, 500, (1, 8)), cache=cache)
    for plane in cache:
        assert plane.trim(8) == 8
        plane.stop_speculation()
    kept = [(c.keys, c.offset) for c in cache[0::2] if type(c) is KVCache]
    assert [o for _, o in kept] == [o for _, o in snapshot]
    assert all(k.dtype == mx.int8 for k, _ in kept)
    got = model(tokens[:, :1], cache=cache)
    assert mx.abs(got - want).max().item() < 1e-5


# ----- fail closed -----


def test_formats_never_mix_in_a_plane():
    model = _tiny_model()
    tokens = mx.random.randint(0, 500, (1, 10))
    bf16 = model.make_cache()
    model(tokens, cache=bf16)
    model.configure_kernels(kv_q8=True)
    with pytest.raises(ValueError, match="Q8 and BF16 caches do not mix"):
        model(tokens[:, :1], cache=bf16)
    q8 = model.make_cache()
    model(tokens, cache=q8)
    model.configure_kernels()
    with pytest.raises(ValueError, match="inkling2_kv_q8 off"):
        model(tokens[:, :1], cache=q8)


def test_configure_refuses_a_head_dim_without_a_layout():
    from mlx2.runtime.models.inkling2 import Model, TextConfig

    model = Model(TextConfig.from_dict({**TINY, "head_dim": 96}))
    before = dict(model.kernel_policy) if hasattr(model, "kernel_policy") else None
    with pytest.raises(ValueError, match="no int8 layout for head_dim 96"):
        model.configure_kernels(kv_q8=True)
    assert (dict(model.kernel_policy) if hasattr(model, "kernel_policy") else None) == before
    assert not any(layer.self_attn.__dict__.get("_kv_q8") for layer in model.model.layers)


def test_policy_key():
    from mlx2.adapters.inkling2 import KERNEL_OPTIONS, validate_policy

    assert KERNEL_OPTIONS["inkling2_kv_q8"] == "kv_q8"
    assert validate_policy(None)[0]["kv_q8"] is False
    assert validate_policy({"inkling2_kv_q8": True})[0]["kv_q8"] is True
    for bad in (1, "true", None):
        with pytest.raises(TypeError, match="inkling2_kv_q8 must be a boolean"):
            validate_policy({"inkling2_kv_q8": bad})


def test_feature_grid_lists_the_option():
    from pathlib import Path

    text = (Path(__file__).parents[1] / "scripts/build_model_feature_grid.py").read_text()
    assert '"inkling2_kv_q8"' in text


def test_prompt_logprobs_refuse_the_approximate_route():
    from mlx2.prompt_logprobs import PromptLogprobsRefused, scoring_fidelity

    adapter = _adapter({"inkling2_kv_q8": True})
    with pytest.raises(PromptLogprobsRefused):
        scoring_fidelity(adapter, numerical_contract=adapter.execution_numerics_contract())


# ----- APCv2 identity -----


def _adapter(policy, model=None):
    from mlx2.adapters.inkling2 import Inkling2Adapter, validate_policy

    adapter = Inkling2Adapter.__new__(Inkling2Adapter)
    adapter.kernel_policy = validate_policy(policy)[0]
    if model is not None:
        adapter.model = model
        adapter.layout = model.apc_v2_layout
    return adapter


def test_numerics_operation_layout_and_receipt_differ_from_default():
    from mlx2.adapters.inkling2 import (
        NUMERICS,
        NUMERICS_KV_Q8,
        NUMERICS_W4A4_FAKE_KV_Q8,
        kv_q8_receipt,
    )

    off = _adapter(None).execution_numerics_contract()
    on = _adapter({"inkling2_kv_q8": True}).execution_numerics_contract()
    both = _adapter({"inkling2_kv_q8": True, "inkling2_moe_act_quant": True}
                    ).execution_numerics_contract()
    assert off["operation"] == NUMERICS and "approximate_state" not in off
    assert "inkling2/kv_q8.py" not in off["source_sha256"]
    assert on["operation"] == NUMERICS_KV_Q8 == "inkling2-native-w4a16-kvq8-v2"
    assert both["operation"] == NUMERICS_W4A4_FAKE_KV_Q8
    assert len(on["source_sha256"]["inkling2/kv_q8.py"]) == 64
    assert on["approximate_state"]["operation"] == "inkling2_kv_q8"
    assert on["approximate_state"]["fidelity"] == "approximate"
    assert on["approximate_state"]["qualified"] is False
    assert on["qualification"] == "unqualified"
    receipt = kv_q8_receipt(False)
    assert receipt["selected"] is False and receipt["fidelity"] == "exact"
    model = _tiny_model()
    base_layout = model.apc_v2_layout
    model.configure_kernels(kv_q8=True)
    assert model.apc_v2_layout == base_layout + "-kvq8v2"
    model.configure_kernels(kv_q8=True, loop_passes=2)
    assert model.apc_v2_layout == base_layout + "-passes2-kvq8v2"


def test_budget_charges_int8_rows_for_the_global_visits():
    import dataclasses

    from test_inkling2_memory_cpu import _allocated

    from mlx2.adapters.inkling2_memory import Inkling2CacheBudget
    from mlx2.runtime.models.inkling2 import TextConfig

    config = TextConfig.from_dict(TINY)
    plain = dataclasses.replace(Inkling2CacheBudget.from_config(config), prefill_rows=16,
                                item_bytes=4)
    q8 = dataclasses.replace(Inkling2CacheBudget.from_config(config, kv_q8=True),
                             prefill_rows=16, item_bytes=4)
    assert q8.global_row_bytes == 48 and plain.global_row_bytes == 0
    assert q8.project(300) < plain.project(300)
    model = _tiny_model()
    model.configure_kernels(kv_q8=True)
    cache = model.make_cache()
    ids = mx.random.randint(1, 400, (1, 300))
    for a in range(0, 300, 16):
        mx.eval(model(ids[:, a:a + 16], cache=cache))
    assert _allocated(cache) <= q8.project(300)
    adapter = _adapter({"inkling2_kv_q8": True}, model)
    assert adapter.cache_budget(mtp=False).global_row_bytes == 48


def _apc_key(adapter, model):
    from mlx2.runtime.apc_v2 import APCKey
    from mlx2.serving import adapter_numerical_contract, apc_semantic_namespace

    semantic = apc_semantic_namespace(
        "tenant", adapter_execution_numerics=adapter_numerical_contract(adapter))
    return APCKey("inkling2-tiny", cache_layout_fingerprint=model.apc_v2_layout,
                  semantic_fingerprint=semantic)


def test_apcv2_freeze_restore_round_trip_and_namespace():
    """Cache-level readiness for a qualified publication later (serving
    refuses it today, see below): a Q8 lane stored directly in APCv2
    restores to a cache that decodes exactly as the live one, and the BF16
    namespace (layout and numerics) never sees it."""
    from mlx2.runtime.apc_v2 import APCv2

    model = _tiny_model()
    bf16_key = _apc_key(_adapter(None), model)
    model.configure_kernels(kv_q8=True)
    q8_key = _apc_key(_adapter({"inkling2_kv_q8": True}), model)
    assert q8_key != bf16_key
    assert q8_key.cache_layout_fingerprint != bf16_key.cache_layout_fingerprint
    assert q8_key.semantic_fingerprint != bf16_key.semantic_fingerprint

    mx.random.seed(9)
    prompt = [int(t) for t in mx.random.randint(1, 400, (24,)).tolist()]
    live = model.make_cache()
    model(mx.array([prompt]), cache=live)
    apc = APCv2(max_size=4, layout_name=model.apc_v2_layout)
    apc.store(q8_key, prompt, live)
    follow = [5, 9, 13]
    want = [model(mx.array([[t]]), cache=live)[0, -1] for t in follow]

    assert apc.lookup(bf16_key, prompt + follow).cached_tokens == 0
    hit = apc.lookup(q8_key, prompt + follow)
    assert hit.cached_tokens == len(prompt)
    restored = hit.cache
    assert restored[10].keys.dtype == mx.int8
    got = [model(mx.array([[t]]), cache=restored)[0, -1] for t in follow]
    for a, b in zip(got, want):
        assert mx.array_equal(a, b)


# ----- DSpark -----


def test_dspark_greedy_equals_q8_ordinary(monkeypatch):
    """DSpark verify rows (4) and their rollback over Q8 global planes: greedy
    output equals ordinary Q8 decode, with the kernel route engaged on the
    CPU (oracle in place of the Metal kernel) and without."""
    import test_inkling2_dspark_cpu as d

    target = d.Model(d.TextConfig.from_dict(d.TEXT))
    d._random(target, 0)
    draft = d.Inkling2DSparkDraftModel(d.Inkling2DSparkConfig.from_hf(d._config()))
    d._random(draft, 1)
    draft.bind(target)
    prompt = [int(t) for t in mx.random.randint(1, 400, (40,)).tolist()]
    target.configure_kernels(kv_q8=True)
    expect = d._ordinary_greedy(target, prompt)
    assert d._dspark_greedy(target, draft, prompt) == expect
    seen = _engage_reference(monkeypatch)
    assert d._dspark_greedy(target, draft, prompt) == expect
    assert 4 in {rows for rows, _, _ in seen}


# ----- APCv2 publication refused (unqualified approximate state) -----


def test_adapter_refuses_reuse_and_publication_while_unqualified():
    from mlx2.adapters import inkling2 as adapter_mod

    assert adapter_mod.KV_Q8_PUBLICATION_QUALIFIED is False
    off = _adapter(None)
    on = _adapter({"inkling2_kv_q8": True})
    assert "apcv2_reuse_disabled_reason" not in off.execution_config(max_lanes=2, prefill_step=16)
    config = on.execution_config(max_lanes=2, prefill_step=16)
    assert config["apcv2_reuse_disabled_reason"] == adapter_mod.KV_Q8_APC_REASON
    model = _tiny_model()
    exact = model.make_cache()
    model(mx.random.randint(0, 500, (1, 6)), cache=exact)
    assert off.apc_publication_refusal(exact, list(range(6))) is None
    model.configure_kernels(kv_q8=True)
    q8 = model.make_cache()
    model(mx.random.randint(0, 500, (1, 6)), cache=q8)
    assert on.apc_publication_refusal(q8, list(range(6))) == adapter_mod.KV_Q8_APC_REASON
    # Structural: int8 rows are refused even under a policy that says off.
    assert off.apc_publication_refusal(q8, list(range(6))) == adapter_mod.KV_Q8_APC_REASON
    assert on.__dict__["kv_q8_apc_declines"] == 1 and off.__dict__["kv_q8_apc_declines"] == 1
    receipt = adapter_mod.kv_q8_receipt(True)["apcv2_publication"]
    assert receipt == {"enabled": False, "reason": adapter_mod.KV_Q8_APC_REASON,
                       "qualified_switch": False}
    assert adapter_mod.kv_q8_receipt(False)["apcv2_publication"]["enabled"] is True


def _q8_engine(monkeypatch, kv_q8_on):
    import route_harness as h

    from mlx2.adapters.inkling2 import Inkling2Adapter, validate_policy

    model = _tiny_model()
    model.configure_kernels(kv_q8=kv_q8_on)
    policy = validate_policy({"inkling2_kv_q8": kv_q8_on})[0]

    class Inkling2Route:
        kernel_policy = policy
        bind_prefill_step = Inkling2Adapter.bind_prefill_step
        _route_execution_config = Inkling2Adapter._route_execution_config
        execution_config = Inkling2Adapter.execution_config
        apc_publication_refusal = Inkling2Adapter.apc_publication_refusal

    h.patch_host(monkeypatch)
    return h, h.make_engine(model, 500, mtp=False, adapter_mixin=Inkling2Route)


def _cached(result):
    apc = result["receipt"].get("apcv2") or {}
    return result["receipt"].get("cached_tokens", apc.get("cached_tokens", 0))


@pytest.mark.parametrize("kv_q8_on", [False, True])
def test_serving_q8_request_is_never_stored_nor_reused(monkeypatch, kv_q8_on):
    """Two identical requests through a real ServingEngine: with kv_q8 off the
    second reuses the first's APCv2 entry (control); with it on nothing is
    stored, the second gets 0 cached tokens and the declines are counted."""
    h, engine = _q8_engine(monkeypatch, kv_q8_on)
    request = {"tokens": [int(t) for t in mx.random.randint(1, 400, (40,)).tolist()],
               "max_tokens": 3, "temperature": 0}
    try:
        first = h.run(engine, request)
        second = h.run(engine, request)
        status = engine.status()
        from mlx2.runtime.apc_v2 import _iter_trie_entries

        entries = len(list(_iter_trie_entries(engine.apc._trie)))
        held = (engine.apc._n_bytes, engine.apc._disk_bytes)
    finally:
        engine.close()
    assert "error" not in first and "error" not in second, (first, second)
    counts = status["counts"]
    if not kv_q8_on:
        assert entries > 0 and held[0] > 0
        assert _cached(second) == len(request["tokens"]) - 1, second["receipt"]
        assert not counts.get("apcv2_route_disabled_requests")
        return
    assert entries == 0 and held == (0, 0)  # nothing resident, nothing on disk
    assert _cached(first) == 0 and _cached(second) == 0, second["receipt"]
    assert counts["apcv2_route_disabled_requests"] == 2
    assert counts["apcv2_write_suppressed_requests"] == 2
    assert counts["apcv2_lookup_bypassed_route"] == 2
    assert status["settings"]["apcv2_reuse"] == {
        "enabled": False, "reason": "inkling2_kv_q8_unqualified_approximate_state"}
    assert second["receipt"]["apcv2_reuse"]["enabled"] is False


# ----- prefill without a whole-prefix dequantize -----


def _attn_inputs(b, length, size, seed, pad=None, heads=4, kv_heads=2, dim=64, extent=64,
                 dtype=mx.float32):
    mx.random.seed(seed)
    q = (mx.random.normal((b, length, heads, dim)) * 0.5).astype(dtype).transpose(0, 2, 1, 3)
    k = kv_q8.pack((mx.random.normal((b, kv_heads, size, dim)) * 0.5).astype(dtype))
    v = kv_q8.pack(mx.random.normal((b, kv_heads, size, dim)).astype(dtype))
    rel = (mx.random.normal((b, length, heads, extent)) * 2).astype(dtype)
    sinks = (mx.random.normal((b, length, heads)) * 2).astype(dtype)
    left = None if pad is None else mx.array(pad, dtype=mx.int32)
    return q, k, v, rel, sinks, left


@pytest.mark.parametrize("length", [32, 100, 512])
@pytest.mark.parametrize("window", [0, 300])
@pytest.mark.parametrize("b", [1, 2])
def test_blockwise_fallback_matches_the_oracle(length, window, b):
    """Prefill chunks of 32/100/512 rows over 5000 keys, key blocks of 4096
    (two blocks) and 700 (eight), with the bias band, window, sink and left
    padding: within the split-K kernels' FP32 tolerance."""
    pad = None if b == 1 else [0, 37]
    size = 5000
    q, k, v, rel, sinks, left = _attn_inputs(b, length, size, seed=length + window + b, pad=pad)
    kw = dict(offset=size - length, left_padding=left, window=window, scale=64 ** -0.5)
    ref = kv_q8.reference_attention(q, k, v, rel, sinks, **kw)
    for block in (None, 700):
        counts = {}
        got = kv_q8.blockwise_attention(q, k, v, rel, sinks, block=block, counts=counts, **kw)
        diff = mx.abs(got - ref).max().item()
        assert diff <= 2e-5 * max(1.0, mx.abs(ref).max().item()), (block, diff)
        n_keys = min(size, window + length - 1) if window else size
        assert counts["block_dequant_calls"] >= -(-n_keys // (block or kv_q8.KEY_BLOCK))


@pytest.mark.parametrize("length", [32, 100, 512])
def test_tiled_prefill_matches_the_oracle(monkeypatch, length):
    """The GPU prefill route's tiling (31-row tiles, each at its own query
    offset, with its own bias rows and sinks) with the oracle standing in
    for the Metal kernel per tile: equal to one oracle call over the chunk."""
    size = 5000
    q, k, v, rel, sinks, left = _attn_inputs(2, length, size, seed=length, pad=[0, 37])
    kw = dict(offset=size - length, left_padding=left, window=0, scale=64 ** -0.5)
    ref = kv_q8.reference_attention(q, k, v, rel, sinks, **kw)
    tiles = []

    def fake(q, k, v, rel, sinks, **kw):
        tiles.append((q.shape[2], kw["offset"]))
        return kv_q8.reference_attention(q, k, v, rel, sinks, **kw)

    monkeypatch.setattr(kv_q8, "q8_attention", fake)
    counts = {}
    got = kv_q8.tiled_attention(q, k, v, rel, sinks, counts=counts, **kw)
    assert counts["prefill_tiles"] == len(tiles) == -(-length // kv_q8.PREFILL_TILE)
    assert all(n <= kv_q8.PREFILL_TILE for n, _ in tiles)
    assert [o for _, o in tiles] == list(range(size - length, size, kv_q8.PREFILL_TILE))
    assert mx.abs(got - ref).max().item() <= 2e-5 * max(1.0, mx.abs(ref).max().item())


def test_model_prefill_routes_through_tiles_and_matches(monkeypatch):
    """The model's prefill chunks (40 rows: two tiles) take the tiled route
    when the kernel is available (oracle per tile on the CPU) and match the
    blockwise route, with the ring wrapped."""
    model = _tiny_model()
    model.configure_kernels(kv_q8=True)
    tokens = mx.random.randint(0, 500, (1, 90))
    plan = [(0, 40), (40, 80)] + [(i, i + 1) for i in range(80, 90)]
    base = _run(model, tokens, plan)
    _engage_reference(monkeypatch)
    got = _run(model, tokens, plan)
    receipt = model.kernel_receipt()["kv_q8"]
    assert receipt["calls"]["kernel_prefill_tiled"] == 8 * 2
    assert receipt["kv_q8_prefill_tiles"] == 8 * 2 * 2
    assert mx.abs(got - base).max().item() < 1e-3  # FP32 order noise through the loop


def test_serving_prefill_never_dequantizes_the_whole_prefix(monkeypatch):
    """Chunked prefill plus decode over 300 tokens with 64-key blocks: the
    oracle (whole-prefix dequantize) is never called, and no dequantize ever
    sees more than one key block."""
    model = _tiny_model()
    model.configure_kernels(kv_q8=True)
    monkeypatch.setattr(kv_q8, "KEY_BLOCK", 64)

    def forbidden(*a, **kw):
        raise AssertionError("serving called the whole-prefix dequantize")

    widths = []
    original = kv_q8.dequantize

    def watched(packed, head_dim, dtype):
        widths.append(packed.shape[-2])
        return original(packed, head_dim, dtype)

    monkeypatch.setattr(kv_q8, "reference_attention", forbidden)
    monkeypatch.setattr(kv_q8, "dequantize", watched)
    before = kv_q8.full_prefix_dequant_calls()
    tokens = mx.random.randint(0, 500, (1, 300))
    _run(model, tokens, [(a, a + 64) for a in range(0, 256, 64)] + [(256, 290)]
         + [(i, i + 1) for i in range(290, 300)])
    receipt = model.kernel_receipt()["kv_q8"]
    assert kv_q8.full_prefix_dequant_calls() == before
    assert receipt["kv_q8_block_dequant_calls"] > 8 * 15
    assert widths and max(widths) <= 64
