"""CPU tests for the Qwen3.8 27B fused GDN prefill (omlx #3903 kernels).

The 27B variant reuses the Flash-Next prework/norm-gate kernels with the
shared Qwen3.5 numerics: RMS q/k normalization (``normalize_gdn_qk``) and a
swish output gate (``Qwen3NextRMSNormGated``).  Metal cannot run here, so the
kernel arithmetic is mirrored in numpy at its rounding sites, the mirrors are
held against the eager 27B path on CPU, and the wiring runs with the kernel
wrappers monkeypatched to the mirrors.  The GPU bit gate is
``scripts/check_qwen38_27b_fused_gdn_prefill.py --i-own-the-gpu``.
"""

import hashlib

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx2.runtime.models import qwen38_fused_gdn as route
from mlx2.runtime.models import qwen4_fused_gdn_prefill as gdn_prefill
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.qwen3_5 import TextModelArgs
from mlx2.runtime.models.qwen38_27b import GatedDeltaNet

HK, HV, DK, DV, K = 16, 48, 128, 128, 4
C = 2 * HK * DK + HV * DV
NKEEP = K - 1
HIDDEN = 32


# --------------------------------------------------------------------------
# numpy mirrors
# --------------------------------------------------------------------------


def bf16(x):
    """Round float32 values to bfloat16 (round-to-nearest-even), as float32."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    out = rounded.astype(np.uint32).view(np.float32)
    return np.where(np.isnan(x), x, out).astype(np.float32)


def f32(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32))


def _butterfly(lanes):
    vals = lanes.astype(np.float32)
    for mask in (16, 8, 4, 2, 1):  # stand-in for simd_sum, fp32
        vals = (vals + vals[..., np.arange(32) ^ mask]).astype(np.float32)
    return vals[..., 0]


def ref_prework_qwen35(qkv, conv_state, conv_w):
    """Mirror of the prework kernel's RMS branch (``L2 = 0``), Qwen3.5 eps.

    qkv (S, C), conv_state (3, C), conv_w (C, 4): float32 holding bf16 values.
    """
    S = qkv.shape[0]
    x = np.concatenate([conv_state, qkv], axis=0)
    acc = np.zeros((S, C), dtype=np.float32)
    for tap in range(K):  # fp32 accumulate in tap order, one bf16 round
        acc = acc + x[tap : tap + S] * conv_w[None, :, tap]
    conv = bf16(acc)
    sy = bf16(np.float32(1) / (np.float32(1) + np.exp(np.abs(conv))))
    sig = np.where(conv < 0, sy, bf16(np.float32(1) - sy))
    act = bf16(conv * sig)
    inv_scale = DK ** (-0.5)
    eps = np.float32(np.float32(1e-6) / np.float32(DK))

    def rms(block, scale):
        lanes = block.reshape(S, HK, 32, 4)
        lane_sum = np.zeros((S, HK, 32), dtype=np.float32)
        for i in range(4):  # exact fp32 squares, sequential per lane
            lane_sum = (lane_sum + lanes[..., i] * lanes[..., i]).astype(np.float32)
        total = _butterfly(lane_sum)
        inv = (1.0 / np.sqrt((total / np.float32(DK) + eps).astype(np.float64))).astype(
            np.float32
        )
        normed = bf16(lanes * inv[..., None, None])
        return bf16(normed * bf16(np.float32(scale))).reshape(S, HK, DK)

    kd = HK * DK
    q = rms(act[:, :kd], inv_scale**2)
    k = rms(act[:, kd : 2 * kd], inv_scale)
    v = act[:, 2 * kd :].reshape(S, HV, DV)
    return q, k, v, x[-NKEEP:]


def ref_norm_gate_swish(y, z, w, eps):
    """Mirror of the swish norm-gate kernel: y (S, HV, DV), z (S, HV*DV)."""
    S = y.shape[0]
    lanes = y.reshape(S, HV, 32, 4)
    lane_sum = np.zeros((S, HV, 32), dtype=np.float32)
    for i in range(4):
        lane_sum = (lane_sum + lanes[..., i] * lanes[..., i]).astype(np.float32)
    total = _butterfly(lane_sum)
    inv = (1.0 / np.sqrt((total / np.float32(DV) + np.float32(eps)).astype(np.float64)))
    inv = inv.astype(np.float32)
    normed = bf16(w[None, None, :] * bf16(y * inv[..., None]))
    zf = z.reshape(S, HV, DV).astype(np.float32)
    sigmoid = (1.0 / (1.0 + np.exp(-zf.astype(np.float64)))).astype(np.float32)
    gate = (zf * sigmoid).astype(np.float32)
    return bf16(normed * gate).reshape(S, HV * DV)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _args():
    return TextModelArgs(
        hidden_size=HIDDEN, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=32,
        linear_num_key_heads=HK, linear_num_value_heads=HV,
        linear_key_head_dim=DK, linear_value_head_dim=DV, linear_conv_kernel_dim=K,
    )


@pytest.fixture
def layer():
    mx.random.seed(3838)
    gdn = GatedDeltaNet(_args())
    gdn.set_dtype(mx.bfloat16)
    gdn.A_log = gdn.A_log.astype(mx.float32)
    gdn.norm.weight = (1.0 + 0.1 * mx.random.normal((DV,))).astype(mx.bfloat16)
    gdn.conv1d.weight = (0.5 * mx.random.normal((C, K, 1))).astype(mx.bfloat16)
    gdn.eval()
    return gdn


@pytest.fixture
def kernels_on_cpu(monkeypatch):
    """Route the Metal wrappers to the numpy mirrors; record the variants."""
    calls = {"prework": [], "norm_gate": []}

    def prework(qkv, conv_state, conv_weight, *, numerics="qwen4", **_):
        calls["prework"].append(numerics)
        assert numerics == "qwen35"
        state = (
            np.zeros((NKEEP, C), np.float32)
            if conv_state is None
            else f32(conv_state)[0]
        )
        q, k, v, nxt = ref_prework_qwen35(
            f32(qkv)[0], state, f32(conv_weight)[:, :, 0]
        )
        to = lambda a: mx.array(a[None]).astype(qkv.dtype)
        return to(q), to(k), to(v), to(nxt)

    def norm_gate(y, z, norm_weight, eps, *, gate="sigmoid", **_):
        calls["norm_gate"].append(gate)
        assert gate == "swish"
        out = ref_norm_gate_swish(f32(y)[0], f32(z)[0], f32(norm_weight), eps)
        return mx.array(out[None]).astype(y.dtype)

    monkeypatch.setattr(gdn_prefill, "runtime_supported", lambda: True)
    monkeypatch.setattr(route.kernels, "served_silu_refusal", lambda: None)
    monkeypatch.setattr(gdn_prefill, "qwen4_gdn_prefill_prework", prework)
    monkeypatch.setattr(gdn_prefill, "qwen4_gdn_prefill_norm_gate", norm_gate)
    return calls


def _inputs(S, seed=0, rows=1):
    return (0.5 * mx.random.normal((rows, S, HIDDEN), key=mx.random.key(seed))).astype(
        mx.bfloat16
    )


# --------------------------------------------------------------------------
# kernel sources: Flash-Next untouched, 27B variant differs where it must
# --------------------------------------------------------------------------


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_flash_next_sources_are_byte_identical_to_the_qualified_ones():
    # Pinned from origin/main 27a7f923b, the sources the 2026-09-25 GPU gate
    # found bit-identical; the 27B variants must not perturb them.
    assert _sha(gdn_prefill.prefill_prework_source()) == (
        "493af85eca7ab4d4ebb8c4625670741569e9972065c9334726ac2d85a5b8659c"
    )
    assert _sha(gdn_prefill.prefill_norm_gate_source()) == (
        "2289d0e785731668c8636575fa1342e50863fc4bc57db98aeff3d98a1a4ef2e2"
    )
    assert gdn_prefill.prefill_prework_source("qwen4") == gdn_prefill.prefill_prework_source()
    assert gdn_prefill.prefill_norm_gate_source("sigmoid") == (
        gdn_prefill.prefill_norm_gate_source()
    )


def test_qwen35_variants_change_only_eps_and_gate():
    base = gdn_prefill.prefill_prework_source().splitlines()
    var = gdn_prefill.prefill_prework_source("qwen35").splitlines()
    changed = [(a, b) for a, b in zip(base, var) if a != b]
    assert len(base) == len(var) and len(changed) == 1
    assert "sumsq / float(DK) + 1e-6f" in changed[0][0]
    assert "sumsq / float(DK) + 1.0e-6f / float(DK)" in changed[0][1]
    base = gdn_prefill.prefill_norm_gate_source().splitlines()
    var = gdn_prefill.prefill_norm_gate_source("swish").splitlines()
    changed = [(a, b) for a, b in zip(base, var) if a != b]
    assert len(base) == len(var) and len(changed) == 2
    assert "metal::exp(metal::abs(zv))" in changed[0][1]
    assert "precise" not in changed[0][1]
    assert "zv * (zv < 0.0f ? sy : 1.0f - sy)" in changed[1][1]
    with pytest.raises(ValueError):
        gdn_prefill.prefill_prework_source("agnes")
    with pytest.raises(ValueError):
        gdn_prefill.prefill_norm_gate_source("silu")


def test_qwen35_eps_equals_the_eager_rms_norm_eps():
    inv_scale = DK ** (-0.5)
    eager = np.float32(1e-06 * inv_scale**2)  # normalize_gdn_qk's float arg
    kernel = np.float32(np.float32(1.0e-6) / np.float32(DK))
    assert eager == kernel


# --------------------------------------------------------------------------
# mirrors describe the eager 27B semantics (CPU: tolerance-level)
# --------------------------------------------------------------------------


def test_qwen35_prework_mirror_tracks_eager_normalize_gdn_qk(layer):
    S = 64
    rng = np.random.default_rng(1)
    qkv = mx.array(rng.normal(size=(1, S, C)).astype(np.float32)).astype(mx.bfloat16)
    state = mx.array(rng.normal(size=(1, NKEEP, C)).astype(np.float32)).astype(
        mx.bfloat16
    )
    conv_input = mx.concatenate([state, qkv], axis=1)
    conv_out = nn.silu(layer.conv1d(conv_input))
    kd = HK * DK
    q = conv_out[..., :kd].reshape(1, S, HK, DK)
    k = conv_out[..., kd : 2 * kd].reshape(1, S, HK, DK)
    v = conv_out[..., 2 * kd :].reshape(1, S, HV, DV)
    q, k = layer._normalize_qk(q, k)
    # Mirror the RMS branch from the eager activations, isolating it from the
    # CPU-vs-Metal SiLU difference.
    act = f32(conv_out)[0]
    _, _, rv, rstate = ref_prework_qwen35(
        f32(qkv)[0], f32(state)[0], f32(layer.conv1d.weight)[:, :, 0]
    )
    np.testing.assert_allclose(rv, f32(v)[0], rtol=2**-5, atol=2**-14)
    inv_scale = DK ** (-0.5)
    eps = np.float32(np.float32(1e-6) / np.float32(DK))
    for name, eager, block, scale in (
        ("q", q, act[:, :kd], inv_scale**2),
        ("k", k, act[:, kd : 2 * kd], inv_scale),
    ):
        lanes = block.reshape(S, HK, 32, 4)
        lane_sum = np.zeros((S, HK, 32), np.float32)
        for i in range(4):
            lane_sum = (lane_sum + lanes[..., i] * lanes[..., i]).astype(np.float32)
        total = _butterfly(lane_sum)
        inv = (1.0 / np.sqrt((total / np.float32(DK) + eps).astype(np.float64))).astype(
            np.float32
        )
        mirror = bf16(bf16(lanes * inv[..., None, None]) * bf16(np.float32(scale)))
        got = f32(eager)[0].reshape(S, HK, 32, 4)
        # CPU rms_norm reduces in a different fp32 order: at most one bf16 ulp
        # on a small fraction of elements.
        diff = np.abs(mirror - got)
        ulp = np.abs(got) * 2.0**-7 + 1e-30
        assert np.all(diff <= ulp), (name, float((diff / ulp).max()))
        assert np.mean(diff > 0) < 0.01, (name, float(np.mean(diff > 0)))
    assert np.array_equal(rstate, f32(conv_input[:, -NKEEP:])[0])


def test_swish_norm_gate_mirror_tracks_eager_qwen3next_norm(layer):
    S = 64
    rng = np.random.default_rng(2)
    y = mx.array(rng.normal(size=(1, S, HV, DV)).astype(np.float32)).astype(mx.bfloat16)
    z = mx.array(rng.normal(size=(1, S, HV * DV)).astype(np.float32)).astype(mx.bfloat16)
    eager = layer.norm(y, z.reshape(1, S, HV, DV)).reshape(1, S, -1)
    ref = ref_norm_gate_swish(f32(y)[0], f32(z)[0], f32(layer.norm.weight), layer.norm.eps)
    got = f32(eager)[0]
    diff = np.abs(ref - got)
    assert np.all(diff <= np.abs(got) * 2.0**-7 + 1e-30), float(diff.max())
    assert np.mean(diff > 0) < 0.01


# --------------------------------------------------------------------------
# admission
# --------------------------------------------------------------------------


def _admission_kwargs(S=64, **overrides):
    kw = dict(
        qkv=mx.zeros((1, S, C), mx.bfloat16),
        z=mx.zeros((1, S, HV * DV), mx.bfloat16),
        b=mx.zeros((1, S, HV), mx.bfloat16),
        a=mx.zeros((1, S, HV), mx.bfloat16),
        conv_state=None,
        recurrent_state=None,
        conv_weight=mx.zeros((C, K, 1), mx.bfloat16),
        norm_weight=mx.zeros((DV,), mx.bfloat16),
        mask=None,
        has_cache=True,
        lengths=None,
        left_padding=None,
        speculating=False,
        training=False,
        sharded=False,
        num_key_heads=HK,
        num_value_heads=HV,
        key_head_dim=DK,
        value_head_dim=DV,
        conv_kernel=K,
        gate_activation="swish",
        architecture="qwen38",
    )
    kw.update(overrides)
    return kw


def test_admission_accepts_the_27b_contract():
    assert gdn_prefill.admit_qwen4_fused_gdn_prefill(**_admission_kwargs()).accepted
    warm = _admission_kwargs(
        S=512,
        conv_state=mx.zeros((1, NKEEP, C), mx.bfloat16),
        recurrent_state=mx.zeros((1, HV, DV, DK), mx.float16),
    )
    assert gdn_prefill.admit_qwen4_fused_gdn_prefill(**warm).accepted


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (dict(gate_activation="sigmoid"), "output gate 'sigmoid'"),
        (dict(architecture="qwen4"), "output gate 'swish'"),
        (dict(architecture="qwen35"), "unsupported architecture 'qwen35'"),
        (dict(num_value_heads=32), "unsupported geometry"),
        (dict(speculating=True), "speculative rollback"),
        (dict(mask=mx.ones((1, 64), mx.bool_)), "masked prefill"),
        (dict(qkv=mx.zeros((2, 64, C), mx.bfloat16)), "batch of 2 rows"),
        (dict(qkv=mx.zeros((1, 63, C), mx.bfloat16)), "rows 63 < 64"),
        (dict(norm_weight=mx.zeros((DV,), mx.float32)), "norm_weight dtype"),
    ],
)
def test_admission_refuses_with_reason(overrides, reason):
    admission = gdn_prefill.admit_qwen4_fused_gdn_prefill(**_admission_kwargs(**overrides))
    assert not admission.accepted
    assert admission.reason.startswith(reason), admission.reason


# --------------------------------------------------------------------------
# wiring in the 27B layer
# --------------------------------------------------------------------------


def _run(layer, chunks, prefill, cache=None):
    layer.set_fused_gdn_prefill_enabled(prefill)
    cache = ArraysCache(2) if cache is None else cache
    advanced = []
    original = cache.advance

    def spy(n):
        advanced.append(n)
        return original(n)

    cache.advance = spy
    outs = [layer(x, cache.make_mask(x.shape[1]), cache) for x in chunks]
    mx.eval(outs, cache[0], cache[1])
    return outs, cache, advanced


def test_default_is_off_and_untouched(layer, kernels_on_cpu):
    assert layer.fused_gdn_prefill_enabled is False
    _run(layer, [_inputs(64)], False)
    assert kernels_on_cpu == {"prework": [], "norm_gate": []}
    assert layer.fused_gdn_counters["prefill_calls"] == 0
    assert layer.fused_gdn_counters["fallbacks"] == 0


@pytest.mark.parametrize("decode_switch", [False, True])
def test_fused_prefill_cold_and_warm_chunks_match_eager(
    layer, kernels_on_cpu, decode_switch
):
    """The uninitialized-cache first chunk and a warm 512-row chunk both fuse."""
    layer.set_fused_gdn_enabled(decode_switch)
    chunks = [_inputs(64, seed=1), _inputs(512, seed=2), _inputs(96, seed=3)]
    eager_outs, eager_cache, eager_adv = _run(layer, chunks, False)
    if decode_switch:
        # The decode switch alone books today's refusals for these chunks.
        reasons = layer.fused_gdn_counters["reasons"]
        assert reasons.pop("uninitialized cache") == 1
        assert sorted(reasons) == sorted(
            r for r in reasons if r.startswith(("verify width 512 above",
                                               "verify width 96 above"))
        ) and len(reasons) == 2
    layer.fused_gdn_counters.update(fallbacks=0, reasons={})
    fused_outs, fused_cache, fused_adv = _run(layer, chunks, True)
    assert kernels_on_cpu == {
        "prework": ["qwen35"] * 3, "norm_gate": ["swish"] * 3,
    }
    counts = layer.fused_gdn_counters
    assert counts["prefill_calls"] == counts["prefill_chunk_calls"] == 3
    assert counts["prefill_tokens"] == counts["prefill_chunk_tokens"] == 672
    assert counts["fallbacks"] == 0 and counts["reasons"] == {}
    assert fused_adv == eager_adv == [64, 512, 96]
    assert np.array_equal(f32(fused_cache[0]), f32(eager_cache[0]))
    assert fused_cache[1].shape == (1, HV, DV, DK)
    assert fused_cache[1].dtype == eager_cache[1].dtype
    for fused, eager in zip(fused_outs, eager_outs):
        assert fused.shape == eager.shape and fused.dtype == eager.dtype
        diff = np.abs(f32(fused) - f32(eager))
        assert diff.max() <= 0.05 * np.abs(f32(eager)).max() + 1e-3
    np.testing.assert_allclose(
        f32(fused_cache[1]), f32(eager_cache[1]), rtol=0.05, atol=0.05
    )
    report = route.stats(_Model(layer))
    assert report["prefill_enabled"] is True and report["enabled"] is decode_switch
    assert report["prefill_chunk_calls"] == 3


def test_zero_left_padding_merged_cache_fuses(layer, kernels_on_cpu):
    """The ordinary route's merged cache carries host-known [0] padding."""
    cache = ArraysCache.merge([ArraysCache(2)])
    assert cache.host_left_padding() == [0]
    _run(layer, [_inputs(64, seed=5), _inputs(64, seed=6)], True, cache)
    assert layer.fused_gdn_counters["prefill_chunk_calls"] == 2
    assert layer.fused_gdn_counters["reasons"] == {}


def test_short_chunks_and_speculation_keep_their_existing_routes(layer, kernels_on_cpu):
    layer.set_fused_gdn_prefill_enabled(True)
    # Below the prefill floor with the decode switch off: plain reference.
    cache = ArraysCache(2)
    layer(_inputs(32, seed=7), None, cache)
    assert kernels_on_cpu["prework"] == []
    assert layer.fused_gdn_counters["fallbacks"] == 0
    # A speculative 64-row block is never a prefill chunk.
    layer.set_fused_gdn_enabled(True)
    cache.start_speculation()
    layer(_inputs(64, seed=8), None, cache)
    assert kernels_on_cpu["prework"] == []
    assert layer.fused_gdn_counters["prefill_chunk_calls"] == 0
    assert not any(r.startswith("prefill:") for r in layer.fused_gdn_counters["reasons"])


def test_refusals_are_counted_as_prefill_reasons(layer, kernels_on_cpu):
    layer.set_fused_gdn_prefill_enabled(True)
    want = layer(_inputs(64, seed=9, rows=2), None, ArraysCache(2))
    assert layer.fused_gdn_counters["reasons"] == {"prefill: batch of 2 rows": 1}
    padded = ArraysCache(2, left_padding=[3])
    layer(_inputs(64, seed=10), padded.make_mask(64), padded)
    assert layer.fused_gdn_counters["reasons"]["prefill: masked prefill"] == 1
    layer(_inputs(64, seed=11))
    assert layer.fused_gdn_counters["reasons"]["prefill: no cache"] == 1
    assert kernels_on_cpu["prework"] == []
    assert want.shape == (2, 64, HIDDEN)


def test_refusal_returns_the_reference_bits(layer, kernels_on_cpu):
    x = _inputs(64, seed=12, rows=2)
    layer.set_fused_gdn_prefill_enabled(False)
    want = layer(x, None, ArraysCache(2))
    layer.set_fused_gdn_prefill_enabled(True)
    got = layer(x, None, ArraysCache(2))
    assert np.array_equal(f32(got), f32(want))


def test_dispatch_failure_falls_back_with_cache_untouched(
    layer, kernels_on_cpu, monkeypatch
):
    def boom(*_, **__):
        raise RuntimeError("no pipeline")

    monkeypatch.setattr(gdn_prefill, "qwen4_gdn_prefill_norm_gate", boom)
    x = _inputs(64, seed=13)
    eager, eager_cache, _ = _run(layer, [x], False)
    got, cache, adv = _run(layer, [x], True)
    assert layer.fused_gdn_counters["reasons"] == {
        "prefill: Metal kernel dispatch failed: RuntimeError": 1
    }
    assert adv == [64]  # advanced once, by the reference path
    assert np.array_equal(f32(got[0]), f32(eager[0]))
    assert np.array_equal(f32(cache[0]), f32(eager_cache[0]))
    assert np.array_equal(f32(cache[1]), f32(eager_cache[1]))


def test_runtime_and_silu_refusals(layer, kernels_on_cpu, monkeypatch):
    monkeypatch.setattr(gdn_prefill, "runtime_supported", lambda: False)
    _run(layer, [_inputs(64)], True)
    assert layer.fused_gdn_counters["last_fallback"] == "prefill: Metal runtime unavailable"
    monkeypatch.setattr(gdn_prefill, "runtime_supported", lambda: True)
    monkeypatch.setattr(route.kernels, "served_silu_refusal", lambda: "served SiLU uses x")
    _run(layer, [_inputs(64)], True)
    assert layer.fused_gdn_counters["last_fallback"] == "prefill: served SiLU uses x"
    assert kernels_on_cpu["prework"] == []


def test_qwen35_architecture_layers_are_refused(layer, kernels_on_cpu):
    layer.set_fused_gdn_architecture("qwen35")
    _run(layer, [_inputs(64)], True)
    assert layer.fused_gdn_counters["last_fallback"] == (
        "prefill: unsupported architecture 'qwen35'"
    )


# --------------------------------------------------------------------------
# configure / adapter policy
# --------------------------------------------------------------------------


class _Model:
    def __init__(self, *layers):
        self.layers = layers

    def named_modules(self):
        for i, obj in enumerate(self.layers):
            yield f"layers.{i}.linear_attn", obj


def test_configure_sets_prefill_independently(layer):
    route.configure(_Model(layer), False, prefill=True)
    assert layer.fused_gdn_enabled is False and layer.fused_gdn_prefill_enabled is True
    route.configure(_Model(layer), True)
    assert layer.fused_gdn_enabled is True and layer.fused_gdn_prefill_enabled is False
    with pytest.raises(ValueError, match="fused_gdn_prefill must be boolean"):
        route.configure(_Model(layer), True, prefill=1)
    with pytest.raises(ValueError, match="fused_gdn_prefill must be boolean"):
        layer.set_fused_gdn_prefill_enabled("on")


def test_adapter_policy_is_opt_in_and_strict():
    from mlx2.adapters.qwen36_27b import Qwen3627BAdapter
    from mlx2.adapters.qwen38_27b import (
        TARGET_POLICY_KEYS,
        Qwen3827BAdapter,
        fused_gdn_prefill_policy,
    )

    assert Qwen3827BAdapter.default_fused_gdn_prefill is False
    assert "default_fused_gdn_prefill" not in vars(Qwen3627BAdapter)
    assert fused_gdn_prefill_policy({}) is False
    assert fused_gdn_prefill_policy({"fused_gdn_prefill": True}) is True
    for bad in (1, "on", None):
        with pytest.raises(ValueError, match="fused_gdn_prefill must be boolean"):
            fused_gdn_prefill_policy({"fused_gdn_prefill": bad})
    assert {"fused_gdn", "fused_gdn_prefill", "gdn_state_dtype"} <= TARGET_POLICY_KEYS


def test_qualification_requires_prefill_engagement_when_selected():
    from mlx2.qualification import required_feature_checks

    settings = {"environment": {"MLX2_QWEN38_FUSED_GDN_PREFILL": "1"},
                "max_context": 32768, "max_lanes": 4}
    assert "feature_qwen38_fused_gdn_prefill" in required_feature_checks(settings)
    assert "feature_qwen38_fused_gdn_prefill" not in required_feature_checks(
        {"environment": {}, "max_context": 32768, "max_lanes": 4}
    )


# --------------------------------------------------------------------------
# qualification: fused prefill chunks are not decode-switch engagement
# --------------------------------------------------------------------------


def _route_diagnostics(layer):
    # Qwen3827BAdapter._fused_gdn_diagnostics on a linear (non-tree) route.
    return {"architecture": "qwen38", **route.stats(_Model(layer)),
            "tree_calls": 0, "tree_rows": 0}


def test_fused_prefill_chunks_are_not_decode_switch_engagement(
    layer, kernels_on_cpu, monkeypatch
):
    """qwen3x-gdn#0 (sweep 2026-10-08): the decode step kernel declines every
    call and only prefill chunks fuse.  feature_qwen38_fused_gdn (the decode
    switch, MLX2_QWEN38_FUSED_GDN=1) must read 0; the prefill feature reads 1.
    """
    from scripts.qualify_serving import default_on_observations

    monkeypatch.setattr(route.kernels, "fused_gdn_runtime_supported", lambda: True)
    monkeypatch.setattr(route.kernels, "probe_qwen4_fused_gdn_decode",
                        lambda *a, **k: None)
    layer.set_fused_gdn_enabled(True)
    layer.set_fused_gdn_prefill_enabled(True)
    initial = {"execution": {"fused_gdn": _route_diagnostics(layer)}}

    cache = ArraysCache(2)
    mx.eval(layer(_inputs(64, seed=1), cache.make_mask(64), cache))
    for step in range(4):
        mx.eval(layer(_inputs(1, seed=10 + step), cache.make_mask(1), cache))

    diag = _route_diagnostics(layer)
    assert diag["prefill_calls"] == diag["prefill_chunk_calls"] == 1
    assert diag["decode_calls"] == diag["verify_calls"] == 0
    assert diag["reasons"] == {"Metal kernel probe declined": 4}
    observed = default_on_observations({"execution": {"fused_gdn": diag}}, initial)
    assert observed["qwen38_fused_gdn_prefill"] == 1
    # The decode switch's kernels never ran: its feature must fail closed.
    assert observed["qwen38_fused_gdn"] == 0


def test_catch_up_blocks_and_tree_launches_still_count_for_the_decode_switch():
    """Bounded catch-up blocks (prefill_calls without a fused chunk) and the
    owned tree kernel (tree_calls, 1324fd63) are the decode switch's own."""
    from scripts.qualify_serving import default_on_observations

    base = {"enabled": True, "prefill_enabled": True, "decode_calls": 0,
            "batch_decode_calls": 0, "verify_calls": 0, "prefill_calls": 0,
            "prefill_chunk_calls": 0, "tree_calls": 0}
    initial = {"execution": {"fused_gdn": dict(base)}}

    def observed(**counts):
        final = {"execution": {"fused_gdn": {**base, **counts}}}
        return default_on_observations(final, initial)["qwen38_fused_gdn"]

    assert observed(prefill_calls=2) == 2
    assert observed(prefill_calls=5, prefill_chunk_calls=3) == 2
    assert observed(prefill_calls=3, prefill_chunk_calls=3, tree_calls=7) == 7
