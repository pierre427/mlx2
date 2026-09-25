"""CPU plumbing tests for the omlx #3903 port (GDN prefill prework, MoE sum).

The Metal kernels cannot run here (tests are CPU-only and the GPU is
reserved), so each kernel's arithmetic is mirrored in numpy below, following
its rounding sites as written, and the wiring is exercised with the kernel
call monkeypatched to that mirror.  The GPU bit/ULP gate is
``scripts/check_omlx3903_port.py --i-own-the-gpu``.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx2.runtime.models import qwen3_next
from mlx2.runtime.models import qwen4_exp
from mlx2.runtime.models import qwen4_fused_gdn_prefill as gdn_prefill
from mlx2.runtime.models import qwen4_moe_weighted_sum as moe_wsum
from mlx2.runtime.models.cache import ArraysCache

HK, HV, DK, DV, K = 16, 48, 128, 128, 4
C = 2 * HK * DK + HV * DV
NKEEP = K - 1


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


def ref_prework(qkv, conv_state, conv_w, q_scale):
    """Mirror of the prefill prework kernel (L2 variant) for one batch row.

    qkv (S, C), conv_state (3, C), conv_w (C, 4): float32 holding bf16 values.
    Returns q, k (S, HK, DK), v (S, HV, DV), next conv state (3, C).
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

    def l2(block, heads, scale):
        lanes = block.reshape(S, heads, 32, 4)
        sq = bf16(lanes * lanes)
        acc_lane = np.zeros((S, heads, 32), dtype=np.float32)
        for i in range(4):  # sequential bf16 accumulation per lane
            acc_lane = bf16(acc_lane + sq[..., i])
        vals = acc_lane
        for mask in (16, 8, 4, 2, 1):  # xor butterfly, fp32
            vals = (vals + vals[..., np.arange(32) ^ mask]).astype(np.float32)
        total = vals[..., 0]
        eps = bf16(bf16(total) + bf16(np.float32(1e-6)))
        inv = bf16((1.0 / np.sqrt(eps.astype(np.float64))).astype(np.float32))
        out = bf16(lanes * inv[..., None, None])
        if scale is not None:
            out = bf16(out * scale)
        return out.reshape(S, heads, -1)

    kd = HK * DK
    q = l2(act[:, :kd], HK, q_scale)
    k = l2(act[:, kd : 2 * kd], HK, None)
    v = act[:, 2 * kd :].reshape(S, HV, DV)
    return q, k, v, x[-NKEEP:]


def ref_norm_gate(y, z, w, eps):
    """Mirror of the prefill norm-gate kernel: y (S, HV, DV), z (S, HV*DV)."""
    S = y.shape[0]
    sumsq = np.sum(y * y, axis=-1, dtype=np.float32)
    inv = (1.0 / np.sqrt(sumsq / np.float32(DV) + np.float32(eps))).astype(np.float32)
    normed = bf16(w[None, None, :] * bf16(y * inv[..., None]))
    zf = z.reshape(S, HV, DV)
    sig = (1.0 / (1.0 + np.exp(-zf.astype(np.float64)))).astype(np.float32)
    return bf16(normed * sig).reshape(S, HV * DV)


def ref_weighted_sum(x_sorted, inv_order, scores, round_product):
    """Mirror of the MoE weighted-sum kernel: fp32 accumulation, slot order."""
    tokens, top_k = scores.shape
    rows = x_sorted[:, 0, :]
    acc = np.zeros((tokens, rows.shape[-1]), dtype=np.float32)
    for k in range(top_k):
        term = rows[inv_order.reshape(tokens, top_k)[:, k]] * scores[:, k : k + 1]
        if round_product:
            term = bf16(term)
        acc = acc + term
    return acc


# --------------------------------------------------------------------------
# GDN prefill: fixtures
# --------------------------------------------------------------------------


def _text_args(hidden=64, gate="sigmoid"):
    return qwen4_exp.TextModelArgs(
        model_type="qwen4_exp_text", hidden_size=hidden, intermediate_size=0,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, vocab_size=64,
        linear_num_value_heads=HV, linear_num_key_heads=HK,
        linear_key_head_dim=DK, linear_value_head_dim=DV,
        linear_conv_kernel_dim=K,
        layer_types=["linear_attention", "full_attention"],
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, hc_count=2, hc_lowrank=8,
        ple_layer_ids=[1], ple_embed_dim=32, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=128,
        make_ngram_vocab_size_divisible_by=128, split_ngram_parts=1,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=8, indexer_compress_ratio=2, mtp_num_hidden_layers=0,
        output_gate_type=gate,
        rope_parameters={
            "type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25,
        },
    )


@pytest.fixture
def layer():
    mx.random.seed(3903)
    gdn = qwen4_exp.GatedDeltaNet(_text_args())
    gdn.set_dtype(mx.bfloat16)
    gdn.A_log = gdn.A_log.astype(mx.float32)
    gdn.norm.weight = (1.0 + 0.1 * mx.random.normal((DV,))).astype(mx.bfloat16)
    gdn.conv1d.weight = (0.5 * mx.random.normal((C, K, 1))).astype(mx.bfloat16)
    gdn.eval()
    return gdn


@pytest.fixture
def kernels_on_cpu(monkeypatch):
    """Route the Metal wrappers to the numpy mirrors; count the dispatches."""
    calls = {"prework": 0, "norm_gate": 0}

    def prework(qkv, conv_state, conv_weight, **_):
        calls["prework"] += 1
        state = (
            np.zeros((NKEEP, C), np.float32)
            if conv_state is None
            else f32(conv_state)[0]
        )
        q, k, v, nxt = ref_prework(
            f32(qkv)[0], state, f32(conv_weight)[:, :, 0], bf16(np.float32(DK**-0.5))
        )
        to = lambda a: mx.array(a[None]).astype(qkv.dtype)  # noqa: E731
        return to(q), to(k), to(v), to(nxt)

    def norm_gate(y, z, norm_weight, eps, **_):
        calls["norm_gate"] += 1
        out = ref_norm_gate(f32(y)[0], f32(z)[0], f32(norm_weight), eps)
        return mx.array(out[None]).astype(y.dtype)

    monkeypatch.setattr(gdn_prefill, "runtime_supported", lambda: True)
    monkeypatch.setattr(gdn_prefill, "qwen4_gdn_prefill_prework", prework)
    monkeypatch.setattr(gdn_prefill, "qwen4_gdn_prefill_norm_gate", norm_gate)
    return calls


def _inputs(S, hidden=64, seed=0):
    return (0.5 * mx.random.normal((1, S, hidden), key=mx.random.key(seed))).astype(
        mx.bfloat16
    )


# --------------------------------------------------------------------------
# GDN prefill: the mirrors describe the eager semantics
# --------------------------------------------------------------------------


def _ulp_mismatch(a, b):
    """Fraction of elements that differ, and the max abs difference."""
    return float(np.mean(a != b)), float(np.max(np.abs(a - b)))


def test_prework_mirror_tracks_eager_path_on_cpu(layer):
    """The mirror reproduces eager conv/silu/L2/q-scale to bf16 tolerance.

    CPU reductions and exp differ from Metal's, so this is tolerance-level;
    exact identity is the GPU gate's job.
    """
    S = 64
    rng = np.random.default_rng(1)
    qkv = mx.array(rng.normal(size=(1, S, C)).astype(np.float32)).astype(mx.bfloat16)
    state = mx.array(rng.normal(size=(1, NKEEP, C)).astype(np.float32)).astype(
        mx.bfloat16
    )
    conv_input = mx.concatenate([state, qkv], axis=1)
    conv_out = nn.silu(layer.conv1d(conv_input))
    kd = HK * DK
    q, k, v = (
        conv_out[..., :kd].reshape(1, S, HK, DK),
        conv_out[..., kd : 2 * kd].reshape(1, S, HK, DK),
        conv_out[..., 2 * kd :].reshape(1, S, HV, DV),
    )
    q, k = layer._normalize_qk(q, k)
    rq, rk, rv, rstate = ref_prework(
        f32(qkv)[0],
        f32(state)[0],
        f32(layer.conv1d.weight)[:, :, 0],
        bf16(np.float32(DK**-0.5)),
    )
    # conv is exact on CPU too (fp32 tap accumulation, one rounding).
    x = f32(conv_input)[0]
    w = f32(layer.conv1d.weight)[:, :, 0]
    acc = np.zeros((S, C), np.float32)
    for tap in range(K):
        acc = acc + x[tap : tap + S] * w[None, :, tap]
    assert np.array_equal(bf16(acc), f32(layer.conv1d(conv_input))[0])
    # SiLU: CPU sigmoid is not Metal's formula; within a few bf16 ulps.
    np.testing.assert_allclose(rv, f32(v)[0], rtol=2**-5, atol=2**-14)
    # L2: the mirror is closer to exact math than the CPU eager chain, whose
    # bf16 reduction differs from the GPU row-reduce the kernel mirrors.
    act = f32(conv_out)[0].astype(np.float64)
    kd_ = HK * DK
    for name, ref, eager, block, scale in (
        ("q", rq, q, act[:, :kd_], DK**-0.5),
        ("k", rk, k, act[:, kd_ : 2 * kd_], 1.0),
    ):
        heads = block.reshape(S, HK, DK)
        exact = scale * heads / np.sqrt((heads**2).sum(-1, keepdims=True) + 1e-6)
        ref_err = np.abs(ref - exact).max()
        eager_err = np.abs(f32(eager)[0] - exact).max()
        assert ref_err <= 4 * 2**-8 * np.abs(exact).max(), (name, ref_err)
        assert ref_err <= eager_err + 1e-6, (name, ref_err, eager_err)
    # The next conv state is a pure copy: exact.
    assert np.array_equal(rstate, f32(conv_input[:, -NKEEP:])[0])


def test_norm_gate_mirror_tracks_eager_rmsnormgated_on_cpu(layer):
    S = 64
    rng = np.random.default_rng(2)
    y = mx.array(rng.normal(size=(1, S, HV, DV)).astype(np.float32)).astype(mx.bfloat16)
    z = mx.array(rng.normal(size=(1, S, HV * DV)).astype(np.float32)).astype(mx.bfloat16)
    eager = layer.norm(y, z.reshape(1, S, HV, DV)).reshape(1, S, -1)
    ref = ref_norm_gate(f32(y)[0], f32(z)[0], f32(layer.norm.weight), layer.norm.eps)
    frac, worst = _ulp_mismatch(ref, f32(eager)[0])
    assert frac < 0.05 and worst < 0.05, (frac, worst)


def test_prefill_sources_are_the_row_runtime_variants():
    prework = gdn_prefill.prefill_prework_source()
    assert "uint(S)" not in prework and "S_rt" in prework
    assert prework.startswith("    const uint S_rt = uint(s_len);")
    assert "simd_shuffle_xor" in prework  # the L2 reduction the mirror follows
    norm_gate = gdn_prefill.prefill_norm_gate_source()
    assert "threadgroup_position_in_grid.y * uint(HV)" in norm_gate


# --------------------------------------------------------------------------
# GDN prefill: admission
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
        gate_activation="sigmoid",
    )
    kw.update(overrides)
    return kw


def test_admission_accepts_flash_next_prefill():
    assert gdn_prefill.admit_qwen4_fused_gdn_prefill(**_admission_kwargs()).accepted
    warm = _admission_kwargs(
        S=2048,
        conv_state=mx.zeros((1, NKEEP, C), mx.bfloat16),
        recurrent_state=mx.zeros((1, HV, DV, DK), mx.float32),
    )
    assert gdn_prefill.admit_qwen4_fused_gdn_prefill(**warm).accepted


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (dict(training=True), "training"),
        (dict(sharded=True), "distributed sharding"),
        (dict(has_cache=False), "no cache"),
        (dict(speculating=True), "speculative rollback"),
        (dict(mask=mx.ones((1, 64), mx.bool_)), "masked prefill"),
        (dict(lengths=mx.array([64])), "ragged lengths"),
        (dict(left_padding=mx.array([0])), "left padding"),
        (dict(gate_activation="silu"), "output gate 'silu'"),
        (dict(num_value_heads=32), "unsupported geometry"),
        (dict(conv_kernel=3), "unsupported geometry"),
        (dict(qkv=mx.zeros((2, 64, C), mx.bfloat16)), "batch of 2 rows"),
        (dict(qkv=mx.zeros((1, 63, C), mx.bfloat16)), "rows 63 < 64"),
        (dict(qkv=mx.zeros((1, 64, C), mx.float16)), "qkv dtype"),
        (dict(norm_weight=mx.zeros((DV,), mx.float32)), "norm_weight dtype"),
        (dict(conv_state=mx.zeros((1, NKEEP, C), mx.float32)), "conv_state dtype"),
        (
            dict(recurrent_state=mx.zeros((1, HV, DV, DK), mx.bfloat16)),
            "recurrent_state must be float32",
        ),
        (dict(z=mx.zeros((1, 64, 10), mx.bfloat16)), "z shape"),
    ],
)
def test_admission_refuses_with_reason(overrides, reason):
    kw = _admission_kwargs()
    kw.update(overrides)
    admission = gdn_prefill.admit_qwen4_fused_gdn_prefill(**kw)
    assert not admission.accepted
    assert admission.reason.startswith(reason), admission.reason


def test_env_default_is_off(monkeypatch):
    monkeypatch.delenv(gdn_prefill.PREFILL_ENV, raising=False)
    assert gdn_prefill.prefill_enabled_from_env() is False
    monkeypatch.setenv(gdn_prefill.PREFILL_ENV, "0")
    assert gdn_prefill.prefill_enabled_from_env() is False
    monkeypatch.setenv(gdn_prefill.PREFILL_ENV, "1")
    assert gdn_prefill.prefill_enabled_from_env() is True


# --------------------------------------------------------------------------
# GDN prefill: wiring
# --------------------------------------------------------------------------


def _run(layer, chunks, mode):
    layer.set_fused_gdn_prefill_mode(mode)
    cache = ArraysCache(2)
    advanced = []
    original = cache.advance

    def spy(n):
        advanced.append(n)
        return original(n)

    cache.advance = spy
    outs = [layer(x, cache=cache) for x in chunks]
    mx.eval(outs, cache[0], cache[1])
    return outs, cache, advanced


def test_default_mode_is_stock_and_leaves_stats_unchanged(layer, kernels_on_cpu):
    assert layer.fused_gdn_prefill_mode == "stock"
    _run(layer, [_inputs(64)], "stock")
    assert kernels_on_cpu == {"prework": 0, "norm_gate": 0}
    assert layer.fused_gdn_prefill_calls == 0
    assert layer.fused_gdn_prefill_fallbacks == 0
    stats = qwen4_exp.qwen4_fused_gdn_stats(None, modules=[layer])
    assert "prefill_calls" not in stats


def test_fused_prefill_plumbing_cold_and_warm_chunks(layer, kernels_on_cpu):
    chunks = [_inputs(64, seed=1), _inputs(96, seed=2)]
    eager_outs, eager_cache, eager_adv = _run(layer, chunks, "stock")
    fused_outs, fused_cache, fused_adv = _run(layer, chunks, "fused")

    assert kernels_on_cpu == {"prework": 2, "norm_gate": 2}
    assert layer.fused_gdn_prefill_calls == 2
    assert layer.fused_gdn_prefill_fallbacks == 0
    assert fused_adv == eager_adv == [64, 96]
    # cache[0] is a raw-row copy: identical to the eager slice.
    assert fused_cache[0].dtype == mx.bfloat16
    assert np.array_equal(f32(fused_cache[0]), f32(eager_cache[0]))
    assert fused_cache[1].shape == eager_cache[1].shape == (1, HV, DV, DK)
    assert fused_cache[1].dtype == mx.float32
    for fused, eager in zip(fused_outs, eager_outs):
        assert fused.shape == eager.shape and fused.dtype == eager.dtype
        diff = np.abs(f32(fused) - f32(eager))
        scale = np.abs(f32(eager)).max()
        assert diff.max() <= 0.05 * scale + 1e-3, (diff.max(), scale)
    np.testing.assert_allclose(
        f32(fused_cache[1]), f32(eager_cache[1]), rtol=0.05, atol=0.05
    )
    stats = qwen4_exp.qwen4_fused_gdn_stats(None, modules=[layer])
    assert stats["prefill_calls"] == 2 and stats["prefill_fallbacks"] == 0


def test_short_or_masked_prefill_falls_back_with_reason(layer, kernels_on_cpu):
    outs_eager, cache_eager, _ = _run(layer, [_inputs(8, seed=4)], "stock")
    outs, cache, _ = _run(layer, [_inputs(8, seed=4)], "fused")
    assert kernels_on_cpu["prework"] == 0
    assert layer.fused_gdn_prefill_fallbacks == 1
    assert layer.fused_gdn_prefill_last_fallback == "rows 8 < 64"
    assert np.array_equal(f32(outs[0]), f32(outs_eager[0]))
    assert np.array_equal(f32(cache[0]), f32(cache_eager[0]))


def test_no_cache_prefill_falls_back(layer, kernels_on_cpu):
    layer.set_fused_gdn_prefill_mode("fused")
    layer(_inputs(64))
    assert kernels_on_cpu["prework"] == 0
    assert layer.fused_gdn_prefill_fallback_reasons == {"no cache": 1}


def test_silu_gate_layer_is_refused(kernels_on_cpu):
    gdn = qwen4_exp.GatedDeltaNet(_text_args(gate="silu"))
    gdn.set_dtype(mx.bfloat16)
    gdn.eval()
    gdn.set_fused_gdn_prefill_mode("fused")
    gdn(_inputs(64), cache=ArraysCache(2))
    assert gdn.fused_gdn_prefill_last_fallback == "output gate 'silu'"


def test_runtime_unavailable_falls_back(layer, kernels_on_cpu, monkeypatch):
    monkeypatch.setattr(gdn_prefill, "runtime_supported", lambda: False)
    _run(layer, [_inputs(64)], "fused")
    assert kernels_on_cpu["prework"] == 0
    assert layer.fused_gdn_prefill_last_fallback == "Metal runtime unavailable"


def test_dispatch_failure_leaves_cache_untouched(layer, kernels_on_cpu, monkeypatch):
    def boom(*_, **__):
        raise RuntimeError("no pipeline")

    monkeypatch.setattr(gdn_prefill, "qwen4_gdn_prefill_prework", boom)
    outs_eager, cache_eager, adv_eager = _run(layer, [_inputs(64, seed=5)], "stock")
    outs, cache, adv = _run(layer, [_inputs(64, seed=5)], "fused")
    assert layer.fused_gdn_prefill_last_fallback == (
        "Metal kernel dispatch failed: RuntimeError"
    )
    assert adv == adv_eager == [64]  # advanced once, by the eager path
    assert np.array_equal(f32(outs[0]), f32(outs_eager[0]))
    assert np.array_equal(f32(cache[1]), f32(cache_eager[1]))


def test_speculating_prefill_never_reaches_the_prefill_route(layer, kernels_on_cpu):
    layer.set_fused_gdn_prefill_mode("fused")
    cache = ArraysCache(2)
    cache.start_speculation()
    layer(_inputs(64), cache=cache)
    assert kernels_on_cpu["prework"] == 0
    assert layer.fused_gdn_prefill_calls == 0
    assert layer.fused_gdn_prefill_fallbacks == 0  # the verify route owns it


def test_stats_reset_clears_prefill_counters(layer, kernels_on_cpu):
    _run(layer, [_inputs(64)], "fused")
    qwen4_exp.qwen4_fused_gdn_stats(None, modules=[layer], reset=True)
    assert layer.fused_gdn_prefill_calls == 0
    assert layer.fused_gdn_prefill_fallback_reasons == {}


def test_mode_setter_validates(layer):
    with pytest.raises(ValueError):
        layer.set_fused_gdn_prefill_mode("fast")


# --------------------------------------------------------------------------
# MoE weighted sum
# --------------------------------------------------------------------------


def test_weighted_sum_mirror_matches_scatter_unsort_tail():
    """The kernel's contract, on CPU: equal to unsort + (x * scores).sum."""
    rng = np.random.default_rng(7)
    B, S, top_k, D = 1, 12, 10, 48
    indices = mx.array(rng.integers(0, 32, size=(B, S, top_k)).astype(np.uint32))
    x = mx.array(rng.normal(size=(B, S, 8)).astype(np.float32))
    x_sorted_in, idx, inv_order = qwen3_next._gather_sort(
        mx.expand_dims(x, (-2, -3)), indices
    )
    down = mx.array(rng.normal(size=(idx.size, 1, D)).astype(np.float32))
    scores = mx.array(rng.uniform(size=(B, S, top_k)).astype(np.float32))
    stock = (
        qwen3_next._scatter_unsort(down, inv_order, indices.shape).squeeze(-2)
        * scores[..., None]
    ).sum(axis=-2)
    ref = ref_weighted_sum(
        np.array(down), np.array(inv_order), np.array(scores).reshape(-1, top_k), False
    )
    np.testing.assert_allclose(ref.reshape(B, S, D), np.array(stock), rtol=1e-5, atol=1e-5)
    # bf16 scores: eager rounds each product to bf16 -- the ROUND_PRODUCT arm.
    down16 = down.astype(mx.bfloat16)
    scores16 = scores.astype(mx.bfloat16)
    stock16 = (
        qwen3_next._scatter_unsort(down16, inv_order, indices.shape).squeeze(-2)
        * scores16[..., None]
    ).sum(axis=-2)
    ref16 = bf16(
        ref_weighted_sum(
            f32(down16), np.array(inv_order), f32(scores16).reshape(-1, top_k), True
        )
    ).reshape(B, S, D)
    # CPU's bf16 reduction is not the GPU's; hold the mirror to exact math.
    exact = (
        f32(qwen3_next._scatter_unsort(down16, inv_order, indices.shape).squeeze(-2))
        .astype(np.float64)
        * f32(scores16)[..., None]
    ).sum(axis=-2)
    tol = 4 * 2**-8 * np.abs(exact).max()
    assert np.abs(ref16 - exact).max() <= tol
    assert np.abs(f32(stock16) - exact).max() <= 4 * tol


def _moe_args(top_k=10):
    return SimpleNamespace(
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=24,
        norm_topk_prob=True,
        num_experts=16,
        num_experts_per_tok=top_k,
    )


@pytest.fixture
def wsum_on_cpu(monkeypatch):
    calls = []

    def kernel(x_sorted, inv_order, scores):
        calls.append(tuple(x_sorted.shape))
        top_k = scores.shape[-1]
        round_product = scores.dtype == x_sorted.dtype and x_sorted.dtype != mx.float32
        out = ref_weighted_sum(
            f32(x_sorted),
            np.array(inv_order).astype(np.int64),
            f32(scores).reshape(-1, top_k),
            round_product,
        )
        dtype = x_sorted.dtype if scores.dtype == x_sorted.dtype else mx.float32
        return mx.array(out.reshape(*scores.shape[:-1], -1)).astype(dtype)

    monkeypatch.setattr(moe_wsum, "runtime_supported", lambda: True)
    monkeypatch.setattr(moe_wsum, "moe_weighted_sum", kernel)
    return calls


def _block(top_k=10):
    mx.random.seed(11)
    block = qwen3_next.Qwen3NextSparseMoeBlock(_moe_args(top_k))
    block.eval()
    return block


def test_moe_weighted_sum_default_off(monkeypatch, wsum_on_cpu):
    monkeypatch.delenv(moe_wsum.WEIGHTED_SUM_ENV, raising=False)
    assert moe_wsum.weighted_sum_enabled_from_env() is False
    block = _block()
    assert block.moe_weighted_sum is False
    block(mx.random.normal((1, 16, 32)))
    assert wsum_on_cpu == []
    assert block.switch_mlp.moe_weighted_sum_calls == 0


def test_moe_weighted_sum_plumbing_matches_stock(wsum_on_cpu):
    block = _block()
    x = mx.random.normal((1, 16, 32), key=mx.random.key(3))
    stock = block(x)
    block.set_moe_weighted_sum(True)
    fused = block(x)
    mx.eval(stock, fused)
    assert len(wsum_on_cpu) == 1
    # sorted rows, one per routed assignment: (tokens * top_k, 1, D)
    assert wsum_on_cpu[0] == (16 * 10, 1, 32)
    assert block.switch_mlp.moe_weighted_sum_calls == 1
    assert fused.shape == stock.shape and fused.dtype == stock.dtype
    np.testing.assert_allclose(np.array(fused), np.array(stock), rtol=1e-5, atol=1e-5)


def test_moe_weighted_sum_stock_expert_mode_route(wsum_on_cpu):
    """With the fused-expert kernel off, the block hands scores over itself."""
    block = _block()
    block.set_fused_expert_kernel_mode("stock")
    x = mx.random.normal((1, 16, 32), key=mx.random.key(5))
    stock = block(x)
    block.set_moe_weighted_sum(True)
    fused = block(x)
    assert len(wsum_on_cpu) == 1
    np.testing.assert_allclose(np.array(fused), np.array(stock), rtol=1e-5, atol=1e-5)


def test_moe_weighted_sum_skips_decode_widths(wsum_on_cpu):
    block = _block()
    block.set_moe_weighted_sum(True)
    block(mx.random.normal((1, 6, 32)))  # 60 routed rows < 64
    assert wsum_on_cpu == []
    assert block.switch_mlp.moe_weighted_sum_calls == 0
    # below the threshold the scores never reach the switch module
    assert block.switch_mlp.moe_weighted_sum_fallbacks == 0


def test_moe_weighted_sum_refuses_other_top_k(wsum_on_cpu):
    block = _block(top_k=8)
    block.set_moe_weighted_sum(True)
    stock_block = _block(top_k=8)
    x = mx.random.normal((1, 16, 32), key=mx.random.key(4))
    out = block(x)
    assert wsum_on_cpu == []
    assert block.switch_mlp.moe_weighted_sum_last_fallback == "top_k 8"
    np.testing.assert_array_equal(np.array(out), np.array(stock_block(x)))


def test_moe_weighted_sum_admission_reasons():
    idx = mx.zeros((1, 8, 10), mx.uint32)
    kw = dict(
        x_sorted=mx.zeros((80, 1, 32), mx.bfloat16),
        inv_order=mx.zeros((80,), mx.uint32),
        scores=mx.zeros((1, 8, 10), mx.bfloat16),
        indices=idx,
        do_sort=True,
        training=False,
    )
    assert moe_wsum.admit_moe_weighted_sum(**kw).accepted
    for overrides, reason in (
        (dict(do_sort=False), "unsorted gather"),
        (dict(scores=None), "no scores"),
        (dict(training=True), "training"),
        (dict(indices=mx.zeros((1, 6, 10), mx.uint32)), "routed rows 60 < 64"),
        (dict(scores=mx.zeros((1, 8, 10), mx.float16)), "scores dtype"),
        (dict(x_sorted=mx.zeros((80, 32), mx.bfloat16)), "x_sorted shape"),
    ):
        args = dict(kw, **overrides)
        admission = moe_wsum.admit_moe_weighted_sum(**args)
        assert not admission.accepted
        assert admission.reason.startswith(reason), admission.reason


def test_folded_shared_row_rejects_weighted_sum():
    block = _block()
    object.__setattr__(block, "shared_folded", True)
    with pytest.raises(ValueError):
        block.set_moe_weighted_sum(True)


def test_gpu_gate_script_refuses_without_ownership_flag(capsys):
    from scripts.check_omlx3903_port import main

    assert main([]) == 2
    assert "--i-own-the-gpu" in capsys.readouterr().err
