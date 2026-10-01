"""Real-Metal gate for the fp16 GDN state storage class (runtime/models/gdn_state.py).

Run under the GPU lock with ``MLX2_RUN_METAL_TESTS=1``.  Every check runs for
the fp32 class (the control: the proven route) and for fp16, and demands
storage-bit identity:

* the generic gated_delta kernels (packed, masked, vectorized-gate) in one
  multi-token launch equal the fp32 kernel run token by token with the state
  rounded to fp16 between tokens -- "fp32 math, fp16 store" -- so the class's
  law holds inside a launch and grouping never changes the stored bits;
* the fused Flash-Next decode (B=1) and batched decode equal the composed
  stock chain (conv1d + SiLU, direct L2 q/k, gated_delta_update, gated norm);
* the snapshot verify, compact-replay verify with template and dynamic
  reconstruction, and catch-up equal the stock block at widths 2..8 over a
  chain of blocks that alternately commit and roll back;
* the qwen4_exp layer: fused prefill (several segment sizes), decode and
  verify-with-rollback equal the same layer on its stock route;
* an fp16 state past 65504 is stored as inf (no clamp), so it reaches the
  non-finite log-probability guard.
"""

import os

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime.models import gated_delta as gd
from mlx2.runtime.models import gdn_state
from mlx2.runtime.models import qwen4_exp
from mlx2.runtime.models import qwen4_fused_gdn as fused_gdn
from mlx2.runtime.models import qwen4_fused_gdn_verify as fused_verify
from mlx2.runtime.models.cache import ArraysCache
from mlx2.runtime.models.gated_delta import gated_delta_update

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (under the GPU lock) for the real-Metal gate",
)

STATES = [mx.float32, mx.float16]
WIDTHS = list(range(2, 9))
BLOCKS = 6


@pytest.fixture
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _eq(x, y):
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    bits = mx.uint16 if x.dtype.size == 2 else mx.uint32
    exact = mx.array_equal(x.view(bits), y.view(bits))
    finite = mx.all(mx.isfinite(x)) & mx.all(mx.isfinite(y))
    mx.eval(exact, finite)
    return bool(exact.item()) and bool(finite.item())


def _sd(state_dtype):
    return None if state_dtype == mx.float32 else state_dtype


# --------------------------------------------------------------------------
# generic kernels: the class law inside one launch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["packed", "masked", "vector_gate"])
def test_generic_kernel_rounds_every_token(gpu, variant):
    B, T, Hk, Hv, Dk, Dv = 2, 7, 4, 8, 128, 64
    keys = mx.random.split(mx.random.key(7), 6)
    q = (mx.random.normal((B, T, Hk, Dk), key=keys[0]) * 0.1).astype(mx.bfloat16)
    k = (mx.random.normal((B, T, Hk, Dk), key=keys[1]) * 0.1).astype(mx.bfloat16)
    v = mx.random.normal((B, T, Hv, Dv), key=keys[2]).astype(mx.bfloat16)
    if variant == "vector_gate":
        g = mx.random.uniform(0.9, 1.0, (B, T, Hv, Dk), key=keys[3])
    else:
        g = mx.random.uniform(0.9, 1.0, (B, T, Hv), key=keys[3])
    beta = mx.random.uniform(0.0, 1.0, (B, T, Hv), key=keys[4])
    mask = None
    if variant == "masked":
        mask = mx.array([[True] * T, [False, False] + [True] * (T - 2)])
    state0 = (mx.random.normal((B, Hv, Dv, Dk), key=keys[5]) * 3).astype(mx.float16)

    y16, s16 = gd.gated_delta_kernel(q, k, v, g, beta, state0, mask)
    state = state0.astype(mx.float32)
    ys = []
    for t in range(T):
        y, state = gd.gated_delta_kernel(
            q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], g[:, t:t + 1],
            beta[:, t:t + 1], state, None if mask is None else mask[:, t:t + 1],
        )
        state = state.astype(mx.float16).astype(mx.float32)
        ys.append(y)
    y_ref = mx.concatenate(ys, axis=1)
    mx.eval(y16, s16, y_ref, state)
    assert s16.dtype == mx.float16
    assert _eq(s16, state.astype(mx.float16))
    assert _eq(y16, y_ref)


def test_fp16_state_overflow_stores_inf_not_a_clamp(gpu):
    B, T, Hk, Hv, Dk, Dv = 1, 1, 1, 1, 128, 8
    q = mx.ones((B, T, Hk, Dk), mx.float32) * 0.01
    k = mx.ones((B, T, Hk, Dk), mx.float32) * 0.1
    v = mx.full((B, T, Hv, Dv), 1.0e6)
    g = mx.ones((B, T, Hv))
    beta = mx.ones((B, T, Hv))
    state0 = mx.full((B, Hv, Dv, Dk), 60000.0).astype(mx.float16)
    _, s = gd.gated_delta_kernel(q, k, v, g, beta, state0)
    mx.eval(s)
    assert bool(mx.any(mx.isinf(s)).item())


# --------------------------------------------------------------------------
# fused Flash-Next kernels vs the composed stock chain
# --------------------------------------------------------------------------


def _stock_block(qkv, z, b, a, conv_state, conv_weight, A_log, dt_bias, norm_weight, state):
    steps = qkv.shape[1]
    keep = fused_gdn.CONV_KERNEL - 1
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    next_conv = mx.contiguous(conv_input[:, -keep:, :])
    convolved = nn.silu(mx.conv1d(conv_input, conv_weight, groups=fused_gdn.CONV_DIM))
    q, k, v = [
        t.reshape(qkv.shape[0], steps, h, d)
        for t, h, d in zip(
            mx.split(convolved, [fused_gdn.KEY_DIM, 2 * fused_gdn.KEY_DIM], axis=-1),
            [fused_gdn.NUM_KEY_HEADS, fused_gdn.NUM_KEY_HEADS, fused_gdn.NUM_VALUE_HEADS],
            [fused_gdn.KEY_HEAD_DIM, fused_gdn.KEY_HEAD_DIM, fused_gdn.VALUE_HEAD_DIM],
        )
    ]
    q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1.0e-6)
    k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1.0e-6)
    q = q * (fused_gdn.KEY_HEAD_DIM ** -0.5)

    def update(m):
        return gated_delta_update(
            q[:, :m], k[:, :m], v[:, :m], a[:, :m], b[:, :m], A_log, dt_bias,
            state, None, use_kernel=True, beta_input_dtype=True,
            state_dtype=_sd(state.dtype),
        )

    out, next_state = update(steps)
    restore_states = [update(m)[1] for m in range(1, steps)]
    restore_convs = [mx.contiguous(conv_input[:, m : m + keep, :]) for m in range(1, steps)]
    gate = mx.sigmoid(
        z.reshape(qkv.shape[0], steps, fused_gdn.NUM_VALUE_HEADS, -1).astype(mx.float32)
    )
    output = (mx.fast.rms_norm(out, norm_weight, 1.0e-6).astype(mx.float32) * gate).astype(qkv.dtype)
    return (output.reshape(qkv.shape[0], steps, fused_gdn.VALUE_DIM), next_conv, next_state,
            restore_states, restore_convs)


def _weights(dtype=mx.bfloat16):
    conv_weight = (mx.random.normal((fused_gdn.CONV_DIM, fused_gdn.CONV_KERNEL, 1),
                                    key=mx.random.key(1)) * 0.02).astype(dtype)
    A_log = (mx.random.normal((fused_gdn.NUM_VALUE_HEADS,), key=mx.random.key(2)) * 0.2).astype(mx.float32)
    dt_bias = (mx.random.normal((fused_gdn.NUM_VALUE_HEADS,), key=mx.random.key(3)) * 0.2).astype(dtype)
    norm_weight = (mx.random.normal((fused_gdn.VALUE_HEAD_DIM,), key=mx.random.key(4)) * 0.05 + 1).astype(dtype)
    return conv_weight, A_log, dt_bias, norm_weight


def _inputs(steps, block, salt, rows=1, dtype=mx.bfloat16):
    seed = salt * 1000 * steps + 10 * block + 7 * rows

    def draw(shape, offset):
        return (mx.random.normal(shape, key=mx.random.key(seed + offset)) * 0.6).astype(dtype)

    return (draw((rows, steps, fused_gdn.CONV_DIM), 1), draw((rows, steps, fused_gdn.VALUE_DIM), 2),
            draw((rows, steps, fused_gdn.NUM_VALUE_HEADS), 3), draw((rows, steps, fused_gdn.NUM_VALUE_HEADS), 4))


def _start(state_dtype, rows=1):
    """A nonzero starting state, so fp16 rounding is exercised from step one."""
    conv = mx.zeros((rows, fused_gdn.CONV_KERNEL - 1, fused_gdn.CONV_DIM), dtype=mx.bfloat16)
    state = (mx.random.normal((rows, fused_gdn.NUM_VALUE_HEADS, fused_gdn.VALUE_HEAD_DIM,
                               fused_gdn.KEY_HEAD_DIM), key=mx.random.key(99)) * 0.5)
    return conv, state.astype(mx.float16).astype(state_dtype)


def _probe_kw(state_dtype):
    return {} if state_dtype == mx.float32 else {"state_dtype": state_dtype}


@pytest.mark.parametrize("state_dtype", STATES)
def test_fused_decode_and_batch_decode_match_stock(gpu, state_dtype):
    ty = fused_gdn.probe_qwen4_fused_gdn_decode(mx.bfloat16, **_probe_kw(state_dtype))
    assert ty is not None
    w = _weights()
    (sc, ss), (fc, fs) = _start(state_dtype), _start(state_dtype)
    for step in range(12):
        qkv, z, b, a = _inputs(1, step, 5)
        stock = _stock_block(qkv, z, b, a, sc, w[0], w[1], w[2], w[3], ss)
        out = fused_gdn.qwen4_fused_gdn_decode(
            qkv, z, b, a, fc, w[0], w[1], w[2], fs, w[3], 1.0e-6, threadgroup_y=ty)
        mx.eval(*stock[:3], *out)
        assert out[2].dtype == state_dtype
        for i, name in enumerate(("output", "conv", "state")):
            assert _eq(stock[i], out[i]), (name, step)
        sc, ss, fc, fs = stock[1], stock[2], out[1], out[2]
    rows = 4
    (sc, ss), (fc, fs) = _start(state_dtype, rows), _start(state_dtype, rows)
    for step in range(6):
        qkv, z, b, a = _inputs(1, step, 6, rows=rows)
        stock = _stock_block(qkv, z, b, a, sc, w[0], w[1], w[2], w[3], ss)
        out = fused_gdn.qwen4_fused_gdn_batch_decode(
            qkv, z, b, a, fc, w[0], w[1], w[2], fs, w[3], 1.0e-6, threadgroup_y=ty)
        mx.eval(*stock[:3], *out)
        for i, name in enumerate(("output", "conv", "state")):
            assert _eq(stock[i], out[i]), ("batch", name, step)
        sc, ss, fc, fs = stock[1], stock[2], out[1], out[2]


@pytest.mark.parametrize("state_dtype", STATES)
@pytest.mark.parametrize("steps", WIDTHS)
def test_compact_replay_verify_and_every_rollback_match_stock(gpu, steps, state_dtype):
    kw = _probe_kw(state_dtype)
    ty = fused_verify.probe_qwen4_fused_gdn_replay_verify(mx.bfloat16, steps, **kw)
    dyn_ty = fused_verify.probe_qwen4_fused_gdn_replay_verify(
        mx.bfloat16, steps, dynamic_accept=True, **kw)
    assert ty is not None and dyn_ty is not None
    w = _weights()
    (sc, ss), (fc, fs) = _start(state_dtype), _start(state_dtype)
    for block in range(BLOCKS):
        qkv, z, b, a = _inputs(steps, block, 1)
        stock = _stock_block(qkv, z, b, a, sc, w[0], w[1], w[2], w[3], ss)
        out = fused_verify.qwen4_fused_gdn_replay_verify(
            qkv, z, b, a, fc, w[0], w[1], w[2], fs, w[3], 1.0e-6, threadgroup_y=ty)
        keys, corrections, decay = out[3:]
        mx.eval(*stock[:3], *stock[3], *stock[4], *out)
        assert out[2].dtype == state_dtype
        for i, name in enumerate(("output", "conv", "state")):
            assert _eq(stock[i], out[i]), (name, steps, block)
        for m in range(1, steps):
            rebuilt = fused_verify.qwen4_fused_gdn_reconstruct(
                fs, keys, corrections, decay, m, threadgroup_y=ty)
            assert _eq(stock[3][m - 1], rebuilt), ("reconstruct", steps, block, m)
        counts = mx.array([block % steps], dtype=mx.int32)
        dynamic = fused_verify.qwen4_fused_gdn_reconstruct(
            fs, keys, corrections, decay, counts, threadgroup_y=dyn_ty)
        expected = fs if block % steps == 0 else stock[3][block % steps - 1]
        assert _eq(expected, dynamic), ("dynamic", steps, block)
        keep = block % steps
        if keep == 0:
            sc, ss, fc, fs = stock[1], stock[2], out[1], out[2]
        else:
            restored_conv = mx.contiguous(mx.concatenate([fc, qkv], axis=1)[:, keep : keep + 3])
            assert _eq(stock[4][keep - 1], restored_conv)
            sc, ss = stock[4][keep - 1], stock[3][keep - 1]
            fc = restored_conv
            fs = fused_verify.qwen4_fused_gdn_reconstruct(
                fs, keys, corrections, decay, keep, threadgroup_y=ty)


@pytest.mark.parametrize("state_dtype", STATES)
@pytest.mark.parametrize("steps", WIDTHS)
def test_snapshot_verify_and_catchup_match_stock(gpu, steps, state_dtype):
    kw = _probe_kw(state_dtype)
    ty = fused_verify.probe_qwen4_fused_gdn_verify(mx.bfloat16, steps, **kw)
    cty = fused_verify.probe_qwen4_fused_gdn_catchup(mx.bfloat16, steps, **kw)
    assert ty is not None and cty is not None
    w = _weights()
    (sc, ss), (fc, fs) = _start(state_dtype), _start(state_dtype)
    for block in range(BLOCKS):
        qkv, z, b, a = _inputs(steps, block, 2)
        stock = _stock_block(qkv, z, b, a, sc, w[0], w[1], w[2], w[3], ss)
        out = fused_verify.qwen4_fused_gdn_verify(
            qkv, z, b, a, fc, w[0], w[1], w[2], fs, w[3], 1.0e-6, threadgroup_y=ty)
        catch = fused_verify.qwen4_fused_gdn_catchup(
            qkv, z, b, a, fc, w[0], w[1], w[2], fs, w[3], 1.0e-6, threadgroup_y=cty)
        mx.eval(*stock[:3], *stock[3], *stock[4], *out, *catch)
        assert out[3].dtype == state_dtype  # rollback snapshots in the state's class
        for i, name in enumerate(("output", "conv", "state")):
            assert _eq(stock[i], out[i]), (name, steps, block)
            assert _eq(stock[i], catch[i]), ("catchup", name, steps, block)
        for p in range(steps - 1):
            assert _eq(stock[3][p], out[3][:, p]), ("state point", steps, block, p)
            assert _eq(stock[4][p], out[4][:, p]), ("conv point", steps, block, p)
        keep = block % steps
        if keep == 0:
            sc, ss, fc, fs = stock[1], stock[2], out[1], out[2]
        else:
            sc, ss = stock[4][keep - 1], stock[3][keep - 1]
            fc, fs = out[4][:, keep - 1], out[3][:, keep - 1]


# --------------------------------------------------------------------------
# the qwen4_exp layer: fused routes vs its own stock route
# --------------------------------------------------------------------------

HK, HV, DK, DV, K = 16, 48, 128, 128, 4


def _layer(state_dtype):
    args = qwen4_exp.TextModelArgs(
        model_type="qwen4_exp_text", hidden_size=256, intermediate_size=0,
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
        output_gate_type="sigmoid",
        rope_parameters={"type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25},
    )
    mx.random.seed(3903)
    gdn = qwen4_exp.GatedDeltaNet(args)
    gdn.set_dtype(mx.bfloat16)
    gdn.A_log = gdn.A_log.astype(mx.float32)
    gdn.norm.weight = (1.0 + 0.1 * mx.random.normal((DV,))).astype(mx.bfloat16)
    gdn.conv1d.weight = (0.5 * mx.random.normal((2 * HK * DK + HV * DV, K, 1))).astype(mx.bfloat16)
    gdn.eval()
    mx.eval(gdn.parameters())
    if state_dtype == mx.float16:
        gdn_state.install_state_dtype(gdn, "float16")
    return gdn


def _modes(layer, fused):
    layer.set_fused_gdn_prefill_mode("fused" if fused else "stock")
    layer.set_fused_gdn_decode_mode("fused" if fused else "stock")
    layer.set_fused_gdn_verify_mode("fused" if fused else "stock")
    layer.set_fused_gdn_replay_rollback_mode("compact")


def _drive(layer, fused):
    """Prefill segments, decodes, verify blocks with rollback; returns outputs, cache."""
    _modes(layer, fused)
    x = lambda rows, seed: (0.5 * mx.random.normal((1, rows, 256), key=mx.random.key(seed))).astype(mx.bfloat16)  # noqa: E731
    cache = ArraysCache(2)
    outs = []
    for seed, rows in enumerate((100, 64, 300)):
        outs.append(layer(x(rows, seed), cache=cache))
    for step in range(4):
        outs.append(layer(x(1, 10 + step), cache=cache))
    for block, (width, keep) in enumerate(((4, 2), (8, 8), (3, 1), (6, 3))):
        cache.start_speculation()
        outs.append(layer(x(width, 20 + block), cache=cache))
        if keep < width:
            cache.trim(width - keep)
        cache.stop_speculation()
        outs.append(layer(x(1, 30 + block), cache=cache))
    mx.eval(*outs, cache[0], cache[1])
    return outs, cache


@pytest.mark.parametrize("state_dtype", STATES)
def test_layer_fused_routes_match_the_stock_route(gpu, state_dtype):
    layer = _layer(state_dtype)
    before = dict(gdn_state.STATS)
    stock_outs, stock_cache = _drive(layer, fused=False)
    fused_outs, fused_cache = _drive(layer, fused=True)
    assert stock_cache[1].dtype == fused_cache[1].dtype == state_dtype
    assert layer.fused_gdn_prefill_calls >= 3
    assert layer.fused_gdn_decode_calls >= 4
    assert layer.fused_gdn_verify_calls >= 4
    assert layer.fused_gdn_replay_rollback_calls >= 3
    for i, (s, f) in enumerate(zip(stock_outs, fused_outs)):
        assert _eq(s, f), ("output", i)
    assert _eq(stock_cache[0], fused_cache[0])
    assert _eq(stock_cache[1], fused_cache[1])
    if state_dtype == mx.float16:
        after = gdn_state.STATS
        for counter in ("kernel_launches", "fused_decode", "fused_replay_verify",
                        "fused_reconstruct"):
            assert after[counter] > before.get(counter, 0), counter
