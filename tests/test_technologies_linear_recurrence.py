"""CPU parity of the linear-recurrence technology against each model's old copy.

Every ``_legacy_*`` function is the model's pre-migration code, verbatim
(names adjusted only). Equality is exact (``mx.array_equal``).
"""
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.runtime.models import differentiable
from mlx2.runtime.models import gated_delta as gd
from mlx2.runtime.models.precise_ops import gate_sigmoid
from mlx2.runtime.technologies import linear_recurrence as lr

DTYPES = (mx.float32, mx.bfloat16, mx.float16)


def _eq(a, b):
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _eq(x, y)
        return
    assert a.dtype == b.dtype and a.shape == b.shape
    assert mx.array_equal(a, b, equal_nan=True).item()


def _rand(shape, dtype, scale=1.0, key=0):
    x = (mx.random.normal(shape, key=mx.random.key(key)) * scale).astype(dtype)
    # Edge cases: exact zeros and repeated values (ties) in every input.
    flat = x.reshape(-1)
    n = flat.shape[0]
    flat = mx.where(mx.arange(n) % 7 == 0, mx.zeros_like(flat), flat)
    flat = mx.where(mx.arange(n) % 11 == 0, mx.ones_like(flat), flat)
    return flat.reshape(shape)


# --------------------------------------------------------------------------
# legacy copies (verbatim)


def _legacy_kimi_l2_qk(q, k):
    # FLA normalizes in fp32, epsilon on SUM, with no bf16 round before scan.
    q, k = q.astype(mx.float32), k.astype(mx.float32)
    return (q * mx.rsqrt(mx.sum(q * q, -1, keepdims=True) + 1e-6) * q.shape[-1]**-0.5,
            k * mx.rsqrt(mx.sum(k * k, -1, keepdims=True) + 1e-6))


@mx.compile
def _legacy_kimi_kda_gates(a, beta, a_log, dt_bias):
    shifted = a.astype(mx.float32) + dt_bias.astype(mx.float32)
    softplus = mx.maximum(shifted, 0) + mx.log1p(mx.exp(-mx.abs(shifted)))
    return (-mx.exp(a_log.astype(mx.float32)) * softplus,
            mx.sigmoid(beta.astype(mx.float32)))


def _legacy_qwen4_exp_normalize_qk(q, k):
    q = q * mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-06)
    k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-06)
    return (q * k.shape[-1] ** (-0.5), k)


def _legacy_d3_normalize_qk(q, k):
    q, k = q.astype(mx.float32), k.astype(mx.float32)
    q = q * mx.rsqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    return q * (q.shape[-1] ** -0.5), k


def _legacy_qwen35_normalize_qk(q, k):
    inv_scale = k.shape[-1] ** (-0.5)
    eps = 1e-06 * inv_scale**2
    q = inv_scale**2 * mx.fast.rms_norm(q, None, eps)
    k = inv_scale * mx.fast.rms_norm(k, None, eps)
    return (q, k)


def _legacy_ling_qk(q, k, head_dim):
    scale = float(head_dim) ** -0.5
    q = (scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = scale * mx.fast.rms_norm(k, None, 1e-6)
    return q, k


def _legacy_glm53_qk(q, k, dim):
    q = q / mx.maximum(mx.sqrt((q * q).sum(-1, keepdims=True)), 1e-6) * dim**-0.5
    k = k / mx.maximum(mx.sqrt((k * k).sum(-1, keepdims=True)), 1e-6)
    return q, k


def _legacy_raon_l2_normalize(x):
    x = x.astype(mx.float32)
    return x / mx.maximum(mx.sqrt(mx.sum(x * x, -1, keepdims=True)), 1e-6)


def _legacy_raon_qk(q, k, dtype, d):
    q = _legacy_raon_l2_normalize(q).astype(dtype) * (d ** -0.5)
    k = _legacy_raon_l2_normalize(k).astype(dtype)
    return q, k


def _legacy_k31_qk(q, k, d):
    q = q * mx.rsqrt(mx.sum(q * q, -1, keepdims=True) + 1e-6) * d ** -0.5
    k = k * mx.rsqrt(mx.sum(k * k, -1, keepdims=True) + 1e-6)
    return q, k


def _legacy_raon_rda_gates_src(decay_raw, dt_bias, a_exp, beta_raw, write_raw, lower_bound):
    scaled = a_exp * (decay_raw.astype(mx.float32) + dt_bias.astype(mx.float32))
    log_alpha = lower_bound * mx.sigmoid(scaled)
    b32 = beta_raw.astype(mx.float32)
    beta = mx.sigmoid(b32)
    ratio = mx.exp(nn.log_sigmoid(b32 + write_raw.astype(mx.float32)) - nn.log_sigmoid(b32))
    return mx.exp(log_alpha), beta, ratio


_legacy_raon_rda_gates = mx.compile(_legacy_raon_rda_gates_src, shapeless=True)


def _legacy_glm53_gates(a, beta, stored_a_h1, dt_bias_hd):
    a = a + dt_bias_hd
    decay = mx.exp(-5 * gate_sigmoid(a * stored_a_h1))
    beta = gate_sigmoid(beta)
    return decay, beta


def _legacy_compute_lower_bound_g_src(A_log, a, dt_bias, lower_bound):
    return mx.exp(
        lower_bound
        * mx.sigmoid(mx.exp(A_log.astype(mx.float32)) * (a.astype(mx.float32) + dt_bias))
    )


def _legacy_compute_g_src(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


def _legacy_compute_g_f32_src(A_log, a, dt_bias):
    a = a.astype(mx.float32)
    dt_bias = dt_bias.astype(mx.float32)
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


_legacy_compute_g = mx.compile(_legacy_compute_g_src, shapeless=True)
_legacy_compute_g_f32 = mx.compile(_legacy_compute_g_f32_src, shapeless=True)
_legacy_compute_lower_bound_g = mx.compile(_legacy_compute_lower_bound_g_src, shapeless=True)


def _legacy_ling_short_conv(conv, kernel_size, x, state, mask, lengths):
    if mask is not None:
        x = mx.where(mask[..., None], x, 0)
    if state is None:
        state = mx.zeros((x.shape[0], kernel_size - 1, x.shape[-1]), dtype=x.dtype)
    conv_input = mx.concatenate([state, x], axis=1)
    output = nn.silu(conv(conv_input))
    n_keep = kernel_size - 1
    if lengths is not None:
        ends = mx.clip(lengths, 0, x.shape[1])
        positions = (ends[:, None] + mx.arange(n_keep))[..., None]
        new_state = mx.take_along_axis(conv_input, positions, axis=1)
    else:
        new_state = mx.contiguous(conv_input[:, -n_keep:, :])
    return output, new_state


def _legacy_raon_short_conv(weight, kernel_size, x, state, mask, lengths):
    if mask is not None:
        x = mx.where(mask[..., None], x, 0)
    if state is None:
        state = mx.zeros((x.shape[0], kernel_size - 1, x.shape[-1]), dtype=x.dtype)
    conv_input = mx.concatenate([state, x], axis=1)
    out = mx.conv1d(conv_input.astype(mx.float32), weight.astype(mx.float32),
                    groups=x.shape[-1])
    output = nn.silu(out).astype(x.dtype)
    keep = kernel_size - 1
    if lengths is not None:
        ends = mx.clip(lengths, 0, x.shape[1])
        positions = (ends[:, None] + mx.arange(keep))[..., None]
        new_state = mx.take_along_axis(conv_input, positions, axis=1)
    else:
        new_state = mx.contiguous(conv_input[:, -keep:, :])
    return output, new_state


def _legacy_qwen35_conv(qkv, conv_state, mask, lengths, kernel, S):
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    n_keep = kernel - 1
    if lengths is not None:
        ends = mx.clip(lengths, 0, S)
        positions = (ends[:, None] + mx.arange(n_keep))[..., None]
        tail = mx.take_along_axis(conv_input, positions, axis=1)
    else:
        tail = mx.contiguous(conv_input[:, -n_keep:, :])
    return conv_input, tail


def _legacy_qwen36_pick(lengths, counts, steps, rows, snaps, before, final):
    ends = mx.broadcast_to(mx.array(lengths, mx.int32).reshape(-1), (rows,))
    ends = mx.minimum(mx.maximum(ends, 0), mx.array(counts, mx.int32))
    idx = mx.minimum(mx.maximum(ends - 1, 0), steps - 2)
    index = idx.reshape((rows, 1) + (1,) * (snaps.ndim - 2))
    selected = mx.take_along_axis(snaps, index, axis=1).squeeze(1)
    full = (ends == mx.array(counts)).reshape((rows,) + (1,) * (before.ndim - 1))
    zero = (ends == 0).reshape((rows,) + (1,) * (before.ndim - 1))
    return mx.contiguous(mx.where(zero, before, mx.where(full, final, selected)))


def _legacy_qwen38_pick(lengths, row_steps, rows, snapshots, before, final):
    if isinstance(lengths, mx.array):
        lengths = lengths.tolist()
    values = list(lengths)
    if len(values) == 1 and rows > 1:
        values *= rows
    if len(values) != rows:
        raise ValueError('batched GDN rollback length count changed')
    ends = [min(max(int(value), 0), int(count)) for value, count in zip(values, row_steps)]
    selected = []
    for lane, (end, count) in enumerate(zip(ends, row_steps)):
        if end == 0:
            value = before[lane:lane + 1]
        elif end == count:
            value = final[lane:lane + 1]
        else:
            value = snapshots[lane:lane + 1, end - 1]
        selected.append(value)
    return mx.contiguous(mx.concatenate(selected, axis=0))


def _legacy_tree_cohorts(parents, levels, initial, operands, step):
    width = len(parents)
    outputs, states = [None] * width, [None] * width
    for level in levels:
        indices = mx.array(level, dtype=mx.int32)
        parent_states = mx.concatenate([
            initial if parents[row] < 0 else states[parents[row]] for row in level
        ])
        output, state = step(*(mx.take(value, indices, axis=0) for value in operands),
                             parent_states)
        for index, row in enumerate(level):
            outputs[row], states[row] = output[index:index + 1], state[index:index + 1]
    return outputs, states


def _legacy_gated_delta_update_reference(q, k, v, a, b, A_log, dt_bias, state=None, mask=None,
                                         lower_bound=None, beta_input_dtype=False,
                                         decay_input_float32=False, state_dtype=None):
    beta = gate_sigmoid(b) if beta_input_dtype else gate_sigmoid(b.astype(mx.float32))
    if lower_bound is None:
        g = (_legacy_compute_g_f32(A_log, a, dt_bias) if decay_input_float32
             else _legacy_compute_g(A_log, a, dt_bias))
    else:
        g = _legacy_compute_lower_bound_g(A_log, a, dt_bias, lower_bound)
    if state is None:
        (B, _, Hk, Dk) = q.shape
        (Hv, Dv) = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=state_dtype or mx.float32)
    return lr.gated_delta_ops(q, k, v, g, beta, state, mask)


# --------------------------------------------------------------------------
# q/k laws


@pytest.mark.parametrize("dtype", DTYPES)
def test_qk_laws_match_every_model_copy(dtype):
    shape = (2, 5, 3, 32)
    q, k = _rand(shape, dtype, 0.5, 1), _rand(shape, dtype, 0.5, 2)
    q = q.at[0, 0].multiply(0)  # an all-zero row (epsilon branch)
    k = k.at[1, 2].multiply(0)
    spec = lr.QWEN35_GDN
    _eq(lr.normalize_qk(spec, q, k), _legacy_qwen35_normalize_qk(q, k))
    _eq(lr.normalize_qk(lr.replace(spec, qk_norm="l2_rsqrt"), q, k),
        _legacy_qwen4_exp_normalize_qk(q, k))
    _eq(lr.normalize_qk(lr.replace(spec, qk_norm="l2_rsqrt", qk_norm_dtype="float32"), q, k),
        _legacy_d3_normalize_qk(q, k))
    from mlx2.runtime.models import kimi_linear
    _eq(kimi_linear.l2_qk(q, k), _legacy_kimi_l2_qk(q, k))
    _eq(lr.normalize_qk(lr.replace(spec, qk_norm="rms_mean"), q, k), _legacy_ling_qk(q, k, 32))
    from mlx2.runtime.models.glm53_gguf import GLM53_KDA
    _eq(lr.normalize_qk(GLM53_KDA, q, k), _legacy_glm53_qk(q, k, 32))
    from mlx2.runtime.models.raon1_mini import RAON_RDA_LAWS
    raon = lr.LinearRecurrenceSpec(num_k_heads=1, num_v_heads=1, lower_bound=-5.0, **RAON_RDA_LAWS)
    _eq(lr.normalize_qk(raon, q, k), _legacy_raon_qk(q, k, dtype, 32))
    from mlx2.runtime.models.kimi_k31 import _K31_QK
    q32, k32 = q.astype(mx.float32), k.astype(mx.float32)
    _eq(lr.normalize_qk(_K31_QK, q32, k32), _legacy_k31_qk(q32, k32, 32))


def test_qwen_layer_classes_carry_their_laws():
    from mlx2.decisions.candidates.d3_model import D3GatedDeltaNet
    from mlx2.runtime.models import qwen3_5, qwen4_exp

    assert qwen3_5.GatedDeltaNet.recurrence_spec.qk_norm == "rms_folded"
    assert lr.gdn_update_flags(qwen3_5.GatedDeltaNet.recurrence_spec) == {}
    q4 = qwen4_exp.GatedDeltaNet.recurrence_spec
    assert (q4.qk_norm, q4.qk_norm_dtype) == ("l2_rsqrt", "input")
    assert lr.gdn_update_flags(q4) == {"beta_input_dtype": True, "decay_input_float32": True}
    d3 = D3GatedDeltaNet.recurrence_spec
    assert (d3.qk_norm, d3.qk_norm_dtype) == ("l2_rsqrt", "float32")
    q = _rand((1, 2, 2, 16), mx.bfloat16, key=3)
    _eq(qwen4_exp.GatedDeltaNet._normalize_qk(SimpleNamespace(recurrence_spec=q4), q, q),
        _legacy_qwen4_exp_normalize_qk(q, q))


# --------------------------------------------------------------------------
# gate laws


@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_gate_laws_match_gated_delta_update_copy(dtype):
    B, T, Hk, Hv, D = 2, 4, 1, 2, 32
    q, k = _rand((B, T, Hk, D), dtype, 0.3, 1), _rand((B, T, Hk, D), dtype, 0.3, 2)
    v = _rand((B, T, Hv, D), dtype, 1, 3)
    a, b = _rand((B, T, Hv), dtype, 2, 4), _rand((B, T, Hv), dtype, 2, 5)
    A_log, dt = _rand((Hv,), mx.float32, 1, 6), _rand((Hv,), dtype, 1, 7)
    mask = mx.array([[True] * T, [False, True, True, True]])
    for kw in ({}, {"beta_input_dtype": True}, {"decay_input_float32": True},
               {"lower_bound": -5.0}, {"beta_input_dtype": True, "decay_input_float32": True}):
        for m in (None, mask):
            got = gd.gated_delta_update(q, k, v, a, b, A_log, dt, None, m, **kw)
            _eq(got, _legacy_gated_delta_update_reference(q, k, v, a, b, A_log, dt, None, m, **kw))


@pytest.mark.parametrize("dtype", DTYPES)
def test_kda_gate_laws_match_model_copies(dtype):
    b, t, h, d = 2, 3, 2, 16
    a = _rand((b, t, h, d), dtype, 3, 1)
    beta = _rand((b, t, h), dtype, 3, 2)
    a_log = _rand((h, 1), mx.float32, 1, 3)
    dt = _rand((h, d), dtype, 1, 4)
    from mlx2.runtime.models import kimi_linear
    kimi = lr.LinearRecurrenceSpec(num_k_heads=h, num_v_heads=h, head_k_dim=d, head_v_dim=d,
                                   **kimi_linear.KIMI_KDA_LAWS)
    got = lr.gates(kimi, a, beta, a_log, dt)
    log_decay, ref_beta = _legacy_kimi_kda_gates(a, beta, a_log, dt)
    _eq((got.decay, got.beta, got.log_decay), (mx.exp(log_decay), ref_beta, log_decay))
    # Raon RDA (resident fp32 lower bound array).
    from mlx2.runtime.models.raon1_mini import RAON_RDA_LAWS
    raon = lr.LinearRecurrenceSpec(num_k_heads=h, num_v_heads=h, lower_bound=-5.0, **RAON_RDA_LAWS)
    write = _rand((b, t, h), dtype, 2, 5)
    a_exp = mx.exp(a_log.astype(mx.float32))
    bound = mx.array(-5.0, mx.float32)
    got = lr.gates(raon, a, beta, a_exp, dt, write=write, lower_bound=bound)
    _eq((got.decay, got.beta, got.value_ratio),
        _legacy_raon_rda_gates(a, dt, a_exp, beta, write, bound))
    # GLM-5.3: stored A, gate_sigmoid, int -5.
    from mlx2.runtime.models.glm53_gguf import GLM53_KDA
    stored = _rand((h, 1), dtype, 1, 6)
    _eq(lr.gates(GLM53_KDA, a, beta, stored, dt)[:2], _legacy_glm53_gates(a, beta, stored, dt))
    # Kimi K3.1: lower-bounded sigmoid, fp32 sigmoid beta.
    dt32 = dt.astype(mx.float32)
    _eq(lr.decay("lower_bound_sigmoid", a_log, a, dt32, -5.0),
        _legacy_compute_lower_bound_g(a_log, a, dt32, -5.0))
    _eq(lr.beta("sigmoid_float32", beta), mx.sigmoid(beta.astype(mx.float32)))


# --------------------------------------------------------------------------
# short conv


@pytest.mark.parametrize("dtype", DTYPES)
def test_short_conv_matches_ling_raon_and_qwen_copies(dtype):
    B, T, C, K = 2, 5, 8, 4
    x = _rand((B, T, C), dtype, 1, 1)
    conv = nn.Conv1d(C, C, K, bias=False, groups=C)
    conv.weight = _rand(conv.weight.shape, dtype, 0.5, 2)
    mask = mx.array([[True] * T, [False, False, True, True, True]])
    lengths = mx.array([5, 3])
    for state in (None, _rand((B, K - 1, C), dtype, 1, 3)):
        for m in (None, mask):
            for ln in (None, lengths):
                _eq(lr.short_conv(x, state, m, ln, conv, kernel_size=K),
                    _legacy_ling_short_conv(conv, K, x, state, m, ln))
                _eq(lr.short_conv(x, state, m, ln, None, kernel_size=K, precision="float32",
                                  weight=conv.weight),
                    _legacy_raon_short_conv(conv.weight, K, x, state, m, ln))
                cs = state if state is not None else mx.zeros((B, K - 1, C), dtype)
                window = lr.conv_window(x, cs, m, K)
                _eq((window, lr.conv_tail(window, ln, K - 1, T)),
                    _legacy_qwen35_conv(x, cs, m, ln, K, T))


# --------------------------------------------------------------------------
# dispatcher selection (CPU: kernels stubbed, never launched)


def _operands(B=1, T=3, H=2, D=32, vector=True):
    q = _rand((B, T, H, D), mx.float32, 0.2, 1)
    v = _rand((B, T, H, D), mx.float32, 1, 2)
    g = mx.full((B, T, H, D) if vector else (B, T, H), 0.9)
    beta = mx.full((B, T, H), 0.5)
    state = mx.zeros((B, H, D, D))
    return q, q, v, g, beta, state


def _fake_gpu(monkeypatch, calls):
    monkeypatch.setattr(lr.mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(lr.mx.metal, "is_available", lambda: True)

    def kernel(q, k, v, g, beta, state, mask=None):
        calls.append("kernel")
        return lr.gated_delta_ops(q, k, v, g, beta, state, mask)

    monkeypatch.setattr(gd, "gated_delta_kernel", kernel)


def _legacy_kda_choice(reference, gpu, chunk_wy, chunk32, chunked):
    """Kimi/Raon's old branch order."""
    if chunk_wy and chunked:
        return "chunk_wy"
    if chunk32 and chunked:
        return "chunk32"
    if not reference and gpu:
        return "step_kernel"
    return "reference"


@pytest.mark.parametrize("gpu", (False, True))
def test_kda_dispatch_order_matches_kimi_and_raon(monkeypatch, gpu):
    calls = []
    if gpu:
        _fake_gpu(monkeypatch, calls)
    q, k, v, g, beta, state = _operands(H=4, D=128)
    for reference in (False, True):
        for wy in (False, True):
            for c32 in (False, True):
                for chunked in (False, True):
                    chunk = ("wy" if wy else "chunk32" if c32 else None) if chunked else None
                    _, receipt = lr.RECURRENCE.select(
                        None, q=q, k=k, v=v, g=g, beta=beta, state=state, mask=None,
                        kernels=None if (not reference and gpu) else "kernels_disallowed",
                        chunk=chunk, log_decay=mx.log(g), allow_core=False, prefill_chunk_size=0)
                    assert receipt.selected == _legacy_kda_choice(reference, gpu, wy, c32, chunked)


def test_guard_routes_every_kernel_site_to_the_reference(monkeypatch):
    calls = []
    _fake_gpu(monkeypatch, calls)
    q, k, v, g, beta, state = _operands()
    out, st, receipt = lr.recurrence(q, k, v, g, beta, state, allow_core=False, with_receipt=True)
    assert receipt.selected == "step_kernel" and calls == ["kernel"]
    differentiable.reset_reached()
    with differentiable.reference_path():
        ref, ref_state, receipt = lr.recurrence(q, k, v, g, beta, state, allow_core=False,
                                                with_receipt=True)
    assert receipt.selected == "reference" and calls == ["kernel"]
    assert differentiable.reached() == {"gated_delta": 1}
    _eq((out, st), (ref, ref_state))
    # use_kernel=False never reaches (counts) the guard: legacy short circuit.
    differentiable.reset_reached()
    with differentiable.reference_path():
        lr.recurrence(q, k, v, g, beta, state, use_kernel=False)
    assert differentiable.reached() == {}


def test_prefill_scan_refusals_keep_their_stats_keys(monkeypatch):
    calls = []
    _fake_gpu(monkeypatch, calls)
    q, k, v, g, beta, state = _operands(T=3, vector=False)
    stats = Counter()
    _, _, receipt = lr.recurrence(q, k, v, g, beta, state, prefill_chunk_size=8,
                                  prefill_stats=stats, allow_core=False, with_receipt=True)
    assert receipt.selected == "step_kernel"
    reason = gd._prefill_scan_refusal(q, k, v, g, beta, state, None, 8)
    assert stats == Counter({"fallback_" + reason: 1})
    # The module entry still counts exactly once per refusal.
    stats2 = Counter()
    assert gd._chunked_prefill(q, k, v, g, beta, state, None, 8, stats2) is None
    assert stats2 == stats
    with pytest.raises(ValueError):
        lr.recurrence(q, k, v, g, beta, state, prefill_chunk_size=4)


# --------------------------------------------------------------------------
# rollback selection and tree cohorts


@pytest.mark.parametrize("dtype", (mx.float32, mx.bfloat16))
def test_rollback_selection_matches_qwen36_and_qwen38(dtype):
    rows, steps = 3, 5
    counts = (5, 3, 1)
    snaps = _rand((rows, steps - 1, 2, 4, 4), dtype, 1, 1)
    before = _rand((rows, 2, 4, 4), dtype, 1, 2)
    final = _rand((rows, 2, 4, 4), dtype, 1, 3)
    for lengths in ([0, 0, 0], [5, 3, 1], [2, 1, 0], [9, -1, 1], [3], mx.array([4, 2, 1])):
        got = lr.select_rollback_rows(lengths, counts, snaps, before, final)
        _eq(got, _legacy_qwen38_pick(lengths, counts, rows, snaps, before, final))
        if not isinstance(lengths, list) or len(lengths) == rows:
            _eq(got, _legacy_qwen36_pick(lengths, counts, steps, rows, snaps, before, final))
    with pytest.raises(ValueError):
        lr.select_rollback_rows([1, 2], counts, snaps, before, final)


def test_tree_cohorts_match_ling_and_kimi_loop():
    parents = (-1, 0, 0, 1, 1, 2)
    depths = [0, 1, 1, 2, 2, 2]
    levels = [[r for r, dd in enumerate(depths) if dd == lv] for lv in range(3)]
    width, h, d = len(parents), 2, 32
    q = _rand((width, 1, h, d), mx.float32, 0.2, 1)
    v = _rand((width, 1, h, d), mx.float32, 1, 2)
    g = mx.full((width, 1, h, d), 0.8)
    beta = mx.full((width, 1, h), 0.5)
    initial = _rand((1, h, d, d), mx.float32, 0.1, 3)

    def step(q, k, v, g, beta, parent):
        return lr.gated_delta_ops(q, k, v, g, beta, parent)

    got = lr.cohort_tree_recurrence(parents, levels, initial, (q, q, v, g, beta), step)
    ref = _legacy_tree_cohorts(parents, levels, initial, (q, q, v, g, beta), step)
    _eq(got[0], ref[0])
    _eq(got[1], ref[1])


# --------------------------------------------------------------------------
# spec derivation


def test_from_config_derives_geometry_and_gate():
    qwen = lr.LinearRecurrenceSpec.from_config({
        "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4})
    assert (qwen.num_k_heads, qwen.num_v_heads, qwen.gate_kind, qwen.conv_layout,
            qwen.decay_law) == (16, 48, "scalar", "packed_qkv", "softplus")
    kimi = lr.LinearRecurrenceSpec.from_config({
        "linear_attn_config": {"num_heads": 32, "head_dim": 128, "short_conv_kernel_size": 4}})
    assert (kimi.num_v_heads, kimi.gate_kind, kimi.decay_law, kimi.beta_law) == (
        32, "vector", "log_softplus", "sigmoid_float32")
    ling = lr.LinearRecurrenceSpec.from_config({
        "num_attention_heads": 16, "head_dim": 128, "short_conv_kernel_size": 4,
        "kda_lower_bound": -5.0})
    assert (ling.num_k_heads, ling.decay_law, ling.lower_bound) == (16, "lower_bound_sigmoid", -5.0)
    unbounded = lr.LinearRecurrenceSpec.from_config({
        "num_attention_heads": 8, "head_dim": 64, "short_conv_kernel_size": 4})
    assert unbounded.decay_law == "softplus"

    @dataclass(frozen=True)
    class K31:
        kda_heads: int = 8
        kda_head_dim: int = 64
        conv_kernel: int = 4
        gate_lower_bound: float = -5.0

    k31 = lr.LinearRecurrenceSpec.from_config(K31())
    assert (k31.num_v_heads, k31.head_k_dim, k31.lower_bound, k31.gate_kind) == (8, 64, -5.0, "vector")
    with pytest.raises(ValueError):
        lr.LinearRecurrenceSpec(decay_law="rda")
    with pytest.raises(ValueError):
        lr.LinearRecurrenceSpec(num_k_heads=3, num_v_heads=4)


# --------------------------------------------------------------------------
# GDN selection order (scalar gate): prefill_scan -> core_scan -> step_kernel


def _gdn_stubs(monkeypatch, *, scan_refusal, core_admits):
    calls = []
    _fake_gpu(monkeypatch, calls)
    monkeypatch.setattr(gd, "_prefill_scan_refusal", lambda *a: scan_refusal)

    def scan(q, k, v, g, beta, state, chunk, stats, rows):
        calls.append("prefill_scan")
        return lr.gated_delta_ops(q, k, v, g, beta, state)

    def core(q, k, v, g, beta, *, initial_state, **kw):
        calls.append("core_scan")
        return lr.gated_delta_ops(q, k, v, g, beta, initial_state)

    monkeypatch.setattr(gd, "_prefill_scan_run", scan)
    monkeypatch.setattr(gd, "_can_use_core_gated_delta", lambda *a: core_admits)
    monkeypatch.setattr(gd, "_core_gated_delta_update", core)
    return calls


def test_gdn_order_prefill_scan_admitted(monkeypatch):
    calls = _gdn_stubs(monkeypatch, scan_refusal=None, core_admits=True)
    q, k, v, g, beta, state = _operands(T=20, vector=False)
    stats = Counter()
    _, _, receipt = lr.recurrence(q, k, v, g, beta, state, prefill_chunk_size=8,
                                  prefill_stats=stats, with_receipt=True)
    assert receipt.selected == "prefill_scan" and calls == ["prefill_scan"]
    assert set(receipt.refusals) == {"chunk_wy", "chunk32"} and not stats


def test_gdn_order_prefill_scan_refused_then_core_scan(monkeypatch):
    calls = _gdn_stubs(monkeypatch, scan_refusal="unsupported_layout", core_admits=True)
    q, k, v, g, beta, state = _operands(T=20, vector=False)
    stats = Counter()
    before = gd._GDN_CORE_CALLS
    _, _, receipt = lr.recurrence(q, k, v, g, beta, state, prefill_chunk_size=8,
                                  prefill_stats=stats, with_receipt=True)
    assert receipt.selected == "core_scan" and calls == ["core_scan"]
    assert receipt.refusals["prefill_scan"] == "unsupported_layout"
    assert stats == Counter({"fallback_unsupported_layout": 1})
    assert gd._GDN_CORE_CALLS == before + 1


@pytest.mark.parametrize("allow_core,core_admits,reason", [
    (True, False, "core_refused"), (False, True, "not_allowed")])
def test_gdn_order_core_refused_or_disallowed_then_step_kernel(monkeypatch, allow_core,
                                                                core_admits, reason):
    calls = _gdn_stubs(monkeypatch, scan_refusal=None, core_admits=core_admits)
    q, k, v, g, beta, state = _operands(T=20, vector=False)
    _, _, receipt = lr.recurrence(q, k, v, g, beta, state, allow_core=allow_core,
                                  with_receipt=True)
    assert receipt.selected == "step_kernel" and calls == ["kernel"]
    assert receipt.refusals["prefill_scan"] == "not_requested"
    assert receipt.refusals["core_scan"] == reason


def test_gdn_order_kernels_disallowed_is_reference(monkeypatch):
    calls = _gdn_stubs(monkeypatch, scan_refusal=None, core_admits=True)
    q, k, v, g, beta, state = _operands(T=20, vector=False)
    stats = Counter()
    _, _, receipt = lr.recurrence(q, k, v, g, beta, state, use_kernel=False,
                                  prefill_chunk_size=8, prefill_stats=stats, with_receipt=True)
    assert receipt.selected == "reference" and calls == [] and not stats
    assert {receipt.refusals[p] for p in ("prefill_scan", "core_scan", "step_kernel")} == {
        "kernels_disallowed"}


@pytest.mark.parametrize("chunk,path", [("wy", "chunk_wy"), ("chunk32", "chunk32")])
def test_masked_chunk_request_preserves_padding(chunk, path):
    q, k, v, g, beta, state = _operands()
    mask = mx.array([[False, True, True]])
    out, final, receipt = lr.recurrence(q, k, v, g, beta, state, mask,
                                        chunk=chunk, log_decay=mx.log(g), with_receipt=True)
    assert receipt.selected == "reference" and receipt.refusals[path] == "masked"
    _eq((out, final), lr.gated_delta_ops(q, k, v, g, beta, state, mask))
    with pytest.raises(RuntimeError, match="masked"):
        lr.recurrence(q, k, v, g, beta, state, mask, chunk=chunk,
                      log_decay=mx.log(g), policy=path)


def test_one_tap_short_conv_keeps_empty_decode_history():
    x = mx.ones((1, 2, 3))
    first, history = lr.short_conv(x, None, None, None, lambda values: values, kernel_size=1)
    second, history = lr.short_conv(x, history, None, None, lambda values: values, kernel_size=1)
    assert history.shape == (1, 0, 3)
    _eq(first, second)

@pytest.mark.parametrize("dim,state_present,refusal", [(31, True, "incomplete_simd_strip"),
                                                       (32, False, "state_required")])
def test_dispatch_receipt_matches_actual_step_kernel_support(monkeypatch, dim, state_present, refusal):
    calls = []
    _fake_gpu(monkeypatch, calls)
    q, k, v, g, beta, state = _operands(D=dim)
    state = state if state_present else None
    out, final, receipt = lr.recurrence(q, k, v, g, beta, state, allow_core=False, with_receipt=True)
    assert receipt.selected == "reference" and receipt.refusals["step_kernel"] == refusal
    assert not calls
    _eq((out, final), lr.gated_delta_ops(q, k, v, g, beta, state))
    with pytest.raises(RuntimeError, match=refusal):
        lr.recurrence(q, k, v, g, beta, state, allow_core=False, policy="step_kernel")


def test_single_step_rollback_uses_checkpoint_and_final_only():
    before = mx.array([[2], [3], [4]])
    final = mx.array([[5], [6], [7]])
    got = lr.select_rollback_rows([0, 1, 0], [1, 1, 0], mx.zeros((3, 0, 1)), before, final)
    _eq(got, mx.array([[2], [6], [4]]))
