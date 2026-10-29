"""Default-off four-row weight reuse and COW-safe physical RoPE/append.

Original Metal kernels; provenance/nanbeige43-physical-kernels.json. A physical
single custom invocation may still lose to native kernels and copy preparation.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
from mlx import nn

FOUR_ROW_LAW = "nanbeige43-four-actual-row-q8g64-fp32-v1"
ROPE_APPEND_LAW = "nanbeige43-functional-physical-rope-append-v2-runtime-offset"

_FOUR_ROW_SOURCE = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint col = threadgroup_position_in_grid.x * 4 + simdgroup_index_in_threadgroup;
    if (col >= N) return;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint base = 0; base < K; base += 128) {
        const uint first = base + lane * 4;
        // One packed weight load is shared by all FOUR real activation rows.
        const uint word = W[size_t(col) * (K / 4) + first / 4];
        const uint group = size_t(col) * (K / 64) + first / 64;
        const float scale = float(S[group]), bias = float(B[group]);
        float weight[4];
        for (uint i = 0; i < 4; ++i)
            weight[i] = fma(scale, float((word >> (8*i)) & 255u), bias);
        for (uint row = 0; row < 4; ++row) {
            for (uint i = 0; i < 4; ++i)
                acc[row] = fma(float(X[size_t(row) * K + first + i]), weight[i], acc[row]);
        }
    }
    for (uint row = 0; row < 4; ++row) {
        float total = simd_sum(acc[row]);
        if (lane == 0) OUT[size_t(row) * N + col] = T(total);
    }
"""

_ROPE_APPEND_SOURCE = r"""
    const uint i = thread_position_in_grid.x;
    constexpr uint HALF = 64;
    constexpr float BASE = 120000000.0f;
    const uint offset = uint(OFF[0]);
    if (i < QSIZE) {
        const uint d = i % 128;
        const uint row = (i / 128) % ROWS;
        const uint h = i / (128 * ROWS);
        const uint pair = d % HALF;
        const float angle = float(P[row]) * metal::pow(float(BASE), -float(pair) / float(HALF));
        const float sine = metal::sin(angle), cosine = metal::cos(angle);
        const size_t index = (size_t(h) * ROWS + row) * 128 + pair;
        const float first = float(Q[index]), second = float(Q[index + HALF]);
        QOUT[i] = T(d < HALF ? first * cosine - second * sine : first * sine + second * cosine);
    } else if (i < QSIZE + KVSIZE) {
        const uint j = i - QSIZE;
        const uint d = j % 128;
        const uint slot = (j / 128) % CAPACITY;
        const uint h = j / (128 * CAPACITY);
        if (slot >= offset && slot < offset + ROWS) {
            const uint row = slot - offset;
            const uint pair = d % HALF;
            const float angle = float(P[row]) * metal::pow(float(BASE), -float(pair) / float(HALF));
            const float sine = metal::sin(angle), cosine = metal::cos(angle);
            const size_t index = (size_t(h) * ROWS + row) * 128 + pair;
            const float first = float(K[index]), second = float(K[index + HALF]);
            KOUT[j] = T(d < HALF ? first * cosine - second * sine : first * sine + second * cosine);
            VOUT[j] = V[(size_t(h) * ROWS + row) * 128 + d];
        } else {
            KOUT[j] = OLDK[j];
            VOUT[j] = OLDV[j];
        }
    }
"""


def source_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def four_row_eligible(x, module):
    return (
        isinstance(module, nn.QuantizedLinear)
        and module.bits == 8
        and module.group_size == 64
        and module.mode == "affine"
        and "bias" not in module
        and x.ndim >= 2
        and x.size // x.shape[-1] == 4
        and x.shape[-1] % 128 == 0
        and module.weight.shape[1] * 4 == x.shape[-1]
        and module.weight.dtype == mx.uint32
        and x.dtype in (mx.bfloat16, mx.float16)
        and module.scales.dtype == x.dtype
        and module.biases.dtype == x.dtype
    )


@lru_cache(maxsize=1)
def _four_row_kernel():
    digest = hashlib.sha256(_FOUR_ROW_SOURCE.encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name="nanbeige_q8_shared4_" + digest,
        input_names=["X", "W", "S", "B"],
        output_names=["OUT"],
        source=_FOUR_ROW_SOURCE,
        ensure_row_contiguous=True,
    )


def q8_four_row(x, module):
    """Exactly four physical rows share each weight word; no padded row tile."""
    if not four_row_eligible(x, module):
        raise ValueError(
            "four-row Q8 requires four bf16/fp16 rows, affine g64, K divisible by128"
        )
    if mx.default_device() != mx.gpu:
        # Independent tensor reference only; never reported as Metal execution.
        return mx.quantized_matmul(
            x,
            module.weight,
            scales=module.scales,
            biases=module.biases,
            transpose=True,
            group_size=64,
            bits=8,
            mode="affine",
        )
    k, n = x.shape[-1], module.weight.shape[0]
    out = _four_row_kernel()(
        inputs=[x.reshape(4, k), module.weight, module.scales, module.biases],
        template=[("T", x.dtype), ("K", k), ("N", n)],
        grid=(((n + 3) // 4) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(4, n)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(*x.shape[:-1], n)


def rope_append_eligible(q, k, v, keys, values, offset, positions, base):
    if (
        type(offset) is not int
        or offset < 0
        or type(base) not in (int, float)
        or float(base) != 120000000.0
    ):
        return False
    if any(a is None for a in (q, k, v, keys, values, positions)):
        return False
    rows = k.shape[2] if k.ndim == 4 else 0
    return (
        q.ndim == k.ndim == v.ndim == keys.ndim == values.ndim == 4
        and q.shape == (1, 56, rows, 128)
        and k.shape == v.shape == (1, 8, rows, 128)
        and keys.shape == values.shape
        and keys.shape[:2] == (1, 8)
        and keys.shape[-1] == 128
        and 1 <= rows <= 32
        and offset + rows <= keys.shape[2]
        and positions.shape == (rows,)
        and positions.dtype == mx.int32
        and q.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and all(a.dtype == q.dtype for a in (k, v, keys, values))
    )


@lru_cache(maxsize=1)
def _rope_append_kernel():
    digest = hashlib.sha256(_ROPE_APPEND_SOURCE.encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name="nanbeige_physical_rope_append_" + digest,
        input_names=["Q", "K", "V", "OLDK", "OLDV", "P", "OFF"],
        output_names=["QOUT", "KOUT", "VOUT"],
        source=_ROPE_APPEND_SOURCE,
        ensure_row_contiguous=True,
    )


def _reference_rotate(x, positions, base):
    # Per-row positions, including siblings at equal depth and arbitrary branches.
    heads, rows, dim = x.shape[1:]
    view = x.transpose(0, 2, 1, 3).reshape(rows, heads, 1, dim)
    out = mx.fast.rope(
        view, dim, traditional=False, base=base, scale=1.0, offset=positions
    )
    return out.reshape(1, rows, heads, dim).transpose(0, 2, 1, 3)


def fused_rope_append_arrays(
    q, k, v, keys, values, offset, positions, *, base=120000000.0
):
    """Pure successor arrays; None refuses before dispatch and owner mutation.

    One original Metal invocation per admitted lane rotates Q/newK and appends
    K/V while preserving untouched state in distinct full-capacity outputs.
    Contiguous-input preparation and copying old state are explicit extra costs.
    Offset is a runtime int32 operand; advancing the committed boundary does
    not specialize another pipeline when the capacity/row geometry is stable.
    """
    if not rope_append_eligible(q, k, v, keys, values, offset, positions, base):
        return None
    rows, capacity = k.shape[2], keys.shape[2]
    if mx.default_device() != mx.gpu:
        qr, kr = (
            _reference_rotate(q, positions, base),
            _reference_rotate(k, positions, base),
        )
        start = mx.array([offset], mx.int32)
        return (
            qr,
            mx.slice_update(keys, kr, start, axes=(2,)),
            mx.slice_update(values, v, start, axes=(2,)),
            offset + rows,
        )
    qsize, kvsize = q.size, keys.size
    out = _rope_append_kernel()(
        inputs=[q, k, v, keys, values, positions, mx.array([offset], mx.int32)],
        template=[
            ("T", q.dtype),
            ("ROWS", rows),
            ("CAPACITY", capacity),
            ("QSIZE", qsize),
            ("KVSIZE", kvsize),
        ],
        grid=(qsize + kvsize, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape, keys.shape, values.shape],
        output_dtypes=[q.dtype] * 3,
    )
    return *out, offset + rows
