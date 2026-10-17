# SPDX-License-Identifier: Apache-2.0
"""Bounded affine fitting with the stored BF16 parameters in the error oracle.

Weighted clipping/refit mechanism mined from oMLX; see
provenance/nemotron35-super-vl.json. The emitted layout uses ordinary MLX
affine kernels, with no activation casts or per-token conversion.
"""
from __future__ import annotations

import numpy as np

from .nvfp4 import bf16_bits


def bf16_float(values):
    return (bf16_bits(values).astype(np.uint32) << 16).view(np.float32)


def pack_codes(codes, bits):
    """Pack row-major codes, least-significant bit first, as MLX U32 words."""
    codes = np.asarray(codes)
    if (bits not in (4, 5, 6, 8) or codes.ndim < 2
            or codes.shape[-1] * bits % 32
            or np.any(codes < 0) or np.any(codes >= 1 << bits)):
        raise ValueError("Invalid affine codes or packing geometry")
    # Eight codes fit in a uint64 even for 8-bit; slicing bytes also covers
    # the word-crossing 5/6-bit layouts without a full unpacked bit matrix.
    groups = codes.astype(np.uint64).reshape(*codes.shape[:-1], -1, 8)
    words = np.bitwise_or.reduce(groups << (np.arange(8, dtype=np.uint64) * bits), axis=-1)
    raw = words.astype("<u8").view(np.uint8).reshape(*words.shape, 8)[..., :bits].copy()
    return raw.reshape(*codes.shape[:-1], -1).view("<u4")


def quantize_rows(values, *, bits=4, group_size=64, importance=None, refine=True):
    """Fit a bounded 2-D row chunk; importance is mean squared input activation.

    Without importance this minimizes weight MSE, not activation error. Every
    accepted refit strictly improves the chosen error at *stored* precision.
    """
    values = np.asarray(values, dtype=np.float32)
    if (values.ndim != 2 or not values.size or not np.isfinite(values).all()
            or bits not in (4, 5, 6, 8) or group_size not in (32, 64, 128)
            or values.shape[-1] % group_size):
        raise ValueError("Invalid affine input, bits, or group size")
    grouped = values.reshape(values.shape[0], -1, group_size)
    if importance is None:
        imp = np.ones((1, grouped.shape[1], group_size), np.float32)
    else:
        imp = np.asarray(importance, np.float32)
        if imp.shape != (values.shape[-1],) or not np.isfinite(imp).all() or np.any(imp < 0) or not np.any(imp > 0):
            raise ValueError("Importance must contain finite nonnegative channel statistics")
        # Normalize per group so a tiny but positive group remains meaningful.
        imp = imp.reshape(1, -1, group_size)
        imp = np.maximum(imp / np.maximum(imp.mean(axis=-1, keepdims=True), 1e-30), 1e-8)
    bins = (1 << bits) - 1
    low, high = grouped.min(axis=-1, keepdims=True), grouped.max(axis=-1, keepdims=True)
    minimum = np.float32(1e-12)

    def fit(scale, bias):
        codes = np.clip(np.rint((grouped - bias) / scale), 0, bins)
        error = np.sum(imp * (grouped - (codes * scale + bias)) ** 2, axis=-1, keepdims=True)
        return codes, error

    best_scale = bf16_float(np.maximum((high - low) / bins, minimum))
    best_bias = bf16_float(low)
    if importance is not None:
        # Refinement must compete against the already-produced weight-MSE
        # candidate, not merely against a weaker min/max starting point.
        _, initial_scales, initial_biases, _ = quantize_rows(
            values, bits=bits, group_size=group_size, refine=refine
        )
        best_scale = (initial_scales.astype(np.uint32) << 16).view(np.float32)[..., None]
        best_bias = (initial_biases.astype(np.uint32) << 16).view(np.float32)[..., None]
    _, best_error = fit(best_scale, best_bias)
    initial_error = best_error.copy()
    if refine and bits < 8:
        total = imp.sum(axis=-1, keepdims=True)
        sum_v = (imp * grouped).sum(axis=-1, keepdims=True)
        starts = [(best_scale.copy(), best_bias.copy())]
        for factor in (.6, .8, 1., 1.15):
            scale = np.maximum((high - low) * factor / bins, minimum)
            starts.append((scale, (high + low) / 2 - bins * scale / 2))
        for scale, bias in starts:
            scale, bias = bf16_float(scale), bf16_float(bias)
            for _ in range(4):
                codes, error = fit(scale, bias)
                take = error < best_error
                best_error = np.where(take, error, best_error)
                best_scale = np.where(take, scale, best_scale)
                best_bias = np.where(take, bias, best_bias)
                sum_c = (imp * codes).sum(axis=-1, keepdims=True)
                denom = (imp * codes * codes).sum(axis=-1, keepdims=True) - sum_c * sum_c / total
                covariance = (imp * codes * grouped).sum(axis=-1, keepdims=True) - sum_c * sum_v / total
                candidate = covariance / np.maximum(denom, minimum)
                okay = (denom > minimum) & (candidate > minimum) & np.isfinite(candidate)
                scale = bf16_float(np.where(okay, candidate, scale))
                bias = bf16_float(np.where(okay, (sum_v - scale * sum_c) / total, bias))
    codes, error = fit(best_scale, best_bias)
    packed = pack_codes(codes.reshape(values.shape).astype(np.uint32), bits)
    shape = (values.shape[0], values.shape[-1] // group_size)
    return (packed, bf16_bits(best_scale.reshape(shape)), bf16_bits(best_bias.reshape(shape)),
            {"error_before": float(initial_error.sum(dtype=np.float64)),
             "error_after": float(error.sum(dtype=np.float64)),
             "elements": values.size, "activation_weighted": importance is not None,
             "baseline": "unweighted_lsq" if importance is not None else "stored_minmax"})
