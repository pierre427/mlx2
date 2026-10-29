# SPDX-License-Identifier: BSD-3-Clause
# Algorithm adapted from t-head/pla dfd7eb2, FlashQLA chunk/WY pipeline.
# See provenance/pla-xpu-kernels-20261024.json. Original MLX implementation.
"""Unsplit chunk32 GDN candidate, with full FP32 state and no decay cutoff.

The triangular solve and matrix readout reorder floating point operations;
this is an explicit arithmetic candidate, not exact ordinary token parity.
Unlike PLA's optional CP warmup, every token and incoming state contributes.
Zero gates reset the carry exactly without log(0) or inf-inf intermediates.
"""
from __future__ import annotations

import mlx.core as mx

CHUNK_SIZE = 32
HEAD_CONFIGS = frozenset({(1, 8), (2, 4), (2, 6), (2, 8), (2, 16),
    (4, 8), (4, 12), (4, 16), (4, 32), (8, 16), (8, 24), (8, 64),
    (16, 16), (16, 24), (16, 32), (16, 48), (16, 64), (16, 128),
    (4, 4), (6, 6), (8, 8), (12, 12), (24, 24), (32, 32),
    (48, 48), (64, 64), (96, 96), (128, 128)})


def refusal(q, k, v, g, beta, state, mask=None):
    if mask is not None:
        return "masked"
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        return "qkv_rank_or_shape"
    b, t, hk, dk = q.shape
    hv, dv = v.shape[-2:]
    if (hk, hv) not in HEAD_CONFIGS or hv < hk or hv % hk or dk != 128 or dv != 128:
        return "head_geometry"
    if t < 1 or v.shape[:2] != (b, t):
        return "token_geometry"
    if g.shape not in ((b, t, hv), (b, t, hv, dk)) or beta.shape != (b, t, hv):
        return "gate_geometry"
    if state.shape != (b, hv, dv, dk) or state.dtype != mx.float32:
        return "fp32_state_required"
    if g.dtype != mx.float32 or q.dtype not in (mx.float32, mx.bfloat16, mx.float16):
        return "activation_or_gate_dtype"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return "mixed_qkv_dtype"
    return None


def chunk32(q, k, v, g, beta, state, mask=None):
    reason = refusal(q, k, v, g, beta, state, mask)
    if reason:
        raise ValueError("chunk32 GDN: " + reason)
    dtype = q.dtype
    hk, hv = q.shape[2], v.shape[2]
    outputs = []
    # Matmul convention is [B,H,T,D]; public recurrent state is [B,H,V,K].
    carry = state.swapaxes(-1, -2)
    for start in range(0, q.shape[1], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, q.shape[1])
        qc, kc = (x[:, start:end].astype(mx.float32).transpose(0, 2, 1, 3)
                  for x in (q, k))
        if hk != hv:
            qc = mx.repeat(qc, hv // hk, axis=1)
            kc = mx.repeat(kc, hv // hk, axis=1)
        vc = v[:, start:end].astype(mx.float32).transpose(0, 2, 1, 3)
        vector = g.ndim == 4
        gc = g[:, start:end].transpose(0, 2, 1, 3) if vector else g[:, start:end].transpose(0, 2, 1)
        bc = beta[:, start:end].astype(mx.float32).transpose(0, 2, 1)
        n = end - start
        # D[i,j] = product(g[j+1:i+1]); a zero anywhere in that interval
        # masks the factor, while g[j] is intentionally excluded.
        axis = -2 if vector else -1
        logs = mx.cumsum(mx.log(mx.where(gc > 0, gc, 1.0)), axis=axis)
        zeros = mx.cumsum((gc == 0).astype(mx.int32), axis=axis)
        if vector:
            delta_log = logs[..., :, None, :] - logs[..., None, :, :]
            same_zeros = zeros[..., :, None, :] == zeros[..., None, :, :]
        else:
            delta_log = logs[..., :, None] - logs[..., None, :]
            same_zeros = zeros[..., :, None] == zeros[..., None, :]
        ids = mx.arange(n)
        causal = ids[:, None] >= ids[None, :]
        decay = mx.where((causal[..., None] if vector else causal) & same_zeros,
                         mx.exp(mx.minimum(delta_log, 0)), 0)
        incoming = mx.cumprod(gc, axis=axis)
        if vector:
            gram = mx.sum(kc[..., :, None, :] * kc[..., None, :, :] * decay, axis=-1) * bc[..., :, None]
            rhs = bc[..., None] * (vc - (kc * incoming) @ carry)
        else:
            gram = (kc @ kc.swapaxes(-1, -2)) * decay * bc[..., :, None]
            rhs = bc[..., None] * (vc - incoming[..., None] * (kc @ carry))
        # Forward substitution of the unit lower triangular I + beta K K^T D.
        rows = []
        for i in range(n):
            row = rhs[..., i:i+1, :]
            if i:
                row = row - gram[..., i:i+1, :i] @ mx.concatenate(rows, axis=-2)
            rows.append(row)
        update = mx.concatenate(rows, axis=-2)
        if vector:
            readout = (qc * incoming) @ carry
            readout = readout + mx.sum(qc[..., :, None, :] * kc[..., None, :, :] * decay, axis=-1) @ update
            carry = incoming[..., -1, :, None] * carry + (
                kc * decay[..., -1, :, :]).swapaxes(-1, -2) @ update
        else:
            readout = incoming[..., None] * (qc @ carry)
            readout = readout + ((qc @ kc.swapaxes(-1, -2)) * decay) @ update
            carry = incoming[..., -1:, None] * carry + kc.swapaxes(-1, -2) @ (
                decay[..., -1, :, None] * update)
        outputs.append(readout.transpose(0, 2, 1, 3).astype(dtype))
    return mx.concatenate(outputs, axis=1), carry.swapaxes(-1, -2)
