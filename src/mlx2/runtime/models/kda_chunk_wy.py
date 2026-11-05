# SPDX-License-Identifier: MIT
"""Chunked WY prefill for the vector-gated delta rule (KDA); original MLX code.

Algorithm structure follows FLA's chunk_kda (MIT, see
provenance/kimi-linear-reap.json): every chunk's UT/WY factors are built in
parallel, a short sequential pass carries the state chunk to chunk, and the
outputs are read in parallel afterwards.

Per head, with K-by-V state S, chunk-initial state S0, log decay g (vector
over K), cumulative c_t = sum_{i<=t} g_i inside the chunk:

    G[t,j] = sum_d k_t[d] k_j[d] exp(c_t[d]-c_j[d])          (j < t)
    (I + diag(beta) G) u = beta v - beta (exp(c) * k) S0
    u = U - W S0,   U = T beta v,   W = T beta (exp(c) * k),   T = (I+L)^-1
    o_t = S0^T (exp(c_t) * q_t) + sum_{j<=t} P[t,j] u_j
    S_C = exp(c_C) * S0 + (k * exp(c_C - c))^T u

Every exponent is a difference c_t - c_j with j <= t, so each factor is at
most one: the checkpoint's unbounded softplus gate needs no lower bound and
no secondary chunking.  The pairwise [C, C, K] tensors keep that exactness
at C=16.  T comes from the nilpotent product (I+N)(I+N^2)(I+N^4)(I+N^8),
N = -L, which is exact algebra for a strictly lower 16 x 16 L.  This is a
reordered FP32 arithmetic candidate, not exact parity with the recurrence.
"""
from __future__ import annotations

import mlx.core as mx

CHUNK = 16


def refusal(q, k, v, log_g, beta, state, mask=None):
    if mask is not None:
        return "masked"
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        return "qkv_geometry"
    b, t, h, dk = q.shape
    if log_g.shape != (b, t, h, dk) or beta.shape != (b, t, h):
        return "gate_geometry"
    if state.shape != (b, h, v.shape[-1], dk) or state.dtype != mx.float32:
        return "fp32_state_required"
    if log_g.dtype != mx.float32:
        return "fp32_log_gate_required"
    return None


def chunk_wy(q, k, v, log_g, beta, state, *, segment=512):
    """q,k: [B,T,H,K]; v: [B,T,H,V]; log_g: [B,T,H,K] FP32 log decay;
    beta: [B,T,H]; state: [B,H,V,K] FP32.  Returns (o [B,T,H,V] FP32, state).

    ``segment`` bounds the pairwise tensors to segment*C*K floats per head.
    """
    reason = refusal(q, k, v, log_g, beta, state)
    if reason:
        raise ValueError("chunk_wy KDA: " + reason)
    outputs = []
    carry = state.swapaxes(-1, -2)  # [B,H,K,V]
    for start in range(0, q.shape[1], segment):
        end = min(start + segment, q.shape[1])
        out, carry = _segment(q[:, start:end], k[:, start:end], v[:, start:end],
                              log_g[:, start:end], beta[:, start:end], carry)
        outputs.append(out)
    out = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)
    return out, carry.swapaxes(-1, -2)


def _segment(q, k, v, log_g, beta, carry):
    b, t, h, dk = q.shape
    dv = v.shape[-1]
    pad = (-t) % CHUNK
    if pad:
        # Zero key/value/beta and zero log decay leave the state unchanged.
        widths = [(0, 0), (0, pad), (0, 0), (0, 0)]
        q, k, v, log_g = (mx.pad(x, widths) for x in (q, k, v, log_g))
        beta = mx.pad(beta, widths[:3])
    n = (t + pad) // CHUNK

    def chunks(x):  # [B,T,H,D] -> [B,H,N,C,D]
        return x.reshape(b, n, CHUNK, h, x.shape[-1]).transpose(0, 3, 1, 2, 4)
    qc, kc = chunks(q.astype(mx.float32)), chunks(k.astype(mx.float32))
    vc, gc = chunks(v.astype(mx.float32)), chunks(log_g)
    bc = beta.astype(mx.float32).reshape(b, n, CHUNK, h).transpose(0, 3, 1, 2)[..., None]
    c = mx.cumsum(gc, axis=-2)
    ids = mx.arange(CHUNK)
    lower = (ids[:, None] > ids[None, :])[..., None]
    causal = (ids[:, None] >= ids[None, :])[..., None]
    diff = c[..., :, None, :] - c[..., None, :, :]
    decay = mx.where(causal, mx.exp(mx.minimum(diff, 0)), 0)
    gram = mx.sum(kc[..., :, None, :] * kc[..., None, :, :] * mx.where(lower, decay, 0), axis=-1)
    readin = mx.sum(qc[..., :, None, :] * kc[..., None, :, :] * decay, axis=-1)
    neg = -(bc * gram)
    inverse = mx.eye(CHUNK, dtype=mx.float32) + neg
    power = neg
    for _ in range(3):
        power = power @ power
        inverse = inverse + inverse @ power
    upd_v = inverse @ (bc * vc)
    upd_w = inverse @ (bc * (mx.exp(c) * kc))
    q_decay = mx.exp(c) * qc
    last = c[..., -1:, :]
    k_tail = kc * mx.exp(last - c)
    total = mx.exp(last[..., 0, :])[..., None]
    starts, residuals = [], []
    for i in range(n):
        starts.append(carry)
        residual = upd_v[:, :, i] - upd_w[:, :, i] @ carry
        residuals.append(residual)
        carry = total[:, :, i] * carry + k_tail[:, :, i].swapaxes(-1, -2) @ residual
    s0 = mx.stack(starts, axis=2)
    u = mx.stack(residuals, axis=2)
    out = q_decay @ s0 + readin @ u
    out = out.transpose(0, 2, 3, 1, 4).reshape(b, n * CHUNK, h, dv)[:, :t]
    return out, carry
