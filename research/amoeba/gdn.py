"""Differentiable Gated DeltaNet recurrence in plain MLX ops (for training the host).

Recurrence per head (state S is Dv x Dk):

    S_t = a_t * S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T,     o_t = S_t q_t

``chunked`` is the chunkwise-parallel form (Yang, Kautz & Hatamizadeh, "Gated
Delta Networks", 2024; the FLA naive chunk reference). Within a chunk of C
steps, the delta-rule correction is the inverse of a unit lower-triangular
matrix, computed by recursive block inversion. Everything is matmuls, so
autograd and the GPU handle it; only the loop over chunks is sequential.

``sequential`` is the step-by-step reference used to validate it.
Inputs: q, k, v [B, H, T, D]; log_a [B, H, T] (log decay, <= 0); b [B, H, T].
q and k are expected L2-normalised; q may carry the 1/sqrt(Dk) scale.
"""
from __future__ import annotations

import math

import mlx.core as mx


def sequential(q, k, v, log_a, b, state=None):
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    S = mx.zeros((B, H, Dv, Dk)) if state is None else state
    outs = []
    for t in range(T):
        a = mx.exp(log_a[:, :, t])[..., None, None]
        kt, vt, qt, bt = k[:, :, t], v[:, :, t], q[:, :, t], b[:, :, t][..., None]
        Sk = (S @ kt[..., None])[..., 0]                      # S_{t-1} k_t, scaled below
        S = a * S + (bt * (vt - (a[..., 0] * Sk)))[..., None] * kt[..., None, :]
        outs.append((S @ qt[..., None])[..., 0])
    return mx.stack(outs, axis=2), S


def _unit_lower_inverse(M, C):
    """(I - M)^-1 for strictly lower-triangular M (..., C, C), by recursive block inversion.

    [[A, 0], [Cb, D]]^-1 = [[A^-1, 0], [-D^-1 Cb A^-1, D^-1]] with A = I - M11, D = I - M22.
    Stable like forward substitution; repeated squaring is not (near-identical keys make the
    powers of M grow combinatorially before they cancel)."""
    if C == 1:
        return mx.ones(M.shape)
    h = C // 2
    Ai = _unit_lower_inverse(M[..., :h, :h], h)
    Di = _unit_lower_inverse(M[..., h:, h:], C - h)
    lower = Di @ M[..., h:, :h] @ Ai          # -D^-1 (-M21) A^-1
    top = mx.concatenate([Ai, mx.zeros(Ai.shape[:-1] + (C - h,))], axis=-1)
    bot = mx.concatenate([lower, Di], axis=-1)
    return mx.concatenate([top, bot], axis=-2)


def chunked(q, k, v, log_a, b, chunk=64, state=None):
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    C = chunk
    pad = (-T) % C
    if pad:
        z = lambda x: mx.pad(x, [(0, 0), (0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 3))
        q, k, v, log_a, b = z(q), z(k), z(v), z(log_a), z(b)
    N = (T + pad) // C
    r = lambda x: x.reshape(B, H, N, C, *x.shape[3:])
    q, k, v, log_a, b = r(q), r(k), r(v), r(log_a), r(b)
    g = mx.cumsum(log_a, axis=-1)                                 # (B,H,N,C) cumulative log decay
    kb = k * b[..., None]
    vb = v * b[..., None]
    lower_strict = mx.tril(mx.ones((C, C)), k=-1)
    lower_incl = mx.tril(mx.ones((C, C)))
    diff = g[..., :, None] - g[..., None, :]
    decay = mx.exp(mx.where(lower_incl > 0, diff, -mx.inf))      # exp(g_i - g_j) for i >= j, else 0
    M = -(kb @ mx.swapaxes(k, -1, -2)) * decay * lower_strict
    Tm = _unit_lower_inverse(M, C)                                # (I - M)^-1
    u = Tm @ vb                                                   # (B,H,N,C,Dv)
    w = Tm @ (kb * mx.exp(g)[..., None])                          # (B,H,N,C,Dk)
    S = mx.zeros((B, H, Dv, Dk)) if state is None else state
    outs = []
    for n in range(N):
        qn, kn, gn = q[:, :, n], k[:, :, n], g[:, :, n]
        attn = (qn @ mx.swapaxes(kn, -1, -2)) * decay[:, :, n]    # causal incl. diagonal
        v_new = u[:, :, n] - w[:, :, n] @ mx.swapaxes(S, -1, -2)  # (C,Dv)
        o_inter = (qn * mx.exp(gn)[..., None]) @ mx.swapaxes(S, -1, -2)
        outs.append(o_inter + attn @ v_new)
        g_last = gn[..., -1:]
        kd = kn * mx.exp(g_last - gn)[..., None]
        S = S * mx.exp(g_last)[..., None] + mx.swapaxes(v_new, -1, -2) @ kd
    o = mx.concatenate(outs, axis=2)
    return o[:, :, :T], S
