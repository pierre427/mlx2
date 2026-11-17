"""H0, the amoeba's recurrent host (plan section 2.2).

* Lexicon box (box 0): E = E_donor @ P, the donor's frozen embedding table times
  a trainable 1024 -> 512 projection, tied to the unembedding
  (logits = h @ P^T @ E_donor^T). P starts as the top-512 right singular vectors
  of E_donor, so the host begins with the donor's dominant embedding directions.
* 8 layers, each a Gated DeltaNet mixer (4 heads x 128, short causal conv,
  gated output norm) followed by a small SwiGLU FFN (512 -> 1024 -> 512).
* Box slots are added by later milestones; H0 has none.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from research.amoeba import gdn


class ShortConv(nn.Module):
    def __init__(self, ch, k=4):
        super().__init__()
        self.k = k
        self.w = mx.zeros((k, ch)).at[k - 1].add(1.0)   # identity at init

    def __call__(self, x):                               # x (B,T,ch), causal depthwise
        xp = mx.pad(x, [(0, 0), (self.k - 1, 0), (0, 0)])
        T = x.shape[1]
        return sum(xp[:, j:j + T] * self.w[j] for j in range(self.k))


class GDNLayer(nn.Module):
    def __init__(self, d=512, heads=4, hd=128):
        super().__init__()
        self.h, self.hd = heads, hd
        inner = heads * hd
        self.norm = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * inner, bias=False)
        self.conv = ShortConv(3 * inner)
        self.ab = nn.Linear(d, 2 * heads, bias=True)
        self.A_log = mx.log(mx.linspace(1.0, 16.0, heads))
        self.gate = nn.Linear(d, inner, bias=False)
        self.onorm = nn.RMSNorm(hd)
        self.out = nn.Linear(inner, d, bias=False)

    def __call__(self, x):
        B, T, _ = x.shape
        xn = self.norm(x)
        qkv = nn.silu(self.conv(self.qkv(xn)))
        q, k, v = mx.split(qkv, 3, axis=-1)
        sh = lambda t: t.reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        l2 = lambda t: t * mx.rsqrt(mx.sum(t * t, axis=-1, keepdims=True) + 1e-6)
        q, k = l2(q) * self.hd ** -0.5, l2(k)
        a, b = mx.split(self.ab(xn), 2, axis=-1)
        beta = mx.sigmoid(b).transpose(0, 2, 1)
        log_a = (-mx.exp(self.A_log) * nn.softplus(a - 4.0)).transpose(0, 2, 1)
        o, _ = gdn.chunked(q, k, v, log_a, beta, chunk=64)
        o = self.onorm(o.transpose(0, 2, 1, 3)) * nn.silu(self.gate(xn).reshape(B, T, self.h, self.hd))
        return x + self.out(o.reshape(B, T, -1))


class FFN(nn.Module):
    def __init__(self, d=512, hidden=1024):
        super().__init__()
        self.norm = nn.RMSNorm(d)
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def __call__(self, x):
        xn = self.norm(x)
        return x + self.down(nn.silu(self.gate(xn)) * self.up(xn))


class Host(nn.Module):
    def __init__(self, E_donor: mx.array, d=512, layers=8):
        super().__init__()
        self._E = E_donor                                  # frozen, not a parameter (leading underscore)
        U, S, Vt = mx.linalg.svd(E_donor[::8].astype(mx.float32), stream=mx.cpu)   # subsample rows for speed
        self.P = Vt[:d].T * (d ** 0.5 / mx.sqrt(mx.mean(mx.sum((E_donor[::64] @ Vt[:d].T) ** 2, -1))))
        self.layers = [m for _ in range(layers) for m in (GDNLayer(d), FFN(d))]
        self.norm = nn.RMSNorm(d)
        # learnable logit scale in log space, initialised so logits of a unit-RMS state have unit
        # std. (A raw scalar of ~0.05 is driven to 0 by Adam within a few steps, which makes every
        # logit 0 and the output exactly uniform; seen 2026-11-11.)
        probe = mx.random.normal((64, d)) @ self.P.T @ E_donor[::16].T
        self.log_logit_scale = -mx.log(mx.std(probe))

    def embed(self, ids):
        return self._E[ids] @ self.P

    def states(self, ids):
        """Residual stream after each (mixer, FFN) pair: list of (B,T,d)."""
        h = self.embed(ids)
        out = []
        for j, m in enumerate(self.layers):
            h = m(h)
            if j % 2 == 1:
                out.append(h)
        return out

    def __call__(self, ids, vocab=None):
        h = self.embed(ids)
        for m in self.layers:
            h = m(h)
        h = self.norm(h) @ self.P.T                         # back to donor embedding space
        E = self._E if vocab is None else self._E[vocab]
        return (h @ E.T) * mx.exp(self.log_logit_scale)


def n_params(model):
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
