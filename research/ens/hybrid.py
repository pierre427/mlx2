"""Hybrid core for H40: GDN (Gated DeltaNet), RoPE softmax attention and block-sparse attention, plus the bus.

Design note: wiki/docs/ens/hybrid-core.md. Toy-scale reference implementations in plain MLX, written for
correctness first (the sparse block masks a dense score matrix, so it has sparse *semantics*, not sparse
cost). The layer pattern is a string, e.g. "GGGGA": G = GDN, A = RoPE softmax attention, S = sparse attention.

Bus targets (H40):
  residual   bounded write into the residual stream at the field position after a site layer (as in A0)
  gdn_state  a key->value outer-product write into a GDN layer's state matrix just before the field position
Both obey a structural bypass: with no bus call the forward pass is the ordinary graph.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from research.ens import vocab as V
from research.ens.model import BusCall, bounded_update, _gather_rows


@dataclass(frozen=True)
class HybridConfig:
    vocab: int = V.VOCAB_SIZE
    d: int = 128
    heads: int = 4
    ffn: int = 512
    seq_len: int = 512
    pattern: str = "GGGGA"            # one char per block: G gdn, A rope attention, S sparse attention
    bus_width: int = 64
    tau: float = 0.5
    site: int = 1                     # residual bus write after this block index
    sparse_block: int = 32            # sparse attention block size
    sparse_topk: int = 4              # past blocks each query block may read (plus its own block)
    rope_base: float = 10000.0
    gdn_chunk: int = 0                # >0: chunkwise GDN scan of this chunk size (equal to the sequential scan)

    def to_json(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ RoPE
def rope(x, base=10000.0, offset=0):
    """Rotary embedding on the last dim of x (B, H, T, hd); hd must be even."""
    B, H, T, hd = x.shape
    half = hd // 2
    freqs = base ** (-mx.arange(half, dtype=mx.float32) / half)
    pos = mx.arange(offset, offset + T, dtype=mx.float32)[:, None] * freqs[None, :]   # (T, half)
    cos, sin = mx.cos(pos), mx.sin(pos)
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class _FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n = nn.RMSNorm(cfg.d)
        self.up = nn.Linear(cfg.d, 2 * cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, cfg.d, bias=False)

    def __call__(self, h):
        g, u = mx.split(self.up(self.n(h)), 2, axis=-1)
        return h + self.down(nn.silu(g) * u)


class AttnBlock(nn.Module):
    """Causal softmax attention with RoPE, optionally block-sparse (top-k past blocks by block-mean keys)."""

    def __init__(self, cfg: HybridConfig, sparse: bool = False):
        super().__init__()
        self.cfg, self.sparse = cfg, sparse
        self.n1 = nn.RMSNorm(cfg.d)
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.o = nn.Linear(cfg.d, cfg.d, bias=False)
        self.ffn = _FFN(cfg)

    def sparse_mask(self, q, k):
        """(B, H, T, T) boolean mask: causal; each token sees its own block (causally) plus the top-k *strictly
        earlier* blocks, chosen from its own query against block-mean keys. Selection uses only the token's
        own query and completed past blocks, so it is causal (a block-mean query would leak later tokens)."""
        B, H, T, hd = q.shape
        bs, K = self.cfg.sparse_block, self.cfg.sparse_topk
        nb = (T + bs - 1) // bs
        pad = nb * bs - T
        kp = mx.pad(k, [(0, 0), (0, 0), (0, pad), (0, 0)]) if pad else k
        kb = mx.mean(kp.reshape(B, H, nb, bs, hd), axis=3)                     # (B, H, nb, hd) block summaries
        sc = q @ kb.transpose(0, 1, 3, 2)                                      # (B, H, T, nb)
        tok_blk = mx.arange(T) // bs
        past = mx.arange(nb)[None, :] < tok_blk[:, None]                       # (T, nb) strictly earlier blocks
        sc = mx.where(past[None, None], sc, -mx.inf)
        k_eff = min(K, max(nb - 1, 0))
        if k_eff > 0:
            thr = mx.sort(sc, axis=-1)[..., -k_eff][..., None]
            chosen = (sc >= thr) & past[None, None]                            # (B, H, T, nb)
        else:
            chosen = mx.zeros(sc.shape, dtype=mx.bool_)
        own = (mx.arange(nb)[None, :] == tok_blk[:, None])[None, None]
        blocks = chosen | own
        m = blocks[..., tok_blk]                                               # (B, H, T, T): key position's block
        causal = mx.arange(T)[None, :] <= mx.arange(T)[:, None]
        return m & causal[None, None]

    def __call__(self, h):
        B, T, D = h.shape
        H = self.cfg.heads
        hd = D // H
        q, k, v = mx.split(self.qkv(self.n1(h)), 3, axis=-1)
        q, k, v = (t.reshape(B, T, H, hd).transpose(0, 2, 1, 3) for t in (q, k, v))
        q, k = rope(q, self.cfg.rope_base), rope(k, self.cfg.rope_base)
        if self.sparse:
            mask = self.sparse_mask(q, k)
            sc = (q @ k.transpose(0, 1, 3, 2)) * hd ** -0.5
            a = mx.softmax(mx.where(mask, sc, -mx.inf), axis=-1) @ v
        else:
            a = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5, mask="causal")
        h = h + self.o(a.transpose(0, 2, 1, 3).reshape(B, T, D))
        return self.ffn(h)


def gdn_chunked(q, k, v, log_alpha, beta, S0, C=32):
    """Chunkwise gated delta rule, equal to the sequential scan. q, k, v (B,T,H,hd); log_alpha, beta (B,T,H)
    (the log-decay is passed directly: log(alpha) would NaN if alpha underflows to 0, review);
    S0 (B,H,hd,hd). Returns (o (B,T,H,hd), S_T).

    Within a chunk, with g = cumsum(log alpha) and w_t = beta_t (v_t - alpha_t S_{t-1} k_t):
        S_t = gamma_t S0 + sum_{i<=t} (gamma_t/gamma_i) w_i k_i^T,   gamma_t = exp(g_t)
        (I + L) W = diag(beta) (V - diag(gamma) K S0^T),   L = strict_lower(diag(beta) (Gamma o K K^T))
        O = diag(gamma) Q S0^T + lower_incl(Gamma o Q K^T) W,   Gamma_ti = exp(g_t - g_i)
        S_C = gamma_C S0 + W^T diag(exp(g_C - g)) K
    (I + L)^-1 = (I - L)(I + L^2)(I + L^4)..., exact because L is nilpotent. Every decay factor is <= 1."""
    B, T, H, hd = q.shape
    tr = lambda x: x.transpose(0, 2, 1, 3)                                    # (B,H,T,hd)
    q, k, v = tr(q), tr(k), tr(v)
    la, be = log_alpha.transpose(0, 2, 1), beta.transpose(0, 2, 1)           # (B,H,T)
    S, outs = S0, []
    for c0 in range(0, T, C):
        c1 = min(c0 + C, T)
        n = c1 - c0
        Q, K, Vc, bc = q[:, :, c0:c1], k[:, :, c0:c1], v[:, :, c0:c1], be[:, :, c0:c1]
        g = mx.cumsum(la[:, :, c0:c1], axis=-1)                               # (B,H,n)
        idx = mx.arange(n)
        strict = (idx[:, None] > idx[None, :])
        incl = (idx[:, None] >= idx[None, :])
        Gam = mx.exp(mx.where(incl, g[..., :, None] - g[..., None, :], -mx.inf))   # (B,H,n,n), 0 above diag
        L = mx.where(strict, bc[..., :, None] * Gam * (K @ K.transpose(0, 1, 3, 2)), 0.0)
        gam = mx.exp(g)[..., None]                                            # (B,H,n,1)
        X = bc[..., None] * (Vc - gam * (K @ S.transpose(0, 1, 3, 2)))        # (B,H,n,hd)
        X = X - L @ X
        P, m = L @ L, 2
        while m < n:
            X = X + P @ X
            P, m = P @ P, m * 2
        W = X
        O = gam * (Q @ S.transpose(0, 1, 3, 2)) + (Gam * (Q @ K.transpose(0, 1, 3, 2))) @ W
        outs.append(O)
        dec = mx.exp(g[..., -1:] - g)[..., None]                              # (B,H,n,1)
        S = mx.exp(g[..., -1])[..., None, None] * S + (W * dec).transpose(0, 1, 3, 2) @ K
    o = mx.concatenate(outs, axis=2).transpose(0, 2, 1, 3)
    return o, S


class GDNBlock(nn.Module):
    """Gated DeltaNet (published form, reference scan): per head a state S (dv x dk); for each token
         S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T,    o_t = S_t q_t
    - q, k, v pass a width-4 depthwise causal conv (so one layer can bind adjacent tokens, e.g. "GR a b"),
      then SiLU; q and k are L2-normalised;
    - decay alpha_t = exp(-softplus(a_t)), write strength beta_t = sigmoid(b_t);
    - output: per-head RMSNorm, times a SiLU output gate from the input, then the output projection.
    An external write (gdn_state bus) adds a bounded outer product w_v w_k^T just before a chosen position.
    A sequential scan: correct, not fast (a chunked WY/UT form is the speed-up)."""

    CONV = 4

    def __init__(self, cfg: HybridConfig):
        super().__init__()
        self.cfg = cfg
        d, H = cfg.d, cfg.heads
        self.hd = d // H
        self.n1 = nn.RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = mx.random.normal((3 * d, self.CONV)) * (1.0 / self.CONV)    # depthwise causal kernel
        self.conv = self.conv.at[:, -1].add(1.0)                                 # starts close to identity
        self.ab = nn.Linear(d, 2 * H, bias=True)
        # softplus(a) = 0.0513 -> alpha ~ 0.95 at init; beta 0.5
        self.ab.bias = mx.concatenate([mx.full((H,), -2.943), mx.zeros((H,))])
        self.gate = nn.Linear(d, d, bias=False)
        self.onorm = nn.RMSNorm(self.hd)
        self.o = nn.Linear(d, d, bias=False)
        self.ffn = _FFN(cfg)

    def prepare(self, h):
        """q, k, v (B,T,H,hd) after conv/SiLU/L2, alpha, beta (B,T,H), and the normed input x."""
        B, T, D = h.shape
        H, hd = self.cfg.heads, self.hd
        x = self.n1(h)
        z = self.qkv(x)                                                          # (B, T, 3d)
        zp = mx.pad(z, [(0, 0), (self.CONV - 1, 0), (0, 0)])
        z = sum(zp[:, j:j + T] * self.conv[:, j] for j in range(self.CONV))     # causal depthwise conv
        z = nn.silu(z)
        q, k, v = (t.reshape(B, T, H, hd) for t in mx.split(z, 3, axis=-1))
        q = q / mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k = k / mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        ab = self.ab(x).reshape(B, T, 2, H)
        self._log_alpha = -nn.softplus(ab[:, :, 0])                            # kept for the chunked scan
        alpha = mx.exp(self._log_alpha)
        beta = mx.sigmoid(ab[:, :, 1])
        return q, k, v, alpha, beta, x

    def scan(self, h, write=None):
        """Returns (raw o (B,T,H,hd), final state, x). write: None or (pos (B,), (wk, wv) each (B,H,hd)).
        With cfg.gdn_chunk > 0 the chunkwise form runs (equal to the sequential scan); a state write splits the
        scan at its position, which must then be the same for every row (else the sequential scan runs)."""
        q, k, v, alpha, beta, x = self.prepare(h)
        la = self._log_alpha
        B, T, H, hd = q.shape
        C = self.cfg.gdn_chunk
        if C > 0:
            S0 = mx.zeros((B, H, hd, hd))
            if write is None:
                o, S = gdn_chunked(q, k, v, la, beta, S0, C)
                return o, S, x
            pos, (wk, wv) = write
            ps = set(pos.tolist())
            if len(ps) == 1:
                p0 = ps.pop()
                parts, S = [], S0
                if p0 > 0:
                    o1, S = gdn_chunked(q[:, :p0], k[:, :p0], v[:, :p0], la[:, :p0], beta[:, :p0], S, C)
                    parts.append(o1)
                if p0 < T:                     # the sequential scan never applies a write at or past T (review)
                    S = S + wv[..., :, None] * wk[..., None, :]
                    o2, S = gdn_chunked(q[:, p0:], k[:, p0:], v[:, p0:], la[:, p0:], beta[:, p0:], S, C)
                    parts.append(o2)
                return mx.concatenate(parts, axis=1), S, x
        S = mx.zeros((B, H, hd, hd))
        outs = []
        for t in range(T):
            if write is not None:
                pos, (wk, wv) = write
                at = (pos == t).astype(S.dtype)[:, None, None, None]
                S = S + at * (wv[..., :, None] * wk[..., None, :])
            kt, vt = k[:, t], v[:, t]
            a_t, b_t = alpha[:, t][..., None, None], beta[:, t][..., None, None]
            Sk = S @ kt[..., None]
            S = a_t * (S - b_t * Sk * kt[..., None, :]) + b_t * vt[..., :, None] * kt[..., None, :]
            outs.append((S @ q[:, t][..., None])[..., 0])
        return mx.stack(outs, axis=1), S, x

    def __call__(self, h, write=None):
        o, _, x = self.scan(h, write)
        B, T, D = h.shape
        o = self.onorm(o).reshape(B, T, D) * nn.silu(self.gate(x))
        return self.ffn(h + self.o(o))


class HybridCore(nn.Module):
    def __init__(self, cfg: HybridConfig):
        super().__init__()
        assert set(cfg.pattern) <= set("GAS"), cfg.pattern
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = [GDNBlock(cfg) if c == "G" else AttnBlock(cfg, sparse=(c == "S")) for c in cfg.pattern]
        self.norm = nn.RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        w, d = cfg.bus_width, cfg.d
        self.reader = nn.Linear(w, d, bias=False)                       # residual bus reader
        self.gate = mx.array([1.0])
        self.state_k = nn.Linear(w, d, bias=False)                      # gdn_state bus: message -> per-head key, value
        self.state_v = nn.Linear(w, d, bias=False)

    def __call__(self, tokens, bus: Optional[BusCall] = None, *, target: str = "residual",
                 gdn_layer: Optional[int] = None):
        """No bus call: the ordinary graph. target 'residual' (site = cfg.site) or 'gdn_state' (gdn_layer).
        Every bus argument is validated up front, so a bus experiment can never be a silent no-op (review)."""
        if bus is not None:
            if callable(bus.message):
                raise ValueError("the hybrid bus takes an array message; tap-style callables are not wired yet")
            if target == "residual":
                if not 0 <= self.cfg.site < len(self.blocks):
                    raise ValueError(f"site {self.cfg.site} outside the {len(self.blocks)} blocks")
            elif target == "gdn_state":
                if gdn_layer is None or not 0 <= gdn_layer < len(self.blocks):
                    raise ValueError(f"gdn_layer {gdn_layer} is not a block index")
                if not isinstance(self.blocks[gdn_layer], GDNBlock):
                    raise ValueError(f"block {gdn_layer} is not a GDN block")
            else:
                raise ValueError(f"unknown bus target {target!r}")
        h = self.embed(tokens)
        for i, blk in enumerate(self.blocks):
            if bus is not None and target == "gdn_state" and i == gdn_layer:
                h = blk(h, write=self._state_write(bus))
            else:
                h = blk(h)
            if bus is not None and target == "residual" and i == self.cfg.site:
                h = self._residual_write(h, bus)
        return self.head(self.norm(h))

    def _residual_write(self, h, bus: BusCall):
        h_row = _gather_rows(h, bus.field_idx)
        delta = bounded_update(self.gate * self.reader(bus.message), h_row, self.cfg.tau)
        T = h.shape[1]
        mask = (mx.arange(T)[None, :] == bus.field_idx[:, None]).astype(h.dtype)[..., None]
        return h + mask * delta[:, None, :]

    def _state_write(self, bus: BusCall):
        """Bound (units: absolute L2 norm of the written value vector, per head): ||gate * w_v|| <= tau. The
        clamp is applied after the gate, so a learned gate cannot exceed it (review). Keys are unit vectors,
        as the scan's keys are."""
        B = bus.message.shape[0]
        H, hd = self.cfg.heads, self.cfg.d // self.cfg.heads
        wk = self.state_k(bus.message).reshape(B, H, hd)
        wk = wk / mx.sqrt(mx.sum(wk * wk, axis=-1, keepdims=True) + 1e-6)
        wv = self.gate * self.state_v(bus.message).reshape(B, H, hd)
        n = mx.sqrt(mx.sum(wv * wv, axis=-1, keepdims=True) + 1e-12)
        wv = wv * mx.minimum(1.0, self.cfg.tau / n)
        return bus.field_idx, (wk, wv)
