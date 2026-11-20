"""CPU references for the gated block-sparse MLP (numpy; never Metal).

Two references:

* :func:`masked_reference`: float64, dequantized weights, the plain definition
  ``down((silu(gate x) * up x) * mask)`` with closed neurons zeroed.  This is the function the
  kernels compute; dtype tolerance applies.
* :func:`emulate`: the kernels' *structure* in fp32 (work tiles, chunk skips, the zero writes
  through ``perm``, group signatures, groups in index order, ``acc = s*P + (b*xs + acc)``), with a
  read recorder.  It is what the CPU tests use to prove the plumbing: that the packed layout gives
  the unpacked layout's bits, that skipping equals masking bit for bit, and that a closed block
  is never read.  It does not reproduce the GPU's in-group summation order, so its bits are not
  the GPU's bits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .layout import LayerBlocks


@dataclass
class QArrays:
    """One quantized projection as numpy arrays: weight uint32 [E, N, K*bits/32], scales/biases
    [E, N, K/group] (any float dtype numpy holds), affine ``bits``/``group_size``."""

    weight: np.ndarray
    scales: np.ndarray
    biases: np.ndarray
    bits: int
    group_size: int

    @property
    def experts(self) -> int:
        return int(self.weight.shape[0])

    @property
    def n(self) -> int:
        return int(self.weight.shape[1])

    @property
    def k(self) -> int:
        return int(self.weight.shape[2]) * 32 // self.bits


def codes(weight: np.ndarray, bits: int) -> np.ndarray:
    """uint8 codes [..., K] from MLX's packed uint32 words (LSB first)."""
    w = np.asarray(weight, np.uint32)
    per = 32 // bits
    shifts = (np.arange(per, dtype=np.uint32) * bits).astype(np.uint32)
    q = (w[..., None] >> shifts) & np.uint32((1 << bits) - 1)
    return q.reshape(*w.shape[:-1], w.shape[-1] * per).astype(np.uint8)


def pack_codes(q: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of :func:`codes`."""
    q = np.asarray(q, np.uint32)
    per = 32 // bits
    q = q.reshape(*q.shape[:-1], q.shape[-1] // per, per)
    shifts = (np.arange(per, dtype=np.uint32) * bits).astype(np.uint32)
    return np.bitwise_or.reduce(q << shifts, axis=-1).astype(np.uint32)


def dequant(p: QArrays, dtype=np.float64) -> np.ndarray:
    q = codes(p.weight, p.bits).astype(dtype)
    s = np.repeat(np.asarray(p.scales, dtype), p.group_size, axis=-1)
    b = np.repeat(np.asarray(p.biases, dtype), p.group_size, axis=-1)
    return q * s + b


def silu_mul(g, u):
    g = np.asarray(g)
    one = g.dtype.type(1)
    return (g * (one / (one + np.exp(-g)))) * u


def masked_reference(x, gate: QArrays, up: QArrays, down: QArrays, owner, topen, *, experts=None,
                     tokens=None) -> np.ndarray:
    """float64 ``down((silu(gate x) * up x) * open(owner))`` per work row.

    ``x`` [R, d] work rows, ``experts`` [R] the expert of each row (None: 0), ``tokens`` [R] the
    row of ``topen`` each work row reads (None: the row itself; ``topen`` with one row: shared)."""
    x = np.asarray(x, np.float64)
    R = x.shape[0]
    experts = np.zeros(R, np.int64) if experts is None else np.asarray(experts, np.int64)
    topen = np.asarray(topen, bool)
    if tokens is None:
        tokens = np.zeros(R, np.int64) if topen.shape[0] == 1 else np.arange(R)
    owner = np.asarray(owner).reshape(-1, gate.n)
    Wg, Wu, Wd = dequant(gate), dequant(up), dequant(down)
    y = np.zeros((R, down.n))
    for r in range(R):
        e = int(experts[r])
        g = Wg[e] @ x[r]
        u = Wu[e] @ x[r]
        a = silu_mul(g, u) * topen[int(tokens[r])][owner[e]]
        y[r] = Wd[e] @ a
    return y


@dataclass
class Reads:
    """Every weight block the emulated kernels touched."""

    gate_up_rows: set = field(default_factory=set)     # (expert, packed row)
    down_groups: set = field(default_factory=set)      # (expert, group, work tile)
    chunk_reads: int = 0
    rows_read: int = 0
    down_group_tiles: int = 0


def _group_term_rows(qrow, s, b, v, gs):
    """fp32 row-wise group law over one weight row: acc = s*P + (b*xs + acc), groups in order."""
    acc = np.zeros(v.shape[0], np.float32)
    for g in range(qrow.shape[-1] // gs):
        sl = slice(g * gs, (g + 1) * gs)
        P = (v[:, sl] @ qrow[sl].astype(np.float32)).astype(np.float32)
        xs = v[:, sl].sum(axis=1, dtype=np.float32)
        acc = (np.float32(s[g]) * P + (np.float32(b[g]) * xs + acc)).astype(np.float32)
    return acc


def bf16_round(v) -> np.ndarray:
    """fp32 values rounded to bf16 (ties to even), kept as fp32 (finite inputs)."""
    u = np.ascontiguousarray(np.asarray(v, np.float32)).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)).view(np.float32)


def _deq_rows(q, s, b, gs):
    """The prefill law's weight tile: s*q + b rounded once to fp32, as the kernel's fma (numpy
    evaluates it in float64, where s*q is exact), then rounded to the activation dtype (bf16)."""
    s64 = np.repeat(np.asarray(s, np.float32).astype(np.float64), gs, axis=-1)
    b64 = np.repeat(np.asarray(b, np.float32).astype(np.float64), gs, axis=-1)
    return bf16_round((s64 * q.astype(np.float64) + b64).astype(np.float32))


def emulate(x, gate: QArrays, up: QArrays, down: QArrays, lb: LayerBlocks, worktiles, topen, *,
            rmap=None, omap=None, rows_out=None, per_row=True, skip=True, down_cols=32,
            reads: Reads | None = None, law: str = "lane"):
    """fp32 emulation of gate/up+SwiGLU and down over the block tables (gate/up packed).

    ``law``: ``lane`` (per-group ``acc = s*P + (b*xs + acc)``) or ``prefill`` (dequantized
    weights rounded to bf16, fp32 products accumulated over the open groups in index order).
    Returns ``(a, y)``: ``a`` [R, F] the activation in original neuron order, ``y`` [rows_out, d].
    ``a`` entries no work tile covers stay NaN, so a test notices a missing write."""
    if law not in ("lane", "prefill"):
        raise ValueError("law: lane | prefill")
    x = np.asarray(x, np.float32)
    topen = np.asarray(topen, bool)
    worktiles = np.asarray(worktiles)
    R = int(max((r0 + n for r0, n, _, _ in worktiles), default=0))
    F, d = lb.width, down.n
    a = np.full((R, F), np.nan, np.float32)
    rows_out = R if rows_out is None else rows_out
    y = np.full((rows_out, d), np.nan, np.float32)
    qg, qu, qd = codes(gate.weight, gate.bits), codes(up.weight, up.bits), codes(down.weight, down.bits)
    G = lb.groups
    tiles_down = -(-d // down_cols)
    for u_i, (row0, nrows, e, _) in enumerate(worktiles):
        if nrows <= 0:
            continue
        rows = np.arange(row0, row0 + nrows)
        toks = [(int(rmap[r]) if rmap is not None else int(r)) if per_row else 0 for r in rows]
        xin = x[[int(rmap[r]) if rmap is not None else int(r) for r in rows]]
        for off, ln, t in lb.chunks[lb.chunk_ptr[e]:lb.chunk_ptr[e + 1]]:
            ro = np.array([topen[k, t] for k in toks])
            dst = lb.perm[e, off:off + ln].astype(np.int64)
            if skip and not ro.any():
                a[np.ix_(rows, dst)] = 0.0
                continue
            if reads is not None:
                reads.chunk_reads += 1
                reads.rows_read += int(ln)
            for j in range(ln):
                p = off + j
                if reads is not None:
                    reads.gate_up_rows.add((int(e), int(p)))
                if law == "prefill":
                    gv = (xin @ _deq_rows(qg[e, p], gate.scales[e, p], gate.biases[e, p], gate.group_size)
                          ).astype(np.float32)
                    uv = (xin @ _deq_rows(qu[e, p], up.scales[e, p], up.biases[e, p], up.group_size)
                          ).astype(np.float32)
                else:
                    gv = _group_term_rows(qg[e, p], gate.scales[e, p], gate.biases[e, p], xin, gate.group_size)
                    uv = _group_term_rows(qu[e, p], up.scales[e, p], up.biases[e, p], xin, up.group_size)
                val = silu_mul(gv, uv).astype(np.float32)
                a[rows, dst[j]] = np.where(ro, val, np.float32(0.0))
        # down: original layout, groups in index order, a group read only when some owner is open
        gopen = np.zeros(G, bool)
        for g in range(G):
            sig = lb.gsig[lb.gsig_ptr[e * G + g]:lb.gsig_ptr[e * G + g + 1]]
            gopen[g] = (not skip) or any(topen[k, t] for k in toks for t in sig)
        acc = np.zeros((nrows, d), np.float32)
        av = a[rows]
        gs = down.group_size
        for g in range(G):
            if not gopen[g]:
                continue
            if reads is not None:
                reads.down_groups.add((int(e), g, u_i))
                reads.down_group_tiles += tiles_down
            sl = slice(g * gs, (g + 1) * gs)
            if law == "prefill":
                wd = _deq_rows(qd[e][:, sl], down.scales[e][:, g:g + 1], down.biases[e][:, g:g + 1], gs)
                acc = (acc + av[:, sl] @ wd.T).astype(np.float32)
                continue
            P = (av[:, sl] @ qd[e][:, sl].T.astype(np.float32)).astype(np.float32)
            xs = av[:, sl].sum(axis=1, dtype=np.float32)
            acc = (down.scales[e][:, g].astype(np.float32) * P
                   + (down.biases[e][:, g].astype(np.float32) * xs[:, None] + acc)).astype(np.float32)
        dest = rows if omap is None else np.asarray(omap)[rows]
        y[dest] = acc
    return a, y
