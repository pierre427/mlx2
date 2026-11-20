"""MLX launchers for the gated block-sparse MLP kernels (opt-in, unqualified, never selected).

Granularity.  A kernel decides what to read per *work tile* (the rows that share one weight
read: 16-64 rows on dense mpp, 32 on sorted MoE mpp, up to 8 on simd).  ``per_row=True`` gives every row its own tile open
bits (``topen`` [tokens, tiles]); a block is read when any row of the work tile opens it, and the
rows that do not open it get exact zeros, so each row equals its own masked reference whatever
the other rows are.  ``per_row=False`` shares one open set (``topen`` [1, tiles], e.g. the batch
union from :func:`tile_gates` or a request-level router set) across all rows.

* decode, MoE: one work tile per (token, expert) pair, so skipping is per token;
* decode, dense B <= 16 (mpp) or <= 8 (simd): one work tile, per-row masks; the union of the
  rows' blocks is read once (reading them per row would multiply weight reads by B);
* prefill and verify: per-row masks over the work tiles above, which never read more than the
  batch union does (``per_row=False`` with the union is the batch-union law).

No copies on the hot path: gate/up read the packed weights in place, routed x rows are staged in
threadgroup memory (never gathered into a device copy), the activation is written once in
original neuron order, ``down`` reads its original weights in place, and the MoE output is
written straight to each pair's row (no unsort copy).  No host synchronization: work tiles,
masks and the sorted route table are built with device ops of static shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx
import numpy as np

from . import kernels as K
from .layout import LayerBlocks

_XT = {mx.bfloat16: "bfloat", mx.float16: "half"}


class BlockSparseUnsupported(ValueError):
    """The request cannot run on the block-sparse kernels without changing their contract."""


@dataclass(frozen=True)
class QW:
    """One affine-quantized projection as MLX arrays with an expert axis: weight uint32
    [E, N, K*bits/32], scales/biases [E, N, K/group]."""

    weight: mx.array
    scales: mx.array
    biases: mx.array
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

    @classmethod
    def from_module(cls, module) -> "QW":
        """A view of an ``nn.QuantizedLinear`` or ``QuantizedSwitchLinear`` (no copy: a rank-2
        weight gains its expert axis by a reshape of the same buffer)."""
        w, s, b = module["weight"], module["scales"], module.get("biases")
        if b is None or getattr(module, "mode", "affine") != "affine":
            raise BlockSparseUnsupported("block-sparse kernels take affine weights with biases")
        if module.get("bias") is not None:
            raise BlockSparseUnsupported("block-sparse kernels take projections without an additive bias")
        if w.ndim == 2:
            w, s, b = w[None], s[None], b[None]
        return cls(w, s, b, int(module.bits), int(module.group_size))

    def check(self) -> None:
        if self.bits not in K.BITS:
            raise BlockSparseUnsupported(f"{self.bits}-bit weights: the kernels take {K.BITS}")
        if self.group_size not in K.GROUP_SIZES:
            raise BlockSparseUnsupported(f"group size {self.group_size} not in {K.GROUP_SIZES}")
        if self.weight.dtype != mx.uint32 or self.scales.dtype not in _XT or self.biases.dtype != self.scales.dtype:
            raise BlockSparseUnsupported("packed uint32 weights with bf16/fp16 scales and biases")
        if tuple(self.scales.shape) != (self.experts, self.n, self.k // self.group_size):
            raise BlockSparseUnsupported("scale geometry does not match the weight")
        if self.k % 8:
            raise BlockSparseUnsupported("input width must be a multiple of 8")
        if self.n % 4 or self.n < 4:
            raise BlockSparseUnsupported("output width must be a positive multiple of 4")


@dataclass(frozen=True)
class DeviceBlocks:
    """A layer's block tables as device arrays (uploaded once)."""

    layout: LayerBlocks
    perm: mx.array
    chunk_ptr: mx.array
    chunks: mx.array
    gsig_ptr: mx.array
    gsig: mx.array

    @classmethod
    def from_layout(cls, lb: LayerBlocks) -> "DeviceBlocks":
        gsig = lb.gsig if lb.gsig.size else np.zeros(1, np.int32)
        return cls(lb, mx.array(lb.perm.astype(np.uint16)), mx.array(lb.chunk_ptr.astype(np.int32)),
                   mx.array(lb.chunks.astype(np.int32).reshape(-1)), mx.array(lb.gsig_ptr.astype(np.int32)),
                   mx.array(gsig.astype(np.int32)))


def pack_qweight(q: QW, lb: LayerBlocks) -> QW:
    """gate/up rows in packed order (``mx.take_along_axis`` on rows, with their scales and biases).

    Exact: :func:`unpack_qweight` returns the input bit for bit.  Run it once at load or offline;
    the packed arrays replace the originals (they are the same format)."""
    if (q.experts, q.n) != (lb.experts, lb.width):
        raise BlockSparseUnsupported("projection rows do not match the block table")
    idx = mx.array(lb.perm.astype(np.int32))[..., None]

    def take(a):
        return mx.take_along_axis(a, mx.broadcast_to(idx, (lb.experts, lb.width, int(a.shape[2]))), axis=1)

    return QW(take(q.weight), take(q.scales), take(q.biases), q.bits, q.group_size)


def unpack_qweight(q: QW, lb: LayerBlocks) -> QW:
    inv = mx.array(lb.inverse().astype(np.int32))[..., None]

    def take(a):
        return mx.take_along_axis(a, mx.broadcast_to(inv, (lb.experts, lb.width, int(a.shape[2]))), axis=1)

    return QW(take(q.weight), take(q.scales), take(q.biases), q.bits, q.group_size)


# ---------------------------------------------------------------------------
# Backend and geometry
# ---------------------------------------------------------------------------


def backend(name: str | None = None) -> str:
    """``mpp`` (M5 tensor units / NAX) or ``simd`` (M1-M4), by the lane matmul's detection;
    ``name`` pins one ("simd" also runs on an M5).  Raises off the GPU."""
    from ..lane import matmul as lane

    if name is not None:
        if name not in K.BACKENDS:
            raise BlockSparseUnsupported(f"backend must be one of {K.BACKENDS}")
        if name == "mpp" and not lane._tensor_units():
            raise BlockSparseUnsupported("the mpp (NAX) backend needs M5 tensor units")
        if mx.default_device() != mx.gpu:
            raise BlockSparseUnsupported("block-sparse kernels run on the GPU")
        return name
    found = lane.backend()
    if found not in K.BACKENDS:
        raise BlockSparseUnsupported("no Metal backend for the block-sparse kernels on this device")
    return found


def split_k(n: int, k: int, group_size: int, bits: int, *, variant: str, kind: str) -> int:
    """mpp K slices: the lane matmul's shape-only rule, capped by the threadgroup budget at the
    variant's largest work tile.  Never depends on the row count or the mask."""
    from ..lane.matmul import split_k as lane_split_k

    sk = lane_split_k(n, k, group_size, bits)
    groups = k // group_size
    while sk > 1 and (K.threadgroup_bytes(kind, "mpp", variant, sk=sk, tmr=max_tmr(variant), gs=group_size,
                                          groups=groups) > K.TG_MEMORY_BUDGET):
        sk //= 2
    return sk


def max_tmr(variant: str) -> int:
    return 2 if variant == "rowmap" else K.MAX_TMR


def tile_rows(backend_name: str, rows: int, variant: str) -> int:
    """Rows per work tile for a call of ``rows`` rows."""
    if backend_name == "simd":
        return max(1, min(K.MAX_MR, rows))
    cap = max_tmr(variant)
    return 16 * min(cap, max(1, -(-rows // 16)))


_DUMMY: dict[str, mx.array] = {}


def _dummy(kind: str) -> mx.array:
    if kind not in _DUMMY:
        _DUMMY[kind] = mx.zeros((1,), {"i32": mx.int32, "u8": mx.uint8, "f32": mx.float32}[kind])
    return _DUMMY[kind]


# ---------------------------------------------------------------------------
# Work tiles (static shapes, device ops only)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=256)
def dense_worktiles(rows: int, per_tile: int) -> mx.array:
    """int32 [U, 4] (row0, nrows, expert 0, 0) over ``rows`` consecutive rows.

    Static per (rows, per_tile): uploaded once and reused (callers never write it)."""
    starts = np.arange(0, rows, per_tile, dtype=np.int32)
    wt = np.stack([starts, np.minimum(per_tile, rows - starts), np.zeros_like(starts), np.zeros_like(starts)], 1)
    return mx.array(wt.astype(np.int32).reshape(-1))


def _flat_i32(indices: mx.array) -> mx.array:
    """Expert ids as flat int32 without a dtype copy: a uint32 router output (argpartition) is
    reinterpreted in place (ids < 2**31, so the bits are the same numbers)."""
    if indices.dtype == mx.uint32:
        indices = indices.view(mx.int32)
    elif indices.dtype != mx.int32:
        indices = indices.astype(mx.int32)
    return indices.reshape(-1)


@lru_cache(maxsize=64)
def _pair_static(p: int, k: int):
    """The routing-independent columns of the pair work tiles, and rmap (built once per shape)."""
    ar = mx.arange(p, dtype=mx.int32)
    return ar, mx.ones((p,), mx.int32), mx.zeros((p,), mx.int32), ar // k


def pair_worktiles(indices: mx.array):
    """Decode routing: one work tile per (token, expert) pair, in pair order.

    Returns ``(wt, rmap, pairs)``: ``rmap[p]`` is pair p's token; the output rows are the pairs."""
    k = int(indices.shape[-1])
    flat = _flat_i32(indices)
    p = int(flat.shape[0])
    ar, ones, zeros, rmap = _pair_static(p, k)
    wt = mx.stack([ar, ones, flat, zeros], axis=1).reshape(-1)
    return wt, rmap, p


def sorted_worktiles(indices: mx.array, num_experts: int, per_tile: int):
    """Prefill routing: pairs sorted by expert, cut into work tiles of at most ``per_tile`` rows
    of one expert.  Returns ``(wt, rmap, omap, pairs, tiles_bound)``: work row r is pair
    ``omap[r]`` of token ``rmap[r]``.  Tiles past the real count have ``nrows == 0``."""
    k = int(indices.shape[-1])
    flat = _flat_i32(indices)
    p = int(flat.shape[0])
    order = mx.argsort(flat).view(mx.int32)          # uint32 positions < 2**31: same bits, no copy
    s = flat[order]
    idx = mx.arange(p, dtype=mx.int32)
    start = mx.concatenate([mx.array([True]), s[1:] != s[:-1]]) if p > 1 else mx.array([True])
    run0 = mx.cummax(mx.where(start, idx, 0))
    tstart = mx.logical_or(start, ((idx - run0) % per_tile) == 0)
    tid = mx.cumsum(tstart.astype(mx.int32)) - 1
    bound = -(-p // per_tile) + min(num_experts, p)
    key = mx.where(tstart, tid, bound)
    zeros = mx.zeros((bound + 1,), mx.int32)
    row0 = zeros.at[key].add(mx.where(tstart, idx, 0))
    nrows = zeros.at[tid].add(1)
    expert = zeros.at[key].add(mx.where(tstart, s, 0))
    wt = mx.stack([row0, nrows, expert, zeros], axis=1)[:bound].reshape(-1)
    return wt, order // k, order, p, bound


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def gate_up_swiglu(x2, gate: QW, up: QW, blocks: DeviceBlocks, wt, n_tiles_work: int, topen, *,
                   rows_out: int, rmap=None, per_row=True, skip=True, count=False, backend_name: str,
                   per_tile: int):
    """``a`` [rows_out, F] (original neuron order) for every work row; closed neurons exact 0."""
    lb = blocks.layout
    if lb.chunk > K.NT or (lb.chunks.size and int(lb.chunks[:, 1].max()) > K.NT):
        raise BlockSparseUnsupported(f"chunks of more than {K.NT} rows: a gate/up threadgroup covers {K.NT}")
    for q in (gate, up):
        q.check()
        if (q.experts, q.n, q.k) != (lb.experts, lb.width, int(x2.shape[-1])):
            raise BlockSparseUnsupported("gate/up geometry does not match the block table or x")
    if (gate.bits, gate.group_size, gate.scales.dtype) != (up.bits, up.group_size, up.scales.dtype):
        raise BlockSparseUnsupported("gate and up must share bits, group size and scale dtype (one kernel)")
    xt = _XT.get(x2.dtype)
    if xt is None:
        raise BlockSparseUnsupported("activations must be bf16 or fp16")
    st = _XT[gate.scales.dtype]
    variant = "dense" if rmap is None else "rowmap"
    kk, F, gs = gate.k, lb.width, gate.group_size
    ntiles = int(topen.shape[-1])
    tmpl = [("K", kk), ("F", F), ("GS", gs), ("NTILES", ntiles), ("MS", int(per_row)),
            ("SKIP", int(skip)), ("COUNT", int(count))]
    if backend_name == "mpp":
        sk = split_k(F, kk, gs, gate.bits, variant=variant, kind="gu")
        tmr = per_tile // 16
        tmpl += [("SK", sk), ("TMR", tmr)]
        threads = 32 * sk
    else:
        tmpl += [("MR", per_tile)]
        threads = K.SIMD_GU_THREADS
    kern = K.build("gu", backend_name, variant, gate.bits, xt, st)
    out = kern(
        inputs=[x2, gate.weight, gate.scales, gate.biases, up.weight, up.scales, up.biases,
                blocks.perm, blocks.chunk_ptr, blocks.chunks, wt,
                rmap if rmap is not None else _dummy("i32"), topen],
        template=tmpl, grid=(lb.max_chunks * threads, n_tiles_work, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(rows_out, F), (4,)], output_dtypes=[x2.dtype, mx.uint32],
        init_value=0 if count else None,
    )
    return out[0], out[1]


def down_proj(a, down: QW, blocks: DeviceBlocks, wt, n_tiles_work: int, topen, *, rows_out: int,
              rmap=None, omap=None, per_row=True, skip=True, count=False, backend_name: str, per_tile: int):
    """``y`` [rows_out, d] from the original ``down`` weights; unopened groups are never read."""
    lb = blocks.layout
    down.check()
    if (down.experts, down.k) != (lb.experts, lb.width) or down.group_size != lb.group_size:
        raise BlockSparseUnsupported("down geometry does not match the block table")
    xt, st = _XT[a.dtype], _XT[down.scales.dtype]
    variant = "dense" if rmap is None else "rowmap"
    n, F, gs = down.n, lb.width, down.group_size
    tmpl = [("N", n), ("F", F), ("GS", gs), ("NTILES", int(topen.shape[-1])), ("MS", int(per_row)),
            ("SKIP", int(skip)), ("COUNT", int(count)), ("OUTMAP", int(omap is not None))]
    if backend_name == "mpp":
        sk = split_k(n, F, gs, down.bits, variant=variant, kind="dn")
        tmpl += [("SK", sk), ("TMR", per_tile // 16)]
        grid, tg = (-(-n // K.NT) * 32 * sk, n_tiles_work, 1), (32 * sk, 1, 1)
    else:
        tmpl += [("MR", per_tile)]
        grid, tg = (-(-n // K.SIMD_DOWN_COLS) * 64, n_tiles_work, 1), (64, 1, 1)
    kern = K.build("dn", backend_name, variant, down.bits, xt, st)
    out = kern(
        inputs=[a, down.weight, down.scales, down.biases, blocks.gsig_ptr, blocks.gsig, wt,
                rmap if rmap is not None else _dummy("i32"), omap if omap is not None else _dummy("i32"), topen],
        template=tmpl, grid=grid, threadgroup=tg,
        output_shapes=[(rows_out, n), (4,)], output_dtypes=[a.dtype, mx.uint32],
        init_value=0 if count else None,
    )
    return out[0], out[1]


def _check_topen(topen, per_row: bool, tokens: int, ntiles: int):
    if topen.dtype != mx.uint8 or topen.ndim != 2 or int(topen.shape[1]) != ntiles:
        raise BlockSparseUnsupported(f"topen must be uint8 [rows, {ntiles}]")
    want = tokens if per_row else 1
    if int(topen.shape[0]) != want:
        raise BlockSparseUnsupported(f"topen has {topen.shape[0]} rows, the call needs {want}")


def block_mlp(x, gate: QW, up: QW, down: QW, blocks: DeviceBlocks, topen, *, per_row=True, skip=True,
              count=False, backend_name: str | None = None):
    """Dense gated MLP over packed gate/up; ``x`` [..., d] -> [..., d] (and the read counters)."""
    be = backend(backend_name)
    d = int(x.shape[-1])
    lead = x.shape[:-1]
    x2 = x.reshape(-1, d)
    m = int(x2.shape[0])
    _check_topen(topen, per_row, m, blocks.layout.tiles)
    per_tile = tile_rows(be, m, "dense")
    wt = dense_worktiles(m, per_tile)
    u = -(-m // per_tile)
    a, c1 = gate_up_swiglu(x2, gate, up, blocks, wt, u, topen, rows_out=m, per_row=per_row, skip=skip,
                           count=count, backend_name=be, per_tile=per_tile)
    y, c2 = down_proj(a, down, blocks, wt, u, topen, rows_out=m, per_row=per_row, skip=skip, count=count,
                      backend_name=be, per_tile=per_tile)
    return y.reshape(*lead, down.n), (c1 + c2 if count else None)


SORT_MIN_PAIRS = 64     # below this the pairs run unsorted (one work tile each), as decode does


def block_switch_glu(x, indices, gate: QW, up: QW, down: QW, blocks: DeviceBlocks, topen, *, per_row=True,
                     skip=True, count=False, backend_name: str | None = None, sort: bool | None = None):
    """Routed experts: ``x`` [..., d], ``indices`` [..., k] -> [..., k, d] (the SwitchGLU contract).

    A pair whose expert has no open block for its token reads no weight at all (every chunk and
    every group of its work tile is skipped) and yields exact zeros."""
    be = backend(backend_name)
    d = int(x.shape[-1])
    k = int(indices.shape[-1])
    lead = x.shape[:-1]
    x2 = x.reshape(-1, d)
    tokens = int(x2.shape[0])
    _check_topen(topen, per_row, tokens, blocks.layout.tiles)
    pairs = tokens * k
    if sort is None:
        sort = pairs >= SORT_MIN_PAIRS
    if sort:
        per_tile = tile_rows(be, 64 if be == "mpp" else K.MAX_MR, "rowmap")
        wt, rmap, omap, p, u = sorted_worktiles(indices, blocks.layout.experts, per_tile)
    else:
        per_tile = 16 if be == "mpp" else 1
        wt, rmap, p = pair_worktiles(indices)
        omap, u = None, p
    a, c1 = gate_up_swiglu(x2, gate, up, blocks, wt, u, topen, rows_out=p, rmap=rmap, per_row=per_row,
                           skip=skip, count=count, backend_name=be, per_tile=per_tile)
    y, c2 = down_proj(a, down, blocks, wt, u, topen, rows_out=p, rmap=rmap, omap=omap, per_row=per_row,
                      skip=skip, count=count, backend_name=be, per_tile=per_tile)
    return y.reshape(*lead, k, down.n), (c1 + c2 if count else None)


# ---------------------------------------------------------------------------
# Tile gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateTables:
    """Device tables of one gate layer: box probes (fp32 [boxes, d], bias [boxes]) and the
    closure-folded tile membership bits (uint32 [tiles, words]) with the always-open flags."""

    probe_w: mx.array | None
    probe_b: mx.array | None
    tile_bits: mx.array
    always: mx.array
    boxes: int

    @classmethod
    def from_graph(cls, graph, probe_w=None, probe_b=None) -> "GateTables":
        bits = graph.tile_box_bits()
        pw = None if probe_w is None else mx.array(np.asarray(probe_w, np.float32))
        pb = None if probe_b is None else mx.array(np.asarray(probe_b, np.float32).reshape(-1))
        if pw is not None and (pw.ndim != 2 or int(pw.shape[0]) != len(graph.boxes) or int(pb.shape[0]) != len(graph.boxes)):
            raise BlockSparseUnsupported("one probe row and bias per box")
        return cls(pw, pb, mx.array(bits.reshape(-1)), mx.array(graph.always()), len(graph.boxes))

    @property
    def words(self) -> int:
        return max(1, -(-self.boxes // 32))

    @property
    def tiles(self) -> int:
        return int(self.always.shape[0])


def tile_gates(h, tables: GateTables, *, box_open=None, union: bool = False):
    """One launch: ``topen`` uint8 [tokens, tiles] (and the union over tokens, uint32 [tiles]).

    Probe law: box b is open when ``sum_i h_i * w_bi + b_b > 0`` (sigmoid > 0.5), fp32, lane
    strided over d with one ``simd_sum``.  ``box_open`` (uint8 [tokens, boxes]) replaces the
    probes for router-law boxes decided outside the model."""
    d = int(h.shape[-1])
    h2 = h.reshape(-1, d)
    t = int(h2.shape[0])
    mode = 0 if box_open is None else 1
    if mode == 0 and tables.probe_w is None:
        raise BlockSparseUnsupported("no probes: pass box_open (router law)")
    if mode == 0 and int(tables.probe_w.shape[1]) != d:
        raise BlockSparseUnsupported("probe width does not match the hidden state")
    xt = _XT.get(h2.dtype)
    if xt is None:
        raise BlockSparseUnsupported("hidden states must be bf16 or fp16")
    kern = K.build("gate", "simd", "dense", 4, xt, xt)
    out = kern(
        inputs=[h2, tables.probe_w if mode == 0 else _dummy("f32"), tables.probe_b if mode == 0 else _dummy("f32"),
                box_open if mode == 1 else _dummy("u8"), tables.tile_bits, tables.always],
        template=[("D", d), ("NB", tables.boxes), ("NTILES", tables.tiles), ("NW", tables.words),
                  ("MODE", mode), ("UNION", int(union))],
        grid=(256 * t, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(t, tables.tiles), (tables.tiles,)], output_dtypes=[mx.uint8, mx.uint32],
        init_value=0 if union else None,     # every topen byte is written; only the atomic union needs zeros
    )
    return out[0], (out[1] if union else None)


def union_open(un) -> mx.array:
    """The union bits as a shared ``topen`` row (uint8 [1, tiles])."""
    return un.astype(mx.uint8)[None]


# ---------------------------------------------------------------------------
# Bytes
# ---------------------------------------------------------------------------


def counter_reads(ctr) -> dict:
    """Kernel counters -> the ``layout.predicted_reads`` keys (down in units of its column tile)."""
    c = np.array(ctr).astype(np.int64)
    return {"chunks": int(c[0]), "gate_up_rows": int(c[1]), "down_group_tiles": int(c[2])}


def down_cols(backend_name: str) -> int:
    return K.NT if backend_name == "mpp" else K.SIMD_DOWN_COLS


def dense_bytes(gate: QW, up: QW, down: QW) -> int:
    """Weight bytes of one full gate/up/down read (the base path's per-expert traffic)."""
    def b(q):
        return math.prod(q.weight.shape) * 4 + 2 * math.prod(q.scales.shape) * q.scales.itemsize
    return b(gate) + b(up) + b(down)
