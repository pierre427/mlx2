# SPDX-License-Identifier: MIT
"""GLM-5.3 gathered sparse MLA attention over selected latent rows.

See docs/design/glm53-sparse-dsa.md (sections 4.2, 4.3, 5.1, 7) and
provenance/glm53-sparse-dsa.json. Every path here is opt-in; the mask
consumer in glm53_gguf._mla stays the reference law.

A query attends to the ``W`` latent rows its selection names (``-1`` =
unused) instead of to every committed key under a boolean mask, so decode
work is independent of the context length. Laws:

``gathered_v1``  gather ``(L, W, 512)`` rows, one SDPA with the query rows on
                 the batch axis. Each query row is computed by the same
                 one-row arithmetic whatever ``L`` or the block size, so
                 prefill blocks, decode steps and verify rows agree bitwise.
``fused_decode_v1``  one Metal launch per <= 8 query rows of one latent
                 source (gather + online softmax over 128-key splits) and an
                 ascending-split combine. Not bitwise to ``gathered_v1``; a
                 first-use gate bounds it (1e-5 absolute) or the policy
                 raises. Exposed only if the profile shows it pays (R4).
``fused_v2``     glm53_sparse_attention_v2 (design glm53-speed-round2.md
                 1.1): each gathered tile staged once for all 64 heads, MMA
                 scores and values, rows on grid z. A new law (near-tie
                 acceptance cross-policy); any grouping of rows is bitwise
                 equal to one row per launch, so decode, prompt blocks and
                 verify rows agree bitwise within it. Prompt rows take it
                 under ``gathered_v2_from_<n>``.

Failure contract (design 4.1): a query with no valid id yields a NaN output
row in every law and nothing raises on the device; the final-logit guard
fails the request closed. The mask is additive (``0`` / ``-inf``): MLX's SDPA
fallback replaces a boolean mask's false entries by ``finfo.min``, which
would turn an empty row into a finite average of unrelated rows.

Tree frontiers (``TreeLatentView``, ``Glm53MLAFrontier``) let verify rows read
committed base rows plus a short private tail without copying the base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import mlx.core as mx

SCALE = 256**-0.5
TOPK = 2048
LATENT, RAW = 512, 256
PREFILL_BLOCK = 32
# Query positions pos with pos + 1 >= threshold take the gathered prefill law.
PREFILL_THRESHOLDS = {
    "dense_reference": None,
    "gathered_v1_8k": 8192,
    "gathered_v1_16k": 16384,
    "gathered_v1_all": TOPK + 1,
}
# Round 2 (WP1 -> WP5 interface): the v2 prompt law carries its threshold
# position; these nine values are the only v2 prefill policies.
PREFILL_POSITIONS_V2 = (2049, 3072, 4096, 5120, 6144, 7168, 8192, 12288, 16384)
PREFILL_THRESHOLDS.update({f"gathered_v2_from_{n}": n for n in PREFILL_POSITIONS_V2})
# gathered_f16_from_<n> (2026-11-02): the fp16-operand fused kernel on the
# gathered prompt rows from position n (same positions as v2; rows below n
# keep the reference dense pieces). Prompt-only: decode keeps the fp32 law
# ``glm53_sparse_attention`` names.
PREFILL_POSITIONS_F16 = PREFILL_POSITIONS_V2
PREFILL_THRESHOLDS.update({f"gathered_f16_from_{n}": n for n in PREFILL_POSITIONS_F16})
POLICIES_ATTENTION = ("mask_reference", "gathered_v1", "fused_decode_v1", "fused_v2")
DECODE_POLICIES = ("gathered_v1", "fused_decode_v1", "fused_v2")
PREFILL_LAWS = ("gathered_v1", "fused_v2", "fused_f16")
WIDTH = TOPK + 3  # 2051: 512 pools x 4 rows + the 3-row tail


def prefill_law(sparse_prefill):
    """The ``sparse_prefill`` consumer policy a prompt threshold value selects."""
    if sparse_prefill not in PREFILL_THRESHOLDS:
        raise ValueError(f"unknown GLM-5.3 sparse prefill policy {sparse_prefill!r}")
    if sparse_prefill.startswith("gathered_v2_from_"):
        return "fused_v2"
    if sparse_prefill.startswith("gathered_f16_from_"):
        return "fused_f16"
    return "gathered_v1"


def dense_row_ids(positions, width=WIDTH):
    """(L, W) int32 causal ids ``where(arange(W) <= pos, arange(W), -1)``.

    Dense rows (no selection yet, ``T <= 2048``) through the fused_v2 kernel:
    every row keeps its own causal visibility; ``W = 2051 >= 2048`` fits.
    """
    if not isinstance(positions, mx.array):
        positions = mx.array(list(positions), dtype=mx.int32)
    positions = positions.astype(mx.int32).reshape(-1)
    arange = mx.arange(width, dtype=mx.int32)
    return mx.where(arange[None] <= positions[:, None], arange[None], mx.array(-1, mx.int32))


def fused_v2_workspace_bytes(rows, *, split=64):
    """Partials of one fused_v2 launch: rows * ceil(W / split) * 64 * (512 + 2) * 4."""
    return rows * (-(-WIDTH // split)) * 64 * (LATENT + 2) * 4
# Batched SDPA over R verify rows is row-exact on the builds checked so far;
# a GPU check that finds otherwise sets this False (R7) and rows_attention
# runs one SDPA per row until the fused rows kernel lands.
# Metal SDPA picks a different kernel for a batch of one query than for two
# or more (bitwise different; every batch of >= 2 agrees with every other),
# so batched verify rows cannot equal one-row decode: law V1 runs per row.
# GLM-5.3-Flash-REAP50-Q3_K_M.gguf@5a865918, mlx 0.32.4, M5 Max, 2026-10-31.
BATCHED_ROWS_EXACT = False
FUSED_HEADS, FUSED_SPLIT, FUSED_STAGE, FUSED_GROUP = 64, 128, 8, 8
FUSED_THREADS = 256  # 8 simdgroups, one head each
_SCHEDULED = {"gathered": 0, "fused_launches": 0, "rows_batched": 0, "rows_per_row": 0, "prefill_blocks": 0}
_FUSED_GATE = {"passed": None}


def _count(counters, name, n=1):
    if counters is not None:
        counters[name] = counters.get(name, 0) + n


@dataclass(frozen=True)
class TreeLatentView:
    """Committed base rows ``(T0, 512)`` then a path tail ``(d, 512)``.

    Ids below ``split`` read ``base``, the others ``tail[id - split]``; the
    gathered values equal a gather from the concatenation, without building it.
    ``origin`` names the committed cache ``base`` was sliced from (a frontier
    sets it): fused_v2 batches rows into one launch only when every row has
    the same origin and split, i.e. the same committed base rows.
    """

    base: mx.array
    tail: mx.array
    split: int
    origin: object = field(default=None, compare=False)

    def __post_init__(self):
        if self.base.ndim != 2 or self.tail.ndim != 2 or self.base.shape[1] != self.tail.shape[1]:
            raise ValueError("GLM-5.3 tree latent view needs rank-2 base and tail rows of one width")
        if type(self.split) is not int or self.split != self.base.shape[0]:
            raise ValueError("GLM-5.3 tree latent view split must be the base row count")

    @property
    def total(self):
        return self.split + self.tail.shape[0]


def _latent_total(latent):
    if isinstance(latent, TreeLatentView):
        return latent.total
    if latent.ndim != 4 or latent.shape[:2] != (1, 1):
        raise ValueError("GLM-5.3 gathered attention needs a (1, 1, T, 512) latent")
    return latent.shape[2]


def gather_latent(latent, ids):
    """(rows (L, W, D), valid (L, 1, 1, W)) for ids (1, L, W); -1 ids are masked."""
    if ids.ndim != 3 or ids.shape[0] != 1:
        raise ValueError("GLM-5.3 gathered attention needs B1 ids (1, L, W)")
    total = _latent_total(latent)
    if total < 1:
        raise ValueError("GLM-5.3 gathered attention needs at least one latent row")
    ids = ids[0]
    if isinstance(latent, TreeLatentView):
        base, tail, split = latent.base, latent.tail, latent.split
        if tail.shape[0] == 0:
            rows = mx.take(base, mx.clip(ids, 0, split - 1), axis=0)
        elif split == 0:
            rows = mx.take(tail, mx.clip(ids, 0, tail.shape[0] - 1), axis=0)
        else:
            # Both branches copy exact rows; -1 reads base row 0 like the
            # clipped gather from the concatenation, and is masked anyway.
            rows = mx.where(
                (ids < split)[..., None],
                mx.take(base, mx.clip(ids, 0, split - 1), axis=0),
                mx.take(tail, mx.clip(ids - split, 0, tail.shape[0] - 1), axis=0),
            )
    else:
        rows = mx.take(latent[0, 0], mx.clip(ids, 0, total - 1), axis=0)
    valid = (ids >= 0)[:, None, None, :]
    return rows, valid


def _attend(qb, rows, valid):
    """qb (N, H, 1, D), rows (N, W, D), valid (N, 1, 1, W) -> (N, H, 1, D)."""
    mask = mx.where(valid, mx.array(0.0, mx.float32), mx.array(-mx.inf, mx.float32))
    return _attend_masked(qb, rows, mask)


def _attend_masked(qb, rows, mask):
    """``_attend`` with its additive fp32 mask (N, 1, 1, W) already built."""
    keys = rows[:, None]
    return mx.fast.scaled_dot_product_attention(qb, keys, keys, scale=SCALE, mask=mask)


def gathered_attention(q, rows, valid):
    """q (1, H, L, D) over each query's own gathered rows -> (1, H, L, D)."""
    if q.ndim != 4 or q.shape[0] != 1 or rows.shape[0] != q.shape[2]:
        raise ValueError("GLM-5.3 gathered attention query rows and gathered rows differ")
    out = _attend(q.transpose(2, 1, 0, 3), rows, valid)
    return out.transpose(2, 1, 0, 3)


def gathered_attention_masked(q, rows, mask):
    """``gathered_attention`` from rows and the additive mask that
    ``where(valid, 0, -inf)`` builds (glm53_mla_fused's gather emits both);
    the same SDPA call on the same operands, so the same bits."""
    if q.ndim != 4 or q.shape[0] != 1 or rows.shape[0] != q.shape[2] or mask.shape[0] != q.shape[2]:
        raise ValueError("GLM-5.3 gathered attention query rows and gathered rows differ")
    out = _attend_masked(q.transpose(2, 1, 0, 3), rows, mask)
    return out.transpose(2, 1, 0, 3)


def _check_ids(q, ids):
    if ids is None or ids.ndim != 3 or ids.shape[:2] != (1, q.shape[2]):
        raise ValueError("GLM-5.3 sparse attention needs ids (1, L, W) for every query row")


def sparse_decode(q, latent, ids, *, policy, counters=None):
    """Decode-phase consumer (decode, verify rows, MTP head, draft steps); any L.

    Under ``fused_v2`` ``ids`` may be None for dense rows (``T <= 2048``): the
    L query rows are the source's last L positions and get causal ids.
    """
    if policy == "fused_v2":
        return fused_v2_attention(q, latent, ids, counters=counters)
    _check_ids(q, ids)
    if policy == "gathered_v1":
        _count(counters, "sparse_attention_gathered")
        _count(counters, "sparse_gathered_rows", ids.shape[1] * ids.shape[2])
        _SCHEDULED["gathered"] += 1
        if q.shape[2] == 1:
            return gathered_attention(q, *gather_latent(latent, ids))
        # Law V1: every decode-phase row is its own one-query launch (see
        # BATCHED_ROWS_EXACT), so a verify row equals its one-row decode.
        return mx.concatenate(
            [gathered_attention(q[:, :, r : r + 1], *gather_latent(latent, ids[:, r : r + 1]))
             for r in range(q.shape[2])], axis=2)
    if policy == "fused_decode_v1":
        out = fused_decode_attention(q, latent, ids, counters=counters)
        _count(counters, "sparse_attention_fused")
        return out
    raise ValueError(f"GLM-5.3 sparse decode policy {policy!r} is not a gathered law")


def sparse_prefill(q, latent, ids, *, block=PREFILL_BLOCK, counters=None, policy="gathered_v1"):
    """Prompt-phase gathered consumer in blocks of ``block`` queries.

    ``policy="fused_v2"`` (a ``gathered_v2_from_<n>`` value, see
    ``prefill_law``): one fused_v2 launch per block; rows share nothing, so
    no single-row pad is needed and any block size gives the same bits.
    ``ids`` may then be None for dense causal rows (``T <= 2048``).

    Each block is evaluated before the next is built, so one block's gather
    and scores are live at a time (the bound Glm53CacheBudget books; an
    mx.depends chain alone left every block's gather live on Metal). A
    one-query block is padded to two queries: Metal SDPA computes a batch of
    one with another kernel, and the pad keeps every prompt query on the
    batch-of-two-or-more law whatever the chunking (cold/warm exactness).
    The AST route verifies its anchor (the last prompt token) as a decode
    row, so ordinary and AST serving can differ there at a near tie, as the
    dense mask path already does. The per-block eval makes this eager-only:
    do not wrap a multi-block call in mx.compile.
    """
    if type(block) is not int or block < 1:
        raise ValueError("GLM-5.3 sparse prefill block must be a positive integer")
    if policy == "fused_v2":
        return _fused_v2_prefill(q, latent, ids, block, counters)
    if policy == "fused_f16":
        # The fp16-operand kernel: half q rows and one half copy of the chunk's
        # visible latent rows per call (the cached plane stays fp32).
        return _fused_v2_prefill(q, latent, ids, block, counters, half=True)
    if policy != "gathered_v1":
        raise ValueError(f"GLM-5.3 sparse prefill policy {policy!r} is not a gathered law")
    _check_ids(q, ids)
    length = q.shape[2]
    pieces = []
    for start in range(0, length, block):
        stop = min(start + block, length)
        piece_q, piece_ids = q[:, :, start:stop], ids[:, start:stop]
        single = stop - start == 1
        if single:
            piece_q = mx.concatenate([piece_q, piece_q], axis=2)
            piece_ids = mx.concatenate([piece_ids, piece_ids], axis=1)
        rows, valid = gather_latent(latent, piece_ids)
        piece = gathered_attention(piece_q, rows, valid)
        if single:
            piece = piece[:, :, :1]
        if length > block:
            mx.eval(piece)
        pieces.append(piece)
    _count(counters, "sparse_prefill_gathered_blocks", len(pieces))
    _count(counters, "sparse_gathered_rows", length * ids.shape[2])
    _SCHEDULED["prefill_blocks"] += len(pieces)
    return pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=2)


def prefill_gathered_start(total, length, threshold):
    """First query index of a chunk that takes the gathered prefill law.

    The rule is per query position (``pos + 1 >= threshold``, design 5.1), so
    a chunk straddling the threshold runs [0, start) dense and [start, L)
    gathered, and cold/warm chunkings put each query under the same law.
    Returns ``length`` when no query is gathered (including ``total <= 2048``,
    where no ids exist).
    """
    if type(total) is not int or type(length) is not int or not 0 <= length <= total:
        raise ValueError("GLM-5.3 prefill split needs integer total >= length >= 0")
    if threshold is None or total <= TOPK:
        return length
    return min(length, max(0, threshold - 1 - (total - length)))


def rows_attention(q_rows, sources, *, policy, counters=None):
    """Verify/lane rows (7.1): q_rows (R, H, 1, D); sources [(latent_r, ids_r (1, 1, W))].

    Each row's arithmetic is that of its one-row decode (law V1).
    """
    if q_rows.ndim != 4 or q_rows.shape[2] != 1 or q_rows.shape[0] != len(sources) or not sources:
        raise ValueError("GLM-5.3 rows attention needs q_rows (R, H, 1, D) and R sources")
    for _, ids in sources:
        if ids is None or ids.ndim != 3 or ids.shape[:2] != (1, 1):
            raise ValueError("GLM-5.3 rows attention needs ids (1, 1, W) per row")
    if len({ids.shape[2] for _, ids in sources}) != 1:
        raise ValueError("GLM-5.3 rows attention needs one selection width")
    if policy == "fused_v2":
        return _fused_v2_rows(q_rows, sources, counters)
    if policy == "fused_decode_v1":
        # One launch per row: one latent source each (4.3). A (1, H, 1, D)
        # row is its own one-query decode layout.
        out = [fused_decode_attention(q_rows[r : r + 1], latent, ids, counters=counters)
               for r, (latent, ids) in enumerate(sources)]
        _count(counters, "sparse_attention_fused", len(sources))
        return mx.concatenate(out, axis=0)
    if policy != "gathered_v1":
        raise ValueError(f"GLM-5.3 rows attention policy {policy!r} is not a gathered law")
    gathered = [gather_latent(latent, ids) for latent, ids in sources]
    _count(counters, "sparse_gathered_rows", len(sources) * sources[0][1].shape[2])
    if not BATCHED_ROWS_EXACT:
        _count(counters, "tree_rows_attention_per_row_fallback")
        _SCHEDULED["rows_per_row"] += 1
        return mx.concatenate(
            [_attend(q_rows[r : r + 1], rows, valid) for r, (rows, valid) in enumerate(gathered)], axis=0
        )
    _count(counters, "tree_rows_attention_batched")
    _SCHEDULED["rows_batched"] += 1
    rows = mx.concatenate([g[0] for g in gathered], axis=0)
    valid = mx.concatenate([g[1] for g in gathered], axis=0)
    return _attend(q_rows, rows, valid)


# fused_v2 orchestration (glm53_sparse_attention_v2 holds the kernels).
PREFILL_BLOCK_V2 = 64  # 278 MB of partials per block (fused_v2_workspace_bytes)


def _v2():
    from . import glm53_sparse_attention_v2

    return glm53_sparse_attention_v2


def _v2_source(latent):
    """(base (T0, 512), tail (d, 512) or None) of one latent source."""
    if isinstance(latent, TreeLatentView):
        return latent.base, (latent.tail if latent.tail.shape[0] else None)
    _latent_total(latent)
    return latent[0, 0], None


def _v2_buffers(base, tail):
    # No zero-size buffers: T0 = 0 never reads base, d = 0 never reads tails.
    if base.shape[0] == 0:
        base = mx.zeros((1, LATENT), mx.float32)
    return base, (mx.zeros((1, LATENT), mx.float32) if tail is None else tail)


def _v2_rows_ids(q, latent, ids):
    """(flat ids (L, W), total) for q (1, 64, L, 512); dense causal ids when ids is None."""
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != 64 or q.shape[3] != LATENT:
        raise ValueError("GLM-5.3 fused_v2 is built for q (1, 64, L, 512)")
    total = _latent_total(latent)
    length = q.shape[2]
    if ids is None:
        if total > TOPK or length > total:
            raise ValueError("GLM-5.3 fused_v2 dense rows need T <= 2048 and L <= T")
        return dense_row_ids(range(total - length, total)), total
    _check_ids(q, ids)
    return ids[0], total


def _fused_v2_launch_source(q, base, tail, flat_ids, positions, counters, half=False):
    """One launch over prepared source buffers (``_v2_source`` output, cast by the caller)."""
    v2 = _v2()
    v2.require_metal()
    split, depth = base.shape[0], (0 if tail is None else tail.shape[0])
    base, tails = _v2_buffers(base, tail)
    if half:
        base, tails = base.astype(mx.float16), tails.astype(mx.float16)
    meta = mx.array([[split, depth, 0, p] for p in positions], dtype=mx.int32)
    out = v2.checked_launch(q[0].transpose(1, 0, 2), base, tails, flat_ids, meta, half=half)
    law = "fused_f16" if half else "fused_v2"
    _count(counters, f"sparse_attention_{law}")
    _count(counters, f"sparse_attention_{law}_rows", len(positions))
    return out.transpose(1, 0, 2)[None]


def _fused_v2_launch(q, latent, flat_ids, positions, counters):
    base, tail = _v2_source(latent)
    return _fused_v2_launch_source(q, base, tail, flat_ids, positions, counters)


def fused_v2_attention(q, latent, ids, *, counters=None):
    """fused_v2 over one latent source, all L rows in one launch: (1, 64, L, 512)."""
    flat, total = _v2_rows_ids(q, latent, ids)
    length = q.shape[2]
    return _fused_v2_launch(q, latent, flat, list(range(total - length, total)), counters)


def _fused_v2_prefill(q, latent, ids, block, counters, half=False):
    flat, total = _v2_rows_ids(q, latent, ids)
    length = q.shape[2]
    positions = list(range(total - length, total))
    base, tail = _v2_source(latent)
    if half:
        # One half copy of the source per call: every block of this chunk
        # reads it (the first block's eval materializes it once).
        q = q.astype(mx.float16)
        base = base.astype(mx.float16)
        tail = None if tail is None else tail.astype(mx.float16)
    pieces = []
    for start in range(0, length, block):
        stop = min(start + block, length)
        piece = _fused_v2_launch_source(q[:, :, start:stop], base, tail, flat[start:stop], positions[start:stop],
                                        counters, half)
        if length > block:
            mx.eval(piece)  # one block's partials live at a time
        pieces.append(piece)
    _count(counters, "sparse_prefill_fused_f16_blocks" if half else "sparse_prefill_fused_v2_blocks", len(pieces))
    _SCHEDULED["prefill_blocks"] += len(pieces)
    return pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=2)


def _fused_v2_rows(q_rows, sources, counters):
    """One launch when every source is a TreeLatentView over the same committed
    base (same origin and split, tails_v1); otherwise one launch per row with
    the row's own source (the clone frontier), counted."""
    views = [latent for latent, _ in sources]
    first = views[0]
    shared = all(
        isinstance(v, TreeLatentView) and v.origin is not None and v.origin is first.origin and v.split == first.split
        for v in views
    )
    if not shared:
        _count(counters, "sparse_attention_fused_v2_per_row_fallback")
        return mx.concatenate(
            [fused_v2_attention(q_rows[r : r + 1], latent, ids, counters=counters)
             for r, (latent, ids) in enumerate(sources)], axis=0)
    v2 = _v2()
    v2.require_metal()
    tails, meta, offset = [], [], 0
    for view in views:
        depth = view.tail.shape[0]
        if depth:
            tails.append(view.tail)
        meta.append([view.split, depth, offset, view.split + depth - 1])
        offset += depth
    base, tail = _v2_buffers(first.base, mx.concatenate(tails, axis=0) if tails else None)
    flat = mx.concatenate([ids[0] for _, ids in sources], axis=0)
    out = v2.checked_launch(q_rows[:, :, 0], base, tail, flat, mx.array(meta, dtype=mx.int32))
    _count(counters, "sparse_attention_fused_v2")
    _count(counters, "sparse_attention_fused_v2_rows", len(sources))
    return out[:, :, None]


# Optional fused decode kernel mlx2_glm53_sparse_decode_v1 (design 4.3).
# Partial pass: threadgroup (split s, head group g, query row z) stages 8
# gathered rows at a time (-1 rows skipped); each simdgroup scores one head
# with lane-strided 16-dim partial dots and simd_sum, online softmax in fp32,
# scale applied before the exponent. An empty split writes (-inf, 0, 0).
_PARTIAL_SOURCE = r"""
const uint split = threadgroup_position_in_grid.x;
const uint group = threadgroup_position_in_grid.y;
const uint row = threadgroup_position_in_grid.z;
const uint tid = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint head = group * (TG / 32) + simdgroup_index_in_threadgroup;
const int T0 = meta[0];
const int D = meta[1];
const int W = meta[2];
const int LAST_ID = T0 + D - 1;
threadgroup float stage[STAGE * DIM];
threadgroup int sid[STAGE];
float qv[DIM / 32];
float acc[DIM / 32];
for (uint i = 0; i < DIM / 32; i++) {
    qv[i] = q[(size_t(row) * HEADS + head) * DIM + lane + 32 * i];
    acc[i] = 0.0f;
}
float m = -INFINITY;
float l = 0.0f;
const int first = int(split) * SPLIT;
const int last = min(first + SPLIT, W);
for (int k0 = first; k0 < last; k0 += STAGE) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < STAGE) {
        const int k = k0 + int(tid);
        sid[tid] = k < last ? min(ids[size_t(row) * W + k], LAST_ID) : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < STAGE * DIM; e += TG) {
        const int id = sid[e / DIM];
        if (id >= 0) {
            const uint c = e % DIM;
            stage[e] = id < T0 ? base[size_t(id) * DIM + c] : tail[size_t(id - T0) * DIM + c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0; r < STAGE; r++) {
        if (sid[r] < 0) continue;
        float part = 0.0f;
        for (uint i = 0; i < DIM / 32; i++) part += qv[i] * stage[r * DIM + lane + 32 * i];
        const float score = simd_sum(part) * SCALE;
        // A -inf score has zero weight, as in the ops softmax; skipping it
        // keeps a later finite score from meeting exp(-inf - -inf). A split
        // of only -inf scores stays empty (l = 0).
        if (score == -INFINITY) continue;
        // NaN propagates like the ops path (max() would drop it).
        const float next = isnan(score) ? score : max(m, score);
        const float a = metal::precise::exp(m - next);
        const float p = metal::precise::exp(score - next);
        l = l * a + p;
        for (uint i = 0; i < DIM / 32; i++) acc[i] = acc[i] * a + p * stage[r * DIM + lane + 32 * i];
        m = next;
    }
}
const size_t slot = (size_t(row) * HEADS + head) * SPLITS + split;
if (lane == 0) {
    partial_m[slot] = m;
    partial_l[slot] = l;
}
for (uint i = 0; i < DIM / 32; i++) partial_o[slot * DIM + lane + 32 * i] = acc[i];
"""

# Combine in ascending split order: m = max over non-empty splits,
# a_s = exp(m_s - m) (0 for empty), out = sum a_s o_s / sum a_s l_s.
# No valid id anywhere -> l = 0 -> 0/0 = NaN (the shared failure contract).
_COMBINE_SOURCE = r"""
const uint c = thread_position_in_grid.x;
const uint head = thread_position_in_grid.y;
const uint row = thread_position_in_grid.z;
const size_t slots = (size_t(row) * HEADS + head) * SPLITS;
float m = -INFINITY;
for (uint s = 0; s < SPLITS; s++)
    if (partial_l[slots + s] != 0.0f) m = max(m, partial_m[slots + s]);
float l = 0.0f;
float acc = 0.0f;
for (uint s = 0; s < SPLITS; s++) {
    const float ls = partial_l[slots + s];
    const float a = ls != 0.0f ? metal::precise::exp(partial_m[slots + s] - m) : 0.0f;
    l += a * ls;
    acc += a * partial_o[(slots + s) * DIM + c];
}
out[(size_t(row) * HEADS + head) * DIM + c] = acc / l;
"""

_HEADER = "constant constexpr float SCALE = 0.0625f;  // 256 ** -0.5, exact\n"


def kernel_spec():
    return {
        "partial": {"name": "mlx2_glm53_sparse_decode_v1", "input_names": ["q", "base", "tail", "ids", "meta"],
                    "output_names": ["partial_m", "partial_l", "partial_o"], "header": _HEADER,
                    "source": _PARTIAL_SOURCE},
        "combine": {"name": "mlx2_glm53_sparse_decode_combine_v1",
                    "input_names": ["partial_m", "partial_l", "partial_o"], "output_names": ["out"],
                    "header": _HEADER, "source": _COMBINE_SOURCE},
    }


def launch_plan(rows, width):
    """Grid of one fused launch over ``rows`` (<= 8) query rows and ``width`` ids."""
    if type(rows) is not int or not 1 <= rows <= FUSED_GROUP or type(width) is not int or width < 1:
        raise ValueError("GLM-5.3 fused decode launch needs 1..8 rows and a positive width")
    splits = -(-width // FUSED_SPLIT)
    groups = FUSED_HEADS // (FUSED_THREADS // 32)
    return {
        "splits": splits,
        "partial": {"grid": (FUSED_THREADS * splits, groups, rows), "threadgroup": (FUSED_THREADS, 1, 1),
                    "output_shapes": [(rows, FUSED_HEADS, splits)] * 2 + [(rows, FUSED_HEADS, splits, LATENT)]},
        "combine": {"grid": (LATENT, FUSED_HEADS, rows), "threadgroup": (LATENT, 1, 1),
                    "output_shapes": [(rows, FUSED_HEADS, LATENT)]},
        "template": [("DIM", LATENT), ("HEADS", FUSED_HEADS), ("SPLIT", FUSED_SPLIT), ("STAGE", FUSED_STAGE),
                     ("TG", FUSED_THREADS), ("SPLITS", splits)],
    }


@lru_cache(maxsize=2)
def _kernel(kind):
    spec = kernel_spec()[kind]
    return mx.fast.metal_kernel(**spec, ensure_row_contiguous=True)


def _require_metal():
    if not mx.metal.is_available() or mx.default_device() != mx.gpu:
        raise RuntimeError("GLM-5.3 fused_decode_v1 requires Metal on the default GPU device; no fallback law")


def _fused_sources(latent):
    if isinstance(latent, TreeLatentView):
        base, tail = latent.base, latent.tail
    else:
        _latent_total(latent)
        base, tail = latent[0, 0], mx.zeros((0, LATENT), mx.float32)
    depth = tail.shape[0]
    if depth == 0:
        tail = mx.zeros((1, LATENT), mx.float32)  # no zero-size buffers; meta d = 0 never reads it
    return base.astype(mx.float32), tail.astype(mx.float32), base.shape[0], depth


def _fused_launch(q_rows, base, tail, ids, meta):
    rows, width = ids.shape
    plan = launch_plan(rows, width)
    m, l, o = _kernel("partial")(
        inputs=[q_rows, base, tail, ids, meta], template=plan["template"],
        grid=plan["partial"]["grid"], threadgroup=plan["partial"]["threadgroup"],
        output_shapes=plan["partial"]["output_shapes"], output_dtypes=[mx.float32] * 3)
    (out,) = _kernel("combine")(
        inputs=[m, l, o], template=plan["template"],
        grid=plan["combine"]["grid"], threadgroup=plan["combine"]["threadgroup"],
        output_shapes=plan["combine"]["output_shapes"], output_dtypes=[mx.float32])
    _SCHEDULED["fused_launches"] += 1
    return out


def _fused_unchecked(q, latent, ids, counters=None):
    base, tail, split, depth = _fused_sources(latent)
    length, width = ids.shape[1], ids.shape[2]
    meta = mx.array([split, depth, width], dtype=mx.int32)
    q_rows = q[0].transpose(1, 0, 2).astype(mx.float32)  # (L, 64, 512)
    flat_ids = ids[0].astype(mx.int32)
    out = []
    for start in range(0, length, FUSED_GROUP):
        stop = min(start + FUSED_GROUP, length)
        out.append(_fused_launch(q_rows[start:stop], base, tail, flat_ids[start:stop], meta))
        _count(counters, "sparse_attention_fused_launches")
    out = out[0] if len(out) == 1 else mx.concatenate(out, axis=0)
    return out.transpose(1, 0, 2)[None]


def fused_decode_attention(q, latent, ids, *, counters=None):
    """Fused gather + flash decode; q (1, 64, L, 512) -> (1, 64, L, 512). Fails closed."""
    _check_ids(q, ids)
    if q.shape[1] != FUSED_HEADS or q.shape[3] != LATENT:
        raise ValueError("GLM-5.3 fused decode kernel is built for 64 heads of 512")
    _require_metal()
    self_check()
    return _fused_unchecked(q, latent, ids, counters)


def _gate_ids(rng, total, width, rows):
    import numpy as np

    ids = np.full((rows, width), -1, dtype=np.int32)
    filled = rows - 1 if rows > 1 else rows  # with several rows the last stays empty: the NaN contract
    for r in range(filled):
        count = width if r == 0 else width // 2
        ids[r, :count] = np.sort(rng.choice(total, size=count, replace=False))
    return mx.array(ids[None])


def self_check():
    """First-use GPU gate: within 1e-5 absolute of gathered_attention, NaN rows kept."""
    import numpy as np

    if _FUSED_GATE["passed"] is True:
        return True
    if _FUSED_GATE["passed"] is False:
        raise RuntimeError("GLM-5.3 fused decode kernel failed its first-use gate")
    _require_metal()
    rng = np.random.default_rng(5302)
    total, width = 4096, TOPK + 3
    ok = True
    for split in (total, total - 5):
        latent = mx.array(rng.normal(size=(1, 1, total, LATENT)).astype(np.float32))
        source = latent if split == total else TreeLatentView(latent[0, 0, :split], latent[0, 0, split:], split)
        for rows in (1, 3, 9):
            q = mx.array(rng.normal(size=(1, FUSED_HEADS, rows, LATENT)).astype(np.float32))
            ids = _gate_ids(rng, total, width, rows)
            got = _fused_unchecked(q, source, ids)
            want = gathered_attention(q, *gather_latent(source, ids))
            mx.eval(got, want)
            g, w = np.array(got), np.array(want)
            empty = (np.array(ids)[0] < 0).all(axis=-1)
            finite = ~empty
            ok &= bool(np.isnan(g[:, :, empty]).all() and np.isnan(w[:, :, empty]).all())
            ok &= bool(np.isfinite(g[:, :, finite]).all())
            ok &= bool(np.max(np.abs(g[:, :, finite] - w[:, :, finite])) <= 1e-5)
    _FUSED_GATE["passed"] = ok
    if not ok:
        raise RuntimeError("GLM-5.3 fused decode kernel failed its first-use gate")
    return True


class Glm53MLAFrontier:
    """A verify path's MLA state: committed base plus private tail rows (7.3).

    ``base`` is never written. ``base_tokens`` is the integer frontier the
    tree passes from its KDA offset; an MLA offset may be an array in
    serving and is never read for it.
    """

    def __init__(self, base, base_tokens, *, tail_latent=None, tail_raw=None, pools=None):
        if type(base_tokens) is not int or base_tokens < 0:
            raise ValueError("GLM-5.3 MLA frontier needs an integer base frontier")
        offset = getattr(base, "offset", None)
        if type(offset) is int and offset != base_tokens:
            raise ValueError("GLM-5.3 MLA frontier base offset differs from the tree frontier")
        latent_plane = base.caches[0]
        if not hasattr(latent_plane, "keys") or (
            latent_plane.keys is not None and latent_plane.keys.dtype != mx.float32
        ):
            raise ValueError("GLM-5.3 MLA frontier requires the fp32 latent plane")
        self.base = base
        self.base_tokens = base_tokens
        self.tail_latent = mx.zeros((0, LATENT), mx.float32) if tail_latent is None else tail_latent
        self.tail_raw = mx.zeros((0, RAW), mx.float32) if tail_raw is None else tail_raw
        if self.tail_latent.shape[0] != self.tail_raw.shape[0]:
            raise ValueError("GLM-5.3 MLA frontier tails differ in length")
        if pools is None:
            if getattr(base, "pools", None) is None:
                from .glm53_gguf_indexer import Glm53PoolCache

                base.pools = Glm53PoolCache()
            pools = base.pools.clone()
        self.pools = pools

    @property
    def depth(self):
        return self.tail_latent.shape[0]

    @property
    def offset(self):
        return self.base_tokens + self.depth

    @property
    def state(self):
        leaves = [self.tail_latent, self.tail_raw]
        pooled = getattr(self.pools, "pooled", None)
        return leaves if pooled is None else leaves + [pooled]

    def _base_rows(self, plane, width, stop):
        keys = self.base.caches[plane].keys
        if keys is None:
            if self.base_tokens:
                raise ValueError("GLM-5.3 MLA frontier base plane is empty")
            return mx.zeros((0, width), mx.float32)
        return keys[0, 0, :stop]

    def append(self, latent_row, raw_row):
        """Append one row on both planes, in place (callers keep this object)."""
        latent_row = latent_row.reshape(-1, LATENT)
        raw_row = raw_row.reshape(-1, RAW)
        if latent_row.shape[0] != 1 or raw_row.shape[0] != 1:
            raise ValueError("GLM-5.3 MLA frontier appends one row per verify step")
        self.tail_latent = mx.concatenate([self.tail_latent, latent_row.astype(mx.float32)], axis=0)
        self.tail_raw = mx.concatenate([self.tail_raw, raw_row.astype(mx.float32)], axis=0)

    def latent_source(self):
        return TreeLatentView(self._base_rows(0, LATENT, self.base_tokens), self.tail_latent, self.base_tokens,
                              origin=self.base)

    def raw_rows(self, start, stop):
        """Raw indexer rows [start, stop) as (stop - start, 256): base then tail."""
        if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= self.offset:
            raise ValueError("GLM-5.3 MLA frontier raw row range is outside the frontier")
        split = self.base_tokens
        pieces = []
        if start < split:
            pieces.append(self._base_rows(1, RAW, split)[start : min(stop, split)])
        if stop > split:
            pieces.append(self.tail_raw[max(start, split) - split : stop - split])
        if not pieces:
            return mx.zeros((0, RAW), mx.float32)
        return pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=0)

    def dense_keys(self):
        """(1, 1, T0 + d, 512) for the dense causal path when no ids exist."""
        rows = mx.concatenate([self._base_rows(0, LATENT, self.base_tokens), self.tail_latent], axis=0)
        return rows[None, None]

    def fork(self):
        """The tree's private successor: same base, aliased tails, cloned pools."""
        return type(self)(self.base, self.base_tokens, tail_latent=self.tail_latent,
                          tail_raw=self.tail_raw, pools=self.pools.clone())

    def materialize(self, clone=None):
        """A real Glm53MLACache: clone(base), then one-row appends of each tail row.

        Row at a time reproduces the clone-and-append allocation sequence, so
        ``state`` buffers (shapes included) and ``meta_state`` match it.
        ``clone`` maps a cache list to descriptor clones (default: the APCv2
        COW snapshot, as glm53_ast_target.clone).
        """
        from .glm53_gguf import Glm53MLACache

        if not isinstance(self.base, Glm53MLACache):
            raise TypeError("GLM-5.3 MLA frontier base is not a Glm53MLACache")
        if clone is None:
            from ..cow_cache import snapshot_prompt_cache_descriptors

            def clone(caches):
                return snapshot_prompt_cache_descriptors(caches)[0]

        result = clone([self.base])[0]
        dtypes = [
            mx.float32 if plane.values is None else plane.values.dtype for plane in result.caches
        ]
        for row in range(self.depth):
            for plane, tail, dtype in zip(result.caches, (self.tail_latent, self.tail_raw), dtypes, strict=True):
                plane.update_and_fetch(tail[row : row + 1][None, None], mx.zeros((1, 1, 1, 0), dtype))
        if self.pools is not None:
            result.pools = self.pools.clone()
        return result


def _v2_status():
    import sys

    module = sys.modules.get(__name__.rsplit(".", 1)[0] + ".glm53_sparse_attention_v2")
    return {"first_use_gate": "not_run"} if module is None else module.route_status()


def route_status():
    gate = _FUSED_GATE["passed"]
    return {
        "policies": {
            "gathered_v1": {"implemented": True, "qualified": False, "selected_by_default": True},
            "fused_decode_v1": {"implemented": True, "qualified": False, "selected_by_default": False,
                                "first_use_gate": "not_run" if gate is None else ("passed" if gate else "failed")},
            "fused_v2": {"implemented": True, "qualified": False, "selected_by_default": False,
                         **_v2_status()},
            # Prompt-only (gathered_f16_from_<n>); its gate is keyed "_f16" in the v2 status.
            "fused_f16": {"implemented": True, "qualified": False, "selected_by_default": False,
                          "prompt_only": True},
        },
        "batched_rows_exact": BATCHED_ROWS_EXACT,
        "scheduled_calls": dict(_SCHEDULED),
    }
