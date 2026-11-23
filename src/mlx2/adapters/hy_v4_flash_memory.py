"""Hy4-flash lane cache projection, decode/prefill transients and media bounds.

Without a projection, lane admission charged Hy4-flash the controller's
full-attention envelope, 0.44 GiB per 1K tokens: 14.2 GiB for a 32K request
whose retained state is about 1.2 GiB.  That envelope, not the model,
refused 32K at an 8 GiB service reserve and 64K at 4 GiB
(docs/hy4-flash/context-128k.md).

``project`` is the bytes the lane's cache arrays hold after ``context``
tokens, read off the cache classes in ``runtime/models/hy_v4_flash.py``:

* indexed (sparse) layers keep three append-only planes: ``kv`` (latent ‖
  RoPE row, 1152 B native bf16 or 656 B packed fp8), ``tk`` (indexer token
  keys, activation dtype) and ``gk`` (pooled group keys, fp32, one per
  ``ratio`` tokens).  Each grows in 256-row steps to exactly
  ``ceil(n / 256) * 256`` rows;
* sliding layers keep a ring of ``window - 1 + ROLLBACK`` rows whose buffer
  is reallocated to ``ceil((retained + chunk) / 256) * 256`` rows by a
  prefill chunk (the server's ``prefill_step``) and is not shrunk until its
  next compaction;
* DFly2 Index Share records the last ``ROLLBACK`` selections of each
  recorded layer.  They are slices of the forward's ``[B, L, K]`` int32
  recall array, so a prefill chunk pins the whole chunk's array and decode
  rounds pin one ``[B, verify_rows, K]`` array per step.

The drafter keeps no KV of its own: each round reads the target's latent rows
at the recalled positions.

``decode_transient_bytes`` (context-dependent, charged on every cycle through
the controller's ``context_transient_estimator``) is what one verify round
allocates above the retained cache: every 256 rows ``_buf_append`` reallocates
the append-only planes while the live cache (or an APCv2 checkpoint, or a
recovery point) keeps the old buffer -- one full copy; each child chain of a
tree tile aliases its parent's buffers and its first append copies them all
(append-only planes and sliding rings) -- one more copy per child chain; plus
the verify forward's own working set.  The plain ``transient_gib_per_lane``
stays the controller's calibrated figure on top.

``prefill_transient_bytes`` is a shape bound (not a Metal measurement) on one
prefill chunk's working set: recall masks, the score blocks the attention and
indexer chunk to ``SCORE_BUDGET`` (one block live with
``BOUNDED_SCORE_BLOCKS``; all of them without), projections of the widest
attention layer, routed-expert gathers (fp32 around each gather on ``q2_0``
layers) and the hyper-connection streams, summed as if live at once.  The
append-only copy a chunk's growth makes is charged once, by
``decode_transient_bytes``, on every path that charges a prefill.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

STEP = 256  # runtime/models/hy_v4_flash.py ``_STEP``
PAGE = 16384  # Apple silicon page; MLX buffers are page-rounded


def _rows(n: int) -> int:
    return math.ceil(n / STEP) * STEP if n > 0 else 0


def _blocks(elements: int, rows: int, budget: int) -> int:
    """Score blocks ``mla_attend``/``select`` split ``rows`` query rows into."""
    if rows <= 0:
        return 0
    per_row = max(1, elements // rows)
    step = max(1, budget // per_row)
    return 1 if step >= rows else math.ceil(rows / step)


def expert_row_bytes(fmt: str, *, routed_hidden: int, intermediate: int, item_bytes: int) -> int:
    """Gather working set per (token, routed expert) for one expert format.

    ``affine`` (MLX quantized or dense): the sorted input and the down output
    (activation dtype), gate/up/activation, and the fp32 weighted sum.
    ``q2_0`` (``Q2ZeroSwitchLinear``): each of the three gathers widens its
    input to fp32, returns fp32 and casts back, so every projection holds an
    fp32 input, an fp32 output and the cast.
    """
    a = int(item_bytes)
    if fmt == "q2_0":
        gate_up = 2 * (routed_hidden * 4 + intermediate * 4 + intermediate * a)
        down = intermediate * 4 + routed_hidden * 4 + routed_hidden * a
        return gate_up + down + intermediate * a + routed_hidden * 4
    return 2 * routed_hidden * a + 3 * intermediate * a + routed_hidden * 4


@dataclass(frozen=True)
class HY4FlashCacheBudget:
    indexed_layers: int
    sliding_layers: int
    record_layers: int
    indexed_row_bytes: int
    sliding_row_bytes: int
    index_key_bytes: int
    group_key_bytes: int
    ratio: int
    sliding_window: int
    rollback: int
    candidate_topk: int
    hidden_size: int
    hc_mult: int
    indexed_heads: int
    sliding_heads: int
    indexer_heads: int
    kv_lora_rank: int
    rope_dim: int
    nope_dim: int
    v_dim: int
    experts_per_token: int
    routed_hidden: int
    moe_intermediate: int
    dense_intermediate: int
    item_bytes: int
    score_budget: int
    gather_max_query: int
    # Widest prefill chunk the server runs (its ``prefill_step``; the
    # prompt-length autoscale picks at most this).
    prefill_rows: int = 8192
    # Widest single-lane verify forward of the route (anchor + tile nodes,
    # or num_draft + 1 on a chain); 1 on the ordinary route.
    verify_rows: int = 1
    # Child chains of one lane's tree tile (0 on a chain).
    child_chains: int = 0
    # Widest per-(token, expert) gather working set over the trunk's layers.
    expert_bytes_per_row: int = 0
    # A media (image) prefill runs its layers in slices of this many rows.
    media_prefill_rows: int = 2048
    # Unmeasured on Hy4-flash: the controller's calibrated k=2 figure, which
    # it already charged this adapter before it declared a budget.
    transient_gib_per_lane: float = 1.76

    @classmethod
    def from_model(cls, model, *, prefill_rows: int = 8192, verify_rows: int = 1,
                   child_chains: int = 0):
        """Geometry of a loaded ``runtime.models.hy_v4_flash.Model``."""
        import mlx.core as mx

        from ..runtime.models.hy_v4_flash import Q2ZeroSwitchLinear

        # Rows are stored in the activation dtype: the embedding's output
        # (built lazily, never evaluated).
        item = model.model.embed_tokens(mx.array([[0]], dtype=mx.int32)).dtype.size
        q2 = any(isinstance(module, Q2ZeroSwitchLinear) for _, module in model.named_modules())
        formats = ("affine", "q2_0") if q2 else ("affine",)
        return cls.from_args(model.args, item_bytes=item,
                             record_layers=len(getattr(model, "record_layers", ()) or ()),
                             prefill_rows=prefill_rows, verify_rows=verify_rows,
                             child_chains=child_chains, expert_formats=formats)

    @classmethod
    def from_args(cls, a, *, item_bytes: int, record_layers: int, prefill_rows: int = 8192,
                  verify_rows: int = 1, child_chains: int = 0, expert_formats=("affine",)):
        """Geometry of ``ModelArgs`` with ``item_bytes`` activations."""
        from ..runtime.models import hy_v4_flash as H
        from ..runtime.models.hy_v4_flash_fp8kv import native_row_bytes, row_bytes

        kinds = list(a.layer_types)
        indexed = sum(k == H.INDEXED for k in kinds)
        item = int(item_bytes)
        rope_idx, rope_swa = a.qk_rope_head_dim, a.swa_qk_rope_head_dim
        if a.latent_kv == "fp8":
            idx_row, swa_row = row_bytes(a.kv_lora_rank, rope_idx), row_bytes(a.kv_lora_rank, rope_swa)
        else:
            idx_row = native_row_bytes(a.kv_lora_rank, rope_idx, item)
            swa_row = native_row_bytes(a.kv_lora_rank, rope_swa, item)
        for name, value, low in (("prefill rows", prefill_rows, 1), ("verify rows", verify_rows, 1),
                                 ("child chains", child_chains, 0)):
            if type(value) is not int or value < low:
                raise ValueError(f"Hy4-flash {name} must be an integer >= {low}")
        routed = a.routed_expert_hidden_size or a.hidden_size
        expert_bytes = max(expert_row_bytes(f, routed_hidden=routed,
                                            intermediate=a.moe_intermediate_size, item_bytes=item)
                           for f in expert_formats)
        return cls(
            indexed_layers=indexed,
            sliding_layers=len(kinds) - indexed,
            record_layers=int(record_layers),
            indexed_row_bytes=idx_row,
            sliding_row_bytes=swa_row,
            index_key_bytes=a.compressed_indexer_head_dim * item,
            group_key_bytes=a.compressed_indexer_head_dim * 4,
            ratio=a.compressed_indexer_compress_ratio,
            sliding_window=a.sliding_window,
            rollback=H.ROLLBACK,
            candidate_topk=a.candidate_topk,
            hidden_size=a.hidden_size,
            hc_mult=a.hc_mult,
            indexed_heads=a.num_attention_heads,
            sliding_heads=a.swa_num_attention_heads,
            indexer_heads=a.compressed_indexer_num_attention_heads,
            kv_lora_rank=a.kv_lora_rank,
            rope_dim=max(rope_idx, rope_swa),
            nope_dim=max(a.qk_nope_head_dim, a.swa_qk_nope_head_dim),
            v_dim=max(a.v_head_dim, a.swa_v_head_dim),
            experts_per_token=a.num_experts_per_tok,
            routed_hidden=routed,
            moe_intermediate=a.moe_intermediate_size,
            dense_intermediate=a.intermediate_size,
            item_bytes=item,
            score_budget=H.SCORE_BUDGET,
            gather_max_query=H.GATHER_MAX_QUERY,
            prefill_rows=prefill_rows,
            verify_rows=verify_rows,
            child_chains=child_chains,
            expert_bytes_per_row=expert_bytes,
            media_prefill_rows=H.MEDIA_PREFILL_STEP,
        )

    @staticmethod
    def _count(value):
        if type(value) is not int or value < 0:
            raise ValueError("Hy4-flash token counts must be nonnegative integers")
        return value

    # -- retained state ------------------------------------------------------
    def append_only_bytes(self, context_tokens: int) -> int:
        """Indexed layers' ``kv`` + ``tk`` + ``gk`` buffers at ``context_tokens``."""
        n = self._count(context_tokens)
        per_layer = (_rows(n) * (self.indexed_row_bytes + self.index_key_bytes)
                     + _rows(n // self.ratio) * self.group_key_bytes)
        return self.indexed_layers * per_layer

    def ring_bytes(self, context_tokens: int) -> int:
        n = self._count(context_tokens)
        retained = self.sliding_window - 1 + self.rollback
        widest = max(self.prefill_rows, self.verify_rows)
        return self.sliding_layers * _rows(min(n, retained + widest)) * self.sliding_row_bytes

    def project_parts(self, context_tokens: int) -> dict:
        n = self._count(context_tokens)
        # A recorded layer pins its last prefill chunk's recall array and,
        # once decoding, up to ROLLBACK verify rounds' arrays.  Positions at
        # or below the candidate count are dense steps and record nothing.
        pinned_rows = (min(n, self.prefill_rows) + self.rollback * self.verify_rows
                       if n > self.candidate_topk else 0)
        buffers = 3 * self.indexed_layers + self.sliding_layers + 2 * self.record_layers
        return {
            "append_only": self.append_only_bytes(n),
            "sliding_rings": self.ring_bytes(n),
            "index_share_selections": self.record_layers * pinned_rows * self.candidate_topk * 4,
            # Allocators hand out whole pages: one 16 KiB page of rounding per
            # buffer (and Engram tails, offsets and spans are host objects).
            "allocation_rounding": buffers * PAGE if n else 0,
        }

    def project(self, context_tokens: int) -> int:
        return sum(self.project_parts(context_tokens).values())

    # -- one verify round ----------------------------------------------------
    def decode_transient_parts(self, context_tokens: int) -> dict:
        n = self._count(context_tokens)
        if n == 0:
            return {}
        grown = n + self.verify_rows
        working = self.prefill_transient_parts(grown, self.verify_rows)
        return {
            "append_only_growth_copy": self.append_only_bytes(grown),
            "child_chain_copies": self.child_chains * (
                self.append_only_bytes(grown) + self.ring_bytes(grown)),
            "verify_working_set": sum(working.values()),
        }

    def decode_transient_bytes(self, context_tokens: int) -> int:
        return sum(self.decode_transient_parts(context_tokens).values())

    # -- one prefill chunk ---------------------------------------------------
    def _score_block(self, elements: int) -> int:
        """Live score elements of one chunked forward.

        With ``BOUNDED_SCORE_BLOCKS`` each block is evaluated before the next
        is built, so one ``SCORE_BUDGET`` block is live.  Without it the lazy
        graph holds every block at once: the whole score tensor.
        """
        from ..runtime.models import hy_v4_flash as H

        return min(self.score_budget, elements) if H.BOUNDED_SCORE_BLOCKS else elements

    def prefill_transient_parts(self, context_tokens: int, chunk_rows: int) -> dict:
        n, L = self._count(context_tokens), self._count(chunk_rows)
        if L == 0:
            return {}
        D, s = self.hidden_size, self.hc_mult
        a = self.item_bytes
        K = self.candidate_topk
        groups = n // self.ratio
        swa_keys = min(n, L + self.sliding_window - 1 + self.rollback)
        block = self._score_block
        if n > K and L > self.gather_max_query:
            # Masked path over the whole history: scores, masked scores and
            # softmax per block, and hit (zeros, scatter), causal compare and
            # ``allowed`` as bool [L, T].
            indexed_scores = 3 * 4 * block(self.indexed_heads * L * n)
            recall_masks = 4 * L * (n + 1)
        elif n > K:
            # Gather path: one recalled key set per query row.
            indexed_scores = 3 * 4 * block(self.indexed_heads * L * K)
            recall_masks = L * K * (self.kv_lora_rank + self.rope_dim) * a
        else:
            indexed_scores = 3 * 4 * block(self.indexed_heads * L * n)
            recall_masks = 0
        sliding_scores = 3 * 4 * block(self.sliding_heads * L * swa_keys)
        indexer = (3 * 4 * block(self.indexer_heads * L * max(groups, 1))
                   + 4 * L * K * 4) if n > K else 0
        heads = max(self.indexed_heads, self.sliding_heads)
        projections = L * heads * (2 * (self.nope_dim + self.rope_dim)
                                   + 3 * (self.kv_lora_rank + self.rope_dim)
                                   + 3 * self.v_dim) * a
        experts = L * self.experts_per_token * self.expert_bytes_per_row
        dense = 3 * L * self.dense_intermediate * a
        streams = 8 * L * s * D * 4
        return {
            "attention_scores": max(indexed_scores, sliding_scores),
            "recall_masks": recall_masks,
            "indexer": indexer,
            "projections": projections,
            "routed_experts": max(experts, dense),
            "streams": streams,
        }

    def prefill_transient_bytes(self, context_tokens: int, chunk_rows: int) -> int:
        return sum(self.prefill_transient_parts(context_tokens, chunk_rows).values())

    def media_prefill_transient_bytes(self, context_tokens: int, uncached_tokens: int) -> int:
        """A cold media prefill: one ``uncached_tokens`` chunk run in slices.

        ``Model.hidden`` runs the layers in ``MEDIA_PREFILL_STEP`` slices with
        an evaluation between them, so one slice's working set is live, plus
        the whole tail's embeddings, the features scattered into them, each
        slice's retained output and their concatenation (4 x L x D).
        """
        L = self._count(uncached_tokens)
        if L == 0:
            return 0
        rows = min(L, self.media_prefill_rows)
        return (self.prefill_transient_bytes(self._count(context_tokens), rows)
                + 4 * L * self.hidden_size * self.item_bytes)

    def prefill_score_syncs(self, context_tokens: int, chunk_rows: int) -> int:
        """Block evaluations ``BOUNDED_SCORE_BLOCKS`` adds to one prefill chunk.

        A forward with one block (decode, verify, any chunk whose whole score
        tensor fits ``SCORE_BUDGET``) adds none.
        """
        n, L = self._count(context_tokens), self._count(chunk_rows)
        K = self.candidate_topk
        swa_keys = min(n, L + self.sliding_window - 1 + self.rollback)
        if n > K and L > self.gather_max_query:
            indexed = _blocks(self.indexed_heads * L * n, L, self.score_budget)
        elif n > K:
            indexed = _blocks(self.indexed_heads * L * K, L, self.score_budget)
        else:
            indexed = 1  # "causal" string mask: one SDPA call
        indexer = _blocks(self.indexer_heads * L * max(n // self.ratio, 1), L,
                          self.score_budget) if n > K else 1
        sliding = 1 if L == 1 else _blocks(self.sliding_heads * L * swa_keys, L, self.score_budget)
        extra = lambda blocks: blocks if blocks > 1 else 0  # noqa: E731
        return (self.indexed_layers * (extra(indexed) + extra(indexer))
                + self.sliding_layers * extra(sliding))

    def as_dict(self):
        return dict(
            schema="hy4-flash-cache-budget-v2",
            **asdict(self),
            bound=("retained: exact 256-row capacity of the indexed append-only planes, "
                   "sliding rings at the widest prefill chunk, pinned Index Share recalls"),
            transient_basis=("decode: append-only growth copy + one copy per tree child chain "
                             "+ verify working set (context-dependent); prefill: shape bound of "
                             "one chunk (recall masks, SCORE_BUDGET blocks, projections, expert "
                             "gathers per layer format, streams), not a Metal measurement; "
                             "flat 1.76 GiB/lane controller calibration on top, unmeasured"),
        )


def vision_tower_peak_bytes(args, grids, *, item_bytes: int, score_budget: int) -> int:
    """Working set of one ``VisionTower`` call over ``grids`` (shape bound).

    Per patch: the fp32 pixel row, the residual, norms, q/k/v and attention
    output (8 x D), and the MLP's two ``intermediate`` rows.  Per frame: the
    resampled position table (``[h, E, D]`` and ``[h, w, D]`` fp32).  One
    attention score block (three fp32 copies; ``BOUNDED_SCORE_BLOCKS``) or a
    whole frame's scores without it.  The merged features and the merger's
    ``m*m`` concatenation.
    """
    from ..runtime.models import hy_v4_flash_vision as V

    D, a = args.hidden_size, int(item_bytes)
    E = args.max_image_size // args.patch_size
    m = args.spatial_merge_size
    patches = sum(int(t) * int(h) * int(w) for t, h, w in grids)
    if patches == 0:
        return 0
    per_patch = args.num_channels * args.patch_size ** 2 * 4 + 8 * D * a + 2 * args.intermediate_size * a
    frame = max(int(h) * E * D * 4 + int(h) * int(w) * D * 4 for _, h, w in grids)
    widest = max(int(h) * int(w) for _, h, w in grids)
    scores = args.num_attention_heads * widest * widest
    block = min(score_budget, scores) if V.BOUNDED_SCORE_BLOCKS else scores
    merged = sum(int(t) * (int(h) // m) * (int(w) // m + 1) for t, h, w in grids)
    return (patches * per_patch + frame + 3 * 4 * block
            + merged * (m * m * D + 2 * args.out_hidden_size) * a)


def route_verify_shape(policy, *, draft_selected: bool, max_width: int = 4):
    """``(verify_rows, child_chains)`` of one lane on the selected route.

    Rows: the widest single-lane tile (``ast_tiles x ast_tile_rows``, its
    node budget plus the anchor) or a chain of ``num_draft`` drafts plus the
    anchor, whichever is wider.  Child chains: 0 on a chain, one per extra
    branch of a fixed topology, and every node but the anchor and one leaf
    for a drafter-shaped tree; a grafted prompt-lookup copy adds a branch.
    """
    if not draft_selected:
        return 1, 0
    from ..runtime.ast_tree_proposals import parse_topology
    from .qwen38_27b import _tile_node_budgets

    nodes = _tile_node_budgets(dict(policy), max_width)[1]
    rows = max(nodes + 1, int(policy.get("num_draft", 5)) + 1)
    topology = parse_topology(policy.get("ast_topology", "chain"))
    if topology == "chain":
        children = 0
    elif isinstance(topology, tuple):
        children = min(len(topology) - 1, max(0, rows - 2))
    else:
        children = max(0, rows - 2)
    if policy.get("ast_graft_prompt_lookup"):
        children = max(children, 1)
    return rows, children
