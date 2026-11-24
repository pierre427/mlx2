"""Inkling 2 per-lane cache geometry for admission and APCv2 checkpoint budgets.

Original mlx2 code (2026-11-22), modelled on ``inkling_memory.py`` (the
Inkling 1 budget). It is a shape bound, not a measured speed claim.

The lane cache is one (attention, short-conv) pair per *visit*: 24 layers with
the core (layers 6-17) run three times give 48 visits, 8 of them global
(``KVCache``, grows with context) and 40 sliding (``RotatingKVCache`` of
``sliding_window`` tokens, which holds up to ``window + prefill rows - 1``
while a prefill chunk is in flight). The short-conv cache holds the last
``kernel - 1`` inputs of four convolutions per visit (k and v at
``kv_heads * head_dim`` channels, attention and MLP at ``hidden_size``), in
FP32. On the DSpark route the drafter adds its own three-layer KV (one global
layer, the rest sliding) and the last target feature.

``prefill_transient_bytes`` (``inkling2_prefill_transient_pricing``, default
off; 2026-11-23, MTPLX ingest candidate 11) prices one prefill chunk's working
memory beyond the lane cache from the shapes: see ``transient_terms``.

Checked against the real checkpoint's APCv2 copy-on-write receipts
(qualification/runs/inkling2-gpu-20261122/16eed37f5-dsgsm8k, ~650-token
GSM8K prompts): ring 122 MB, global KV 29 MB, short conv 3.8 MB and draft
9.8 MB per entry, i.e. BF16 K/V, FP32 conv state.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Inkling2CacheBudget:
    global_visits: int
    sliding_visits: int
    kv_heads: int
    head_dim: int
    hidden_size: int
    sliding_window: int
    conv_kernel: int
    draft_global_layers: int = 0
    draft_sliding_layers: int = 0
    draft_kv_heads: int = 0
    draft_head_dim: int = 0
    draft_sliding_window: int = 0
    draft_feature_width: int = 0
    # Upper bound on prefill rows in flight (the ring's transient overhang).
    prefill_rows: int = 2048
    item_bytes: int = 2  # BF16 K/V (the dense checkpoint dtype)
    conv_item_bytes: int = 4  # FP32 short-conv histories
    # inkling2_kv_q8: bytes of one global-visit K (or V) row per KV head (int8
    # codes plus inline FP16 scales); 0 = head_dim * item_bytes.
    global_row_bytes: int = 0
    # Speculation keeps short-conv rollback records for the verify block
    # (DSpark depth up to 8, plus the committed state).
    conv_copies: int = 1
    checkpoint_copies: int = 0
    checkpoint_stride: int = 2048
    verify_rows: int = 0
    allocation_step: int = 256
    transcript_bytes_per_token: int = 16
    # Prefill transient pricing (inkling2_prefill_transient_pricing; 0 = off,
    # the chunk is charged nothing as before). Shapes it is derived from:
    transient_pricing: int = 0
    heads: int = 0
    d_rel: int = 0
    rel_extent: int = 0
    intermediate_size: int = 0
    routed_experts: int = 0
    experts_per_token: int = 0
    residual_block: int = 0
    # The composed attention's mask slice budget (inkling2_prefill_mask_mb;
    # language._MASK_BUDGET_BYTES), counted at 4 bytes per element.
    mask_budget_bytes: int = 64 << 20
    # 0 for a prompt attention that builds no mask, bias or index planes
    # (none ships yet; the composed path builds them).
    mask_planes: int = 1
    # inkling2_prefill_wide_chunk: chunks wider than ``schedule_narrow_rows``
    # run only below ``schedule_crossover`` keys and never cross it; past it a
    # chunk is ``schedule_narrow_rows`` wide (0: a fixed step).
    schedule_crossover: int = 0
    schedule_narrow_rows: int = 0
    # inkling2_native_replay: global K/V buffers are held at whole multiples
    # of this many rows past the fill (capacity_state.CAPACITY_STEP; up to
    # step - 1 spare rows per global visit). 0 = the default growth envelope.
    fixed_capacity_step: int = 0

    @staticmethod
    def _count(value):
        if type(value) is not int or value < 0:
            raise ValueError("Inkling 2 memory dimensions must be nonnegative integers")
        return value

    @classmethod
    def from_config(cls, text, *, visits=None, prefill_rows=2048, draft=None, num_draft=0,
                    kv_q8=False, transient_pricing=False, mask_budget_mb=None,
                    mask_planes=True, schedule=None, fixed_capacity_step=0):
        """``text``: the Inkling 2 ``TextConfig``; ``draft``: the DSpark
        drafter's ``mtp_config`` dict when the DSpark route is selected."""
        visits = text.layer_indices if visits is None else visits
        sliding = sum(1 for index in visits if text.layer_is_sliding(index))
        fields = dict(
            prefill_rows=prefill_rows,
            global_visits=len(visits) - sliding,
            sliding_visits=sliding,
            kv_heads=int(text.num_key_value_heads),
            head_dim=int(text.head_dim),
            hidden_size=int(text.hidden_size),
            sliding_window=int(text.sliding_window_size),
            conv_kernel=int(text.sconv_kernel_size),
            transient_pricing=int(bool(transient_pricing)),
            heads=int(text.num_attention_heads),
            d_rel=int(text.d_rel),
            rel_extent=int(text.rel_extent),
            intermediate_size=int(text.intermediate_size),
            routed_experts=int(text.n_routed_experts),
            experts_per_token=int(text.num_experts_per_tok),
            residual_block=int(text.n_layers_per_attn_res_block),
            mask_planes=int(bool(mask_planes)),
            fixed_capacity_step=int(fixed_capacity_step),
        )
        if mask_budget_mb is not None:
            fields["mask_budget_bytes"] = int(mask_budget_mb) << 20
        if schedule is not None:
            fields["schedule_crossover"], fields["schedule_narrow_rows"] = map(int, schedule)
        if kv_q8:
            from ..runtime.models.inkling2 import kv_q8 as _kv_q8

            fields["global_row_bytes"] = _kv_q8.packed_width(int(text.head_dim))
        if getattr(text, "swa_num_key_value_heads", None) not in (None, fields["kv_heads"]) or (
            getattr(text, "swa_head_dim", None) not in (None, fields["head_dim"])
        ):
            raise ValueError("Inkling 2 budget assumes equal global and sliding K/V geometry")
        if draft is not None:
            layers = int(draft["num_hidden_layers"])
            local = len(draft.get("local_layer_ids") or ())
            fields.update(
                draft_global_layers=layers - local,
                draft_sliding_layers=local,
                draft_kv_heads=int(draft["num_key_value_heads"]),
                draft_head_dim=int(draft["head_dim"]),
                draft_sliding_window=int(draft["sliding_window_size"]),
                draft_feature_width=len(draft["target_layer_ids"]) * int(draft["hidden_size"]),
                conv_copies=1 + max(0, int(num_draft)) + 1,
                verify_rows=max(0, int(num_draft)) + 1,
            )
        else:
            from .mlx_vlm_memory import _mlx2_rotating_checkpoint_policy

            fields["checkpoint_copies"], fields["checkpoint_stride"] = _mlx2_rotating_checkpoint_policy()
        budget = cls(**fields)
        for name, value in asdict(budget).items():
            cls._count(value)
        if not (budget.global_visits or budget.sliding_visits) or not budget.kv_heads:
            raise ValueError("invalid Inkling 2 cache dimensions")
        return budget

    def _capacity(self, n):
        return (math.ceil(n / self.allocation_step) + 1) * self.allocation_step

    @property
    def conv_bytes(self):
        channels = 2 * self.kv_heads * self.head_dim + 2 * self.hidden_size
        return (
            (self.global_visits + self.sliding_visits)
            * max(0, self.conv_kernel - 1)
            * channels
            * self.conv_item_bytes
            * self.conv_copies
        )

    def project(self, context_tokens):
        """Upper bound on one lane's cache bytes at ``context_tokens``."""
        n = self._count(context_tokens)
        capacity = self._capacity(n)
        token = 2 * self.kv_heads * self.head_dim * self.item_bytes
        global_token = 2 * self.kv_heads * (self.global_row_bytes or self.head_dim * self.item_bytes)
        ring = min(capacity, self.sliding_window + self.prefill_rows)
        # Ordinary prefill retains stride-spaced window snapshots and can
        # force one conv snapshot at each turn end. DSpark instead holds the
        # pre-verify ring plus the current window and the new rollback rows.
        windows = min(self.checkpoint_copies, n // self.checkpoint_stride)
        saved_ring = windows * min(self.sliding_window, n)
        if self.verify_rows:
            saved_ring += min(n, self.sliding_window + 2 * self.verify_rows)
        saved_conv = min(n, self.checkpoint_copies) * self.conv_bytes
        global_capacity = capacity
        if self.fixed_capacity_step:
            # The decode step's fixed layout: ceil((n + 1) / step) * step rows.
            step = self.fixed_capacity_step
            global_capacity = max(capacity, -(-(n + 1) // step) * step)
        total = (
            self.global_visits * global_capacity * global_token
            + self.sliding_visits * (ring + saved_ring) * token
            + self.conv_bytes + saved_conv
            + capacity * self.transcript_bytes_per_token
        )
        if self.draft_global_layers or self.draft_sliding_layers:
            draft_token = 2 * self.draft_kv_heads * self.draft_head_dim * self.item_bytes
            draft_ring = min(capacity, self.draft_sliding_window + self.prefill_rows)
            total += (
                self.draft_global_layers * capacity * draft_token
                + self.draft_sliding_layers * draft_ring * draft_token
                + self.draft_feature_width * 4
            )
        return int(total)

    def transient_terms(self, context_tokens, chunk_rows):
        """One prefill chunk's working memory beyond the lane cache, by term
        (bytes), for ``chunk_rows`` rows ending at ``context_tokens`` keys.

        A shape bound for the route as it runs (B = 1 lane; intermediates
        counted at FP32 where the code may widen them):

        * ``residual``: the attention-residual block states a forward keeps
          (one per ``residual_block`` visits, plus the partial, the prelude
          copies and the loop's message bank), [rows, hidden] each.
        * ``attention``: one visit's projections (q, gate, k, v, r, sink) with
          their FP32 clip copies, and the output with the FP32 gate product.
        * ``mask``: the dense relative-bias mask slices (the gathered bias, the
          select, the sink concatenation and the per-slice bias table), each
          slice bounded by the mask budget, plus the forward-memoized index
          planes (int32 gather index, two bool bands: 6 bytes per row per
          key) of the global and the sliding visits. Zero for a mask-free
          attention (``mask_planes`` 0; none ships yet).
        * ``moe``: one visit's routed experts (router logits, the sorted
          inputs, gate/up, the activation, the down product, the combine).
        * ``growth``: a global visit's K/V buffer while the cache grows it
          (old and new alive during the copy).

        The vocabulary head is not evaluated during chunked prefill (the
        engine evaluates the cache state only), so it is not charged."""
        n, rows = self._count(context_tokens), self._count(chunk_rows)
        keys = max(n, 1)
        hidden, heads, dim = self.hidden_size, self.heads, self.head_dim
        visits = self.global_visits + self.sliding_visits
        states = visits // max(1, self.residual_block) + 4
        residual = states * rows * hidden * 4
        project_cols = 2 * heads * dim + 2 * self.kv_heads * dim + heads * self.d_rel + heads
        attention = rows * (project_cols * (2 + 4) + heads * dim * (2 + 4 + 4))
        mask = 0
        if self.mask_planes:
            budget_elems = max(1, self.mask_budget_bytes // 4)
            local_keys = min(keys, self.sliding_window + rows - 1) + 1
            visible = ([keys + 1] * bool(self.global_visits)
                       + [local_keys] * bool(self.sliding_visits))
            widest = max(visible) if visible else 0
            slice_rows = min(rows, max(1, budget_elems // max(1, heads * widest)))
            slice_elems = heads * slice_rows * widest
            table = heads * slice_rows * max(1, self.rel_extent) * 2
            mask = 4 * slice_elems * 2 + table + sum(6 * rows * width for width in visible)
        top = self.experts_per_token
        moe = rows * (
            self.routed_experts * 4 * 2
            + top * hidden * 2
            + top * 2 * self.intermediate_size * 4
            + top * self.intermediate_size * 4
            + top * hidden * 4
            + hidden * 4
        )
        growth = 0
        if self.global_visits:
            row = self.global_row_bytes or self.head_dim * self.item_bytes
            growth = 2 * 2 * self._capacity(n) * self.kv_heads * row
        return {"residual": residual, "attention": attention, "mask": mask, "moe": moe,
                "growth": growth}

    def prefill_transient_bytes(self, context_tokens, chunk_rows):
        """The chunk transient admission charges (0 unless pricing is on).

        Under a width schedule, the largest chunk the prompt can actually run:
        the max over the schedule's cases, a ``chunk_rows`` chunk ending at
        most at the crossover and a narrow chunk at the full depth."""
        n, rows = self._count(context_tokens), self._count(chunk_rows)
        if not self.transient_pricing:
            return 0

        def cost(keys, width):
            return int(sum(self.transient_terms(keys, width).values()))

        narrow = self.schedule_narrow_rows
        if not self.schedule_crossover or rows <= narrow:
            return cost(n, rows)
        charge = cost(min(n, self.schedule_crossover), rows)
        if n > self.schedule_crossover:
            charge = max(charge, cost(n, narrow))
        return charge

    def as_dict(self):
        return dict(
            schema="inkling2-visit-kv-shortconv-geometry-v1",
            **asdict(self),
            conv_bytes=self.conv_bytes,
            bound="BF16 K/V per visit (global grows, sliding rings plus prefill overhang), "
                  "FP32 short-conv histories, DSpark drafter KV when selected",
            basis="shape bound checked against APCv2 copy-on-write plane bytes; not a speed claim",
        )
