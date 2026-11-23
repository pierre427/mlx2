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
    # Speculation keeps short-conv rollback records for the verify block
    # (DSpark depth up to 8, plus the committed state).
    conv_copies: int = 1
    checkpoint_copies: int = 0
    checkpoint_stride: int = 2048
    verify_rows: int = 0
    allocation_step: int = 256
    transcript_bytes_per_token: int = 16

    @staticmethod
    def _count(value):
        if type(value) is not int or value < 0:
            raise ValueError("Inkling 2 memory dimensions must be nonnegative integers")
        return value

    @classmethod
    def from_config(cls, text, *, visits=None, prefill_rows=2048, draft=None, num_draft=0):
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
        )
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
        ring = min(capacity, self.sliding_window + self.prefill_rows)
        # Ordinary prefill retains stride-spaced window snapshots and can
        # force one conv snapshot at each turn end. DSpark instead holds the
        # pre-verify ring plus the current window and the new rollback rows.
        windows = min(self.checkpoint_copies, n // self.checkpoint_stride)
        saved_ring = windows * min(self.sliding_window, n)
        if self.verify_rows:
            saved_ring += min(n, self.sliding_window + 2 * self.verify_rows)
        saved_conv = min(n, self.checkpoint_copies) * self.conv_bytes
        total = (
            self.global_visits * capacity * token
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

    def as_dict(self):
        return dict(
            schema="inkling2-visit-kv-shortconv-geometry-v1",
            **asdict(self),
            conv_bytes=self.conv_bytes,
            bound="BF16 K/V per visit (global grows, sliding rings plus prefill overhang), "
                  "FP32 short-conv histories, DSpark drafter KV when selected",
            basis="shape bound checked against APCv2 copy-on-write plane bytes; not a speed claim",
        )
