"""Conservative Inkling KV/shortconv geometry, without a measured speed claim."""

import math
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class InklingCacheBudget:
    global_layers: int = 7
    sliding_layers: int = 35
    kv_heads: int = 8
    head_dim: int = 128
    hidden_size: int = 4096
    sliding_window: int = 512
    prefill_step: int = 128
    checkpoint_copies: int = 4
    item_bytes: int = 4  # upper bound including promoted FP32 activations
    transient_gib_per_lane: float = 2.0  # provisional; not a measured model bound

    @staticmethod
    def _count(value):
        if type(value) is not int or value < 0:
            raise ValueError("Inkling memory dimensions must be nonnegative integers")
        return value

    @classmethod
    def from_config(cls, config, *, mtp):
        if mtp:
            raise ValueError("Inkling mixed conversion has no MTP tensors")
        text = config["text_config"]
        local = len(text["local_layer_ids"])
        copies = int(os.environ.get("MLX_LM_STATE_CHECKPOINT_MAX", "4"))
        cls._count(copies)
        return cls(
            global_layers=text["num_hidden_layers"] - local,
            sliding_layers=local,
            checkpoint_copies=copies,
        )

    @property
    def recurrent_bytes(self):
        return (
            (self.global_layers + self.sliding_layers)
            * 3
            * (2 * self.kv_heads * self.head_dim + 2 * self.hidden_size)
            * 4
        )

    def project(self, context_tokens):
        n = self._count(context_tokens)
        capacity = (math.ceil(n / 256) + 1) * 256
        token_bytes = 2 * self.kv_heads * self.head_dim * self.item_bytes
        global_kv = capacity * self.global_layers * token_bytes
        sliding = (
            self.sliding_layers
            * token_bytes
            * (
                self.sliding_window
                + self.prefill_step
                + self.sliding_window * self.checkpoint_copies
            )
        )
        return (
            global_kv
            + sliding
            + self.recurrent_bytes * (1 + self.checkpoint_copies)
            + capacity * 16
        )

    def prefill_transient_bytes(self, context_tokens, chunk_rows):
        n, rows = self._count(context_tokens), self._count(chunk_rows)
        # One layer's relative-bias mask + SDPA workspace, projected activations,
        # routed intermediates and cache growth. This is a shape bound, not measured.
        return (
            3 * 32 * min(rows, 128) * max(n, 1) * 4
            + rows * (16 * self.hidden_size + 6 * 2048 * 3) * 4
        )

    def as_dict(self):
        return dict(
            schema="inkling-kv-four-shortconv-geometry-v1",
            **asdict(self),
            recurrent_bytes=self.recurrent_bytes,
            bound="fp32 KV upper bound, bounded local rings and four fp32 histories",
            transient_basis="provisional geometry; not measured checkpoint evidence",
        )
