"""Static LFM2.5-VL hybrid cache geometry for candidate admission.

The convolution state is the last two projected B*x rows per layer.  APCv2
also retains up to the configured number of exact convolution checkpoints.
The forward workspace is provisional until source-bound device measurement.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LFM25CacheBudget:
    attention_layers: int
    conv_layers: int
    kv_heads: int
    head_dim: int
    hidden_size: int
    conv_history: int
    checkpoint_copies: int
    item_bytes: int = 4  # fp32 upper bound for BF16/FP16/FP32 activations
    allocation_step: int = 256
    transcript_bytes_per_token: int = 16
    transient_gib_per_lane: float = 3.1  # provisional dense forward workspace

    @classmethod
    def from_config(cls, config: dict, *, mtp: bool):
        if mtp:
            raise ValueError("LFM2.5-VL DSpark is not a qualified serving route")
        layers = config.get("layer_types")
        if (not isinstance(layers, list) or len(layers) != 30 or
                layers.count("full_attention") != 8 or
                set(layers) != {"full_attention", "conv"}):
            raise ValueError("unknown LFM hybrid cache topology")
        width = int(config["hidden_size"])
        heads = int(config["num_attention_heads"])
        if width % heads:
            raise ValueError("LFM hidden size does not divide by heads")
        try:
            checkpoints = max(0, int(os.environ.get("MLX_LM_STATE_CHECKPOINT_MAX", "4")))
        except ValueError:
            checkpoints = 4
        result = cls(
            attention_layers=8, conv_layers=22,
            kv_heads=int(config["num_key_value_heads"]),
            head_dim=width // heads, hidden_size=width,
            conv_history=int(config["conv_L_cache"]) - 1,
            checkpoint_copies=checkpoints,
        )
        values = asdict(result)
        if (any(not math.isfinite(v) or v < 0 for v in values.values()) or
                not all((result.kv_heads, result.head_dim, result.hidden_size,
                         result.conv_history, result.item_bytes,
                         result.allocation_step))):
            raise ValueError("invalid LFM cache dimensions")
        return result

    @property
    def recurrent_live_bytes(self):
        return self.conv_layers * self.conv_history * self.hidden_size * self.item_bytes

    def project(self, context_tokens: int):
        if type(context_tokens) is not int or context_tokens < 0:
            raise ValueError("context_tokens must be nonnegative integer")
        capacity = (math.ceil(context_tokens / self.allocation_step) + 1) * self.allocation_step
        kv = (self.attention_layers * 2 * self.kv_heads * self.head_dim
              * self.item_bytes * capacity)
        return (kv + (self.checkpoint_copies + 1) * self.recurrent_live_bytes
                + capacity * self.transcript_bytes_per_token
                + (self.attention_layers + self.conv_layers) * 4096)

    def prefill_transient_bytes(self, context_tokens: int, chunk_rows: int):
        if type(context_tokens) is not int or context_tokens < 0 or chunk_rows < 0:
            raise ValueError("invalid LFM prefill dimensions")
        # Full media prefill can be one prompt-sized forward.  Charge a copy
        # of the growing K/V plus provisional encoder/projector workspace.
        kv_copy = (self.attention_layers * 2 * self.kv_heads * self.head_dim
                   * self.item_bytes * context_tokens)
        return int(kv_copy + (2 << 30) + chunk_rows * self.hidden_size * 16)

    def as_dict(self):
        return {"schema": "lfm25-vl-hybrid-cache-v1", **asdict(self),
                "bound": "fp32-KV-plus-conv-state-and-APCv2-checkpoints",
                "workspace": "provisional; no GPU measurement"}
