# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see provenance/muse-glimmer.json and .NOTICE.
"""GPU-free Muse text configuration and cache topology."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelArgs:
    model_type: str = "muse_glimmer"
    hidden_size: int = 6656
    num_hidden_layers: int = 52
    intermediate_size: int = 19968
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    vocab_size: int = 202048
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    sliding_window: int = 2048
    qk_scale_factor: float = 3.87
    hidden_activation: str = "silu"
    max_position_embeddings: int = 131072
    rope_theta: float = 500000.0
    rope_parameters: Optional[dict] = None
    layer_types: Optional[List[str]] = None
    layer_rope_theta: Optional[List[float]] = None
    tie_word_embeddings: bool = False
    final_logit_softcapping: float = 20.0
    output_multiplier: float = 0.19611613513818404

    @classmethod
    def from_dict(cls, params):
        # Meta's original checkpoint nests the language-model fields under
        # "text_config"; the mlx-community conversions flatten them. Accept both.
        if "text_config" in params:
            merged = dict(params)
            merged.update(params["text_config"])
            params = merged
        params = dict(params, model_type="muse_glimmer")
        if params.get("attention_bias", False):
            raise ValueError("Muse port requires bias-free attention")
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in params.items() if k in fields})

    def __post_init__(self):
        if self.hidden_activation != "silu":
            raise ValueError("Muse port requires silu activation")
        if self.num_hidden_layers <= 0 or self.sliding_window <= 0:
            raise ValueError("Layer count and attention window must be positive")
        if (
            self.num_key_value_heads <= 0
            or self.num_attention_heads <= 0
            or self.num_attention_heads % self.num_key_value_heads
        ):
            raise ValueError("Query heads must be divisible by KV heads")
        if self.final_logit_softcapping <= 0:
            raise ValueError("Muse logit softcap must be positive")
        # rope_theta may live under rope_parameters (as in Meta's config)
        if self.rope_parameters and "rope_theta" in self.rope_parameters:
            self.rope_theta = self.rope_parameters["rope_theta"]
        if self.layer_types is None:
            # default [local, local, local, global] repeating
            pat = ["sliding_attention"] * 3 + ["full_attention"]
            self.layer_types = (pat * (self.num_hidden_layers // 4 + 1))[
                : self.num_hidden_layers
            ]
        if self.layer_rope_theta is None:
            self.layer_rope_theta = [
                0 if t == "full_attention" else self.rope_theta
                for t in self.layer_types
            ]
        self.validate_topology()

    @property
    def cache_layout(self):
        import hashlib
        import json

        topology = (
            self.layer_types,
            self.layer_rope_theta,
            self.sliding_window,
            self.num_key_value_heads,
            self.head_dim,
        )
        return (
            "muse-glimmer-layer-segments-v1:"
            + hashlib.sha256(json.dumps(topology).encode()).hexdigest()[:16]
        )

    def validate_topology(self):
        if (
            len(self.layer_types) != self.num_hidden_layers
            or len(self.layer_rope_theta) != self.num_hidden_layers
        ):
            raise ValueError("Muse layer topology length does not match layer count")
        if any(
            t not in {"sliding_attention", "full_attention"} for t in self.layer_types
        ):
            raise ValueError("Unknown Muse attention layer type")
        if any(
            (t == "full_attention") != (theta == 0)
            for t, theta in zip(self.layer_types, self.layer_rope_theta)
        ):
            raise ValueError("Muse requires local RoPE and global NoPE")
