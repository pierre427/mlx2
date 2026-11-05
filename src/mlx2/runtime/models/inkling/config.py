# SPDX-License-Identifier: MIT
# Adapted from Blaizzy/mlx-vlm; see provenance/inkling-small.json.
from dataclasses import dataclass

from ..base import BaseModelArgs as BaseModelConfig


def sanitize_quantization_config(quantization):
    """Alias per-layer quantization entries onto the load-time fused ``qkvr_proj``.

    Checkpoints quantized before the fusion existed key their per-layer
    overrides by the constituent paths (``...self_attn.q_proj`` and friends).
    The stacked projection inherits the ``q_proj`` entry: row-concat keeps each
    output row's own quantization groups (see ``fuse_qkvr``), so the
    constituents' settings apply unchanged to the fused matrix.
    """
    if not isinstance(quantization, dict):
        return quantization
    out = dict(quantization)
    for key, value in quantization.items():
        if key.endswith(".self_attn.q_proj"):
            out.setdefault(key[: -len("q_proj")] + "qkvr_proj", value)
    return out


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "inkling"
    hidden_size: int = 6144
    num_hidden_layers: int = 66
    vocab_size: int = 201024
    unpadded_vocab_size: int | None = None
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    use_embed_norm: bool = True
    logits_mup_width_multiplier: float = 1.0
    max_position_embeddings: int = 1048576
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int = 64
    swa_num_key_value_heads: int = 16
    swa_head_dim: int = 128
    sliding_window_size: int = 512
    local_layer_ids: list[int] | None = None
    layer_types: list[str] | None = None
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: int | None = None
    log_scaling_alpha: float = 0.1
    sconv_kernel_size: int = 4
    dense_mlp_idx: int = 0
    mlp_layer_types: list[str] | None = None
    # inkling nomenclature uses intermediate_size for MoE, then dense_intermediate_size
    moe_intermediate_size: int | None = None
    intermediate_size: int = 24576
    dense_intermediate_size: int | None = None
    n_routed_experts: int = 256
    num_experts_per_tok: int = 6
    n_shared_experts: int = 2
    route_scale: float = 8.0

    def layer_is_sliding(self, i: int) -> bool:
        """Sliding-window (local) vs global full-attention layer. Real checkpoints set
        either ``layer_types`` (0.6B) or ``local_layer_ids`` (975B); the modulo is the
        reference default (every 6th layer global)."""
        if self.layer_types is not None:
            return self.layer_types[i] == "hybrid_sliding"
        if self.local_layer_ids is not None:
            return i in set(self.local_layer_ids)
        return bool((i + 1) % 6)

    def layer_is_dense(self, i: int) -> bool:
        if self.mlp_layer_types is not None:
            return self.mlp_layer_types[i] == "dense"
        return i < self.dense_mlp_idx

    def __post_init__(self):
        if self.moe_intermediate_size is not None:
            self.intermediate_size = self.moe_intermediate_size
        if self.dense_intermediate_size is None:
            self.dense_intermediate_size = self.intermediate_size
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Inkling requires divisible GQA heads")
