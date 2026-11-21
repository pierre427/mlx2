# SPDX-License-Identifier: MIT
# Adapted from Blaizzy/mlx-vlm@111ac668 mlx_vlm/models/inkling2/config.py;
# see provenance/inkling2-mini.json.
from dataclasses import dataclass

from ..base import BaseModelArgs


@dataclass(frozen=True)
class LoopConfig:
    core_start: int
    core_end: int
    num_passes: int
    gate_rank: int
    is_varloop: bool = False


@dataclass
class TextConfig(BaseModelArgs):
    model_type: str = "inkling2_loop_text"
    hidden_size: int = 2304
    num_hidden_layers: int = 24
    vocab_size: int = 201024
    unpadded_vocab_size: int | None = 200058
    model_max_length: int = 1048576
    rms_norm_eps: float = 1e-6
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    swa_num_attention_heads: int = 24
    swa_num_key_value_heads: int = 8
    swa_head_dim: int = 128
    sliding_window_size: int = 1024
    local_layer_ids: list[int] | None = None
    layer_types: list[str] | None = None
    d_rel: int = 16
    rel_extent: int = 1024
    log_scaling_n_floor: int | None = 8192
    log_scaling_alpha: float = 0.1
    sconv_kernel_size: int = 4
    intermediate_size: int = 1152
    n_routed_experts: int = 128
    num_experts_per_tok: int = 8
    n_shared_experts: int = 0
    route_scale: float = 8.0
    use_gate_bias: bool = True
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    n_layers_per_attn_res_block: int = 6
    attn_res_normalize_values: bool = True
    use_value_norm_no_gamma: bool = True
    q_bias: bool = False
    o_bias: bool = False
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    final_logit_softcapping: float | None = 30.0
    loop_config: LoopConfig | None = None

    def __post_init__(self):
        if isinstance(self.loop_config, dict):
            self.loop_config = LoopConfig(**self.loop_config)
        if self.n_shared_experts:
            raise ValueError("Inkling 2 shared experts are not implemented")
        if self.q_bias or self.o_bias or not self.use_gate_bias:
            raise ValueError("Inkling 2 attention biases / unbiased routing are not implemented")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Inkling 2 requires divisible GQA heads")
        loop = self.loop_config
        if loop is None:
            return
        block = self.n_layers_per_attn_res_block
        if not (
            0 < loop.core_start < loop.core_end <= self.num_hidden_layers
            and block > 0
            and loop.core_start % block == 0
            and loop.core_end % block == 0
            and loop.num_passes >= 1
            and loop.gate_rank >= 1
            and not loop.is_varloop
        ):
            raise ValueError(f"Unsupported Inkling 2 loop_config: {loop}")

    @property
    def layer_indices(self) -> tuple[int, ...]:
        """Physical layer run at each visit (the middle block repeats)."""
        indices = tuple(range(self.num_hidden_layers))
        loop = self.loop_config
        if loop is None:
            return indices
        return (
            indices[: loop.core_start]
            + indices[loop.core_start : loop.core_end] * loop.num_passes
            + indices[loop.core_end :]
        )

    def layer_is_sliding(self, index: int) -> bool:
        if self.layer_types is not None:
            return self.layer_types[index] == "hybrid_sliding"
        if self.local_layer_ids is not None:
            return index in self.local_layer_ids
        return (index + 1) % 6 != 0
