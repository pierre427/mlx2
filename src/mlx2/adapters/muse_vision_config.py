# SPDX-License-Identifier: MIT
# Adapted from mlx-vlm; see provenance/muse-vision-serving.json and .NOTICE.
"""GPU-free Muse image geometry; unknown numerical laws refuse."""
from dataclasses import dataclass, field
import math


@dataclass
class VisionConfig:
    patch_size: int = 14
    patch_temporal: int = 2
    merge_size: int = 2
    pos_emb_height: int = 32
    pos_emb_width: int = 32
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_attention_heads: int = 16
    num_hidden_layers: int = 50
    hidden_act: str = 'gelu'
    layer_norm_eps: float = 1e-5
    rope_parameters: dict = field(default_factory=lambda: {'rope_theta': 10000., 'rope_type': 'default'})
    layer_types: list | None = None

    def __post_init__(self):
        for name in ('patch_size', 'patch_temporal', 'merge_size', 'pos_emb_height',
                     'pos_emb_width', 'hidden_size', 'intermediate_size',
                     'num_attention_heads', 'num_hidden_layers'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('Muse vision dimensions must be positive integers')
        if (self.hidden_act != 'gelu' or self.pos_emb_height != self.pos_emb_width
            or self.hidden_size % self.num_attention_heads
            or (self.hidden_size // self.num_attention_heads) % 4
            or not math.isfinite(self.layer_norm_eps) or self.layer_norm_eps <= 0):
            raise ValueError('Unsupported Muse vision geometry/activation')
        rope = self.rope_parameters
        if (set(rope) - {'rope_theta', 'rope_type'} or rope.get('rope_type', 'default') != 'default'
            or not math.isfinite(rope.get('rope_theta', 10000.)) or rope.get('rope_theta', 10000.) <= 0):
            raise ValueError('Unsupported Muse vision RoPE')
        if self.layer_types is None:
            self.layer_types = ['full_attention' if (i + 1) % 4 == 0 or i == self.num_hidden_layers - 1
                                else 'window_attention' for i in range(self.num_hidden_layers)]
        if len(self.layer_types) != self.num_hidden_layers or set(self.layer_types) - {'full_attention', 'window_attention'}:
            raise ValueError('Unsupported Muse vision attention layout')

    @classmethod
    def from_dict(cls, config):
        if config.get('model_type', 'muse_glimmer_vision') != 'muse_glimmer_vision':
            raise ValueError('Expected a Muse vision config')
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})
