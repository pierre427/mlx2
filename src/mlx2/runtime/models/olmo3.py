# Copyright © 2025 Apple Inc.
#
# Only the MLP block of mlx-lm's OLMo3 model is kept: it is the support
# dependency of olmo_hils.py (provenance/hils-attention.json).  mlx2 has no
# OLMo3 serving route, so the rest of the source model was dead code.

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu


class Olmo3MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
