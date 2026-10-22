"""NVFP4 linear with a per-tensor FP32 global scale (ModelOpt W4A16 law).

Original mlx2 code.  A ModelOpt NVFP4 weight is ``E2M1 value x E4M3 group-16
scale x FP32 per-tensor scale``.  ``conversion/nvfp4.py`` (mode ``native``)
repacks those bytes without changing them; MLX's ``nvfp4`` mode reads the
same E2M1 nibble order and E4M3 group scales (``mx.dequantize`` of an export
equals ``conversion.nvfp4.decode_rows`` bit for bit before the global scale).
MLX 0.32's ``quantized_matmul`` has no global-scale argument, so the scalar is
applied to the product in float32: ``y = (x @ W_q.T) * s + b``.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

GROUP_SIZE = 16
BITS = 4


class GlobalScaledNVFP4Linear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, bias: bool = False):
        super().__init__()
        if input_dims % GROUP_SIZE:
            raise ValueError("NVFP4 input width must be a multiple of 16")
        self.weight = mx.zeros((output_dims, input_dims * BITS // 32), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // GROUP_SIZE), dtype=mx.uint8)
        self.global_scale = mx.array(1.0, dtype=mx.float32)
        if bias:
            self.bias = mx.zeros((output_dims,))
        self.freeze()

    def __call__(self, x):
        y = mx.quantized_matmul(
            x, self["weight"], scales=self["scales"], biases=None, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, mode="nvfp4",
        )
        y = (y.astype(mx.float32) * self["global_scale"]).astype(y.dtype)
        if "bias" in self:
            y = y + self["bias"]
        return y


def _resolve(model, path):
    parent = model
    parts = path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def install(model, modules) -> list[str]:
    """Replace each named ``nn.Linear`` with a global-scaled NVFP4 linear."""
    installed = []
    for path in modules:
        parent, name = _resolve(model, path)
        linear = getattr(parent, name)
        if type(linear) is not nn.Linear:
            raise ValueError(f"NVFP4 module {path} is not a dense linear")
        out_dims, in_dims = linear.weight.shape
        setattr(parent, name, GlobalScaledNVFP4Linear(in_dims, out_dims, bias="bias" in linear))
        installed.append(path)
    return installed


__all__ = ["BITS", "GROUP_SIZE", "GlobalScaledNVFP4Linear", "install"]
