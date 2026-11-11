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


def _apply_scale(y, scale):
    return (y.astype(mx.float32) * scale).astype(y.dtype)


class ScaledNVFP4Linear(nn.Module):
    """NVFP4 dense projection whose FP32 per-tensor scale is ``weight_scale_2``.

    The parameter name is the one MLX-VLM exports write (ModelOpt
    ``weight_scale_2``, a multiplier on the product).  Deliberately not an
    ``nn.QuantizedLinear``: fast paths that read a QuantizedLinear's packed
    weights directly would silently drop the scale, so they see an unknown
    module and take its own call (row_exact ``per_row_unsupported``).
    """

    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        if input_dims % GROUP_SIZE:
            raise ValueError("NVFP4 input width must be a multiple of 16")
        self.weight = mx.zeros((output_dims, input_dims * BITS // 32), dtype=mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // GROUP_SIZE), dtype=mx.uint8)
        self.weight_scale_2 = mx.array(1.0, dtype=mx.float32)
        self.freeze()

    def __call__(self, x):
        y = mx.quantized_matmul(
            x, self["weight"], scales=self["scales"], biases=None, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, mode="nvfp4",
        )
        return _apply_scale(y, self["weight_scale_2"])


class ScaledNVFP4SwitchLinear(nn.Module):
    """Routed NVFP4 experts with one FP32 ``weight_scale_2`` per expert.

    ``y[e] = gather_qmm(x, W_q[e]) * s[e]``, the MLX-VLM
    ``ScaledQuantizedSwitchLinear`` law.  Same call signature as
    ``QuantizedSwitchLinear`` but not a subclass, for the reason given on
    ``ScaledNVFP4Linear``: the NAX gather, fused ReLU2/down, int8 expert and
    expert-streaming paths all key on ``QuantizedSwitchLinear`` and none
    applies a per-expert global scale.
    """

    # moe_nax_gather admits NVFP4 only from this owned module (scope gate).
    nax_nvfp4_owned = True

    def __init__(self, input_dims: int, output_dims: int, num_experts: int):
        super().__init__()
        # Stock QuantizedSwitchLinear reroutes K % 32 != 0 (dense) and sorted
        # K % 64 != 0 (MLX sorted-tail bug) gathers; this module has neither
        # reroute, so it refuses those geometries instead.
        if input_dims % 64:
            raise ValueError("NVFP4 switch input width must be a multiple of 64")
        self.weight = mx.zeros(
            (num_experts, output_dims, input_dims * BITS // 32), dtype=mx.uint32
        )
        self.scales = mx.zeros(
            (num_experts, output_dims, input_dims // GROUP_SIZE), dtype=mx.uint8
        )
        self.weight_scale_2 = mx.ones((num_experts,), dtype=mx.float32)
        self.group_size, self.bits, self.mode = GROUP_SIZE, BITS, "nvfp4"
        self.freeze()

    @property
    def input_dims(self):
        return self.scales.shape[2] * GROUP_SIZE

    @property
    def output_dims(self):
        return self.weight.shape[1]

    @property
    def num_experts(self):
        return self.weight.shape[0]

    # Set by ``fold_relu2_expert_scales``: the owning MoE then multiplies
    # its routing scores by the folded per-expert scale instead.
    scale_folded = False

    def __call__(self, x, indices, sorted_indices=False):
        from .models import invariant_prefill as invariant

        if invariant.active():
            # No slice-invariance evidence for this gather: never claim it.
            invariant.not_invariant("nvfp4_scaled_gather")
        y = None
        if sorted_indices and "bias" not in self:
            from .models import moe_nax_gather as nax

            # Prefill forwards only, as for QuantizedSwitchLinear: the NAX
            # NVFP4 kernel is canary-proven bit-identical to the stock sorted
            # gather below, and returns None (stock) for anything else.
            if nax.MODE != "off" and nax.admit_phase():
                y = nax.try_gather(x, self, indices)
        if y is None:
            y = mx.gather_qmm(
                x, self["weight"], self["scales"], None, rhs_indices=indices,
                transpose=True, group_size=GROUP_SIZE, bits=BITS, mode="nvfp4",
                sorted_indices=sorted_indices,
            )
        if self.scale_folded:
            return y
        # The gather output is indices.shape + (rows, outputs).
        scale = self["weight_scale_2"][indices][..., None, None]
        return _apply_scale(y, scale)


FP8_GROUP = 32
FP8_UNIT_SCALE = 127  # E8M0 code of 2**0


class ScaledFP8Linear(nn.Module):
    """Vendor per-tensor FP8 (E4M3) projection on MLX ``mxfp8`` kernels.

    ModelOpt FP8 weights are ``e4m3 * weight_scale`` with one FP32 scale per
    tensor.  MLX ``mxfp8`` decodes E4M3 exactly (all 254 finite codes) and
    with every E8M0 block scale at 2**0 it is the bare E4M3 matrix, so
    ``y = (x @ e4m3.T) * weight_scale`` in float32, cast back.  W8A16: the
    vendor's static activation scale is not applied.  Not an
    ``nn.QuantizedLinear`` for the same reason as ``ScaledNVFP4Linear``.
    """

    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        if input_dims % FP8_GROUP:
            raise ValueError("FP8 input width must be a multiple of 32")
        self.weight = mx.zeros((output_dims, input_dims // 4), dtype=mx.uint32)
        self.scales = mx.full((output_dims, input_dims // FP8_GROUP), FP8_UNIT_SCALE,
                              dtype=mx.uint8)
        self.weight_scale = mx.array(1.0, dtype=mx.float32)
        self.freeze()

    def __call__(self, x):
        y = mx.quantized_matmul(
            x, self["weight"], scales=self["scales"], biases=None, transpose=True,
            group_size=FP8_GROUP, bits=8, mode="mxfp8",
        )
        return _apply_scale(y, self["weight_scale"])


def encode_vendor_fp8(weight, weight_scale):
    """Recover the vendor E4M3 codes of a ``bf16(e4m3 * s)`` weight, exactly.

    Returns ``(packed_uint32, unit_scales)``.  Refuses unless re-decoding the
    codes reproduces every input bit, so a weight that was not produced by
    the vendor FP8 law (or a wrong scale) can never load as FP8.
    """
    if weight.dtype != mx.bfloat16 or weight.ndim != 2 or weight.shape[1] % FP8_GROUP:
        raise ValueError("vendor FP8 recovery needs a 2-D bf16 weight, K % 32 == 0")
    scale = mx.array(weight_scale, dtype=mx.float32)
    codes = mx.to_fp8(weight.astype(mx.float32) / scale)
    rebuilt = (mx.from_fp8(codes, dtype=mx.float32) * scale).astype(mx.bfloat16)
    exact = mx.all(rebuilt.view(mx.uint16) == weight.view(mx.uint16))
    mx.eval(codes, exact)
    if not bool(exact):
        raise ValueError("weight is not bf16(e4m3 * weight_scale): refusing FP8 load")
    rows, cols = weight.shape
    packed = codes.reshape(rows, cols // 4, 4).view(mx.uint32).reshape(rows, cols // 4)
    unit = mx.full((rows, cols // FP8_GROUP), FP8_UNIT_SCALE, dtype=mx.uint8)
    return packed, unit


def install_vendor_fp8(model, weights, scales) -> list[str]:
    """Convert ``weights`` in place and swap each named ``nn.Linear``.

    ``scales`` maps module path -> vendor ``weight_scale``.  Each BF16
    weight is replaced by its exactly recovered E4M3 codes (checked bit for
    bit) plus unit block scales and the FP32 tensor scale.
    """
    installed = []
    for path, scale in scales.items():
        parent, name = _resolve(model, path)
        linear = getattr(parent, name)
        if type(linear) is not nn.Linear or "bias" in linear:
            raise ValueError(f"vendor FP8 module {path} is not an unbiased dense linear")
        key = path + ".weight"
        packed, unit = encode_vendor_fp8(weights[key], scale)
        weights[key] = packed
        weights[path + ".scales"] = unit
        weights[path + ".weight_scale"] = mx.array(scale, dtype=mx.float32)
        out_dims, in_dims = linear.weight.shape
        setattr(parent, name, ScaledFP8Linear(in_dims, out_dims))
        installed.append(path)
    return installed


class _FoldedScale:
    """Non-parameter holder, so the folded table stays out of the weights."""

    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def fold_relu2_expert_scales(switch_mlps, activation_dtype) -> int:
    """Fold each expert's two ``weight_scale_2`` into its routing weight.

    For ``y = sum_k w_k * fc2_e(relu2(fc1_e(x)))`` with
    ``fc_e(x) = qmm_e(x) * s_e`` and ``s1_e > 0``, ReLU2 is homogeneous of
    degree two, so ``y = sum_k (w_k * s1_e**2 * s2_e) * qmm2_e(relu2(qmm1_e(x)))``.
    The two fp32 scale passes over every routed row disappear; the product
    is rounded differently (the unscaled bf16 intermediates keep the same
    relative precision), so this is a distinct numerics profile, recorded
    in the adapter layout.  Refuses anything but ReLU2 over two
    ``ScaledNVFP4SwitchLinear`` projections with positive fc1 scales.
    Returns the number of folded expert tables.
    """
    if activation_dtype != mx.bfloat16:
        # Folded expert values carry 1 / (s1**2 * s2): bf16/fp32 range only.
        raise ValueError("NVFP4 scale fold needs bfloat16 activations")
    folded = 0
    for switch in switch_mlps:
        fc1, fc2 = switch.fc1, switch.fc2
        if not (isinstance(fc1, ScaledNVFP4SwitchLinear)
                and isinstance(fc2, ScaledNVFP4SwitchLinear)):
            raise ValueError("NVFP4 scale fold needs two ScaledNVFP4SwitchLinear projections")
        if type(switch.activation) is not nn.ReLU2:
            raise ValueError("NVFP4 scale fold needs a ReLU2 expert activation")
        s1, s2 = fc1["weight_scale_2"], fc2["weight_scale_2"]
        if not bool(mx.all(s1 > 0)):
            raise ValueError("NVFP4 scale fold needs positive fc1 scales")
        table = (s1 * s1 * s2).astype(mx.float32)
        mx.eval(table)
        switch.expert_output_scale = _FoldedScale(table)
        fc1.scale_folded = fc2.scale_folded = True
        folded += 1
    return folded


def install_weight_scale2(model, paths) -> dict[str, str]:
    """Replace each named dense or switch projection with its NVFP4 module.

    ``paths`` are module paths whose checkpoint carries ``weight_scale_2``.
    Returns ``{path: installed class name}``; anything other than an
    unquantized ``nn.Linear`` or ``SwitchLinear`` refuses.
    """
    from .models.switch_layers import SwitchLinear

    installed = {}
    for path in paths:
        parent, name = _resolve(model, path)
        module = getattr(parent, name)
        if type(module) is nn.Linear:
            if "bias" in module:
                raise ValueError(f"NVFP4 module {path} has a bias")
            out_dims, in_dims = module.weight.shape
            replacement = ScaledNVFP4Linear(in_dims, out_dims)
        elif type(module) is SwitchLinear:
            if "bias" in module:
                raise ValueError(f"NVFP4 switch module {path} has a bias")
            experts, out_dims, in_dims = module.weight.shape
            replacement = ScaledNVFP4SwitchLinear(in_dims, out_dims, experts)
        else:
            raise ValueError(f"NVFP4 module {path} is not a dense or switch linear")
        setattr(parent, name, replacement)
        installed[path] = type(replacement).__name__
    return installed


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


__all__ = [
    "BITS",
    "GROUP_SIZE",
    "GlobalScaledNVFP4Linear",
    "ScaledFP8Linear",
    "ScaledNVFP4Linear",
    "encode_vendor_fp8",
    "install_vendor_fp8",
    "ScaledNVFP4SwitchLinear",
    "fold_relu2_expert_scales",
    "install",
    "install_weight_scale2",
]
