"""Block-scaled FP linear with a per-tensor FP32 scale (ModelOpt W4A16 law).

Original mlx2 code.  One technology: ``y = x @ (codes x block_scale x s).T + b``
where ``codes`` are E2M1 (``nvfp4``, E4M3 scale per 16) or E4M3 (``mxfp8``,
E8M0 scale per 32, all at 2**0 for vendor per-tensor FP8) and ``s`` is the
FP32 per-tensor multiplier ModelOpt stores as ``weight_scale_2`` (NVFP4) or
``weight_scale`` (FP8).  Activations are never quantized (W4A16 / W8A16);
a checkpoint's ``input_scale`` is validated by the loader and ignored.

MLX 0.32's ``quantized_matmul`` takes no global scale, so ``s`` is applied
outside the kernel.  Where it is applied is the one numerics choice that
differs between models, kept as :attr:`NVFP4Spec.global_scale_law`:

``"scale_rounded_product"``
    ``y = round(fp32(qmm(x)) * s) [+ b]``: the product is formed and rounded
    in the activation dtype, then scaled in FP32 and rounded again; a bias is
    added afterwards in its own dtype.  The MLX-VLM / vendor law
    (Nemotron 3.5 Super, Inkling 2 experts, Raon-1-mini dense, LiLiCorr
    native Nemotron drafter).
``"fp32_product"``
    ``y = round(qmm(fp32(x)) * s + fp32(b))``: input cast to FP32, product,
    scale and bias in FP32, one rounding.  The ModelOpt-native law
    (LiLiCorr ModelOpt drafter, Ripple, Ricursive).

Fast paths (``LINEAR`` registry) run only for the law whose arithmetic they
implement; see ``docs/technologies/nvfp4_linear.md``.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx import nn

from .dispatch import FastPath, KernelRegistry, Receipt, _record, spec_from_config

#: MLX's own NVFP4 ``global_scale`` argument is relative to the E4M3 and E2M1
#: maxima; a ModelOpt direct multiplier ``s`` is ``global_scale = s * 448 * 6``.
MLX_GLOBAL_SCALE_ENCODING = 448 * 6

GLOBAL_SCALE_LAWS = ("scale_rounded_product", "fp32_product")
EPILOGUES = ("eager", "compiled_env")
_MODES = {
    # mode: (bits, group_size, block-scale dtype)
    "nvfp4": (4, 16, "e4m3"),
    "mxfp8": (8, 32, "ue8m0"),
}


@dataclass(frozen=True)
class NVFP4Spec:
    """Every parameter of a block-scaled FP linear that differs between models.

    ``mode`` / ``bits`` / ``group_size`` / ``scale_dtype``
        MLX quantization mode and its fixed geometry: ``nvfp4`` = E2M1 x E4M3
        per 16, ``mxfp8`` = E4M3 x E8M0 per 32.
    ``global_scale_law``
        Where the FP32 per-tensor scale (and bias) is applied; see module
        docstring.  Not derivable from a checkpoint: a runtime numerics
        profile each model pins.
    ``epilogue``
        For ``scale_rounded_product`` only: ``eager`` runs cast-multiply-cast
        as three primitives; ``compiled_env`` runs the same arithmetic as one
        compiled element-wise kernel unless ``MLX2_SCALED_EPILOGUE=eager``.
    ``global_scale_name``
        Module parameter holding ``s`` (``global_scale`` for mlx2 / Raon /
        ModelOpt-repacked artifacts, ``weight_scale_2`` for MLX-VLM NVFP4
        exports, ``weight_scale`` for vendor FP8).
    ``bias`` / ``bias_dtype``
        Whether the projection has a bias, and its initial parameter dtype.
    ``unit_block_scale``
        Initial block-scale code (``127`` = E8M0 2**0 for vendor per-tensor
        FP8, whose block scales are all unit); ``0`` otherwise.
    ``input_scale``
        ``"ignored"``: the checkpoint carries a static activation scale that
        this W4A16 law validates and drops; ``"absent"``: the checkpoint must
        not carry one (sanitize refuses it).
    """

    mode: str = "nvfp4"
    bits: int = 4
    group_size: int = 16
    scale_dtype: str = "e4m3"
    global_scale_law: str = "fp32_product"
    epilogue: str = "eager"
    global_scale_name: str = "global_scale"
    bias: bool = False
    bias_dtype: str = "bfloat16"
    unit_block_scale: int = 0
    input_scale: str = "absent"

    def __post_init__(self):
        if self.mode not in _MODES:
            raise ValueError(f"unsupported block-scaled mode {self.mode!r}")
        if (self.bits, self.group_size, self.scale_dtype) != _MODES[self.mode]:
            raise ValueError(
                f"{self.mode} is bits/group/scale {_MODES[self.mode]}, got "
                f"{(self.bits, self.group_size, self.scale_dtype)}"
            )
        if self.global_scale_law not in GLOBAL_SCALE_LAWS:
            raise ValueError(f"unknown global_scale_law {self.global_scale_law!r}")
        if self.epilogue not in EPILOGUES:
            raise ValueError(f"unknown epilogue {self.epilogue!r}")
        if self.epilogue != "eager" and self.global_scale_law != "scale_rounded_product":
            raise ValueError("a compiled epilogue exists only for scale_rounded_product")
        if self.bias_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"unsupported bias_dtype {self.bias_dtype!r}")
        if self.input_scale not in ("ignored", "absent"):
            raise ValueError(f"unknown input_scale handling {self.input_scale!r}")

    @classmethod
    def from_config(cls, config: Any, **overrides) -> NVFP4Spec:
        """Derive geometry and input-scale handling from a quantization config.

        Accepts a model ``config.json`` (``quantization_config`` or MLX
        ``quantization``), a ModelOpt ``hf_quant_config.json``
        (``{"quantization": {"quant_algo": ...}}``) or the inner dict itself.
        ``global_scale_law``, ``epilogue`` and ``global_scale_name`` are
        runtime choices and come from ``overrides``.
        """
        quant = _quant_section(config)
        mode = overrides.get("mode") or _derive_mode(quant)
        if mode is None:
            raise ValueError(
                "cannot derive a block-scaled FP mode (nvfp4 / mxfp8) from this "
                f"quantization config: {dict(quant)!r}; pass mode= explicitly if it is one"
            )

        def geometry(index):
            return lambda _: None if mode is None else _MODES[mode][index]

        return spec_from_config(cls, quant, {
            "mode": lambda _: mode,
            "bits": geometry(0),
            "group_size": geometry(1),
            "scale_dtype": geometry(2),
            "input_scale": _derive_input_scale,
        }, **overrides)

    def weight_columns(self, input_dims: int) -> int:
        return input_dims * self.bits // 32


def _quant_section(config):
    if not isinstance(config, Mapping):
        config = {k: getattr(config, k) for k in ("quantization_config", "quantization")
                  if getattr(config, k, None) is not None}
    for key in ("quantization_config", "quantization"):
        inner = config.get(key)
        if isinstance(inner, Mapping):
            return inner
    return config


def _weights_group(quant):
    groups = quant.get("config_groups")
    if isinstance(groups, Mapping) and groups:
        first = next(iter(groups.values()))
        if isinstance(first, Mapping):
            return first
    return {}


def _derive_mode(quant):
    algo = str(quant.get("quant_algo") or "").upper()
    if algo in ("NVFP4", "W4A16_NVFP4", "NVFP4_AWQ"):
        mode = "nvfp4"
    elif algo == "FP8":
        mode = "mxfp8"
    elif quant.get("mode") in _MODES:
        mode = quant["mode"]
    else:
        weights = _weights_group(quant).get("weights") or {}
        if weights.get("type") == "float" and weights.get("num_bits") == 4:
            mode = "nvfp4"
        else:
            return None
    bits, group, _ = _MODES[mode]
    weights = _weights_group(quant).get("weights") or {}
    declared_group = quant.get("group_size", weights.get("group_size"))
    declared_bits = quant.get("bits", weights.get("num_bits"))
    if mode == "mxfp8" and algo == "FP8":
        declared_group = declared_bits = None  # per-tensor FP8: no block grid declared
    if declared_group is not None and declared_group != group:
        raise ValueError(f"{mode} requires group size {group}, config declares {declared_group}")
    if declared_bits is not None and declared_bits != bits:
        raise ValueError(f"{mode} requires {bits} bits, config declares {declared_bits}")
    return mode


def _derive_input_scale(quant):
    algo = str(quant.get("quant_algo") or "").upper()
    if algo.startswith(("W4A16", "W8A16")):
        return "absent"
    group = _weights_group(quant)
    if group.get("input_activations") is not None or algo in ("NVFP4", "FP8"):
        # ModelOpt W4A4 / W8A8 recipes ship input_scale; this law is W4A16.
        return "ignored"
    return None


# ---------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------

def _scale_law(y, scale):
    return (y.astype(mx.float32) * scale).astype(y.dtype)


_COMPILED_SCALE = None


def apply_global_scale(y, scale, spec: NVFP4Spec | None = None):
    """The ``scale_rounded_product`` epilogue: ``round(fp32(y) * scale)``.

    Shared by dense and routed (gather) projections.  ``spec.epilogue ==
    "compiled_env"`` (default since 2026-11-10, Pierre) runs the same
    cast-multiply-cast as one compiled element-wise kernel instead of three
    (one fp32 multiply either way; census1/routed2 in
    qualification/runs/super-vl-nvfp4-20261110: -386 kernels a decode step,
    same greedy tokens); ``MLX2_SCALED_EPILOGUE=eager`` keeps three primitives.
    """
    global _COMPILED_SCALE

    if spec is None or spec.epilogue == "eager":
        return _scale_law(y, scale)
    mode = os.environ.get("MLX2_SCALED_EPILOGUE", "compiled")
    if mode == "eager":
        return _scale_law(y, scale)
    if mode != "compiled":
        raise ValueError("MLX2_SCALED_EPILOGUE must be eager or compiled")
    if _COMPILED_SCALE is None:
        _COMPILED_SCALE = mx.compile(_scale_law, shapeless=True)
    return _COMPILED_SCALE(y, scale)


def reference(x, weight, scales, global_scale, bias, spec: NVFP4Spec):
    """Ordinary-decode NVFP4 / FP8 projection, parameterised only by ``spec``.

    ``global_scale`` may be a scalar or any array broadcastable against the
    output (a per-column vector for concatenated projections).
    """
    if spec.global_scale_law == "fp32_product":
        result = mx.quantized_matmul(
            x.astype(mx.float32), weight, scales=scales,
            group_size=spec.group_size, bits=spec.bits, mode=spec.mode, transpose=True,
        ) * global_scale
        if bias is not None:
            result = result + bias.astype(mx.float32)
        return result.astype(x.dtype)
    y = mx.quantized_matmul(
        x, weight, scales=scales, biases=None, transpose=True,
        group_size=spec.group_size, bits=spec.bits, mode=spec.mode,
    )
    y = apply_global_scale(y, global_scale, spec)
    if bias is not None:
        y = y + bias
    return y


def unscaled_product(x, weight, scales, spec: NVFP4Spec):
    """``qmm(x)`` in the activation dtype, before the global scale."""
    return mx.quantized_matmul(
        x, weight, scales=scales, biases=None, transpose=True,
        group_size=spec.group_size, bits=spec.bits, mode=spec.mode,
    )


def decode_weight(weight, scales, global_scale, spec: NVFP4Spec, dtype=mx.bfloat16):
    """Dequantized ``codes x block_scale x s`` through MLX's encoded global scale."""
    if spec.mode != "nvfp4":
        raise ValueError("MLX's dequantize global_scale is defined for nvfp4 only")
    return mx.dequantize(
        weight, scales=scales, group_size=spec.group_size, bits=spec.bits, mode=spec.mode,
        global_scale=global_scale * MLX_GLOBAL_SCALE_ENCODING, dtype=dtype,
    )


# ---------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------

class NVFP4Linear(nn.Module):
    """The one block-scaled FP linear module; its numerics are its ``spec``.

    Deliberately not an ``nn.QuantizedLinear``: fast paths that read a
    QuantizedLinear's packed weights directly would silently drop the global
    scale, so they see an unknown module and take its own call.
    """

    spec: NVFP4Spec = NVFP4Spec()
    _width_error = "NVFP4 input width must be a multiple of 16"

    def __init__(self, input_dims: int, output_dims: int, *, spec: NVFP4Spec | None = None,
                 bias: bool | None = None):
        super().__init__()
        if spec is not None:
            object.__setattr__(self, "spec", spec)
        spec = self.spec
        has_bias = spec.bias if bias is None else bool(bias)
        if input_dims % spec.group_size:
            raise ValueError(self._width_error)
        self.weight = mx.zeros((output_dims, spec.weight_columns(input_dims)), dtype=mx.uint32)
        if spec.unit_block_scale:
            self.scales = mx.full((output_dims, input_dims // spec.group_size),
                                  spec.unit_block_scale, dtype=mx.uint8)
        else:
            self.scales = mx.zeros((output_dims, input_dims // spec.group_size), dtype=mx.uint8)
        setattr(self, spec.global_scale_name, mx.array(1.0, dtype=mx.float32))
        if has_bias:
            dtype = mx.float32 if spec.bias_dtype == "float32" else mx.bfloat16
            self.bias = mx.zeros((output_dims,), dtype=dtype)
        self.freeze()

    @property
    def global_scale_value(self):
        return self[self.spec.global_scale_name]

    def __call__(self, x):
        return reference(
            x, self["weight"], self["scales"], self[self.spec.global_scale_name],
            self.get("bias"), self.spec,
        )


def module_reference(module, x):
    """The registry's reference: a module's own spec through :func:`reference`."""
    return NVFP4Linear.__call__(module, x)


# ---------------------------------------------------------------------------
# fast paths
# ---------------------------------------------------------------------------

def _fp32_kernel_refusal(spec):
    if spec.global_scale_law != "fp32_product":
        return "global_scale_law"
    if spec.mode != "nvfp4":
        return "mode"
    if spec.global_scale_name != "global_scale":
        return "global_scale_name"
    return None


def _admit_decoded(spec, *, module, x, **_):
    """The bounded decoded-weight cache, for at least ``threshold`` rows."""
    if spec.global_scale_law != "fp32_product":
        return "global_scale_law"
    decoded = getattr(module, "_decoded_weight", None)
    threshold = getattr(module, "_decoded_large_row_threshold", None)
    if decoded is None:
        return "decoded_not_resident"
    if threshold is None:
        return "no_threshold"
    if int(x.size // x.shape[-1]) < threshold:
        return "rows_below_threshold"
    return None


def _run_decoded(module, x):
    result = x @ module._decoded_weight.T
    if "bias" in module:
        result = result.astype(mx.float32) + module.bias.astype(mx.float32)
        result = result.astype(x.dtype)
    return result


def _admit_b1(spec, *, module, x, **_):
    from ..models import modelopt_nvfp4_b1 as b1

    return _fp32_kernel_refusal(spec) or b1.refusal(module, x)


def _run_b1(module, x):
    from ..models import modelopt_nvfp4_b1 as b1

    return b1.project(module, x)


def _admit_tile(spec, *, module, x, require_gpu=True, **_):
    from ..models import modelopt_nvfp4_tile as tile

    reason = _fp32_kernel_refusal(spec)
    if reason is not None:
        return reason
    admission = tile.admit(module, x, require_gpu=require_gpu)
    return None if admission.accepted else admission.reason


def _run_tile(module, x):
    from ..models import modelopt_nvfp4_tile as tile

    return tile.project(module, x)


def _run_tile_row_exact(module, x):
    from ..models import modelopt_nvfp4_tile as tile

    return tile.project_row_exact(module, x)


LINEAR = KernelRegistry("nvfp4_linear.project", module_reference)
DECODED_LARGE_ROWS = LINEAR.register(FastPath("decoded_large_rows", _admit_decoded, _run_decoded))
FUSED_B1 = LINEAR.register(FastPath("fused_b1", _admit_b1, _run_b1))
ROW_TILE = LINEAR.register(FastPath("row_tile", _admit_tile, _run_tile))
ROW_TILE_ROW_EXACT = LINEAR.register(
    FastPath("row_tile_row_exact", _admit_tile, _run_tile_row_exact)
)


_MODULE_PATHS = {"native": "reference", "decoded_large_rows": "decoded_large_rows",
                 "fused_b1": "fused_b1"}


def _project_owned(module, x, policy):
    """A module with its own dispatch policy (``ModelOptNVFP4Linear``).

    Runs the module's call so its policy semantics and its ``_dispatch_counts``
    (which feed ``nvfp4_model_status`` ``observed_used``) are the ones used;
    the receipt names the path its counters recorded.
    """
    owned = module._dispatch_policy
    if policy is not None and policy != owned:
        raise ValueError(
            f"policy: module dispatch policy is {owned!r}, project() asked for {policy!r}; "
            "set it with set_dispatch_policy"
        )
    before = dict(module._dispatch_counts)
    refusals_before = dict(module._refusal_counts)
    refusals: dict[str, str] = {}
    try:
        y = module(x)
    except ValueError as error:
        reason = str(error).rsplit(": ", 1)[-1]
        refusals["fused_b1"] = reason
        receipt = Receipt(LINEAR.operation, "failed_closed", refusals)
        _record(LINEAR.technology, LINEAR.op, receipt)
        raise
    taken = [name for name, count in module._dispatch_counts.items()
             if count > before.get(name, 0)]
    selected = _MODULE_PATHS.get(taken[0], taken[0]) if taken else "reference"
    if owned == "auto" and selected == "reference":
        refusals["decoded_large_rows"] = _admit_decoded(module.spec, module=module, x=x) or "unknown"
    if module._refusal_counts.get("fused_b1_rows", 0) > refusals_before.get("fused_b1_rows", 0):
        refusals["fused_b1"] = "rows: fused_b1 runs one row only"
    receipt = Receipt(LINEAR.operation, selected, refusals)
    _record(LINEAR.technology, LINEAR.op, receipt)
    return y, receipt


def project(module, x, *, policy: str | None = None) -> tuple[Any, Receipt]:
    """Dispatch ``module(x)`` through the registry; returns ``(y, receipt)``.

    A module that owns a dispatch policy (``ModelOptNVFP4Linear``: native,
    auto = decoded large rows only, fused_b1) runs under that policy and its
    own counters; ``policy`` must then be ``None`` or equal to it.  For other
    modules ``policy=None`` runs the reference (no fast path is a default),
    ``"auto"`` takes the first admitting fast path, and a fast-path name
    fails closed if it refuses.
    """
    if hasattr(module, "_dispatch_policy") and hasattr(module, "_dispatch_counts"):
        return _project_owned(module, x, policy)
    if policy is None:
        policy = "reference"
    fn, receipt = LINEAR.select(module.spec, policy=None if policy == "auto" else policy,
                                module=module, x=x)
    return fn(module, x), receipt


# ---------------------------------------------------------------------------
# checkpoint names
# ---------------------------------------------------------------------------

def _checked_tensor_scale(name, value):
    """A per-tensor scale must be one finite, positive number (ripple loader rule)."""
    if value.size != 1 or value.ndim != 0:
        raise ValueError(f"{name}: per-tensor scale must be a scalar, got shape {tuple(value.shape)}")
    scale = value.astype(mx.float32)
    ok = mx.isfinite(scale) & (scale > 0)
    mx.eval(ok)
    if not bool(ok):
        raise ValueError(f"{name}: per-tensor scale must be finite and positive")
    return scale


def _check_e4m3_codes(name, codes):
    """Refuse negative or NaN E4M3 block-scale codes (``conversion.nvfp4.decode_e4m3``)."""
    exponent, mantissa = (codes >> 3) & 15, codes & 7
    bad = mx.any(((codes & 128) != 0) | ((exponent == 15) & (mantissa == 7)))
    mx.eval(bad)
    if bool(bad):
        raise ValueError(f"{name}: NVFP4 scales must be finite and nonnegative")


def sanitize_modelopt(weights: Mapping[str, Any], prefixes, spec: NVFP4Spec) -> dict:
    """Rename ModelOpt tensors of each ``prefix`` to the module's parameters.

    ``.weight`` (U8 ``[rows, cols/2]`` for NVFP4, ``[rows, cols]`` for FP8)
    becomes the packed ``uint32`` view; ``.weight_scale`` (E4M3 bytes, read
    as ``uint8``) becomes ``.scales`` for NVFP4; the per-tensor scale
    (``.weight_scale_2`` for NVFP4, ``.weight_scale`` for FP8) becomes
    ``.<spec.global_scale_name>``; ``.input_scale`` is dropped when
    ``spec.input_scale == "ignored"`` and refused otherwise.  Other tensors
    pass through unchanged.
    """
    prefixes = set(prefixes)
    tensor_scale = ".weight_scale_2" if spec.mode == "nvfp4" else ".weight_scale"
    out = {}
    for name, value in weights.items():
        prefix, dot, leaf = name.rpartition(".")
        if prefix not in prefixes:
            out[name] = value
            continue
        leaf = dot + leaf
        if leaf == ".input_scale":
            if spec.input_scale != "ignored":
                raise ValueError(f"{name}: checkpoint carries input_scale but spec says absent")
            continue
        if leaf == tensor_scale:
            out[prefix + "." + spec.global_scale_name] = _checked_tensor_scale(name, value)
        elif leaf == ".weight_scale" and spec.mode == "nvfp4":
            if value.dtype != mx.uint8:
                raise ValueError(f"{name}: E4M3 block scales must load as uint8 bytes")
            _check_e4m3_codes(name, value)
            out[prefix + ".scales"] = value
        elif leaf == ".weight":
            group_bytes = spec.group_size * spec.bits // 8
            if value.dtype != mx.uint8 or value.ndim != 2 or value.shape[1] % group_bytes:
                raise ValueError(f"{name}: expected a uint8 matrix of complete {spec.mode} groups")
            block_scales = weights.get(prefix + ".weight_scale")
            if spec.mode == "nvfp4" and block_scales is not None:
                expected = (value.shape[0], value.shape[1] // group_bytes)
                if block_scales.shape != expected:
                    raise ValueError(f"{prefix}.weight_scale: expected block-scale grid {expected}")
            out[prefix + ".weight"] = value.view(mx.uint32)
            if spec.mode == "mxfp8":
                out.setdefault(prefix + ".scales", mx.full(
                    (value.shape[0], value.shape[1] // spec.group_size),
                    spec.unit_block_scale or 127, dtype=mx.uint8))
        else:
            out[name] = value
    return out


__all__ = [
    "DECODED_LARGE_ROWS",
    "EPILOGUES",
    "FUSED_B1",
    "GLOBAL_SCALE_LAWS",
    "LINEAR",
    "MLX_GLOBAL_SCALE_ENCODING",
    "ROW_TILE",
    "ROW_TILE_ROW_EXACT",
    "NVFP4Linear",
    "NVFP4Spec",
    "apply_global_scale",
    "decode_weight",
    "module_reference",
    "project",
    "reference",
    "sanitize_modelopt",
    "unscaled_product",
]
