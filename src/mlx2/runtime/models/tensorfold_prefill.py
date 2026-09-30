# SPDX-License-Identifier: MIT
"""Candidate q4/g64 prefill projections; see provenance/tensorfold-prefill.json.

TensorFold-inspired row grouping with two explicit experimental backends.
Native MLX packs large projection weights and makes the original projections
views of those arrays. The M5 Metal candidate uses 32x64x64 tiles, reusing
staged activations and dequantized weights without a full dequantized table.
Its fused SwiGLU writes only the final activation.
Installation is explicit, default-off, and does not change decode kernels.
"""

import hashlib
import inspect
from collections import Counter
from contextvars import ContextVar
from functools import lru_cache

import mlx.core as mx
from mlx import nn

from ..prefill_plan import prefill_rows
from .prefill_metal import (
    HEADER,
    THREADS,
    TILE_K,
    TILE_M,
    TILE_N,
    source_for,
    swiglu_source,
)

_PREFILL_SCOPE = ContextVar("mlx2_tensorfold_prefill_scope", default=False)


@lru_cache(maxsize=8)
def _scoped_decoder_type(original):
    """Scope installed layers so verification never selects prefill arithmetic."""
    parameters = list(inspect.signature(original.__call__).parameters)
    cache_index = parameters.index("cache") - 1

    class PrefillDecoder(original):
        def __call__(self, *args, **kwargs):
            cache = kwargs.get(
                "cache", args[cache_index] if len(args) > cache_index else None
            )
            x = args[0] if args else kwargs[parameters[1]]
            token = _PREFILL_SCOPE.set(
                bool(prefill_rows(x.shape))
                and not bool(getattr(cache, "speculating", False))
            )
            try:
                return super().__call__(*args, **kwargs)
            finally:
                _PREFILL_SCOPE.reset(token)

    return PrefillDecoder


@lru_cache(maxsize=64)
def _kernel(widths, biases):
    source, inputs = source_for(widths, biases)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name=f"mlx2_tensorfold_prefill_{digest}",
        input_names=inputs,
        output_names=[f"Y{i}" for i in range(len(widths))],
        header=HEADER,
        source=source,
    )


@lru_cache(maxsize=16)
def _swiglu_kernel(width, biases):
    source, inputs = swiglu_source(width, biases)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name=f"mlx2_prefill_swiglu_{digest}",
        input_names=inputs,
        output_names=["OUT"],
        header=HEADER,
        source=source,
    )


def eligible(module):
    if not isinstance(module, nn.QuantizedLinear):
        return False
    if (
        module.bits != 4
        or module.group_size != 64
        or getattr(module, "mode", "affine") != "affine"
    ):
        return False
    w, s, b = module.weight, module.scales, module.biases
    if w.ndim != 2 or s.ndim != 2 or b.shape != s.shape:
        return False
    n, packed_k = w.shape
    k = packed_k * 8
    return (
        n > 0
        and k > 0
        and k % 64 == 0
        and s.shape == (n, k // 64)
        and w.dtype == mx.uint32
        and s.dtype == b.dtype == mx.bfloat16
        and (
            "bias" not in module
            or (module.bias.shape == (n,) and module.bias.dtype == mx.bfloat16)
        )
    )


def project(x, modules):
    """Project arbitrary prefill rows, with masked M/N tails and fused bias."""
    modules = tuple(modules)
    if not modules or len(modules) > 4 or x.ndim < 2 or x.dtype != mx.bfloat16:
        raise ValueError("prefill projection requires bf16 rows and 1..4 projections")
    k = x.shape[-1]
    rows = x.size // k if k else 0
    if rows < 1 or any(not eligible(m) or m.weight.shape[1] * 8 != k for m in modules):
        raise ValueError("prefill projection requires matching affine q4/g64 geometry")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("prefill projection requires Metal")
    if "M5" not in str(mx.device_info().get("device_name", "")):
        raise RuntimeError("Metal prefill candidate requires M5")
    widths = tuple(int(m.weight.shape[0]) for m in modules)
    biases = tuple("bias" in m for m in modules)
    inputs = [mx.contiguous(x.reshape(rows, k)), rows]
    for m, biased in zip(modules, biases):
        inputs.extend(
            [mx.contiguous(m.weight), mx.contiguous(m.scales), mx.contiguous(m.biases)]
        )
        if biased:
            inputs.append(mx.contiguous(m.bias))
    result = _kernel(widths, biases)(
        inputs=inputs,
        template=[("K", k)],
        grid=(
            THREADS * ((rows + TILE_M - 1) // TILE_M),
            sum((n + TILE_N - 1) // TILE_N for n in widths),
            1,
        ),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, n) for n in widths],
        output_dtypes=[mx.bfloat16] * len(widths),
    )
    return tuple(y.reshape(*x.shape[:-1], n) for y, n in zip(result, widths))


def _admitted(x):
    return bool(
        _PREFILL_SCOPE.get()
        and prefill_rows(x.shape)
        and x.dtype == mx.bfloat16
        and mx.default_device() == mx.gpu
        and mx.metal.is_available()
    )


class PackedProjectionGroup:
    """One native prefill matmul; original projections become views of its weights.

    Evaluate each packed group before replacing the original arrays, bounding
    load-time duplication to one group. No second resident copy is retained.
    A later weight replacement invalidates the handle before it can be used.
    """

    def __init__(self, modules):
        modules = tuple(modules)
        self.widths = tuple(int(m.weight.shape[0]) for m in modules)
        self.weight, self.scales, self.biases = (
            mx.concatenate([getattr(m, key) for m in modules], axis=0)
            for key in ("weight", "scales", "biases")
        )
        mx.eval(self.weight, self.scales, self.biases)
        self.expected = []
        start = 0
        for m, n in zip(modules, self.widths):
            m.weight = self.weight[start : start + n]
            m.scales = self.scales[start : start + n]
            m.biases = self.biases[start : start + n]
            self.expected.append((m.weight, m.scales, m.biases))
            start += n

    def matches(self, modules):
        return len(modules) == len(self.expected) and all(
            all(
                getattr(m, key) is array
                for key, array in zip(("weight", "scales", "biases"), expected)
            )
            for m, expected in zip(modules, self.expected)
        )

    def __call__(self, x, modules):
        y = mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=64,
            bits=4,
        )
        cuts, total = [], 0
        for n in self.widths[:-1]:
            total += n
            cuts.append(total)
        return tuple(
            value + m.bias if "bias" in m else value
            for value, m in zip(mx.split(y, cuts, axis=-1), modules)
        )


def project_swiglu(x, gate, up):
    """Fused dense gate/up projection and activation; final output only."""
    if (
        x.ndim < 2
        or x.dtype != mx.bfloat16
        or not eligible(gate)
        or not eligible(up)
        or gate.weight.shape != up.weight.shape
        or x.shape[-1] != gate.weight.shape[1] * 8
    ):
        raise ValueError(
            "prefill SwiGLU requires matching bf16 affine q4/g64 projections"
        )
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("prefill SwiGLU requires Metal")
    if "M5" not in str(mx.device_info().get("device_name", "")):
        raise RuntimeError("Metal prefill candidate requires M5")
    k, n = x.shape[-1], gate.weight.shape[0]
    rows = x.size // k
    if rows < 1:
        raise ValueError("prefill SwiGLU requires nonempty rows")
    modules = (gate, up)
    biases = tuple("bias" in m for m in modules)
    inputs = [mx.contiguous(x.reshape(rows, k)), rows]
    for m in modules:
        inputs.extend(
            [mx.contiguous(m.weight), mx.contiguous(m.scales), mx.contiguous(m.biases)]
        )
    inputs.extend(mx.contiguous(m.bias) for m in modules if "bias" in m)
    out = _swiglu_kernel(n, biases)(
        inputs=inputs,
        template=[("K", k)],
        grid=(THREADS * ((rows + TILE_M - 1) // TILE_M), (n + TILE_N - 1) // TILE_N, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return out.reshape(*x.shape[:-1], n)


def fused_mlp(layer, x):
    counts = getattr(layer, "_prefill_counts", None)
    if (
        counts is None
        or layer.training
        or not getattr(layer, "_prefill_enabled", True)
        or not _admitted(x)
    ):
        return None
    gate, up = layer.gate_proj, layer.up_proj
    if not eligible(gate) or not eligible(up) or gate.weight.shape != up.weight.shape:
        counts["swiglu_reference_calls"] += 1
        return None
    packed = getattr(layer, "_prefill_mlp_group", None)
    if getattr(layer, "_prefill_backend", "metal") == "native":
        if x.shape[1] < 64:
            return None
        if packed is None or not packed.matches((gate, up)):
            object.__setattr__(layer, "_prefill_mlp_group", None)
            counts["stale_group_fallbacks"] += 1
            return None
        from .activations import swiglu

        y = swiglu(*packed(x, (gate, up)))
        counts["native_swiglu_calls"] += 1
        counts["projection_launches_saved"] += 1
    else:
        y = project_swiglu(x, gate, up)
        counts["swiglu_intermediate_elements_avoided"] += 2 * y.size
    counts["swiglu_calls"] += 1
    return layer.down_proj(y)


class TensorFoldPrefillLinear(nn.QuantizedLinear):
    def __call__(self, x):
        counts = self._prefill_counts
        if (
            self._prefill_backend == "metal"
            and not self.training
            and _admitted(x)
            and eligible(self)
        ):
            y = project(x, (self,))[0]
            counts["projection_calls"] += 1
            counts["projection_rows"] += prefill_rows(x.shape)
            return y
        counts["reference_calls"] += 1
        return super().__call__(x)


def grouped_input(layer, x):
    """Group GDN projections; native packing preserves small gate reductions."""
    counts = getattr(layer, "_prefill_counts", None)
    if (
        counts is None
        or layer.training
        or not getattr(layer, "_prefill_enabled", True)
        or not _admitted(x)
    ):
        return None
    names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    modules = tuple(getattr(layer, name, None) for name in names)
    if any(not eligible(m) for m in modules):
        counts["group_reference_calls"] += 1
        return None
    if getattr(layer, "_prefill_backend", "metal") == "native":
        if x.shape[1] < 64:
            return None
        packed = getattr(layer, "_prefill_input_group", None)
        if packed is None or not packed.matches(modules[:2]):
            object.__setattr__(layer, "_prefill_input_group", None)
            counts["stale_group_fallbacks"] += 1
            return None
        # Small b/a projections choose a different split-K reduction when
        # stacked with qkv/z. Keep their reference arithmetic: full-model
        # tests amplified that otherwise-small gate error across 64 layers.
        ys = (*packed(x, modules[:2]), modules[2](x), modules[3](x))
        counts["native_grouped_calls"] += 1
        counts["projection_launches_saved"] += 1
    else:
        ys = project(x, modules)
        counts["projection_launches_saved"] += 3
    counts["grouped_calls"] += 1
    counts["grouped_rows"] += prefill_rows(x.shape)
    return ys


def install(model, *, backend="native"):
    """Install a prefill-only candidate; retain any selected decode matvec."""
    from .flash_tensorfold_qmv import TensorFoldQMVLinear
    from .qwen3_5 import GatedDeltaNet
    from .qwen3_next import Qwen3NextMLP
    from .qwen4_exp import DecoderLayer as FlashDecoder
    from .qwen38_27b import DecoderLayer as DenseDecoder

    if backend not in ("native", "metal"):
        raise ValueError("prefill backend must be native or metal")
    if (
        backend == "metal"
        and mx.default_device() == mx.gpu
        and "M5" not in str(mx.device_info().get("device_name", ""))
    ):
        raise ValueError("Metal prefill candidate requires M5")

    class PrefillAndDecodeLinear(TensorFoldPrefillLinear, TensorFoldQMVLinear):
        pass

    counts = Counter()
    installed, groups, mlps = [], 0, 0
    skip = {"switch_mlp", "shared_expert", "ple", "lm_head", "mtp_draft_head", "mtp"}
    modules = list(model.named_modules())
    for name, module in modules:
        if skip.intersection(name.split(".")):
            continue
        if (
            backend == "metal"
            and type(module) in (nn.QuantizedLinear, TensorFoldQMVLinear)
            and eligible(module)
        ):
            module.__class__ = (
                PrefillAndDecodeLinear
                if type(module) is TensorFoldQMVLinear
                else TensorFoldPrefillLinear
            )
            object.__setattr__(module, "_prefill_counts", counts)
            object.__setattr__(module, "_prefill_backend", backend)
            installed.append(name)
        if isinstance(module, (DenseDecoder, FlashDecoder)):
            module.__class__ = _scoped_decoder_type(type(module))
        if isinstance(module, GatedDeltaNet) and all(
            type(getattr(module, part, None))
            in (nn.QuantizedLinear, TensorFoldQMVLinear)
            and eligible(getattr(module, part))
            for part in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        ):
            object.__setattr__(module, "_prefill_counts", counts)
            object.__setattr__(module, "_prefill_backend", backend)
            if backend == "native":
                object.__setattr__(
                    module,
                    "_prefill_input_group",
                    PackedProjectionGroup(
                        tuple(
                            getattr(module, part)
                            for part in (
                                "in_proj_qkv",
                                "in_proj_z",
                            )
                        )
                    ),
                )
                installed.extend(
                    f"{name}.{part}".lstrip(".")
                    for part in ("in_proj_qkv", "in_proj_z")
                )
            groups += 1
        if isinstance(module, Qwen3NextMLP) and all(
            type(getattr(module, part, None))
            in (nn.QuantizedLinear, TensorFoldQMVLinear)
            and eligible(getattr(module, part))
            for part in ("gate_proj", "up_proj")
        ):
            # Only the explicit Qwen3NextMLP hook consumes this marker.
            object.__setattr__(module, "_prefill_counts", counts)
            object.__setattr__(module, "_prefill_backend", backend)
            if backend == "native":
                object.__setattr__(
                    module,
                    "_prefill_mlp_group",
                    PackedProjectionGroup((module.gate_proj, module.up_proj)),
                )
                installed.extend(
                    f"{name}.{part}".lstrip(".") for part in ("gate_proj", "up_proj")
                )
            mlps += 1
    if not installed:
        raise ValueError("prefill candidate found no eligible dense q4/g64 projections")
    return {
        "kernel": "q4g64_prefill_m32n64k64_nax"
        if backend == "metal"
        else "q4g64_prefill_native_qkv_z_swiglu",
        "installed": len(installed),
        "grouped_layers": groups,
        "mlp_layers": mlps,
        "tile": [TILE_M, TILE_N, TILE_K],
        "names_sha256": hashlib.sha256(
            "\n".join(sorted(installed)).encode()
        ).hexdigest(),
        "counters": counts,
    }
