"""Default-off, per-layer Agnes GDN projection-loading candidate.

The ordinary four-projection path remains the reference. This installer is
not called by the serving adapter: qualification and route admission must be
explicit before it can become a selected route.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mlx.core as mx
from mlx import nn


@dataclass(frozen=True)
class FusionReceipt:
    expected_layers: int
    fused_layers: int
    declined_layers: tuple[int, ...]
    activation_dtypes: tuple[str, ...]
    selected: bool = False


def _parts(layer):
    return tuple(getattr(layer, name) for name in (
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"
    ))


def _eligible(layer) -> bool:
    if hasattr(layer, "in_proj_fused") or getattr(layer, "sharding_group", None) is not None:
        return False
    try:
        parts = _parts(layer)
    except AttributeError:
        return False
    if not all(type(part) is nn.QuantizedLinear for part in parts):
        return False
    first = parts[0]
    if getattr(first, "mode", "affine") != "affine":
        return False
    return all(
        (part.group_size, part.bits, getattr(part, "mode", "affine"))
        == (first.group_size, first.bits, "affine")
        and "bias" not in part
        and part.get("biases") is not None
        and all(part[name].dtype == first[name].dtype for name in
                ("weight", "scales", "biases"))
        and part.weight.shape[1] == first.weight.shape[1]
        for part in parts
    )


def _concatenate(parts):
    return tuple(mx.concatenate([part[name] for part in parts], axis=0)
                 for name in ("weight", "scales", "biases"))


def _raw_bytes(array) -> bytes:
    import numpy as np

    dtype = getattr(mx, {1: "uint8", 2: "uint16", 4: "uint32", 8: "uint64"}[
        array.dtype.size
    ])
    return np.array(mx.view(array, dtype), copy=True).tobytes()


def probe_agnes_projection_layer(layer, *, dtypes=("bfloat16", "float16")) -> tuple[str, ...]:
    """Probe the loaded layer's real quantized weights on the current device."""
    if not _eligible(layer):
        return ()
    parts = _parts(layer)
    weight, scales, biases = _concatenate(parts)
    bounds = []
    total = 0
    for part in parts:
        total += part.weight.shape[0]
        bounds.append(total)
    hidden_size = parts[0].scales.shape[1] * parts[0].group_size
    passed = []
    for dtype_name in dtypes:
        dtype = getattr(mx, dtype_name)
        try:
            exact = True
            for batch, rows in ((1, 1), (1, 4), (1, 8), (2, 4)):
                inputs = (mx.random.normal(
                    (batch, rows, hidden_size), key=mx.random.key(batch * 100 + rows)
                ) * 0.3).astype(dtype)
                fused = mx.quantized_matmul(
                    inputs, weight, scales, biases, transpose=True,
                    group_size=parts[0].group_size, bits=parts[0].bits,
                )
                candidate = mx.split(fused, bounds[:-1], axis=-1)
                reference = tuple(part(inputs) for part in parts)
                mx.eval(candidate, reference)
                if any(_raw_bytes(a) != _raw_bytes(b)
                       for a, b in zip(candidate, reference)):
                    exact = False
                    break
            if exact:
                passed.append(dtype_name)
        except (RuntimeError, ValueError, TypeError):
            continue
    return tuple(passed)


def install_agnes_projection_fusion(model, *, enabled: bool = False) -> FusionReceipt:
    """Fuse only byte-proven loaded layers; never change serving selection."""
    if getattr(model, "model_type", None) != "agnes":
        raise ValueError("Agnes projection fusion requires an Agnes model")
    layers = tuple(model.layers)
    targets = tuple((index, layer.delta_attn) for index, layer in enumerate(layers)
                    if layer.is_linear)
    if not enabled:
        return FusionReceipt(len(targets), 0, (), ())
    declined = []
    installed = 0
    supported = set()
    for index, layer in targets:
        dtypes = probe_agnes_projection_layer(layer)
        if not dtypes:
            declined.append(index)
            continue
        parts = _parts(layer)
        weight, scales, biases = _concatenate(parts)
        mx.eval(weight, scales, biases)
        bounds = []
        total = 0
        for part in parts:
            total += part.weight.shape[0]
            bounds.append(total)
        fused = copy.deepcopy(parts[0])
        fused.weight, fused.scales, fused.biases = weight, scales, biases
        layer._gdn_fused_bounds = tuple(bounds)
        layer._gdn_fused_dtypes = frozenset(getattr(mx, name) for name in dtypes)
        layer.in_proj_fused = fused
        del layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a
        supported.update(dtypes)
        installed += 1
    return FusionReceipt(len(targets), installed, tuple(declined),
                         tuple(sorted(supported)))
