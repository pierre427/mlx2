# SPDX-License-Identifier: Apache-2.0
"""SwiGLU producer for the existing packed Q4/Q5/Q8 INT8 consumers.

Inspired by Uzu's producer/consumer interface; see
provenance/uzu-metal-mining-20261027.json. No rotation or new cache format.
The product's two activation-dtype roundings are retained before the FP32
quantizer. The served sigmoid spelling is gated against the installed MLX.
"""

from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[3] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import hashlib
from functools import lru_cache

_PRODUCT = r"""
template <typename T>
inline float mlx2_swiglu_value(T gate, T up) {
    const T activated = gate * omlx_mlx_sigmoid<T>(gate);
    // PR #913: an ordinary float(bfloat(v)) round trip can be optimized away.
    const float rounded_activation = as_type<float>(uint(as_type<ushort>(activated)) << 16);
    const T product = T(rounded_activation * float(up));
    return as_type<float>(uint(as_type<ushort>(product)) << 16);
}
"""


def source(layout: str, *, tiled: bool = False) -> str:
    """Reuse the consumer's exact quantizer, replacing only its input loads."""
    from mlx2.runtime import int8_prefill as ip

    if layout not in ("q8", "q45"):
        raise ValueError("fused MLP layout must be q8 or q45")
    text = ip._Q8_STAGE_A_SRC if layout == "q8" else ip._Q45_STAGE_A_SRC
    pointer = "const device TX* xr = x + size_t(row) * K;"
    if text.count(pointer) != 1:
        raise RuntimeError("Stage A input pointer changed")
    text = text.replace(pointer, "const size_t input_offset = size_t(row) * K;")
    for index in ("base", "base + 32"):
        old = f"float(xr[{index}])"
        if text.count(old) != 2:
            raise RuntimeError("Stage A input loads changed")
        text = text.replace(
            old,
            f"mlx2_swiglu_value<TX>(gate[input_offset + {index}], up[input_offset + {index}])",
        )
    if tiled:
        text = text.replace("int(threadgroup_position_in_grid.x);", "int(threadgroup_position_in_grid.y);", 1)
        loop = "for (int g = sg; g < G; g += NSG) {"
        last = text.rfind(loop)
        if last < 0:
            raise RuntimeError("Stage A quantization loop changed")
        # Group64 has no cross-group reduction. Give each eight-group tile
        # its own threadgroup instead of serializing an entire wide MLP row.
        text = text[:last] + text[last:].replace(loop, """for (int g = int(threadgroup_position_in_grid.x) * NSG + sg;
         g < min(G, (int(threadgroup_position_in_grid.x) + 1) * NSG); g += NSG) {""", 1)
    return text


def revision() -> str:
    from mlx2.runtime.models.qwen4_routed_decode import SIGMOID

    programs = "".join(source(layout, tiled=tiled) for layout in ("q8", "q45") for tiled in (False, True))
    return hashlib.sha256((_PRODUCT + SIGMOID + programs).encode()).hexdigest()


@lru_cache(maxsize=32)
def _kernel(k, dtype, mode, layout, spelling):
    import mlx.core as mx

    from mlx2.runtime import int8_prefill as ip
    from mlx2.runtime.models.qwen4_routed_decode import SIGMOID
    from mlx2.runtime.models.served_exp import respell

    body = source(layout, tiled=mode == 1)
    helper = respell(SIGMOID, "metal::exp", spelling) + _PRODUCT
    digest = hashlib.sha256((helper + body).encode()).hexdigest()[:16]
    return mx.fast.metal_kernel(
        name=f"mlx2_fused_mlp_stage_a_{layout}_{k}_{ip._tname(dtype)}_{mode}_{digest}",
        input_names=["gate", "up", "m_dim"],
        output_names=["qa", "sa", "ra"],
        header=ip._Q8_HEADER + helper,
        source=ip._q8_prefix(K=k, ACT_MODE=mode, TX=ip._tname(dtype)) + body,
        ensure_row_contiguous=True,
    )


def prepare(gate, up, *, layout: str, act_scale: str = "group64", m_dim=None):
    """Return exactly the existing Stage A layout from two BF16 matrices."""
    import mlx.core as mx

    from mlx2.runtime import int8_prefill as ip
    from mlx2.runtime.models.qwen4_routed_decode import SWIGLU_GATE

    if layout not in ("q8", "q45") or act_scale not in ip.Q8_ACT_SCALES:
        raise ValueError("unsupported fused MLP layout or activation scale")
    if gate.ndim != 2 or gate.shape != up.shape or gate.dtype != mx.bfloat16 or up.dtype != gate.dtype:
        raise ValueError("fused MLP Stage A requires matching rank2 BF16 gate/up")
    m, k = gate.shape
    if m < 1 or k < 64 or k % 64:
        raise ValueError("fused MLP Stage A requires positive M and K divisible by 64")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise ValueError("fused MLP Stage A requires Metal")
    spelling = SWIGLU_GATE.spelling(gate.dtype)
    if spelling is None:
        raise ip.Int8PrefillError("fused MLP Stage A cannot reproduce served SwiGLU")
    mode = ip._Q8_ACT_MODE[act_scale]
    if m_dim is None:
        m_dim = mx.array([m], dtype=mx.int32)
    groups = k // 64
    grid = (((groups + 7) // 8) * 256, m, 1) if mode == 1 else (m * 256, 1, 1)
    return tuple(_kernel(k, gate.dtype, mode, layout, spelling)(
        inputs=[gate, up, m_dim], grid=grid, threadgroup=(256, 1, 1),
        output_shapes=[(m, k), (m,) if mode == 0 else (groups, m), (groups, m)],
        output_dtypes=[mx.int8, mx.float32, mx.int16],
    ))
