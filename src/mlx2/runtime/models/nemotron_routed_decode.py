"""One-token Nemotron-H routed experts in fewer launches, stock-exact.

Original mlx2 code after the design of ``qwen4_routed_decode`` (omlx #3912 /
#4113 concept); the arithmetic is the installed MLX's own.  At one decode
token the stock routed path is ``SwitchMLP``: an argsort of the 22 slots,
gathers, ``gather_qmm`` for fc1, ReLU2, ``gather_qmm`` for fc2, the unsort and
its gathers.  With 22 rows over 512 experts MLX's ``GatherQMM`` takes no
streaming kernel (that needs >= 4 sorted rows per expert): each row runs the
one-row ``fp_qmv_fast`` / ``fp_qmv`` on its expert, and sorting only permutes
rows.  Here one launch per projection runs exactly those one-row
implementations (``fp_row_exact_qmv`` header) with every slot's expert offset
read on the device, in slot order, so the stock ReLU2 between them and the
stock weighted combine after them see the same values.  NVFP4 experts with
folded ``weight_scale_2`` only (``fold_relu2_expert_scales``); anything else
declines and the caller runs ``SwitchMLP``.  Default on
(``MLX2_NEMOTRON_ROUTED_DECODE=off`` disables).  Provenance:
provenance/nemotron-routed-decode.json.
"""

from __future__ import annotations

import os
import threading

import mlx.core as mx

from . import fp_row_exact_qmv as fq

_KERNELS: dict = {}
_LOCK = threading.Lock()
COUNTS = {"engaged": 0, "declined": 0}
_SOURCE = """
    const int K = params[0];
    const int N = params[1];
    const uint slot = threadgroup_position_in_grid.z;
    const uint expert = indices[slot];
    const device uint32_t* we = w + (size_t)expert * (size_t)N * (size_t)(K * BITS / 32);
    const device uint8_t* se = scales + (size_t)expert * (size_t)N * (size_t)(K / GS);
    const device T* xs = x + (XSLOT ? (size_t)slot * (size_t)K : 0);
    device T* ys = y + (size_t)slot * (size_t)N;
    uint3 tg = uint3(0, threadgroup_position_in_grid.y, 0);
    IMPL
"""
_FAST = ("fp_qmv_fast_impl<T, GS, BITS, false, false, 4>(we, se, (const device float*)nullptr, "
         "xs, ys, K, N, tg, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);")
_GENERAL = ("fp_qmv_impl<T, GS, BITS, false>(we, se, (const device float*)nullptr, xs, ys, K, N, "
            "tg, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);")


def enabled() -> bool:
    """On by default since 2026-11-10 (Pierre), after routed2 (64 greedy steps,
    logits bitwise identical) and unit7 (served ordinary GSM8K 37.7 -> 40.0
    tok/s, native MTP 52.8, greedy-equal); ``MLX2_NEMOTRON_ROUTED_DECODE=off``
    restores ``SwitchMLP``.  It only engages on folded NVFP4 experts."""
    mode = os.environ.get("MLX2_NEMOTRON_ROUTED_DECODE", "on")
    if mode not in {"off", "on"}:
        raise ValueError("MLX2_NEMOTRON_ROUTED_DECODE must be off or on")
    return mode == "on"


def _kernel(fast: bool):
    with _LOCK:
        kernel = _KERNELS.get(fast)
        if kernel is None:
            kernel = mx.fast.metal_kernel(
                name="mlx2_nemotron_routed_" + ("fast" if fast else "general"),
                input_names=["x", "w", "scales", "indices", "params"],
                output_names=["y"],
                header=fq._header(),
                source=_SOURCE.replace("IMPL", _FAST if fast else _GENERAL),
            )
            _KERNELS[fast] = kernel
        return kernel


_PARAMS: dict = {}


def _params(k: int, n: int):
    key = (k, n)
    value = _PARAMS.get(key)
    if value is None:
        value = _PARAMS[key] = mx.array([k, n], dtype=mx.int32)
    return value


def gather_rows(x, layer, indices, *, x_per_slot: bool):
    """``[slots, N]``: row s is the stock one-row qmv of expert ``indices[s]``.

    ``x`` is ``[1, K]`` (one token for every slot) or ``[slots, K]``; inputs
    that are not row-contiguous are copied by ``metal_kernel`` itself.
    """
    weight, scales = layer["weight"], layer["scales"]
    n, k = int(weight.shape[1]), int(x.shape[-1])
    slots = int(indices.size)
    flat = indices.reshape(-1)
    if flat.dtype != mx.uint32:
        flat = flat.astype(mx.uint32)
    fast = fq.fast_layout(k, n, 4)
    (y,) = _kernel(fast)(
        inputs=[x, weight, scales, flat, _params(k, n)],
        template=[("T", x.dtype), ("GS", 16), ("BITS", 4), ("XSLOT", int(x_per_slot))],
        grid=(32, 2 * ((n + 7) // 8), slots),
        threadgroup=(32, 2, 1),
        output_shapes=[(slots, n)],
        output_dtypes=[x.dtype],
    )
    return y


def _static_reason(switch, k: int) -> str | None:
    """The per-module part of the admission (types, folding, geometry)."""
    from ..nvfp4_linear import ScaledNVFP4SwitchLinear

    fc1, fc2 = getattr(switch, "fc1", None), getattr(switch, "fc2", None)
    if not (isinstance(fc1, ScaledNVFP4SwitchLinear) and isinstance(fc2, ScaledNVFP4SwitchLinear)):
        return "not_scaled_nvfp4"
    if not (fc1.scale_folded and fc2.scale_folded):
        return "scales_not_folded"
    for layer, width in ((fc1, k), (fc2, int(fc1.weight.shape[1]))):
        weight, scales = layer["weight"], layer["scales"]
        n = int(weight.shape[1])
        if (int(weight.shape[2]) * 32 != width * 4 or int(scales.shape[2]) * 16 != width
                or width <= 128 or n < 8):
            return "geometry"
    return None


def decline_reason(switch, x, indices) -> str | None:
    from . import invariant_prefill as invariant

    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != 1:
        return "not_one_token"
    if x.dtype not in (mx.bfloat16, mx.float16) or indices.shape[:2] != (1, 1):
        return "layout"
    k = int(x.shape[-1])
    # Weight types, folding and geometry do not change after load; held on
    # the module next to the arrays they were derived from.
    held = switch.__dict__.get("_mlx2_routed_decode_static")
    fc1, fc2 = getattr(switch, "fc1", None), getattr(switch, "fc2", None)
    if fc1 is None or fc2 is None:
        return _static_reason(switch, k)
    # Keyed on the weight arrays and the fold flags (folding changes the
    # flags, not the arrays).
    key = (k, fc1.get("weight"), fc2.get("weight"),
           getattr(fc1, "scale_folded", None), getattr(fc2, "scale_folded", None))
    if held is None or held[0][0] != key[0] or any(
        a is not b for a, b in zip(held[0][1:3], key[1:3])
    ) or held[0][3:] != key[3:]:
        held = (key, _static_reason(switch, k))
        object.__setattr__(switch, "_mlx2_routed_decode_static", held)
    if held[1] is not None:
        return held[1]
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return "device"
    if invariant.active():
        return "invariant_prefill"
    return None


def routed_experts(switch, x, indices):
    """``switch(x, indices)`` for one token: ``[1, 1, k, D]``, or None."""
    if decline_reason(switch, x, indices) is not None:
        COUNTS["declined"] += 1
        return None
    COUNTS["engaged"] += 1
    hidden = switch.activation(
        gather_rows(x.reshape(1, x.shape[-1]), switch.fc1, indices, x_per_slot=False))
    out = gather_rows(hidden, switch.fc2, indices, x_per_slot=True)
    return out.reshape(1, 1, int(indices.shape[-1]), out.shape[-1])


__all__ = ["COUNTS", "decline_reason", "enabled", "gather_rows", "routed_experts"]
