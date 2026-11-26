# SPDX-License-Identifier: MIT
"""Muse-owned D128 row-exact attention windows.

The served Muse reference is one MLX vector-SDPA call per token row.  Each
row therefore has a different *physical* reduction domain: row ``r`` reads
exactly ``n_first + r`` keys.  A native multi-query call over the final K/V
width does not preserve that contract merely by masking its future keys; it
changes the kernel specialization and was rejected by the AST geometry gate.

This module transcribes the relevant MLX 0.32.4 vector-SDPA arithmetic into
Muse-owned D128 kernels.  The only window change is to place independent
serial rows on the grid and set ``N = n_first + row`` in the source.  Runs are
split whenever the one-row MLX plan changes.  Optional boolean masks retain
the concatenated one-token mask rows; all-valid rows stay on the unmasked
source, exactly as ordinary Muse decode does.

The source proof is MLX ``scaled_dot_product_attention.cpp`` and
``sdpa_vector.h`` at the revision and hashes recorded below (MIT).  This file
does not import, widen, or dispatch through the separately evidenced Qwen
D256 kernel.  Runtime use remains fail-closed behind Muse's build-, device-
and source-bound bitwise canary, and policy selection remains default off.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from functools import cache

import mlx.core as mx

from ..technologies.window_attention import vector_launch as _VL
from ..technologies.window_attention import vector_sources as _VS

MLX_SOURCE_REVISION = "39400a0d4cf1641bc6e543cb5e3585757f491854"
MLX_SDPA_DISPATCH_SHA256 = (
    "575e29a36b2d0c653f17b4f55062b644243d2e3dbc362f6725686bef959dd9dd"
)
# Content identity of the MLX vector-SDPA arithmetic transcribed here.  Kept
# separate from the dispatch hash so a source update cannot retain a stale
# numerical verdict by changing only one side of the proof.
MLX_SDPA_VECTOR_SHA256 = (
    "2100a4d1eaa8a524c5147c82c771cad75197495c72daffa03e7ea4c259aebf10"
)
ALGORITHM = "muse-d128-variable-n-v2"
HEAD_DIM = 128
MAX_ROWS = 16
MAX_VECTOR_SIMDS = 32
TWO_PASS_KEYS = 1024
_ONE_PASS_HPT = 2
_ONE_PASS_SG = 4


class ShapeOnly:
    """Array geometry for complete admission before the first live append."""

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.ndim = len(self.shape)


@cache
def _device_class() -> str:
    try:
        if not mx.metal.is_available():
            return ""
        return str(mx.device_info().get("architecture", ""))[-1:]
    except (AttributeError, RuntimeError, ValueError):
        return ""


def _select_blocks(device_class: str, n_keys: int, n_simds: int) -> int | None:
    """MLX 0.32.4 ``select_sdpa_blocks`` for local s/d device classes."""
    return _VL.select_sdpa_blocks(device_class, n_keys, n_simds)


def _serial_plan(device_class: str, n_keys: int, gqa: int):
    """The MLX vector plan of one ordinary Muse token row."""
    if device_class not in ("s", "d") or n_keys < 1 or not 1 <= gqa <= 32:
        return None
    if n_keys < TWO_PASS_KEYS:
        # The one-pass transcription assigns two query heads per simdgroup.
        return (1, None) if gqa % _ONE_PASS_HPT == 0 else None
    return (2, _select_blocks(device_class, n_keys, gqa))


def row_plans(
    n_first: int,
    rows: int,
    heads: int,
    kv_heads: int,
    *,
    device_class: str | None = None,
):
    """One-row plan for each exact physical reduction domain."""
    device_class = _device_class() if device_class is None else device_class
    if not kv_heads or heads % kv_heads:
        return ()
    gqa = heads // kv_heads
    return tuple(
        _serial_plan(device_class, n_first + row, gqa) for row in range(rows)
    )


def _plan_runs(plans):
    """Maximal consecutive row runs with the same ordinary one-row plan."""
    return tuple(_VL.plan_runs(plans))


def plan_signature(
    n_first: int,
    rows: int,
    heads: int,
    kv_heads: int,
    *,
    device_class: str | None = None,
):
    """Per-row serial plans grouped only at exact plan boundaries."""
    plans = row_plans(
        n_first, rows, heads, kv_heads, device_class=device_class
    )
    if not plans or any(plan is None for plan in plans):
        return ()
    return _plan_runs(plans)


def window_supported(queries, keys, values, n_first: int) -> str | None:
    if mx.default_device() != mx.gpu or _device_class() not in ("s", "d"):
        return "device"
    if not hasattr(mx.fast, "metal_kernel"):
        return "metal_kernel"
    if os.environ.get("MLX_SDPA_BLOCKS"):
        return "mlx_sdpa_blocks"
    # The ordinary Muse call can opt into a different experimental decoder.
    # Do not compare or replace that arithmetic through this capability.
    if os.environ.get("MLX2_FP_DECODE_KERNEL", "0") == "1" or os.environ.get("MLX2_FP_VERIFY_KERNEL") == "1":
        return "fp_decode_override"
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return "rank"
    batch, heads, rows, head_dim = queries.shape
    if batch != 1 or keys.shape[0] != 1 or values.shape[0] != 1:
        return "batch"
    if head_dim != HEAD_DIM or keys.shape[-1] != HEAD_DIM:
        return "head_dim"
    if not 2 <= rows <= MAX_ROWS:
        return "rows"
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return "dtype"
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return "kv_dtype"
    if (
        keys.shape[1] != values.shape[1]
        or keys.shape[2:] != values.shape[2:]
    ):
        return "kv_shape"
    if n_first < 1 or keys.shape[2] < n_first + rows - 1:
        return "key_count"
    if not plan_signature(n_first, rows, heads, keys.shape[1]):
        return "plan"
    return None


# The Metal text is the window-attention technology's vector-SDPA source
# generator: ``_VS.MUSE_D128_WINDOW`` (row ``r`` reads ``n_first + r`` keys,
# two-pass rows on the grid, concatenated one-token mask rows, the compact
# text style this module carried).  Byte-identical to this module's previous
# copy (tests/test_technologies_window_attention_sources.py); it does not
# share compiled kernels, names or admission with the Qwen D256 family.


def _one_pass_source(masked: bool) -> str:
    return _VS.kernel_kwargs(_VS.MUSE_D128_WINDOW, "one_chains", masked=masked)["source"]


def _two_pass_source(masked: bool) -> str:
    return _VS.kernel_kwargs(_VS.MUSE_D128_WINDOW, "two_chains", masked=masked)["source"]


_KERNELS: dict = {}


def _kernel(name: str):
    kernel = _KERNELS.get(name)
    if kernel is not None:
        return kernel
    masked = name.endswith("_mask")
    stage = name[: -len("_mask")] if masked else name
    if stage not in _VS.STAGES:  # pragma: no cover - every caller uses a fixed internal name
        raise KeyError(name)
    kernel = mx.fast.metal_kernel(
        **_VS.kernel_kwargs(_VS.MUSE_D128_WINDOW, stage, masked=masked)
    )
    _KERNELS[name] = kernel
    return kernel


_SCALARS: dict = {}
_INTS: dict = {}


def _scalar(value: float) -> mx.array:
    key = float(value)
    array = _SCALARS.get(key)
    if array is None:
        array = mx.array([key], dtype=mx.float32)
        _SCALARS[key] = array
    return array


def _int(value: int) -> mx.array:
    value = int(value)
    array = _INTS.get(value)
    if array is None:
        array = mx.array([value], dtype=mx.int32)
        if len(_INTS) < 4096:
            _INTS[value] = array
    return array


def window_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    *,
    n_first: int,
    masks: mx.array | None = None,
) -> mx.array:
    """Attention-only D128 rows over exactly ``n_first + row`` keys."""
    _, heads, rows, _head_dim = queries.shape
    kv_heads = keys.shape[1]
    plans = plan_signature(n_first, rows, heads, kv_heads)
    suffix = "_mask" if masks is not None else ""
    return _VL.run_window(
        lambda passes: _kernel(("one" if passes == 1 else "two") + "_chains" + suffix),
        lambda passes: _kernel(("one" if passes == 1 else "two") + "_combine"),
        queries,
        keys,
        values,
        _scalar(float(scale)),
        runs=plans,
        n_first=n_first,
        int_array=_int,
        hpt=_ONE_PASS_HPT,
        sg=_ONE_PASS_SG,
        masks=masks,
    )


@cache
def source_identity() -> str:
    """Bind numerical verdicts to both upstream proof and local Metal text."""
    pieces = [
        ALGORITHM,
        MLX_SOURCE_REVISION,
        MLX_SDPA_DISPATCH_SHA256,
        MLX_SDPA_VECTOR_SHA256,
        repr((HEAD_DIM, MAX_ROWS, TWO_PASS_KEYS, _ONE_PASS_HPT, _ONE_PASS_SG)),
        *(
            repr(sorted(_VS.kernel_kwargs(_VS.MUSE_D128_WINDOW, stage, masked=masked).items()))
            for stage, masked in (
                ("one_chains", False),
                ("one_chains", True),
                ("two_chains", False),
                ("two_chains", True),
                ("one_combine", False),
                ("two_combine", False),
            )
        ),
        *(
            inspect.getsource(fn)
            for fn in (
                _select_blocks,
                _VL.select_sdpa_blocks,
                _serial_plan,
                row_plans,
                _plan_runs,
                _VL.plan_runs,
                plan_signature,
                window_supported,
                _one_pass_source,
                _two_pass_source,
                _kernel,
                _scalar,
                _int,
                window_sdpa,
                _VL.run_window,
                _VL._one_pass,
                _VL._two_pass,
            )
        ),
    ]
    return hashlib.sha256("\n".join(pieces).encode()).hexdigest()
