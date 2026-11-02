# SPDX-License-Identifier: Apache-2.0
"""Row padding past MLX 0.32.4's legacy split-K quantized matmul (opt-in).

On NAX GPUs (M5), stock MLX 0.32.4 sends a transposed, non-batched
``quantized_matmul`` with ``M >= get_qmv_batch_limit(K, N)`` rows to
``qmm_splitk``.  That kernel only reaches the NAX ``qmm`` when its split
factor comes out as 1; otherwise it runs the legacy 32x32 non-NAX split-K
tiles (ml-explore/mlx issue #4720; the open fix is PR #4793).  For wide-K
projections (``o``, ``down``, GDN ``out``) this is 1.4-3x slower than the
NAX kernel the same shape gets a few rows later.

This module zero-pads such calls up to the first row count that 0.32.4
routes to the NAX ``qmm`` and slices the result back.  Padding is applied
only when the legacy split factor is at most ``MAX_SPLIT``: on the M5 Max
check (pad, matmul and slice timed together) every such cell won 1.2-2.7x,
split 5 tied (0.98-1.00x), and larger splits (narrow N: ``k/v``, small
shared experts) lose, because those calls are cheap and padding them costs
hundreds of extra rows (qualification/runs/qmm-row-pad-20261030).

The route model replicates 0.32.4's ``QuantizedMatmul::eval_gpu``,
``get_qmv_batch_limit`` (architecture generation 17, non-``d`` size) and
``qmm_splitk``; it is admitted only on those builds and devices and fails
closed elsewhere (the dispatch differs on MLX main and on M1-M4).

Not bitwise identical to stock: padded rows use the NAX ``qmm`` arithmetic
instead of the split-K partial sums.  Selection is gated by
``MLX2_QMM_ROW_PAD`` (default off).

Original mlx2 code; see provenance/qmm-row-pad.json.
"""

from __future__ import annotations

import re

import mlx.core as mx

ADMITTED_MLX_BUILDS = frozenset({"0.32.4"})
MAX_SPLIT = 4
BITS = (4,)
GROUP_SIZES = (64,)
DTYPES = (mx.bfloat16, mx.float16)
POLICY_ID = "splitk-le4-m5max-20261030"

STATS = {"routed": 0, "stock": 0}


def vector_limit(K: int, N: int) -> int:
    """``get_qmv_batch_limit`` for architecture generation >= 17, size != 'd'."""
    if K <= 2048 and N <= 2048:
        return 33
    if K <= 4096 and N <= 4096:
        return 25
    return 13


def split_k(M: int, N: int, K: int, group_size: int) -> int:
    """The split factor 0.32.4's ``qmm_splitk`` picks (1 means NAX ``qmm``)."""
    current = -(-N // 32) * -(-M // 32)
    split = max(1, 512 // current)
    k_align = group_size if group_size > 32 else 32
    split = min(split, K // k_align)
    while split > 1 and K % (split * k_align) != 0:
        split -= 1
    return split


def pad_target(M: int, K: int, N: int, group_size: int, *, max_split: int = MAX_SPLIT):
    """Rows to pad ``M`` to, or None when the call should stay on stock."""
    if K % 64 or M < vector_limit(K, N):
        return None
    split = split_k(M, N, K, group_size)
    if split <= 1 or split > max_split:
        return None
    target = M
    while split_k(target, N, K, group_size) > 1:
        target = 32 * (-(-target // 32)) + 1  # first row of the next 32-row tile
    return target


def device_admitted(architecture: str | None = None) -> bool:
    if architecture is None:
        if not mx.metal.is_available():
            return False
        architecture = mx.device_info().get("architecture", "")
    match = re.fullmatch(r"applegpu_g(\d+)([a-z])", architecture or "")
    return bool(match) and int(match.group(1)) >= 17 and match.group(2) != "d"


def refusal(build: str | None = None, architecture: str | None = None):
    """Why this process cannot use the row padding, or None."""
    if build is None:
        from ..mlx_build import installed_mlx_build

        build = installed_mlx_build()
    if build not in ADMITTED_MLX_BUILDS:
        return f"qmm_row_pad models MLX {sorted(ADMITTED_MLX_BUILDS)} dispatch, installed {build}"
    if not device_admitted(architecture):
        return "qmm_row_pad needs a NAX GPU of architecture generation 17 or later"
    return None


class _RowPadQuantizedLinear:
    """Mixin swapped onto selected ``nn.QuantizedLinear`` instances."""

    _row_pad_max_split = MAX_SPLIT

    def __call__(self, x):
        K = x.shape[-1]
        M = x.size // K if K else 0
        N = self["weight"].shape[0]
        target = None
        if x.dtype in DTYPES and M:
            target = pad_target(M, K, N, self.group_size, max_split=self._row_pad_max_split)
        if target is None:
            STATS["stock"] += 1
            return super().__call__(x)
        STATS["routed"] += 1
        rows = x.reshape(M, K)
        padded = mx.concatenate([rows, mx.zeros((target - M, K), dtype=x.dtype)], axis=0)
        return super().__call__(padded)[:M].reshape(*x.shape[:-1], -1)


_SUBCLASSES: dict = {}


def _subclass(cls):
    sub = _SUBCLASSES.get(cls)
    if sub is None:
        sub = type("RowPad" + cls.__name__, (_RowPadQuantizedLinear, cls), {})
        _SUBCLASSES[cls] = sub
    return sub


def apply(model, *, max_split: int = MAX_SPLIT):
    """Route eligible ``QuantizedLinear`` instances of ``model`` through the
    row padding.  Instances another mechanism already swapped (the lane
    matmul, sp_qmm) are left alone.  Returns a handle for :func:`remove`."""
    import mlx.nn as nn

    handle = []
    for _, module in model.named_modules():
        if (type(module) is nn.QuantizedLinear
                and getattr(module, "mode", "affine") == "affine"
                and module.bits in BITS and module.group_size in GROUP_SIZES):
            handle.append((module, type(module)))
            module.__class__ = _subclass(type(module))
            module._row_pad_max_split = max_split
    return handle


def remove(handle) -> None:
    for module, cls in handle:
        module.__class__ = cls
