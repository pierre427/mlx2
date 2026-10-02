# SPDX-License-Identifier: MIT
"""Slice-invariant prefill lane (opt-in, default off).

Problem.  MLX picks kernels on the row count of each forward, and those
kernels round differently, so a prompt's prefilled state depends on how the
prompt was cut into forwards.  On Flash-Next the last-row logits of an 8K
prompt differ by up to 1.64 between 512- and 1024-row slices
(qualification/runs/prefill-slices-20261001).  The row-count switches, read
from MLX 39400a0d4:

* ``quantized_matmul`` with 2-D weights folds every row into one M and runs
  split-K (``qmm_splitk``) for narrow outputs, ``qmv``/``qmv_nax`` below
  ~18 rows, and the NAX ``qmm`` with a 32-row tile up to 32 rows and a
  64-row tile above (quantized.cpp ``QuantizedMatmul::eval_gpu``, ``qmm_nax``).
* a dense ``x @ W.T`` runs gemv/gemv_wide for few rows and NAX split-K while
  ``K >= 3 * max(M, N)`` (matmul.cpp ``steel_matmul_axpby``).
* ``gather_qmm`` streams each expert once (``gather_qmm_rhs``) only with at
  least 4 sorted rows per expert, else a per-row ``gather_qmv``.
* ``scaled_dot_product_attention`` at head dim 256 runs the fused kernel
  from 1024 query rows, an unfused matmul/softmax/matmul from 9 to 1023, and
  the vector kernel at 8 rows or fewer.

The lane (one ContextVar scope, opened only around the target trunk of a
prefill forward) runs every one of those on a single kernel configuration:

* affine quantized projections as a two-batch matmul with at least 33 rows
  per batch (broadcast weights): batched calls never split K and always take
  the 64-row NAX tile;
* dense projections the same way against a cached two-copy weight stack (a
  broadcast weight would be folded back into M by MLX);
* sorted expert gathers padded to MLX's streaming floor (4 rows per expert);
* attention forced onto the fused kernel (``force_fused``), with zero query
  rows in front of fewer than 9 rows.

Model-specific stages that pick their own routes by width (norm fast paths,
fused decode kernels, sparse-attention kernels) consult ``active()`` in the
model module.  Decode and verify forwards never open the scope, so they run
exactly the stock kernels; the only decode cost is one ContextVar read per
swapped module.

Design input: MTPLX PR #549 (youssofal/MTPLX, Apache-2.0), read for the
mechanism; this module is written for mlx2 and copies no code.  See
provenance/mtplx-549-invariant-prefill.json.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

SCHEMA = "mlx2.invariant-prefill.v1"

# Two batches of more than 32 rows: the batched NAX qmm then never splits K
# and always uses its 64-row tile (bm is 32 only while M <= 32).
LINEAR_BATCHES = 2
MIN_BATCH_ROWS = 33
# The fused attention kernel serves more than 8 query rows; 8 or fewer take
# the vector kernel whatever ``force_fused`` says.
MIN_FUSED_QUERY_ROWS = 9
FUSED_HEAD_DIMS = frozenset({64, 72, 80, 96, 128, 192, 256})
# MLX streams a sorted gather (gather_qmm_rhs) from 4 rows per expert, and
# only when B >= 16.
RHS_ROWS_PER_EXPERT = 4
RHS_MIN_ROWS = 16
# MTPLX #549 measured SwitchGLU rows invariant under padding from 16 experts
# up; with 8 experts MLX picked another expert kernel below 33 tokens.
MIN_EXPERTS = 16
# A dense projection is stacked twice (the batch stride must be nonzero);
# refuse anything larger than this per module rather than double big tables.
DENSE_STACK_LIMIT_BYTES = 64 << 20

_ACTIVE: ContextVar[bool] = ContextVar("mlx2_invariant_prefill", default=False)
_STATS: Counter = Counter()
_REASON_LIMIT = 24


def active() -> bool:
    """Whether the current forward is an invariant-lane prefill forward."""
    return _ACTIVE.get()


@contextmanager
def scope(enabled: bool = True):
    token = _ACTIVE.set(bool(enabled))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def note(key: str, amount: int = 1) -> None:
    _STATS[key] += int(amount)


def _bounded(prefix: str, reason: str) -> str:
    key = f"{prefix}:{reason}"
    if key not in _STATS and sum(1 for k in _STATS if k.startswith(prefix + ":")) >= _REASON_LIMIT:
        key = f"{prefix}:other"
    return key


def not_invariant(reason: str) -> None:
    """A lane forward ran a stage on a width-dependent route.

    A receipt may claim slice invariance only while this count stays zero.
    """
    _STATS[_bounded("not_invariant", reason)] += 1


def decline(reason: str) -> None:
    """A prefill-shaped forward did not open the lane (verify, speculation)."""
    _STATS[_bounded("declined", reason)] += 1


def status(*, reset: bool = False) -> Dict[str, Any]:
    counts = dict(_STATS)
    out = {
        "schema": SCHEMA,
        "counts": counts,
        "not_invariant": sum(v for k, v in counts.items() if k.startswith("not_invariant:")),
    }
    if reset:
        _STATS.clear()
    return out


def _rows(x: mx.array) -> mx.array:
    return x.reshape(-1, x.shape[-1])


def _batch_rows(rows: mx.array):
    """``rows`` [n, K] -> ([2, m, K] zero padded, n)."""
    n = int(rows.shape[0])
    m = max(MIN_BATCH_ROWS, -(-n // LINEAR_BATCHES))
    total = LINEAR_BATCHES * m
    if total > n:
        rows = mx.concatenate(
            [rows, mx.zeros((total - n, rows.shape[-1]), dtype=rows.dtype)], axis=0
        )
    note("linear_pad_rows", total - n)
    return rows.reshape(LINEAR_BATCHES, m, rows.shape[-1]), n


def _unbatch(out: mx.array, n: int, lead_shape) -> mx.array:
    out = out.reshape(-1, out.shape[-1])[:n]
    return out.reshape(*lead_shape, out.shape[-1])


def quantized_matmul(x, weight, scales, biases, *, group_size, bits, mode="affine"):
    """``x @ dequantize(weight).T`` whose rows do not depend on the row count."""
    batched, n = _batch_rows(_rows(x))

    def wide(a):
        return None if a is None else mx.broadcast_to(a, (LINEAR_BATCHES, *a.shape))

    out = mx.quantized_matmul(
        batched,
        wide(weight),
        wide(scales),
        wide(biases),
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    note("quantized_calls")
    return _unbatch(out, n, x.shape[:-1])


def dense_matmul(x, stacked_t, out_features=None):
    """``x @ W.T`` with ``stacked_t`` = [2, K, N] (two copies of ``W.T``)."""
    batched, n = _batch_rows(_rows(x))
    out = mx.matmul(batched, stacked_t)
    if out_features is not None and out.shape[-1] != out_features:
        out = out[..., :out_features]
    note("dense_calls")
    return _unbatch(out, n, x.shape[:-1])


def stack_dense(weight: mx.array) -> mx.array:
    """The two-copy ``W.T`` stack for ``dense_matmul``.

    A one-column projection (N == 1, e.g. a shared-expert gate) would run
    MLX's gemv, whose tiling follows the row count (Metal: 64-row slices
    differed from 512-row slices in one row of 8K); a zero second column
    keeps it on the GEMM, and ``dense_matmul`` drops it again.
    """
    wt = weight.T
    if wt.shape[-1] == 1:
        wt = mx.concatenate([wt, mx.zeros_like(wt)], axis=-1)
    return mx.contiguous(mx.stack([wt, wt]))


class InvariantQuantizedLinear(nn.QuantizedLinear):
    """``QuantizedLinear`` whose lane rows are row-count invariant.

    Outside the lane its call is the plain class's, so the decode kernels that
    transcribe a plain ``QuantizedLinear`` (HC decode, routed/shared decode,
    MoE windows) keep admitting it through the base-class markers they read.
    """

    _row_exact_base = nn.QuantizedLinear
    _mlx2_row_exact_base = nn.QuantizedLinear

    def __call__(self, x):
        if not _ACTIVE.get():
            return super().__call__(x)
        out = quantized_matmul(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if "bias" in self:
            out = out + self["bias"]
        return out


class InvariantLinear(nn.Linear):
    """Dense ``Linear`` whose lane rows are row-count invariant (see above)."""

    _row_exact_base = nn.Linear
    _mlx2_row_exact_base = nn.Linear

    def _lane_stack(self):
        weight = self["weight"]
        cached = getattr(self, "_mlx2_lane_stack", None)
        if cached is not None and cached[0] is weight:
            return cached[1]
        stacked = stack_dense(weight)
        object.__setattr__(self, "_mlx2_lane_stack", (weight, stacked))
        return stacked

    def __call__(self, x):
        if not _ACTIVE.get():
            return super().__call__(x)
        out = dense_matmul(x, self._lane_stack(), self["weight"].shape[0])
        if "bias" in self:
            out = out + self["bias"]
        return out


def sorted_gather_pad(rows: int, num_experts: int) -> int:
    """Pad rows that lift a sorted gather of ``rows`` to MLX's streaming floor."""
    floor = max(RHS_ROWS_PER_EXPERT * int(num_experts), RHS_MIN_ROWS)
    pad = max(0, floor - int(rows))
    note("moe_gathers")
    if pad:
        note("moe_pad_rows", pad)
    return pad


def sdpa_refusal(q, k, v, mask) -> Optional[str]:
    """Why ``sdpa`` cannot take this call, or None."""
    if isinstance(k, tuple) or isinstance(v, tuple):
        return "quantized_kv"
    if q.shape[-1] not in FUSED_HEAD_DIMS or v.shape[-1] != q.shape[-1]:
        return "head_dim"
    padded = max(int(q.shape[2]), MIN_FUSED_QUERY_ROWS)
    if isinstance(mask, str):
        if mask != "causal":
            return "mask_kind"
        if padded > int(k.shape[2]):
            return "causal_short_keys"
    elif mask is not None and mask.ndim != 4:
        return "mask_rank"
    return None


def sdpa(q, k, v, *, scale, mask):
    """Fused attention for every query count (call ``sdpa_refusal`` first).

    Fewer than 9 query rows get zero query rows in front (and copies of the
    first mask row); a query row's result depends only on its own row, so the
    real rows keep their bits and the front rows are dropped.
    """
    rows = int(q.shape[2])
    lead = max(0, MIN_FUSED_QUERY_ROWS - rows)
    if lead:
        shape = list(q.shape)
        shape[2] = lead
        q = mx.concatenate([mx.zeros(shape, dtype=q.dtype), q], axis=2)
        if mask is not None and not isinstance(mask, str) and mask.shape[-2] == rows:
            front = mx.broadcast_to(
                mask[..., :1, :], (*mask.shape[:-2], lead, mask.shape[-1])
            )
            mask = mx.concatenate([front, mask], axis=-2)
        note("sdpa_pad_rows", lead)
    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask, force_fused=True
    )
    note("sdpa_calls")
    return out[:, :, lead:] if lead else out


# --- installation ---------------------------------------------------------


def coverage_refusal(root: nn.Module) -> Optional[str]:
    """Why the lane cannot cover every projection under ``root``, or None.

    Half a lane would change prefill bits without the invariance that makes
    the change worth having, so any uncovered projection refuses the lane.
    """
    from .switch_layers import QuantizedSwitchLinear, SwitchLinear

    for name, module in root.named_modules():
        cls = type(module)
        if cls in (nn.QuantizedLinear, InvariantQuantizedLinear):
            if module.mode != "affine":
                return f"quantized_mode:{module.mode}:{name}"
        elif cls in (nn.Linear, InvariantLinear):
            nbytes = int(module["weight"].nbytes)
            if 4 * nbytes > DENSE_STACK_LIMIT_BYTES:
                return f"dense_too_large:{name}"
        elif isinstance(module, (nn.Linear, nn.QuantizedLinear)):
            return f"unsupported_linear:{cls.__name__}:{name}"
        elif isinstance(module, QuantizedSwitchLinear):
            if cls is not QuantizedSwitchLinear:
                return f"unsupported_switch:{cls.__name__}:{name}"
            if module.mode != "affine":
                return f"switch_mode:{module.mode}:{name}"
            if module.num_experts < MIN_EXPERTS:
                return f"switch_experts:{module.num_experts}:{name}"
        elif isinstance(module, SwitchLinear):
            return f"dense_switch:{name}"
    return None


def swap_projections(root: nn.Module) -> Dict[str, int]:
    """Swap every exact ``QuantizedLinear``/``Linear`` class under ``root``.

    Class swaps only: the parameter tree and the decode arithmetic stay as
    loaded.
    """
    report = {"quantized_linears": 0, "dense_linears": 0}
    for _name, module in root.named_modules():
        cls = type(module)
        if cls is nn.QuantizedLinear:
            module.__class__ = InvariantQuantizedLinear
            report["quantized_linears"] += 1
        elif cls is nn.Linear:
            module.__class__ = InvariantLinear
            report["dense_linears"] += 1
    return report


def restore_projections(root: nn.Module) -> None:
    for _name, module in root.named_modules():
        if type(module) is InvariantQuantizedLinear:
            module.__class__ = nn.QuantizedLinear
        elif type(module) is InvariantLinear:
            object.__setattr__(module, "_mlx2_lane_stack", None)
            module.__class__ = nn.Linear


def prefill_decline_reason(inputs, cache) -> Optional[str]:
    """None when a trunk forward is a prefill forward the lane may take."""
    from .. import row_exact_verify, verify_scope

    if inputs is None or inputs.ndim < 2 or int(inputs.shape[1]) < 2:
        return "single_row"
    if verify_scope.active():
        return "verify_scope"
    if row_exact_verify.active():
        return "row_exact_window"
    if cache is not None and any(
        bool(getattr(entry, "speculating", False)) for entry in cache if entry is not None
    ):
        return "speculating"
    return None


@lru_cache(maxsize=16)
def scoped_trunk_type(original):
    """A subclass of the trunk class that opens the lane for prefill forwards.

    The trunk's ``__call__(inputs, cache, ...)`` is the boundary: everything
    inside it is the target's prefill arithmetic; the LM head and the MTP
    head stay outside.  A nested call (a trunk that recurses per token)
    keeps the outer decision.
    """

    class InvariantPrefillTrunk(original):
        def __call__(self, inputs, cache=None, *args, **kwargs):
            if _ACTIVE.get():
                return super().__call__(inputs, cache, *args, **kwargs)
            handle = getattr(self, "_mlx2_invariant_prefill", None)
            if handle is None or not handle.enabled:
                return super().__call__(inputs, cache, *args, **kwargs)
            reason = prefill_decline_reason(inputs, cache)
            if reason is not None:
                if reason != "single_row":
                    decline(reason)
                return super().__call__(inputs, cache, *args, **kwargs)
            note("forwards")
            note("rows", int(inputs.shape[0]) * int(inputs.shape[1]))
            with scope(True):
                return super().__call__(inputs, cache, *args, **kwargs)

    InvariantPrefillTrunk.__name__ = f"InvariantPrefill{original.__name__}"
    InvariantPrefillTrunk.__qualname__ = InvariantPrefillTrunk.__name__
    return InvariantPrefillTrunk


class InvariantPrefill:
    """Install handle: what was swapped, or why nothing was."""

    def __init__(self, trunk, refusal: Optional[str], report: Dict[str, int]):
        self.trunk = trunk
        self.refusal = refusal
        self.report = report
        self.enabled = refusal is None

    @property
    def installed(self) -> bool:
        return self.refusal is None

    def identity(self) -> Dict[str, Any]:
        """The numerical law the lane imposes (for the APCv2 prefill identity)."""
        return {
            "schema": SCHEMA,
            "law": {
                "linear_batches": LINEAR_BATCHES,
                "min_batch_rows": MIN_BATCH_ROWS,
                "min_fused_query_rows": MIN_FUSED_QUERY_ROWS,
                "rhs_rows_per_expert": RHS_ROWS_PER_EXPERT,
            },
        }

    def audit(self) -> Optional[str]:
        """Re-check coverage after other mechanisms installed their modules."""
        if not self.installed:
            return self.refusal
        return coverage_refusal(self.trunk)

    def status(self, *, reset: bool = False) -> Dict[str, Any]:
        base = {"installed": self.installed, "enabled": self.enabled}
        if self.refusal is not None:
            base["reason"] = self.refusal
        else:
            base["report"] = dict(self.report)
        base.update(status(reset=reset))
        return base

    def uninstall(self) -> None:
        if not self.installed:
            return
        restore_projections(self.trunk)
        cls = type(self.trunk)
        if cls.__name__.startswith("InvariantPrefill"):
            self.trunk.__class__ = cls.__mro__[1]
        object.__setattr__(self.trunk, "_mlx2_invariant_prefill", None)
        self.enabled = False
        self.refusal = "uninstalled"


def install(trunk: nn.Module, *, extra_refusal: Optional[str] = None) -> InvariantPrefill:
    """Install the lane on ``trunk`` (the module whose call is one trunk forward).

    All or nothing: a refusal installs nothing and is reported by the handle.
    """
    existing = getattr(trunk, "_mlx2_invariant_prefill", None)
    if existing is not None:
        return existing
    refusal = extra_refusal or coverage_refusal(trunk)
    if refusal is not None:
        return InvariantPrefill(trunk, refusal, {})
    report = swap_projections(trunk)
    trunk.__class__ = scoped_trunk_type(type(trunk))
    handle = InvariantPrefill(trunk, None, report)
    object.__setattr__(trunk, "_mlx2_invariant_prefill", handle)
    return handle
