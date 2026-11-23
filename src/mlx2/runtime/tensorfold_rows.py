"""Opt-in TensorFold row law for an adapter's dense verify-path projections.

Family-agnostic installer over the existing TensorFold-derived row matvec
(``runtime/models/flash_tensorfold_qmv.py``; TensorFold bb4b4a35, MIT, see
provenance/LICENSE.TensorFold-MIT and provenance/tensorfold-rows-installer.json).
The adapter names the module subtrees and the name parts to leave alone; this
module never branches on a model family.

One numerical law runs at every row count from 1 to ``MAX_ROWS``: one-row
decode, a plain-anchor step and a packed multi-row verify all compute each
row with the same arithmetic, independent of how many rows share the call.
Calls outside that band (prefill chunks, wide cohorts) or with a non-bf16
activation keep the stock ``nn.QuantizedLinear`` path and are counted.

Backends:

* ``metal``: the Flash TensorFold row kernel (one simdgroup per row).
* ``metal_shared2``: its two-rows-per-simdgroup variant (decoded weights
  shared by a row pair; same per-row arithmetic, candidate only).
* ``reference``: a CPU transcription of the same row law in plain MLX ops,
  for oracle tests.  It follows the kernel's bf16 activation-sum chain and
  its per-group ``scale * dot + sum * bias`` form; its fp32 summation order
  is MLX's, not the GPU simdgroup's, so it is not bit-identical to Metal.

Everything here is implemented and unqualified.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from contextlib import contextmanager

import mlx.core as mx
from mlx import nn

MAX_ROWS = 17  # the row kernel's admitted band (1..17 rows per call)
BACKENDS = ("metal", "metal_shared2", "reference")
KERNEL_SOURCE_REVISION = "bb4b4a35863af562fc4ccb2586300d8f94b5d6de"
LAW = "tensorfold-rows-affine-q4q8-g64-bf16-v1"


def refusal_reason(module) -> str | None:
    """``None`` when the row law admits ``module``; otherwise why it does not."""
    if not isinstance(module, nn.QuantizedLinear):
        return "not_quantized_linear"
    if getattr(module, "mode", "affine") != "affine" or "biases" not in module:
        return "not_affine"
    bits = int(module.bits)
    if bits not in (4, 8):
        return f"bits_{bits}"
    if int(module.group_size) != 64:
        return f"group_size_{int(module.group_size)}"
    if module.scales.dtype != mx.bfloat16 or module.biases.dtype != mx.bfloat16:
        return "scales_not_bf16"
    n = int(module.weight.shape[0])
    k = int(module.weight.shape[1]) * (32 // bits)
    if k % 512:
        return "k_not_multiple_of_512"
    if n % 4:
        return "n_not_multiple_of_4"
    return None


def _codes(module):
    """Unsigned quantized codes ``[N, K]`` (MLX affine packing, low bits first)."""
    raw = mx.view(module.weight, mx.uint8)
    if int(module.bits) == 8:
        return raw.astype(mx.float32)
    lo = (raw & 0x0F).astype(mx.float32)
    hi = (raw >> 4).astype(mx.float32)
    return mx.stack([lo, hi], axis=-1).reshape(raw.shape[0], -1)


def _bf(v):
    return v.astype(mx.bfloat16).astype(mx.float32)


def reference_rows(x, module):
    """CPU transcription of the TensorFold row law (row-independent by construction).

    Per row and per 16-element chunk (one simdgroup lane's slice of a
    512-element block): the activation sum is the kernel's bf16 chain
    ``bf16(bf16(bf16(a + b) + c) + d)`` over four quads, summed in fp32; the
    chunk's contribution is ``scale * dot(x, q) + sum * bias`` of its group;
    lanes accumulate over the blocks, then reduce, and the row rounds to bf16.
    """
    if refusal_reason(module) is not None:
        raise ValueError("TensorFold row law requires affine q4/q8-g64 bf16 projections")
    shape = x.shape
    k = int(shape[-1])
    n = int(module.weight.shape[0])
    if k != int(module.weight.shape[1]) * (32 // int(module.bits)):
        raise ValueError("TensorFold row law activation width mismatch")
    rows = x.reshape(-1, k)
    chunks, blocks = k // 16, k // 512
    codes = _codes(module).reshape(n, chunks, 16)
    group = mx.arange(chunks) * 16 // 64
    scales = mx.take(module.scales.astype(mx.float32), group, axis=1)  # [N, chunks]
    biases = mx.take(module.biases.astype(mx.float32), group, axis=1)
    out = []
    for r in range(rows.shape[0]):
        xr = rows[r].astype(mx.float32)
        quads = xr.reshape(chunks, 4, 4)
        chain = _bf(_bf(_bf(quads[..., 0] + quads[..., 1]) + quads[..., 2]) + quads[..., 3])
        sums = ((chain[:, 0] + chain[:, 1]) + chain[:, 2]) + chain[:, 3]  # [chunks]
        dot = (codes * xr.reshape(1, chunks, 16)).sum(axis=-1)  # [N, chunks]
        part = (scales * dot + biases * sums[None, :]).reshape(n, blocks, 32)
        lanes = part[:, 0]
        for b in range(1, blocks):
            lanes = lanes + part[:, b]
        out.append(lanes.sum(axis=-1).astype(mx.bfloat16))
    y = mx.stack(out).reshape(*shape[:-1], n)
    if "bias" in module:
        y = y + module.bias
    return y


class TensorFoldRows:
    """One installation: its modules, backend, refusals and call counters."""

    def __init__(self, backend: str):
        if backend not in BACKENDS:
            raise ValueError(f"TensorFold rows backend must be one of {BACKENDS}")
        self.backend = backend
        self.enabled = True
        self.installed: list[str] = []
        self.refused: dict[str, str] = {}
        self.modules: list[nn.QuantizedLinear] = []
        self.counts = Counter()
        self.row_histogram = Counter()
        self._scope = "other"

    @contextmanager
    def scope(self, name: str):
        """Attribute the calls made inside to ``name`` (e.g. the AST verify)."""
        previous, self._scope = self._scope, str(name)
        try:
            yield
        finally:
            self._scope = previous

    def rows(self, x, module):
        if self.backend == "reference":
            return reference_rows(x, module)
        from .models.flash_tensorfold_qmv import qmv_rows

        return qmv_rows(x, module, shared_weights=self.backend == "metal_shared2")

    def uninstall(self):
        """Restore every installed module to the stock class (A/B on one load)."""
        for module in self.modules:
            module.__class__ = nn.QuantizedLinear
        self.modules = []
        self.enabled = False

    def counters(self) -> dict:
        return {**dict(self.counts),
                "row_histogram": {str(k): v for k, v in sorted(self.row_histogram.items())}}

    def receipt(self) -> dict:
        counts = self.counts
        names = "\n".join(sorted(self.installed))
        reasons = Counter(self.refused.values())
        return {
            "implemented": True,
            "qualified": False,
            "selected": bool(self.installed) and self.enabled,
            "observed_used": counts["kernel_calls"] > 0,
            "verify_observed_used": counts["scope_ast_verify_kernel_calls"] > 0,
            "law": LAW,
            "backend": self.backend,
            "kernel_source_revision": KERNEL_SOURCE_REVISION,
            "max_rows": MAX_ROWS,
            "installed": len(self.installed),
            "names_sha256": hashlib.sha256(names.encode()).hexdigest(),
            "refused": len(self.refused),
            "refused_by_reason": dict(sorted(reasons.items())),
            "refused_names": dict(sorted(self.refused.items())),
            "counters": self.counters(),
        }


class TensorFoldRowsLinear(nn.QuantizedLinear):
    """``nn.QuantizedLinear`` whose 1..17-row bf16 calls run the row law."""

    def __call__(self, x):
        owner = self._tensorfold_rows
        if not owner.enabled:
            return nn.QuantizedLinear.__call__(self, x)
        counts = owner.counts
        rows = x.size // x.shape[-1]
        if x.dtype != mx.bfloat16:
            counts["stock_calls_dtype"] += 1
            return nn.QuantizedLinear.__call__(self, x)
        if not 1 <= rows <= MAX_ROWS:
            counts["stock_calls_rows"] += 1
            counts[f"scope_{owner._scope}_stock_calls_rows"] += 1
            return nn.QuantizedLinear.__call__(self, x)
        counts["kernel_calls"] += 1
        counts["kernel_rows"] += rows
        counts[f"scope_{owner._scope}_kernel_calls"] += 1
        if rows > 1:
            counts["multi_row_kernel_calls"] += 1
        owner.row_histogram[rows] += 1
        return owner.rows(x, self)


def _metal_ready() -> bool:
    try:
        return bool(mx.metal.is_available()) and mx.default_device() == mx.gpu
    except (AttributeError, RuntimeError):
        return False


def install(roots, *, skip_parts=(), backend="metal") -> TensorFoldRows:
    """Install the row law on the eligible projections under ``roots``.

    ``roots`` maps a display prefix to a module subtree; only plain
    ``nn.QuantizedLinear`` modules are candidates (embeddings, expert banks
    and unquantized layers are not projections of this law).  A candidate
    whose name has a part in ``skip_parts`` is left alone silently; a
    candidate the law does not admit is refused, recorded, and keeps the
    stock path.  Weights are never copied.  Fails closed when nothing is
    admitted or when a Metal backend has no Metal device.
    """
    owner = TensorFoldRows(backend)
    if backend != "reference" and not _metal_ready():
        raise ValueError("TensorFold rows Metal backend needs a Metal default device "
                         "(the reference backend is the CPU oracle)")
    skip = frozenset(skip_parts)
    seen = set()
    for prefix, root in dict(roots).items():
        for name, module in root.named_modules():
            if type(module) is not nn.QuantizedLinear or id(module) in seen:
                continue
            full = f"{prefix}.{name}" if name else prefix
            if skip.intersection(full.split(".")):
                continue
            seen.add(id(module))
            reason = refusal_reason(module)
            if reason is not None:
                owner.refused[full] = reason
                continue
            module.__class__ = TensorFoldRowsLinear
            object.__setattr__(module, "_tensorfold_rows", owner)
            owner.installed.append(full)
            owner.modules.append(module)
    if not owner.installed:
        raise ValueError("TensorFold rows found no eligible affine q4/q8-g64 bf16 projections "
                         f"(refused: {dict(Counter(owner.refused.values()))})")
    return owner
