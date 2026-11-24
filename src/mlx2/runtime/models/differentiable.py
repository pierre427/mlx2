"""Explicit differentiable-reference switch for research code that takes gradients.

MLX cannot differentiate through ``mx.fast.metal_kernel`` launches
(``[Primitive::vjp] Not implemented for CustomKernel``).  Serving routes pick
those kernels on the GPU; a caller that needs ``mx.grad`` through a model
forward opens :func:`reference_path` around the traced call, and every kernel
selection site that asks :func:`routed` takes its pure-MLX reference instead
(the same reference the CPU and ``module.train()`` already run).

Off by default, a ``ContextVar`` (per thread / task), never an environment
variable: serving paths, receipts and import-time profiles are unchanged.

The switch is read when Python runs the selection, i.e. while ``mx.grad`` /
``value_and_grad`` trace the function (which is when the backward graph is
built), not when the lazy graph is evaluated.  A site inside an
``mx.compile``-d function is fixed by that function's first trace: such a site
does not follow the switch, so keep selection sites outside compiled regions
(the one exception today, ``precise_ops.gate_sigmoid``, only selects a kernel
inside the GLM-5.3 compile regions' precise span).

Sites that honour it (keep ``ROUTED_KERNELS`` in step with the code):
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_ACTIVE: ContextVar[bool] = ContextVar("mlx2_differentiable_reference", default=False)

# site -> what it routes; the receipt names these and counts which ones a forward reached.
ROUTED_KERNELS = {
    "gated_delta": "gated_delta.gated_delta_update: GatedDeltaNet recurrence Metal kernel, chunked prefill scan "
                   "and MLX core scan -> gated_delta_ops",
    "fused_gdn": "qwen38_fused_gdn.GatedDeltaNet._try_fused_decode: fused GDN decode / verify / catch-up / "
                 "prefill-prework kernels -> GatedDeltaNet._recurrent_core",
    "sdpa": "base.scaled_dot_product_attention, plain (unquantized, unbucketed) cache only: opt-in FP verify "
            "(fp_verify_tensor) and FP decode (qsdpa_decode_metal) kernels -> mx.fast.scaled_dot_product_attention",
    "precise_sigmoid": "precise_ops.gate_sigmoid inside a precise span: precise sigmoid kernel -> mx.sigmoid",
    "kda_fused": "bailing_hybrid.KimiDeltaAttention.fused_decode: one-launch Ling KDA decode body "
                 "(ling_kda_fused) -> the eager KimiDeltaAttention chain",
}
if __doc__:  # None under python -OO
    __doc__ += "".join(f"\n* ``{k}``: {v}" for k, v in ROUTED_KERNELS.items()) + "\n"

_REACHED: dict[str, int] = {}


def active() -> bool:
    """True inside :func:`reference_path`."""
    return _ACTIVE.get()


def routed(site: str) -> bool:
    """A selection site's question: take the reference?  Counts the sites reached."""
    if not _ACTIVE.get():
        return False
    _REACHED[site] = _REACHED.get(site, 0) + 1
    return True


def reached() -> dict[str, int]:
    """Calls per site that took the reference since the last :func:`reset_reached`."""
    return dict(_REACHED)


def reset_reached() -> None:
    _REACHED.clear()


@contextmanager
def reference_path():
    """Route honouring kernel sites to their differentiable MLX reference."""
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
