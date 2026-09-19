# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from contextvars import ContextVar
import mlx.core as mx

PRECISE_DTYPES = (mx.float32, mx.bfloat16)
_SOURCE = "\n    uint i = thread_position_in_grid.x;\n    T x = inp[i];\n    T y = 1 / (1 + metal::precise::exp(metal::abs(x)));\n    out[i] = (x < 0) ? y : 1 - y;\n"
_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="precise_sigmoid",
            input_names=["inp"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _kernel


def sigmoid(x: mx.array) -> mx.array:
    """``mx.sigmoid``'s bytes, in a primitive ``mx.compile`` cannot fuse."""
    if x.dtype not in PRECISE_DTYPES or not mx.metal.is_available():
        return mx.sigmoid(x)
    n = x.size
    return _get_kernel()(
        inputs=[x],
        template=[("T", x.dtype)],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


_IN_PRECISE_SPAN = ContextVar("mlx_lm_in_precise_span", default=False)


def gate_sigmoid(x: mx.array) -> mx.array:
    """``mx.sigmoid``, except inside a traced span where it would lose bits."""
    return sigmoid(x) if _IN_PRECISE_SPAN.get() else mx.sigmoid(x)
