"""Build-independent initialisation for seeded tiny CPU fixtures.

Tiny random models stand in for real checkpoints in CPU oracles, and some of
those oracles depend on the exact weights a seed produces (which drafts are
rejected, which QSA block a draft closes).  MLX 0.32.4 changed the grouped
``nn.Conv1d``/``nn.Conv2d`` initialiser bound from ``1/sqrt(in * kernel)`` to
``1/sqrt(in / groups * kernel)``, so the same seed now builds a different GDN
depthwise convolution.  Constructing a fixture under ``legacy_conv_init()``
keeps the earlier bound: the weights, and the RNG draws that follow them, are
identical on every admitted MLX build.  Checkpoint loads are unaffected,
because loading replaces these weights.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import math


@contextlib.contextmanager
def legacy_conv_init():
    import mlx.core as mx
    from mlx import nn

    uniform = mx.random.uniform
    originals = {cls: cls.__init__ for cls in (nn.Conv1d, nn.Conv2d)}

    def wrap(original):
        signature = inspect.signature(original)

        @functools.wraps(original)  # nested use binds the real signature
        def init(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            kernel = values["kernel_size"]
            kernel = (kernel,) if isinstance(kernel, int) else tuple(kernel)
            if len(kernel) == 1 and original is originals[nn.Conv2d]:
                kernel = kernel * 2
            scale = math.sqrt(1 / (values["in_channels"] * math.prod(kernel)))

            def legacy_uniform(*a, **kw):
                return uniform(*a, **{**kw, "low": -scale, "high": scale})

            mx.random.uniform = legacy_uniform
            try:
                original(self, *args, **kwargs)
            finally:
                mx.random.uniform = uniform

        return init

    try:
        for cls, original in originals.items():
            cls.__init__ = wrap(original)
        yield
    finally:
        for cls, original in originals.items():
            cls.__init__ = original
        mx.random.uniform = uniform
