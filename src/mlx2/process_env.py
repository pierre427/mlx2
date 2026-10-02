"""Process-wide numerics defaults that MLX latches once per process.

``MLX_ENABLE_TF32`` is read by MLX at the first fp32 matmul, quantized matmul
or attention dispatch on Metal and never re-read, so the value must be in the
environment before any such dispatch.  Pinning it in each adapter's serving
profile (fifteen copies by 2026-09-30) is the wrong layer: a harness that ran
one fp32 GEMM before constructing the adapter kept TF32 on while the receipt
said otherwise.  The package applies the default at import, before any
entrypoint can dispatch; adapters spread the same constant into their
profiles so receipts keep recording it.  An explicit value already in the
environment wins.
"""

from __future__ import annotations

import os

TF32_ENV = "MLX_ENABLE_TF32"
PROCESS_NUMERICS = {TF32_ENV: "0"}


def tf32_enabled(environ=None) -> bool:
    """Whether this process runs fp32 matmuls at TF32 (the owner of the
    variable reads it for kernels whose bits depend on it)."""
    target = os.environ if environ is None else environ
    return target.get(TF32_ENV, PROCESS_NUMERICS[TF32_ENV]).strip() not in ("0", "")


def apply_process_numerics(environ=None) -> dict:
    """Set every default that is not already present; return what was set."""
    target = os.environ if environ is None else environ
    applied = {}
    for name, value in PROCESS_NUMERICS.items():
        if name not in target:
            target[name] = value
            applied[name] = value
    return applied
