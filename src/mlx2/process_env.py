"""Process-wide numerics defaults that MLX latches once per process.

``MLX_ENABLE_TF32`` is read by MLX at the first fp32 matmul, quantized matmul
or attention dispatch on Metal and never re-read, so the value must be in the
environment before any such dispatch.  Pinning it in each adapter's serving
profile (fifteen copies by 2026-09-30) is the wrong layer: a harness that ran
one fp32 GEMM before constructing the adapter kept TF32 on while the receipt
said otherwise.  The package applies the default at import, before any
entrypoint can dispatch; adapters spread the same constant into their
profiles so receipts keep recording it.

An explicit value already in the environment is left alone at package import,
so tools that never construct an adapter may run with TF32.  Adapter profiles
do not honour it: every qualified profile and the APCv2 numerics identity are
pinned to "0", and once MLX has latched TF32 on there is no way to observe or
undo it, so a profile that wrote "0" over it would publish a receipt the
process does not run.  ``require_process_numerics`` therefore refuses a
TF32-enabling value instead (sweep 2026-10-02 P3): before a profile is pinned
(the Qwen adapters), and in ``import_env.assert_profile_applied`` for a value
that was present at package import and has since been overwritten.
"""

from __future__ import annotations

import os
from typing import Optional

TF32_ENV = "MLX_ENABLE_TF32"
PROCESS_NUMERICS = {TF32_ENV: "0"}

# The TF32-enabling value present in ``os.environ`` when the package defaults
# were first applied, if any; MLX may have latched it, whatever came later.
_EXPLICIT_AT_IMPORT: Optional[str] = None


class ProcessNumericsConflict(RuntimeError):
    """An explicit process numerics value conflicts with the pinned profile."""


def _enables_tf32(value) -> bool:
    return value is not None and value.strip() not in ("0", "")


def tf32_enabled(environ=None) -> bool:
    """Whether this process runs fp32 matmuls at TF32 (the owner of the
    variable reads it for kernels whose bits depend on it)."""
    target = os.environ if environ is None else environ
    return _enables_tf32(target.get(TF32_ENV, PROCESS_NUMERICS[TF32_ENV]))


def apply_process_numerics(environ=None) -> dict:
    """Set every default that is not already present; return what was set."""
    global _EXPLICIT_AT_IMPORT
    target = os.environ if environ is None else environ
    if environ is None and _EXPLICIT_AT_IMPORT is None and _enables_tf32(target.get(TF32_ENV)):
        _EXPLICIT_AT_IMPORT = target[TF32_ENV]
    applied = {}
    for name, value in PROCESS_NUMERICS.items():
        if name not in target:
            target[name] = value
            applied[name] = value
    return applied


def require_process_numerics(owner: str, environ=None) -> None:
    """Refuse an explicit TF32-enabling value a pinned profile would override."""
    target = os.environ if environ is None else environ
    value = target.get(TF32_ENV)
    if not _enables_tf32(value) and environ is None:
        value = _EXPLICIT_AT_IMPORT
    if _enables_tf32(value):
        raise ProcessNumericsConflict(
            f"{TF32_ENV}={value!r} was set explicitly, but {owner} pins "
            f"{TF32_ENV}={PROCESS_NUMERICS[TF32_ENV]!r} (its receipts and the APCv2 "
            f"numerics identity record that value) and MLX latches TF32 at the first "
            f"fp32 dispatch.  Unset {TF32_ENV} or set it to \"0\" before starting the "
            f"process."
        )


# Operator knobs mlx2 itself reads at call time, process-wide, that change
# only how often exact hybrid state is checkpointed (memory, not numerics):
# runtime/models/cache.py and the lane-budget estimators read them.  A
# profile's lab-namespace wipe used to delete them silently (sweep
# 2026-10-06 LEAD-02), so an operator's setting never reached the run.
PRESERVED_OPERATOR_KNOBS = frozenset(
    {"MLX_LM_STATE_CHECKPOINT_STRIDE", "MLX_LM_STATE_CHECKPOINT_MAX"}
)


def clear_inherited_profile(prefixes, environ=None) -> None:
    """Delete inherited lab experiment variables under ``prefixes`` before a
    profile is pinned, keeping ``PRESERVED_OPERATOR_KNOBS``."""
    target = os.environ if environ is None else environ
    for name in tuple(target):
        if name.startswith(tuple(prefixes)) and name not in PRESERVED_OPERATOR_KNOBS:
            del target[name]
