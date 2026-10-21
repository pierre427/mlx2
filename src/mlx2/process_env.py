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


# Host/device synchronization mechanism, latched by MLX when Metal starts.
# It never changes a computed bit: MLX_METAL_FAST_SYNCH makes a host wait on
# device work poll a shared event instead of blocking in Metal's completion
# wait.  Each blocking wait cost ~0.15 ms of wake-up latency on M5 Max, paid
# at every host read of a device result (two to three per speculative tree
# round, plus the ordinary decode's per-token read), so the default is on.
# An explicit value (``0`` included) is left alone.
PROCESS_RUNTIME_DEFAULTS = {"MLX_METAL_FAST_SYNCH": "1"}


def apply_process_runtime_defaults(environ=None) -> dict:
    """Set each runtime default that is not already present; return what was set."""
    target = os.environ if environ is None else environ
    applied = {}
    for name, value in PROCESS_RUNTIME_DEFAULTS.items():
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
    {
        "MLX_LM_STATE_CHECKPOINT_STRIDE",
        "MLX_LM_STATE_CHECKPOINT_MAX",
        # Request-local observability; these do not select model math.
        "MLX_LM_SYNC_TRACE",
        "MLX_LM_SEGMENTED_SELF_MTP_TIMING",
        # Documented process-owner choices.  NAX_GATHER and RHS_PAD_POLICY
        # are rebound and conflict-checked by adapters.process_globals; the
        # other values tune those modes and are exposed in their status.
        "MLX2_MOE_NAX_GATHER",
        "MLX2_MOE_NAX_GATHER_PLAN",
        "MLX2_MOE_NAX_GATHER_BITS",
        "MLX2_MOE_RHS_PAD_POLICY",
        "MLX2_MOE_RHS_PAD_MIN_ROWS",
        # Source ownership for the explicitly selected TensorFold route.
        "MLX2_TENSORFOLD_SOURCE",
    }
)

# Model profiles are process-global.  Every adapter profile transition clears
# these selector namespaces before pinning its declared environment, so a
# previous model's kernel or external-route choice cannot survive outside the
# new adapter receipt.  Exact operator-owned values are preserved above, and
# so is whatever the operator set under these namespaces before the first
# profile applied (``_OPERATOR_SELECTORS``; sweep 2026-10-12 L1-F2).
# MLX_ENABLE_TF32 is deliberately absent: require_process_numerics must reject
# an explicit enabling value instead of erasing evidence that MLX may have
# latched it.
INHERITED_MODEL_PROFILE_PREFIXES = (
    "MLX_QWEN",
    "MLX_LM_",
    "MLXUAG_",
    "MLX_GDN_",
    "MLX2_DFLASH_",
    "MLX2_FUSED_SDPA_",
    "MLX2_MOE_",
    "MLX2_QSDPA_",
    "MLX2_QWEN38_",
    "MLX2_QWEN_",
    "MLX2_TENSORFOLD_",
    "MLX2_TREE_",
    "MLX2_XING_",
    "MLX2_EAGER_DISPATCH_",
    "MLX_SDPA_",
    "MLX_LAGUNA_",
)

# These process-global selections are applied through live setters and held
# by adapters.process_globals.  Import snapshots may legitimately contain an
# earlier value; the adapter claim, rather than environment text, is the
# authoritative transition.
LIVE_MOE_PROFILE_SELECTIONS = frozenset(
    {"MLX2_MOE_NAX_GATHER", "MLX2_MOE_RHS_PAD_POLICY"}
)


# The process environment under INHERITED_MODEL_PROFILE_PREFIXES when the
# first profile transition ran: the operator's selections (tree gate kill
# switches, env_switches.SERVING_ENV_SWITCHES, the cohort qualifier's gates).
# Anything that appears or changes there later was written by a profile or an
# in-process arm, so the next profile clears it.  Latched once per process.
_OPERATOR_SELECTORS: dict | None = None


def _operator_selectors(target) -> dict:
    global _OPERATOR_SELECTORS
    if _OPERATOR_SELECTORS is None:
        _OPERATOR_SELECTORS = {
            name: value
            for name, value in target.items()
            if name.startswith(INHERITED_MODEL_PROFILE_PREFIXES)
        }
    return _OPERATOR_SELECTORS


def clear_inherited_profile(prefixes, environ=None, *, operator=None) -> None:
    """Delete inherited model selectors before a profile is pinned.

    ``prefixes`` are the adapter's own namespaces: everything under them is
    cleared except ``PRESERVED_OPERATOR_KNOBS``.  Under the shared
    ``INHERITED_MODEL_PROFILE_PREFIXES`` the operator's values (``operator``;
    by default, for the process environment, the values present when the
    first profile applied) are kept, and put back if an earlier profile
    cleared or overwrote them; every other value there is cleared.  An
    explicit ``environ`` without ``operator`` has no operator values.
    """
    target = os.environ if environ is None else environ
    if operator is None:
        operator = _operator_selectors(target) if environ is None else {}
    own = tuple(prefixes)
    shared = INHERITED_MODEL_PROFILE_PREFIXES
    for name in tuple(target):
        if name in PRESERVED_OPERATOR_KNOBS:
            continue
        if name.startswith(own) or (
            name.startswith(shared) and operator.get(name) != target[name]
        ):
            del target[name]
    for name, value in operator.items():
        if name not in target and not name.startswith(own) and name.startswith(shared):
            target[name] = value


def assert_process_profile_applied(owner: str, *, live=()) -> None:
    """Fail when an imported tensor module latched another model profile."""
    from .runtime.models.import_env import assert_profile_applied

    assert_profile_applied(
        owner,
        live=frozenset(live) | LIVE_MOE_PROFILE_SELECTIONS,
    )
