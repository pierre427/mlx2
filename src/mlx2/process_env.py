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
#
# Hazard (apc-warm-hit-hang-20261021): MLX's maintainers mark fast synch
# unsupported (ml-explore/mlx#3142, won't-fix; #3830).  Inside one
# evaluation, an array produced on one GPU stream and consumed on another is
# bridged by a Fence; in fast mode that is a ``fence_wait`` kernel spinning
# on one command queue for a ``fence_update`` on another, and the two command
# buffers are not guaranteed to overlap: the GPU can wedge at 100% with no
# error.  mlx2 therefore keeps the generation thread single-stream
# (``runtime.generation_stream.bind_generation_thread_stream``) and hands lanes
# evaluated state (``models.cache.materialize_restored_state``), and the
# eval watchdog (``runtime.eval_watchdog``) reports a wedge instead of
# serving silently.  Any new cross-stream lazy handoff reopens the hazard.
PROCESS_RUNTIME_DEFAULTS = {"MLX_METAL_FAST_SYNCH": "1"}


# Metal command-buffer commit cadence, latched by MLX when Metal starts. It
# never changes a computed bit: it decides only where MLX ends one command
# buffer and starts the next. MLX commits after max_ops_per_buffer kernels or
# once the *input* bytes bound in the buffer pass max_mb_per_buffer (M5 Max
# defaults 50 ops / 50 MB, mlx/backend/metal/device.cpp). An input is billed
# at its whole buffer size, so a decode step of a large model whose kernels
# bind whole packed weight banks (GLM-5.3: a 520 MB expert bank per routed
# launch) commits after nearly every weight kernel, and each commit leaves the
# GPU waiting on the next buffer's scheduling. MLX2_COMMAND_BUFFERS selects:
#
#   "mlx_default" (default)  MLX's own values.
#   "ops100_v1"              commit every 100 kernels; no byte trigger
#                            (MLX_MAX_OPS_PER_BUFFER=100,
#                            MLX_MAX_MB_PER_BUFFER=1000000).
#
# Evidence for ops100_v1 (GLM-5.3-Flash-REAP50-Q3_K_M.gguf@5a865918, mlx2
# 1892dab2a, process-level ABBA of recorded 2K/32K decode replays,
# qualification/runs/glm53-ordinary-roofline-20261030/scout-envsweep-*):
# pipelined decode 24.7-25.3 -> 20.2-20.5 ms per step (-19%), every greedy
# token identical. No byte trigger with the default 50 ops: -14%; neither
# trigger (one buffer per step): -2% (the GPU then waits for the whole step to
# be encoded). Opt-in, unqualified; explicit MLX_MAX_* values win.
COMMAND_BUFFER_POLICY_ENV = "MLX2_COMMAND_BUFFERS"
COMMAND_BUFFER_POLICIES = {
    "mlx_default": {},
    "ops100_v1": {"MLX_MAX_OPS_PER_BUFFER": "100", "MLX_MAX_MB_PER_BUFFER": "1000000"},
}


# The command-buffer values present when the package defaults were first
# applied (before any mlx2 entrypoint can start Metal). A value written after
# that may never have reached MLX, so status reports this snapshot.
_COMMAND_BUFFERS_AT_IMPORT: dict | None = None


def command_buffers_at_import() -> dict | None:
    return None if _COMMAND_BUFFERS_AT_IMPORT is None else dict(_COMMAND_BUFFERS_AT_IMPORT)


# A byte cap at or above this many MB no longer bounds a prefill chunk's
# command buffers (MLX's own M5 Max default is 50).
RAISED_BYTE_CAP_MB = 512


def raised_command_buffer_byte_cap(environ=None) -> bool:
    """True when the byte cap MLX latched (the import snapshot) or the one now
    in the environment is raised to RAISED_BYTE_CAP_MB or more, whichever way
    it was set (MLX2_COMMAND_BUFFERS or an explicit MLX_MAX_MB_PER_BUFFER)."""
    target = os.environ if environ is None else environ
    values = [target.get("MLX_MAX_MB_PER_BUFFER")]
    if environ is None and _COMMAND_BUFFERS_AT_IMPORT is not None:
        values.append(_COMMAND_BUFFERS_AT_IMPORT.get("MLX_MAX_MB_PER_BUFFER"))
    for value in values:
        try:
            if value is not None and int(value) >= RAISED_BYTE_CAP_MB:
                return True
        except ValueError:
            return True  # unparseable: assume the worst
    return False


def command_buffer_policy(environ=None) -> str:
    target = os.environ if environ is None else environ
    value = target.get(COMMAND_BUFFER_POLICY_ENV, "mlx_default")
    if value not in COMMAND_BUFFER_POLICIES:
        raise ValueError(f"unsupported {COMMAND_BUFFER_POLICY_ENV}={value!r}; "
                         f"expected one of {sorted(COMMAND_BUFFER_POLICIES)}")
    return value


def apply_process_runtime_defaults(environ=None) -> dict:
    """Set each runtime default that is not already present; return what was set."""
    target = os.environ if environ is None else environ
    applied = {}
    global _COMMAND_BUFFERS_AT_IMPORT
    defaults = {**PROCESS_RUNTIME_DEFAULTS, **COMMAND_BUFFER_POLICIES[command_buffer_policy(target)]}
    for name, value in defaults.items():
        if name not in target:
            target[name] = value
            applied[name] = value
    if environ is None and _COMMAND_BUFFERS_AT_IMPORT is None:
        _COMMAND_BUFFERS_AT_IMPORT = {
            "policy": command_buffer_policy(target),
            **{name: target.get(name) for name in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER")},
        }
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
