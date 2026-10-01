# SPDX-License-Identifier: MIT
"""Process-level numerical laws that the APCv2 key binds.

Idea from arXiv 2609.38706 ("KV provenance in shared caches"): a shared KV
tier keyed on tokens alone hands out state that another configuration
computed. mlx2's ``APCKey`` already binds the artifact, the runtime
source/MLX revision, the tenant, the media/LoRA/hyper-directory scope, and
three numerical-law wrappers (int8 prefill, lane matmul, prefill execution).
This wrapper adds the remaining process-level switches that select a
candidate or reduced-precision law for the K/V and recurrent state that
reaches APCv2. The persistent tier needs them most: a restart with a flipped
switch keeps the same source revision, so without this binding it would adopt
blocks written under the other law.

A switch enters the identity only when it differs from its default, so
default namespaces, and the blocks persisted under them, keep their identity.
Some levers are deliberately not bound: qualified exact-family levers, and
geometry that varies at run time (prefill slicing, batch width). APCv2 reuse
already moves chunk boundaries, so their bits stay within the tolerance APCv2
promises. tests/test_apc_key_differential.py lists both groups.
"""

from __future__ import annotations

import hashlib
import json
import os

from ..process_env import PROCESS_NUMERICS

TAG = "execution-numerics-v1"

# Environment switches that select a KV-producing law outside the qualified
# exact default.  "flag": on unless empty/0/false/off/no; "int": bound when
# nonzero.  Each entry says why.
CANDIDATE_ENV = {
    # Reduced-precision fp32 matmuls (process_env pins "0"; explicit wins).
    **{name: "flag" for name in PROCESS_NUMERICS},
    # MLX gated_delta_update for 17..256-row prefill chunks; parity unestablished.
    "MLX_GDN_CORE": "flag",
    # Opt-in A/B kernels with no mlx2 full-model evidence (flash_next_policy).
    "MLX_QWEN4_MOE_ROUTER_KERNEL": "flag",
    "MLX_QWEN4_QSA_NAX_DECODE": "flag",
    # Tiled quantized-SDPA scores; checked on the pinned M5 build only.
    "MLX2_QSDPA_SCORES_BUDGET_BYTES": "int",
}
_FALSE = {"", "0", "false", "off", "no"}


def _bound_value(kind: str, raw: str):
    value = raw.strip()
    if kind == "flag":
        return None if value.lower() in _FALSE else "1"
    try:
        return None if int(value) == 0 else str(int(value))
    except ValueError:
        return value


def execution_numerics_identity(environ=None, *, sp_qmm=False, verify_bitexact=False):
    """The non-default process numerics, or None when everything is default.

    ``sp_qmm``: the small-M quantized matmul (not bitwise identical, M=2..16
    rows, which includes short prefill tails).  ``verify_bitexact``: the
    process-global mode that gives every matmul of <= max_m rows the
    single-row arithmetic (prefill tails included).
    """
    env = os.environ if environ is None else environ
    bound = {}
    for name, kind in CANDIDATE_ENV.items():
        raw = env.get(name)
        if raw is not None:
            value = _bound_value(kind, raw)
            if value is not None:
                bound[name] = value
    if sp_qmm:
        bound["sp_qmm"] = True
    if verify_bitexact:
        bound["verify_bitexact"] = True
    return {"version": 1, **bound} if bound else None


def apc_execution_fingerprint(base, identity):
    """APCv2 semantic namespace under ``identity``; None is the identity."""
    if identity is None:
        return base
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return (base, TAG, hashlib.sha256(payload).hexdigest())
