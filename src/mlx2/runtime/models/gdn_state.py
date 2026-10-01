# SPDX-License-Identifier: MIT
"""Storage dtype of the gated-delta (GDN) recurrent state.

The recurrent state is fp32 by default.  ``float16`` is a candidate storage
class (idea: DAMP, arXiv 2608.27513, which narrows the live state to INT8;
the recon-20261001 l6 drift probe measured an fp16 round-trip per step at the
perturbation floor, bf16 above it).  Under it every kernel loads the fp16
state, runs the token's update and readout in fp32 registers, and rounds the
state to fp16 (round to nearest even) after **every token** -- also inside a
multi-token prefill, verify or replay launch.  So the stored trajectory does
not depend on how tokens are grouped into launches: chunked prefill, a verify
block with rollback to ``m`` tokens, and token-by-token decode all store the
same fp16 states, and the ordinary composed path stays the bit reference for
the fused kernels.  The readout of a token is computed from the unrounded
fp32 state, exactly as the fp32 route does.

Selection is per layer (``install_state_dtype``), from the adapter's
execution policy; nothing reads a process environment variable.  An adapter
that selects it also changes its APCv2 cache layout fingerprint, so fp16 and
fp32 states never share a cache namespace (memory or disk).
"""

from __future__ import annotations

from collections import Counter

import mlx.core as mx

STATE_DTYPES = {"float32": mx.float32, "float16": mx.float16}
DEFAULT_STATE_DTYPE = "float32"
LAYOUT_TAG = "gdn-state-fp16-v1"

# Graph-build counters (no device sync): which launches ran an fp16 state.
STATS: Counter = Counter()


def validate_state_dtype(value) -> str:
    """The policy value, or a ValueError naming the accepted ones."""
    if value is None:
        return DEFAULT_STATE_DTYPE
    if not isinstance(value, str) or value not in STATE_DTYPES:
        raise ValueError(
            "gdn_state_dtype must be one of " + ", ".join(sorted(STATE_DTYPES))
        )
    return value


def layout_with_state_dtype(layout: str, value: str) -> str:
    """APCv2 cache layout fingerprint for a state dtype; fp32 is unchanged."""
    return layout if validate_state_dtype(value) == "float32" else f"{layout}:{LAYOUT_TAG}"


def is_gdn_layer(module) -> bool:
    return callable(getattr(module, "_gated_delta_update", None)) and hasattr(
        module, "head_k_dim"
    )


def install_state_dtype(model, value: str) -> dict:
    """Bind the recurrent-state dtype to every GDN layer; returns the receipt.

    ``float32`` installs nothing (the layers keep their default and their
    receipts), so a default policy leaves the model untouched.
    """
    value = validate_state_dtype(value)
    if value == "float32":
        return {"state_dtype": "float32", "layers": 0}
    dtype = STATE_DTYPES[value]
    count = 0
    for _, layer in model.named_modules():
        if is_gdn_layer(layer):
            object.__setattr__(layer, "_gdn_state_dtype", dtype)
            count += 1
    if not count:
        raise ValueError("gdn_state_dtype: model has no gated-delta layer")
    return {
        "state_dtype": value,
        "layers": count,
        "rounding": "per-token round-to-nearest-even, fp32 compute",
        "qualification": "unqualified",
        "counters": STATS,
    }


def check_state(state, expected) -> None:
    """Fail closed when a cache carries a state of the other storage class.

    ``expected`` is the layer's selected dtype, ``None`` for a layer that
    never selected one (the default class: an fp16 state is refused there).
    """
    if state is None:
        return
    if expected is None:
        if state.dtype == mx.float16:
            raise ValueError(
                "GDN recurrent state is float16 but this layer did not select "
                "the fp16 storage class; states of different classes never mix"
            )
        return
    if state.dtype != expected:
        raise ValueError(
            f"GDN recurrent state is {state.dtype}, the layer stores {expected}; "
            "states of different storage classes never mix"
        )


def nonfinite_state_layers(caches) -> list:
    """Indices of cache entries whose recurrent state holds inf/NaN.

    A diagnostic for an already-failed step (it syncs): the per-step
    non-finite log-probability check drops the lane, and this names the GDN
    layers that overflowed, so the failure receipt says why.
    """
    bad = []
    for index, cache in enumerate(caches or ()):
        try:
            state = cache[1]
        except Exception:  # noqa: BLE001 - not an ArraysCache
            continue
        if isinstance(state, mx.array) and state.ndim == 4:
            if not bool(mx.all(mx.isfinite(state)).item()):
                bad.append(index)
    return bad


def derive_source(source: str, replacements, *, what: str) -> str:
    """Apply checked textual replacements; each ``old`` must occur ``count`` times."""
    for old, new, count in replacements:
        found = source.count(old)
        if found != count:
            raise RuntimeError(
                f"{what}: fp16-state derivation drifted ({found} != {count} for {old[:40]!r})"
            )
        source = source.replace(old, new)
    return source
