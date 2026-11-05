# SPDX-License-Identifier: Apache-2.0
"""Explicit model-owned output-only recurrent candidate; serving opt-in pending.

Inputs are already normalized/activated Qwen3.8 GDN operands. The caller owns
convolution, attention KV and publication of the complete state-plane pair.
This module never automatically selects the candidate for a serving request.
"""

from __future__ import annotations

from ..experimental.flashinfer_gdn_replay import OutputOnlyReplay, ReplayBinding


def output_only_replay_candidate(
    state,
    *,
    checkpoint,
    source_revision,
    enabled=False,
    layout="value_key",
    max_steps=17,
    max_tape_bytes=64 << 20,
):
    """Create an opt-in CPU executor from canonical [L,S,HV,DV,DK] state.

    This separate numerical law does not inherit the qualification of the
    ordinary Qwen3.8 route or existing compact replay. Native launch is offered
    separately through build_native_output_only_candidate and stages output
    state for the adapter to validate; it does not mutate the live cache.
    """
    return OutputOnlyReplay(
        state,
        ReplayBinding(
            checkpoint, source_revision, "qwen38-gdn-output-only-candidate-v1"
        ),
        enabled=enabled,
        layout=layout,
        max_steps=max_steps,
        max_tape_bytes=max_tape_bytes,
    )


def build_native_output_only_candidate(*args, **kwargs):
    """Lazy native staging API; no MLX import until explicitly requested."""
    from ..experimental.flashinfer_gdn_replay import build_native_candidate

    return build_native_candidate(*args, **kwargs)
