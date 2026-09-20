"""The self-MTP draft-depth ceiling, and the opt-in that raises it.

Four adapters -- Qwen3.8-27B, Qwen3.6-35B-A3B, Flash-Next and Xing4.0 --
shipped an identical ``1 <= num_draft <= 3`` guard.  Each was introduced in the
commit that created its file, with no comment and no cited measurement, and the
same repository admits ``1-7`` for the North EAGLE chain and ``1-15`` for the
Laguna DFlash block (docs/SERVING.md), so three was never a house rule.

Nothing in the artifacts bounds the depth at three either.  Qwen3.8-27B's
``config.json`` declares ``mtp_num_hidden_layers = 1`` and no horizon of any
kind; its safetensors ``__metadata__`` is the bare ``{"format": "pt"}``; and the
model card states only that the head was "trained with multiple steps", plural
and unquantified.  The weights are one decoder layer plus
``fc: [2H, H]`` over ``[norm(embed(t_{p+1})); norm(h_p)]``, reusing the target's
own ``lm_head`` -- a recurrent next-token head, not a bank of per-offset heads.
Depth is obtained by feeding the head its own post-norm hidden back in
(``runtime/hybrid_speculative.py``, ``for depth in range(1, max_k)``), which is
the recurrence the vendor says it trained, so a fourth draft token is the same
operator applied once more rather than a new one.

That makes the guard a tested-range guard, not a correctness bound.  The
distinction matters because verification is exact at every depth: a draft the
target rejects costs time and changes nothing about the emitted tokens.  A
lifted cap can therefore lose throughput but cannot lose correctness.

The default is unchanged at three.  ``MLX2_MTP_DEPTH_CAP`` raises it, and is
meant for qualification sweeps: a deployment that has not set it validates
exactly as it did before, down to the error message.
"""

from __future__ import annotations

import os

#: What every adapter accepted before the opt-in existed, and still accepts
#: when ``MLX2_MTP_DEPTH_CAP`` is unset.
DEFAULT_SELF_MTP_DEPTH_CAP = 3

#: A hard ceiling on the opt-in itself.  Each draft token adds a row to the
#: target verification forward (``num_draft + 1`` rows per lane, see
#: ``runtime/int8_prefill.py``) and a KV entry to the MTP cache, so an
#: unbounded environment variable would be an unbounded admission charge.
MAX_SELF_MTP_DEPTH_CAP = 16

_ENV = "MLX2_MTP_DEPTH_CAP"


def self_mtp_depth_cap(environ=None) -> int:
    """Return the largest ``num_draft`` a self-MTP adapter will accept."""
    raw = (os.environ if environ is None else environ).get(_ENV)
    if raw is None or raw == "":
        return DEFAULT_SELF_MTP_DEPTH_CAP
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"{_ENV} must be an integer between 1 and {MAX_SELF_MTP_DEPTH_CAP}; got {raw!r}"
        ) from None
    if not 1 <= value <= MAX_SELF_MTP_DEPTH_CAP:
        raise ValueError(
            f"{_ENV} must be an integer between 1 and {MAX_SELF_MTP_DEPTH_CAP}; got {raw!r}"
        )
    return value


def validate_self_mtp_num_draft(value, environ=None) -> int:
    """Validate one ``num_draft`` against the active cap and return it.

    With no opt-in the message is the one adapters raised before, so callers
    and tests that match on it are unaffected.
    """
    cap = self_mtp_depth_cap(environ)
    if type(value) is not int or not 1 <= value <= cap:
        if cap == DEFAULT_SELF_MTP_DEPTH_CAP:
            raise ValueError("num_draft must be 1, 2, or 3")
        raise ValueError(f"num_draft must be an integer between 1 and {cap} ({_ENV})")
    return value
