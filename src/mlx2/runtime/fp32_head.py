"""Opt-in fp32 logits from a quantized vocabulary head.

The quantized ``lm_head`` of the Qwen3.8 text adapters takes a bf16 hidden
state and returns bf16 logits.  Its Metal kernels already accumulate in fp32;
the only precision lost is the final store, which rounds each logit to bf16.
At logit magnitudes of 16 to 32 the bf16 spacing is 0.125, so near-ties round
together and greedy argmax can flip.

MLX's ``quantized_matmul`` computes in ``promote(x.dtype, scales.dtype)``.
Widening the head's bf16 scales and biases to fp32 once (exact) therefore
makes the same quantized weights run with an fp32 input (the bf16 hidden state
widens exactly) and store their fp32 sums instead of rounding them.  Nothing
else changes: parameter names, the weights, and every caller of the head --
ordinary sampling, MTP drafting and verification, logprobs -- which now
receive fp32 logits.

Opt-in (``fp32_head_logits`` in the 27B and Flash-Next execution policies),
not a default: on the 27B it takes greedy top-1 agreement with an fp32-trunk
reference from 98.80% to 99.50% and removes every head-rounding flip, but the
native-MTP server decodes 1.2% slower at B1 and about 9% slower at B4
(qualification/runs/sp-fp32-head-20260925/serving-abba-27b.json).

Design input (idea only, no code): Inco Splash, Apache-2.0, f786bed, PR #141
("Keep all logits in fp32").  See ``provenance/fp32-head-logits.json``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def enable_fp32_head_logits(language_model) -> dict:
    """Make ``language_model.lm_head`` return fp32 logits.

    Fails closed (``ValueError``) on a tied, unquantized or biased head: the
    request for fp32 logits is then unmet and the caller must not serve as if
    it were.  Idempotent.  Returns a receipt for adapter diagnostics.
    """
    args = getattr(language_model, "args", None)
    if getattr(args, "tie_word_embeddings", False):
        raise ValueError("fp32_head_logits requires an untied lm_head")
    head = getattr(language_model, "lm_head", None)
    if not isinstance(head, nn.QuantizedLinear):
        raise ValueError("fp32_head_logits requires a quantized lm_head")
    if "bias" in head:
        raise ValueError("fp32_head_logits supports a bias-free lm_head only")
    before = int(head["scales"].nbytes) + int(
        0 if head.get("biases") is None else head["biases"].nbytes
    )
    head.scales = head["scales"].astype(mx.float32)
    if head.get("biases") is not None:
        head.biases = head["biases"].astype(mx.float32)
    mx.eval(head.parameters())
    after = int(head["scales"].nbytes) + int(
        0 if head.get("biases") is None else head["biases"].nbytes
    )
    return {
        "enabled": True,
        "bits": int(head.bits),
        "group_size": int(head.group_size),
        "mode": str(head.mode),
        "extra_resident_bytes": after - before,
    }

