# SPDX-License-Identifier: MIT
"""Targeted float32 -> compute-dtype normalization of RMSNorm gammas.

Some community conversions (the Qwen3.8-27B CRACK 4/8-bit and bf16 artifacts)
store the RMSNorm gammas as float32 where the official MLX conversions store
bf16. ``mx.fast.rms_norm(bf16 x, f32 w)`` returns float32, so a single float32
residual-stream norm (``input_layernorm``, ``post_attention_layernorm``, the
final ``norm``) or attention ``q_norm``/``k_norm`` promotes the whole residual
stream to float32 from layer 0: the KV cache then allocates float32, quantized
matmuls take float32 kernels, and every bf16 weight is converted inside each
matmul (measured 1.66x slower on the quantized CRACK variants, ~4x on bf16).

The cast is deliberately targeted, not a blanket ``astype``:

* Only tensors whose names end in a caller-supplied norm suffix are touched.
  Gated-delta ``A_log``, ``dt_bias`` and ``linear_attn.norm`` stay float32:
  the decay gate is computed in float32 anyway, the recurrent state is
  float32 by design, and ``Qwen3NextRMSNormGated`` casts its result back to
  the input dtype, so none of them promotes the stream (checked on CPU in
  ``tests/test_norm_dtype_normalization.py``).
* Other families carry intentional float32 tensors (Xing4.0 / DeepSeek-V4 /
  glm52 routing biases and hyper-connection scales); they never call this.
* The cast runs after the +1 norm-convention fold, so the fold is done in
  float32 and rounded once.
"""

from __future__ import annotations

import logging
from typing import Iterable, MutableMapping, Optional

logger = logging.getLogger(__name__)

_DTYPE_NAMES = ("bfloat16", "float16", "float32")


def resolve_compute_dtype(configured: Optional[str], weights=None):
    """The model's compute dtype: config ``dtype``, else the embedding's.

    Returns an ``mx.Dtype`` or ``None`` when neither source is decisive.
    """
    import mlx.core as mx

    if isinstance(configured, str) and configured in _DTYPE_NAMES:
        return getattr(mx, configured)
    if weights:
        for key, value in weights.items():
            if key.endswith("embed_tokens.scales") or key.endswith(
                "embed_tokens.weight"
            ):
                if mx.issubdtype(value.dtype, mx.floating):
                    return value.dtype
    return None


def normalize_norm_dtypes(
    weights: MutableMapping, suffixes: Iterable[str], target
) -> Optional[dict]:
    """Cast float32 tensors ending in ``suffixes`` to ``target`` in place.

    Returns the load receipt ``{"cast": N, "from": "float32", "to": ...}``
    (``cast`` may be 0), or ``None`` when ``target`` is unknown or is itself
    float32 (a genuinely float32 model is left alone).
    """
    import mlx.core as mx

    if target is None or target == mx.float32:
        return None
    suffixes = tuple(suffixes)
    cast = 0
    for key in list(weights):
        value = weights[key]
        if value.dtype == mx.float32 and key.endswith(suffixes):
            weights[key] = value.astype(target)
            cast += 1
    receipt = {"cast": cast, "from": "float32", "to": _name(target)}
    if cast:
        logger.info("dtype_normalized: %s", receipt)
    return receipt


def check_compute_dtype(text_model, expected) -> dict:
    """Load-time check: a tiny trunk forward must stay in ``expected``.

    Runs two tokens through the trunk with a fresh cache (two, not one, so
    decode-only fused paths and their call counters are not touched) and
    compares the hidden and full-attention KV dtypes with the compute dtype.
    A mismatch is logged as a warning and recorded, so a future promotion is
    caught at load rather than surfacing as a throughput mystery.
    """
    import mlx.core as mx

    if expected is None:
        return {"ok": None, "expected": None, "reason": "compute dtype unknown"}
    cache = text_model.make_cache()
    hidden = text_model.model(mx.array([[0, 0]]), cache=cache)
    kv = [
        c.keys.dtype
        for c in cache
        if getattr(c, "keys", None) is not None and hasattr(c.keys, "dtype")
    ]
    mx.eval(hidden)
    observed = {
        "hidden": _name(hidden.dtype),
        "kv": sorted({_name(d) for d in kv}),
    }
    ok = hidden.dtype == expected and all(d == expected for d in kv)
    result = {"ok": ok, "expected": _name(expected), "observed": observed}
    if not ok:
        logger.warning(
            "compute dtype promoted at load: expected %s, observed %s; "
            "a float32 norm or bias is widening the residual stream",
            _name(expected),
            observed,
        )
    return result


def _name(dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]
