"""Behaviour-changing ``MLX2_*`` environment switches a serving route records.

Adapter profiles pin and record their own ``MLX_QWEN*``/``MLX_LM_*``/
``MLX_GDN_*`` environment, but leave every ``MLX2_*`` switch alone, so a
process started with one of these set served a different law or schedule
under settings identical to a default receipt.  Serving records each one that
is explicitly present as ``settings["process_env"]``; absent switches are not
recorded, so default settings are unchanged.  The switches that change the
K/V or recurrent state reaching APCv2 are also bound into its key
(``apc_numerics.CANDIDATE_ENV``).
"""

from __future__ import annotations

import os

# name -> what it changes.  Allow-listed: unrelated MLX2_* variables (worker
# counts, paths, test aids) are not route identity.
SERVING_ENV_SWITCHES = {
    "MLX2_MIXED_PREFILL_DECODE": "fused prompt-slice and decode forward; not bit-identical (APCv2-bound)",
    "MLX2_SELF_MTP_HOST_ACCEPT": "candidate host-side B1 self-MTP acceptance",
    "MLX2_DECODE_MASK": "single-token decode SDPA mask",
    "MLX2_STEP_VALIDITY": "where the sampled-row corruption check runs",
    "MLX2_PREFILL_CLEAR_CACHE": "allocator clears after prefill chunks",
    "MLX2_ALLOCATOR_RECLAIM_STEP_INTERVAL": "decode rounds between allocator clears",
    "MLX2_GREEDY_BATCH_SAMPLER": "shared or per-lane greedy sampling",
    "MLX2_MTP_DEPTH_CAP": "largest accepted self-MTP num_draft",
    "MLX2_XING_MHC_KERNEL": "Xing fused mHC kernels or the compiled path (APCv2-bound when off)",
    "MLX2_NORTH_NORM": "North legacy LayerNorm A/B arm (APCv2-bound)",
    "MLX2_LFM25_FUSED_SHORTCONV": "LFM2.5-VL fused one-token ShortConv candidate",
    "MLX2_FUSED_SDPA_MIN_L": "fused d256 prefill SDPA crossover (APCv2 exemption)",
    "MLX2_DECODE_FIRST": "decode-first publication kill switch",
}


def serving_env_switches(environ=None) -> dict:
    """``{name: value}`` for every allow-listed switch explicitly set."""
    env = os.environ if environ is None else environ
    return {name: str(env[name]) for name in sorted(SERVING_ENV_SWITCHES) if name in env}
