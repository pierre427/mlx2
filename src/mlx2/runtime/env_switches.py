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
    "MLX2_COMMAND_BUFFERS": "Metal command-buffer commit cadence (process_env; no computed bit changes)",
    "MLX2_NVFP4_DENSE": "Super VL NVFP4 export dense projections: vendor fp8 (default) or exported bf16 (APCv2-bound)",
    "MLX2_NVFP4_EXPERT_SCALE": "Super VL NVFP4 export expert weight_scale_2: fold (default) or apply (APCv2-bound)",
    "MLX2_SCALED_EPILOGUE": "scaled FP8/NVFP4 projection epilogue: one compiled kernel (default) or eager (same fp32 multiply)",
    "MLX2_NEMOTRON_ROUTED_DECODE": "Nemotron-H one-token NVFP4 routed experts in three stock-exact launches (on by default)",
    "MLX2_FP_ROW_EXACT_KERNEL": "exact FP8/NVFP4 verify rows: single-pass fused kernel (default), fused_chunked (opt-in: 9..16 rows as <=8-row fused launches) or per-row loop (bit-identical, self-checked)",
    # MLX's own commit caps, latched at Metal start (set by MLX2_COMMAND_BUFFERS
    # or by the operator): schedule and transient memory, never values.
    "MLX_MAX_OPS_PER_BUFFER": "MLX command-buffer kernel cap",
    "MLX_MAX_MB_PER_BUFFER": "MLX command-buffer input-size cap",
}


def serving_env_switches(environ=None) -> dict:
    """``{name: value}`` for every allow-listed switch explicitly set."""
    env = os.environ if environ is None else environ
    return {name: str(env[name]) for name in sorted(SERVING_ENV_SWITCHES) if name in env}
