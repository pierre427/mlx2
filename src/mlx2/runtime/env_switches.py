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
    "MLX2_EXTERNAL_VERIFY_PREBUILD": "B1 greedy external chain: build and queue the verify while the drafter runs (opt-in; adopted only when the host proposal equals it; no computed bit changes)",
    "MLX2_EXTERNAL_DRAFT_STREAM": "under MLX2_EXTERNAL_VERIFY_PREBUILD=1: run the LiLiCorr drafter graph on the default stream (shared, default) or its own GPU stream (draft, opt-in: deadlocked served without phase syncs); scheduling only, no computed bit changes",
    "MLX2_NEMOTRON_EXACT_ATTENTION_ROWS": "Nemotron-H exact verify attention blocks: unset = adapter default (Super VL on, others off), off (whole block per row) or on (norm and q/k/v/o on the strip through exact row ops; cache append and SDPA per row; bit-identical)",
    "MLX2_SUPER_TREE_COST_HANDOFF": "Super VL (not Bolt) B1 tree route: guarded (default: reversible park below 80% of ordinary over 8 rounds, re-probe; for long-context low acceptance) or off (tree kept unconditionally)",
    "MLX2_GEMV_ROWS_KERNEL": "exact dense verify-row matvecs (MoE latent projections, router): loop (default, one broadcast matvec per row) or fused (opt-in: one weight read per <=8 rows, bit-identical, self-checked)",
    "MLX2_FP_ROW_EXACT_GENERAL": "exact FP verify rows on layouts that miss qmv_fast (e.g. K 5376): loop (default) or fused (opt-in single-pass, <=6 rows per launch, bit-identical, self-checked)",
    "MLX2_DRAFTER_ROWS_QMV": "external-drafter 4-bit (affine q4 / NVFP4) linears: 2..16-row one-weight-read kernel (on) or stock quantized_matmul (off, default); draft numerics only, target-verified",
    "MLX2_FP_ROW_EXACT_KERNEL": "exact FP8/NVFP4 verify rows: single-pass fused kernel (default), fused_chunked (opt-in: 9..16 rows as <=8-row fused launches) or per-row loop (bit-identical, self-checked)",
    "MLX2_NEMOTRON_SSD_MMA": "Nemotron-H Mamba-2 prefill scan as 8x8 simdgroup products, 32-row chunks (opt-in; not bit-identical, APCv2-bound)",
    "MLX2_NEMOTRON_ATTN_STAGED": "Nemotron-H GQA-16 causal prefill attention with staged K/V tiles (opt-in, measured slower than MLX SDPA; APCv2-bound)",
    # MLX's own commit caps, latched at Metal start (set by MLX2_COMMAND_BUFFERS
    # or by the operator): schedule and transient memory, never values.
    "MLX_MAX_OPS_PER_BUFFER": "MLX command-buffer kernel cap",
    "MLX_MAX_MB_PER_BUFFER": "MLX command-buffer input-size cap",
}


def serving_env_switches(environ=None) -> dict:
    """``{name: value}`` for every allow-listed switch explicitly set."""
    env = os.environ if environ is None else environ
    return {name: str(env[name]) for name in sorted(SERVING_ENV_SWITCHES) if name in env}
