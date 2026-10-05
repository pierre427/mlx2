"""Exact, default-off admission profile for one native Qwen3 B2 cohort."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping

from .paged_pack_price import _identity

SCHEMA = "mlx2.native-qwen3-b2-research-admission.v1"
SCHEMA_V2 = "mlx2.native-qwen3-b2-research-admission.v2"
SCHEMA_V3 = "mlx2.native-qwen3-b2-research-admission.v3"
SCHEMA_V4 = "mlx2.native-qwen3-b2-research-admission.v4"
SCHEMA_V5 = "mlx2.native-qwen3-b2-research-admission.v5"
SCHEMA_V6 = "mlx2.native-qwen3-b2-research-admission.v6"
SCHEMA_V7 = "mlx2.native-qwen3-b2-research-admission.v7"
CONTEXTS = (63, 65)
CONTEXT_BOUNDS = {"minimum": 32, "maximum": 127, "distinct": True}
SUSTAINED_CONTEXT_BOUNDS = {"minimum": 32, "maximum": 97, "distinct": True}
PACKED_FLAG = "MLX2_PAGED_B2_PACKED_PREFILL"
SUSTAINED_FLAG = "MLX2_PAGED_B2_SUSTAINED_DECODE"
VECTOR_ROPE_FLAG = "MLX2_PAGED_Q1_VECTOR_ROPE"
DIRECT_FENCE_FLAG = "MLX2_PAGED_GROUPED_DIRECT_FENCE"
DEFERRED_EVAL_FLAG = "MLX2_PAGED_B2_DEFERRED_EVAL"
DEFERRED_WRITE_EVAL_FLAG = "MLX2_PAGED_B2_DEFERRED_WRITE_EVAL"
INLINE_METADATA_FLAG = "MLX2_PAGED_Q1_INLINE_METADATA"
GROUPED_SAMPLER_FLAG = "MLX2_PAGED_GROUPED_SAMPLER"
STRIPES_FLAG = "MLX2_PAGED_Q1_SIMD_STRIPES"
STOCK_SDPA_FLAG = "MLX2_PAGED_Q1_STOCK_SDPA"
COMBINED_FLAGS = (DEFERRED_EVAL_FLAG, DEFERRED_WRITE_EVAL_FLAG,
                  INLINE_METADATA_FLAG,
                  GROUPED_SAMPLER_FLAG)
SUSTAINED_TOKENS = 32
FLAGS = (
    "MLX2_PAGED_Q1_SIMD_TILE",
    "MLX2_PAGED_GROUPED_Q1_WRITE",
    "MLX2_PAGED_PRIVATE_TAIL_REUSE",
)


def validate_combined_environment(profile: dict, environment: Mapping[str, str]) -> bool:
    """Old profiles reject undeclared optimization flags, including malformed values."""
    combined = profile["schema"] == SCHEMA_V7
    options = profile["combined_optimizations"] if combined else {}
    values = (options.get("deferred_eval", False),
              options.get("deferred_write_eval", False),
              options.get("inline_metadata", False),
              options.get("grouped_sampler", False))
    if any(environment.get(key, "0") != ("1" if enabled else "0")
           for key, enabled in zip(COMBINED_FLAGS, values)):
        raise ValueError("native B2 combined optimization flags differ from profile")
    stripes = environment.get(STRIPES_FLAG, "4")
    if stripes != str(profile["q1_simd_stripes"] if combined else 4):
        raise ValueError("native B2 Q1 stripe geometry differs from profile")
    if environment.get(STOCK_SDPA_FLAG, "0") != (
            "1" if options.get("stock_sdpa", False) else "0"):
        raise ValueError("native B2 stock SDPA diagnostic differs from profile")
    return combined


def load_b2_research_profile(
    path: str | Path, *, live_identity: dict,
    context_lengths: tuple[int, int] | None = None,
) -> dict:
    """Validate a source-bound capability permit, never a performance price."""
    data = json.loads(Path(path).read_text())
    if type(data) is not dict or data.get("schema") not in (SCHEMA, SCHEMA_V2,
                                                            SCHEMA_V3, SCHEMA_V4,
                                                            SCHEMA_V5,
                                                            SCHEMA_V6,
                                                            SCHEMA_V7):
        raise ValueError("native B2 research profile schema differs")
    bounded = data["schema"] in (SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5,
                                  SCHEMA_V6, SCHEMA_V7)
    packed = data["schema"] in (SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6,
                                 SCHEMA_V7)
    sustained = data["schema"] in (SCHEMA_V4, SCHEMA_V5, SCHEMA_V6, SCHEMA_V7)
    vector_rope = data["schema"] == SCHEMA_V5
    combined = data["schema"] == SCHEMA_V7
    direct_fence = data["schema"] in (SCHEMA_V6, SCHEMA_V7)
    options = data.get("combined_optimizations") if combined else {}
    if combined and (type(options) is not dict or
                     set(options) != {"deferred_eval", "deferred_write_eval",
                                      "inline_metadata",
                                      "grouped_sampler", "stock_sdpa"} or
                     any(type(value) is not bool for value in options.values())):
        raise ValueError("native B2 combined optimization scope differs")
    if combined and options["deferred_eval"] and options["deferred_write_eval"]:
        raise ValueError("native B2 write-only and full deferral are exclusive")
    if combined and options["deferred_write_eval"] and options["stock_sdpa"]:
        raise ValueError("native B2 write-only deferral requires custom Q1 read")
    if set(data) != {
        "schema", "profile_id", "identity",
        "context_bounds" if bounded else "ordered_context_tokens",
        "max_tokens", "sampling", "required_environment", "qualified",
        "price_usable", "serving_default", "warm_apcv2",
    } | ({"packed_prefill"} if packed else set()) | (
        {"sustained_decode"} if sustained else set()) | (
        {"vector_q1_rope"} if vector_rope else set()) | (
        {"direct_grouped_fence"} if direct_fence else set()) | (
        {"combined_optimizations", "q1_simd_stripes"} if combined else set()):
        raise ValueError("native B2 research profile fields differ")
    if (type(data["profile_id"]) is not str or not data["profile_id"] or
            _identity(data["identity"]) != _identity(live_identity) or
            type(data["max_tokens"]) is not int or
            data["max_tokens"] != (SUSTAINED_TOKENS if sustained else 2) or
            type(data["sampling"]) is not dict or
            set(data["sampling"]) != {"mode", "processors"} or
            data["sampling"]["mode"] != "greedy" or
            data["sampling"]["processors"] is not False or
            data["required_environment"] != {
                key: "1" for key in (*FLAGS, *((PACKED_FLAG,) if packed else ()),
                                     *((SUSTAINED_FLAG,) if sustained else ()),
                                     *((VECTOR_ROPE_FLAG,) if vector_rope else ()),
                                     *((DIRECT_FENCE_FLAG,) if direct_fence else ()))
            } | ({key: "1" if enabled else "0" for key, enabled in zip(
                COMBINED_FLAGS, (options["deferred_eval"],
                                 options["deferred_write_eval"],
                                 options["inline_metadata"],
                                 options["grouped_sampler"]))} |
                 {STRIPES_FLAG: str(data["q1_simd_stripes"]),
                  STOCK_SDPA_FLAG: "1" if options["stock_sdpa"] else "0"}
                 if combined else {}) or
            data["qualified"] is not False or
            data["price_usable"] is not False or
            data["serving_default"] is not False or
            data["warm_apcv2"] is not False):
        raise ValueError("native B2 research profile identity or scope differs")
    if packed and data["packed_prefill"] is not True:
        raise ValueError("native B2 packed prefill scope differs")
    if sustained and data["sustained_decode"] is not True:
        raise ValueError("native B2 sustained decode scope differs")
    if vector_rope and data["vector_q1_rope"] is not True:
        raise ValueError("native B2 vector Q1 RoPE scope differs")
    if direct_fence and data["direct_grouped_fence"] is not True:
        raise ValueError("native B2 direct grouped fence scope differs")
    if combined and (type(data["q1_simd_stripes"]) is not int or
                     data["q1_simd_stripes"] not in (4, 8, 16, 32)):
        raise ValueError("native B2 Q1 stripe geometry scope differs")
    if bounded:
        bounds = data["context_bounds"]
        expected_bounds = SUSTAINED_CONTEXT_BOUNDS if sustained else CONTEXT_BOUNDS
        if (type(bounds) is not dict or bounds != expected_bounds or
                type(bounds["minimum"]) is not int or
                type(bounds["maximum"]) is not int or
                type(bounds["distinct"]) is not bool or
                type(context_lengths) is not tuple or len(context_lengths) != 2 or
                any(type(length) is not int or not 32 <= length <= expected_bounds["maximum"]
                    for length in context_lengths) or
                context_lengths[0] == context_lengths[1]):
            raise ValueError("native B2 research profile context scope differs")
    elif (data["ordered_context_tokens"] != list(CONTEXTS) or
          context_lengths is not None and context_lengths != CONTEXTS):
        raise ValueError("native B2 research profile identity or scope differs")
    return data
