#!/usr/bin/env python3
"""Generate the 'everything we built' comparison ladder (Pierre 2026-11-09).

All non-exact arms use the 90 GB total; tier arms carry the selected default kernel bundle
explicitly (the default guard skips tier profiles, so earlier tier speeds ran reference kernels).
"""
import json
from pathlib import Path

M = "~/mlx-models"
U = f"{M}/.ds41-routing-capture-20261108/usage-train-split.json"
Q1 = f"{M}/DeepSeek-V4.1-Flash-1017-Source-Expert-Q1-g64"
Q2 = {"g64": f"{M}/DeepSeek-V4.1-Flash-1017-Source-Expert-Q2-g64", "g32h": f"{M}/DeepSeek-V4.1-Flash-1017-Source-Expert-Q2-g32h"}
DEFAULTS = {"deepseek_attention_epilogue": "fused", "deepseek_engram_cache": "bf16_lut", "deepseek_index_policy": "finite_all_retained",
            "deepseek_pointwise": "fused", "deepseek_qat": "fused_fp32", "deepseek_selected_kv_policy": "decode_reuse",
            "deepseek_source_attention_math": "all_retained_broadcast", "deepseek_source_bf16_loads": "direct_f32",
            "deepseek_source_completion": "bounded_2", "deepseek_source_dense_qat": "fused_fp32",
            "deepseek_source_expert_math": "routed_fused_fp32", "deepseek_source_fp8_pp": "native_bf16_qb_wob_v1",
            "deepseek_source_hc_coefficients": "fused_fp32", "deepseek_source_hc_mega": "mega_v1",
            "deepseek_source_prefill_batch": "gate_up", "deepseek_source_qakv": "qat_reuse", "deepseek_source_woa_storage": "mxfp8_gemv_v1"}
SENSITIVE = [(10, 14), (20, 24), (30, 34)]
# Re-ablation 2026-11-09: L10-12 tolerate Q2 again; L13, L20 and L30 do not.
REFINED = [(13, 14), (20, 24), (30, 34)]
B90 = 90_000_000_000


def orig(budget=B90):
    return {"deepseek_source_experts": {"mode": "original_lfu", "budget_bytes": budget}}


def tiers(fmt, layers, rank_overrides, q2_total=40_000_000_000, q1_total=1_000_000_000):
    lo, hi = layers
    n = hi - lo + 1
    q2_total = min(q2_total, n * 2_000_000_000)   # a short span cannot hold 40 GB of one layer's 384 experts
    ov = [{"layers": [0, 39], "q2_bytes": 0, "q1_bytes": 0},
          {"layers": [lo, hi], "q2_bytes": q2_total // n, "q1_bytes": q1_total // n}] + rank_overrides
    return {"deepseek_source_experts": {"mode": "original_lfu", "budget_bytes": B90, "q1_root": Q1,
            "tiers": {"usage_map": U, "q2_root": Q2[fmt], "q2_format": fmt, "q2_bytes": (q2_total // n) * n,
                      "q1_bytes": (q1_total // n) * n, "max_weight": 1.0, "layers": [lo, hi], "overrides": ov}}, **DEFAULTS}


def recovered(fmt, spans=SENSITIVE):
    ov = [{"layers": [4, 39], "rank_tiers": {"q4": 0, "q2": 6, "rest": "q1", "strict": True}}]
    ov += [{"layers": [a, b], "rank_tiers": {"q4": 6, "q2": 0, "rest": "q1"}} for a, b in spans]
    return tiers(fmt, (4, 39), ov)


def drop(mode, spans, **extra):
    return {"deepseek_source_route_drop": mode,
            "deepseek_source_route_drop_spans": [{"layers": list(l), "count": c} for l, c in spans], **extra}


def merge(*parts):
    out = {}
    for p in parts:
        for k, v in p.items():
            out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


arms = [
    ("exact_75", orig(75_000_000_000)),
    ("exact_90", orig()),
    ("drop2_l13_26_skip", merge(orig(), drop("drop_min_v1", [((13, 26), 2)]))),
    ("drop_miss_all_c2", merge(orig(), drop("drop_miss_v1", [((0, 39), 2)]))),
    ("route_bias_guarded", merge(orig(), {"deepseek_source_route_bias": "resident_guarded_v1"})),
    ("recovered_g64", recovered("g64")),
    ("recovered_g32h", recovered("g32h")),
    ("refined_g64", recovered("g64", REFINED)),
    ("refined_g32h", recovered("g32h", REFINED)),
    ("keep1_q4_l15_19", merge(orig(), drop("drop_min_v1", [((15, 19), 5)]))),
    ("keep1_g32h_l15_19", merge(tiers("g32h", (15, 19), [{"layers": [15, 19], "rank_tiers": {"q4": 0, "q2": 1, "rest": "q1", "strict": True}}]),
                                drop("drop_min_v1", [((15, 19), 5)]))),
    ("keep2_q4_l13_26", merge(orig(), drop("drop_min_v1", [((13, 26), 4)]))),
    ("keep2_q4_g32h_l13_26", merge(tiers("g32h", (13, 26), [{"layers": [13, 26], "rank_tiers": {"q4": 1, "q2": 1, "rest": "q1", "strict": True}}]),
                                   drop("drop_min_v1", [((13, 26), 4)]))),
    ("recovered_g32h_drop_miss", merge(recovered("g32h"), drop("drop_miss_v1", [((0, 39), 2)]))),
]
spec = {"base": {}, "arms": [{"name": n, "policy": p} for n, p in arms], "max_tokens": 192}
Path(__file__).with_name("compare_all.json").write_text(json.dumps(spec, indent=1))
print(len(arms), "arms")
