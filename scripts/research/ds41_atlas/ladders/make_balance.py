#!/usr/bin/env python3
"""Balance ladder (Pierre 2026-11-09: route bias + fused Q2, keep as much in memory as possible).

Lessons folded in: strong route bias was fastest with clean quality (plain 0.7: 6.2 tok/s);
the earlier 40 GB Q2 pools starved the original cache (49 GB left) and strict rank caps re-read Q2
for originals already in RAM, so reads did not fall. Here the Q2 pool is small, only on layers
the re-ablation found tolerant, and non-strict (a resident original always wins).
"""
import json
from pathlib import Path

from make_compare_all import DEFAULTS, Q1, Q2, U, B90, merge, drop

TOLERANT = [(4, 12), (15, 19), (25, 29), (35, 39)]
REFINED_Q4 = [(13, 14), (20, 24), (30, 34)]
N_TOL = sum(b - a + 1 for a, b in TOLERANT)


def bias(mode="resident_v1", lam=0.7):
    return {"deepseek_source_route_bias": mode, "deepseek_source_route_bias_lambda": lam}


def orig():
    return {"deepseek_source_experts": {"mode": "original_lfu", "budget_bytes": B90}}


def tiers(q2_total, q4_ranks=2, fmt="g32h", mixed=True):
    per2, per1 = q2_total // N_TOL, 250_000_000
    ov = [{"layers": [0, 39], "q2_bytes": 0, "q1_bytes": 0}]
    ov += [{"layers": [a, b], "q2_bytes": per2, "q1_bytes": per1} for a, b in TOLERANT]
    ov += [{"layers": [a, b], "rank_tiers": {"q4": q4_ranks, "q2": 6 - q4_ranks, "rest": "q1"}} for a, b in TOLERANT]
    ov += [{"layers": [a, b], "rank_tiers": {"q4": 6, "q2": 0, "rest": "q1"}} for a, b in REFINED_Q4]
    p = {"deepseek_source_experts": {"mode": "original_lfu", "budget_bytes": B90, "q1_root": Q1,
         "tiers": {"usage_map": U, "q2_root": Q2[fmt], "q2_format": fmt, "q2_bytes": per2 * N_TOL,
                   "q1_bytes": per1 * N_TOL, "max_weight": 1.0, "layers": [4, 39], "overrides": ov}}, **DEFAULTS}
    if mixed:
        p["deepseek_source_q2_tier_math"] = "mixed_v1"
    return p


G = 1_000_000_000
arms = [
    ("exact_90", orig()),
    ("plain_l0p7", merge(orig(), bias())),
    ("plain_l1p5", merge(orig(), bias(lam=1.5))),
    ("guarded_l3p0", merge(orig(), bias("resident_guarded_v1", 3.0))),
    ("plain_l0p7_drop_miss", merge(orig(), bias(), drop("drop_miss_v1", [((0, 39), 2)]))),
    ("plain_l0p7_q2tol10", merge(tiers(10 * G), bias())),
    ("plain_l0p7_q2tol20", merge(tiers(20 * G), bias())),
    ("plain_l0p7_q2tol20_ref", merge(tiers(20 * G, mixed=False), bias())),
    ("plain_l0p7_q2tol20_drop_miss", merge(tiers(20 * G), bias(), drop("drop_miss_v1", [((0, 39), 2)]))),
]
spec = {"base": {}, "arms": [{"name": n, "policy": p} for n, p in arms], "max_tokens": 192}
Path(__file__).with_name("balance.json").write_text(json.dumps(spec, indent=1))
print(len(arms), "arms")
