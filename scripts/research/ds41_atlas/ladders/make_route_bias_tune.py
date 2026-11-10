#!/usr/bin/env python3
"""Route-bias tuning ladder (Pierre 2026-11-09: balance push starts from route bias).

Lambda strength, guarded vs plain, adaptive margin, and smaller expert budgets at the
strongest clean setting. Non-tier arms: the default kernel bundle applies automatically.
"""
import json
from pathlib import Path

G = 1_000_000_000


def rb(mode="resident_guarded_v1", lam=0.35, budget=90 * G, adaptive=False):
    p = {"deepseek_source_experts": {"mode": "original_lfu", "budget_bytes": budget},
         "deepseek_source_route_bias": mode}
    if adaptive:
        p["deepseek_source_route_bias_adaptive"] = "margin_v1"
    else:
        p["deepseek_source_route_bias_lambda"] = lam
    return p


arms = [("exact_90", {"deepseek_source_experts": {"mode": "original_lfu", "budget_bytes": 90 * G}})]
arms += [(f"guarded_l{str(l).replace('.', 'p')}", rb(lam=l)) for l in (0.35, 0.7, 1.5, 3.0)]
arms += [(f"plain_l{str(l).replace('.', 'p')}", rb("resident_v1", l)) for l in (0.35, 0.7)]
arms += [("guarded_adaptive", rb(adaptive=True))]
arms += [(f"guarded_l0p7_{b}gb", rb(lam=0.7, budget=b * G)) for b in (60, 40)]
spec = {"base": {}, "arms": [{"name": n, "policy": p} for n, p in arms], "max_tokens": 192}
Path(__file__).with_name("route_bias_tune.json").write_text(json.dumps(spec, indent=1))
print(len(arms), "arms")
