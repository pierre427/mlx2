# SPDX-License-Identifier: MIT
"""Confidence-scheduled DSpark verify length (V4.1 tech report section 2.4.3).

The embedded confidence head predicts each draft position's conditional
acceptance probability p_i = sigmoid(c_i). Prefix survival is S_k = prod p_i
for i <= k, so verifying k drafts emits E(k) = 1 + sum_{j<=k} S_j tokens in
expectation. With a profiled relative cost C(w) for a w-row target verify and
a draft cost d, the scheduler picks k in [0, maximum] maximizing
E(k) / (d + C(k + 1)). The draft cost is already paid when the scores exist,
so it does not change the argmax; it is kept for the reported expected rate.

Scheduling changes only how many proposals are verified. Every committed
token is still the target's greedy choice (exact). Provenance:
provenance/deepseek-v41-dspark-chain-verify.json.
"""
import math

POLICIES = ("fixed", "survival_v1")
# Relative w-row verify cost, w = 1..6 (one ordinary target token = 1.0).
# Serial width-one rows cost w. Lockstep multirow reuses post-attention dense
# coefficients and the head (about 8.7 of 14.3 GB per token on the resident
# byte model), so each extra row is estimated at ~0.45 of a token. These are
# planning estimates, not measurements; calibrate with
# deepseek_dspark_verify_costs from a controlled run.
DEFAULT_COSTS = {
    "serial": (1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    "lockstep_multirow_v1": (1.0, 1.45, 1.9, 2.35, 2.8, 3.25),
}
DEFAULT_DRAFT_COST = 0.35


def validate(policy, costs=None, draft_cost=None):
    if policy not in POLICIES:
        raise ValueError("DSpark verify schedule must be fixed or survival_v1")
    if costs is not None:
        if (not isinstance(costs, (list, tuple)) or len(costs) != 6
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in costs)
                or any(b < a for a, b in zip(costs, costs[1:]))):
            raise ValueError("DSpark verify costs must be six positive nondecreasing numbers")
    if draft_cost is not None and (type(draft_cost) not in (int, float)
                                   or not math.isfinite(draft_cost) or draft_cost < 0):
        raise ValueError("DSpark draft cost must be a finite nonnegative number")
    if policy == "fixed" and (costs is not None or draft_cost is not None):
        raise ValueError("DSpark verify costs require survival_v1")


def survival(confidence_logits):
    result, alive = [], 1.0
    for value in confidence_logits:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("DSpark confidence must be finite")
        p = 1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))
        alive *= p
        result.append(alive)
    return result


def choose(confidence_logits, *, maximum, costs, draft_cost=DEFAULT_DRAFT_COST):
    """Return (k, receipt). Ties prefer the shorter verify."""
    if type(maximum) is not int or not 0 <= maximum <= 5 or len(costs) < maximum + 1:
        raise ValueError("DSpark schedule maximum/cost geometry mismatch")
    alive = survival(list(confidence_logits)[:maximum])
    best, best_rate, expected = 0, 1.0 / (draft_cost + costs[0]), 1.0
    total = 1.0
    for k, s in enumerate(alive, 1):
        total += s
        rate = total / (draft_cost + costs[k])
        if rate > best_rate * (1 + 1e-12):
            best, best_rate, expected = k, rate, total
    return best, {"verify_drafts": best, "expected_tokens": expected,
                  "expected_rate": best_rate, "survival": alive}
