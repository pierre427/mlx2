#!/usr/bin/env python3
"""Oracle-verification ceiling: how much speedup adaptive speculation leaves.

vLLM's Q4 2026 list asks how much opportunity remains between adaptive
speculative decoding and an "oracle" that knows each round's outcome in
advance.  This tool answers it offline from receipts we already write
(``scripts/spec_acceptance_gsm8k.py``: per-request ``accept_hist`` -- accepted
drafted tokens per verify round -- and ``depth_hist`` -- drafted depth per
round), with no GPU work.

Round cost model ``c(d)``: the time of one round that drafts ``d`` tokens and
verifies ``d + 1`` rows.  Two measured points anchor it: ``c(0)``, one
ordinary decode step (``--ordinary-arm``, from that arm's decode tok/s), and
the speculative arm's own measured seconds per round.  Between them the
shape is unknown without a depth sweep, so the tool prices two bounding
shapes and reports both:

* ``flat``: every round with ``d >= 1`` costs the measured round time.  Right
  for a parallel drafter (DFlash/DSpark draft every position in one forward)
  on a memory-bound verify; the oracle can then only save the rounds that
  accept nothing.  Lower bound on the ceiling.
* ``linear``: ``c(d)`` grows linearly from ``c(0)`` to ``c(K)``.  Right for an
  autoregressive drafter (EAGLE/MTP chains, one drafter step per position).
  Upper bound for a parallel drafter.

``--costs`` replaces both with explicit per-depth seconds (or relative units)
when a depth sweep measured them.

Policies, all priced with the same ``c(d)`` and the same per-round accepted
counts, so the ratios compare like with like:

* ``actual``: the arm's realized depth mix (``depth_hist``) and tokens
  (``accepted + 1`` per round).
* ``best_fixed``: the best single depth ``d <= K`` chosen with hindsight;
  a round that accepted ``a`` yields ``min(a, d) + 1`` tokens.
* ``oracle``: each round drafts exactly its accepted count ``a`` (all
  accepted, plus the bonus): ``a + 1`` tokens at cost ``c(a)``.  A round that
  accepted its whole drafted depth might have accepted more, so the oracle
  is censored at ``K``: a lower bound on the true ceiling (reported as
  ``full_accept_censored_rounds``).

``opportunity_vs_actual = oracle / actual - 1`` is the most an adaptive depth
policy could still gain on this traffic at depth cap ``K``; ``best_fixed``
separates the part a static retune would already capture.
Sanity/attribution numbers, not publishable statistics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _hist(value):
    return {int(k): int(v) for k, v in (value or {}).items() if int(v)}


def arm_rounds(rows):
    """Pooled accepted-count and depth histograms plus measured decode time."""
    accepted, depths, rounds, seconds, tokens = {}, {}, 0, 0.0, 0
    for row in rows:
        a = _hist(row.get("accept_hist"))
        completion, tps = row.get("completion_tokens") or 0, row.get("decode_tps")
        # Rounds and seconds come from the same rows, or the measured
        # seconds per round would be diluted by untimed rows.
        if not a or not (tps and completion > 1):
            continue
        for k, v in a.items():
            accepted[k] = accepted.get(k, 0) + v
        for k, v in _hist(row.get("depth_hist")).items():
            depths[k] = depths.get(k, 0) + v
        rounds += sum(a.values())
        seconds += (completion - 1) / tps
        tokens += completion - 1
    return {"accepted": accepted, "depths": depths, "rounds": rounds,
            "decode_seconds": seconds, "decode_tokens": tokens}


def ordinary_step_seconds(rows):
    seconds = tokens = 0.0
    for row in rows:
        completion, tps = row.get("completion_tokens") or 0, row.get("decode_tps")
        if tps and completion > 1:
            seconds += (completion - 1) / tps
            tokens += completion - 1
    if not tokens:
        raise ValueError("ordinary arm has no decode timing")
    return seconds / tokens


def linear_costs(c0, ck, depth):
    if depth < 1 or c0 <= 0 or ck <= 0:
        raise ValueError("need depth >= 1 and positive costs")
    return [c0 + (ck - c0) * d / depth for d in range(depth + 1)]


def flat_costs(c0, ck, depth):
    if depth < 1 or c0 <= 0 or ck <= 0:
        raise ValueError("need depth >= 1 and positive costs")
    return [c0] + [ck] * depth


def ceiling(accepted, depths, costs):
    """Price the three policies; ``costs[d]`` is the cost of a depth-d round."""
    if not accepted:
        raise ValueError("no speculation rounds")
    cap = max(depths) if depths else max(accepted)
    if len(costs) <= cap:
        raise ValueError(f"costs cover depths 0..{len(costs) - 1}, receipts reach {cap}")
    if any(not c > 0 for c in costs[: cap + 1]):
        raise ValueError("costs must be positive")
    rounds = sum(accepted.values())
    if depths and sum(depths.values()) != rounds:
        raise ValueError("depth and accept histograms count different rounds")
    tokens = sum((a + 1) * n for a, n in accepted.items())
    actual_cost = (sum(costs[d] * n for d, n in depths.items()) if depths
                   else costs[cap] * rounds)
    actual = tokens / actual_cost
    fixed = {}
    for d in range(cap + 1):
        fixed[d] = sum((min(a, d) + 1) * n for a, n in accepted.items()) / (costs[d] * rounds)
    best_depth = max(fixed, key=fixed.get)
    oracle_cost = sum(costs[min(a, cap)] * n for a, n in accepted.items())
    oracle = tokens / oracle_cost
    censored = sum(n for a, n in accepted.items() if a >= cap)
    return {
        "rounds": rounds, "depth_cap": cap,
        "mean_accepted_length": tokens / rounds,
        "tokens_per_cost": {"actual": actual, "best_fixed": fixed[best_depth], "oracle": oracle},
        "best_fixed_depth": best_depth,
        "fixed_depth_tokens_per_cost": {str(d): v for d, v in fixed.items()},
        "opportunity_vs_actual": oracle / actual - 1,
        "best_fixed_vs_actual": fixed[best_depth] / actual - 1,
        "oracle_vs_best_fixed": oracle / fixed[best_depth] - 1,
        # Counted at the depth cap only: with mixed drafted depths a round
        # that accepted a shallower full draft is censored too, which pooled
        # histograms cannot show; the oracle stays a lower bound either way.
        "full_accept_censored_rounds": censored,
        "oracle_is_lower_bound": censored > 0 or len(depths) > 1,
    }


def _arm(receipt, name):
    for arm in receipt["arms"]:
        if arm["name"] == name:
            return arm
    raise SystemExit(f"arm {name!r} not in receipt")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("receipt", type=Path, help="spec_acceptance_gsm8k receipt.json")
    parser.add_argument("--arm", action="append", required=True, help="speculative arm name (repeatable)")
    parser.add_argument("--ordinary-arm", help="ordinary arm in the same or --ordinary-receipt receipt")
    parser.add_argument("--ordinary-receipt", type=Path)
    parser.add_argument("--costs", help="JSON list: cost of a round drafting d = 0..K tokens")
    args = parser.parse_args(argv)
    receipt = json.loads(args.receipt.read_text())
    explicit = json.loads(args.costs) if args.costs else None
    c0 = None
    if explicit is None:
        if not args.ordinary_arm:
            parser.error("give --ordinary-arm (linear cost model) or --costs")
        source = json.loads(args.ordinary_receipt.read_text()) if args.ordinary_receipt else receipt
        c0 = ordinary_step_seconds(_arm(source, args.ordinary_arm)["rows"])
    out = {"receipt": str(args.receipt), "checkpoint": receipt.get("checkpoint"), "arms": {}}
    for name in args.arm:
        arm = _arm(receipt, name)
        pooled = arm_rounds(arm["rows"])
        if not pooled["rounds"]:
            out["arms"][name] = {"error": "no speculation rounds"}
            continue
        cap = max(pooled["depths"]) if pooled["depths"] else max(pooled["accepted"])
        if explicit is not None:
            result = ceiling(pooled["accepted"], pooled["depths"], [float(c) for c in explicit])
            result["cost_model"] = "explicit"
            out["arms"][name] = result
            continue
        result = {"measured_tok_s": pooled["decode_tokens"] / pooled["decode_seconds"],
                  "ordinary_tok_s": 1 / c0, "models": {}}
        zero = pooled["depths"].get(0, 0)
        if zero:
            result["note"] = f"{zero} rounds drafted nothing; priced at c(0)"
        # Measured round seconds are a mixture when depths vary; the flat
        # shape needs only the d >= 1 rounds' mean, after the d = 0 rounds
        # are taken out at c(0).
        drafting = pooled["rounds"] - zero
        ck = (pooled["decode_seconds"] - zero * c0) / drafting if drafting else None
        if ck is None or ck <= 0:
            result["error"] = "no drafting rounds with positive measured time"
        else:
            result["models"]["flat"] = ceiling(pooled["accepted"], pooled["depths"], flat_costs(c0, ck, cap))
            if len(pooled["depths"]) == 1:
                result["models"]["linear"] = ceiling(pooled["accepted"], pooled["depths"],
                                                     linear_costs(c0, ck, cap))
            else:
                result["models"]["linear"] = {"error": "mixed drafted depths: a linear shape cannot be "
                                                       "anchored on the mixture; pass --costs"}
        out["arms"][name] = result
    json.dump(out, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
