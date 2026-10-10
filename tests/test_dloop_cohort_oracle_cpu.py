"""Cohort draft-loop oracle: padded-cohort rules over sampled lanes."""

import importlib.util
import math
import random
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dloop_cohort_oracle.py"
SPEC = importlib.util.spec_from_file_location("dloop_cohort_oracle", SCRIPT)
cohort = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cohort)

CONFIDENT = {"match": 7, "logq": [math.log(0.99)] * 3 + [0.0] * 4}
SHAKY = {"match": 0, "logq": [math.log(0.2)] * 3 + [0.0] * 4}


def _costs():
    # Width 2: base depth 3 is 8 rows (one tile); depth 7 is 16 rows.
    return {(2, 3): 0.130, (2, 7): 0.200}


def test_rules_differ_only_in_how_many_lanes_must_pass():
    rng = random.Random(1)
    result = cohort.evaluate_width([CONFIDENT, SHAKY], _costs(), 2, 3, [7], [-0.5], 4000, 0.0, rng)
    any_rate = result["any_-0.5_7"]["extend_rate"]
    all_rate = result["all_-0.5_7"]["extend_rate"]
    assert 0.70 < any_rate < 0.80 and 0.20 < all_rate < 0.30  # 1-(1/2)^2, (1/2)^2
    assert result["half_-0.5_7"]["extend_rate"] == any_rate  # half of 2 lanes is 1


def test_extending_fills_every_lane_and_the_oracle_bounds_the_rules():
    rng = random.Random(2)
    result = cohort.evaluate_width([CONFIDENT], _costs(), 2, 3, [7], [-0.5], 200, 0.0, rng)
    # Two confident lanes: extended, both verify 7 drafts -> 8 tokens each.
    assert result["any_-0.5_7"]["tokens_per_cycle"] == 16
    assert result["fixed_3"]["tokens_per_cycle"] == 8
    best_rule = min(v["ms_per_token"] for k, v in result.items() if not k.startswith("oracle"))
    assert result["oracle_7"]["ms_per_token"] <= best_rule + 1e-9
