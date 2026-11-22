"""--joint-floor auto, rule coverage-v2 (Pierre 2026-11-21). CPU only: pure selection over a sweep table."""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("amoeba_fn_partition_jf", ROOT / "scripts/amoeba_fn_partition.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _mod()
FLOORS = (5000, 10000, 20000, 50000)


def _sweep(sizes, floors=FLOORS):
    """The table main() builds: per floor, the claim sets with >= floor units."""
    return {str(f): {"tiles": sum(s >= f for s in sizes), "units_moved": sum(s for s in sizes if s >= f)}
            for f in floors}


def test_mass_in_a_few_large_sets_takes_the_highest_floor():
    sizes = [400_000, 300_000, 6_000, 7_000, 12_000]            # 700k of 725k sits in the two large sets
    floor, why, rule = P.choose_joint_floor(_sweep(sizes))
    assert floor == 50000 and "coverage-v2" in why and rule["name"] == "coverage-v2"
    assert rule["reference_floor"] == 5000 and rule["reference_units"] == sum(sizes)
    assert rule["share_of_reference"]["50000"] == round(700_000 / 725_000, 6)


def test_mass_spread_over_mid_size_sets_steps_down():
    sizes = [60_000] + [25_000] * 8 + [12_000] * 10 + [6_000] * 5   # 410k: >=50k 15%, >=20k 63%, >=10k 93%
    floor, why, _ = P.choose_joint_floor(_sweep(sizes))
    assert floor == 10000 and "19 tiles" in why
    # the rule before 2026-11-21 (largest units per tile) took the highest floor with a tile here
    sweep = _sweep(sizes)
    old = max(sweep, key=lambda f: (sweep[f]["units_moved"] / max(sweep[f]["tiles"], 1), -int(f)))
    assert old == "50000"


def test_coverage_threshold_is_inclusive_and_adjustable():
    sizes = [80, 20]                                             # floor 50 moves exactly 0.8 of floor 10's units
    sweep = _sweep(sizes, floors=(10, 50))
    assert P.choose_joint_floor(sweep, 0.8)[0] == 50
    assert P.choose_joint_floor(sweep, 0.81)[0] == 10
    assert P.choose_joint_floor(sweep, 1.0)[0] == 10
    assert P.choose_joint_floor(_sweep([80, 20], floors=(10, 15)), 1.0)[0] == 15   # same sets: the higher floor


def test_no_set_at_the_lowest_floor_writes_no_tile():
    floor, why, rule = P.choose_joint_floor(_sweep([100, 200]))
    assert floor == 5000 and "no joint tile" in why and rule["reference_units"] == 0


def test_tile_cap_steps_up_and_refuses_when_nothing_fits():
    sizes = [60_000] + [25_000] * 8 + [12_000] * 10 + [6_000] * 5
    sweep = _sweep(sizes)
    floor, why, rule = P.choose_joint_floor(sweep, 0.8, max_tiles=9)
    assert floor == 20000 and "capped to floor 20000" in why and rule["max_tiles"] == 9
    assert P.choose_joint_floor(sweep, 0.8, max_tiles=19)[0] == 10000    # the cap already holds: unchanged
    assert P.choose_joint_floor(sweep, 0.8, max_tiles=1)[0] == 50000
    with pytest.raises(SystemExit, match="--joint-max-tiles 1"):
        P.choose_joint_floor(_sweep([60_000, 70_000] + [12_000] * 30), 0.8, max_tiles=1)
    # a floor that would write no tile does not "fit" the cap
    empty_above = _sweep([12_000] * 4 + [6_000] * 11)                # 5000: 15 tiles, 10000: 4, 20000 and 50000: 0
    assert P.choose_joint_floor(empty_above, 0.8, max_tiles=4)[0] == 10000
    with pytest.raises(SystemExit, match="between 1 and 3 tiles"):
        P.choose_joint_floor(empty_above, 0.8, max_tiles=3)


def test_threshold_is_exact_for_large_unit_counts():
    ref = 60_000_000                                                 # 0.55 * ref is not exact in binary floating point
    sweep = {"10": {"tiles": 9, "units_moved": ref}, "20": {"tiles": 2, "units_moved": 33_000_000},
             "30": {"tiles": 1, "units_moved": 32_999_999}}
    assert P.choose_joint_floor(sweep, 0.55)[0] == 20


def test_choice_is_not_always_the_highest_floor_with_a_tile():
    """The defect this replaces: over random size distributions the old rule returned the highest floor with a
    tile every time; coverage-v2 returns each swept floor for some distribution and always meets its coverage."""
    rng = random.Random(20261121)
    seen = set()
    for _ in range(400):
        sizes = [int(rng.lognormvariate(rng.uniform(8.5, 11.5), 1.0)) for _ in range(rng.randint(3, 60))]
        sweep = _sweep(sizes)
        floor, _, _ = P.choose_joint_floor(sweep)
        ref = sweep["5000"]["units_moved"]
        if not ref:
            continue
        seen.add(floor)
        assert sweep[str(floor)]["units_moved"] >= 0.8 * ref
        higher = [f for f in FLOORS if f > floor]
        assert all(sweep[str(f)]["units_moved"] < 0.8 * ref for f in higher)
    assert seen == set(FLOORS)
