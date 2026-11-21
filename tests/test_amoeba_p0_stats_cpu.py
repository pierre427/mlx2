"""CPU tests of the P0 statistics and decision rules (scripts/amoeba_battery_stats.py) and the CPU halves of the two P0
drivers (closing orders and depth schedules of scripts/amoeba_fn_pareto.py; random controls of amoeba_p0_lib):
docs/research/2026-11-20-amoeba-next-experiments.md 1.5, 4.3-4.5, 7.1, 7.5. No model."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import amoeba_battery_stats as ST  # noqa: E402
import amoeba_fn_pareto as PA  # noqa: E402
import amoeba_p0_lib as P0  # noqa: E402


# ----------------------------------------------------------------------------- per-document rows and bootstrap

def test_perdoc_sidecar_roundtrip_and_npz(tmp_path):
    p = tmp_path / "perdoc.jsonl"
    ST.append_perdoc(p, "a", "core_plus", "a", {"ce": [6.0, 12.0], "clean_ce": [3.0, 6.0], "n": [6, 6]})
    ST.append_perdoc(p, "a", "core_plus", "a", {"ce": [9.0, 12.0], "clean_ce": [3.0, 6.0], "n": [6, 6]})  # resumed
    rows = ST.load_perdoc(p)
    assert list(rows) == [("a", "core_plus", "a")]
    np.testing.assert_allclose(ST.doc_dce(rows[("a", "core_plus", "a")]), [1.0, 1.0])
    assert ST.to_npz(p, tmp_path / "battery.perdoc.npz") == 1
    z = np.load(tmp_path / "battery.perdoc.npz")
    np.testing.assert_allclose(z["a|core_plus|a|dce"], [1.0, 1.0])


def test_doc_meta_groups_and_strata():
    rows = [{"id": "x1", "group": "fineweb-edu:<u1>", "source": "fineweb-edu"},
            {"id": "x2", "group": "wikipedia-eng:7", "source": "wikipedia-eng"},
            {"id": "x3", "group": "se/law/1", "register": "qa", "source": "stackexchange:law/1"},
            {"question": "q"}]
    m = ST.doc_meta(rows, "s")
    assert m["groups"] == ["fineweb-edu:<u1>", "wikipedia-eng:7", "se/law/1", "doc:3"]
    assert m["strata"] == ["fineweb-edu", "wikipedia-eng", "qa", "all"]


def test_group_bootstrap_moves_groups_together_and_keeps_strata():
    groups = ["g0"] * 5 + ["g1"] * 5 + ["h0", "h1"]
    strata = ["s"] * 10 + ["t", "t"]
    for ix in ST.boot_indices(groups, strata, n_boot=200, seed=3):
        assert ix.size == 12
        c = np.bincount(ix, minlength=12)
        assert all(len(set(c[i] for i in range(g * 5, g * 5 + 5))) == 1 for g in (0, 1))   # whole groups
        assert c[:10].sum() == 10 and c[10:].sum() == 2                                     # strata sizes kept
    v = np.r_[np.zeros(5), np.ones(5), [5.0, 5.0]]
    lo, hi = ST.group_bootstrap(v, groups, strata, n_boot=500)
    assert lo < v.mean() < hi


def test_paired_se_and_ci():
    rng = np.random.default_rng(0)
    a = rng.normal(0.1, 0.05, 80)
    b = a - 0.02 + rng.normal(0, 0.001, 80)
    r = ST.paired(a, b, [str(i) for i in range(80)], None, n_boot=1000)
    d = a - b
    assert r["mean"] == pytest.approx(d.mean()) and r["se"] == pytest.approx(d.std(ddof=1) / math.sqrt(80))
    assert r["ci95"][0] < 0.02 < r["ci95"][1] and r["ci95"][0] > 0.0195 - 0.002


# ----------------------------------------------------------------------------- 1.5 rules

def _own(cp, ca, rd, ri, n=40, noise=0.002, seed=0):
    g = np.random.default_rng(seed)
    e = lambda: g.normal(0, noise, n)                                                   # noqa: E731
    return {"core_plus": cp + e(), "core_alone": ca + e(), "core_plus_rand": rd + e(), "core_plus_rand_inexp": ri + e()}


def test_suff_rules_useful_served_and_four_way():
    # language case: core alone fails badly, the box recovers it, random reopenings do not
    r = ST.suff_box(_own(0.03, 0.9, 0.88, 0.85), necessity_net=0.5)
    assert r["useful"] and r["served"] == "yes" and r["four_way"] == "real_box"
    assert r["recovery"] == pytest.approx((r["core_alone_own"] - r["core_plus_own"]) / r["core_alone_own"])
    # topic case: core alone already passes -> vacuous; the box adds ~nothing -> cosmetic
    r = ST.suff_box(_own(0.013, 0.014, 0.0135, 0.0138), necessity_net=0.001)
    assert r["served"] == "vacuous" and not r["useful"] and r["four_way"] == "cosmetic"
    # vacuous and still useful (1.5: core_alone 0.04, core_plus 0.01)
    r = ST.suff_box(_own(0.01, 0.04, 0.04, 0.04), necessity_net=0.002)
    assert r["served"] == "vacuous" and r["useful"] and r["four_way"] == "redundant_or"
    # "no" wins over "vacuous" (order of the rule): core_plus over the bar
    r = ST.suff_box(_own(0.07, 0.04, 0.04, 0.04), necessity_net=0.02)
    assert r["served"] == "no" and r["four_way"] == "needed_core_plus_fails"


def test_suff_random_ratio_and_floor():
    # gain 0.02 but an equal-size random reopening gains 0.015 -> 0.02 < 2 x 0.015: not useful
    r = ST.suff_box(_own(0.06, 0.08, 0.065, 0.07), necessity_net=0.0)
    assert r["suff_gain"]["mean"] == pytest.approx(0.02, abs=2e-3) and not r["useful"]
    assert r["rand_gain_used"] == pytest.approx(0.015, abs=2e-3)
    # random gains nothing: the 0.005 floor keeps the ratio defined (0.012 >= 2 x 0.005)
    r = ST.suff_box(_own(0.06, 0.072, 0.072, 0.072, noise=0.0005), necessity_net=0.0)
    assert r["rand_gain_used"] == 0.005 and r["useful"]
    # below the absolute bar: 0.008 < 0.01
    r = ST.suff_box(_own(0.06, 0.068, 0.068, 0.068, noise=0.0005), necessity_net=0.0)
    assert not r["useful"]


def test_suff_ci_lower_bound_gate():
    g = np.random.default_rng(1)
    own = _own(0.05, 0.065, 0.065, 0.065, noise=0.0)
    own["core_alone"] = own["core_alone"] + g.normal(0, 0.2, 40)                    # huge spread: CI crosses 0
    r = ST.suff_box(own, necessity_net=0.0)
    assert r["suff_gain"]["ci95"][0] < 0 and not r["useful"]


def test_suff_ablation_sensitivity():
    own = _own(0.01, 0.06, 0.06, 0.06)
    mean_own = {"core_plus": own["core_plus"], "core_alone": own["core_plus"] + 0.001}    # mean ablation: no gain
    r = ST.suff_box(own, necessity_net=0.02, mean_own=mean_own, mean_necessity_net=0.02)
    assert r["four_way"] == "real_box" and r["mean_ablation"]["four_way"] == "needed_core_plus_fails"
    assert r["ablation_sensitive"] is True
    r = ST.suff_box(own, necessity_net=0.02, mean_own={"core_plus": own["core_plus"], "core_alone": own["core_alone"]},
                    mean_necessity_net=0.02)
    assert r["ablation_sensitive"] is False


def test_expected_false_positives():
    assert ST.expected_false_positives(100, 0.005) == pytest.approx(100 * 0.5 * math.erfc(2 / math.sqrt(2)))
    assert ST.expected_false_positives(100, None) is None


# ----------------------------------------------------------------------------- 4.5 rules

def test_cf_interpolation_censoring_and_cliff():
    cf, c = ST.cf_at([0.1, 0.2, 0.3, 0.4], [0.01, 0.03, 0.07, 0.2], 0.05)
    assert cf == pytest.approx(0.2 + 0.1 * (0.02 / 0.04)) and not c
    cf, c = ST.cf_at([0.1, 0.2], [0.01, 0.02], 0.05)
    assert cf == 0.2 and c
    cf, c = ST.cf_at([0.1, 0.2], [0.10, 0.2], 0.05)
    assert cf == pytest.approx(0.1 * 0.05 / 0.10)
    # a cliff-flagged cell counts as over the bar, without interpolating into it
    cf, _ = ST.cf_at([0.1, 0.2, 0.3], [0.01, 0.02, 0.03], 0.05, [False, True, False])
    assert cf == pytest.approx(0.3)          # the largest fraction under the bar that is not flagged
    cf, _ = ST.cf_at([0.1, 0.2], [0.01, 0.02], 0.05, [False, True])
    assert cf == pytest.approx(0.1)


def test_pareto_cf_band_and_ub():
    rng = np.random.default_rng(0)
    groups = [str(i) for i in range(80)]
    cells = [{"fraction": f, "english": c + rng.normal(0, 0.01, 80)} for f, c in ((0.1, 0.01), (0.2, 0.04),
                                                                                  (0.3, 0.08), (0.4, 0.15))]
    r = ST.pareto_cf(cells, groups, None, n_boot=400)
    assert 0.2 <= r["cf"] <= 0.3
    assert r["cells"][1]["english_ub95"] > r["cells"][1]["english_mean"]
    assert r["band"][0] <= r["band"][1] and r["band"][1] >= r["cf"]     # the mean curve crosses later than the ub


# ----------------------------------------------------------------------------- closing orders (4.3) and controls (7.5)

def test_global_and_expert_orders():
    L, E, F = 3, 4, 8
    N = E * F
    key = np.arange(L * N, dtype=float)[::-1].reshape(L, N).copy()        # descending: the last unit closes first
    o = PA.global_order("k", key, F, 0)
    c = o.closed(0.25)
    assert c.sum() == round(0.25 * L * N) and c[-1, -24:].all()            # the 24 smallest keys
    lc = o.closed(0.25, per_layer=[0.5, 0.25, 0.0])
    assert list(lc.sum(1)) == [16, 8, 0] and lc[0, 16:].all()
    ek = np.array([[3, 1, 2, 0], [5, 6, 7, 8], [9, 10, 4, 11]], float)
    eo = PA.expert_order("e", ek, F, 0)
    c = eo.closed(0.2)                                                      # 0.2 x 96 = 19.2 units -> 2 experts
    assert c.sum() == 2 * F
    closed_e = sorted({(l, u // F) for l, u in zip(*np.nonzero(c))})
    assert closed_e == [(0, 1), (0, 3)] and all(c[l, e * F:(e + 1) * F].all() for l, e in closed_e)
    dc = eo.closed(0.25, per_layer=[0.25, 0.25, 0.25])                     # one whole expert per layer
    assert list(dc.sum(1)) == [F, F, F] and dc[2, 2 * F:3 * F].all()


def test_ties_are_seeded_and_rand_is_layer_matched():
    L, N, F = 3, 32, 8
    z = np.zeros((L, N))
    a, b = PA.global_order("z", z, F, 1).closed(0.3), PA.global_order("z", z, F, 1).closed(0.3)
    assert np.array_equal(a, b) and not np.array_equal(a, PA.global_order("z", z, F, 2).closed(0.3))
    r = PA.rand_order(L, N, F, 0).closed(0.25)
    assert list(r.sum(1)) == [8, 8, 8]
    r2 = PA.rand_order(L, N, F, 0).closed(0.5)
    assert (r2 | ~r).all() or (r & r2).sum() == r.sum()                    # nested: 25% inside 50%


def test_fire_order_selected_close_last():
    mt = np.array([[0.5, 0.1, 0.9, 0.2]])
    sel = np.array([[False, True, False, False]])
    glob = PA._fire_glob(mt, sel, 0)
    assert list(np.argsort(glob)) == [3, 0, 2, 1]                           # 0.2, 0.5, 0.9, then the selected unit


class _Tiles:
    def __init__(self, masks):
        self.keys = list(masks)
        self.shape = next(iter(masks.values())).shape
        self.idx = {k: np.flatnonzero(v).astype(np.int32) for k, v in masks.items()}

    def units(self, k):
        return int(self.idx[k].size)


def test_part_order_sequence_and_limit():
    L, N = 2, 16
    m = lambda cols: np.isin(np.arange(N), cols)[None].repeat(L, 0)        # noqa: E731
    tiles = {"core": m([0, 1, 2, 3, 4, 5, 6, 7]), "box:a": m([8, 9]), "box:lang.eng": m([10]),
             "pair:a__b": m([11]), "group:g": m([12, 13, 14]), "joint:a+b+c": m([15])}
    jprep = {"english_box": {"name": "lang.eng", "closed_for": []}}
    seq = PA.part_sequence(jprep, [k for k in tiles if k != "core"], {k: int(v.sum()) for k, v in tiles.items()})
    assert seq == ["joint:a+b+c", "box:a", "group:g", "pair:a__b"]          # by size, pairs last, no English box
    o, info = PA.part_order(_Tiles(tiles), jprep, 8, 0)
    assert info["closable_units"] == 2 * 7
    c = o.closed(2 / 32)                                                    # 2 units: the joint tile
    assert c.sum() == 2 and c[:, 15].all()
    assert o.closed(0.5) is None                                            # past the closable share: n/a
    assert not (o.closed(14 / 32) & tiles["core"]).any() and not o.closed(14 / 32)[:, 10].any()


def test_depth_schedules():
    for f in (0.1, 0.2, 0.4):
        t = PA.depth_fractions("D-tenp", f, 48)
        assert t.mean() == pytest.approx(f) and t[0] > t[24] > t[47] and (np.diff(t) <= 1e-12).all()
        e = PA.depth_fractions("D-ends", f, 48)
        assert e.mean() == pytest.approx(f) and np.allclose(e[:6], f / 2) and np.allclose(e[42:], f / 2)
        assert np.allclose(e[6:42], e[6])
    t = PA.depth_fractions("D-tenp", 0.9, 48)
    assert t.max() <= 1.0 + 1e-12 and t.mean() == pytest.approx(0.9)


def test_random_controls():
    g = np.random.default_rng(0)
    L, E, F = 2, 4, 8
    mask = np.zeros((L, E * F), bool)
    mask[0, [0, 1, 9]] = True
    mask[1, [17, 18, 19]] = True
    cand = np.ones_like(mask)
    cand[0, :4] = False
    r, short = P0.random_in_experts(mask, cand, F, g)
    for l in range(L):
        assert (r[l].reshape(E, F).sum(1) == mask[l].reshape(E, F).sum(1)).all()
    assert not (r & ~cand).any() and short == 0
    cand2 = np.zeros_like(mask)
    cand2[0, 8] = True                                                      # expert 1 of layer 0 has one candidate
    r2, short2 = P0.random_in_experts(mask, cand2, F, g)
    assert r2.sum() == 1 and short2 == 5
    r3, s3 = P0.random_like(mask, g, cand=cand)
    assert (r3.sum(1) == mask.sum(1)).all() and not (r3 & ~cand).any() and s3 == 0


def test_suff_auto_subset_and_repeat():
    import amoeba_fn_suff as S
    rows = []
    for b, nec in [("deu", 1.0), ("cmn", 0.2), ("sci.physics", 0.004)] + [(f"t{i}", i / 1000) for i in range(12)]:
        rows.append({"box": b, "without": {b: nec}, "random": {b: 0.0}})
    sub = S.auto_subset(rows)
    assert sub[:2] == ["cmn", "deu"] and sub[-1] == "sci.physics" and len(sub) == 12
    assert sub[2:11] == [f"t{i}" for i in range(11, 2, -1)]
    rep = S.auto_repeat(sub, rows)
    assert rep[:2] == ["deu", "cmn"] and rep[2:] == ["t11", "t10", "t9", "t8", "t7"]


def test_rules_are_recorded_in_json():
    json.dumps(ST.SUFF_RULES)
    json.dumps(ST.PARETO_RULES)
    assert ST.SUFF_RULES["min_gain"] == 0.01 and ST.PARETO_RULES["bar"] == 0.05 and ST.N_BOOT == 2000


def test_append_line_closes_a_partial_last_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"key": "a", "arm": "x", "set": "s", "ce": [1.0], "clean_ce": [0.0], "n": [1]}\n{"key": "b", "ar')
    ST.append_perdoc(p, "c", "x", "s", {"ce": [2.0], "clean_ce": [0.0], "n": [1]})
    rows = ST.load_perdoc(p)
    assert set(rows) == {("a", "x", "s"), ("c", "x", "s")}


def test_unmatched_random_reopening_leaves_the_ratio():
    own = _own(0.06, 0.08, 0.079, 0.065, noise=0.0005)      # rand gains 0.001, rand_inexp 0.015 (but 10% filled)
    r = ST.suff_box(own, necessity_net=0.0, fill={"core_plus_rand": 1.0, "core_plus_rand_inexp": 0.1})
    assert r["rand_gain_inexp"]["size_matched"] is False and r["rand_gain"]["size_matched"] is True
    assert r["rand_gain_used"] == 0.005 and r["useful"]
    r = ST.suff_box(own, necessity_net=0.0, fill={"core_plus_rand": 1.0, "core_plus_rand_inexp": 0.95})
    assert r["rand_gain_used"] == pytest.approx(0.015, abs=2e-3) and not r["useful"]


def test_cf_mean_is_reported_beside_the_upper_bound():
    rng = np.random.default_rng(1)
    cells = [{"fraction": f, "english": c + rng.normal(0, 0.02, 80)} for f, c in ((0.1, 0.02), (0.2, 0.045),
                                                                                  (0.3, 0.06), (0.4, 0.1))]
    r = ST.pareto_cf(cells, [str(i) for i in range(80)], None, n_boot=400)
    assert r["cf_mean"] >= r["cf"] and r["band"][0] <= r["cf_mean"] <= r["band"][1]
