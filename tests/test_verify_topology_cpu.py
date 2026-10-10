"""Load-time verify topology: tile edges, candidate ends, selection, cache."""

import json

import pytest

from mlx2.runtime import verify_topology as vt

# M3 Pro, Qwen3.8-27B 4-bit, lane-simd crossover: whole-cycle seconds.
M3_LANE = {3: 0.1464, 4: 0.1779, 5: 0.2112, 6: 0.2487, 7: 0.1863, 8: 0.2656, 9: 0.2743}
M3_STOCK = {3: 0.1464, 4: 0.1780, 5: 0.2115, 6: 0.2500, 7: 0.2992, 8: 0.3378, 9: 0.3472}


def test_tile_edges_find_the_step_after_a_full_tile():
    rows = {r: 0.040 for r in range(1, 9)} | {r: 0.060 for r in range(9, 17)}
    assert vt.tile_edges(rows) == [8]
    assert vt.candidate_ends(3, rows, 9) == [7, 9]
    # Two padded lanes: 2 x (depth + 1) rows must fit under the edge.
    wide = {r: 0.040 for r in range(1, 17)} | {r: 0.060 for r in range(17, 21)}
    assert vt.candidate_ends(3, wide, 9, width=2) == [7, 9]
    assert vt.candidate_ends(3, rows, 9, width=2) == [9]  # 8 rows: depth 3 is the edge
    assert vt.candidate_ends(3, {r: 0.04 + 0.004 * r for r in range(1, 11)}, 9) == [9]


# M5 Max, production 27B oQ4e MTP, lane auto: whole-cycle seconds at 2 lanes.
M5_TWO = {3: 0.0656, 4: 0.0670, 5: 0.0711, 6: 0.0752, 7: 0.0784, 9: 0.1052}
M5_FOUR = {3: 0.0808, 4: 0.1073, 5: 0.1120, 6: 0.1195, 7: 0.1260, 9: 0.1665}


def test_selection_follows_the_host_cost_curve():
    grid = vt.load_prior_grid()
    lane = vt.select_topology(3, {1: M3_LANE}, grid)
    assert lane["selected"] == {"by_width": {"1": [3, 7]}, "threshold": -0.4, "cohort": "any"}
    best = max(lane["candidates"]["1"], key=lambda c: c["predicted_gain"])
    assert best["end"] == 7 and best["predicted_gain"] > 1.1
    stock = vt.select_topology(3, {1: M3_STOCK}, grid)
    assert stock["selected"] is None
    assert all(c["predicted_gain"] < 1.0 for c in stock["candidates"]["1"])


def test_width_two_pays_only_where_two_lanes_fit_one_tile():
    grid = vt.load_prior_grid()
    m5 = vt.select_topology(3, {2: M5_TWO, 4: M5_FOUR}, grid)
    assert m5["selected"]["by_width"] == {"2": [3, 7]}
    gain = next(c for c in m5["candidates"]["2"] if c["end"] == 7)["predicted_gain"]
    assert 1.15 < gain < 1.22  # the cohort oracle's Monte Carlo: 1.187
    m3_two = {3: 0.1695, 4: 0.2490, 5: 0.2605, 6: 0.2694, 7: 0.2793, 9: 0.5144}
    assert vt.select_topology(3, {2: m3_two}, grid)["selected"] is None


def test_one_lane_prediction_is_the_per_lane_gate():
    prior = vt.prior_at(vt.load_prior_grid(), -0.4)
    fixed, looped = vt.predict(3, 7, M3_LANE, prior, width=1)
    p = prior["extend_rate"]
    tokens = (1 - p) * prior["tokens_if_stopped"] + p * prior["tokens_if_extended"]["7"]
    seconds = (1 - p) * M3_LANE[3] + p * M3_LANE[7] + vt.GATE_SYNC_SECONDS
    assert looped == pytest.approx(1e3 * seconds / tokens)


def test_selection_refuses_a_base_or_threshold_the_prior_does_not_cover():
    grid = vt.load_prior_grid()
    assert vt.select_topology(2, {1: M3_LANE}, grid)["selected"] is None
    assert vt.select_topology(3, {1: M3_LANE}, grid, threshold=-2.5)["selected"] is None
    assert vt.select_topology(3, {1: M3_LANE}, grid, threshold=-0.01)["selected"] is None


@pytest.mark.parametrize("threshold", [-0.3, -0.4, -0.5, -0.45, -1.0])
def test_operator_thresholds_select_with_their_own_prior(threshold):
    choice = vt.select_topology(3, {1: M3_LANE}, vt.load_prior_grid(), threshold=threshold)
    assert choice["selected"]["by_width"] == {"1": [3, 7]}
    assert choice["selected"]["threshold"] == threshold


def test_prior_grid_is_exact_at_grid_points_and_linear_between():
    grid = vt.load_prior_grid()
    assert grid["base"] == 3 and vt.threshold_range(grid) == (-2.0, -0.05)
    at = vt.prior_at(grid, -0.4)
    row = next(r for r in grid["grid"] if r["threshold"] == -0.4)
    assert at["extend_rate"] == pytest.approx(row["extend_rate"])
    lo, hi = vt.prior_at(grid, -0.45), vt.prior_at(grid, -0.4)
    mid = vt.prior_at(grid, -0.425)
    assert mid["extend_rate"] == pytest.approx((lo["extend_rate"] + hi["extend_rate"]) / 2)
    # Looser gates extend more often.
    rates = [vt.prior_at(grid, t)["extend_rate"] for t in (-0.1, -0.4, -1.0, -2.0)]
    assert rates == sorted(rates)
    assert vt.prior_at(grid, -3.0) is None


def test_resolve_probes_once_then_reads_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX2_VERIFY_TOPOLOGY_CACHE", str(tmp_path))
    calls = {"rows": 0, "cycles": [], "max_rows": []}

    def rows(model, vocab, max_rows, reps=3):
        calls["rows"] += 1
        calls["max_rows"].append(max_rows)
        return {r: 0.04 if r <= 8 else (0.06 if r <= 16 else 0.09) for r in range(1, max_rows + 1)}

    def cycles(model, prompt, depths, width=1, **kwargs):
        if list(depths) == [3] and width == 1:  # the closing contention check
            return {3: M3_LANE[3]}
        calls["cycles"].append((width, list(depths)))
        table = M3_LANE if width == 1 else M5_TWO
        return {d: table[d] for d in depths}

    monkeypatch.setattr(vt, "probe_row_costs", rows)
    monkeypatch.setattr(vt, "probe_cycle_costs", cycles)
    kwargs = dict(identity={"device": "test", "law": "lane-simd-v1"}, base=3, max_end=9,
                  vocab_size=10, prompt=[1, 2, 3], widths=(1, 2))
    first = vt.resolve_topology(None, **kwargs)
    assert first["source"] == "probe" and first["tile_edges"] == [8, 16]
    assert first["stable"] and first["base_cycle_drift"] == 0
    assert calls["cycles"] == [(1, [3, 7, 9]), (2, [3, 7, 9])]
    assert first["selected"]["by_width"]["1"] == [3, 7]
    assert first["selected"]["by_width"]["2"] == [3, 7]
    second = vt.resolve_topology(None, **kwargs)
    assert second["source"] == "cache" and second["selected"] == first["selected"]
    assert calls["rows"] == 1 and len(calls["cycles"]) == 2
    # A different lane law, or a different set of widths, is a different topology.
    vt.resolve_topology(None, **{**kwargs, "identity": {"device": "test", "law": "stock"}})
    vt.resolve_topology(None, **{**kwargs, "widths": (1,)})
    assert calls["rows"] == 3
    assert calls["max_rows"] == [20, 20, 10]  # widest probed width x (9 + 1)
    assert len(list(tmp_path.glob("*.json"))) == 3
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["schema"] == vt.SCHEMA


def test_cache_is_keyed_by_the_self_mtp_policy_the_cycles_ran_under(tmp_path, monkeypatch):
    """The cycle probe runs real self-MTP cycles under the route's execution
    policy; another policy (shared-QSA indices, segmentation, persistence)
    has other cycle costs and must time its own, not read this one's."""
    monkeypatch.setenv("MLX2_VERIFY_TOPOLOGY_CACHE", str(tmp_path))
    probed = []

    def rows(model, vocab, max_rows, reps=3):
        return {r: 0.04 if r <= 8 else 0.06 for r in range(1, max_rows + 1)}

    def cycles(model, prompt, depths, *, self_mtp=None, **kwargs):
        probed.append(dict(self_mtp or {}))
        if (self_mtp or {}).get("share_qsa_indices"):
            # Deep cycles are expensive under this policy: no loop pays.
            return {d: M3_LANE[3] * (1 + 0.4 * (d - 3)) for d in depths}
        return {d: M3_LANE[d] for d in depths}

    monkeypatch.setattr(vt, "probe_row_costs", rows)
    monkeypatch.setattr(vt, "probe_cycle_costs", cycles)
    common = dict(identity={"device": "test", "law": "lane-simd-v1"}, base=3, max_end=9,
                  vocab_size=10, prompt=[1, 2, 3], widths=(1,))
    first = vt.resolve_topology(
        None, self_mtp={"num_draft": 3, "persistent": True}, **common
    )
    assert first["source"] == "probe" and first["selected"]["by_width"] == {"1": [3, 7]}
    second = vt.resolve_topology(
        None, self_mtp={"num_draft": 3, "persistent": True, "share_qsa_indices": True}, **common
    )
    assert second["source"] == "probe" and second["selected"] is None, second
    # Main's contention guard times every probe twice: two policies, four cycles.
    distinct = [dict(t) for t in {tuple(sorted(p.items())) for p in probed}]
    assert len(distinct) == 2 and any(p.get("share_qsa_indices") for p in probed[2:])
    probes_before = len(probed)
    # The same policy reads its own cache; a draft_loop key does not
    # change the identity (the probe strips it before timing).
    again = vt.resolve_topology(
        None, self_mtp={"num_draft": 3, "persistent": True}, **common
    )
    assert again["source"] == "cache" and again["selected"] == first["selected"]
    looped = vt.resolve_topology(
        None,
        self_mtp={"num_draft": 3, "persistent": True, "draft_loop": {"stage": 2, "threshold": -1.0}},
        **common,
    )
    assert looped["source"] == "cache" and len(probed) == probes_before
    assert first["key"]["self_mtp"] == looped["key"]["self_mtp"] != second["key"]["self_mtp"]


def test_a_contended_probe_selects_nothing_and_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX2_VERIFY_TOPOLOGY_CACHE", str(tmp_path))
    monkeypatch.setattr(vt, "probe_row_costs",
                        lambda model, vocab, max_rows, reps=3: {r: 0.04 if r <= 8 else 0.06
                                                                for r in range(1, max_rows + 1)})
    timings = iter([{3: 0.1464, 7: 0.1863, 9: 0.2743}, {3: 0.2100}])  # +43%: contended
    monkeypatch.setattr(vt, "probe_cycle_costs", lambda *a, **k: next(timings))
    receipt = vt.resolve_topology(None, identity={"d": 1}, base=3, max_end=9, vocab_size=10,
                                  prompt=[1], widths=(1,))
    assert receipt["stable"] is False and receipt["selected"] is None
    assert "unstable" in receipt["reason"]
    assert not list(tmp_path.glob("*.json"))
