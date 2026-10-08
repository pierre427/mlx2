from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/research/plan_dflash_multitile.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dflash_multitile_planner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evidence(module, *, explicit_rates=True):
    histograms = {
        4: {0: 2, 2: 2, 4: 4},
        8: {0: 2, 4: 4, 8: 2},
        12: {0: 4, 6: 3, 12: 1},
        15: {0: 4, 3: 3, 7: 1},
    }
    rates = {
        4: (0.5, 0.25),
        8: (0.25, 0.125),
        12: (0.125, 0.0625),
        15: (0.0, 0.0),
    }
    return module.EmpiricalAcceptance(
        accepted_count_histograms=histograms,
        full_tile_rates_by_k=rates if explicit_rates else None,
    )


def costs(module):
    return module.CostInputs(
        draft_ms_by_k={4: 10.0, 8: 14.0, 12: 18.0, 15: 22.0},
        verify_ms_by_k={4: 20.0, 8: 27.0, 12: 34.0, 15: 40.0},
        scheduler_boundary_ms=2.0,
    )


def test_source_separates_mechanisms_without_claiming_artifact_geometry():
    module = load_module()
    audit = module.audit_source(ROOT)
    assert audit["native_monolithic_ready"] is True
    assert audit["serial_committed_state_ready"] is True
    assert audit["progressive_same_proposal_api_ready"] is False
    assert audit["eager_frontier_exact_state_ready"] is False
    assert "artifact_block_size" not in audit
    plan = module.build_plan(root=ROOT, artifact_block_size=16)
    assert plan["artifact_observation"] == {"block_size": 16, "k15_legal": True}


def test_structural_plan_separates_probe_and_candidate_state_without_mlx():
    before = set(sys.modules)
    module = load_module()
    plan = module.build_plan(root=ROOT, artifact_block_size=16)
    assert len(plan["native_monolithic"]) == 4
    assert len(plan["exact_staged_serial"]) == 12
    assert len(plan["progressive_same_proposal"]) == 6
    assert len(plan["eager_frontier_precompute"]) == 24
    assert plan["probe_state"]["implemented"] is True
    assert plan["scope"]["authorizes_serving"] is False
    assert all(row["candidate_state"]["implemented"] for row in plan["native_monolithic"])
    assert all(
        not row["candidate_state"]["implemented"]
        for row in plan["progressive_same_proposal"]
        if not row["fixed_control"]
    )
    imported = set(sys.modules) - before
    assert not any(name == "mlx" or name.startswith("mlx.") for name in imported)


def test_serial_uses_histograms_and_empirical_tile_transitions():
    module = load_module()
    observed = evidence(module)
    committed, reach, source = module.expected_serial_commits(4, 3, observed)
    # K=4 mean accepted is 2.5; reach is 1, .5, .5*.25.
    assert reach == pytest.approx((1.0, 0.5, 0.125))
    assert committed == pytest.approx(3.5 * 1.625)
    assert source == "empirical_conditional_full_tile_rates"


def test_stationary_histogram_full_fraction_is_disclosed():
    module = load_module()
    committed, reach, source = module.expected_serial_commits(
        4, 3, evidence(module, explicit_rates=False)
    )
    assert reach == pytest.approx((1.0, 0.5, 0.25))
    assert committed == pytest.approx(3.5 * 1.75)
    assert source == "histogram_full_fraction_stationary_reuse"


def test_serial_keeps_reach_weighted_work_and_only_saves_boundaries():
    module = load_module()
    row = module.serial_cell(
        4, 3, module.audit_source(ROOT), 16, evidence(module), costs(module)
    )
    assert row["expected_draft_plus_verify_ms"] == pytest.approx(48.75)
    assert row["separate_rounds_expected_ms"] == pytest.approx(52.0)
    assert row["staged_serial_expected_ms"] == pytest.approx(50.75)
    assert row["scheduler_boundary_savings_ms"] == pytest.approx(1.25)
    assert row["savings_source"] == "scheduler_boundary_only"


def test_progressive_geometry_uses_one_long_proposal_and_stops_after_miss():
    module = load_module()
    histogram = {0: 4, 3: 3, 7: 1}
    tile3 = module.progressive_geometry(15, 3, histogram)
    tile4 = module.progressive_geometry(15, 4, histogram)
    # Needed rows are accepted+1, rounded to the target verification tile.
    assert tile3["expected_target_rows"] == pytest.approx((3 * 4 + 6 * 3 + 9) / 8)
    assert tile3["expected_target_launches"] == pytest.approx((1 * 4 + 2 * 3 + 3) / 8)
    assert tile4["expected_target_rows"] == pytest.approx((4 * 4 + 4 * 3 + 8) / 8)
    assert tile4["expected_target_launches"] == pytest.approx((1 * 7 + 2) / 8)
    assert tile3["fixed_target_rows"] == 16


def test_progressive_candidate_requires_shape_sensitive_parity():
    module = load_module()
    row = module.progressive_cell(
        3, module.audit_source(ROOT), 16, evidence(module)
    )
    assert row["dflash_calls"] == 1
    assert row["redrafts_after_partial_verify"] == 0
    assert row["shape_sensitive_parity_gate"]["required"] is True
    assert row["shape_sensitive_parity_gate"]["state"] == "unverified"
    assert row["candidate_state"]["implemented"] is False


def test_progressive_ladder_includes_fixed_k15_control():
    module = load_module()
    plan = module.build_plan(
        root=ROOT, artifact_block_size=16, evidence=evidence(module)
    )
    assert [row["verify_tile"] for row in plan["progressive_same_proposal"]] == [
        2, 3, 4, 5, 8, 15
    ]
    control = plan["progressive_same_proposal"][-1]
    assert control["fixed_control"] is True
    assert control["candidate_state"]["implemented"] is True
    assert control["decision"] == "fixed_native_control"
    assert control["shape_sensitive_parity_gate"]["required"] is False
    assert control["expected_target_rows"] == 16
    assert control["expected_target_launches"] == 1


def test_eager_frontier_stays_refused_and_may_waste_work():
    module = load_module()
    row = module.eager_cell(4, 2, 4, module.audit_source(ROOT), 16)
    assert row["candidate_state"]["exact_state_available"] is False
    assert row["may_waste_work"] is True
    assert row["decision"] == "refused_exact_uncommitted_frontier_api_missing"


def test_invalid_histogram_and_cost_without_evidence_fail_closed():
    module = load_module()
    bad = module.EmpiricalAcceptance(
        accepted_count_histograms={4: {5: 1}, 8: {0: 1}, 12: {0: 1}, 15: {0: 1}}
    )
    with pytest.raises(ValueError, match="invalid bin"):
        bad.validate()
    with pytest.raises(ValueError, match="requires empirical"):
        module.build_plan(root=ROOT, artifact_block_size=16, costs=costs(module))


def test_k15_requires_an_explicit_artifact_block_of_at_least_16():
    module = load_module()
    unknown = module.build_plan(root=ROOT)
    assert unknown["artifact_observation"]["k15_legal"] is None
    assert unknown["native_monolithic"][-1]["decision"] == (
        "requires_observed_artifact_block_size"
    )
    too_short = module.build_plan(root=ROOT, artifact_block_size=15)
    assert too_short["artifact_observation"]["k15_legal"] is False
    assert too_short["native_monolithic"][-1]["decision"] == (
        "refused_artifact_trained_block_too_short"
    )


def test_cli_refuses_required_eager_frontier(capsys):
    module = load_module()
    assert module.main(["--artifact-block-size", "16", "--require-eager-frontier"]) == 2
    assert '"eager_frontier_exact_state_ready": false' in capsys.readouterr().out


def test_cli_accepts_empirical_histograms_rates_and_costs(capsys):
    module = load_module()
    args = []
    for k in module.K_VALUES:
        args.extend(["--histogram", f"{k}=0:1,{k}:1"])
        args.extend(["--full-tile-rates", f"{k}=0.5,0.25"])
        args.extend(["--draft-ms", f"{k}=10"])
        args.extend(["--verify-ms", f"{k}=20"])
    args.extend(["--scheduler-boundary-ms", "2", "--artifact-block-size", "16"])
    assert module.main(args) == 0
    output = capsys.readouterr().out
    assert '"per_token_independence_assumed": false' in output
    assert '"savings_source": "scheduler_boundary_only"' in output
    assert '"progressive_target_rows"' in output
