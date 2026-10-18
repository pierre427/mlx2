from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "single_model_ladder", ROOT / "scripts/run_single_model_smoke_ladder.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_validate_ownership_requires_matching_receipts(monkeypatch, tmp_path):
    locks = (tmp_path / "shared", tmp_path / "tmp")
    owner = {"session": "session-a", "label": "model-a", "lease_id": "lease-a"}
    for lock in locks:
        lock.mkdir()
        (lock / "owner.json").write_text(json.dumps(owner))
    monkeypatch.setattr(MODULE, "LOCKS", locks)
    assert MODULE.validate_ownership("session-a", "model-a") == owner


def test_campaign_ownership_requires_cpg_and_paired_receipts(monkeypatch, tmp_path):
    locks = (tmp_path / "shared", tmp_path / "tmp")
    paired = {"session": "session-a", "label": "model-a", "lease_id": "lease-a"}
    for lock in locks:
        lock.mkdir()
        (lock / "owner.json").write_text(json.dumps(paired))
    cpg_lock = tmp_path / "cpg"
    cpg_lock.mkdir()
    seen = {}

    def require_qualification_lease(**kwargs):
        seen.update(kwargs)
        return {"radio_claim": {"claimed": True}}

    monkeypatch.setattr(MODULE, "LOCKS", locks)
    monkeypatch.setitem(
        sys.modules,
        "qualification_gpu_ownership",
        SimpleNamespace(require_qualification_lease=require_qualification_lease),
    )
    receipt = MODULE.validate_campaign_ownership(
        "session-a",
        "model-a",
        cpg_task="task-a",
        cpg_owner_lock=cpg_lock,
        cpg_generation=7,
    )
    assert receipt["paired_lock_owner"] == paired
    assert receipt["radio_claim"] == {"claimed": True}
    assert seen == {
        "task_id": "task-a",
        "cpg_owner_lock": cpg_lock,
        "generation": 7,
    }


def test_campaign_locks_still_owned_compares_only_paired_receipt(monkeypatch):
    paired = {"session": "s", "label": "l", "pid": 7}
    monkeypatch.setattr(MODULE, "validate_ownership", lambda session, label: paired)
    combined = {
        "paired_lock_owner": paired,
        "radio_claim": {"claimed": True},
        "cpg_owner": {"cpg_generation": 3},
    }
    assert MODULE.campaign_locks_still_owned("s", "l", combined) is True
    assert MODULE.campaign_locks_still_owned("s", "l", {}) is False


def test_thermal_sample_requires_structured_state(monkeypatch):
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout='{"thermal_state": 1}'),
    )
    assert MODULE.thermal_sample()["thermal_state"] == 1


def test_stop_server_is_noop_for_completed_process():
    class Completed:
        def poll(self):
            return 0

    MODULE.stop_server(Completed())


def test_validate_startup_status_binds_ordinary_candidate_identity():
    status = {
        "artifact": "artifact-a",
        "profile": "profile-a",
        "qualification": "candidate",
        "settings": {"route": "ordinary", "mtp": False},
        "execution": {"cache_layout": "layout-a"},
    }
    MODULE.validate_startup_status(
        status,
        expected_artifact="artifact-a",
        expected_profile="profile-a",
        expected_cache_layout="layout-a",
    )


def test_validate_startup_status_rejects_wrong_route():
    status = {
        "artifact": "artifact-a",
        "profile": "profile-a",
        "qualification": "candidate",
        "settings": {"route": "native_mtp", "mtp": True},
        "execution": {"cache_layout": "layout-a"},
    }
    import pytest

    with pytest.raises(RuntimeError, match="ordinary non-MTP"):
        MODULE.validate_startup_status(
            status,
            expected_artifact="artifact-a",
            expected_profile="profile-a",
            expected_cache_layout="layout-a",
        )


def test_validate_startup_status_accepts_external_draft_candidate_identity():
    status = {
        "artifact": "artifact-a",
        "profile": "profile-a",
        "qualification": "candidate",
        "settings": {"route": "external_draft", "mtp": False},
        "execution": {"cache_layout": "layout-a"},
    }
    MODULE.validate_startup_status(
        status,
        expected_route="external_draft",
        expected_artifact="artifact-a",
        expected_profile="profile-a",
        expected_cache_layout="layout-a",
    )


def test_validate_startup_status_accepts_legacy_layout_key():
    status = {
        "artifact": "artifact-a",
        "profile": "profile-a",
        "qualification": "candidate",
        "settings": {"route": "ordinary", "mtp": False},
        "execution": {"layout": "layout-a"},
    }
    MODULE.validate_startup_status(
        status,
        expected_artifact="artifact-a",
        expected_profile="profile-a",
        expected_cache_layout="layout-a",
    )


def test_ladder_semantics_keeps_exploratory_and_qualification_modes_distinct():
    assert "exploratory" in MODULE.ladder_semantics(
        qualification_ladder=False, runs=1
    )
    assert "qualification-ladder evidence" in MODULE.ladder_semantics(
        qualification_ladder=True, runs=3
    )


def test_ladder_semantics_rejects_wrong_replication_for_each_mode():
    import pytest

    with pytest.raises(ValueError, match="three runs"):
        MODULE.ladder_semantics(qualification_ladder=True, runs=1)
    with pytest.raises(ValueError, match="one run"):
        MODULE.ladder_semantics(qualification_ladder=False, runs=3)


def test_server_capacity_preserves_default_and_allows_historical_headroom():
    assert MODULE.server_capacity(wide=4, max_inflight=None) == (4, 4)
    assert MODULE.server_capacity(wide=4, max_inflight=8) == (4, 8)


def test_server_capacity_rejects_less_inflight_than_width():
    import pytest

    with pytest.raises(ValueError, match="at least"):
        MODULE.server_capacity(wide=4, max_inflight=3)


def test_external_policy_binds_fidelity_and_ast_shape(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy = {
        "draft_model": "/draft",
        "progressive_verification_fidelity": "near_tie_bounded",
        "progressive_verification_tile": 3,
        "progressive_alternative_branches": 4,
        "progressive_alternative_strategy": "confidence_tree",
        "progressive_cross_session_pack_rows": 8,
        "progressive_cross_session_pack_mode": "ordered_live_frontier",
    }
    policy_path.write_text(json.dumps(policy))
    assert MODULE.load_execution_policy(
        policy_path,
        route="external_draft",
        expected_fidelity="near_tie_bounded",
        require_ast=True,
    ) == policy


def test_external_policy_rejects_ast_cap_that_cannot_reach_progressive_tile(
    tmp_path,
):
    import pytest

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "draft_model": "/draft",
                "progressive_verification_tile": 3,
                "progressive_multilane_draft_cap": 3,
                "progressive_alternative_branches": 4,
                "progressive_alternative_strategy": "confidence_tree",
                "progressive_cross_session_pack_rows": 8,
                "progressive_cross_session_pack_mode": "ordered_live_frontier",
            }
        )
    )
    with pytest.raises(ValueError, match="multilane_draft_cap to exceed"):
        MODULE.load_execution_policy(
            policy_path,
            route="external_draft",
            expected_fidelity="exact",
            require_ast=True,
        )


def test_external_policy_rejects_wrong_fidelity_and_nonconfidence_ast(tmp_path):
    import pytest

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "draft_model": "/draft",
                "progressive_verification_tile": 3,
                "progressive_alternative_branches": 4,
                "progressive_alternative_strategy": "distinct_root",
                "progressive_cross_session_pack_rows": 8,
                "progressive_cross_session_pack_mode": "ordered_live_frontier",
            }
        )
    )
    with pytest.raises(ValueError, match="fidelity"):
        MODULE.load_execution_policy(
            policy_path,
            route="external_draft",
            expected_fidelity="near_tie_bounded",
            require_ast=True,
        )
    with pytest.raises(ValueError, match="confidence_tree"):
        MODULE.load_execution_policy(
            policy_path,
            route="external_draft",
            expected_fidelity="exact",
            require_ast=True,
        )


def _ast_report(*, fidelity="exact", observed=True, observation_failures=0):
    counters = {
        "external_progressive_verify_max_cohort_width": 4,
        "external_progressive_alternative_rounds": 8,
        "external_progressive_alternative_branches": 32,
        "external_progressive_cross_session_ast_rounds": 4,
        "external_progressive_cross_session_ast_sessions": 16,
        "external_progressive_cross_session_ast_packs": 8,
        "external_progressive_ast_frontier_rows_admitted": 32,
        "external_progressive_topology_observation_failures": observation_failures,
    }
    speculation = {
        "progressive_verification": {
            "fidelity": fidelity,
            "observed_used": True,
            "ordered_alternatives": {
                "mechanism_id": "adaptive_speculation_topology",
                "configured_branches": 4,
                "rounds": 2,
                "cross_session_ast_packing": {
                    "configured_pack_rows": 8,
                    "configured_mode": "ordered_live_frontier",
                    "mode": "ordered_live_frontier",
                    "observed_used": observed,
                },
            },
        }
    }
    response = {"receipt": {"speculation": speculation}}
    request = {"rows": [response]}
    run = {"requests": {"cold": request, "warm": request}}
    return {
        "final": {"scheduler": counters},
        "cells": [{"warmup": run, "runs": [run]}],
    }


def test_ast_report_requires_counters_fidelity_and_observed_receipt():
    receipt = MODULE.validate_ast_report(
        _ast_report(),
        expected_fidelity="exact",
        expected_width=4,
        expected_branches=4,
        expected_pack_rows=8,
    )
    assert receipt["passed"] is True
    assert receipt["matched_response_receipts"] == 4

    failed = MODULE.validate_ast_report(
        _ast_report(
            fidelity="near_tie_bounded",
            observed=False,
            observation_failures=1,
        ),
        expected_fidelity="exact",
        expected_width=4,
        expected_branches=4,
        expected_pack_rows=8,
    )
    assert failed["passed"] is False
    assert any("observation_failures" in item for item in failed["failures"])
    assert any("no response receipt" in item for item in failed["failures"])

    wrong_mode_report = _ast_report()
    packing = wrong_mode_report["cells"][0]["warmup"]["requests"]["cold"][
        "rows"
    ][0]["receipt"]["speculation"]["progressive_verification"][
        "ordered_alternatives"
    ]["cross_session_ast_packing"]
    packing["mode"] = "eager_all_branches"
    wrong_mode = MODULE.validate_ast_report(
        wrong_mode_report,
        expected_fidelity="exact",
        expected_width=4,
        expected_branches=4,
        expected_pack_rows=8,
    )
    assert wrong_mode["passed"] is False
    assert "AST receipt has the wrong observed packing mode" in wrong_mode["failures"]

    wrong_config_report = _ast_report()
    packing = wrong_config_report["cells"][0]["warmup"]["requests"]["cold"][
        "rows"
    ][0]["receipt"]["speculation"]["progressive_verification"][
        "ordered_alternatives"
    ]["cross_session_ast_packing"]
    packing["configured_mode"] = None
    wrong_config = MODULE.validate_ast_report(
        wrong_config_report,
        expected_fidelity="exact",
        expected_width=4,
        expected_branches=4,
        expected_pack_rows=8,
    )
    assert wrong_config["passed"] is False
    assert (
        "AST receipt has the wrong configured packing mode"
        in wrong_config["failures"]
    )
