from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
        "execution": {"layout": "layout-a"},
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
        "execution": {"layout": "layout-a"},
    }
    import pytest

    with pytest.raises(RuntimeError, match="ordinary non-MTP"):
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
