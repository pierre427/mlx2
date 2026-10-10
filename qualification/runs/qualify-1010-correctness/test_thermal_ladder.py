from types import SimpleNamespace

import pytest
from thermal_ladder import foreign_snapshot, validate_serving_profile

PROFILE = {
    "cache_bytes": 4 << 30,
    "max_lanes": 4,
    "max_inflight": 8,
    "prefill_step": 2048,
    "apc_persistence": False,
    "apc_persist_on_shutdown": False,
    "mtp": {"enabled": True, "mode": "self"},
    "draft_loop": "auto",
    "draft_loop_threshold": 0.2,
    "draft_loop_widths": "1,2,4",
    "prefill_depth_budget": 4096,
    "prefill_step_autoscale": True,
}


def validate(settings=PROFILE):
    return validate_serving_profile(
        {
            "settings": settings,
            "healthy": True,
            "host": "h",
            "artifact": "a",
            "max_context": 8192,
            "route_receipt": {"route": "self_mtp"},
        },
        expected_route="self_mtp",
        expected_artifact="a",
        max_context=8192,
        cache_bytes=4 << 30,
        max_lanes=4,
        max_inflight=8,
        prefill_step=2048,
        apc_persistence=False,
        apc_persist_on_shutdown=False,
        mtp_policy={"enabled": True, "mode": "self"},
        draft_loop_policy={
            "draft_loop": "auto",
            "draft_loop_threshold": 0.2,
            "draft_loop_widths": "1,2,4",
        },
        prefill_policy={"prefill_depth_budget": 4096, "prefill_step_autoscale": True},
    )


def test_profile_binds_live_budget_concurrency_and_route_settings():
    assert validate()["cache_bytes"] == 4 << 30


def test_profile_rejects_live_setting_drift():
    changed = {**PROFILE, "max_lanes": 2}
    with pytest.raises(ValueError, match="profile mismatch"):
        validate(changed)


def test_profile_rejects_missing_live_settings():
    with pytest.raises(TypeError, match="missing serving settings"):
        validate_serving_profile(
            {},
            expected_route="r",
            expected_artifact="a",
            max_context=1,
            cache_bytes=1,
            max_lanes=1,
            max_inflight=1,
            prefill_step=1,
            apc_persistence=False,
            apc_persist_on_shutdown=False,
            mtp_policy={},
            draft_loop_policy={},
            prefill_policy={},
        )


def test_profile_rejects_wrong_route_artifact_or_context():
    status = {
        "settings": PROFILE,
        "route_receipt": {"route": "self_mtp"},
        "artifact": "a",
        "max_context": 8192,
    }
    with pytest.raises(ValueError, match="route/artifact/context mismatch"):
        validate_serving_profile(
            {**status, "artifact": "other"},
            expected_route="self_mtp",
            expected_artifact="a",
            max_context=8192,
            cache_bytes=4 << 30,
            max_lanes=4,
            max_inflight=8,
            prefill_step=2048,
            apc_persistence=False,
            apc_persist_on_shutdown=False,
            mtp_policy={"enabled": True, "mode": "self"},
            draft_loop_policy={
                "draft_loop": "auto",
                "draft_loop_threshold": 0.2,
                "draft_loop_widths": "1,2,4",
            },
            prefill_policy={
                "prefill_depth_budget": 4096,
                "prefill_step_autoscale": True,
            },
        )


def test_foreign_snapshot_matches_decision_server_only_at_module_boundary(monkeypatch):
    rows = "\n".join(
        (
            "101 00:00.01 python -m mlx2.decisions.server",
            "102 00:00.02 python -m mlx2.top.server",
            "103 00:00.03 python -m mlx2.server_helpers",
            "104 00:00.04 python -m mlx2.decisions.server",
            "105 00:00.05 python -m mlx2.server",
        )
    )

    def fake_ps(*args, **kwargs):
        assert args[0] == ["/bin/ps", "-Ao", "pid=,time=,command="]
        return SimpleNamespace(stdout=rows)

    monkeypatch.setattr("thermal_ladder.subprocess.run", fake_ps)
    snapshot = foreign_snapshot({104})

    assert set(snapshot) == {101, 105}
    assert snapshot[101]["command"].endswith("mlx2.decisions.server")


@pytest.mark.parametrize("shutdown", [False, True])
def test_cli_binds_shutdown_policy_before_requests(monkeypatch, tmp_path, shutdown):
    import json
    import thermal_ladder

    class StopBeforeRequests(Exception):
        pass

    settings = {**PROFILE, "mtp": True, "apc_persist_on_shutdown": shutdown}

    class StatusOnly:
        def __init__(self, *args):
            pass

        def status(self):
            return {
                "settings": settings, "artifact": "a", "max_context": 8192,
                "route_receipt": {"route": "self_mtp"},
            }

    original = thermal_ladder.validate_serving_profile

    def stop_after_validation(status, **kwargs):
        # Exercise the actual CLI caller and required-keyword contract, then
        # stop before thermal admission, tokenization or inference requests.
        result = original(status, **kwargs)
        assert result["apc_persist_on_shutdown"] is shutdown
        raise StopBeforeRequests

    monkeypatch.setattr(thermal_ladder, "Stream", StatusOnly)
    monkeypatch.setattr(
        thermal_ladder, "validate_serving_profile", stop_after_validation
    )
    args = [
        "--url", "http://127.0.0.1:1", "--output", str(tmp_path / "unused.json"),
        "--model", "test", "--model-id", "test", "--route", "self_mtp",
        "--artifact-identity", "a", "--max-context", "8192",
        "--cache-bytes", str(4 << 30), "--max-lanes", "4", "--max-inflight", "8",
        "--prefill-step", "2048", "--apc-persistence", "off",
        "--apc-persist-on-shutdown", "on" if shutdown else "off",
        "--mtp-policy", "true", "--server-pid", "123",
        "--draft-loop-policy", json.dumps({key: PROFILE[key] for key in (
            "draft_loop", "draft_loop_threshold", "draft_loop_widths"
        )}),
        "--prefill-policy", json.dumps({key: PROFILE[key] for key in (
            "prefill_depth_budget", "prefill_step_autoscale"
        )}),
    ]
    with pytest.raises(StopBeforeRequests):
        thermal_ladder.main(args)
