import copy
import hashlib
import json
from typing import ClassVar

import pytest

from mlx2.adaptive_profile import (
    PostLoadAutotuner,
    bind_live_profile,
    live_telemetry_evidence,
    live_tuning_fingerprint,
    load_receipt,
    qualification_evidence,
    tuning_fingerprint,
    validate_receipt,
    write_receipt,
)


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _qualification(name):
    return {
        "qualification_receipt_sha256": _sha(name + "-receipt"),
        "runtime_sha256": _sha(name + "-runtime"),
        "settings_sha256": _sha(name + "-settings"),
    }


def _fingerprint(**shape):
    return tuning_fingerprint(
        artifact={"target": "a", "draft": "b"},
        topology={"kind": "external", "layout": "hybrid"},
        platform={"chip": "fixture", "memory_bytes": 128 << 30},
        serving_shape={"lanes": 4, "cache_bytes": 4 << 30, **shape},
        telemetry={"power": True, "thermal": True},
    )


def _candidate(name, policy, *, qualified=True):
    return {
        "id": name,
        "policy": policy,
        "qualification": _qualification(name) if qualified else None,
    }


def _sample(*, watts=40, state="nominal", age=0.1, complete=True):
    return {
        "power_complete": complete,
        "power_sample_age_seconds": age,
        "thermal_state": state,
        "thermal_sample_age_seconds": age,
        "gpu_dram_watts": watts,
    }


def _tuner(**kwargs):
    return PostLoadAutotuner(
        _fingerprint(),
        [
            _candidate("baseline", {"branches": 1}),
            _candidate("ast", {"branches": 4}),
            _candidate("probe", {"branches": 3}, qualified=False),
        ],
        baseline_id="baseline",
        **kwargs,
    )


def _observe(tuner, name, rate, *, energy=4, watts=40, **kwargs):
    tuner.observe(
        name,
        correct=True,
        tokens_per_second=rate,
        joules_per_token=energy,
        telemetry=_sample(watts=watts),
        **kwargs,
    )


def test_throughput_selects_fastest_qualified_profile_and_persists_candidates():
    tuner = _tuner(minimum_samples=2)
    for rate in (100, 102):
        _observe(tuner, "baseline", rate, energy=4.1)
    for energy in (3.1, 3.3):
        _observe(tuner, "ast", 110, energy=energy)
    _observe(tuner, "probe", 999, energy=1)
    receipt = tuner.finalize()
    assert receipt["selected_profile_id"] == "ast"
    assert receipt["selected_policy"] == {"branches": 4}
    assert receipt["selection_valid"] is True
    assert receipt["summaries"]["probe"]["qualified"] is False
    assert {row["id"] for row in receipt["candidates"]} == {"baseline", "ast", "probe"}


def test_efficiency_and_power_budget_are_explicit_objectives():
    efficient = _tuner(objective="efficiency", max_throughput_regression=0.1)
    _observe(efficient, "baseline", 100, energy=4, watts=35)
    _observe(efficient, "ast", 95, energy=2, watts=50)
    assert efficient.finalize()["selected_profile_id"] == "ast"

    budget = _tuner(
        objective="power_budget",
        power_budget_watts=40,
        max_throughput_regression=0.1,
    )
    _observe(budget, "baseline", 100, energy=4, watts=35)
    _observe(budget, "ast", 120, energy=3, watts=50)
    receipt = budget.finalize()
    assert receipt["selected_profile_id"] == "baseline"
    assert receipt["selection_reason"] == (
        "fastest_no_regression_profile_within_power_budget"
    )


def test_failed_and_refused_cells_allow_null_performance_and_remain_auditable():
    tuner = _tuner()
    _observe(tuner, "baseline", 100)
    tuner.observe(
        "ast",
        outcome="failed",
        failure_reason="decode fault",
        telemetry=None,
    )
    tuner.observe(
        "probe",
        outcome="refused",
        failure_reason="capability unavailable",
        telemetry=None,
    )
    receipt = tuner.finalize()
    assert receipt["selected_profile_id"] == "baseline"
    assert receipt["observations"][1]["tokens_per_second"] is None
    assert "failed" in receipt["observations"][1]["invalid_reasons"]
    assert validate_receipt(receipt) == receipt


def test_stale_incomplete_hot_and_contaminated_samples_fail_closed_but_do_not_poison_clean_epoch():
    tuner = _tuner(max_sample_age_seconds=1)
    _observe(tuner, "baseline", 100)
    for telemetry, contaminated in (
        (_sample(age=2), False),
        (_sample(complete=False), False),
        (_sample(state="fair"), False),
        (_sample(), True),
    ):
        tuner.observe(
            "ast",
            correct=True,
            tokens_per_second=200,
            joules_per_token=1,
            telemetry=telemetry,
            contaminated=contaminated,
        )
    _observe(tuner, "ast", 110, energy=3)
    receipt = tuner.finalize()
    assert receipt["summaries"]["ast"]["valid_samples"] == 1
    assert receipt["selected_profile_id"] == "ast"
    assert receipt["thermally_controlled"] is True


def test_receipt_recomputes_derived_fields_and_rejects_tampering(tmp_path):
    tuner = _tuner()
    _observe(tuner, "baseline", 100)
    receipt = tuner.finalize()
    path = tmp_path / "autotune.json"
    write_receipt(path, receipt)
    assert load_receipt(path, _fingerprint()) == receipt
    assert load_receipt(path, _fingerprint(lanes=1)) is None
    assert json.loads(path.read_text())["schema"] == "mlx2.adaptive-profile.v2"

    tampered = copy.deepcopy(receipt)
    tampered["summaries"]["baseline"]["tokens_per_second_median"] = 999
    with pytest.raises(ValueError, match="integrity"):
        validate_receipt(tampered)
    tampered["integrity_sha256"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in tampered.items()
                if key != "integrity_sha256"
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="derived"):
        validate_receipt(tampered)


def test_candidate_qualification_and_fingerprint_digest_validation():
    with pytest.raises(ValueError, match="baseline must be qualified"):
        PostLoadAutotuner(
            _fingerprint(), [_candidate("b", {}, qualified=False)], baseline_id="b"
        )
    broken = _fingerprint()
    broken["serving_shape"]["lanes"] = 9
    with pytest.raises(ValueError, match="fingerprint digest"):
        PostLoadAutotuner(broken, [_candidate("b", {})], baseline_id="b")
    value = _candidate("b", {})
    value["qualification"]["runtime_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="lowercase sha256"):
        PostLoadAutotuner(_fingerprint(), [value], baseline_id="b")


class _Descriptor:
    capabilities: ClassVar = frozenset({"external_draft", "apcv2"})


class _Adapter:
    layout = "hybrid"
    descriptor = _Descriptor()
    identity: ClassVar = {
        "fingerprint": "route-sha",
        "target_fingerprint": "target-sha",
        "draft_fingerprint": "draft-sha",
        "sidecar_fingerprints": {"projection": "sidecar-sha"},
    }

    def adaptive_profile_topology(self):
        return {"attention": "hybrid", "routed_moe": True}


def _live_fingerprint(**shape):
    return live_tuning_fingerprint(
        adapter=_Adapter(),
        runtime={"source_sha256": "source", "mlx": "fixture"},
        settings={
            "route": "external_draft",
            "speculation": "external_draft",
            "process_env": {"MLX2_FIXTURE": "1"},
        },
        serving_shape={
            "max_lanes": 4,
            "max_inflight": 8,
            "cache_bytes": 123,
            "cache_persistence": True,
            "coalesce_window_ms": 2,
            "prefill_step": 512,
            "power_governor": {"mode": "off"},
            **shape,
        },
        telemetry={"power": {"enabled": True}, "thermal": True},
        memory_bytes=128 << 30,
        hardware={"machdep.cpu.brand_string": "Apple Fixture Max", "gpu_cores": 40},
    )


def test_live_fingerprint_covers_artifacts_capability_topology_hardware_and_shape():
    value = _live_fingerprint()
    assert value["artifact"] == {
        "route": "route-sha",
        "target": "target-sha",
        "draft": "draft-sha",
        "sidecars": {"projection": "sidecar-sha"},
    }
    assert value["topology"]["declared"]["routed_moe"] is True
    assert value["platform"]["hardware"]["gpu_cores"] == 40
    for changed in (
        _live_fingerprint(max_lanes=1),
        live_tuning_fingerprint(
            adapter=_Adapter(),
            runtime={"source_sha256": "source", "mlx": "fixture"},
            settings={"route": "external_draft", "speculation": "external_draft"},
            serving_shape={"max_lanes": 4},
            telemetry={"power": True},
            memory_bytes=128 << 30,
            hardware={"machdep.cpu.brand_string": "Apple Other"},
        ),
    ):
        assert changed["sha256"] != value["sha256"]


def _live_status(*, age=0.1, state="nominal", complete=True):
    return {
        "available": True,
        "last_sample_age_seconds": age,
        "last": {
            "watts": {"gpu": 30, "dram": 5},
            "system_thermal_state": state,
        },
        "governor_signal": {"complete": complete},
    }


def test_live_telemetry_requires_fresh_complete_nominal_sample():
    assert (
        live_telemetry_evidence(_live_status(), max_sample_age_seconds=1)[
            "gpu_dram_watts"
        ]
        == 35
    )
    for status, message in (
        (_live_status(age=2), "stale"),
        (_live_status(state="serious"), "thermal_not_nominal"),
        (_live_status(complete=False), "power_incomplete"),
        ({"available": False}, "available power telemetry"),
    ):
        with pytest.raises(ValueError, match=message):
            live_telemetry_evidence(status, max_sample_age_seconds=1)


def test_qualification_evidence_hashes_exact_file_runtime_and_settings(tmp_path):
    path = tmp_path / "qualification.json"
    path.write_text('{"qualified":true}\n')
    first = qualification_evidence(path, runtime={"source": "a"}, settings={"lanes": 4})
    second = qualification_evidence(
        path, runtime={"source": "a"}, settings={"lanes": 1}
    )
    assert (
        first["qualification_receipt_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert first["settings_sha256"] != second["settings_sha256"]


def test_live_reuse_requires_exact_qualification_policy_and_current_telemetry(tmp_path):
    policy = {"adaptive_speculation_topology": {"enabled": True, "branches": 4}}
    fingerprint = _live_fingerprint()
    binding = _qualification("live")
    tuner = PostLoadAutotuner(
        fingerprint,
        [
            {
                "id": "selected",
                "policy": {"execution_policy": policy},
                "qualification": binding,
            }
        ],
        baseline_id="selected",
    )
    tuner.observe(
        "selected",
        correct=True,
        tokens_per_second=100,
        joules_per_token=2,
        telemetry=_sample(watts=35),
    )
    path = tmp_path / "adaptive.json"
    write_receipt(path, tuner.finalize())
    result = bind_live_profile(
        path,
        fingerprint,
        active_execution_policy=policy,
        qualification_state="qualified",
        qualification_binding=binding,
        telemetry_status=_live_status(),
    )
    assert result["selected"] is True
    assert result["defaulted"] is False
    assert result["observed_used"] is False

    with pytest.raises(ValueError, match="qualification binding"):
        bind_live_profile(
            path,
            fingerprint,
            active_execution_policy=policy,
            qualification_state="qualified",
            qualification_binding=_qualification("other"),
            telemetry_status=_live_status(),
        )
    with pytest.raises(ValueError, match="not active"):
        bind_live_profile(
            path,
            fingerprint,
            active_execution_policy={},
            qualification_state="qualified",
            qualification_binding=binding,
            telemetry_status=_live_status(),
        )


def test_server_cli_forwards_opt_in_receipt_and_constructor_requires_qualification():
    from mlx2.server import build_parser, serving_engine_kwargs
    from mlx2.serving import ServingEngine

    parser = build_parser()
    args = parser.parse_args(
        [
            "--model",
            "fixture",
            "--qualification",
            "qualified.json",
            "--adaptive-profile-receipt",
            "adaptive.json",
        ]
    )
    kwargs = serving_engine_kwargs(
        args,
        {},
        native_mtp=False,
        approximate_kv=None,
        max_request_bytes=2 << 20,
    )
    assert str(kwargs["adaptive_profile_receipt"]) == "adaptive.json"
    with pytest.raises(ValueError, match="requires a matching route qualification"):
        ServingEngine.validate_arguments(
            "fixture", adaptive_profile_receipt="adaptive.json"
        )
