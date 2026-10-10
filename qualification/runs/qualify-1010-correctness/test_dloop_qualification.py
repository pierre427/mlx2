import hashlib
import json

from dloop_qualification import audit_case, audit_manifest, dloop_producer_sha256

from scripts.dloop_ab import (
    _metadata_artifact_fingerprint,
    _validate_arms,
    _validate_identity_args,
    _validate_numeric_args,
)

DEPTHS = list(range(1, 9))


def state(prompt):
    prefix = [101, prompt + 1, 7]
    return {
        "schema": "mlx2.dloop-state-continuation.v1",
        "prefix_tokens": len(prefix),
        "prefix_token_ids": prefix,
        "prefix_sha256": hashlib.sha256(
            json.dumps(prefix, separators=(",", ":")).encode()
        ).hexdigest(),
        "target_cache_offsets": [
            {"index": 0, "type": "KVCache", "offset": 3},
            {"index": 1, "type": "LinearCache", "offset": None},
        ],
        "mtp_cache_offsets": [{"index": 0, "type": "KVCache", "offset": 2}],
        "saved_ordinary_tokens": [31, 32, 33],
        "saved_self_mtp_tokens": [31, 32, 33],
        "cold_ordinary_tokens": [31, 32, 33],
    }


def report(width=1):
    arms = {f"fixed{d}": {"num_draft": d} for d in DEPTHS}
    arms["loop8"] = {
        "num_draft": 8,
        "draft_loop": {"boundaries": DEPTHS, "threshold": -1e9, "cohort": "any"},
    }
    results = []
    for pair in range(3):
        for arm in arms:
            rows = []
            for prompt in range(3):
                row = {
                    "prompt": prompt,
                    "tokens": [prompt, 10, 11],
                    "route": "segmented_self_mtp",
                    "verify_span_hist": {"9": 1}
                    if arm in ("fixed8", "loop8")
                    else {str(arms[arm]["num_draft"] + 1): 1},
                    "state_continuation": state(prompt),
                }
                if arm == "loop8":
                    row["draft_loop"] = {
                        "observed_used": True,
                        "decisions": 2,
                        "extensions": 1,
                        "boundaries": DEPTHS,
                        "qualified": False,
                    }
                if arm == "fixed1":
                    row["draft_loop"] = None
                if arm == "loop8":
                    row["true_batched"] = {"true_batched_engaged": 1}
                rows.append(row)
            results.append({"pair": pair, "arm": arm, "rows": rows})
    return {
        "schema": "mlx2.dloop_ab.v1",
        "model": "test-27b",
        "host": "test-host",
        "source_revision": "a" * 40,
        "artifact_identity": "b" * 64,
        "runtime_identity": {
            "source_sha256": "c" * 64,
            "mlx_native_sha256": "d" * 64,
            "python": "3.12",
            "macos": "15.0",
            "mlx": "0.30",
            "transformers": "4.50",
            "dependencies": {},
        },
        "runtime_source_sha256": "c" * 64,
        "runtime_native_sha256": "d" * 64,
        "producer_sha256": dloop_producer_sha256(),
        "width": width,
        "pairs": 3,
        "state_oracle_tokens": 3,
        "arms": arms,
        "results": results,
        "summary": {"speedup_vs_baseline": 0.0},
    }


def test_all_widths_and_depths_prove_exact_tokens_state_and_max_span():
    for width in (1, 2, 4, 8):
        result = audit_case(width, report(width), 3)
        assert result["passed"] is True
        assert result["fixed8_max_verify_span"] == 9
        assert result["loop8_engaged_requests"] == 9


def test_speed_values_do_not_affect_feature_behavior_evidence():
    value = report()
    value["summary"]["speedup_vs_baseline"] = -999
    assert audit_case(1, value, 3)["passed"] is True


def test_token_mismatch_fails_even_if_summary_claims_pass():
    value = report()
    next(
        row
        for item in value["results"]
        if item["arm"] == "loop8"
        for row in item["rows"]
    )["tokens"] = [1, 2, 3]
    assert audit_case(1, value, 3)["passed"] is False


def test_saved_state_continuation_mismatch_fails():
    value = report()
    next(
        row
        for item in value["results"]
        if item["arm"] == "loop8"
        for row in item["rows"]
    )["state_continuation"]["saved_self_mtp_tokens"] = [9]
    assert audit_case(1, value, 3)["passed"] is False


def test_cold_continuation_mismatch_fails():
    value = report()
    next(
        row
        for item in value["results"]
        if item["arm"] == "loop8"
        for row in item["rows"]
    )["state_continuation"]["cold_ordinary_tokens"] = [8]
    assert audit_case(1, value, 3)["passed"] is False


def test_one_correctly_declined_extension_does_not_fail():
    value = report()
    next(
        row
        for item in value["results"]
        if item["arm"] == "loop8"
        for row in item["rows"]
    )["draft_loop"]["extensions"] = 0
    assert audit_case(1, value, 3)["passed"] is True


def test_no_max_depth_execution_fails():
    value = report()
    row = next(
        row
        for item in value["results"]
        if item["arm"] == "loop8"
        for row in item["rows"]
    )
    row["verify_span_hist"] = {"4": 10}
    assert audit_case(1, value, 3)["passed"] is False


def test_missing_counter_or_width_engagement_fails():
    value = report(8)
    row = next(
        row
        for item in value["results"]
        if item["arm"] == "loop8"
        for row in item["rows"]
    )
    row["draft_loop"]["extensions"] = "invalid"
    assert audit_case(8, value, 3)["passed"] is False
    value = report(8)
    row = next(
        row
        for item in value["results"]
        if item["arm"] == "loop8"
        for row in item["rows"]
    )
    row["true_batched"]["true_batched_engaged"] = True
    assert audit_case(8, value, 3)["passed"] is False


def test_manifest_requires_exact_width_coverage_and_hashes(tmp_path):
    cases = []
    for width in (1, 2, 4, 8):
        path = tmp_path / f"width-{width}.json"
        path.write_text(json.dumps(report(width)))
        cases.append(
            {
                "width": width,
                "report": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema": "mlx2.dloop-behavior-campaign.v1",
        "pairs": 3,
        "cases": cases,
        "host": "test-host",
        "model": "test-27b",
        "source_identity": "a" * 40,
        "artifact_identity": "b" * 64,
        "runtime_source_sha256": "c" * 64,
        "runtime_native_sha256": "d" * 64,
        "dloop_producer_sha256": dloop_producer_sha256(),
        "settings": {
            "fixed_depths": DEPTHS,
            "gate_boundaries": DEPTHS,
            "gate_threshold": -1e9,
            "cohort": "any",
            "non_dloop_control": "fixed1",
            "explicit_dloop_arm": "loop8",
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert audit_manifest(path)["evidence_passed"] is True
    manifest["cases"] = cases[:-1]
    path.write_text(json.dumps(manifest))
    assert audit_manifest(path)["evidence_passed"] is False


def test_report_identity_is_actual_and_matches_expected_manifest():
    value = report()
    assert audit_case(1, value, 3)["passed"] is True
    value["runtime_identity"]["source_sha256"] = "f" * 64
    assert audit_case(1, value, 3)["passed"] is False


def test_identity_cli_formats_are_distinct_and_validated():
    _validate_identity_args("a" * 40, "b" * 64, "c" * 64)
    import pytest

    with pytest.raises(ValueError, match="40-character"):
        _validate_identity_args("b" * 64, "b" * 64, "c" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        _validate_identity_args("a" * 40, "git-head", "c" * 64)


def test_arm_depths_are_validated_before_model_loading():
    _validate_arms([("fixed1", {"num_draft": 1})])
    import pytest

    with pytest.raises(ValueError, match="1..8"):
        _validate_arms([("fixed9", {"num_draft": 9})])
    with pytest.raises(ValueError, match="boundaries"):
        _validate_arms(
            [("loop8", {"num_draft": 8, "draft_loop": {"boundaries": [1, 9]}})]
        )


def test_dloop_numeric_arguments_reject_zero_tokens_before_model_load():
    _validate_numeric_args(1, 2, 0, 1)
    import pytest

    with pytest.raises(ValueError, match="at least 2"):
        _validate_numeric_args(1, 1, 0, 1)
    with pytest.raises(ValueError, match="positive integer"):
        _validate_numeric_args(0, 2, 0, 1)


def test_artifact_metadata_fingerprint_is_read_without_adapter_load():
    assert (
        _metadata_artifact_fingerprint({"identity": {"fingerprint": "a" * 64}})
        == "a" * 64
    )
    assert _metadata_artifact_fingerprint({"identity": None}) is None
    assert _metadata_artifact_fingerprint(None) is None
