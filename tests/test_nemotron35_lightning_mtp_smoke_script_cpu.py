from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_nemotron35_lightning_mtp_smoke.py"


def load_script():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(
            "run_nemotron35_lightning_mtp_smoke", SCRIPT
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ROOT / "scripts"))


def test_mtp_receipt_gate_requires_exact_serial_mechanism():
    smoke = load_script()
    response = {
        "mlx2": {
            "route": "native_mtp",
            "route_selection_source": "explicit_flag",
            "ordinary_compute_width": None,
            "mtp": {
                "route": "segmented_self_mtp",
                "verification": "exact",
                "num_draft": 2,
                "observed_compute_widths": [1],
                "stats": {"draft_cycles": 3, "draft_proposed": 6},
            },
        }
    }
    assert smoke.mtp_receipt_errors(response) == []
    response["mlx2"]["mtp"]["observed_compute_widths"] = [2]
    assert "mtp.observed_compute_widths" in smoke.mtp_receipt_errors(response)


def test_mechanism_gate_requires_positive_serial_deltas_and_no_forbidden_delta():
    smoke = load_script()
    before = {name: 0 for name in (
        "requests", "engaged", "b1_target_forwards", "b1_draft_forwards",
        "transaction_branches", "committed_cycles", "failures",
        "full_prefix_materializations", "physical_b2_formations",
        "true_batched_engaged",
    )}
    after = {**before, **{name: 1 for name in (
        "requests", "engaged", "b1_target_forwards", "b1_draft_forwards",
        "transaction_branches", "committed_cycles",
    )}}
    assert smoke.mechanism_errors(before, after) == []
    assert "no_positive_delta:engaged" in smoke.mechanism_errors(
        before, {**after, "engaged": 0}
    )
    assert "unexpected_delta:true_batched_engaged" in smoke.mechanism_errors(
        before, {**after, "true_batched_engaged": 1}
    )


def test_ordinary_reference_must_be_passed_and_contain_four_b1_responses(tmp_path):
    smoke = load_script()
    path = tmp_path / "ordinary.json"
    value = {
        "schema": smoke.REFERENCE_SCHEMA,
        "passed": True,
        "checks": {
            "b1_bn_token_parity": {
                "passed": True,
                "evidence": {"b1": [{"choices": []}] * 4},
            }
        },
    }
    path.write_text(json.dumps(value))
    assert len(smoke.load_ordinary_references(path)) == 4
    value["passed"] = False
    path.write_text(json.dumps(value))
    try:
        smoke.load_ordinary_references(path)
    except ValueError as error:
        assert "not a passed" in str(error)
    else:
        raise AssertionError("failed ordinary receipt must be rejected")


def test_server_command_is_fresh_single_lane_mtp2_and_always_cleaned_up():
    source = SCRIPT.read_text()
    server = source[source.index("server_command = ["):source.index("server = None")]
    assert '"--native-mtp"' in server
    assert '"--max-lanes", "1"' in server
    assert '"--max-inflight", "1"' in server
    assert '"--execution-policy"' in server
    assert 'common.atomic_json(policy_path, {"num_draft": 2})' in source
    assert "common.stop_server(server)" in source
    assert "common.validate_ownership(args.session, args.label)" in source
