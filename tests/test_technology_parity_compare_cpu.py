import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from technology_parity_compare import compare


@pytest.fixture
def receipt():
    rows = [{"index": i, "test_index": index, "text": "4", "finish_reason": "stop",
             "completion_tokens": 1, "accept_hist": {"2": 1}, "depth_hist": {"3": 1}}
            for i, index in enumerate([2, 7])]
    arm = {"passed": True, "rows": rows, "summary": {
        "routes": ['"external_draft"'], "acceptance": {"mean_acceptance_length": 3}}}
    return {"checkpoint": "fixture@sha256:123", "dataset": {"questions": 2, "indices": [2, 7]},
            "arms": [dict(copy.deepcopy(arm), name=f"{tree}-{route}", revision={"head": tree})
                     for route in ("default", "ordinary") for tree in ("old", "new")]}


def evaluate(tmp_path, receipt):
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    return compare(tmp_path, ["default", "ordinary"])


def test_exact_uses_gsm8k_summary_fields(tmp_path, receipt):
    result = evaluate(tmp_path, receipt)
    assert result["verdict"] == "exact"
    pair = result["pairs"][0]
    assert pair["routes"]["new"] == ['"external_draft"']
    assert pair["mean_acceptance_length"]["new"] == 3
    receipt["arms"][0]["rows"][0]["text"] = "5"
    result = evaluate(tmp_path, receipt)
    assert result["verdict"] == "mismatch"
    assert result["pairs"][0]["mismatches"][0]["first_text_difference"] == 0


@pytest.mark.parametrize("damage", ["partial", "orphan", "missing_pair", "duplicate_arm",
                                    "duplicate_row", "wrong_question", "missing_histogram", "missing_revision"])
def test_incomplete_evidence_cannot_establish_parity(tmp_path, receipt, damage):
    arms = receipt["arms"]
    if damage == "partial":
        for arm in arms:
            arm["rows"].pop()
            arm["stopped_at_deadline"] = True
    elif damage == "orphan":
        del arms[2]
    elif damage == "missing_pair":
        del arms[2:]
    elif damage == "duplicate_arm":
        arms.append(arms[-1])
    elif damage == "duplicate_row":
        arms[0]["rows"][0]["index"] = 1
    elif damage == "wrong_question":
        arms[0]["rows"][0]["test_index"] = 7
    elif damage == "missing_revision":
        del arms[0]["revision"]
    elif damage == "missing_histogram":
        del arms[0]["rows"][0]["accept_hist"]
    assert evaluate(tmp_path, receipt)["verdict"] == "incomplete"


def test_speculative_missing_and_zero_rounds_are_distinct(tmp_path, receipt):
    for arm in receipt["arms"]:
        row = arm["rows"][0]
        row.update(accept_hist=None, depth_hist=None)
    assert evaluate(tmp_path, receipt)["verdict"] == "incomplete"
    for arm in receipt["arms"]:
        arm["rows"][0].update(accept_hist={}, depth_hist={})
    assert evaluate(tmp_path, receipt)["verdict"] == "exact"
    receipt["arms"][0]["rows"][0].update(accept_hist={"0": 1})
    assert evaluate(tmp_path, receipt)["verdict"] == "incomplete"


@pytest.mark.parametrize("field", ["text", "completion_tokens", "finish_reason"])
def test_null_compact_values_are_incomplete_even_if_both_arms_agree(tmp_path, receipt, field):
    for arm in receipt["arms"]:
        arm["rows"][0][field] = None
    assert evaluate(tmp_path, receipt)["verdict"] == "incomplete"


def test_checkpoint_needs_an_artifact_fingerprint(tmp_path, receipt):
    receipt["checkpoint"] = "fixture@sha256:"
    assert evaluate(tmp_path, receipt)["verdict"] == "incomplete"


def test_legacy_mixed_arm_reads_request_route_from_full_rows(tmp_path, receipt):
    for arm in receipt["arms"]:
        arm["summary"]["routes"] = ['"external_draft"', '"ordinary"']
        arm["rows"][0].update(accept_hist=None, depth_hist=None)
        out = tmp_path / arm["name"]
        out.mkdir()
        (out / "rows.json").write_text(json.dumps(
            [dict(row, route="external_draft" if i else "ordinary") for i, row in enumerate(arm["rows"])]))
    assert evaluate(tmp_path, receipt)["verdict"] == "exact"
