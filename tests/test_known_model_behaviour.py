"""Declared, artifact-scoped known model behaviour in qualification (CPU).

The Muse-Glimmer 30B 8-bit artifact answers the qualifier's ``tools`` probe
(tool_choice auto) in text, asking for the city, on every route; required
and named tool calls work (qualification/runs/qualify-1007-extra/QUALIFIED.md).
The qualifier used to stop there, so 19 later checks never ran.  The declared
exception records ``tools`` as known model behaviour (never a pass) only for
that check, those artifact fingerprints and that exact failure shape, and the
run continues.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import qualify_serving  # noqa: E402

from mlx2.adapters.qwen import QWEN4_FLASH_NEXT  # noqa: E402
from mlx2.known_model_behaviour import (  # noqa: E402
    KNOWN_MODEL_BEHAVIOUR_STATUS,
    known_model_behaviour_for,
)
from mlx2.qualification import (  # noqa: E402
    APPROVED_QUALIFICATION_HARNESS,
    REQUIRED_CHECKS,
    load_qualified_route,
)

MUSE_8BIT = "e4f23559c63fcf614116136c1205fd9bf1795e289001c2e2441f783f356c4793"
MUSE_8BIT_DFLASH2 = "116136e41ac5017b9823bcbbd5a8806deaced3bed049694efa3d3ea05a53f065"
OTHER = "0" * 64

ASKS_FOR_CITY = {
    "choices": [{
        "index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": (
            "I can help you with that. Could you please specify the city you "
            "would like to check the weather for?")},
    }],
}


def call(city="Toronto", name="weather", finish="tool_calls"):
    return {
        "choices": [{
            "index": 0, "finish_reason": finish,
            "message": {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": name, "arguments": json.dumps({"city": city})},
            }]},
        }],
    }


class Server:
    """Answers the probe: auto gets ``auto``, an enforced choice ``forced``."""

    def __init__(self, auto, forced):
        self.auto, self.forced, self.requests = auto, forced, []

    def __call__(self, body):
        self.requests.append(copy.deepcopy(body))
        return copy.deepcopy(self.forced if body.get("tool_choice") else self.auto)


def test_declaration_is_bound_to_check_and_artifact():
    assert known_model_behaviour_for("tools", MUSE_8BIT) is not None
    assert known_model_behaviour_for("tools", MUSE_8BIT_DFLASH2) is not None
    assert known_model_behaviour_for("tools", OTHER) is None
    assert known_model_behaviour_for("tool_roundtrip", MUSE_8BIT) is None
    assert known_model_behaviour_for("reasoning", MUSE_8BIT) is None


@pytest.mark.parametrize("artifact", [MUSE_8BIT, MUSE_8BIT_DFLASH2])
def test_matching_artifact_and_shape_is_known_behaviour_not_a_pass(artifact):
    server = Server(ASKS_FOR_CITY, call())
    probe = qualify_serving.run_tools_probe(server, artifact)
    assert probe.status == KNOWN_MODEL_BEHAVIOUR_STATUS
    assert probe.known["check"] == "tools"
    assert probe.known["artifact"] == artifact
    assert probe.known["reason"] and probe.known["evidence"]
    assert probe.evidence["auto"]["choices"][0]["finish_reason"] == "stop"
    # The enforced re-ask is the same request plus tool_choice "required".
    assert [r.get("tool_choice") for r in server.requests] == [None, "required"]
    assert server.requests[1]["messages"] == server.requests[0]["messages"]
    # The round trip continues from the model's own enforced call.
    assert probe.call_response["choices"][0]["message"]["tool_calls"]


def test_other_artifact_still_fails_and_makes_no_extra_request():
    server = Server(ASKS_FOR_CITY, call())
    probe = qualify_serving.run_tools_probe(server, OTHER)
    assert probe.status == "failed" and probe.known is None
    assert len(server.requests) == 1


@pytest.mark.parametrize("auto, forced", [
    # Truncated, not a deliberate text reply.
    ({"choices": [{"index": 0, "finish_reason": "length",
                   "message": {"role": "assistant", "content": "I can"}}]}, call()),
    # Empty answer.
    ({"choices": [{"index": 0, "finish_reason": "stop",
                   "message": {"role": "assistant", "content": ""}}]}, call()),
    # A wrong tool call under auto is a different failure.
    (call(city="Ottawa"), call()),
    # The enforced call no longer works: not the declared behaviour.
    (ASKS_FOR_CITY, call(city="Ottawa")),
    (ASKS_FOR_CITY, ASKS_FOR_CITY),
])
def test_other_failure_shapes_still_fail(auto, forced):
    probe = qualify_serving.run_tools_probe(Server(auto, forced), MUSE_8BIT)
    assert probe.status == "failed" and probe.known is None
    assert probe.evidence["known_model_behaviour_rejected"]["reasons"]


def test_passing_tools_is_unchanged():
    server = Server(call(), call())
    probe = qualify_serving.run_tools_probe(server, MUSE_8BIT)
    assert probe.status == "passed" and probe.known is None
    assert len(server.requests) == 1
    assert probe.evidence == call()


def test_producer_records_known_behaviour_and_continues():
    """The producer's check recorder: a known result is recorded with the
    exception and does not raise, so the remaining checks run."""
    report = {"checks": {}}
    probe = qualify_serving.run_tools_probe(Server(ASKS_FOR_CITY, call()), MUSE_8BIT)
    qualify_serving.record_tools_probe(report, probe)
    entry = report["checks"]["tools"]
    assert entry["passed"] is False
    assert entry["status"] == KNOWN_MODEL_BEHAVIOUR_STATUS
    assert entry["known_model_behaviour"]["id"] == probe.known["id"]
    assert report["known_model_behaviour"]["tools"] == probe.known
    # A plain failure still stops the run.
    failed = qualify_serving.run_tools_probe(Server(ASKS_FOR_CITY, call()), OTHER)
    with pytest.raises(AssertionError, match="tools"):
        qualify_serving.record_tools_probe({"checks": {}}, failed)


def _receipt(artifact, tools_entry):
    checks = {c: {"passed": True} for c in REQUIRED_CHECKS | {"mtp_execution", "structured_output"}}
    checks["tools"] = tools_entry
    return {
        "passed": True, "runtime": {"source": "abc"}, "artifact": artifact,
        "settings": {"mtp": True, "max_context": 16384, "default_max_tokens": 65_536},
        "qualification_harness": APPROVED_QUALIFICATION_HARNESS,
        "checks": checks,
    }


def _known_entry(artifact):
    report = {"checks": {}}
    probe = qualify_serving.run_tools_probe(Server(ASKS_FOR_CITY, call()), artifact)
    qualify_serving.record_tools_probe(report, probe)
    return report["checks"]["tools"]


def _load(tmp_path, record, artifact):
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(record))
    return load_qualified_route(
        path, runtime=record["runtime"], artifact=artifact,
        settings=record["settings"], descriptor=QWEN4_FLASH_NEXT, name="known",
    )


def test_route_receipt_carries_the_exception(tmp_path):
    record = _receipt(MUSE_8BIT, _known_entry(MUSE_8BIT))
    route = _load(tmp_path, record, MUSE_8BIT)
    entry = known_model_behaviour_for("tools", MUSE_8BIT)
    assert f"known_model_behaviour=tools:{entry.id}" in route.receipt


def test_loader_refuses_known_entry_on_other_artifact(tmp_path):
    # A receipt of another artifact carrying a copied exception.
    record = _receipt(OTHER, _known_entry(MUSE_8BIT))
    with pytest.raises(ValueError, match="failed"):
        _load(tmp_path, record, OTHER)


def test_loader_refuses_edited_known_entry(tmp_path):
    entry = _known_entry(MUSE_8BIT)
    entry["evidence"]["forced"] = ASKS_FOR_CITY  # shape no longer matches
    with pytest.raises(ValueError, match="failed"):
        _load(tmp_path, _receipt(MUSE_8BIT, entry), MUSE_8BIT)
    entry = _known_entry(MUSE_8BIT)
    entry["known_model_behaviour"]["reason"] = "edited"
    with pytest.raises(ValueError, match="failed"):
        _load(tmp_path, _receipt(MUSE_8BIT, entry), MUSE_8BIT)


def test_known_behaviour_is_only_for_the_declared_check(tmp_path):
    record = _receipt(MUSE_8BIT, _known_entry(MUSE_8BIT))
    record["checks"]["reasoning"] = {**record["checks"]["tools"]}
    with pytest.raises(ValueError, match="failed"):
        _load(tmp_path, record, MUSE_8BIT)


def test_plain_passing_receipt_has_no_exception_in_route_receipt(tmp_path):
    record = _receipt(MUSE_8BIT, {"passed": True})
    assert "known_model_behaviour" not in _load(tmp_path, record, MUSE_8BIT).receipt
