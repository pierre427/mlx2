from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlx2.desktop_capsules import CapsuleController
from mlx2.desktop_memory import MemoryController


@pytest.fixture
def harness(tmp_path):
    host = SimpleNamespace(home=tmp_path, preferences={})
    host.memory = MemoryController(host)
    host.capsules = CapsuleController(host)
    yield host
    host.capsules.close()
    host.memory.close()


def note(host, scope="project:mlx2"):
    return host.memory.dispatch({"command": "memory_add", "title": "Conversion lesson", "body": "Reserve disk space before conversion", "scope": scope})


def actor(scope="project:mlx2"):
    return {"principal_id": "worker-one", "allowed_scopes": [scope], "session_ids": ["session-one"], "worker_ids": [], "can_promote": False}


def test_note_capsule_uses_real_integrity_store_and_never_applies_activation(harness):
    source = note(harness)
    result = harness.capsules.dispatch({"command": "capsule_capture", "id": source["id"], "revision": 1, "situation": "user_requested"})
    assert len(result["digest"]) == 64
    value = harness.capsules.dispatch({"command": "capsule_get", "digest": result["digest"]})
    assert value["kind"] == "evaluation"
    assert value["data"]["note"]["body"] == source["body"]
    assert value["provenance"]["note_revision"] == 1
    assert value["provenance"]["activation_applied"] is False
    assert value["provenance"]["authority"] == "operator-confirmed"
    with pytest.raises(ValueError, match="qualified runtime"):
        harness.capsules.dispatch({"command": "capsule_apply_approximate", "digest": result["digest"]})


def test_autonomy_off_scope_denial_and_no_privilege_arguments(harness):
    source = note(harness)
    command = {"command": "capsule_capture", "id": source["id"], "revision": 1, "situation": "correction"}
    with pytest.raises(PermissionError, match="off"):
        harness.capsules.dispatch(command, context=actor())
    with pytest.raises(ValueError, match="grant authority"):
        harness.capsules.dispatch({**command, "capsule_operation_approved": True}, context=actor())
    with pytest.raises(PermissionError):
        harness.capsules.dispatch(command, context=actor("project:other"))
    assert harness.capsules.snapshot()["capsules"] == []


def test_situational_authority_and_budget_remain_explicit(harness):
    source = note(harness)
    harness.capsules.configure({"autonomy": "situational", "allowed_situations": ["correction"], "max_operations_per_session": 1})
    command = {"command": "capsule_capture", "id": source["id"], "revision": 1, "situation": "tool_failure"}
    with pytest.raises(PermissionError, match="situation"):
        harness.capsules.dispatch(command, context=actor())
    result = harness.capsules.dispatch({**command, "situation": "correction"}, context=actor())
    assert result["authority"] == "model-policy-unverified"
    assert result["activation_applied"] is False
    value = harness.capsules.dispatch({"command": "capsule_get", "digest": result["digest"]}, context=actor())
    assert "note" not in value["data"]
    with pytest.raises(ValueError, match="budget exhausted"):
        harness.capsules.dispatch({**command, "situation": "correction"}, context=actor())


def test_policy_approved_model_promotion_is_unverified_not_operator_confirmation(harness):
    source = note(harness)
    harness.capsules.configure({"autonomy": "always", "allow_promote": True})
    result = harness.capsules.dispatch({"command": "capsule_review_note", "id": source["id"], "revision": 1,
        "status": "promoted", "reason": "Repeated successful tool use", "situation": "useful_feedback"}, context=actor())
    loaded = harness.memory.dispatch({"command": "memory_get", "id": source["id"]})
    assert loaded["status"] == "promoted"
    assert "model-policy-unverified" in loaded["reason"]
    capsule = harness.capsules.dispatch({"command": "capsule_get", "digest": result["digest"]})
    assert capsule["data"]["operator_confirmed"] is False
    assert capsule["data"]["authority"] == "model-policy-unverified"


def test_promotion_and_policy_changes_need_explicit_saved_authority(harness):
    source = note(harness)
    with pytest.raises(PermissionError, match="operator approval"):
        harness.capsules.configure({"autonomy": "always"}, context=actor())
    harness.capsules.configure({"autonomy": "always"})
    with pytest.raises(PermissionError, match="promotion"):
        harness.capsules.dispatch({"command": "capsule_review_note", "id": source["id"], "revision": 1,
            "status": "promoted", "reason": "trust me", "situation": "user_requested"}, context=actor())
    assert harness.memory.dispatch({"command": "memory_get", "id": source["id"]})["revision"] == 1


def test_parent_references_block_deletion_and_cross_scope_linking(harness):
    one = note(harness)
    first = harness.capsules.dispatch({"command": "capsule_capture", "id": one["id"], "revision": 1})
    second = harness.capsules.dispatch({"command": "capsule_capture", "id": one["id"], "revision": 1, "parents": [first["digest"]]})
    with pytest.raises(ValueError, match="child snapshot"):
        harness.capsules.dispatch({"command": "capsule_delete", "digest": first["digest"]})
    other = note(harness, "private:other")
    with pytest.raises(PermissionError, match="same scope"):
        harness.capsules.dispatch({"command": "capsule_capture", "id": other["id"], "revision": 1, "parents": [first["digest"]]})
    harness.capsules.dispatch({"command": "capsule_delete", "digest": second["digest"]})
    assert harness.capsules.dispatch({"command": "capsule_delete", "digest": first["digest"]})["deleted"] is True
