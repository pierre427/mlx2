"""Synthetic research progress lifecycle; no models, downloads or network."""
from types import SimpleNamespace

import pytest

from mlx2.desktop_research_plans import ResearchPlans


@pytest.fixture
def plans(tmp_path):
    controller = SimpleNamespace(home=tmp_path, user_profiles=SimpleNamespace(active="alice"), tool_sessions={})
    ledger = ResearchPlans(controller)
    yield ledger
    ledger.db.close()


def session(parent="chat:a", profile="alice"):
    return {"user_profile": profile, "chat_session_id": parent}


def test_plan_exists_before_real_call_and_records_only_observed_stages(plans):
    def search():
        value = plans.snapshot()["plans"][0]
        assert value["status"] == "active"
        assert value["steps"][0]["status"] == "running"
        assert value["steps"][1]["status"] == "pending"
        return {"results": [{"id": "reference", "url": "https://example.test/source", "title": "Synthetic"}]}
    result, receipt = plans.execute("tools:a", session(), "web_search", {"query": "synthetic"}, search)
    assert result["results"][0]["id"] == "reference"
    assert receipt["status"] == "active"
    value = plans.snapshot()["plans"][0]
    assert value["steps"][0]["status"] == "completed"
    assert value["steps"][1]["status"] == "pending"
    assert value["steps"][2]["status"] == "pending"
    assert value["events"][0]["evidence"]["request"] == "synthetic"
    assert value["events"][0]["evidence"]["returned_references"] == 1
    plans.finish("tools:a", "completed")
    value = plans.snapshot()["plans"][0]
    assert value["steps"][1]["status"] == "not_observed"
    assert value["steps"][2]["status"] == "completed"
    assert "not independently verified" in value["notice"]


def test_failure_preserves_error_type_without_provider_secret_or_content(plans):
    def fail():
        raise RuntimeError("secret-provider-token")
    with pytest.raises(RuntimeError):
        plans.execute("tools:a", session(), "fetch_page", {"url": "https://example.test"}, fail)
    value = plans.snapshot()["plans"][0]
    assert value["steps"][1]["status"] == "failed"
    assert value["events"][0]["evidence"]["error_type"] == "RuntimeError"
    assert "secret-provider-token" not in str(value)
    plans.finish("tools:a", "failed")
    assert plans.snapshot()["plans"][0]["status"] == "failed"


def test_profiles_and_chats_scope_history_and_completion(plans):
    plans.execute("tools:a", session(), "research_read", {"source_id": "ref:a"}, lambda: {"revision": {"revid": 42}, "extract": "private text"})
    assert plans.snapshot(parent_id="chat:other")["plans"] == []
    assert "private text" not in str(plans.snapshot())
    plans.controller.user_profiles.active = "bob"
    assert plans.snapshot()["plans"] == []
    with pytest.raises(ValueError, match="different profile"):
        plans.finish("tools:a", "completed")
    with pytest.raises(ValueError, match="different profile"):
        plans.begin("tools:b", session(), "web_search", {"query": "x"})
    plans.controller.user_profiles.active = "alice"
    plans.controller.tool_sessions["continuation"] = session("chat:other")
    with pytest.raises(ValueError, match="scope"):
        plans.finish("continuation", "completed", parent_id="chat:a")


def test_child_research_waits_for_parent_response_after_child_finishes(plans):
    receipt = plans.attach_child("tools:a", session(), "job:a", "Find and read references")
    assert receipt["child_job_id"] == "job:a"
    before = plans.snapshot()["plans"][0]
    assert before["events"] == []
    assert all(step["status"] == "pending" for step in before["steps"])
    plans.controller.tool_sessions["tools:a"] = session()
    first_finish = plans.finish("tools:a", "completed", parent_id="chat:a")
    assert first_finish["status"] == "active"
    plans.execute_child("job:a", "web_search", {"query": "synthetic"}, lambda: {"results": []})
    plans.execute_child("job:a", "fetch_page", {"url": "https://example.test"}, lambda: {"url": "https://example.test", "text": "do not persist"})
    plans.child_finished("job:a", "completed")
    waiting = plans.snapshot()["plans"][0]
    assert waiting["status"] == "active"
    assert "awaiting the parent" in waiting["waiting"]
    assert waiting["steps"][2]["status"] == "pending"
    plans.controller.tool_sessions["continuation"] = session()
    completed = plans.finish("continuation", "completed", parent_id="chat:a")
    assert completed["status"] == "completed"
    assert completed["plans"][0]["steps"][2]["status"] == "completed"
    assert "do not persist" not in str(completed)


def test_cancelled_plan_blocks_further_child_network_activity(plans):
    plans.attach_child("tools:a", session(), "job:a", "Research")
    plans.finish("tools:a", "cancelled")
    calls = []
    with pytest.raises(ValueError, match="inactive"):
        plans.execute_child("job:a", "web_search", {"query": "x"}, lambda: calls.append(True))
    assert calls == []
    assert plans.finish("tools:a", "completed")["plans"][0]["status"] == "cancelled"


def test_restart_retains_progress_and_marks_interrupted_work(plans):
    plans.begin("tools:a", session(), "web_search", {"query": "synthetic"})
    restored = ResearchPlans(plans.controller)
    try:
        value = restored.snapshot()["plans"][0]
        assert value["status"] == "interrupted"
        assert value["events"][0]["status"] == "interrupted"
        assert value["steps"][2]["status"] == "not_observed"
    finally:
        restored.db.close()


def test_no_research_does_not_create_a_plan(plans):
    assert plans.finish("tools:no-research", "completed") == {"status": "not_started"}
    assert plans.snapshot()["plans"] == []


def test_http_error_does_not_claim_successful_source_read(plans):
    plans.execute("tools:a", session(), "fetch_page", {"url": "https://example.test/missing"},
                  lambda: {"status": "404", "url": "https://example.test/missing", "content": "Not found"})
    value = plans.snapshot()["plans"][0]
    assert value["steps"][1]["status"] == "failed"
    assert value["events"][0]["evidence"]["status"] == "404"
    assert "successfully" not in value["events"][0]["evidence"]["observation"]


def test_fast_child_does_not_finish_before_original_parent_receives_report(plans):
    plans.attach_child("tools:a", session(), "job:a", "Research")
    plans.child_finished("job:a", "completed")
    plans.controller.tool_sessions["tools:a"] = session()
    assert plans.finish("tools:a", "completed", parent_id="chat:a")["status"] == "active"
    plans.mark_child_consumed("job:a", "tools:a")
    assert plans.finish("tools:a", "completed", parent_id="chat:a")["status"] == "completed"
