from types import SimpleNamespace

import pytest

from mlx2.desktop_research import Research


def test_search_read_retain_exact_source_revision_and_consumption(tmp_path, monkeypatch):
    research = Research(SimpleNamespace(home=tmp_path))
    calls = []
    def query(parameters):
        calls.append(parameters)
        if parameters.get("list") == "search":
            return {"query": {"search": [{"pageid": 42, "title": "A source", "snippet": "A <b>result</b>"}]}}
        assert parameters["pageids"] == 42
        return {"query": {"pages": [{"pageid": 42, "extract": "Reference content", "revisions": [{"revid": 11, "timestamp": "2026-10-22T12:00:00Z"}]}]}}
    monkeypatch.setattr(research, "query", query)
    result = research.dispatch({"command": "research_search", "query": "source", "limit": 1}, principal="chat:a")
    source = result["results"][0]
    assert source["snippet"] == "A result"
    read = research.dispatch({"command": "research_read", "source_id": source["id"]}, principal="chat:a")
    assert read["revision"]["revid"] == 11
    assert read["consumed"][0]["principal"] == "chat:a"
    assert "extract" not in research.snapshot()["sources"][0]
    restored = Research(SimpleNamespace(home=tmp_path))
    assert restored.sources[source["id"]]["extract"] == "Reference content"
    with pytest.raises(ValueError, match="admitted"):
        research.dispatch({"command": "research_read", "source_id": "https://arbitrary.invalid"})
    assert len(calls) == 2


@pytest.mark.parametrize("arguments", [{"query": "x", "limit": 1000}, {"query": "", "limit": 1}, {"query": "a" * 301, "limit": 1}])
def test_search_budget_before_network(tmp_path, arguments):
    research = Research(SimpleNamespace(home=tmp_path))
    with pytest.raises(ValueError):
        research.dispatch({"command": "research_search", **arguments})


def test_research_results_are_private_to_searching_principal(tmp_path, monkeypatch):
    research = Research(SimpleNamespace(home=tmp_path))
    calls = []
    def query(parameters):
        calls.append(parameters)
        if parameters.get("list") == "search":
            return {"query": {"search": [{"pageid": 42, "title": "Private research interest", "snippet": "Source reference"}]}}
        return {"query": {"pages": [{"pageid": 42, "extract": "Public source, private consumption", "revisions": [{"revid": 11}]}]}}
    monkeypatch.setattr(research, "query", query)
    source = research.dispatch({"command": "research_search", "query": "private", "limit": 1}, principal="chat:owner")["results"][0]
    assert research.snapshot(principal="chat:other")["sources"] == []
    assert research.snapshot(principal="chat:owner")["sources"][0]["id"] == source["id"]
    assert research.snapshot()["sources"][0]["id"] == source["id"]
    with pytest.raises(PermissionError, match="different principal"):
        research.dispatch({"command": "research_read", "source_id": source["id"]}, principal="chat:other")
    assert len(calls) == 1
    assert research.dispatch({"command": "research_read", "source_id": source["id"]})["revision"]["revid"] == 11


def test_old_admitted_source_remains_readable_after_cache_eviction_and_restart(tmp_path, monkeypatch):
    import json
    import os
    import time
    home = tmp_path / "research"
    home.mkdir()
    oldest = "0" * 32
    for index in range(105):
        identity = f"{index:032x}"
        path = home / (identity + ".json")
        path.write_text(json.dumps({"id": identity, "provider": "Wikipedia", "title": "source", "page_id": index + 1,
            "searched_by": "chat:owner", "searched_at": index, "consumed": []}))
        os.utime(path, (time.time() - 1000 + index, time.time() - 1000 + index))
    research = Research(SimpleNamespace(home=tmp_path))
    assert len(research.sources) == 100 and oldest not in research.sources
    calls = []
    def query(parameters):
        calls.append(parameters)
        return {"query": {"pages": [{"pageid": 1, "extract": "Durably recovered source", "revisions": [{"revid": 12}]}]}}
    monkeypatch.setattr(research, "query", query)
    with pytest.raises(PermissionError, match="different principal"):
        research.dispatch({"command": "research_read", "source_id": oldest}, principal="chat:other")
    assert calls == []
    result = research.dispatch({"command": "research_read", "source_id": oldest}, principal="chat:owner")
    assert result["extract"] == "Durably recovered source"
    assert len(research.sources) == 100


def test_corrupt_saved_research_does_not_prevent_startup_or_grant_sources(tmp_path):
    home = tmp_path / "research"
    home.mkdir()
    (home / (("a" * 32) + ".json")).write_text("not JSON")
    research = Research(SimpleNamespace(home=tmp_path))
    assert research.snapshot()["errors"]
    assert research.snapshot(principal="chat:owner")["errors"] == []
    with pytest.raises(ValueError):
        research.dispatch({"command": "research_read", "source_id": "a" * 32}, principal="chat:owner")
