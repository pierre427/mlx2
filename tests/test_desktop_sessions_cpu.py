"""Lossless private archives and self-session retrieval using synthetic JSON."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from mlx2.desktop_sessions import FIELDS, SessionArchive, canonical


@pytest.fixture
def archive(tmp_path):
    value = SessionArchive(SimpleNamespace(home=tmp_path))
    yield value
    value.close()


def ctx(profile="profile", session="session"):
    return {"profile_id": profile, "session_id": session}


def read_all(archive, context, **kwargs):
    chunks, offset = [], 0
    while True:
        reply = archive.get(context=context, offset=offset, max_bytes=2048, **kwargs)
        assert len(canonical(reply)) <= 2048
        chunks.append(reply["chunk"])
        if reply["eof"]:
            break
        assert reply["next_offset"] > offset
        offset = reply["next_offset"]
    return "".join(chunks).encode()


def test_multimodal_tool_groups_roundtrip_exactly_without_raw_snapshot(archive):
    document = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + "AQID" * 2048}}, {"type": "text", "text": "日本語 🦊\n" * 256}]},
        {"role": "assistant", "tool_calls": [{"id": "call_one", "function": {"name": "read", "arguments": '{"path":"x"}'}}]},
        {"role": "tool", "tool_call_id": "call_one", "content": "Exact-result evidence alpha"}],
        "attachments": [{"blobref": "private-sha256:example", "name": "source.mov", "media_type": "video/quicktime"}],
        "draft": "unfinished", "tools": [{"name": "read", "schema": {"type": "object"}}]}
    receipt = archive.archive("profile", "session", document)
    recovered = read_all(archive, ctx())
    assert recovered == canonical(document)
    assert hashlib.sha256(recovered).hexdigest() == receipt["digest"]
    assert json.loads(recovered) == document
    assert "Exact-result" not in json.dumps(archive.snapshot())
    assert archive.search("evidence alpha", context=ctx())["matches"]
    assert archive.search("source.mov", context=ctx())["matches"]
    eof = archive.get(context=ctx(), offset=receipt["bytes"])
    assert eof["eof"] and eof["chunk"] == ""


def test_append_only_revisions_deduplicate_and_reject_stale_writes(archive):
    first = archive.archive("profile", "session", {"message": "one"}, expected_revision=0)
    assert archive.archive("profile", "session", {"message": "one"}) == first
    second = archive.archive("profile", "session", {"message": "two"}, expected_revision=1)
    assert second["revision"] == 2
    with pytest.raises(ValueError, match="changed"):
        archive.archive("profile", "session", {"message": "stale"}, expected_revision=1)
    assert json.loads(read_all(archive, ctx(), revision=1)) == {"message": "one"}
    assert json.loads(read_all(archive, ctx(), revision=2)) == {"message": "two"}
    assert archive.snapshot()["revisions"] == 2


def test_profile_siblings_descendants_and_ancestor_future_turns_are_isolated(archive):
    archive.archive("profile", "parent", {"message": "shared ancestor"})
    archive.archive("profile", "child", {"message": "child private"}, parent_id="parent", parent_revision=1)
    archive.archive("profile", "sibling", {"message": "sibling secret"}, parent_id="parent", parent_revision=1)
    archive.archive("profile", "parent", {"message": "future ancestor secret"})
    archive.archive("other", "child", {"message": "other profile secret"})
    context = ctx(session="child")
    assert archive.search("shared", context=context)["matches"]
    for query in ("sibling secret", "future ancestor", "other profile"):
        assert archive.search(query, context=context)["matches"] == []
    for session in ("sibling", "unknown"):
        with pytest.raises(ValueError, match="outside"):
            archive.get(context=context, session_id=session)
    with pytest.raises(ValueError, match="Revision"):
        archive.get(context=context, session_id="parent", revision=2)
    assert not archive.search("child private", context=ctx(session="parent"))["matches"]
    with pytest.raises(ValueError, match="immutable"):
        archive.archive("profile", "child", {"message": "changed"}, parent_id="sibling")


def test_compaction_is_revision_bound_candidate_and_original_stays_lossless(archive):
    original = {"wire": [{"role": "tool", "tool_call_id": "a", "content": "kept exactly"}], "goal": "test"}
    receipt = archive.archive("profile", "session", original)
    summary = {field: ["candidate note"] for field in FIELDS}
    checkpoint = archive.compact(context=ctx(), expected_revision=receipt["revision"], expected_digest=receipt["digest"], summary=summary)
    assert checkpoint["original_retained"] is True
    assert checkpoint["authority"] == "model-authored-candidate"
    assert read_all(archive, ctx()) == canonical(original)
    archive.archive("profile", "session", {"later": True})
    with pytest.raises(ValueError, match="input changed"):
        archive.compact(context=ctx(), expected_revision=receipt["revision"], expected_digest=receipt["digest"], summary=summary)
    assert archive.snapshot()["checkpoints"] == 1
    assert read_all(archive, ctx(), revision=1) == canonical(original)


def test_search_and_chunks_are_bounded_and_invalid_offsets_fail_closed(archive):
    archive.archive("profile", "session", {"items": ["needle " * 1000 for _ in range(100)], "unicode": "🦊"})
    result = archive.search("needle", context=ctx(), limit=20, max_bytes=2048)
    assert len(canonical(result)) <= 2048
    assert result["matches"] and result["next_offset"]
    with pytest.raises(ValueError, match="between"):
        archive.get(context=ctx(), max_bytes=100000)
    raw = read_all(archive, ctx())
    bad = raw.index("🦊".encode()) + 1
    with pytest.raises(ValueError, match="UTF-8 boundary"):
        archive.get(context=ctx(), offset=bad)
    with pytest.raises(ValueError, match="authority"):
        archive.dispatch({"command": "session_search", "query": "needle", "profile_id": "other"}, context=ctx())


def test_archive_transaction_rolls_back_index_failure_without_partial_revision(archive, monkeypatch):
    archive.archive("profile", "session", {"old": "kept"})
    def fail(document):
        yield "/new", 0, "first segment"
        raise RuntimeError("synthetic indexing failure")
    monkeypatch.setattr("mlx2.desktop_sessions.text_leaves", fail)
    with pytest.raises(RuntimeError, match="indexing"):
        archive.archive("profile", "session", {"new": "never published"})
    assert archive.snapshot()["revisions"] == 1
    assert json.loads(read_all(archive, ctx())) == {"old": "kept"}


def operator(session="session", profile="profile"):
    return {**ctx(profile, session), "archive_operator": True}


def begin_upload(archive, document, **kwargs):
    raw = json.dumps(document, ensure_ascii=False, indent=2).encode()
    receipt = archive.dispatch({"command": "session_archive_begin", "session_id": "session", "total_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), **kwargs}, context=operator())
    return receipt["upload_id"], raw


def chunk(archive, identity, raw, index=0, **kwargs):
    import base64
    return archive.dispatch({"command": "session_archive_chunk", "upload_id": identity, "index": index, "data_base64": base64.b64encode(raw).decode()}, context=operator(**kwargs))


def test_native_chunk_protocol_preserves_large_exact_document_and_cleans_staging(archive):
    document = {"large": "🦊 tool-result " * 100000, "media": {"url": "data:image/png;base64,AQID"}}
    identity, raw = begin_upload(archive, document)
    for index, offset in enumerate(range(0, len(raw), 1 << 20)):
        chunk(archive, identity, raw[offset:offset + (1 << 20)], index)
    receipt = archive.dispatch({"command": "session_archive_commit", "upload_id": identity}, context=operator())
    assert receipt["digest"] == hashlib.sha256(canonical(document)).hexdigest()
    assert read_all(archive, ctx()) == canonical(document)
    assert list(archive.upload_root.iterdir()) == []
    assert archive.uploads == {}
    status = archive.dispatch({"command": "session_archive_status", "session_id": "session"}, context=operator())
    assert status["archive"]["digest"] == receipt["digest"]


def test_native_upload_scope_order_hash_and_revision_cas_fail_closed(archive):
    identity, raw = begin_upload(archive, {"new": "state"})
    with pytest.raises(ValueError, match="native operator"):
        archive.dispatch({"command": "session_archive_chunk", "upload_id": identity, "index": 0, "data_base64": "eA=="}, context=ctx())
    with pytest.raises(ValueError, match="active profile"):
        chunk(archive, identity, raw, profile="other")
    with pytest.raises(ValueError, match="sequential"):
        chunk(archive, identity, raw, index=1)
    with pytest.raises(ValueError, match="length or digest"):
        archive.dispatch({"command": "session_archive_commit", "upload_id": identity}, context=operator())
    chunk(archive, identity, raw)
    archive.archive("profile", "session", {"concurrent": "newer state"})
    with pytest.raises(ValueError, match="revision changed"):
        archive.dispatch({"command": "session_archive_commit", "upload_id": identity}, context=operator())
    assert json.loads(read_all(archive, ctx())) == {"concurrent": "newer state"}
    assert archive.uploads == {}
    assert list(archive.upload_root.iterdir()) == []


def test_invalid_native_document_rolls_back_and_releases_upload(archive):
    identity, raw = begin_upload(archive, {"original": "unchanged"})
    broken = b"x" * len(raw)
    archive.uploads[identity]["digest"] = hashlib.sha256(broken).hexdigest()
    chunk(archive, identity, broken)
    with pytest.raises(json.JSONDecodeError):
        archive.dispatch({"command": "session_archive_commit", "upload_id": identity}, context=operator())
    assert archive.snapshot()["sessions"] == archive.snapshot()["revisions"] == 0
    assert archive.uploads == {}


def test_ancestor_old_archive_cannot_leak_future_turns_before_frozen_prefix_revision(archive):
    archive.archive("profile", "parent", {"messages": ["before fork", "PRIVATE FUTURE TURN"]})
    fork_snapshot = archive.archive("profile", "parent", {"messages": ["before fork"]})
    assert fork_snapshot["revision"] == 2
    archive.archive("profile", "child", {"messages": ["before fork", "child turn"]}, parent_id="parent", parent_revision=2)
    context = ctx(session="child")
    assert archive.search("before fork", context=context)["matches"]
    assert archive.search("PRIVATE FUTURE TURN", context=context)["matches"] == []
    with pytest.raises(ValueError, match="exact frozen fork snapshot"):
        archive.get(context=context, session_id="parent", revision=1)
    assert json.loads(read_all(archive, context, session_id="parent")) == {"messages": ["before fork"]}
    assert archive.search("PRIVATE FUTURE TURN", context=ctx(session="parent"))["matches"] == []


def test_search_indexes_latest_revision_once_and_preserves_frozen_ancestor(archive):
    archive.archive("profile", "parent", {"messages": ["frozen fox"]})
    archive.archive("profile", "parent", {"messages": ["latest fox"]})
    # A late branch may choose an old immutable revision after its FTS was pruned.
    archive.archive("profile", "child", {"messages": ["child"]}, parent_id="parent", parent_revision=1)
    for index in range(5):
        archive.archive("profile", "parent", {"messages": ["latest fox", str(index)]})
    matches = archive.search("fox", context=ctx(session="parent"))["matches"]
    assert len(matches) == 1 and matches[0]["revision"] == 7
    assert archive.search("frozen fox", context=ctx(session="child"))["matches"][0]["revision"] == 1
    assert archive.db.execute("SELECT COUNT(*) FROM session_search WHERE session='parent'").fetchone()[0] == 3
