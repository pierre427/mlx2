"""Synthetic procedural guidance only; no tools, model loads or subprocesses."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mlx2.desktop_skills import Skills


@pytest.fixture
def skills(tmp_path):
    controller = SimpleNamespace(home=tmp_path, user_profiles=SimpleNamespace(active="profile-one"))
    value = Skills(controller)
    yield value
    value.close()


def create(value, name="example", **changes):
    return value.dispatch({"command": "skill_manage", "action": "create", "name": name, "description": "When to use this example", "body": "first instruction", "files": {"references/details.md": "Reference evidence"}, "expected_revision": 0, **changes}, operator=True)


def test_revision_cas_patch_and_history_are_immutable(skills):
    assert create(skills)["revision"] == 1
    updated = skills.dispatch({"command": "skill_manage", "action": "patch", "name": "example", "expected_revision": 1, "old_text": "first", "new_text": "second"}, operator=True)
    assert updated["revision"] == 2
    assert skills.dispatch({"command": "skill_view", "name": "example", "revision": 1})["content"] == "first instruction"
    with pytest.raises(ValueError, match="changed"):
        skills.dispatch({"command": "skill_manage", "action": "archive", "name": "example", "expected_revision": 1}, operator=True)
    with pytest.raises(ValueError, match="exactly once"):
        skills.dispatch({"command": "skill_manage", "action": "patch", "name": "example", "expected_revision": 2, "old_text": "absent", "new_text": "change"}, operator=True)
    assert skills.snapshot()["skills"][0]["revision"] == 2


def test_unicode_paging_references_and_metadata_do_not_dump_bodies(skills):
    text = "🦊 日本語é\n" * 200
    create(skills, body=text)
    offset, pages = 0, []
    while True:
        page = skills.dispatch({"command": "skill_view", "name": "example", "offset": offset, "max_bytes": 256})
        assert len(page["content"].encode()) <= 256
        pages.append(page["content"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert "".join(pages) == text
    reference = skills.dispatch({"command": "skill_view", "name": "example", "path": "references/details.md"})
    assert reference["content"] == "Reference evidence" and reference["reference_only"] is True
    assert "first instruction" not in json.dumps(skills.snapshot())
    with pytest.raises(ValueError, match="Unknown skill reference"):
        skills.dispatch({"command": "skill_view", "name": "example", "path": "../outside"})


@pytest.mark.parametrize("files", [
    {"../escape": "x"}, {"/absolute": "x"}, {"refs//file": "x"}, {"refs/./file": "x"},
    {"refs\\file": "x"}, {"refs/line\nbreak": "x"}, {"SKILL.md": "x"}, {"skill.md/child": "x"},
    {"Refs/a": "x", "refs/A": "y"}, {"refs": "x", "refs/a": "y"}, {"é" * 128: "x"},
])
def test_reference_paths_are_safe_and_portable_before_writing(skills, files):
    with pytest.raises(ValueError):
        create(skills, files=files)
    assert skills.snapshot()["total"] == 0


def test_disabled_archive_restore_and_update_enabled_are_atomic(skills):
    create(skills)
    result = skills.dispatch({"command": "skill_manage", "action": "update", "name": "example", "expected_revision": 1, "body": "updated", "enabled": False}, operator=True)
    assert result["revision"] == 2 and result["enabled"] is False
    assert skills.dispatch({"command": "skills_list"})["skills"] == []
    with pytest.raises(ValueError, match="unavailable"):
        skills.dispatch({"command": "skill_view", "name": "example"})
    assert skills.dispatch({"command": "skill_view", "name": "example"}, operator=True)["content"] == "updated"
    skills.dispatch({"command": "skill_manage", "action": "archive", "name": "example", "expected_revision": 2}, operator=True)
    with pytest.raises(ValueError, match="Restore"):
        skills.dispatch({"command": "skill_manage", "action": "set_enabled", "name": "example", "expected_revision": 3, "enabled": True}, operator=True)
    restored = skills.dispatch({"command": "skill_manage", "action": "restore", "name": "example", "expected_revision": 3}, operator=True)
    assert restored["deleted"] is False and restored["enabled"] is False
    enabled = skills.dispatch({"command": "skill_manage", "action": "set_enabled", "name": "example", "expected_revision": 4, "enabled": True}, operator=True)
    assert enabled["enabled"] is True
    assert skills.dispatch({"command": "skill_view", "name": "example"})["content"] == "updated"


def test_import_is_staged_disabled_and_requires_operator(skills):
    doc = {"name": "imported", "description": "Imported procedure", "body": "Reference only", "files": {}, "enabled": True}
    with pytest.raises(ValueError, match="operator"):
        skills.dispatch({"command": "skill_import", "document": doc})
    result = skills.dispatch({"command": "skill_import", "document": doc}, operator=True)
    assert result["enabled"] is False
    exported = skills.dispatch({"command": "skill_export", "name": "imported"}, operator=True)["document"]
    assert exported["body"] == doc["body"] and exported["enabled"] is False
    assert skills.dispatch({"command": "skills_list"})["skills"] == []


def test_profile_policy_and_documents_remain_isolated_and_stale_ui_guarded(skills):
    create(skills)
    skills.dispatch({"command": "skill_policy", "changes": {"autonomous_writes": True}}, operator=True)
    skills.controller.user_profiles.active = "profile-two"
    assert skills.snapshot()["total"] == 0
    assert skills.policy()["autonomous_writes"] is False
    with pytest.raises(ValueError, match="profile changed"):
        skills.dispatch({"command": "skill_view", "name": "example"}, profile_id="profile-one")
    with pytest.raises(ValueError, match="profile changed"):
        skills.dispatch({"command": "skill_manage", "action": "create", "name": "other", "profile_id": "profile-one"}, operator=True)
    create(skills, body="independent profile-two instructions")
    assert skills.dispatch({"command": "skill_view", "name": "example"})["content"].startswith("independent")
    skills.controller.user_profiles.active = "profile-one"
    assert skills.dispatch({"command": "skill_view", "name": "example"})["content"] == "first instruction"


def test_autonomous_writes_require_operator_policy_and_enabled_system(skills):
    request = {"command": "skill_manage", "action": "create", "name": "agent-tip", "description": "Procedure", "body": "Candidate guidance", "files": {}, "expected_revision": 0}
    with pytest.raises(ValueError, match="approval"):
        skills.dispatch(request)
    with pytest.raises(ValueError, match="operator approval"):
        skills.dispatch({"command": "skill_policy", "changes": {"autonomous_writes": True}})
    skills.dispatch({"command": "skill_policy", "changes": {"autonomous_writes": True}}, operator=True)
    assert skills.dispatch(request)["revision"] == 1
    skills.dispatch({"command": "skill_policy", "changes": {"enabled": False}}, operator=True)
    for request in ({"command": "skills_list"}, {"command": "skill_view", "name": "agent-tip"}):
        with pytest.raises(ValueError, match="disabled"):
            skills.dispatch(request)
    assert skills.dispatch({"command": "skills_list", "includeall": True}, operator=True)["total"] == 1


def test_restart_private_files_and_bounded_metadata_pagination(tmp_path):
    controller = SimpleNamespace(home=tmp_path, user_profiles=SimpleNamespace(active="profile"))
    skills = Skills(controller)
    for index in range(5):
        create(skills, name=f"skill-{index}")
    skills.dispatch({"command": "skill_policy", "changes": {"autonomous_writes": True}}, operator=True)
    skills.close()
    restored = Skills(controller)
    try:
        assert restored.policy()["autonomous_writes"] is True
        page = restored.snapshot(offset=2, limit=2)
        assert page["total"] == 5 and page["next_offset"] == 4
        assert [item["name"] for item in page["skills"]] == ["skill-2", "skill-3"]
        assert (tmp_path / "skills.sqlite3").stat().st_mode & 0o777 == 0o600
        assert restored.dispatch({"command": "skill_view", "name": "skill-0"})["content"] == "first instruction"
    finally:
        restored.close()


def test_reference_catalog_pages_long_filenames_without_inflating_instruction_reads(skills):
    files = {"references/" + f"{index:02d}-" + "x" * 220 + ".md": "bounded reference" for index in range(64)}
    create(skills, body="Small instruction", files=files)
    body = skills.dispatch({"command": "skill_view", "name": "example", "max_bytes": 256})
    assert body["content"] == "Small instruction"
    assert body["reference_count"] == 64
    assert body["reference_catalog_path"] == "."
    assert "files" not in body and "references/" not in json.dumps(body)
    offset, chunks = 0, []
    while True:
        page = skills.dispatch({"command": "skill_view", "name": "example", "path": ".", "offset": offset, "max_bytes": 256})
        assert len(page["content"].encode()) <= 256
        assert "files" not in page
        chunks.append(page["content"])
        if page["next_offset"] is None:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert json.loads("".join(chunks)) == sorted(files)
    reference = skills.dispatch({"command": "skill_view", "name": "example", "path": min(files), "max_bytes": 256})
    assert reference["content"] == "bounded reference"
    with pytest.raises(ValueError):
        create(skills, name="invalid-dot", files={".": "not a catalog override"})
