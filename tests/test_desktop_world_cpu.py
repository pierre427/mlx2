"""Declarative portability, permission and scene validation without GPU work."""
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2.desktop_world import Worlds, default_world, validate_world


def test_saved_world_round_trip_import_does_not_alias_or_open_resources(tmp_path):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    original = worlds.snapshot()["active"]
    document = copy.deepcopy(original)
    document["objects"][1]["resource"] = "/private/not-a-grant.png"
    imported = worlds.dispatch({"command": "world_import", "document": document})["active"]
    assert imported["id"] != original["id"]
    assert imported["objects"][1]["resource"] == "/private/not-a-grant.png"
    assert worlds.dispatch({"command": "world_export", "world_id": imported["id"]})["document"] == {"world": imported, "assets": {}}
    worlds.dispatch({"command": "world_policy", "changes": {"autonomous_scene": True}})
    loaded = Worlds(SimpleNamespace(home=tmp_path)).snapshot()
    assert loaded["active"] == imported
    assert loaded["policy"] == {"autonomous_scene": True}
    assert (tmp_path / "worlds/state.json").stat().st_mode & 0o777 == 0o600
    assert document["id"] == original["id"]


@pytest.mark.parametrize("mutation", [
    lambda w: w.update(script="arbitrary code"),
    lambda w: w.update(id="../../escape"),
    lambda w: w["objects"][0].update(position=[float("nan"), 0, 0]),
    lambda w: w["objects"][0].update(resource="javascript:alert(1)"),
    lambda w: w["objects"][0].update(animation={"kind": "shader", "speed": 1}),
    lambda w: w["objects"][0].update(scale=[10000, 1, 1]),
    lambda w: w.update(objects=[]),
    lambda w: w.update(objects=w["objects"] * 100),
    lambda w: w["transitions"].update(duration=float("inf")),
])
def test_import_validation_refuses_executable_or_unbounded_content(mutation):
    value = default_world()
    mutation(value)
    with pytest.raises(ValueError):
        validate_world(value)


def test_last_chat_cannot_be_removed_and_rejected_edit_does_not_change_disk(tmp_path):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    before = json.loads((tmp_path / "worlds/state.json").read_text())
    with pytest.raises(ValueError, match="chat"):
        worlds.dispatch({"command": "world_object_remove", "object_id": "chat"})
    assert worlds.active == before["active"]
    assert json.loads((tmp_path / "worlds/state.json").read_text()) == before


def test_texture_bundle_is_content_bound_and_portable_across_app_homes(tmp_path):
    from PIL import Image
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    source, target = Worlds(SimpleNamespace(home=first)), Worlds(SimpleNamespace(home=second))
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), "blue").save(stream, format="PNG")
    resource = source.store_image(stream.getvalue())
    document = source.snapshot()["active"]
    document["objects"][1]["resource"] = resource
    source.dispatch({"command": "world_save", "world": document})
    bundle = source.dispatch({"command": "world_export", "world_id": document["id"]})["document"]
    imported = target.dispatch({"command": "world_import", "document": bundle})["active"]
    from pathlib import Path
    imported_file = Path(imported["objects"][1]["resource"])
    assert imported_file.parent == second / "worlds/assets"
    assert imported_file.read_bytes() == stream.getvalue()
    bad = copy.deepcopy(bundle)
    bad["assets"][next(iter(bad["assets"]))] = "ZGlmZmVyZW50"
    with pytest.raises(ValueError, match="digest"):
        target.dispatch({"command": "world_import", "document": bad})


def test_generated_artifact_representation_does_not_overwrite_agent_edits(tmp_path):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    jobs = [{"id": "job1", "kind": "media", "state": "completed", "spec": {"capability": "video"},
             "result": {"artifacts": [{"path": "/not-automatically-opened/video.mp4"}]}}]
    worlds.sync_artifacts(jobs)
    obj = next(o for o in worlds.active["objects"] if o["id"] == "artifact-job1-0")
    obj["label"] = "An agent's composition"
    worlds.dispatch({"command": "world_object_upsert", "object": obj})
    worlds.sync_artifacts(jobs)
    assert next(o for o in worlds.active["objects"] if o["id"] == obj["id"])["label"] == obj["label"]
    worlds.dispatch({"command": "world_object_remove", "object_id": obj["id"]})
    worlds.sync_artifacts(jobs)
    assert not any(o["id"] == obj["id"] for o in worlds.active["objects"])


def test_provider_artifact_paths_cannot_grant_unrelated_local_image_reads(tmp_path):
    from PIL import Image
    external = tmp_path / "personal.png"
    Image.new("RGB", (2, 2)).save(external)
    home = tmp_path / "app"
    home.mkdir()
    worlds = Worlds(SimpleNamespace(home=home))
    job = {"id": "external-image", "kind": "media", "state": "completed", "spec": {"capability": "image"}, "result": {"path": str(external)}}
    worlds.sync_artifacts([job])
    obj = next(o for o in worlds.active["objects"] if o["id"] == "artifact-external-image-0")
    assert obj["resource"] == str(external)
    assert not list(worlds.assets.iterdir())
    managed = home / "shared-media/images/artifacts/generated.png"
    managed.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(managed)
    job = {**job, "id": "managed-image", "result": {"path": str(managed)}}
    worlds.sync_artifacts([job])
    obj = next(o for o in worlds.active["objects"] if o["id"] == "artifact-managed-image-0")
    assert Path(obj["resource"]).parent == worlds.assets
    assert len(list(worlds.assets.iterdir())) == 1


def test_stored_world_autonomy_must_not_treat_string_false_as_true(tmp_path):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    path = worlds.home / "state.json"
    document = json.loads(path.read_text())
    document["policy"]["autonomous_scene"] = "false"
    path.write_text(json.dumps(document))
    # A non-boolean grant never enables autonomy, and it does not stop the app starting.
    assert Worlds(SimpleNamespace(home=tmp_path)).policy == {"autonomous_scene": False}


@pytest.mark.parametrize('document', ['{', '[]', '{"id":"broken","name":"broken"}'])
def test_invalid_inactive_world_does_not_break_status_snapshot(tmp_path, document):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    original = copy.deepcopy(worlds.active)
    broken = worlds.home / 'broken.json'
    broken.write_text(document)
    state = worlds.snapshot()
    assert state['active'] == original
    assert len(state['worlds']) == 1
    assert len(state['errors']) == 1
    assert 'broken.json' in state['errors'][0]
    assert broken.read_text() == document


def test_texture_reuse_does_not_charge_storage_twice(tmp_path):
    from PIL import Image
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    stream = io.BytesIO()
    Image.new('RGB', (4, 4), 'blue').save(stream, format='PNG')
    data = stream.getvalue()
    resource = worlds.store_image(data)
    worlds.dispatch({"command": "world_object_upsert", "object": {"id": "agent", "resource": resource}})
    with (worlds.assets / 'other-fixture').open('wb') as output:
        output.truncate(64 * 1024 * 1024 - len(data))
    assert worlds.store_image(data) == resource
    stream = io.BytesIO()
    Image.new('RGB', (4, 4), 'red').save(stream, format='PNG')
    with pytest.raises(ValueError, match='storage'):
        worlds.store_image(stream.getvalue())


@pytest.mark.parametrize('world', [[], {'objects': [None]}, {'objects': None}])
def test_invalid_bundled_world_is_explicit_validation_error(tmp_path, world):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    with pytest.raises(ValueError, match='bundled world'):
        worlds.import_bundle({'world': world, 'assets': {}})


def test_rejected_bundle_does_not_leave_orphan_textures_or_change_scene(tmp_path):
    import base64
    import hashlib

    from PIL import Image
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    image = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(image, format="PNG")
    raw = image.getvalue()
    name = hashlib.sha256(raw).hexdigest() + ".image"
    before = worlds.snapshot()["active"]
    invalid = copy.deepcopy(before)
    invalid["objects"][1]["resource"] = "asset:" + name
    invalid["theme"]["exposure"] = 100
    with pytest.raises(ValueError, match="Exposure"):
        worlds.import_bundle({"world": invalid, "assets": {name: base64.b64encode(raw).decode()}})
    assert list(worlds.assets.iterdir()) == []
    valid = copy.deepcopy(before)
    valid["objects"][1]["resource"] = "asset:" + name
    bad_name = hashlib.sha256(b"not an image").hexdigest() + ".image"
    with pytest.raises(ValueError, match="valid bounded image"):
        worlds.import_bundle({"world": valid, "assets": {name: base64.b64encode(raw).decode(), bad_name: base64.b64encode(b"not an image").decode()}})
    assert list(worlds.assets.iterdir()) == []
    assert worlds.snapshot()["active"] == before


def test_partial_object_edits_merge_in_place_and_new_objects_take_defaults(tmp_path):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    worlds.dispatch({"command": "world_object_upsert", "object": {"id": "notes", "kind": "memory", "shape": "cube",
                                                                  "animation": {"kind": "spin"}}})
    notes = worlds.dispatch({"command": "world_get", "object_id": "notes"})["object"]
    assert notes["label"] == "notes" and notes["scale"] == [1, 1, 1] and notes["animation"] == {"kind": "spin", "speed": 1}
    worlds.dispatch({"command": "world_object_upsert", "object": {"id": "chat", "position": [0, 1, 0]}})
    assert [o["id"] for o in worlds.active["objects"]] == ["chat", "agent", "notes"]
    assert worlds.active["objects"][0]["label"] == "Chat" and worlds.active["objects"][0]["position"] == [0, 1, 0]
    with pytest.raises(ValueError, match="needs a kind"):
        worlds.dispatch({"command": "world_object_upsert", "object": {"id": "ghost"}})
    with pytest.raises(ValueError, match="unexpected.*script|Unexpected.*script"):
        worlds.dispatch({"command": "world_object_upsert", "object": {"id": "notes", "script": "x"}})


def test_missing_targets_and_reserved_identities_fail_without_leaking_paths(tmp_path):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    before = json.loads((tmp_path / "worlds/state.json").read_text())
    for request in ({"command": "world_object_remove", "object_id": "nope"},
                    {"command": "world_load", "world_id": "nope"},
                    {"command": "world_export", "world_id": "nope"},
                    {"command": "world_load", "world_id": "state"},
                    {"command": "world_create", "name": "   "}):
        with pytest.raises(ValueError) as error:
            worlds.dispatch(request)
        assert str(tmp_path) not in str(error.value)
    for reserved in ("state", "assets"):
        document = copy.deepcopy(worlds.active)
        document["id"] = reserved
        with pytest.raises(ValueError, match="reserved"):
            worlds.dispatch({"command": "world_save", "world": document})
    assert json.loads((tmp_path / "worlds/state.json").read_text()) == before


def test_style_accepts_partial_changes_and_state_is_compact(tmp_path):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    worlds.dispatch({"command": "world_object_upsert", "object": {"id": "chat", "text": "x" * 4000}})
    result = worlds.dispatch({"command": "world_style", "theme": {"background": "#000000"}})
    assert result["theme"] == {"background": "#000000", "accent": "#38CBFF", "exposure": 1,
                               "environment": "", "environment_intensity": 1, "fog": 0, "floor": "grid"}
    assert result["transitions"] == {"duration": 0.6, "style": "ease"}
    with pytest.raises(ValueError):
        worlds.dispatch({"command": "world_style"})
    state = worlds.dispatch({"command": "world_state"})
    chat = state["active"]["objects"][0]
    assert chat["position"] == [0, 0, 0] and len(chat["text_preview"]) == 160 and "text" not in chat
    assert state["worlds"] == [{"id": worlds.active["id"], "name": "My world"}]


def test_full_texture_store_reclaims_only_unreferenced_images(tmp_path):
    from PIL import Image
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    def png(colour):
        stream = io.BytesIO()
        Image.new("RGB", (4, 4), colour).save(stream, format="PNG")
        return stream.getvalue()
    kept = worlds.store_image(png("blue"))
    orphan = worlds.store_image(png("green"))
    recent = worlds.store_image(png("yellow"))
    profiled = worlds.store_image(png("white"))
    (tmp_path / "user-profiles").mkdir()
    (tmp_path / "user-profiles" / "p.json").write_text(json.dumps({"world": {"resource": profiled}}))
    worlds.dispatch({"command": "world_object_upsert", "object": {"id": "agent", "resource": kept}})
    import os
    old = 0  # Older than the grace period.
    for path in (kept, orphan, profiled):
        os.utime(path, (old, old))
    with (worlds.assets / "other-fixture").open("wb") as output:
        output.truncate(64 * 1024 * 1024 - sum(len(png(c)) for c in ("blue", "yellow", "white", "red")))
    worlds.store_image(png("red"))
    assert not Path(orphan).exists()
    assert Path(kept).exists() and Path(recent).exists() and Path(profiled).exists()


def test_damaged_generated_image_does_not_break_artifact_sync(tmp_path):
    shared = tmp_path / "shared-media"
    shared.mkdir()
    damaged = shared / "bad.png"
    from PIL import Image
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), "blue").save(stream, format="PNG")
    data = bytearray(stream.getvalue())
    data[-20] ^= 0xFF  # corrupt the IDAT checksum
    damaged.write_bytes(bytes(data))
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    job = {"id": "a" * 32, "kind": "media", "state": "completed", "result": {"path": str(damaged)}, "spec": {"capability": "image"}}
    assert worlds.sync_artifacts([job])
    assert worlds.active["objects"][-1]["resource"] == str(damaged)


def test_artifacts_are_represented_once_across_worlds_and_partial_jobs_are_revisited(tmp_path):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    partial = {"id": "b" * 32, "kind": "document", "state": "partial", "result": {"path": "/tmp/one.txt"}}
    worlds.sync_artifacts([partial])
    worlds.sync_artifacts([{**partial, "state": "completed", "result": {"paths": [{"path": "/tmp/one.txt"}, {"path": "/tmp/two.txt"}]}}])
    assert [o["label"] for o in worlds.active["objects"][2:]] == ["one.txt", "two.txt"]
    worlds.dispatch({"command": "world_create", "name": "Fresh"})
    assert not worlds.sync_artifacts([{**partial, "state": "completed"}])
    assert len(worlds.active["objects"]) == 2


@pytest.mark.parametrize("active", [{"name": "   "}, {"id": "state"}, {"objects": []}])
def test_startup_migrates_or_sets_aside_a_rejected_active_world(tmp_path, active):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    path = worlds.home / "state.json"
    document = json.loads(path.read_text())
    document["active"].update(active)
    path.write_text(json.dumps(document))
    reloaded = Worlds(SimpleNamespace(home=tmp_path))
    validate_world(reloaded.active)
    if "objects" in active:
        assert list(worlds.home.glob("state.json.invalid-*")) and reloaded.snapshot()["errors"]


def test_autonomous_save_covers_only_the_active_world_and_saved_worlds_are_capped(tmp_path, monkeypatch):
    from mlx2 import desktop_world
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    monkeypatch.setattr(desktop_world, "MAX_SAVED_WORLDS", 2)
    worlds.dispatch({"command": "world_create", "name": "Two"})
    with pytest.raises(ValueError, match="At most 2"):
        worlds.dispatch({"command": "world_create", "name": "Three"})


def test_removed_artifact_objects_stay_removed_even_for_partial_jobs(tmp_path):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    job = {"id": "c" * 32, "kind": "document", "state": "partial", "result": {"path": "/tmp/part.txt"}}
    worlds.sync_artifacts([job])
    identity = worlds.active["objects"][-1]["id"]
    worlds.dispatch({"command": "world_object_remove", "object_id": identity})
    worlds.sync_artifacts([job])
    assert all(o["id"] != identity for o in worlds.active["objects"])
    worlds.dispatch({"command": "world_create", "name": "Other"})
    worlds.sync_artifacts([job])
    assert len(worlds.active["objects"]) == 2


def test_saved_and_bundled_worlds_from_earlier_releases_still_load(tmp_path):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    legacy = default_world("   ")  # Earlier releases accepted blank names.
    (worlds.home / (legacy["id"] + ".json")).write_text(json.dumps(legacy))
    assert worlds.dispatch({"command": "world_load", "world_id": legacy["id"]})["active"]["name"] == "My world"
    imported = worlds.dispatch({"command": "world_import", "document": {"world": default_world(" "), "assets": {}}})
    assert imported["active"]["name"] == "My world"


def test_import_is_checked_before_approval_without_storing_textures(tmp_path):
    worlds = Worlds(SimpleNamespace(home=tmp_path))
    with pytest.raises(ValueError, match="incomplete"):
        world = default_world()
        world["objects"][1]["resource"] = "asset:" + "0" * 64 + ".image"
        worlds.check({"command": "world_import", "document": {"world": world, "assets": {}}})
    assert not list(worlds.assets.iterdir())
