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
    with pytest.raises(ValueError, match="explicitly boolean"):
        Worlds(SimpleNamespace(home=tmp_path))


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
