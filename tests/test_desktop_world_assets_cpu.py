"""World schema v2 (parts, materials, lights, sky) and the Poly Haven asset cache. CPU only, no network."""
import copy
import hashlib
import json
import os
import time
from types import SimpleNamespace

import pytest

from mlx2 import desktop_world_assets as assets
from mlx2.desktop_world import Worlds, compact_state, migrate_world, validate_world

USDC = b"PXR-USDC" + b"\0" * 64
JPG = b"\xff\xd8\xff\xe0fake-jpeg"
HDR = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n\0\0\0\0"
CATALOGUE = {
    "model": {"jacaranda_tree": {"name": "Jacaranda Tree", "tags": ["tree", "purple"], "categories": ["nature", "trees"],
                                 "authors": {"Someone": "All"}, "polycount": 312356, "dimensions": [9000, 8000, 7000]},
              "fir_tree_01": {"name": "Fir Tree 01", "tags": ["tree"], "categories": ["trees"], "authors": {}, "polycount": 7853731},
              "ArmChair_01": {"name": "Arm Chair 01", "tags": ["chair"], "categories": ["furniture"], "authors": {}, "polycount": 5626}},
    "hdri": {"sunny_sky": {"name": "Sunny Sky", "tags": ["sky", "clear"], "categories": ["outdoor"], "authors": {}}},
}


def entry(data, url):
    return {"url": url, "size": len(data), "md5": hashlib.md5(data).hexdigest()}


@pytest.fixture
def network(monkeypatch):
    files = {"https://dl.polyhaven.org/m.usdc": USDC, "https://dl.polyhaven.org/d.jpg": JPG, "https://dl.polyhaven.org/s.hdr": HDR}
    listings = {
        "jacaranda_tree": {"usd": {"1k": {"usd": {**entry(USDC, "https://dl.polyhaven.org/m.usdc"),
                                                  "include": {"textures/bark_diff_1k.jpg": entry(JPG, "https://dl.polyhaven.org/d.jpg")}}}}},
        "sunny_sky": {"hdri": {"2k": {"hdr": entry(HDR, "https://dl.polyhaven.org/s.hdr")}}},
    }
    calls = []

    def get(url, limit, hosts, timeout=30):
        calls.append(url)
        assert assets._allowed(url, hosts)
        if "/assets?t=" in url:
            kind = "model" if url.endswith("models") else "hdri"
            return json.dumps(CATALOGUE[kind]).encode()
        if "/files/" in url:
            return json.dumps(listings[url.rsplit("/", 1)[1]]).encode()
        return files[url]
    monkeypatch.setattr(assets, "_get", get)
    return SimpleNamespace(files=files, listings=listings, calls=calls)


def worlds(tmp_path):
    return Worlds(SimpleNamespace(home=tmp_path))


def test_version_one_documents_migrate_and_render_unchanged(tmp_path):
    world = worlds(tmp_path)
    old = copy.deepcopy(world.active)
    old["schema_version"] = 1
    for key in ("environment", "environment_intensity", "fog", "floor"):
        del old["theme"][key]
    for obj in old["objects"]:
        del obj["parent"], obj["material"], obj["light"]
    upgraded = validate_world(migrate_world(old))
    assert upgraded["schema_version"] == 2 and upgraded["theme"]["floor"] == "grid"
    assert all(o["parent"] == "" and o["material"]["opacity"] == 1 for o in upgraded["objects"])
    # world_save accepts a version 1 document from a model that read old docs.
    old["id"] = world.active["id"]
    assert world.dispatch({"command": "world_save", "world": old})["active"]["schema_version"] == 2


def test_parts_lights_and_materials_validate(tmp_path):
    world = worlds(tmp_path)
    world.dispatch({"command": "world_object_upsert", "object": {"id": "trunk", "kind": "media", "shape": "cylinder", "color": "#5A3A22"}})
    world.dispatch({"command": "world_object_upsert", "object": {"id": "crown", "kind": "media", "shape": "sphere", "parent": "trunk",
                                                                  "material": {"roughness": 0.9}}})
    crown = next(o for o in world.active["objects"] if o["id"] == "crown")
    assert crown["material"] == {"roughness": 0.9, "metalness": 0.2, "emission": 0.12, "opacity": 1, "tiling": 1}
    with pytest.raises(ValueError, match="parent"):
        world.dispatch({"command": "world_object_upsert", "object": {"id": "trunk", "parent": "crown"}})
    with pytest.raises(ValueError, match="does not exist"):
        world.dispatch({"command": "world_object_upsert", "object": {"id": "leaf", "kind": "media", "parent": "nope"}})
    with pytest.raises(ValueError, match="opacity"):
        world.dispatch({"command": "world_object_upsert", "object": {"id": "crown", "material": {"opacity": 0}}})
    for index in range(8):
        world.dispatch({"command": "world_object_upsert", "object": {"id": f"l{index}", "kind": "media", "shape": "light",
                                                                      "light": {"type": "spot", "intensity": 800}}})
    with pytest.raises(ValueError, match="at most 8 light"):
        world.dispatch({"command": "world_object_upsert", "object": {"id": "l9", "kind": "media", "shape": "light"}})
    state = compact_state(world.snapshot())
    rows = {o["id"]: o for o in state["active"]["objects"]}
    assert "material" not in rows["trunk"] and rows["crown"]["material"]["roughness"] == 0.9
    assert rows["crown"]["parent"] == "trunk" and rows["l0"]["light"]["type"] == "spot" and "light" not in rows["trunk"]


def test_model_references_need_a_download_and_never_name_paths(tmp_path, network):
    world = worlds(tmp_path)
    with pytest.raises(ValueError, match="needs resource polyhaven:model"):
        world.dispatch({"command": "world_object_upsert", "object": {"id": "tree", "kind": "media", "shape": "model"}})
    with pytest.raises(ValueError, match="not downloaded"):
        world.check({"command": "world_object_upsert", "object": {"id": "tree", "kind": "media", "shape": "model",
                                                                  "resource": "polyhaven:model/jacaranda_tree"}})
    with pytest.raises(ValueError):
        world.dispatch({"command": "world_object_upsert", "object": {"id": "tree", "kind": "media", "shape": "model",
                                                                     "resource": "polyhaven:model/../../etc"}})
    with pytest.raises(ValueError, match="theme environment"):
        world.dispatch({"command": "world_object_upsert", "object": {"id": "sky", "kind": "media", "resource": "polyhaven:hdri/sunny_sky"}})
    fetched = world.dispatch({"command": "world_model_fetch", "kind": "model", "asset_id": "jacaranda_tree"})
    assert fetched["resource"] == "polyhaven:model/jacaranda_tree" and fetched["license"] == "CC0"
    root = tmp_path / "worlds/assets/polyhaven/models/jacaranda_tree"
    assert (root / "model.usdc").read_bytes() == USDC and (root / "textures/bark_diff_1k.jpg").read_bytes() == JPG
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["files"]["model.usdc"] == hashlib.sha256(USDC).hexdigest()
    world.dispatch({"command": "world_object_upsert", "object": {"id": "tree", "kind": "media", "shape": "model",
                                                                 "resource": fetched["resource"], "position": [0, -1.8, -3]}})
    row = next(o for o in compact_state(world.snapshot())["active"]["objects"] if o["id"] == "tree")
    assert row["resource"] == "polyhaven:model/jacaranda_tree"
    # A second fetch reuses the cache.
    count = len(network.calls)
    world.dispatch({"command": "world_model_fetch", "kind": "model", "asset_id": "jacaranda_tree"})
    assert not any("/files/" in url for url in network.calls[count:])
    sky = world.dispatch({"command": "world_model_fetch", "kind": "hdri", "asset_id": "sunny_sky"})
    result = world.dispatch({"command": "world_style", "theme": {"environment": sky["resource"], "fog": 40}})
    assert result["theme"]["environment"] == "polyhaven:hdri/sunny_sky"
    # An imported scene that names an asset not downloaded here does not block other edits.
    world.active["objects"].append({**copy.deepcopy(world.active["objects"][1]), "id": "imported", "shape": "model",
                                    "resource": "polyhaven:model/ArmChair_01"})
    world.dispatch({"command": "world_object_upsert", "object": {"id": "chat", "label": "Still editable"}})


def test_search_is_local_and_filters_heavy_models(tmp_path, network):
    world = worlds(tmp_path)
    found = world.dispatch({"command": "world_model_search", "query": "tree", "kind": "model", "limit": 6})
    assert [r["asset_id"] for r in found["results"]] == ["jacaranda_tree"]
    assert found["results"][0]["size_m"] == [9.0, 8.0, 7.0] and found["results"][0]["downloaded"] is False
    assert all("tree" not in url for url in network.calls)  # The query never leaves the machine.
    with pytest.raises(ValueError, match="too detailed"):
        world.dispatch({"command": "world_model_fetch", "kind": "model", "asset_id": "fir_tree_01"})
    with pytest.raises(ValueError, match="not in the Poly Haven catalogue"):
        world.dispatch({"command": "world_model_fetch", "kind": "model", "asset_id": "made_up"})


@pytest.mark.parametrize("damage", ["md5", "size", "host", "path", "magic"])
def test_fetch_refuses_damaged_or_unexpected_files(tmp_path, network, damage):
    listing = network.listings["jacaranda_tree"]["usd"]["1k"]["usd"]
    if damage == "md5":
        listing["md5"] = "0" * 32
    elif damage == "size":
        listing["size"] += 1
    elif damage == "host":
        listing["url"] = "https://evil.example/m.usdc"
    elif damage == "path":
        listing["include"]["../../escape.jpg"] = listing["include"].pop("textures/bark_diff_1k.jpg")
    else:
        network.files["https://dl.polyhaven.org/m.usdc"] = b"NOT-USDC" + USDC[8:]
    world = worlds(tmp_path)
    with pytest.raises((ValueError, AssertionError)):
        world.dispatch({"command": "world_model_fetch", "kind": "model", "asset_id": "jacaranda_tree"})
    base = tmp_path / "worlds/assets/polyhaven"
    assert not (base / "models/jacaranda_tree").exists()
    assert not list(base.glob(".fetch-*")) and not (tmp_path / "escape.jpg").exists()


def test_reclaim_keeps_referenced_and_recent_assets(tmp_path, network):
    world = worlds(tmp_path)
    world.dispatch({"command": "world_model_fetch", "kind": "model", "asset_id": "jacaranda_tree"})
    world.dispatch({"command": "world_model_fetch", "kind": "hdri", "asset_id": "sunny_sky"})
    world.dispatch({"command": "world_style", "theme": {"environment": "polyhaven:hdri/sunny_sky"}})
    assert world.polyhaven.reclaim() == 0  # Both are within the grace period.
    old = time.time() - 3 * 24 * 3600
    for path in (tmp_path / "worlds/assets/polyhaven").rglob("*"):
        os.utime(path, (old, old))
    assert world.polyhaven.reclaim() == 1  # The unreferenced model goes; the sky in use stays.
    assert not world.polyhaven.cached("model", "jacaranda_tree") and world.polyhaven.cached("hdri", "sunny_sky")


def test_removing_a_parent_removes_its_parts(tmp_path):
    world = worlds(tmp_path)
    for identity, parent in (("table", ""), ("leg", "table"), ("foot", "leg"), ("lamp", "")):
        world.dispatch({"command": "world_object_upsert", "object": {"id": identity, "kind": "media", "shape": "cube", "parent": parent}})
    world.dispatch({"command": "world_object_remove", "object_id": "table"})
    assert {o["id"] for o in world.active["objects"]} == {"chat", "agent", "lamp"}


def test_cached_fetch_needs_no_network_and_stale_staging_is_cleaned(tmp_path, network, monkeypatch):
    world = worlds(tmp_path)
    world.dispatch({"command": "world_model_fetch", "kind": "model", "asset_id": "jacaranda_tree"})
    world.polyhaven.catalogues.clear()

    def offline(*args, **kwargs):
        raise OSError("offline")
    monkeypatch.setattr(assets, "_get", offline)
    again = world.dispatch({"command": "world_model_fetch", "kind": "model", "asset_id": "jacaranda_tree"})
    assert again["name"] == "Jacaranda Tree" and again["authors"] == ["Someone"]
    staging = tmp_path / "worlds/assets/polyhaven/.fetch-dead"
    staging.mkdir()
    old = time.time() - 7200
    os.utime(staging, (old, old))
    world.polyhaven.reclaim()
    assert not staging.exists()
