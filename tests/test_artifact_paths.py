"""Weight shards may live in a Hub snapshot's own blob store, nowhere else."""

import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

from mlx2.adapters.artifact_paths import (
    hub_blob_identity,
    hub_snapshot_revision,
    shard_within_artifact,
)


def test_snapshot_blobs_belong_to_the_same_repository(tmp_path):
    repo = tmp_path / "models--org--name"
    revision = "a" * 40
    snapshot = repo / "snapshots" / revision
    (repo / "blobs").mkdir(parents=True)
    snapshot.mkdir(parents=True)
    assert shard_within_artifact(snapshot, snapshot / "model.safetensors")
    assert shard_within_artifact(snapshot, repo / "blobs" / "abc")
    assert not shard_within_artifact(snapshot, tmp_path / "elsewhere" / "abc")
    assert not shard_within_artifact(snapshot, tmp_path / "models--other" / "blobs" / "abc")
    plain = tmp_path / "plain"
    assert not shard_within_artifact(plain, tmp_path / "blobs" / "abc")
    assert hub_snapshot_revision(snapshot) == revision
    assert hub_blob_identity(snapshot, repo / "blobs" / ("b" * 64)) == "b" * 64
    assert hub_blob_identity(snapshot, repo / "blobs" / "not-a-hash") is None
    assert hub_snapshot_revision(plain) is None


def test_candidate_fingerprint_binds_content_addressed_blob_identity(tmp_path):
    from mlx2.decisions.candidates.base import inspect_index

    repo = tmp_path / "models--org--decision"
    snapshot = repo / "snapshots" / ("a" * 40)
    blobs = repo / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (snapshot / "config.json").write_text("{}")
    index = snapshot / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"model.weight": "model.safetensors"}}))

    def inspect(blob_name):
        blob = blobs / blob_name
        blob.write_bytes(b"12345678")
        shard = snapshot / "model.safetensors"
        if shard.exists() or shard.is_symlink():
            shard.unlink()
        shard.symlink_to(os.path.relpath(blob, snapshot))
        return inspect_index(
            snapshot,
            index,
            metadata_paths=(snapshot / "config.json", index),
            allowed_prefixes=("model.",),
            required_prefixes=("model.",),
        )["identity"]

    first = inspect("b" * 64)
    second = inspect("c" * 64)
    assert first["fingerprint_kind"] == "hub-blob-identity"
    assert first["fingerprint"] != second["fingerprint"]


def test_granite_inspects_from_a_hub_cache_snapshot(tmp_path):
    """Snapshot entries are symlinks into ../../blobs; Granite (and the Agnes,
    HY V3, LLaDA, HiLS and Nemotron 3 Super gates) refused them as missing."""
    from mlx2.adapters import granite_swa

    source = Path("~/mlx-models/granite-swash-3b-a600m").expanduser()
    if not (source / "model.safetensors").is_file():
        pytest.skip("optional local Granite pack is absent")
    repo = tmp_path / "models--ibm-granite--granite-swash"
    blobs, snapshot = repo / "blobs", repo / "snapshots" / "rev0"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)

    def link(name, data):
        blob = blobs / hashlib.sha1(data).hexdigest()
        blob.write_bytes(data)
        (snapshot / name).symlink_to(os.path.relpath(blob, snapshot))

    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if (source / name).is_file():
            link(name, (source / name).read_bytes())
    with (source / "model.safetensors").open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        link("model.safetensors", struct.pack("<Q", size) + stream.read(size))
    receipt = granite_swa.inspect_artifact(snapshot)
    assert [name for name, *_ in receipt["identity"]["files"]] == ["model.safetensors"]


def _as_hub_snapshot(flat, repo):
    """Move a flat artifact into ``repo/blobs`` behind ``repo/snapshots/rev0`` links."""
    blobs, snapshot = repo / "blobs", repo / "snapshots" / "rev0"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    for item in sorted(flat.iterdir()):
        blob = blobs / hashlib.sha256(item.name.encode()).hexdigest()
        item.rename(blob)  # keeps sparse shard fixtures sparse
        (snapshot / item.name).symlink_to(os.path.relpath(blob, snapshot))
    return snapshot


MUSE_CONFIG = b'{"model_type": "muse_glimmer"}'


@pytest.mark.parametrize("sharded", [False, True])
def test_muse_inspects_from_a_hub_cache_snapshot(tmp_path, sharded):
    """Muse resolved each shard and required the blob to sit inside the
    snapshot with a .safetensors suffix, so every snapshot was refused."""
    from mlx2.adapters import muse_glimmer, registry

    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "config.json").write_bytes(MUSE_CONFIG)
    shards = ["model.safetensors"]
    if sharded:
        shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        (flat / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}}))
    for name in shards:
        (flat / name).write_bytes(name.encode())
    snapshot = _as_hub_snapshot(flat, tmp_path / "models--org--Muse-Glimmer")
    receipt = muse_glimmer.inspect_artifact(snapshot)
    assert [name for name, *_ in receipt["files"]] == shards
    assert registry.inspect_model(snapshot).artifact["files"] == receipt["files"]


def test_muse_still_refuses_shards_outside_the_repository(tmp_path):
    from mlx2.adapters import muse_glimmer

    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    (foreign / "model.safetensors").write_bytes(b"foreign")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_bytes(MUSE_CONFIG)
    (artifact / "model.safetensors").symlink_to(foreign / "model.safetensors")
    with pytest.raises(ValueError, match="local safetensors"):
        muse_glimmer.inspect_artifact(artifact)
    # Another repository's blob store is foreign as well.
    other = tmp_path / "other"
    other.mkdir()
    (other / "model.safetensors").write_bytes(b"other")
    other = _as_hub_snapshot(other, tmp_path / "models--other")
    snapshot = tmp_path / "models--x" / "snapshots" / "rev0"
    snapshot.mkdir(parents=True)
    (tmp_path / "models--x" / "blobs").mkdir()
    (snapshot / "config.json").write_bytes(MUSE_CONFIG)
    (snapshot / "model.safetensors").symlink_to((other / "model.safetensors").resolve())
    with pytest.raises(ValueError, match="local safetensors"):
        muse_glimmer.inspect_artifact(snapshot)


def test_xing_inspects_from_a_hub_cache_snapshot(tmp_path):
    from test_xing_adapter import _artifact

    from mlx2.adapters import xing

    flat = xing.inspect_artifact(_artifact(tmp_path / "flat"))
    snapshot = _as_hub_snapshot(_artifact(tmp_path / "moved"), tmp_path / "models--org--Xing")
    receipt = xing.inspect_artifact(snapshot)
    assert [name for name, *_ in receipt["identity"]["files"]] == [
        name for name, *_ in flat["identity"]["files"]
    ]


def test_north_inspects_from_a_hub_cache_snapshot(tmp_path):
    from test_north_mini_code_port import artifact

    from mlx2.adapters import north_mini_code

    flat = tmp_path / "flat"
    flat.mkdir()
    snapshot = _as_hub_snapshot(artifact(flat), tmp_path / "models--org--North")
    receipt = north_mini_code.inspect_artifact(snapshot)
    assert [name for name, *_ in receipt["identity"]["files"]] == ["model.safetensors"]
