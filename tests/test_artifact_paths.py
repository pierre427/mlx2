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
