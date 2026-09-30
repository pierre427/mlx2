"""Flash-Next resolution keeps Hugging Face snapshot layouts, like standard_decoder."""

import json
import os

import pytest

from mlx2.adapters.registry import resolve_adapter
from test_adapter_registry import artifact


def test_symlinked_snapshot_shard_is_accepted(tmp_path):
    snapshot = tmp_path / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "sha256-deadbeef").write_bytes(b"not loaded")
    path = artifact(snapshot, {"model_type": "qwen4_exp"}, ["mtp.fc.weight"])
    (path / "ple_rows.bin").write_bytes(b"not loaded")
    (path / "model.safetensors").unlink()
    os.symlink("../../blobs/sha256-deadbeef", path / "model.safetensors")
    assert resolve_adapter(path, mtp=True).__name__ == "FlashNextAdapter"


@pytest.mark.parametrize("name", ["../model.safetensors", "sub/model.safetensors", "model.bin"])
def test_traversal_and_foreign_shard_names_are_refused(tmp_path, name):
    path = artifact(tmp_path, {"model_type": "qwen4_exp"}, ["mtp.fc.weight"])
    (path / "ple_rows.bin").write_bytes(b"not loaded")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc.weight": name}})
    )
    with pytest.raises(ValueError, match="stay within the artifact"):
        resolve_adapter(path, mtp=True)
