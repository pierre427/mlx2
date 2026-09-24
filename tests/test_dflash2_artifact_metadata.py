"""DFlash2 artifact closure without importing or allocating MLX tensors."""

import json
import math
from dataclasses import asdict

import pytest

from mlx2.adapters.dflash2 import _expected_weight_shapes, inspect_drafter
from mlx2.runtime.drafters.dflash2_config import DFlash2Config


@pytest.fixture
def draft_artifact(tmp_path):
    args = DFlash2Config(
        hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=32, num_target_layers=4, target_layer_ids=[0, 3],
        conv_kernel_size=2, conv_group_size=2, selector_rank=4,
        selector_top_k=4, block_size=4, mask_token_id=31,
        max_position_embeddings=128,
    )
    target, draft = tmp_path / "target", tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    config = asdict(args)
    config.update(architectures=["DFlash2DraftModel"], dtype="bfloat16", dflash_config={})
    (draft / "config.json").write_text(json.dumps(config))
    (target / "config.json").write_text(json.dumps({
        "hidden_size": args.hidden_size, "vocab_size": args.vocab_size,
        "num_hidden_layers": args.num_target_layers,
    }))
    header, end = {}, 0
    for name, shape in _expected_weight_shapes(args).items():
        start, end = end, end + 2 * math.prod(shape)
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, end]}
    raw = json.dumps(header).encode()
    (draft / "model.safetensors").write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(end))
    return draft, target, header


@pytest.mark.parametrize("indexed", [False, True])
def test_snapshot_shard_symlinks_keep_lexical_names(draft_artifact, tmp_path, indexed):
    draft, target, header = draft_artifact
    shard = draft / "model.safetensors"
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    blob = blobs / "content-hash-without-extension"
    shard.rename(blob)
    shard.symlink_to(blob)
    if indexed:
        (draft / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: shard.name for name in header},
        }))
    record = inspect_drafter(draft, target)
    assert record["files"][0][0] == "model.safetensors"
    assert len(record["header_sha256"]) == 1


@pytest.mark.parametrize("name", ["../model.safetensors", "/outside/model.safetensors", "nested/model.safetensors", 1, ["model.safetensors"]])
def test_index_still_rejects_nonlocal_shard_names(draft_artifact, name):
    draft, target, header = draft_artifact
    (draft / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {tensor: name for tensor in header},
    }))
    with pytest.raises(ValueError, match="shard path"):
        inspect_drafter(draft, target)
