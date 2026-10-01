"""Full source schema reconciliation without an MLX import or GPU access."""

import json
import math
import os
import subprocess
import sys
from dataclasses import asdict

import pytest

from mlx2.adapters.xpress import (
    XPressConfig,
    content_revision,
    expected_weight_shapes,
    inspect_drafter,
    load_drafter,
)


@pytest.fixture
def artifact(tmp_path):
    args = XPressConfig(
        hidden_size=4,
        intermediate_size=7,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        vocab_size=9,
        mask_token_id=8,
        num_target_layers=3,
        target_layer_ids=[0, 2],
        block_size=4,
        layer_types=["full_attention"],
        xpress_rank=3,
        xpress_mlp_hidden=5,
        xpress_num_passes=3,
    )
    config = asdict(args)
    config.update(
        architectures=["Qwen3XPressModel"],
        model_type="qwen3",
        dtype="bfloat16",
        xpress_block_size=4,
        dflash_config={"mask_token_id": 8, "target_layer_ids": [0, 2]},
    )
    config.pop("mask_token_id")
    config.pop("target_layer_ids")
    draft, target = tmp_path / "draft", tmp_path / "target"
    draft.mkdir()
    target.mkdir()
    (draft / "config.json").write_text(json.dumps(config))
    target_config = {
        key: config[key]
        for key in (
            "model_type",
            "hidden_size",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "rope_theta",
            "rms_norm_eps",
        )
    }
    target_config["num_hidden_layers"] = 3
    (target / "config.json").write_text(json.dumps(target_config))
    header, end = {}, 0
    for name, shape in expected_weight_shapes(args).items():
        start, end = end, end + 2 * math.prod(shape)
        header[name] = {"shape": shape, "dtype": "BF16", "data_offsets": [start, end]}
    write_weights(draft, header)
    return draft, target, config, header


def write_weights(draft, header, tail=0):
    raw = json.dumps(header).encode()
    size = max(v["data_offsets"][1] for v in header.values())
    (draft / "model.safetensors").write_bytes(
        len(raw).to_bytes(8, "little") + raw + bytes(size + tail)
    )


def test_metadata_inspector_never_imports_mlx(artifact):
    draft, target, _, _ = artifact
    code = f"""import builtins
original = builtins.__import__
def guarded(name, *a, **kw):
    if name == 'mlx' or name.startswith('mlx.'):
        raise AssertionError('MLX import during metadata inspection')
    return original(name, *a, **kw)
builtins.__import__ = guarded
from mlx2.adapters.xpress import inspect_drafter
record = inspect_drafter({str(draft)!r}, {str(target)!r})
assert len(record['weight_sha256'][0]) == 64
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())


@pytest.mark.parametrize("indexed", [False, True])
def test_snapshot_symlink_and_content_pin(artifact, tmp_path, indexed):
    draft, target, _, header = artifact
    blob = tmp_path / "blob"
    (draft / "model.safetensors").rename(blob)
    (draft / "model.safetensors").symlink_to(blob)
    if indexed:
        (draft / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: "model.safetensors" for k in header}})
        )
    first = inspect_drafter(draft, target)
    old = content_revision(first)
    with blob.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\x01")
    new = inspect_drafter(draft, target)
    assert content_revision(new) != old
    with pytest.raises(ValueError, match="changed since inspection"):
        load_drafter(first, object())


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "shape", "dtype", "overlap", "tail"]
)
def test_source_schema_fail_closed(artifact, mutation):
    draft, target, _, header = artifact
    if mutation == "missing":
        header.pop("fc.weight")
    elif mutation == "extra":
        header["oops"] = header["norm.weight"]
    elif mutation == "shape":
        header["fc.weight"]["shape"] = [4, 4]
    elif mutation == "dtype":
        header["fc.weight"]["dtype"] = "F16"
    elif mutation == "overlap":
        record = header["hidden_norm.weight"]
        size = record["data_offsets"][1] - record["data_offsets"][0]
        record["data_offsets"] = [0, size]
    write_weights(draft, header, tail=2 if mutation == "tail" else 0)
    with pytest.raises(ValueError):
        inspect_drafter(draft, target)


@pytest.mark.parametrize(
    "key,value",
    [
        ("xpress_block_size", 3),
        ("xpress_num_passes", True),
        ("xpress_rank", 0),
        ("sample_from_anchor", True),
        ("unknown_head_variant", True),
        ("rope_scaling", {"factor": 2}),
        ("draft_vocab_size", 8),
    ],
)
def test_config_fail_closed(artifact, key, value):
    draft, target, config, _ = artifact
    config[key] = value
    (draft / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        inspect_drafter(draft, target)


@pytest.mark.parametrize(
    "key", ["hidden_size", "vocab_size", "num_hidden_layers", "head_dim", "rope_theta"]
)
def test_target_geometry_fail_closed(artifact, key):
    draft, target, _, _ = artifact
    text = json.loads((target / "config.json").read_text())
    text[key] += 1
    (target / "config.json").write_text(json.dumps(text))
    with pytest.raises(ValueError, match="target"):
        inspect_drafter(draft, target)


def test_duplicate_json_and_index_path_rejection(artifact):
    draft, target, config, header = artifact
    (draft / "config.json").write_text(json.dumps(config)[:-1] + ', "xpress_rank": 3}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        inspect_drafter(draft, target)
    (draft / "config.json").write_text(json.dumps(config))
    (draft / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "../model.safetensors" for k in header}})
    )
    with pytest.raises(ValueError, match="shard path"):
        inspect_drafter(draft, target)


@pytest.mark.parametrize("passes", [0, True, 5])
def test_load_options_rejected_before_mlx(artifact, passes):
    draft, target, _, _ = artifact
    record = inspect_drafter(draft, target)
    with pytest.raises(ValueError, match="num_passes"):
        load_drafter(record, object(), num_passes=passes)


def test_strict_tiny_bf16_source_load_and_pass_override(artifact):
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx import nn

    mx.set_default_device(mx.cpu)
    draft, target, _, _ = artifact
    record = inspect_drafter(draft, target)
    embedding = nn.Embedding(9, 4)
    target_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding))
    model = load_drafter(record, target_model, num_passes=2)
    assert record["args"].xpress_num_passes == 3
    assert model.config.xpress_num_passes == 2
    np_mix = __import__("numpy").asarray(model.xpress_head.mix_L.astype(mx.float32))
    __import__("numpy").testing.assert_array_equal(
        np_mix, __import__("numpy").broadcast_to(__import__("numpy").eye(4), (3, 4, 4))
    )
    tokens, laws = model.draft_distributions(
        [1],
        mx.zeros((1, 0, 8)),
        model.make_cache(),
        3,
        [object()],
        [0.0],
        processor_histories=[[]],
    )
    assert tokens == [[0, 0, 0]]
    assert all(q[0] == 1 for q in laws[0])


def test_attention_override_validated_before_allocation_and_source_unchanged(artifact):
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx import nn

    mx.set_default_device(mx.cpu)
    draft, target, _, _ = artifact
    record = inspect_drafter(draft, target)
    for windows in ([], [True], [0]):
        with pytest.raises(ValueError, match="draft_attention_windows"):
            load_drafter(record, object(), draft_attention_windows=windows)
    model = load_drafter(
        record,
        SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(9, 4))),
        draft_attention_windows=[2],
    )
    assert record["args"].layer_types == ["full_attention"]
    assert model.make_cache()[0].max_size == 2
    assert model.receipt_settings["draft_attention_windows"] == [2]
