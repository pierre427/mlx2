"""Exact trained-head and convolution tensor closure, CPU metadata only."""

import json
import math
import os
import subprocess
import sys
from dataclasses import asdict

import pytest

from mlx2.adapters.lilicorr import (
    LiLiCorrConfig,
    expected_weight_shapes,
    inspect_drafter,
    load_drafter,
)


@pytest.fixture(params=[False, True])
def artifact(tmp_path, request):
    args = LiLiCorrConfig(
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
        lilicorr_hidden_size=6,
        lilicorr_candidate_topk=2,
        lilicorr_num_layers=2,
        lilicorr_num_heads=2,
        lilicorr_mlp_ratio=1.5,
        lilicorr_factor_dim=3,
        conv_kernel_size=2 if request.param else 0,
        conv_group_size=2 if request.param else 0,
    )
    cfg = asdict(args)
    nested = {}
    for key in list(cfg):
        if key.startswith("lilicorr_") or key in (
            "mask_token_id",
            "target_layer_ids",
            "conv_kernel_size",
            "conv_group_size",
        ):
            nested[key] = cfg.pop(key)
    cfg.update(
        architectures=["LiLiCorrDraftModel"],
        model_type="qwen3",
        dtype="bfloat16",
        dflash_config=nested,
    )
    draft, target = tmp_path / "draft", tmp_path / "target"
    draft.mkdir()
    target.mkdir()
    (draft / "config.json").write_text(json.dumps(cfg))
    text = {
        k: cfg[k]
        for k in (
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
    text["num_hidden_layers"] = 3
    (target / "config.json").write_text(json.dumps(text))
    header, end = {}, 0
    for name, shape in expected_weight_shapes(args).items():
        start, end = end, end + 2 * math.prod(shape)
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, end]}
    write_shard(draft, header)
    return draft, target, cfg, header


def write_shard(draft, header):
    raw = json.dumps(header).encode()
    end = max(v["data_offsets"][1] for v in header.values())
    (draft / "model.safetensors").write_bytes(
        len(raw).to_bytes(8, "little") + raw + bytes(end)
    )


def test_inspection_without_mlx_import(artifact):
    draft, target, _, _ = artifact
    code = f"""import builtins
original=builtins.__import__
def guarded(name,*a,**kw):
    if name=='mlx' or name.startswith('mlx.'):
        raise AssertionError('MLX import from metadata')
    return original(name,*a,**kw)
builtins.__import__=guarded
from mlx2.adapters.lilicorr import inspect_drafter
record=inspect_drafter({str(draft)!r},{str(target)!r})
assert record['args'].lilicorr_candidate_topk==2
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ.copy())


@pytest.mark.parametrize(
    "change",
    [
        "missing_head",
        "extra_head",
        "shape",
        "convolution_drop",
        "convolution_config_drop",
    ],
)
def test_strict_head_and_backbone_closure(artifact, change):
    draft, target, cfg, header = artifact
    if change == "missing_head":
        header.pop("lilicorr.out_head.weight")
    elif change == "extra_head":
        header["lilicorr.untrained_projection.weight"] = header[
            "lilicorr.out_head.weight"
        ]
    elif change == "shape":
        header["lilicorr.slot_embedding"]["shape"] = [1, 1, 2, 1, 6]
    elif change == "convolution_drop":
        if not cfg["dflash_config"]["conv_kernel_size"]:
            return
        header.pop("layers.0.attention_conv.base_kernel")
    elif change == "convolution_config_drop":
        if not cfg["dflash_config"]["conv_kernel_size"]:
            return
        cfg["dflash_config"]["conv_kernel_size"] = cfg["dflash_config"][
            "conv_group_size"
        ] = 0
        (draft / "config.json").write_text(json.dumps(cfg))
    write_shard(draft, header)
    with pytest.raises(ValueError):
        inspect_drafter(draft, target)


@pytest.mark.parametrize(
    "key,value",
    [
        ("lilicorr_candidate_topk", 3),
        ("lilicorr_hidden_size", 5),
        ("lilicorr_num_layers", 0),
        ("lilicorr_vector_eps", 0),
        ("lilicorr_enabled", False),
        ("untrained_head_variant", True),
    ],
)
def test_config_variant_rejection(artifact, key, value):
    draft, target, cfg, _ = artifact
    cfg["dflash_config"][key] = value
    (draft / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        inspect_drafter(draft, target)


def test_tiny_strict_bf16_load_is_candidate_only(artifact):
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx import nn

    mx.set_default_device(mx.cpu)
    draft, target, _, _ = artifact
    record = inspect_drafter(draft, target)
    model = load_drafter(
        record, SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(9, 4)))
    )
    tokens, laws = model.draft_distributions(
        [1], mx.zeros((1, 0, 8)), model.make_cache(), 3, [object()], [1.0]
    )
    assert len(tokens[0]) == 3
    assert model.receipt_kind == "external_lilicorr"
    assert model.receipt_settings["co_trained_head_required"]
    assert all(q[t] == 1 for q, t in zip(laws[0], tokens[0]))
    with pytest.raises(ValueError, match="quantization"):
        load_drafter(
            record, object(), runtime_quantization={"bits": 4, "group_size": 32}
        )


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
