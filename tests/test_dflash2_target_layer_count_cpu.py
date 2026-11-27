"""A DFlash/DFlash2 config without ``num_target_layers`` takes the target's
count, never an assumed 32 (mlx-vlm #2841).  CPU only: no MLX tensors."""

import json
import math
from dataclasses import asdict

import pytest

from mlx2.adapters.dflash2 import _expected_weight_shapes, inspect_drafter
from mlx2.runtime.drafters.base_config import DFlashConfig
from mlx2.runtime.drafters.dflash2_config import DFlash2Config

TAPS = [1, 6, 11, 16, 22, 27, 32, 37]  # mlx-community Qwen3.6-35B-A3B-DFlash2-4bit


def _args(**overrides):
    values = dict(
        hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=32, num_target_layers=40, target_layer_ids=list(TAPS),
        conv_kernel_size=2, conv_group_size=2, selector_rank=4,
        selector_top_k=4, block_size=4, mask_token_id=31,
        max_position_embeddings=128,
    )
    values.update(overrides)
    return DFlash2Config(**values)


def _raw(args, *, top=True, nested=None):
    config = asdict(args)
    config.update(architectures=["DFlash2DraftModel"], dtype="bfloat16", dflash_config={})
    if not top:
        del config["num_target_layers"]
    if nested is not None:
        config["dflash_config"]["num_target_layers"] = nested
    return config


def test_missing_key_takes_target_count():
    args = DFlash2Config.from_dict(_raw(_args(), top=False), num_target_layers=40)
    assert args.num_target_layers == 40 and args.target_layer_ids == TAPS


def test_missing_key_without_target_count_fails_closed():
    with pytest.raises(ValueError, match="omits num_target_layers"):
        DFlash2Config.from_dict(_raw(_args(), top=False))


def test_explicit_key_is_authoritative():
    raw = _raw(_args())
    assert DFlash2Config.from_dict(raw).num_target_layers == 40
    # A differing target count never overrides the declared one; the
    # binding check refuses the pair instead (see the inspect test below).
    assert DFlash2Config.from_dict(raw, num_target_layers=48).num_target_layers == 40


def test_nested_key_is_read_and_conflicts_refused():
    raw = _raw(_args(), top=False, nested=40)
    assert DFlash2Config.from_dict(raw, num_target_layers=48).num_target_layers == 40
    with pytest.raises(ValueError, match="num_target_layers conflict"):
        DFlash2Config.from_dict(_raw(_args(), nested=41))


def test_base_dflash_parser_takes_target_count():
    raw = {"target_layer_ids": [1, 20, 38], "dflash_config": {}}
    assert DFlashConfig.from_dict(raw, num_target_layers=40).num_target_layers == 40
    with pytest.raises(ValueError, match="omits num_target_layers"):
        DFlashConfig.from_dict(raw)
    assert DFlashConfig.from_dict({**raw, "num_target_layers": 40}).num_target_layers == 40


def _artifact(tmp_path, config, args, target_layers):
    target, draft = tmp_path / "target", tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps(config))
    (target / "config.json").write_text(json.dumps({
        "hidden_size": args.hidden_size, "vocab_size": args.vocab_size,
        "num_hidden_layers": target_layers,
    }))
    header, end = {}, 0
    for name, shape in _expected_weight_shapes(args).items():
        start, end = end, end + 2 * math.prod(shape)
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [start, end]}
    blob = json.dumps(header).encode()
    (draft / "model.safetensors").write_bytes(len(blob).to_bytes(8, "little") + blob + bytes(end))
    return draft, target


def test_inspect_missing_key_binds_to_target_depth(tmp_path):
    args = _args()
    draft, target = _artifact(tmp_path, _raw(args, top=False), args, 40)
    assert inspect_drafter(draft, target)["args"].num_target_layers == 40


def test_inspect_missing_key_refuses_shallower_target(tmp_path):
    args = _args()
    draft, target = _artifact(tmp_path, _raw(args, top=False), args, 36)
    with pytest.raises(ValueError, match="target_layer_ids"):
        inspect_drafter(draft, target)


def test_inspect_explicit_key_mismatch_refused(tmp_path):
    args = _args()
    draft, target = _artifact(tmp_path, _raw(args), args, 48)
    with pytest.raises(ValueError, match="target layer count mismatch"):
        inspect_drafter(draft, target)
