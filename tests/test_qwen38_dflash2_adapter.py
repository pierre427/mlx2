"""Qwen3.8 27B DFlash2 policy, revision pins and tensor mapping; header-only.

No payload is loaded.  The real-artifact checks skip when the local
checkpoints are absent.
"""
import json
import struct
from pathlib import Path

import pytest

from mlx2.adapters.dflash2 import _expected_weight_shapes, validate_runtime_quantization
from mlx2.runtime.drafters.dflash2_config import DFlash2Config

DRAFT = Path.home() / "mlx-models/Qwen3.8-27B-DFlash2"
TARGET = Path.home() / "mlx-models/Qwen3.8-27B-oQ4e-mtp"
POLICY = Path(__file__).parents[1] / "qualification/policies/qwen38-27b-dflash2.json"


def _draft_config(**dflash):
    config = {
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "dtype": "bfloat16",
        "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
        "vocab_size": 64, "num_target_layers": 8, "rms_norm_eps": 1e-6,
        "max_position_embeddings": 512, "sliding_window": 6,
        "layer_types": ["sliding_attention"] * 2,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "dflash_config": {
            "block_size": 8, "mask_token_id": 63, "target_layer_ids": [1, 6],
            "conv_group_size": 4, "conv_kernel_size": 2, "selector_rank": 8,
            "selector_top_k": 4, **dflash,
        },
    }
    return config


def _write_artifacts(tmp_path, *, target_layers=8, draft=None):
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "text_config": {"hidden_size": 16, "vocab_size": 64, "num_hidden_layers": target_layers},
    }))
    (target / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "a.safetensors"}}))
    path = tmp_path / "draft"
    path.mkdir()
    config = draft or _draft_config()
    (path / "config.json").write_text(json.dumps(config))
    shapes = _expected_weight_shapes(DFlash2Config.from_dict(config))
    offset, header = 0, {}
    for name, shape in shapes.items():
        count = 1
        for value in shape:
            count *= value
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + 2 * count]}
        offset += 2 * count
    raw = json.dumps(header).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
    return target, path


def _pins(target, draft):
    from mlx2.adapters.dflash2 import content_revision, inspect_drafter
    from mlx2.adapters.qwen38_27b import content_revision as target_revision

    return {
        "draft_model": str(draft),
        "draft_revision": content_revision(inspect_drafter(draft, target)),
        "target_revision": target_revision(target),
    }


def test_pinned_policy_inspects_without_payload(tmp_path):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    record = inspect_external_policy({**_pins(target, draft), "num_draft": 5}, target)
    assert record["args"].target_layer_ids == [1, 6]
    assert record["runtime_quantization"] is None


@pytest.mark.parametrize("key", ["draft_revision", "target_revision"])
def test_missing_or_wrong_revision_pin_fails_closed(tmp_path, key):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    policy = _pins(target, draft)
    with pytest.raises(ValueError, match="must pin"):
        inspect_external_policy({k: v for k, v in policy.items() if k != key}, target)
    with pytest.raises(ValueError, match="revision mismatch"):
        inspect_external_policy({**policy, key: "0" * 64}, target)


def test_changed_target_or_draft_content_breaks_the_pin(tmp_path):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    policy = _pins(target, draft)
    config = json.loads((draft / "config.json").read_text())
    config["dflash_config"]["mask_token_id"] = 62
    (draft / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="draft revision mismatch"):
        inspect_external_policy(policy, target)
    index = target / "model.safetensors.index.json"
    index.write_text(index.read_text() + " ")
    with pytest.raises(ValueError, match="target revision mismatch"):
        inspect_external_policy(policy, target)


def test_topology_mismatch_fails_before_pins_matter(tmp_path):
    from mlx2.adapters.dflash2 import inspect_drafter

    target, draft = _write_artifacts(tmp_path, target_layers=6)
    with pytest.raises(ValueError, match="layer count"):
        inspect_drafter(draft, target)


@pytest.mark.parametrize(
    "extra, error",
    [
        ({"num_draft": 8}, "below the draft block size"),
        ({"num_draft": 0}, "below the draft block size"),
        ({"pairwise_selection": "gpu"}, "pairwise_selection"),
        ({"gdn_core": True}, "unknown keys"),
        ({"draft_quantization": {"bits": 3, "group_size": 64}}, "draft_quantization"),
    ],
)
def test_policy_negative_cases(tmp_path, extra, error):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    with pytest.raises(ValueError, match=error):
        inspect_external_policy({**_pins(target, draft), **extra}, target)


def test_runtime_quantization_is_part_of_the_draft_identity(tmp_path):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    policy = _pins(target, draft)
    plain = inspect_external_policy(policy, target)
    q4 = inspect_external_policy(
        {**policy, "draft_quantization": {"bits": 4, "group_size": 64}}, target
    )
    assert q4["fingerprint"] != plain["fingerprint"]
    assert q4["runtime_quantization"] == {"bits": 4, "group_size": 64}
    assert validate_runtime_quantization(None) is None


def _mapped_parameter_shapes(config):
    """Checkpoint names/shapes after ``sanitize``, against the lazy module tree."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel

    args = DFlash2Config.from_dict(config)
    model = DFlash2DraftModel(args)  # parameters stay lazy: nothing allocates
    module = {name: list(value.shape) for name, value in tree_flatten(model.parameters())}
    fake = {name: mx.zeros((1,)) for name in _expected_weight_shapes(args)}
    renamed = model.sanitize(fake)
    expected = _expected_weight_shapes(args)
    mapped = {}
    for source, target_name in zip(fake, renamed):
        mapped[target_name] = expected[source]
    return module, mapped


def test_tensor_mapping_covers_every_draft_parameter():
    module, mapped = _mapped_parameter_shapes(_draft_config())
    assert module == mapped


def test_real_checkpoint_headers_map_onto_the_drafter():
    if not (DRAFT / "config.json").exists() or not (TARGET / "config.json").exists():
        pytest.skip("local Qwen3.8 27B DFlash2 artifacts are not installed")
    from mlx2.adapters.dflash2 import _read_safetensors_header
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    policy = json.loads(POLICY.read_text())
    record = inspect_external_policy(policy, TARGET)
    args = record["args"]
    assert args.target_layer_ids == [5, 19, 33, 47, 61]
    assert (args.block_size, args.sliding_window, args.selector_top_k, args.selector_rank) == (8, 2048, 16, 256)
    assert (args.conv_kernel_size, args.conv_group_size, args.mask_token_id) == (2, 16, 248070)
    _, header, _ = _read_safetensors_header(DRAFT / "model.safetensors")
    header.pop("__metadata__", None)
    assert len(header) == 81
    config = json.loads((DRAFT / "config.json").read_text())
    module, mapped = _mapped_parameter_shapes(config)
    assert module == mapped
    assert {name: record["shape"] for name, record in header.items()} == _expected_weight_shapes(args)
