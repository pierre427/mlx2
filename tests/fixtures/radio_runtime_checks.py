"""Isolated MLX CPU checks invoked by test_radio_cpu.py."""

import json
from dataclasses import replace

import pytest

from mlx2.adapters.radio_config import RadioConfig


def recipe(**args):
    return {
        "args": {
            "model": "vit_base_patch16_224",
            "cls_token_per_teacher": True,
            "teachers": [{"name": "a"}, {"name": "b", "use_summary": False}],
            "register_multiple": 8,
            **args,
        },
        "patch_size": 16,
        "max_resolution": 32,
        "preferred_resolution": [32, 32],
    }


@pytest.fixture
def cpu():
    mx = pytest.importorskip("mlx.core")
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield mx
    mx.set_default_device(device)


@pytest.fixture
def tiny(cpu):
    from mlx2.runtime.models.radio import RadioModel

    cfg = replace(
        RadioConfig.from_dict(recipe()),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
    )
    model = RadioModel(cfg)
    model.eval()
    return cfg, model


def test_summary_selection_and_patch_geometry(cpu, tiny):
    _cfg, model = tiny
    image = cpu.zeros((2, 3, 16, 32))
    output = model(image)
    cpu.eval(output.summary, output.features)
    assert output.summary.shape == (2, 8)
    assert output.features.shape == (2, 2, 8)
    with pytest.raises(ValueError, match="patch-aligned"):
        model(cpu.zeros((1, 3, 17, 32)))
    with pytest.raises(ValueError, match="NCHW"):
        model(cpu.zeros((1, 4, 16, 32)))


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "summary", "prefix"])
def test_loader_rejects_incomplete_or_wrong_checkpoint(
    cpu, tiny, tmp_path, monkeypatch, damage
):
    from mlx.utils import tree_flatten

    from mlx2.adapters import radio

    cfg, model = tiny
    weights = {"radio_model." + k: v for k, v in tree_flatten(model.parameters())}
    if damage == "missing":
        weights.pop("radio_model.model.blocks.0.attn.qkv.weight")
    elif damage == "extra":
        weights["radio_model.unexpected"] = cpu.zeros((1,))
    elif damage == "shape":
        weights["radio_model.model.blocks.0.attn.qkv.weight"] = cpu.zeros((1, 1))
    elif damage == "summary":
        weights["radio_model.summary_idxs"] = cpu.array([1])
    else:
        weights = {k.removeprefix("radio_model."): v for k, v in weights.items()}
    cpu.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    monkeypatch.setattr(radio, "inspect_artifact", lambda _: (tmp_path, cfg, "test"))
    with pytest.raises(ValueError):
        radio.RadioImageAdapter(tmp_path)


def test_loaded_encoder_receipt(cpu, tiny, tmp_path, monkeypatch):
    from mlx.utils import tree_flatten

    from mlx2.adapters import radio

    cfg, model = tiny
    weights = {"radio_model." + k: v for k, v in tree_flatten(model.parameters())}
    cpu.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    monkeypatch.setattr(radio, "inspect_artifact", lambda _: (tmp_path, cfg, "test"))
    adapter = radio.RadioImageAdapter(tmp_path)
    assert adapter.receipt["observed_used"] is False
    adapter.encode(cpu.zeros((1, 3, 32, 32)))
    assert adapter.receipt["observed_used"] is True
    assert adapter.receipt["qualified"] is False
    assert adapter.receipt["features_shape"] == [1, 4, 8]
    json.dumps(adapter.receipt)


@pytest.mark.parametrize("aligned", [True, False])
@pytest.mark.parametrize("size", [(1, 1), (3, 5), (9, 11)])
def test_position_interpolation_against_torch(cpu, aligned, size):
    import numpy as np

    torch = pytest.importorskip("torch")
    from mlx2.runtime.models.radio import _interpolate_bilinear_nchw

    pixels = np.random.default_rng(7).normal(size=(1, 2, 7, 8)).astype(np.float32)
    actual = _interpolate_bilinear_nchw(cpu.array(pixels), size, aligned)
    expected = torch.nn.functional.interpolate(
        torch.from_numpy(pixels), size=size, mode="bilinear", align_corners=aligned
    )
    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), atol=1e-6, rtol=1e-5
    )
