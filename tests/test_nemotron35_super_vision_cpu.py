"""Source-layout and spatial/temporal packing oracles for Super VL vision."""
import mlx.core as mx
import numpy as np
from mlx import nn

from mlx2.adapters.radio_config import RadioConfig
from mlx2.runtime.models.nemotron35_super import (
    SuperRadio,
    VisionProjector,
    pixel_shuffle_v2,
)


def test_pixel_shuffle_matches_explicit_spatial_groups():
    x = np.arange(2 * 4 * 6 * 3).reshape(2, 4, 6, 3).astype(np.float32)
    expected = np.empty((2, 2, 3, 12), np.float32)
    for batch in range(2):
        for row in range(2):
            for col in range(3):
                # NVIDIA v2 groups adjacent width channels before height.
                expected[batch, row, col] = x[batch, row*2:row*2+2, col*2:col*2+2].reshape(-1)
    np.testing.assert_array_equal(np.asarray(pixel_shuffle_v2(mx.array(x))), expected)


def test_radio_composition_uses_external_normalization_once():
    mx.set_default_device(mx.cpu)
    cfg = RadioConfig(8, 1, 2, 16, 2, 8, 1, 1, (0,), (4, 4), False)
    model = SuperRadio(cfg)
    pixels = mx.arange(48, dtype=mx.float32).reshape(1, 3, 4, 4) / 48
    first = model(pixels)
    model.radio_model.input_conditioner.norm_mean = mx.full((3, 1, 1), 100.)
    model.radio_model.input_conditioner.norm_std = mx.full((3, 1, 1), .001)
    # These checkpoint buffers are retained, but the Super VL composition
    # delegates normalization to its processor exactly as NVIDIA does.
    np.testing.assert_array_equal(np.asarray(first), np.asarray(model(pixels)))
    assert first.shape == (1, 4, 8)
    projector = VisionProjector(8, 16, 8)
    assert projector(first, grid=(2, 2)).shape == (1, 1, 8)
    assert model(mx.concatenate([pixels, pixels], axis=1), video=True).shape == (1, 4, 8)


def test_image_prefill_replaces_only_owned_image_token_embeddings():
    from mlx2.runtime.models.nemotron35_super import Model

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(32, 8)

        def __call__(self, inputs, cache=None, inputs_embeds=None):
            return self.embeddings(inputs) if inputs_embeds is None else inputs_embeds

    class TinyModel(Model):
        def __init__(self):
            nn.Module.__init__(self)
            self.backbone = Backbone()
            self.lm_head = nn.Identity()

        def encode_images(self, pixels):
            return mx.full((1, 1, 8), .25)

    model = TinyModel()
    tokens = mx.array([[2, 18, 3]])
    output = np.asarray(model.prefill_forward(tokens, image_pixels=[np.ones((1, 3, 32, 32), np.float32)]))
    expected = np.asarray(model.embed_tokens(tokens)).copy()
    expected[:, 1] = .25
    np.testing.assert_array_equal(output, expected)


def test_complete_runtime_tensor_layout_matches_streaming_conversion_without_payloads():
    import json
    from pathlib import Path

    from mlx.utils import tree_flatten

    from mlx2.adapters.nemotron35_super_config import language_config, tensor_schema
    from mlx2.conversion.nemotron35_plan import make_plan
    from mlx2.runtime.models.nemotron35_super import Model
    from mlx2.runtime.models.nemotron_h import ModelArgs

    config = json.loads((Path(__file__).parent / "fixtures/nemotron35_super_config.json").read_text())
    # MLX's lazy constructors expose shapes without allocating the 249 GB
    # parameters. Never evaluate this model in a metadata-only CPU check.
    model = Model(ModelArgs.from_dict(language_config(config)))

    def mapped(name, shape):
        name = name.removeprefix("language_model.")
        if name.startswith("mlp1."):
            parts = name.split(".")
            parts[1] = {"0": "norm", "1": "linear1", "3": "linear2"}[parts[1]]
            name = "vision_projector." + ".".join(parts)
        if name == "vision_model.summary_idxs":
            name = "vision_model.radio_model.summary_idxs"
        if name.endswith("conv1d.weight"):
            shape = [shape[0], shape[2], shape[1]]
        return name, shape

    expected = dict(mapped(k, v["shape"]) for k, v in tensor_schema(config, fused=True).items())
    assert {k: list(v.shape) for k, v in tree_flatten(model.parameters())} == expected
    plan = make_plan(config)
    modules = {k.removeprefix("language_model."): v for k, v in plan["modules"].items()}
    nn.quantize(model, class_predicate=lambda name, module: dict(modules[name]) if name in modules else False)
    expected = dict(mapped(k, v["shape"]) for k, v in plan["schema"].items())
    assert {k: list(v.shape) for k, v in tree_flatten(model.parameters())} == expected
