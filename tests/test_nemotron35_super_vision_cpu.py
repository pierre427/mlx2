"""Source-layout and spatial/temporal packing oracles for Super VL vision."""
import mlx.core as mx
import numpy as np
import pytest
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


def _tiny_super():
    from mlx2.runtime.models.nemotron35_super import Model
    from mlx2.runtime.models.nemotron_h import ModelArgs

    mx.random.seed(11)
    args = ModelArgs(
        model_type="nemotron_h", vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=4, max_position_embeddings=256,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        attention_bias=False, mamba_num_heads=2, mamba_head_dim=8,
        mamba_proj_bias=False, ssm_state_size=4, conv_kernel=4, n_groups=1,
        mlp_bias=False, layer_norm_epsilon=1e-5, use_bias=False,
        use_conv_bias=True, hybrid_override_pattern=["M", "*", "-", "M"],
    )
    radio = RadioConfig(8, 1, 2, 16, 2, 8, 1, 1, (0,), (4, 4), False)
    model = Model(args, radio_config=radio, projector_hidden_size=16)
    features = mx.random.normal((1, 2, 16))
    model.encode_images = lambda pixels: features
    return model


def test_external_media_prefill_leaves_the_ordinary_image_prefill_cache():
    mx.set_default_device(mx.cpu)
    model = _tiny_super()
    tokens = mx.array([[2, 5, 18, 18, 7, 9, 4]])
    pixels = [np.ones((1, 3, 32, 32), np.float32)]
    ordinary = model.make_cache()
    mx.eval(model.prefill_forward(tokens, ordinary, image_pixels=pixels))
    external = model.make_cache()
    taps = model.media_prefill_body(tokens, external, (0, 3), image_pixels=pixels)
    mx.eval(taps)
    assert taps.shape == (1, 7, 32)
    from mlx.utils import tree_flatten

    for left, right in zip(external, ordinary, strict=True):
        a_leaves = [v for _, v in tree_flatten(left.state) if isinstance(v, mx.array)]
        b_leaves = [v for _, v in tree_flatten(right.state) if isinstance(v, mx.array)]
        assert a_leaves and len(a_leaves) == len(b_leaves)
        for a, b in zip(a_leaves, b_leaves, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # The image features reach the taps (they differ from a text-only prefill).
    text = model.make_cache()
    plain = model.prefill_body(tokens, text, (0, 3))
    assert not np.array_equal(np.asarray(plain[:, 2:]), np.asarray(taps[:, 2:]))
    np.testing.assert_array_equal(np.asarray(plain[:, :2]), np.asarray(taps[:, :2]))
    with pytest.raises(ValueError, match="placeholders"):
        model.media_prefill_body(mx.array([[2, 18, 3]]), model.make_cache(), (0,),
                                 image_pixels=pixels)


def test_super_declares_external_media_and_drops_the_refusal():
    import inspect

    from mlx2.adapters.nemotron35_super import Nemotron35SuperAdapter
    from mlx2.serving import media_route_policy

    assert media_route_policy(Nemotron35SuperAdapter, "external_draft")["media"] == "route_native"
    assert media_route_policy(Nemotron35SuperAdapter, "native_mtp")["media"] == "refused"
    source = inspect.getsource(Nemotron35SuperAdapter.prepare_multimodal_request)
    assert "external draft with media prefill is not implemented" not in source


def _source_video_patches(width, height, target=1024, divisor=2):
    """The pinned image processor's ``_compute_target_patches_video``, verbatim."""
    import math

    aspect_wh = width / max(height, 1)
    ph = max(round(math.sqrt(target / aspect_wh)), 1)
    pw = max(round(math.sqrt(target * aspect_wh)), 1)
    rem_h, rem_w = ph % divisor, pw % divisor
    ph_up = ph + (divisor - rem_h if rem_h else 0)
    ph_down = ph - rem_h
    pw_up = pw + (divisor - rem_w if rem_w else 0)
    pw_down = pw - rem_w
    if ph_up * pw_up <= target:
        ph, pw = ph_up, pw_up
    else:
        ph = max(divisor, ph_down)
        pw = max(divisor, pw_down)
    return pw, ph


def test_video_grid_matches_the_source_rule():
    from mlx2.adapters.nemotron35_media import video_grid

    for width, height in [(640, 480), (1920, 1080), (480, 640), (100, 100), (1280, 200), (33, 999)]:
        pw, ph = _source_video_patches(width, height)
        assert video_grid(width, height) == (ph, pw)


def test_video_prompt_matches_the_source_labels():
    from mlx2.adapters.nemotron35_media import video_prompt

    prompt = video_prompt(3, 2, source_fps=30.0, indices=(0, 15, 30))
    chunk = "<img><image><image></img>"
    assert prompt == (
        "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.49 seconds: " + chunk
        + "\nFrame 3 sampled at 0.99 seconds: " + chunk
    )
    assert video_prompt(2, 1) == "Frame 1 and frame 2: <img><image></img>"


def test_video_and_image_features_fill_placeholders_in_prompt_order():
    mx.set_default_device(mx.cpu)
    model = _tiny_super()
    image = mx.full((1, 1, 16), 2.0)
    video = mx.stack([mx.full((2, 16), 3.0), mx.full((2, 16), 4.0)])  # 2 tubelets x 2 tokens
    model.encode_images = lambda pixels: image
    model.encode_video = lambda frames: video
    tokens = mx.array([[2, 18, 5, 18, 18, 18, 18, 7]])
    out = model.image_embeddings(
        tokens, [np.zeros((1, 3, 32, 32), np.float32)],
        video_pixels=[np.zeros((3, 3, 32, 32), np.float32)],
        media_order=["image", "video"],
    )
    values = np.asarray(out[0, :, 0])
    assert values[1] == 2.0 and list(values[3:7]) == [3.0, 3.0, 4.0, 4.0]
    plain = np.asarray(model.embed_tokens(tokens))
    np.testing.assert_array_equal(np.asarray(out)[0, [0, 2, 7]], plain[0, [0, 2, 7]])
    with pytest.raises(ValueError, match="order"):
        model.image_embeddings(tokens, [], video_pixels=[np.zeros((1, 3, 32, 32))],
                               media_order=["image"])


def test_super_prepares_a_video_request(monkeypatch):
    from types import SimpleNamespace

    from mlx2.adapters import nemotron35_super as sv
    from mlx2.multimodal import MediaValue

    frames = [np.full((48, 64, 3), value, np.uint8) for value in (0, 80, 160)]
    video = MediaValue("video", "video/mp4", "ab" * 32, 10, frames,
                       {"source_fps": 30.0, "sampled_indices": (0, 15, 30)})
    monkeypatch.setattr("mlx2.multimodal.resolve_media", lambda *a, **k: video)
    adapter = object.__new__(sv.Nemotron35SuperAdapter)
    adapter.source_config = {"norm_mean": [0.5, 0.5, 0.5], "norm_std": [0.25, 0.25, 0.25]}
    adapter.identity = {"fingerprint": "f" * 64}
    adapter.draft_model = object()
    rendered = {}

    def tokens(self, request):
        text = request["messages"][0]["content"]
        rendered["text"] = text
        return [18 if piece == "<image>" else 1 for piece in
                text.replace("<image>", "\0<image>\0").split("\0") if piece]

    monkeypatch.setattr(sv.Nemotron3SuperAdapter, "prompt_tokens", tokens)
    request = {"messages": [{"role": "user", "content": [
        {"type": "input_video", "video_url": {"url": "data:video/mp4;base64,AA=="}},
        {"type": "text", "text": "What changes?"},
    ]}]}
    prepared = adapter.prepare_multimodal_request(request)
    inputs = prepared["_mlx2_prefill_inputs"]
    assert inputs["media_order"] == ["video"] and inputs["image_pixels"] == []
    pixels = inputs["video_pixels"][0]
    h, w = sv_grid = __import__("mlx2.adapters.nemotron35_media", fromlist=["x"]).video_grid(64, 48)
    assert pixels.shape == (3, 3, h * 16, w * 16)
    per_tubelet = h * w // 4
    assert prepared["_mlx2_prompt_tokens"].count(18) == 2 * per_tubelet
    assert rendered["text"].startswith("Frame 1 sampled at 0.00 seconds and frame 2")
    with pytest.raises(ValueError, match="pruning"):
        adapter.prepare_multimodal_request({**request, "video_pruning_rate": 0.7})
