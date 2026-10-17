"""Native tensor/layout parity against BFL's pinned independent reference.

Set FLUX3_REFERENCE_ROOT to a clean checkout of the provenance-pinned source
to run the complete small-shape transformer and autoencoder CPU controls.
"""

import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

from mlx2.adapters.flux3_autoencoder import Flux3AutoEncoder, convert_vae_weights
from mlx2.adapters.flux3_transformer import (
    Flux3Transformer,
    latent_ids,
    rmsnorm_rope,
    rotary_angles,
    timestep_embedding,
    timesteps,
)


@pytest.fixture(autouse=True)
def cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    torch.set_num_threads(4)
    yield
    mx.set_default_device(previous)


def _reference(name):
    root = os.environ.get("FLUX3_REFERENCE_ROOT")
    if not root:
        pytest.skip(
            "set FLUX3_REFERENCE_ROOT for the independent pinned BFL CPU control"
        )
    root = Path(root)
    assert (
        subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        == "45e57c9fb06f9751cdced29a65c5d2171003be05"
    )
    assert not subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--", "src"], text=True
    ).strip()
    spec = importlib.util.spec_from_file_location(
        "flux3_reference_" + name, root / "src" / "flux3" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rotary_and_schedule():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((1, 2, 5, 16)).astype(np.float32)
    ids = rng.standard_normal((5, 4)).astype(np.float32)
    scales = rng.standard_normal(16).astype(np.float32)
    result = rmsnorm_rope(
        mx.array(x), mx.array(scales), rotary_angles(mx.array(ids), (4, 4, 4, 4))
    )
    t = torch.tensor(x)
    t = t * torch.rsqrt((t * t).mean(-1, keepdim=True) + 1e-6) * torch.tensor(scales)
    phase = torch.cat(
        [
            torch.tensor(ids[:, i : i + 1])
            * (10000 ** (-torch.arange(0, 4, 2, dtype=torch.float64) / 4))
            for i in range(4)
        ],
        -1,
    ).float()[None, None]
    a, b = t[..., ::2], t[..., 1::2]
    expected = torch.stack(
        (phase.cos() * a - phase.sin() * b, phase.sin() * a + phase.cos() * b), -1
    ).reshape(t.shape)
    np.testing.assert_allclose(np.array(result), expected.numpy(), atol=2e-6, rtol=2e-6)
    schedule = torch.linspace(1, 0, 17)
    schedule = 6.93 * schedule / (1 + 5.93 * schedule)
    np.testing.assert_allclose(
        np.array(timesteps(16)), schedule.numpy(), atol=1e-7, rtol=1e-6
    )


def test_bfloat16_timestep_rounding():
    reference = _reference("dit")
    times = torch.tensor([0.993, 0.731, 0.412, 0.081], dtype=torch.bfloat16)
    expected = reference.timestep_embedding(times).float().numpy()
    actual = timestep_embedding(mx.array(times.float().numpy(), dtype=mx.bfloat16))
    # exp/sin/cos implementations can differ by a BF16 rounding boundary.
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), expected, atol=1e-5, rtol=1 / 128
    )


@pytest.mark.parametrize("references", [0, 2])
def test_complete_small_transformer(references):
    reference = _reference("dit")
    reference.HIDDEN, reference.HEADS = 32, 2
    reference.DEPTH, reference.DEPTH_SINGLE = 2, 2
    reference.AXES_DIM = (4, 4, 4, 4)
    reference.IN_CHANNELS = {"image": 8, "image_cond": 8, "text": 12}
    torch.manual_seed(19)
    ordinary = reference.DiT(flow_map=True).eval()
    candidate = Flux3Transformer(
        {
            "hidden": 32,
            "heads": 2,
            "depth": 2,
            "depth_single": 2,
            "axes_dim": [4] * 4,
            "theta": 10000,
            "in_channels": reference.IN_CHANNELS,
            "mlp_ratio": 3.0,
            "flow_map": True,
        }
    )
    candidate.load_weights(
        [(k, mx.array(v.detach().numpy())) for k, v in ordinary.state_dict().items()],
        strict=True,
    )
    lengths = {"image": 4, "image_cond": references, "text": 6}
    data = {
        m: torch.randn(lengths[m], reference.IN_CHANNELS[m]) for m in reference.MODES
    }
    ids = {m: torch.randn(lengths[m], 4) for m in reference.MODES}
    times = {
        m: torch.full((lengths[m],), 0.7 if m == "image" else 0.0)
        for m in reference.MODES
    }
    packed = reference.Packed(
        data, ids, times, {m: [(0, lengths[m])] for m in reference.MODES}
    )
    with torch.no_grad():
        expected = ordinary(packed, timesteps_ref=torch.tensor([0.4])).numpy()
    result = candidate(
        {k: mx.array(v.numpy()) for k, v in data.items()},
        {k: mx.array(v.numpy()) for k, v in ids.items()},
        0.7,
        0.4,
    )
    np.testing.assert_allclose(np.array(result), expected, atol=3e-6, rtol=2e-5)


def test_complete_autoencoder_layout():
    reference = _reference("autoencoder")
    torch.manual_seed(2)
    ordinary = reference.AutoEncoder().eval()
    candidate = Flux3AutoEncoder()
    weights = {
        k: mx.array(v.detach().numpy()) for k, v in ordinary.state_dict().items()
    }
    candidate.load_weights(list(convert_vae_weights(weights).items()), strict=True)
    candidate.eval()
    image = torch.randn(3, 32, 48)
    with torch.no_grad():
        expected_tokens, ids = ordinary.encode(image)
        expected_image = ordinary.decode(expected_tokens, ids).permute(1, 2, 0).numpy()
    tokens, native_ids = candidate.encode(mx.array(image.permute(1, 2, 0).numpy()))
    np.testing.assert_array_equal(np.array(native_ids), ids.numpy())
    np.testing.assert_array_equal(np.array(latent_ids(2, 3)), ids.numpy())
    np.testing.assert_allclose(
        np.array(tokens), expected_tokens.numpy(), atol=2e-5, rtol=1e-4
    )
    result = candidate.decode(mx.array(expected_tokens.numpy()), 2, 3)
    np.testing.assert_allclose(np.array(result), expected_image, atol=2e-4, rtol=2e-3)


def test_text_hidden_layers_and_padding(monkeypatch):
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.models.qwen3_vl.config import TextConfig
    from mlx_vlm.models.qwen3_vl.language import Qwen3VLModel
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    from mlx2.adapters import flux3_pipeline

    config = {
        "model_type": "qwen3_vl_text",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000,
        "max_position_embeddings": 128,
        "tie_word_embeddings": True,
        "rope_scaling": {
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    }
    torch.manual_seed(3)
    ordinary = Qwen3VLTextModel(Qwen3VLTextConfig(**config)).eval()
    candidate = Qwen3VLModel(TextConfig.from_dict(config))
    candidate.load_weights(
        [(k, mx.array(v.detach().numpy())) for k, v in ordinary.state_dict().items()],
        strict=True,
    )
    monkeypatch.setattr(flux3_pipeline, "OUTPUT_LAYERS", (1, 2, 3))
    token_ids = np.array([[7, 8, 9, 10, 11] + [0] * 75])
    mask = token_ids != 0

    class Tokenizer:
        def __call__(self, text, **kwargs):
            if "return_tensors" in kwargs:
                return {"input_ids": token_ids, "attention_mask": mask}
            return {"input_ids": token_ids[0, :5].tolist()}

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, *args, **kwargs):
            return "a teapot"

    encoder = object.__new__(flux3_pipeline.TextEncoder)
    encoder.model, encoder.processor = candidate, Processor()
    actual, ids = encoder.encode("a teapot")
    positions = np.where(mask, np.cumsum(mask, axis=-1) - 1, 1)
    with torch.no_grad():
        output = ordinary(
            input_ids=torch.tensor(token_ids),
            attention_mask=torch.tensor(mask),
            position_ids=torch.tensor(np.broadcast_to(positions[None], (3, 1, 80))),
            output_hidden_states=True,
            use_cache=False,
        )
        expected = (
            torch.stack([output.hidden_states[k] for k in (1, 2, 3)], dim=2)
            .reshape(80, -1)
            .bfloat16()
            .float()
            .numpy()
        )
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), expected, atol=1e-5, rtol=1e-3
    )
    np.testing.assert_array_equal(np.array(ids[:, 3]), np.arange(80))


@pytest.mark.parametrize("failure", ["state", "decoder"])
def test_nonfinite_values_fail_before_png_conversion(failure):
    from mlx2.adapters.flux3_pipeline import Flux3Pipeline

    pipeline = object.__new__(Flux3Pipeline)
    pipeline.text_encoder = SimpleNamespace(
        encode=lambda prompt: (
            mx.zeros((1, 20480), dtype=mx.bfloat16),
            mx.zeros((1, 4)),
        )
    )
    pipeline.transformer = lambda data, ids, current, target: mx.full(
        data["image"].shape,
        float("nan") if failure == "state" else 0,
        dtype=mx.bfloat16,
    )
    pipeline.vae = SimpleNamespace(
        decode=lambda tokens, h, w: mx.full((h * 16, w * 16, 3), float("nan"))
    )
    with pytest.raises(FloatingPointError, match="nonfinite FLUX 3"):
        pipeline.generate("a teapot", [], width=64, height=64, steps=1, seed=0)


def _noise_image(width, height, seed=0):
    from PIL import Image

    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def test_single_reference_reflect_pads_like_the_reference():
    from mlx2.adapters.flux3_pipeline import prepare_reference

    image = _noise_image(70, 100)
    padding = ((6, 6), (5, 5), (0, 0))  # centred on the 80x112 canvas
    np.testing.assert_array_equal(
        np.asarray(prepare_reference(image, multiple=False)),
        np.pad(np.asarray(image), padding, mode="reflect"),
    )
    # The reference falls back to symmetric only when a side is too short.
    tiny = _noise_image(10, 12)
    np.testing.assert_array_equal(
        np.asarray(prepare_reference(tiny, multiple=False)),
        np.pad(np.asarray(tiny), ((26, 26), (27, 27), (0, 0)), mode="symmetric"),
    )


def test_multi_reference_floors_and_centre_crops_like_the_reference():
    from PIL import Image

    from mlx2.adapters.flux3_pipeline import prepare_reference

    image = _noise_image(70, 100)
    np.testing.assert_array_equal(
        np.asarray(prepare_reference(image, multiple=True)),
        np.asarray(image)[2:98, 3:67],
    )
    large = _noise_image(1500, 1000, seed=1)
    scale = math.sqrt(1024 * 1024 / (1500 * 1000))
    fitted = large.resize(
        (int(1500 * scale), int(1000 * scale)), Image.Resampling.LANCZOS
    )
    assert fitted.size == (1254, 836)
    np.testing.assert_array_equal(
        np.asarray(prepare_reference(large, multiple=True)),
        np.asarray(fitted)[2:834, 3:1251],
    )
    with pytest.raises(ValueError, match="reference"):
        prepare_reference(_noise_image(15, 40), multiple=True)
