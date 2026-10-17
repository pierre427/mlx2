"""CPU-only MOSS-Image artifact and native math contracts."""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from mlx2.adapters import moss_image as module


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    entries = []
    for name, raw in (
        ("model_index.json", b'{"_class_name":"MOSSImagePipeline"}'),
        ("transformer/config.json", b'{"_class_name":"MOSSImageTransformer"}'),
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        entries.append(
            {
                "path": name,
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob": None,
            }
        )
    manifest = {
        "repo": module.MOSS_IMAGE_REPO,
        "revision": module.MOSS_IMAGE_REVISION,
        "files": entries,
    }
    monkeypatch.setattr(module, "snapshot_manifest", lambda: manifest)
    (tmp_path / ".hf-download-manifest.json").write_text(json.dumps(manifest))
    (tmp_path / ".hf-download-complete.json").write_text(
        json.dumps(
            {
                "repo": module.MOSS_IMAGE_REPO,
                "revision": module.MOSS_IMAGE_REVISION,
                "files": len(entries),
                "total_bytes": sum(entry["size"] for entry in entries),
            }
        )
    )
    return tmp_path


def test_artifact_hash_and_revision_gate(artifact):
    result = module.inspect_moss_image(artifact)
    assert result.kind == "moss-image-sft-bf16"
    assert result.execution_qualified is False
    (artifact / "model_index.json").write_text("changed")
    with pytest.raises(ValueError, match="hash differs|incomplete"):
        module.inspect_moss_image(artifact)


def test_adapter_receipt_and_default_optimization(artifact):
    class Backend:
        def generate(self, prompt, **kwargs):
            assert prompt == "a teapot"
            assert kwargs["uncond_skip_layers"] == (22,)
            return Image.new("RGB", (kwargs["width"], kwargs["height"]), "blue")

    adapter = module.MOSSImageSFTAdapter(
        artifact, backend_factory=lambda _root: Backend()
    )
    image = adapter.generate_image("a teapot", width=256, height=256, steps=2)
    assert image.width == image.height == 256
    assert adapter.last_route_receipt["success"] is True
    assert adapter.last_route_receipt["observed_used"] is True
    assert adapter.status()["qualified"] is False
    assert adapter.status()["apcv2"] == "not_applicable"
    assert adapter.status()["capabilities"]["negative_branch_layer_skip"] is True


def test_changed_artifact_fails_before_backend_load(artifact):
    adapter = module.MOSSImageSFTAdapter(
        artifact,
        backend_factory=lambda _root: pytest.fail("changed artifact loaded backend"),
    )
    (artifact / "model_index.json").write_text("changed")
    with pytest.raises(ValueError, match="changed since inspection"):
        adapter.generate_image("a teapot", width=256, height=256, steps=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": True},
        {"height": 255},
        {"steps": 0},
        {"seed": -1},
        {"guidance_scale": float("nan")},
        {"uncond_skip_layers": (42,)},
        {"uncond_skip_layers": 22},
    ],
)
def test_invalid_request_fails_before_backend_load(artifact, kwargs):
    adapter = module.MOSSImageSFTAdapter(
        artifact,
        backend_factory=lambda _root: pytest.fail("invalid request loaded backend"),
    )
    with pytest.raises((TypeError, ValueError)):
        adapter.generate_image("a teapot", **kwargs)
    assert adapter.last_route_receipt["success"] is False
    assert adapter.last_route_receipt["observed_used"] is False


def test_none_uncond_skip_layers_disables_skipping(artifact):
    # The publisher documents None (or an empty sequence) as "disable".
    seen = {}

    class Backend:
        def generate(self, prompt, **kwargs):
            seen.update(kwargs)
            return Image.new("RGB", (kwargs["width"], kwargs["height"]), "blue")

    adapter = module.MOSSImageSFTAdapter(
        artifact, backend_factory=lambda _root: Backend()
    )
    adapter.generate_image(
        "a teapot", width=256, height=256, steps=2, uncond_skip_layers=None
    )
    assert seen["uncond_skip_layers"] == ()
    assert adapter.last_route_receipt["success"] is True
    assert adapter.last_route_receipt["parameters"]["uncond_skip_layers"] == []


def test_q8_receipt_is_separate_and_unqualified(tmp_path):
    record = tmp_path / "model_index.json"
    record.write_text("{}")
    proof = {
        "source_repo": module.MOSS_IMAGE_REPO,
        "source_revision": module.MOSS_IMAGE_REVISION,
        "source_fingerprint": module._fingerprint(module.snapshot_manifest()),
        "quantization": module.Q8,
        "output_files": {
            "model_index.json": {
                "size": record.stat().st_size,
                "sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
            }
        },
        "execution_qualification": "pending",
        "ab_comparison": "pending",
    }
    (tmp_path / "mlx2-conversion.json").write_text(json.dumps(proof))
    adapter = module.MOSSImageSFTAdapter(tmp_path, backend_factory=lambda _: None)
    assert adapter.artifact.kind == "moss-image-sft-mlx-q8"
    assert adapter.status()["capabilities"]["quantized_weights"] is True
    assert adapter.status()["qualified"] is False


def test_q8_runtime_preserves_embeddings_and_quantizes_linears():
    mx = pytest.importorskip("mlx.core")
    from mlx import nn

    from mlx2.adapters.moss_image_pipeline import Q8, _quantize_if_declared
    from mlx2.adapters.moss_image_transformer import _linear_input_dtype

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(64, 64)
            self.projection = nn.Linear(64, 64, bias=False)

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = Model()
        _quantize_if_declared(model, {"quantization": Q8, "mlx_format": True})
        assert isinstance(model.embed, nn.Embedding)
        assert isinstance(model.projection, nn.QuantizedLinear)
        assert model.projection.weight.dtype == mx.uint32
        assert _linear_input_dtype(model.projection) == model.projection.scales.dtype
        output = model.projection(mx.ones((1, 64), dtype=_linear_input_dtype(model.projection)))
        mx.eval(output)
        assert mx.all(mx.isfinite(output)).item()
    finally:
        mx.set_default_device(previous)


def test_native_activation_rotary_and_schedule_match_reference():
    mx = pytest.importorskip("mlx.core")
    torch = pytest.importorskip("torch")
    from mlx2.adapters.moss_image_transformer import (
        flow_schedule,
        rotary_embedding,
        situ,
    )

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        rng = np.random.default_rng(9)
        hidden = rng.standard_normal((3, 7)).astype(np.float32)
        gate = rng.standard_normal((3, 7)).astype(np.float32)
        expected = (
            4 * torch.tanh(torch.tensor(gate) / 4) * torch.sigmoid(torch.tensor(gate))
        ) * (25 * torch.tanh(torch.tensor(hidden) / 25))
        np.testing.assert_allclose(
            np.array(situ(mx.array(hidden), mx.array(gate))),
            expected.numpy(),
            atol=2e-6,
        )

        height, width, text, head = 2, 3, 5, 16
        bands = 1 / (10000 ** (np.arange(head // 8, dtype=np.float32) / (head // 8)))
        rows, columns = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        image = (
            np.stack((np.zeros_like(rows), np.full_like(rows, text), rows, columns), -1)
            .reshape(-1, 4)
            .astype(np.float32)
        )
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        words = np.zeros((text, 4), dtype=np.float32)
        words[:, 1] = np.arange(text)
        phase = np.concatenate((image, words))[..., None] * bands
        expected_rope = np.concatenate((np.sin(phase), np.cos(phase)), -2).reshape(
            -1, head
        )
        np.testing.assert_allclose(
            np.array(rotary_embedding(height, width, text, head)),
            expected_rope,
            atol=2e-6,
        )

        total = 1000
        base = torch.linspace(1, 5 / (total + 4), 7)
        requested = 5 * base / (1 + 4 * base)
        training = torch.linspace(1, 1 / total, total).flip(0)
        expected_schedule = []
        for sigma in requested:
            value = training[torch.argmin((training - sigma).abs())].item()
            if not expected_schedule or value != expected_schedule[-1]:
                expected_schedule.append(value)
        expected_schedule.append(0.0)
        np.testing.assert_allclose(
            np.array(flow_schedule(7)), expected_schedule, atol=2e-7, rtol=0
        )
    finally:
        mx.set_default_device(previous)


def test_text_encoder_offsets_only_centred_norms(tmp_path, monkeypatch):
    # HF Qwen3.5 stores Qwen3_5RMSNorm weights centred on zero (it computes
    # 1 + weight), but the GDN gated norm linear_attn.norm is a plain-weight
    # Qwen3_5RMSNormGated and must load unchanged.
    mx = pytest.importorskip("mlx.core")
    from mlx.utils import tree_flatten
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import Qwen3_5Model
    from transformers import AutoTokenizer

    from mlx2.adapters.moss_image_pipeline import TextEncoder

    text_config = {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "num_key_value_heads": 1,
        "max_position_embeddings": 512,
        "head_dim": 32,
        "full_attention_interval": 4,
        "rope_parameters": {
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "rope_type": "default",
        },
    }
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = Qwen3_5Model(TextConfig.from_dict(text_config))
        rng = np.random.default_rng(5)
        checkpoint = {}
        for name, value in tree_flatten(model.parameters()):
            value = mx.array(rng.standard_normal(value.shape).astype(np.float32))
            if "conv1d.weight" in name:
                value = value.moveaxis(1, 2)  # HF (C, 1, K) layout
            checkpoint["language_model." + name] = value
        mx.save_safetensors(str(tmp_path / "m.safetensors"), checkpoint)
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": dict.fromkeys(checkpoint, "m.safetensors")})
        )
        (tmp_path / "config.json").write_text(json.dumps({"text_config": text_config}))
        monkeypatch.setattr(
            AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: None)
        )
        loaded = dict(tree_flatten(TextEncoder(tmp_path, tmp_path).model.parameters()))
        for name, offset in (
            ("layers.0.linear_attn.norm.weight", 0),
            ("layers.0.input_layernorm.weight", 1),
            ("layers.0.post_attention_layernorm.weight", 1),
            ("layers.3.self_attn.q_norm.weight", 1),
            ("layers.3.self_attn.k_norm.weight", 1),
            ("norm.weight", 1),
        ):
            np.testing.assert_array_equal(
                np.array(loaded[name]),
                np.array(checkpoint["language_model." + name]) + offset,
                err_msg=name,
            )
    finally:
        mx.set_default_device(previous)


def test_pixels_round_like_the_publisher_postprocess():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from mlx2.adapters.moss_image_pipeline import MOSSImagePipeline

    # Each decoded value maps to k + 0.75: Diffusers' numpy_to_pil rounds it
    # up to k + 1, a truncating cast would give k.
    levels = np.arange(16 * 16 * 3, dtype=np.float64) % 255
    decoded = ((levels + 0.75) / 127.5 - 1).astype(np.float32).reshape(1, 16, 16, 3)
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        pipeline = object.__new__(MOSSImagePipeline)
        pipeline.text_encoder = SimpleNamespace(
            encode=lambda _prompt: mx.zeros((1, 4, 8), dtype=mx.bfloat16)
        )
        pipeline.transformer = lambda latents, *_args, **_kwargs: mx.zeros_like(
            latents
        )
        pipeline.vae = SimpleNamespace(decode=lambda _latents: mx.array(decoded))
        image = pipeline.generate(
            "a teapot", width=16, height=16, steps=1, seed=0, guidance_scale=1.0
        )
    finally:
        mx.set_default_device(previous)
    np.testing.assert_array_equal(
        np.asarray(image), (levels + 1).astype(np.uint8).reshape(16, 16, 3)
    )
