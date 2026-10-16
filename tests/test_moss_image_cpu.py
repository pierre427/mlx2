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
