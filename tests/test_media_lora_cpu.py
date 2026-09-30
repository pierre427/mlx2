"""Real MLX arithmetic on CPU; no model weights and no GPU dispatch."""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from safetensors.numpy import load_file, save_file

from mlx2.runtime.media_lora import (
    convert_media_lora,
    export_ltx_native,
    inspect_media_lora,
    install_media_lora,
    target_key,
    write_media_lora,
)
from mlx2.runtime.media_lora_training import (
    FlowExample,
    image_flow_example,
    music_flow_example,
    train_media_lora,
)

BASE, REV = "a" * 64, "b" * 40


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def model(family="qwen-image-2.1", quantized=False):
    m = nn.Module()
    block = nn.Module()
    block.attn = nn.Module()
    if family == "minimax-music3":
        block.attn.to_qkv = nn.Linear(64 if quantized else 4, 3, bias=False)
        m.blocks = [block]
        key = "blocks.0.attn.to_qkv"
    elif family == "ltx-2.5":
        block.attn1 = nn.Module()
        block.attn1.to_q = nn.Linear(4, 3, bias=False)
        m.transformer_blocks = [block]
        key = "transformer_blocks.0.attn1.to_q"
    else:
        block.attn.to_q = nn.Linear(64 if quantized else 4, 3, bias=False)
        m.transformer_blocks = [block]
        key = "transformer_blocks.0.attn.to_q"
    if quantized:
        nn.quantize(m, bits=4, group_size=64)
    return m, key


def artifact(path, family, key, *, inputs=4, b=1, scale=0.7):
    tensors = {
        key + ".lora_a": np.arange(inputs * 2, dtype=np.float32).reshape(inputs, 2)
        / 10,
        key + ".lora_b": np.full((2, 3), b, dtype=np.float32),
    }
    return write_media_lora(
        path,
        tensors,
        family=family,
        base_fingerprint=BASE,
        backend_revision=REV,
        scale=scale,
    )


@pytest.mark.parametrize("family", ["qwen-image-2.1", "ltx-2.5", "minimax-music3"])
@pytest.mark.parametrize("b", [0, 1])
def test_effect_exact_restore_and_reload(tmp_path, family, b):
    m, key = model(family)
    linear = dict(m.named_modules())[key]
    x = mx.ones((1, 4))
    reference = linear(x)
    mx.eval(reference)
    a = artifact(tmp_path / "a", family, key, b=b)
    session = install_media_lora(m, a)
    replacement = dict(m.named_modules())[key]
    expected = reference + 0.7 * (x @ replacement.lora_a @ replacement.lora_b)
    assert mx.allclose(replacement(x), expected, atol=1e-6).item()
    if b:
        assert not mx.allclose(replacement(x), reference).item()
    else:
        assert mx.array_equal(replacement(x), reference).item()
    saved = session.export(tmp_path / "saved")
    session.restore()
    assert dict(m.named_modules())[key] is linear
    assert mx.array_equal(linear(x), reference).item()
    reloaded = install_media_lora(m, saved)
    assert mx.allclose(dict(m.named_modules())[key](x), expected).item()
    reloaded.restore()
    reloaded.restore()


def test_quantized_wrapper_parity(tmp_path):
    m, key = model(quantized=True)
    linear = dict(m.named_modules())[key]
    x = mx.ones((1, 64))
    reference = linear(x)
    a = artifact(tmp_path / "a", "qwen-image-2.1", key, inputs=64)
    session = install_media_lora(m, a)
    wrapper = dict(m.named_modules())[key]
    assert mx.allclose(
        wrapper(x), reference + 0.7 * (x @ wrapper.lora_a @ wrapper.lora_b), atol=1e-5
    ).item()
    session.restore()
    assert dict(m.named_modules())[key] is linear


@pytest.mark.parametrize(
    "prefix,suffix",
    [
        ("", ".default"),
        ("transformer.", ""),
        ("diffusion_model.", ""),
        ("base_model.model.", ""),
    ],
)
def test_peft_scale_and_orientation(tmp_path, prefix, suffix):
    key = "transformer_blocks.0.attn.to_q"
    source = tmp_path / "peft.safetensors"
    save_file(
        {
            prefix + key + ".lora_A" + suffix + ".weight": np.ones((2, 4), np.float32),
            prefix + key + ".lora_B" + suffix + ".weight": np.ones((3, 2), np.float32),
        },
        str(source),
    )
    converted = convert_media_lora(
        source,
        tmp_path / "out",
        family="qwen-image-2.1",
        base_fingerprint=BASE,
        backend_revision=REV,
        alpha=8,
        strength=0.5,
    )
    assert converted.config["scale"] == 2
    assert converted.tensors()[key + ".lora_a"].shape == (4, 2)
    assert converted.tensors()[key + ".lora_b"].shape == (2, 3)


def test_ltx_export_bakes_alpha_once(tmp_path):
    key = target_key("ltx-2.5", "diffusion_model.transformer_blocks.0.attn1.to_out.0")
    assert key.endswith(".to_out")
    a = artifact(tmp_path / "a", "ltx-2.5", key, scale=3)
    export_ltx_native(a, tmp_path / "native.safetensors")
    native = load_file(str(tmp_path / "native.safetensors"))
    a0, b0 = a.tensors()[key + ".lora_a"], a.tensors()[key + ".lora_b"]
    assert np.allclose(
        native["diffusion_model." + key + ".lora_B.weight"]
        @ native["diffusion_model." + key + ".lora_A.weight"],
        3 * (a0 @ b0).T,
    )


@pytest.mark.parametrize(
    "bad", ["dora_scale", "lora_magnitude_vector", "lokr_w1", "lora_A.weight"]
)
def test_unknown_or_partial_format_rejected(tmp_path, bad):
    source = tmp_path / "bad.safetensors"
    save_file(
        {"transformer_blocks.0.attn.to_q." + bad: np.ones((2, 4), np.float32)},
        str(source),
    )
    with pytest.raises(ValueError):
        convert_media_lora(
            source,
            tmp_path / "out",
            family="qwen-image-2.1",
            base_fingerprint=BASE,
            backend_revision=REV,
            alpha=2,
        )
    assert not (tmp_path / "out").exists()


def test_fail_before_mutation_and_tamper(tmp_path):
    m, key = model()
    original = dict(m.named_modules())[key]
    a = artifact(tmp_path / "a", "qwen-image-2.1", key, inputs=5)
    with pytest.raises(ValueError, match="shape"):
        install_media_lora(m, a)
    assert dict(m.named_modules())[key] is original
    with pytest.raises(ValueError, match="identity"):
        inspect_media_lora(
            a.path,
            family="qwen-image-2.1",
            base_fingerprint="c" * 64,
            backend_revision=REV,
        )
    (a.path / "adapters.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        install_media_lora(m, a)
    assert dict(m.named_modules())[key] is original


def test_nonfinite_rejected(tmp_path):
    _m, key = model()
    with pytest.raises(ValueError, match="finite"):
        artifact(tmp_path / "a", "qwen-image-2.1", key, b=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        artifact(tmp_path / "b", "qwen-image-2.1", key, scale=float("inf"))


class TrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        b = nn.Module()
        b.attn = nn.Module()
        b.attn.to_q = nn.Linear(4, 3, bias=False)
        b.attn.to_q.weight = mx.zeros((3, 4))
        self.transformer_blocks = [b]

    def __call__(self, hidden_states):
        return self.transformer_blocks[0].attn.to_q(hidden_states)


def test_training_only_lora_export_restore_and_resume(tmp_path):
    m = TrainModel()
    key = "transformer_blocks.0.attn.to_q"
    original = m.transformer_blocks[0].attn.to_q
    before = mx.array(original.weight)
    train = FlowExample(
        "qwen-image-2.1",
        BASE,
        {"hidden_states": mx.ones((1, 2, 4))},
        (mx.ones((1, 2, 3)),),
    )
    held = FlowExample(
        "qwen-image-2.1",
        BASE,
        {"hidden_states": mx.full((1, 2, 4), 0.5)},
        (mx.full((1, 2, 3), 0.5),),
    )
    result = train_media_lora(
        m,
        family="qwen-image-2.1",
        base_fingerprint=BASE,
        backend_revision=REV,
        keys=[key],
        examples=[train],
        validation_examples=[held],
        output=tmp_path / "trained",
        steps=20,
        learning_rate=0.01,
    )
    assert result.config["training"]["trained"]
    assert (
        result.config["training"]["final_validation_loss"]
        < result.config["training"]["initial_validation_loss"]
    )
    assert mx.array_equal(original.weight, before).item()
    assert m.transformer_blocks[0].attn.to_q is original
    assert "weight" in original.trainable_parameters()
    second = train_media_lora(
        m,
        family="qwen-image-2.1",
        base_fingerprint=BASE,
        backend_revision=REV,
        keys=[key],
        examples=[train],
        validation_examples=[held],
        output=tmp_path / "resumed",
        steps=2,
        learning_rate=0.001,
        resume=result.path,
    )
    assert second.config["training"]["resumed_from"] == result.fingerprint
    assert mx.array_equal(original.weight, before).item()


def test_training_failure_restores_model_and_flags(tmp_path):
    m = TrainModel()
    key = "transformer_blocks.0.attn.to_q"
    original = m.transformer_blocks[0].attn.to_q
    original.freeze()
    bad = FlowExample(
        "qwen-image-2.1",
        BASE,
        {"hidden_states": mx.ones((1, 2, 4))},
        (mx.ones((1, 2, 7)),),
    )
    with pytest.raises(ValueError, match="shape"):
        train_media_lora(
            m,
            family="qwen-image-2.1",
            base_fingerprint=BASE,
            backend_revision=REV,
            keys=[key],
            examples=[bad],
            validation_examples=[bad],
            output=tmp_path / "bad",
        )
    assert m.transformer_blocks[0].attn.to_q is original
    assert not original.trainable_parameters()
    assert not (tmp_path / "bad").exists()


def test_flow_signs_and_bounds():
    clean, noise = mx.ones((1, 128, 2)), mx.zeros((1, 128, 2))
    music = music_flow_example(
        clean, noise, mx.zeros((1, 2, 2048)), time=0.25, base_fingerprint=BASE
    )
    assert mx.array_equal(music.inputs["hidden"], mx.full(clean.shape, 0.25)).item()
    assert mx.array_equal(music.targets[0], clean).item()
    image = image_flow_example(
        mx.ones((1, 4, 3)),
        mx.zeros((1, 4, 3)),
        mx.zeros((1, 2, 8)),
        time=0.25,
        img_shape=(1, 2, 2),
        base_fingerprint=BASE,
    )
    assert mx.array_equal(image.targets[0], -mx.ones((1, 4, 3))).item()
    with pytest.raises(ValueError):
        music_flow_example(
            clean, noise, mx.zeros((1, 3, 2048)), time=0.25, base_fingerprint=BASE
        )


def test_manifest_tamper_rejected_after_inspection(tmp_path):
    m, key = model()
    a = artifact(tmp_path / "a", "qwen-image-2.1", key)
    data = json.loads((a.path / "adapter_config.json").read_text())
    data["scale"] = 42
    (a.path / "adapter_config.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest changed"):
        install_media_lora(m, a)


def test_multi_backend_failure_rolls_back(tmp_path):
    from mlx2.adapters.media_lora_control import MediaLoRAControl

    first, key = model()
    second, _ = model()
    second.transformer_blocks[0].attn.to_q = nn.Linear(5, 3)
    original = first.transformer_blocks[0].attn.to_q
    a = artifact(tmp_path / "a", "qwen-image-2.1", key)

    class Owner(MediaLoRAControl):
        def _lora_models(self):
            return [first, second]

    owner = Owner()
    owner.artifact = SimpleNamespace(fingerprint=BASE)
    owner._init_lora("qwen-image-2.1", REV)
    with pytest.raises(ValueError, match="shape"):
        owner.load_lora(a.path)
    assert first.transformer_blocks[0].attn.to_q is original
    assert owner.lora_receipt()["state_epoch"] == 0
    assert not owner.lora_receipt()["selected"]


def test_qwen_generation_edit_lora_effect_and_epochs(tmp_path, monkeypatch):
    import sys
    import types

    from mlx2.adapters import generative_media as media

    image_request = types.ModuleType("mlx_vlm.generate.image")
    edit_request = types.ModuleType("mlx_vlm.generate.edit_image")
    image_request.ImageGenerationRequest = lambda **kw: SimpleNamespace(**kw)
    edit_request.ImageEditRequest = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate.image", image_request)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate.edit_image", edit_request)
    monkeypatch.setattr(
        media,
        "inspect_qwen_image21",
        lambda p: media.MediaArtifact("qwen-image-2.1", tmp_path, "s", BASE),
    )
    backends = []

    def factory(path, edit):
        m, _key = model()
        m.transformer_blocks[0].attn.to_q.weight = mx.zeros((3, 4))

        class Backend:
            pipeline = SimpleNamespace(transformer=m)

            def generate(self, request):
                value = int(
                    mx.sum(
                        self.pipeline.transformer.transformer_blocks[0].attn.to_q(
                            mx.ones((1, 4))
                        )
                    ).item()
                    * 10
                )
                return SimpleNamespace(array=np.full((256, 256, 3), value, np.uint8))

            edit = generate

        b = Backend()
        backends.append(b)
        return b

    owner = media.QwenImage21Adapter(tmp_path, backend_factory=factory)
    reference = owner.generate_image("test", width=256, height=256)
    a = write_media_lora(
        tmp_path / "a",
        {
            "transformer_blocks.0.attn.to_q.lora_a": np.ones((4, 2), np.float32) * 0.1,
            "transformer_blocks.0.attn.to_q.lora_b": np.ones((2, 3), np.float32),
        },
        family="qwen-image-2.1",
        base_fingerprint=BASE,
        backend_revision=media.QWEN_BACKEND_REVISION,
        scale=0.7,
    )
    loaded = owner.load_lora(a.path)
    assert (
        loaded["selected"] and not loaded["observed_used"] and not loaded["qualified"]
    )
    changed = owner.generate_image("test", width=256, height=256)
    assert reference.data != changed.data and changed.lora_fingerprint == a.fingerprint
    assert owner.lora_receipt()["observed_used"]
    image = tmp_path / "reference.png"
    image.write_bytes(reference.data)
    edited = owner.edit_image("test", [image], width=256, height=256)
    assert edited.data == changed.data
    assert len(owner._lora_sessions) == 2
    assert owner.unload_lora()["state_epoch"] == 2
    restored = owner.generate_image("test", width=256, height=256)
    assert restored.data == reference.data and restored.lora_fingerprint is None
    assert (
        owner.edit_image("test", [image], width=256, height=256).data == reference.data
    )


def test_music_generation_backend_contract_and_lora(tmp_path, monkeypatch):
    import io
    import wave

    from mlx2.adapters import music3
    from mlx2.adapters.generative_media import MediaArtifact

    monkeypatch.setattr(
        music3,
        "inspect_music3",
        lambda p: MediaArtifact("minimax-music3", tmp_path, "s", BASE),
    )
    m, key = model("minimax-music3")
    m.blocks[0].attn.to_qkv.weight = mx.zeros((3, 4))

    def generate(caption, lyrics, **kw):
        value = m.blocks[0].attn.to_qkv(mx.ones((1, 4))).mean().item()
        return np.full((2, 64), value, np.float32), min(2, kw["max_frames"])

    (tmp_path / "mlx2-music3-manifest.json").write_text(json.dumps({"files": {}}))
    adapter = music3.Music3Adapter(
        tmp_path,
        runtime_root=tmp_path,
        backend_factory=lambda p: SimpleNamespace(dit=m, generate=generate),
    )
    reference = adapter.generate_music("test")
    a = write_media_lora(
        tmp_path / "a",
        {
            key + ".lora_a": np.ones((4, 2), np.float32) * 0.1,
            key + ".lora_b": np.ones((2, 3), np.float32),
        },
        family="minimax-music3",
        base_fingerprint=BASE,
        backend_revision=music3.MUSIC3_RUNTIME_REVISION,
        scale=0.7,
    )
    adapter.load_lora(a.path)
    changed = adapter.generate_music("test")
    assert changed.data != reference.data and changed.lora_fingerprint == a.fingerprint
    with wave.open(io.BytesIO(changed.data)) as f:
        assert (f.getnchannels(), f.getframerate(), f.getnframes()) == (2, 32000, 64)
    adapter.unload_lora()
    assert adapter.generate_music("test").data == reference.data
    with pytest.raises(ValueError):
        adapter.generate_music("test", max_frames=201)
    adapter._backend.generate = lambda *a, **kw: (np.full((2, 64), float("nan")), 2)
    with pytest.raises(ValueError, match="waveform"):
        adapter.generate_music("test")


def test_ltx_native_forwarding_failure_and_unload(tmp_path, monkeypatch):
    from mlx2.adapters import generative_media
    from mlx2.adapters.generative_media import LTX_RUNTIME_REVISION, LTX25Adapter

    owner = object.__new__(LTX25Adapter)
    owner.artifact = SimpleNamespace(fingerprint=BASE)
    owner._init_lora("ltx-2.5", LTX_RUNTIME_REVISION)
    owner.mlx_model = tmp_path / "model"
    owner.mlx_model.mkdir()
    owner.runtime_root = tmp_path
    owner.executable = tmp_path / "python"
    owner._conversion_identity = {}
    owner._bound_inputs = set()
    key = "transformer_blocks.0.attn1.to_q"
    save_file(
        {"transformer." + key + ".weight": np.zeros((3, 4), np.float32)},
        str(owner.mlx_model / "transformer-distilled.safetensors"),
    )
    a = write_media_lora(
        tmp_path / "a",
        {
            key + ".lora_a": np.ones((4, 2), np.float32),
            key + ".lora_b": np.ones((2, 3), np.float32),
        },
        family="ltx-2.5",
        base_fingerprint=BASE,
        backend_revision=LTX_RUNTIME_REVISION,
        scale=3,
    )
    owner._bound_inputs = {"transformer-distilled.safetensors"}
    seen = []
    fail = [False]

    def run(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(
                stdout=LTX_RUNTIME_REVISION + "\n" if "rev-parse" in command else ""
            )
        if "--output" not in command:
            return SimpleNamespace(returncode=0, stdout="")
        if "--lora" in command:
            native = load_file(command[command.index("--lora") + 1])
            assert np.array_equal(
                native["diffusion_model." + key + ".lora_B.weight"],
                np.full((3, 2), 3, np.float32),
            )
            assert command[command.index("--lora") + 2] == "1.0"
        seen.append(command)
        if not fail[0]:
            from pathlib import Path

            Path(command[command.index("--output") + 1]).write_bytes(b"mp4")
        return SimpleNamespace(returncode=int(fail[0]), stderr="failed")

    monkeypatch.setattr(generative_media.subprocess, "run", run)
    owner.load_lora(a.path)
    result = owner.generate_video(
        "test", output=tmp_path / "good.mp4", width=256, height=256, frames=9
    )
    assert (
        result.lora_fingerprint == a.fingerprint
        and owner.lora_receipt()["observed_used"]
    )
    fail[0] = True
    with pytest.raises(RuntimeError, match="failed"):
        owner.generate_video(
            "test", output=tmp_path / "bad.mp4", width=256, height=256, frames=9
        )
    assert not (tmp_path / "bad.mp4").exists()
    owner.unload_lora()
    fail[0] = False
    result = owner.generate_video(
        "test", output=tmp_path / "ordinary.mp4", width=256, height=256, frames=9
    )
    assert "--lora" not in seen[-1] and result.lora_fingerprint is None


def test_ltx_quantized_targets_fail_closed(tmp_path, monkeypatch):
    from mlx2.adapters.generative_media import LTX_RUNTIME_REVISION, LTX25Adapter

    owner = object.__new__(LTX25Adapter)
    owner.artifact = SimpleNamespace(fingerprint=BASE)
    owner._init_lora("ltx-2.5", LTX_RUNTIME_REVISION)
    owner.mlx_model = tmp_path
    monkeypatch.setattr(owner, "_verify_execution_identity", lambda: None)
    key = "transformer_blocks.0.attn1.to_q"
    save_file(
        {
            "transformer." + key + ".weight": np.zeros((3, 4), np.float32),
            "transformer." + key + ".scales": np.ones((3, 1), np.float32),
        },
        str(tmp_path / "transformer-distilled.safetensors"),
    )
    a = write_media_lora(
        tmp_path / "a",
        {
            key + ".lora_a": np.ones((4, 2), np.float32),
            key + ".lora_b": np.ones((2, 3), np.float32),
        },
        family="ltx-2.5",
        base_fingerprint=BASE,
        backend_revision=LTX_RUNTIME_REVISION,
    )
    with pytest.raises(ValueError, match="quantized"):
        owner.load_lora(a.path)
    assert not owner.lora_receipt()["selected"]


def test_music_manifest_binds_all_shards_and_refuses_partials(tmp_path):
    import hashlib

    from mlx2.adapters.music3 import MUSIC3_SOURCE_REVISION, inspect_music3

    files = {
        "flowmatching_vae.pth": b"p",
        "dav.pth": b"d",
        "qwen_7B/qwen_7B/config.json": b"{}",
        "qwen_7B/qwen_7B/model.safetensors.index.json": json.dumps(
            {"weight_map": {"a": "model.safetensors"}}
        ).encode(),
        "qwen_7B/qwen_7B/model.safetensors": b"s",
        "qwen_7B/qwen3-8B-tokenizer-music/tokenizer.json": b"{}",
    }
    records = {}
    for name, value in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        records[name] = {
            "size": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    proof = {
        "source_repo": "MiniMaxAI/MiniMax-Music3",
        "source_revision": MUSIC3_SOURCE_REVISION,
        "files": records,
    }
    manifest = tmp_path / "mlx2-music3-manifest.json"
    manifest.write_text(json.dumps(proof))
    assert inspect_music3(tmp_path).execution_qualified is False
    (tmp_path / "partial.aria2").write_bytes(b"x")
    with pytest.raises(ValueError, match="partial"):
        inspect_music3(tmp_path)
    (tmp_path / "partial.aria2").unlink()
    del records["qwen_7B/qwen_7B/model.safetensors"]
    records["qwen_7B/qwen_7B/another.safetensors"] = {
        "size": 1,
        "sha256": hashlib.sha256(b"s").hexdigest(),
    }
    manifest.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="coverage"):
        inspect_music3(tmp_path)


def test_training_heldout_rejection_rolls_back_after_updates(tmp_path):
    m = TrainModel()
    key = "transformer_blocks.0.attn.to_q"
    original = m.transformer_blocks[0].attn.to_q
    before = mx.array(original.weight)
    train = FlowExample(
        "qwen-image-2.1",
        BASE,
        {"hidden_states": mx.ones((1, 2, 4))},
        (mx.ones((1, 2, 3)),),
    )
    held = FlowExample(
        "qwen-image-2.1",
        BASE,
        {"hidden_states": mx.ones((1, 2, 4))},
        (-mx.ones((1, 2, 3)),),
    )
    with pytest.raises(ValueError, match="held-out loss gate failed"):
        train_media_lora(
            m,
            family="qwen-image-2.1",
            base_fingerprint=BASE,
            backend_revision=REV,
            keys=[key],
            examples=[train],
            validation_examples=[held],
            output=tmp_path / "rejected",
            steps=4,
            learning_rate=0.01,
        )
    assert m.transformer_blocks[0].attn.to_q is original
    assert mx.array_equal(original.weight, before).item()
    assert not (tmp_path / "rejected").exists()


def test_training_export_failure_restores_prior_training_flags(tmp_path):
    m = TrainModel()
    m.eval()
    m.transformer_blocks[0].attn.train(True)
    prior = {key: module.training for key, module in m.named_modules()}
    key = "transformer_blocks.0.attn.to_q"
    original = m.transformer_blocks[0].attn.to_q
    train = FlowExample(
        "qwen-image-2.1",
        BASE,
        {"hidden_states": mx.ones((1, 2, 4))},
        (mx.ones((1, 2, 3)),),
    )
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("unrelated")
    with pytest.raises(FileExistsError):
        train_media_lora(
            m,
            family="qwen-image-2.1",
            base_fingerprint=BASE,
            backend_revision=REV,
            keys=[key],
            examples=[train],
            validation_examples=[train],
            output=existing,
            steps=2,
        )
    assert m.transformer_blocks[0].attn.to_q is original
    assert {key: module.training for key, module in m.named_modules()} == prior
    assert (existing / "keep").read_text() == "unrelated"


def test_bf16_conversion_and_peft_metadata(tmp_path):
    source = tmp_path / "peft.safetensors"
    key = "transformer_blocks.0.attn.to_q"
    mx.save_safetensors(
        str(source),
        {
            key + ".lora_A.weight": mx.full((2, 4), 0.5, dtype=mx.bfloat16),
            key + ".lora_B.weight": mx.full((3, 2), 0.25, dtype=mx.bfloat16),
        },
    )
    config = tmp_path / "adapter_config.json"
    config.write_text(
        json.dumps(
            {
                "r": 2,
                "lora_alpha": 4,
                "alpha_pattern": {},
                "rank_pattern": {},
                "use_rslora": False,
                "bias": "none",
            }
        )
    )
    result = convert_media_lora(
        source,
        tmp_path / "converted",
        family="qwen-image-2.1",
        base_fingerprint=BASE,
        backend_revision=REV,
        alpha=4,
    )
    assert result.config["scale"] == 2
    assert np.array_equal(result.tensors()[key + ".lora_a"], np.full((4, 2), 0.5))
    for metadata, alpha, match in [
        ({"r": 3}, 4, "rank"),
        ({"lora_alpha": 8}, 4, "alpha"),
    ]:
        config.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match=match):
            convert_media_lora(
                source,
                tmp_path / "bad",
                family="qwen-image-2.1",
                base_fingerprint=BASE,
                backend_revision=REV,
                alpha=alpha,
            )


@pytest.mark.parametrize(
    "source,target",
    [
        ("transformer.layers.2.self_attn.to_qkv", "blocks.2.attn.to_qkv"),
        ("transformer.layers.2.self_attn.to_out", "blocks.2.attn.to_out"),
        ("transformer.layers.2.ff.ff.0.proj", "blocks.2.ff_in"),
        ("transformer.layers.2.ff.ff.2", "blocks.2.ff_out"),
    ],
)
def test_music_reference_target_mapping(source, target):
    assert target_key("minimax-music3", source) == target
    with pytest.raises(ValueError):
        target_key("minimax-music3", "diffusion_transformer." + source)


def test_artifact_consumes_same_bytes_it_hashes(tmp_path, monkeypatch):
    from pathlib import Path

    from safetensors.numpy import save

    _m, key = model()
    a = artifact(tmp_path / "a", "qwen-image-2.1", key)
    weights = a.path / "adapters.safetensors"
    original = Path.read_bytes
    replacement = save(
        {
            key + ".lora_a": np.full((4, 2), 9, np.float32),
            key + ".lora_b": np.full((2, 3), 9, np.float32),
        }
    )

    def read(path):
        data = original(path)
        if path == weights:
            weights.write_bytes(replacement)
        return data

    monkeypatch.setattr(Path, "read_bytes", read)
    assert np.array_equal(a.tensors()[key + ".lora_b"], np.ones((2, 3)))
    with pytest.raises(ValueError, match="weights changed"):
        a.tensors()


def test_trained_requires_complete_loss_evidence(tmp_path):
    _m, key = model()
    with pytest.raises(ValueError, match="training evidence"):
        write_media_lora(
            tmp_path / "a",
            {
                key + ".lora_a": np.ones((4, 2), np.float32),
                key + ".lora_b": np.ones((2, 3), np.float32),
            },
            family="qwen-image-2.1",
            base_fingerprint=BASE,
            backend_revision=REV,
            training={"trained": True},
        )
    assert not (tmp_path / "a").exists()


def test_editor_only_lora_load_and_lazy_generator(tmp_path, monkeypatch):
    from mlx2.adapters import generative_media as media

    monkeypatch.setattr(
        media,
        "inspect_qwen_image21",
        lambda p: media.MediaArtifact("qwen-image-2.1", tmp_path, "s", BASE),
    )
    calls = []

    def factory(path, edit):
        calls.append(edit)
        m, _ = model()
        return SimpleNamespace(
            pipeline=SimpleNamespace(transformer=m),
            generate=lambda request: SimpleNamespace(
                array=np.zeros((256, 256, 3), np.uint8)
            ),
        )

    owner = media.QwenImage21Adapter(tmp_path, backend_factory=factory)
    owner._editor = owner._model(edit=True)
    key = "transformer_blocks.0.attn.to_q"
    a = write_media_lora(
        tmp_path / "a",
        {
            key + ".lora_a": np.ones((4, 2), np.float32),
            key + ".lora_b": np.ones((2, 3), np.float32),
        },
        family="qwen-image-2.1",
        base_fingerprint=BASE,
        backend_revision=media.QWEN_BACKEND_REVISION,
    )
    owner.load_lora(a.path)
    assert (
        calls == [True] and owner._generator is None and len(owner._lora_sessions) == 1
    )
    owner.generate_image("test", width=256, height=256)
    assert calls == [True, False] and len(owner._lora_sessions) == 2
    owner.unload_lora()
    assert all(
        not hasattr(b.pipeline.transformer.transformer_blocks[0].attn.to_q, "lora_a")
        for b in (owner._editor, owner._generator)
    )


@pytest.mark.parametrize(
    "shadow", ["transformer.safetensors", "transformer-distilled-1.1.safetensors"]
)
def test_ltx_rejects_native_checkpoint_shadows(tmp_path, shadow):
    from mlx2.adapters.generative_media import LTX25Adapter

    owner = object.__new__(LTX25Adapter)
    owner.mlx_model = tmp_path
    (tmp_path / shadow).write_bytes(b"shadow")
    with pytest.raises(ValueError, match="shadow"):
        owner._verify_execution_identity()


def test_music_lazy_loader_rejects_changed_or_new_inputs(tmp_path, monkeypatch):
    from mlx2.adapters import music3

    monkeypatch.setattr(
        music3,
        "inspect_music3",
        lambda p: music3.MediaArtifact("minimax-music3", tmp_path, "s", BASE),
    )
    (tmp_path / "flowmatching_vae.pth").write_bytes(b"original")
    (tmp_path / "mlx2-music3-manifest.json").write_text(
        json.dumps({"files": {"flowmatching_vae.pth": {}}})
    )
    calls = []
    owner = music3.Music3Adapter(
        tmp_path, runtime_root=tmp_path, backend_factory=lambda p: calls.append(p)
    )
    (tmp_path / "flowmatching_vae.pth").write_bytes(b"changed!")
    with pytest.raises(ValueError, match="changed"):
        owner._ensure_backend()
    assert not calls and owner._backend is None


def test_generation_and_unload_serialize(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from mlx2.adapters import generative_media as media

    monkeypatch.setattr(
        media,
        "inspect_qwen_image21",
        lambda p: media.MediaArtifact("qwen-image-2.1", tmp_path, "s", BASE),
    )
    entered, release, unload_started = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    m, key = model()

    def generate(request):
        assert hasattr(m.transformer_blocks[0].attn.to_q, "lora_a")
        entered.set()
        assert release.wait(5)
        assert hasattr(m.transformer_blocks[0].attn.to_q, "lora_a")
        return SimpleNamespace(array=np.zeros((256, 256, 3), np.uint8))

    owner = media.QwenImage21Adapter(
        tmp_path,
        backend_factory=lambda path, edit: SimpleNamespace(
            pipeline=SimpleNamespace(transformer=m), generate=generate
        ),
    )
    a = write_media_lora(
        tmp_path / "a",
        {
            key + ".lora_a": np.ones((4, 2), np.float32),
            key + ".lora_b": np.ones((2, 3), np.float32),
        },
        family="qwen-image-2.1",
        base_fingerprint=BASE,
        backend_revision=media.QWEN_BACKEND_REVISION,
    )
    owner.load_lora(a.path)

    def unload():
        unload_started.set()
        return owner.unload_lora()

    with ThreadPoolExecutor(max_workers=2) as pool:
        render = pool.submit(owner.generate_image, "test", width=256, height=256)
        try:
            assert entered.wait(5)
            removing = pool.submit(unload)
            assert unload_started.wait(5)
            assert not removing.done()
        finally:
            release.set()
        assert render.result().lora_fingerprint == a.fingerprint
        assert not removing.result()["selected"]
    assert not hasattr(m.transformer_blocks[0].attn.to_q, "lora_a")


def test_ltx_rejects_dirty_runtime_and_checks_import_origin(tmp_path, monkeypatch):
    from mlx2.adapters import generative_media as media

    owner = object.__new__(media.LTX25Adapter)
    owner.mlx_model = tmp_path
    owner.runtime_root = tmp_path
    owner.executable = tmp_path / "python"
    owner._conversion_identity = {}
    owner._bound_inputs = set()
    calls = []
    dirty = [True]

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            stdout=media.LTX_RUNTIME_REVISION
            if "rev-parse" in command
            else (" M packages/a.py" if dirty[0] and "status" in command else "")
        )

    monkeypatch.setattr(media.subprocess, "run", run)
    with pytest.raises(ValueError, match="packages"):
        owner._verify_execution_identity()
    assert len(calls) == 2
    dirty[0] = False
    owner._verify_execution_identity()
    assert calls[-1][0] == str(owner.executable)
    assert "__file__" in calls[-1][2]


def test_music_pins_package_initialization_before_import(tmp_path, monkeypatch):
    from mlx2.adapters import music3

    monkeypatch.setattr(music3, "SOURCE_SHA256", {})
    monkeypatch.setenv("MM3_MLX_LM_UNIFIED", str(tmp_path / "unified"))
    package = tmp_path / "unified/mlx_lm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="Python package"):
        music3._runtime_modules(tmp_path)


@pytest.mark.parametrize(
    "extra",
    [
        "spatial_upscaler_x2_v1_1.safetensors",
        "spatial_upscaler_x2-1.1.safetensors",
        "text_encoder/model-extra.safetensors",
        "text_encoder/tokenizer_config.json",
    ],
)
def test_ltx_all_loader_inputs_must_be_bound(tmp_path, extra):
    from mlx2.adapters.generative_media import LTX25Adapter

    owner = object.__new__(LTX25Adapter)
    owner.mlx_model = tmp_path
    owner._bound_inputs = {"spatial_upscaler_x2.safetensors"}
    (tmp_path / "spatial_upscaler_x2.safetensors").write_bytes(b"bound")
    owner._verify_input_listing()
    path = tmp_path / extra
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"unbound")
    with pytest.raises(ValueError, match="unbound"):
        owner._verify_input_listing()


def test_ltx_origin_probe_survives_optimization_and_stale_bytecode(tmp_path):
    import importlib.util
    import os
    import py_compile
    import struct
    import subprocess
    import sys

    from mlx2.adapters.generative_media import _ltx_origin_probe

    root = tmp_path / "runtime"
    packages = root / "packages"
    for name in ("ltx_core_mlx", "ltx_pipelines_mlx"):
        folder = packages / name
        folder.mkdir(parents=True)
        (folder / "__init__.py").write_text("MARKER = 'source'\n")
    source = packages / "ltx_core_mlx/__init__.py"
    evil = tmp_path / "evil.py"
    evil.write_text("raise RuntimeError('stale compiled code')\n")
    cached = importlib.util.cache_from_source(str(source))
    py_compile.compile(str(evil), cfile=cached, doraise=True)
    from pathlib import Path

    data = Path(cached).read_bytes()
    st = source.stat()
    Path(cached).write_bytes(
        data[:8] + struct.pack("<II", int(st.st_mtime), st.st_size) + data[16:]
    )
    env = {**os.environ, "PYTHONPATH": str(packages)}
    env.pop("PYTHONPYCACHEPREFIX", None)
    unguarded = subprocess.run(
        [sys.executable, "-c", "import ltx_core_mlx"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert unguarded.returncode != 0 and "stale compiled code" in unguarded.stderr
    guarded = subprocess.run(
        [sys.executable, "-c", _ltx_origin_probe(), str(root)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert guarded.returncode == 0, guarded.stderr
    wrong_root = tmp_path / "other"
    wrong_root.mkdir()
    optimized = subprocess.run(
        [sys.executable, "-O", "-c", _ltx_origin_probe(), str(wrong_root)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert optimized.returncode != 0 and "origin differs" in optimized.stderr


@pytest.mark.parametrize(
    "metadata",
    [
        {"lora_alpha": "8"},
        {"transformer.lora_adapter_metadata": json.dumps({"r": 2, "lora_alpha": 8})},
        {"network_alphas": json.dumps({"transformer.block.alpha": 8})},
        {"rank": "3"},
        {"lora_adapter_metadata": json.dumps({"use_rslora": True})},
        {"lora_adapter_metadata": json.dumps({"use_dora": True})},
        {"lora_adapter_metadata": json.dumps({"alpha_pattern": {"block": 4}})},
        {"lora_adapter_metadata": json.dumps({"rank_pattern": {"block": 2}})},
    ],
)
def test_embedded_alpha_and_rank_metadata_cannot_be_ignored(tmp_path, metadata):
    key = "transformer_blocks.0.attn.to_q"
    path = tmp_path / "source.safetensors"
    save_file(
        {
            key + ".lora_A.weight": np.ones((2, 4), np.float32),
            key + ".lora_B.weight": np.ones((3, 2), np.float32),
        },
        str(path),
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="embedded PEFT"):
        convert_media_lora(
            path,
            tmp_path / "bad",
            family="qwen-image-2.1",
            base_fingerprint=BASE,
            backend_revision=REV,
            alpha=4,
        )
    assert not (tmp_path / "bad").exists()


def test_embedded_matching_alpha_metadata_converts(tmp_path):
    key = "transformer_blocks.0.attn.to_q"
    path = tmp_path / "source.safetensors"
    save_file(
        {
            key + ".lora_A.weight": np.ones((2, 4), np.float32),
            key + ".lora_B.weight": np.ones((3, 2), np.float32),
        },
        str(path),
        metadata={
            "transformer.lora_adapter_metadata": json.dumps(
                {
                    "r": 2,
                    "lora_alpha": 4,
                    "alpha_pattern": {},
                    "rank_pattern": {},
                    "use_rslora": False,
                    "bias": "none",
                }
            )
        },
    )
    result = convert_media_lora(
        path,
        tmp_path / "good",
        family="qwen-image-2.1",
        base_fingerprint=BASE,
        backend_revision=REV,
        alpha=4,
    )
    assert result.config["scale"] == 2
