"""Regression tests from the 2026-10-06 sweep: non-Qwen adapter findings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

MODELS = Path.home() / "mlx-models"
NEMOTRON = {
    "super": MODELS / "Nemotron-3-Super-120B-A12B-5bit-MTP",
    "lightning": MODELS / "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-mlx-8Bit",
}
CHAT = {"messages": [{"role": "user", "content": "Say hi."}]}


def _nemotron(kind):
    from mlx2.adapters.nemotron35_lightning import Nemotron35LightningAdapter
    from mlx2.adapters.nemotron3_super import Nemotron3SuperAdapter
    from mlx2.runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

    path = NEMOTRON[kind]
    if not (path / "tokenizer.json").is_file():
        pytest.skip("artifact tokenizer absent")
    from transformers import AutoTokenizer

    cls = Nemotron3SuperAdapter if kind == "super" else Nemotron35LightningAdapter
    adapter = object.__new__(cls)
    adapter.tokenizer = TokenizerWrapper(
        AutoTokenizer.from_pretrained(path, local_files_only=True),
        detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=[2, 11])
    adapter.identity = {"fingerprint": "sweep-test", "path": str(path)}
    return adapter


@pytest.mark.parametrize("kind", sorted(NEMOTRON))
def test_nemotron_incremental_render_keeps_thinking_default(kind):
    adapter = _nemotron(kind)
    assert adapter.thinking_enabled(CHAT) is True
    rendered = type(adapter).render_incremental_prompt(adapter.tokenizer._tokenizer, CHAT)
    assert rendered == adapter.render_prompt(CHAT)
    assert adapter.incremental_tokenizer_renderer_revision != "flash-next-renderer-v1"


@pytest.mark.parametrize("kind", sorted(NEMOTRON))
def test_nemotron_incremental_cache_serves_the_ordinary_prompt(kind):
    from mlx2.runtime.incremental_tokenizer_cache import IncrementalPromptTokenizerCache

    adapter = _nemotron(kind)
    cache = IncrementalPromptTokenizerCache(max_entries=8)
    if not cache.bind(adapter):
        pytest.skip(f"bind refused: {cache.status()['refusal']}")
    first = {**CHAT, "enable_thinking": True}
    tokens, _ = cache.tokenize(cache.prepare(first), lambda: adapter.prompt_tokens(first))
    assert tokens == adapter.prompt_tokens(first)
    later = {"messages": CHAT["messages"] + [
        {"role": "assistant", "content": "hi"}, {"role": "user", "content": "Again."}]}
    tokens, receipt = cache.tokenize(cache.prepare(later), lambda: adapter.prompt_tokens(later))
    assert tokens == adapter.prompt_tokens(later), receipt


def _tiny_gemma4():
    pytest.importorskip("mlx_vlm")
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.gemma4.config import ModelConfig, TextConfig, VisionConfig
    from mlx_vlm.models.gemma4.gemma4 import Model

    class StubTower(nn.Module):
        def __call__(self, pixels, position_ids=None):
            return pixels  # the features are the pixels

    text = TextConfig(hidden_size=32, num_hidden_layers=2, intermediate_size=48,
                      num_attention_heads=2, head_dim=16, global_head_dim=16,
                      num_key_value_heads=1, num_kv_shared_layers=0,
                      hidden_size_per_layer_input=0, vocab_size=64,
                      vocab_size_per_layer_input=64, sliding_window=8,
                      use_double_wide_mlp=False)
    vision = VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                          global_head_dim=8, default_output_length=4,
                          pooling_kernel_size=1, patch_size=2, position_embedding_size=16)
    mx.random.seed(0)
    model = Model(ModelConfig(text_config=text, vision_config=vision, image_token_id=7))
    model.vision_tower = StubTower()
    return model


def test_vision_feature_cache_is_bound_to_the_request_lora(tmp_path):
    """Features computed under LoRA A must not be served to a base request
    for the same image (G5-01)."""
    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.gemma4 import Gemma4A4BAdapter, _Gemma4LogitsModel
    from mlx2.adapters.multimodal import MediaFeatureCache
    from mlx2.runtime.multi_lora import (
        MultiLoRAManager, bind_lora_rows, clear_lora_rows, write_adapter,
    )

    model = _tiny_gemma4()
    wrapped = _Gemma4LogitsModel(model)
    manager = MultiLoRAManager(wrapped, max_loras=1, max_lora_rank=4)
    key_name = "embed_vision.embedding_projection"
    write_adapter(tmp_path / "A", keys=[key_name], dims={key_name: (16, 32)}, rank=4, scale=4.0)
    manager.register("A", tmp_path / "A")
    slot, _ = manager.acquire("A")
    manager.bind_uid(1, slot)  # uid 1 = LoRA request, uid 2 = base request

    adapter = object.__new__(Gemma4A4BAdapter)
    ids = mx.array([[1, 7, 7, 7, 7, 2]])
    pixels = mx.random.normal((1, 4, 16))
    cache = MediaFeatureCache()
    prepared = dict(pixel_values=pixels, vision_cache=cache,
                    _image_key="artifact-fp:media-fp:image")
    lora_kwargs = adapter.validate_prefill_inputs(
        {"_mlx2_lora_fingerprint": "lora-a"}, [], dict(prepared))
    base_kwargs = adapter.validate_prefill_inputs({}, [], dict(prepared))
    assert base_kwargs["_image_key"] == prepared["_image_key"]
    assert lora_kwargs["_image_key"] != prepared["_image_key"]

    rows = bind_lora_rows(wrapped, [1])
    lora_embeds = model.get_input_embeddings(ids, **lora_kwargs).inputs_embeds
    clear_lora_rows(rows)
    rows = bind_lora_rows(wrapped, [2])
    base_cached = model.get_input_embeddings(ids, **base_kwargs).inputs_embeds
    base_true = model.get_input_embeddings(ids, pixel_values=pixels).inputs_embeds
    clear_lora_rows(rows)
    assert not np.allclose(np.array(lora_embeds), np.array(base_true))
    np.testing.assert_allclose(np.array(base_cached), np.array(base_true), atol=1e-5)


GEMMA4_31B = MODELS / "gemma-4-31B-MLX-8bit"


@pytest.fixture(scope="module")
def gemma4_media_adapter():
    if not (GEMMA4_31B / "processor_config.json").is_file() and not (
        GEMMA4_31B / "preprocessor_config.json"
    ).is_file():
        pytest.skip("Gemma 4 processor files absent")
    import mlx_vlm.models.gemma4  # noqa: F401  registers the pinned processor
    from mlx_vlm.utils import load_processor

    from mlx2.adapters.gemma4 import Gemma431BAdapter
    from mlx2.adapters.multimodal import MediaFeatureCache

    adapter = object.__new__(Gemma431BAdapter)
    adapter.processor = load_processor(GEMMA4_31B)
    adapter.identity = {"fingerprint": "x"}
    adapter.media_feature_cache = MediaFeatureCache()
    return adapter


def _mp4_data_url(seconds, fps=1.0, size=96):
    import base64
    import tempfile

    import cv2
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
        out = cv2.VideoWriter(handle.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
        for i in range(int(seconds * fps)):
            out.write(np.full((size, size, 3), (i * 7) % 255, np.uint8))
        out.release()
        payload = Path(handle.name).read_bytes()
    return "data:video/mp4;base64," + base64.b64encode(payload).decode()


def test_gemma4_long_video_stamps_follow_the_sampled_frames(gemma4_media_adapter):
    import re

    adapter = gemma4_media_adapter
    request = {"messages": [{"role": "user", "content": [
        {"type": "input_video", "video_url": _mp4_data_url(60)},
        {"type": "text", "text": "What happens?"}]}]}
    prepared = adapter.prepare_multimodal_request(request)
    text = adapter.processor.tokenizer.decode(prepared["_mlx2_prompt_tokens"])
    seconds = [int(m) * 60 + int(s) for m, s in re.findall(r"(\d\d):(\d\d) <\|image>", text)]
    assert len(seconds) == 32
    assert seconds[0] == 0 and seconds[-1] == 59, seconds
    assert seconds == sorted(seconds)


def test_gemma4_literal_media_marker_in_text_is_a_client_error(gemma4_media_adapter):
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (9, 9, 9)).save(buffer, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    request = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "is this the same as <|image|> ?"}]}]}
    with pytest.raises(ValueError, match="placeholder"):
        gemma4_media_adapter.prepare_multimodal_request(request)


def _tiny_llada(seed=0):
    import mlx.core as mx

    from mlx2.runtime.models.llada import Model, ModelArgs

    mx.random.seed(seed)
    return Model(ModelArgs(model_type="llada", d_model=32, n_layers=2, n_heads=4,
                           n_kv_heads=4, mlp_hidden_size=64, vocab_size=64,
                           embedding_size=64, weight_tying=False))


def test_llada_runtime_refuses_a_prompt_holding_the_mask_id():
    """A prompt mask is a denoising slot to the reference loop: it spent the
    block's reveal budget, left masks in the output and made the 'exact'
    active-block route diverge (G4-01)."""
    import mlx.core as mx

    from mlx2.runtime.models.llada import generate

    model = _tiny_llada()
    common = dict(steps=4, gen_length=4, block_length=4, mask_id=63, return_stats=True)
    with pytest.raises(ValueError, match="mask token"):
        generate(model, mx.array([[1, 63, 63, 4]]), **common)
    out, stats = generate(model, mx.array([[1, 2, 3, 4]]), **common)
    assert stats["revealed"] == int(mx.sum(out != 63).item()) == 4


def test_llada_adapter_refuses_a_prompt_holding_the_mask_id():
    from types import SimpleNamespace

    from mlx2.adapters.llada import LLaDADenoisingAdapter

    adapter = object.__new__(LLaDADenoisingAdapter)
    adapter.model = object()
    adapter.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [1, 126336, 2])
    adapter.config = {"mask_token_id": 126336, "eos_token_id": 126081}
    with pytest.raises(ValueError, match="mask token"):
        adapter.generate(prompt="<|mdm_mask|>", gen_length=8, block_length=8, steps=8)


@pytest.mark.parametrize(
    "module,name,expected",
    [
        # Hy3 generation_config.json: temperature 0.9, top_p 1, top_k -1 (off).
        ("hy_v3", "HYV3Adapter", {"temperature": 0.9, "top_p": 1.0, "top_k": 0}),
        # Agnes-3.0-Flash-Preview generation_config.json.
        ("agnes_3_flash", "Agnes3FlashAdapter",
         {"temperature": 1.0, "top_p": 0.95, "top_k": 20}),
        # LFM2.5-VL-3B generation_config.json and model card.
        ("lfm25_vl", "LFM25VLAdapter",
         {"temperature": 0.2, "top_k": 50, "repetition_penalty": 1.0}),
    ],
)
def test_adapter_declares_its_generation_config_sampling(module, name, expected):
    import importlib

    from mlx2.sampling_defaults import resolve_sampling, vendor_sampling

    cls = getattr(importlib.import_module(f"mlx2.adapters.{module}"), name)
    effective, record = resolve_sampling({}, vendor_sampling(object.__new__(cls)), thinking=False)
    assert record["fallback_kind"] == "neutral"
    assert {key: effective[key] for key in expected} == expected


def test_hy_v3_runtime_defaults_follow_the_reference_config():
    """HF HYV3Config defaults router_scaling_factor to 2.826 and
    enable_moe_fp32_combine to True; mlx2 defaulted 1.0 / False (G1-03)."""
    from mlx2.runtime.models.hy_v3 import ModelArgs

    args = ModelArgs.from_dict({
        "model_type": "hy_v3", "vocab_size": 64, "hidden_size": 32,
        "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "num_experts": 4,
        "num_experts_per_tok": 2, "num_shared_experts": 1, "expert_hidden_dim": 16,
        "first_k_dense_replace": 1, "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    })
    assert (args.router_scaling_factor, args.enable_moe_fp32_combine) == (2.826, True)


def test_hy_v3_runtime_carries_no_dead_mtp_surface():
    """The model never builds an MTP sidecar, so mtp_step / make_mtp_cache /
    the remap were unreachable and the provenance overclaimed a loader."""
    from mlx2.runtime.models import hy_v3

    for name in ("mtp_step", "make_mtp_cache", "_remap_mtp_weights"):
        assert not hasattr(hy_v3.Model, name)
    assert not hasattr(hy_v3, "HYV3MTP")
    # The public mirror's provenance/ records are frozen; this record
    # check runs on the source tree only.


HY_FULL = Path("/Volumes/T7/models/kernelpool/Hy3-6bit")


@pytest.mark.parametrize("field,value", [
    ("enable_attention_fp32_softmax", True),
    ("router_scaling_factor", 0),
    ("enable_moe_fp32_combine", "yes"),
])
def test_hy_v3_refuses_unimplemented_or_invalid_numerics(tmp_path, field, value):
    if not HY_FULL.is_dir():
        pytest.skip("HY V3 artifact absent")
    from mlx2.adapters.hy_v3 import inspect_artifact

    for name in ("config.json", "model.safetensors.index.json"):
        (tmp_path / name).write_bytes((HY_FULL / name).read_bytes())
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())["weight_map"]
    for name in set(index.values()):
        (tmp_path / name).write_bytes(b"metadata")
    config = json.loads((tmp_path / "config.json").read_text())
    inspect_artifact(tmp_path)
    config[field] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="HY V3"):
        inspect_artifact(tmp_path)


def test_gpt_oss_carries_no_dead_speculative_hooks():
    """kv_sink had no consumer and gpt-oss declares no speculative route
    (G1-04); provenance no longer calls reasoning excluded (G1-05)."""
    import inspect

    from mlx2.runtime.models import gpt_oss, gpt_oss_puzzle

    for cls in (gpt_oss.Model, gpt_oss.AttentionBlock, gpt_oss_puzzle.Model):
        assert "kv_sink" not in inspect.signature(cls.__call__).parameters
    assert not getattr(gpt_oss.Model, "supports_speculative_rollback", False)
    # The public mirror's provenance/ records are frozen; this record
    # check runs on the source tree only.


NORTH = MODELS / "North-Mini-Code-1.0-mlx-4bit"


@pytest.mark.skipif(not (NORTH / "config.json").is_file(), reason="North artifact absent")
def test_north_budget_receipt_names_the_transient_admission_uses():
    from mlx2.adapters.north_memory import NorthCacheBudget

    config = json.loads((NORTH / "config.json").read_text())
    receipt = NorthCacheBudget.from_config(config, mtp=False).as_dict()
    assert receipt["workspace"].startswith(f"{receipt['transient_gib_per_lane']:g}-GiB")
    assert "3.1" not in receipt["workspace"]


def test_lab_written_hils_port_has_no_apple_copyright_header():
    """AGENTS.md: never add an Apple copyright header to original project code."""
    root = Path(__file__).resolve().parents[1] / "src" / "mlx2" / "runtime" / "models"
    head = (root / "olmo_hils.py").read_text().splitlines()[:3]
    assert not any("Apple" in line for line in head)
    assert head[0] == "# SPDX-License-Identifier: MIT"
    from mlx2.runtime.models import olmo3

    assert [name for name in vars(olmo3) if name.startswith("Olmo3")] == ["Olmo3MLP"]


@pytest.mark.parametrize("name", ["gpt_oss_puzzle.py", "granitemoe_swa.py"])
def test_lab_written_model_ports_have_no_apple_copyright_header(name):
    """The lab wrote these in mlx-lm-unified; NOTICE carries the mlx-lm notice."""
    root = Path(__file__).resolve().parents[1] / "src" / "mlx2" / "runtime" / "models"
    head = (root / name).read_text().splitlines()[:5]
    assert not any("Apple" in line for line in head)
    assert head[0] == "# SPDX-License-Identifier: MIT"


def _diffusion_gemma(monkeypatch, tmp_path, captured):
    import sys
    import types

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_diffusion_gemma_cpu import fixture

    from mlx2.adapters.diffusion_gemma import DiffusionGemmaAdapter

    def fake_generate(model, processor, formatted, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(text="ok", finish_reason="stop",
                                     diffusion_canvas_tokens=256,
                                     diffusion_denoising_steps=48)

    generate = types.ModuleType("mlx_vlm.generate")
    generate.generate = fake_generate
    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.apply_chat_template = lambda *a, **k: "formatted"
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generate)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)
    fixture(tmp_path)
    model = types.SimpleNamespace(config=types.SimpleNamespace())
    return DiffusionGemmaAdapter(tmp_path, backend_factory=lambda p: (model, None))


def test_diffusion_gemma_defaults_to_the_reference_sampling_law(monkeypatch, tmp_path):
    """HF draws each denoising step from the schedule-scaled softmax over a
    full 256-token canvas; mlx-vlm's default temperature 0 is argmax (G4-02)."""
    captured = {}
    adapter = _diffusion_gemma(monkeypatch, tmp_path, captured)
    result = adapter.generate_text("hello")
    assert captured["temperature"] == 1.0
    assert captured["diffusion_full_canvas"] is True
    assert captured["seed"] == 0
    assert (result.temperature, result.seed, result.full_canvas) == (1.0, 0, True)

    captured.clear()
    argmax = adapter.generate_text("hello", temperature=0.0, seed=None, full_canvas=False)
    assert captured["temperature"] == 0.0 and "seed" not in captured
    assert (argmax.temperature, argmax.seed, argmax.full_canvas) == (0.0, None, False)
    with pytest.raises(ValueError, match="seed"):
        adapter.generate_text("hello", seed=-1)


def _qwen_image_adapter(monkeypatch, tmp_path, seen):
    import sys
    from types import SimpleNamespace

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_generative_media_cpu import _snapshot, _stub_qwen_request_type, _write

    from mlx2.adapters.generative_media import QWEN_REVISION, QwenImage21Adapter

    _stub_qwen_request_type(monkeypatch)
    _write(tmp_path / "model_index.json", b'{"_class_name":"QwenImage21Pipeline"}')
    _write(tmp_path / "transformer/config.json",
           b'{"_class_name":"QwenImage21Transformer2DModel","num_layers":32,"num_attention_heads":32}')
    _snapshot(tmp_path, "Qwen/Qwen-Image-2.1", QWEN_REVISION,
              ["model_index.json", "transformer/config.json"])

    class Backend:
        def generate(self, request):
            seen.append(request)
            return SimpleNamespace(array=np.zeros((256, 256, 3), dtype=np.uint8))

    return QwenImage21Adapter(tmp_path, backend_factory=lambda path, edit: Backend())


def test_qwen_image_receipt_carries_request_parameters_and_bounds(monkeypatch, tmp_path):
    """G4-03/G4-04: the receipt omitted seed and steps; width/height had no
    upper bound and prompt/seed were not type-checked."""
    seen = []
    adapter = _qwen_image_adapter(monkeypatch, tmp_path, seen)
    result = adapter.generate_image("test", width=256, height=256, steps=7, seed=11)
    assert (result.seed, result.steps) == (11, 7)
    for bad in ({"width": 4096}, {"height": 8192}, {"width": 256.0}):
        with pytest.raises(ValueError, match="dimensions"):
            adapter.generate_image("test", **{"width": 256, "height": 256, **bad})
    with pytest.raises(ValueError, match="dimensions"):
        adapter.generate_image(b"test", width=256, height=256)
    with pytest.raises(ValueError, match="seed"):
        adapter.generate_image("test", width=256, height=256, seed=-1)
    assert len(seen) == 1


def test_ltx_prompt_is_one_argv_token_and_receipt_carries_parameters(monkeypatch, tmp_path):
    """G4-07: a prompt starting with '-' parsed as an option; G4-03/04:
    receipts lacked seed/frames/rate and sizes had no cap."""
    import argparse
    from types import SimpleNamespace

    from mlx2.adapters import generative_media
    from mlx2.adapters.generative_media import LTX_RUNTIME_REVISION, LTX25Adapter

    owner = object.__new__(LTX25Adapter)
    owner.artifact = SimpleNamespace(fingerprint="f" * 64)
    owner._init_lora("ltx-2.5", LTX_RUNTIME_REVISION)
    owner.mlx_model = tmp_path / "model"
    owner.runtime_root = tmp_path
    owner.executable = tmp_path / "python"
    monkeypatch.setattr(owner, "_verify_execution_identity", lambda: None)
    seen = []

    def run(command, **kwargs):
        seen.append((command, kwargs.get("input")))
        Path(command[command.index("--output") + 1]).write_bytes(b"mp4")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(generative_media.subprocess, "run", run)
    result = owner.generate_video("-dash first", output=tmp_path / "a.mp4",
                                  width=256, height=256, frames=17, frame_rate=12, seed=5)
    command, prompt = seen[0]
    runner = command[command.index("-c") + 1]
    assert "'--prompt=' + sys.stdin.read()" in runner and prompt == "-dash first"
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt")
    assert parser.parse_args(["--prompt=" + prompt]).prompt == "-dash first"
    assert (result.seed, result.width, result.height, result.frames, result.frame_rate) == (
        5, 256, 256, 17, 12
    )
    for bad in ({"width": 4096}, {"frames": 1025}):
        with pytest.raises(ValueError):
            owner.generate_video("x", output=tmp_path / "b.mp4",
                                 **{"width": 256, "height": 256, "frames": 9, **bad})


def test_music3_receipt_carries_seed_steps_and_dit_seed_law(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    from mlx2.adapters import music3
    from mlx2.adapters.generative_media import MediaArtifact

    monkeypatch.setattr(music3, "inspect_music3",
                        lambda p: MediaArtifact("minimax-music3", tmp_path, "s", "f" * 64))
    (tmp_path / "mlx2-music3-manifest.json").write_text(json.dumps({"files": {}}))
    adapter = music3.Music3Adapter(
        tmp_path, runtime_root=tmp_path,
        backend_factory=lambda p: SimpleNamespace(
            dit=None,
            generate=lambda caption, lyrics, **kw: (np.zeros((2, 64), np.float32), 1),
        ),
    )
    result = adapter.generate_music("test", seed=9, steps=12)
    assert (result.seed, result.steps) == (9, 12)
    assert result.dit_seed_derivation.startswith("single-window")


def test_native_mtp_beam_ends_the_cycle_on_every_branch():
    """G4-06: pruned beams' private caches were never ended, and the head
    cycle was never started (unlike native_mtp_source)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_continuation_providers_cpu import context
    from test_proposal_composition_cpu import _native_qsa_pair

    from mlx2.adapters.proposal_path_sources import NativeMTPContinuationSource

    model, draft = _native_qsa_pair("xpress")
    started, ended = [], []
    real_start, real_end = model.mtp_start_cycle, model.mtp_end_cycle
    model.mtp_start_cycle = lambda cache, **kw: (started.append(id(cache)), real_start(cache, **kw))[1]
    model.mtp_end_cycle = lambda cache: (ended.append(id(cache)), real_end(cache))[1]
    ctx = context(model, draft, history=(1, 2, 3), anchor=17, depth=2)
    paths = NativeMTPContinuationSource(model)(ctx, 4)
    assert len(paths) == 4 and len(started) == 1
    # mtp_start_cycle ends any previous cycle first (ended[0]); after that the
    # head cache, 1 branch at depth 1 and 4 at depth 2 are each ended once.
    assert ended[0] == started[0]
    assert len(ended[1:]) == len(set(ended[1:])) == 6
    assert started[0] in ended[1:]


def test_llada_records_and_applies_its_seed(monkeypatch):
    import sys
    import types

    from mlx2.adapters.llada import LLaDADenoisingAdapter

    class Out(list):
        def tolist(self):
            return list(self)

    seeds = []
    core = types.ModuleType("mlx.core")
    core.array = lambda value: value
    core.random = types.SimpleNamespace(seed=seeds.append)
    runtime = types.ModuleType("mlx2.runtime.models.llada")
    runtime.generate = lambda model, prompt, **kw: ([Out([11])], "x", {"forwards": 1})
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx2.runtime.models.llada", runtime)
    adapter = object.__new__(LLaDADenoisingAdapter)
    adapter.model = object()
    adapter.tokenizer = types.SimpleNamespace(
        encode=lambda text, **kw: [1, 2], convert_tokens_to_ids=lambda value: 126348,
        decode=lambda values, **kw: "done")
    adapter.config = {"mask_token_id": 126336, "eos_token_id": 126081}
    adapter.identity = {"fingerprint": "x"}
    result = adapter.generate(prompt="hi", gen_length=8, block_length=8, steps=8,
                              temperature=0.5, seed=3)
    assert seeds == [3] and (result["seed"], result["temperature"]) == (3, 0.5)


def _muse_pair(dtype):
    pytest.importorskip("mlx_vlm")
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_map
    from mlx_vlm.models.muse_glimmer.config import TextConfig
    from mlx_vlm.models.muse_glimmer.language import LanguageModel

    from mlx2.adapters.muse_glimmer_config import ModelArgs
    from mlx2.runtime.models.muse_glimmer import Model

    types = (["sliding_attention"] * 3 + ["full_attention"]) * 2
    common = dict(hidden_size=64, intermediate_size=96, num_hidden_layers=8,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                  vocab_size=256, sliding_window=8, layer_types=types,
                  layer_rope_theta=[0 if t == "full_attention" else 500000.0 for t in types],
                  rope_parameters={"rope_theta": 500000.0, "rope_type": "default"})
    reference, ours = LanguageModel(TextConfig(**common)), Model(ModelArgs(**common))
    mx.random.seed(3)
    params = tree_map(lambda p: (mx.random.normal(p.shape) * 0.08).astype(dtype),
                      reference.parameters())
    reference.update(params)
    ours.load_weights(list(dict(tree_flatten(params)).items()), strict=True)
    return reference, ours


def test_muse_bf16_centered_norm_follows_the_fp32_scale_reference():
    """HF and mlx-vlm apply 1 + w in fp32; bf16 rounding of the scale gave
    max |dlogit| 0.0082 and 0.89 argmax agreement on this tiny model (G5-04)."""
    import mlx.core as mx
    import numpy as np

    reference, ours = _muse_pair(mx.bfloat16)
    tokens = mx.array([np.random.default_rng(0).integers(0, 256, size=64).tolist()])
    a = np.array(reference(tokens).logits[0].astype(mx.float32))
    b = np.array(ours(tokens)[0].astype(mx.float32))
    assert np.abs(a - b).max() < 2e-3
    assert (a.argmax(-1) == b.argmax(-1)).mean() >= 0.95


def test_off_pin_mlx_vlm_candidate_names_both_revisions(tmp_path, monkeypatch):
    """G2-04/G5-05: candidates pinned to 8a5e704e can never load under the
    project's mlx-vlm pin; the refusal must say so instead of a bare SHA."""
    import subprocess

    from mlx2.adapters import _direct_mlx_vlm
    from mlx2.adapters.lfm25_vl import SOURCE_REVISION
    from mlx2.adapters.mlx_vlm_pin import MLX_VLM_REVISION

    monkeypatch.setattr(
        _direct_mlx_vlm.subprocess, "check_output",
        lambda *a, **k: MLX_VLM_REVISION + "\n",
    )
    with pytest.raises(RuntimeError) as raised:
        _direct_mlx_vlm.load_backend(tmp_path, SOURCE_REVISION, ())
    message = str(raised.value)
    assert SOURCE_REVISION in message and MLX_VLM_REVISION[:8] in message
    assert "PYTHONPATH" in message
