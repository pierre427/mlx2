"""LLaDA metadata and direct-route checks without loading real MLX."""

import importlib.abc
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from mlx2.adapters.llada import LLADA, LLaDADenoisingAdapter, inspect_artifact, validate_generation
from mlx2.contracts import Capability, StatePlane


_ROOT = Path("~/Desktop/mlx-uag/models")


def _require_artifact(path):
    if not all(((path / "config.json").is_file(), (path / "model.safetensors.index.json").is_file())):
        pytest.skip(f"optional local LLaDA fixture is absent: {path}")


@pytest.mark.parametrize("name,quantized", [
    ("LLaDA-8B-Instruct", False),
    ("LLaDA-8B-Instruct-q4", True),
    ("LLaDA-8B-Instruct-q6", True),
])
def test_local_artifacts_are_denoising_only(name, quantized):
    path = _ROOT / name
    _require_artifact(path)
    record = inspect_artifact(path)
    assert record["quantized"] is quantized
    assert record["execution_kind"] == "bidirectional-denoising"
    assert record["has_autoregressive_cache"] is False
    assert record["identity"]["fingerprint"]
    assert Capability.APC_V2 not in LLADA.capabilities
    assert Capability.STREAMING not in LLADA.capabilities
    assert StatePlane.ATTENTION_KV not in LLADA.state_planes
    assert not hasattr(LLaDADenoisingAdapter, "default_route")


def test_inspection_does_not_import_mlx():
    path = _ROOT / "LLaDA-8B-Instruct-q4"
    _require_artifact(path)
    script = r'''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError("real MLX import attempted")
sys.meta_path.insert(0, Block())
from mlx2.adapters.llada import inspect_artifact
inspect_artifact(sys.argv[1])
assert "mlx.core" not in sys.modules
'''
    proc = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("fields", [
    {"prompt_length": 0}, {"gen_length": 0}, {"block_length": 0},
    {"steps": 0}, {"gen_length": 129}, {"steps": 3},
    {"prompt_length": 4000}, {"temperature": -1},
])
def test_generation_geometry_fails_closed(fields):
    args = dict(prompt_length=4, gen_length=128, block_length=64, steps=128, temperature=0.0)
    args.update(fields)
    with pytest.raises(ValueError):
        validate_generation(**args)


def test_direct_route_wires_exact_sampler(monkeypatch):
    class FakeArray(list):
        def tolist(self):
            return list(self)

    calls = []
    def fake_generate(model, prompt, **kwargs):
        calls.append((model, prompt, kwargs))
        return [FakeArray([11, 12])], "done", {"forwards": 2}

    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    fake_core.array = lambda value: value
    fake_model = types.ModuleType("mlx2.runtime.models.llada")
    fake_model.generate = fake_generate
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setitem(sys.modules, "mlx2.runtime.models.llada", fake_model)
    adapter = object.__new__(LLaDADenoisingAdapter)
    adapter.model = object()
    adapter.tokenizer = types.SimpleNamespace(
        encode=lambda text, **kwargs: [1, 2],
        convert_tokens_to_ids=lambda value: 126348,
        decode=lambda values, **kwargs: "done",
    )
    adapter.config = {"mask_token_id": 126336, "eos_token_id": 126081}
    adapter.identity = {"fingerprint": "test-fingerprint"}
    result = adapter.generate(prompt="hi", gen_length=8, block_length=8, steps=8)
    assert result["text"] == "done"
    assert result["token_ids"] == [11, 12]
    kwargs = calls[0][2]
    assert kwargs["kv_cache"] is False
    assert kwargs["dual_cache"] is False
    assert kwargs["incremental_cache"] is False
    assert kwargs["parallel_threshold"] is None
    assert kwargs["cfg_scale"] == 0.0


def test_chat_template_batch_encoding_is_unwrapped(monkeypatch):
    class FakeArray(list):
        def tolist(self):
            return list(self)

    seen = []
    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    fake_core.array = lambda value: seen.append(value) or value
    fake_model = types.ModuleType("mlx2.runtime.models.llada")
    fake_model.generate = lambda model, prompt, **kwargs: (
        [FakeArray([11])], "Hello!", {"forwards": 1}
    )
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setitem(sys.modules, "mlx2.runtime.models.llada", fake_model)
    adapter = object.__new__(LLaDADenoisingAdapter)
    adapter.model = object()
    adapter.tokenizer = types.SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: {
            "input_ids": [1, 2, 3], "attention_mask": [1, 1, 1],
        },
        convert_tokens_to_ids=lambda value: 126348,
        decode=lambda values, **kwargs: "Hello!",
    )
    adapter.config = {"mask_token_id": 126336, "eos_token_id": 126081}
    adapter.identity = {"fingerprint": "test-fingerprint"}
    result = adapter.generate(
        messages=[{"role": "user", "content": "hi"}],
        gen_length=8, block_length=8, steps=8,
    )
    assert seen == [[[1, 2, 3]]]
    assert result["text"] == "Hello!"


def test_direct_route_stops_client_output_before_canvas_fill(monkeypatch):
    class FakeArray(list):
        def tolist(self):
            return list(self)

    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    fake_core.array = lambda value: value
    fake_model = types.ModuleType("mlx2.runtime.models.llada")
    fake_model.generate = lambda model, prompt, **kwargs: (
        [FakeArray([14455, 126348, 126081, 126081])], "raw special tokens", {"forwards": 4}
    )
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setitem(sys.modules, "mlx2.runtime.models.llada", fake_model)
    adapter = object.__new__(LLaDADenoisingAdapter)
    adapter.model = object()
    adapter.tokenizer = types.SimpleNamespace(
        encode=lambda text, **kwargs: [1, 2],
        convert_tokens_to_ids=lambda value: 126348,
        decode=lambda values, **kwargs: "Hello!" if values == [14455] else "bad",
    )
    adapter.config = {"mask_token_id": 126336, "eos_token_id": 126081}
    adapter.identity = {"fingerprint": "test-fingerprint"}
    result = adapter.generate(prompt="hi", gen_length=8, block_length=8, steps=8)
    assert result["text"] == "Hello!"
    assert result["token_ids"] == [14455]
    assert result["canvas_token_ids"] == [14455, 126348, 126081, 126081]
    assert result["stop_index"] == 1


def test_bad_topology_rejected_before_tensor_load(tmp_path):
    source = _ROOT / "LLaDA-8B-Instruct-q4"
    _require_artifact(source)
    config = json.loads((source / "config.json").read_text())
    config["use_cache"] = True
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="bidirectional topology"):
        inspect_artifact(tmp_path)
