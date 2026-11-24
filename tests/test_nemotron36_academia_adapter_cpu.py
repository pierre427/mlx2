"""Nemotron 3.6 Academia adapter: resolution, fail-closed gates, CPU load.

Synthetic artifact only (no checkpoint has been released); CPU only.
"""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import nemotron36_academia as adapter_module
from mlx2.adapters.registry import inspect_model
from mlx2.contracts import Capability

from test_nemotron36_academia_cpu import config, hf_weights, run_mlx, tokens


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def write_artifact(root, cfg, *, drop=(), rename=None):
    weights = hf_weights(cfg, seed=9)
    for name in drop:
        weights.pop(name)
    if rename:
        weights = {rename(k): v for k, v in weights.items()}
    root.mkdir(parents=True, exist_ok=True)
    cfg = {**cfg, "architectures": ["Nemotron3_6AcademiaForCausalLM"], "eos_token_id": 2}
    (root / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(str(root / "model.safetensors"), {k: mx.array(v) for k, v in weights.items()})
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {f"w{i}": i for i in range(cfg["vocab_size"])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="w0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "w2", "unk_token": "w0"}))
    return weights


@pytest.mark.parametrize("mode", ["per_loop", "last_loop"])
def test_registry_resolves_and_declares_by_cache_mode(tmp_path, mode):
    cfg = config(recurrent_kv_cache_mode=mode)
    write_artifact(tmp_path / "a", cfg)
    resolution = inspect_model(tmp_path / "a")
    assert resolution.adapter_type is adapter_module.Nemotron36AcademiaAdapter
    descriptor = resolution.descriptor
    assert descriptor.metadata["qualification"] == "unqualified"
    assert descriptor.metadata["qualified"] is False
    assert (Capability.APC_V2 in descriptor.capabilities) is (mode == "per_loop")
    assert mode in descriptor.cache_layout and "loops2" in descriptor.cache_layout
    assert resolution.default_route == "ordinary"


def test_missing_tensor_and_bad_config_fail_closed(tmp_path):
    cfg = config()
    write_artifact(tmp_path / "a", cfg, drop=("model.layers.1.mixer.conv1d.weight",))
    with pytest.raises(ValueError, match="tensors missing"):
        adapter_module.inspect_artifact(tmp_path / "a")
    write_artifact(tmp_path / "b", cfg)
    bad = json.loads((tmp_path / "b" / "config.json").read_text())
    (tmp_path / "b" / "config.json").write_text(json.dumps({**bad, "hidden_act": "silu"}))
    with pytest.raises(ValueError, match="relu2"):
        adapter_module.inspect_artifact(tmp_path / "b")


def test_adapter_loads_and_matches_runtime(tmp_path):
    cfg = config(n_recurrent_loops=3)
    write_artifact(tmp_path / "a", cfg, rename=lambda k: k.replace(".norm.weight", ".input_layernorm.weight")
                   if k.startswith("model.layers.") and ".mixer." not in k else k)
    adapter = adapter_module.Nemotron36AcademiaAdapter(str(tmp_path / "a"))
    try:
        assert adapter.diagnostics()["topology"]["virtual_layers"] == 1 + 3 * 3 + 1
        prompt = tokens(7)
        cache = adapter.model.make_cache()
        logits = adapter.model(mx.array([prompt]), cache=cache)
        reference = run_mlx(adapter.model, [prompt])
        np.testing.assert_allclose(np.array(logits[0]), reference, rtol=0, atol=0)
        assert adapter.prompt_tokens({"prompt": "w5 w6"}) == [5, 6]
        with pytest.raises(ValueError):
            adapter_module.Nemotron36AcademiaAdapter(str(tmp_path / "a"), execution_policy={"x": 1})
    finally:
        adapter.close()
