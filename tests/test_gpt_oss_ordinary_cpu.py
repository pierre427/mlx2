"""GPT-OSS artifact and route checks that cannot import or execute MLX."""

import json
import subprocess
import sys

import pytest

from mlx2.adapters.gpt_oss import (
    GPT_OSS,
    GPT_OSS_PUZZLE,
    GptOssAdapter,
    GptOssPuzzleAdapter,
    inspect_artifact,
)
from mlx2.contracts import Capability


def _artifact(path, *, puzzle=False):
    n = 36 if puzzle else 24
    windows = [128 if i % 2 == 0 else None for i in range(n)]
    if puzzle:
        windows[3] = 8192
    config = {
        "model_type": "gpt_oss_puzzle" if puzzle else "gpt_oss",
        "architectures": ["GptOssPuzzleForCausalLM"] if puzzle else ["GptOssForCausalLM"],
        "hidden_size": 2880, "intermediate_size": 2880, "vocab_size": 201088,
        "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64,
        "num_experts_per_tok": 4, "num_hidden_layers": n,
        "tie_word_embeddings": False, "max_position_embeddings": 131072,
        "eos_token_id": 200002,
        "layer_types": ["full_attention" if w is None else "sliding_attention" for w in windows],
    }
    if puzzle:
        config["block_configs"] = [
            {"num_local_experts": 64 if i >= 15 else 128, "sliding_window": w}
            for i, w in enumerate(windows)
        ]
    else:
        config.update(num_local_experts=32, sliding_window=128)
    keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer in range(n):
        prefix = f"model.layers.{layer}."
        keys.update({prefix + "self_attn.q_proj.weight", prefix + "self_attn.sinks", prefix + "mlp.router.weight"})
        keys.add(prefix + ("mlp.experts.gate_proj.weight" if puzzle else "mlp.experts.gate_up_proj_blocks"))
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in keys}}))
    (path / "model.safetensors").write_bytes(b"metadata test only")
    return config


def test_import_and_inspection_cannot_load_mlx(tmp_path):
    _artifact(tmp_path)
    script = r'''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError("real MLX import attempted")
sys.meta_path.insert(0, Block())
from mlx2.adapters.gpt_oss import inspect_artifact
inspect_artifact(sys.argv[1])
assert "mlx.core" not in sys.modules
'''
    proc = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("puzzle", [False, True])
def test_ordinary_descriptors_and_cache_geometry(tmp_path, puzzle):
    _artifact(tmp_path, puzzle=puzzle)
    record = inspect_artifact(tmp_path)
    descriptor = GPT_OSS_PUZZLE if puzzle else GPT_OSS
    adapter = GptOssPuzzleAdapter if puzzle else GptOssAdapter
    assert adapter.default_route == "ordinary"
    assert Capability.APC_V2 in descriptor.capabilities
    assert Capability.MTP not in descriptor.capabilities
    assert Capability.TOOLS not in descriptor.capabilities
    assert descriptor.metadata["qualification"] == "pending"
    assert len(record["windows"]) == (36 if puzzle else 24)
    if puzzle:
        assert record["windows"][3] == 8192
        assert record["windows"][1] is None
    with pytest.raises(ValueError, match="MTP"):
        adapter.profile_name(adapter, True)


def test_puzzle_rejects_block_and_layer_disagreement(tmp_path):
    config = _artifact(tmp_path, puzzle=True)
    config["block_configs"][3]["sliding_window"] = None
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="layer order"):
        inspect_artifact(tmp_path)


def test_missing_and_unsafe_shards_fail_closed(tmp_path):
    _artifact(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    value = json.loads(index_path.read_text())
    value["weight_map"]["lm_head.weight"] = "../elsewhere.safetensors"
    index_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="unsafe shard"):
        inspect_artifact(tmp_path)
