"""scripts/bake_hy4_q2_layers.py: a baked q2 layer equals the scan's in-memory requant (CPU)."""

import importlib.util
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_cpu import TOKENS, checkpoint  # noqa: E402
from test_hy_v4_flash_mlx_pack_cpu import pack_config, write_pack  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAYER = 4  # an affine-4 layer of the tiny pack (layers 1-2 are q2_0)


def _baker():
    spec = importlib.util.spec_from_file_location("bake_hy4_q2_layers", ROOT / "scripts" / "bake_hy4_q2_layers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_experts_per_layer(root: Path):
    """The real pack keeps each layer's experts in its own shard; do the same."""
    index = json.loads((root / "model.safetensors.index.json").read_text())
    tensors = load_file(str(root / "model-experts.safetensors"))
    by_layer = {}
    for name, value in tensors.items():
        by_layer.setdefault(int(name.split(".")[2]), {})[name] = value
    for layer, part in by_layer.items():
        shard = f"model-experts-L{layer}.safetensors"
        save_file(part, str(root / shard))
        for name in part:
            index["weight_map"][name] = shard
    (root / "model-experts.safetensors").unlink()
    (root / "model.safetensors.index.json").write_text(json.dumps(index))


@pytest.fixture(scope="module")
def baked(tmp_path_factory):
    config = pack_config()
    W = checkpoint(config, seed=7, scale=0.2)
    root, _, _ = write_pack(tmp_path_factory.mktemp("pack"), config, W)
    _split_experts_per_layer(root)
    out = tmp_path_factory.mktemp("baked") / "q2"
    assert _baker().main([str(root), str(out), "--layers", str(LAYER), "--bits", "2",
                          "--group-size", "32", "--cpu"]) == 0
    return root, out


def _load(path):
    from mlx2.adapters import hy_v4_flash as A

    model, _, readers = A.load_model(path, A.inspect_artifact(path), keep_dfly=False,
                                     engram_table_format="iq4_nl", vision=False)
    return model, readers


def test_baked_layer_equals_the_scans_in_memory_requant(baked):
    from mlx2 import layer_kl as K

    root, out = baked
    source, r1 = _load(root)
    derived, r2 = _load(out)
    try:
        K.apply_substitution(source, LAYER, ["mlp.experts"], bits=2, group_size=32)
        ids = mx.array([TOKENS[:16]])
        np.testing.assert_array_equal(np.array(derived(ids)), np.array(source(ids)))
        bank = derived.model.layers[LAYER].mlp.experts.gate_proj
        assert bank.bits == 2 and bank.group_size == 32
    finally:
        for r in (*r1.values(), *r2.values()):
            r.close()


def test_derivation_record_links_and_refusal(baked):
    root, out = baked
    config = json.loads((out / "config.json").read_text())
    record = config["mlx2_derivation"]
    assert record["layers"] == [LAYER] and record["bits"] == 2 and record["qualified"] is False
    assert record["source_checkpoint"].startswith(root.name + "@sha256:")
    assert config["quantization"][f"model.layers.{LAYER}.mlp.switch_mlp.down_proj"]["bits"] == 2
    assert (out / "model-00001.safetensors").stat().st_ino == (root / "model-00001.safetensors").stat().st_ino
    assert (out / f"model-experts-L{LAYER}.safetensors").stat().st_ino != \
        (root / f"model-experts-L{LAYER}.safetensors").stat().st_ino
    with pytest.raises(SystemExit, match="not empty"):
        _baker().main([str(root), str(out), "--layers", str(LAYER), "--cpu"])
