"""Static local Granite SWA and HiLS checks; real MLX must never load."""

import importlib.abc
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlx2.adapters.granite_swa import DESCRIPTOR as GRANITE, GraniteSWAAdapter
from mlx2.adapters.granite_swa import inspect_artifact as inspect_granite
from mlx2.adapters.olmo_hils import DESCRIPTOR as HILS, OlmoHiLSAdapter
from mlx2.adapters.olmo_hils import inspect_artifact as inspect_hils
from mlx2.contracts import Capability

GRANITE_ROOT = Path("~/mlx-models")
HILS_ROOT = Path("~/Desktop/mlx-uag/models")


def _require_artifact(path):
    if not all(((path / "config.json").is_file(), (path / "model.safetensors.index.json").is_file())):
        pytest.skip(f"optional local model fixture is absent: {path}")


@pytest.mark.parametrize("name,quantized", [
    ("granite-swash-3b-a600m", False),
    ("granite-swash-3b-a600m-mlx-4bit", True),
])
def test_granite_local_artifacts(name, quantized):
    path = GRANITE_ROOT / name
    _require_artifact(path)
    record = inspect_granite(path)
    assert record["quantized"] is quantized
    assert (record["full_layers"], record["sliding_layers"]) == (8, 20)
    assert record["identity"]["fingerprint"]
    assert Capability.APC_V2 in GRANITE.capabilities
    assert Capability.MTP not in GRANITE.capabilities
    assert GraniteSWAAdapter.default_route == "ordinary"


@pytest.mark.parametrize("name,quantized", [
    ("HiLS-Attention-7B", False),
    ("HiLS-Attention-7B-q6", True),
])
def test_hils_local_artifacts(name, quantized):
    path = HILS_ROOT / name
    _require_artifact(path)
    record = inspect_hils(path)
    assert record["quantized"] is quantized
    assert (record["hils_layers"], record["swa_layers"]) == (8, 24)
    assert Capability.APC_V2 not in HILS.capabilities
    assert Capability.CONTINUOUS_BATCH not in HILS.capabilities
    with pytest.raises(ValueError, match="one lane"):
        OlmoHiLSAdapter.execution_config(OlmoHiLSAdapter, max_lanes=2, prefill_step=128)


def test_static_inspection_blocks_real_mlx():
    _require_artifact(GRANITE_ROOT / "granite-swash-3b-a600m")
    _require_artifact(HILS_ROOT / "HiLS-Attention-7B-q6")
    script = r'''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError("real MLX import attempted")
sys.meta_path.insert(0, Block())
from mlx2.adapters.granite_swa import inspect_artifact as g
from mlx2.adapters.olmo_hils import inspect_artifact as h
g("~/mlx-models/granite-swash-3b-a600m")
h("~/Desktop/mlx-uag/models/HiLS-Attention-7B-q6")
assert "mlx.core" not in sys.modules
'''
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_hils_flag_mismatch_rejected(tmp_path):
    source = HILS_ROOT / "HiLS-Attention-7B/config.json"
    if not source.is_file():
        pytest.skip("optional local HiLS fixture is absent")
    config = json.loads(open(source).read())
    config["chunk_size"] = 32
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="landmark topology"):
        inspect_hils(tmp_path)


def test_granite_layer_order_rejected(tmp_path):
    source = GRANITE_ROOT / "granite-swash-3b-a600m-mlx-4bit/config.json"
    if not source.is_file():
        pytest.skip("optional local Granite fixture is absent")
    config = json.loads(open(source).read())
    config["layer_types"][3] = "sliding_attention"
    (tmp_path / "config.json").write_text(json.dumps(config))
    # Metadata reading needs an index before topology checks, so copy only
    # a minimal index and one inert shard, never the model payload.
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model.safetensors"}}))
    (tmp_path / "model.safetensors").write_bytes(b"metadata only")
    with pytest.raises(ValueError, match="layer order"):
        inspect_granite(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("layer_rope_theta", [10000.0] * 28, "layer_rope_theta"),
        ("head_dim", 128, "head_dim"),
    ],
)
def test_granite_unimplemented_config_fails_closed(tmp_path, field, value, message):
    """Per-layer rope and an explicit head_dim were ignored: a same-shaped
    checkpoint carrying them loaded and served wrong logits."""
    source = GRANITE_ROOT / "granite-swash-3b-a600m-mlx-4bit/config.json"
    if not source.is_file():
        pytest.skip("optional local Granite fixture is absent")
    config = json.loads(open(source).read())
    config[field] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model.safetensors"}}))
    (tmp_path / "model.safetensors").write_bytes(b"metadata only")
    with pytest.raises(ValueError, match=message):
        inspect_granite(tmp_path)
