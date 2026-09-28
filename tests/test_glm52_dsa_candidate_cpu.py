"""CPU-only GLM-5.2 DSA artifact and source-compatibility gates."""

import importlib.abc
import json
import sys
from pathlib import Path

import pytest

from mlx2.adapters.glm52_dsa_candidate import (
    DESCRIPTOR, _required_tensors, inspect_artifact, validate_topology,
)
from mlx2.contracts import Capability

FRAGMENT = Path("~/mlx-models/glm52-mtp-src")
SOURCE = Path("~/Desktop/mlx-uag/mlx-lm-unified/mlx_lm/models")


class _BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError("real MLX import in CPU-only GLM test")
        return None


@pytest.fixture(autouse=True)
def block_mlx(monkeypatch):
    assert "mlx" not in sys.modules and "mlx.core" not in sys.modules
    monkeypatch.setattr(sys, "meta_path", [_BlockMLX(), *sys.meta_path])


def test_local_config_matches_glm52_topology_but_fragment_fails_closed():
    if not FRAGMENT.is_dir():
        pytest.skip("local GLM fragment absent")
    config = json.loads((FRAGMENT / "config.json").read_text())
    validate_topology(config)
    with pytest.raises(ValueError, match="complete indexed checkpoint"):
        inspect_artifact(FRAGMENT)
    assert len(list(FRAGMENT.glob("*.safetensors"))) == 5
    assert Capability.APC_V2 not in DESCRIPTOR.capabilities
    assert Capability.MTP not in DESCRIPTOR.capabilities
    assert "mlx.core" not in sys.modules


def test_shared_indexer_and_moe_schedule_fail_before_weights(tmp_path):
    if not (FRAGMENT / "config.json").is_file():
        pytest.skip("optional local GLM fragment is absent")
    config = json.loads((FRAGMENT / "config.json").read_text())
    config["indexer_types"][6] = "shared"
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="shared-indexer schedule"):
        inspect_artifact(tmp_path)
    config["indexer_types"][6] = "full"
    config["mlp_layer_types"][3] = "dense"
    with pytest.raises(ValueError, match="dense/MoE schedule"):
        validate_topology(config)


def test_full_trunk_keyset_requires_every_expert_and_excludes_mtp():
    required = _required_tensors()
    assert "model.layers.77.mlp.experts.255.down_proj.weight" in required
    assert "model.layers.74.self_attn.indexer.wq_b.weight" in required
    assert "model.layers.78.eh_proj.weight" not in required
    assert len(required) > 58_000


def test_pinned_unified_source_cannot_model_shared_indexer():
    if not SOURCE.is_dir():
        pytest.skip("pinned unified source absent")
    glm = (SOURCE / "glm_moe_dsa.py").read_text()
    base = (SOURCE / "deepseek_v32.py").read_text()
    assert "class Model(DSV32Model):" in glm
    assert "self.indexer = Indexer(config)" in base
    assert "indexer_types" not in glm + base
    assert "mlx.core" not in sys.modules
